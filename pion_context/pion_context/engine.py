"""Semantic codebase search — retrieves relevant code and memories from Pion.

Combines codebase search, conversation memory, and semantic cache
into a single unified context retrieval interface.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import redis as redis_lib

from .embeddings import embed_text, embed_dim
from .indexer import INDEX_NAME, HASH_PREFIX


class SearchError(RuntimeError):
    """A codebase search that could not run: the query could not be embedded,
    or the server refused it (no index, or another index has replaced it)."""


@dataclass
class ContextResult:
    """A single context retrieval result."""
    source: str  # "codebase", "memory", "cache"
    content: str
    score: float = 0.0
    file_path: str = ""
    start_line: int = 0
    name: str = ""


class ContextEngine:
    """Unified semantic context retrieval from Pion.

    Searches three sources:
    1. Codebase index (__codebase__) — code chunks
    2. Agent memory (__agent_memory__) — past decisions/facts
    3. Semantic cache (AI.SEMANTIC_CACHE) — cached Q&A
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1974):
        self.conn = redis_lib.Redis(
            host=host, port=port,
            decode_responses=False,
            socket_keepalive=True,
        )

    def search_codebase(self, query: str, k: int = 10, ef_runtime: int = 150) -> list[ContextResult]:
        """Search the codebase index for semantically relevant code.

        Raises SearchError when the search cannot run. It used to return []
        then, so an embedding outage, a missing index, or an index replaced
        by another one (Pion serves one at a time) all read as "nothing
        relevant".
        """
        try:
            query_vec = embed_text(query)
        except Exception as e:
            raise SearchError(f"could not embed the query: {e}") from e

        try:
            raw = self.conn.execute_command(
                "FT.SEARCH", INDEX_NAME,
                f"*=>[KNN {k} @vec $vec EF_RUNTIME {ef_runtime}]",
                "PARAMS", "2", "vec", query_vec,
            )
        except redis_lib.ResponseError as e:
            raise SearchError(f"the server refused the codebase search: {e}") from e

        return self._parse_hnsw_results(raw, "codebase", HASH_PREFIX)

    def search_memory(self, query: str, k: int = 5) -> list[ContextResult]:
        """Search agent memory for relevant past context."""
        try:
            query_vec = embed_text(query)
        except Exception:
            return []

        try:
            raw = self.conn.execute_command(
                "FT.SEARCH", "__agent_memory__",
                f"*=>[KNN {k} @embedding $vec EF_RUNTIME 64]",
                "PARAMS", "2", "vec", query_vec,
                "RETURN", "3", "text", "session_id", "timestamp",
            )
        except Exception:
            return []

        return self._parse_search_results(raw, "memory")

    def search_cache(self, query: str, threshold: float = 0.92) -> Optional[str]:
        """Check semantic cache for a similar past query."""
        try:
            query_vec = embed_text(query)
        except Exception:
            return None

        try:
            # Use AI.SEMANTIC_CACHE GET via raw RESP
            resp = self.conn.execute_command(
                "AI.SEMANTIC_CACHE", "GET", query, "THRESHOLD", f"{threshold:.2f}"
            )
            if resp and resp != b"$-1\r\n":
                if isinstance(resp, bytes):
                    return resp.decode("utf-8", errors="replace")
            return None
        except Exception:
            return None

    def retrieve_context(
        self,
        query: str,
        code_k: int = 8,
        memory_k: int = 3,
        check_cache: bool = True,
    ) -> list[ContextResult]:
        """Unified context retrieval — searches all sources.

        Returns results sorted by relevance (highest score first).
        """
        results: list[ContextResult] = []

        # 1. Codebase search (a failure is reported, not hidden: see search_codebase)
        code_results = self.search_codebase(query, k=code_k)
        results.extend(code_results)

        # 2. Memory search
        memory_results = self.search_memory(query, k=memory_k)
        results.extend(memory_results)

        # 3. Semantic cache (returns a single cached response, not ranked)
        if check_cache:
            cached = self.search_cache(query)
            if cached:
                results.append(ContextResult(
                    source="cache",
                    content=cached,
                    score=1.0,  # cache hit = high relevance
                ))

        return results

    def format_context(self, results: list[ContextResult], max_chars: int = 8000) -> str:
        """Format context results into a string for injection into Claude's context."""
        if not results:
            return ""

        parts = []
        total = 0

        # Group by source
        code_results = [r for r in results if r.source == "codebase"]
        memory_results = [r for r in results if r.source == "memory"]
        cache_results = [r for r in results if r.source == "cache"]

        if code_results:
            parts.append("## Relevant Code (from Pion codebase index)\n")
            for r in code_results:
                loc = f"{r.file_path}:{r.start_line}" if r.file_path else ""
                entry = f"### {loc} ({r.name})\n```\n{r.content}\n```\n"
                if total + len(entry) > max_chars:
                    break
                parts.append(entry)
                total += len(entry)

        if memory_results:
            parts.append("\n## Relevant Memories (from past sessions)\n")
            for r in memory_results:
                entry = f"- {r.content}\n"
                if total + len(entry) > max_chars:
                    break
                parts.append(entry)
                total += len(entry)

        if cache_results:
            parts.append("\n## Cached Answer (semantic match)\n")
            parts.append(cache_results[0].content + "\n")

        return "".join(parts)

    def _parse_hnsw_results(self, raw, source: str, prefix: str) -> list[ContextResult]:
        """Parse FT.SEARCH response and fetch metadata from hash keys.

        Pion returns the original hash key for each result (it returned
        bare HNSW node ids long ago, hence the prefix fallback), then the
        fields are fetched via HGETALL.
        """
        results = []
        if not raw or not isinstance(raw, list) or len(raw) < 2:
            return results

        # Format: [count, node_id_1, [id, val, score, val], node_id_2, ...]
        i = 1
        while i < len(raw):
            node_id = raw[i]
            if isinstance(node_id, bytes):
                node_id = node_id.decode()
            elif isinstance(node_id, int):
                node_id = str(node_id)
            i += 1

            # Skip the [id, val, score, val] array if present
            if i < len(raw) and isinstance(raw[i], list):
                i += 1

            # Fetch metadata from hash key
            # Pion returns the real key; prefixing it again made
            # "ctx:" + "ctx:doc" and every HGETALL came back empty.
            hash_key = node_id if node_id.startswith(prefix) else f"{prefix}{node_id}"
            try:
                fields_raw = self.conn.hgetall(hash_key)
                fields = {
                    k.decode(): v.decode("utf-8", errors="replace")
                    for k, v in fields_raw.items()
                    if k != b"embedding" and k != b"vector"
                }
            except Exception:
                fields = {}

            results.append(ContextResult(
                source=source,
                content=fields.get("text", ""),
                file_path=fields.get("file_path", ""),
                start_line=int(fields.get("start_line", 0)),
                name=fields.get("name", node_id),
            ))

        return results

    def _parse_search_results(self, raw, source: str) -> list[ContextResult]:
        """Parse FT.SEARCH response into ContextResult list."""
        results = []
        if not raw or not isinstance(raw, list):
            return results

        # FT.SEARCH returns: [total_count, doc_id_1, [field, value, ...], doc_id_2, ...]
        total = raw[0] if isinstance(raw[0], int) else int(raw[0])
        i = 1
        while i < len(raw) - 1:
            doc_id = raw[i]
            if isinstance(doc_id, bytes):
                doc_id = doc_id.decode()
            fields_list = raw[i + 1] if i + 1 < len(raw) else []
            i += 2

            # Parse field pairs
            fields = {}
            if isinstance(fields_list, list):
                for j in range(0, len(fields_list) - 1, 2):
                    k = fields_list[j]
                    v = fields_list[j + 1]
                    if isinstance(k, bytes):
                        k = k.decode()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="replace")
                    fields[k] = v

            results.append(ContextResult(
                source=source,
                content=fields.get("text", ""),
                file_path=fields.get("file_path", ""),
                start_line=int(fields.get("start_line", 0)),
                name=fields.get("name", doc_id),
            ))

        return results
