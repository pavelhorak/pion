"""PionMemoryStore — AutoGen Memory protocol backed by Pion's HNSW vector index.

Implements autogen_core.memory.Memory protocol:
  - add(content) → None
  - query(text, k, score_threshold) → MemoryQueryResult
  - update_context(model_context) → UpdateContextResult
  - clear() → None
  - close() → None

Uses FT.CREATE/FT.SEARCH for semantic retrieval, HSET for storage.
Embedding via mock (deterministic, no deps), OpenAI, or Ollama.
"""
from __future__ import annotations

import struct
import time
from typing import Any, Optional

import redis as redis_lib

from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage


# ── Embedding helpers ────────────────────────────────────────────────────────

def _embed_mock(text: str, dim: int = 384) -> bytes:
    import hashlib
    import math
    # Seeded Gaussian per dimension. The old sin(h + i*1.618) added a small float to a
    # 256-bit hash — as a float h + i*1.618 == h — so every text embedded to the same
    # constant vector and no search over mock embeddings could tell results apart.
    import random
    rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
    floats = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    floats = [v / norm for v in floats]
    return struct.pack(f"{dim}f", *floats)


def _embed_openai(text: str, model: str) -> bytes:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=model, input=[text])
    floats = resp.data[0].embedding
    return struct.pack(f"{len(floats)}f", *floats)


def _embed_ollama(text: str, model: str, host: str, port: int) -> bytes:
    import json
    import urllib.request
    url = f"http://{host}:{port}/api/embeddings"
    payload = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    floats = data["embedding"]
    return struct.pack(f"{len(floats)}f", *floats)


def _embed(text: str, provider: str, model: str, dim: int,
           host: str, port: int) -> bytes:
    if provider == "mock":
        return _embed_mock(text, dim)
    elif provider == "openai":
        return _embed_openai(text, model or "text-embedding-3-small")
    elif provider == "ollama":
        return _embed_ollama(text, model or "nomic-embed-text", host, port)
    raise ValueError(f"Unknown embed_provider={provider!r}")


_OPTIMIZE_EVERY = 50
_MIN_HNSW_NODES = 8


class PionMemoryStore(Memory):
    """AutoGen Memory backed by Pion's HNSW vector index.

    Args:
        host:           Pion server host (default: 127.0.0.1)
        port:           Pion server port (default: 1974)
        index_name:     HNSW index name (default: autogen_memory)
        dimensions:     Embedding dimensions (default: 384)
        embed_provider: "mock" | "openai" | "ollama"
        embed_model:    Model name (provider-specific)
        embed_host:     Embedding server host (Ollama default: 127.0.0.1)
        embed_port:     Embedding server port (Ollama default: 11434)
        context_k:      Number of memories to inject via update_context (default: 5)
    """

    component_type = "memory"
    component_provider_override = "pion_autogen.PionMemoryStore"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1974,
        index_name: str = "autogen_memory",
        dimensions: int = 384,
        embed_provider: str = "mock",
        embed_model: str = "",
        embed_host: str = "127.0.0.1",
        embed_port: int = 11434,
        context_k: int = 5,
    ):
        self._r = redis_lib.Redis(
            host=host, port=port,
            socket_timeout=30,
            decode_responses=False,
        )
        self._index = index_name
        self._dim = dimensions
        self._provider = embed_provider
        self._model = embed_model
        self._emb_host = embed_host
        self._emb_port = embed_port
        self._context_k = context_k
        self._seq_key = f"__{index_name}_seq__"
        self._cnt_key = f"__{index_name}_count__"
        self._prefix = f"{index_name}:"
        self._ensure_index()

    @property
    def name(self) -> str:
        return f"PionMemoryStore({self._index})"

    def _ensure_index(self) -> None:
        try:
            self._r.execute_command("FT.INFO", self._index)
        except Exception:
            self._r.execute_command(
                "FT.CREATE", self._index,
                "SCHEMA", "embedding", "VECTOR", "HNSW",
                "10", "TYPE", "FLOAT32",
                "DIM", str(self._dim),
                "DISTANCE_METRIC", "COSINE",
                "M", "16", "EF_CONSTRUCTION", "128",
            )

    def _embed_text(self, text: str) -> bytes:
        return _embed(text, self._provider, self._model, self._dim,
                      self._emb_host, self._emb_port)

    # ── Memory protocol ──────────────────────────────────────────────────

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Optional[Any] = None,
    ) -> None:
        text = str(content.content) if not isinstance(content.content, str) else content.content
        mime_type = str(content.mime_type) if not isinstance(content.mime_type, str) else content.mime_type

        vec = self._embed_text(text)
        seq_id = int(self._r.execute_command("INCR", self._seq_key))
        key = self._prefix + str(seq_id)

        fields: dict[bytes, Any] = {
            b"embedding": vec,
            b"text": text.encode(),
            b"mime_type": mime_type.encode(),
            b"timestamp": str(int(time.time())).encode(),
        }
        if content.metadata:
            for mk, mv in content.metadata.items():
                fields[mk.encode() if isinstance(mk, str) else mk] = (
                    str(mv).encode() if not isinstance(mv, (bytes, bytearray)) else mv
                )

        self._r.hset(key, mapping=fields)

        count = int(self._r.execute_command("INCR", self._cnt_key))
        if count % _OPTIMIZE_EVERY == 0:
            self._r.execute_command("FT.OPTIMIZE", self._index)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        query_text = str(query.content) if isinstance(query, MemoryContent) else str(query)
        k = kwargs.get("k", 5)
        score_threshold = kwargs.get("score_threshold", 0.0)
        ef = kwargs.get("ef", 64)

        vec = self._embed_text(query_text)
        results = self._search(vec, k=k, ef=ef)

        contents: list[MemoryContent] = []
        for r in results:
            score = r.get("score", 0.0)
            if score_threshold > 0 and score < score_threshold:
                continue
            mime = r.get("mime_type", "text/plain")
            text = r.get("text", "")
            contents.append(MemoryContent(
                content=text,
                mime_type=MemoryMimeType(mime) if mime in [e.value for e in MemoryMimeType] else mime,
                metadata={"score": score, "id": r.get("id")},
            ))

        return MemoryQueryResult(results=contents)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Query recent messages and inject relevant memories into context."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        # Use the last user message as query
        query_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                query_text = msg.content
                break

        if not query_text:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        result = await self.query(query_text, k=self._context_k)

        if result.results:
            memory_strings = [
                f"{i}. {str(m.content)}"
                for i, m in enumerate(result.results, 1)
            ]
            memory_context = (
                "\nRelevant memory content:\n"
                + "\n".join(memory_strings)
                + "\n"
            )
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=result)

    async def clear(self) -> None:
        seq_raw = self._r.get(self._seq_key)
        max_seq = int(seq_raw) if seq_raw else 0
        for i in range(1, max_seq + 1):
            self._r.delete(self._prefix + str(i))
        self._r.delete(self._seq_key)
        self._r.delete(self._cnt_key)

    async def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass

    # ── Internal search ──────────────────────────────────────────────────

    def _search(self, vec: bytes, k: int = 5, ef: int = 64) -> list[dict[str, Any]]:
        count_raw = self._r.get(self._cnt_key)
        total = int(count_raw) if count_raw else 0

        if total < _MIN_HNSW_NODES:
            return self._linear_search(vec, k)

        raw = self._r.execute_command(
            "FT.SEARCH", self._index,
            f"*=>[KNN {k} @embedding $vec EF_RUNTIME {ef}]",
            "PARAMS", "2", "vec", vec,
        )
        return self._parse_ft_search(raw, k)

    def _linear_search(self, query_vec: bytes, k: int) -> list[dict[str, Any]]:
        seq_raw = self._r.get(self._seq_key)
        max_seq = int(seq_raw) if seq_raw else 0
        q = struct.unpack_from(f"{len(query_vec)//4}f", query_vec)
        qn = sum(v * v for v in q) ** 0.5 or 1.0

        candidates: list[dict[str, Any]] = []
        for i in range(1, max_seq + 1):
            key = self._prefix + str(i)
            emb_raw = self._r.hget(key, "embedding")
            if emb_raw is None:
                continue
            d = struct.unpack_from(f"{len(emb_raw)//4}f", emb_raw)
            dn = sum(v * v for v in d) ** 0.5 or 1.0
            dot = sum(qi * di for qi, di in zip(q, d))
            sim = dot / (qn * dn)

            text_raw = self._r.hget(key, "text")
            mime_raw = self._r.hget(key, "mime_type")
            candidates.append({
                "id": i,
                "text": text_raw.decode(errors="replace") if text_raw else "",
                "mime_type": mime_raw.decode() if mime_raw else "text/plain",
                "score": round(sim, 4),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def _parse_ft_search(self, raw: Any, k: int) -> list[dict[str, Any]]:
        if not raw or not isinstance(raw, list):
            return []

        items = list(raw)
        i = 1 if (items and isinstance(items[0], (int, bytes)) and _is_int(items[0])) else 0
        results: list[dict[str, Any]] = []

        while i + 1 < len(items) and len(results) < k:
            doc_id_raw = items[i]
            fields_raw = items[i + 1]
            i += 2

            doc_id_str = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
            entry: dict[str, Any] = {"id": int(doc_id_str) if doc_id_str.isdigit() else doc_id_str}

            if isinstance(fields_raw, list):
                fi = 0
                while fi + 1 < len(fields_raw):
                    fname = fields_raw[fi]
                    fval = fields_raw[fi + 1]
                    fname = fname.decode() if isinstance(fname, bytes) else fname
                    if fname == "embedding":
                        fi += 2
                        continue
                    fval_str = fval.decode(errors="replace") if isinstance(fval, bytes) else str(fval)
                    if fname == "score":
                        # The server's score is the COSINE index's distance,
                        # 1 - cos (gh #365); this store reports similarity, the
                        # same scale its linear-search fallback computes.
                        try:
                            entry["score"] = 1.0 - float(fval_str)
                        except ValueError:
                            pass
                    else:
                        entry[fname] = fval_str
                    fi += 2

            if "text" not in entry:
                # Pion returns the original key; prefixing it again missed every HGET.
                key = doc_id_str if doc_id_str.startswith(self._prefix) else self._prefix + doc_id_str
                text_raw = self._r.hget(key, "text")
                if text_raw:
                    entry["text"] = text_raw.decode(errors="replace")

            results.append(entry)

        return results


def _is_int(val: Any) -> bool:
    if isinstance(val, int):
        return True
    if isinstance(val, bytes):
        try:
            int(val)
            return True
        except ValueError:
            return False
    return False
