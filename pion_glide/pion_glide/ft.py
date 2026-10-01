"""Typed wrappers for Pion's FT.* vector-search commands.

All methods delegate to ``PionClient.execute()`` which uses GLIDE's
``custom_command()`` under the hood — no monkey-patching or protocol hacks.

Workflow
--------
::

    # 1. Create index schema
    await client.ft.create("products", field="embedding", dim=1536)

    # 2. Ingest vectors (HSET  index  doc_id  field  <float32-blob>)
    await client.ft.add_vector("products", "sku:123", [0.1, 0.2, ...])

    # 3. Build HNSW graph
    await client.ft.optimize("products")

    # 4. Search (k-NN)
    results = await client.ft.search("products", query_vec, k=10)
    for r in results:
        print(r.doc_id, r.score)

    # 5. Text search (server-side embedding via Ollama/MAX — requires --flare)
    await client.ft.add_text("docs", "doc:1", "Pion achieves 10K QPS on Linux")
    results = await client.ft.search_text("docs", "vector database performance", k=5)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from .client import PionClient


@dataclass
class FTSearchResult:
    """A single result returned by :meth:`FTIndex.search` or :meth:`FTIndex.search_text`.

    Attributes
    ----------
    doc_id:   The document identifier (e.g. ``"sku:123"`` or a raw HNSW node id).
    score:    Approximate distance score (lower = closer for L2).
    fields:   Returned hash fields (populated when ``RETURN`` is passed to FT.SEARCH).
    """
    doc_id: str
    score: float = 0.0
    fields: Dict[str, str] = field(default_factory=dict)


class FTIndex:
    """Typed helpers for Pion's FT.* (HNSW vector search) commands.

    Obtain an instance from :class:`~pion_glide.PionClient`::

        client.ft.create(...)
        client.ft.search(...)
    """

    def __init__(self, client: "PionClient") -> None:
        self._client = client

    async def create(
        self,
        index_name: str,
        field: str = "embedding",
        dim: int = 1536,
        metric: str = "L2",
        M: int = 16,
        ef_construction: int = 128,
    ) -> str:
        """Create an HNSW vector index.

        Equivalent to::

            FT.CREATE index_name SCHEMA field VECTOR HNSW 6 TYPE FLOAT32 DIM dim DISTANCE_METRIC metric

        Parameters
        ----------
        index_name:       Name used in subsequent FT.SEARCH calls.
        field:            Hash field that contains the float32 embedding blob.
        dim:              Vector dimension (must match ingested vectors).
        metric:           ``"L2"`` (default) or ``"COSINE"``.
        M:                HNSW M parameter (neighbours per node, default 16).
        ef_construction:  Build-time beam width (default 128).
        """
        return await self._client.execute(
            "FT.CREATE", index_name,
            "SCHEMA", field, "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", metric,
        )

    async def optimize(self, index_name: str) -> str:
        """Build the HNSW graph from all ingested vectors.

        Call once after bulk ingest.  Additional vectors ingested after
        ``FT.OPTIMIZE`` are added incrementally (no rebuild required).
        """
        return await self._client.execute("FT.OPTIMIZE", index_name)

    async def drop(self, index_name: str) -> str:
        """Delete the index and all associated vectors."""
        return await self._client.execute("FT.DROPINDEX", index_name)

    async def info(self, index_name: str) -> Any:
        """Return index metadata (dimensions, node count, build status)."""
        return await self._client.execute("FT.INFO", index_name)

    async def add_vector(
        self,
        index_name: str,
        doc_id: str,
        vector: Sequence[float],
        field: str = "embedding",
    ) -> int:
        """Ingest a single float32 vector.

        Sends ``HSET index_name doc_id field <float32-blob>`` where the blob
        is a packed little-endian float32 array.

        Parameters
        ----------
        index_name:   Target index (must be created first).
        doc_id:       Document identifier (e.g. ``"product:42"``).
        vector:       Python list or array of floats.
        field:        Hash field name (must match the field in FT.CREATE).
        """
        blob = struct.pack(f"{len(vector)}f", *vector)
        return await self._client.execute("HSET", index_name, doc_id, field, blob)

    async def search(
        self,
        index_name: str,
        query_vector: Sequence[float],
        k: int = 10,
        ef_runtime: int = 150,
        field: str = "embedding",
        return_fields: Optional[List[str]] = None,
    ) -> List[FTSearchResult]:
        """k-NN vector search.

        Sends::

            FT.SEARCH index_name "*=>[KNN k @field $vec EF_RUNTIME ef]"
                PARAMS 2 vec <float32-blob>

        Parameters
        ----------
        index_name:     Index to search.
        query_vector:   Query embedding (must match index dimension).
        k:              Number of neighbours to return.
        ef_runtime:     HNSW search beam width (larger = higher recall, slower).
        field:          Vector field name (must match FT.CREATE schema).
        return_fields:  Additional hash fields to include in results.

        Returns
        -------
        List of :class:`FTSearchResult` ordered by ascending distance.
        """
        blob = struct.pack(f"{len(query_vector)}f", *query_vector)
        query = f"*=>[KNN {k} @{field} $vec EF_RUNTIME {ef_runtime}]"
        args: List[Any] = ["FT.SEARCH", index_name, query, "PARAMS", "2", "vec", blob]
        if return_fields:
            args += ["RETURN", str(len(return_fields))] + return_fields
        raw = await self._client.execute(*args)
        return _parse_ft_results(raw)

    async def add_text(
        self,
        index_name: str,
        doc_id: str,
        text: str,
    ) -> str:
        """Add a text document with server-side embedding.

        Requires Pion started with ``--flare`` (auto-detects Ollama).
        The server embeds the text via Ollama/MAX Serve and inserts the vector.

        Equivalent to ``FT.ADDTEXT index_name doc_id text``.
        """
        return await self._client.execute("FT.ADDTEXT", index_name, doc_id, text)

    async def search_text(
        self,
        index_name: str,
        query: str,
        k: int = 10,
    ) -> List[FTSearchResult]:
        """Text-to-vector search with server-side embedding.

        Requires Pion started with ``--flare``.  The server embeds ``query``
        and performs k-NN search — no float vectors needed on the client side.

        Equivalent to ``FT.SEARCHTEXT index_name query K k``.
        """
        raw = await self._client.execute("FT.SEARCHTEXT", index_name, query, "K", str(k))
        return _parse_ft_results(raw)


# ── Response parser ──────────────────────────────────────────────────────────

def _parse_ft_results(raw: Any) -> List[FTSearchResult]:
    """Parse FT.SEARCH / FT.SEARCHTEXT RESP response into typed objects.

    Pion returns one of two layouts depending on whether the index was
    created via FT.CREATE (returns HNSW node ids as integers in a flat
    array) or the full Redis-style ``[count, doc_id, [field, val], …]`` array.
    We handle both.
    """
    if not raw or not isinstance(raw, (list, tuple)):
        return []

    results: List[FTSearchResult] = []
    try:
        idx = 1  # skip leading count element
        while idx < len(raw):
            item = raw[idx]
            if item is None:
                idx += 1
                continue
            doc_id = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            result = FTSearchResult(doc_id=doc_id)
            # Optional field list following the doc_id
            if idx + 1 < len(raw) and isinstance(raw[idx + 1], (list, tuple)):
                fields_raw = raw[idx + 1]
                pairs: Dict[str, str] = {}
                j = 0
                while j + 1 < len(fields_raw):
                    k_name = fields_raw[j]
                    v = fields_raw[j + 1]
                    if isinstance(k_name, bytes):
                        k_name = k_name.decode("utf-8")
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    pairs[str(k_name)] = str(v)
                    j += 2
                result.fields = pairs
                # Score may be in __vector_score field
                if "__vector_score" in pairs:
                    try:
                        result.score = float(pairs["__vector_score"])
                    except ValueError:
                        pass
                idx += 2
            else:
                idx += 1
            results.append(result)
    except Exception:
        pass
    return results
