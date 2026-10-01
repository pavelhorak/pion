"""PionVectorStore — LlamaIndex BasePydanticVectorStore backed by Pion.

Uses FT.CREATE/FT.SEARCH for HNSW vector retrieval, HSET for document storage.
No RedisJSON module required — documents stored as HASH fields.
"""
from __future__ import annotations

import struct
from typing import Any, List, Optional

import redis as redis_lib

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


_OPTIMIZE_EVERY = 100
_MIN_HNSW_NODES = 8


class PionVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore using Pion's HNSW index.

    Args:
        host: Pion server host (default: 127.0.0.1)
        port: Pion server port (default: 1974)
        index_name: HNSW index name (default: llamaindex)
        dimensions: Embedding dimensions (default: 1536 for OpenAI)
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    host: str = "127.0.0.1"
    port: int = 1974
    index_name: str = "llamaindex"
    dimensions: int = 1536

    _client: Any = None
    _prefix: str = ""
    _seq_key: str = ""
    _cnt_key: str = ""
    _id_map_prefix: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = redis_lib.Redis(
            host=self.host, port=self.port,
            socket_timeout=30,
            decode_responses=False,
        )
        self._prefix = f"{self.index_name}:"
        self._seq_key = f"__{self.index_name}_seq__"
        self._cnt_key = f"__{self.index_name}_count__"
        self._id_map_prefix = f"__{self.index_name}_idmap__:"
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._client.execute_command("FT.INFO", self.index_name)
        except Exception:
            self._client.execute_command(
                "FT.CREATE", self.index_name,
                "SCHEMA", "embedding", "VECTOR", "HNSW",
                "10", "TYPE", "FLOAT32",
                "DIM", str(self.dimensions),
                "DISTANCE_METRIC", "COSINE",
                "M", "16", "EF_CONSTRUCTION", "128",
            )

    @classmethod
    def class_name(cls) -> str:
        return "PionVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    # ── Add ──────────────────────────────────────────────────────────────

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        ids: List[str] = []
        for node in nodes:
            embedding = node.get_embedding()
            if not embedding:
                continue

            vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            seq_id = int(self._client.execute_command("INCR", self._seq_key))
            key = self._prefix + str(seq_id)
            node_id = node.node_id

            text = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata_str = node.get_content(metadata_mode=MetadataMode.ALL)

            fields: dict[bytes, Any] = {
                b"embedding": vec_bytes,
                b"text": text.encode(),
                b"node_id": node_id.encode(),
                b"metadata": metadata_str.encode(),
            }

            ref_doc_id = node.ref_doc_id
            if ref_doc_id:
                fields[b"ref_doc_id"] = ref_doc_id.encode()

            self._client.hset(key, mapping=fields)

            # node_id → seq_id mapping for delete
            self._client.set(self._id_map_prefix + node_id, str(seq_id))

            count = int(self._client.execute_command("INCR", self._cnt_key))
            if count % _OPTIMIZE_EVERY == 0:
                self._client.execute_command("FT.OPTIMIZE", self.index_name)

            ids.append(node_id)

        return ids

    # ── Delete ───────────────────────────────────────────────────────────

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete all nodes with matching ref_doc_id."""
        seq_raw = self._client.get(self._seq_key)
        max_seq = int(seq_raw) if seq_raw else 0
        for i in range(1, max_seq + 1):
            key = self._prefix + str(i)
            doc_id_raw = self._client.hget(key, "ref_doc_id")
            if doc_id_raw is None:
                continue
            doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else doc_id_raw
            if doc_id == ref_doc_id:
                # clean up id map
                node_id_raw = self._client.hget(key, "node_id")
                if node_id_raw:
                    nid = node_id_raw.decode() if isinstance(node_id_raw, bytes) else node_id_raw
                    self._client.delete(self._id_map_prefix + nid)
                self._client.delete(key)

    # ── Query ────────────────────────────────────────────────────────────

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        embedding = query.query_embedding
        if not embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        k = query.similarity_top_k or 10
        ef = kwargs.get("ef", max(64, k * 2))
        vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        count_raw = self._client.get(self._cnt_key)
        total = int(count_raw) if count_raw else 0

        if total < _MIN_HNSW_NODES:
            return self._linear_query(vec_bytes, k)

        raw = self._client.execute_command(
            "FT.SEARCH", self.index_name,
            f"*=>[KNN {k} @embedding $vec EF_RUNTIME {ef}]",
            "PARAMS", "2", "vec", vec_bytes,
        )
        return self._parse_results(raw, k)

    def _parse_results(self, raw: Any, k: int) -> VectorStoreQueryResult:
        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        if not raw or not isinstance(raw, list):
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        items = list(raw)
        i = 1 if (items and _is_int(items[0])) else 0

        while i + 1 < len(items) and len(nodes) < k:
            doc_id_raw = items[i]
            fields_raw = items[i + 1]
            i += 2

            doc_id_str = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
            # Pion returns the original key; prefixing it again made
            # "idx:" + "idx:1" and every node came back with empty text.
            key = doc_id_str if doc_id_str.startswith(self._prefix) else self._prefix + doc_id_str

            text = ""
            node_id = doc_id_str
            score = 0.0
            metadata_str = ""

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
                    if fname == "text":
                        text = fval_str
                    elif fname == "node_id":
                        node_id = fval_str
                    elif fname == "score":
                        # Server score = COSINE distance, 1 - cos (gh #365);
                        # LlamaIndex `similarities` are similarities.
                        try:
                            score = 1.0 - float(fval_str)
                        except ValueError:
                            pass
                    elif fname == "metadata":
                        metadata_str = fval_str
                    fi += 2

            # Fetch missing text if not in FT.SEARCH response
            if not text:
                text_raw = self._client.hget(key, "text")
                if text_raw:
                    text = text_raw.decode(errors="replace")
            if node_id == doc_id_str:
                nid_raw = self._client.hget(key, "node_id")
                if nid_raw:
                    node_id = nid_raw.decode()

            node = TextNode(id_=node_id, text=text)
            nodes.append(node)
            similarities.append(score)
            ids.append(node_id)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _linear_query(self, query_vec: bytes, k: int) -> VectorStoreQueryResult:
        seq_raw = self._client.get(self._seq_key)
        max_seq = int(seq_raw) if seq_raw else 0
        q = struct.unpack_from(f"{len(query_vec)//4}f", query_vec)
        qn = sum(v * v for v in q) ** 0.5 or 1.0

        candidates: list[tuple[float, str, str]] = []
        for i in range(1, max_seq + 1):
            key = self._prefix + str(i)
            emb_raw = self._client.hget(key, "embedding")
            if emb_raw is None:
                continue
            d = struct.unpack_from(f"{len(emb_raw)//4}f", emb_raw)
            dn = sum(v * v for v in d) ** 0.5 or 1.0
            dot = sum(qi * di for qi, di in zip(q, d))
            sim = dot / (qn * dn)

            text_raw = self._client.hget(key, "text")
            nid_raw = self._client.hget(key, "node_id")
            text = text_raw.decode(errors="replace") if text_raw else ""
            nid = nid_raw.decode() if nid_raw else str(i)
            candidates.append((sim, nid, text))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:k]

        nodes = [TextNode(id_=nid, text=text) for _, nid, text in top]
        similarities = [sim for sim, _, _ in top]
        ids = [nid for _, nid, _ in top]
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


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
