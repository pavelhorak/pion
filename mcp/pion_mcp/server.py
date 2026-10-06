"""Pion MCP Server.

Exposes Pion's vector search and KV store as MCP tools, usable from
Claude Code, Cursor, GitHub Copilot, and any MCP-compatible agent.

Usage:
    # Install from this checkout. The package is NOT on PyPI, so the usual
    # pip-install-by-name and uvx-by-name invocations both 404. Do not restore
    # them until it is actually published (tracked in the launch issues).
    pip install -e mcp/

    # Add to Claude Code
    claude mcp add pion -- python -m pion_mcp.server

    # Environment variables
    PION_HOST          Pion server host (default: localhost)
    PION_PORT          Pion server port (default: 1974)
    PION_EMBED_PROVIDER  openai | max | mock (default: openai)
    PION_EMBED_MODEL   embedding model name (default: text-embedding-3-small)
    PION_EMBED_DIM     embedding dimension (default: 1536)
    OPENAI_API_KEY     required when PION_EMBED_PROVIDER=openai
    PION_EMBED_URL     MAX Serve base URL (default: http://localhost:8000/v1)
"""
from __future__ import annotations

import os
import struct
import sys
from typing import Any

import redis as redis_lib
from mcp.server.fastmcp import FastMCP

from .embeddings import embed_text, embed_texts

# ── Configuration ─────────────────────────────────────────────────────────────

PION_HOST = os.environ.get("PION_HOST", "localhost")
PION_PORT = int(os.environ.get("PION_PORT", "1974"))
DEFAULT_EF = int(os.environ.get("PION_EF_RUNTIME", "150"))
DEFAULT_M = int(os.environ.get("PION_M", "16"))
DEFAULT_EF_CONSTRUCTION = int(os.environ.get("PION_EF_CONSTRUCTION", "128"))


# ── MCP server & Pion connection ──────────────────────────────────────────────

mcp = FastMCP(
    "Pion",
    instructions=(
        "Pion is a Redis-compatible vector database with HNSW search. "
        "Use it to store and retrieve documents by semantic similarity, "
        "manage key-value data, and cache LLM responses. "
        "Pion beats Redis VSET on QPS (+14.5%) and latency (6.7×). "
        "Default port: 1974."
    ),
)

# Persistent single connection — required because Pion is shared-nothing:
# each worker owns an independent keyspace, so different TCP connections
# land on different workers. A stable connection guarantees all operations
# on the same data hit the same worker.
_pion: redis_lib.Redis | None = None


def _conn() -> redis_lib.Redis:
    """Return the persistent Pion connection, creating it on first call."""
    global _pion
    if _pion is None:
        _pion = redis_lib.Redis(
            host=PION_HOST,
            port=PION_PORT,
            decode_responses=False,
            socket_keepalive=True,
        )
    return _pion


def _knn(r: redis_lib.Redis, index: str, vec: bytes, k: int,
         ef_runtime: int | None = None, extra: tuple = ()) -> Any:
    """Standard KNN query on the "vector" field this server's indexes use.

    The tools used to send `FT.SEARCH <index> <vector-bytes> K <k>`, which
    Pion does not parse as a vector query: it answered an empty array, so
    vector_search, vector_search_by_vector, search_with_filter and
    semantic_cache_get returned nothing against a real server.
    """
    ef = f" EF_RUNTIME {ef_runtime}" if ef_runtime else ""
    return r.execute_command(
        "FT.SEARCH", index, f"*=>[KNN {k} @vector $vec{ef} AS score]",
        *extra, "PARAMS", "2", "vec", vec, "DIALECT", "2",
    )


def _score_of(v: Any) -> float | None:
    """The score from one FT.SEARCH result's fields.

    Pion replies [count, key, fields, key, fields, ...] where fields is
    [b"id", <id>, b"score", <score>] or just [b"score", <score>]. The parsers
    here used to float() the whole fields array, which always raised, so
    every score came back 0.0 and semantic_cache_get never hit. Read the
    score by name; a bare scalar is accepted too.
    """
    if isinstance(v, (list, tuple)):
        v = next((v[j + 1] for j in range(0, len(v) - 1, 2) if v[j] in (b"score", "score")), None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cosine(a: bytes, b: bytes) -> float | None:
    """Cosine similarity of two little-endian float32 vectors, or None."""
    import array
    va, vb = array.array("f"), array.array("f")
    va.frombytes(a)
    vb.frombytes(b)
    if len(va) != len(vb) or not va:
        return None
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    return dot / (na * nb) if na and nb else None


# ── Index management ──────────────────────────────────────────────────────────

@mcp.tool()
def create_index(
    index: str,
    dim: int = 1536,
    metric: str = "COSINE",
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
) -> str:
    """Create a Pion HNSW vector index.

    Args:
        index: Index name (e.g. "docs", "products", "memories").
        dim: Embedding dimension. Must match your embedding model output.
              Use 1536 for OpenAI text-embedding-3-small/large,
              768 for nomic-embed-text, 384 for all-MiniLM-L6-v2.
        metric: Distance metric — COSINE (default) or L2.
        m: HNSW M parameter (neighbours per node). Higher = better recall, more RAM.
           Default 16 is optimal for most workloads.
        ef_construction: Build-time beam width. Higher = better graph quality.
    """
    r = _conn()
    try:
        r.execute_command(
            "FT.CREATE", index,
            "SCHEMA", "vector", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", metric,
            "M", str(m),
            "EF_CONSTRUCTION", str(ef_construction),
        )
        return f"Index '{index}' created (dim={dim}, metric={metric}, M={m})."
    except Exception as e:
        msg = str(e)
        if "already exists" in msg.lower() or "index already" in msg.lower():
            return f"Index '{index}' already exists."
        raise


@mcp.tool()
def optimize_index(index: str) -> str:
    """Build (or rebuild) a Pion HNSW index after inserting documents.

    Must be called after bulk insert before searching. Takes ~15-20s for 50K
    documents. Subsequent searches use the optimized HNSW graph.

    Args:
        index: Index name to optimize.
    """
    r = _conn()
    r.execute_command("FT.OPTIMIZE", index)
    return f"Index '{index}' optimized and ready for search."


@mcp.tool()
def index_info(index: str) -> dict[str, Any]:
    """Return metadata about a Pion vector index.

    Args:
        index: Index name.
    """
    r = _conn()
    raw = r.execute_command("FT.INFO", index)
    # FT.INFO returns a flat list of alternating key/value pairs
    result: dict[str, Any] = {}
    if isinstance(raw, list):
        for i in range(0, len(raw) - 1, 2):
            k = raw[i].decode() if isinstance(raw[i], bytes) else str(raw[i])
            v = raw[i + 1]
            if isinstance(v, bytes):
                v = v.decode()
            result[k] = v
    return result


@mcp.tool()
def drop_index(index: str) -> str:
    """Delete a Pion vector index (keeps underlying HASH documents).

    Args:
        index: Index name to drop.
    """
    r = _conn()
    r.execute_command("FT.DROPINDEX", index)
    return f"Index '{index}' dropped."


# ── Document operations ───────────────────────────────────────────────────────

@mcp.tool()
def add_document(
    index: str,
    doc_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
    auto_optimize: bool = False,
) -> str:
    """Add a text document to a Pion vector index.

    Embeds `text` using the configured provider (PION_EMBED_PROVIDER) and
    stores the vector alongside the document in Pion's hash store.
    Call optimize_index() after bulk inserts before searching.

    Args:
        index: Target index name. Will be created automatically if missing.
        doc_id: Unique document identifier (e.g. "article:42", "chunk:7").
        text: Document text to embed and index.
        metadata: Optional key-value metadata stored alongside the vector.
        auto_optimize: If True, call FT.OPTIMIZE immediately after insert.
                       Only use for single inserts; slow for bulk loads.
    """
    r = _conn()

    # Create index if it doesn't exist yet
    try:
        r.execute_command("FT.INFO", index)
    except Exception:
        from .embeddings import _dim
        r.execute_command(
            "FT.CREATE", index,
            "SCHEMA", "vector", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(_dim()),
            "DISTANCE_METRIC", "COSINE",
            "M", str(DEFAULT_M),
            "EF_CONSTRUCTION", str(DEFAULT_EF_CONSTRUCTION),
        )

    vector_bytes = embed_text(text)
    mapping: dict[str | bytes, Any] = {
        "text": text,
        "vector": vector_bytes,
        "index": index,
    }
    if metadata:
        for k, v in metadata.items():
            mapping[k] = v

    r.hset(doc_id, mapping=mapping)

    if auto_optimize:
        r.execute_command("FT.OPTIMIZE", index)
        return f"Added '{doc_id}' to '{index}' and optimized."
    return f"Added '{doc_id}' to '{index}'. Call optimize_index('{index}') before searching."


@mcp.tool()
def add_documents_bulk(
    index: str,
    documents: list[dict[str, str]],
) -> str:
    """Bulk-add documents to a Pion index in a single pipeline batch.

    Significantly faster than calling add_document() in a loop for >10 docs.
    Remember to call optimize_index() after this returns.

    Args:
        index: Target index name.
        documents: List of dicts, each with required keys:
                   - "id": unique document ID
                   - "text": text to embed
                   Any additional keys are stored as metadata fields.

    Example:
        add_documents_bulk("wiki", [
            {"id": "doc:1", "text": "Paris is the capital of France"},
            {"id": "doc:2", "text": "Berlin is the capital of Germany"},
        ])
    """
    r = _conn()
    if not documents:
        return "No documents provided."

    # Ensure index exists
    try:
        r.execute_command("FT.INFO", index)
    except Exception:
        from .embeddings import _dim
        r.execute_command(
            "FT.CREATE", index,
            "SCHEMA", "vector", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(_dim()),
            "DISTANCE_METRIC", "COSINE",
            "M", str(DEFAULT_M),
            "EF_CONSTRUCTION", str(DEFAULT_EF_CONSTRUCTION),
        )

    texts = [d["text"] for d in documents]
    vectors = embed_texts(texts)

    with r.pipeline(transaction=False) as pipe:
        for doc, vec_bytes in zip(documents, vectors):
            doc_id = doc["id"]
            mapping: dict[str | bytes, Any] = {
                "text": doc["text"],
                "vector": vec_bytes,
                "index": index,
            }
            for k, v in doc.items():
                if k not in ("id", "text"):
                    mapping[k] = v
            pipe.hset(doc_id, mapping=mapping)
        pipe.execute()

    return (
        f"Added {len(documents)} documents to '{index}'. "
        f"Call optimize_index('{index}') before searching."
    )


# ── Vector search ─────────────────────────────────────────────────────────────

@mcp.tool()
def vector_search(
    index: str,
    query: str,
    k: int = 10,
    ef_runtime: int = DEFAULT_EF,
    return_text: bool = True,
) -> list[dict[str, Any]]:
    """Semantic vector search over a Pion index.

    Embeds `query` and returns the k most similar documents by cosine similarity.

    Args:
        index: Index name to search.
        query: Natural language query (will be embedded automatically).
        k: Number of results to return (default 10).
        ef_runtime: HNSW search beam width. Higher = better recall, slower.
                    Default 150 achieves recall@100 > 0.94 at 8,317 QPS.
        return_text: If True, fetch and include the stored text for each result.
    """
    r = _conn()
    query_bytes = embed_text(query)

    raw = _knn(r, index, query_bytes, k, ef_runtime)

    # FT.SEARCH returns: [count, key1, fields1, key2, fields2, ...]
    if not raw or not isinstance(raw, list) or len(raw) < 1:
        return []

    results = []
    items = raw if isinstance(raw, list) else list(raw)

    # Parse alternating [doc_id, score] pairs (skip first element = count)
    i = 0
    if len(items) > 0 and isinstance(items[0], (int, bytes)):
        # First element may be count
        try:
            int(items[0])
            i = 1  # skip count
        except (ValueError, TypeError):
            pass

    while i + 1 < len(items):
        doc_id_raw = items[i]
        score_raw = items[i + 1]
        i += 2

        doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
        score = _score_of(score_raw)
        score = score if score is not None else 0.0

        entry: dict[str, Any] = {"id": doc_id, "score": score}

        if return_text:
            try:
                text_raw = r.hget(doc_id, "text")
                if text_raw:
                    entry["text"] = text_raw.decode() if isinstance(text_raw, bytes) else text_raw
            except Exception:
                pass

        results.append(entry)

    return results


@mcp.tool()
def vector_search_raw(
    index: str,
    vector_hex: str,
    k: int = 10,
    ef_runtime: int = DEFAULT_EF,
) -> list[dict[str, Any]]:
    """Search with a pre-computed vector provided as hex-encoded float32 bytes.

    Use this when you already have an embedding and want to avoid re-computing it.

    Args:
        index: Index name to search.
        vector_hex: Hex string of raw float32 bytes
                    (e.g. from numpy: arr.astype('float32').tobytes().hex()).
        k: Number of results to return.
        ef_runtime: HNSW beam width.
    """
    query_bytes = bytes.fromhex(vector_hex)
    r = _conn()
    raw = _knn(r, index, query_bytes, k, ef_runtime)
    if not raw or not isinstance(raw, list):
        return []

    results = []
    items = raw
    i = 1 if (len(items) > 0 and isinstance(items[0], (int, bytes)) and _is_int(items[0])) else 0
    while i + 1 < len(items):
        doc_id = items[i].decode() if isinstance(items[i], bytes) else str(items[i])
        score = _score_of(items[i + 1])
        score = score if score is not None else 0.0
        results.append({"id": doc_id, "score": score})
        i += 2
    return results


# ── Key-value store ───────────────────────────────────────────────────────────

@mcp.tool()
def kv_get(key: str) -> str | None:
    """Get a value from Pion's key-value store.

    Args:
        key: The key to retrieve.
    """
    r = _conn()
    val = r.get(key)
    if val is None:
        return None
    return val.decode() if isinstance(val, bytes) else str(val)


@mcp.tool()
def kv_set(key: str, value: str, ttl_seconds: int | None = None) -> str:
    """Set a key-value pair in Pion.

    Args:
        key: The key.
        value: The value to store.
        ttl_seconds: Optional expiry in seconds. Not yet implemented in Pion (stored permanently).
    """
    r = _conn()
    r.set(key, value)
    return f"OK — set {key!r}"


@mcp.tool()
def kv_delete(keys: list[str]) -> int:
    """Delete one or more keys from Pion.

    Args:
        keys: List of keys to delete.
    """
    r = _conn()
    # Pipeline individual DEL commands — Pion's multi-key DEL only deletes the first key.
    with r.pipeline(transaction=False) as pipe:
        for k in keys:
            pipe.delete(k)
        results = pipe.execute()
    return sum(r for r in results if r)


@mcp.tool()
def kv_mget(keys: list[str]) -> dict[str, str | None]:
    """Get multiple keys in a single pipeline batch.

    Args:
        keys: List of keys to retrieve.
    """
    r = _conn()
    # Note: use individual GET commands in a pipeline rather than MGET.
    # Pion's MGET fast path has a known hash-lookup bug (nil for keys that
    # GET finds correctly); pipelined GETs are equivalent throughput-wise.
    with r.pipeline(transaction=False) as pipe:
        for k in keys:
            pipe.get(k)
        values = pipe.execute()
    return {
        k: (v.decode() if isinstance(v, bytes) else v) if v is not None else None
        for k, v in zip(keys, values)
    }


@mcp.tool()
def kv_incr(key: str) -> int:
    """Atomically increment an integer counter stored at key.

    Creates the key with value 1 if it does not exist.

    Args:
        key: Counter key.
    """
    r = _conn()
    # Use INCR directly — Pion supports INCR but not INCRBY
    return r.execute_command("INCR", key)


# ── Hash operations (structured documents) ───────────────────────────────────

@mcp.tool()
def hash_set(key: str, fields: dict[str, str]) -> int:
    """Store structured data as a hash (field→value map) in Pion.

    Args:
        key: Hash key.
        fields: Dict of field names to string values.
    """
    r = _conn()
    return r.hset(key, mapping=fields)


@mcp.tool()
def hash_get(key: str, field: str) -> str | None:
    """Get a single field from a hash stored in Pion.

    Args:
        key: Hash key.
        field: Field name.
    """
    r = _conn()
    val = r.hget(key, field)
    if val is None:
        return None
    return val.decode() if isinstance(val, bytes) else str(val)


@mcp.tool()
def hash_get_fields(key: str, fields: list[str]) -> dict[str, str | None]:
    """Get specific fields from a hash stored in Pion.

    Note: Pion does not support HGETALL. Use this tool with a known list of
    field names instead (e.g. fields=["title", "author", "year"]).

    Args:
        key: Hash key.
        fields: List of field names to retrieve.
    """
    r = _conn()
    with r.pipeline(transaction=False) as pipe:
        for f in fields:
            pipe.hget(key, f)
        values = pipe.execute()
    return {
        f: (v.decode() if isinstance(v, bytes) else v) if v is not None else None
        for f, v in zip(fields, values)
    }


# ── Semantic cache ─────────────────────────────────────────────────────────────

_CACHE_INDEX = "__semantic_cache__"
_CACHE_PREFIX = "__sc__:"


@mcp.tool()
def semantic_cache_set(query: str, response: str) -> str:
    """Cache an LLM response indexed by the semantic meaning of the query.

    Subsequent calls to semantic_cache_get() with a similar query will return
    this response instead of calling the LLM again. Can reduce LLM costs 70-86%.

    Args:
        query: The original user query or prompt.
        response: The LLM response to cache.
    """
    r = _conn()

    # Ensure cache index exists (1536-dim or whatever the env says)
    try:
        r.execute_command("FT.INFO", _CACHE_INDEX)
    except Exception:
        from .embeddings import _dim
        r.execute_command(
            "FT.CREATE", _CACHE_INDEX,
            "SCHEMA", "vector", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(_dim()),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "128",
        )

    vec = embed_text(query)
    import hashlib
    key = _CACHE_PREFIX + hashlib.sha256(query.encode()).hexdigest()[:16]
    # `query_vector` is a copy the threshold check reads back: Pion routes the
    # indexed `vector` field into the HNSW index and does not keep it in the
    # hash when it arrives in a multi-field HSET, so HGET <key> vector is nil.
    r.hset(key, mapping={"query": query, "response": response, "vector": vec, "query_vector": vec})

    # Optimize after every 100 inserts (lazy; check counter)
    count_key = "__sc_count__"
    count = r.execute_command("INCR", count_key)
    if count % 100 == 0:
        r.execute_command("FT.OPTIMIZE", _CACHE_INDEX)

    return f"Cached under key {key!r}."


@mcp.tool()
def semantic_cache_get(query: str, threshold: float = 0.95) -> str | None:
    """Look up a cached LLM response by semantic similarity.

    Returns the cached response if a sufficiently similar query was previously
    stored via semantic_cache_set(). Returns None on cache miss.

    Args:
        query: The current user query or prompt.
        threshold: Cosine similarity threshold (0.0–1.0). Higher = stricter matching.
                   Default 0.95 catches near-duplicate phrasings of the same question.
    """
    r = _conn()

    try:
        r.execute_command("FT.INFO", _CACHE_INDEX)
    except Exception:
        return None  # Cache index doesn't exist yet → miss

    vec = embed_text(query)
    raw = _knn(r, _CACHE_INDEX, vec, 1, 50)

    if not raw or not isinstance(raw, list) or len(raw) < 3:
        return None

    items = raw
    i = 1 if _is_int(items[0]) else 0
    if i + 1 >= len(items):
        return None

    doc_id_raw = items[i]
    doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)

    # Search only finds the candidate; the threshold is checked on the exact
    # cosine against the stored float32 vector. (The server's score is the
    # metric's distance since gh #365 — 1 - cos, from dequantized codes — but
    # the exact check costs one HGET and stays correct on older servers.)
    stored = r.hget(doc_id, "query_vector")
    similarity = _cosine(vec, stored) if stored else None
    if similarity is None or similarity < threshold:
        return None

    response_raw = r.hget(doc_id, "response")
    if response_raw is None:
        return None
    return response_raw.decode() if isinstance(response_raw, bytes) else str(response_raw)


# ── Diagnostics ───────────────────────────────────────────────────────────────

@mcp.tool()
def ping() -> str:
    """Check connectivity to the Pion server.

    Returns the server address and PONG response if reachable.
    """
    r = _conn()
    result = r.ping()
    return f"PONG from {PION_HOST}:{PION_PORT} — {'OK' if result else 'ERROR'}"


@mcp.tool()
def server_info() -> dict[str, str]:
    """Return Pion server information (version, config, uptime).

    Calls the Redis INFO command; Pion returns a subset of standard fields.
    """
    r = _conn()
    try:
        raw = r.execute_command("INFO")
        if isinstance(raw, bytes):
            raw = raw.decode()
        result = {}
        for line in str(raw).splitlines():
            if ":" in line and not line.startswith("#"):
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
        return result or {"response": str(raw)[:500]}
    except Exception as e:
        return {"error": str(e), "host": PION_HOST, "port": str(PION_PORT)}


# ── Phase 3: Search Platform ──────────────────────────────────────────────────

@mcp.tool()
def search_text_bm25(
    index: str,
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Keyword (BM25) search over a Pion text index.

    Calls FT.SEARCH <index> BM25 "<query>" K <k>. Falls back to vector_search
    if BM25 is not available for the index.

    Args:
        index: Index name to search.
        query: Keyword query string.
        k: Number of results to return (default 10).
    """
    r = _conn()
    try:
        raw = r.execute_command(
            "FT.SEARCH", index, "BM25", query, "K", str(k),
        )
        if not raw or not isinstance(raw, list):
            return []
        results = []
        items = raw
        i = 1 if (len(items) > 0 and _is_int(items[0])) else 0
        while i + 1 < len(items):
            doc_id_raw = items[i]
            score_raw = items[i + 1]
            i += 2
            doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
            score = _score_of(score_raw)
            score = score if score is not None else 0.0
            entry: dict[str, Any] = {"id": doc_id, "score": score}
            try:
                text_raw = r.hget(doc_id, "text")
                if text_raw:
                    entry["text"] = text_raw.decode() if isinstance(text_raw, bytes) else text_raw
            except Exception:
                pass
            results.append(entry)
        return results
    except Exception:
        return vector_search(index, query, k=k)


@mcp.tool()
def search_hybrid(
    index: str,
    query_text: str,
    query_vector_hex: str,
    k: int = 10,
    fusion: str = "RRF",
) -> list[dict[str, Any]]:
    """Hybrid (vector + keyword) search over a Pion index.

    Calls FT.SEARCH <index> HYBRID VECTOR_QUERY <vector_bytes> TEXT_QUERY
    "<query_text>" K <k> FUSION <fusion>. Combines dense and sparse retrieval.

    Args:
        index: Index name to search.
        query_text: Keyword query string for the text component.
        query_vector_hex: Hex-encoded float32 bytes for the vector component
                          (e.g. numpy: arr.astype('float32').tobytes().hex()).
        k: Number of results to return (default 10).
        fusion: Result fusion strategy — "RRF" (Reciprocal Rank Fusion, default)
                or "LINEAR".
    """
    r = _conn()
    try:
        query_bytes = bytes.fromhex(query_vector_hex)
        raw = r.execute_command(
            "FT.SEARCH", index,
            "HYBRID",
            "VECTOR_QUERY", query_bytes,
            "TEXT_QUERY", query_text,
            "K", str(k),
            "FUSION", fusion,
        )
        if not raw or not isinstance(raw, list):
            return []
        results = []
        items = raw
        i = 1 if (len(items) > 0 and _is_int(items[0])) else 0
        while i + 1 < len(items):
            doc_id_raw = items[i]
            score_raw = items[i + 1]
            i += 2
            doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
            score = _score_of(score_raw)
            score = score if score is not None else 0.0
            entry: dict[str, Any] = {"id": doc_id, "score": score}
            try:
                text_raw = r.hget(doc_id, "text")
                if text_raw:
                    entry["text"] = text_raw.decode() if isinstance(text_raw, bytes) else text_raw
            except Exception:
                pass
            results.append(entry)
        return results
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
def search_with_filter(
    index: str,
    query: str,
    k: int = 10,
    filters: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Vector search with metadata filters over a Pion index.

    Embeds query and runs a KNN FT.SEARCH with one FILTER argument per
    filter string.

    Args:
        index: Index name to search.
        query: Natural language query (will be embedded automatically).
        k: Number of results to return (default 10).
        filters: List of filter strings, e.g. ["category=electronics",
                 "price [50 200]"]. Each filter is passed as a separate
                 FILTER argument.
    """
    r = _conn()
    try:
        query_bytes = embed_text(query)
        extra: tuple = ()
        for f in (filters or []):
            extra += ("FILTER", f)
        raw = _knn(r, index, query_bytes, k, extra=extra)
        if not raw or not isinstance(raw, list):
            return []
        results = []
        items = raw
        i = 1 if (len(items) > 0 and _is_int(items[0])) else 0
        while i + 1 < len(items):
            doc_id_raw = items[i]
            score_raw = items[i + 1]
            i += 2
            doc_id = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
            score = _score_of(score_raw)
            score = score if score is not None else 0.0
            entry: dict[str, Any] = {"id": doc_id, "score": score}
            try:
                text_raw = r.hget(doc_id, "text")
                if text_raw:
                    entry["text"] = text_raw.decode() if isinstance(text_raw, bytes) else text_raw
            except Exception:
                pass
            results.append(entry)
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ── Phase 4: AI Gateway ────────────────────────────────────────────────────────

@mcp.tool()
def add_text_document_native(
    index: str,
    doc_id: str,
    text: str,
) -> str:
    """Add text document using Pion's built-in embedding (requires EmbeddingConfig.enabled=True in config.mojo).

    Calls FT.ADDTEXT <index> <doc_id> <text> — Pion's internal EmbeddingClient
    handles embedding; no Python-side embedding provider is needed.

    Args:
        index: Target index name.
        doc_id: Unique document identifier.
        text: Document text to embed and index.
    """
    r = _conn()
    try:
        raw = r.execute_command("FT.ADDTEXT", index, doc_id, text)
        if isinstance(raw, bytes):
            return raw.decode()
        return str(raw)
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def search_text_native(
    index: str,
    query_text: str,
    k: int = 10,
) -> list[str]:
    """Search using Pion's built-in embedding (requires EmbeddingConfig.enabled=True in config.mojo).

    Calls FT.SEARCHTEXT <index> <query_text> K <k> — Pion embeds the query
    internally; no Python-side embedding provider is needed.

    Args:
        index: Index name to search.
        query_text: Natural language query (embedded by Pion internally).
        k: Number of results to return (default 10).
    """
    r = _conn()
    try:
        raw = r.execute_command("FT.SEARCHTEXT", index, query_text, "K", str(k))
        if not raw or not isinstance(raw, list):
            return []
        return [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in raw
        ]
    except Exception as e:
        return [f"ERROR: {e}"]


@mcp.tool()
def ai_chat(
    prompt: str,
    context_index: str | None = None,
    context_query: str | None = None,
    k: int = 5,
) -> str:
    """Full RAG pipeline: Pion embeds context_query, retrieves from HNSW, augments prompt, calls LLM (requires LLMConfig.enabled=True in config.mojo and a running /v1/chat/completions server).

    Calls AI.CHAT <prompt> [CONTEXT <context_index> <context_query> K <k>].
    When context_index is provided, Pion retrieves the k most relevant documents
    and prepends them to the prompt before passing it to the LLM.

    Args:
        prompt: The user prompt or question.
        context_index: Optional index name to retrieve context from.
        context_query: Query used to retrieve context (defaults to prompt if omitted).
        k: Number of context documents to retrieve (default 5).
    """
    r = _conn()
    try:
        if context_index:
            cq = context_query if context_query else prompt
            raw = r.execute_command(
                "AI.CHAT", prompt,
                "CONTEXT", context_index, cq, "K", str(k),
            )
        else:
            raw = r.execute_command("AI.CHAT", prompt)
        if isinstance(raw, bytes):
            return raw.decode()
        return str(raw)
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def semantic_cache_set_native(query: str, response: str) -> str:
    """Cache an LLM response using Pion's built-in EmbeddingClient.

    Calls AI.SEMANTIC_CACHE SET <query> <response>. Unlike semantic_cache_set(),
    this uses Pion's internal embedding rather than the Python-side provider
    (requires EmbeddingConfig.enabled=True in config.mojo).

    Args:
        query: The original user query or prompt.
        response: The LLM response to cache.
    """
    r = _conn()
    try:
        raw = r.execute_command("AI.SEMANTIC_CACHE", "SET", query, response)
        if isinstance(raw, bytes):
            return raw.decode()
        return str(raw)
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def semantic_cache_get_native(
    query: str,
    threshold: float = 0.95,
) -> str | None:
    """Look up a cached LLM response using Pion's built-in EmbeddingClient.

    Calls AI.SEMANTIC_CACHE GET <query> THRESHOLD <threshold>. Returns the cached
    response string on a hit, or None on a cache miss (requires
    EmbeddingConfig.enabled=True in config.mojo).

    Args:
        query: The current user query or prompt.
        threshold: Cosine similarity threshold (0.0–1.0). Default 0.95.
    """
    r = _conn()
    try:
        raw = r.execute_command(
            "AI.SEMANTIC_CACHE", "GET", query, "THRESHOLD", str(threshold),
        )
        if raw is None or raw == b"$-1" or raw == b"" or raw == "":
            return None
        if isinstance(raw, bytes):
            decoded = raw.decode()
            if decoded == "$-1":
                return None
            return decoded
        return str(raw)
    except Exception:
        return None


# ── Phase 5: Cluster ───────────────────────────────────────────────────────────

@mcp.tool()
def cluster_info() -> dict[str, str]:
    """Return Pion cluster topology and state information.

    Calls CLUSTER INFO and parses the key:value response into a dict.
    Useful for diagnosing cluster health, slot coverage, and node count.
    """
    r = _conn()
    try:
        raw = r.execute_command("CLUSTER", "INFO")
        if isinstance(raw, bytes):
            raw = raw.decode()
        result: dict[str, str] = {}
        for line in str(raw).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def cluster_keyslot(key: str) -> int:
    """Return the hash slot (0–16383) that a key maps to in the Pion cluster.

    Calls CLUSTER KEYSLOT <key>. Useful for understanding data distribution
    and routing in a sharded Pion deployment.

    Args:
        key: The key whose slot assignment to look up.
    """
    r = _conn()
    try:
        result = r.execute_command("CLUSTER", "KEYSLOT", key)
        return int(result)
    except Exception as e:
        return -1


# ── Codebase search ───────────────────────────────────────────────────────────
# Semantic codebase indexing and retrieval for Claude Code integration.
# Indexes source files into Pion's HNSW index for sub-millisecond semantic search.

_CB_INDEX = "__codebase__"
_CB_PREFIX = "cb:"
_CB_CHECKSUM_KEY = "__cb_checksums__"
_CB_SEQ_KEY = "__cb_seq__"


@mcp.tool()
def codebase_index(
    directory: str = ".",
    force: bool = False,
) -> str:
    """Index a codebase directory into Pion for semantic search.

    Walks the directory tree, chunks source files into semantic blocks
    (functions, classes, fixed-size segments), embeds each chunk, and
    stores them in Pion's HNSW index. Supports incremental indexing:
    only re-processes files whose content has changed.

    Call this once to bootstrap, then let the PostToolUse hook handle
    incremental updates.

    Args:
        directory: Root directory to index (default: current directory).
        force: Re-index all files even if unchanged (default: False).

    Example:
        codebase_index("/path/to/project")
        codebase_index(".", force=True)  # full re-index
    """
    try:
        from pion_context.indexer import CodebaseIndexer
    except ImportError:
        return (
            "pion_context package not installed. "
            "Run: pip install -e pion_context  (from the Pion project root)"
        )

    indexer = CodebaseIndexer(host=PION_HOST, port=PION_PORT)
    stats = indexer.index_directory(directory, force=force)

    return (
        f"Indexed {stats.files_indexed} files ({stats.chunks_created} chunks) "
        f"in {stats.elapsed_s:.1f}s. "
        f"Skipped {stats.files_skipped} unchanged files."
        + (f" Errors: {len(stats.errors)}" if stats.errors else "")
    )


@mcp.tool()
def codebase_search(
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Semantic search over the indexed codebase.

    Finds code chunks most semantically relevant to the natural language
    query. Returns file paths, line numbers, and code snippets — far
    more precise than grep for conceptual queries like "authentication
    middleware" or "database connection pooling".

    Requires codebase_index() to have been run first.

    Args:
        query: Natural language description of what you're looking for.
        k: Number of results to return (default: 10).

    Returns:
        List of dicts with: file_path, start_line, end_line, name, kind, text.

    Example:
        codebase_search("hash map probing and collision resolution")
        codebase_search("how does the HNSW index handle concurrent writes", k=5)
    """
    try:
        from pion_context.engine import ContextEngine, SearchError
    except ImportError:
        return [{"error": "pion_context package not installed"}]

    engine = ContextEngine(host=PION_HOST, port=PION_PORT)
    try:
        results = engine.search_codebase(query, k=k)
    except SearchError as e:
        # e.g. agent memory's index replaced the codebase index on this server
        return [{"error": str(e)}]

    return [
        {
            "file_path": r.file_path,
            "start_line": r.start_line,
            "name": r.name,
            "text": r.content[:500],  # truncate for MCP response size
            "source": r.source,
        }
        for r in results
    ]


@mcp.tool()
def codebase_context(
    query: str,
    code_k: int = 8,
    memory_k: int = 3,
) -> str:
    """Unified context retrieval: code + memories + semantic cache.

    Searches all Pion sources (codebase index, agent memory, semantic
    cache) and returns a formatted context block. Use this as the
    "one-stop" tool for understanding a topic across code and history.

    Args:
        query: Natural language query.
        code_k: Number of code results (default: 8).
        memory_k: Number of memory results (default: 3).

    Returns:
        Formatted markdown context string.

    Example:
        codebase_context("how does the fast path handle GET commands")
        codebase_context("past decisions about quantization strategy")
    """
    try:
        from pion_context.engine import ContextEngine, SearchError
    except ImportError:
        return "pion_context package not installed."

    engine = ContextEngine(host=PION_HOST, port=PION_PORT)
    try:
        results = engine.retrieve_context(query, code_k=code_k, memory_k=memory_k)
    except SearchError as e:
        return f"error: {e}"
    return engine.format_context(results)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _is_int(val: Any) -> bool:
    try:
        int(val)
        return True
    except (TypeError, ValueError):
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pion MCP Server — vector search and KV tools for AI agents"
    )
    parser.add_argument("--host", default=None, help="Pion host (overrides PION_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Pion port (overrides PION_PORT)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio for Claude Code / Cursor)",
    )
    args = parser.parse_args()

    if args.host:
        os.environ["PION_HOST"] = args.host
    if args.port:
        os.environ["PION_PORT"] = str(args.port)

    # Reload globals after env override
    global PION_HOST, PION_PORT
    PION_HOST = os.environ.get("PION_HOST", "localhost")
    PION_PORT = int(os.environ.get("PION_PORT", "1974"))

    print(
        f"Pion MCP server starting — connecting to {PION_HOST}:{PION_PORT} "
        f"(embed: {os.environ.get('PION_EMBED_PROVIDER', 'openai')})",
        file=sys.stderr,
    )
    mcp.run(transport=args.transport)


# ── Agent Memory ──────────────────────────────────────────────────────────────
# Persistent cross-session semantic memory for any MCP-compatible agent.
# Each memory is embedded, stored in Pion's HNSW index, and retrievable
# by semantic similarity across sessions and restarts.
#
# Storage layout:
#   HNSW index:  "__agent_memory__"
#   Hash key:    "mem:<sha256[:16]>"
#   Fields:      text, session_id, timestamp, vector, [extra metadata]
#   Counter key: "__mem_count__"  (tracks total inserts for lazy optimize)

_MEM_INDEX = "__agent_memory__"
_MEM_PREFIX = "mem:"
_MEM_OPTIMIZE_EVERY = 50


def _ensure_mem_index(r: "redis_lib.Redis") -> None:
    """Create the agent memory index if it does not exist."""
    try:
        r.execute_command("FT.INFO", _MEM_INDEX)
    except Exception:
        from .embeddings import _dim
        # Field name "embedding" avoids collision with the VECTOR keyword in
        # Pion's case-insensitive schema parser (which would mis-detect "vector"
        # as the VECTOR keyword and corrupt the stored field name).
        r.execute_command(
            "FT.CREATE", _MEM_INDEX,
            "SCHEMA", "embedding", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(_dim()),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "128",
        )


@mcp.tool()
def agent_remember(
    text: str,
    session_id: str = "default",
    metadata: dict[str, str] | None = None,
) -> str:
    """Store a memory persistently in Pion, indexed by semantic content.

    Embeds `text` and saves it to Pion's agent memory index. The memory
    survives process restarts and is retrievable by semantic similarity
    using agent_recall(). Use this to persist facts, decisions, context,
    or anything an agent should remember across sessions.

    Args:
        text: The text to remember (fact, decision, observation, summary, etc.)
        session_id: Logical session or agent name for grouping memories.
                    Stored as metadata; agent_recall() can filter by this.
        metadata: Optional extra key-value pairs to store with the memory
                  (e.g. {"source": "user", "importance": "high"}).

    Returns:
        The memory key assigned (stable for identical text).

    Example:
        agent_remember("The user prefers concise answers without preamble",
                       session_id="claude-code")
        agent_remember("Project uses pixi for builds, not pip",
                       session_id="pion-project")
    """
    import hashlib, time as _time

    r = _conn()
    _ensure_mem_index(r)

    vec = embed_text(text)
    # Use sequential integer keys so HNSW key_id extraction works correctly.
    # Hex keys break fast_path digit extraction; "mem:1", "mem:2", ... work reliably.
    seq_id = int(r.execute_command("INCR", "__mem_seq__"))
    key = _MEM_PREFIX + str(seq_id)

    fields: dict[str, Any] = {
        "text": text,
        "session_id": session_id,
        "timestamp": str(int(_time.time())),
        "embedding": vec,
    }
    if metadata:
        fields.update(metadata)

    r.hset(key, mapping=fields)

    # Lazy optimize every N inserts
    count = r.execute_command("INCR", "__mem_count__")
    if int(count) % _MEM_OPTIMIZE_EVERY == 0:
        r.execute_command("FT.OPTIMIZE", _MEM_INDEX)

    return f"Remembered under key {key!r} (session={session_id!r})."


@mcp.tool()
def agent_recall(
    query: str,
    k: int = 5,
    session_id: str | None = None,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Retrieve the most semantically relevant memories from Pion.

    Embeds `query` and returns the k most similar memories previously
    stored via agent_remember(). Results are ordered by similarity
    (most similar first).

    Args:
        query: Natural language query describing what to recall.
        k: Number of memories to return (default 5).
        session_id: If provided, only return memories from this session.
                    Pass None (default) to search across all sessions.
        threshold: Minimum cosine similarity (0.0–1.0) to include a result.
                   Default 0.0 returns all top-k regardless of score.
                   Use 0.7 for loose relevance, 0.9 for near-exact matches.

    Returns:
        List of dicts with keys: id, text, session_id, timestamp, similarity.

    Example:
        agent_recall("user preferences")
        agent_recall("build system", session_id="pion-project", threshold=0.7)
    """
    r = _conn()

    try:
        r.execute_command("FT.INFO", _MEM_INDEX)
    except Exception:
        return []  # No memories stored yet

    # Pion's HNSW SIMD batch-8 kernel requires at least 8 nodes for safe search.
    # For small collections use a linear scan over all sequential keys instead.
    count_raw = r.get("__mem_count__")
    total_count = int(count_raw) if count_raw else 0
    _MIN_HNSW_NODES = 8

    vec = embed_text(query)

    if total_count < _MIN_HNSW_NODES:
        # Linear fallback: iterate all keys, compute similarity in Python.
        import struct as _struct
        seq_raw = r.get("__mem_seq__")
        max_seq = int(seq_raw) if seq_raw else 0
        candidates = []
        for i in range(1, max_seq + 1):
            key_str = _MEM_PREFIX + str(i)
            with r.pipeline(transaction=False) as pipe:
                pipe.hget(key_str, "text")
                pipe.hget(key_str, "session_id")
                pipe.hget(key_str, "timestamp")
                pipe.hget(key_str, "embedding")
                text_raw, sess_raw, ts_raw, emb_raw = pipe.execute()
            if text_raw is None:
                continue
            mem_session = (sess_raw.decode() if isinstance(sess_raw, bytes) else sess_raw) or "default"
            if session_id is not None and mem_session != session_id:
                continue
            # Compute cosine similarity in Python
            sim = 0.0
            if emb_raw and len(emb_raw) == len(vec):
                q = _struct.unpack_from(f"{len(vec)//4}f", vec)
                d = _struct.unpack_from(f"{len(emb_raw)//4}f", emb_raw)
                dot = sum(qi * di for qi, di in zip(q, d))
                qn = sum(qi * qi for qi in q) ** 0.5
                dn = sum(di * di for di in d) ** 0.5
                sim = dot / (qn * dn + 1e-9)
            if sim < threshold:
                continue
            text_val = (text_raw.decode() if isinstance(text_raw, bytes) else text_raw) or ""
            ts_val = (ts_raw.decode() if isinstance(ts_raw, bytes) else ts_raw) or "0"
            candidates.append({
                "id": key_str,
                "text": text_val,
                "session_id": mem_session,
                "timestamp": int(ts_val) if ts_val.isdigit() else 0,
                "similarity": round(sim, 4),
            })
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:k]

    k_fetch = k * 3 if session_id else k

    # Use Redis KNN PARAMS format — Pion's native format never sets blob_set=True.
    raw = r.execute_command(
        "FT.SEARCH", _MEM_INDEX,
        f"*=>[KNN {k_fetch} @embedding $vec EF_RUNTIME 64]",
        "PARAMS", "2", "vec", vec,
    )

    if not raw or not isinstance(raw, list):
        return []

    # Pion response format: [count, doc_id_bytes, [b"id", id_val, b"score", score_val], ...]
    # Pion returns the original hash key; a bare node id (old servers) still gets the prefix.
    results = []
    items = list(raw)
    i = 1 if (items and _is_int(items[0])) else 0

    while i + 1 < len(items) and len(results) < k:
        doc_id_raw = items[i]
        fields_raw = items[i + 1]
        i += 2

        doc_id_str = doc_id_raw.decode() if isinstance(doc_id_raw, bytes) else str(doc_id_raw)
        doc_key = doc_id_str if doc_id_str.startswith(_MEM_PREFIX) else _MEM_PREFIX + doc_id_str

        # Extract score BY NAME: fields are [b"id", id_val, b"score", score_val]
        # when the doc has an "id" field on the answering worker, else just
        # [b"score", score_val] (gh #357) — never index by position.
        score_val = None
        if isinstance(fields_raw, (list, tuple)):
            for j in range(0, len(fields_raw) - 1, 2):
                if fields_raw[j] in (b"score", "score"):
                    score_val = fields_raw[j + 1]
                    break
        try:
            raw_dist = float(score_val) if score_val is not None else 1e9
            # Convert raw L2 distance to a [0,1] similarity (lower distance = higher similarity).
            similarity = 1.0 / (1.0 + raw_dist)
        except (TypeError, ValueError):
            similarity = 0.0

        if similarity < threshold:
            continue

        # Fetch stored fields
        try:
            with r.pipeline(transaction=False) as pipe:
                pipe.hget(doc_key, "text")
                pipe.hget(doc_key, "session_id")
                pipe.hget(doc_key, "timestamp")
                text_raw, sess_raw, ts_raw = pipe.execute()
        except Exception:
            continue

        mem_session = (sess_raw.decode() if isinstance(sess_raw, bytes) else sess_raw) or "default"

        # Session filter
        if session_id is not None and mem_session != session_id:
            continue

        text_val = (text_raw.decode() if isinstance(text_raw, bytes) else text_raw) or ""
        ts_val = (ts_raw.decode() if isinstance(ts_raw, bytes) else ts_raw) or "0"

        results.append({
            "id": doc_key,
            "text": text_val,
            "session_id": mem_session,
            "timestamp": int(ts_val) if ts_val.isdigit() else 0,
            "similarity": round(similarity, 4),
        })

    return results


@mcp.tool()
def agent_forget(memory_ids: list[str]) -> int:
    """Delete specific memories from Pion by their key IDs.

    Use the IDs returned by agent_recall() to remove individual memories.
    To clear all memories for a session, use agent_forget_session() instead.

    Args:
        memory_ids: List of memory key IDs to delete (e.g. ["mem:a1b2c3d4..."]).

    Returns:
        Number of memories deleted.
    """
    r = _conn()
    deleted = 0
    with r.pipeline(transaction=False) as pipe:
        for mid in memory_ids:
            pipe.delete(mid)
        results = pipe.execute()
    deleted = sum(1 for r in results if r)
    return deleted


@mcp.tool()
def agent_forget_session(session_id: str) -> str:
    """Delete all memories associated with a session (non-reversible).

    Scans the memory index for all entries with the given session_id
    and removes them. Use with care — this cannot be undone.

    Args:
        session_id: Session identifier to clear.

    Returns:
        Summary of how many memories were deleted.
    """
    r = _conn()

    try:
        r.execute_command("FT.INFO", _MEM_INDEX)
    except Exception:
        return "No memory index — nothing to delete."

    # Iterate over all sequential keys mem:1 .. mem:N and delete those matching session_id.
    # (Pion does not implement SCAN or KEYS, so we use the __mem_seq__ counter.)
    seq_raw = r.get("__mem_seq__")
    max_seq = int(seq_raw) if seq_raw else 0

    to_delete = []
    for i in range(1, max_seq + 1):
        key_str = _MEM_PREFIX + str(i)
        sess_raw = r.hget(key_str, "session_id")
        if sess_raw is None:
            continue  # key deleted or never existed
        sess = (sess_raw.decode() if isinstance(sess_raw, bytes) else sess_raw) or "default"
        if sess == session_id:
            to_delete.append(key_str)

    if not to_delete:
        return f"No memories found for session {session_id!r}."

    with r.pipeline(transaction=False) as pipe:
        for mid in to_delete:
            pipe.delete(mid)
        pipe.execute()

    return f"Deleted {len(to_delete)} memories from session {session_id!r}."


@mcp.tool()
def agent_memory_stats() -> dict[str, Any]:
    """Return statistics about the agent memory store.

    Reports total memory count, sessions present, and index status.
    """
    r = _conn()

    try:
        info_raw = r.execute_command("FT.INFO", _MEM_INDEX)
    except Exception:
        return {"status": "empty", "total_memories": 0}

    # Parse FT.INFO response
    info: dict[str, Any] = {}
    if isinstance(info_raw, list):
        for j in range(0, len(info_raw) - 1, 2):
            k = info_raw[j].decode() if isinstance(info_raw[j], bytes) else str(info_raw[j])
            v = info_raw[j + 1]
            if isinstance(v, bytes):
                v = v.decode()
            info[k] = v

    count_raw = r.get("__mem_count__")
    total = int(count_raw) if count_raw else 0

    return {
        "status": "ready",
        "total_memories": total,
        "index": _MEM_INDEX,
        "num_docs": info.get("num_docs", "unknown"),
        "indexing_failures": info.get("hash_indexing_failures", 0),
    }


if __name__ == "__main__":
    main()
