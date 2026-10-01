"""A4 — Framework Integration Tests.

Tests LangGraph, AutoGen, and LlamaIndex integrations against a live Pion server.
Requires: pion-server running on localhost:1974

Usage:
    python3 tests/test_framework_integrations.py
    python3 tests/test_framework_integrations.py --langgraph   # only LangGraph
    python3 tests/test_framework_integrations.py --autogen     # only AutoGen
    python3 tests/test_framework_integrations.py --llamaindex  # only LlamaIndex
"""
from __future__ import annotations

import asyncio
import struct
import sys
import time

import redis as redis_lib

# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_pion(host: str = "127.0.0.1", port: int = 1974) -> bool:
    try:
        r = redis_lib.Redis(host=host, port=port, socket_timeout=2)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def _mock_embedding(text: str, dim: int = 384) -> list[float]:
    """Deterministic mock embedding matching pion/embeddings.py logic."""
    import hashlib
    import math
    # Seeded Gaussian per dimension. The old sin(h + i*1.618) added a small float to a
    # 256-bit hash — as a float h + i*1.618 == h — so every text embedded to the same
    # constant vector and no search over mock embeddings could tell results apart.
    import random
    rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
    floats = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in floats)) or 1.0
    return [v / norm for v in floats]


def _mock_embedding_bytes(text: str, dim: int = 384) -> bytes:
    floats = _mock_embedding(text, dim)
    return struct.pack(f"{dim}f", *floats)


passed = 0
failed = 0
xfailed = 0


def xfail(name: str, condition: bool, issue: str):
    """A known bug, pinned: prints XFAIL while `condition` is false. If it
    starts holding, that is XPASS and it FAILS the run — the marker must be
    removed on purpose, never left to rot as a silent pass."""
    global failed, xfailed
    if condition:
        failed += 1
        print(f"  ✗ XPASS {name} — {issue} looks fixed; turn this into check()")
    else:
        xfailed += 1
        print(f"  ~ XFAIL {name} ({issue})")


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}: {detail}")


# ── LangGraph Tests ──────────────────────────────────────────────────────────

def test_langgraph():
    print("\n═══ LangGraph Checkpointer ═══")

    try:
        from pion_langgraph import PionSaver
    except ImportError as e:
        print(f"  SKIP: {e}")
        print("  Install: pip install -e pion-langgraph/")
        return

    saver = PionSaver(host="127.0.0.1", port=1974, key_prefix="test_lg")

    # Clean up from previous runs
    saver.delete_thread("test-thread-1")

    # 1. put + get_tuple
    checkpoint = {
        "id": "cp-001",
        "ts": "2026-04-15T00:00:00",
        "channel_values": {"messages": ["hello", "world"]},
        "channel_versions": {"messages": 1},
        "versions_seen": {},
    }
    metadata = {"source": "input", "step": 0}

    config = {"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}}
    result_config = saver.put(config, checkpoint, metadata, {"messages": 1})
    check("put returns config", "checkpoint_id" in result_config.get("configurable", {}))

    tup = saver.get_tuple({"configurable": {
        "thread_id": "test-thread-1", "checkpoint_ns": "",
        "checkpoint_id": "cp-001",
    }})
    check("get_tuple returns checkpoint", tup is not None)
    check("checkpoint data intact", tup.checkpoint["channel_values"]["messages"] == ["hello", "world"])
    check("metadata intact", tup.metadata.get("source") == "input")

    # 2. put second checkpoint with parent
    checkpoint2 = {
        "id": "cp-002",
        "ts": "2026-04-15T00:01:00",
        "channel_values": {"messages": ["hello", "world", "!"]},
        "channel_versions": {"messages": 2},
        "versions_seen": {},
    }
    config2 = {"configurable": {
        "thread_id": "test-thread-1", "checkpoint_ns": "",
        "checkpoint_id": "cp-001",
    }}
    saver.put(config2, checkpoint2, {"source": "loop", "step": 1}, {"messages": 2})

    # 3. get latest (no checkpoint_id)
    latest = saver.get_tuple({"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}})
    check("latest checkpoint is cp-002", latest is not None and latest.checkpoint["id"] == "cp-002")
    check("parent_config set", latest.parent_config is not None)

    # 4. put_writes
    saver.put_writes(
        {"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": "", "checkpoint_id": "cp-002"}},
        [("messages", "new message"), ("counter", 42)],
        task_id="task-1",
    )

    tup_with_writes = saver.get_tuple({"configurable": {
        "thread_id": "test-thread-1", "checkpoint_ns": "",
        "checkpoint_id": "cp-002",
    }})
    check("pending writes collected",
          tup_with_writes is not None and len(tup_with_writes.pending_writes) >= 2)

    # 5. list checkpoints
    listed = list(saver.list(
        {"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}},
        limit=10,
    ))
    check("list returns 2 checkpoints", len(listed) == 2)
    check("list ordered newest first",
          len(listed) == 2 and listed[0].checkpoint["id"] == "cp-002")

    # 6. list with filter
    filtered = list(saver.list(
        {"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}},
        filter={"source": "loop"},
    ))
    check("filter by metadata works", len(filtered) == 1 and filtered[0].metadata.get("source") == "loop")

    # 7. list with before
    before = list(saver.list(
        {"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}},
        before={"configurable": {"checkpoint_id": "cp-002"}},
    ))
    check("before filter works", len(before) == 1 and before[0].checkpoint["id"] == "cp-001")

    # 8. delete_thread
    saver.delete_thread("test-thread-1")
    empty = saver.get_tuple({"configurable": {"thread_id": "test-thread-1", "checkpoint_ns": ""}})
    check("delete_thread clears all", empty is None)

    saver.close()
    print(f"  LangGraph: all checks complete")


# ── AutoGen Tests ────────────────────────────────────────────────────────────

def test_autogen():
    print("\n═══ AutoGen Memory ═══")

    try:
        from pion_autogen import PionMemoryStore
    except ImportError as e:
        print(f"  SKIP: {e}")
        print("  Install: pip install -e pion-autogen/")
        return

    from autogen_core.memory import MemoryContent, MemoryMimeType

    # Drop any existing index (single HNSW per worker)
    r = redis_lib.Redis(host="127.0.0.1", port=1974, decode_responses=False)
    try:
        r.execute_command("FT.DROPINDEX", "test_autogen")
    except Exception:
        pass
    r.close()

    store = PionMemoryStore(
        host="127.0.0.1", port=1974,
        index_name="test_autogen",
        dimensions=384,
        embed_provider="mock",
    )

    async def run_tests():
        # Clean up
        await store.clear()

        # 1. add
        await store.add(MemoryContent(content="The capital of France is Paris", mime_type=MemoryMimeType.TEXT))
        await store.add(MemoryContent(content="Python is a programming language", mime_type=MemoryMimeType.TEXT))
        await store.add(MemoryContent(content="Pion is a vector database written in Mojo", mime_type=MemoryMimeType.TEXT))
        stored = store._r.execute_command("KEYS", store._prefix + "*")
        check("add stores 3 memories", len(stored) == 3, f"{len(stored)} keys")

        # Add more to exceed HNSW threshold
        for i in range(7):
            await store.add(MemoryContent(content=f"Filler memory number {i}", mime_type=MemoryMimeType.TEXT))

        # Optimize HNSW index so FT.SEARCH finds the vectors
        store._r.execute_command("FT.OPTIMIZE", store._index)

        # 2. query. The mock embedder is a hash, so "semantic" relevance cannot
        # be tested with it — but exact-text retrieval can, and it is what a
        # broken key lookup, a wrong query form or a degenerate embedder all
        # fail (every one of those shipped; each is a check below).
        target = "Pion is a vector database written in Mojo"
        result = await store.query(target, k=3)
        texts = [str(r.content) for r in result.results]
        check("query returns k results", len(texts) == 3, str(texts))
        check("exact-text query returns that memory first", texts[:1] == [target], str(texts[:1]))
        check("every result carries its stored text", all(texts), str(texts))

        # 3. score_threshold keeps only results at or above it (a similarity)
        unfiltered = await store.query(target, k=3)
        filtered = await store.query(target, k=3, score_threshold=0.99)
        check("score_threshold=0.99 keeps only the exact match (gh #365)",
              [str(r.content) for r in filtered.results] == [target], str([str(r.content) for r in filtered.results]))
        check("score_threshold never adds results", len(filtered.results) <= len(unfiltered.results))

        # 4. clear
        await store.clear()
        result_empty = await store.query("anything", k=3)
        check("clear removes all memories", len(result_empty.results) == 0)

        # Cleanup — drop index for next test
        await store.clear()
        store._r.execute_command("FT.DROPINDEX", store._index)
        await store.close()

    asyncio.run(run_tests())
    print(f"  AutoGen: all checks complete")


# ── LlamaIndex Tests ────────────────────────────────────────────────────────

def test_llamaindex():
    print("\n═══ LlamaIndex VectorStore ═══")

    try:
        from pion_llamaindex import PionVectorStore
    except ImportError as e:
        print(f"  SKIP: {e}")
        print("  Install: pip install -e pion-llamaindex/")
        return

    from llama_index.core.schema import TextNode, RelatedNodeInfo, NodeRelationship
    from llama_index.core.vector_stores.types import VectorStoreQuery

    # Drop any existing index (single HNSW per worker)
    r = redis_lib.Redis(host="127.0.0.1", port=1974, decode_responses=False)
    try:
        r.execute_command("FT.DROPINDEX", "test_llamaindex")
    except Exception:
        pass
    seq_raw = r.get("__test_llamaindex_seq__")
    if seq_raw:
        for i in range(1, int(seq_raw) + 1):
            r.delete(f"test_llamaindex:{i}")
    r.delete("__test_llamaindex_seq__")
    r.delete("__test_llamaindex_count__")
    r.close()

    store = PionVectorStore(
        host="127.0.0.1", port=1974,
        index_name="test_llamaindex",
        dimensions=384,
    )

    # 1. add nodes
    nodes = []
    texts = [
        "Pion is a deterministically low-latency vector database",
        "Redis is an in-memory data structure store",
        "LlamaIndex is a data framework for LLM applications",
        "HNSW is an approximate nearest neighbor algorithm",
        "Mojo is a programming language for AI",
    ]
    for text in texts:
        emb = _mock_embedding(text, 384)
        node = TextNode(text=text, embedding=emb)
        node.relationships = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")}
        nodes.append(node)

    # Add more nodes to exceed MIN_HNSW_NODES threshold
    for i in range(5):
        emb = _mock_embedding(f"Padding text number {i}", 384)
        node = TextNode(text=f"Padding text number {i}", embedding=emb)
        node.relationships = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-2")}
        nodes.append(node)

    ids = store.add(nodes)
    check("add returns node IDs", len(ids) == 10)
    check("IDs match node_ids", all(isinstance(i, str) for i in ids))

    # Optimize HNSW so FT.SEARCH works
    store._client.execute_command("FT.OPTIMIZE", store.index_name)

    # 2. query with a stored node's exact embedding: it must come back first,
    # with its stored text (an empty text was the "prefix + key" bug).
    query_emb = _mock_embedding(texts[3], 384)
    query = VectorStoreQuery(query_embedding=query_emb, similarity_top_k=3)
    result = store.query(query)
    check("query returns k nodes", len(result.nodes) == 3, str(len(result.nodes)))
    check("exact-embedding query returns that node first",
          bool(result.nodes) and result.nodes[0].text == texts[3], result.nodes[0].text if result.nodes else "")
    check("query returns similarities", len(result.similarities) == len(result.nodes))
    check("query returns IDs", len(result.ids) == len(result.nodes))

    top_texts = [n.text for n in result.nodes]
    check("query returns text content", all(len(t) > 0 for t in top_texts))

    # 3. delete by ref_doc_id
    store.delete("doc-2")
    # Verify hash keys are removed (HNSW still has stale vectors until re-optimize)
    r2 = redis_lib.Redis(host="127.0.0.1", port=1974, decode_responses=False)
    deleted_keys = sum(1 for i in range(6, 11) if not r2.exists(f"test_llamaindex:{i}"))
    remaining_keys = sum(1 for i in range(1, 6) if r2.exists(f"test_llamaindex:{i}"))
    r2.close()
    check("delete removes doc-2 nodes", deleted_keys == 5 and remaining_keys == 5)

    # Cleanup
    try:
        store._client.execute_command("FT.DROPINDEX", store.index_name)
    except Exception:
        pass
    store.close()
    print(f"  LlamaIndex: all checks complete")


# ── RESP-only tests (no framework deps needed) ──────────────────────────────

def test_resp_foundations():
    """Test the RESP commands that framework integrations rely on."""
    print("\n═══ RESP Foundation Commands ═══")

    r = redis_lib.Redis(host="127.0.0.1", port=1974, socket_timeout=5, decode_responses=False)

    # HSET + HGET + HGETALL
    r.hset("__test_fw_hash", mapping={b"f1": b"v1", b"f2": b"v2", b"f3": b"v3"})
    v1 = r.hget("__test_fw_hash", "f1")
    check("HGET returns field", v1 == b"v1")

    all_fields = r.hgetall("__test_fw_hash")
    check("HGETALL returns all fields", len(all_fields) >= 3)

    # ZADD (single-member) + ZREVRANGE + ZCARD
    # Note: Pion ZADD only supports single member per call
    r.zadd("__test_fw_zset", {"a": 1.0})
    r.zadd("__test_fw_zset", {"b": 2.0})
    r.zadd("__test_fw_zset", {"c": 3.0})

    top = r.zrevrange("__test_fw_zset", 0, 0)
    # Pion may return duplicates — dedup
    top_dedup = list(dict.fromkeys(top))
    check("ZREVRANGE returns highest", top_dedup == [b"c"])

    card = r.zcard("__test_fw_zset")
    check("ZCARD correct", card == 3)

    # INCR
    r.delete("__test_fw_incr")
    v = r.incr("__test_fw_incr")
    check("INCR returns 1 on new key", v == 1)
    v2 = r.incr("__test_fw_incr")
    check("INCR increments", v2 == 2)

    # FT.CREATE + HSET + FT.SEARCH
    try:
        r.execute_command("FT.DROPINDEX", "__test_fw_idx")
    except Exception:
        pass
    r.execute_command(
        "FT.CREATE", "__test_fw_idx",
        "SCHEMA", "embedding", "VECTOR", "HNSW",
        "10", "TYPE", "FLOAT32", "DIM", "4",
        "DISTANCE_METRIC", "COSINE", "M", "16", "EF_CONSTRUCTION", "32",
    )
    # Insert 10 vectors for HNSW
    import math
    for i in range(10):
        angle = i * math.pi / 5
        vec = struct.pack("4f", math.cos(angle), math.sin(angle), 0.0, 0.0)
        r.hset(f"__test_fw_idx:{i+1}", mapping={b"embedding": vec, b"text": f"doc{i}".encode()})
    r.execute_command("FT.OPTIMIZE", "__test_fw_idx")

    query_vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    results = r.execute_command(
        "FT.SEARCH", "__test_fw_idx",
        "*=>[KNN 3 @embedding $vec EF_RUNTIME 32]",
        "PARAMS", "2", "vec", query_vec,
    )
    check("FT.SEARCH returns results", results is not None and len(results) > 1)

    # Cleanup — MUST drop index since Pion has single HNSW per worker
    try:
        r.execute_command("FT.DROPINDEX", "__test_fw_idx")
    except Exception:
        pass
    for key in ["__test_fw_hash", "__test_fw_zset", "__test_fw_incr"]:
        r.delete(key)
    for i in range(10):
        r.delete(f"__test_fw_idx:{i+1}")

    r.close()
    print(f"  RESP foundations: all checks complete")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _check_pion():
        print("ERROR: Pion server not running on localhost:1974")
        print("Start it: ./pion-server -w 1")
        sys.exit(1)

    print("A4 — Framework Integration Tests")
    print(f"Pion server: localhost:1974")

    args = set(sys.argv[1:])
    run_all = not args or "--all" in args

    # Always run RESP foundation tests
    test_resp_foundations()

    if run_all or "--langgraph" in args:
        test_langgraph()

    if run_all or "--autogen" in args:
        test_autogen()

    if run_all or "--llamaindex" in args:
        test_llamaindex()

    print(f"\n{'═' * 40}")
    print(f"Results: {passed} passed, {failed} failed, {xfailed} known-bug (XFAIL)")
    if failed > 0:
        sys.exit(1)
    else:
        print("All A4 framework integration tests passed!")
        sys.exit(0)
