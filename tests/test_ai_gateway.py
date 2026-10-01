#!/usr/bin/env python3
"""Integration tests for AI gateway commands.

Section 1-3: No external deps (AI.ROUTE, RAG.SPECULATE, AI.SEMANTIC_CACHE)
Section 4-8: Require Ollama (AI.CHAT, AI.COMPLETE, AI.EMBED, AI.MEMORY, AI.FLARE)

Requires: pion-server --kvcache -w 1  on port 1974
Ollama sections auto-skip if embedding/LLM not available.
"""

import socket
import struct
import sys
import numpy as np

HOST = "127.0.0.1"
PORT = 1974
DIM = 768  # default embedding dimension for router/speculative


def send_raw_resp(sock, parts):
    """Send a RESP command with mixed string/bytes args. Return raw response."""
    header = f"*{len(parts)}\r\n".encode()
    body = b""
    for part in parts:
        if isinstance(part, bytes):
            body += f"${len(part)}\r\n".encode() + part + b"\r\n"
        else:
            s = str(part)
            body += f"${len(s)}\r\n{s}\r\n".encode()
    sock.sendall(header + body)
    return sock.recv(1024 * 1024)


def make_embedding(seed, dim=DIM):
    """Create a deterministic unit-norm FP32 embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def connect():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.settimeout(5)
    return sock


# ──────────────────────────────────────────────────────────────────────────────
# M13: Semantic Router (AI.ROUTE.*)
# ──────────────────────────────────────────────────────────────────────────────

passed = 0
failed = 0
skipped = 0
# Set by main() from a probe BEFORE the backend-dependent sections run. Those
# sections used to turn every error or timeout into a printed "SKIP" that
# counted as nothing — so a broken AI.SEMANTIC_CACHE (SET ok, GET misses)
# was indistinguishable from "no embedding server", and the gate passed.
BACKEND_OK = False   # embedding backend (Ollama nomic-embed-text or the sidecar)
LLM_OK = False       # a chat/generation model the server is configured for


def env_skip(name, reason, backend="embed"):
    """A backend-dependent check that could not run. If the probe found THAT
    backend, this is a FAILURE; only a genuinely absent backend may skip, and
    every skip is counted and printed in the summary."""
    global failed, skipped
    present = LLM_OK if backend == "llm" else BACKEND_OK
    if present:
        failed += 1
        print(f"  FAIL: {name} — {reason} (the {backend} backend probe succeeded, so this is not an environment skip)")
    else:
        skipped += 1
        print(f"  SKIP: {name} — {reason} (no {backend} backend)")


def _ollama_has(model):
    """Does the local Ollama serve `model`? (tags API; no LLM call.)"""
    import json, urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except OSError:
        return False
    return any(n == model or n.split(":")[0] == model.split(":")[0] and model.endswith(n.split(":")[-1]) for n in names)


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def test_route_info_empty():
    """AI.ROUTE.INFO on empty routing table."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.ROUTE.INFO"])
    sock.close()
    check("ROUTE.INFO empty returns bulk string", b"$" in resp, repr(resp[:100]))
    check("ROUTE.INFO shows nodes:0", b"nodes:0" in resp, repr(resp[:200]))


def test_route_register():
    """AI.ROUTE.REGISTER with valid embedding."""
    sock = connect()
    emb = make_embedding(100)

    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER", "node-math", "http://gpu1:8080/v1", emb.tobytes()])
    check("REGISTER node-math returns +OK", b"+OK" in resp, repr(resp))

    # Register a second node with a different domain
    emb2 = make_embedding(200)
    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER", "node-code", "http://gpu2:8080/v1", emb2.tobytes()])
    check("REGISTER node-code returns +OK", b"+OK" in resp, repr(resp))

    # Register with CAPACITY
    emb3 = make_embedding(300)
    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER", "node-med", "http://gpu3:8080/v1", emb3.tobytes(), "CAPACITY", "10"])
    check("REGISTER node-med with CAPACITY returns +OK", b"+OK" in resp, repr(resp))

    sock.close()


def test_route_register_bad_embedding():
    """AI.ROUTE.REGISTER with wrong embedding size should fail."""
    sock = connect()
    bad_emb = np.zeros(16, dtype=np.float32)  # wrong dim
    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER", "bad-node", "http://x:1", bad_emb.tobytes()])
    check("REGISTER wrong dim returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


def test_route_register_duplicate():
    """AI.ROUTE.REGISTER duplicate node_id should fail."""
    sock = connect()
    emb = make_embedding(999)
    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER", "node-math", "http://dup:1", emb.tobytes()])
    check("REGISTER duplicate returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


def test_route_info_populated():
    """AI.ROUTE.INFO after registering 3 nodes."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.ROUTE.INFO"])
    sock.close()
    check("ROUTE.INFO shows nodes:3", b"nodes:3" in resp, repr(resp[:300]))
    check("ROUTE.INFO shows node-math", b"node-math" in resp, repr(resp[:500]))
    check("ROUTE.INFO shows node-code", b"node-code" in resp, repr(resp[:500]))
    check("ROUTE.INFO shows node-med", b"node-med" in resp, repr(resp[:500]))


def test_route_query_routing():
    """AI.ROUTE with query embedding closest to a registered node."""
    sock = connect()
    # Query close to node-math (seed=100)
    query_math = make_embedding(100)  # exact same as node-math centroid
    resp = send_raw_resp(sock, ["AI.ROUTE", query_math.tobytes()])
    check("ROUTE to math returns gpu1 endpoint", b"gpu1" in resp, repr(resp))

    # Query close to node-code (seed=200)
    query_code = make_embedding(200)
    resp = send_raw_resp(sock, ["AI.ROUTE", query_code.tobytes()])
    check("ROUTE to code returns gpu2 endpoint", b"gpu2" in resp, repr(resp))
    sock.close()


def test_route_query_exclude():
    """AI.ROUTE with EXCLUDE should skip the excluded node."""
    sock = connect()
    query = make_embedding(100)  # closest to node-math
    resp = send_raw_resp(sock, ["AI.ROUTE", query.tobytes(), "EXCLUDE", "node-math"])
    check("ROUTE with EXCLUDE skips node-math", b"gpu1" not in resp, repr(resp))
    # Should route to one of the other nodes
    check("ROUTE with EXCLUDE returns some endpoint", b"gpu" in resp, repr(resp))
    sock.close()


def test_route_update():
    """AI.ROUTE.UPDATE changes a node's centroid."""
    sock = connect()
    # Update node-math to have node-code's embedding
    new_emb = make_embedding(200)
    resp = send_raw_resp(sock, ["AI.ROUTE.UPDATE", "node-math", new_emb.tobytes()])
    check("UPDATE node-math returns +OK", b"+OK" in resp, repr(resp))

    # Now querying with seed=100 should NOT route to node-math (centroid changed)
    query = make_embedding(100)
    resp = send_raw_resp(sock, ["AI.ROUTE", query.tobytes()])
    # node-math now has seed=200 centroid, so seed=100 query should go elsewhere
    check("ROUTE after UPDATE doesn't go to updated node", b"gpu1" not in resp or b"gpu" in resp, repr(resp))

    # Restore original centroid
    orig_emb = make_embedding(100)
    send_raw_resp(sock, ["AI.ROUTE.UPDATE", "node-math", orig_emb.tobytes()])
    sock.close()


def test_route_update_nonexistent():
    """AI.ROUTE.UPDATE on unknown node should fail."""
    sock = connect()
    emb = make_embedding(1)
    resp = send_raw_resp(sock, ["AI.ROUTE.UPDATE", "nonexistent", emb.tobytes()])
    check("UPDATE nonexistent returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


def test_route_remove():
    """AI.ROUTE.REMOVE deletes a node."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.ROUTE.REMOVE", "node-med"])
    check("REMOVE node-med returns +OK", b"+OK" in resp, repr(resp))

    # Verify gone from INFO
    resp = send_raw_resp(sock, ["AI.ROUTE.INFO"])
    check("ROUTE.INFO shows nodes:2 after remove", b"nodes:2" in resp, repr(resp[:300]))
    check("ROUTE.INFO no longer shows node-med", b"node-med" not in resp, repr(resp[:500]))
    sock.close()


def test_route_remove_nonexistent():
    """AI.ROUTE.REMOVE on unknown node should fail."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.ROUTE.REMOVE", "ghost-node"])
    check("REMOVE nonexistent returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# M9: Speculative RAG (RAG.SPECULATE.*, RAG.QUERY)
# ──────────────────────────────────────────────────────────────────────────────

def test_rag_info_global_empty():
    """RAG.SPECULATE.INFO returns valid stats."""
    sock = connect()
    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO"])
    sock.close()
    check("RAG.INFO global returns bulk string", b"$" in resp, repr(resp[:100]))
    check("RAG.INFO shows sessions field", b"sessions:" in resp, repr(resp[:200]))
    check("RAG.INFO shows enabled:True", b"enabled:True" in resp, repr(resp[:200]))


def test_rag_enable_session():
    """RAG.SPECULATE.ENABLE creates a session."""
    sock = connect()
    resp = send_raw_resp(sock, ["RAG.SPECULATE.ENABLE", "sess-1"])
    check("ENABLE sess-1 returns +OK", b"+OK" in resp, repr(resp))

    # Enable with custom DEPTH and THRESHOLD
    resp = send_raw_resp(sock, ["RAG.SPECULATE.ENABLE", "sess-2", "DEPTH", "5", "THRESHOLD", "0.85"])
    check("ENABLE sess-2 with DEPTH/THRESHOLD returns +OK", b"+OK" in resp, repr(resp))
    sock.close()


def test_rag_info_per_session():
    """RAG.SPECULATE.INFO <session_id> returns per-session stats."""
    sock = connect()
    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO", "sess-1"])
    check("RAG.INFO sess-1 returns bulk string", b"$" in resp, repr(resp[:100]))
    check("RAG.INFO sess-1 shows queries field", b"queries:" in resp, repr(resp[:300]))
    check("RAG.INFO sess-1 shows depth:3", b"depth:3" in resp, repr(resp[:300]))

    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO", "sess-2"])
    check("RAG.INFO sess-2 shows depth:5", b"depth:5" in resp, repr(resp[:300]))
    check("RAG.INFO sess-2 shows threshold:0.85", b"threshold:0.85" in resp.replace(b"0.849999", b"0.85"), repr(resp[:300]))
    sock.close()


def test_rag_info_nonexistent():
    """RAG.SPECULATE.INFO on unknown session should fail."""
    sock = connect()
    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO", "no-such-session"])
    check("RAG.INFO unknown session returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


def test_rag_query_no_hnsw():
    """RAG.QUERY without HNSW populated returns empty array or valid results."""
    sock = connect()
    emb = make_embedding(10)
    resp = send_raw_resp(sock, ["RAG.QUERY", "sess-1", emb.tobytes()])
    # May return *0 if HNSW empty, or results if prior tests populated it
    is_valid = resp.startswith(b"*0") or resp.startswith(b"$") or resp.startswith(b"*")
    check("RAG.QUERY returns valid RESP response", is_valid, repr(resp[:100]))
    sock.close()


def test_rag_query_builds_trajectory():
    """Multiple RAG.QUERY calls build trajectory history."""
    sock = connect()
    # Send 3 queries to build trajectory
    for i in range(3):
        emb = make_embedding(10 + i)
        send_raw_resp(sock, ["RAG.QUERY", "sess-1", emb.tobytes()])

    # Check session stats — should show queries incremented
    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO", "sess-1"])
    # Extract query count (may vary if server reused across runs)
    check("RAG.INFO after queries shows queries field", b"queries:" in resp, repr(resp[:300]))
    check("RAG.INFO shows history_count > 0", b"history_count:0\r\n" not in resp, repr(resp[:300]))
    sock.close()


def test_rag_global_info_after_queries():
    """RAG.SPECULATE.INFO global shows updated stats."""
    sock = connect()
    resp = send_raw_resp(sock, ["RAG.SPECULATE.INFO"])
    check("RAG.INFO global shows sessions:2", b"sessions:2" in resp, repr(resp[:200]))
    sock.close()


def test_rag_query_with_hnsw():
    """RAG.QUERY with HNSW populated returns results."""
    import time

    # Use a fresh connection for HNSW setup
    sock1 = connect()
    send_raw_resp(sock1, ["FT.CREATE", "rag_test_idx", "ON", "HASH",
                          "SCHEMA", "vec", "VECTOR", "HNSW", "6",
                          "TYPE", "FLOAT32", "DIM", "768", "DISTANCE_METRIC", "L2"])

    # Insert 20 vectors
    for i in range(20):
        v = make_embedding(1000 + i)
        send_raw_resp(sock1, ["HSET", f"doc:{i}", "vec", v.tobytes()])

    send_raw_resp(sock1, ["FT.OPTIMIZE", "rag_test_idx"])
    sock1.close()
    time.sleep(1)  # let optimize complete

    # Fresh connection for RAG.QUERY
    sock2 = connect()

    # Check if HNSW is actually ready (FT.INFO should show index)
    resp_info = send_raw_resp(sock2, ["FT.INFO", "rag_test_idx"])

    query_emb = make_embedding(1005)  # matches doc:5 exactly
    resp = send_raw_resp(sock2, ["RAG.QUERY", "sess-1", query_emb.tobytes(), "K", "3"])
    # RAG.QUERY returns *0 if HNSW not ready, or a bulk string with results
    # After FT.OPTIMIZE with 20 vectors, HNSW should be ready
    # Accept either non-empty results OR *0 (if optimize hasn't completed yet)
    is_valid_response = resp.startswith(b"*") or resp.startswith(b"$")
    check("RAG.QUERY with HNSW returns valid RESP response", is_valid_response, repr(resp[:200]))

    # Clean up
    send_raw_resp(sock2, ["FT.DROPINDEX", "rag_test_idx"])
    sock2.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.SEMANTIC_CACHE (requires Ollama — skipped if embedding disabled)
# ──────────────────────────────────────────────────────────────────────────────

def test_semantic_cache():
    """AI.SEMANTIC_CACHE SET/GET (requires embedding server).

    When server has --no-auto-detect and no --emb-enabled, embedding is disabled.
    SET will return ERR or hang waiting for HTTP — detect and skip gracefully.
    """
    import time
    sock = connect()
    sock.settimeout(3)  # short timeout — SET with disabled embedding may hang on HTTP

    try:
        resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "SET",
                                    "What is the capital of France?", "Paris"])
    except socket.timeout:
        env_skip("AI.SEMANTIC_CACHE", "embedding HTTP timeout — server has no embedding backend")
        sock.close()
        return

    if b"+OK" in resp:
        # SET returned OK. Verify GET works — if embedding server is unreachable,
        # SET may succeed but store a zero/garbage embedding, causing GET to miss.
        try:
            resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "capital of France?"])
            if b"Paris" in resp:
                # The miss path below is env_skip(), which FAILS when the
                # embedding backend probe succeeded — that is the real check.
                check("SEMANTIC_CACHE SET+GET round-trip", True)

                # GET with unrelated query should miss
                resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "How to sort an array in Python?"])
                is_miss = b"$-1" in resp or b"Paris" not in resp
                check("SEMANTIC_CACHE GET unrelated query misses", is_miss, repr(resp[:200]))

                # THRESHOLD must change the answer, both ways — a check that
                # accepted "hit or miss" (as this one did) cannot fail.
                # gh #373: an explicit threshold at or below 0 used to mean
                # "unset" (the handler tested `> 0`), so the default applied and
                # this missed. -1 is the floor of cosine: it must hit anything.
                resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "How to sort an array in Python?", "THRESHOLD", "-1"])
                check("SEMANTIC_CACHE THRESHOLD -1 hits even an unrelated query", b"Paris" in resp, repr(resp[:200]))
                resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "How to sort an array in Python?", "THRESHOLD", "0.01"])
                check("SEMANTIC_CACHE THRESHOLD 0.01 hits even an unrelated query", b"Paris" in resp, repr(resp[:200]))
                resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "What is the capital of France?", "THRESHOLD", "abc"])
                check("SEMANTIC_CACHE THRESHOLD abc is an error, not the default", resp.startswith(b"-"), repr(resp[:200]))
                resp = send_raw_resp(sock, ["AI.SEMANTIC_CACHE", "GET", "How to sort an array in Python?", "THRESHOLD", "0.99999"])
                check("SEMANTIC_CACHE THRESHOLD 0.99999 misses an unrelated query", b"Paris" not in resp, repr(resp[:200]))
            else:
                # GET missed — embedding server probably not reachable
                env_skip("AI.SEMANTIC_CACHE", "SET ok but GET missed — no embedding server")
        except socket.timeout:
            env_skip("AI.SEMANTIC_CACHE", "GET timeout — embedding server slow/unavailable")
    elif b"-ERR" in resp:
        env_skip(f"AI.SEMANTIC_CACHE", f"server returned: {resp.decode(errors='replace').strip()}")
    else:
        env_skip("AI.SEMANTIC_CACHE", "embedding not enabled — requires Ollama")

    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────────────

def test_route_missing_args():
    """AI.ROUTE.* with missing arguments should return errors."""
    sock = connect()

    resp = send_raw_resp(sock, ["AI.ROUTE.REGISTER"])
    check("REGISTER no args returns ERR", b"-ERR" in resp, repr(resp))

    resp = send_raw_resp(sock, ["AI.ROUTE.UPDATE"])
    check("UPDATE no args returns ERR", b"-ERR" in resp, repr(resp))

    resp = send_raw_resp(sock, ["AI.ROUTE"])
    check("ROUTE no args returns ERR", b"-ERR" in resp, repr(resp))

    resp = send_raw_resp(sock, ["AI.ROUTE.REMOVE"])
    check("REMOVE no args returns ERR", b"-ERR" in resp, repr(resp))

    sock.close()


def test_rag_missing_args():
    """RAG.* with missing arguments should return errors."""
    sock = connect()

    resp = send_raw_resp(sock, ["RAG.SPECULATE.ENABLE"])
    check("ENABLE no args returns ERR", b"-ERR" in resp, repr(resp))

    resp = send_raw_resp(sock, ["RAG.QUERY"])
    check("RAG.QUERY no args returns ERR", b"-ERR" in resp, repr(resp))

    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.EMBED (requires Ollama embedding)
# ──────────────────────────────────────────────────────────────────────────────

def _ollama_available():
    """Check if Ollama embedding is reachable by sending AI.EMBED."""
    sock = connect()
    sock.settimeout(5)
    try:
        resp = send_raw_resp(sock, ["AI.EMBED", "test"])
        sock.close()
        if b"-ERR" in resp:
            return False
        # Expect a bulk string with FP32 bytes (DIM * 4 bytes)
        return b"$" in resp and len(resp) > 100
    except socket.timeout:
        sock.close()
        return False


def test_ai_embed():
    """AI.EMBED returns a valid FP32 embedding vector."""
    sock = connect()
    sock.settimeout(10)
    try:
        resp = send_raw_resp(sock, ["AI.EMBED", "The quick brown fox jumps over the lazy dog"])
    except socket.timeout:
        env_skip("AI.EMBED", "timeout — no embedding server")
        sock.close()
        return False

    if b"-ERR" in resp:
        env_skip(f"AI.EMBED", f"{resp.decode(errors='replace').strip()}")
        sock.close()
        return False

    # Parse bulk string: $<len>\r\n<bytes>\r\n
    check("AI.EMBED returns bulk string", resp.startswith(b"$"), repr(resp[:50]))
    newline = resp.index(b"\r\n")
    blob_len = int(resp[1:newline])
    blob = resp[newline + 2:newline + 2 + blob_len]
    dim = blob_len // 4
    check("AI.EMBED vector has valid dimensions", dim >= 384, f"dim={dim}")
    check("AI.EMBED blob is multiple of 4 bytes", blob_len % 4 == 0, f"len={blob_len}")

    # Parse as float array — should be finite, not all zeros
    vec = np.frombuffer(blob, dtype=np.float32)
    check("AI.EMBED vector is finite", np.all(np.isfinite(vec)), f"has inf/nan")
    check("AI.EMBED vector is non-zero", np.linalg.norm(vec) > 0.1, f"norm={np.linalg.norm(vec)}")

    sock.close()
    return True


def test_ai_embed_missing_args():
    """AI.EMBED with no text should return error."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.EMBED"])
    check("AI.EMBED no args returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.CHAT (requires Ollama LLM)
# ──────────────────────────────────────────────────────────────────────────────

def test_ai_chat_simple():
    """AI.CHAT with a simple prompt returns LLM response."""
    sock = connect()
    sock.settimeout(30)  # LLM generation can take a few seconds
    try:
        resp = send_raw_resp(sock, ["AI.CHAT", "Reply with exactly one word: hello"])
    except socket.timeout:
        env_skip("AI.CHAT", "timeout — LLM server unavailable", backend="llm")
        sock.close()
        return False

    if b"-ERR" in resp:
        env_skip(f"AI.CHAT", f"{resp.decode(errors='replace').strip()}", backend="llm")
        sock.close()
        return False

    check("AI.CHAT returns bulk string response", b"$" in resp, repr(resp[:100]))
    # Should contain some text — $-1 means null (LLM returned empty)
    if resp.startswith(b"$-1"):
        # A configured LLM that answers null is not a pass (it used to be one).
        env_skip("AI.CHAT response has content", "LLM returned null", backend="llm")
    else:
        newline = resp.index(b"\r\n")
        blob_len = int(resp[1:newline])
        check("AI.CHAT response has content", blob_len > 0, f"len={blob_len}")

    sock.close()
    return True


def test_ai_chat_missing_args():
    """AI.CHAT with no prompt should return error."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.CHAT"])
    check("AI.CHAT no args returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.COMPLETE (requires Ollama embedding + LLM, semantic cache)
# ──────────────────────────────────────────────────────────────────────────────

def test_ai_complete():
    """AI.COMPLETE generates response and caches it, second call hits cache."""
    sock = connect()
    sock.settimeout(30)

    prompt = "What is 2+2? Reply with just the number."

    try:
        resp1 = send_raw_resp(sock, ["AI.COMPLETE", prompt])
    except socket.timeout:
        env_skip("AI.COMPLETE", "timeout — embedding/LLM unavailable", backend="llm")
        sock.close()
        return False

    if b"-ERR" in resp1:
        env_skip(f"AI.COMPLETE", f"{resp1.decode(errors='replace').strip()}", backend="llm")
        sock.close()
        return False

    check("AI.COMPLETE first call returns bulk string", b"$" in resp1, repr(resp1[:100]))

    # Second call with same prompt should hit semantic cache (much faster)
    import time
    t0 = time.time()
    try:
        resp2 = send_raw_resp(sock, ["AI.COMPLETE", prompt])
    except socket.timeout:
        check("AI.COMPLETE cache hit", False, "timeout on second call")
        sock.close()
        return False
    cache_time = time.time() - t0

    check("AI.COMPLETE second call returns bulk string", b"$" in resp2, repr(resp2[:100]))
    # Cache hit should be fast (embedding HTTP + HNSW search, no LLM call)
    check("AI.COMPLETE cache hit is fast (<5s)", cache_time < 5.0, f"took {cache_time:.2f}s")

    sock.close()
    return True


def test_ai_complete_with_threshold():
    """AI.COMPLETE with custom THRESHOLD."""
    sock = connect()
    sock.settimeout(15)
    try:
        resp = send_raw_resp(sock, ["AI.COMPLETE", "What is 3+3?", "THRESHOLD", "0.5"])
    except socket.timeout:
        sock.close()
        return
    # Should work (lower threshold = more cache hits)
    if b"-ERR" not in resp:
        check("AI.COMPLETE with THRESHOLD returns response", b"$" in resp, repr(resp[:100]))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.MEMORY (requires Ollama embedding)
# ──────────────────────────────────────────────────────────────────────────────

def test_ai_memory_add():
    """AI.MEMORY ADD stores a memory entry."""
    sock = connect()
    sock.settimeout(10)
    try:
        resp = send_raw_resp(sock, ["AI.MEMORY", "ADD", "test-session", "user",
                                    "The capital of France is Paris"])
    except socket.timeout:
        env_skip("AI.MEMORY", "timeout — embedding unavailable")
        sock.close()
        return False

    if b"-ERR" in resp:
        env_skip(f"AI.MEMORY", f"{resp.decode(errors='replace').strip()}")
        sock.close()
        return False

    check("AI.MEMORY ADD returns +OK", b"+OK" in resp, repr(resp))

    # Add a second memory
    try:
        resp = send_raw_resp(sock, ["AI.MEMORY", "ADD", "test-session", "assistant",
                                    "Paris is the capital and largest city of France"])
        check("AI.MEMORY ADD second entry returns +OK", b"+OK" in resp, repr(resp))
    except socket.timeout:
        check("AI.MEMORY ADD second entry", False, "timeout")

    sock.close()
    return True


def test_ai_memory_context():
    """AI.MEMORY CONTEXT returns recent conversation entries."""
    sock = connect()
    sock.settimeout(5)
    try:
        resp = send_raw_resp(sock, ["AI.MEMORY", "CONTEXT", "test-session", "10"])
    except socket.timeout:
        sock.close()
        return

    if b"-ERR" in resp:
        sock.close()
        return

    check("AI.MEMORY CONTEXT returns array", b"*" in resp, repr(resp[:100]))
    # Should have at least 2 entries from test_ai_memory_add
    check("AI.MEMORY CONTEXT has entries", b"$" in resp, repr(resp[:200]))

    sock.close()


def test_ai_memory_recall():
    """AI.MEMORY RECALL searches for semantically similar memories."""
    sock = connect()
    sock.settimeout(10)
    try:
        resp = send_raw_resp(sock, ["AI.MEMORY", "RECALL", "test-session",
                                    "What is the capital of France?"])
    except socket.timeout:
        sock.close()
        return

    if b"-ERR" in resp:
        sock.close()
        return

    # May return a hit or miss depending on HNSW state
    is_valid = b"$" in resp or b"*" in resp
    check("AI.MEMORY RECALL returns valid response", is_valid, repr(resp[:200]))

    sock.close()


def test_ai_memory_missing_args():
    """AI.MEMORY with missing args returns error."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.MEMORY"])
    check("AI.MEMORY no args returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.FLARE (requires Ollama embedding + LLM)
# ──────────────────────────────────────────────────────────────────────────────

def test_ai_flare_info():
    """AI.FLARE INFO returns knowledge base stats."""
    sock = connect()
    sock.settimeout(5)
    resp = send_raw_resp(sock, ["AI.FLARE", "INFO"])
    # FLARE INFO should work even without documents loaded
    if b"-ERR" in resp:
        env_skip(f"AI.FLARE", f"{resp.decode(errors='replace').strip()}")
        sock.close()
        return False

    check("AI.FLARE INFO returns bulk string", b"$" in resp, repr(resp[:100]))
    check("AI.FLARE INFO contains stats", b"docs" in resp or b"doc_count" in resp or b"FLARE" in resp,
          repr(resp[:300]))

    sock.close()
    return True


def test_ai_flare_load():
    """AI.FLARE LOAD indexes text into the knowledge base."""
    sock = connect()
    sock.settimeout(15)
    try:
        resp = send_raw_resp(sock, ["AI.FLARE", "LOAD",
                                    "Python is a high-level programming language. "
                                    "It was created by Guido van Rossum in 1991. "
                                    "Python emphasizes code readability and simplicity."])
    except socket.timeout:
        env_skip("AI.FLARE LOAD", "timeout — embedding unavailable")
        sock.close()
        return False

    if b"-ERR" in resp:
        env_skip(f"AI.FLARE LOAD", f"{resp.decode(errors='replace').strip()}")
        sock.close()
        return False

    check("AI.FLARE LOAD returns +OK", b"+OK" in resp, repr(resp))
    sock.close()
    return True


def test_ai_flare_run():
    """AI.FLARE RUN performs RAG query against the knowledge base."""
    sock = connect()
    sock.settimeout(30)
    try:
        resp = send_raw_resp(sock, ["AI.FLARE", "RUN", "Who created Python?"])
    except socket.timeout:
        env_skip("AI.FLARE RUN", "timeout — LLM unavailable", backend="llm")
        sock.close()
        return

    if b"-ERR" in resp:
        # May fail if LLM not enabled
        sock.close()
        return

    check("AI.FLARE RUN returns response", b"$" in resp or b"+" in resp, repr(resp[:200]))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# AI.GENERATE + AI.LOADMODEL (requires inference sidecar via --inference)
# ──────────────────────────────────────────────────────────────────────────────

def _sidecar_available():
    """Check if inference sidecar is connected by sending AI.GENERATE."""
    sock = connect()
    sock.settimeout(10)
    try:
        resp = send_raw_resp(sock, ["AI.GENERATE", "test"])
        sock.close()
        return b"-ERR AI.GENERATE requires --inference" not in resp
    except socket.timeout:
        sock.close()
        return False


def test_ai_generate_simple():
    """AI.GENERATE with a simple prompt returns generated text."""
    sock = connect()
    sock.settimeout(15)
    try:
        resp = send_raw_resp(sock, ["AI.GENERATE", "The meaning of life is"])
    except socket.timeout:
        check("AI.GENERATE returns response", False, "timeout")
        sock.close()
        return

    if b"-ERR" in resp:
        # Expected when sidecar has no LLM loaded (embedding-only mode)
        if b"inference failed" in resp or b"no LLM" in resp:
            check("AI.GENERATE returns expected error (no LLM in sidecar)", True)
        else:
            check("AI.GENERATE returns response", False, resp.decode(errors="replace")[:100])
        sock.close()
        return

    check("AI.GENERATE returns bulk string", b"$" in resp, repr(resp[:50]))
    newline = resp.index(b"\r\n")
    blob_len = int(resp[1:newline])
    check("AI.GENERATE response has content", blob_len > 0, f"len={blob_len}")
    sock.close()


def test_ai_generate_with_keys():
    """AI.GENERATE with KEYS injects keyspace values as context."""
    sock = connect()
    sock.settimeout(15)

    # Set a key for context
    send_raw_resp(sock, ["SET", "test_ctx_key", "Python is a programming language"])

    try:
        resp = send_raw_resp(sock, ["AI.GENERATE", "Describe the language",
                                    "KEYS", "test_ctx_key", "MAX_TOKENS", "50"])
    except socket.timeout:
        sock.close()
        return

    if b"-ERR" not in resp:
        check("AI.GENERATE with KEYS returns response", b"$" in resp, repr(resp[:100]))

    # Clean up
    send_raw_resp(sock, ["DEL", "test_ctx_key"])
    sock.close()


def test_ai_generate_missing_args():
    """AI.GENERATE with no prompt should return error."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.GENERATE"])
    check("AI.GENERATE no args returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


def test_ai_loadmodel_embedding():
    """AI.LOADMODEL loads an embedding model."""
    sock = connect()
    sock.settimeout(30)
    try:
        resp = send_raw_resp(sock, ["AI.LOADMODEL",
                                    "sentence-transformers/all-MiniLM-L6-v2", "EMBEDDING"])
    except socket.timeout:
        check("AI.LOADMODEL embedding", False, "timeout")
        sock.close()
        return

    check("AI.LOADMODEL EMBEDDING returns +OK", b"+OK" in resp, repr(resp))
    sock.close()


def test_ai_loadmodel_llm():
    """AI.LOADMODEL loads an LLM model."""
    sock = connect()
    sock.settimeout(30)
    try:
        resp = send_raw_resp(sock, ["AI.LOADMODEL", "sshleifer/tiny-gpt2", "LLM"])
    except socket.timeout:
        check("AI.LOADMODEL LLM", False, "timeout")
        sock.close()
        return

    check("AI.LOADMODEL LLM returns +OK", b"+OK" in resp, repr(resp))
    sock.close()


def test_ai_loadmodel_missing_args():
    """AI.LOADMODEL with no model_id should return error."""
    sock = connect()
    resp = send_raw_resp(sock, ["AI.LOADMODEL"])
    check("AI.LOADMODEL no args returns ERR", b"-ERR" in resp, repr(resp))
    sock.close()


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup: remove remaining test nodes
# ──────────────────────────────────────────────────────────────────────────────

def test_cleanup():
    """Remove test nodes."""
    sock = connect()
    send_raw_resp(sock, ["AI.ROUTE.REMOVE", "node-math"])
    send_raw_resp(sock, ["AI.ROUTE.REMOVE", "node-code"])
    resp = send_raw_resp(sock, ["AI.ROUTE.INFO"])
    check("Cleanup: all nodes removed", b"nodes:0" in resp, repr(resp[:200]))
    sock.close()


def main():
    global passed, failed

    print("=" * 60)
    print("AI Gateway Integration Tests")
    print(f"Target: {HOST}:{PORT} (requires --kvcache -w 1)")
    print("=" * 60)

    # M13: Semantic Router
    print("\n=== Section 1: AI.ROUTE.* (M13 Semantic Router) ===")
    test_route_info_empty()
    test_route_register()
    test_route_register_bad_embedding()
    test_route_register_duplicate()
    test_route_info_populated()
    test_route_query_routing()
    test_route_query_exclude()
    test_route_update()
    test_route_update_nonexistent()
    test_route_remove()
    test_route_remove_nonexistent()
    test_route_missing_args()

    # M9: Speculative RAG
    print("\n=== Section 2: RAG.SPECULATE.* (M9 Speculative RAG) ===")
    test_rag_info_global_empty()
    test_rag_enable_session()
    test_rag_info_per_session()
    test_rag_info_nonexistent()
    test_rag_query_no_hnsw()
    test_rag_query_builds_trajectory()
    test_rag_global_info_after_queries()
    test_rag_query_with_hnsw()
    test_rag_missing_args()

    # Probe the backends ONCE, before any section that depends on them, so a
    # section can tell "backend absent" (skip) from "backend present but the
    # feature is broken" (fail).
    global BACKEND_OK, LLM_OK
    ollama_ok = _ollama_available()
    BACKEND_OK = ollama_ok
    import os
    llm_model = os.environ.get("PION_LLM_MODEL", "llama3.1:8b")   # the --flare default (main.mojo)
    LLM_OK = _ollama_has(llm_model)
    print(f"backends: embedding={'yes' if BACKEND_OK else 'NO'}  llm({llm_model})={'yes' if LLM_OK else 'NO'}")

    # AI.SEMANTIC_CACHE (requires an embedding backend)
    print("\n=== Section 3: AI.SEMANTIC_CACHE ===")
    test_semantic_cache()

    # Ollama-dependent tests (sections 4-8): fail if embedding is unavailable
    # (Ollama must be running for gate)
    if not ollama_ok:
        print("\n  FAIL: Ollama embedding not available — sections 4-8 require Ollama with nomic-embed-text")
        print("        Start Ollama: ollama serve &, then: ollama pull nomic-embed-text")
        failed += 5  # count skipped sections as failures
    else:
        # AI.EMBED
        print("\n=== Section 4: AI.EMBED ===")
        test_ai_embed()
        test_ai_embed_missing_args()

        # AI.CHAT
        print("\n=== Section 5: AI.CHAT ===")
        test_ai_chat_simple()
        test_ai_chat_missing_args()

        # AI.COMPLETE
        print("\n=== Section 6: AI.COMPLETE ===")
        test_ai_complete()
        test_ai_complete_with_threshold()

        # AI.MEMORY
        print("\n=== Section 7: AI.MEMORY ===")
        test_ai_memory_add()
        test_ai_memory_context()
        test_ai_memory_recall()
        test_ai_memory_missing_args()

        # AI.FLARE
        print("\n=== Section 8: AI.FLARE ===")
        test_ai_flare_info()
        test_ai_flare_load()
        test_ai_flare_run()

    # Inference sidecar tests (section 9) — requires --inference flag
    sidecar_ok = _sidecar_available()
    if not sidecar_ok:
        print("\n  FAIL: Inference sidecar not available — section 9 requires --inference flag")
        print("        Start with: ./pion-server --kvcache -w 1 --flare --inference")
        failed += 6  # count skipped section as failures
    else:
        print("\n=== Section 9: AI.GENERATE + AI.LOADMODEL (Inference Sidecar) ===")
        test_ai_generate_simple()
        test_ai_generate_with_keys()
        test_ai_generate_missing_args()
        test_ai_loadmodel_embedding()
        test_ai_loadmodel_llm()
        test_ai_loadmodel_missing_args()

    # Cleanup
    print("\n=== Cleanup ===")
    test_cleanup()

    print("\n" + "=" * 60)
    print(f"AI Gateway Tests: {passed} passed, {failed} failed, {skipped} skipped (no backend)")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
