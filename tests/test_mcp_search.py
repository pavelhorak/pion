#!/usr/bin/env python3
"""pion-mcp search tools (and pion-serve's RAG lookup) against a live server.

Every search tool in mcp/pion_mcp/server.py was broken against a real Pion,
three ways at once, and nothing noticed:

1. vector_search, vector_search_raw, search_with_filter and
   semantic_cache_get sent `FT.SEARCH <index> <vector-bytes> K <k>`, which
   Pion does not parse as a vector query — it answered an empty array.
2. Every parser did float() on the per-result fields ARRAY ([b"score", ...]),
   which raised, so every score read as 0.0 and the cache's threshold check
   returned None on every lookup.
3. The cache computed cosine as 1 - distance/2 from the server's score, which
   is a distance in the quantized search space (a vector against itself
   scores ~770 here), not a normalized L2.

pion-serve's _rag_retrieve had the same unparsed query form
(`FT.SEARCH <idx> <blob> KNN k` -> []) and then looked for the document text
in the reply fields, which only ever carry id/score — so --rag-index injected
nothing. Section [4] runs when pion-serve's dependencies (flask) import.

Also exercised: the mock embedding provider, which embedded every text to the
same constant vector (sin(h + i*1.618) with a 256-bit h).

Needs `mcp[cli]<2` and `redis<5` importable (pip install -e mcp/). Starts its
own ./pion-server on --port.

Usage: python3 tests/test_mcp_search.py [--port 6398]   (exit 0 = pass, 2 = skipped)
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def fresh_server(port):
    # A private working directory: the server writes its WAL/snapshot into its
    # cwd, and this used to be the repo root (deleting pion.* files there).
    workdir = tempfile.mkdtemp(prefix="pion_mcp_search_")
    binary = os.environ.get("PION_BIN", os.path.join(REPO, "pion-server"))
    proc = subprocess.Popen([binary, "-p", str(port), "-w", "1",
                             "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                            cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.workdir = workdir
    import redis
    for _ in range(100):
        try:
            redis.Redis(port=port).ping()
            return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("server did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6398)
    args = ap.parse_args()
    os.environ["PION_EMBED_PROVIDER"] = "mock"
    os.environ["PION_PORT"] = str(args.port)
    sys.path.insert(0, os.path.join(REPO, "mcp"))
    try:
        from pion_mcp import server as s
    except ImportError as e:
        print(f"SKIP: pion_mcp not importable ({e}); pip install -e mcp/")
        return 2

    print("[1] helpers")
    check("score read by name from a fields array", s._score_of([b"id", b"7", b"score", b"12.5"]) == 12.5)
    check("score-only fields array", s._score_of([b"score", b"3"]) == 3.0)
    check("no score -> None", s._score_of([b"id", b"7"]) is None)
    a, b = s.embed_text("alpha"), s.embed_text("beta")
    check("mock embeds different texts differently", a != b)
    check("cosine(x, x) == 1", abs(s._cosine(a, a) - 1.0) < 1e-5)
    check("mock vectors are near-orthogonal", abs(s._cosine(a, b)) < 0.2, f"{s._cosine(a, b):.3f}")

    # One index per server (gh #145): the cache and the docs index each get a
    # fresh server.
    print("[2] semantic cache")
    proc = fresh_server(args.port)
    try:
        s._pion = None
        r = s._conn()
        q = "how do I reset my password?"
        s.semantic_cache_set(q, "Use the reset link.")
        r.execute_command("FT.OPTIMIZE", s._CACHE_INDEX)
        check("same query hits", s.semantic_cache_get(q) == "Use the reset link.")
        check("unrelated query misses", s.semantic_cache_get("capital of France?") is None)
    finally:
        proc.kill(); proc.wait(); shutil.rmtree(proc.workdir, ignore_errors=True)

    print("[3] vector search tools")
    proc = fresh_server(args.port)
    try:
        s._pion = None
        for i in range(20):
            s.add_document("docs", f"doc:{i}", f"document number {i} about topic {i % 5}")
        s.optimize_index("docs")
        text = "document number 3 about topic 3"
        res = s.vector_search("docs", text, k=3)
        check("vector_search returns results", len(res) == 3, str(res)[:120])
        check("exact document ranks first", bool(res) and res[0]["id"] == "doc:3", str([x["id"] for x in res]))
        check("scores are real, not the 0.0 fallback", bool(res) and res[0]["score"] > 0 and res[0]["score"] < res[-1]["score"])
        raw = s.vector_search_raw("docs", s.embed_text(text).hex(), k=3)
        check("vector_search_raw agrees", [x["id"] for x in raw] == [x["id"] for x in res])
        flt = s.search_with_filter("docs", text, k=3)
        check("search_with_filter returns results", bool(flt) and flt[0]["id"] == "doc:3")
    finally:
        proc.kill(); proc.wait(); shutil.rmtree(proc.workdir, ignore_errors=True)

    print("[4] pion-serve RAG retrieval")
    try:
        sys.path.insert(0, os.path.join(REPO, "pion-serve"))
        import serve
        import numpy as np
    except ImportError as e:
        print(f"  SKIP  pion-serve not importable ({e})")
    else:
        proc = fresh_server(args.port)
        try:
            import random
            import redis
            r = redis.Redis(port=args.port, decode_responses=False)
            r.execute_command("FT.CREATE", "rag", "SCHEMA", "emb", "VECTOR", "HNSW", "6",
                              "TYPE", "FLOAT32", "DIM", "1536", "DISTANCE_METRIC", "COSINE")

            def vec(i):
                rnd = random.Random(i)
                return np.array([rnd.gauss(0, 1) for _ in range(1536)], dtype=np.float32)
            for i in range(10):
                r.hset(f"doc:{i}", mapping={"content": f"passage {i}", "emb": vec(i).tobytes()})
            r.execute_command("FT.OPTIMIZE", "rag")
            serve._pion, serve._embed_fn = r, (lambda q: vec(int(q.split()[-1])))
            serve._config["rag_index"] = "rag"
            got = serve._rag_retrieve("query 4", k=3)
            check("RAG returns passages, nearest first", got[:1] == ["passage 4"] and len(got) == 3, str(got))
        finally:
            proc.kill(); proc.wait(); shutil.rmtree(proc.workdir, ignore_errors=True)

    print(f"\n{'FAIL: ' + ', '.join(FAIL) if FAIL else 'ALL PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
