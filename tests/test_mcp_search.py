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
    sys.path.insert(0, os.path.join(REPO, "pion_context"))
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        # The gate's Python has no `mcp` package, and this test used to SKIP
        # on every gate run for that alone. The tools are plain functions; a
        # stub FastMCP that registers nothing lets them be tested here. The
        # MCP transport itself is not exercised without the real package.
        import types

        class _StubFastMCP:
            def __init__(self, *a, **k):
                pass

            def tool(self, *a, **k):
                return lambda f: f

            def run(self, *a, **k):
                raise SystemExit("stub FastMCP cannot serve")

        for name in ("mcp", "mcp.server"):
            sys.modules.setdefault(name, types.ModuleType(name))
        stub = types.ModuleType("mcp.server.fastmcp")
        stub.FastMCP = _StubFastMCP
        sys.modules["mcp.server.fastmcp"] = stub
        print("  note: no `mcp` package; the tools run under a stub FastMCP (transport not exercised)")
    try:
        from pion_mcp import server as s
    except ImportError as e:
        print(f"SKIP: pion_mcp not importable ({e})")
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
            check("RAG returns passages, nearest first (vector field learned from the server)",
                  got[:1] == ["passage 4"] and len(got) == 3, str(got))
            check("the learned field is the index's own", serve._config.get("rag_field") == "emb",
                  str(serve._config.get("rag_field")))
            errors_before = serve._stats.get("rag_errors", 0)
            serve._config.update(rag_field="nope", rag_field_pinned=True)
            got = serve._rag_retrieve("query 4", k=3)
            check("a failing RAG lookup is counted in rag_errors, not silent",
                  got == [] and serve._stats.get("rag_errors", 0) == errors_before + 1,
                  f"{got} rag_errors={serve._stats.get('rag_errors')}")
        finally:
            proc.kill(); proc.wait(); shutil.rmtree(proc.workdir, ignore_errors=True)

    print("[5] codebase tools through pion-mcp, and agent memory replacing their index")
    proc = fresh_server(args.port)
    repo = tempfile.mkdtemp(prefix="pion_mcp_repo_")
    try:
        from pion_context import indexer as ix
        files = {"pkg/models/user.py": "class User:\n    def set_password(self, raw):\n        self.h = hash(raw)\n\n\n\n",
                 "pkg/notify.py": "def send_email(to, body):\n    return (to, body)\n\n\n\n\n"}
        for rel, text in files.items():
            os.makedirs(os.path.join(repo, os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(repo, rel), "w") as f:
                f.write(text)
        # pion-context lists files the way git does; the harness's temp dir sits
        # in an ignored directory of this checkout, so the repo needs its own git.
        subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True)
        s._pion = None
        st = s.codebase_index(repo)
        check("codebase_index indexes every file, models/ included", "Indexed 2 files" in str(st), str(st)[:160])
        path = os.path.join(repo, "pkg/models/user.py")
        chunk = ix.chunk_file(path, files["pkg/models/user.py"])[0]
        query = f"{chunk.file_path}:{chunk.start_line} ({chunk.kind} {chunk.name})\n{chunk.content}"   # mock: exact text
        got = s.codebase_search(query, k=2)
        check("codebase_search finds the file", bool(got) and got[0].get("file_path") == path, str(got)[:200])
        for i in range(50):        # pion-mcp builds __agent_memory__ at the 50th memory
            s.agent_remember(f"fact number {i} about the deployment", session_id="t")
        got = s.codebase_search(query, k=2)
        err = got[0].get("error", "") if got else ""
        check("after 50 memories codebase_search reports the replaced index, not []",
              "__agent_memory__" in err, str(got)[:200])
    finally:
        proc.kill(); proc.wait(); shutil.rmtree(proc.workdir, ignore_errors=True)
        shutil.rmtree(repo, ignore_errors=True)

    print("[6] PION_EMBED_PROVIDER=ollama works in pion-mcp, with pion-context's vectors")
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _FakeOllama(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["content-length"])))
            texts = body.get("input") or [body.get("prompt", "")]
            vecs = [[float((hash(t) >> (i % 32)) & 7) - 3.5 for i in range(768)] for t in texts]
            out = {"embeddings": vecs} if self.path == "/api/embed" else {"embedding": vecs[0]}
            data = _json.dumps(out).encode()
            self.send_response(200); self.send_header("content-length", str(len(data))); self.end_headers()
            self.wfile.write(data)

    fake = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    os.environ.update(PION_EMBED_PROVIDER="ollama", PION_OLLAMA_URL=f"http://127.0.0.1:{fake.server_port}")
    try:
        from pion_mcp import embeddings as me
        from pion_context import embeddings as ce
        try:
            v = me.embed_text("hello world")
            raised = ""
        except Exception as e:
            v, raised = b"", f"{type(e).__name__}: {e}"
        check("pion-mcp accepts ollama", not raised and len(v) == 1536 * 4, raised or f"{len(v)} bytes")
        check("and embeds exactly as pion-context does", v == ce.embed_text("hello world"))
    finally:
        os.environ["PION_EMBED_PROVIDER"] = "mock"
        fake.shutdown()

    print(f"\n{'FAIL: ' + ', '.join(FAIL) if FAIL else 'ALL PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
