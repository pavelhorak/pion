#!/usr/bin/env python3
"""Drop-in, phase A: a vector index answers without FT.OPTIMIZE.

A stock Redis client creates an index, HSETs documents and searches; it never
sends FT.OPTIMIZE. Pion used to answer every such search with `[0]`. Now
(src/network/vector_auto.mojo):

  * before any build, FT.SEARCH answers EXACTLY from the ingest buffer;
  * once ingest is quiet, the graph builds on a background thread and is
    published, and searches use it;
  * vectors written after that build are scanned exactly and merged in, and a
    delta of 1/10 of the index triggers a rebuild;
  * an explicit FT.OPTIMIZE keeps its meaning (build now, free the buffer;
    later HSETs are stored, not indexed, and counted).

Every answer is checked against numpy brute force over the same vectors, so an
"exact" path that is not exact fails here. Multi-worker cases reach each
worker through its affinity port (port + 2 + worker), never the accept race.

Usage:
    python3 tests/test_vector_auto_build.py [--binary ./pion-server] [--port 7471]
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, ok, detail: object = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    return ok


class Client:
    def __init__(self, port, timeout=60):
        import socket
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.f = self.sock.makefile("rb")

    @staticmethod
    def encode(*args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            elif isinstance(a, int):
                a = str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        return out

    def cmd(self, *args):
        self.sock.sendall(self.encode(*args))
        return self.read()

    def read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self.read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def pipe(self, cmds):
        self.sock.sendall(b"".join(self.encode(*c) for c in cmds))
        return [self.read() for _ in cmds]

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def serve(binary, port, workers, cwd, log_name, dim=32):
    # --dim: a saved index loads only at the server's own dim
    cmd = [binary, "--profile", "vector", "-w", str(workers), "-p", str(port), "--dim", str(dim),
           "--no-auto-detect", "--no-auto-embed", "--no-wal"]
    if workers > 1:
        cmd.append("--independent-workers")
    log = open(os.path.join(cwd, log_name), "w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def wait_ready(port, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = Client(port, timeout=2)
            r = c.cmd("PING")
            c.close()
            if r == b"PONG":
                return True
        except (OSError, ConnectionError):
            pass
        time.sleep(0.3)
    return False


def info(c):
    out = {}
    for line in c.cmd("INFO").decode(errors="replace").split("\r\n"):
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k] = v
    return out


def wait_info(c, key, want, timeout=120):
    deadline = time.time() + timeout
    v = None
    while time.time() < deadline:
        v = info(c).get(key)
        if v == want:
            return True
        time.sleep(0.1)
    print(f"    (waited {timeout}s for {key}={want}, last {v})")
    return False


def schema(dim, metric):
    return ["ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA", "n", "NUMERIC",
            "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(dim),
            "DISTANCE_METRIC", metric]


def ingest(c, vecs, start=0, prefix="doc:"):
    cmds = [("HSET", f"{prefix}{start + i}", "n", str(start + i), "vec", v.tobytes())
            for i, v in enumerate(vecs)]
    out = []
    for b in range(0, len(cmds), 1000):
        out += c.pipe(cmds[b:b + 1000])
    return out


def knn(c, q, k=10, extra=()):
    return c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
                 q.astype(np.float32).tobytes(), *extra)


def parse(r):
    """[(key, score)] from a KNN reply (score = __vec_score)."""
    if not isinstance(r, list) or not r:
        return None
    out = []
    i = 1
    while i + 1 < len(r):
        key, fields = r[i], r[i + 1]
        sc = None
        if isinstance(fields, list):
            for j in range(0, len(fields) - 1, 2):
                if fields[j] == b"__vec_score":
                    sc = float(fields[j + 1])
        out.append((key.decode(), sc))
        i += 2
    return out


def brute(vecs, keys, q, k, metric, mask=None):
    """Exact top-k as (key, score), in Pion's units: 1 - cos, or squared L2."""
    if metric == "COSINE":
        vn = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-30)
        qn = q / max(np.linalg.norm(q), 1e-30)
        d = 1.0 - vn @ qn
    else:
        d = ((vecs - q) ** 2).sum(axis=1)
    if mask is not None:
        d = np.where(mask, d, np.inf)
    idx = np.argsort(d, kind="stable")[:k]
    return [(keys[i], float(d[i])) for i in idx if np.isfinite(d[i])]


def same_answer(got, want, tol=2e-3):
    """Same keys (ties may swap), same scores within tol."""
    if got is None or len(got) != len(want):
        return False
    if sorted(k for k, _ in got) != sorted(k for k, _ in want):
        # a tie at the cut can swap the last key: accept when scores agree
        gs = [s for _, s in got]
        ws = [s for _, s in want]
        return all(abs(a - b) <= tol for a, b in zip(gs, ws))
    return all(abs(a - b) <= tol * max(1.0, abs(b)) for (_, a), (_, b) in zip(got, want))


def recall(got, want):
    if not got:
        return 0.0
    g = {k for k, _ in got}
    return sum(1 for k, _ in want if k in g) / max(1, len(want))


# ── single worker ────────────────────────────────────────────────────────────
def phase_single(binary, port, workdir, metric):
    print(f"\n[A/{metric}] -w 1: exact before the build, graph after, delta merged")
    d = os.path.join(workdir, "single_" + metric)
    os.makedirs(d)
    rng = np.random.default_rng(471 if metric == "COSINE" else 472)
    dim, n = 32, 6000
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    keys = [f"doc:{i}" for i in range(n)]
    proc = serve(binary, port, 1, d, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return
        c = Client(port)
        check("FT.CREATE", c.cmd("FT.CREATE", "idx", *schema(dim, metric)) == b"OK")
        r = knn(c, vecs[0])
        check("before any vector: well-formed empty reply", r == [0], repr(r)[:80])
        check("INFO vector_index: empty", info(c).get("vector_index") == "empty", info(c).get("vector_index"))

        ok = all(x in (2, 3) for x in ingest(c, vecs))
        check(f"ingest {n} vectors", ok)
        # Within the quiet window: no graph yet, every answer is exact.
        t0 = time.time()
        st = info(c).get("vector_index")
        check("right after ingest: vector_index exact", st == "exact", st)
        exact_ok = 0
        probes = list(range(0, n, n // 25))
        for i in probes:
            q = vecs[i] + 0.05 * rng.standard_normal(dim).astype(np.float32)
            if same_answer(parse(knn(c, q)), brute(vecs, keys, q, 10, metric)):
                exact_ok += 1
        took = time.time() - t0
        check(f"before the build, FT.SEARCH equals brute force ({exact_ok}/{len(probes)})",
              exact_ok == len(probes))
        if took > 0.2:
            print(f"    (note: the probes took {took:.2f}s; a build may have started meanwhile)")
        r = c.cmd("FT.INFO", "idx")
        kv = {r[i]: r[i + 1] for i in range(0, len(r) - 1, 2)} if isinstance(r, list) else {}
        check("FT.INFO num_docs counts searchable vectors before a build",
              int(kv.get(b"num_docs", b"-1")) == n, repr(r)[:120])

        # FILTER and deletes on the exact path
        q = vecs[10]
        mask = np.arange(n) < 1000
        got = parse(knn(c, q, 10, ("FILTER", "n", "0", "999")))
        check("exact path: FILTER honoured, equals brute force over the filtered set",
              same_answer(got, brute(vecs, keys, q, 10, metric, mask)), got[:3] if got else got)
        check("DEL doc:10", c.cmd("DEL", "doc:10") == 1)
        got = parse(knn(c, vecs[10], 5))
        check("exact path: a deleted document is gone",
              got is not None and "doc:10" not in [k for k, _ in got], got)
        live = np.ones(n, bool)
        live[10] = False

        # Pipeline integrity on the exact path
        c.sock.sendall(Client.encode("FT.SEARCH", "idx", "*=>[KNN 3 @vec $q]", "PARAMS", "2", "q",
                                     vecs[1].tobytes()) + Client.encode("PING"))
        a, b = c.read(), c.read()
        check("exact FT.SEARCH + PING in one write: two replies", isinstance(a, list) and b == b"PONG",
              f"{a!r:.60} {b!r}")

        # The background build
        check("auto-build finishes: vector_index graph", wait_info(c, "vector_index", "graph", 300))
        inf = info(c)
        check("vector_auto_builds 1, vector_graph_rows n",
              inf.get("vector_auto_builds") == "1" and inf.get("vector_graph_rows") == str(n), inf.get("vector_graph_rows"))
        log = open(os.path.join(d, "server.log")).read()
        check("server log names the background build", "[vector] auto-build: indexing" in log
              and "vectors indexed in" in log)
        rs = []
        for i in probes:
            q = vecs[i] + 0.05 * rng.standard_normal(dim).astype(np.float32)
            rs.append(recall(parse(knn(c, q, 10)), brute(vecs, keys, q, 10, metric, live)))
        check(f"after the build: recall@10 {np.mean(rs):.3f} >= 0.9", np.mean(rs) >= 0.9)
        got = parse(knn(c, vecs[10], 5))
        check("after the build: the deleted document stays gone",
              got is not None and "doc:10" not in [k for k, _ in got], got)
        check("pion.hnsw.0 saved by the build thread", os.path.exists(os.path.join(d, "pion.hnsw.0")))

        # The delta: vectors written after the build are found at once
        new = rng.standard_normal((50, dim)).astype(np.float32)
        ingest(c, new, start=n)
        hits = sum(1 for j in range(50) if (parse(knn(c, new[j], 1)) or [("", 0)])[0][0] == f"doc:{n + j}")
        check(f"delta: each new vector is its own nearest at once ({hits}/50)", hits == 50)
        inf = info(c)
        check("INFO: graph+delta, vector_exact_rows 50",
              inf.get("vector_index") == "graph+delta" and inf.get("vector_exact_rows") == "50",
              f"{inf.get('vector_index')} {inf.get('vector_exact_rows')}")
        allv = np.vstack([vecs, new])
        allk = keys + [f"doc:{n + j}" for j in range(50)]
        live2 = np.concatenate([live, np.ones(50, bool)])
        rs = []
        for i in probes[:10]:
            q = allv[i]
            rs.append(recall(parse(knn(c, q, 10)), brute(allv, allk, q, 10, metric, live2)))
        check(f"graph + delta: recall@10 {np.mean(rs):.3f} >= 0.9", np.mean(rs) >= 0.9)

        # A delta of >= 1/10 rebuilds
        more = rng.standard_normal((n // 10, dim)).astype(np.float32)
        ingest(c, more, start=n + 50)
        check("delta >= 1/10 of the index: rebuilt (vector_auto_builds 2)",
              wait_info(c, "vector_auto_builds", "2", 300))
        check("... and vector_index is graph again", wait_info(c, "vector_index", "graph", 60))
        hit = (parse(knn(c, more[7], 1)) or [("", 0)])[0][0]
        check("rebuilt graph holds the delta", hit == f"doc:{n + 50 + 7}", hit)
        # the file the build thread saved (vector_index is graph only after the
        # thread is joined, i.e. after its save)
        warm = os.path.join(d, "warm")
        os.makedirs(warm)
        shutil.copy(os.path.join(d, "pion.hnsw.0"), warm)

        # An identical rewrite does not list a document twice
        c.cmd("HSET", "doc:0", "vec", vecs[0].tobytes())
        got = parse(knn(c, vecs[0], 5))
        keys0 = [k for k, _ in got] if got else []
        check("identical re-HSET: doc:0 once", keys0.count("doc:0") == 1, keys0)

        # Explicit FT.OPTIMIZE keeps its meaning
        check("FT.OPTIMIZE idx", c.cmd("FT.OPTIMIZE", "idx") == b"OK")
        log = open(os.path.join(d, "server.log")).read()
        check("FT.OPTIMIZE over a graph that holds every vector does not build again",
              "nodes built" not in log)
        inf = info(c)
        check("after FT.OPTIMIZE: vector_index graph", inf.get("vector_index") == "graph", inf.get("vector_index"))
        late = rng.standard_normal((3, dim)).astype(np.float32)
        ingest(c, late, start=900000)
        inf = info(c)
        check("HSET after FT.OPTIMIZE: counted in vector_unindexed_writes",
              inf.get("vector_unindexed_writes") == "3", inf.get("vector_unindexed_writes"))
        got = parse(knn(c, late[0], 1))
        check("... and not searchable (the documented contract)",
              got is not None and got[0][0] != "doc:900000", got)
        c.close()
    finally:
        stop(proc)

    # warm restart from the file the automatic rebuild's thread wrote
    proc = serve(binary, port, 1, os.path.join(d, "warm"), "server2.log")
    try:
        if not check("restart ready", wait_ready(port)):
            return
        c = Client(port)
        hit = (parse(knn(c, vecs[123], 1)) or [("", 0)])[0][0]
        check("warm restart serves the automatically built index", hit == "doc:123", hit)
        hit = (parse(knn(c, more[7], 1)) or [("", 0)])[0][0]
        check("... including the rebuilt delta", hit == f"doc:{n + 50 + 7}", hit)
        c.close()
    finally:
        stop(proc)


# ── the delta merge when graph scores are off ────────────────────────────────
def phase_delta_merge(binary, port, workdir):
    """The graph scores its results from INT8 codes, and the quantization noise
    adds to every distance: on OpenAI-50K embeddings (COSINE) the reported
    1 - cos is about 0.094 above the exact one. The delta is exact, so merged
    by score as they were, delta rows outranked the true neighbours: in-stream
    recall@100 fell to 0.07-0.10 (VectorDBBench S50, Mac). The merge now
    rescores the graph's rows from the FP32 ingest buffer first.

    Synthetic COSINE data does not show the inflation; L2 data with one large
    constant component per 32-d quantization group does (each group's range
    is wide, so its steps are coarse): about +24 on squared L2 here. Constant
    components do not move L2 distances, so brute force is still the truth.
    Checked: with a delta, every returned score is the exact one, and the
    merge does not lose recall against the graph alone."""
    print("\n[C] -w 1, 1536-d L2, coarse quantization: graph + delta merge on exact scores")
    d = os.path.join(workdir, "delta_merge")
    os.makedirs(d)
    rng = np.random.default_rng(1536)
    dim, n, nd = 1536, 4000, 300           # nd * 10 < n: no rebuild
    centers = rng.standard_normal((40, dim)).astype(np.float32)

    def draw(m):
        v = (centers[rng.integers(0, 40, m)] + 0.35 * rng.standard_normal((m, dim))).astype(np.float32)
        v[:, ::32] += 20.0
        return v
    vecs = draw(n)
    qs = draw(30)
    proc = serve(binary, port, 1, d, "server.log", dim=dim)
    try:
        if not check("server ready", wait_ready(port)):
            return
        c = Client(port)
        check("FT.CREATE (L2, DIM 1536)", c.cmd("FT.CREATE", "idx", *schema(dim, "L2")) == b"OK")
        ingest(c, vecs)
        check("auto-build finishes", wait_info(c, "vector_index", "graph", 300))
        keys = [f"doc:{i}" for i in range(n + nd)]
        base = [recall(parse(knn(c, q, 10)), brute(vecs, keys[:n], q, 10, "L2")) for q in qs]
        delta = draw(nd)
        ingest(c, delta, start=n)
        st = info(c)
        check("graph + 300 delta rows", st.get("vector_index") == "graph+delta"
              and st.get("vector_exact_rows") == str(nd), f"{st.get('vector_index')} {st.get('vector_exact_rows')}")
        allv = np.vstack([vecs, delta])
        rs, worst = [], 0.0
        for q in qs:
            got = parse(knn(c, q, 10))
            rs.append(recall(got, brute(allv, keys, q, 10, "L2")))
            exact = ((allv - q) ** 2).sum(axis=1)
            for k2, sc in got or []:
                worst = max(worst, abs(sc - exact[int(k2.split(":")[1])]) / max(1.0, exact[int(k2.split(":")[1])]))
        check(f"graph + delta: every score is the exact one (worst relative error {worst:.5f})", worst <= 2e-3)
        check(f"graph + delta: recall@10 {np.mean(rs):.3f} not below the graph's own {np.mean(base):.3f} - 0.05",
              np.mean(rs) >= np.mean(base) - 0.05)
        c.close()
    finally:
        stop(proc)


# ── four workers ─────────────────────────────────────────────────────────────
WORKERS = 4


def phase_multi(binary, port, workdir):
    print(f"\n[B] -w {WORKERS}: exact and built answers on every worker")
    d = os.path.join(workdir, "multi")
    os.makedirs(d)
    rng = np.random.default_rng(4710)
    dim, n = 32, 5000
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    keys = [f"doc:{i}" for i in range(n)]
    proc = serve(binary, port, WORKERS, d, "server.log")
    try:
        if not check("server ready", wait_ready(port)):
            return
        cs = [Client(port + 2 + w) for w in range(WORKERS)]
        ws = [int(info(c).get("worker_id", -1)) for c in cs]
        if not check("affinity ports reach workers 0..3", ws == list(range(WORKERS)), ws):
            return
        check("FT.CREATE on worker 0", cs[0].cmd("FT.CREATE", "idx", *schema(dim, "COSINE")) == b"OK")
        ingest(cs[2], vecs)
        ok = 0
        for w in range(WORKERS):
            q = vecs[w * 97]
            if same_answer(parse(knn(cs[w], q)), brute(vecs, keys, q, 10, "COSINE")):
                ok += 1
        check(f"before the build: exact on every worker ({ok}/{WORKERS})", ok == WORKERS)
        check("auto-build finishes", wait_info(cs[3], "vector_index", "graph", 300))
        rs = []
        for w in range(WORKERS):
            for i in range(w, n, n // 10):
                q = vecs[i]
                rs.append(recall(parse(knn(cs[w], q)), brute(vecs, keys, q, 10, "COSINE")))
        check(f"after the build: recall@10 {np.mean(rs):.3f} >= 0.9 on every worker", np.mean(rs) >= 0.9)
        new = rng.standard_normal((20, dim)).astype(np.float32)
        ingest(cs[1], new, start=n)
        hits = sum(1 for j in range(20)
                   if (parse(knn(cs[j % WORKERS], new[j], 1)) or [("", 0)])[0][0] == f"doc:{n + j}")
        check(f"delta written on worker 1 found on every worker ({hits}/20)", hits == 20)

        # Drop on a worker that does not own the graph, then a new index from
        # scratch: every worker must serve the NEW vectors (stale borrows and a
        # stale owner graph are the hazard). SKIPINITIALSCAN keeps the old
        # documents out: FT.CREATE indexes the hashes already in the keyspace
        # (test_ft_create_backfill.py), and this phase is about the new ones.
        check("FT.DROPINDEX on worker 3", cs[3].cmd("FT.DROPINDEX", "idx") == b"OK")
        sch = schema(dim, "COSINE")
        at = sch.index("SCHEMA")
        check("FT.CREATE again on worker 2 (SKIPINITIALSCAN)",
              cs[2].cmd("FT.CREATE", "idx", *sch[:at], "SKIPINITIALSCAN", *sch[at:]) == b"OK")
        vecs2 = rng.standard_normal((3000, dim)).astype(np.float32)
        keys2 = [f"doc:{i}" for i in range(3000)]
        ingest(cs[0], vecs2)
        ok = 0
        for w in range(WORKERS):
            q = vecs2[w * 31]
            if same_answer(parse(knn(cs[w], q)), brute(vecs2, keys2, q, 10, "COSINE")):
                ok += 1
        check(f"new index, before its build: exact on every worker ({ok}/{WORKERS})", ok == WORKERS)
        check("second auto-build finishes", wait_info(cs[1], "vector_auto_builds", "2", 300)
              and wait_info(cs[1], "vector_index", "graph", 60))
        rs = []
        for w in range(WORKERS):
            for i in range(w, 3000, 300):
                q = vecs2[i]
                rs.append(recall(parse(knn(cs[w], q)), brute(vecs2, keys2, q, 10, "COSINE")))
        check(f"new index built: recall@10 {np.mean(rs):.3f} >= 0.9 on every worker", np.mean(rs) >= 0.9)
        for c in cs:
            c.close()
    finally:
        stop(proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(REPO, "pion-server"))
    ap.add_argument("--port", type=int, default=7471)
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    a = ap.parse_args()
    workdir = tempfile.mkdtemp(prefix="pion_auto_build_")
    try:
        phase_single(a.binary, a.port, workdir, "COSINE")
        phase_single(a.binary, a.port, workdir, "L2")
        phase_delta_merge(a.binary, a.port, workdir)
        phase_multi(a.binary, a.port, workdir)
    finally:
        if a.keep:
            print("scratch:", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAIL:", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
