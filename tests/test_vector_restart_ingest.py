#!/usr/bin/env python3
"""A restart keeps a vector index taking writes, or says that it does not.

WHY
After a restart, an index loaded from its file served the vectors it was built
with, and no vector written afterwards was ever indexed: FT.SEARCH missed it
with no error, and INFO vector_unindexed_writes read 0. Only FT.DROPINDEX then
FT.CREATE brought those vectors in. A restart is routine, so this was the most
likely way to get a wrong answer from 1.0.

Now a restart leaves each index as its build left it:

  * an automatic build (no FT.OPTIMIZE) keeps indexing. Its file records that
    it was one, every worker refills the FP32 rows of its own hashes before
    any serves (the file holds the graph, not those rows), and the hashes the
    graph does not hold become the delta again;
  * an index FT.OPTIMIZE built (or whose file predates 1.0.2) keeps ingest
    closed, as FT.OPTIMIZE says. A vector written to it is counted in
    vector_unindexed_writes, and the log says so at startup.

Answers are checked against numpy brute force. The multi-worker case reaches
each worker through its affinity port (port + 2 + worker), never the accept
race.

    python3 tests/test_vector_restart_ingest.py [--binary ./pion-server] [--port 7491]
"""
import argparse
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16
# recall@10 against numpy; the graph alone reads about 0.93 here. What this
# test catches reads far lower: an unindexed delta, rows never refilled.
FLOOR = 0.90
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)
    return ok


def serve(binary, port, cwd, workers=1, wal=True):
    cmd = [binary, "-p", str(port), "-w", str(workers), "--dim", str(DIM),
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"]
    if workers > 1:
        cmd.append("--independent-workers")
    if not wal:
        cmd.append("--no-wal")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(os.path.join(cwd, "log"), "a"),
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with Conn(port, timeout=2) as c:
                if c.cmd("PING") == "PONG":
                    return p
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server on {port} did not start")


def stop(p, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(p.pid), sig)
        p.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait(timeout=10)
        except Exception:
            pass


def log_text(d):
    with open(os.path.join(d, "log"), errors="replace") as f:
        return f.read()


def info(c):
    out = {}
    for line in c.cmd("INFO").decode(errors="replace").split("\r\n"):
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k] = v
    return out


def vinfo(c):
    i = info(c)
    return (i.get("vector_index"), int(i.get("vector_graph_rows", 0)), int(i.get("vector_exact_rows", 0)),
            int(i.get("vector_unindexed_writes", 0)))


def wait_state(c, want, timeout=60):
    deadline = time.time() + timeout
    v = None
    while time.time() < deadline:
        v = info(c).get("vector_index")
        if v == want:
            return True
        time.sleep(0.05)
    print(f"    (waited {timeout}s for vector_index={want}, last {v})")
    return False


def create(c, metric="L2"):
    return c.cmd("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", "n", "NUMERIC",
                 "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
                 "DISTANCE_METRIC", metric)


def write(c, docs):
    """docs: {key: vector}"""
    cmds = [("HSET", k, "n", "1", "vec", v.astype(np.float32).tobytes()) for k, v in docs.items()]
    for b in range(0, len(cmds), 500):
        c.pipeline(cmds[b:b + 500])


def knn(c, q, k=10):
    """[(key, score)]"""
    r = c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
              q.astype(np.float32).tobytes(), "DIALECT", "2")
    if isinstance(r, RespError) or not isinstance(r, list):
        return r
    out = []
    for i in range(1, len(r) - 1, 2):
        sc = None
        f = r[i + 1]
        for j in range(0, len(f) - 1, 2):
            if f[j] == b"__vec_score":
                sc = float(f[j + 1])
        out.append((r[i].decode(), sc))
    return out


def dist(docs, q, metric):
    keys = list(docs)
    m = np.stack([docs[k] for k in keys]).astype(np.float64)
    q = q.astype(np.float64)
    if metric == "COSINE":
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        d = 1.0 - m @ (q / np.linalg.norm(q))
    else:
        d = ((m - q) ** 2).sum(axis=1)
    return keys, d


def recall(c, docs, queries, metric="L2", k=10):
    hit = 0
    for q in queries:
        got = knn(c, q, k)
        if not isinstance(got, list):
            return -1.0
        keys, d = dist(docs, q, metric)
        want = {keys[j] for j in np.argsort(d)[:k]}
        hit += len({g for g, _ in got} & want)
    return hit / (k * len(queries))


def self_misses(c, docs):
    """The documents whose own vector does not find them first."""
    return [key for key, v in docs.items() if [g for g, _ in (knn(c, v, 1) or [])] != [key]]


def vecs(rng, n, scale=False):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    if scale:
        v *= rng.uniform(0.2, 5.0, (n, 1)).astype(np.float32)
    return v


def flag_word(d):
    with open(os.path.join(d, "pion.hnsw.0"), "rb") as f:
        f.seek(25 * 8)
        return struct.unpack("<Q", f.read(8))[0]


def build_auto(c, docs, metric="L2"):
    create(c, metric)
    write(c, docs)
    return wait_state(c, "graph")


def phase_auto(binary, port, d, rng):
    print("=== an automatic build keeps indexing across a restart (-w 1) ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 300))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs))
            delta = {f"doc:{300 + i}": v for i, v in enumerate(vecs(rng, 20))}
            # a long key: it resolves through `__hk__<slot>`, not the 31-byte map
            delta["doc:with-a-key-longer-than-thirty-one-bytes"] = vecs(rng, 1)[0]
            write(c, delta)
            docs.update(delta)
            # before the restart: a graph document deleted, a delta one deleted
            # (its tombstone names a slot past the graph, which a restart
            # numbers again), a graph document's vector rewritten
            old20 = docs["doc:20"]
            docs["doc:20"] = vecs(rng, 1)[0]
            write(c, {"doc:20": docs["doc:20"]})
            c.cmd("DEL", "doc:10", "doc:305")
            del docs["doc:10"], docs["doc:305"]
            st = vinfo(c)
            # 21 written + doc:20's new vector; doc:305's row is dead, not gone
            check("before: graph 300 + delta 22", st[:3] == ("graph+delta", 300, 22), st)
            check("its file says it was built automatically", flag_word(d) & 256 != 0, hex(flag_word(d)))
    finally:
        stop(p, signal.SIGKILL)

    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            st = vinfo(c)
            # the 21 live documents the graph does not hold, doc:20 among them
            check("after the restart: the graph serves and the delta is back", st[:3] == ("graph+delta", 300, 21), st)
            log = log_text(d)
            check("the log says it keeps indexing", "built automatically): it keeps indexing" in log)
            misses = self_misses(c, docs)
            check(f"every one of the {len(docs)} documents finds itself first", not misses, misses[:5])
            got = knn(c, old20, 1)
            check("doc:20's old vector no longer finds doc:20", not got or got[0][0] != "doc:20", got)
            gone = [k for q in (vecs(rng, 30)) for k, _ in knn(c, q, 20) if k in ("doc:10", "doc:305")]
            check("deleted documents stay out", not gone, gone)
            queries = vecs(rng, 20)
            r = recall(c, docs, queries)
            check(f"recall@10 vs numpy >= {FLOOR}", r >= FLOOR, r)

            new = {f"new:{i}": v for i, v in enumerate(vecs(rng, 5))}
            write(c, new)
            docs.update(new)
            st = vinfo(c)
            check("5 vectors written after the restart are indexed", st == ("graph+delta", 300, 26, 0), st)
            check("and found at once", not self_misses(c, new), self_misses(c, new))

            more = {f"more:{i}": v for i, v in enumerate(vecs(rng, 20))}
            write(c, more)
            docs.update(more)
            check("a delta of a tenth rebuilds the index, as before a restart",
                  wait_state(c, "graph") and info(c).get("vector_auto_builds") == "1", info(c).get("vector_auto_builds"))
            misses = self_misses(c, docs)
            check(f"after it every one of the {len(docs)} documents finds itself", not misses, misses[:5])
            r = recall(c, docs, queries)
            check(f"recall@10 >= {FLOOR}", r >= FLOOR, r)
    finally:
        stop(p)

    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            st = vinfo(c)
            check("a second restart (SIGTERM) loads the rebuilt index, no delta", st[:3] == ("graph", 346, 0), st)
            misses = self_misses(c, docs)
            check("every document finds itself", not misses, misses[:5])
            write(c, {"last:0": vecs(rng, 1)[0]})
            check("and it still takes writes", vinfo(c)[:3] == ("graph+delta", 346, 1), vinfo(c))
    finally:
        stop(p)


def phase_cosine(binary, port, d, rng):
    print("=== COSINE: refilled rows are normalized as ingest normalizes them ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 300, scale=True))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs, "COSINE"))
            delta = {f"doc:{300 + i}": v for i, v in enumerate(vecs(rng, 20, scale=True))}
            write(c, delta)
            docs.update(delta)
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("graph + delta after the restart", vinfo(c)[:3] == ("graph+delta", 300, 20), vinfo(c))
            # With a delta, the graph's results are rescored from the FP32 rows
            # to merge with it: each score is then the exact distance.
            bad = []
            for q in vecs(rng, 20, scale=True):
                keys, dd = dist(docs, q, "COSINE")
                exact = dict(zip(keys, dd))
                for k, s in knn(c, q, 10):
                    if abs(s - exact[k]) > 1e-3:
                        bad.append((k, s, round(float(exact[k]), 4)))
            check("every reported score is the exact 1 - cos", not bad, bad[:3])
            r = recall(c, docs, vecs(rng, 20, scale=True), "COSINE")
            check(f"recall@10 vs numpy >= {FLOOR}", r >= FLOOR, r)
    finally:
        stop(p)


def phase_optimize(binary, port, d, rng):
    print("=== FT.OPTIMIZE ended ingest: a restart keeps it ended, and counts what it misses ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 200))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, docs)
            check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            check("its file is not an automatic build's", flag_word(d) & 256 == 0, hex(flag_word(d)))
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("loaded", vinfo(c) == ("graph", 200, 0, 0), vinfo(c))
            check("the log says ingest is closed",
                  "loaded (200 vectors) with ingest closed" in log_text(d))
            new = {f"new:{i}": v for i, v in enumerate(vecs(rng, 5))}
            write(c, new)
            st = vinfo(c)
            check("5 writes after the restart are counted as unindexed", st == ("graph", 200, 0, 5), st)
            check("FT.DROPINDEX then FT.CREATE indexes every hash",
                  c.cmd("FT.DROPINDEX", "idx") == "OK" and create(c) == "OK"
                  and sum(vinfo(c)[1:3]) == 205, vinfo(c))
            docs.update(new)
            check("the five included", not self_misses(c, new), self_misses(c, new))
    finally:
        stop(p)


def phase_covered(binary, port, d, rng):
    print("=== FT.OPTIMIZE after an automatic build (nothing new) ends ingest after a restart too ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 200))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs))
            check("its file says so", flag_word(d) & 256 != 0, hex(flag_word(d)))
            check("FT.OPTIMIZE keeps the graph and ends ingest", c.cmd("FT.OPTIMIZE", "idx") == "OK")
            check("and clears the flag in the file", flag_word(d) & 256 == 0, hex(flag_word(d)))
    finally:
        stop(p)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, {"new:0": vecs(rng, 1)[0]})
            st = vinfo(c)
            check("after the restart a write is counted, not indexed", st == ("graph", 200, 0, 1), st)
    finally:
        stop(p)


def phase_legacy(binary, port, d, rng):
    print("=== a file from before 1.0.2 (no flag) loads with ingest closed and says so ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 200))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs))
    finally:
        stop(p)
    path = os.path.join(d, "pion.hnsw.0")
    with open(path, "r+b") as f:
        f.seek(25 * 8)
        w = struct.unpack("<Q", f.read(8))[0]
        f.seek(25 * 8)
        f.write(struct.pack("<Q", w & ~256))
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("the log says ingest is closed", "or its file predates 1.0.2" in log_text(d))
            write(c, {"new:0": vecs(rng, 1)[0]})
            check("a write is counted, not indexed", vinfo(c) == ("graph", 200, 0, 1), vinfo(c))
            check("the index still answers", not self_misses(c, dict(list(docs.items())[:20])))
    finally:
        stop(p)


def phase_no_data(binary, port, d, rng):
    print("=== --no-wal: the documents are gone, so the index serves as loaded ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 200))}
    p = serve(binary, port, d, wal=False)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs))
    finally:
        stop(p)
    p = serve(binary, port, d, wal=False)
    try:
        with Conn(port, timeout=60) as c:
            check("the log says so", "none of the index's 200 documents is in the keyspace" in log_text(d))
            check("it serves as loaded", vinfo(c) == ("graph", 200, 0, 0), vinfo(c))
            check("its documents still answer by key", not self_misses(c, dict(list(docs.items())[:20])))
            write(c, {"new:0": vecs(rng, 1)[0]})
            check("a write is counted, not indexed", vinfo(c) == ("graph", 200, 0, 1), vinfo(c))
    finally:
        stop(p)


def phase_multi(binary, port, d, rng):
    print("=== -w 4: every worker refills its own documents before any serves ===")
    docs = {}
    p = serve(binary, port, d, workers=4)
    conns = []
    try:
        conns = [Conn(port + 2 + w, timeout=60) for w in range(4)]
        check("FT.CREATE on worker 0", create(conns[0]) == "OK")
        for w in range(4):
            part = {f"w{w}:{i}": v for i, v in enumerate(vecs(rng, 100))}
            write(conns[w], part)
            docs.update(part)
        check("built by itself, all 400", wait_state(conns[0], "graph") and vinfo(conns[0])[1] == 400, vinfo(conns[0]))
        for w in range(4):
            part = {f"w{w}:d{i}": v for i, v in enumerate(vecs(rng, 8))}
            write(conns[w], part)
            docs.update(part)
        conns[2].cmd("DEL", "w2:5")
        conns[3].cmd("DEL", "w3:d3")
        del docs["w2:5"], docs["w3:d3"]
        check("graph 400 + delta 32", vinfo(conns[1])[:3] == ("graph+delta", 400, 32), vinfo(conns[1]))
    finally:
        for c in conns:
            c.close()
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d, workers=4)
    conns = []
    try:
        conns = [Conn(port + 2 + w, timeout=60) for w in range(4)]
        check("after the restart: graph 400 + delta 31", vinfo(conns[3])[:3] == ("graph+delta", 400, 31), vinfo(conns[3]))
        for w in range(4):
            misses = self_misses(conns[w], docs)
            check(f"worker {w}: all {len(docs)} documents find themselves", not misses, misses[:5])
        new = {"w1:new": vecs(rng, 1)[0]}
        write(conns[1], new)
        check("a write on worker 1 is found from worker 3", not self_misses(conns[3], new))
    finally:
        for c in conns:
            c.close()
        stop(p)


def phase_dropped(binary, port, d, rng):
    print("=== a dropped index stays dropped across a restart ===")
    docs = {f"doc:{i}": v for i, v in enumerate(vecs(rng, 100))}
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            check("built by itself", build_auto(c, docs))
            check("FT.DROPINDEX", c.cmd("FT.DROPINDEX", "idx") == "OK")
            check("its file is gone", not os.path.exists(os.path.join(d, "pion.hnsw.0")))
    finally:
        stop(p)
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            r = c.cmd("FT.INFO", "idx")
            check("after the restart FT.INFO does not know it", isinstance(r, RespError), r)
            st = vinfo(c)
            check("and nothing is indexed", st[:3] == ("none", 0, 0), st)
            check("FT.CREATE then indexes the 100 hashes", create(c) == "OK" and sum(vinfo(c)[1:3]) == 100, vinfo(c))
    finally:
        stop(p)


def phase_long_key_slot(binary, port, d, rng):
    print("=== a long key on a slot an earlier index numbered answers as itself ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, {"short:0": vecs(rng, 1)[0]})
            c.cmd("FT.DROPINDEX", "idx")
            c.cmd("DEL", "short:0")
            create(c)
            long_key = "doc:" + "x" * 40
            v = vecs(rng, 1)[0]
            write(c, {long_key: v})
            got = knn(c, v, 1)
            check("slot 0 answers with the long key, not short:0", got[:1] and got[0][0] == long_key, got)
    finally:
        stop(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7491)
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    rng = np.random.default_rng(11)
    root = tempfile.mkdtemp(prefix="pion_restart_ingest_")
    try:
        for n, phase in enumerate([phase_auto, phase_cosine, phase_optimize, phase_covered, phase_legacy,
                                   phase_no_data, phase_multi, phase_dropped, phase_long_key_slot]):
            d = os.path.join(root, str(n))
            os.makedirs(d)
            phase(binary, a.port, d, rng)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
