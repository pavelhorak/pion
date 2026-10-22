#!/usr/bin/env python3
"""FT.CREATE indexes the hashes already in the keyspace, as RediSearch does.

WHY
FT.CREATE used to index only vectors written after it. An index created over
documents that were already there stayed empty: FT.SEARCH answered `[0]` with
no error, and every document had to be written again. Creating the index after
loading the data is one way there. A restart was the other: an index whose DIM
differs from the server's --dim is not restored from its file, and re-issuing
FT.CREATE gave back an empty index while every hash was still in the keyspace.
SKIPINITIALSCAN keeps the old behaviour, as it does in RediSearch.

Answers are checked against numpy brute force over the same vectors. The
multi-worker case reaches each worker through its affinity port
(port + 2 + worker), never through the accept race.

    python3 tests/test_ft_create_backfill.py [--binary ./pion-server] [--port 7481]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, encode  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16            # not the server's --dim (1536): the restart case depends on it
# An answer may come from the exact scan or, once the automatic build lands, from
# the graph (about 0.93 recall@10 here against numpy). What this test catches
# reads far lower: no backfill finds nothing, one worker's alone finds a quarter.
FLOOR = 0.90
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)


def serve(binary, port, cwd, workers=1, wal=False, dim=None):
    cmd = [binary, "-p", str(port), "-w", str(workers), "--no-auto-detect", "--no-auto-embed",
           "--no-crash-log"]
    if dim:
        cmd += ["--dim", str(dim)]
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


def info(c):
    out = {}
    for line in c.cmd("INFO").decode(errors="replace").split("\r\n"):
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            out[k] = v
    return out


def rows(c):
    """Vectors the index holds: the graph's plus the exactly scanned ones."""
    i = info(c)
    return int(i.get("vector_graph_rows", 0)) + int(i.get("vector_exact_rows", 0))


def create(c, *opts, metric="L2"):
    return c.cmd("FT.CREATE", "idx", "ON", "HASH", *opts, "SCHEMA", "n", "NUMERIC",
                 "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM),
                 "DISTANCE_METRIC", metric)


def write(c, vecs, prefix, start=0):
    cmds = [("HSET", f"{prefix}{start + i}", "n", str(start + i), "vec", v.tobytes())
            for i, v in enumerate(vecs)]
    for b in range(0, len(cmds), 500):
        c.pipeline(cmds[b:b + 500])


def knn(c, q, k=10):
    r = c.cmd("FT.SEARCH", "idx", f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
              q.astype(np.float32).tobytes(), "NOCONTENT", "DIALECT", "2")
    if isinstance(r, RespError):
        return r
    return [x.decode() for x in r[1:]]


def brute(vecs, keys, q, k=10):
    d = ((vecs - q) ** 2).sum(axis=1)
    return [keys[j] for j in np.argsort(d)[:k]]


def recall(c, vecs, keys, queries, k=10):
    hit = tot = 0
    dups = 0
    for q in queries:
        got = knn(c, q, k)
        if isinstance(got, RespError):
            return -1.0, 0
        dups += len(got) - len(set(got))
        hit += len(set(got) & set(brute(vecs, keys, q, k)))
        tot += k
    return hit / tot, dups


def wait_rows(c, want, timeout=10):
    deadline = time.time() + timeout
    n = rows(c)
    while n != want and time.time() < deadline:
        time.sleep(0.05)
        n = rows(c)
    return n


def phase_after_data(binary, port, d, rng):
    print("=== FT.CREATE after the data: the index holds what is already there ===")
    p = serve(binary, port, d)
    c = Conn(port, timeout=60)
    try:
        vecs = rng.standard_normal((300, DIM)).astype(np.float32)
        keys = [f"doc:{i}" for i in range(300)]
        write(c, vecs, "doc:")
        c.pipeline([("HSET", f"other:{i}", "x", "1") for i in range(20)])               # no vector field
        c.pipeline([("HSET", f"bad:{i}", "vec", np.ones(8, np.float32).tobytes())        # wrong size
                    for i in range(10)])
        check("FT.CREATE over 330 existing hashes", create(c) == "OK")
        check("the 300 valid hashes are indexed when FT.CREATE answers (not the 30 others)",
              rows(c) == 300, rows(c))
        queries = rng.standard_normal((20, DIM)).astype(np.float32)
        r, dups = recall(c, vecs, keys, queries)
        check(f"KNN finds them (recall@10 vs numpy >= {FLOOR})", r >= FLOOR, r)
        i = info(c)
        check("nothing counted as an unindexed write", i.get("vector_unindexed_writes") == "0",
              i.get("vector_unindexed_writes"))

        print("=== a defensive re-create adds nothing ===")
        deadline = time.time() + 30
        while info(c).get("vector_index") != "graph" and time.time() < deadline:
            time.sleep(0.1)
        check("the graph built by itself", info(c).get("vector_index") == "graph", info(c).get("vector_index"))
        r_graph, _ = recall(c, vecs, keys, queries)
        check("FT.CREATE again, same name and DIM", create(c) == "OK")
        check("still 300 rows", wait_rows(c, 300, timeout=1) == 300, rows(c))
        r, dups = recall(c, vecs, keys, queries)
        check("no key twice in any answer", dups == 0, dups)
        check("the same answers as before it", r == r_graph, (r_graph, r))
        nv = rng.standard_normal((1, DIM)).astype(np.float32)
        write(c, nv, "doc:", start=300)
        check("a hash written afterwards is found", knn(c, nv[0], 1) == ["doc:300"], knn(c, nv[0], 1))
    finally:
        c.close()
        stop(p)


def phase_restart(binary, port, d, rng):
    print("=== restart: an index of another DIM than --dim comes back by FT.CREATE ===")
    p = serve(binary, port, d, wal=True)
    vecs = rng.standard_normal((300, DIM)).astype(np.float32)
    keys = [f"doc:{i}" for i in range(300)]
    queries = rng.standard_normal((20, DIM)).astype(np.float32)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, vecs, "doc:")
            check("indexed before the restart", wait_rows(c, 300) == 300, rows(c))
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d, wal=True)
    try:
        with Conn(port, timeout=60) as c:
            before = knn(c, queries[0])
            restored = not isinstance(before, RespError)
            print(f"    (index file {'restored' if restored else 'refused'} after the restart: "
                  f"DIM {DIM} vs --dim 1536)")
            check("the hashes survived the restart (WAL)", c.cmd("EXISTS", "doc:0", "doc:299") == 2)
            r = create(c)
            check("FT.CREATE after the restart", r == "OK", r)
            check("all 300 indexed again", wait_rows(c, 300) == 300, rows(c))
            rc, dups = recall(c, vecs, keys, queries)
            check(f"KNN finds them (recall@10 >= {FLOOR}), no duplicates", rc >= FLOOR and dups == 0, (rc, dups))
    finally:
        stop(p)


def phase_warm_restart(binary, port, d, rng):
    print("=== warm restart: FT.DROPINDEX then FT.CREATE takes what was written since ===")
    # At --dim 16 the index file loads and serves, but takes no new vector, and
    # re-issuing FT.CREATE leaves a served index as it is. A new index takes
    # every hash, the ones written after the restart included.
    p = serve(binary, port, d, wal=True, dim=DIM)
    vecs = rng.standard_normal((300, DIM)).astype(np.float32)
    try:
        with Conn(port, timeout=60) as c:
            create(c)
            write(c, vecs, "doc:")
            deadline = time.time() + 30
            while info(c).get("vector_index") != "graph" and time.time() < deadline:
                time.sleep(0.1)
            check("built before the restart", info(c).get("vector_index") == "graph", info(c).get("vector_index"))
    finally:
        stop(p, signal.SIGKILL)
    p = serve(binary, port, d, wal=True, dim=DIM)
    try:
        with Conn(port, timeout=60) as c:
            check("the index is restored", info(c).get("vector_index") == "graph" and rows(c) == 300, rows(c))
            nv = rng.standard_normal((2, DIM)).astype(np.float32)
            write(c, nv, "new:")
            check("FT.DROPINDEX", c.cmd("FT.DROPINDEX", "idx") == "OK")
            check("FT.CREATE takes all 302 hashes", create(c) == "OK" and rows(c) == 302, rows(c))
            check("including the two written after the restart",
                  knn(c, nv[0], 1) == ["new:0"] and knn(c, nv[1], 1) == ["new:1"], (knn(c, nv[0], 1), knn(c, nv[1], 1)))
    finally:
        stop(p)


def phase_skip_initial_scan(binary, port, d, rng):
    print("=== SKIPINITIALSCAN: only what is written afterwards ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, rng.standard_normal((100, DIM)).astype(np.float32), "old:")
            check("FT.CREATE ... SKIPINITIALSCAN", create(c, "SKIPINITIALSCAN") == "OK")
            check("nothing indexed", rows(c) == 0, rows(c))
            nv = rng.standard_normal((5, DIM)).astype(np.float32)
            write(c, nv, "new:")
            check("the 5 written afterwards are", rows(c) == 5, rows(c))
            got = knn(c, nv[2], 10)
            check("and only they answer", not isinstance(got, RespError) and set(got) <= {f"new:{i}" for i in range(5)}
                  and got[:1] == ["new:2"], got)
    finally:
        stop(p)


def phase_refused(binary, port, d, rng):
    print("=== a refused FT.CREATE indexes nothing ===")
    p = serve(binary, port, d)
    try:
        with Conn(port, timeout=60) as c:
            write(c, rng.standard_normal((50, DIM)).astype(np.float32), "doc:")
            r = create(c, metric="IP")
            check("DISTANCE_METRIC IP is refused", isinstance(r, RespError), r)
            i = info(c)
            check("no index, no rows", i.get("vector_index") == "none" and rows(c) == 0,
                  (i.get("vector_index"), rows(c)))
            check("a valid FT.CREATE then indexes all 50", create(c) == "OK" and rows(c) == 50, rows(c))
    finally:
        stop(p)


def phase_multi(binary, port, d, rng):
    print("=== -w 4: every worker indexes its own keyspace ===")
    p = serve(binary, port, d, workers=4)
    conns = []
    try:
        conns = [Conn(port + 2 + w, timeout=60) for w in range(4)]
        vecs = rng.standard_normal((200, DIM)).astype(np.float32)
        keys = [f"w{i // 50}:{i}" for i in range(200)]
        for w in range(4):
            write(conns[w], vecs[w * 50:(w + 1) * 50], f"w{w}:", start=w * 50)
        check("FT.CREATE on worker 0", create(conns[0]) == "OK")
        check("worker 0's 50 are indexed when it answers", rows(conns[0]) >= 50, rows(conns[0]))
        check("all 200 are indexed within 5 s", wait_rows(conns[0], 200, timeout=5) == 200, rows(conns[0]))
        queries = rng.standard_normal((20, DIM)).astype(np.float32)
        for w in range(4):
            r, dups = recall(conns[w], vecs, keys, queries)
            check(f"worker {w} answers over all four keyspaces (recall@10 >= {FLOOR})",
                  r >= FLOOR and dups == 0, (r, dups))
    finally:
        for c in conns:
            c.close()
        stop(p)


def phase_race(binary, port, d, rng):
    print("=== -w 4: FT.DROPINDEX and FT.CREATE while other workers backfill ===")
    # A worker backfills from housekeeping, between dispatch batches, where the
    # epoch wait counts it idle. It must hold the build state (AUTO_HOLD) so a
    # DROPINDEX or re-create on another worker cannot free the ingest buffer
    # under it. A stress, not a deterministic repro: the server must survive
    # and end where the last FT.CREATE puts it.
    p = serve(binary, port, d, workers=4)
    conns = []
    try:
        conns = [Conn(port + 2 + w, timeout=120) for w in range(4)]
        per = 20000
        vecs = rng.standard_normal((4 * per, DIM)).astype(np.float32)
        keys = [f"r{i // per}:{i}" for i in range(4 * per)]
        for w in range(4):
            write(conns[w], vecs[w * per:(w + 1) * per], f"r{w}:", start=w * per)
        ft_create = ("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", "vec", "VECTOR", "HNSW", "6",
                     "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")
        for _ in range(10):
            # sent back to back, read after: the three run on three workers at once
            conns[0].sock.sendall(encode(ft_create))
            conns[1].sock.sendall(encode(("FT.DROPINDEX", "idx")))
            conns[2].sock.sendall(encode(ft_create))
            for w in range(3):
                conns[w].read()
        alive = all(c.cmd("PING") == "PONG" for c in conns)
        check("every worker still answers after 10 rounds", alive)
        conns[3].cmd("FT.DROPINDEX", "idx")
        check("a last FT.CREATE", create(conns[0]) == "OK")
        n = wait_rows(conns[0], 4 * per, timeout=30)
        check(f"indexes all {4 * per}", n == 4 * per, n)
        queries = rng.standard_normal((10, DIM)).astype(np.float32)
        r, dups = recall(conns[3], vecs, keys, queries)
        check(f"and answers over them (recall@10 >= {FLOOR})", r >= FLOOR and dups == 0, (r, dups))
    finally:
        for c in conns:
            c.close()
        stop(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7481)
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    rng = np.random.default_rng(7)
    root = tempfile.mkdtemp(prefix="pion_backfill_")
    try:
        for n, phase in enumerate([phase_after_data, phase_restart, phase_warm_restart, phase_skip_initial_scan,
                                   phase_refused, phase_multi, phase_race]):
            d = os.path.join(root, str(n))
            os.makedirs(d)
            phase(binary, a.port, d, rng)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
