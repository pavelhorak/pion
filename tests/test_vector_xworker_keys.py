#!/usr/bin/env python3
"""Every worker names every document of the vector index by its key.

WHY
FT.SEARCH replies with each document's key. Workers find it in the slot map
they share (slot -> key), which until 1.0.6 held keys of up to 31 bytes only.
A longer key stayed in the slot map of the worker that ingested it, so under
-w N every other worker answered with the slot number as the key: `0`, `17`.
LangChain's keys (`doc:<index>:<uuid hex>`) are always longer, and RedisVL's
(`<index>:<ULID>`) nearly always, so at -w N most replies named their
documents by number. Nothing errored. The README promised "the same keys on
every worker".

Second, FT.DROPINDEX (or an FT.CREATE that replaces the index) starts the
slot numbering again, and only the worker that ran it dropped its records of
the old one. Another worker's records then named slots of the new index: that
worker answered a new long-key document with an OLD document's key, and after
a restart it linked its old hashes to the new index's slots, so deleting an
old document took a new one out of the results.

Every check reaches a chosen worker through its affinity port
(`port + 2 + worker`): an accept-race probe that lands on the ingesting
worker proves nothing.

    python3 tests/test_vector_xworker_keys.py [--binary ./pion-server] [--port 7551]
        [--legacy-binary /path/to/pion-1.0.5/pion-server.sh]

--legacy-binary adds the upgrade: an index a 1.0.5 server built at -w 2 (its
file holds no key over 31 bytes) serves every key on every worker once this
server loads it.
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
from resp_strict import Conn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16
fails = []


def check(name, ok, detail: object = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — {detail}"))
    if not ok:
        fails.append(name)
    return ok


def serve(binary, port, cwd, workers):
    cmd = [binary, "-p", str(port), "-w", str(workers), "--dim", str(DIM), "--wal-size", "16",
           "--no-auto-detect", "--no-auto-embed", "--no-crash-log"]
    if workers > 1:
        cmd.append("--independent-workers")
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(os.path.join(cwd, "log"), "a"),
                         stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 120
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"server on {port} exited {p.returncode}")
        try:
            ok = True
            for w in range(workers):
                with Conn(port + 2 + w if workers > 1 else port, timeout=2) as c:
                    ok = ok and c.cmd("PING") == "PONG"
            if ok:
                return p
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server on {port} did not start")


def stop(p, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(p.pid), sig)
        p.wait(timeout=30)
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


def wait_graph(c, rows, timeout=120):
    """The automatic build has served a graph of `rows` documents."""
    deadline = time.time() + timeout
    i = {}
    while time.time() < deadline:
        i = info(c)
        if i.get("vector_index") == "graph" and i.get("vector_graph_rows") == str(rows):
            return True
        time.sleep(0.1)
    print(f"    (waited {timeout}s for a graph of {rows}: {i.get('vector_index')} "
          f"{i.get('vector_graph_rows')})")
    return False


def create(c, name="idx", prefix=None):
    pre = ["PREFIX", "1", prefix] if prefix else []
    return c.cmd("FT.CREATE", name, "ON", "HASH", *pre, "SCHEMA", "t", "TEXT",
                 "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")


def write(c, docs):
    rs = c.pipeline([("HSET", k, "vec", v.astype(np.float32).tobytes(), "t", "x") for k, v in docs.items()])
    return all(r == 2 for r in rs)


def knn(c, q, k=1, name="idx"):
    r = c.cmd("FT.SEARCH", name, f"*=>[KNN {k} @vec $q]", "PARAMS", "2", "q",
              q.astype(np.float32).tobytes(), "NOCONTENT", "DIALECT", "2")
    if not isinstance(r, list):
        return r
    return [x.decode(errors="replace") for x in r[1:]]


def wrong_names(c, docs, name="idx"):
    """Documents a query for their own vector does not answer with their key."""
    out = []
    for k, v in docs.items():
        r = knn(c, v, 1, name)
        if r != [k]:
            out.append((k[:24], r if not isinstance(r, list) else [x[:24] for x in r]))
    return out


def lc_key(index, i, tag):
    """LangChain's shape: doc:<index>:<uuid4 hex> (38 + len(index) bytes)."""
    return f"doc:{index}:" + (f"{tag}{i:02d}" + "0123456789abcdef" * 2)[:32]


def all_workers(port, n, label, docs, name="idx"):
    for w in range(n):
        with Conn(port + 2 + w) as c:
            bad = wrong_names(c, docs, name)
            check(f"{label}: worker {w} names all {len(docs)} documents by their key", not bad, bad[:3])


def vecs(rng, n):
    return rng.standard_normal((n, DIM)).astype(np.float32)


# ── -w 2: long keys, the restart, and a renumbering ─────────────────────────

def phase_multi(binary, port, root):
    print("\n[A] -w 2: a key over 31 bytes, written on either worker, is the key on both")
    d = os.path.join(root, "multi")
    os.makedirs(d)
    rng = np.random.default_rng(1006)
    # Worker 0 writes first, so its documents hold slots 0.. of the first numbering
    old0 = {f"old:{i}": v for i, v in enumerate(vecs(rng, 6))}
    old0.update({lc_key("langchain_idx", i, "a"): v for i, v in enumerate(vecs(rng, 4))})
    old0["doc:" + "x" * 300] = vecs(rng, 1)[0]             # far past any inline limit
    old1 = {f"w1:{i}": v for i, v in enumerate(vecs(rng, 3))}
    old1.update({lc_key("langchain_idx", i, "b"): v for i, v in enumerate(vecs(rng, 4))})
    old = dict(old0, **old1)
    p = serve(binary, port, d, 2)
    try:
        ids = []
        for w in range(2):
            with Conn(port + 2 + w) as c:
                ids.append(info(c).get("worker_id"))
        if not check("affinity port port+2+w reaches worker w", ids == ["0", "1"], ids):
            return
        with Conn(port + 2) as c0, Conn(port + 3) as c1:
            check("FT.CREATE on worker 0", create(c0) == "OK")
            check("worker 0 writes its documents", write(c0, old0))
            check("worker 1 writes its documents", write(c1, old1))
        all_workers(port, 2, "exact scan, before a build", old)
        with Conn(port + 2) as c0:
            check("the index builds itself", wait_graph(c0, len(old)))
        all_workers(port, 2, "the built graph", old)
    finally:
        stop(p)

    print("\n[B] -w 2: the restart (the index file carries the long keys)")
    p = serve(binary, port, d, 2)
    try:
        with Conn(port + 2) as c0:
            check("the index loads", wait_graph(c0, len(old)))
        all_workers(port, 2, "after the restart", old)
        gone = lc_key("langchain_idx", 0, "b")             # worker 1's
        with Conn(port + 3) as c1:
            check("worker 1 deletes one of its long-key documents", c1.cmd("DEL", gone) == 1)
        for w in range(2):
            with Conn(port + 2 + w) as c:
                r = knn(c, old[gone], 5)
                check(f"worker {w}: the deleted document left the results", isinstance(r, list) and gone not in r, r)
        del old[gone]
        del old1[gone]

        print("\n[C] -w 2: FT.DROPINDEX on worker 1, a new index, new documents at the same slots")
        with Conn(port + 3) as c1:
            check("FT.DROPINDEX on worker 1", c1.cmd("FT.DROPINDEX", "idx") == "OK")
            # PREFIX keeps the old documents out of the new index, so the new
            # numbering's slots 0.. are worker 1's new documents, the very
            # numbers worker 0's old documents held
            check("FT.CREATE idx2 PREFIX new: on worker 1", create(c1, "idx2", "new:") == "OK")
        new1 = {lc_key("new_idx", i, "c").replace("doc:", "new:", 1): v for i, v in enumerate(vecs(rng, 8))}
        new0 = {lc_key("new_idx", i, "d").replace("doc:", "new:", 1): v for i, v in enumerate(vecs(rng, 3))}
        new0.update({f"new:s{i}": v for i, v in enumerate(vecs(rng, 2))})
        new = dict(new1, **new0)
        with Conn(port + 3) as c1, Conn(port + 2) as c0:
            check("worker 1 writes the new documents first", write(c1, new1))
            check("worker 0 writes its new documents", write(c0, new0))
        all_workers(port, 2, "the new index (was: an old document's key on worker 0)", new, "idx2")
        for w in range(2):
            with Conn(port + 2 + w) as c:
                r = knn(c, vecs(rng, 1)[0], len(new) + 5, "idx2")
                stale = [x for x in r if not x.startswith("new:")] if isinstance(r, list) else r
                check(f"worker {w}: only the new index's documents come back", isinstance(r, list) and not stale,
                      stale[:3] if isinstance(stale, list) else stale)
        with Conn(port + 2) as c0:
            check("the new index builds itself", wait_graph(c0, len(new)))
    finally:
        stop(p)

    print("\n[D] -w 2: the restart after the renumbering")
    p = serve(binary, port, d, 2)
    try:
        with Conn(port + 2) as c0:
            check("idx2 loads", wait_graph(c0, len(new)))
        all_workers(port, 2, "idx2 after the restart", new, "idx2")
        with Conn(port + 2) as c0:
            # worker 0's old documents held slots 0..10 of the first numbering;
            # were they linked to the new index's slots, deleting them would
            # take worker 1's new documents out of the results
            n = c0.cmd("DEL", *old0.keys())
            check(f"worker 0 deletes its {len(old0)} old documents", n == len(old0), n)
        all_workers(port, 2, "after the old documents went (was: new ones went with them)", new, "idx2")
        gone = next(iter(new1))
        with Conn(port + 3) as c1:
            check("worker 1 deletes a new document", c1.cmd("DEL", gone) == 1)
        for w in range(2):
            with Conn(port + 2 + w) as c:
                r = knn(c, new[gone], 5, "idx2")
                check(f"worker {w}: it left the results", isinstance(r, list) and gone not in r, r)
    finally:
        stop(p)


# ── -w 1: the same paths with one worker ────────────────────────────────────

def phase_single(binary, port, root):
    print("\n[E] -w 1: long keys through a build, a restart and two renumberings")
    d = os.path.join(root, "single")
    os.makedirs(d)
    rng = np.random.default_rng(1061)
    docs = {lc_key("one", i, "e"): v for i, v in enumerate(vecs(rng, 6))}
    docs.update({f"s:{i}": v for i, v in enumerate(vecs(rng, 3))})
    p = serve(binary, port, d, 1)
    try:
        with Conn(port) as c:
            check("FT.CREATE", create(c) == "OK")
            check("writes", write(c, docs))
            check("names every document (exact scan)", not wrong_names(c, docs), wrong_names(c, docs)[:3])
            check("builds", wait_graph(c, len(docs)))
            check("names every document (graph)", not wrong_names(c, docs), wrong_names(c, docs)[:3])
    finally:
        stop(p)
    p = serve(binary, port, d, 1)
    try:
        with Conn(port) as c:
            check("loads after the restart", wait_graph(c, len(docs)))
            check("names every document after the restart", not wrong_names(c, docs), wrong_names(c, docs)[:3])
            gone = next(iter(docs))
            check("DEL a long-key document", c.cmd("DEL", gone) == 1)
            r = knn(c, docs[gone], 5)
            check("it left the results", isinstance(r, list) and gone not in r, r)
            del docs[gone]
            for cycle in range(3):
                check(f"cycle {cycle}: FT.DROPINDEX", c.cmd("FT.DROPINDEX", "idx") == "OK")
                check(f"cycle {cycle}: FT.CREATE indexes the hashes again", create(c) == "OK")
                bad = wrong_names(c, docs)
                check(f"cycle {cycle}: every document by its key", not bad, bad[:3])
            # an FT.CREATE that replaces the index (another name) renumbers too
            check("FT.CREATE idx3 replaces idx", create(c, "idx3") == "OK")
            bad = wrong_names(c, docs, "idx3")
            check("idx3 names every document by its key", not bad, bad[:3])
            check("idx3 builds", wait_graph(c, len(docs)))
    finally:
        stop(p)
    p = serve(binary, port, d, 1)
    try:
        with Conn(port) as c:
            check("idx3 loads", wait_graph(c, len(docs)))
            bad = wrong_names(c, docs, "idx3")
            check("idx3 after the restart", not bad, bad[:3])
    finally:
        stop(p)


# ── an index 1.0.5 built ────────────────────────────────────────────────────

def phase_legacy(binary, legacy, port, root):
    print("\n[F] -w 2: an index a 1.0.5 server built, loaded by this one")
    d = os.path.join(root, "legacy")
    os.makedirs(d)
    rng = np.random.default_rng(1005)
    d0 = {lc_key("legacy", i, "f"): v for i, v in enumerate(vecs(rng, 5))}
    d1 = {lc_key("legacy", i, "g"): v for i, v in enumerate(vecs(rng, 5))}
    docs = dict(d0, **d1)
    p = serve(legacy, port, d, 2)
    try:
        with Conn(port + 2) as c0, Conn(port + 3) as c1:
            check("1.0.5: FT.CREATE and writes on both workers",
                  create(c0) == "OK" and write(c0, d0) and write(c1, d1))
            check("1.0.5: builds", wait_graph(c0, len(docs)))
            bad = wrong_names(c1, d0)
            check("1.0.5 itself names worker 0's long keys on worker 1 by slot number (the bug)",
                  len(bad) == len(d0), bad[:2])
    finally:
        stop(p)
    p = serve(binary, port, d, 2)
    try:
        with Conn(port + 2) as c0:
            check("this server loads 1.0.5's index", wait_graph(c0, len(docs)))
        all_workers(port, 2, "1.0.5's index", docs)
        gone = next(iter(d1))
        with Conn(port + 3) as c1:
            check("worker 1 deletes one of its documents", c1.cmd("DEL", gone) == 1)
        for w in range(2):
            with Conn(port + 2 + w) as c:
                r = knn(c, docs[gone], 5)
                check(f"worker {w}: it left the results", isinstance(r, list) and gone not in r, r)
    finally:
        stop(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7551)
    ap.add_argument("--legacy-binary", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the data directories")
    a = ap.parse_args()
    binary = os.path.abspath(a.binary)
    root = tempfile.mkdtemp(prefix="pion_xworker_keys_")
    try:
        phase_multi(binary, a.port, root)
        phase_single(binary, a.port, root)
        if a.legacy_binary:
            phase_legacy(binary, os.path.abspath(a.legacy_binary), a.port, root)
    finally:
        if a.keep:
            print(f"\n  data kept in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
