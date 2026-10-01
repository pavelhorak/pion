#!/usr/bin/env python3
"""gh #405 — every HNSW node is reachable, and a warm restart serves the SAME graph.

Two defects behind "quant-mode builds sometimes serve 1.5-10pp lower recall;
the live index serves less than the same index warm-loaded":

1. **Islands.** The insert paths evict a full list's farthest link AND the
   evicted node's link back, so a dense region can lose every inbound link.
   Whether it does depends on insert order, which the parallel link phase makes
   nondeterministic: most OpenAI-50K builds left ~25 level-0 nodes unreachable
   from the entry point, some left an island of ~1,300. Island nodes are never
   returned, and a query whose upper-level descent lands inside the island is
   trapped in it (turbo recall 0.953 healthy, 0.880 live on an island build).
   FT.OPTIMIZE now links every unreachable node back in before compaction.

2. **A warm load dropped the upper layers.** load_from_disk read the neighbor
   pool — per-level counts included — and then built each node with the
   HNSWNode constructor, which zeroes those counts. Level 0 kept working (the
   beam reads l0_compact), so recall barely moved on a healthy build and nobody
   noticed, but every search after a restart started at the entry point. On an
   island build that happened to AVOID the trap, which is why warm looked better
   than live.

Checks: the saved index is parsed and level 0 must be fully reachable from the
entry point; the upper levels in the file must be non-empty; and a warm restart
must return byte-identical result lists to the live index for every query
(same graph, same search — before the fix they differed).

The dataset is clustered with near-duplicates, the shape that strands nodes:
the build on main leaves nodes unreachable on it.

Usage:
    python3 tests/test_gh405_hnsw_reachability.py [--binary ./pion-server] [--port 7405]
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gh403_ft_info_name import Client, serve, stop, wait_ready  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 1536          # warm load only restores the server's startup dim
PASS, FAIL = [], []


def check(name, ok, detail: object = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def dataset(rng):
    """Tight clusters plus exact duplicates: many near-equal distances make the
    evict-both-directions shrink strand whole regions."""
    centers = rng.standard_normal((40, DIM)).astype(np.float32)
    parts = []
    for c in centers:
        parts.append(c + 0.05 * rng.standard_normal((140, DIM)).astype(np.float32))
    base = np.concatenate(parts)
    dups = np.repeat(base[:40], 10, axis=0) + 1e-4 * rng.standard_normal((400, DIM)).astype(np.float32)
    v = np.concatenate([base, dups, rng.standard_normal((200, DIM)).astype(np.float32)])
    return v[rng.permutation(len(v))].astype(np.float32)


def graph_from_file(path):
    with open(path, "rb") as f:
        h = np.frombuffer(f.read(256), dtype=np.uint64)
        n, m, entry, maxl = int(h[2]), int(h[4]), int(h[6]), int(h[7])
        pool_per, maxe, stride = int(h[9]), int(h[10]), int(h[20])
        f.seek(256 + n * stride + maxe * 8)
        l0 = np.frombuffer(f.read(n * 33 * 4), dtype=np.uint32).reshape(n, 33)
        pool = np.frombuffer(f.read(maxe * pool_per * 4), dtype=np.uint32).reshape(maxe, pool_per)[:n]
        lv = np.frombuffer(f.read(n * 16), dtype=np.int64).reshape(n, 2)[:, 1]
    return dict(n=n, m=m, entry=entry, maxl=maxl, l0=l0, pool=pool, lv=lv)


def l0_reach(g):
    seen = np.zeros(g["n"], bool)
    seen[g["entry"]] = True
    q = deque([g["entry"]])
    while q:
        x = q.popleft()
        for nb in g["l0"][x, 1:1 + g["l0"][x, 0]]:
            if not seen[nb]:
                seen[nb] = True
                q.append(int(nb))
    return int(seen.sum())


def search_all(port, queries):
    c = Client(port)
    out = []
    for q in queries:
        r = c.cmd("FT.SEARCH", "idx", "*=>[KNN 20 @vector $q]", "PARAMS", "2", "q", q.tobytes())
        out.append([r[i] for i in range(1, len(r), 2)] if isinstance(r, list) else r)
    c.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7405)
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)  # the server runs in a temp cwd
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2
    workdir = tempfile.mkdtemp(prefix="pion-gh405-")
    try:
        return run(args, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run(args, workdir):
    print(f"gh #405 HNSW reachability + warm-load layers — binary={args.binary} dir={workdir}")
    rng = np.random.default_rng(405)
    vecs = dataset(rng)
    queries = (vecs[rng.choice(len(vecs), 150, replace=False)]
               + 0.02 * rng.standard_normal((150, DIM)).astype(np.float32))

    proc = serve(args.binary, args.port, 1, workdir, "server.log")
    try:
        if not check("server ready", wait_ready(args.port)):
            return 1
        c = Client(args.port)
        c.cmd("FT.CREATE", "idx", "ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA", "vector", "VECTOR",
              "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2")
        buf = b"".join(Client.encode("HSET", f"doc:{i}", "vector", v.tobytes()) for i, v in enumerate(vecs))
        c.sock.sendall(buf)
        acks = [c.read() for _ in range(len(vecs))]
        check(f"ingest {len(vecs)} vectors", all(a in (1, 2) for a in acks))
        check("FT.OPTIMIZE", c.cmd("FT.OPTIMIZE", "idx") == b"OK")
        c.close()
        live = search_all(args.port, queries)
    finally:
        stop(proc)

    g = graph_from_file(os.path.join(workdir, "pion.hnsw.0"))
    reach = l0_reach(g)
    check(f"level 0: every node reachable from the entry point ({reach}/{g['n']})",
          reach == g["n"], f"{g['n'] - reach} stranded")
    ent = g["entry"]
    upper = [int(g["pool"][ent, 8 * g["m"] + level]) for level in range(1, int(g["lv"][ent]) + 1)]
    check(f"the entry point has upper-level links in the file (levels 1..{g['lv'][ent]})",
          len(upper) > 0 and all(x > 0 for x in upper), upper)

    proc = serve(args.binary, args.port, 1, workdir, "server-warm.log")
    try:
        if not check("warm server ready", wait_ready(args.port)):
            return 1
        warm = search_all(args.port, queries)
    finally:
        stop(proc)
    loaded = "HNSW loaded from disk" in open(os.path.join(workdir, "server-warm.log")).read()
    check("the warm server really loaded the index", loaded)
    same = sum(1 for a, b in zip(live, warm) if a == b)
    check(f"warm results identical to live for every query ({same}/{len(live)}) "
          "(was: the load zeroed the upper layers)", same == len(live))

    print(f"\n{'=' * 60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
