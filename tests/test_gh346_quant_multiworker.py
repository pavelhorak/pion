#!/usr/bin/env python3
"""Regression gate for gh #346: every worker must serve a quantized index.

`publish_to_shared` / `borrow_from_shared` carried TurboQuant's format flag but
not NanoQuant's (`compact_is_2bit`), and no quant variant's FP32 re-rank buffer.
A worker that BORROWED the index therefore ran the INT8 beam over INT2 bytes
(NanoQuant) or skipped the re-rank (all three), while the worker that ran
FT.OPTIMIZE answered correctly — so which answer you got depended on which
worker won the accept race.

Two measurement rules, both from gh #253:
  * connections are opened CONCURRENTLY — serially-opened connections all land
    on one worker and an agreeing result proves nothing;
  * the test does not assume a spread, it MEASURES it: workers keep private
    keyspaces, so each connection SETs a marker key and two connections share a
    worker iff each sees the other's marker. At least two groups are required,
    or the run is inconclusive and fails.

Recall is then computed per worker group against brute force.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_gh211_warm_restart as base  # noqa: E402  (start/stop/cleanup helpers)

import redis  # noqa: E402

N = 2000
K = 10
N_QUERIES = 40
N_CONNS = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6412)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "pion-server"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quant", choices=sorted(base.QUANT_FLAGS), default="nanoquant")
    args = ap.parse_args()

    base.cleanup_state()
    rng = np.random.default_rng(0x346)
    vecs = rng.standard_normal((N, base.DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    queries = (vecs[rng.choice(N, N_QUERIES, replace=False)]
               + 0.01 * rng.standard_normal((N_QUERIES, base.DIM))).astype(np.float32)
    truth = [{f"doc:{j}" for j in np.argsort(-(vecs @ q))[:K]} for q in queries]

    log = f"/tmp/pion_gh346_{args.port}.log"
    proc = base.start_pion(args.port, args.binary, args.workers, log, args.quant)
    try:
        r = redis.Redis(port=args.port)
        r.execute_command(
            "FT.CREATE", base.INDEX, "ON", "HASH", "PREFIX", "1", "doc:",
            "SCHEMA", "vector", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(base.DIM), "DISTANCE_METRIC", "COSINE")
        pipe = r.pipeline(transaction=False)
        for i in rng.permutation(N):
            pipe.hset(f"doc:{i}", mapping={"id": str(i), "vector": vecs[i].tobytes()})
        pipe.execute()
        r.execute_command("FT.OPTIMIZE", base.INDEX)
        r.close()

        marker = base.QUANT_BUILD_MARKER.get(args.quant)
        if marker and marker not in open(log).read():
            print(f"FAIL: FT.OPTIMIZE never logged '{marker}' (gh #350)")
            return 1

        # Open all connections at once so the accept race spreads them.
        conns: list = [None] * N_CONNS
        barrier = threading.Barrier(N_CONNS)

        def opener(ci: int) -> None:
            barrier.wait()
            c = redis.Redis(port=args.port)
            c.ping()  # force the socket open now, inside the race
            conns[ci] = c

        threads = [threading.Thread(target=opener, args=(ci,)) for ci in range(N_CONNS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for ci, c in enumerate(conns):
            c.set(f"gh346:marker:{ci}", "1")
        groups: dict = {}
        for ci, c in enumerate(conns):
            seen = tuple(cj for cj in range(N_CONNS) if c.exists(f"gh346:marker:{cj}"))
            groups.setdefault(seen, []).append(ci)
        print(f"{N_CONNS} concurrent connections landed on {len(groups)} worker(s): "
              + ", ".join(str(len(v)) for v in groups.values()))
        if len(groups) < 2:
            print("FAIL: inconclusive — every connection landed on one worker")
            return 1

        worst = 1.0
        for gi, members in enumerate(groups.values()):
            c = conns[members[0]]
            hits = 0
            for q, t in zip(queries, truth):
                hits += len(set(base.knn_keys(c, q)) & t)
            rec = hits / (K * len(truth))
            worst = min(worst, rec)
            print(f"  worker group {gi} ({len(members)} conns): recall@{K} = {rec:.3f}")
        for c in conns:
            c.close()
    finally:
        base.stop_pion(proc)
        base.cleanup_state()

    if worst < base.MIN_RECALL[args.quant]:
        print(f"FAIL: a worker serves {args.quant} at recall {worst:.3f} < {base.MIN_RECALL[args.quant]} "
              "(gh #346: quant state not published to the shared view)")
        return 1
    print(f"PASS: every worker serves {args.quant} (min recall {worst:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
