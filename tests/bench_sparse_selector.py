#!/usr/bin/env python3
"""gh #191 — end-to-end cost of the sparse block-score selector.

The selector runs on the CPU inside `ATTEND.PREFIX.QUERY_SPARSE_AUTO`, before
the Metal SDPA dispatch. A standalone kernel harness measures the loop in
isolation; this measures what a caller actually waits for, which is always the
smaller number — the same call also pays RESP framing, the precompute read and
the GPU dispatch. Report both or the speedup reads as an end-to-end claim it
is not.

Both selectors are exercised: they have different arithmetic (`block_mean` is
a dot product, `quest` is a per-lane max of two products) and different
speedups.

Usage:
    ./pion-server --kvcache --metal-attention -w 1 &
    python3 tests/bench_sparse_selector.py [--ctx 65536] [--iters 40]
"""
import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))

from pion_vllm_mlx.prompt_cache import PionPromptCache   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--ktop", type=int, default=400)
    args = ap.parse_args()

    n_blocks = (args.ctx + args.block - 1) // args.block
    k_top = min(args.ktop, n_blocks)

    pc = PionPromptCache(host="127.0.0.1", port=args.port)
    ns = "gh191_%d" % args.ctx
    rng = np.random.default_rng(4242)

    K = rng.standard_normal((args.hkv, args.ctx, args.d)).astype(np.float32)
    V = rng.standard_normal((args.hkv, args.ctx, args.d)).astype(np.float32)
    pc.attend_store_layer(ns, 0, K, V)

    Q = rng.standard_normal((args.hq, args.d)).astype(np.float32)
    head_map = np.array([h // (args.hq // args.hkv) for h in range(args.hq)],
                        dtype=np.uint8)

    print("ctx=%d  n_blocks=%d  D=%d  H_q=%d  H_kv=%d  K_top=%d  iters=%d"
          % (args.ctx, n_blocks, args.d, args.hq, args.hkv, k_top, args.iters))
    out = {}
    for selector in ("block_mean", "quest"):
        for _ in range(3):
            pc.attend_query_sparse_auto(ns, 0, Q, args.block, k_top,
                                        H_kv=args.hkv, head_map=head_map,
                                        selector=selector)
        r = None
        t0 = time.perf_counter()
        for _ in range(args.iters):
            r = pc.attend_query_sparse_auto(ns, 0, Q, args.block, k_top,
                                            H_kv=args.hkv, head_map=head_map,
                                            selector=selector)
        dt = (time.perf_counter() - t0) / args.iters * 1000.0
        out[selector] = np.asarray(r)
        print("  %-11s %7.3f ms/call end-to-end" % (selector, dt))

    # Print a checksum so an A/B across binaries can confirm the OUTPUT is
    # unchanged, not just that both were fast.
    for selector, r in out.items():
        a = np.asarray(r, dtype=np.float64)
        print("  %-11s out sum=%.9f  norm=%.9f" % (selector, a.sum(),
                                                   float(np.linalg.norm(a))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
