#!/usr/bin/env python3
"""§15.4 workload through the PionPromptCache user-facing API.

Equivalent to tests/test_kv_prefix_workload.py in the parent repo, but exercises
the package-level PionPromptCache (drop-in for make_prompt_cache) instead of the
hand-rolled V.STOREBATCH/V.FETCH RANGE calls.

If this passes the same way the lower-level harness did (4× TTFT, 3× throughput,
100% first-token agreement on fp16), the public API is ready for end users.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import argparse
import random
import socket
import sys
import time

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Allow `pip install -e pion-vllm-mlx/` OR running from repo
sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


SYSTEM_PROMPTS = [
    "You are an expert technical writer. Be precise, lead with the answer, and use one short example. ",
    "You are a senior backend engineer. Optimize for correctness, then for clarity, then for speed. ",
    "You are an SRE. Always quote the exact log line, the runbook step, and the rollback. ",
    "You are a security analyst. Threat-model first, then propose mitigations with explicit assumptions. ",
    "You are a compiler engineer. Frame answers in terms of the IR transformation and its invariants. ",
]
SYSTEM_PADDING = (
    "House style: short sentences, no filler, no apologies. Cite your source if you have one. "
    "Avoid markdown headers in answers under 200 words. Format code blocks with the language tag. "
    "When the user is wrong, say so plainly. When the user asks for an opinion, give one and own the tradeoffs. "
    "If you do not know, say so and suggest the next investigation step. "
    "Prefer the load-bearing detail. Skip pleasantries. Do not invent function names or paths. "
)
USER_QUERIES = [
    "Question: how do I structure a tagged-union value type for a low-level engine?",
    "Question: when should I prefer a slab allocator over a general-purpose allocator?",
    "Question: what is the right way to evict entries from a fixed-capacity pool?",
    "Question: how do I detect and prevent priority inversion in a worker pool?",
    "Question: what's a robust pattern for gracefully draining an event loop on shutdown?",
    "Question: how do I avoid head-of-line blocking in a multi-tenant request queue?",
    "Question: when does it make sense to inline a hot function and when to split it?",
    "Question: what's the cleanest way to add backpressure to a producer/consumer pipeline?",
    "Question: how should I version a binary protocol that has to evolve?",
    "Question: how do I bound the worst-case latency of a generational GC for soft-realtime work?",
]


def system_prompt(i: int, repeats: int) -> str:
    return (SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)] + SYSTEM_PADDING) * repeats


def main(args) -> int:
    print(f"PionPromptCache workload  model={args.model}  prompts={args.prompts}  q={args.queries}  vquant={args.vquant}")

    try:
        s = socket.create_connection(("127.0.0.1", 1974), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e}). Start with: ./pion-server --kvcache -w 1")
        return 2

    print("loading model...")
    model, tok = load(args.model)

    # Pre-tokenize prompts
    sys_tokens = [tok.encode(system_prompt(pi, args.prompt_repeats)) for pi in range(args.prompts)]

    # Build workload
    workload = []
    for pi in range(args.prompts):
        for qi in range(args.queries):
            workload.append((pi, USER_QUERIES[(pi * 7 + qi) % len(USER_QUERIES)]))
    random.Random(0).shuffle(workload)

    # Warmup
    full_w = mx.array([sys_tokens[0] + tok.encode(USER_QUERIES[0])])
    for _ in range(args.warmup):
        out = model(full_w, cache=make_prompt_cache(model))
        mx.eval(out)

    # ── A: standalone (cold every request) ──────────────────────────────────
    print("[A] standalone — cold prefill every request")
    a_ttfts = []
    a_first_by_key = {}
    a_t0 = time.perf_counter()
    for (pi, q) in workload:
        full = mx.array([sys_tokens[pi] + tok.encode(q)])
        t0 = time.perf_counter()
        out = model(full, cache=make_prompt_cache(model))
        mx.eval(out)
        a_ttfts.append((time.perf_counter() - t0) * 1000)
        a_first_by_key[(pi, q)] = int(mx.argmax(out[0, -1]).item())
    a_wall = time.perf_counter() - a_t0
    a_throughput = len(workload) / a_wall

    # ── C: PionPromptCache ─────────────────────────────────────────────────
    if args.boundary_protect > 0:
        print(f"[C] PionPromptCache (vquant={args.vquant}, boundary_protect={args.boundary_protect})")
    else:
        print(f"[C] PionPromptCache (vquant={args.vquant})")
    pc = PionPromptCache(model, vquant=args.vquant, boundary_protect=args.boundary_protect)
    c_ttfts = []
    c_ttfts_cold = []  # first request per prompt (Pion MISS)
    c_ttfts_warm = []  # all subsequent requests on the same prompt (Pion HIT)
    seen_prompts = set()
    c_first_by_key = {}
    c_t0 = time.perf_counter()
    for (pi, q) in workload:
        ns = PionPromptCache.make_namespace(args.model, "tok-v1", args.vquant, f"prompt{pi}")
        suffix = mx.array([tok.encode(q)])
        is_cold = pi not in seen_prompts
        t0 = time.perf_counter()
        cache = pc.get_or_prefill(sys_tokens[pi], ns)
        out = model(suffix, cache=cache)
        mx.eval(out)
        dt = (time.perf_counter() - t0) * 1000
        c_ttfts.append(dt)
        if is_cold:
            c_ttfts_cold.append(dt)
            seen_prompts.add(pi)
        else:
            c_ttfts_warm.append(dt)
        c_first_by_key[(pi, q)] = int(mx.argmax(out[0, -1]).item())
    c_wall = time.perf_counter() - c_t0
    c_throughput = len(workload) / c_wall

    # Stats
    matches = sum(1 for k in a_first_by_key if a_first_by_key[k] == c_first_by_key.get(k))
    correctness = matches / len(a_first_by_key) if a_first_by_key else 0.0

    def stats(arr):
        a = np.array(arr, dtype=np.float64)
        return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 99))

    a_mean, a_p50, a_p99 = stats(a_ttfts)
    c_mean, c_p50, c_p99 = stats(c_ttfts)
    pc_stats = pc.stats()

    # Warm-only (every request after the first cold prefill per prompt)
    cw_mean = float(np.mean(c_ttfts_warm)) if c_ttfts_warm else 0.0
    cw_p50 = float(np.percentile(c_ttfts_warm, 50)) if c_ttfts_warm else 0.0
    cw_p99 = float(np.percentile(c_ttfts_warm, 99)) if c_ttfts_warm else 0.0
    cc_mean = float(np.mean(c_ttfts_cold)) if c_ttfts_cold else 0.0

    print("\n──────── PionPromptCache workload result ────────")
    print(f"  Config A (cold every): mean {a_mean:6.1f}ms  p50 {a_p50:6.1f}  p99 {a_p99:6.1f}  thr {a_throughput:.2f} req/s")
    print(f"  Config C (PionCache):  mean {c_mean:6.1f}ms  p50 {c_p50:6.1f}  p99 {c_p99:6.1f}  thr {c_throughput:.2f} req/s")
    print(f"    Pion warm-only ({len(c_ttfts_warm)}/{len(c_ttfts)}): mean {cw_mean:6.1f}ms  p50 {cw_p50:6.1f}  p99 {cw_p99:6.1f}")
    print(f"    Pion cold-only ({len(c_ttfts_cold)}/{len(c_ttfts)}): mean {cc_mean:6.1f}ms")
    print(f"  TTFT mean speedup       {a_mean / c_mean:.2f}×")
    print(f"  throughput speedup      {c_throughput / a_throughput:.2f}×")
    print(f"  hit rate                {pc_stats['hit_rate']*100:.1f}%  ({pc_stats['hits']}/{pc_stats['hits']+pc_stats['misses']})")
    print(f"  first-token agreement   {correctness*100:.1f}% ({matches}/{len(a_first_by_key)})")

    pass_ttft = c_mean < a_mean
    pass_throughput = c_throughput > a_throughput
    pass_corr = correctness >= 0.99
    overall = pass_ttft and pass_throughput and pass_corr
    print(f"\n  TTFT win        {'PASS' if pass_ttft else 'FAIL'}")
    print(f"  throughput win  {'PASS' if pass_throughput else 'FAIL'}")
    print(f"  correctness     {'PASS' if pass_corr else 'FAIL'}")
    print(f"  PUBLIC API      {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--prompt-repeats", type=int, default=2)
    ap.add_argument("--vquant", default="fp16", choices=["int8", "turbo4", "fp16", "fp8"])
    ap.add_argument("--boundary-protect", type=int, default=0,
                    help="boundary-layer fp16 protection count (gh #29 SCHEMA path); 0 = uniform vquant")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    sys.exit(main(args))
