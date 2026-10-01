#!/usr/bin/env python3
"""W1.2 — prefix-length sweep for V-store consumer (mlx-lm).

Runs the W1/W1.1 100-request workload at prompt-repeats ∈ {2, 4, 8, 16}
(prefix sizes ≈ 316 / 632 / 1264 / 2528 tokens) under three configs:

  A: vanilla mlx-lm — cold prefill every request (baseline)
  C: Pion Stage 1 — V.STOREBATCH/V.FETCH RANGE
  D: Pion Stage 2 — ATTEND.PREFIX.STORE/QUERY via mlx_lm_patch

Tests the claim "TTFT win scales linearly with prefill cost." Vanilla TTFT grows ~linearly with prefix
length. Pion warm-path TTFT grows much slower (Stage 1: V.FETCH RANGE
linear in bytes; Stage 2: ATTEND.PREFIX.QUERY linear in N for QK^T but
small constant).

Outputs a single table comparing TTFT mean speedups across prefix sizes.

Requires: ./pion-server --kvcache --metal-attention -w 1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import socket
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from test_kv_prefix_workload import SYSTEM_PROMPTS, SYSTEM_PADDING, USER_QUERIES, system_prompt
from test_kv_prefix_mlx import (
    PionVStore, layout_from, cache_to_arrays, arrays_to_cache,
    store_prefix, fetch_prefix, forward_logits,
)

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def stats(arr):
    if not arr:
        return (0.0, 0.0, 0.0, 0.0)
    a = np.array(arr, dtype=np.float64)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 99)), float(a.max())


def run_workload_for_repeats(model, tok, layout, args, prompt_repeats: int,
                              run_id: str, install_pion_attention_patch,
                              uninstall_pion_attention_patch, make_pion_prompt_cache,
                              PionPromptCache):
    """Run all three configs at this prefix size; return per-config metrics."""
    workload = []
    for pi in range(args.prompts):
        for qi in range(args.queries):
            workload.append((pi, USER_QUERIES[(pi * 7 + qi) % len(USER_QUERIES)]))
    random.Random(0).shuffle(workload)

    sys_tokens = [tok.encode(system_prompt(pi, prompt_repeats)) for pi in range(args.prompts)]
    prefix_size = sys_tokens[0]
    print(f"\n=== prompt-repeats={prompt_repeats}  prefix≈{len(sys_tokens[0])} tokens  workload={len(workload)} reqs ===")

    # ── Config A: vanilla baseline ─────────────────────────────────────────
    print(f"  [A] vanilla...")
    a_ttfts = []
    a_t0 = time.perf_counter()
    for (pi, q) in workload:
        full = mx.array([sys_tokens[pi] + tok.encode(q)])
        ttft, _ = forward_logits(model, full)
        a_ttfts.append(ttft)
    a_wall = time.perf_counter() - a_t0
    a_mean, a_p50, a_p99, _ = stats(a_ttfts)
    a_throughput = len(workload) / a_wall

    # ── Config C: Pion Stage 1 ─────────────────────────────────────────────
    print(f"  [C] Pion Stage 1...")
    pion = PionVStore()
    prompt_cache_meta = {}
    c_ttfts = []
    c_t0 = time.perf_counter()
    for (pi, q) in workload:
        suffix_ids_list = tok.encode(q)
        suffix_ids = mx.array([suffix_ids_list])
        if pi not in prompt_cache_meta:
            prefix_ids_list = sys_tokens[pi]
            full_ids = mx.array([prefix_ids_list + suffix_ids_list])
            ttft, _ = forward_logits(model, full_ids)
            c_ttfts.append(ttft)
            cache = make_prompt_cache(model)
            prefix_ids = mx.array([prefix_ids_list])
            _ = model(prefix_ids, cache=cache); mx.eval(cache[0].keys)
            arrays = cache_to_arrays(cache, layout, len(prefix_ids_list))
            prefix_hash = hashlib.sha256(
                f"{run_id}|w1_2_s1|r{prompt_repeats}|p{pi}".encode()
            ).hexdigest()[:16]
            sid_k, sid_v, sid_kb, sid_vb, _, _, _ = store_prefix(
                pion, prefix_hash, arrays, layout, "fp16", 0,
            )
            prompt_cache_meta[pi] = (sid_k, sid_v, sid_kb, sid_vb, len(prefix_ids_list))
        else:
            sid_k, sid_v, sid_kb, sid_vb, prefix_len = prompt_cache_meta[pi]
            per_layer_q, fetch_ms, _ = fetch_prefix(
                pion, layout, prefix_len,
                sid_k_main=sid_k, sid_v_main=sid_v,
                sid_k_boundary=sid_kb, sid_v_boundary=sid_vb,
                boundary_layers=0,
            )
            rebuilt = arrays_to_cache(model, per_layer_q, layout, prefix_len)
            ttft, _ = forward_logits(model, suffix_ids, cache=rebuilt)
            c_ttfts.append(fetch_ms + ttft)
    c_wall = time.perf_counter() - c_t0
    c_mean, c_p50, c_p99, _ = stats(c_ttfts)
    c_throughput = len(workload) / c_wall

    # ── Config D: Pion Stage 2 ─────────────────────────────────────────────
    print(f"  [D] Pion Stage 2...")
    install_pion_attention_patch()
    try:
        pc = PionPromptCache(model, vquant="fp16", host="127.0.0.1", port=args.port,
                              stage2=True)
        prompt_meta = {}
        d_ttfts = []
        d_t0 = time.perf_counter()
        for (pi, q) in workload:
            if pi not in prompt_meta:
                full_ids = mx.array([sys_tokens[pi] + tok.encode(q)])
                ttft, _ = forward_logits(model, full_ids)
                ns = f"w1_2_{run_id}_r{prompt_repeats}_p{pi}"
                prefix_ids_list = sys_tokens[pi][:-1]
                prefix_len = len(prefix_ids_list)
                _ = pc.get_or_prefill(prefix_ids_list, namespace=ns)
                prompt_meta[pi] = (ns, prefix_len)
            else:
                ns, prefix_len = prompt_meta[pi]
                cache = make_pion_prompt_cache(model, namespace=ns,
                                                prompt_cache=pc, prefix_len=prefix_len)
                suffix_ids = mx.array([[sys_tokens[pi][-1]] + tok.encode(q)])
                ttft, _ = forward_logits(model, suffix_ids, cache=cache)
            d_ttfts.append(ttft)
        d_wall = time.perf_counter() - d_t0
        d_mean, d_p50, d_p99, _ = stats(d_ttfts)
        d_throughput = len(workload) / d_wall
        aq_calls = pc.attend_query_calls
        aq_per_call = (pc.attend_query_ms_total / aq_calls) if aq_calls else 0.0
    finally:
        uninstall_pion_attention_patch()

    return {
        "prefix_tokens": len(sys_tokens[0]),
        "n_reqs": len(workload),
        "A": {"mean": a_mean, "p50": a_p50, "p99": a_p99, "throughput": a_throughput, "wall": a_wall},
        "C": {"mean": c_mean, "p50": c_p50, "p99": c_p99, "throughput": c_throughput, "wall": c_wall},
        "D": {"mean": d_mean, "p50": d_p50, "p99": d_p99, "throughput": d_throughput, "wall": d_wall,
              "aq_calls": aq_calls, "aq_per_call_ms": aq_per_call},
    }


def main(args) -> int:
    print(f"W1.2 prefix-length sweep  model={args.model}")
    print(f"  pion: 127.0.0.1:{args.port}")

    try:
        s = socket.create_connection(("127.0.0.1", args.port), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable on port {args.port} ({e})"); return 2

    print("loading model...")
    model, tok = load(args.model)
    layout = layout_from(model)
    print(f"  layers={layout.n_layers}  n_kv_heads={layout.n_kv_heads}  head_dim={layout.head_dim}")

    from pion_vllm_mlx import PionPromptCache, install_pion_attention_patch, uninstall_pion_attention_patch
    from pion_vllm_mlx.mlx_lm_patch import make_pion_prompt_cache

    # Warmup
    full_w = mx.array([tok.encode(SYSTEM_PROMPTS[0] + USER_QUERIES[0])])
    for _ in range(args.warmup):
        forward_logits(model, full_w)

    run_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:8]
    repeats_list = [int(r) for r in args.repeats.split(",")]
    results = []
    for r in repeats_list:
        try:
            results.append(run_workload_for_repeats(
                model, tok, layout, args, r, run_id,
                install_pion_attention_patch, uninstall_pion_attention_patch,
                make_pion_prompt_cache, PionPromptCache,
            ))
        except Exception as e:
            print(f"\n  ERROR at prompt-repeats={r}: {type(e).__name__}: {e}")
            print(f"  Skipping remaining repeats; printing partial table.")
            break

    if not results:
        print("FAIL: no completed sweep points")
        return 1

    # ── Final table ────────────────────────────────────────────────────────
    print("\n\n──────── W1.2 sweep results ────────")
    print(f"  {args.prompts} prompts × {args.queries} queries each = {results[0]['n_reqs']} reqs per prefix size")
    print()
    print(f"  {'prefix':>7} | {'vanilla':>11} | {'Stage 1 fp16':>20} | {'Stage 2 metal':>20}")
    print(f"  {'tokens':>7} | {'TTFT mean':>11} | {'TTFT mean (×)':>20} | {'TTFT mean (×)':>20}")
    print(f"  {'-'*7}-+-{'-'*11}-+-{'-'*20}-+-{'-'*20}")
    for r in results:
        a = r["A"]; c = r["C"]; d = r["D"]
        cs = a["mean"] / c["mean"] if c["mean"] else float("inf")
        ds = a["mean"] / d["mean"] if d["mean"] else float("inf")
        print(f"  {r['prefix_tokens']:>7} | {a['mean']:>8.1f} ms | {c['mean']:>10.1f} ms ({cs:>4.2f}×) | {d['mean']:>10.1f} ms ({ds:>4.2f}×)")

    print()
    print(f"  {'prefix':>7} | {'vanilla':>11} | {'Stage 1 fp16':>20} | {'Stage 2 metal':>20}")
    print(f"  {'tokens':>7} | {'throughput':>11} | {'throughput (×)':>20} | {'throughput (×)':>20}")
    print(f"  {'-'*7}-+-{'-'*11}-+-{'-'*20}-+-{'-'*20}")
    for r in results:
        a = r["A"]; c = r["C"]; d = r["D"]
        cs = c["throughput"] / a["throughput"] if a["throughput"] else float("inf")
        ds = d["throughput"] / a["throughput"] if a["throughput"] else float("inf")
        print(f"  {r['prefix_tokens']:>7} | {a['throughput']:>5.2f} req/s | {c['throughput']:>5.2f} req/s ({cs:>4.2f}×) | {d['throughput']:>5.2f} req/s ({ds:>4.2f}×)")

    print()
    print(f"  Stage 2 ATTEND.PREFIX.QUERY cost growth with N:")
    print(f"  {'prefix':>7} | {'aq calls':>9} | {'ms/call':>9}")
    print(f"  {'-'*7}-+-{'-'*9}-+-{'-'*9}")
    for r in results:
        d = r["D"]
        print(f"  {r['prefix_tokens']:>7} | {d['aq_calls']:>9} | {d['aq_per_call_ms']:>7.2f}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--repeats", default="2,4,8,16",
                    help="Comma-separated prompt-repeats values for prefix sweep")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--port", type=int, default=1974)
    args = ap.parse_args()
    sys.exit(main(args))
