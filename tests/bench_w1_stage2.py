#!/usr/bin/env python3
"""W1.1 — Stage 2 workload bench (ATTEND.PREFIX.QUERY via mlx_lm_patch).

Mirrors tests/test_kv_prefix_workload.py (W1 Stage 1) but replaces the
V.STOREBATCH/V.FETCH RANGE cache-rebuild flow with the Stage-2 monkey-patch:
K/V stay resident in Pion's native Metal SDPA cache across sessions, and
mlx-lm's scaled_dot_product_attention is patched to route prefix attention
through ATTEND.PREFIX.QUERY (Q-only on the wire, online-softmax merge with
locally computed suffix attention).

Two configs:
  A: vanilla mlx-lm — cold prefill every request (same baseline as W1)
  D: Pion S2 — first request per prompt is cold (prefill + ATTEND.PREFIX.STORE);
     every subsequent request warm-skips prefill, attention runs on Pion-resident
     K/V via ATTEND.PREFIX.QUERY at every layer

Same workload as W1 (5 prompts × 20 queries × prompt-repeats=2 = ~315-token
prefixes, 100 requests) for direct comparison with the W1 run.

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

# Reuse workload definition from W1 bench
from test_kv_prefix_workload import (  # type: ignore
    SYSTEM_PROMPTS, SYSTEM_PADDING, USER_QUERIES, request_ids, system_prompt,
)
from test_kv_prefix_mlx import forward_logits  # type: ignore
from _prompt_ids import piece  # type: ignore

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def main(args) -> int:
    print(f"W1.1 Stage 2 workload  model={args.model}  prompts={args.prompts}  q_per_prompt={args.queries}")
    print(f"  pion: 127.0.0.1:{args.port}  vquant=fp16")

    try:
        s = socket.create_connection(("127.0.0.1", args.port), timeout=2)
        s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable on port {args.port} ({e})")
        return 2

    print("loading model...")
    if args.gemma4_filter:
        from _gemma4_text_filter_load import load_text_only_from_cached
        model, tok = load_text_only_from_cached(args.model)
    else:
        model, tok = load(args.model)
    args_obj = model.args
    n_layers = args_obj.num_hidden_layers
    n_kv = getattr(args_obj, "num_key_value_heads", args_obj.num_attention_heads)
    head_dim = getattr(args_obj, "head_dim", args_obj.hidden_size // args_obj.num_attention_heads)
    print(f"  layers={n_layers}  n_kv_heads={n_kv}  head_dim={head_dim}")

    # Workload
    workload = []
    for pi in range(args.prompts):
        for qi in range(args.queries):
            workload.append((pi, USER_QUERIES[(pi * 7 + qi) % len(USER_QUERIES)]))
    random.Random(0).shuffle(workload)
    print(f"  total requests: {len(workload)}")

    sys_tokens = [tok.encode(system_prompt(pi, args.prompt_repeats)) for pi in range(args.prompts)]
    print(f"  prompt sizes (tokens): {[len(t) for t in sys_tokens]}")

    # Warmup — vanilla path only (separate from Stage 2 install)
    full_w = mx.array([request_ids(tok, sys_tokens[0], USER_QUERIES[0])])
    for _ in range(args.warmup):
        forward_logits(model, full_w)

    # ── Config A: vanilla baseline (cold every request) ────────────────────
    print("\n[A] vanilla mlx-lm — cold prefill every request")
    a_ttfts = []
    a_first_tokens_by_key = {}
    a_t0 = time.perf_counter()
    for (pi, q) in workload:
        full = mx.array([request_ids(tok, sys_tokens[pi], q)])
        ttft, last = forward_logits(model, full)
        a_ttfts.append(ttft)
        a_first_tokens_by_key[(pi, q)] = int(mx.argmax(last).item())
    a_wall = time.perf_counter() - a_t0
    a_throughput = len(workload) / a_wall

    # ── Config D: Pion Stage 2 (ATTEND.PREFIX.QUERY) ───────────────────────
    print(f"\n[D] Pion S2 — ATTEND.PREFIX.STORE on first encounter, ATTEND.PREFIX.QUERY on hit")
    from pion_vllm_mlx import PionPromptCache
    from pion_vllm_mlx import install_pion_attention_patch, uninstall_pion_attention_patch
    from pion_vllm_mlx.mlx_lm_patch import make_pion_prompt_cache

    install_pion_attention_patch()
    try:
        pc = PionPromptCache(model, vquant="fp16", host="127.0.0.1", port=args.port,
                             stage2=True)
        # Per-prompt prefix metadata (built lazily on first MISS)
        prompt_meta = {}  # pi -> (namespace, prefix_len)
        # Use a unique run_id so namespaces don't collide with previous runs' V-store WAL
        run_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:8]

        d_ttfts = []
        d_first_tokens_by_key = {}
        d_t0 = time.perf_counter()
        for (pi, q) in workload:
            if pi not in prompt_meta:
                # COLD: vanilla forward for TTFT (apples-to-apples with Stage 1's
                # bench methodology), then separately push K/V to Pion. The
                # ATTEND.PREFIX.STORE cost lands in wall-clock throughput, not
                # per-request TTFT — same accounting as Stage 1's V.STOREBATCH.
                full_ids = mx.array([request_ids(tok, sys_tokens[pi], q)])
                ttft, last = forward_logits(model, full_ids)
                ns = f"w1_2_{run_id}_p{pi}"
                prefix_ids_list = sys_tokens[pi][:-1]
                prefix_len = len(prefix_ids_list)
                _ = pc.get_or_prefill(prefix_ids_list, namespace=ns)
                prompt_meta[pi] = (ns, prefix_len)
            else:
                # WARM: build PionPrefixCache list, run suffix-only forward
                # through the patched SDPA. TTFT = full perceived latency.
                ns, prefix_len = prompt_meta[pi]
                cache = make_pion_prompt_cache(model, namespace=ns,
                                                prompt_cache=pc, prefix_len=prefix_len)
                # Suffix = last prefix token + user query (mlx-lm convention).
                suffix_ids = mx.array([[sys_tokens[pi][-1]] + piece(tok, q)])
                ttft, last = forward_logits(model, suffix_ids, cache=cache)
            d_ttfts.append(ttft)
            d_first_tokens_by_key[(pi, q)] = int(mx.argmax(last).item())
        d_wall = time.perf_counter() - d_t0
        d_throughput = len(workload) / d_wall
    finally:
        uninstall_pion_attention_patch()

    # ── Correctness ────────────────────────────────────────────────────────
    matches = sum(1 for k in a_first_tokens_by_key
                  if a_first_tokens_by_key[k] == d_first_tokens_by_key.get(k))
    correctness = matches / len(a_first_tokens_by_key) if a_first_tokens_by_key else 0.0

    def stats(arr):
        if not arr:
            return (0.0, 0.0, 0.0, 0.0)
        a = np.array(arr, dtype=np.float64)
        return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 99)), float(a.max())

    a_mean, a_p50, a_p99, a_max = stats(a_ttfts)
    d_mean, d_p50, d_p99, d_max = stats(d_ttfts)
    pc_stats = pc.stats()
    hit_rate = pc_stats["hits"] / max(1, pc_stats["hits"] + pc_stats["misses"])
    aq_calls = pc.attend_query_calls
    aq_ms = pc.attend_query_ms_total
    aq_per_call = (aq_ms / aq_calls) if aq_calls else 0.0

    print("\n──────── W1.1 Stage 2 results ────────")
    print(f"  workload          {args.prompts} system prompts × {args.queries} queries = {len(workload)} reqs")
    print(f"  prefix size       {len(sys_tokens[0])} tokens")
    print(f"")
    print(f"  Config A (vanilla, cold every request):")
    print(f"    TTFT mean={a_mean:7.1f} ms  p50={a_p50:7.1f}  p99={a_p99:7.1f}  max={a_max:7.1f}")
    print(f"    throughput {a_throughput:6.2f} req/s  wall {a_wall:.1f}s")
    print(f"")
    print(f"  Config D (Pion S2, --metal-attention):")
    print(f"    TTFT mean={d_mean:7.1f} ms  p50={d_p50:7.1f}  p99={d_p99:7.1f}  max={d_max:7.1f}")
    print(f"    throughput {d_throughput:6.2f} req/s  wall {d_wall:.1f}s")
    print(f"    cache hits        {pc_stats['hits']}/{pc_stats['hits']+pc_stats['misses']} ({hit_rate*100:.1f}%)")
    print(f"    ATTEND.PREFIX.QUERY  {aq_calls} calls, {aq_ms:.0f} ms total, {aq_per_call:.2f} ms/call")
    print(f"")
    print(f"  Pion S2 vs vanilla:")
    print(f"    TTFT mean speedup       {a_mean / d_mean if d_mean else float('inf'):.2f}×")
    print(f"    TTFT p99 speedup        {a_p99 / d_p99 if d_p99 else float('inf'):.2f}×")
    print(f"    throughput speedup      {d_throughput / a_throughput if a_throughput else float('inf'):.2f}×")
    print(f"    first-token agreement   {correctness*100:.1f}% ({matches}/{len(a_first_tokens_by_key)})")

    pass_ttft = d_mean < a_mean
    pass_throughput = d_throughput > a_throughput
    pass_corr = correctness >= 0.95   # Stage 2 should be near-bit-equivalent
    print(f"\n──────── verdict ────────")
    print(f"  TTFT win        {'PASS' if pass_ttft else 'FAIL'}")
    print(f"  throughput win  {'PASS' if pass_throughput else 'FAIL'}")
    print(f"  correctness ≥95% {'PASS' if pass_corr else 'FAIL'}  ({correctness*100:.1f}%)")
    print(f"  W1.1 OVERALL    {'PASS' if pass_ttft and pass_throughput and pass_corr else 'FAIL'}")
    return 0 if (pass_ttft and pass_throughput and pass_corr) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--queries", type=int, default=20, help="queries per prompt")
    ap.add_argument("--prompt-repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--gemma4-filter", action="store_true",
                    help="load text-only filter view of a multimodal Gemma 4 4-bit quant (gh #16)")
    args = ap.parse_args()
    sys.exit(main(args))
