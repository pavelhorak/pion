#!/usr/bin/env python3
"""§15.4 comparative benchmark — multi-query workload, MLX edition.

Simulates a multi-tenant SaaS where the same N system prompts are reused
across many user queries. Compares two configurations on the same Mac:

  A: mlx-lm standalone, cold prefill every request (state-of-the-art baseline)
  C: Pion S1 — first request per prompt is cold, every subsequent request
     fetches the prefix KV from Pion via V.FETCH RANGE and skips prefill

(Configs B (LMCache+Redis) and D (Pion+MLX sidecar attention) from §15.4 are
omitted: Redis/LMCache aren't installed locally, and the MLX sidecar attention
path is documented separately in test_stage2_attention.py.)

Workload: PROMPTS distinct system prompts × QUERIES_PER_PROMPT user queries each,
shuffled, run sequentially. All requests use greedy decode.

Measurements:
  Mean TTFT, P50, P99 TTFT
  Throughput (req/s wall-clock)
  Cache hit rate (Pion path)
  KV cache memory footprint
  Correctness: a hit's first token against the same prefix cache kept in
    process (and, for fp16 storage of an fp16 cache, bit-equal logits); a
    miss's against the cold baseline. Agreement of hits with the cold one-pass
    baseline is printed but not gated (see the comment at the check).

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import argparse
import hashlib
import random
import socket
import sys
import time

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Reuse the MLX adapter and PionVStore client from the G1 harness
sys.path.insert(0, "tests")
from _prompt_ids import one_bos, piece  # type: ignore
from test_kv_prefix_mlx import (  # type: ignore
    PionVStore, CacheLayout, layout_from,
    cache_to_arrays, arrays_to_cache,
    store_prefix, fetch_prefix,
    forward_logits, _log_softmax,
)

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# 5 distinct system prompts (each repeated 4× to push prefill ~600 tokens)
SYSTEM_PROMPTS = [
    "You are an expert technical writer. Be precise, lead with the answer, and use one short example. ",
    "You are a senior backend engineer. Optimize for correctness, then for clarity, then for speed. ",
    "You are an SRE. Always quote the exact log line, the runbook step, and the rollback. ",
    "You are a security analyst. Threat-model first, then propose mitigations with explicit assumptions. ",
    "You are a compiler engineer. Frame answers in terms of the IR transformation and its invariants. ",
]
# Pad each system prompt up to ~700 tokens so prefill cost is meaningful
SYSTEM_PADDING = (
    "House style: short sentences, no filler, no apologies. Cite your source if you have one. "
    "Avoid markdown headers in answers under 200 words. Format code blocks with the language tag. "
    "When the user is wrong, say so plainly and supply the correct fact. "
    "When the user asks for an opinion, give one and own the tradeoffs. "
    "If you do not know, say so and suggest the next investigation step. "
    "Prefer the load-bearing detail over a comprehensive overview. "
    "Skip pleasantries unless the user opens with one. "
    "Do not invent function names, file paths, or commands. "
    "If you are about to give a long answer, ask if a shorter one would do. "
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


def request_ids(tok, sys_ids, query):
    """A request: the system prompt (its own leading <bos>) and the query as
    a piece. A plain tok.encode(query) prepends a second <bos> mid-prompt,
    on Llama 3 as on Gemma 4 (tests/_prompt_ids.py)."""
    return one_bos(tok, list(sys_ids) + piece(tok, query))


def main(args) -> int:
    print(f"§15.4 workload  model={args.model}  prompts={args.prompts}  q_per_prompt={args.queries}  vquant={args.vquant}")
    print(f"  pion: 127.0.0.1:1974")

    try:
        s = socket.create_connection(("127.0.0.1", 1974), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e})")
        return 2

    print("loading model...")
    model, tok = load(args.model)
    layout = layout_from(model)
    print(f"  layers={layout.n_layers}  n_kv_heads={layout.n_kv_heads}  head_dim={layout.head_dim}")

    # Build the workload: (prompt_idx, query_text) tuples
    workload = []
    for pi in range(args.prompts):
        for qi in range(args.queries):
            workload.append((pi, USER_QUERIES[(pi * 7 + qi) % len(USER_QUERIES)]))
    random.Random(0).shuffle(workload)
    print(f"  total requests: {len(workload)}")

    # Pre-tokenize prompts and queries
    sys_tokens = [tok.encode(system_prompt(pi, args.prompt_repeats)) for pi in range(args.prompts)]
    sys_tokens_set = [tuple(t) for t in sys_tokens]
    print(f"  prompt sizes (tokens): {[len(t) for t in sys_tokens]}")

    # Warmup
    full_w = mx.array([request_ids(tok, sys_tokens[0], USER_QUERIES[0])])
    for _ in range(args.warmup):
        forward_logits(model, full_w)

    # ── Config A: standalone (cold every request) ───────────────────────────
    print("\n[A] standalone mlx-lm — cold prefill every request")
    a_ttfts = []
    a_first_tokens_by_key = {}  # for correctness check vs Pion path
    a_margin_by_key = {}        # vanilla's top-1 minus top-2 logit: how close a call it was
    a_t0 = time.perf_counter()
    for (pi, q) in workload:
        full = mx.array([request_ids(tok, sys_tokens[pi], q)])
        ttft, last = forward_logits(model, full)
        a_ttfts.append(ttft)
        key = (pi, q)
        a_first_tokens_by_key[key] = int(mx.argmax(last).item())
        top2 = mx.sort(mx.topk(last.reshape(-1).astype(mx.float32), 2))
        a_margin_by_key[key] = float((top2[1] - top2[0]).item())
    a_wall = time.perf_counter() - a_t0
    a_throughput = len(workload) / a_wall

    # ── Config C: Pion S1 (cold once per prompt, warm thereafter) ───────────
    print(f"[C] Pion S1 — V.STOREBATCH on first encounter, V.FETCH RANGE on hit ({args.vquant})")
    pion = PionVStore()
    prompt_cache_meta = {}  # pi -> (sid_k, sid_v, sid_kb, sid_vb, prefix_len)

    c_ttfts = []
    c_fetch_ms = []
    c_hits = 0
    c_first_tokens_by_key = {}
    c_hit_logits_by_key = {}   # a hit's last-token logits, for the bit-equality check
    local_arrays = {}          # pi -> the K/V exactly as sent to Pion, kept in process
    kv_dtype = None
    bytes_up_total = 0
    bytes_dn_total = 0
    c_t0 = time.perf_counter()
    for (pi, q) in workload:
        suffix_ids_list = piece(tok, q)
        suffix_ids = mx.array([suffix_ids_list])
        if pi not in prompt_cache_meta:
            # MISS: cold path, prefill prompt, push K/V to Pion
            prefix_ids_list = sys_tokens[pi]
            prefix_ids = mx.array([prefix_ids_list])
            full_ids = mx.array([prefix_ids_list + suffix_ids_list])
            ttft, last = forward_logits(model, full_ids)
            c_ttfts.append(ttft)
            # Capture prefix-only KV for storage
            cache = make_prompt_cache(model)
            _ = model(prefix_ids, cache=cache); mx.eval(cache[0].keys)
            kv_dtype = cache[0].keys.dtype
            arrays = cache_to_arrays(cache, layout, len(prefix_ids_list))
            local_arrays[pi] = arrays
            prefix_hash = hashlib.sha256(
                f"{args.model}|{args.vquant}|b{args.boundary}|p{pi}".encode()
            ).hexdigest()[:16]
            sid_k, sid_v, sid_kb, sid_vb, store_ms, bytes_up, _ = store_prefix(
                pion, prefix_hash, arrays, layout, args.vquant, args.boundary,
            )
            bytes_up_total += bytes_up
            prompt_cache_meta[pi] = (sid_k, sid_v, sid_kb, sid_vb, len(prefix_ids_list))
        else:
            # HIT: fetch KV, run only suffix
            sid_k, sid_v, sid_kb, sid_vb, prefix_len = prompt_cache_meta[pi]
            per_layer_q, fetch_ms, bytes_dn = fetch_prefix(
                pion, layout, prefix_len,
                sid_k_main=sid_k, sid_v_main=sid_v,
                sid_k_boundary=sid_kb, sid_v_boundary=sid_vb,
                boundary_layers=args.boundary,
            )
            rebuilt = arrays_to_cache(model, per_layer_q, layout, prefix_len)
            ttft, last = forward_logits(model, suffix_ids, cache=rebuilt)
            c_ttfts.append(fetch_ms + ttft)
            c_fetch_ms.append(fetch_ms)
            c_hits += 1
            bytes_dn_total += bytes_dn
            c_hit_logits_by_key[(pi, q)] = last
        c_first_tokens_by_key[(pi, q)] = int(mx.argmax(last).item())
    c_wall = time.perf_counter() - c_t0
    c_throughput = len(workload) / c_wall

    # ── Correctness ────────────────────────────────────────────────────────
    # What Pion must preserve is the cache: a hit has to answer as the same
    # prefix cache would had it never left the process. So a hit is checked
    # against that cache rebuilt locally from the arrays sent to Pion, and a
    # miss (a cold full prefill on both sides) against A.
    #
    # Agreement with A on hits is reported, not gated. Prefilling the prefix
    # and then the suffix rounds differently from one pass over both, with no
    # Pion involved: mlx-lm's own in-memory prompt cache does the same. On
    # 2026-10-10 the in-process cache, the fp16-rebuilt cache and Pion's
    # fetched cache all chose ' ' where one pass chose ' \n\n' (prompt 0,
    # "how should I version a binary ...", top-2 margin 0.016), and Pion's
    # logits were bit-equal to the in-process cache's on all 50 requests.
    ref_tokens = dict(a_first_tokens_by_key)       # misses: the cold prefill
    hits_bit_equal = 0
    for key, c_last in c_hit_logits_by_key.items():
        pi, q = key
        local = arrays_to_cache(model, local_arrays[pi], layout, prompt_cache_meta[pi][4])
        _, r_last = forward_logits(model, mx.array([piece(tok, q)]), cache=local)
        ref_tokens[key] = int(mx.argmax(r_last).item())
        hits_bit_equal += bool(mx.array_equal(c_last, r_last).item())
    matches = sum(1 for k, t in ref_tokens.items() if c_first_tokens_by_key.get(k) == t)
    correctness = matches / len(ref_tokens) if ref_tokens else 0.0
    oneshot_matches = sum(1 for k in a_first_tokens_by_key
                          if a_first_tokens_by_key[k] == c_first_tokens_by_key.get(k))
    # fp16 storage of an fp16 cache is lossless, so a hit must reproduce the
    # in-process logits bit for bit; any other format or model dtype may round.
    lossless = args.vquant == "fp16" and kv_dtype == mx.float16

    def stats(arr):
        if not arr:
            return (0.0, 0.0, 0.0, 0.0)
        a = np.array(arr, dtype=np.float64)
        return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 99)), float(a.max())

    a_mean, a_p50, a_p99, a_max = stats(a_ttfts)
    c_mean, c_p50, c_p99, c_max = stats(c_ttfts)
    fetch_p50 = float(np.percentile(c_fetch_ms, 50)) if c_fetch_ms else 0.0
    hit_rate = c_hits / len(workload)

    print("\n──────── §15.4 results ────────")
    print(f"  workload          {args.prompts} system prompts × {args.queries} queries = {len(workload)} reqs")
    print(f"")
    print(f"  Config A (cold every request):")
    print(f"    TTFT mean={a_mean:7.1f} ms  p50={a_p50:7.1f}  p99={a_p99:7.1f}  max={a_max:7.1f}")
    print(f"    throughput {a_throughput:6.2f} req/s  wall {a_wall:.1f}s")
    print(f"")
    print(f"  Config C (Pion S1, {args.vquant} b={args.boundary}):")
    print(f"    TTFT mean={c_mean:7.1f} ms  p50={c_p50:7.1f}  p99={c_p99:7.1f}  max={c_max:7.1f}")
    print(f"    throughput {c_throughput:6.2f} req/s  wall {c_wall:.1f}s")
    print(f"    cache hit rate    {hit_rate*100:5.1f}%  ({c_hits}/{len(workload)})")
    print(f"    fetch p50         {fetch_p50:5.1f} ms")
    print(f"    bytes up/down     {bytes_up_total/1024/1024:.1f}MB / {bytes_dn_total/1024/1024:.1f}MB")
    print(f"")
    print(f"  Pion vs standalone:")
    print(f"    TTFT mean speedup       {a_mean / c_mean if c_mean else float('inf'):.2f}×")
    print(f"    TTFT p99 speedup        {a_p99 / c_p99 if c_p99 else float('inf'):.2f}×")
    print(f"    throughput speedup      {c_throughput / a_throughput if a_throughput else float('inf'):.2f}×")
    print(f"    first-token agreement   {correctness*100:.1f}% ({matches}/{len(ref_tokens)})"
          f"  hits vs the same cache kept in process, misses vs A")
    for k, r_tok in ref_tokens.items():
        c_tok = c_first_tokens_by_key.get(k)
        if c_tok != r_tok:
            pi, q = k
            print(f"      disagrees: prompt {pi}, query {q[:40]!r}: reference {tok.decode([r_tok])!r}, "
                  f"pion {tok.decode([c_tok]) if c_tok is not None else None!r}")
    print(f"    hit logits bit-equal    {hits_bit_equal}/{len(c_hit_logits_by_key)}"
          f"  ({'required' if lossless else 'not required'}: {args.vquant} storage of a {kv_dtype} cache)")
    print(f"    agreement with A        {oneshot_matches}/{len(a_first_tokens_by_key)}"
          f"  (informational: one pass vs prefix-then-suffix)")
    for k, a_tok in a_first_tokens_by_key.items():
        c_tok = c_first_tokens_by_key.get(k)
        if c_tok != a_tok:
            pi, q = k
            print(f"      differs from A: prompt {pi}, query {q[:40]!r}: A {tok.decode([a_tok])!r}, "
                  f"pion {tok.decode([c_tok]) if c_tok is not None else None!r}, "
                  f"A's top-2 margin {a_margin_by_key[k]:.3f}")

    # Pass criteria from §15.4 win condition: Pion beats standalone on TTFT AND quality equivalent
    pass_ttft = c_mean < a_mean
    pass_corr = correctness >= 0.99 and (
        not lossless or hits_bit_equal == len(c_hit_logits_by_key))
    pass_throughput = c_throughput > a_throughput
    print(f"\n──────── verdict ────────")
    print(f"  TTFT win        {'PASS' if pass_ttft else 'FAIL'}")
    print(f"  throughput win  {'PASS' if pass_throughput else 'FAIL'}")
    print(f"  correctness     {'PASS' if pass_corr else 'FAIL'}")
    print(f"  §15.4 OVERALL   {'PASS' if pass_ttft and pass_corr and pass_throughput else 'FAIL'}")
    return 0 if (pass_ttft and pass_corr and pass_throughput) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--queries", type=int, default=10, help="queries per prompt")
    ap.add_argument("--prompt-repeats", type=int, default=2)
    ap.add_argument("--vquant", default="fp16", choices=["int8", "turbo4", "fp16"])
    ap.add_argument("--boundary", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    sys.exit(main(args))
