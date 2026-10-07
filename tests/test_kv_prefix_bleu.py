#!/usr/bin/env python3
"""§15.2 G1 — BLEU acceptance harness.

Replaces the strict logprob-delta criterion (which is too tight on small models
and on top-10-only slices, per §16-§18 findings) with the answer-equivalence
metric the doc itself names: BLEU > 0.95 on a 20-question eval set, with
50-token greedy completions per query, comparing the warm path (Pion via
PionPromptCache) against cold standalone forward.

This is the metric §15.2 G1 was always supposed to use; the logprob version
was a placeholder while we got the wire path working.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import List

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


SYSTEM = (
    "You are a senior systems engineer. Answer concisely, no filler, lead with the action. "
    "If the user is wrong about a fact, say so plainly. If you do not know, say so. "
    "Style: tight, imperative, one short example when it helps. "
)
EVAL_QUERIES = [
    "Question: how do tagged unions differ from sum types in low-level code?",
    "Question: what is a slab allocator good at and what is it bad at?",
    "Question: when should I evict from a fixed-capacity pool by LRU vs random?",
    "Question: name three patterns to graceful drain an event loop on shutdown.",
    "Question: what causes priority inversion and what is the simplest fix?",
    "Question: when does inlining a hot function backfire?",
    "Question: how do I add backpressure to a producer/consumer pipeline?",
    "Question: what makes a binary protocol easy to evolve later?",
    "Question: what is the difference between epoll edge-triggered and level-triggered?",
    "Question: why is io_uring sometimes slower than epoll at low concurrency?",
    "Question: when should I prefer flat arrays over linked lists for queues?",
    "Question: what does TCP_NODELAY actually do and when is it wrong to set it?",
    "Question: why does mmap append beat write() on append-only logs?",
    "Question: when should a server use SO_REUSEPORT?",
    "Question: how do I bound tail latency under load shedding?",
    "Question: what is the cost of false sharing on a hot atomic counter?",
    "Question: when is reference counting cheaper than tracing GC?",
    "Question: what is the right concurrency primitive for a single-producer multi-consumer queue?",
    "Question: how do I detect a memory leak that only shows up under sustained load?",
    "Question: what makes a benchmark reproducible across machines?",
]


def greedy_complete(model, prompt_ids: List[int], cache, max_new: int) -> List[int]:
    """Greedy decode max_new tokens. cache is mutated in place."""
    out_ids = []
    cur = mx.array([prompt_ids])
    logits = model(cur, cache=cache)
    next_id = int(mx.argmax(logits[0, -1]).item())
    out_ids.append(next_id)
    for _ in range(max_new - 1):
        cur = mx.array([[next_id]])
        logits = model(cur, cache=cache)
        next_id = int(mx.argmax(logits[0, -1]).item())
        out_ids.append(next_id)
    return out_ids


def bleu_score(reference: List[int], candidate: List[int], max_n: int = 4) -> float:
    """Token-level BLEU-4 with brevity penalty. Cheap, uses int IDs (no detokenize)."""
    if not candidate:
        return 0.0
    weights = [1.0 / max_n] * max_n
    log_precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        if not cand_ngrams:
            return 0.0
        ref_counts: dict[tuple, int] = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = 0
        used: dict[tuple, int] = {}
        for g in cand_ngrams:
            avail = ref_counts.get(g, 0) - used.get(g, 0)
            if avail > 0:
                match += 1
                used[g] = used.get(g, 0) + 1
        # smoothed precision
        precision = (match + 1) / (len(cand_ngrams) + 1)
        log_precisions.append(np.log(precision))
    bp = 1.0 if len(candidate) >= len(reference) else float(np.exp(1.0 - len(reference) / max(1, len(candidate))))
    return float(bp * np.exp(np.dot(weights, log_precisions)))


def main(args) -> int:
    print(f"§15.2 G1 BLEU eval  model={args.model}  vquant={args.vquant}  max_new={args.max_new}")

    try:
        s = socket.create_connection(("127.0.0.1", 1974), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e})")
        return 2

    print("loading model...")
    model, tok = load(args.model)

    prefix_ids = tok.encode(SYSTEM)
    pc = PionPromptCache(model, vquant=args.vquant)
    ns = PionPromptCache.make_namespace(args.model, "tok-v1", args.vquant, "bleu_eval_system")

    # Warmup
    out = model(mx.array([prefix_ids[:32]]), cache=make_prompt_cache(model)); mx.eval(out)

    # Build a single prefix cache in Pion (one register + store, then 20 reuses)
    initial_cache = pc.get_or_prefill(prefix_ids, ns)
    del initial_cache  # populated; subsequent get_or_prefill calls hit Pion

    bleus = []
    cold_lens = []
    warm_lens = []
    cold_total_ms = 0.0
    warm_total_ms = 0.0
    first_match = 0
    for qi, q in enumerate(EVAL_QUERIES[:args.queries]):
        # The question follows the prefix: no second <bos> (until 2026-10-07 it had one).
        suffix_ids = tok.encode(q, add_special_tokens=False)
        # COLD: standalone, full forward + greedy decode
        t0 = time.perf_counter()
        cold_cache = make_prompt_cache(model)
        cold_ids = greedy_complete(model, prefix_ids + suffix_ids, cold_cache, args.max_new)
        mx.eval(cold_cache[0].keys)
        cold_total_ms += (time.perf_counter() - t0) * 1000

        # WARM: PionPromptCache fetch + greedy decode of suffix only
        t0 = time.perf_counter()
        warm_cache = pc.get_or_prefill(prefix_ids, ns)
        warm_ids = greedy_complete(model, suffix_ids, warm_cache, args.max_new)
        mx.eval(warm_cache[0].keys)
        warm_total_ms += (time.perf_counter() - t0) * 1000

        bl = bleu_score(cold_ids, warm_ids)
        bleus.append(bl)
        cold_lens.append(len(cold_ids))
        warm_lens.append(len(warm_ids))
        first_match += int(cold_ids[0] == warm_ids[0])
        flag = "OK" if bl >= 0.95 else "warn"
        print(f"  q{qi:02d}: BLEU {bl:.3f}  first_token cold={cold_ids[0]} warm={warm_ids[0]} {flag}")

    mean_bleu = float(np.mean(bleus))
    n = len(bleus)
    print(f"\n──────── BLEU summary  ({args.vquant}) ────────")
    print(f"  queries          {n}")
    print(f"  mean BLEU        {mean_bleu:.4f}  (target > 0.95)")
    print(f"  median BLEU      {np.median(bleus):.4f}")
    print(f"  min   BLEU       {np.min(bleus):.4f}")
    print(f"  first-token match{first_match}/{n}")
    print(f"  cold total time  {cold_total_ms/n:.0f} ms/query")
    print(f"  warm total time  {warm_total_ms/n:.0f} ms/query")
    print(f"  speedup (full decode loop) {cold_total_ms / warm_total_ms:.2f}×")

    pass_bleu = mean_bleu >= 0.95
    print(f"\n  G1 BLEU criterion  {'PASS' if pass_bleu else 'FAIL'}")
    return 0 if pass_bleu else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--vquant", default="fp16", choices=["int8", "turbo4", "fp16"])
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--queries", type=int, default=20)
    args = ap.parse_args()
    sys.exit(main(args))
