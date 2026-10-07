#!/usr/bin/env python3
"""Cross-instance KV cache sharing — closes the §1 "shared" claim.

Two independent PionPromptCache instances against the same pion-server:
  Client A: prefill prompt locally, register namespace, push K/V via STOREBATCH.
  Client B: a *fresh* instance, *separate socket*, *different model object*.
            B never prefills the prompt itself. It calls get_or_prefill on the
            same namespace and must get a HIT — proving the K/V it received came
            from Pion, not from B's own forward pass.

Pass criteria:
  1. B sees a HIT (KV.PREFIX.LOOKUP returns +HIT before B touches the model).
  2. B's first generated token matches A's standalone forward of the same prompt.
  3. B's BLEU against the standalone reference is ≥ 0.95.

Each client uses an independent mlx_lm.load() call to make sure there's no
shared MLX cache state — only Pion is the channel between them.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import socket
import sys
import time

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

SYSTEM = (
    "You are a senior systems engineer. Answer concisely, no filler, lead with the action. "
    "If the user is wrong about a fact, say so plainly. If you do not know, say so. "
    "Style: tight, imperative, one short example when it helps. "
    "House conventions: prefer fewer files over more, name things after what they do, never reach for "
    "a framework when a function will do, write tests before refactors, treat warnings as bugs. "
)
QUERY = " Question: how do I detect priority inversion in a worker pool?"


def greedy_complete(model, prompt_ids, cache, max_new: int):
    cur = mx.array([prompt_ids])
    out = model(cur, cache=cache)
    next_id = int(mx.argmax(out[0, -1]).item())
    out_ids = [next_id]
    for _ in range(max_new - 1):
        cur = mx.array([[next_id]])
        out = model(cur, cache=cache)
        next_id = int(mx.argmax(out[0, -1]).item())
        out_ids.append(next_id)
    return out_ids


def bleu_score(reference, candidate, max_n: int = 4) -> float:
    if not candidate:
        return 0.0
    weights = [1.0 / max_n] * max_n
    log_precisions = []
    for n in range(1, max_n + 1):
        cand = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        ref = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        if not cand:
            return 0.0
        rc: dict[tuple, int] = {}
        for g in ref:
            rc[g] = rc.get(g, 0) + 1
        match = 0
        used: dict[tuple, int] = {}
        for g in cand:
            avail = rc.get(g, 0) - used.get(g, 0)
            if avail > 0:
                match += 1
                used[g] = used.get(g, 0) + 1
        precision = (match + 1) / (len(cand) + 1)
        log_precisions.append(np.log(precision))
    bp = 1.0 if len(candidate) >= len(reference) else float(np.exp(1.0 - len(reference) / max(1, len(candidate))))
    return float(bp * np.exp(np.dot(weights, log_precisions)))


def main() -> int:
    print("Cross-instance KV cache test")

    try:
        s = socket.create_connection(("127.0.0.1", 1974), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e})")
        return 2

    print("loading two independent model instances...")
    model_a, tok_a = load(MODEL)
    model_b, tok_b = load(MODEL)
    assert model_a is not model_b, "models must be distinct objects"

    prefix_ids = tok_a.encode(SYSTEM)
    # The query follows the prefix: no second <bos> (until 2026-10-07 it had one).
    suffix_ids_list = tok_a.encode(QUERY, add_special_tokens=False)
    suffix_ids = mx.array([suffix_ids_list])
    full_ids = mx.array([prefix_ids + suffix_ids_list])

    # Reference: standalone cold forward, used as ground truth for B's correctness
    ref_cache = make_prompt_cache(model_a)
    ref_ids = greedy_complete(model_a, prefix_ids + suffix_ids_list, ref_cache, max_new=50)

    # Use a unique namespace so we're not picking up anything cached from prior runs
    ts = int(time.time())
    namespace = PionPromptCache.make_namespace(MODEL, "tok-v1", "fp16", "cross_instance", str(ts))

    # ── Client A: prefill + register + store ────────────────────────────────
    print(f"\n[A] client A registers namespace and pushes K/V")
    pc_a = PionPromptCache(model_a, vquant="fp16")
    t0 = time.perf_counter()
    cache_a = pc_a.get_or_prefill(prefix_ids, namespace)  # MISS: prefill + store
    a_ms = (time.perf_counter() - t0) * 1000
    a_stats = pc_a.stats()
    print(f"  client A: {a_ms:.1f}ms, hits={a_stats['hits']} misses={a_stats['misses']}")
    assert a_stats["misses"] == 1 and a_stats["hits"] == 0, "A should be a miss"

    # ── Client B: fresh instance, fresh socket; never prefilled this prompt ─
    print(f"[B] client B (fresh instance, fresh socket, never prefilled) looks up namespace")
    pc_b = PionPromptCache(model_b, vquant="fp16")
    # Pre-check: lookup returns HIT (so we know B's HIT is purely from Pion's state)
    pre_lookup = pc_b.lookup(namespace)
    print(f"  B.lookup before any get_or_prefill call: {'+HIT' if pre_lookup else '+MISS'}")
    assert pre_lookup, "B must see a HIT from Pion before doing any local work"

    t0 = time.perf_counter()
    cache_b = pc_b.get_or_prefill(prefix_ids, namespace)  # HIT: pure fetch
    b_fetch_ms = (time.perf_counter() - t0) * 1000
    b_stats = pc_b.stats()
    print(f"  client B fetch: {b_fetch_ms:.1f}ms, hits={b_stats['hits']} misses={b_stats['misses']}")
    assert b_stats["hits"] == 1 and b_stats["misses"] == 0, "B should be a pure hit"

    # B uses the fetched cache to generate. If correct, it proves the K/V came
    # from A → Pion → B and is faithful enough to drive the model.
    b_ids = greedy_complete(model_b, suffix_ids_list, cache_b, max_new=50)

    first_match = (b_ids[0] == ref_ids[0])
    bleu = bleu_score(ref_ids, b_ids)

    print(f"\n──────── cross-instance result ────────")
    print(f"  A miss-path latency        {a_ms:7.1f} ms (prefill + store)")
    print(f"  B hit-path latency         {b_fetch_ms:7.1f} ms (fetch only)")
    print(f"  B speedup over A           {a_ms / b_fetch_ms:.2f}×")
    print(f"  pre-lookup HIT before B's work?  {'YES' if pre_lookup else 'NO'}")
    print(f"  B first-token == ref?            {'YES' if first_match else 'NO'}  (B={b_ids[0]} ref={ref_ids[0]})")
    print(f"  B BLEU vs ref (50 tokens)        {bleu:.4f}  (target ≥ 0.95)")

    pass_lookup = pre_lookup
    pass_first = first_match
    pass_bleu = bleu >= 0.95
    overall = pass_lookup and pass_first and pass_bleu
    print(f"\n  cross-instance LOOKUP   {'PASS' if pass_lookup else 'FAIL'}")
    print(f"  cross-instance correctness (first-token)  {'PASS' if pass_first else 'FAIL'}")
    print(f"  cross-instance correctness (BLEU)         {'PASS' if pass_bleu else 'FAIL'}")
    print(f"  CROSS-INSTANCE          {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
