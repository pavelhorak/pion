# ---------------------------------------------------------------------------
# REPRODUCER -- backs a published number.
#
# Claim: Stage-0 hybrid K/V injection spike -- the chunk-keyed cache's first evidence.
#
# Requires: Apple Silicon + MLX + Llama-3.2-1B-Instruct-4bit.
#
# This is research code, moved here from the private research tree so the
# number it produces can be checked. It was not written to be read; it was
# written to answer one question. Expect rough edges, and read the
# prerequisites above before running -- most of the cost is the model
# download, and a run on a loaded machine produces a wrong number rather
# than an error.
# ---------------------------------------------------------------------------
"""
Stage 0 spike for gh #54: hybrid retrieval — verify K/V injection preserves
generation quality and quantify the latency win vs text-only RAG.

Method A (baseline): tokenize(chunk + query) -> prefill -> generate
Method B (hydrate):  pre-prefill chunk to cache, save K/V state, fresh cache,
                     restore K/V state, then prefill query -> generate
                     (and time the hydration step vs full prefill)

Pass criteria:
  Q1: token-level agreement on first 50 generated tokens >= 0.98
  Q2: hydration latency < 30% of prefill latency for the same chunk
  Q4: per-chunk INT4 K/V storage cost reported

Cache hit rate (Q3) handled in stage0_hybrid_hitrate.py.
"""

import argparse
import json
import time
from typing import List, Tuple

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache


def encode(tokenizer, text: str) -> mx.array:
    ids = tokenizer.encode(text)
    return mx.array([ids])


def prefill(model, cache, x: mx.array) -> None:
    _ = model(x, cache=cache)
    mx.eval([c.state for c in cache])


def greedy_step(model, cache, last_tok: mx.array) -> int:
    logits = model(last_tok, cache=cache)
    mx.eval(logits)
    next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    return next_id


def generate_n(model, tokenizer, cache, prompt_ids: mx.array, n: int) -> List[int]:
    """Greedy generate n tokens given a cache that may already be primed."""
    if prompt_ids.shape[1] > 0:
        prefill(model, cache, prompt_ids)
    out = []
    last = prompt_ids[:, -1:] if prompt_ids.shape[1] > 0 else mx.array([[tokenizer.bos_token_id or 1]])
    for _ in range(n):
        logits = model(last, cache=cache)
        mx.eval(logits)
        nid = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(nid)
        last = mx.array([[nid]])
    return out


def save_cache_state(cache) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Snapshot every layer's KV cache as numpy arrays plus offset."""
    snap = []
    for c in cache:
        # Attributes, not `c.state`: mlx-lm 0.32 widened `state` with scalars
        # and made it return the step-padded buffers.
        keys, values = c.keys[..., : c.offset, :], c.values[..., : c.offset, :]
        snap.append((np.array(keys, copy=True),
                     np.array(values, copy=True),
                     c.offset))
    return snap


def restore_cache_state(cache, snap) -> None:
    """Populate every layer's cache from a snapshot. Mirrors KV.HYDRATE."""
    for c, (k_np, v_np, off) in zip(cache, snap):
        c.keys, c.values = mx.array(k_np), mx.array(v_np)
        c.offset = off


def kv_storage_bytes(snap, dtype_bits: int) -> int:
    """How many bytes does this snapshot take at the given dtype precision?"""
    total = 0
    for k_np, v_np, _ in snap:
        # only count the actual offset slice, not the over-allocated buffer
        off = _
        elems = k_np[..., :off, :].size + v_np[..., :off, :].size
        total += elems * dtype_bits // 8
    return total


def run_one(args, model, tokenizer, chunk_text: str, query_text: str,
            n_gen: int, label: str) -> dict:
    print(f"\n--- {label} ---")
    print(f"chunk first 60 chars: {chunk_text[:60]!r}")
    print(f"query: {query_text!r}")

    # Method A: text-only RAG baseline (encode chunk+query together).
    chunk_ids = encode(tokenizer, chunk_text)
    query_ids = encode(tokenizer, query_text)
    full_ids = mx.concatenate([chunk_ids, query_ids], axis=1)
    chunk_len = chunk_ids.shape[1]

    cache_a = make_prompt_cache(model)
    t0 = time.perf_counter()
    prefill(model, cache_a, full_ids)
    t_prefill_full = time.perf_counter() - t0
    last_a = full_ids[:, -1:]
    out_a = []
    for _ in range(n_gen):
        nid = greedy_step(model, cache_a, last_a)
        out_a.append(nid)
        last_a = mx.array([[nid]])
    t_a_total = time.perf_counter() - t0

    # Method B: hydrate. First we have to *create* the chunk cache. In
    # production this happens once and is amortized; we still measure it.
    cache_chunk = make_prompt_cache(model)
    t0 = time.perf_counter()
    prefill(model, cache_chunk, chunk_ids)
    t_chunk_only_prefill = time.perf_counter() - t0
    snap = save_cache_state(cache_chunk)
    int4_bytes = kv_storage_bytes(snap, 4)
    fp16_bytes = kv_storage_bytes(snap, 16)

    # Now the hot path: hydrate fresh cache from snap, then prefill only the
    # query, then generate. This is what every cache hit does in production.
    cache_b = make_prompt_cache(model)
    t0 = time.perf_counter()
    restore_cache_state(cache_b, snap)
    t_hydrate = time.perf_counter() - t0
    t1 = time.perf_counter()
    prefill(model, cache_b, query_ids)
    t_query_prefill = time.perf_counter() - t1
    last_b = query_ids[:, -1:]
    out_b = []
    for _ in range(n_gen):
        nid = greedy_step(model, cache_b, last_b)
        out_b.append(nid)
        last_b = mx.array([[nid]])
    t_b_total = time.perf_counter() - t0

    # Token agreement.
    matches = sum(1 for a, b in zip(out_a, out_b) if a == b)
    agreement = matches / n_gen
    text_a = tokenizer.decode(out_a)
    text_b = tokenizer.decode(out_b)

    return {
        "label": label,
        "chunk_tokens": int(chunk_len),
        "query_tokens": int(query_ids.shape[1]),
        "n_generated": n_gen,
        "token_agreement": round(agreement, 4),
        "first_divergence": next((i for i, (a, b) in enumerate(zip(out_a, out_b)) if a != b), None),
        "method_a_text_first_80": text_a[:80],
        "method_b_text_first_80": text_b[:80],
        "t_full_prefill_ms": round(t_prefill_full * 1000, 2),
        "t_chunk_prefill_ms": round(t_chunk_only_prefill * 1000, 2),
        "t_hydrate_ms": round(t_hydrate * 1000, 2),
        "t_query_prefill_ms": round(t_query_prefill * 1000, 2),
        "t_method_a_total_ms": round(t_a_total * 1000, 2),
        "t_method_b_total_ms": round(t_b_total * 1000, 2),
        "ttft_savings_pct": round((t_prefill_full - (t_hydrate + t_query_prefill)) / t_prefill_full * 100, 2),
        "kv_int4_kb_per_chunk": round(int4_bytes / 1024, 1),
        "kv_fp16_kb_per_chunk": round(fp16_bytes / 1024, 1),
    }


# ---- chunks: hand-picked from existing pion-serve qa dataset ----

CHUNKS_AND_QUERIES = [
    (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Locally nicknamed 'La dame de fer', it was constructed from 1887 to 1889 as the centerpiece "
        "of the 1889 World's Fair. The tower is 330 metres tall, about the same height as an 81-storey "
        "building, and the tallest structure in Paris. Its base is square, measuring 125 metres on each "
        "side. During its construction, the Eiffel Tower surpassed the Washington Monument to become "
        "the tallest man-made structure in the world, a title it held for 41 years until the Chrysler "
        "Building in New York City was finished in 1930. ",
        "Question: How tall is the Eiffel Tower?\nAnswer:",
    ),
    (
        "Photosynthesis is a process used by plants and other organisms to convert light energy into "
        "chemical energy that, through cellular respiration, can later be released to fuel the organism's "
        "activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are "
        "synthesized from carbon dioxide and water. In most cases, oxygen is released as a by-product. "
        "Most plants, most algae, and cyanobacteria perform photosynthesis; such organisms are called "
        "photoautotrophs. Photosynthesis is largely responsible for producing and maintaining the oxygen "
        "content of the Earth's atmosphere, and supplies most of the energy necessary for life on Earth. ",
        "Question: What gas do plants release during photosynthesis?\nAnswer:",
    ),
    (
        "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from "
        "the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by the continents "
        "of Asia and Australia in the west and the Americas in the east. At 165,250,000 square kilometers "
        "in area (as defined with an Antarctic southern border), this largest division of the World Ocean "
        "covers about 46% of Earth's water surface and about 32% of its total surface area, larger than all "
        "of Earth's land area combined. The Mariana Trench in the western North Pacific is the deepest "
        "point in the world, reaching a depth of 10,928 meters. ",
        "Question: What is the deepest point in the Pacific Ocean?\nAnswer:",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--out", type=str, default="stage0_hybrid_results.json")
    args = ap.parse_args()

    print("=== Stage 0 spike: gh #54 hybrid retrieval ===")
    print("Loading mlx-community/Llama-3.2-1B-Instruct-4bit...")
    model, tok = mlx_load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    n_layers = len(model.layers)
    print(f"loaded — {n_layers} layers")

    results = {
        "config": vars(args),
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "n_layers": n_layers,
        "trials": [],
    }

    for i, (chunk, query) in enumerate(CHUNKS_AND_QUERIES):
        # warmup once on first trial
        if i == 0:
            print("warmup pass...")
            cache = make_prompt_cache(model)
            prefill(model, cache, encode(tok, chunk[:200]))
            _ = greedy_step(model, cache, encode(tok, "The")[:, -1:])
        trial = run_one(args, model, tok, chunk, query, args.n_gen, label=f"trial {i+1}")
        results["trials"].append(trial)
        print(json.dumps({k: v for k, v in trial.items()
                          if k not in ("method_a_text_first_80", "method_b_text_first_80")}, indent=2))
        print(f"  A: {trial['method_a_text_first_80']!r}")
        print(f"  B: {trial['method_b_text_first_80']!r}")

    # Aggregate.
    agreements = [t["token_agreement"] for t in results["trials"]]
    ttft_savings = [t["ttft_savings_pct"] for t in results["trials"]]
    kv_int4 = [t["kv_int4_kb_per_chunk"] for t in results["trials"]]
    summary = {
        "Q1_token_agreement_mean": round(sum(agreements) / len(agreements), 4),
        "Q1_token_agreement_min": round(min(agreements), 4),
        "Q1_pass_threshold": 0.98,
        "Q1_pass": all(a >= 0.98 for a in agreements),
        "Q2_ttft_savings_pct_mean": round(sum(ttft_savings) / len(ttft_savings), 2),
        "Q2_ttft_savings_pct_min": round(min(ttft_savings), 2),
        "Q2_pass_threshold_pct": 20.0,
        "Q2_pass": all(s >= 20.0 for s in ttft_savings),
        "Q4_kv_int4_kb_mean": round(sum(kv_int4) / len(kv_int4), 1),
        "Q4_kv_int4_kb_per_token": round(sum(kv_int4) / sum(t["chunk_tokens"] for t in results["trials"]) * len(kv_int4), 3),
    }
    results["summary"] = summary

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
