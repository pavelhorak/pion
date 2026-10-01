#!/usr/bin/env python3
"""gh #60 Step 3b — Layer-type wall-time profile (sliding vs full).

Question for Step 2: should the sparse-mask kernel target FULL layers (D=512,
unsupported by Pion's Metal SDPA today; needs new PSO — option 2a) or just
SLIDING layers (D=256, already supported, but sliding layers are bounded at
window=512 and don't grow with context — option 2b)?

Rule of thumb: if full layers consume ≥40% of decode wall-time at 16K, the
D=512 PSO is mandatory; below that, 2b captures most of the value.

Method: monkey-patch `pion_scaled_dot_product_attention` in mlx_lm_patch.py
to record (cache.layer_idx, cache.fa_window, elapsed_ns) for every call. Run
N decode steps over a 16K prefix. Aggregate by fa_window category.

Honest limitations:
- Wall-time per call ≠ wall-time per layer in the model (the model has KV-shared
  layers that DON'T go through PionPrefixCache — they call _orig_sdpa with
  shared_kv from intermediates). We attribute those to the donor layer's
  category in a second aggregation step.
- Times include MLX lazy-graph realization, not just kernel execution. Use
  `mx.eval` per call to force realization at the right point — adds barrier
  overhead but makes the timing measurable.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import mlx.core as mx

sys.path.insert(0, "pion-vllm-mlx")
sys.path.insert(0, "tests")
from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx import mlx_lm_patch as _patch
from pion_vllm_mlx.mlx_lm_patch import (
    install_pion_attention_patch, make_pion_prompt_cache,
)
from _gemma4_text_filter_load import load_text_only_from_cached


def main(args) -> int:
    print(f"layer-type wall-time profile  model={args.model}  prefix={args.prefix_tokens}  decode={args.decode_tokens}")
    print("loading model...")
    model, tok = load_text_only_from_cached(args.model)
    layer_types = model.args.layer_types
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
    print(f"  layers: {len(layer_types)} total, {n_full} full, {n_sliding} sliding")

    # Build a long prefix.
    base = tok.encode("The quick brown fox jumps over the lazy dog. " * 64)
    while len(base) < args.prefix_tokens:
        base = base + base
    prefix_ids = base[: args.prefix_tokens]

    install_pion_attention_patch()
    pc = PionPromptCache(
        model, vquant="fp16", stage2=True,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    ns = "profile_layer_wall_time_v1"
    pc.get_or_prefill(prefix_ids, ns)
    cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids))

    # Monkey-patch pion_scaled_dot_product_attention to record per-call timing.
    layer_times_ns = defaultdict(list)   # key: (cache.fa_window category)
    orig_sdpa = _patch.pion_scaled_dot_product_attention

    def timed_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        # Classify by D (head dim): Gemma 4 sliding layers have head_dim=256,
        # full layers global_head_dim=512. KV-shared layers inherit the donor's
        # D, so they fall into the same category as their donor type.
        D = queries.shape[-1] if hasattr(queries, "shape") else 0
        is_pion = isinstance(cache, _patch.PionPrefixCache)
        if D == 512:
            category = "full(D=512)"
        elif D == 256:
            category = "sliding(D=256)"
        else:
            category = f"other(D={D})"
        # Sub-tag pion-routed vs kv-shared for diagnostics.
        category = f"{category} {'[pion]' if is_pion else '[shared]'}"
        # Force eval to bound the timing window.
        t0 = time.perf_counter_ns()
        out = orig_sdpa(queries, keys, values, cache, scale, mask, sinks)
        mx.eval(out)
        elapsed = time.perf_counter_ns() - t0
        layer_times_ns[category].append(elapsed)
        return out

    # Walk every mlx-lm model module that imported `scaled_dot_product_attention`
    # and replace its bound reference with our timed wrapper. install_pion_attention_patch
    # already pointed each module at `pion_scaled_dot_product_attention`; we
    # now point them at `timed_sdpa` (which itself calls the original via
    # `orig_sdpa = _patch.pion_scaled_dot_product_attention` captured above).
    import sys as _sys
    import mlx_lm.models.base as _base_mod
    _base_mod.scaled_dot_product_attention = timed_sdpa
    for _name, _mod in list(_sys.modules.items()):
        if not _name.startswith("mlx_lm.models."):
            continue
        if _name.endswith(".base"):
            continue
        if hasattr(_mod, "scaled_dot_product_attention") and \
                getattr(_mod, "scaled_dot_product_attention") is orig_sdpa:
            _mod.scaled_dot_product_attention = timed_sdpa

    # Warm decode.
    x = mx.array([[prefix_ids[-1]]])
    out = model(x, cache=cache); mx.eval(out)
    tok_id = int(mx.argmax(out[0, -1]).item())
    print("\nrunning decode steps (warm path)...")
    t_total = time.perf_counter()
    for step in range(args.decode_tokens):
        nxt = mx.array([[tok_id]])
        out = model(nxt, cache=cache); mx.eval(out)
        tok_id = int(mx.argmax(out[0, -1]).item())
    wall_total_ms = (time.perf_counter() - t_total) * 1000

    # Aggregate.
    print(f"\n{args.decode_tokens} decode steps over prefix={args.prefix_tokens}: total {wall_total_ms:.1f}ms\n")
    print(f"  {'category':<22}  {'calls':>6}  {'sum_ms':>10}  {'mean_ms':>10}  {'pct':>6}")
    print(f"  {'-' * 22}  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 6}")
    grand_sum = sum(sum(v) for v in layer_times_ns.values())
    for cat, samples in sorted(layer_times_ns.items()):
        s = sum(samples)
        n = len(samples)
        mean = s / n
        pct = (s / grand_sum * 100) if grand_sum else 0.0
        print(f"  {cat:<22}  {n:>6}  {s/1e6:>10.2f}  {mean/1e6:>10.3f}  {pct:>5.1f}%")
    print()

    # Step 2 decision hint — sum over BOTH pion-routed and shared kv buckets
    # for the same D (since shared layers inherit donor type and D).
    full_ns = sum(s for cat, samples in layer_times_ns.items()
                  if cat.startswith("full(D=512)") for s in samples)
    sliding_ns = sum(s for cat, samples in layer_times_ns.items()
                     if cat.startswith("sliding(D=256)") for s in samples)
    full_pct = full_ns / max(1, grand_sum) * 100
    sliding_pct = sliding_ns / max(1, grand_sum) * 100
    print(f"Step 2 decision hint  prefix={args.prefix_tokens}:")
    print(f"  full(D=512)    total wall-time = {full_pct:5.1f}%")
    print(f"  sliding(D=256) total wall-time = {sliding_pct:5.1f}%")
    print(f"  (full layers scale with N; sliding layers are bounded at window=512)")
    if full_pct >= 40:
        print("  → full layers dominate. Step 2a (D=512 PSO) is mandatory; 2b would leave most savings on the table.")
    elif full_pct >= 20:
        print("  → full layers material. Step 2a recommended; 2b is half-measure.")
    else:
        print("  → full layers minority at this prefix length. BUT they scale linearly with N "
              "while sliding stays at 512 — re-profile at 64K/128K to see crossover.")

    pc._mlx_prefix_kv.pop(ns, None)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--prefix-tokens", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=8)
    ap.add_argument("--prefill-chunk-size", type=int, default=None,
                    help="If set, cold prefill is chunked. Needed at 32K+ on M4 16GB.")
    args = ap.parse_args()
    sys.exit(main(args))
