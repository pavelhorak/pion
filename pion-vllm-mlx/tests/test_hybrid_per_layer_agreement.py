#!/usr/bin/env python3
"""gh #60 Step 1 — token-agreement gate for hybrid (Gemma 4 / SWA) models.

The in-proc Stage-2 fast lane stashes per-layer prefix K/V from the post-prefill
cache and reuses them across warm forwards. For hybrid architectures with
sliding-window layers, the post-prefill cache holds the *full un-rotated*
prefix (RotatingKVCache only trims on the first decode step). Vanilla mlx-lm
applies the sliding window via the rotated cache + a None mask at decode time.
Our path concats [full_prefix | suffix] and runs SDPA with mask=None — for
sliding layers this attends to the whole prefix instead of the last `window`
tokens, producing incorrect logits.

This test asserts token-by-token agreement between vanilla mlx-lm and the
patched Pion-aware path for a 1-2 KTok prefix + N decode steps. With Step 1
applied (per-layer window slicing), agreement should be ≥ 99% on Gemma 4.
Without Step 1, sliding layers diverge as soon as the prefix exceeds the
training window (512 tokens).

Usage:
    python pion-vllm-mlx/tests/test_hybrid_per_layer_agreement.py
        [--model mlx-community/gemma-4-e2b-it-4bit]
        [--prefix-tokens 2048] [--decode-tokens 16]

No Pion server required — in-proc lane only (`stage2=True`, no `--kvcache`).
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
sys.path.insert(0, "tests")
from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx.mlx_lm_patch import (
    install_pion_attention_patch, make_pion_prompt_cache,
)
from _gemma4_text_filter_load import load_text_only_from_cached


def greedy_decode(model, prompt_ids, n_steps: int, cache):
    """Run prefill + n_steps greedy decode, return list of token ids."""
    out = model(prompt_ids, cache=cache)
    mx.eval(out)
    tok = int(mx.argmax(out[0, -1]).item())
    decoded = [tok]
    for _ in range(n_steps - 1):
        nxt = mx.array([[tok]])
        out = model(nxt, cache=cache)
        mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
    return decoded


def main(args) -> int:
    print(f"hybrid per-layer agreement  model={args.model}  prefix={args.prefix_tokens}  decode={args.decode_tokens}")

    print("loading model (text-only filter view of cached multimodal 4bit)...")
    model, tok = load_text_only_from_cached(args.model)

    layer_types = getattr(model.args, "layer_types", []) or []
    sliding_window = getattr(model.args, "sliding_window", 0)
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
    print(f"  layers: {len(layer_types)} total, {n_full} full, {n_sliding} sliding (window={sliding_window})")

    # Build a prefix that's long enough to expose the sliding-window bug.
    # Tokenize a paragraph and tile until we reach prefix_tokens.
    base = tok.encode(
        "The quick brown fox jumps over the lazy dog. " * 64
    )
    while len(base) < args.prefix_tokens:
        base = base + base
    prefix_ids = base[: args.prefix_tokens]
    suffix_ids = tok.encode(" Continue: ")

    full_ids = mx.array([prefix_ids + suffix_ids])

    # ── Vanilla baseline ───────────────────────────────────────────────────
    print("[A] vanilla mlx-lm — cold every request")
    cache_v = make_prompt_cache(model)
    vanilla_tokens = greedy_decode(model, full_ids, args.decode_tokens, cache_v)
    print(f"   vanilla: {vanilla_tokens[:8]}...")

    # ── Pion-patched (in-proc lane) ────────────────────────────────────────
    print("[B] Pion-patched (in-proc lane, stage2=True)")
    install_pion_attention_patch()
    pc = PionPromptCache(model, vquant="fp16", stage2=True)
    namespace = "test_hybrid_agreement_v1"
    # Cold prefill via PionPromptCache — stashes _mlx_prefix_kv for warm reuse.
    _ = pc.get_or_prefill(prefix_ids, namespace)
    # Warm decode: build a Pion-aware cache list, decode the suffix tokens.
    cache_p = make_pion_prompt_cache(
        model, namespace=namespace, prompt_cache=pc, prefix_len=len(prefix_ids),
    )
    suffix_arr = mx.array([suffix_ids])
    pion_tokens = greedy_decode(model, suffix_arr, args.decode_tokens, cache_p)
    print(f"   pion:    {pion_tokens[:8]}...")

    # ── Comparison ────────────────────────────────────────────────────────
    matches = sum(1 for a, b in zip(vanilla_tokens, pion_tokens) if a == b)
    agreement = matches / len(vanilla_tokens)
    print(f"\n  agreement: {agreement * 100:.1f}% ({matches}/{len(vanilla_tokens)})")
    threshold = args.threshold
    ok = agreement >= threshold
    print(f"  gate ≥ {threshold * 100:.0f}%: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--prefix-tokens", type=int, default=2048,
                    help="prefix length in tokens (must exceed sliding_window=512 to expose the bug)")
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()
    sys.exit(main(args))
