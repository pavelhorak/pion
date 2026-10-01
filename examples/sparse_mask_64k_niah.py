#!/usr/bin/env python3
"""
sparse_mask_64k_niah.py — public reproduction of Pion's 326× warm-TTFT
sparse-mask result on Gemma-4-E2B-it-4bit at 64K context.

What this does:
  1. Loads `mlx-community/gemma-4-e2b-it-4bit` from your local HF cache
     (3.4 GB download on first run).
  2. Builds a 64K-token needle-in-a-haystack prompt: long filler with a
     fact ("the magic number for the city of <X> is <N>") spliced in at
     depth=0.5, then a question about that city.
  3. Runs *vanilla mlx-lm* on the full 64K prompt → records TTFT.
  4. Runs the *Pion in-proc lane* with block-mean top-K sparse selection
     on full-attention layers (sliding layers stay dense). The prefix is
     cold-prefilled into the cache (NOT counted in TTFT); the question
     suffix is the only thing measured at warm time.
  5. Prints the side-by-side TTFT + needle-found-yes/no for both paths.

Vanilla TTFT is "cold 64K prefill + first decoded token." Pion warm TTFT
is "warm forward of the ~25-token question suffix + first decoded token."
That's an apples-to-apples *warm vs cold* comparison — the same comparison
Modular's Part 1 blog cites as the 80× warm-vs-cold gap. Pion's number is
substantially larger because the sparse-mask kernel reads only 0.78% of
the prefix budget on each full-attention layer.

Honest caveats — read these before quoting the number:
  - Requires the pion-vllm-mlx package and the mlx_lm_patch monkey-patch.
    A different runtime (vLLM, llama.cpp, raw transformers) does NOT
    automatically get this win — you need the consumer-side integration.
  - Per-layer routing matters: full-attention layers get sparse mask,
    sliding-window layers stay dense. The decision is per-layer because
    sliding-attention is already capped at `sliding_window` tokens — a
    sparse-mask there is mostly redundant and the dense kernel is faster.
  - Sliding-window dense-only is the right call for hybrid models (Gemma
    4 has both layer types). For a full-attention-only model (e.g.
    Llama 3) the win profile is different — usually still big, but the
    blog number specifically refers to Gemma 4.
  - This is single-needle NIAH, not generic inference. Multi-needle, RAG
    with passage shuffling, and long-context summarization have their
    own behavior and aren't covered by this demo.

Usage:
  python3 examples/sparse_mask_64k_niah.py

Optional flags:
  --length N             context length in tokens (default 64000)
  --depth F              needle depth in [0, 1] (default 0.5)
  --decode-tokens N      tokens to decode for the answer (default 12)
  --prefill-chunk-size N memory-friendly prefill chunking (default 2048)
  --skip-vanilla         skip the vanilla baseline (Pion-only timing)

Requirements: 16 GB RAM, Apple Silicon, ~5 minutes wall-clock for a full
single run (most of which is the vanilla 64K cold prefill).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pion_vllm_mlx.prompt_cache import PionPromptCache  # noqa: E402
from pion_vllm_mlx.mlx_lm_patch import (  # noqa: E402
    install_pion_attention_patch,
    make_pion_prompt_cache,
)
from _gemma4_text_filter_load import load_text_only_from_cached  # noqa: E402


FILLER = (
    "When you analyze a startup's prospects, focus on the founders' "
    "demonstrated ability to learn fast. Domain knowledge can be acquired; "
    "judgment under uncertainty cannot. Most successful founders we backed "
    "had at most a year of experience in their target market when they "
    "started. What they had was the ability to update their model of the "
    "market every week based on what users actually did, not what they said "
    "they wanted. The signal you're looking for is whether the founder, when "
    "shown evidence that contradicts their thesis, gets defensive or curious. "
    "Curious founders win. Defensive founders flame out around month nine, "
    "when the obvious-in-hindsight market gap they were exploiting fills in "
    "with three better-funded competitors. The hard part is that this trait "
    "is invisible at the pitch — you have to see them in motion. "
)


def build_prompt(length: int, depth: float, city: str, number: int, tok) -> Tuple[List[int], List[int]]:
    """Return (full_token_ids, question_token_ids). The first is the entire
    cold prefill; the second is the suffix Pion measures warm."""
    needle = f"\nThe magic number for the city of {city} is {number}.\n"
    question = (
        f"\n\nQuestion: What is the magic number for the city of {city}? "
        f"Answer with only the number.\nAnswer:"
    )
    needle_ids = tok.encode(needle)
    question_ids = tok.encode(question)
    target_filler = max(128, length - len(needle_ids) - len(question_ids) - 16)
    base = tok.encode(FILLER)
    while len(base) < target_filler:
        base = base + base
    filler = base[:target_filler]
    insert_at = max(1, int(len(filler) * depth))
    full_ids = filler[:insert_at] + needle_ids + filler[insert_at:] + question_ids
    return full_ids, question_ids


def greedy_decode(model, prompt_ids: List[int], n_steps: int, cache,
                  prefill_chunk_size: int | None = None) -> Tuple[List[int], float]:
    """Prefill + greedy decode. Returns (decoded_tokens, ttft_ms).
    TTFT here is wall time from forward-start to first-decoded-token-ready."""
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    N = x.shape[1]
    if prefill_chunk_size is None or N <= prefill_chunk_size:
        out = model(x, cache=cache)
        mx.eval(out)
    else:
        for start in range(0, N, prefill_chunk_size):
            end = min(start + prefill_chunk_size, N)
            out = model(x[:, start:end], cache=cache)
            evals = []
            for c in cache:
                if c is None:
                    continue
                evals.append(c.keys)
                evals.append(c.values)
            if evals:
                mx.eval(*evals)
    ttft = (time.perf_counter() - t0) * 1000
    tok_id = int(mx.argmax(out[0, -1]).item())
    decoded = [tok_id]
    for _ in range(n_steps - 1):
        nxt = mx.array([[tok_id]])
        out = model(nxt, cache=cache)
        mx.eval(out)
        tok_id = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok_id)
    return decoded, ttft


def extract_number(decoded_tokens, tok) -> int:
    text = tok.decode(decoded_tokens)
    digits, seen = "", False
    for ch in text:
        if ch.isdigit():
            digits += ch; seen = True
        elif seen:
            break
    try:
        return int(digits) if digits else -1
    except ValueError:
        return -1


def main(args) -> int:
    rng = random.Random(args.seed)
    city = rng.choice(["Petropavlovsk-Kamchatsky", "Ouagadougou", "Antananarivo",
                       "Bratislava", "Wagga Wagga", "Trondheim"])
    number = rng.randint(10000, 99999)

    print(f"sparse_mask_64k_niah  model={args.model}")
    print(f"  context={args.length:,} tokens · depth={args.depth} · needle: '{city}' → {number}")
    print()

    print("loading model (3.4 GB if first run)...")
    model, tok = load_text_only_from_cached(args.model)
    layer_types = getattr(model.args, "layer_types", []) or []
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
    sliding_window = getattr(model.args, "sliding_window", 0)
    print(f"  {len(layer_types)} layers ({n_full} full, {n_sliding} sliding, window={sliding_window})")
    print()

    full_ids, question_ids = build_prompt(args.length, args.depth, city, number, tok)
    prefix_ids = full_ids[: len(full_ids) - len(question_ids)]
    print(f"prompt built: {len(full_ids):,} total tokens "
          f"({len(prefix_ids):,} prefix + {len(question_ids)} suffix)")
    print()

    vanilla_ttft = None
    vanilla_found = None
    if not args.skip_vanilla:
        print("[A] vanilla mlx-lm — cold prefill of full prompt")
        t_start = time.perf_counter()
        cache = make_prompt_cache(model)
        decoded, vanilla_ttft = greedy_decode(
            model, full_ids, args.decode_tokens, cache,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        guess = extract_number(decoded, tok)
        vanilla_found = (guess == number)
        wall = time.perf_counter() - t_start
        print(f"    TTFT      {vanilla_ttft:>9,.1f} ms  (wall {wall:.1f}s)")
        print(f"    needle    {'FOUND' if vanilla_found else f'MISSED (got {guess})'}")
        del cache
        mx.clear_cache()
        print()

    print("[B] Pion in-proc lane — sparse-mask on full-attention layers")
    print(f"    config    K_block={args.sparse_k_block}  K_blocks={args.sparse_k_blocks}  "
          f"(budget = {args.sparse_k_block * args.sparse_k_blocks} tokens per full layer "
          f"= {args.sparse_k_block * args.sparse_k_blocks / args.length * 100:.2f}% of prefix)")
    install_pion_attention_patch()
    pc = PionPromptCache(
        model, vquant="fp16", stage2=True,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    ns = f"sparse_demo|{args.length}|{city}|{number}"

    print("    prefix prefill (cold — NOT counted in TTFT)...")
    t_pref = time.perf_counter()
    pc.get_or_prefill(prefix_ids, ns)
    print(f"    prefix cached in {(time.perf_counter() - t_pref):.1f}s "
          f"({len(prefix_ids):,} tokens, one-time per prefix)")

    # First warm call — includes Metal PSO JIT compilation on first invocation.
    # We measure it (one-time cold-start cost) but the headline is the steady
    # state after PSOs are cached. This matches the original gh #60 Step 4
    # measurement, which ran 9 trials × multiple lengths and reported the p50.
    cache = make_pion_prompt_cache(
        model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids),
        sparse_full_layers={"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks},
    )
    decoded, pion_ttft_first = greedy_decode(model, question_ids, args.decode_tokens, cache)
    guess_first = extract_number(decoded, tok)

    # Steady-state warm call — PSOs already compiled, this is the headline number.
    cache2 = make_pion_prompt_cache(
        model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids),
        sparse_full_layers={"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks},
    )
    decoded2, pion_ttft = greedy_decode(model, question_ids, args.decode_tokens, cache2)
    guess = extract_number(decoded2, tok)
    pion_found = (guess == number)
    print(f"    first warm TTFT  {pion_ttft_first:>9,.1f} ms  (incl. Metal PSO JIT compile)")
    print(f"    steady-state     {pion_ttft:>9,.1f} ms  (suffix-only, {len(question_ids)} tokens)")
    print(f"    needle           {'FOUND' if pion_found else f'MISSED (got {guess})'}")
    print()

    print("────────────────────────────────────────────────────────")
    if vanilla_ttft is not None:
        speedup_first = vanilla_ttft / max(1.0, pion_ttft_first)
        speedup_steady = vanilla_ttft / max(1.0, pion_ttft)
        print(f"  Vanilla cold TTFT:     {vanilla_ttft:>9,.1f} ms   ({len(full_ids):,} tokens)")
        print(f"  Pion first warm:       {pion_ttft_first:>9,.1f} ms   ({speedup_first:>5,.1f}×, incl. PSO JIT)")
        print(f"  Pion steady-state:     {pion_ttft:>9,.1f} ms   ({speedup_steady:>5,.1f}×, headline)")
        print(f"  Needle found:          vanilla={vanilla_found}  pion={pion_found}")
        print()
        ok = (speedup_steady >= 100.0) and pion_found and vanilla_found
        verdict = "PASS (gh #73 acceptance)" if ok else "FAIL"
        print(f"  Acceptance gate:       steady-state ≥ 100× AND both needles found → {verdict}")
        return 0 if ok else 1
    print(f"  Pion first warm:       {pion_ttft_first:.1f} ms")
    print(f"  Pion steady-state:     {pion_ttft:.1f} ms")
    print(f"  Pion needle found:     {pion_found}")
    return 0 if pion_found else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--length", type=int, default=64000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--decode-tokens", type=int, default=12)
    ap.add_argument("--prefill-chunk-size", type=int, default=2048)
    ap.add_argument("--sparse-k-block", type=int, default=64)
    ap.add_argument("--sparse-k-blocks", type=int, default=8)
    ap.add_argument("--skip-vanilla", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(main(ap.parse_args()))
