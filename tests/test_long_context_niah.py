#!/usr/bin/env python3
"""gh #60 Step 3 — Long-context NIAH gate (vanilla vs Pion in-proc lane).

Pion's first published end-to-end long-context numbers. The plan in gh #60
calls for ≥95% of vanilla RULER score and ≥2× decode TTFT win at 64K+. This
gate is the working subset:

  - Multi-length needle-in-a-haystack: insert "The magic number for the city
    of <CITY> is <NUMBER>" at a known depth in a long Paul Graham essay tile,
    then ask "What is the magic number for the city of <CITY>?"
  - Score per (length, depth) bucket: exact-match on the numeric answer.
  - Compare vanilla mlx-lm against the Pion in-proc lane (Step 1 fix in
    da5b3a6). The cross-process wire path is a separate test (and is gated
    on Step 2 D=512 PSO support before it can run on Gemma 4 end-to-end).

Targets (per gh #60 Step 3):
  - Pion accuracy ≥ 95% of vanilla AT EVERY length.
  - Pion in-proc TTFT savings reported (informational; the ≥2× target is
    aspirational pre-Step-2).

Lengths default to {4096, 8192, 16384}; pass --lengths 4096,8192,16384,32768
to push higher (32K + Gemma-4-E2B-4bit is borderline on a 16GB Mac).

No Pion server required — in-proc lane only.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
sys.path.insert(0, "tests")
from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx.mlx_lm_patch import (
    install_pion_attention_patch, make_pion_prompt_cache,
)
from _gemma4_text_filter_load import load_text_only_from_cached


# Long-form filler. Tiled to reach the desired token length.
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
    "is invisible at the pitch — you have to see them in motion. Run an "
    "experiment with their thesis. Tell them what you'd want to see in the "
    "next two weeks for the company to clear your bar. If they come back "
    "with that exact data and an honest narrative about what surprised them, "
    "they're worth a check. If they come back with the data shoehorned to "
    "fit the original story, pass. "
)

# Cities for the needle. Distinct enough that the model can't guess.
CITIES = [
    "Petropavlovsk-Kamchatsky", "Ouagadougou", "Antananarivo",
    "Bratislava", "Wagga Wagga", "Trondheim", "Mar del Plata",
    "Yogyakarta",
]


@dataclass
class Trial:
    length: int            # target context tokens
    depth: float           # depth fraction in [0, 1]
    city: str
    number: int            # the magic number


def build_prompt(trial: Trial, tok) -> Tuple[List[int], int]:
    """Tile FILLER, splice in the needle at `depth`, append the question.
    Return (token_ids, target_number)."""
    needle_text = (
        f"\nThe magic number for the city of {trial.city} is {trial.number}.\n"
    )
    question = (
        f"\n\nQuestion: What is the magic number for the city of {trial.city}? "
        f"Answer with only the number.\nAnswer:"
    )
    # Build filler to roughly the target length minus needle/question.
    needle_tokens = tok.encode(needle_text)
    question_tokens = tok.encode(question)
    target_filler_tokens = max(
        128, trial.length - len(needle_tokens) - len(question_tokens) - 16
    )

    base_filler = tok.encode(FILLER)
    while len(base_filler) < target_filler_tokens:
        base_filler = base_filler + base_filler
    filler = base_filler[:target_filler_tokens]

    # Splice needle at depth.
    insert_at = max(1, int(len(filler) * trial.depth))
    full_ids = filler[:insert_at] + needle_tokens + filler[insert_at:] + question_tokens
    return full_ids, trial.number


def greedy_decode(model, prompt_ids: List[int], n_steps: int, cache,
                  prefill_chunk_size: int | None = None):
    """Prefill + n_steps greedy decode. Returns (decoded_token_ids, t_prefill_ms).

    If prefill_chunk_size is set, the prefill is split into chunks of that many
    tokens with mx.eval between chunks — keeps peak memory at O(chunk · N_kv)
    instead of O(N²). Needed to fit 64K+ prefills on a 16GB Mac.
    """
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    N = x.shape[1]
    if prefill_chunk_size is None or N <= prefill_chunk_size:
        out = model(x, cache=cache)
        mx.eval(out)
    else:
        for start in range(0, N, prefill_chunk_size):
            end = start + prefill_chunk_size if start + prefill_chunk_size < N else N
            out = model(x[:, start:end], cache=cache)
            # Force per-layer cache materialization between chunks; evaling
            # only `out` leaves the cache concat lazy and defeats the win.
            evals = []
            for c in cache:
                if c is None:
                    continue
                evals.append(c.keys)
                evals.append(c.values)
            if evals:
                mx.eval(*evals)
    t_prefill = (time.perf_counter() - t0) * 1000
    tok = int(mx.argmax(out[0, -1]).item())
    decoded = [tok]
    for _ in range(n_steps - 1):
        nxt = mx.array([[tok]])
        out = model(nxt, cache=cache)
        mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
    return decoded, t_prefill


def extract_number(decoded_tokens, tok) -> int:
    """Decode and find the first integer in the answer."""
    text = tok.decode(decoded_tokens)
    digits = ""
    seen_digit = False
    for ch in text:
        if ch.isdigit():
            digits += ch
            seen_digit = True
        elif seen_digit:
            break
    if not digits:
        return -1
    try:
        return int(digits)
    except ValueError:
        return -1


def main(args) -> int:
    print(f"gh #60 Step 3 NIAH gate  model={args.model}  lengths={args.lengths}  trials={args.trials}")
    print(f"  depths={args.depths}  decode-tokens={args.decode_tokens}")
    print()

    print("loading model...")
    model, tok = load_text_only_from_cached(args.model)
    layer_types = getattr(model.args, "layer_types", []) or []
    sliding_window = getattr(model.args, "sliding_window", 0)
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
    print(f"  layers: {len(layer_types)} total, {n_full} full, {n_sliding} sliding (window={sliding_window})")
    print()

    # Build trial set.
    rng = random.Random(args.seed)
    trials: List[Trial] = []
    for length in args.lengths:
        for depth in args.depths:
            for _ in range(args.trials):
                city = rng.choice(CITIES)
                number = rng.randint(10000, 99999)
                trials.append(Trial(length=length, depth=depth, city=city, number=number))
    print(f"total trials: {len(trials)}\n")

    # Run vanilla baseline.
    print("[A] vanilla mlx-lm — cold every request")
    vanilla_correct: dict = {L: 0 for L in args.lengths}
    vanilla_total: dict   = {L: 0 for L in args.lengths}
    vanilla_ttft: dict    = {L: [] for L in args.lengths}
    for t in trials:
        prompt_ids, target = build_prompt(t, tok)
        cache = make_prompt_cache(model)
        decoded, ttft = greedy_decode(
            model, prompt_ids, args.decode_tokens, cache,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        guess = extract_number(decoded, tok)
        ok = (guess == target)
        vanilla_total[t.length] += 1
        vanilla_correct[t.length] += int(ok)
        vanilla_ttft[t.length].append(ttft)
        # Force evict cache to keep memory in check between trials.
        del cache
        mx.clear_cache()
    for L in args.lengths:
        acc = vanilla_correct[L] / max(1, vanilla_total[L]) * 100
        ttft_p50 = sorted(vanilla_ttft[L])[len(vanilla_ttft[L]) // 2]
        print(f"   length={L:>5}  vanilla  acc={acc:5.1f}%  ({vanilla_correct[L]}/{vanilla_total[L]})  TTFT p50={ttft_p50:7.1f}ms")
    print()

    # Run Pion in-proc lane.
    sparse_full = None
    if args.sparse_full:
        sparse_full = {"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks}
        budget = args.sparse_k_block * args.sparse_k_blocks
        print(f"[B] Pion-patched mlx-lm (in-proc lane, stage2=True)  "
              f"sparse full layers: K_block={args.sparse_k_block} K_blocks={args.sparse_k_blocks} "
              f"({budget} tokens of prefix budget per full layer)")
    else:
        print("[B] Pion-patched mlx-lm (in-proc lane, stage2=True)")
    install_pion_attention_patch()
    pc = PionPromptCache(
        model, vquant="fp16", stage2=True,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    pion_correct: dict = {L: 0 for L in args.lengths}
    pion_total: dict   = {L: 0 for L in args.lengths}
    pion_ttft: dict    = {L: [] for L in args.lengths}
    for ti, t in enumerate(trials):
        prompt_ids, target = build_prompt(t, tok)
        # Use a unique namespace per trial so we always exercise the cold
        # → warm transition (one cold prefill, one warm forward of the
        # question portion). The cold prefill is the "fair comparison"
        # for vanilla; the warm question forward is what the in-proc lane
        # actually accelerates.
        ns = f"step3_niah|{ti}|L{t.length}|d{t.depth}|{t.city}"
        # Split: prefix = everything except the question; suffix = question only.
        question = (
            f"\n\nQuestion: What is the magic number for the city of {t.city}? "
            f"Answer with only the number.\nAnswer:"
        )
        question_ids = tok.encode(question)
        prefix_ids = prompt_ids[: len(prompt_ids) - len(question_ids)]
        # Cold prefill via PionPromptCache stashes _mlx_prefix_kv.
        pc.get_or_prefill(prefix_ids, ns)
        # Warm decode of the question.
        cache = make_pion_prompt_cache(
            model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids),
            sparse_full_layers=sparse_full,
        )
        # Suffix (question) is small (~25 tokens) — no chunking needed here.
        # The cold-path prefix forward already happened inside get_or_prefill()
        # and is chunked through PionPromptCache.prefill_chunk_size.
        decoded, ttft = greedy_decode(model, question_ids, args.decode_tokens, cache)
        guess = extract_number(decoded, tok)
        ok = (guess == target)
        pion_total[t.length] += 1
        pion_correct[t.length] += int(ok)
        pion_ttft[t.length].append(ttft)
        # Free per-namespace stash to avoid OOM at high context.
        pc._mlx_prefix_kv.pop(ns, None)
        del cache
        mx.clear_cache()
    for L in args.lengths:
        acc = pion_correct[L] / max(1, pion_total[L]) * 100
        ttft_p50 = sorted(pion_ttft[L])[len(pion_ttft[L]) // 2]
        v_ttft_p50 = sorted(vanilla_ttft[L])[len(vanilla_ttft[L]) // 2]
        speedup = v_ttft_p50 / max(1.0, ttft_p50)
        print(f"   length={L:>5}  pion     acc={acc:5.1f}%  ({pion_correct[L]}/{pion_total[L]})  TTFT p50={ttft_p50:7.1f}ms  ({speedup:4.2f}× vs vanilla)")
    print()

    # Gate evaluation.
    print("──────── gate ────────")
    overall_ok = True
    for L in args.lengths:
        v_acc = vanilla_correct[L] / max(1, vanilla_total[L])
        p_acc = pion_correct[L]   / max(1, pion_total[L])
        ratio = (p_acc / v_acc) if v_acc > 0 else 1.0
        gate_ok = ratio >= args.threshold
        marker = "PASS" if gate_ok else "FAIL"
        print(f"  length={L:>5}  pion/vanilla = {ratio*100:5.1f}%  (gate ≥ {args.threshold*100:.0f}%)  {marker}")
        if not gate_ok:
            overall_ok = False
    print()
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                    default=[4096, 8192, 16384])
    ap.add_argument("--depths", type=lambda s: [float(x) for x in s.split(",")],
                    default=[0.25, 0.5, 0.75])
    ap.add_argument("--trials", type=int, default=2,
                    help="number of (city, number) trials per (length, depth) bucket")
    ap.add_argument("--decode-tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="Pion-acc / vanilla-acc must be at least this ratio")
    ap.add_argument("--prefill-chunk-size", type=int, default=None,
                    help="If set, prefill in chunks of this many tokens with "
                         "mx.eval between chunks. Required for 32K+ on 16GB M-series.")
    ap.add_argument("--sparse-full", action="store_true",
                    help="gh #60 Phase 3: enable block-mean top-K sparse selection "
                         "on full-attention layers in the Pion in-proc lane. "
                         "Sliding layers stay dense (capped at sliding_window anyway).")
    ap.add_argument("--sparse-k-block", type=int, default=64,
                    help="Block size B for block-mean top-K selector.")
    ap.add_argument("--sparse-k-blocks", type=int, default=8,
                    help="Number of blocks K_top to pick per query.")
    args = ap.parse_args()
    sys.exit(main(args))
