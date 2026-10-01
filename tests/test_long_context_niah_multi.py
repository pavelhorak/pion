#!/usr/bin/env python3
"""gh #60 Phase 3 follow-up — multi-needle NIAH gate.

Hardens the Step 4 decision: single-needle NIAH (tests/test_long_context_niah.py)
showed sparse-mask holding 100% accuracy at 64K with 0.78% prefix budget, but
single-needle is the easy case for block-mean top-K selection — exactly ONE
block contains the magic number, and the selector just has to pick it.

Multi-needle is harder. N needles compete for the K_top block budget; if the
block-mean score doesn't sharply differentiate the target needle's block from
the others, sparse may pick the wrong needle's context and the model produces
a confident-but-wrong answer.

Setup per trial:
  - Insert N needles at evenly-spaced depths (0.5/N, 1.5/N, 2.5/N, ...).
  - Each needle: "The magic number for the city of {C_i} is {N_i}."
  - Ask about ONE specific needle: "What is the magic number for the city of {C_target}?"
  - Score EM on the numeric answer.

Comparison:
  [A] Vanilla mlx-lm (cold every request)        — accuracy baseline
  [B] Pion dense (in-proc, no sparse)            — confirms Pion in-proc preserves multi-needle
  [C] Pion sparse (K_block=64, K_blocks=8)       — the actual gate

Acceptance: Pion-sparse accuracy ≥ 95% of vanilla at 3-needle, 64K. If passes,
the 326× speedup story is robust beyond single-needle. If fails, the Phase 3
caveat (single-needle only) is real and the user-facing claim needs to be
tightened.

Requires: ./pion-server --kvcache -w 1 (for PionPromptCache construction).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
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
    "with three better-funded competitors. "
)

CITIES = [
    "Petropavlovsk-Kamchatsky", "Ouagadougou", "Antananarivo",
    "Bratislava", "Wagga Wagga", "Trondheim", "Mar del Plata",
    "Yogyakarta", "Reykjavik", "Tegucigalpa", "Asmara", "Vilnius",
]


@dataclass
class Trial:
    length: int
    needles: List[Tuple[str, int]]        # [(city, number), ...] — all N
    target_idx: int                       # which needle to ask about
    depths: List[float]                   # one per needle


def build_prompt(trial: Trial, tok) -> Tuple[List[int], int]:
    """Tile filler, splice in ALL needles at their depths, append the question
    asking only about the target needle. Returns (token_ids, target_number)."""
    needle_texts = [
        f"\nThe magic number for the city of {c} is {n}.\n"
        for (c, n) in trial.needles
    ]
    target_city = trial.needles[trial.target_idx][0]
    question = (
        f"\n\nQuestion: What is the magic number for the city of {target_city}? "
        f"Answer with only the number.\nAnswer:"
    )
    needle_token_lists = [tok.encode(nt) for nt in needle_texts]
    question_tokens = tok.encode(question)
    needles_total = sum(len(nt) for nt in needle_token_lists)
    target_filler_tokens = max(
        128, trial.length - needles_total - len(question_tokens) - 16
    )

    base_filler = tok.encode(FILLER)
    while len(base_filler) < target_filler_tokens:
        base_filler = base_filler + base_filler
    filler = base_filler[:target_filler_tokens]

    # Plant needles at requested depths, sorted ascending so insert positions
    # don't drift each other.
    plan = sorted(zip(trial.depths, needle_token_lists), key=lambda x: x[0])
    out: List[int] = []
    last_filler_pos = 0
    for depth, ntoks in plan:
        insert_at = max(1, int(len(filler) * depth))
        if insert_at < last_filler_pos:
            insert_at = last_filler_pos
        out.extend(filler[last_filler_pos:insert_at])
        out.extend(ntoks)
        last_filler_pos = insert_at
    out.extend(filler[last_filler_pos:])
    out.extend(question_tokens)
    return out, trial.needles[trial.target_idx][1]


def greedy_decode(model, prompt_ids: List[int], n_steps: int, cache,
                  prefill_chunk_size: int | None = None):
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    N = x.shape[1]
    if prefill_chunk_size is None or N <= prefill_chunk_size:
        out = model(x, cache=cache); mx.eval(out)
    else:
        for start in range(0, N, prefill_chunk_size):
            end = start + prefill_chunk_size if start + prefill_chunk_size < N else N
            out = model(x[:, start:end], cache=cache)
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
        out = model(nxt, cache=cache); mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
    return decoded, t_prefill


def extract_number(decoded_tokens, tok) -> int:
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


def make_trials(args, rng) -> List[Trial]:
    trials = []
    for length in args.lengths:
        for _ in range(args.trials_per_length):
            cities = rng.sample(CITIES, args.needles)
            numbers = [rng.randint(10000, 99999) for _ in range(args.needles)]
            # Evenly-spaced depths: 1/(N+1), 2/(N+1), ..., N/(N+1).
            depths = [(i + 1) / (args.needles + 1) for i in range(args.needles)]
            target = rng.randint(0, args.needles - 1)
            trials.append(Trial(length=length,
                                needles=list(zip(cities, numbers)),
                                target_idx=target,
                                depths=depths))
    return trials


def run_eval(model, tok, trials, args, label: str,
             cache_builder, sparse_full=None) -> dict:
    """Returns {length: (correct, total, ttft_list)}."""
    print(f"[{label}]")
    by_length = {L: [0, 0, []] for L in args.lengths}
    for ti, t in enumerate(trials):
        prompt_ids, target = build_prompt(t, tok)
        cache = cache_builder(ti, t, prompt_ids)
        decoded, ttft = greedy_decode(model, prompt_ids, args.decode_tokens, cache,
                                      prefill_chunk_size=args.prefill_chunk_size)
        guess = extract_number(decoded, tok)
        ok = (guess == target)
        by_length[t.length][0] += int(ok)
        by_length[t.length][1] += 1
        by_length[t.length][2].append(ttft)
        del cache
        mx.clear_cache()
    for L in args.lengths:
        c, n, tt = by_length[L]
        acc = c / max(1, n) * 100
        ttft_p50 = sorted(tt)[len(tt) // 2]
        print(f"   length={L:>5}  acc={acc:5.1f}% ({c}/{n})  TTFT p50={ttft_p50:7.1f}ms")
    print()
    return by_length


def main(args) -> int:
    print(f"gh #60 Phase 3 multi-needle NIAH  model={args.model}")
    print(f"  lengths={args.lengths}  needles={args.needles}  trials/length={args.trials_per_length}")
    print(f"  decode-tokens={args.decode_tokens}")
    print()

    print("loading model...")
    model, tok = load_text_only_from_cached(args.model)
    layer_types = getattr(model.args, "layer_types", []) or []
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_sliding = sum(1 for t in layer_types if t == "sliding_attention")
    sliding_window = getattr(model.args, "sliding_window", 0)
    print(f"  layers: {len(layer_types)} total, {n_full} full, {n_sliding} sliding (window={sliding_window})")
    print()

    rng = random.Random(args.seed)
    trials = make_trials(args, rng)
    print(f"total trials: {len(trials)}\n")

    # [A] vanilla
    def vanilla_builder(ti, t, prompt_ids):
        return make_prompt_cache(model)
    vanilla = run_eval(model, tok, trials, args, "A vanilla mlx-lm — cold every request",
                       vanilla_builder)

    # [B] Pion dense
    install_pion_attention_patch()
    pc = PionPromptCache(model, vquant="fp16", stage2=True,
                         prefill_chunk_size=args.prefill_chunk_size)
    def pion_dense_builder(ti, t, prompt_ids):
        # Use the question portion as suffix, the rest as prefix.
        target_city = t.needles[t.target_idx][0]
        question = (
            f"\n\nQuestion: What is the magic number for the city of {target_city}? "
            f"Answer with only the number.\nAnswer:"
        )
        qtoks = tok.encode(question)
        prefix_ids = prompt_ids[: len(prompt_ids) - len(qtoks)]
        ns = f"multi_niah|{ti}|L{t.length}|needles{args.needles}|target{t.target_idx}|dense"
        pc.get_or_prefill(prefix_ids, ns)
        cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc,
                                       prefix_len=len(prefix_ids))
        # Hand over the question portion as the "prompt" the model decodes —
        # greedy_decode prefills the suffix tokens then decodes.
        # But greedy_decode here takes the FULL prompt_ids; for the Pion path
        # we need to pass only the question. Workaround: swap prompt_ids on
        # the fly via a closure.
        return cache
    # The Pion path needs prompt_ids to be just the question; rewrite by
    # building a tiny shim list per trial.
    print("[B] Pion-patched mlx-lm DENSE (in-proc lane, stage2=True)")
    pion_dense_by_length = {L: [0, 0, []] for L in args.lengths}
    for ti, t in enumerate(trials):
        prompt_ids, target = build_prompt(t, tok)
        target_city = t.needles[t.target_idx][0]
        question = (
            f"\n\nQuestion: What is the magic number for the city of {target_city}? "
            f"Answer with only the number.\nAnswer:"
        )
        qtoks = tok.encode(question)
        prefix_ids = prompt_ids[: len(prompt_ids) - len(qtoks)]
        ns = f"multi_niah|{ti}|L{t.length}|needles{args.needles}|target{t.target_idx}|dense"
        pc.get_or_prefill(prefix_ids, ns)
        cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc,
                                       prefix_len=len(prefix_ids))
        decoded, ttft = greedy_decode(model, qtoks, args.decode_tokens, cache)
        guess = extract_number(decoded, tok)
        ok = (guess == target)
        pion_dense_by_length[t.length][0] += int(ok)
        pion_dense_by_length[t.length][1] += 1
        pion_dense_by_length[t.length][2].append(ttft)
        pc._mlx_prefix_kv.pop(ns, None)
        del cache
        mx.clear_cache()
    for L in args.lengths:
        c, n, tt = pion_dense_by_length[L]
        acc = c / max(1, n) * 100
        ttft_p50 = sorted(tt)[len(tt) // 2]
        print(f"   length={L:>5}  acc={acc:5.1f}% ({c}/{n})  TTFT p50={ttft_p50:7.1f}ms")
    print()

    # [C] Pion sparse
    print(f"[C] Pion-patched mlx-lm SPARSE  K_block={args.sparse_k_block} "
          f"K_blocks={args.sparse_k_blocks} "
          f"({args.sparse_k_block * args.sparse_k_blocks} tokens of prefix budget per full layer)")
    sparse_cfg = {"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks}
    pion_sparse_by_length = {L: [0, 0, []] for L in args.lengths}
    for ti, t in enumerate(trials):
        prompt_ids, target = build_prompt(t, tok)
        target_city = t.needles[t.target_idx][0]
        question = (
            f"\n\nQuestion: What is the magic number for the city of {target_city}? "
            f"Answer with only the number.\nAnswer:"
        )
        qtoks = tok.encode(question)
        prefix_ids = prompt_ids[: len(prompt_ids) - len(qtoks)]
        ns = f"multi_niah|{ti}|L{t.length}|needles{args.needles}|target{t.target_idx}|sparse"
        pc.get_or_prefill(prefix_ids, ns)
        cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc,
                                       prefix_len=len(prefix_ids),
                                       sparse_full_layers=sparse_cfg)
        decoded, ttft = greedy_decode(model, qtoks, args.decode_tokens, cache)
        guess = extract_number(decoded, tok)
        ok = (guess == target)
        pion_sparse_by_length[t.length][0] += int(ok)
        pion_sparse_by_length[t.length][1] += 1
        pion_sparse_by_length[t.length][2].append(ttft)
        pc._mlx_prefix_kv.pop(ns, None)
        del cache
        mx.clear_cache()
    for L in args.lengths:
        c, n, tt = pion_sparse_by_length[L]
        acc = c / max(1, n) * 100
        ttft_p50 = sorted(tt)[len(tt) // 2]
        print(f"   length={L:>5}  acc={acc:5.1f}% ({c}/{n})  TTFT p50={ttft_p50:7.1f}ms")
    print()

    # Gate
    print("──────── multi-needle gate ────────")
    overall_ok = True
    for L in args.lengths:
        vc, vn, _ = vanilla[L]
        pc_c, pc_n, _ = pion_sparse_by_length[L]
        v_acc = vc / max(1, vn)
        p_acc = pc_c / max(1, pc_n)
        ratio = (p_acc / v_acc) if v_acc > 0 else 1.0
        gate_ok = ratio >= args.threshold
        marker = "PASS" if gate_ok else "FAIL"
        print(f"  length={L:>5}  pion-sparse/vanilla = {ratio*100:5.1f}%  (gate ≥ {args.threshold*100:.0f}%)  {marker}")
        if not gate_ok:
            overall_ok = False
    print()
    print(f"OVERALL ({args.needles}-needle): {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                    default=[32768, 65536])
    ap.add_argument("--needles", type=int, default=3)
    ap.add_argument("--trials-per-length", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--sparse-k-block", type=int, default=64)
    ap.add_argument("--sparse-k-blocks", type=int, default=8)
    ap.add_argument("--prefill-chunk-size", type=int, default=2048)
    args = ap.parse_args()
    sys.exit(main(args))
