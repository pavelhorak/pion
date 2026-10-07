#!/usr/bin/env python3
"""gh #62 (1.5b) — RULER subset gate for Pion sparse-mask attention.

Hardens the gh #60 / Step 4 story beyond simple NIAH by running the most
diagnostic RULER-style task for sparse-attention failure modes that
Gemma-4-E2B can actually solve at vanilla baseline:

  Multi-Value NIAH (MV): one key, M values scattered at distinct depths.
  Query asks for ALL values. **Sparse failure mode:** if any value's block
  isn't selected, that value is missing from the answer.

Sliding-window layers see only window=512 tokens; for the model to answer
correctly across a long prefix, the full-attention layers must surface ALL
M values' blocks via the sparse top-K selector. F1 < 1.0 means at least
one value was missed; F1 = 1.0 means the selector picked all M needles.

Variable Tracking (VT) — chained assignments `VAR X1 = root; VAR X2 = X1;
...` is the classic RULER multi-hop probe. An earlier note here said vanilla
Gemma-4-E2B-4bit gets 0% at 4K with 4-hop chains; on 2026-10-07 it solved
3/3 at 4K, and so did Pion dense and sparse
(benchmarks/results/2026-10-07-mac-m4/ruler_vt_4k.txt). Still opt-in with
`--include-vt`: it has not been run at 32K or 64K.

Acceptance: Pion-sparse F1 ≥ 95% of vanilla F1 on MV at every length tested.

Requires: ./pion-server --kvcache -w 1.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
sys.path.insert(0, "tests")
from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx.mlx_lm_patch import (
    install_pion_attention_patch, make_pion_prompt_cache,
)
from _gemma4_text_filter_load import load_text_only_from_cached
from _prompt_ids import bos, one_bos, piece


FILLER = (
    "When you analyze a startup's prospects, focus on the founders' "
    "demonstrated ability to learn fast. Domain knowledge can be acquired; "
    "judgment under uncertainty cannot. Most successful founders we backed "
    "had at most a year of experience in their target market when they "
    "started. What they had was the ability to update their model of the "
    "market every week based on what users actually did. The signal you're "
    "looking for is whether the founder gets defensive or curious when shown "
    "evidence that contradicts their thesis. "
)

CITIES = [
    "Petropavlovsk-Kamchatsky", "Ouagadougou", "Antananarivo",
    "Bratislava", "Wagga Wagga", "Trondheim", "Mar del Plata",
    "Yogyakarta", "Reykjavik", "Tegucigalpa", "Asmara", "Vilnius",
]


# ─── Variable Tracking ────────────────────────────────────────────────────


@dataclass
class VTTrial:
    length: int
    var_names: List[str]      # X1, X2, ..., XK+1
    root_value: int
    depths: List[float]


def build_vt_prompt(trial: VTTrial, tok) -> Tuple[List[int], int]:
    statements = []
    for i, name in enumerate(trial.var_names):
        if i == 0:
            statements.append(f"\nVAR {name} = {trial.root_value}.\n")
        else:
            statements.append(f"\nVAR {name} = {trial.var_names[i - 1]}.\n")

    question = vt_suffix(trial)
    stmt_tok_lists = [piece(tok, s) for s in statements]
    qtoks = piece(tok, question)
    stmt_total = sum(len(s) for s in stmt_tok_lists)
    target_filler_tokens = max(128, trial.length - len(bos(tok)) - stmt_total - len(qtoks) - 16)

    base_filler = piece(tok, FILLER)
    while len(base_filler) < target_filler_tokens:
        base_filler = base_filler + base_filler
    filler = base_filler[:target_filler_tokens]

    plan = sorted(zip(trial.depths, stmt_tok_lists), key=lambda x: x[0])
    out: List[int] = bos(tok)
    last = 0
    for depth, stmt in plan:
        pos = max(1, int(len(filler) * depth))
        if pos < last:
            pos = last
        out.extend(filler[last:pos])
        out.extend(stmt)
        last = pos
    out.extend(filler[last:])
    out.extend(qtoks)
    return one_bos(tok, out), trial.root_value


def vt_suffix(t: VTTrial) -> str:
    return (
        f"\n\nQuestion: What is the value of {t.var_names[-1]}? "
        f"Answer with only the number.\nAnswer:"
    )


def make_vt_trials(args, rng) -> List[VTTrial]:
    out = []
    for length in args.lengths:
        for _ in range(args.trials_per_length):
            chain = args.chain_length
            names = [f"X{i+1}" for i in range(chain + 1)]
            root = rng.randint(10000, 99999)
            depths = [(i + 1) / (chain + 2) for i in range(chain + 1)]
            out.append(VTTrial(length=length, var_names=names, root_value=root, depths=depths))
    return out


def score_vt(text: str, expected: int) -> bool:
    m = re.search(r"-?\d+", text)
    if not m:
        return False
    try:
        return int(m.group(0)) == expected
    except ValueError:
        return False


# ─── Multi-Value NIAH ─────────────────────────────────────────────────────


@dataclass
class MVTrial:
    length: int
    city: str
    values: List[int]
    depths: List[float]


def build_mv_prompt(trial: MVTrial, tok) -> Tuple[List[int], List[int]]:
    ordinals = ["first", "second", "third", "fourth", "fifth"]
    statements = []
    for i, v in enumerate(trial.values):
        ord_str = ordinals[i] if i < len(ordinals) else f"{i+1}th"
        statements.append(
            f"\nThe {ord_str} magic number for the city of {trial.city} is {v}.\n"
        )
    question = mv_suffix(trial)
    stmt_tok_lists = [piece(tok, s) for s in statements]
    qtoks = piece(tok, question)
    stmt_total = sum(len(s) for s in stmt_tok_lists)
    target_filler_tokens = max(128, trial.length - len(bos(tok)) - stmt_total - len(qtoks) - 16)

    base_filler = piece(tok, FILLER)
    while len(base_filler) < target_filler_tokens:
        base_filler = base_filler + base_filler
    filler = base_filler[:target_filler_tokens]

    plan = sorted(zip(trial.depths, stmt_tok_lists), key=lambda x: x[0])
    out: List[int] = bos(tok)
    last = 0
    for depth, stmt in plan:
        pos = max(1, int(len(filler) * depth))
        if pos < last:
            pos = last
        out.extend(filler[last:pos])
        out.extend(stmt)
        last = pos
    out.extend(filler[last:])
    out.extend(qtoks)
    return one_bos(tok, out), list(trial.values)


def mv_suffix(t: MVTrial) -> str:
    return (
        f"\n\nQuestion: List ALL the magic numbers for the city of {t.city}, "
        f"comma-separated.\nAnswer:"
    )


def make_mv_trials(args, rng) -> List[MVTrial]:
    out = []
    for length in args.lengths:
        for _ in range(args.trials_per_length):
            city = rng.choice(CITIES)
            M = args.mv_values
            values = [rng.randint(10000, 99999) for _ in range(M)]
            depths = [(i + 1) / (M + 1) for i in range(M)]
            out.append(MVTrial(length=length, city=city, values=values, depths=depths))
    return out


def score_mv(text: str, expected: List[int]) -> Tuple[bool, float]:
    found = re.findall(r"\d{4,6}", text)
    found_set = set(int(x) for x in found if x.isdigit())
    expected_set = set(expected)
    if not expected_set:
        return False, 0.0
    tp = len(found_set & expected_set)
    if tp == 0:
        return False, 0.0
    prec = tp / max(1, len(found_set))
    rec  = tp / len(expected_set)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return (expected_set.issubset(found_set), f1)


# ─── Driver ───────────────────────────────────────────────────────────────


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


def run_task(model, tok, trials, args, label, mode, pc, sparse_cfg,
             build_prompt_fn, suffix_fn, score_fn, task_name):
    """Generic runner — mode in {"vanilla", "dense", "sparse"}."""
    print(f"[{label}]  {task_name}")
    # results: by length → [correct, total, ttft_list, extra_metric_list]
    by_length = {L: [0, 0, [], []] for L in args.lengths}
    for ti, t in enumerate(trials):
        prompt_ids, expected = build_prompt_fn(t, tok)
        if mode == "vanilla":
            cache = make_prompt_cache(model)
            decode_input = prompt_ids
        else:
            qtoks = piece(tok, suffix_fn(t))
            prefix_ids = prompt_ids[: len(prompt_ids) - len(qtoks)]
            ns = f"ruler|{ti}|L{t.length}|{id(t)}|{mode}"
            pc.get_or_prefill(prefix_ids, ns)
            cfg = sparse_cfg if mode == "sparse" else None
            cache = make_pion_prompt_cache(
                model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids),
                sparse_full_layers=cfg,
            )
            decode_input = qtoks
        decoded, ttft = greedy_decode(model, decode_input, args.decode_tokens, cache,
                                      prefill_chunk_size=args.prefill_chunk_size)
        text = tok.decode(decoded)
        result = score_fn(text, expected)
        ok = result if isinstance(result, bool) else result[0]
        f1 = None if isinstance(result, bool) else result[1]
        by_length[t.length][0] += int(ok)
        by_length[t.length][1] += 1
        by_length[t.length][2].append(ttft)
        if f1 is not None:
            by_length[t.length][3].append(f1)
        if mode != "vanilla":
            pc._mlx_prefix_kv.pop(ns, None)
        del cache
        mx.clear_cache()
    for L in args.lengths:
        c, n, tt, f1s = by_length[L]
        acc = c / max(1, n) * 100
        ttft_p50 = sorted(tt)[len(tt) // 2]
        extra = f"  F1={sum(f1s)/max(1,len(f1s)):.3f}" if f1s else ""
        print(f"   length={L:>5}  acc={acc:5.1f}% ({c}/{n}){extra}  TTFT p50={ttft_p50:7.1f}ms")
    print()
    return by_length


def main(args) -> int:
    print(f"gh #62 (1.5b) RULER-subset gate  model={args.model}")
    print(f"  lengths={args.lengths}  trials/length={args.trials_per_length}")
    print(f"  VT chain={args.chain_length}  MV M={args.mv_values}  "
          f"sparse K_block={args.sparse_k_block} K_blocks={args.sparse_k_blocks} "
          f"(budget={args.sparse_k_block*args.sparse_k_blocks})")
    print()

    print("loading model...")
    model, tok = load_text_only_from_cached(args.model)
    layer_types = getattr(model.args, "layer_types", []) or []
    print(f"  layers: {len(layer_types)} total, "
          f"{sum(1 for t in layer_types if t == 'full_attention')} full, "
          f"{sum(1 for t in layer_types if t == 'sliding_attention')} sliding")
    print()

    rng = random.Random(args.seed)
    mv_trials = make_mv_trials(args, rng)
    vt_trials = make_vt_trials(args, rng) if args.include_vt else []
    print(f"MV trials: {len(mv_trials)}" + (f"   VT trials: {len(vt_trials)}" if vt_trials else "") + "\n")

    install_pion_attention_patch()
    pc = PionPromptCache(model, vquant="fp16", stage2=True,
                         prefill_chunk_size=args.prefill_chunk_size)
    sparse_cfg = {"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks}

    print(f"━━━ Task: Multi-Value NIAH (M={args.mv_values}) ━━━\n")
    mv_vanilla = run_task(model, tok, mv_trials, args, "A vanilla mlx-lm (cold)", "vanilla",
                          pc, sparse_cfg, build_mv_prompt, mv_suffix, score_mv, "MV")
    mv_dense   = run_task(model, tok, mv_trials, args, "B Pion in-proc DENSE",  "dense",
                          pc, sparse_cfg, build_mv_prompt, mv_suffix, score_mv, "MV")
    mv_sparse  = run_task(model, tok, mv_trials, args, "C Pion in-proc SPARSE", "sparse",
                          pc, sparse_cfg, build_mv_prompt, mv_suffix, score_mv, "MV")

    vt_results = None
    if vt_trials:
        print(f"━━━ Task: Variable Tracking (chain={args.chain_length}) ━━━\n")
        vt_vanilla = run_task(model, tok, vt_trials, args, "A vanilla mlx-lm (cold)", "vanilla",
                              pc, sparse_cfg, build_vt_prompt, vt_suffix, score_vt, "VT")
        vt_dense   = run_task(model, tok, vt_trials, args, "B Pion in-proc DENSE",  "dense",
                              pc, sparse_cfg, build_vt_prompt, vt_suffix, score_vt, "VT")
        vt_sparse  = run_task(model, tok, vt_trials, args, "C Pion in-proc SPARSE", "sparse",
                              pc, sparse_cfg, build_vt_prompt, vt_suffix, score_vt, "VT")
        vt_results = (vt_vanilla, vt_sparse)

    print("━━━ Gate (MV — F1-based; EM is too strict at 32-64 decode tokens) ━━━")
    overall = True
    for L in args.lengths:
        v_f1s = mv_vanilla[L][3]
        p_f1s = mv_sparse[L][3]
        v_f1 = sum(v_f1s) / max(1, len(v_f1s))
        p_f1 = sum(p_f1s) / max(1, len(p_f1s))
        ratio = (p_f1 / v_f1) if v_f1 > 0 else 1.0
        gate_ok = ratio >= args.threshold
        mark = "PASS" if gate_ok else "FAIL"
        print(f"  MV  length={L:>5}  "
              f"sparse F1={p_f1:.3f}  vanilla F1={v_f1:.3f}  "
              f"ratio = {ratio*100:5.1f}%  (gate ≥ {args.threshold*100:.0f}%)  {mark}")
        if not gate_ok:
            overall = False
    if vt_results:
        vt_vanilla, vt_sparse = vt_results
        print("━━━ Gate (VT — EM) ━━━")
        for L in args.lengths:
            vc, vn = vt_vanilla[L][0], vt_vanilla[L][1]
            pc_c, pc_n = vt_sparse[L][0], vt_sparse[L][1]
            v_acc = vc / max(1, vn)
            p_acc = pc_c / max(1, pc_n)
            ratio = (p_acc / v_acc) if v_acc > 0 else 1.0
            gate_ok = ratio >= args.threshold
            mark = "PASS" if gate_ok else "FAIL"
            print(f"  VT  length={L:>5}  "
                  f"sparse/vanilla EM = {ratio*100:5.1f}%  (gate ≥ {args.threshold*100:.0f}%)  {mark}")
            if not gate_ok:
                overall = False
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e2b-it-4bit")
    ap.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                    default=[32768, 65536])
    ap.add_argument("--trials-per-length", type=int, default=3)
    ap.add_argument("--include-vt", action="store_true",
                    help="Also run Variable Tracking. Gemma-4-E2B fails vanilla at 4K; "
                         "enable only for ≥7B models.")
    ap.add_argument("--chain-length", type=int, default=4)
    ap.add_argument("--mv-values", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=64,
                    help="MV needs room for `N1, N2, N3` answer plus model preamble.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--sparse-k-block", type=int, default=64)
    ap.add_argument("--sparse-k-blocks", type=int, default=8)
    ap.add_argument("--prefill-chunk-size", type=int, default=2048)
    args = ap.parse_args()
    sys.exit(main(args))
