#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# REPRODUCER -- backs a published number.
#
# Claim: Hybrid retrieval cache recall (gh #54/#75/#150).
#
# Requires: Apple Silicon + MLX + Llama-3.2-1B-Instruct-4bit. Needs a Pion server.
#
# This is research code, moved here from the private research tree so the
# number it produces can be checked. It was not written to be read; it was
# written to answer one question. Expect rough edges, and read the
# prerequisites above before running -- most of the cost is the model
# download, and a run on a loaded machine produces a wrong number rather
# than an error.
# ---------------------------------------------------------------------------
"""gh #75 Stage 1 sub-task 1 — hybrid retrieval recall benchmark.

Replays N RAG-shaped queries against three paths and reports TTFT +
answer-EM + token agreement vs the text-RAG baseline.

Dataset: SQuAD v2 validation (proxy for NQ-shaped single-passage RAG —
NQ-open doesn't include passages; SQuAD v2 has gold context + question
+ short answer, which is the right shape for testing HYBRID's K/V
caching effect in isolation from retrieval quality).

Paths:
  A) text-RAG baseline   — cold prefill (context + question), decode, score
  B) hybrid (inproc)     — ingest context as chunk (NOT timed), hydrate its
                           K/V and decode the question (both timed), score
  C) hybrid (pion lane)  — same but cross-process via pion-server --kvcache

For each query and each path: TTFT (ms) for the timed portion, and whether
the gold answer string appears in the decoded text. Hard gate per gh #75:
hybrid answer-found-rate ≥ 95% of baseline rate.

Usage:
  ./pion-server --kvcache --metal-attention -w 1     # for path C
  python3 benchmarks/reproducers/stage1_hybrid_recall_bench.py --n 100

  # Smoke test
  python3 benchmarks/reproducers/stage1_hybrid_recall_bench.py --n 5

Output: JSON to stage1_hybrid_results.json (--out to change).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parents[2]   # benchmarks/reproducers/<this> -> repo root
sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))
from pion_vllm_mlx import HybridRetrievalCache  # noqa: E402


PROMPT_TEMPLATE = "\n\nQuestion: {q}\nAnswer:"


def load_dataset_sample(n: int, seed: int = 0):
    """Return a list of (id, context, question, [answers]) tuples — only
    examples with non-empty answers, shuffled with fixed seed for repro."""
    from datasets import load_dataset
    import random
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    items = [(ex["id"], ex["context"], ex["question"], ex["answers"]["text"])
             for ex in ds if ex["answers"]["text"]]
    random.Random(seed).shuffle(items)
    return items[:n]


def greedy_decode(model, prompt_ids: List[int], n_steps: int, cache) -> tuple:
    """Returns (decoded_token_ids, ttft_ms).

    The first token is produced the way mlx_lm.generate_step does it: every
    prompt token but the last in 2,048-token chunks with only the cache state
    evaluated, then the last token alone. Until 2026-10-02 this evaluated one
    forward's logits at every prompt position, which no generation computes,
    and that made the text-RAG baseline too slow.
    """
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    done, n = 0, x.shape[1]
    while n - done > 1:
        step = min(2048, n - done - 1)
        model(x[:, done:done + step], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()       # as generate_step does after each prefill chunk
        done += step
    out = model(x[:, done:], cache=cache)
    mx.eval(out)
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


def warm_decode_via_hybrid(model, hr: HybridRetrievalCache, chunk_id: str,
                            suffix_ids: List[int], n_steps: int) -> tuple:
    """Hybrid path: K/V already ingested. Hydrate cache + forward suffix.
    Returns (decoded_token_ids, ttft_ms) — the hydration is inside the time to
    first token (until 2026-10-02 it was not, which flattered this side)."""
    t0 = time.perf_counter()
    cache, suffix = hr.prepare(chunk_id, suffix_ids)
    prepare_ms = (time.perf_counter() - t0) * 1000
    decoded, ttft = greedy_decode(model, suffix, n_steps, cache)
    return decoded, prepare_ms + ttft


def answer_found(decoded_text: str, gold_answers: List[str]) -> bool:
    """Substring match: gold answer appears (case-insensitive) in the decoded
    text. Models tend to wrap short answers in phrases ("The answer is X").
    Substring match captures the same intent as squad_v2's official EM but
    is forgiving of wrapping. (Strict EM is too tight for greedy decode.)"""
    decoded_lower = decoded_text.lower().strip()
    return any(g.lower().strip() in decoded_lower for g in gold_answers if g.strip())


def token_agreement(a: List[int], b: List[int]) -> float:
    """Fraction of positions where a[i] == b[i] over min(len(a), len(b))."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def run_baseline(model, tok, context: str, question: str, n_gen: int) -> dict:
    """Cold prefill of (context_ids + suffix_ids). Tokenize the two halves
    SEPARATELY then concat — matches the hybrid path's tokenization exactly,
    so token-agreement comparisons are apples-to-apples. (Joining the strings
    first then tokenizing can produce different BPE boundaries at the join.)"""
    chunk_ids = tok.encode(context)
    suffix_ids = tok.encode(PROMPT_TEMPLATE.format(q=question))
    prompt_ids = chunk_ids + suffix_ids
    cache = make_prompt_cache(model)
    decoded, ttft = greedy_decode(model, prompt_ids, n_gen, cache)
    text = tok.decode(decoded)
    del cache
    mx.clear_cache()
    return {"ttft_ms": ttft, "decoded": text, "decoded_tokens": decoded,
            "prompt_tokens": len(prompt_ids),
            "chunk_tokens": len(chunk_ids), "suffix_tokens": len(suffix_ids)}


def run_hybrid(model, tok, hr: HybridRetrievalCache, chunk_id: str,
                context: str, question: str, n_gen: int) -> dict:
    """Hybrid: ingest context (not timed if already in cache), warm-forward
    suffix. Returns dict with ttft (warm portion only) + decoded + tokens."""
    chunk_ids = tok.encode(context)
    if not hr.has(chunk_id):
        hr.ingest(chunk_id, chunk_ids)
    suffix_ids = tok.encode(PROMPT_TEMPLATE.format(q=question))
    decoded, ttft = warm_decode_via_hybrid(model, hr, chunk_id, suffix_ids, n_gen)
    text = tok.decode(decoded)
    return {"ttft_ms": ttft, "decoded": text, "decoded_tokens": decoded,
            "context_tokens": len(chunk_ids), "suffix_tokens": len(suffix_ids)}


def run_one(model, tok, idx: int, item: tuple, n_gen: int,
            hr_inproc: HybridRetrievalCache,
            hr_pion: Optional[HybridRetrievalCache]) -> dict:
    qid, context, question, answers = item
    base = run_baseline(model, tok, context, question, n_gen)
    inproc = run_hybrid(model, tok, hr_inproc, f"sq2_{qid}_inproc", context, question, n_gen)
    pion_res: Optional[dict] = None
    if hr_pion is not None:
        pion_res = run_hybrid(model, tok, hr_pion, f"sq2_{qid}_pion", context, question, n_gen)

    out = {
        "idx": idx, "id": qid, "gold_answers": answers,
        "baseline": {**base, "answer_found": answer_found(base["decoded"], answers)},
        "inproc":   {**inproc, "answer_found": answer_found(inproc["decoded"], answers),
                     "token_agreement_vs_baseline": token_agreement(inproc["decoded_tokens"], base["decoded_tokens"])},
    }
    if pion_res is not None:
        out["pion"] = {**pion_res, "answer_found": answer_found(pion_res["decoded"], answers),
                       "token_agreement_vs_baseline": token_agreement(pion_res["decoded_tokens"], base["decoded_tokens"])}
    return out


def aggregate(trials: List[dict]) -> dict:
    """Compute pass-rates + median TTFT + answer-found-rate per path."""
    def rate(path_key, field):
        vals = [t[path_key][field] for t in trials if path_key in t]
        return (sum(vals) / len(vals)) if vals else 0.0

    def p50_ms(path_key):
        vals = sorted(t[path_key]["ttft_ms"] for t in trials if path_key in t)
        return vals[len(vals)//2] if vals else 0.0

    out = {
        "n_trials": len(trials),
        "baseline": {
            "answer_found_rate": rate("baseline", "answer_found"),
            "ttft_p50_ms": p50_ms("baseline"),
        },
        "inproc": {
            "answer_found_rate": rate("inproc", "answer_found"),
            "ttft_p50_ms": p50_ms("inproc"),
            "token_agreement_mean": rate("inproc", "token_agreement_vs_baseline"),
        },
    }
    if any("pion" in t for t in trials):
        out["pion"] = {
            "answer_found_rate": rate("pion", "answer_found"),
            "ttft_p50_ms": p50_ms("pion"),
            "token_agreement_mean": rate("pion", "token_agreement_vs_baseline"),
        }

    # Gate evaluation: hybrid answer-found-rate ≥ 95% of baseline.
    b = out["baseline"]["answer_found_rate"]
    out["gate_threshold"] = 0.95
    out["gate"] = {}
    for key in ("inproc", "pion"):
        if key not in out:
            continue
        ratio = out[key]["answer_found_rate"] / b if b > 0 else 1.0
        out["gate"][key] = {"ratio_to_baseline": ratio,
                            "pass": ratio >= 0.95}
    return out


def main(args) -> int:
    print(f"gh #75 Stage 1 recall benchmark  n={args.n}  model={args.model}")
    print("loading dataset...")
    items = load_dataset_sample(args.n, seed=args.seed)
    print(f"  {len(items)} queries with non-empty answers")

    print("loading model (Llama-3.2-1B-Instruct-4bit)...")
    model, tok = mlx_load(args.model)

    hr_inproc = HybridRetrievalCache(model, backend="inproc")
    hr_pion: Optional[HybridRetrievalCache] = None
    if args.with_pion:
        print(f"connecting to pion-server at {args.pion_host}:{args.pion_port}...")
        try:
            hr_pion = HybridRetrievalCache(model, backend="pion",
                                            host=args.pion_host, port=args.pion_port)
        except Exception as e:
            print(f"  pion backend not reachable ({e}); skipping path C")
            hr_pion = None

    print("warmup pass (JIT compile)...")
    _ = run_baseline(model, tok, items[0][1][:200], items[0][2], 4)

    trials: List[dict] = []
    print(f"\nrunning {len(items)} trials...")
    t_total0 = time.perf_counter()
    for i, item in enumerate(items):
        try:
            trial = run_one(model, tok, i, item, args.n_gen, hr_inproc, hr_pion)
            trials.append(trial)
        except Exception as e:
            print(f"  trial {i} failed: {e}")
            continue
        if (i + 1) % args.progress == 0 or (i + 1) == len(items):
            elapsed = time.perf_counter() - t_total0
            rate_per_sec = (i + 1) / max(0.001, elapsed)
            eta = (len(items) - (i + 1)) / max(0.001, rate_per_sec)
            print(f"  {i+1:>3}/{len(items)}  ({elapsed:>5.0f}s, {rate_per_sec:.1f}/s, eta {eta:.0f}s)")

    summary = aggregate(trials)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"config": vars(args), "summary": summary, "trials": trials}, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")

    ok_inproc = summary.get("gate", {}).get("inproc", {}).get("pass", False)
    ok_pion = summary.get("gate", {}).get("pion", {}).get("pass", True)  # pass-by-default if skipped
    return 0 if (ok_inproc and ok_pion) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-gen", type=int, default=20)
    ap.add_argument("--with-pion", action="store_true",
                    help="Include path C (cross-process pion lane). Requires pion-server --kvcache.")
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--out", default="stage1_hybrid_results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=int, default=10)
    sys.exit(main(ap.parse_args()))
