#!/usr/bin/env python3
"""Cross-process time to first token: a cold prefill vs a hit from a FRESH process.

Backs the "across processes" row of the README / docs Overview / one-pager
(2026-10-02, M4 Mac mini, Pion 0.9.1, mlx 0.32.3 / mlx-lm 0.31.3:
1,241.9 ms -> 73.9 ms, 16.8x at a 2,049-token prefix, Llama-3.2-1B-Instruct-4bit;
raw runs in results/cross_process_ttft_2026_10_02.json), its "same process" row
with --same (61.9 ms, 20.1x), and the prefix-length regime in "Where this does
not help": 1.50x at 34 tokens, 4.51x at 268, 11.24x at 1,035.

What each side measures, so the ratio means what it says:

  cold  a new process loads the model, then times the first token of
        prefix + question exactly as mlx-lm's generate_step produces it:
        every token but the last in 2,048-token chunks with only the cache
        evaluated, then the last token alone — nothing cached anywhere.
  hit   a new process loads the model, then times PionPromptCache.get_or_prefill
        (lookup + fetch over the wire + cache rebuild) and the question's
        first token through the same function.
  same  the process that stored the prefix asks again: Stage 2's in-process
        lane, where the prefix K/V it computed stayed resident as MLX arrays.
        Times the lookup, the cache it builds and the question's first token.
        The README's "same process" row; it is what an in-process prompt cache
        gives, and the only row here that needs no second process.

Until 2026-10-02 the cold side ran ONE forward over the whole prompt and
evaluated its logits — a vocabulary projection at all 2,065 positions, which no
generation computes. That made cold ~30% slow (1,624 vs 1,246 ms) and the ratio
~30% high: the 24x published from it (2026-09-23) re-measures at 17x.

Model load is outside both clocks. Each process first runs a short warm-up
forward so neither side pays Metal's one-time kernel compile. A single `store`
process prefills and registers the prefix once per length; cold and hit then
alternate so drift on a busy machine lands on both sides equally. Every hit
must report hits=1, misses=0, and the same first token as cold — the script
fails otherwise, because a "hit" that silently re-prefilled would be a
flattering wrong number.

Needs: Apple Silicon, mlx-lm, mlx-community/Llama-3.2-1B-Instruct-4bit, and a
running server:

    ./pion-server --kvcache -w 1                      # port 1974
    python3 benchmarks/reproducers/cross_process_ttft.py
    python3 benchmarks/reproducers/cross_process_ttft.py --prefix-tokens 34 268 1035 2049 --pairs 5 \
        --same --out cross_process_ttft.json

It is a timing measurement: on a loaded machine it produces a wrong number,
not an error. Close everything else first.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
PREFILL_STEP = 2048          # mlx_lm.generate_step's default prefill_step_size
QUESTION = " Question: what does the document say about the Eiffel Tower? Answer:"
FILLER = (
    "The Eiffel Tower is 330 metres tall and located in Paris. It was completed "
    "in 1889 for the World's Fair. The Empire State Building in New York City "
    "was completed in 1931 and stands 381 metres tall. The Golden Gate Bridge, "
    "completed in 1937 in San Francisco, has a main span of 1280 metres. "
)


def build_prefix_ids(tok, n_tokens: int) -> list[int]:
    """Exactly `n_tokens` token ids of a document-style system prompt."""
    text = "You are a careful assistant. Answer using only the document below.\n\nDocument: "
    ids = tok.encode(text, add_special_tokens=False)
    filler = tok.encode(FILLER, add_special_tokens=False)
    while len(ids) < n_tokens:
        ids += filler
    return ids[:n_tokens]


def child(mode: str, n_tokens: int, port: int) -> dict:
    """One measurement in THIS process (invoked as a subprocess by main)."""
    sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pion_vllm_mlx import PionPromptCache

    model, tok = load(MODEL)
    prefix = build_prefix_ids(tok, n_tokens)
    question = tok.encode(QUESTION, add_special_tokens=False)
    namespace = f"repro|cross_process_ttft|llama-3.2-1b-4bit|fp16|{n_tokens}"

    def first_token(cache, ids) -> int:
        # mlx_lm.generate_step's prefill: all but the last token, chunked, with
        # only the cache state evaluated; then the last token gives the logits.
        x = mx.array(ids)[None]
        done, n = 0, x.shape[1]
        while n - done > 1:
            step = min(PREFILL_STEP, n - done - 1)
            model(x[:, done:done + step], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()       # as generate_step does after each prefill chunk
            done += step
        tok_id = mx.argmax(model(x[:, done:], cache=cache)[0, -1])
        mx.eval(tok_id)
        return int(tok_id.item())

    first_token(make_prompt_cache(model), prefix[:64] + question)   # warm-up, untimed

    if mode == "same":
        # The in-process lane holds the prefix only in the process that
        # prefilled it, so this child stores under a namespace of its own and
        # then asks again: its timed request finds the prefix resident.
        from pion_vllm_mlx import install_pion_attention_patch, make_pion_prompt_cache
        install_pion_attention_patch()
        namespace = f"repro|cross_process_ttft|same|{n_tokens}|{os.getpid()}|{time.time_ns()}"
        pc = PionPromptCache(model, vquant="fp16", port=port, stage2=True)
        pc.get_or_prefill(prefix, namespace=namespace)        # MISS: prefill, store, keep resident
        t0 = time.perf_counter()
        pc.lookup(namespace)
        cache = make_pion_prompt_cache(model, namespace=namespace, prompt_cache=pc,
                                       prefix_len=len(prefix))
        tok_id = first_token(cache, question)
        ms = (time.perf_counter() - t0) * 1000
        return {"mode": mode, "prefix_tokens": len(prefix), "ms": ms, "first_token": tok_id,
                "hits": 1, "misses": 0, "stored_misses": pc.misses}

    pc = None if mode == "cold" else PionPromptCache(model, vquant="fp16", port=port)
    t0 = time.perf_counter()
    if mode == "cold":
        tok_id = first_token(make_prompt_cache(model), prefix + question)
    else:
        tok_id = first_token(pc.get_or_prefill(prefix, namespace=namespace), question)
    ms = (time.perf_counter() - t0) * 1000
    out = {"mode": mode, "prefix_tokens": len(prefix), "ms": ms, "first_token": tok_id}
    if pc is not None:
        out.update(hits=pc.hits, misses=pc.misses)
    return out


def run_child(mode: str, n_tokens: int, port: int) -> dict:
    r = subprocess.run([sys.executable, __file__, "--child", mode, str(n_tokens), "--port", str(port)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefix-tokens", type=int, nargs="+", default=[2049])
    ap.add_argument("--pairs", type=int, default=5, help="interleaved cold/hit pairs per length")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--same", action="store_true",
                    help="also time the same-process row (Stage 2's in-process lane)")
    ap.add_argument("--out", help="write the raw runs as JSON here")
    ap.add_argument("--child", nargs=2, metavar=("MODE", "TOKENS"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(child(args.child[0], int(args.child[1]), args.port)))
        return 0

    report = {"model": MODEL, "measured": time.strftime("%Y-%m-%d"), "lengths": []}
    print(f"{'prefix':>7}  {'cold':>10}  {'hit (fresh process)':>20}  {'ratio':>7}"
          + (f"  {'same process':>13}  {'ratio':>7}" if args.same else ""))
    for n in args.prefix_tokens:
        store = run_child("store", n, args.port)
        runs = []
        for _ in range(args.pairs):
            runs.append(run_child("cold", n, args.port))
            runs.append(run_child("hit", n, args.port))
            if args.same:
                runs.append(run_child("same", n, args.port))
        cold = [r["ms"] for r in runs if r["mode"] == "cold"]
        hit = [r for r in runs if r["mode"] == "hit"]
        same = [r for r in runs if r["mode"] == "same"]
        bad = [r for r in hit if (r["hits"], r["misses"]) != (1, 0)]
        bad += [r for r in same if r["stored_misses"] != 1]
        tokens = {r["first_token"] for r in runs} | {store["first_token"]}
        if bad or len(tokens) != 1:
            print(f"FAIL at {n} tokens: non-hits {bad} / first tokens {sorted(tokens)}", file=sys.stderr)
            return 1
        c, h = statistics.median(cold), statistics.median(r["ms"] for r in hit)
        row = {"prefix_tokens": n, "cold_median_ms": round(c, 1),
               "hit_median_ms": round(h, 1), "ratio": round(c / h, 2)}
        line = f"{n:>7}  {c:>8.1f}ms  {h:>18.1f}ms  {c / h:>6.2f}x"
        if same:
            sm = statistics.median(r["ms"] for r in same)
            row.update(same_median_ms=round(sm, 1), same_ratio=round(c / sm, 2))
            line += f"  {sm:>11.1f}ms  {c / sm:>6.2f}x"
        print(line)
        report["lengths"].append({**row, "store": store, "runs": runs})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"raw runs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
