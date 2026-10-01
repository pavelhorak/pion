#!/usr/bin/env python3
"""3B BLEU strict-PASS attempt via Stage-2 attend_query.

§28.4 root-caused the 3B BLEU drop (0.911 vs 1B's 0.969) to mlx-lm's own
greedy-decode numerical depth across 28 layers — Pion's wire path is
bit-perfect for fp16 input. The escape hatch is Stage-2: the cache stays
in MLX-resident memory, so there's no client-side cache rebuild step
that would propagate fp16-cast precision loss.

This bench compares THREE generation paths on Llama-3.2-3B-Instruct-4bit
and reports BLEU + first-token agreement against vanilla:

  A. Vanilla mlx-lm (the reference).
  B. PionPromptCache cache-rebuild (path that §28.4 measured at 0.911).
  C. install_pion_attention_patch + Stage-2 monkey-patch (this session).

Hypothesis: C ≥ A on agreement (no cache rebuild, no quantization loss in
the inference path) and C ≥ 0.95 BLEU vs A.

Run:
  ./pion-server --kvcache -w 1
  .pixi/envs/default/bin/python tests/bench_bleu_3b_stage2.py [--max-tokens 64]
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--start", action="store_true")
    p.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    p.add_argument("--prompts", default="tests:bench_bleu_prompts",
                   help="Comma list or 'tests:bench_bleu_prompts' for built-in.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--vquant", default="fp16")
    return p.parse_args()


# Built-in prompts: short, medium, long — common LLM workloads.
BUILTIN_PROMPTS = [
    "The capital of France is",
    "Write a haiku about distributed systems:",
    "Summarize the following in one sentence: A KV cache is a memory of attention keys and values from previous tokens, used to avoid recomputing them on each generation step.",
    "Explain why prefix sharing helps LLM inference:",
    "What's 2+2? Answer with just the number.",
]


def start_pion(port, log_path):
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")  # gh #429
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 60
    import socket
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(3)
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def stop_pion(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def bleu_simple(ref_tokens, cand_tokens, max_n=4):
    """Simple corpus-style BLEU: brevity penalty × geometric mean of n-gram
    precisions for n=1..max_n. Smoothed by adding 1 to numerator+denominator
    to avoid log(0)."""
    import math
    if not cand_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        cand_ngrams = Counter(tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1))
        match = 0
        total = 0
        for ng, cnt in cand_ngrams.items():
            match += min(cnt, ref_ngrams.get(ng, 0))
            total += cnt
        # +1 smoothing (Lin & Och)
        precisions.append((match + 1) / (total + 1))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(cand_tokens))
    return bp * geo


def gen_vanilla(model, tok, prompt_ids, max_tokens):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    x = mx.array([prompt_ids])
    logits = model(x, cache=cache)
    out = []
    next_id = int(mx.argmax(logits[0, -1]))
    out.append(next_id)
    for _ in range(max_tokens - 1):
        x = mx.array([[next_id]])
        logits = model(x, cache=cache)
        next_id = int(mx.argmax(logits[0, -1]))
        out.append(next_id)
    return out


def gen_path_b(model, tok, prompt_ids, max_tokens, namespace, port, vquant):
    import mlx.core as mx
    from pion_vllm_mlx import PionPromptCache
    pc = PionPromptCache(model, vquant=vquant, host="127.0.0.1", port=port,
                         stage2=False)
    cache = pc.get_or_prefill(prompt_ids, namespace=namespace)
    out = []
    x = mx.array([[prompt_ids[-1]]])
    logits = model(x, cache=cache)
    next_id = int(mx.argmax(logits[0, -1]))
    out.append(next_id)
    for _ in range(max_tokens - 1):
        x = mx.array([[next_id]])
        logits = model(x, cache=cache)
        next_id = int(mx.argmax(logits[0, -1]))
        out.append(next_id)
    return out


def gen_path_c(model, tok, prompt_ids, max_tokens, namespace, port, vquant):
    import mlx.core as mx
    from pion_vllm_mlx import (
        PionPromptCache, install_pion_attention_patch,
        uninstall_pion_attention_patch, make_pion_prompt_cache,
    )
    install_pion_attention_patch()
    try:
        pc = PionPromptCache(model, vquant=vquant, host="127.0.0.1", port=port,
                              stage2=True)
        _ = pc.get_or_prefill(prompt_ids, namespace=namespace)
        cache_c = make_pion_prompt_cache(
            model, namespace=namespace, prompt_cache=pc,
            prefix_len=len(prompt_ids) - 1,
        )
        # Feed the LAST prompt token as the first suffix token.
        out = []
        x = mx.array([[prompt_ids[-1]]])
        logits = model(x, cache=cache_c)
        next_id = int(mx.argmax(logits[0, -1]))
        out.append(next_id)
        for _ in range(max_tokens - 1):
            x = mx.array([[next_id]])
            logits = model(x, cache=cache_c)
            next_id = int(mx.argmax(logits[0, -1]))
            out.append(next_id)
        return out
    finally:
        uninstall_pion_attention_patch()


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0

    for f in ("pion.vstore.0", "pion.vstore.wal.0", "pion.wal.0", "pion.snapshot.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p): os.remove(p)

    try:
        if args.start:
            proc = start_pion(args.port, "/tmp/pion_bleu_3b.log")

        from mlx_lm import load
        print(f"  Loading {args.model}...")
        model, tok = load(args.model)

        prompts = BUILTIN_PROMPTS
        agg_a, agg_b, agg_c = [], [], []
        bleu_a_b = []
        bleu_a_c = []
        first_match_b = 0
        first_match_c = 0

        for pi, prompt in enumerate(prompts):
            print(f"\n  Prompt {pi+1}/{len(prompts)}: {prompt!r}")
            ids = tok.encode(prompt)
            print(f"    {len(ids)} tokens")
            ref = gen_vanilla(model, tok, ids, args.max_tokens)
            ns = f"bleu3b|p{pi}|{args.vquant}"
            b = gen_path_b(model, tok, ids, args.max_tokens, ns + "|B", args.port, args.vquant)
            c = gen_path_c(model, tok, ids, args.max_tokens, ns + "|C", args.port, args.vquant)
            agg_a.append(ref); agg_b.append(b); agg_c.append(c)
            bleu_a_b.append(bleu_simple(ref, b))
            bleu_a_c.append(bleu_simple(ref, c))
            if ref[0] == b[0]: first_match_b += 1
            if ref[0] == c[0]: first_match_c += 1
            print(f"    A vanilla:  {tok.decode(ref[:12])!r}...")
            print(f"    B rebuild:  {tok.decode(b[:12])!r}...   BLEU={bleu_a_b[-1]:.4f}  first={'==A' if ref[0]==b[0] else '!=A'}")
            print(f"    C Stage-2:  {tok.decode(c[:12])!r}...   BLEU={bleu_a_c[-1]:.4f}  first={'==A' if ref[0]==c[0] else '!=A'}")

        mean_b = sum(bleu_a_b) / len(bleu_a_b)
        mean_c = sum(bleu_a_c) / len(bleu_a_c)
        print(f"\n=== 3B BLEU SUMMARY (mean over {len(prompts)} prompts) ===")
        print(f"  Path B (cache-rebuild): BLEU = {mean_b:.4f}, first-token agree = {first_match_b}/{len(prompts)}")
        print(f"  Path C (Stage-2):       BLEU = {mean_c:.4f}, first-token agree = {first_match_c}/{len(prompts)}")
        print(f"  G2 target: BLEU ≥ 0.95")
        b_pass = mean_b >= 0.95
        c_pass = mean_c >= 0.95
        print(f"  Path B: {'✅' if b_pass else '❌'}    Path C: {'✅' if c_pass else '❌'}")
        if c_pass:
            print(f"\n  Stage-2 closes 3B BLEU strict pass.")
        elif mean_c > mean_b:
            print(f"\n  Stage-2 narrows the gap ({mean_c:.4f} > {mean_b:.4f}) but still under 0.95.")
        else:
            print(f"\n  Stage-2 didn't help BLEU at 3B; cache-rebuild precision loss isn't the dominant factor.")
        if not c_pass:
            rc = 1
    except Exception as e:
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        if args.start:
            stop_pion(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
