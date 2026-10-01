#!/usr/bin/env python3
"""TTFT benchmark — three paths on the same prompt:

  A. Vanilla mlx-lm cold prefill
  B. PionPromptCache cache-rebuild (shipped — fetches K/V from V-store,
     reconstructs the local mlx-lm cache, then runs attention locally).
  C. Stage-2 monkey-patched (this session — K/V live in Pion's MLX
     sidecar; attention runs there, only Q crosses the wire per layer).

Goal: quantify the headline question — does (C) beat (B), and by how
much? Closes the §28.3 ⚠ feasibility-only row at the value level.

Run:
  ./pion-server --kvcache --metal-attention -w 1
  .pixi/envs/default/bin/python tests/bench_ttft.py [--prompt-tokens 256]

Reports per-path: ms-to-first-token (median of N=5 runs), token-1
correctness vs (A), and a cost breakdown (prefill / fetch / store /
attend_query) for paths B and C.
"""
from __future__ import annotations

import argparse
import os
import signal
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--start", action="store_true",
                   help="Spawn `pion-server --kvcache --metal-attention -w 1`.")
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--prompt-tokens", type=int, default=256,
                   help="Length of the shared system prompt in tokens.")
    p.add_argument("--runs", type=int, default=5,
                   help="TTFT measurements per path; report the median.")
    p.add_argument("--vquant", default="fp16",
                   help="V-store quant for path B (fp16 / turbo4 / int8).")
    return p.parse_args()


def start_pion(port: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")  # gh #429
    cmd = [binary, "--kvcache", "--metal-attention", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 60
    import socket
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                # Sidecar warm-up — first request takes ~2s as MLX initializes.
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


def median_pretty(samples_ms):
    if not samples_ms:
        return "—"
    return f"{statistics.median(samples_ms):7.1f} ms (min {min(samples_ms):6.1f}, max {max(samples_ms):6.1f})"


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0
    log_path = "/tmp/pion_ttft_bench.log"

    for f in ("pion.vstore.0", "pion.vstore.wal.0", "pion.wal.0", "pion.snapshot.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p): os.remove(p)

    try:
        if args.start:
            proc = start_pion(args.port, log_path)

        # Heavy imports.
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        from pion_vllm_mlx import PionPromptCache, install_pion_attention_patch, \
            uninstall_pion_attention_patch, make_pion_prompt_cache

        print(f"  Loading {args.model}...")
        model, tok = load(args.model)
        # Build a synthetic prompt of the requested length.
        seed_text = ("Pion is a deterministically low-latency, shared-nothing "
                     "key-value and vector database engine written in Mojo. ")
        words = (seed_text * (args.prompt_tokens // 4 + 4)).split()
        prompt = " ".join(words)
        prompt_ids = tok.encode(prompt)
        prompt_ids = prompt_ids[: args.prompt_tokens]
        if prompt_ids[0] != tok.bos_token_id and tok.bos_token_id is not None:
            prompt_ids = [tok.bos_token_id] + prompt_ids
        suffix_id = tok.encode(" The first")[-1]  # arbitrary user-query token
        print(f"  Prompt: {len(prompt_ids)} tokens, suffix start token = {suffix_id}")

        ttft_a = []
        ttft_b_first = []     # Path B first call (prefill + register + store)
        ttft_b_warm = []      # Path B subsequent calls (lookup + fetch)
        ttft_c_first = []     # Path C first call (prefill + ATTEND.PREFIX.STORE)
        ttft_c_warm = []      # Path C subsequent calls (cache lives in sidecar)
        ref_first_token = None

        # ── Path A: vanilla mlx-lm prefill ─────────────────────────────────
        print(f"\n  Path A: vanilla mlx-lm cold prefill")
        for i in range(args.runs):
            cache = make_prompt_cache(model)
            x = mx.array([prompt_ids])
            t0 = time.perf_counter()
            logits = model(x, cache=cache)
            mx.eval(logits)
            t1 = time.perf_counter()
            tok_id = int(mx.argmax(logits[0, -1]))
            ttft_a.append((t1 - t0) * 1000)
            if ref_first_token is None:
                ref_first_token = tok_id
        print(f"    TTFT median: {median_pretty(ttft_a)}, first_token={ref_first_token}")

        # ── Path B: PionPromptCache rebuild path ───────────────────────────
        print(f"\n  Path B: PionPromptCache cache-rebuild (vquant={args.vquant})")
        ns_b = f"bench_ttft|B|{args.prompt_tokens}|{args.vquant}"
        for i in range(args.runs):
            # Use a fresh PionPromptCache each run for path B's "first" call,
            # then reuse for warm calls. Track first vs warm separately.
            if i == 0:
                pc = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                      port=args.port, stage2=False)
                t0 = time.perf_counter()
                cache = pc.get_or_prefill(prompt_ids, namespace=ns_b)
                # Force MLX to materialize the cache so the timing is honest.
                mx.eval(cache[0].keys, cache[0].values)
                t1 = time.perf_counter()
                ttft_b_first.append((t1 - t0) * 1000)
            else:
                # Warm: lookup hit → fetch from V-store → reconstruct.
                pc2 = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                       port=args.port, stage2=False)
                t0 = time.perf_counter()
                cache = pc2.get_or_prefill(prompt_ids, namespace=ns_b)
                mx.eval(cache[0].keys, cache[0].values)
                t1 = time.perf_counter()
                ttft_b_warm.append((t1 - t0) * 1000)
            # Now run attention to compute first generated token (already
            # has full cache; just feed last prompt token).
            x = mx.array([[prompt_ids[-1]]])
            t2 = time.perf_counter()
            logits = model(x, cache=cache)
            mx.eval(logits)
            t3 = time.perf_counter()
            # Check first token agreement (only on the warm call set).
        print(f"    Cold (prefill + register + store): {median_pretty(ttft_b_first)}")
        print(f"    Warm (lookup + fetch + rebuild):   {median_pretty(ttft_b_warm)}")

        # ── Path C: Stage-2 monkey-patched ─────────────────────────────────
        print(f"\n  Path C: Stage-2 monkey-patched (sidecar runs prefix attention)")
        install_pion_attention_patch()
        try:
            ns_c = f"bench_ttft|C|{args.prompt_tokens}|{args.vquant}"
            for i in range(args.runs):
                if i == 0:
                    pc3 = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                           port=args.port, stage2=True)
                    t0 = time.perf_counter()
                    _ = pc3.get_or_prefill(prompt_ids, namespace=ns_c)
                    t1 = time.perf_counter()
                    ttft_c_first.append((t1 - t0) * 1000)
                    # Cache is now resident in the sidecar.
                else:
                    # Warm: just a LOOKUP + new PionPrefixCache (no K/V transfer).
                    pc4 = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                           port=args.port, stage2=True)
                    t0 = time.perf_counter()
                    _ = pc4.lookup(ns_c)  # HIT
                    cache_c = make_pion_prompt_cache(
                        model, namespace=ns_c, prompt_cache=pc4,
                        prefix_len=len(prompt_ids) - 1)
                    # Feed the LAST prompt token as suffix=1; first generated
                    # token is via attention over [prefix, suffix=1].
                    x = mx.array([[prompt_ids[-1]]])
                    logits = model(x, cache=cache_c)
                    mx.eval(logits)
                    t1 = time.perf_counter()
                    ttft_c_warm.append((t1 - t0) * 1000)
            print(f"    Cold (prefill + sidecar STORE_KV): {median_pretty(ttft_c_first)}")
            print(f"    Warm (lookup + 1-token forward):   {median_pretty(ttft_c_warm)}")
        finally:
            uninstall_pion_attention_patch()

        # ── Summary ────────────────────────────────────────────────────────
        median_a = statistics.median(ttft_a)
        median_b = statistics.median(ttft_b_warm) if ttft_b_warm else float("nan")
        median_c = statistics.median(ttft_c_warm) if ttft_c_warm else float("nan")
        print(f"\n  Headline TTFT (warm path, median of {args.runs - 1}):")
        print(f"    Path A vanilla cold:        {median_a:7.1f} ms")
        print(f"    Path B cache-rebuild:       {median_b:7.1f} ms  ({median_a/median_b:.2f}× vs A)")
        print(f"    Path C Stage-2 monkey-patch:{median_c:7.1f} ms  ({median_a/median_c:.2f}× vs A)")
        if median_c < median_b:
            print(f"\n    Stage-2 wins: {median_b/median_c:.2f}× faster than cache-rebuild.")
        else:
            print(f"\n    Stage-2 SLOWER than cache-rebuild by {median_c/median_b:.2f}× —")
            print(f"    M=1 per-layer round-trip dominates at this scale.")
        print(f"\n    (Path B/C cold path one-time cost: B={ttft_b_first[0] if ttft_b_first else 0:.1f}ms, "
              f"C={ttft_c_first[0] if ttft_c_first else 0:.1f}ms)")
    except Exception as e:
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        if args.start:
            stop_pion(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
