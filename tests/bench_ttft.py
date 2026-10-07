#!/usr/bin/env python3
"""TTFT benchmark — five paths on the same prompt:

  A. Vanilla mlx-lm cold prefill
  B. PionPromptCache cache-rebuild (shipped — fetches K/V from V-store,
     reconstructs the local mlx-lm cache, then runs attention locally).
  C. Stage 2, wire lane — a fresh PionPromptCache, as another consumer
     would have: K/V live in Pion, attention runs there, only Q crosses
     the wire per layer.
  D. Stage 2, in-process lane — the PionPromptCache that prefilled keeps
     the prefix K/V resident as MLX arrays; attention runs locally.
  E. mlx-lm's own prompt-cache FILE — save_prompt_cache once, then each run
     times load_prompt_cache(file) + the first token. Free, built into mlx-lm,
     and the baseline every Pion number here has to be read against (the
     file sits in the page cache, as Pion's rows sit in a RAM-resident server).

Goal: quantify the headline question — does (C) beat (B), and by how
much? Closes the §28.3 ⚠ feasibility-only row at the value level.

Run:
  ./pion-server --kvcache --metal-attention -w 1
  .pixi/envs/default/bin/python tests/bench_ttft.py [--prompt-tokens 256]

Reports per-path: ms-to-first-token (median of N=5 runs), token-1
correctness vs (A), and a cost breakdown (prefill / fetch / store /
attend_query) for paths B and C.

All five paths produce the first token after the same prompt: A prefills it
the way mlx_lm.generate_step does (see first_token_logits), B, C, D and E restore
every token but the last and run the last one, and each must produce A's token. Until 2026-10-02 path A
evaluated the logits of one forward over the whole prompt — a vocabulary
projection at every position, which no generation computes — and path B
stopped its clock before the first-token forward. Both made the ratios too
high (the old 1,530 ms baseline and its 50.6x).
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
PREFILL_STEP = 2048          # mlx_lm.generate_step's default prefill_step_size


def first_token_logits(model, ids, cache):
    """The first token's logits after `ids`, computed the way mlx_lm.generate_step does.

    Every token but the last runs through the model in PREFILL_STEP chunks with
    only the cache state evaluated; the last token alone yields the logits.
    """
    import mlx.core as mx
    x = mx.array([ids])
    done, n = 0, x.shape[1]
    while n - done > 1:
        step = min(PREFILL_STEP, n - done - 1)
        model(x[:, done:done + step], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()       # as generate_step does after each prefill chunk
        done += step
    last = model(x[:, done:], cache=cache)[0, -1]
    mx.eval(last)
    return last


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
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        prompt_ids = (bos + tok.encode(prompt, add_special_tokens=False))[: args.prompt_tokens]
        suffix_id = tok.encode(" The first")[-1]  # arbitrary user-query token
        print(f"  Prompt: {len(prompt_ids)} tokens, suffix start token = {suffix_id}")

        ttft_a = []
        ttft_b_first = []     # Path B first call (prefill + register + store)
        ttft_b_warm = []      # Path B subsequent calls (lookup + fetch)
        ttft_c_first = []     # Path C first call (prefill + ATTEND.PREFIX.STORE)
        ttft_c_warm = []      # Path C subsequent calls (wire lane: attention in Pion)
        ttft_d_warm = []      # Path D (in-process lane: prefix K/V resident as MLX arrays)
        ttft_e_warm = []      # Path E (mlx-lm's own prompt-cache file: load_prompt_cache)
        file_bytes = 0
        ref_first_token = None

        # ── Path A: vanilla mlx-lm prefill ─────────────────────────────────
        print(f"\n  Path A: vanilla mlx-lm cold prefill")
        for i in range(args.runs):
            cache = make_prompt_cache(model)
            t0 = time.perf_counter()
            last = first_token_logits(model, prompt_ids, cache)
            t1 = time.perf_counter()
            tok_id = int(mx.argmax(last))
            ttft_a.append((t1 - t0) * 1000)
            if ref_first_token is None:
                ref_first_token = tok_id
        print(f"    TTFT median: {median_pretty(ttft_a)}, first_token={ref_first_token}")

        # ── Path B: PionPromptCache rebuild path ───────────────────────────
        print(f"\n  Path B: PionPromptCache cache-rebuild (vquant={args.vquant})")
        # The prefix is every prompt token but the last, as in path C; the
        # clock covers the first token, which is when the rebuilt cache is
        # actually used (MLX is lazy, so a cache "materialized" without the
        # forward is not a time to first token).
        ns_b = f"bench_ttft|B|{args.prompt_tokens}|{args.vquant}|n-1"
        for i in range(args.runs):
            # A fresh PionPromptCache each run: run 0 misses (prefill + register
            # + store), the rest hit (lookup + fetch from V-store + rebuild).
            pc = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                 port=args.port, stage2=False)
            t0 = time.perf_counter()
            cache = pc.get_or_prefill(prompt_ids[:-1], namespace=ns_b)
            tok_b = int(mx.argmax(first_token_logits(model, prompt_ids[-1:], cache)))
            t1 = time.perf_counter()
            (ttft_b_first if i == 0 else ttft_b_warm).append((t1 - t0) * 1000)
            if tok_b != ref_first_token:
                print(f"    !! run {i}: first token {tok_b} != vanilla {ref_first_token}")
                rc = 1
        print(f"    Cold (prefill + register + store + first token): {median_pretty(ttft_b_first)}")
        print(f"    Warm (lookup + fetch + rebuild + first token):   {median_pretty(ttft_b_warm)}")

        # ── Paths C and D: Stage 2 (mlx-lm patched) ────────────────────────
        # Both store every prompt token but the last, then run the last one.
        # C builds a fresh PionPromptCache per request, as another consumer
        # would: it holds no MLX copy of the prefix, so it takes the WIRE lane —
        # Pion computes the prefix attention and the query crosses per layer.
        # D reuses the instance that prefilled, whose prefix K/V stayed resident
        # as MLX arrays: the in-process lane, zero wire round trips. Until
        # 2026-10-02 only C existed, and the docs called its number the
        # "same-process" / in-process-lane row.
        print(f"\n  Path C: Stage 2, wire lane (Pion computes the prefix attention)")
        install_pion_attention_patch()
        try:
            ns_c = f"bench_ttft|C|{args.prompt_tokens}|{args.vquant}|n-1"
            prefix_c = prompt_ids[:-1]
            pc3 = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                  port=args.port, stage2=True)
            t0 = time.perf_counter()
            _ = pc3.get_or_prefill(prefix_c, namespace=ns_c)
            ttft_c_first.append((time.perf_counter() - t0) * 1000)

            def stage2_first_token(pc):
                t0 = time.perf_counter()
                _ = pc.lookup(ns_c)  # HIT
                cache_s2 = make_pion_prompt_cache(
                    model, namespace=ns_c, prompt_cache=pc, prefix_len=len(prefix_c))
                tok_s2 = int(mx.argmax(first_token_logits(model, prompt_ids[-1:], cache_s2)))
                return (time.perf_counter() - t0) * 1000, tok_s2

            for i in range(args.runs - 1):
                pc4 = PionPromptCache(model, vquant=args.vquant, host="127.0.0.1",
                                      port=args.port, stage2=True)
                ms, tok_c = stage2_first_token(pc4)
                ttft_c_warm.append(ms)
                if tok_c != ref_first_token:
                    print(f"    !! C run {i}: first token {tok_c} != vanilla {ref_first_token}")
                    rc = 1
            print(f"    Cold (prefill + register + store, both lanes): {median_pretty(ttft_c_first)}")
            print(f"    Warm (lookup + wire attention + first token):  {median_pretty(ttft_c_warm)}")

            print(f"\n  Path D: Stage 2, in-process lane (the instance that prefilled)")
            for i in range(args.runs - 1):
                ms, tok_d = stage2_first_token(pc3)
                ttft_d_warm.append(ms)
                if tok_d != ref_first_token:
                    print(f"    !! D run {i}: first token {tok_d} != vanilla {ref_first_token}")
                    rc = 1
            print(f"    Warm (lookup + local attention + first token): {median_pretty(ttft_d_warm)}")
        finally:
            uninstall_pion_attention_patch()

        # ── Path E: mlx-lm's own prompt-cache file ─────────────────────────
        # The free alternative a reader already has. Prefill every token but
        # the last once and save it (untimed, like B's and C's first call);
        # each run then maps the file back and answers the first token.
        print(f"\n  Path E: mlx-lm prompt-cache file (save_prompt_cache / load_prompt_cache)")
        import tempfile
        from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
        with tempfile.TemporaryDirectory(prefix="bench_ttft_file_") as d:
            fpath = os.path.join(d, "prefix.safetensors")
            cache_e = make_prompt_cache(model)
            first_token_logits(model, prompt_ids[:-1] + prompt_ids[-1:], cache_e)  # warm the same shapes
            cache_e = make_prompt_cache(model)
            x = mx.array([prompt_ids[:-1]])
            done, n = 0, x.shape[1]
            while done < n:
                step = min(PREFILL_STEP, n - done)
                model(x[:, done:done + step], cache=cache_e)
                mx.eval([c.state for c in cache_e])
                done += step
            save_prompt_cache(fpath, cache_e)
            file_bytes = os.path.getsize(fpath)
            for i in range(args.runs - 1):
                t0 = time.perf_counter()
                cache_f = load_prompt_cache(fpath)
                tok_e = int(mx.argmax(first_token_logits(model, prompt_ids[-1:], cache_f)))
                ttft_e_warm.append((time.perf_counter() - t0) * 1000)
                if tok_e != ref_first_token:
                    print(f"    !! E run {i}: first token {tok_e} != vanilla {ref_first_token}")
                    rc = 1
        print(f"    Warm (load_prompt_cache + first token), file {file_bytes / 1e6:.1f} MB: "
              f"{median_pretty(ttft_e_warm)}")

        # ── Summary ────────────────────────────────────────────────────────
        median_a = statistics.median(ttft_a)
        median_b = statistics.median(ttft_b_warm) if ttft_b_warm else float("nan")
        median_c = statistics.median(ttft_c_warm) if ttft_c_warm else float("nan")
        median_d = statistics.median(ttft_d_warm) if ttft_d_warm else float("nan")
        median_e = statistics.median(ttft_e_warm) if ttft_e_warm else float("nan")
        print(f"\n  Headline TTFT (warm path, median of {args.runs - 1}):")
        print(f"    Path A vanilla cold:        {median_a:7.1f} ms")
        print(f"    Path B cache-rebuild:       {median_b:7.1f} ms  ({median_a/median_b:.2f}× vs A)")
        print(f"    Path C Stage 2, wire lane:  {median_c:7.1f} ms  ({median_a/median_c:.2f}× vs A)")
        print(f"    Path D Stage 2, in-process: {median_d:7.1f} ms  ({median_a/median_d:.2f}× vs A)")
        print(f"    Path E mlx-lm file:         {median_e:7.1f} ms  ({median_a/median_e:.2f}× vs A)"
              f"  [{file_bytes / 1e6:.1f} MB]")
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
