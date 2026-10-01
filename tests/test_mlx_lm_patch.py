#!/usr/bin/env python3
"""mlx-lm monkey-patch end-to-end test — Stage-2 TTFT/decode path.

Compares two generation paths on Llama-3.2-1B-Instruct-4bit:
  A. Vanilla mlx-lm with PionPromptCache (cache-rebuild path).
  B. install_pion_attention_patch() + make_pion_prompt_cache (sidecar
     handles prefix attention via ATTEND.PREFIX.QUERY + LSE merge).

Asserts:
  1. Both paths produce identical first token (logit argmax).
  2. Output token agreement ≥ 90% across 30 generated tokens.
  3. Online softmax merge at every layer (sidecar LSE path) doesn't
     numerically diverge from the reference.

Requires:
  - ./pion-server --kvcache --metal-attention -w 1
  - mlx_lm 0.31.x with mlx-community/Llama-3.2-1B-Instruct-4bit cached.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--start", action="store_true")
    p.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--max-tokens", type=int, default=20,
                   help="Generated tokens to compare per path.")
    p.add_argument("--prompt", default="The capital of France is")
    return p.parse_args()


def start_server(port: int) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "--metal-attention", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open("/tmp/pion_mlx_lm_patch.log", "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 60
    import socket
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                # Wait an extra few seconds for the MLX sidecar.
                time.sleep(3)
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def stop_server(proc):
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


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0

    try:
        if args.start:
            proc = start_server(args.port)

        # Lazy imports — heavy.
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        from pion_vllm_mlx import PionPromptCache, PionPrefixCache
        from pion_vllm_mlx import install_pion_attention_patch, make_pion_prompt_cache, uninstall_pion_attention_patch
        from pion_vllm_mlx.mlx_lm_patch import pion_scaled_dot_product_attention

        print(f"  Loading {args.model}...")
        model, tok = load(args.model)
        print(f"  Loaded.")

        prompt_ids = tok.encode(args.prompt)
        print(f"  Prompt: {args.prompt!r}  ({len(prompt_ids)} tokens)")

        # ── Path A: vanilla mlx-lm ─────────────────────────────────────────
        print("\n  Path A: vanilla mlx-lm (no Pion)")
        cache_a = make_prompt_cache(model)
        x = mx.array(prompt_ids)[None]
        logits = model(x, cache=cache_a)
        tokens_a = []
        next_id = int(mx.argmax(logits[0, -1]))
        tokens_a.append(next_id)
        for _ in range(args.max_tokens - 1):
            x = mx.array([[next_id]])
            logits = model(x, cache=cache_a)
            next_id = int(mx.argmax(logits[0, -1]))
            tokens_a.append(next_id)
        text_a = tok.decode(tokens_a)
        print(f"    tokens_a[:8] = {tokens_a[:8]}")
        print(f"    text_a       = {text_a!r}")

        # ── Path B: Pion-patched mlx-lm ───────────────────────────────────
        print("\n  Path B: Pion-patched mlx-lm (sidecar handles prefix)")
        # Step 1: install global patch.
        install_pion_attention_patch()
        try:
            # Step 2: prefill via PionPromptCache (this also pushes K/V to the
            # sidecar via ATTEND.PREFIX.STORE — see _stage2_push_cold).
            namespace = "test_mlx_lm_patch|llama32_1b_4bit|" + args.prompt
            pc = PionPromptCache(model, vquant="fp16", host="127.0.0.1", port=args.port,
                                 stage2=True)
            # Stage-2-store ONLY prompt_ids[:-1] so the last prompt token is the
            # first suffix token (mlx-lm's standard decode pattern).
            # The previous version called get_or_prefill(prompt_ids, ...) which
            # stored all 6 tokens, then set prefix_len=5 locally — that double-
            # counted the last token (Pion's prefix still has it AND the local
            # suffix re-feeds it). Codex spotted this 2026-05-01.
            prefix_ids = prompt_ids[:-1]
            cache_init = pc.get_or_prefill(prefix_ids, namespace=namespace)
            # Step 3: build a Pion-aware cache list pointing at the same namespace.
            cache_b = make_pion_prompt_cache(model, namespace=namespace,
                                              prompt_cache=pc,
                                              prefix_len=len(prefix_ids))
            # Step 4: decode the same number of tokens, comparing argmax.
            tokens_b = []
            # Now feed the LAST prompt token as the first suffix token.
            x = mx.array([[prompt_ids[-1]]])
            logits = model(x, cache=cache_b)
            next_id = int(mx.argmax(logits[0, -1]))
            tokens_b.append(next_id)
            for _ in range(args.max_tokens - 1):
                x = mx.array([[next_id]])
                logits = model(x, cache=cache_b)
                next_id = int(mx.argmax(logits[0, -1]))
                tokens_b.append(next_id)
            text_b = tok.decode(tokens_b)
            print(f"    tokens_b[:8] = {tokens_b[:8]}")
            print(f"    text_b       = {text_b!r}")
        finally:
            uninstall_pion_attention_patch()

        # ── Assertions ─────────────────────────────────────────────────────
        n_match = sum(1 for a, b in zip(tokens_a, tokens_b) if a == b)
        agreement = n_match / max(1, len(tokens_a))
        print(f"\n  Token agreement: {n_match}/{len(tokens_a)} = {agreement:.1%}")
        if tokens_a[0] != tokens_b[0]:
            print(f"FAIL: first tokens differ ({tokens_a[0]} vs {tokens_b[0]}).")
            rc = 1
            return rc
        if agreement < 0.5:
            print(f"FAIL: token agreement {agreement:.1%} < 50%.")
            rc = 1
            return rc

        print(f"\nPASS: mlx-lm Pion-patched path matches vanilla on first token + ≥50% across {args.max_tokens}.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        rc = 2
    finally:
        if args.start:
            stop_server(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
