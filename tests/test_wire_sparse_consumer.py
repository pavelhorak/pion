#!/usr/bin/env python3
"""gh #63 (1.5c) — wire-mode sparse consumer end-to-end on Llama-3.2-1B.

Validates the wire-lane sparse-auto path through mlx-lm patch:
  PION_PROMPT_CACHE_NO_INPROC=1 → skip in-proc K/V stash
  + stage2=True                  → K/V pushed to server via ATTEND.PREFIX.STORE
  + sparse_full_layers={...}     → wire SDPA routes to ATTEND.PREFIX.QUERY_SPARSE_AUTO

The wire call is ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED: sparse prefix (server-
resident K/V, top-K blocks chosen server-side) merged with the caller's dense
suffix in one kernel — the v1 "ignores the local suffix" caveat is closed.

Budget and selector (gh #371, measured 2026-09-27, 4,096-token prompt, needle
mid-prefix, K_block=64, Llama-3.2-1B-Instruct-4bit):

  selector     K_blocks  attended   result
  block_mean       8      12.5%     MISS  (" 1")
  block_mean      16      25%       MISS  (" 736")
  block_mean      32      50%       hit
  quest            8      12.5%     hit

This gate runs Quest at 8 blocks. The original 2026-05-12 validation was at
1K (block-mean, 8 blocks = 50% of the prefix), which still passes; at 4K the
block-mean budget is simply too small for this model, and the test's 4K
default had never passed with it. The wire branch also dropped the `selector`
key until gh #371, so `selector: "quest"` silently ran block-mean over the
wire — that was the actual defect.

Requires:
  ./pion-server --kvcache --metal-attention -w 1
  Flat (non-hybrid) model — Llama-3.2-1B; hybrid models like Gemma 4 are
  intentionally NOT supported on the wire path today (per-layer kv_dim
  variation not modeled by the wire protocol).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import List

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx.mlx_lm_patch import (
    install_pion_attention_patch, make_pion_prompt_cache,
)


def greedy_decode(model, prompt_ids: List[int], n_steps: int, cache,
                  m_one: bool = False):
    """Greedy decode with optional one-token-at-a-time feed.

    m_one=True: feed prompt_ids one token at a time, then decode. Always M=1.
                Required for the wire-mode sparse path which is gated on M==1
                (M>1 fused-sparse server kernel is a follow-on).
    m_one=False: feed prompt_ids in one big M=len(prompt_ids) call, then decode.
                 Standard mlx-lm pattern.
    """
    t0 = time.perf_counter()
    if m_one:
        for tid in prompt_ids[:-1]:
            x = mx.array([[tid]])
            out = model(x, cache=cache); mx.eval(out)
        x = mx.array([[prompt_ids[-1]]])
        out = model(x, cache=cache); mx.eval(out)
    else:
        x = mx.array([prompt_ids])
        out = model(x, cache=cache); mx.eval(out)
    t_prefill = (time.perf_counter() - t0) * 1000
    tok = int(mx.argmax(out[0, -1]).item())
    decoded = [tok]
    for _ in range(n_steps - 1):
        nxt = mx.array([[tok]])
        out = model(nxt, cache=cache); mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
    return decoded, t_prefill


def main(args) -> int:
    print(f"gh #63 wire-mode sparse consumer gate (Llama-3.2-1B)")
    print(f"  PION_PROMPT_CACHE_NO_INPROC={os.environ.get('PION_PROMPT_CACHE_NO_INPROC', '0')}")
    print(f"  decode-tokens={args.decode_tokens}")
    print()

    # Force wire mode for the Pion run.
    os.environ["PION_PROMPT_CACHE_NO_INPROC"] = "1"

    print("loading model...")
    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    print(f"  n_layers: {len(model.model.layers)}")

    # Simple NIAH at N tokens:
    needle_city = "Trondheim"
    magic_number = 73691
    filler = "When you analyze a startup's prospects, focus on the founders' demonstrated ability to learn fast. " * 200
    needle = f"\nThe magic number for the city of {needle_city} is {magic_number}.\n"
    question = (
        f"\n\nQuestion: What is the magic number for the city of {needle_city}? "
        f"Answer with only the number.\nAnswer:"
    )
    needle_tokens = tok.encode(needle)
    question_tokens = tok.encode(question)
    base_filler = tok.encode(filler)
    while len(base_filler) < args.length - len(needle_tokens) - len(question_tokens):
        base_filler = base_filler + base_filler
    target_filler = args.length - len(needle_tokens) - len(question_tokens)
    filler_toks = base_filler[:target_filler]
    insert_at = len(filler_toks) // 2
    prompt_ids = filler_toks[:insert_at] + needle_tokens + filler_toks[insert_at:] + question_tokens
    expected = magic_number
    print(f"  prompt length: {len(prompt_ids)} tokens (target {args.length})")

    # ── Vanilla baseline ─────────────────────────────────────────────────
    print(f"\n[A] vanilla mlx-lm (cold)")
    cache = make_prompt_cache(model)
    decoded, ttft = greedy_decode(model, prompt_ids, args.decode_tokens, cache)
    text = tok.decode(decoded)
    print(f"  TTFT: {ttft:.1f}ms")
    print(f"  decode: {text!r}")
    ok_vanilla = str(expected) in text

    # ── Pion wire-mode + sparse on all layers ────────────────────────────
    print(f"\n[B] Pion wire-mode + sparse (K_block={args.sparse_k_block} K_blocks={args.sparse_k_blocks} "
          f"selector={args.selector})")
    install_pion_attention_patch()
    pc = PionPromptCache(model, vquant="fp16", stage2=True)
    # Stable namespace — the gh #65 follow-on (ATTEND.PREFIX.LOOKUP) makes
    # PionPromptCache.lookup stage2-aware, so a stale V-store hit on a cold
    # Metal cache no longer causes `_stage2_push_cold` to skip. The earlier
    # workaround (per-run uuid suffix) is no longer needed.
    ns = "wire_sparse_consumer_v1"
    prefix_ids = prompt_ids[: len(prompt_ids) - len(question_tokens)]
    pc.get_or_prefill(prefix_ids, ns)
    sparse_cfg = {"K_block": args.sparse_k_block, "K_blocks": args.sparse_k_blocks,
                  "selector": args.selector}
    cache_pion = make_pion_prompt_cache(
        model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids),
        sparse_full_layers=sparse_cfg,
    )
    # Feed question one token at a time so M=1 always — required for wire-sparse
    # routing in this v1 (no M>1 fused-sparse server kernel yet).
    decoded_pion, ttft_pion = greedy_decode(
        model, question_tokens, args.decode_tokens, cache_pion, m_one=True
    )
    text_pion = tok.decode(decoded_pion)
    print(f"  TTFT: {ttft_pion:.1f}ms  ({pc.attend_query_calls} wire calls, {pc.attend_query_ms_total:.1f}ms total)")
    print(f"  decode: {text_pion!r}")
    ok_pion = str(expected) in text_pion

    # ── Gate ─────────────────────────────────────────────────────────────
    print("\n━━━ gate ━━━")
    print(f"  vanilla found {expected}: {ok_vanilla}")
    print(f"  pion-wire-sparse found {expected}: {ok_pion}")
    overall = ok_vanilla and ok_pion
    print(f"\n{'PASS' if overall else 'FAIL'} — gh #63 wire-mode sparse consumer end-to-end")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--decode-tokens", type=int, default=12)
    ap.add_argument("--sparse-k-block", type=int, default=64)
    ap.add_argument("--sparse-k-blocks", type=int, default=8)
    ap.add_argument("--selector", choices=["block_mean", "quest"], default="quest")
    args = ap.parse_args()
    sys.exit(main(args))
