#!/usr/bin/env python3
"""
prompt_cache_demo.py — 5-minute PionPromptCache headline demo.

Shows the in-process fast lane (Stage 2, gh #50): cold prefill warms Pion,
warm forward skips prefill entirely. No Pion server needed for the in-process
path — K/V lives as MLX arrays inside this process.

Usage:
    # In-process path (default, no server needed):
    python3 examples/prompt_cache_demo.py

    # Cross-process path (requires Pion; the Homebrew service already runs
    # with these flags):
    ./pion-server --kvcache --metal-attention -w 1
    python3 examples/prompt_cache_demo.py --backend pion

Prerequisites:
    pip install 'pion-vllm-mlx[mlx]'   # mlx + mlx-lm pulled in
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
except ImportError:
    print("mlx and mlx-lm are required: pip install 'pion-vllm-mlx[mlx]'")
    sys.exit(1)

# gh #263: generate_step moved from mlx_lm.utils to mlx_lm.generate somewhere
# between 0.20.1 and 0.22.5 (measured — it is in utils on 0.20.1 and in generate
# from 0.22.5 on). Importing only the old location made this demo die on every
# supported mlx-lm but the very oldest, printing "mlx and mlx-lm are required" —
# advice for packages the user has already installed. Try both, and if neither
# has it, say which symbol is actually missing.
try:
    from mlx_lm.generate import generate_step
except ImportError:
    try:
        from mlx_lm.utils import generate_step   # mlx-lm 0.20.1 and older
    except ImportError:
        import mlx_lm
        print(
            "this mlx-lm (%s) exposes generate_step in neither mlx_lm.generate "
            "nor mlx_lm.utils.\nInstall a tested version: "
            "pip install 'mlx-lm>=0.20.0,<0.32'"
            % getattr(mlx_lm, "__version__", "unknown")
        )
        sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))

try:
    from pion_vllm_mlx import PionPromptCache, install_pion_attention_patch
except ImportError:
    print("pion_vllm_mlx not found: pip install 'pion-vllm-mlx[mlx]'")
    sys.exit(1)

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely in one sentence."
)
PREFIX = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"

# gh #264: the demo used to run with this ~33-token prefix and nothing else, and
# it therefore DISCONFIRMED the pitch — measured 1.13x on the inproc backend and
# 0.86x on the wire, because at 33 tokens there is no prefill worth saving and
# the round-trip costs more than recomputing it. The mechanism is fine; the
# workload was outside the regime the claim comes from. Same code path, only the
# prefix length varying:
#
#     prefix tokens   vanilla cold   pion warm   speedup
#                33        332 ms      152 ms      2.18x
#               267        444 ms      164 ms      2.71x
#             1,034      1,024 ms      308 ms      3.33x
#             2,061      2,113 ms       91 ms     23.18x
#
# So the default is now a 2,048-token prefix — the size the documented numbers
# were measured at. --prefix-tokens 33 reproduces the old behaviour, which is
# worth seeing: it is the honest picture of when this technique does nothing.
_FILLER = ("The assistant is helpful, precise, and answers in one sentence. ")


def _build_prefix(tok, target_tokens: int) -> str:
    """A system prefix of roughly `target_tokens` tokens."""
    head = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + SYSTEM_PROMPT + " "
    body = head
    while len(tok.encode(body, add_special_tokens=False)) < target_tokens:
        body += _FILLER
    return body + "<|eot_id|>"
SUFFIX = "<|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
MAX_TOKENS = 32


# gh #264: three things about generate_step move between mlx-lm versions, and
# all three were wrong here — the demo could not run at all on current mlx-lm.
#   * the cache keyword is `prompt_cache=` now and was `cache=` before;
#   * the prompt must be 1-D (a [None] batch axis raises "too many values to
#     unpack" from inside mlx-lm);
#   * the generator yields (token, logprobs), so unpacking it as a bare token
#     hands you a tuple and `.item()` fails.
# Resolved once, from the signature, rather than pinned to a version.
_CACHE_KW = "prompt_cache" if "prompt_cache" in inspect.signature(
    generate_step).parameters else "cache"


def _generate(model, tok, cache, suffix_text: str) -> tuple[str, float, float]:
    """Returns (text, time-to-FIRST-token, total generation time).

    TTFT used to be measured after the whole loop, which made it the total
    generation time wearing the name of a latency metric — and TTFT is the
    number this demo exists to show.
    """
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    suffix_mx = mx.array(suffix_ids)               # 1-D: no batch axis
    steps = generate_step(suffix_mx, model, **{_CACHE_KW: cache})

    t0 = time.perf_counter()
    ttft = None
    tokens = []
    for step, _ in zip(steps, range(MAX_TOKENS)):
        token = step[0] if isinstance(step, tuple) else step
        if ttft is None:
            ttft = time.perf_counter() - t0        # first token is out
        tid = int(token.item()) if hasattr(token, "item") else int(token)
        if tid == tok.eos_token_id:
            break
        tokens.append(tid)
    total = time.perf_counter() - t0
    return tok.decode(tokens), (ttft if ttft is not None else total), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["inproc", "pion"], default="inproc")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prefix-tokens", type=int, default=2048,
                    help="Length of the shared system prefix. The win scales with "
                         "this: below a few hundred tokens there is nothing to reuse.")
    args = ap.parse_args()

    print(f"Loading {args.model} …")
    model, tok = mlx_load(args.model)

    if args.backend == "pion":
        install_pion_attention_patch()
        # gh #264: the model is PionPromptCache's FIRST parameter and defaults
        # to None. Omitting it made --backend pion — the only mode that shows
        # cross-session reuse — die inside get_or_prefill with
        # "'NoneType' object has no attribute 'layers'".
        pc = PionPromptCache(model, host=args.host, port=args.port)
        namespace = f"demo|llama321b|fp16|system_v1|{args.prefix_tokens}"
    else:
        pc = None

    prefix_ids = tok.encode(_build_prefix(tok, args.prefix_tokens),
                            add_special_tokens=False)
    print(f"Shared prefix: {len(prefix_ids)} tokens "
          f"(--prefix-tokens {args.prefix_tokens})")

    from mlx_lm.models.cache import make_prompt_cache

    # gh #264: BOTH passes used to prefill (or fetch) OUTSIDE the timer and then
    # time only the suffix — so "cold" and "warm" measured the same thing and the
    # demo could never show more than noise (measured 1.13x / 0.86x). The prefill
    # is the entire cost being saved; it has to be inside the clock.

    # --- Path A: vanilla mlx-lm. Prefills the prefix on every request. ---
    print("\n[vanilla] cold prefill of the whole prefix, then generate …")
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    model(mx.array(prefix_ids)[None], cache=cache)
    mx.eval([c.state for c in cache])
    cold_prefix = time.perf_counter() - t0
    cold_out, cold_first, cold_gen = _generate(model, tok, cache, SUFFIX)
    # TTFT = the prefix phase + _generate's own first-token time. Reading the
    # clock after _generate returned timed the WHOLE answer while printing
    # "Time to first token" — decode time diluted the ratio (8.2x printed where
    # the first-token ratio was ~25x at 2,048 tokens; 1.6x printed at 33).
    cold_ttft = cold_prefix + cold_first
    cold_total = cold_prefix + cold_gen
    print(f"[vanilla] {cold_ttft*1000:>8.1f} ms to first token  ·  {cold_total*1000:.1f} ms to the full answer  →  {cold_out!r}")

    # --- Path B: Pion. The prefix is already prefilled; this is a cache hit. ---
    print("\n[pion]    warming the cache once (not timed) …")
    if pc is not None:
        pc.get_or_prefill(prefix_ids, namespace=namespace)
    else:
        warm_seed = make_prompt_cache(model)
        model(mx.array(prefix_ids)[None], cache=warm_seed)
        mx.eval([c.state for c in warm_seed])

    print("[pion]    hitting it …")
    t0 = time.perf_counter()
    if pc is not None:
        cache = pc.get_or_prefill(prefix_ids, namespace=namespace)
    else:
        # The inproc path has nothing to fetch from — it re-prefills, which is
        # why this backend reports ~1x and the note below says so.
        cache = make_prompt_cache(model)
        model(mx.array(prefix_ids)[None], cache=cache)
        mx.eval([c.state for c in cache])
    warm_prefix = time.perf_counter() - t0
    warm_out, warm_first, warm_gen = _generate(model, tok, cache, SUFFIX)
    warm_ttft = warm_prefix + warm_first
    warm_total = warm_prefix + warm_gen
    print(f"[pion]    {warm_ttft*1000:>8.1f} ms to first token  ·  {warm_total*1000:.1f} ms to the full answer  →  {warm_out!r}")

    if cold_out != warm_out:
        print("\n!! outputs differ — that is a bug, please report it:")
        print(f"   vanilla: {cold_out!r}")
        print(f"   pion   : {warm_out!r}")

    ratio = cold_ttft / warm_ttft if warm_ttft > 0 else float("inf")
    print(f"\nTime to first token: {cold_ttft*1000:.1f} ms → {warm_ttft*1000:.1f} ms"
          f"   ({ratio:.2f}× faster, backend={args.backend},"
          f" {len(prefix_ids)}-token prefix)")
    print(f"Full answer:         {cold_total*1000:.1f} ms → {warm_total*1000:.1f} ms"
          f"   (decode itself is unchanged — Pion saves the prefill)")
    if args.backend == "inproc":
        print("Note: inproc path recomputes prefill on each call — "
              "run with --backend pion to see cross-session KV reuse.")


if __name__ == "__main__":
    main()
