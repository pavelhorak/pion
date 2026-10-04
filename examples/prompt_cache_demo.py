#!/usr/bin/env python3
"""
prompt_cache_demo.py — the README's four PionPromptCache lines, timed.

Prefills a ~2,048-token shared prefix the way vanilla mlx-lm does, then gets
the same prefix's K/V from Pion instead, and generates the same answer both
ways. Prints the time to first token for each, and both answers, which must
match.

Usage — Pion must be running with --kvcache (the Homebrew service is):
    brew services start pion                        # or, from a tarball or build:
    ./pion-server --kvcache --metal-attention -w 1
    python3 examples/prompt_cache_demo.py

The first run stores the prefix in Pion; later runs, in any process, find it
there. Each run times five requests the way an app makes them (fetch the
prefix's K/V, then generate a full answer), prints every time to first token,
and takes the ratio from their median. The first request in a process also
pays that process's one-time setup.

    python3 examples/prompt_cache_demo.py --prefix-tokens 33
shows the small end: a 34-token prefix leaves about 15 ms of prefill to skip.

    python3 examples/prompt_cache_demo.py --backend inproc
needs no server and only checks that mlx-lm runs: it re-prefills every time,
so it prints ~1x by construction.

Prerequisites:
    pip install 'pion-vllm-mlx[mlx]'   # mlx + mlx-lm pulled in
"""
from __future__ import annotations

import argparse
import inspect
import statistics
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
            "pip install 'mlx-lm>=0.20.1,<0.33'"
            % getattr(mlx_lm, "__version__", "unknown")
        )
        sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))

try:
    from pion_vllm_mlx import PionPromptCache
except ImportError:
    print("pion_vllm_mlx not found: pip install 'pion-vllm-mlx[mlx]'")
    sys.exit(1)

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely in one sentence."
)
PREFIX = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"

# gh #264: the demo used to run with this ~33-token prefix and nothing else, and
# it therefore DISCONFIRMED the pitch — 1.13x on the inproc backend and 0.86x on
# the wire, because at 33 tokens there is almost no prefill to save. The default
# is now a 2,048-token prefix; --prefix-tokens 33 still shows the small end. The
# whole curve, each point from separate processes, is cross_process_ttft.py's
# sweep (2026-10-02, M4 Mac mini, Pion 0.9.1): 1.5x at 34 tokens, 4.5x at 268,
# 11x at 1,035 and 17x at 2,049.
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


def _not_reachable(args, err) -> int:
    print(f"\nPion is not answering on {args.host}:{args.port} ({err}). Start it first:")
    print("    brew services start pion                        # Homebrew")
    print("    ./pion-server --kvcache --metal-attention -w 1  # tarball or source build")
    print("or run this demo with --backend inproc (no server; it re-prefills, so it shows ~1x).")
    return 2


def _refused(err) -> int:
    print(f"\nPion refused to store the prefix: {err}")
    print("The prompt cache needs the server started with --kvcache "
          "(the Homebrew service already is).")
    return 2


def main():
    ap = argparse.ArgumentParser(
        description="Time the README's four PionPromptCache lines against vanilla mlx-lm.")
    ap.add_argument("--backend", choices=["pion", "inproc"], default="pion",
                    help="pion (default): get the prefix's K/V from a Pion server. "
                         "inproc: no server; re-prefills every time, so it only "
                         "checks that mlx-lm runs and prints ~1x.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prefix-tokens", type=int, default=2048,
                    help="Length of the shared system prefix. The win scales with "
                         "this: below a few hundred tokens there is nothing to reuse.")
    ap.add_argument("--requests", type=int, default=5,
                    help="Timed requests on the cached side; the ratio uses their median.")
    args = ap.parse_args()

    print(f"Loading {args.model} …")
    model, tok = mlx_load(args.model)

    from mlx_lm.models.cache import make_prompt_cache

    prefix_ids = tok.encode(_build_prefix(tok, args.prefix_tokens),
                            add_special_tokens=False)
    print(f"Shared prefix: {len(prefix_ids)} tokens "
          f"(--prefix-tokens {args.prefix_tokens})")

    pc = None
    stored = False
    namespace = None
    hits_before = 0
    if args.backend == "pion":
        # The key names everything the K/V depends on: model, token ids, dtype.
        namespace = PionPromptCache.make_namespace(
            args.model, "prompt_cache_demo", "fp16", ",".join(map(str, prefix_ids)))
        try:
            pc = PionPromptCache(model, host=args.host, port=args.port)
            stored = pc.lookup(namespace)
        except OSError as e:
            return _not_reachable(args, e)

    # One short untimed forward first, so neither side pays Metal's one-time
    # kernel compile inside its clock (cross_process_ttft.py does the same).
    warm = make_prompt_cache(model)
    model(mx.array(prefix_ids[:64])[None], cache=warm)
    mx.eval([c.state for c in warm])

    # gh #264: the prefill is the whole cost being saved, so it is inside the
    # clock on both sides. Vanilla is mlx-lm's own path: the prefix prefilled
    # with only the cache evaluated, then generate_step over the suffix.
    print("\n[vanilla] cold prefill of the whole prefix, then generate …")
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    model(mx.array(prefix_ids)[None], cache=cache)
    mx.eval([c.state for c in cache])
    cold_prefix = time.perf_counter() - t0
    cold_out, cold_first, cold_gen = _generate(model, tok, cache, SUFFIX)
    cold_ttft = cold_prefix + cold_first
    cold_total = cold_prefix + cold_gen
    print(f"[vanilla] {cold_ttft*1000:>8.1f} ms to first token  ·  "
          f"{cold_total*1000:.1f} ms to the full answer  →  {cold_out!r}")

    if pc is not None:
        if stored:
            print("\n[pion]    the prefix is already in Pion (an earlier run, or "
                  "another process, stored it)")
        else:
            print("\n[pion]    the prefix is not in Pion yet: prefilling it once and "
                  "storing it (not timed) …")
            try:
                pc.get_or_prefill(prefix_ids, namespace=namespace)
            except RuntimeError as e:
                return _refused(e)
        hits_before = pc.hits

    def request():
        """One request as an app makes it: get the prefix's cache, then generate."""
        t0 = time.perf_counter()
        if pc is not None:
            cache = pc.get_or_prefill(prefix_ids, namespace=namespace)
        else:
            # inproc has nothing to fetch from: it re-prefills, by construction.
            cache = make_prompt_cache(model)
            model(mx.array(prefix_ids)[None], cache=cache)
            mx.eval([c.state for c in cache])
        got = time.perf_counter() - t0
        out, first, gen = _generate(model, tok, cache, SUFFIX)
        return out, got + first, got + gen

    label = "pion" if pc is not None else "inproc"
    runs = [request() for _ in range(max(1, args.requests))]
    ttfts = [r[1] * 1000 for r in runs]
    each = "a fetch and a full answer" if pc is not None else "a re-prefill and a full answer"
    print(f"[{label}]{' ' * (9 - len(label))}{len(runs)} requests, each {each}, "
          f"ms to first token: " + " · ".join(f"{t:.1f}" for t in ttfts))
    print(f"[{label}]{' ' * (9 - len(label))}answer: {runs[-1][0]!r}")

    if pc is not None and pc.hits - hits_before != len(runs):
        print("\n!! a timed request was not a cache hit, so its time includes a "
              "prefill — please report it")
    if any(r[0] != cold_out for r in runs):
        print("\n!! outputs differ — that is a bug, please report it:")
        print(f"   vanilla: {cold_out!r}")
        for i, r in enumerate(runs):
            if r[0] != cold_out:
                print(f"   request {i + 1}: {r[0]!r}")

    warm_ttft = statistics.median(r[1] for r in runs)
    warm_total = statistics.median(r[2] for r in runs)
    ratio = cold_ttft / warm_ttft if warm_ttft > 0 else float("inf")
    print(f"\nTime to first token: {cold_ttft*1000:.1f} ms → {warm_ttft*1000:.1f} ms, the median"
          f"   ({ratio:.2f}×, backend={args.backend}, {len(prefix_ids)}-token prefix)")
    print(f"Full answer:         {cold_total*1000:.1f} ms → {warm_total*1000:.1f} ms"
          + ("   (decode itself is unchanged — Pion saves the prefill)" if pc is not None else ""))
    if pc is not None and not stored:
        print("The first request ran right after this process stored the prefix and "
              "pays its one-time setup;\nrun the demo again to time requests in a "
              "fresh process, the way another process would find the prefix.")
    if pc is not None:
        print("Every request here follows a full answer in this process. Requests like that "
              "measure slower than the\nREADME's separate-process row, which times a fresh "
              "process's first request\n(benchmarks/reproducers/cross_process_ttft.py).")
    if pc is None:
        print("Note: inproc re-prefills on every request, so ~1x is the expected "
              "result — run without --backend inproc, with Pion up, to see the "
              "prefix reused.")


if __name__ == "__main__":
    sys.exit(main())
