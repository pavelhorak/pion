#!/usr/bin/env python3
"""
hybrid_retrieval_demo.py — single-query demonstration of Pion's
chunk-id-keyed K/V hybrid retrieval (gh #54 substrate + gh #75 Stage 1).

What this shows:
  Same RAG query handled two ways:
    A) text-RAG baseline — encode (chunk + question), cold prefill, decode.
    B) hybrid retrieval — chunk's K/V was previously cached by chunk_id;
       hydrate the cache, warm-forward only the question, decode.

  Both paths produce equivalent answers (within fp16 noise on the pion
  lane); the hybrid path skips the chunk prefill, saving ~4-5× TTFT on
  Llama-3.2-1B at SQuAD-shaped contexts.

Backends:
  - `--backend inproc` (default): K/V held as MLX arrays in this process.
    Bit-perfect generation parity; no Pion server needed.
  - `--backend pion`: K/V serialized cross-process via Pion's KV.PREFIX.*
    + V.STOREBATCH. Requires `./pion-server --kvcache --metal-attention -w 1`.

Usage:
  python3 examples/hybrid_retrieval_demo.py                       # inproc
  python3 examples/hybrid_retrieval_demo.py --backend pion        # cross-process

Run-time: ~30-45 seconds (~30s model load, ~10s per path).

Scope: single retrieved chunk per query. Multi-chunk routes to text-RAG
per gh #75 sub-task 2; see issue thread for the architectural decision.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pion-vllm-mlx"))
from pion_vllm_mlx import HybridRetrievalCache  # noqa: E402


CHUNK = (
    "The Eiffel Tower (French: la Tour Eiffel) is a wrought-iron lattice "
    "tower on the Champ de Mars in Paris, France. It is named after the "
    "engineer Gustave Eiffel, whose company designed and built the tower "
    "for the 1889 World's Fair. The tower is 330 metres tall, about the "
    "same height as an 81-storey building, and the tallest structure in "
    "Paris. Its base is square, measuring 125 metres on each side. During "
    "its construction, the Eiffel Tower surpassed the Washington Monument "
    "to become the tallest man-made structure in the world, a title it "
    "held for 41 years until the Chrysler Building in New York City was "
    "finished in 1930."
)

QUERY = "\n\nQuestion: How tall is the Eiffel Tower?\nAnswer:"


def greedy_decode(model, prompt_ids: list, n_steps: int, cache) -> tuple:
    """Returns (decoded_token_ids, ttft_ms)."""
    x = mx.array([prompt_ids])
    t0 = time.perf_counter()
    out = model(x, cache=cache)
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


def main(args) -> int:
    print(f"hybrid_retrieval_demo  backend={args.backend}  model={args.model}")
    print(f"  chunk: {CHUNK[:60]!r}...")
    print(f"  query: {QUERY.strip()!r}")
    print()

    print("loading model...")
    model, tok = mlx_load(args.model)

    chunk_ids = tok.encode(CHUNK)
    suffix_ids = tok.encode(QUERY)
    print(f"  chunk={len(chunk_ids)} tokens · suffix={len(suffix_ids)} tokens")
    print()

    # Path A — text-RAG baseline (same tokenization as the hybrid path for
    # an apples-to-apples comparison: encode chunk and suffix separately and
    # concat the token lists, not the strings).
    print("[A] text-RAG baseline — cold prefill of (chunk + query)")
    cache_a = make_prompt_cache(model)
    # Warmup pass (JIT compile) so the timed path doesn't pay it.
    _ = greedy_decode(model, chunk_ids[:32] + suffix_ids, 1, make_prompt_cache(model))
    cache_a = make_prompt_cache(model)
    decoded_a, ttft_a = greedy_decode(model, chunk_ids + suffix_ids, args.n_gen, cache_a)
    text_a = tok.decode(decoded_a)
    print(f"    TTFT   {ttft_a:>9,.1f} ms  ({len(chunk_ids) + len(suffix_ids)} tokens prefill)")
    print(f"    answer {text_a[:80]!r}")
    print()

    # Path B — hybrid retrieval
    print(f"[B] hybrid retrieval ({args.backend}) — chunk K/V cached, warm-forward query")
    hr_kwargs = {"backend": args.backend}
    if args.backend == "pion":
        hr_kwargs.update({"host": args.pion_host, "port": args.pion_port})
    hr = HybridRetrievalCache(model, **hr_kwargs)

    print("    ingest (one-time per chunk; NOT counted in TTFT)...")
    t_ing = time.perf_counter()
    hr.ingest("eiffel_passage", chunk_ids)
    print(f"    chunk ingested in {(time.perf_counter() - t_ing) * 1000:>9,.1f} ms "
          f"(amortized across many queries against the same chunk)")

    cache_b, suffix = hr.prepare("eiffel_passage", suffix_ids)
    decoded_b, ttft_b = greedy_decode(model, suffix, args.n_gen, cache_b)
    text_b = tok.decode(decoded_b)
    print(f"    TTFT   {ttft_b:>9,.1f} ms  (only {len(suffix_ids)} suffix tokens — chunk skipped)")
    print(f"    answer {text_b[:80]!r}")
    print()

    # Token agreement
    n_agree = sum(1 for a, b in zip(decoded_a, decoded_b) if a == b)
    agreement = n_agree / max(1, min(len(decoded_a), len(decoded_b)))

    speedup = ttft_a / max(1.0, ttft_b)
    print("────────────────────────────────────────────────────────")
    print(f"  text-RAG TTFT:           {ttft_a:>9,.1f} ms")
    print(f"  hybrid TTFT (warm):      {ttft_b:>9,.1f} ms")
    print(f"  speedup:                 {speedup:>9,.1f}×")
    print(f"  token agreement:         {agreement:>9.1%}  ({n_agree}/{min(len(decoded_a), len(decoded_b))})")
    if "330" in text_a and "330" in text_b:
        print(f"  both answers found '330': True")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--backend", choices=["inproc", "pion"], default="inproc")
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--n-gen", type=int, default=12)
    sys.exit(main(ap.parse_args()))
