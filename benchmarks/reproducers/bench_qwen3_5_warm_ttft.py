#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# REPRODUCER -- backs a published number.
#
# Claim: Single-point version of the Qwen3.5-4B hybrid warm-TTFT measurement.
#
# Requires: Apple Silicon + MLX + Qwen3.5-4B-MLX-4bit. Needs a Pion server with --kvcache.
#
# This is research code, moved here from the private research tree so the
# number it produces can be checked. It was not written to be read; it was
# written to answer one question. Expect rough edges, and read the
# prerequisites above before running -- most of the cost is the model
# download, and a run on a loaded machine produces a wrong number rather
# than an error.
# ---------------------------------------------------------------------------
"""gh #61 Path-2 / gh #66 — Qwen3.5-4B hybrid warm-TTFT speedup benchmark.

The value-side companion to `spike_qwen3_5_cross_process_wire.py` (which proved
bit-equality). This script measures the *TTFT speedup* a multi-query workload
gets from prefix-sharing through Pion's wire surface, on a hybrid Mamba-style
model (24 GatedDeltaNet linear-attn layers + 8 Qwen3NextAttention softmax
layers).

Workload: one shared prefix (system prompt + RAG-style document) + N suffix
queries that all use the same prefix. The canonical multi-turn-agent or
RAG-with-shared-system shape.

Two paths:
  A) Vanilla mlx-lm — cold prefill of (prefix + suffix) for each query.
  B) Pion warm — prefill prefix ONCE, ship to Pion via SSM.PREFIX.STORE,
     then each query fetches via SSM.PREFIX.FETCH and forwards only its
     suffix.

Setup:
    ./pion-server --kvcache --metal-attention -w 1
    python3 benchmarks/reproducers/bench_qwen3_5_warm_ttft.py

Result on 2026-10-02 (M4 Mac mini, Pion 0.9.1, Qwen3.5-4B-MLX-4bit, 174-token
prefix, 3 queries), vanilla prefilled the way mlx-lm's generate_step does:
    Vanilla cold per-query: ~470 ms TTFT
    Pion warm per-query:    ~285 ms TTFT
    Mean speedup:           1.67×
    Token agreement:        100% (8/8, all 3 queries)

That is below the 2x this script requires before it exits 0. The bar was set
against the 2026-05-13 run, which reported 3.92x with a vanilla side that also
computed logits at every prompt position. At 174 tokens there is little
prefill to skip; sweep_qwen3_5_warm_ttft.py has 2K-8K (14x / 25x / 29x).

Why this matters:
- Pion carries both layer types of a hybrid Mamba+Transformer model across
  processes: the linear layers' state through SSM.PREFIX.*, the softmax
  layers' K/V alongside it, bit-equal on replay.

Caveats:
- Spike uses uniform SSM.PREFIX.* path for both layer types. Production-shape
  routes softmax K/V through KV.PREFIX.REGISTER + V.STOREBATCH so the substrate
  can also serve attention against quantized V. Bit-equality result is the same.
- Serialization is mx.save_safetensors per layer (bf16/fp16 dtype preserved).
- Single-host loopback wire; multi-host wins would be measured separately.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import mlx.core as mx
import redis
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


PREFIX_DEFAULT = (
    "You are a careful assistant. Answer using only the document below. "
    "If the document does not state the answer, say 'I don't know'.\n\n"
    "Document: The Eiffel Tower is 330 metres tall and located in Paris. "
    "The tower was completed in 1889 for the World's Fair and held the title of "
    "tallest man-made structure until 1930. Its base is square, 125 metres on each side. "
    "The Empire State Building, located in New York City, was completed in 1931 "
    "and stands 381 metres tall. The Burj Khalifa in Dubai, completed in 2010, "
    "stands 828 metres tall. The Golden Gate Bridge, completed in 1937 in San Francisco, "
    "has a main span of 1280 metres.\n\n"
)

SUFFIXES_DEFAULT = [
    "Question: How tall is the Eiffel Tower?\nAnswer:",
    "Question: When was the Empire State Building completed?\nAnswer:",
    "Question: Where is the Burj Khalifa located?\nAnswer:",
]


def serialize_layer(c):
    arrays = {}
    meta_bits = []
    ctype = type(c).__name__
    if ctype == "KVCache":
        k, v = c.state
        mx.eval(k, v)
        arrays["k"] = k
        arrays["v"] = v
        meta_bits.append("type=KVCache")
        meta_bits.append(f"offset={c.offset}")
    elif ctype == "ArraysCache":
        for i, a in enumerate(c.state):
            if a is not None:
                mx.eval(a)
                arrays[f"a{i}"] = a
        meta_bits.append("type=ArraysCache")
        meta_bits.append(f"narr={len(c.state)}")
    meta = "|".join(meta_bits)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
    mx.save_safetensors(path, arrays, metadata={"meta": meta})
    with open(path, "rb") as f:
        blob = f.read()
    os.unlink(path)
    return blob


def deserialize_into(c, blob):
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(blob)
        path = f.name
    arrays = mx.load(path)
    _, md = mx.load(path, return_metadata=True)
    os.unlink(path)
    meta = dict(p.split("=", 1) for p in md.get("meta", "").split("|") if "=" in p)
    if meta.get("type") == "KVCache":
        c.state = (arrays["k"], arrays["v"])
        c.offset = int(meta["offset"])
    elif meta.get("type") == "ArraysCache":
        n = int(meta["narr"])
        c.state = [arrays.get(f"a{i}") for i in range(n)]


def greedy_first_token(model, ids, cache):
    """First token after `ids`, prefilled the way mlx_lm.generate_step does.

    Every token but the last runs in 2,048-token chunks with only the cache
    state evaluated; the last token alone gives the logits. Until 2026-10-02
    this evaluated one forward's logits over every position, which no
    generation computes, so the vanilla side was too slow and the speedup
    too high.
    """
    x = mx.array([ids])
    done, n = 0, x.shape[1]
    while n - done > 1:
        step = min(2048, n - done - 1)
        model(x[:, done:done + step], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()       # as generate_step does after each prefill chunk
        done += step
    out = model(x[:, done:], cache=cache)
    mx.eval(out)
    return out, int(mx.argmax(out[0, -1]).item())


def greedy_continue(model, first_tok, n, cache):
    decoded = [first_tok]
    tok_id = first_tok
    for _ in range(n - 1):
        out = model(mx.array([[tok_id]]), cache=cache)
        mx.eval(out)
        tok_id = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok_id)
    return decoded


def main(args) -> int:
    print(f"loading {args.model}...")
    model, tok = load(args.model)
    n_gdn = sum(1 for L in model.layers if hasattr(L, "linear_attn"))
    n_attn = sum(1 for L in model.layers if hasattr(L, "self_attn"))
    print(f"  {len(model.layers)} layers ({n_gdn} GatedDeltaNet + {n_attn} Qwen3NextAttention)")

    prefix_ids = tok.encode(args.prefix)
    print(f"  prefix: {len(prefix_ids)} tokens; {len(args.suffixes)} suffix queries")

    r = redis.Redis(host=args.pion_host, port=args.pion_port, decode_responses=False)
    r.ping()
    r.execute_command("SSM.PREFIX.DROP", args.session_id)

    # Warmup (kernel JIT compile)
    _ = greedy_first_token(model, prefix_ids[:32], make_prompt_cache(model))
    mx.clear_cache()

    # === A: vanilla cold-prefill per query ===
    print("[A] vanilla — cold prefill of (prefix + suffix) per query")
    vanilla_ttfts = []
    vanilla_decodeds = []
    for s_idx, suf in enumerate(args.suffixes):
        suffix_ids = tok.encode(suf)
        full_ids = prefix_ids + suffix_ids
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        _, first = greedy_first_token(model, full_ids, cache)
        ttft = (time.perf_counter() - t0) * 1000
        decoded = greedy_continue(model, first, args.n_gen, cache)
        vanilla_ttfts.append(ttft)
        vanilla_decodeds.append(decoded)
        print(f"   Q{s_idx+1}: TTFT={ttft:>6.0f} ms   answer={tok.decode(decoded)!r}")

    # === B: Pion warm — one-time prefill+ship, per-query fetch+suffix ===
    print("[B] Pion warm — prefill ONCE, ship via wire, fetch per query")
    t0 = time.perf_counter()
    cache_prefix = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=cache_prefix)
    mx.eval([c.state for c in cache_prefix])    # the cache is what ships; no logits needed
    total_bytes = 0
    for i, c in enumerate(cache_prefix):
        blob = serialize_layer(c)
        total_bytes += len(blob)
        r.execute_command("SSM.PREFIX.STORE", args.session_id, str(i), blob)
    setup_ms = (time.perf_counter() - t0) * 1000
    print(f"   one-time setup: {setup_ms:.0f} ms (prefill + ship {total_bytes:,} bytes), amortized")

    pion_ttfts = []
    pion_decodeds = []
    for s_idx, suf in enumerate(args.suffixes):
        suffix_ids = tok.encode(suf)
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        for i, c in enumerate(cache):
            blob = r.execute_command("SSM.PREFIX.FETCH", args.session_id, str(i))
            deserialize_into(c, blob)
        _, first = greedy_first_token(model, suffix_ids, cache)
        ttft = (time.perf_counter() - t0) * 1000
        decoded = greedy_continue(model, first, args.n_gen, cache)
        pion_ttfts.append(ttft)
        pion_decodeds.append(decoded)
        print(f"   Q{s_idx+1}: TTFT={ttft:>6.0f} ms   answer={tok.decode(decoded)!r}")

    # === Summary ===
    print()
    print("===== summary =====")
    speedups, agreements = [], []
    for i, suf in enumerate(args.suffixes):
        vt, pt = vanilla_ttfts[i], pion_ttfts[i]
        speedup = vt / pt if pt > 0 else 0
        agree = sum(1 for a, b in zip(vanilla_decodeds[i], pion_decodeds[i]) if a == b)
        speedups.append(speedup)
        agreements.append(agree)
        print(f"   Q{i+1}: vanilla {vt:>6.0f} ms / pion warm {pt:>6.0f} ms / speedup {speedup:>4.1f}× / agreement {agree}/{args.n_gen}")
    mean_speedup = sum(speedups) / len(speedups)
    mean_agreement = sum(agreements) / (len(args.suffixes) * args.n_gen)
    print(f"   MEAN speedup: {mean_speedup:.2f}×   MEAN token agreement: {mean_agreement*100:.1f}%")
    print(f"   one-time prefix-publish cost: {setup_ms:.0f} ms (amortized across {len(args.suffixes)} queries)")

    r.execute_command("SSM.PREFIX.DROP", args.session_id)
    return 0 if mean_speedup >= 2.0 and mean_agreement >= 0.95 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--prefix", default=PREFIX_DEFAULT)
    ap.add_argument("--suffixes", nargs="+", default=SUFFIXES_DEFAULT)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--session-id", default="qwen3_5_warm_demo")
    sys.exit(main(ap.parse_args()))
