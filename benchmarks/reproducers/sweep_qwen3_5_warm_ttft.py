#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# REPRODUCER -- backs a published number.
#
# Claim: Qwen3.5-4B hybrid warm-TTFT sweep: 14.0x / 25.3x / 29.0x at 2K / 4K / 8K
# (2026-10-02, M4 Mac mini, Pion 0.9.1; results/stage1_qwen3_5_prefix_sweep_2026_10_02.json).
# Before the vanilla side prefilled the way mlx-lm's generate_step does, this
# line read 24.77x / 36.0x / 29.5x.
#
# Requires: Apple Silicon + MLX + Qwen3.5-4B-MLX-4bit (~2.5 GB download).
#   Needs a Pion server with --kvcache.
#
# This is research code, moved here from the private research tree so the
# number it produces can be checked. It was not written to be read; it was
# written to answer one question. Expect rough edges, and read the
# prerequisites above before running -- most of the cost is the model
# download, and a run on a loaded machine produces a wrong number rather
# than an error.
# ---------------------------------------------------------------------------
"""gh #61 / gh #76 follow-on — Qwen3.5-4B warm-TTFT scaling sweep at L=1024/2048.

The pre-gh #76 commit (bb3f5e5) capped the sweep at L=512 because longer
prefixes triggered the SSM.PREFIX.FETCH overflow crash on >4 MB blobs. With
the writev fix landed in dc8c0d4 the cap is lifted.

This script:
  1. Loads Qwen3.5-4B once,
  2. For each target prefix length L ∈ {128, 256, 512, 1024, 2048}, pads the
     filler document to hit that token count,
  3. Runs vanilla cold-prefill TTFT vs Pion warm TTFT (3 suffix queries × 6
     decode tokens, identical to the prior sweep),
  4. Writes the consolidated JSON.

Pre-req: `./pion-server --kvcache --metal-attention -w 1`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import mlx.core as mx
import redis
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


PREFIX_HEAD = (
    "You are a careful assistant. Answer using only the document below. "
    "If the document does not state the answer, say 'I don't know'.\n\n"
    "Document: "
)
PREFIX_TAIL = "\n\n"
# Filler paragraph — historical facts, repeated until target token count.
FILLER_PARAGRAPH = (
    "The Eiffel Tower is 330 metres tall and located in Paris. The tower was "
    "completed in 1889 for the World's Fair and held the title of tallest "
    "man-made structure until 1930. Its base is square, 125 metres on each side. "
    "The Empire State Building, located in New York City, was completed in 1931 "
    "and stands 381 metres tall. The Burj Khalifa in Dubai, completed in 2010, "
    "stands 828 metres tall. The Golden Gate Bridge, completed in 1937 in San "
    "Francisco, has a main span of 1280 metres. The Great Wall of China stretches "
    "across northern China for over 13,000 miles and was built primarily during "
    "the Ming dynasty between 1368 and 1644. The Statue of Liberty was a gift "
    "from France, dedicated in 1886, standing 93 metres from ground to torch. "
    "The Colosseum in Rome was completed in 80 AD under Emperor Titus and could "
    "hold 50,000 to 80,000 spectators. Machu Picchu, in Peru, was built in the "
    "15th century at an elevation of 2,430 metres above sea level. "
)

SUFFIXES = [
    "Question: How tall is the Eiffel Tower?\nAnswer:",
    "Question: When was the Empire State Building completed?\nAnswer:",
    "Question: Where is the Burj Khalifa located?\nAnswer:",
]


def build_prefix_text(tok, target_tokens: int) -> tuple[str, int]:
    """Pad filler paragraph until prefix encodes to ≥ target_tokens; return
    the prefix text and the actual token count."""
    n = 1
    while True:
        text = PREFIX_HEAD + (FILLER_PARAGRAPH * n) + PREFIX_TAIL
        ids = tok.encode(text)
        if len(ids) >= target_tokens:
            # Trim by re-tokenizing on truncated text. Easier: keep the first
            # `target_tokens` ids and decode back; or just accept the slight
            # over-shoot. We accept overshoot (same convention as gh #61
            # bench: the original 174-token "128" data point was an upper
            # bound). For exactness, slice the id list and decode.
            ids = ids[:target_tokens]
            return tok.decode(ids), target_tokens
        n += 1


def serialize_layer(c):
    arrays = {}
    meta_bits = []
    ctype = type(c).__name__
    # Attributes, not `c.state`: mlx-lm 0.32 widened `state` with scalars and
    # made a KVCache's `state` return its step-padded buffers.
    if ctype == "KVCache":
        k, v = c.keys[..., : c.offset, :], c.values[..., : c.offset, :]
        mx.eval(k, v)
        arrays["k"] = k
        arrays["v"] = v
        meta_bits.append("type=KVCache")
        meta_bits.append(f"offset={c.offset}")
    elif ctype == "ArraysCache":
        for i, a in enumerate(c.cache):
            if a is not None:
                mx.eval(a)
                arrays[f"a{i}"] = a
        meta_bits.append("type=ArraysCache")
        meta_bits.append(f"narr={len(c.cache)}")
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
        c.keys, c.values = arrays["k"], arrays["v"]
        c.offset = int(meta["offset"])
    elif meta.get("type") == "ArraysCache":
        n = int(meta["narr"])
        c.cache = [arrays.get(f"a{i}") for i in range(n)]


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


def run_one_L(model, tok, r, L: int, session_id: str, n_gen: int):
    print(f"\n===== L={L} tokens =====")
    prefix_text, actual_L = build_prefix_text(tok, L)
    prefix_ids = tok.encode(prefix_text)
    print(f"  prefix tokens: {len(prefix_ids)} (target {L})")
    r.execute_command("SSM.PREFIX.DROP", session_id)

    # === A: vanilla cold-prefill per query ===
    print("  [A] vanilla cold prefill per query")
    vanilla_ttfts = []
    vanilla_decodeds = []
    for s_idx, suf in enumerate(SUFFIXES):
        suffix_ids = tok.encode(suf)
        full_ids = prefix_ids + suffix_ids
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        _, first = greedy_first_token(model, full_ids, cache)
        ttft = (time.perf_counter() - t0) * 1000
        decoded = greedy_continue(model, first, n_gen, cache)
        vanilla_ttfts.append(ttft)
        vanilla_decodeds.append(decoded)
        print(f"     Q{s_idx+1}: TTFT={ttft:>7.0f} ms")

    # === B: Pion warm — one-time prefill+ship, per-query fetch+suffix ===
    print("  [B] Pion warm — one-time prefill, fetch per query")
    t0 = time.perf_counter()
    cache_prefix = make_prompt_cache(model)
    model(mx.array([prefix_ids]), cache=cache_prefix)
    mx.eval([c.state for c in cache_prefix])    # the cache is what ships; no logits needed
    total_bytes = 0
    max_blob = 0
    for i, c in enumerate(cache_prefix):
        blob = serialize_layer(c)
        total_bytes += len(blob)
        max_blob = max(max_blob, len(blob))
        r.execute_command("SSM.PREFIX.STORE", session_id, str(i), blob)
    setup_ms = (time.perf_counter() - t0) * 1000
    print(f"     one-time setup: {setup_ms:.0f} ms (shipped {total_bytes/1e6:.1f} MB, "
          f"max layer {max_blob/1e6:.2f} MB)")

    pion_ttfts = []
    pion_decodeds = []
    for s_idx, suf in enumerate(SUFFIXES):
        suffix_ids = tok.encode(suf)
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        for i, c in enumerate(cache):
            blob = r.execute_command("SSM.PREFIX.FETCH", session_id, str(i))
            deserialize_into(c, blob)
        _, first = greedy_first_token(model, suffix_ids, cache)
        ttft = (time.perf_counter() - t0) * 1000
        decoded = greedy_continue(model, first, n_gen, cache)
        pion_ttfts.append(ttft)
        pion_decodeds.append(decoded)
        print(f"     Q{s_idx+1}: TTFT={ttft:>7.0f} ms")

    # === per-L summary ===
    speedups, agreements = [], []
    for i in range(len(SUFFIXES)):
        vt, pt = vanilla_ttfts[i], pion_ttfts[i]
        speedup = vt / pt if pt > 0 else 0
        agree = sum(1 for a, b in zip(vanilla_decodeds[i], pion_decodeds[i]) if a == b)
        speedups.append(speedup)
        agreements.append(agree)
    mean_v = sum(vanilla_ttfts) / len(vanilla_ttfts)
    mean_p = sum(pion_ttfts) / len(pion_ttfts)
    mean_speedup = sum(speedups) / len(speedups)
    total_agree = sum(agreements)
    total_possible = len(SUFFIXES) * n_gen
    print(f"  → vanilla {mean_v:.0f} ms / pion warm {mean_p:.0f} ms / "
          f"speedup {mean_speedup:.2f}× / agreement {total_agree}/{total_possible}")

    r.execute_command("SSM.PREFIX.DROP", session_id)
    return {
        "vanilla_ms": round(mean_v, 1),
        "pion_ms": round(mean_p, 1),
        "speedup": round(mean_speedup, 2),
        "agreement": f"{total_agree}/{total_possible}",
        "setup_ms": round(setup_ms, 1),
        "shipped_mb": round(total_bytes / 1e6, 1),
        "max_layer_mb": round(max_blob / 1e6, 2),
        "actual_prefix_tokens": len(prefix_ids),
    }


def main(args) -> int:
    print(f"loading {args.model}...")
    model, tok = load(args.model)
    n_gdn = sum(1 for L in model.layers if hasattr(L, "linear_attn"))
    n_attn = sum(1 for L in model.layers if hasattr(L, "self_attn"))
    print(f"  {len(model.layers)} layers ({n_gdn} GatedDeltaNet + {n_attn} Qwen3NextAttention)")

    r = redis.Redis(host=args.pion_host, port=args.pion_port, decode_responses=False)
    r.ping()

    # Warmup (kernel JIT compile)
    sample = tok.encode("warmup")
    _ = greedy_first_token(model, sample, make_prompt_cache(model))
    mx.clear_cache()

    results = {}
    for L in args.target_tokens:
        try:
            results[str(L)] = run_one_L(model, tok, r, L, args.session_id, args.n_gen)
        except Exception as e:
            print(f"  L={L} FAILED: {e}")
            results[str(L)] = {"error": str(e)}
            if args.stop_on_error:
                break

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--target-tokens", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--session-id", default="qwen3_5_sweep_v2")
    ap.add_argument("--out", default="stage1_qwen3_5_prefix_sweep_v2.json")
    ap.add_argument("--stop-on-error", action="store_true")
    sys.exit(main(ap.parse_args()))
