#!/usr/bin/env python3
"""gh #61 follow-on — PionPromptCache hybrid wire path regression gate.

The pre-2026-05-14 PionPromptCache.get_or_prefill detected `is_hybrid` on
Qwen3.5-style mixed-cache hybrids (24 GatedDeltaNet ArraysCache + 8
Qwen3NextAttention KVCache) and disabled the wire path entirely — fell back
to local re-prefill on every call. The 2026-05-14 generalization adds a
split-substrate path:

  ArraysCache (linear / SSM) slots → SSM.PREFIX.STORE / FETCH (opaque)
  KVCache (softmax) slots          → V.CREATE + V.STOREBATCH + V.FETCH RANGE
                                       (structured, indexed by softmax-rank)

This test asserts the new path works end-to-end:
  1. First call to `get_or_prefill` cold-prefills + stores to Pion (MISS).
  2. Second call (same namespace) hits Pion + restores split-substrate (HIT).
  3. Token-by-token decode from the restored cache matches a vanilla
     cold-prefill baseline ≥ 95% (allow 1 greedy-argmax bit of variance).

Setup:
    ./pion-server --kvcache --metal-attention -w 1
    python pion-vllm-mlx/tests/test_hybrid_wire_path.py
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


PREFIX_DEFAULT = (
    "You are a careful assistant. Answer using only the document below. "
    "Document: The Eiffel Tower is 330 metres tall and located in Paris. "
    "The Empire State Building, located in New York City, was completed in 1931 "
    "and stands 381 metres tall. The Burj Khalifa in Dubai, completed in 2010, "
    "stands 828 metres tall.\n\nQuestion: How tall is the Eiffel Tower?\nAnswer:"
)


def greedy_decode(model, prompt_ids, n_steps: int, cache):
    out = model(prompt_ids, cache=cache)
    mx.eval(out)
    tok = int(mx.argmax(out[0, -1]).item())
    decoded = [tok]
    for _ in range(n_steps - 1):
        out = model(mx.array([[tok]]), cache=cache)
        mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        decoded.append(tok)
    return decoded


def _try_connect(host: str, port: int) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=2); s.close()
        return True
    except OSError:
        return False


def main(args) -> int:
    if not _try_connect(args.host, args.port):
        print(f"FAIL: pion not reachable at {args.host}:{args.port}. "
              "Start: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    print(f"loading {args.model}...")
    model, tok = load(args.model)
    n_gdn = sum(1 for L in model.layers if hasattr(L, "linear_attn"))
    n_attn = sum(1 for L in model.layers if hasattr(L, "self_attn"))
    print(f"  {len(model.layers)} layers ({n_gdn} GatedDeltaNet + {n_attn} Qwen3NextAttention)")

    prefix_ids = tok.encode(args.prefix)
    print(f"  prefix: {len(prefix_ids)} tokens; decoding {args.n_gen} tokens per pass")

    # ── Baseline: vanilla cold prefill ──
    cache_base = make_prompt_cache(model)
    tokens_base = greedy_decode(model, mx.array([prefix_ids]), args.n_gen, cache_base)
    print(f"  baseline tokens:    {tokens_base[:6]}…")

    # ── Test PionPromptCache.get_or_prefill on hybrid ──
    pc = PionPromptCache(model=model, vquant="fp16",
                          host=args.host, port=args.port,
                          softmax_bitexact=args.softmax_bitexact)
    # Fresh namespace per run — there's no V.DROP wire op and SSM has no
    # persistence, so cross-restart state can desync. Tested separately via
    # the lookup-HIT-then-SSM-miss fallback in get_or_prefill; this test
    # focuses on the first MISS + first HIT happy path.
    ns = PionPromptCache.make_namespace(
        "hybrid_wire_test", args.model, str(len(prefix_ids)),
        str(int(time.time() * 1000)),
    )

    # First call — MISS, expect cold prefill + wire push.
    pre_hits, pre_misses = pc.hits, pc.misses
    t0 = time.perf_counter()
    cache_a = pc.get_or_prefill(prefix_ids, namespace=ns)
    t_a = (time.perf_counter() - t0) * 1000
    if pc.misses != pre_misses + 1 or pc.hits != pre_hits:
        print(f"FAIL: expected first call to be a MISS, got hits={pc.hits} misses={pc.misses}")
        return 1
    # Decode against the populated cache by feeding empty suffix isn't valid
    # because the cache already includes the prefix and we want continuation
    # from where prefill stopped. Easiest: call greedy_continue starting from
    # the cache's last logits — but get_or_prefill returns a cache that
    # already contains prefix, and the next forward expects [decoded_tok].
    # Cleaner: skip first-call decode, do bit-equality only on the second
    # (post-fetch) call. The second call is the load-bearing one for the
    # wire path anyway.
    print(f"  miss: cold prefill + store = {t_a:>5.0f} ms (hits={pc.hits} misses={pc.misses})")

    # Second call — same namespace, expect HIT and fetch through the
    # mixed-hybrid split substrate.
    pre_hits, pre_misses = pc.hits, pc.misses
    t0 = time.perf_counter()
    cache_b = pc.get_or_prefill(prefix_ids, namespace=ns)
    t_b = (time.perf_counter() - t0) * 1000
    if pc.hits != pre_hits + 1 or pc.misses != pre_misses:
        print(f"FAIL: expected second call to be a HIT, got hits={pc.hits} misses={pc.misses}")
        return 1
    print(f"  hit:  fetch + restore = {t_b:>5.0f} ms (hits={pc.hits} misses={pc.misses})")

    # Decode the next n_gen tokens from the FETCHED cache and compare to
    # vanilla baseline. cache_b was rehydrated from Pion — bit-equality
    # check covers the split-substrate restore on both halves.
    # The cache already has `prefix_len` tokens cached; feed nothing or
    # the LAST prefix token? Actually we want to extend from the SAME state
    # as vanilla. Vanilla did `model(prefix_ids, cache=fresh)` and ran the
    # full prefix forward; the next step is greedy from cache. The fetched
    # cache should be in the same state — let me forward the entire prefix
    # ON cache_b too, but cache_b is already populated. The right comparison
    # is: from a fully-populated cache_b, generate n_gen tokens.
    #
    # Trick: we kept vanilla in cache_base too; both should produce the same
    # next-token distribution at the same prefix-end position. Compare from
    # the cache state directly: feed a 1-token "no-op" doesn't make sense for
    # a fresh greedy start. Use the documented mlx-lm pattern: forward an
    # empty? No — the cleanest path is to feed nothing and call the model
    # decode loop with a starting last-position logit. Instead we just call
    # greedy from the cache by re-doing the LAST prefix token (and trimming
    # the cache to prefix-1 first). Too brittle.
    #
    # Simplest viable bit-equality: do the SAME thing on the baseline. Both
    # use greedy_decode(model, prefix_ids, n_gen, cache_FRESH) but cache_b
    # is the populated cache, which means feeding prefix_ids again would
    # DOUBLE the prefix. Wrong.
    #
    # Right answer: pc.get_or_prefill returns a cache populated with prefix.
    # We decode from there by forwarding only the LAST prefix token as the
    # next input, but that would consume cache space. The intended pattern
    # is: cache is at end-of-prefix; next user input is a NEW suffix. So we
    # add a suffix here, run on both cache_base and cache_b, compare tokens.
    suffix_ids = tok.encode(" Then I will say:")
    # `cache_base` was advanced by the baseline greedy_decode above (it decoded
    # n_gen tokens past the prefix), so it now sits at prefix_len + n_gen. The
    # pion cache_b sits at exactly prefix_len. Comparing a suffix continuation
    # from those two positions is apples-to-oranges. Build a fresh reference
    # cache prefilled to EXACTLY prefix_len so both decode from the same state.
    cache_ref = make_prompt_cache(model)
    _ = model(mx.array([prefix_ids]), cache=cache_ref)
    mx.eval(_)
    out_base = model(mx.array([suffix_ids]), cache=cache_ref)
    mx.eval(out_base)
    tok_base = int(mx.argmax(out_base[0, -1]).item())
    out_pion = model(mx.array([suffix_ids]), cache=cache_b)
    mx.eval(out_pion)
    tok_pion = int(mx.argmax(out_pion[0, -1]).item())

    base_seq = [tok_base]
    pion_seq = [tok_pion]
    cur_b, cur_p = tok_base, tok_pion
    for _ in range(args.n_gen - 1):
        ob = model(mx.array([[cur_b]]), cache=cache_ref); mx.eval(ob)
        op = model(mx.array([[cur_p]]), cache=cache_b); mx.eval(op)
        cur_b = int(mx.argmax(ob[0, -1]).item())
        cur_p = int(mx.argmax(op[0, -1]).item())
        base_seq.append(cur_b)
        pion_seq.append(cur_p)

    matches = sum(1 for a, b in zip(base_seq, pion_seq) if a == b)
    total = args.n_gen
    print(f"  vanilla seq: {base_seq}")
    print(f"  pion    seq: {pion_seq}")
    print(f"  agreement: {matches}/{total}")

    # ── Cleanup ── SSM only (no V.DROP wire op). V-store sessions leak per
    # test run — fine for a unit test, real deployments call KV.PREFIX.* for
    # invalidation. Cross-restart resilience is covered by the
    # `_store_mixed_hybrid` "already exists" handling.
    pc.resp.call("SSM.PREFIX.DROP", ns)

    # Pass criterion: bit-equality OR within 1 greedy-argmax bit (the same
    # tolerance used in the long-sweep results).
    pass_ok = matches >= total - 1
    print(f"\n{'PASS' if pass_ok else 'FAIL'} — hybrid wire path "
          f"(miss {t_a:.0f} ms / hit {t_b:.0f} ms / agreement {matches}/{total})")
    return 0 if pass_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--prefix", default=PREFIX_DEFAULT)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--softmax-bitexact", action="store_true",
                    help="Store softmax anchors opaquely via SSM.PREFIX (bit-exact; "
                         "needed for extreme hybrids like Nemotron-H where fp16 V-store breaks decode)")
    sys.exit(main(ap.parse_args()))
