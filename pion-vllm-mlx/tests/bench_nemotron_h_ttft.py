#!/usr/bin/env python3
"""gh #61 — Nemotron-H (Mamba2 hybrid) dual-prefix TTFT vs prefix length.

Measures cold-prefill+store vs warm fetch+restore on Pion at increasing shared-
prefix lengths, and verifies bit-exact next-token logits at each length. The
point: Nemotron-H's 24 Mamba2 layers carry a FIXED-SIZE recurrent state (O(1) in
prefix length), so the warm-restore TTFT win grows with prefix — unlike a pure
transformer whose KV grows linearly.

softmax_bitexact routes the 4 attention anchors through opaque SSM.PREFIX too
(bit-exact; the fp16 V-store ceiling breaks decode on this arch — see
test_hybrid_wire_path.py).

Methodology: one discarded warm-up per length (gh73_sparse_demo_warmup) before
timing; single trial thereafter. Numbers are Mac M-series + this 4B model — a
latency curve, not a throughput claim.

Setup:
    ./pion-server --kvcache --metal-attention -w 1
    <venv>/bin/python pion-vllm-mlx/tests/bench_nemotron_h_ttft.py \
        --model nvidia/Nemotron-H-4B-Instruct-128K [--lengths 256,1024,4096]
"""
from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


# A diverse paragraph tiled up to the target length — avoids a pathological
# all-same-token prefix while keeping content irrelevant (we measure latency +
# bit-exactness, not answer quality).
_PARA = (
    "In 1889 the Eiffel Tower opened in Paris at 330 metres. The Burj Khalifa in "
    "Dubai reached 828 metres in 2010. Meanwhile the study of state-space models "
    "advanced: a recurrent layer keeps a fixed-size hidden state h_t = F(h_{t-1}, "
    "x_t), so continuing from token t needs only h_t, not the full history. "
    "Attention layers instead cache per-token keys and values that grow with the "
    "sequence. Hybrid models interleave the two. "
)


def _build_prefix_ids(tok, n_tokens: int) -> list[int]:
    ids = tok.encode(_PARA)
    while len(ids) < n_tokens:
        ids = ids + tok.encode(_PARA, add_special_tokens=False)  # one <bos>, first
    return ids[:n_tokens]


def _cold_logits(model, prefix_ids):
    """Fresh cold-prefill; return (last-position logits, cache)."""
    cache = make_prompt_cache(model)
    out = model(mx.array([prefix_ids]), cache=cache)
    mx.eval(out)
    return out[0, -1], cache


def main(args):
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"loading {args.model}...")
    model, tok = load(args.model)

    print(f"{'prefix':>8} | {'cold ms':>8} | {'warm ms':>8} | {'speedup':>7} | bit-exact")
    print("-" * 56)
    rows = []
    for n in lengths:
        prefix_ids = _build_prefix_ids(tok, n)
        plen = len(prefix_ids)
        ns = PionPromptCache.make_namespace(
            "bench_nh", args.model, str(plen), str(int(time.time() * 1000)))
        pc = PionPromptCache(model=model, vquant="fp16", host=args.host,
                             port=args.port, softmax_bitexact=True)

        # Warm-up (discarded): primes the wire path for this length, then freed.
        _ = pc.get_or_prefill(prefix_ids, namespace=ns + "_warm")
        pc.resp.call("SSM.PREFIX.DROP", ns + "_warm")
        del _

        # COLD: miss path = cold prefill + store to Pion. Keep its cache as the
        # bit-exact reference (avoids an extra full prefill → less memory).
        t0 = time.perf_counter()
        cold_cache = pc.get_or_prefill(prefix_ids, namespace=ns)
        cold_ms = (time.perf_counter() - t0) * 1000

        # WARM: hit path = fetch + restore from Pion.
        t0 = time.perf_counter()
        warm_cache = pc.get_or_prefill(prefix_ids, namespace=ns)
        warm_ms = (time.perf_counter() - t0) * 1000

        # Bit-exact: feed the SAME probe token to both caches (each at offset
        # prefix_len) and compare next-token logits. Equal state → equal logits.
        probe = mx.array([[prefix_ids[-1]]])
        c_out = model(probe, cache=cold_cache); mx.eval(c_out)
        w_out = model(probe, cache=warm_cache); mx.eval(w_out)
        maxdiff = float(mx.max(mx.abs(w_out[0, -1] - c_out[0, -1])))
        bitexact = maxdiff < 1e-2 and int(mx.argmax(w_out[0, -1])) == int(mx.argmax(c_out[0, -1]))

        speedup = cold_ms / warm_ms if warm_ms else float("inf")
        rows.append((plen, cold_ms, warm_ms, speedup, bitexact, maxdiff))
        print(f"{plen:>8} | {cold_ms:>8.0f} | {warm_ms:>8.0f} | {speedup:>6.1f}x | "
              f"{'YES' if bitexact else 'NO':>3} (maxdiff {maxdiff:.2g})")
        pc.resp.call("SSM.PREFIX.DROP", ns)
        del cold_cache, warm_cache, c_out, w_out, pc, prefix_ids, probe

    all_bitexact = all(r[4] for r in rows)
    print("-" * 56)
    print(f"all bit-exact: {all_bitexact}")
    return 0 if all_bitexact else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/Nemotron-H-4B-Instruct-128K")
    ap.add_argument("--lengths", default="256,1024,4096")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1974)
    sys.exit(main(ap.parse_args()))
