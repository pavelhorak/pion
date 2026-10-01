"""gh #49: microbench — fused suffix-SDPA + merge vs legacy 2-step path.

Approximates Stage-2 warm TTFT by running 16 attention calls back-to-back
(stand-in for a 16-layer Llama forward) on a prefix already registered in
Pion. Measures per-layer wall-clock for both paths.

Server: ./pion-server-dev --kvcache --metal-attention -w 1 --no-auto-detect --no-auto-embed
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))

import mlx.core as mx                                                       # noqa: E402
from pion_vllm_mlx.prompt_cache import PionPromptCache, _RESP               # noqa: E402
from pion_vllm_mlx.mlx_lm_patch import (                                    # noqa: E402
    PionPrefixCache,
    pion_scaled_dot_product_attention,
    _legacy_pion_attention,
)


PORT = int(os.environ.get("PION_PORT", "1974"))


def setup_pc() -> PionPromptCache:
    pc = PionPromptCache.__new__(PionPromptCache)
    pc.resp = _RESP("127.0.0.1", PORT)
    pc._local = {}
    pc._stage2_pushed = set()
    pc.attend_query_ms_total = 0.0
    pc.attend_query_calls = 0
    return pc


def main() -> int:
    rng = np.random.default_rng(7)
    # Default: TTFT-prefill regime (M = S_suf = 64 new tokens). At decode-step
    # (M=1, S_suf=1) host suffix-SDPA is trivially small and the win shrinks;
    # the ~50 ms host overhead was measured on
    # the prefill path with many suffix tokens.
    H_kv, H_q, D = int(os.environ.get("H_KV", 8)), int(os.environ.get("H_Q", 32)), int(os.environ.get("D", 64))
    N_pref = int(os.environ.get("N_PREF", 316))
    S_suf  = int(os.environ.get("S_SUF", 64))
    M      = int(os.environ.get("M",     64))
    n_layers = int(os.environ.get("LAYERS", 16))
    namespace = "gh49_bench"
    prefix_len = N_pref
    scale = 1.0 / np.sqrt(D)

    pc = setup_pc()

    # Push 16 layers' worth of prefix.
    K_pref = (rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1)
    V_pref = (rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1)
    sid = pc._attend_session(namespace)
    for li in range(n_layers):
        rep = pc.resp.call(
            "ATTEND.PREFIX.STORE", sid, str(li),
            str(H_kv), str(N_pref), str(D),
            K_pref.tobytes(), V_pref.tobytes(),
        )
        if not rep.startswith(b"+OK"):
            print(f"FAIL: STORE layer {li}: {rep[:160]!r}")
            return 1

    # Build per-layer caches with synthetic suffix K/V.
    K_suf = (rng.standard_normal((H_kv, S_suf, D)).astype(np.float32) * 0.1)
    V_suf = (rng.standard_normal((H_kv, S_suf, D)).astype(np.float32) * 0.1)
    Q     = (rng.standard_normal((H_q,  M,     D)).astype(np.float32) * 0.1)
    Q_mx  = mx.array(Q[None, ...])
    Ks_mx = mx.array(K_suf[None, ...])
    Vs_mx = mx.array(V_suf[None, ...])

    caches = []
    for li in range(n_layers):
        c = PionPrefixCache(layer_idx=li, namespace=namespace,
                            prompt_cache=pc, prefix_len=prefix_len)
        c.keys = Ks_mx
        c.values = Vs_mx
        c.offset = prefix_len + S_suf
        caches.append(c)

    def run_fused() -> None:
        for c in caches:
            o = pion_scaled_dot_product_attention(
                Q_mx, Ks_mx, Vs_mx, c, scale=scale, mask="causal")
            mx.eval(o)

    def run_legacy() -> None:
        for c in caches:
            o = _legacy_pion_attention(Q_mx, Ks_mx, Vs_mx, c, scale, "causal")
            mx.eval(o)

    # Warmup + measure.
    for _ in range(3):
        run_fused()
        run_legacy()

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        run_fused()
    t_fused = (time.perf_counter() - t0) / iters * 1000   # ms / forward

    t0 = time.perf_counter()
    for _ in range(iters):
        run_legacy()
    t_legacy = (time.perf_counter() - t0) / iters * 1000

    print(f"H_q={H_q} H_kv={H_kv} D={D} N_pref={N_pref} S_suf={S_suf} M={M} "
          f"layers={n_layers} iters={iters}")
    print(f"  legacy (2-step suffix SDPA + host merge):  {t_legacy:7.3f} ms/forward "
          f"({t_legacy/n_layers:.3f} ms/layer)")
    print(f"  fused  (gh #49):                           {t_fused:7.3f} ms/forward "
          f"({t_fused/n_layers:.3f} ms/layer)")
    print(f"  speedup: {t_legacy/t_fused:.2f}×, saved {(t_legacy - t_fused):.2f} ms / forward")

    try:
        pc.resp.sock.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
