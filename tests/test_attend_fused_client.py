"""gh #49: end-to-end client-side test for the fused attention patch.

Synthesizes the call shape `pion_scaled_dot_product_attention` receives from
mlx-lm during a real forward pass, runs it both ways:

  - Patched (gh #49 path): registers prefix in Pion, calls fused kernel.
  - Reference: vanilla mlx-lm scaled_dot_product_attention over the
    concatenated K/V (prefix ‖ suffix) with causal mask.

Asserts cosine ≥ 0.9999 and max|Δ| ≤ 5e-4. Avoids the heavy LLM download.

Server: ./pion-server-dev --kvcache --metal-attention -w 1
"""
from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-vllm-mlx"))

import mlx.core as mx                                                       # noqa: E402
from pion_vllm_mlx.prompt_cache import PionPromptCache                      # noqa: E402
from pion_vllm_mlx.mlx_lm_patch import (                                    # noqa: E402
    PionPrefixCache,
    pion_scaled_dot_product_attention,
)


PORT = int(os.environ.get("PION_PORT", "1974"))


def reference_sdpa(Q: mx.array, K_full: mx.array, V_full: mx.array,
                   prefix_len: int, scale: float) -> mx.array:
    """Vanilla MLX SDPA over (B=1, H, M, D) Q, (B=1, H_kv, N+S, D) K/V with
    causal mask: prefix fully visible, suffix queries see suffix [0..mq]."""
    B, Hq, M, D = Q.shape
    Hkv = K_full.shape[1]
    rep = Hq // Hkv
    if rep > 1:
        K_full = mx.repeat(K_full, rep, axis=1)
        V_full = mx.repeat(V_full, rep, axis=1)
    NS = K_full.shape[2]
    S_suf = NS - prefix_len
    suf_off = max(0, S_suf - M)
    rows = mx.arange(M)[:, None]
    cols = mx.arange(NS)[None, :]
    # Within suffix (cols >= prefix_len), query row mq sees suffix indices
    # 0..(suf_off + mq); i.e. global cols up to prefix_len + suf_off + mq.
    cap = prefix_len + suf_off + rows
    mask = mx.where(cols > cap, -1e9, 0.0)
    scores = mx.matmul(Q, K_full.transpose(0, 1, 3, 2)) * scale
    scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, V_full)


def cos(a: mx.array, b: mx.array) -> float:
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    rng = np.random.default_rng(42)
    H_kv, H_q, D = 8, 32, 64           # Llama-3.2-1B
    N_pref = 96
    S_suf = 6
    M = 4
    namespace = "gh49_e2e_test"
    prefix_len = N_pref

    K_pref = (rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1)
    V_pref = (rng.standard_normal((H_kv, N_pref, D)).astype(np.float32) * 0.1)
    K_suf  = (rng.standard_normal((H_kv, S_suf,  D)).astype(np.float32) * 0.1)
    V_suf  = (rng.standard_normal((H_kv, S_suf,  D)).astype(np.float32) * 0.1)
    Q      = (rng.standard_normal((H_q,  M,      D)).astype(np.float32) * 0.1)
    scale = 1.0 / np.sqrt(D)

    # ── 1. PionPromptCache (no model — wire-only ops). ──
    pc = PionPromptCache.__new__(PionPromptCache)
    from pion_vllm_mlx.prompt_cache import _RESP
    pc.resp = _RESP("127.0.0.1", PORT)
    pc._local = {}
    pc._stage2_pushed = set()
    pc.attend_query_ms_total = 0.0
    pc.attend_query_calls = 0
    # A __new__ stub skips __init__, so every attribute the class gains must
    # be added here too — _binary_* (the port+1 lane) arrived after this test
    # and made it AttributeError. Pin the RESP path this test was written for.
    pc._binary_disabled = True
    pc._binary = None
    pc._binary_attempted = False

    # Push prefix to Pion via ATTEND.PREFIX.STORE for layer 0.
    sid = pc._attend_session(namespace)
    rep = pc.resp.call(
        "ATTEND.PREFIX.STORE", sid, "0",
        str(H_kv), str(N_pref), str(D),
        K_pref.tobytes(), V_pref.tobytes(),
    )
    if not rep.startswith(b"+OK"):
        print(f"FAIL: STORE: {rep[:160]!r}")
        return 1

    # ── 2. Run patched attention. ──
    Q_mx = mx.array(Q[None, ...])               # (1, H_q, M, D)
    Ks_mx = mx.array(K_suf[None, ...])           # (1, H_kv, S, D)
    Vs_mx = mx.array(V_suf[None, ...])
    cache = PionPrefixCache(layer_idx=0, namespace=namespace,
                            prompt_cache=pc, prefix_len=prefix_len)
    cache.keys = Ks_mx
    cache.values = Vs_mx
    cache.offset = prefix_len + S_suf

    fused_out = pion_scaled_dot_product_attention(
        Q_mx, Ks_mx, Vs_mx, cache, scale=scale, mask="causal", sinks=None,
    )

    # ── 3. Reference: vanilla SDPA over [prefix | suffix] K/V. ──
    K_full = mx.array(np.concatenate([K_pref, K_suf], axis=1)[None, ...])
    V_full = mx.array(np.concatenate([V_pref, V_suf], axis=1)[None, ...])
    ref_out = reference_sdpa(Q_mx, K_full, V_full, prefix_len, scale)

    cos_v = cos(fused_out, ref_out)
    max_abs = float(np.abs(np.asarray(fused_out) - np.asarray(ref_out)).max())
    print(f"cosine(patched, vanilla MLX SDPA) = {cos_v:.10f}")
    print(f"max|Δ| = {max_abs:.4e}")

    try:
        pc.resp.sock.close()
    except Exception:
        pass
    if cos_v < 0.9999 or max_abs > 5e-4:
        print("FAIL: end-to-end client-side patch parity below threshold")
        return 1
    print("\nPASS — gh #49 client-side patch matches vanilla MLX SDPA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
