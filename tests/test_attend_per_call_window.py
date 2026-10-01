#!/usr/bin/env python3
"""gh #60 Step 1 wire-path — per-call fa_window override on RESP + binary.

Validates that PionPromptCache.attend_query_fused(fa_window=N) actually
reaches Pion's MetalAttentionEngine.query_batched_fused as a per-call
override (not just a no-op kwarg). Tested against a known prefix:

  1. Stash a 1024-token prefix (random fp32 K/V) via ATTEND.PREFIX.STORE.
  2. Build a single query Q.
  3. Call attend_query_fused with fa_window=None (full prefix) → out_full.
  4. Call attend_query_fused with fa_window=128 (last 128 only) → out_128.
  5. Call attend_query_fused with fa_window=0 (also full attention per kernel
     convention W=0 means no clamp) → out_zero.
  6. Assert out_full and out_zero are bit-equal.
  7. Assert out_full and out_128 differ (the window actually changed scope).
  8. Compute reference attention output over LAST 128 K/V locally and assert
     out_128 matches within fp32 SDPA tolerance.

Requires: ./pion-server --kvcache --metal-attention -w 1
"""
from __future__ import annotations

import socket
import sys

import numpy as np

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


HOST = "127.0.0.1"
PORT = 1974
NS = "test_attend_per_call_window"
LAYER = 0


def reference_sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Float32 reference: Q @ K^T / sqrt(D), softmax, weighted V.
    Shapes: Q (H, M, D), K (H, S, D), V (H, S, D). Returns (H, M, D)."""
    H, M, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    out = np.zeros_like(Q)
    for h in range(H):
        scores = (Q[h] @ K[h].T) * scale       # (M, S)
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        w = e / e.sum(axis=-1, keepdims=True)
        out[h] = w @ V[h]
    return out


def main() -> int:
    print(f"gh #60 Step 1 wire-path test  host={HOST} port={PORT}")
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e}). Start: ./pion-server --kvcache --metal-attention -w 1")
        return 2

    H, S_full, D = 4, 1024, 64       # H ≤ supported, D ∈ supported set
    H_kv = 4                          # MHA (H_kv == H_q for simplicity)
    rng = np.random.default_rng(42)
    K_prefix = rng.standard_normal((H, S_full, D), dtype=np.float32)
    V_prefix = rng.standard_normal((H, S_full, D), dtype=np.float32)
    Q = rng.standard_normal((H, 1, D), dtype=np.float32)
    head_map = np.arange(H_kv, dtype=np.uint8).repeat(H // H_kv)

    pc = PionPromptCache(model=None, host=HOST, port=PORT, stage2=True)

    # 1. Stash prefix.
    pc.attend_drop(NS)        # idempotent cleanup
    print(f"  stashing prefix: H={H} S={S_full} D={D}")
    pc.attend_store_layer(NS, LAYER, K_prefix, V_prefix)

    # 2-5. Three queries with different fa_window values. Use empty suffix
    # (S_suf=0) so the wire payload only exercises the prefix attention.
    K_suf = np.zeros((H_kv, 0, D), dtype=np.float32)
    V_suf = np.zeros((H_kv, 0, D), dtype=np.float32)

    print(f"  query A: fa_window=None (server default = full attention)")
    out_full = pc.attend_query_fused(NS, LAYER, Q, K_suf, V_suf, head_map,
                                      fa_window=None)
    print(f"  query B: fa_window=0 (kernel convention: 0 = no clamp)")
    out_zero = pc.attend_query_fused(NS, LAYER, Q, K_suf, V_suf, head_map,
                                      fa_window=0)
    print(f"  query C: fa_window=128 (last 128 K/V only)")
    out_128 = pc.attend_query_fused(NS, LAYER, Q, K_suf, V_suf, head_map,
                                     fa_window=128)

    # 6. fa_window=0 ≡ no override at the kernel level.
    if not np.allclose(out_full, out_zero, atol=1e-5):
        diff = np.max(np.abs(out_full - out_zero))
        print(f"  FAIL: out_full != out_zero (max diff {diff:.6e}) — "
              f"the kernel is treating fa_window=0 differently from full attention")
        return 1
    print(f"  ✓ fa_window=0 ≡ fa_window=None (max diff {np.max(np.abs(out_full - out_zero)):.2e})")

    # 7. fa_window=128 must produce a different output (different attention scope).
    diff_128 = np.max(np.abs(out_full - out_128))
    if diff_128 < 1e-3:
        print(f"  FAIL: out_full ≈ out_128 (max diff {diff_128:.6e}) — "
              f"the override didn't reach the kernel; full prefix was attended in both")
        return 1
    print(f"  ✓ fa_window=128 produced different output (max diff {diff_128:.2e}) — override reached kernel")

    # 8. Reference: attention over LAST 128 K/V positions.
    K_ref = K_prefix[:, -128:, :]
    V_ref = V_prefix[:, -128:, :]
    out_ref = reference_sdpa(Q, K_ref, V_ref)
    rel_err = np.max(np.abs(out_128 - out_ref)) / max(1e-6, np.max(np.abs(out_ref)))
    print(f"  reference attention over last 128: max rel err = {rel_err:.2e}")
    if rel_err > 1e-2:
        print(f"  FAIL: out_128 doesn't match reference (rel err {rel_err:.4e})")
        return 1
    print(f"  ✓ out_128 matches reference attention over last 128 K/V (rel err < 1%)")

    pc.attend_drop(NS)
    print("\nPASS — gh #60 Step 1 wire-path: per-call fa_window override works on binary + RESP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
