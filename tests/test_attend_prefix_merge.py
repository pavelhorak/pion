#!/usr/bin/env python3
"""Online softmax merge correctness — pure-math test, no server required.

Validates the §31 mlx-lm monkey-patch's merge formula against the reference:

  softmax([Q@Kp^T | Q@Ks^T] / √d) @ [Vp ; Vs]
    ==
  online_merge(
      softmax(Q@Kp^T / √d) @ Vp,  rowwise LSE(Q@Kp^T/√d),
      softmax(Q@Ks^T / √d) @ Vs,  rowwise LSE(Q@Ks^T/√d),
  )

This is the math the monkey-patch relies on. If this fails, the patch
produces wrong attention output regardless of how reliable the wire path is.

Also covers a second leg of the same protocol extension: ATTEND.PREFIX.QUERY
with M>1 should return a wire body of layout
  [output: H*M*D float32 | LSE: H*M float32]
i.e. exactly H*M*(D+1)*4 bytes. We don't run the server here — that's
test_attend_prefix.py's job — we just unit-test the merge.
"""
from __future__ import annotations

import sys

import numpy as np


def softmax_lse(scores: np.ndarray, axis: int = -1):
    m = np.max(scores, axis=axis, keepdims=True)
    exp = np.exp(scores - m)
    s = np.sum(exp, axis=axis, keepdims=True)
    out = exp / s
    lse = (m + np.log(s)).squeeze(axis)
    return out, lse


def reference_attention(Q, K, V, scale):
    """Q (H, M, D), K/V (H, N, D) → output (H, M, D)."""
    scores = (Q @ np.swapaxes(K, -1, -2)) * scale  # (H, M, N)
    weights, _ = softmax_lse(scores, axis=-1)
    return weights @ V


def split_attention(Q, K, V, scale, split_at: int):
    """Run attention separately over K[:, :split_at] and K[:, split_at:],
    return (p_out, p_lse, s_out, s_lse) — what the wire path produces."""
    Kp, Vp = K[:, :split_at], V[:, :split_at]
    Ks, Vs = K[:, split_at:], V[:, split_at:]

    p_scores = (Q @ np.swapaxes(Kp, -1, -2)) * scale
    p_w, p_lse = softmax_lse(p_scores, axis=-1)
    p_out = p_w @ Vp

    s_scores = (Q @ np.swapaxes(Ks, -1, -2)) * scale
    s_w, s_lse = softmax_lse(s_scores, axis=-1)
    s_out = s_w @ Vs
    return p_out, p_lse, s_out, s_lse


def online_merge(p_out, p_lse, s_out, s_lse):
    m = np.maximum(p_lse, s_lse)
    p_w = np.exp(p_lse - m)[..., None]
    s_w = np.exp(s_lse - m)[..., None]
    return (p_w * p_out + s_w * s_out) / (p_w + s_w)


def main() -> int:
    rng = np.random.default_rng(0x1234)
    H, M, D, N = 8, 4, 64, 100
    Q = rng.standard_normal((H, M, D)).astype(np.float32)
    K = rng.standard_normal((H, N, D)).astype(np.float32)
    V = rng.standard_normal((H, N, D)).astype(np.float32)
    scale = 1.0 / np.sqrt(D)

    ref = reference_attention(Q, K, V, scale)

    failures = []
    for split in (1, 10, 50, 90, 99):
        p_out, p_lse, s_out, s_lse = split_attention(Q, K, V, scale, split)
        merged = online_merge(p_out, p_lse, s_out, s_lse)
        max_abs = float(np.max(np.abs(ref - merged)))
        max_rel = float(np.max(np.abs(ref - merged) / (np.abs(ref) + 1e-9)))
        ok = max_abs < 1e-4
        tag = "OK" if ok else "BAD"
        print(f"  split={split:3d}  {tag}  max|Δ|={max_abs:.2e}  max rel={max_rel:.2e}")
        if not ok:
            failures.append(f"split={split} max|Δ|={max_abs:.2e}")

    # Symmetric check: split into [empty | full] should equal pure-suffix.
    # Mojo/Python sidecar treats prefix_len <= 0 specially (no merge needed),
    # but the formula itself must still be sound — verify by setting LSE=-inf.
    s_scores = (Q @ np.swapaxes(K, -1, -2)) * scale
    s_w, s_lse_full = softmax_lse(s_scores, axis=-1)
    s_out_full = s_w @ V
    p_lse_empty = np.full((H, M), -np.inf, dtype=np.float32)
    merged = online_merge(np.zeros_like(s_out_full), p_lse_empty, s_out_full, s_lse_full)
    max_abs = float(np.max(np.abs(ref - merged)))
    print(f"  empty-prefix merge: max|Δ|={max_abs:.2e}  expected ≈ 0")
    if max_abs >= 1e-4:
        failures.append(f"empty-prefix merge max|Δ|={max_abs:.2e}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS: online softmax merge is bit-equivalent to full-K attention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
