#!/usr/bin/env python3
"""
MLX vs CPU Attention Benchmark — Phase 2
=========================================

Compares three attention backends:
  1. CPU-only (NumPy, cblas_sgemv via @ operator) — single-head and multi-head
  2. MLX GPU-only (mx.matmul batched) — single-head and multi-head
  3. MLX GPU-only with mx.fast.scaled_dot_product_attention (if available)

Configs: H in {1,8,16,32}, N in {1024,4096,16384,65536[,131072]}, D=128, top_k=32.

Usage:
  python mlx_comparison.py                  # standard configs (up to 16K seq)
  python mlx_comparison.py --full           # include 65K and 128K
  python mlx_comparison.py --markdown       # emit paper-ready markdown tables
  python mlx_comparison.py --full --markdown
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Check for scaled_dot_product_attention
HAS_SDPA = False
if HAS_MLX:
    try:
        from mlx import nn as mlx_nn

        # mlx.fast.scaled_dot_product_attention landed in mlx >= 0.6
        import mlx.core.fast

        if hasattr(mx.fast, "scaled_dot_product_attention"):
            HAS_SDPA = True
    except (ImportError, AttributeError):
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_ITERS = 5
MEASURE_ITERS = 20
TOP_K = 32
HEAD_DIM = 128  # D per head

HEADS_LIST = [1, 8, 16, 32]
SEQ_LENGTHS_STANDARD = [1024, 4096, 16384]
SEQ_LENGTHS_FULL = [1024, 4096, 16384, 65536, 131072]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(xs: List[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + d * (s[c] - s[f])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays (flattened)."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data_numpy(
    H: int, N: int, D: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random Q, K, V, and a query vector q for multi-head attention.

    Returns:
        q:  (H, 1, D)   — single query token per head
        K:  (H, N, D)   — key cache
        V:  (H, N, D)   — value cache
        full_ref: (H, 1, D) — full-attention reference output (for cosine check)
    """
    rng = np.random.RandomState(seed)
    q = rng.randn(H, 1, D).astype(np.float32)
    K = rng.randn(H, N, D).astype(np.float32)
    V = rng.randn(H, N, D).astype(np.float32)

    # Compute full-attention reference (softmax over all N keys)
    # scores: (H, 1, N) = q @ K^T / sqrt(D)
    scale = 1.0 / math.sqrt(D)
    scores = np.matmul(q, K.transpose(0, 2, 1)) * scale  # (H, 1, N)
    # Numerically stable softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    full_ref = np.matmul(attn_weights, V)  # (H, 1, D)

    return q, K, V, full_ref


# ---------------------------------------------------------------------------
# Backend 1: CPU (NumPy) — top-k sparse attention
# ---------------------------------------------------------------------------

def cpu_topk_attention(
    q: np.ndarray, K: np.ndarray, V: np.ndarray, top_k: int
) -> np.ndarray:
    """
    Top-k sparse attention on CPU using NumPy.

    q:  (H, 1, D)
    K:  (H, N, D)
    V:  (H, N, D)

    Returns: (H, 1, D)
    """
    H, _, D = q.shape
    N = K.shape[1]
    k = min(top_k, N)
    scale = 1.0 / math.sqrt(D)

    # scores: (H, 1, N) via cblas_sgemv under the hood
    scores = np.matmul(q, K.transpose(0, 2, 1)) * scale  # (H, 1, N)
    scores_2d = scores.reshape(H, N)  # (H, N)

    # top-k per head
    # argpartition is O(N) average — faster than full sort
    topk_idx = np.argpartition(scores_2d, -k, axis=-1)[:, -k:]  # (H, k)

    # Gather top-k scores and apply softmax
    topk_scores = np.take_along_axis(scores_2d, topk_idx, axis=-1)  # (H, k)
    topk_max = topk_scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(topk_scores - topk_max)
    attn_w = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (H, k)

    # Gather top-k values and weighted sum
    # V_topk: (H, k, D)
    out = np.zeros((H, 1, D), dtype=np.float32)
    for h in range(H):
        V_sel = V[h, topk_idx[h], :]  # (k, D)
        out[h, 0, :] = attn_w[h] @ V_sel  # (1, k) @ (k, D) -> (1, D)

    return out


def bench_cpu(
    q: np.ndarray, K: np.ndarray, V: np.ndarray, top_k: int
) -> Tuple[np.ndarray, List[float]]:
    """Warmup + measure CPU top-k attention. Returns (result, latencies_ms)."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = cpu_topk_attention(q, K, V, top_k)

    latencies: List[float] = []
    result = None
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        result = cpu_topk_attention(q, K, V, top_k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return result, latencies


# ---------------------------------------------------------------------------
# Backend 2: MLX GPU — batched matmul top-k attention
# ---------------------------------------------------------------------------

def mlx_topk_attention(
    q_mx: "mx.array",
    K_mx: "mx.array",
    V_mx: "mx.array",
    top_k: int,
    D: int,
) -> "mx.array":
    """
    Top-k sparse attention on MLX GPU using batched matmul.

    q_mx:  (H, 1, D)
    K_mx:  (H, N, D)
    V_mx:  (H, N, D)

    Returns: (H, 1, D) on MLX
    """
    scale = 1.0 / math.sqrt(D)
    # scores: (H, 1, N)
    scores = mx.matmul(q_mx, mx.transpose(K_mx, axes=(0, 2, 1))) * scale
    scores_2d = mx.reshape(scores, (scores.shape[0], scores.shape[2]))  # (H, N)

    N = scores_2d.shape[1]
    k = min(top_k, N)

    # MLX top-k: returns (values, indices) along last axis
    # mx.topk does NOT exist in all versions — use argsort fallback
    try:
        topk_idx = mx.argpartition(scores_2d, kth=N - k, axis=-1)[:, -k:]
    except (AttributeError, TypeError):
        # Fallback: full argsort
        sorted_idx = mx.argsort(scores_2d, axis=-1)
        topk_idx = sorted_idx[:, -k:]

    # Gather top-k scores
    H_dim = q_mx.shape[0]
    # Advanced indexing: we need per-head gather
    # Use mx.take_along_axis if available, else loop
    topk_scores = mx.take_along_axis(scores_2d, topk_idx, axis=-1)  # (H, k)

    # Softmax over top-k
    topk_max = mx.max(topk_scores, axis=-1, keepdims=True)
    exp_s = mx.exp(topk_scores - topk_max)
    attn_w = exp_s / mx.sum(exp_s, axis=-1, keepdims=True)  # (H, k)

    # Gather V values for top-k indices and compute weighted sum
    # We need V[h, topk_idx[h], :] for each head h
    # Expand indices for gather: (H, k) -> (H, k, D)
    topk_idx_exp = mx.expand_dims(topk_idx, axis=-1)  # (H, k, 1)
    topk_idx_exp = mx.broadcast_to(topk_idx_exp, (H_dim, k, q_mx.shape[2]))  # (H, k, D)
    V_sel = mx.take_along_axis(V_mx, topk_idx_exp, axis=1)  # (H, k, D)

    # attn_w: (H, k) -> (H, 1, k) for matmul
    attn_w_3d = mx.expand_dims(attn_w, axis=1)  # (H, 1, k)
    out = mx.matmul(attn_w_3d, V_sel)  # (H, 1, D)

    return out


def bench_mlx_matmul(
    q: np.ndarray, K: np.ndarray, V: np.ndarray, top_k: int
) -> Tuple[np.ndarray, List[float]]:
    """Warmup + measure MLX batched matmul top-k attention. Returns (result_np, latencies_ms)."""
    D = q.shape[2]
    q_mx = mx.array(q)
    K_mx = mx.array(K)
    V_mx = mx.array(V)
    mx.eval(q_mx, K_mx, V_mx)  # ensure transferred to GPU

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = mlx_topk_attention(q_mx, K_mx, V_mx, top_k, D)
        mx.eval(out)

    latencies: List[float] = []
    result_mx = None
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        result_mx = mlx_topk_attention(q_mx, K_mx, V_mx, top_k, D)
        mx.eval(result_mx)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    result_np = np.array(result_mx)
    return result_np, latencies


# ---------------------------------------------------------------------------
# Backend 3: MLX GPU — mx.fast.scaled_dot_product_attention (full attention)
# ---------------------------------------------------------------------------

def bench_mlx_sdpa(
    q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
    """
    Full attention via mx.fast.scaled_dot_product_attention.
    This is NOT top-k — it computes full softmax over all N keys.
    Included to show the fused-kernel ceiling.

    q:  (H, 1, D) — treated as (batch=1, H, 1, D) for SDPA
    K:  (H, N, D) — treated as (batch=1, H, N, D)
    V:  (H, N, D) — treated as (batch=1, H, N, D)

    Returns (result_np (H, 1, D), latencies_ms) or (None, None) if unavailable.
    """
    if not HAS_SDPA:
        return None, None

    D = q.shape[2]
    scale = 1.0 / math.sqrt(D)

    # SDPA expects (batch, heads, seq_q, head_dim) and (batch, heads, seq_kv, head_dim)
    q_mx = mx.array(q[np.newaxis, ...])   # (1, H, 1, D)
    K_mx = mx.array(K[np.newaxis, ...])   # (1, H, N, D)
    V_mx = mx.array(V[np.newaxis, ...])   # (1, H, N, D)
    mx.eval(q_mx, K_mx, V_mx)

    # Warmup
    for _ in range(WARMUP_ITERS):
        out = mx.fast.scaled_dot_product_attention(q_mx, K_mx, V_mx, scale=scale)
        mx.eval(out)

    latencies: List[float] = []
    result_mx = None
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        result_mx = mx.fast.scaled_dot_product_attention(q_mx, K_mx, V_mx, scale=scale)
        mx.eval(result_mx)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    # (1, H, 1, D) -> (H, 1, D)
    result_np = np.array(result_mx).squeeze(0)
    return result_np, latencies


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    backend: str  # "cpu", "mlx_matmul", "mlx_sdpa"
    H: int
    N: int
    D: int
    top_k: int
    median_ms: float
    min_ms: float
    p95_ms: float
    cosine_vs_full: float  # cosine similarity to full-attention reference
    speedup_vs_cpu: float  # median_cpu / median_this (filled in post)


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_config(
    H: int, N: int, D: int, top_k: int, verbose: bool = True
) -> List[BenchResult]:
    """Run all backends for a single (H, N, D, top_k) config."""
    results: List[BenchResult] = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"  H={H}, N={N:,}, D={D}, top_k={top_k}")
        print(f"  Memory: K+V = {2 * H * N * D * 4 / (1024**2):.1f} MB (FP32)")
        print(f"{'='*70}")

    # Generate data
    q, K, V, full_ref = generate_data_numpy(H, N, D)

    # --- Backend 1: CPU ---
    if verbose:
        print(f"  [CPU] running {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters ... ", end="", flush=True)
    cpu_out, cpu_lats = bench_cpu(q, K, V, top_k)
    cpu_cos = cosine_similarity(cpu_out, full_ref)
    cpu_med = median(cpu_lats)
    cpu_min = min(cpu_lats)
    cpu_p95 = percentile(cpu_lats, 95)
    if verbose:
        print(f"median={cpu_med:.3f} ms, cos={cpu_cos:.6f}")

    results.append(BenchResult(
        backend="cpu", H=H, N=N, D=D, top_k=top_k,
        median_ms=cpu_med, min_ms=cpu_min, p95_ms=cpu_p95,
        cosine_vs_full=cpu_cos, speedup_vs_cpu=1.0,
    ))

    # --- Backend 2: MLX matmul ---
    if HAS_MLX:
        if verbose:
            print(f"  [MLX matmul] running {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters ... ", end="", flush=True)
        mlx_out, mlx_lats = bench_mlx_matmul(q, K, V, top_k)
        mlx_cos = cosine_similarity(mlx_out, full_ref)
        mlx_med = median(mlx_lats)
        mlx_min = min(mlx_lats)
        mlx_p95 = percentile(mlx_lats, 95)
        speedup = cpu_med / mlx_med if mlx_med > 0 else float("inf")
        if verbose:
            print(f"median={mlx_med:.3f} ms, cos={mlx_cos:.6f}, speedup={speedup:.2f}x")

        results.append(BenchResult(
            backend="mlx_matmul", H=H, N=N, D=D, top_k=top_k,
            median_ms=mlx_med, min_ms=mlx_min, p95_ms=mlx_p95,
            cosine_vs_full=mlx_cos, speedup_vs_cpu=speedup,
        ))
    else:
        if verbose:
            print("  [MLX matmul] SKIPPED (mlx not installed)")

    # --- Backend 3: MLX SDPA (full attention, no top-k) ---
    if HAS_MLX:
        if verbose:
            print(f"  [MLX SDPA] running {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters ... ", end="", flush=True)
        sdpa_out, sdpa_lats = bench_mlx_sdpa(q, K, V)
        if sdpa_out is not None and sdpa_lats is not None:
            sdpa_cos = cosine_similarity(sdpa_out, full_ref)
            sdpa_med = median(sdpa_lats)
            sdpa_min = min(sdpa_lats)
            sdpa_p95 = percentile(sdpa_lats, 95)
            speedup = cpu_med / sdpa_med if sdpa_med > 0 else float("inf")
            if verbose:
                print(f"median={sdpa_med:.3f} ms, cos={sdpa_cos:.6f}, speedup={speedup:.2f}x")

            results.append(BenchResult(
                backend="mlx_sdpa", H=H, N=N, D=D, top_k=top_k,
                median_ms=sdpa_med, min_ms=sdpa_min, p95_ms=sdpa_p95,
                cosine_vs_full=sdpa_cos, speedup_vs_cpu=speedup,
            ))
        else:
            if verbose:
                print("SKIPPED (mx.fast.scaled_dot_product_attention not available)")
    else:
        if verbose:
            print("  [MLX SDPA] SKIPPED (mlx not installed)")

    return results


def print_summary_table(all_results: List[BenchResult], markdown: bool) -> None:
    """Print a consolidated results table."""
    if not all_results:
        print("No results to display.")
        return

    # Group by (H, N)
    configs = sorted(set((r.H, r.N) for r in all_results))
    backends = sorted(set(r.backend for r in all_results),
                      key=lambda b: ["cpu", "mlx_matmul", "mlx_sdpa"].index(b)
                      if b in ["cpu", "mlx_matmul", "mlx_sdpa"] else 99)

    backend_labels = {
        "cpu": "CPU (NumPy)",
        "mlx_matmul": "MLX matmul",
        "mlx_sdpa": "MLX SDPA",
    }

    # Build lookup
    lookup = {}
    for r in all_results:
        lookup[(r.H, r.N, r.backend)] = r

    sep = "|" if markdown else " | "
    header_line = "-" if markdown else "-"

    # --- Latency table ---
    print("\n")
    title = "Latency (ms): median / min / p95"
    if markdown:
        print(f"### {title}\n")
    else:
        print(f"  {title}")
        print(f"  {'=' * 90}")

    # Header
    cols = ["H", "N"]
    for b in backends:
        cols.append(backend_labels.get(b, b))
    if markdown:
        print(f"| {' | '.join(cols)} |")
        print(f"| {' | '.join(['---'] * len(cols))} |")
    else:
        widths = [4, 8] + [22] * len(backends)
        hdr = ""
        for c, w in zip(cols, widths):
            hdr += f"{c:>{w}}  "
        print(f"  {hdr}")
        print(f"  {'-' * (sum(widths) + 2 * len(widths))}")

    for H, N in configs:
        row = [f"{H}", f"{N:,}"]
        for b in backends:
            r = lookup.get((H, N, b))
            if r:
                row.append(f"{r.median_ms:.2f} / {r.min_ms:.2f} / {r.p95_ms:.2f}")
            else:
                row.append("--")
        if markdown:
            print(f"| {' | '.join(row)} |")
        else:
            parts = ""
            widths_data = [4, 8] + [22] * len(backends)
            for val, w in zip(row, widths_data):
                parts += f"{val:>{w}}  "
            print(f"  {parts}")

    # --- Cosine similarity table ---
    print("\n")
    title2 = "Cosine similarity vs full attention"
    if markdown:
        print(f"### {title2}\n")
    else:
        print(f"  {title2}")
        print(f"  {'=' * 70}")

    cols2 = ["H", "N"]
    for b in backends:
        cols2.append(backend_labels.get(b, b))
    if markdown:
        print(f"| {' | '.join(cols2)} |")
        print(f"| {' | '.join(['---'] * len(cols2))} |")
    else:
        widths2 = [4, 8] + [14] * len(backends)
        hdr2 = ""
        for c, w in zip(cols2, widths2):
            hdr2 += f"{c:>{w}}  "
        print(f"  {hdr2}")
        print(f"  {'-' * (sum(widths2) + 2 * len(widths2))}")

    for H, N in configs:
        row = [f"{H}", f"{N:,}"]
        for b in backends:
            r = lookup.get((H, N, b))
            if r:
                row.append(f"{r.cosine_vs_full:.6f}")
            else:
                row.append("--")
        if markdown:
            print(f"| {' | '.join(row)} |")
        else:
            parts = ""
            widths_data2 = [4, 8] + [14] * len(backends)
            for val, w in zip(row, widths_data2):
                parts += f"{val:>{w}}  "
            print(f"  {parts}")

    # --- Speedup table ---
    print("\n")
    title3 = "Speedup vs CPU (median latency)"
    if markdown:
        print(f"### {title3}\n")
    else:
        print(f"  {title3}")
        print(f"  {'=' * 60}")

    cols3 = ["H", "N"]
    for b in backends:
        if b == "cpu":
            continue
        cols3.append(backend_labels.get(b, b))
    if markdown:
        print(f"| {' | '.join(cols3)} |")
        print(f"| {' | '.join(['---'] * len(cols3))} |")
    else:
        widths3 = [4, 8] + [14] * (len(backends) - 1)
        hdr3 = ""
        for c, w in zip(cols3, widths3):
            hdr3 += f"{c:>{w}}  "
        print(f"  {hdr3}")
        print(f"  {'-' * (sum(widths3) + 2 * len(widths3))}")

    for H, N in configs:
        row = [f"{H}", f"{N:,}"]
        for b in backends:
            if b == "cpu":
                continue
            r = lookup.get((H, N, b))
            if r:
                row.append(f"{r.speedup_vs_cpu:.2f}x")
            else:
                row.append("--")
        if markdown:
            print(f"| {' | '.join(row)} |")
        else:
            parts = ""
            widths_data3 = [4, 8] + [14] * (len(backends) - 1)
            for val, w in zip(row, widths_data3):
                parts += f"{val:>{w}}  "
            print(f"  {parts}")


def main() -> None:
    global WARMUP_ITERS, MEASURE_ITERS

    parser = argparse.ArgumentParser(
        description="MLX vs CPU attention benchmark (Phase 2)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include 65K and 128K sequence lengths",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output paper-ready markdown tables",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=None,
        help="Comma-separated head counts (default: 1,8,16,32)",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default=None,
        help="Comma-separated sequence lengths (overrides --full)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-k for sparse attention (default: {TOP_K})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERS,
        help=f"Warmup iterations (default: {WARMUP_ITERS})",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=MEASURE_ITERS,
        help=f"Measurement iterations (default: {MEASURE_ITERS})",
    )
    args = parser.parse_args()

    # Override globals if specified
    WARMUP_ITERS = args.warmup
    MEASURE_ITERS = args.iters

    # Determine configs
    if args.heads:
        heads = [int(h) for h in args.heads.split(",")]
    else:
        heads = HEADS_LIST

    if args.seq_lengths:
        seq_lengths = [int(n) for n in args.seq_lengths.split(",")]
    elif args.full:
        seq_lengths = SEQ_LENGTHS_FULL
    else:
        seq_lengths = SEQ_LENGTHS_STANDARD

    top_k = args.top_k

    # Print header
    print("=" * 70)
    print("  MLX vs CPU Attention Benchmark — Phase 2")
    print("=" * 70)
    print(f"  Heads:        {heads}")
    print(f"  Seq lengths:  {[f'{n:,}' for n in seq_lengths]}")
    print(f"  Head dim (D): {HEAD_DIM}")
    print(f"  Top-k:        {top_k}")
    print(f"  Warmup:       {WARMUP_ITERS} iters")
    print(f"  Measure:      {MEASURE_ITERS} iters")
    print(f"  MLX:          {'available' if HAS_MLX else 'NOT INSTALLED'}")
    print(f"  SDPA:         {'available' if HAS_SDPA else 'not available'}")

    if HAS_MLX:
        # Print MLX device info
        try:
            default_device = mx.default_device()
            print(f"  MLX device:   {default_device}")
        except Exception:
            pass

    print()

    # Run all configs
    all_results: List[BenchResult] = []
    for H in heads:
        for N in seq_lengths:
            # Memory check: K+V in FP32
            mem_mb = 2 * H * N * HEAD_DIM * 4 / (1024 ** 2)
            if mem_mb > 16384:
                print(f"\n  SKIPPING H={H}, N={N:,} — would need {mem_mb:.0f} MB (>16 GB)")
                continue
            try:
                results = run_config(H, N, HEAD_DIM, top_k, verbose=not args.markdown)
                all_results.extend(results)
            except Exception as e:
                print(f"\n  ERROR at H={H}, N={N:,}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print_summary_table(all_results, markdown=args.markdown)

    # Final summary line
    print()
    if all_results:
        mlx_results = [r for r in all_results if r.backend == "mlx_matmul"]
        if mlx_results:
            best = max(mlx_results, key=lambda r: r.speedup_vs_cpu)
            worst = min(mlx_results, key=lambda r: r.speedup_vs_cpu)
            print(f"  MLX matmul speedup range: {worst.speedup_vs_cpu:.2f}x - {best.speedup_vs_cpu:.2f}x vs CPU")
            avg_cos = sum(r.cosine_vs_full for r in mlx_results) / len(mlx_results)
            print(f"  MLX matmul avg cosine vs full: {avg_cos:.6f}")

        sdpa_results = [r for r in all_results if r.backend == "mlx_sdpa"]
        if sdpa_results:
            best_s = max(sdpa_results, key=lambda r: r.speedup_vs_cpu)
            worst_s = min(sdpa_results, key=lambda r: r.speedup_vs_cpu)
            print(f"  MLX SDPA speedup range:   {worst_s.speedup_vs_cpu:.2f}x - {best_s.speedup_vs_cpu:.2f}x vs CPU")
            avg_cos_s = sum(r.cosine_vs_full for r in sdpa_results) / len(sdpa_results)
            print(f"  MLX SDPA avg cosine vs full:   {avg_cos_s:.6f}")

    print()


if __name__ == "__main__":
    main()
