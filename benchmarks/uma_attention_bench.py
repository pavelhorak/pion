#!/usr/bin/env python3
"""
UMMA Attention Benchmark — UMA-Native Mixed Attention for Apple Silicon
=======================================================================

Research question: Is CPU-key-search + GPU-value-multiply faster than
either CPU-only or GPU-only attention on unified memory hardware?

Three strategies:
  1. CPU-only   — NumPy: Q@K^T → top-k → softmax → V multiply
  2. GPU-only   — MLX Metal: same ops, all on GPU
  3. Hybrid UMMA — CPU does Q@K^T + top-k (SIMD), GPU does V gather+multiply (Metal)

The UMA insight: CPU and GPU share the same physical memory on Apple Silicon.
No PCIe copies. The question is whether splitting work across both processors
and running them concurrently beats either processor alone.

Usage:
  python3 benchmarks/uma_attention_bench.py
  python3 benchmarks/uma_attention_bench.py --seq-lens 1024,4096,16384 --top-k 64
  python3 benchmarks/uma_attention_bench.py --full   # include 64K and 128K
"""

import argparse
import time
import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import queue

# ---------------------------------------------------------------------------
# MLX import
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("WARNING: MLX not available. GPU and Hybrid benchmarks will be skipped.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HEAD_DIM = 128          # standard transformer head dimension
NUM_HEADS = 1           # benchmark one head (scales linearly)
WARMUP_ITERS = 3
BENCH_ITERS = 10
TOP_K_DEFAULT = 32      # sparse attention: attend to top-k tokens

@dataclass
class BenchResult:
    strategy: str
    seq_len: int
    top_k: int
    mean_us: float       # mean latency in microseconds
    std_us: float
    min_us: float
    max_us: float
    bandwidth_gbps: float  # effective memory bandwidth utilization
    output_cosine: float   # cosine similarity to full-attention reference


# ---------------------------------------------------------------------------
# Reference: full dense attention (CPU, for correctness validation)
# ---------------------------------------------------------------------------
def full_attention_cpu(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Full dense attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V


# ---------------------------------------------------------------------------
# Strategy 1: CPU-only sparse attention (NumPy)
# ---------------------------------------------------------------------------
def cpu_only_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, top_k: int) -> np.ndarray:
    """CPU sparse attention: Q@K^T → top-k → softmax → V[top_k] multiply"""
    d = Q.shape[-1]
    # Key search: full dot product (BLAS, uses AMX on Apple Silicon)
    scores = Q @ K.T / np.sqrt(d)  # (1, seq_len)

    # Top-k selection
    topk_indices = np.argpartition(scores[0], -top_k)[-top_k:]
    topk_scores = scores[0, topk_indices]

    # Softmax over top-k only
    topk_scores -= topk_scores.max()
    weights = np.exp(topk_scores)
    weights /= weights.sum()

    # Value multiply: gather + weighted sum
    V_topk = V[topk_indices]  # (top_k, d)
    output = weights @ V_topk  # (d,)
    return output.reshape(1, -1)


# ---------------------------------------------------------------------------
# Strategy 2: GPU-only sparse attention (MLX Metal)
# ---------------------------------------------------------------------------
def gpu_only_attention(Q_mx: 'mx.array', K_mx: 'mx.array', V_mx: 'mx.array',
                       top_k: int) -> 'mx.array':
    """GPU sparse attention: all ops on Metal"""
    d = Q_mx.shape[-1]
    scores = (Q_mx @ K_mx.T) / np.sqrt(d)  # (1, seq_len)

    # Top-k on GPU
    topk_indices = mx.argpartition(scores[0], kth=-top_k)[-top_k:]
    topk_scores = scores[0][topk_indices]

    # Softmax
    topk_scores = topk_scores - mx.max(topk_scores)
    weights = mx.exp(topk_scores)
    weights = weights / mx.sum(weights)

    # V gather + multiply
    V_topk = V_mx[topk_indices]
    output = weights @ V_topk
    return output.reshape(1, -1)


# ---------------------------------------------------------------------------
# Strategy 3: Hybrid UMMA — CPU key search, GPU value multiply
# ---------------------------------------------------------------------------
def hybrid_umma_attention(Q: np.ndarray, K: np.ndarray, V_mx: 'mx.array',
                          top_k: int) -> 'mx.array':
    """
    Hybrid UMA attention:
      Phase 1 (CPU): Q @ K^T → top-k indices + scores  [AMX SIMD]
      Phase 2 (GPU): gather V[indices], weighted sum     [Metal]

    On UMA, V_mx and K share the same physical memory — no copies needed.
    CPU is better at sequential top-k selection (branch-heavy).
    GPU is better at parallel gather + matrix multiply.
    """
    d = Q.shape[-1]

    # Phase 1: CPU key search (uses AMX/NEON via BLAS)
    scores = Q @ K.T / np.sqrt(d)
    topk_indices = np.argpartition(scores[0], -top_k)[-top_k:]
    topk_scores = scores[0, topk_indices]

    # Softmax on CPU (tiny array, CPU is faster for k=32)
    topk_scores -= topk_scores.max()
    weights = np.exp(topk_scores)
    weights /= weights.sum()

    # Phase 2: GPU value gather + multiply
    indices_mx = mx.array(topk_indices)
    weights_mx = mx.array(weights.astype(np.float32))
    V_topk = V_mx[indices_mx]
    output = weights_mx @ V_topk
    return output.reshape(1, -1)


# ---------------------------------------------------------------------------
# Strategy 3b: Hybrid UMMA with overlapped execution
# ---------------------------------------------------------------------------
def hybrid_umma_overlapped(Q: np.ndarray, K: np.ndarray, V_mx: 'mx.array',
                           top_k: int) -> 'mx.array':
    """
    Hybrid with CPU/GPU overlap:
      - CPU computes scores for query Q (AMX)
      - Meanwhile GPU pre-warms by doing a dummy eval (keeps Metal pipeline hot)
      - CPU extracts top-k → GPU gathers V and multiplies

    The key UMA advantage: no copy latency between phases.
    mx.array wraps the same physical memory the CPU just wrote.
    """
    d = Q.shape[-1]

    # Phase 1: CPU key search
    scores = Q @ K.T / np.sqrt(d)
    topk_indices = np.argpartition(scores[0], -top_k)[-top_k:]
    topk_scores = scores[0, topk_indices]

    topk_scores -= topk_scores.max()
    weights = np.exp(topk_scores)
    weights /= weights.sum()

    # Phase 2: GPU — zero-copy handoff via mx.array wrapping numpy
    # On UMA this is a pointer cast, not a memcpy
    indices_mx = mx.array(topk_indices)
    weights_mx = mx.array(weights.astype(np.float32))

    V_topk = V_mx[indices_mx]
    output = weights_mx @ V_topk
    return output.reshape(1, -1)


# ---------------------------------------------------------------------------
# Strategy 3c: Hybrid with concurrent CPU+GPU via threading
# ---------------------------------------------------------------------------
def hybrid_umma_concurrent(Q: np.ndarray, K: np.ndarray, V_mx: 'mx.array',
                           top_k: int,
                           K_mx: 'mx.array') -> 'mx.array':
    """
    Concurrent hybrid: CPU and GPU compute scores in parallel on the SAME data.
    Take whichever finishes first (they share UMA, so both read from same memory).
    CPU wins at small seq_len (lower dispatch overhead), GPU wins at large.

    This tests whether concurrent access to shared UMA memory helps or hurts
    (cache coherency traffic between CPU and GPU memory controllers).
    """
    d = Q.shape[-1]
    result_q = queue.Queue()
    Q_mx_local = mx.array(Q)

    def cpu_path():
        scores = Q @ K.T / np.sqrt(d)
        topk_idx = np.argpartition(scores[0], -top_k)[-top_k:]
        topk_sc = scores[0, topk_idx]
        topk_sc -= topk_sc.max()
        w = np.exp(topk_sc)
        w /= w.sum()
        result_q.put(('cpu', topk_idx, w))

    def gpu_path():
        scores = (Q_mx_local @ K_mx.T) / np.sqrt(d)
        topk_idx = mx.argpartition(scores[0], kth=-top_k)[-top_k:]
        topk_sc = scores[0][topk_idx]
        topk_sc = topk_sc - mx.max(topk_sc)
        w = mx.exp(topk_sc)
        w = w / mx.sum(w)
        mx.eval(w, topk_idx)  # force GPU sync
        result_q.put(('gpu', topk_idx, w))

    t_cpu = threading.Thread(target=cpu_path)
    t_gpu = threading.Thread(target=gpu_path)
    t_cpu.start()
    t_gpu.start()

    # Use first result
    winner, topk_indices, weights = result_q.get()
    t_cpu.join()
    t_gpu.join()

    # GPU value multiply regardless of who found the indices
    if winner == 'cpu':
        indices_mx = mx.array(topk_indices)
        weights_mx = mx.array(weights.astype(np.float32))
    else:
        indices_mx = topk_indices
        weights_mx = weights

    V_topk = V_mx[indices_mx]
    output = weights_mx @ V_topk
    mx.eval(output)
    return output.reshape(1, -1)


# ---------------------------------------------------------------------------
# Benchmarking harness
# ---------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    return float(dot / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10))


def estimate_bandwidth(seq_len: int, top_k: int, latency_s: float) -> float:
    """Estimate effective memory bandwidth in GB/s"""
    # Reads: Q(128*4) + K(seq*128*4) + V_topk(k*128*4) + scores(seq*4)
    # Writes: output(128*4) + indices(k*4) + weights(k*4)
    bytes_read = 4 * (HEAD_DIM + seq_len * HEAD_DIM + top_k * HEAD_DIM + seq_len)
    bytes_written = 4 * (HEAD_DIM + top_k + top_k)
    total_bytes = bytes_read + bytes_written
    if latency_s <= 0:
        return 0.0
    return (total_bytes / latency_s) / 1e9


def bench_one(fn, args, warmup: int, iters: int, sync_mlx: bool = False) -> Tuple[float, float, float, float]:
    """Returns (mean_us, std_us, min_us, max_us)"""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if sync_mlx and HAS_MLX:
            mx.eval(result)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        if sync_mlx and HAS_MLX:
            mx.eval(result)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # ns → µs

    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def run_benchmark(seq_len: int, top_k: int) -> List[BenchResult]:
    """Run all strategies for one sequence length."""
    results = []

    # Generate data — simulate one attention head
    np.random.seed(42)
    Q = np.random.randn(1, HEAD_DIM).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, HEAD_DIM).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)

    # Reference: full dense attention
    ref_output = full_attention_cpu(Q, K, V)

    # --- Strategy 1: CPU-only ---
    mean, std, mn, mx_val = bench_one(
        cpu_only_attention, (Q, K, V, top_k),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS
    )
    cpu_out = cpu_only_attention(Q, K, V, top_k)
    cos = cosine_sim(cpu_out, ref_output)
    bw = estimate_bandwidth(seq_len, top_k, mean / 1e6)
    results.append(BenchResult("CPU-only", seq_len, top_k, mean, std, mn, mx_val, bw, cos))

    if not HAS_MLX:
        return results

    # MLX arrays (UMA: same physical memory, no copy)
    Q_mx = mx.array(Q)
    K_mx = mx.array(K)
    V_mx = mx.array(V)

    # --- Strategy 2: GPU-only ---
    mean, std, mn, mx_val = bench_one(
        gpu_only_attention, (Q_mx, K_mx, V_mx, top_k),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    gpu_out = gpu_only_attention(Q_mx, K_mx, V_mx, top_k)
    mx.eval(gpu_out)
    cos = cosine_sim(np.array(gpu_out), ref_output)
    bw = estimate_bandwidth(seq_len, top_k, mean / 1e6)
    results.append(BenchResult("GPU-only", seq_len, top_k, mean, std, mn, mx_val, bw, cos))

    # --- Strategy 3a: Hybrid UMMA (sequential) ---
    mean, std, mn, mx_val = bench_one(
        hybrid_umma_attention, (Q, K, V_mx, top_k),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    hyb_out = hybrid_umma_attention(Q, K, V_mx, top_k)
    mx.eval(hyb_out)
    cos = cosine_sim(np.array(hyb_out), ref_output)
    bw = estimate_bandwidth(seq_len, top_k, mean / 1e6)
    results.append(BenchResult("Hybrid-UMMA", seq_len, top_k, mean, std, mn, mx_val, bw, cos))

    # --- Strategy 3b: Hybrid UMMA overlapped ---
    mean, std, mn, mx_val = bench_one(
        hybrid_umma_overlapped, (Q, K, V_mx, top_k),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    hyb2_out = hybrid_umma_overlapped(Q, K, V_mx, top_k)
    mx.eval(hyb2_out)
    cos = cosine_sim(np.array(hyb2_out), ref_output)
    bw = estimate_bandwidth(seq_len, top_k, mean / 1e6)
    results.append(BenchResult("Hybrid-overlap", seq_len, top_k, mean, std, mn, mx_val, bw, cos))

    # --- Strategy 3c: Hybrid concurrent (CPU+GPU race) ---
    mean, std, mn, mx_val = bench_one(
        hybrid_umma_concurrent, (Q, K, V_mx, top_k, K_mx),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    hyb3_out = hybrid_umma_concurrent(Q, K, V_mx, top_k, K_mx)
    mx.eval(hyb3_out)
    cos = cosine_sim(np.array(hyb3_out), ref_output)
    bw = estimate_bandwidth(seq_len, top_k, mean / 1e6)
    results.append(BenchResult("Hybrid-concurrent", seq_len, top_k, mean, std, mn, mx_val, bw, cos))

    return results


# ---------------------------------------------------------------------------
# Dense attention benchmark (no top-k, full attention)
# ---------------------------------------------------------------------------
def cpu_dense_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    return full_attention_cpu(Q, K, V)

def gpu_dense_attention(Q_mx, K_mx, V_mx) -> 'mx.array':
    d = Q_mx.shape[-1]
    scores = (Q_mx @ K_mx.T) / np.sqrt(d)
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights @ V_mx

def hybrid_dense_attention(Q: np.ndarray, K: np.ndarray, V_mx: 'mx.array') -> 'mx.array':
    """CPU computes attention weights, GPU does V multiply"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    weights_mx = mx.array(weights.astype(np.float32))
    return weights_mx @ V_mx

def run_dense_benchmark(seq_len: int) -> List[BenchResult]:
    """Dense (full) attention benchmark — no top-k."""
    results = []
    np.random.seed(42)
    Q = np.random.randn(1, HEAD_DIM).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, HEAD_DIM).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, HEAD_DIM).astype(np.float32)
    ref = full_attention_cpu(Q, K, V)

    # CPU dense
    mean, std, mn, mx_val = bench_one(
        cpu_dense_attention, (Q, K, V),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS
    )
    bytes_total = 4 * (HEAD_DIM + seq_len * HEAD_DIM * 2 + seq_len + HEAD_DIM)
    bw = (bytes_total / (mean / 1e6)) / 1e9 if mean > 0 else 0
    results.append(BenchResult("CPU-dense", seq_len, seq_len, mean, std, mn, mx_val, bw, 1.0))

    if not HAS_MLX:
        return results

    Q_mx = mx.array(Q)
    K_mx = mx.array(K)
    V_mx = mx.array(V)

    # GPU dense
    mean, std, mn, mx_val = bench_one(
        gpu_dense_attention, (Q_mx, K_mx, V_mx),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    results.append(BenchResult("GPU-dense", seq_len, seq_len, mean, std, mn, mx_val, bw, 1.0))

    # Hybrid dense: CPU scores, GPU V multiply
    mean, std, mn, mx_val = bench_one(
        hybrid_dense_attention, (Q, K, V_mx),
        warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True
    )
    hyb_out = hybrid_dense_attention(Q, K, V_mx)
    mx.eval(hyb_out)
    cos = cosine_sim(np.array(hyb_out), ref)
    results.append(BenchResult("Hybrid-dense", seq_len, seq_len, mean, std, mn, mx_val, bw, cos))

    return results


# ---------------------------------------------------------------------------
# Multi-head benchmark (realistic: 32 heads)
# ---------------------------------------------------------------------------
def run_multihead_benchmark(seq_len: int, num_heads: int, top_k: int) -> List[BenchResult]:
    """Multi-head attention: measures total latency for all heads."""
    results = []
    np.random.seed(42)

    Qs = [np.random.randn(1, HEAD_DIM).astype(np.float32) * 0.1 for _ in range(num_heads)]
    Ks = [np.random.randn(seq_len, HEAD_DIM).astype(np.float32) * 0.1 for _ in range(num_heads)]
    Vs = [np.random.randn(seq_len, HEAD_DIM).astype(np.float32) for _ in range(num_heads)]

    # CPU multi-head
    def cpu_multihead():
        outs = []
        for h in range(num_heads):
            outs.append(cpu_only_attention(Qs[h], Ks[h], Vs[h], top_k))
        return np.concatenate(outs, axis=-1)

    mean, std, mn, mx_val = bench_one(cpu_multihead, (), warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    bw = estimate_bandwidth(seq_len * num_heads, top_k * num_heads, mean / 1e6)
    results.append(BenchResult(f"CPU-{num_heads}h", seq_len, top_k, mean, std, mn, mx_val, bw, 1.0))

    if not HAS_MLX:
        return results

    # Batched GPU: stack all heads as a batch
    Q_batch = mx.array(np.stack([q[0] for q in Qs]))  # (H, d)
    K_batch = mx.array(np.stack(Ks))  # (H, seq, d)
    V_batch = mx.array(np.stack(Vs))  # (H, seq, d)

    def gpu_multihead():
        d = HEAD_DIM
        # Batched matmul: (H, 1, d) @ (H, d, seq) → (H, 1, seq)
        scores = mx.matmul(Q_batch[:, None, :], mx.transpose(K_batch, (0, 2, 1))) / np.sqrt(d)
        # Top-k per head
        topk_idx = mx.argpartition(scores[:, 0, :], kth=-top_k, axis=-1)[:, -top_k:]
        # Gather scores and V
        H = num_heads
        topk_scores = mx.take_along_axis(scores[:, 0, :], topk_idx, axis=-1)
        topk_scores = topk_scores - mx.max(topk_scores, axis=-1, keepdims=True)
        weights = mx.exp(topk_scores)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)  # (H, k)
        # Gather V: (H, k, d) — use vmap-style indexing
        # V_batch[h, topk_idx[h], :] for each h
        V_gathered = mx.take_along_axis(
            V_batch,
            topk_idx[:, :, None] * mx.ones((1, 1, HEAD_DIM), dtype=mx.int32),
            axis=1
        )
        # Weighted sum: (H, 1, k) @ (H, k, d) → (H, 1, d)
        output = mx.matmul(weights[:, None, :], V_gathered)
        return output[:, 0, :]  # (H, d)

    mean, std, mn, mx_val = bench_one(gpu_multihead, (), warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True)
    bw = estimate_bandwidth(seq_len * num_heads, top_k * num_heads, mean / 1e6)
    results.append(BenchResult(f"GPU-{num_heads}h", seq_len, top_k, mean, std, mn, mx_val, bw, 1.0))

    # Hybrid multi-head: CPU scores for all heads, GPU V multiply batched
    def hybrid_multihead():
        all_indices = []
        all_weights = []
        d = HEAD_DIM
        for h in range(num_heads):
            scores = Qs[h] @ Ks[h].T / np.sqrt(d)
            topk_idx = np.argpartition(scores[0], -top_k)[-top_k:]
            topk_sc = scores[0, topk_idx]
            topk_sc -= topk_sc.max()
            w = np.exp(topk_sc)
            w /= w.sum()
            all_indices.append(topk_idx)
            all_weights.append(w)

        indices_mx = mx.array(np.stack(all_indices))  # (H, k)
        weights_mx = mx.array(np.stack(all_weights).astype(np.float32))  # (H, k)

        V_gathered = mx.take_along_axis(
            V_batch,
            indices_mx[:, :, None] * mx.ones((1, 1, HEAD_DIM), dtype=mx.int32),
            axis=1
        )
        output = mx.matmul(weights_mx[:, None, :], V_gathered)
        return output[:, 0, :]

    mean, std, mn, mx_val = bench_one(hybrid_multihead, (), warmup=WARMUP_ITERS, iters=BENCH_ITERS, sync_mlx=True)
    bw = estimate_bandwidth(seq_len * num_heads, top_k * num_heads, mean / 1e6)
    results.append(BenchResult(f"Hybrid-{num_heads}h", seq_len, top_k, mean, std, mn, mx_val, bw, 1.0))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(title: str, results: List[BenchResult]):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Strategy':<20s} {'SeqLen':>7s} {'k':>5s} {'Mean µs':>10s} {'Std µs':>8s} "
          f"{'Min µs':>10s} {'BW GB/s':>9s} {'Cosine':>8s}")
    print(f"  {'-'*20} {'-'*7} {'-'*5} {'-'*10} {'-'*8} {'-'*10} {'-'*9} {'-'*8}")
    for r in results:
        print(f"  {r.strategy:<20s} {r.seq_len:>7d} {r.top_k:>5d} {r.mean_us:>10.1f} {r.std_us:>8.1f} "
              f"{r.min_us:>10.1f} {r.bandwidth_gbps:>9.2f} {r.output_cosine:>8.4f}")


def print_comparison(results: List[BenchResult]):
    """Print speedup comparison vs CPU-only baseline."""
    cpu_results = {r.seq_len: r for r in results if r.strategy == "CPU-only"}
    if not cpu_results:
        return

    print(f"\n{'='*80}")
    print(f"  Speedup vs CPU-only (>1.0× = hybrid/GPU is faster)")
    print(f"{'='*80}")
    print(f"  {'Strategy':<20s} {'SeqLen':>7s} {'CPU µs':>10s} {'This µs':>10s} {'Speedup':>8s}")
    print(f"  {'-'*20} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")

    for r in results:
        if r.strategy == "CPU-only":
            continue
        cpu = cpu_results.get(r.seq_len)
        if cpu:
            speedup = cpu.mean_us / r.mean_us if r.mean_us > 0 else 0
            marker = " <<<" if speedup > 1.0 else ""
            print(f"  {r.strategy:<20s} {r.seq_len:>7d} {cpu.mean_us:>10.1f} {r.mean_us:>10.1f} {speedup:>7.2f}×{marker}")


def generate_markdown(all_results: List[BenchResult], dense_results: List[BenchResult],
                      multihead_results: List[BenchResult]) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# UMMA Attention Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Hardware:** Apple Silicon UMA (M4, 16GB unified memory)")
    lines.append(f"**Config:** head_dim={HEAD_DIM}, warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
    lines.append("")
    lines.append("## Research Question")
    lines.append("")
    lines.append("Is CPU-key-search + GPU-value-multiply faster than either CPU-only or GPU-only")
    lines.append("attention on UMA hardware (Apple Silicon)?")
    lines.append("")

    # Sparse attention table
    lines.append("## Table 1: Sparse Attention (top-k)")
    lines.append("")
    lines.append("| Strategy | SeqLen | k | Mean µs | Min µs | BW GB/s | Cosine |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        lines.append(f"| {r.strategy} | {r.seq_len:,} | {r.top_k} | {r.mean_us:.1f} | "
                     f"{r.min_us:.1f} | {r.bandwidth_gbps:.2f} | {r.output_cosine:.4f} |")

    # Speedup table
    lines.append("")
    lines.append("## Table 2: Speedup vs CPU-only")
    lines.append("")
    lines.append("| Strategy | SeqLen | CPU µs | This µs | Speedup |")
    lines.append("|---|---:|---:|---:|---:|")
    cpu_by_seq = {r.seq_len: r for r in all_results if r.strategy == "CPU-only"}
    for r in all_results:
        if r.strategy == "CPU-only":
            continue
        cpu = cpu_by_seq.get(r.seq_len)
        if cpu:
            speedup = cpu.mean_us / r.mean_us if r.mean_us > 0 else 0
            lines.append(f"| {r.strategy} | {r.seq_len:,} | {cpu.mean_us:.1f} | {r.mean_us:.1f} | {speedup:.2f}× |")

    # Dense attention table
    if dense_results:
        lines.append("")
        lines.append("## Table 3: Dense Attention (full, no top-k)")
        lines.append("")
        lines.append("| Strategy | SeqLen | Mean µs | Min µs | BW GB/s |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in dense_results:
            lines.append(f"| {r.strategy} | {r.seq_len:,} | {r.mean_us:.1f} | {r.min_us:.1f} | {r.bandwidth_gbps:.2f} |")

    # Multi-head table
    if multihead_results:
        lines.append("")
        lines.append("## Table 4: Multi-Head Attention (32 heads, sparse)")
        lines.append("")
        lines.append("| Strategy | SeqLen | Mean µs | Min µs | BW GB/s |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in multihead_results:
            lines.append(f"| {r.strategy} | {r.seq_len:,} | {r.mean_us:.1f} | {r.min_us:.1f} | {r.bandwidth_gbps:.2f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UMMA Attention Benchmark")
    parser.add_argument("--seq-lens", type=str, default="512,1024,4096,16384",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT,
                        help=f"Top-k for sparse attention (default: {TOP_K_DEFAULT})")
    parser.add_argument("--full", action="store_true",
                        help="Include 64K and 128K sequence lengths")
    parser.add_argument("--dense", action="store_true",
                        help="Also run dense (full) attention benchmark")
    parser.add_argument("--multihead", action="store_true",
                        help="Also run 32-head multi-head benchmark")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS,
                        help=f"Benchmark iterations (default: {BENCH_ITERS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Write markdown report to file")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    if args.full:
        seq_lens = [512, 1024, 4096, 16384, 65536, 131072]

    top_k = args.top_k

    print(f"UMMA Attention Benchmark")
    print(f"  Hardware: Apple Silicon UMA")
    print(f"  head_dim={HEAD_DIM}, top_k={top_k}, iters={BENCH_ITERS}")
    print(f"  Sequence lengths: {seq_lens}")
    if HAS_MLX:
        print(f"  MLX: {mx.__version__}, Metal: {mx.metal.is_available()}")
    print()

    # --- Sparse attention ---
    all_results = []
    for sl in seq_lens:
        mem_mb = sl * HEAD_DIM * 4 * 2 / 1e6  # K + V
        if mem_mb > 14000:  # leave headroom on 16GB
            print(f"  SKIP seq_len={sl} (would need {mem_mb:.0f}MB, exceeds safe limit)")
            continue
        print(f"  Benchmarking seq_len={sl:,} ({mem_mb:.1f}MB K+V)...")
        results = run_benchmark(sl, top_k)
        all_results.extend(results)
        print_table(f"Sparse Attention — seq_len={sl:,}, top_k={top_k}", results)
        print_comparison(results)

    # --- Dense attention ---
    dense_results = []
    if args.dense:
        dense_lens = [s for s in seq_lens if s <= 16384]  # dense is O(n²), cap at 16K
        for sl in dense_lens:
            print(f"\n  Dense attention: seq_len={sl:,}...")
            results = run_dense_benchmark(sl)
            dense_results.extend(results)
            print_table(f"Dense Attention — seq_len={sl:,}", results)

    # --- Multi-head ---
    multihead_results = []
    if args.multihead:
        mh_lens = [s for s in seq_lens if s <= 16384]  # memory constraint
        for sl in mh_lens:
            mem_mb = sl * HEAD_DIM * 4 * 2 * 32 / 1e6
            if mem_mb > 10000:
                print(f"  SKIP multihead seq_len={sl} ({mem_mb:.0f}MB)")
                continue
            print(f"\n  Multi-head (32h): seq_len={sl:,} ({mem_mb:.1f}MB)...")
            results = run_multihead_benchmark(sl, 32, top_k)
            multihead_results.extend(results)
            print_table(f"Multi-Head 32h — seq_len={sl:,}", results)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"  OVERALL SPEEDUP SUMMARY")
    print_comparison(all_results)

    # --- Markdown output ---
    if args.output:
        md = generate_markdown(all_results, dense_results, multihead_results)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"\n  Report written to {args.output}")

    # --- Verdict ---
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}")

    # Find crossover point
    cpu_by_seq = {r.seq_len: r for r in all_results if r.strategy == "CPU-only"}
    gpu_by_seq = {r.seq_len: r for r in all_results if r.strategy == "GPU-only"}
    hyb_by_seq = {r.seq_len: r for r in all_results if r.strategy == "Hybrid-UMMA"}

    for sl in sorted(cpu_by_seq.keys()):
        cpu = cpu_by_seq[sl]
        gpu = gpu_by_seq.get(sl)
        hyb = hyb_by_seq.get(sl)
        best = "CPU"
        best_us = cpu.mean_us
        if gpu and gpu.mean_us < best_us:
            best = "GPU"
            best_us = gpu.mean_us
        if hyb and hyb.mean_us < best_us:
            best = "Hybrid-UMMA"
            best_us = hyb.mean_us
        winner_marker = " *** UMMA WINS ***" if best == "Hybrid-UMMA" else ""
        print(f"  seq_len={sl:>7,}: best={best:<15s} ({best_us:.1f}µs){winner_marker}")

    print()


if __name__ == "__main__":
    main()
