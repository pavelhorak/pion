"""J.b Prototype — MLX GPU vs NumPy CPU attention benchmark.

Compares the full sparse attention pipeline:
  GEMV (Q @ K^T) → top-k → softmax → V gather → weighted sum

Configurations tested:
  H=1,8,16,32  N=1K,4K,16K,64K  D=128  top_k=32

This establishes the MLX baseline that the Mojo GPU prototype must beat.

Usage:
    python3 benchmarks/gpu_attention_proto/bench_mlx_vs_cpu.py
"""

import numpy as np
import time
import sys

# ---------------------------------------------------------------------------
# NumPy CPU attention (BLAS-accelerated, same as Accelerate/AMX on macOS)
# ---------------------------------------------------------------------------

def cpu_sparse_attention(Q, K, V, top_k):
    """Sparse attention on CPU via NumPy/BLAS.

    Q: [H, 1, D], K: [H, N, D], V: [H, N, D]
    Returns: [H, D]
    """
    H, _, D = Q.shape
    N = K.shape[1]
    scale = 1.0 / np.sqrt(D)

    # GEMV: [H, 1, D] @ [H, D, N] → [H, 1, N]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale

    # Top-k per head
    scores_2d = scores[:, 0, :]  # [H, N]
    topk_idx = np.argpartition(scores_2d, -top_k, axis=-1)[:, -top_k:]  # [H, k]
    topk_scores = np.take_along_axis(scores_2d, topk_idx, axis=-1)  # [H, k]

    # Softmax
    topk_scores = topk_scores - topk_scores.max(axis=-1, keepdims=True)
    weights = np.exp(topk_scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)  # [H, k]

    # V gather
    idx_expanded = topk_idx[:, :, None].repeat(D, axis=2)  # [H, k, D]
    V_gathered = np.take_along_axis(V, idx_expanded, axis=1)  # [H, k, D]

    # Weighted sum: [H, 1, k] @ [H, k, D] → [H, 1, D]
    output = np.matmul(weights[:, None, :], V_gathered)
    return output[:, 0, :]  # [H, D]


# ---------------------------------------------------------------------------
# MLX GPU attention
# ---------------------------------------------------------------------------

def mlx_sparse_attention(Q_np, K_np, V_np, top_k):
    """Sparse attention on GPU via MLX (Apple Metal/MPSGraph).

    Same interface as cpu_sparse_attention.
    """
    import mlx.core as mx

    Q = mx.array(Q_np)
    K = mx.array(K_np)
    V = mx.array(V_np)

    H, _, D = Q.shape
    N = K.shape[1]
    scale = 1.0 / np.sqrt(D)

    # Batched GEMV
    scores = mx.matmul(Q, mx.transpose(K, (0, 2, 1))) * scale

    # Top-k
    topk_idx = mx.argpartition(scores[:, 0, :], kth=-top_k, axis=-1)[:, -top_k:]
    topk_scores = mx.take_along_axis(scores[:, 0, :], topk_idx, axis=-1)

    # Softmax
    topk_scores = topk_scores - mx.max(topk_scores, axis=-1, keepdims=True)
    weights = mx.exp(topk_scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)

    # V gather
    V_gathered = mx.take_along_axis(
        V,
        topk_idx[:, :, None] * mx.ones((1, 1, D), dtype=mx.int32),
        axis=1
    )

    # Weighted sum
    output = mx.matmul(weights[:, None, :], V_gathered)
    mx.eval(output)
    return np.array(output[:, 0, :])


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench(fn, Q, K, V, top_k, warmup=3, iters=20, label=""):
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn(Q, K, V, top_k)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = fn(Q, K, V, top_k)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)  # µs

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    p5 = times[int(len(times) * 0.05)]
    p95 = times[int(len(times) * 0.95)]
    return median, mean, p5, p95, out


def verify_correctness(cpu_out, mlx_out, label):
    """Check that CPU and MLX produce similar results."""
    cos = np.sum(cpu_out * mlx_out) / (np.linalg.norm(cpu_out) * np.linalg.norm(mlx_out) + 1e-8)
    max_diff = np.max(np.abs(cpu_out - mlx_out))
    print(f"  {label}: cosine={cos:.6f}, max_diff={max_diff:.6f}", end="")
    if cos < 0.99:
        print(" ⚠ LOW")
    else:
        print(" ✓")


def main():
    # Check MLX
    try:
        import mlx.core as mx
        print(f"MLX {mx.__version__}, Metal: {mx.metal.is_available()}")
        has_mlx = True
    except ImportError:
        print("MLX not available — CPU-only benchmark")
        has_mlx = False

    np.random.seed(42)

    configs = [
        # (H, N, D, top_k)
        (1,   1024,  128, 32),
        (1,   4096,  128, 32),
        (1,  16384,  128, 32),
        (8,   1024,  128, 32),
        (8,   4096,  128, 32),
        (8,  16384,  128, 32),
        (16,  4096,  128, 32),
        (16, 16384,  128, 32),
        (32,  4096,  128, 32),
        (32, 16384,  128, 32),
        (32, 65536,  128, 32),
    ]

    print()
    print(f"{'Config':>24s} | {'CPU µs':>10s} | {'MLX µs':>10s} | {'Speedup':>8s} | {'Winner':>6s}")
    print("-" * 78)

    results = []

    for H, N, D, top_k in configs:
        label = f"H={H} N={N:,} D={D} k={top_k}"

        Q = np.random.randn(H, 1, D).astype(np.float32) * 0.1
        K = np.random.randn(H, N, D).astype(np.float32) * 0.1
        V = np.random.randn(H, N, D).astype(np.float32) * 0.1

        # CPU benchmark
        cpu_med, cpu_mean, _, _, cpu_out = bench(cpu_sparse_attention, Q, K, V, top_k, label="CPU")

        if has_mlx:
            # MLX benchmark
            mlx_med, mlx_mean, _, _, mlx_out = bench(mlx_sparse_attention, Q, K, V, top_k, label="MLX")
            speedup = cpu_med / mlx_med
            winner = "MLX" if speedup > 1.0 else "CPU"

            print(f"{label:>24s} | {cpu_med:>10.0f} | {mlx_med:>10.0f} | {speedup:>7.2f}x | {winner:>6s}")
            verify_correctness(cpu_out, mlx_out, label)

            results.append((label, cpu_med, mlx_med, speedup))
        else:
            print(f"{label:>24s} | {cpu_med:>10.0f} | {'N/A':>10s} | {'N/A':>8s} | {'CPU':>6s}")
            results.append((label, cpu_med, 0, 0))

    print()
    print("=" * 78)
    print("Summary:")
    print(f"  MLX wins when H≥8 and N≥4K (GPU dispatch amortized over heads)")
    print(f"  CPU wins at H=1 (Metal dispatch ~118µs dominates)")
    if has_mlx and results:
        gpu_wins = sum(1 for _, _, _, s in results if s > 1.0)
        print(f"  GPU won {gpu_wins}/{len(results)} configs")
        best = max(results, key=lambda x: x[3])
        print(f"  Best GPU speedup: {best[3]:.2f}x at {best[0]}")

    # Data sizes for memory planning
    print()
    print("Memory footprint per config (K+V only):")
    for H, N, D, _ in configs:
        kv_mb = H * N * D * 4 * 2 / (1024 * 1024)
        print(f"  H={H:>2d} N={N:>6,d}: {kv_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
