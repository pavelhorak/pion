"""Head-to-head: MLX raw compute time vs Mojo+Metal kernel 1 at Pion's
decoder shape (H=8, N=2048, d_head=128).

Same methodology as tests/bench_metal_flash_attention.mojo:
  - 20 warmup iters + 100 measured iters
  - Median + p95 over 100 measurements
  - Fixed RNG seed for reproducibility
  - Q/K/V uploaded once outside the timing loop (kernel-only timing)

The "MLX through socket" production path is NOT measured here; that adds
~200 µs Unix-domain round-trip plus framing. This bench measures raw
MLX compute, the optimistic case for MLX.

Run: python3.11 tests/bench_mlx_vs_mojo.py
"""

import time

import mlx.core as mx
import numpy as np

H = 8
N = 2048
D = 128
WARMUP = 20
ITERS = 100

print(f"shape: H={H} N={N} d_head={D}")
print(f"MLX {mx.__version__}, Metal: {mx.metal.is_available()}")
print()


def attend_dense_mlx(Q, K, V):
    """Same as src/inference/mlx_attention_worker.py:attend_dense."""
    scale = 1.0 / np.sqrt(D)
    scores = mx.matmul(Q, mx.transpose(K, (0, 2, 1))) * scale
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    output = mx.matmul(weights, V)
    mx.eval(output)
    return output


def attend_sdpa_mlx(Q, K, V):
    """MLX's tuned scaled_dot_product_attention path (uses Apple's
    optimized attention kernel — fairest representation of MLX peak).
    SDPA expects rank-4 (B, H, N, D); add a batch dim of 1.
    """
    scale = 1.0 / np.sqrt(D)
    Q4 = mx.expand_dims(Q, 0)
    K4 = mx.expand_dims(K, 0)
    V4 = mx.expand_dims(V, 0)
    out = mx.fast.scaled_dot_product_attention(Q4, K4, V4, scale=scale)
    mx.eval(out)
    return out


np.random.seed(0)
Q_np = np.random.uniform(-1, 1, (H, 1, D)).astype(np.float32)
K_np = np.random.uniform(-1, 1, (H, N, D)).astype(np.float32)
V_np = np.random.uniform(-1, 1, (H, N, D)).astype(np.float32)

# Resident on GPU once
Q = mx.array(Q_np)
K = mx.array(K_np)
V = mx.array(V_np)
mx.eval(Q, K, V)


def bench(name, fn):
    for _ in range(WARMUP):
        fn(Q, K, V)
    times_ms = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn(Q, K, V)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)
    times_ms.sort()
    median = times_ms[ITERS // 2]
    p95 = times_ms[int(ITERS * 0.95)]
    minimum = times_ms[0]
    print(f"  {name:<30s} median={median:.3f} ms  p95={p95:.3f} ms  min={minimum:.3f} ms")
    return median


print("MLX raw compute (no socket):")
median_naive = bench("naive matmul + softmax", attend_dense_mlx)
median_sdpa = bench("fast.scaled_dot_product_attention", attend_sdpa_mlx)

print()
print("Reference numbers from doc (same shape):")
print(f"  Mojo CPU SIMD reference        median=2.610 ms")
print(f"  Mojo+Metal FA32 BC=24 resident median=4.850 ms")
print()
print("Verdict:")
print(f"  MLX SDPA vs Mojo CPU SIMD:   {2.610 / median_sdpa:.2f}× ({'MLX faster' if median_sdpa < 2.610 else 'CPU faster'})")
print(f"  MLX SDPA vs Mojo+Metal FA32: {4.850 / median_sdpa:.2f}× ({'MLX faster' if median_sdpa < 4.850 else 'Mojo+Metal faster'})")
print(f"  MLX naive vs Mojo+Metal FA32:{4.850 / median_naive:.2f}× ({'MLX faster' if median_naive < 4.850 else 'Mojo+Metal faster'})")
