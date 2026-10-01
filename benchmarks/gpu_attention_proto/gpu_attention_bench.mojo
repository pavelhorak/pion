# J.b Prototype — Mojo GPU Attention Benchmark
# Tests GEMV + top-k + softmax + gather pipeline via DeviceContext (Metal on macOS)
# Compares against CPU baseline (same operations, no GPU)
#
# Build & run:
#   pixi run mojo build -I ../.. gpu_attention_bench.mojo -o gpu_attention_bench
#   ./gpu_attention_bench

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import memcpy
from std.sys.info import CompilationTarget
from std.math import ceildiv, sqrt, exp
from std.time import perf_counter_ns


# ---------------------------------------------------------------------------
# CPU baseline: sparse attention (GEMV + top-k + softmax + gather)
# ---------------------------------------------------------------------------

@always_inline
def cpu_gemv(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    scores: UnsafePointer[Float32, MutAnyOrigin],
    N: Int, D: Int, scale: Float32,
):
    for i in range(N):
        var dot: Float32 = 0.0
        var k_row = K + i * D
        for j in range(D):
            dot += Q[j] * k_row[j]
        scores[i] = dot * scale


@always_inline
def cpu_topk(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    N: Int, k: Int,
    out_idx: UnsafePointer[Int32, MutAnyOrigin],
    out_scores: UnsafePointer[Float32, MutAnyOrigin],
):
    var heap_size = 0
    for i in range(min(k, N)):
        out_scores[heap_size] = scores[i]
        out_idx[heap_size] = Int32(i)
        heap_size += 1
        var c = heap_size - 1
        while c > 0:
            var p = (c - 1) // 2
            if out_scores[c] < out_scores[p]:
                var ts = out_scores[c]; out_scores[c] = out_scores[p]; out_scores[p] = ts
                var ti = out_idx[c]; out_idx[c] = out_idx[p]; out_idx[p] = ti
                c = p
            else:
                break
    for i in range(k, N):
        if scores[i] > out_scores[0]:
            out_scores[0] = scores[i]
            out_idx[0] = Int32(i)
            var c = 0
            while True:
                var left = 2 * c + 1
                var right = 2 * c + 2
                var smallest = c
                if left < heap_size and out_scores[left] < out_scores[smallest]:
                    smallest = left
                if right < heap_size and out_scores[right] < out_scores[smallest]:
                    smallest = right
                if smallest != c:
                    var ts = out_scores[c]; out_scores[c] = out_scores[smallest]; out_scores[smallest] = ts
                    var ti = out_idx[c]; out_idx[c] = out_idx[smallest]; out_idx[smallest] = ti
                    c = smallest
                else:
                    break


@always_inline
def cpu_softmax(
    scores: UnsafePointer[Float32, MutAnyOrigin],
    k: Int,
):
    var max_val: Float32 = scores[0]
    for i in range(1, k):
        if scores[i] > max_val:
            max_val = scores[i]
    var sum_exp: Float32 = 0.0
    for i in range(k):
        var e = exp(scores[i] - max_val)
        scores[i] = e
        sum_exp += e
    var inv_sum = Float32(1.0) / sum_exp
    for i in range(k):
        scores[i] *= inv_sum


@always_inline
def cpu_gather_weighted_sum(
    V: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int32, MutAnyOrigin],
    weights: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    k: Int, D: Int,
):
    for d in range(D):
        output[d] = 0.0
    for i in range(k):
        var w = weights[i]
        var v_row = V + Int(indices[i]) * D
        for d in range(D):
            output[d] += w * v_row[d]


def cpu_sparse_attention(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    H: Int, N: Int, D: Int, top_k: Int,
):
    var scale = Float32(1.0 / sqrt(Float64(D)))
    var scores = alloc[Float32](N)
    var topk_idx = alloc[Int32](top_k)
    var topk_scores = alloc[Float32](top_k)

    for h in range(H):
        var q_head = Q + h * D
        var k_head = K + h * N * D
        var v_head = V + h * N * D
        var o_head = output + h * D

        cpu_gemv(q_head, k_head, scores, N, D, scale)
        cpu_topk(scores, N, top_k, topk_idx, topk_scores)
        cpu_softmax(topk_scores, top_k)
        cpu_gather_weighted_sum(v_head, topk_idx, topk_scores, o_head, top_k, D)

    scores.free()
    topk_idx.free()
    topk_scores.free()


# ---------------------------------------------------------------------------
# GPU attention via DeviceContext (Metal on macOS)
# ---------------------------------------------------------------------------

comptime GPU_BLOCK: Int = 256


def gpu_sparse_attention(
    Q: UnsafePointer[Float32, MutAnyOrigin],
    K: UnsafePointer[Float32, MutAnyOrigin],
    V: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    H: Int, N: Int, D: Int, top_k: Int,
) raises:
    comptime if CompilationTarget.is_macos():
        from std.gpu.host import DeviceContext, DeviceBuffer
        from std.gpu import block_dim, block_idx, thread_idx

        var ctx = DeviceContext()

        var total_scores = H * N
        var scale_val = Float32(1.0 / sqrt(Float64(D)))

        # Upload Q, K to GPU
        var q_buf = ctx.enqueue_create_buffer[DType.float32](H * D)
        with q_buf.map_to_host() as h_q:
            memcpy(dest=h_q.unsafe_ptr().bitcast[UInt8](), src=Q.bitcast[UInt8](), count=H * D * 4)

        var k_buf = ctx.enqueue_create_buffer[DType.float32](H * N * D)
        with k_buf.map_to_host() as h_k:
            memcpy(dest=h_k.unsafe_ptr().bitcast[UInt8](), src=K.bitcast[UInt8](), count=H * N * D * 4)

        var scores_buf = ctx.enqueue_create_buffer[DType.float32](total_scores)

        # Pack params: H, N, D, scale (as float reinterpreted as int32)
        var params_buf = ctx.enqueue_create_buffer[DType.int32](4)
        with params_buf.map_to_host() as h_p:
            h_p[0] = Int32(H)
            h_p[1] = Int32(N)
            h_p[2] = Int32(D)
            h_p.unsafe_ptr().bitcast[Float32]()[3] = scale_val

        # GEMV kernel: one thread per (h, n) pair
        # D=128 fixed to avoid Metal compiler issues with dynamic loops
        def gemv_kernel(
            q_ptr: UnsafePointer[Float32, MutAnyOrigin],
            k_ptr: UnsafePointer[Float32, MutAnyOrigin],
            s_ptr: UnsafePointer[Float32, MutAnyOrigin],
            p_ptr: UnsafePointer[Int32, MutAnyOrigin],
        ):
            var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
            var n_val = Int(p_ptr[1])
            var total = Int(p_ptr[0]) * n_val
            if gid >= total:
                return
            var scale = p_ptr.bitcast[Float32]()[3]
            var head = gid // n_val
            var tok = gid - head * n_val
            var q_off = head * 128
            var k_off = (head * n_val + tok) * 128
            var dot: Float32 = 0.0
            for i in range(128):
                dot += q_ptr[q_off + i] * k_ptr[k_off + i]
            s_ptr[gid] = dot * scale

        var num_blocks = ceildiv(total_scores, GPU_BLOCK)
        ctx.enqueue_function[gemv_kernel, gemv_kernel](
            q_buf, k_buf, scores_buf, params_buf,
            grid_dim=num_blocks, block_dim=GPU_BLOCK,
        )
        ctx.synchronize()

        # Download scores and do top-k + softmax + gather on CPU
        var scores_cpu = alloc[Float32](total_scores)
        with scores_buf.map_to_host() as h_s:
            memcpy(dest=scores_cpu.bitcast[UInt8](), src=h_s.unsafe_ptr().bitcast[UInt8](), count=total_scores * 4)

        var topk_idx = alloc[Int32](top_k)
        var topk_scores = alloc[Float32](top_k)

        for h_idx in range(H):
            var head_scores = scores_cpu + h_idx * N
            cpu_topk(head_scores, N, top_k, topk_idx, topk_scores)
            cpu_softmax(topk_scores, top_k)
            var v_head = V + h_idx * N * D
            var o_head = output + h_idx * D
            cpu_gather_weighted_sum(v_head, topk_idx, topk_scores, o_head, top_k, D)

        scores_cpu.free()
        topk_idx.free()
        topk_scores.free()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def bench_cpu(H: Int, N: Int, D: Int, top_k: Int, warmup: Int, iters: Int) -> Float64:
    var Q = alloc[Float32](H * D)
    var K = alloc[Float32](H * N * D)
    var V_buf = alloc[Float32](H * N * D)
    var O = alloc[Float32](H * D)

    for i in range(H * D):
        Q[i] = Float32(0.01) * Float32(i % 97 - 48)
    for i in range(H * N * D):
        K[i] = Float32(0.01) * Float32(i % 89 - 44)
        V_buf[i] = Float32(0.01) * Float32(i % 83 - 41)

    for _ in range(warmup):
        cpu_sparse_attention(Q, K, V_buf, O, H, N, D, top_k)

    var times = alloc[Float64](iters)
    for it in range(iters):
        var t0 = perf_counter_ns()
        cpu_sparse_attention(Q, K, V_buf, O, H, N, D, top_k)
        var t1 = perf_counter_ns()
        times[it] = Float64(t1 - t0) / 1000.0

    for i in range(iters):
        for j in range(i + 1, iters):
            if times[j] < times[i]:
                var tmp = times[i]
                times[i] = times[j]
                times[j] = tmp

    var median = times[iters // 2]

    Q.free(); K.free(); V_buf.free(); O.free(); times.free()
    return median


def bench_gpu(H: Int, N: Int, D: Int, top_k: Int, warmup: Int, iters: Int) raises -> Float64:
    var Q = alloc[Float32](H * D)
    var K = alloc[Float32](H * N * D)
    var V_buf = alloc[Float32](H * N * D)
    var O = alloc[Float32](H * D)

    for i in range(H * D):
        Q[i] = Float32(0.01) * Float32(i % 97 - 48)
    for i in range(H * N * D):
        K[i] = Float32(0.01) * Float32(i % 89 - 44)
        V_buf[i] = Float32(0.01) * Float32(i % 83 - 41)

    for _ in range(warmup):
        gpu_sparse_attention(Q, K, V_buf, O, H, N, D, top_k)

    var times = alloc[Float64](iters)
    for it in range(iters):
        var t0 = perf_counter_ns()
        gpu_sparse_attention(Q, K, V_buf, O, H, N, D, top_k)
        var t1 = perf_counter_ns()
        times[it] = Float64(t1 - t0) / 1000.0

    for i in range(iters):
        for j in range(i + 1, iters):
            if times[j] < times[i]:
                var tmp = times[i]
                times[i] = times[j]
                times[j] = tmp

    var median = times[iters // 2]

    Q.free(); K.free(); V_buf.free(); O.free(); times.free()
    return median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() raises:
    print("=" * 78)
    print("J.b Prototype: Mojo GPU (DeviceContext/Metal) vs CPU Attention")
    print("=" * 78)
    print()

    comptime WARMUP: Int = 3
    comptime ITERS: Int = 20

    def run_config(H: Int, N: Int, D: Int, top_k: Int) raises:
        var label = String("H=") + String(H) + " N=" + String(N) + " D=" + String(D) + " k=" + String(top_k)

        var cpu_us = bench_cpu(H, N, D, top_k, WARMUP, ITERS)

        comptime if CompilationTarget.is_macos():
            var gpu_us = bench_gpu(H, N, D, top_k, WARMUP, ITERS)
            var speedup = cpu_us / gpu_us
            var winner = String("GPU") if speedup > 1.0 else String("CPU")
            var sp_int = Int(speedup * 100)
            var sp_whole = sp_int // 100
            var sp_frac = sp_int % 100
            print(label, "| CPU:", Int(cpu_us), "µs | GPU:", Int(gpu_us), "µs |",
                  String(sp_whole) + "." + String(sp_frac) + "x |", winner)
        else:
            print(label, "| CPU:", Int(cpu_us), "µs | GPU: N/A")

    print("Config                   | CPU µs     | GPU µs     | Speedup  | Winner")
    print("-" * 78)

    run_config(1,   1024,  128, 32)
    run_config(1,   4096,  128, 32)
    run_config(1,  16384,  128, 32)
    run_config(8,   1024,  128, 32)
    run_config(8,   4096,  128, 32)
    run_config(8,  16384,  128, 32)
    run_config(16,  4096,  128, 32)
    run_config(16, 16384,  128, 32)
    run_config(32,  4096,  128, 32)
    run_config(32, 16384,  128, 32)
    run_config(32, 65536,  128, 32)

    print()
    print("Notes:")
    print("  GPU = GEMV on Metal, top-k/softmax/gather on CPU (hybrid)")
    print("  CPU = all on CPU (scalar loops, no BLAS)")
    print("  Next step: full GPU pipeline or use MAX matmul kernel")
