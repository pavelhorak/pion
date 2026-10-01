# FlashAttention-style fused kernel benchmark.
# Comparison points: CPU multi-head reference, naive multi-head GPU (the
# scores+matvec two-kernel design from MetalMHAttentionContext), FlashAttention.
#
# Shape: H=8 N=2048 d_head=128 — same as the multi-head bench so deltas line up.

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.math import sqrt
from std.random import random_float64
from std.time import perf_counter_ns

from src.vector.metal_attention_kernels import (
    MetalMHAttentionContext,
    MetalFlashAttentionContext,
    MetalFlashAttentionFP16Context,
    MetalFlashAttentionSplitKContext,
    cpu_mh_attend_reference,
)


def fill_random(p: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Float32(random_float64() * 2.0 - 1.0)


def cosine(a: UnsafePointer[Float32, MutAnyOrigin],
           b: UnsafePointer[Float32, MutAnyOrigin], n: Int) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb))


def insertion_sort(a: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    for i in range(1, n):
        var key = a[i]
        var j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


def main() raises:
    var H = 8
    var N = 2048
    var D = 128

    var q = alloc[Float32](H * D)
    var k = alloc[Float32](H * N * D)
    var v = alloc[Float32](H * N * D)
    var out_cpu = alloc[Float32](H * D)
    var out_mh = alloc[Float32](H * D)
    var out_fa = alloc[Float32](H * D)

    fill_random(q, H * D)
    fill_random(k, H * N * D)
    fill_random(v, H * N * D)

    print("FlashAttention vs naive multi-head vs CPU. H=", H, " N=", N, " d=", D)
    print("---------------------------------------------------------------")

    cpu_mh_attend_reference(q, k, v, H, N, D, out_cpu)

    # Naive multi-head GPU
    var mh_ctx = MetalMHAttentionContext()
    mh_ctx.init_device(D, H, N)
    var mh_ok = mh_ctx.attend(q, k, v, N, out_mh)
    if mh_ok != H * D:
        print("Naive MH attend failed: ret=", mh_ok)
        return
    var mh_sim = cosine(out_cpu, out_mh, H * D)
    print("Correctness: cosine(cpu, naive-mh) =", mh_sim)
    if mh_sim < 0.999:
        print("FAIL: naive-mh correctness, stopping")
        return

    # FlashAttention
    var fa_ctx = MetalFlashAttentionContext()
    fa_ctx.init_device(D, H, N)
    var fa_ok = fa_ctx.attend(q, k, v, N, out_fa)
    if fa_ok != H * D:
        print("FlashAttention attend failed: ret=", fa_ok)
        return
    var fa_sim = cosine(out_cpu, out_fa, H * D)
    print("Correctness: cosine(cpu, flash) =", fa_sim)
    if fa_sim < 0.999:
        print("FAIL: FlashAttention correctness, stopping before perf bench")
        return

    # FP16 FlashAttention
    var fa16_ctx = MetalFlashAttentionFP16Context()
    fa16_ctx.init_device(D, H, N)
    fa16_ctx.upload_kv(k, v, N)
    var out_fa16 = alloc[Float32](H * D)
    var fa16_ok = fa16_ctx.attend_q_resident(q, N, out_fa16)
    if fa16_ok != H * D:
        print("FP16 FlashAttention attend failed: ret=", fa16_ok)
        return
    var fa16_sim = cosine(out_cpu, out_fa16, H * D)
    print("Correctness: cosine(cpu, flash-fp16) =", fa16_sim)
    if fa16_sim < 0.995:
        print("FAIL: FP16 FlashAttention cosine below 0.995, stopping")
        return

    # Split-K FlashAttention
    var fasp_ctx = MetalFlashAttentionSplitKContext()
    fasp_ctx.init_device(D, H, N)
    fasp_ctx.upload_kv(k, v, N)
    var out_fasp = alloc[Float32](H * D)
    var fasp_ok = fasp_ctx.attend_q_resident(q, N, out_fasp)
    if fasp_ok != H * D:
        print("Split-K FlashAttention attend failed: ret=", fasp_ok)
        return
    var fasp_sim = cosine(out_cpu, out_fasp, H * D)
    print("Correctness: cosine(cpu, flash-split-k) =", fasp_sim)
    if fasp_sim < 0.999:
        print("FAIL: Split-K cosine below 0.999, stopping")
        return

    var WARMUP = 20
    var ITERS = 100
    var fa_times = alloc[Float64](ITERS)
    var fa_resident_times = alloc[Float64](ITERS)
    var fa16_times = alloc[Float64](ITERS)
    var fasp_times = alloc[Float64](ITERS)
    var mh_times = alloc[Float64](ITERS)
    var cpu_times = alloc[Float64](ITERS)

    # Upload K/V once outside the timing loop for the resident case.
    fa_ctx.upload_kv(k, v, N)

    for _ in range(WARMUP):
        _ = fa_ctx.attend(q, k, v, N, out_fa)
    for _ in range(WARMUP):
        _ = fa_ctx.attend_q_resident(q, N, out_fa)
    for _ in range(WARMUP):
        _ = fa16_ctx.attend_q_resident(q, N, out_fa16)
    for _ in range(WARMUP):
        _ = fasp_ctx.attend_q_resident(q, N, out_fasp)
    for _ in range(WARMUP):
        _ = mh_ctx.attend(q, k, v, N, out_mh)
    for _ in range(WARMUP):
        cpu_mh_attend_reference(q, k, v, H, N, D, out_cpu)

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        _ = fa_ctx.attend(q, k, v, N, out_fa)
        var t1 = perf_counter_ns()
        fa_times[i] = Float64(t1 - t0) / 1_000_000.0

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        _ = fa_ctx.attend_q_resident(q, N, out_fa)
        var t1 = perf_counter_ns()
        fa_resident_times[i] = Float64(t1 - t0) / 1_000_000.0

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        _ = fa16_ctx.attend_q_resident(q, N, out_fa16)
        var t1 = perf_counter_ns()
        fa16_times[i] = Float64(t1 - t0) / 1_000_000.0

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        _ = fasp_ctx.attend_q_resident(q, N, out_fasp)
        var t1 = perf_counter_ns()
        fasp_times[i] = Float64(t1 - t0) / 1_000_000.0

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        _ = mh_ctx.attend(q, k, v, N, out_mh)
        var t1 = perf_counter_ns()
        mh_times[i] = Float64(t1 - t0) / 1_000_000.0

    for i in range(ITERS):
        var t0 = perf_counter_ns()
        cpu_mh_attend_reference(q, k, v, H, N, D, out_cpu)
        var t1 = perf_counter_ns()
        cpu_times[i] = Float64(t1 - t0) / 1_000_000.0

    insertion_sort(fa_times, ITERS)
    insertion_sort(fa_resident_times, ITERS)
    insertion_sort(fa16_times, ITERS)
    insertion_sort(fasp_times, ITERS)
    insertion_sort(mh_times, ITERS)
    insertion_sort(cpu_times, ITERS)

    var fa_med = fa_times[ITERS // 2]
    var fa_res_med = fa_resident_times[ITERS // 2]
    var fa16_med = fa16_times[ITERS // 2]
    var fasp_med = fasp_times[ITERS // 2]
    var mh_med = mh_times[ITERS // 2]
    var cpu_med = cpu_times[ITERS // 2]

    print("FlashAttention FP32 fresh K/V upload    median=", fa_med, "ms")
    print("FlashAttention FP32 K/V resident        median=", fa_res_med, "ms")
    print("FlashAttention FP16 K/V resident        median=", fa16_med, "ms")
    print("FlashAttention Split-K K/V resident     median=", fasp_med, "ms")
    print("Naive multi-head 2-dispatch             median=", mh_med, "ms")
    print("CPU multi-head reference (FP32)         median=", cpu_med, "ms")
    print("speedup vs CPU: FA32=", cpu_med / fa_med, "x  FA32-res=", cpu_med / fa_res_med, "x  FA16=", cpu_med / fa16_med, "x  Split-K=", cpu_med / fasp_med, "x")

    if cpu_med / fasp_med > 1.0:
        print("→ Split-K FlashAttention BEATS CPU")
    elif cpu_med / fa_res_med > 1.0:
        print("→ FP32 FlashAttention (resident) BEATS CPU")
    else:
        print("→ CPU still wins. Gap (Split-K - CPU) =", fasp_med - cpu_med, "ms")

    q.free(); k.free(); v.free()
    out_cpu.free(); out_mh.free(); out_fa.free()
    fa_times.free(); mh_times.free(); cpu_times.free()
