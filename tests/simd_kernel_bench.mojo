"""Standalone SIMD kernel microbenchmark for Mojo version regression testing.

Measures the throughput of INT8 L2 distance kernels used in HNSW vector search.
Run on different Mojo versions to detect codegen regressions.

Usage:
    mojo run tests/simd_kernel_bench.mojo

Expected output: ~X million distance computations per second.
Compare across Mojo versions (0.26.2 vs 0.26.3+) to detect regressions.

Filed as: https://github.com/modular/modular/issues/XXXX (update when filed)
"""

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import unsafe_memset
from std.sys import simd_width_of
from std.sys import CompilationTarget
from std.sys.intrinsics import prefetch, llvm_intrinsic
from std.time import perf_counter_ns
from std.random import random_ui64


# ── Kernels (extracted from src/vector/kernels.mojo) ────────────────────────

@always_inline
fn sdot_int8_portable(acc: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.int32, 4]:
    """Portable INT8 dot-product: widen to int32, multiply, group-sum 4 lanes."""
    var prod = a.cast[DType.int32]() * b.cast[DType.int32]()
    return acc + SIMD[DType.int32, 4](
        prod[0] + prod[1] + prod[2] + prod[3],
        prod[4] + prod[5] + prod[6] + prod[7],
        prod[8] + prod[9] + prod[10] + prod[11],
        prod[12] + prod[13] + prod[14] + prod[15]
    )

@always_inline
fn sdot_int8(acc: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.int32, 4]:
    comptime if CompilationTarget.is_linux():
        return sdot_int8_portable(acc, a, b)
    else:
        return llvm_intrinsic[
            "llvm.aarch64.neon.sdot.v4i32.v16i8",
            SIMD[DType.int32, 4],
            SIMD[DType.int32, 4],
            SIMD[DType.int8, 16],
            SIMD[DType.int8, 16]
        ](acc, a, b)


# Kernel 1: Simple L2 distance (no early exit)
@no_inline
fn l2_distance_int8_simple[dim: Int](
    v1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """Basic INT8 L2 distance. Single accumulator, no unrolling tricks."""
    comptime width = 16
    var acc = SIMD[DType.int32, width](0)
    for i in range(0, dim, width):
        var d = v1.load[width=width](i).cast[DType.int32]() - v2.load[width=width](i).cast[DType.int32]()
        acc += d * d
    return acc.reduce_add().cast[DType.float32]()


# Kernel 2: 2x-unrolled L2 distance (the pattern used in suffix early-exit)
@no_inline
fn l2_distance_int8_unrolled[dim: Int](
    v1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """2x-unrolled INT8 L2 distance. Two accumulator chains to hide latency."""
    comptime width = 16
    var sum_a = SIMD[DType.int32, width](0)
    var sum_b = SIMD[DType.int32, width](0)
    for i in range(0, dim, 2 * width):
        var da = v1.load[width=width](i).cast[DType.int32]() - v2.load[width=width](i).cast[DType.int32]()
        sum_a += da * da
        var db = v1.load[width=width](i + width).cast[DType.int32]() - v2.load[width=width](i + width).cast[DType.int32]()
        sum_b += db * db
    return (sum_a + sum_b).reduce_add().cast[DType.float32]()


# Kernel 3: comptime-for unrolled L2 with early exit (the actual hot kernel)
@no_inline
fn l2_distance_int8_suffix_early_exit[suffix: Int, block: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v: UnsafePointer[Int8, MutUntrackedOrigin],
    prefix_dist: Float32,
    threshold: Float32) -> Float32:
    """Suffix L2 with early exit at each block boundary.
    comptime for unrolls the block loop at compile time.
    For suffix=1280, block=256: 5 checkpoints."""
    comptime width = 16
    var running = prefix_dist
    comptime for b in range(suffix // block):
        var sum_a = SIMD[DType.int32, width](0)
        var sum_b = SIMD[DType.int32, width](0)
        comptime for i in range(0, block // (2 * width)):
            var base = b * block + i * 2 * width
            var qa = q.load[width=width](base).cast[DType.int32]()
            var da = qa - v.load[width=width](base).cast[DType.int32]()
            sum_a += da * da
            var qb = q.load[width=width](base + width).cast[DType.int32]()
            var db = qb - v.load[width=width](base + width).cast[DType.int32]()
            sum_b += db * db
        running += (sum_a + sum_b).reduce_add().cast[DType.float32]()
        if running > threshold:
            return Float32(1e30)
    return running


# Kernel 4: Batch-8 dot product (prefix pruning kernel)
@no_inline
fn dot_int8_batch8[dim: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v0: UnsafePointer[Int8, MutUntrackedOrigin],
    v1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin],
    v3: UnsafePointer[Int8, MutUntrackedOrigin],
    v4: UnsafePointer[Int8, MutUntrackedOrigin],
    v5: UnsafePointer[Int8, MutUntrackedOrigin],
    v6: UnsafePointer[Int8, MutUntrackedOrigin],
    v7: UnsafePointer[Int8, MutUntrackedOrigin]) -> SIMD[DType.float32, 8]:
    """SDOT-based batch-8 dot products."""
    var acc0 = SIMD[DType.int32, 4](0); var acc1 = SIMD[DType.int32, 4](0)
    var acc2 = SIMD[DType.int32, 4](0); var acc3 = SIMD[DType.int32, 4](0)
    var acc4 = SIMD[DType.int32, 4](0); var acc5 = SIMD[DType.int32, 4](0)
    var acc6 = SIMD[DType.int32, 4](0); var acc7 = SIMD[DType.int32, 4](0)
    for i in range(0, dim, 16):
        var qv = q.load[width=16](i)
        acc0 = sdot_int8(acc0, qv, v0.load[width=16](i))
        acc1 = sdot_int8(acc1, qv, v1.load[width=16](i))
        acc2 = sdot_int8(acc2, qv, v2.load[width=16](i))
        acc3 = sdot_int8(acc3, qv, v3.load[width=16](i))
        acc4 = sdot_int8(acc4, qv, v4.load[width=16](i))
        acc5 = sdot_int8(acc5, qv, v5.load[width=16](i))
        acc6 = sdot_int8(acc6, qv, v6.load[width=16](i))
        acc7 = sdot_int8(acc7, qv, v7.load[width=16](i))
    return SIMD[DType.float32, 8](
        acc0.reduce_add().cast[DType.float32](), acc1.reduce_add().cast[DType.float32](),
        acc2.reduce_add().cast[DType.float32](), acc3.reduce_add().cast[DType.float32](),
        acc4.reduce_add().cast[DType.float32](), acc5.reduce_add().cast[DType.float32](),
        acc6.reduce_add().cast[DType.float32](), acc7.reduce_add().cast[DType.float32]()
    )


# ── Benchmark Harness ───────────────────────────────────────────────────────

fn fill_random(ptr: UnsafePointer[Int8, MutUntrackedOrigin], n: Int):
    """Fill buffer with pseudo-random INT8 values."""
    for i in range(0, n, 8):
        var r = random_ui64(0, 255)
        for j in range(min(8, n - i)):
            ptr[i + j] = Int8((r >> UInt64(j * 8)) & 0xFF)

fn bench_kernel(name: String, iters: Int, ns_total: Int):
    var ns_per = ns_total / iters
    var ops_per_sec = Int(1_000_000_000.0 / Float64(ns_per)) if ns_per > 0 else 0
    print("  " + name + ": " + String(iters) + " iters, " + String(ns_per) + " ns/op, " + String(ops_per_sec) + " ops/sec")

fn main():
    alias DIM = 1536
    alias PREFIX = 256
    alias SUFFIX = 1280
    alias BLOCK = 256
    alias WARMUP = 1000
    alias ITERS = 100_000

    print("=== Pion SIMD Kernel Microbenchmark ===")
    print("Dim: " + String(DIM) + ", Prefix: " + String(PREFIX) + ", Suffix: " + String(SUFFIX))
    print("Iterations: " + String(ITERS))
    comptime if CompilationTarget.is_linux():
        print("Platform: Linux (portable INT32 fallback for sdot)")
    else:
        print("Platform: macOS/ARM (NEON SDOT intrinsic)")
    print("")

    # Allocate test vectors
    var q = alloc[Int8](DIM)
    var v = alloc[Int8](DIM)
    var v0 = alloc[Int8](DIM); var v1 = alloc[Int8](DIM)
    var v2 = alloc[Int8](DIM); var v3 = alloc[Int8](DIM)
    var v4 = alloc[Int8](DIM); var v5 = alloc[Int8](DIM)
    var v6 = alloc[Int8](DIM); var v7 = alloc[Int8](DIM)
    fill_random(q, DIM); fill_random(v, DIM)
    fill_random(v0, DIM); fill_random(v1, DIM)
    fill_random(v2, DIM); fill_random(v3, DIM)
    fill_random(v4, DIM); fill_random(v5, DIM)
    fill_random(v6, DIM); fill_random(v7, DIM)

    # Volatile sink: store result to heap pointer so compiler cannot prove it's unused
    var sink_ptr = alloc[Float32](1)
    sink_ptr[0] = 0

    # Allocate N different target vectors to prevent loop hoisting
    alias N_VECS = 64
    var targets = alloc[UnsafePointer[Int8, MutUntrackedOrigin]](N_VECS)
    for i in range(N_VECS):
        targets[i] = alloc[Int8](DIM)
        fill_random(targets[i], DIM)

    # ── Kernel 1: Simple L2 ─────────────────────────────────────────────────
    for i in range(WARMUP):
        sink_ptr[0] += l2_distance_int8_simple[DIM](q, targets[i & (N_VECS - 1)])
    var t0 = perf_counter_ns()
    for i in range(ITERS):
        sink_ptr[0] += l2_distance_int8_simple[DIM](q, targets[i & (N_VECS - 1)])
    var t1 = perf_counter_ns()
    bench_kernel("L2 simple (1536d)", ITERS, Int(t1 - t0))

    # ── Kernel 2: 2x-unrolled L2 ───────────────────────────────────────────
    for i in range(WARMUP):
        sink_ptr[0] += l2_distance_int8_unrolled[DIM](q, targets[i & (N_VECS - 1)])
    t0 = perf_counter_ns()
    for i in range(ITERS):
        sink_ptr[0] += l2_distance_int8_unrolled[DIM](q, targets[i & (N_VECS - 1)])
    t1 = perf_counter_ns()
    bench_kernel("L2 2x-unrolled (1536d)", ITERS, Int(t1 - t0))

    # ── Kernel 3: comptime-for suffix early exit ────────────────────────────
    for i in range(WARMUP):
        sink_ptr[0] += l2_distance_int8_suffix_early_exit[SUFFIX, BLOCK](
            q + PREFIX, targets[i & (N_VECS - 1)] + PREFIX, Float32(50000.0), Float32(999999.0))
    t0 = perf_counter_ns()
    for i in range(ITERS):
        sink_ptr[0] += l2_distance_int8_suffix_early_exit[SUFFIX, BLOCK](
            q + PREFIX, targets[i & (N_VECS - 1)] + PREFIX, Float32(50000.0), Float32(999999.0))
    t1 = perf_counter_ns()
    bench_kernel("Suffix early-exit (1280d, block=256, no prune)", ITERS, Int(t1 - t0))

    # ── Kernel 3b: suffix with early pruning ────────────────────────────────
    for i in range(WARMUP):
        sink_ptr[0] += l2_distance_int8_suffix_early_exit[SUFFIX, BLOCK](
            q + PREFIX, targets[i & (N_VECS - 1)] + PREFIX, Float32(50000.0), Float32(50001.0))
    t0 = perf_counter_ns()
    for i in range(ITERS):
        sink_ptr[0] += l2_distance_int8_suffix_early_exit[SUFFIX, BLOCK](
            q + PREFIX, targets[i & (N_VECS - 1)] + PREFIX, Float32(50000.0), Float32(50001.0))
    t1 = perf_counter_ns()
    bench_kernel("Suffix early-exit (1280d, block=256, early prune)", ITERS, Int(t1 - t0))

    # ── Kernel 4: Batch-8 dot product ───────────────────────────────────────
    for i in range(WARMUP):
        var j = (i * 8) & (N_VECS - 8)
        sink_ptr[0] += dot_int8_batch8[DIM](q, targets[j], targets[j+1], targets[j+2], targets[j+3],
            targets[j+4], targets[j+5], targets[j+6], targets[j+7]).reduce_add()
    t0 = perf_counter_ns()
    for i in range(ITERS):
        var j = (i * 8) & (N_VECS - 8)
        sink_ptr[0] += dot_int8_batch8[DIM](q, targets[j], targets[j+1], targets[j+2], targets[j+3],
            targets[j+4], targets[j+5], targets[j+6], targets[j+7]).reduce_add()
    t1 = perf_counter_ns()
    bench_kernel("Batch-8 dot (1536d, 8 vectors)", ITERS, Int(t1 - t0))

    for i in range(N_VECS):
        targets[i].free()
    targets.free()

    # Print sink to fully prevent DCE
    print("sink: " + String(sink_ptr[0]))
    sink_ptr.free()

    print("")
    print("To compare across Mojo versions:")
    print("  1. Run with Mojo 0.26.2: mojo run tests/simd_kernel_bench.mojo")
    print("  2. Run with Mojo 0.26.3: mojo run tests/simd_kernel_bench.mojo")
    print("  3. Compare ops/sec — >10% regression indicates compiler codegen issue")
    print("")
    print("Context: Pion HNSW vector search dropped from 10,283 QPS to 6,803 QPS (-34%)")
    print("between Mojo 0.26.2 and 0.26.3 with identical algorithm and recall (0.937).")

    q.free(); v.free()
    v0.free(); v1.free(); v2.free(); v3.free()
    v4.free(); v5.free(); v6.free(); v7.free()
