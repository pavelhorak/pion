# gh #127: the batch distance kernels rebuilt on RegTile (src/vector/reg_tile.mojo)
# must return the same bits as the hand-unrolled forms they replaced. The old
# bodies are frozen below, verbatim except for an _old_ prefix, and every live
# batch entry point is compared against them and against a scalar reference:
#
#   - l2_distance_int8_int8_batch8_jit / batch4_jit (INT32, Float32 tail)
#   - l2_int8_sabd_udot_batch8 (NEON SABD+UDOT; elsewhere the batch-8 above)
#   - l2_distance_fp32_int8_pervec_batch4 (FP32: bit-exact only if each row's
#     expression tree is unchanged)
#
# at the dims the engine dispatches on plus dims with a SIMD and a scalar
# tail, on random codes and on the extreme codes (-128 / 127).
#
#   pixi run mojo build -I . tests/test_reg_tile_kernels.mojo -o /tmp/rt && /tmp/rt
from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from std.random import random_si64, random_float64, seed
from std.sys import simd_width_of, CompilationTarget
from std.sys.intrinsics import llvm_intrinsic
from std.memory.unsafe import bitcast

from src.vector.fma_mad import fma_mad
from src.vector.kernels import (
    l2_distance_int8_int8_batch8_jit, l2_distance_int8_int8_batch4_jit,
    l2_int8_sabd_udot_batch8, l2_distance_fp32_int8_pervec_batch4,
    _sabd_u8, _udot_u8,
)


# ── the pre-refactor bodies (git show 4b672bb:src/vector/kernels.mojo) ──────

@always_inline
def _old_l2_distance_int8_int8_batch8_jit[dim: Int](
    q:    UnsafePointer[Int8, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_4: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_5: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_6: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_7: UnsafePointer[Int8, MutUntrackedOrigin]) -> SIMD[DType.float32, 8]:
    """INT8-INT8 L2 batch-8: no dequantization, matches graph build metric.
    Query quantized to INT8 once per search call; ~1.3x faster than FP32-INT8.
    Accumulates in INT32 (max: 1536 * 254^2 = 99M < INT32_MAX)."""
    comptime width = 16
    var sum0 = SIMD[DType.int32, width](0); var sum1 = SIMD[DType.int32, width](0)
    var sum2 = SIMD[DType.int32, width](0); var sum3 = SIMD[DType.int32, width](0)
    var sum4 = SIMD[DType.int32, width](0); var sum5 = SIMD[DType.int32, width](0)
    var sum6 = SIMD[DType.int32, width](0); var sum7 = SIMD[DType.int32, width](0)
    for i in range(0, dim - width + 1, width):
        var qv = q.load[width=width](i).cast[DType.int32]()
        var d0 = qv - v2_0.load[width=width](i).cast[DType.int32](); sum0 += d0 * d0
        var d1 = qv - v2_1.load[width=width](i).cast[DType.int32](); sum1 += d1 * d1
        var d2 = qv - v2_2.load[width=width](i).cast[DType.int32](); sum2 += d2 * d2
        var d3 = qv - v2_3.load[width=width](i).cast[DType.int32](); sum3 += d3 * d3
        var d4 = qv - v2_4.load[width=width](i).cast[DType.int32](); sum4 += d4 * d4
        var d5 = qv - v2_5.load[width=width](i).cast[DType.int32](); sum5 += d5 * d5
        var d6 = qv - v2_6.load[width=width](i).cast[DType.int32](); sum6 += d6 * d6
        var d7 = qv - v2_7.load[width=width](i).cast[DType.int32](); sum7 += d7 * d7
    var r0 = sum0.reduce_add().cast[DType.float32](); var r1 = sum1.reduce_add().cast[DType.float32]()
    var r2 = sum2.reduce_add().cast[DType.float32](); var r3 = sum3.reduce_add().cast[DType.float32]()
    var r4 = sum4.reduce_add().cast[DType.float32](); var r5 = sum5.reduce_add().cast[DType.float32]()
    var r6 = sum6.reduce_add().cast[DType.float32](); var r7 = sum7.reduce_add().cast[DType.float32]()
    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var qi = q[i].cast[DType.int32]()
        var d0i = qi - v2_0[i].cast[DType.int32](); r0 += Float32(d0i * d0i)
        var d1i = qi - v2_1[i].cast[DType.int32](); r1 += Float32(d1i * d1i)
        var d2i = qi - v2_2[i].cast[DType.int32](); r2 += Float32(d2i * d2i)
        var d3i = qi - v2_3[i].cast[DType.int32](); r3 += Float32(d3i * d3i)
        var d4i = qi - v2_4[i].cast[DType.int32](); r4 += Float32(d4i * d4i)
        var d5i = qi - v2_5[i].cast[DType.int32](); r5 += Float32(d5i * d5i)
        var d6i = qi - v2_6[i].cast[DType.int32](); r6 += Float32(d6i * d6i)
        var d7i = qi - v2_7[i].cast[DType.int32](); r7 += Float32(d7i * d7i)
    return SIMD[DType.float32, 8](r0, r1, r2, r3, r4, r5, r6, r7)


@always_inline
def _old_l2_distance_int8_int8_batch4_jit[dim: Int](
    q:    UnsafePointer[Int8, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin]) -> SIMD[DType.float32, 4]:
    """INT8-INT8 L2 batch-4: no dequantization, matches graph build metric."""
    comptime width = 16
    var sum0 = SIMD[DType.int32, width](0); var sum1 = SIMD[DType.int32, width](0)
    var sum2 = SIMD[DType.int32, width](0); var sum3 = SIMD[DType.int32, width](0)
    for i in range(0, dim - width + 1, width):
        var qv = q.load[width=width](i).cast[DType.int32]()
        var d0 = qv - v2_0.load[width=width](i).cast[DType.int32](); sum0 += d0 * d0
        var d1 = qv - v2_1.load[width=width](i).cast[DType.int32](); sum1 += d1 * d1
        var d2 = qv - v2_2.load[width=width](i).cast[DType.int32](); sum2 += d2 * d2
        var d3 = qv - v2_3.load[width=width](i).cast[DType.int32](); sum3 += d3 * d3
    var r0 = sum0.reduce_add().cast[DType.float32](); var r1 = sum1.reduce_add().cast[DType.float32]()
    var r2 = sum2.reduce_add().cast[DType.float32](); var r3 = sum3.reduce_add().cast[DType.float32]()
    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var qi = q[i].cast[DType.int32]()
        var d0i = qi - v2_0[i].cast[DType.int32](); r0 += Float32(d0i * d0i)
        var d1i = qi - v2_1[i].cast[DType.int32](); r1 += Float32(d1i * d1i)
        var d2i = qi - v2_2[i].cast[DType.int32](); r2 += Float32(d2i * d2i)
        var d3i = qi - v2_3[i].cast[DType.int32](); r3 += Float32(d3i * d3i)
    return SIMD[DType.float32, 4](r0, r1, r2, r3)


@always_inline
def _old_l2_int8_sabd_udot_batch8[dim: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v0: UnsafePointer[Int8, MutUntrackedOrigin], v1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin], v3: UnsafePointer[Int8, MutUntrackedOrigin],
    v4: UnsafePointer[Int8, MutUntrackedOrigin], v5: UnsafePointer[Int8, MutUntrackedOrigin],
    v6: UnsafePointer[Int8, MutUntrackedOrigin], v7: UnsafePointer[Int8, MutUntrackedOrigin],
) -> SIMD[DType.float32, 8]:
    """gh #395: `l2_int8_sabd_udot` against eight vectors at once — one query
    load per 16 lanes and eight independent chains, so the eight vectors' cache
    misses overlap. Exact, like the single form."""
    comptime if CompilationTarget.has_neon_int8_dotprod() and dim % 16 == 0:
        var s0 = SIMD[DType.uint32, 4](0); var s1 = SIMD[DType.uint32, 4](0)
        var s2 = SIMD[DType.uint32, 4](0); var s3 = SIMD[DType.uint32, 4](0)
        var s4 = SIMD[DType.uint32, 4](0); var s5 = SIMD[DType.uint32, 4](0)
        var s6 = SIMD[DType.uint32, 4](0); var s7 = SIMD[DType.uint32, 4](0)
        for i in range(0, dim, 16):
            var qv = q.load[width=16](i)
            var d0 = _sabd_u8(qv, v0.load[width=16](i)); s0 = _udot_u8(s0, d0, d0)
            var d1 = _sabd_u8(qv, v1.load[width=16](i)); s1 = _udot_u8(s1, d1, d1)
            var d2 = _sabd_u8(qv, v2.load[width=16](i)); s2 = _udot_u8(s2, d2, d2)
            var d3 = _sabd_u8(qv, v3.load[width=16](i)); s3 = _udot_u8(s3, d3, d3)
            var d4 = _sabd_u8(qv, v4.load[width=16](i)); s4 = _udot_u8(s4, d4, d4)
            var d5 = _sabd_u8(qv, v5.load[width=16](i)); s5 = _udot_u8(s5, d5, d5)
            var d6 = _sabd_u8(qv, v6.load[width=16](i)); s6 = _udot_u8(s6, d6, d6)
            var d7 = _sabd_u8(qv, v7.load[width=16](i)); s7 = _udot_u8(s7, d7, d7)
        return SIMD[DType.float32, 8](
            s0.reduce_add().cast[DType.float32](), s1.reduce_add().cast[DType.float32](),
            s2.reduce_add().cast[DType.float32](), s3.reduce_add().cast[DType.float32](),
            s4.reduce_add().cast[DType.float32](), s5.reduce_add().cast[DType.float32](),
            s6.reduce_add().cast[DType.float32](), s7.reduce_add().cast[DType.float32]())
    else:
        return _old_l2_distance_int8_int8_batch8_jit[dim](q, v0, v1, v2, v3, v4, v5, v6, v7)



@always_inline
def _old_l2_distance_fp32_int8_pervec_batch4(
    v1:   UnsafePointer[Float32, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
    min0: Float32, range0: Float32,
    min1: Float32, range1: Float32,
    min2: Float32, range2: Float32,
    min3: Float32, range3: Float32,
) -> SIMD[DType.float32, 4]:
    """Per-vector SQ8 batch-4 (gh #42 follow-up). Each of the 4 neighbors
    has its own (min, range) — unlike `*_batch4_jit` which assumes shared
    quantization. Loads the query chunk ONCE and reuses it for 4 dequant +
    L2² accumulators. Used by KNNHNSW._search_layer to attack memory stall
    from random graph traversal: 4 neighbor base loads share 1 query load.
    Returns SIMD[float32, 4] = (dist0, dist1, dist2, dist3)."""
    comptime width = simd_width_of[DType.float32]()
    var s0 = range0 / 254.0; var o0 = 127.0 * s0 + min0
    var s1 = range1 / 254.0; var o1 = 127.0 * s1 + min1
    var s2 = range2 / 254.0; var o2 = 127.0 * s2 + min2
    var s3 = range3 / 254.0; var o3 = 127.0 * s3 + min3
    var sv0 = SIMD[DType.float32, width](s0); var ov0 = SIMD[DType.float32, width](o0)
    var sv1 = SIMD[DType.float32, width](s1); var ov1 = SIMD[DType.float32, width](o1)
    var sv2 = SIMD[DType.float32, width](s2); var ov2 = SIMD[DType.float32, width](o2)
    var sv3 = SIMD[DType.float32, width](s3); var ov3 = SIMD[DType.float32, width](o3)
    var sum0 = SIMD[DType.float32, width](0.0)
    var sum1 = SIMD[DType.float32, width](0.0)
    var sum2 = SIMD[DType.float32, width](0.0)
    var sum3 = SIMD[DType.float32, width](0.0)
    var n_simd = (dim // width) * width
    var i = 0
    while i < n_simd:
        var f1 = v1.load[width=width](i)
        var dq0 = fma_mad[width](v2_0.load[width=width](i).cast[DType.float32](), sv0, ov0)
        var dq1 = fma_mad[width](v2_1.load[width=width](i).cast[DType.float32](), sv1, ov1)
        var dq2 = fma_mad[width](v2_2.load[width=width](i).cast[DType.float32](), sv2, ov2)
        var dq3 = fma_mad[width](v2_3.load[width=width](i).cast[DType.float32](), sv3, ov3)
        var d0 = f1 - dq0; sum0 = fma_mad[width](d0, d0, sum0)
        var d1 = f1 - dq1; sum1 = fma_mad[width](d1, d1, sum1)
        var d2 = f1 - dq2; sum2 = fma_mad[width](d2, d2, sum2)
        var d3 = f1 - dq3; sum3 = fma_mad[width](d3, d3, sum3)
        i += width
    var r0 = sum0.reduce_add()
    var r1 = sum1.reduce_add()
    var r2 = sum2.reduce_add()
    var r3 = sum3.reduce_add()
    while i < dim:
        var f1 = v1[i]
        var d0 = f1 - (v2_0[i].cast[DType.float32]() * s0 + o0); r0 += d0 * d0
        var d1 = f1 - (v2_1[i].cast[DType.float32]() * s1 + o1); r1 += d1 * d1
        var d2 = f1 - (v2_2[i].cast[DType.float32]() * s2 + o2); r2 += d2 * d2
        var d3 = f1 - (v2_3[i].cast[DType.float32]() * s3 + o3); r3 += d3 * d3
        i += 1
    return SIMD[DType.float32, 4](r0, r1, r2, r3)




# ── checks ───────────────────────────────────────────────────────────────────

def fill_i8(v: UnsafePointer[Int8, MutUntrackedOrigin], n: Int, mode: Int):
    for i in range(n):
        if mode == 0:
            v[i] = Int8(random_si64(-128, 127))
        elif mode == 1:
            v[i] = Int8(-128) if (i % 2) == 0 else Int8(127)
        else:
            v[i] = Int8(127) if (i % 2) == 0 else Int8(-128)


def ref_l2(q: UnsafePointer[Int8, MutUntrackedOrigin], v: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int) -> Int:
    var s = 0
    for i in range(dim):
        var d = Int(q[i]) - Int(v[i])
        s += d * d
    return s


def check_int8[dim: Int](trials: Int) -> Int:
    var q = alloc[Int8](dim)
    var vs = alloc[Int8](dim * 8)
    var bad = 0
    for t in range(trials):
        var mq = 1 if t % 5 == 0 else 0
        fill_i8(q, dim, mq)
        for j in range(8):
            fill_i8(vs + j * dim, dim, 2 if (t % 5 == 0 and j % 2 == 0) else 0)
        var p = vs
        var n8 = l2_distance_int8_int8_batch8_jit[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim,
                                                       p + 4 * dim, p + 5 * dim, p + 6 * dim, p + 7 * dim)
        var o8 = _old_l2_distance_int8_int8_batch8_jit[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim,
                                                            p + 4 * dim, p + 5 * dim, p + 6 * dim, p + 7 * dim)
        var n4 = l2_distance_int8_int8_batch4_jit[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim)
        var o4 = _old_l2_distance_int8_int8_batch4_jit[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim)
        var s8 = l2_int8_sabd_udot_batch8[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim,
                                               p + 4 * dim, p + 5 * dim, p + 6 * dim, p + 7 * dim)
        var so8 = _old_l2_int8_sabd_udot_batch8[dim](q, p, p + dim, p + 2 * dim, p + 3 * dim,
                                                     p + 4 * dim, p + 5 * dim, p + 6 * dim, p + 7 * dim)
        for j in range(8):
            if n8[j].to_bits() != o8[j].to_bits(): bad += 1
            if s8[j].to_bits() != so8[j].to_bits(): bad += 1
            if j < 4 and n4[j].to_bits() != o4[j].to_bits(): bad += 1
            # Against the exact integer: equal when there is no scalar tail
            # (one cast of an exact INT32). With a tail the kernels add it in
            # Float32, which rounds once a row passes 2^24, by design.
            comptime if dim % 16 == 0:
                var want = Float32(ref_l2(q, p + j * dim, dim))
                if n8[j].to_bits() != want.to_bits(): bad += 1
                if s8[j].to_bits() != want.to_bits(): bad += 1
                if j < 4 and n4[j].to_bits() != want.to_bits(): bad += 1
    q.free(); vs.free()
    if bad > 0:
        print("  FAIL int8 batch dim=" + String(dim) + ": " + String(bad) + " mismatches")
    return bad


def check_pervec(dim: Int, trials: Int) -> Int:
    var q = alloc[Float32](dim)
    var vs = alloc[Int8](dim * 4)
    var bad = 0
    for t in range(trials):
        for i in range(dim):
            q[i] = Float32(random_float64(-3.0, 3.0))
        fill_i8(vs, dim * 4, 1 if t % 7 == 0 else 0)
        var mins = SIMD[DType.float32, 4](0)
        var ranges = SIMD[DType.float32, 4](0)
        for j in range(4):
            mins[j] = Float32(random_float64(-2.0, 0.0))
            ranges[j] = Float32(random_float64(0.01, 4.0))
        var p = vs
        var n = l2_distance_fp32_int8_pervec_batch4(q, p, p + dim, p + 2 * dim, p + 3 * dim, dim,
            mins[0], ranges[0], mins[1], ranges[1], mins[2], ranges[2], mins[3], ranges[3])
        var o = _old_l2_distance_fp32_int8_pervec_batch4(q, p, p + dim, p + 2 * dim, p + 3 * dim, dim,
            mins[0], ranges[0], mins[1], ranges[1], mins[2], ranges[2], mins[3], ranges[3])
        for j in range(4):
            if n[j].to_bits() != o[j].to_bits(): bad += 1
    q.free(); vs.free()
    if bad > 0:
        print("  FAIL pervec dim=" + String(dim) + ": " + String(bad) + " mismatches")
    return bad


def main() raises:
    seed(127)
    var bad = 0
    bad += check_int8[16](200)
    bad += check_int8[100](200)     # SIMD body + scalar tail
    bad += check_int8[128](200)
    bad += check_int8[256](200)
    bad += check_int8[384](200)
    bad += check_int8[768](200)
    bad += check_int8[777](100)     # odd tail
    bad += check_int8[1024](100)
    bad += check_int8[1536](100)
    for d in [1, 3, 15, 16, 17, 100, 384, 768, 777, 1536]:
        bad += check_pervec(d, 200)
    if bad > 0:
        print("FAIL: " + String(bad) + " mismatches")
        raise Error("RegTile kernels differ from the unrolled forms")
    print("ALL PASS: RegTile batch kernels are bit-identical to the unrolled forms")
