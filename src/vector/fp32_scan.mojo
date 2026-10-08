"""gh #400: the exact FP32 row kernels behind VSIM and kNN-LM's brute force.

Both used one 8-wide accumulator, so each row was two dependent chains of
4-lane adds: on a cache-resident set the scan waited on its own additions.
Four independent 4-lane FMA chains keep the pipes busy (1.4-1.6x per row in
the 2026-09-29 kernel review). VSIM's whole-set scan goes through BLAS GEMV
where one exists (`scan_dot_f32`), and these are the fallback.

Reassociation moves the last bit of a score; nothing stored changes.
"""
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call

from src.vector.fma_mad import fma_mad


@always_inline
def dot_f32_4chain(q: Pointer[Float32, MutUntrackedOrigin],
                   v: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
    """q·v over `dim` floats."""
    var a0 = SIMD[DType.float32, 4](0)
    var a1 = SIMD[DType.float32, 4](0)
    var a2 = SIMD[DType.float32, 4](0)
    var a3 = SIMD[DType.float32, 4](0)
    var d = 0
    while d + 16 <= dim:
        a0 = fma_mad[4]((q + d).load[width=4](), (v + d).load[width=4](), a0)
        a1 = fma_mad[4]((q + d + 4).load[width=4](), (v + d + 4).load[width=4](), a1)
        a2 = fma_mad[4]((q + d + 8).load[width=4](), (v + d + 8).load[width=4](), a2)
        a3 = fma_mad[4]((q + d + 12).load[width=4](), (v + d + 12).load[width=4](), a3)
        d += 16
    while d + 4 <= dim:
        a0 = fma_mad[4]((q + d).load[width=4](), (v + d).load[width=4](), a0)
        d += 4
    var s = ((a0 + a1) + (a2 + a3)).reduce_add()
    while d < dim:
        s += q[unsafe_offset=d] * v[unsafe_offset=d]
        d += 1
    return s


@always_inline
def l2sq_f32_4chain(a: Pointer[Float32, MutUntrackedOrigin],
                    b: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
    """Squared L2 distance over `dim` floats."""
    var a0 = SIMD[DType.float32, 4](0)
    var a1 = SIMD[DType.float32, 4](0)
    var a2 = SIMD[DType.float32, 4](0)
    var a3 = SIMD[DType.float32, 4](0)
    var d = 0
    while d + 16 <= dim:
        var e0 = (a + d).load[width=4]() - (b + d).load[width=4]()
        var e1 = (a + d + 4).load[width=4]() - (b + d + 4).load[width=4]()
        var e2 = (a + d + 8).load[width=4]() - (b + d + 8).load[width=4]()
        var e3 = (a + d + 12).load[width=4]() - (b + d + 12).load[width=4]()
        a0 = fma_mad[4](e0, e0, a0)
        a1 = fma_mad[4](e1, e1, a1)
        a2 = fma_mad[4](e2, e2, a2)
        a3 = fma_mad[4](e3, e3, a3)
        d += 16
    while d + 4 <= dim:
        var e = (a + d).load[width=4]() - (b + d).load[width=4]()
        a0 = fma_mad[4](e, e, a0)
        d += 4
    var s = ((a0 + a1) + (a2 + a3)).reduce_add()
    while d < dim:
        var e = a[unsafe_offset=d] - b[unsafe_offset=d]
        s += e * e
        d += 1
    return s


@always_inline
def scan_dot_f32(vecs: Pointer[Float32, MutUntrackedOrigin], rows: Int, dim: Int,
                 q: Pointer[Float32, MutUntrackedOrigin],
                 dst: Pointer[Float32, MutUntrackedOrigin]):
    """dst[r] = q·vecs[r] for every row of a contiguous row-major [rows x dim]
    block: one BLAS GEMV where the platform has one (Accelerate on macOS),
    else the four-chain kernel row by row."""
    if rows <= 0:
        return
    var rc = external_call["pion_sgemv_f32", Int32](
        Int64(rows), Int64(dim), vecs, q, dst)
    if rc == 0:
        return
    for r in range(rows):
        dst[unsafe_offset=r] = dot_f32_4chain(q, vecs + r * dim, dim)
