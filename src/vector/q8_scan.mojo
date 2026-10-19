"""Drop-in, phase B: the INT8 copy of the ingest rows that exact scans read.

The vectors FT.SEARCH scans exactly (all of them before the first build, the
delta after it: vector.mojo exact_knn) were read as FP32, 6 KB a row at 1536
dims, and the scan was memory-bound: 0.116 ms per 1,000 rows per query on the
Mac. Each ingest row now also gets Q8 codes: 32-dim blocks, one FP32 scale
per block (absmax / 127), rounded to nearest. The scan reads 1,728 bytes a
row and its dot products run on the platform's INT8 dot instruction
(kernels.sdot_int8: SDOT on ARM, VNNI or pmaddwd on x86).

The scan only ranks: it keeps a few more candidates than asked for, and those
are scored again from the FP32 row, so every score a client sees is exact.
Per-block scales matter on embeddings: one near-constant outlier dimension
(OpenAI's reach -0.69 while the rest sit within +-0.03) sets a per-row scale
so coarse that the other dimensions keep a few levels each; a block's scale
confines that to its own 32 dimensions.
"""
from std.memory.unsafe_pointer import Pointer
from src.vector.kernels import sdot_int8, quantize_fp32_to_block_int8
from src.vector.fma_mad import fma_mad

comptime Q8_BLOCK = 32


@always_inline
def q8_ok(dim: Int) -> Bool:
    """The Q8 copy exists for dims that are whole blocks (every embedding size
    in use); other dims are scanned from FP32 only."""
    return dim >= Q8_BLOCK and dim % Q8_BLOCK == 0


@always_inline
def q8_quantize(src: Pointer[Float32, MutUntrackedOrigin],
                codes: Pointer[Int8, MutUntrackedOrigin],
                scales: Pointer[Float32, MutUntrackedOrigin], dim: Int):
    """`dim` floats to `dim` codes and dim / 32 block scales."""
    _ = quantize_fp32_to_block_int8(src, codes, scales, dim)


@always_inline
def q8_dot(qc: Pointer[Int8, MutUntrackedOrigin], qs: Pointer[Float32, MutUntrackedOrigin],
           rc: Pointer[Int8, MutUntrackedOrigin], rs: Pointer[Float32, MutUntrackedOrigin],
           nb: Int) -> Float32:
    """About q . r from two Q8 vectors of `nb` blocks: per block, the INT8 dot
    times both scales. Four blocks per step on four accumulators, so the FMA
    chain does not wait on itself."""
    var a0 = SIMD[DType.float32, 4](0)
    var a1 = SIMD[DType.float32, 4](0)
    var a2 = SIMD[DType.float32, 4](0)
    var a3 = SIMD[DType.float32, 4](0)
    var b = 0
    while b + 4 <= nb:
        var o = b * Q8_BLOCK
        var i0 = sdot_int8(SIMD[DType.int32, 4](0), (qc + o).load[width=16](), (rc + o).load[width=16]())
        var i1 = sdot_int8(SIMD[DType.int32, 4](0), (qc + o + 32).load[width=16](), (rc + o + 32).load[width=16]())
        var i2 = sdot_int8(SIMD[DType.int32, 4](0), (qc + o + 64).load[width=16](), (rc + o + 64).load[width=16]())
        var i3 = sdot_int8(SIMD[DType.int32, 4](0), (qc + o + 96).load[width=16](), (rc + o + 96).load[width=16]())
        i0 = sdot_int8(i0, (qc + o + 16).load[width=16](), (rc + o + 16).load[width=16]())
        i1 = sdot_int8(i1, (qc + o + 48).load[width=16](), (rc + o + 48).load[width=16]())
        i2 = sdot_int8(i2, (qc + o + 80).load[width=16](), (rc + o + 80).load[width=16]())
        i3 = sdot_int8(i3, (qc + o + 112).load[width=16](), (rc + o + 112).load[width=16]())
        a0 = fma_mad[4](i0.cast[DType.float32](), SIMD[DType.float32, 4](qs[unsafe_offset=b] * rs[unsafe_offset=b]), a0)
        a1 = fma_mad[4](i1.cast[DType.float32](), SIMD[DType.float32, 4](qs[unsafe_offset=b + 1] * rs[unsafe_offset=b + 1]), a1)
        a2 = fma_mad[4](i2.cast[DType.float32](), SIMD[DType.float32, 4](qs[unsafe_offset=b + 2] * rs[unsafe_offset=b + 2]), a2)
        a3 = fma_mad[4](i3.cast[DType.float32](), SIMD[DType.float32, 4](qs[unsafe_offset=b + 3] * rs[unsafe_offset=b + 3]), a3)
        b += 4
    while b < nb:
        var o = b * Q8_BLOCK
        var i0 = sdot_int8(SIMD[DType.int32, 4](0), (qc + o).load[width=16](), (rc + o).load[width=16]())
        i0 = sdot_int8(i0, (qc + o + 16).load[width=16](), (rc + o + 16).load[width=16]())
        a0 = fma_mad[4](i0.cast[DType.float32](), SIMD[DType.float32, 4](qs[unsafe_offset=b] * rs[unsafe_offset=b]), a0)
        b += 1
    return ((a0 + a1) + (a2 + a3)).reduce_add()
