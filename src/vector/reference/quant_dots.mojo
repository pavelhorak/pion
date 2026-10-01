"""Open reference distance kernels for the quantized 1536 beams (D11).

Plain evaluations of the block formats — portable 16-lane widening dots, the
same standard as reference/beam_1536.mojo (no batching, no ISA dot
instruction, no prefetch) — the tuned kernels read, with the
SAME float operation order, so the reference beams return bit-identical
results: per 32-dim block, an exact int32 dot, cast to f32, times
(query_scale * vector_scale), accumulated block by block.

Formats (per vector, after a 4-byte f32 norm; 48 blocks of 32 dims, each
block starting with an f16 scale):
  INT4 (PolarQuant)  16 bytes: byte i = (dim i+16) << 4 | dim i, value = nibble - 8
  INT3 (TurboQuant)  6 + 6 bytes: two groups of 8 x 3-bit per 3 bytes, value = code - 4
  INT2 (NanoQuant)   4 + 4 bytes: 16 x 2-bit per 4 bytes, value = code * 2 - 3

Bytes are widened through Int and masked: `UInt64(p[i].cast[DType.uint8]())`
sign-extends under Mojo 1.0 (the bug #342 fixed in the tuned kernels).
"""
from std.memory.unsafe_pointer import UnsafePointer
from std.bit import pop_count

# The register unpackers are the same open primitives the single-vector kernels
# use: unpacking is format decoding, not tuning. (Until 2026-09-25 the reference
# decoded one code per lane in scalar code, which made the TurboQuant reference
# ~8x slower than tuned — a gap that measured scalar decoding, not tuning.)
from ..kernels import _unpack16_int3, _unpack16_int2

comptime _BLOCKS = 48
comptime _BLOCK_DIM = 32


@always_inline
def _dot16(q: UnsafePointer[Int8, MutUntrackedOrigin], v: SIMD[DType.int32, 16]) -> Int:
    """Portable widening dot of 16 query bytes with 16 decoded codes (exact)."""
    return Int((q.load[width=16]().cast[DType.int32]() * v).reduce_add())


@always_inline
def _byte(p: UnsafePointer[Int8, MutUntrackedOrigin], i: Int) -> Int:
    return Int(p[i]) & 0xFF


@always_inline
def _scale(p: UnsafePointer[Int8, MutUntrackedOrigin], off: Int) -> Float32:
    return (p + off).bitcast[Float16]()[0].cast[DType.float32]()


def ref_int4_dot(
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    qs: UnsafePointer[Float32, MutUntrackedOrigin],
    v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    var total: Float32 = 0.0
    var off = 4
    for b in range(_BLOCKS):
        var combined = qs[b] * _scale(v, off)
        off += 2
        var packed = (v + off).load[width=16]().cast[DType.int32]() & SIMD[DType.int32, 16](0xFF)
        var lo = (packed & SIMD[DType.int32, 16](0x0F)) - SIMD[DType.int32, 16](8)
        var hi = (packed >> 4) - SIMD[DType.int32, 16](8)
        var acc = _dot16(q + b * _BLOCK_DIM, lo) + _dot16(q + b * _BLOCK_DIM + 16, hi)
        total += Float32(acc) * combined
        off += 16
    return total


def ref_int3_dot(
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    qs: UnsafePointer[Float32, MutUntrackedOrigin],
    v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    var total: Float32 = 0.0
    var off = 4
    for b in range(_BLOCKS):
        var combined = qs[b] * _scale(v, off)
        off += 2
        var acc = _dot16(q + b * _BLOCK_DIM, _unpack16_int3(v, off).cast[DType.int32]()) + _dot16(q + b * _BLOCK_DIM + 16, _unpack16_int3(v, off + 6).cast[DType.int32]())
        total += Float32(acc) * combined
        off += 12
    return total


def ref_int2_dot(
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    qs: UnsafePointer[Float32, MutUntrackedOrigin],
    v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    var total: Float32 = 0.0
    var off = 4
    for b in range(_BLOCKS):
        var combined = qs[b] * _scale(v, off)
        off += 2
        var acc = _dot16(q + b * _BLOCK_DIM, _unpack16_int2(v, off).cast[DType.int32]()) + _dot16(q + b * _BLOCK_DIM + 16, _unpack16_int2(v, off + 4).cast[DType.int32]())
        total += Float32(acc) * combined
        off += 8
    return total


def ref_hamming(
    a: UnsafePointer[UInt64, MutUntrackedOrigin],
    b: UnsafePointer[UInt64, MutUntrackedOrigin],
    words: Int) -> Int:
    var total = 0
    for w in range(words):
        total += Int(pop_count(a[w] ^ b[w]))
    return total
