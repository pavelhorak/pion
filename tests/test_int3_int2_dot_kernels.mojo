# The open INT3/INT2 single-vector dot kernels must equal a scalar evaluation
# of the documented block format, bit for bit, and the INT2 quantizer must
# round-trip its input. (The batch-of-8 kernels are part of libpion_vector;
# tests/test_vector_differential.mojo checks them through the beams.)
#
#   pixi run mojo build -I . tests/test_int3_int2_dot_kernels.mojo -o /tmp/k32 && /tmp/k32
#
# Block format (kernels.mojo): [4B norm] then 48 blocks of
#   INT3: [FP16 scale][6B: 16 x 3-bit][6B: 16 x 3-bit]   value = code - 4
#   INT2: [FP16 scale][4B: 16 x 2-bit][4B: 16 x 2-bit]   value = code*2 - 3
# per block: dot_i32(q_block, v_block) cast to f32, times q_scale*v_scale,
# accumulated block by block into one f32.
from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from std.random import random_si64, random_float64, seed
from std.ffi import external_call
from std.sys import CompilationTarget

from src.vector.kernels import (
    int3_dot_single_simd, int2_dot_single_simd,
    NUM_BLOCKS_1536, BLOCK_DIM, INT3_BLOCK_BYTES, INT3_VEC_BYTES_1536,
    INT2_BLOCK_BYTES, INT2_VEC_BYTES_1536,
    quantize_fp32_to_block_int2, dequantize_block_int2_to_fp32,
    quantize_fp32_to_block_int3,
    _unpack16_int3, _unpack16_int2, unpack32_int3, unpack32_int2,
)


def spec_code(v: UnsafePointer[Int8, MutUntrackedOrigin], data_off: Int, i: Int, bits: Int) -> Int:
    """Value i (0..15) of a 16-value group starting at data_off."""
    if bits == 3:
        var grp = i // 8
        var base = data_off + grp * 3
        var u = Int(v[base].cast[DType.uint8]()) | (Int(v[base + 1].cast[DType.uint8]()) << 8) | (Int(v[base + 2].cast[DType.uint8]()) << 16)
        return ((u >> (3 * (i % 8))) & 7) - 4
    var u2 = Int(v[data_off].cast[DType.uint8]()) | (Int(v[data_off + 1].cast[DType.uint8]()) << 8) | (Int(v[data_off + 2].cast[DType.uint8]()) << 16) | (Int(v[data_off + 3].cast[DType.uint8]()) << 24)
    return ((u2 >> (2 * i)) & 3) * 2 - 3


def spec_dot(q: UnsafePointer[Int8, MutUntrackedOrigin], qs: UnsafePointer[Float32, MutUntrackedOrigin],
             v: UnsafePointer[Int8, MutUntrackedOrigin], bits: Int) -> Float32:
    var blk = INT3_BLOCK_BYTES if bits == 3 else INT2_BLOCK_BYTES
    var half = 6 if bits == 3 else 4
    var total: Float32 = 0.0
    for b in range(NUM_BLOCKS_1536):
        var off = 4 + b * blk
        var vs = (v + off).bitcast[Float16]()[0].cast[DType.float32]()
        var acc = 0
        for i in range(16):
            acc += Int(q[b * BLOCK_DIM + i]) * spec_code(v, off + 2, i, bits)
            acc += Int(q[b * BLOCK_DIM + 16 + i]) * spec_code(v, off + 2 + half, i, bits)
        total += Float32(acc) * (qs[b] * vs)
    return total


def fill(v: UnsafePointer[Int8, MutUntrackedOrigin], nbytes: Int, blk: Int):
    for i in range(nbytes):
        v[i] = Int8(random_si64(-128, 127))
    for b in range(NUM_BLOCKS_1536):
        (v + 4 + b * blk).bitcast[Float16]()[0] = Float16(random_float64(0.001, 0.5))


def check[bits: Int](trials: Int) -> Int:
    comptime vb = INT3_VEC_BYTES_1536 if bits == 3 else INT2_VEC_BYTES_1536
    comptime blk = INT3_BLOCK_BYTES if bits == 3 else INT2_BLOCK_BYTES
    var q = alloc[Int8](1536)
    var qs = alloc[Float32](NUM_BLOCKS_1536)
    var v = alloc[Int8](vb * 8)
    var bad = 0
    for _ in range(trials):
        for i in range(1536):
            q[i] = Int8(random_si64(-127, 127))
        for b in range(NUM_BLOCKS_1536):
            qs[b] = Float32(random_float64(0.001, 0.1))
        for j in range(8):
            fill(v + j * vb, vb, blk)
        for j in range(8):
            var got: Float32
            comptime if bits == 3:
                got = int3_dot_single_simd(q, qs, v + j * vb)
            else:
                got = int2_dot_single_simd(q, qs, v + j * vb)
            if got.to_bits() != spec_dot(q, qs, v + j * vb, bits).to_bits():
                bad += 1
    q.free(); qs.free(); v.free()
    return bad


def check_int2_roundtrip(trials: Int) -> Int:
    """The bit-exact check above compares the kernels with the FORMAT, so it
    cannot see a quantizer that writes the format wrong. Until 2026-09-25 the
    INT2 quantizer stored twice the scale its rounding used: every decoded value
    came back 2x and the stored norm 4x, and NanoQuant recall on the gate
    dataset was 0.06. Round-trip against the INPUT instead: each value must
    decode within half a level (absmax/3) of what went in, and the header norm
    must be the squared norm of the decoded vector."""
    var src = alloc[Float32](1536)
    var dst = alloc[Float32](1536)
    var buf = alloc[Int8](INT2_VEC_BYTES_1536)
    var bad = 0
    for _ in range(trials):
        for i in range(1536):
            src[i] = Float32(random_float64(-0.1, 0.1))
        quantize_fp32_to_block_int2(src, buf, 1536)
        dequantize_block_int2_to_fp32(buf, dst, 1536)
        var norm: Float32 = 0.0
        for b in range(NUM_BLOCKS_1536):
            var absmax: Float32 = 0.0
            for i in range(BLOCK_DIM):
                absmax = max(absmax, abs(src[b * BLOCK_DIM + i]))
            for i in range(BLOCK_DIM):
                var d = dst[b * BLOCK_DIM + i]
                norm += d * d
                # + 1% for the FP16 scale
                if abs(d - src[b * BLOCK_DIM + i]) > absmax / 3.0 * 1.01:
                    bad += 1
        var stored = buf.bitcast[Float32]()[0]
        if abs(stored - norm) > 0.01 * norm:
            bad += 1
    src.free(); dst.free(); buf.free()
    return bad


def check_unpack32[bits: Int](vectors: Int) -> Int:
    """gh #396: the TBL unpacker must give the same 32 codes as two scalar
    `_unpack16` calls, lane for lane, for every block of `vectors` random
    vectors (random BYTES, so every bit pattern the format can hold, not only
    what the quantizer writes) plus an all-0x00 and an all-0xFF vector."""
    comptime vb = INT3_VEC_BYTES_1536 if bits == 3 else INT2_VEC_BYTES_1536
    comptime blk = INT3_BLOCK_BYTES if bits == 3 else INT2_BLOCK_BYTES
    comptime half = 6 if bits == 3 else 4
    var v = alloc[Int8](vb)
    var bad = 0
    for t in range(vectors + 2):
        for i in range(vb):
            if t == vectors: v[i] = Int8(0)
            elif t == vectors + 1: v[i] = Int8(-1)
            else: v[i] = Int8(random_si64(-128, 127))
        for b in range(NUM_BLOCKS_1536):
            var off = 4 + b * blk
            var lo: SIMD[DType.int8, 16]
            var hi: SIMD[DType.int8, 16]
            var got: Tuple[SIMD[DType.int8, 16], SIMD[DType.int8, 16]]
            comptime if bits == 3:
                lo = _unpack16_int3(v, off + 2); hi = _unpack16_int3(v, off + 2 + half)
                got = unpack32_int3(v, off)
            else:
                lo = _unpack16_int2(v, off + 2); hi = _unpack16_int2(v, off + 2 + half)
                got = unpack32_int2(v, off)
            for i in range(16):
                if got[0][i] != lo[i] or got[1][i] != hi[i]: bad += 1
                # and both must match the documented format
                if Int(lo[i]) != spec_code(v, off + 2, i, bits): bad += 1
                if Int(hi[i]) != spec_code(v, off + 2 + half, i, bits): bad += 1
    v.free()
    return bad


def check_quantized_roundtrip(trials: Int) -> Int:
    """Quantizer output through the TBL unpackers: the INT3 codes decode to
    within half a level of the input (scale = absmax/3, levels -4..3)."""
    var src = alloc[Float32](1536)
    var buf = alloc[Int8](INT3_VEC_BYTES_1536)
    var bad = 0
    for _ in range(trials):
        for i in range(1536):
            src[i] = Float32(random_float64(-0.1, 0.1))
        quantize_fp32_to_block_int3(src, buf, 1536)
        for b in range(NUM_BLOCKS_1536):
            var off = 4 + b * INT3_BLOCK_BYTES
            var sc = (buf + off).bitcast[Float16]()[0].cast[DType.float32]()
            var u = unpack32_int3(buf, off)
            for i in range(BLOCK_DIM):
                var code = Int(u[0][i]) if i < 16 else Int(u[1][i - 16])
                var x = src[b * BLOCK_DIM + i]
                # half a level, +1% for the FP16 scale; the top level clips at +3
                if abs(Float32(code) * sc - x) > sc * 0.505 and not (code == 3 and x > 3.0 * sc):
                    bad += 1
    src.free(); buf.free()
    return bad


def check_no_overread() -> Int:
    """The TBL loads must stay inside the vector: place one INT3 vector and
    one INT2 vector so each ends exactly at a PROT_NONE guard page, then run
    every block through the unpackers and the single-vector kernels. An
    over-read is a SIGSEGV here, not a silent wrong answer."""
    var page = Int(external_call["getpagesize", Int32]())
    # RW, MAP_PRIVATE|MAP_ANON. The value differs per OS: 0x1002 is macOS's,
    # and on Linux mmap rejected it, so the test stopped here on every Linux
    # box and its later checks never ran there (#27).
    var anon_private = Int32(0x1002)
    comptime if CompilationTarget.is_linux():
        anon_private = Int32(0x22)
    var base = external_call["mmap", UnsafePointer[UInt8, MutUntrackedOrigin]](
        Int(0), 2 * page,
        Int32(3), anon_private, Int32(-1), Int(0))
    if Int(base) == -1 or Int(base) == 0:
        print("mmap failed"); return 1
    if external_call["mprotect", Int32](base + page, page, Int32(0)) != 0:
        print("mprotect failed"); return 1
    var q = alloc[Int8](1536)
    var qs = alloc[Float32](NUM_BLOCKS_1536)
    for i in range(1536): q[i] = Int8(random_si64(-127, 127))
    for b in range(NUM_BLOCKS_1536): qs[b] = Float32(0.01)
    var bad = 0
    var v3 = (base + page - INT3_VEC_BYTES_1536).bitcast[Int8]()
    fill(v3, INT3_VEC_BYTES_1536, INT3_BLOCK_BYTES)
    var s3 = 0
    for b in range(NUM_BLOCKS_1536):
        var u = unpack32_int3(v3, 4 + b * INT3_BLOCK_BYTES)
        s3 += Int(u[0].reduce_add()) + Int(u[1].reduce_add())
    if int3_dot_single_simd(q, qs, v3).to_bits() != spec_dot(q, qs, v3, 3).to_bits(): bad += 1
    var v2 = (base + page - INT2_VEC_BYTES_1536).bitcast[Int8]()
    fill(v2, INT2_VEC_BYTES_1536, INT2_BLOCK_BYTES)
    for b in range(NUM_BLOCKS_1536):
        var u = unpack32_int2(v2, 4 + b * INT2_BLOCK_BYTES)
        s3 += Int(u[0].reduce_add()) + Int(u[1].reduce_add())
    if int2_dot_single_simd(q, qs, v2).to_bits() != spec_dot(q, qs, v2, 2).to_bits(): bad += 1
    _ = external_call["munmap", Int32](base, 2 * page)
    q.free(); qs.free()
    _ = s3
    return bad


def main() raises:
    seed(1536)
    # ~10^6 blocks per format (21,000 vectors x 48 blocks).
    var u3 = check_unpack32[3](21000)
    var u2 = check_unpack32[2](21000)
    print("unpack32 vs scalar: int3 differ", u3, "  int2 differ", u2, "(1,008,096 blocks each)")
    if u3 + u2 != 0:
        raise Error("TBL unpackers disagree with the scalar unpackers / the format")
    var qr = check_quantized_roundtrip(500)
    print("int3 quantize->unpack32 out of band:", qr)
    if qr != 0:
        raise Error("INT3 quantizer output does not round-trip through unpack32_int3")
    var ov = check_no_overread()
    print("guard-page vectors (no over-read): mismatches", ov)
    if ov != 0:
        raise Error("kernels over a guard-page vector disagree with the spec")
    var b3 = check[3](2000)
    var b2 = check[2](2000)
    print("int3 differ", b3, "/ 16000   int2 differ", b2, "/ 16000")
    if b3 + b2 != 0:
        raise Error("INT3/INT2 dot kernels disagree with the block-format spec")
    var rt = check_int2_roundtrip(200)
    print("int2 quantize->dequantize out of band:", rt)
    if rt != 0:
        raise Error("INT2 quantizer does not round-trip its input")
