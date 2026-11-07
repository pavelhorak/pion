# pion_simd: every kernel against a plain scalar loop.
from std.memory.alloc import unsafe_alloc
from pion_simd import dot_product_simd, l2_distance_int8, hamming_distance, quantize_fp32_to_int8_simd
from std.memory import alloc
from std.sys import simd_width_of
from std.testing import assert_equal, assert_true, assert_almost_equal

comptime DIM = 1536 + 7   # not a multiple of any SIMD width: the tails run too


def lcg(mut s: UInt64) -> UInt64:
    s = s * 6364136223846793005 + 1442695040888963407
    return s >> 33


def test_dot_product() raises:
    var a = unsafe_alloc[Float32](DIM)
    var b = unsafe_alloc[Float32](DIM)
    var s: UInt64 = 1
    var want: Float64 = 0
    for i in range(DIM):
        a[unsafe_offset=i] = Float32(Int(lcg(s) % 2001) - 1000) / 1000.0
        b[unsafe_offset=i] = Float32(Int(lcg(s) % 2001) - 1000) / 1000.0
        want += Float64(a[unsafe_offset=i]) * Float64(b[unsafe_offset=i])
    var got = dot_product_simd[simd_width_of[DType.float32]()](a, b, DIM)
    assert_almost_equal(Float64(got), want, atol=1e-3)


def test_l2_int8() raises:
    var a = unsafe_alloc[Int8](DIM)
    var b = unsafe_alloc[Int8](DIM)
    var s: UInt64 = 2
    var want: Int = 0
    for i in range(DIM):
        a[unsafe_offset=i] = Int8(Int(lcg(s) % 255) - 127)
        b[unsafe_offset=i] = Int8(Int(lcg(s) % 255) - 127)
        var d = Int(a[unsafe_offset=i]) - Int(b[unsafe_offset=i])
        want += d * d
    assert_equal(l2_distance_int8(a, b, DIM), Float32(want))


def test_hamming() raises:
    comptime N = 24
    var a = unsafe_alloc[UInt64](N)
    var b = unsafe_alloc[UInt64](N)
    var s: UInt64 = 3
    var want = 0
    for i in range(N):
        a[unsafe_offset=i] = lcg(s) << 31 | lcg(s)
        b[unsafe_offset=i] = lcg(s) << 31 | lcg(s)
        var x = a[unsafe_offset=i] ^ b[unsafe_offset=i]
        while x != 0:
            want += Int(x & 1)
            x >>= 1
    assert_equal(hamming_distance(a, b, N), Float32(want))


def test_quantize_int8() raises:
    # dst = clamp(round((src - qmin) * scale), 0, 254) - 127
    var src = unsafe_alloc[Float32](DIM)
    var dst = unsafe_alloc[Int8](DIM)
    for i in range(DIM):
        src[unsafe_offset=i] = Float32(i % 255) / 254.0
    quantize_fp32_to_int8_simd(src, dst, DIM, 0.0, 254.0)
    for i in range(DIM):
        assert_equal(Int(dst[unsafe_offset=i]), (i % 255) - 127)


def main() raises:
    test_dot_product()
    test_l2_int8()
    test_hamming()
    test_quantize_int8()
    print("pion_simd: all tests passed")
