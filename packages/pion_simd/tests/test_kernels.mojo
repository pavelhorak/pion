# pion_simd: every kernel against a plain scalar loop.
from pion_simd import dot_product_simd, l2_distance_int8, hamming_distance, quantize_fp32_to_int8_simd
from std.memory import alloc
from std.sys import simd_width_of
from std.testing import assert_equal, assert_true, assert_almost_equal

comptime DIM = 1536 + 7   # not a multiple of any SIMD width: the tails run too


def lcg(mut s: UInt64) -> UInt64:
    s = s * 6364136223846793005 + 1442695040888963407
    return s >> 33


def test_dot_product() raises:
    var a = alloc[Float32](DIM)
    var b = alloc[Float32](DIM)
    var s: UInt64 = 1
    var want: Float64 = 0
    for i in range(DIM):
        a[i] = Float32(Int(lcg(s) % 2001) - 1000) / 1000.0
        b[i] = Float32(Int(lcg(s) % 2001) - 1000) / 1000.0
        want += Float64(a[i]) * Float64(b[i])
    var got = dot_product_simd[simd_width_of[DType.float32]()](a, b, DIM)
    assert_almost_equal(Float64(got), want, atol=1e-3)


def test_l2_int8() raises:
    var a = alloc[Int8](DIM)
    var b = alloc[Int8](DIM)
    var s: UInt64 = 2
    var want: Int = 0
    for i in range(DIM):
        a[i] = Int8(Int(lcg(s) % 255) - 127)
        b[i] = Int8(Int(lcg(s) % 255) - 127)
        var d = Int(a[i]) - Int(b[i])
        want += d * d
    assert_equal(l2_distance_int8(a, b, DIM), Float32(want))


def test_hamming() raises:
    comptime N = 24
    var a = alloc[UInt64](N)
    var b = alloc[UInt64](N)
    var s: UInt64 = 3
    var want = 0
    for i in range(N):
        a[i] = lcg(s) << 31 | lcg(s)
        b[i] = lcg(s) << 31 | lcg(s)
        var x = a[i] ^ b[i]
        while x != 0:
            want += Int(x & 1)
            x >>= 1
    assert_equal(hamming_distance(a, b, N), Float32(want))


def test_quantize_int8() raises:
    # dst = clamp(round((src - qmin) * scale), 0, 254) - 127
    var src = alloc[Float32](DIM)
    var dst = alloc[Int8](DIM)
    for i in range(DIM):
        src[i] = Float32(i % 255) / 254.0
    quantize_fp32_to_int8_simd(src, dst, DIM, 0.0, 254.0)
    for i in range(DIM):
        assert_equal(Int(dst[i]), (i % 255) - 127)


def main() raises:
    test_dot_product()
    test_l2_int8()
    test_hamming()
    test_quantize_int8()
    print("pion_simd: all tests passed")
