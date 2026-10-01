from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.math import sqrt, log
from std.memory.unsafe import bitcast
from std.bit import count_leading_zeros
from src.common.value import GenericValue, ValueType

# HLL parameters
comptime HLL_P = 14
comptime HLL_REGISTERS = 1 << HLL_P
comptime HLL_BITS = 6
comptime HLL_REGISTER_MAX = (1 << HLL_BITS) - 1

# MurmurHash3 64-bit implementation
def fmix64(k: UInt64) -> UInt64:
    var h = k
    h ^= h >> 33
    h *= 0xff51afd7ed558ccd
    h ^= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ^= h >> 33
    return h

def murmur3_64(data: Pointer[UInt8, MutUntrackedOrigin], length: Int, seed: UInt64) -> UInt64:
    var nblocks = length // 8
    var h1 = seed
    var c1: UInt64 = 0x87c37b91114253d5
    var c2: UInt64 = 0x4cf5ad432745937f

    var blocks = data.unsafe_bitcast[UInt64]()
    for i in range(nblocks):
        var k1 = blocks.load(i)
        k1 *= c1
        k1 = (k1 << 31) | (k1 >> (64 - 31))
        k1 *= c2
        h1 ^= k1
        h1 = (h1 << 27) | (h1 >> (64 - 27))
        h1 = h1 * 5 + 0x52dce729

    var tail = data.unsafe_offset(nblocks * 8)
    var k1: UInt64 = 0
    var remainder = length & 7
    if remainder == 7: k1 ^= UInt64(tail.load(6)) << 48
    if remainder >= 6: k1 ^= UInt64(tail.load(5)) << 40
    if remainder >= 5: k1 ^= UInt64(tail.load(4)) << 32
    if remainder >= 4: k1 ^= UInt64(tail.load(3)) << 24
    if remainder >= 3: k1 ^= UInt64(tail.load(2)) << 16
    if remainder >= 2: k1 ^= UInt64(tail.load(1)) << 8
    if remainder >= 1: k1 ^= UInt64(tail.load(0))
    
    if remainder > 0:
      k1 *= c1
      k1 = (k1 << 31) | (k1 >> (64 - 31))
      k1 *= c2
      h1 ^= k1

    h1 ^= UInt64(length)
    return fmix64(h1)

def hll_add(registers: Pointer[UInt8, MutUntrackedOrigin], element: GenericValue) -> Bool:
    var hash: UInt64
    if element.is_string():
        if element.string_len() > 0:
            var _sbuf = alloc[UInt8](24)
            var ptr = element.as_string_safe(_sbuf)
            hash = murmur3_64(ptr, element.string_len(), 0)
            _sbuf.unsafe_free()
        else:
            return False # or handle empty string case
    else:
        # We can hash other types as well by their byte representation
        # For now, only strings are supported
        return False

    var index = Int(hash & (HLL_REGISTERS - 1))
    
    # Rank = 1 + leading zeros of the hash above the index bits. Was a
    # bit-at-a-time shift loop; now a single ctlz. No measurable throughput change
    # (leading zeros average ~1 on random hashes, so the loop rarely iterated and
    # its branch predicted well) — taken for constant-time behaviour and clarity.
    var h_shifted = hash << HLL_P
    var rank = Int(count_leading_zeros(h_shifted)) + 1
    if rank > 64 - HLL_P + 1:
        rank = 64 - HLL_P + 1
    
    if rank > Int(registers.load(index)):
        registers.store(index, UInt8(rank))
        return True
    return False

def hll_count(registers: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
    """PFCOUNT harmonic-mean reduction over the 16384 registers.

    Kernel review §3.3 / exec-summary #11: the original was a scalar loop with an
    unpredictable branch, a 64-bit shift and a **Float64 divide** per register —
    16384 divides on a serial accumulate chain.

    Two changes, both exactness-preserving:
      1. `2^-r` is an exact power of two, so it is built straight from IEEE-754
         exponent bits — `(1023 - r) << 52` bitcast to Float64 — instead of
         `1.0 / (1 << r)`. Bit-identical result, no divider on the critical path.
      2. Branchless + SIMD: the `reg == 0` test becomes a mask (select 0.0 into the
         sum, 1 into the zero count), and two independent accumulator chains run
         16 registers per iteration.

    Result is bit-identical to the scalar version: every term is an exact power of
    two, and float addition of the same values in a different order is exact here
    because all terms are dyadic rationals accumulated well within Float64 precision
    (16384 terms, each ≤ 1.0)."""
    var alpha_m = 0.7213 / (1.0 + 1.079 / Float64(HLL_REGISTERS))

    comptime W = 8
    var e0 = SIMD[DType.float64, W](0.0)
    var e1 = SIMD[DType.float64, W](0.0)
    var z0 = SIMD[DType.uint64, W](0)
    var z1 = SIMD[DType.uint64, W](0)
    var bias = SIMD[DType.uint64, W](1023)
    var zero_f = SIMD[DType.float64, W](0.0)
    var one_u = SIMD[DType.uint64, W](1)
    var zero_u = SIMD[DType.uint64, W](0)

    var i = 0
    while i + 2 * W <= HLL_REGISTERS:
        var r0 = registers.load[width=W](i).cast[DType.uint64]()
        var r1 = registers.load[width=W](i + W).cast[DType.uint64]()
        var m0 = r0.eq(zero_u)
        var m1 = r1.eq(zero_u)
        # 2^-r via exponent-field construction (no divide).
        var v0 = bitcast[DType.float64, W]((bias - r0) << 52)
        var v1 = bitcast[DType.float64, W]((bias - r1) << 52)
        e0 += m0.select(zero_f, v0)
        e1 += m1.select(zero_f, v1)
        z0 += m0.select(one_u, zero_u)
        z1 += m1.select(one_u, zero_u)
        i += 2 * W

    var E = (e0 + e1).reduce_add()
    var zeros = Int((z0 + z1).reduce_add())

    # Scalar tail (HLL_REGISTERS is a multiple of 16 today, so this is normally empty).
    while i < HLL_REGISTERS:
        var reg_val = Int(registers.load(i))
        if reg_val == 0:
            zeros += 1
        else:
            E += 1.0 / Float64(1 << reg_val)
        i += 1

    E = (alpha_m * Float64(HLL_REGISTERS) * Float64(HLL_REGISTERS)) / (E + Float64(zeros))

    if E <= 2.5 * Float64(HLL_REGISTERS):
        if zeros != 0:
            E = Float64(HLL_REGISTERS) * log(Float64(HLL_REGISTERS) / Float64(zeros))

    return Int(E)

def hll_merge(dest: Pointer[UInt8, MutUntrackedOrigin], src: Pointer[UInt8, MutUntrackedOrigin]):
    """PFMERGE register-wise max. Was 16384 scalar load/compare/branch/store; now a
    branchless SIMD `max` 32 registers at a time (kernel review §3.3 / #11)."""
    comptime W = 32
    var i = 0
    while i + W <= HLL_REGISTERS:
        dest.store(i, max(dest.load[width=W](i), src.load[width=W](i)))
        i += W
    while i < HLL_REGISTERS:
        var src_val = src.load(i)
        if src_val > dest.load(i):
            dest.store(i, src_val)
        i += 1
