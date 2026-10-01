# is_valid_float_arg — the strict validator behind INCRBYFLOAT / HINCRBYFLOAT
# (gh #229), THRESHOLD (gh #373) and FT.SEARCH numeric filter bounds (gh #367).
#
# Found 2026-09-27: with `@always_inline`, Mojo 1.0.0 (ed45d567) miscompiled the
# original loop shape — `break` out of the mantissa loop, then the exponent
# loop — and "10" was rejected, at -O0 and -O3, in some inlining contexts
# (FT.SEARCH's filter-bound parser, and a plain call like the ones below) while
# the INCRBYFLOAT sites happened to be right. Every FILTER @price:[10 20] was
# refused as "invalid numeric filter range". The validator no longer uses
# `break`; this file calls it the way the failing contexts did.
from std.memory import alloc
from std.testing import assert_equal
from src.common.utils import is_valid_float_arg, parse_float64


def check_bound(p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int, mut out_v: Float32) -> Bool:
    """The shape of vector.mojo's _parse_bound — the context that failed."""
    if not is_valid_float_arg(p, n):
        return False
    out_v = Float32(parse_float64(p, n))
    return True


def main() raises:
    var buf = alloc[UInt8](32)
    var good: List[String] = ["10", "0", "0.2", "-3", "+4.5", "1e5", "1E-3", "2.5e+10", "007", ".5", "5."]
    var bad: List[String] = ["", "abc", "1.2.3", "e5", ".", "-", "1e", "1e+", "10x", "--1", "1 0", "0x10"]
    for t in range(len(good)):
        var s = good[t]
        var n = s.byte_length()
        for k in range(n):
            buf[unsafe_offset=k] = s.as_bytes()[k]
        assert_equal(is_valid_float_arg(buf, n), True, "should accept '" + s + "'")
        var v = Float32(0)
        assert_equal(check_bound(buf, n, v), True, "bound context should accept '" + s + "'")
    for t in range(len(bad)):
        var s = bad[t]
        var n = s.byte_length()
        for k in range(n):
            buf[unsafe_offset=k] = s.as_bytes()[k]
        assert_equal(is_valid_float_arg(buf, n), False, "should reject '" + s + "'")
    var v2 = Float32(0)
    buf[unsafe_offset=0] = 49; buf[unsafe_offset=1] = 48
    _ = check_bound(buf, 2, v2)
    assert_equal(v2, Float32(10.0))
    print("is_valid_float_arg: " + String(len(good)) + " accepted, " + String(len(bad)) + " rejected — OK")
