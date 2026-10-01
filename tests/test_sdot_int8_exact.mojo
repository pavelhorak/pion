# sdot_int8 must be an EXACT signed int8 dot on every ISA it dispatches to:
# acc[i] += sum(a[4i+k] * b[4i+k]), k = 0..3, for all int8 inputs incl. -128.
#
#   pixi run mojo build -I . tests/test_sdot_int8_exact.mojo -o /tmp/sdot && /tmp/sdot
#
# The x86 non-VNNI arm used to treat `a` as UNSIGNED (pmaddubsw): 99.8% of
# random signed dots were wrong and an x86-64-v2 build's INT8 beam search
# returned recall@10 0.001 against 0.134 for the same routine on ARM. A Mac run
# of this test exercises SDOT; run it on an x86 host (with and without
# --target-cpu x86-64-v2) to exercise SSE and VNNI.
from std.random import random_si64, seed
from std.sys.info import CompilationTarget

from src.vector.kernels import sdot_int8


def main() raises:
    seed(1974)
    var bad = 0
    var trials = 200000
    for t in range(trials):
        var a = SIMD[DType.int8, 16](0)
        var b = SIMD[DType.int8, 16](0)
        for i in range(16):
            # every 8th trial pins the extremes, where saturation bugs live
            a[i] = Int8(-128) if (t % 8 == 0 and i % 3 == 0) else Int8(random_si64(-128, 127))
            b[i] = Int8(-128) if (t % 8 == 0 and i % 5 == 0) else Int8(random_si64(-128, 127))
        var acc0 = SIMD[DType.int32, 4](Int32(t % 1000) - 500)
        var got = sdot_int8(acc0, a, b)
        for lane in range(4):
            var want = Int(acc0[lane])
            for k in range(4):
                want += Int(a[4 * lane + k]) * Int(b[4 * lane + k])
            if Int(got[lane]) != want:
                bad += 1
    var isa = "x86" if CompilationTarget.is_x86() else "arm/other"
    print("sdot_int8 [", isa, "]: lanes wrong", bad, "/", trials * 4)
    if bad != 0:
        raise Error("sdot_int8 is not an exact signed dot on this ISA")
