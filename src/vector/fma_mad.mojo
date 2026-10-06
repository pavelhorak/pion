"""A fused multiply-add where the target has one (#15, #25).

`fma(a, b, c)` rounds once. An x86 target without FMA, the release's
x86-64-v2 build among them, has no instruction for that, so LLVM lowers each
lane of a vector `fma` to a libm `fmaf` call: in metric_scores that was about
300K calls and 1.3 ms per FT.SEARCH on an EPYC 8124P (#15), and every other
kernel written with `fma` paid the same on every Linux x86 release binary.
There `fma_mad` is `a * b + c`; on every other target it is `fma`. `has_fma()`
names an x86 feature and reads False on AArch64, so the test is x86-only.

Call it with the width: Mojo 1.1 does not infer a SIMD width parameter from
an argument. tests/test_audit_raw_fma.py keeps `fma(` out of src/ elsewhere.
"""
from std.math import fma
from std.sys import CompilationTarget


@always_inline
def fma_mad[w: Int](a: SIMD[DType.float32, w], b: SIMD[DType.float32, w],
                    c: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    comptime if CompilationTarget.is_x86() and not CompilationTarget.has_fma():
        return a * b + c
    else:
        return fma(a, b, c)
