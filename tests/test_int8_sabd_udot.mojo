# gh #395: the SABD+UDOT INT8 distance (the upper-level greedy's metric) must
# return EXACTLY the integer `l2_distance_int8_jit` returns, single and
# batch-of-8, at every dim the greedy dispatches on — including the extreme
# codes (-128 and 127, |a-b| = 255, the most a u8 lane can hold).
#
#   pixi run mojo build -I . tests/test_int8_sabd_udot.mojo -o /tmp/sabd && /tmp/sabd
#
# It also prints a single-thread timing next to the FP32-dequant distance the
# greedy used before (cache-hot, like the upper levels), for the record.
from std.memory import alloc
from std.memory.unsafe_pointer import UnsafePointer
from std.random import random_si64, random_float64, seed
from std.time import perf_counter_ns

from src.vector.kernels import (
    l2_distance_int8_jit, l2_int8_sabd_udot, l2_int8_sabd_udot_batch8,
    l2_distance_fp32_int8_fused_jit,
)


def fill(v: UnsafePointer[Int8, MutUntrackedOrigin], n: Int, mode: Int):
    for i in range(n):
        if mode == 0:
            v[i] = Int8(random_si64(-127, 127))
        elif mode == 1:
            v[i] = Int8(-128) if (i % 2) == 0 else Int8(127)
        else:
            v[i] = Int8(127) if (i % 2) == 0 else Int8(-128)


def check[dim: Int](trials: Int) -> Int:
    var q = alloc[Int8](dim)
    var vs = alloc[Int8](dim * 8)
    var bad = 0
    for t in range(trials):
        # every 7th trial is an extreme pair: q and v at opposite ends
        var mode_q = 1 if t % 7 == 0 else 0
        var mode_v = 2 if t % 7 == 0 else 0
        fill(q, dim, mode_q)
        for j in range(8):
            fill(vs + j * dim, dim, mode_v if j % 2 == 0 else 0)
        var d8 = l2_int8_sabd_udot_batch8[dim](
            q, vs, vs + dim, vs + 2 * dim, vs + 3 * dim,
            vs + 4 * dim, vs + 5 * dim, vs + 6 * dim, vs + 7 * dim)
        for j in range(8):
            var want = l2_distance_int8_jit[dim](q, vs + j * dim)
            var got = l2_int8_sabd_udot[dim](q, vs + j * dim)
            if got.to_bits() != want.to_bits(): bad += 1
            if d8[j].to_bits() != want.to_bits(): bad += 1
    q.free(); vs.free()
    return bad


def bench() -> Tuple[Float64, Float64, Float64]:
    comptime dim = 1536
    var q = alloc[Int8](dim)
    var qf = alloc[Float32](dim)
    var vs = alloc[Int8](dim * 16)
    fill(q, dim, 0)
    fill(vs, dim * 16, 0)
    for i in range(dim): qf[i] = Float32(random_float64(-0.2, 0.2))
    comptime N = 200000
    var acc = Float32(0)
    var t0 = perf_counter_ns()
    for i in range(N):
        acc += l2_distance_fp32_int8_fused_jit[dim](qf, vs + (i & 15) * dim, Float32(-0.2), Float32(0.4))
    var t1 = perf_counter_ns()
    for i in range(N):
        acc += l2_distance_int8_jit[dim](q, vs + (i & 15) * dim)
    var t2 = perf_counter_ns()
    for i in range(N):
        acc += l2_int8_sabd_udot[dim](q, vs + (i & 15) * dim)
    var t3 = perf_counter_ns()
    if acc == 1.0: print(acc)   # keep the loops
    q.free(); qf.free(); vs.free()
    return (Float64(t1 - t0) / N, Float64(t2 - t1) / N, Float64(t3 - t2) / N)


def main() raises:
    seed(395)
    var bad = check[128](3000) + check[256](3000) + check[384](2000) + check[768](2000) + check[1536](2000)
    print("SABD+UDOT vs l2_distance_int8_jit (single + batch8, 5 dims):", bad, "differ")
    if bad != 0:
        raise Error("SABD+UDOT distance is not exact")
    var t = bench()
    print("ns/distance at 1536: fp32-dequant", t[0], " int8 widening", t[1], " sabd+udot", t[2])
