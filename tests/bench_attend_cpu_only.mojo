from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.math import sqrt
from std.random import random_float64
from src.vector.metal_attention_kernels import cpu_attend_reference

def main() raises:
    var d = 128
    var n = 256
    var q = alloc[Float32](d)
    var k = alloc[Float32](n * d)
    var v = alloc[Float32](n * d)
    var out_a = alloc[Float32](d)
    var out_b = alloc[Float32](d)

    for i in range(d): q[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(n * d): k[i] = Float32(random_float64() * 2.0 - 1.0)
    for i in range(n * d): v[i] = Float32(random_float64() * 2.0 - 1.0)

    cpu_attend_reference(q, k, v, n, d, out_a)
    cpu_attend_reference(q, k, v, n, d, out_b)

    var max_diff: Float32 = 0.0
    var sum_a: Float32 = 0.0
    for i in range(d):
        var diff = out_a[i] - out_b[i]
        if diff < 0: diff = -diff
        if diff > max_diff: max_diff = diff
        sum_a += out_a[i] * out_a[i]

    print("CPU reference produces |out|² =", sum_a)
    print("Determinism check: max |a - b| =", max_diff, "(should be 0)")
    if max_diff < Float32(1e-6) and sum_a > Float32(0.0):
        print("CPU REFERENCE OK")
    else:
        print("CPU REFERENCE FAIL")
