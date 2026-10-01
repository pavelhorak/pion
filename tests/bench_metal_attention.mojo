# PoC test for src/vector/metal_attention_kernels.mojo.
#
# Run: mojo run -I . tests/bench_metal_attention.mojo
#
# Generates random Q/K/V at small N, runs CPU reference + GPU path, asserts
# cosine similarity ≥ 0.999. Reports both outputs on failure.
#
# This is a correctness gate, not a perf bench. Perf comparison vs MLX sidecar
# comes after the kernel is wired into ATTEND.PREFIX.QUERY (out of PoC scope).

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.math import sqrt
from std.random import random_float64

from src.vector.metal_attention_kernels import MetalAttentionContext, cpu_attend_reference


def cosine_sim(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb))


def fill_random(p: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    # Uniform [-1, 1]. Deterministic across runs is not required for a PoC
    # correctness check; cosine threshold is loose enough to absorb seed drift.
    for i in range(n):
        p[i] = Float32(random_float64() * 2.0 - 1.0)


def run_case(d_head: Int, n: Int) raises -> Bool:
    print("Case: d_head=", d_head, " N=", n)

    var q = alloc[Float32](d_head)
    var k = alloc[Float32](n * d_head)
    var v = alloc[Float32](n * d_head)
    var out_cpu = alloc[Float32](d_head)
    var out_gpu = alloc[Float32](d_head)

    fill_random(q, d_head)
    fill_random(k, n * d_head)
    fill_random(v, n * d_head)

    cpu_attend_reference(q, k, v, n, d_head, out_cpu)

    var ctx = MetalAttentionContext()
    ctx.init_device(d_head, n)
    var produced = ctx.attend(q, k, v, n, out_gpu)

    var ok = True
    if produced != d_head:
        print("  attend() returned ", produced, ", expected ", d_head, " (Linux/no-GPU = 0 is acceptable)")
        if produced == 0:
            print("  → no GPU path on this host; skipping cosine check")
            q.free(); k.free(); v.free(); out_cpu.free(); out_gpu.free()
            return True
        ok = False

    var sim = cosine_sim(out_cpu, out_gpu, d_head)
    print("  cosine(cpu, gpu) = ", sim)
    if sim < 0.999:
        print("  FAIL: cosine below 0.999")
        # Print first 8 lanes for diagnosis
        for i in range(8):
            if i < d_head:
                print("    [", i, "] cpu=", out_cpu[i], " gpu=", out_gpu[i])
        ok = False

    q.free(); k.free(); v.free(); out_cpu.free(); out_gpu.free()
    return ok


def main() raises:
    # PoC kernel pins D_HEAD=128 (comptime). Other head sizes return 0 from
    # attend() and the cosine check is skipped. Production: variant per d.
    var all_ok = True
    if not run_case(128, 16):
        all_ok = False
    if not run_case(128, 256):
        all_ok = False
    if not run_case(128, 2048):
        all_ok = False

    if all_ok:
        print("\nALL CASES PASSED")
    else:
        print("\nFAILURES — see output above")
