"""Decision-gate microbench for issue #47 (Apple AMX / Accelerate FFI for HNSW build).

Compares the existing NEON INT8×INT8 batch-8 SDOT path
(`dot_int8_int8_batch8_jit[1536]`) against Accelerate `cblas_sgemv` running
over an FP32-dequantized candidate tile. Accelerate dispatches AMX
internally on M-series.

The issue's decision gate: Accelerate must be ≥ 3× the NEON baseline at K=64
to justify integrating into `compact_overflows()` / FT.OPTIMIZE.

Reports two Accelerate variants:
  - Mode B: candidates pre-dequantized once (best case — cache the FP32 tile)
  - Mode C: dequantize-on-the-fly per call (worst case)

Build / run:
    pixi run bench-amx
"""

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import unsafe_memset
from std.time import perf_counter_ns
from std.ffi import external_call
from src.vector.kernels import dot_int8_int8_batch8_jit


comptime DIM: Int = 1536
comptime ITERS: Int = 2000  # repeats per K to amortise timer noise


# ── CBLAS constants (cblas.h) ────────────────────────────────────────────────
comptime CBLAS_ROW_MAJOR: Int32 = 101
comptime CBLAS_NO_TRANS: Int32 = 111


@always_inline
def cblas_sgemv_k_dim(
    K: Int,
    A: UnsafePointer[Float32, MutUntrackedOrigin],
    x: UnsafePointer[Float32, MutUntrackedOrigin],
    y: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """Compute y = A * x where A is K×DIM row-major FP32, x is DIM, y is K."""
    _ = external_call["cblas_sgemv", Int32](
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        Int32(K),
        Int32(DIM),
        Float32(1.0),
        A,
        Int32(DIM),
        x,
        Int32(1),
        Float32(0.0),
        y,
        Int32(1),
    )


@always_inline
def cblas_sgemm_M_K(
    M: Int,
    K: Int,
    queries_MxDim: UnsafePointer[Float32, MutUntrackedOrigin],
    cands_KxDim: UnsafePointer[Float32, MutUntrackedOrigin],
    out_MxK: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """Batched: C[M×K] = Q[M×DIM] * cands[DIM×K]; cands stored as K rows so transpose_b=True.

    This matches the pattern that arises if compact_overflows() is restructured
    to compute an M×K all-pairs distance matrix in one shot. AMX wins here only
    if M × K is large enough to amortise the matrix-coprocessor dispatch.
    """
    _ = external_call["cblas_sgemm", Int32](
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,                # transA
        Int32(112),                    # transB = TRANSPOSE
        Int32(M),                      # M
        Int32(K),                      # N
        Int32(DIM),                    # K (gemm)
        Float32(1.0),                  # alpha
        queries_MxDim,                 # A (M×DIM)
        Int32(DIM),                    # lda
        cands_KxDim,                   # B (K×DIM, transposed)
        Int32(DIM),                    # ldb
        Float32(0.0),                  # beta
        out_MxK,                       # C (M×K)
        Int32(K),                      # ldc
    )


@always_inline
def cblas_sgemm_q_K(
    K: Int,
    A_KxDim: UnsafePointer[Float32, MutUntrackedOrigin],
    x_Dim: UnsafePointer[Float32, MutUntrackedOrigin],
    y_K: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """SGEMM variant: C[1xK] = A[1xDIM] * B[DIMxK], where B is candidates transposed.

    cblas_sgemm sometimes dispatches to AMX more aggressively than sgemv at small M.
    Here we treat the query as the [1×DIM] left matrix and candidates as the
    [DIM×K] right matrix — which means we need candidates laid out as DIM-major
    (one column per candidate). To avoid a transpose we instead compute via
    transpose_b: A=[1×DIM] (query), B=[K×DIM] (candidates as K rows), C=[1×K].
    """
    _ = external_call["cblas_sgemm", Int32](
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,                # transA
        Int32(112),                    # transB = TRANSPOSE
        Int32(1),                      # M
        Int32(K),                      # N
        Int32(DIM),                    # K (gemm)
        Float32(1.0),                  # alpha
        x_Dim,                         # A (1×DIM)
        Int32(DIM),                    # lda
        A_KxDim,                       # B (K×DIM, transposed)
        Int32(DIM),                    # ldb
        Float32(0.0),                  # beta
        y_K,                           # C (1×K)
        Int32(K),                      # ldc
    )


def fill_int8(buf: UnsafePointer[Int8, MutUntrackedOrigin], n: Int, seed: Int):
    var s = UInt64(seed)
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        # Keep values in [-100, 100] so SDOT accumulators don't saturate
        var v = Int(s >> 32) % 201 - 100
        buf[i] = Int8(v)


def fill_fp32_from_int8(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    scale: Float32,
):
    for i in range(n):
        dst[i] = Float32(Int(src[i])) * scale


def run_neon_batch8(
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    cands: UnsafePointer[Int8, MutUntrackedOrigin],
    K: Int,
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """K must be a multiple of 8."""
    var k = 0
    while k < K:
        var dots = dot_int8_int8_batch8_jit[DIM](
            q,
            cands + (k + 0) * DIM,
            cands + (k + 1) * DIM,
            cands + (k + 2) * DIM,
            cands + (k + 3) * DIM,
            cands + (k + 4) * DIM,
            cands + (k + 5) * DIM,
            cands + (k + 6) * DIM,
            cands + (k + 7) * DIM,
        )
        dst[k + 0] = dots[0]
        dst[k + 1] = dots[1]
        dst[k + 2] = dots[2]
        dst[k + 3] = dots[3]
        dst[k + 4] = dots[4]
        dst[k + 5] = dots[5]
        dst[k + 6] = dots[6]
        dst[k + 7] = dots[7]
        k += 8


def bench_batched_for(M: Int, K: Int):
    """Batched M queries × K candidates, all FP32, one sgemm call.

    NEON baseline runs M independent sgemv-shaped 1×K dot-product passes.
    """
    print("─── M=" + String(M) + " × K=" + String(K) + " ──────────────────────────────")

    var qs_int8 = alloc[Int8](M * DIM)
    var qs_fp32 = alloc[Float32](M * DIM)
    var cands_int8 = alloc[Int8](K * DIM)
    var cands_fp32 = alloc[Float32](K * DIM)
    var out_neon = alloc[Float32](M * K)
    var out_acc_e = alloc[Float32](M * K)

    fill_int8(qs_int8, M * DIM, 0xC0FFEE)
    fill_int8(cands_int8, K * DIM, 0xDEADBEEF)
    var scale: Float32 = 1.0 / 127.0
    fill_fp32_from_int8(qs_int8, qs_fp32, M * DIM, scale)
    fill_fp32_from_int8(cands_int8, cands_fp32, K * DIM, scale)

    # NEON: M passes of (1 query × K cands)
    for q in range(M):
        run_neon_batch8(qs_int8 + q * DIM, cands_int8, K, out_neon + q * K)
    cblas_sgemm_M_K(M, K, qs_fp32, cands_fp32, out_acc_e)

    var iters = ITERS // (1 if M <= 16 else (M // 16))   # cap wall-time per case
    if iters < 200: iters = 200

    var t0 = perf_counter_ns()
    for _ in range(iters):
        for q in range(M):
            run_neon_batch8(qs_int8 + q * DIM, cands_int8, K, out_neon + q * K)
    var t1 = perf_counter_ns()
    var neon_ns = Float64(t1 - t0) / Float64(iters)

    var t2 = perf_counter_ns()
    for _ in range(iters):
        cblas_sgemm_M_K(M, K, qs_fp32, cands_fp32, out_acc_e)
    var t3 = perf_counter_ns()
    var acc_e_ns = Float64(t3 - t2) / Float64(iters)

    var macs: Float64 = Float64(M) * Float64(K) * Float64(DIM)
    var ratio_e = neon_ns / acc_e_ns
    print(
        "  NEON loop                : "
        + String(Int(neon_ns)) + " ns/call   "
        + String(macs / neon_ns) + " GMAC/s   (1.00×)"
    )
    print(
        "  Accelerate sgemm batched : "
        + String(Int(acc_e_ns)) + " ns/call   "
        + String(macs / acc_e_ns) + " GMAC/s   ("
        + String(ratio_e) + "×)"
    )

    qs_int8.free()
    qs_fp32.free()
    cands_int8.free()
    cands_fp32.free()
    out_neon.free()
    out_acc_e.free()


def bench_for_K(K: Int):
    print("─── K=" + String(K) + " ──────────────────────────────")

    # --- Allocate ---------------------------------------------------------
    var q_int8 = alloc[Int8](DIM)
    var q_fp32 = alloc[Float32](DIM)
    var cands_int8 = alloc[Int8](K * DIM)
    var cands_fp32 = alloc[Float32](K * DIM)
    var out_neon = alloc[Float32](K)
    var out_acc_b = alloc[Float32](K)
    var out_acc_c = alloc[Float32](K)
    var out_acc_d = alloc[Float32](K)

    # --- Fill ---------------------------------------------------------------
    fill_int8(q_int8, DIM, 0xC0FFEE)
    fill_int8(cands_int8, K * DIM, 0xDEADBEEF)
    var scale: Float32 = 1.0 / 127.0
    fill_fp32_from_int8(q_int8, q_fp32, DIM, scale)
    fill_fp32_from_int8(cands_int8, cands_fp32, K * DIM, scale)

    # Warm-up — fault all pages into TLB and warm the i-cache
    run_neon_batch8(q_int8, cands_int8, K, out_neon)
    cblas_sgemv_k_dim(K, cands_fp32, q_fp32, out_acc_b)

    # --- NEON timing --------------------------------------------------------
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        run_neon_batch8(q_int8, cands_int8, K, out_neon)
    var t1 = perf_counter_ns()
    var neon_ns = Float64(t1 - t0) / Float64(ITERS)

    # --- Accelerate Mode B (candidates pre-dequantized) ---------------------
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        cblas_sgemv_k_dim(K, cands_fp32, q_fp32, out_acc_b)
    var t3 = perf_counter_ns()
    var acc_b_ns = Float64(t3 - t2) / Float64(ITERS)

    # --- Accelerate Mode C (dequantize-on-the-fly) --------------------------
    var t4 = perf_counter_ns()
    for _ in range(ITERS):
        fill_fp32_from_int8(cands_int8, cands_fp32, K * DIM, scale)
        cblas_sgemv_k_dim(K, cands_fp32, q_fp32, out_acc_c)
    var t5 = perf_counter_ns()
    var acc_c_ns = Float64(t5 - t4) / Float64(ITERS)

    # --- Accelerate Mode D (sgemm with transpose_b, candidates pre-dequant) -
    var t6 = perf_counter_ns()
    for _ in range(ITERS):
        cblas_sgemm_q_K(K, cands_fp32, q_fp32, out_acc_d)
    var t7 = perf_counter_ns()
    var acc_d_ns = Float64(t7 - t6) / Float64(ITERS)

    # --- Summary ------------------------------------------------------------
    var macs: Float64 = Float64(K) * Float64(DIM)
    var neon_gmac = macs / neon_ns                    # ns × 1e-9 → s; macs/s × 1e-9 → GMAC/s
    var acc_b_gmac = macs / acc_b_ns
    var acc_c_gmac = macs / acc_c_ns
    var acc_d_gmac = macs / acc_d_ns
    var ratio_b = neon_ns / acc_b_ns
    var ratio_c = neon_ns / acc_c_ns
    var ratio_d = neon_ns / acc_d_ns

    print(
        "  NEON batch-8                 : "
        + String(Int(neon_ns)) + " ns/call   "
        + String(neon_gmac) + " GMAC/s   (1.00×)"
    )
    print(
        "  Accelerate sgemv (cached B)  : "
        + String(Int(acc_b_ns)) + " ns/call   "
        + String(acc_b_gmac) + " GMAC/s   ("
        + String(ratio_b) + "×)"
    )
    print(
        "  Accelerate sgemm (cached D)  : "
        + String(Int(acc_d_ns)) + " ns/call   "
        + String(acc_d_gmac) + " GMAC/s   ("
        + String(ratio_d) + "×)"
    )
    print(
        "  Accelerate sgemv (dequant C) : "
        + String(Int(acc_c_ns)) + " ns/call   "
        + String(acc_c_gmac) + " GMAC/s   ("
        + String(ratio_c) + "×)"
    )

    # --- Sanity: spot-check Mode B vs NEON on first few outputs ------------
    # NEON returns INT8×INT8 dot in Float32; Accelerate returns FP32×FP32 dot
    # over the same numbers scaled by `scale`. Expected: out_acc_b[i] ≈ out_neon[i] * scale * scale.
    var s2: Float32 = scale * scale
    var max_err: Float32 = 0.0
    for i in range(min(K, 8)):
        var expected = out_neon[i] * s2
        var got = out_acc_b[i]
        var err: Float32 = expected - got
        if err < 0: err = -err
        var rel: Float32 = err / (expected if expected > 0 else 1.0)
        if rel > max_err: max_err = rel
    print("  rel-err vs NEON (first 8) : " + String(max_err))

    q_int8.free()
    q_fp32.free()
    cands_int8.free()
    cands_fp32.free()
    out_neon.free()
    out_acc_b.free()
    out_acc_c.free()
    out_acc_d.free()


def main():
    print("=== Pion AMX/Accelerate decision-gate microbench (issue #47) ===")
    print("dim=" + String(DIM) + ", iters=" + String(ITERS))
    print("Decision gate: Mode B must be ≥ 3.0× NEON at K=64")
    print("")

    bench_for_K(32)
    bench_for_K(64)
    bench_for_K(128)
    bench_for_K(256)
    bench_for_K(1024)

    print("")
    print("=== Mode E: batched M×K GEMM (compact_overflows reformulation) ===")
    print("Measures the AMX-friendly path: many queries vs K candidates in one sgemm.")
    print("")
    bench_batched_for(16, 32)
    bench_batched_for(16, 64)
    bench_batched_for(32, 64)
    bench_batched_for(16, 256)
    bench_batched_for(64, 64)
    bench_batched_for(64, 256)
