from std.math import sqrt
from src.vector.fma_mad import fma_mad
from std.memory.unsafe_pointer import UnsafePointer
from std.sys import simd_width_of
from std.sys import CompilationTarget
from std.sys.intrinsics import prefetch, llvm_intrinsic
from std.memory.unsafe import bitcast
from std.collections import Array
from src.vector.reg_tile import RegTile, pack_rows, ptr_rows4, ptr_rows8

# ── Platform-intrinsic rounding (inspired by MAX quantization/_utils.mojo) ────
# Fused float→int32 with round-to-nearest-even: single instruction on ARM NEON
# (fcvtns) instead of separate frintn + fcvtzs. Fallback to round().cast[] elsewhere.

@always_inline
def _fcvtns_4(val: SIMD[DType.float32, 4]) -> SIMD[DType.int32, 4]:
    """ARM NEON fcvtns: fused round-to-nearest-even + convert to signed int32.
    Single instruction on Apple M-series. Only available for width=4."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.fcvtns.v4i32.v4f32",
        SIMD[DType.int32, 4],
        SIMD[DType.float32, 4]
    ](val)

@always_inline
def roundeven_to_int32[width: Int](val: SIMD[DType.float32, width]) -> SIMD[DType.int32, width]:
    """FP32→INT32 with round-to-nearest-even.
    - x86 AVX512: vcvtps2dq with embedded rounding control (single instruction per 16 floats)
    - macOS ARM NEON: fcvtns (single instruction per 4-lane chunk)
    - Fallback: round() + cast (2 instructions)"""
    comptime if CompilationTarget.has_avx512f() and width == 16:
        return rebind[SIMD[DType.int32, width]](llvm_intrinsic[
            "llvm.x86.avx512.mask.cvtps2dq.512",
            SIMD[DType.int32, 16],
            has_side_effect=False,
        ](
            rebind[SIMD[DType.float32, 16]](val),
            SIMD[DType.int32, 16](0),
            Int16(-1),   # no mask
            Int32(8),    # round to nearest even
        ))
    elif CompilationTarget.has_neon() and width == 4:
        # gh #122 (same landmine as sdot_int8 below): this arm was gated on
        # `not is_linux()`, so ARM Linux fell to round()+cast even though fcvtns
        # is base ARMv8 NEON, not an extension — and a macOS x86 build would
        # have taken the branch and emitted an aarch64 intrinsic. NEON is the
        # property actually required.
        return rebind[SIMD[DType.int32, width]](_fcvtns_4(rebind[SIMD[DType.float32, 4]](val)))
    else:
        return round(val).cast[DType.int32]()

@always_inline
def quantize_fp32_to_int8_simd(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
    qmin: Float32,
    scale: Float32):
    """Vectorized FP32→INT8 quantization: dst[i] = clamp(round((src[i]-qmin)*scale), 0,254) - 127.
    Uses fused fcvtns (round-to-nearest + convert) on ARM NEON — single instruction per SIMD lane.
    2×-unrolled SIMD: processes 2×fp_width per iteration (8 floats on M4 NEON)."""
    comptime fp_width = simd_width_of[DType.float32]()
    comptime stride = fp_width * 2
    var scale_v = SIMD[DType.float32, fp_width](scale)
    var qmin_v  = SIMD[DType.float32, fp_width](qmin)
    var lo_v    = SIMD[DType.int32,   fp_width](0)
    var hi_v    = SIMD[DType.int32,   fp_width](254)
    var off_v   = SIMD[DType.int32,   fp_width](127)
    var qi = 0
    # 2×-unrolled main loop: two independent clamp chains per iteration
    while qi + stride <= dim:
        var norm_a = (src.load[width=fp_width](qi) - qmin_v) * scale_v
        var norm_b = (src.load[width=fp_width](qi + fp_width) - qmin_v) * scale_v
        # Fused round-to-nearest + convert (fcvtns on NEON), then int32 clamp
        var rounded_a = roundeven_to_int32(norm_a)
        var rounded_b = roundeven_to_int32(norm_b)
        var clamped_a = min(max(rounded_a, lo_v), hi_v)
        var clamped_b = min(max(rounded_b, lo_v), hi_v)
        dst.store[width=fp_width](qi, (clamped_a - off_v).cast[DType.int8]())
        dst.store[width=fp_width](qi + fp_width, (clamped_b - off_v).cast[DType.int8]())
        qi += stride
    # Single-width remainder
    while qi + fp_width <= dim:
        var norm = (src.load[width=fp_width](qi) - qmin_v) * scale_v
        var rounded = roundeven_to_int32(norm)
        var clamped = min(max(rounded, lo_v), hi_v)
        dst.store[width=fp_width](qi, (clamped - off_v).cast[DType.int8]())
        qi += fp_width
    # Scalar tail
    while qi < dim:
        var norm = (src[qi] - qmin) * scale
        var r = Int(norm + 0.5) if norm >= 0.0 else Int(norm - 0.5)
        if r > 254: r = 254
        if r < 0:   r = 0
        dst[qi] = Int8(r - 127)
        qi += 1

# ── Welford online statistics (inspired by MAX nn/normalization.mojo) ──────────
# Single-pass mean + variance computation for FP32 vector buffers.
# Used for sigma-based calibration range clipping in FP32→INT8 quantization.

@no_inline
def welford_calibrate(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    count: Int, dim: Int,
    sigma: Float32 = 3.0,
) -> Tuple[Float32, Float32]:
    """Compute sigma-clipped [mean - sigma*std, mean + sigma*std] range over all values.
    Single pass via Welford online algorithm. Returns (qmin, qmax).
    count = number of vectors, dim = dimensions per vector. Total values = count * dim."""
    var total = count * dim
    if total == 0:
        return (-0.20, 0.20)
    # SIMD-accelerated Welford: process fp_width elements per step
    comptime fp_width = simd_width_of[DType.float32]()
    var mean_v = SIMD[DType.float32, fp_width](0.0)
    var m2_v = SIMD[DType.float32, fp_width](0.0)
    var count_v = SIMD[DType.float32, fp_width](0.0)
    var one_v = SIMD[DType.float32, fp_width](1.0)

    var i = 0
    while i + fp_width <= total:
        var val = data.load[width=fp_width](i)
        count_v += one_v
        var d1 = val - mean_v
        mean_v += d1 / count_v
        var d2 = val - mean_v
        m2_v = fma_mad[fp_width](d1, d2, m2_v)
        i += fp_width

    # Reduce SIMD lanes using Welford combine
    var mean: Float32 = 0.0
    var m2: Float32 = 0.0
    var n: Float32 = 0.0
    for lane in range(fp_width):
        var n_b = count_v[lane]
        if n_b == 0: continue
        var mean_b = mean_v[lane]
        var m2_b = m2_v[lane]
        var n_total = n + n_b
        var delta = mean_b - mean
        mean += delta * n_b / n_total
        m2 += m2_b + delta * delta * n * n_b / n_total
        n = n_total

    # Scalar tail
    while i < total:
        n += 1.0
        var d1 = data[i] - mean
        mean += d1 / n
        var d2 = data[i] - mean
        m2 += d1 * d2
        i += 1

    var variance = m2 / n if n > 0 else Float32(0.0)
    # gh #376: a real sqrt. This was three Newton steps seeded at x = variance,
    # which converges only near variance 1: OpenAI embeddings (variance
    # 0.00065, sigma 0.0255) came out sigma 0.127, so every "3 sigma" range was
    # ~15 sigma wide (±0.38) and wasted most of the INT8 codes.
    var std = sqrt(variance) if variance > 0 else Float32(0.0)

    var qmin = mean - sigma * std
    var qmax = mean + sigma * std
    # Ensure non-degenerate range
    if qmax - qmin < 1e-6:
        qmin = mean - 0.20
        qmax = mean + 0.20
    return (qmin, qmax)

@no_inline
def calibrate_per_group(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    count: Int, dim: Int,
    group_size: Int,
    group_qmins: UnsafePointer[Float32, MutUntrackedOrigin],
    group_scales: UnsafePointer[Float32, MutUntrackedOrigin],
    sigma: Float32 = 3.0):
    """Compute per-group (Q8_K-style) calibration from buffer of FP32 vectors.
    For each group of group_size dimensions, computes Welford mean±sigma clipping
    independently. Result: group_qmins[g] and group_scales[g] = 254 / (qmax-qmin).
    data: [count × dim] FP32 vectors (row-major)."""
    var num_groups = dim // group_size
    for g in range(num_groups):
        var goff = g * group_size
        # Welford per-group: iterate over all vectors, collecting stats for dims [goff, goff+group_size)
        var mean: Float32 = 0.0
        var m2: Float32 = 0.0
        var n: Float32 = 0.0
        for vi in range(count):
            for di in range(group_size):
                n += 1.0
                var val = data[vi * dim + goff + di]
                var d1 = val - mean
                mean += d1 / n
                var d2 = val - mean
                m2 += d1 * d2
        var variance = m2 / n if n > 0 else Float32(0.0)
        # gh #376: real sqrt — see welford_calibrate
        var std = sqrt(variance) if variance > 0 else Float32(0.0)
        var gmin = mean - sigma * std
        var gmax = mean + sigma * std
        if gmax - gmin < 1e-6:
            gmin = mean - 0.20
            gmax = mean + 0.20
        group_qmins[g] = gmin
        var range_val = gmax - gmin
        group_scales[g] = Float32(254.0) / range_val if range_val > 0 else Float32(1.0)

# ── x86 INT8 dot product intrinsics ──
# Signed INT8×INT8 → INT32 dot product. On VNNI (after MAX kernels'
# vnni_intrinsics.mojo): fused vpdpbusd over a biased (unsigned) first operand,
# minus a correction term. Without VNNI: both operands sign-extended to int16,
# then pmaddwd + phaddd.

@always_inline
def _x86_sdot_int8_vnni(acc: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.int32, 4]:
    """128-bit AVX512-VNNI vpdpbusd: unsigned×signed byte dot product with accumulate.
    Bias a to unsigned (+128), compute vpdpbusd, subtract correction term.
    VNNI fuses 4 multiply-adds per int32 lane in a single µop."""
    # Bias a to unsigned: XOR with 0x80 flips sign bit (signed → offset binary)
    var a_u8 = bitcast[DType.uint8, 16](a ^ SIMD[DType.int8, 16](-128))
    # vpdpbusd: acc += a_u8[4i]*b[4i] + a_u8[4i+1]*b[4i+1] + ...
    var result = llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.128", SIMD[DType.int32, 4]
    ](acc, a_u8, bitcast[DType.uint8, 16](b))
    # Subtract bias correction: 128 * sum(b[4i..4i+3]) per lane.
    # Use vpdpbusd(0, ones, b) to compute group sums efficiently.
    var bsum = llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.128", SIMD[DType.int32, 4]
    ](SIMD[DType.int32, 4](0), SIMD[DType.uint8, 16](1), bitcast[DType.uint8, 16](b))
    return result - bsum * Int32(128)

@always_inline
def _x86_sdot_int8_vnni_512(acc: SIMD[DType.int32, 16], a: SIMD[DType.int8, 64], b: SIMD[DType.int8, 64]) -> SIMD[DType.int32, 16]:
    """512-bit AVX512-VNNI vpdpbusd: processes 64 bytes per instruction (4× throughput).
    16 int32 lanes × 4 bytes each = 64 byte dot product with accumulate.
    EPYC Zen4 / Intel SPR execute this in a single µop on ZMM registers."""
    var a_u8 = bitcast[DType.uint8, 64](a ^ SIMD[DType.int8, 64](-128))
    var result = llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512", SIMD[DType.int32, 16]
    ](acc, a_u8, bitcast[DType.uint8, 64](b))
    var bsum = llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512", SIMD[DType.int32, 16]
    ](SIMD[DType.int32, 16](0), SIMD[DType.uint8, 64](1), bitcast[DType.uint8, 64](b))
    return result - bsum * Int32(128)

@always_inline
def _x86_sdot_int8_sse(acc: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.int32, 4]:
    """Exact signed INT8 dot for x86 without VNNI: acc[i] += sum(a[4i+k]*b[4i+k]).

    Both operands are sign-extended to int16, `pmaddwd` multiplies and sums
    adjacent pairs into int32 (|a*b| <= 16384, so a pair never saturates), and
    `phaddd` folds the pairs into the 4-byte groups the lane contract names.

    The previous body was a pmaddubsw "mask decomposition" lifted from a kernel
    whose first operand is UNSIGNED. pmaddubsw treats `a` as u8, so every
    negative query byte was multiplied as a+256: 99.8% of random signed 16-byte
    dots came out wrong, and the INT8 beam search on an x86-64-v2 build returned
    recall@10 0.001 where the same routine on ARM SDOT returned 0.134 (a C-ABI
    harness driving the beam on both ISAs). It went unnoticed because the Linux boxes that benchmarked
    vector search either had VNNI or ran the CUDA build, whose FP32 rerank hides
    wrong INT8 distances — and the GPU-free x86-64-v2 build is exactly what
    release.yml ships for Linux x86_64."""
    var a16 = a.cast[DType.int16]()
    var b16 = b.cast[DType.int16]()
    var lo = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        a16.slice[8, offset=0](), b16.slice[8, offset=0]())
    var hi = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        a16.slice[8, offset=8](), b16.slice[8, offset=8]())
    return acc + llvm_intrinsic["llvm.x86.ssse3.phadd.d.128", SIMD[DType.int32, 4]](lo, hi)

@always_inline
def sdot_int8(acc: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.int32, 4]:
    """ISA-dispatched INT8 dot product: acc[i] += sum(a[4i+k]*b[4i+k], k=0..3).
    - x86 VNNI: AVX512-VNNI vpdpbusd (single µop, fused 4-byte dot)
    - x86 without VNNI: sign-extend to int16, pmaddwd + phaddd (exact; see
      `_x86_sdot_int8_sse` for the unsigned-by-signed bug it replaced)
    - ARM +dotprod (ARMv8.2): NEON SDOT (single instruction)
    - anything else: portable int32 widening fallback (SMULL/SMLAL)

    gh #122: the ARM arm used to be selected by `CompilationTarget.is_linux()`,
    which sent **every** ARM Linux target to the widening fallback — 4–8× slower
    than SDOT — even though Graviton3/4, Ampere and NVIDIA Grace all ship
    +dotprod. The OS was never the property being tested; the ISA is. Gating on
    `has_neon_int8_dotprod()` keeps macOS ARM on exactly the same instruction it
    used before and lets ARM Linux take it too, while a genuinely
    dotprod-less target still lands on the portable path.

    Note this reads the *target* ISA, so it follows `--target-cpu`. Builds that
    pin a baseline (the CI x86-64-v2 pin, gh #83) get the baseline's answer,
    which is the correct conservative behavior.
    """
    comptime if CompilationTarget.is_x86():
        comptime if CompilationTarget.has_vnni():
            return _x86_sdot_int8_vnni(acc, a, b)
        else:
            return _x86_sdot_int8_sse(acc, a, b)
    elif CompilationTarget.has_neon_int8_dotprod():
        # ARMv8.2+ with +dotprod — macOS ARM and ARM Linux alike.
        return llvm_intrinsic[
            "llvm.aarch64.neon.sdot.v4i32.v16i8",
            SIMD[DType.int32, 4],
            SIMD[DType.int32, 4],
            SIMD[DType.int8, 16],
            SIMD[DType.int8, 16]
        ](acc, a, b)
    else:
        # No fused int8 dot on this target: portable int32 widening fallback.
        # LLVM maps this to SMULL/SMULL2/SMLAL on NEON.
        var prod = a.cast[DType.int32]() * b.cast[DType.int32]()
        return acc + SIMD[DType.int32, 4](
            prod[0] + prod[1] + prod[2] + prod[3],
            prod[4] + prod[5] + prod[6] + prod[7],
            prod[8] + prod[9] + prod[10] + prod[11],
            prod[12] + prod[13] + prod[14] + prod[15]
        )

@always_inline
def l2_normalize_fp32(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """Scale `src` to unit L2 length into `dst` (gh #271, DISTANCE_METRIC COSINE).

    Cosine ordering equals L2 ordering on unit vectors, and Pion's search is
    squared L2 over affinely-quantized codes (`q_norm + n_norm - 2*dot`). So
    honouring COSINE is exactly: normalize before quantizing, at ingest AND at
    query. Doing it here rather than in the distance function is not a style
    choice — the quantizer is affine WITH AN OFFSET, and an offset cancels in a
    difference but not in a dot product, so a cosine computed from quantized
    dots would be wrong.

    A zero vector has no direction; it is copied through unchanged rather than
    turned into NaNs. `src` and `dst` may alias.
    """
    comptime width = simd_width_of[DType.float32]()
    var acc = SIMD[DType.float32, width](0.0)
    var i = 0
    while i + width <= dim:
        var v = src.load[width=width](i)
        acc += v * v
        i += width
    var norm_sq = acc.reduce_add()
    while i < dim:
        norm_sq += src[i] * src[i]
        i += 1
    if norm_sq <= 0.0:
        if dst != src:
            for j in range(dim): dst[j] = src[j]
        return
    var inv = Float32(1.0) / sqrt(norm_sq)
    var inv_v = SIMD[DType.float32, width](inv)
    var k = 0
    while k + width <= dim:
        dst.store[width=width](k, src.load[width=width](k) * inv_v)
        k += width
    while k < dim:
        dst[k] = src[k] * inv
        k += 1


@no_inline
def norm_sq_int8_jit[dim: Int](v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """INT8 L2 norm squared via SDOT: sum(v[i]^2). Computed once per query at search time.
    dim must be a multiple of 64 (1536 qualifies)."""
    comptime if CompilationTarget.is_x86() and CompilationTarget.has_vnni():
        var acc = SIMD[DType.int32, 16](0)
        for i in range(0, dim, 64):
            var vv = v.load[width=64](i)
            acc = _x86_sdot_int8_vnni_512(acc, vv, vv)
        return acc.reduce_add().cast[DType.float32]()
    else:
        var acc = SIMD[DType.int32, 4](0)
        for i in range(0, dim, 16):
            var vv = v.load[width=16](i)
            acc = sdot_int8(acc, vv, vv)
        return acc.reduce_add().cast[DType.float32]()

# dot_int8_int8_batch8_jit: tuned batch kernel, closed in libpion_vector (D11).

# dot_int8_int8_batch4_jit: tuned batch kernel, closed in libpion_vector (D11).

@always_inline
def manual_popcount(val: UInt64) -> Int:
    """Hardware popcount via LLVM ctpop intrinsic. Single CNT instruction on M4 NEON.
    Replaces 12-op bit-twiddling sequence with 1-cycle hardware instruction."""
    return Int(llvm_intrinsic["llvm.ctpop.i64", UInt64](val))

@always_inline
def l2_distance_int8_128(v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    comptime width = 16
    # Two independent accumulator chains (sum0/sum1) to hide INT32 multiply-add latency.
    # sum0 accumulates even iterations, sum1 accumulates odd iterations.
    # The CPU can issue on sum1 while sum0's result is still in flight.
    var sum0 = SIMD[DType.int32, width](0)
    var sum1 = SIMD[DType.int32, width](0)

    var d0 = v1.load[width=width](0).cast[DType.int32]() - v2.load[width=width](0).cast[DType.int32]()
    sum0 += d0 * d0
    var d1 = v1.load[width=width](16).cast[DType.int32]() - v2.load[width=width](16).cast[DType.int32]()
    sum1 += d1 * d1
    var d2 = v1.load[width=width](32).cast[DType.int32]() - v2.load[width=width](32).cast[DType.int32]()
    sum0 += d2 * d2
    var d3 = v1.load[width=width](48).cast[DType.int32]() - v2.load[width=width](48).cast[DType.int32]()
    sum1 += d3 * d3
    var d4 = v1.load[width=width](64).cast[DType.int32]() - v2.load[width=width](64).cast[DType.int32]()
    sum0 += d4 * d4
    var d5 = v1.load[width=width](80).cast[DType.int32]() - v2.load[width=width](80).cast[DType.int32]()
    sum1 += d5 * d5
    var d6 = v1.load[width=width](96).cast[DType.int32]() - v2.load[width=width](96).cast[DType.int32]()
    sum0 += d6 * d6
    var d7 = v1.load[width=width](112).cast[DType.int32]() - v2.load[width=width](112).cast[DType.int32]()
    sum1 += d7 * d7

    return (sum0 + sum1).reduce_add().cast[DType.float32]()

@always_inline
def l2_distance_int8(v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int) -> Float32:
    comptime width = simd_width_of[DType.int8]()
    var sum = SIMD[DType.int32, width](0)

    for i in range(0, dim - width + 1, width):
        var d = v1.load[width=width](i).cast[DType.int32]() - v2.load[width=width](i).cast[DType.int32]()
        sum += d * d

    # Scalar tail (not hit for dim=1536 which is always SIMD-aligned)
    var result = sum.reduce_add().cast[DType.float32]()
    for i in range(dim - (dim % width), dim):
        var d = v1[i].cast[DType.int32]() - v2[i].cast[DType.int32]()
        result += (d * d).cast[DType.float32]()

    return result

@always_inline
def l2_distance_fp32_int8(v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int, min_val: Float32, range_val: Float32) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    # Pre-calculate fused dequant constants (same as JIT variant)
    var scale = range_val / 254.0
    var scale_v = SIMD[DType.float32, width](scale)
    var offset_v = SIMD[DType.float32, width](127.0 * scale + min_val)
    # Dual FMA chains to hide latency
    var sum_a = SIMD[DType.float32, width](0.0)
    var sum_b = SIMD[DType.float32, width](0.0)
    var i = 0
    while i + width * 2 <= dim:
        var d_a = v1.load[width=width](i) - fma_mad[width](v2.load[width=width](i).cast[DType.float32](), scale_v, offset_v)
        sum_a = fma_mad[width](d_a, d_a, sum_a)
        var d_b = v1.load[width=width](i + width) - fma_mad[width](v2.load[width=width](i + width).cast[DType.float32](), scale_v, offset_v)
        sum_b = fma_mad[width](d_b, d_b, sum_b)
        i += width * 2
    while i + width <= dim:
        var d = v1.load[width=width](i) - fma_mad[width](v2.load[width=width](i).cast[DType.float32](), scale_v, offset_v)
        sum_a = fma_mad[width](d, d, sum_a)
        i += width

    var result = (sum_a + sum_b).reduce_add()
    for j in range(i, dim):
        var f2 = fma_mad[1](v2[j].cast[DType.float32](), scale, 127.0 * scale + min_val)
        var d = v1[j] - f2
        result += d * d

    return result

@always_inline
def l2_distance_int8_jit[dim: Int](v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    comptime width = simd_width_of[DType.int8]()
    # Two independent accumulator chains to hide INT32 multiply-add latency (3-4 cycles on M4).
    # sum_a and sum_b are updated in alternating iterations; neither depends on the other,
    # so the CPU can issue sum_b's operation while sum_a's result is still in flight.
    var sum_a = SIMD[DType.int32, width](0)
    var sum_b = SIMD[DType.int32, width](0)

    for i in range(0, dim - 2 * width + 1, 2 * width):
        var d_a = v1.load[width=width](i).cast[DType.int32]() - v2.load[width=width](i).cast[DType.int32]()
        sum_a += d_a * d_a
        var d_b = v1.load[width=width](i + width).cast[DType.int32]() - v2.load[width=width](i + width).cast[DType.int32]()
        sum_b += d_b * d_b

    var total_sum = (sum_a + sum_b).reduce_add()

    # Handle last width-block if dim/width is odd (at most one remaining)
    comptime simd_tail = ((dim // width) // 2) * 2 * width
    comptime simd_end = (dim // width) * width
    for i in range(simd_tail, simd_end, width):
        var d = v1.load[width=width](i).cast[DType.int32]() - v2.load[width=width](i).cast[DType.int32]()
        total_sum += (d * d).reduce_add()

    # Scalar tail if dim is not a multiple of width
    for i in range(simd_end, dim):
        var d = v1[i].cast[DType.int32]() - v2[i].cast[DType.int32]()
        total_sum += d * d

    return total_sum.cast[DType.float32]()

@always_inline
def _sabd_u8(a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]) -> SIMD[DType.uint8, 16]:
    """NEON SABD: |a - b| per lane. For int8 inputs it is at most 255, so the
    unsigned reading of the result is exact."""
    return bitcast[DType.uint8, 16](llvm_intrinsic[
        "llvm.aarch64.neon.sabd.v16i8", SIMD[DType.int8, 16], has_side_effect=False](a, b))


@always_inline
def _udot_u8(acc: SIMD[DType.uint32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.uint8, 16]) -> SIMD[DType.uint32, 4]:
    """NEON UDOT (+dotprod): acc[i] += sum of a[4i+k] * b[4i+k], unsigned."""
    return llvm_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.uint32, 4],
                          has_side_effect=False](acc, a, b)


@always_inline
def l2_int8_sabd_udot[dim: Int](a: UnsafePointer[Int8, MutUntrackedOrigin],
                                b: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """gh #395: squared L2 between two INT8 code vectors, exactly — the same
    integer `l2_distance_int8_jit` returns. |a-b| fits a u8 lane (<= 255), so
    UDOT(d, d) accumulates (a-b)^2 with no widening, in four independent
    chains. The widening kernel spends two multiply-adds and two widenings per
    8 lanes on the same sum. `dim` must be a multiple of 64. Targets without
    +dotprod use `l2_distance_int8_jit`."""
    comptime if CompilationTarget.has_neon_int8_dotprod() and dim % 64 == 0:
        var s0 = SIMD[DType.uint32, 4](0); var s1 = SIMD[DType.uint32, 4](0)
        var s2 = SIMD[DType.uint32, 4](0); var s3 = SIMD[DType.uint32, 4](0)
        comptime for i in range(0, dim, 64):
            var d0 = _sabd_u8(a.load[width=16](i), b.load[width=16](i)); s0 = _udot_u8(s0, d0, d0)
            var d1 = _sabd_u8(a.load[width=16](i + 16), b.load[width=16](i + 16)); s1 = _udot_u8(s1, d1, d1)
            var d2 = _sabd_u8(a.load[width=16](i + 32), b.load[width=16](i + 32)); s2 = _udot_u8(s2, d2, d2)
            var d3 = _sabd_u8(a.load[width=16](i + 48), b.load[width=16](i + 48)); s3 = _udot_u8(s3, d3, d3)
        return ((s0 + s1) + (s2 + s3)).reduce_add().cast[DType.float32]()
    else:
        return l2_distance_int8_jit[dim](a, b)


@always_inline
def _l2_sabd_udot_rows[dim: Int, rows: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v: Array[UnsafePointer[Int8, MutUntrackedOrigin], rows],
) -> SIMD[DType.float32, rows]:
    """gh #127: the SABD+UDOT microkernel behind `l2_int8_sabd_udot_batch8`
    (NEON +dotprod only): one query load per 16 lanes, `rows` UINT32 chains
    in a RegTile. `dim` must be a multiple of 16."""
    var acc = RegTile[DType.uint32, rows, 4]()
    for i in range(0, dim, 16):
        var qv = q.load[width=16](i)
        comptime for r in range(rows):
            var d = _sabd_u8(qv, v[r].load[width=16](i))
            acc[r] = _udot_u8(acc[r], d, d)
    return pack_rows[rows](acc.reduce_f32())

@always_inline
def l2_int8_sabd_udot_batch8[dim: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v0: UnsafePointer[Int8, MutUntrackedOrigin], v1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin], v3: UnsafePointer[Int8, MutUntrackedOrigin],
    v4: UnsafePointer[Int8, MutUntrackedOrigin], v5: UnsafePointer[Int8, MutUntrackedOrigin],
    v6: UnsafePointer[Int8, MutUntrackedOrigin], v7: UnsafePointer[Int8, MutUntrackedOrigin],
) -> SIMD[DType.float32, 8]:
    """gh #395: `l2_int8_sabd_udot` against eight vectors at once — one query
    load per 16 lanes and eight independent chains, so the eight vectors' cache
    misses overlap. Exact, like the single form."""
    comptime if CompilationTarget.has_neon_int8_dotprod() and dim % 16 == 0:
        return _l2_sabd_udot_rows[dim, 8](q, ptr_rows8(v0, v1, v2, v3, v4, v5, v6, v7))
    else:
        return l2_distance_int8_int8_batch8_jit[dim](q, v0, v1, v2, v3, v4, v5, v6, v7)

@always_inline
def l2_distance_gpu(v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int, min_val: Float32, range_val: Float32) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var scale = range_val / 254.0
    var scale_v = SIMD[DType.float32, width](scale)
    var offset_v = SIMD[DType.float32, width](127.0 * scale + min_val)
    var sum = SIMD[DType.float32, width](0.0)

    for i in range(0, dim - width + 1, width):
        var d = v1.load[width=width](i) - fma_mad[width](v2.load[width=width](i).cast[DType.float32](), scale_v, offset_v)
        sum = fma_mad[width](d, d, sum)

    var result = sum.reduce_add()
    for i in range(dim - (dim % width), dim):
        var f2 = fma_mad[1](v2[i].cast[DType.float32](), scale, 127.0 * scale + min_val)
        var d = v1[i] - f2
        result += d * d

    return result

@always_inline
def l2_distance_fp32_int8_jit[dim: Int](v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], min_val: Float32, range_val: Float32) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var sum = SIMD[DType.float32, width](0.0)
    
    for i in range(0, dim - width + 1, width):
        var f1 = v1.load[width=width](i)
        var q2 = v2.load[width=width](i).cast[DType.float32]()
        var f2 = (q2 + 127.0) / 254.0 * range_val + min_val
        var d = f1 - f2
        sum += d * d

    var total_sum = sum.reduce_add()

    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var f1 = v1[i]
        var q2 = v2[i].cast[DType.float32]()
        var f2 = (q2 + 127.0) / 254.0 * range_val + min_val
        var d = f1 - f2
        total_sum += d * d

    return total_sum

@always_inline
def dot_product_simd[simd_width: Int](
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int
) -> Float32:
    # Two independent FMA chains to hide FMA latency (4 cycles on M4)
    var sum_a = SIMD[DType.float32, simd_width](0.0)
    var sum_b = SIMD[DType.float32, simd_width](0.0)
    var i = 0

    while i + simd_width * 2 <= dim:
        sum_a = fma_mad[simd_width](a.load[width=simd_width](i), b.load[width=simd_width](i), sum_a)
        sum_b = fma_mad[simd_width](a.load[width=simd_width](i + simd_width), b.load[width=simd_width](i + simd_width), sum_b)
        i += simd_width * 2

    while i + simd_width <= dim:
        sum_a = fma_mad[simd_width](a.load[width=simd_width](i), b.load[width=simd_width](i), sum_a)
        i += simd_width

    var total = (sum_a + sum_b).reduce_add()
    # Scalar tail (not hit for dim=1536)
    while i < dim:
        total += a[i] * b[i]
        i += 1

    return total

@always_inline
def hamming_distance(v1: UnsafePointer[UInt64, MutUntrackedOrigin], v2: UnsafePointer[UInt64, MutUntrackedOrigin], dim_u64: Int) -> Float32:
    var total_popcount: Int = 0
    for i in range(dim_u64):
        total_popcount += manual_popcount(v1[i] ^ v2[i])
        
    return Float32(total_popcount)

@always_inline
def hamming_distance_jit[dim_u64: Int](v1: UnsafePointer[UInt64, MutUntrackedOrigin], v2: UnsafePointer[UInt64, MutUntrackedOrigin]) -> Float32:
    var total_popcount: Int = 0
    for i in range(dim_u64):
        total_popcount += manual_popcount(v1[i] ^ v2[i])
        
    return Float32(total_popcount)
    
@always_inline
def l2_distance_int4(v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int) -> Float32:
    # dim is original dimensions, each byte stores 2 dimensions
    var bytes = dim // 2
    comptime width = simd_width_of[DType.int8]()
    var sum = SIMD[DType.int32, width](0)
    
    for i in range(0, bytes - width + 1, width):
        var b1 = v1.load[width=width](i)
        var b2 = v2.load[width=width](i)
        
        # Low nibbles
        var l1 = (b1 & 0x0F).cast[DType.int32]()
        var l2 = (b2 & 0x0F).cast[DType.int32]()
        var dl = l1 - l2
        sum += dl * dl
        
        # High nibbles
        var h1 = (b1 >> 4).cast[DType.int32]()
        var h2 = (b2 >> 4).cast[DType.int32]()
        var dh = h1 - h2
        sum += dh * dh
        
    var total_sum = sum.reduce_add()
    
    # Tail
    for i in range(bytes - (bytes % width), bytes):
        var b1 = v1[i]
        var b2 = v2[i]
        var l1 = (b1 & 0x0F).cast[DType.int32]()
        var l2 = (b2 & 0x0F).cast[DType.int32]()
        var dl = l1 - l2
        total_sum += dl * dl
        var h1 = (b1 >> 4).cast[DType.int32]()
        var h2 = (b2 >> 4).cast[DType.int32]()
        var dh = h1 - h2
        total_sum += dh * dh
        
    return Float32(total_sum)

@always_inline
def l2_distance_int4_jit[dim: Int](v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    comptime bytes = dim // 2
    comptime width = simd_width_of[DType.int8]()
    var sum = SIMD[DType.int32, width](0)
    
    for i in range(0, bytes - width + 1, width):
        var b1 = v1.load[width=width](i)
        var b2 = v2.load[width=width](i)
        
        var l1 = (b1 & 0x0F).cast[DType.int32]()
        var l2 = (b2 & 0x0F).cast[DType.int32]()
        var dl = l1 - l2
        sum += dl * dl
        
        var h1 = (b1 >> 4).cast[DType.int32]()
        var h2 = (b2 >> 4).cast[DType.int32]()
        var dh = h1 - h2
        sum += dh * dh
        
    var total_sum = sum.reduce_add()
    
    # Tail
    comptime tail_start = (bytes // width) * width
    for i in range(tail_start, bytes):
        var b1 = v1[i]
        var b2 = v2[i]
        var l1 = (b1 & 0x0F).cast[DType.int32]()
        var l2 = (b2 & 0x0F).cast[DType.int32]()
        var dl = l1 - l2
        total_sum += dl * dl
        var h1 = (b1 >> 4).cast[DType.int32]()
        var h2 = (b2 >> 4).cast[DType.int32]()
        var dh = h1 - h2
        total_sum += dh * dh
        
    return Float32(total_sum)

@always_inline
def l2_distance_fp32_int4(v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], dim: Int, min_val: Float32, range_val: Float32) -> Float32:
    var total_sum: Float32 = 0.0
    for i in range(0, dim, 2):
        var b2 = v2[i // 2]
        var q2_low = (b2 & 0x0F).cast[DType.float32]()
        var q2_high = ((b2 >> 4) & 0x0F).cast[DType.float32]()
        
        var f2_low = q2_low / 15.0 * range_val + min_val
        var f2_high = q2_high / 15.0 * range_val + min_val
        
        var d_low = v1[i] - f2_low
        total_sum += d_low * d_low
        
        if i + 1 < dim:
            var d_high = v1[i + 1] - f2_high
            total_sum += d_high * d_high
    return total_sum

@always_inline
def l2_distance_fp32_int4_jit[dim: Int](v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], min_val: Float32, range_val: Float32) -> Float32:
    # INT4 compact buffer: byte b_i → lo nibble = element[2*b_i], hi nibble = element[2*b_i+1].
    # Optimized scalar JIT: compile-time dim eliminates i//2 and i+1<dim from scalar kernel.
    # The Mojo compiler auto-vectorizes this loop (no branch, stride-2 FP32 access pattern).
    comptime bytes = dim // 2
    var total_sum: Float32 = 0.0
    var scale = range_val / 15.0
    for b_i in range(bytes):
        var b = Int(v2[b_i])
        var q_lo = Float32(b & 0xF)
        var q_hi = Float32((b >> 4) & 0xF)
        var d0 = v1[2 * b_i]     - (q_lo * scale + min_val)
        var d1 = v1[2 * b_i + 1] - (q_hi * scale + min_val)
        total_sum += d0 * d0 + d1 * d1
    return total_sum

@always_inline
def l2_distance_fp32_int8_fused_jit[dim: Int](v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin], min_val: Float32, range_val: Float32) -> Float32:
    comptime width = simd_width_of[DType.float32]()
    var sum = SIMD[DType.float32, width](0.0)
    
    # Pre-calculate constants for dequantization
    # Quantizer: q = Int8(Int(norm * 254) - 127)  →  norm = (q + 127) / 254
    # f2 = norm * range_val + min_val = q * (range_val / 254) + (127 * range_val / 254 + min_val)
    var scale = range_val / 254.0
    var scale_simd = SIMD[DType.float32, width](scale)
    var offset_simd = SIMD[DType.float32, width](127.0 * scale + min_val)

    for i in range(0, dim - width + 1, width):
        var f1 = v1.load[width=width](i)
        var q2 = v2.load[width=width](i).cast[DType.float32]()
        
        # Dequantize and compute distance in two FMAs
        var f2_scaled = fma_mad[width](q2, scale_simd, offset_simd)
        var d = f1 - f2_scaled
        sum = fma_mad[width](d, d, sum) # sum += d * d

    var total_sum = sum.reduce_add()

    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var f1 = v1[i]
        var q2 = v2[i].cast[DType.float32]()
        var f2 = fma_mad[1](q2, scale, 127.0 * scale + min_val)
        var d = f1 - f2
        total_sum += d * d

    return total_sum

@always_inline
def l2_distance_fp32_int8_sq8_jit[dim: Int](
    v1: UnsafePointer[Float32, MutUntrackedOrigin],
    v2: UnsafePointer[Int8, MutUntrackedOrigin],
    sq8_scale: UnsafePointer[Float32, MutUntrackedOrigin],
    sq8_offset: UnsafePointer[Float32, MutUntrackedOrigin]) -> Float32:
    """SQ8 per-dimension: dequant = q * sq8_scale[d] + sq8_offset[d]. 7.9 effective bits."""
    comptime width = simd_width_of[DType.float32]()
    var sum = SIMD[DType.float32, width](0.0)
    for i in range(0, dim - width + 1, width):
        var dq = fma_mad[width](v2.load[width=width](i).cast[DType.float32](),
                     sq8_scale.load[width=width](i),
                     sq8_offset.load[width=width](i))
        var d = v1.load[width=width](i) - dq
        sum = fma_mad[width](d, d, sum)
    var total = sum.reduce_add()
    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var dq = v2[i].cast[DType.float32]() * sq8_scale[i] + sq8_offset[i]
        var d = v1[i] - dq
        total += d * d
    return total

@always_inline
def _l2_fp32_int8_pervec_rows[rows: Int](
    v1: UnsafePointer[Float32, MutUntrackedOrigin],
    v: Array[UnsafePointer[Int8, MutUntrackedOrigin], rows],
    dim: Int,
    mins: Array[Float32, rows],
    ranges: Array[Float32, rows],
) -> SIMD[DType.float32, rows]:
    """gh #127: the per-vector SQ8 microkernel behind
    `l2_distance_fp32_int8_pervec_batch4`. FP32, so each row runs exactly the
    expression tree the unrolled form ran (dequantize with fma_mad, subtract,
    fma_mad into the row's chain; the same scalar tail); only the rows'
    interleaving differs, and rows never touch each other."""
    comptime width = simd_width_of[DType.float32]()
    var s = Array[Float32, rows](uninitialized=True)
    var o = Array[Float32, rows](uninitialized=True)
    comptime for r in range(rows):
        s[r] = ranges[r] / 254.0
        o[r] = 127.0 * s[r] + mins[r]
    var acc = RegTile[DType.float32, rows, width]()
    var n_simd = (dim // width) * width
    var i = 0
    while i < n_simd:
        var f1 = v1.load[width=width](i)
        comptime for r in range(rows):
            var dq = fma_mad[width](v[r].load[width=width](i).cast[DType.float32](),
                                    SIMD[DType.float32, width](s[r]), SIMD[DType.float32, width](o[r]))
            var d = f1 - dq
            acc[r] = fma_mad[width](d, d, acc[r])
        i += width
    var red = acc.reduce_f32()
    while i < dim:
        var f1 = v1[i]
        comptime for r in range(rows):
            var d = f1 - (v[r][i].cast[DType.float32]() * s[r] + o[r])
            red[r] += d * d
        i += 1
    return pack_rows[rows](red)

@always_inline
def l2_distance_fp32_int8_pervec_batch4(
    v1:   UnsafePointer[Float32, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
    min0: Float32, range0: Float32,
    min1: Float32, range1: Float32,
    min2: Float32, range2: Float32,
    min3: Float32, range3: Float32,
) -> SIMD[DType.float32, 4]:
    """Per-vector SQ8 batch-4 (gh #42 follow-up). Each of the 4 neighbors
    has its own (min, range), unlike a shared-quantization batch. Loads the
    query chunk ONCE and reuses it for 4 dequant + L2² accumulators. Used by
    KNNHNSW._search_layer to attack memory stall from random graph traversal:
    4 neighbor base loads share 1 query load.
    Returns SIMD[float32, 4] = (dist0, dist1, dist2, dist3)."""
    var mins = Array[Float32, 4](uninitialized=True)
    mins[0] = min0; mins[1] = min1; mins[2] = min2; mins[3] = min3
    var ranges = Array[Float32, 4](uninitialized=True)
    ranges[0] = range0; ranges[1] = range1; ranges[2] = range2; ranges[3] = range3
    return _l2_fp32_int8_pervec_rows[4](
        v1, ptr_rows4(v2_0, v2_1, v2_2, v2_3), dim, mins, ranges)

@always_inline
def _l2_int8_int8_rows[dim: Int, rows: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v: Array[UnsafePointer[Int8, MutUntrackedOrigin], rows],
) -> SIMD[DType.float32, rows]:
    """gh #127: the INT8-INT8 L2 microkernel behind the batch-4 and batch-8
    entry points: one query load per 16 lanes, `rows` INT32 chains in a
    RegTile, Float32 tail per row. Exact, like the unrolled forms it replaces
    (INT32 sums are order-free; the per-row tail order is unchanged)."""
    comptime width = 16
    var acc = RegTile[DType.int32, rows, width]()
    for i in range(0, dim - width + 1, width):
        var qv = q.load[width=width](i).cast[DType.int32]()
        comptime for r in range(rows):
            var d = qv - v[r].load[width=width](i).cast[DType.int32]()
            acc[r] += d * d
    var red = acc.reduce_f32()
    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var qi = q[i].cast[DType.int32]()
        comptime for r in range(rows):
            var di = qi - v[r][i].cast[DType.int32]()
            red[r] += Float32(di * di)
    return pack_rows[rows](red)

@always_inline
def l2_distance_int8_int8_batch8_jit[dim: Int](
    q:    UnsafePointer[Int8, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_4: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_5: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_6: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_7: UnsafePointer[Int8, MutUntrackedOrigin]) -> SIMD[DType.float32, 8]:
    """INT8-INT8 L2 batch-8: no dequantization, matches graph build metric.
    Query quantized to INT8 once per search call; ~1.3x faster than FP32-INT8.
    Accumulates in INT32 (max: 1536 * 254^2 = 99M < INT32_MAX)."""
    return _l2_int8_int8_rows[dim, 8](
        q, ptr_rows8(v2_0, v2_1, v2_2, v2_3, v2_4, v2_5, v2_6, v2_7))

@always_inline
def cosine_distance_int8_jit[dim: Int](v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    comptime width = 16
    var sum = SIMD[DType.int32, width](0)
    for i in range(0, dim - width + 1, width):
        var v1_val = v1.load[width=width](i).cast[DType.int32]()
        var v2_val = v2.load[width=width](i).cast[DType.int32]()
        sum += v1_val * v2_val
    var result = -sum.reduce_add().cast[DType.float32]()
    comptime tail_start = (dim // width) * width
    for i in range(tail_start, dim):
        var v1i = v1[i].cast[DType.int32]()
        var v2i = v2[i].cast[DType.int32]()
        result -= Float32(v1i * v2i)
    return result

@always_inline
def l2_distance_int8_int8_batch4_jit[dim: Int](
    q:    UnsafePointer[Int8, MutUntrackedOrigin],
    v2_0: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_1: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_2: UnsafePointer[Int8, MutUntrackedOrigin],
    v2_3: UnsafePointer[Int8, MutUntrackedOrigin]) -> SIMD[DType.float32, 4]:
    """INT8-INT8 L2 batch-4: no dequantization, matches graph build metric."""
    return _l2_int8_int8_rows[dim, 4](q, ptr_rows4(v2_0, v2_1, v2_2, v2_3))

@always_inline
def pq_distance_8way[nsub: Int, pq_k: Int](
    pq_table: UnsafePointer[Float32, MutUntrackedOrigin],
    codes0: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes1: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes2: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes3: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes4: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes5: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes6: UnsafePointer[UInt8, MutUntrackedOrigin],
    codes7: UnsafePointer[UInt8, MutUntrackedOrigin]
) -> SIMD[DType.float32, 8]:
    var dists = SIMD[DType.float32, 8](0.0)
    for m in range(nsub):
        var table_m = pq_table + m * pq_k
        var indices = SIMD[DType.int32, 8](
            codes0[m].cast[DType.int32](), codes1[m].cast[DType.int32](),
            codes2[m].cast[DType.int32](), codes3[m].cast[DType.int32](),
            codes4[m].cast[DType.int32](), codes5[m].cast[DType.int32](),
            codes6[m].cast[DType.int32](), codes7[m].cast[DType.int32]()
        )
        dists += table_m.gather(indices)
    return dists

# l2_distance_int8_suffix_early_exit_jit: tuned batch kernel, closed in libpion_vector (D11).

@no_inline
def wht_fp32_inplace_512(data: UnsafePointer[Float32, MutUntrackedOrigin]):
    """In-place Fast Walsh-Hadamard Transform for a 512-element FP32 block.
    O(N log N) butterfly: 9 stages × 512 ops = 4608 add/subs.
    Normalizes by 1/sqrt(512) to preserve L2 norm."""
    var h = 1
    while h < 512:
        var i = 0
        while i < 512:
            # SIMD butterfly when stride h ≥ 4 (NEON 128-bit = 4 floats)
            if h >= 4:
                var j = 0
                while j < h:
                    var x = (data + i + j).load[width=4]()
                    var y = (data + i + j + h).load[width=4]()
                    (data + i + j).store[width=4](x + y)
                    (data + i + j + h).store[width=4](x - y)
                    j += 4
            else:
                for j in range(h):
                    var x = data[i + j]
                    var y = data[i + j + h]
                    data[i + j] = x + y
                    data[i + j + h] = x - y
            i += h * 2
        h *= 2
    # Normalize: 1/sqrt(512) ≈ 0.044194173824159216
    var norm_v = SIMD[DType.float32, 4](0.044194173824159216)
    for i in range(0, 512, 4):
        (data + i).store[width=4]((data + i).load[width=4]() * norm_v)

@no_inline
def wht_fp32_1536(data: UnsafePointer[Float32, MutUntrackedOrigin]):
    """Block-diagonal WHT for 1536 dims: 3 independent 512-dim blocks.
    1536 = 3 × 512 = 3 × 2^9. Each block gets full WHT decorrelation.
    Preserves L2 norm: ||WHT(x)||² = ||x||² (orthogonal transform)."""
    wht_fp32_inplace_512(data)
    wht_fp32_inplace_512(data + 512)
    wht_fp32_inplace_512(data + 1024)

@always_inline
def quantize_fp32_to_int4_simd(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
    qmin: Float32,
    scale: Float32):
    """Vectorized FP32→INT4 quantization: packs 2 values per byte as (hi<<4)|lo.
    dst has dim/2 bytes. scale = 15.0 / (global_max - global_min)."""
    comptime fp_width = simd_width_of[DType.float32]()  # 4 on M4 NEON
    var scale_v = SIMD[DType.float32, fp_width](scale)
    var qmin_v  = SIMD[DType.float32, fp_width](qmin)
    var lo_v    = SIMD[DType.float32, fp_width](0.0)
    var hi_v    = SIMD[DType.float32, fp_width](15.0)
    var di = 0
    # Process pairs: 2 × fp_width floats → fp_width packed bytes
    while di + fp_width * 2 <= dim:
        var norm_even = (src.load[width=fp_width](di) - qmin_v) * scale_v
        var norm_odd  = (src.load[width=fp_width](di + fp_width) - qmin_v) * scale_v
        var even = min(max(norm_even, lo_v), hi_v).cast[DType.int8]()
        var odd  = min(max(norm_odd,  lo_v), hi_v).cast[DType.int8]()
        # Pack: byte[j] = (odd[j] << 4) | even[j]  — even dims in lo nibble
        var packed = (odd << 4) | even
        dst.store[width=fp_width](di // 2, packed)
        di += fp_width * 2
    # Scalar tail
    while di + 1 < dim:
        var v0 = (src[di] - qmin) * scale
        var v1 = (src[di + 1] - qmin) * scale
        if v0 > 15.0: v0 = 15.0
        if v0 < 0.0:  v0 = 0.0
        if v1 > 15.0: v1 = 15.0
        if v1 < 0.0:  v1 = 0.0
        dst[di // 2] = (Int8(Int(v1)) << 4) | Int8(Int(v0))
        di += 2
    if di < dim:
        var v0 = (src[di] - qmin) * scale
        if v0 > 15.0: v0 = 15.0
        if v0 < 0.0:  v0 = 0.0
        dst[di // 2] = Int8(Int(v0))

@no_inline
def norm_sq_int4_jit[dim: Int](v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """INT4 norm squared: sum(v_i^2) where v is packed INT4 (dim/2 bytes).
    Values in [0,15]. Max per-dim contribution: 225. Max total: dim×225."""
    comptime bytes = dim // 2
    comptime width = simd_width_of[DType.int8]()  # 16
    var sum = SIMD[DType.int32, width](0)
    var mask = SIMD[DType.int8, width](0x0F)
    for i in range(0, bytes - width + 1, width):
        var b = v.load[width=width](i)
        var lo = (b & mask).cast[DType.int32]()
        var hi = ((b >> 4) & mask).cast[DType.int32]()
        sum += lo * lo + hi * hi
    var total = sum.reduce_add()
    # Tail
    comptime tail_start = (bytes // width) * width
    for i in range(tail_start, bytes):
        var b = v[i]
        var lo = (b & 0x0F).cast[DType.int32]()
        var hi = ((b >> 4) & 0x0F).cast[DType.int32]()
        total += lo * lo + hi * hi
    return total.cast[DType.float32]()

@no_inline
def l2_distance_int4_suffix_early_exit_jit[suffix: Int, block: Int](
    q: UnsafePointer[Int8, MutUntrackedOrigin],
    v: UnsafePointer[Int8, MutUntrackedOrigin],
    prefix_dist: Float32,
    threshold: Float32) -> Float32:
    """INT4 suffix L2 with block-wise early exit. q/v are packed INT4 pointers
    starting at the suffix offset. suffix and block are in logical dimensions.
    Returns 1e30 sentinel if pruned, else total distance."""
    comptime width = simd_width_of[DType.int8]()  # 16
    var mask = SIMD[DType.int8, width](0x0F)
    var running_dist = prefix_dist
    comptime packed_block = block // 2  # bytes per block (128 for block=256)
    comptime for b in range(suffix // block):
        var sum = SIMD[DType.int32, width](0)
        for i in range(0, packed_block - width + 1, width):
            var qi = q.load[width=width](b * packed_block + i)
            var vi = v.load[width=width](b * packed_block + i)
            var dl = (qi & mask).cast[DType.int32]() - (vi & mask).cast[DType.int32]()
            var dh = ((qi >> 4) & mask).cast[DType.int32]() - ((vi >> 4) & mask).cast[DType.int32]()
            sum += dl * dl + dh * dh
        running_dist += sum.reduce_add().cast[DType.float32]()
        if running_dist > threshold:
            return Float32(1e30)
    return running_dist

# ═══════════════════════════════════════════════════════════════════════════════
# M6b: Block-wise INT4 quantization (SQ4_Block32)
# 48 blocks × 32 dims, each block: [FP16 scale (2B)] [INT4 packed (16B)] = 18B
# Total per vector: 4B norm + 48 × 18B = 868B (vs 1536B INT8 = −44%)
# ═══════════════════════════════════════════════════════════════════════════════

comptime BLOCK_DIM = 32
comptime NUM_BLOCKS_1536 = 48  # 1536 / 32
comptime BLOCK_BYTES = 18      # 2B scale + 16B data
comptime VEC_BYTES_1536 = 868  # 4B norm + 48 * 18B

@no_inline
def quantize_fp32_to_block_int4(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int):
    """Block-wise symmetric INT4 quantization.
    Layout per vector: [FP32 norm (4B)] [block0: FP16 scale (2B) + INT4 data (16B)] × 48
    INT4 values are symmetric [-8, 7], packed as unsigned [0, 15] in nibbles.
    dst must have at least 4 + num_blocks * 18 bytes."""
    var num_blocks = dim // BLOCK_DIM
    # Compute reconstructed norm (for distance correctness)
    var recon_norm: Float32 = 0.0
    var out_off = 4  # skip 4-byte norm header

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        # Find block absmax for symmetric quantization
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src[base + i]
            var av = v if v >= 0 else -v
            if av > absmax: absmax = av
        # Scale: maps [-absmax, absmax] → [-8, 7] (symmetric, 16 levels)
        # Quantized = round(val / scale), clamped to [-8, 7], stored as unsigned [0, 15]
        var scale = absmax / Float32(7.0) if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale

        # Write FP16 scale
        var scale_f16 = scale.cast[DType.float16]()
        (dst + out_off).bitcast[Float16]()[0] = scale_f16
        out_off += 2

        # Quantize 32 dims and pack in SDOT-friendly layout:
        # byte[i] = (dim[i+16] << 4) | dim[i] for i=0..15
        # After unpack: lo[0..15] = dims 0..15, hi[0..15] = dims 16..31
        # This aligns directly with query INT8 layout → enables SDOT without deinterleave.
        var q_vals = Array[Int, 32](uninitialized=True)
        for i in range(BLOCK_DIM):
            var q_f = src[base + i] * inv_scale
            var q = Int(q_f + Float32(0.5)) if q_f >= 0 else Int(q_f - Float32(0.5))
            if q > 7: q = 7
            if q < -8: q = -8
            q_vals[i] = q
            var r = Float32(q) * scale
            recon_norm += r * r
        # Pack: lo nibble = dim[i], hi nibble = dim[i+16]
        for i in range(16):
            var u_lo = UInt8(q_vals[i] + 8)       # dims 0..15
            var u_hi = UInt8(q_vals[i + 16] + 8)  # dims 16..31
            dst[out_off] = ((u_hi << 4) | u_lo).cast[DType.int8]()
            out_off += 1

    # Write reconstructed norm at offset 0
    dst.bitcast[Float32]()[0] = recon_norm

@no_inline
def dequantize_block_int4_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """M4: Inverse of quantize_fp32_to_block_int4 — used by ATTEND.QUERY to return
    FP32 V vectors from turbo4-packed storage.

    Layout per vector: [FP32 norm (4B)] [block: FP16 scale (2B) + 16B INT4 data] × (dim/32)
    INT4 nibbles are stored unsigned [0, 15] representing symmetric [-8, 7]:
        byte[i] = (dim[i+16] << 4) | dim[i]  for i = 0..15 within a block
    So lo nibble = dim i (0..15), hi nibble = dim i+16 (16..31)."""
    var num_blocks = dim // BLOCK_DIM
    var in_off = 4  # skip FP32 norm header
    for b in range(num_blocks):
        var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
        in_off += 2
        var base = b * BLOCK_DIM
        # Unpack 16 bytes → 32 dims
        for i in range(16):
            var byte = UInt8(src[in_off + i])
            var lo_u = Int((byte & UInt8(0x0F)))      # 0..15
            var hi_u = Int((byte >> 4) & UInt8(0x0F))  # 0..15
            var lo_q = lo_u - 8                        # -8..7
            var hi_q = hi_u - 8
            dst[base + i]      = Float32(lo_q) * scale
            dst[base + i + 16] = Float32(hi_q) * scale
        in_off += 16


@no_inline
def fused_topk_dequant_v_turbo4(
    token_ids: UnsafePointer[Int, MutUntrackedOrigin],
    k: Int,
    v_blob: UnsafePointer[Int8, MutUntrackedOrigin],
    bytes_per_token: Int,
    dim: Int,
    dst_fp32: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """M4: Fused scatter-gather dequant — takes k token IDs, gathers their turbo4-packed
    V vectors from the blob, and writes all k results as contiguous FP32 in one pass.
    Avoids per-result function call overhead and keeps prefetch pattern tight.
    dst_fp32 must have space for k * dim Float32 values."""
    var num_blocks = dim // BLOCK_DIM
    for ri in range(k):
        var tid = token_ids[ri]
        var src = v_blob + tid * bytes_per_token
        var dst = dst_fp32 + ri * dim
        # Prefetch next token's data to overlap with current decode
        if ri + 1 < k:
            var next_src = v_blob + token_ids[ri + 1] * bytes_per_token
            prefetch(next_src)
            prefetch(next_src + 64)
        # Dequant this token: skip 4B norm, then decode blocks
        var in_off = 4
        for b in range(num_blocks):
            var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
            in_off += 2
            var base = b * BLOCK_DIM
            for i in range(16):
                var byte = UInt8(src[in_off + i])
                var lo_u = Int((byte & UInt8(0x0F)))
                var hi_u = Int((byte >> 4) & UInt8(0x0F))
                dst[base + i]      = Float32(lo_u - 8) * scale
                dst[base + i + 16] = Float32(hi_u - 8) * scale
            in_off += 16


def fused_topk_dequant_v_turbo3(
    token_ids: UnsafePointer[Int, MutUntrackedOrigin],
    k: Int,
    v_blob: UnsafePointer[Int8, MutUntrackedOrigin],
    bytes_per_token: Int,
    dim: Int,
    dst_fp32: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """gh #131 §5.3: fused scatter-gather dequant for turbo3 (block-INT3), mirroring
    fused_topk_dequant_v_turbo4 — gathers k tokens' packed V vectors with a tight
    prefetch pattern and inlines the INT3 decode, avoiding per-result call overhead."""
    var num_blocks = dim // BLOCK_DIM
    for ri in range(k):
        var tid = token_ids[ri]
        var src = v_blob + tid * bytes_per_token
        var dst = dst_fp32 + ri * dim
        if ri + 1 < k:
            var next_src = v_blob + token_ids[ri + 1] * bytes_per_token
            prefetch(next_src)
            prefetch(next_src + 64)
        var in_off = 4
        for b in range(num_blocks):
            var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
            in_off += 2
            var base = b * BLOCK_DIM
            for half in range(2):  # 0=lo(dims 0-15), 1=hi(dims 16-31)
                for g in range(2):  # 2 groups of 8 per half
                    var goff = in_off + half * 6 + g * 3
                    var b0 = Int(src[goff].cast[DType.uint8]())
                    var b1 = Int(src[goff + 1].cast[DType.uint8]())
                    var b2 = Int(src[goff + 2].cast[DType.uint8]())
                    var word = b0 | (b1 << 8) | (b2 << 16)
                    var dim_base = base + half * 16 + g * 8
                    dst[dim_base]     = Float32((word & 7) - 4) * scale
                    dst[dim_base + 1] = Float32(((word >> 3) & 7) - 4) * scale
                    dst[dim_base + 2] = Float32(((word >> 6) & 7) - 4) * scale
                    dst[dim_base + 3] = Float32(((word >> 9) & 7) - 4) * scale
                    dst[dim_base + 4] = Float32(((word >> 12) & 7) - 4) * scale
                    dst[dim_base + 5] = Float32(((word >> 15) & 7) - 4) * scale
                    dst[dim_base + 6] = Float32(((word >> 18) & 7) - 4) * scale
                    dst[dim_base + 7] = Float32(((word >> 21) & 7) - 4) * scale
            in_off += 12


def fused_topk_dequant_v_turbo2(
    token_ids: UnsafePointer[Int, MutUntrackedOrigin],
    k: Int,
    v_blob: UnsafePointer[Int8, MutUntrackedOrigin],
    bytes_per_token: Int,
    dim: Int,
    dst_fp32: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """gh #131 §5.3: fused scatter-gather dequant for turbo2 (block-INT2), mirroring
    fused_topk_dequant_v_turbo4 with an inlined INT2 decode."""
    var num_blocks = dim // BLOCK_DIM
    for ri in range(k):
        var tid = token_ids[ri]
        var src = v_blob + tid * bytes_per_token
        var dst = dst_fp32 + ri * dim
        if ri + 1 < k:
            var next_src = v_blob + token_ids[ri + 1] * bytes_per_token
            prefetch(next_src)
            prefetch(next_src + 64)
        var in_off = 4
        for b in range(num_blocks):
            var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
            in_off += 2
            var base = b * BLOCK_DIM
            for half in range(2):  # 0=lo(dims 0-15), 1=hi(dims 16-31)
                var off = in_off + half * 4
                var b0 = Int(src[off].cast[DType.uint8]())
                var b1 = Int(src[off + 1].cast[DType.uint8]())
                var b2 = Int(src[off + 2].cast[DType.uint8]())
                var b3 = Int(src[off + 3].cast[DType.uint8]())
                var word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                var dim_base = base + half * 16
                for i in range(16):
                    var val = ((word >> (i * 2)) & 3) * 2 - 3  # {-3,-1,1,3}
                    dst[dim_base + i] = Float32(val) * scale
            in_off += 8


def fused_topk_dequant_v_fp8(
    token_ids: UnsafePointer[Int, MutUntrackedOrigin],
    k: Int,
    v_blob: UnsafePointer[Int8, MutUntrackedOrigin],
    bytes_per_token: Int,
    dim: Int,
    dst_fp32: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """gh #131 §5.3: fused scatter-gather dequant for fp8 (block E4M3), mirroring
    fused_topk_dequant_v_turbo4 with an inlined FP8 decode."""
    var num_blocks = dim // BLOCK_DIM
    for ri in range(k):
        var tid = token_ids[ri]
        var src = v_blob + tid * bytes_per_token
        var dst = dst_fp32 + ri * dim
        if ri + 1 < k:
            var next_src = v_blob + token_ids[ri + 1] * bytes_per_token
            prefetch(next_src)
            prefetch(next_src + 64)
        var in_off = 4
        for b in range(num_blocks):
            var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
            in_off += 2
            var base = b * BLOCK_DIM
            for i in range(BLOCK_DIM):
                var v_e4m3 = (src + in_off + i).bitcast[Scalar[DType.float8_e4m3fn]]()[0]
                dst[base + i] = v_e4m3.cast[DType.float32]() * scale
            in_off += BLOCK_DIM


@no_inline
def dequantize_block_int3_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """M4: Inverse of quantize_fp32_to_block_int3.
    Layout per vector: [FP32 norm (4B)] [block: FP16 scale (2B) + 6B lo + 6B hi] × (dim/32)
    3-bit packed: 8 values in 3 bytes. Unsigned [0,7] → signed [-4,3]."""
    var num_blocks = dim // BLOCK_DIM
    var in_off = 4  # skip norm header
    for b in range(num_blocks):
        var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
        in_off += 2
        var base = b * BLOCK_DIM
        # Decode all 32 dims from 12 packed bytes (4 groups of 8 values × 3 bytes each)
        # lo: dims 0..15 in bytes [in_off..in_off+5] (2 groups of 8)
        # hi: dims 16..31 in bytes [in_off+6..in_off+11] (2 groups of 8)
        for half in range(2):  # 0=lo(dims 0-15), 1=hi(dims 16-31)
            for g in range(2):  # 2 groups of 8 per half
                var goff = in_off + half * 6 + g * 3
                var b0 = Int(src[goff].cast[DType.uint8]())
                var b1 = Int(src[goff + 1].cast[DType.uint8]())
                var b2 = Int(src[goff + 2].cast[DType.uint8]())
                var word = b0 | (b1 << 8) | (b2 << 16)
                var dim_base = base + half * 16 + g * 8
                dst[dim_base]     = Float32((word & 7) - 4) * scale
                dst[dim_base + 1] = Float32(((word >> 3) & 7) - 4) * scale
                dst[dim_base + 2] = Float32(((word >> 6) & 7) - 4) * scale
                dst[dim_base + 3] = Float32(((word >> 9) & 7) - 4) * scale
                dst[dim_base + 4] = Float32(((word >> 12) & 7) - 4) * scale
                dst[dim_base + 5] = Float32(((word >> 15) & 7) - 4) * scale
                dst[dim_base + 6] = Float32(((word >> 18) & 7) - 4) * scale
                dst[dim_base + 7] = Float32(((word >> 21) & 7) - 4) * scale
        in_off += 12


@no_inline
def dequantize_block_int2_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """M4: Inverse of quantize_fp32_to_block_int2.
    Layout per vector: [FP32 norm (4B)] [block: FP16 scale (2B) + 4B lo + 4B hi] × (dim/32)
    2-bit packed: 16 values in 4 bytes. Unsigned [0,3] → signed {-3,-1,1,3} via val*2-3."""
    var num_blocks = dim // BLOCK_DIM
    var in_off = 4  # skip norm header
    for b in range(num_blocks):
        var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
        in_off += 2
        var base = b * BLOCK_DIM
        # Decode all 32 dims from 8 packed bytes (4B lo + 4B hi)
        for half in range(2):  # 0=lo(dims 0-15), 1=hi(dims 16-31)
            var off = in_off + half * 4
            var b0 = Int(src[off].cast[DType.uint8]())
            var b1 = Int(src[off + 1].cast[DType.uint8]())
            var b2 = Int(src[off + 2].cast[DType.uint8]())
            var b3 = Int(src[off + 3].cast[DType.uint8]())
            var word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            var dim_base = base + half * 16
            for i in range(16):
                var val = ((word >> (i * 2)) & 3) * 2 - 3  # {-3,-1,1,3}
                dst[dim_base + i] = Float32(val) * scale
        in_off += 8


@no_inline
def quantize_fp32_to_block_int8(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst_int8: UnsafePointer[Int8, MutUntrackedOrigin],
    dst_scales: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int) -> Float32:
    """Block-wise symmetric INT8 quantization for the QUERY vector.
    Each 32-dim block → FP32 scale + 32 × INT8 values in [-127, 127].
    Returns the reconstructed norm squared (for distance computation).
    dst_int8: dim bytes. dst_scales: num_blocks × 4 bytes."""
    var num_blocks = dim // BLOCK_DIM
    var recon_norm: Float32 = 0.0

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src[base + i]
            var av = v if v >= 0 else -v
            if av > absmax: absmax = av
        var scale = absmax / Float32(127.0) if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale
        dst_scales[b] = scale

        for i in range(BLOCK_DIM):
            var q_f = src[base + i] * inv_scale
            var q = Int(q_f + Float32(0.5)) if q_f >= 0 else Int(q_f - Float32(0.5))
            if q > 127: q = 127
            if q < -127: q = -127
            dst_int8[base + i] = Int8(q)
            var r = Float32(q) * scale
            recon_norm += r * r

    return recon_norm

@no_inline
def block_int4_dot_single_simd(
    q_int8: UnsafePointer[Int8, MutUntrackedOrigin],
    q_scales: UnsafePointer[Float32, MutUntrackedOrigin],
    v_block: UnsafePointer[Int8, MutUntrackedOrigin],
) -> Float32:
    """SDOT-optimized asymmetric dot: INT8 query × block-INT4 vector.
    SDOT-friendly layout: byte[i] = (dim[i+16]<<4) | dim[i].
    After unpack: lo=dims 0..15, hi=dims 16..31 → aligns with query directly.
    2 SDOT calls per 32-dim block = 96 SDOT calls total for 1536 dims."""
    var total_dot: Float32 = 0.0
    var v_off = 4  # skip norm header

    for b in range(NUM_BLOCKS_1536):
        var scale_v = (v_block + v_off).bitcast[Float16]()[0].cast[DType.float32]()
        v_off += 2
        var combined_scale = q_scales[b] * scale_v

        # Load 16 packed bytes, unpack to 2×16 INT8
        var packed = (v_block + v_off).load[width=16]()
        var v_lo = (packed & SIMD[DType.int8, 16](0x0F)) - SIMD[DType.int8, 16](8)  # dims 0..15
        var v_hi = ((packed >> 4) & SIMD[DType.int8, 16](0x0F)) - SIMD[DType.int8, 16](8)  # dims 16..31

        # Query halves align directly with unpacked vectors
        var q_base = b * BLOCK_DIM
        var q_lo = (q_int8 + q_base).load[width=16]()       # dims 0..15
        var q_hi = (q_int8 + q_base + 16).load[width=16]()  # dims 16..31

        # 2× SDOT: each produces 4 int32 partial sums from 16 int8×int8 products
        var acc = SIMD[DType.int32, 4](0)
        acc = sdot_int8(acc, q_lo, v_lo)
        acc = sdot_int8(acc, q_hi, v_hi)

        total_dot += acc.reduce_add().cast[DType.float32]() * combined_scale
        v_off += 16

    return total_dot

# block_int4_dot_batch8: tuned batch kernel, closed in libpion_vector (D11).

# ═══════════════════════════════════════════════════════════════════════════════
# M7: TurboQuant — 3-bit block quantization + QJL error correction
# 48 blocks × 32 dims, each block: [FP16 scale (2B)] [3-bit packed (12B)] = 14B
# Total per vector: 4B norm + 48 × 14B = 676B (vs 868B INT4 = −22%)
# QJL: 192B/vector sign bits in separate buffer for Hamming distance correction.
# ═══════════════════════════════════════════════════════════════════════════════

comptime INT3_BLOCK_DATA = 12       # 32 dims × 3 bits / 8
comptime INT3_BLOCK_BYTES = 14      # 2B scale + 12B data
comptime INT3_VEC_BYTES_1536 = 676  # 4B norm + 48 * 14B
comptime QJL_U64S_1536 = 24        # 1536 bits / 64
comptime QJL_BYTES_1536 = 192      # 1536 bits / 8

@always_inline
def _pack_8_int3(vals: Array[Int, 8], dst: UnsafePointer[Int8, MutUntrackedOrigin], off: Int):
    """Pack 8 unsigned 3-bit values [0,7] into 3 bytes.
    Layout: byte[0] = v0|(v1<<3)|((v2&3)<<6), byte[1] = (v2>>2)|(v3<<1)|(v4<<4)|((v5&1)<<7),
            byte[2] = (v5>>1)|(v6<<2)|(v7<<5)."""
    var v0 = UInt8(vals[0]); var v1 = UInt8(vals[1]); var v2 = UInt8(vals[2]); var v3 = UInt8(vals[3])
    var v4 = UInt8(vals[4]); var v5 = UInt8(vals[5]); var v6 = UInt8(vals[6]); var v7 = UInt8(vals[7])
    dst[off]     = (v0 | (v1 << 3) | ((v2 & 3) << 6)).cast[DType.int8]()
    dst[off + 1] = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 1) << 7)).cast[DType.int8]()
    dst[off + 2] = ((v5 >> 1) | (v6 << 2) | (v7 << 5)).cast[DType.int8]()



@always_inline
def _byte_u64(p: UnsafePointer[Int8, MutUntrackedOrigin], i: Int) -> UInt64:
    """Byte i of p as an UNSIGNED UInt64 (0..255).

    Do not write `UInt64(p[i].cast[DType.uint8]())`: Mojo 1.0 folds that
    int8 -> uint8 -> uint64 chain into one SIGN extension, at -O0 too, so any
    byte >= 128 comes back as 0xFFFF_FFFF_FFFF_FFxx and its high ones OR over
    every neighbouring packed code. That silently corrupted every INT3/INT2
    unpack (TurboQuant/NanoQuant search, the QJL residual at build).
    Widening through Int and masking cannot be folded wrong."""
    return UInt64(Int(p[i]) & 0xFF)


@always_inline
def _unpack16_int3(src: UnsafePointer[Int8, MutUntrackedOrigin], off: Int) -> SIMD[DType.int8, 16]:
    """`_unpack_16_int3` returning the 16 values in a register instead of
    writing a caller buffer — the dot kernels used to heap-alloc that buffer
    per call (D11: a closed routine must not allocate)."""
    var u0 = _byte_u64(src, off) | (_byte_u64(src, off + 1) << 8) | (_byte_u64(src, off + 2) << 16)
    var u1 = _byte_u64(src, off + 3) | (_byte_u64(src, off + 4) << 8) | (_byte_u64(src, off + 5) << 16)
    var r = SIMD[DType.int8, 16](0)
    comptime for k in range(8):
        r[k] = Int8(Int((u0 >> UInt64(3 * k)) & 7) - 4)
        r[8 + k] = Int8(Int((u1 >> UInt64(3 * k)) & 7) - 4)
    return r

@always_inline
def _tbl16(t: SIMD[DType.uint8, 16], idx: SIMD[DType.uint8, 16]) -> SIMD[DType.uint8, 16]:
    """NEON TBL: lane i = t[idx[i]], 0 for an index >= 16."""
    return llvm_intrinsic["llvm.aarch64.neon.tbl1.v16i8", SIMD[DType.uint8, 16],
                          has_side_effect=False](t, idx)


@always_inline
def unpack32_int3(v: UnsafePointer[Int8, MutUntrackedOrigin], blk: Int
                  ) -> Tuple[SIMD[DType.int8, 16], SIMD[DType.int8, 16]]:
    """gh #396: all 32 codes of one block-INT3 block, as (dims 0-15, dims 16-31).
    `blk` is the byte offset of the block's FP16 scale; its 12 data bytes follow.

    `_unpack16_int3` builds 16 codes from six scalar byte loads and 16
    shift/mask/insert steps, twice per block, in the innermost loop of every
    TurboQuant distance. Here: ONE 16-byte load, four TBLs that gather each
    code's two covering bytes into a u16 lane, one variable shift, one mask.
    The load starts at `blk - 2` so it ends exactly at the block's last data
    byte (data byte j is lane 4 + j): no over-read, so no `volatile` needed.
    Outputs are bit-identical to two `_unpack16_int3` calls; non-NEON targets
    use exactly those."""
    comptime if CompilationTarget.has_neon():
        var raw = bitcast[DType.uint8, 16]((v + blk - 2).load[width=16]())
        # Code k of a 3-byte group starts at bit 3k: byte 3k//8, shift 3k%8.
        comptime I0 = SIMD[DType.uint8, 16](4, 5, 4, 5, 4, 5, 5, 6, 5, 6, 5, 6, 6, 7, 6, 7)
        comptime I1 = SIMD[DType.uint8, 16](7, 8, 7, 8, 7, 8, 8, 9, 8, 9, 8, 9, 9, 10, 9, 10)
        comptime I2 = SIMD[DType.uint8, 16](10, 11, 10, 11, 10, 11, 11, 12, 11, 12, 11, 12, 12, 13, 12, 13)
        comptime I3 = SIMD[DType.uint8, 16](13, 14, 13, 14, 13, 14, 14, 15, 14, 15, 14, 15, 15, 255, 15, 255)
        comptime SH = SIMD[DType.uint16, 8](0, 3, 6, 1, 4, 7, 2, 5)
        var w0 = bitcast[DType.uint16, 8](_tbl16(raw, I0)) >> SH
        var w1 = bitcast[DType.uint16, 8](_tbl16(raw, I1)) >> SH
        var w2 = bitcast[DType.uint16, 8](_tbl16(raw, I2)) >> SH
        var w3 = bitcast[DType.uint16, 8](_tbl16(raw, I3)) >> SH
        var lo = w0.cast[DType.uint8]().join(w1.cast[DType.uint8]()) & 7
        var hi = w2.cast[DType.uint8]().join(w3.cast[DType.uint8]()) & 7
        return (bitcast[DType.int8, 16](lo) - 4, bitcast[DType.int8, 16](hi) - 4)
    else:
        return (_unpack16_int3(v, blk + 2), _unpack16_int3(v, blk + 8))


@no_inline
def quantize_fp32_to_block_int3(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int):
    """Block-wise symmetric INT3 quantization.
    Layout per vector: [FP32 norm (4B)] [block0: FP16 scale (2B) + 3-bit packed (12B)] × 48
    3-bit values: symmetric [-4, 3], packed as unsigned [0, 7] in groups of 8 → 3 bytes.
    Within each 32-dim block: first 16 dims packed into 6 bytes, next 16 into 6 bytes.
    dst must have at least 4 + num_blocks * 14 bytes."""
    var num_blocks = dim // BLOCK_DIM
    var recon_norm: Float32 = 0.0
    var out_off = 4  # skip 4-byte norm header

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        # Find block absmax for symmetric quantization
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src[base + i]
            var av = v if v >= 0 else -v
            if av > absmax: absmax = av
        # Scale: maps [-absmax, absmax] → [-4, 3] (8 levels, symmetric)
        var scale = absmax / Float32(3.0) if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale

        # Write FP16 scale
        (dst + out_off).bitcast[Float16]()[0] = scale.cast[DType.float16]()
        out_off += 2

        # Quantize 32 dims
        var q_vals = Array[Int, 32](uninitialized=True)
        for i in range(BLOCK_DIM):
            var q_f = src[base + i] * inv_scale
            var q = Int(q_f + Float32(0.5)) if q_f >= 0 else Int(q_f - Float32(0.5))
            if q > 3: q = 3
            if q < -4: q = -4
            q_vals[i] = q
            var r = Float32(q) * scale
            recon_norm += r * r

        # Pack lo 16 dims (0..15) into 6 bytes: 2 groups of 8 → 2×3 bytes
        var u_lo0 = Array[Int, 8](uninitialized=True)
        var u_lo1 = Array[Int, 8](uninitialized=True)
        for i in range(8):
            u_lo0[i] = q_vals[i] + 4      # unsigned [0, 7]
            u_lo1[i] = q_vals[i + 8] + 4
        _pack_8_int3(u_lo0, dst, out_off)
        _pack_8_int3(u_lo1, dst, out_off + 3)
        out_off += 6

        # Pack hi 16 dims (16..31) into 6 bytes
        var u_hi0 = Array[Int, 8](uninitialized=True)
        var u_hi1 = Array[Int, 8](uninitialized=True)
        for i in range(8):
            u_hi0[i] = q_vals[16 + i] + 4
            u_hi1[i] = q_vals[24 + i] + 4
        _pack_8_int3(u_hi0, dst, out_off)
        _pack_8_int3(u_hi1, dst, out_off + 3)
        out_off += 6

    # Write reconstructed norm at offset 0
    dst.bitcast[Float32]()[0] = recon_norm

@no_inline
def int3_dot_single_simd(
    q_int8: UnsafePointer[Int8, MutUntrackedOrigin],
    q_scales: UnsafePointer[Float32, MutUntrackedOrigin],
    v_block: UnsafePointer[Int8, MutUntrackedOrigin],
) -> Float32:
    """Asymmetric dot: INT8 query × block-INT3 vector.
    Unpacks 3-bit → Int8, then 2× SDOT per 32-dim block = 96 SDOT calls for 1536 dims.
    Combined scale = q_scale × v_scale per block."""
    var total_dot: Float32 = 0.0
    var v_off = 4  # skip norm header

    for b in range(NUM_BLOCKS_1536):
        var scale_v = (v_block + v_off).bitcast[Float16]()[0].cast[DType.float32]()
        var combined_scale = q_scales[b] * scale_v
        var q_base = b * BLOCK_DIM

        # All 32 codes of the block (gh #396: one load + TBL on NEON)
        var v32 = unpack32_int3(v_block, v_off)
        var q_lo = (q_int8 + q_base).load[width=16]()
        var q_hi = (q_int8 + q_base + 16).load[width=16]()
        v_off += INT3_BLOCK_BYTES

        var acc = SIMD[DType.int32, 4](0)
        acc = sdot_int8(acc, q_lo, v32[0])
        acc = sdot_int8(acc, q_hi, v32[1])
        total_dot += acc.reduce_add().cast[DType.float32]() * combined_scale

    return total_dot

# int3_dot_batch8: tuned batch kernel, closed in libpion_vector (D11).

@no_inline
def qjl_compute_signs(
    residual: UnsafePointer[Float32, MutUntrackedOrigin],
    random_signs: UnsafePointer[UInt64, MutUntrackedOrigin],
    wht_scratch: UnsafePointer[Float32, MutUntrackedOrigin],
    dst_signs: UnsafePointer[UInt64, MutUntrackedOrigin],
    dim: Int) -> Float32:
    """QJL sign computation: apply random sign flips to residual, WHT, store signs.
    Returns residual L2 norm squared.
    random_signs: 24 UInt64s = 1536 random ±1 bits (seeded, same for all vectors).
    dst_signs: 24 UInt64s output (1536 sign bits)."""
    var res_norm_sq: Float32 = 0.0
    # Apply random sign flips and copy to wht_scratch
    for i in range(dim):
        var bit = (random_signs[i >> 6] >> UInt64(i & 63)) & UInt64(1)
        var sign_val = Float32(1.0) if bit == UInt64(1) else Float32(-1.0)
        var r = residual[i]
        wht_scratch[i] = r * sign_val
        res_norm_sq += r * r
    # Apply WHT (block-diagonal 3×512)
    wht_fp32_1536(wht_scratch)
    # Extract sign bits
    for w in range(dim >> 6):  # 24 words
        var bits: UInt64 = 0
        for b in range(64):
            if wht_scratch[w * 64 + b] > 0:
                bits |= (UInt64(1) << UInt64(b))
        dst_signs[w] = bits
    return res_norm_sq

# qjl_hamming_batch8: tuned batch kernel, closed in libpion_vector (D11).

# ═══════════════════════════════════════════════════════════════════════════════
# N4: NanoQuant — 2-bit block quantization
# 48 blocks × 32 dims, each block: [FP16 scale (2B)] [2-bit packed (8B)] = 10B
# Total per vector: 4B norm + 48 × 10B = 484B (vs 676B INT3, 868B INT4)
# Signed levels: {-3, -1, 1, 3} via (unsigned * 2 - 3); stored scale = absmax / 3,
# so the outer levels decode to ±absmax (see quantize_fp32_to_block_int2)
# ═══════════════════════════════════════════════════════════════════════════════

comptime INT2_BLOCK_DATA = 8        # 32 dims × 2 bits / 8
comptime INT2_BLOCK_BYTES = 10      # 2B scale + 8B data
comptime INT2_VEC_BYTES_1536 = 484  # 4B norm + 48 * 10B

@always_inline
def _pack_16_int2(vals: Array[Int, 16], dst: UnsafePointer[Int8, MutUntrackedOrigin], off: Int):
    """Pack 16 unsigned 2-bit values [0,3] into 4 bytes.
    Layout: byte[0] = v0|(v1<<2)|(v2<<4)|(v3<<6), etc."""
    var b0 = UInt8(vals[0]) | (UInt8(vals[1]) << 2) | (UInt8(vals[2]) << 4) | (UInt8(vals[3]) << 6)
    var b1 = UInt8(vals[4]) | (UInt8(vals[5]) << 2) | (UInt8(vals[6]) << 4) | (UInt8(vals[7]) << 6)
    var b2 = UInt8(vals[8]) | (UInt8(vals[9]) << 2) | (UInt8(vals[10]) << 4) | (UInt8(vals[11]) << 6)
    var b3 = UInt8(vals[12]) | (UInt8(vals[13]) << 2) | (UInt8(vals[14]) << 4) | (UInt8(vals[15]) << 6)
    dst[off]     = b0.cast[DType.int8]()
    dst[off + 1] = b1.cast[DType.int8]()
    dst[off + 2] = b2.cast[DType.int8]()
    dst[off + 3] = b3.cast[DType.int8]()



@always_inline
def _unpack16_int2(src: UnsafePointer[Int8, MutUntrackedOrigin], off: Int) -> SIMD[DType.int8, 16]:
    """`_unpack_16_int2` returning the 16 values in a register (see
    `_unpack16_int3`)."""
    var packed = _byte_u64(src, off) | (_byte_u64(src, off + 1) << 8) | (_byte_u64(src, off + 2) << 16) | (_byte_u64(src, off + 3) << 24)
    var r = SIMD[DType.int8, 16](0)
    comptime for k in range(16):
        r[k] = Int8(Int((packed >> UInt64(2 * k)) & 3) * 2 - 3)
    return r

@always_inline
def unpack32_int2(v: UnsafePointer[Int8, MutUntrackedOrigin], blk: Int
                  ) -> Tuple[SIMD[DType.int8, 16], SIMD[DType.int8, 16]]:
    """gh #396: all 32 codes of one block-INT2 block, as (dims 0-15, dims 16-31),
    decoded to {-3,-1,1,3}. `blk` is the byte offset of the block's FP16
    scale; its 8 data bytes follow. One exact 8-byte load, two TBLs that copy
    each byte into the four lanes it feeds, a per-lane shift and a mask —
    instead of 16 scalar steps and a multiply per lane, twice per block.
    Bit-identical to two `_unpack16_int2` calls; non-NEON targets use those."""
    comptime if CompilationTarget.has_neon():
        var raw8 = bitcast[DType.uint8, 8]((v + blk + 2).load[width=8]())
        var raw = raw8.join(raw8)
        comptime J0 = SIMD[DType.uint8, 16](0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)
        comptime J1 = SIMD[DType.uint8, 16](4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7)
        comptime S = SIMD[DType.uint8, 16](0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6)
        var lo = (_tbl16(raw, J0) >> S) & 3
        var hi = (_tbl16(raw, J1) >> S) & 3
        return (bitcast[DType.int8, 16](lo) * 2 - 3, bitcast[DType.int8, 16](hi) * 2 - 3)
    else:
        return (_unpack16_int2(v, blk + 2), _unpack16_int2(v, blk + 6))


@no_inline
def quantize_fp32_to_block_int2(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int):
    """Block-wise symmetric INT2 quantization.
    Layout per vector: [FP32 norm (4B)] [block0: FP16 scale (2B) + 2-bit packed (8B)] × 48
    4 levels: unsigned [0,3] → signed {-3,-1,1,3}, scale = absmax / 1.5.
    Within each 32-dim block: first 16 dims packed into 4 bytes, next 16 into 4 bytes.
    dst must have at least 4 + num_blocks * 10 bytes."""
    var num_blocks = dim // BLOCK_DIM
    var recon_norm: Float32 = 0.0
    var out_off = 4  # skip 4-byte norm header

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        # Find block absmax for symmetric quantization
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src[base + i]
            var av = v if v >= 0 else -v
            if av > absmax: absmax = av
        # Scale: maps [-absmax, absmax] → signed {-3,-1,1,3} (4 levels)
        # Reconstruction: signed_val * scale, where scale = absmax / 1.5
        var scale = absmax / Float32(1.5) if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale

        # Rounding below puts level q at v ≈ (q - 1.5)·scale = (2q - 3)·(scale/2),
        # and every decoder (dot kernels, dequantize, fused turbo2) computes
        # (2q - 3)·stored_scale — so the STORED scale is scale/2. Storing
        # `scale` itself (as shipped until 2026-09-25) doubled every decoded
        # value and quadrupled the stored norm, so the beam's
        # `q_norm + v_norm - 2·dot` over-weighted the norm term 2× and
        # NanoQuant scored recall 0.06 on the gate dataset.
        var stored_scale = scale * Float32(0.5)
        (dst + out_off).bitcast[Float16]()[0] = stored_scale.cast[DType.float16]()
        out_off += 2

        # Quantize 32 dims: val → round(val/scale + 1.5) → clamp [0,3]
        var q_vals = Array[Int, 32](uninitialized=True)
        for i in range(BLOCK_DIM):
            var q_f = src[base + i] * inv_scale + Float32(1.5)
            var q = Int(q_f + Float32(0.5)) if q_f >= 0 else Int(q_f - Float32(0.5))
            if q > 3: q = 3
            if q < 0: q = 0
            q_vals[i] = q
            # Reconstruction value: (q * 2 - 3) * scale
            var signed_q = q * 2 - 3
            var r = Float32(signed_q) * stored_scale
            recon_norm += r * r

        # Pack lo 16 dims (0..15) into 4 bytes
        var u_lo = Array[Int, 16](uninitialized=True)
        for i in range(16):
            u_lo[i] = q_vals[i]
        _pack_16_int2(u_lo, dst, out_off)
        out_off += 4

        # Pack hi 16 dims (16..31) into 4 bytes
        var u_hi = Array[Int, 16](uninitialized=True)
        for i in range(16):
            u_hi[i] = q_vals[16 + i]
        _pack_16_int2(u_hi, dst, out_off)
        out_off += 4

    # Write reconstructed norm at offset 0
    dst.bitcast[Float32]()[0] = recon_norm

@no_inline
def int2_dot_single_simd(
    q_int8: UnsafePointer[Int8, MutUntrackedOrigin],
    q_scales: UnsafePointer[Float32, MutUntrackedOrigin],
    v_block: UnsafePointer[Int8, MutUntrackedOrigin],
) -> Float32:
    """Asymmetric dot: INT8 query × block-INT2 vector.
    Unpacks 2-bit → Int8 {-3,-1,1,3}, then 2× SDOT per 32-dim block.
    Combined scale = q_scale × v_scale per block."""
    var total_dot: Float32 = 0.0
    var v_off = 4  # skip norm header

    for b in range(NUM_BLOCKS_1536):
        var scale_v = (v_block + v_off).bitcast[Float16]()[0].cast[DType.float32]()
        var combined_scale = q_scales[b] * scale_v
        v_off += 2

        var q_base = b * BLOCK_DIM
        var acc = SIMD[DType.int32, 4](0)

        # All 32 codes of the block (gh #396: one load + TBL on NEON)
        var v32 = unpack32_int2(v_block, v_off - 2)
        var q_lo = (q_int8 + q_base).load[width=16]()
        acc = sdot_int8(acc, q_lo, v32[0])
        var q_hi = (q_int8 + q_base + 16).load[width=16]()
        acc = sdot_int8(acc, q_hi, v32[1])
        v_off += INT2_BLOCK_DATA

        total_dot += acc.reduce_add().cast[DType.float32]() * combined_scale

    return total_dot

# int2_dot_batch8: tuned batch kernel, closed in libpion_vector (D11).


# ═══════════════════════════════════════════════════════════════════════════════
# A2 (gh #30): FP8 (E4M3) and BF16-RoPE / FP8-body hybrid V-cache kernels
# Source: DeepSeek-V4 §2.3.4 ("BF16 precision is used for the rotary positional
# embedding (RoPE) dimensions, while FP8 precision is applied to the remaining
# dimensions. This hybrid representation reduces the KV cache size by nearly
# half compared with pure BF16 storage.")
#
# Both formats use Mojo's native DType.float8_e4m3fn (OFP8 standard, sign +
# 4-bit exp biased by 7 + 3-bit mantissa, finite range [-448, 448]).
#
# FP8 layout per token (matches turbo4 family — usable by ATTEND.QUERY):
#   [FP32 norm 4B] [block: FP16 scale 2B + 32 × E4M3 1B] × (dim/32)
#   bpt = 4 + (dim/32) * 34   (vs 4 + (dim/32) * 18 for turbo4)
#
# BF16-RoPE / FP8-body layout per token:
#   [rope_dim × BF16 (2B each)] [(body/32) × (FP16 scale 2B + 32 × E4M3 1B)]
#   bpt = rope_dim * 2 + ((dim - rope_dim) / 32) * 34
#   rope_dim must be ≤ dim; (dim - rope_dim) must be divisible by 32.
# ═══════════════════════════════════════════════════════════════════════════════

# Per-block scale target: keeps per-element values in E4M3's well-quantized
# normal range (avoid the saturating tail near ±448).
comptime FP8_BLOCK_SCALE_MAX = Float32(240.0)

@no_inline
def quantize_fp32_to_block_fp8(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int):
    """A2: FP8 (E4M3) per-block-of-32 quantization.

    Layout per vector: [FP32 norm 4B] [block: FP16 scale 2B + 32 × E4M3 1B] × (dim/32)
    `dim` must be divisible by 32. `dst` must hold at least `4 + (dim/32) * 34`
    bytes. The reconstructed L2 norm is written at offset 0 — same convention as
    the turbo4 family.
    """
    var num_blocks = dim // BLOCK_DIM
    var recon_norm: Float32 = 0.0
    var out_off = 4

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src[base + i]
            var av = v if v >= 0 else -v
            if av > absmax:
                absmax = av
        var scale = absmax / FP8_BLOCK_SCALE_MAX if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale

        var scale_f16 = scale.cast[DType.float16]()
        (dst + out_off).bitcast[Float16]()[0] = scale_f16
        out_off += 2

        for i in range(BLOCK_DIM):
            var v_scaled = src[base + i] * inv_scale
            var v_e4m3 = v_scaled.cast[DType.float8_e4m3fn]()
            (dst + out_off + i).bitcast[Scalar[DType.float8_e4m3fn]]()[0] = v_e4m3
            var v_back = v_e4m3.cast[DType.float32]() * scale
            recon_norm += v_back * v_back
        out_off += BLOCK_DIM

    dst.bitcast[Float32]()[0] = recon_norm


@no_inline
def dequantize_block_fp8_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """A2: Inverse of quantize_fp32_to_block_fp8."""
    var num_blocks = dim // BLOCK_DIM
    var in_off = 4
    for b in range(num_blocks):
        var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
        in_off += 2
        var base = b * BLOCK_DIM
        for i in range(BLOCK_DIM):
            var v_e4m3 = (src + in_off + i).bitcast[Scalar[DType.float8_e4m3fn]]()[0]
            dst[base + i] = v_e4m3.cast[DType.float32]() * scale
        in_off += BLOCK_DIM


@no_inline
def quantize_fp32_to_bf16_rope_fp8_body(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
    rope_dim: Int):
    """A2: BF16/FP8 hybrid quant — first `rope_dim` values stored as BF16
    verbatim, remaining `dim - rope_dim` values stored as block-FP8.

    Storage per token:
        [rope_dim × BF16 2B] [body_blocks × (FP16 scale 2B + 32 × E4M3 1B)]
    where body_blocks = (dim - rope_dim) // 32.
    """
    if rope_dim < 0 or rope_dim > dim:
        return
    if (dim - rope_dim) % BLOCK_DIM != 0:
        return

    var bf16_dst = dst.bitcast[BFloat16]()
    for i in range(rope_dim):
        bf16_dst[i] = src[i].cast[DType.bfloat16]()

    var body_dim = dim - rope_dim
    var num_blocks = body_dim // BLOCK_DIM
    var out_off = rope_dim * 2
    var src_body = src + rope_dim

    for b in range(num_blocks):
        var base = b * BLOCK_DIM
        var absmax: Float32 = 0.0
        for i in range(BLOCK_DIM):
            var v = src_body[base + i]
            var av = v if v >= 0 else -v
            if av > absmax:
                absmax = av
        var scale = absmax / FP8_BLOCK_SCALE_MAX if absmax > 0 else Float32(1.0)
        var inv_scale = Float32(1.0) / scale
        (dst + out_off).bitcast[Float16]()[0] = scale.cast[DType.float16]()
        out_off += 2
        for i in range(BLOCK_DIM):
            var v_scaled = src_body[base + i] * inv_scale
            var v_e4m3 = v_scaled.cast[DType.float8_e4m3fn]()
            (dst + out_off + i).bitcast[Scalar[DType.float8_e4m3fn]]()[0] = v_e4m3
        out_off += BLOCK_DIM


@no_inline
def dequantize_bf16_rope_fp8_body_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int,
    rope_dim: Int):
    """A2: Inverse of quantize_fp32_to_bf16_rope_fp8_body."""
    if rope_dim < 0 or rope_dim > dim:
        return
    if (dim - rope_dim) % BLOCK_DIM != 0:
        return

    var bf16_src = src.bitcast[BFloat16]()
    for i in range(rope_dim):
        dst[i] = bf16_src[i].cast[DType.float32]()

    var body_dim = dim - rope_dim
    var num_blocks = body_dim // BLOCK_DIM
    var in_off = rope_dim * 2
    var dst_body = dst + rope_dim
    for b in range(num_blocks):
        var scale = (src + in_off).bitcast[Float16]()[0].cast[DType.float32]()
        in_off += 2
        var base = b * BLOCK_DIM
        for i in range(BLOCK_DIM):
            var v_e4m3 = (src + in_off + i).bitcast[Scalar[DType.float8_e4m3fn]]()[0]
            dst_body[base + i] = v_e4m3.cast[DType.float32]() * scale
        in_off += BLOCK_DIM


# ── mlx-compatible INT4 group-32, affine (gh #148 Phase 1) ─────────────────
#
# The layout mlx's QuantizedKVCache uses, stored verbatim so a warm attach is a
# straight `mx.array` view with no repack:
#
#   per token, per layer, D = value_dim
#     [packed uint32 × D/8]  4-bit codes, element j of a group at bits 4*(j%8)
#     [scales fp16  × D/32]
#     [biases fp16  × D/32]
#
# Affine (scale + bias), not symmetric: gh #148 gate 2 measured affine +2.92 pp
# on K over symmetric, and gate 4 then showed K's error dominates the end-to-end
# result, so this is the axis that mattered. Group 32, not 64: gate 0 found g64
# corrupts recalled facts at digit grain ("2430-04-22" for "2030-04-22") while
# g32 was clean on the same probes. Knowledge held as KV is bits-fragile the
# same way weight-held knowledge is.
comptime MLX_Q_GROUP = 32
comptime MLX_Q_BITS = 4


@always_inline
def mlx4g32_bytes_per_token(dim: Int) -> Int:
    """D/2 packed bytes + 2B scale + 2B bias per group of 32."""
    return (dim // 2) + (dim // MLX_Q_GROUP) * 4


@no_inline
def quantize_fp32_to_mlx4_g32(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int):
    """fp32 → mlx int4/g32 affine. dst needs mlx4g32_bytes_per_token(dim) bytes."""
    var groups = dim // MLX_Q_GROUP
    var packed = dst.bitcast[UInt32]()
    var scales = (dst + (dim // 2)).bitcast[Float16]()
    var biases = (dst + (dim // 2) + groups * 2).bitcast[Float16]()

    for g in range(groups):
        var base = g * MLX_Q_GROUP
        var lo = src[base]
        var hi = src[base]
        for i in range(1, MLX_Q_GROUP):
            var v = src[base + i]
            if v < lo: lo = v
            if v > hi: hi = v
        # 4 bits → 16 levels; bias is the group minimum, matching mlx's
        # `w_q = round((w - bias) / scale)` convention.
        var scale = (hi - lo) / Float32(15.0)
        if scale <= Float32(0.0):
            scale = Float32(1e-8)
        # Round-trip the scale through fp16 *before* quantizing, so the codes are
        # chosen against the scale the reader will actually see. Skipping this
        # costs up to half a level on every element.
        var s16 = scale.cast[DType.float16]()
        var b16 = lo.cast[DType.float16]()
        scales[g] = s16
        biases[g] = b16
        var s = s16.cast[DType.float32]()
        var b = b16.cast[DType.float32]()
        var inv = Float32(1.0) / s

        for w in range(MLX_Q_GROUP // 8):
            var word = UInt32(0)
            for j in range(8):
                var q_f = (src[base + w * 8 + j] - b) * inv
                var q = Int(q_f + Float32(0.5))
                if q > 15: q = 15
                if q < 0: q = 0
                word |= UInt32(q) << UInt32(4 * j)
            packed[g * (MLX_Q_GROUP // 8) + w] = word


@no_inline
def dequantize_mlx4_g32_to_fp32(
    src: UnsafePointer[Int8, MutUntrackedOrigin],
    dst: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int):
    """mlx int4/g32 affine → fp32. Inverse of quantize_fp32_to_mlx4_g32."""
    var groups = dim // MLX_Q_GROUP
    var packed = src.bitcast[UInt32]()
    var scales = (src + (dim // 2)).bitcast[Float16]()
    var biases = (src + (dim // 2) + groups * 2).bitcast[Float16]()

    for g in range(groups):
        var base = g * MLX_Q_GROUP
        var s = scales[g].cast[DType.float32]()
        var b = biases[g].cast[DType.float32]()
        for w in range(MLX_Q_GROUP // 8):
            var word = packed[g * (MLX_Q_GROUP // 8) + w]
            for j in range(8):
                var q = (word >> UInt32(4 * j)) & UInt32(0xF)
                dst[base + w * 8 + j] = Float32(Int(q)) * s + b
