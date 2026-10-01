"""PKM kernels — product-key memory exact top-k (gh #146).

A product-key memory table (Lample et al. 2019, "Large Memory Layers with
Product Keys"; Meta 2024, "Memory Layers at Scale") stores N = S² slots whose
keys are the Cartesian product of two *half-dim* codebooks C0, C1 ∈ R^(S × D)
with D = dim/2:

    key[i·S + j] = concat(C0[i], C1[j])
    score(q, i·S + j) = ⟨q0, C0[i]⟩ + ⟨q1, C1[j]⟩          (q = concat(q0, q1))

**Why this is exact, not approximate.** If (i, j) is in the true top-k of the
sum, then i must be in the top-k of `s0` — otherwise there exist k indices i′
with s0[i′] > s0[i], and each pair (i′, j) outscores (i, j), contradicting
(i, j) being top-k. Symmetrically for j. So scoring only the 2S half-dots and
then combining the two per-half top-k lists yields the **exact** global top-k
over all N slots. No ef, no recall trade.

**Cost.** 2·S·D = 2·√N·(dim/2) MACs plus a ≤ k² combine, against HNSW's
memory-stall-bound graph traversal. At N = 1M, dim = 896: ~0.9M MACs,
~3.7 MB of FP32 key traffic (0.9 MB in the INT8 mirror).

Kernel set
----------
`pkm_scores_f32`      FP32 codebook GEMV/GEMM, NQ ∈ {1,2,4,8} specializations.
`pkm_scores_i8`       INT8 mirror via the platform-dispatched `sdot_int8`
                      (NEON SDOT / AVX512-VNNI / SSE decomposition).
`pkm_topk_max`        Bounded min-heap top-k with a SIMD `reduce_max` prefilter
                      — one 16-wide compare rejects a whole block of rows.
`pkm_combine_topk`    Exact k₀×k₁ combine with double early-break (both input
                      lists are sorted descending, so the scan prunes hard).
`pkm_rescore_f32`     FP32 re-score of an INT8-selected candidate set.
`pkm_softmax`         In-place temperature softmax over the k winners.
`pkm_accumulate_f32`  Σ wₜ·V[idₜ] gather-FMA (FP32 and FP16 value rows).
`pkm_accumulate_f16`

**Multi-query (multi-head) is the structural win.** Memory layers run several
query heads per layer against the *same* codebooks. `pkm_scores_*` keeps one
key row in registers and FMAs it against every query, so H heads cost one pass
over the codebook instead of H — turning a bandwidth-bound GEMV into a
compute-bound GEMM. Heads within a layer are parallel; only layers are
sequential.

**Padding contract.** Every row (codebook rows and query rows alike) is padded
to `pkm_pad_dim(D)` — a multiple of 64 elements — with zeros. 64 floats is a
multiple of every FP32 unroll step used here, and 64 int8 bytes is a multiple
of the 16-byte SDOT step, so no kernel has a scalar tail and the same stride
serves both storages. Zero padding contributes 0 to every dot product.
"""

from std.collections import Array
from std.math import exp, fma
from std.memory import unsafe_memset
from std.memory.unsafe_pointer import Pointer

from std.ffi import external_call
from src.vector.vector_abi import HELD_VECTOR
from src.vector.reference.pkm_ref import (
    ref_pkm_scores_f32, ref_pkm_rescore_f32, ref_pkm_scores_i8,
    ref_pkm_topk_max, ref_pkm_sort_desc, ref_pkm_combine_topk,
)

# D11: the six scoring / selection routines below are closed. Under
# -D PION_HELD_VECTOR (what `pixi run build` passes) each one is a single C-ABI
# call into libpion_vector (vendor/pion-vector/); every other build runs the
# open reference in src/vector/reference/pkm_ref.mojo — same results, bit for
# bit, slower (doc/vector_engine.md § Open build). Padding, quantization,
# softmax and accumulate stay open here.


comptime PKM_DPAD_MULT = 64          # row padding granularity (elements)
comptime PKM_SW = 8                  # FP32 SIMD lane count in the score kernels
comptime PKM_PREFILTER_W = 16        # lanes per top-k prefilter block
comptime PKM_NEG_INF = Float32(-3.4028235e38)


@always_inline
def pkm_pad_dim(d: Int) -> Int:
    """Round a half-dim up to the row stride every kernel here assumes."""
    return ((d + PKM_DPAD_MULT - 1) // PKM_DPAD_MULT) * PKM_DPAD_MULT


# ─────────────────────────────────────────────────────────────────────────
# FP32 codebook scoring
# ─────────────────────────────────────────────────────────────────────────


@always_inline
def pkm_scores_f32(
    keys: Pointer[Float32, MutUntrackedOrigin],
    s_rows: Int,
    d_pad: Int,
    q: Pointer[Float32, MutUntrackedOrigin],
    nq: Int,
    dst: Pointer[Float32, MutUntrackedOrigin],
):
    """dst[t·s_rows + r] = ⟨q[t], keys[r]⟩ for t < nq, r < s_rows.

    `q` is nq × d_pad, `keys` is s_rows × d_pad, both zero-padded. Query batches
    are peeled 8 → 4 → 2 → 1 so any nq gets the widest specialization that fits.
    """
    comptime if HELD_VECTOR:
        external_call["pion_v_pkm_scores_f32", NoneType](keys, s_rows, d_pad, q, nq, dst)
    else:
        ref_pkm_scores_f32(keys, s_rows, d_pad, q, nq, dst)


@always_inline
def pkm_rescore_f32(
    keys: Pointer[Float32, MutUntrackedOrigin],
    d_pad: Int,
    q: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    cnt: Int,
    dst: Pointer[Float32, MutUntrackedOrigin],
):
    """Exact FP32 re-score of `cnt` gathered rows — the INT8 path's refine step."""
    comptime if HELD_VECTOR:
        external_call["pion_v_pkm_rescore_f32", NoneType](keys, d_pad, q, idx, cnt, dst)
    else:
        ref_pkm_rescore_f32(keys, d_pad, q, idx, cnt, dst)


# ─────────────────────────────────────────────────────────────────────────
# INT8 codebook scoring (per-row symmetric scale, FP32 query scale)
# ─────────────────────────────────────────────────────────────────────────


@always_inline
def pkm_scores_i8(
    keys: Pointer[Int8, MutUntrackedOrigin],
    s_rows: Int,
    d_pad: Int,
    row_scale: Pointer[Float32, MutUntrackedOrigin],
    q: Pointer[Int8, MutUntrackedOrigin],
    q_scale: Pointer[Float32, MutUntrackedOrigin],
    nq: Int,
    dst: Pointer[Float32, MutUntrackedOrigin],
):
    """INT8 mirror of `pkm_scores_f32` — 4× less key traffic, ~4× less latency.

    Dequant is folded into the epilogue: `dot_i32 · row_scale[r] · q_scale[t]`.
    Used to *select* a widened candidate set; `pkm_rescore_f32` then restores
    exactness on the survivors.
    """
    comptime if HELD_VECTOR:
        external_call["pion_v_pkm_scores_i8", NoneType](keys, s_rows, d_pad, row_scale, q, q_scale, nq, dst)
    else:
        ref_pkm_scores_i8(keys, s_rows, d_pad, row_scale, q, q_scale, nq, dst)


@always_inline
def pkm_quantize_rows_i8(
    src: Pointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    d: Int,
    d_pad: Int,
    dst: Pointer[Int8, MutUntrackedOrigin],
    dst_scale: Pointer[Float32, MutUntrackedOrigin],
):
    """Per-row symmetric INT8 quantization: `scale[r] = max|src[r]| / 127`.

    Symmetric (rather than min/range affine) keeps the dequant a single
    multiply, so `sdot_int8`'s int32 accumulator needs no offset correction
    term. Padding columns are zeroed. Off the hot path — runs once per SETKEYS.
    """
    for r in range(n_rows):
        var sp = src.unsafe_offset(r * d_pad)
        var m = Float32(0.0)
        for c in range(d):
            var a = sp[unsafe_offset=c]
            if a < 0.0:
                a = -a
            if a > m:
                m = a
        var scale = m / 127.0
        if m == 0.0:
            scale = 1.0
        var inv = 1.0 / scale
        var dp = dst.unsafe_offset(r * d_pad)
        for c in range(d):
            var qv = sp[unsafe_offset=c] * inv
            var qi = Int(qv + 0.5) if qv >= 0.0 else Int(qv - 0.5)
            if qi > 127:
                qi = 127
            if qi < -127:
                qi = -127
            dp[unsafe_offset=c] = Int8(qi)
        for c in range(d, d_pad):
            dp[unsafe_offset=c] = Int8(0)
        dst_scale[unsafe_offset=r] = scale


# ─────────────────────────────────────────────────────────────────────────
# Bounded top-k (max) with SIMD prefilter
# ─────────────────────────────────────────────────────────────────────────


@always_inline
def pkm_sort_desc(
    val: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin],
    n: Int,
):
    """Min-heapsort in place — the smallest sinks to the end, so the array
    comes dst sorted *descending*, which is the order every consumer wants."""
    comptime if HELD_VECTOR:
        external_call["pion_v_pkm_sort_desc", NoneType](val, idx, n)
    else:
        ref_pkm_sort_desc(val, idx, n)


@always_inline
def pkm_topk_max(
    scores: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    k: Int,
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    out_val: Pointer[Float32, MutUntrackedOrigin],
) -> Int:
    """Top-k *largest* of `scores[0:n]`, written descending. Returns min(k, n).

    `out_idx`/`out_val` double as the heap storage — no scratch allocation. The
    heap root is the current admission threshold, so a single 16-wide
    `reduce_max` discards a whole block of rows without touching the heap; with
    k ≪ n almost every block takes that branch.
    """
    comptime if HELD_VECTOR:
        return external_call["pion_v_pkm_topk_max", Int](scores, n, k, out_idx, out_val)
    else:
        return ref_pkm_topk_max(scores, n, k, out_idx, out_val)


# ─────────────────────────────────────────────────────────────────────────
# Exact product-key combine
# ─────────────────────────────────────────────────────────────────────────

@always_inline
def pkm_combine_topk(
    s0: Pointer[Float32, MutUntrackedOrigin],
    i0: Pointer[Int32, MutUntrackedOrigin],
    k0: Int,
    s1: Pointer[Float32, MutUntrackedOrigin],
    i1: Pointer[Int32, MutUntrackedOrigin],
    k1: Int,
    s_cols: Int,
    k: Int,
    out_id: Pointer[Int32, MutUntrackedOrigin],
    out_score: Pointer[Float32, MutUntrackedOrigin],
) -> Int:
    """Top-k of `{s0[a] + s1[b]}` over the k0×k1 grid, slot id = i0[a]·s_cols + i1[b].

    Both inputs must be sorted descending (they come straight dst of
    `pkm_topk_max`). Two early breaks make the scan pay only for what it keeps:
    the inner one fires once `s0[a] + s1[b]` drops under the k-th best (all
    later b are smaller), the outer once `s0[a] + s1[0]` does (all later a are
    smaller). Results are written descending.
    """
    comptime if HELD_VECTOR:
        return external_call["pion_v_pkm_combine_topk", Int](s0, i0, k0, s1, i1, k1, s_cols, k, out_id, out_score)
    else:
        return ref_pkm_combine_topk(s0, i0, k0, s1, i1, k1, s_cols, k, out_id, out_score)


# ─────────────────────────────────────────────────────────────────────────
# Fused FFN epilogue: softmax(scores) · value rows
# ─────────────────────────────────────────────────────────────────────────

@always_inline
def pkm_softmax(
    v: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    temperature: Float32,
):
    """In-place max-shifted softmax with temperature. n is k (≤ 4096)."""
    if n <= 0:
        return
    var mx = v[unsafe_offset=0]
    for i in range(1, n):
        if v[unsafe_offset=i] > mx:
            mx = v[unsafe_offset=i]
    var inv_t = Float32(1.0) / temperature
    var total = Float32(0.0)
    for i in range(n):
        var e = exp((v[unsafe_offset=i] - mx) * inv_t)
        v[unsafe_offset=i] = e
        total += e
    if total > 0.0:
        var inv = Float32(1.0) / total
        for i in range(n):
            v[unsafe_offset=i] = v[unsafe_offset=i] * inv


@always_inline
def pkm_accumulate_f32(
    vals: Pointer[Float32, MutUntrackedOrigin],
    vdim: Int,
    ids: Pointer[Int32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    k: Int,
    dst: Pointer[Float32, MutUntrackedOrigin],
):
    """dst[0:vdim] = Σₜ w[t] · vals[ids[t]]. FP32 value rows."""
    unsafe_memset(dst.unsafe_bitcast[UInt8](), 0, vdim * 4)
    for t in range(k):
        var wv = SIMD[DType.float32, PKM_PREFILTER_W](w[unsafe_offset=t])
        var vp = vals.unsafe_offset(Int(ids[unsafe_offset=t]) * vdim)
        var i = 0
        while i + PKM_PREFILTER_W <= vdim:
            dst.store[width=PKM_PREFILTER_W](
                i,
                fma(
                    vp.load[width=PKM_PREFILTER_W](i),
                    wv,
                    dst.load[width=PKM_PREFILTER_W](i),
                ),
            )
            i += PKM_PREFILTER_W
        while i < vdim:
            dst[unsafe_offset=i] = fma(vp[unsafe_offset=i], w[unsafe_offset=t], dst[unsafe_offset=i])
            i += 1


@always_inline
def pkm_accumulate_f16(
    vals: Pointer[Float16, MutUntrackedOrigin],
    vdim: Int,
    ids: Pointer[Int32, MutUntrackedOrigin],
    w: Pointer[Float32, MutUntrackedOrigin],
    k: Int,
    dst: Pointer[Float32, MutUntrackedOrigin],
):
    """dst[0:vdim] = Σₜ w[t] · vals[ids[t]]. FP16 value rows, FP32 accumulation.

    FP16 is the realistic memory-layer format — at vdim=2560 a row is 5 KB,
    the size the gh #146 measurements assumed — and halving value-row traffic
    is what keeps the fused FFN cheaper than shipping k rows over the wire.
    """
    unsafe_memset(dst.unsafe_bitcast[UInt8](), 0, vdim * 4)
    for t in range(k):
        var wv = SIMD[DType.float32, PKM_PREFILTER_W](w[unsafe_offset=t])
        var vp = vals.unsafe_offset(Int(ids[unsafe_offset=t]) * vdim)
        var i = 0
        while i + PKM_PREFILTER_W <= vdim:
            dst.store[width=PKM_PREFILTER_W](
                i,
                fma(
                    vp.load[width=PKM_PREFILTER_W](i).cast[DType.float32](),
                    wv,
                    dst.load[width=PKM_PREFILTER_W](i),
                ),
            )
            i += PKM_PREFILTER_W
        while i < vdim:
            dst[unsafe_offset=i] = fma(vp[unsafe_offset=i].cast[DType.float32](), w[unsafe_offset=t], dst[unsafe_offset=i])
            i += 1
