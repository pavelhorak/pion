"""Open reference for the product-key memory kernels (D11, gh #146).

Same algorithms as src/vector/pkm_kernels.mojo with the tuning removed: one
query and one row at a time (no multi-query register blocking), one portable
SIMD accumulator per dot (8-wide f32, 16-wide widening int8 — the standard
of reference/beam_1536.mojo), no prefilter in top-k, no early breaks in the
combine.

Equivalence to the tuned kernels:
  - INT8 scores: bit-exact (exact int32 dot; same `dot * row_scale * q_scale`
    epilogue).
  - FP32 scores / rescore: within float rounding — the tuned kernels already
    change their own summation order with the query batch size (NQ 1/2/4/8),
    so no fixed order is "the" answer; the differential uses a tolerance.
  - top-k / combine: the same bounded min-heap and heapsort as the tuned
    code, minus the SIMD prefilter and the early breaks — identical output.
"""
from std.memory.unsafe_pointer import Pointer

comptime _NEG_INF = Float32(-3.4028235e38)


def ref_pkm_scores_f32(
    keys: Pointer[Float32, MutUntrackedOrigin], s_rows: Int, d_pad: Int,
    q: Pointer[Float32, MutUntrackedOrigin], nq: Int,
    dst: Pointer[Float32, MutUntrackedOrigin]):
    for t in range(nq):
        for r in range(s_rows):
            var acc = SIMD[DType.float32, 8](0.0)
            for i in range(0, d_pad, 8):
                acc += keys.unsafe_offset(r * d_pad + i).load[width=8]() * q.unsafe_offset(t * d_pad + i).load[width=8]()
            dst[unsafe_offset=t * s_rows + r] = acc.reduce_add()


def ref_pkm_rescore_f32(
    keys: Pointer[Float32, MutUntrackedOrigin], d_pad: Int,
    q: Pointer[Float32, MutUntrackedOrigin],
    idx: Pointer[Int32, MutUntrackedOrigin], cnt: Int,
    dst: Pointer[Float32, MutUntrackedOrigin]):
    for t in range(cnt):
        var row = Int(idx[unsafe_offset=t]) * d_pad
        var acc = SIMD[DType.float32, 8](0.0)
        for i in range(0, d_pad, 8):
            acc += keys.unsafe_offset(row + i).load[width=8]() * q.load[width=8](i)
        dst[unsafe_offset=t] = acc.reduce_add()


def ref_pkm_scores_i8(
    keys: Pointer[Int8, MutUntrackedOrigin], s_rows: Int, d_pad: Int,
    row_scale: Pointer[Float32, MutUntrackedOrigin],
    q: Pointer[Int8, MutUntrackedOrigin],
    q_scale: Pointer[Float32, MutUntrackedOrigin], nq: Int,
    dst: Pointer[Float32, MutUntrackedOrigin]):
    for t in range(nq):
        for r in range(s_rows):
            var acc = SIMD[DType.int32, 16](0)
            for i in range(0, d_pad, 16):
                acc += (keys.unsafe_offset(r * d_pad + i).load[width=16]().cast[DType.int32]()
                        * q.unsafe_offset(t * d_pad + i).load[width=16]().cast[DType.int32]())
            dst[unsafe_offset=t * s_rows + r] = (
                acc.reduce_add().cast[DType.float32]() * row_scale[unsafe_offset=r] * q_scale[unsafe_offset=t])


@always_inline
def _sift_down_min(val: Pointer[Float32, MutUntrackedOrigin], idx: Pointer[Int32, MutUntrackedOrigin], start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            break
        if child + 1 < n and val[unsafe_offset=child + 1] < val[unsafe_offset=child]:
            child += 1
        if val[unsafe_offset=child] < val[unsafe_offset=root]:
            var tv = val[unsafe_offset=root]
            val[unsafe_offset=root] = val[unsafe_offset=child]
            val[unsafe_offset=child] = tv
            var ti = idx[unsafe_offset=root]
            idx[unsafe_offset=root] = idx[unsafe_offset=child]
            idx[unsafe_offset=child] = ti
            root = child
        else:
            break


@always_inline
def _heapify_min(val: Pointer[Float32, MutUntrackedOrigin], idx: Pointer[Int32, MutUntrackedOrigin], n: Int):
    var s = n // 2 - 1
    while s >= 0:
        _sift_down_min(val, idx, s, n)
        s -= 1


def ref_pkm_sort_desc(
    val: Pointer[Float32, MutUntrackedOrigin], idx: Pointer[Int32, MutUntrackedOrigin], n: Int):
    """Min-heapsort: the smallest sinks to the end, leaving the array descending."""
    if n <= 1:
        return
    _heapify_min(val, idx, n)
    var e = n - 1
    while e > 0:
        var tv = val[unsafe_offset=0]
        val[unsafe_offset=0] = val[unsafe_offset=e]
        val[unsafe_offset=e] = tv
        var ti = idx[unsafe_offset=0]
        idx[unsafe_offset=0] = idx[unsafe_offset=e]
        idx[unsafe_offset=e] = ti
        _sift_down_min(val, idx, 0, e)
        e -= 1


def ref_pkm_topk_max(
    scores: Pointer[Float32, MutUntrackedOrigin], n: Int, k: Int,
    out_idx: Pointer[Int32, MutUntrackedOrigin],
    out_val: Pointer[Float32, MutUntrackedOrigin]) -> Int:
    """Bounded min-heap over every score — the standard algorithm, without the
    tuned kernel's 16-wide reduce_max prefilter."""
    var kk = k if k < n else n
    if kk <= 0:
        return 0
    for i in range(kk):
        out_idx[unsafe_offset=i] = Int32(i)
        out_val[unsafe_offset=i] = scores[unsafe_offset=i]
    _heapify_min(out_val, out_idx, kk)
    for i in range(kk, n):
        var sv = scores[unsafe_offset=i]
        if sv > out_val[unsafe_offset=0]:
            out_val[unsafe_offset=0] = sv
            out_idx[unsafe_offset=0] = Int32(i)
            _sift_down_min(out_val, out_idx, 0, kk)
    ref_pkm_sort_desc(out_val, out_idx, kk)
    return kk


def ref_pkm_combine_topk(
    s0: Pointer[Float32, MutUntrackedOrigin], i0: Pointer[Int32, MutUntrackedOrigin], k0: Int,
    s1: Pointer[Float32, MutUntrackedOrigin], i1: Pointer[Int32, MutUntrackedOrigin], k1: Int,
    s_cols: Int, k: Int,
    out_id: Pointer[Int32, MutUntrackedOrigin],
    out_score: Pointer[Float32, MutUntrackedOrigin]) -> Int:
    """Bounded min-heap over the full k0 x k1 grid — no early breaks."""
    var lim = k0 * k1
    var cap = k if k < lim else lim
    if cap <= 0:
        return 0
    var cnt = 0
    for a in range(k0):
        for b in range(k1):
            var sc = s0[unsafe_offset=a] + s1[unsafe_offset=b]
            var sid = Int32(Int(i0[unsafe_offset=a]) * s_cols + Int(i1[unsafe_offset=b]))
            if cnt < cap:
                out_score[unsafe_offset=cnt] = sc
                out_id[unsafe_offset=cnt] = sid
                cnt += 1
                if cnt == cap:
                    _heapify_min(out_score, out_id, cap)
            elif sc > out_score[unsafe_offset=0]:
                out_score[unsafe_offset=0] = sc
                out_id[unsafe_offset=0] = sid
                _sift_down_min(out_score, out_id, 0, cap)
    ref_pkm_sort_desc(out_score, out_id, cnt)
    return cnt
