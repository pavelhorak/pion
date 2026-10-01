"""PKMIndex — product-key memory tables for the NEURON.PKM.* surface (gh #146).

Per-worker registry of memory-layer tables. A table of N = S² slots is stored
as two half-dim codebooks of S rows each; `src/vector/pkm_kernels.mojo` does the
arithmetic. See that file's header for why the two per-half top-k lists combine
to the **exact** global top-k — there is no ef and no recall trade here.

Why this exists (measured, gh #146 Rung P). Serving one memory-layer lookup out
of the generic `AI.KNN_LM.*` SQ8-HNSW datastore costs 5.13 ms p50 at 1M slots,
dim 896, k=32 — 18% of a 30.3 ms token budget, and 34–75% for the 2–4 layer
stacks memory-layer configs actually want. Transport was exonerated in the same
measurement (value fetch adds 0.3 ms of the 5.46); the cost is ANN search
compute. Product-key decomposition replaces the stall-bound graph walk with
2·√N·(dim/2) streaming MACs.

Storage per table
-----------------
  keys_f32   2 × S × d_pad FP32   source of truth, exact scoring
  keys_i8    2 × S × d_pad INT8   mirror for the FAST path (4× less traffic)
  key_scale  2 × S       FP32     per-row symmetric dequant scale
  vals       n_slots × vdim       FP32 or FP16 value rows, allocated lazily on
                                  the first SETVALS (a table used only for
                                  QUERY — client fetches its own values — never
                                  pays for them)

At S=1024 (1M slots), dim 896: keys are 3.67 MB FP32 + 0.92 MB INT8. The value
matrix dominates everything else (1M × 2560 FP16 = 5.1 GB) which is exactly why
`NEURON.PKM.FFN` keeps it server-side.

All query scratch is allocated at CREATE, so the query path itself allocates
nothing — no allocation on the event loop.

Per-worker, shared-nothing: a table created on worker A is invisible to worker
B. Memory-layer serving is single-stream decode (`-w 1`), which is the shape
this was priced for.

Wire commands are handled in `src/commands/pkm.mojo`.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.collections import Array
from std.math import sqrt
from std.memory import unsafe_memcpy, unsafe_memset
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc

from src.vector.pkm_kernels import (
    pkm_accumulate_f16,
    pkm_accumulate_f32,
    pkm_combine_topk,
    pkm_pad_dim,
    pkm_quantize_rows_i8,
    pkm_rescore_f32,
    pkm_scores_f32,
    pkm_scores_i8,
    pkm_softmax,
    pkm_sort_desc,
    pkm_topk_max,
)


comptime MAX_PKM_TABLES = 8
comptime PKM_NAME_INLINE = 64
comptime PKM_MAX_S = 4096          # ⇒ n_slots ≤ 16,777,216, ids fit Int32
comptime PKM_MAX_DIM = 8192
comptime PKM_MAX_VDIM = 65536
comptime PKM_MAX_K = 512
comptime PKM_MAX_NQ = 16           # query heads per request
comptime PKM_QCHUNK = 8            # queries scored per pass over the codebook

comptime PKM_VT_F32 = 0
comptime PKM_VT_F16 = 1

# FAST-path candidate width: select this many rows per half on INT8 scores,
# re-score them in FP32, then keep the top k. 4× k (floor 64) puts the
# quantization error budget far below the score gaps that decide the top-k.
comptime PKM_FAST_CAND_MULT = 4
comptime PKM_FAST_CAND_MIN = 64

comptime PKM_NEG_INF_F = Float32(-3.4028235e38)


@always_inline
def pkm_isqrt(n: Int) -> Int:
    """Integer square root, corrected — `sqrt` on a large Float64 can land off
    by one and we need S·S == n_slots to hold exactly."""
    if n <= 0:
        return 0
    var r = Int(sqrt(Float64(n)))
    while r * r > n:
        r -= 1
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


struct PKMTable(Movable, Copyable):
    """One product-key table. Copies are shallow pointer aliases — the registry
    creates then mutates in place, same convention as KNNDatastore."""

    var name: Array[UInt8, PKM_NAME_INLINE]
    var name_len: Int
    var active: Bool

    var dim: Int          # full key dim (queries are this wide)
    var half: Int         # dim // 2 — codebook row width
    var d_pad: Int        # padded row stride, multiple of 64
    var s_rows: Int       # S
    var n_slots: Int      # S²
    var vdim: Int
    var val_type: Int
    var val_rows: Int     # highest row index written + 1
    var keys_set: Int     # bit 0 = half 0 loaded, bit 1 = half 1 loaded
    var queries: Int      # served QUERY + FFN count, for INFO

    var keys_f32: Pointer[Float32, MutUntrackedOrigin]
    var keys_i8: Pointer[Int8, MutUntrackedOrigin]
    var key_scale: Pointer[Float32, MutUntrackedOrigin]
    var vals: Pointer[UInt8, MutUntrackedOrigin]

    # ── query scratch, all sized at CREATE ──
    var q_pad: Pointer[Float32, MutUntrackedOrigin]      # 2 × QCHUNK × d_pad
    var q_i8: Pointer[Int8, MutUntrackedOrigin]          # 2 × QCHUNK × d_pad
    var q_scale: Pointer[Float32, MutUntrackedOrigin]    # 2 × QCHUNK
    var scores: Pointer[Float32, MutUntrackedOrigin]     # 2 × QCHUNK × S
    var h_idx: Pointer[Int32, MutUntrackedOrigin]        # 2 × PKM_MAX_K
    var h_val: Pointer[Float32, MutUntrackedOrigin]      # 2 × PKM_MAX_K
    var ffn_w: Pointer[Float32, MutUntrackedOrigin]      # PKM_MAX_K

    def __init__(out self):
        self.name = Array[UInt8, PKM_NAME_INLINE](fill=UInt8(0))
        self.name_len = 0
        self.active = False
        self.dim = 0
        self.half = 0
        self.d_pad = 0
        self.s_rows = 0
        self.n_slots = 0
        self.vdim = 0
        self.val_type = PKM_VT_F32
        self.val_rows = 0
        self.keys_set = 0
        self.queries = 0
        self.keys_f32 = null_ptr[Float32, MutUntrackedOrigin]()
        self.keys_i8 = null_ptr[Int8, MutUntrackedOrigin]()
        self.key_scale = null_ptr[Float32, MutUntrackedOrigin]()
        self.vals = null_ptr[UInt8, MutUntrackedOrigin]()
        self.q_pad = null_ptr[Float32, MutUntrackedOrigin]()
        self.q_i8 = null_ptr[Int8, MutUntrackedOrigin]()
        self.q_scale = null_ptr[Float32, MutUntrackedOrigin]()
        self.scores = null_ptr[Float32, MutUntrackedOrigin]()
        self.h_idx = null_ptr[Int32, MutUntrackedOrigin]()
        self.h_val = null_ptr[Float32, MutUntrackedOrigin]()
        self.ffn_w = null_ptr[Float32, MutUntrackedOrigin]()

    def __copyinit__(out self, existing: Self):
        self.name = Array[UInt8, PKM_NAME_INLINE](fill=UInt8(0))
        for i in range(PKM_NAME_INLINE):
            self.name[i] = existing.name[i]
        self.name_len = existing.name_len
        self.active = existing.active
        self.dim = existing.dim
        self.half = existing.half
        self.d_pad = existing.d_pad
        self.s_rows = existing.s_rows
        self.n_slots = existing.n_slots
        self.vdim = existing.vdim
        self.val_type = existing.val_type
        self.val_rows = existing.val_rows
        self.keys_set = existing.keys_set
        self.queries = existing.queries
        self.keys_f32 = existing.keys_f32
        self.keys_i8 = existing.keys_i8
        self.key_scale = existing.key_scale
        self.vals = existing.vals
        self.q_pad = existing.q_pad
        self.q_i8 = existing.q_i8
        self.q_scale = existing.q_scale
        self.scores = existing.scores
        self.h_idx = existing.h_idx
        self.h_val = existing.h_val
        self.ffn_w = existing.ffn_w

    @always_inline
    def val_elt_bytes(self) -> Int:
        return 2 if self.val_type == PKM_VT_F16 else 4

    @always_inline
    def keys_ready(self) -> Bool:
        return self.keys_set == 3


struct PKMIndex:
    """Per-worker product-key table registry."""

    var enabled: Bool
    var tables: Array[PKMTable, MAX_PKM_TABLES]

    # Shared reply scratch — one set for the worker, not per table, so the
    # query/FFN path is zero-alloc without paying 8× for idle tables.
    var res_id: Pointer[Int32, MutUntrackedOrigin]      # PKM_MAX_NQ × PKM_MAX_K
    var res_score: Pointer[Float32, MutUntrackedOrigin]
    var pack: Pointer[UInt8, MutUntrackedOrigin]        # 8 bytes per pair
    var ffn_buf: Pointer[Float32, MutUntrackedOrigin]   # grow-only, nq × vdim
    var ffn_buf_cap: Int

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.tables = Array[PKMTable, MAX_PKM_TABLES](fill=PKMTable())
        self.res_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.res_score = null_ptr[Float32, MutUntrackedOrigin]()
        self.pack = null_ptr[UInt8, MutUntrackedOrigin]()
        self.ffn_buf = null_ptr[Float32, MutUntrackedOrigin]()
        self.ffn_buf_cap = 0
        if enabled:
            var n = PKM_MAX_NQ * PKM_MAX_K
            var _ri = alloc[Int32](n)
            self.res_id = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_ri))
            var _rs = alloc[Float32](n)
            self.res_score = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_rs))
            var _pk = alloc[UInt8](n * 8)
            self.pack = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_pk))

    @always_inline
    def ensure_ffn_buf(mut self, floats_needed: Int) -> Bool:
        """Grow-only activation buffer for NEURON.PKM.FFN. Reallocs only when a
        request is wider than anything seen before; steady state is zero-alloc."""
        if floats_needed <= self.ffn_buf_cap:
            return True
        if is_not_null(self.ffn_buf):
            self.ffn_buf.unsafe_free()
        var p = alloc[Float32](floats_needed)
        if is_null(p):
            self.ffn_buf_cap = 0
            self.ffn_buf = null_ptr[Float32, MutUntrackedOrigin]()
            return False
        self.ffn_buf = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(p))
        self.ffn_buf_cap = floats_needed
        return True

    @always_inline
    def _name_eq(self, slot: Int, name_ptr: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Bool:
        if not self.tables[slot].active:
            return False
        if self.tables[slot].name_len != name_len:
            return False
        for i in range(name_len):
            if self.tables[slot].name[i] != name_ptr[unsafe_offset=i]:
                return False
        return True

    def find(self, name_ptr: Pointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
        """Return slot index, or -1 if not found."""
        for s in range(MAX_PKM_TABLES):
            if self._name_eq(s, name_ptr, name_len):
                return s
        return -1

    def create(
        mut self,
        name_ptr: Pointer[UInt8, MutUntrackedOrigin],
        name_len: Int,
        dim: Int,
        n_slots: Int,
        vdim: Int,
        val_type: Int,
    ) -> Int:
        """Allocate a table. Returns slot idx, or a negative reason code:
        -1 registry full / duplicate name, -2 bad name, -3 bad dim,
        -4 n_slots is not a perfect square or out of range, -5 bad vdim."""
        if name_len <= 0 or name_len > PKM_NAME_INLINE:
            return -2
        if dim <= 1 or dim > PKM_MAX_DIM or (dim % 2) != 0:
            return -3
        var s_rows = pkm_isqrt(n_slots)
        if s_rows <= 0 or s_rows > PKM_MAX_S or s_rows * s_rows != n_slots:
            return -4
        if vdim < 0 or vdim > PKM_MAX_VDIM:
            return -5
        if self.find(name_ptr, name_len) >= 0:
            return -1

        for s in range(MAX_PKM_TABLES):
            if self.tables[s].active:
                continue

            var half = dim // 2
            var d_pad = pkm_pad_dim(half)

            self.tables[s].active = True
            self.tables[s].name_len = name_len
            for i in range(name_len):
                self.tables[s].name[i] = name_ptr[unsafe_offset=i]
            self.tables[s].dim = dim
            self.tables[s].half = half
            self.tables[s].d_pad = d_pad
            self.tables[s].s_rows = s_rows
            self.tables[s].n_slots = n_slots
            self.tables[s].vdim = vdim
            self.tables[s].val_type = val_type
            self.tables[s].val_rows = 0
            self.tables[s].keys_set = 0
            self.tables[s].queries = 0

            var kn = 2 * s_rows * d_pad
            var _kf = alloc[Float32](kn)
            self.tables[s].keys_f32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_kf))
            unsafe_memset(self.tables[s].keys_f32.unsafe_bitcast[UInt8](), 0, kn * 4)

            var _ki = alloc[Int8](kn)
            self.tables[s].keys_i8 = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_ki))
            unsafe_memset(self.tables[s].keys_i8.unsafe_bitcast[UInt8](), 0, kn)

            var _ks = alloc[Float32](2 * s_rows)
            self.tables[s].key_scale = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_ks))
            unsafe_memset(self.tables[s].key_scale.unsafe_bitcast[UInt8](), 0, 2 * s_rows * 4)

            # Query scratch — sized once so the query path never allocates.
            var qn = 2 * PKM_QCHUNK * d_pad
            var _qp = alloc[Float32](qn)
            self.tables[s].q_pad = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_qp))
            unsafe_memset(self.tables[s].q_pad.unsafe_bitcast[UInt8](), 0, qn * 4)

            var _qi = alloc[Int8](qn)
            self.tables[s].q_i8 = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_qi))
            unsafe_memset(self.tables[s].q_i8.unsafe_bitcast[UInt8](), 0, qn)

            var _qs = alloc[Float32](2 * PKM_QCHUNK)
            self.tables[s].q_scale = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_qs))

            var _sc = alloc[Float32](2 * PKM_QCHUNK * s_rows)
            self.tables[s].scores = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_sc))

            var _hi = alloc[Int32](2 * PKM_MAX_K)
            self.tables[s].h_idx = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_hi))
            var _hv = alloc[Float32](2 * PKM_MAX_K)
            self.tables[s].h_val = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_hv))
            var _fw = alloc[Float32](PKM_MAX_K)
            self.tables[s].ffn_w = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_fw))

            return s
        return -1

    def drop(
        mut self,
        name_ptr: Pointer[UInt8, MutUntrackedOrigin],
        name_len: Int,
    ) -> Bool:
        var s = self.find(name_ptr, name_len)
        if s < 0:
            return False
        if is_not_null(self.tables[s].keys_f32):
            self.tables[s].keys_f32.unsafe_free()
        if is_not_null(self.tables[s].keys_i8):
            self.tables[s].keys_i8.unsafe_free()
        if is_not_null(self.tables[s].key_scale):
            self.tables[s].key_scale.unsafe_free()
        if is_not_null(self.tables[s].vals):
            self.tables[s].vals.unsafe_free()
        if is_not_null(self.tables[s].q_pad):
            self.tables[s].q_pad.unsafe_free()
        if is_not_null(self.tables[s].q_i8):
            self.tables[s].q_i8.unsafe_free()
        if is_not_null(self.tables[s].q_scale):
            self.tables[s].q_scale.unsafe_free()
        if is_not_null(self.tables[s].scores):
            self.tables[s].scores.unsafe_free()
        if is_not_null(self.tables[s].h_idx):
            self.tables[s].h_idx.unsafe_free()
        if is_not_null(self.tables[s].h_val):
            self.tables[s].h_val.unsafe_free()
        if is_not_null(self.tables[s].ffn_w):
            self.tables[s].ffn_w.unsafe_free()
        self.tables[s] = PKMTable()
        return True

    # ── ingest ───────────────────────────────────────────────────────────

    def set_keys(
        mut self,
        slot: Int,
        half: Int,
        src: Pointer[Float32, MutUntrackedOrigin],
    ):
        """Load codebook `half` (0 or 1): S × (dim/2) FP32, row-major.

        Rows are copied into the padded stride and the INT8 mirror is rebuilt
        for that half. Caller has already validated the blob length.
        """
        var s_rows = self.tables[slot].s_rows
        var hd = self.tables[slot].half
        var d_pad = self.tables[slot].d_pad
        var base = half * s_rows * d_pad
        var dst = self.tables[slot].keys_f32.unsafe_offset(base)
        for r in range(s_rows):
            unsafe_memcpy(dest=(dst.unsafe_offset(r * d_pad)).unsafe_bitcast[UInt8](),
                   src=(src.unsafe_offset(r * hd)).unsafe_bitcast[UInt8](), count=hd * 4)
        pkm_quantize_rows_i8(
            dst,
            s_rows,
            hd,
            d_pad,
            self.tables[slot].keys_i8.unsafe_offset(base),
            self.tables[slot].key_scale.unsafe_offset(half * s_rows),
        )
        self.tables[slot].keys_set = self.tables[slot].keys_set | (1 << half)

    def ensure_vals(mut self, slot: Int) -> Bool:
        """Allocate the value matrix on first SETVALS. At 1M × 2560 FP16 this
        is 5.1 GB, so a QUERY-only table never pays it."""
        if is_not_null(self.tables[slot].vals):
            return True
        var vdim = self.tables[slot].vdim
        if vdim <= 0:
            return False
        var total = self.tables[slot].n_slots * vdim * self.tables[slot].val_elt_bytes()
        var _v = alloc[UInt8](total)
        if is_null(_v):
            return False
        self.tables[slot].vals = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_v))
        unsafe_memset(self.tables[slot].vals, 0, total)
        return True

    def set_vals(
        mut self,
        slot: Int,
        off: Int,
        n: Int,
        src: Pointer[UInt8, MutUntrackedOrigin],
    ):
        """Write `n` value rows starting at slot `off`. Caller validated bounds."""
        var stride = self.tables[slot].vdim * self.tables[slot].val_elt_bytes()
        unsafe_memcpy(dest=self.tables[slot].vals.unsafe_offset(off * stride), src=src, count=n * stride)
        if off + n > self.tables[slot].val_rows:
            self.tables[slot].val_rows = off + n

    # ── query ────────────────────────────────────────────────────────────

    @always_inline
    def _stage_queries(mut self, slot: Int, q: Pointer[Float32, MutUntrackedOrigin], base: Int, cnt: Int):
        """Split `cnt` queries into the two zero-padded half-dim scratch rows."""
        var hd = self.tables[slot].half
        var dim = self.tables[slot].dim
        var d_pad = self.tables[slot].d_pad
        for h in range(2):
            for t in range(cnt):
                var dst = self.tables[slot].q_pad.unsafe_offset((h * PKM_QCHUNK + t) * d_pad)
                unsafe_memcpy(dest=dst.unsafe_bitcast[UInt8](),
                       src=(q.unsafe_offset((base + t) * dim).unsafe_offset(h * hd)).unsafe_bitcast[UInt8](), count=hd * 4)
                for c in range(hd, d_pad):
                    dst[unsafe_offset=c] = 0.0

    @always_inline
    def _half_topk(mut self, slot: Int, h: Int, t: Int, k: Int, fast: Bool) -> Int:
        """Top-k rows of codebook `h` for staged query `t`, written descending
        into `h_idx`/`h_val` at offset h·PKM_MAX_K. Returns the count.

        FAST first selects a widened candidate set on INT8 scores, then
        re-scores those rows in FP32 and re-sorts — the selection is INT8 but
        every returned score is exact, so the combine stage stays exact too."""
        var s_rows = self.tables[slot].s_rows
        var d_pad = self.tables[slot].d_pad
        var srow = self.tables[slot].scores.unsafe_offset((h * PKM_QCHUNK + t) * s_rows)
        var oi = self.tables[slot].h_idx.unsafe_offset(h * PKM_MAX_K)
        var ov = self.tables[slot].h_val.unsafe_offset(h * PKM_MAX_K)

        if not fast:
            return pkm_topk_max(srow, s_rows, k, oi, ov)

        var cw = k * PKM_FAST_CAND_MULT
        if cw < PKM_FAST_CAND_MIN:
            cw = PKM_FAST_CAND_MIN
        if cw > PKM_MAX_K:
            cw = PKM_MAX_K
        if cw > s_rows:
            cw = s_rows
        var got = pkm_topk_max(srow, s_rows, cw, oi, ov)
        pkm_rescore_f32(
            self.tables[slot].keys_f32.unsafe_offset(h * s_rows * d_pad),
            d_pad,
            self.tables[slot].q_pad.unsafe_offset((h * PKM_QCHUNK + t) * d_pad),
            oi,
            got,
            ov,
        )
        pkm_sort_desc(ov, oi, got)
        return k if k < got else got

    def query(
        mut self,
        slot: Int,
        k: Int,
        q: Pointer[Float32, MutUntrackedOrigin],
        nq: Int,
        fast: Bool,
    ) -> Int:
        """Exact top-k over all n_slots for each of `nq` queries.

        Writes nq × k pairs into `self.res_id` / `self.res_score`, descending
        per query, padded with (-1, -inf) when a table has fewer than k
        reachable slots. Returns nq.
        """
        var out_id = self.res_id
        var out_score = self.res_score
        var s_rows = self.tables[slot].s_rows
        var d_pad = self.tables[slot].d_pad

        var base = 0
        while base < nq:
            var cnt = nq - base
            if cnt > PKM_QCHUNK:
                cnt = PKM_QCHUNK
            self._stage_queries(slot, q, base, cnt)

            # One pass over each codebook scores the whole chunk — this is the
            # multi-head win: H heads cost one codebook read, not H.
            for h in range(2):
                var kf = self.tables[slot].keys_f32.unsafe_offset(h * s_rows * d_pad)
                var qp = self.tables[slot].q_pad.unsafe_offset(h * PKM_QCHUNK * d_pad)
                var sc = self.tables[slot].scores.unsafe_offset(h * PKM_QCHUNK * s_rows)
                if fast:
                    var qi = self.tables[slot].q_i8.unsafe_offset(h * PKM_QCHUNK * d_pad)
                    var qs = self.tables[slot].q_scale.unsafe_offset(h * PKM_QCHUNK)
                    pkm_quantize_rows_i8(qp, cnt, self.tables[slot].half, d_pad, qi, qs)
                    pkm_scores_i8(
                        self.tables[slot].keys_i8.unsafe_offset(h * s_rows * d_pad),
                        s_rows,
                        d_pad,
                        self.tables[slot].key_scale.unsafe_offset(h * s_rows),
                        qi,
                        qs,
                        cnt,
                        sc,
                    )
                else:
                    pkm_scores_f32(kf, s_rows, d_pad, qp, cnt, sc)

            for t in range(cnt):
                var k0 = self._half_topk(slot, 0, t, k, fast)
                var k1 = self._half_topk(slot, 1, t, k, fast)
                var op = (base + t) * k
                var got = pkm_combine_topk(
                    self.tables[slot].h_val,
                    self.tables[slot].h_idx,
                    k0,
                    self.tables[slot].h_val.unsafe_offset(PKM_MAX_K),
                    self.tables[slot].h_idx.unsafe_offset(PKM_MAX_K),
                    k1,
                    s_rows,
                    k,
                    out_id.unsafe_offset(op),
                    out_score.unsafe_offset(op),
                )
                for j in range(got, k):
                    out_id[unsafe_offset=op + j] = Int32(-1)
                    out_score[unsafe_offset=op + j] = PKM_NEG_INF_F

            base += cnt

        self.tables[slot].queries += nq
        return nq

    def ffn(
        mut self,
        slot: Int,
        k: Int,
        q: Pointer[Float32, MutUntrackedOrigin],
        nq: Int,
        fast: Bool,
        temperature: Float32,
    ) -> Int:
        """Fused lookup + softmax-weighted value read into `self.ffn_buf`:
        out[t] = Σ softmax(s)·V[id].

        Only activations cross the wire (nq × vdim FP32), so the value-row
        bandwidth — the 5.1 GB matrix at 1M × 2560 FP16 — stays server-side.
        Returns nq, -1 if the table has no value matrix, -2 on scratch OOM.
        """
        if is_null(self.tables[slot].vals):
            return -1
        var vdim = self.tables[slot].vdim
        if not self.ensure_ffn_buf(nq * vdim):
            return -2
        var scratch_id = self.res_id
        var scratch_score = self.res_score
        var dst = self.ffn_buf
        _ = self.query(slot, k, q, nq, fast)

        for t in range(nq):
            var op = t * k
            # Drop the (-1, -inf) padding slots before softmax.
            var live = 0
            for j in range(k):
                if scratch_id[unsafe_offset=op + j] >= 0:
                    self.tables[slot].ffn_w[unsafe_offset=live] = scratch_score[unsafe_offset=op + j]
                    scratch_id[unsafe_offset=op + live] = scratch_id[unsafe_offset=op + j]
                    live += 1
            if live == 0:
                unsafe_memset((dst.unsafe_offset(t * vdim)).unsafe_bitcast[UInt8](), 0, vdim * 4)
                continue
            pkm_softmax(self.tables[slot].ffn_w, live, temperature)
            if self.tables[slot].val_type == PKM_VT_F16:
                pkm_accumulate_f16(
                    self.tables[slot].vals.unsafe_bitcast[Float16](),
                    vdim,
                    scratch_id.unsafe_offset(op),
                    self.tables[slot].ffn_w,
                    live,
                    dst.unsafe_offset(t * vdim),
                )
            else:
                pkm_accumulate_f32(
                    self.tables[slot].vals.unsafe_bitcast[Float32](),
                    vdim,
                    scratch_id.unsafe_offset(op),
                    self.tables[slot].ffn_w,
                    live,
                    dst.unsafe_offset(t * vdim),
                )
        return nq
