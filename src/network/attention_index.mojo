"""AttentionIndex — per-session, per-layer HNSW index for externalized attention.

Phase 3 of M14 (Externalized Attention). Indexes individual token KV pairs
per transformer layer. During inference, query vectors are matched against
stored tokens via HNSW, returning top-k (key, value) pairs.

Architecture:
  - Each session has up to MAX_ATTEND_LAYERS HNSW indexes
  - Each index stores concatenated KV head vectors (dim = head_dim * num_kv_heads * 2)
  - For Llama 70B GQA: 128 head_dim * 8 kv_heads = 1024d key + 1024d value
  - One HNSW lookup per layer returns all heads for top-k tokens

Memory per 128K context, 40 layers, INT8 1024d:
  ~68 MB per layer * 40 = 2.7 GB per session (fits single Pion worker)
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import List

from src.vector.hnsw import HNSWGraph
from src.vector.kernels import (
    quantize_fp32_to_block_int4,
    dequantize_block_int4_to_fp32,
    quantize_fp32_to_block_int3,
    dequantize_block_int3_to_fp32,
    quantize_fp32_to_block_int2,
    dequantize_block_int2_to_fp32,
    fused_topk_dequant_v_turbo4,
    fused_topk_dequant_v_turbo3,
    fused_topk_dequant_v_turbo2,
    fused_topk_dequant_v_fp8,
    quantize_fp32_to_block_fp8,
    dequantize_block_fp8_to_fp32,
    quantize_fp32_to_bf16_rope_fp8_body,
    dequantize_bf16_rope_fp8_body_to_fp32,
    l2_normalize_fp32,
)
from std.math import sqrt

# Maximum sessions with attention indexes
comptime MAX_ATTEND_SESSIONS = 32
# Maximum layers per session to externalize
comptime MAX_ATTEND_LAYERS = 80
# Maximum tokens per layer index (128K + headroom)
# HNSW capacity for graph construction. Set to actual token count in finalize_layer.
# Value/staging buffers now allocate to actual size, not this max.
comptime MAX_ATTEND_TOKENS = 150000
# Initial staging buffer size (grows if needed). Smaller = less memory waste.
comptime INITIAL_STAGING_TOKENS = 2000
# Default key dimension (1024 = 128 head_dim * 8 kv_heads for Llama 70B)
comptime DEFAULT_ATTEND_DIM = 1024


# gh #391: search width for ATTEND.QUERY (RESP and the binary lane). At 64 a
# stored 128-d key found itself 99.2% of the time at 128K tokens; at 128,
# 1000/1000, and a random query's best hit is in the true L2 top-10 97% of
# the time. Costs ~0.2 ms per query at 128K.
comptime ATTEND_QUERY_EF = 128

# M4: Quantization formats for K and V.
# Stored as UInt8 inside AttendSessionMeta + per-layer format array.
comptime QFMT_INT8   = UInt8(0)   # baseline, 1.0 B/val
comptime QFMT_TURBO4 = UInt8(1)   # Block-INT4 (dim must be multiple of 32), 0.5625 B/val
comptime QFMT_FP16   = UInt8(2)   # boundary-layer high-precision V (2.0 B/val)
comptime QFMT_TURBO3 = UInt8(3)   # Block-INT3 (dim must be multiple of 32), 0.4375 B/val
comptime QFMT_TURBO2 = UInt8(4)   # Block-INT2 (dim must be multiple of 32), 0.3125 B/val
# A2 (gh #30, gh #39): FP8 (E4M3) and BF16/FP8 hybrid for V4-class attention.
# Tags MUST stay aligned with src/network/v_store.mojo VFMT_FP8/VFMT_BF16_ROPE_FP8
# so a snapshot or wire encoding can roundtrip across both stores.
comptime QFMT_FP8           = UInt8(5)   # E4M3 per-block-of-32, ~1.06 B/val (4B norm + (dim/32)×34)
comptime QFMT_BF16_ROPE_FP8 = UInt8(6)   # rope_dim×2 + ((dim-rope_dim)/32)×34 — V4 §2.3.4 layout


struct AttendSessionMeta(TrivialRegisterPassable):
    """Metadata for an attention session."""
    var active: Bool
    var session_hash: UInt64
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: Int
    var key_dim: Int            # dimension of concatenated keys (e.g., 1024)
    var value_dim: Int          # dimension of concatenated values (e.g., 1024)
    var num_layers: Int         # how many layers have been externalized
    var tokens_per_layer: Int   # longest layer's token count (ATTEND.INFO); per-layer counts: AttentionIndex.layer_tokens
    # M4: Asymmetric quantization config.
    # k_format is currently informational; K storage goes through the HNSW
    # quantizer, which always uses INT8 today. v_format actually switches
    # the per-layer V format. boundary_layers_n > 0 protects first-n + last-n
    # layers by using boundary_v_format instead.
    var k_format: UInt8
    var v_format: UInt8
    var boundary_layers_n: Int
    var boundary_v_format: UInt8
    # A2 (gh #39): per-session rope_dim — only meaningful when v_format or
    # boundary_v_format is QFMT_BF16_ROPE_FP8. ATTEND.* layers within one
    # session share an architectural rope_dim (V4: same across all CSA/HCA
    # layers), so per-session storage is sufficient — no per-layer array.
    var rope_dim: Int

    def __init__(out self):
        self.active = False
        self.session_hash = 0
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.key_dim = DEFAULT_ATTEND_DIM
        self.value_dim = DEFAULT_ATTEND_DIM
        self.num_layers = 0
        self.tokens_per_layer = 0
        self.k_format = QFMT_INT8
        self.v_format = QFMT_INT8
        self.boundary_layers_n = 0
        self.boundary_v_format = QFMT_INT8
        self.rope_dim = 0


struct AttentionIndex(Movable):
    """Per-worker attention index manager.

    Manages multiple sessions, each with per-layer HNSW indexes for token KV pairs.
    """
    var sessions: Pointer[AttendSessionMeta, MutUntrackedOrigin]

    # Per-layer HNSW indexes: indexes[session_idx * MAX_ATTEND_LAYERS + layer_id]
    var indexes: Pointer[Pointer[HNSWGraph, MutUntrackedOrigin], MutUntrackedOrigin]

    # Per-layer value storage as INT8 quantized: values_int8[slot][token_id * val_dim]
    # Quantization: per-layer min/max → scale to [-127,127]. Dequantized on query.
    # 4x smaller than FP32 (1 byte/dim vs 4 bytes/dim).
    var values_int8: Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin]
    var values_scale: Pointer[Float32, MutUntrackedOrigin]  # [slots] per-layer scale factor
    var values_min: Pointer[Float32, MutUntrackedOrigin]    # [slots] per-layer min value

    # Legacy FP32 values (kept for backward compat; NULL when INT8 is used)
    var values: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]

    # Per-layer FP32 key staging buffer for batch build (populated by store_tokens, consumed by finalize_layer)
    var key_staging: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]

    # M4: Per-layer V format (one entry per slot = session_idx * MAX_ATTEND_LAYERS + layer_id).
    # Resolved at finalize_layer time from session v_format + boundary_layers config.
    # Read at query_topk time to dispatch the correct dequantizer.
    var values_format: Pointer[UInt8, MutUntrackedOrigin]
    # M4: Per-layer turbo4-packed V blob (used when values_format[slot] == QFMT_TURBO4).
    # Layout per token: [FP32 norm (4B)] [block: FP16 scale (2B) + 16B INT4 data] × (val_dim/32)
    var values_turbo4: Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin]

    # gh #131 §3.3: per-worker grow-only scratch for ATTEND.QUERY output buffers,
    # replacing 3 alloc/free per query (no-alloc-in-event-loop). keys+values share
    # one float buffer (keys at [0, k*key_dim), values after); token ids in an int buffer.
    var query_scratch_f: Pointer[Float32, MutUntrackedOrigin]
    var query_scratch_i: Pointer[Int, MutUntrackedOrigin]
    var query_scratch_f_cap: Int
    var query_scratch_i_cap: Int

    var session_count: Int
    var enabled: Bool

    # Stats
    var total_tokens_stored: Int
    var total_queries: Int
    var total_query_hits: Int

    # Per-slot staging capacity in TOKENS, shared by key_staging and the value
    # buffer allocated alongside it (values or values_turbo4). 0 = no staging
    # buffer live. Guards the store_tokens append path: the buffers are sized
    # at first store and must grow before an append past the current capacity.
    var staging_cap: Pointer[Int, MutUntrackedOrigin]
    # gh #368: tokens stored PER LAYER slot. The session-wide
    # `tokens_per_layer` was both the append offset and the finalize count,
    # and every store — for any layer — added to it: storing 32 layers of
    # 1,000 tokens left it at 32,000, wrote layer L at offset L*1000 (past the
    # buffer sized for its own tokens) and made finalize_layer quantize 32,000
    # rows out of a 1,000-row staging buffer — SIGSEGV in HNSWGraph.quantize.
    var layer_tokens: Pointer[Int, MutUntrackedOrigin]
    # gh #391: each finalized layer's key norms, by token id. The graph is
    # built over the keys' DIRECTIONS; query_topk re-ranks its candidates by
    # exact L2 from these norms. Null for a layer not finalized yet.
    var key_norms: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]

    def __init__(out self, enabled: Bool = False):
        self.enabled = enabled
        self.session_count = 0
        self.total_tokens_stored = 0
        self.total_queries = 0
        self.total_query_hits = 0

        # Only allocate when enabled — avoid wasting memory/cache when --kvcache not set
        var n_slots = MAX_ATTEND_SESSIONS * MAX_ATTEND_LAYERS if enabled else 1

        var _s = alloc[AttendSessionMeta](MAX_ATTEND_SESSIONS if enabled else 1)
        self.sessions = Pointer[AttendSessionMeta, MutUntrackedOrigin](unsafe_from_address=Int(_s))
        var n_sessions = MAX_ATTEND_SESSIONS if enabled else 1
        for si in range(n_sessions):
            self.sessions[unsafe_offset=si] = AttendSessionMeta()

        var _idx = alloc[Pointer[HNSWGraph, MutUntrackedOrigin]](n_slots)
        self.indexes = Pointer[Pointer[HNSWGraph, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_idx))
        for ii in range(n_slots):
            self.indexes[unsafe_offset=ii] = null_ptr[HNSWGraph, MutUntrackedOrigin]()

        var _val = alloc[Pointer[Float32, MutUntrackedOrigin]](n_slots)
        self.values = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_val))
        for vi in range(n_slots):
            self.values[unsafe_offset=vi] = null_ptr[Float32, MutUntrackedOrigin]()

        # INT8 value compression
        var _vi8 = alloc[Pointer[Int8, MutUntrackedOrigin]](n_slots)
        self.values_int8 = Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_vi8))
        for vi in range(n_slots):
            self.values_int8[unsafe_offset=vi] = null_ptr[Int8, MutUntrackedOrigin]()
        var _vs = alloc[Float32](n_slots)
        self.values_scale = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_vs))
        var _vm = alloc[Float32](n_slots)
        self.values_min = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_vm))

        var _ks = alloc[Pointer[Float32, MutUntrackedOrigin]](n_slots)
        self.key_staging = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_ks))
        for ki in range(n_slots):
            self.key_staging[unsafe_offset=ki] = null_ptr[Float32, MutUntrackedOrigin]()

        # M4: per-layer V format + turbo4 blob arrays
        var _vf = alloc[UInt8](n_slots)
        self.values_format = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_vf))
        for fi in range(n_slots):
            self.values_format[unsafe_offset=fi] = QFMT_INT8
        var _vt = alloc[Pointer[Int8, MutUntrackedOrigin]](n_slots)
        self.values_turbo4 = Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_vt))
        for ti in range(n_slots):
            self.values_turbo4[unsafe_offset=ti] = null_ptr[Int8, MutUntrackedOrigin]()

        # Lazily grown on the first ATTEND.QUERY (see _ensure_query_scratch).
        self.query_scratch_f = null_ptr[Float32, MutUntrackedOrigin]()
        self.query_scratch_i = null_ptr[Int, MutUntrackedOrigin]()
        self.query_scratch_f_cap = 0
        self.query_scratch_i_cap = 0

        var _sc = alloc[Int](n_slots)
        self.staging_cap = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_sc))
        for ci in range(n_slots):
            self.staging_cap[unsafe_offset=ci] = 0
        var _lt = alloc[Int](n_slots)
        self.layer_tokens = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_lt))
        for ci in range(n_slots):
            self.layer_tokens[unsafe_offset=ci] = 0
        var _kn = alloc[Pointer[Float32, MutUntrackedOrigin]](n_slots)
        self.key_norms = Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin](
            unsafe_from_address=Int(_kn))
        for ci in range(n_slots):
            self.key_norms[unsafe_offset=ci] = null_ptr[Float32, MutUntrackedOrigin]()

    @always_inline
    def _ensure_query_scratch(mut self, floats_needed: Int, ints_needed: Int):
        """Grow-only per-worker scratch for ATTEND.QUERY outputs. Reallocs only when
        a request needs more than the current capacity; steady state is zero-alloc."""
        if floats_needed > self.query_scratch_f_cap:
            if is_not_null(self.query_scratch_f):
                self.query_scratch_f.unsafe_free()
            var p = alloc[Float32](floats_needed)
            self.query_scratch_f = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(p))
            self.query_scratch_f_cap = floats_needed
        if ints_needed > self.query_scratch_i_cap:
            if is_not_null(self.query_scratch_i):
                self.query_scratch_i.unsafe_free()
            var p = alloc[Int](ints_needed)
            self.query_scratch_i = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(p))
            self.query_scratch_i_cap = ints_needed

    def __moveinit__(out self, deinit take: Self):
        self.sessions = take.sessions
        self.indexes = take.indexes
        self.values = take.values
        self.values_int8 = take.values_int8
        self.values_scale = take.values_scale
        self.values_min = take.values_min
        self.key_staging = take.key_staging
        self.values_format = take.values_format
        self.values_turbo4 = take.values_turbo4
        self.query_scratch_f = take.query_scratch_f
        self.query_scratch_i = take.query_scratch_i
        self.query_scratch_f_cap = take.query_scratch_f_cap
        self.query_scratch_i_cap = take.query_scratch_i_cap
        self.staging_cap = take.staging_cap
        self.layer_tokens = take.layer_tokens
        self.key_norms = take.key_norms
        self.session_count = take.session_count
        self.enabled = take.enabled
        self.total_tokens_stored = take.total_tokens_stored
        self.total_queries = take.total_queries
        self.total_query_hits = take.total_query_hits

    def _hash_bytes(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt64:
        var h = UInt64(0x517cc1b727220a95)
        for bi in range(length):
            h = (h ^ UInt64(ptr[unsafe_offset=bi])) * UInt64(0x9e3779b97f4a7c15)
        return h

    def _find_session(self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int) -> Int:
        var h = self._hash_bytes(sid_ptr, sid_len)
        for si in range(MAX_ATTEND_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                continue
            if self.sessions[unsafe_offset=si].session_hash != h:
                continue
            if self.sessions[unsafe_offset=si].session_id_len != sid_len:
                continue
            # Compare bytes
            var mismatch = False
            for bi in range(sid_len):
                if self.sessions[unsafe_offset=si].session_id_ptr[unsafe_offset=bi] != sid_ptr[unsafe_offset=bi]:
                    mismatch = True
                    break
            if not mismatch:
                return si
        return -1

    def create_session(mut self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int,
                      key_dim: Int, value_dim: Int,
                      k_format: UInt8 = QFMT_INT8,
                      v_format: UInt8 = QFMT_INT8,
                      boundary_layers_n: Int = 0,
                      boundary_v_format: UInt8 = QFMT_INT8,
                      rope_dim: Int = 0) -> Int:
        """Create a new attention session. Returns session index or -1 if full.

        M4: k_format / v_format / boundary_layers_n / boundary_v_format default to
        INT8/INT8/0/INT8 — backward compatible with pre-M4 sessions.

        gh #36: k_format is silently coerced to INT8. K is stored INT8-only in
        the HNSW retrieval graph by design.
        The wire/Python kwarg is preserved for forward-compat but no longer
        round-trips through ATTEND.INFO.

        Block quant (turbo4/3/2 + fp8) requires val_dim divisible by 32. If
        value_dim % 32 != 0, those formats fall back to INT8 silently to avoid
        hard failures on short head dims.

        A2 (gh #39): rope_dim is required when v_format or boundary_v_format is
        QFMT_BF16_ROPE_FP8. Validation: 0 < rope_dim < value_dim and
        (value_dim - rope_dim) % 32 == 0; otherwise the hybrid fmt falls back
        to INT8 to preserve correctness.
        """
        if not self.enabled:
            return -1
        # gh #36: K storage goes through the HNSW INT8 quantizer regardless of
        # k_format. Coerce so ATTEND.INFO reflects reality. Sub-4-bit K with
        # FWHT/QJL Q-rotation (the dflash TQ3_0 design) is intentionally out of
        # scope: in retrieval-attention,
        # K error corrupts top-k candidate selection before V dequant matters,
        # so it cannot be added without a fresh recall/PPL gate.
        var eff_k = QFMT_INT8
        # Block quant formats require 32-dim blocks; fall back to INT8 if not compatible
        var eff_v = v_format
        if (eff_v == QFMT_TURBO4 or eff_v == QFMT_TURBO3 or eff_v == QFMT_TURBO2 or eff_v == QFMT_FP8) and (value_dim % 32) != 0:
            eff_v = QFMT_INT8
        var eff_bv = boundary_v_format
        if (eff_bv == QFMT_TURBO4 or eff_bv == QFMT_TURBO3 or eff_bv == QFMT_TURBO2 or eff_bv == QFMT_FP8) and (value_dim % 32) != 0:
            eff_bv = QFMT_INT8
        # Hybrid validation
        if eff_v == QFMT_BF16_ROPE_FP8 and (rope_dim <= 0 or rope_dim >= value_dim or (value_dim - rope_dim) % 32 != 0):
            eff_v = QFMT_INT8
        if eff_bv == QFMT_BF16_ROPE_FP8 and (rope_dim <= 0 or rope_dim >= value_dim or (value_dim - rope_dim) % 32 != 0):
            eff_bv = QFMT_INT8
        # Find free slot
        for si in range(MAX_ATTEND_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                var meta = AttendSessionMeta()
                meta.active = True
                meta.session_hash = self._hash_bytes(sid_ptr, sid_len)
                var id_copy = alloc[UInt8](sid_len)
                unsafe_memcpy(dest=id_copy, src=sid_ptr, count=sid_len)
                meta.session_id_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(id_copy))
                meta.session_id_len = sid_len
                meta.key_dim = key_dim
                meta.value_dim = value_dim
                meta.k_format = eff_k
                meta.v_format = eff_v
                meta.boundary_layers_n = boundary_layers_n
                meta.boundary_v_format = eff_bv
                meta.rope_dim = rope_dim
                self.sessions[unsafe_offset=si] = meta
                for li in range(MAX_ATTEND_LAYERS):
                    self.layer_tokens[unsafe_offset=si * MAX_ATTEND_LAYERS + li] = 0
                self.session_count += 1
                return si
        return -1

    @always_inline
    def _resolve_layer_v_format(self, session_idx: Int, layer_id: Int) -> UInt8:
        """M4: Resolve per-layer V format, honoring boundary_layers config.
        Boundary layers use boundary_v_format; middle layers use v_format."""
        var meta = self.sessions[unsafe_offset=session_idx]
        var n = meta.boundary_layers_n
        if n > 0:
            if layer_id < n or layer_id >= meta.num_layers - n:
                return meta.boundary_v_format
        return meta.v_format

    @always_inline
    def _bytes_per_token(self, vf: UInt8, val_dim: Int, rope_dim: Int) -> Int:
        """Per-token packed byte size for a block-quantized V format. Single source
        of truth for the layouts used by store (ingest-quant), finalize, and query."""
        var num_blocks = val_dim // 32
        if vf == QFMT_TURBO4:
            return 4 + num_blocks * 18
        elif vf == QFMT_TURBO3:
            return 4 + num_blocks * 14
        elif vf == QFMT_TURBO2:
            return 4 + num_blocks * 10
        elif vf == QFMT_FP8:
            return 4 + num_blocks * 34
        else:  # QFMT_BF16_ROPE_FP8
            return rope_dim * 2 + ((val_dim - rope_dim) // 32) * 34

    def store_tokens(mut self, session_idx: Int, layer_id: Int,
                    keys_fp32: Pointer[Float32, MutUntrackedOrigin],
                    values_fp32: Pointer[Float32, MutUntrackedOrigin],
                    num_tokens: Int) raises -> Bool:
        """Store token KV pairs into a layer's HNSW index.

        Args:
            session_idx: From create_session()
            layer_id: Transformer layer index
            keys_fp32: [num_tokens * key_dim] FP32 key vectors (concatenated heads)
            values_fp32: [num_tokens * value_dim] FP32 value vectors
            num_tokens: Number of tokens to store

        Returns True on success.
        """
        if session_idx < 0 or session_idx >= MAX_ATTEND_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=session_idx].active:
            return False
        if layer_id < 0 or layer_id >= MAX_ATTEND_LAYERS:
            return False

        var meta = self.sessions[unsafe_offset=session_idx]
        var slot = session_idx * MAX_ATTEND_LAYERS + layer_id
        var key_dim = meta.key_dim
        var val_dim = meta.value_dim

        # Allocate staging + value buffers on first use (small initial size, grows as needed)
        var base = self.layer_tokens[unsafe_offset=slot]   # gh #368: this layer's count
        var needed = base + num_tokens
        if is_null(self.key_staging[unsafe_offset=slot]):
            # First allocation: size for exactly what we need (+ headroom)
            var cap = max(needed * 2, INITIAL_STAGING_TOKENS)
            var ks_ptr = alloc[Float32](cap * key_dim)
            self.key_staging[unsafe_offset=slot] = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(ks_ptr))
            # gh #131 §6: quantize block-format V on ingest — allocate the packed blob
            # instead of an FP32 staging buffer, skipping one full FP32 pass + buffer at
            # finalize. Gated to uniform block formats (boundary_layers_n == 0, because the
            # final num_layers isn't known until every layer is stored, so per-layer
            # boundary resolution is unsafe here) with val_dim % 32 == 0. INT8 (needs a
            # global per-layer min/max), FP16, hybrid, and boundary configs keep the
            # FP32-staged path and are quantized in finalize_layer.
            var vf0 = meta.v_format
            var ingest_quant = (meta.boundary_layers_n == 0
                and (vf0 == QFMT_TURBO4 or vf0 == QFMT_TURBO3 or vf0 == QFMT_TURBO2 or vf0 == QFMT_FP8)
                and (val_dim % 32) == 0)
            if ingest_quant:
                var bpt = self._bytes_per_token(vf0, val_dim, meta.rope_dim)
                var _tb = alloc[Int8](cap * bpt)
                self.values_turbo4[unsafe_offset=slot] = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_tb))
                self.values_format[unsafe_offset=slot] = vf0
            else:
                var val_ptr = alloc[Float32](cap * val_dim)
                self.values[unsafe_offset=slot] = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(val_ptr))
            self.staging_cap[unsafe_offset=slot] = cap
        elif needed > self.staging_cap[unsafe_offset=slot]:
            # Grow every live staged buffer before the append below — pre-fix
            # the append ran unchecked past the first allocation's capacity
            # (heap overflow → empty replies, then SIGSEGV on ATTEND.QUERY;
            # reproduced at ~3,800 stored tokens by tests/test_attend_scale.py).
            var new_cap = self.staging_cap[unsafe_offset=slot] * 2
            while new_cap < needed:
                new_cap *= 2
            var old_ks = self.key_staging[unsafe_offset=slot]
            var _nks = alloc[Float32](new_cap * key_dim)
            var nks = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_nks))
            unsafe_memcpy(dest=nks, src=old_ks, count=base * key_dim)
            old_ks.unsafe_free()
            self.key_staging[unsafe_offset=slot] = nks
            if is_not_null(self.values[unsafe_offset=slot]):
                var old_v = self.values[unsafe_offset=slot]
                var _nv = alloc[Float32](new_cap * val_dim)
                var nv = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_nv))
                unsafe_memcpy(dest=nv, src=old_v, count=base * val_dim)
                old_v.unsafe_free()
                self.values[unsafe_offset=slot] = nv
            if is_not_null(self.values_turbo4[unsafe_offset=slot]):
                var vfg = self.values_format[unsafe_offset=slot]
                var bptg = self._bytes_per_token(vfg, val_dim, meta.rope_dim)
                var old_t = self.values_turbo4[unsafe_offset=slot]
                var _nt = alloc[Int8](new_cap * bptg)
                var nt = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_nt))
                unsafe_memcpy(dest=nt, src=old_t, count=base * bptg)
                old_t.unsafe_free()
                self.values_turbo4[unsafe_offset=slot] = nt
            self.staging_cap[unsafe_offset=slot] = new_cap

        var staging = self.key_staging[unsafe_offset=slot]

        # Keys are always FP32-staged (consumed by finalize_layer to build the HNSW).
        unsafe_memcpy(dest=staging.unsafe_offset(base * key_dim), src=keys_fp32, count=num_tokens * key_dim)

        # Values: quantize-on-ingest into the packed blob (block formats), else FP32-stage.
        if is_not_null(self.values_turbo4[unsafe_offset=slot]) and is_null(self.values[unsafe_offset=slot]):
            var vf0 = self.values_format[unsafe_offset=slot]
            var bpt = self._bytes_per_token(vf0, val_dim, meta.rope_dim)
            var blob = self.values_turbo4[unsafe_offset=slot]
            for t in range(num_tokens):
                var src = values_fp32.unsafe_offset(t * val_dim)
                var dst = blob.unsafe_offset((base + t) * bpt)
                if vf0 == QFMT_TURBO4:
                    quantize_fp32_to_block_int4(src, dst, val_dim)
                elif vf0 == QFMT_TURBO3:
                    quantize_fp32_to_block_int3(src, dst, val_dim)
                elif vf0 == QFMT_TURBO2:
                    quantize_fp32_to_block_int2(src, dst, val_dim)
                else:  # QFMT_FP8
                    quantize_fp32_to_block_fp8(src, dst, val_dim)
        else:
            var val_store = self.values[unsafe_offset=slot]
            unsafe_memcpy(dest=val_store.unsafe_offset(base * val_dim), src=values_fp32, count=num_tokens * val_dim)

        self.layer_tokens[unsafe_offset=slot] += num_tokens
        # Session-wide figure for ATTEND.INFO: the longest layer.
        if self.layer_tokens[unsafe_offset=slot] > self.sessions[unsafe_offset=session_idx].tokens_per_layer:
            self.sessions[unsafe_offset=session_idx].tokens_per_layer = self.layer_tokens[unsafe_offset=slot]
        if layer_id >= self.sessions[unsafe_offset=session_idx].num_layers:
            self.sessions[unsafe_offset=session_idx].num_layers = layer_id + 1
        self.total_tokens_stored += num_tokens
        return True

    def finalize_layer(mut self, session_idx: Int, layer_id: Int) raises -> Bool:
        """Build HNSW index from staged FP32 keys. Must be called after all store_tokens() before querying.
        This is the batch build path — constructs the entire HNSW graph in one shot from the staging buffer."""
        if session_idx < 0 or session_idx >= MAX_ATTEND_SESSIONS:
            return False
        if layer_id < 0 or layer_id >= MAX_ATTEND_LAYERS:
            return False

        var slot = session_idx * MAX_ATTEND_LAYERS + layer_id
        if is_null(self.key_staging[unsafe_offset=slot]):
            return False

        var meta = self.sessions[unsafe_offset=session_idx]
        var n_tokens = self.layer_tokens[unsafe_offset=slot]   # gh #368: not the session total
        var key_dim = meta.key_dim
        var staging = self.key_staging[unsafe_offset=slot]

        # Create HNSW sized to actual token count (not MAX — saves memory)
        var hnsw_cap = max(n_tokens + 100, 1000)  # small headroom
        var hnsw_ptr = alloc[HNSWGraph](1)
        # gh #391: M=16 / ef_construction=100 (was 4 / 16, where a stored key
        # found itself 1 time in 5 at 128K tokens).
        hnsw_ptr.unsafe_write(HNSWGraph(hnsw_cap, key_dim, M=16, ef_construction=100))
        self.indexes[unsafe_offset=slot] = Pointer[HNSWGraph, MutUntrackedOrigin](unsafe_from_address=Int(hnsw_ptr))

        var hnsw = self.indexes[unsafe_offset=slot]
        # gh #391: the graph is built over each key's DIRECTION and the norms
        # are kept for query_topk's exact-L2 re-rank (see _rerank_l2). Over raw
        # keys, L2 is dominated by the norms: long keys sit far from every
        # other key, lose their inbound links, and a query for one could not
        # reach it — 52-61% self-match at 128K even with M=16, 92.5% at
        # ef=1000. The staging buffer is freed below, so normalize in place.
        var kn = alloc[Float32](max(n_tokens, 1))
        for ti in range(n_tokens):
            var row = staging.unsafe_offset(ti * key_dim)
            var ss = Float32(0.0)
            for d in range(key_dim):
                ss += row[unsafe_offset=d] * row[unsafe_offset=d]
            kn[ti] = sqrt(ss)
            l2_normalize_fp32(row, row, key_dim)
        self.key_norms[unsafe_offset=slot] = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(kn))
        hnsw[].distance_metric = 1       # search_fp32_scored normalizes the query
        # The INT8 range comes from the keys themselves (gh #376's rule); the
        # fixed [-0.2, 0.2] default fitted only keys a client had rescaled.
        _ = hnsw[]._calibrate(staging, n_tokens)
        # Insert all tokens from staging buffer
        for ti in range(n_tokens):
            hnsw[].insert_no_compact(ti, staging.unsafe_offset(ti * key_dim))

        # Single compact at the end (BFS reorder + INT8 quantization)
        hnsw[].finalize_compact()

        # Free key staging buffer; capacity now describes nothing (the value
        # buffers below are replaced by exact-size finalized blobs).
        self.key_staging[unsafe_offset=slot].unsafe_free()
        self.key_staging[unsafe_offset=slot] = null_ptr[Float32, MutUntrackedOrigin]()
        self.staging_cap[unsafe_offset=slot] = 0

        # M4: Resolve per-layer V format (session default + boundary override).
        var vf = self._resolve_layer_v_format(session_idx, layer_id)
        # Block-quant formats need val_dim divisible by 32; fall back to INT8 if not.
        if (vf == QFMT_TURBO4 or vf == QFMT_TURBO3 or vf == QFMT_TURBO2 or vf == QFMT_FP8) and (meta.value_dim % 32) != 0:
            vf = QFMT_INT8
        # A2 hybrid: needs valid rope_dim. create_session pre-validated, but
        # fallback applies if a stale session is reused with a new value_dim.
        if vf == QFMT_BF16_ROPE_FP8 and (meta.rope_dim <= 0 or meta.rope_dim >= meta.value_dim or (meta.value_dim - meta.rope_dim) % 32 != 0):
            vf = QFMT_INT8
        self.values_format[unsafe_offset=slot] = vf

        var val_dim = meta.value_dim
        var fp32_vals = self.values[unsafe_offset=slot]
        var total_vals = n_tokens * val_dim
        if is_not_null(fp32_vals) and total_vals > 0:
            if vf == QFMT_TURBO4 or vf == QFMT_TURBO3 or vf == QFMT_TURBO2 or vf == QFMT_FP8 or vf == QFMT_BF16_ROPE_FP8:
                # Block-quantized path (turbo4/turbo3/turbo2/fp8/hybrid).
                var bytes_per_token = self._bytes_per_token(vf, val_dim, meta.rope_dim)
                var blob_size = n_tokens * bytes_per_token
                var _tb = alloc[Int8](blob_size)
                var turbo_buf = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_tb))
                for ti in range(n_tokens):
                    var src = fp32_vals.unsafe_offset(ti * val_dim)
                    var dst_t = turbo_buf.unsafe_offset(ti * bytes_per_token)
                    if vf == QFMT_TURBO4:
                        quantize_fp32_to_block_int4(src, dst_t, val_dim)
                    elif vf == QFMT_TURBO3:
                        quantize_fp32_to_block_int3(src, dst_t, val_dim)
                    elif vf == QFMT_TURBO2:
                        quantize_fp32_to_block_int2(src, dst_t, val_dim)
                    elif vf == QFMT_FP8:
                        quantize_fp32_to_block_fp8(src, dst_t, val_dim)
                    else:  # QFMT_BF16_ROPE_FP8
                        quantize_fp32_to_bf16_rope_fp8_body(src, dst_t, val_dim, meta.rope_dim)
                self.values_turbo4[unsafe_offset=slot] = turbo_buf  # reuse the same pointer array for all non-INT8 formats
                self.values[unsafe_offset=slot].unsafe_free()
                self.values[unsafe_offset=slot] = null_ptr[Float32, MutUntrackedOrigin]()
            elif vf == QFMT_FP16:
                # FP16 path: simple truncation, 2× memory reduction vs FP32.
                var fp16_count = total_vals
                var _f16 = alloc[Int8](fp16_count * 2)  # Float16 = 2 bytes
                var fp16_buf = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_f16))
                var f16_ptr = fp16_buf.unsafe_bitcast[Float16]()
                for vi in range(fp16_count):
                    f16_ptr[unsafe_offset=vi] = fp32_vals[unsafe_offset=vi].cast[DType.float16]()
                self.values_turbo4[unsafe_offset=slot] = fp16_buf  # reuse pointer array
                self.values[unsafe_offset=slot].unsafe_free()
                self.values[unsafe_offset=slot] = null_ptr[Float32, MutUntrackedOrigin]()
            else:
                # INT8 path (default): per-layer global min/max → scale to [-127, 127]
                var vmin = fp32_vals[unsafe_offset=0]
                var vmax = fp32_vals[unsafe_offset=0]
                for vi in range(total_vals):
                    var v = fp32_vals[unsafe_offset=vi]
                    if v < vmin: vmin = v
                    if v > vmax: vmax = v

                var vrange = vmax - vmin
                if vrange < Float32(1e-8):
                    vrange = Float32(1.0)
                var scale = Float32(254.0) / vrange

                var _i8 = alloc[Int8](total_vals)
                var int8_vals = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_i8))
                for vi in range(total_vals):
                    var normalized = (fp32_vals[unsafe_offset=vi] - vmin) * scale
                    var clamped = min(max(normalized, Float32(0.0)), Float32(254.0))
                    int8_vals[unsafe_offset=vi] = Int8(Int(clamped) - 127)

                self.values_int8[unsafe_offset=slot] = int8_vals
                self.values_scale[unsafe_offset=slot] = scale
                self.values_min[unsafe_offset=slot] = vmin

                self.values[unsafe_offset=slot].unsafe_free()
                self.values[unsafe_offset=slot] = null_ptr[Float32, MutUntrackedOrigin]()

        return True

    def _rerank_l2(self, hnsw: Pointer[HNSWGraph, MutUntrackedOrigin],
                   norms: Pointer[Float32, MutUntrackedOrigin],
                   query: Pointer[Float32, MutUntrackedOrigin], key_dim: Int,
                   mut ids: List[Int], mut scores: List[Float32]):
        """gh #391: order the graph's candidates by squared L2 to the query.

        The layer's graph holds each key's DIRECTION (finalize_layer), so the
        search returns the keys nearest the query's direction. L2 over raw
        keys is dominated by key norms: short keys sit near everything and
        long ones near nothing, so a graph over raw keys could not reach a
        long key even with ef=1000 (a stored 128-d Gaussian key found itself
        52% of the time at 128K tokens). Directions have no such hubs, and
        |q - k|^2 = |q|^2 + |k|^2 - 2|q||k|cos, with cos from the INT8 codes
        and |k| kept exactly per token, ranks the candidates by L2: a stored
        key, queried, has distance ~0 and comes first once it is among them."""
        if not hnsw[].metric_scores(query, ids, scores):   # COSINE: 1 - cos
            return
        var qss = Float32(0.0)
        for d in range(key_dim):
            qss += query[unsafe_offset=d] * query[unsafe_offset=d]
        var qn = sqrt(qss)
        var n = min(len(ids), len(scores))
        for r in range(n):
            var kn = norms[unsafe_offset=ids[r]]
            scores[r] = qss + kn * kn - 2.0 * qn * kn * (1.0 - scores[r])
        for a in range(1, n):
            var b = a
            while b > 0 and scores[b] < scores[b - 1]:
                var ts = scores[b]; scores[b] = scores[b - 1]; scores[b - 1] = ts
                var ti = ids[b]; ids[b] = ids[b - 1]; ids[b - 1] = ti
                b -= 1

    def query_topk(mut self, session_idx: Int, layer_id: Int,
                  query_fp32: Pointer[Float32, MutUntrackedOrigin],
                  k: Int, ef: Int,
                  out_keys: Pointer[Float32, MutUntrackedOrigin],
                  out_values: Pointer[Float32, MutUntrackedOrigin],
                  out_token_ids: Pointer[Int, MutUntrackedOrigin],
                  ) raises -> Int:
        """Query the HNSW index for top-k matching tokens.

        Args:
            session_idx: Session index
            layer_id: Layer index
            query_fp32: [key_dim] FP32 query vector (concatenated heads)
            k: Number of results
            ef: HNSW search expansion factor
            out_keys: [k * key_dim] buffer for retrieved keys
            out_values: [k * value_dim] buffer for retrieved values
            out_token_ids: [k] buffer for token IDs

        Returns: actual number of results (may be < k if fewer tokens stored).
        """
        self.total_queries += 1
        if session_idx < 0 or session_idx >= MAX_ATTEND_SESSIONS:
            return 0
        if not self.sessions[unsafe_offset=session_idx].active:
            return 0
        if layer_id < 0 or layer_id >= MAX_ATTEND_LAYERS:
            return 0

        var slot = session_idx * MAX_ATTEND_LAYERS + layer_id
        if is_null(self.indexes[unsafe_offset=slot]):
            return 0

        var hnsw = self.indexes[unsafe_offset=slot]
        var val_store = self.values[unsafe_offset=slot]
        var meta = self.sessions[unsafe_offset=session_idx]

        # Search HNSW for top-k keys
        var scores = List[Float32]()
        var norms = self.key_norms[unsafe_offset=slot]
        var cand = k
        if is_not_null(norms):
            cand = max(k, ef)
        # gh #404: a beam narrower than k cannot return k results — widen it,
        # as FT.SEARCH does (ef >= k). No change for k <= ATTEND_QUERY_EF.
        var ef_q = max(ef, k)
        var results = hnsw[].search_fp32_scored(query_fp32, cand, scores, ef_q)
        if is_not_null(norms) and len(results) > 0:
            self._rerank_l2(hnsw, norms, query_fp32, meta.key_dim, results, scores)

        var num_results = len(results)
        if num_results > k:
            num_results = k

        # M4 + gh #131 §5.3: fused batch scatter-gather dequant for every uniform block
        # format (turbo4/turbo3/turbo2/fp8) — avoids per-result call overhead and keeps
        # a tight prefetch pattern. Hybrid/FP16/INT8/FP32 fall through to the loop below.
        var vf = self.values_format[unsafe_offset=slot]
        var turbo_store = self.values_turbo4[unsafe_offset=slot]
        if (vf == QFMT_TURBO4 or vf == QFMT_TURBO3 or vf == QFMT_TURBO2 or vf == QFMT_FP8) and is_not_null(turbo_store) and num_results > 0:
            var bytes_per_token = self._bytes_per_token(vf, meta.value_dim, meta.rope_dim)
            # Build token_ids array for fused kernel
            for ri in range(num_results):
                out_token_ids[unsafe_offset=ri] = results[ri]
            if vf == QFMT_TURBO4:
                fused_topk_dequant_v_turbo4(
                    out_token_ids, num_results, turbo_store,
                    bytes_per_token, meta.value_dim, out_values)
            elif vf == QFMT_TURBO3:
                fused_topk_dequant_v_turbo3(
                    out_token_ids, num_results, turbo_store,
                    bytes_per_token, meta.value_dim, out_values)
            elif vf == QFMT_TURBO2:
                fused_topk_dequant_v_turbo2(
                    out_token_ids, num_results, turbo_store,
                    bytes_per_token, meta.value_dim, out_values)
            else:  # QFMT_FP8
                fused_topk_dequant_v_fp8(
                    out_token_ids, num_results, turbo_store,
                    bytes_per_token, meta.value_dim, out_values)
            self.total_query_hits += 1
            return num_results

        # Copy results: keys from HNSW nodes, values from parallel array
        for ri in range(num_results):
            var token_id = results[ri]
            out_token_ids[unsafe_offset=ri] = token_id

            # Copy key (from HNSW compact buffer or node vector)
            # For now, use the FP32 query approach — HNSW stores INT8 internally,
            # but for Phase 3 MVP we just return the token ID and let the caller
            # reconstruct from its own cache. In production, we'd store FP16 keys
            # and return them directly.

            # M4: Dispatch per-layer V format (reuses vf/turbo_store from above).
            var i8_store = self.values_int8[unsafe_offset=slot]
            if (vf == QFMT_TURBO4 or vf == QFMT_TURBO3 or vf == QFMT_TURBO2 or vf == QFMT_FP8 or vf == QFMT_BF16_ROPE_FP8) and is_not_null(turbo_store):
                # Block-quantized dequant dispatch (hybrid; turbo4/3/2/fp8 handled by the
                # fused fast path above, this loop only runs for QFMT_BF16_ROPE_FP8).
                var bytes_per_token = self._bytes_per_token(vf, meta.value_dim, meta.rope_dim)
                var src_t = turbo_store.unsafe_offset(token_id * bytes_per_token)
                var dst_t = out_values.unsafe_offset(ri * meta.value_dim)
                if vf == QFMT_TURBO4:
                    dequantize_block_int4_to_fp32(src_t, dst_t, meta.value_dim)
                elif vf == QFMT_TURBO3:
                    dequantize_block_int3_to_fp32(src_t, dst_t, meta.value_dim)
                elif vf == QFMT_TURBO2:
                    dequantize_block_int2_to_fp32(src_t, dst_t, meta.value_dim)
                elif vf == QFMT_FP8:
                    dequantize_block_fp8_to_fp32(src_t, dst_t, meta.value_dim)
                else:  # QFMT_BF16_ROPE_FP8
                    dequantize_bf16_rope_fp8_body_to_fp32(src_t, dst_t, meta.value_dim, meta.rope_dim)
            elif vf == QFMT_FP16 and is_not_null(turbo_store):
                # FP16 → FP32 widen (gh #121: SIMD-8, scalar tail)
                var f16_ptr = turbo_store.unsafe_bitcast[Float16]()
                var src_off = token_id * meta.value_dim
                var dst_off = ri * meta.value_dim
                comptime WF = 8
                var d = 0
                while d + WF <= meta.value_dim:
                    out_values.store(dst_off + d, f16_ptr.load[width=WF](src_off + d).cast[DType.float32]())
                    d += WF
                while d < meta.value_dim:
                    out_values[unsafe_offset=dst_off + d] = f16_ptr[unsafe_offset=src_off + d].cast[DType.float32]()
                    d += 1
            elif is_not_null(i8_store):
                # Dequantize INT8 → FP32: val = (int8 + 127) * inv_scale + min.
                # gh #121: inv_scale hoisted (already) + SIMD-16 widen/fma, scalar tail.
                var scale = self.values_scale[unsafe_offset=slot]
                var vmin = self.values_min[unsafe_offset=slot]
                var inv_scale = Float32(1.0) / scale if scale > Float32(1e-8) else Float32(1.0)
                var src_off = token_id * meta.value_dim
                var dst_off = ri * meta.value_dim
                comptime WI = 16
                var inv_v = SIMD[DType.float32, WI](inv_scale)
                var vmin_v = SIMD[DType.float32, WI](vmin)
                var c127 = SIMD[DType.float32, WI](127.0)
                var d = 0
                while d + WI <= meta.value_dim:
                    var f = i8_store.load[width=WI](src_off + d).cast[DType.float32]() + c127
                    out_values.store(dst_off + d, f * inv_v + vmin_v)
                    d += WI
                while d < meta.value_dim:
                    var i8_val = Float32(Int(i8_store[unsafe_offset=src_off + d]) + 127)
                    out_values[unsafe_offset=dst_off + d] = i8_val * inv_scale + vmin
                    d += 1
            elif is_not_null(val_store):
                unsafe_memcpy(dest=out_values.unsafe_offset(ri * meta.value_dim),
                       src=val_store.unsafe_offset(token_id * meta.value_dim),
                       count=meta.value_dim)

        self.total_query_hits += 1 if num_results > 0 else 0
        return num_results
