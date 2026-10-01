"""ATTEND.PREFIX.STORE / ATTEND.PREFIX.QUERY — native Metal attention path.

K/V live in Metal-side memory across calls; only Q crosses the wire on each
subsequent attention call. Backed by `MetalAttentionEngine` (kernels in
`src/ffi/metal_compute.metal`: `sdpa_q1_fp32`, `sdpa_batched_q_fp32`, plus
fp16 variants). Requires `--metal-attention` (or `--metal-attention-fp16`)
at server start.

Wire forms:
  ATTEND.PREFIX.STORE <session_id> <layer_id> <H> <N> <D> <K_blob> <V_blob>
    K_blob, V_blob: H*N*D float32 (host-order). Resident in Metal memory
    until DROP_SESSION or LRU eviction. Returns +OK / -ERR.

  ATTEND.PREFIX.QUERY <session_id> <layer_id> <H> <D> <top_k> <Q_blob>
    Q_blob: H*M*D float32. M is derived from blob length (per_token = H*D*4).
      M=1 → decode-step path (sdpa_q1_fp32 kernel), returns H*D*4 bytes.
      M>1 → batched suffix-Q path (sdpa_batched_q_fp32), returns H*M*D*4
            bytes of attention output FOLLOWED BY H*M*4 bytes of rowwise
            log-sum-exp (LSE = max + log(sum(exp(scores - max)))). The LSE
            trailer is required by the mlx-lm monkey-patch's online softmax
            merge between cached prefix attention and locally computed
            suffix attention.

Used by `pion-vllm-mlx`'s mlx-lm monkey-patch (see
`pion-vllm-mlx/pion_vllm_mlx/mlx_lm_patch.py`).
"""
from src.common.utils import strict_atol

from src.common.ptr import is_not_null, null_ptr
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.metal_attention_engine import MetalAttentionEngine
from src.network.cuda_attention_engine import CudaAttentionEngine
from std.sys.info import CompilationTarget
from std.collections import Array
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy


@always_inline
def handle_attend_prefix_store(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
    mut cuda_engine: CudaAttentionEngine,
) raises -> Int:
    """ATTEND.PREFIX.STORE <session_id> <layer_id> <H> <N> <D> <K_blob> <V_blob>

    gh #9: on Linux, when cuda_engine.available, route to the CUDA path.
    On macOS the cuda_engine is comptime-disabled and we always fall through
    to metal_engine. If neither is available, returns the historical error.
    """
    var have_cuda = cuda_engine.available
    var have_metal = metal_engine.available
    if not have_cuda and not have_metal:
        writer.append_error_response("ERR attention engine not available (start with --metal-attention or --cuda-attention)")
        return 1
    if start + 7 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.STORE requires: session_id layer_id H N D K_blob V_blob")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H = strict_atol(tokens[unsafe_offset=start + 3].value())
    var N = strict_atol(tokens[unsafe_offset=start + 4].value())
    var D = strict_atol(tokens[unsafe_offset=start + 5].value())

    if H <= 0 or N <= 0 or D <= 0:
        writer.append_error_response("ERR invalid H/N/D (must be > 0)")
        return 1

    var K_tok = tokens[unsafe_offset=start + 6]
    var V_tok = tokens[unsafe_offset=start + 7]
    var expected = Int(H) * Int(N) * Int(D) * 4
    if Int(K_tok.length) != expected or Int(V_tok.length) != expected:
        writer.append_error_response("ERR K/V blob size mismatch (expected H*N*D*4 each)")
        return 1

    var K_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(K_tok.ptr.unsafe_bitcast[Float32]()))
    var V_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(V_tok.ptr.unsafe_bitcast[Float32]()))
    var sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))

    var ok = False
    if have_cuda:
        ok = cuda_engine.store_kv(sid_ext, sid_len, Int(layer_id), Int(H), Int(N), Int(D), K_ptr, V_ptr)
    else:
        ok = metal_engine.store_kv(sid_ext, sid_len, Int(layer_id), Int(H), Int(N), Int(D), K_ptr, V_ptr)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.STORE failed (unsupported D? supported: 32,64,96,128,160,192,256)")
    return 1


@always_inline
def handle_attend_prefix_query(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
    mut cuda_engine: CudaAttentionEngine,
) raises -> Int:
    """ATTEND.PREFIX.QUERY <session_id> <layer_id> <H> <D> <top_k> <Q_blob> [<fa_window>]

    Q_blob: H*M*D*4 bytes. M is derived from the blob length (per_token = H*D*4).
      M=1 → decode-step path (sparse top_k attention), returns H*D*4 bytes.
      M>1 → batched suffix-Q path (dense softmax, top_k ignored), returns
            H*M*D*4 bytes of attention output FOLLOWED BY H*M*4 bytes of
            rowwise log-sum-exp (LSE = max + log(sum(exp(scores - max)))).

    gh #60 Step 1: optional `<fa_window>` trailing positional. When present,
    overrides the engine-state `--fa-window` for THIS call only — lets the
    consumer pass a per-layer sliding window (e.g. 512 for Gemma 4 sliding
    layers, 0 for full layers) without changing the server-wide flag. Absent
    = use engine default (backward compatible with pre-#60 consumers).
    """
    var have_cuda = cuda_engine.available
    var have_metal = metal_engine.available
    if not have_cuda and not have_metal:
        writer.append_error_response("ERR attention engine not available (start with --metal-attention or --cuda-attention)")
        return 1
    if start + 6 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY requires: session_id layer_id H D top_k Q_blob [fa_window]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H = strict_atol(tokens[unsafe_offset=start + 3].value())
    var D = strict_atol(tokens[unsafe_offset=start + 4].value())
    var top_k = strict_atol(tokens[unsafe_offset=start + 5].value())
    # gh #60 Step 1: optional positional fa_window override (token at start+7).
    var fa_window_override: Int = -1
    if start + 7 < num_tokens:
        var w_tok = tokens[unsafe_offset=start + 7]
        # Heuristic: a numeric ascii token is the window override; non-numeric
        # tokens (or e.g. another bulk string of binary data) means "no window
        # arg, this is something else" — but the wire is positional so any
        # extra token is treated as the window. Use atol's permissive parse.
        fa_window_override = atol(w_tok.value())

    if H <= 0 or D <= 0 or top_k <= 0:
        writer.append_error_response("ERR invalid H/D/top_k (must be > 0)")
        return 1

    var Q_tok = tokens[unsafe_offset=start + 6]
    var per_tok_bytes = Int(H) * Int(D) * 4
    var Q_bytes = Int(Q_tok.length)
    if per_tok_bytes <= 0 or Q_bytes % per_tok_bytes != 0 or Q_bytes < per_tok_bytes:
        writer.append_error_response("ERR Q blob size must be a positive multiple of H*D*4")
        return 1
    var M = Q_bytes // per_tok_bytes

    var Q_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Q_tok.ptr.unsafe_bitcast[Float32]()))
    var sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))

    # gh #67: if the (sid, layer) was demoted to the cold tier, fail fast with
    # a distinct error code so the client can issue KV.PREFIX.WARM + retry.
    # Without this branch the client sees a generic "session not found" and
    # cannot tell whether it's a stale prefix or a recoverable eviction.
    # CUDA path doesn't yet have cold-tier support — only check on Mac/Metal.
    if have_metal and not have_cuda:
        if metal_engine.session_state(sid_ext, sid_len, Int(layer_id)) == 2:
            writer.append_error_response("COLDMISS session evicted to cold tier (call KV.PREFIX.WARM)")
            return 1

    var out_floats = Int(H) * M * Int(D)
    var lse_floats = Int(H) * M
    var _out = alloc[Float32](out_floats)
    var out_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))
    var lse_ptr = null_ptr[Float32, MutUntrackedOrigin]()
    if M > 1:
        var _lse = alloc[Float32](lse_floats)
        lse_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_lse))

    # gh #9: route M=1 to CUDA when available (Linux + --cuda-attention).
    # M>1 batched + cold-tier paths still Metal-only — those kernels are
    # next-session work. CUDA only has the M=1 dense kernel right now.
    var nr = 0
    if M == 1 and have_cuda:
        if cuda_engine.query(sid_ext, sid_len, Int(layer_id), Int(H), Int(D), Q_ptr, out_ptr,
                             fa_window_override=fa_window_override):
            nr = Int(H) * Int(D)
    elif M == 1:
        if metal_engine.query(sid_ext, sid_len, Int(layer_id), Int(H), Int(D), Q_ptr, out_ptr,
                              fa_window_override=fa_window_override):
            nr = Int(H) * Int(D)
    else:
        if metal_engine.query_batched(sid_ext, sid_len, Int(layer_id), Int(H), M, Int(D), Q_ptr, out_ptr, lse_ptr,
                                      fa_window_override=fa_window_override):
            nr = Int(H) * M * Int(D)

    if nr > 0:
        # M=1 path: just the output. M>1: output ‖ LSE concatenated.
        if M > 1 and is_not_null(lse_ptr):
            var total = (out_floats + lse_floats) * 4
            var combined = alloc[UInt8](total)
            unsafe_memcpy(dest=combined, src=out_ptr.unsafe_bitcast[UInt8](), count=out_floats * 4)
            unsafe_memcpy(dest=combined.unsafe_offset(out_floats * 4), src=lse_ptr.unsafe_bitcast[UInt8](), count=lse_floats * 4)
            var combined_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(combined))
            writer.append_bulk_string_response(combined_ext, total)
            combined.unsafe_free()
        else:
            var out_u8 = out_ptr.unsafe_bitcast[UInt8]()
            var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
            writer.append_bulk_string_response(out_ext, out_floats * 4)
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY failed (session not found, or unsupported D)")

    out_ptr.unsafe_free()
    if is_not_null(lse_ptr):
        lse_ptr.unsafe_free()
    return 1


@always_inline
def handle_attend_prefix_query_fused(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """gh #49: ATTEND.PREFIX.QUERY_FUSED — server-side fused suffix-SDPA + merge.

    Wire form:
      ATTEND.PREFIX.QUERY_FUSED <session_id> <layer_id> <H_q> <D> <S_suf> <H_kv>
                                <Q_blob> <K_suf_blob> <V_suf_blob> <head_map_blob>
                                [<fa_window>]

      Q_blob:        H_q * M * D * 4   bytes  (M derived from blob length)
      K_suf_blob:    H_kv * S_suf * D * 4 bytes  (empty/zero-length OK if S_suf=0)
      V_suf_blob:    H_kv * S_suf * D * 4 bytes  (empty/zero-length OK if S_suf=0)
      head_map_blob: H_q bytes (uint8 per q-head, value < H_kv)
      fa_window:     gh #60 Step 1 — OPTIONAL trailing positional. When
                     present, overrides --fa-window for THIS call only
                     (per-layer routing for hybrid SWA models). Absent =
                     engine default.

    Returns H_q * M * D * 4 bytes — merged attention output. No LSE trailer
    (merge already done server-side). Causal-within-suffix mask is applied
    by the kernel. Prefix is fully visible (within fa_window if set).
    """
    if not metal_engine.available:
        writer.append_error_response("ERR Metal attention not available (start with --metal-attention)")
        return 1
    if start + 10 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_FUSED requires: session_id layer_id H_q D S_suf H_kv Q K_suf V_suf head_map [fa_window]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H_q = strict_atol(tokens[unsafe_offset=start + 3].value())
    var D = strict_atol(tokens[unsafe_offset=start + 4].value())
    var S_suf = strict_atol(tokens[unsafe_offset=start + 5].value())
    var H_kv = strict_atol(tokens[unsafe_offset=start + 6].value())

    if H_q <= 0 or D <= 0 or H_kv <= 0 or S_suf < 0:
        writer.append_error_response("ERR invalid H_q/D/H_kv (must be > 0) or S_suf (>= 0)")
        return 1
    if (H_q % H_kv) != 0:
        writer.append_error_response("ERR H_q must be a multiple of H_kv (GQA)")
        return 1

    # gh #67: COLDMISS short-circuit (see handle_attend_prefix_query for rationale).
    var _sid_ext_cold = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    if metal_engine.session_state(_sid_ext_cold, sid_len, Int(layer_id)) == 2:
        writer.append_error_response("COLDMISS session evicted to cold tier (call KV.PREFIX.WARM)")
        return 1

    var Q_tok = tokens[unsafe_offset=start + 7]
    var per_tok_bytes = Int(H_q) * Int(D) * 4
    var Q_bytes = Int(Q_tok.length)
    if per_tok_bytes <= 0 or Q_bytes % per_tok_bytes != 0 or Q_bytes < per_tok_bytes:
        writer.append_error_response("ERR Q blob size must be a positive multiple of H_q*D*4")
        return 1
    var M = Q_bytes // per_tok_bytes

    var Ks_tok = tokens[unsafe_offset=start + 8]
    var Vs_tok = tokens[unsafe_offset=start + 9]
    var HM_tok = tokens[unsafe_offset=start + 10]
    var ks_expected = Int(H_kv) * Int(S_suf) * Int(D) * 4
    if Int(Ks_tok.length) != ks_expected or Int(Vs_tok.length) != ks_expected:
        writer.append_error_response("ERR K_suf / V_suf size mismatch (expected H_kv*S_suf*D*4 each)")
        return 1
    if Int(HM_tok.length) != Int(H_q):
        writer.append_error_response("ERR head_map size mismatch (expected H_q bytes)")
        return 1

    var Q_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Q_tok.ptr.unsafe_bitcast[Float32]()))
    var Ks_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Ks_tok.ptr.unsafe_bitcast[Float32]()))
    var Vs_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Vs_tok.ptr.unsafe_bitcast[Float32]()))
    var HM_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(HM_tok.ptr))
    var sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))

    var out_floats = Int(H_q) * M * Int(D)
    var _out = alloc[Float32](out_floats)
    var out_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))

    # gh #60 Step 1: optional positional fa_window override at end.
    # Args 0..10: cmd, sid, layer_id, H_q, D, S_suf, H_kv, Q, Ks, Vs, HM
    # Optional arg 11: <fa_window>.
    var fa_window_override: Int = -1
    if start + 11 < num_tokens:
        fa_window_override = strict_atol(tokens[unsafe_offset=start + 11].value())

    var ok = metal_engine.query_batched_fused(
        sid_ext, sid_len, Int(layer_id),
        Int(H_q), M, Int(D), Int(H_kv), Int(S_suf),
        Q_ptr, Ks_ptr, Vs_ptr, HM_ptr, out_ptr,
        fa_window_override=fa_window_override,
    )
    if ok:
        var out_u8 = out_ptr.unsafe_bitcast[UInt8]()
        var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
        writer.append_bulk_string_response(out_ext, out_floats * 4)
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_FUSED failed (session not found, GQA mismatch, or unsupported D)")

    out_ptr.unsafe_free()
    return 1


@always_inline
def handle_attend_prefix_query_sparse(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """gh #60 Phase 2: ATTEND.PREFIX.QUERY_SPARSE — M=1 sparse-mask SDPA.

    Wire form:
      ATTEND.PREFIX.QUERY_SPARSE <session_id> <layer_id> <H> <D>
                                 <K_sparse_max> <Q_blob> <indices_blob>
                                 <counts_blob> [<fa_window>]

      Q_blob:       H * D * 4              bytes (M=1 only — batched comes later)
      indices_blob: H * K_sparse_max * 4   bytes (int32, per-head token IDs)
      counts_blob:  H * 4                  bytes (uint32, per-head actual count
                                                  ≤ K_sparse_max)
      fa_window:    OPTIONAL trailing positional — overrides --fa-window for
                    THIS call (e.g. dense local window layered on top of
                    sparse global picks).

    Returns H * D * 4 bytes of attention output (no LSE — M=1 is decode path).
    """
    if not metal_engine.available:
        writer.append_error_response("ERR Metal attention not available (start with --metal-attention)")
        return 1
    if start + 8 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE requires: session_id layer_id H D K_sparse_max Q indices counts [fa_window]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H = strict_atol(tokens[unsafe_offset=start + 3].value())
    var D = strict_atol(tokens[unsafe_offset=start + 4].value())
    var K_sparse_max = strict_atol(tokens[unsafe_offset=start + 5].value())

    if H <= 0 or D <= 0 or K_sparse_max <= 0:
        writer.append_error_response("ERR invalid H/D/K_sparse_max (must be > 0)")
        return 1

    # gh #67: COLDMISS short-circuit (see handle_attend_prefix_query for rationale).
    var _sid_ext_cold = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    if metal_engine.session_state(_sid_ext_cold, sid_len, Int(layer_id)) == 2:
        writer.append_error_response("COLDMISS session evicted to cold tier (call KV.PREFIX.WARM)")
        return 1

    var Q_tok   = tokens[unsafe_offset=start + 6]
    var Idx_tok = tokens[unsafe_offset=start + 7]
    var Cnt_tok = tokens[unsafe_offset=start + 8]
    var q_expected   = Int(H) * Int(D) * 4
    var idx_expected = Int(H) * Int(K_sparse_max) * 4
    var cnt_expected = Int(H) * 4
    if Int(Q_tok.length) != q_expected:
        writer.append_error_response("ERR Q blob size must be H*D*4")
        return 1
    if Int(Idx_tok.length) != idx_expected:
        writer.append_error_response("ERR indices blob size must be H*K_sparse_max*4")
        return 1
    if Int(Cnt_tok.length) != cnt_expected:
        writer.append_error_response("ERR counts blob size must be H*4")
        return 1

    var fa_window_override: Int = -1
    if start + 9 < num_tokens:
        fa_window_override = strict_atol(tokens[unsafe_offset=start + 9].value())

    var Q_ptr   = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Q_tok.ptr.unsafe_bitcast[Float32]()))
    var Idx_ptr = Pointer[Int32,   MutUntrackedOrigin](unsafe_from_address=Int(Idx_tok.ptr.unsafe_bitcast[Int32]()))
    var Cnt_ptr = Pointer[UInt32,  MutUntrackedOrigin](unsafe_from_address=Int(Cnt_tok.ptr.unsafe_bitcast[UInt32]()))
    var sid_ext = Pointer[UInt8,   MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))

    var out_floats = Int(H) * Int(D)
    var _out = alloc[Float32](out_floats)
    var out_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))

    # Wire form passes head_map only when GQA is desired. For non-GQA callers
    # we synthesize a NULL head_map below (FFI builds identity mapping). v1
    # of the wire command doesn't accept head_map in RESP; passing NULL means
    # the kernel paths use H_q==H_kv assumption.
    var head_map_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
    var ok = metal_engine.query_sparse(
        sid_ext, sid_len, Int(layer_id),
        Int(H), Int(D), Int(K_sparse_max),
        Int(H),                        # H_kv == H_q on this RESP path (no GQA passed)
        Q_ptr, Idx_ptr, Cnt_ptr, head_map_ptr, out_ptr,
        fa_window_override=fa_window_override,
    )
    if ok:
        var out_u8 = out_ptr.unsafe_bitcast[UInt8]()
        var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
        writer.append_bulk_string_response(out_ext, out_floats * 4)
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE failed (session not found, K_sparse_max=0, or unsupported D)")

    out_ptr.unsafe_free()
    return 1


@always_inline
def handle_attend_prefix_lookup(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """gh #65 follow-on: ATTEND.PREFIX.LOOKUP <session_id> <layer_id>.

    Probes the Metal SDPA session cache for (sid, layer_id). Returns:
      +HIT  if the slot exists with non-nil K/V (ATTEND.PREFIX.QUERY would succeed)
      +COLD if the slot was demoted to the cold tier under LRU pressure
            (gh #67) — caller can issue KV.PREFIX.WARM to rehydrate
      +MISS if the slot was never stored, or was explicitly dropped

    Symmetric to KV.PREFIX.LOOKUP for V-store state. Wire-mode consumers
    (pion-vllm-mlx's `mlx_lm_patch`) use this to check ATTEND.PREFIX.* state
    independently of KV.PREFIX.* — a stale KV.PREFIX hit shouldn't cause
    the consumer to skip ATTEND.PREFIX.STORE when the Metal cache is cold.
    """
    if not metal_engine.available:
        writer.append_error_response("ERR Metal attention not available (start with --metal-attention)")
        return 1
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.LOOKUP requires: session_id layer_id")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    if layer_id < 0:
        writer.append_error_response("ERR invalid layer_id (must be >= 0)")
        return 1

    var sid_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_tok.ptr))
    # gh #67: tri-state probe — 1 = WARM, 2 = COLD, 0 = MISSING.
    var state = metal_engine.session_state(sid_ext, Int(sid_tok.length), Int(layer_id))
    if state == 1:
        writer.append_to_response("+HIT\r\n".unsafe_ptr(), 6)
    elif state == 2:
        writer.append_to_response("+COLD\r\n".unsafe_ptr(), 7)
    else:
        writer.append_to_response("+MISS\r\n".unsafe_ptr(), 7)
    return 1


@always_inline
def handle_attend_prefix_query_sparse_auto(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
    mut cuda_engine: CudaAttentionEngine,
) raises -> Int:
    """gh #63 Phase 3b: ATTEND.PREFIX.QUERY_SPARSE_AUTO — server picks
    top-K via the selector + runs sparse attention in one round-trip.

    Wire form:
      ATTEND.PREFIX.QUERY_SPARSE_AUTO <session_id> <layer_id> <H_q> <D>
                                      <B> <K_top> <H_kv> <Q_blob>
                                      <head_map_blob> [<fa_window>] [<selector_id>]

      H_q:           query head count
      H_kv:          kv head count (= H_q on non-GQA; H_q % H_kv must == 0)
      B:             block size for the selector (e.g. 64)
      K_top:         number of blocks to pick (e.g. 8)
      Q_blob:        H_q * D * 4 bytes (M=1 only)
      head_map_blob: H_q bytes (uint8 per q-head, value < H_kv). May be
                     empty (0 bytes) on non-GQA — server synthesizes identity.
      fa_window:     optional trailing positional (default -1 = use engine default)
      selector_id:   optional trailing positional (W11 Phase 2 / gh #9)
                     0 = block-mean Q·mean(K) (default; gh #60 / gh #63 — NIAH-class)
                     1 = Quest upper-bound (recovers factual QA; see W11)
                     Mac (Metal) supports both. Linux (CUDA) currently only
                     supports 0; sending 1 on CUDA returns an error rather
                     than silently degrading to block-mean.

    Returns H_q * D * 4 bytes of attention output.
    """
    var have_cuda = cuda_engine.available
    var have_metal = metal_engine.available
    if not have_cuda and not have_metal:
        writer.append_error_response("ERR attention engine not available (start with --metal-attention or --cuda-attention)")
        return 1
    if start + 9 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE_AUTO requires: session_id layer_id H_q D B K_top H_kv Q head_map [fa_window]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H_q = strict_atol(tokens[unsafe_offset=start + 3].value())
    var D = strict_atol(tokens[unsafe_offset=start + 4].value())
    var B = strict_atol(tokens[unsafe_offset=start + 5].value())
    var K_top = strict_atol(tokens[unsafe_offset=start + 6].value())
    var H_kv = strict_atol(tokens[unsafe_offset=start + 7].value())

    if H_q <= 0 or D <= 0 or B <= 0 or K_top <= 0 or H_kv <= 0:
        writer.append_error_response("ERR invalid H_q/D/B/K_top/H_kv (must be > 0)")
        return 1
    if (H_q % H_kv) != 0:
        writer.append_error_response("ERR H_q must be a multiple of H_kv (GQA)")
        return 1

    # gh #67: COLDMISS short-circuit (see handle_attend_prefix_query for rationale).
    var _sid_ext_cold = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    if metal_engine.session_state(_sid_ext_cold, sid_len, Int(layer_id)) == 2:
        writer.append_error_response("COLDMISS session evicted to cold tier (call KV.PREFIX.WARM)")
        return 1

    var Q_tok = tokens[unsafe_offset=start + 8]
    var HM_tok = tokens[unsafe_offset=start + 9]
    var q_expected = Int(H_q) * Int(D) * 4
    if Int(Q_tok.length) != q_expected:
        writer.append_error_response("ERR Q blob size must be H_q*D*4")
        return 1
    # head_map: empty (0 bytes) → identity (FFI synthesizes); otherwise H_q bytes.
    if Int(HM_tok.length) != 0 and Int(HM_tok.length) != Int(H_q):
        writer.append_error_response("ERR head_map size must be 0 (identity) or H_q bytes")
        return 1

    var fa_window_override: Int = -1
    if start + 10 < num_tokens:
        fa_window_override = strict_atol(tokens[unsafe_offset=start + 10].value())

    var selector_id: Int = 0
    if start + 11 < num_tokens:
        selector_id = strict_atol(tokens[unsafe_offset=start + 11].value())

    var Q_ptr   = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Q_tok.ptr.unsafe_bitcast[Float32]()))
    var sid_ext = Pointer[UInt8,   MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    var hm_ptr  = null_ptr[UInt8,   MutUntrackedOrigin]()
    if Int(HM_tok.length) > 0:
        hm_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(HM_tok.ptr))

    var out_floats = Int(H_q) * Int(D)
    var _out = alloc[Float32](out_floats)
    var out_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))

    var ok = False
    if have_cuda:
        # CUDA path: K_block + K_blocks come straight from B + K_top. The
        # bridge derives h_kv from head_map; if head_map is empty the kernel
        # uses identity (h_kv = h). H_kv is implicit.
        # W11 Phase 2: Quest selector (selector_id=1) is Mac/MSL-only at
        # the moment; reject loudly on CUDA so operators know it didn't
        # silently degrade. CUDA Quest is the Phase 3 follow-up.
        if selector_id != 0:
            writer.append_error_response("ERR selector_id=1 (Quest) is Metal-only; the CUDA path supports selector_id=0 only")
            _out.unsafe_free()
            return 1
        ok = cuda_engine.query_sparse_auto(
            sid_ext, sid_len, Int(layer_id),
            Int(H_q), Int(D),
            Q_ptr, out_ptr,
            Int(B), Int(K_top),
            hm_ptr,
            fa_window_override=fa_window_override,
        )
    else:
        ok = metal_engine.query_sparse_auto(
            sid_ext, sid_len, Int(layer_id),
            Int(H_q), Int(D), Int(B), Int(K_top),
            Int(H_kv),
            Q_ptr, hm_ptr, out_ptr,
            fa_window_override=fa_window_override,
            selector_id=selector_id,
        )
    if ok:
        var out_u8 = out_ptr.unsafe_bitcast[UInt8]()
        var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
        writer.append_bulk_string_response(out_ext, out_floats * 4)
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE_AUTO failed (session not found, B=0, K_top=0, or unsupported D)")

    out_ptr.unsafe_free()
    return 1


@always_inline
def handle_attend_prefix_query_sparse_auto_fused(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut metal_engine: MetalAttentionEngine,
) raises -> Int:
    """gh #63 follow-on: ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED — server picks
    top-K via the selector + runs sparse-prefix + dense-suffix attention with
    online-softmax merge, all in one wire call.

    Wire form:
      ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED <sid> <layer> <H_q> <D> <B> <K_top>
                                            <H_kv> <S_suf> <Q> <K_suf> <V_suf>
                                            <head_map> [<fa_window>] [<selector_id>]

      Q:           H_q * D * 4 bytes
      K_suf:       H_kv * S_suf * D * 4 bytes (empty if S_suf=0)
      V_suf:       H_kv * S_suf * D * 4 bytes (empty if S_suf=0)
      head_map:    H_q bytes (empty = identity)
      fa_window:   optional positional (default -1 = use engine default)
      selector_id: optional positional (W11 Phase 2 / gh #9)
                   0 = block-mean (default), 1 = Quest UB

    Returns H_q * D * 4 bytes — merged attention output. No LSE trailer
    (merge is fused). When S_suf=0 this degenerates to sparse_auto.
    """
    if not metal_engine.available:
        writer.append_error_response("ERR Metal attention not available (start with --metal-attention)")
        return 1
    if start + 12 >= num_tokens:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED requires: sid layer H_q D B K_top H_kv S_suf Q K_suf V_suf head_map [fa_window]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var H_q = strict_atol(tokens[unsafe_offset=start + 3].value())
    var D = strict_atol(tokens[unsafe_offset=start + 4].value())
    var B = strict_atol(tokens[unsafe_offset=start + 5].value())
    var K_top = strict_atol(tokens[unsafe_offset=start + 6].value())
    var H_kv = strict_atol(tokens[unsafe_offset=start + 7].value())
    var S_suf = strict_atol(tokens[unsafe_offset=start + 8].value())

    if H_q <= 0 or D <= 0 or B <= 0 or K_top <= 0 or H_kv <= 0:
        writer.append_error_response("ERR invalid H_q/D/B/K_top/H_kv (must be > 0)")
        return 1
    if S_suf < 0:
        writer.append_error_response("ERR S_suf must be >= 0")
        return 1
    if (H_q % H_kv) != 0:
        writer.append_error_response("ERR H_q must be a multiple of H_kv (GQA)")
        return 1

    # gh #67: COLDMISS short-circuit (see handle_attend_prefix_query for rationale).
    var _sid_ext_cold = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    if metal_engine.session_state(_sid_ext_cold, sid_len, Int(layer_id)) == 2:
        writer.append_error_response("COLDMISS session evicted to cold tier (call KV.PREFIX.WARM)")
        return 1

    var Q_tok  = tokens[unsafe_offset=start + 9]
    var Ks_tok = tokens[unsafe_offset=start + 10]
    var Vs_tok = tokens[unsafe_offset=start + 11]
    var HM_tok = tokens[unsafe_offset=start + 12]
    var q_expected  = Int(H_q) * Int(D) * 4
    var ks_expected = Int(H_kv) * Int(S_suf) * Int(D) * 4
    if Int(Q_tok.length) != q_expected:
        writer.append_error_response("ERR Q blob size must be H_q*D*4")
        return 1
    if Int(Ks_tok.length) != ks_expected or Int(Vs_tok.length) != ks_expected:
        writer.append_error_response("ERR K_suf / V_suf size mismatch (expected H_kv*S_suf*D*4 each)")
        return 1
    if Int(HM_tok.length) != 0 and Int(HM_tok.length) != Int(H_q):
        writer.append_error_response("ERR head_map size must be 0 (identity) or H_q bytes")
        return 1

    var fa_window_override: Int = -1
    if start + 13 < num_tokens:
        fa_window_override = strict_atol(tokens[unsafe_offset=start + 13].value())

    var selector_id: Int = 0
    if start + 14 < num_tokens:
        selector_id = strict_atol(tokens[unsafe_offset=start + 14].value())

    var Q_ptr   = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Q_tok.ptr.unsafe_bitcast[Float32]()))
    var sid_ext = Pointer[UInt8,   MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr))
    var hm_ptr  = null_ptr[UInt8,   MutUntrackedOrigin]()
    if Int(HM_tok.length) > 0:
        hm_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(HM_tok.ptr))
    var Ks_ptr = null_ptr[Float32, MutUntrackedOrigin]()
    var Vs_ptr = null_ptr[Float32, MutUntrackedOrigin]()
    if Int(Ks_tok.length) > 0:
        Ks_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Ks_tok.ptr.unsafe_bitcast[Float32]()))
        Vs_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(Vs_tok.ptr.unsafe_bitcast[Float32]()))

    var out_floats = Int(H_q) * Int(D)
    var _out = alloc[Float32](out_floats)
    var out_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))

    var ok = metal_engine.query_sparse_auto_fused(
        sid_ext, sid_len, Int(layer_id),
        Int(H_q), Int(D), Int(B), Int(K_top),
        Int(H_kv), Int(S_suf),
        Q_ptr, hm_ptr, Ks_ptr, Vs_ptr, out_ptr,
        fa_window_override=fa_window_override,
        selector_id=selector_id,
    )
    if ok:
        var out_u8 = out_ptr.unsafe_bitcast[UInt8]()
        var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
        writer.append_bulk_string_response(out_ext, out_floats * 4)
    else:
        writer.append_error_response("ERR ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED failed (session not found, invalid params, or unsupported D)")

    out_ptr.unsafe_free()
    return 1
