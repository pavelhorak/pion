"""Attend commands — ATTEND.CREATE, ATTEND.STORE, ATTEND.QUERY, ATTEND.INFO

Phase 3 of M14 (Externalized Attention).

Protocol:
  ATTEND.CREATE <session_id> <key_dim> <value_dim>
  ATTEND.STORE <session_id> <layer_id> <num_tokens> <keys_blob:FP32> <values_blob:FP32>
  ATTEND.QUERY <session_id> <layer_id> <k> <query_blob:FP32>
  ATTEND.INFO
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.collections import Array
from std.memory import unsafe_memcpy

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.common.utils import parse_int64_strict
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.network.attention_index import (
    MAX_ATTEND_LAYERS,
    MAX_ATTEND_TOKENS,
    AttentionIndex,
    ATTEND_QUERY_EF,
    QFMT_INT8,
    QFMT_TURBO4,
    QFMT_TURBO3,
    QFMT_TURBO2,
    QFMT_FP16,
    QFMT_FP8,
    QFMT_BF16_ROPE_FP8,
)


@always_inline
def _parse_qfmt(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt8:
    """Parse a quant format token. Recognizes: int8, turbo4/3/2, fp16, fp8,
    bf16_rope_fp8_body. Default: INT8. Case-insensitive ASCII."""
    if length == 3:
        # fp8: f(102),p(112),8(56)
        if (ptr[unsafe_offset=0]|0x20) == 102 and (ptr[unsafe_offset=1]|0x20) == 112 and ptr[unsafe_offset=2] == 56:
            return QFMT_FP8
    if length == 4:
        # int8: i(105),n(110),t(116),8(56)
        if (ptr[unsafe_offset=0]|0x20) == 105 and (ptr[unsafe_offset=1]|0x20) == 110 and (ptr[unsafe_offset=2]|0x20) == 116 and ptr[unsafe_offset=3] == 56:
            return QFMT_INT8
        # fp16: f(102),p(112),1(49),6(54)
        if (ptr[unsafe_offset=0]|0x20) == 102 and (ptr[unsafe_offset=1]|0x20) == 112 and ptr[unsafe_offset=2] == 49 and ptr[unsafe_offset=3] == 54:
            return QFMT_FP16
    if length == 6:
        # turbo4/turbo3/turbo2: t(116),u(117),r(114),b(98),o(111),{4|3|2}
        if ((ptr[unsafe_offset=0]|0x20) == 116 and (ptr[unsafe_offset=1]|0x20) == 117 and (ptr[unsafe_offset=2]|0x20) == 114
            and (ptr[unsafe_offset=3]|0x20) == 98 and (ptr[unsafe_offset=4]|0x20) == 111):
            if ptr[unsafe_offset=5] == 52: return QFMT_TURBO4   # '4'
            if ptr[unsafe_offset=5] == 51: return QFMT_TURBO3   # '3'
            if ptr[unsafe_offset=5] == 50: return QFMT_TURBO2   # '2'
    if length == 18:
        # bf16_rope_fp8_body — full literal compare (case-insensitive)
        var lit = "bf16_rope_fp8_body"
        var lp = lit.unsafe_ptr()
        var ok = True
        for i in range(18):
            var b = ptr[unsafe_offset=i]
            if b >= UInt8(65) and b <= UInt8(90):
                b = b | UInt8(0x20)
            if b != lp[unsafe_offset=i]:
                ok = False
                break
        if ok:
            return QFMT_BF16_ROPE_FP8
    return QFMT_INT8


@always_inline
def _tok_matches(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int,
                 expected_lower: StringLiteral) -> Bool:
    """Case-insensitive ASCII match against a lowercase literal."""
    if length != expected_lower.byte_length():
        return False
    var elit = expected_lower.unsafe_ptr()
    for i in range(length):
        if (ptr[unsafe_offset=i] | 0x20) != elit[unsafe_offset=i]:
            return False
    return True


comptime ATTEND_MAX_DIM = 65536
comptime ATTEND_MAX_K = 4096


def _attend_int(tokens: Pointer[RESP3Token, MutUntrackedOrigin], idx: Int,
                lo: Int, hi: Int) -> Int:
    """A strictly parsed integer argument in [lo, hi], or -1.

    Every numeric argument here used to go through `n = n * 10 + (b - 48)`
    over every byte: "abc" and "-5" became dimensions, a count of 99999
    against a 128-byte blob read ~6 MB past the request (other clients'
    bytes) into the store, and k = 100000000 sized a ~13 GB scratch buffer.
    All replied success. (gh #229's rule, reaching the substrate.)"""
    var p = parse_int64_strict(tokens[unsafe_offset=idx].ptr, tokens[unsafe_offset=idx].length)
    if not p.ok or p.value < Int64(lo) or p.value > Int64(hi):
        return -1
    return Int(p.value)


@always_inline
def handle_attend_create(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut attn_idx: AttentionIndex,
) raises -> Int:
    """ATTEND.CREATE <session_id> <key_dim> <value_dim>
                    [KQUANT int8|fp16]
                    [VQUANT int8|turbo4|fp16]
                    [BOUNDARY <n>]
                    [BOUNDARY_VQUANT int8|fp16]

    M4: trailing KQUANT/VQUANT/BOUNDARY/BOUNDARY_VQUANT args are optional.
    Defaults preserve backward compatibility (INT8/INT8, no boundary protection).
    """
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR ATTEND.CREATE requires: <session_id> <key_dim> <value_dim>")
        return 3  # skip 3 args even on error to avoid phantom commands
    if not attn_idx.enabled:
        writer.append_error_response("ERR attention index disabled (use --kvcache)")
        return 3

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length

    var key_dim = _attend_int(tokens, start + 2, 1, ATTEND_MAX_DIM)
    var value_dim = _attend_int(tokens, start + 3, 1, ATTEND_MAX_DIM)
    if key_dim < 0 or value_dim < 0:
        writer.append_error_response("ERR ATTEND.CREATE key_dim and value_dim must be integers in 1..65536")
        return 3

    # M4: optional trailing args (KQUANT <fmt>, VQUANT <fmt>, BOUNDARY <n>,
    # BOUNDARY_VQUANT <fmt>). A2 (gh #39) adds ROPE <n> for the
    # bf16_rope_fp8_body hybrid format — per-session rope_dim.
    var kfmt = QFMT_INT8
    var vfmt = QFMT_INT8
    var boundary_n = 0
    var boundary_vfmt = QFMT_INT8
    var rope_dim = 0
    var consumed = 3  # base: sid + key_dim + value_dim
    var ci = start + 4
    while ci + 1 < num_tokens and consumed < 13:  # cap to avoid runaway parse
        var kw_ptr = tokens[unsafe_offset=ci].ptr
        var kw_len = tokens[unsafe_offset=ci].length
        if _tok_matches(kw_ptr, kw_len, "kquant"):
            kfmt = _parse_qfmt(tokens[unsafe_offset=ci + 1].ptr, tokens[unsafe_offset=ci + 1].length)
            ci += 2; consumed += 2
        elif _tok_matches(kw_ptr, kw_len, "vquant"):
            vfmt = _parse_qfmt(tokens[unsafe_offset=ci + 1].ptr, tokens[unsafe_offset=ci + 1].length)
            ci += 2; consumed += 2
        elif _tok_matches(kw_ptr, kw_len, "boundary"):
            boundary_n = _attend_int(tokens, ci + 1, 0, MAX_ATTEND_LAYERS)
            if boundary_n < 0:
                writer.append_error_response("ERR ATTEND.CREATE BOUNDARY is not an integer or out of range")
                return consumed + 2
            ci += 2; consumed += 2
        elif _tok_matches(kw_ptr, kw_len, "boundary_vquant"):
            boundary_vfmt = _parse_qfmt(tokens[unsafe_offset=ci + 1].ptr, tokens[unsafe_offset=ci + 1].length)
            ci += 2; consumed += 2
        elif _tok_matches(kw_ptr, kw_len, "rope"):
            rope_dim = _attend_int(tokens, ci + 1, 0, key_dim)
            if rope_dim < 0:
                writer.append_error_response("ERR ATTEND.CREATE ROPE must be an integer in 0..key_dim")
                return consumed + 2
            ci += 2; consumed += 2
        else:
            break  # unknown token — stop parsing (it belongs to the next command)

    var si = attn_idx.create_session(sid_ptr, sid_len, key_dim, value_dim,
                                       k_format=kfmt, v_format=vfmt,
                                       boundary_layers_n=boundary_n,
                                       boundary_v_format=boundary_vfmt,
                                       rope_dim=rope_dim)
    if si >= 0:
        writer.append_int_response(Int64(si))
    else:
        writer.append_error_response("ERR ATTEND.CREATE failed (max sessions reached or disabled)")
    return consumed


@always_inline
def handle_attend_store(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut attn_idx: AttentionIndex,
) raises -> Int:
    """ATTEND.STORE <session_id> <layer_id> <num_tokens> <keys_blob> <values_blob>"""
    if start + 5 >= num_tokens:
        writer.append_error_response("ERR ATTEND.STORE requires: <session_id> <layer_id> <num_tokens> <keys> <values>")
        return 5

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length

    var layer_id = _attend_int(tokens, start + 2, 0, MAX_ATTEND_LAYERS - 1)
    var n_tokens = _attend_int(tokens, start + 3, 1, MAX_ATTEND_TOKENS)
    if layer_id < 0 or n_tokens < 0:
        writer.append_error_response("ERR ATTEND.STORE layer_id/num_tokens is not an integer or out of range")
        return 5

    var keys_ptr = tokens[unsafe_offset=start + 4].ptr.unsafe_bitcast[Float32]()
    var values_ptr = tokens[unsafe_offset=start + 5].ptr.unsafe_bitcast[Float32]()

    # Find session
    var si = attn_idx._find_session(sid_ptr, sid_len)
    if si < 0:
        writer.append_error_response("ERR session not found (call ATTEND.CREATE first)")
        return 5
    # The blobs must hold exactly num_tokens rows: the store reads that many.
    var smeta = attn_idx.sessions[unsafe_offset=si]
    if tokens[unsafe_offset=start + 4].length != n_tokens * smeta.key_dim * 4 \
       or tokens[unsafe_offset=start + 5].length != n_tokens * smeta.value_dim * 4:
        writer.append_error_response("ERR ATTEND.STORE blob sizes must be num_tokens * dim * 4 bytes")
        return 5

    var ok = attn_idx.store_tokens(si, layer_id, keys_ptr, values_ptr, n_tokens)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR ATTEND.STORE failed")
    return 5  # cmd + 5 args


@always_inline
def handle_attend_query(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut attn_idx: AttentionIndex,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
) raises -> Int:
    """ATTEND.QUERY <session_id> <layer_id> <k> <query_blob>

    Reply: a bulk string of the top-k value rows (FP32, best first), or `*0`
    when the layer holds nothing."""
    if start + 4 >= num_tokens:
        writer.append_error_response("ERR ATTEND.QUERY requires: <session_id> <layer_id> <k> <query>")
        return 4

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length

    var layer_id = _attend_int(tokens, start + 2, 0, MAX_ATTEND_LAYERS - 1)
    var k = _attend_int(tokens, start + 3, 1, ATTEND_MAX_K)
    if layer_id < 0 or k < 0:
        writer.append_error_response("ERR ATTEND.QUERY layer_id/k is not an integer or out of range")
        return 4

    var query_ptr = tokens[unsafe_offset=start + 4].ptr.unsafe_bitcast[Float32]()

    # Find session. An unknown session is an ERROR: `[]` looked exactly like
    # "nothing relevant stored", so a typo'd session id was a silent miss.
    var si = attn_idx._find_session(sid_ptr, sid_len)
    if si < 0:
        writer.append_error_response("ERR session not found (call ATTEND.CREATE first)")
        return 4

    var meta = attn_idx.sessions[unsafe_offset=si]
    if tokens[unsafe_offset=start + 4].length != meta.key_dim * 4:
        writer.append_error_response("ERR ATTEND.QUERY query must be key_dim * 4 bytes")
        return 4
    var ef = ATTEND_QUERY_EF   # gh #391

    # gh #131 §3.3: per-worker grow-only scratch instead of 3 alloc/free per query.
    # One float buffer holds keys [0, k*key_dim) then values [k*key_dim, …); ids separate.
    var kd = meta.key_dim
    var vd = meta.value_dim
    attn_idx._ensure_query_scratch(k * (kd + vd), k)
    var out_keys = attn_idx.query_scratch_f              # keys (unused in RESP response)
    var out_values = attn_idx.query_scratch_f.unsafe_offset(k * kd)   # values
    var out_token_ids = attn_idx.query_scratch_i

    var num_results = attn_idx.query_topk(si, layer_id, query_ptr, k, ef,
                                           out_keys, out_values, out_token_ids)

    # gh #404: the reply is ALL `num_results` value rows, best match first, as
    # one bulk string of num_results * value_dim FP32s — `num_results` is k, or
    # fewer when fewer tokens are stored, and the caller recovers it as
    # len / (value_dim * 4). This used to be the first row only, whatever k
    # was: a k=32 caller attended over one token with no error and no count.
    # k=1 replies are byte-identical to before.
    if num_results == 0:
        writer.append_empty_array_response()
    else:
        var val_bytes = num_results * meta.value_dim * 4
        var val_ptr = (out_values).unsafe_bitcast[UInt8]()
        writer.append_bulk_string_response(val_ptr, val_bytes)

    # No free: out_keys/out_values/out_token_ids are per-worker scratch, reused next query.

    return 4  # cmd + 4 args


@always_inline
def handle_attend_finalize(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut attn_idx: AttentionIndex,
) raises -> Int:
    """ATTEND.FINALIZE <session_id> <layer_id> — compact HNSW index after all tokens stored."""
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR ATTEND.FINALIZE requires: <session_id> <layer_id>")
        return 2

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length

    var layer_id = _attend_int(tokens, start + 2, 0, MAX_ATTEND_LAYERS - 1)
    if layer_id < 0:
        writer.append_error_response("ERR ATTEND.FINALIZE layer_id is not an integer or out of range")
        return 2

    var si = attn_idx._find_session(sid_ptr, sid_len)
    if si < 0:
        writer.append_error_response("ERR session not found")
        return 2

    var ok = attn_idx.finalize_layer(si, layer_id)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR ATTEND.FINALIZE failed")
    return 2  # cmd + 2 args

@always_inline
def _qfmt_str(fmt: UInt8) -> String:
    if fmt == QFMT_TURBO4:
        return String("turbo4")
    if fmt == QFMT_TURBO3:
        return String("turbo3")
    if fmt == QFMT_TURBO2:
        return String("turbo2")
    if fmt == QFMT_FP16:
        return String("fp16")
    if fmt == QFMT_FP8:
        return String("fp8")
    if fmt == QFMT_BF16_ROPE_FP8:
        return String("bf16_rope_fp8_body")
    return String("int8")


@always_inline
def handle_attend_info(
    mut writer: ResponseWriter,
    attn_idx: AttentionIndex,
) raises -> Int:
    """ATTEND.INFO — statistics for attention indexes (M4: per-session quant info)."""
    var info = String("sessions:") + String(attn_idx.session_count) + "\r\n"
    info += "total_tokens:" + String(attn_idx.total_tokens_stored) + "\r\n"
    info += "total_queries:" + String(attn_idx.total_queries) + "\r\n"
    info += "query_hits:" + String(attn_idx.total_query_hits) + "\r\n"
    info += "enabled:" + String(attn_idx.enabled) + "\r\n"
    # M4: per-session quant config for active sessions
    for si in range(32):  # MAX_ATTEND_SESSIONS
        if not attn_idx.sessions[unsafe_offset=si].active:
            continue
        var m = attn_idx.sessions[unsafe_offset=si]
        info += "session[" + String(si) + "].kquant:" + _qfmt_str(m.k_format) + "\r\n"
        info += "session[" + String(si) + "].vquant:" + _qfmt_str(m.v_format) + "\r\n"
        info += "session[" + String(si) + "].boundary:" + String(m.boundary_layers_n) + "\r\n"
        if m.boundary_layers_n > 0:
            info += "session[" + String(si) + "].boundary_vquant:" + _qfmt_str(m.boundary_v_format) + "\r\n"
        info += "session[" + String(si) + "].tokens:" + String(m.tokens_per_layer) + "\r\n"
        info += "session[" + String(si) + "].layers:" + String(m.num_layers) + "\r\n"
    writer.append_bulk_string_response(info.unsafe_ptr(), info.byte_length())
    return 0  # no args (just the command token)
