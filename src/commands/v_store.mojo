"""V.CREATE / V.STOREBATCH / V.FETCH / V.INFO — token-ID-indexed V-cache.

GPU keeps K for attention routing. Pion stores V (quantized) by token ID.
On decode, GPU sends top-k token IDs → Pion returns dequantized V.

A1 (gh #29) extends V.CREATE with a SCHEMA form for per-layer formats.
"""

from src.common.ptr import is_not_null
from src.common.utils import strict_atol, arg_eq
from src.network.v_store import (
    VStoreIndex, VFMT_INT8, VFMT_TURBO4, VFMT_TURBO3, VFMT_TURBO2, VFMT_FP16,
    VFMT_FP8, VFMT_BF16_ROPE_FP8,
    vstore_dir_lookup, MAX_VS_LAYERS,
    VFMT_MLX4G32,
)
from src.network.response_writer import ResponseWriter
from src.common.metrics import ValueLedger
from src.network.server import TCPServer
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from std.collections import Array
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy

# Maximum heterogeneous schema length. The RESP token-array frame
# limit caps a SCHEMA call at: cmd + sid + (default_dim) + SCHEMA + N + N specs
# — leaves room for ≤59 layer specs without changing the parser. Refused
# beyond this; doc §2.2 spells out the continuation form for >60-layer models.
comptime MAX_SCHEMA_LAYERS = 59


@always_inline
def _spec_starts_with(spec_ptr: Pointer[UInt8, MutUntrackedOrigin],
                       spec_len: Int, key: String) -> Int:
    """Return the offset of `=` if spec begins with `<key>=`, else -1.
    `key` is matched byte-for-byte in lowercase."""
    var key_bytes = key.as_bytes()
    var key_len = key.byte_length()
    if spec_len < key_len + 2:
        return -1
    for k in range(key_len):
        var b = spec_ptr[unsafe_offset=k] | UInt8(0x20)
        if b != key_bytes[k]:
            return -1
    if spec_ptr[unsafe_offset=key_len] != UInt8(61):  # '='
        return -1
    return key_len + 1


@always_inline
def _parse_fmt_token(p: Pointer[UInt8, MutUntrackedOrigin], n: Int,
                      mut out_fmt: UInt8, mut out_unsupported: Bool,
                      mut out_reserved: Bool) -> Bool:
    """Parse a format substring (case-insensitive). Returns True if it parsed
    to a known supported tag. Sets `out_unsupported` for fp8 / bf16_rope_fp8_body
    (A2 territory) and `out_reserved` for `s=` (D3 reservation; reported separately)."""
    out_unsupported = False
    out_reserved = False
    # Lowercase comparison via temp string. `b | 0x20` only safely lowercases
    # ASCII letters [A-Z]; for non-letters (e.g. `_` = 0x5f → 0x7f DEL) it
    # corrupts the byte. So we lowercase only when the byte is in the
    # uppercase letter range.
    var s = String("")
    for i in range(n):
        var b = p[unsafe_offset=i]
        if b >= UInt8(65) and b <= UInt8(90):
            b = b | UInt8(0x20)
        s += chr(Int(b))
    if s == "int8":
        out_fmt = VFMT_INT8
        return True
    if s == "turbo4":
        out_fmt = VFMT_TURBO4
        return True
    if s == "turbo3":
        out_fmt = VFMT_TURBO3
        return True
    if s == "turbo2":
        out_fmt = VFMT_TURBO2
        return True
    if s == "fp16":
        out_fmt = VFMT_FP16
        return True
    if s == "mlx4g32":
        out_fmt = VFMT_MLX4G32
        return True
    if s == "fp8":
        out_fmt = VFMT_FP8
        return True
    if s == "bf16_rope_fp8_body":
        out_fmt = VFMT_BF16_ROPE_FP8
        return True
    return False


@always_inline
def handle_v_create(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """V.CREATE <session_id> <value_dim> [VQUANT int8|turbo4|turbo3|turbo2|fp16|mlx4g32]

    Creates a V-store session. Returns session index.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1

    if start + 2 >= num_tokens:
        writer.append_error_response("ERR V.CREATE requires: session_id value_dim [VQUANT fmt] OR sid [default_dim] SCHEMA N spec_0 ... spec_{N-1}")
        return 1

    # Parse session_id
    var sid_tok = tokens[unsafe_offset=start + 1]
    var sid_ptr = sid_tok.ptr
    var sid_len = Int(sid_tok.length)

    # ── A1 SCHEMA detection ──────────────────────────────────────────────
    # Two heterogeneous forms:
    #   V.CREATE sid SCHEMA N spec_0 ...               (default_dim = 0)
    #   V.CREATE sid <default_dim> SCHEMA N spec_0 ... (default_dim from arg)
    # Otherwise legacy uniform form.
    var sec = tokens[unsafe_offset=start + 2]
    var sec_is_schema = False
    if Int(sec.length) == 6:
        var sp = sec.ptr
        # SCHEMA (6 bytes: s=115,c=99,h=104,e=101,m=109,a=97 case-insensitive)
        if (sp[unsafe_offset=0] | 0x20) == 115 and (sp[unsafe_offset=1] | 0x20) == 99 and (sp[unsafe_offset=2] | 0x20) == 104 and (sp[unsafe_offset=3] | 0x20) == 101 and (sp[unsafe_offset=4] | 0x20) == 109 and (sp[unsafe_offset=5] | 0x20) == 97:
            sec_is_schema = True

    # Two-arg lookahead for `<dim> SCHEMA` form.
    var schema_at = -1   # token index of the literal "SCHEMA"
    var default_dim = 0
    if sec_is_schema:
        schema_at = start + 2
    elif start + 3 < num_tokens:
        var th = tokens[unsafe_offset=start + 3]
        if Int(th.length) == 6:
            var tp = th.ptr
            if (tp[unsafe_offset=0] | 0x20) == 115 and (tp[unsafe_offset=1] | 0x20) == 99 and (tp[unsafe_offset=2] | 0x20) == 104 and (tp[unsafe_offset=3] | 0x20) == 101 and (tp[unsafe_offset=4] | 0x20) == 109 and (tp[unsafe_offset=5] | 0x20) == 97:
                schema_at = start + 3
                var dd = atol(sec.value())
                # 0 = sentinel for "no default supplied; every layer must
                # carry its own dim=" (matches V4 indexer-vs-main pattern
                # where layers genuinely have nothing in common dim-wise).
                if dd < 0 or dd > 8192:
                    writer.append_error_response("ERR V.CREATE SCHEMA: default_dim must be 0-8192")
                    return 1
                default_dim = Int(dd)

    if schema_at >= 0:
        # Heterogeneous SCHEMA form
        if schema_at + 1 >= num_tokens:
            writer.append_error_response("ERR V.CREATE SCHEMA requires N (layer count)")
            return 1
        var n_layers_raw = strict_atol(tokens[unsafe_offset=schema_at + 1].value())
        if n_layers_raw <= 0:
            writer.append_error_response("ERR V.CREATE SCHEMA: N must be positive")
            return 1
        if n_layers_raw > MAX_SCHEMA_LAYERS:
            writer.append_error_response("ERR V.CREATE SCHEMA: N exceeds " + String(MAX_SCHEMA_LAYERS) + "; split into multiple sessions or wait for V.CREATE.SCHEMA.APPEND")
            return 1
        var n_layers = Int(n_layers_raw)
        if schema_at + 1 + n_layers >= num_tokens:
            writer.append_error_response("ERR V.CREATE SCHEMA: not enough layer specs (need " + String(n_layers) + ")")
            return 1

        var ldim_buf = alloc[Int](n_layers)
        var lfmt_buf = alloc[UInt8](n_layers)
        var lrope_buf = alloc[Int](n_layers)
        var ldim = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(ldim_buf))
        var lfmt = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(lfmt_buf))
        var lrope = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(lrope_buf))
        for li in range(n_layers):
            ldim[unsafe_offset=li] = default_dim
            lfmt[unsafe_offset=li] = VFMT_INT8
            lrope[unsafe_offset=li] = 0

        for li in range(n_layers):
            var spec = tokens[unsafe_offset=schema_at + 2 + li]
            var sl = Int(spec.length)
            var sptr = spec.ptr
            # Walk comma-separated key=value pairs.
            var pos = 0
            while pos < sl:
                # Find end of next k=v segment
                var seg_end = pos
                while seg_end < sl and sptr[unsafe_offset=seg_end] != UInt8(44):  # ','
                    seg_end += 1
                var seg_len = seg_end - pos
                var seg = sptr.unsafe_offset(pos)
                # Key checks (longest first to avoid `s` matching `s` from a longer key).
                # rope= must come before s= because `s` is a prefix of nothing here, but
                # match-fmt vs match-rope vs match-dim ordering is independent.
                var off_fmt  = _spec_starts_with(seg, seg_len, "fmt")
                var off_dim  = _spec_starts_with(seg, seg_len, "dim")
                var off_rope = _spec_starts_with(seg, seg_len, "rope")
                var off_s    = _spec_starts_with(seg, seg_len, "s")
                if off_fmt > 0:
                    var v_off = off_fmt
                    var v_len = seg_len - v_off
                    var v_ptr = seg.unsafe_offset(v_off)
                    var parsed_fmt = VFMT_INT8
                    var unsup = False
                    var reserved = False
                    var got = _parse_fmt_token(v_ptr, v_len, parsed_fmt, unsup, reserved)
                    if got:
                        lfmt[unsafe_offset=li] = parsed_fmt
                    else:
                        ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                        writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " unknown fmt")
                        return 1
                elif off_dim > 0:
                    # Parse decimal int into a temp String for atol.
                    var ds = String("")
                    for k in range(seg_len - off_dim):
                        ds += chr(Int(seg[unsafe_offset=off_dim + k]))
                    var dv = atol(ds)
                    if dv <= 0 or dv > 8192:
                        ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                        writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " dim must be 1-8192")
                        return 1
                    ldim[unsafe_offset=li] = Int(dv)
                elif off_rope > 0:
                    # A2: rope=N — number of leading dims stored as BF16 in
                    # the bf16_rope_fp8_body hybrid layout. Ignored when fmt
                    # is anything else (still parsed for consistency).
                    var rs = String("")
                    for k in range(seg_len - off_rope):
                        rs += chr(Int(seg[unsafe_offset=off_rope + k]))
                    var rv = atol(rs)
                    if rv <= 0 or rv > 8192:
                        ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                        writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " rope must be 1-8192")
                        return 1
                    lrope[unsafe_offset=li] = Int(rv)
                elif off_s > 0:
                    # D3: `s=` (third per-layer tensor) reserved syntactically,
                    # refused at runtime in v0.1. See doc §5.
                    ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                    writer.append_error_response("ERR V.CREATE SCHEMA: S-tensor reserved (route SSM/SWA-tail state through STATE.*)")
                    return 1
                else:
                    ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                    writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " unknown key")
                    return 1
                pos = seg_end + 1  # skip ','

            # Per-layer dim must be set somewhere (either default or explicit).
            if ldim[unsafe_offset=li] <= 0:
                ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " missing dim and no default supplied")
                return 1
            # Per-layer dim must be valid for chosen format
            if (lfmt[unsafe_offset=li] == VFMT_TURBO4 or lfmt[unsafe_offset=li] == VFMT_TURBO3 or lfmt[unsafe_offset=li] == VFMT_TURBO2 or lfmt[unsafe_offset=li] == VFMT_FP8) and (ldim[unsafe_offset=li] % 32 != 0):
                ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " turbo/fp8 dim must be multiple of 32")
                return 1
            # A2: hybrid validation — rope_dim required, in-range, body
            # divisible by 32.
            if lfmt[unsafe_offset=li] == VFMT_BF16_ROPE_FP8:
                if lrope[unsafe_offset=li] <= 0 or lrope[unsafe_offset=li] >= ldim[unsafe_offset=li]:
                    ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                    writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " bf16_rope_fp8_body needs rope=N (0 < N < dim)")
                    return 1
                if (ldim[unsafe_offset=li] - lrope[unsafe_offset=li]) % 32 != 0:
                    ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
                    writer.append_error_response("ERR V.CREATE SCHEMA: layer " + String(li) + " bf16_rope_fp8_body body (dim - rope) must be multiple of 32")
                    return 1

        var idx = vstore.create_session_schema(sid_ptr, sid_len, default_dim, n_layers, ldim, lfmt, lrope)
        if idx < 0:
            ldim_buf.unsafe_free(); lfmt_buf.unsafe_free(); lrope_buf.unsafe_free()
            writer.append_error_response("ERR V.CREATE SCHEMA failed (no free slots)")
            return 1
        vstore.wal_append_create_schema(sid_ptr, sid_len, default_dim, n_layers, ldim, lfmt, lrope)
        ldim_buf.unsafe_free()
        lfmt_buf.unsafe_free()
        lrope_buf.unsafe_free()
        writer.append_int_response(Int64(idx))
        return 1

    # ── Legacy uniform form ──────────────────────────────────────────────
    var dim_tok = tokens[unsafe_offset=start + 2]
    var val_dim = atol(dim_tok.value())
    if val_dim <= 0 or val_dim > 8192:
        writer.append_error_response("ERR invalid value_dim (must be 1-8192)")
        return 1

    # Parse optional VQUANT
    var v_format = VFMT_INT8
    var j = start + 3
    while j + 1 < num_tokens:
        var kw_tok = tokens[unsafe_offset=j]
        var kw_len = Int(kw_tok.length)
        var kw_ptr = kw_tok.ptr
        # VQUANT (6 bytes: v=118,q=113,u=117,a=97,n=110,t=116)
        if kw_len == 6 and (kw_ptr[unsafe_offset=0] | 0x20) == 118 and (kw_ptr[unsafe_offset=1] | 0x20) == 113:
            var fmt_tok = tokens[unsafe_offset=j + 1]
            # Case-insensitive on the VALUE, strict on the SET. `FP16` used to
            # fall through to the int8 default exactly like a typo did, so a
            # client asking for fp16 in the wrong case silently got a lossier
            # format. Lenient about case, loud about anything unrecognised.
            var fmt_str = fmt_tok.text_value().lower()
            if fmt_str == "turbo4":
                v_format = VFMT_TURBO4
            elif fmt_str == "turbo3":
                v_format = VFMT_TURBO3
            elif fmt_str == "turbo2":
                v_format = VFMT_TURBO2
            elif fmt_str == "fp16":
                v_format = VFMT_FP16
            elif fmt_str == "mlx4g32":
                v_format = VFMT_MLX4G32
            elif fmt_str == "fp8":
                v_format = VFMT_FP8
            elif fmt_str == "bf16_rope_fp8_body":
                # Hybrid format requires per-layer rope_dim — only expressible
                # via the SCHEMA form. Refuse the legacy uniform path.
                writer.append_error_response("ERR V.CREATE: bf16_rope_fp8_body requires SCHEMA form with per-layer rope=N")
                return 1
            elif fmt_str == "int8":
                v_format = VFMT_INT8
            else:
                # There was no else here, so an unrecognised format kept the
                # VFMT_INT8 default and V.CREATE answered +OK: `VQUANT
                # notarealtier`, `garbage!!` and `int9` all silently stored
                # int8. A client that typos `fp16` therefore gets a lossier
                # format than it asked for, with no error and no way to tell —
                # the same silent-substitution class gh #148 fixed once
                # already on the uniform path. A KV tier is a fidelity
                # contract; refuse rather than guess.
                writer.append_error_response(
                    "ERR V.CREATE: unknown VQUANT format '" + fmt_str
                    + "' (want int8|turbo4|turbo3|turbo2|fp16|fp8|mlx4g32)")
                return 1
            j += 2
        else:
            j += 1

    var idx = vstore.create_session(sid_ptr, sid_len, Int(val_dim), v_format)
    if idx < 0:
        writer.append_error_response("ERR V.CREATE failed (no free slots)")
        return 1
    vstore.wal_append_create(sid_ptr, sid_len, Int(val_dim), v_format)

    writer.append_int_response(Int64(idx))
    return 1


@always_inline
def handle_v_storebatch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """V.STOREBATCH <session_id> <layer_id> <start_token_id> <num_tokens> <values_blob> [FMT F16]

    Store a batch of V vectors. values_blob = num_tokens * value_dim * 4 bytes
    of fp32, or * 2 bytes of fp16 with FMT F16.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1

    if start + 5 >= num_tokens:
        writer.append_error_response("ERR V.STOREBATCH requires: session_id layer_id start_id count values_blob")
        return 1

    # Parse session
    var sid_tok = tokens[unsafe_offset=start + 1]
    var session_idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
    if session_idx < 0:
        # Cross-worker awareness: if the session lives on another worker
        # consult the directory and return -MOVED-style hint so the client
        # can reconnect to the right worker.
        if is_not_null(vstore.directory):
            var owner = vstore_dir_lookup(vstore.directory, sid_tok.ptr, Int(sid_tok.length))
            if owner >= 0 and owner != vstore.my_worker_id:
                writer.append_error_response(
                    "ERR session lives on worker " + String(owner) +
                    " (this is worker " + String(vstore.my_worker_id) +
                    "); reconnect for V.STOREBATCH")
                return 1
        writer.append_error_response("ERR session not found")
        return 1

    # Parse layer_id, start_id, count
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    var start_id = strict_atol(tokens[unsafe_offset=start + 3].value())
    var count = strict_atol(tokens[unsafe_offset=start + 4].value())

    if layer_id < 0 or start_id < 0 or count <= 0:
        writer.append_error_response("ERR invalid layer_id/start_id/count")
        return 1
    if Int(layer_id) >= MAX_VS_LAYERS:
        writer.append_error_response("ERR V.STOREBATCH: layer_id exceeds MAX_VS_LAYERS")
        return 1

    # A1: per-layer dim is the source of truth. Strict size validation:
    # refuse both undersized AND oversized blobs — silent acceptance of oversize was a corruption
    # surface for SCHEMA sessions where the consumer might pad to the wrong
    # default dim.
    var slot_idx = session_idx * MAX_VS_LAYERS + Int(layer_id)
    var per_layer_dim = vstore.layer_value_dim[unsafe_offset=slot_idx]
    if per_layer_dim <= 0:
        per_layer_dim = vstore.sessions[unsafe_offset=session_idx].value_dim

    # Parse values blob: fp32 by default, fp16 with a trailing `FMT F16`,
    # which halves the wire for K/V that is fp16 in the model and stored as
    # fp16. The format is declared, never inferred from the size: a half-dim
    # fp32 blob is exactly as long as a full-dim fp16 one, and accepting it
    # would reopen the silent-corruption surface strict sizing closed (A1).
    # `num_tokens` is this command's own end (cmd_end_tok), so the option scan
    # cannot read a pipelined neighbour.
    var val_tok = tokens[unsafe_offset=start + 5]
    var val_ptr = val_tok.ptr
    var val_len = Int(val_tok.length)
    var fp16_in = False
    if start + 7 < num_tokens:
        var kw = tokens[unsafe_offset=start + 6]
        var fv = tokens[unsafe_offset=start + 7]
        if arg_eq(kw.ptr, Int(kw.length), "fmt") and arg_eq(fv.ptr, Int(fv.length), "f16"):
            fp16_in = True
        else:
            writer.append_error_response("ERR V.STOREBATCH: unknown option (expected FMT F16)")
            return 1
    elif start + 6 < num_tokens:
        writer.append_error_response("ERR V.STOREBATCH: unknown option (expected FMT F16)")
        return 1
    var n_vals = Int(count) * per_layer_dim
    var elem = 2 if fp16_in else 4
    var expected = n_vals * elem
    if val_len != expected:
        writer.append_error_response("ERR V.STOREBATCH: blob size " + String(val_len) + " != expected " + String(expected) + " (count=" + String(Int(count)) + " × per_layer_dim=" + String(per_layer_dim) + " × " + String(elem) + ")")
        return 1

    var widened = alloc[Float32](n_vals if fp16_in else 1)
    if fp16_in:
        var src16 = val_ptr.bitcast[Float16]()
        for vi in range(n_vals):
            widened[vi] = src16[vi].cast[DType.float32]()
    var fp32_addr = Int(widened) if fp16_in else Int(val_ptr.bitcast[Float32]())
    var fp32_ext = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=fp32_addr)

    var ok = vstore.store_batch(session_idx, Int(layer_id), Int(start_id), Int(count), fp32_ext)
    if ok:
        # Append AFTER quantize+store succeeded. Note: WAL append happens before
        # flush_response below — we don't fsync per record (group commit via OS
        # page cache). KV.PREFIX.SAVE forces a snapshot + WAL truncate when
        # callers need bounded recovery loss.
        # An fp16-stored layer logs the fp16 values it actually stores: half
        # the bytes of the fp32 record, and replay (fp16 -> fp32 -> fp16) is
        # bit-exact. Other formats log fp32 and replay re-quantizes it.
        if vstore.v_fmt[unsafe_offset=slot_idx] == VFMT_FP16:
            vstore.wal_append_storebatch_f16(
                sid_tok.ptr, Int(sid_tok.length), Int(layer_id), Int(start_id),
                Int(count), per_layer_dim, fp32_ext)
        else:
            vstore.wal_append_storebatch(
                sid_tok.ptr, Int(sid_tok.length), Int(layer_id), Int(start_id),
                Int(count), per_layer_dim, fp32_ext)
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR V.STOREBATCH failed")
    widened.free()

    return 1


@always_inline
def handle_v_fetch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
    mut ledger: ValueLedger,
) raises -> Int:
    """V.FETCH <session_id> <layer_id> <token_id_1> [token_id_2 ...]
       V.FETCH <session_id> <layer_id> RANGE <start_id> <end_id_exclusive> [FMT NATIVE]
       V.FETCH <session_id> BATCH <start_id> <end_id_exclusive> [num_layers] [FMT NATIVE]

    Returns concatenated dequantized FP32 values for requested token IDs.
    Response: bulk string of num_ids * value_dim * 4 bytes (legacy/RANGE) or
    a RESP array of one bulk string per layer (BATCH form).

    gh #193 FMT NATIVE: fp16-stored layers reply with the raw stored fp16
    bytes (num_ids * value_dim * 2) — no fp32 expansion, half the wire.
    Non-fp16 formats ignore the flag and reply fp32; clients disambiguate by
    payload length. Contiguous forms only (RANGE/BATCH).

    The RANGE form fits any prefix length in 5 RESP tokens, sidestepping the
    RESP token-array frame limit. Used by KV-prefix cache fetches.

    The BATCH form returns ALL active layers (or `num_layers` if specified)
    in a single wire round-trip — replaces N×V.FETCH RANGE calls that
    PionPromptCache today does sequentially per layer (16 layers × K + V =
    32 round-trips on Llama-1B). At short prefixes this is the dominant
    fetch cost.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1

    if start + 3 >= num_tokens:
        writer.append_error_response("ERR V.FETCH requires: session_id layer_id token_id [...]")
        return 1

    # Parse session
    var sid_tok = tokens[unsafe_offset=start + 1]
    var session_idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
    if session_idx < 0:
        # Cross-worker awareness — see V.STOREBATCH above.
        if is_not_null(vstore.directory):
            var owner = vstore_dir_lookup(vstore.directory, sid_tok.ptr, Int(sid_tok.length))
            if owner >= 0 and owner != vstore.my_worker_id:
                writer.append_error_response(
                    "ERR session lives on worker " + String(owner) +
                    " (this is worker " + String(vstore.my_worker_id) +
                    "); reconnect for V.FETCH")
                return 1
        writer.append_error_response("ERR session not found")
        return 1

    # ── gh #193: optional trailing `FMT NATIVE` ──────────────────────────
    # fp16-stored layers reply with raw fp16 bytes (half the wire) instead of
    # the fp32 expansion; every other format falls back to fp32 unchanged.
    # The scan is bounded by this command's tokens (call site passes
    # cmd_end_tok), so it can never walk into a pipelined neighbor.
    var fmt_native = False
    var fmt_pos = -1
    for j in range(start + 2, num_tokens - 1):
        var ft = tokens[unsafe_offset=j]
        if Int(ft.length) == 3:
            var fp = ft.ptr
            if (fp[unsafe_offset=0] | 0x20) == 102 and (fp[unsafe_offset=1] | 0x20) == 109 and (fp[unsafe_offset=2] | 0x20) == 116:  # "fmt"
                var nt = tokens[unsafe_offset=j + 1]
                if Int(nt.length) == 6:
                    var np = nt.ptr
                    if (np[unsafe_offset=0] | 0x20) == 110 and (np[unsafe_offset=1] | 0x20) == 97 and (np[unsafe_offset=2] | 0x20) == 116 \
                       and (np[unsafe_offset=3] | 0x20) == 105 and (np[unsafe_offset=4] | 0x20) == 118 and (np[unsafe_offset=5] | 0x20) == 101:  # "native"
                        fmt_native = True
                        fmt_pos = j
                        break

    # ── BATCH subcommand: multi-layer fetch in one wire reply ────────────
    # tokens[start+2] = "BATCH" (5 bytes, case-insensitive) selects this form.
    var second = tokens[unsafe_offset=start + 2]
    var is_batch = False
    if Int(second.length) == 5:
        var p = second.ptr
        # b=98/66, a=97/65, t=116/84, c=99/67, h=104/72
        if (p[unsafe_offset=0] | 0x20) == 98 and (p[unsafe_offset=1] | 0x20) == 97 and (p[unsafe_offset=2] | 0x20) == 116 and (p[unsafe_offset=3] | 0x20) == 99 and (p[unsafe_offset=4] | 0x20) == 104:
            is_batch = True

    if is_batch:
        # tokens[start+3] = start_id, [start+4] = end_id, [start+5] optional num_layers
        if start + 4 >= num_tokens:
            writer.append_error_response("ERR V.FETCH BATCH requires: session_id BATCH start end [num_layers]")
            return 1
        var rs = strict_atol(tokens[unsafe_offset=start + 3].value())
        var re = strict_atol(tokens[unsafe_offset=start + 4].value())
        if rs < 0 or re <= rs:
            writer.append_error_response("ERR V.FETCH BATCH invalid range: end must exceed start (>=0)")
            return 1
        var b_range_start = Int(rs)
        var b_num_ids = Int(re - rs)

        # Determine layer count: explicit if provided, else session's num_layers.
        # gh #193: the slot may instead hold the FMT keyword — only parse a
        # token that starts with a digit (atol raises on non-numeric).
        var b_layers = vstore.sessions[unsafe_offset=session_idx].num_layers
        if start + 5 < num_tokens and start + 5 != fmt_pos and Int(tokens[unsafe_offset=start + 5].length) > 0 \
           and tokens[unsafe_offset=start + 5].ptr[unsafe_offset=0] >= 48 and tokens[unsafe_offset=start + 5].ptr[unsafe_offset=0] <= 57:
            var nl = strict_atol(tokens[unsafe_offset=start + 5].value())
            if nl > 0 and Int(nl) <= b_layers:
                b_layers = Int(nl)
        if b_layers <= 0:
            writer.append_error_response("ERR V.FETCH BATCH: session has no stored layers")
            return 1

        # Pre-build the RESP array header with the layer count.
        var hdr_buf = String("*") + String(b_layers) + String("\r\n")
        writer.append_to_response(hdr_buf.unsafe_ptr(), hdr_buf.byte_length())

        # Single token-id buffer reused across layers (RANGE is contiguous so
        # the IDs are identical for every layer).
        var b_ids_alloc = alloc[Int32](b_num_ids)
        var b_ids = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(b_ids_alloc))
        for j in range(b_num_ids):
            b_ids[unsafe_offset=j] = Int32(b_range_start + j)

        # A1: per-layer dim — buffer sized to the MAX across scanned layers,
        # then per-iteration writev uses the actual per-layer byte count.
        var b_max_dim = 0
        for li in range(b_layers):
            var ld = vstore.layer_value_dim[unsafe_offset=session_idx * MAX_VS_LAYERS + li]
            if ld <= 0:
                ld = vstore.sessions[unsafe_offset=session_idx].value_dim
            if ld > b_max_dim:
                b_max_dim = ld
        var b_out_size = b_num_ids * b_max_dim
        var b_out_alloc = alloc[Float32](b_out_size)
        var b_out_fp32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(b_out_alloc))

        for li in range(b_layers):
            var li_dim = vstore.layer_value_dim[unsafe_offset=session_idx * MAX_VS_LAYERS + li]
            if li_dim <= 0:
                li_dim = vstore.sessions[unsafe_offset=session_idx].value_dim
            # gh #193: fp16 layers go out raw under FMT NATIVE (fits in the
            # fp32-sized scratch buffer — half the bytes). Other formats keep
            # the fp32 expansion; the client disambiguates by payload length.
            if fmt_native and vstore.v_fmt[unsafe_offset=session_idx * MAX_VS_LAYERS + li] == VFMT_FP16:
                var out16 = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(b_out_fp32.bitcast[UInt8]()))
                var n_live = vstore.fetch_range_fp16_raw(session_idx, li, b_range_start, b_num_ids, out16)
                if n_live > 0:
                    writer.append_bulk_bytes_writev(fd, out16, b_num_ids * li_dim * 2)
                    ledger.record_kvprefix_bytes(b_num_ids * li_dim * 2)
                else:
                    writer.append_null_response()
            else:
                var n_fetched = vstore.fetch_tokens(session_idx, li, b_ids, b_num_ids, b_out_fp32)
                if n_fetched > 0:
                    var out_u8 = b_out_fp32.bitcast[UInt8]()
                    var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
                    var li_bytes = b_num_ids * li_dim * 4
                    # Use writev-aware bulk helper — single layer can exceed
                    # RESP_BUF_SIZE at long prefixes (same overflow class that
                    # bit V.FETCH RANGE before commit 10ad333).
                    writer.append_bulk_bytes_writev(fd, out_ext, li_bytes)
                    ledger.record_kvprefix_bytes(li_bytes)
                else:
                    # Layer absent — emit a $-1 nil bulk to keep the array
                    # length aligned. Caller can check per-layer.
                    writer.append_null_response()

        b_ids.unsafe_free()
        b_out_fp32.unsafe_free()
        return 1

    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())

    # Detect RANGE form: third arg is the literal "RANGE" (5 bytes).
    var third = tokens[unsafe_offset=start + 3]
    var is_range = False
    if Int(third.length) == 5:
        var p = third.ptr
        # case-insensitive: r=114/82, a=97/65, n=110/78, g=103/71, e=101/69
        if (p[unsafe_offset=0] | 0x20) == 114 and (p[unsafe_offset=1] | 0x20) == 97 and (p[unsafe_offset=2] | 0x20) == 110 and (p[unsafe_offset=3] | 0x20) == 103 and (p[unsafe_offset=4] | 0x20) == 101:
            is_range = True

    var num_ids = 0
    var id_start = start + 3
    var range_start = 0

    if is_range:
        if start + 5 >= num_tokens:
            writer.append_error_response("ERR V.FETCH RANGE requires: session_id layer_id RANGE start end")
            return 1
        var rs = strict_atol(tokens[unsafe_offset=start + 4].value())
        var re = strict_atol(tokens[unsafe_offset=start + 5].value())
        if rs < 0 or re <= rs:
            writer.append_error_response("ERR invalid RANGE: end must exceed start (>=0)")
            return 1
        range_start = Int(rs)
        num_ids = Int(re - rs)
    else:
        if fmt_native:
            # gh #193: only contiguous forms can memcpy raw fp16.
            writer.append_error_response("ERR FMT NATIVE requires the RANGE or BATCH form")
            return 1
        # Legacy: count remaining tokens as IDs.
        while id_start + num_ids < num_tokens:
            var t = tokens[unsafe_offset=id_start + num_ids]
            if Int(t.length) == 0:
                break
            num_ids += 1
            if num_ids >= 1024:
                break
        if num_ids == 0:
            writer.append_error_response("ERR no token IDs provided")
            return 1

    # Parse token IDs into array
    var _ids = alloc[Int32](num_ids)
    var ids = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_ids))
    if is_range:
        for i in range(num_ids):
            ids[unsafe_offset=i] = Int32(range_start + i)
    else:
        for i in range(num_ids):
            ids[unsafe_offset=i] = Int32(strict_atol(tokens[unsafe_offset=id_start + i].value()))

    # Allocate output buffer (per-layer dim — A1)
    var val_dim = vstore.layer_value_dim[unsafe_offset=session_idx * MAX_VS_LAYERS + Int(layer_id)]
    if val_dim <= 0:
        val_dim = vstore.sessions[unsafe_offset=session_idx].value_dim

    # gh #193 FMT NATIVE fast path (RANGE form only — legacy errored above):
    # raw fp16 straight from the store, half the wire bytes, no expansion.
    # Non-fp16 formats fall through to the fp32 path; the client tells the
    # two apart by payload length.
    if fmt_native and vstore.v_fmt[unsafe_offset=session_idx * MAX_VS_LAYERS + Int(layer_id)] == VFMT_FP16:
        var nat_bytes = num_ids * val_dim * 2
        var _o16 = alloc[UInt8](nat_bytes)
        var o16 = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_o16))
        var n_live = vstore.fetch_range_fp16_raw(session_idx, Int(layer_id), range_start, num_ids, o16)
        if n_live > 0:
            writer.append_bulk_bytes_writev(fd, o16, nat_bytes)
            ledger.record_kvprefix_bytes(nat_bytes)
        else:
            writer.append_null_response()
        o16.unsafe_free()
        _ids.unsafe_free()
        return 1

    var out_size = num_ids * val_dim
    var _out = alloc[Float32](out_size)
    var out_fp32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_out))

    var fetched = vstore.fetch_tokens(session_idx, Int(layer_id), ids, num_ids, out_fp32)

    if fetched > 0:
        var out_bytes = out_size * 4
        var out_u8 = out_fp32.bitcast[UInt8]()
        var out_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(out_u8))
        # writev for large responses: a 2K-token prefix at val_dim=512 already
        # produces 4 MB which overflows RESP_BUF_SIZE (also 4 MB) → heap
        # corruption → SIGSEGV. Route via writev when in danger of overflow;
        # the helper auto-selects the buffer path when it fits.
        writer.append_bulk_bytes_writev(fd, out_ext, out_bytes)
        ledger.record_kvprefix_bytes(out_bytes)
    else:
        writer.append_null_response()

    # Free temp buffers
    ids.unsafe_free()
    out_fp32.unsafe_free()

    return 1


@always_inline
def handle_v_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    vstore: VStoreIndex,
) raises -> Int:
    """V.INFO [session_id]

    Without args: global stats. With session_id: per-session details.
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1

    var info = String("")

    if start + 1 < num_tokens:
        # Per-session info
        var sid_tok = tokens[unsafe_offset=start + 1]
        var idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
        if idx < 0:
            writer.append_error_response("ERR session not found")
            return 1
        var meta = vstore.sessions[unsafe_offset=idx]
        var fmt_name: String
        if meta.v_format == VFMT_TURBO4: fmt_name = "turbo4"
        elif meta.v_format == VFMT_TURBO3: fmt_name = "turbo3"
        elif meta.v_format == VFMT_TURBO2: fmt_name = "turbo2"
        elif meta.v_format == VFMT_FP16: fmt_name = "fp16"
        elif meta.v_format == VFMT_MLX4G32: fmt_name = "mlx4g32"
        elif meta.v_format == VFMT_FP8: fmt_name = "fp8"
        elif meta.v_format == VFMT_BF16_ROPE_FP8: fmt_name = "bf16_rope_fp8_body"
        else: fmt_name = "int8"

        info += "session_idx:" + String(idx) + "\r\n"
        info += "value_dim:" + String(meta.value_dim) + "\r\n"
        info += "v_format:" + fmt_name + "\r\n"
        info += "num_layers:" + String(meta.num_layers) + "\r\n"
        # A1: detect heterogeneous schema (any layer disagrees with default).
        var heterogeneous = False
        for li in range(meta.num_layers):
            var li_slot = idx * MAX_VS_LAYERS + li
            if vstore.layer_value_dim[unsafe_offset=li_slot] != meta.value_dim:
                heterogeneous = True
                break
            if vstore.v_fmt[unsafe_offset=li_slot] != meta.v_format and vstore.tokens_per_layer[unsafe_offset=li_slot] > 0:
                heterogeneous = True
                break
        info += "schema:" + (String("heterogeneous") if heterogeneous else String("uniform")) + "\r\n"
        var total_tok = 0
        for li in range(meta.num_layers):
            var tpl = vstore.tokens_per_layer[unsafe_offset=idx * MAX_VS_LAYERS + li]
            total_tok += tpl
            if tpl > 0:
                info += "layer_" + String(li) + "_tokens:" + String(tpl) + "\r\n"
            if heterogeneous:
                var li_slot = idx * MAX_VS_LAYERS + li
                var ldim = vstore.layer_value_dim[unsafe_offset=li_slot]
                var lfmt = vstore.v_fmt[unsafe_offset=li_slot]
                var lfmt_name: String
                if lfmt == VFMT_TURBO4: lfmt_name = "turbo4"
                elif lfmt == VFMT_TURBO3: lfmt_name = "turbo3"
                elif lfmt == VFMT_TURBO2: lfmt_name = "turbo2"
                elif lfmt == VFMT_FP16: lfmt_name = "fp16"
                elif lfmt == VFMT_MLX4G32: lfmt_name = "mlx4g32"
                elif lfmt == VFMT_FP8: lfmt_name = "fp8"
                elif lfmt == VFMT_BF16_ROPE_FP8: lfmt_name = "bf16_rope_fp8_body"
                else: lfmt_name = "int8"
                info += "layer_" + String(li) + "_dim:" + String(ldim) + "\r\n"
                info += "layer_" + String(li) + "_fmt:" + lfmt_name + "\r\n"
                if lfmt == VFMT_BF16_ROPE_FP8:
                    info += "layer_" + String(li) + "_rope:" + String(vstore.layer_rope_dim[unsafe_offset=li_slot]) + "\r\n"
        info += "total_tokens:" + String(total_tok) + "\r\n"
    else:
        # Global info
        info += "enabled:1\r\n"
        info += "sessions:" + String(vstore.session_count) + "\r\n"
        info += "total_tokens:" + String(vstore.total_tokens) + "\r\n"
        info += "total_fetches:" + String(vstore.total_fetches) + "\r\n"

    var info_bytes = info.as_bytes()
    var info_ptr = info_bytes.unsafe_ptr()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(info_ptr))
    writer.append_bulk_string_response(info_ext, info.byte_length())

    return 1


# ── A10 (gh #37): V.SNAPSHOT / V.RESTORE / V.COMMIT ──────────────────────


@always_inline
def handle_v_snapshot(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """V.SNAPSHOT <session_id>  →  :<snap_id>

    Records current per-layer length. Cheap: O(MAX_VS_LAYERS) memcpy of an
    Int array. snap_id is a monotonic UInt64 — never reused.

    Returns -ERR on disabled, missing sid, or no free snapshot slot
    (MAX_VS_SNAPSHOTS_PER_SESSION = 16 concurrent per session)."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled (use --kvcache)")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR V.SNAPSHOT requires: session_id")
        return 1
    var sid_tok = tokens[unsafe_offset=start + 1]
    var session_idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
    if session_idx < 0:
        writer.append_error_response("ERR session not found")
        return 1
    var sid_value = vstore.snapshot_session(session_idx)
    if sid_value == UInt64(0):
        writer.append_error_response("ERR V.SNAPSHOT: no free snapshot slot (max 16 per session)")
        return 1
    writer.append_int_response(Int64(sid_value))
    return 1


@always_inline
def handle_v_restore(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """V.RESTORE <session_id> <snap_id>  →  +OK

    Truncates per-layer token counts back to the recorded snapshot. Buffer
    bytes are NOT freed — V-store buffers are sized at first STORE; subsequent
    writes overwrite the trailing portion. The snapshot stays valid (multi-restore
    against the same snap_id is allowed; useful for nested speculative branches).
    """
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR V.RESTORE requires: session_id snap_id")
        return 1
    var sid_tok = tokens[unsafe_offset=start + 1]
    var session_idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
    if session_idx < 0:
        writer.append_error_response("ERR session not found")
        return 1
    var snap_raw = strict_atol(tokens[unsafe_offset=start + 2].value())
    if snap_raw <= 0:
        writer.append_error_response("ERR V.RESTORE: snap_id must be positive")
        return 1
    var snap_id_v = UInt64(snap_raw)
    if not vstore.restore_session(session_idx, snap_id_v):
        writer.append_error_response("ERR V.RESTORE: unknown snap_id for this session")
        return 1
    writer.append_ok_response()
    return 1


@always_inline
def handle_v_commit(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut vstore: VStoreIndex,
) raises -> Int:
    """V.COMMIT <session_id> <snap_id>  →  :1 if released / :0 if absent

    Releases the snapshot slot. Subsequent V.RESTORE for that snap_id returns
    -ERR. Idempotent — calling COMMIT twice returns :0 the second time."""
    if not vstore.enabled:
        writer.append_error_response("ERR V-store not enabled")
        return 1
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR V.COMMIT requires: session_id snap_id")
        return 1
    var sid_tok = tokens[unsafe_offset=start + 1]
    var session_idx = vstore._find_session(sid_tok.ptr, Int(sid_tok.length))
    if session_idx < 0:
        writer.append_error_response("ERR session not found")
        return 1
    var snap_raw = strict_atol(tokens[unsafe_offset=start + 2].value())
    if snap_raw <= 0:
        writer.append_error_response("ERR V.COMMIT: snap_id must be positive")
        return 1
    var ok = vstore.commit_session_snapshot(session_idx, UInt64(snap_raw))
    writer.append_int_response(Int64(1) if ok else Int64(0))
    return 1
