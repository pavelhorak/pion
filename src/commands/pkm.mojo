"""NEURON.PKM.* wire handlers — product-key memory-layer serving (gh #146).

Store implementation: `src/network/pkm.mojo`. Kernels: `src/vector/pkm_kernels.mojo`.

    NEURON.PKM.CREATE  <table> <dim> <n_slots> [VDIM <v>] [VALTYPE F32|F16]
    NEURON.PKM.SETKEYS <table> <half:0|1> <blob>      # S × (dim/2) FP32 LE
    NEURON.PKM.SETVALS <table> <off> <n> <blob>       # n × vdim, VALTYPE-typed
    NEURON.PKM.QUERY   <table> <k> <q_blob> [FAST]    # → nq·k × (Int32 id, Float32 score)
    NEURON.PKM.FFN     <table> <k> <q_blob> [FAST] [TEMP <t>]   # → nq × vdim FP32
    NEURON.PKM.INFO    <table>
    NEURON.PKM.DROP    <table>

`n_slots` must be a perfect square S²; the two codebooks hold S rows each and
slot id is `i·S + j`. QUERY/FFN infer the head count from the blob:
`nq = len(q_blob) / (dim·4)`, so a multi-head memory layer sends all its heads
in one command and they share a single pass over the codebooks.

Replies for QUERY and FFN are packed binary bulk strings — same convention as
AI.KNN_LM.QUERY, unpacked client-side with numpy/struct:

    QUERY → nq·k × 8 bytes: Int32 LE slot id, Float32 LE score.
            Padding entries (fewer than k reachable slots) are (-1, -inf).
    FFN   → nq × vdim × 4 bytes: Float32 LE activations.

`FAST` selects on the INT8 codebook mirror and re-scores the survivors in FP32
— ~4× less key traffic, and every returned score is still an exact FP32 dot.
Without it the whole scan is FP32. Both are exact top-k in the product-key
sense (see the kernel file's header); FAST's only approximation is *which*
rows enter the candidate set, and the widened candidate width makes that a
non-event at any realistic score spread.
"""

from std.collections import Array
from std.memory.unsafe_pointer import Pointer

from src.network.pkm import (
    MAX_PKM_TABLES,
    PKMIndex,
    PKM_MAX_K,
    PKM_MAX_NQ,
    PKM_VT_F16,
    PKM_VT_F32,
)
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter


# FFN replies go through the 4 MB response buffer; keep a wide margin below it.
comptime PKM_MAX_FFN_BYTES = 3 * 1024 * 1024

comptime _PKM_GATE_ERR = "ERR NEURON.PKM.* requires --kvcache or --inference"


@always_inline
def _atol(s: String) raises -> Int64:
    return Int64(atol(s))


@always_inline
def _atof(s: String) raises -> Float64:
    return Float64(atof(s))


# Every handler takes `cmd_end` — the exclusive end token of THIS command —
# rather than the batch-wide token count. CREATE/QUERY/FFN scan trailing
# options in a loop, so an unbounded scan would read the next pipelined
# command's tokens as options and reject a perfectly good request.


@always_inline
def _tok_eq(tok: RESP3Token, lit: String) -> Bool:
    """Case-insensitive ASCII compare of a RESP token against a literal."""
    var n = Int(tok.length)
    var lb = lit.as_bytes()
    if n != len(lb):
        return False
    for i in range(n):
        var a = tok.ptr[unsafe_offset=i]
        if a >= 65 and a <= 90:
            a = a | 0x20
        var b = lb[i]
        if b >= 65 and b <= 90:
            b = b | 0x20
        if a != b:
            return False
    return True


@always_inline
def _name_ptr(tok: RESP3Token) -> Pointer[UInt8, MutUntrackedOrigin]:
    return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(tok.ptr))


@always_inline
def handle_neuron_pkm_create(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.CREATE <table> <dim> <n_slots> [VDIM <v>] [VALTYPE F32|F16]."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 3 >= cmd_end:
        writer.append_error_response(
            "ERR syntax: NEURON.PKM.CREATE <table> <dim> <n_slots> [VDIM <v>] [VALTYPE F32|F16]"
        )
        return 1

    var dim = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    var n_slots = Int(_atol(tokens[unsafe_offset=start + 3].value()))
    var vdim = 0
    var val_type = PKM_VT_F32

    var a = start + 4
    while a < cmd_end:
        if _tok_eq(tokens[unsafe_offset=a], "VDIM") and a + 1 < cmd_end:
            vdim = Int(_atol(tokens[unsafe_offset=a + 1].value()))
            a += 2
        elif _tok_eq(tokens[unsafe_offset=a], "VALTYPE") and a + 1 < cmd_end:
            if _tok_eq(tokens[unsafe_offset=a + 1], "F16"):
                val_type = PKM_VT_F16
            elif _tok_eq(tokens[unsafe_offset=a + 1], "F32"):
                val_type = PKM_VT_F32
            else:
                writer.append_error_response("ERR NEURON.PKM.CREATE: VALTYPE must be F32 or F16")
                return 1
            a += 2
        else:
            writer.append_error_response(
                "ERR NEURON.PKM.CREATE: unknown option '" + tokens[unsafe_offset=a].value() + "'"
            )
            return 1

    var slot = pkm.create(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length), dim, n_slots, vdim, val_type)
    if slot == -1:
        writer.append_error_response(
            "ERR NEURON.PKM.CREATE: duplicate table name or registry full (max "
            + String(MAX_PKM_TABLES) + ")"
        )
    elif slot == -2:
        writer.append_error_response("ERR NEURON.PKM.CREATE: table name must be 1..64 bytes")
    elif slot == -3:
        writer.append_error_response("ERR NEURON.PKM.CREATE: dim must be even and in [2, 8192]")
    elif slot == -4:
        writer.append_error_response(
            "ERR NEURON.PKM.CREATE: n_slots must be a perfect square S*S with S in [1, 4096] (got "
            + String(n_slots) + ")"
        )
    elif slot == -5:
        writer.append_error_response("ERR NEURON.PKM.CREATE: vdim must be in [0, 65536]")
    else:
        writer.append_ok_response()
    return 1


@always_inline
def handle_neuron_pkm_setkeys(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.SETKEYS <table> <half:0|1> <blob> — S × (dim/2) FP32 LE."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 3 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.SETKEYS <table> <half> <blob>")
        return 1
    var slot = pkm.find(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    if slot < 0:
        writer.append_error_response("ERR NEURON.PKM.SETKEYS: table not found (call NEURON.PKM.CREATE first)")
        return 1
    var half = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    if half != 0 and half != 1:
        writer.append_error_response("ERR NEURON.PKM.SETKEYS: half must be 0 or 1")
        return 1

    var blob = tokens[unsafe_offset=start + 3]
    var expected = pkm.tables[slot].s_rows * pkm.tables[slot].half * 4
    if Int(blob.length) != expected:
        writer.append_error_response(
            "ERR NEURON.PKM.SETKEYS: blob is " + String(Int(blob.length)) + " bytes, expected "
            + String(expected) + " (S=" + String(pkm.tables[slot].s_rows)
            + " × dim/2=" + String(pkm.tables[slot].half) + " × 4)"
        )
        return 1

    pkm.set_keys(
        slot,
        half,
        Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(blob.ptr.unsafe_bitcast[Float32]())),
    )
    writer.append_ok_response()
    return 1


@always_inline
def handle_neuron_pkm_setvals(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.SETVALS <table> <off> <n> <blob> — n × vdim value rows."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 4 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.SETVALS <table> <off> <n> <blob>")
        return 1
    var slot = pkm.find(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    if slot < 0:
        writer.append_error_response("ERR NEURON.PKM.SETVALS: table not found")
        return 1
    if pkm.tables[slot].vdim <= 0:
        writer.append_error_response("ERR NEURON.PKM.SETVALS: table was created without VDIM")
        return 1

    var off = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    var n = Int(_atol(tokens[unsafe_offset=start + 3].value()))
    if off < 0 or n <= 0 or off + n > pkm.tables[slot].n_slots:
        writer.append_error_response(
            "ERR NEURON.PKM.SETVALS: [off, off+n) must lie in [0, " + String(pkm.tables[slot].n_slots) + ")"
        )
        return 1

    var blob = tokens[unsafe_offset=start + 4]
    var expected = n * pkm.tables[slot].vdim * pkm.tables[slot].val_elt_bytes()
    if Int(blob.length) != expected:
        writer.append_error_response(
            "ERR NEURON.PKM.SETVALS: blob is " + String(Int(blob.length)) + " bytes, expected "
            + String(expected) + " (n=" + String(n) + " × vdim=" + String(pkm.tables[slot].vdim)
            + " × " + String(pkm.tables[slot].val_elt_bytes()) + ")"
        )
        return 1

    if not pkm.ensure_vals(slot):
        writer.append_error_response("ERR NEURON.PKM.SETVALS: value matrix allocation failed")
        return 1
    pkm.set_vals(slot, off, n, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(blob.ptr)))
    writer.append_int_response(Int64(n))
    return 1


@always_inline
def _parse_query_args(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    first: Int,
    cmd_end: Int,
    mut fast: Bool,
    mut temperature: Float32,
) raises -> Bool:
    """Parse the trailing [FAST] [TEMP <t>] options. False on an unknown token."""
    var a = first
    while a < cmd_end:
        if _tok_eq(tokens[unsafe_offset=a], "FAST"):
            fast = True
            a += 1
        elif _tok_eq(tokens[unsafe_offset=a], "TEMP") and a + 1 < cmd_end:
            temperature = Float32(_atof(tokens[unsafe_offset=a + 1].value()))
            a += 2
        else:
            return False
    return True


@always_inline
def handle_neuron_pkm_query(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.QUERY <table> <k> <q_blob> [FAST] → nq·k × (Int32 id, Float32 score)."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 3 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.QUERY <table> <k> <q_blob> [FAST]")
        return 1
    var slot = pkm.find(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    if slot < 0:
        writer.append_error_response("ERR NEURON.PKM.QUERY: table not found")
        return 1
    if not pkm.tables[slot].keys_ready():
        writer.append_error_response(
            "ERR NEURON.PKM.QUERY: codebooks incomplete — call NEURON.PKM.SETKEYS for half 0 and half 1"
        )
        return 1

    var k = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    if k <= 0 or k > PKM_MAX_K:
        writer.append_error_response("ERR NEURON.PKM.QUERY: k must be in (0, " + String(PKM_MAX_K) + "]")
        return 1

    var blob = tokens[unsafe_offset=start + 3]
    var dim = pkm.tables[slot].dim
    var row_bytes = dim * 4
    if Int(blob.length) == 0 or Int(blob.length) % row_bytes != 0:
        writer.append_error_response(
            "ERR NEURON.PKM.QUERY: query blob is " + String(Int(blob.length))
            + " bytes, must be a positive multiple of dim*4 = " + String(row_bytes)
        )
        return 1
    var nq = Int(blob.length) // row_bytes
    if nq > PKM_MAX_NQ:
        writer.append_error_response("ERR NEURON.PKM.QUERY: at most " + String(PKM_MAX_NQ) + " query heads per call")
        return 1

    var fast = False
    var temperature = Float32(1.0)
    if not _parse_query_args(tokens, start + 4, cmd_end, fast, temperature):
        writer.append_error_response("ERR NEURON.PKM.QUERY: unknown option (expected FAST)")
        return 1

    _ = pkm.query(
        slot,
        k,
        Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(blob.ptr.unsafe_bitcast[Float32]())),
        nq,
        fast,
    )

    # Pack nq·k × [Int32 id][Float32 score] into the preallocated reply buffer.
    var total = nq * k
    for i in range(total):
        var e = pkm.pack.unsafe_offset(i * 8)
        e.unsafe_bitcast[Int32]()[unsafe_offset=0] = pkm.res_id[unsafe_offset=i]
        (e.unsafe_offset(4)).unsafe_bitcast[Float32]()[unsafe_offset=0] = pkm.res_score[unsafe_offset=i]
    writer.append_bulk_string_response(pkm.pack, total * 8)
    return 1


@always_inline
def handle_neuron_pkm_ffn(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.FFN <table> <k> <q_blob> [FAST] [TEMP <t>] → nq × vdim FP32."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 3 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.FFN <table> <k> <q_blob> [FAST] [TEMP <t>]")
        return 1
    var slot = pkm.find(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    if slot < 0:
        writer.append_error_response("ERR NEURON.PKM.FFN: table not found")
        return 1
    if not pkm.tables[slot].keys_ready():
        writer.append_error_response(
            "ERR NEURON.PKM.FFN: codebooks incomplete — call NEURON.PKM.SETKEYS for half 0 and half 1"
        )
        return 1

    var k = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    if k <= 0 or k > PKM_MAX_K:
        writer.append_error_response("ERR NEURON.PKM.FFN: k must be in (0, " + String(PKM_MAX_K) + "]")
        return 1

    var blob = tokens[unsafe_offset=start + 3]
    var dim = pkm.tables[slot].dim
    var row_bytes = dim * 4
    if Int(blob.length) == 0 or Int(blob.length) % row_bytes != 0:
        writer.append_error_response(
            "ERR NEURON.PKM.FFN: query blob is " + String(Int(blob.length))
            + " bytes, must be a positive multiple of dim*4 = " + String(row_bytes)
        )
        return 1
    var nq = Int(blob.length) // row_bytes
    if nq > PKM_MAX_NQ:
        writer.append_error_response("ERR NEURON.PKM.FFN: at most " + String(PKM_MAX_NQ) + " query heads per call")
        return 1

    var vdim = pkm.tables[slot].vdim
    var out_bytes = nq * vdim * 4
    if out_bytes > PKM_MAX_FFN_BYTES:
        writer.append_error_response(
            "ERR NEURON.PKM.FFN: reply would be " + String(out_bytes) + " bytes, limit is "
            + String(PKM_MAX_FFN_BYTES) + " — send fewer heads per call"
        )
        return 1

    var fast = False
    var temperature = Float32(1.0)
    if not _parse_query_args(tokens, start + 4, cmd_end, fast, temperature):
        writer.append_error_response("ERR NEURON.PKM.FFN: unknown option (expected FAST or TEMP <t>)")
        return 1
    if temperature <= 0.0:
        writer.append_error_response("ERR NEURON.PKM.FFN: TEMP must be > 0")
        return 1

    var rc = pkm.ffn(
        slot,
        k,
        Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(blob.ptr.unsafe_bitcast[Float32]())),
        nq,
        fast,
        temperature,
    )
    if rc == -1:
        writer.append_error_response("ERR NEURON.PKM.FFN: table has no value rows (call NEURON.PKM.SETVALS first)")
        return 1
    if rc < 0:
        writer.append_error_response("ERR NEURON.PKM.FFN: activation buffer allocation failed")
        return 1

    writer.append_bulk_string_response(pkm.ffn_buf.unsafe_bitcast[UInt8](), out_bytes)
    return 1


@always_inline
def handle_neuron_pkm_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.INFO <table> → status string."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 1 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.INFO <table>")
        return 1
    var slot = pkm.find(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    if slot < 0:
        writer.append_error_response("ERR NEURON.PKM.INFO: table not found")
        return 1

    var info = String("dim=") + String(pkm.tables[slot].dim)
    info += " half=" + String(pkm.tables[slot].half)
    info += " s_rows=" + String(pkm.tables[slot].s_rows)
    info += " n_slots=" + String(pkm.tables[slot].n_slots)
    info += " d_pad=" + String(pkm.tables[slot].d_pad)
    info += " keys_ready=" + String(1 if pkm.tables[slot].keys_ready() else 0)
    info += " vdim=" + String(pkm.tables[slot].vdim)
    info += " valtype=" + ("f16" if pkm.tables[slot].val_type == PKM_VT_F16 else "f32")
    info += " val_rows=" + String(pkm.tables[slot].val_rows)
    info += " queries=" + String(pkm.tables[slot].queries)
    var bytes = info.as_bytes()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bytes.unsafe_ptr()))
    writer.append_bulk_string_response(info_ext, len(bytes))
    return 1


@always_inline
def handle_neuron_pkm_drop(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    cmd_end: Int,
    mut writer: ResponseWriter,
    mut pkm: PKMIndex,
) raises -> Int:
    """NEURON.PKM.DROP <table> → :1 if dropped, :0 if it wasn't there."""
    if not pkm.enabled:
        writer.append_error_response(_PKM_GATE_ERR)
        return 1
    if start + 1 >= cmd_end:
        writer.append_error_response("ERR syntax: NEURON.PKM.DROP <table>")
        return 1
    var ok = pkm.drop(_name_ptr(tokens[unsafe_offset=start + 1]), Int(tokens[unsafe_offset=start + 1].length))
    writer.append_int_response(Int64(1) if ok else Int64(0))
    return 1
