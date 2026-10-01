"""STATE.* command handlers — per-request fixed-size state cache.

Issue #31 (A4) + A12 fold-in. Wire surface:

  STATE.ALLOC  <sid> <size> [MODE fixed|ring]
  STATE.WRITE  <sid> <offset> <bytes>
  STATE.READ   <sid> <offset> <length>
  STATE.FREE   <sid>
  STATE.INFO   [sid]
"""
from src.common.utils import strict_atol

from src.network.state_store import (
    StateStore, STATE_MODE_FIXED, STATE_MODE_RING,
    MAX_STATE_SESSIONS, MAX_BUFFER_SIZE, MAX_TOTAL_BYTES,
)
from src.network.response_writer import ResponseWriter
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from std.collections import Array
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc


@always_inline
def _err_disabled(mut writer: ResponseWriter):
    writer.append_error_response("ERR state cache not enabled (use --kvcache)")


@always_inline
def handle_state_alloc(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut state: StateStore,
) raises -> Int:
    """STATE.ALLOC <sid> <size> [MODE fixed|ring]

    Returns +OK on success. Errors:
      -ERR sid already exists / -ERR invalid size / -ERR budget exhausted /
      -ERR no free slot.
    """
    if not state.enabled:
        _err_disabled(writer)
        return 1
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR STATE.ALLOC requires: sid size [MODE fixed|ring]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var size = strict_atol(tokens[unsafe_offset=start + 2].value())
    if size <= 0:
        writer.append_error_response("ERR STATE.ALLOC: size must be positive")
        return 1
    if size > MAX_BUFFER_SIZE:
        writer.append_error_response("ERR STATE.ALLOC: size exceeds per-session cap " + String(MAX_BUFFER_SIZE))
        return 1

    var mode = STATE_MODE_FIXED
    if start + 4 < num_tokens:
        var kw = tokens[unsafe_offset=start + 3]
        # MODE keyword (4 bytes: m=109,o=111,d=101,e=101 case-insensitive)
        if Int(kw.length) == 4 and (kw.ptr[unsafe_offset=0] | 0x20) == 109 and (kw.ptr[unsafe_offset=1] | 0x20) == 111 and (kw.ptr[unsafe_offset=2] | 0x20) == 100 and (kw.ptr[unsafe_offset=3] | 0x20) == 101:
            var mode_str = tokens[unsafe_offset=start + 4].value()
            if mode_str == "ring":
                mode = STATE_MODE_RING
            elif mode_str == "fixed":
                mode = STATE_MODE_FIXED
            else:
                writer.append_error_response("ERR STATE.ALLOC: MODE must be 'fixed' or 'ring'")
                return 1

    var slot = state.alloc_session(sid_tok.ptr, Int(sid_tok.length), Int(size), mode)
    if slot >= 0:
        writer.append_ok_response()
    elif slot == -2:
        writer.append_error_response("ERR STATE.ALLOC: sid already allocated (FREE first)")
    elif slot == -3:
        writer.append_error_response("ERR STATE.ALLOC: invalid size")
    elif slot == -4:
        writer.append_error_response("ERR STATE.ALLOC: total memory budget exhausted")
    elif slot == -5:
        writer.append_error_response("ERR STATE.ALLOC: no free session slot (max " + String(MAX_STATE_SESSIONS) + ")")
    else:
        writer.append_error_response("ERR STATE.ALLOC failed")
    return 1


@always_inline
def handle_state_write(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut state: StateStore,
) raises -> Int:
    """STATE.WRITE <sid> <offset> <bytes>

    Returns :<bytes_written>. Errors on inactive sid, fixed-mode overflow,
    or zero-length payload.
    """
    if not state.enabled:
        _err_disabled(writer)
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR STATE.WRITE requires: sid offset bytes")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var slot = state._find_session(sid_tok.ptr, Int(sid_tok.length))
    if slot < 0:
        writer.append_error_response("ERR STATE.WRITE: sid not allocated")
        return 1

    var offset = strict_atol(tokens[unsafe_offset=start + 2].value())
    var blob = tokens[unsafe_offset=start + 3]
    var length = Int(blob.length)
    var src = blob.ptr

    var rc = state.write_bytes(slot, Int(offset), src, length)
    if rc >= 0:
        writer.append_int_response(Int64(rc))
    elif rc == -2:
        writer.append_error_response("ERR STATE.WRITE: offset must be non-negative")
    elif rc == -3:
        writer.append_error_response("ERR STATE.WRITE: offset+length out of range (fixed mode)")
    elif rc == -4:
        writer.append_error_response("ERR STATE.WRITE: payload empty")
    else:
        writer.append_error_response("ERR STATE.WRITE failed")
    return 1


@always_inline
def handle_state_read(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut state: StateStore,
) raises -> Int:
    """STATE.READ <sid> <offset> <length>

    Returns the requested bytes as a bulk string, or -ERR on overflow / bad
    sid / bad args.
    """
    if not state.enabled:
        _err_disabled(writer)
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR STATE.READ requires: sid offset length")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var slot = state._find_session(sid_tok.ptr, Int(sid_tok.length))
    if slot < 0:
        writer.append_error_response("ERR STATE.READ: sid not allocated")
        return 1

    var offset = strict_atol(tokens[unsafe_offset=start + 2].value())
    var length = strict_atol(tokens[unsafe_offset=start + 3].value())
    if length <= 0:
        writer.append_error_response("ERR STATE.READ: length must be positive")
        return 1
    # Single-allocation cap on read size — the response buffer is 4 MB; we
    # cap reads at the same ceiling so STATE.READ can never blow past the
    # writer in a single shot. Larger reads should be split by the caller.
    if length > 4 * 1024 * 1024:
        writer.append_error_response("ERR STATE.READ: length exceeds 4MB single-read cap")
        return 1

    var _dst = alloc[UInt8](Int(length))
    var dst = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_dst))
    var rc = state.read_bytes(slot, Int(offset), Int(length), dst)
    if rc >= 0:
        writer.append_bulk_string_response(dst, Int(length))
    elif rc == -2:
        writer.append_error_response("ERR STATE.READ: offset must be non-negative")
    elif rc == -3:
        writer.append_error_response("ERR STATE.READ: offset+length out of range (fixed mode)")
    else:
        writer.append_error_response("ERR STATE.READ failed")
    dst.unsafe_free()
    return 1


@always_inline
def handle_state_free(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut state: StateStore,
) raises -> Int:
    """STATE.FREE <sid>

    Returns :1 if released, :0 if no such sid.
    """
    if not state.enabled:
        _err_disabled(writer)
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR STATE.FREE requires: sid")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var slot = state._find_session(sid_tok.ptr, Int(sid_tok.length))
    if slot < 0:
        writer.append_int_response(Int64(0))
        return 1
    var ok = state.free_session(slot)
    writer.append_int_response(Int64(1) if ok else Int64(0))
    return 1


@always_inline
def handle_state_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    state: StateStore,
) raises -> Int:
    """STATE.INFO [sid]

    Without args: global stats. With sid: per-session details.
    """
    if not state.enabled:
        _err_disabled(writer)
        return 1

    var info = String("")
    if start + 1 < num_tokens and Int(tokens[unsafe_offset=start + 1].length) > 0:
        var sid_tok = tokens[unsafe_offset=start + 1]
        var slot = state._find_session(sid_tok.ptr, Int(sid_tok.length))
        if slot < 0:
            writer.append_error_response("ERR STATE.INFO: sid not allocated")
            return 1
        var s = state.sessions[unsafe_offset=slot]
        var mode_str: String
        if s.mode == STATE_MODE_RING:
            mode_str = "ring"
        else:
            mode_str = "fixed"
        info += "slot:" + String(slot) + "\r\n"
        info += "size:" + String(s.size) + "\r\n"
        info += "mode:" + mode_str + "\r\n"
        info += "bytes_written:" + String(s.bytes_written) + "\r\n"
        info += "bytes_read:" + String(s.bytes_read) + "\r\n"
    else:
        info += "enabled:1\r\n"
        info += "sessions:" + String(state.session_count) + "\r\n"
        info += "max_sessions:" + String(MAX_STATE_SESSIONS) + "\r\n"
        info += "total_bytes:" + String(state.total_bytes_allocated) + "\r\n"
        info += "max_total_bytes:" + String(MAX_TOTAL_BYTES) + "\r\n"
        info += "max_buffer_size:" + String(MAX_BUFFER_SIZE) + "\r\n"
        info += "total_allocs:" + String(state.total_allocs) + "\r\n"
        info += "total_frees:" + String(state.total_frees) + "\r\n"
        info += "total_writes:" + String(state.total_writes) + "\r\n"
        info += "total_reads:" + String(state.total_reads) + "\r\n"

    var info_bytes = info.as_bytes()
    var info_ptr = info_bytes.unsafe_ptr()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(info_ptr))
    writer.append_bulk_string_response(info_ext, info.byte_length())
    return 1
