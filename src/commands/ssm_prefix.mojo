"""SSM.PREFIX.STORE / FETCH / DROP — opaque byte-blob storage for SSM state.

gh #65: server-side substrate for prefix-sharing on state-space models (Mamba,
RWKV, RetNet, etc.). The server stores byte blobs keyed by (session_id,
layer_id); the consumer serializes per-layer recurrent state (e.g. Mamba's
[conv_state, ssm_state] arrays) into a blob, ships it, fetches on a new
request that shares the prefix.

The server NEVER parses the blob. This keeps the substrate model-family-
agnostic — each consumer chooses its own serialization. Recommended:

    uint32 version
    uint32 n_arrays
    for each array: uint32 ndim, uint32[ndim] shape, uint32 dtype_code, raw bytes

Companion to KV.PREFIX.* (the transformer-attention substrate). Together they
let Pion prefix-share both kinds of per-layer state on hybrid models,
validated bit-perfect against the in-process path (gh #61, gh #64).

Wire surface:
    SSM.PREFIX.STORE <session_id> <layer_id> <state_blob>
    SSM.PREFIX.FETCH <session_id> <layer_id>
    SSM.PREFIX.DROP  <session_id> [<layer_id>]    # layer_id omitted = drop all
"""
from src.common.utils import strict_atol

from src.common.ptr import null_ptr
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.sys.info import CompilationTarget

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter

from std.collections import Array


@always_inline
def ssm_init() -> Bool:
    """Idempotent worker-local init. Safe to call multiple times."""
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        var rc = external_call["pion_ssm_init", Int32]()
        return rc == 0
    else:
        return False


def ssm_durability_startup(worker_id: Int) -> None:
    """gh #94 — per-worker SSM.PREFIX.* durability bootstrap.

    Sequence (mirrors the V-store path at slow_path.mojo:284):
      1. pion_ssm_init() — ensure slot tables zeroed.
      2. pion_ssm_load_snapshot("pion.ssm.<wid>")  — restore last KV.PREFIX.SAVE.
      3. pion_ssm_wal_replay("pion.ssm.wal.<wid>") — replay post-snapshot ops.
      4. pion_ssm_wal_open("pion.ssm.wal.<wid>")   — open for ongoing append.

    Steps 2-4 are best-effort: missing files mean cold start, file-IO errors
    leave the substrate operational (durability is degraded, not correctness).
    """
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        _ = ssm_init()
        var snap_path = "pion.ssm." + String(worker_id) + "\0"
        var wal_path  = "pion.ssm.wal." + String(worker_id) + "\0"
        _ = external_call["pion_ssm_load_snapshot", Int32](
            UInt32(worker_id), snap_path.as_c_string_slice())
        _ = external_call["pion_ssm_wal_replay", Int32](
            UInt32(worker_id), wal_path.as_c_string_slice())
        _ = external_call["pion_ssm_wal_open", Int32](
            UInt32(worker_id), wal_path.as_c_string_slice())


def ssm_save_snapshot(worker_id: Int) -> Int32:
    """gh #94 — write pion.ssm.<wid> snapshot + truncate the WAL.

    Mirrors the KV.PREFIX.SAVE path for V-store (kv_prefix.mojo:519). On
    success the WAL is reopened against a freshly truncated file, so post-
    snapshot mutations remain durable. Returns the number of slots written,
    or -1 on error.
    """
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        var snap_path = "pion.ssm." + String(worker_id) + "\0"
        var rc = external_call["pion_ssm_save_snapshot", Int32](
            UInt32(worker_id), snap_path.as_c_string_slice())
        if rc >= 0:
            _ = external_call["pion_ssm_wal_truncate", Int32](UInt32(worker_id))
        return rc
    else:
        return Int32(-1)


@always_inline
def handle_ssm_prefix_store(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    worker_id: Int,
) raises -> Int:
    """SSM.PREFIX.STORE <session_id> <layer_id> <state_blob>"""
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR SSM.PREFIX.STORE requires: session_id layer_id state_blob")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    if layer_id < 0:
        writer.append_error_response("ERR invalid layer_id (must be >= 0)")
        return 1
    var blob_tok = tokens[unsafe_offset=start + 3]
    var blob_len = Int(blob_tok.length)

    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        _ = ssm_init()
        var blob_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(blob_tok.ptr))
        var sid_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_tok.ptr))
        var rc = external_call["pion_ssm_store", Int32](
            UInt32(worker_id),
            sid_ptr, UInt32(sid_tok.length), UInt32(layer_id),
            blob_ptr, UInt32(blob_len),
        )
        if rc == 0:
            writer.append_ok_response()
        else:
            writer.append_error_response("ERR SSM.PREFIX.STORE failed (alloc or worker bound)")
    else:
        writer.append_error_response("ERR SSM.PREFIX.STORE not built for this platform")
    return 1


@always_inline
def handle_ssm_prefix_fetch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    worker_id: Int,
    fd: Int32,
) raises -> Int:
    """SSM.PREFIX.FETCH <session_id> <layer_id>

    Routes payloads ≥ ~3 MB through `append_bulk_bytes_writev` so the 4 MB
    `RESP_BUF_SIZE` cannot be overflowed (gh #76). Small payloads stay on the
    buffered fast path.
    """
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR SSM.PREFIX.FETCH requires: session_id layer_id")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
    if layer_id < 0:
        writer.append_error_response("ERR invalid layer_id (must be >= 0)")
        return 1

    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        _ = ssm_init()
        var sid_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_tok.ptr))
        # Two-call pattern: peek size first, then allocate + fetch. Slots
        # are allocated on the host heap (no contention) so the second call
        # is guaranteed to see the same size unless an evict happens between
        # — bounded race; if mismatch we return -ERR rather than partial data.
        var _size_buf = alloc[UInt32](1)
        var size_out_ptr = Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(_size_buf))
        var rc_size = external_call["pion_ssm_size", Int32](
            UInt32(worker_id),
            sid_ptr, UInt32(sid_tok.length), UInt32(layer_id),
            size_out_ptr,
        )
        if rc_size != 0:
            _size_buf.unsafe_free()
            writer.append_null_response()
            return 1
        var blob_len = Int(size_out_ptr[unsafe_offset=0])
        _size_buf.unsafe_free()
        if blob_len == 0:
            writer.append_bulk_string_response(null_ptr[UInt8, MutUntrackedOrigin](), 0)
            return 1
        var _buf = alloc[UInt8](blob_len)
        var buf_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_buf))
        var _len_buf = alloc[UInt32](1)
        var out_len_ptr = Pointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(_len_buf))
        var rc = external_call["pion_ssm_fetch", Int32](
            UInt32(worker_id),
            sid_ptr, UInt32(sid_tok.length), UInt32(layer_id),
            buf_ptr, UInt32(blob_len), out_len_ptr,
        )
        if rc == 0:
            writer.append_bulk_bytes_writev(fd, buf_ptr, Int(out_len_ptr[unsafe_offset=0]))
        else:
            writer.append_error_response("ERR SSM.PREFIX.FETCH unexpected race / size mismatch")
        _buf.unsafe_free()
        _len_buf.unsafe_free()
    else:
        writer.append_error_response("ERR SSM.PREFIX.FETCH not built for this platform")
    return 1


@always_inline
def handle_ssm_prefix_drop(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    worker_id: Int,
) raises -> Int:
    """SSM.PREFIX.DROP <session_id> [<layer_id>] — layer_id omitted = drop all."""
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR SSM.PREFIX.DROP requires: session_id [layer_id]")
        return 1

    var sid_tok = tokens[unsafe_offset=start + 1]
    var layer_id: Int = -1
    if start + 2 < num_tokens:
        layer_id = strict_atol(tokens[unsafe_offset=start + 2].value())
        if layer_id < 0:
            writer.append_error_response("ERR layer_id must be >= 0 (omit to drop all)")
            return 1

    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        _ = ssm_init()
        var sid_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_tok.ptr))
        var rc = external_call["pion_ssm_drop", Int32](
            UInt32(worker_id),
            sid_ptr, UInt32(sid_tok.length), Int32(layer_id),
        )
        if rc == 0:
            writer.append_ok_response()
        else:
            writer.append_error_response("ERR SSM.PREFIX.DROP failed (bad worker)")
    else:
        writer.append_error_response("ERR SSM.PREFIX.DROP not built for this platform")
    return 1
