"""KV Cache commands — KV.STORE, KV.FETCH, KV.INFO

Phase 1 of M14 (Externalized Attention).

Protocol:
  KV.STORE <cache_id> <embedding_blob:FP32> <tensor_blob:bytes> [TTL <sec>] [MODEL <name>]
  KV.FETCH <embedding_blob:FP32> [THRESHOLD <cosine>] [MODEL <name>]
  KV.INFO

gh #80: at capacity KV.STORE evicts the LRU entry; TTL is enforced lazily
on FETCH; FETCH filters by MODEL when given. KV.EVICT was named in earlier
docs but no handler exists — eviction is automatic.
"""

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.collections import Array

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.network.kv_cache_store import KVCacheStore
from src.common.utils import is_valid_float_arg, parse_float64, THRESHOLD_UNSET


@always_inline
def handle_kv_store(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut kvcache: KVCacheStore,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
) raises -> Int:
    """Handle KV.STORE <cache_id> <embedding> <blob> [TTL <sec>] [MODEL <name>]"""
    # Need at least: KV.STORE cache_id embedding blob = 4 tokens (start is at KV.STORE)
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR KV.STORE requires: <cache_id> <embedding> <blob>")
        return 1

    if not kvcache.enabled:
        writer.append_error_response("ERR KV cache store is disabled (use --kvcache flag)")
        return 1

    var cache_id_ptr = tokens[unsafe_offset=start + 1].ptr
    var cache_id_len = tokens[unsafe_offset=start + 1].length

    var embed_ptr = tokens[unsafe_offset=start + 2].ptr
    var embed_len = tokens[unsafe_offset=start + 2].length

    var blob_ptr = tokens[unsafe_offset=start + 3].ptr
    var blob_size = tokens[unsafe_offset=start + 3].length

    # Validate embedding size: must be dimensions * 4 (FP32)
    var expected_embed_bytes = kvcache.dimensions * 4
    if embed_len != expected_embed_bytes:
        writer.append_error_response("ERR embedding size mismatch: expected " + String(expected_embed_bytes) + " bytes, got " + String(embed_len))
        return 1

    # Parse optional args: TTL and MODEL
    var ttl_sec = 0
    var model_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
    var model_len = 0
    var j = start + 4
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        # TTL (3 bytes: t,t,l)
        if jl == 3 and (jp[unsafe_offset=0] | 0x20) == 116 and (jp[unsafe_offset=1] | 0x20) == 116 and (jp[unsafe_offset=2] | 0x20) == 108:
            # Parse integer TTL value
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            var val = 0
            for vi in range(vl):
                val = val * 10 + Int(vp[unsafe_offset=vi] - 48)
            ttl_sec = val
            j += 2
        # MODEL (5 bytes: m,o,d,e,l)
        elif jl == 5 and (jp[unsafe_offset=0] | 0x20) == 109 and (jp[unsafe_offset=1] | 0x20) == 111 and (jp[unsafe_offset=2] | 0x20) == 100 and (jp[unsafe_offset=3] | 0x20) == 101 and (jp[unsafe_offset=4] | 0x20) == 108:
            model_ptr = tokens[unsafe_offset=j + 1].ptr
            model_len = tokens[unsafe_offset=j + 1].length
            j += 2
        else:
            j += 1

    # Cast embedding bytes to Float32 pointer
    var fp32_ptr = embed_ptr.unsafe_bitcast[Float32]()

    var ok = kvcache.store(cache_id_ptr, cache_id_len, fp32_ptr, blob_ptr, blob_size, model_ptr, model_len, ttl_sec)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR KV.STORE failed (store disabled)")
    return 1


@always_inline
def handle_kv_fetch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut kvcache: KVCacheStore,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
) raises -> Int:
    """Handle KV.FETCH <embedding> [THRESHOLD <cosine>] [MODEL <name>]"""
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR KV.FETCH requires: <embedding>")
        return 1

    if not kvcache.enabled:
        writer.append_error_response("ERR KV cache store is disabled (use --kvcache flag)")
        return 1

    var embed_ptr = tokens[unsafe_offset=start + 1].ptr
    var embed_len = tokens[unsafe_offset=start + 1].length

    var expected_embed_bytes = kvcache.dimensions * 4
    if embed_len != expected_embed_bytes:
        writer.append_error_response("ERR embedding size mismatch: expected " + String(expected_embed_bytes) + " bytes, got " + String(embed_len))
        return 1

    # Parse optional THRESHOLD and MODEL
    var threshold = THRESHOLD_UNSET  # gh #373: unset = default; an explicit 0 is honoured
    var model_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
    var model_len = 0
    var j = start + 2
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        # THRESHOLD (9 bytes)
        if jl == 9 and (jp[unsafe_offset=0] | 0x20) == 116 and (jp[unsafe_offset=1] | 0x20) == 104 and (jp[unsafe_offset=2] | 0x20) == 114:
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            # gh #373: strict — the old loop skipped non-digits, so "abc" was
            # 0.0 and "-0.5" was 0.5.
            if not is_valid_float_arg(vp, vl):
                writer.append_error_response("ERR THRESHOLD value is not a valid float")
                return 1
            threshold = Float32(parse_float64(vp, vl))
            j += 2
        # MODEL (5 bytes: m,o,d,e,l)
        elif jl == 5 and (jp[unsafe_offset=0] | 0x20) == 109 and (jp[unsafe_offset=1] | 0x20) == 111 and (jp[unsafe_offset=2] | 0x20) == 100 and (jp[unsafe_offset=3] | 0x20) == 101 and (jp[unsafe_offset=4] | 0x20) == 108:
            model_ptr = tokens[unsafe_offset=j + 1].ptr
            model_len = tokens[unsafe_offset=j + 1].length
            j += 2
        else:
            j += 1

    var fp32_ptr = embed_ptr.unsafe_bitcast[Float32]()

    var hit = kvcache.fetch(fp32_ptr, threshold, model_ptr, model_len, writer, server, fd, kq)
    if not hit:
        writer.append_null_response()
    return 1


@always_inline
def handle_kv_info(
    mut writer: ResponseWriter,
    kvcache: KVCacheStore,
) raises -> Int:
    """Handle KV.INFO — return cache statistics."""
    kvcache.info(writer)
    return 1
