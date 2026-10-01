"""AI.KNN_LM.* wire handlers — substrate for client-side kNN-LM augmentation.

See `src/network/knn_lm.mojo` for the datastore implementation. Pion ships the
wire surface WITHOUT committing to a kNN-LM mixing distribution: the retrieval
substrate is useful on its own, and the mixing step did not survive QA (see the
note in src/network/knn_lm.mojo).

Wire format for QUERY response (RESP bulk string of k × 8 bytes):
  Each entry: 4-byte little-endian Int32 token_id, 4-byte little-endian Float32 distance.
  Caller unpacks via numpy / struct. Sentinel: token_id = -1, distance = +INF for
  unfilled slots when k > count.
"""

from std.collections import Array
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc

from src.network.knn_lm import KNNLMIndex
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter


@always_inline
def _atol(s: String) raises -> Int64:
    return Int64(atol(s))


@always_inline
def handle_ai_knn_lm_create(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.CREATE <ds_id> <dim> [<max_entries>]"""
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.CREATE <ds_id> <dim> [<max_entries>]")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var dim = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    var max_entries = 100_000
    if start + 3 < num_tokens:
        max_entries = Int(_atol(tokens[unsafe_offset=start + 3].value()))
    var slot = knn_lm.create(name_ext, Int(name_tok.length), dim, max_entries)
    if slot < 0:
        writer.append_error_response(
            "ERR AI.KNN_LM.CREATE failed (duplicate name, registry full, or invalid dim/max_entries)"
        )
    else:
        writer.append_ok_response()
    return 1


@always_inline
def handle_ai_knn_lm_store(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.STORE <ds_id> <next_token_id> <embedding_blob>"""
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.STORE <ds_id> <next_token_id> <embedding_blob>")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = knn_lm.find(name_ext, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("ERR AI.KNN_LM.STORE: datastore not found (call AI.KNN_LM.CREATE first)")
        return 1
    var next_token_id = Int32(Int(_atol(tokens[unsafe_offset=start + 2].value())))
    var emb_tok = tokens[unsafe_offset=start + 3]
    var dim = knn_lm.datastores[slot].dim
    if Int(emb_tok.length) != dim * 4:
        writer.append_error_response("ERR embedding blob length mismatch (expected dim*4 bytes)")
        return 1
    var emb_ext = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(emb_tok.ptr.bitcast[Float32]()),
    )
    var ok = knn_lm.store_one(slot, next_token_id, emb_ext)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR AI.KNN_LM.STORE: datastore full")
    return 1


@always_inline
def handle_ai_knn_lm_storebatch(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.STOREBATCH <ds_id> <n> <token_ids_blob> <embeddings_blob>"""
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 4 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.STOREBATCH <ds_id> <n> <token_ids_blob> <embeddings_blob>")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = knn_lm.find(name_ext, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("ERR AI.KNN_LM.STOREBATCH: datastore not found")
        return 1
    var n = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    if n <= 0:
        writer.append_error_response("ERR AI.KNN_LM.STOREBATCH: n must be > 0")
        return 1
    var ids_tok = tokens[unsafe_offset=start + 3]
    var emb_tok = tokens[unsafe_offset=start + 4]
    var dim = knn_lm.datastores[slot].dim
    if Int(ids_tok.length) != n * 4:
        writer.append_error_response("ERR token_ids blob length mismatch (expected n*4 bytes)")
        return 1
    if Int(emb_tok.length) != n * dim * 4:
        writer.append_error_response("ERR embeddings blob length mismatch (expected n*dim*4 bytes)")
        return 1
    var ids_ext = Pointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=Int(ids_tok.ptr.bitcast[Int32]()),
    )
    var emb_ext = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(emb_tok.ptr.bitcast[Float32]()),
    )
    var stored = knn_lm.store_batch(slot, n, ids_ext, emb_ext)
    writer.append_int_response(Int64(stored))
    return 1


@always_inline
def handle_ai_knn_lm_query(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.QUERY <ds_id> <k> <embedding_blob>

    Response: bulk string of k × 8 bytes (Int32 token_id LE, Float32 distance LE).
    Sentinel: token_id = -1, distance = +INF for slots beyond datastore count.
    """
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.QUERY <ds_id> <k> <embedding_blob>")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = knn_lm.find(name_ext, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("ERR AI.KNN_LM.QUERY: datastore not found")
        return 1
    var k = Int(_atol(tokens[unsafe_offset=start + 2].value()))
    if k <= 0 or k > 4096:
        writer.append_error_response("ERR k must be in (0, 4096]")
        return 1
    var emb_tok = tokens[unsafe_offset=start + 3]
    var dim = knn_lm.datastores[slot].dim
    if Int(emb_tok.length) != dim * 4:
        writer.append_error_response("ERR query embedding blob length mismatch (expected dim*4 bytes)")
        return 1
    var emb_ext = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(emb_tok.ptr.bitcast[Float32]()),
    )
    # Allocate output buffers: k tokens + k distances, packed as k × (Int32, Float32) = 8k bytes.
    var _ids = alloc[Int32](k)
    var _dists = alloc[Float32](k)
    var ids_ext = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_ids))
    var dists_ext = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_dists))
    _ = knn_lm.query(slot, k, emb_ext, ids_ext, dists_ext)
    # Pack into k × 8-byte stride: [tok0][dist0][tok1][dist1]...
    var pack_bytes = k * 8
    var _packed = alloc[UInt8](pack_bytes)
    var packed_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_packed))
    for i in range(k):
        var tok_ptr = (packed_ext.unsafe_offset(i * 8)).bitcast[Int32]()
        var dist_ptr = (packed_ext.unsafe_offset(i * 8).unsafe_offset(4)).bitcast[Float32]()
        tok_ptr[] = ids_ext[unsafe_offset=i]
        dist_ptr[] = dists_ext[unsafe_offset=i]
    writer.append_bulk_string_response(packed_ext, pack_bytes)
    _ids.unsafe_free()
    _dists.unsafe_free()
    _packed.unsafe_free()
    return 1


@always_inline
def handle_ai_knn_lm_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.INFO <ds_id> → status string."""
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.INFO <ds_id>")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var slot = knn_lm.find(name_ext, Int(name_tok.length))
    if slot < 0:
        writer.append_error_response("ERR AI.KNN_LM.INFO: datastore not found")
        return 1
    var info = String("count=") + String(knn_lm.datastores[slot].count)
    info += " dim=" + String(knn_lm.datastores[slot].dim)
    info += " max_entries=" + String(knn_lm.datastores[slot].max_entries)
    var bytes = info.as_bytes()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bytes.unsafe_ptr()))
    writer.append_bulk_string_response(info_ext, len(bytes))
    return 1


@always_inline
def handle_ai_knn_lm_drop(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut knn_lm: KNNLMIndex,
) raises -> Int:
    """AI.KNN_LM.DROP <ds_id>"""
    if not knn_lm.enabled:
        writer.append_error_response("ERR AI.KNN_LM.* requires --kvcache or --inference")
        return 1
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR syntax: AI.KNN_LM.DROP <ds_id>")
        return 1
    var name_tok = tokens[unsafe_offset=start + 1]
    var name_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name_tok.ptr))
    var ok = knn_lm.drop(name_ext, Int(name_tok.length))
    if ok:
        writer.append_int_response(Int64(1))
    else:
        writer.append_int_response(Int64(0))
    return 1
