"""Speculative RAG commands — RAG.SPECULATE.ENABLE, RAG.QUERY, RAG.SPECULATE.INFO

M9: Branch prediction for the RAG pipeline.
"""

from src.common.ptr import is_not_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.collections import Array, List

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.network.speculative_rag import SpeculativeRAG
from src.vector.hnsw import HNSWGraph, SharedHNSWView


@always_inline
def handle_rag_speculate_enable(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut spec: SpeculativeRAG,
) raises -> Int:
    """RAG.SPECULATE.ENABLE <session_id> [DEPTH <n>] [THRESHOLD <cosine>]"""
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR RAG.SPECULATE.ENABLE requires: <session_id>")
        return 1
    if not spec.enabled:
        writer.append_error_response("ERR speculative RAG disabled (use --kvcache)")
        return 1

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length
    var depth = 3
    var threshold = Float32(0.0)

    var j = start + 2
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        # DEPTH (5 bytes)
        if jl == 5 and (jp[unsafe_offset=0] | 0x20) == 100 and (jp[unsafe_offset=1] | 0x20) == 101 and (jp[unsafe_offset=2] | 0x20) == 112:
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            depth = 0
            for vi in range(vl):
                depth = depth * 10 + Int(vp[unsafe_offset=vi] - 48)
            j += 2
        # THRESHOLD (9 bytes)
        elif jl == 9 and (jp[unsafe_offset=0] | 0x20) == 116 and (jp[unsafe_offset=1] | 0x20) == 104 and (jp[unsafe_offset=2] | 0x20) == 114:
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            var int_part = 0
            var frac = Float32(0.0)
            var frac_div = Float32(1.0)
            var past_dot = False
            for vi in range(vl):
                var c = Int(vp[unsafe_offset=vi])
                if c == 46: past_dot = True
                elif c >= 48 and c <= 57:
                    if past_dot:
                        frac_div *= 10.0
                        frac += Float32(c - 48) / frac_div
                    else:
                        int_part = int_part * 10 + (c - 48)
            threshold = Float32(int_part) + frac
            j += 2
        else:
            j += 1

    var ok = spec.enable_session(sid_ptr, sid_len, depth, threshold)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR RAG.SPECULATE.ENABLE failed")
    return 1


@always_inline
def handle_rag_query(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut spec: SpeculativeRAG,
    mut hnsw: HNSWGraph,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
) raises -> Int:
    """RAG.QUERY <session_id> <query_embedding_fp32> [K <k>]

    1. Check speculative cache for a prediction match
    2. If hit: return pre-computed results immediately
    3. If miss: execute HNSW search, cache result, return
    4. Always: update trajectory history, generate new predictions
    """
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR RAG.QUERY requires: <session_id> <query_embedding>")
        return 1

    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length
    var emb_ptr = tokens[unsafe_offset=start + 2].ptr.unsafe_bitcast[Float32]()

    var k = 5  # default top-k
    var j = start + 3
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        if jl == 1 and (jp[unsafe_offset=0] | 0x20) == 107:  # K
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            k = 0
            for vi in range(vl):
                k = k * 10 + Int(vp[unsafe_offset=vi] - 48)
            j += 2
        else:
            j += 1

    var si = spec._find_session(sid_ptr, sid_len)
    if si < 0:
        # No speculative session — just do regular HNSW search
        if not hnsw.index_ready:
            writer.append_empty_array_response()
            return 1
        var scores = List[Float32]()
        var results = hnsw.search_fp32_scored(emb_ptr, k, scores, hnsw.ef_runtime)
        _write_rag_results(writer, results, scores)
        return 1

    # Process through speculative pipeline
    var hit_idx = spec.process_query(si, emb_ptr)

    if hit_idx >= 0 and spec.sessions[unsafe_offset=si].cache[unsafe_offset=hit_idx].num_results > 0:
        # SPECULATION HIT — return pre-computed results
        var entry = spec.sessions[unsafe_offset=si].cache[unsafe_offset=hit_idx]
        var n = min(entry.num_results, k)
        # Build response array
        var ids = List[Int]()
        var dists = List[Float32]()
        for ri in range(n):
            ids.append(Int(entry.result_ids[unsafe_offset=ri]))
            dists.append(entry.result_scores[unsafe_offset=ri])
        _write_rag_results(writer, ids, dists)
    else:
        # SPECULATION MISS — execute HNSW search
        if hnsw.index_ready:
            var scores = List[Float32]()
            var results = hnsw.search_fp32_scored(emb_ptr, k, scores, hnsw.ef_runtime)
            _write_rag_results(writer, results, scores)
        else:
            writer.append_empty_array_response()

    # Speculatively pre-execute HNSW search for predictions (fills cache for next query)
    if hnsw.index_ready:
        for ci in range(spec.sessions[unsafe_offset=si].cache_count):
            var pred_emb = spec.get_prediction_embedding(si, ci)
            if is_not_null(pred_emb):
                var pred_scores = List[Float32]()
                var pred_results = hnsw.search_fp32_scored(pred_emb, k, pred_scores, hnsw.ef_runtime)
                if len(pred_results) > 0:
                    var _rids = alloc[Int32](len(pred_results))
                    var rids = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(_rids))
                    var _rscores = alloc[Float32](len(pred_results))
                    var rscores = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_rscores))
                    for ri in range(len(pred_results)):
                        rids[unsafe_offset=ri] = Int32(pred_results[ri])
                        rscores[unsafe_offset=ri] = pred_scores[ri]
                    spec.store_prediction_results(si, ci, rids, rscores, len(pred_results))

    return 1


@always_inline
def handle_rag_speculate_info(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    spec: SpeculativeRAG,
) raises -> Int:
    """RAG.SPECULATE.INFO [session_id]"""
    # Global stats if no session_id
    if start + 1 >= num_tokens:
        var info = String("sessions:") + String(spec.session_count) + "\r\n"
        info += "total_hits:" + String(spec.total_hits) + "\r\n"
        info += "total_misses:" + String(spec.total_misses) + "\r\n"
        info += "total_predictions:" + String(spec.total_predictions) + "\r\n"
        var total = spec.total_hits + spec.total_misses
        if total > 0:
            var rate = Float64(spec.total_hits) / Float64(total)
            info += "hit_rate:" + String(rate) + "\r\n"
        else:
            info += "hit_rate:0.0\r\n"
        info += "enabled:" + String(spec.enabled) + "\r\n"
        writer.append_bulk_string_response(info.unsafe_ptr(), info.byte_length())
        return 1

    # Per-session stats
    var sid_ptr = tokens[unsafe_offset=start + 1].ptr
    var sid_len = tokens[unsafe_offset=start + 1].length
    var si = spec._find_session(sid_ptr, sid_len)
    if si < 0:
        writer.append_error_response("ERR session not found")
        return 1

    var sess = spec.sessions[unsafe_offset=si]
    var info = String("queries:") + String(sess.total_queries) + "\r\n"
    info += "hits:" + String(sess.spec_hits) + "\r\n"
    info += "misses:" + String(sess.spec_misses) + "\r\n"
    var total = sess.spec_hits + sess.spec_misses
    if total > 0:
        var rate = Float64(sess.spec_hits) / Float64(total)
        info += "hit_rate:" + String(rate) + "\r\n"
    else:
        info += "hit_rate:0.0\r\n"
    info += "depth:" + String(sess.depth) + "\r\n"
    info += "threshold:" + String(sess.threshold) + "\r\n"
    info += "history_count:" + String(sess.history_count) + "\r\n"
    info += "cached_predictions:" + String(sess.cache_count) + "\r\n"
    writer.append_bulk_string_response(info.unsafe_ptr(), info.byte_length())
    return 1


@always_inline
def _write_rag_results(mut writer: ResponseWriter, results: List[Int], scores: List[Float32]):
    """Write RAG results as RESP array of [doc_id, score] pairs."""
    var n = len(results)
    if n == 0:
        writer.append_empty_array_response()
        return
    # Return doc IDs as integer array (scores available via RAG.SPECULATE.INFO)
    # Format: *N\r\n :id1\r\n :id2\r\n ...
    var resp = String("*") + String(n) + "\r\n"
    for i in range(n):
        resp += ":" + String(results[i]) + "\r\n"
    writer.append_bulk_string_response(resp.unsafe_ptr(), resp.byte_length())
