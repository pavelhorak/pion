"""Routing commands — AI.ROUTE.REGISTER, AI.ROUTE.UPDATE, AI.ROUTE, AI.ROUTE.REMOVE, AI.ROUTE.INFO

M13: Semantic Load Balancer for inference fleet management.
"""

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.collections import Array

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.network.semantic_router import SemanticRouter


@always_inline
def handle_ai_route_register(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut router: SemanticRouter,
) raises -> Int:
    """AI.ROUTE.REGISTER <node_id> <endpoint> <embedding> [CAPACITY <n>]"""
    if start + 3 >= num_tokens:
        writer.append_error_response("ERR AI.ROUTE.REGISTER requires: <node_id> <endpoint> <embedding>")
        return 1
    if not router.enabled:
        writer.append_error_response("ERR semantic router disabled (use --kvcache)")
        return 1

    var nid_ptr = tokens[unsafe_offset=start + 1].ptr
    var nid_len = tokens[unsafe_offset=start + 1].length
    var ep_ptr = tokens[unsafe_offset=start + 2].ptr
    var ep_len = tokens[unsafe_offset=start + 2].length
    var emb_ptr = tokens[unsafe_offset=start + 3].ptr
    var emb_len = tokens[unsafe_offset=start + 3].length

    var expected = router.dimensions * 4
    if emb_len != expected:
        writer.append_error_response("ERR embedding size mismatch")
        return 1

    # Parse optional CAPACITY
    var capacity = 0
    var j = start + 4
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        if jl == 8 and (jp[unsafe_offset=0] | 0x20) == 99 and (jp[unsafe_offset=1] | 0x20) == 97 and (jp[unsafe_offset=2] | 0x20) == 112:
            var vp = tokens[unsafe_offset=j + 1].ptr
            var vl = tokens[unsafe_offset=j + 1].length
            for vi in range(vl):
                capacity = capacity * 10 + Int(vp[unsafe_offset=vi] - 48)
            j += 2
        else:
            j += 1

    var ok = router.register_node(nid_ptr, nid_len, ep_ptr, ep_len, emb_ptr.unsafe_bitcast[Float32](), capacity)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR AI.ROUTE.REGISTER failed (full or duplicate)")
    return 1


@always_inline
def handle_ai_route_update(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut router: SemanticRouter,
) raises -> Int:
    """AI.ROUTE.UPDATE <node_id> <new_embedding>"""
    if start + 2 >= num_tokens:
        writer.append_error_response("ERR AI.ROUTE.UPDATE requires: <node_id> <embedding>")
        return 1

    var nid_ptr = tokens[unsafe_offset=start + 1].ptr
    var nid_len = tokens[unsafe_offset=start + 1].length
    var emb_ptr = tokens[unsafe_offset=start + 2].ptr

    var ok = router.update_centroid(nid_ptr, nid_len, emb_ptr.unsafe_bitcast[Float32]())
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR node not found")
    return 1


@always_inline
def handle_ai_route(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut router: SemanticRouter,
    server: TCPServer,
    fd: Int32,
    kq: Int32,
) raises -> Int:
    """AI.ROUTE <query_embedding> [EXCLUDE <node_id>]"""
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR AI.ROUTE requires: <query_embedding>")
        return 1
    if not router.enabled:
        writer.append_error_response("ERR semantic router disabled")
        return 1

    var emb_ptr = tokens[unsafe_offset=start + 1].ptr.unsafe_bitcast[Float32]()

    # Parse optional EXCLUDE
    var exclude_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
    var exclude_len = 0
    var j = start + 2
    while j + 1 < num_tokens:
        var jp = tokens[unsafe_offset=j].ptr
        var jl = tokens[unsafe_offset=j].length
        if jl == 7 and (jp[unsafe_offset=0] | 0x20) == 101 and (jp[unsafe_offset=1] | 0x20) == 120 and (jp[unsafe_offset=2] | 0x20) == 99:
            exclude_ptr = tokens[unsafe_offset=j + 1].ptr
            exclude_len = tokens[unsafe_offset=j + 1].length
            j += 2
        else:
            j += 1

    var hit = router.route_query(emb_ptr, exclude_ptr, exclude_len, writer)
    if not hit:
        writer.append_null_response()
    return 1


@always_inline
def handle_ai_route_remove(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin],
    start: Int,
    num_tokens: Int,
    mut writer: ResponseWriter,
    mut router: SemanticRouter,
) raises -> Int:
    """AI.ROUTE.REMOVE <node_id>"""
    if start + 1 >= num_tokens:
        writer.append_error_response("ERR AI.ROUTE.REMOVE requires: <node_id>")
        return 1

    var ok = router.remove_node(tokens[unsafe_offset=start + 1].ptr, tokens[unsafe_offset=start + 1].length)
    if ok:
        writer.append_ok_response()
    else:
        writer.append_error_response("ERR node not found")
    return 1


@always_inline
def handle_ai_route_info(
    mut writer: ResponseWriter,
    router: SemanticRouter,
) raises -> Int:
    """AI.ROUTE.INFO — routing table stats + per-node details."""
    router.info(writer)
    return 1
