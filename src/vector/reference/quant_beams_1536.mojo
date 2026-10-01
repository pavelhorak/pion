"""Open reference for the three quantized 1536-dim beams (D11).

Same traversal, same heaps, same per-lane float formula as the tuned routines
in libpion_vector, so results are bit-identical. What it leaves out is the
tuning: no prefetch, no batch-of-8 scoring, plain scalar block dots
(quant_dots.mojo) where the tuned routines use batched SDOT kernels.
"""
from std.memory.unsafe_pointer import UnsafePointer

from src.common.heap import HeapNode
from src.common.ptr import is_not_null
from ..quant_beam_view import QuantBeamView1536, QUANT_KIND_POLAR, QUANT_KIND_NANO
from ..kernels import QJL_U64S_1536
from .quant_dots import ref_int4_dot, ref_int3_dot, ref_int2_dot, ref_hamming


@always_inline
def _is_deleted(deleted: UnsafePointer[UInt8, MutUntrackedOrigin], idx: Int) -> Bool:
    return Bool((deleted[idx >> 3] >> UInt8(idx & 7)) & 1)


@no_inline
def quant_beam_search_1536_ref(v: UnsafePointer[QuantBeamView1536, MutUntrackedOrigin]):
    var kind = v[].kind
    var ef = v[].ef
    var candidates = v[].candidates
    var results = v[].results
    var q8 = v[].query_block_int8
    var qs = v[].query_block_scales
    var qnorm = v[].query_block_norm
    var l0_compact = v[].l0_compact
    var l0_slots = v[].l0_slots
    var num_nodes = v[].num_nodes
    var deleted = v[].deleted
    var visited = v[].visited
    var cur_epoch = v[].cur_epoch
    var compact_buffer = v[].compact_buffer
    var compact_stride = v[].compact_stride
    var compact_hdr = v[].compact_hdr
    var nodes = v[].nodes
    var qjl_buffer = v[].qjl_buffer
    var qjl_res_norms = v[].qjl_res_norms
    var qjl_query_signs = v[].qjl_query_signs
    var qjl_query_res_norm = v[].qjl_query_res_norm
    var qjl_lambda = v[].qjl_lambda
    var has_qjl = is_not_null(qjl_buffer) and is_not_null(qjl_res_norms)
    var inv_m = Float32(1.0) / Float32(1536)
    while len(candidates[].data) > 0:
        var c = candidates[].pop()
        var worst = results[].peek_distance()
        if c.distance > worst and len(results[].data) >= ef: break
        var l0_slot = l0_compact + c.id * 33
        var l0_srow = l0_slots + c.id * 33
        var neighbor_count = Int(l0_slot[0])
        for i in range(neighbor_count):
            var nidx = Int(l0_slot[1 + i])
            if nidx < 0 or nidx >= num_nodes: continue
            if _is_deleted(deleted, nidx): continue
            if visited[nidx] == cur_epoch: continue
            visited[nidx] = cur_epoch
            var nslot = l0_srow[1 + i]
            var vptr: UnsafePointer[Int8, MutUntrackedOrigin]
            if nslot != UInt32(0xFFFFFFFF):
                vptr = compact_buffer + Int(nslot) * compact_stride + compact_hdr
            else:
                vptr = nodes[nidx].vector
            var norm_v = vptr.bitcast[Float32]()[0]
            var d: Float32
            if kind == QUANT_KIND_POLAR:
                d = qnorm + norm_v - 2.0 * ref_int4_dot(q8, qs, vptr)
            elif kind == QUANT_KIND_NANO:
                d = qnorm + norm_v - 2.0 * ref_int2_dot(q8, qs, vptr)
            else:
                var dot_val = ref_int3_dot(q8, qs, vptr)
                var qjl_c: Float32 = 0.0
                if has_qjl:
                    var ham = Float32(ref_hamming(qjl_query_signs, qjl_buffer + nidx * QJL_U64S_1536, QJL_U64S_1536))
                    var cos_est = (Float32(1536) - Float32(2) * ham) * inv_m
                    qjl_c = qjl_lambda * cos_est * (qjl_query_res_norm * qjl_res_norms[nidx])
                d = qnorm + norm_v - 2.0 * dot_val - 2.0 * qjl_c
            candidates[].push(HeapNode(d, nidx))
            _ = results[].push_bounded(d, nidx, ef)
