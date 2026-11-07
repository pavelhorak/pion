"""D11 seam: everything the 1536-dim beam search reads or writes, as one
plain struct passed by pointer — to the inline tuned routine, the open
reference, or the held static library (C ABI). The engine owns every buffer;
the routine never allocates. Field order is ABI: append only."""

from src.common.heap import LinearPool
from .hnsw_types import HNSWNode


@fieldwise_init
struct BeamView1536(Copyable, Movable):
    var pool: Pointer[LinearPool, MutUntrackedOrigin]
    var ef: Int
    var query_norm_sq: Float32
    var query_prefix_norm_sq: Float32
    var query_int8: Pointer[Int8, MutUntrackedOrigin]
    var l0_compact: Pointer[UInt32, MutUntrackedOrigin]
    var l0_slots: Pointer[UInt32, MutUntrackedOrigin]
    var num_nodes: Int
    var deleted: Pointer[UInt8, MutUntrackedOrigin]
    var visited: Pointer[UInt16, MutUntrackedOrigin]
    var cur_epoch: UInt16
    var compact_buffer: Pointer[Int8, MutUntrackedOrigin]
    var compact_stride: Int
    var compact_hdr: Int
    var nodes: Pointer[HNSWNode, MutUntrackedOrigin]
