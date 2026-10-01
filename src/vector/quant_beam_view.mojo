"""D11 seam for the three quantized 1536-dim beams (PolarQuant INT4,
NanoQuant INT2, TurboQuant INT3 + QJL): everything the level-0 beam reads or
writes, as one struct passed by pointer to the open reference or to
libpion_vector. The engine owns every buffer. Field order is ABI: append
only, and bump VECTOR_ABI_VERSION (vector_abi.mojo) on any change.

Unlike the INT8 beam, these beams keep the older dual-heap discipline
(`candidates` MinHeap + `results` MaxHeap, both engine-owned `List`s), and
`candidates` can grow past its reserve — so a push inside the library may
allocate through the Mojo runtime the engine links. Same allocator on both
sides of the call; recorded because the INT8 beam's view does not allocate.
"""
from std.memory.unsafe_pointer import UnsafePointer

from src.common.heap import MinHeap, MaxHeap
from .hnsw_types import HNSWNode

comptime QUANT_KIND_POLAR = 0
comptime QUANT_KIND_NANO = 1
comptime QUANT_KIND_TURBO = 2


@fieldwise_init
struct QuantBeamView1536(Copyable, Movable):
    var kind: Int
    var ef: Int
    var candidates: UnsafePointer[MinHeap, MutUntrackedOrigin]
    var results: UnsafePointer[MaxHeap, MutUntrackedOrigin]
    var query_block_int8: UnsafePointer[Int8, MutUntrackedOrigin]
    var query_block_scales: UnsafePointer[Float32, MutUntrackedOrigin]
    var query_block_norm: Float32
    var l0_compact: UnsafePointer[UInt32, MutUntrackedOrigin]
    var l0_slots: UnsafePointer[UInt32, MutUntrackedOrigin]
    var num_nodes: Int
    var deleted: UnsafePointer[UInt8, MutUntrackedOrigin]
    var visited: UnsafePointer[UInt16, MutUntrackedOrigin]
    var cur_epoch: UInt16
    var compact_buffer: UnsafePointer[Int8, MutUntrackedOrigin]
    var compact_stride: Int
    var compact_hdr: Int
    var nodes: UnsafePointer[HNSWNode, MutUntrackedOrigin]
    var qjl_buffer: UnsafePointer[UInt64, MutUntrackedOrigin]
    var qjl_res_norms: UnsafePointer[Float32, MutUntrackedOrigin]
    var qjl_query_signs: UnsafePointer[UInt64, MutUntrackedOrigin]
    var qjl_query_res_norm: Float32
    var qjl_lambda: Float32
