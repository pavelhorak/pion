"""Open reference for the 1536-dim INT8 beam search (D11).

Same algorithm as the tuned routine in `libpion_vector`, and bit-identical
results: the same traversal, the same `LinearPool` discipline, the same
float formula for every lane. Integer dot and L2 sums are exact in int32, so
any summation order gives the same value and the same float after the cast.

What it leaves out is the tuning, and nothing else:
  - no prefetch (the tuned routine stages 4 + 25 lines per neighbour);
  - no batching: one vector at a time, where the tuned routine scores 8 or 4
    neighbours per call and shares the query loads across lanes;
  - no ISA-specific dot instruction: a portable widening multiply-add, where
    the tuned routine uses SDOT on ARM and VNNI on x86.

The tuned routine picks a formula per lane by where the lane falls: lanes in a
batch of 8 or 4 use `q_norm + v_norm - 2*dot`, the last `valid_count % 4` use
a direct L2 (with the early-exit prefix/suffix test in the prune phase). The
reference reproduces that split exactly, because it decides which float
rounding each lane gets and therefore which candidates enter the pool.
"""
from std.memory.unsafe_pointer import UnsafePointer
from std.collections import Array

from src.common.ptr import is_null
from ..beam_view import BeamView1536

comptime _W = 16


@always_inline
def _ref_dot[n: Int](a: UnsafePointer[Int8, MutUntrackedOrigin], b: UnsafePointer[Int8, MutUntrackedOrigin]) -> Int32:
    var acc = SIMD[DType.int32, _W](0)
    for i in range(0, n, _W):
        acc += a.load[width=_W](i).cast[DType.int32]() * b.load[width=_W](i).cast[DType.int32]()
    return acc.reduce_add()


@always_inline
def _ref_l2[n: Int](a: UnsafePointer[Int8, MutUntrackedOrigin], b: UnsafePointer[Int8, MutUntrackedOrigin]) -> Int32:
    var acc = SIMD[DType.int32, _W](0)
    for i in range(0, n, _W):
        var d = a.load[width=_W](i).cast[DType.int32]() - b.load[width=_W](i).cast[DType.int32]()
        acc += d * d
    return acc.reduce_add()


@always_inline
def _is_deleted(deleted: UnsafePointer[UInt8, MutUntrackedOrigin], idx: Int) -> Bool:
    return Bool((deleted[idx >> 3] >> UInt8(idx & 7)) & 1)


@always_inline
def _norm(vptr: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
    """Node L2 norm, stored as Float32 in the 8-byte slot header."""
    return (vptr - 8).bitcast[Float32]()[0]


@always_inline
def _gather(
    v: UnsafePointer[BeamView1536, MutUntrackedOrigin], c_id: Int,
    mut idxs: Array[Int, 65],
    mut ptrs: Array[UnsafePointer[Int8, MutUntrackedOrigin], 33]) -> Int:
    """Unvisited, undeleted level-0 neighbours of `c_id`, in adjacency order,
    marked visited. Vector address: compact slot, else the node's own pointer."""
    var row = v[].l0_compact + c_id * 33
    var slots = v[].l0_slots + c_id * 33
    var count = Int(row[0])
    var n = 0
    for i in range(count):
        var nb = Int(row[1 + i])
        if nb < 0 or nb >= v[].num_nodes: continue
        if _is_deleted(v[].deleted, nb): continue
        if v[].visited[nb] == v[].cur_epoch: continue
        v[].visited[nb] = v[].cur_epoch
        idxs[n] = nb
        var slot = slots[1 + i]
        if slot != UInt32(0xFFFFFFFF):
            ptrs[n] = v[].compact_buffer + Int(slot) * v[].compact_stride + v[].compact_hdr
        else:
            ptrs[n] = v[].nodes[nb].vector
        n += 1
    return n


@no_inline
def beam_search_1536_ref(v: UnsafePointer[BeamView1536, MutUntrackedOrigin]):
    var pool = v[].pool
    var ef = v[].ef
    var qn = v[].query_norm_sq
    var q = v[].query_int8
    var idxs = Array[Int, 65](uninitialized=True)
    var ptrs = Array[UnsafePointer[Int8, MutUntrackedOrigin], 33](uninitialized=True)

    # Fill: expand the nearest unexpanded entry until the pool holds ef.
    while pool[].has_next() and pool[].size < ef:
        var c = pool[].pop()
        var n = _gather(v, c.id, idxs, ptrs)
        var batched = n - n % 4
        for j in range(n):
            var d: Float32
            if j < batched:
                d = qn + _norm(ptrs[j]) - 2.0 * _ref_dot[1536](q, ptrs[j]).cast[DType.float32]()
            elif is_null(q) or is_null(ptrs[j]):
                d = Float32(1e30)
            else:
                d = _ref_l2[1536](q, ptrs[j]).cast[DType.float32]()
            _ = pool[].insert(d, idxs[j])

    # Prune: expand until every pool entry is expanded.
    while pool[].has_next():
        var c = pool[].pop()
        var worst = pool[].worst()
        var n = _gather(v, c.id, idxs, ptrs)
        var batched = n - n % 4
        for j in range(n):
            var p = ptrs[j]
            if j < batched:
                var dot = _ref_dot[256](q, p).cast[DType.float32]() + _ref_dot[1280](q + 256, p + 256).cast[DType.float32]()
                var d = qn + _norm(p) - 2.0 * dot
                if pool[].insert(d, idxs[j]):
                    worst = pool[].worst()
            else:
                # Prefix L2, a coarse gate, then the suffix in 256-dim blocks
                # with an exit as soon as the running sum passes `worst`.
                var d_prefix = _ref_l2[256](q, p).cast[DType.float32]()
                if d_prefix < worst * Float32(0.175):
                    var running = d_prefix
                    var exited = False
                    for b in range(5):
                        var off = 256 + b * 256
                        running += _ref_l2[256](q + off, p + off).cast[DType.float32]()
                        if running > worst:
                            exited = True
                            break
                    if not exited and running < worst:
                        _ = pool[].insert(running, idxs[j])
