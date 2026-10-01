"""pion-memory: in-process vector memory library for AI workloads.

Shared library entry point. Exports C-compatible functions callable from
Python (ctypes), Rust (unsafe extern), C/C++, and MAX inference kernels.

Build:
  pixi run build-lib
  # => libpion_memory.dylib

API:
  pion_memory_create(dim, max_elements, M, ef_construction) -> handle
  pion_memory_destroy(handle)
  pion_remember(handle, id, fp32_vector) -> 0 ok / -1 error
  pion_optimize(handle)               -> 0 ok / -1 error
  pion_recall(handle, query, k, ef, out_ids, out_scores) -> count
  pion_forget(handle, id)             -> -1 (Phase 1)
  pion_count(handle)                  -> number of indexed vectors
  pion_is_ready(handle)              -> 1 if ready / 0 if not
"""

from src.common.ptr import is_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from src.vector.hnsw import HNSWGraph


# ---------------------------------------------------------------------------
# Internal context struct — heap-allocated, returned as opaque handle
# ---------------------------------------------------------------------------

struct PionMemoryCtx(Movable):
    var hnsw: HNSWGraph
    var scores: List[Float32]

    def __init__(out self, dim: Int, max_elements: Int, M: Int, ef_construction: Int):
        self.hnsw = HNSWGraph(max_elements, dim, M, ef_construction)
        self.scores = List[Float32]()

    def __moveinit__(out self, deinit take: PionMemoryCtx):
        self.hnsw = take.hnsw^
        self.scores = take.scores^


# ---------------------------------------------------------------------------
# Exported C functions
# ---------------------------------------------------------------------------

@export
def pion_memory_create(
    dim: Int32,
    max_elements: Int32,
    M: Int32,
    ef_construction: Int32,
) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Allocate and initialize a PionMemory context. Returns opaque handle."""
    var ptr = alloc[PionMemoryCtx](1)
    ptr.unsafe_write(PionMemoryCtx(
        Int(dim), Int(max_elements), Int(M), Int(ef_construction)
    ))
    return ptr.unsafe_bitcast[UInt8]()


@export
def pion_memory_destroy(handle: Pointer[UInt8, MutUntrackedOrigin]):
    """Free a PionMemory context created by pion_memory_create."""
    if is_null(handle):
        return
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    ctx.unsafe_deinit_pointee()
    ctx.unsafe_free()


@export
def pion_remember(
    handle: Pointer[UInt8, MutUntrackedOrigin],
    id: Int32,
    vector: Pointer[Float32, MutUntrackedOrigin],
) -> Int32:
    """Stage a vector for indexing (O(1)). Call pion_optimize() when done.

    Returns 0 on success, -1 on error.
    """
    if is_null(handle) or is_null(vector):
        return -1
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    try:
        ctx[].hnsw.add_vector(Int(id), vector)
        return 0
    except:
        return -1


@export
def pion_optimize(handle: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    """Build the HNSW index from staged vectors. Must call before pion_recall().

    Returns 0 on success, -1 if no vectors staged or on error.
    """
    if is_null(handle):
        return -1
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    try:
        ctx[].hnsw.build_index()
        ctx[].hnsw.index_ready = True
        return 0
    except:
        return -1


@export
def pion_recall(
    handle: Pointer[UInt8, MutUntrackedOrigin],
    query: Pointer[Float32, MutUntrackedOrigin],
    k: Int32,
    ef: Int32,
    out_ids: Pointer[Int32, MutUntrackedOrigin],
    out_scores: Pointer[Float32, MutUntrackedOrigin],
) -> Int32:
    """Search for k nearest neighbors of query.

    Returns the number of results written (<=k), or -1 on error.
    """
    if is_null(handle) or is_null(query) or is_null(out_ids) or is_null(out_scores):
        return -1
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    if not ctx[].hnsw.index_ready:
        return -1
    try:
        var results = ctx[].hnsw.search_fp32_scored(
            query, Int(k), ctx[].scores, Int(ef)
        )
        var n = len(results)
        var n_scores = len(ctx[].scores)
        for i in range(n):
            out_ids[i] = Int32(results[i])
            if i < n_scores:
                out_scores[i] = ctx[].scores[i]
            else:
                out_scores[i] = Float32(0.0)
        return Int32(n)
    except:
        return -1


@export
def pion_forget(
    handle: Pointer[UInt8, MutUntrackedOrigin],
    id: Int32,
) -> Int32:
    """Remove vector by ID. Phase 1 — not yet implemented. Returns -1."""
    return -1


@export
def pion_count(handle: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    """Return the number of indexed vectors."""
    if is_null(handle):
        return 0
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    return Int32(ctx[].hnsw.num_nodes)


@export
def pion_is_ready(handle: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    """Returns 1 if recall() is available, 0 if pion_optimize() is needed."""
    if is_null(handle):
        return 0
    var ctx = handle.unsafe_bitcast[PionMemoryCtx]()
    return 1 if ctx[].hnsw.index_ready else 0
