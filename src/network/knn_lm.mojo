"""KNNLMIndex — token-id-tagged kNN datastore for AI.KNN_LM.* substrate.

Stores (next_token_id, embedding) pairs indexed by datastore name. Two query
backends, selected automatically by datastore size:

  count < KNN_HNSW_BUILD_THRESHOLD  → brute-force linear scan (SIMD FP32 L2)
  count ≥ KNN_HNSW_BUILD_THRESHOLD  → per-vector SQ8 HNSW (this file's KNNHNSW)

The cutover at ~5000 entries is a heuristic — at that scale HNSW's
~3M ops/query (ef≈200, M=32, ~50 hops) starts beating brute-force's
N×dim=5K×768 ≈ 4M ops, and the gap widens fast: at 1M entries brute-force
is ~750M ops, HNSW still ~3M.

**Asymmetric SQ8 storage (gh #42).** Each base vector is mirrored as INT8
with per-vec (qmin, qrange); queries stay FP32. Search uses the asymmetric
`l2_distance_fp32_int8` kernel from src/vector/kernels.mojo with dual-FMA
dequant fused into the L2² accumulator. Cuts per-distance load 4×
(3 KB FP32 → 768 B INT8 + 8 B scale/offset). The brute-force path stays
pure FP32 — small N is bandwidth-cheap and exact. We keep both storages
(FP32 for brute-force / rerank, INT8 for HNSW) at +25% memory.

**Heap-based PQ + per-vec batch-4 (gh #42 follow-up).** _search_layer
maintains a binary min-heap of candidates (closest at root) and a
max-heap of results (worst at root, bounded ef), pre-allocated on
KNNHNSW. The inner neighbor-visit loop collects unvisited neighbors
into a stack-allocated Array then dispatches in groups of 4 to
`l2_distance_fp32_int8_pervec_batch4`, sharing one query load across
4 dequant chains.

**Lazy reciprocal pruning (gh #42).** Bidirectional back-links during
build append up to KNN_HNSW_M0_STORAGE = 2× M0; the eager M² heuristic
rebuild only fires at 2× saturation. compact_overflows() runs at end
of bulk insert and prunes any saturated row back to M0/M using the
same heuristic — same final selection, only timing shifts.

**Performance vs scale (measured 2026-05-04 with `tests/bench_knn_lm_hnsw.py`,
dim=768, k=10, 50 random unit-norm queries, Mac):**

  | N      | path        | recall@10 | median latency | bulk-build |
  |---:    |---          |---:       |---:            |---:        |
  | 2K     | brute-force | 1.00      | ~0.40 ms       | ~7 ms      |
  | 8K     | HNSW        | 1.00      | ~0.73 ms       | ~5.7 s     |
  | 30K    | HNSW        | 0.90      | ~1.94 ms       | ~36 s      |

Sub-linear latency scaling holds (size 3.75× → latency 2.7×). Hard gates:
recall ≥ 0.85, latency ≤ 2.5 ms, bulk-build ≤ 60 s. The remaining latency
budget at 30K is memory-stall on random graph traversal, not compute —
identifiable per-query work (distance + heap + visited checks) accounts
for ~250 µs of ~2.5 ms server time. Further wins would require structural
rework (graph reordering, sub-4-bit base storage) — defer until a consumer
measures PPL impact at higher scales.

Use case: client-side kNN-LM augmentation (Khandelwal et al. 2020). The PPL
win is real for in-domain text infill / code completion / log generation,
even though QA accuracy doesn't lift — the literature autopsy holds, and our
own measurement at 1B reproduced it. Pion ships the wire surface so consumers
can build PPL-improving workloads on a sub-2ms kNN substrate.

Wire commands (handled in src/commands/knn_lm.mojo):
  AI.KNN_LM.CREATE     <ds_id> <dim> [<max_entries>]
  AI.KNN_LM.STORE      <ds_id> <next_token_id> <embedding_blob>
  AI.KNN_LM.STOREBATCH <ds_id> <n> <token_ids_blob> <embeddings_blob>
  AI.KNN_LM.QUERY      <ds_id> <k> <embedding_blob>  → k × (Int32 token, Float32 dist)
  AI.KNN_LM.INFO       <ds_id>
  AI.KNN_LM.DROP       <ds_id>
"""

from src.vector.fp32_scan import l2sq_f32_4chain
from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import Array
from std.math import log, sqrt
from std.sys import simd_width_of

from src.vector.kernels import (
    quantize_fp32_to_int8_simd,
    l2_distance_fp32_int8,
    l2_distance_fp32_int8_pervec_batch4,
)


comptime MAX_KNN_DATASTORES = 16
comptime KNN_DS_NAME_INLINE = 64
comptime DEFAULT_MAX_ENTRIES = 100_000

# ── HNSW tunables ────────────────────────────────────────────────────────
# Standard HNSW parameters from Malkov & Yashunin 2018 §4. These match the
# defaults the FT.SEARCH path uses (M=16, ef_construction=200) so any future
# unification with HNSWGraph won't churn topology.
comptime KNN_HNSW_M = 32                 # max neighbors per layer (gh #38: 16→32)
comptime KNN_HNSW_M0 = 64                # max neighbors at layer 0 (2× M, standard)
comptime KNN_HNSW_MAX_LAYERS = 8         # bounds memory; ln(1M)/ln(M) ≈ 5
comptime KNN_HNSW_EF_CONSTRUCTION = 200  # ef during insert (gh #38: kept at 200; ec=400 doubled build for marginal recall)
comptime KNN_HNSW_DEFAULT_EF = 200       # ef during search if caller didn't override
# Below this count, brute-force is faster than HNSW (graph build/search overhead
# exceeds linear-scan time). Picked conservatively; bench can refine.
comptime KNN_HNSW_BUILD_THRESHOLD = 5_000
# Pre-allocated PQ scratch caps (gh #42 follow-up). Result max-heap is
# bounded by ef ≤ KNN_HNSW_PQ_RES_CAP. Candidate min-heap can grow beyond
# ef during search (each visit pushes up to M0 candidates after dedup);
# bound it generously at 4× the result cap.
comptime KNN_HNSW_PQ_RES_CAP = 512
comptime KNN_HNSW_PQ_CAND_CAP = 2048
# Per-insert scratch caps. ef_c never exceeds KNN_HNSW_EF_CONSTRUCTION (200);
# heuristic picks up to M0=64. Pre-allocated on KNNHNSW so insert() avoids
# 4 alloc/frees per layer × 5 layers per insert (gh #42 follow-up build win).
comptime KNN_HNSW_INSERT_SL_CAP = 256       # ≥ KNN_HNSW_EF_CONSTRUCTION
comptime KNN_HNSW_INSERT_PICKED_CAP = 64    # ≥ KNN_HNSW_M0
# Lazy reciprocal pruning (gh #42 follow-up, Codex-recommended). During
# build, allow neighbor lists to grow up to 2× the heuristic cap before
# triggering a rebuild — most nodes never saturate, so the rebuild fires
# far less often. compact_overflows() at end of bulk insert runs the
# normal heuristic over saturated rows to bring them back to M0/M.
# Slot stride uses STORAGE so layer-0 fan-out and upper-layer fan-out
# share the same stride math (existing convention).
comptime KNN_HNSW_M0_STORAGE = 128          # 2× M0 — layer-0 build-time slot count
comptime KNN_HNSW_M_STORAGE = 64            # 2× M — upper-layer build-time cap
# 1.0/ln(M) for the random-layer geometric distribution. Inlined for speed.
comptime KNN_HNSW_LEVEL_MULT = Float32(1.0) / Float32(3.4657359)  # 1/ln(32) — must match KNN_HNSW_M (Malkov-Yashunin, gh #38)


# ── Heap helpers (gh #42 follow-up) ──────────────────────────────────────
# Replace the O(ef²) sorted-array PQ in _search_layer with a binary heap:
# O(log ef) push/pop. Two flavors: min-heap (candidates, root = closest)
# and max-heap (results, root = worst). Plain free functions taking
# parallel `(dist, id)` arrays — no struct, so the compiler can inline
# them into the hot loop.

@always_inline
def _heap_sift_up_min(
    dist: UnsafePointer[Float32, MutUntrackedOrigin],
    id_: UnsafePointer[Int32, MutUntrackedOrigin],
    start: Int,
):
    var idx = start
    while idx > 0:
        var parent = (idx - 1) >> 1
        if dist[parent] > dist[idx]:
            var td = dist[idx]; dist[idx] = dist[parent]; dist[parent] = td
            var ti = id_[idx]; id_[idx] = id_[parent]; id_[parent] = ti
            idx = parent
        else:
            return


@always_inline
def _heap_sift_down_min(
    dist: UnsafePointer[Float32, MutUntrackedOrigin],
    id_: UnsafePointer[Int32, MutUntrackedOrigin],
    start: Int,
    n: Int,
):
    var idx = start
    while True:
        var left = (idx << 1) + 1
        if left >= n: return
        var smallest = left
        var right = left + 1
        if right < n and dist[right] < dist[left]:
            smallest = right
        if dist[smallest] >= dist[idx]:
            return
        var td = dist[idx]; dist[idx] = dist[smallest]; dist[smallest] = td
        var ti = id_[idx]; id_[idx] = id_[smallest]; id_[smallest] = ti
        idx = smallest


@always_inline
def _heap_sift_up_max(
    dist: UnsafePointer[Float32, MutUntrackedOrigin],
    id_: UnsafePointer[Int32, MutUntrackedOrigin],
    start: Int,
):
    var idx = start
    while idx > 0:
        var parent = (idx - 1) >> 1
        if dist[parent] < dist[idx]:
            var td = dist[idx]; dist[idx] = dist[parent]; dist[parent] = td
            var ti = id_[idx]; id_[idx] = id_[parent]; id_[parent] = ti
            idx = parent
        else:
            return


@always_inline
def _heap_sift_down_max(
    dist: UnsafePointer[Float32, MutUntrackedOrigin],
    id_: UnsafePointer[Int32, MutUntrackedOrigin],
    start: Int,
    n: Int,
):
    var idx = start
    while True:
        var left = (idx << 1) + 1
        if left >= n: return
        var largest = left
        var right = left + 1
        if right < n and dist[right] > dist[left]:
            largest = right
        if dist[largest] <= dist[idx]:
            return
        var td = dist[idx]; dist[idx] = dist[largest]; dist[largest] = td
        var ti = id_[idx]; id_[idx] = id_[largest]; id_[largest] = ti
        idx = largest


# ── Pseudo-random for layer assignment ───────────────────────────────────
# A node's layer is sampled from a geometric distribution with mean 1/ln(M).
# We need cheap per-insert randomness; use a per-graph LCG that's NOT seeded
# by clock (so insert order is the only entropy source). Same node_id always
# gets the same layer — deterministic, reproducible, and good enough for
# HNSW's "random level" property.
@always_inline
def _knn_hnsw_node_level(node_id: Int, mut state: UInt64) -> Int:
    # xorshift64 for per-node randomness
    var x = state
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    state = x
    var u = (Float32(Int(x & 0xFFFFFFFF)) + Float32(1.0)) / Float32(0x100000000)
    var f = -log(u) * KNN_HNSW_LEVEL_MULT
    var lvl = Int(f)
    if lvl < 0:
        lvl = 0
    if lvl >= KNN_HNSW_MAX_LAYERS:
        lvl = KNN_HNSW_MAX_LAYERS - 1
    return lvl


# ── Per-vector SQ8 (asymmetric scalar quantization) ──────────────────────
# Each vector gets its own (qmin, qrange) so dequant is `(b + 127) * qrange/254 + qmin`.
# Pairs with `quantize_fp32_to_int8_simd` (kernels.mojo) and the asymmetric L2
# kernel `l2_distance_fp32_int8` (gh #42).
@always_inline
def _vec_sq8_quantize(
    src: UnsafePointer[Float32, MutUntrackedOrigin],
    dst: UnsafePointer[Int8, MutUntrackedOrigin],
    dim: Int,
) -> Tuple[Float32, Float32]:
    """Find per-vec min/max and quantize FP32→INT8. Returns (qmin, qrange).
    qrange clamped to ≥ 1e-9 so a constant vector stays representable."""
    comptime W = simd_width_of[DType.float32]()
    var n_simd = (dim // W) * W
    var mn: Float32 = src[0]
    var mx: Float32 = src[0]
    var i = 0
    if n_simd >= W:
        var mn_v = src.load[width=W](0)
        var mx_v = mn_v
        i = W
        while i < n_simd:
            var v = src.load[width=W](i)
            mn_v = min(mn_v, v)
            mx_v = max(mx_v, v)
            i += W
        mn = mn_v.reduce_min()
        mx = mx_v.reduce_max()
    else:
        i = 1
    while i < dim:
        var v = src[i]
        if v < mn: mn = v
        if v > mx: mx = v
        i += 1
    var qrange = mx - mn
    if qrange < Float32(1e-9):
        qrange = Float32(1e-9)
    var scale = Float32(254.0) / qrange
    quantize_fp32_to_int8_simd(src, dst, dim, mn, scale)
    return (mn, qrange)


@always_inline
def _l2_distance_fp32(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    dim: Int,
) -> Float32:
    """Squared L2 distance over two FP32 vectors (gh #400: four FMA chains)."""
    return l2sq_f32_4chain(a, b, dim)


struct KNNHNSW(Movable, Copyable):
    """Per-datastore HNSW graph topology. Asymmetric distance: queries stay
    FP32, base vectors live as per-vector SQ8 (gh #42). Both `embeddings_ref`
    (FP32) and `embeddings_int8_ref` + `sq_min_ref` + `sq_range_ref` are
    borrowed from KNNDatastore — this struct owns only the graph (neighbor
    table, layer levels, visited stamps).

    Memory: max_elements × (KNN_HNSW_MAX_LAYERS × KNN_HNSW_M × 4 + 8 + 4 + 1) bytes.
    At max_elements=100K with defaults: 100K × (8×16×4 + 13) ≈ 52 MB.
    At 1M: 525 MB. Allocated lazily — only when build_threshold is crossed.
    """
    var built: Bool                # graph allocated?
    var max_elements: Int
    var num_nodes: Int             # nodes inserted into the graph so far
    var dim: Int
    var entry_point: Int            # -1 if empty
    var max_layer: Int              # max layer reached so far
    var rng_state: UInt64           # xorshift state for per-node layer sampling

    # neighbors[node][layer][slot]: flat Int32 array of size
    # max_elements * KNN_HNSW_MAX_LAYERS * KNN_HNSW_M0. We pad ALL layers to
    # M0 (32) so layer 0 (which gets the wider fan-out) shares the same
    # stride math as upper layers. Wastes (M0-M)/M0 × upper-layer slots ≈
    # 50% of upper-layer memory; cheap given upper layers are tiny.
    var neighbors: UnsafePointer[Int32, MutUntrackedOrigin]
    # neighbor_count[node][layer] populated count.
    var neighbor_counts: UnsafePointer[Int32, MutUntrackedOrigin]
    # node_level[node] = top layer this node participates in.
    var node_level: UnsafePointer[Int8, MutUntrackedOrigin]

    # Visited tracking: per-node generation stamp. Each query increments
    # `visited_gen`; visited check is `node_visited_gen[id] == visited_gen`.
    # Avoids per-query bitset memset.
    var visited_gen: UInt32
    var node_visited_gen: UnsafePointer[UInt32, MutUntrackedOrigin]

    # Borrowed pointers to KNNDatastore — NOT owned, do not free.
    var embeddings_ref: UnsafePointer[Float32, MutUntrackedOrigin]
    # gh #42: per-vector SQ8 mirror. Used by _distance for asymmetric L2.
    var embeddings_int8_ref: UnsafePointer[Int8, MutUntrackedOrigin]
    var sq_min_ref: UnsafePointer[Float32, MutUntrackedOrigin]
    var sq_range_ref: UnsafePointer[Float32, MutUntrackedOrigin]
    # gh #42 follow-up: pre-allocated PQ scratch (heap-based search).
    # Owned by KNNHNSW — freed in free_buffers(). Reused across all
    # _search_layer calls; safe under the shared-nothing model where each
    # worker has its own KNNHNSW and queries are sequential within a worker.
    var pq_cand_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var pq_cand_id: UnsafePointer[Int32, MutUntrackedOrigin]
    var pq_res_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var pq_res_id: UnsafePointer[Int32, MutUntrackedOrigin]
    # gh #42 follow-up: pre-allocated insert() scratch. Eliminates alloc
    # churn (4 buffers × 5 layers × 30K inserts = 600K malloc/free).
    var ins_sl_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var ins_sl_id: UnsafePointer[Int32, MutUntrackedOrigin]
    var ins_picked_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var ins_picked_id: UnsafePointer[Int32, MutUntrackedOrigin]
    # Compact-pass / rebuild scratch (used by insert's rebuild branch and
    # by compact_overflows). Pool holds candidates ASC; rebuilt holds the
    # heuristic output without clobbering the outer loop's picked_*.
    var ins_pool_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var ins_pool_id: UnsafePointer[Int32, MutUntrackedOrigin]
    var ins_rebuilt_dist: UnsafePointer[Float32, MutUntrackedOrigin]
    var ins_rebuilt_id: UnsafePointer[Int32, MutUntrackedOrigin]

    def __init__(out self):
        self.built = False
        self.max_elements = 0
        self.num_nodes = 0
        self.dim = 0
        self.entry_point = -1
        self.max_layer = 0
        self.rng_state = UInt64(0xC0DEBA5E5EED1234)
        self.neighbors = null_ptr[Int32, MutUntrackedOrigin]()
        self.neighbor_counts = null_ptr[Int32, MutUntrackedOrigin]()
        self.node_level = null_ptr[Int8, MutUntrackedOrigin]()
        self.visited_gen = 0
        self.node_visited_gen = null_ptr[UInt32, MutUntrackedOrigin]()
        self.embeddings_ref = null_ptr[Float32, MutUntrackedOrigin]()
        self.embeddings_int8_ref = null_ptr[Int8, MutUntrackedOrigin]()
        self.sq_min_ref = null_ptr[Float32, MutUntrackedOrigin]()
        self.sq_range_ref = null_ptr[Float32, MutUntrackedOrigin]()
        self.pq_cand_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.pq_cand_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.pq_res_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.pq_res_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.ins_sl_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.ins_sl_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.ins_picked_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.ins_picked_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.ins_pool_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.ins_pool_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.ins_rebuilt_dist = null_ptr[Float32, MutUntrackedOrigin]()
        self.ins_rebuilt_id = null_ptr[Int32, MutUntrackedOrigin]()

    def __copyinit__(out self, existing: Self):
        # Shallow copy — graph buffers are pointer-aliased. Used only by
        # Array fill / KNNDatastore copyinit during registry creation;
        # never used to clone an active graph (the registry is "create then
        # mutate in place"). Same convention as KNNDatastore.
        self.built = existing.built
        self.max_elements = existing.max_elements
        self.num_nodes = existing.num_nodes
        self.dim = existing.dim
        self.entry_point = existing.entry_point
        self.max_layer = existing.max_layer
        self.rng_state = existing.rng_state
        self.neighbors = existing.neighbors
        self.neighbor_counts = existing.neighbor_counts
        self.node_level = existing.node_level
        self.visited_gen = existing.visited_gen
        self.node_visited_gen = existing.node_visited_gen
        self.embeddings_ref = existing.embeddings_ref
        self.embeddings_int8_ref = existing.embeddings_int8_ref
        self.sq_min_ref = existing.sq_min_ref
        self.sq_range_ref = existing.sq_range_ref
        self.pq_cand_dist = existing.pq_cand_dist
        self.pq_cand_id = existing.pq_cand_id
        self.pq_res_dist = existing.pq_res_dist
        self.pq_res_id = existing.pq_res_id
        self.ins_sl_dist = existing.ins_sl_dist
        self.ins_sl_id = existing.ins_sl_id
        self.ins_picked_dist = existing.ins_picked_dist
        self.ins_picked_id = existing.ins_picked_id
        self.ins_pool_dist = existing.ins_pool_dist
        self.ins_pool_id = existing.ins_pool_id
        self.ins_rebuilt_dist = existing.ins_rebuilt_dist
        self.ins_rebuilt_id = existing.ins_rebuilt_id

    def __moveinit__(out self, var existing: Self):
        self.built = existing.built
        self.max_elements = existing.max_elements
        self.num_nodes = existing.num_nodes
        self.dim = existing.dim
        self.entry_point = existing.entry_point
        self.max_layer = existing.max_layer
        self.rng_state = existing.rng_state
        self.neighbors = existing.neighbors
        self.neighbor_counts = existing.neighbor_counts
        self.node_level = existing.node_level
        self.visited_gen = existing.visited_gen
        self.node_visited_gen = existing.node_visited_gen
        self.embeddings_ref = existing.embeddings_ref
        self.embeddings_int8_ref = existing.embeddings_int8_ref
        self.sq_min_ref = existing.sq_min_ref
        self.sq_range_ref = existing.sq_range_ref
        self.pq_cand_dist = existing.pq_cand_dist
        self.pq_cand_id = existing.pq_cand_id
        self.pq_res_dist = existing.pq_res_dist
        self.pq_res_id = existing.pq_res_id
        self.ins_sl_dist = existing.ins_sl_dist
        self.ins_sl_id = existing.ins_sl_id
        self.ins_picked_dist = existing.ins_picked_dist
        self.ins_picked_id = existing.ins_picked_id
        self.ins_pool_dist = existing.ins_pool_dist
        self.ins_pool_id = existing.ins_pool_id
        self.ins_rebuilt_dist = existing.ins_rebuilt_dist
        self.ins_rebuilt_id = existing.ins_rebuilt_id

    @always_inline
    def _slot_idx(self, node: Int, layer: Int) -> Int:
        """Flat index into `neighbors` for the start of (node, layer)'s slot row."""
        return (node * KNN_HNSW_MAX_LAYERS + layer) * KNN_HNSW_M0_STORAGE

    @always_inline
    def _count_idx(self, node: Int, layer: Int) -> Int:
        return node * KNN_HNSW_MAX_LAYERS + layer

    @always_inline
    def _max_neighbors(self, layer: Int) -> Int:
        """Heuristic target: how many neighbors to KEEP after compaction."""
        return KNN_HNSW_M0 if layer == 0 else KNN_HNSW_M

    @always_inline
    def _max_neighbors_storage(self, layer: Int) -> Int:
        """Build-time overflow cap: how many neighbors a row may TEMPORARILY
        hold before triggering an eager rebuild. compact_overflows() at end
        of bulk insert prunes back to _max_neighbors(layer)."""
        return KNN_HNSW_M0_STORAGE if layer == 0 else KNN_HNSW_M_STORAGE

    @always_inline
    def _distance(self, query: UnsafePointer[Float32, MutUntrackedOrigin], node: Int) -> Float32:
        # gh #42: asymmetric L2² via per-vector SQ8 base. Matches
        # `quantize_fp32_to_int8_simd` (kernels.mojo): dequant uses
        # (qmin, qrange) and reconstructs `(b + 127) * qrange/254 + qmin`.
        return l2_distance_fp32_int8(
            query,
            self.embeddings_int8_ref + node * self.dim,
            self.dim,
            self.sq_min_ref[node],
            self.sq_range_ref[node],
        )

    def build(
        mut self,
        max_elements: Int,
        dim: Int,
        embeddings_ref: UnsafePointer[Float32, MutUntrackedOrigin],
        embeddings_int8_ref: UnsafePointer[Int8, MutUntrackedOrigin],
        sq_min_ref: UnsafePointer[Float32, MutUntrackedOrigin],
        sq_range_ref: UnsafePointer[Float32, MutUntrackedOrigin],
    ):
        """Allocate graph buffers. Called lazily when count first reaches
        KNN_HNSW_BUILD_THRESHOLD; bulk-inserts the existing entries."""
        if self.built: return
        if max_elements <= 0 or dim <= 0: return
        self.max_elements = max_elements
        self.dim = dim
        self.embeddings_ref = embeddings_ref
        self.embeddings_int8_ref = embeddings_int8_ref
        self.sq_min_ref = sq_min_ref
        self.sq_range_ref = sq_range_ref

        var n_slots = max_elements * KNN_HNSW_MAX_LAYERS * KNN_HNSW_M0_STORAGE
        var n_counts = max_elements * KNN_HNSW_MAX_LAYERS
        self.neighbors = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](n_slots))
        )
        self.neighbor_counts = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](n_counts))
        )
        self.node_level = UnsafePointer[Int8, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int8](max_elements))
        )
        self.node_visited_gen = UnsafePointer[UInt32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[UInt32](max_elements))
        )
        # Zero the count and visited arrays (neighbors content doesn't need
        # init since we read up to neighbor_count items per row).
        unsafe_memset(self.neighbor_counts.bitcast[UInt8](), 0, n_counts * 4)
        unsafe_memset(self.node_visited_gen.bitcast[UInt8](), 0, max_elements * 4)
        # gh #42 follow-up: pre-allocate PQ scratch once. _search_layer
        # reuses these buffers — zero per-query alloc.
        self.pq_cand_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_PQ_CAND_CAP)),
        )
        self.pq_cand_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_PQ_CAND_CAP)),
        )
        self.pq_res_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_PQ_RES_CAP)),
        )
        self.pq_res_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_PQ_RES_CAP)),
        )
        # Pre-allocate insert() scratch — reused across all inserts.
        self.ins_sl_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_INSERT_SL_CAP)),
        )
        self.ins_sl_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_INSERT_SL_CAP)),
        )
        self.ins_picked_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_INSERT_PICKED_CAP)),
        )
        self.ins_picked_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_INSERT_PICKED_CAP)),
        )
        # compact_overflows() / rebuild pool — needs M0_STORAGE + 1 slots
        # for the rebuild path (nb's M0_STORAGE existing neighbors + 1 new).
        # Round up to 2× M0_STORAGE for headroom + alignment.
        self.ins_pool_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_M0_STORAGE * 2)),
        )
        self.ins_pool_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_M0_STORAGE * 2)),
        )
        # Rebuild output: heuristic returns ≤ M0=64 neighbors; pad to M0_STORAGE.
        self.ins_rebuilt_dist = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Float32](KNN_HNSW_M0_STORAGE)),
        )
        self.ins_rebuilt_id = UnsafePointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(alloc[Int32](KNN_HNSW_M0_STORAGE)),
        )
        self.entry_point = -1
        self.max_layer = 0
        self.num_nodes = 0
        self.visited_gen = 0
        self.built = True

    def free_buffers(mut self):
        if is_not_null(self.neighbors):
            self.neighbors.free()
            self.neighbors = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.neighbor_counts):
            self.neighbor_counts.free()
            self.neighbor_counts = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.node_level):
            self.node_level.free()
            self.node_level = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.node_visited_gen):
            self.node_visited_gen.free()
            self.node_visited_gen = null_ptr[UInt32, MutUntrackedOrigin]()
        if is_not_null(self.pq_cand_dist):
            self.pq_cand_dist.free()
            self.pq_cand_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.pq_cand_id):
            self.pq_cand_id.free()
            self.pq_cand_id = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.pq_res_dist):
            self.pq_res_dist.free()
            self.pq_res_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.pq_res_id):
            self.pq_res_id.free()
            self.pq_res_id = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.ins_sl_dist):
            self.ins_sl_dist.free()
            self.ins_sl_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.ins_sl_id):
            self.ins_sl_id.free()
            self.ins_sl_id = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.ins_picked_dist):
            self.ins_picked_dist.free()
            self.ins_picked_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.ins_picked_id):
            self.ins_picked_id.free()
            self.ins_picked_id = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.ins_pool_dist):
            self.ins_pool_dist.free()
            self.ins_pool_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.ins_pool_id):
            self.ins_pool_id.free()
            self.ins_pool_id = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.ins_rebuilt_dist):
            self.ins_rebuilt_dist.free()
            self.ins_rebuilt_dist = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.ins_rebuilt_id):
            self.ins_rebuilt_id.free()
            self.ins_rebuilt_id = null_ptr[Int32, MutUntrackedOrigin]()
        self.built = False
        self.entry_point = -1
        self.num_nodes = 0

    # ── ef-search at one layer ────────────────────────────────────────────
    # Standard "ef-search" routine (Malkov-Yashunin 2018, Algorithm 2).
    # gh #42 follow-up: heap-based PQ, O(log ef) push/pop instead of the
    # original O(ef) sorted-array shifts. Two parallel structures:
    #   candidates: min-heap on distance, root = closest (next to expand)
    #   results:    max-heap on distance, root = worst (bounded at ef)
    # Stop condition: if results full and the closest candidate is worse
    # than the current worst result, no further expansion can improve
    # results — break.
    #
    # Final pass pops the result max-heap into out_* in DESCENDING index
    # order so out_* ends up ASCENDING — the contract expected by both
    # `_select_heuristic` (insert path) and the top-level search (which
    # writes out the first k entries directly).

    def _search_layer(
        mut self,
        query: UnsafePointer[Float32, MutUntrackedOrigin],
        entry: Int,
        ef: Int,
        layer: Int,
        out_dist: UnsafePointer[Float32, MutUntrackedOrigin],
        out_id: UnsafePointer[Int32, MutUntrackedOrigin],
    ) -> Int:
        """Returns the number of populated entries (≤ ef), sorted ascending
        by distance. out_dist/out_id are caller-owned scratch (≥ ef long).
        Uses pre-allocated heap scratch (no per-query alloc)."""
        if entry < 0:
            return 0
        # Cap ef to result-heap capacity (configured at build time).
        var ef_eff = ef
        if ef_eff > KNN_HNSW_PQ_RES_CAP:
            ef_eff = KNN_HNSW_PQ_RES_CAP

        # Pre-allocated PQ scratch (zero per-query alloc).
        var cand_dist = self.pq_cand_dist
        var cand_id = self.pq_cand_id
        var res_dist = self.pq_res_dist
        var res_id = self.pq_res_id
        var cand_n = 0
        var res_n = 0

        # Bump generation stamp for this search. If it wraps, reset all stamps.
        self.visited_gen = self.visited_gen + 1
        if self.visited_gen == 0:
            unsafe_memset(self.node_visited_gen.bitcast[UInt8](), 0, self.max_elements * 4)
            self.visited_gen = 1

        var d0 = self._distance(query, entry)
        cand_dist[0] = d0
        cand_id[0] = Int32(entry)
        cand_n = 1
        res_dist[0] = d0
        res_id[0] = Int32(entry)
        res_n = 1
        self.node_visited_gen[entry] = self.visited_gen

        while cand_n > 0:
            # Closest candidate is at the min-heap root.
            var c_dist = cand_dist[0]
            var c_id = Int(cand_id[0])
            # Stop if we've filled results and the closest cand is worse
            # than the current worst result.
            if res_n >= ef_eff and c_dist > res_dist[0]:
                break
            # Pop min: swap root with last, sift down.
            cand_n -= 1
            if cand_n > 0:
                cand_dist[0] = cand_dist[cand_n]
                cand_id[0] = cand_id[cand_n]
                _heap_sift_down_min(cand_dist, cand_id, 0, cand_n)

            # gh #42 follow-up: collect unvisited neighbors first, then
            # batch the distance compute in groups of 4 via the per-vector
            # SQ8 batch4 kernel. Sharing one query load across 4 dequants
            # attacks the memory-stall bottleneck on random graph traversal
            # — the dominant cost beyond ~250 µs of identifiable work
            # (per Codex consultation).
            var nbase = self._slot_idx(c_id, layer)
            var nc = Int(self.neighbor_counts[self._count_idx(c_id, layer)])
            # Stack-allocated scratch sized at M0_STORAGE (build-time cap;
            # neighbor list can hit 2× M0 in lazy-pruning mode before
            # compact_overflows runs at end of bulk insert).
            var unvisited_ids = Array[Int32, KNN_HNSW_M0_STORAGE](fill=Int32(0))
            var unvisited_n = 0
            for ni in range(nc):
                var n = Int(self.neighbors[nbase + ni])
                if n < 0:
                    continue
                if self.node_visited_gen[n] == self.visited_gen:
                    continue
                self.node_visited_gen[n] = self.visited_gen
                unvisited_ids[unvisited_n] = Int32(n)
                unvisited_n += 1

            # Process unvisited neighbors in groups of 4 (batch4) + tail.
            var ui = 0
            while ui + 4 <= unvisited_n:
                var n0 = Int(unvisited_ids[ui])
                var n1 = Int(unvisited_ids[ui + 1])
                var n2 = Int(unvisited_ids[ui + 2])
                var n3 = Int(unvisited_ids[ui + 3])
                var d4 = l2_distance_fp32_int8_pervec_batch4(
                    query,
                    self.embeddings_int8_ref + n0 * self.dim,
                    self.embeddings_int8_ref + n1 * self.dim,
                    self.embeddings_int8_ref + n2 * self.dim,
                    self.embeddings_int8_ref + n3 * self.dim,
                    self.dim,
                    self.sq_min_ref[n0], self.sq_range_ref[n0],
                    self.sq_min_ref[n1], self.sq_range_ref[n1],
                    self.sq_min_ref[n2], self.sq_range_ref[n2],
                    self.sq_min_ref[n3], self.sq_range_ref[n3],
                )
                # Admit each of the 4 results into the heaps.
                for k in range(4):
                    var nd = d4[k]
                    var nk = Int(unvisited_ids[ui + k])
                    var added_to_res = False
                    if res_n < ef_eff:
                        res_dist[res_n] = nd
                        res_id[res_n] = Int32(nk)
                        res_n += 1
                        _heap_sift_up_max(res_dist, res_id, res_n - 1)
                        added_to_res = True
                    elif nd < res_dist[0]:
                        res_dist[0] = nd
                        res_id[0] = Int32(nk)
                        _heap_sift_down_max(res_dist, res_id, 0, res_n)
                        added_to_res = True
                    if added_to_res and cand_n < KNN_HNSW_PQ_CAND_CAP:
                        cand_dist[cand_n] = nd
                        cand_id[cand_n] = Int32(nk)
                        cand_n += 1
                        _heap_sift_up_min(cand_dist, cand_id, cand_n - 1)
                ui += 4

            # Tail: process remaining 0-3 neighbors with the single-distance kernel.
            while ui < unvisited_n:
                var nk = Int(unvisited_ids[ui])
                var nd = self._distance(query, nk)
                var added_to_res = False
                if res_n < ef_eff:
                    res_dist[res_n] = nd
                    res_id[res_n] = Int32(nk)
                    res_n += 1
                    _heap_sift_up_max(res_dist, res_id, res_n - 1)
                    added_to_res = True
                elif nd < res_dist[0]:
                    res_dist[0] = nd
                    res_id[0] = Int32(nk)
                    _heap_sift_down_max(res_dist, res_id, 0, res_n)
                    added_to_res = True
                if added_to_res and cand_n < KNN_HNSW_PQ_CAND_CAP:
                    cand_dist[cand_n] = nd
                    cand_id[cand_n] = Int32(nk)
                    cand_n += 1
                    _heap_sift_up_min(cand_dist, cand_id, cand_n - 1)
                ui += 1

        # Pop result max-heap into out_* descending index → ascending order.
        var n_out = res_n
        var i = n_out - 1
        while i >= 0:
            out_dist[i] = res_dist[0]
            out_id[i] = res_id[0]
            res_n -= 1
            if res_n > 0:
                res_dist[0] = res_dist[res_n]
                res_id[0] = res_id[res_n]
                _heap_sift_down_max(res_dist, res_id, 0, res_n)
            i -= 1
        return n_out

    # ── Heuristic neighbor selection (M&Y 2018 Algorithm 4) ─────────────
    # Standard "M-closest" pruning collapses graph quality at scale: nodes
    # form a tight clique around the densest region and far queries can't
    # navigate in. Empirically: recall fell 1.00 → 0.90 → 0.50 across
    # n ∈ {2K, 8K, 30K} with the simple M-closest selection.
    #
    # Heuristic: a candidate `c` is rejected if some already-selected
    # neighbor `r` is closer to `c` than the query is — i.e. `c` would be
    # better served by routing through `r`. Keeps neighbors spread across
    # different angular directions, preserving navigability.
    #
    # Inputs:
    #   query: the query vector (or, for eviction-pruning, the "owner"
    #          node's embedding — that node is the one whose neighbor list
    #          we're rebuilding)
    #   cands_dist[n_cand] / cands_id[n_cand]: candidate pool, sorted ASC
    #   m: max neighbors to pick
    # Outputs:
    #   out_id[m] / out_dist[m]: selected neighbors, ASC by distance
    # Returns: count of selected.
    def _select_heuristic(
        self,
        query: UnsafePointer[Float32, MutUntrackedOrigin],
        cands_dist: UnsafePointer[Float32, MutUntrackedOrigin],
        cands_id: UnsafePointer[Int32, MutUntrackedOrigin],
        n_cand: Int,
        m: Int,
        out_id: UnsafePointer[Int32, MutUntrackedOrigin],
        out_dist: UnsafePointer[Float32, MutUntrackedOrigin],
    ) -> Int:
        var picked = 0
        var d = self.dim
        for ci in range(n_cand):
            if picked >= m:
                break
            var cid = Int(cands_id[ci])
            if cid < 0: continue
            var c_to_q = cands_dist[ci]
            var keep = True
            # Reject if some already-picked r is closer to c than q is.
            # gh #42: kept on FP32-FP32 — both `cid` and `rid` are base
            # vectors, and `rid` is a small repeated set (already-picked,
            # ≤ M=32) that lives in L1 across the inner loop. The FP32-FP32
            # path has half the compute of the FP32-INT8 dequant kernel, so
            # the bandwidth saving from INT8 is wasted when both sides are
            # cache-resident. The INT8 win is on the search hot path
            # (_distance), where node visits cycle through 30K unique
            # vectors that don't fit in L1.
            for ri in range(picked):
                var rid = Int(out_id[ri])
                if rid < 0: continue
                var c_to_r = _l2_distance_fp32(
                    self.embeddings_ref + cid * d,
                    self.embeddings_ref + rid * d,
                    d,
                )
                if c_to_r < c_to_q:
                    keep = False
                    break
            if keep:
                out_id[picked] = Int32(cid)
                out_dist[picked] = c_to_q
                picked += 1
        return picked

    # ── 1-NN greedy descent at upper layers ──────────────────────────────
    def _greedy_descent_layer(
        mut self,
        query: UnsafePointer[Float32, MutUntrackedOrigin],
        entry: Int,
        layer: Int,
    ) -> Int:
        """Walk from `entry` along `layer`-edges to a local minimum. Returns
        the node id. Used to find a good entry for the next-lower layer.
        Each iteration picks the BEST improving neighbor (not the first), so
        the trajectory is consistent with HNSW's standard greedy semantics."""
        var cur = entry
        var cur_dist = self._distance(query, cur)
        while True:
            var nbase = self._slot_idx(cur, layer)
            var nc = Int(self.neighbor_counts[self._count_idx(cur, layer)])
            var best_n = -1
            var best_d = cur_dist
            for ni in range(nc):
                var n = Int(self.neighbors[nbase + ni])
                if n < 0: continue
                var nd = self._distance(query, n)
                if nd < best_d:
                    best_d = nd
                    best_n = n
            if best_n < 0:
                break
            cur = best_n
            cur_dist = best_d
        return cur

    # ── Insert a new node ────────────────────────────────────────────────
    def insert(mut self, node_id: Int):
        """Insert `node_id` into the graph. The embedding at
        `embeddings_ref + node_id*dim` must already be populated."""
        if not self.built or node_id < 0 or node_id >= self.max_elements:
            return
        var node_layer = _knn_hnsw_node_level(node_id, self.rng_state)
        self.node_level[node_id] = Int8(node_layer)
        # Initialize all of this node's neighbor counts to 0 (they were
        # zero from build() but only for nodes ≤ then-num_nodes).
        for L in range(KNN_HNSW_MAX_LAYERS):
            self.neighbor_counts[self._count_idx(node_id, L)] = 0

        if self.entry_point < 0:
            self.entry_point = node_id
            self.max_layer = node_layer
            self.num_nodes += 1
            return

        # Greedy descent from max_layer down to node_layer+1.
        var query_emb = self.embeddings_ref + node_id * self.dim
        var ep = self.entry_point
        for L in range(self.max_layer, node_layer, -1):
            ep = self._greedy_descent_layer(query_emb, ep, L)

        # ef-search + neighbor selection at each layer from node_layer down.
        # gh #42 follow-up: pre-allocated scratch (ins_sl_*, ins_picked_*)
        # eliminates ~25 allocs per insert. Reciprocal back-link is LAZY:
        # nb's neighbor list grows up to KNN_HNSW_M{0,}_STORAGE (2× the
        # heuristic cap) before triggering a rebuild — most nodes never
        # saturate during the bulk insert window, so the M² rebuild fires
        # far less often. compact_overflows() at end of bulk insert (in
        # _maybe_build_hnsw) prunes any saturated rows back to M0/M.
        var sl_dist = self.ins_sl_dist
        var sl_id = self.ins_sl_id
        var picked_id = self.ins_picked_id
        var picked_dist = self.ins_picked_dist
        var top = node_layer
        if top > self.max_layer: top = self.max_layer
        for L in range(top, -1, -1):
            var ef_c = KNN_HNSW_EF_CONSTRUCTION
            var found = self._search_layer(query_emb, ep, ef_c, L, sl_dist, sl_id)
            var max_n = self._max_neighbors(L)
            var to_pick = self._select_heuristic(
                query_emb, sl_dist, sl_id, found, max_n, picked_id, picked_dist
            )

            # Connect node_id ↔ selected, bidirectionally.
            var nb_max_storage = self._max_neighbors_storage(L)
            var node_slot = self._slot_idx(node_id, L)
            for k in range(to_pick):
                var nb = Int(picked_id[k])
                self.neighbors[node_slot + k] = Int32(nb)
                # Lazy back-link: append to nb's list up to STORAGE cap.
                # The original M² rebuild path only fires if this back-link
                # is the one that pushes nb past STORAGE — rare during
                # bulk insert at typical M0=64 → STORAGE=128. compact_
                # overflows() runs the same heuristic at end of bulk to
                # prune to the M0/M target.
                var nb_count_idx = self._count_idx(nb, L)
                var nb_count = Int(self.neighbor_counts[nb_count_idx])
                var nb_slot = self._slot_idx(nb, L)
                if nb_count < nb_max_storage:
                    self.neighbors[nb_slot + nb_count] = Int32(node_id)
                    self.neighbor_counts[nb_count_idx] = Int32(nb_count + 1)
                else:
                    # nb hit the STORAGE cap (= 2× heuristic cap). Run the
                    # eager heuristic rebuild over nb's neighbors ∪ {node}
                    # to compact back to the heuristic target before
                    # admitting the new back-link.
                    var nb_max = self._max_neighbors(L)
                    var nb_emb = self.embeddings_ref + nb * self.dim
                    var pool_n = nb_count + 1
                    var pool_dist = self.ins_pool_dist
                    var pool_id = self.ins_pool_id
                    for j in range(nb_count):
                        var jid = Int(self.neighbors[nb_slot + j])
                        var jd = self._distance(nb_emb, jid) if jid >= 0 else Float32(1e38)
                        var pp = j
                        while pp > 0 and pool_dist[pp - 1] > jd:
                            pool_dist[pp] = pool_dist[pp - 1]
                            pool_id[pp] = pool_id[pp - 1]
                            pp -= 1
                        pool_dist[pp] = jd
                        pool_id[pp] = Int32(jid)
                    var node_d_to_nb = self._distance(nb_emb, node_id)
                    var pp2 = nb_count
                    while pp2 > 0 and pool_dist[pp2 - 1] > node_d_to_nb:
                        pool_dist[pp2] = pool_dist[pp2 - 1]
                        pool_id[pp2] = pool_id[pp2 - 1]
                        pp2 -= 1
                    pool_dist[pp2] = node_d_to_nb
                    pool_id[pp2] = Int32(node_id)

                    # Use ins_rebuilt_* so picked_*/picked_id stay intact
                    # for the outer `for k in range(to_pick)` loop.
                    var rebuilt_id = self.ins_rebuilt_id
                    var rebuilt_dist = self.ins_rebuilt_dist
                    var rebuilt_n = self._select_heuristic(
                        nb_emb, pool_dist, pool_id, pool_n, nb_max,
                        rebuilt_id, rebuilt_dist,
                    )
                    for j in range(rebuilt_n):
                        self.neighbors[nb_slot + j] = rebuilt_id[j]
                    self.neighbor_counts[nb_count_idx] = Int32(rebuilt_n)
            self.neighbor_counts[self._count_idx(node_id, L)] = Int32(to_pick)
            # Use closest selected neighbor as entry for the next-lower layer.
            if to_pick > 0:
                ep = Int(picked_id[0])
            elif found > 0:
                ep = Int(sl_id[0])

        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id
        self.num_nodes += 1

    # ── Compact overflowed rows after bulk insert ────────────────────────
    # gh #42 follow-up: insert() leaves neighbor lists at up to STORAGE
    # cap. compact_overflows() walks every (node, layer) and, for each row
    # that exceeds the heuristic target M0/M, runs the same heuristic
    # neighbor selection to prune it back to M0/M. Same final selection
    # rule as the original eager rebuild — the only change is timing.
    def compact_overflows(mut self):
        if not self.built or self.num_nodes <= 0: return
        var pool_dist = self.ins_pool_dist
        var pool_id = self.ins_pool_id
        var rebuilt_dist = self.ins_rebuilt_dist
        var rebuilt_id = self.ins_rebuilt_id
        for node in range(self.num_nodes):
            var nb_emb = self.embeddings_ref + node * self.dim
            for L in range(KNN_HNSW_MAX_LAYERS):
                var ci = self._count_idx(node, L)
                var count = Int(self.neighbor_counts[ci])
                var target = self._max_neighbors(L)
                if count <= target:
                    continue
                # Build pool ASC by distance to `node`.
                var slot = self._slot_idx(node, L)
                for j in range(count):
                    var jid = Int(self.neighbors[slot + j])
                    var jd = self._distance(nb_emb, jid) if jid >= 0 else Float32(1e38)
                    var pp = j
                    while pp > 0 and pool_dist[pp - 1] > jd:
                        pool_dist[pp] = pool_dist[pp - 1]
                        pool_id[pp] = pool_id[pp - 1]
                        pp -= 1
                    pool_dist[pp] = jd
                    pool_id[pp] = Int32(jid)
                # Heuristic prune ASC pool → ≤ target diverse neighbors.
                var rebuilt_n = self._select_heuristic(
                    nb_emb, pool_dist, pool_id, count, target,
                    rebuilt_id, rebuilt_dist,
                )
                for j in range(rebuilt_n):
                    self.neighbors[slot + j] = rebuilt_id[j]
                self.neighbor_counts[ci] = Int32(rebuilt_n)

    # ── Top-k search ─────────────────────────────────────────────────────
    def search(
        mut self,
        query: UnsafePointer[Float32, MutUntrackedOrigin],
        k: Int,
        ef: Int,
        out_dist: UnsafePointer[Float32, MutUntrackedOrigin],
        out_id: UnsafePointer[Int32, MutUntrackedOrigin],
    ) -> Int:
        """Returns the number of populated results (≤ k). Results sorted ASC
        by distance. `ef` is the search beam width — should be ≥ k for
        recall; KNN_HNSW_DEFAULT_EF is a sane default."""
        if not self.built or self.entry_point < 0 or k <= 0:
            return 0

        # Greedy descent from max_layer down to layer 1.
        var ep = self.entry_point
        for L in range(self.max_layer, 0, -1):
            ep = self._greedy_descent_layer(query, ep, L)

        # ef-search at layer 0 with ef = max(ef, k).
        var ef_eff = ef
        if ef_eff < k: ef_eff = k
        var sl_dist_p = alloc[Float32](ef_eff)
        var sl_id_p = alloc[Int32](ef_eff)
        var sl_dist = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(sl_dist_p))
        var sl_id = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(sl_id_p))
        var found = self._search_layer(query, ep, ef_eff, 0, sl_dist, sl_id)

        var n = found
        if n > k: n = k
        for i in range(n):
            out_dist[i] = sl_dist[i]
            out_id[i] = sl_id[i]
        sl_dist.free()
        sl_id.free()
        return n


struct KNNDatastore(Movable, Copyable):
    """One kNN-LM datastore: token_ids[N], embeddings[N × dim] flat arrays.
    Optional `hnsw` field is built lazily once count crosses
    KNN_HNSW_BUILD_THRESHOLD; until then queries scan brute-force.

    Storage is dual: `embeddings` (FP32) is the source-of-truth for the
    brute-force path and for restore/reranking. `embeddings_int8` is the
    per-vector SQ8 mirror used by the HNSW asymmetric distance kernel
    (gh #42). SQ8 cuts vector size 4× (3 KB → 768 B at dim=768), keeping
    the working set in L2 cache and roughly tripling HNSW search speed.
    """
    var name: Array[UInt8, KNN_DS_NAME_INLINE]
    var name_len: Int
    var active: Bool
    var dim: Int
    var max_entries: Int
    var count: Int
    var token_ids: UnsafePointer[Int32, MutUntrackedOrigin]
    var embeddings: UnsafePointer[Float32, MutUntrackedOrigin]
    # gh #42: per-vector SQ8 mirror for the HNSW path. dequant[d] =
    # (embeddings_int8[d] + 127) * sq_range[i]/254 + sq_min[i].
    var embeddings_int8: UnsafePointer[Int8, MutUntrackedOrigin]
    var sq_min: UnsafePointer[Float32, MutUntrackedOrigin]
    var sq_range: UnsafePointer[Float32, MutUntrackedOrigin]
    var hnsw: KNNHNSW

    def __init__(out self):
        self.name = Array[UInt8, KNN_DS_NAME_INLINE](fill=UInt8(0))
        self.name_len = 0
        self.active = False
        self.dim = 0
        self.max_entries = 0
        self.count = 0
        self.token_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.embeddings = null_ptr[Float32, MutUntrackedOrigin]()
        self.embeddings_int8 = null_ptr[Int8, MutUntrackedOrigin]()
        self.sq_min = null_ptr[Float32, MutUntrackedOrigin]()
        self.sq_range = null_ptr[Float32, MutUntrackedOrigin]()
        self.hnsw = KNNHNSW()

    def __copyinit__(out self, existing: Self):
        self.name = Array[UInt8, KNN_DS_NAME_INLINE](fill=UInt8(0))
        for i in range(KNN_DS_NAME_INLINE):
            self.name[i] = existing.name[i]
        self.name_len = existing.name_len
        self.active = existing.active
        self.dim = existing.dim
        self.max_entries = existing.max_entries
        self.count = existing.count
        self.token_ids = existing.token_ids
        self.embeddings = existing.embeddings
        self.embeddings_int8 = existing.embeddings_int8
        self.sq_min = existing.sq_min
        self.sq_range = existing.sq_range
        self.hnsw = existing.hnsw

    def __moveinit__(out self, var existing: Self):
        self.name = Array[UInt8, KNN_DS_NAME_INLINE](fill=UInt8(0))
        for i in range(KNN_DS_NAME_INLINE):
            self.name[i] = existing.name[i]
        self.name_len = existing.name_len
        self.active = existing.active
        self.dim = existing.dim
        self.max_entries = existing.max_entries
        self.count = existing.count
        self.token_ids = existing.token_ids
        self.embeddings = existing.embeddings
        self.embeddings_int8 = existing.embeddings_int8
        self.sq_min = existing.sq_min
        self.sq_range = existing.sq_range
        self.hnsw = existing.hnsw^


struct KNNLMIndex:
    """Per-worker kNN-LM datastore registry (linear-probe table by name)."""
    var enabled: Bool
    var datastores: Array[KNNDatastore, MAX_KNN_DATASTORES]

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.datastores = Array[KNNDatastore, MAX_KNN_DATASTORES](
            fill=KNNDatastore(),
        )

    @always_inline
    def _name_eq(self, slot: Int, name_ptr: UnsafePointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Bool:
        if not self.datastores[slot].active:
            return False
        if self.datastores[slot].name_len != name_len:
            return False
        for i in range(name_len):
            if self.datastores[slot].name[i] != name_ptr[i]:
                return False
        return True

    def find(self, name_ptr: UnsafePointer[UInt8, MutUntrackedOrigin], name_len: Int) -> Int:
        """Return slot index, or -1 if not found."""
        for s in range(MAX_KNN_DATASTORES):
            if self._name_eq(s, name_ptr, name_len):
                return s
        return -1

    def create(
        mut self,
        name_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
        name_len: Int,
        dim: Int,
        max_entries: Int,
    ) -> Int:
        """Allocate a new datastore. Returns slot idx, or -1 if full / duplicate / invalid."""
        if name_len <= 0 or name_len > KNN_DS_NAME_INLINE:
            return -1
        if dim <= 0 or max_entries <= 0:
            return -1
        if self.find(name_ptr, name_len) >= 0:
            return -1  # duplicate
        for s in range(MAX_KNN_DATASTORES):
            if not self.datastores[s].active:
                self.datastores[s].active = True
                self.datastores[s].name_len = name_len
                for i in range(name_len):
                    self.datastores[s].name[i] = name_ptr[i]
                self.datastores[s].dim = dim
                self.datastores[s].max_entries = max_entries
                self.datastores[s].count = 0
                self.datastores[s].token_ids = UnsafePointer[Int32, MutUntrackedOrigin](
                    unsafe_from_address=Int(alloc[Int32](max_entries)),
                )
                self.datastores[s].embeddings = UnsafePointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=Int(alloc[Float32](max_entries * dim)),
                )
                # gh #42: per-vector SQ8 mirror for HNSW asymmetric distance.
                self.datastores[s].embeddings_int8 = UnsafePointer[Int8, MutUntrackedOrigin](
                    unsafe_from_address=Int(alloc[Int8](max_entries * dim)),
                )
                self.datastores[s].sq_min = UnsafePointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=Int(alloc[Float32](max_entries)),
                )
                self.datastores[s].sq_range = UnsafePointer[Float32, MutUntrackedOrigin](
                    unsafe_from_address=Int(alloc[Float32](max_entries)),
                )
                return s
        return -1

    def drop(
        mut self,
        name_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
        name_len: Int,
    ) -> Bool:
        var s = self.find(name_ptr, name_len)
        if s < 0:
            return False
        # Free graph buffers BEFORE the parallel arrays so the borrowed
        # embeddings_ref isn't pointing at freed memory at any point.
        self.datastores[s].hnsw.free_buffers()
        if is_not_null(self.datastores[s].token_ids):
            self.datastores[s].token_ids.free()
            self.datastores[s].token_ids = null_ptr[Int32, MutUntrackedOrigin]()
        if is_not_null(self.datastores[s].embeddings):
            self.datastores[s].embeddings.free()
            self.datastores[s].embeddings = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.datastores[s].embeddings_int8):
            self.datastores[s].embeddings_int8.free()
            self.datastores[s].embeddings_int8 = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.datastores[s].sq_min):
            self.datastores[s].sq_min.free()
            self.datastores[s].sq_min = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.datastores[s].sq_range):
            self.datastores[s].sq_range.free()
            self.datastores[s].sq_range = null_ptr[Float32, MutUntrackedOrigin]()
        self.datastores[s].active = False
        self.datastores[s].count = 0
        return True

    @always_inline
    def _maybe_build_hnsw(mut self, slot: Int):
        """Build HNSW lazily when count crosses threshold. Bulk-inserts all
        existing entries. Idempotent: subsequent calls are no-ops once
        `hnsw.built == True`."""
        if self.datastores[slot].hnsw.built:
            return
        if self.datastores[slot].count < KNN_HNSW_BUILD_THRESHOLD:
            return
        self.datastores[slot].hnsw.build(
            self.datastores[slot].max_entries,
            self.datastores[slot].dim,
            self.datastores[slot].embeddings,
            self.datastores[slot].embeddings_int8,
            self.datastores[slot].sq_min,
            self.datastores[slot].sq_range,
        )
        # Bulk-insert existing entries. Each insert lazily appends back-links
        # up to KNN_HNSW_M{0,}_STORAGE; compact_overflows() at the end runs
        # the heuristic prune over saturated rows in one batch (gh #42).
        for i in range(self.datastores[slot].count):
            self.datastores[slot].hnsw.insert(i)
        self.datastores[slot].hnsw.compact_overflows()

    def store_one(
        mut self, slot: Int, next_token_id: Int32,
        emb: UnsafePointer[Float32, MutUntrackedOrigin],
    ) -> Bool:
        """Append one (token_id, embedding) pair. False if datastore full."""
        if slot < 0 or slot >= MAX_KNN_DATASTORES or not self.datastores[slot].active:
            return False
        var c = self.datastores[slot].count
        if c >= self.datastores[slot].max_entries:
            return False
        self.datastores[slot].token_ids[c] = next_token_id
        var d = self.datastores[slot].dim
        var dst = self.datastores[slot].embeddings + c * d
        unsafe_memcpy(dest=dst.bitcast[UInt8](), src=emb.bitcast[UInt8](),
               count=d * 4)
        # gh #42: mirror into per-vector SQ8 for HNSW asymmetric distance.
        var i8_dst = self.datastores[slot].embeddings_int8 + c * d
        var qm_qr = _vec_sq8_quantize(dst, i8_dst, d)
        self.datastores[slot].sq_min[c] = qm_qr[0]
        self.datastores[slot].sq_range[c] = qm_qr[1]
        self.datastores[slot].count = c + 1
        # HNSW lifecycle: build at threshold, then incrementally insert.
        if self.datastores[slot].hnsw.built:
            self.datastores[slot].hnsw.insert(c)
        else:
            self._maybe_build_hnsw(slot)
        return True

    def store_batch(
        mut self, slot: Int, n: Int,
        token_ids: UnsafePointer[Int32, MutUntrackedOrigin],
        embeddings: UnsafePointer[Float32, MutUntrackedOrigin],
    ) -> Int:
        """Bulk append. Returns number of pairs actually stored (≤ n)."""
        if slot < 0 or slot >= MAX_KNN_DATASTORES or not self.datastores[slot].active:
            return 0
        var c = self.datastores[slot].count
        var room = self.datastores[slot].max_entries - c
        var to_store = min(n, room)
        if to_store <= 0:
            return 0
        var d = self.datastores[slot].dim
        unsafe_memcpy(
            dest=(self.datastores[slot].token_ids + c).bitcast[UInt8](),
            src=token_ids.bitcast[UInt8](), count=to_store * 4,
        )
        unsafe_memcpy(
            dest=(self.datastores[slot].embeddings + c * d).bitcast[UInt8](),
            src=embeddings.bitcast[UInt8](), count=to_store * d * 4,
        )
        # gh #42: per-vector SQ8 mirror. Quantize each new row before HNSW
        # build/insert so the asymmetric distance kernel reads populated INT8.
        for i in range(to_store):
            var slot_idx = c + i
            var src_fp32 = self.datastores[slot].embeddings + slot_idx * d
            var dst_i8 = self.datastores[slot].embeddings_int8 + slot_idx * d
            var qm_qr = _vec_sq8_quantize(src_fp32, dst_i8, d)
            self.datastores[slot].sq_min[slot_idx] = qm_qr[0]
            self.datastores[slot].sq_range[slot_idx] = qm_qr[1]
        self.datastores[slot].count = c + to_store
        # HNSW lifecycle: bulk-build at threshold, else insert each new entry.
        if self.datastores[slot].hnsw.built:
            for i in range(c, c + to_store):
                self.datastores[slot].hnsw.insert(i)
        else:
            self._maybe_build_hnsw(slot)
        return to_store

    def query(
        mut self, slot: Int, k: Int,
        query_emb: UnsafePointer[Float32, MutUntrackedOrigin],
        out_token_ids: UnsafePointer[Int32, MutUntrackedOrigin],
        out_distances: UnsafePointer[Float32, MutUntrackedOrigin],
    ) -> Int:
        """Top-k. Returns number of results (min(k, count)). Routes through
        HNSW when the graph is built, brute-force otherwise.

        out_token_ids and out_distances are pre-allocated by caller (length k).
        Distances are sorted ascending. Padded with INFINITY/-1 if fewer than k matches.
        """
        if slot < 0 or slot >= MAX_KNN_DATASTORES or not self.datastores[slot].active:
            return 0
        var c = self.datastores[slot].count
        var d = self.datastores[slot].dim
        var top = min(k, c)
        if top <= 0:
            return 0
        # Initialize output arrays with sentinels: distance = +INF, token_id = -1.
        for i in range(k):
            out_distances[i] = 1e38
            out_token_ids[i] = -1

        # ── HNSW path ────────────────────────────────────────────────────
        # When the graph is built, use ANN with FP32-rerank on the returned
        # top-K. Rerank is essentially free (k small distances) and recovers
        # any approximation slop from the HNSW traversal — the graph returns
        # internal node indices into our embeddings array, so we can look up
        # both the token_id and recompute the FP32 distance directly.
        if self.datastores[slot].hnsw.built:
            var node_ids_p = alloc[Int32](k)
            var node_dists_p = alloc[Float32](k)
            var node_ids = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(node_ids_p))
            var node_dists = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(node_dists_p))
            # gh #42: per-vector SQ8 base + asymmetric FP32×INT8 kernel +
            # heap-based PQ + per-vec batch-4 distance kernel together cut
            # 30K median latency 5.17 → ~1.94 ms with recall stable at 0.90.
            # ef policy unchanged from gh #38 (sqrt growth, cap 400). Per
            # Codex/Gemini consultation, the dominant cost in the original
            # 5.17 ms was memory stall on random graph traversal, not
            # distance compute or sort overhead — `batch4` shares the
            # query load across 4 dequant FMA chains, attacking that stall.
            var ef = k * 8
            if ef < KNN_HNSW_DEFAULT_EF: ef = KNN_HNSW_DEFAULT_EF
            if c > KNN_HNSW_BUILD_THRESHOLD:
                var scale_ratio = Float64(c) / Float64(KNN_HNSW_BUILD_THRESHOLD)
                var scaled_ef = Int(Float64(KNN_HNSW_DEFAULT_EF) * sqrt(scale_ratio))
                if scaled_ef > ef: ef = scaled_ef
                if ef > 400:
                    ef = 400
            var found = self.datastores[slot].hnsw.search(query_emb, k, ef, node_dists, node_ids)
            for i in range(found):
                var nid = Int(node_ids[i])
                if nid < 0 or nid >= c:
                    continue
                # node_ids[i] is an entry index into the embeddings/token_ids
                # arrays. Map directly to the next-token id; the distance
                # already came from FP32 _l2 inside _search_layer.
                out_token_ids[i] = self.datastores[slot].token_ids[nid]
                out_distances[i] = node_dists[i]
            node_ids.free()
            node_dists.free()
            return found

        # ── Brute-force path (small datastores) ──────────────────────────
        # Maintain top-k by linear scan + sorted insertion.
        for ei in range(c):
            var dist = _l2_distance_fp32(
                query_emb,
                self.datastores[slot].embeddings + ei * d,
                d,
            )
            if dist < out_distances[k - 1]:
                var j = k - 1
                while j > 0 and out_distances[j - 1] > dist:
                    out_distances[j] = out_distances[j - 1]
                    out_token_ids[j] = out_token_ids[j - 1]
                    j -= 1
                out_distances[j] = dist
                out_token_ids[j] = self.datastores[slot].token_ids[ei]
        return top
