from src.common.ptr import is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.math import sqrt
from std.memory import alloc
from std.atomic import Atomic, Ordering
from src.common.lock_free import ShardQueryBus
# gh #87.1: ivf_pq module deleted.
from std.collections import Array


# gh #5 / #87.2: HNSW multi-worker sharded ingest is force-disabled. Until
# the merge-path SIGSEGV / recall ≈ 0 at 500K+ on Linux multi-worker is
# root-caused, `num_shards = 1` is pinned at main.mojo:591.
#
# Flipping this alias back to True re-enters the sharded-ingest paths gated
# by `comptime if HNSW_SHARDED_INGEST_ENABLED:`:
#   - src/vector/hnsw.mojo  : `build_index_from_shared` id-modulo skip
#   - src/network/engine.mojo : 3× housekeeping blocks (uring/epoll/kqueue)
#     covering shard-build coordination + cross-worker shard-query bus serve
#
# Pattern mirrors the disabled-then-deleted `KV_BUS_ENABLED` flag (gh #48 /
# gh #85b). When gh #5 lands, decide: delete the gated bodies entirely, or
# flip the alias and ship sharded ingest again.
comptime HNSW_SHARDED_INGEST_ENABLED : Bool = False

struct HNSWNode(Movable):
    var id: Int
    var vector: Pointer[Int8, MutUntrackedOrigin]
    var neighbors: Pointer[UInt32, MutUntrackedOrigin]
    var neighbor_counts: Pointer[UInt32, MutUntrackedOrigin]
    var max_level: Int
    var M: Int
    var inline_vector: SIMD[DType.int8, 16]

    def __init__(out self, id: Int, vector: Pointer[Int8, MutUntrackedOrigin], max_level: Int, M: Int, neighbor_block: Pointer[UInt32, MutUntrackedOrigin]):
        self.id = id
        self.vector = vector
        self.max_level = max_level
        self.M = M
        self.inline_vector = SIMD[DType.int8, 16](0)
        # Level 0 has 2*M capacity, others have M capacity.
        var total_capacity = (2 * M) + (6 * M) # Max level 6: supports M^6=16.7M nodes; saves ~34MB at N=50K
        self.neighbors = neighbor_block
        self.neighbor_counts = neighbor_block.unsafe_offset(total_capacity)
        for i in range(max_level + 1):
            self.neighbor_counts[unsafe_offset=i] = 0

    def __moveinit__(out self, deinit take: Self):
        self.id = take.id
        self.vector = take.vector
        self.neighbors = take.neighbors
        self.neighbor_counts = take.neighbor_counts
        self.max_level = take.max_level
        self.M = take.M
        self.inline_vector = take.inline_vector
        # Consume fields to prevent deinit from freeing
        _ = take.neighbors
        _ = take.neighbor_counts

    def _get_offset(self, level: Int) -> Int:
        if level == 0: return 0
        return (2 * self.M) + (level - 1) * self.M

    def add_neighbor(self, level: Int, neighbor_idx: Int):
        if level <= self.max_level:
            var count = self.neighbor_counts[unsafe_offset=level]
            var capacity = 2 * self.M if level == 0 else self.M
            if count < UInt32(capacity):
                var offset = self._get_offset(level)
                self.neighbors[unsafe_offset=offset + Int(count.cast[DType.int64]().cast[DType.int32]())] = UInt32(neighbor_idx)
                self.neighbor_counts[unsafe_offset=level] = count + 1

    def get_neighbor(self, level: Int, index: Int) -> Int:
        var offset = self._get_offset(level)
        return Int(self.neighbors[unsafe_offset=offset + index])

    def get_neighbor_count(self, level: Int) -> Int:
        if level <= self.max_level:
            return Int(self.neighbor_counts[unsafe_offset=level])
        return 0

    def deinit(owned self):
        pass # Neighbors are part of a global pool now

# V2.6: shared read-only view of the built HNSW index, written by the owning worker
# after FT.OPTIMIZE and read by all other workers on first FT.SEARCH.
# All pointers are read-only after ready=True; visited_map stays per-worker.
struct SharedHNSWView(Movable):
    # Published index (ready=True after FT.OPTIMIZE)
    var nodes: Pointer[HNSWNode, MutUntrackedOrigin]
    var node_map: Pointer[Int, MutUntrackedOrigin]
    var neighbor_pool: Pointer[UInt32, MutUntrackedOrigin]
    var neighbor_pool_per_node: Int
    var compact_buffer: Pointer[Int8, MutUntrackedOrigin]
    var compact_is_int4: Bool
    var node_norms: Pointer[Float32, MutUntrackedOrigin]
    var node_prefix_norms: Pointer[Float32, MutUntrackedOrigin]
    var num_nodes: Int
    var deleted_bitset: Pointer[UInt8, MutUntrackedOrigin]
    var deleted_count: Int
    var entry_point_id: Int
    var max_level: Int
    var M: Int
    var dim: Int
    var global_min: Float32
    var global_max: Float32
    var ef_runtime: Int
    var index_name: Array[UInt8, 64]
    var index_name_len: Int
    var ready: Bool
    # Atomic version of `ready` — separately allocated UInt64 read with ACQUIRE
    # ordering and written with RELEASE ordering. The plain Bool above is kept
    # for diagnostics; the source of truth on the FT.SEARCH borrow path is
    # `ready_atomic`. Reason: publish_to_shared writes many fields then sets
    # ready=True; on weak memory order (ARM) other workers can observe
    # ready=True before observing the field writes, then borrow stale pointers
    # — symptom is recall≈0 after a flaky FT.OPTIMIZE handoff.
    var ready_atomic: Pointer[UInt64, MutUntrackedOrigin]
    # gh #14 — PHASE-2 EPOCH RCU. Phase 1 was `usleep(10000)` before freeing
    # the published buffers: correct in practice, unbounded in principle. A
    # search that had already passed its ACQUIRE-load of `ready_atomic` when
    # DROPINDEX fired still holds pointers into memory that is about to be
    # freed, and 10 ms is a guess about how long that search takes, not a fact.
    #
    # `reclaim_epoch` is a monotonically increasing counter bumped by the
    # reclaimer. `worker_epoch[w * 8]` is 0 when worker w is between dispatch
    # batches and `(epoch << 1) | 1` while it is inside one. The reclaimer bumps
    # the epoch, then waits until every worker is either idle or running at an
    # epoch at least as new — at which point no worker can still be holding a
    # pointer it borrowed before the bump.
    #
    # 8 slots per worker (64 B) rather than 1: adjacent workers sharing a cache
    # line would turn a per-batch relaxed store into cross-core invalidation
    # traffic on the hot dispatch path.
    var reclaim_epoch: Pointer[UInt64, MutUntrackedOrigin]
    var worker_epoch: Pointer[UInt64, MutUntrackedOrigin]
    var worker_epoch_slots: Int
    # Pre-build config: set by FT.CREATE, read by any worker during HSET routing
    var pre_index_ready: Bool
    var pre_vector_field_name: Array[UInt8, 32]
    var pre_vector_field_len: Int
    var pre_dim: Int
    # Shared FP32 ingest buffer: written by any worker's HSET, consumed by FT.OPTIMIZE
    var ingest_fp32: Pointer[Float32, MutUntrackedOrigin]
    var ingest_ids: Pointer[Int32, MutUntrackedOrigin]
    var ingest_count: Pointer[UInt64, MutUntrackedOrigin]  # separately allocated; pointer VALUE preserved in struct copies
    var ingest_capacity_warned: Bool                            # one-time warning latch
    # Sharding coordination (set up by main() before parallelize)
    var num_shards: Int
    var shard_bus: Pointer[ShardQueryBus, MutUntrackedOrigin]
    var optimize_trigger: Pointer[UInt64, MutUntrackedOrigin]  # pointer to alloc'd UInt64; pointer value preserved in struct copies
    var shard_ready: Pointer[UInt64, MutUntrackedOrigin]  # [num_shards] 0=building 1=ready; atomic RELEASE/ACQUIRE ops
    var build_requested: Pointer[UInt64, MutUntrackedOrigin] # [num_shards] flag to explicitly request shard build
    var dbg_counters: Pointer[UInt64, MutUntrackedOrigin]  # [num_shards*4]: per-worker [poll_count, saw_trigger, build_done, build_failed]; plain writes (each worker owns its own slots)
    # Random projection metadata: published by coordinator, borrowed by all workers
    var proj_dim: Int                                             # projected dim (0 = disabled)
    var original_dim: Int                                        # original FP32 input dim
    var proj_matrix: Pointer[Int8, MutUntrackedOrigin]     # [proj_dim × original_dim] ±1, owned by coordinator
    # gh #87.1: ivf_pq / ivf_ready removed alongside src/vector/ivf_pq.mojo.
    # DiskANN-style V32: PQ-compressed beam traversal; pointers owned by coordinator, borrowed by others.
    var pq_codes: Pointer[UInt8, MutUntrackedOrigin]     # [num_nodes × 64] UInt8
    var pq_codebook: Pointer[Float32, MutUntrackedOrigin] # [64 × 256 × 24] FP32
    var pq_is_built: Bool
    # V33: compact level-0 neighbor list — 33 UInt32/node = 6.6MB vs 61MB neighbor_pool
    # [count, id0, id1, ..., id31] per node, indexed by internal node ID after compact_vectors().
    var l0_compact: Pointer[UInt32, MutUntrackedOrigin]
    # V34: prefix_buffer — 256-byte prefix of each node; 12.8MB, fits SLC better than 77MB compact_buffer.
    var prefix_buffer: Pointer[Int8, MutUntrackedOrigin]
    # M7 TurboQuant: shared 3-bit compact + QJL buffers
    var compact_is_3bit: Bool
    var qjl_buffer: Pointer[UInt64, MutUntrackedOrigin]
    var qjl_res_norms: Pointer[Float32, MutUntrackedOrigin]
    var qjl_random_signs: Pointer[UInt64, MutUntrackedOrigin]
    # Phase 3.1: Index schema — up to 16 non-vector fields, stored inline (zero heap).
    # Written at FT.CREATE time, read-only by all workers during FT.SEARCH filtering.
    # Field types: 0=NONE, 1=TEXT, 2=TAG, 3=NUMERIC
    var schema_field_count: Int
    var schema_field_types: Array[UInt8, 16]
    var schema_field_name_lens: Array[UInt8, 16]
    var schema_field_names: Array[Array[UInt8, 32], 16]   # 512 bytes inline
    # Per-group Q8_K calibration: shared across all workers
    var group_qmins: Array[Float32, 48]     # max 48 groups (1536/32)
    var group_scales: Array[Float32, 48]
    var grouped_calibrated: Bool
    # Cross-worker slot → original-hash-key mapping. Each slot is 32 bytes:
    # byte 0 = key length (0 = empty), bytes 1..31 = key data. Sized [max_elements * 32]
    # at startup and shared by all workers — fixes multi-worker recall=0 when HSETs
    # land on one worker (storing __hk__<slot> in its local keyspace) but FT.SEARCH
    # runs on another worker and the local lookup misses.
    var hk_keys_buf: Pointer[UInt8, MutUntrackedOrigin]
    var hk_max_elements: Int
    # Cross-worker V-store session directory (forward declared). Pointed-at type
    # is `VStoreDirectory` from `src.network.v_store`; kept as opaque pointer
    # here to avoid an import cycle. Used by KV.PREFIX.LOOKUP / REGISTER /
    # INFO so they answer correctly across workers under `--kvcache -w N`.
    # NULL when --kvcache is off.
    var vstore_directory: Pointer[NoneType, MutUntrackedOrigin]
    # K1 beam kernel: parallel slot array —
    # l0_slots[ni*33 + 1 + k] = compact-buffer SLOT of l0_compact's k-th
    # neighbor. Lets the beam gather loop compute a neighbor's vector address
    # arithmetically (compact_buffer + slot*compact_stride + compact_hdr)
    # instead of the dependent random load `nodes[nid].vector` — one
    # serialized cache miss per gathered neighbor, ~1000×/query.
    # New fields go at the END of the struct (gh #149 layout discipline).
    var l0_slots: Pointer[UInt32, MutUntrackedOrigin]
    var compact_stride: Int   # bytes per compact slot (INT8: dim+8; polar: 868; ...)
    var compact_hdr: Int      # byte offset of vector data within a slot (INT8: 8)
    # gh #271: FT.CREATE DISTANCE_METRIC, published so ANY worker's HSET routes
    # through the same normalization the querying worker will apply.
    # 0 = L2 (default, and what every pre-#271 index is), 1 = COSINE.
    var pre_distance_metric: UInt8
    # gh #346: NanoQuant's compact-format flag and the quant variants' FP32
    # re-rank buffer ([num_nodes × dim], node-indexed, owned by the publisher).
    # Without them a borrowing worker ran the INT8 beam over INT2 bytes and
    # skipped the re-rank.
    var compact_is_2bit: Bool
    var gpu_rerank_fp32: Pointer[Float32, MutUntrackedOrigin]
    # gh #407: FT.CREATE's EF_CONSTRUCTION (0 = none given). With pre_dim,
    # pre_vector_field_* and pre_distance_metric this is the index config every
    # worker can see; FT.OPTIMIZE adopts it, because the builder is usually NOT
    # the worker that handled FT.CREATE and used to build with its defaults.
    var pre_ef_construction: Int

    def __init__(out self):
        self.nodes = null_ptr[HNSWNode, MutUntrackedOrigin]()
        self.node_map = null_ptr[Int, MutUntrackedOrigin]()
        self.neighbor_pool = null_ptr[UInt32, MutUntrackedOrigin]()
        self.neighbor_pool_per_node = 0
        self.compact_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        self.compact_is_int4 = False
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.num_nodes = 0
        self.deleted_bitset = null_ptr[UInt8, MutUntrackedOrigin]()
        self.deleted_count = 0
        self.entry_point_id = -1
        self.max_level = -1
        self.M = 16
        self.dim = 0
        self.global_min = -0.20
        self.global_max = 0.20
        self.ef_runtime = 200
        self.index_name = Array[UInt8, 64](uninitialized=True)
        self.index_name_len = 0
        self.ready = False
        self.ready_atomic = null_ptr[UInt64, MutUntrackedOrigin]()
        self.reclaim_epoch = null_ptr[UInt64, MutUntrackedOrigin]()
        self.worker_epoch = null_ptr[UInt64, MutUntrackedOrigin]()
        self.worker_epoch_slots = 0
        self.pre_index_ready = False
        self.pre_vector_field_name = Array[UInt8, 32](uninitialized=True)
        self.pre_vector_field_len = 0
        self.pre_dim = 0
        self.pre_distance_metric = 0
        self.compact_is_2bit = False
        self.gpu_rerank_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
        self.pre_ef_construction = 0
        self.ingest_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
        self.ingest_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.ingest_count = null_ptr[UInt64, MutUntrackedOrigin]()
        self.ingest_capacity_warned = False
        self.num_shards = 0
        self.shard_bus = null_ptr[ShardQueryBus, MutUntrackedOrigin]()
        self.optimize_trigger = null_ptr[UInt64, MutUntrackedOrigin]()
        self.shard_ready = null_ptr[UInt64, MutUntrackedOrigin]()
        self.build_requested = null_ptr[UInt64, MutUntrackedOrigin]()
        self.dbg_counters = null_ptr[UInt64, MutUntrackedOrigin]()
        self.proj_dim = 0
        self.original_dim = 0
        self.proj_matrix = null_ptr[Int8, MutUntrackedOrigin]()
        self.pq_codes = null_ptr[UInt8, MutUntrackedOrigin]()
        self.pq_codebook = null_ptr[Float32, MutUntrackedOrigin]()
        self.pq_is_built = False
        self.l0_compact = null_ptr[UInt32, MutUntrackedOrigin]()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        self.compact_is_3bit = False
        self.qjl_buffer = null_ptr[UInt64, MutUntrackedOrigin]()
        self.qjl_res_norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.qjl_random_signs = null_ptr[UInt64, MutUntrackedOrigin]()
        self.schema_field_count = 0
        self.schema_field_types = Array[UInt8, 16](fill=UInt8(0))
        self.schema_field_name_lens = Array[UInt8, 16](fill=UInt8(0))
        self.schema_field_names = Array[Array[UInt8, 32], 16](uninitialized=True)
        for _si in range(16):
            for _bi in range(32): self.schema_field_names[_si][_bi] = 0
        self.grouped_calibrated = False
        self.group_qmins = Array[Float32, 48](fill=Float32(-0.20))
        self.group_scales = Array[Float32, 48](fill=Float32(254.0 / 0.40))
        self.hk_keys_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.hk_max_elements = 0
        self.vstore_directory = null_ptr[NoneType, MutUntrackedOrigin]()
        self.l0_slots = null_ptr[UInt32, MutUntrackedOrigin]()
        self.compact_stride = 0
        self.compact_hdr = 0

    def __moveinit__(out self, deinit take: Self):
        self.nodes = take.nodes
        self.node_map = take.node_map
        self.neighbor_pool = take.neighbor_pool
        self.neighbor_pool_per_node = take.neighbor_pool_per_node
        self.compact_buffer = take.compact_buffer
        self.compact_is_int4 = take.compact_is_int4
        self.node_norms = take.node_norms
        self.node_prefix_norms = take.node_prefix_norms
        self.num_nodes = take.num_nodes
        self.deleted_bitset = take.deleted_bitset
        self.deleted_count = take.deleted_count
        self.entry_point_id = take.entry_point_id
        self.max_level = take.max_level
        self.M = take.M
        self.dim = take.dim
        self.global_min = take.global_min
        self.global_max = take.global_max
        self.ef_runtime = take.ef_runtime
        self.index_name = Array[UInt8, 64](uninitialized=True)
        for ii in range(64):
            self.index_name[ii] = take.index_name[ii]
        self.index_name_len = take.index_name_len
        self.ready = take.ready
        self.ready_atomic = take.ready_atomic  # pointer value copy — same alloc
        self.reclaim_epoch = take.reclaim_epoch      # gh #14: pointer value copy
        self.worker_epoch = take.worker_epoch        # gh #14: pointer value copy
        self.worker_epoch_slots = take.worker_epoch_slots
        self.pre_index_ready = take.pre_index_ready
        self.pre_vector_field_name = Array[UInt8, 32](uninitialized=True)
        for ii in range(32):
            self.pre_vector_field_name[ii] = take.pre_vector_field_name[ii]
        self.pre_vector_field_len = take.pre_vector_field_len
        self.pre_dim = take.pre_dim
        self.ingest_fp32 = take.ingest_fp32
        self.ingest_ids = take.ingest_ids
        self.ingest_count = take.ingest_count  # pointer value copy — both point to same alloc'd UInt64
        self.ingest_capacity_warned = take.ingest_capacity_warned
        self.num_shards = take.num_shards
        self.shard_bus = take.shard_bus
        self.optimize_trigger = take.optimize_trigger
        self.shard_ready = take.shard_ready
        self.build_requested = take.build_requested
        self.dbg_counters = take.dbg_counters
        self.proj_dim = take.proj_dim
        self.original_dim = take.original_dim
        self.proj_matrix = take.proj_matrix
        self.pq_codes = take.pq_codes
        self.pq_codebook = take.pq_codebook
        self.pq_is_built = take.pq_is_built
        self.l0_compact = take.l0_compact
        self.prefix_buffer = take.prefix_buffer
        self.compact_is_3bit = take.compact_is_3bit
        self.qjl_buffer = take.qjl_buffer
        self.qjl_res_norms = take.qjl_res_norms
        self.qjl_random_signs = take.qjl_random_signs
        self.schema_field_count = take.schema_field_count
        self.schema_field_types = Array[UInt8, 16](uninitialized=True)
        for _bi in range(16): self.schema_field_types[_bi] = take.schema_field_types[_bi]
        self.schema_field_name_lens = Array[UInt8, 16](uninitialized=True)
        for _bi in range(16): self.schema_field_name_lens[_bi] = take.schema_field_name_lens[_bi]
        self.schema_field_names = Array[Array[UInt8, 32], 16](uninitialized=True)
        for _si in range(16):
            for _bi in range(32): self.schema_field_names[_si][_bi] = take.schema_field_names[_si][_bi]
        self.grouped_calibrated = take.grouped_calibrated
        self.group_qmins = Array[Float32, 48](uninitialized=True)
        self.group_scales = Array[Float32, 48](uninitialized=True)
        for _gi in range(48):
            self.group_qmins[_gi] = take.group_qmins[_gi]
            self.group_scales[_gi] = take.group_scales[_gi]
        self.hk_keys_buf = take.hk_keys_buf
        self.hk_max_elements = take.hk_max_elements
        self.vstore_directory = take.vstore_directory
        self.l0_slots = take.l0_slots
        self.compact_stride = take.compact_stride
        self.compact_hdr = take.compact_hdr
        self.pre_distance_metric = take.pre_distance_metric
        self.compact_is_2bit = take.compact_is_2bit
        self.gpu_rerank_fp32 = take.gpu_rerank_fp32
        self.pre_ef_construction = take.pre_ef_construction

    def add_ingest_vector(mut self, id: Int, vector: Pointer[Float32, MutUntrackedOrigin]) -> Int:
        """Buffer a FP32 vector during load phase. Called from any worker's HSET handler.
        Returns the slot index (used as the HNSW node ID for __hk__ key mapping),
        or -1 if rejected (buffer not initialised OR ingest_fp32 capacity exhausted)."""
        var dim = self.pre_dim
        if dim <= 0 or is_null(self.ingest_fp32): return -1

        var slot = Int(Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](
            self.ingest_count, UInt64(1)
        ))
        # Bounds check: ingest_fp32 was allocated for hk_max_elements slots
        # (see main.mojo §549-553). Without this check, slot >= max_elements
        # writes past the end of the buffer → memory corruption → SIGSEGV
        # (manifested as Pion process crash during HSET multi-field at scale).
        if self.hk_max_elements > 0 and slot >= self.hk_max_elements:
            # We don't roll back ingest_count — once exhausted, every
            # subsequent call also returns -1, so the counter just stops
            # being meaningful. Clean rollback would require a CAS loop
            # since multiple workers race here. Counter overshoot is
            # harmless because nothing reads it for storage sizing.
            #
            # One-time warning so the user knows to bump --max-elements.
            if not self.ingest_capacity_warned:
                print("WARN: HSET vector ingest dropped — buffer full at "
                      + String(self.hk_max_elements)
                      + " vectors. Restart with --max-elements <N> to raise the cap.")
                self.ingest_capacity_warned = True
            return -1
        var off = slot * dim
        for i in range(dim):
            self.ingest_fp32[unsafe_offset=off + i] = vector[unsafe_offset=i]
        # gh #271: under COSINE the index stores unit vectors. FT.OPTIMIZE both
        # calibrates the quantizer and builds the graph straight out of this
        # buffer, so normalizing at the moment the vector lands here is the one
        # place that covers every consumer. Scalar and in place, matching the
        # copy loop above — this is the ingest path, not a query path, and a
        # zero vector has no direction so it is left alone rather than made NaN.
        if self.pre_distance_metric == 1:
            var norm_sq = Float32(0.0)
            for i in range(dim):
                var v = self.ingest_fp32[unsafe_offset=off + i]
                norm_sq += v * v
            if norm_sq > 0.0:
                var inv = Float32(1.0) / sqrt(norm_sq)
                for i in range(dim):
                    self.ingest_fp32[unsafe_offset=off + i] = self.ingest_fp32[unsafe_offset=off + i] * inv
        self.ingest_ids[unsafe_offset=slot] = Int32(slot)  # Use slot as node ID (not key-parsed digits)
        return slot
