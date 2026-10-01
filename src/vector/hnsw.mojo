from src.common.ptr import is_not_null, is_null, null_ptr
from src.memory.slab_allocator import SlabAllocator
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from src.common.heap import HeapNode, MinHeap, MaxHeap, LinearPool
from std.atomic import Atomic, Ordering
from src.common.lock_free import ShardQueryBus
from .hnsw_types import HNSWNode, SharedHNSWView, HNSW_SHARDED_INGEST_ENABLED
from .beam_view import BeamView1536
from .vector_abi import beam_search_1536, quant_beam_search_1536
from .quant_beam_view import QuantBeamView1536, QUANT_KIND_POLAR, QUANT_KIND_NANO, QUANT_KIND_TURBO
# gh #87.1: ivf_pq module deleted.
from .gpu_search import GPUSearchContext, GPU_RERANK_THRESHOLD, GPU_RERANK_K_MAX
from .kernels import (
    quantize_fp32_to_int8_simd,
    quantize_fp32_to_int4_simd,
    welford_calibrate,
    l2_distance_int8,
    l2_distance_fp32_int8,
    l2_distance_int8_jit,
    l2_distance_fp32_int8_fused_jit,
    l2_distance_int8_int8_batch8_jit,
    l2_distance_int8_int8_batch4_jit,
    l2_distance_int8_prefix_suffix_fused_batch8_jit,
    l2_distance_int8_prefix_suffix_fused_batch4_jit, 
    cosine_distance_int8_jit,
    cosine_distance_int8_int8_batch8_jit,
    cosine_distance_int8_int8_batch4_jit,
    l2_distance_fp32_int4,
    l2_distance_fp32_int4_jit,
    l2_distance_int4,
    l2_distance_int4_jit,
    l2_distance_int4_int4_batch8_jit,
    l2_distance_int4_int4_batch4_jit,
    l2_distance_int4_suffix_early_exit_jit,
    norm_sq_int4_jit,
    hamming_distance,
    hamming_distance_jit,  
    norm_sq_int8_jit,
    l2_normalize_fp32,
    wht_fp32_1536,
    quantize_fp32_to_block_int4,
    quantize_fp32_to_block_int8,
    block_int4_dot_single_simd, 
    VEC_BYTES_1536,
    NUM_BLOCKS_1536,
    INT3_BLOCK_BYTES,
    INT3_VEC_BYTES_1536,
    QJL_U64S_1536,
    QJL_BYTES_1536,
    BLOCK_DIM,
    quantize_fp32_to_block_int3,
    int3_dot_single_simd,
    _byte_u64, 
    qjl_compute_signs, 
    INT2_BLOCK_BYTES,
    INT2_VEC_BYTES_1536,
    quantize_fp32_to_block_int2,
    int2_dot_single_simd, 
    calibrate_per_group,
    l2_int8_sabd_udot,
    l2_int8_sabd_udot_batch8,
)
from std.random import random_float64
from std.math import log, min, fma
from std.collections import List, Array
from std.sys.intrinsics import prefetch
from std.ffi import external_call
from std.sys.info import CompilationTarget

# gh #376: vectors the HSET-ingest build calibrates on. All of them for the
# 50K gate corpus; a bounded prefix past that, since the per-group pass is a
# scalar Welford over count*dim values and a 5M build would spend ~20 s in it.
comptime CALIBRATION_SAMPLE_MAX = 65536




struct HNSWGraph(Movable):
    var max_elements: Int
    var dim: Int
    var nodes: UnsafePointer[HNSWNode, MutUntrackedOrigin]
    var num_nodes: Int
    var node_map: UnsafePointer[Int, MutUntrackedOrigin]
    var neighbor_pool: UnsafePointer[UInt32, MutUntrackedOrigin]
    var neighbor_pool_per_node: Int
    var vector_allocator: SlabAllocator[Int8]
    var node_allocator: SlabAllocator[HNSWNode]
    var entry_point_id: Int
    var max_level: Int
    var M: Int
    var ml: Float32
    var ef_construction: Int
    var global_min: Float32
    var global_max: Float32
    # Visit Filter Trick (gh #118: epoch-stamped visited set — no per-query memset).
    # A node is "visited this query" iff visited_epoch[nid] == cur_epoch; _reset_visited
    # just bumps cur_epoch (memset only every 65535 queries on UInt16 wrap).
    var visited_epoch: UnsafePointer[UInt16, MutUntrackedOrigin]
    var cur_epoch: UInt16
    var visited_bitset_bytes: Int  # still sizes deleted_bitset
    var visited_map: UnsafePointer[UInt32, MutUntrackedOrigin] # Kept for struct compatibility
    var cur_num: UInt32
    # A1b: Tombstone deletion support
    var deleted_bitset: UnsafePointer[UInt8, MutUntrackedOrigin]
    var deleted_count: Int
    var compact_dirty: Bool
    var use_int4: Bool
    var use_bq: Bool
    var has_gpu: Bool
    # FP32 rerank: True after pion_metal_register_rerank_buffer succeeds for the
    # current gpu_rerank_fp32 pointer. Reset on every compact (pointer changes).
    var gpu_rerank_registered: Bool
    var is_borrowed: Bool # Phase 1.1: Track if we own the memory or borrowed it
    var vector_field_name: Array[UInt8, 32]
    var vector_field_len: Int
    var index_ready: Bool
    # V2.2B: FP32 staging buffer for two-phase calibrated ingest
    var fp32_buffer: UnsafePointer[Float32, MutUntrackedOrigin]
    var fp32_ids: UnsafePointer[Int32, MutUntrackedOrigin]
    var fp32_count: Int
    var streaming_mode: Bool  # When True, add_vector() inserts directly (no fp32_buffer)
    var streaming_calibrated: Bool  # True after first 1000 vectors calibrate global_min/max
    # V2.3: BFS-compacted contiguous vector buffer (set by compact_vectors)
    var compact_buffer: UnsafePointer[Int8, MutUntrackedOrigin]
    var compact_is_int4: Bool
    var node_norms: UnsafePointer[Float32, MutUntrackedOrigin]         # [num_nodes] full 1536-dim INT8 norm_sq, indexed by node idx
    var node_prefix_norms: UnsafePointer[Float32, MutUntrackedOrigin]  # [num_nodes] first-256-dim INT8 norm_sq, indexed by node idx
    var ef_runtime: Int
    var index_name: Array[UInt8, 64]
    var index_name_len: Int
    # V8.1: INT8 query scratch buffer — quantized once per search, used for INT8-INT8 batch kernels
    var query_int8: UnsafePointer[Int8, MutUntrackedOrigin]
    # Instrumentation counters — nodes evaluated and queries run (for diagnosing prefix pruning impact)
    var stat_query_count: Int
    var stat_total_evals: Int
    # gh #199: per-node spinlocks for the parallel FT.OPTIMIZE link phase.
    # 0 = free, 1 = held. Only the MT build path touches them; runtime searches
    # never lock (post-build reads are ordered by the pool's pthread_join).
    var node_locks: UnsafePointer[UInt32, MutUntrackedOrigin]
    # Phase 3.2: Pre-allocated search heaps and lists to eliminate per-query allocation
    var candidates: MinHeap
    var results: MaxHeap
    var reranked: MaxHeap
    var scratch_ids: List[Int]
    var scratch_dists: List[Float32]
    var final_results: List[Int]
    var rerank_ids: List[Int]
    # Random projection: 1536 → proj_dim to shrink compact_buffer from 77MB to ~12.8MB (fits M4 SLC)
    var proj_matrix: UnsafePointer[Int8, MutUntrackedOrigin]    # [proj_dim × original_dim] ±1 entries
    var proj_dim: Int                                            # projected dim (0 = disabled)
    var original_dim: Int                                        # original FP32 input dim before projection
    var proj_scratch: UnsafePointer[Float32, MutUntrackedOrigin] # [proj_dim] temp buffer per vector/query
    # V33: compact level-0 neighbor list — 33 UInt32/node = 6.6MB vs 61MB neighbor_pool
    # [count, id0, ..., id31] per node; built by compact_vectors(), shared read-only by borrowers.
    var l0_compact: UnsafePointer[UInt32, MutUntrackedOrigin]
    # V34: prefix_buffer — separate 12.8MB buffer holding only first 256 INT8 dims per node.
    # Fits in M4 SLC (12MB) better than 77MB compact_buffer → fewer DRAM misses in prune phase.
    # Layout: prefix_buffer[nidx × 256 .. nidx × 256 + 255] = first 256 dims of node nidx.
    var prefix_buffer: UnsafePointer[Int8, MutUntrackedOrigin]
    # §7: GPU brute-force search support
    # compact_buffer slot i → gpu_slot_ext_ids[i] = external ID of the vector in that slot.
    # Required because compact_buffer is BFS-ordered, not insertion-ordered.
    var gpu_slot_ext_ids: UnsafePointer[Int32, MutUntrackedOrigin]
    # Pre-allocated scratch buffers for GPU search (avoid per-query alloc)
    var gpu_distances: UnsafePointer[Float32, MutUntrackedOrigin]
    var gpu_topk_ids: UnsafePointer[Int32, MutUntrackedOrigin]
    var gpu_topk_dists: UnsafePointer[Float32, MutUntrackedOrigin]
    # §7v2: FP32 re-rank buffer — BFS-ordered FP32 vectors for GPU oversample re-ranking
    # Same slot ordering as compact_buffer. Built during compact_vectors() from fp32_buffer.
    var gpu_rerank_fp32: UnsafePointer[Float32, MutUntrackedOrigin]
    # §7v2: Native Mojo GPU context — zero ObjC overhead, direct GPU dispatch
    var gpu_ctx: GPUSearchContext
    # Phase 3.1: per-node metadata for O(1) filter checks (populated after FT.OPTIMIZE by slow_path)
    # Indexed by [nidx * 8 + schema_slot]; up to 8 filterable schema fields per node
    var node_meta_tag_hashes: UnsafePointer[UInt32, MutUntrackedOrigin]  # FNV-1a hash of TAG value
    var node_meta_numerics:   UnsafePointer[Float32, MutUntrackedOrigin] # Float32 for NUMERIC field
    var node_meta_field_set:  UnsafePointer[UInt8, MutUntrackedOrigin]   # 1=tag set, 3=numeric set
    # Phase 3.2: BM25 inverted index — built from TEXT schema fields after FT.OPTIMIZE
    # term_hashes[] sorted ascending for O(log V) binary search at query time.
    var bm25_term_hashes:    UnsafePointer[UInt32, MutUntrackedOrigin]   # sorted unique FNV-1a hashes [bm25_vocab_count]
    var bm25_term_idf:       UnsafePointer[Float32, MutUntrackedOrigin]  # cached IDF per term [bm25_vocab_count]
    var bm25_postings_start: UnsafePointer[UInt32, MutUntrackedOrigin]   # per-term start in postings_buf [bm25_vocab_count]
    var bm25_postings_count: UnsafePointer[UInt32, MutUntrackedOrigin]   # per-term pair count [bm25_vocab_count]
    var bm25_postings_buf:   UnsafePointer[UInt32, MutUntrackedOrigin]   # packed [ext_id, tf, ...] all terms
    var bm25_doc_lengths:    UnsafePointer[UInt16, MutUntrackedOrigin]   # token count per ext_id [max_elements]
    var bm25_scores_scratch: UnsafePointer[Float32, MutUntrackedOrigin]  # [max_elements] per-search reuse (always owned)
    var bm25_vocab_count:    Int
    var bm25_total_tokens:   Int
    var bm25_is_built:       Bool
    # gh #139: BM25's document set is NOT the HNSW node set. FT.ADDTEXT registers
    # text-only docs (no vector, hence no node) here so build_bm25 can index them;
    # bm25_doc_ids is the union it actually indexed, and what search_bm25 scans.
    var bm25_text_ids:       UnsafePointer[Int32, MutUntrackedOrigin]    # FT.ADDTEXT-registered ext_ids [bm25_text_cap]
    var bm25_text_count:     Int
    var bm25_text_cap:       Int
    var bm25_doc_ids:        UnsafePointer[Int32, MutUntrackedOrigin]    # docs in the built index [bm25_doc_count]
    var bm25_doc_count:      Int
    # gh #145: which index these postings were built for. A server holds ONE
    # index, so FT.OPTIMIZE on index B replaces the postings that were serving
    # index A; recording the owner lets a query on A say so instead of silently
    # answering `*0`. Captured at build time rather than at FT.CREATE time — the
    # postings belong to whichever schema actually produced them.
    var bm25_index_name:     Array[UInt8, 64]
    var bm25_index_name_len: Int
    # M6 PolarQuant: WHT rotation + INT4 quantization
    var polarquant: Bool                                       # enable block-wise INT4 in compact_vectors/search
    var query_int4: UnsafePointer[Int8, MutUntrackedOrigin]     # INT4 query scratch (dim/2 bytes)
    var wht_scratch: UnsafePointer[Float32, MutUntrackedOrigin] # per-query FP32 scratch for WHT rotation
    var wht_global_min: Float32                                # post-WHT calibration min
    var wht_global_max: Float32                                # post-WHT calibration max
    var query_block_int8: UnsafePointer[Int8, MutUntrackedOrigin]     # M6b: block-INT8 query (dim bytes)
    var query_block_scales: UnsafePointer[Float32, MutUntrackedOrigin] # M6b: per-block FP32 scales (num_blocks × 4B)
    var query_block_norm: Float32                                     # M6b: reconstructed query norm squared
    # M7 TurboQuant: 3-bit block quantization + QJL error correction
    var turboquant: Bool                                               # enable block-3bit + QJL in compact/search
    var compact_is_3bit: Bool                                          # compact_buffer format flag
    var qjl_buffer: UnsafePointer[UInt64, MutUntrackedOrigin]          # [num_nodes × 24] QJL sign bits
    var qjl_res_norms: UnsafePointer[Float32, MutUntrackedOrigin]      # [num_nodes] residual norm squared
    var qjl_random_signs: UnsafePointer[UInt64, MutUntrackedOrigin]    # [24] fixed random ±1 bits (seeded)
    var qjl_query_signs: UnsafePointer[UInt64, MutUntrackedOrigin]     # [24] per-query QJL signs
    var qjl_query_res_norm: Float32                                    # per-query residual norm squared
    var qjl_lambda: Float32                                            # QJL correction weight
    # N4 NanoQuant
    var nanoquant: Bool                                                # N4: NanoQuant 2-bit enabled
    var compact_is_2bit: Bool                                          # compact buffer uses INT2 format
    # Per-group calibration (Q8_K-style): each 32-dim group gets independent min/scale.
    # Stored once per index (not per vector). Improves quantization precision at large scale.
    var group_qmins: UnsafePointer[Float32, MutUntrackedOrigin]         # [num_groups] per-group qmin
    var group_scales: UnsafePointer[Float32, MutUntrackedOrigin]        # [num_groups] per-group scale = 254/(qmax-qmin)
    var num_groups: Int                                                 # dim // 32 (0 = grouped quant disabled)
    var grouped_calibrated: Bool                                       # True after grouped calibration completes
    # K1 beam kernel: l0_slots mirrors
    # l0_compact's shape (33 UInt32/node) but holds compact-buffer SLOTS, so
    # the beam gather loop computes each neighbor's vector address as
    # `compact_buffer + slot*compact_stride + compact_hdr` — pure arithmetic —
    # instead of the dependent random load `nodes[nid].vector` (one serialized
    # cache miss per gathered neighbor, ~1000×/query = the bulk of the 62%
    # `_beam_search_1536` self-time measured 2026-08-05).
    # SLOT_NONE entries fall back to the nodes[] dereference.
    # New fields at the END of the struct (gh #149 layout discipline).
    var l0_slots: UnsafePointer[UInt32, MutUntrackedOrigin]
    var compact_stride: Int   # bytes per compact slot (INT8: dim+8 64B-padded; polar: 868; turbo: 676; ...)
    var compact_hdr: Int      # byte offset of vector data within a slot (INT8: 8, quant variants: 0)
    # gh #197: single sorted pool for the 1536 query beam (replaces the
    # candidates/results pair there only — _search_layer builds and the quant
    # beams stay on the heaps). At END of struct per the field-ordering rule.
    var pool: LinearPool
    # gh #212: parallel-build link-loss accounting. The MT link lanes race on
    # it, so it lives in a separately allocated cell updated with relaxed
    # atomics. Counts links the serial algorithm would have kept but the
    # concurrent build lost (own-list full of backlinks AND the pruned pick
    # not closer than the current worst). Reset per build, reported after join.
    var link_drops: UnsafePointer[UInt64, MutUntrackedOrigin]
    # gh #271: FT.CREATE DISTANCE_METRIC. 0 = L2 (what this engine has always
    # computed — squared L2 over affinely-quantized codes), 1 = COSINE (the
    # same search, over vectors L2-normalized at ingest and at query time).
    # At END of struct per the field-ordering rule. Default 0 keeps every
    # existing caller — and every pre-#271 index file — on today's behaviour.
    var distance_metric: UInt8
    # Scratch for COSINE normalization; sized `dim`, allocated lazily on first
    # use so an L2-only server never pays for it. Shared by the query path and
    # the streaming direct-insert path: a worker is single-task (V16), so it is
    # inside exactly one dispatch batch and an ingest never overlaps a query on
    # the same graph. The PARALLEL build lanes (`_insert_to_graph_mt`) do NOT
    # touch it — they read `fp32_buffer`, which was already normalized at
    # staging time, so there is no race to have.
    var fp32_norm_scratch: UnsafePointer[Float32, MutUntrackedOrigin]
    # D11: the 1536 beam's argument block, rewritten every query. Heap, not
    # stack: under Mojo 1.0 -O3, stores into a `stack_allocation` that is then
    # passed to a @no_inline Mojo function were eliminated (the callee read
    # zeros — tests/test_vector_differential.mojo reproduced it). At END of
    # struct per the field-ordering rule.
    var beam_view: UnsafePointer[BeamView1536, MutUntrackedOrigin]
    # D11: same for the three quantized beams (quant_beam_view.mojo).
    var quant_beam_view: UnsafePointer[QuantBeamView1536, MutUntrackedOrigin]
    # Floats fp32_norm_scratch holds. The scratch used to be allocated once, at
    # the dim of the first COSINE vector, and reused forever: after FT.CREATE
    # changed dim (4 -> 384) l2_normalize_fp32 wrote 380 floats past the end.
    # _norm_scratch() grows it. At END of struct per the field-ordering rule.
    var fp32_norm_scratch_cap: Int
    # gh #396: TurboQuant's per-query QJL residual (dim floats). It was an
    # alloc + free on every query. Lazily sized like fp32_norm_scratch; it
    # cannot share that buffer, which may hold the (normalized) query itself.
    # At END of struct per the field-ordering rule.
    var residual_scratch: UnsafePointer[Float32, MutUntrackedOrigin]
    var residual_scratch_cap: Int

    def __init__(out self, max_elements: Int, dim: Int, M: Int = 16, ef_construction: Int = 100, use_int4: Bool = False, use_bq: Bool = False, has_gpu: Bool = False, use_huge_pages: Bool = False, polarquant: Bool = False, turboquant: Bool = False, nanoquant: Bool = False):
        self.max_elements = max_elements
        self.dim = dim
        self.nodes = alloc[HNSWNode](max_elements)
        self.num_nodes = 0
        self.node_map = alloc[Int](max_elements)
        for i in range(max_elements):
            self.node_map[i] = -1

        # Neighbor Pool Allocation
        self.neighbor_pool_per_node = (2 * M) + (6 * M) + 7 # Max level 6: supports M^6=16.7M nodes; saves ~34MB at N=50K
        self.neighbor_pool = alloc[UInt32](max_elements * self.neighbor_pool_per_node)

        # Vector Allocation Strategy
        # PolarQuant/TurboQuant: graph construction always uses INT8 (block quantization applied in compact_vectors)
        var vector_bytes: Int
        if use_bq:
            vector_bytes = ((dim + 63) // 64) * 8
        elif use_int4 and not polarquant and not turboquant:
            vector_bytes = (dim + 1) // 2
        else:
            vector_bytes = dim

        self.vector_allocator = SlabAllocator[Int8](max_elements, vector_bytes, use_huge_pages=use_huge_pages)
        self.node_allocator = SlabAllocator[HNSWNode](max_elements, use_huge_pages=use_huge_pages)
        self.entry_point_id = -1
        self.max_level = -1
        self.M = M
        self.ml = 1.0 / log(Float32(M))
        self.ef_construction = ef_construction
        self.global_min = -0.20  # V2.2C: narrowed for OpenAI unit-norm embeddings (actual range ~[-0.12,0.12])
        self.global_max = 0.20   # consistent calibration for all vectors; better precision than [-1,1]

        # Initialize epoch-stamped visited set (gh #118). Epoch 0 = "never visited";
        # first _reset_visited() bumps cur_epoch to 1, so the zeroed array reads as unvisited.
        self.visited_bitset_bytes = (max_elements + 7) // 8
        self.visited_epoch = alloc[UInt16](max_elements)
        unsafe_memset(self.visited_epoch.bitcast[UInt8](), 0, max_elements * 2)
        self.cur_epoch = 0
        self.visited_map = null_ptr[UInt32, MutUntrackedOrigin]()
        self.cur_num = 1
        # A1b: Tombstone deletion
        self.deleted_bitset = alloc[UInt8](self.visited_bitset_bytes)
        unsafe_memset(self.deleted_bitset, 0, self.visited_bitset_bytes)
        self.deleted_count = 0
        self.compact_dirty = False
        self.use_int4 = use_int4
        self.use_bq = use_bq
        self.has_gpu = has_gpu
        self.gpu_rerank_registered = False
        self.is_borrowed = False  # By default, we own our memory
        self.vector_field_name = Array[UInt8, 32](uninitialized=True)
        # default field name = "vector" (v=118,e=101,c=99,t=116,o=111,r=114)
        self.vector_field_name[0] = 118; self.vector_field_name[1] = 101
        self.vector_field_name[2] = 99;  self.vector_field_name[3] = 116
        self.vector_field_name[4] = 111; self.vector_field_name[5] = 114
        self.vector_field_len = 6
        self.index_ready = False
        self.ef_runtime = 150
        self.index_name = Array[UInt8, 64](uninitialized=True)
        self.index_name_len = 0
        self.fp32_buffer = null_ptr[Float32, MutUntrackedOrigin]()
        self.fp32_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.fp32_count = 0
        self.streaming_mode = max_elements > 1000000  # Auto-enable for large datasets
        self.streaming_calibrated = False
        self.compact_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        self.compact_is_int4 = False
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()
        # Pre-allocate query_int8 scratch buffer (1536 bytes, reused every search call)
        self.query_int8 = alloc[Int8](max(dim, 1536))
        self.stat_query_count = 0
        self.stat_total_evals = 0
        self.node_locks = alloc[UInt32](max_elements)
        unsafe_memset(self.node_locks.bitcast[UInt8](), 0, max_elements * 4)
        self.link_drops = alloc[UInt64](1)
        self.link_drops[0] = 0
        self.is_borrowed = False # Ensure explicitly owned if created via __init__
        self.candidates = MinHeap()
        self.results = MaxHeap()
        self.reranked = MaxHeap()
        self.pool = LinearPool()
        self.scratch_ids = List[Int]()
        self.scratch_dists = List[Float32]()
        self.final_results = List[Int]()
        self.rerank_ids = List[Int]()
        self.proj_matrix = null_ptr[Int8, MutUntrackedOrigin]()
        self.proj_dim = 0
        self.original_dim = 0
        self.proj_scratch = null_ptr[Float32, MutUntrackedOrigin]()
        self.l0_compact = null_ptr[UInt32, MutUntrackedOrigin]()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        self.l0_slots = null_ptr[UInt32, MutUntrackedOrigin]()
        self.compact_stride = 0
        self.compact_hdr = 0
        self.distance_metric = 0        # gh #271: 0 = L2 (default), 1 = COSINE
        self.beam_view = alloc[BeamView1536](1)
        self.quant_beam_view = alloc[QuantBeamView1536](1)
        self.fp32_norm_scratch_cap = 0
        self.fp32_norm_scratch = null_ptr[Float32, MutUntrackedOrigin]()
        self.residual_scratch_cap = 0
        self.residual_scratch = null_ptr[Float32, MutUntrackedOrigin]()
        self.gpu_slot_ext_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.gpu_distances = null_ptr[Float32, MutUntrackedOrigin]()
        self.gpu_topk_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.gpu_topk_dists = null_ptr[Float32, MutUntrackedOrigin]()
        self.gpu_rerank_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
        self.gpu_ctx = GPUSearchContext()
        self.node_meta_tag_hashes = null_ptr[UInt32, MutUntrackedOrigin]()
        self.node_meta_numerics = null_ptr[Float32, MutUntrackedOrigin]()
        self.node_meta_field_set = null_ptr[UInt8, MutUntrackedOrigin]()
        self.bm25_term_hashes    = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_term_idf       = null_ptr[Float32, MutUntrackedOrigin]()
        self.bm25_postings_start = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_postings_count = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_postings_buf   = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_doc_lengths    = null_ptr[UInt16, MutUntrackedOrigin]()
        self.bm25_scores_scratch = alloc[Float32](max_elements)
        self.bm25_vocab_count    = 0
        self.bm25_total_tokens   = 0
        self.bm25_is_built       = False
        self.bm25_text_ids       = null_ptr[Int32, MutUntrackedOrigin]()
        self.bm25_text_count     = 0
        self.bm25_text_cap       = 0
        self.bm25_doc_ids        = null_ptr[Int32, MutUntrackedOrigin]()
        self.bm25_doc_count      = 0
        self.bm25_index_name     = Array[UInt8, 64](uninitialized=True)
        self.bm25_index_name_len = 0
        # M6 PolarQuant
        self.polarquant = polarquant
        self.query_int4 = alloc[Int8](max(dim // 2, 768)) if polarquant else null_ptr[Int8, MutUntrackedOrigin]()
        self.wht_scratch = alloc[Float32](max(dim, 1536)) if polarquant else null_ptr[Float32, MutUntrackedOrigin]()
        self.wht_global_min = Float32(0.0)
        self.wht_global_max = Float32(0.0)
        # M6b: Block-wise INT4 query scratch
        var need_block_query = polarquant or turboquant or nanoquant
        self.query_block_int8 = alloc[Int8](max(dim, 1536)) if need_block_query else null_ptr[Int8, MutUntrackedOrigin]()
        self.query_block_scales = alloc[Float32](max(dim // 32, 48)) if need_block_query else null_ptr[Float32, MutUntrackedOrigin]()
        self.query_block_norm = Float32(0.0)
        # M7 TurboQuant
        self.turboquant = turboquant
        self.compact_is_3bit = False
        self.qjl_buffer = null_ptr[UInt64, MutUntrackedOrigin]()
        self.qjl_res_norms = null_ptr[Float32, MutUntrackedOrigin]()
        self.qjl_query_signs = alloc[UInt64](QJL_U64S_1536) if turboquant else null_ptr[UInt64, MutUntrackedOrigin]()
        self.qjl_query_res_norm = Float32(0.0)
        self.qjl_lambda = Float32(1.0)  # calibrated after first benchmark
        # Generate fixed random signs for QJL (seeded PRNG, same for all vectors)
        if turboquant:
            self.qjl_random_signs = alloc[UInt64](QJL_U64S_1536)
            # Simple LCG seeded at 42 for reproducibility
            var rng_state = UInt64(42)
            for w in range(QJL_U64S_1536):
                var bits: UInt64 = 0
                for b in range(64):
                    rng_state = rng_state * 6364136223846793005 + 1442695040888963407
                    if (rng_state >> UInt64(33)) & UInt64(1):
                        bits |= (UInt64(1) << UInt64(b))
                self.qjl_random_signs[w] = bits
            # Also allocate WHT scratch for QJL (reuse polarquant scratch if available)
            if is_null(self.wht_scratch):
                self.wht_scratch = alloc[Float32](max(dim, 1536))
        else:
            self.qjl_random_signs = null_ptr[UInt64, MutUntrackedOrigin]()
        # N4 NanoQuant
        self.nanoquant = nanoquant
        self.compact_is_2bit = False
        if nanoquant and is_null(self.wht_scratch):
            self.wht_scratch = alloc[Float32](max(dim, 1536))
        # Per-group calibration (Q8_K-style)
        var ng = dim // 32 if dim >= 32 else 0
        self.num_groups = ng
        self.grouped_calibrated = False
        if ng > 0:
            self.group_qmins = alloc[Float32](ng)
            self.group_scales = alloc[Float32](ng)
            for gi in range(ng):
                self.group_qmins[gi] = Float32(-0.20)
                self.group_scales[gi] = Float32(254.0 / 0.40)
        else:
            self.group_qmins = null_ptr[Float32, MutUntrackedOrigin]()
            self.group_scales = null_ptr[Float32, MutUntrackedOrigin]()

    def _init_projection(mut self, proj_dim: Int):
        """Build Achlioptas ±1 random projection matrix original_dim → proj_dim.
        Uses a fixed Xorshift64 seed so every worker produces the identical matrix."""
        self.original_dim = self.dim
        self.proj_dim = proj_dim
        var n = proj_dim * self.dim
        self.proj_matrix = alloc[Int8](n)
        var state: UInt64 = 0x9E3779B97F4A7C15  # deterministic seed
        for i in range(n):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            self.proj_matrix[i] = Int8(1) if (state & 1) == 0 else Int8(-1)
        if is_not_null(self.proj_scratch):
            self.proj_scratch.free()
        self.proj_scratch = alloc[Float32](proj_dim)

    @always_inline
    def _project_vector(self, input: UnsafePointer[Float32, MutUntrackedOrigin], output: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Project original_dim-FP32 → proj_dim-FP32, scale by 1/sqrt(proj_dim)=1/16 for proj_dim=256."""
        var scale = Float32(0.0625)  # 1/16 = 1/sqrt(256)
        for r in range(self.proj_dim):
            var row = self.proj_matrix + r * self.original_dim
            var acc: Float32 = 0.0
            for c in range(self.original_dim):
                acc += Float32(Int(row[c])) * input[c]
            output[r] = acc * scale

    def _random_level(self) -> Int:
        var r = random_float64(0, 1).cast[DType.float32]()
        if r <= 0: r = 1e-7
        var level = Int(-log(r) * self.ml)
        if level > 6: level = 6
        return level

    def _calibrate(mut self, data: UnsafePointer[Float32, MutUntrackedOrigin],
                   count: Int) -> Tuple[Float32, Float32]:
        """gh #376: the one calibration every build runs — a single global
        INT8 range over mean ± 8 sigma of `count` row-major vectors (Welford).
        One scale for every dimension keeps code-space L2 proportional to
        true L2. Returns the range.

        Measured on the gate corpus (OpenAI 1536-d, 50K; interleaved A/B,
        3 runs per arm, against the old fixed ±0.2):
            fixed ±0.2 (pre-fix)        recall 0.9594
            mean ± 8 sigma, global      recall 0.9592
            mean ± 3 sigma, per-group   recall 0.9175
        The corpus has sigma 0.0255 but a few near-constant outlier
        dimensions reaching -0.69. Those cancel in every L2 difference, so
        clipping them costs nothing, while 3 sigma (±0.077) clips the 0.34%
        of components that really vary. The hand-tuned ±0.2 turns out to be
        8 sigma of this corpus, so 8 sigma reproduces it there and follows any
        other data's scale instead of clipping it (N(0,1): ±8).

        Per-group (Q8_K-style) ranges are no longer computed: at equal width
        they measured no better than one global range (0.951 both), and
        different scales per group weight the groups unequally in code-space
        L2. Index files written with group ranges still load and search.

        History worth knowing: until gh #376 welford_calibrate took its sqrt
        with three Newton steps seeded at the variance, which converges only
        near variance 1 — every "3 sigma" range here was really ~15 sigma on
        embedding data, which is why the old 3-sigma setting looked harmless."""
        var cal = welford_calibrate(data, count, self.dim, 8.0)
        self.global_min = cal[0]
        self.global_max = cal[1]
        self.grouped_calibrated = False
        print("[HNSW] calibrated [" + String(cal[0]) + ", " + String(cal[1])
              + "] from " + String(count) + " vectors")
        return cal

    def quantize(mut self, vector: UnsafePointer[Float32, MutUntrackedOrigin]) -> UnsafePointer[Int8, MutUntrackedOrigin]:
        var range_val = self.global_max - self.global_min
        if range_val <= 0: range_val = 1.0

        if self.use_int4 and not self.polarquant:
            var q_vector = self.vector_allocator.allocate()
            for i in range(0, self.dim, 2):
                var val1 = vector[i]
                var val2 = vector[i+1] if i + 1 < self.dim else 0.0
                var q1 = Int8(Int((val1 - self.global_min) / range_val * 15.0))
                var q2 = Int8(Int((val2 - self.global_min) / range_val * 15.0))
                if q1 > 15: q1 = 15
                if q1 < 0: q1 = 0
                if q2 > 15: q2 = 15
                if q2 < 0: q2 = 0
                q_vector[i // 2] = (q2 << 4) | q1
            return q_vector

        var q_vector = self.vector_allocator.allocate()
        if self.grouped_calibrated and self.num_groups > 0:
            # Per-group Q8_K quantization: each 32-dim group uses its own qmin/scale
            for g in range(self.num_groups):
                var goff = g * 32
                var gmin = self.group_qmins[g]
                var gscale = self.group_scales[g]
                for j in range(32):
                    var idx = goff + j
                    if idx >= self.dim: break
                    var norm = (vector[idx] - gmin) * gscale
                    var r = Int(norm + 0.5) if norm >= 0 else Int(norm - 0.5)
                    if r > 254: r = 254
                    if r < 0: r = 0
                    q_vector[idx] = Int8(r - 127)
        else:
            for i in range(self.dim):
                var val = vector[i]
                var normalized = (val - self.global_min) / range_val
                if normalized > 1.0: normalized = 1.0
                if normalized < 0: normalized = 0
                q_vector[i] = Int8(Int(normalized * 254.0) - 127)
        return q_vector

    @always_inline
    def _quantize_query_to_int8(mut self, query_in: UnsafePointer[Float32, MutUntrackedOrigin]):
        """Quantize FP32 query to self.query_int8 using per-group scales if calibrated.

        gh #271: under DISTANCE_METRIC COSINE the stored vectors were normalized
        at ingest, so the query must be normalized on the SAME side of the
        quantizer or the two live in different spaces. This is the single choke
        point for all three INT8 search entry points (brute force, GPU native,
        beam) — normalizing per call site is how this class of bug takes eleven
        passes to fix.
        """
        var query = query_in
        if self.distance_metric == 1:
            l2_normalize_fp32(query_in, self._norm_scratch(), self.dim)
            query = self.fp32_norm_scratch
        if self.grouped_calibrated and self.num_groups > 0:
            for g in range(self.num_groups):
                quantize_fp32_to_int8_simd(
                    query + g * 32, self.query_int8 + g * 32, 32,
                    self.group_qmins[g], self.group_scales[g])
        else:
            var range_val = self.global_max - self.global_min
            if range_val <= 0.0: range_val = 1.0
            var scale = Float32(254.0) / range_val
            quantize_fp32_to_int8_simd(query, self.query_int8, self.dim, self.global_min, scale)

    def _search_layer(mut self, query: UnsafePointer[Int8, MutUntrackedOrigin], entry_point_idx: Int, ef: Int, level: Int) raises:
        # Reuse pre-allocated self.candidates / self.results (no heap alloc per call)
        self._reset_visited()
        self.candidates.clear()
        self.results.clear()
        # gh #131 §2.5: bound the heaps up front (ef + max l0 degree) so push()
        # never reallocs mid-search on a cold graph.
        self.candidates.reserve(ef + 32)
        self.results.reserve(ef + 32)

        var dist = self._dist_int8_int8(query, self.nodes[entry_point_idx].vector)
        self.candidates.push(HeapNode(dist, entry_point_idx))
        self.results.push(HeapNode(dist, entry_point_idx))
        self.visited_epoch[entry_point_idx] = self.cur_epoch

        while len(self.candidates.data) > 0:
            var c = self.candidates.pop()

            if c.distance > self.results.peek_distance() and len(self.results.data) >= ef:
                break

            var neighbor_count = self.nodes[c.id].get_neighbor_count(level)
            for i in range(neighbor_count):
                var neighbor_idx = self.nodes[c.id].get_neighbor(level, i) # 3.1: internal index
                if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                if self.is_deleted(neighbor_idx): continue

                # Software Prefetching: Prefetch next neighbor's vector
                if i + 1 < neighbor_count:
                    var next_idx = self.nodes[c.id].get_neighbor(level, i + 1)
                    if next_idx >= 0 and next_idx < self.num_nodes:
                        prefetch((self.nodes + next_idx).bitcast[Int8]())
                        var nvptr = self.nodes[next_idx].vector
                        prefetch(nvptr); prefetch(nvptr + 64); prefetch(nvptr + 128); prefetch(nvptr + 192)
                        prefetch(self.nodes[next_idx].neighbors.bitcast[Int8]())
                if (self.visited_epoch[neighbor_idx] == self.cur_epoch):
                    continue

                self.visited_epoch[neighbor_idx] = self.cur_epoch
                var d = self._dist_int8_int8(query, self.nodes[neighbor_idx].vector)
                if self.results.push_bounded(d, neighbor_idx, ef):
                    self.candidates.push(HeapNode(d, neighbor_idx))

    @always_inline
    def is_deleted(self, idx: Int) -> Bool:
        return Bool((self.deleted_bitset[idx >> 3] >> UInt8(idx & 7)) & 1)

    @always_inline
    def mark_deleted(mut self, idx: Int):
        self.deleted_bitset[idx >> 3] |= UInt8(1 << (idx & 7))
        self.deleted_count += 1

    @always_inline
    def _norm_scratch(mut self) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """COSINE normalization scratch, at least `dim` floats. Grows when an
        FT.CREATE raised dim after the first allocation (never shrinks)."""
        if self.fp32_norm_scratch_cap < self.dim:
            if is_not_null(self.fp32_norm_scratch):
                self.fp32_norm_scratch.free()
            self.fp32_norm_scratch = alloc[Float32](self.dim)
            self.fp32_norm_scratch_cap = self.dim
        return self.fp32_norm_scratch

    @always_inline
    def _residual_scratch(mut self) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """gh #396: the QJL residual scratch, at least `dim` floats."""
        if self.residual_scratch_cap < self.dim:
            if is_not_null(self.residual_scratch):
                self.residual_scratch.free()
            self.residual_scratch = alloc[Float32](self.dim)
            self.residual_scratch_cap = self.dim
        return self.residual_scratch

    def reset_index(mut self):
        # gh #139: BM25 state is per-worker and never published to / borrowed
        # from the shared view, so it is owned even on a borrowing graph and has
        # to be cleared before the is_borrowed early-return. Leaving it behind
        # made FT.DROPINDEX a no-op for lexical search — a dropped index kept
        # answering `FT.SEARCH … BM25` from the stale postings. (A worker
        # self-borrows its own published index at -w 1, so this is the common
        # path, not an exotic multi-worker one.)
        if is_not_null(self.bm25_term_hashes):    self.bm25_term_hashes.free()
        if is_not_null(self.bm25_term_idf):       self.bm25_term_idf.free()
        if is_not_null(self.bm25_postings_start): self.bm25_postings_start.free()
        if is_not_null(self.bm25_postings_count): self.bm25_postings_count.free()
        if is_not_null(self.bm25_postings_buf):   self.bm25_postings_buf.free()
        if is_not_null(self.bm25_doc_lengths):    self.bm25_doc_lengths.free()
        if is_not_null(self.bm25_doc_ids):        self.bm25_doc_ids.free()
        if is_not_null(self.bm25_text_ids):       self.bm25_text_ids.free()
        self.bm25_term_hashes    = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_term_idf       = null_ptr[Float32, MutUntrackedOrigin]()
        self.bm25_postings_start = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_postings_count = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_postings_buf   = null_ptr[UInt32, MutUntrackedOrigin]()
        self.bm25_doc_lengths    = null_ptr[UInt16, MutUntrackedOrigin]()
        self.bm25_doc_ids        = null_ptr[Int32, MutUntrackedOrigin]()
        self.bm25_text_ids       = null_ptr[Int32, MutUntrackedOrigin]()
        self.bm25_vocab_count    = 0
        self.bm25_total_tokens   = 0
        self.bm25_is_built       = False
        self.bm25_doc_count      = 0
        self.bm25_text_count     = 0
        self.bm25_text_cap       = 0
        self.bm25_index_name_len = 0

        if self.is_borrowed:
            self.num_nodes = 0
            self.entry_point_id = -1
            self.index_ready = False
            return

        self.num_nodes = 0
        self.entry_point_id = -1
        self.max_level = -1
        self.cur_num = 1
        # gh #118: no visited memset — epoch counter stays monotonic across graph resets,
        # so stale stamps (all < cur_epoch) never read as visited.
        unsafe_memset(self.deleted_bitset, 0, self.visited_bitset_bytes)
        self.deleted_count = 0
        self.compact_dirty = False
        for i in range(self.max_elements):
            self.node_map[i] = -1
        self.index_ready = False
        self.index_name_len = 0
        self.global_min = -0.20
        self.global_max = 0.20

        if is_not_null(self.fp32_buffer): self.fp32_buffer.free()
        if is_not_null(self.fp32_ids): self.fp32_ids.free()
        self.fp32_buffer = null_ptr[Float32, MutUntrackedOrigin]()
        self.fp32_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.fp32_count = 0

        if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
        self.gpu_rerank_fp32 = null_ptr[Float32, MutUntrackedOrigin]()

        if is_not_null(self.compact_buffer): self.compact_buffer.free()
        self.compact_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        self.compact_is_int4 = False
        if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.l0_slots): self.l0_slots.free()
        self.l0_slots = null_ptr[UInt32, MutUntrackedOrigin]()
        if is_not_null(self.node_norms): self.node_norms.free()
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_prefix_norms): self.node_prefix_norms.free()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_meta_tag_hashes): self.node_meta_tag_hashes.free()
        if is_not_null(self.node_meta_numerics): self.node_meta_numerics.free()
        if is_not_null(self.node_meta_field_set): self.node_meta_field_set.free()
        self.node_meta_tag_hashes = null_ptr[UInt32, MutUntrackedOrigin]()
        self.node_meta_numerics = null_ptr[Float32, MutUntrackedOrigin]()
        self.node_meta_field_set = null_ptr[UInt8, MutUntrackedOrigin]()
        self.vector_allocator.reset()

    def bm25_register_text_doc(mut self, ext_id: Int):
        """gh #139: record a doc that carries text but may carry no vector.

        FT.ADDTEXT writes the text into the keyspace HASH but never creates an
        HNSW node, so `build_bm25` — which used to walk `nodes[]` — could not
        see it and `FT.SEARCH … BM25` silently returned `[0]`. Registering the
        ext_id here is what makes an ADDTEXT-only corpus searchable.

        Duplicates are allowed (dedup happens once, at build time); ids outside
        [0, max_elements) are dropped because the scoring scratch and doc-length
        arrays are indexed by ext_id."""
        if ext_id < 0 or ext_id >= self.max_elements: return
        if self.bm25_text_count >= self.bm25_text_cap:
            var new_cap = self.bm25_text_cap * 2 if self.bm25_text_cap > 0 else 1024
            var grown = alloc[Int32](new_cap)
            if is_not_null(self.bm25_text_ids):
                unsafe_memcpy(dest=grown, src=self.bm25_text_ids, count=self.bm25_text_count)
                self.bm25_text_ids.free()
            self.bm25_text_ids = grown
            self.bm25_text_cap = new_cap
        self.bm25_text_ids[self.bm25_text_count] = Int32(ext_id)
        self.bm25_text_count += 1

    def publish_to_shared(self, shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]):
        # V2.6: after build_index()+compact_vectors(), publish read-only pointers for other workers
        shared[].nodes = self.nodes
        shared[].node_map = self.node_map
        shared[].neighbor_pool = self.neighbor_pool
        shared[].neighbor_pool_per_node = self.neighbor_pool_per_node
        shared[].compact_buffer = self.compact_buffer
        shared[].compact_is_int4 = self.compact_is_int4
        shared[].compact_is_3bit = self.compact_is_3bit
        # gh #346: NanoQuant's flag and every quant variant's FP32 re-rank
        # buffer were never published, so a borrowing worker ran the INT8 beam
        # over an INT2 compact buffer (nano) or skipped the re-rank (all
        # three). Read-only borrow; the owner frees it (borrowers return early
        # from reset/deinit), under the same epoch reclamation as compact_buffer.
        shared[].compact_is_2bit = self.compact_is_2bit
        shared[].gpu_rerank_fp32 = self.gpu_rerank_fp32
        shared[].qjl_buffer = self.qjl_buffer
        shared[].qjl_res_norms = self.qjl_res_norms
        shared[].qjl_random_signs = self.qjl_random_signs
        shared[].node_norms = self.node_norms
        shared[].node_prefix_norms = self.node_prefix_norms
        shared[].num_nodes = self.num_nodes
        shared[].deleted_bitset = self.deleted_bitset
        shared[].deleted_count = self.deleted_count
        shared[].entry_point_id = self.entry_point_id
        shared[].max_level = self.max_level
        shared[].M = self.M
        shared[].dim = self.dim
        shared[].global_min = self.global_min
        shared[].global_max = self.global_max
        shared[].grouped_calibrated = self.grouped_calibrated
        if self.grouped_calibrated and self.num_groups > 0:
            for gi in range(self.num_groups):
                shared[].group_qmins[gi] = self.group_qmins[gi]
                shared[].group_scales[gi] = self.group_scales[gi]
        shared[].ef_runtime = self.ef_runtime
        # Only overwrite index_name if this worker actually saw FT.CREATE
        if self.index_name_len > 0:
            shared[].index_name_len = self.index_name_len
            for i in range(self.index_name_len):
                shared[].index_name[i] = self.index_name[i]
        # Projection metadata: borrowers need proj_dim + original_dim to project queries
        shared[].proj_dim = self.proj_dim
        shared[].original_dim = self.original_dim
        shared[].proj_matrix = self.proj_matrix
        # V33: compact level-0 neighbor array
        shared[].l0_compact = self.l0_compact
        # V34: prefix buffer — 256-byte prefix per node for SLC-resident prune phase
        shared[].prefix_buffer = self.prefix_buffer
        # K1: slot-space adjacency mirror + compact layout constants
        shared[].l0_slots = self.l0_slots
        shared[].compact_stride = self.compact_stride
        shared[].compact_hdr = self.compact_hdr
        # gh #271: a worker that ADOPTS this index must quantize its queries the
        # same way the ingesting worker normalized. Publishing the graph without
        # the metric is how a warm restart or a cross-worker adopt ends up
        # querying a normalized graph with a raw query.
        shared[].pre_distance_metric = self.distance_metric
        # gh #407: the vector field goes with the graph. FT.CREATE is what set
        # the shared copy, and a warm restart has no FT.CREATE, so every worker
        # refused `@<field>` for any field not named `vector` (the default).
        if self.vector_field_len > 0 and self.vector_field_len <= 32:
            shared[].pre_vector_field_len = self.vector_field_len
            for i in range(self.vector_field_len):
                shared[].pre_vector_field_name[i] = self.vector_field_name[i]
        shared[].ready = True  # legacy plain-Bool, kept for diagnostics
        # Real visibility barrier: atomic Release store after all field writes.
        # FT.SEARCH borrow check uses Atomic Acquire load to pair with this
        # — fixes recall≈0 race on weak memory (ARM) when other workers
        # observed ready=True but stale pointers (compact_buffer/num_nodes/etc).
        if is_not_null(shared[].ready_atomic):
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                shared[].ready_atomic, UInt64(1))

    def borrow_from_shared(mut self, shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]):
        # V2.6: lazy one-time init — borrow read-only pointers from the index-owning worker.
        # All pointers are shared read-only after ready=True; visited_map stays per-worker private.
        self.nodes = shared[].nodes
        self.node_map = shared[].node_map
        self.neighbor_pool = shared[].neighbor_pool
        self.neighbor_pool_per_node = shared[].neighbor_pool_per_node
        self.compact_buffer = shared[].compact_buffer
        self.compact_is_int4 = shared[].compact_is_int4
        self.compact_is_3bit = shared[].compact_is_3bit
        self.compact_is_2bit = shared[].compact_is_2bit   # gh #346
        # gh #346: CPU re-rank reads it; the Metal registration stays with the
        # owner (gpu_rerank_registered is False here, so _try_gpu_rerank falls
        # through to the CPU loop).
        self.gpu_rerank_fp32 = shared[].gpu_rerank_fp32
        self.qjl_buffer = shared[].qjl_buffer
        self.qjl_res_norms = shared[].qjl_res_norms
        # Borrow random signs but allocate per-worker query signs
        if is_not_null(shared[].qjl_random_signs):
            self.qjl_random_signs = shared[].qjl_random_signs
            if is_null(self.qjl_query_signs):
                self.qjl_query_signs = alloc[UInt64](QJL_U64S_1536)
            self.turboquant = True
        self.node_norms = shared[].node_norms
        self.node_prefix_norms = shared[].node_prefix_norms
        self.num_nodes = shared[].num_nodes
        self.deleted_count = shared[].deleted_count
        if is_not_null(shared[].deleted_bitset):
            self.deleted_bitset = shared[].deleted_bitset
        self.entry_point_id = shared[].entry_point_id
        self.max_level = shared[].max_level
        self.M = shared[].M
        self.dim = shared[].dim
        self.global_min = shared[].global_min
        self.global_max = shared[].global_max
        self.grouped_calibrated = shared[].grouped_calibrated
        if shared[].grouped_calibrated and self.num_groups > 0:
            for gi in range(self.num_groups):
                self.group_qmins[gi] = shared[].group_qmins[gi]
                self.group_scales[gi] = shared[].group_scales[gi]
        self.ef_runtime = shared[].ef_runtime
        self.index_name_len = shared[].index_name_len
        for i in range(shared[].index_name_len):
            self.index_name[i] = shared[].index_name[i]
        self.stat_query_count = 0
        self.stat_total_evals = 0
        self.is_borrowed = True # Phase 1.1: We are now a borrower
        self.index_ready = True
        # Projection: borrow matrix pointer so this worker can project queries
        self.proj_dim = shared[].proj_dim
        self.original_dim = shared[].original_dim
        self.proj_matrix = shared[].proj_matrix  # read-only borrow, coordinator owns memory
        # Allocate per-worker proj_scratch if projection is active
        if self.proj_dim > 0 and is_null(self.proj_scratch):
            self.proj_scratch = alloc[Float32](self.proj_dim)
        # V33: borrow compact level-0 neighbor array
        self.l0_compact = shared[].l0_compact
        # V34: borrow prefix buffer
        self.prefix_buffer = shared[].prefix_buffer
        # K1: borrow slot-space adjacency mirror + compact layout constants
        self.l0_slots = shared[].l0_slots
        self.compact_stride = shared[].compact_stride
        self.compact_hdr = shared[].compact_hdr
        self.distance_metric = shared[].pre_distance_metric   # gh #271

    # ── Disk persistence (Phase 1) ────────────────────────────────────────────

    def _pion_write_all(self, fd: Int32, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int):
        """Write exactly n bytes, retrying on short writes."""
        var remaining = n
        var p = ptr
        while remaining > 0:
            var written = external_call["pion_write", Int](fd, p, remaining)
            if written <= 0:
                break
            p = p + written
            remaining -= written

    def _pion_read_all(self, fd: Int32, ptr: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        """Read exactly n bytes. Returns False on short read or error."""
        var remaining = n
        var p = ptr
        while remaining > 0:
            var got = external_call["pion_read", Int](fd, p, remaining)
            if got <= 0:
                return False
            p = p + got
            remaining -= got
        return True

    def save_to_disk(self, path: String,
                     hk_keys: UnsafePointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin](),
                     hk_max: Int = 0):
        """Persist compact_buffer, neighbor graph, and metadata to disk.
        Called after FT.OPTIMIZE. Enables instant recovery on restart (skips FT.OPTIMIZE).
        gh #211: pass the SharedHNSWView's hk_keys_buf/hk_max_elements so the
        slot→original-key map survives the restart — without it a warm-loaded
        graph returns raw slot numbers from FT.SEARCH (recall ≈ random)."""
        if is_null(self.compact_buffer) or self.num_nodes == 0 or is_null(self.l0_compact):
            return

        var cpath = path
        var path_bytes = cpath.as_c_string_slice()
        var fd = external_call["pion_creat", Int32](path_bytes)
        if fd < 0:
            print("HNSW save: cannot open " + path)
            return
        self._save_body(fd, hk_keys, hk_max)
        _ = external_call["close", Int32](fd)

    def _save_body(self, fd: Int32,
                   hk_keys: UnsafePointer[UInt8, MutUntrackedOrigin],
                   hk_max: Int):
        # Header (256 bytes)
        var hdr = alloc[UInt64](32)  # 32 × 8 = 256 bytes
        unsafe_memset(hdr.bitcast[UInt8](), 0, 256)
        hdr[0] = UInt64(0x574E534E4F494E50)  # "PIONHNSW" LE
        # v2 (K1, 2026-08-05): the compact section is now written with the TRUE
        # slot stride. v1 wrote `num_nodes * dim` bytes, but the INT8 layout is
        # `num_nodes * (dim + 8)` (8B norm header/slot) — the file silently
        # truncated the last ~8·N bytes, and the v1 loader reconstructed vector
        # pointers with the (dim+8) stride into an undersized (num_nodes*dim)
        # allocation: garbage norms/distances for the tail slots + OOB reads.
        # Quant variants (stride 868/676/...) were saved with stride `dim` too.
        # v2 stores stride+hdr+format explicitly; v1 files are refused (loud) —
        # they were tail-corrupt anyway, and a cold FT.OPTIMIZE rebuilds.
        # v3 (gh #211, 2026-08-11): appends two sections after node_prefix_norms
        # — per-group INT8 calibration (queries were re-quantized with the
        # GLOBAL scale after a warm load while the stored codes were built
        # per-group) and the slot→original-key map (without it every key-
        # resolution tier misses after restart and FT.SEARCH emits raw slot
        # numbers: the silent recall ≈ 0.002). v2 files are refused loudly —
        # they reproduce exactly that bug.
        hdr[1] = UInt64(3)                   # version
        hdr[2] = UInt64(self.num_nodes)
        hdr[3] = UInt64(self.dim)
        hdr[4] = UInt64(self.M)
        hdr[5] = UInt64(self.ef_construction)
        hdr[6] = UInt64(self.entry_point_id)
        hdr[7] = UInt64(self.max_level)
        # global_min / global_max stored as Float32 via bitcast into UInt8 header bytes
        (hdr.bitcast[UInt8]() + 64).bitcast[Float32]()[0] = self.global_min
        (hdr.bitcast[UInt8]() + 68).bitcast[Float32]()[0] = self.global_max
        hdr[9]  = UInt64(self.neighbor_pool_per_node)
        hdr[10] = UInt64(self.max_elements)
        hdr[11] = UInt64(self.index_name_len)
        # v2 fields live at words 20-22 (bytes 160-183): the 64-byte index name
        # occupies bytes 96-159 = words 12-19, so anything written there gets
        # stomped by the name copy below (found the hard way: hdr[12] loaded
        # back as the ASCII bytes of "index" → a 517 GB compact alloc).
        hdr[20] = UInt64(self.compact_stride)
        hdr[21] = UInt64(self.compact_hdr)
        # Compact format bits: 1=int4, 2=3bit, 4=2bit (0 = plain INT8+hdr)
        var _fmt: UInt64 = 0
        if self.compact_is_int4: _fmt |= 1
        if self.compact_is_3bit: _fmt |= 2
        if self.compact_is_2bit: _fmt |= 4
        hdr[22] = _fmt
        # v3: number of calibration groups persisted (0 = not group-calibrated)
        hdr[23] = UInt64(self.num_groups) if self.grouped_calibrated else UInt64(0)
        # gh #271: DISTANCE_METRIC (0 = L2, 1 = COSINE). Word 24 is free and the
        # header is memset to 0 above, so a file written before this field
        # existed loads as 0 — which is precisely what those indexes are. That
        # is why this needs no version bump: the default IS the old behaviour.
        # It has to be persisted at all because the metric decides whether the
        # QUERY gets normalized, and a warm restart that forgot it would query
        # a normalized graph with an unnormalized query.
        hdr[24] = UInt64(self.distance_metric)
        # gh #350: optional quant sections, appended after the v3 key map.
        # bit 0 = FP32 re-rank buffer (num_nodes × dim Float32), bit 1 = QJL
        # signs + residual norms (TurboQuant). Every quant search is "beam over
        # the compact codes, then exact FP32 re-rank", and QJL is part of the
        # INT3 distance, so a warm load without them answers from a different
        # (worse) distance than the build did. Same reasoning as word 24: the
        # memset header makes every older file read 0 = no sections, so no
        # version bump. Plain INT8 keeps its file size — its re-rank buffer
        # exists only for the GPU oversample path.
        var _qsec: UInt64 = 0
        if _fmt != 0 and is_not_null(self.gpu_rerank_fp32): _qsec |= 1
        if self.compact_is_3bit and is_not_null(self.qjl_buffer) and is_not_null(self.qjl_res_norms): _qsec |= 2
        hdr[25] = _qsec
        # gh #407: the vector field name — word 26 = length, bytes 216-247 (words
        # 27-30) = up to 32 bytes. Same no-bump reasoning as words 24/25: an
        # older file reads length 0 and keeps the default field.
        hdr[26] = UInt64(self.vector_field_len) if self.vector_field_len <= 32 else UInt64(0)
        var vf_dst = hdr.bitcast[UInt8]() + 216
        if self.vector_field_len <= 32:
            for i in range(self.vector_field_len):
                vf_dst[i] = self.vector_field_name[i]
        # index_name: 64 bytes at offset 96
        var name_dst = hdr.bitcast[UInt8]() + 96
        for i in range(self.index_name_len):
            name_dst[i] = self.index_name[i]

        self._pion_write_all(fd, hdr.bitcast[UInt8](), 256)
        hdr.free()

        # Data sections — v2: TRUE slot stride (see version note above)
        var compact_bytes = self.num_nodes * self.compact_stride
        self._pion_write_all(fd, self.compact_buffer.bitcast[UInt8](), compact_bytes)

        var node_map_bytes = self.max_elements * 8
        self._pion_write_all(fd, self.node_map.bitcast[UInt8](), node_map_bytes)

        var l0_bytes = self.num_nodes * 33 * 4
        self._pion_write_all(fd, self.l0_compact.bitcast[UInt8](), l0_bytes)

        var pool_bytes = self.max_elements * self.neighbor_pool_per_node * 4
        self._pion_write_all(fd, self.neighbor_pool.bitcast[UInt8](), pool_bytes)

        # Node id+level array (2×Int per node = 16 bytes per node)
        var id_buf = alloc[Int](self.num_nodes * 2)
        for i in range(self.num_nodes):
            id_buf[i * 2]     = self.nodes[i].id
            id_buf[i * 2 + 1] = self.nodes[i].max_level
        self._pion_write_all(fd, id_buf.bitcast[UInt8](), self.num_nodes * 16)
        id_buf.free()

        # v3 (gh #211 root cause): per-node compact SLOT (UInt32 × num_nodes).
        # nodes[] is insertion-ordered but compact_buffer is BFS-ordered by
        # compact_vectors(); the v1/v2 loader reconstructed
        # nodes[i].vector = compact_buffer + i*stride — a silent BFS
        # permutation of every node's vector. The graph stayed navigable (same
        # vector set, near-identical distances), so a warm restart found the
        # RIGHT neighbors and returned them under the WRONG ids: the silent
        # recall ≈ 0.002 with healthy-looking logs.
        var slot_buf = alloc[UInt32](self.num_nodes)
        for i in range(self.num_nodes):
            slot_buf[i] = UInt32(
                (Int(self.nodes[i].vector) - Int(self.compact_buffer) - self.compact_hdr)
                // self.compact_stride)
        self._pion_write_all(fd, slot_buf.bitcast[UInt8](), self.num_nodes * 4)
        slot_buf.free()

        # node_norms (Float32 × num_nodes)
        if is_not_null(self.node_norms):
            self._pion_write_all(fd, self.node_norms.bitcast[UInt8](), self.num_nodes * 4)
        else:
            var zeros = alloc[UInt8](self.num_nodes * 4)
            unsafe_memset(zeros, 0, self.num_nodes * 4)
            self._pion_write_all(fd, zeros, self.num_nodes * 4)
            zeros.free()

        # node_prefix_norms (Float32 × num_nodes)
        if is_not_null(self.node_prefix_norms):
            self._pion_write_all(fd, self.node_prefix_norms.bitcast[UInt8](), self.num_nodes * 4)
        else:
            var zeros2 = alloc[UInt8](self.num_nodes * 4)
            unsafe_memset(zeros2, 0, self.num_nodes * 4)
            self._pion_write_all(fd, zeros2, self.num_nodes * 4)
            zeros2.free()

        # v3 section 1: per-group calibration (hdr[23] floats × 2) — the INT8
        # codes above were quantized with these; a query quantized with the
        # global fallback scores against mismatched codes.
        if self.grouped_calibrated and self.num_groups > 0:
            self._pion_write_all(fd, self.group_qmins.bitcast[UInt8](), self.num_groups * 4)
            self._pion_write_all(fd, self.group_scales.bitcast[UInt8](), self.num_groups * 4)

        # v3 section 2 (gh #211): slot→original-key map, one 32-byte record per
        # node in node order — record i belongs to external id nodes[i].id
        # (byte 0 = key length, 0 = unknown/>31B; bytes 1..31 = key). Sourced
        # from the SharedHNSWView's cross-worker map, which any worker's HSET
        # ingest writes. Records for slots outside [0, hk_max) stay zero; the
        # load-side resolver then falls through to the __hk__ keyspace probe
        # (rebuilt by WAL replay).
        var hk_section = alloc[UInt8](self.num_nodes * 32)
        unsafe_memset(hk_section, 0, self.num_nodes * 32)
        if is_not_null(hk_keys):
            for i in range(self.num_nodes):
                var ext_id = self.nodes[i].id
                if ext_id >= 0 and ext_id < hk_max:
                    unsafe_memcpy(dest=hk_section + i * 32, src=hk_keys + ext_id * 32, count=32)
        self._pion_write_all(fd, hk_section, self.num_nodes * 32)
        hk_section.free()

        # gh #350 quant sections (flagged in header word 25, see above).
        if _qsec & 1:
            self._pion_write_all(fd, self.gpu_rerank_fp32.bitcast[UInt8](),
                                 self.num_nodes * self.dim * 4)
        if _qsec & 2:
            self._pion_write_all(fd, self.qjl_buffer.bitcast[UInt8](),
                                 self.num_nodes * QJL_BYTES_1536)
            self._pion_write_all(fd, self.qjl_res_norms.bitcast[UInt8](),
                                 self.num_nodes * 4)

        print("HNSW saved: " + String(self.num_nodes) + " nodes → pion.hnsw.0")

    def load_from_disk(mut self, path: String,
                       hk_keys: UnsafePointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin](),
                       hk_max: Int = 0) -> Bool:
        """Load persisted HNSW index from disk. Returns True if loaded, False if file absent/corrupt.
        On success: index_ready=True, no FT.OPTIMIZE needed.
        gh #211: pass the SharedHNSWView's hk_keys_buf/hk_max_elements so the
        persisted slot→key map is restored into the cross-worker resolver."""
        var cpath = path
        var path_bytes = cpath.as_c_string_slice()
        var fd = external_call["pion_open_rdonly", Int32](path_bytes)
        if fd < 0:
            return False  # No saved index — normal cold start

        var ok = self._load_body(fd, hk_keys, hk_max)
        _ = external_call["close", Int32](fd)
        if ok:
            print("HNSW loaded from disk: " + String(self.num_nodes) + " nodes (" + path + ")")
            self.index_ready = True
        else:
            print("HNSW load failed — starting fresh")
        return ok

    def _load_body(mut self, fd: Int32,
                   hk_keys: UnsafePointer[UInt8, MutUntrackedOrigin],
                   hk_max: Int) -> Bool:
        # Read and validate 256-byte header
        var hdr = alloc[UInt64](32)
        if not self._pion_read_all(fd, hdr.bitcast[UInt8](), 256):
            hdr.free(); return False
        if hdr[0] != UInt64(0x574E534E4F494E50):  # magic check
            hdr.free(); return False
        if hdr[1] != UInt64(3):  # version check (v3 = group calibration + slot→key map)
            # v1 files were tail-corrupt; v2 files lack the slot→key map, so a
            # warm restart serves recall ≈ 0.002 silently (gh #211) — refuse
            # both loudly rather than load an index that returns wrong keys.
            print("HNSW snapshot version " + String(hdr[1]) + " unsupported (want 3) — cold rebuild (re-ingest + FT.OPTIMIZE)")
            hdr.free(); return False
        var saved_stride = Int(hdr[20])
        var saved_slot_hdr = Int(hdr[21])
        var saved_fmt = hdr[22]
        var _cur_fmt: UInt64 = 0
        if self.polarquant: _cur_fmt |= 1
        if self.turboquant: _cur_fmt |= 2
        if self.nanoquant:  _cur_fmt |= 4
        if self.use_int4:   _cur_fmt |= 1
        if saved_fmt != _cur_fmt:
            print("HNSW snapshot quant format mismatch (file=" + String(saved_fmt) + " server=" + String(_cur_fmt) + ") — cold rebuild")
            hdr.free(); return False
        if saved_stride <= 0:
            hdr.free(); return False

        var saved_num_nodes = Int(hdr[2])
        var saved_dim       = Int(hdr[3])
        var saved_M         = Int(hdr[4])
        var saved_entry     = Int(hdr[6])
        var saved_max_level = Int(hdr[7])
        var saved_pool_per  = Int(hdr[9])
        var saved_max_elem  = Int(hdr[10])
        var saved_name_len  = Int(hdr[11])
        var saved_groups    = Int(hdr[23])   # v3: calibration groups (0 = none)
        var saved_metric    = UInt8(hdr[24]) if hdr[24] <= 1 else UInt8(0)  # gh #271
        var saved_qsec      = hdr[25]        # gh #350: optional quant sections
        var saved_vf_len    = Int(hdr[26])   # gh #407: 0 = file predates the field

        # Sanity checks: must match this worker's config
        if saved_dim != self.dim or saved_M != self.M or saved_max_elem != self.max_elements:
            print("HNSW load: config mismatch (saved dim=" + String(saved_dim) + " M=" + String(saved_M) + ")")
            hdr.free(); return False
        if saved_num_nodes > self.max_elements or saved_num_nodes <= 0:
            hdr.free(); return False

        # global_min/max
        var gmin_bytes = hdr.bitcast[UInt8]() + 64
        var gmax_bytes = hdr.bitcast[UInt8]() + 68
        self.global_min = gmin_bytes.bitcast[Float32]()[0]
        self.global_max = gmax_bytes.bitcast[Float32]()[0]

        # index_name
        var name_src = hdr.bitcast[UInt8]() + 96
        self.index_name_len = saved_name_len
        for i in range(saved_name_len):
            self.index_name[i] = name_src[i]
        # gh #407: vector field name
        if saved_vf_len > 0 and saved_vf_len <= 32:
            var vf_src = hdr.bitcast[UInt8]() + 216
            self.vector_field_len = saved_vf_len
            for i in range(saved_vf_len):
                self.vector_field_name[i] = vf_src[i]

        hdr.free()

        # Allocate / reset compact_buffer — v2: TRUE slot stride from the header
        # (v1 used `dim` here while pointers stepped by dim+8: tail corruption).
        if is_not_null(self.compact_buffer):
            self.compact_buffer.free()
        # gh #196.2: keep the 64B base alignment on the load path too, or a
        # warm restart silently re-introduces the line-straddling the padded
        # stride paid to remove.
        self.compact_buffer = alloc[Int8](saved_num_nodes * saved_stride, alignment=64)
        if not self._pion_read_all(fd, self.compact_buffer.bitcast[UInt8](), saved_num_nodes * saved_stride):
            return False

        # node_map
        if not self._pion_read_all(fd, self.node_map.bitcast[UInt8](), self.max_elements * 8):
            return False

        # l0_compact
        if is_not_null(self.l0_compact):
            self.l0_compact.free()
        self.l0_compact = alloc[UInt32](saved_num_nodes * 33)
        if not self._pion_read_all(fd, self.l0_compact.bitcast[UInt8](), saved_num_nodes * 33 * 4):
            return False

        # neighbor_pool
        if not self._pion_read_all(fd, self.neighbor_pool.bitcast[UInt8](), self.max_elements * saved_pool_per * 4):
            return False

        # Node id+level
        var id_buf = alloc[Int](saved_num_nodes * 2)
        if not self._pion_read_all(fd, id_buf.bitcast[UInt8](), saved_num_nodes * 16):
            id_buf.free(); return False

        # v3: per-node compact slot — see _save_body. The vector pointer MUST
        # go through this indirection; assuming node i ↔ slot i (v1/v2) loads
        # a BFS-permuted labeling that silently returns wrong keys.
        var slot_buf = alloc[UInt32](saved_num_nodes)
        if not self._pion_read_all(fd, slot_buf.bitcast[UInt8](), saved_num_nodes * 4):
            slot_buf.free(); id_buf.free(); return False

        # Reconstruct nodes[i].vector pointer into compact_buffer.
        # v2+: stride and in-slot header offset come from the file header —
        # correct for INT8 (dim+8, +8) and every quant variant (868/676/..., +0).
        var slot_hdr = saved_slot_hdr
        var bpv_total = saved_stride

        for i in range(saved_num_nodes):
            var node_id    = id_buf[i * 2]
            var node_level = id_buf[i * 2 + 1]
            var nb_block   = self.neighbor_pool + i * saved_pool_per
            var vslot      = Int(slot_buf[i])
            if vslot < 0 or vslot >= saved_num_nodes:
                print("HNSW load: slot index out of range (node " + String(i) + " slot " + String(vslot) + ") — cold rebuild")
                slot_buf.free(); id_buf.free(); return False
            var vec_ptr    = self.compact_buffer + vslot * bpv_total + slot_hdr
            # gh #405: the per-level neighbor COUNTS live in this block (after
            # the 8*M list slots) and were just read from the file — and the
            # HNSWNode constructor zeroes them. Every warm load therefore ran
            # with empty neighbor lists at every level: level 0 still worked
            # (the beam reads l0_compact) but the upper layers were gone and
            # each search started at the entry point. Keep them.
            var cnt_base = nb_block + 8 * self.M
            var saved_cnts = Array[UInt32, 8](fill=UInt32(0))
            var nl = node_level + 1 if node_level + 1 <= 7 else 7
            for li in range(nl): saved_cnts[li] = cnt_base[li]
            (self.nodes + i).unsafe_write(
                HNSWNode(node_id, vec_ptr, node_level, self.M, nb_block))
            for li in range(nl): cnt_base[li] = saved_cnts[li]
        id_buf.free()
        slot_buf.free()

        # node_norms
        if is_not_null(self.node_norms):
            self.node_norms.free()
        self.node_norms = alloc[Float32](saved_num_nodes)
        if not self._pion_read_all(fd, self.node_norms.bitcast[UInt8](), saved_num_nodes * 4):
            return False

        # node_prefix_norms
        if is_not_null(self.node_prefix_norms):
            self.node_prefix_norms.free()
        self.node_prefix_norms = alloc[Float32](saved_num_nodes)
        if not self._pion_read_all(fd, self.node_prefix_norms.bitcast[UInt8](), saved_num_nodes * 4):
            return False

        # V34 prefix_buffer: not persisted; reconstruct from compact_buffer.
        # Each entry is the first 256 INT8 dims of the node's vector — used by the
        # SLC-resident prefix-prune phase in _beam_search_1536. Without this, search
        # dereferences a NULL prefix_buffer → SEGV (or random results in non-1536 path).
        # v3: indexed by NODE, sourced through the node's true slot (gh #211).
        # Plain INT8 only: a quant slot holds block scales + packed codes, not
        # INT8 dims, and no quant search reads prefix_buffer.
        if not self.use_int4 and saved_dim >= 256 and saved_fmt == 0:
            if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
            self.prefix_buffer = alloc[Int8](saved_num_nodes * 256)
            for i in range(saved_num_nodes):
                unsafe_memcpy(dest=self.prefix_buffer + i * 256,
                       src=self.nodes[i].vector, count=256)

        # v3 section 1: per-group calibration — restore so query quantization
        # matches the persisted INT8 codes (global-scale fallback silently
        # degrades warm-restart recall).
        if saved_groups > 0:
            if saved_groups != self.num_groups:
                print("HNSW load: calibration group count mismatch (file=" + String(saved_groups) + " server=" + String(self.num_groups) + ") — cold rebuild")
                return False
            if not self._pion_read_all(fd, self.group_qmins.bitcast[UInt8](), saved_groups * 4):
                return False
            if not self._pion_read_all(fd, self.group_scales.bitcast[UInt8](), saved_groups * 4):
                return False
            self.grouped_calibrated = True

        # v3 section 2 (gh #211): slot→original-key map, scattered back into the
        # cross-worker resolver buffer by external id. Without this every
        # FT.SEARCH after a warm restart falls through to raw slot numbers.
        var hk_section = alloc[UInt8](saved_num_nodes * 32)
        if not self._pion_read_all(fd, hk_section, saved_num_nodes * 32):
            hk_section.free(); return False
        if is_not_null(hk_keys):
            for i in range(saved_num_nodes):
                var ext_id = self.nodes[i].id
                var klen = Int(hk_section[i * 32])
                if ext_id >= 0 and ext_id < hk_max and klen > 0 and klen <= 31:
                    unsafe_memcpy(dest=hk_keys + ext_id * 32, src=hk_section + i * 32, count=32)
        hk_section.free()

        # gh #350 quant sections. A file that carries a section this server
        # has no use for is still read (it has to be consumed either way).
        if saved_qsec & 1:
            if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
            self.gpu_rerank_fp32 = alloc[Float32](saved_num_nodes * saved_dim)
            if not self._pion_read_all(fd, self.gpu_rerank_fp32.bitcast[UInt8](),
                                       saved_num_nodes * saved_dim * 4):
                return False
        if saved_qsec & 2:
            if is_not_null(self.qjl_buffer): self.qjl_buffer.free()
            self.qjl_buffer = alloc[UInt64](saved_num_nodes * QJL_U64S_1536)
            if not self._pion_read_all(fd, self.qjl_buffer.bitcast[UInt8](),
                                       saved_num_nodes * QJL_BYTES_1536):
                return False
            if is_not_null(self.qjl_res_norms): self.qjl_res_norms.free()
            self.qjl_res_norms = alloc[Float32](saved_num_nodes)
            if not self._pion_read_all(fd, self.qjl_res_norms.bitcast[UInt8](),
                                       saved_num_nodes * 4):
                return False
        # gh #350: the format word was CHECKED above but never applied, so a
        # warm PolarQuant server ran the INT8 beam over INT4 bytes (recall
        # ≈ 0.002 against its own fresh build). The search dispatch keys on
        # these flags, not on the server's quant options.
        self.compact_is_int4 = (saved_fmt & 1) != 0
        self.compact_is_3bit = (saved_fmt & 2) != 0
        self.compact_is_2bit = (saved_fmt & 4) != 0

        self.num_nodes       = saved_num_nodes
        self.entry_point_id  = saved_entry
        self.max_level       = saved_max_level
        self.neighbor_pool_per_node = saved_pool_per
        # K1: rebuild the slot-space adjacency mirror (not persisted — derived
        # in one pass from the reconstructed vector pointers).
        self.compact_stride = saved_stride
        self.compact_hdr = saved_slot_hdr
        self.distance_metric = saved_metric   # gh #271
        self._build_l0_slots()
        self._register_metal_rerank_buffer()  # needs num_nodes; no-op without GPU/buffer
        return True

    def add_vector(mut self, id: Int, vector: UnsafePointer[Float32, MutUntrackedOrigin]) raises:
        if self.streaming_mode:
            # Streaming mode: insert directly into graph — no FP32 staging buffer.
            # Saves max_elements × dim × 4 bytes (33.8GB at 5M×1536D).
            # Calibration: buffer first 1000 vectors, run Welford to set global_min/max,
            # then insert all 1000 and switch to direct insert for the rest.
            if not self.streaming_calibrated:
                if is_null(self.fp32_buffer):
                    self.fp32_buffer = alloc[Float32](1000 * self.dim)
                    self.fp32_ids = alloc[Int32](1000)
                for i in range(self.dim):
                    self.fp32_buffer[self.fp32_count * self.dim + i] = vector[i]
                # gh #271: normalize BEFORE calibration, not after. Welford runs
                # over this buffer to pick the quantizer range, so calibrating
                # on raw magnitudes and then storing unit vectors would size the
                # range for values that never get stored.
                if self.distance_metric == 1:
                    var _slot = self.fp32_buffer + self.fp32_count * self.dim
                    l2_normalize_fp32(_slot, _slot, self.dim)
                self.fp32_ids[self.fp32_count] = Int32(id)
                self.fp32_count += 1
                if self.fp32_count >= 1000:
                    var cal = self._calibrate(self.fp32_buffer, 1000)
                    print("Streaming calibration from first 1000 vectors: [" + String(cal[0]) + ", " + String(cal[1]) + "]")
                    # Insert all buffered vectors with calibrated range
                    for bi in range(1000):
                        self._insert_to_graph(Int(self.fp32_ids[bi]), self.fp32_buffer + bi * self.dim)
                    self.fp32_buffer.free()
                    self.fp32_buffer = null_ptr[Float32, MutUntrackedOrigin]()
                    self.fp32_ids.free()
                    self.fp32_ids = null_ptr[Int32, MutUntrackedOrigin]()
                    self.fp32_count = 0
                    self.streaming_calibrated = True
                return
            if self.distance_metric == 1:
                l2_normalize_fp32(vector, self._norm_scratch(), self.dim)
                self._insert_to_graph(id, self.fp32_norm_scratch)
                return
            self._insert_to_graph(id, vector)
            return
        # V2.2B: buffer FP32 for two-phase calibrated build (triggered by FT.OPTIMIZE)
        if is_null(self.fp32_buffer):
            self.fp32_buffer = alloc[Float32](self.max_elements * self.dim)
            self.fp32_ids = alloc[Int32](self.max_elements)
        if self.fp32_count >= self.max_elements:
            raise Error("FP32 buffer full")
        var offset = self.fp32_count * self.dim
        for i in range(self.dim):
            self.fp32_buffer[offset + i] = vector[i]
        # gh #271: COSINE stores unit vectors; FT.OPTIMIZE calibrates and builds
        # straight out of this buffer, so normalizing here covers both.
        if self.distance_metric == 1:
            l2_normalize_fp32(self.fp32_buffer + offset, self.fp32_buffer + offset, self.dim)
        self.fp32_ids[self.fp32_count] = Int32(id)
        self.fp32_count += 1

    def add_and_insert(mut self, id: Int, vector: UnsafePointer[Float32, MutUntrackedOrigin]) raises:
        """Immediate insert for SemanticCache — bypasses FP32 staging, rebuilds compact_buffer after each insert.
        compact_vectors() allocates a new buffer but does not free the old one — free it here to avoid leaks."""
        self._insert_to_graph(id, vector)
        # gh #140: DO NOT free compact_buffer before compact_vectors(). Every
        # existing node's .vector points INTO compact_buffer, and
        # compact_vectors() reads those pointers to rebuild. Freeing first left
        # them dangling; compact_vectors' own new_buf / l0_compact allocations
        # then reused the freed region, so the rebuild memcpy'd partially
        # overwritten bytes and silently corrupted the graph as it grew —
        # correct under ~100 nodes, random retrieval by a few hundred, and
        # embedder-independent. Capture, rebuild, then free.
        var old_compact = self.compact_buffer
        self.compact_vectors()   # repoints every node into a fresh buffer, sets self.compact_buffer
        if is_not_null(old_compact):
            old_compact.free()
        self.index_ready = True

    def insert_no_compact(mut self, id: Int, vector: UnsafePointer[Float32, MutUntrackedOrigin]) raises:
        """Insert a vector into the HNSW graph without compacting. Call finalize_compact() after batch inserts."""
        self._insert_to_graph(id, vector)

    def finalize_compact(mut self) raises:
        """Compact vectors and mark index as ready. Call after batch insert_no_compact() calls."""
        # gh #140: same use-after-free as add_and_insert — the old compact_buffer
        # is still referenced by every node's .vector until compact_vectors()
        # repoints them, so it can only be freed afterwards.
        var old_compact = self.compact_buffer
        self.compact_vectors()
        if is_not_null(old_compact):
            old_compact.free()
        self.index_ready = True

    def _insert_to_graph(mut self, id: Int, vector: UnsafePointer[Float32, MutUntrackedOrigin]) raises:
        var level = self._random_level()
        var q_vector = self.quantize(vector)

        var internal_id = self.num_nodes
        self.num_nodes += 1
        var neighbor_block = self.neighbor_pool + (internal_id * self.neighbor_pool_per_node)
        (self.nodes + internal_id).unsafe_write(HNSWNode(id, q_vector, level, self.M, neighbor_block))

        if self.entry_point_id == -1:
            if id >= 0 and id < self.max_elements:
                self.node_map[id] = internal_id
            self.entry_point_id = internal_id
            self.max_level = level
            return

        # Navigate down levels (Greedy)
        var search_entry_point = self.entry_point_id
        for l in range(self.max_level, level, -1):
            var changed = True
            var best_dist = self._dist_fp32_int8(vector, self.nodes[search_entry_point].vector)
            while changed:
                changed = False
                var neighbor_count = self.nodes[search_entry_point].get_neighbor_count(l)
                for i in range(neighbor_count):
                    var neighbor_idx = self.nodes[search_entry_point].get_neighbor(l, i) # 3.1: internal index
                    if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                    if self.is_deleted(neighbor_idx): continue

                    # Software Prefetching
                    if i + 1 < neighbor_count:
                        var next_idx = self.nodes[search_entry_point].get_neighbor(l, i + 1)
                        if next_idx >= 0 and next_idx < self.num_nodes:
                            prefetch((self.nodes + next_idx).bitcast[Int8]())
                            var nvptr2 = self.nodes[next_idx].vector
                            prefetch(nvptr2); prefetch(nvptr2 + 64); prefetch(nvptr2 + 128); prefetch(nvptr2 + 192)
                            prefetch(self.nodes[next_idx].neighbors.bitcast[Int8]())
                    var d = self._dist_fp32_int8(vector, self.nodes[neighbor_idx].vector)
                    if d < best_dist:
                        best_dist = d
                        search_entry_point = neighbor_idx
                        changed = True

        # Connect from level down to 0
        for l in range(min(level, self.max_level), -1, -1):
            self._search_layer(q_vector, search_entry_point, self.ef_construction, l)
            if len(self.results.data) > 0:
                var all_neighbors = List[HeapNode]()
                while len(self.results.data) > 0:
                    all_neighbors.append(self.results.pop())

                var m_limit = 2 * self.M if l == 0 else self.M

                # Closest neighbor for descent to next level
                search_entry_point = all_neighbors[len(all_neighbors)-1].id

                var pruned = self._prune_neighbors(q_vector, all_neighbors, m_limit)
                for i in range(len(pruned)):
                    var neighbor_idx = pruned[i] # 3.1: internal index
                    var neighbor_node_ptr = self.nodes + neighbor_idx
                    var cap = 2 * self.M if l == 0 else self.M
                    if Int(neighbor_node_ptr[].neighbor_counts[l]) < cap:
                        neighbor_node_ptr[].add_neighbor(l, internal_id) # Store internal index
                    else:
                        # Shrink (O(M)): replace the farthest existing neighbor if new candidate is closer
                        var nc = Int(neighbor_node_ptr[].neighbor_counts[l])
                        var worst_dist = Float32(-1.0)
                        var worst_pos = 0
                        var off = neighbor_node_ptr[]._get_offset(l)
                        for ni in range(nc):
                            var ex_nidx = Int(neighbor_node_ptr[].neighbors[off + ni]) # 3.1: internal index
                            if ex_nidx == -1:
                                worst_pos = ni; worst_dist = Float32(1e30); break
                            var d = self._dist_int8_int8(neighbor_node_ptr[].vector, self.nodes[ex_nidx].vector)
                            if worst_dist < 0 or d > worst_dist:
                                worst_dist = d; worst_pos = ni
                        var d_new = self._dist_int8_int8(neighbor_node_ptr[].vector, self.nodes[internal_id].vector)
                        if d_new < worst_dist:
                            # A1b: Capture evicted node before overwrite, remove stale backlink
                            var evicted_idx = Int(neighbor_node_ptr[].neighbors[off + worst_pos])
                            neighbor_node_ptr[].neighbors[off + worst_pos] = UInt32(internal_id)
                            # Remove evicted node's forward link to this neighbor (enforce bidirectionality)
                            if evicted_idx >= 0 and evicted_idx < self.num_nodes:
                                var evicted_ptr = self.nodes + evicted_idx
                                var evicted_off = evicted_ptr[]._get_offset(l)
                                var evicted_nc = Int(evicted_ptr[].neighbor_counts[l])
                                for ei in range(evicted_nc):
                                    if Int(evicted_ptr[].neighbors[evicted_off + ei]) == neighbor_idx:
                                        evicted_ptr[].neighbors[evicted_off + ei] = evicted_ptr[].neighbors[evicted_off + evicted_nc - 1]
                                        evicted_ptr[].neighbor_counts[l] = UInt32(evicted_nc - 1)
                                        break
                    self.nodes[internal_id].add_neighbor(l, neighbor_idx)

        if id >= 0 and id < self.max_elements:
            self.node_map[id] = internal_id

        if level > self.max_level:
            self.max_level = level
            self.entry_point_id = internal_id

    # ── gh #199: parallel FT.OPTIMIZE link phase ─────────────────────────────
    # Design (validated by the 2026-08-07 racy probe: 2.2× ceiling, zero recall
    # delta): serial pre-pass creates every node (levels, quantized vectors,
    # node_map, entry = global max-level node), then N pthread lanes (the
    # build_pool_wrap shim — Mojo's `parallelize` pool is occupied by the
    # server's event loops and deadlocks, see the issue) link strided subsets
    # concurrently. Correctness rules:
    #   - every neighbor-LIST mutation holds that node's spinlock;
    #   - never hold two node locks (the evict-backlink cleanup re-locks after
    #     releasing — bidirectionality is best-effort during the build, exactly
    #     as in hnswlib's concurrent addPoint);
    #   - counts are release-published after payload stores; concurrent lane
    #     readers acquire-load them (_search_layer_mt). Post-build readers are
    #     ordered by pthread_join, so the serial hot paths stay untouched.

    @always_inline
    def _node_lock(self, idx: Int):
        var lp = self.node_locks + idx
        while True:
            var expected: UInt32 = 0
            if Atomic[Scalar[DType.uint32]].compare_exchange(lp, expected, UInt32(1)):
                return

    @always_inline
    def _node_unlock(self, idx: Int):
        Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELEASE](self.node_locks + idx, UInt32(0))

    @always_inline
    def _add_neighbor_published(self, node_ptr: UnsafePointer[HNSWNode, MutUntrackedOrigin], level: Int, neighbor_idx: Int):
        """add_neighbor with a release-published count. Caller holds the node's
        lock. Payload store first, then the count store with RELEASE so a
        concurrent acquire-load reader never sees an unwritten slot."""
        if level <= node_ptr[].max_level:
            var count = node_ptr[].neighbor_counts[level]
            var capacity = 2 * self.M if level == 0 else self.M
            if count < UInt32(capacity):
                var offset = node_ptr[]._get_offset(level)
                node_ptr[].neighbors[offset + Int(count)] = UInt32(neighbor_idx)
                Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELEASE](
                    node_ptr[].neighbor_counts + level, count + 1)

    def _search_layer_mt(mut self, query: UnsafePointer[Int8, MutUntrackedOrigin],
                         entry_point_idx: Int, ef: Int, level: Int,
                         mut cand: MinHeap, mut res: MaxHeap,
                         visited: UnsafePointer[UInt16, MutUntrackedOrigin],
                         epoch: UInt16) raises:
        """_search_layer with caller-owned scratch and acquire-loaded neighbor
        counts (safe against concurrent lane publishes)."""
        cand.clear()
        res.clear()
        cand.reserve(ef + 32)
        res.reserve(ef + 32)
        var dist = self._dist_int8_int8(query, self.nodes[entry_point_idx].vector)
        cand.push(HeapNode(dist, entry_point_idx))
        res.push(HeapNode(dist, entry_point_idx))
        visited[entry_point_idx] = epoch
        while len(cand.data) > 0:
            var c = cand.pop()
            if c.distance > res.peek_distance() and len(res.data) >= ef:
                break
            var node_ptr = self.nodes + c.id
            if level > node_ptr[].max_level:
                continue
            var neighbor_count = Int(Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](
                node_ptr[].neighbor_counts + level))
            var off = node_ptr[]._get_offset(level)
            for i in range(neighbor_count):
                var neighbor_idx = Int(node_ptr[].neighbors[off + i])
                if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                if self.is_deleted(neighbor_idx): continue
                if visited[neighbor_idx] == epoch: continue
                visited[neighbor_idx] = epoch
                var d = self._dist_int8_int8(query, self.nodes[neighbor_idx].vector)
                if res.push_bounded(d, neighbor_idx, ef):
                    cand.push(HeapNode(d, neighbor_idx))

    def _insert_to_graph_mt(mut self, internal_id: Int, level: Int,
                            vector: UnsafePointer[Float32, MutUntrackedOrigin],
                            mut cand: MinHeap, mut res: MaxHeap,
                            visited: UnsafePointer[UInt16, MutUntrackedOrigin],
                            epoch_cell: UnsafePointer[UInt16, MutUntrackedOrigin]) raises:
        """Link phase of _insert_to_graph under the gh #199 locking rules.
        The node itself exists already (pre-pass)."""
        var q_vector = self.nodes[internal_id].vector
        var search_entry_point = self.entry_point_id
        for l in range(self.max_level, level, -1):
            var changed = True
            var best_dist = self._dist_fp32_int8(vector, self.nodes[search_entry_point].vector)
            while changed:
                changed = False
                var ep_ptr = self.nodes + search_entry_point
                if l > ep_ptr[].max_level:
                    break
                var neighbor_count = Int(Atomic[Scalar[DType.uint32]].load[ordering=Ordering.ACQUIRE](
                    ep_ptr[].neighbor_counts + l))
                var ep_off = ep_ptr[]._get_offset(l)
                for i in range(neighbor_count):
                    var neighbor_idx = Int(ep_ptr[].neighbors[ep_off + i])
                    if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                    if self.is_deleted(neighbor_idx): continue
                    var d = self._dist_fp32_int8(vector, self.nodes[neighbor_idx].vector)
                    if d < best_dist:
                        best_dist = d
                        search_entry_point = neighbor_idx
                        changed = True
        for l in range(min(level, self.max_level), -1, -1):
            epoch_cell[0] += 1
            if epoch_cell[0] == 0:
                unsafe_memset(visited.bitcast[UInt8](), 0, self.max_elements * 2)
                epoch_cell[0] = 1
            self._search_layer_mt(q_vector, search_entry_point, self.ef_construction, l,
                                  cand, res, visited, epoch_cell[0])
            if len(res.data) > 0:
                var all_neighbors = List[HeapNode]()
                while len(res.data) > 0:
                    all_neighbors.append(res.pop())
                var m_limit = 2 * self.M if l == 0 else self.M
                search_entry_point = all_neighbors[len(all_neighbors)-1].id
                var pruned = self._prune_neighbors(q_vector, all_neighbors, m_limit)
                for i in range(len(pruned)):
                    var neighbor_idx = pruned[i]
                    var neighbor_node_ptr = self.nodes + neighbor_idx
                    var cap = 2 * self.M if l == 0 else self.M
                    # ── backlink into the neighbor, under its lock ──
                    var evicted_idx = -1
                    self._node_lock(neighbor_idx)
                    var nc = Int(neighbor_node_ptr[].neighbor_counts[l])
                    if nc < cap:
                        self._add_neighbor_published(neighbor_node_ptr, l, internal_id)
                    else:
                        var worst_dist = Float32(-1.0)
                        var worst_pos = 0
                        var off = neighbor_node_ptr[]._get_offset(l)
                        for ni in range(nc):
                            var ex_nidx = Int(neighbor_node_ptr[].neighbors[off + ni])
                            if ex_nidx < 0 or ex_nidx >= self.num_nodes:
                                worst_pos = ni; worst_dist = Float32(1e30); break
                            var d = self._dist_int8_int8(neighbor_node_ptr[].vector, self.nodes[ex_nidx].vector)
                            if worst_dist < 0 or d > worst_dist:
                                worst_dist = d; worst_pos = ni
                        var d_new = self._dist_int8_int8(neighbor_node_ptr[].vector, self.nodes[internal_id].vector)
                        if d_new < worst_dist:
                            evicted_idx = Int(neighbor_node_ptr[].neighbors[off + worst_pos])
                            neighbor_node_ptr[].neighbors[off + worst_pos] = UInt32(internal_id)
                    self._node_unlock(neighbor_idx)
                    # ── evicted node's stale backlink: separate lock, never
                    # nested (deadlock-freedom by single-lock discipline) ──
                    if evicted_idx >= 0 and evicted_idx < self.num_nodes:
                        self._node_lock(evicted_idx)
                        var evicted_ptr = self.nodes + evicted_idx
                        if l <= evicted_ptr[].max_level:
                            var evicted_off = evicted_ptr[]._get_offset(l)
                            var evicted_nc = Int(evicted_ptr[].neighbor_counts[l])
                            for ei in range(evicted_nc):
                                if Int(evicted_ptr[].neighbors[evicted_off + ei]) == neighbor_idx:
                                    evicted_ptr[].neighbors[evicted_off + ei] = evicted_ptr[].neighbors[evicted_off + evicted_nc - 1]
                                    Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELEASE](
                                        evicted_ptr[].neighbor_counts + l, UInt32(evicted_nc - 1))
                                    break
                        self._node_unlock(evicted_idx)
                    # ── own list, under our lock (other lanes backlink into us).
                    # gh #212: the serial path never sees a full own list (it
                    # only holds the ≤ m_limit pruned picks), but here other
                    # lanes may have backlinked into us first. The old
                    # capacity-guarded add silently dropped the pruned pick —
                    # and the drop count scales with contention (racier
                    # interleaving → fuller lists at link time), which is what
                    # dragged loaded-machine builds to recall 0.89–0.935.
                    # Mirror the serial shrink instead: evict the farthest
                    # existing neighbor when the pruned pick is closer.
                    #
                    # gh #405: and leave the evicted node's edge TO US alone.
                    # This eviction exists only in the parallel build, and the
                    # entry it evicts is a backlink — the evicted node chose us
                    # as one of ITS pruned (diverse, often long-range) picks.
                    # Removing that reverse edge, as the backlink shrink does,
                    # stripped exactly the bridges between regions, and in
                    # some builds all of them: the upper-level descent then
                    # left a query in a region its level-0 beam could not get
                    # out of (polar 0.892 / turbo 0.880 where the serial build
                    # gives 0.968 / 0.953, one build in 6-8). Keeping it:
                    # 10/10 polar builds 0.9675-0.9680, the serial build's
                    # 0.9679; keeping reverse edges at EVERY eviction site
                    # instead costs ~0.5pp (0.962) — the backlink shrink's
                    # removal is part of how the serial build gets its recall.
                    self._node_lock(internal_id)
                    var own_ptr = self.nodes + internal_id
                    var onc = Int(own_ptr[].neighbor_counts[l])
                    if onc < cap:
                        self._add_neighbor_published(own_ptr, l, neighbor_idx)
                    else:
                        var o_worst_dist = Float32(-1.0)
                        var o_worst_pos = 0
                        var o_off = own_ptr[]._get_offset(l)
                        for oi in range(onc):
                            var o_nidx = Int(own_ptr[].neighbors[o_off + oi])
                            if o_nidx < 0 or o_nidx >= self.num_nodes:
                                o_worst_pos = oi; o_worst_dist = Float32(1e30); break
                            var od = self._dist_int8_int8(own_ptr[].vector, self.nodes[o_nidx].vector)
                            if o_worst_dist < 0 or od > o_worst_dist:
                                o_worst_dist = od; o_worst_pos = oi
                        var o_d_new = self._dist_int8_int8(own_ptr[].vector, self.nodes[neighbor_idx].vector)
                        if o_d_new < o_worst_dist:
                            own_ptr[].neighbors[o_off + o_worst_pos] = UInt32(neighbor_idx)
                        else:
                            _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](
                                self.link_drops, UInt64(1))
                    self._node_unlock(internal_id)

    def _relink_node_l0(mut self, internal_id: Int) raises:
        """gh #212 repair: re-run the level-0 link step for one under-linked
        node, serially (post-join — no locks needed). Searches from the entry
        point, prunes, then adds bidirectional links with the serial shrink
        semantics from _insert_to_graph."""
        if self.entry_point_id < 0 or self.num_nodes <= 1:
            return
        var q = self.nodes[internal_id].vector
        self._search_layer(q, self.entry_point_id, self.ef_construction, 0)
        if len(self.results.data) == 0:
            return
        var all_neighbors = List[HeapNode]()
        while len(self.results.data) > 0:
            all_neighbors.append(self.results.pop())
        var m_limit = 2 * self.M
        var pruned = self._prune_neighbors(q, all_neighbors, m_limit)
        var own_ptr = self.nodes + internal_id
        var own_off = own_ptr[]._get_offset(0)
        for i in range(len(pruned)):
            var neighbor_idx = pruned[i]
            if neighbor_idx == internal_id:
                continue
            # Skip picks we already link to (backlinks from the MT phase).
            var already = False
            var onc = Int(own_ptr[].neighbor_counts[0])
            for oi in range(onc):
                if Int(own_ptr[].neighbors[own_off + oi]) == neighbor_idx:
                    already = True
                    break
            if already:
                continue
            if onc >= m_limit:
                break
            own_ptr[].add_neighbor(0, neighbor_idx)
            # Backlink with the serial full-list shrink (mirror of
            # _insert_to_graph's A1b block, level 0 only).
            var neighbor_node_ptr = self.nodes + neighbor_idx
            var nc = Int(neighbor_node_ptr[].neighbor_counts[0])
            var off = neighbor_node_ptr[]._get_offset(0)
            var has_back = False
            for bi in range(nc):
                if Int(neighbor_node_ptr[].neighbors[off + bi]) == internal_id:
                    has_back = True
                    break
            if has_back:
                continue
            if nc < m_limit:
                neighbor_node_ptr[].add_neighbor(0, internal_id)
            else:
                var worst_dist = Float32(-1.0)
                var worst_pos = 0
                for ni in range(nc):
                    var ex_nidx = Int(neighbor_node_ptr[].neighbors[off + ni])
                    if ex_nidx < 0 or ex_nidx >= self.num_nodes:
                        worst_pos = ni; worst_dist = Float32(1e30); break
                    var d = self._dist_int8_int8(neighbor_node_ptr[].vector, self.nodes[ex_nidx].vector)
                    if worst_dist < 0 or d > worst_dist:
                        worst_dist = d; worst_pos = ni
                var d_new = self._dist_int8_int8(neighbor_node_ptr[].vector, own_ptr[].vector)
                if d_new < worst_dist:
                    var evicted_idx = Int(neighbor_node_ptr[].neighbors[off + worst_pos])
                    neighbor_node_ptr[].neighbors[off + worst_pos] = UInt32(internal_id)
                    if evicted_idx >= 0 and evicted_idx < self.num_nodes:
                        var evicted_ptr = self.nodes + evicted_idx
                        var evicted_off = evicted_ptr[]._get_offset(0)
                        var evicted_nc = Int(evicted_ptr[].neighbor_counts[0])
                        for ei in range(evicted_nc):
                            if Int(evicted_ptr[].neighbors[evicted_off + ei]) == neighbor_idx:
                                evicted_ptr[].neighbors[evicted_off + ei] = evicted_ptr[].neighbors[evicted_off + evicted_nc - 1]
                                evicted_ptr[].neighbor_counts[0] = UInt32(evicted_nc - 1)
                                break

    def _mark_reachable_l0(self, start: Int, reach: UnsafePointer[UInt8, MutUntrackedOrigin],
                           mut queue: List[Int]):
        """BFS over level-0 lists from `start`, setting reach[] for every node
        it gets to (already-marked nodes are not re-expanded)."""
        if reach[start] != 0: return
        reach[start] = 1
        queue.clear()
        queue.append(start)
        var head = 0
        while head < len(queue):
            var x = queue[head]; head += 1
            var c = self.nodes[x].get_neighbor_count(0)
            for k in range(c):
                var nb = self.nodes[x].get_neighbor(0, k)
                if nb < 0 or nb >= self.num_nodes or reach[nb] != 0: continue
                if self.is_deleted(nb): continue
                reach[nb] = 1
                queue.append(nb)

    def _repair_l0_reachability(mut self) raises -> Int:
        """gh #405: link every level-0 node that the entry point cannot reach.

        The insert paths evict a full list's farthest link AND the evicted
        node's link back (A1b "enforce bidirectionality"), so a dense cluster
        can lose every inbound link. Whether it does depends on insert order,
        which the parallel link phase (gh #199) makes nondeterministic: most
        OpenAI-50K builds leave ~25 nodes unreachable, some leave an island of
        ~1,300. An island costs recall twice. Its nodes can never be returned,
        and a query whose upper-level descent lands INSIDE it is trapped there
        (turbo: 0.953 healthy, 0.925 without the island, 0.880 live).

        Serial, post-join, before compact_vectors() so l0_compact/l0_slots see
        the links. For each unreachable node u (index order): search level 0
        from the entry point, which only walks the reachable graph, and link
        the closest result r -> u, into a free slot when one of the results
        has one, else over r's farthest link w when w keeps another in-link.
        u -> r is added when u has room, so a search that enters u's region
        can leave it. Then BFS from u marks everything u newly makes
        reachable. Repeats (bounded) in case a replacement stranded a node.
        Returns the number of links made."""
        var n = self.num_nodes
        if n <= 1 or self.entry_point_id < 0: return 0
        var cap = 2 * self.M
        var reach = alloc[UInt8](n)
        var indeg = alloc[Int32](n)
        var queue = List[Int]()
        var linked_total = 0
        for _pass in range(3):
            unsafe_memset(reach, 0, n)
            unsafe_memset(indeg.bitcast[UInt8](), 0, n * 4)
            for i in range(n):
                if self.is_deleted(i): continue
                var c = self.nodes[i].get_neighbor_count(0)
                for k in range(c):
                    var nb = self.nodes[i].get_neighbor(0, k)
                    if nb >= 0 and nb < n: indeg[nb] += 1
            self._mark_reachable_l0(self.entry_point_id, reach, queue)
            var linked_pass = 0
            for u in range(n):
                if reach[u] != 0 or self.is_deleted(u): continue
                self._search_layer(self.nodes[u].vector, self.entry_point_id, self.ef_construction, 0)
                var found = List[HeapNode]()
                while len(self.results.data) > 0:
                    found.append(self.results.pop())   # farthest first
                var r_pick = -1
                # 1) the closest reachable node with a free slot
                var fi = len(found) - 1
                while fi >= 0:
                    var r = found[fi].id
                    if r != u and reach[r] != 0 and Int(self.nodes[r].neighbor_counts[0]) < cap:
                        self.nodes[r].add_neighbor(0, u)
                        r_pick = r
                        break
                    fi -= 1
                # 2) else over the farthest link of the closest node whose
                #    farthest neighbor keeps another in-link
                fi = len(found) - 1
                while r_pick < 0 and fi >= 0:
                    var r = found[fi].id
                    fi -= 1
                    if r == u or reach[r] == 0: continue
                    var r_ptr = self.nodes + r
                    var off = r_ptr[]._get_offset(0)
                    var nc = Int(r_ptr[].neighbor_counts[0])
                    var worst_pos = -1
                    var worst_d = Float32(-1.0)
                    for ni in range(nc):
                        var w = Int(r_ptr[].neighbors[off + ni])
                        if w < 0 or w >= n: continue
                        var d = self._dist_int8_int8(r_ptr[].vector, self.nodes[w].vector)
                        if d > worst_d:
                            worst_d = d; worst_pos = ni
                    if worst_pos < 0: continue
                    var w_idx = Int(r_ptr[].neighbors[off + worst_pos])
                    if indeg[w_idx] < 2: continue
                    r_ptr[].neighbors[off + worst_pos] = UInt32(u)
                    indeg[w_idx] -= 1
                    r_pick = r
                if r_pick < 0: continue
                indeg[u] += 1
                if Int(self.nodes[u].neighbor_counts[0]) < cap:
                    var has = False
                    var uoff = self.nodes[u]._get_offset(0)
                    for k in range(Int(self.nodes[u].neighbor_counts[0])):
                        if Int(self.nodes[u].neighbors[uoff + k]) == r_pick:
                            has = True; break
                    if not has:
                        self.nodes[u].add_neighbor(0, r_pick)
                        indeg[r_pick] += 1
                linked_pass += 1
                self._mark_reachable_l0(u, reach, queue)
            linked_total += linked_pass
            if linked_pass == 0: break
        reach.free()
        indeg.free()
        return linked_total

    def build_index_from_shared(mut self, shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin], shard_id: Int = -1, num_shards: Int = 1) raises:
        """V3.1: build graph from the shared ingest buffer (any worker can call this after distributed HSET).
        V8.1 path: build graph in 1536-dim INT8, BFS-compact, search with INT8-INT8 batch-8."""
        if is_null(shared[].ingest_count) or Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].ingest_count, UInt64(0)) == 0: return
        self.num_nodes = 0
        self.entry_point_id = -1
        self.max_level = -1
        self.index_ready = True  # V3.1 fix: mark ready so FT.SEARCH on this same worker works
        self.cur_num = 1
        # gh #118: epoch-stamped visited set needs no reset here (monotonic counter).
        for i in range(self.max_elements):
            self.node_map[i] = -1
        self.vector_allocator.reset()

        var n = Int(Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](shared[].ingest_count, UInt64(0)))
        # Cap n at max_elements: ingest_count can over-increment past the buffer
        # capacity if HSET multi-field overflows (see SharedHNSWView.add_ingest_vector
        # bounds check). Without this cap, iterating past the buffer reads garbage
        # → SIGSEGV in compact_vectors. Companion to the add_ingest_vector fix.
        if n > self.max_elements:
            n = self.max_elements
        if shard_id < 0:
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                shared[].ingest_count, UInt64(0)
            )

        # gh #376: calibrate the quantizer from the vectors about to be
        # quantized. This build used to skip it and quantize with the
        # constructor's fixed ±0.2 range, so any component outside it was
        # clipped at ingest — invisible on OpenAI-scale embeddings (±0.03),
        # a self-match scoring ~1,134 on N(0,1) data. See _calibrate for
        # the range rule and the recall measurement behind it. The ingest buffer is
        # already normalized under COSINE (gh #271), so this calibrates the
        # vectors that actually get stored. publish_to_shared carries the
        # result to every borrowing worker.
        if n > 0:
            _ = self._calibrate(shared[].ingest_fp32, min(n, CALIBRATION_SAMPLE_MAX))

        # ── gh #199: parallel build. Serial pre-pass (levels/quantize/node
        # init/node_map, entry = global max-level node — deterministic, no
        # racy entry updates), then the pthread pool links strided subsets
        # under the per-node-lock rules above. n_threads=1 for small builds
        # keeps one code path. (The dormant gh #5 shard-skip was dropped with
        # the serial loop — num_shards is pinned at 1.)
        var bp_max_lvl = -1
        var bp_max_idx = 0
        for buf_idx in range(n):
            var lvl = self._random_level()
            if lvl > bp_max_lvl:
                bp_max_lvl = lvl
                bp_max_idx = buf_idx
            var q = self.quantize(shared[].ingest_fp32 + buf_idx * self.dim)
            var nb = self.neighbor_pool + buf_idx * self.neighbor_pool_per_node
            (self.nodes + buf_idx).unsafe_write(
                HNSWNode(Int(shared[].ingest_ids[buf_idx]), q, lvl, self.M, nb))
            var vid = Int(shared[].ingest_ids[buf_idx])
            if vid >= 0 and vid < self.max_elements:
                self.node_map[vid] = buf_idx
        self.num_nodes = n
        self.entry_point_id = bp_max_idx
        self.max_level = bp_max_lvl
        if n > 0:
            var n_threads = 4 if n >= 4096 else 1
            self.link_drops[0] = 0  # gh #212: per-build accounting
            var bctx = alloc[Int64](5)
            bctx[0] = Int64(Int(UnsafePointer(to=self)))
            bctx[1] = Int64(Int(shared[].ingest_fp32))
            bctx[2] = Int64(n)
            bctx[3] = Int64(bp_max_idx)
            bctx[4] = Int64(n_threads)
            var bp_sym = String("pion_gh199_build_lane") + "\0"
            var bp_rc = external_call["pion_build_pool_run_sym", Int32](
                bp_sym.unsafe_ptr(), bctx, Int64(n_threads))
            if bp_rc != 0:
                # -1: some lanes ran serially on the caller (still complete);
                # -2: dlsym miss — build produced an UNLINKED graph, fail loud.
                print("[HNSW] parallel build rc=", bp_rc)
            bctx.free()
            # gh #212: post-join repair — single-threaded and deterministic
            # (pthread_join above orders every lane's stores). Nodes that left
            # the concurrent link phase with a degenerate level-0 list are
            # re-linked serially; a sparse node near the entry region poisons
            # every search that routes through it, which is how loaded-machine
            # builds fell to 0.89 recall. Runs BEFORE compact_vectors() so the
            # repaired lists flow into l0_compact/l0_slots.
            var _repaired = 0
            if n_threads > 1:
                for ni in range(n):
                    if Int(self.nodes[ni].neighbor_counts[0]) < self.M // 2:
                        self._relink_node_l0(ni)
                        _repaired += 1
            var _drops = self.link_drops[0]  # plain read — lanes joined
            if _drops > 0 or _repaired > 0:
                print("[HNSW] parallel link: " + String(_drops)
                      + " contended link drops, " + String(_repaired)
                      + " under-linked nodes repaired")
            var _l0_links = self._repair_l0_reachability()   # gh #405
            if _l0_links > 0:
                print("[HNSW] level-0 reachability: " + String(_l0_links)
                      + " unreachable node(s) linked into the graph")
        # §7v2 / M6: Borrow shared FP32 data for the GPU rerank buffer and for
        # every block quantizer. gh #350: TurboQuant and NanoQuant were missing
        # here, and both compactors are gated on `fp32_count > 0` — so on the
        # HSET ingest route (the only route FT.OPTIMIZE builds from) they fell
        # through to plain INT8 compaction. The server said `TurboQuant: True`,
        # searched INT8, and wrote format 0 to disk, which its own loader then
        # refused.
        var saved_fp32_buf = self.fp32_buffer
        var saved_fp32_ids = self.fp32_ids
        var saved_fp32_count = self.fp32_count
        if (self.has_gpu or self.polarquant or self.turboquant or self.nanoquant) and is_not_null(shared[].ingest_fp32):
            self.fp32_buffer = shared[].ingest_fp32
            self.fp32_ids = shared[].ingest_ids
            self.fp32_count = n
        self.compact_vectors()
        # Restore original pointers (don't free — shared owns them)
        self.fp32_buffer = saved_fp32_buf
        self.fp32_ids = saved_fp32_ids
        self.fp32_count = saved_fp32_count
        # V32 PQ: permanently disabled — see build_index() comment.

    def build_index(mut self) raises:
        if self.streaming_mode:
            # Streaming mode: graph already built during add_vector() calls.
            # Flush any remaining calibration buffer (< 1000 vectors ingested total).
            if not self.streaming_calibrated and self.fp32_count > 0 and is_not_null(self.fp32_buffer):
                _ = self._calibrate(self.fp32_buffer, self.fp32_count)
                for bi in range(self.fp32_count):
                    self._insert_to_graph(Int(self.fp32_ids[bi]), self.fp32_buffer + bi * self.dim)
                self.fp32_buffer.free()
                self.fp32_buffer = null_ptr[Float32, MutUntrackedOrigin]()
                self.fp32_ids.free()
                self.fp32_ids = null_ptr[Int32, MutUntrackedOrigin]()
                self.fp32_count = 0
                self.streaming_calibrated = True
            if self.num_nodes == 0: return
            self.compact_vectors()
            return
        # V2.2B: rebuild HNSW from FP32 buffer using final calibration
        if self.fp32_count == 0: return

        # Welford sigma-clipped calibration: compute optimal quantization range from actual data.
        # Replaces hardcoded [-0.20, 0.20] with data-driven [mean-3σ, mean+3σ].
        # Better range utilization → ~0.5-1% recall improvement for non-OpenAI embeddings.
        _ = self._calibrate(self.fp32_buffer, self.fp32_count)

        # Reset graph state
        self.num_nodes = 0
        self.entry_point_id = -1
        self.max_level = -1
        self.cur_num = 1
        # gh #118: epoch-stamped visited set needs no reset here (monotonic counter).
        self.vector_allocator.reset()

        var n = self.fp32_count

        # Build graph from all buffered FP32 vectors
        for buf_idx in range(n):
            var id = Int(self.fp32_ids[buf_idx])
            var fp32_ptr = self.fp32_buffer + buf_idx * self.dim
            self._insert_to_graph(id, fp32_ptr)
        var _l0_links = self._repair_l0_reachability()   # gh #405
        if _l0_links > 0:
            print("[HNSW] level-0 reachability: " + String(_l0_links)
                  + " unreachable node(s) linked into the graph")
        # V2.3: compaction (fp32_count kept alive so compact_vectors can build rerank buffer)
        self.compact_vectors()
        # Reset count after compact (keep buffer allocated for potential re-use)
        self.fp32_count = 0
        # V32 DiskANN-style PQ beam: PERMANENTLY DISABLED.
        # V32 (M=64, Ks=256, 0.33 bits/dim): recall=0.7161, QPS=5,326 — beam misdirection.
        # V32b (M=128, Ks=64, 0.50 bits/dim): recall=0.8682, QPS=5,866 — still misdirected.
        # Root cause: PQ noise at any SLC-resident bits/dim (≤1 bit/dim) is sufficient to cause
        # beam to terminate in the wrong graph region for 1536-dim cosine embeddings.
        # ef inflation and sorted re-rank do not help when candidates are never pushed to the heap.
        # PQ infrastructure preserved for future FT.MSEARCH multi-query NEON batching.

    def _register_metal_rerank_buffer(mut self):
        """Wire gpu_rerank_fp32 into the Metal FFI rerank pipeline. Idempotent —
        safe to call after every compact (each variant frees and reallocates the
        underlying buffer, so the pointer changes). No-op when the buffer is
        absent (streaming mode / cold warm-start) or GPU is off."""
        self.gpu_rerank_registered = False
        if not self.has_gpu: return
        if is_null(self.gpu_rerank_fp32): return
        comptime if CompilationTarget.is_macos():
            var _unreg = external_call["pion_metal_unregister_rerank_buffer", Int32]()
            var rc = external_call["pion_metal_register_rerank_buffer", Int32](
                self.gpu_rerank_fp32, UInt32(self.num_nodes), UInt32(self.dim))
            if rc == 0:
                self.gpu_rerank_registered = True

    @always_inline
    def _try_gpu_rerank(
        mut self,
        query: UnsafePointer[Float32, MutUntrackedOrigin],
        ids_i32: UnsafePointer[Int32, MutUntrackedOrigin],
        out_dists: UnsafePointer[Float32, MutUntrackedOrigin],
        k: Int,
        worker_id: Int = 0,
    ) -> Bool:
        """Dispatch the FP32 gather-rerank kernel. Returns True on success
        (caller skips its CPU rerank loop); False to fall through to CPU.
        Threshold-gated: small K is faster on CPU than GPU dispatch."""
        if not self.has_gpu: return False
        if not self.gpu_rerank_registered: return False
        if k < GPU_RERANK_THRESHOLD: return False
        if k > GPU_RERANK_K_MAX: return False
        var success: Bool = False
        comptime if CompilationTarget.is_macos():
            var gate = Int(external_call["pion_metal_should_use_gpu", Int32]())
            if gate == 1:
                var ret = Int(external_call["pion_metal_rerank", Int32](
                    query, ids_i32, UInt32(k), out_dists, UInt32(worker_id)))
                if ret == k: success = True
        return success

    def _build_l0_slots(mut self):
        """K1 beam kernel: derive the
        slot-space mirror of l0_compact. Must run AFTER the final l0_compact of
        a build (compact or snapshot load) and after compact_stride/compact_hdr
        are set. Slots are derived from each neighbor's nodes[].vector pointer,
        so the pass is layout-agnostic across INT8/INT4/polar/turbo/nano compact
        formats. 0xFFFFFFFF marks entries with no compact slot (deleted or
        pointer outside the compact buffer) — the beam gather falls back to the
        nodes[] dereference for those. Build-time only; never on the query path."""
        if is_not_null(self.l0_slots): self.l0_slots.free()
        self.l0_slots = alloc[UInt32](self.num_nodes * 33)
        var base = Int(self.compact_buffer) + self.compact_hdr
        var span = self.num_nodes * self.compact_stride
        for ni in range(self.num_nodes):
            var cnt = Int(self.l0_compact[ni * 33])
            self.l0_slots[ni * 33] = UInt32(cnt)
            for k in range(cnt):
                var nid = Int(self.l0_compact[ni * 33 + 1 + k])
                var slot_v = UInt32(0xFFFFFFFF)
                if nid >= 0 and nid < self.num_nodes and self.compact_stride > 0:
                    var vp = self.nodes[nid].vector
                    if is_not_null(vp):
                        var off = Int(vp) - base
                        if off >= 0 and off < span:
                            slot_v = UInt32(off // self.compact_stride)
                self.l0_slots[ni * 33 + 1 + k] = slot_v

    def compact_vectors(mut self):
        # V2.3: Reorder vectors into a single contiguous buffer in BFS order.
        # 2.5: For INT8, prepend 8-byte norm header to each slot:
        #   [full_norm_sq: Float32 (4)] [prefix_norm_sq: Float32 (4)] [vector: Int8×dim]
        # nodes[idx].vector points 8 bytes into the slot (past the header).
        # Norm access in search: (vptr-8).bitcast[Float32]()[0] = full, (vptr-4) = prefix.
        # Header lands in the same cache line as the vector start — free when prefetched.
        if self.num_nodes == 0 or self.entry_point_id == -1: return

        # N4: Block-wise INT2 compaction (NanoQuant) — takes priority over M7/M6b
        if self.nanoquant and self.fp32_count > 0 and not self.streaming_mode and self.dim == 1536:
            self._compact_vectors_nano2bit()
            return

        # M7: Block-wise INT3 compaction + QJL — takes priority over M6b
        # B2 fix: Skip INT3 compaction in streaming mode (>1M elements).
        # Streaming mode has no FP32 buffer — would need INT8→FP32→INT3 double quantization
        # which destroys recall (10.5% at 5M vs 93.7% at 50K). Fall back to INT8 compact.
        if self.turboquant and self.fp32_count > 0 and not self.streaming_mode and self.dim == 1536:
            self._compact_vectors_turbo3bit()
            return

        # M6b: Block-wise INT4 compaction (SQ4_Block32) — per-block FP16 scales fix recall cliff
        if self.polarquant and is_not_null(self.fp32_buffer) and self.fp32_count > 0 and self.dim == 1536:
            self._compact_vectors_polarquant()
            return

        var bpv = self.dim // 2 if self.use_int4 else self.dim
        var hdr = 8 if not self.use_int4 else 0  # norm header bytes per slot
        # gh #196.2: round the slot stride up to a 64B multiple (1544 → 1600 for
        # INT8-1536) and 64-align the buffer base, so no slot straddles an extra
        # cache line — at stride 1544, 7 of 8 slots did, and every 16B NEON load
        # in the batch kernels was potentially line-crossing. +3.6% memory at 50K.
        # Addressing is compact_stride-driven everywhere (K1 slots, save/load
        # header words 20-21), so pre-pad files keep loading with their own stride.
        var bpv_total = (bpv + hdr + 63) // 64 * 64
        var new_buf = alloc[Int8](self.num_nodes * bpv_total, alignment=64)
        # Zero the pad bytes: they are written to the index file on save (whole
        # stride rows) — keep them deterministic instead of heap garbage.
        unsafe_memset(new_buf, 0, self.num_nodes * bpv_total)
        # §7: GPU slot→external ID mapping (parallel to compact_buffer slots)
        if is_not_null(self.gpu_slot_ext_ids): self.gpu_slot_ext_ids.free()
        self.gpu_slot_ext_ids = alloc[Int32](self.num_nodes)

        # §7v2: Build FP32 re-rank buffer in BFS order for GPU oversample re-ranking.
        # Maps compact_buffer slot i → FP32 vector at gpu_rerank_fp32[i * dim .. (i+1)*dim-1].
        var build_fp32_rerank = self.has_gpu and is_not_null(self.fp32_buffer) and self.fp32_count > 0
        var eid_to_fp32 = null_ptr[Int32, MutUntrackedOrigin]()
        if build_fp32_rerank:
            if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
            self.gpu_rerank_fp32 = alloc[Float32](self.num_nodes * self.dim)
            eid_to_fp32 = alloc[Int32](self.max_elements)
            for i in range(self.max_elements): eid_to_fp32[i] = Int32(-1)
            for i in range(self.fp32_count):
                var eid = Int(self.fp32_ids[i])
                if eid >= 0 and eid < self.max_elements:
                    eid_to_fp32[eid] = Int32(i)

        var queue = List[Int]()
        self._reset_visited()

        queue.append(self.entry_point_id)
        self.visited_epoch[self.entry_point_id] = self.cur_epoch
        var head = 0
        var slot = 0

        while head < len(queue) and slot < self.num_nodes:
            var idx = queue[head]; head += 1
            var slot_start = new_buf + slot * bpv_total
            var src = self.nodes[idx].vector
            # 2.5: Write norms into 8-byte header, then copy vector
            if not self.use_int4:
                var norm_sq: Float32 = 0.0
                var prefix_norm_sq: Float32 = 0.0
                for j in range(self.dim):
                    var vj = Int(src[j])
                    var vj2 = Float32(vj * vj)
                    norm_sq += vj2
                    if j < 256: prefix_norm_sq += vj2
                slot_start.bitcast[Float32]()[0] = norm_sq
                slot_start.bitcast[Float32]()[1] = prefix_norm_sq
            unsafe_memcpy(dest=slot_start + hdr, src=src, count=bpv)
            # V24: Populate inline dimensions for zero-latency early pruning
            for i in range(16):
                self.nodes[idx].inline_vector[i] = src[i]
            self.nodes[idx].vector = slot_start + hdr
            self.gpu_slot_ext_ids[slot] = Int32(self.nodes[idx].id)
            # §7v2: Copy FP32 vector to BFS-ordered rerank buffer
            if build_fp32_rerank:
                var eid = self.nodes[idx].id
                if eid >= 0 and eid < self.max_elements:
                    var fp32_idx = Int(eid_to_fp32[eid])
                    if fp32_idx >= 0:
                        unsafe_memcpy(dest=(self.gpu_rerank_fp32 + slot * self.dim).bitcast[UInt8](),
                               src=(self.fp32_buffer + fp32_idx * self.dim).bitcast[UInt8](),
                               count=self.dim * 4)
            slot += 1
            for l in range(self.nodes[idx].max_level + 1):
                var nc = self.nodes[idx].get_neighbor_count(l)
                for i in range(nc):
                    # Neighbors store INTERNAL indices — use directly, not via node_map.
                    # node_map maps EXTERNAL→INTERNAL; passing internal IDs through it
                    # corrupted vector pointers for sharded builds (internal != external).
                    var nid = self.nodes[idx].get_neighbor(l, i)
                    if nid < 0 or nid >= self.num_nodes: continue
                    if self.is_deleted(nid): continue
                    if (self.visited_epoch[nid] == self.cur_epoch): continue
                    self.visited_epoch[nid] = self.cur_epoch
                    queue.append(nid)

        # Safety: handle any nodes not reached by BFS (disconnected components)
        for i in range(self.num_nodes):
            if self.is_deleted(i): continue
            if (self.visited_epoch[i] != self.cur_epoch) and slot < self.num_nodes:
                var src = self.nodes[i].vector
                var slot_start = new_buf + slot * bpv_total
                if not self.use_int4:
                    var norm_sq: Float32 = 0.0
                    var prefix_norm_sq: Float32 = 0.0
                    for j in range(self.dim):
                        var vj = Int(src[j])
                        var vj2 = Float32(vj * vj)
                        norm_sq += vj2
                        if j < 256: prefix_norm_sq += vj2
                    slot_start.bitcast[Float32]()[0] = norm_sq
                    slot_start.bitcast[Float32]()[1] = prefix_norm_sq
                unsafe_memcpy(dest=slot_start + hdr, src=src, count=bpv)
                self.nodes[i].vector = slot_start + hdr
                self.gpu_slot_ext_ids[slot] = Int32(self.nodes[i].id)
                # §7v2: Copy FP32 for disconnected node
                if build_fp32_rerank:
                    var eid = self.nodes[i].id
                    if eid >= 0 and eid < self.max_elements:
                        var fp32_idx = Int(eid_to_fp32[eid])
                        if fp32_idx >= 0:
                            unsafe_memcpy(dest=(self.gpu_rerank_fp32 + slot * self.dim).bitcast[UInt8](),
                                   src=(self.fp32_buffer + fp32_idx * self.dim).bitcast[UInt8](),
                                   count=self.dim * 4)
                slot += 1

        if is_not_null(eid_to_fp32): eid_to_fp32.free()
        if build_fp32_rerank:
            print("[GPU] Built FP32 re-rank buffer: " + String(self.num_nodes) + " vectors (" + String(self.num_nodes * self.dim * 4 // 1048576) + " MB)")

        # Null out deleted nodes' vector pointers before reclaiming slab memory.
        # Deleted nodes were skipped by BFS + safety loop, so their .vector still
        # points into the slab allocator. After reset(), those pointers dangle.
        # Nulling prevents SEGV if search accidentally traverses a deleted neighbor
        # (e.g. via stale upper-level links or l0_compact entries).
        for di in range(self.num_nodes):
            if self.is_deleted(di):
                self.nodes[di].vector = null_ptr[Int8, MutUntrackedOrigin]()
        # Reclaim slab allocator memory — all vectors now live in compact_buffer
        self.vector_allocator.reset()
        self.compact_buffer = new_buf
        self.compact_is_int4 = self.use_int4
        if is_not_null(self.node_norms): self.node_norms.free()
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_prefix_norms): self.node_prefix_norms.free()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()
        # V33: build l0_compact — 33 UInt32/node = count + up to 32 level-0 neighbor IDs
        # 6.6MB total vs 61MB neighbor_pool; hot nodes fit in SLC → fewer DRAM misses per beam step
        if is_not_null(self.l0_compact): self.l0_compact.free()
        self.l0_compact = alloc[UInt32](self.num_nodes * 33)
        for ni in range(self.num_nodes):
            if self.is_deleted(ni):
                self.l0_compact[ni * 33] = UInt32(0)
                continue
            var count = self.nodes[ni].get_neighbor_count(0)
            # A1b: Filter out deleted neighbors from l0_compact
            var actual_count = 0
            for i in range(count):
                var nid = self.nodes[ni].get_neighbor(0, i)
                if nid >= 0 and nid < self.num_nodes and not self.is_deleted(nid):
                    self.l0_compact[ni * 33 + 1 + actual_count] = UInt32(nid)
                    actual_count += 1
            self.l0_compact[ni * 33] = UInt32(actual_count)
        # V34: build prefix_buffer — 256-byte prefix of each node, indexed by nidx.
        # 12.8MB vs 77MB compact_buffer → much higher SLC hit rate in prune phase.
        if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
        self.prefix_buffer = alloc[Int8](self.num_nodes * 256)
        # A vector shorter than 256 bytes (dim < 256: ATTEND's 128-d graphs)
        # copies what it has and zero-fills; copying 256 read past the end of
        # compact_buffer on the last slot.
        var pre = min(bpv, 256)
        for nidx in range(self.num_nodes):
            if self.is_deleted(nidx) or pre < 256:
                unsafe_memset(self.prefix_buffer + nidx * 256, 0, 256)
                if self.is_deleted(nidx): continue
            unsafe_memcpy(dest=self.prefix_buffer + nidx * 256, src=self.nodes[nidx].vector, count=pre)
        # K1: slot-space adjacency mirror for arithmetic vector addressing.
        self.compact_stride = bpv_total
        self.compact_hdr = hdr
        self._build_l0_slots()

        # §7: Register compact_buffer with Metal GPU engine (if --gpu active)
        if self.has_gpu:
            comptime if CompilationTarget.is_macos():
                var gpu_avail = external_call["pion_metal_available", Int32]()
                if gpu_avail == 1:
                    var rc = external_call["pion_metal_register_vectors", Int32](
                        self.compact_buffer, UInt32(self.num_nodes),
                        UInt32(self.compact_stride))
                    if rc == 0:
                        print("[GPU] Registered " + String(self.num_nodes) + " vectors with Metal")
            # §7v2: Register with native Mojo GPU context (zero ObjC overhead path)
            try:
                self.gpu_ctx.register_vectors(self.compact_buffer.bitcast[Int8](), self.num_nodes,
                                              self.compact_stride)
                print("[GPU] Native Mojo GPU context registered " + String(self.num_nodes) + " vectors")
            except:
                print("[GPU] Native Mojo GPU context init failed — Metal FFI fallback active")

        # FP32 rerank buffer (BFS-ordered) → Metal gather-rerank pipeline.
        # Pointer changes on every compact, so unregister-then-register here.
        self._register_metal_rerank_buffer()

    def _compact_vectors_polarquant(mut self):
        """M6b: Block-wise INT4 compaction (SQ4_Block32).
        Each vector → 48 blocks × 32 dims, each with FP16 scale + 16B INT4 data.
        Layout: [FP32 norm (4B)] [FP16 scale (2B) + INT4 packed (16B)] × 48 = 868B/vector.
        50K × 868B = 43.4 MB (vs 77MB INT8 = −44%). No prefix/suffix split — uniform block loop."""
        # Build eid → fp32_buffer index mapping
        var eid_to_fp32 = alloc[Int32](self.max_elements)
        for i in range(self.max_elements): eid_to_fp32[i] = Int32(-1)
        for i in range(self.fp32_count):
            var eid = Int(self.fp32_ids[i])
            if eid >= 0 and eid < self.max_elements:
                eid_to_fp32[eid] = Int32(i)

        # Allocate block-INT4 compact buffer
        var bpv = VEC_BYTES_1536  # 868 bytes per vector
        var new_buf = alloc[Int8](self.num_nodes * bpv)
        if is_not_null(self.gpu_slot_ext_ids): self.gpu_slot_ext_ids.free()
        self.gpu_slot_ext_ids = alloc[Int32](self.num_nodes)

        # BFS traverse — quantize each node's FP32 → block INT4
        var queue = List[Int]()
        self._reset_visited()
        queue.append(self.entry_point_id)
        self.visited_epoch[self.entry_point_id] = self.cur_epoch
        var head_ = 0
        var slot = 0

        while head_ < len(queue) and slot < self.num_nodes:
            var idx = queue[head_]; head_ += 1
            var eid = self.nodes[idx].id
            var slot_start = new_buf + slot * bpv

            var fp32_idx = Int(-1)
            if eid >= 0 and eid < self.max_elements:
                fp32_idx = Int(eid_to_fp32[eid])

            if fp32_idx >= 0:
                quantize_fp32_to_block_int4(
                    self.fp32_buffer + fp32_idx * self.dim, slot_start, self.dim)
            else:
                unsafe_memset(slot_start, 0, bpv)

            self.nodes[idx].vector = slot_start
            self.gpu_slot_ext_ids[slot] = Int32(eid)
            slot += 1
            for l in range(self.nodes[idx].max_level + 1):
                var nc = self.nodes[idx].get_neighbor_count(l)
                for i in range(nc):
                    # Neighbors store INTERNAL indices — use directly, not via node_map.
                    # node_map maps EXTERNAL→INTERNAL; passing internal IDs through it
                    # corrupted vector pointers for sharded builds (internal != external).
                    var nid = self.nodes[idx].get_neighbor(l, i)
                    if nid < 0 or nid >= self.num_nodes: continue
                    if self.is_deleted(nid): continue
                    if (self.visited_epoch[nid] == self.cur_epoch): continue
                    self.visited_epoch[nid] = self.cur_epoch
                    queue.append(nid)

        # Handle disconnected nodes
        for i in range(self.num_nodes):
            if (self.visited_epoch[i] != self.cur_epoch) and slot < self.num_nodes:
                var eid = self.nodes[i].id
                var slot_start = new_buf + slot * bpv
                var fp32_idx = Int(-1)
                if eid >= 0 and eid < self.max_elements:
                    fp32_idx = Int(eid_to_fp32[eid])
                if fp32_idx >= 0:
                    quantize_fp32_to_block_int4(
                        self.fp32_buffer + fp32_idx * self.dim, slot_start, self.dim)
                else:
                    unsafe_memset(slot_start, 0, bpv)
                self.nodes[i].vector = slot_start
                self.gpu_slot_ext_ids[slot] = Int32(eid)
                slot += 1

        # Build FP32 re-rank buffer (BFS-ordered, for post-beam-search re-ranking)
        if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
        self.gpu_rerank_fp32 = alloc[Float32](self.num_nodes * self.dim)
        for nidx in range(self.num_nodes):
            var eid = self.nodes[nidx].id
            var fp32_idx = Int(-1)
            if eid >= 0 and eid < self.max_elements:
                fp32_idx = Int(eid_to_fp32[eid])
            if fp32_idx >= 0:
                unsafe_memcpy(dest=(self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](),
                       src=(self.fp32_buffer + fp32_idx * self.dim).bitcast[UInt8](),
                       count=self.dim * 4)
            else:
                unsafe_memset((self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](), 0, self.dim * 4)
        var rerank_mb = self.num_nodes * self.dim * 4 // 1048576
        print("[PolarQuant] FP32 re-rank buffer: " + String(rerank_mb) + " MB")

        eid_to_fp32.free()
        for di in range(self.num_nodes):
            if self.is_deleted(di):
                self.nodes[di].vector = null_ptr[Int8, MutUntrackedOrigin]()
        self.vector_allocator.reset()
        self.compact_buffer = new_buf
        self.compact_is_int4 = True  # signals block-INT4 format

        # Build l0_compact (same as standard path)
        if is_not_null(self.l0_compact): self.l0_compact.free()
        self.l0_compact = alloc[UInt32](self.num_nodes * 33)
        for ni in range(self.num_nodes):
            if self.is_deleted(ni):
                self.l0_compact[ni * 33] = UInt32(0)
                continue
            var count = self.nodes[ni].get_neighbor_count(0)
            var actual_count = 0
            for i in range(count):
                var nid = self.nodes[ni].get_neighbor(0, i)
                if nid >= 0 and nid < self.num_nodes and not self.is_deleted(nid):
                    self.l0_compact[ni * 33 + 1 + actual_count] = UInt32(nid)
                    actual_count += 1
            self.l0_compact[ni * 33] = UInt32(actual_count)

        # K1: slot-space adjacency mirror (vector ptr = slot start, no header).
        self.compact_stride = bpv
        self.compact_hdr = 0
        self._build_l0_slots()

        # No separate prefix_buffer — block-wise search uses unified compact_buffer
        if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.node_norms): self.node_norms.free()
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_prefix_norms): self.node_prefix_norms.free()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()

        var compact_mb = self.num_nodes * bpv // 1048576
        print("[PolarQuant] Block-INT4 compact buffer: " + String(self.num_nodes) + " vectors × " + String(bpv) + "B = " + String(compact_mb) + " MB (target: ~43MB)")
        self._register_metal_rerank_buffer()

    def _compact_vectors_turbo3bit(mut self):
        """M7 TurboQuant: Block-wise INT3 compaction + QJL error correction.
        Each vector → 48 blocks × 32 dims, each with FP16 scale + 12B 3-bit packed data.
        Layout: [FP32 norm (4B)] [FP16 scale (2B) + 3-bit packed (12B)] × 48 = 676B/vector.
        QJL: 192B/vector sign bits in separate buffer (WHT(random_signs ⊙ residual) → sign bits).
        50K × 676B = 33.8 MB compact + 9.6 MB QJL = 43.4 MB total (vs 77MB INT8 = −44%)."""
        # Build eid → fp32_buffer index mapping
        var eid_to_fp32 = alloc[Int32](self.max_elements)
        for i in range(self.max_elements): eid_to_fp32[i] = Int32(-1)
        for i in range(self.fp32_count):
            var eid = Int(self.fp32_ids[i])
            if eid >= 0 and eid < self.max_elements:
                eid_to_fp32[eid] = Int32(i)

        # Allocate block-INT3 compact buffer
        var bpv = INT3_VEC_BYTES_1536  # 676 bytes per vector
        var new_buf = alloc[Int8](self.num_nodes * bpv)
        if is_not_null(self.gpu_slot_ext_ids): self.gpu_slot_ext_ids.free()
        self.gpu_slot_ext_ids = alloc[Int32](self.num_nodes)

        # Allocate QJL buffers
        if is_not_null(self.qjl_buffer): self.qjl_buffer.free()
        self.qjl_buffer = alloc[UInt64](self.num_nodes * QJL_U64S_1536)
        if is_not_null(self.qjl_res_norms): self.qjl_res_norms.free()
        self.qjl_res_norms = alloc[Float32](self.num_nodes)

        # Temporary buffers for QJL residual computation and INT8→FP32 reconstruction
        var residual_buf = alloc[Float32](self.dim)
        var wht_tmp = alloc[Float32](self.dim)
        var recon_fp32 = alloc[Float32](self.dim)  # Streaming mode: dequantized INT8→FP32
        var range_val = self.global_max - self.global_min
        var inv_scale = range_val / 255.0

        # BFS traverse — quantize each node's FP32 → block INT3 + compute QJL signs
        var queue = List[Int]()
        self._reset_visited()
        queue.append(self.entry_point_id)
        self.visited_epoch[self.entry_point_id] = self.cur_epoch
        var head_ = 0
        var slot = 0

        while head_ < len(queue) and slot < self.num_nodes:
            var idx = queue[head_]; head_ += 1
            var eid = self.nodes[idx].id
            var slot_start = new_buf + slot * bpv

            # Get FP32 source: from fp32_buffer (buffered mode) or reconstruct from INT8 (streaming mode)
            var src_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
            var fp32_idx = Int(-1)
            if is_not_null(self.fp32_buffer) and not self.streaming_mode:
                if eid >= 0 and eid < self.max_elements:
                    fp32_idx = Int(eid_to_fp32[eid])
                if fp32_idx >= 0:
                    src_fp32 = self.fp32_buffer + fp32_idx * self.dim
            elif is_not_null(self.nodes[idx].vector):
                # Streaming mode: dequantize INT8 → FP32 from vector_allocator
                var int8_src = self.nodes[idx].vector
                for d in range(self.dim):
                    recon_fp32[d] = Float32(Int(int8_src[d]) + 128) * inv_scale + self.global_min
                src_fp32 = recon_fp32

            if is_not_null(src_fp32):
                quantize_fp32_to_block_int3(src_fp32, slot_start, self.dim)
                # Compute QJL: residual = fp32 - dequantized(quantized)
                var q_off = 4
                for b in range(NUM_BLOCKS_1536):
                    var base = b * BLOCK_DIM
                    var scale_v = (slot_start + q_off).bitcast[Float16]()[0].cast[DType.float32]()
                    q_off += 2
                    for g in range(4):
                        var byte_off = q_off + g * 3
                        var u = _byte_u64(slot_start, byte_off) | (_byte_u64(slot_start, byte_off + 1) << 8) | (_byte_u64(slot_start, byte_off + 2) << 16)
                        for vi in range(8):
                            var q_val = Int((u >> UInt64(vi * 3)) & UInt64(7)) - 4
                            var dim_idx = base + g * 8 + vi
                            if g >= 2: dim_idx = base + 16 + (g - 2) * 8 + vi
                            residual_buf[dim_idx] = src_fp32[dim_idx] - Float32(q_val) * scale_v
                    q_off += 12
                # QJL rows are indexed by NODE, not slot: the beam reads
                # qjl_buffer[nidx] with nidx a node index (l0_compact ids).
                # Writing by BFS slot paired every node with another node's
                # residual signs.
                var qjl_dst = self.qjl_buffer + idx * QJL_U64S_1536
                self.qjl_res_norms[idx] = qjl_compute_signs(
                    residual_buf, self.qjl_random_signs, wht_tmp, qjl_dst, self.dim)
            else:
                unsafe_memset(slot_start, 0, bpv)
                unsafe_memset((self.qjl_buffer + idx * QJL_U64S_1536).bitcast[UInt8](), 0, QJL_BYTES_1536)
                self.qjl_res_norms[idx] = Float32(0.0)

            self.nodes[idx].vector = slot_start
            self.gpu_slot_ext_ids[slot] = Int32(eid)
            slot += 1
            for l in range(self.nodes[idx].max_level + 1):
                var nc = self.nodes[idx].get_neighbor_count(l)
                for i in range(nc):
                    # Neighbors store INTERNAL indices — use directly, not via node_map.
                    # node_map maps EXTERNAL→INTERNAL; passing internal IDs through it
                    # corrupted vector pointers for sharded builds (internal != external).
                    var nid = self.nodes[idx].get_neighbor(l, i)
                    if nid < 0 or nid >= self.num_nodes: continue
                    if self.is_deleted(nid): continue
                    if (self.visited_epoch[nid] == self.cur_epoch): continue
                    self.visited_epoch[nid] = self.cur_epoch
                    queue.append(nid)

        # Handle disconnected nodes
        for i in range(self.num_nodes):
            if (self.visited_epoch[i] != self.cur_epoch) and slot < self.num_nodes:
                var eid = self.nodes[i].id
                var slot_start = new_buf + slot * bpv

                # Get FP32 source for disconnected node
                var dc_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
                if is_not_null(self.fp32_buffer) and not self.streaming_mode:
                    var fp32_idx = Int(-1)
                    if eid >= 0 and eid < self.max_elements:
                        fp32_idx = Int(eid_to_fp32[eid])
                    if fp32_idx >= 0:
                        dc_fp32 = self.fp32_buffer + fp32_idx * self.dim
                elif is_not_null(self.nodes[i].vector):
                    var int8_src = self.nodes[i].vector
                    for d in range(self.dim):
                        recon_fp32[d] = Float32(Int(int8_src[d]) + 128) * inv_scale + self.global_min
                    dc_fp32 = recon_fp32

                if is_not_null(dc_fp32):
                    quantize_fp32_to_block_int3(dc_fp32, slot_start, self.dim)
                    var q_off = 4
                    for b in range(NUM_BLOCKS_1536):
                        var base = b * BLOCK_DIM
                        var scale_v = (slot_start + q_off).bitcast[Float16]()[0].cast[DType.float32]()
                        q_off += 2
                        for g in range(4):
                            var byte_off = q_off + g * 3
                            var u = _byte_u64(slot_start, byte_off) | (_byte_u64(slot_start, byte_off + 1) << 8) | (_byte_u64(slot_start, byte_off + 2) << 16)
                            for vi in range(8):
                                var q_val = Int((u >> UInt64(vi * 3)) & UInt64(7)) - 4
                                var dim_idx = base + g * 8 + vi
                                if g >= 2: dim_idx = base + 16 + (g - 2) * 8 + vi
                                residual_buf[dim_idx] = dc_fp32[dim_idx] - Float32(q_val) * scale_v
                        q_off += 12
                    var qjl_dst = self.qjl_buffer + i * QJL_U64S_1536
                    self.qjl_res_norms[i] = qjl_compute_signs(
                        residual_buf, self.qjl_random_signs, wht_tmp, qjl_dst, self.dim)
                else:
                    unsafe_memset(slot_start, 0, bpv)
                    unsafe_memset((self.qjl_buffer + i * QJL_U64S_1536).bitcast[UInt8](), 0, QJL_BYTES_1536)
                    self.qjl_res_norms[i] = Float32(0.0)
                self.nodes[i].vector = slot_start
                self.gpu_slot_ext_ids[slot] = Int32(eid)
                slot += 1

        residual_buf.free()
        wht_tmp.free()
        recon_fp32.free()

        # Build FP32 re-rank buffer — skip in streaming mode (too expensive for large datasets)
        if not self.streaming_mode:
            if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
            self.gpu_rerank_fp32 = alloc[Float32](self.num_nodes * self.dim)
            for nidx in range(self.num_nodes):
                var eid = self.nodes[nidx].id
                var fp32_idx = Int(-1)
                if eid >= 0 and eid < self.max_elements:
                    fp32_idx = Int(eid_to_fp32[eid])
                if fp32_idx >= 0:
                    unsafe_memcpy(dest=(self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](),
                           src=(self.fp32_buffer + fp32_idx * self.dim).bitcast[UInt8](),
                           count=self.dim * 4)
                else:
                    unsafe_memset((self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](), 0, self.dim * 4)
            var rerank_mb = self.num_nodes * self.dim * 4 // 1048576
            print("[TurboQuant] FP32 re-rank buffer: " + String(rerank_mb) + " MB")

        eid_to_fp32.free()
        for di in range(self.num_nodes):
            if self.is_deleted(di):
                self.nodes[di].vector = null_ptr[Int8, MutUntrackedOrigin]()
        self.vector_allocator.reset()
        self.compact_buffer = new_buf
        self.compact_is_3bit = True
        self.compact_is_int4 = False

        # Build l0_compact (same as standard path)
        if is_not_null(self.l0_compact): self.l0_compact.free()
        self.l0_compact = alloc[UInt32](self.num_nodes * 33)
        for ni in range(self.num_nodes):
            if self.is_deleted(ni):
                self.l0_compact[ni * 33] = UInt32(0)
                continue
            var count = self.nodes[ni].get_neighbor_count(0)
            var actual_count = 0
            for i in range(count):
                var nid = self.nodes[ni].get_neighbor(0, i)
                if nid >= 0 and nid < self.num_nodes and not self.is_deleted(nid):
                    self.l0_compact[ni * 33 + 1 + actual_count] = UInt32(nid)
                    actual_count += 1
            self.l0_compact[ni * 33] = UInt32(actual_count)

        # K1: slot-space adjacency mirror (vector ptr = slot start, no header).
        self.compact_stride = bpv
        self.compact_hdr = 0
        self._build_l0_slots()

        # Clear unused buffers
        if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.node_norms): self.node_norms.free()
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_prefix_norms): self.node_prefix_norms.free()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()

        var compact_mb = self.num_nodes * bpv // 1048576
        var qjl_mb = self.num_nodes * QJL_BYTES_1536 // 1048576
        var rerank_mb2 = 0
        if not self.streaming_mode:
            rerank_mb2 = self.num_nodes * self.dim * 4 // 1048576
        print("[TurboQuant] Block-INT3 compact buffer: " + String(self.num_nodes) + " vectors × " + String(bpv) + "B = " + String(compact_mb) + " MB")
        print("[TurboQuant] QJL sign buffer: " + String(qjl_mb) + " MB (" + String(QJL_BYTES_1536) + "B/vec)")
        print("[TurboQuant] Total working set: " + String(compact_mb + qjl_mb + rerank_mb2) + " MB")
        self._register_metal_rerank_buffer()

    def _compact_vectors_nano2bit(mut self):
        """N4 NanoQuant: Block-wise INT2 compaction (no QJL).
        Each vector → 48 blocks × 32 dims, each with FP16 scale + 8B 2-bit packed data.
        Layout: [FP32 norm (4B)] [FP16 scale (2B) + 2-bit packed (8B)] × 48 = 484B/vector.
        50K × 484B = 24.2 MB compact (vs 33.8 MB INT3, 43.4 MB INT4)."""
        # Build eid → fp32_buffer index mapping
        var eid_to_fp32 = alloc[Int32](self.max_elements)
        for i in range(self.max_elements): eid_to_fp32[i] = Int32(-1)
        for i in range(self.fp32_count):
            var eid = Int(self.fp32_ids[i])
            if eid >= 0 and eid < self.max_elements:
                eid_to_fp32[eid] = Int32(i)

        # Allocate block-INT2 compact buffer
        var bpv = INT2_VEC_BYTES_1536  # 484 bytes per vector
        var new_buf = alloc[Int8](self.num_nodes * bpv)
        if is_not_null(self.gpu_slot_ext_ids): self.gpu_slot_ext_ids.free()
        self.gpu_slot_ext_ids = alloc[Int32](self.num_nodes)

        # Temporary buffer for INT8→FP32 reconstruction (streaming mode)
        var recon_fp32 = alloc[Float32](self.dim)
        var range_val = self.global_max - self.global_min
        var inv_scale = range_val / 255.0

        # BFS traverse — quantize each node's FP32 → block INT2
        var queue = List[Int]()
        self._reset_visited()
        queue.append(self.entry_point_id)
        self.visited_epoch[self.entry_point_id] = self.cur_epoch
        var head_ = 0
        var slot = 0

        while head_ < len(queue) and slot < self.num_nodes:
            var idx = queue[head_]; head_ += 1
            var eid = self.nodes[idx].id
            var slot_start = new_buf + slot * bpv

            # Get FP32 source: from fp32_buffer (buffered mode) or reconstruct from INT8 (streaming mode)
            var src_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
            var fp32_idx = Int(-1)
            if is_not_null(self.fp32_buffer) and not self.streaming_mode:
                if eid >= 0 and eid < self.max_elements:
                    fp32_idx = Int(eid_to_fp32[eid])
                if fp32_idx >= 0:
                    src_fp32 = self.fp32_buffer + fp32_idx * self.dim
            elif is_not_null(self.nodes[idx].vector):
                var int8_src = self.nodes[idx].vector
                for d in range(self.dim):
                    recon_fp32[d] = Float32(Int(int8_src[d]) + 128) * inv_scale + self.global_min
                src_fp32 = recon_fp32

            if is_not_null(src_fp32):
                quantize_fp32_to_block_int2(src_fp32, slot_start, self.dim)
            else:
                unsafe_memset(slot_start, 0, bpv)

            self.nodes[idx].vector = slot_start
            self.gpu_slot_ext_ids[slot] = Int32(eid)
            slot += 1
            for l in range(self.nodes[idx].max_level + 1):
                var nc = self.nodes[idx].get_neighbor_count(l)
                for i in range(nc):
                    # Neighbors store INTERNAL indices — use directly, not via node_map.
                    # node_map maps EXTERNAL→INTERNAL; passing internal IDs through it
                    # corrupted vector pointers for sharded builds (internal != external).
                    var nid = self.nodes[idx].get_neighbor(l, i)
                    if nid < 0 or nid >= self.num_nodes: continue
                    if self.is_deleted(nid): continue
                    if (self.visited_epoch[nid] == self.cur_epoch): continue
                    self.visited_epoch[nid] = self.cur_epoch
                    queue.append(nid)

        # Handle disconnected nodes
        for i in range(self.num_nodes):
            if (self.visited_epoch[i] != self.cur_epoch) and slot < self.num_nodes:
                var eid = self.nodes[i].id
                var slot_start = new_buf + slot * bpv

                var dc_fp32 = null_ptr[Float32, MutUntrackedOrigin]()
                if is_not_null(self.fp32_buffer) and not self.streaming_mode:
                    var fp32_idx2 = Int(-1)
                    if eid >= 0 and eid < self.max_elements:
                        fp32_idx2 = Int(eid_to_fp32[eid])
                    if fp32_idx2 >= 0:
                        dc_fp32 = self.fp32_buffer + fp32_idx2 * self.dim
                elif is_not_null(self.nodes[i].vector):
                    var int8_src = self.nodes[i].vector
                    for d in range(self.dim):
                        recon_fp32[d] = Float32(Int(int8_src[d]) + 128) * inv_scale + self.global_min
                    dc_fp32 = recon_fp32

                if is_not_null(dc_fp32):
                    quantize_fp32_to_block_int2(dc_fp32, slot_start, self.dim)
                else:
                    unsafe_memset(slot_start, 0, bpv)
                self.nodes[i].vector = slot_start
                self.gpu_slot_ext_ids[slot] = Int32(eid)
                slot += 1

        recon_fp32.free()

        # Build FP32 re-rank buffer
        if not self.streaming_mode:
            if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
            self.gpu_rerank_fp32 = alloc[Float32](self.num_nodes * self.dim)
            for nidx in range(self.num_nodes):
                var eid = self.nodes[nidx].id
                var fp32_idx3 = Int(-1)
                if eid >= 0 and eid < self.max_elements:
                    fp32_idx3 = Int(eid_to_fp32[eid])
                if fp32_idx3 >= 0:
                    unsafe_memcpy(dest=(self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](),
                           src=(self.fp32_buffer + fp32_idx3 * self.dim).bitcast[UInt8](),
                           count=self.dim * 4)
                else:
                    unsafe_memset((self.gpu_rerank_fp32 + nidx * self.dim).bitcast[UInt8](), 0, self.dim * 4)
            var rerank_mb = self.num_nodes * self.dim * 4 // 1048576
            print("[NanoQuant] FP32 re-rank buffer: " + String(rerank_mb) + " MB")

        eid_to_fp32.free()
        for di in range(self.num_nodes):
            if self.is_deleted(di):
                self.nodes[di].vector = null_ptr[Int8, MutUntrackedOrigin]()
        self.vector_allocator.reset()
        self.compact_buffer = new_buf
        self.compact_is_2bit = True
        self.compact_is_3bit = False
        self.compact_is_int4 = False

        # Build l0_compact (same as standard path)
        if is_not_null(self.l0_compact): self.l0_compact.free()
        self.l0_compact = alloc[UInt32](self.num_nodes * 33)
        for ni in range(self.num_nodes):
            if self.is_deleted(ni):
                self.l0_compact[ni * 33] = UInt32(0)
                continue
            var count = self.nodes[ni].get_neighbor_count(0)
            var actual_count = 0
            for i in range(count):
                var nid = self.nodes[ni].get_neighbor(0, i)
                if nid >= 0 and nid < self.num_nodes and not self.is_deleted(nid):
                    self.l0_compact[ni * 33 + 1 + actual_count] = UInt32(nid)
                    actual_count += 1
            self.l0_compact[ni * 33] = UInt32(actual_count)

        # K1: slot-space adjacency mirror (vector ptr = slot start, no header).
        self.compact_stride = bpv
        self.compact_hdr = 0
        self._build_l0_slots()

        # Clear unused buffers
        if is_not_null(self.prefix_buffer): self.prefix_buffer.free()
        self.prefix_buffer = null_ptr[Int8, MutUntrackedOrigin]()
        if is_not_null(self.node_norms): self.node_norms.free()
        self.node_norms = null_ptr[Float32, MutUntrackedOrigin]()
        if is_not_null(self.node_prefix_norms): self.node_prefix_norms.free()
        self.node_prefix_norms = null_ptr[Float32, MutUntrackedOrigin]()

        var compact_mb = self.num_nodes * bpv // 1048576
        var rerank_mb2 = 0
        if not self.streaming_mode:
            rerank_mb2 = self.num_nodes * self.dim * 4 // 1048576
        print("[NanoQuant] Block-INT2 compact buffer: " + String(self.num_nodes) + " vectors × " + String(bpv) + "B = " + String(compact_mb) + " MB")
        print("[NanoQuant] Total working set: " + String(compact_mb + rerank_mb2) + " MB (no QJL)")
        self._register_metal_rerank_buffer()

    @always_inline
    def _fast_passes_filters(
        self,
        nidx: Int,
        filter_count: Int,
        filter_schema_slots: Array[Int, 4],
        filter_types: Array[UInt8, 4],
        filter_tag_hashes: Array[UInt32, 4],
        filter_lo: Array[Float32, 4],
        filter_hi: Array[Float32, 4],
    ) -> Bool:
        """Phase 3.1: O(1) per-node filter check using pre-computed metadata arrays.
        Returns True if all filters pass. Falls through to True if metadata not built
        (caller then runs _passes_filters keyspace lookup as fallback)."""
        for fi in range(filter_count):
            var slot = filter_schema_slots[fi]
            if slot < 0 or slot >= 8: return False
            var meta_idx = nidx * 8 + slot
            var field_set = self.node_meta_field_set[meta_idx]
            if field_set == 0: return False  # field not in this doc
            if filter_types[fi] == 0:  # TAG_EQ: compare FNV-1a hash
                if self.node_meta_tag_hashes[meta_idx] != filter_tag_hashes[fi]: return False
            else:  # NUMERIC_RANGE
                var v = self.node_meta_numerics[meta_idx]
                if v < filter_lo[fi] or v > filter_hi[fi]: return False
        return True

    def _prune_l0_mrng(mut self):
        """V38E: NSG-style MRNG pruning of l0_compact L0 edges.
        For each node u, sort neighbors by dist(u,v), then keep v only if no
        already-kept neighbor w has dist(w,v) < dist(u,v).
        Minimum 4 edges kept per node to preserve graph connectivity.
        Operates entirely on l0_compact — upper-layer neighbors unchanged."""
        var tmp_ids   = Array[Int, 33](uninitialized=True)
        var tmp_dists = Array[Float32, 33](uninitialized=True)
        var kept_ids  = Array[Int, 33](uninitialized=True)

        for ni in range(self.num_nodes):
            var count = Int(self.l0_compact[ni * 33])
            if count <= 1: continue
            var u_vec = self.nodes[ni].vector

            # Load neighbors and compute dist(u, v) for each
            for i in range(count):
                var vid = Int(self.l0_compact[ni * 33 + 1 + i])
                tmp_ids[i] = vid
                tmp_dists[i] = self._dist_int8_int8(u_vec, self.nodes[vid].vector)

            # Insertion-sort by distance ascending (M≤32, so O(M²) is fine)
            for i in range(1, count):
                var key_id   = tmp_ids[i]
                var key_dist = tmp_dists[i]
                var j = i - 1
                while j >= 0 and tmp_dists[j] > key_dist:
                    tmp_ids[j + 1]   = tmp_ids[j]
                    tmp_dists[j + 1] = tmp_dists[j]
                    j -= 1
                tmp_ids[j + 1]   = key_id
                tmp_dists[j + 1] = key_dist

            # MRNG rule: keep v if no already-kept w has dist(w,v) < dist(u,v)
            var kept_count = 0
            for i in range(count):
                var vi     = tmp_ids[i]
                var d_u_vi = tmp_dists[i]
                var dominated = False
                for j in range(kept_count):
                    var d_w_vi = self._dist_int8_int8(
                        self.nodes[kept_ids[j]].vector, self.nodes[vi].vector)
                    if d_w_vi < d_u_vi:
                        dominated = True
                        break
                if not dominated:
                    kept_ids[kept_count] = vi
                    kept_count += 1

            # Guarantee minimum connectivity: keep at least 4 even if pruned
            if kept_count < 4:
                var extra = 0
                for i in range(count):
                    if extra >= 4 - kept_count: break
                    var vi = tmp_ids[i]
                    var already = False
                    for j in range(kept_count):
                        if kept_ids[j] == vi: already = True; break
                    if not already:
                        kept_ids[kept_count] = vi
                        kept_count += 1
                        extra += 1

            # Write pruned list back
            self.l0_compact[ni * 33] = UInt32(kept_count)
            for i in range(kept_count):
                self.l0_compact[ni * 33 + 1 + i] = UInt32(kept_ids[i])

    def _prune_neighbors(self, query: UnsafePointer[Int8, MutUntrackedOrigin], all_neighbors: List[HeapNode], m_limit: Int) -> List[Int]:
        var results = List[Int]()
        var discarded = List[Int]()

        for i in range(len(all_neighbors)-1, -1, -1):
            if len(results) >= m_limit:
                break
            var candidate_id = all_neighbors[i].id
            var candidate_dist = all_neighbors[i].distance
            var candidate_vec = self.nodes[candidate_id].vector

            var good = True
            for j in range(len(results)):
                var r_id = results[j]
                var r_vec = self.nodes[r_id].vector
                var dist_to_r = self._dist_int8_int8(candidate_vec, r_vec)
                if dist_to_r < candidate_dist:
                    good = False
                    break

            if good:
                results.append(candidate_id)
            else:
                discarded.append(candidate_id)

        # Keep pruned connections if we haven't reached m_limit
        var idx = 0
        while len(results) < m_limit and idx < len(discarded):
            results.append(discarded[idx])
            idx += 1

        return results^

    @always_inline
    def _reset_visited(mut self):
        # gh #118: O(1) per query — just advance the epoch. The full memset only runs
        # when the UInt16 counter wraps (every 65535 queries), amortizing to nothing.
        self.cur_epoch += 1
        if self.cur_epoch == 0:
            unsafe_memset(self.visited_epoch.bitcast[UInt8](), 0, self.max_elements * 2)
            self.cur_epoch = 1

    def unlink_node(mut self, internal_id: Int):
        """A1b: Remove a node from the HNSW graph — clear its neighbor lists, remove backlinks,
        mark as deleted, replace entry point if needed, and reconnect orphaned neighbors."""
        if internal_id < 0 or internal_id >= self.num_nodes: return
        if self.is_deleted(internal_id): return
        var node = self.nodes + internal_id

        # 1. Collect neighbors per level BEFORE clearing (needed for reconnection)
        var max_lvl = node[].max_level
        # Use a flat array: orphan_levels[l][j] stored as orphan_ids[l * 64 + j], orphan_counts[l]
        var orphan_ids = alloc[Int](7 * 64)  # max 7 levels × max 64 neighbors (2*M=32 at L0)
        var orphan_counts = alloc[Int](7)
        for l in range(7):
            orphan_counts[l] = 0

        for l in range(max_lvl + 1):
            var nc = node[].get_neighbor_count(l)
            var off = node[]._get_offset(l)
            var oc = 0
            for i in range(nc):
                var neighbor_idx = Int(node[].neighbors[off + i])
                if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                if self.is_deleted(neighbor_idx): continue
                orphan_ids[l * 64 + oc] = neighbor_idx
                oc += 1
            orphan_counts[l] = oc

        # 2. Remove backlinks: for each neighbor, remove internal_id from their neighbor list
        for l in range(max_lvl + 1):
            var nc = node[].get_neighbor_count(l)
            var off = node[]._get_offset(l)
            for i in range(nc):
                var neighbor_idx = Int(node[].neighbors[off + i])
                if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                var nb_ptr = self.nodes + neighbor_idx
                var nb_off = nb_ptr[]._get_offset(l)
                var nb_nc = Int(nb_ptr[].neighbor_counts[l])
                for ni in range(nb_nc):
                    if Int(nb_ptr[].neighbors[nb_off + ni]) == internal_id:
                        # Swap with last and decrement
                        nb_ptr[].neighbors[nb_off + ni] = nb_ptr[].neighbors[nb_off + nb_nc - 1]
                        nb_ptr[].neighbor_counts[l] = UInt32(nb_nc - 1)
                        break

        # 3. Clear deleted node's neighbor counts
        for l in range(max_lvl + 1):
            node[].neighbor_counts[l] = 0

        # 4. Mark as deleted
        self.mark_deleted(internal_id)
        self.compact_dirty = True

        # 5. Replace entry point if needed
        if internal_id == self.entry_point_id:
            self.entry_point_id = -1
            var best_level = -1
            for i in range(self.num_nodes):
                if not self.is_deleted(i):
                    if self.nodes[i].max_level > best_level:
                        best_level = self.nodes[i].max_level
                        self.entry_point_id = i
            if self.entry_point_id != -1:
                self.max_level = best_level

        # 6. Reconnect orphaned neighbors
        self._reconnect_orphans(orphan_ids, orphan_counts, max_lvl)

        orphan_ids.free()
        orphan_counts.free()

    def _reconnect_orphans(mut self, orphan_ids: UnsafePointer[Int, MutUntrackedOrigin], orphan_counts: UnsafePointer[Int, MutUntrackedOrigin], max_lvl: Int):
        """A1b: Reconnect neighbors that lost a connection due to node deletion.
        For each level, if a former neighbor's connection count dropped below threshold,
        try to connect it to other former neighbors."""
        for l in range(max_lvl + 1):
            var oc = orphan_counts[l]
            if oc < 2: continue
            var cap = 2 * self.M if l == 0 else self.M
            var threshold = self.M if l == 0 else self.M // 2

            for i in range(oc):
                var nid = orphan_ids[l * 64 + i]
                if self.is_deleted(nid): continue
                var current_count = self.nodes[nid].get_neighbor_count(l)
                if current_count >= threshold: continue

                # Try connecting to other former neighbors
                for j in range(oc):
                    if i == j: continue
                    if current_count >= cap: break
                    var cid = orphan_ids[l * 64 + j]
                    if self.is_deleted(cid): continue

                    # Check not already connected
                    var already = False
                    var off = self.nodes[nid]._get_offset(l)
                    var nc = Int(self.nodes[nid].neighbor_counts[l])
                    for k in range(nc):
                        if Int(self.nodes[nid].neighbors[off + k]) == cid:
                            already = True
                            break
                    if already: continue

                    # Add bidirectional link
                    self.nodes[nid].add_neighbor(l, cid)
                    self.nodes[cid].add_neighbor(l, nid)
                    current_count += 1

    @always_inline
    def _greedy_in_code_space(self) -> Bool:
        """gh #395: the upper-level greedy runs on INT8 codes (the space the
        graph was built and the beam navigates in) whenever the stored codes
        are plain INT8 at a JIT-specialized dim. INT4 / binary formats keep the
        FP32-query path; the block-quant formats never reach it (see
        _upper_level_greedy)."""
        if self.use_bq or self.use_int4 or self.compact_is_int4: return False
        if self.compact_is_3bit or self.compact_is_2bit: return False
        return self.dim == 1536 or self.dim == 768 or self.dim == 384 or self.dim == 256 or self.dim == 128

    @always_inline
    def _code_dist(self, q: UnsafePointer[Int8, MutUntrackedOrigin], v: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
        if self.dim == 1536: return l2_int8_sabd_udot[1536](q, v)
        elif self.dim == 768: return l2_int8_sabd_udot[768](q, v)
        elif self.dim == 384: return l2_int8_sabd_udot[384](q, v)
        elif self.dim == 256: return l2_int8_sabd_udot[256](q, v)
        elif self.dim == 128: return l2_int8_sabd_udot[128](q, v)
        return l2_distance_int8(q, v, self.dim)

    @always_inline
    def _code_dist8(self, q: UnsafePointer[Int8, MutUntrackedOrigin], ids: Array[Int, 8]) -> SIMD[DType.float32, 8]:
        var v0 = self.nodes[ids[0]].vector; var v1 = self.nodes[ids[1]].vector
        var v2 = self.nodes[ids[2]].vector; var v3 = self.nodes[ids[3]].vector
        var v4 = self.nodes[ids[4]].vector; var v5 = self.nodes[ids[5]].vector
        var v6 = self.nodes[ids[6]].vector; var v7 = self.nodes[ids[7]].vector
        if self.dim == 1536: return l2_int8_sabd_udot_batch8[1536](q, v0, v1, v2, v3, v4, v5, v6, v7)
        elif self.dim == 768: return l2_int8_sabd_udot_batch8[768](q, v0, v1, v2, v3, v4, v5, v6, v7)
        elif self.dim == 384: return l2_int8_sabd_udot_batch8[384](q, v0, v1, v2, v3, v4, v5, v6, v7)
        elif self.dim == 256: return l2_int8_sabd_udot_batch8[256](q, v0, v1, v2, v3, v4, v5, v6, v7)
        return l2_int8_sabd_udot_batch8[128](q, v0, v1, v2, v3, v4, v5, v6, v7)

    @always_inline
    def _upper_greedy_codes(mut self) -> Int:
        """gh #395: upper-level greedy descent in INT8 code space, on the query
        already in `query_int8`. Returns the level-1 winner.

        It used to dequantize every neighbor against the FP32 query, one at a
        time: a single 384-deep FMA chain per 1536-d distance (285-328 ns),
        13% of INT8 search CPU. The exact SABD+UDOT distance on codes takes
        ~14 ns single-threaded (278 ns before; tests/test_int8_sabd_udot.mojo),
        and eight neighbors at a time overlap their cache misses. It is also
        the metric the graph was built in and the beam runs in; the FP32
        greedy used the GLOBAL range even on a per-group-calibrated index.

        Each step moves to the nearest neighbor strictly closer than the
        current node (first one on ties), reading the list of the node the
        step started at. The old loop switched to the new node's list
        mid-scan, so entry points can differ on near-ties; recall is the
        check (vector gate)."""
        var q = self.query_int8
        var curr = self.entry_point_id
        var curr_dist = self._code_dist(q, self.nodes[curr].vector)
        var ids = Array[Int, 8](fill=0)
        for l in range(self.max_level, 0, -1):
            var changed = True
            while changed:
                changed = False
                var node = curr
                var nc = self.nodes[node].get_neighbor_count(l)
                var best = curr_dist
                var best_idx = curr
                var i = 0
                while i < nc:
                    var cnt = 0
                    while i < nc and cnt < 8:
                        var nidx = self.nodes[node].get_neighbor(l, i)
                        i += 1
                        if nidx < 0 or nidx >= self.num_nodes: continue
                        if self.is_deleted(nidx): continue
                        ids[cnt] = nidx
                        cnt += 1
                    for j in range(cnt):
                        var vp = self.nodes[ids[j]].vector
                        prefetch(vp); prefetch(vp + 64); prefetch(vp + 128); prefetch(vp + 192)
                    if cnt == 8:
                        var d8 = self._code_dist8(q, ids)
                        for j in range(8):
                            if d8[j] < best:
                                best = d8[j]
                                best_idx = ids[j]
                    else:
                        for j in range(cnt):
                            var d = self._code_dist(q, self.nodes[ids[j]].vector)
                            if d < best:
                                best = d
                                best_idx = ids[j]
                if best_idx != curr:
                    curr = best_idx
                    curr_dist = best
                    changed = True
        return curr

    @always_inline
    def _upper_level_greedy(mut self, query: UnsafePointer[Float32, MutUntrackedOrigin]) -> Int:
        """Run upper-level greedy descent (levels max_level..1); return best entry node for base-level search.
        P3: called once per batch; result shared across all batch queries to save N-1 greedy passes."""
        # The block-quant searches (polar INT4, turbo INT3, nano INT2) run their
        # own greedy and ignore the P3 start node. Descending here read each
        # node's 676/484/868-byte block slot as 1536 INT8 codes — past the end
        # of the compact buffer on its last slot.
        if self.compact_is_3bit or self.compact_is_2bit or (self.polarquant and self.compact_is_int4):
            return self.entry_point_id
        if self._greedy_in_code_space():
            # gh #395: quantizing here also normalizes a COSINE query, which
            # the FP32 descent below never did for this batch entry point.
            self._quantize_query_to_int8(query)
            return self._upper_greedy_codes()
        var curr_node_idx = self.entry_point_id
        var curr_dist = self._dist_fp32_int8(query, self.nodes[curr_node_idx].vector)
        for l in range(self.max_level, 0, -1):
            var changed = True
            while changed:
                changed = False
                var neighbor_count = self.nodes[curr_node_idx].get_neighbor_count(l)
                for i in range(neighbor_count):
                    var neighbor_idx = self.nodes[curr_node_idx].get_neighbor(l, i)
                    if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                    if self.is_deleted(neighbor_idx): continue
                    if i + 1 < neighbor_count:
                        var nxt_idx = self.nodes[curr_node_idx].get_neighbor(l, i + 1)
                        if nxt_idx >= 0 and nxt_idx < self.num_nodes:
                            prefetch(self.nodes[nxt_idx].vector)
                    var d = self._dist_fp32_int8(query, self.nodes[neighbor_idx].vector)
                    if d < curr_dist:
                        curr_dist = d
                        curr_node_idx = neighbor_idx
                        changed = True
        return curr_node_idx

    @no_inline
    def _beam_search_1536(mut self, ef: Int, query_norm_sq: Float32, query_prefix_norm_sq: Float32):
        """Builds the view and hands the routine to `vector_abi` (D11):
        libpion_vector under -D PION_HELD_VECTOR, the open reference otherwise."""
        var vp = self.beam_view
        vp.unsafe_write(BeamView1536(
            rebind[UnsafePointer[LinearPool, MutUntrackedOrigin]](UnsafePointer(to=self.pool)),
            ef, query_norm_sq, query_prefix_norm_sq, self.query_int8,
            self.l0_compact, self.l0_slots, self.num_nodes, self.deleted_bitset,
            self.visited_epoch, self.cur_epoch,
            self.compact_buffer, self.compact_stride, self.compact_hdr, self.nodes))
        beam_search_1536(vp)

    def _quant_beam_1536(mut self, kind: Int, ef: Int):
        """Builds the view and hands the level-0 beam to `vector_abi` (D11):
        libpion_vector under -D PION_HELD_VECTOR, the open reference otherwise.
        KIND: QUANT_KIND_POLAR (INT4), _NANO (INT2), _TURBO (INT3 + QJL)."""
        var vp = self.quant_beam_view
        vp.unsafe_write(QuantBeamView1536(
            kind, ef,
            rebind[UnsafePointer[MinHeap, MutUntrackedOrigin]](UnsafePointer(to=self.candidates)),
            rebind[UnsafePointer[MaxHeap, MutUntrackedOrigin]](UnsafePointer(to=self.results)),
            self.query_block_int8, self.query_block_scales, self.query_block_norm,
            self.l0_compact, self.l0_slots, self.num_nodes, self.deleted_bitset,
            self.visited_epoch, self.cur_epoch,
            self.compact_buffer, self.compact_stride, self.compact_hdr, self.nodes,
            self.qjl_buffer, self.qjl_res_norms, self.qjl_query_signs,
            self.qjl_query_res_norm, self.qjl_lambda))
        quant_beam_search_1536(vp)

    @no_inline
    def search_gpu_brute_force(mut self, query: UnsafePointer[Float32, MutUntrackedOrigin], k: Int,
                              mut scores: List[Float32], worker_id: Int = 0) raises -> List[Int]:
        """§7v2 GPU Vector Engine: Metal brute-force search with FP32 re-ranking.
        GPU INT8 oversample (top-K×3) → CPU FP32 re-rank → top-K.
        Achieves recall ≥0.937 (vs 0.899 without re-rank) by eliminating INT8 quantization noise."""
        if self.num_nodes == 0: return List[Int]()
        if is_null(self.gpu_slot_ext_ids):
            return self.search_fp32_scored(query, k, scores)

        # Quantize query to INT8 (per-group Q8_K if calibrated, else global)
        self._quantize_query_to_int8(query)
        var query_norm_sq = norm_sq_int8_jit[1536](self.query_int8) if self.dim == 1536 else Float32(0.0)

        # §7v2: If FP32 rerank buffer exists, oversample 2× then re-rank.
        # If not (auto-calibrated INT8 is precise enough), use exact k.
        var final_k = min(k, self.num_nodes)
        var do_rerank = is_not_null(self.gpu_rerank_fp32)
        var oversample_k = min(final_k * 2, min(self.num_nodes, 256)) if do_rerank else min(final_k, 256)
        # Pre-alloc scratch if needed
        if is_null(self.gpu_topk_ids):
            # Sized for GPU_RERANK_K_MAX so the same scratch serves both §7v2
            # full-scan oversample (capped at 256) and the quantized rerank paths.
            self.gpu_topk_ids = alloc[Int32](GPU_RERANK_K_MAX)
            self.gpu_topk_dists = alloc[Float32](GPU_RERANK_K_MAX)

        # Fused GPU search + top-K (per-worker buffers, zero mutex contention)
        var found: Int = 0
        comptime if CompilationTarget.is_macos():
            found = Int(external_call["pion_metal_search_topk_w", Int32](
                self.query_int8, query_norm_sq, UInt32(self.num_nodes),
                UInt32(oversample_k), self.gpu_topk_ids, self.gpu_topk_dists,
                UInt32(worker_id)))
        if found <= 0:
            return self.search_fp32_scored(query, k, scores)

        # §7v2: FP32 re-rank — overwrite INT8 distances with exact FP32 L2.
        # GPU gather kernel for K ≥ threshold; CPU SIMD (NEON 128-bit × 2 pipes) below.
        if do_rerank and found > 0:
            var gpu_done = self._try_gpu_rerank(
                query, self.gpu_topk_ids, self.gpu_topk_dists, found, worker_id)
            if not gpu_done:
                var dim8 = self.dim >> 3  # dim / 8
                for i in range(found):
                    var slot_idx = Int(self.gpu_topk_ids[i])
                    if slot_idx >= 0 and slot_idx < self.num_nodes:
                        var fp32_vec = self.gpu_rerank_fp32 + slot_idx * self.dim
                        var acc = SIMD[DType.float32, 8](0)
                        for d in range(dim8):
                            var q = (query + d * 8).load[width=8]()
                            var v = (fp32_vec + d * 8).load[width=8]()
                            var diff = q - v
                            acc = acc + diff * diff
                        self.gpu_topk_dists[i] = acc.reduce_add()

        # Sort by distance (insertion sort — K is small)
        for i in range(1, found):
            var j = i
            while j > 0 and self.gpu_topk_dists[j] < self.gpu_topk_dists[j - 1]:
                var tmp_d = self.gpu_topk_dists[j]; self.gpu_topk_dists[j] = self.gpu_topk_dists[j-1]; self.gpu_topk_dists[j-1] = tmp_d
                var tmp_i = self.gpu_topk_ids[j]; self.gpu_topk_ids[j] = self.gpu_topk_ids[j-1]; self.gpu_topk_ids[j-1] = tmp_i
                j -= 1

        # Truncate to final_k after re-ranking
        if found > final_k: found = final_k

        # Build result list (slot_idx → external ID via gpu_slot_ext_ids)
        self.final_results.clear()
        scores.clear()
        for i in range(found):
            var slot_idx = Int(self.gpu_topk_ids[i])
            if slot_idx >= 0 and slot_idx < self.num_nodes:
                self.final_results.append(Int(self.gpu_slot_ext_ids[slot_idx]))
                scores.append(self.gpu_topk_dists[i])

        return self.final_results.copy()

    @no_inline
    def search_gpu_native(mut self, query: UnsafePointer[Float32, MutUntrackedOrigin], k: Int,
                         mut scores: List[Float32], worker_id: Int = 0) raises -> List[Int]:
        """§7v2 Native Mojo GPU search — zero ObjC overhead, direct DeviceContext dispatch.
        Falls back to Metal FFI path if native context not ready."""
        if not self.gpu_ctx.ready or self.num_nodes == 0 or is_null(self.gpu_slot_ext_ids):
            return self.search_gpu_brute_force(query, k, scores, worker_id)

        # Quantize query to INT8 (per-group Q8_K if calibrated)
        self._quantize_query_to_int8(query)
        var query_norm_sq = norm_sq_int8_jit[1536](self.query_int8) if self.dim == 1536 else Float32(0.0)

        # Pre-alloc scratch if needed
        if is_null(self.gpu_topk_ids):
            # Sized for GPU_RERANK_K_MAX so the same scratch serves both §7v2
            # full-scan oversample (capped at 256) and the quantized rerank paths.
            self.gpu_topk_ids = alloc[Int32](GPU_RERANK_K_MAX)
            self.gpu_topk_dists = alloc[Float32](GPU_RERANK_K_MAX)

        var actual_k = min(k, min(self.num_nodes, 256))
        var found = self.gpu_ctx.search(
            self.query_int8.bitcast[Int8](), query_norm_sq, actual_k,
            self.gpu_topk_ids, self.gpu_topk_dists)

        if found <= 0:
            return self.search_fp32_scored(query, k, scores)

        # Sort by distance (insertion sort)
        for i in range(1, found):
            var j = i
            while j > 0 and self.gpu_topk_dists[j] < self.gpu_topk_dists[j - 1]:
                var tmp_d = self.gpu_topk_dists[j]; self.gpu_topk_dists[j] = self.gpu_topk_dists[j-1]; self.gpu_topk_dists[j-1] = tmp_d
                var tmp_i = self.gpu_topk_ids[j]; self.gpu_topk_ids[j] = self.gpu_topk_ids[j-1]; self.gpu_topk_ids[j-1] = tmp_i
                j -= 1

        # Build result list
        self.final_results.clear()
        scores.clear()
        for i in range(found):
            var slot_idx = Int(self.gpu_topk_ids[i])
            if slot_idx >= 0 and slot_idx < self.num_nodes:
                self.final_results.append(Int(self.gpu_slot_ext_ids[slot_idx]))
                scores.append(self.gpu_topk_dists[i])

        return self.final_results.copy()

    @always_inline
    def _prefetch_rerank_row(self, ri: Int, n: Int):
        """Prefetch every line of the FP32 re-rank row of scratch_ids[ri]."""
        if ri >= n: return
        var nidx = self.scratch_ids[ri]
        if nidx < 0 or nidx >= self.num_nodes: return
        var p = (self.gpu_rerank_fp32 + nidx * self.dim).bitcast[Int8]()
        var off = 0
        var row_bytes = self.dim * 4
        while off < row_bytes:
            prefetch(p + off)
            off += 64

    def _rerank_fp32_sorted(mut self, query: UnsafePointer[Float32, MutUntrackedOrigin],
                            k: Int, mut scores: List[Float32]) -> List[Int]:
        """gh #396: the tail shared by the three block-quant searches (nano,
        turbo, polar): drain the beam, re-rank against the FP32 copy when there
        is one, sort, and map the first k to external ids.

        Two costs this removes, both measured on the quant modes' pipelined
        profile (27% of polar, 17% of turbo busy samples):
          - The heap pops farthest-first and the re-ranked distances track the
            beam's order, so an insertion sort over the popped order started
            near its worst case: ~n^2/4 swaps, 33 us per query at ef=150.
            Reversed, the input is nearly sorted and the same sort takes
            0.45 us. Only exactly-equal distances (duplicate vectors) can end
            up in a different order, which recall cannot see.
          - Each re-rank row is 6 KB of random FP32 reads. Prefetching three
            lines of the NEXT row left the rest of it to demand misses; the
            whole row two ahead is prefetched instead (1.20-1.25x on the scan).
        `query` must be heap memory: this function is out of line (gh #349)."""
        self.scratch_ids.clear()
        self.scratch_dists.clear()
        while len(self.results.data) > 0:
            var r = self.results.pop()
            self.scratch_ids.append(r.id)
            self.scratch_dists.append(r.distance)
        var n = len(self.scratch_ids)
        var lo = 0
        var hi = n - 1
        while lo < hi:
            var ti = self.scratch_ids[lo]; self.scratch_ids[lo] = self.scratch_ids[hi]; self.scratch_ids[hi] = ti
            var td = self.scratch_dists[lo]; self.scratch_dists[lo] = self.scratch_dists[hi]; self.scratch_dists[hi] = td
            lo += 1
            hi -= 1
        if is_not_null(self.gpu_rerank_fp32):
            # GPU gather kernel for K >= threshold; SIMD CPU scan below.
            var gpu_done = False
            if n >= GPU_RERANK_THRESHOLD and n <= GPU_RERANK_K_MAX and is_not_null(self.gpu_topk_ids):
                for ri in range(n):
                    self.gpu_topk_ids[ri] = Int32(self.scratch_ids[ri])
                if self._try_gpu_rerank(query, self.gpu_topk_ids, self.gpu_topk_dists, n, 0):
                    for ri in range(n):
                        self.scratch_dists[ri] = self.gpu_topk_dists[ri]
                    gpu_done = True
            if not gpu_done:
                self._prefetch_rerank_row(0, n)
                self._prefetch_rerank_row(1, n)
                for ri in range(n):
                    self._prefetch_rerank_row(ri + 2, n)
                    var nidx = self.scratch_ids[ri]
                    if nidx < 0 or nidx >= self.num_nodes: continue
                    var fp32_vec = self.gpu_rerank_fp32 + nidx * self.dim
                    var acc_a = SIMD[DType.float32, 8](0)
                    var acc_b = SIMD[DType.float32, 8](0)
                    var di = 0
                    while di + 16 <= self.dim:
                        var qa = (query + di).load[width=8]()
                        var va = (fp32_vec + di).load[width=8]()
                        var da = qa - va; acc_a = acc_a + da * da
                        var qb = (query + di + 8).load[width=8]()
                        var vb = (fp32_vec + di + 8).load[width=8]()
                        var db = qb - vb; acc_b = acc_b + db * db
                        di += 16
                    while di + 8 <= self.dim:
                        var qa = (query + di).load[width=8]()
                        var va = (fp32_vec + di).load[width=8]()
                        var da = qa - va; acc_a = acc_a + da * da
                        di += 8
                    self.scratch_dists[ri] = (acc_a + acc_b).reduce_add()
        # Insertion sort by (re-ranked) distance — near-sorted input, see above.
        for i in range(1, n):
            var j = i
            while j > 0 and self.scratch_dists[j] < self.scratch_dists[j - 1]:
                var tmp_d = self.scratch_dists[j]; self.scratch_dists[j] = self.scratch_dists[j-1]; self.scratch_dists[j-1] = tmp_d
                var tmp_i = self.scratch_ids[j]; self.scratch_ids[j] = self.scratch_ids[j-1]; self.scratch_ids[j-1] = tmp_i
                j -= 1
        self.final_results.clear()
        scores.clear()
        for ri in range(min(k, n)):
            var nidx = self.scratch_ids[ri]
            if nidx >= 0 and nidx < self.num_nodes:
                self.final_results.append(self.nodes[nidx].id)
                scores.append(self.scratch_dists[ri])
        return self.final_results.copy()

    @no_inline
    def search_fp32_scored(mut self, query_in: UnsafePointer[Float32, MutUntrackedOrigin], k: Int,
                          mut scores: List[Float32], ef: Int = 100, start_node: Int = -1) raises -> List[Int]:
        """Like search_fp32 but also fills `scores` with L2 distance for each returned ID.
        P3: when start_node >= 0, skips upper-level greedy (shared across batch queries)."""
        if self.entry_point_id == -1: return List[Int]()
        # gh #271: normalize HERE, not only in _quantize_query_to_int8. Two
        # things in this function consume the FP32 query directly and would
        # otherwise run in a different space from the normalized nodes: the
        # upper-level greedy descent (`_dist_fp32_int8`, which runs BEFORE the
        # query is quantized) and the quant-variant block quantizers. Both would
        # still return an answer — a worse entry point, quietly worse recall —
        # which is the failure mode this issue is about. Normalization is
        # idempotent, so _quantize_query_to_int8 re-doing it below is a no-op.
        var query = query_in
        if self.distance_metric == 1:
            l2_normalize_fp32(query_in, self._norm_scratch(), self.dim)
            query = self.fp32_norm_scratch
        # A1b: Lazy compact rebuild after VREM deletions
        if self.compact_dirty and is_not_null(self.l0_compact):
            self.compact_vectors()
            self.compact_dirty = False
        # ef cap removed: capping ef at 64 dropped recall from 0.9366 → 0.64

        # ── N4 NanoQuant: Block-wise INT2 search ──────────────────────────────
        if self.nanoquant and self.compact_is_2bit and self.dim == 1536:
            # Block-quantize query to INT8 (once per search)
            self.query_block_norm = quantize_fp32_to_block_int8(
                query, self.query_block_int8, self.query_block_scales, self.dim)

            # Upper levels: greedy with block-INT2 distance
            var nq_node = self.entry_point_id
            var nq_norm_v = self.nodes[nq_node].vector.bitcast[Float32]()[0]
            var nq_dot = int2_dot_single_simd(
                self.query_block_int8, self.query_block_scales, self.nodes[nq_node].vector)
            var nq_dist = self.query_block_norm + nq_norm_v - 2.0 * nq_dot
            for l in range(self.max_level, 0, -1):
                var changed = True
                while changed:
                    changed = False
                    var nc = self.nodes[nq_node].get_neighbor_count(l)
                    for i in range(nc):
                        var nidx = self.nodes[nq_node].get_neighbor(l, i)
                        if nidx < 0 or nidx >= self.num_nodes: continue
                        if self.is_deleted(nidx): continue
                        var nv = self.nodes[nidx].vector
                        var n_norm = nv.bitcast[Float32]()[0]
                        var n_dot = int2_dot_single_simd(
                            self.query_block_int8, self.query_block_scales, nv)
                        var d = self.query_block_norm + n_norm - 2.0 * n_dot
                        if d < nq_dist:
                            nq_dist = d
                            nq_node = nidx
                            changed = True

            # Beam search
            self._reset_visited()
            self.candidates.clear()
            self.results.clear()
            self.candidates.reserve(ef + 32)  # gh #131 §2.5
            self.results.reserve(ef + 32)
            self.candidates.push(HeapNode(nq_dist, nq_node))
            self.results.push(HeapNode(nq_dist, nq_node))
            self.visited_epoch[nq_node] = self.cur_epoch
            self._quant_beam_1536(QUANT_KIND_NANO, ef)

            # FP32 re-rank, sort, first k (gh #396: shared tail).
            return self._rerank_fp32_sorted(query, k, scores)
        # ── End N4 NanoQuant ──────────────────────────────────────────────────

        # ── M7 TurboQuant: Block-wise INT3 search + QJL correction ──────────────
        if self.turboquant and self.compact_is_3bit and self.dim == 1536:
            # Block-quantize query to INT8 (once per search)
            self.query_block_norm = quantize_fp32_to_block_int8(
                query, self.query_block_int8, self.query_block_scales, self.dim)

            # Compute QJL signs for query residual
            if is_not_null(self.qjl_random_signs) and is_not_null(self.wht_scratch):
                # gh #396: a per-worker scratch, not an alloc/free per query.
                var residual_tmp = self._residual_scratch()
                # Compute dequantized query from INT8 blocks
                for b in range(NUM_BLOCKS_1536):
                    var scale = self.query_block_scales[b]
                    for i in range(BLOCK_DIM):
                        var q_val = Float32(Int(self.query_block_int8[b * BLOCK_DIM + i]))
                        residual_tmp[b * BLOCK_DIM + i] = query[b * BLOCK_DIM + i] - q_val * scale
                self.qjl_query_res_norm = qjl_compute_signs(
                    residual_tmp, self.qjl_random_signs, self.wht_scratch,
                    self.qjl_query_signs, self.dim)

            # Upper levels: greedy with block-INT3 distance
            var tq_node = self.entry_point_id
            var tq_norm_v = self.nodes[tq_node].vector.bitcast[Float32]()[0]
            var tq_dot = int3_dot_single_simd(
                self.query_block_int8, self.query_block_scales, self.nodes[tq_node].vector)
            var tq_dist = self.query_block_norm + tq_norm_v - 2.0 * tq_dot
            for l in range(self.max_level, 0, -1):
                var changed = True
                while changed:
                    changed = False
                    var nc = self.nodes[tq_node].get_neighbor_count(l)
                    for i in range(nc):
                        var nidx = self.nodes[tq_node].get_neighbor(l, i)
                        if nidx < 0 or nidx >= self.num_nodes: continue
                        var nv = self.nodes[nidx].vector
                        var n_norm = nv.bitcast[Float32]()[0]
                        var n_dot = int3_dot_single_simd(
                            self.query_block_int8, self.query_block_scales, nv)
                        var d = self.query_block_norm + n_norm - 2.0 * n_dot
                        if d < tq_dist:
                            tq_dist = d
                            tq_node = nidx
                            changed = True

            # Beam search
            self._reset_visited()
            self.candidates.clear()
            self.results.clear()
            self.candidates.reserve(ef + 32)  # gh #131 §2.5
            self.results.reserve(ef + 32)
            self.candidates.push(HeapNode(tq_dist, tq_node))
            self.results.push(HeapNode(tq_dist, tq_node))
            self.visited_epoch[tq_node] = self.cur_epoch
            self._quant_beam_1536(QUANT_KIND_TURBO, ef)

            # FP32 re-rank, sort, first k (gh #396: shared tail).
            return self._rerank_fp32_sorted(query, k, scores)
        # ── End M7 TurboQuant ──────────────────────────────────────────────────

        # ── M6b PolarQuant: Block-wise INT4 search (SQ4_Block32) ────────────────
        if self.polarquant and self.compact_is_int4 and self.dim == 1536:
            # Block-quantize query to INT8 (once per search)
            self.query_block_norm = quantize_fp32_to_block_int8(
                query, self.query_block_int8, self.query_block_scales, self.dim)

            # Upper levels: greedy with block-INT4 distance
            var pq_node = self.entry_point_id
            var pq_norm_v = self.nodes[pq_node].vector.bitcast[Float32]()[0]
            var pq_dot = block_int4_dot_single_simd(
                self.query_block_int8, self.query_block_scales, self.nodes[pq_node].vector)
            var pq_dist = self.query_block_norm + pq_norm_v - 2.0 * pq_dot
            for l in range(self.max_level, 0, -1):
                var changed = True
                while changed:
                    changed = False
                    var nc = self.nodes[pq_node].get_neighbor_count(l)
                    for i in range(nc):
                        var nidx = self.nodes[pq_node].get_neighbor(l, i)
                        if nidx < 0 or nidx >= self.num_nodes: continue
                        var nv = self.nodes[nidx].vector
                        var n_norm = nv.bitcast[Float32]()[0]
                        var n_dot = block_int4_dot_single_simd(
                            self.query_block_int8, self.query_block_scales, nv)
                        var d = self.query_block_norm + n_norm - 2.0 * n_dot
                        if d < pq_dist:
                            pq_dist = d
                            pq_node = nidx
                            changed = True

            # Beam search
            self._reset_visited()
            self.candidates.clear()
            self.results.clear()
            self.candidates.reserve(ef + 32)  # gh #131 §2.5
            self.results.reserve(ef + 32)
            self.candidates.push(HeapNode(pq_dist, pq_node))
            self.results.push(HeapNode(pq_dist, pq_node))
            self.visited_epoch[pq_node] = self.cur_epoch
            self._quant_beam_1536(QUANT_KIND_POLAR, ef)

            # FP32 re-rank, sort, first k (gh #396: shared tail).
            return self._rerank_fp32_sorted(query, k, scores)
        # ── End M6b PolarQuant ─────────────────────────────────────────────────

        var curr_node_idx: Int
        var query_quantized = False
        if start_node >= 0 and start_node < self.num_nodes:
            # P3: skip upper-level greedy — reuse shared start node from batch coordinator
            curr_node_idx = start_node
        elif self._greedy_in_code_space():
            # gh #395: quantize first and descend in code space.
            self._quantize_query_to_int8(query)
            query_quantized = True
            curr_node_idx = self._upper_greedy_codes()
        else:
            var curr_dist: Float32
            curr_node_idx = self.entry_point_id
            curr_dist = self._dist_fp32_int8(query, self.nodes[curr_node_idx].vector)

            # Upper levels: greedy search
            for l in range(self.max_level, 0, -1):
                var changed = True
                while changed:
                    changed = False
                    var neighbor_count = self.nodes[curr_node_idx].get_neighbor_count(l)
                    for i in range(neighbor_count):
                        var neighbor_idx = self.nodes[curr_node_idx].get_neighbor(l, i) # 3.1: internal index
                        if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                        if self.is_deleted(neighbor_idx): continue
                        if i + 1 < neighbor_count:
                            var nxt_idx = self.nodes[curr_node_idx].get_neighbor(l, i + 1)
                            if nxt_idx >= 0 and nxt_idx < self.num_nodes:
                                prefetch(self.nodes[nxt_idx].vector)
                        var d = self._dist_fp32_int8(query, self.nodes[neighbor_idx].vector)
                        if d < curr_dist:
                            curr_dist = d
                            curr_node_idx = neighbor_idx
                            changed = True

        # Base level: beam search
        self._reset_visited()
        self.visited_epoch[curr_node_idx] = self.cur_epoch

        # V10: quantize query to INT8 once — per-group Q8_K if calibrated
        if not query_quantized:
            self._quantize_query_to_int8(query)
        # 2.5: SDOT-based distance — compute query norms once per search.
        # Node norms are inline in compact_buffer (8-byte header before each vector).
        var query_norm_sq = norm_sq_int8_jit[1536](self.query_int8) if self.dim == 1536 else Float32(0.0)
        var query_prefix_norm_sq = norm_sq_int8_jit[256](self.query_int8) if self.dim == 1536 else Float32(0.0)

        # V8.1: INT8-INT8 path — query_int8 already computed above.
        # Batch kernels match graph build metric (no dequantization per neighbor).
        # Distances in INT32 space (not FP32); ordering is monotone → recall unchanged.
        # Seed with the INT8-INT8 distance directly (consistent metric; the old
        # fp32-push-then-repop dance ended in this exact state).
        var curr_dist_i8 = self._dist_int8_int8(self.query_int8, self.nodes[curr_node_idx].vector)
        if self.dim == 1536:
            # gh #197: LinearPool replaces the candidates/results heap pair on
            # this path only. 1.10's fill/prune split is preserved inside.
            self.pool.reset(ef)
            _ = self.pool.insert(curr_dist_i8, curr_node_idx)
            self._beam_search_1536(ef, query_norm_sq, query_prefix_norm_sq)
        else:
            self.candidates.clear()
            self.results.clear()
            self.candidates.reserve(ef + 32)  # gh #131 §2.5
            self.results.reserve(ef + 32)
            self.candidates.push(HeapNode(curr_dist_i8, curr_node_idx))
            self.results.push(HeapNode(curr_dist_i8, curr_node_idx))
            # Non-1536 dims: combined fill+prune loop, no prefix pruning
            var valid_idxs = Array[Int, 65](uninitialized=True)
            var _evals_this_query = 0
            while len(self.candidates.data) > 0:
                var c = self.candidates.pop()
                var worst = self.results.peek_distance()
                if c.distance > worst and len(self.results.data) >= ef: break
                var neighbor_count = self.nodes[c.id].get_neighbor_count(0)
                var valid_count = 0
                for i in range(neighbor_count):
                    var neighbor_idx = self.nodes[c.id].get_neighbor(0, i)
                    if neighbor_idx < 0 or neighbor_idx >= self.num_nodes: continue
                    if self.is_deleted(neighbor_idx): continue
                    if self.visited_epoch[neighbor_idx] == self.cur_epoch: continue
                    self.visited_epoch[neighbor_idx] = self.cur_epoch
                    valid_idxs[valid_count] = neighbor_idx
                    valid_count += 1
                _evals_this_query += valid_count
                for i in range(valid_count):
                    var vptr = self.nodes[valid_idxs[i]].vector
                    prefetch(vptr); prefetch(vptr + 64); prefetch(vptr + 128); prefetch(vptr + 192)
                var vi = 0
                while vi + 8 <= valid_count:
                    var i0 = valid_idxs[vi];     var i1 = valid_idxs[vi + 1]
                    var i2 = valid_idxs[vi + 2]; var i3 = valid_idxs[vi + 3]
                    var i4 = valid_idxs[vi + 4]; var i5 = valid_idxs[vi + 5]
                    var i6 = valid_idxs[vi + 6]; var i7 = valid_idxs[vi + 7]
                    var dists8: SIMD[DType.float32, 8]
                    if self.dim == 256:
                        dists8 = l2_distance_int8_int8_batch8_jit[256](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector, self.nodes[i4].vector,
                            self.nodes[i5].vector, self.nodes[i6].vector, self.nodes[i7].vector)
                    elif self.dim == 768:
                        dists8 = l2_distance_int8_int8_batch8_jit[768](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector, self.nodes[i4].vector,
                            self.nodes[i5].vector, self.nodes[i6].vector, self.nodes[i7].vector)
                    elif self.dim == 384:
                        dists8 = l2_distance_int8_int8_batch8_jit[384](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector, self.nodes[i4].vector,
                            self.nodes[i5].vector, self.nodes[i6].vector, self.nodes[i7].vector)
                    else:
                        dists8 = SIMD[DType.float32, 8](
                            self._dist_int8_int8(self.query_int8, self.nodes[i0].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i1].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i2].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i3].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i4].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i5].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i6].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i7].vector))
                    for j in range(8):
                        var d = dists8[j]; var nidx = valid_idxs[vi + j]
                        if self.results.push_bounded(d, nidx, ef):
                            self.candidates.push(HeapNode(d, nidx))
                    vi += 8
                while vi + 4 <= valid_count:
                    var i0 = valid_idxs[vi]; var i1 = valid_idxs[vi + 1]
                    var i2 = valid_idxs[vi + 2]; var i3 = valid_idxs[vi + 3]
                    var dists4: SIMD[DType.float32, 4]
                    if self.dim == 256:
                        dists4 = l2_distance_int8_int8_batch4_jit[256](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector)
                    elif self.dim == 768:
                        dists4 = l2_distance_int8_int8_batch4_jit[768](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector)
                    elif self.dim == 384:
                        dists4 = l2_distance_int8_int8_batch4_jit[384](
                            self.query_int8, self.nodes[i0].vector, self.nodes[i1].vector,
                            self.nodes[i2].vector, self.nodes[i3].vector)
                    else:
                        dists4 = SIMD[DType.float32, 4](
                            self._dist_int8_int8(self.query_int8, self.nodes[i0].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i1].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i2].vector),
                            self._dist_int8_int8(self.query_int8, self.nodes[i3].vector))
                    for j in range(4):
                        var d = dists4[j]; var nidx = valid_idxs[vi + j]
                        if self.results.push_bounded(d, nidx, ef):
                            self.candidates.push(HeapNode(d, nidx))
                    vi += 4
                while vi < valid_count:
                    var nidx = valid_idxs[vi]
                    var d = self._dist_int8_int8(self.query_int8, self.nodes[nidx].vector)
                    if self.results.push_bounded(d, nidx, ef):
                        self.candidates.push(HeapNode(d, nidx))
                    vi += 1

        # Instrumentation: accumulate and print every 1000 queries
        var _evals_this_query = 0
        self.stat_total_evals += _evals_this_query
        self.stat_query_count += 1
        if self.stat_query_count % 1000 == 0:
            _ = self.stat_total_evals / self.stat_query_count
            # print("[STATS] queries=" + String(self.stat_query_count) + " avg_evals=" + String(avg))

        # gh #197: pool drain is a bounded read of the first k entries — the
        # array is already nearest-first (no full-depth pop sifts, no reversal).
        if self.dim == 1536:
            self.final_results.clear()
            var n_out = min(k, self.pool.size)
            for i in range(n_out):
                self.final_results.append(self.nodes[self.pool.entry_id(i)].id)
                scores.append(self.pool.entry_dist(i))
            return self.final_results.copy()

        # Pop from MaxHeap (highest distance first), reverse to get nearest-first
        self.scratch_ids.clear()
        self.scratch_dists.clear()
        while len(self.results.data) > 0:
            var node = self.results.pop()
            self.scratch_ids.append(self.nodes[node.id].id)
            self.scratch_dists.append(node.distance)

        self.final_results.clear()
        for i in range(len(self.scratch_ids) - 1, -1, -1):
            self.final_results.append(self.scratch_ids[i])
            scores.append(self.scratch_dists[i])
            if len(self.final_results) >= k: break

        return self.final_results.copy()

    def metric_scores(mut self, query_in: UnsafePointer[Float32, MutUntrackedOrigin],
                      ids: List[Int], mut scores: List[Float32]) -> Bool:
        """gh #365: replace search scores with the index METRIC's distance for
        the k results a reply returns — squared L2 under L2, `1 - cos` under
        COSINE (RediSearch's units). The beam ranks in quantized code space,
        where a vector scored ~750-1500 against itself and a neighbour ~780,000,
        so every client that read `score` as a distance or similarity misread it.

        Source per node: the FP32 re-rank copy when one exists, else the INT8
        codes dequantized with the calibration they were written with. Formats
        with neither (INT4 / binary) keep their scores and return False.
        `ids` are EXTERNAL ids, as FT.SEARCH replies carry them."""
        var n = len(ids)
        if n == 0: return True
        var have_fp32 = is_not_null(self.gpu_rerank_fp32)
        var int8_ok = not (self.compact_is_int4 or self.use_int4 or self.use_bq
                           or self.polarquant or self.turboquant or self.nanoquant)
        if not have_fp32 and not int8_ok: return False
        var query = query_in
        if self.distance_metric == 1:
            l2_normalize_fp32(query_in, self._norm_scratch(), self.dim)
            query = self.fp32_norm_scratch
        var dim = self.dim
        var grouped = self.grouped_calibrated and self.num_groups > 0 and dim % 32 == 0
        var range_val = self.global_max - self.global_min
        if range_val <= 0.0: range_val = 1.0
        var g_inv = range_val / Float32(254.0)
        # gh #395: dequantize as one FMA, c * a + b, with the constants hoisted
        # out of the row loop (per group under per-group calibration: 1/scale
        # used to be recomputed for every group of every row), and four
        # independent accumulator chains instead of one. The chains
        # reassociate the FP32 sum, so a score can move in its last bits.
        var ga = self._residual_scratch()          # [2 * groups]: a then b
        if grouped:
            for g in range(dim // 32):
                var inv = Float32(1.0) / self.group_scales[g]
                ga[g] = inv
                ga[dim // 32 + g] = 127.0 * inv + self.group_qmins[g]
        var a8 = SIMD[DType.float32, 8](g_inv)
        var b8 = SIMD[DType.float32, 8](127.0 * g_inv + self.global_min)
        for r in range(n):
            var ext = ids[r]
            if ext < 0 or ext >= self.max_elements: continue
            var nidx = self.node_map[ext]
            if nidx < 0 or nidx >= self.num_nodes: continue
            var c0 = SIMD[DType.float32, 8](0); var c1 = SIMD[DType.float32, 8](0)
            var c2 = SIMD[DType.float32, 8](0); var c3 = SIMD[DType.float32, 8](0)
            var tail = Float32(0.0)
            if have_fp32:
                var v = self.gpu_rerank_fp32 + nidx * dim
                var d = 0
                while d + 32 <= dim:
                    var t0 = (query + d).load[width=8]() - (v + d).load[width=8]();           c0 = fma(t0, t0, c0)
                    var t1 = (query + d + 8).load[width=8]() - (v + d + 8).load[width=8]();   c1 = fma(t1, t1, c1)
                    var t2 = (query + d + 16).load[width=8]() - (v + d + 16).load[width=8](); c2 = fma(t2, t2, c2)
                    var t3 = (query + d + 24).load[width=8]() - (v + d + 24).load[width=8](); c3 = fma(t3, t3, c3)
                    d += 32
                while d + 8 <= dim:
                    var t = (query + d).load[width=8]() - (v + d).load[width=8]()
                    c0 = fma(t, t, c0)
                    d += 8
                while d < dim:
                    var t = query[d] - v[d]; tail += t * t; d += 1
            else:
                var codes = self.nodes[nidx].vector
                if is_null(codes): continue
                if grouped:
                    var ng = dim // 32
                    for g in range(ng):
                        var ag = SIMD[DType.float32, 8](ga[g])
                        var bg = SIMD[DType.float32, 8](ga[ng + g])
                        var off = g * 32
                        var t0 = (query + off).load[width=8]() - fma((codes + off).load[width=8]().cast[DType.float32](), ag, bg)
                        c0 = fma(t0, t0, c0)
                        var t1 = (query + off + 8).load[width=8]() - fma((codes + off + 8).load[width=8]().cast[DType.float32](), ag, bg)
                        c1 = fma(t1, t1, c1)
                        var t2 = (query + off + 16).load[width=8]() - fma((codes + off + 16).load[width=8]().cast[DType.float32](), ag, bg)
                        c2 = fma(t2, t2, c2)
                        var t3 = (query + off + 24).load[width=8]() - fma((codes + off + 24).load[width=8]().cast[DType.float32](), ag, bg)
                        c3 = fma(t3, t3, c3)
                else:
                    var d = 0
                    while d + 32 <= dim:
                        var t0 = (query + d).load[width=8]() - fma((codes + d).load[width=8]().cast[DType.float32](), a8, b8)
                        c0 = fma(t0, t0, c0)
                        var t1 = (query + d + 8).load[width=8]() - fma((codes + d + 8).load[width=8]().cast[DType.float32](), a8, b8)
                        c1 = fma(t1, t1, c1)
                        var t2 = (query + d + 16).load[width=8]() - fma((codes + d + 16).load[width=8]().cast[DType.float32](), a8, b8)
                        c2 = fma(t2, t2, c2)
                        var t3 = (query + d + 24).load[width=8]() - fma((codes + d + 24).load[width=8]().cast[DType.float32](), a8, b8)
                        c3 = fma(t3, t3, c3)
                        d += 32
                    while d + 8 <= dim:
                        var t = (query + d).load[width=8]() - fma((codes + d).load[width=8]().cast[DType.float32](), a8, b8)
                        c0 = fma(t, t, c0)
                        d += 8
                    while d < dim:
                        var x1 = fma(Float32(Int(codes[d])), g_inv, 127.0 * g_inv + self.global_min)
                        var t = query[d] - x1; tail += t * t; d += 1
            scores[r] = ((c0 + c1) + (c2 + c3)).reduce_add() + tail
            if self.distance_metric == 1:
                # unit vectors: |q - v|^2 = 2 - 2cos, so 1 - cos = |q - v|^2 / 2
                scores[r] = scores[r] * 0.5
        return True

    def _dist_fp32_int8(self, v1: UnsafePointer[Float32, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
        if is_null(v1) or is_null(v2): return Float32(1e30)
        var range_val = self.global_max - self.global_min
        if range_val <= 0: range_val = 1.0

        # V5.2: compact_buffer uses INT4 when compact_is_int4=True (set by compact_vectors)
        if self.compact_is_int4:
            if self.dim == 1536:
                return l2_distance_fp32_int4_jit[1536](v1, v2, self.global_min, range_val)
            elif self.dim == 768:
                return l2_distance_fp32_int4_jit[768](v1, v2, self.global_min, range_val)
            elif self.dim == 384:
                return l2_distance_fp32_int4_jit[384](v1, v2, self.global_min, range_val)
            else:
                return l2_distance_fp32_int4(v1, v2, self.dim, self.global_min, range_val)

        if self.use_int4 and not self.polarquant:
            return l2_distance_fp32_int4(v1, v2, self.dim, self.global_min, range_val)

        # JIT Specialization for common dimensions
        if self.dim == 128:
            return l2_distance_fp32_int8_fused_jit[128](v1, v2, self.global_min, range_val)
        elif self.dim == 256:
            return l2_distance_fp32_int8_fused_jit[256](v1, v2, self.global_min, range_val)
        elif self.dim == 384:
            return l2_distance_fp32_int8_fused_jit[384](v1, v2, self.global_min, range_val)
        elif self.dim == 768:
            return l2_distance_fp32_int8_fused_jit[768](v1, v2, self.global_min, range_val)
        elif self.dim == 1536:
            return l2_distance_fp32_int8_fused_jit[1536](v1, v2, self.global_min, range_val)

        return l2_distance_fp32_int8(v1, v2, self.dim, self.global_min, range_val)

    def _dist_int8_int8(self, v1: UnsafePointer[Int8, MutUntrackedOrigin], v2: UnsafePointer[Int8, MutUntrackedOrigin]) -> Float32:
        if is_null(v1) or is_null(v2): return Float32(1e30)
        if self.use_bq:
            var dim_u64 = (self.dim + 63) // 64
            if dim_u64 == 2:
                return hamming_distance_jit[2](v1.bitcast[UInt64](), v2.bitcast[UInt64]())
            elif dim_u64 == 6:
                return hamming_distance_jit[6](v1.bitcast[UInt64](), v2.bitcast[UInt64]())
            elif dim_u64 == 12:
                return hamming_distance_jit[12](v1.bitcast[UInt64](), v2.bitcast[UInt64]())
            elif dim_u64 == 24:
                return hamming_distance_jit[24](v1.bitcast[UInt64](), v2.bitcast[UInt64]())
            return hamming_distance(v1.bitcast[UInt64](), v2.bitcast[UInt64](), dim_u64)

        if self.use_int4 and not self.polarquant:
            # JIT Specialization for common dimensions
            if self.dim == 128:
                return l2_distance_int4_jit[128](v1, v2)
            elif self.dim == 384:
                return l2_distance_int4_jit[384](v1, v2)
            elif self.dim == 768:
                return l2_distance_int4_jit[768](v1, v2)
            elif self.dim == 1536:
                return l2_distance_int4_jit[1536](v1, v2)
            return l2_distance_int4(v1, v2, self.dim)

        # JIT Specialization for common dimensions
        if self.dim == 128:
            return l2_distance_int8_jit[128](v1, v2)
        elif self.dim == 256:
            return l2_distance_int8_jit[256](v1, v2)
        elif self.dim == 384:
            return l2_distance_int8_jit[384](v1, v2)
        elif self.dim == 768:
            return l2_distance_int8_jit[768](v1, v2)
        elif self.dim == 1536:
            return l2_distance_int8_jit[1536](v1, v2)

        return l2_distance_int8(v1, v2, self.dim)

    def deinit(owned self):
        if is_not_null(self.query_int8): self.query_int8.free()
        if is_not_null(self.visited_epoch): self.visited_epoch.free()
        if is_not_null(self.bm25_scores_scratch): self.bm25_scores_scratch.free()  # always owned (pre-alloc)
        # gh #139: the BM25 doc set is never published to / borrowed from the
        # shared view, so it is owned even by a borrowing graph — free it above
        # the is_borrowed early-return or it leaks on every borrower.
        if is_not_null(self.bm25_doc_ids):  self.bm25_doc_ids.free()
        if is_not_null(self.bm25_text_ids): self.bm25_text_ids.free()
        if is_not_null(self.residual_scratch): self.residual_scratch.free()   # gh #396: per-worker

        if self.is_borrowed:
            return

        for i in range(self.num_nodes):
            (self.nodes + i).unsafe_deinit_pointee()
        self.nodes.free()
        self.node_map.free()
        self.neighbor_pool.free()

        if is_not_null(self.fp32_buffer): self.fp32_buffer.free()
        if is_not_null(self.fp32_ids): self.fp32_ids.free()
        if is_not_null(self.gpu_rerank_fp32): self.gpu_rerank_fp32.free()
        if is_not_null(self.compact_buffer): self.compact_buffer.free()
        if is_not_null(self.l0_compact): self.l0_compact.free()
        if is_not_null(self.l0_slots): self.l0_slots.free()
        if is_not_null(self.bm25_term_hashes):    self.bm25_term_hashes.free()
        if is_not_null(self.bm25_term_idf):       self.bm25_term_idf.free()
        if is_not_null(self.bm25_postings_start): self.bm25_postings_start.free()
        if is_not_null(self.bm25_postings_count): self.bm25_postings_count.free()
        if is_not_null(self.bm25_postings_buf):   self.bm25_postings_buf.free()
        if is_not_null(self.bm25_doc_lengths):    self.bm25_doc_lengths.free()
        # M7 TurboQuant
        if is_not_null(self.qjl_buffer): self.qjl_buffer.free()
        if is_not_null(self.qjl_res_norms): self.qjl_res_norms.free()
        if is_not_null(self.qjl_random_signs): self.qjl_random_signs.free()
        if is_not_null(self.qjl_query_signs): self.qjl_query_signs.free()

