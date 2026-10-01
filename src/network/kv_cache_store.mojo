"""KVCacheStore — per-worker store for LLM KV cache blobs, indexed by prompt embeddings via HNSW.

Phase 1 of M14 (Externalized Attention). Stores complete KV cache tensors from
prefill workers and returns the closest semantic match for decode workers.

Commands:
  KV.STORE <cache_id> <embedding_blob> <tensor_blob> [TTL <seconds>] [MODEL <name>]
  KV.FETCH <embedding_blob> [THRESHOLD <cosine_sim>] [MODEL <name>]
  KV.INFO

Embeddings are passed as raw FP32 binary blobs (dim * 4 bytes).
Tensor blobs are heap-copied (alloc + memcpy on STORE; we own the memory).

Semantic matching: HNSW search on prompt embeddings returns the closest cached
KV tensor when cosine similarity exceeds threshold (default 0.95).

Lifecycle (gh #80):
  - LRU eviction: at KVCACHE_MAX_ENTRIES, STORE evicts the least-recently-
    accessed entry instead of rejecting the write.
  - TTL: lazily enforced on FETCH — expired entries are reclaimed and skipped.
  - MODEL: FETCH filters candidates by model tag when MODEL is given.
  - HNSW internal nodes are append-only, so evicted slots tombstone their node
    (unlink_node) and the graph is rebuilt from stored FP32 embedding copies
    once internal node usage reaches KVCACHE_HNSW_CAPACITY.
"""

from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import List
from std.math import sqrt
from std.ffi import external_call

from src.vector.hnsw import HNSWGraph
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.common.utils import format_int_to_buf, int_string_len
from src.common.utils import resolve_threshold

# Max live entries in the KV cache store per worker.
# Each entry is a (embedding, blob) pair. For 65 MB blobs, 100 entries = ~6.5 GB.
comptime KVCACHE_MAX_ENTRIES = 1000

# HNSW internal node capacity. Inserts append internal nodes (never reused), so
# evictions consume headroom; when num_nodes reaches this, the graph is rebuilt
# from the live entries' stored FP32 embeddings.
comptime KVCACHE_HNSW_CAPACITY = 2 * KVCACHE_MAX_ENTRIES

# Default embedding dimensions (768 for nomic-embed-text / typical sentence transformers).
# Overridden by config.
comptime KVCACHE_DEFAULT_DIM = 768

# Cosine similarity → INT8 L2 conversion constant for unit-norm embeddings.
# Same formula as SemanticCache: cos_sim ≈ 1 - L2_int8 / UNIT_NORM_SQ
comptime KVCACHE_UNIT_NORM_SQ = Float32(806450.0)

# Candidates examined per FETCH — must be > 1 so TTL-expired or model-mismatched
# nearest neighbors don't mask a valid second-best match.
comptime KVCACHE_FETCH_K = 8


@always_inline
def _kvcache_now_ns() -> Int64:
    """Current CLOCK_REALTIME in nanoseconds (same as fast_path._get_now_ns)."""
    var ts = alloc[Int64](2)
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    var result = ts[unsafe_offset=0] * Int64(1_000_000_000) + ts[unsafe_offset=1]
    ts.unsafe_free()
    return result


struct KVCacheEntry(TrivialRegisterPassable):
    """Metadata for a single cached KV tensor."""
    var blob_ptr: Pointer[UInt8, MutUntrackedOrigin]  # raw tensor bytes
    var blob_size: Int                                      # tensor byte length
    var cache_id_ptr: Pointer[UInt8, MutUntrackedOrigin]  # cache_id string bytes
    var cache_id_len: Int
    var model_tag_ptr: Pointer[UInt8, MutUntrackedOrigin]  # model family tag
    var model_tag_len: Int
    var embed_ptr: Pointer[Float32, MutUntrackedOrigin]  # FP32 embedding copy (for HNSW rebuild)
    var created_at: Int64                                    # unix timestamp (ns)
    var last_access: Int64                                   # unix timestamp (ns) — LRU clock
    var ttl_ns: Int64                                        # TTL in nanoseconds (0 = no expiry)
    var access_count: Int                                    # hit counter
    var active: Bool                                         # slot holds a live entry

    def __init__(out self):
        self.blob_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.blob_size = 0
        self.cache_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.cache_id_len = 0
        self.model_tag_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.model_tag_len = 0
        self.embed_ptr = null_ptr[Float32, MutUntrackedOrigin]()
        self.created_at = 0
        self.last_access = 0
        self.ttl_ns = 0
        self.access_count = 0
        self.active = False


struct KVCacheStore(Movable):
    """Per-worker KV cache store with HNSW-indexed prompt embeddings."""
    var hnsw: HNSWGraph
    var entries: Pointer[KVCacheEntry, MutUntrackedOrigin]  # slot array: entries[s], s = HNSW external id
    var count: Int                  # live entries
    var next_slot: Int              # high-water mark of slot allocation (≤ KVCACHE_MAX_ENTRIES)
    var free_slots: List[Int]       # reusable slots from eviction/expiry
    var dimensions: Int
    var threshold: Float32          # cosine similarity threshold (default 0.95)
    var enabled: Bool
    var total_blob_bytes: Int       # total bytes stored (for KV.INFO)
    var hits: Int                   # cache hit counter
    var misses: Int                 # cache miss counter
    var evictions: Int              # LRU evictions (for KV.INFO)
    var expired: Int                # TTL expiries reclaimed on FETCH (for KV.INFO)

    def __init__(out self, dimensions: Int = KVCACHE_DEFAULT_DIM,
                threshold: Float32 = Float32(0.95), enabled: Bool = False):
        self.dimensions = dimensions
        self.threshold = threshold
        self.enabled = enabled
        self.count = 0
        self.next_slot = 0
        self.free_slots = List[Int]()
        self.total_blob_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired = 0
        if enabled:
            self.hnsw = HNSWGraph(KVCACHE_HNSW_CAPACITY, dimensions, M=16, ef_construction=64)
            self.entries = alloc[KVCacheEntry](KVCACHE_MAX_ENTRIES)
            for idx in range(KVCACHE_MAX_ENTRIES):
                self.entries[unsafe_offset=idx] = KVCacheEntry()
        else:
            # Minimal init — no heavy allocations when disabled
            self.hnsw = HNSWGraph(1, 1)
            var _e = alloc[KVCacheEntry](1)
            self.entries = Pointer[KVCacheEntry, MutUntrackedOrigin](unsafe_from_address=Int(_e))

    def __moveinit__(out self, deinit take: Self):
        self.hnsw = take.hnsw^
        self.entries = take.entries
        self.count = take.count
        self.next_slot = take.next_slot
        self.free_slots = take.free_slots^
        self.dimensions = take.dimensions
        self.threshold = take.threshold
        self.enabled = take.enabled
        self.total_blob_bytes = take.total_blob_bytes
        self.hits = take.hits
        self.misses = take.misses
        self.evictions = take.evictions
        self.expired = take.expired

    def _release_slot(mut self, slot: Int):
        """Free a slot's heap memory, tombstone its HNSW node, and recycle the slot."""
        var internal = self.hnsw.node_map[unsafe_offset=slot]
        if internal >= 0:
            self.hnsw.unlink_node(internal)
            self.hnsw.node_map[unsafe_offset=slot] = -1
        var entry = self.entries[unsafe_offset=slot]
        self.total_blob_bytes -= entry.blob_size
        if is_not_null(entry.blob_ptr):
            entry.blob_ptr.unsafe_free()
        if is_not_null(entry.cache_id_ptr):
            entry.cache_id_ptr.unsafe_free()
        if entry.model_tag_len > 0:
            entry.model_tag_ptr.unsafe_free()
        if is_not_null(entry.embed_ptr):
            entry.embed_ptr.unsafe_free()
        self.entries[unsafe_offset=slot] = KVCacheEntry()
        self.free_slots.append(slot)
        self.count -= 1

    def _evict_lru(mut self):
        """Evict the least-recently-accessed live entry."""
        var lru_slot = -1
        var lru_ts = Int64(0)
        for s in range(self.next_slot):
            if not self.entries[unsafe_offset=s].active:
                continue
            if lru_slot < 0 or self.entries[unsafe_offset=s].last_access < lru_ts:
                lru_slot = s
                lru_ts = self.entries[unsafe_offset=s].last_access
        if lru_slot >= 0:
            self._release_slot(lru_slot)
            self.evictions += 1

    def _rebuild_index(mut self) raises:
        """Rebuild the HNSW graph from live entries' FP32 embedding copies.

        Internal node ids are append-only; tombstones from eviction consume
        capacity until a rebuild repacks the graph to `count` nodes.

        Inserts directly into the live graph (no compaction): nodes keep their
        per-vector slots in the HNSW vector allocator, which is stable for the
        graph's lifetime. compact_vectors() would repoint every node into a
        freshly-allocated buffer and free the old one mid-rebuild — see store().
        """
        self.hnsw.reset_index()
        for s in range(self.next_slot):
            if not self.entries[unsafe_offset=s].active:
                continue
            self.hnsw.insert_no_compact(s, self.entries[unsafe_offset=s].embed_ptr)

    def store(mut self, cache_id_ptr: Pointer[UInt8, MutUntrackedOrigin], cache_id_len: Int,
             embedding_ptr: Pointer[Float32, MutUntrackedOrigin],
             blob_ptr: Pointer[UInt8, MutUntrackedOrigin], blob_size: Int,
             model_ptr: Pointer[UInt8, MutUntrackedOrigin], model_len: Int,
             ttl_sec: Int) raises -> Bool:
        """Store a KV cache blob indexed by its prompt embedding. Returns True on success.

        At capacity, evicts the LRU entry (never rejects)."""
        if not self.enabled:
            return False
        var now = _kvcache_now_ns()

        if self.count >= KVCACHE_MAX_ENTRIES:
            self._evict_lru()
        if self.hnsw.num_nodes + 1 >= KVCACHE_HNSW_CAPACITY:
            self._rebuild_index()

        var slot: Int
        if len(self.free_slots) > 0:
            slot = self.free_slots.pop()
        else:
            slot = self.next_slot
            self.next_slot += 1

        # Insert embedding into HNSW (external id = slot).
        # Use insert_no_compact, NOT add_and_insert: the latter frees the old
        # compact_buffer before compact_vectors() re-reads every node's .vector
        # (which still points into that freed buffer), corrupting stored vectors
        # as the heap is reused — recall silently collapses past a few hundred
        # entries. The uncompacted graph keeps each node's vector in the stable
        # vector allocator; search reads it directly (dim=384 needs no norm
        # header), so no compaction is required for a bounded ≤1000-entry cache.
        self.hnsw.insert_no_compact(slot, embedding_ptr)

        # Copy blob bytes (we own this memory — caller's buffer may be reused)
        var blob_copy = alloc[UInt8](blob_size)
        unsafe_memcpy(dest=blob_copy, src=blob_ptr.unsafe_bitcast[UInt8](), count=blob_size)

        # Copy cache_id
        var id_copy = alloc[UInt8](cache_id_len)
        unsafe_memcpy(dest=id_copy, src=cache_id_ptr, count=cache_id_len)

        # Copy model tag
        var mt_copy = null_ptr[UInt8, MutUntrackedOrigin]()
        if model_len > 0:
            mt_copy = alloc[UInt8](model_len)
            unsafe_memcpy(dest=mt_copy, src=model_ptr, count=model_len)

        # Copy FP32 embedding (needed to re-insert on HNSW rebuild)
        var embed_copy = alloc[Float32](self.dimensions)
        unsafe_memcpy(dest=embed_copy.unsafe_bitcast[UInt8](), src=embedding_ptr.unsafe_bitcast[UInt8](),
               count=self.dimensions * 4)

        # Build entry — store pointers by reconstructing with the raw address
        var entry = KVCacheEntry()
        entry.blob_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(blob_copy))
        entry.blob_size = blob_size
        entry.cache_id_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(id_copy))
        entry.cache_id_len = cache_id_len
        entry.model_tag_ptr = mt_copy
        entry.model_tag_len = model_len
        entry.embed_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(embed_copy))
        entry.created_at = now
        entry.last_access = now
        entry.ttl_ns = Int64(ttl_sec) * 1_000_000_000 if ttl_sec > 0 else Int64(0)
        entry.access_count = 0
        entry.active = True

        self.entries[unsafe_offset=slot] = entry
        self.count += 1
        self.total_blob_bytes += blob_size
        return True

    def fetch(mut self, embedding_ptr: Pointer[Float32, MutUntrackedOrigin],
             threshold_override: Float32,
             model_ptr: Pointer[UInt8, MutUntrackedOrigin], model_len: Int,
             mut writer: ResponseWriter,
             server: TCPServer, fd: Int32, kq: Int32) raises -> Bool:
        """Search for the closest cached KV blob. Writes blob to writer on hit, returns True.

        TTL is enforced lazily (expired candidates are reclaimed and skipped).
        When model_len > 0, only entries with a byte-identical model tag match."""
        if not self.enabled or self.count == 0:
            self.misses += 1
            return False

        var now = _kvcache_now_ns()
        var ef = 32
        var scores = List[Float32]()
        var results = self.hnsw.search_fp32_scored(embedding_ptr, KVCACHE_FETCH_K, scores, ef)
        var thresh = resolve_threshold(threshold_override, self.threshold)   # gh #373

        for ri in range(len(results)):
            if ri >= len(scores):
                break
            var slot = results[ri]
            if slot < 0 or slot >= self.next_slot:
                continue
            if not self.entries[unsafe_offset=slot].active:
                continue

            # Results are sorted by distance — once similarity drops below the
            # threshold, no later candidate can pass.
            var cos_sim = Float32(1.0) - scores[ri] / KVCACHE_UNIT_NORM_SQ
            if cos_sim < thresh:
                break

            # Lazy TTL expiry
            var entry = self.entries[unsafe_offset=slot]
            if entry.ttl_ns > 0 and now - entry.created_at > entry.ttl_ns:
                self._release_slot(slot)
                self.expired += 1
                continue

            # MODEL filter
            if model_len > 0:
                if entry.model_tag_len != model_len:
                    continue
                var mismatch = False
                for bi in range(model_len):
                    if entry.model_tag_ptr[unsafe_offset=bi] != model_ptr[unsafe_offset=bi]:
                        mismatch = True
                        break
                if mismatch:
                    continue

            # Cache hit
            self.entries[unsafe_offset=slot].access_count += 1
            self.entries[unsafe_offset=slot].last_access = now
            self.hits += 1
            writer.append_bulk_string_response(entry.blob_ptr, entry.blob_size)
            return True

        self.misses += 1
        return False

    def info(self, mut writer: ResponseWriter) raises:
        """Write KV.INFO response with cache statistics."""
        # Build a simple key-value response as a bulk string
        var buf = alloc[UInt8](4096)
        var off = 0

        # entries
        var line1 = String("entries:") + String(self.count) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line1.unsafe_ptr(), count=line1.byte_length())
        off += line1.byte_length()

        # total_blob_bytes
        var line2 = String("total_blob_bytes:") + String(self.total_blob_bytes) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line2.unsafe_ptr(), count=line2.byte_length())
        off += line2.byte_length()

        # hits
        var line3 = String("hits:") + String(self.hits) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line3.unsafe_ptr(), count=line3.byte_length())
        off += line3.byte_length()

        # misses
        var line4 = String("misses:") + String(self.misses) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line4.unsafe_ptr(), count=line4.byte_length())
        off += line4.byte_length()

        # evictions
        var line_ev = String("evictions:") + String(self.evictions) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line_ev.unsafe_ptr(), count=line_ev.byte_length())
        off += line_ev.byte_length()

        # expired
        var line_ex = String("expired:") + String(self.expired) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line_ex.unsafe_ptr(), count=line_ex.byte_length())
        off += line_ex.byte_length()

        # hit_rate
        var total = self.hits + self.misses
        var rate_str: String
        if total > 0:
            var rate = Float64(self.hits) / Float64(total)
            rate_str = String("hit_rate:") + String(rate) + "\r\n"
        else:
            rate_str = String("hit_rate:0.0\r\n")
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=rate_str.unsafe_ptr(), count=rate_str.byte_length())
        off += rate_str.byte_length()

        # dimensions
        var line5 = String("dimensions:") + String(self.dimensions) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line5.unsafe_ptr(), count=line5.byte_length())
        off += line5.byte_length()

        # enabled
        var line6 = String("enabled:") + String(self.enabled) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line6.unsafe_ptr(), count=line6.byte_length())
        off += line6.byte_length()

        # capacity
        var line7 = String("capacity:") + String(KVCACHE_MAX_ENTRIES) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line7.unsafe_ptr(), count=line7.byte_length())
        off += line7.byte_length()

        writer.append_bulk_string_response(buf, off)
        buf.unsafe_free()
