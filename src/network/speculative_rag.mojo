"""SpeculativeRAG — M9: Branch prediction for the RAG pipeline.

Observes per-session query embedding trajectories and speculatively
pre-executes HNSW search for predicted next queries. When a predicted
query matches the actual query (cosine > threshold), the pre-computed
search results are returned instantly, skipping the full RAG pipeline.

Prediction strategy (MVP): Embedding momentum.
  predicted = normalize(last_query + alpha * (last_query - prev_query))
  Generates N predictions by varying alpha (0.5, 1.0, 1.5).
  This captures linear trajectory patterns (define → compare → measure).
  Production: replace with GRU over embedding trajectories.

Commands:
  RAG.SPECULATE.ENABLE <session_id> [DEPTH <n>] [THRESHOLD <cosine>]
  RAG.QUERY <session_id> <query_embedding_fp32> [K <k>]
  RAG.SPECULATE.INFO <session_id>
"""

from src.common.ptr import is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import List
from std.math import sqrt
from src.vector.fma_mad import fma_mad

from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.vector.kernels import dot_product_simd

# Max concurrent speculative sessions
comptime MAX_SPEC_SESSIONS = 64
# Max predictions per session
comptime MAX_PREDICTIONS = 5
# Max query history per session
comptime MAX_HISTORY = 10
# Default embedding dimensions
comptime SPEC_EMBED_DIM = 768
# Default cosine threshold for speculation hit
comptime DEFAULT_SPEC_THRESHOLD = Float32(0.90)
# Speculative cache entry TTL in ticks (evicted after this many queries without hit)
comptime SPEC_CACHE_TTL = 50


struct SpecCacheEntry(TrivialRegisterPassable):
    """A speculative pre-computed search result."""
    var predicted_embedding: Pointer[Float32, MutUntrackedOrigin]  # [dim] predicted query
    var result_ids: Pointer[Int32, MutUntrackedOrigin]             # [k] doc IDs from HNSW search
    var result_scores: Pointer[Float32, MutUntrackedOrigin]        # [k] distances
    var num_results: Int
    var active: Bool
    var age: Int  # ticks since creation (evicted when > TTL)

    def __init__(out self):
        self.predicted_embedding = null_ptr[Float32, MutUntrackedOrigin]()
        self.result_ids = null_ptr[Int32, MutUntrackedOrigin]()
        self.result_scores = null_ptr[Float32, MutUntrackedOrigin]()
        self.num_results = 0
        self.active = False
        self.age = 0


struct SpecSession(TrivialRegisterPassable):
    """Per-session speculation state."""
    var active: Bool
    var session_hash: UInt64
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: Int
    var dimensions: Int
    var threshold: Float32
    var depth: Int                                                       # number of predictions to generate
    # Query history: ring buffer of last MAX_HISTORY embeddings
    var history: Pointer[Float32, MutUntrackedOrigin]               # [MAX_HISTORY * dim]
    var history_count: Int
    var history_head: Int                                                 # index of most recent
    # Speculative cache: pre-computed search results for predicted queries
    var cache: Pointer[SpecCacheEntry, MutUntrackedOrigin]          # [MAX_PREDICTIONS]
    var cache_count: Int
    # Stats
    var total_queries: Int
    var spec_hits: Int
    var spec_misses: Int

    def __init__(out self):
        self.active = False
        self.session_hash = 0
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.dimensions = SPEC_EMBED_DIM
        self.threshold = DEFAULT_SPEC_THRESHOLD
        self.depth = 3
        self.history = null_ptr[Float32, MutUntrackedOrigin]()
        self.history_count = 0
        self.history_head = 0
        self.cache = null_ptr[SpecCacheEntry, MutUntrackedOrigin]()
        self.cache_count = 0
        self.total_queries = 0
        self.spec_hits = 0
        self.spec_misses = 0


struct SpeculativeRAG(Movable):
    """Speculative RAG pipeline manager."""
    var sessions: Pointer[SpecSession, MutUntrackedOrigin]
    var session_count: Int
    var dimensions: Int
    var enabled: Bool
    var total_hits: Int
    var total_misses: Int
    var total_predictions: Int

    def __init__(out self, dimensions: Int = SPEC_EMBED_DIM, enabled: Bool = False):
        self.dimensions = dimensions
        self.enabled = enabled
        self.session_count = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_predictions = 0
        var n = MAX_SPEC_SESSIONS if enabled else 1
        var _s = alloc[SpecSession](n)
        self.sessions = Pointer[SpecSession, MutUntrackedOrigin](unsafe_from_address=Int(_s))
        for i in range(n):
            self.sessions[unsafe_offset=i] = SpecSession()

    def __moveinit__(out self, deinit take: Self):
        self.sessions = take.sessions
        self.session_count = take.session_count
        self.dimensions = take.dimensions
        self.enabled = take.enabled
        self.total_hits = take.total_hits
        self.total_misses = take.total_misses
        self.total_predictions = take.total_predictions

    def _hash_bytes(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt64:
        var h = UInt64(0x517cc1b727220a95)
        for bi in range(length):
            h = (h ^ UInt64(ptr[unsafe_offset=bi])) * UInt64(0x9e3779b97f4a7c15)
        return h

    def _find_session(self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int) -> Int:
        var h = self._hash_bytes(sid_ptr, sid_len)
        for si in range(MAX_SPEC_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                continue
            if self.sessions[unsafe_offset=si].session_hash != h:
                continue
            if self.sessions[unsafe_offset=si].session_id_len != sid_len:
                continue
            var mismatch = False
            for bi in range(sid_len):
                if self.sessions[unsafe_offset=si].session_id_ptr[unsafe_offset=bi] != sid_ptr[unsafe_offset=bi]:
                    mismatch = True
                    break
            if not mismatch:
                return si
        return -1

    def _dot(self, a: Pointer[Float32, MutUntrackedOrigin],
            b: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
        # gh #120: SIMD dual-FMA dot product (was a scalar per-dim loop).
        return dot_product_simd[8](a, b, dim)

    def _norm(self, v: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
        return sqrt(self._dot(v, v, dim))

    def _cosine(self, a: Pointer[Float32, MutUntrackedOrigin],
               b: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
        var d = self._dot(a, b, dim)
        var na = self._norm(a, dim)
        var nb = self._norm(b, dim)
        if na < Float32(1e-8) or nb < Float32(1e-8):
            return Float32(0.0)
        return d / (na * nb)

    def enable_session(mut self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int,
                      depth: Int, threshold: Float32) raises -> Bool:
        """Enable speculation for a session."""
        if not self.enabled:
            return False
        # Check if already exists
        if self._find_session(sid_ptr, sid_len) >= 0:
            return True  # already enabled

        # Find free slot
        for si in range(MAX_SPEC_SESSIONS):
            if self.sessions[unsafe_offset=si].active:
                continue

            var id_copy = alloc[UInt8](sid_len)
            unsafe_memcpy(dest=id_copy, src=sid_ptr, count=sid_len)

            var dim = self.dimensions
            var hist = alloc[Float32](MAX_HISTORY * dim)
            var cache = alloc[SpecCacheEntry](MAX_PREDICTIONS)
            var cache_ptr = Pointer[SpecCacheEntry, MutUntrackedOrigin](unsafe_from_address=Int(cache))
            for ci in range(MAX_PREDICTIONS):
                cache_ptr[unsafe_offset=ci] = SpecCacheEntry()

            var sess = SpecSession()
            sess.active = True
            sess.session_hash = self._hash_bytes(sid_ptr, sid_len)
            sess.session_id_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(id_copy))
            sess.session_id_len = sid_len
            sess.dimensions = dim
            sess.threshold = threshold if threshold > Float32(0.0) else DEFAULT_SPEC_THRESHOLD
            sess.depth = depth if depth > 0 else 3
            sess.history = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(hist))
            sess.cache = cache_ptr
            self.sessions[unsafe_offset=si] = sess
            self.session_count += 1
            return True
        return False

    def _add_to_history(mut self, si: Int, embedding: Pointer[Float32, MutUntrackedOrigin]):
        """Add a query embedding to the session's history ring buffer."""
        var dim = self.sessions[unsafe_offset=si].dimensions
        var idx = self.sessions[unsafe_offset=si].history_count % MAX_HISTORY
        unsafe_memcpy(dest=self.sessions[unsafe_offset=si].history.unsafe_offset(idx * dim), src=embedding, count=dim)
        self.sessions[unsafe_offset=si].history_count += 1
        self.sessions[unsafe_offset=si].history_head = idx

    def _get_history(self, si: Int, steps_back: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        """Get a historical embedding (0 = most recent, 1 = one before, etc.)."""
        var count = self.sessions[unsafe_offset=si].history_count
        if steps_back >= count or steps_back >= MAX_HISTORY:
            return null_ptr[Float32, MutUntrackedOrigin]()
        var idx = (self.sessions[unsafe_offset=si].history_head - steps_back + MAX_HISTORY) % MAX_HISTORY
        return self.sessions[unsafe_offset=si].history.unsafe_offset(idx * self.sessions[unsafe_offset=si].dimensions)

    def _generate_predictions(mut self, si: Int):
        """Generate speculative predictions from the query trajectory.

        MVP: Embedding momentum — predict by extrapolating the direction
        from the previous query to the current query.

        predicted[i] = normalize(current + alpha[i] * (current - previous))

        With alphas [0.5, 1.0, 1.5] this generates 3 predictions:
        - Conservative (half step forward)
        - Linear (same step forward)
        - Aggressive (1.5x step forward)
        """
        var count = self.sessions[unsafe_offset=si].history_count
        if count < 2:
            return  # need at least 2 queries for trajectory

        var dim = self.sessions[unsafe_offset=si].dimensions
        var current = self._get_history(si, 0)
        var previous = self._get_history(si, 1)
        if is_null(current) or is_null(previous):
            return

        var depth = self.sessions[unsafe_offset=si].depth
        if depth > MAX_PREDICTIONS:
            depth = MAX_PREDICTIONS

        # Clear old predictions
        for ci in range(MAX_PREDICTIONS):
            self.sessions[unsafe_offset=si].cache[unsafe_offset=ci].active = False
        self.sessions[unsafe_offset=si].cache_count = 0

        # Generate predictions with different momentum alphas
        var alphas = List[Float32]()
        alphas.append(Float32(0.5))
        alphas.append(Float32(1.0))
        alphas.append(Float32(1.5))
        if depth > 3:
            alphas.append(Float32(2.0))
        if depth > 4:
            alphas.append(Float32(0.25))

        for pi in range(min(depth, len(alphas))):
            var alpha = alphas[pi]
            # Allocate prediction embedding
            var pred = alloc[Float32](dim)
            var pred_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(pred))

            # predicted = current + alpha * (current - previous); accumulate ‖pred‖².
            # gh #120: SIMD-8 — fused delta/predict/norm in one vector pass.
            comptime W = 8
            var alpha_v = SIMD[DType.float32, W](alpha)
            var nsq_v = SIMD[DType.float32, W](0.0)
            var norm_sq = Float32(0.0)
            var d = 0
            while d + W <= dim:
                var cur = current.load[width=W](d)
                var p = cur + alpha_v * (cur - previous.load[width=W](d))
                pred_ptr.store(d, p)
                nsq_v = fma_mad[W](p, p, nsq_v)
                d += W
            norm_sq = nsq_v.reduce_add()
            while d < dim:  # scalar tail (empty for dim % 8 == 0)
                var delta = current[unsafe_offset=d] - previous[unsafe_offset=d]
                pred_ptr[unsafe_offset=d] = current[unsafe_offset=d] + alpha * delta
                norm_sq += pred_ptr[unsafe_offset=d] * pred_ptr[unsafe_offset=d]
                d += 1

            # Normalize to unit norm
            var norm_val = sqrt(norm_sq)
            if norm_val > Float32(1e-8):
                var nv = SIMD[DType.float32, W](norm_val)
                var d2 = 0
                while d2 + W <= dim:
                    pred_ptr.store(d2, pred_ptr.load[width=W](d2) / nv)
                    d2 += W
                while d2 < dim:
                    pred_ptr[unsafe_offset=d2] = pred_ptr[unsafe_offset=d2] / norm_val
                    d2 += 1

            var entry = SpecCacheEntry()
            entry.predicted_embedding = pred_ptr
            entry.active = True
            entry.age = 0
            entry.num_results = 0  # results filled by external HNSW search
            # Allocate result buffers (filled by the caller after HNSW search)
            var rids = alloc[Int32](64)  # max k=64
            entry.result_ids = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(rids))
            var rscores = alloc[Float32](64)
            entry.result_scores = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(rscores))

            self.sessions[unsafe_offset=si].cache[unsafe_offset=pi] = entry
            self.sessions[unsafe_offset=si].cache_count += 1
            self.total_predictions += 1

    def check_speculation(mut self, si: Int,
                         query_embedding: Pointer[Float32, MutUntrackedOrigin]) -> Int:
        """Check if any speculative prediction matches the actual query.
        Returns the cache entry index (0..MAX_PREDICTIONS-1) on hit, -1 on miss."""
        var dim = self.sessions[unsafe_offset=si].dimensions
        var threshold = self.sessions[unsafe_offset=si].threshold

        for ci in range(MAX_PREDICTIONS):
            if not self.sessions[unsafe_offset=si].cache[unsafe_offset=ci].active:
                continue
            var cos = self._cosine(query_embedding, self.sessions[unsafe_offset=si].cache[unsafe_offset=ci].predicted_embedding, dim)
            if cos >= threshold:
                return ci

        return -1

    def process_query(mut self, si: Int,
                     query_embedding: Pointer[Float32, MutUntrackedOrigin]) raises -> Int:
        """Process a query: check speculation, update history, generate new predictions.

        Returns cache entry index on hit, -1 on miss.
        Always updates history and regenerates predictions for next query."""
        self.sessions[unsafe_offset=si].total_queries += 1

        # Check speculation
        var hit = self.check_speculation(si, query_embedding)

        if hit >= 0:
            self.sessions[unsafe_offset=si].spec_hits += 1
            self.total_hits += 1
        else:
            self.sessions[unsafe_offset=si].spec_misses += 1
            self.total_misses += 1

        # Add to history
        self._add_to_history(si, query_embedding)

        # Generate predictions for next query
        self._generate_predictions(si)
        return hit

    def get_prediction_embedding(self, si: Int, cache_idx: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        """Get a prediction's embedding for external HNSW search."""
        if cache_idx < 0 or cache_idx >= MAX_PREDICTIONS:
            return null_ptr[Float32, MutUntrackedOrigin]()
        if not self.sessions[unsafe_offset=si].cache[unsafe_offset=cache_idx].active:
            return null_ptr[Float32, MutUntrackedOrigin]()
        return self.sessions[unsafe_offset=si].cache[unsafe_offset=cache_idx].predicted_embedding

    def store_prediction_results(mut self, si: Int, cache_idx: Int,
                                result_ids: Pointer[Int32, MutUntrackedOrigin],
                                result_scores: Pointer[Float32, MutUntrackedOrigin],
                                num_results: Int):
        """Store HNSW search results for a prediction (called after external search)."""
        if cache_idx < 0 or cache_idx >= MAX_PREDICTIONS:
            return
        unsafe_memcpy(dest=self.sessions[unsafe_offset=si].cache[unsafe_offset=cache_idx].result_ids, src=result_ids, count=num_results)
        unsafe_memcpy(dest=self.sessions[unsafe_offset=si].cache[unsafe_offset=cache_idx].result_scores, src=result_scores, count=num_results)
        self.sessions[unsafe_offset=si].cache[unsafe_offset=cache_idx].num_results = num_results
