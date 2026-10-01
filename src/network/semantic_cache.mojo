"""SemanticCache — per-worker AI semantic cache backed by a small HNSWGraph.

Commands:
  AI.SEMANTIC_CACHE SET <query> <response>          — embed query, insert into HNSW, store response
  AI.SEMANTIC_CACHE GET <query> [THRESHOLD <t>]     — embed query, search HNSW, return cached response if similar enough

Similarity metric:
  Uses INT8 L2 distance from the HNSW graph.  For unit-norm embeddings (OpenAI / nomic-embed-text)
  quantized to [-0.20, 0.20] with scale=635, the INT8 L2 distance approximates cosine distance by:

    cos_sim ≈ 1 - L2_int8 / (2 * scale² * ||v||²)
            ≈ 1 - L2_int8 / 806450           (for unit-norm vectors)

  The config threshold (0–1 cosine similarity) is converted to an INT8 L2 threshold internally.
  Default threshold = 0.95, which corresponds to INT8 L2 ≈ 40322.

Capacity: 10,000 entries per worker (configurable via CACHE_MAX_ENTRIES comptime).
"""

from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.collections import List, Span
from std.math import sqrt

from src.vector.hnsw import HNSWGraph
from src.network.embedding_client import EmbeddingClient
from src.network.nle_embedding_engine import NLEmbeddingEngine
from src.network.inference_bridge import InferenceBridge
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.common.utils import resolve_threshold

comptime CACHE_MAX_ENTRIES = 10000

# Approximate INT8 squared-norm for unit-norm embeddings in [-0.20, 0.20] quantized with scale=635.
# scale² = 635² = 403225.  For unit-norm v: ||v_int8||² ≈ scale² = 403225.
# INT8 L2 = ||q_int8||² + ||v_int8||² - 2*dot(q,v)
#         ≈ 2 * scale² * (1 - cos_sim) = 806450 * (1 - cos_sim)
comptime CACHE_UNIT_NORM_SQ = Float32(806450.0)


struct SemanticCache(Movable):
    var hnsw: HNSWGraph
    var client: EmbeddingClient
    var responses: List[String]     # responses[i] = cached response for entry i
    # gh #115: optional audit trail — the caller's workspace snapshot (J-lens
    # top-k, base64) recorded at cache-WRITE time, so a hit can be explained
    # later without any inference infrastructure. Opaque to the server.
    var workspaces: List[String]
    var count: Int
    var threshold: Float32          # cosine similarity threshold (0–1)
    var dimensions: Int
    var embed_buf: Pointer[Float32, MutUntrackedOrigin]   # per-worker scratch for embeddings
    var enabled: Bool
    var bridge: Pointer[InferenceBridge, MutUntrackedOrigin]  # A3: optional inference bridge for auto-embed
    var nle: NLEmbeddingEngine                                      # Apple NLEmbedding (macOS) — preferred when available
    # gh #140: instruction prefixes for asymmetric retrievers. Empty = disabled.
    var query_prefix: String
    var doc_prefix: String
    var prefix_scratch: Pointer[UInt8, MutUntrackedOrigin]
    var prefix_scratch_cap: Int
    # gh #262: value receipt. Counted INSIDE cache_get so every caller
    # (AI.SEMANTIC_CACHE GET, AI.MEMORY RECALL, the gateway) is covered.
    var hits: UInt64
    var misses: UInt64

    def __init__(out self, host: String, port: Int, model: String, dimensions: Int,
                threshold: Float32, enabled: Bool, nle_enabled: Bool = False,
                query_prefix: String = "", doc_prefix: String = ""):
        self.client = EmbeddingClient(host, port, model, dimensions)
        self.hnsw = HNSWGraph(CACHE_MAX_ENTRIES, dimensions, M=16, ef_construction=32)
        self.responses = List[String]()
        self.workspaces = List[String]()
        self.count = 0
        self.threshold = threshold
        self.dimensions = dimensions
        self.embed_buf = alloc[Float32](dimensions)
        self.enabled = enabled
        self.bridge = null_ptr[InferenceBridge, MutUntrackedOrigin]()
        self.nle = NLEmbeddingEngine(nle_enabled)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.prefix_scratch = null_ptr[UInt8, MutUntrackedOrigin]()
        self.prefix_scratch_cap = 0
        self.hits = UInt64(0)
        self.misses = UInt64(0)

    def __moveinit__(out self, deinit take: Self):
        self.hnsw = take.hnsw^
        self.client = take.client^
        self.responses = take.responses^
        self.workspaces = take.workspaces^
        self.count = take.count
        self.threshold = take.threshold
        self.dimensions = take.dimensions
        self.embed_buf = take.embed_buf
        self.enabled = take.enabled
        self.bridge = take.bridge
        self.nle = take.nle^
        self.query_prefix = take.query_prefix^
        self.doc_prefix = take.doc_prefix^
        self.prefix_scratch = take.prefix_scratch
        self.prefix_scratch_cap = take.prefix_scratch_cap
        self.hits = take.hits
        self.misses = take.misses

    @always_inline
    def set_bridge(mut self, bridge_ptr: Pointer[InferenceBridge, MutUntrackedOrigin]):
        """A3: Set inference bridge for auto-embedding (called after worker init)."""
        self.bridge = bridge_ptr

    @always_inline
    def embed_into(mut self,
                   text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int,
                   out_buf: Pointer[Float32, MutUntrackedOrigin],
                   is_query: Bool = True) -> Bool:
        """Public embed cascade — fills *caller's* buffer at self.dimensions.
        Cascade: NLEmbedding (Mac native, ANE) → InferenceBridge (PyTorch sidecar)
        → HTTP EmbeddingClient (Ollama / OpenAI / MAX). Use this from any
        embedding call site; keeps the cascade in one place so --nle-embed
        flips on for every embed-using surface, not just AI.SEMANTIC_CACHE.

        gh #140: `is_query` picks the instruction prefix for asymmetric
        retrievers. Indexing a document uses the document prefix; everything
        else (search, semantic cache, memory recall) is a query. Both prefixes
        default to empty, so symmetric models are untouched."""
        # gh #140: prepend the instruction prefix into a reusable scratch buffer.
        # Only pays an allocation when a prefix is actually configured, and only
        # grows — FT.ADDTEXT/FT.SEARCHTEXT are slow-path commands.
        var pfx = self.query_prefix if is_query else self.doc_prefix
        var pfx_len = pfx.byte_length()
        if pfx_len > 0:
            var need = pfx_len + text_len
            if need > self.prefix_scratch_cap:
                if is_not_null(self.prefix_scratch):
                    self.prefix_scratch.unsafe_free()
                var newcap = need * 2
                self.prefix_scratch = alloc[UInt8](newcap)
                self.prefix_scratch_cap = newcap
            unsafe_memcpy(dest=self.prefix_scratch, src=pfx.unsafe_ptr(), count=pfx_len)
            unsafe_memcpy(dest=self.prefix_scratch.unsafe_offset(pfx_len), src=text_ptr, count=text_len)
            return self._embed_cascade(self.prefix_scratch, need, out_buf)
        return self._embed_cascade(text_ptr, text_len, out_buf)

    @always_inline
    def _embed_cascade(mut self,
                       text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int,
                       out_buf: Pointer[Float32, MutUntrackedOrigin]) -> Bool:
        # Apple NLEmbedding — Mac-only, no Python, ANE-accelerated.
        # Only valid if the cache was sized to the NLE dimension (512).
        if self.nle.available and Int(self.nle.dimension) == self.dimensions:
            var dim = self.nle.embed(text_ptr, text_len, out_buf, self.dimensions)
            if dim == self.dimensions:
                return True
            # NLE didn't recognize the text → fall through.
        # Try InferenceBridge (PyTorch sidecar)
        if is_not_null(self.bridge):
            if self.bridge[].connected:
                var dim = self.bridge[].embed_blocking(text_ptr, text_len, out_buf)
                if dim == self.dimensions:
                    return True
                # dim mismatch or failure — fall through to HTTP client
        # Fallback: HTTP EmbeddingClient (Ollama/OpenAI/MAX)
        return self.client.embed(text_ptr, text_len, out_buf)

    @always_inline
    def _embed(mut self, text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int) -> Bool:
        """Internal: embed into self.embed_buf. Used by cache_set / cache_get.
        External callers should use embed_into() with their own buffer."""
        return self.embed_into(text_ptr, text_len, self.embed_buf)

    def cache_set(mut self, query_ptr: Pointer[UInt8, MutUntrackedOrigin], query_len: Int,
                 response_ptr: Pointer[UInt8, MutUntrackedOrigin], response_len: Int,
                 ws_ptr: Pointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin](),
                 ws_len: Int = 0) raises -> Int:
        """Embed query and store response.

        gh #421: returns a status so the caller can reply honestly instead of
        answering +OK when nothing was stored — 0 stored, 1 embedding failed
        (backend unavailable), 2 cache full, 3 semantic cache disabled."""
        if not self.enabled: return 3
        if self.count >= CACHE_MAX_ENTRIES: return 2
        if not self._embed(query_ptr, query_len): return 1

        self.hnsw.add_and_insert(self.count, self.embed_buf)
        # gh #232/#115: this used to spell every byte >= 128 as '?', ONE PER
        # BYTE — so a UTF-8 response came back mangled and longer: "café"
        # stored as "caf??". This is a cache for MODEL OUTPUT, where curly
        # quotes, accents, dashes and emoji are the norm rather than the
        # exception, so the common case was the corrupted one. Bulk-construct
        # from the bytes instead, exactly as RESP3Token.value() does for its
        # all-ASCII path — the String copies them out, so no borrow into the
        # recv buffer survives.
        var resp = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=response_ptr, length=response_len)))
        self.responses.append(resp)
        # gh #115: keep `workspaces` index-aligned with `responses` even though
        # OTHER callers (MCP memory in ai.mojo, RAG ingest in vector.mojo)
        # append to `responses` directly and know nothing about workspaces.
        # Padding here self-heals rather than requiring all five append sites to
        # stay in step — the failure mode otherwise is EXPLAIN silently
        # returning some other entry's audit trail, which is worse than none.
        while len(self.workspaces) < self.count: self.workspaces.append(String(""))
        if ws_len > 0 and is_not_null(ws_ptr):
            self.workspaces.append(String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                    unsafe_ptr=ws_ptr, length=ws_len))))
        else:
            self.workspaces.append(String(""))
        self.count += 1
        return 0

    def workspace_at(self, idx: Int) -> String:
        """Stored workspace for an entry, or empty when absent (gh #115)."""
        if idx < 0 or idx >= len(self.workspaces): return String("")
        return self.workspaces[idx]

    def cache_lookup(mut self, query_ptr: Pointer[UInt8, MutUntrackedOrigin], query_len: Int,
                     threshold_override: Float32) raises -> Int:
        """Index of the best match above threshold, or -1 (gh #115).

        Factored out of `cache_get` so EXPLAIN can resolve a query to an entry
        without also emitting the cached response."""
        if not self.enabled or self.count == 0: return -1
        if not self._embed(query_ptr, query_len): return -1
        var ef = 32
        var scores = List[Float32]()
        var results = self.hnsw.search_fp32_scored(self.embed_buf, 1, scores, ef)
        if len(results) == 0 or len(scores) == 0: return -1
        var cos_sim = Float32(1.0) - scores[0] / CACHE_UNIT_NORM_SQ
        var thresh = resolve_threshold(threshold_override, self.threshold)   # gh #373
        if cos_sim < thresh: return -1
        var hit_id = results[0]
        if hit_id < 0 or hit_id >= self.count: return -1
        return hit_id

    def cache_get(mut self, query_ptr: Pointer[UInt8, MutUntrackedOrigin], query_len: Int,
                 threshold_override: Float32, mut writer: ResponseWriter,
                 server: TCPServer, fd: Int32, kq: Int32,
                 with_workspace: Bool = False) raises -> Bool:
        """Search cache for a similar query.  Writes response to writer and returns True on hit."""
        if not self.enabled: return False
        if self.count == 0:
            self.misses += UInt64(1)
            return False
        if not self._embed(query_ptr, query_len): return False

        var ef = 32   # small ef: cache is small, speed is key
        var scores = List[Float32]()
        var results = self.hnsw.search_fp32_scored(self.embed_buf, 1, scores, ef)
        if len(results) == 0 or len(scores) == 0:
            self.misses += UInt64(1)
            return False

        # Convert INT8 L2² to approximate cosine similarity for unit-norm embeddings.
        var l2_int8 = scores[0]
        var cos_sim = Float32(1.0) - l2_int8 / CACHE_UNIT_NORM_SQ
        var thresh = resolve_threshold(threshold_override, self.threshold)   # gh #373
        if cos_sim < thresh:
            self.misses += UInt64(1)
            return False

        # Cache hit: write stored response
        var hit_id = results[0]
        if hit_id < 0 or hit_id >= self.count:
            self.misses += UInt64(1)
            return False
        var resp = self.responses[hit_id]
        # gh #115: WITHWORKSPACE is OPT-IN and the default reply shape is
        # unchanged. The issue proposed always returning a RESP3 map, but that
        # would break every RESP3 client already reading GET as a bulk string —
        # a compatibility break to add observability is the wrong trade.
        if with_workspace:
            var ws = self.workspace_at(hit_id)
            writer.append_map_header(2)
            writer.append_bulk_string_response("response".unsafe_ptr(), 8)
            writer.append_bulk_string_response(resp.unsafe_ptr(), resp.byte_length())
            writer.append_bulk_string_response("workspace".unsafe_ptr(), 9)
            if ws.byte_length() > 0:
                writer.append_bulk_string_response(ws.unsafe_ptr(), ws.byte_length())
            else:
                writer.append_null_response()
        else:
            writer.append_bulk_string_response(resp.unsafe_ptr(), resp.byte_length())
        self.hits += UInt64(1)
        return True
