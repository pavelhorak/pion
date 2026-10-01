"""SemanticRouter — M13: HNSW-indexed inference load balancer.

Routes queries to the inference node whose cached KV state / LoRA adapter
is semantically closest to the query. Uses an HNSW index over per-node
embedding centroids.

Commands:
  AI.ROUTE.REGISTER <node_id> <endpoint> <embedding_fp32> [CAPACITY <n>]
  AI.ROUTE.UPDATE <node_id> <new_embedding_fp32>
  AI.ROUTE <query_embedding_fp32> [EXCLUDE <node_id>]
  AI.ROUTE.REMOVE <node_id>
  AI.ROUTE.INFO
"""

from src.common.ptr import null_ptr, is_not_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import List

from src.vector.hnsw import HNSWGraph
from src.vector.kernels import dot_product_simd
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer

# Max inference nodes in the routing table
comptime MAX_ROUTE_NODES = 256
# AI.ROUTE.INFO response buffer, and the per-field clamp for node id /
# endpoint reads. Both exist because info() previously trusted an
# uninitialized length and wrote into a fixed buffer unchecked.
comptime ROUTE_INFO_BUF_SIZE = 8192
comptime ROUTE_INFO_MAX_FIELD = 256
# Embedding dimensions for centroids (same as EmbeddingConfig default)
comptime ROUTE_EMBED_DIM = 768
# Cosine similarity → INT8 L2 conversion (same as SemanticCache)
comptime ROUTE_UNIT_NORM_SQ = Float32(806450.0)


struct RouteNode(TrivialRegisterPassable):
    """An inference node registered in the routing table."""
    var active: Bool
    var node_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var node_id_len: Int
    var endpoint_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var endpoint_len: Int
    var capacity: Int           # max concurrent requests (0 = unlimited)
    var active_requests: Int    # current in-flight requests
    var total_routed: Int       # lifetime routed count
    var hnsw_internal_id: Int   # internal HNSW node index (-1 if not indexed)

    def __init__(out self):
        self.active = False
        self.node_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.node_id_len = 0
        self.endpoint_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.endpoint_len = 0
        self.capacity = 0
        self.active_requests = 0
        self.total_routed = 0
        self.hnsw_internal_id = -1


struct SemanticRouter(Movable):
    """HNSW-indexed semantic router for inference fleet management."""
    var hnsw: HNSWGraph
    var nodes: Pointer[RouteNode, MutUntrackedOrigin]
    var centroids_fp32: Pointer[Float32, MutUntrackedOrigin]  # [MAX_ROUTE_NODES * dim] FP32 centroids for brute-force
    var node_count: Int
    var dimensions: Int
    var enabled: Bool
    var total_queries: Int
    var total_hits: Int         # queries that found a matching node

    def __init__(out self, dimensions: Int = ROUTE_EMBED_DIM, enabled: Bool = False):
        self.dimensions = dimensions
        self.enabled = enabled
        self.node_count = 0
        self.total_queries = 0
        self.total_hits = 0
        if enabled:
            self.hnsw = HNSWGraph(MAX_ROUTE_NODES, dimensions, M=8, ef_construction=32)
            var _n = alloc[RouteNode](MAX_ROUTE_NODES)
            self.nodes = Pointer[RouteNode, MutUntrackedOrigin](unsafe_from_address=Int(_n))
            for i in range(MAX_ROUTE_NODES):
                self.nodes[unsafe_offset=i] = RouteNode()
            var _c = alloc[Float32](MAX_ROUTE_NODES * dimensions)
            self.centroids_fp32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_c))
        else:
            # A disabled router still has to survive being ASKED about. This
            # allocated ONE RouteNode and left it uninitialized, while info()
            # (and register/remove/route) iterate range(MAX_ROUTE_NODES) — so
            # entries 1..255 were out-of-bounds reads and entry 0 was garbage.
            # When that garbage `active` read truthy, info() took an
            # uninitialized `node_id_len` and ran
            # `for bi in range(node_id_len): nid += chr(...)`, i.e. an
            # effectively unbounded String append: `AI.ROUTE.INFO` on a default
            # server (no --kvcache/--inference) wedged the worker permanently —
            # the process stayed alive and listening but never answered another
            # command — and later died with SIGSEGV inside info().
            #
            # 256 RouteNodes is ~16 KB per worker, which is nothing next to a
            # remotely reachable hang, and it makes the disabled router
            # structurally identical to the enabled one for every reader.
            self.hnsw = HNSWGraph(1, 1)
            var _n = alloc[RouteNode](MAX_ROUTE_NODES)
            self.nodes = Pointer[RouteNode, MutUntrackedOrigin](unsafe_from_address=Int(_n))
            for i in range(MAX_ROUTE_NODES):
                self.nodes[unsafe_offset=i] = RouteNode()
            self.centroids_fp32 = null_ptr[Float32, MutUntrackedOrigin]()

    def __moveinit__(out self, deinit take: Self):
        self.hnsw = take.hnsw^
        self.nodes = take.nodes
        self.centroids_fp32 = take.centroids_fp32
        self.node_count = take.node_count
        self.dimensions = take.dimensions
        self.enabled = take.enabled
        self.total_queries = take.total_queries
        self.total_hits = take.total_hits

    def _find_node_by_id(self, node_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                         node_id_len: Int) -> Int:
        """Find node slot index by node_id string. Returns -1 if not found."""
        for i in range(MAX_ROUTE_NODES):
            if not self.nodes[unsafe_offset=i].active:
                continue
            if self.nodes[unsafe_offset=i].node_id_len != node_id_len:
                continue
            var mismatch = False
            for bi in range(node_id_len):
                if self.nodes[unsafe_offset=i].node_id_ptr[unsafe_offset=bi] != node_id_ptr[unsafe_offset=bi]:
                    mismatch = True
                    break
            if not mismatch:
                return i
        return -1

    def register_node(mut self,
                     node_id_ptr: Pointer[UInt8, MutUntrackedOrigin], node_id_len: Int,
                     endpoint_ptr: Pointer[UInt8, MutUntrackedOrigin], endpoint_len: Int,
                     embedding_ptr: Pointer[Float32, MutUntrackedOrigin],
                     capacity: Int) raises -> Bool:
        """Register a new inference node with its semantic centroid."""
        if not self.enabled or self.node_count >= MAX_ROUTE_NODES:
            return False

        # Check if node already exists
        var existing = self._find_node_by_id(node_id_ptr, node_id_len)
        if existing >= 0:
            return False  # already registered

        # Find free slot
        var slot = -1
        for i in range(MAX_ROUTE_NODES):
            if not self.nodes[unsafe_offset=i].active:
                slot = i
                break
        if slot < 0:
            return False

        # Copy strings
        var id_copy = alloc[UInt8](node_id_len)
        unsafe_memcpy(dest=id_copy, src=node_id_ptr, count=node_id_len)
        var ep_copy = alloc[UInt8](endpoint_len)
        unsafe_memcpy(dest=ep_copy, src=endpoint_ptr, count=endpoint_len)

        # Store FP32 centroid (for brute-force at small fleet sizes)
        unsafe_memcpy(dest=self.centroids_fp32.unsafe_offset(slot * self.dimensions), src=embedding_ptr, count=self.dimensions)

        # Also insert into HNSW (for large fleet O(log N) search)
        self.hnsw.add_and_insert(slot, embedding_ptr)

        var node = RouteNode()
        node.active = True
        node.node_id_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(id_copy))
        node.node_id_len = node_id_len
        node.endpoint_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(ep_copy))
        node.endpoint_len = endpoint_len
        node.capacity = capacity
        node.hnsw_internal_id = slot
        self.nodes[unsafe_offset=slot] = node
        self.node_count += 1
        return True

    def update_centroid(mut self,
                       node_id_ptr: Pointer[UInt8, MutUntrackedOrigin], node_id_len: Int,
                       embedding_ptr: Pointer[Float32, MutUntrackedOrigin]) raises -> Bool:
        """Update a node's semantic centroid (e.g., when KV cache changes)."""
        if not self.enabled:
            return False
        var slot = self._find_node_by_id(node_id_ptr, node_id_len)
        if slot < 0:
            return False

        # Re-insert into HNSW with updated embedding
        self.hnsw.add_and_insert(slot, embedding_ptr)
        return True

    def _dot_fp32(self, a: Pointer[Float32, MutUntrackedOrigin],
                  b: Pointer[Float32, MutUntrackedOrigin], dim: Int) -> Float32:
        """Compute dot product of two FP32 vectors (gh #120: SIMD dual-FMA)."""
        return dot_product_simd[8](a, b, dim)

    def route_query(mut self,
                   query_ptr: Pointer[Float32, MutUntrackedOrigin],
                   exclude_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   exclude_id_len: Int,
                   mut writer: ResponseWriter) raises -> Bool:
        """Route a query to the best inference node. Returns True if a node was found."""
        self.total_queries += 1
        if not self.enabled or self.node_count == 0:
            return False

        # For small fleet (<=16 nodes): brute-force FP32 cosine for perfect accuracy.
        # HNSW is overkill and lossy at INT8 with few nodes.
        # For large fleet (>16): use HNSW for O(log N) scalability.
        var best_slot = -1
        var best_score = Float32(-1e30)

        if self.node_count <= 16:
            # Brute-force FP32 cosine: compute dot product against all active centroids.
            # Much more accurate than HNSW INT8 at small node counts.
            for i in range(MAX_ROUTE_NODES):
                if not self.nodes[unsafe_offset=i].active:
                    continue
                if self.nodes[unsafe_offset=i].capacity > 0 and self.nodes[unsafe_offset=i].active_requests >= self.nodes[unsafe_offset=i].capacity:
                    continue
                # Check exclude
                if exclude_id_len > 0 and self.nodes[unsafe_offset=i].node_id_len == exclude_id_len:
                    var is_excluded = True
                    for bi in range(exclude_id_len):
                        if self.nodes[unsafe_offset=i].node_id_ptr[unsafe_offset=bi] != exclude_id_ptr[unsafe_offset=bi]:
                            is_excluded = False
                            break
                    if is_excluded:
                        continue

                # Cosine similarity = dot product (both unit-norm)
                var score = self._dot_fp32(query_ptr, self.centroids_fp32.unsafe_offset(i * self.dimensions), self.dimensions)
                if score > best_score:
                    best_score = score
                    best_slot = i

            if best_slot >= 0:
                self.nodes[unsafe_offset=best_slot].total_routed += 1
                self.total_hits += 1
                writer.append_bulk_string_response(self.nodes[unsafe_offset=best_slot].endpoint_ptr, self.nodes[unsafe_offset=best_slot].endpoint_len)
                return True
        else:
            # Large fleet: HNSW top-5 with filtering
            var k = min(5, self.node_count)
            var scores = List[Float32]()
            var results = self.hnsw.search_fp32_scored(query_ptr, k, scores, 32)

            for ri in range(len(results)):
                var slot = results[ri]
                if slot < 0 or slot >= MAX_ROUTE_NODES:
                    continue
                if not self.nodes[unsafe_offset=slot].active:
                    continue
                if self.nodes[unsafe_offset=slot].capacity > 0 and self.nodes[unsafe_offset=slot].active_requests >= self.nodes[unsafe_offset=slot].capacity:
                    continue
                if exclude_id_len > 0 and self.nodes[unsafe_offset=slot].node_id_len == exclude_id_len:
                    var is_excluded = True
                    for bi in range(exclude_id_len):
                        if self.nodes[unsafe_offset=slot].node_id_ptr[unsafe_offset=bi] != exclude_id_ptr[unsafe_offset=bi]:
                            is_excluded = False
                            break
                    if is_excluded:
                        continue

                self.nodes[unsafe_offset=slot].total_routed += 1
                self.total_hits += 1
                writer.append_bulk_string_response(self.nodes[unsafe_offset=slot].endpoint_ptr, self.nodes[unsafe_offset=slot].endpoint_len)
                return True

        return False

    def remove_node(mut self,
                   node_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   node_id_len: Int) -> Bool:
        """Remove a node from the routing table."""
        var slot = self._find_node_by_id(node_id_ptr, node_id_len)
        if slot < 0:
            return False
        self.nodes[unsafe_offset=slot].active = False
        self.node_count -= 1
        return True

    def info(self, mut writer: ResponseWriter) raises:
        """Write AI.ROUTE.INFO response."""
        var buf = alloc[UInt8](ROUTE_INFO_BUF_SIZE)
        var off = 0

        var line = String("nodes:") + String(self.node_count) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        line = String("total_queries:") + String(self.total_queries) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        line = String("total_hits:") + String(self.total_hits) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        var total = self.total_queries
        if total > 0:
            var rate = Float64(self.total_hits) / Float64(total)
            line = String("hit_rate:") + String(rate) + "\r\n"
        else:
            line = String("hit_rate:0.0\r\n")
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        line = String("dimensions:") + String(self.dimensions) + "\r\n"
        unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        # Per-node stats.
        #
        # Two bounds here, both load-bearing:
        #
        #  * The id/endpoint reads are clamped. They used to iterate whatever
        #    `node_id_len` held, so ONE bad length was an unbounded String
        #    append that wedged the worker (see __init__ — the disabled router
        #    left these uninitialized). The allocation bug is fixed, but a
        #    length is attacker-adjacent data and this loop should not be the
        #    thing that trusts it.
        #  * `buf` is 8192 bytes and was written with no capacity check at all.
        #    256 active nodes at ~80 bytes each is ~20 KB — a fully populated
        #    router would have run off the end of the allocation. Stop cleanly
        #    instead; a truncated INFO beats a heap overflow.
        for i in range(MAX_ROUTE_NODES):
            if not self.nodes[unsafe_offset=i].active:
                continue
            var idl = self.nodes[unsafe_offset=i].node_id_len
            if idl < 0: idl = 0
            if idl > ROUTE_INFO_MAX_FIELD: idl = ROUTE_INFO_MAX_FIELD
            var epl = self.nodes[unsafe_offset=i].endpoint_len
            if epl < 0: epl = 0
            if epl > ROUTE_INFO_MAX_FIELD: epl = ROUTE_INFO_MAX_FIELD
            # node:<id>:endpoint=<ep>,routed=<n>,active=<n>,capacity=<n>
            var nid = String("")
            if is_not_null(self.nodes[unsafe_offset=i].node_id_ptr):
                for bi in range(idl):
                    nid += chr(Int(self.nodes[unsafe_offset=i].node_id_ptr[unsafe_offset=bi]))
            var ep = String("")
            if is_not_null(self.nodes[unsafe_offset=i].endpoint_ptr):
                for bi in range(epl):
                    ep += chr(Int(self.nodes[unsafe_offset=i].endpoint_ptr[unsafe_offset=bi]))
            line = String("node:") + nid + ":endpoint=" + ep + ",routed=" + String(self.nodes[unsafe_offset=i].total_routed) + ",capacity=" + String(self.nodes[unsafe_offset=i].capacity) + "\r\n"
            if off + line.byte_length() > ROUTE_INFO_BUF_SIZE:
                break
            unsafe_memcpy(dest=buf.unsafe_offset(off), src=line.unsafe_ptr(), count=line.byte_length()); off += line.byte_length()

        writer.append_bulk_string_response(buf, off)
