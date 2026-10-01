"""LayerStore — per-worker store for individual KV cache layers.

Phase 2 of M14 (Externalized Attention). Stores per-layer KV tensors
for selective layer offloading from GPU to Pion.

Sessions hold per-layer tensor blobs. The inference engine stores layers
after prefill and fetches them during decode for externalized layers.

Storage: mmap'd arena with bump allocation. Sessions are evicted by TTL
or manual deletion.
"""

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.collections import List
from std.ffi import external_call

from src.common.hash_map import SlabHashMap
from src.common.value import GenericValue, ValueType

# Maximum concurrent sessions
comptime MAX_SESSIONS = 256
# Maximum layers per session (covers models up to 200 layers)
comptime MAX_LAYERS = 200
# Default arena size: 8 GB (configurable)
comptime DEFAULT_ARENA_SIZE = 8 * 1024 * 1024 * 1024


@always_inline
def _layerstore_now_ns() -> Int64:
    """Current CLOCK_REALTIME in nanoseconds (LRU clock; same as kv_cache_store)."""
    var ts = alloc[Int64](2)
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    var result = ts[unsafe_offset=0] * Int64(1_000_000_000) + ts[unsafe_offset=1]
    ts.unsafe_free()
    return result


struct LayerEntry(TrivialRegisterPassable):
    """One layer's KV tensor within a session."""
    var offset: Int     # byte offset within the session's arena allocation
    var size: Int       # tensor byte length
    var stored: Bool    # True if this layer has been stored

    def __init__(out self):
        self.offset = 0
        self.size = 0
        self.stored = False


struct SessionEntry(TrivialRegisterPassable):
    """Metadata for a single inference session's KV cache layers."""
    var session_hash: UInt64           # hash of session_id for fast lookup
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: Int
    var num_layers: Int                # total layers in this session
    var blob_ptr: Pointer[UInt8, MutUntrackedOrigin]  # base pointer for all layer tensors
    var blob_capacity: Int             # total bytes allocated for this session
    var blob_used: Int                 # bytes used so far
    var active: Bool                   # True if session is active
    var created_at: Int64              # timestamp
    var last_access: Int64             # timestamp of last access
    var ttl_sec: Int                   # TTL in seconds (0 = no expiry)
    # Per-layer metadata (stored in a separate flat array indexed by session_idx * MAX_LAYERS + layer_id)

    def __init__(out self):
        self.session_hash = 0
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.num_layers = 0
        self.blob_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.blob_capacity = 0
        self.blob_used = 0
        self.active = False
        self.created_at = 0
        self.last_access = 0
        self.ttl_sec = 3600  # 1 hour default


struct LayerStore(Movable):
    """Per-worker layer-granular KV cache store."""

    var sessions: Pointer[SessionEntry, MutUntrackedOrigin]
    var layers: Pointer[LayerEntry, MutUntrackedOrigin]  # [MAX_SESSIONS * MAX_LAYERS]
    var session_count: Int
    var enabled: Bool

    # Stats
    var total_bytes_stored: Int
    var store_count: Int
    var fetch_count: Int
    var fetch_hits: Int
    var fetch_misses: Int
    var evictions: Int              # LRU session evictions when MAX_SESSIONS is full

    def __init__(out self, enabled: Bool = False):
        self.enabled = enabled
        self.session_count = 0
        self.total_bytes_stored = 0
        self.store_count = 0
        self.fetch_count = 0
        self.fetch_hits = 0
        self.fetch_misses = 0
        self.evictions = 0

        var n_sess = MAX_SESSIONS if enabled else 1
        var n_layers = n_sess * MAX_LAYERS
        var _s = alloc[SessionEntry](n_sess)
        self.sessions = Pointer[SessionEntry, MutUntrackedOrigin](unsafe_from_address=Int(_s))
        var _l = alloc[LayerEntry](n_layers)
        self.layers = Pointer[LayerEntry, MutUntrackedOrigin](unsafe_from_address=Int(_l))

        for si in range(n_sess):
            self.sessions[unsafe_offset=si] = SessionEntry()
        for li in range(n_layers):
            self.layers[unsafe_offset=li] = LayerEntry()

    def __moveinit__(out self, deinit take: Self):
        self.sessions = take.sessions
        self.layers = take.layers
        self.session_count = take.session_count
        self.enabled = take.enabled
        self.total_bytes_stored = take.total_bytes_stored
        self.store_count = take.store_count
        self.fetch_count = take.fetch_count
        self.fetch_hits = take.fetch_hits
        self.fetch_misses = take.fetch_misses
        self.evictions = take.evictions

    def _find_session(self, session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                     session_id_len: Int) -> Int:
        """Find session index by ID. Returns -1 if not found."""
        var h = self._hash_bytes(session_id_ptr, session_id_len)
        for si in range(MAX_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                continue
            if self.sessions[unsafe_offset=si].session_hash != h:
                continue
            if self.sessions[unsafe_offset=si].session_id_len != session_id_len:
                continue
            var mismatch = False
            for bi in range(session_id_len):
                if self.sessions[unsafe_offset=si].session_id_ptr[unsafe_offset=bi] != session_id_ptr[unsafe_offset=bi]:
                    mismatch = True
                    break
            if not mismatch:
                return si
        return -1

    def _init_session_slot(mut self, si: Int,
                          session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          session_id_len: Int, now: Int64) -> Int:
        """Populate an inactive slot `si` with a fresh session. Returns `si`."""
        var entry = SessionEntry()
        entry.session_hash = self._hash_bytes(session_id_ptr, session_id_len)
        var id_copy = alloc[UInt8](session_id_len)
        unsafe_memcpy(dest=id_copy, src=session_id_ptr, count=session_id_len)
        entry.session_id_ptr = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(id_copy)
        )
        entry.session_id_len = session_id_len
        entry.active = True
        entry.created_at = now
        entry.last_access = now
        # Allocate initial blob space (64 MB per session — enough for ~16 layers of 4MB each)
        var initial_cap = 64 * 1024 * 1024
        var blob = alloc[UInt8](initial_cap)
        entry.blob_ptr = Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(blob)
        )
        entry.blob_capacity = initial_cap
        entry.blob_used = 0
        self.sessions[unsafe_offset=si] = entry
        self.session_count += 1
        return si

    def _lru_session(self) -> Int:
        """Return the slot of the least-recently-accessed active session, or -1."""
        var lru_slot = -1
        var lru_ts = Int64(0)
        for si in range(MAX_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                continue
            if lru_slot < 0 or self.sessions[unsafe_offset=si].last_access < lru_ts:
                lru_slot = si
                lru_ts = self.sessions[unsafe_offset=si].last_access
        return lru_slot

    def _alloc_session(mut self, session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      session_id_len: Int) -> Int:
        """Allocate a new session. At MAX_SESSIONS, evicts the LRU session and
        reuses its slot (never rejects)."""
        var now = _layerstore_now_ns()
        # Find a free slot
        for si in range(MAX_SESSIONS):
            if not self.sessions[unsafe_offset=si].active:
                return self._init_session_slot(si, session_id_ptr, session_id_len, now)
        # All slots active — evict the LRU session and reuse its slot. Safe because
        # zero-copy fetch pointers are consumed within the request that produced
        # them (see store_layer); the victim is never the session being stored.
        var victim = self._lru_session()
        if victim < 0:
            return -1
        self._free_session_slot(victim)
        self.evictions += 1
        return self._init_session_slot(victim, session_id_ptr, session_id_len, now)

    @always_inline
    def _hash_bytes(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> UInt64:
        """Simple wyhash-like hash for session ID lookup."""
        var h = UInt64(0x517cc1b727220a95)
        for bi in range(length):
            h = (h ^ UInt64(ptr[unsafe_offset=bi])) * UInt64(0x9e3779b97f4a7c15)
        return h

    def store_layer(mut self,
                   session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   session_id_len: Int,
                   layer_id: Int,
                   tensor_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   tensor_len: Int) raises -> Bool:
        """Store a single layer's KV tensor for a session."""
        if not self.enabled:
            return False
        if layer_id < 0 or layer_id >= MAX_LAYERS:
            return False

        # Find or create session
        var si = self._find_session(session_id_ptr, session_id_len)
        if si < 0:
            si = self._alloc_session(session_id_ptr, session_id_len)
            if si < 0:
                return False  # All session slots full
        self.sessions[unsafe_offset=si].last_access = _layerstore_now_ns()

        var session = self.sessions[unsafe_offset=si]

        # Check capacity — grow 4× (offsets are blob-relative, so a copy-realloc is
        # safe; zero-copy fetch pointers are consumed within the same request and never
        # held across stores). The initial blob is already 64 MB (~16×4 MB layers), so
        # growth only fires on large-context sessions; gh #131 §3.3: 4× halves the
        # re-memcpy passes vs doubling (e.g. a 40-layer/~160 MB session copies 64 MB once
        # instead of 64+128 MB across two doublings).
        if session.blob_used + tensor_len > session.blob_capacity:
            var new_cap = session.blob_capacity * 4
            while session.blob_used + tensor_len > new_cap:
                new_cap *= 4
            var new_blob = alloc[UInt8](new_cap)
            unsafe_memcpy(dest=new_blob, src=session.blob_ptr, count=session.blob_used)
            session.blob_ptr.unsafe_free()
            self.sessions[unsafe_offset=si].blob_ptr = Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(new_blob)
            )
            self.sessions[unsafe_offset=si].blob_capacity = new_cap
            session = self.sessions[unsafe_offset=si]

        # Copy tensor into session blob
        var offset = session.blob_used
        unsafe_memcpy(dest=session.blob_ptr.unsafe_offset(offset), src=tensor_ptr, count=tensor_len)

        # Update session
        self.sessions[unsafe_offset=si].blob_used = session.blob_used + tensor_len
        if layer_id >= self.sessions[unsafe_offset=si].num_layers:
            self.sessions[unsafe_offset=si].num_layers = layer_id + 1

        # Update layer entry
        var li = si * MAX_LAYERS + layer_id
        self.layers[unsafe_offset=li].offset = offset
        self.layers[unsafe_offset=li].size = tensor_len
        self.layers[unsafe_offset=li].stored = True

        self.total_bytes_stored += tensor_len
        self.store_count += 1
        return True

    def fetch_layer(mut self,
                   session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                   session_id_len: Int,
                   layer_id: Int,
                   out_ptr: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                   out_len: Pointer[Int, MutUntrackedOrigin]) raises -> Bool:
        """Fetch a single layer's KV tensor. Sets out_ptr/out_len on hit. Returns True on hit."""
        self.fetch_count += 1
        if not self.enabled:
            self.fetch_misses += 1
            return False
        if layer_id < 0 or layer_id >= MAX_LAYERS:
            self.fetch_misses += 1
            return False

        var si = self._find_session(session_id_ptr, session_id_len)
        if si < 0:
            self.fetch_misses += 1
            return False

        var li = si * MAX_LAYERS + layer_id
        if not self.layers[unsafe_offset=li].stored:
            self.fetch_misses += 1
            return False

        # Return pointer directly into the session blob (zero-copy)
        out_ptr.unsafe_write(self.sessions[unsafe_offset=si].blob_ptr.unsafe_offset(self.layers[unsafe_offset=li].offset))
        out_len.unsafe_write(self.layers[unsafe_offset=li].size)

        self.sessions[unsafe_offset=si].last_access = _layerstore_now_ns()
        self.fetch_hits += 1
        return True

    def _free_session_slot(mut self, si: Int):
        """Free slot `si`'s layer entries, blob arena, and id copy, then reset it."""
        # Clear layer entries
        for li in range(MAX_LAYERS):
            self.layers[unsafe_offset=si * MAX_LAYERS + li] = LayerEntry()

        # Free the session's blob arena and id copy, then reset the slot
        self.total_bytes_stored -= self.sessions[unsafe_offset=si].blob_used
        if self.sessions[unsafe_offset=si].blob_capacity > 0:
            self.sessions[unsafe_offset=si].blob_ptr.unsafe_free()
        if self.sessions[unsafe_offset=si].session_id_len > 0:
            self.sessions[unsafe_offset=si].session_id_ptr.unsafe_free()
        self.sessions[unsafe_offset=si] = SessionEntry()
        self.session_count -= 1

    def delete_session(mut self,
                      session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      session_id_len: Int) -> Bool:
        """Delete a session and free its resources."""
        var si = self._find_session(session_id_ptr, session_id_len)
        if si < 0:
            return False
        self._free_session_slot(si)
        return True
