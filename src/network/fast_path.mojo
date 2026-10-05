from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc, unsafe_memcpy, unsafe_memset, stack_allocation
from std.collections import Array, Span
from std.sys.intrinsics import prefetch
from std.ffi import external_call

from src.network.server import TCPServer
from src.network.response_writer import ResponseWriter
from src.common.container_free import free_container, remove_and_free
from src.common.hash_map import SlabHashMap, StripedHashMap, _movemask16
from std.bit import count_trailing_zeros
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.memory.slab_allocator import SlabAllocator
from src.memory.object_pool import ObjectPool
from src.common.value import GenericValue, ValueType
from src.common.utils import acc_digit_checked, format_int_to_buf, parse_int64_strict
from src.common.bitmap import getbit, setbit, bitcount
from src.common.hll import hll_add, hll_count, HLL_REGISTERS
from src.vector.hnsw import HNSWGraph, SharedHNSWView
from src.io.wal import WAL, gv_bytes
from src.io.blob_store import BlobStore, BLOB_TIER_OFF
from src.commands.transaction import TransactionState
from src.commands.command_table import PION_COMMAND_COUNT, command_arity, command_is_write
from src.network.raft import RaftNode, RaftLogEntry
from src.common.lock_free import LockFreeRingBuffer, AITask
# gh #85: the KV_BUS import is gone with the last of the P2 fields — see
# `NetworkEngine` for why every cross-worker routing branch was unreachable.
from src.network.cluster import ClusterState
from src.common.prng import Xoshiro256PlusPlus


# gh #102 (C4): `proto-max-bulk-len` ceiling for the fast path. The bulk-length
# accumulators below (`X = X*10 + digit`) share the resp3.mojo overflow class —
# a ~19-digit prefix wraps `Int` negative, which slips past the positive-only
# `it_pos + X + 2 > n` frame-fit checks and reaches a raw pointer read with a
# negative length. Guarding `X < 0 or X > MAX_BULK_LEN` at the frame shell
# (num_args, cmd_len) and the key/value primitives bails such frames to the
# (hardened) slow path. 512 MB matches the Redis default.
comptime MAX_BULK_LEN = 536870912   # 512 MB

# gh #175: batched GET burst kernel constants. A simple GET frame is
# `*2\r\n$3\r\nGET\r\n$<klen>\r\n<key>\r\n` — the first 8 bytes are two fixed
# u32 words (checked as single unaligned loads), bytes 8-10 are the
# case-insensitive command name, bytes 11-13 are `\r\n$`.
comptime GET_HDR0 = UInt32(0x0A0D322A)   # '*' '2' '\r' '\n' (LE)
comptime GET_HDR1 = UInt32(0x0A0D3324)   # '$' '3' '\r' '\n' (LE)
# 16, not 32: A/B'd 2026-08-03 on pipeline_deep (P=30, t=4 c=30) — a 32-window
# measured 2.506M mean vs 2.598M for 16 (interleaved ABBA). More in-flight
# prefetches (~224 slots at 32×7 lines) thrash the fill buffers and stage-A
# metadata lines age out before stage-C uses them. P=30 batches simply take
# two bursts; each stays fully overlapped internally.
comptime GET_BURST_MAX = 16

@no_inline
def _bulks_end(buffer: UnsafePointer[UInt8, MutUntrackedOrigin], pos: Int, n: Int,
               count: Int) -> Int:
    """Offset just past `count` complete bulk strings starting at `pos`, or -1
    when the recv buffer does not hold all of them yet. Out of line: every
    caller is a cold arm or behind has_cluster, and inlined copies grew
    process_data_plane (the gate's MSET row is sensitive to its shape).

    For an arm that would otherwise act on part of a frame: a pipelined frame
    split across two reads is re-run from its start once the rest arrives, so
    anything done before the arm noticed the frame was short is done twice —
    a reply written twice, or a mutation applied twice (PFADD then answered 0,
    because its first half-run had already moved the registers). Check first,
    then act."""
    var p = pos
    for _ in range(count):
        if p >= n or buffer[p] != 36:
            return -1
        p += 1
        var l = 0
        while p < n and buffer[p] != 13:
            l = l * 10 + Int(buffer[p] - 48)
            p += 1
        p += 2
        if l < 0 or p + l + 2 > n:
            return -1
        p += l + 2
    return p


@always_inline
def cmd_eq(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], length: Int, name: StaticString) -> Bool:
    """gh #225: whole-name, case-insensitive command match against a literal.

    The `cmd_matches_N` family stops at 8 bytes, so every longer name — the
    whole substrate surface, `FT.*`, `KV.PREFIX.*`, `ATTEND.*` — was matched by
    hand as "length plus a few bytes". 166 arms did that, and a prefix match
    accepts names that are not the command: `PINX` ran as `PING`, `DBSIZ<NUL>`
    as `DBSIZE`. The live hazard is less the garbage input than the shadowing —
    a future 4-byte `PI**` command is silently swallowed by PING's arm.

    `name` must be lowercase ASCII. The fold is an explicit A-Z range test, not
    `| 0x20`: that trick maps `_` (0x5F) to 0x7F, so a literal underscore in a
    name like `QUERY_SPARSE` cannot be expressed with it — one of the reasons
    the hand-rolled arms skipped bytes in the first place.

    The length compare is first and rejects almost every arm immediately, so
    this costs what the old `tl == N` guard cost."""
    if length != name.byte_length():
        return False
    var np = name.unsafe_ptr()
    for k in range(length):
        var c = ptr[k]
        if c >= 65 and c <= 90:
            c += 32
        if c != np[k]:
            return False
    return True

@always_inline
def cmd_matches_3(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2

@always_inline
def cmd_matches_4(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2 and (ptr[3] | 0x20) == b3

@always_inline
def cmd_matches_5(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8, b4: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2 and (ptr[3] | 0x20) == b3 and (ptr[4] | 0x20) == b4

@always_inline
def cmd_matches_6(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8, b4: UInt8, b5: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2 and (ptr[3] | 0x20) == b3 and (ptr[4] | 0x20) == b4 and (ptr[5] | 0x20) == b5

@always_inline
def cmd_matches_7(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8, b4: UInt8, b5: UInt8, b6: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2 and (ptr[3] | 0x20) == b3 and (ptr[4] | 0x20) == b4 and (ptr[5] | 0x20) == b5 and (ptr[6] | 0x20) == b6

@always_inline
def cmd_matches_8(ptr: UnsafePointer[UInt8, MutUntrackedOrigin], b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8, b4: UInt8, b5: UInt8, b6: UInt8, b7: UInt8) -> Bool:
    return (ptr[0] | 0x20) == b0 and (ptr[1] | 0x20) == b1 and (ptr[2] | 0x20) == b2 and (ptr[3] | 0x20) == b3 and (ptr[4] | 0x20) == b4 and (ptr[5] | 0x20) == b5 and (ptr[6] | 0x20) == b6 and (ptr[7] | 0x20) == b7

@always_inline
def _get_now_ns() -> Int64:
    """Return current time in nanoseconds from CLOCK_REALTIME (id=0).

    gh #202: the timespec used to be `alloc[Int64](2)` + `free` — a full
    tcmalloc round trip per call, on a function with 25 call sites including
    the TTL-checked GET arm and the TTL sweep slot. `stack_allocation` is the
    house rule for exactly this shape (see the kevent buffers); the alloca is
    static so it does not grow the frame per iteration.
    """
    var ts = stack_allocation[2, Int64]()
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    return ts[0] * Int64(1_000_000_000) + ts[1]


@always_inline
def _replica_name_is(buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
                     cmd_start: Int, cmd_len: Int, lit: StringLiteral) -> Bool:
    """Full-name, case-insensitive match for the replica read/write classifier.

    Every name it is used with is pure ASCII letters, so `| 0x20` folding is
    safe here (unlike the substrate names, which contain '_').
    """
    if cmd_len != lit.byte_length():
        return False
    var lp = lit.unsafe_ptr()
    for k in range(cmd_len):
        if (buffer[unsafe_offset=cmd_start + k] | 0x20) != lp[unsafe_offset=k]:
            return False
    return True


struct FastPathHandler(Movable):
    var keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin]
    var hash_map_pool: UnsafePointer[ObjectPool[SlabHashMap], MutUntrackedOrigin]
    var skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]
    var list_pool: UnsafePointer[ObjectPool[SlabList], MutUntrackedOrigin]
    var ai_queue: UnsafePointer[LockFreeRingBuffer, MutUntrackedOrigin]
    var wal: UnsafePointer[WAL, MutUntrackedOrigin]
    var raft: UnsafePointer[RaftNode, MutUntrackedOrigin]
    var shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin]
    var worker_id: Int
    var num_workers: Int
    # Transaction: per-fd MULTI mode flag (shared pointer from slow_path.tx_state.in_multi)
    var tx_in_multi: UnsafePointer[UInt8, MutUntrackedOrigin]
    # WATCH: key version array (shared pointer from slow_path.tx_state.key_versions)
    var key_versions: UnsafePointer[UInt64, MutUntrackedOrigin]
    # P4: per-fd local-affinity flag. True when the connection was accepted on this worker's
    # secondary listen port (1974+worker_id). These connections skip cross-worker routing
    # entirely — all commands are served directly from this worker's keyspace.
    var local_affinity: UnsafePointer[UInt8, MutUntrackedOrigin]
    # Phase 5: Cluster state (shared, read-only after init)
    var cluster: UnsafePointer[ClusterState, MutUntrackedOrigin]
    # C1: Per-fd ASKING flag — set by ASKING command, cleared after one command
    var asking_flags: UnsafePointer[UInt8, MutUntrackedOrigin]
    # TTL: per-worker map of key → expiry_ns (GenericValue.INT). Shared with SlowPathHandler.
    var ttl_map: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
    # Active TTL sweep: rotating cursor into ttl_map metadata array.
    var ttl_sweep_cursor: Int
    # gh #100 (C2): per-fd authenticated flag (shared pointer from
    # slow_path.tx_state.authed). Only consulted when auth_required is True.
    var authed: UnsafePointer[UInt8, MutUntrackedOrigin]
    # gh #101: per-fd tenant binding (shared pointer from
    # slow_path.tx_state.tenant_id). Only consulted when tenant_mode is True.
    var tenant_ids: UnsafePointer[Int16, MutUntrackedOrigin]
    # Fast-path flags (pre-computed, avoid pointer dereference per command)
    var has_cluster: Bool
    var has_ttl: Bool
    var has_wal: Bool
    # gh #100 (C2): True when --requirepass is set. When True, an unauthenticated
    # fd falls through to the slow path (which owns AUTH + the NOAUTH gate).
    var auth_required: Bool
    # gh #101: True when --tenant is configured. Tenant-bound fds never take
    # the fast path — the slow-path pre-pass owns the allowlist + key rewrite.
    var tenant_mode: Bool
    var prng: Xoshiro256PlusPlus
    # gh #163: file-backed arena for large values. Wired post-construction by
    # Pion.__init__ (the arena is built in phase 4, the engine in phase 3).
    # Declared LAST deliberately: inserting fields mid-struct shifted every
    # downstream hot field by 16 B and cost a measured ~1.5% on the memtier
    # pipeline profile. New fields go at the tail.
    var blobs: UnsafePointer[BlobStore, MutUntrackedOrigin]
    var blob_threshold: Int          # BLOB_TIER_OFF disables the tier (--no-blob-tier)
    var field_sweep_cursor: Int      # gh #392: position in keyspace.field_ttl_index (last: hot struct)

    @no_inline
    def _set_blob_value(mut self, key_val: GenericValue,
                        key_ptr: UnsafePointer[UInt8, MutUntrackedOrigin], k_len: Int,
                        val_ptr: UnsafePointer[UInt8, MutUntrackedOrigin], val_len: Int):
        """gh #163 cold path: route a >= blob_threshold SET through the
        file-backed arena (WAL records a 24-byte pointer). Outlined so
        process_data_plane only grows by a compare + call — inlining this block
        measurably regressed the pipeline profile via code-size alone."""
        if is_not_null(self.blobs):
            var blob_seg = -1
            var blob_off = -1
            _ = self.blobs[].append(val_ptr, val_len, blob_seg, blob_off)
            var bp = null_ptr[UInt8, MutUntrackedOrigin]()
            if blob_seg >= 0:
                bp = self.blobs[].ptr_at(blob_seg, blob_off, val_len)
            if is_not_null(bp):
                if self.has_wal:
                    _ = self.wal[].append_blob_ref(key_ptr, k_len,
                                                   blob_seg, blob_off, val_len)
                self.keyspace[].set(key_val,
                                    GenericValue.from_blob_ptr(bp, val_len))
                return
        # Arena full or unavailable — fall back to the heap so the write still
        # succeeds (see BlobStore.append).
        if self.has_wal:
            _ = self.wal[].append_kv(1, key_ptr, k_len, val_ptr, val_len)
        self.keyspace[].set(key_val, GenericValue.borrow_buf(val_ptr, val_len))

    def __init__(
        out self,
        keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
        hash_map_pool: UnsafePointer[ObjectPool[SlabHashMap], MutUntrackedOrigin],
        skip_list_pool: UnsafePointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
        list_pool: UnsafePointer[ObjectPool[SlabList], MutUntrackedOrigin],
        ai_queue: UnsafePointer[LockFreeRingBuffer, MutUntrackedOrigin],
        wal: UnsafePointer[WAL, MutUntrackedOrigin],
        raft: UnsafePointer[RaftNode, MutUntrackedOrigin],
        shared_hnsw: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
        worker_id: Int = 0,
        num_workers: Int = 1,
        local_affinity: UnsafePointer[UInt8, MutUntrackedOrigin] = null_ptr[UInt8, MutUntrackedOrigin](),
        cluster: UnsafePointer[ClusterState, MutUntrackedOrigin] = null_ptr[ClusterState, MutUntrackedOrigin](),
        ttl_map: UnsafePointer[SlabHashMap, MutUntrackedOrigin] = null_ptr[SlabHashMap, MutUntrackedOrigin](),
    ):
        self.keyspace = keyspace
        self.hash_map_pool = hash_map_pool
        self.skip_list_pool = skip_list_pool
        self.list_pool = list_pool
        self.ai_queue = ai_queue
        self.wal = wal
        self.blobs = null_ptr[BlobStore, MutUntrackedOrigin]()
        self.blob_threshold = BLOB_TIER_OFF
        self.raft = raft
        self.shared_hnsw = shared_hnsw
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.local_affinity = local_affinity
        self.cluster = cluster
        self.ttl_map = ttl_map
        self.ttl_sweep_cursor = 0
        self.field_sweep_cursor = 0
        self.has_cluster = is_not_null(cluster) and cluster[].enabled
        self.has_ttl = is_not_null(ttl_map)  # True if TTL map exists (lazy expiry checks enabled)
        self.has_wal = True  # default on; NetworkEngine sets False when --no-wal
        self.auth_required = False  # NetworkEngine sets True when --requirepass is set
        self.authed = null_ptr[UInt8, MutUntrackedOrigin]()
        self.tenant_mode = False  # NetworkEngine sets True when --tenant is set (gh #101)
        self.tenant_ids = null_ptr[Int16, MutUntrackedOrigin]()
        self.tx_in_multi = null_ptr[UInt8, MutUntrackedOrigin]()
        self.key_versions = null_ptr[UInt64, MutUntrackedOrigin]()
        self.prng = Xoshiro256PlusPlus(UInt64(worker_id + 1) * 0x9E3779B97F4A7C15)
        # C1: ASKING flags — 65536 entries, one per fd
        self.asking_flags = alloc[UInt8](65536)
        unsafe_memset(self.asking_flags, 0, 65536)

    @always_inline
    def sweep_expired_keys(mut self, max_scan: Int):
        """Scan up to max_scan slots in ttl_map and evict any expired keys, then
        up to max_scan hashes that carry field TTLs.

        Uses a rotating cursor so each call advances through a different region
        of the metadata array, amortising the full scan across many loop ticks.
        Safe to call while the map is being written: removals mark slots DELETED
        (0xFF) rather than moving entries, so forward iteration remains valid.

        gh #392: field TTLs no longer live in ttl_map (as ambiguous
        `key::field` keys that this sweep had to split — and that turned a
        plain key containing `::` into a phantom field deletion, leaving the
        key alive for good); they are in each hash, found via the keyspace's
        field_ttl_index."""
        if not self.keyspace[].active_expire or self.keyspace[].expire_hides_only:
            return     # DEBUG SET-ACTIVE-EXPIRE 0, or a replica: the primary expires (#45)
        self._sweep_field_ttls(max_scan)
        if is_null(self.ttl_map) or self.ttl_map[].size == 0:
            return
        var cap = self.ttl_map[].capacity
        if cap == 0:
            return
        var now_ns = _get_now_ns()
        var cursor = self.ttl_sweep_cursor & (cap - 1)
        var scanned = 0
        while scanned < max_scan:
            var meta = self.ttl_map[].metadata[cursor]
            # Valid occupied slot: fingerprint is 0x00..0x7F (not EMPTY=0x80 or DELETED=0xFF)
            if meta != UInt8(0x80) and meta != UInt8(0xFF):
                var exp_ns = self.ttl_map[].values[cursor].as_int()
                if now_ns > exp_ns:
                    # `key` is a shallow copy of the TTL map's own key, whose
                    # heap payload remove_generic FREES — so the TTL entry goes
                    # LAST. Removing it first (as this did) read freed memory
                    # for every key over 23 bytes: the keyspace removal hashed
                    # tcmalloc's free-list word instead of the key, missed, and
                    # the expired key stayed readable forever with no TTL left.
                    # The keyspace's remove now drops the TTL entry
                    # itself, so it gets an OWNED copy of the key.
                    var key = self.ttl_map[].keys[cursor].clone()
                    # Remove from the keyspace, freeing an aggregate's container
                    # (gh #369) rather than leaking it, and log the deletion
                    # (#45): replay must not revive the key, or give a key
                    # created again under its name the old deadline.
                    var gone = self.keyspace[].expire_key(key, UInt64(key.__hash__()), False)
                    if gone.type.value != ValueType.NONE:
                        free_container(gone)
                    _ = self.ttl_map[].remove_generic(key)   # a TTL whose key was already gone
                    key.free_str_payload()
            cursor = (cursor + 1) & (cap - 1)
            scanned += 1
        self.ttl_sweep_cursor = cursor

    def _sweep_field_ttls(mut self, max_scan: Int):
        """gh #392: active expiry of hash fields. Walks the index of hashes that
        carry field TTLs, deletes their due fields, drops the key when that
        empties it, and drops index entries that no longer lead anywhere (the
        key went, was renamed, or has no field TTLs left)."""
        var idx = self.keyspace[].field_ttl_index
        if idx[].size == 0:
            return
        var cap = idx[].capacity
        var now_ns = _get_now_ns()
        var cursor = self.field_sweep_cursor & (cap - 1)
        for _ in range(min(max_scan, cap)):
            var m = idx[].metadata[cursor]
            if m != UInt8(0x80) and m != UInt8(0xFF):
                var key = idx[].keys[cursor]   # the index's own key: remove it LAST
                var hv = self.keyspace[].get(key)
                var drop = True
                if hv.type.value == ValueType.HASH:
                    var hp = hv.as_hash().bitcast[SlabHashMap]()
                    _ = hp[].purge_expired_fields(now_ns)
                    if hp[].size == 0:
                        _ = remove_and_free(self.keyspace, key)
                    elif hp[].has_field_ttls():
                        drop = False
                if drop:
                    _ = idx[].remove_generic(key)
            cursor = (cursor + 1) & (cap - 1)
        self.field_sweep_cursor = cursor

    # gh #85b (gh #48 / eae8975): serve_bus_requests removed — KV_BUS routing
    # was permanently disabled. The body was a per-slot poll over inbound KV
    # request slots that served GET/SET/DEL/INCR/DECR/EXISTS locally and posted
    # the response back to the originating worker. Shared-nothing is the
    # committed model; restore from git history if a future cross-worker
    # routing design needs to start from this scaffold.

    @always_inline
    def _ask_if_migrating(
        self,
        key_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
        key_len: Int,
        mut writer: ResponseWriter,
    ) -> Bool:
        """§3: Returns True and writes -ASK if slot is MIGRATING and key not found locally.
        Called after _moved_if_needed returns False (we own the slot)."""
        if is_null(self.cluster) or not self.cluster[].enabled:
            return False
        var slot = self.cluster[].keyslot(key_ptr, key_len)
        if not self.cluster[].is_slot_migrating(slot):
            return False
        # Slot is migrating — check if key exists locally
        var key_v = GenericValue.borrow_buf(key_ptr, key_len)
        var val = self.keyspace[].get(key_v)
        if not val.is_none():
            return False  # key exists locally, serve it
        # Key not here — redirect to importing node
        var pi = Int(self.cluster[].slot_migrating[slot])
        if pi < 0 or pi >= self.cluster[].peer_count:
            return False
        var peer_host_len = self.cluster[].peer_host_lens[pi]
        var peer_port = self.cluster[].peer_ports[pi]
        # Write: -ASK <slot> <host>:<port>\r\n
        var rb = writer.buffer + writer.offset
        rb[0] = 45; rb[1] = 65; rb[2] = 83; rb[3] = 75  # -ASK
        rb[4] = 32  # space
        var off = 5
        off += format_int_to_buf(rb + off, 0, Int64(slot))
        rb[off] = 32; off += 1
        unsafe_memcpy(dest=rb + off, src=self.cluster[].peer_host_ptr(pi), count=peer_host_len)
        off += peer_host_len
        rb[off] = 58; off += 1  # ':'
        off += format_int_to_buf(rb + off, 0, Int64(peer_port))
        rb[off] = 13; rb[off + 1] = 10
        writer.offset += off + 2
        return True

    def _moved_if_needed(
        self,
        key_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
        key_len: Int,
        mut writer: ResponseWriter,
        fd: Int32 = Int32(-1),
    ) -> Bool:
        """Returns True and writes -MOVED if this key belongs to another cluster node.
        Respects ASKING flag: if slot is IMPORTING and ASKING was sent, serve locally."""
        if is_null(self.cluster) or not self.cluster[].enabled or self.cluster[].peer_count == 0:
            return False
        var slot = self.cluster[].keyslot(key_ptr, key_len)
        if self.cluster[].owns_slot(slot):
            return False
        # C1: If slot is IMPORTING and client sent ASKING, serve locally
        if self.cluster[].is_slot_importing(slot) and fd >= 0 and Int(fd) < 65536 and self.asking_flags[Int(fd)] != 0:
            self.asking_flags[Int(fd)] = 0  # clear after one use
            return False
        var pi = self.cluster[].find_peer_for_slot(slot)
        if pi < 0:
            return False
        var peer_host_len = self.cluster[].peer_host_lens[pi]
        var peer_port = self.cluster[].peer_ports[pi]
        # Write: -MOVED <slot> <host>:<port>\r\n
        var rb = writer.buffer + writer.offset
        rb[0] = 45; rb[1] = 77; rb[2] = 79; rb[3] = 86; rb[4] = 69; rb[5] = 68  # -MOVED
        rb[6] = 32  # space
        var off = 7
        off += format_int_to_buf(rb + off, 0, Int64(slot))
        rb[off] = 32; off += 1  # space
        unsafe_memcpy(dest=rb + off, src=self.cluster[].peer_host_ptr(pi), count=peer_host_len)
        off += peer_host_len
        rb[off] = 58; off += 1  # ':'
        off += format_int_to_buf(rb + off, 0, Int64(peer_port))
        rb[off] = 13; rb[off + 1] = 10  # \r\n
        writer.offset += off + 2
        return True

    @always_inline
    def _reject_if_replica_write(self, mut writer: ResponseWriter, fd: Int32) -> Bool:
        """Reject write commands on replica nodes.
        READONLY connections (local_affinity==3) can read but not write.
        Non-READONLY connections on replicas get -MOVED to the primary."""
        if not self.has_cluster or not self.cluster[].is_replica:
            return False
        # READONLY mode: reject writes with -READONLY
        if Int(fd) < 65536 and self.local_affinity[Int(fd)] == 3:
            var err = String("-READONLY You can't write against a read only replica.\r\n")
            writer.append_to_response(err.unsafe_ptr(), err.byte_length())
            return True
        # Non-READONLY: redirect to primary with -MOVED
        var pi = self.cluster[].primary_peer_idx
        if pi >= 0 and pi < self.cluster[].peer_count:
            var rb = writer.buffer + writer.offset
            rb[0] = 45; rb[1] = 77; rb[2] = 79; rb[3] = 86; rb[4] = 69; rb[5] = 68  # -MOVED
            rb[6] = 32
            var off = 7
            off += format_int_to_buf(rb + off, 0, Int64(0))  # slot 0 (any slot)
            rb[off] = 32; off += 1
            var peer_host_len = self.cluster[].peer_host_lens[pi]
            unsafe_memcpy(dest=rb + off, src=self.cluster[].peer_host_ptr(pi), count=peer_host_len)
            off += peer_host_len
            rb[off] = 58; off += 1
            off += format_int_to_buf(rb + off, 0, Int64(self.cluster[].peer_ports[pi]))
            rb[off] = 13; rb[off + 1] = 10
            writer.offset += off + 2
            return True
        # No primary known — reject with error
        var err2 = String("-READONLY You can't write against a read only replica.\r\n")
        writer.append_to_response(err2.unsafe_ptr(), err2.byte_length())
        return True

    @always_inline
    def _skip_args(
        self,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        start: Int,
        n: Int,
        count: Int,
    ) -> Int:
        """Skip `count` bulk string args starting at buffer[start]. Returns new position."""
        var pos = start
        for _ in range(count):
            if pos >= n or buffer[pos] != 36:
                break
            pos += 1
            var arg_len = 0
            while pos < n and buffer[pos] != 13:
                arg_len = arg_len * 10 + Int(buffer[pos] - 48)
                pos += 1
            pos += 2 + arg_len + 2
        return pos

    # gh #85b: _post_remote_async + _forward_and_respond removed alongside
    # serve_bus_requests (above). Both were callers of the disabled KV_BUS
    # routing; no remaining references after the gated `comptime if False:`
    # call sites in process_data_plane were dropped.

    # gh #175: batched GET burst kernel. The pipeline profile delivers ~10
    # pipelined commands per recv, and executing them one at a time serializes
    # a DRAM-miss chain per key (metadata line → key/value slot → value
    # payload) against a table far larger than LLC. This kernel parses a run
    # of simple SSO-key GETs first, then walks the batch in stages — hash +
    # metadata prefetch, slot resolve + slot prefetch, key compare + payload
    # prefetch, response emit — so the misses of up to GET_BURST_MAX
    # independent lookups overlap instead of queuing behind each other.
    #
    # Preconditions (enforced by the caller): no cluster routing, no live
    # TTLs. Keys >23B, wrong arity, or malformed frames stop the parse and
    # fall to the generic per-command arm, which handles everything.
    # @no_inline: runs once per batch, so call overhead is amortized ~10×,
    # and keeping ~150 lines out of process_data_plane preserves its I-cache
    # footprint (the gh #163 lesson).
    # MSET's whole-frame parse and apply, OUT of process_data_plane on purpose.
    # process_data_plane is one ~240 KB function and LLVM allocates registers
    # across all of it, so code added to ANY arm changed this loop: the gh #394
    # branch added frame checks and borrows elsewhere, and MSET's loop began
    # reloading `buffer` and friends from the stack every iteration — same
    # instruction count, ~15% more user time per MSET, the gate row 5-9% down
    # (measured with in-process PC sampling, 2026-09-29; the 09-27 maxmemory
    # branch was the same effect). As its own function its codegen no longer
    # depends on the other arms. One call per MSET, amortized over its pairs,
    # as _get_burst does for GET.
    #
    # Returns the offset just past the frame, or -1 when the recv buffer does
    # not hold all of it yet — in which case NOTHING has been applied (the pairs
    # are parsed into the stash first), so the engine can retry it whole.
    @no_inline
    def _mset_frame(
        mut self,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        start: Int,
        n: Int,
        num_args: Int,
        mut writer: ResponseWriter,
    ) -> Int:
        var it_pos = start
        var n_pairs = (num_args - 1) // 2
        # Whole frame first. Pairs were stored as they were parsed,
        # so a split MSET was half-applied — visible to every other
        # connection until the rest arrived (MSET is atomic) — and
        # retried from the start. Up to 16 pairs are parsed ONCE into
        # a stash and applied from it; a separate header walk before
        # the apply loop cost the gate's MSET row ~15% (measured,
        # interleaved A/B). Larger MSETs keep the walk.
        var mset_stash = stack_allocation[64, Int]()   # key_off, key_len, val_off, val_len
        var mset_start = it_pos    # the WAL bound below counts from here
        var mset_stashed = n_pairs <= 16
        if mset_stashed:
            var _sp = 0
            while _sp < n_pairs:
                if it_pos >= n or buffer[it_pos] != 36:
                    return -1
                it_pos += 1
                var _kl = 0
                while it_pos < n and buffer[it_pos] != 13:
                    _kl = _kl * 10 + Int(buffer[it_pos] - 48)
                    it_pos += 1
                it_pos += 2
                if _kl < 0 or it_pos + _kl + 2 > n:
                    return -1
                mset_stash[_sp * 4] = it_pos
                mset_stash[_sp * 4 + 1] = _kl
                it_pos += _kl + 2
                if it_pos >= n or buffer[it_pos] != 36:
                    return -1
                it_pos += 1
                var _vl = 0
                while it_pos < n and buffer[it_pos] != 13:
                    _vl = _vl * 10 + Int(buffer[it_pos] - 48)
                    it_pos += 1
                it_pos += 2
                if _vl < 0 or it_pos + _vl + 2 > n:
                    return -1
                mset_stash[_sp * 4 + 2] = it_pos
                mset_stash[_sp * 4 + 3] = _vl
                it_pos += _vl + 2
                _sp += 1
        elif _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
            return -1
        # gh #230: everything invariant across the pairs is
        # hoisted. These were per-key branches on state that
        # cannot change inside the loop.
        var mset_wal = self.has_wal and is_not_null(self.wal)
        # One capacity check for the whole command. The bound is
        # 13 header bytes per record plus every byte of the recv
        # buffer still unparsed — which is >= the key+value bytes
        # those records will carry, since they are copied FROM
        # those bytes. Too tight to prove only near a segment
        # end, where the per-record path takes over and rotates.
        var mset_batch = mset_wal and \
            self.wal[].batch_fits(13 * n_pairs + (n - mset_start))
        # cmd 31: one record for the whole MSET. Its header is
        # written last (mset_seal), so the pairs start 13 bytes in.
        var rec_off = Int(self.wal[].tail_offset) if mset_batch else 0
        var w_off = rec_off + 13
        var mset_watch = is_not_null(self.key_versions)
        # Nothing in this loop ADDS a TTL, so an empty ttl_map
        # stays empty and the probe stays skippable (gh #175).
        var mset_ttl = is_not_null(self.ttl_map) and self.ttl_map[].size > 0
        var pairs_parsed = 0
        while pairs_parsed < n_pairs:
            var key_off: Int
            var key_len: Int
            var val_off: Int
            var val_len: Int
            if mset_stashed:
                key_off = mset_stash[pairs_parsed * 4]
                key_len = mset_stash[pairs_parsed * 4 + 1]
                val_off = mset_stash[pairs_parsed * 4 + 2]
                val_len = mset_stash[pairs_parsed * 4 + 3]
            else:
                # The header walk above proved the frame complete.
                it_pos += 1
                key_len = 0
                while buffer[it_pos] != 13:
                    key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                    it_pos += 1
                key_off = it_pos + 2
                it_pos = key_off + key_len + 3
                val_len = 0
                while buffer[it_pos] != 13:
                    val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                    it_pos += 1
                val_off = it_pos + 2
                it_pos = val_off + val_len + 2
            var key_val = GenericValue.borrow(buffer + key_off, key_len)
            var mset_key_ptr = buffer + key_off   # gh #217: key bytes for the effect record
            var val_val = GenericValue.borrow(buffer + val_off, val_len)
            var mset_val_ptr = buffer + val_off
            # gh #230: hash the key ONCE and spend it twice —
            # the keyspace store and the WATCH version slot.
            var kh = UInt64(key_val.__hash__())
            self.keyspace[].set_with_hash(key_val, val_val, kh)
            # gh #217: this arm logged nothing, so a fast-path MSET
            # was lost entirely on replay. The slow-path arm routes
            # through execute_set and was always durable, which is
            # why MSET durability looked path-dependent.
            if mset_wal:
                # gh #229: batched — the header is published once
                # per MSET below, not once per key. gh #230: and
                # the tail cursor stays in a register across the
                # whole run of records.
                if mset_batch:
                    w_off = self.wal[].mset_pair_at(w_off,
                        mset_key_ptr, key_len, mset_val_ptr, val_len)
                else:
                    _ = self.wal[].append_kv_batched(UInt8(1), mset_key_ptr, key_len,
                        mset_val_ptr, val_len)
            # gh #230: MSET is a SET and must do what SET does to
            # the key's metadata. It did neither, so `WATCH k` did
            # not see an MSET of k (EXEC ran anyway — the silent
            # lost update WATCH exists to prevent), and `SET k v
            # EX 100; MSET k v2` left the TTL in place, so the
            # NEW value expired on the OLD value's deadline.
            if mset_watch:
                self.key_versions[TransactionState.key_slot_from_hash(kh)] += 1
            if mset_ttl:
                _ = self.ttl_map[].remove_generic(key_val)
            pairs_parsed += 1
        if mset_wal:
            if mset_batch:
                self.wal[].mset_seal(rec_off, w_off)
                self.wal[].commit_batch(w_off, 1)
            else:
                self.wal[].publish_header()
        writer.append_ok_response()
        return it_pos

    @no_inline
    def _get_burst(
        mut self,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        n: Int,
        start_pos: Int,
        fd: Int32,
        mut writer: ResponseWriter,
    ) -> Int:
        var key_off = Array[Int32, GET_BURST_MAX](uninitialized=True)
        var key_len_a = Array[Int32, GET_BURST_MAX](uninitialized=True)
        var h_a = Array[UInt64, GET_BURST_MAX](uninitialized=True)
        var d0_a = Array[UInt64, GET_BURST_MAX](uninitialized=True)
        var d1_a = Array[UInt64, GET_BURST_MAX](uninitialized=True)
        var d2_a = Array[UInt64, GET_BURST_MAX](uninitialized=True)
        var slot_a = Array[Int, GET_BURST_MAX](uninitialized=True)
        var vals = Array[GenericValue, GET_BURST_MAX](uninitialized=True)

        # ── Parse: collect a run of complete `*2\r\n$3\r\nGET\r\n$…` frames ──
        var p = start_pos
        var k = 0
        while k < GET_BURST_MAX and p + 14 <= n:
            if (buffer + p).bitcast[UInt32]()[0] != GET_HDR0: break
            if (buffer + p + 4).bitcast[UInt32]()[0] != GET_HDR1: break
            if (buffer[p + 8] | 0x20) != 103 or (buffer[p + 9] | 0x20) != 101 \
                    or (buffer[p + 10] | 0x20) != 116: break
            if buffer[p + 11] != 13 or buffer[p + 12] != 10 or buffer[p + 13] != 36: break
            var q = p + 14
            var klen = 0
            while q < n and buffer[q] != 13:
                klen = klen * 10 + Int(buffer[q] - 48)
                if klen > MAX_BULK_LEN: break
                q += 1
            # SSO keys only — a >23B key falls to the generic GET arm.
            if klen < 0 or klen > 23: break
            q += 2   # \r\n
            if q + klen + 2 > n: break   # incomplete frame: stop, don't consume
            key_off[k] = Int32(q)
            key_len_a[k] = Int32(klen)
            k += 1
            p = q + klen + 2
        if k == 0:
            return start_pos

        # ── Stage A: hash every key, prefetch its metadata group ──
        for i in range(k):
            var packed = GenericValue.hash_and_pack_sso(
                buffer + Int(key_off[i]), Int(key_len_a[i]))
            var h = packed[0]
            h_a[i] = h
            d0_a[i] = packed[1]
            d1_a[i] = packed[2]
            d2_a[i] = packed[3]
            var shard = self.keyspace[].shards + Int(h & 7)
            var idx = Int(h >> 7) & (shard[].capacity - 1)
            prefetch(shard[].metadata + idx)

        # ── Stage B: resolve first candidate slot, prefetch key+value slots ──
        comptime width = 16
        for i in range(k):
            var h = h_a[i]
            var shard = self.keyspace[].shards + Int(h & 7)
            var mask = shard[].capacity - 1
            var idx = Int(h >> 7) & mask
            var meta_chunk = (shard[].metadata + idx).load[width=width]()
            var mbits = _movemask16(
                meta_chunk.eq(SIMD[DType.uint8, width](UInt8(h & 0x7F))))
            if mbits != 0:
                var slot = (idx + count_trailing_zeros(mbits)) & mask
                slot_a[i] = slot
                prefetch((shard[].keys + slot).bitcast[UInt8]())
                prefetch((shard[].values + slot).bitcast[UInt8]())
            elif _movemask16(
                    meta_chunk.eq(SIMD[DType.uint8, width](SlabHashMap.EMPTY))) != 0:
                slot_a[i] = -1   # definite miss: no candidate, group has EMPTY
            else:
                slot_a[i] = -2   # full group, no candidate: rare long probe

        # ── Stage C: compare keys, fetch values, prefetch value payloads ──
        for i in range(k):
            var h = h_a[i]
            var shard = self.keyspace[].shards + Int(h & 7)
            var slot = slot_a[i]
            var val = GenericValue()
            if slot >= 0:
                var key_slot = shard[].keys[slot]
                if key_slot.type.value == ValueType.STRING_SSO \
                        and key_slot._data0 == d0_a[i] \
                        and key_slot._data1 == d1_a[i] \
                        and key_slot._data2 == d2_a[i]:
                    val = shard[].values[slot]
                else:
                    # h2 collision or later candidate — full probe (rare).
                    val = shard[].get_with_sso(h, d0_a[i], d1_a[i], d2_a[i])
            elif slot == -2:
                val = shard[].get_with_sso(h, d0_a[i], d1_a[i], d2_a[i])
            vals[i] = val
            if val.type.value == ValueType.STRING:
                # Overlap the payload fetch with the remaining lookups; the
                # response memcpy in stage D then hits warm lines. Cap at 512B
                # — larger values amortize their own miss cost.
                var vp = val.as_string()
                var vl = val.string_len()
                var off = 0
                while off < vl and off < 512:
                    prefetch(vp + off)
                    off += 64

        # ── Stage D: emit responses in order ──
        for i in range(k):
            writer.append_large_value_response_writev(fd, vals[i])
        return p

    def process_data_plane(
        mut self,
        fd: Int32,
        buffer: UnsafePointer[UInt8, MutUntrackedOrigin],
        n: Int,
        mut writer: ResponseWriter,
        server: TCPServer,
        kq: Int32,
        mut hnsw: HNSWGraph,
        mut db_size: Int,
    ) -> Int:
        if n < 4: return 0
        # Transaction: if fd is in MULTI mode, route everything to slow path for queuing
        if is_not_null(self.tx_in_multi) and self.tx_in_multi[Int(fd)] == 1:
            return 0
        # gh #100 (C2): when a password is configured, no fast-path command runs
        # until this connection has authenticated. Fall through to the slow path,
        # which owns AUTH itself and rejects every other command with -NOAUTH.
        # The `auth_required` bool short-circuits to zero cost when unset (default).
        if self.auth_required and (is_null(self.authed) or self.authed[Int(fd)] == 0):
            return 0
        # gh #101: tenant-bound fds never run fast-path commands — the slow
        # path's namespacing pre-pass owns the allowlist and the key rewrite.
        # `tenant_mode` is False unless --tenant is configured (zero cost by
        # default); admin fds (tenant_id == -1) keep the fast path.
        if self.tenant_mode and is_not_null(self.tenant_ids) and self.tenant_ids[Int(fd)] >= 0:
            return 0
        # gh #260: the WAL can no longer persist writes. Hand the whole buffer to
        # the slow path, which refuses the mutating commands individually and
        # still serves the reads. Cost in normal operation is one load of an
        # already-hot WAL cache line and a predicted-not-taken branch, per RECV
        # BUFFER — not per command, and never on the append path itself.
        if is_not_null(self.wal) and self.wal[].refusing_writes():
            return 0

        # Check first byte: '*' (0x2A) means RESP command, anything else is inline
        var num_cmds = 1 if buffer[0] == 42 else 0
        var is_complex = False

        # Inline PING: batch consecutive `PING\r\n` (or `PING\n`) sequences
        # in the same buffer into one bulk PONG write + one flush syscall.
        # Pre-fix this branch consumed one PING per call → on `redis-benchmark
        # -t PING_INLINE -P 10` (10 pipelined inline PINGs/socket write) we
        # paid 10× the send() syscall cost; PING_INLINE bottomed out at ~600K
        # RPS instead of the Redis-comparable ~1.4M.
        if num_cmds == 0 and n >= 4:
            if (buffer[0] | 0x20) == 112 and (buffer[1] | 0x20) == 105 and (buffer[2] | 0x20) == 110 and (buffer[3] | 0x20) == 103:
                var count = 0
                var ci = 0
                while ci + 4 <= n:
                    if ((buffer[ci] | 0x20) != 112
                            or (buffer[ci + 1] | 0x20) != 105
                            or (buffer[ci + 2] | 0x20) != 110
                            or (buffer[ci + 3] | 0x20) != 103):
                        break
                    var after = ci + 4
                    # End-of-buffer: accept this PING (mirrors original
                    # lenient single-shot behaviour — CR/LF can arrive next read).
                    if after >= n:
                        count += 1
                        ci = after
                        break
                    # Non-terminator after PING: stop and let the rest fall
                    # through to slow-path. Don't consume bytes we can't safely batch.
                    if buffer[after] != 13 and buffer[after] != 10:
                        count += 1
                        ci = after
                        break
                    count += 1
                    ci = after
                    if ci < n and buffer[ci] == 13: ci += 1
                    if ci < n and buffer[ci] == 10: ci += 1
                writer.append_pong_bulk(count)
                writer.flush_response(fd, server, kq)
                return ci

        if not is_complex and num_cmds > 0:
            var it_pos = 0
            var fast_path_ok = True
            var consumed = 0

            while it_pos < n:
                var cmd_start_pos = it_pos
                if buffer[it_pos] != 42: # *
                    fast_path_ok = False
                    break

                # gh #175: batched GET burst — overlap DRAM misses across the
                # pipelined batch. Preconditions checked once here: no cluster
                # routing (covers MOVED/ASK + replica gating) and no live TTLs
                # (lazy-expiry check needs the per-key probe). Non-GET frames
                # fail the first u32 compare inside (`*2` vs `*3`…) — one load.
                if (not self.has_cluster) \
                        and (not self.has_ttl or self.ttl_map[].size == 0):
                    var burst_end = self._get_burst(buffer, n, it_pos, fd, writer)
                    if burst_end > it_pos:
                        it_pos = burst_end
                        consumed = it_pos
                        continue

                it_pos += 1

                var num_args = 0
                while it_pos < n and buffer[it_pos] != 13: # \r
                    num_args = num_args * 10 + Int(buffer[it_pos] - 48)
                    if num_args > MAX_BULK_LEN: break  # gh #102: bail before Int overflow
                    it_pos += 1
                it_pos += 2 # \r\n
                # gh #102 (C4): a wrapped/absurd arg count → hand to the slow path.
                # gh #147: flush first — `consumed` covers commands already
                # executed above, and their replies are still sitting in the
                # response buffer.
                if num_args < 0 or num_args > MAX_BULK_LEN:
                    if consumed > 0: writer.flush_response(fd, server, kq)
                    return consumed

                if it_pos >= n or buffer[it_pos] != 36: # $
                    if consumed > 0: writer.flush_response(fd, server, kq)
                    return consumed

                it_pos += 1

                var cmd_len = 0
                while it_pos < n and buffer[it_pos] != 13: # \r
                    cmd_len = cmd_len * 10 + Int(buffer[it_pos] - 48)
                    if cmd_len > MAX_BULK_LEN: break  # gh #102: bail before Int overflow
                    it_pos += 1
                it_pos += 2 # \r\n

                # gh #102 (C4): `cmd_len < 0` catches a negative-overflowed length
                # that would otherwise slip past the positive-only fit check.
                if cmd_len < 0 or it_pos + cmd_len + 2 > n:
                    if consumed > 0: writer.flush_response(fd, server, kq)
                    return consumed

                var cmd_start = it_pos
                it_pos += cmd_len + 2 # Skip cmd and \r\n

                var b0 = buffer[cmd_start]
                var b0_lower = b0 | 0x20

                # Cluster mode: every MOVED/ASK redirect and the replica write
                # refusal below write their error FIRST and then skip the rest
                # of the frame without bounds — on a frame split across two
                # reads that answered twice and skipped past the data. One
                # whole-frame check here covers all of them; standalone mode
                # (has_cluster false) never pays it.
                if self.has_cluster and _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
                    if consumed > 0: writer.flush_response(fd, server, kq)
                    return consumed

                # Replica write rejection: block write commands on replica nodes.
                # Read commands (GET, MGET, LRANGE, LLEN, SRANDMEMBER, EXISTS, DBSIZE,
                # PING, CLUSTER, CONFIG, INFO, HELLO, READONLY, READWRITE, ASKING, etc.)
                # pass through. Write commands are rejected early.
                if self.has_cluster and self.cluster[].is_replica:
                    # gh #225: these arms matched FIRST BYTE + LENGTH only, so a
                    # replica classified writes as reads and accepted them:
                    #   MSET   (m,4) matched MGET      HSET/HDEL (h,4) matched HGET
                    #   LPUSH/LTRIM (l,5) matched the LLEN arm (LLEN is 4 bytes —
                    #                     that arm never matched LLEN at all)
                    #   EXPIRE (e,6) matched EXISTS    GETSET (g,6) matched GETBIT
                    #   DECRBY (d,6) matched DBSIZE    APPEND (a,6) matched ASKING
                    #   PERSIST(p,7) matched PFCOUNT
                    # A read-only replica silently took the write and diverged from
                    # its primary. Match the whole name; this block only runs on a
                    # replica, so the extra compares cost nothing on the hot path.
                    # The allowlist below was the WHOLE classifier, so every
                    # Redis read it did not name — SMEMBERS, ZRANGE, HGETALL,
                    # XLEN, SCAN, TTL, … — was refused on a replica as if it
                    # were a write. Redis's own `write` flag (generated into
                    # command_table) now decides for every command Redis knows
                    # (command_arity != 0); Pion-only commands keep needing an
                    # explicit entry, since "not flagged write" means nothing
                    # for a command Redis never classified.
                    var _cmd_p = buffer.unsafe_offset(cmd_start)
                    var _is_read = (
                        (command_arity(_cmd_p, cmd_len) != 0
                         and not command_is_write(_cmd_p, cmd_len))
                        or _replica_name_is(buffer, cmd_start, cmd_len, "get")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "mget")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "ping")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "exists")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "hget")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "hmget")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "llen")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "lrange")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "dbsize")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "getbit")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "bitcount")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "pfcount")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "cluster")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "config")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "info")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "hello")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "readonly")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "readwrite")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "reset")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "asking")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "quit")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "echo")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "type")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "ttl")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "strlen")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "scard")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "zcard")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "hlen")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "getrange")
                        or _replica_name_is(buffer, cmd_start, cmd_len, "srandmember")
                    )
                    if not _is_read:
                        if self._reject_if_replica_write(writer, fd):
                            # Skip remaining args of this command
                            for _sa in range(num_args - 1):
                                if it_pos < n and buffer[it_pos] == 36:
                                    it_pos += 1
                                    var _al = 0
                                    while it_pos < n and buffer[it_pos] != 13:
                                        _al = _al * 10 + Int(buffer[it_pos] - 48)
                                        it_pos += 1
                                    it_pos += 2 + _al + 2
                            consumed = it_pos
                            continue

                if b0_lower == 103 and cmd_len == 3 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 116: # 'g' 'e' 't' - GET
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed

                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed

                    # Phase 5: MOVED redirect if cluster enabled and key belongs to another node
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    # #45: the lookup applies the key's TTL (the keyspace's
                    # lazy expiry), as every other lookup does.
                    var val = self.keyspace[].get_with_ptr(buffer + it_pos, key_len)
                    # A6: values >512B use writev (bypasses 4MB response buffer for LMCache blobs)
                    writer.append_large_value_response_writev(fd, val)
                    it_pos += key_len + 2
                elif b0_lower == 115 and cmd_len == 3 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 116: # 's' 'e' 't' - SET
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed

                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed

                    var key_ptr = buffer + it_pos
                    var k_len = key_len

                    # Phase 5: MOVED redirect if cluster enabled and key belongs to another node
                    if self.has_cluster and self._moved_if_needed(key_ptr, k_len, writer):
                        # Skip past key + val to consume full command
                        it_pos += key_len + 2
                        # Skip val_len header and value
                        if it_pos < n and buffer[it_pos] == 36:
                            it_pos += 1
                            var skip_vlen = 0
                            while it_pos < n and buffer[it_pos] != 13:
                                skip_vlen = skip_vlen * 10 + Int(buffer[it_pos] - 48)
                                it_pos += 1
                            it_pos += 2 + skip_vlen + 2
                        consumed = it_pos
                        continue

                    it_pos += key_len + 2

                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed

                    it_pos += 1
                    var val_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if val_len < 0 or it_pos + val_len + 2 > n:
                        return consumed

                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    var key_val = GenericValue.borrow_buf(key_ptr, k_len)

                    # gh #163: a value at or above the threshold goes to the
                    # file-backed arena instead of the heap, and the WAL records
                    # a 24-byte pointer instead of the payload. Disabled is
                    # BLOB_TIER_OFF, not 0, so the predicate is exactly one
                    # compare, the taken branch is an outlined @no_inline call
                    # (keeping this — the hottest function in the codebase — at
                    # its pre-gh #163 code size), and the sub-threshold branch
                    # below is byte-for-byte the old hot path.
                    if val_len >= self.blob_threshold:
                        self._set_blob_value(key_val, key_ptr, k_len,
                                             buffer + it_pos, val_len)
                    else:
                        if self.has_wal:
                            _ = self.wal[].append_kv(1, key_ptr, k_len, buffer + it_pos, val_len)
                        # gh #175: overwrite an equal-length heap payload in
                        # place — removes a tcmalloc alloc+free per overwrite.
                        self.keyspace[].set_str_reuse(key_val, buffer + it_pos, val_len)
                    # WATCH: bump key version on mutation (gh #230: from the
                    # hash this arm already has, not a second byte-wise one)
                    if is_not_null(self.key_versions):
                        var _kv_slot = TransactionState.key_slot_from_hash(UInt64(key_val.__hash__()))
                        self.key_versions[_kv_slot] += 1
                    # TTL: SET clears any existing TTL for the key (skip the
                    # probe entirely while no key holds a TTL — gh #175)
                    if is_not_null(self.ttl_map) and self.ttl_map[].size > 0:
                        _ = self.ttl_map[].remove_generic(key_val)

                    writer.append_ok_response()
                    it_pos += val_len + 2
                # Match all four bytes, not just 'p','i': the two-byte form
                # accepted PINX, PIxx and PI\0G as PING (gh #162's rule — match
                # the whole command name). Cheap here, and it stops a future
                # 4-byte `PI**` command being silently shadowed by PING.
                elif b0_lower == 112 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 105 and (buffer[cmd_start + 2] | 0x20) == 110 and (buffer[cmd_start + 3] | 0x20) == 103: # 'p','i','n','g' - PING
                    if num_args == 1:
                        writer.append_pong_response()
                    elif num_args == 2:
                        if it_pos >= n or buffer[it_pos] != 36:
                            return consumed

                        it_pos += 1
                        var msg_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            msg_len = msg_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if msg_len < 0 or it_pos + msg_len + 2 > n:
                            return consumed

                        writer.buffer[writer.offset] = 36 # '$'
                        writer.offset += 1
                        writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(msg_len))
                        writer.buffer[writer.offset] = 13 # '\r'
                        writer.buffer[writer.offset + 1] = 10 # '\n'
                        writer.offset += 2
                        unsafe_memcpy(dest=writer.buffer + writer.offset, src=buffer + it_pos, count=msg_len)
                        writer.offset += msg_len
                        writer.buffer[writer.offset] = 13 # '\r'
                        writer.buffer[writer.offset + 1] = 10 # '\n'
                        writer.offset += 2

                        it_pos += msg_len + 2
                    else:
                        fast_path_ok = False
                        break
                elif b0_lower == 102 and cmd_len == 5: # 'f' - FCALL → slow path (Lua engine)
                    return consumed
                elif b0_lower == 102 and cmd_len == 8: # 'f' - FUNCTION or FLUSHALL
                    # FLUSHALL and FUNCTION both go to the slow path: FLUSHALL is
                    # logged there and parses its ASYNC|SYNC argument.
                    return consumed
                elif b0_lower == 120 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 97 and (buffer[cmd_start + 2] | 0x20) == 100 and (buffer[cmd_start + 3] | 0x20) == 100: # 'x' 'a' 'd' 'd' - XADD
                    # Route to slow path for real stream storage
                    return consumed
                elif b0_lower == 105 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 110 and (buffer[cmd_start + 2] | 0x20) == 99 and (buffer[cmd_start + 3] | 0x20) == 114: # 'i' 'n' 'c' 'r' - INCR
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed

                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed

                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)

                    # Fast path: in-place increment for INT type (single hash lookup, no GenericValue construction)
                    var vptr = self.keyspace[].get_value_ptr(key_val)
                    if is_not_null(vptr) and vptr[].type.value == ValueType.INT:
                        var cur = Int64(vptr[]._data0)
                        if cur == 9223372036854775807:
                            writer.append_error_response("ERR increment or decrement would overflow")
                            it_pos += key_len + 2
                            consumed = it_pos
                            continue
                        vptr[]._data0 = UInt64(cur + 1)
                        writer.append_int_response(Int64(vptr[]._data0))
                        if self.has_wal and is_not_null(self.wal):
                            # gh #216: log the RESOLVED counter, not an empty SET.
                            # A null-value cmd-1 record replays as an empty string,
                            # so the increment was silently lost on recovery.
                            var _cb = stack_allocation[24, UInt8]()
                            var _cl = format_int_to_buf(_cb, 0, Int64(vptr[]._data0))
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue

                    # Key not found: set to 1 directly (skip redundant get())
                    if is_null(vptr):
                        self.keyspace[].set(key_val, GenericValue.from_int(1))
                        writer.append_int_response(1)
                        if self.has_wal and is_not_null(self.wal):
                            var _cb = stack_allocation[24, UInt8]()   # gh #216
                            var _cl = format_int_to_buf(_cb, 0, Int64(1))
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue

                    # Key exists but not INT: must be string — parse it
                    var val = vptr[]
                    var new_val: Int64 = 0
                    var valid = True
                    if val.is_string():
                        var parsed_val: Int64 = 0
                        var is_neg = False
                        if val.type.value == ValueType.STRING_SSO:
                            var length = Int(val._data0 & 0xFF)
                            var d0 = val._data0 >> 8
                            var d1 = val._data1
                            var d2 = val._data2

                            var start_idx = 0
                            if length > 0:
                                var first_char = Int(d0 & 0xFF)
                                if first_char == 45: # '-'
                                    is_neg = True
                                    start_idx = 1
                            if start_idx >= length: valid = False
                            for j in range(start_idx, length):
                                var c: Int
                                if j < 7:
                                    c = Int((d0 >> UInt64(j * 8)) & 0xFF)
                                elif j < 15:
                                    c = Int((d1 >> UInt64((j - 7) * 8)) & 0xFF)
                                else:
                                    c = Int((d2 >> UInt64((j - 15) * 8)) & 0xFF)

                                if c >= 48 and c <= 57:
                                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                                    if parsed_val == -1:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        else:
                            var length = val.string_len()
                            var ptr = val.as_string()
                            var start_idx = 0
                            if length > 0 and ptr[0] == 45:
                                is_neg = True
                                start_idx = 1
                            if start_idx >= length: valid = False
                            for j in range(start_idx, length):
                                var c = Int(ptr[j])
                                if c >= 48 and c <= 57:
                                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                                    if parsed_val == -1:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        if valid:
                            if is_neg: parsed_val = -parsed_val
                            if parsed_val == 9223372036854775807:
                                valid = False
                            else:
                                new_val = parsed_val + 1
                    else:
                        valid = False

                    if valid:
                        self.keyspace[].set(key_val, GenericValue.from_int(new_val))
                        writer.append_int_response(new_val)
                        if self.has_wal and is_not_null(self.wal):
                            # gh #216: this arm (counter held as a string, e.g. SET
                            # then INCR) logged nothing at all, so the increment was
                            # dropped and the key replayed at its pre-INCR value.
                            var _cb = stack_allocation[24, UInt8]()
                            var _cl = format_int_to_buf(_cb, 0, new_val)
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                    elif val.is_container():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_error_response("ERR value is not an integer or out of range")

                    it_pos += key_len + 2
                elif b0_lower == 100 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 99 and (buffer[cmd_start + 3] | 0x20) == 114: # 'd' 'e' 'c' 'r' - DECR
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed

                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed

                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)

                    # Fast path: in-place decrement for INT type (single hash lookup)
                    var vptr = self.keyspace[].get_value_ptr(key_val)
                    if is_not_null(vptr) and vptr[].type.value == ValueType.INT:
                        var cur = Int64(vptr[]._data0)
                        if cur == -9223372036854775808:
                            writer.append_error_response("ERR increment or decrement would overflow")
                            it_pos += key_len + 2
                            consumed = it_pos
                            continue
                        vptr[]._data0 = UInt64(cur - 1)
                        writer.append_int_response(Int64(vptr[]._data0))
                        if self.has_wal and is_not_null(self.wal):
                            var _cb = stack_allocation[24, UInt8]()   # gh #216
                            var _cl = format_int_to_buf(_cb, 0, Int64(vptr[]._data0))
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue

                    # Key not found: set to -1 directly (skip redundant get())
                    if is_null(vptr):
                        self.keyspace[].set(key_val, GenericValue.from_int(-1))
                        writer.append_int_response(-1)
                        if self.has_wal and is_not_null(self.wal):
                            var _cb = stack_allocation[24, UInt8]()   # gh #216
                            var _cl = format_int_to_buf(_cb, 0, Int64(-1))
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue

                    # Key exists but not INT: must be string — parse it
                    var val = vptr[]
                    var new_val: Int64 = 0
                    var valid = True
                    if val.is_string():
                        var parsed_val: Int64 = 0
                        var is_neg = False
                        if val.type.value == ValueType.STRING_SSO:
                            var length = Int(val._data0 & 0xFF)
                            var d0 = val._data0 >> 8
                            var d1 = val._data1
                            var d2 = val._data2

                            var start_idx = 0
                            if length > 0:
                                var first_char = Int(d0 & 0xFF)
                                if first_char == 45: # '-'
                                    is_neg = True
                                    start_idx = 1
                            if start_idx >= length: valid = False
                            for j in range(start_idx, length):
                                var c: Int
                                if j < 7:
                                    c = Int((d0 >> UInt64(j * 8)) & 0xFF)
                                elif j < 15:
                                    c = Int((d1 >> UInt64((j - 7) * 8)) & 0xFF)
                                else:
                                    c = Int((d2 >> UInt64((j - 15) * 8)) & 0xFF)

                                if c >= 48 and c <= 57:
                                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                                    if parsed_val == -1:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        else:
                            var length = val.string_len()
                            var ptr = val.as_string()
                            var start_idx = 0
                            if length > 0 and ptr[0] == 45:
                                is_neg = True
                                start_idx = 1
                            if start_idx >= length: valid = False
                            for j in range(start_idx, length):
                                var c = Int(ptr[j])
                                if c >= 48 and c <= 57:
                                    parsed_val = acc_digit_checked(parsed_val, Int64(c - 48), is_neg, j + 1 == length)
                                    if parsed_val == -1:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                        if valid:
                            if is_neg: parsed_val = -parsed_val
                            if parsed_val == -9223372036854775808:
                                valid = False
                            else:
                                new_val = parsed_val - 1
                    else:
                        valid = False

                    if valid:
                        self.keyspace[].set(key_val, GenericValue.from_int(new_val))
                        writer.append_int_response(new_val)
                        if self.has_wal and is_not_null(self.wal):
                            # gh #216: this arm (counter held as a string, e.g. SET
                            # then INCR) logged nothing at all, so the increment was
                            # dropped and the key replayed at its pre-INCR value.
                            var _cb = stack_allocation[24, UInt8]()
                            var _cl = format_int_to_buf(_cb, 0, new_val)
                            _ = self.wal[].append_kv(UInt8(1), buffer + it_pos, key_len,
                                _cb, _cl)
                    elif val.is_container():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_error_response("ERR value is not an integer or out of range")

                    it_pos += key_len + 2
                elif b0_lower == 100 and cmd_len == 3 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 108: # 'd' 'e' 'l' - DEL
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    var key_ptr = buffer + it_pos
                    var key_val = GenericValue.borrow_buf(key_ptr, key_len)
                    it_pos += key_len + 2
                    var _del_taken = GenericValue()
                    var removed = self.keyspace[].remove_generic_taking(key_val, _del_taken)
                    if removed:
                        free_container(_del_taken)   # gh #369: a DEL'd aggregate is freed
                        if self.has_wal:
                            _ = self.wal[].append(2, key_ptr, key_len)
                        if is_not_null(self.key_versions):
                            var _kv_slot = TransactionState.key_slot_from_hash(UInt64(key_val.__hash__()))
                            self.key_versions[_kv_slot] += 1
                        if is_not_null(self.ttl_map):
                            _ = self.ttl_map[].remove_generic(key_val)
                    writer.append_int_response(Int64(1 if removed else 0))
                elif b0_lower == 101 and cmd_len == 6 and cmd_matches_6(buffer + cmd_start, 101, 120, 105, 115, 116, 115): # EXISTS (gh #225: shared 'e'+6 with EXPIRE)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    # gh #85b: P2 cross-worker routing removed (shared-nothing, gh #48).
                    var exists = not self.keyspace[].get_with_ptr(buffer + it_pos, key_len).is_none()
                    it_pos += key_len + 2
                    writer.append_int_response(Int64(1 if exists else 0))
                elif b0_lower == 104 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 104, 103, 101, 116): # HGET (gh #225: every byte)
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var hget_key_ptr = buffer + it_pos
                    var hget_key_len = key_len
                    var outer_val = self.keyspace[].get_with_ptr(buffer + it_pos, key_len)
                    it_pos += key_len + 2

                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var field_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        field_len = field_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if field_len < 0 or it_pos + field_len + 2 > n:
                        return consumed
                    var field_ptr = buffer + it_pos
                    it_pos += field_len + 2

                    if outer_val.type.value == ValueType.HASH:
                        var hash_ptr = outer_val.as_hash().bitcast[SlabHashMap]()
                        # R3: Check field TTL before returning
                        var field_expired_fp = False
                        # gh #392: the hash's own field TTLs — one null check when
                        # it has none. This built two Strings and a heap key per
                        # HGET whenever ANY key had a TTL, to look up an ambiguous
                        # `key::field` entry in the global table.
                        if Int(hash_ptr[].field_ttl) != 0:
                            var _fg = GenericValue.borrow_buf(field_ptr, field_len)
                            if hash_ptr[].expire_field(_fg, _get_now_ns()):
                                field_expired_fp = True
                                if hash_ptr[].size == 0:   # the last field went: so does the key
                                    _ = remove_and_free(self.keyspace, GenericValue.borrow_buf(hget_key_ptr, hget_key_len))
                        if field_expired_fp:
                            writer.append_null_response()
                        else:
                            var field_val = hash_ptr[].get_with_ptr(field_ptr, field_len)
                            writer.append_bulk_value_response(field_val)
                    elif not outer_val.is_none():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_null_response()
                elif b0_lower == 109 and cmd_len == 4: # 'm' - MGET or MSET
                    var b1_lower = buffer[cmd_start + 1] | 0x20
                    if cmd_matches_4(buffer + cmd_start, 109, 103, 101, 116): # MGET (gh #225: shares 'm'+4 with MOVE)
                        if num_args < 2:
                            fast_path_ok = False
                            break
                        var num_keys = num_args - 1
                        # The header goes out BEFORE the keys are parsed, so a frame
                        # that is not complete yet (a pipelined MGET split across two
                        # reads) must take it back: `return consumed` left `*N` in the
                        # reply buffer, the retry wrote it again, and the client read
                        # an array whose elements were the NEXT replies — silently
                        # desynced for the rest of the connection.
                        var mget_w0 = writer.offset
                        writer.buffer[writer.offset] = 42 # '*'
                        writer.offset += 1
                        writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(num_keys))
                        writer.buffer[writer.offset] = 13 # '\r'
                        writer.buffer[writer.offset + 1] = 10 # '\n'
                        writer.offset += 2
                        if num_keys <= 16:
                            # Two-phase prefetch: parse all keys + prefetch metadata, then lookup
                            var mget_offs = Array[Int, 16](uninitialized=True)
                            var mget_lens = Array[Int, 16](uninitialized=True)
                            var mget_hashes = Array[UInt64, 16](uninitialized=True)
                            var k = 0
                            # Phase 1: collect key positions, compute hashes, prefetch groups
                            while k < num_keys:
                                if it_pos >= n or buffer[it_pos] != 36:
                                    writer.offset = mget_w0
                                    return consumed
                                it_pos += 1
                                var kl = 0
                                while it_pos < n and buffer[it_pos] != 13:
                                    kl = kl * 10 + Int(buffer[it_pos] - 48)
                                    it_pos += 1
                                it_pos += 2
                                if kl < 0 or it_pos + kl + 2 > n:
                                    writer.offset = mget_w0
                                    return consumed
                                mget_offs[k] = it_pos
                                mget_lens[k] = kl
                                var key = GenericValue.borrow_buf(buffer + it_pos, kl)
                                var h = UInt64(key.__hash__())
                                mget_hashes[k] = h
                                var shard_idx = Int(h & 7)
                                var cap_mask = self.keyspace[].shards[shard_idx].capacity - 1
                                var h1 = Int(h >> 7) & cap_mask
                                prefetch(self.keyspace[].shards[shard_idx].metadata + h1)
                                it_pos += kl + 2
                                k += 1
                            # Phase 2: lookup with warmed metadata groups
                            k = 0
                            while k < num_keys:
                                var key = GenericValue.borrow_buf(buffer + mget_offs[k], mget_lens[k])
                                var val = self.keyspace[].get_with_hash(key, mget_hashes[k])
                                writer.append_bulk_value_response(val)
                                k += 1
                        else:
                            # Sequential fallback for large MGET (>16 keys)
                            var keys_parsed = 0
                            while keys_parsed < num_keys:
                                if it_pos >= n or buffer[it_pos] != 36:
                                    writer.offset = mget_w0
                                    return consumed
                                it_pos += 1
                                var key_len = 0
                                while it_pos < n and buffer[it_pos] != 13:
                                    key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                                    it_pos += 1
                                it_pos += 2
                                if key_len < 0 or it_pos + key_len + 2 > n:
                                    writer.offset = mget_w0
                                    return consumed
                                var val = self.keyspace[].get_with_ptr(buffer + it_pos, key_len)
                                writer.append_bulk_value_response(val)
                                it_pos += key_len + 2
                                keys_parsed += 1
                    elif cmd_matches_4(buffer + cmd_start, 109, 115, 101, 116): # MSET (gh #225: every byte)
                        if num_args < 3 or num_args % 2 == 0:
                            fast_path_ok = False
                            break
                        var _mset_end = self._mset_frame(buffer, it_pos, n, num_args, writer)
                        if _mset_end < 0:
                            return consumed
                        it_pos = _mset_end
                    else:
                        fast_path_ok = False
                        break
                elif b0_lower == 104 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 104, 115, 101, 116): # HSET single or multi-field (gh #225: every byte)
                    if num_args == 4:
                        if it_pos >= n or buffer[it_pos] != 36:
                            return consumed
                        it_pos += 1
                        var key_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if key_len < 0 or it_pos + key_len + 2 > n:
                            return consumed
                        # Phase 5: MOVED redirect
                        if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                            it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                            consumed = it_pos
                            continue
                        var key_ptr = buffer + it_pos
                        var k_len = key_len
                        it_pos += key_len + 2
                        if it_pos >= n or buffer[it_pos] != 36:
                            return consumed
                        it_pos += 1
                        var field_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            field_len = field_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if field_len < 0 or it_pos + field_len + 2 > n:
                            return consumed
                        var field_ptr = buffer + it_pos
                        var f_len = field_len
                        it_pos += field_len + 2
                        if it_pos >= n or buffer[it_pos] != 36:
                            return consumed
                        it_pos += 1
                        var val_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if val_len < 0 or it_pos + val_len + 2 > n:
                            return consumed
                        # Route vector field to HNSW ingest buffer (single-field HSET)
                        var _sh_ptr = self.shared_hnsw
                        if is_not_null(_sh_ptr):
                            if _sh_ptr[].pre_index_ready:
                                if val_len == _sh_ptr[].pre_dim * 4:
                                    if f_len == _sh_ptr[].pre_vector_field_len:
                                        var _is_vf = True
                                        var _vfn_ptr = _sh_ptr[].pre_vector_field_name.unsafe_ptr()
                                        for _vbi in range(f_len):
                                            if (field_ptr[_vbi] | 0x20) != _vfn_ptr[_vbi]:
                                                _is_vf = False
                                                break
                                        if _is_vf:
                                            var _slot = _sh_ptr[].add_ingest_vector(0, (buffer + it_pos).bitcast[Float32]())
                                            if _slot >= 0:
                                                # Store reverse mapping: slot ID → original hash key
                                                var _hk_buf = stack_allocation[30, UInt8]()
                                                _hk_buf[0]=95;_hk_buf[1]=95;_hk_buf[2]=104;_hk_buf[3]=107;_hk_buf[4]=95;_hk_buf[5]=95 # __hk__
                                                var _hk_end = format_int_to_buf(_hk_buf, 6, Int64(_slot))
                                                self.keyspace[].set(GenericValue.borrow(_hk_buf, _hk_end), GenericValue.borrow_buf(key_ptr, k_len))
                                                # gh #211: effect-log the __hk__ SET (gh #170 rule —
                                                # every keyspace mutation replays); it's the only
                                                # durable slot→key source for keys >31B.
                                                if self.has_wal:
                                                    _ = self.wal[].append_kv(1, _hk_buf, _hk_end, key_ptr, k_len)
                                                # Cross-worker shared mapping: byte 0 = len, bytes 1..31 = key.
                                                # gh #211: >31B keys must stay len 0 (fall through to the
                                                # __hk__ probe) — a truncated key stored as complete
                                                # resolves to a wrong doc key.
                                                if is_not_null(_sh_ptr[].hk_keys_buf) and _slot < _sh_ptr[].hk_max_elements and k_len <= 31:
                                                    var _dst = _sh_ptr[].hk_keys_buf + _slot * 32
                                                    _dst[0] = UInt8(k_len)
                                                    unsafe_memcpy(dest=_dst + 1, src=key_ptr, count=k_len)
                        var key_val = GenericValue.borrow_buf(key_ptr, k_len)
                        var field_val = GenericValue.borrow_buf(field_ptr, f_len)
                        var val_val = GenericValue.borrow_buf(buffer + it_pos, val_len)
                        if is_not_null(self.key_versions):
                            var _kv_slot = TransactionState.key_slot_from_hash(UInt64(key_val.__hash__()))
                            self.key_versions[_kv_slot] += 1
                        var val = self.keyspace[].get(key_val)
                        var valid = True
                        if val.is_none():
                            var hash_ptr: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
                            if self.hash_map_pool[].head < self.hash_map_pool[].capacity:
                                hash_ptr = self.hash_map_pool[].acquire(); hash_ptr[].reset()
                            else:
                                hash_ptr = alloc[SlabHashMap](1); hash_ptr.unsafe_write(SlabHashMap(16))
                            var new_val = GenericValue()
                            new_val.type = ValueType(ValueType.HASH)
                            new_val.set_ptr(hash_ptr.bitcast[NoneType]())
                            self.keyspace[].set(key_val, new_val)
                            hash_ptr[].set(field_val, val_val)
                            # gh #170: effect record (cmd 5) — replayable, unlike the old key-only stub
                            if self.has_wal:
                                _ = self.wal[].append_field_kv(5, key_ptr, k_len,
                                                               field_ptr, f_len,
                                                               buffer + it_pos, val_len)
                            writer.append_int_response(Int64(1))
                        elif val.type.value == ValueType.HASH:
                            var hash_ptr = val.as_hash().bitcast[SlabHashMap]()
                            # gh #232: Redis returns the number of fields ADDED, so a
                            # pure update is 0. This answered 1 unconditionally, which
                            # breaks the documented way to ask "did I create this field
                            # or overwrite one" (first-writer-wins, dedup, presence).
                            # `size` only moves on insert, so comparing it costs two
                            # field reads rather than a second hash probe on a hot arm.
                            var _hs_before = hash_ptr[].size
                            hash_ptr[].set(field_val, val_val)
                            if Int(hash_ptr[].field_ttl) != 0:   # gh #392: HSET clears an overwritten field's TTL (Redis)
                                _ = hash_ptr[].clear_field_deadline(field_val)
                            if self.has_wal:
                                _ = self.wal[].append_field_kv(5, key_ptr, k_len,
                                                               field_ptr, f_len,
                                                               buffer + it_pos, val_len)
                            writer.append_int_response(Int64(1) if hash_ptr[].size > _hs_before else Int64(0))
                        else:
                            valid = False
                        if not valid:
                            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                        it_pos += val_len + 2
                    elif num_args >= 6 and (num_args - 2) % 2 == 0:
                        # Multi-field HSET: num_args = 2 + 2*num_fields
                        # VectorDBBench sends: HSET key id val metadata val vector <float32_bytes>
                        # Whole frame first: fields are applied as they are parsed,
                        # so a split frame applied some, retried, and answered with
                        # a count missing them (and the WRONGTYPE skip had no bounds).
                        if _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
                            return consumed
                        it_pos += 1
                        var k_len2 = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            k_len2 = k_len2 * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if k_len2 < 0 or it_pos + k_len2 + 2 > n:
                            return consumed
                        # Phase 5: MOVED redirect
                        if self.has_cluster and self._moved_if_needed(buffer + it_pos, k_len2, writer, fd):
                            it_pos = self._skip_args(buffer, it_pos + k_len2 + 2, n, num_args - 2)
                            consumed = it_pos
                            continue
                        var key_ptr2 = buffer + it_pos
                        it_pos += k_len2 + 2
                        # Parse integer key for HNSW node ID
                        var key_id = 0
                        for ki in range(k_len2):
                            var b = key_ptr2[ki]
                            if b >= 48 and b <= 57:
                                key_id = key_id * 10 + Int(b - 48)
                        # Ensure HASH entry in keyspace
                        var key_val2 = GenericValue.borrow_buf(key_ptr2, k_len2)
                        var kv2 = self.keyspace[].get(key_val2)
                        var hash_ptr2: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
                        if kv2.is_none():
                            if self.hash_map_pool[].head < self.hash_map_pool[].capacity:
                                hash_ptr2 = self.hash_map_pool[].acquire(); hash_ptr2[].reset()
                            else:
                                hash_ptr2 = alloc[SlabHashMap](1); hash_ptr2.unsafe_write(SlabHashMap(16))
                            var nv2 = GenericValue()
                            nv2.type = ValueType(ValueType.HASH)
                            nv2.set_ptr(hash_ptr2.bitcast[NoneType]())
                            self.keyspace[].set(key_val2, nv2)
                        elif kv2.type.value == ValueType.HASH:
                            hash_ptr2 = kv2.as_hash().bitcast[SlabHashMap]()
                        else:
                            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                            # Skip remaining field-value tokens
                            var _skip = num_args - 2
                            while _skip > 0 and it_pos < n:
                                if buffer[it_pos] == 36:
                                    it_pos += 1
                                    var _slen = 0
                                    while it_pos < n and buffer[it_pos] != 13:
                                        _slen = _slen * 10 + Int(buffer[it_pos] - 48); it_pos += 1
                                    it_pos += 2 + _slen + 2
                                _skip -= 1
                            consumed = it_pos
                            continue
                        var nf = (num_args - 2) // 2
                        # gh #232: reply is the number of fields ADDED, not the
                        # number supplied — a size delta counts only new fields.
                        var _hms_before = hash_ptr2[].size
                        for _ in range(nf):
                            if it_pos >= n or buffer[it_pos] != 36:
                                return consumed
                            it_pos += 1
                            var f_len2 = 0
                            while it_pos < n and buffer[it_pos] != 13:
                                f_len2 = f_len2 * 10 + Int(buffer[it_pos] - 48)
                                it_pos += 1
                            it_pos += 2
                            if f_len2 < 0 or it_pos + f_len2 + 2 > n:
                                return consumed
                            var f_ptr2 = buffer + it_pos
                            it_pos += f_len2 + 2
                            if it_pos >= n or buffer[it_pos] != 36:
                                return consumed
                            it_pos += 1
                            var v_len2 = 0
                            while it_pos < n and buffer[it_pos] != 13:
                                v_len2 = v_len2 * 10 + Int(buffer[it_pos] - 48)
                                it_pos += 1
                            it_pos += 2
                            if v_len2 < 0 or it_pos + v_len2 + 2 > n:
                                return consumed
                            # Route to shared ingest buffer if this is the vector field
                            # V3.1: use SharedHNSWView for cross-worker coordination
                            var is_vec2 = False
                            if is_not_null(self.shared_hnsw):
                                if self.shared_hnsw[].pre_index_ready:
                                    if v_len2 == self.shared_hnsw[].pre_dim * 4:
                                        is_vec2 = True
                            if is_vec2 and f_len2 == self.shared_hnsw[].pre_vector_field_len:
                                var _vfn2 = self.shared_hnsw[].pre_vector_field_name.unsafe_ptr()
                                for bi in range(f_len2):
                                    if (f_ptr2[bi] | 0x20) != _vfn2[bi]:
                                        is_vec2 = False
                                        break
                            elif is_vec2:
                                is_vec2 = False
                            if is_vec2:
                                var _slot2 = self.shared_hnsw[].add_ingest_vector(0, (buffer + it_pos).bitcast[Float32]())
                                if _slot2 >= 0:
                                    var _hk2_buf = stack_allocation[30, UInt8]()
                                    _hk2_buf[0]=95;_hk2_buf[1]=95;_hk2_buf[2]=104;_hk2_buf[3]=107;_hk2_buf[4]=95;_hk2_buf[5]=95
                                    var _hk2_end = format_int_to_buf(_hk2_buf, 6, Int64(_slot2))
                                    self.keyspace[].set(GenericValue.borrow(_hk2_buf, _hk2_end), GenericValue.borrow_buf(key_ptr2, k_len2))
                                    # gh #211: effect-log the __hk__ SET (see single-field HSET above)
                                    if self.has_wal:
                                        _ = self.wal[].append_kv(1, _hk2_buf, _hk2_end, key_ptr2, k_len2)
                                    # Cross-worker shared mapping (see single-field HSET above for
                                    # rationale; gh #211: >31B keys stay len 0, never truncated)
                                    if is_not_null(self.shared_hnsw[].hk_keys_buf) and _slot2 < self.shared_hnsw[].hk_max_elements and k_len2 <= 31:
                                        var _dst2 = self.shared_hnsw[].hk_keys_buf + _slot2 * 32
                                        _dst2[0] = UInt8(k_len2)
                                        unsafe_memcpy(dest=_dst2 + 1, src=key_ptr2, count=k_len2)
                            # gh #360: the vector field is ALSO stored in the hash.
                            # It used to go to the HNSW ingest buffer only, so
                            # `HGET key <vector-field>` answered nil after a
                            # multi-field HSET but not after a single-field one,
                            # and RETURN / RedisVL return_fields got nothing.
                            # Redis stores every field; so does Pion now.
                            var fv2 = GenericValue.borrow_buf(f_ptr2, f_len2)
                            var vv2 = GenericValue.borrow_buf(buffer + it_pos, v_len2)
                            hash_ptr2[].set(fv2, vv2)
                            if Int(hash_ptr2[].field_ttl) != 0:   # gh #392: HSET clears an overwritten field's TTL (Redis)
                                _ = hash_ptr2[].clear_field_deadline(fv2)
                            # gh #170: effect record per stored field
                            if self.has_wal:
                                _ = self.wal[].append_field_kv(5, key_ptr2, k_len2,
                                                               f_ptr2, f_len2,
                                                               buffer + it_pos, v_len2)
                            it_pos += v_len2 + 2
                        writer.append_int_response(Int64(hash_ptr2[].size - _hms_before))
                    else:
                        return consumed
                elif b0_lower == 108 and cmd_len == 5 and (buffer[cmd_start + 1] | 0x20) == 112 and (buffer[cmd_start + 2] | 0x20) == 117 and (buffer[cmd_start + 3] | 0x20) == 115 and (buffer[cmd_start + 4] | 0x20) == 104: # 'l' 'p' 'u' 's' 'h' - LPUSH
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_ptr = buffer + it_pos
                    var k_len = key_len
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var val_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if val_len < 0 or it_pos + val_len + 2 > n:
                        return consumed
                    var key_val = GenericValue.borrow_buf(key_ptr, k_len)
                    var val_val = GenericValue.from_ptr(buffer + it_pos, val_len)
                    var val = self.keyspace[].get(key_val)
                    var valid = True
                    if val.is_none():
                        var list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
                        var new_val = GenericValue()
                        new_val.type = ValueType(ValueType.LIST)
                        new_val.set_ptr(list_ptr.bitcast[NoneType]())
                        self.keyspace[].set(key_val, new_val)
                        list_ptr[].lpush(val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_kv(6, key_ptr, k_len, buffer + it_pos, val_len)
                        writer.append_int_response(Int64(list_ptr[].llen()))
                    elif val.type.value == ValueType.LIST:
                        var list_ptr = val.as_list().bitcast[SlabList]()
                        list_ptr[].lpush(val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_kv(6, key_ptr, k_len, buffer + it_pos, val_len)
                        writer.append_int_response(Int64(list_ptr[].llen()))
                    else:
                        valid = False
                    if not valid:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    it_pos += val_len + 2
                elif b0_lower == 108 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 112 and (buffer[cmd_start + 2] | 0x20) == 111 and (buffer[cmd_start + 3] | 0x20) == 112: # 'l' 'p' 'o' 'p' - LPOP
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var val = self.keyspace[].get(key_val)
                    if val.type.value == ValueType.LIST:
                        var list_ptr = val.as_list().bitcast[SlabList]()
                        if list_ptr[].size == 0:
                            # Empty list fast path — avoid full lpop() call + GenericValue construction
                            writer.append_null_response()
                        else:
                            var popped = list_ptr[].lpop()
                            if self.has_wal:   # gh #170
                                _ = self.wal[].append(13, buffer + it_pos, key_len)
                            writer.append_bulk_value_response(popped)
                            # lpop/rpop hand back an OWNED value (a heap copy from the ziplist, or the
                            # stored element itself when segmented); the reply above copied it. Never
                            # freed: RPUSH/LPOP of 100 KB messages grew RSS by ~92 KB per message.
                            popped.free_str_payload()
                            # gh #234: Redis removes an aggregate the moment its
                            # last element goes. This MUST follow the response
                            # write — `popped` borrows into the list we drop.
                            if list_ptr[].size == 0:
                                free_container(val)   # gh #369: the emptied container is freed, not leaked
                                _ = self.keyspace[].remove_generic(key_val)
                                if self.has_wal:
                                    _ = self.wal[].append(2, buffer + it_pos, key_len)
                    # gh #232: a key holding a non-list answered nil, not
                    # WRONGTYPE. The extra compare rides the not-a-list branch
                    # only — the served path above is untouched.
                    elif not val.is_none():
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_null_response()
                    it_pos += key_len + 2
                elif b0_lower == 114 and cmd_len == 5 and (buffer[cmd_start + 1] | 0x20) == 112 and (buffer[cmd_start + 2] | 0x20) == 117 and (buffer[cmd_start + 3] | 0x20) == 115 and (buffer[cmd_start + 4] | 0x20) == 104: # 'r' 'p' 'u' 's' 'h' - RPUSH
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_ptr = buffer + it_pos
                    var k_len = key_len
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var val_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if val_len < 0 or it_pos + val_len + 2 > n:
                        return consumed
                    var key_val = GenericValue.borrow_buf(key_ptr, k_len)
                    var val_val = GenericValue.from_ptr(buffer + it_pos, val_len)
                    var val = self.keyspace[].get(key_val)
                    var valid = True
                    if val.is_none():
                        var list_ptr = self.list_pool[].acquire(); list_ptr[].reset()
                        var new_val = GenericValue()
                        new_val.type = ValueType(ValueType.LIST)
                        new_val.set_ptr(list_ptr.bitcast[NoneType]())
                        self.keyspace[].set(key_val, new_val)
                        list_ptr[].rpush(val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_kv(7, key_ptr, k_len, buffer + it_pos, val_len)
                        writer.append_int_response(Int64(list_ptr[].llen()))
                    elif val.type.value == ValueType.LIST:
                        var list_ptr = val.as_list().bitcast[SlabList]()
                        list_ptr[].rpush(val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_kv(7, key_ptr, k_len, buffer + it_pos, val_len)
                        writer.append_int_response(Int64(list_ptr[].llen()))
                    else:
                        valid = False
                    if not valid:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    it_pos += val_len + 2
                elif b0_lower == 114 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 112 and (buffer[cmd_start + 2] | 0x20) == 111 and (buffer[cmd_start + 3] | 0x20) == 112: # 'r' 'p' 'o' 'p' - RPOP
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var val = self.keyspace[].get(key_val)
                    if val.type.value == ValueType.LIST:
                        var list_ptr = val.as_list().bitcast[SlabList]()
                        if list_ptr[].size == 0:
                            writer.append_null_response()
                        else:
                            var popped = list_ptr[].rpop()
                            if self.has_wal:   # gh #170
                                _ = self.wal[].append(14, buffer + it_pos, key_len)
                            writer.append_bulk_value_response(popped)
                            # lpop/rpop hand back an OWNED value (a heap copy from the ziplist, or the
                            # stored element itself when segmented); the reply above copied it. Never
                            # freed: RPUSH/LPOP of 100 KB messages grew RSS by ~92 KB per message.
                            popped.free_str_payload()
                            # gh #234: Redis removes an aggregate the moment its
                            # last element goes. This MUST follow the response
                            # write — `popped` borrows into the list we drop.
                            if list_ptr[].size == 0:
                                free_container(val)   # gh #369: the emptied container is freed, not leaked
                                _ = self.keyspace[].remove_generic(key_val)
                                if self.has_wal:
                                    _ = self.wal[].append(2, buffer + it_pos, key_len)
                    elif not val.is_none():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_null_response()
                    it_pos += key_len + 2
                elif b0_lower == 122 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 97 and (buffer[cmd_start + 2] | 0x20) == 100 and (buffer[cmd_start + 3] | 0x20) == 100: # 'z' 'a' 'd' 'd' - ZADD
                    if num_args != 4 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_ptr = buffer + it_pos
                    var k_len = key_len
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var score_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        score_len = score_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if score_len < 0 or it_pos + score_len + 2 > n:
                        return consumed    # split frame: the score is not all here yet
                    var score_ptr = buffer + it_pos
                    var s_len = score_len
                    it_pos += score_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var val_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if val_len < 0 or it_pos + val_len + 2 > n:
                        return consumed
                    # gh #179: this arm only serves integer scores. Any other
                    # byte ('.', 'e', flags like NX, garbage) bails the whole
                    # command to the slow path's atof — the old code broke at
                    # '.' and stored the truncated integer part. The check runs
                    # BEFORE the member GenericValue is built: from_ptr heap-
                    # copies >23B members and the bail path would leak the copy.
                    var s_is_neg = False
                    var s_idx = 0
                    if s_len > 0 and score_ptr[0] == 45:
                        s_is_neg = True
                        s_idx = 1
                    var int_score: Int64 = 0
                    # <= 15 digits: exact in a Float64 and no Int64 wrap. A longer
                    # score wrapped here (99999999999999999999 stored as
                    # 7766279631452241920) where Redis reads it as 1e20 — the
                    # slow path parses it the way Redis does (gh #393).
                    var s_is_int = s_len > s_idx and s_len - s_idx <= 15
                    for j in range(s_idx, s_len):
                        var c = Int(score_ptr[j])
                        if c >= 48 and c <= 57:
                            int_score = int_score * 10 + Int64(c - 48)
                        else:
                            s_is_int = False
                            break
                    if not s_is_int:
                        return consumed
                    var parsed_score: Float64
                    if s_is_neg:
                        parsed_score = Float64(-int_score)
                    else:
                        parsed_score = Float64(int_score)
                    var val_val = GenericValue.from_ptr(buffer + it_pos, val_len)
                    var key_val = GenericValue.borrow_buf(key_ptr, k_len)
                    var val = self.keyspace[].get(key_val)
                    var valid = True
                    if val.is_none():
                        var zset_ptr = self.skip_list_pool[].acquire()   # gh: NO reset() — construction follows
                        zset_ptr.unsafe_write(SlabSkipList(16))
                        var new_val = GenericValue()
                        new_val.type = ValueType(ValueType.ZSET)
                        new_val.set_ptr(zset_ptr.bitcast[NoneType]())
                        self.keyspace[].set(key_val, new_val)
                        var zadd_added = zset_ptr[].upsert(parsed_score, val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_scored(9, key_ptr, k_len, parsed_score,
                                                         buffer + it_pos, val_len)
                        writer.append_int_response(Int64(zadd_added))
                    elif val.type.value == ValueType.ZSET:
                        var zset_ptr = val.as_zset().bitcast[SlabSkipList]()
                        # gh #187: upsert dedups; reply counts new members only
                        var zadd_added = zset_ptr[].upsert(parsed_score, val_val)
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_scored(9, key_ptr, k_len, parsed_score,
                                                         buffer + it_pos, val_len)
                        writer.append_int_response(Int64(zadd_added))
                    else:
                        valid = False
                    if not valid:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    it_pos += val_len + 2
                elif b0_lower == 122 and cmd_len == 7 and cmd_matches_7(buffer + cmd_start, 122, 112, 111, 112, 109, 105, 110):  # ZPOPMIN (gh #225: every byte)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var val = self.keyspace[].get(key_val)
                    if val.type.value == ValueType.ZSET:
                        var zset_ptr = val.as_zset().bitcast[SlabSkipList]()
                        var res = zset_ptr[].pop_min()
                        if not res.valid or res.obj.is_none():
                            writer.append_empty_array_response()
                        else:
                            var score = res.score
                            var obj = res.obj
                            if self.has_wal:   # gh #170: resolved effect = ZREM
                                var _zb = stack_allocation[64, UInt8]()
                                var _zl = 0
                                var _zp = gv_bytes(obj, _zb, _zl)
                                _ = self.wal[].append_kv(12, buffer + it_pos, key_len, _zp, _zl)
                            writer.buffer[writer.offset] = 42 # '*'
                            writer.offset += 1
                            writer.buffer[writer.offset] = 50 # '2'
                            writer.offset += 1
                            writer.buffer[writer.offset] = 13 # '\r'
                            writer.buffer[writer.offset + 1] = 10 # '\n'
                            writer.offset += 2
                            writer.append_bulk_value_response(obj)
                            # gh #251: `Int64(score)` TRUNCATED — ZPOPMIN on a
                            # member scored 1.5 replied "1". ZPOPMAX was already
                            # correct, so the two ends of the same command family
                            # disagreed. Integer-valued scores still emit bare
                            # digits (Redis prints "3", not "3.0").
                            # #18: and never through Int64(), which read ±inf
                            # back as INT64_MIN on x86. RESP3: a double, as
                            # Redis sends it (same bytes as before on RESP2).
                            writer.append_score_response(score)
                            # gh #394: pop_min hands back the node's own payload; the
                            # reply above copied it, and nothing else holds it.
                            obj.free_str_payload()
                            # gh #234: after the response.
                            if zset_ptr[].length == 0:
                                free_container(val)   # gh #369: the emptied container is freed, not leaked
                                _ = self.keyspace[].remove_generic(key_val)
                                if self.has_wal:
                                    _ = self.wal[].append(2, buffer + it_pos, key_len)
                    elif val.is_none():
                        # gh #251: a missing key is an EMPTY ARRAY, not nil —
                        # the reply shape has to match the popped-pair shape so
                        # a client can treat it uniformly. The slow path already
                        # answered this correctly; only the fast path did not.
                        writer.append_empty_array_response()
                    else:
                        # gh #232: the deferred "Pion permissive" remainder,
                        # now closed. The compare rides the not-a-zset branch,
                        # which ZPOPMIN's gate row never takes.
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    it_pos += key_len + 2
                elif b0_lower == 115 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 115, 97, 100, 100): # SADD (gh #225: every byte)
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_ptr = buffer + it_pos
                    var k_len = key_len
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var val_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        val_len = val_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if val_len < 0 or it_pos + val_len + 2 > n:
                        return consumed
                    var key_val = GenericValue.borrow_buf(key_ptr, k_len)
                    var val_val = GenericValue.borrow_buf(buffer + it_pos, val_len)
                    var val = self.keyspace[].get(key_val)
                    var valid = True
                    if val.is_none():
                        var set_ptr: UnsafePointer[SlabHashMap, MutUntrackedOrigin]
                        if self.hash_map_pool[].head < self.hash_map_pool[].capacity:
                            set_ptr = self.hash_map_pool[].acquire(); set_ptr[].reset()
                        else:
                            set_ptr = alloc[SlabHashMap](1); set_ptr.unsafe_write(SlabHashMap(16))
                        var new_val = GenericValue()
                        new_val.type = ValueType(ValueType.SET)
                        new_val.set_ptr(set_ptr.bitcast[NoneType]())
                        self.keyspace[].set(key_val, new_val)
                        set_ptr[].set(val_val, GenericValue.from_int(1))
                        if self.has_wal:   # gh #170
                            _ = self.wal[].append_kv(8, key_ptr, k_len, buffer + it_pos, val_len)
                        writer.append_int_response(Int64(1))
                    elif val.type.value == ValueType.SET:
                        var set_ptr = val.as_set().bitcast[SlabHashMap]()
                        var exists = set_ptr[].get(val_val)
                        if exists.is_none():
                            set_ptr[].set(val_val, GenericValue.from_int(1))
                            if self.has_wal:   # gh #170
                                _ = self.wal[].append_kv(8, key_ptr, k_len, buffer + it_pos, val_len)
                            writer.append_int_response(Int64(1))
                        else:
                            writer.append_int_response(Int64(0))
                    else:
                        valid = False
                    if not valid:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    it_pos += val_len + 2
                elif b0_lower == 115 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 115, 112, 111, 112): # SPOP (gh #225: every byte)
                    if (num_args != 2 and num_args != 3) or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var _sp_kp = buffer + it_pos   # gh #170: key bytes for effect records
                    it_pos += key_len + 2
                    # Parse optional count argument (SPOP key count)
                    var spop_count = 1
                    if num_args == 3:
                        if it_pos >= n or buffer[it_pos] != 36:
                            return consumed
                        it_pos += 1
                        var cnt_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            cnt_len = cnt_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if cnt_len < 0 or it_pos + cnt_len + 2 > n:
                            return consumed
                        # Strict: `SPOP k abc` folded the letters into a huge
                        # count and popped (and deleted) the whole set; <= 0
                        # was coerced to 1. Redis refuses both and pops nothing
                        # for 0.
                        var _spc = parse_int64_strict(buffer + it_pos, cnt_len)
                        it_pos += cnt_len + 2
                        if not _spc.ok:
                            key_val.free_str_payload()
                            writer.append_error_response("ERR value is not an integer or out of range")
                            consumed = it_pos
                            continue
                        if _spc.value < 0:
                            key_val.free_str_payload()
                            writer.append_error_response("ERR value is out of range, must be positive")
                            consumed = it_pos
                            continue
                        spop_count = Int(_spc.value)
                    var val = self.keyspace[].get(key_val)
                    if val.type.value == ValueType.SET:
                        var set_ptr = val.as_set().bitcast[SlabHashMap]()
                        if num_args == 2:
                            # No count arg: return single bulk string (Redis compat)
                            var popped = set_ptr[].pop_random(self.prng)
                            if self.has_wal and not popped.is_none():  # gh #170: effect = SREM
                                var _sb = stack_allocation[64, UInt8]()
                                var _sl = 0
                                var _sp = gv_bytes(popped, _sb, _sl)
                                _ = self.wal[].append_kv(11, _sp_kp, key_len, _sp, _sl)
                            writer.append_bulk_value_response(popped)
                            popped.free_str_payload()   # gh #394: pop_random hands the member over
                        else:
                            # Count arg: return array of popped elements
                            var actual_count = min(spop_count, set_ptr[].size)
                            # RESP3: a set, as Redis; and no String built on the fast path.
                            writer.append_set_header(actual_count)
                            for _pi in range(actual_count):
                                var popped = set_ptr[].pop_random(self.prng)
                                if self.has_wal and not popped.is_none():  # gh #170
                                    var _sb = stack_allocation[64, UInt8]()
                                    var _sl = 0
                                    var _sp = gv_bytes(popped, _sb, _sl)
                                    _ = self.wal[].append_kv(11, _sp_kp, key_len, _sp, _sl)
                                writer.append_bulk_value_response(popped)
                                popped.free_str_payload()   # gh #394
                        # gh #234: both SPOP shapes land here, after every
                        # response that borrows into the set.
                        if set_ptr[].size == 0:
                            free_container(val)   # gh #369: the emptied container is freed, not leaked
                            _ = self.keyspace[].remove_generic(key_val)
                            if self.has_wal:
                                _ = self.wal[].append(2, _sp_kp, key_len)
                    elif not val.is_none():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        # gh #251: WITH and WITHOUT `count` are different reply
                        # types, and on a missing key the split flips.
                        if num_args == 2:
                            writer.append_null_response()
                        else:
                            writer.append_set_header(0)   # RESP3 `~0`, RESP2 `*0`
                elif b0_lower == 108 and cmd_len == 6 and cmd_matches_6(buffer + cmd_start, 108, 114, 97, 110, 103, 101): # LRANGE (gh #225: every byte)
                    if num_args != 4 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var lrange_key_ptr = buffer + it_pos
                    var lrange_key_len = key_len
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var start_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        start_len = start_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if start_len < 0 or it_pos + start_len + 2 > n:
                        return consumed    # split frame (this answered with garbage bounds)
                    var start_ptr = buffer + it_pos
                    var st_len = start_len
                    it_pos += start_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var stop_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        stop_len = stop_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if stop_len < 0 or it_pos + stop_len + 2 > n:
                        return consumed
                    var stop_ptr = buffer + it_pos
                    var sp_len = stop_len
                    it_pos += stop_len + 2
                    # gh #393: strict, as Redis's string2ll. The digit loop
                    # folded ANY byte into the number, so `LRANGE k abc 2` and
                    # `LRANGE k 0 1.5` answered a plausible range, not an error.
                    var _lr_s = parse_int64_strict(start_ptr, st_len)
                    var _lr_e = parse_int64_strict(stop_ptr, sp_len)
                    if not _lr_s.ok or not _lr_e.ok:
                        writer.append_error_response("ERR value is not an integer or out of range")
                        consumed = it_pos
                        continue
                    var start_idx = Int(_lr_s.value)
                    var stop_idx = Int(_lr_e.value)
                    var val = self.keyspace[].get_with_ptr(lrange_key_ptr, lrange_key_len)
                    if val.type.value == ValueType.LIST:
                        var list_ptr = val.as_list().bitcast[SlabList]()
                        var size = list_ptr[].size
                        var start = start_idx
                        var stop = stop_idx
                        if start < 0:
                            start = size + start
                            if start < 0: start = 0
                        if stop < 0:
                            stop = size + stop
                            if stop < 0: stop = -1
                        if stop >= size: stop = size - 1
                        if start > stop or start >= size:
                            writer.append_empty_array_response()
                        else:
                            var count = stop - start + 1
                            writer.buffer[writer.offset] = 42 # '*'
                            writer.offset += 1
                            writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(count))
                            writer.buffer[writer.offset] = 13 # '\r'
                            writer.buffer[writer.offset + 1] = 10 # '\n'
                            writer.offset += 2
                            if is_not_null(list_ptr[].zip_buf):
                                var offset = 0
                                var idx = 0
                                # Skip entries before start
                                while idx < start and idx < size:
                                    var v_len = Int((list_ptr[].zip_buf + offset).bitcast[UInt16]()[])
                                    offset += 2 + v_len
                                    idx += 1
                                # Write entries from start to stop
                                while idx <= stop and idx < size:
                                    var v_len = Int((list_ptr[].zip_buf + offset).bitcast[UInt16]()[])
                                    writer.buffer[writer.offset] = 36 # '$'
                                    writer.offset += 1
                                    # Inline 1-or-2-digit length write (v_len is typically 1-64)
                                    if v_len < 10:
                                        writer.buffer[writer.offset] = UInt8(48 + v_len)
                                        writer.offset += 1
                                    else:
                                        writer.buffer[writer.offset] = UInt8(48 + v_len // 10)
                                        writer.buffer[writer.offset + 1] = UInt8(48 + v_len % 10)
                                        writer.offset += 2
                                    writer.buffer[writer.offset] = 13 # '\r'
                                    writer.buffer[writer.offset + 1] = 10 # '\n'
                                    writer.offset += 2
                                    unsafe_memcpy(dest=writer.buffer + writer.offset, src=list_ptr[].zip_buf + offset + 2, count=v_len)
                                    writer.offset += v_len
                                    writer.buffer[writer.offset] = 13 # '\r'
                                    writer.buffer[writer.offset + 1] = 10 # '\n'
                                    writer.offset += 2
                                    offset += 2 + v_len
                                    idx += 1
                            else:
                                # Quicklist sequential traversal (no pointer chasing)
                                var global_idx = 0
                                comptime SEG = SlabList.SEG_SIZE
                                # Phase 1: active_head_data [head_off..SEG-1]
                                var hi = list_ptr[].head_off
                                # live range only: head_end < SEG after a pop refill, and head
                                # segs below head_segs_start were handed to the tail side
                                while hi < list_ptr[].head_end and global_idx <= stop:
                                    if global_idx >= start:
                                        writer.append_bulk_value_response(list_ptr[].active_head_data[hi])
                                    hi += 1
                                    global_idx += 1
                                # Phase 2: committed head segs (most recent first)
                                var seg = list_ptr[].head_seg_count - 1
                                while seg >= list_ptr[].head_segs_start and global_idx <= stop:
                                    var j = 0
                                    while j < SEG and global_idx <= stop:
                                        if global_idx >= start:
                                            writer.append_bulk_value_response(list_ptr[].head_segs[seg][j])
                                        j += 1
                                        global_idx += 1
                                    seg -= 1
                                # Phase 3: committed tail segs (oldest first)
                                var tseg = 0
                                while tseg < list_ptr[].tail_seg_count and global_idx <= stop:
                                    var j = 0
                                    while j < SEG and global_idx <= stop:
                                        if global_idx >= start:
                                            writer.append_bulk_value_response(list_ptr[].tail_segs[tseg][j])
                                        j += 1
                                        global_idx += 1
                                    tseg += 1
                                # Phase 4: active_tail_data [0..tail_count-1]
                                var tk = 0
                                while tk < list_ptr[].tail_count and global_idx <= stop:
                                    if global_idx >= start:
                                        writer.append_bulk_value_response(list_ptr[].active_tail_data[tk])
                                    tk += 1
                                    global_idx += 1
                    elif not val.is_none():   # gh #232
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                    else:
                        writer.append_empty_array_response()
                elif b0_lower == 108 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 108, 108, 101, 110): # LLEN (gh #225: every byte)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var llen_val = self.keyspace[].get_with_ptr(buffer + it_pos, key_len)
                    it_pos += key_len + 2
                    if llen_val.is_none():
                        writer.append_int_response(0)
                    elif llen_val.type.value == ValueType.LIST:
                        var list_ptr = llen_val.as_list().bitcast[SlabList]()
                        writer.append_int_response(Int64(list_ptr[].llen()))
                    else:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                elif b0_lower == 103 and cmd_len == 6 and cmd_matches_6(buffer + cmd_start, 103, 101, 116, 98, 105, 116): # GETBIT
                    if num_args != 3 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var off_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        off_len = off_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if off_len < 0 or it_pos + off_len + 2 > n:
                        return consumed
                    # Strict, and bounded as Redis bounds it (0 <= offset < 2^32).
                    # The digit loop this replaces folded '-' in as -3, so
                    # `GETBIT k -9223372036854775808` passed the byte_len bound
                    # and read far outside the bitmap: one command, SIGSEGV.
                    var _gbo = parse_int64_strict(buffer + it_pos, off_len)
                    var bit_offset = Int(_gbo.value)
                    it_pos += off_len + 2
                    if not _gbo.ok or _gbo.value < 0 or _gbo.value >= 4294967296:
                        writer.append_error_response("ERR bit offset is not an integer or out of range")
                        consumed = it_pos
                        continue
                    var gb_val = self.keyspace[].get(key_val)
                    if gb_val.is_none():
                        writer.append_int_response(0)
                    elif gb_val.is_string_like():
                        # gh #232: a plain SET string is a valid bitmap in Redis
                        # (`SET k hello; GETBIT k 0` -> 0, not WRONGTYPE).
                        var _gb_scratch = stack_allocation[24, UInt8]()
                        var byte_len = 0
                        var bitmap_ptr = gb_val.bitmap_view(_gb_scratch, byte_len)
                        if bit_offset // 8 >= byte_len:
                            writer.append_int_response(0)
                        else:
                            writer.append_int_response(Int64(getbit(bitmap_ptr, byte_len, bit_offset)))
                    else:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                elif b0_lower == 115 and cmd_len == 6 and cmd_matches_6(buffer + cmd_start, 115, 101, 116, 98, 105, 116): # SETBIT
                    if num_args != 4 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    it_pos += key_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var off_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        off_len = off_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if off_len < 0 or it_pos + off_len + 2 > n:
                        return consumed
                    # Strict and bounded (see GETBIT): a negative offset used to
                    # pass setbit()'s grow check and WRITE before the bitmap.
                    var _sbo = parse_int64_strict(buffer + it_pos, off_len)
                    var bit_offset = Int(_sbo.value)
                    var _sbo_ok = _sbo.ok and _sbo.value >= 0 and _sbo.value < 4294967296
                    it_pos += off_len + 2
                    if it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var bv_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        bv_len = bv_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if bv_len < 0 or it_pos + bv_len + 2 > n:
                        return consumed
                    # The value must be exactly "0" or "1": only its first byte
                    # was read, so "1abc" and "10" set the bit.
                    var bit_value = Int(buffer[it_pos] - 48) if bv_len == 1 else -1
                    it_pos += bv_len + 2
                    if not _sbo_ok:
                        writer.append_error_response("ERR bit offset is not an integer or out of range")
                    elif bit_value != 0 and bit_value != 1:
                        writer.append_error_response("ERR bit is not an integer or out of range")
                    else:
                        var sb_val = self.keyspace[].get(key_val)
                        if sb_val.is_none():
                            var byte_len = bit_offset // 8 + 1
                            var bitmap_ptr = alloc[UInt8](byte_len)
                            unsafe_memset(bitmap_ptr, 0, byte_len)
                            var old_bit = getbit(bitmap_ptr, byte_len, bit_offset)
                            var result = setbit(byte_len, bitmap_ptr, bit_offset, bit_value)
                            var new_val = GenericValue()
                            new_val.type = ValueType(ValueType.BITMAP)
                            new_val._data0 = UInt64(Int(result.ptr))
                            new_val._data1 = UInt64(result.len)
                            self.keyspace[].set(key_val, new_val)
                            if self.has_wal:   # gh #170
                                var _bb = stack_allocation[64, UInt8]()
                                var _bl = 0
                                var _bp = gv_bytes(key_val, _bb, _bl)
                                _ = self.wal[].append_u64_val(18, _bp, _bl,
                                        (UInt64(bit_offset) << 1) | UInt64(bit_value & 1),
                                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                            writer.append_int_response(Int64(old_bit))
                        elif sb_val.type.value == ValueType.BITMAP:
                            var bitmap_ptr = sb_val.as_bitmap()
                            var byte_len = sb_val.bitmap_len()
                            var old_bit = getbit(bitmap_ptr, byte_len, bit_offset)
                            var result = setbit(byte_len, bitmap_ptr, bit_offset, bit_value)
                            sb_val._data0 = UInt64(Int(result.ptr))
                            sb_val._data1 = UInt64(result.len)
                            self.keyspace[].set(key_val, sb_val)
                            if self.has_wal:   # gh #170
                                var _bb = stack_allocation[64, UInt8]()
                                var _bl = 0
                                var _bp = gv_bytes(key_val, _bb, _bl)
                                _ = self.wal[].append_u64_val(18, _bp, _bl,
                                        (UInt64(bit_offset) << 1) | UInt64(bit_value & 1),
                                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                            writer.append_int_response(Int64(old_bit))
                        elif sb_val.is_string_like():
                            # gh #232: a bitmap IS a string in Redis, so
                            # `SET k "hello"; SETBIT k 10 1` is textbook usage
                            # there and Pion answered WRONGTYPE — the rarer and
                            # more confusing direction (refusing something valid
                            # rather than accepting something invalid).
                            #
                            # The reason it was refused is real: `setbit` frees
                            # and reallocs on growth, and a STRING's payload may
                            # live in the gh #163 blob arena, which the heap
                            # allocator must never free. So COPY out first —
                            # `owned_bitmap_copy` always returns plain heap,
                            # whatever the source shape (SSO, heap, arena) — and
                            # free the ORIGINAL through the arena-safe helper.
                            var _sb_need = bit_offset // 8 + 1
                            var _sb_len = 0
                            var _sb_buf = sb_val.owned_bitmap_copy(_sb_need, _sb_len)
                            var old_bit = getbit(_sb_buf, _sb_len, bit_offset)
                            var result = setbit(_sb_len, _sb_buf, bit_offset, bit_value)
                            # NOTE: do NOT free the original here. `keyspace.set()`
                            # already calls `free_str_payload()` on the value it
                            # overwrites (hash_map.mojo:138/228/374), and that helper
                            # is the arena-safe one. Freeing it here too is a DOUBLE
                            # FREE — it corrupted the tcmalloc free list and the
                            # server SIGSEGV'd on a LATER alloc inside
                            # `owned_bitmap_copy`, several commands downstream of the
                            # actual fault. Caught by the union test below.
                            var new_val = GenericValue()
                            new_val.type = ValueType(ValueType.BITMAP)
                            new_val._data0 = UInt64(Int(result.ptr))
                            new_val._data1 = UInt64(result.len)
                            self.keyspace[].set(key_val, new_val)
                            if self.has_wal:   # gh #170
                                var _bb = stack_allocation[64, UInt8]()
                                var _bl = 0
                                var _bp = gv_bytes(key_val, _bb, _bl)
                                _ = self.wal[].append_u64_val(18, _bp, _bl,
                                        (UInt64(bit_offset) << 1) | UInt64(bit_value & 1),
                                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
                            writer.append_int_response(Int64(old_bit))
                        else:
                            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                elif b0_lower == 98 and cmd_len == 8 and cmd_matches_8(buffer + cmd_start, 98, 105, 116, 99, 111, 117, 110, 116): # BITCOUNT
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    it_pos += key_len + 2
                    var bc_val = self.keyspace[].get(key_val)
                    if bc_val.is_none():
                        writer.append_int_response(0)
                    elif bc_val.is_string_like():
                        # gh #232: `SET k "hello"; BITCOUNT k` -> 21 in Redis.
                        # Textbook usage, and it was answering WRONGTYPE.
                        var _bc_scratch = stack_allocation[24, UInt8]()
                        var byte_len = 0
                        var bitmap_ptr = bc_val.bitmap_view(_bc_scratch, byte_len)
                        writer.append_int_response(Int64(bitcount(bitmap_ptr, byte_len)))
                    else:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                elif b0_lower == 112 and cmd_len == 5 and cmd_matches_5(buffer + cmd_start, 112, 102, 97, 100, 100): # PFADD (gh #225: every byte)
                    # Whole frame present before anything is applied or answered
                    # (see _bulks_end): the element loop below mutates as it goes.
                    if num_args < 2 or _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos = self._skip_args(buffer, it_pos + key_len + 2, n, num_args - 2)
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var pf_key_ptr = buffer + it_pos   # gh #174: key bytes for effect records
                    it_pos += key_len + 2
                    var hll_stored = self.keyspace[].get(key_val)
                    var hll_ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
                    if hll_stored.is_none():
                        hll_ptr = alloc[UInt8](HLL_REGISTERS)
                        unsafe_memset(hll_ptr, 0, HLL_REGISTERS)
                        var new_val = GenericValue()
                        new_val.type = ValueType(ValueType.HLL)
                        new_val.set_ptr(hll_ptr.bitcast[NoneType]())
                        self.keyspace[].set(key_val, new_val)
                    elif hll_stored.type.value == ValueType.HLL:
                        hll_ptr = hll_stored.as_hll()
                    else:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                        # skip remaining args
                        var args_to_skip = num_args - 2
                        for _ in range(args_to_skip):
                            if it_pos >= n or buffer[it_pos] != 36: break
                            it_pos += 1
                            var al = 0
                            while it_pos < n and buffer[it_pos] != 13:
                                al = al * 10 + Int(buffer[it_pos] - 48)
                                it_pos += 1
                            it_pos += 2 + al + 2
                        # `continue` skips the `consumed = it_pos` at the bottom of
                        # the loop, so without this the fast path returned a STALE
                        # consumed, the caller handed the same bytes to the slow
                        # path, and PFADD-on-wrong-type answered TWICE — one
                        # command in, two replies out, and every later reply on
                        # that connection paired with the wrong request. Pipelining
                        # masked it: a following command advanced `consumed` past
                        # the whole buffer, so it only bit when PFADD was last.
                        consumed = it_pos
                        continue
                    var pfadd_updated = False
                    var pfadd_ok = True
                    var elem_count = num_args - 2
                    for _ in range(elem_count):
                        if it_pos >= n or buffer[it_pos] != 36:
                            pfadd_ok = False
                            break
                        it_pos += 1
                        var elem_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            elem_len = elem_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if elem_len < 0 or it_pos + elem_len + 2 > n:
                            pfadd_ok = False
                            break
                        var elem_val = GenericValue.from_ptr(buffer + it_pos, elem_len)
                        var pf_changed = hll_add(hll_ptr, elem_val)
                        if pf_changed:
                            pfadd_updated = True
                        # gh #174: HLL used to be snapshot-only, so every PFADD
                        # since the last SAVE vanished on crash. Log only
                        # elements that actually moved a register — hll_add is
                        # deterministic, so a no-op add contributes nothing to
                        # replay and logging it would just cost WAL bytes.
                        if pf_changed and self.has_wal:
                            _ = self.wal[].append_kv(24, pf_key_ptr, key_len,
                                                     buffer + it_pos, elem_len)
                        it_pos += elem_len + 2
                    if not pfadd_ok:
                        return consumed
                    writer.append_int_response(Int64(1 if pfadd_updated else 0))
                elif b0_lower == 112 and cmd_len == 7 and cmd_matches_7(buffer + cmd_start, 112, 102, 99, 111, 117, 110, 116): # PFCOUNT single key (gh #225: shared 'p','f'+7 with PFDEBUG)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    # Phase 5: MOVED redirect
                    if self.has_cluster and self._moved_if_needed(buffer + it_pos, key_len, writer, fd):
                        it_pos += key_len + 2
                        consumed = it_pos
                        continue
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    it_pos += key_len + 2
                    var pfc_val = self.keyspace[].get(key_val)
                    if pfc_val.is_none():
                        writer.append_int_response(0)
                    elif pfc_val.type.value == ValueType.HLL:
                        var hll_ptr = pfc_val.as_hll()
                        writer.append_int_response(Int64(hll_count(hll_ptr)))
                    else:
                        writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
                elif b0_lower == 101 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 101, 99, 104, 111): # ECHO (gh #225: every byte)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var msg_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        msg_len = msg_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if msg_len < 0 or it_pos + msg_len + 2 > n:
                        return consumed
                    writer.append_bulk_string_response(buffer + it_pos, msg_len)
                    it_pos += msg_len + 2
                elif b0_lower == 116 and cmd_len == 4 and cmd_matches_4(buffer + cmd_start, 116, 121, 112, 101): # TYPE (gh #225: every byte)
                    if num_args != 2 or it_pos >= n or buffer[it_pos] != 36:
                        return consumed
                    it_pos += 1
                    var key_len = 0
                    while it_pos < n and buffer[it_pos] != 13:
                        key_len = key_len * 10 + Int(buffer[it_pos] - 48)
                        it_pos += 1
                    it_pos += 2
                    if key_len < 0 or it_pos + key_len + 2 > n:
                        return consumed
                    var key_val = GenericValue.borrow_buf(buffer + it_pos, key_len)
                    var val = self.keyspace[].get(key_val)
                    var vt = val.type.value
                    var type_resp: String
                    if vt == ValueType.NONE: type_resp = "+none\r\n"
                    elif vt == ValueType.LIST: type_resp = "+list\r\n"
                    elif vt == ValueType.SET: type_resp = "+set\r\n"
                    elif vt == ValueType.ZSET or vt == ValueType.GEO: type_resp = "+zset\r\n"
                    elif vt == ValueType.HASH: type_resp = "+hash\r\n"
                    elif vt == ValueType.STREAM: type_resp = "+stream\r\n"
                    # gh #378: this arm answers TYPE before the slow path's
                    # handle_type ever sees it, and it lacked gh #366's arm
                    elif vt == ValueType.VSET: type_resp = "+vectorset\r\n"
                    else: type_resp = "+string\r\n"
                    writer.append_to_response(type_resp.unsafe_ptr(), type_resp.byte_length())
                    it_pos += key_len + 2
                elif b0_lower == 115 and cmd_len == 6 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 108 and (buffer[cmd_start + 3] | 0x20) == 101 and (buffer[cmd_start + 4] | 0x20) == 99 and (buffer[cmd_start + 5] | 0x20) == 116: # SELECT
                    # Only `SELECT 0` is answered here. Any other index, a
                    # non-integer or a wrong arity goes to the slow path, which
                    # refuses it (this answered +OK to everything).
                    if num_args != 2 or it_pos + 7 > n or buffer[it_pos] != 36 \
                            or buffer[it_pos + 1] != 49 or buffer[it_pos + 2] != 13 \
                            or buffer[it_pos + 3] != 10 or buffer[it_pos + 4] != 48 \
                            or buffer[it_pos + 5] != 13 or buffer[it_pos + 6] != 10:
                        return consumed
                    it_pos += 7
                    writer.append_ok_response()
                elif b0_lower == 99 and cmd_len == 6 and (buffer[cmd_start + 1] | 0x20) == 108 and (buffer[cmd_start + 2] | 0x20) == 105 and (buffer[cmd_start + 3] | 0x20) == 101 and (buffer[cmd_start + 4] | 0x20) == 110 and (buffer[cmd_start + 5] | 0x20) == 116: # CLIENT
                    if _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
                        return consumed    # split frame: the sub-arms skip without bounds
                    if num_args >= 2 and it_pos < n and buffer[it_pos] == 36:
                        it_pos += 1
                        var sub_len = 0
                        while it_pos < n and buffer[it_pos] != 13:
                            sub_len = sub_len * 10 + Int(buffer[it_pos] - 48)
                            it_pos += 1
                        it_pos += 2
                        if sub_len < 0 or it_pos + sub_len + 2 > n:
                            return consumed
                        var sub_at = it_pos
                        it_pos += sub_len + 2
                        # #30: only ID and NO-EVICT / NO-TOUCH here, by their
                        # whole names. GETNAME answered nil and SETNAME +OK
                        # without storing anything; both now run in the slow
                        # path, which keeps the name (as does SETINFO).
                        if cmd_eq(buffer + sub_at, sub_len, "id"):
                            writer.append_int_response(Int64(fd))
                        elif cmd_eq(buffer + sub_at, sub_len, "no-evict") or cmd_eq(buffer + sub_at, sub_len, "no-touch"):
                            if num_args >= 3 and it_pos < n and buffer[it_pos] == 36:
                                it_pos += 1
                                var al = 0
                                while it_pos < n and buffer[it_pos] != 13:
                                    al = al * 10 + Int(buffer[it_pos] - 48)
                                    it_pos += 1
                                it_pos += 2 + al + 2
                            writer.append_ok_response()
                        else:
                            fast_path_ok = False
                            break
                    else:
                        fast_path_ok = False
                        break
                elif b0_lower == 99 and cmd_len == 7 and (buffer[cmd_start + 1] | 0x20) == 111 and (buffer[cmd_start + 2] | 0x20) == 109 and (buffer[cmd_start + 3] | 0x20) == 109 and (buffer[cmd_start + 4] | 0x20) == 97 and (buffer[cmd_start + 5] | 0x20) == 110 and (buffer[cmd_start + 6] | 0x20) == 100: # COMMAND
                    # Whole frame first: with the subcommand still in flight,
                    # `it_pos < n` failed and the arm answered `*0` for it.
                    if _bulks_end(buffer, it_pos, n, num_args - 1) < 0:
                        return consumed
                    if num_args >= 2 and it_pos < n and buffer[it_pos] == 36:
                        # gh #220: this replied :200 to EVERY subcommand — so
                        # `COMMAND DOCS` answered an integer where Redis answers
                        # an array, and the count itself was invented (the real
                        # surface is PION_COMMAND_COUNT, derived from the
                        # dispatch chains). Clients use COMMAND for routing, so
                        # both the shape and the number matter. Peek at the
                        # first subcommand byte instead of ignoring it.
                        #
                        # Note this arm is why the slow path's handle_command is
                        # effectively unreachable for ordinary traffic: COMMAND
                        # never falls through. Keep the two in step.
                        var _cmd_sub0: UInt8 = 0
                        var _cmd_first = True
                        for _ in range(num_args - 1):
                            if it_pos >= n or buffer[it_pos] != 36: break
                            it_pos += 1
                            var al = 0
                            while it_pos < n and buffer[it_pos] != 13:
                                al = al * 10 + Int(buffer[it_pos] - 48)
                                it_pos += 1
                            it_pos += 2
                            if _cmd_first and al > 0 and it_pos < n:
                                _cmd_sub0 = buffer[it_pos] | 0x20
                                _cmd_first = False
                            it_pos += al + 2
                        if _cmd_sub0 == 99:  # COUNT
                            writer.append_int_response(Int64(PION_COMMAND_COUNT))
                        else:
                            # DOCS / INFO / LIST / GETKEYS — array-shaped in
                            # Redis; empty is incomplete but at least the right
                            # type, which an integer was not.
                            writer.append_empty_array_response()
                    else:
                        writer.append_empty_array_response()
                elif b0_lower == 100 and cmd_len == 6 and (buffer[cmd_start + 1] | 0x20) == 98 and (buffer[cmd_start + 2] | 0x20) == 115 and (buffer[cmd_start + 3] | 0x20) == 105 and (buffer[cmd_start + 4] | 0x20) == 122 and (buffer[cmd_start + 5] | 0x20) == 101: # DBSIZE
                    # Step over any arguments before replying: this arm answers a
                    # no-argument command, and `consumed = it_pos` below must cover the
                    # WHOLE frame, or the surplus bytes are re-parsed as a fresh command
                    # (`DBSIZE k` + `PING` answered :N, then -ERR unknown command 'k').
                    # ...and only once the whole frame is here: this skipped
                    # past the end of a split frame and answered anyway.
                    var _fe = _bulks_end(buffer, it_pos, n, num_args - 1)
                    if _fe < 0:
                        return consumed
                    it_pos = _fe
                    var _dbsz: Int64 = 0
                    for _si in range(8): _dbsz += Int64(self.keyspace[].shards[_si].size)
                    writer.append_int_response(_dbsz)
                elif b0_lower == 114 and cmd_len == 5 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 115 and (buffer[cmd_start + 3] | 0x20) == 101 and (buffer[cmd_start + 4] | 0x20) == 116: # RESET
                    # Redis answers RESET with +RESET (the slow path does); this
                    # arm said +OK, so the reply depended on which path ran it.
                    # RESET is rare — hand it over.
                    return consumed
                elif b0_lower == 113 and cmd_len == 4 and (buffer[cmd_start + 1] | 0x20) == 117 and (buffer[cmd_start + 2] | 0x20) == 105 and (buffer[cmd_start + 3] | 0x20) == 116: # QUIT
                    # Step over any arguments before replying: this arm answers a
                    # no-argument command, and `consumed = it_pos` below must cover the
                    # WHOLE frame, or the surplus bytes are re-parsed as a fresh command
                    # (`DBSIZE k` + `PING` answered :N, then -ERR unknown command 'k').
                    # ...and only once the whole frame is here: this skipped
                    # past the end of a split frame and answered anyway.
                    var _fe = _bulks_end(buffer, it_pos, n, num_args - 1)
                    if _fe < 0:
                        return consumed
                    it_pos = _fe
                    writer.append_ok_response()
                elif b0_lower == 114 and cmd_len == 8 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 97 and (buffer[cmd_start + 3] | 0x20) == 100 and (buffer[cmd_start + 4] | 0x20) == 111 and (buffer[cmd_start + 5] | 0x20) == 110 and (buffer[cmd_start + 6] | 0x20) == 108 and (buffer[cmd_start + 7] | 0x20) == 121: # READONLY
                    # Step over any arguments before replying: this arm answers a
                    # no-argument command, and `consumed = it_pos` below must cover the
                    # WHOLE frame, or the surplus bytes are re-parsed as a fresh command
                    # (`DBSIZE k` + `PING` answered :N, then -ERR unknown command 'k').
                    # ...and only once the whole frame is here: this skipped
                    # past the end of a split frame and answered anyway.
                    var _fe = _bulks_end(buffer, it_pos, n, num_args - 1)
                    if _fe < 0:
                        return consumed
                    it_pos = _fe
                    # C1.2: Allow reads on replica — set per-fd flag
                    if Int(fd) < 65536:
                        self.local_affinity[Int(fd)] = 3  # 3 = READONLY mode
                    writer.append_ok_response()
                elif b0_lower == 114 and cmd_len == 9 and (buffer[cmd_start + 1] | 0x20) == 101 and (buffer[cmd_start + 2] | 0x20) == 97 and (buffer[cmd_start + 3] | 0x20) == 100 and (buffer[cmd_start + 4] | 0x20) == 119 and (buffer[cmd_start + 5] | 0x20) == 114 and (buffer[cmd_start + 6] | 0x20) == 105 and (buffer[cmd_start + 7] | 0x20) == 116 and (buffer[cmd_start + 8] | 0x20) == 101: # READWRITE
                    # Step over any arguments before replying: this arm answers a
                    # no-argument command, and `consumed = it_pos` below must cover the
                    # WHOLE frame, or the surplus bytes are re-parsed as a fresh command
                    # (`DBSIZE k` + `PING` answered :N, then -ERR unknown command 'k').
                    # ...and only once the whole frame is here: this skipped
                    # past the end of a split frame and answered anyway.
                    var _fe = _bulks_end(buffer, it_pos, n, num_args - 1)
                    if _fe < 0:
                        return consumed
                    it_pos = _fe
                    # C1.2: Restore default — reject reads on replica
                    if Int(fd) < 65536:
                        self.local_affinity[Int(fd)] = 0
                    writer.append_ok_response()
                elif b0_lower == 97 and cmd_len == 6 and (buffer[cmd_start + 1] | 0x20) == 115 and (buffer[cmd_start + 2] | 0x20) == 107 and (buffer[cmd_start + 3] | 0x20) == 105 and (buffer[cmd_start + 4] | 0x20) == 110 and (buffer[cmd_start + 5] | 0x20) == 103: # ASKING
                    # Step over any arguments before replying: this arm answers a
                    # no-argument command, and `consumed = it_pos` below must cover the
                    # WHOLE frame, or the surplus bytes are re-parsed as a fresh command
                    # (`DBSIZE k` + `PING` answered :N, then -ERR unknown command 'k').
                    # ...and only once the whole frame is here: this skipped
                    # past the end of a split frame and answered anyway.
                    var _fe = _bulks_end(buffer, it_pos, n, num_args - 1)
                    if _fe < 0:
                        return consumed
                    it_pos = _fe
                    # C1: Set per-fd asking flag — next command on IMPORTING slot served locally
                    if Int(fd) < 65536:
                        self.asking_flags[Int(fd)] = 1
                    writer.append_ok_response()
                else:
                    fast_path_ok = False
                    break
                consumed = it_pos
            # gh #147: `fast_path_ok = False` covers two very different exits —
            # a batch ending on a partial frame, and a batch whose tail command
            # isn't fast-path-handled. In BOTH, `consumed` already covers the
            # commands that executed above and their replies are queued in the
            # response buffer. Returning 0 here dropped that work on the floor:
            # the caller saw consumed==0 and re-ran the slow path over the very
            # same bytes, re-executing them (INCR landed twice) and appending a
            # second reply, which desyncs the connection permanently. Return
            # what actually executed and flush it; the caller memmoves past
            # those bytes and re-dispatches only the unhandled remainder.
            if fast_path_ok or consumed > 0:
                writer.flush_response(fd, server, kq)
                return consumed

        return 0
