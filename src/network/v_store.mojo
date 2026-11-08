"""VStoreIndex — token-ID-indexed value store for externalized attention.

Stores quantized V vectors indexed by (session, layer, token_id). No HNSW,
no key storage. The inference engine keeps K on GPU for attention routing
and offloads V to Pion for memory savings.

Flow:
  1. V.CREATE <session_id> <value_dim> [VQUANT turbo4]
  2. V.STOREBATCH <session_id> <layer_id> <start_id> <num_tokens> <values_fp32>
  3. V.FETCH <session_id> <layer_id> <token_id_1> [id_2 ...]
  4. V.INFO [session_id]

Memory per 128K tokens, 32 layers, dim=1024, turbo4:
  128K × 1024 × 0.5625 B × 32 = 2.25 GB per session
"""

from std.memory.alloc import unsafe_alloc
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory import alloc
from std.memory import unsafe_memcpy, unsafe_memset
from std.ffi import external_call
from std.atomic import Atomic, Ordering
from std.collections import Array

from src.vector.kernels import (
    quantize_fp32_to_block_int4,
    dequantize_block_int4_to_fp32,
    quantize_fp32_to_block_int3,
    dequantize_block_int3_to_fp32,
    quantize_fp32_to_block_int2,
    dequantize_block_int2_to_fp32,
    quantize_fp32_to_block_fp8,
    dequantize_block_fp8_to_fp32,
    quantize_fp32_to_bf16_rope_fp8_body,
    dequantize_bf16_rope_fp8_body_to_fp32,
    quantize_fp32_to_mlx4_g32,
    dequantize_mlx4_g32_to_fp32,
    mlx4g32_bytes_per_token,
)

# Quantization format tags. Stable on-disk values — DO NOT renumber existing
# entries; only append new ones.
comptime VFMT_INT8              = UInt8(0)
comptime VFMT_TURBO4            = UInt8(1)
comptime VFMT_FP16              = UInt8(2)
comptime VFMT_TURBO3            = UInt8(3)
comptime VFMT_TURBO2            = UInt8(4)
comptime VFMT_FP8               = UInt8(5)   # A2 (gh #30) — E4M3 per-block-of-32
comptime VFMT_BF16_ROPE_FP8     = UInt8(6)   # A2 — V4 §2.3.4 mixed layout
# gh #148 Phase 1: mlx QuantizedKVCache layout, int4 group-32 affine. Stored
# verbatim so a warm attach is an mx.array view with no repack. g32 (not g64)
# and affine (not symmetric) are both measured requirements — see the kernel.
comptime VFMT_MLX4G32           = UInt8(7)

# 32 held 16 KV.PREFIX namespaces (each is a _pk + _pv pair) — too few for a
# prefix tree shared by several agents, where every divergence is a namespace.
# Per-slot metadata costs ~20 KB per session per worker; buffers are lazy.
comptime MAX_VS_SESSIONS = 256
comptime MAX_VS_LAYERS   = 80
comptime MAX_VS_TOKENS   = 200000  # 200K headroom
# A10 (gh #37): per-session snapshot pool. Speculative-decoding consumers
# (dflash, EAGLE-3, REST) take a snapshot before each verify, write candidate
# K/V, then RESTORE rejected branches. 16 concurrent snapshots per session is
# enough for realistic tree depths without bloating per-session state.
comptime MAX_VS_SNAPSHOTS_PER_SESSION = 16
# Cross-worker directory: each pion-server can have up to MAX_DIR_ENTRIES
# session names known across all workers, regardless of which worker physically
# stores the V buffers. KV.PREFIX.LOOKUP queries this directory so a fresh
# connection to ANY worker can answer "do we have this prefix cached?".
comptime MAX_DIR_ENTRIES = 256   # tracks MAX_VS_SESSIONS: one entry per session name
comptime DIR_SID_INLINE  = 64


def _sort_uint64_inplace(
    a: Pointer[UInt64, MutUntrackedOrigin],
    lo: Int,
    hi: Int,
):
    """gh #71: Lomuto-partition quicksort over a UInt64 array. Used to build
    the sorted block-hash copy in `set_block_hashes` so MEMBERSHIP can do
    binary search instead of an O(N²) linear scan."""
    if lo >= hi:
        return
    # Insertion sort for small ranges (avoids deep recursion + faster on short arrays).
    if hi - lo <= 16:
        for i in range(lo + 1, hi + 1):
            var key = a[unsafe_offset=i]
            var j = i - 1
            while j >= lo and a[unsafe_offset=j] > key:
                a[unsafe_offset=j + 1] = a[unsafe_offset=j]
                j -= 1
            a[unsafe_offset=j + 1] = key
        return
    # Median-of-three pivot to dodge worst-case quadratic on sorted input.
    var mid = (lo + hi) // 2
    if a[unsafe_offset=lo] > a[unsafe_offset=mid]:
        var t = a[unsafe_offset=lo]; a[unsafe_offset=lo] = a[unsafe_offset=mid]; a[unsafe_offset=mid] = t
    if a[unsafe_offset=lo] > a[unsafe_offset=hi]:
        var t = a[unsafe_offset=lo]; a[unsafe_offset=lo] = a[unsafe_offset=hi]; a[unsafe_offset=hi] = t
    if a[unsafe_offset=mid] > a[unsafe_offset=hi]:
        var t = a[unsafe_offset=mid]; a[unsafe_offset=mid] = a[unsafe_offset=hi]; a[unsafe_offset=hi] = t
    # Move the chosen pivot (mid) to hi-1 and partition.
    var tmp = a[unsafe_offset=mid]; a[unsafe_offset=mid] = a[unsafe_offset=hi - 1]; a[unsafe_offset=hi - 1] = tmp
    var pivot = a[unsafe_offset=hi - 1]
    var i = lo
    var j = hi - 1
    while True:
        i += 1
        while a[unsafe_offset=i] < pivot:
            i += 1
        j -= 1
        while a[unsafe_offset=j] > pivot:
            j -= 1
        if i >= j:
            break
        var swap = a[unsafe_offset=i]; a[unsafe_offset=i] = a[unsafe_offset=j]; a[unsafe_offset=j] = swap
    # Restore pivot.
    var swap2 = a[unsafe_offset=i]; a[unsafe_offset=i] = a[unsafe_offset=hi - 1]; a[unsafe_offset=hi - 1] = swap2
    _sort_uint64_inplace(a, lo, i - 1)
    _sort_uint64_inplace(a, i + 1, hi)


struct VStoreDirEntry(Movable, Copyable):
    """One row of the cross-worker session directory.

    All fields are written by the owning worker on KV.PREFIX.REGISTER /
    DROP. Other workers read with ACQUIRE on the parallel `published[]`
    UInt64 array to see whether the entry is visible. Plain reads of
    `sid` etc. are safe after the ACQUIRE because they were written
    before the publish RELEASE."""
    var active: Bool
    var sid_hash: UInt64
    var sid: Array[UInt8, DIR_SID_INLINE]
    var sid_len: Int
    var kv_dim: Int
    var v_format: UInt8
    var owner_worker_id: Int
    var last_ts: UInt64
    # Bundle (gh #29 / §8.2 follow-up): schema digest. For uniform sessions
    # this matches a hash of (kv_dim, v_format, 0, 1) so legacy LOOKUPs can
    # still equate. For heterogeneous sessions it captures the per-layer
    # (dim, fmt, rope_dim) tuple stream so a cross-instance consumer can
    # detect "same prefix sid but DIFFERENT schema". 0 = unset (legacy).
    var schema_digest: UInt32

    def __init__(out self):
        self.active = False
        self.sid_hash = 0
        self.sid = Array[UInt8, DIR_SID_INLINE](fill=UInt8(0))
        self.sid_len = 0
        self.kv_dim = 0
        self.v_format = VFMT_INT8
        self.owner_worker_id = -1
        self.last_ts = 0
        self.schema_digest = 0

    def __copyinit__(out self, existing: Self):
        self.active = existing.active
        self.sid_hash = existing.sid_hash
        self.sid = Array[UInt8, DIR_SID_INLINE](fill=UInt8(0))
        for i in range(DIR_SID_INLINE):
            self.sid[i] = existing.sid[i]
        self.sid_len = existing.sid_len
        self.kv_dim = existing.kv_dim
        self.v_format = existing.v_format
        self.owner_worker_id = existing.owner_worker_id
        self.last_ts = existing.last_ts
        self.schema_digest = existing.schema_digest

    def __moveinit__(out self, owned existing: Self):
        self.active = existing.active
        self.sid_hash = existing.sid_hash
        self.sid = Array[UInt8, DIR_SID_INLINE](fill=UInt8(0))
        for i in range(DIR_SID_INLINE):
            self.sid[i] = existing.sid[i]
        self.sid_len = existing.sid_len
        self.kv_dim = existing.kv_dim
        self.v_format = existing.v_format
        self.owner_worker_id = existing.owner_worker_id
        self.last_ts = existing.last_ts
        self.schema_digest = existing.schema_digest


struct VStoreDirectory(Movable):
    """Cross-worker session directory backing KV.PREFIX.LOOKUP across all
    workers. Allocated once in main() before parallelize; every worker holds
    the same pointer.

    Concurrency: a `published` UInt64 array (one slot per row) is written
    with RELEASE on registration and read with ACQUIRE on lookup, pairing
    the field writes with the visibility flag the same way SharedHNSWView
    handles its own `ready_atomic`."""
    var entries: Pointer[VStoreDirEntry, MutUntrackedOrigin]
    var published: Pointer[UInt64, MutUntrackedOrigin]  # 0 = empty/in-flight, 1 = visible

    def __init__(out self):
        var _e = unsafe_alloc[VStoreDirEntry](MAX_DIR_ENTRIES)
        var ep = Pointer[VStoreDirEntry, MutUntrackedOrigin](unsafe_from_address=Int(_e))
        for i in range(MAX_DIR_ENTRIES):
            ep[unsafe_offset=i] = VStoreDirEntry()
        var _p = unsafe_alloc[UInt64](MAX_DIR_ENTRIES)
        unsafe_memset(_p.unsafe_bitcast[UInt8](), 0, MAX_DIR_ENTRIES * 8)
        var pp = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(_p))
        self.entries = ep
        self.published = pp

    def __moveinit__(out self, owned existing: Self):
        self.entries = existing.entries
        self.published = existing.published


@always_inline
def _dir_hash(sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int) -> UInt64:
    var h: UInt64 = 0
    for i in range(sid_len):
        h = h * UInt64(0x100000001b3) + UInt64(Int(sid_ptr[unsafe_offset=i]))
    return h


def vstore_dir_register(
    dir_ptr: Pointer[VStoreDirectory, MutUntrackedOrigin],
    sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
    sid_len: Int,
    kv_dim: Int,
    v_format: UInt8,
    owner_worker_id: Int,
    ts: UInt64,
    schema_digest: UInt32 = 0,
) -> Int:
    """Insert or refresh a directory entry. Returns slot index, or -1 if full.
    Idempotent on (sid_hash, sid_len, sid bytes) — re-registering the same
    namespace updates owner/ts/schema_digest in place. schema_digest=0 means
    "uniform / unset" (legacy LOOKUP behavior preserved)."""
    if is_null(dir_ptr) or sid_len <= 0 or sid_len > DIR_SID_INLINE:
        return -1
    var h = _dir_hash(sid_ptr, sid_len)
    var entries = dir_ptr[].entries
    var published = dir_ptr[].published

    # First pass: find existing entry to refresh.
    for i in range(MAX_DIR_ENTRIES):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                published.unsafe_offset(i), UInt64(0)) == 0:
            continue
        if entries[unsafe_offset=i].sid_hash != h or entries[unsafe_offset=i].sid_len != sid_len:
            continue
        var same = True
        for j in range(sid_len):
            if entries[unsafe_offset=i].sid[j] != sid_ptr[unsafe_offset=j]:
                same = False
                break
        if same:
            entries[unsafe_offset=i].owner_worker_id = owner_worker_id
            entries[unsafe_offset=i].last_ts = ts
            entries[unsafe_offset=i].kv_dim = kv_dim
            entries[unsafe_offset=i].v_format = v_format
            entries[unsafe_offset=i].schema_digest = schema_digest
            return i

    # Second pass: claim a free slot (published==0).
    for i in range(MAX_DIR_ENTRIES):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                published.unsafe_offset(i), UInt64(0)) != 0:
            continue
        # Write fields under the assumption that published is still 0 — if
        # another worker also picks this slot we'll lose the race; the loser's
        # writes get overwritten by the winner's RELEASE store. With 64 slots
        # and 4 workers this is rare; for a stricter solution use an atomic CAS
        # on `published` before writing.
        entries[unsafe_offset=i].active = True
        entries[unsafe_offset=i].sid_hash = h
        entries[unsafe_offset=i].sid_len = sid_len
        for j in range(sid_len):
            entries[unsafe_offset=i].sid[j] = sid_ptr[unsafe_offset=j]
        entries[unsafe_offset=i].kv_dim = kv_dim
        entries[unsafe_offset=i].v_format = v_format
        entries[unsafe_offset=i].owner_worker_id = owner_worker_id
        entries[unsafe_offset=i].last_ts = ts
        entries[unsafe_offset=i].schema_digest = schema_digest
        Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
            published.unsafe_offset(i), UInt64(1))
        return i

    return -1  # directory full


def vstore_compute_schema_digest(
    num_layers: Int,
    layer_value_dim: Pointer[Int, MutUntrackedOrigin],
    v_fmt: Pointer[UInt8, MutUntrackedOrigin],
    layer_rope_dim: Pointer[Int, MutUntrackedOrigin],
) -> UInt32:
    """FNV-1a hash over the per-layer (dim, fmt, rope_dim) tuple stream.
    Returns 0 only when num_layers == 0 (caller should treat as "unset")."""
    if num_layers <= 0:
        return UInt32(0)
    var h: UInt32 = UInt32(2166136261)
    for li in range(num_layers):
        var d = UInt32(layer_value_dim[unsafe_offset=li])
        var f = UInt32(v_fmt[unsafe_offset=li])
        var r = UInt32(layer_rope_dim[unsafe_offset=li])
        # Mix each field byte-wise.
        for shift in range(4):
            h = (h ^ ((d >> (UInt32(shift) * UInt32(8))) & UInt32(0xFF))) * UInt32(16777619)
        h = (h ^ f) * UInt32(16777619)
        for shift in range(4):
            h = (h ^ ((r >> (UInt32(shift) * UInt32(8))) & UInt32(0xFF))) * UInt32(16777619)
    # Reserve 0 for "unset".
    if h == UInt32(0):
        return UInt32(1)
    return h


def vstore_dir_get_schema_digest(
    dir_ptr: Pointer[VStoreDirectory, MutUntrackedOrigin],
    sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
    sid_len: Int,
) -> UInt32:
    """Return the schema_digest for the entry, or 0 if not found / unset."""
    if is_null(dir_ptr) or sid_len <= 0 or sid_len > DIR_SID_INLINE:
        return UInt32(0)
    var h = _dir_hash(sid_ptr, sid_len)
    var entries = dir_ptr[].entries
    var published = dir_ptr[].published
    for i in range(MAX_DIR_ENTRIES):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                published.unsafe_offset(i), UInt64(0)) == 0:
            continue
        if entries[unsafe_offset=i].sid_hash != h or entries[unsafe_offset=i].sid_len != sid_len:
            continue
        var same = True
        for j in range(sid_len):
            if entries[unsafe_offset=i].sid[j] != sid_ptr[unsafe_offset=j]:
                same = False
                break
        if same:
            return entries[unsafe_offset=i].schema_digest
    return UInt32(0)


def vstore_dir_lookup(
    dir_ptr: Pointer[VStoreDirectory, MutUntrackedOrigin],
    sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
    sid_len: Int,
) -> Int:
    """Return the owner worker_id for the namespace, or -1 if not registered."""
    if is_null(dir_ptr) or sid_len <= 0 or sid_len > DIR_SID_INLINE:
        return -1
    var h = _dir_hash(sid_ptr, sid_len)
    var entries = dir_ptr[].entries
    var published = dir_ptr[].published
    for i in range(MAX_DIR_ENTRIES):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                published.unsafe_offset(i), UInt64(0)) == 0:
            continue
        if entries[unsafe_offset=i].sid_hash != h or entries[unsafe_offset=i].sid_len != sid_len:
            continue
        var same = True
        for j in range(sid_len):
            if entries[unsafe_offset=i].sid[j] != sid_ptr[unsafe_offset=j]:
                same = False
                break
        if same:
            return entries[unsafe_offset=i].owner_worker_id
    return -1


def vstore_dir_drop(
    dir_ptr: Pointer[VStoreDirectory, MutUntrackedOrigin],
    sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
    sid_len: Int,
) -> Bool:
    """Remove the entry for this namespace. Returns True if something was removed."""
    if is_null(dir_ptr) or sid_len <= 0 or sid_len > DIR_SID_INLINE:
        return False
    var h = _dir_hash(sid_ptr, sid_len)
    var entries = dir_ptr[].entries
    var published = dir_ptr[].published
    for i in range(MAX_DIR_ENTRIES):
        if Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.ACQUIRE](
                published.unsafe_offset(i), UInt64(0)) == 0:
            continue
        if entries[unsafe_offset=i].sid_hash != h or entries[unsafe_offset=i].sid_len != sid_len:
            continue
        var same = True
        for j in range(sid_len):
            if entries[unsafe_offset=i].sid[j] != sid_ptr[unsafe_offset=j]:
                same = False
                break
        if same:
            # Clear publish flag first so concurrent lookups don't read mid-clear.
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                published.unsafe_offset(i), UInt64(0))
            entries[unsafe_offset=i].active = False
            entries[unsafe_offset=i].sid_len = 0
            return True
    return False


struct VSSessionMeta(TrivialRegisterPassable):
    """Per-session metadata."""
    var active: Bool
    var session_hash: UInt64
    var session_id_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var session_id_len: Int
    var value_dim: Int
    var v_format: UInt8
    var num_layers: Int             # highest layer stored + 1
    var last_access_ts: UInt64      # monotonic counter — for LRU eviction
    # gh #71: optional block-hash table for within-prefix block visibility.
    # Populated by `KV.PREFIX.REGISTER ... BLOCKS <block_size> <hash_count> <blob>`
    # on the K-side session ("<ns>_pk"); V-side stays unset. block_size == 0 OR
    # block_hashes is null means no table registered → wire reply +UNKNOWN.
    #
    # `block_hashes` preserves token order for the BLOCKS read-back.
    # `block_hashes_sorted` is a parallel sorted copy used by MEMBERSHIP for
    # O(log N) binary search — at 1,562 hashes the linear scan was 6.5 ms p50
    # (issue target ≤ 100 µs). Both arrays have `block_count` entries.
    var block_size: UInt32
    var block_count: UInt32
    var block_hashes: Pointer[UInt64, MutUntrackedOrigin]
    var block_hashes_sorted: Pointer[UInt64, MutUntrackedOrigin]
    # gh #468: VStoreIndex.gen_clock at this session's last change. An export
    # file is named by its sessions' gens, so a changed session never serves
    # an old file.
    var gen: UInt64

    def __init__(out self):
        self.active = False
        self.session_hash = 0
        self.session_id_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.session_id_len = 0
        self.value_dim = 1024
        self.v_format = VFMT_INT8
        self.num_layers = 0
        self.last_access_ts = 0
        self.block_size = UInt32(0)
        self.block_count = UInt32(0)
        self.block_hashes = null_ptr[UInt64, MutUntrackedOrigin]()
        self.block_hashes_sorted = null_ptr[UInt64, MutUntrackedOrigin]()
        self.gen = 0


struct VStoreIndex(Movable):
    """Token-ID-indexed value store for externalized attention V-cache.

    No HNSW. No keys. Just flat quantized V arrays indexed by token ID.
    GPU keeps K for routing; Pion stores V for memory savings.
    """
    var sessions: Pointer[VSSessionMeta, MutUntrackedOrigin]

    # Per-session per-layer token count. tokens_per_layer[session * MAX_VS_LAYERS + layer]
    var tokens_per_layer: Pointer[Int, MutUntrackedOrigin]

    # Per-layer V storage. Slot = session_idx * MAX_VS_LAYERS + layer_id.
    # INT8 path:
    var v_int8: Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin]
    var v_scale: Pointer[Float32, MutUntrackedOrigin]
    var v_min: Pointer[Float32, MutUntrackedOrigin]
    # Turbo/FP16 path (reuse same pointer for turbo4/turbo3/turbo2/fp16):
    var v_turbo: Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin]
    # Per-slot format (resolved at store time):
    var v_fmt: Pointer[UInt8, MutUntrackedOrigin]
    # A1 (gh #29): per-layer value_dim. Populated either by broadcasting the
    # session-level value_dim (legacy `V.CREATE sid dim [VQUANT fmt]`) or by
    # parsing per-layer specs (heterogeneous `V.CREATE sid [default_dim]
    # SCHEMA N spec_0 ... spec_{N-1}`). store_batch / fetch_tokens / save /
    # load / wal_replay all read THIS array, never `meta.value_dim`, so
    # heterogeneous and uniform sessions follow the same code path.
    var layer_value_dim: Pointer[Int, MutUntrackedOrigin]
    # A2 (gh #30): per-layer rope_dim — meaningful ONLY for VFMT_BF16_ROPE_FP8.
    # 0 for every other format. Layout impact: the first rope_dim values of
    # each token are stored as BF16 (2 bytes), the remaining (val_dim - rope_dim)
    # values as block-FP8.
    var layer_rope_dim: Pointer[Int, MutUntrackedOrigin]
    # gh #41 fix: per-slot buffer capacity in TOKENS. store_batch reads this
    # to decide whether to realloc-grow before writing past first-batch end_id.
    # Pre-#41, multi-batch STOREs past the initial capacity silently corrupted
    # the heap (buffer was sized at first-STORE end_id × bpt, never resized).
    # Snapshot/restore tests are the first workload that exercises multi-batch
    # routinely; without this fix they trigger tcmalloc heap corruption.
    var buf_capacity: Pointer[Int, MutUntrackedOrigin]
    # A10 (gh #37 / #41): per-session snapshot pool.
    #
    # Slot indexed by `session_idx * MAX_VS_SNAPSHOTS_PER_SESSION + snap_slot`.
    # snap_id == 0 means slot is free; the counter is monotonic and never reused.
    # snap_lens captures tokens_per_layer at snap time. snap_scales / snap_mins
    # capture v_scale / v_min so INT8 multi-batch sessions can roll back the
    # global rescale that STOREBATCH applies. Block formats (turbo*/fp8/hybrid)
    # embed scale per-block in the value blob and don't read snap_scales on
    # restore — but the capture is unconditional for uniformity (cheap).
    #
    # In-memory only — snapshots are not persisted to WAL or disk.
    #
    # First attempt at this pool (originally co-shipped with the dir-register
    # reorder in load_from_disk) tripped a tcmalloc heap-corruption that
    # surfaced only in cross-test setups. Bisecting under #41 found the pool
    # itself was clean; the corruption traced to other state I couldn't pin
    # down but that no longer manifests on this revision.
    var snap_id: Pointer[UInt64, MutUntrackedOrigin]
    var snap_lens: Pointer[Int, MutUntrackedOrigin]
    var snap_scales: Pointer[Float32, MutUntrackedOrigin]
    var snap_mins: Pointer[Float32, MutUntrackedOrigin]
    var snap_id_counter: UInt64

    var session_count: Int
    var enabled: Bool
    var total_tokens: Int
    var total_fetches: Int
    var access_counter: UInt64      # monotonic LRU clock
    var total_evictions: Int        # diagnostic counter
    # WAL: append-only log of mutating ops (CREATE / STORE / DROP). Replayed on
    # startup after the snapshot load to recover writes that arrived between
    # snapshots. Truncated by KV.PREFIX.SAVE once a fresh snapshot is durable.
    # When wal_fd < 0, WAL is disabled (cold path / replay in progress).
    var wal_fd: Int32
    var wal_path_buf: Pointer[UInt8, MutUntrackedOrigin]
    var wal_path_len: Int
    var wal_appended: Int           # diagnostic: ops appended since open
    var wal_replayed: Int           # diagnostic: ops replayed at startup
    # Cross-worker session directory (shared across workers, owned by main).
    # Used by KV.PREFIX.LOOKUP to answer for sessions registered on any worker
    # — without this, a fresh connection that race-accepts onto a non-owner
    # worker reports MISS even though the prefix is cached. NULL = no directory
    # (single-worker deployments).
    var directory: Pointer[VStoreDirectory, MutUntrackedOrigin]
    var my_worker_id: Int
    # Multi-tenant: when non-empty, every KV.PREFIX.* namespace must start
    # with this exact byte sequence. Set from config.server.ns_prefix.
    var ns_prefix: String
    # gh #262: the client's MEASURED cold prefill time per session
    # (`KV.PREFIX.REGISTER ... PREFILL_MS`), microseconds; 0 = not reported.
    # Indexed by session slot like `sessions`. Deliberately NOT persisted in
    # the V-store WAL/snapshot: a restart forgets the measurement and the
    # ledger falls back to its per-token estimate, which it labels as such.
    var prefix_prefill_us: Pointer[UInt64, MutUntrackedOrigin]
    # gh #468: export lane state. gen_clock only ever grows within a process;
    # export_nonce (pid ^ start time) separates this process's export files
    # from a previous one's, which the first export of a process deletes.
    var gen_clock: UInt64
    var export_nonce: UInt64
    var export_pruned: Bool

    def __init__(out self, enabled: Bool = False):
        self.enabled = enabled
        self.session_count = 0
        self.total_tokens = 0
        self.total_fetches = 0
        self.access_counter = 0
        self.total_evictions = 0
        self.wal_fd = Int32(-1)
        self.wal_path_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.wal_path_len = 0
        self.wal_appended = 0
        self.wal_replayed = 0
        self.directory = null_ptr[VStoreDirectory, MutUntrackedOrigin]()
        self.my_worker_id = -1
        self.ns_prefix = String("")
        var _ppu = unsafe_alloc[UInt64](MAX_VS_SESSIONS)
        self.prefix_prefill_us = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(_ppu))
        self.gen_clock = 0
        self.export_nonce = 0
        self.export_pruned = False
        for i in range(MAX_VS_SESSIONS):
            self.prefix_prefill_us[unsafe_offset=i] = UInt64(0)

        var num_slots = MAX_VS_SESSIONS * MAX_VS_LAYERS

        var _s = unsafe_alloc[VSSessionMeta](MAX_VS_SESSIONS)
        self.sessions = Pointer[VSSessionMeta, MutUntrackedOrigin](unsafe_from_address=Int(_s))
        for i in range(MAX_VS_SESSIONS):
            self.sessions[unsafe_offset=i] = VSSessionMeta()

        var _tpl = unsafe_alloc[Int](num_slots)
        self.tokens_per_layer = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_tpl))
        for i in range(num_slots):
            self.tokens_per_layer[unsafe_offset=i] = 0

        var _i8 = unsafe_alloc[Pointer[Int8, MutUntrackedOrigin]](num_slots)
        self.v_int8 = Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(_i8))
        for i in range(num_slots):
            self.v_int8[unsafe_offset=i] = null_ptr[Int8, MutUntrackedOrigin]()

        var _sc = unsafe_alloc[Float32](num_slots)
        self.v_scale = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_sc))
        var _mn = unsafe_alloc[Float32](num_slots)
        self.v_min = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_mn))
        for i in range(num_slots):
            self.v_scale[unsafe_offset=i] = Float32(1.0)
            self.v_min[unsafe_offset=i] = Float32(0.0)

        var _tb = unsafe_alloc[Pointer[Int8, MutUntrackedOrigin]](num_slots)
        self.v_turbo = Pointer[Pointer[Int8, MutUntrackedOrigin], MutUntrackedOrigin](unsafe_from_address=Int(_tb))
        for i in range(num_slots):
            self.v_turbo[unsafe_offset=i] = null_ptr[Int8, MutUntrackedOrigin]()

        var _vf = unsafe_alloc[UInt8](num_slots)
        self.v_fmt = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_vf))
        for i in range(num_slots):
            self.v_fmt[unsafe_offset=i] = VFMT_INT8

        # A1: per-layer value_dim. Initialised to 0 (= unset). create_session
        # populates with the session value_dim (broadcast); the SCHEMA path
        # writes per-layer.
        var _ld = unsafe_alloc[Int](num_slots)
        self.layer_value_dim = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_ld))
        for i in range(num_slots):
            self.layer_value_dim[unsafe_offset=i] = 0

        # A2: per-layer rope_dim. 0 unless VFMT_BF16_ROPE_FP8.
        var _rd = unsafe_alloc[Int](num_slots)
        self.layer_rope_dim = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_rd))
        for i in range(num_slots):
            self.layer_rope_dim[unsafe_offset=i] = 0

        # #41 fix: per-slot buffer capacity (in tokens). 0 = no buffer yet.
        var _bc = unsafe_alloc[Int](num_slots)
        self.buf_capacity = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_bc))
        for i in range(num_slots):
            self.buf_capacity[unsafe_offset=i] = 0

        # A10 BISECT: snap_id (4 KB) + snap_lens (320 KB).
        var snap_count = MAX_VS_SESSIONS * MAX_VS_SNAPSHOTS_PER_SESSION
        var _si = unsafe_alloc[UInt64](snap_count)
        unsafe_memset(_si.unsafe_bitcast[UInt8](), 0, snap_count * 8)
        self.snap_id = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(_si))
        var snap_lens_count = snap_count * MAX_VS_LAYERS
        var _sl = unsafe_alloc[Int](snap_lens_count)
        unsafe_memset(_sl.unsafe_bitcast[UInt8](), 0, snap_lens_count * 8)
        self.snap_lens = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(_sl))
        # BISECT: snap_scales + snap_mins (4-byte Float32 each — half the size
        # of snap_lens).
        var _ss = unsafe_alloc[Float32](snap_lens_count)
        unsafe_memset(_ss.unsafe_bitcast[UInt8](), 0, snap_lens_count * 4)
        self.snap_scales = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_ss))
        var _sm = unsafe_alloc[Float32](snap_lens_count)
        unsafe_memset(_sm.unsafe_bitcast[UInt8](), 0, snap_lens_count * 4)
        self.snap_mins = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(_sm))
        self.snap_id_counter = 0


    def __moveinit__(out self, owned existing: Self):
        self.sessions = existing.sessions
        self.tokens_per_layer = existing.tokens_per_layer
        self.v_int8 = existing.v_int8
        self.v_scale = existing.v_scale
        self.v_min = existing.v_min
        self.v_turbo = existing.v_turbo
        self.v_fmt = existing.v_fmt
        self.layer_value_dim = existing.layer_value_dim
        self.layer_rope_dim = existing.layer_rope_dim
        self.buf_capacity = existing.buf_capacity
        self.snap_id = existing.snap_id
        self.snap_lens = existing.snap_lens
        self.snap_scales = existing.snap_scales
        self.snap_mins = existing.snap_mins
        self.snap_id_counter = existing.snap_id_counter
        self.session_count = existing.session_count
        self.enabled = existing.enabled
        self.total_tokens = existing.total_tokens
        self.total_fetches = existing.total_fetches
        self.access_counter = existing.access_counter
        self.total_evictions = existing.total_evictions
        self.wal_fd = existing.wal_fd
        self.wal_path_buf = existing.wal_path_buf
        self.wal_path_len = existing.wal_path_len
        self.wal_appended = existing.wal_appended
        self.wal_replayed = existing.wal_replayed
        self.directory = existing.directory
        self.my_worker_id = existing.my_worker_id
        self.ns_prefix = existing.ns_prefix
        self.prefix_prefill_us = existing.prefix_prefill_us
        self.gen_clock = existing.gen_clock
        self.export_nonce = existing.export_nonce
        self.export_pruned = existing.export_pruned

    # ── Session management ──────────────────────────────────────────────

    @always_inline
    def _bump_gen(mut self, idx: Int):
        """gh #468: mark session `idx` changed (see VSSessionMeta.gen)."""
        if idx >= 0 and idx < MAX_VS_SESSIONS:
            self.gen_clock += 1
            self.sessions[unsafe_offset=idx].gen = self.gen_clock

    def _find_session(self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int) -> Int:
        """Find session by ID. Returns index or -1."""
        # Simple wyhash on session ID
        var h: UInt64 = 0
        for i in range(sid_len):
            h = h * UInt64(0x100000001b3) + UInt64(Int(sid_ptr[unsafe_offset=i]))
        for i in range(MAX_VS_SESSIONS):
            if self.sessions[unsafe_offset=i].active and self.sessions[unsafe_offset=i].session_hash == h:
                if self.sessions[unsafe_offset=i].session_id_len == sid_len:
                    var found = True
                    for j in range(sid_len):
                        if self.sessions[unsafe_offset=i].session_id_ptr[unsafe_offset=j] != sid_ptr[unsafe_offset=j]:
                            found = False
                            break
                    if found:
                        return i
        return -1

    @always_inline
    def _touch(mut self, slot: Int):
        """Bump LRU timestamp on the given slot. Caller must have validated slot."""
        self.access_counter += 1
        self.sessions[unsafe_offset=slot].last_access_ts = self.access_counter

    def find_and_touch(mut self, sid_ptr: Pointer[UInt8, MutUntrackedOrigin], sid_len: Int) -> Int:
        """Find session and bump LRU timestamp. Use on cache-access calls
        (LOOKUP / STOREBATCH / FETCH). Read-only stats paths use _find_session."""
        var idx = self._find_session(sid_ptr, sid_len)
        if idx >= 0:
            self._touch(idx)
        return idx

    def drop_session(mut self, slot: Int) -> Bool:
        """Free all per-layer buffers and the session ID for this slot.
        Returns True if a session was dropped, False if slot was invalid/inactive.

        Used by LRU eviction in create_session() when MAX_VS_SESSIONS is full.
        """
        if slot < 0 or slot >= MAX_VS_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=slot].active:
            return False

        # Account for tokens being freed
        var freed_tokens = 0
        for li in range(MAX_VS_LAYERS):
            var li_slot = slot * MAX_VS_LAYERS + li
            freed_tokens += self.tokens_per_layer[unsafe_offset=li_slot]
            if is_not_null(self.v_int8[unsafe_offset=li_slot]):
                self.v_int8[unsafe_offset=li_slot].unsafe_free()
                self.v_int8[unsafe_offset=li_slot] = null_ptr[Int8, MutUntrackedOrigin]()
            if is_not_null(self.v_turbo[unsafe_offset=li_slot]):
                self.v_turbo[unsafe_offset=li_slot].unsafe_free()
                self.v_turbo[unsafe_offset=li_slot] = null_ptr[Int8, MutUntrackedOrigin]()
            self.tokens_per_layer[unsafe_offset=li_slot] = 0
            self.v_scale[unsafe_offset=li_slot] = Float32(1.0)
            self.v_min[unsafe_offset=li_slot] = Float32(0.0)
            self.v_fmt[unsafe_offset=li_slot] = VFMT_INT8
            self.layer_value_dim[unsafe_offset=li_slot] = 0
            self.layer_rope_dim[unsafe_offset=li_slot] = 0
            self.buf_capacity[unsafe_offset=li_slot] = 0
        # A10: invalidate any snapshots taken against this slot.
        for sn in range(MAX_VS_SNAPSHOTS_PER_SESSION):
            self.snap_id[unsafe_offset=slot * MAX_VS_SNAPSHOTS_PER_SESSION + sn] = UInt64(0)

        # Unpublish from the cross-worker directory BEFORE freeing the id.
        # Every drop path must do this — LRU eviction (create_session under
        # slot pressure) reached here without it, leaving a stale entry that
        # kept KV.PREFIX.LOOKUP answering +HIT for a session whose V buffers
        # were gone (Gate 2c / test_kv_prefix_lru, 2026-08-01).
        if is_not_null(self.directory) and is_not_null(self.sessions[unsafe_offset=slot].session_id_ptr):
            _ = vstore_dir_drop(self.directory,
                                self.sessions[unsafe_offset=slot].session_id_ptr,
                                self.sessions[unsafe_offset=slot].session_id_len)

        if is_not_null(self.sessions[unsafe_offset=slot].session_id_ptr):
            self.sessions[unsafe_offset=slot].session_id_ptr.unsafe_free()

        # gh #71: free optional block-hash table before clearing the meta slot.
        if is_not_null(self.sessions[unsafe_offset=slot].block_hashes):
            self.sessions[unsafe_offset=slot].block_hashes.unsafe_free()
        if is_not_null(self.sessions[unsafe_offset=slot].block_hashes_sorted):
            self.sessions[unsafe_offset=slot].block_hashes_sorted.unsafe_free()

        self.sessions[unsafe_offset=slot] = VSSessionMeta()
        self.session_count -= 1
        self.total_tokens -= freed_tokens
        self.total_evictions += 1
        return True

    # ── gh #71: block-hash table ────────────────────────────────────────
    #
    # Block hashes are content-identifiers of fixed-size token blocks within
    # a prefix (typically 16 or 64 tokens, content-hashed by the consumer).
    # Stored ONLY on the K-side session ("<ns>_pk"); the V-side stays unset.
    # set_block_hashes() overwrites any prior table (frees the old one).
    #
    # We store two parallel arrays:
    #   block_hashes         — token-order copy, returned verbatim by BLOCKS
    #   block_hashes_sorted  — sorted copy, binary-searched by MEMBERSHIP
    # The sort runs once per REGISTER (or per WAL/snapshot reload); MEMBERSHIP
    # then does O(K log N) lookups (Modular blog's "binary search over
    # cumulative block hashes" — linear-scan p50 was 6.5 ms at N=K=1562).
    def set_block_hashes(
        mut self,
        slot: Int,
        block_size: UInt32,
        block_count: UInt32,
        src: Pointer[UInt8, MutUntrackedOrigin],
    ) -> Bool:
        if slot < 0 or slot >= MAX_VS_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=slot].active:
            return False
        # Free any previously registered tables.
        if is_not_null(self.sessions[unsafe_offset=slot].block_hashes):
            self.sessions[unsafe_offset=slot].block_hashes.unsafe_free()
            self.sessions[unsafe_offset=slot].block_hashes = null_ptr[UInt64, MutUntrackedOrigin]()
        if is_not_null(self.sessions[unsafe_offset=slot].block_hashes_sorted):
            self.sessions[unsafe_offset=slot].block_hashes_sorted.unsafe_free()
            self.sessions[unsafe_offset=slot].block_hashes_sorted = null_ptr[UInt64, MutUntrackedOrigin]()
        self.sessions[unsafe_offset=slot].block_size = block_size
        self.sessions[unsafe_offset=slot].block_count = block_count
        if Int(block_count) == 0:
            return True
        var nbytes = Int(block_count) * 8
        # Token-order copy (returned by KV.PREFIX.BLOCKS).
        var _h = unsafe_alloc[UInt64](Int(block_count))
        var hp = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(_h))
        unsafe_memcpy(dest=hp.unsafe_bitcast[UInt8](), src=src, count=nbytes)
        self.sessions[unsafe_offset=slot].block_hashes = hp
        # Sorted copy (binary-searched by KV.PREFIX.MEMBERSHIP).
        var _hs = unsafe_alloc[UInt64](Int(block_count))
        var hsp = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(_hs))
        unsafe_memcpy(dest=hsp.unsafe_bitcast[UInt8](), src=src, count=nbytes)
        _sort_uint64_inplace(hsp, 0, Int(block_count) - 1)
        self.sessions[unsafe_offset=slot].block_hashes_sorted = hsp
        return True

    # ── A10 (gh #37): snapshot / restore / commit ──────────────────────────
    #
    # V-store layers are append-only — `tokens_per_layer[slot]` is monotonic
    # within a STOREBATCH window. A snapshot just records the current per-layer
    # length; restore truncates `tokens_per_layer[]` back to that length.
    # Buffers stay allocated (V-store sizes them at first STORE to end_id × bpt);
    # subsequent writes overwrite the trailing portion. No alloc churn, no leaks.
    #
    # snap_id is a monotonic counter — never reuses a value, so a stale snap_id
    # from a freed slot can't be confused with a live snapshot. snap_id == 0 is
    # reserved for "free slot".

    # A10 BISECT: re-adding method bodies that touch snap_id + snap_lens.

    def snapshot_session(mut self, session_idx: Int) -> UInt64:
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return UInt64(0)
        if not self.sessions[unsafe_offset=session_idx].active:
            return UInt64(0)
        var snap_base = session_idx * MAX_VS_SNAPSHOTS_PER_SESSION
        var free_slot = -1
        for sn in range(MAX_VS_SNAPSHOTS_PER_SESSION):
            if self.snap_id[unsafe_offset=snap_base + sn] == UInt64(0):
                free_slot = sn
                break
        if free_slot < 0:
            return UInt64(0)
        self.snap_id_counter += 1
        var sid_value = self.snap_id_counter
        self.snap_id[unsafe_offset=snap_base + free_slot] = sid_value
        var lens_base = (snap_base + free_slot) * MAX_VS_LAYERS
        var tpl_base = session_idx * MAX_VS_LAYERS
        for li in range(MAX_VS_LAYERS):
            self.snap_lens[unsafe_offset=lens_base + li] = self.tokens_per_layer[unsafe_offset=tpl_base + li]
            # BISECT step: write to snap_scales / snap_mins.
            self.snap_scales[unsafe_offset=lens_base + li] = self.v_scale[unsafe_offset=tpl_base + li]
            self.snap_mins[unsafe_offset=lens_base + li] = self.v_min[unsafe_offset=tpl_base + li]
        return sid_value

    def restore_session(mut self, session_idx: Int, snap_id_v: UInt64) -> Bool:
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=session_idx].active or snap_id_v == UInt64(0):
            return False
        var snap_base = session_idx * MAX_VS_SNAPSHOTS_PER_SESSION
        for sn in range(MAX_VS_SNAPSHOTS_PER_SESSION):
            if self.snap_id[unsafe_offset=snap_base + sn] == snap_id_v:
                var lens_base = (snap_base + sn) * MAX_VS_LAYERS
                var tpl_base = session_idx * MAX_VS_LAYERS
                var diff = 0
                for li in range(MAX_VS_LAYERS):
                    var cur = self.tokens_per_layer[unsafe_offset=tpl_base + li]
                    var snap_len = self.snap_lens[unsafe_offset=lens_base + li]
                    if cur > snap_len:
                        diff += (cur - snap_len)
                        self.tokens_per_layer[unsafe_offset=tpl_base + li] = snap_len
                        # BISECT step: restore scale + min for INT8 dequant
                        # correctness across multi-batch sessions.
                        self.v_scale[unsafe_offset=tpl_base + li] = self.snap_scales[unsafe_offset=lens_base + li]
                        self.v_min[unsafe_offset=tpl_base + li] = self.snap_mins[unsafe_offset=lens_base + li]
                self.total_tokens -= diff
                self._bump_gen(session_idx)
                return True
        return False

    def commit_session_snapshot(mut self, session_idx: Int, snap_id_v: UInt64) -> Bool:
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=session_idx].active or snap_id_v == UInt64(0):
            return False
        var snap_base = session_idx * MAX_VS_SNAPSHOTS_PER_SESSION
        for sn in range(MAX_VS_SNAPSHOTS_PER_SESSION):
            if self.snap_id[unsafe_offset=snap_base + sn] == snap_id_v:
                self.snap_id[unsafe_offset=snap_base + sn] = UInt64(0)
                return True
        return False

    def _pick_lru_slot(self) -> Int:
        """Return the active slot with the smallest last_access_ts. -1 if none."""
        var best_slot = -1
        var best_ts: UInt64 = UInt64(0)
        var first = True
        for i in range(MAX_VS_SESSIONS):
            if self.sessions[unsafe_offset=i].active:
                var ts = self.sessions[unsafe_offset=i].last_access_ts
                if first or ts < best_ts:
                    best_ts = ts
                    best_slot = i
                    first = False
        return best_slot

    def create_session(
        mut self,
        sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
        sid_len: Int,
        value_dim: Int,
        v_format: UInt8 = VFMT_INT8,
    ) -> Int:
        """Create a new V-store session. Returns session index or -1."""
        if not self.enabled:
            return -1

        # Check for existing session
        var existing = self._find_session(sid_ptr, sid_len)
        if existing >= 0:
            self._touch(existing)
            return existing

        # Find free slot
        var slot = -1
        for i in range(MAX_VS_SESSIONS):
            if not self.sessions[unsafe_offset=i].active:
                slot = i
                break
        if slot < 0:
            # Evict the LRU active session and reuse its slot.
            var victim = self._pick_lru_slot()
            if victim < 0:
                return -1  # nothing active — should never happen
            # Log the eviction: without a DROP record the WAL still holds the
            # victim's CREATE and rows, and a restart replays them back. (No-op
            # during replay itself, where wal_fd < 0.)
            self.wal_append_drop(self.sessions[unsafe_offset=victim].session_id_ptr,
                                 self.sessions[unsafe_offset=victim].session_id_len)
            _ = self.drop_session(victim)
            slot = victim

        # Validate value_dim for the block formats (all quantize in groups of 32).
        var actual_fmt = v_format
        if (actual_fmt == VFMT_TURBO4 or actual_fmt == VFMT_TURBO3
            or actual_fmt == VFMT_TURBO2 or actual_fmt == VFMT_FP8
            or actual_fmt == VFMT_MLX4G32) and (value_dim % 32 != 0):
            actual_fmt = VFMT_INT8  # fallback

        # Copy session ID
        var _sid = unsafe_alloc[UInt8](sid_len)
        var sid_copy = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_sid))
        unsafe_memcpy(dest=sid_copy, src=sid_ptr, count=sid_len)

        var h: UInt64 = 0
        for i in range(sid_len):
            h = h * UInt64(0x100000001b3) + UInt64(Int(sid_ptr[unsafe_offset=i]))

        self.sessions[unsafe_offset=slot].active = True
        self.sessions[unsafe_offset=slot].session_hash = h
        self.sessions[unsafe_offset=slot].session_id_ptr = sid_copy
        self.sessions[unsafe_offset=slot].session_id_len = sid_len
        self.sessions[unsafe_offset=slot].value_dim = value_dim
        self.sessions[unsafe_offset=slot].v_format = actual_fmt
        self.sessions[unsafe_offset=slot].num_layers = 0
        for i in range(MAX_VS_LAYERS):
            var li_slot = slot * MAX_VS_LAYERS + i
            self.tokens_per_layer[unsafe_offset=li_slot] = 0
            # A1: broadcast the uniform session value_dim across all layer
            # slots so store_batch / fetch_tokens can read layer_value_dim
            # unconditionally.
            self.layer_value_dim[unsafe_offset=li_slot] = value_dim
            # gh #148: broadcast the session's format too. This used to be left
            # at the VFMT_INT8 that __init__ wrote, on the assumption that
            # "store_batch will overwrite per layer at first STORE" — but
            # store_batch *reads* v_fmt[slot] and only consults the session
            # format when layer_value_dim is unset, which it never is after the
            # line above. Net effect: `V.CREATE ... VQUANT turbo4` (and
            # `KV.PREFIX.REGISTER <ns> <dim> <quant>`) silently stored INT8 —
            # the requested quantization never engaged on the uniform path.
            # Verified before the fix: V.INFO reported layer_0_fmt:int8 for
            # turbo4, fp16 and mlx4g32 alike. The SCHEMA path was unaffected,
            # since it writes v_fmt per layer itself.
            self.v_fmt[unsafe_offset=li_slot] = actual_fmt
        self.session_count += 1
        self._touch(slot)
        self._bump_gen(slot)

        return slot

    def create_session_schema(
        mut self,
        sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
        sid_len: Int,
        default_value_dim: Int,
        num_layers: Int,
        per_layer_dim: Pointer[Int, MutUntrackedOrigin],
        per_layer_fmt: Pointer[UInt8, MutUntrackedOrigin],
        per_layer_rope: Pointer[Int, MutUntrackedOrigin] = null_ptr[Int, MutUntrackedOrigin](),
    ) -> Int:
        """A1 (gh #29): Create a session with a per-layer schema.

        `default_value_dim` becomes the session-level fallback (shown in
        V.INFO) and is broadcast to layer indices >= num_layers so that
        out-of-schema STORE calls behave consistently. Per-layer dim/fmt
        override the default for indices in [0, num_layers).
        """
        if not self.enabled:
            return -1
        if num_layers <= 0 or num_layers > MAX_VS_LAYERS:
            return -1
        # Reuse the legacy create_session to allocate the slot, copy the sid,
        # and broadcast the default. We then overwrite layer_value_dim / v_fmt
        # for indices [0, num_layers).
        var slot = self.create_session(sid_ptr, sid_len, default_value_dim, VFMT_INT8)
        if slot < 0:
            return slot
        for li in range(num_layers):
            var li_slot = slot * MAX_VS_LAYERS + li
            var ld = per_layer_dim[unsafe_offset=li]
            if ld <= 0:
                ld = default_value_dim
            var lf = per_layer_fmt[unsafe_offset=li]
            # Re-apply turbo / FP8 dim divisibility constraint per layer.
            if (lf == VFMT_TURBO4 or lf == VFMT_TURBO3 or lf == VFMT_TURBO2 or lf == VFMT_FP8) and (ld % 32 != 0):
                lf = VFMT_INT8
            self.layer_value_dim[unsafe_offset=li_slot] = ld
            self.v_fmt[unsafe_offset=li_slot] = lf
            # A2: hybrid format stores per-layer rope_dim. For all other
            # formats the rope_dim slot stays 0.
            if lf == VFMT_BF16_ROPE_FP8 and is_not_null(per_layer_rope):
                var rd = per_layer_rope[unsafe_offset=li]
                if rd <= 0 or rd >= ld or (ld - rd) % 32 != 0:
                    # Bad rope_dim → fall back to plain INT8 to avoid silent
                    # mis-quantization. Caller gets a working session but
                    # not the format they asked for.
                    self.v_fmt[unsafe_offset=li_slot] = VFMT_INT8
                else:
                    self.layer_rope_dim[unsafe_offset=li_slot] = rd
        # Set num_layers to the declared schema length so V.INFO reports it
        # immediately (legacy path leaves this at 0 until first STORE bumps it).
        self.sessions[unsafe_offset=slot].num_layers = num_layers
        # A2: set session-level v_format to layer 0's fmt — gives V.INFO a
        # sensible "representative" format string for single-layer SCHEMA
        # sessions (which would otherwise look uniform-INT8 because
        # create_session() was called with VFMT_INT8 internally).
        if num_layers > 0:
            self.sessions[unsafe_offset=slot].v_format = self.v_fmt[unsafe_offset=slot * MAX_VS_LAYERS]
        return slot

    # ── Store ───────────────────────────────────────────────────────────

    @always_inline
    def _bytes_per_token(self, vf: UInt8, val_dim: Int) -> Int:
        """Compute bytes per token for a given V format. Hybrid format uses
        _bytes_per_token_with_rope below — never call this with VFMT_BF16_ROPE_FP8."""
        if vf == VFMT_TURBO4:
            return 4 + (val_dim // 32) * 18
        elif vf == VFMT_TURBO3:
            return 4 + (val_dim // 32) * 14
        elif vf == VFMT_TURBO2:
            return 4 + (val_dim // 32) * 10
        elif vf == VFMT_FP16:
            return val_dim * 2
        elif vf == VFMT_FP8:
            # A2: [FP32 norm 4B] + (dim/32) × (2B scale + 32B E4M3) = 4 + (dim/32) × 34
            return 4 + (val_dim // 32) * 34
        elif vf == VFMT_MLX4G32:
            return mlx4g32_bytes_per_token(val_dim)
        else:  # INT8
            return val_dim

    @always_inline
    def _bytes_per_token_hybrid(self, val_dim: Int, rope_dim: Int) -> Int:
        """A2: bytes per token for VFMT_BF16_ROPE_FP8.
        rope_dim × 2 (BF16) + ((val_dim - rope_dim) / 32) × 34 (block-FP8)."""
        var body = val_dim - rope_dim
        return rope_dim * 2 + (body // 32) * 34

    @always_inline
    def _bytes_per_token_for_slot(self, slot: Int) -> Int:
        """Resolve bytes-per-token for a per-slot fmt+dim+rope_dim. Used by
        store/fetch/save/load to keep the hybrid-vs-uniform branch in one place."""
        var vf = self.v_fmt[unsafe_offset=slot]
        var ld = self.layer_value_dim[unsafe_offset=slot]
        if vf == VFMT_BF16_ROPE_FP8:
            return self._bytes_per_token_hybrid(ld, self.layer_rope_dim[unsafe_offset=slot])
        return self._bytes_per_token(vf, ld)

    def store_batch(
        mut self,
        session_idx: Int,
        layer_id: Int,
        start_id: Int,
        num_tokens: Int,
        values_fp32: Pointer[Float32, MutUntrackedOrigin],
    ) -> Bool:
        """Store a batch of tokens' V, quantizing according to session format.

        values_fp32: [num_tokens * value_dim] contiguous FP32 values.
        Tokens are stored at positions [start_id, start_id + num_tokens).
        """
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return False
        if not self.sessions[unsafe_offset=session_idx].active:
            return False
        if layer_id < 0 or layer_id >= MAX_VS_LAYERS:
            return False

        var slot = session_idx * MAX_VS_LAYERS + layer_id
        # A1: per-layer dim/fmt are the source of truth. Legacy uniform
        # sessions populate every layer slot with the session default, so this
        # branch reads identically for both paths.
        var val_dim = self.layer_value_dim[unsafe_offset=slot]
        var vf = self.v_fmt[unsafe_offset=slot]
        if val_dim <= 0:
            # Defensive: should not happen — create_session and create_session_schema
            # both populate every layer slot. Fall back to the session-level value.
            val_dim = self.sessions[unsafe_offset=session_idx].value_dim
            vf = self.sessions[unsafe_offset=session_idx].v_format
        var end_id = start_id + num_tokens

        if end_id > MAX_VS_TOKENS:
            return False

        # A2: refuse formats we don't have kernels for. Today everything up to
        # VFMT_MLX4G32 (= 7) is implemented; any newer tag is a stale snapshot
        # or a parser bug — refuse rather than silently misquantize.
        if vf > VFMT_MLX4G32:
            return False

        # bytes-per-token: hybrid format also depends on per-layer rope_dim.
        var bpt: Int
        if vf == VFMT_BF16_ROPE_FP8:
            bpt = self._bytes_per_token_hybrid(val_dim, self.layer_rope_dim[unsafe_offset=slot])
        else:
            bpt = self._bytes_per_token(vf, val_dim)

        # Ensure buffer is allocated for the maximum token ID we'll write.
        # FP8 and the hybrid format use the v_turbo slot (same allocation
        # convention as the other non-INT8 paths).
        if vf == VFMT_TURBO4 or vf == VFMT_TURBO3 or vf == VFMT_TURBO2 or vf == VFMT_FP16 or vf == VFMT_FP8 or vf == VFMT_BF16_ROPE_FP8 or vf == VFMT_MLX4G32:
            # Turbo/FP16 path. #41 fix: realloc-grow when end_id exceeds capacity.
            if is_null(self.v_turbo[unsafe_offset=slot]):
                # First batch — round capacity up to a power-of-2 multiple of
                # end_id to amortize the next few extends. Min cap = end_id.
                var cap = end_id
                if cap < 32:
                    cap = 32
                var buf_size = cap * bpt
                var _buf = unsafe_alloc[Int8](buf_size)
                self.v_turbo[unsafe_offset=slot] = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_buf))
                unsafe_memset(self.v_turbo[unsafe_offset=slot], 0, buf_size)
                self.buf_capacity[unsafe_offset=slot] = cap
            elif end_id > self.buf_capacity[unsafe_offset=slot]:
                # Grow: 2× current or end_id, whichever is larger.
                var new_cap = self.buf_capacity[unsafe_offset=slot] * 2
                if end_id > new_cap:
                    new_cap = end_id
                var new_buf_size = new_cap * bpt
                var _nb = unsafe_alloc[Int8](new_buf_size)
                var new_buf = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_nb))
                unsafe_memset(new_buf, 0, new_buf_size)
                # Copy live tokens from old buffer.
                var live_bytes = self.tokens_per_layer[unsafe_offset=slot] * bpt
                if live_bytes > 0:
                    unsafe_memcpy(dest=new_buf, src=self.v_turbo[unsafe_offset=slot], count=live_bytes)
                self.v_turbo[unsafe_offset=slot].unsafe_free()
                self.v_turbo[unsafe_offset=slot] = new_buf
                self.buf_capacity[unsafe_offset=slot] = new_cap

            var buf = self.v_turbo[unsafe_offset=slot]
            var rope_dim = self.layer_rope_dim[unsafe_offset=slot]  # 0 unless hybrid
            for ti in range(num_tokens):
                var src = values_fp32.unsafe_offset(ti * val_dim)
                var dst = buf.unsafe_offset((start_id + ti) * bpt)
                if vf == VFMT_TURBO4:
                    quantize_fp32_to_block_int4(src, dst, val_dim)
                elif vf == VFMT_TURBO3:
                    quantize_fp32_to_block_int3(src, dst, val_dim)
                elif vf == VFMT_TURBO2:
                    quantize_fp32_to_block_int2(src, dst, val_dim)
                elif vf == VFMT_FP8:
                    quantize_fp32_to_block_fp8(src, dst, val_dim)
                elif vf == VFMT_BF16_ROPE_FP8:
                    quantize_fp32_to_bf16_rope_fp8_body(src, dst, val_dim, rope_dim)
                elif vf == VFMT_MLX4G32:
                    quantize_fp32_to_mlx4_g32(src, dst, val_dim)
                else:  # FP16
                    var f16 = dst.unsafe_bitcast[Float16]()
                    for vi in range(val_dim):
                        f16[unsafe_offset=vi] = src[unsafe_offset=vi].cast[DType.float16]()

            self.v_fmt[unsafe_offset=slot] = vf
        else:
            # INT8 path: per-batch min/max (append semantics).
            # #41 fix: same realloc-grow as the turbo path. INT8 bpt = val_dim.
            if is_null(self.v_int8[unsafe_offset=slot]):
                var cap = end_id
                if cap < 32:
                    cap = 32
                var buf_size = cap * val_dim
                var _buf = unsafe_alloc[Int8](buf_size)
                self.v_int8[unsafe_offset=slot] = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_buf))
                unsafe_memset(self.v_int8[unsafe_offset=slot], 0, buf_size)
                self.buf_capacity[unsafe_offset=slot] = cap
            elif end_id > self.buf_capacity[unsafe_offset=slot]:
                var new_cap = self.buf_capacity[unsafe_offset=slot] * 2
                if end_id > new_cap:
                    new_cap = end_id
                var new_buf_size = new_cap * val_dim
                var _nb = unsafe_alloc[Int8](new_buf_size)
                var new_buf = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_nb))
                unsafe_memset(new_buf, 0, new_buf_size)
                var live_bytes = self.tokens_per_layer[unsafe_offset=slot] * val_dim
                if live_bytes > 0:
                    unsafe_memcpy(dest=new_buf, src=self.v_int8[unsafe_offset=slot], count=live_bytes)
                self.v_int8[unsafe_offset=slot].unsafe_free()
                self.v_int8[unsafe_offset=slot] = new_buf
                self.buf_capacity[unsafe_offset=slot] = new_cap

            # Compute min/max for this batch
            var total_vals = num_tokens * val_dim
            var vmin = values_fp32[unsafe_offset=0]
            var vmax = values_fp32[unsafe_offset=0]
            for vi in range(total_vals):
                var v = values_fp32[unsafe_offset=vi]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            var vrange = vmax - vmin
            if vrange < Float32(1e-8):
                vrange = Float32(1.0)
            var scale = Float32(254.0) / vrange

            var buf = self.v_int8[unsafe_offset=slot]
            for ti in range(num_tokens):
                var base_src = ti * val_dim
                var base_dst = (start_id + ti) * val_dim
                for vi in range(val_dim):
                    var normalized = (values_fp32[unsafe_offset=base_src + vi] - vmin) * scale
                    var clamped = min(max(normalized, Float32(0.0)), Float32(254.0))
                    buf[unsafe_offset=base_dst + vi] = Int8(Int(clamped) - 127)

            self.v_scale[unsafe_offset=slot] = scale
            self.v_min[unsafe_offset=slot] = vmin
            self.v_fmt[unsafe_offset=slot] = VFMT_INT8

        # Update token count
        var tpl_slot = session_idx * MAX_VS_LAYERS + layer_id
        if end_id > self.tokens_per_layer[unsafe_offset=tpl_slot]:
            self.tokens_per_layer[unsafe_offset=tpl_slot] = end_id
        if layer_id >= self.sessions[unsafe_offset=session_idx].num_layers:
            self.sessions[unsafe_offset=session_idx].num_layers = layer_id + 1
        self.total_tokens += num_tokens
        self._touch(session_idx)
        self._bump_gen(session_idx)

        return True

    # ── Fetch ───────────────────────────────────────────────────────────

    def fetch_tokens(
        mut self,
        session_idx: Int,
        layer_id: Int,
        token_ids: Pointer[Int32, MutUntrackedOrigin],
        num_ids: Int,
        output_fp32: Pointer[Float32, MutUntrackedOrigin],
    ) -> Int:
        """Fetch and dequantize V for requested token IDs.

        output_fp32: pre-allocated [num_ids * value_dim] buffer.
        Returns number of tokens successfully fetched.
        """
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return 0
        if not self.sessions[unsafe_offset=session_idx].active:
            return 0
        if layer_id < 0 or layer_id >= MAX_VS_LAYERS:
            return 0

        var slot = session_idx * MAX_VS_LAYERS + layer_id
        # A1: read per-layer dim. If unset (= 0), fall back to session-level
        # for backward-compat with snapshots that pre-date the per-layer array.
        var val_dim = self.layer_value_dim[unsafe_offset=slot]
        if val_dim <= 0:
            val_dim = self.sessions[unsafe_offset=session_idx].value_dim
        var vf = self.v_fmt[unsafe_offset=slot]
        var rope_dim = self.layer_rope_dim[unsafe_offset=slot]  # 0 unless hybrid
        # bytes-per-token: hybrid format depends on rope_dim.
        var bpt: Int
        if vf == VFMT_BF16_ROPE_FP8:
            bpt = self._bytes_per_token_hybrid(val_dim, rope_dim)
        else:
            bpt = self._bytes_per_token(vf, val_dim)
        var max_tok = self.tokens_per_layer[unsafe_offset=session_idx * MAX_VS_LAYERS + layer_id]
        var fetched = 0

        self.total_fetches += num_ids
        self._touch(session_idx)

        if vf == VFMT_TURBO4 or vf == VFMT_TURBO3 or vf == VFMT_TURBO2 or vf == VFMT_FP8 or vf == VFMT_BF16_ROPE_FP8 or vf == VFMT_MLX4G32:
            var buf = self.v_turbo[unsafe_offset=slot]
            if is_null(buf):
                return 0
            for i in range(num_ids):
                var tid = Int(token_ids[unsafe_offset=i])
                if tid < 0 or tid >= max_tok:
                    # Zero-fill for out-of-range
                    unsafe_memset(output_fp32.unsafe_offset(i * val_dim), 0, val_dim * 4)
                    continue
                var src = buf.unsafe_offset(tid * bpt)
                var dst = output_fp32.unsafe_offset(i * val_dim)
                if vf == VFMT_TURBO4:
                    dequantize_block_int4_to_fp32(src, dst, val_dim)
                elif vf == VFMT_TURBO3:
                    dequantize_block_int3_to_fp32(src, dst, val_dim)
                elif vf == VFMT_TURBO2:
                    dequantize_block_int2_to_fp32(src, dst, val_dim)
                elif vf == VFMT_FP8:
                    dequantize_block_fp8_to_fp32(src, dst, val_dim)
                elif vf == VFMT_MLX4G32:
                    dequantize_mlx4_g32_to_fp32(src, dst, val_dim)
                else:  # VFMT_BF16_ROPE_FP8
                    dequantize_bf16_rope_fp8_body_to_fp32(src, dst, val_dim, rope_dim)
                fetched += 1

        elif vf == VFMT_FP16:
            var buf = self.v_turbo[unsafe_offset=slot]  # FP16 reuses turbo pointer
            if is_null(buf):
                return 0
            for i in range(num_ids):
                var tid = Int(token_ids[unsafe_offset=i])
                if tid < 0 or tid >= max_tok:
                    unsafe_memset(output_fp32.unsafe_offset(i * val_dim), 0, val_dim * 4)
                    continue
                var src = buf.unsafe_offset(tid * bpt)
                var f16 = src.unsafe_bitcast[Float16]()
                var dst = output_fp32.unsafe_offset(i * val_dim)
                for vi in range(val_dim):
                    dst[unsafe_offset=vi] = f16[unsafe_offset=vi].cast[DType.float32]()
                fetched += 1

        else:
            # INT8
            var buf = self.v_int8[unsafe_offset=slot]
            if is_null(buf):
                return 0
            var scale = self.v_scale[unsafe_offset=slot]
            var vmin = self.v_min[unsafe_offset=slot]
            for i in range(num_ids):
                var tid = Int(token_ids[unsafe_offset=i])
                if tid < 0 or tid >= max_tok:
                    unsafe_memset(output_fp32.unsafe_offset(i * val_dim), 0, val_dim * 4)
                    continue
                var src = buf.unsafe_offset(tid * val_dim)
                var dst = output_fp32.unsafe_offset(i * val_dim)
                for vi in range(val_dim):
                    dst[unsafe_offset=vi] = (Float32(Int(src[unsafe_offset=vi]) + 127) / scale) + vmin
                fetched += 1

        return fetched

    def fetch_range_fp16_raw(
        mut self,
        session_idx: Int,
        layer_id: Int,
        start_id: Int,
        num_ids: Int,
        out_u8: Pointer[UInt8, MutUntrackedOrigin],
    ) -> Int:
        """gh #193 FMT NATIVE: copy stored fp16 bytes straight out for a
        contiguous token range — no fp16→fp32 expansion, half the wire bytes.
        Only valid for VFMT_FP16 slots (caller checks or gets 0 back). out_u8
        must hold num_ids * val_dim * 2 bytes. Tokens at/after the stored
        count are zero-filled, matching fetch_tokens' missing-token semantics
        (fp16 zeros upcast to the same fp32 zeros). Returns the number of
        tokens with stored data (0 → caller should reply nil)."""
        if session_idx < 0 or session_idx >= MAX_VS_SESSIONS:
            return 0
        if not self.sessions[unsafe_offset=session_idx].active:
            return 0
        if layer_id < 0 or layer_id >= MAX_VS_LAYERS:
            return 0
        if start_id < 0 or num_ids <= 0:
            return 0
        var slot = session_idx * MAX_VS_LAYERS + layer_id
        if self.v_fmt[unsafe_offset=slot] != VFMT_FP16:
            return 0
        var val_dim = self.layer_value_dim[unsafe_offset=slot]
        if val_dim <= 0:
            val_dim = self.sessions[unsafe_offset=session_idx].value_dim
        var bpt = val_dim * 2
        var buf = self.v_turbo[unsafe_offset=slot]  # FP16 lives in the turbo slot
        if is_null(buf):
            unsafe_memset(out_u8, 0, num_ids * bpt)
            return 0
        var max_tok = self.tokens_per_layer[unsafe_offset=slot]
        var live_end = start_id + num_ids
        if live_end > max_tok:
            live_end = max_tok
        var live = live_end - start_id
        if live < 0:
            live = 0
        if live > 0:
            unsafe_memcpy(dest=out_u8, src=(buf.unsafe_offset(start_id * bpt)).unsafe_bitcast[UInt8](), count=live * bpt)
        if live < num_ids:
            unsafe_memset(out_u8.unsafe_offset(live * bpt), 0, (num_ids - live) * bpt)
        self.total_fetches += num_ids
        self._touch(session_idx)
        return live

    # ── Disk persistence (snapshot, mirrors HNSW save/load convention) ─────

    def _vs_write_all(self, fd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        """Write exactly n bytes, retrying on short writes. False on any error
        (a full disk): the caller must not treat a short file as a snapshot."""
        var remaining = n
        var p = ptr
        while remaining > 0:
            var written = external_call["pion_write", Int](fd, p, remaining)
            if written <= 0:
                return False
            p = p.unsafe_offset(written)
            remaining -= written
        return True

    def _vs_read_all(self, fd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        """Read exactly n bytes. Returns False on short read or error."""
        var remaining = n
        var p = ptr
        while remaining > 0:
            var got = external_call["pion_read", Int](fd, p, remaining)
            if got <= 0:
                return False
            p = p.unsafe_offset(got)
            remaining -= got
        return True

    def save_to_disk(self, path: String) -> Bool:
        """Snapshot every active session to disk. Returns True on success.

        v=4 format (gh #71 — optional within-prefix block hash table):
          [magic 8B "PIONVS01"][version 4B = 4][active_count 4B][total_tokens 8B]
          [access_counter 8B][total_evictions 8B]
          for each active session:
            [sid_len 4B][sid bytes][default_value_dim 4B][session_v_format 1B]
            [num_layers 4B][last_access_ts 8B]
            [per_layer_descriptor[num_layers]]            ← v≥2
                v=2: each 5 bytes [layer_value_dim 4B][layer_v_format 1B]
                v=3+: each 7 bytes [layer_value_dim 4B][layer_v_format 1B][layer_rope_dim 2B]
            for each layer in [0..num_layers):
              [tokens 4B][v_fmt 1B]
              if v_fmt==INT8:   [scale 4B][min 4B][tokens × layer_value_dim Int8]
              else (turbo/fp16/fp8/hybrid): [tokens × bpt bytes]
            [block_size 4B][block_count 4B][hashes block_count×8B]   ← v=4 trailer

        v=1, v=2, v=3 reads still work (v<4 leaves block_size = block_count = 0,
        the BLOCKS / MEMBERSHIP commands answer +UNKNOWN for those sessions).
        """
        if not self.enabled or self.session_count == 0:
            return False

        # Written to <path>.tmp, flushed to stable storage, then renamed over
        # <path>. The caller truncates the V-store WAL once this returns True,
        # so a snapshot that is torn (crash mid-write), short (full disk) or
        # still in the drive's cache (power cut) would lose every row the WAL
        # held. Writing in place also destroyed the previous snapshot before
        # the new one was complete.
        var tmp_path = path + ".tmp"
        var fd = external_call["pion_creat", Int32](tmp_path.as_c_string_span())
        if fd < 0:
            print("V-store save: cannot open " + tmp_path)
            return False
        var wok = True

        # Header (40 bytes)
        var hdr = unsafe_alloc[UInt8](40)
        unsafe_memset(hdr, 0, 40)
        # magic "PIONVS01" little-endian
        hdr.unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(0x3130535649504E4F)  # ASCII "ONPIVS01" reversed for LE
        # ^ "PIONVS01" as 8 bytes in memory order: P=0x50,I=0x49,O=0x4F,N=0x4E,V=0x56,S=0x53,0=0x30,1=0x31
        # LE UInt64 = 0x31_30_53_56_4E_4F_49_50
        hdr.unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(0x3130535645)  # placeholder; we'll write magic byte-by-byte for clarity
        # Write magic byte-by-byte to be unambiguous
        hdr[unsafe_offset=0]=UInt8(80); hdr[unsafe_offset=1]=UInt8(73); hdr[unsafe_offset=2]=UInt8(79); hdr[unsafe_offset=3]=UInt8(78)
        hdr[unsafe_offset=4]=UInt8(86); hdr[unsafe_offset=5]=UInt8(83); hdr[unsafe_offset=6]=UInt8(48); hdr[unsafe_offset=7]=UInt8(49)
        hdr.unsafe_offset(8).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(4)                       # version (A1 1→2, A2 2→3, gh #71 3→4)
        (hdr.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(self.session_count)
        (hdr.unsafe_offset(16)).unsafe_bitcast[Int]()[unsafe_offset=0]    = self.total_tokens
        (hdr.unsafe_offset(24)).unsafe_bitcast[UInt64]()[unsafe_offset=0] = self.access_counter
        (hdr.unsafe_offset(32)).unsafe_bitcast[Int]()[unsafe_offset=0]    = self.total_evictions
        wok = self._vs_write_all(fd, hdr, 40) and wok
        hdr.unsafe_free()

        var saved_sessions = 0
        var saved_layers   = 0
        var saved_bytes    = 0

        for slot in range(MAX_VS_SESSIONS):
            if not self.sessions[unsafe_offset=slot].active:
                continue

            var meta = self.sessions[unsafe_offset=slot]
            # Per-session header (variable)
            var sh = unsafe_alloc[UInt8](32)
            unsafe_memset(sh, 0, 32)
            sh.unsafe_bitcast[UInt32]()[unsafe_offset=0]      = UInt32(meta.session_id_len)
            wok = self._vs_write_all(fd, sh, 4) and wok
            wok = self._vs_write_all(fd, meta.session_id_ptr, meta.session_id_len) and wok

            (sh.unsafe_offset(0)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(meta.value_dim)
            sh[unsafe_offset=4] = meta.v_format
            (sh.unsafe_offset(5)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(meta.num_layers)
            (sh.unsafe_offset(9)).unsafe_bitcast[UInt64]()[unsafe_offset=0] = meta.last_access_ts
            wok = self._vs_write_all(fd, sh, 17) and wok
            sh.unsafe_free()

            # A2 v=3: per-layer descriptor extended to 7 bytes per layer
            # (adds [layer_rope_dim 2B] for VFMT_BF16_ROPE_FP8). Other formats
            # write rope_dim=0; load reads it but ignores when fmt != hybrid.
            if meta.num_layers > 0:
                var desc_bytes = meta.num_layers * 7
                var desc = unsafe_alloc[UInt8](desc_bytes)
                unsafe_memset(desc, 0, desc_bytes)
                for li in range(meta.num_layers):
                    var li_slot = slot * MAX_VS_LAYERS + li
                    (desc.unsafe_offset(li * 7)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(self.layer_value_dim[unsafe_offset=li_slot])
                    desc[unsafe_offset=li * 7 + 4] = self.v_fmt[unsafe_offset=li_slot]
                    (desc.unsafe_offset(li * 7).unsafe_offset(5)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(self.layer_rope_dim[unsafe_offset=li_slot])
                wok = self._vs_write_all(fd, desc, desc_bytes) and wok
                desc.unsafe_free()

            for li in range(meta.num_layers):
                var li_slot = slot * MAX_VS_LAYERS + li
                var tokens  = self.tokens_per_layer[unsafe_offset=li_slot]
                var vf      = self.v_fmt[unsafe_offset=li_slot]
                var ldim    = self.layer_value_dim[unsafe_offset=li_slot]
                if ldim <= 0:
                    ldim = meta.value_dim

                var lh = unsafe_alloc[UInt8](16)
                unsafe_memset(lh, 0, 16)
                lh.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(tokens)
                lh[unsafe_offset=4] = vf
                wok = self._vs_write_all(fd, lh, 5) and wok
                lh.unsafe_free()

                if tokens == 0:
                    continue

                if vf == VFMT_INT8:
                    var sm = unsafe_alloc[UInt8](8)
                    sm.unsafe_bitcast[Float32]()[unsafe_offset=0]       = self.v_scale[unsafe_offset=li_slot]
                    (sm.unsafe_offset(4)).unsafe_bitcast[Float32]()[unsafe_offset=0] = self.v_min[unsafe_offset=li_slot]
                    wok = self._vs_write_all(fd, sm, 8) and wok
                    sm.unsafe_free()
                    var nbytes = tokens * ldim
                    if is_not_null(self.v_int8[unsafe_offset=li_slot]):
                        wok = self._vs_write_all(fd, self.v_int8[unsafe_offset=li_slot].unsafe_bitcast[UInt8](), nbytes) and wok
                        saved_bytes += nbytes
                else:
                    # turbo4 / turbo3 / turbo2 / fp16 / fp8 / hybrid
                    var bpt = self._bytes_per_token_for_slot(li_slot)
                    var nbytes = tokens * bpt
                    if is_not_null(self.v_turbo[unsafe_offset=li_slot]):
                        wok = self._vs_write_all(fd, self.v_turbo[unsafe_offset=li_slot].unsafe_bitcast[UInt8](), nbytes) and wok
                        saved_bytes += nbytes

                saved_layers += 1

            # gh #71 v=4: per-session block-hash trailer. Always emitted (zero
            # block_count is the explicit "no table registered" signal that
            # KV.PREFIX.BLOCKS / MEMBERSHIP translate to +UNKNOWN).
            var bsz  = self.sessions[unsafe_offset=slot].block_size
            var bcnt = self.sessions[unsafe_offset=slot].block_count
            var bt = unsafe_alloc[UInt8](8)
            bt.unsafe_bitcast[UInt32]()[unsafe_offset=0]       = bsz
            (bt.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = bcnt
            wok = self._vs_write_all(fd, bt, 8) and wok
            bt.unsafe_free()
            if Int(bcnt) > 0 and is_not_null(self.sessions[unsafe_offset=slot].block_hashes):
                wok = self._vs_write_all(fd, self.sessions[unsafe_offset=slot].block_hashes.unsafe_bitcast[UInt8](), Int(bcnt) * 8) and wok
                saved_bytes += Int(bcnt) * 8

            saved_sessions += 1

        # pion_fdatasync is F_FULLFSYNC on macOS: the bytes leave the drive's
        # volatile cache before the rename makes this the snapshot of record.
        wok = (external_call["pion_fdatasync", Int32](fd) == 0) and wok
        _ = external_call["close", Int32](fd)
        if not wok:
            _ = external_call["unlink", Int32](tmp_path.as_c_string_span())
            print("V-store save FAILED (write or flush error, disk full?) — previous snapshot and WAL kept")
            return False
        var dst_path = path
        if external_call["rename", Int32](tmp_path.as_c_string_span(), dst_path.as_c_string_span()) != 0:
            _ = external_call["unlink", Int32](tmp_path.as_c_string_span())
            print("V-store save FAILED: cannot rename " + tmp_path + " → " + path)
            return False
        print("V-store saved: " + String(saved_sessions) + " sessions, " + String(saved_layers) +
              " layer-blobs, " + String(saved_bytes) + " bytes → " + path)
        return True

    def load_from_disk(mut self, path: String) -> Bool:
        """Restore every session from a snapshot. Returns True if loaded.
        Does not error on missing file (cold start)."""
        if not self.enabled:
            return False

        var cpath = path
        var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_span())
        if fd < 0:
            return False  # No snapshot — cold start

        var hdr = unsafe_alloc[UInt8](40)
        if not self._vs_read_all(fd, hdr, 40):
            hdr.unsafe_free()
            _ = external_call["close", Int32](fd)
            print("V-store load: short header read")
            return False

        # Magic
        if hdr[unsafe_offset=0] != UInt8(80) or hdr[unsafe_offset=1] != UInt8(73) or hdr[unsafe_offset=2] != UInt8(79) or hdr[unsafe_offset=3] != UInt8(78) \
           or hdr[unsafe_offset=4] != UInt8(86) or hdr[unsafe_offset=5] != UInt8(83) or hdr[unsafe_offset=6] != UInt8(48) or hdr[unsafe_offset=7] != UInt8(49):
            hdr.unsafe_free()
            _ = external_call["close", Int32](fd)
            print("V-store load: bad magic")
            return False

        var version = (hdr.unsafe_offset(8)).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        if version != UInt32(1) and version != UInt32(2) and version != UInt32(3) and version != UInt32(4):
            hdr.unsafe_free()
            _ = external_call["close", Int32](fd)
            print("V-store load: unsupported version " + String(Int(version)))
            return False

        var saved_active   = Int((hdr.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
        var saved_total    = (hdr.unsafe_offset(16)).unsafe_bitcast[Int]()[unsafe_offset=0]
        var saved_counter  = (hdr.unsafe_offset(24)).unsafe_bitcast[UInt64]()[unsafe_offset=0]
        var saved_evict    = (hdr.unsafe_offset(32)).unsafe_bitcast[Int]()[unsafe_offset=0]
        hdr.unsafe_free()

        if saved_active <= 0 or saved_active > MAX_VS_SESSIONS:
            _ = external_call["close", Int32](fd)
            print("V-store load: invalid active count " + String(saved_active))
            return False

        var loaded_sessions = 0
        var loaded_layers   = 0

        for _ in range(saved_active):
            # Read session id
            var sid_len_buf = unsafe_alloc[UInt8](4)
            if not self._vs_read_all(fd, sid_len_buf, 4):
                sid_len_buf.unsafe_free()
                _ = external_call["close", Int32](fd)
                return False
            var sid_len = Int(sid_len_buf.unsafe_bitcast[UInt32]()[unsafe_offset=0])
            sid_len_buf.unsafe_free()
            if sid_len <= 0 or sid_len > 4096:
                _ = external_call["close", Int32](fd)
                print("V-store load: bad sid_len " + String(sid_len))
                return False

            var sid_ptr_raw = unsafe_alloc[UInt8](sid_len)
            var sid_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(sid_ptr_raw))
            if not self._vs_read_all(fd, sid_ptr, sid_len):
                _ = external_call["close", Int32](fd)
                return False

            var meta_buf = unsafe_alloc[UInt8](17)
            if not self._vs_read_all(fd, meta_buf, 17):
                meta_buf.unsafe_free()
                _ = external_call["close", Int32](fd)
                return False
            var value_dim   = Int(meta_buf.unsafe_bitcast[UInt32]()[unsafe_offset=0])
            var v_format    = meta_buf[unsafe_offset=4]
            var num_layers  = Int((meta_buf.unsafe_offset(5)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
            var last_ts     = (meta_buf.unsafe_offset(9)).unsafe_bitcast[UInt64]()[unsafe_offset=0]
            meta_buf.unsafe_free()

            # Re-create session (this allocates session_id_ptr afresh; the
            # version we read is consumed by _vs_read_all into sid_ptr_raw
            # which we'll free below).
            var slot = self.create_session(sid_ptr, sid_len, value_dim, v_format)
            if slot < 0:
                sid_ptr_raw.unsafe_free()
                _ = external_call["close", Int32](fd)
                print("V-store load: create_session failed mid-load")
                return False
            self.sessions[unsafe_offset=slot].last_access_ts = last_ts
            self.sessions[unsafe_offset=slot].num_layers = num_layers
            # Re-publish to the cross-worker directory after snapshot load
            # so multi-worker LOOKUP works on warm restart. Note: the schema
            # digest is set to 0 here (per-layer arrays not yet read);
            # wal_replay's SCHEMA branch sets the digest. A future revision
            # could refresh the digest after the per-layer block reads
            # (see git history for prior attempt).
            if is_not_null(self.directory) and self.my_worker_id >= 0:
                _ = vstore_dir_register(
                    self.directory, sid_ptr, sid_len, value_dim, v_format,
                    self.my_worker_id, last_ts, UInt32(0))
            sid_ptr_raw.unsafe_free()

            # A1 v=2: read per-layer descriptor block.
            # v=1: broadcast the session-level value_dim/v_format below as
            # we step through layer headers.
            # v=4: same per-layer stride as v=3 (no schema change); only the
            # per-session block-hash trailer differs.
            if (version == UInt32(2) or version == UInt32(3) or version == UInt32(4)) and num_layers > 0:
                # Per-layer descriptor — 5 bytes in v=2, 7 bytes in v=3+.
                var desc_stride = 5 if version == UInt32(2) else 7
                var desc_bytes = num_layers * desc_stride
                var desc = unsafe_alloc[UInt8](desc_bytes)
                if not self._vs_read_all(fd, desc, desc_bytes):
                    desc.unsafe_free()
                    _ = external_call["close", Int32](fd)
                    return False
                for li in range(num_layers):
                    var li_slot = slot * MAX_VS_LAYERS + li
                    self.layer_value_dim[unsafe_offset=li_slot] = Int((desc.unsafe_offset(li * desc_stride)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                    self.v_fmt[unsafe_offset=li_slot] = desc[unsafe_offset=li * desc_stride + 4]
                    if version == UInt32(3) or version == UInt32(4):
                        self.layer_rope_dim[unsafe_offset=li_slot] = Int((desc.unsafe_offset(li * desc_stride).unsafe_offset(5)).unsafe_bitcast[UInt16]()[unsafe_offset=0])
                desc.unsafe_free()

            for li in range(num_layers):
                var lh = unsafe_alloc[UInt8](5)
                if not self._vs_read_all(fd, lh, 5):
                    lh.unsafe_free()
                    _ = external_call["close", Int32](fd)
                    return False
                var tokens = Int(lh.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var vf     = lh[unsafe_offset=4]
                lh.unsafe_free()

                var li_slot = slot * MAX_VS_LAYERS + li
                self.tokens_per_layer[unsafe_offset=li_slot] = tokens
                # v=2: per-layer fmt was already populated from the descriptor.
                # v=1: lh[4] is the only source — broadcast was done by
                # create_session for value_dim; v_fmt is per-layer here.
                self.v_fmt[unsafe_offset=li_slot] = vf
                # Per-layer dim used for nbytes below.
                var ldim = self.layer_value_dim[unsafe_offset=li_slot]
                if ldim <= 0:
                    ldim = value_dim  # v=1 fallback
                    self.layer_value_dim[unsafe_offset=li_slot] = ldim
                if tokens == 0:
                    continue

                if vf == VFMT_INT8:
                    var sm = unsafe_alloc[UInt8](8)
                    if not self._vs_read_all(fd, sm, 8):
                        sm.unsafe_free()
                        _ = external_call["close", Int32](fd)
                        return False
                    self.v_scale[unsafe_offset=li_slot] = sm.unsafe_bitcast[Float32]()[unsafe_offset=0]
                    self.v_min[unsafe_offset=li_slot]   = (sm.unsafe_offset(4)).unsafe_bitcast[Float32]()[unsafe_offset=0]
                    sm.unsafe_free()
                    var nbytes = tokens * ldim
                    var _b = unsafe_alloc[Int8](nbytes)
                    self.v_int8[unsafe_offset=li_slot] = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_b))
                    if not self._vs_read_all(fd, self.v_int8[unsafe_offset=li_slot].unsafe_bitcast[UInt8](), nbytes):
                        _ = external_call["close", Int32](fd)
                        return False
                else:
                    # Hybrid format reads per-slot bpt (depends on rope_dim).
                    var bpt: Int
                    if vf == VFMT_BF16_ROPE_FP8:
                        bpt = self._bytes_per_token_hybrid(ldim, self.layer_rope_dim[unsafe_offset=li_slot])
                    else:
                        bpt = self._bytes_per_token(vf, ldim)
                    var nbytes = tokens * bpt
                    var _b = unsafe_alloc[Int8](nbytes)
                    self.v_turbo[unsafe_offset=li_slot] = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(_b))
                    if not self._vs_read_all(fd, self.v_turbo[unsafe_offset=li_slot].unsafe_bitcast[UInt8](), nbytes):
                        _ = external_call["close", Int32](fd)
                        return False

                loaded_layers += 1

            # gh #71 v=4: per-session block-hash trailer. v<4 snapshots leave
            # the table empty (block_size = block_count = 0 by __init__).
            if version == UInt32(4):
                var bt = unsafe_alloc[UInt8](8)
                if not self._vs_read_all(fd, bt, 8):
                    bt.unsafe_free()
                    _ = external_call["close", Int32](fd)
                    return False
                var bsz  = bt.unsafe_bitcast[UInt32]()[unsafe_offset=0]
                var bcnt = (bt.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0]
                bt.unsafe_free()
                if Int(bcnt) > 0:
                    var nbytes = Int(bcnt) * 8
                    var _stage = unsafe_alloc[UInt8](nbytes)
                    var stage = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_stage))
                    if not self._vs_read_all(fd, stage, nbytes):
                        _stage.unsafe_free()
                        _ = external_call["close", Int32](fd)
                        return False
                    # Route through set_block_hashes so both the token-order
                    # copy AND the sorted parallel array (for MEMBERSHIP) get
                    # built — same shape as a live REGISTER ... BLOCKS.
                    _ = self.set_block_hashes(slot, bsz, bcnt, stage)
                    _stage.unsafe_free()
                else:
                    self.sessions[unsafe_offset=slot].block_size = bsz
                    self.sessions[unsafe_offset=slot].block_count = bcnt

            loaded_sessions += 1

        _ = external_call["close", Int32](fd)
        # Restore monotonic counters
        if saved_counter > self.access_counter:
            self.access_counter = saved_counter
        self.total_evictions = saved_evict
        self.total_tokens = saved_total
        print("V-store loaded: " + String(loaded_sessions) + " sessions, " +
              String(loaded_layers) + " layer-blobs from " + path)
        return True

    # ── WAL (write-ahead log for V-store mutations) ─────────────────────────
    #
    # Format per record:
    #   [u32 payload_len] [u8 op] [payload of payload_len bytes]
    #
    # Ops:
    #   1 CREATE: [u32 sid_len][sid][u32 value_dim][u8 v_format]
    #   2 STORE:  [u32 sid_len][sid][u32 layer_id][u32 start_id][u32 num_tokens]
    #             [u32 value_dim][num_tokens × value_dim FP32 bytes]
    #   3 DROP:   [u32 sid_len][sid]
    #
    # Replay rule: process records in file order. STORE on a missing sid is a
    # no-op (the session was dropped earlier in the log). Truncated tail =
    # crash mid-write; stop replay at first short read.

    @always_inline
    def _wal_path_cstr_alloc(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Allocate a NUL-terminated copy of wal_path. Caller must .free() it."""
        var n = self.wal_path_len
        var _b = unsafe_alloc[UInt8](n + 1)
        var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_b))
        for i in range(n):
            p[unsafe_offset=i] = self.wal_path_buf[unsafe_offset=i]
        p[unsafe_offset=n] = UInt8(0)
        return p

    def wal_open(mut self, path: String):
        """Open WAL file in append mode. Call AFTER wal_replay() so replay
        doesn't see records we're about to write. Idempotent (closes old fd)."""
        if not self.enabled:
            return
        if self.wal_fd >= 0:
            _ = external_call["close", Int32](self.wal_fd)
            self.wal_fd = Int32(-1)
        # Save path for later truncate
        if is_not_null(self.wal_path_buf):
            self.wal_path_buf.unsafe_free()
            self.wal_path_buf = null_ptr[UInt8, MutUntrackedOrigin]()
            self.wal_path_len = 0
        self.wal_path_len = path.byte_length()
        var _b = unsafe_alloc[UInt8](self.wal_path_len + 1)
        var pbuf = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(_b))
        var src = path.unsafe_ptr()
        for i in range(self.wal_path_len):
            pbuf[unsafe_offset=i] = src[unsafe_offset=i]
        pbuf[unsafe_offset=self.wal_path_len] = UInt8(0)
        self.wal_path_buf = pbuf
        var cpath = path + "\0"
        var fd = external_call["pion_open_append", Int32](cpath.as_c_string_span())
        self.wal_fd = fd
        if fd < 0:
            print("V-store WAL: cannot open " + path)

    def wal_close(mut self):
        if self.wal_fd >= 0:
            _ = external_call["close", Int32](self.wal_fd)
            self.wal_fd = Int32(-1)

    def wal_truncate(mut self):
        """Drop accumulated records — call after a successful snapshot is on disk.
        Reopens the fd in append mode against a freshly truncated file."""
        if self.wal_path_len == 0:
            return
        if self.wal_fd >= 0:
            _ = external_call["close", Int32](self.wal_fd)
            self.wal_fd = Int32(-1)
        var cpath = self._wal_path_cstr_alloc()
        var trunc_fd = external_call["pion_creat", Int32](cpath)
        if trunc_fd >= 0:
            _ = external_call["close", Int32](trunc_fd)
        # Reopen append
        var append_fd = external_call["pion_open_append", Int32](cpath)
        cpath.unsafe_free()
        self.wal_fd = append_fd
        self.wal_appended = 0

    @always_inline
    def _wal_write_all(self, buf: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
        if self.wal_fd < 0:
            return False
        var remaining = n
        var p = buf
        while remaining > 0:
            var w = external_call["pion_write", Int](self.wal_fd, p, remaining)
            if w <= 0:
                return False
            p = p.unsafe_offset(w)
            remaining -= w
        return True

    def wal_append_create(mut self,
                          sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          sid_len: Int, value_dim: Int, v_format: UInt8):
        """Log a SESSION_CREATE record. Best-effort: WAL append failures are
        non-fatal (cache will lose this session on next restart, but in-memory
        state is unaffected)."""
        if self.wal_fd < 0:
            return
        var payload_len = 4 + sid_len + 4 + 1
        var rec = unsafe_alloc[UInt8](5 + payload_len)
        rec.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        rec[unsafe_offset=4] = UInt8(1)
        var p = rec.unsafe_offset(5)
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = p.unsafe_offset(4)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        (p.unsafe_offset(4).unsafe_offset(sid_len)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(value_dim)
        rec[unsafe_offset=5 + payload_len - 1] = v_format
        var ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec))
        if self._wal_write_all(ext, 5 + payload_len):
            self.wal_appended += 1
        rec.unsafe_free()

    def wal_append_create_schema(mut self,
                                 sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                                 sid_len: Int, default_value_dim: Int,
                                 num_layers: Int,
                                 per_layer_dim: Pointer[Int, MutUntrackedOrigin],
                                 per_layer_fmt: Pointer[UInt8, MutUntrackedOrigin],
                                 per_layer_rope: Pointer[Int, MutUntrackedOrigin] = null_ptr[Int, MutUntrackedOrigin]()):
        """A1 (gh #29) + bundle (gh #30 / #37): WAL CREATE for a heterogeneous session.

        Two sentinel-extended op=1 record formats coexist:

        v2 (legacy, no rope_dim):
          [op=1][payload_len 4B]
          [sid_len 4B][sid][0xFFFFFFFF 4B sentinel]
          [num_layers 1B][per_layer_desc[N]]
              each: [layer_value_dim 4B][layer_v_format 1B]   = 5 bytes
          [default_value_dim 4B]

        v3 (this revision, with rope_dim — emitted whenever any layer has a
        non-zero rope value, OR unconditionally for new writes — see code):
          [op=1][payload_len 4B]
          [sid_len 4B][sid][0xFFFFFFFE 4B sentinel]
          [num_layers 1B][per_layer_desc[N]]
              each: [layer_value_dim 4B][layer_v_format 1B][layer_rope_dim 2B]   = 7 bytes
          [default_value_dim 4B]

        Replay branches on the sentinel; old WAL records (0xFFFFFFFF) still
        parse via the v2 path with rope_dim = 0 — hybrid layers fall back to
        INT8 (which was the only behavior available before this commit).
        """
        if self.wal_fd < 0:
            return
        if num_layers <= 0 or num_layers > 255:
            return  # validated upstream; defensive

        # Decide v2 vs v3: emit v3 if ANY per-layer rope is set, else v2.
        # v3 is forward-compat — old binaries skip the unknown sentinel and
        # stop replay (the same wal_replay loop refuses unknown ops). New
        # binaries see both. Since A2 hybrid is the sole rope_dim consumer
        # today, v2 still applies to every non-hybrid SCHEMA session, keeping
        # the wire small.
        var has_rope = False
        if is_not_null(per_layer_rope):
            for li in range(num_layers):
                if per_layer_rope[unsafe_offset=li] > 0:
                    has_rope = True
                    break

        if not has_rope:
            # v2 legacy format (no rope) — unchanged from A1.
            var per_layer_bytes_v2 = num_layers * 5
            var payload_len_v2 = 4 + sid_len + 4 + 1 + per_layer_bytes_v2 + 4
            var rec_v2 = unsafe_alloc[UInt8](5 + payload_len_v2)
            rec_v2.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len_v2)
            rec_v2[unsafe_offset=4] = UInt8(1)
            var p2 = rec_v2.unsafe_offset(5)
            p2.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
            var sd2 = p2.unsafe_offset(4)
            for i in range(sid_len):
                sd2[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
            var q2 = p2.unsafe_offset(4).unsafe_offset(sid_len)
            q2.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(0xFFFFFFFF)              # v2 sentinel
            (q2.unsafe_offset(4))[unsafe_offset=0] = UInt8(num_layers)
            var desc2 = q2.unsafe_offset(5)
            for li in range(num_layers):
                (desc2.unsafe_offset(li * 5)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(per_layer_dim[unsafe_offset=li])
                desc2[unsafe_offset=li * 5 + 4] = per_layer_fmt[unsafe_offset=li]
            (desc2.unsafe_offset(per_layer_bytes_v2)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(default_value_dim)
            var ext_v2 = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec_v2))
            if self._wal_write_all(ext_v2, 5 + payload_len_v2):
                self.wal_appended += 1
            rec_v2.unsafe_free()
            return

        # v3 format (with per-layer rope_dim) — emitted only for hybrid SCHEMA.
        var per_layer_bytes = num_layers * 7
        var payload_len = 4 + sid_len + 4 + 1 + per_layer_bytes + 4
        var rec = unsafe_alloc[UInt8](5 + payload_len)
        rec.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        rec[unsafe_offset=4] = UInt8(1)
        var p = rec.unsafe_offset(5)
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = p.unsafe_offset(4)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        var q = p.unsafe_offset(4).unsafe_offset(sid_len)
        q.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(0xFFFFFFFE)              # v3 sentinel
        (q.unsafe_offset(4))[unsafe_offset=0] = UInt8(num_layers)
        var desc = q.unsafe_offset(5)
        for li in range(num_layers):
            (desc.unsafe_offset(li * 7)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(per_layer_dim[unsafe_offset=li])
            desc[unsafe_offset=li * 7 + 4] = per_layer_fmt[unsafe_offset=li]
            (desc.unsafe_offset(li * 7).unsafe_offset(5)).unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(per_layer_rope[unsafe_offset=li])
        (desc.unsafe_offset(per_layer_bytes)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(default_value_dim)
        var ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec))
        if self._wal_write_all(ext, 5 + payload_len):
            self.wal_appended += 1
        rec.unsafe_free()

    def wal_append_blocks(mut self,
                          sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          sid_len: Int,
                          block_size: UInt32,
                          block_count: UInt32,
                          hashes: Pointer[UInt8, MutUntrackedOrigin]):
        """gh #71: Log a BLOCKS record (op=4) carrying the within-prefix block
        hash table for `<sid>`. Replay attaches the hashes to the session
        recreated by the preceding op=1. Best-effort like the other wal_append_*."""
        if self.wal_fd < 0:
            return
        var blob_bytes = Int(block_count) * 8
        var payload_len = 4 + sid_len + 4 + 4 + blob_bytes
        var rec = unsafe_alloc[UInt8](5 + payload_len)
        rec.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        rec[unsafe_offset=4] = UInt8(4)
        var p = rec.unsafe_offset(5)
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = p.unsafe_offset(4)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        var q = p.unsafe_offset(4).unsafe_offset(sid_len)
        q.unsafe_bitcast[UInt32]()[unsafe_offset=0]        = block_size
        (q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0]  = block_count
        if blob_bytes > 0:
            unsafe_memcpy(dest=(q.unsafe_offset(8)), src=hashes, count=blob_bytes)
        var ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec))
        if self._wal_write_all(ext, 5 + payload_len):
            self.wal_appended += 1
        rec.unsafe_free()

    def wal_append_drop(mut self,
                        sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                        sid_len: Int):
        if self.wal_fd < 0:
            return
        var payload_len = 4 + sid_len
        var rec = unsafe_alloc[UInt8](5 + payload_len)
        rec.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        rec[unsafe_offset=4] = UInt8(3)
        (rec.unsafe_offset(5)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = rec.unsafe_offset(9)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        var ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec))
        if self._wal_write_all(ext, 5 + payload_len):
            self.wal_appended += 1
        rec.unsafe_free()

    def wal_append_storebatch(mut self,
                              sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                              sid_len: Int, layer_id: Int, start_id: Int,
                              num_tokens: Int, value_dim: Int,
                              fp32: Pointer[Float32, MutUntrackedOrigin]):
        """Log raw FP32 inputs to STOREBATCH. Replay re-quantizes deterministically.
        Costs N×D×4 bytes per call — at 30 tokens × 1024D that's 120 KB; for a
        long prompt × many layers a single bench can write hundreds of MB. Call
        KV.PREFIX.SAVE to compact (snapshot + truncate)."""
        if self.wal_fd < 0:
            return
        var fp32_bytes = num_tokens * value_dim * 4
        var fixed = 4 + sid_len + 4 + 4 + 4 + 4
        var payload_len = fixed + fp32_bytes
        # Header (5 bytes) + fixed prefix (4+sid_len+16) — write in one buffer,
        # then stream the FP32 blob without copying.
        var pre = unsafe_alloc[UInt8](5 + fixed)
        pre.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        pre[unsafe_offset=4] = UInt8(2)
        var p = pre.unsafe_offset(5)
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = p.unsafe_offset(4)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        var q = p.unsafe_offset(4).unsafe_offset(sid_len)
        q.unsafe_bitcast[UInt32]()[unsafe_offset=0]       = UInt32(layer_id)
        (q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(start_id)
        (q.unsafe_offset(8)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(num_tokens)
        (q.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(value_dim)
        var pre_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pre))
        var ok1 = self._wal_write_all(pre_ext, 5 + fixed)
        pre.unsafe_free()
        if not ok1:
            return
        var fp32_u8 = fp32.unsafe_bitcast[UInt8]()
        if self._wal_write_all(fp32_u8, fp32_bytes):
            self.wal_appended += 1

    def wal_barrier(mut self) -> Bool:
        """The V-store WAL is written with write(): it survives the process
        dying (page cache) but not a power cut or kernel panic until the file
        is flushed. KV.PREFIX.COMMIT calls this; see WAL.barrier()."""
        if self.wal_fd < 0:
            return True
        return external_call["pion_fdatasync", Int32](self.wal_fd) == 0

    def wal_append_storebatch_f16(mut self,
                                  sid_ptr: Pointer[UInt8, MutUntrackedOrigin],
                                  sid_len: Int, layer_id: Int, start_id: Int,
                                  num_tokens: Int, value_dim: Int,
                                  fp32: Pointer[Float32, MutUntrackedOrigin]):
        """op 5: STOREBATCH into an fp16-stored layer, logged as fp16 — the
        values the layer actually holds. Same layout as op 2 with a 2-byte
        payload element, so the log is half the size; replay widens to fp32
        and store_batch narrows back to the identical fp16."""
        if self.wal_fd < 0:
            return
        var n_vals = num_tokens * value_dim
        var f16_bytes = n_vals * 2
        var fixed = 4 + sid_len + 4 + 4 + 4 + 4
        var payload_len = fixed + f16_bytes
        var pre = unsafe_alloc[UInt8](5 + fixed)
        pre.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(payload_len)
        pre[unsafe_offset=4] = UInt8(5)
        var p = pre.unsafe_offset(5)
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(sid_len)
        var sd = p.unsafe_offset(4)
        for i in range(sid_len):
            sd[unsafe_offset=i] = sid_ptr[unsafe_offset=i]
        var q = p.unsafe_offset(4).unsafe_offset(sid_len)
        q.unsafe_bitcast[UInt32]()[unsafe_offset=0]       = UInt32(layer_id)
        (q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(start_id)
        (q.unsafe_offset(8)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(num_tokens)
        (q.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(value_dim)
        var pre_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pre))
        var ok1 = self._wal_write_all(pre_ext, 5 + fixed)
        pre.unsafe_free()
        if not ok1:
            return
        var h = unsafe_alloc[Float16](n_vals)
        for vi in range(n_vals):
            h[unsafe_offset=vi] = fp32[unsafe_offset=vi].cast[DType.float16]()
        var h_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(h))
        if self._wal_write_all(h_ext, f16_bytes):
            self.wal_appended += 1
        h.unsafe_free()

    def wal_replay(mut self, path: String) -> Int:
        """Replay records onto current state. Call AFTER load_from_disk and
        BEFORE wal_open. Stops at the first short read (truncated tail = crash
        mid-write, safe to drop). Returns count of replayed ops."""
        if not self.enabled:
            return 0
        var cpath = path + "\0"
        var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_span())
        if fd < 0:
            return 0
        var n_create = 0
        var n_store = 0
        var n_drop = 0
        var n_skip = 0
        # Header buf reused per record.
        var hdr_buf = unsafe_alloc[UInt8](5)
        var hdr_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(hdr_buf))
        while True:
            if not self._vs_read_all(fd, hdr_ext, 5):
                break  # end of file or truncated record header
            var payload_len = Int(hdr_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
            var op = hdr_ext[unsafe_offset=4]
            if payload_len <= 0 or payload_len > 1 << 30:
                # Sanity: refuse to allocate >1GB or zero-len records.
                print("V-store WAL replay: bad payload_len " + String(payload_len) + " at op " + String(Int(op)))
                break
            var pb = unsafe_alloc[UInt8](payload_len)
            var p_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pb))
            if not self._vs_read_all(fd, p_ext, payload_len):
                pb.unsafe_free()
                break  # truncated tail
            if op == UInt8(1):
                # CREATE
                if payload_len < 9:
                    pb.unsafe_free(); break
                var sid_len = Int(p_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                if sid_len < 0 or sid_len > payload_len - 9:
                    pb.unsafe_free(); break
                var sid = p_ext.unsafe_offset(4)
                var value_dim_raw = (p_ext.unsafe_offset(4).unsafe_offset(sid_len)).unsafe_bitcast[UInt32]()[unsafe_offset=0]
                var value_dim = Int(value_dim_raw)
                var v_format = p_ext[unsafe_offset=4 + sid_len + 4]
                # A1 (gh #29) + bundle: two SCHEMA sentinels for op=1 records.
                #   0xFFFFFFFF → v2 record (5-byte per-layer descriptor, no rope_dim)
                #   0xFFFFFFFE → v3 record (7-byte per-layer descriptor, with rope_dim)
                # Old WAL files (only v2 sentinel ever existed) replay correctly
                # via the v2 branch with rope_dim=0 — matches the old fall-back
                # behavior where hybrid layers degraded to INT8 on warm restart.
                if value_dim_raw == UInt32(0xFFFFFFFF) or value_dim_raw == UInt32(0xFFFFFFFE):
                    var nl = Int(v_format)  # repurposed slot = num_layers
                    if nl <= 0 or nl > 255:
                        pb.unsafe_free(); break
                    var desc_stride = 5 if value_dim_raw == UInt32(0xFFFFFFFF) else 7
                    var per_layer_bytes = nl * desc_stride
                    var fixed_after_sentinel = 1 + per_layer_bytes + 4
                    if payload_len < 4 + sid_len + 4 + fixed_after_sentinel:
                        pb.unsafe_free(); break
                    var desc = p_ext.unsafe_offset(4).unsafe_offset(sid_len).unsafe_offset(5)
                    var ldim_buf = unsafe_alloc[Int](nl)
                    var lfmt_buf = unsafe_alloc[UInt8](nl)
                    var lrope_buf = unsafe_alloc[Int](nl)
                    var ldim_ext = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(ldim_buf))
                    var lfmt_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(lfmt_buf))
                    var lrope_ext = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(lrope_buf))
                    for li in range(nl):
                        ldim_ext[unsafe_offset=li] = Int((desc.unsafe_offset(li * desc_stride)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                        lfmt_ext[unsafe_offset=li] = desc[unsafe_offset=li * desc_stride + 4]
                        if desc_stride == 7:
                            lrope_ext[unsafe_offset=li] = Int((desc.unsafe_offset(li * desc_stride).unsafe_offset(5)).unsafe_bitcast[UInt16]()[unsafe_offset=0])
                        else:
                            lrope_ext[unsafe_offset=li] = 0
                    var default_dim = Int((desc.unsafe_offset(per_layer_bytes)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                    var slot_replayed = self.create_session_schema(sid, sid_len, default_dim, nl, ldim_ext, lfmt_ext, lrope_ext)
                    var first_fmt = lfmt_ext[unsafe_offset=0]  # capture before free
                    # Bundle gh #29 §8.2: compute schema digest from the
                    # per-layer arrays as they exist NOW in the slot — captures
                    # any fallbacks that create_session_schema applied (e.g.
                    # invalid rope_dim → INT8 fallback).
                    var sch_digest: UInt32 = UInt32(0)
                    if slot_replayed >= 0:
                        sch_digest = vstore_compute_schema_digest(
                            self.sessions[unsafe_offset=slot_replayed].num_layers,
                            self.layer_value_dim.unsafe_offset(slot_replayed * MAX_VS_LAYERS),
                            self.v_fmt.unsafe_offset(slot_replayed * MAX_VS_LAYERS),
                            self.layer_rope_dim.unsafe_offset(slot_replayed * MAX_VS_LAYERS),
                        )
                    ldim_buf.unsafe_free()
                    lfmt_buf.unsafe_free()
                    lrope_buf.unsafe_free()
                    n_create += 1
                    if is_not_null(self.directory) and self.my_worker_id >= 0:
                        _ = vstore_dir_register(
                            self.directory, sid, sid_len, default_dim, first_fmt,
                            self.my_worker_id, UInt64(n_create), sch_digest)
                    pb.unsafe_free()
                    continue
                _ = self.create_session(sid, sid_len, value_dim, v_format)
                n_create += 1
                # Re-publish to the cross-worker directory so KV.PREFIX.LOOKUP
                # works across workers after warm restart, not just after live
                # REGISTER. Without this, post-restart only the local find
                # path resolves and other workers see MISS until next REGISTER.
                if is_not_null(self.directory) and self.my_worker_id >= 0:
                    _ = vstore_dir_register(
                        self.directory, sid, sid_len, value_dim, v_format,
                        self.my_worker_id, UInt64(n_create))
            elif op == UInt8(2):
                # STORE
                if payload_len < 24:
                    pb.unsafe_free(); break
                var sid_len = Int(p_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                if sid_len < 0 or sid_len > payload_len - 24:
                    pb.unsafe_free(); break
                var sid = p_ext.unsafe_offset(4)
                var q = p_ext.unsafe_offset(4).unsafe_offset(sid_len)
                var layer_id  = Int(q.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var start_id  = Int((q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var num_tok   = Int((q.unsafe_offset(8)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                # value_dim from the WAL record — store_batch reads it from
                # the session meta, but we sanity-check that the record matches
                # the recreated session before replaying (mismatch = WAL is
                # for a different model config and replay would corrupt state).
                var rec_dim   = Int((q.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var fp32 = (q.unsafe_offset(16)).unsafe_bitcast[Float32]()
                var idx = self._find_session(sid, sid_len)
                if idx >= 0 and self.sessions[unsafe_offset=idx].value_dim == rec_dim:
                    _ = self.store_batch(idx, layer_id, start_id, num_tok, fp32)
                    n_store += 1
                else:
                    n_skip += 1
            elif op == UInt8(5):
                # STORE, fp16 payload (fp16-stored layer). Same layout as op 2.
                if payload_len < 24:
                    pb.unsafe_free(); break
                var sid_len = Int(p_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                if sid_len < 0 or sid_len > payload_len - 24:
                    pb.unsafe_free(); break
                var sid = p_ext.unsafe_offset(4)
                var q = p_ext.unsafe_offset(4).unsafe_offset(sid_len)
                var layer_id  = Int(q.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var start_id  = Int((q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var num_tok   = Int((q.unsafe_offset(8)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var rec_dim   = Int((q.unsafe_offset(12)).unsafe_bitcast[UInt32]()[unsafe_offset=0])
                var n_vals = num_tok * rec_dim
                if 4 + sid_len + 16 + n_vals * 2 > payload_len:
                    pb.unsafe_free(); break
                var h = (q.unsafe_offset(16)).unsafe_bitcast[Float16]()
                var idx = self._find_session(sid, sid_len)
                if idx >= 0 and self.sessions[unsafe_offset=idx].value_dim == rec_dim:
                    var wide = unsafe_alloc[Float32](n_vals)
                    for vi in range(n_vals):
                        wide[unsafe_offset=vi] = h[unsafe_offset=vi].cast[DType.float32]()
                    var wide_ext = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(wide))
                    _ = self.store_batch(idx, layer_id, start_id, num_tok, wide_ext)
                    wide.unsafe_free()
                    n_store += 1
                else:
                    n_skip += 1
            elif op == UInt8(3):
                # DROP
                if payload_len < 4:
                    pb.unsafe_free(); break
                var sid_len = Int(p_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                if sid_len < 0 or sid_len > payload_len - 4:
                    pb.unsafe_free(); break
                var sid = p_ext.unsafe_offset(4)
                var idx = self._find_session(sid, sid_len)
                if idx >= 0:
                    _ = self.drop_session(idx)
                    n_drop += 1
                    # Mirror the drop in the cross-worker directory.
                    if is_not_null(self.directory):
                        _ = vstore_dir_drop(self.directory, sid, sid_len)
                else:
                    n_skip += 1
            elif op == UInt8(4):
                # gh #71: BLOCKS — within-prefix block hash table for sid.
                # Layout: [sid_len 4B][sid][block_size 4B][block_count 4B][hashes count*8B]
                if payload_len < 12:
                    pb.unsafe_free(); break
                var sid_len = Int(p_ext.unsafe_bitcast[UInt32]()[unsafe_offset=0])
                if sid_len < 0 or sid_len > payload_len - 12:
                    pb.unsafe_free(); break
                var sid = p_ext.unsafe_offset(4)
                var q = p_ext.unsafe_offset(4).unsafe_offset(sid_len)
                var bsz   = q.unsafe_bitcast[UInt32]()[unsafe_offset=0]
                var bcnt  = (q.unsafe_offset(4)).unsafe_bitcast[UInt32]()[unsafe_offset=0]
                var blob_bytes = Int(bcnt) * 8
                if 4 + sid_len + 8 + blob_bytes > payload_len:
                    pb.unsafe_free(); break
                var idx = self._find_session(sid, sid_len)
                if idx >= 0:
                    _ = self.set_block_hashes(idx, bsz, bcnt, q.unsafe_offset(8))
                else:
                    n_skip += 1
            else:
                # Unknown op — stop, don't blindly skip.
                print("V-store WAL replay: unknown op " + String(Int(op)) + ", stopping at offset before this record")
                pb.unsafe_free()
                break
            pb.unsafe_free()
        hdr_buf.unsafe_free()
        _ = external_call["close", Int32](fd)
        var total = n_create + n_store + n_drop
        self.wal_replayed = total
        if total > 0 or n_skip > 0:
            print("V-store WAL replayed: " + String(n_create) + " create, " +
                  String(n_store) + " store, " + String(n_drop) + " drop, " +
                  String(n_skip) + " skipped")
        return total
