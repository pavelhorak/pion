"""WAL — Write-Ahead Log (Phase 1: mmap ring buffer, group commit, startup recovery).

File format (pion.wal.{worker_id}):

  Header (64 bytes):
    [8B] magic: 0x314C415750494F50  ("PIONWAL1" little-endian)
    [8B] tail_offset — byte offset from data-section start of next write
    [8B] worker_id
    [8B] seq — monotonic entry counter (incremented per append)
    [32B] reserved

  Data section (byte 64 → file_size):
    Entries packed contiguously. No wrap.
    Entry format (all ints little-endian):
      [4B] entry_len  — total byte length including this field
      [1B] cmd_id
      [4B] key_len
      [key_len bytes]
      [4B] val_len
      [val_len bytes]

cmd_id values:
  1 = SET   — restore as keyspace.set(key, val)
  2 = DEL   — restore as keyspace.remove_generic(key)
  3 = legacy HSET stub (key only, pre-gh #170) — ignored on replay
  4 = SET_BLOB (gh #163) — value field is a 24-byte pointer record into the blob
      arena, [8B segment][8B offset][8B length], not the payload. Keeps the WAL
      the single ordered index over both tiers, so a large value overwriting a
      small one (or the reverse) replays correctly with no merge logic — and a
      6 MB value costs 29 log bytes instead of 6 MB.

gh #170 — aggregate effect records. Mutating commands on HASH/LIST/SET/ZSET/
GEO/BITMAP values log the *effect*, decomposed into the primitives below, and
replay applies them through the same semantics as the live execute path (e.g.
ZADD replays as an unconditional skiplist insert, exactly like execute_zadd).
Non-deterministic commands must log their resolved effect, never their
arguments: SPOP logs an SREM of the member it actually removed, ZPOPMIN/ZPOPMAX
log a ZREM. Snapshots (v2) serialize aggregates as these same records, so one
apply function (`wal_apply_aggregate`) serves both files.

  5 = HSET    val = [4B field_len][field][4B value_len][value]
  6 = LPUSH   val = element
  7 = RPUSH   val = element
  8 = SADD    val = member
  9 = ZADD    val = [8B f64 score][member]
 10 = HDEL    val = field
 11 = SREM    val = member
 12 = ZREM    val = member
 13 = LPOP    no val
 14 = RPOP    no val
 15 = GEOADD  val = [8B f64 score(geohash bits)][member]  (GEO-typed zset)
 16 = BITMAP_IMG val = raw bitmap bytes            (snapshot/compact image)
 17 = HLL_IMG    val = 16384B HLL registers        (snapshot/compact image)
 18 = SETBIT  val = [8B packed u64: (offset << 1) | bit]
 19 = LSET    val = [8B resolved index][element]
 20 = LTRIM   val = [8B resolved start][8B resolved stop] (inclusive)
 21 = LREM    val = [8B count (two's-complement i64)][element]
 22 = LINSERT val = [1B where (1=BEFORE,0=AFTER)][4B pivot_len][pivot][element]
 31 = MSET    key = empty, val = N x [varint key_len][key][varint val_len][value]
              One record per MSET instead of N cmd-1 records: the 13-byte
              header is paid once and each length is a 1-byte LEB128 below 128.
              Durable MSET is bound by WAL bytes (measured: +31% log bytes cost
              -6% MSET throughput, and -0.5% with --no-wal), and the gate's
              10-key MSET logs 223 bytes instead of 320. Replays exactly like
              the N SETs it replaces.

gh #174 — the remainder #170 left open: streams, TTLs and HLL mutations. Same
rules apply (log the resolved effect, never the argument).

 23 = XADD     val = [8B id_ms][8B id_seq][packed field-value pairs]
               The ID is the *resolved* one — `XADD key *` logs the ID the
               server generated, so a replay reproduces the same stream rather
               than re-deriving IDs from a different wall clock. The payload is
               byte-identical to StreamEntry.data ([u16 flen][f][u16 vlen][v]…),
               so replay is a straight StreamData.append with no re-encoding.
 24 = PFADD    val = element. Replay re-runs hll_add, which is deterministic
               (murmur3 of the element → register max), so replaying an element
               twice is idempotent and converges to the same registers.
 25 = EXPIREAT val = [8B i64 absolute expiry, nanoseconds] — the resolved
               deadline, so `EXPIRE key 60` replays to the same wall-clock
               instant it originally meant instead of 60 s after recovery.
 26 = PERSIST  no val — drops the key's TTL.
 27 = XDEL     val = [8B id_ms][8B id_seq] — tombstones one entry (XDEL, and
               the resolved effect of MAXLEN trimming).

gh #378 — vector sets (ValueType.VSET, gh #366). The set is created by its
first VADD record, at that record's dimension, and removed with its last
element exactly as the live VREM does.

 28 = VADD     val = [4B name_len][name][4B plen][4B f32 norm][dim x f32]
               The STORED representation — the unit vector and norm VADD
               computed — not the caller's vector, so replay restores the set
               bit for bit instead of re-normalizing (dim = (plen - 4) / 4).
               A VADD onto an existing element replaces its vector.
 29 = VREM     val = element name.
 30 = VSETATTR val = [4B name_len][name][4B attr_len][attr]; empty = none.

HLL PFMERGE logs its result as a cmd-17 image rather than a cmd-24 element:
the merge output is a register-wise max over sources, which no element replay
can reconstruct.

Hot path: append() is a pure memcpy into the mmap'd region — zero syscalls.
Group commit: sync() calls msync(MS_ASYNC) once per event-loop tick.

gh #149 — segment rotation. The log used to be a single fixed 256 MB file that
silently stopped recording once full: append() returned without writing and the
client still got +OK, so every write past the 256 MB mark was acknowledged and
then lost on restart. Measured before the fix: of 50 x 6 MB SET blobs, 42
survived a SIGKILL and 8 vanished, while a small value written *after* them
survived — small entries still fit the tail slack that a blob no longer did.
That is the exact signature reported in gh #149 (720 blobs gone, 20 small metas
kept).

Now a full active file is sealed and renamed to `{path}.{n}` (n ascending from
1) and a fresh active file takes its place; recover() replays the sealed
segments in order and then the active one. Only when `max_segments` is exhausted
does an append drop — and that drop is counted, logged once, and reported in
INFO Persistence (`wal_dropped_entries`), never silent.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.container_free import remove_and_free
from std.memory.unsafe_pointer import Pointer
from std.memory import unsafe_memcpy, unsafe_memset, alloc
from std.ffi import external_call
from std.collections import List, Span

from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue, ValueType
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.bitmap import getbit, setbit
from src.common.hll import HLL_REGISTERS, hll_add
from src.common.stream_data import StreamData
from src.io.blob_store import BlobStore
from src.common.vector_set import VectorSet, free_vset


comptime WAL_FILE_SIZE    = 256 * 1024 * 1024   # 256 MB — ~2-3M entries at avg 90 B
comptime WAL_HEADER_SIZE  = 64
comptime WAL_DATA_SIZE    = WAL_FILE_SIZE - WAL_HEADER_SIZE
comptime WAL_MAGIC        = UInt64(0x314C415750494F50)   # "PIONWAL1" LE
comptime MS_ASYNC         = Int32(1)
comptime MS_SYNC          = Int32(2)
# gh #149: how many sealed segments may accumulate before appends start dropping.
# 32 x 256 MB = 8 GB of unsnapshotted delta — past that the operator wants a
# SAVE/BGSAVE, not more disk.
comptime WAL_MAX_SEGMENTS = 32
# gh #149 perf: alignment grain for the delta msync in sync(). 16 KB is the
# macOS ARM page size and a multiple of Linux's 4 KB, so a boundary aligned to
# it satisfies msync's page-alignment requirement on both.
comptime WAL_SYNC_PAGE = 16384


@always_inline
def _file_exists(path: String) -> Bool:
    """access(path, F_OK) — probe without creating (pion_wal_open would create)."""
    var p = path
    return external_call["access", Int32](p.as_c_string_slice(), Int32(0)) == 0


# ── gh #170: aggregate record replay (shared by WAL replay + snapshot load) ──
#
# Replay constructs aggregates with direct alloc, mirroring the dispatcher's
# pool-exhaustion fallback. Mixing pooled and alloc'd aggregates is safe: DEL
# only frees STRING payloads (free_str_payload), aggregate structs are never
# pool-returned.

@always_inline
def _rec_u32(p: Pointer[UInt8, MutUntrackedOrigin]) -> UInt32:
    return UInt32(p[unsafe_offset=0]) | (UInt32(p[unsafe_offset=1]) << 8) | \
           (UInt32(p[unsafe_offset=2]) << 16) | (UInt32(p[unsafe_offset=3]) << 24)

@always_inline
def _rec_u64(p: Pointer[UInt8, MutUntrackedOrigin]) -> UInt64:
    return UInt64(_rec_u32(p)) | (UInt64(_rec_u32(p.unsafe_offset(4))) << 32)


def _replay_hash(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 key: GenericValue, create: Bool) -> Pointer[SlabHashMap, MutUntrackedOrigin]:
    var val = keyspace[].get(key)
    if val.is_none():
        if not create:
            return null_ptr[SlabHashMap, MutUntrackedOrigin]()
        var hp = alloc[SlabHashMap](1)
        hp.unsafe_write(SlabHashMap(16))
        var nv = GenericValue()
        nv.type = ValueType(ValueType.HASH)
        nv.set_ptr(hp.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return hp
    if val.type.value == ValueType.HASH:
        return val.as_hash().unsafe_bitcast[SlabHashMap]()
    return null_ptr[SlabHashMap, MutUntrackedOrigin]()


def _replay_stream(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                   key: GenericValue, create: Bool) -> Pointer[StreamData, MutUntrackedOrigin]:
    """gh #174: same shape as _replay_hash, for STREAM-typed values."""
    var val = keyspace[].get(key)
    if val.is_none():
        if not create:
            return null_ptr[StreamData, MutUntrackedOrigin]()
        var sp = alloc[StreamData](1)
        sp.unsafe_write(StreamData())
        var nv = GenericValue()
        nv.type = ValueType(ValueType.STREAM)
        nv.set_ptr(sp.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return sp
    if val.type.value == ValueType.STREAM:
        return val.as_hash().unsafe_bitcast[StreamData]()
    return null_ptr[StreamData, MutUntrackedOrigin]()


def _replay_set(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                key: GenericValue, create: Bool) -> Pointer[SlabHashMap, MutUntrackedOrigin]:
    var val = keyspace[].get(key)
    if val.is_none():
        if not create:
            return null_ptr[SlabHashMap, MutUntrackedOrigin]()
        var sp = alloc[SlabHashMap](1)
        sp.unsafe_write(SlabHashMap(16, 100))
        var nv = GenericValue()
        nv.type = ValueType(ValueType.SET)
        nv.set_ptr(sp.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return sp
    if val.type.value == ValueType.SET:
        return val.as_set().unsafe_bitcast[SlabHashMap]()
    return null_ptr[SlabHashMap, MutUntrackedOrigin]()


def _replay_list(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 key: GenericValue, create: Bool) -> Pointer[SlabList, MutUntrackedOrigin]:
    var val = keyspace[].get(key)
    if val.is_none():
        if not create:
            return null_ptr[SlabList, MutUntrackedOrigin]()
        var lp = alloc[SlabList](1)
        lp.unsafe_write(SlabList())
        var nv = GenericValue()
        nv.type = ValueType(ValueType.LIST)
        nv.set_ptr(lp.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return lp
    if val.type.value == ValueType.LIST:
        return val.as_list().unsafe_bitcast[SlabList]()
    return null_ptr[SlabList, MutUntrackedOrigin]()


def _replay_zset(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 key: GenericValue, create: Bool, geo: Bool) -> Pointer[SlabSkipList, MutUntrackedOrigin]:
    # A geo key is a sorted set (as in Redis); cmd-15 records from before
    # that, and old snapshots' geo values, load as one.
    var vtype = ValueType.ZSET
    var val = keyspace[].get(key)
    if val.is_none():
        if not create:
            return null_ptr[SlabSkipList, MutUntrackedOrigin]()
        var zp = alloc[SlabSkipList](1)
        zp.unsafe_write(SlabSkipList(16))
        var nv = GenericValue()
        nv.type = ValueType(vtype)
        nv.set_ptr(zp.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return zp
    if val.type.value == vtype:
        return val.as_zset().unsafe_bitcast[SlabSkipList]()
    return null_ptr[SlabSkipList, MutUntrackedOrigin]()


def _zset_remove_member(zp: Pointer[SlabSkipList, MutUntrackedOrigin],
                        member: GenericValue):
    """Collect-and-rebuild removal, same approach as handle_zrem."""
    var scores = List[Float64]()
    var objs = List[GenericValue]()
    var curr = zp[].head[].forward[0]
    while is_not_null(curr):
        scores.append(curr[].score)
        objs.append(curr[].obj)
        curr = curr[].forward[0]
    zp[].reset()
    for j in range(len(scores)):
        if not (objs[j] == member):
            zp[].insert(scores[j], objs[j])


def gv_bytes(gv: GenericValue, buf: Pointer[UInt8, MutUntrackedOrigin],
             mut length: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Raw bytes of a scalar GenericValue for serialization. STRING/SSO return
    the payload (SSO copied into buf); INT/FLOAT format into buf (needs ≥32B).
    Cold path only — String construction allocates."""
    if gv.is_string():
        length = gv.string_len()
        return gv.as_string_safe(buf)
    if gv.type.value == ValueType.INT:
        var s = String(gv.as_int())
        length = s.byte_length()
        unsafe_memcpy(dest=buf, src=s.unsafe_ptr(), count=length)
        return buf
    if gv.type.value == ValueType.FLOAT:
        var s = String(gv.as_float())
        length = s.byte_length()
        unsafe_memcpy(dest=buf, src=s.unsafe_ptr(), count=length)
        return buf
    length = 0
    return buf


def _owned_list_elems(lp: Pointer[SlabList, MutUntrackedOrigin]) -> List[GenericValue]:
    """Deep-copy a list's elements. Ziplist-mode lrange returns from_ptr_unsafe
    borrows into zip_buf — reset()+rpush would read the buffer being rewritten
    (the gh #170 replay corruption). from_ptr materializes owned copies."""
    var elems = lp[].get_all()
    var out = List[GenericValue]()
    var buf = alloc[UInt8](64)
    for j in range(len(elems)):
        var ml = 0
        var mp = gv_bytes(elems[j], buf, ml)
        out.append(GenericValue.from_ptr(mp, ml))
    buf.unsafe_free()
    return out^


@always_inline
def wal_is_aggregate(cmd_id: UInt8) -> Bool:
    """Record ids `wal_apply_aggregate` owns — one predicate for the WAL
    replayer and the snapshot loader, so a new record kind cannot reach one
    and be skipped by the other."""
    return (cmd_id >= 5 and cmd_id <= 24) or (cmd_id >= 27 and cmd_id <= 33)


def _replay_vset_field(vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int,
                       mut name_l: Int, mut rest_off: Int, mut rest_l: Int) -> Bool:
    """Split a [4B name_len][name][4B len][bytes] value. False = malformed."""
    if vl < 8:
        return False
    name_l = Int(_rec_u32(vp))
    if 8 + name_l > vl:
        return False
    rest_l = Int(_rec_u32(vp.unsafe_offset(4 + name_l)))
    rest_off = 8 + name_l
    return rest_off + rest_l <= vl


def _replay_vset(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                 key: GenericValue) -> Pointer[VectorSet, MutUntrackedOrigin]:
    """The VSET at `key`, or null when it is missing or another type."""
    var val = keyspace[].get(key)
    if val.is_none() or val.type.value != ValueType.VSET:
        return null_ptr[VectorSet, MutUntrackedOrigin]()
    return val.as_hash().unsafe_bitcast[VectorSet]()


@always_inline
def _read_varint(p: Pointer[UInt8, MutUntrackedOrigin], end: Int, mut off: Int) -> Int:
    """LEB128 u32 at p[off]; advances off. -1 = truncated or longer than 5 bytes."""
    var v = 0
    var shift = 0
    while off < end and shift <= 28:
        var b = Int(p[unsafe_offset=off])
        off += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v
        shift += 7
    return -1


def wal_apply_mset(vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int,
                   keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Int:
    """Apply a cmd-31 MSET record's pairs as SETs, in order; returns how many
    were applied. Shared by WAL replay, the snapshot loader (through
    wal_apply_aggregate) and replication, so the three cannot decode it
    differently. A malformed tail stops the walk; the pairs before it stand,
    as they would have from N separate SET records."""
    var off = 0
    var n = 0
    while off < vl:
        var kl = _read_varint(vp, vl, off)
        if kl < 0 or off + kl > vl:
            break
        var koff = off
        off += kl
        var l = _read_varint(vp, vl, off)
        if l < 0 or off + l > vl:
            break
        keyspace[].set(GenericValue.borrow(vp.unsafe_offset(koff), kl),
                       GenericValue.borrow(vp.unsafe_offset(off), l))
        off += l
        n += 1
    return n


def wal_apply_aggregate(cmd_id: UInt8,
                        kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                        vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int,
                        keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Bool:
    """Apply one gh #170 aggregate record (cmd_id 5..22). Returns True if applied.
    Semantics mirror the live execute path; malformed records are skipped."""
    if cmd_id == 31:     # MSET [varint kl][key][varint vl][val]...
        return wal_apply_mset(vp, vl, keyspace) > 0
    # gh #394: borrowed — the record outlives this call, and every use below is
    # a map call (which copies what it keeps). A from_ptr copy leaked once per
    # replayed or replicated record whose key already existed.
    var key = GenericValue.borrow(kp, kl)

    if cmd_id == 5:      # HSET [4B flen][field][4B vlen][value]
        if vl < 8:
            return False
        var flen = Int(_rec_u32(vp))
        if 8 + flen > vl:
            return False
        var vlen = Int(_rec_u32(vp.unsafe_offset(4).unsafe_offset(flen)))
        if 8 + flen + vlen > vl:
            return False
        var hp = _replay_hash(keyspace, key, True)
        if is_null(hp):
            return False
        hp[].set(GenericValue.borrow(vp.unsafe_offset(4), flen),
                 GenericValue.borrow(vp.unsafe_offset(8).unsafe_offset(flen), vlen))
        if Int(hp[].field_ttl) != 0:   # gh #392: as live HSET — an overwrite clears the TTL
            _ = hp[].clear_field_deadline(GenericValue.borrow(vp.unsafe_offset(4), flen))
        return True

    elif cmd_id == 6 or cmd_id == 7:   # LPUSH / RPUSH
        var lp = _replay_list(keyspace, key, True)
        if is_null(lp):
            return False
        if cmd_id == 6:
            lp[].lpush(GenericValue.from_ptr(vp, vl))
        else:
            lp[].rpush(GenericValue.from_ptr(vp, vl))
        return True

    elif cmd_id == 8:    # SADD
        var sp = _replay_set(keyspace, key, True)
        if is_null(sp):
            return False
        var member = GenericValue.borrow(vp, vl)
        if sp[].get(member).is_none():
            sp[].set(member, GenericValue.from_int(1))
        return True

    elif cmd_id == 9 or cmd_id == 15:  # ZADD / GEOADD [8B score][member]
        if vl < 8:
            return False
        var zp = _replay_zset(keyspace, key, True, cmd_id == 15)
        if is_null(zp):
            return False
        var score = vp.unsafe_bitcast[Float64]()[unsafe_offset=0]
        zp[].insert(score, GenericValue.from_ptr(vp.unsafe_offset(8), vl - 8))
        return True

    elif cmd_id == 10:   # HDEL
        var hp = _replay_hash(keyspace, key, False)
        if is_null(hp):
            return False
        _ = hp[].remove_generic(GenericValue.borrow(vp, vl))
        return True

    elif cmd_id == 32:   # gh #392: HFIELD EXPIREAT [4B flen][field][4B 8][8B deadline ns]
        if vl < 16:
            return False
        var flen = Int(_rec_u32(vp))
        if 8 + flen + 8 > vl:
            return False
        var hp = _replay_hash(keyspace, key, False)
        if is_null(hp):
            return False
        var field = GenericValue.borrow(vp.unsafe_offset(4), flen)
        if hp[].get(field).is_none():
            return False
        hp[].set_field_deadline(field, Int64(_rec_u64(vp.unsafe_offset(8).unsafe_offset(flen))))
        keyspace[].note_field_ttl(key)
        return True

    elif cmd_id == 33:   # gh #392: HFIELD PERSIST [field]
        var hp = _replay_hash(keyspace, key, False)
        if is_null(hp):
            return False
        _ = hp[].clear_field_deadline(GenericValue.borrow(vp, vl))
        return True

    elif cmd_id == 11:   # SREM
        var sp = _replay_set(keyspace, key, False)
        if is_null(sp):
            return False
        _ = sp[].remove_generic(GenericValue.borrow(vp, vl))
        return True

    elif cmd_id == 12:   # ZREM
        var zp = _replay_zset(keyspace, key, False, False)
        if is_null(zp):
            return False
        _zset_remove_member(zp, GenericValue.from_ptr(vp, vl))
        return True

    elif cmd_id == 13 or cmd_id == 14:  # LPOP / RPOP
        var lp = _replay_list(keyspace, key, False)
        if is_null(lp):
            return False
        if cmd_id == 13:
            _ = lp[].lpop()
        else:
            _ = lp[].rpop()
        return True

    elif cmd_id == 16:   # BITMAP image
        if vl <= 0:
            return False
        var bp = alloc[UInt8](vl)
        unsafe_memcpy(dest=bp, src=vp, count=vl)
        var nv = GenericValue()
        nv.type = ValueType(ValueType.BITMAP)
        nv._data0 = UInt64(Int(bp))
        nv._data1 = UInt64(vl)
        keyspace[].set(key, nv)
        return True

    elif cmd_id == 17:   # HLL image
        if vl != HLL_REGISTERS:
            return False
        var hll = alloc[UInt8](HLL_REGISTERS)
        unsafe_memcpy(dest=hll, src=vp, count=HLL_REGISTERS)
        var nv = GenericValue()
        nv.type = ValueType(ValueType.HLL)
        nv.set_ptr(hll.unsafe_bitcast[NoneType]())
        keyspace[].set(key, nv)
        return True

    elif cmd_id == 18:   # SETBIT [8B (offset<<1)|bit]
        if vl < 8:
            return False
        var packed = _rec_u64(vp)
        var offset = Int(packed >> 1)
        var bit = Int(packed & 1)
        var val = keyspace[].get(key)
        if val.is_none():
            var byte_len = offset // 8 + 1
            var bmp = alloc[UInt8](byte_len)
            unsafe_memset(bmp, 0, byte_len)
            var result = setbit(byte_len, bmp, offset, bit)
            var nv = GenericValue()
            nv.type = ValueType(ValueType.BITMAP)
            nv._data0 = UInt64(Int(result.ptr))
            nv._data1 = UInt64(result.len)
            keyspace[].set(key, nv)
            return True
        elif val.type.value == ValueType.BITMAP:
            var result = setbit(val.bitmap_len(), val.as_bitmap(), offset, bit)
            val._data0 = UInt64(Int(result.ptr))
            val._data1 = UInt64(result.len)
            keyspace[].set(key, val)
            return True
        elif val.is_string_like():
            # gh #232: SETBIT on a STRING is legal (a bitmap IS a string), and
            # the live path now accepts it. Replay MUST accept it too, or a
            # server that took the write refuses to reconstruct it — the
            # keyspace would differ from the log it was built from, which is a
            # worse failure than the WRONGTYPE this replaces.
            #
            # Copy out rather than mutate: `setbit` frees on growth and the
            # payload may be blob-arena-backed (gh #163).
            var _need = offset // 8 + 1
            var _blen = 0
            var _buf = val.owned_bitmap_copy(_need, _blen)
            var _res = setbit(_blen, _buf, offset, bit)
            # Do NOT free the original: `keyspace.set()` below already frees the
            # value it overwrites, arena-safely. Freeing here too is a double
            # free (see the fast-path note).
            var _nv = GenericValue()
            _nv.type = ValueType(ValueType.BITMAP)
            _nv._data0 = UInt64(Int(_res.ptr))
            _nv._data1 = UInt64(_res.len)
            keyspace[].set(key, _nv)
            return True
        return False

    elif cmd_id == 19:   # LSET [8B index][element]
        if vl < 8:
            return False
        var lp = _replay_list(keyspace, key, False)
        if is_null(lp):
            return False
        var idx = Int(_rec_u64(vp))
        var elems = _owned_list_elems(lp)
        if idx < 0 or idx >= len(elems):
            return False
        elems[idx].free_str_payload()
        elems[idx] = GenericValue.from_ptr(vp.unsafe_offset(8), vl - 8)
        lp[].reset()
        for j in range(len(elems)):
            lp[].rpush(elems[j])
        return True

    elif cmd_id == 20:   # LTRIM [8B start][8B stop] (resolved, inclusive)
        if vl < 16:
            return False
        var lp = _replay_list(keyspace, key, False)
        if is_null(lp):
            return False
        var start = Int(_rec_u64(vp))
        var stop = Int(_rec_u64(vp.unsafe_offset(8)))
        var elems = _owned_list_elems(lp)
        lp[].reset()
        for j in range(len(elems)):
            if j >= start and j <= stop:
                lp[].rpush(elems[j])
            else:
                elems[j].free_str_payload()
        return True

    elif cmd_id == 21:   # LREM [8B i64 count][element]
        if vl < 8:
            return False
        var lp = _replay_list(keyspace, key, False)
        if is_null(lp):
            return False
        var count = _rec_u64(vp).cast[DType.int64]()
        var target = GenericValue.from_ptr(vp.unsafe_offset(8), vl - 8)
        var elems = _owned_list_elems(lp)
        var keep = List[Bool]()
        for _ in range(len(elems)):
            keep.append(True)
        var budget = count if count > 0 else (-count if count < 0 else Int64(len(elems)))
        if count >= 0:
            for j in range(len(elems)):
                if budget == 0:
                    break
                if elems[j] == target:
                    keep[j] = False
                    budget -= 1
        else:
            for j in range(len(elems) - 1, -1, -1):
                if budget == 0:
                    break
                if elems[j] == target:
                    keep[j] = False
                    budget -= 1
        lp[].reset()
        for j in range(len(elems)):
            if keep[j]:
                lp[].rpush(elems[j])
            else:
                elems[j].free_str_payload()
        target.free_str_payload()
        return True

    elif cmd_id == 22:   # LINSERT [4B plen|BEFORE-bit31][pivot][4B elen][element]
        if vl < 8:
            return False
        var lp = _replay_list(keyspace, key, False)
        if is_null(lp):
            return False
        var p0 = _rec_u32(vp)
        var before = (p0 & 0x80000000) != 0
        var plen = Int(p0 & 0x7FFFFFFF)
        if 8 + plen > vl:
            return False
        var elen = Int(_rec_u32(vp.unsafe_offset(4).unsafe_offset(plen)))
        if 8 + plen + elen > vl:
            return False
        var pivot = GenericValue.from_ptr(vp.unsafe_offset(4), plen)
        var elem = GenericValue.from_ptr(vp.unsafe_offset(8).unsafe_offset(plen), elen)
        var elems = _owned_list_elems(lp)
        lp[].reset()
        var inserted = False
        for j in range(len(elems)):
            # Compare BEFORE the push: rpush takes ownership and, in ziplist
            # mode, frees the element once it is copied in (gh #394).
            var at_pivot = not inserted and elems[j] == pivot
            if at_pivot and before:
                lp[].rpush(elem)
                inserted = True
            lp[].rpush(elems[j])
            if at_pivot and not before:
                lp[].rpush(elem)
                inserted = True
        pivot.free_str_payload()
        if not inserted:
            elem.free_str_payload()
        return True

    # ── gh #174 ──────────────────────────────────────────────────────────
    elif cmd_id == 23:   # XADD [8B id_ms][8B id_seq][packed pairs]
        if vl < 16:
            return False
        var sd = _replay_stream(keyspace, key, True)
        if is_null(sd):
            return False
        var id_ms = _rec_u64(vp)
        var id_seq = _rec_u64(vp.unsafe_offset(8))
        var plen = vl - 16
        # StreamData.append takes ownership of the payload pointer (StreamEntry
        # frees it), so the record bytes must be copied out of the mmap — the
        # WAL mapping is unmapped after replay and the snapshot buffer is
        # reused per record.
        var pack = alloc[UInt8](plen if plen > 0 else 1)
        if plen > 0:
            unsafe_memcpy(dest=pack, src=vp.unsafe_offset(16), count=plen)
        # Recover num_fields by walking the packed pairs rather than trusting a
        # logged count: the walk is the same one the readers do, so a truncated
        # record yields a short entry instead of a reader running off the end.
        var nf = 0
        var w = 0
        while w + 2 <= plen:
            var fl = Int((pack.unsafe_offset(w)).unsafe_bitcast[UInt16]()[unsafe_offset=0]); w += 2 + fl
            if w + 2 > plen:
                break
            var vlen2 = Int((pack.unsafe_offset(w)).unsafe_bitcast[UInt16]()[unsafe_offset=0]); w += 2 + vlen2
            if w > plen:
                break
            nf += 1
        sd[].append(id_ms, id_seq, pack, plen, nf)
        return True

    elif cmd_id == 24:   # PFADD element
        var val = keyspace[].get(key)
        var regs: Pointer[UInt8, MutUntrackedOrigin]
        if val.is_none():
            regs = alloc[UInt8](HLL_REGISTERS)
            unsafe_memset(regs, 0, HLL_REGISTERS)
            var nv = GenericValue()
            nv.type = ValueType(ValueType.HLL)
            nv.set_ptr(regs.unsafe_bitcast[NoneType]())
            keyspace[].set(key, nv)
        elif val.type.value == ValueType.HLL:
            regs = val.as_hash().unsafe_bitcast[UInt8]()
        else:
            return False
        _ = hll_add(regs, GenericValue.from_ptr(vp, vl))
        return True

    elif cmd_id == 28:   # VADD [4B nl][name][4B plen][f32 norm][dim x f32 unit]
        var nl = 0
        var po = 0
        var pl = 0
        if not _replay_vset_field(vp, vl, nl, po, pl):
            return False
        if pl < 8 or (pl - 4) % 4 != 0:
            return False
        var dim = (pl - 4) // 4
        var vs = _replay_vset(keyspace, key)
        if is_null(vs):
            if not keyspace[].get(key).is_none():
                return False           # another type holds the key
            vs = alloc[VectorSet](1)
            vs.unsafe_write(VectorSet(dim))
            var nv = GenericValue()
            nv.type = ValueType(ValueType.VSET)
            nv.set_ptr(vs.unsafe_bitcast[NoneType]())
            keyspace[].set(key, nv)
        elif vs[].dim != dim:
            return False
        var norm = vp.unsafe_offset(po).unsafe_bitcast[Float32]()[unsafe_offset=0]
        vs[].add_stored(vp.unsafe_offset(4), nl,
                        vp.unsafe_offset(po + 4).unsafe_bitcast[Float32](), norm)
        return True

    elif cmd_id == 29:   # VREM element — the last element takes the key with it
        var vs = _replay_vset(keyspace, key)
        if is_null(vs):
            return False
        if vs[].remove(vp, vl) and vs[].live == 0:
            _ = keyspace[].remove_generic(key)
            free_vset(vs)
        return True

    elif cmd_id == 30:   # VSETATTR [4B nl][name][4B al][attr]
        var nl = 0
        var ao = 0
        var al = 0
        if not _replay_vset_field(vp, vl, nl, ao, al):
            return False
        var vs = _replay_vset(keyspace, key)
        if is_null(vs):
            return False
        var slot = vs[].find(vp.unsafe_offset(4), nl)
        if slot < 0:
            return False
        vs[].attrs[slot] = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=vp.unsafe_offset(ao), length=al)))
        return True

    elif cmd_id == 27:   # XDEL [8B id_ms][8B id_seq]
        if vl < 16:
            return False
        var sd = _replay_stream(keyspace, key, False)
        if is_null(sd):
            return False
        var d_ms = _rec_u64(vp)
        var d_seq = _rec_u64(vp.unsafe_offset(8))
        for ei in range(sd[].count):
            if sd[].entries[unsafe_offset=ei].id_ms == d_ms and sd[].entries[unsafe_offset=ei].id_seq == d_seq:
                if not sd[].entries[unsafe_offset=ei].deleted:
                    sd[].kill(ei)
                    sd[].compact()
                return True
        return True

    return False


def wal_apply_ttl(cmd_id: UInt8,
                  kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                  vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int,
                  ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Bool:
    """gh #174: apply a TTL record (cmd 25 EXPIREAT / 26 PERSIST).

    Split out from `wal_apply_aggregate` because it writes the ttl_map, not the
    keyspace, and several replay entry points legitimately have no ttl_map to
    hand (the snapshot loader used at FLUSHALL-rebuild time, for instance). A
    null ttl_map makes these records no-ops rather than a crash."""
    if is_null(ttl_map):
        return False
    var key = GenericValue.borrow(kp, kl)
    if cmd_id == 25:
        if vl < 8:
            return False
        ttl_map[].set(key, GenericValue.from_int(Int64(_rec_u64(vp))))
        return True
    elif cmd_id == 26:
        _ = ttl_map[].remove_generic(key)
        return True
    return False


struct WAL(Movable):
    var path: String
    var fd: Int32
    var map: Pointer[UInt8, MutUntrackedOrigin]
    var tail_offset: UInt64      # write position inside data section
    var seq: UInt64              # monotonic entry counter
    var dirty: Bool
    # kept for legacy callers
    var compaction_threshold: Int
    var current_size: Int
    # gh #149: segment rotation + drop accounting. All new fields sit AFTER the
    # pre-existing ones: current_size/dirty are written by every _commit, and
    # pushing them onto a different cache line than tail_offset (an earlier
    # version inserted this block above them) is a measurable layout change.
    var file_size: Int           # bytes per segment (--wal-size)
    var data_size: Int           # file_size - WAL_HEADER_SIZE
    var max_segments: Int        # sealed-segment ceiling (--wal-max-segments)
    var sealed: Int              # sealed segments currently on disk
    var dropped: UInt64          # entries refused because the log was full
    var dropped_bytes: UInt64
    var warned: Bool             # one-shot stderr warning latch
    # N3 replication streams raw bytes straight out of this mapping from a C
    # thread (PrimaryReplicator.setup passes wal.map + header). Rotating would
    # munmap it under that thread, so a pinned log never rotates.
    var pinned: Bool
    # gh #149 perf: high-water mark of the last sync() — msync covers only
    # [synced_offset, tail_offset), not the whole file (see sync()).
    var synced_offset: UInt64
    # gh #260. Both go at the END for the layout reason stated above: they are
    # read once per recv buffer, never written on the append path.
    #
    # `durability_lost` is STICKY. Once an append has been refused, every later
    # acknowledged mutation is a lie about durability, so the latch never clears
    # on its own — only a successful SAVE/BGSAVE checkpoint (which makes the log
    # droppable) or a restart resets it.
    var durability_lost: Bool
    # False restores the pre-gh #260 behaviour: keep ACKing writes that are no
    # longer being persisted. Only a cache-only deployment should ask for this.
    var refuse_when_full: Bool
    # gh #390: the primary replicator streaming this log, if any. It reads the
    # mapping from its own thread, so rotation and checkpoint must detach it
    # before the mapping or its offsets change, and reattach after. Last field
    # on purpose (new fields go at the END of hot structs).
    var repl_handle: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self, path: String, worker_id: Int = 0,
                 file_size: Int = WAL_FILE_SIZE,
                 max_segments: Int = WAL_MAX_SEGMENTS,
                 enabled: Bool = True):
        self.path = path
        self.file_size = file_size if file_size > WAL_HEADER_SIZE * 2 else WAL_FILE_SIZE
        self.data_size = self.file_size - WAL_HEADER_SIZE
        self.max_segments = max_segments if max_segments >= 0 else WAL_MAX_SEGMENTS
        self.sealed = 0
        self.dropped = 0
        self.dropped_bytes = 0
        self.warned = False
        self.pinned = False
        self.synced_offset = 0
        self.durability_lost = False
        self.refuse_when_full = True   # gh #260: safe default, see ServerConfig
        self.compaction_threshold = self.data_size
        self.current_size = 0
        self.dirty = False
        self.tail_offset = 0
        self.seq = 0
        self.map = null_ptr[UInt8, MutUntrackedOrigin]()
        self.repl_handle = null_ptr[NoneType, MutUntrackedOrigin]()

        # --no-wal (gh #394): no file, no mapping. Every append already returns
        # False on a null map, so this is the whole switch. It used to gate only
        # the FAST path's appends: slow-path writes still logged into a mapped
        # 256 MB file, so "--no-wal" servers grew RSS by every byte a slow-path
        # write logged — which the RSS soak tests then reported as leaks.
        if not enabled:
            self.fd = -1
            return

        # open / create WAL file via C helper (handles variadic open + mode)
        # IMPORTANT: use self.path.as_c_string_slice() — ensures null-terminated C string
        # (unsafe_ptr() does NOT guarantee null at len; use unsafe_cstr_ptr() for all path args)
        var fd = external_call["pion_wal_open", Int32](self.path.as_c_string_slice())
        self.fd = fd

        if fd < 0:
            print("WAL: failed to open " + self.path)
            return

        # extend to file_size (idempotent)
        var tr = external_call["pion_wal_ftruncate", Int32](fd, Int(self.file_size))
        if tr != 0:
            print("WAL: ftruncate failed for " + path)
            _ = external_call["close", Int32](fd)
            self.fd = -1
            return

        # mmap MAP_SHARED for zero-syscall writes
        var mp = external_call["pion_wal_mmap", Pointer[UInt8, MutUntrackedOrigin]](
            fd, Int(self.file_size))

        # MAP_FAILED = (void*)-1; null pointer means success check failed
        if is_null(mp):
            print("WAL: mmap returned null for " + path)
            _ = external_call["close", Int32](fd)
            self.fd = -1
            return

        self.map = mp

        # Check header magic
        var hdr = mp.unsafe_bitcast[UInt64]()
        if hdr[unsafe_offset=0] != WAL_MAGIC:
            # New or corrupt file — write fresh header
            hdr[unsafe_offset=0] = WAL_MAGIC
            hdr[unsafe_offset=1] = 0                   # tail_offset
            hdr[unsafe_offset=2] = UInt64(worker_id)
            hdr[unsafe_offset=3] = 0                   # seq
        else:
            # Existing WAL — pick up where we left off
            self.tail_offset = hdr[unsafe_offset=1]
            self.seq = hdr[unsafe_offset=3]

        self.current_size = Int(self.tail_offset)

        # gh #149: count sealed segments left by a previous run. They are numbered
        # contiguously from 1 and only ever removed all at once (checkpoint), so
        # probing until the first gap is exact.
        while self.sealed < self.max_segments:
            if not _file_exists(self._segment_path(self.sealed + 1)):
                break
            self.sealed += 1

    @always_inline
    def _segment_path(self, n: Int) -> String:
        return self.path + "." + String(n)

    def __moveinit__(out self, deinit take: Self):
        self.path = take.path^
        self.fd = take.fd
        self.map = take.map
        self.tail_offset = take.tail_offset
        self.seq = take.seq
        self.dirty = take.dirty
        self.file_size = take.file_size
        self.data_size = take.data_size
        self.max_segments = take.max_segments
        self.sealed = take.sealed
        self.dropped = take.dropped
        self.dropped_bytes = take.dropped_bytes
        self.warned = take.warned
        self.pinned = take.pinned
        self.synced_offset = take.synced_offset
        self.durability_lost = take.durability_lost
        self.refuse_when_full = take.refuse_when_full
        self.repl_handle = take.repl_handle
        self.compaction_threshold = take.compaction_threshold
        self.current_size = take.current_size

    # ── Entry append (hot path, zero syscalls) ─────────────────────────────

    @always_inline
    def append(mut self, cmd_id: UInt8,
              key_ptr: Pointer[UInt8, _], key_len: Int) -> Bool:
        """Append with no value (DEL, INCR without value). False = not logged."""
        if is_null(self.map):
            return False
        var entry_size = 13 + key_len   # 4+1+4+key+4
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(0))           # val_len = 0
        self._commit(entry_size)
        return True

    @always_inline
    def append_kv(mut self, cmd_id: UInt8,
                 key_ptr: Pointer[UInt8, _], key_len: Int,
                 val_ptr: Pointer[UInt8, _], val_len: Int) -> Bool:
        """Append with value (SET). False = not logged (see gh #149)."""
        if is_null(self.map):
            return False
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        if val_len > 0 and is_not_null(val_ptr):
            unsafe_memcpy(dest=p, src=val_ptr, count=val_len)
        self._commit(entry_size)
        return True

    @always_inline
    def append_kv_batched(mut self, cmd_id: UInt8,
                          key_ptr: Pointer[UInt8, _], key_len: Int,
                          val_ptr: Pointer[UInt8, _], val_len: Int) -> Bool:
        """append_kv without the per-record header publish (gh #229).

        For commands that write N records in one go. The caller MUST call
        `publish_header()` once afterwards. If the append had to rotate the
        segment, this publishes anyway — `_make_room` reopens the mapping, so
        deferring past it would stamp the header of the wrong segment."""
        if is_null(self.map):
            return False
        var entry_size = 13 + key_len + val_len
        var rotated = False
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
            rotated = True
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        if val_len > 0 and is_not_null(val_ptr):
            unsafe_memcpy(dest=p, src=val_ptr, count=val_len)
        if rotated:
            self._commit(entry_size)
        else:
            self._advance(entry_size)
        return True

    # ── Batched append: one bounds check + one accounting for N records ──────
    #
    # gh #230. append_kv_batched already hoisted the header publish out of a
    # multi-record command; what stayed per-record was the capacity compare and
    # `_advance`, which stores FOUR self fields. Those stores sit between two
    # `unsafe_memcpy`s into the same mmap, so the compiler cannot keep the tail
    # in a register across them — a 10-key MSET reloaded and restored the tail
    # cursor ten times to write 10 records it could have written in one run.
    #
    # Contract: batch_fits() first (it proves NO record in the batch can hit the
    # segment end, which is what makes the per-record check droppable), then
    # write_record_at() per record threading the returned offset, then
    # commit_batch() exactly once. A false from batch_fits is not an error —
    # the caller falls back to the per-record path, which rotates correctly.

    @always_inline
    def batch_fits(self, upper_bound: Int) -> Bool:
        """True when `upper_bound` bytes are guaranteed to fit without rotating."""
        if is_null(self.map):
            return False
        return self.tail_offset + UInt64(upper_bound) <= UInt64(self.data_size)

    @always_inline
    def write_record_at(self, off: Int, cmd_id: UInt8,
                        key_ptr: Pointer[UInt8, _], key_len: Int,
                        val_ptr: Pointer[UInt8, _], val_len: Int) -> Int:
        """Write one record at `off`; return the offset just past it.

        No bounds check and no accounting — see the contract above."""
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(off)
        var entry_size = 13 + key_len + val_len
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;            p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        if val_len > 0 and is_not_null(val_ptr):
            unsafe_memcpy(dest=p, src=val_ptr, count=val_len)
        return off + entry_size

    @always_inline
    def commit_batch(mut self, new_off: Int, records: Int):
        """Account for `records` written by write_record_at, then publish."""
        self.tail_offset = UInt64(new_off)
        self.seq += UInt64(records)
        self.current_size = new_off
        self.dirty = True
        self.publish_header()

    # cmd 31 (MSET) under the same batch contract: batch_fits() first, then
    # the pairs start at `rec_off + 13`, mset_pair_at() per pair threading the
    # offset, mset_seal() to write the header, then commit_batch(end, 1).
    # The header goes last because entry_size is known only then; nothing
    # reads the record before commit_batch publishes the tail.

    @always_inline
    def _put_varint(self, p: Pointer[UInt8, MutUntrackedOrigin], v: Int) -> Int:
        """LEB128 at p; returns bytes written. One byte below 128."""
        if v < 0x80:
            p[unsafe_offset=0] = UInt8(v)
            return 1
        var x = v
        var i = 0
        while x >= 0x80:
            p[unsafe_offset=i] = UInt8((x & 0x7F) | 0x80)
            x >>= 7
            i += 1
        p[unsafe_offset=i] = UInt8(x)
        return i + 1

    @always_inline
    def mset_pair_at(self, off: Int,
                     key_ptr: Pointer[UInt8, _], key_len: Int,
                     val_ptr: Pointer[UInt8, _], val_len: Int) -> Int:
        """Write one [varint kl][key][varint vl][val] pair at `off`; return the
        offset past it. No bounds check: batch_fits() proved the room."""
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(off)
        var q = self._put_varint(p, key_len)
        unsafe_memcpy(dest=p.unsafe_offset(q), src=key_ptr, count=key_len)
        q += key_len
        q += self._put_varint(p.unsafe_offset(q), val_len)
        if val_len > 0:
            unsafe_memcpy(dest=p.unsafe_offset(q), src=val_ptr, count=val_len)
        return off + q + val_len

    @always_inline
    def mset_seal(self, rec_off: Int, end_off: Int):
        """Write the cmd-31 header for pairs occupying [rec_off+13, end_off)."""
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(rec_off)
        var entry_size = end_off - rec_off
        self._write_u32(p, UInt32(entry_size))
        p[unsafe_offset=4] = UInt8(31)
        self._write_u32(p.unsafe_offset(5), UInt32(0))
        self._write_u32(p.unsafe_offset(9), UInt32(entry_size - 13))

    @always_inline
    def append_blob_ref(mut self, key_ptr: Pointer[UInt8, _], key_len: Int,
                        seg: Int, off: Int, length: Int) -> Bool:
        """gh #163: log a SET whose payload went to the blob arena. 24-byte value."""
        if is_null(self.map):
            return False
        var entry_size = 13 + key_len + 24
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = UInt8(4);                        p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(24));         p = p.unsafe_offset(4)
        var q = p.unsafe_bitcast[UInt64]()
        q[unsafe_offset=0] = UInt64(seg)
        q[unsafe_offset=1] = UInt64(off)
        q[unsafe_offset=2] = UInt64(length)
        self._commit(entry_size)
        return True

    # ── gh #170 aggregate effect appenders ─────────────────────────────────

    @always_inline
    def append_field_deadline(mut self, key_ptr: Pointer[UInt8, _], key_len: Int,
                              f_ptr: Pointer[UInt8, _], f_len: Int, deadline_ns: Int64) -> Bool:
        """gh #392: cmd 32 — a hash field's absolute deadline (unix ns). The
        8 bytes go through a heap scratch: a stack buffer crossing into the
        out-of-line append is the gh #349 hazard."""
        var d = alloc[UInt8](8)
        d.unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(deadline_ns)
        var ok = self.append_field_kv(32, key_ptr, key_len, f_ptr, f_len, d, 8)
        d.unsafe_free()
        return ok

    def append_field_kv(mut self, cmd_id: UInt8,
                        key_ptr: Pointer[UInt8, _], key_len: Int,
                        f_ptr: Pointer[UInt8, _], f_len: Int,
                        v_ptr: Pointer[UInt8, _], v_len: Int) -> Bool:
        """Two-part value record: val = [4B f_len][f][4B v_len][v] (HSET, LINSERT pivot+elem)."""
        if is_null(self.map):
            return False
        var val_len = 8 + f_len + v_len
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        self._write_u32(p, UInt32(f_len));      p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=f_ptr, count=f_len); p = p.unsafe_offset(f_len)
        self._write_u32(p, UInt32(v_len));      p = p.unsafe_offset(4)
        if v_len > 0:
            unsafe_memcpy(dest=p, src=v_ptr, count=v_len)
        self._commit(entry_size)
        return True

    @always_inline
    def append_scored(mut self, cmd_id: UInt8,
                      key_ptr: Pointer[UInt8, _], key_len: Int,
                      score: Float64,
                      m_ptr: Pointer[UInt8, _], m_len: Int) -> Bool:
        """Scored-member record: val = [8B f64 score][member] (ZADD, GEOADD)."""
        if is_null(self.map):
            return False
        var val_len = 8 + m_len
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        p.unsafe_bitcast[Float64]()[unsafe_offset=0] = score;        p = p.unsafe_offset(8)
        if m_len > 0:
            unsafe_memcpy(dest=p, src=m_ptr, count=m_len)
        self._commit(entry_size)
        return True

    @always_inline
    def append_u64_val(mut self, cmd_id: UInt8,
                       key_ptr: Pointer[UInt8, _], key_len: Int,
                       n: UInt64,
                       v_ptr: Pointer[UInt8, _], v_len: Int) -> Bool:
        """u64-prefixed record: val = [8B n][bytes] (LSET index, LREM count, SETBIT offset+bit)."""
        if is_null(self.map):
            return False
        var val_len = 8 + v_len
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        p.unsafe_bitcast[UInt64]()[unsafe_offset=0] = n;             p = p.unsafe_offset(8)
        if v_len > 0:
            unsafe_memcpy(dest=p, src=v_ptr, count=v_len)
        self._commit(entry_size)
        return True

    @always_inline
    def append_u64x2_val(mut self, cmd_id: UInt8,
                         key_ptr: Pointer[UInt8, _], key_len: Int,
                         a: UInt64, b: UInt64,
                         v_ptr: Pointer[UInt8, _], v_len: Int) -> Bool:
        """gh #174: two-u64-prefixed record: val = [8B a][8B b][bytes].
        Serves XADD ([id_ms][id_seq][packed pairs]) and XDEL ([id_ms][id_seq])."""
        if is_null(self.map):
            return False
        var val_len = 16 + v_len
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        p.unsafe_bitcast[UInt64]()[unsafe_offset=0] = a;             p = p.unsafe_offset(8)
        p.unsafe_bitcast[UInt64]()[unsafe_offset=0] = b;             p = p.unsafe_offset(8)
        if v_len > 0:
            unsafe_memcpy(dest=p, src=v_ptr, count=v_len)
        self._commit(entry_size)
        return True

    def append_linsert(mut self, key_ptr: Pointer[UInt8, _], key_len: Int,
                       before: Bool,
                       p_ptr: Pointer[UInt8, _], p_len: Int,
                       v_ptr: Pointer[UInt8, _], v_len: Int) -> Bool:
        """cmd 22: val = [4B p_len | BEFORE<<31][pivot][4B v_len][element]. Cold path."""
        if is_null(self.map):
            return False
        var val_len = 8 + p_len + v_len
        var entry_size = 13 + key_len + val_len
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = UInt8(22);                       p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(val_len));    p = p.unsafe_offset(4)
        var pl = UInt32(p_len)
        if before:
            pl |= 0x80000000
        self._write_u32(p, pl);                 p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=p_ptr, count=p_len); p = p.unsafe_offset(p_len)
        self._write_u32(p, UInt32(v_len));      p = p.unsafe_offset(4)
        if v_len > 0:
            unsafe_memcpy(dest=p, src=v_ptr, count=v_len)
        self._commit(entry_size)
        return True

    @always_inline
    def append_u64x2(mut self, cmd_id: UInt8,
                     key_ptr: Pointer[UInt8, _], key_len: Int,
                     a: UInt64, b: UInt64) -> Bool:
        """Two-u64 record: val = [8B a][8B b] (LTRIM resolved range)."""
        if is_null(self.map):
            return False
        var entry_size = 13 + key_len + 16
        if self.tail_offset + UInt64(entry_size) > UInt64(self.data_size):
            if not self._make_room(entry_size):
                return False
        var p = self.map.unsafe_offset(WAL_HEADER_SIZE).unsafe_offset(Int(self.tail_offset))
        self._write_u32(p, UInt32(entry_size)); p = p.unsafe_offset(4)
        p[unsafe_offset=0] = cmd_id;                          p = p.unsafe_offset(1)
        self._write_u32(p, UInt32(key_len));    p = p.unsafe_offset(4)
        unsafe_memcpy(dest=p, src=key_ptr, count=key_len); p = p.unsafe_offset(key_len)
        self._write_u32(p, UInt32(16));         p = p.unsafe_offset(4)
        var q = p.unsafe_bitcast[UInt64]()
        q[unsafe_offset=0] = a
        q[unsafe_offset=1] = b
        self._commit(entry_size)
        return True

    @no_inline
    def append_vset_image(mut self, kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                          vs: Pointer[VectorSet, MutUntrackedOrigin]):
        """gh #378: a whole vector set as VADD (+ VSETATTR) records, live slots
        in slot order so VSIM's tie order survives the rewrite."""
        var payload = alloc[UInt8](vs[].payload_len())
        for s in range(vs[].n):
            if vs[].alive[unsafe_offset=s] == 0:
                continue
            var pl = vs[].stored_payload(s, payload)
            var nm = vs[].names[s].copy()
            _ = self.append_field_kv(28, kp, kl, nm.unsafe_ptr(), nm.byte_length(), payload, pl)
            var at = vs[].attrs[s].copy()
            if at.byte_length() > 0:
                _ = self.append_field_kv(30, kp, kl, nm.unsafe_ptr(), nm.byte_length(),
                                         at.unsafe_ptr(), at.byte_length())
        payload.unsafe_free()

    # ── Segment rotation (gh #149) ─────────────────────────────────────────

    @no_inline
    def _make_room(mut self, entry_size: Int) -> Bool:
        """Cold path shared by the appenders: the active segment cannot fit this
        entry. Outlined so each inlined append keeps the pre-gh #149 hot-path
        shape — one compare and a predicted-not-taken branch."""
        if entry_size > self.data_size:
            # Single entry larger than a whole segment: rotating cannot help.
            # Values this large belong in the blob tier (gh #163).
            self._note_drop(entry_size)
            return False
        if not self._rotate():
            self._note_drop(entry_size)
            return False
        return True

    def _repl_detach(self):
        """gh #390: stop the replicator reading this mapping — it is about to
        be unmapped (rotation) or its offsets reset (checkpoint). Every replica
        is dropped and the repl_id changes, so each reconnects into a
        FULLRESYNC instead of reading from offsets the log no longer has."""
        if is_not_null(self.repl_handle):
            external_call["pion_repl_primary_detach_wal", NoneType](
                self.repl_handle, null_ptr[UInt8, MutUntrackedOrigin]())

    def _repl_attach(self):
        if is_not_null(self.repl_handle) and is_not_null(self.map):
            external_call["pion_repl_primary_attach_wal", NoneType](
                self.repl_handle, self.map.unsafe_offset(WAL_HEADER_SIZE),
                self.map.unsafe_bitcast[UInt64]().unsafe_offset(1))

    def _rotate(mut self) -> Bool:
        """Seal the active file as `{path}.{n}` and start a fresh one.

        Returns False when the sealed-segment ceiling is reached — the caller
        must then treat the write as unlogged and say so. Cold path: this runs
        once per `file_size` bytes of log, not per append."""
        if is_null(self.map) or self.fd < 0:
            return False
        if self.pinned:
            # Replication holds a raw pointer into this mapping; sealing it would
            # be a use-after-munmap in the streaming thread. Fall through to the
            # loud drop instead — visible in INFO, unlike the old silent return.
            return False
        if self.sealed >= self.max_segments:
            return False

        # gh #250: stamp the tail into THIS segment's header before sealing it.
        # `_replay_segment` bounds its walk by the sealed file's header word 1,
        # and since gh #229 the publish happens once per COMMAND, not per
        # record — so a multi-record command that rotated mid-way sealed the
        # segment at the end of the PREVIOUS command and orphaned everything it
        # had already written. The records were in the file, the client had its
        # +OK, and replay could never reach them: permanent loss of acked data,
        # with `wal_dropped_entries` still reading 0.
        #
        # `append_kv_batched` appears to cover this (it `_commit`s when it
        # rotated) but that runs after `_make_room` has reopened the mapping, so
        # it stamps the NEW segment. The old one is already sealed by then.
        #
        # This must be the last write to the old mapping, and it must precede
        # the msync below so the flush covers the header page too.
        self.publish_header()
        self._repl_detach()   # gh #390: before the munmap below

        # Schedule writeback and unmap the active file before renaming it.
        # MS_ASYNC, deliberately: the pages are file-backed the moment they were
        # written, so process-crash durability needs no flush at all, and an
        # MS_SYNC here would write back up to a whole segment synchronously
        # inside the event loop. OS-crash durability was never stronger than
        # the per-tick MS_ASYNC group commit, so sealing does not weaken it.
        _ = external_call["pion_wal_msync", Int32](
            self.map, Int(WAL_HEADER_SIZE + Int(self.tail_offset)), MS_ASYNC)
        _ = external_call["pion_wal_munmap", Int32](self.map, Int(self.file_size))
        _ = external_call["close", Int32](self.fd)
        self.map = null_ptr[UInt8, MutUntrackedOrigin]()
        self.fd = Int32(-1)

        var sealed_path = self._segment_path(self.sealed + 1)
        var rc = external_call["rename", Int32](
            self.path.as_c_string_slice(), sealed_path.as_c_string_slice())
        if rc != 0:
            # Rename failed (disk full / permissions). Re-open the file we just
            # closed so the log keeps working, and report the drop.
            _ = self._reopen_active(Int(self.tail_offset), self.seq)
            self._repl_attach()
            return False
        self.sealed += 1

        # Fresh active file, empty, carrying the running seq forward.
        if not self._reopen_active(0, self.seq):
            return False
        self._repl_attach()
        print("WAL: sealed " + sealed_path + " (" + String(self.sealed)
              + "/" + String(self.max_segments) + " segments)")
        return True

    def _reopen_active(mut self, tail: Int, seq: UInt64) -> Bool:
        """(Re)create the active segment at self.path and map it."""
        var fd = external_call["pion_wal_open", Int32](self.path.as_c_string_slice())
        if fd < 0:
            return False
        var tr = external_call["pion_wal_ftruncate", Int32](fd, Int(self.file_size))
        if tr != 0:
            _ = external_call["close", Int32](fd)
            return False
        var mp = external_call["pion_wal_mmap", Pointer[UInt8, MutUntrackedOrigin]](
            fd, Int(self.file_size))
        if is_null(mp):
            _ = external_call["close", Int32](fd)
            return False
        self.fd = fd
        self.map = mp
        self.tail_offset = UInt64(tail)
        self.synced_offset = UInt64(tail)
        self.seq = seq
        self.current_size = tail
        var hdr = mp.unsafe_bitcast[UInt64]()
        hdr[unsafe_offset=0] = WAL_MAGIC
        hdr[unsafe_offset=1] = UInt64(tail)
        hdr[unsafe_offset=3] = seq
        self.dirty = True
        return True

    @always_inline
    def refusing_writes(self) -> Bool:
        """gh #260: should a keyspace mutation be refused right now?

        Read once per recv buffer on the dispatch path, never per append, so
        this stays one load of an already-hot cache line plus a predicted-
        not-taken branch. Both halves matter: `durability_lost` is false in all
        normal operation, and `refuse_when_full` lets a cache-only deployment
        opt back into the old lossy behaviour explicitly."""
        return self.durability_lost and self.refuse_when_full

    def _note_drop(mut self, entry_size: Int):
        """gh #149: an acknowledged write that did not reach the log. Count it and
        say so once — the old code returned here silently, which is how 4.6 GB of
        SET blobs came back as nil after a restart."""
        self.dropped += 1
        self.dropped_bytes += UInt64(entry_size)
        # gh #260: latch BEFORE the warning, so the very first drop already
        # closes the door. Under `refuse_when_full` this write is the last one
        # that gets acknowledged without being persisted — everything after it
        # is refused rather than silently lost.
        self.durability_lost = True
        if not self.warned:
            self.warned = True
            var why = " (log pinned by replication — rotation disabled)" if self.pinned else ""
            print("WAL: FULL" + why + " — " + self.path + " has "
                  + String(self.sealed) + "/" + String(self.max_segments)
                  + " sealed segments; writes are NO LONGER DURABLE."
                  + " Run SAVE/BGSAVE to checkpoint, raise --wal-max-segments,"
                  + " or route large values through the blob tier.")

    @always_inline
    def _commit(mut self, entry_size: Int):
        self._advance(entry_size)
        self.publish_header()

    @always_inline
    def _advance(mut self, entry_size: Int):
        """Account for a written record WITHOUT republishing the header."""
        self.tail_offset += UInt64(entry_size)
        self.seq += 1
        self.current_size = Int(self.tail_offset)
        self.dirty = True

    @always_inline
    def publish_header(mut self):
        """Make every record up to `tail_offset` visible to recovery.

        Split out of `_commit` for multi-record commands (gh #217/#229). The
        header lives at offset 0 while records stream at the tail, so a 10-key
        MSET that committed per-key dirtied that one cache line ten times in
        addition to the tail writes — group-commit work paid per key. Publishing
        once per COMMAND is also better crash semantics: recovery then sees all
        of an MSET or none of it, instead of a prefix.

        Callers using _advance MUST call this before returning to the event
        loop, or the records they wrote are invisible to recovery."""
        if is_null(self.map):
            return
        var hdr = self.map.unsafe_bitcast[UInt64]()
        hdr[unsafe_offset=1] = self.tail_offset
        hdr[unsafe_offset=3] = self.seq

    @always_inline
    def _write_u32(self, p: Pointer[UInt8, MutUntrackedOrigin], v: UInt32):
        # gh #131 §3.3: one (unaligned) 32-bit store instead of 4 scalar byte
        # stores. LE-native, matching _read_u32; mirrors the v_store bitcast idiom
        # which already stores u32 at unaligned mmap offsets on ARM/x86 (both LE).
        p.unsafe_bitcast[UInt32]()[unsafe_offset=0] = v

    @always_inline
    def _read_u32(self, p: Pointer[UInt8, MutUntrackedOrigin]) -> UInt32:
        return UInt32(p[unsafe_offset=0]) | (UInt32(p[unsafe_offset=1]) << 8) | \
               (UInt32(p[unsafe_offset=2]) << 16) | (UInt32(p[unsafe_offset=3]) << 24)

    # ── Group commit (one call per event-loop tick) ─────────────────────────

    @always_inline
    def sync(mut self):
        """Schedule async flush to OS — non-blocking.

        gh #149 perf: covers only the pages dirtied since the previous call,
        not [0, tail). The whole-range form was a page walk that grew with the
        log — up to 16K pages per call as a segment approached 256 MB — paid
        every 64 ticks for as long as appends kept landing. The delta form is
        O(pages written since last tick). The header page is re-dirtied by
        every _commit (hdr[1]/hdr[3]), so it is scheduled separately once the
        delta window has moved past it."""
        if not self.dirty or is_null(self.map):
            return
        var end = WAL_HEADER_SIZE + Int(self.tail_offset)
        var start = (WAL_HEADER_SIZE + Int(self.synced_offset)) \
                    & ~(WAL_SYNC_PAGE - 1)
        if end > start:
            _ = external_call["pion_wal_msync", Int32](
                self.map.unsafe_offset(start), end - start, MS_ASYNC)
        if start > 0:
            _ = external_call["pion_wal_msync", Int32](
                self.map, WAL_SYNC_PAGE, MS_ASYNC)
        self.synced_offset = self.tail_offset
        self.dirty = False

    def sync_durable(mut self):
        """gh #259: flush and WAIT. Called once, on the graceful-shutdown path.

        `sync()` schedules MS_ASYNC once per 64 ticks, which is the right
        trade-off while serving — but it is only a *request* to the VM system,
        so a process that exits promptly afterwards can still lose the last
        tick's acknowledged writes. That is precisely what a routine `kill`
        did before this existed.

        Unlike `sync()` this covers [0, tail) rather than the delta window: the
        delta is bounded by `synced_offset`, and MS_ASYNC pages scheduled by an
        earlier tick may not have reached disk yet, so re-syncing only what is
        newly dirty would leave them in flight. Cost is a full page walk, paid
        exactly once per process lifetime.

        MS_SYNC is deliberate and is the whole point — the blocking is the
        feature. The caller bounds it: gh #259's SIGALRM grace period kills the
        process if the drain overruns."""
        if is_null(self.map):
            return
        var end = WAL_HEADER_SIZE + Int(self.tail_offset)
        if end > 0:
            _ = external_call["pion_wal_msync", Int32](self.map, end, MS_SYNC)
        self.synced_offset = self.tail_offset
        self.dirty = False

    def barrier(mut self) -> Bool:
        """Everything appended so far is on stable storage when this returns.

        KV.PREFIX.COMMIT's half of the barrier (the V-store WAL is the other).
        MS_SYNC writes the mapped pages back and waits; pion_fdatasync then
        flushes the drive's own cache (F_FULLFSYNC on macOS), which MS_SYNC
        does not, so the data also survives a power cut or a kernel panic. The
        caller blocks for it — that is the point; it is called at the pace a
        client asks for durability, not per write."""
        if is_null(self.map) or self.fd < 0:
            return False
        var end = WAL_HEADER_SIZE + Int(self.tail_offset)
        var ok = True
        if end > 0:
            ok = external_call["pion_wal_msync", Int32](self.map, end, MS_SYNC) == 0
        ok = (external_call["pion_fdatasync", Int32](self.fd) == 0) and ok
        self.synced_offset = self.tail_offset
        self.dirty = False
        return ok

    # ── Startup recovery ────────────────────────────────────────────────────

    def recover(mut self, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                blobs: Pointer[BlobStore, MutUntrackedOrigin] =
                    null_ptr[BlobStore, MutUntrackedOrigin](),
                ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] =
                    null_ptr[SlabHashMap, MutUntrackedOrigin]()) -> Int:
        """Replay the log into keyspace on startup: sealed segments oldest-first,
        then the active file. Returns number of entries applied."""
        var replayed = 0

        # gh #149: sealed segments carry everything written before the active file.
        # Replaying them in ascending order preserves write order, so a later
        # overwrite of a key still wins.
        for n in range(1, self.sealed + 1):
            replayed += self._replay_segment(self._segment_path(n), keyspace, blobs, ttl_map)

        replayed += self._replay_map(self.map, self.tail_offset, keyspace, blobs, ttl_map)

        if replayed > 0:
            var seg_note = ""
            if self.sealed > 0:
                seg_note = " (" + String(self.sealed) + " sealed segment(s) + active)"
            print("WAL: recovered " + String(replayed) + " entries from "
                  + self.path + seg_note)
        return replayed

    def _replay_segment(mut self,
                        seg_path: String,
                        keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                        blobs: Pointer[BlobStore, MutUntrackedOrigin],
                        ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
        """Map one sealed segment read-write, replay it, unmap. Sealed segments are
        never written again; the mapping exists only for the replay walk."""
        var sp = seg_path
        var fd = external_call["pion_wal_open", Int32](sp.as_c_string_slice())
        if fd < 0:
            return 0
        var mp = external_call["pion_wal_mmap", Pointer[UInt8, MutUntrackedOrigin]](
            fd, Int(self.file_size))
        if is_null(mp):
            _ = external_call["close", Int32](fd)
            return 0
        var hdr = mp.unsafe_bitcast[UInt64]()
        var applied = 0
        if hdr[unsafe_offset=0] == WAL_MAGIC:
            applied = self._replay_map(mp, hdr[unsafe_offset=1], keyspace, blobs, ttl_map)
        _ = external_call["pion_wal_munmap", Int32](mp, Int(self.file_size))
        _ = external_call["close", Int32](fd)
        return applied

    def _replay_map(mut self,
                    map: Pointer[UInt8, MutUntrackedOrigin],
                    tail: UInt64,
                    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                    blobs: Pointer[BlobStore, MutUntrackedOrigin],
                    ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
        """Replay entries out of one mapped segment. Shared by the sealed and active
        paths so the two can never drift."""
        if is_null(map) or tail == 0:
            return 0

        var replayed = 0
        var offset = UInt64(0)
        var data = map.unsafe_offset(WAL_HEADER_SIZE)

        while offset + 13 <= tail:   # min entry = 4+1+4+0+4+0 = 13 bytes
            var p = data.unsafe_offset(Int(offset))
            var elen = self._read_u32(p)
            if elen < 13 or offset + UInt64(elen) > tail:
                break  # truncated / corrupt

            var cmd_id = p[unsafe_offset=4]
            var kl     = self._read_u32(p.unsafe_offset(5))

            var key_off = Int(offset) + 9
            var vl_off  = key_off + Int(kl)
            if vl_off + 4 > Int(tail):
                break

            var vl     = self._read_u32(data.unsafe_offset(vl_off))
            var val_off = vl_off + 4
            if val_off + Int(vl) > Int(tail):
                break

            if cmd_id == 1:   # SET
                # gh #394: the key borrows the mapped log (set() copies it only
                # if it inserts); a from_ptr copy leaked on every overwrite.
                keyspace[].set(GenericValue.borrow(data.unsafe_offset(key_off), Int(kl)),
                               GenericValue.from_ptr(data.unsafe_offset(val_off), Int(vl)))
                replayed += 1
            elif cmd_id == 2:  # DEL
                _ = remove_and_free(keyspace, GenericValue.borrow(data.unsafe_offset(key_off), Int(kl)))
                replayed += 1
            elif cmd_id == 4 and vl == 24 and is_not_null(blobs):
                # gh #163: pointer record into the blob arena. ptr_at bounds-checks
                # against the mapped segment, so a stale or truncated record drops
                # the key instead of handing out a wild pointer.
                var q = (data.unsafe_offset(val_off)).unsafe_bitcast[UInt64]()
                var bp = blobs[].ptr_at(Int(q[unsafe_offset=0]), Int(q[unsafe_offset=1]), Int(q[unsafe_offset=2]))
                if is_not_null(bp):
                    var key_val = GenericValue.from_ptr(data.unsafe_offset(key_off), Int(kl))
                    keyspace[].set(key_val, GenericValue.from_blob_ptr(bp, Int(q[unsafe_offset=2])))
                    replayed += 1
            elif cmd_id == 25 or cmd_id == 26:
                # gh #174: TTL record — writes ttl_map, not the keyspace.
                if wal_apply_ttl(cmd_id, data.unsafe_offset(key_off), Int(kl),
                                 data.unsafe_offset(val_off), Int(vl), ttl_map):
                    replayed += 1
            elif wal_is_aggregate(cmd_id):
                # gh #170 aggregate effect record, extended by gh #174 (23 XADD,
                # 24 PFADD, 27 XDEL) and gh #378 (28-30, vector sets).
                if wal_apply_aggregate(cmd_id, data.unsafe_offset(key_off), Int(kl),
                                       data.unsafe_offset(val_off), Int(vl), keyspace):
                    replayed += 1

            offset += UInt64(elen)

        return replayed

    # ── Checkpoint (reset after snapshot) ───────────────────────────────────

    def checkpoint(mut self):
        """Truncate WAL after a successful snapshot — safe to replay from scratch next start.

        gh #149: sealed segments describe state the snapshot now contains, so they
        are unlinked here. This is the one place disk goes back down, which is why
        a full log tells the operator to SAVE."""
        if is_null(self.map):
            return
        self._repl_detach()   # gh #390: every stream offset is about to reset
        for n in range(1, self.sealed + 1):
            var seg = self._segment_path(n)
            _ = external_call["unlink", Int32](seg.as_c_string_slice())
        self.sealed = 0
        self.dropped = 0
        self.dropped_bytes = 0
        self.warned = False
        # gh #260: the log is empty again, so the latch clears. This is the ONLY
        # place it does — reset() runs behind a successful SAVE/BGSAVE, which is
        # exactly the condition that makes the refused writes durable again.
        self.durability_lost = False
        self.tail_offset = 0
        self.synced_offset = 0
        self.seq = 0
        self.current_size = 0
        var hdr = self.map.unsafe_bitcast[UInt64]()
        hdr[unsafe_offset=1] = 0
        hdr[unsafe_offset=3] = 0
        _ = external_call["pion_wal_msync", Int32](
            self.map, WAL_HEADER_SIZE, MS_ASYNC)
        self._repl_attach()

    def compact(mut self):
        self.checkpoint()

    def append_key_image(mut self, kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                         val: GenericValue):
        """Log `val` — a key's WHOLE current value — in the record forms replay
        rebuilds it from (the same forms the snapshot serializer writes). The
        caller logs the DEL first, so replay is DEL-then-rebuild: idempotent,
        whatever records the key had before.

        One serializer for BGREWRITEAOF and for log_key_image. The rewrite's
        own copy had no STREAM branch, so BGREWRITEAOF dropped every stream."""
        var vbuf = alloc[UInt8](64)   # scalar serialization scratch (gv_bytes)
        var fbuf = alloc[UInt8](64)
        var t = Int(val.type.value)
        if val.is_string():
            var vl = val.string_len()
            var vp = val.as_string_safe(vbuf)
            _ = self.append_kv(1, kp, kl, vp, vl)
        elif t == ValueType.INT or t == ValueType.FLOAT:
            var vl = 0
            var vp = gv_bytes(val, vbuf, vl)
            _ = self.append_kv(1, kp, kl, vp, vl)
        elif t == ValueType.HASH:
            var hp = val.as_hash().unsafe_bitcast[SlabHashMap]()
            for j in range(hp[].capacity):
                var hm = hp[].metadata[unsafe_offset=j]
                if hm == SlabHashMap.EMPTY or hm == SlabHashMap.DELETED:
                    continue
                var fl = 0
                var fp = gv_bytes(hp[].keys[unsafe_offset=j], fbuf, fl)
                var vl = 0
                var vp = gv_bytes(hp[].values[unsafe_offset=j], vbuf, vl)
                _ = self.append_field_kv(5, kp, kl, fp, fl, vp, vl)
            # gh #392: the field TTLs travel with the image (RENAME, COPY,
            # BGREWRITEAOF, …), in the same cmd-32 form the snapshot writes.
            if Int(hp[].field_ttl) != 0:
                var ft = hp[].field_ttl
                for j in range(ft[].capacity):
                    var tm = ft[].metadata[unsafe_offset=j]
                    if tm == SlabHashMap.EMPTY or tm == SlabHashMap.DELETED:
                        continue
                    var fl2 = 0
                    var fp2 = gv_bytes(ft[].keys[unsafe_offset=j], fbuf, fl2)
                    _ = self.append_field_deadline(kp, kl, fp2, fl2, ft[].values[unsafe_offset=j].as_int())
        elif t == ValueType.LIST:
            var lp = val.as_list().unsafe_bitcast[SlabList]()
            var elems = lp[].get_all()
            for j in range(len(elems)):
                var vl = 0
                var vp = gv_bytes(elems[j], vbuf, vl)
                _ = self.append_kv(7, kp, kl, vp, vl)   # RPUSH in order
        elif t == ValueType.SET:
            var sp = val.as_set().unsafe_bitcast[SlabHashMap]()
            for j in range(sp[].capacity):
                var sm = sp[].metadata[unsafe_offset=j]
                if sm == SlabHashMap.EMPTY or sm == SlabHashMap.DELETED:
                    continue
                var vl = 0
                var vp = gv_bytes(sp[].keys[unsafe_offset=j], vbuf, vl)
                _ = self.append_kv(8, kp, kl, vp, vl)
        elif t == ValueType.ZSET or t == ValueType.GEO:
            var zp = val.as_zset().unsafe_bitcast[SlabSkipList]()
            var cid = UInt8(9) if t == ValueType.ZSET else UInt8(15)
            var curr = zp[].head[].forward[0]
            while is_not_null(curr):
                var vl = 0
                var vp = gv_bytes(curr[].obj, vbuf, vl)
                _ = self.append_scored(cid, kp, kl, curr[].score, vp, vl)
                curr = curr[].forward[0]
        elif t == ValueType.BITMAP:
            _ = self.append_kv(16, kp, kl, val.as_bitmap(), val.bitmap_len())
        elif t == ValueType.HLL:
            _ = self.append_kv(17, kp, kl, val.as_hll(), HLL_REGISTERS)
        elif t == ValueType.VSET:
            self.append_vset_image(kp, kl, val.as_hash().unsafe_bitcast[VectorSet]())
        elif t == ValueType.STREAM:
            # cmd 23: val = [8B id_ms][8B id_seq][packed field/values], live
            # entries only (XDEL/XTRIM tombstones are compacted away).
            var sd = val.as_hash().unsafe_bitcast[StreamData]()
            for ei in range(sd[].count):
                var e = sd[].entries[unsafe_offset=ei]
                if e.deleted:
                    continue
                var rec = alloc[UInt8](8 + e.data_len)
                rec.unsafe_bitcast[UInt64]()[unsafe_offset=0] = e.id_seq
                if e.data_len > 0:
                    unsafe_memcpy(dest=rec.unsafe_offset(8), src=e.data, count=e.data_len)
                _ = self.append_u64_val(23, kp, kl, e.id_ms, rec, 8 + e.data_len)
                rec.unsafe_free()
        vbuf.unsafe_free()
        fbuf.unsafe_free()

    def log_key_image(mut self, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                      ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                      kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int):
        """Make the WAL say exactly what the keyspace now holds for one key:
        DEL, then its whole value (if it exists), then its TTL — or a PERSIST,
        because replaying a DEL does not clear a TTL an earlier record set.

        For write commands whose handler logs nothing of its own: RENAME,
        COPY, BITOP, BITFIELD, the *STORE family, GEOSEARCHSTORE, SETEX/PSETEX/
        GETEX's TTL, XTRIM. Each silently lost its effect on restart and on
        every replica (the replica decodes this same log) — found by
        tests/test_every_write_survives_restart.py. Cold path by construction:
        these are all slow-path commands."""
        if is_null(self.map):
            return
        _ = self.append(2, kp, kl)
        var key = GenericValue.borrow(kp, kl)
        var v = keyspace[].get(key)
        if not v.is_none() and v.type.value != ValueType.NONE:
            self.append_key_image(kp, kl, v)
            var deadline = GenericValue()
            if is_not_null(ttl_map):
                deadline = ttl_map[].get(key)
            if not deadline.is_none() and deadline.type.value == ValueType.INT:
                _ = self.append_u64_val(25, kp, kl, UInt64(deadline.as_int()),
                                        null_ptr[UInt8, MutUntrackedOrigin](), 0)
            else:
                _ = self.append(26, kp, kl)
        key.free_str_payload()

    def compact_rewrite(mut self, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                        ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] =
                            null_ptr[SlabHashMap, MutUntrackedOrigin]()):
        """BGREWRITEAOF: compact the WAL by rewriting the live keyspace as
        fresh records, then syncing. Result: a minimal WAL with no redundant
        history. Streams and TTLs included — the old rewrite had neither, so
        it dropped every stream and every expiry."""
        if is_null(self.map):
            return
        self.checkpoint()
        var kbuf = alloc[UInt8](64)
        for si in range(8):
            var shard = keyspace[].shards.unsafe_offset(si)
            for i in range(shard[].capacity):
                var m = shard[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                var key = shard[].keys[unsafe_offset=i]
                if not key.is_string():
                    continue
                var kl = key.string_len()
                var kp = key.as_string_safe(kbuf)
                self.append_key_image(kp, kl, shard[].values[unsafe_offset=i])
                if is_not_null(ttl_map):
                    var dl = ttl_map[].get(key)
                    if not dl.is_none() and dl.type.value == ValueType.INT:
                        _ = self.append_u64_val(25, kp, kl, UInt64(dl.as_int()),
                                                null_ptr[UInt8, MutUntrackedOrigin](), 0)
        kbuf.unsafe_free()
        self.sync()

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def close(mut self):
        if is_not_null(self.map):
            _ = external_call["pion_wal_munmap", Int32](self.map, Int(self.file_size))
            self.map = null_ptr[UInt8, MutUntrackedOrigin]()
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = Int32(-1)

    def compress_segment(self, data: String) -> String:
        return data

    def deinit(owned self):
        pass
