"""Snapshot Engine — Phase 1: synchronous KV snapshot.

File format: pion.snapshot.{worker_id}  (written atomically via .tmp → rename)

Header (64 bytes, little-endian):
  [ 8B] magic:    0x31504E4150534E50  ("PNSNAPSHOT1" first 8 bytes LE)
  [ 4B] version:  1
  [ 4B] worker_id
  [ 8B] kv_count: number of STRING KV entries in data section
  [ 8B] timestamp: Unix seconds at snapshot time
  [32B] reserved

Data section: WAL-compatible entries packed contiguously.
  Entry: [4B entry_len][1B cmd_id=1][4B key_len][key bytes][4B val_len][val bytes]
  (identical format to WAL data section — load reuses WAL.recover() logic)

Recovery flow:
  startup:  load_snapshot() → restores base keyspace → WAL.recover() replays deltas
  SAVE:     take_snapshot() → WAL.checkpoint() — WAL resets to 0 so only fresh deltas accumulate
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from std.memory import unsafe_memcpy, unsafe_memset, alloc

from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue, ValueType
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.hll import HLL_REGISTERS
from src.io.blob_store import BlobStore
from src.io.wal import wal_apply_aggregate, wal_apply_ttl, wal_is_aggregate, gv_bytes
from src.common.vector_set import VectorSet
from src.common.stream_data import StreamData


comptime SNAP_MAGIC   = UInt64(0x31504E4150534E50)
# gh #170: v2 serializes every persistable value type as WAL-record-format
# entries (HASH → one HSET record per field, LIST → RPUSH per element in order,
# SET → SADD per member, ZSET/GEO → scored records, BITMAP/HLL → raw images,
# INT/FLOAT scalars → formatted SET). v1 files (strings-only) load unchanged —
# the record framing is identical, v1 just never used ids ≥ 5.
# gh #174 adds STREAM entries (cmd 23) and TTL deadlines (cmd 25) to the same
# data section. The version stays 2: the record framing is unchanged and load
# dispatches on cmd_id, so a file written by this build loads on an older one
# with the new records skipped rather than rejected — and an older file loads
# here unchanged. Bump only if the framing itself changes.
# gh #378 adds vector sets (cmd 28 VADD / 30 VSETATTR) on the same terms.
comptime SNAP_VERSION = UInt32(2)
comptime SNAP_HDR_LEN = 64


@always_inline
def _snap_write_u32(buf: Pointer[UInt8, MutUntrackedOrigin], off: Int, v: UInt32):
    buf[unsafe_offset=off]   = UInt8(v & 0xFF)
    buf[unsafe_offset=off+1] = UInt8((v >> 8) & 0xFF)
    buf[unsafe_offset=off+2] = UInt8((v >> 16) & 0xFF)
    buf[unsafe_offset=off+3] = UInt8((v >> 24) & 0xFF)

@always_inline
def _snap_write_u64(buf: Pointer[UInt8, MutUntrackedOrigin], off: Int, v: UInt64):
    buf[unsafe_offset=off]   = UInt8(v & 0xFF)
    buf[unsafe_offset=off+1] = UInt8((v >> 8) & 0xFF)
    buf[unsafe_offset=off+2] = UInt8((v >> 16) & 0xFF)
    buf[unsafe_offset=off+3] = UInt8((v >> 24) & 0xFF)
    buf[unsafe_offset=off+4] = UInt8((v >> 32) & 0xFF)
    buf[unsafe_offset=off+5] = UInt8((v >> 40) & 0xFF)
    buf[unsafe_offset=off+6] = UInt8((v >> 48) & 0xFF)
    buf[unsafe_offset=off+7] = UInt8((v >> 56) & 0xFF)

@always_inline
def _snap_read_u32(buf: Pointer[UInt8, MutUntrackedOrigin], off: Int) -> UInt32:
    return UInt32(buf[unsafe_offset=off]) | (UInt32(buf[unsafe_offset=off+1]) << 8) | \
           (UInt32(buf[unsafe_offset=off+2]) << 16) | (UInt32(buf[unsafe_offset=off+3]) << 24)

@always_inline
def _snap_read_u64(buf: Pointer[UInt8, MutUntrackedOrigin], off: Int) -> UInt64:
    return UInt64(buf[unsafe_offset=off])       | (UInt64(buf[unsafe_offset=off+1]) << 8)  | \
           (UInt64(buf[unsafe_offset=off+2]) << 16) | (UInt64(buf[unsafe_offset=off+3]) << 24) | \
           (UInt64(buf[unsafe_offset=off+4]) << 32) | (UInt64(buf[unsafe_offset=off+5]) << 40) | \
           (UInt64(buf[unsafe_offset=off+6]) << 48) | (UInt64(buf[unsafe_offset=off+7]) << 56)

def _snap_write_all(fd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    var remaining = n
    var p = ptr
    while remaining > 0:
        var written = external_call["pion_write", Int](fd, p, remaining)
        if written <= 0:
            break
        p = p.unsafe_offset(written)
        remaining -= written

def _snap_read_all(fd: Int32, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    var remaining = n
    var p = ptr
    while remaining > 0:
        var got = external_call["pion_read", Int](fd, p, remaining)
        if got <= 0:
            return False
        p = p.unsafe_offset(got)
        remaining -= got
    return True


struct SnapshotEngine(Movable):
    var snapshot_path: String

    def __init__(out self, path: String):
        self.snapshot_path = path

    def __moveinit__(out self, deinit take: Self):
        self.snapshot_path = take.snapshot_path^

    @staticmethod
    def _record_count(val: GenericValue) -> Int:
        """gh #170: how many records this value serializes to (0 = not persistable)."""
        var t = Int(val.type.value)
        if val.is_string() or t == ValueType.INT or t == ValueType.FLOAT:
            return 1
        if t == ValueType.HASH:
            var hp = val.as_hash().unsafe_bitcast[SlabHashMap]()
            var n = 0
            for j in range(hp[].capacity):
                var hm = hp[].metadata[unsafe_offset=j]
                if hm != SlabHashMap.EMPTY and hm != SlabHashMap.DELETED:
                    n += 1
            if Int(hp[].field_ttl) != 0:   # gh #392: one cmd-32 record per field TTL
                n += hp[].field_ttl[].size
            return n
        if t == ValueType.LIST:
            return val.as_list().unsafe_bitcast[SlabList]()[].llen()
        if t == ValueType.SET:
            var sp = val.as_set().unsafe_bitcast[SlabHashMap]()
            var n = 0
            for j in range(sp[].capacity):
                var sm = sp[].metadata[unsafe_offset=j]
                if sm != SlabHashMap.EMPTY and sm != SlabHashMap.DELETED:
                    n += 1
            return n
        if t == ValueType.ZSET or t == ValueType.GEO:
            return val.as_zset().unsafe_bitcast[SlabSkipList]()[].length
        if t == ValueType.BITMAP or t == ValueType.HLL:
            return 1
        if t == ValueType.STREAM:
            # gh #174: one record per live entry — must match what the writer
            # emits below, or the header's kv_count lies about the file.
            return val.as_hash().unsafe_bitcast[StreamData]()[].alive
        if t == ValueType.VSET:
            # gh #378: one VADD per live element, plus one VSETATTR for each
            # that carries an attribute — the writer's predicate, exactly.
            var vs = val.as_hash().unsafe_bitcast[VectorSet]()
            var n = 0
            for s in range(vs[].n):
                if vs[].alive[unsafe_offset=s] != 0:
                    n += 1
                    if vs[].attrs[s].byte_length() > 0:
                        n += 1
            return n
        return 0

    def _write_record(self, fd: Int32, ehdr: Pointer[UInt8, MutUntrackedOrigin],
                      cmd_id: UInt8,
                      kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                      vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int):
        """One [4B elen][1B cmd][4B klen][key][4B vlen][val] record."""
        _snap_write_u32(ehdr, 0, UInt32(13 + kl + vl))
        ehdr[unsafe_offset=4] = cmd_id
        _snap_write_u32(ehdr, 5, UInt32(kl))
        _snap_write_all(fd, ehdr, 9)
        _snap_write_all(fd, kp, kl)
        _snap_write_u32(ehdr, 0, UInt32(vl))
        _snap_write_all(fd, ehdr, 4)
        if vl > 0:
            _snap_write_all(fd, vp, vl)

    def _write_field_record(self, fd: Int32, ehdr: Pointer[UInt8, MutUntrackedOrigin],
                            cmd_id: UInt8,
                            kp: Pointer[UInt8, MutUntrackedOrigin], kl: Int,
                            fp: Pointer[UInt8, MutUntrackedOrigin], fl: Int,
                            vp: Pointer[UInt8, MutUntrackedOrigin], vl: Int):
        """A record whose value is [4B fl][f][4B vl][v] (WAL append_field_kv)."""
        _snap_write_u32(ehdr, 0, UInt32(13 + kl + 8 + fl + vl))
        ehdr[unsafe_offset=4] = cmd_id
        _snap_write_u32(ehdr, 5, UInt32(kl))
        _snap_write_all(fd, ehdr, 9)
        _snap_write_all(fd, kp, kl)
        _snap_write_u32(ehdr, 0, UInt32(8 + fl + vl))
        _snap_write_u32(ehdr, 4, UInt32(fl))
        _snap_write_all(fd, ehdr, 8)
        _snap_write_all(fd, fp, fl)
        _snap_write_u32(ehdr, 0, UInt32(vl))
        _snap_write_all(fd, ehdr, 4)
        if vl > 0:
            _snap_write_all(fd, vp, vl)

    def take_snapshot(mut self,
                     keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     worker_id: Int, wal_seq: UInt64,
                     blobs: Pointer[BlobStore, MutUntrackedOrigin] =
                         null_ptr[BlobStore, MutUntrackedOrigin](),
                     ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] =
                         null_ptr[SlabHashMap, MutUntrackedOrigin](),
                     path_override: String = "") -> Int64:
        """Serialize the keyspace to pion.snapshot.{worker_id} (atomic via .tmp).
        `path_override` writes straight to that path instead, no rename — the
        replication FULLRESYNC image (gh #390), which must never replace the
        persisted snapshot.
        gh #170: v2 covers STRING/INT/FLOAT/HASH/LIST/SET/ZSET/GEO/BITMAP/HLL
        (+ STREAM gh #174, VSET gh #378) —
        SAVE no longer silently drops aggregates. Returns Unix timestamp on
        success, -1 on failure. Caller should call WAL.checkpoint() after a
        successful return."""

        # --- Pass 1: count records ---
        var kv_count = UInt64(0)
        for si in range(8):
            var shard = keyspace[].shards.unsafe_offset(si)
            for i in range(shard[].capacity):
                var m = shard[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                if shard[].keys[unsafe_offset=i].is_string():
                    kv_count += UInt64(Self._record_count(shard[].values[unsafe_offset=i]))

        # gh #174: TTL records are written after the keyspace, so they have to
        # be counted here too — the loader is bounded by this number, and a
        # writer that emits more records than it counts silently truncates the
        # tail on load. (That is exactly how the first cut of this change lost
        # every TTL through the snapshot path while the WAL path was fine.)
        # The predicate must match the writer's below, or the count drifts again.
        if is_not_null(ttl_map):
            for i in range(ttl_map[].capacity):
                var tm = ttl_map[].metadata[unsafe_offset=i]
                if tm == SlabHashMap.EMPTY or tm == SlabHashMap.DELETED:
                    continue
                if not keyspace[].get(ttl_map[].keys[unsafe_offset=i]).is_none():
                    kv_count += 1
        # #36: a FUNCTION FLUSH record, then one FUNCTION LOAD per library
        var n_libs = Int(external_call["pion_lua_tls_library_count", Int64]())
        kv_count += UInt64(1 + n_libs)

        var ts = external_call["pion_get_unix_time", Int64]()

        # --- Open tmp file ---
        var snap_path = "pion.snapshot." + String(worker_id)
        var tmp_path  = snap_path + ".tmp"
        if path_override.byte_length() > 0:
            snap_path = path_override
            tmp_path = path_override
        var ctmp = tmp_path
        var fd = external_call["pion_creat", Int32](ctmp.as_c_string_slice())
        if fd < 0:
            print("Snapshot: failed to open " + tmp_path)
            return -1

        # --- Write header ---
        var hdr = alloc[UInt8](SNAP_HDR_LEN)
        unsafe_memset(hdr, 0, SNAP_HDR_LEN)
        _snap_write_u64(hdr, 0, SNAP_MAGIC)
        _snap_write_u32(hdr, 8, SNAP_VERSION)
        _snap_write_u32(hdr, 12, UInt32(worker_id))
        _snap_write_u64(hdr, 16, kv_count)
        _snap_write_u64(hdr, 24, UInt64(ts))
        _snap_write_all(fd, hdr, SNAP_HDR_LEN)
        hdr.unsafe_free()

        # --- Pass 2: write entries ---
        # entry header: [4B entry_len][1B cmd_id][4B key_len] = 9 bytes, then val_len 4 bytes
        var ehdr = alloc[UInt8](16)
        var sso_buf = alloc[UInt8](64)   # key scratch (SSO/int formatting)
        var vbuf = alloc[UInt8](64)      # element scratch
        var fbuf = alloc[UInt8](64)      # hash-field scratch
        var scored = alloc[UInt8](8)     # f64 score prefix

        for si in range(8):
            var shard = keyspace[].shards.unsafe_offset(si)
            for i in range(shard[].capacity):
                var m = shard[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                var key = shard[].keys[unsafe_offset=i]
                var val = shard[].values[unsafe_offset=i]
                if not key.is_string():
                    continue

                var kl = key.string_len()
                var kp = key.as_string_safe(sso_buf)
                var t = Int(val.type.value)

                if val.is_string():
                    var vl = val.string_len()

                    # gh #163: a blob-backed value is recorded as its arena pointer,
                    # not its bytes. Copying it here would defeat the tier — SAVE
                    # would write a second multi-GB copy and the reload would bring
                    # it back as anonymous heap.
                    var is_blob = val.is_blob_backed()
                    var bseg = 0
                    var boff = 0
                    if is_blob and is_not_null(blobs):
                        if not blobs[].locate(val.as_string(), vl, bseg, boff):
                            is_blob = False     # not resolvable — fall back to bytes

                    if is_blob:
                        var pr = alloc[UInt8](24)
                        var q = pr.unsafe_bitcast[UInt64]()
                        q[unsafe_offset=0] = UInt64(bseg); q[unsafe_offset=1] = UInt64(boff); q[unsafe_offset=2] = UInt64(vl)
                        self._write_record(fd, ehdr, 4, kp, kl, pr, 24)
                        pr.unsafe_free()
                    else:
                        self._write_record(fd, ehdr, 1, kp, kl, val.as_string_safe(vbuf), vl)

                elif t == ValueType.INT or t == ValueType.FLOAT:
                    # gh #170: INCR'd keys hold INT values — v1 dropped them too.
                    # Restored as strings, matching how WAL replays an INCR's SET.
                    var vl = 0
                    var vp = gv_bytes(val, vbuf, vl)
                    self._write_record(fd, ehdr, 1, kp, kl, vp, vl)

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
                        # record: [4B elen][cmd 5][klen][key][vlen][[4B fl][f][4B vl][v]]
                        _snap_write_u32(ehdr, 0, UInt32(13 + kl + 8 + fl + vl))
                        ehdr[unsafe_offset=4] = UInt8(5)
                        _snap_write_u32(ehdr, 5, UInt32(kl))
                        _snap_write_all(fd, ehdr, 9)
                        _snap_write_all(fd, kp, kl)
                        _snap_write_u32(ehdr, 0, UInt32(8 + fl + vl))
                        _snap_write_u32(ehdr, 4, UInt32(fl))
                        _snap_write_all(fd, ehdr, 8)
                        _snap_write_all(fd, fp, fl)
                        _snap_write_u32(ehdr, 0, UInt32(vl))
                        _snap_write_all(fd, ehdr, 4)
                        _snap_write_all(fd, vp, vl)
                    # gh #392: field TTLs, after the fields they belong to.
                    if Int(hp[].field_ttl) != 0:
                        var ft = hp[].field_ttl
                        for j in range(ft[].capacity):
                            var tm = ft[].metadata[unsafe_offset=j]
                            if tm == SlabHashMap.EMPTY or tm == SlabHashMap.DELETED:
                                continue
                            var fl2 = 0
                            var fp2 = gv_bytes(ft[].keys[unsafe_offset=j], fbuf, fl2)
                            scored.unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(ft[].values[unsafe_offset=j].as_int())
                            self._write_field_record(fd, ehdr, 32, kp, kl, fp2, fl2, scored, 8)

                elif t == ValueType.LIST:
                    var lp = val.as_list().unsafe_bitcast[SlabList]()
                    var elems = lp[].get_all()
                    for j in range(len(elems)):
                        var vl = 0
                        var vp = gv_bytes(elems[j], vbuf, vl)
                        self._write_record(fd, ehdr, 7, kp, kl, vp, vl)   # RPUSH in order

                elif t == ValueType.SET:
                    var sp = val.as_set().unsafe_bitcast[SlabHashMap]()
                    for j in range(sp[].capacity):
                        var sm = sp[].metadata[unsafe_offset=j]
                        if sm == SlabHashMap.EMPTY or sm == SlabHashMap.DELETED:
                            continue
                        var vl = 0
                        var vp = gv_bytes(sp[].keys[unsafe_offset=j], vbuf, vl)
                        self._write_record(fd, ehdr, 8, kp, kl, vp, vl)

                elif t == ValueType.ZSET or t == ValueType.GEO:
                    var zp = val.as_zset().unsafe_bitcast[SlabSkipList]()
                    var cid = UInt8(9) if t == ValueType.ZSET else UInt8(15)
                    var curr = zp[].head[].forward[0]
                    while is_not_null(curr):
                        var vl = 0
                        var vp = gv_bytes(curr[].obj, vbuf, vl)
                        _snap_write_u32(ehdr, 0, UInt32(13 + kl + 8 + vl))
                        ehdr[unsafe_offset=4] = cid
                        _snap_write_u32(ehdr, 5, UInt32(kl))
                        _snap_write_all(fd, ehdr, 9)
                        _snap_write_all(fd, kp, kl)
                        _snap_write_u32(ehdr, 0, UInt32(8 + vl))
                        _snap_write_all(fd, ehdr, 4)
                        scored.unsafe_bitcast[Float64]()[unsafe_offset=0] = curr[].score
                        _snap_write_all(fd, scored, 8)
                        _snap_write_all(fd, vp, vl)
                        curr = curr[].forward[0]

                elif t == ValueType.BITMAP:
                    self._write_record(fd, ehdr, 16, kp, kl, val.as_bitmap(), val.bitmap_len())

                elif t == ValueType.HLL:
                    self._write_record(fd, ehdr, 17, kp, kl, val.as_hll(), HLL_REGISTERS)

                elif t == ValueType.STREAM:
                    # gh #174: one XADD record per *live* entry, in ID order:
                    # cmd 34, whose pairs carry u32 lengths.
                    # Tombstoned entries (XDEL, MAXLEN trim) are skipped rather
                    # than written-then-deleted, so a snapshot compacts the
                    # stream instead of carrying its garbage forward.
                    var sd = val.as_hash().unsafe_bitcast[StreamData]()
                    for ei in range(sd[].count):
                        if sd[].entries[unsafe_offset=ei].deleted:
                            continue
                        var e = sd[].entries[unsafe_offset=ei]
                        _snap_write_u32(ehdr, 0, UInt32(13 + kl + 16 + e.data_len))
                        ehdr[unsafe_offset=4] = UInt8(34)
                        _snap_write_u32(ehdr, 5, UInt32(kl))
                        _snap_write_all(fd, ehdr, 9)
                        _snap_write_all(fd, kp, kl)
                        _snap_write_u32(ehdr, 0, UInt32(16 + e.data_len))
                        _snap_write_all(fd, ehdr, 4)
                        scored.unsafe_bitcast[UInt64]()[unsafe_offset=0] = e.id_ms
                        _snap_write_all(fd, scored, 8)
                        scored.unsafe_bitcast[UInt64]()[unsafe_offset=0] = e.id_seq
                        _snap_write_all(fd, scored, 8)
                        if e.data_len > 0:
                            _snap_write_all(fd, e.data, e.data_len)

                elif t == ValueType.VSET:
                    # gh #378: live elements in slot order (VSIM breaks score
                    # ties by slot), each as the WAL's VADD record, then its
                    # attribute. Tombstones are dropped, so a load compacts.
                    var vs = val.as_hash().unsafe_bitcast[VectorSet]()
                    var payload = alloc[UInt8](vs[].payload_len())
                    for s in range(vs[].n):
                        if vs[].alive[unsafe_offset=s] == 0:
                            continue
                        var pl = vs[].stored_payload(s, payload)
                        var nm = vs[].names[s].copy()
                        var np = nm.unsafe_ptr()
                        self._write_field_record(fd, ehdr, 28, kp, kl,
                            Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(np)),
                            nm.byte_length(), payload, pl)
                        var at = vs[].attrs[s].copy()
                        if at.byte_length() > 0:
                            self._write_field_record(fd, ehdr, 30, kp, kl,
                                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(np)),
                                nm.byte_length(),
                                Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(at.unsafe_ptr())),
                                at.byte_length())
                    payload.unsafe_free()

        # gh #174: TTLs, as cmd-25 absolute-deadline records. Emitted after the
        # keyspace so a load applies them to keys that already exist, and only
        # for keys still present — a ttl_map entry whose key is gone is stale
        # bookkeeping, and writing it would resurrect a TTL for a dead key.
        if is_not_null(ttl_map):
            for i in range(ttl_map[].capacity):
                var m = ttl_map[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                var tkey = ttl_map[].keys[unsafe_offset=i]
                if keyspace[].get(tkey).is_none():
                    continue
                var kl2 = 0
                var kp2 = gv_bytes(tkey, sso_buf, kl2)
                var deadline = UInt64(ttl_map[].values[unsafe_offset=i].as_int())
                _snap_write_u32(ehdr, 0, UInt32(13 + kl2 + 8))
                ehdr[unsafe_offset=4] = UInt8(25)
                _snap_write_u32(ehdr, 5, UInt32(kl2))
                _snap_write_all(fd, ehdr, 9)
                _snap_write_all(fd, kp2, kl2)
                _snap_write_u32(ehdr, 0, UInt32(8))
                _snap_write_all(fd, ehdr, 4)
                scored.unsafe_bitcast[UInt64]()[unsafe_offset=0] = deadline
                _snap_write_all(fd, scored, 8)

        # #36: the FUNCTION libraries. FLUSH first, so a replica loading this
        # image (a FULLRESYNC) also drops libraries the primary has deleted.
        self._write_record(fd, ehdr, UInt8(37), sso_buf, 0, sso_buf, 0)
        var clen = alloc[Int64](1)
        for li in range(n_libs):
            clen[unsafe_offset=0] = 0
            var code = external_call["pion_lua_tls_library_code", Pointer[UInt8, MutUntrackedOrigin]](
                Int64(li), clen)
            var lname = external_call["pion_lua_tls_library_name", Pointer[UInt8, MutUntrackedOrigin]](Int64(li))
            var lnl = 0
            while lname[unsafe_offset=lnl] != 0:
                lnl += 1
            self._write_record(fd, ehdr, UInt8(35), lname, lnl, code, Int(clen[unsafe_offset=0]))
        clen.unsafe_free()

        ehdr.unsafe_free()
        sso_buf.unsafe_free()
        vbuf.unsafe_free()
        fbuf.unsafe_free()
        scored.unsafe_free()

        if path_override.byte_length() == 0:   # a replication image is read back at once
            _ = external_call["pion_fdatasync", Int32](fd)
        _ = external_call["close", Int32](fd)

        # Atomic rename .tmp → final
        if path_override.byte_length() == 0:
            var csnap = snap_path
            _ = external_call["pion_snapshot_rename", Int32](
                ctmp.as_c_string_slice(), csnap.as_c_string_slice())

        print("Snapshot: saved " + String(kv_count) + " entries → " + snap_path)
        return ts

    def load_snapshot(mut self,
                     keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     worker_id: Int,
                     blobs: Pointer[BlobStore, MutUntrackedOrigin] =
                         null_ptr[BlobStore, MutUntrackedOrigin](),
                     ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin] =
                         null_ptr[SlabHashMap, MutUntrackedOrigin]()) -> Int64:
        """Load snapshot into keyspace. Returns snapshot timestamp (-1 if no snapshot)."""
        var snap_path = "pion.snapshot." + String(worker_id)
        var cpath = snap_path
        var fd = external_call["pion_open_rdonly", Int32](cpath.as_c_string_slice())
        if fd < 0:
            return -1

        # Read and verify header
        var hdr = alloc[UInt8](SNAP_HDR_LEN)
        if not _snap_read_all(fd, hdr, SNAP_HDR_LEN):
            hdr.unsafe_free()
            _ = external_call["close", Int32](fd)
            return -1

        var magic = _snap_read_u64(hdr, 0)
        if magic != SNAP_MAGIC:
            print("Snapshot: bad magic in " + snap_path + " — skipping")
            hdr.unsafe_free()
            _ = external_call["close", Int32](fd)
            return -1

        var kv_count = Int(_snap_read_u64(hdr, 16))
        var ts       = Int64(_snap_read_u64(hdr, 24))
        hdr.unsafe_free()

        # Read entries
        var ehdr    = alloc[UInt8](16)
        var replayed = 0

        # gh #174: read until EOF rather than exactly `kv_count` records.
        # `_snap_read_all` already returns False at end-of-file, so the loop
        # terminates identically for a well-formed file — but a header whose
        # count under-reports the body no longer silently drops the tail. The
        # count stays in the header (and stays correct) as a cross-check;
        # trusting it as the *bound* is what made a writer/counter mismatch a
        # silent data-loss bug instead of a loud one. kv_count remains the
        # iteration cap-of-last-resort against a corrupt, endlessly-parsing file.
        var guard = 0
        var max_records = Int(kv_count) * 2 + 1024
        while guard < max_records:
            guard += 1
            # 9-byte mini-header: entry_len(4) + cmd_id(1) + key_len(4)
            if not _snap_read_all(fd, ehdr, 9):
                break
            var cmd_id = ehdr[unsafe_offset=4]
            var kl     = Int(_snap_read_u32(ehdr, 5))

            var key_buf = alloc[UInt8](kl + 1)
            if not _snap_read_all(fd, key_buf, kl):
                key_buf.unsafe_free()
                break

            # val_len (4B)
            if not _snap_read_all(fd, ehdr, 4):
                key_buf.unsafe_free()
                break
            var vl = Int(_snap_read_u32(ehdr, 0))

            var val_buf = alloc[UInt8](vl + 1)
            if vl > 0 and not _snap_read_all(fd, val_buf, vl):
                key_buf.unsafe_free()
                val_buf.unsafe_free()
                break

            if cmd_id == 1:   # SET
                var key_gv = GenericValue.from_ptr(key_buf, kl)
                var val_gv = GenericValue.from_ptr(val_buf, vl)
                keyspace[].set(key_gv, val_gv)
                replayed += 1
            elif cmd_id == 4 and vl == 24 and is_not_null(blobs):
                # gh #163: pointer record into the blob arena (see WAL cmd 4).
                var q = val_buf.unsafe_bitcast[UInt64]()
                var bp = blobs[].ptr_at(Int(q[unsafe_offset=0]), Int(q[unsafe_offset=1]), Int(q[unsafe_offset=2]))
                if is_not_null(bp):
                    var key_gv = GenericValue.from_ptr(key_buf, kl)
                    keyspace[].set(key_gv, GenericValue.from_blob_ptr(bp, Int(q[unsafe_offset=2])))
                    replayed += 1
            elif cmd_id >= 35 and cmd_id <= 37:
                # #36: the FUNCTION libraries (a 37 FLUSH, then one 35 per library)
                if external_call["pion_lua_wal_apply", Int64](
                        Int64(cmd_id), key_buf, Int64(kl), val_buf, Int64(vl)) == 1:
                    replayed += 1
            elif cmd_id == 25 or cmd_id == 26:
                # gh #174: TTL record targets the ttl_map, not the keyspace.
                if wal_apply_ttl(cmd_id, key_buf, kl, val_buf, vl, ttl_map):
                    replayed += 1
            elif wal_is_aggregate(cmd_id):
                # gh #170 aggregate record (extended by gh #174: 23 XADD,
                # 24 PFADD, 27 XDEL; gh #378: 28-30 vector sets) — same apply
                # path as WAL replay.
                if wal_apply_aggregate(cmd_id, key_buf, kl, val_buf, vl, keyspace):
                    replayed += 1

            key_buf.unsafe_free()
            val_buf.unsafe_free()

        ehdr.unsafe_free()
        _ = external_call["close", Int32](fd)

        print("Snapshot: loaded " + String(replayed) + " entries from " + snap_path)
        return ts
