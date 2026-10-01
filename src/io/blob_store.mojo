"""BlobStore — file-backed arena for large values (gh #163, gh #149).

Why this exists
---------------
Values above `--blob-threshold` (1 MiB by default) used to live on the anonymous
heap and be memcpy'd into the WAL ring. That is two problems at once:

  * gh #163 — anonymous pages are not reclaimable. A 4.6 GB blob set co-resident
    with an mlx process on a 16 GB Mac is jetsam bait: macOS SIGKILL'd
    pion-server (rc=137) and took every unflushed byte with it.
  * gh #149 — a 6 MB value is a 6 MB WAL entry, so the fixed 256 MB ring filled
    after ~42 blobs and then silently refused every later write while still
    answering +OK.

Blob bytes now live in mmap'd segment files. Pages are file-backed, so the
kernel can evict them under pressure instead of killing the process, and they
are on disk the moment they are written rather than at some later flush.

What is stored where
--------------------
The segments are a **byte arena, not a log** — they hold payload bytes only. All
ordering and naming stays in the structures that already have it:

  * WAL `cmd_id = 4` — key -> (segment, offset, length), a ~29 byte entry
    instead of a 6 MB one. The WAL therefore stays the single ordered index, so
    a large value overwriting a small one (and the reverse) replays correctly
    with no merge logic.
  * Snapshot — the same pointer record, so SAVE does not copy blob bytes back
    out into a second file.

A value read out of the arena is a normal heap-STRING `GenericValue` whose
pointer happens to be inside the mapping, tagged with `BLOB_TAG` in `_data2` so
the gh #123 free path leaves it alone (see `GenericValue.free_str_payload`).

Layout — pion.blob.{worker}.{segment}:

  Header (64 bytes):
    [8B] magic 0x31424C42_4E4F4950 ("PIONBLB1" LE)
    [8B] segment index
    [8B] high_water — bytes used in the data section
    [8B] worker_id
    [32B] reserved
  Data section: payloads, each 16-byte aligned. No per-record header — the WAL
  entry is the record.

Reclaim (gh #167): overwriting or deleting a key strands its bytes. `compact()`
runs at startup, after replay and before the socket serves: it counts what the
keyspace can still reach, and when over half the arena is stranded (and it is
worth the rewrite at all) copies the live payloads into fresh segments,
re-points every value, and unlinks the old files. Startup is the only safe
moment — it is when the complete live set is known, every pointer into the arena
is ours to rewrite, and no reader can be holding a `GenericValue` mid-GET. The
caller must snapshot + checkpoint afterwards, because the WAL's cmd-4 records
still name the pre-compaction offsets. Segments are bounded by
BLOB_MAX_SEGMENTS; on exhaustion large values fall back to the heap path with a
warning rather than failing the write.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue
from std.memory.unsafe_pointer import Pointer
from std.memory import unsafe_memcpy
from std.ffi import external_call
from std.collections import Array


comptime BLOB_MAGIC        = UInt64(0x31424C424E4F4950)   # "PIONBLB1" LE
# "Tier off" sentinel for FastPathHandler/CommandDispatcher.blob_threshold.
# Disabled is a huge threshold rather than 0 so the fast-path SET routing stays
# a single `val_len >= threshold` compare — no separate enabled flag to test.
comptime BLOB_TIER_OFF     = Int(1) << 62
comptime BLOB_HEADER_SIZE  = 64
comptime BLOB_SEG_SIZE     = 1024 * 1024 * 1024            # 1 GiB per segment
comptime BLOB_MAX_SEGMENTS = 64                            # 64 GiB ceiling per worker
comptime BLOB_ALIGN        = 16
# gh #167: don't rewrite a small arena. Below this the stranded bytes are not
# worth the startup cost, and the ratio test alone would fire on a 2-value store.
comptime BLOB_COMPACT_MIN_BYTES = 256 * 1024 * 1024
comptime MS_ASYNC          = Int32(1)


@always_inline
def _align_up(v: Int, a: Int) -> Int:
    return (v + a - 1) & ~(a - 1)


struct BlobStore(Movable):
    var prefix: String                                  # "pion.blob.{worker}"
    var worker_id: Int
    var enabled: Bool
    var seg_count: Int
    var fds: Array[Int32, BLOB_MAX_SEGMENTS]
    var maps: Array[Pointer[UInt8, MutUntrackedOrigin], BLOB_MAX_SEGMENTS]
    var sizes: Array[Int, BLOB_MAX_SEGMENTS]      # mapped bytes per segment
    var high: Array[Int, BLOB_MAX_SEGMENTS]       # bytes used per segment
    var dirty: Bool
    var bytes_used: UInt64                              # arena bytes handed out (incl. stranded)
    # Live blob values at the last startup scan, plus appends since. DELs and
    # overwrites do not decrement (that would need a keyspace walk); the next
    # startup scan re-trues it. gh #167: was "payloads written this run", which
    # read 0 after any restart — including right after a compaction.
    var records: UInt64
    var full_warned: Bool
    # gh #167: compaction bookkeeping. `live_at_last_scan` is what the keyspace
    # could still reach the last time we counted (startup) — not a running total,
    # because keeping one would mean a keyspace walk per INFO.
    var live_at_last_scan: UInt64
    var compactions: UInt64

    def __init__(out self, prefix: String, worker_id: Int = 0, enabled: Bool = True):
        self.prefix = prefix
        self.worker_id = worker_id
        self.enabled = enabled
        self.seg_count = 0
        self.fds = Array[Int32, BLOB_MAX_SEGMENTS](fill=Int32(-1))
        self.maps = Array[Pointer[UInt8, MutUntrackedOrigin], BLOB_MAX_SEGMENTS](
            fill=null_ptr[UInt8, MutUntrackedOrigin]())
        self.sizes = Array[Int, BLOB_MAX_SEGMENTS](fill=0)
        self.high = Array[Int, BLOB_MAX_SEGMENTS](fill=0)
        self.dirty = False
        self.bytes_used = 0
        self.records = 0
        self.full_warned = False
        self.live_at_last_scan = 0
        self.compactions = 0
        if not enabled:
            return
        # Re-map segments left by a previous run. They are numbered contiguously
        # from 0, so the first gap ends the set.
        while self.seg_count < BLOB_MAX_SEGMENTS:
            if not self._map_existing(self.seg_count):
                break
            self.seg_count += 1
        if self.seg_count > 0:
            print("Blob tier: mapped " + String(self.seg_count) + " segment(s), "
                  + String(self.bytes_used // (1024 * 1024)) + " MB in use")

    def __moveinit__(out self, deinit take: Self):
        self.prefix = take.prefix^
        self.worker_id = take.worker_id
        self.enabled = take.enabled
        self.seg_count = take.seg_count
        self.fds = take.fds^
        self.maps = take.maps^
        self.sizes = take.sizes^
        self.high = take.high^
        self.dirty = take.dirty
        self.bytes_used = take.bytes_used
        self.records = take.records
        self.full_warned = take.full_warned
        self.live_at_last_scan = take.live_at_last_scan
        self.compactions = take.compactions

    @always_inline
    def _seg_path(self, n: Int) -> String:
        return self.prefix + "." + String(n)

    def _map_existing(mut self, n: Int) -> Bool:
        """Map segment n if the file is already there. Returns False on the first gap."""
        var path = self._seg_path(n)
        var probe = path
        if external_call["access", Int32](probe.as_c_string_slice(), Int32(0)) != 0:
            return False
        var cpath = path
        var fd = external_call["pion_wal_open", Int32](cpath.as_c_string_slice())
        if fd < 0:
            return False
        # The on-disk size is authoritative: an oversized segment (single payload
        # larger than BLOB_SEG_SIZE) is not BLOB_SEG_SIZE bytes long.
        var fsize = Int(external_call["pion_file_size", Int64](fd))
        if fsize < BLOB_HEADER_SIZE:
            _ = external_call["close", Int32](fd)
            return False
        var mp = external_call["pion_wal_mmap", Pointer[UInt8, MutUntrackedOrigin]](
            fd, Int(fsize))
        if is_null(mp):
            _ = external_call["close", Int32](fd)
            return False
        var hdr = mp.unsafe_bitcast[UInt64]()
        if hdr[unsafe_offset=0] != BLOB_MAGIC:
            _ = external_call["pion_wal_munmap", Int32](mp, Int(fsize))
            _ = external_call["close", Int32](fd)
            return False
        self.fds[n] = fd
        self.maps[n] = mp
        self.sizes[n] = fsize
        self.high[n] = Int(hdr[unsafe_offset=2])
        self.bytes_used += UInt64(Int(hdr[unsafe_offset=2]))
        return True

    def _create_segment(mut self, n: Int, size: Int) -> Bool:
        var path = self._seg_path(n)
        var cpath = path
        var fd = external_call["pion_wal_open", Int32](cpath.as_c_string_slice())
        if fd < 0:
            return False
        if external_call["pion_wal_ftruncate", Int32](fd, Int(size)) != 0:
            _ = external_call["close", Int32](fd)
            return False
        var mp = external_call["pion_wal_mmap", Pointer[UInt8, MutUntrackedOrigin]](
            fd, Int(size))
        if is_null(mp):
            _ = external_call["close", Int32](fd)
            return False
        var hdr = mp.unsafe_bitcast[UInt64]()
        hdr[unsafe_offset=0] = BLOB_MAGIC
        hdr[unsafe_offset=1] = UInt64(n)
        hdr[unsafe_offset=2] = 0                      # high_water
        hdr[unsafe_offset=3] = UInt64(self.worker_id)
        self.fds[n] = fd
        self.maps[n] = mp
        self.sizes[n] = size
        self.high[n] = 0
        return True

    @always_inline
    def is_open(self) -> Bool:
        return self.enabled

    def append(mut self, src: Pointer[UInt8, _], length: Int,
               mut out_seg: Int, mut out_off: Int) -> Bool:
        """Copy `length` bytes into the arena. On success sets (out_seg, out_off).

        Cold-ish path by construction — only values >= --blob-threshold get here,
        so the memcpy dominates and the bookkeeping does not matter."""
        out_seg = -1
        out_off = -1
        if not self.enabled or length <= 0:
            return False

        var need = _align_up(length, BLOB_ALIGN)

        # Current segment first, then a fresh one.
        if self.seg_count > 0:
            var cur = self.seg_count - 1
            if is_not_null(self.maps[cur]) \
               and self.high[cur] + need + BLOB_HEADER_SIZE <= self.sizes[cur]:
                return self._write_into(cur, need, length, src, out_seg, out_off)

        if self.seg_count >= BLOB_MAX_SEGMENTS:
            if not self.full_warned:
                self.full_warned = True
                print("Blob tier: FULL — " + String(BLOB_MAX_SEGMENTS)
                      + " segments in use; large values fall back to heap storage"
                      + " (dead arena space is reclaimed only at startup; see doc/operations.md).")
            return False

        # A payload bigger than the standard segment gets a segment of its own.
        var seg_size = BLOB_SEG_SIZE
        if need + BLOB_HEADER_SIZE > seg_size:
            seg_size = _align_up(need + BLOB_HEADER_SIZE, 4096)
        if not self._create_segment(self.seg_count, seg_size):
            return False
        self.seg_count += 1
        return self._write_into(self.seg_count - 1, need, length, src, out_seg, out_off)

    def _write_into(mut self, seg: Int, need: Int, length: Int,
                    src: Pointer[UInt8, _],
                    mut out_seg: Int, mut out_off: Int) -> Bool:
        var off = self.high[seg]
        var dst = self.maps[seg].unsafe_offset(BLOB_HEADER_SIZE).unsafe_offset(off)
        unsafe_memcpy(dest=dst, src=src, count=length)
        self.high[seg] = off + need
        var hdr = self.maps[seg].unsafe_bitcast[UInt64]()
        hdr[unsafe_offset=2] = UInt64(self.high[seg])
        self.bytes_used += UInt64(need)
        self.records += 1
        self.dirty = True
        out_seg = seg
        out_off = off
        return True

    @always_inline
    def ptr_at(self, seg: Int, off: Int, length: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Resolve a (segment, offset) pair to a pointer, or null if it does not
        address live arena bytes. Recovery feeds unvalidated values in here, so
        the bounds check is load-bearing, not defensive decoration."""
        if seg < 0 or seg >= self.seg_count or length <= 0:
            return null_ptr[UInt8, MutUntrackedOrigin]()
        if is_null(self.maps[seg]):
            return null_ptr[UInt8, MutUntrackedOrigin]()
        if off < 0 or off + length + BLOB_HEADER_SIZE > self.sizes[seg]:
            return null_ptr[UInt8, MutUntrackedOrigin]()
        return self.maps[seg].unsafe_offset(BLOB_HEADER_SIZE).unsafe_offset(off)

    def locate(self, ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int,
               mut out_seg: Int, mut out_off: Int) -> Bool:
        """Inverse of ptr_at: which (segment, offset) does this pointer name?
        SAVE uses it to write a pointer record instead of copying arena bytes."""
        out_seg = -1
        out_off = -1
        if is_null(ptr) or length <= 0:
            return False
        var addr = Int(ptr)
        for n in range(self.seg_count):
            if is_null(self.maps[n]):
                continue
            var base = Int(self.maps[n]) + BLOB_HEADER_SIZE
            var off = addr - base
            if off >= 0 and off + length + BLOB_HEADER_SIZE <= self.sizes[n]:
                out_seg = n
                out_off = off
                return True
        return False

    # ── Compaction (gh #167) ───────────────────────────────────────────────

    def live_scan(self, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                  mut out_records: UInt64) -> UInt64:
        """Arena bytes still reachable from the keyspace (everything else is
        stranded by an overwrite or a DEL), plus the count of reachable blob
        values in `out_records`."""
        var live = UInt64(0)
        out_records = 0
        for si in range(8):
            var shard = keyspace[].shards.unsafe_offset(si)
            for i in range(shard[].capacity):
                var m = shard[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                var val = shard[].values[unsafe_offset=i]
                if val.is_blob_backed():
                    live += UInt64(_align_up(val.string_len(), BLOB_ALIGN))
                    out_records += 1
        return live

    def compact(mut self,
                keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]) -> Bool:
        """Rewrite the live payloads into a fresh set of segments and drop the old
        ones. Returns True when a compaction actually ran.

        Startup-only, deliberately. This is the one moment the complete live set
        is known, every pointer into the arena is ours to rewrite, and no reader
        can be holding a `GenericValue` mid-GET. A live compactor would have to
        move bytes out from under values the keyspace is handing out right now.

        The caller must snapshot + checkpoint afterwards: the WAL's cmd-4 records
        still name the *old* offsets, and nothing else re-indexes them."""
        if not self.enabled or self.seg_count == 0:
            return False
        var live_records = UInt64(0)
        var live = self.live_scan(keyspace, live_records)
        self.live_at_last_scan = live
        # gh #167 residual (2): the startup scan is also the moment `records`
        # becomes knowable again — the arena stores bytes, not records, so a
        # warm start (compacting or not) counts what the keyspace can reach.
        self.records = live_records
        if self.bytes_used == 0:
            return False
        # gh #167 residual (2): gate on DEAD bytes, not total. Gating on total
        # let a 100% dead arena under the floor survive every restart — any
        # workload cycling under the threshold accumulated garbage permanently.
        var dead = self.bytes_used - live if self.bytes_used > live else UInt64(0)
        if live > 0:
            # Bounded garbage: with live data present the rewrite copies every
            # live payload at startup, so require both enough dead bytes to be
            # worth it and a majority-dead arena.
            if dead < UInt64(BLOB_COMPACT_MIN_BYTES):
                return False
            if dead * 2 < self.bytes_used:
                return False        # under 50% stranded — not worth the rewrite
        # live == 0 with bytes in use: nothing to copy, "compaction" is just
        # dropping the files — do it at any size.

        # A leftover set from an interrupted compaction would otherwise be
        # adopted by the fresh store below.
        var tmp_prefix = self.prefix + ".compact"
        for n in range(BLOB_MAX_SEGMENTS):
            var stale = tmp_prefix + "." + String(n)
            _ = external_call["unlink", Int32](stale.as_c_string_slice())

        var fresh = BlobStore(tmp_prefix, self.worker_id, True)
        var moved = 0
        for si in range(8):
            var shard = keyspace[].shards.unsafe_offset(si)
            for i in range(shard[].capacity):
                var m = shard[].metadata[unsafe_offset=i]
                if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                    continue
                var val = shard[].values[unsafe_offset=i]
                if not val.is_blob_backed():
                    continue
                var vlen = val.string_len()
                var nseg = -1
                var noff = -1
                if not fresh.append(val.as_string(), vlen, nseg, noff):
                    # Out of room in the fresh arena: abandon the whole pass
                    # rather than leave half the keyspace pointing at a store we
                    # are about to unlink.
                    print("Blob tier: compaction aborted (fresh arena full) — "
                          + "keeping the existing segments")
                    fresh.close()
                    for n in range(BLOB_MAX_SEGMENTS):
                        var abandoned = tmp_prefix + "." + String(n)
                        _ = external_call["unlink", Int32](abandoned.as_c_string_slice())
                    return False
                var np = fresh.ptr_at(nseg, noff, vlen)
                if is_not_null(np):
                    shard[].values[unsafe_offset=i] = GenericValue.from_blob_ptr(np, vlen)
                    moved += 1

        fresh.sync()
        var reclaimed = self.bytes_used - fresh.bytes_used

        # Drop the old mappings first — every live value now points into `fresh`.
        var old_segs = self.seg_count
        self.close()
        for n in range(old_segs):
            var old = self._seg_path(n)
            _ = external_call["unlink", Int32](old.as_c_string_slice())

        # Rename the fresh files into the canonical names. The mappings survive:
        # they are bound to the inode, not the path.
        for n in range(fresh.seg_count):
            var from_p = tmp_prefix + "." + String(n)
            var to_p = self._seg_path(n)
            _ = external_call["rename", Int32](
                from_p.as_c_string_slice(), to_p.as_c_string_slice())

        # Adopt the fresh store's open segments.
        for n in range(fresh.seg_count):
            self.fds[n] = fresh.fds[n]
            self.maps[n] = fresh.maps[n]
            self.sizes[n] = fresh.sizes[n]
            self.high[n] = fresh.high[n]
        self.seg_count = fresh.seg_count
        self.bytes_used = fresh.bytes_used
        self.live_at_last_scan = fresh.bytes_used
        # gh #167 residual (1): the re-index is a full rewrite of the live set,
        # so the record count is exactly what was moved. Without this the
        # counter read 0 after every startup compaction.
        self.records = UInt64(moved)
        self.compactions += 1
        self.full_warned = False
        print("Blob tier: compacted " + String(moved) + " value(s), reclaimed "
              + String(reclaimed // (1024 * 1024)) + " MB, now "
              + String(self.seg_count) + " segment(s)")
        return True

    def sync(mut self):
        """Group-commit the arena alongside the WAL (once per event-loop tick)."""
        if not self.dirty or not self.enabled:
            return
        for n in range(self.seg_count):
            if is_not_null(self.maps[n]):
                _ = external_call["pion_wal_msync", Int32](
                    self.maps[n], Int(BLOB_HEADER_SIZE + self.high[n]), MS_ASYNC)
        self.dirty = False

    def close(mut self):
        for n in range(self.seg_count):
            if is_not_null(self.maps[n]):
                _ = external_call["pion_wal_munmap", Int32](self.maps[n], Int(self.sizes[n]))
                self.maps[n] = null_ptr[UInt8, MutUntrackedOrigin]()
            if self.fds[n] >= 0:
                _ = external_call["close", Int32](self.fds[n])
                self.fds[n] = Int32(-1)
        self.seg_count = 0

    def deinit(owned self):
        pass
