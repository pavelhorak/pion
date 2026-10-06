"""StreamEntry / StreamData — the storage layer for Redis streams.

gh #174: lifted out of `src/commands/stream.mojo` so `src/io/wal.mojo` can
replay XADD records without importing the command module. The command module
pulls in `fast_path` (for `_get_now_ns`) and `fast_path` pulls in `wal`, so a
direct import would close the cycle wal → stream → fast_path → wal. These two
structs are pure data with no protocol or engine dependencies, which is what
makes `src/common/` the right home — the same reason `SlabList` and
`SlabSkipList` live there rather than beside their handlers.

`stream.mojo` re-exports both names, so every existing `from src.commands.stream
import StreamData` keeps working.
"""
from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.collections import List

# #40: a group's entries-read counter when it is not known (Redis's
# SCG_INVALID_ENTRIES_READ).
comptime SCG_INVALID_ENTRIES_READ = Int64(-1)


@always_inline
def sid_lt(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Bool:
    return a_ms < b_ms or (a_ms == b_ms and a_seq < b_seq)


@always_inline
def sid_cmp(a_ms: UInt64, a_seq: UInt64, b_ms: UInt64, b_seq: UInt64) -> Int:
    if sid_lt(a_ms, a_seq, b_ms, b_seq):
        return -1
    if a_ms == b_ms and a_seq == b_seq:
        return 0
    return 1


# ── #40: consumer groups ──

# How a deletion treats consumer-group references (Redis 8.2's KEEPREF,
# DELREF, ACKED on XADD / XTRIM trimming, XDELEX and XACKDEL): keep the
# pending entries that name a deleted entry, remove them with it, or delete
# only entries no group still needs.
comptime DEL_NONE = 0
comptime DEL_KEEPREF = 1
comptime DEL_DELREF = 2
comptime DEL_ACKED = 3


# A pending-list slot's owner: a consumer's id (>= 1), UNOWNED for the
# moment XCLAIM FORCE has created an entry and not yet given it to anyone,
# DEAD once the entry is acknowledged (the slot stays until compaction).
comptime PEL_UNOWNED = 0
comptime PEL_DEAD = -1


struct StreamNack(Copyable, Movable, ImplicitlyCopyable):
    """One entry of a group's pending entries list: delivered to a consumer,
    not yet acknowledged."""
    var ms: UInt64
    var seq: UInt64
    var consumer: Int          # its consumer's id, PEL_UNOWNED or PEL_DEAD
    var delivery_time: Int64   # unix ms of the last delivery
    var delivery_count: Int64

    def __init__(out self, ms: UInt64, seq: UInt64, consumer: Int, delivery_time: Int64,
                 delivery_count: Int64):
        self.ms = ms
        self.seq = seq
        self.consumer = consumer
        self.delivery_time = delivery_time
        self.delivery_count = delivery_count


struct StreamConsumer(Copyable, Movable):
    var id: Int                 # its slot in StreamGroup.consumers + 1
    var name: List[UInt8]
    var seen_time: Int64        # unix ms of its last attempt to read or claim
    var active_time: Int64      # unix ms of its last successful one, -1 never
    var pending: Int            # pending entries it owns
    var alive: Bool             # False once deleted (the slot stays until compaction)
    var logged_seen: Int64      # the seen time its last record (41) carried; not itself logged

    def __init__(out self, id: Int, var name: List[UInt8], seen_time: Int64):
        self.id = id
        self.name = name^
        self.seen_time = seen_time
        self.active_time = -1
        self.pending = 0
        self.alive = True
        self.logged_seen = seen_time


struct StreamGroup(Copyable, Movable):
    """A consumer group. Both lists are built for a work queue's volumes:

      * `pel` is sorted by id. An acknowledged entry becomes a DEAD slot,
        dropped in bulk once dead slots are most of the list, so XACK costs a
        binary search, not a shift of every later entry; a new delivery has
        the highest id so far and appends.
      * `consumers` is in creation order and a consumer's id is its slot + 1,
        so a pending entry finds its owner directly; `by_name` holds the live
        slots in name order, Redis's listing order, for lookups by name.
        Deleted slots are compacted the same way, renumbering the owners.

    Iterate the pending list with pel_next_live, or skip PEL_DEAD slots;
    iterate consumers in name order through by_name."""
    var name: List[UInt8]
    var last_ms: UInt64        # last-delivered-id
    var last_seq: UInt64
    var entries_read: Int64    # SCG_INVALID_ENTRIES_READ when not known
    var pel: List[StreamNack]
    var pel_dead: Int
    var consumers: List[StreamConsumer]
    var by_name: List[Int]
    var consumers_dead: Int

    def __init__(out self, var name: List[UInt8], last_ms: UInt64, last_seq: UInt64, entries_read: Int64):
        self.name = name^
        self.last_ms = last_ms
        self.last_seq = last_seq
        self.entries_read = entries_read
        self.pel = List[StreamNack]()
        self.pel_dead = 0
        self.consumers = List[StreamConsumer]()
        self.by_name = List[Int]()
        self.consumers_dead = 0

    # ── the pending entries list ──

    @always_inline
    def pel_live(self) -> Int:
        return len(self.pel) - self.pel_dead

    def pel_lower_bound(self, ms: UInt64, seq: UInt64) -> Int:
        """The first slot (live or dead) with an id >= (ms, seq)."""
        var lo = 0
        var hi = len(self.pel)
        while lo < hi:
            var mid = (lo + hi) // 2
            if sid_lt(self.pel[mid].ms, self.pel[mid].seq, ms, seq):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def pel_next_live(self, k: Int) -> Int:
        """The first live slot at or after k, or len(pel)."""
        var j = k
        while j < len(self.pel) and self.pel[j].consumer == PEL_DEAD:
            j += 1
        return j

    def pel_last_live(self) -> Int:
        """The last live slot, or -1."""
        var j = len(self.pel) - 1
        while j >= 0 and self.pel[j].consumer == PEL_DEAD:
            j -= 1
        return j

    def pel_find(self, ms: UInt64, seq: UInt64) -> Int:
        """The slot of the live pending entry with this id, or -1."""
        var k = self.pel_lower_bound(ms, seq)
        if k < len(self.pel) and self.pel[k].ms == ms and self.pel[k].seq == seq \
           and self.pel[k].consumer != PEL_DEAD:
            return k
        return -1

    def set_nack(mut self, ms: UInt64, seq: UInt64, consumer_id: Int, delivery_time: Int64, delivery_count: Int64):
        """Create or update the pending entry for (ms, seq), owned by
        `consumer_id`; the owners' pending counts follow."""
        var k = self.pel_lower_bound(ms, seq)
        if k < len(self.pel) and self.pel[k].ms == ms and self.pel[k].seq == seq:
            var old = self.pel[k].consumer
            if old == PEL_DEAD:
                self.pel_dead -= 1
            if old != consumer_id:
                var oi = self.consumer_by_id(old)
                if oi >= 0:
                    self.consumers[oi].pending -= 1
                var ni = self.consumer_by_id(consumer_id)
                if ni >= 0:
                    self.consumers[ni].pending += 1
            self.pel[k].consumer = consumer_id
            self.pel[k].delivery_time = delivery_time
            self.pel[k].delivery_count = delivery_count
            return
        if k == len(self.pel):
            self.pel.append(StreamNack(ms, seq, consumer_id, delivery_time, delivery_count))
        else:
            self.pel.insert(k, StreamNack(ms, seq, consumer_id, delivery_time, delivery_count))
        var ci = self.consumer_by_id(consumer_id)
        if ci >= 0:
            self.consumers[ci].pending += 1

    def remove_nack_at(mut self, k: Int):
        """Acknowledge live slot k. Slots do not move: an iteration over the
        list stays valid; pel_tidy() compacts once the caller is done."""
        var ci = self.consumer_by_id(self.pel[k].consumer)
        if ci >= 0:
            self.consumers[ci].pending -= 1
        self.pel[k].consumer = PEL_DEAD
        self.pel_dead += 1

    def pel_tidy(mut self):
        """Drop the dead slots once they are most of the list: each
        acknowledgement pays for its own slot's removal, amortized."""
        if self.pel_dead == 0 or (self.pel_dead <= 64 and self.pel_dead < len(self.pel)) \
           or self.pel_dead * 2 < len(self.pel):
            return
        var out = List[StreamNack](capacity=len(self.pel) - self.pel_dead)
        for k in range(len(self.pel)):
            if self.pel[k].consumer != PEL_DEAD:
                out.append(self.pel[k])
        self.pel = out^
        self.pel_dead = 0

    # ── consumers ──

    @always_inline
    def ncons(self) -> Int:
        """Live consumers."""
        return len(self.by_name)

    def _name_pos(self, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
        """The first by_name position whose name is >= p[:n] (byte order,
        shorter first, as Redis's rax keeps consumer names)."""
        var lo = 0
        var hi = len(self.by_name)
        while lo < hi:
            var mid = (lo + hi) // 2
            if _bytes_cmp(self.consumers[self.by_name[mid]].name, p, n) < 0:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def consumer_index(self, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
        """The slot of the live consumer with this name, or -1."""
        var k = self._name_pos(p, n)
        if k < len(self.by_name) and _bytes_cmp(self.consumers[self.by_name[k]].name, p, n) == 0:
            return self.by_name[k]
        return -1

    @always_inline
    def consumer_by_id(self, id: Int) -> Int:
        """The slot of the live consumer with this id, or -1 (PEL_UNOWNED and
        PEL_DEAD never match)."""
        var k = id - 1
        if k >= 0 and k < len(self.consumers) and self.consumers[k].alive:
            return k
        return -1

    def add_consumer(mut self, p: Pointer[UInt8, MutUntrackedOrigin], n: Int, now_ms: Int64) -> Int:
        """The consumer with this name, created if missing; its slot."""
        var k = self._name_pos(p, n)
        if k < len(self.by_name) and _bytes_cmp(self.consumers[self.by_name[k]].name, p, n) == 0:
            return self.by_name[k]
        var name = List[UInt8](capacity=n)
        for b in range(n):
            name.append(p[b])
        var slot = len(self.consumers)
        self.consumers.append(StreamConsumer(slot + 1, name^, now_ms))
        if k == len(self.by_name):
            self.by_name.append(slot)
        else:
            self.by_name.insert(k, slot)
        return slot

    def delete_consumer(mut self, slot: Int) -> Int:
        """Remove a live consumer and its pending entries (Redis's
        streamDelConsumer); how many it had pending. Slots of the consumer
        list may be renumbered: a slot held across this call is stale."""
        var id = self.consumers[slot].id
        var removed = self.consumers[slot].pending
        if removed > 0:
            for k in range(len(self.pel)):
                if self.pel[k].consumer == id:
                    self.pel[k].consumer = PEL_DEAD
                    self.pel_dead += 1
        var nm = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(self.consumers[slot].name.unsafe_ptr()))
        var at = self._name_pos(nm, len(self.consumers[slot].name))
        if at < len(self.by_name) and self.by_name[at] == slot:
            _ = self.by_name.pop(at)
        self.consumers[slot].alive = False
        self.consumers[slot].pending = 0
        self.consumers[slot].name = List[UInt8]()
        self.consumers_dead += 1
        self.pel_tidy()
        self._consumers_tidy()
        return removed

    def _consumers_tidy(mut self):
        """Compact the consumer slots once deleted ones are most of them:
        the live ones are renumbered in name order and their pending entries
        follow."""
        if self.consumers_dead <= 32 or self.consumers_dead * 2 < len(self.consumers):
            return
        var new_id = List[Int](capacity=len(self.consumers) + 1)
        for _ in range(len(self.consumers) + 1):
            new_id.append(PEL_UNOWNED)
        var out = List[StreamConsumer](capacity=len(self.by_name))
        for k in range(len(self.by_name)):
            var old = self.by_name[k]
            new_id[self.consumers[old].id] = k + 1
            var c = self.consumers[old].copy()
            c.id = k + 1
            out.append(c^)
            self.by_name[k] = k
        for k in range(len(self.pel)):
            var o = self.pel[k].consumer
            if o > 0:
                self.pel[k].consumer = new_id[o]
        self.consumers = out^
        self.consumers_dead = 0


def _bytes_cmp(a: List[UInt8], p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    """memcmp order, shorter first on a common prefix."""
    var m = len(a) if len(a) < n else n
    for k in range(m):
        if a[k] != p[k]:
            return -1 if a[k] < p[k] else 1
    if len(a) == n:
        return 0
    return -1 if len(a) < n else 1


# ── StreamEntry: one entry in the stream ──
struct StreamEntry(Copyable, Movable, ImplicitlyCopyable):
    var id_ms: UInt64
    var id_seq: UInt64
    var data: Pointer[UInt8, MutUntrackedOrigin]  # packed field-value pairs: [u32 flen][bytes][u32 vlen][bytes]...
    var data_len: Int
    var num_fields: Int
    var deleted: Bool

    def __init__(out self):
        self.id_ms = 0; self.id_seq = 0
        self.data = null_ptr[UInt8, MutUntrackedOrigin]()
        self.data_len = 0; self.num_fields = 0; self.deleted = False


# ── StreamData: the stream itself ──
struct StreamData(Copyable, Movable):
    var entries: Pointer[StreamEntry, MutUntrackedOrigin]
    var count: Int
    var alive: Int
    var capacity: Int
    var last_id_ms: UInt64
    var last_id_seq: UInt64
    # #40: Redis 7's stream metadata, and the consumer groups
    var entries_added: Int64       # every XADD ever (XSETID ENTRIESADDED sets it)
    var max_del_ms: UInt64         # the largest id XDEL deleted (max-deleted-entry-id)
    var max_del_seq: UInt64
    var groups: List[StreamGroup]  # in creation order

    def __init__(out self):
        self.capacity = 64
        self.entries = alloc[StreamEntry](self.capacity)
        self.count = 0; self.alive = 0
        self.last_id_ms = 0; self.last_id_seq = 0
        self.entries_added = 0
        self.max_del_ms = 0; self.max_del_seq = 0
        self.groups = List[StreamGroup]()

    def append(mut self, id_ms: UInt64, id_seq: UInt64, data: Pointer[UInt8, MutUntrackedOrigin], data_len: Int, num_fields: Int):
        if self.count >= self.capacity:
            var new_cap = self.capacity * 2
            var new_entries = alloc[StreamEntry](new_cap)
            for i in range(self.count):
                (new_entries.unsafe_offset(i)).unsafe_write(self.entries[unsafe_offset=i])
            self.entries.unsafe_free()
            self.entries = new_entries
            self.capacity = new_cap
        var e = StreamEntry()
        e.id_ms = id_ms; e.id_seq = id_seq
        e.data = data; e.data_len = data_len
        e.num_fields = num_fields; e.deleted = False
        (self.entries.unsafe_offset(self.count)).unsafe_write(e)
        self.count += 1; self.alive += 1
        self.last_id_ms = id_ms; self.last_id_seq = id_seq
        self.entries_added += 1

    def kill(mut self, i: Int):
        """Delete entry `i` (XDEL, XTRIM, XADD MAXLEN, their replays) and free
        its field data (gh #394). Entries used to be only FLAGGED deleted, so a
        capped stream kept every trimmed entry's data forever — a MAXLEN log
        grew without bound — and never shrank `entries` either."""
        if self.entries[unsafe_offset=i].deleted:
            return
        self.entries[unsafe_offset=i].deleted = True
        var d = self.entries[unsafe_offset=i].data
        if Int(d) != 0:
            d.unsafe_free()
        self.entries[unsafe_offset=i].data = null_ptr[UInt8, MutUntrackedOrigin]()
        self.alive -= 1

    def compact(mut self):
        """Squeeze out deleted entries once they are the majority, keeping the
        order. Call after a batch of kill()s; nothing holds an entry INDEX
        across commands (readers resume from an id), so moving is safe. Keeps
        a capped stream's `entries` bounded and its trim scans short — the
        MAXLEN trim walks from index 0 over every tombstone ever made."""
        var dead = self.count - self.alive
        if dead < 64 or dead * 2 < self.count:
            return
        var w = 0
        for r in range(self.count):
            if not self.entries[unsafe_offset=r].deleted:
                if w != r:
                    self.entries[unsafe_offset=w] = self.entries[unsafe_offset=r]
                w += 1
        self.count = w

    def release(mut self):
        """Free everything this stream owns — for DEL and every other path that
        drops a stream key (gh #394: none did, ~3 KB per stream)."""
        for i in range(self.count):
            var d = self.entries[unsafe_offset=i].data
            if Int(d) != 0:
                d.unsafe_free()
        if Int(self.entries) != 0:
            self.entries.unsafe_free()
        self.entries = null_ptr[StreamEntry, MutUntrackedOrigin]()
        self.count = 0; self.alive = 0; self.capacity = 0
        self.groups = List[StreamGroup]()   # #40: the groups go too

    def deep_copy(self) -> Self:
        """A stream sharing nothing with this one — for COPY. Live entries only;
        the id high-water mark is kept, so new ids stay monotonic."""
        var out = Self()
        for i in range(self.count):
            var e = self.entries[unsafe_offset=i]
            if e.deleted:
                continue
            var d = alloc[UInt8](max(e.data_len, 1))
            for b in range(e.data_len):
                d[unsafe_offset=b] = e.data[unsafe_offset=b]
            out.append(e.id_ms, e.id_seq, d, e.data_len, e.num_fields)
        out.last_id_ms = self.last_id_ms
        out.last_id_seq = self.last_id_seq
        # #40: and its metadata and consumer groups, as Redis's COPY copies them
        out.entries_added = self.entries_added
        out.max_del_ms = self.max_del_ms
        out.max_del_seq = self.max_del_seq
        out.groups = self.groups.copy()
        return out^

    # ── #40: lookups and Redis 7's group arithmetic ──

    def lower_bound(self, ms: UInt64, seq: UInt64) -> Int:
        """The first entry index (live or deleted) with an id >= (ms, seq)."""
        var lo = 0
        var hi = self.count
        while lo < hi:
            var mid = (lo + hi) // 2
            var e = self.entries[unsafe_offset=mid]
            if sid_lt(e.id_ms, e.id_seq, ms, seq):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def find_live(self, ms: UInt64, seq: UInt64) -> Int:
        """The index of the live entry with this id, or -1."""
        var k = self.lower_bound(ms, seq)
        if k < self.count:
            var e = self.entries[unsafe_offset=k]
            if e.id_ms == ms and e.id_seq == seq and not e.deleted:
                return k
        return -1

    def entry_referenced(self, ms: UInt64, seq: UInt64) -> Bool:
        """Redis's streamEntryIsReferenced: a group has not read this entry
        yet (its id is past the group's last-delivered id) or holds it
        pending. ACKED deletes only what no group references."""
        for g in range(len(self.groups)):
            if sid_lt(self.groups[g].last_ms, self.groups[g].last_seq, ms, seq):
                return True
        for g in range(len(self.groups)):
            if self.groups[g].pel_find(ms, seq) >= 0:
                return True
        return False

    def first_live(self) -> Int:
        """The index of the first live entry, or -1."""
        for k in range(self.count):
            if not self.entries[unsafe_offset=k].deleted:
                return k
        return -1

    def last_live(self) -> Int:
        var k = self.count - 1
        while k >= 0:
            if not self.entries[unsafe_offset=k].deleted:
                return k
            k -= 1
        return -1

    def first_id(self, mut ms: UInt64, mut seq: UInt64):
        """recorded-first-entry-id: the first live entry's id, 0-0 when empty."""
        var k = self.first_live()
        if k < 0:
            ms = 0
            seq = 0
        else:
            ms = self.entries[unsafe_offset=k].id_ms
            seq = self.entries[unsafe_offset=k].id_seq

    def group_index(self, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
        for k in range(len(self.groups)):
            if len(self.groups[k].name) == n:
                var same = True
                for b in range(n):
                    if self.groups[k].name[b] != p[b]:
                        same = False
                        break
                if same:
                    return k
        return -1

    def range_has_tombstones(self, start_ms: UInt64, start_seq: UInt64) -> Bool:
        """Does an XDEL tombstone lie at or after `start` (to the end)?"""
        if self.alive == 0 or (self.max_del_ms == 0 and self.max_del_seq == 0):
            return False
        return not sid_lt(self.max_del_ms, self.max_del_seq, start_ms, start_seq)

    def distance_from_first_ever(self, ms: UInt64, seq: UInt64) -> Int64:
        """An id's logical read counter, when it can be known: how many
        entries were added up to and including it. SCG_INVALID_ENTRIES_READ
        when not (an id between the current first and last entries, a future
        one, or one before a tombstone)."""
        if self.entries_added == 0:
            return 0
        if self.alive == 0 and not sid_lt(self.last_id_ms, self.last_id_seq, ms, seq):
            return self.entries_added
        if not (ms == 0 and seq == 0) and sid_lt(ms, seq, self.max_del_ms, self.max_del_seq):
            return SCG_INVALID_ENTRIES_READ
        var cmp_last = sid_cmp(ms, seq, self.last_id_ms, self.last_id_seq)
        if cmp_last == 0:
            return self.entries_added
        if cmp_last > 0:
            return SCG_INVALID_ENTRIES_READ
        var f_ms = UInt64(0)
        var f_seq = UInt64(0)
        self.first_id(f_ms, f_seq)
        var cmp_first = sid_cmp(ms, seq, f_ms, f_seq)
        var no_del = self.max_del_ms == 0 and self.max_del_seq == 0
        if no_del or sid_lt(self.max_del_ms, self.max_del_seq, f_ms, f_seq):
            if cmp_first < 0:
                return self.entries_added - Int64(self.alive)
            if cmp_first == 0:
                return self.entries_added - Int64(self.alive) + 1
        return SCG_INVALID_ENTRIES_READ

    def group_lag(self, g: Int, mut valid: Bool) -> Int64:
        """XINFO's lag: the entries not yet delivered to group g; `valid`
        False when that cannot be known (it answers nil)."""
        valid = True
        if self.entries_added == 0 or self.alive == 0:
            return 0
        var f_ms = UInt64(0)
        var f_seq = UInt64(0)
        self.first_id(f_ms, f_seq)
        ref grp = self.groups[g]
        if sid_lt(grp.last_ms, grp.last_seq, f_ms, f_seq) and sid_lt(self.max_del_ms, self.max_del_seq, f_ms, f_seq):
            return Int64(self.alive)
        if grp.entries_read != SCG_INVALID_ENTRIES_READ and not self.range_has_tombstones(grp.last_ms, grp.last_seq):
            return self.entries_added - grp.entries_read
        var er = self.distance_from_first_ever(grp.last_ms, grp.last_seq)
        if er != SCG_INVALID_ENTRIES_READ:
            return self.entries_added - er
        valid = False
        return 0

    def advance_group(mut self, g: Int, ms: UInt64, seq: UInt64):
        """Group g delivered (ms, seq), past its last-delivered-id: move that
        and its entries-read on, as Redis's streamReplyWithRange does."""
        var f_ms = UInt64(0)
        var f_seq = UInt64(0)
        self.first_id(f_ms, f_seq)
        if self.groups[g].entries_read != SCG_INVALID_ENTRIES_READ \
           and not sid_lt(self.groups[g].last_ms, self.groups[g].last_seq, f_ms, f_seq) \
           and not self.range_has_tombstones(self.groups[g].last_ms, self.groups[g].last_seq):
            self.groups[g].entries_read += 1
        elif self.entries_added != 0:
            self.groups[g].entries_read = self.distance_from_first_ever(ms, seq)
        self.groups[g].last_ms = ms
        self.groups[g].last_seq = seq


# ── #40: the WAL / snapshot records of stream metadata and consumer groups ──
#
# Effect records (src/io/wal.mojo applies them; src/io/snapshot.mojo writes
# a stream's groups as the same records):
#   38 group create   [u32 glen][group][u64 last ms][u64 last seq][i64 entries-read]
#   39 group set id   the same layout
#   40 group destroy  [u32 glen][group]
#   41 consumer       [u32 glen][group][u32 clen][consumer][i64 seen][i64 active]
#   42 del consumer   [u32 glen][group][u32 clen][consumer]
#   43 pending entry  [u32 glen][group][u32 clen][consumer][u64 ms][u64 seq][i64 time][i64 count]
#   44 pending gone   [u32 glen][group][u64 ms][u64 seq]
#   45 stream meta    [u64 last ms][u64 last seq][i64 entries-added][u64 max-del ms][u64 max-del seq]
# All little-endian (35-37 are FUNCTION's, #36). 45 creates the stream when it
# is missing, so an empty stream (XGROUP CREATE MKSTREAM, XDEL of its last
# entry) survives a reload.

def _put_u32(mut out: List[UInt8], v: Int):
    for k in range(4):
        out.append(UInt8((v >> (8 * k)) & 0xFF))


def _put_u64(mut out: List[UInt8], v: UInt64):
    for k in range(8):
        out.append(UInt8((v >> UInt64(8 * k)) & 0xFF))


def _put_bytes(mut out: List[UInt8], p: Pointer[UInt8, MutUntrackedOrigin], n: Int):
    _put_u32(out, n)
    for k in range(n):
        out.append(p[k])


def _put_list(mut out: List[UInt8], b: List[UInt8]):
    _put_u32(out, len(b))
    for k in range(len(b)):
        out.append(b[k])


def encode_group_rec(g: StreamGroup) -> List[UInt8]:
    var out = List[UInt8]()
    _put_list(out, g.name)
    _put_u64(out, g.last_ms)
    _put_u64(out, g.last_seq)
    _put_u64(out, UInt64(g.entries_read))
    return out^


def encode_group_name_rec(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    _put_bytes(out, p, n)
    return out^


def encode_consumer_rec(group: List[UInt8], c: StreamConsumer) -> List[UInt8]:
    var out = List[UInt8]()
    _put_list(out, group)
    _put_list(out, c.name)
    _put_u64(out, UInt64(c.seen_time))
    _put_u64(out, UInt64(c.active_time))
    return out^


def encode_delconsumer_rec(gp: Pointer[UInt8, MutUntrackedOrigin], gn: Int,
                           cp: Pointer[UInt8, MutUntrackedOrigin], cn: Int) -> List[UInt8]:
    var out = List[UInt8]()
    _put_bytes(out, gp, gn)
    _put_bytes(out, cp, cn)
    return out^


def encode_nack_rec(group: List[UInt8], consumer: List[UInt8], ms: UInt64, seq: UInt64,
                    delivery_time: Int64, delivery_count: Int64) -> List[UInt8]:
    var out = List[UInt8]()
    _put_list(out, group)
    _put_list(out, consumer)
    _put_u64(out, ms)
    _put_u64(out, seq)
    _put_u64(out, UInt64(delivery_time))
    _put_u64(out, UInt64(delivery_count))
    return out^


def encode_pel_del_rec(group: List[UInt8], ms: UInt64, seq: UInt64) -> List[UInt8]:
    var out = List[UInt8]()
    _put_list(out, group)
    _put_u64(out, ms)
    _put_u64(out, seq)
    return out^


def encode_meta_rec(sd: Pointer[StreamData, MutUntrackedOrigin]) -> List[UInt8]:
    var out = List[UInt8]()
    _put_u64(out, sd[].last_id_ms)
    _put_u64(out, sd[].last_id_seq)
    _put_u64(out, UInt64(sd[].entries_added))
    _put_u64(out, sd[].max_del_ms)
    _put_u64(out, sd[].max_del_seq)
    return out^


# Record readers, for the replay. Each checks its bounds; `at` advances.

def rec_u32(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut at: Int, mut ok: Bool) -> Int:
    if at + 4 > n:
        ok = False
        return 0
    var v = Int(p[at]) | (Int(p[at + 1]) << 8) | (Int(p[at + 2]) << 16) | (Int(p[at + 3]) << 24)
    at += 4
    return v


def rec_u64(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut at: Int, mut ok: Bool) -> UInt64:
    if at + 8 > n:
        ok = False
        return 0
    var v = UInt64(0)
    for k in range(8):
        v |= UInt64(Int(p[at + k])) << UInt64(8 * k)
    at += 8
    return v


def rec_name(p: Pointer[UInt8, MutUntrackedOrigin], n: Int, mut at: Int, mut ok: Bool,
             mut name_at: Int) -> Int:
    """A length-prefixed name inside the record: its offset in `name_at`, its
    length returned. The bytes are read in place: `p` belongs to the caller
    and outlives the apply (a List copied out of it would not, if its last
    use came before a call that reads it through a raw pointer)."""
    var l = rec_u32(p, n, at, ok)
    if not ok or l < 0 or at + l > n:
        ok = False
        return 0
    name_at = at
    at += l
    return l


def apply_stream_group_record(cmd_id: UInt8, sd: Pointer[StreamData, MutUntrackedOrigin],
                              p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    """Apply record 38-45 to a stream (the caller resolved or created it).
    False for a malformed record, which changes nothing."""
    var ok = True
    var at = 0
    if cmd_id == 45:
        var lm = rec_u64(p, n, at, ok)
        var ls = rec_u64(p, n, at, ok)
        var ea = rec_u64(p, n, at, ok)
        var dm = rec_u64(p, n, at, ok)
        var ds = rec_u64(p, n, at, ok)
        if not ok:
            return False
        sd[].last_id_ms = lm
        sd[].last_id_seq = ls
        sd[].entries_added = Int64(ea)
        sd[].max_del_ms = dm
        sd[].max_del_seq = ds
        return True
    var g_at = 0
    var gl = rec_name(p, n, at, ok, g_at)
    if not ok:
        return False
    var gp = p + g_at
    var g = sd[].group_index(gp, gl)
    if cmd_id == 38 or cmd_id == 39:
        var lm = rec_u64(p, n, at, ok)
        var ls = rec_u64(p, n, at, ok)
        var er = rec_u64(p, n, at, ok)
        if not ok:
            return False
        if g < 0:
            var name = List[UInt8](capacity=gl)
            for b in range(gl):
                name.append(gp[b])
            sd[].groups.append(StreamGroup(name^, lm, ls, Int64(er)))
        else:
            sd[].groups[g].last_ms = lm
            sd[].groups[g].last_seq = ls
            sd[].groups[g].entries_read = Int64(er)
        return True
    if cmd_id == 40:
        if g >= 0:
            _ = sd[].groups.pop(g)
        return True
    if g < 0:
        return False
    if cmd_id == 44:
        var ms = rec_u64(p, n, at, ok)
        var seq = rec_u64(p, n, at, ok)
        if not ok:
            return False
        var pk = sd[].groups[g].pel_find(ms, seq)
        if pk >= 0:
            sd[].groups[g].remove_nack_at(pk)
            sd[].groups[g].pel_tidy()
        return True
    var c_at = 0
    var cl = rec_name(p, n, at, ok, c_at)
    if not ok:
        return False
    var cp = p + c_at
    if cmd_id == 42:
        var ci = sd[].groups[g].consumer_index(cp, cl)
        if ci >= 0:
            _ = sd[].groups[g].delete_consumer(ci)
        return True
    if cmd_id == 41:
        var seen = rec_u64(p, n, at, ok)
        var active = rec_u64(p, n, at, ok)
        if not ok:
            return False
        var ci = sd[].groups[g].add_consumer(cp, cl, Int64(seen))
        sd[].groups[g].consumers[ci].seen_time = Int64(seen)
        sd[].groups[g].consumers[ci].active_time = Int64(active)
        sd[].groups[g].consumers[ci].logged_seen = Int64(seen)
        return True
    if cmd_id == 43:
        var ms = rec_u64(p, n, at, ok)
        var seq = rec_u64(p, n, at, ok)
        var dt = rec_u64(p, n, at, ok)
        var dc = rec_u64(p, n, at, ok)
        if not ok:
            return False
        var ci = sd[].groups[g].add_consumer(cp, cl, Int64(dt))
        sd[].groups[g].set_nack(ms, seq, sd[].groups[g].consumers[ci].id, Int64(dt), Int64(dc))
        return True
    return False
