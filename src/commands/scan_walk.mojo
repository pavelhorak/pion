"""#50: SCAN, HSCAN, SSCAN and ZSCAN walk their table a step at a time.

SCAN returned the whole keyspace in one reply with cursor 0, whatever COUNT
said, and the other three did the same with a collection; past a few hundred
thousand keys the reply alone ran into #49. Redis walks its dict a bucket at a
time and hands back a cursor where the walk stopped.

Here the cursor is a slot. A Swiss-table entry never moves while the table
lives (an insert takes a free slot, a delete leaves a tombstone), so walking
slots in order returns every entry present from the first call to the last
exactly once. A rebuild (_rehash, on growth or to purge tombstones) moves
every entry, so the cursor also carries the table's rebuild count: a walk
that finds it changed starts that table over. That can return an entry
twice, which SCAN allows, and never misses one.

    keyspace (8 shards):   cursor = slot << 11 | (rehashes & 0xFF) << 3 | shard
    one collection:        cursor = slot << 8  | (rehashes & 0xFF)

A call stops once it has found COUNT live entries (10 by default), and sets
the cursor to the next slot — so a dense table paginates by COUNT, as in
Redis. If it reaches the end of the table first it returns cursor 0, so a
small or sparse keyspace comes back in one call even though Pion's table is
millions of slots preallocated (Redis's dict grows with the data and is tiny
when the data is). MATCH and TYPE filter the found entries AFTER the COUNT
cut, as in Redis, so a page can be short or empty while the walk goes on.
Empty slots are skipped 16 at a time (SIMD), so walking a sparse table to its
end is cheap. A collection of at most SCAN_SMALL entries comes back whole.
"""
from std.memory.unsafe_pointer import Pointer
from std.collections import List
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.utils import format_int_to_buf, int_string_len
from src.network.response_writer import ResponseWriter

comptime SCAN_SMALL = 128


def scan_slots(m: Pointer[SlabHashMap, MutUntrackedOrigin], start: Int, want: Int,
               mut out: List[Int]) -> Int:
    """Append to `out` the live slots of `m` from `start`, until `want` are
    found or the table ends. Returns the slot to resume from: `m[].capacity`
    once the table is done. Empty slots are skipped in SIMD strides."""
    var cap = m[].capacity
    var s = start
    var found = 0
    while s < cap and found < want:
        var nxt = m[].next_live(s, cap)
        if nxt >= cap:
            return cap
        out.append(nxt)
        found += 1
        s = nxt + 1
    return s


@always_inline
def keyspace_cursor(shard: Int, slot: Int, rehashes: Int) -> Int:
    return (slot << 11) | ((rehashes & 0xFF) << 3) | shard


@always_inline
def map_cursor(slot: Int, rehashes: Int) -> Int:
    return (slot << 8) | (rehashes & 0xFF)


def append_scan_header(mut writer: ResponseWriter, cursor: Int, n: Int):
    """`*2`, the cursor as a bulk string, then the header of `n` elements."""
    writer.append_array_header(2)
    writer.append_bulk_string_response_header(int_string_len(Int64(cursor)))
    writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(cursor))
    writer.buffer[unsafe_offset=writer.offset] = 13
    writer.buffer[unsafe_offset=writer.offset + 1] = 10
    writer.offset += 2
    writer.append_array_header(n)


def walk_map(m: Pointer[SlabHashMap, MutUntrackedOrigin], cursor: Int, count: Int,
             mut slots: List[Int]) -> Int:
    """One HSCAN/SSCAN/ZSCAN step over a collection's table: its live slots
    from `cursor` into `slots`; returns the next cursor (0 when done). A
    small collection is returned whole with cursor 0."""
    if m[].size <= SCAN_SMALL:
        _ = scan_slots(m, 0, m[].capacity + 1, slots)
        return 0
    var slot = cursor >> 8
    if cursor != 0 and (cursor & 0xFF) != (m[].rehashes & 0xFF):
        slot = 0                                  # rebuilt since: walk it again
    var nxt = scan_slots(m, slot, count, slots)
    if nxt >= m[].capacity:
        return 0
    return map_cursor(nxt, m[].rehashes)


def walk_keyspace(ks: Pointer[StripedHashMap, MutUntrackedOrigin], cursor: Int, count: Int,
                  mut shards: List[Int], mut slots: List[Int]) -> Int:
    """One SCAN step over the 8-shard keyspace: live (shard, slot) pairs from
    `cursor` into `shards`/`slots`, before MATCH/TYPE/expiry filtering (those
    may drop all of them, as in Redis). The cursor is `slot<<11 | rehashes<<3
    | shard`; a shard rebuilt since the last call is walked from its start.
    Returns the next cursor, 0 when all 8 shards are done. COUNT bounds the
    live entries found; a shard with none left rolls to the next."""
    var shard = cursor & 0x7
    var slot = cursor >> 11
    var want = count
    while shard < 8:
        var sp = ks[].shards.unsafe_offset(shard)
        if slot != 0 and ((cursor >> 3) & 0xFF) != (sp[].rehashes & 0xFF):
            slot = 0                              # rebuilt since: this shard again
        var before = len(slots)
        var nxt = scan_slots(sp, slot, want, slots)
        for _ in range(len(slots) - before):
            shards.append(shard)
        want -= len(slots) - before
        if nxt < sp[].capacity:
            return keyspace_cursor(shard, nxt, sp[].rehashes)
        shard += 1
        slot = 0
        if want <= 0:
            if shard >= 8:
                return 0
            return keyspace_cursor(shard, 0, ks[].shards[unsafe_offset=shard].rehashes)
    return 0
