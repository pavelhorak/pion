"""Eviction: what `--maxmemory` does over the limit under a policy other than
noeviction. Redis's policies except the LRM pair, the same sampled approximation of LRU and
LFU (`maxmemory-samples` keys per eviction, a pool of the 16 best victims seen),
and -OOM when nothing is left to evict.

**What the limit is on.** RSS, as for noeviction: it is what the OS kills on.
Redis limits the bytes its allocator handed out, and that is the number a
controller wants, because freeing memory does not lower RSS. The allocator keeps
freed pages to reuse them. So "evict while RSS is over the limit" never stops
until the keyspace is empty, and a controller that evicts only as RSS grows
falls behind: free a multiple of the growth, and the free memory lasts that many
intervals, after which RSS grows again by a step. RSS creeps up without bound.
Pion cannot read the allocator instead: on Linux it is a tcmalloc inside Mojo's
runtime that exports no statistics.

**So two loops, each steering by what it can see.**
- The inner loop holds the DATA to a target. It estimates the data as live keys
  times their mean size (MEMORY USAGE's estimate of a running sample of keys) and
  evicts down to the target. Data, unlike RSS, falls when a key goes, so this
  loop is stable, and a read-only workload at the limit evicts nothing.
- The outer loop moves the target by RSS. At the first crossing the target is the
  data less the share of the heap that is over. Above the limit, every byte RSS still grows (table
  growth, fragmentation, a list the sample missed) comes off the target, each
  byte once: only a new peak counts, because RSS wobbles. Below 90% of the limit, half the gap goes back on it, so memory the allocator did
  return gets used again.

The estimate is only as good as MEMORY USAGE's, and it undercounts streams and
vector sets. That costs accuracy, not the limit: the outer loop answers for
whatever the inner one cannot see.

**Where it runs.** Between recv batches, in the event loop's tick, never on the
command path: a hot path under noeviction is unchanged, one under eviction stays
fast, and a burst passes the limit by at most what one tick wrote.

**What an eviction does.** What an expiry does (StripedHashMap.expire_key): a WAL
DEL, so a restart does not revive the key and a replica drops it; the TTL entry
goes; the container is freed; a vector document leaves its index. Also what a
write does for WATCH. A replica does not evict, as Redis's
`replica-ignore-maxmemory yes` default: it applies its primary's DELs.
"""
from std.ffi import external_call
from std.memory import alloc
from std.memory.unsafe_pointer import Pointer
from src.common.ptr import is_null, is_not_null
from src.common.value import GenericValue, ValueType
from src.common.hash_map import (
    SlabHashMap, StripedHashMap, EvictState, EvictPoolEntry, evict_tag,
    POLICY_NOEVICTION, POLICY_ALLKEYS_LRU, POLICY_VOLATILE_LRU, POLICY_ALLKEYS_LFU,
    POLICY_VOLATILE_LFU, POLICY_ALLKEYS_RANDOM, POLICY_VOLATILE_RANDOM, POLICY_VOLATILE_TTL,
    LRU_SHIFT, LRU_MASK, EVICT_TABLE_FILL, evict_policy_name,
)
from src.common.memory_usage import memory_usage
from src.common.container_free import free_container
from src.commands.transaction import TransactionState
from src.io.wal import WAL
from src.network.fast_path import _get_now_ns

# Redis 8.10's list also names volatile-lrm and allkeys-lrm (least recently
# MODIFIED). Pion has neither: a read and a write reach touch() the same way.
comptime EVICT_POLICY_ERROR = "ERR CONFIG SET failed (possibly related to argument 'maxmemory-policy') - argument(s) must be one of the following: volatile-lru, volatile-lfu, volatile-random, volatile-ttl, allkeys-lru, allkeys-lfu, allkeys-random, noeviction"
comptime EVICT_POOL_SIZE = 16
# One tick's work: a housekeeping tick must not stall the event loop for long.
# What it leaves, the next tick does.
comptime EVICT_MAX_PER_TICK = 16384
comptime SIZE_SAMPLES_PER_TICK = 32
# What a slot of a keyspace table costs: metadata, key, value, access entry.
comptime SLOT_BYTES = 69


def evict_policy_from_name(ptr: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Int:
    """The policy with this name, in any case as Redis takes it; -1 if none."""
    for p in range(8):
        var name = evict_policy_name(p)
        if name.byte_length() != length:
            continue
        var same = True
        for i in range(length):
            var b = ptr[unsafe_offset=i]
            if b >= 65 and b <= 90:
                b |= 0x20
            if b != name.unsafe_ptr()[unsafe_offset=i]:
                same = False
                break
        if same:
            return p
    return -1


def evict_policy_from_string(s: String) -> Int:
    # On the heap: a short String keeps its bytes inline, on the stack, and a
    # stack pointer passed out of line as MutUntrackedOrigin is miscompiled (gh #349).
    var n = s.byte_length()
    var h = alloc[UInt8](n + 1)
    for k in range(n):
        h[unsafe_offset=k] = s.as_bytes()[k]
    var p = evict_policy_from_name(h, n)
    h.unsafe_free()
    return p


@always_inline
def _is_volatile(p: Int) -> Bool:
    return p == POLICY_VOLATILE_LRU or p == POLICY_VOLATILE_LFU \
        or p == POLICY_VOLATILE_RANDOM or p == POLICY_VOLATILE_TTL


def _apply_policy(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
                  tm: Pointer[SlabHashMap, MutUntrackedOrigin],
                  st: Pointer[EvictState, MutUntrackedOrigin], policy: Int):
    """CONFIG SET reached this worker: the access tables follow the policy. A
    change between LRU and LFU starts them over, since their stamps mean
    different things."""
    var lfu = policy == POLICY_ALLKEYS_LFU or policy == POLICY_VOLATILE_LFU
    var tracked = lfu or policy == POLICY_ALLKEYS_LRU or policy == POLICY_VOLATILE_LRU
    for i in range(8):
        var m = ks[].shards.unsafe_offset(i)
        if policy == POLICY_NOEVICTION or lfu != st[].lfu:
            m[].disable_access_table()
        if policy != POLICY_NOEVICTION:
            m[].enable_access_table()
    if is_not_null(tm):
        if policy == POLICY_NOEVICTION:
            tm[].disable_access_table()
        else:
            tm[].enable_access_table()
    st[].policy = policy
    st[].tracked = tracked
    st[].lfu = lfu
    st[].target = -1
    st[].pool_n = 0
    ks[].track_access = tracked


def _random_live(m: Pointer[SlabHashMap, MutUntrackedOrigin],
                 st: Pointer[EvictState, MutUntrackedOrigin]) -> Int:
    """A random live slot of this table, -1 when it has none. Slots are looked
    at from a random point; after a few sparse windows it walks the whole table,
    16 slots a load."""
    if m[].size == 0:
        return -1
    for _ in range(4):
        var start = Int(st[].next_rand() & UInt64(m[].capacity - 1))
        var limit = start + 256
        var bound = limit if limit < m[].capacity else m[].capacity
        var slot = m[].next_live(start, limit)
        if slot < bound:
            return slot
    var slot = m[].next_live(0, m[].capacity)
    return slot if slot < m[].capacity else -1


def _random_key_slot(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
                     st: Pointer[EvictState, MutUntrackedOrigin]) -> Int:
    """A random key: `shard << 48 | slot`, -1 when the keyspace is empty."""
    var first = Int(st[].next_rand() & 7)
    for i in range(8):
        var s = (first + i) & 7
        var slot = _random_live(ks[].shards.unsafe_offset(s), st)
        if slot >= 0:
            return (s << 48) | slot
    return -1


def _random_volatile_slot(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
                          tm: Pointer[SlabHashMap, MutUntrackedOrigin],
                          st: Pointer[EvictState, MutUntrackedOrigin]) -> Int:
    """A random key with a TTL, by where it lives in the keyspace; -1 if none
    was found (no TTLs, or the sampled entry's key is gone)."""
    if is_null(tm):
        return -1
    var t = _random_live(tm, st)
    if t < 0:
        return -1
    var key = tm[].keys[unsafe_offset=t]
    var h = UInt64(key.__hash__())
    var s = ks[]._shard(h)
    var slot = ks[].shards[unsafe_offset=s].find_slot(key, h)
    return (s << 48) | slot if slot >= 0 else -1


def _remove(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
            st: Pointer[EvictState, MutUntrackedOrigin],
            versions: Pointer[UInt64, MutUntrackedOrigin], s: Int, slot: Int):
    var m = ks[].shards.unsafe_offset(s)
    var table = m[].access_table()
    if Int(table) != 0:
        table[unsafe_offset=slot] = 0
    var key = m[].keys[unsafe_offset=slot]
    var h = UInt64(key.__hash__())
    versions[unsafe_offset=TransactionState.key_slot_from_hash(h)] += 1
    # `key` is the map's own: expire_key is done with it when the removal frees
    # it. It takes care of the WAL DEL and the TTL entry.
    var gone = ks[].expire_key(key, h, False)
    if gone.type.value != ValueType.NONE:
        free_container(gone)
    st[].evicted += 1


def _score(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
           tm: Pointer[SlabHashMap, MutUntrackedOrigin],
           st: Pointer[EvictState, MutUntrackedOrigin],
           m: Pointer[SlabHashMap, MutUntrackedOrigin],
           slot: Int, h: UInt64, now_ns: Int64) -> Int64:
    """How good a victim: higher is better."""
    if st[].policy == POLICY_VOLATILE_TTL:
        var dv = tm[].get_with_hash(m[].keys[unsafe_offset=slot], h)
        if dv.type.value == ValueType.NONE:
            return Int64.MIN                 # its TTL went since it was sampled
        return Int64.MAX - Int64(dv.as_int())
    var e = m[].access_table()[unsafe_offset=slot]
    # Not this key's entry. Reads and writes both stamp, so it has not been
    # touched since tracking began: it was there before the policy was set (or
    # came back from the WAL), and its true age is unknown and large.
    if (e & 0xFF000000) != evict_tag(h):
        return Int64(LRU_MASK + 1)
    if st[].lfu:
        return Int64(255 - st[].lfu_decayed(e, now_ns))
    var now_units = Int((now_ns >> LRU_SHIFT) & LRU_MASK)
    return Int64((now_units - Int(e & UInt32(LRU_MASK))) & LRU_MASK)


def _pool_add(st: Pointer[EvictState, MutUntrackedOrigin], s: Int, slot: Int, tag: UInt32, score: Int64):
    var pool = st[].pool
    var worst = 0
    for i in range(st[].pool_n):
        if pool[unsafe_offset=i].shard == s and pool[unsafe_offset=i].slot == slot:
            pool[unsafe_offset=i] = EvictPoolEntry(s, slot, tag, score)
            return
        if pool[unsafe_offset=i].score < pool[unsafe_offset=worst].score:
            worst = i
    if st[].pool_n < EVICT_POOL_SIZE:
        pool[unsafe_offset=st[].pool_n] = EvictPoolEntry(s, slot, tag, score)
        st[].pool_n += 1
    elif score > pool[unsafe_offset=worst].score:
        pool[unsafe_offset=worst] = EvictPoolEntry(s, slot, tag, score)


def _evict_one(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
               tm: Pointer[SlabHashMap, MutUntrackedOrigin],
               st: Pointer[EvictState, MutUntrackedOrigin],
               versions: Pointer[UInt64, MutUntrackedOrigin], now_ns: Int64) -> Bool:
    """Evict one key. False when there is nothing to evict."""
    var policy = st[].policy
    var vol = _is_volatile(policy)
    if policy == POLICY_ALLKEYS_RANDOM or policy == POLICY_VOLATILE_RANDOM:
        var at = -1
        for _ in range(8):
            at = _random_volatile_slot(ks, tm, st) if vol else _random_key_slot(ks, st)
            if at >= 0:
                break
        if at < 0:
            return False
        _remove(ks, st, versions, at >> 48, at & 0xFFFFFFFFFFFF)
        return True

    for _ in range(4):
        for _ in range(st[].samples):
            var at = _random_volatile_slot(ks, tm, st) if vol else _random_key_slot(ks, st)
            if at < 0:
                continue
            var s = at >> 48
            var slot = at & 0xFFFFFFFFFFFF
            var m = ks[].shards.unsafe_offset(s)
            var h = UInt64(m[].keys[unsafe_offset=slot].__hash__())
            _pool_add(st, s, slot, evict_tag(h), _score(ks, tm, st, m, slot, h, now_ns))
        # The best candidate that is still what it was.
        while st[].pool_n > 0:
            var pool = st[].pool
            var best = 0
            for i in range(st[].pool_n):
                if pool[unsafe_offset=i].score > pool[unsafe_offset=best].score:
                    best = i
            var c = pool[unsafe_offset=best]
            st[].pool_n -= 1
            pool[unsafe_offset=best] = pool[unsafe_offset=st[].pool_n]
            var m = ks[].shards.unsafe_offset(c.shard)
            if c.slot >= m[].capacity or m[].metadata[unsafe_offset=c.slot] >= 128:
                continue
            var h = UInt64(m[].keys[unsafe_offset=c.slot].__hash__())
            if evict_tag(h) != c.tag:
                continue
            # Read since it was sampled: a pool entry can be old, and the score
            # it holds would evict the hottest key in the table.
            var score = _score(ks, tm, st, m, c.slot, h, now_ns)
            if score < c.score:
                _pool_add(st, c.shard, c.slot, c.tag, score)
                continue
            _remove(ks, st, versions, c.shard, c.slot)
            return True
    return False


def _sample_sizes(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
                  st: Pointer[EvictState, MutUntrackedOrigin]):
    """Keep the running mean of a key's size current."""
    for _ in range(SIZE_SAMPLES_PER_TICK):
        var at = _random_key_slot(ks, st)
        if at < 0:
            return
        var m = ks[].shards.unsafe_offset(at >> 48)
        var slot = at & 0xFFFFFFFFFFFF
        var b = Float64(Int(memory_usage(m[].values[unsafe_offset=slot],
                                         m[].keys[unsafe_offset=slot].string_len(), 5)))
        if st[].avg_bytes == 0.0:
            st[].avg_bytes = b
        else:
            st[].avg_bytes += (b - st[].avg_bytes) / 128.0


def evict_configure(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
                    tm: Pointer[SlabHashMap, MutUntrackedOrigin],
                    policy: Int, worker_id: Int) -> Bool:
    """The housekeeping tick: bring this worker up to date with CONFIG SET. True
    when eviction is on, which sends the event loop to evict_tick every tick."""
    var st = ks[].evict
    if is_null(st):
        st = alloc[EvictState](1)
        st.unsafe_write(EvictState(UInt64(_get_now_ns()) ^ (UInt64(worker_id) << 32)))
        ks[].evict = st
    if st[].policy != policy:
        _apply_policy(ks, tm, st, policy)
    if policy == POLICY_NOEVICTION:
        return False
    st[].samples = Int(external_call["pion_get_maxmemory_samples", Int32]())
    st[].log_factor = Int(external_call["pion_get_lfu_log_factor", Int32]())
    st[].decay_time = Int(external_call["pion_get_lfu_decay_time", Int32]())
    return True


def evict_tick(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
               tm: Pointer[SlabHashMap, MutUntrackedOrigin],
               wal: Pointer[WAL, MutUntrackedOrigin],
               worker_id: Int, num_workers: Int, hold: Bool,
               versions: Pointer[UInt64, MutUntrackedOrigin]):
    """Every event-loop tick while a policy is set. A recv buffer can hold tens
    of thousands of writes, so waiting for housekeeping (64 ticks) overruns the
    limit by megabytes on one pipelined connection. `hold`: a replica, or CLIENT
    PAUSE: report, don't evict.

    The inner loop runs every tick. The outer loop, and what it reads, at most
    every 10 ms, which is how often the RSS sample changes."""
    var st = ks[].evict
    var keys = 0
    for i in range(8):
        keys += ks[].shards[unsafe_offset=i].size
    var now_ns = _get_now_ns()
    if now_ns - st[].last_steer_ns >= 10000000:
        st[].last_steer_ns = now_ns
        _steer(ks, tm, wal, st, keys, worker_id, num_workers, hold)
    if hold or st[].target < 0 or st[].avg_bytes <= 0.0:
        st[].prev_keys = keys
        return
    var slots = 0
    for i in range(8):
        slots += ks[].shards[unsafe_offset=i].capacity
    var by_data = Int(Float64(keys) - Float64(st[].target) / st[].avg_bytes)
    var by_table = keys - slots * EVICT_TABLE_FILL // 100
    var n = by_data if by_data > by_table else by_table
    if n > 0:
        # Keep up with what this tick wrote, however large; otherwise spread
        # the work over ticks so one does not stall the loop.
        n += 1
        var cap = 2 * (keys - st[].prev_keys)
        if cap < EVICT_MAX_PER_TICK:
            cap = EVICT_MAX_PER_TICK
        if n > cap:
            n = cap
        for _ in range(n):
            if not _evict_one(ks, tm, st, versions, now_ns):
                break
        keys = 0
        for i in range(8):
            keys += ks[].shards[unsafe_offset=i].size
    st[].prev_keys = keys


def _steer(ks: Pointer[StripedHashMap, MutUntrackedOrigin],
           tm: Pointer[SlabHashMap, MutUntrackedOrigin],
           wal: Pointer[WAL, MutUntrackedOrigin],
           st: Pointer[EvictState, MutUntrackedOrigin],
           keys: Int, worker_id: Int, num_workers: Int, hold: Bool):
    """The outer loop: report to the process, read RSS, move the target."""
    for i in range(8):
        ks[].shards.unsafe_offset(i)[].shrink_if_sparse()
    var ttls = 0
    if is_not_null(tm):
        tm[].shrink_if_sparse()
        ttls = tm[].size
    var wal_bytes = UInt64(wal[].tail_offset) if is_not_null(wal) else UInt64(0)
    external_call["pion_evict_report", NoneType](
        Int32(worker_id), wal_bytes, UInt64(ttls if _is_volatile(st[].policy) else keys))
    var limit = Int(external_call["pion_get_maxmemory", UInt64]())
    if limit == 0:
        st[].target = -1
        st[].last_limit = 0
        for i in range(8):
            ks[].shards.unsafe_offset(i)[].set_may_grow(True)
        if is_not_null(tm):
            tm[].set_may_grow(True)
        return
    _ = external_call["pion_maxmemory_check", Int32]()      # refreshes the sample
    var ctl = Int(external_call["pion_evict_ctl", UInt64]())
    if ctl == 0:
        return
    for i in range(8):
        var m = ks[].shards.unsafe_offset(i)
        m[].set_may_grow(ctl + m[].capacity * SLOT_BYTES <= limit)
    if is_not_null(tm):
        tm[].set_may_grow(ctl + tm[].capacity * SLOT_BYTES <= limit)
    if hold:
        return
    if keys == 0 and (st[].base_ctl == 0 or ctl < st[].base_ctl):
        st[].base_ctl = ctl                                 # what an empty server holds
    if st[].last_limit != limit:                            # CONFIG SET maxmemory
        st[].last_limit = limit
        st[].target = -1
    if st[].target < 0 and ctl * 4 < limit * 3:
        return
    _sample_sizes(ks, st)
    var data = Int(Float64(keys) * st[].avg_bytes)
    if st[].target < 0:
        if ctl <= limit:
            return
        # Cut the data by the share of the heap that is over. Not by the bytes
        # over: the estimate counts less than the heap holds (allocator overhead,
        # empty table slots), so after CONFIG SET maxmemory halves the limit the
        # bytes over can exceed all the data the estimate sees.
        var heap = ctl - st[].base_ctl
        var keep = 1.0 - Float64(ctl - limit + limit // 50) / Float64(heap if heap > 0 else 1)
        st[].target = Int(Float64(data) * keep) if keep > 0.0 else 0
        st[].peak_ctl = ctl
    elif ctl > limit:
        # Only a new peak. RSS wobbles, and taking every rise off the target
        # while no fall puts it back drains the keyspace on noise alone.
        if ctl > st[].peak_ctl:
            var t = st[].target - (ctl - st[].peak_ctl) // num_workers
            st[].target = t if t > 0 else 0
            st[].peak_ctl = ctl
    else:
        st[].peak_ctl = limit
        if ctl * 10 < limit * 9:
            st[].target += (limit * 9 // 10 - ctl) // (2 * num_workers)
