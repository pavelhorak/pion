# gh #197 protocol step 1: standalone microbench — dual MinHeap/MaxHeap vs
# LinearPool on a synthetic beam-search op trace. Kill bar: LinearPool must be
# ≥1.5× faster on the trace or the integration work stops.
#
# Trace shape mirrors the 50K gate (ef=150): seed insert, then pop-expand
# rounds of ~30 scored neighbors each with xorshift-generated distances, run
# to natural termination under each structure's own discipline (that
# divergence is the real divergence — see the issue's recall argument).
#
# Run:  pixi run mojo run -I . benchmarks/micro/bench_linear_pool.mojo

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import memcpy
from std.ffi import external_call
from src.common.heap import HeapNode, MinHeap, MaxHeap
from src.common.ptr import null_ptr, is_not_null

comptime EF = 150
comptime NEIGHBORS = 30
comptime QUERIES = 20_000
comptime CHECKED = UInt32(0x80000000)


def _now_ns() -> Int64:
    var ts = alloc[Int64](2)
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    var result = ts[0] * Int64(1_000_000_000) + ts[1]
    ts.free()
    return result


struct PoolEntry(Copyable, Movable, ImplicitlyCopyable):
    var dist: Float32
    var id: UInt32

    def __init__(out self, dist: Float32, id: UInt32):
        self.dist = dist
        self.id = id


struct LinearPool(Movable):
    """gh #197: one flat sorted (dist,id) array capped at ef + a cursor to the
    first unexpanded entry (pyglass/ParlayANN discipline). Insert = binary
    search + memmove over ≤ef 8B entries (L1-resident); pop = entries[cursor++]
    with a checked bit in the id's high bit so a closer late insert rewinds the
    cursor without re-expanding anything."""
    var entries: UnsafePointer[PoolEntry, MutUntrackedOrigin]
    var capacity: Int
    var size: Int
    var cursor: Int
    var ef: Int

    def __init__(out self):
        self.entries = null_ptr[PoolEntry, MutUntrackedOrigin]()
        self.capacity = 0
        self.size = 0
        self.cursor = 0
        self.ef = 0

    def __del__(deinit self):
        if is_not_null(self.entries):
            self.entries.free()

    @always_inline
    def reset(mut self, ef: Int):
        if ef > self.capacity:
            if is_not_null(self.entries):
                self.entries.free()
            self.entries = alloc[PoolEntry](ef)
            self.capacity = ef
        self.ef = ef
        self.size = 0
        self.cursor = 0

    @always_inline
    def insert(mut self, d: Float32, id: Int) -> Bool:
        if self.size == self.ef and d >= self.entries[self.size - 1].dist:
            return False
        # Upper-bound binary search: first index with dist > d (stable ties).
        var lo = 0
        var hi = self.size
        while lo < hi:
            var mid = (lo + hi) >> 1
            if self.entries[mid].dist > d:
                hi = mid
            else:
                lo = mid + 1
        var move_n = self.size - lo
        if self.size == self.ef:
            move_n -= 1  # last entry falls off the end
        if move_n > 0:
            _ = external_call["memmove", UnsafePointer[NoneType, MutUntrackedOrigin]](
                (self.entries + lo + 1).bitcast[NoneType](),
                (self.entries + lo).bitcast[NoneType](),
                move_n * 8)
        self.entries[lo] = PoolEntry(d, UInt32(id))
        if self.size < self.ef:
            self.size += 1
        if lo < self.cursor:
            self.cursor = lo
        return True

    @always_inline
    def has_next(self) -> Bool:
        return self.cursor < self.size

    @always_inline
    def pop_dist(mut self) -> Float32:
        """Advance past the entry at cursor (marking it expanded) and return its
        distance. Caller reads the id via last_id() before calling if needed."""
        var e = self.entries[self.cursor]
        self.entries[self.cursor].id = e.id | CHECKED
        self.cursor += 1
        while self.cursor < self.size and (self.entries[self.cursor].id & CHECKED) != 0:
            self.cursor += 1
        return e.dist

    @always_inline
    def worst(self) -> Float32:
        return self.entries[self.size - 1].dist


@always_inline
def _xorshift(mut s: UInt64) -> UInt64:
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


comptime RING = 1 << 20  # 1M pre-generated distances (4MB) shared by both arms


def main() raises:
    # Pre-generate the distance stream OUTSIDE the timed region — the timed
    # loops then measure structure ops + one L2-resident array read, not the
    # PRNG + int→float conversion both arms would otherwise share (which only
    # dilutes the ratio the kill bar is about).
    var ring = alloc[Float32](RING)
    var seed = UInt64(0x9E3779B97F4A7C15)
    for i in range(RING):
        var r = _xorshift(seed)
        ring[i] = Float32(Int(r & 0xFFFFF)) * Float32(9.5367431640625e-07)

    # ── dual-heap arm ──────────────────────────────────────────────────────
    var cand = MinHeap()
    var res = MaxHeap()
    cand.reserve(EF + 64)
    res.reserve(EF + 64)
    var heap_checksum = Float64(0)
    var heap_pops = 0
    var rpos = 0
    var t0 = _now_ns()
    for _q in range(QUERIES):
        cand.clear()
        res.clear()
        cand.push(HeapNode(Float32(0.5), 0))
        _ = res.push_bounded(Float32(0.5), 0, EF)
        var node_id = 1
        # fill phase: candidates pushed unconditionally (matches _beam_search_1536)
        while len(cand.data) > 0 and len(res.data) < EF:
            var c = cand.pop()
            heap_pops += 1
            for _j in range(NEIGHBORS):
                var d = ring[rpos & (RING - 1)]
                rpos += 1
                cand.push(HeapNode(d, node_id))
                _ = res.push_bounded(d, node_id, EF)
                node_id += 1
        # prune phase: accept-gated
        while len(cand.data) > 0:
            var c = cand.pop()
            heap_pops += 1
            if c.distance > res.peek_distance():
                break
            for _j in range(NEIGHBORS):
                var d = ring[rpos & (RING - 1)]
                rpos += 1
                if res.push_bounded(d, node_id, EF):
                    cand.push(HeapNode(d, node_id))
                node_id += 1
        # drain like the search epilogue: pop all, sum
        while len(res.data) > 0:
            var n = res.pop()
            heap_checksum += Float64(n.distance)
    var heap_ns = _now_ns() - t0

    # ── LinearPool arm (identical distance stream) ─────────────────────────
    var pool = LinearPool()
    pool.reset(EF)
    var pool_checksum = Float64(0)
    var pool_pops = 0
    rpos = 0
    var t1 = _now_ns()
    for _q in range(QUERIES):
        pool.reset(EF)
        _ = pool.insert(Float32(0.5), 0)
        var node_id = 1
        while pool.has_next():
            _ = pool.pop_dist()
            pool_pops += 1
            for _j in range(NEIGHBORS):
                var d = ring[rpos & (RING - 1)]
                rpos += 1
                _ = pool.insert(d, node_id)
                node_id += 1
        # drain: nearest-first is just the first k entries
        for i in range(pool.size):
            pool_checksum += Float64(pool.entries[i].dist)
    var pool_ns = _now_ns() - t1
    ring.free()

    print("dual-heap : " + String(heap_ns // 1_000_000) + " ms  pops=" +
          String(heap_pops) + "  checksum=" + String(heap_checksum))
    print("LinearPool: " + String(pool_ns // 1_000_000) + " ms  pops=" +
          String(pool_pops) + "  checksum=" + String(pool_checksum))
    print("speedup on trace: " + String(Float64(heap_ns) / Float64(pool_ns)))
