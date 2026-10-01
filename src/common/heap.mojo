from std.collections import List
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from src.common.ptr import null_ptr, is_not_null

struct HeapNode(Copyable, Movable):
    var distance: Float32
    var id: Int

    def __init__(out self, distance: Float32, id: Int):
        self.distance = distance
        self.id = id

    def __copyinit__(out self, copy: Self):
        self.distance = copy.distance
        self.id = copy.id

    def __moveinit__(out self, deinit take: Self):
        self.distance = take.distance
        self.id = take.id

    def copy(self) -> Self:
        return Self(self.distance, self.id)

    def __lt__(self, other: HeapNode) -> Bool:
        return self.distance < other.distance

    def __gt__(self, other: HeapNode) -> Bool:
        return self.distance > other.distance

    def __le__(self, other: HeapNode) -> Bool:
        return self.distance <= other.distance

    def __ge__(self, other: HeapNode) -> Bool:
        return self.distance >= other.distance

@always_inline
def _swap_nodes(mut data: List[HeapNode], a: Int, b: Int):
    """Swap two HeapNodes via field-level exchange — avoids .copy() overhead."""
    var tmp_d = data[a].distance
    var tmp_id = data[a].id
    data[a].distance = data[b].distance
    data[a].id = data[b].id
    data[b].distance = tmp_d
    data[b].id = tmp_id

struct MinHeap(Movable):
    var data: List[HeapNode]

    def __init__(out self):
        self.data = List[HeapNode]()

    def __moveinit__(out self, deinit take: Self):
        self.data = take.data^

    def push(mut self, var node: HeapNode):
        self.data.append(node^)
        self._bubble_up(len(self.data) - 1)

    def pop(mut self) -> HeapNode:
        var root_d = self.data[0].distance
        var root_id = self.data[0].id
        if len(self.data) > 1:
            var last = self.data.pop()
            self.data[0] = last^
            self._bubble_down(0)
        else:
            _ = self.data.pop()
        return HeapNode(root_d, root_id)

    def peek(self) -> HeapNode:
        return HeapNode(self.data[0].distance, self.data[0].id)

    @always_inline
    def peek_distance(self) -> Float32:
        """Return worst (min) distance without constructing HeapNode."""
        return self.data[0].distance

    def clear(mut self):
        self.data.clear()

    @always_inline
    def reserve(mut self, capacity: Int):
        """gh #131 §2.5: pre-size the backing List so push() never reallocs
        mid-search. List.clear() retains capacity, so this is a monotonic (grow-only)
        hint that pays off on the first query of each graph; bit-identical results."""
        self.data.reserve(capacity)

    def __len__(self) -> Int:
        return len(self.data)

    def _bubble_up(mut self, index: Int):
        var curr = index
        while curr > 0:
            var parent = (curr - 1) // 2
            if self.data[curr].distance < self.data[parent].distance:
                _swap_nodes(self.data, curr, parent)
                curr = parent
            else:
                break

    def _bubble_down(mut self, index: Int):
        var curr = index
        var n = len(self.data)
        while True:
            var left = 2 * curr + 1
            var right = 2 * curr + 2
            var smallest = curr

            if left < n and self.data[left].distance < self.data[smallest].distance:
                smallest = left
            if right < n and self.data[right].distance < self.data[smallest].distance:
                smallest = right

            if smallest != curr:
                _swap_nodes(self.data, curr, smallest)
                curr = smallest
            else:
                break

struct PoolEntry(Copyable, Movable, ImplicitlyCopyable):
    """8-byte (dist, id) pair for LinearPool. id's high bit is the expanded
    ('checked') flag — node indices are internal slot indices < 2^31."""
    var dist: Float32
    var id: UInt32

    def __init__(out self, dist: Float32, id: UInt32):
        self.dist = dist
        self.id = id


struct LinearPool(Movable):
    """gh #197: single flat sorted (dist,id) array capped at ef + a cursor to
    the first unexpanded entry (pyglass/ParlayANN discipline). Replaces the
    MinHeap candidates + MaxHeap results pair in the 1536 beam search:
    pop = entries[cursor++]; insert = binary search + memmove over ≤ef 8B
    entries (1.2 KB at ef=150, L1-resident); worst = one load; the result
    drain is a bounded read of the first k entries — already nearest-first,
    no full-depth pop sifts, no reversal. A closer late insert rewinds the
    cursor; the checked bit prevents re-expansion. Microbench (structure ops
    on a gate-shaped trace): 1.56× vs the heap pair. Callers with ef > 512
    should stay on the heaps (memmove cost grows linearly with ef)."""
    comptime CHECKED = UInt32(0x80000000)

    var entries: Pointer[PoolEntry, MutUntrackedOrigin]
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
            self.entries.unsafe_free()

    @always_inline
    def reset(mut self, ef: Int):
        """Clear and (re)bound to ef. Capacity is monotonic — allocates only
        when ef grows past every previous reset."""
        var bound = max(1, ef)  # ef=0 would make insert() read entries[-1]
        if bound > self.capacity:
            if is_not_null(self.entries):
                self.entries.unsafe_free()
            self.entries = alloc[PoolEntry](bound)
            self.capacity = bound
        self.ef = bound
        self.size = 0
        self.cursor = 0

    @always_inline
    def insert(mut self, d: Float32, id: Int) -> Bool:
        """Accept-gated sorted insert. Returns True if the entry entered the
        pool (improved the top-ef set)."""
        if self.size == self.ef and d >= self.entries[unsafe_offset=self.size - 1].dist:
            return False
        # Upper-bound binary search: first index with dist > d (stable ties).
        var lo = 0
        var hi = self.size
        while lo < hi:
            var mid = (lo + hi) >> 1
            if self.entries[unsafe_offset=mid].dist > d:
                hi = mid
            else:
                lo = mid + 1
        var move_n = self.size - lo
        if self.size == self.ef:
            move_n -= 1  # the worst entry falls off the end
        if move_n > 0:
            _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                (self.entries.unsafe_offset(lo).unsafe_offset(1)).unsafe_bitcast[NoneType](),
                (self.entries.unsafe_offset(lo)).unsafe_bitcast[NoneType](),
                move_n * 8)
        self.entries[unsafe_offset=lo] = PoolEntry(d, UInt32(id))
        if self.size < self.ef:
            self.size += 1
        if lo < self.cursor:
            self.cursor = lo
        return True

    @always_inline
    def has_next(self) -> Bool:
        return self.cursor < self.size

    @always_inline
    def pop(mut self) -> HeapNode:
        """Return the nearest unexpanded entry and mark it expanded. The skip
        loop advances past entries already expanded before a cursor rewind."""
        var e = self.entries[unsafe_offset=self.cursor]
        self.entries[unsafe_offset=self.cursor].id = e.id | Self.CHECKED
        self.cursor += 1
        while self.cursor < self.size and (self.entries[unsafe_offset=self.cursor].id & Self.CHECKED) != 0:
            self.cursor += 1
        return HeapNode(e.dist, Int(e.id & ~Self.CHECKED))

    @always_inline
    def worst(self) -> Float32:
        """Current ef-th best distance (valid whenever size > 0)."""
        return self.entries[unsafe_offset=self.size - 1].dist

    @always_inline
    def entry_id(self, i: Int) -> Int:
        """Node id at sorted position i, checked bit stripped."""
        return Int(self.entries[unsafe_offset=i].id & ~Self.CHECKED)

    @always_inline
    def entry_dist(self, i: Int) -> Float32:
        return self.entries[unsafe_offset=i].dist


struct MaxHeap(Movable):
    var data: List[HeapNode]
    var _cached_worst: Float32  # Cached worst (max) distance — avoids peek() on threshold checks

    def __init__(out self):
        self.data = List[HeapNode]()
        self._cached_worst = Float32(-1e30)

    def __moveinit__(out self, deinit take: Self):
        self.data = take.data^
        self._cached_worst = take._cached_worst

    def push(mut self, var node: HeapNode):
        if node.distance > self._cached_worst:
            self._cached_worst = node.distance
        self.data.append(node^)
        self._bubble_up(len(self.data) - 1)

    def pop(mut self) -> HeapNode:
        var root_d = self.data[0].distance
        var root_id = self.data[0].id
        if len(self.data) > 1:
            var last = self.data.pop()
            self.data[0] = last^
            self._bubble_down(0)
            self._cached_worst = self.data[0].distance
        else:
            _ = self.data.pop()
            self._cached_worst = Float32(-1e30)
        return HeapNode(root_d, root_id)

    @always_inline
    def push_bounded(mut self, distance: Float32, id: Int, ef: Int) -> Bool:
        """Combined push + conditional pop for HNSW result set.
        Returns True if the candidate was accepted (improved the result set).
        Avoids separate push()+pop()+peek() sequence — single operation."""
        var n = len(self.data)
        if n < ef:
            # Fill phase: always accept
            self.data.append(HeapNode(distance, id))
            self._bubble_up(n)
            if distance > self._cached_worst:
                self._cached_worst = distance
            return True
        elif distance < self._cached_worst:
            # Prune phase: candidate beats worst — replace root in place
            self.data[0].distance = distance
            self.data[0].id = id
            self._bubble_down(0)
            self._cached_worst = self.data[0].distance
            return True
        return False

    def peek(self) -> HeapNode:
        return HeapNode(self.data[0].distance, self.data[0].id)

    @always_inline
    def peek_distance(self) -> Float32:
        """Return worst (max) distance without constructing HeapNode."""
        return self._cached_worst

    def clear(mut self):
        self.data.clear()
        self._cached_worst = Float32(-1e30)

    @always_inline
    def reserve(mut self, capacity: Int):
        """gh #131 §2.5: pre-size the backing List so push()/push_bounded() never
        realloc mid-search. Monotonic (grow-only) hint; bit-identical results."""
        self.data.reserve(capacity)

    def __len__(self) -> Int:
        return len(self.data)

    def _bubble_up(mut self, index: Int):
        var curr = index
        while curr > 0:
            var parent = (curr - 1) // 2
            if self.data[curr].distance > self.data[parent].distance:
                _swap_nodes(self.data, curr, parent)
                curr = parent
            else:
                break

    def _bubble_down(mut self, index: Int):
        var curr = index
        var n = len(self.data)
        while True:
            var left = 2 * curr + 1
            var right = 2 * curr + 2
            var largest = curr

            if left < n and self.data[left].distance > self.data[largest].distance:
                largest = left
            if right < n and self.data[right].distance > self.data[largest].distance:
                largest = right

            if largest != curr:
                _swap_nodes(self.data, curr, largest)
                curr = largest
            else:
                break
