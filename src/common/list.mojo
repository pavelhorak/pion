from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from src.common.value import GenericValue, ValueType

struct SlabList(Movable):
    # Ziplist (small lists ≤ ZIPLIST_MAX_ENTRIES)
    var size: Int
    var zip_buf: Pointer[UInt8, MutUntrackedOrigin]
    var zip_len: Int
    var zip_cap: Int

    # Head side: items pushed via LPUSH (logical indices 0..)
    # active_head_data[head_off..head_end-1] are valid
    # head_off == head_end means empty; LPUSH decrements head_off; RPOP (fallback) decrements head_end
    var active_head_data: Pointer[GenericValue, MutUntrackedOrigin]
    var head_off: Int       # front pointer (LPOP reads here; LPUSH writes here-1)
    var head_end: Int       # one-past-end of valid items (starts at SEG_SIZE; RPOP shrinks from right)
    var head_segs: Pointer[Pointer[GenericValue, MutUntrackedOrigin], MutUntrackedOrigin]
    var head_seg_count: Int       # one-past-end of valid committed segs (LPOP shrinks from right)
    var head_segs_start: Int      # first valid committed seg index (RPOP oldest-head advances this)
    var head_seg_alloc: Int

    # Tail side: items pushed via RPUSH (logical indices from head_total onwards)
    # active_tail_data[0..tail_count-1] are valid (tail_count == 0 means empty)
    var active_tail_data: Pointer[GenericValue, MutUntrackedOrigin]
    var tail_count: Int
    var tail_segs: Pointer[Pointer[GenericValue, MutUntrackedOrigin], MutUntrackedOrigin]
    var tail_seg_count: Int
    var tail_seg_alloc: Int

    comptime SEG_SIZE = 256
    comptime ZIPLIST_MAX_ENTRIES = 1024
    comptime ZIPLIST_MAX_VALUE_LEN = 64
    comptime ZIPLIST_INITIAL_CAP = 8192

    def __init__(out self):
        self.size = 0
        self.zip_cap = Self.ZIPLIST_INITIAL_CAP
        self.zip_buf = alloc[UInt8](self.zip_cap)
        self.zip_len = 0

        self.active_head_data = alloc[GenericValue](Self.SEG_SIZE)
        self.head_off = Self.SEG_SIZE  # empty
        self.head_end = Self.SEG_SIZE
        self.head_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](8)
        self.head_seg_count = 0
        self.head_segs_start = 0
        self.head_seg_alloc = 8

        self.active_tail_data = alloc[GenericValue](Self.SEG_SIZE)
        self.tail_count = 0
        self.tail_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](8)
        self.tail_seg_count = 0
        self.tail_seg_alloc = 8

    def reset(mut self):
        # Free all segmented data
        if is_not_null(self.active_head_data):
            self.active_head_data.unsafe_free()
        for i in range(self.head_segs_start, self.head_seg_count):
            if is_not_null(self.head_segs[unsafe_offset=i]):
                self.head_segs[unsafe_offset=i].unsafe_free()
        if is_not_null(self.head_segs):
            self.head_segs.unsafe_free()

        if is_not_null(self.active_tail_data):
            self.active_tail_data.unsafe_free()
        for i in range(self.tail_seg_count):
            if is_not_null(self.tail_segs[unsafe_offset=i]):
                self.tail_segs[unsafe_offset=i].unsafe_free()
        if is_not_null(self.tail_segs):
            self.tail_segs.unsafe_free()

        # Free ziplist
        if is_not_null(self.zip_buf):
            self.zip_buf.unsafe_free()

        # Reinitialize
        self.size = 0
        self.zip_cap = Self.ZIPLIST_INITIAL_CAP
        self.zip_buf = alloc[UInt8](self.zip_cap)
        self.zip_len = 0

        self.active_head_data = alloc[GenericValue](Self.SEG_SIZE)
        self.head_off = Self.SEG_SIZE
        self.head_end = Self.SEG_SIZE
        self.head_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](8)
        self.head_seg_count = 0
        self.head_segs_start = 0
        self.head_seg_alloc = 8

        self.active_tail_data = alloc[GenericValue](Self.SEG_SIZE)
        self.tail_count = 0
        self.tail_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](8)
        self.tail_seg_count = 0
        self.tail_seg_alloc = 8

    def release(mut self):
        """gh #369: free everything this list owns when its key is dropped —
        ~24 KB of buffers even when empty, plus heap payloads in segmented mode
        (ziplist entries are inline bytes). The list must not be used after."""
        if is_null(self.zip_buf):
            while self.size > 0:
                var v = self.lpop()
                v.free_str_payload()
        if is_not_null(self.active_head_data):
            self.active_head_data.unsafe_free()
        for i in range(self.head_segs_start, self.head_seg_count):
            if is_not_null(self.head_segs[unsafe_offset=i]):
                self.head_segs[unsafe_offset=i].unsafe_free()
        if is_not_null(self.head_segs):
            self.head_segs.unsafe_free()
        if is_not_null(self.active_tail_data):
            self.active_tail_data.unsafe_free()
        for i in range(self.tail_seg_count):
            if is_not_null(self.tail_segs[unsafe_offset=i]):
                self.tail_segs[unsafe_offset=i].unsafe_free()
        if is_not_null(self.tail_segs):
            self.tail_segs.unsafe_free()
        if is_not_null(self.zip_buf):
            self.zip_buf.unsafe_free()
        self.active_head_data = null_ptr[GenericValue, MutUntrackedOrigin]()
        self.head_segs = null_ptr[Pointer[GenericValue, MutUntrackedOrigin], MutUntrackedOrigin]()
        self.active_tail_data = null_ptr[GenericValue, MutUntrackedOrigin]()
        self.tail_segs = null_ptr[Pointer[GenericValue, MutUntrackedOrigin], MutUntrackedOrigin]()
        self.zip_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.size = 0
        self.head_seg_count = 0
        self.head_segs_start = 0
        self.tail_seg_count = 0

    def __moveinit__(out self, deinit take: Self):
        self.size = take.size
        self.zip_buf = take.zip_buf
        self.zip_len = take.zip_len
        self.zip_cap = take.zip_cap

        self.active_head_data = take.active_head_data
        self.head_off = take.head_off
        self.head_end = take.head_end
        self.head_segs = take.head_segs
        self.head_seg_count = take.head_seg_count
        self.head_segs_start = take.head_segs_start
        self.head_seg_alloc = take.head_seg_alloc

        self.active_tail_data = take.active_tail_data
        self.tail_count = take.tail_count
        self.tail_segs = take.tail_segs
        self.tail_seg_count = take.tail_seg_count
        self.tail_seg_alloc = take.tail_seg_alloc


    @always_inline
    def _grow_head_segs(mut self):
        var new_alloc = self.head_seg_alloc * 2
        var new_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](new_alloc)
        var valid = self.head_seg_count - self.head_segs_start
        for i in range(valid):
            new_segs[unsafe_offset=i] = self.head_segs[unsafe_offset=self.head_segs_start + i]
        self.head_segs.unsafe_free()
        self.head_segs = new_segs
        self.head_seg_alloc = new_alloc
        self.head_seg_count = valid
        self.head_segs_start = 0

    @always_inline
    def _grow_tail_segs(mut self):
        var new_alloc = self.tail_seg_alloc * 2
        var new_segs = alloc[Pointer[GenericValue, MutUntrackedOrigin]](new_alloc)
        for i in range(self.tail_seg_count):
            new_segs[unsafe_offset=i] = self.tail_segs[unsafe_offset=i]
        self.tail_segs.unsafe_free()
        self.tail_segs = new_segs
        self.tail_seg_alloc = new_alloc

    def _convert_to_segmented(mut self):
        # Called when zip_buf is active and we need to switch to segmented mode.
        # zip_buf[0..size-1] (in ziplist format) = logical indices 0..size-1.
        # We push items in REVERSE zip_buf order so zip_buf[0] ends up at logical index 0.
        if is_null(self.zip_buf):
            return

        # Collect entry pointers from zip_buf first
        var ptrs = alloc[Pointer[UInt8, MutUntrackedOrigin]](self.size)
        var lens = alloc[Int](self.size)
        var off = 0
        for i in range(self.size):
            var v_len = Int((self.zip_buf.unsafe_offset(off)).unsafe_bitcast[UInt16]()[])
            ptrs[unsafe_offset=i] = self.zip_buf.unsafe_offset(off).unsafe_offset(2)
            lens[unsafe_offset=i] = v_len
            off += 2 + v_len

        # Push in reverse: zip_buf[size-1] first, zip_buf[0] last
        # After all pushes, zip_buf[0] = logical index 0 (at active_head_data[head_off])
        for i in range(self.size - 1, -1, -1):
            var val = GenericValue.from_ptr(ptrs[unsafe_offset=i], lens[unsafe_offset=i])
            # Inline lpush_seg to avoid infinite recursion risk
            if self.head_off == 0:
                if self.head_seg_count == self.head_seg_alloc:
                    self._grow_head_segs()
                self.head_segs[unsafe_offset=self.head_seg_count] = self.active_head_data
                self.head_seg_count += 1
                self.active_head_data = alloc[GenericValue](Self.SEG_SIZE)
                self.head_off = Self.SEG_SIZE
            self.head_off -= 1
            self.active_head_data[unsafe_offset=self.head_off] = val

        ptrs.unsafe_free()
        lens.unsafe_free()

        self.zip_buf.unsafe_free()
        self.zip_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        self.zip_len = 0
        self.zip_cap = 0

    def lpush(mut self, var value: GenericValue):
        if is_not_null(self.zip_buf):
            var s_len = value.string_len() if value.is_string() else 0
            if self.size >= Self.ZIPLIST_MAX_ENTRIES or s_len > Self.ZIPLIST_MAX_VALUE_LEN or self.zip_len + 2 + s_len > self.zip_cap:
                self._convert_to_segmented()
                # Fall through to segmented push below
            else:
                var entry_len = 2 + s_len
                if self.zip_len > 0:
                    _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                        (self.zip_buf.unsafe_offset(entry_len)).unsafe_bitcast[NoneType](),
                        self.zip_buf.unsafe_bitcast[NoneType](),
                        self.zip_len
                    )
                self.zip_buf.unsafe_bitcast[UInt16]()[] = UInt16(s_len)
                value.copy_to(self.zip_buf.unsafe_offset(2))
                self.zip_len += entry_len
                self.size += 1
                # gh #394: a push takes OWNERSHIP of `value` (the segmented
                # branch below keeps it as-is). Here the bytes were copied into
                # zip_buf, so the heap copy the caller built is ours to free —
                # left alone, every 24-64 byte element leaked its copy.
                value.free_str_payload()
                return

        # Segmented push (head side)
        if self.head_off == 0 and self.head_end < Self.SEG_SIZE:
            # gh #369: the active head can be partial ([0, head_end) after an
            # rpop or an lpop refill). Committed segs must be FULL (rpop and
            # lrange read all SEG_SIZE slots), so slide the live elements to the
            # right end and keep filling this segment instead of committing it.
            var shift = Self.SEG_SIZE - self.head_end
            for q in range(self.head_end - 1, -1, -1):
                self.active_head_data[unsafe_offset=q + shift] = self.active_head_data[unsafe_offset=q]
            self.head_off = shift
            self.head_end = Self.SEG_SIZE
        if self.head_off == 0:
            if self.head_seg_count == self.head_seg_alloc:
                self._grow_head_segs()
            self.head_segs[unsafe_offset=self.head_seg_count] = self.active_head_data
            self.head_seg_count += 1
            self.active_head_data = alloc[GenericValue](Self.SEG_SIZE)
            self.head_off = Self.SEG_SIZE
            self.head_end = Self.SEG_SIZE  # fresh segment
        self.head_off -= 1
        self.active_head_data[unsafe_offset=self.head_off] = value.owned()
        self.size += 1

    def rpush(mut self, var value: GenericValue):
        if is_not_null(self.zip_buf):
            var s_len = value.string_len() if value.is_string() else 0
            if self.size >= Self.ZIPLIST_MAX_ENTRIES or s_len > Self.ZIPLIST_MAX_VALUE_LEN or self.zip_len + 2 + s_len > self.zip_cap:
                self._convert_to_segmented()
                # Fall through to segmented push below
            else:
                var offset = self.zip_len
                (self.zip_buf.unsafe_offset(offset)).unsafe_bitcast[UInt16]()[] = UInt16(s_len)
                value.copy_to(self.zip_buf.unsafe_offset(offset).unsafe_offset(2))
                self.zip_len += 2 + s_len
                self.size += 1
                value.free_str_payload()   # gh #394: copied in; see lpush
                return

        # Segmented push (tail side)
        if self.tail_count == Self.SEG_SIZE:
            if self.tail_seg_count == self.tail_seg_alloc:
                self._grow_tail_segs()
            self.tail_segs[unsafe_offset=self.tail_seg_count] = self.active_tail_data
            self.tail_seg_count += 1
            self.active_tail_data = alloc[GenericValue](Self.SEG_SIZE)
            self.tail_count = 0
        self.active_tail_data[unsafe_offset=self.tail_count] = value.owned()
        self.tail_count += 1
        self.size += 1

    def lpop(mut self) -> GenericValue:
        if self.size == 0:
            return GenericValue()

        if is_not_null(self.zip_buf):
            var v_len = Int(self.zip_buf.unsafe_bitcast[UInt16]()[])
            var val = GenericValue.from_ptr(self.zip_buf.unsafe_offset(2), v_len)
            var entry_len = 2 + v_len
            self.zip_len -= entry_len
            if self.zip_len > 0:
                _ = external_call["memmove", Pointer[NoneType, MutUntrackedOrigin]](
                    self.zip_buf.unsafe_bitcast[NoneType](),
                    (self.zip_buf.unsafe_offset(entry_len)).unsafe_bitcast[NoneType](),
                    self.zip_len
                )
            self.size -= 1
            return val

        # Segmented pop from the logical FRONT. Invariant (the one lrange walks):
        # active_head[head_off..head_end) → head_segs[count-1 .. start] (full)
        # → tail_segs[0 .. count) (full) → active_tail[0..tail_count).
        # gh #369 (found while freeing containers): once the head side ran dry
        # this used to pop the tail's NEWEST element ("best-effort ordering"),
        # so LPOP on a drained head answered from the wrong end. Refill the
        # active head from the next segment in logical order, then pop.
        if self.head_off == self.head_end:
            self._refill_head_for_lpop()
        var val = self.active_head_data[unsafe_offset=self.head_off]
        self.head_off += 1
        self.size -= 1
        return val

    def _refill_head_for_lpop(mut self):
        """Make the active head hold the next elements in logical order.
        Called only when it is empty and size > 0."""
        if self.head_seg_count > self.head_segs_start:
            # Most recently committed head seg = logically next.
            self.active_head_data.unsafe_free()
            self.head_seg_count -= 1
            self.active_head_data = self.head_segs[unsafe_offset=self.head_seg_count]
            self.head_off = 0
            self.head_end = Self.SEG_SIZE
            if self.head_seg_count == self.head_segs_start:
                self.head_seg_count = 0
                self.head_segs_start = 0
        elif self.tail_seg_count > 0:
            # Oldest tail seg becomes the active head; shift the rest down.
            self.active_head_data.unsafe_free()
            self.active_head_data = self.tail_segs[unsafe_offset=0]
            for t in range(1, self.tail_seg_count):
                self.tail_segs[unsafe_offset=t - 1] = self.tail_segs[unsafe_offset=t]
            self.tail_seg_count -= 1
            self.head_off = 0
            self.head_end = Self.SEG_SIZE
        else:
            # Only the active tail holds elements: it becomes the active head
            # ([0, tail_count) — lpush realigns a partial head before it ever
            # commits one, so committed segments stay full).
            self.active_head_data.unsafe_free()
            self.active_head_data = self.active_tail_data
            self.head_off = 0
            self.head_end = self.tail_count
            self.active_tail_data = alloc[GenericValue](Self.SEG_SIZE)
            self.tail_count = 0

    def rpop(mut self) -> GenericValue:
        if self.size == 0:
            return GenericValue()

        if is_not_null(self.zip_buf):
            var offset = 0
            var prev_offset = 0
            for _ in range(self.size):
                prev_offset = offset
                var v_len = Int((self.zip_buf.unsafe_offset(offset)).unsafe_bitcast[UInt16]()[])
                offset += 2 + v_len
            var v_len = Int((self.zip_buf.unsafe_offset(prev_offset)).unsafe_bitcast[UInt16]()[])
            var val = GenericValue.from_ptr(self.zip_buf.unsafe_offset(prev_offset).unsafe_offset(2), v_len)
            self.zip_len = prev_offset
            self.size -= 1
            return val

        # Segmented pop from the logical BACK (invariant: see lpop).
        # gh #369: when the tail side was empty this took the LAST element of
        # the oldest head segment and then freed the whole segment — 255
        # elements gone while `size` still counted them, so the next pops
        # read freed memory (RPUSH 1..1500, RPOP x1500 answered 1022, 767,
        # 511, … and then WRONGTYPE garbage). Refill the active tail instead.
        if self.tail_count == 0:
            if self.tail_seg_count > 0:
                self.active_tail_data.unsafe_free()
                self.tail_seg_count -= 1
                self.active_tail_data = self.tail_segs[unsafe_offset=self.tail_seg_count]
                self.tail_count = Self.SEG_SIZE
            elif self.head_seg_count > self.head_segs_start:
                # Oldest committed head seg = logically last: it becomes the
                # active tail, whole.
                self.active_tail_data.unsafe_free()
                self.active_tail_data = self.head_segs[unsafe_offset=self.head_segs_start]
                self.tail_count = Self.SEG_SIZE
                self.head_segs_start += 1
                if self.head_segs_start == self.head_seg_count:
                    self.head_segs_start = 0
                    self.head_seg_count = 0
            else:
                # Only the active head remains: pop its right end.
                self.head_end -= 1
                var hv = self.active_head_data[unsafe_offset=self.head_end]
                if self.head_end == self.head_off:
                    self.head_off = Self.SEG_SIZE  # empty; back to the sentinel
                    self.head_end = Self.SEG_SIZE
                self.size -= 1
                return hv
        self.tail_count -= 1
        var val = self.active_tail_data[unsafe_offset=self.tail_count]
        self.size -= 1
        return val

    def llen(self) -> Int:
        return self.size

    def owned_elems(self) -> List[GenericValue]:
        """gh #241: deep-copy every element, so the caller may then mutate or
        `reset()` this list without the copies dangling.

        Both representations need it, for opposite reasons. In ziplist mode
        `get_all()` hands back `from_ptr_unsafe` borrows INTO `zip_buf`, so any
        rewrite of that buffer corrupts them (this is the gh #170 replay
        corruption). In segmented mode the values are real `GenericValue`s, but
        the SSO ones live inline in the segment array that `reset()` frees.
        `from_ptr` materializes an owned copy in both cases."""
        var elems = self.get_all()
        var out = List[GenericValue]()
        out.reserve(len(elems))
        var buf = alloc[UInt8](64)
        for j in range(len(elems)):
            var v = elems[j]
            if v.is_string():
                out.append(GenericValue.from_ptr(v.as_string_safe(buf), v.string_len()))
            else:
                out.append(v)
        buf.unsafe_free()
        return out^

    def replace_all(mut self, elems: List[GenericValue]):
        """gh #241: rebuild the list from `elems`, which MUST be deep copies
        (see `owned_elems`) — the old contents are freed before the rebuild.

        This is how LSET/LINSERT/LREM/LTRIM work in quicklist mode. An in-place
        segmented splice would be faster, but these are cold commands and the
        two-sided structure has enough bookkeeping (head_off/head_end,
        head_segs_start, per-side segment arrays) that a partial update is how
        the LTRIM data-loss bug happened in the first place. Rebuilding cannot
        leave the structure and `size` disagreeing.

        Ownership: `rpush` takes ownership of each element in both modes
        (gh #394 — the ziplist branch frees the copy it made), so `elems` must
        not be used after this call."""
        var was_segmented = is_null(self.zip_buf)
        if was_segmented:
            # Ziplist payloads live inside zip_buf, which reset() frees. Calling
            # free_str_payload on a from_ptr_unsafe borrow would hand the
            # allocator an interior pointer, so only the segmented case frees.
            var old = self.get_all()
            for j in range(len(old)):
                old[j].free_str_payload()
        self.reset()
        for j in range(len(elems)):
            self.rpush(elems[j])

    def lrange(self, start_idx: Int, stop_idx: Int) -> List[GenericValue]:
        var sz = self.size
        var start = start_idx
        var stop = stop_idx

        if start < 0:
            start = sz + start
            if start < 0: start = 0
        if stop < 0:
            stop = sz + stop
            if stop < 0: stop = -1
        if stop >= sz: stop = sz - 1

        var res = List[GenericValue]()
        if start > stop or start >= sz:
            return res^
        # gh #119: exact count known upfront — reserve once, no log₂(N) reallocs.
        res.reserve(stop - start + 1)

        if is_not_null(self.zip_buf):
            var offset = 0
            var i = 0
            while i <= stop and i < sz:
                var v_len = Int((self.zip_buf.unsafe_offset(offset)).unsafe_bitcast[UInt16]()[])
                if i >= start:
                    res.append(GenericValue.from_ptr_unsafe(self.zip_buf.unsafe_offset(offset).unsafe_offset(2), v_len))
                offset += 2 + v_len
                i += 1
            return res^

        # Segmented traversal
        var global_idx = 0

        # Phase 1: active_head_data[head_off..head_end-1]
        var i = self.head_off
        while i < self.head_end and global_idx <= stop:
            if global_idx >= start:
                res.append(self.active_head_data[unsafe_offset=i])
            i += 1
            global_idx += 1

        # Phase 2: committed head segs in reverse (most recent = lower indices first)
        var seg = self.head_seg_count - 1
        while seg >= self.head_segs_start and global_idx <= stop:
            var j = 0
            while j < Self.SEG_SIZE and global_idx <= stop:
                if global_idx >= start:
                    res.append(self.head_segs[unsafe_offset=seg][unsafe_offset=j])
                j += 1
                global_idx += 1
            seg -= 1

        # Phase 3: committed tail segs in order
        var tseg = 0
        while tseg < self.tail_seg_count and global_idx <= stop:
            var j = 0
            while j < Self.SEG_SIZE and global_idx <= stop:
                if global_idx >= start:
                    res.append(self.tail_segs[unsafe_offset=tseg][unsafe_offset=j])
                j += 1
                global_idx += 1
            tseg += 1

        # Phase 4: active_tail_data[0..tail_count-1]
        var k = 0
        while k < self.tail_count and global_idx <= stop:
            if global_idx >= start:
                res.append(self.active_tail_data[unsafe_offset=k])
            k += 1
            global_idx += 1

        return res^

    def get_all(self) -> List[GenericValue]:
        return self.lrange(0, self.size - 1)

    def deinit(owned self):
        if is_not_null(self.zip_buf):
            self.zip_buf.unsafe_free()
        if is_not_null(self.active_head_data):
            self.active_head_data.unsafe_free()
        for i in range(self.head_segs_start, self.head_seg_count):
            if is_not_null(self.head_segs[i]):
                self.head_segs[i].unsafe_free()
        if is_not_null(self.head_segs):
            self.head_segs.unsafe_free()
        if is_not_null(self.active_tail_data):
            self.active_tail_data.unsafe_free()
        for i in range(self.tail_seg_count):
            if is_not_null(self.tail_segs[i]):
                self.tail_segs[i].unsafe_free()
        if is_not_null(self.tail_segs):
            self.tail_segs.unsafe_free()
