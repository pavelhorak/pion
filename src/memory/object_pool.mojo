from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, unsafe_memset
from std.sys import size_of

struct ObjectPool[T: AnyType](Movable):
    var free_list: Pointer[Pointer[Self.T, MutUntrackedOrigin], MutUntrackedOrigin]
    var capacity: Int
    var head: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.head = 0
        self.free_list = alloc[Pointer[Self.T, MutUntrackedOrigin]](capacity)

        # Every object starts ZEROED, as `_acquire_overflow` hands them out.
        # A list caller reset()s it (a zero SlabList is an empty list), and a
        # skip-list caller constructs over it. The hash pool's owner constructs
        # its objects after this (state.mojo), because a zero SlabHashMap has
        # no slots. gh #478: the list and skip-list pools used to be filled
        # with CONSTRUCTED objects, and that cost memory twice. A SlabList()
        # allocates 24 KB of buffers, and the reset() on acquire freed them
        # untouched and allocated new ones, which on Linux land on fresh pages:
        # RSS grew ~24 KB for each of a worker's first 1,000 lists, then never
        # again. A constructed SlabSkipList(1024) was overwritten without being
        # freed.
        for i in range(capacity):
            var obj = alloc[Self.T](1)
            unsafe_memset(obj.unsafe_bitcast[UInt8](), 0, size_of[Self.T]())
            self.free_list[unsafe_offset=i] = obj

    def __init__(out self, *, deinit take: Self):
        self.capacity = take.capacity
        self.head = take.head
        self.free_list = take.free_list

    @always_inline
    def acquire(mut self) -> Pointer[Self.T, MutUntrackedOrigin]:
        if self.head < self.capacity:
            var ptr = self.free_list[unsafe_offset=self.head]
            self.head += 1
            return ptr
        # Past capacity we fall back to the heap. The pre-allocated objects
        # above are zeroed at construction, and every caller's `reset()`
        # depends on that. A plain `alloc` here returns RECYCLED heap full of
        # the previous tenant's bytes, so `reset()` walked garbage pointers.
        #
        # Measured on 0.923: the 1001st sorted set created in a worker's
        # lifetime (capacity is 1000, and nothing ever calls `release`) killed
        # the worker with SIGSEGV in `SlabSkipList::reset` — reproducible to the
        # exact cycle, and reachable by plain `ZADD` to 1001 distinct keys with
        # no deletion involved.
        #
        # Zeroing makes the fallback match the state the pooled objects are in,
        # which is the invariant reset() is written against.
        return self._acquire_overflow()

    @no_inline
    def _acquire_overflow(mut self) -> Pointer[Self.T, MutUntrackedOrigin]:
        """Outlined cold path — `acquire` is @always_inline and lands at 26
        sites, several of them arms inside `process_data_plane`. Keeping the
        memset inline bloated the fast-path loop for a branch that is not taken
        until the pool is exhausted. Same reasoning as the WAL's `_make_room`."""
        var fresh = alloc[Self.T](1)
        unsafe_memset(fresh.unsafe_bitcast[UInt8](), 0, size_of[Self.T]())
        return fresh

    @always_inline
    def release(mut self, ptr: Pointer[Self.T, MutUntrackedOrigin]):
        if self.head > 0:
            self.head -= 1
            self.free_list[self.head] = ptr
        else:
            ptr.unsafe_free()
            
    def free_all(var self):
        # Note: In a real move-ready struct, we'd check if free_list is null
        # but for this prototype we assume it is valid if capacity > 0
        if self.capacity > 0:
            # for i in range(self.capacity):
            #      self.free_list[i].unsafe_free()
            # self.free_list.unsafe_free()
            pass
