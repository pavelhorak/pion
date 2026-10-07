from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.sys import size_of, CompilationTarget
from std.collections import List
from std.ffi import external_call

struct SlabAllocator[T: AnyType](Movable):
    var item_size: Int
    var items_per_slab: Int
    var slabs: List[Pointer[UInt8, MutUntrackedOrigin]]
    var slab_sizes: List[Int]
    var free_list: List[Pointer[UInt8, MutUntrackedOrigin]]
    var current_slab: Pointer[UInt8, MutUntrackedOrigin]
    var current_offset: Int
    var use_huge_pages: Bool

    def __init__(out self, items_per_slab: Int, item_size: Int = -1, use_huge_pages: Bool = False):
        if item_size != -1:
            self.item_size = (item_size + 7) & ~7
        else:
            self.item_size = (size_of[Self.T]() + 7) & ~7
        self.items_per_slab = items_per_slab
        self.slabs = List[Pointer[UInt8, MutUntrackedOrigin]]()
        self.slab_sizes = List[Int]()
        self.free_list = List[Pointer[UInt8, MutUntrackedOrigin]]()
        self.current_offset = 0
        self.current_slab = null_ptr[UInt8, MutUntrackedOrigin]()
        self.use_huge_pages = use_huge_pages
        
        var total_size = self.item_size * self.items_per_slab
        var initial_slab = self._mmap_alloc(total_size)
        self.slabs.append(initial_slab)
        self.slab_sizes.append(total_size)
        self.current_slab = initial_slab

    def __init__(out self, *, deinit take: Self):
        self.item_size = take.item_size
        self.items_per_slab = take.items_per_slab
        self.slabs = take.slabs^
        self.slab_sizes = take.slab_sizes^
        self.free_list = take.free_list^
        self.current_slab = take.current_slab
        self.current_offset = take.current_offset
        self.use_huge_pages = take.use_huge_pages

    def _mmap_alloc(self, size: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        var alloc_size = size
        var ptr: Pointer[UInt8, MutUntrackedOrigin]
        comptime if CompilationTarget.is_linux():
            # Linux: MAP_ANONYMOUS=0x20, MAP_PRIVATE=0x02
            var flags = 0x22
            if self.use_huge_pages:
                # MAP_HUGETLB=0x40000; requires hugepages configured in the kernel.
                # Fall back to normal pages if it fails.
                var huge_flags = flags | 0x40000
                alloc_size = (size + 2097151) & ~2097151
                ptr = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
                    null_ptr[UInt8, MutUntrackedOrigin](), alloc_size, 0x3, huge_flags, -1, 0)
                if Int(ptr) == -1:
                    # Hugepage mmap failed (no hugepages configured) — retry with normal pages.
                    alloc_size = size
                    ptr = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
                        null_ptr[UInt8, MutUntrackedOrigin](), alloc_size, 0x3, flags, -1, 0)
            else:
                ptr = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
                    null_ptr[UInt8, MutUntrackedOrigin](), alloc_size, 0x3, flags, -1, 0)
        else:
            # macOS: MAP_ANON=0x1000, MAP_PRIVATE=0x0002
            var flags = 0x1002
            if self.use_huge_pages:
                # VM_FLAGS_SUPERPAGE_SIZE_2MB = 0x10000
                flags |= 0x10000
                alloc_size = (size + 2097151) & ~2097151
            ptr = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
                null_ptr[UInt8, MutUntrackedOrigin](), alloc_size, 0x3, flags, -1, 0)
        return ptr

    def _mmap_free(self, ptr: Pointer[UInt8, MutUntrackedOrigin], size: Int):
        _ = external_call["munmap", Int32](ptr, size)


    def reset(mut self):
        while len(self.slabs) > 1:
            var slab = self.slabs.pop()
            var size = self.slab_sizes.pop()
            self._mmap_free(slab, size)
        if len(self.slabs) > 0:
            self.current_slab = self.slabs[0]
            # `allocate()` grows `items_per_slab` (doubling) every time a slab
            # fills, and it is also the bound that decides when the CURRENT slab
            # is exhausted. Dropping back to slab[0] without restoring the bound
            # left the allocator believing the 16-item first slab held however
            # many items the largest slab had — so it kept bumping
            # `current_offset` and handing out `slab[0] + offset*item_size`
            # addresses far past the end of that mapping. The first ones landed
            # inside the page mmap rounds up to (~93 nodes for a 176 B
            # SkipListNode), which is why small sets appeared to work; past that
            # it is a page-aligned SIGSEGV.
            #
            # Every rebuild-style handler hits this: ZREM / ZREMRANGEBY* / the
            # GEO and ZUNION/ZINTER stores all collect, `reset()`, then reinsert
            # the survivors. `ZREM` on a sorted set of ~200 members killed the
            # server outright. Re-derive the bound from the surviving slab's own
            # byte size so it can never disagree with the memory that is there.
            self.items_per_slab = self.slab_sizes[0] // self.item_size
        self.current_offset = 0
        self.free_list = List[Pointer[UInt8, MutUntrackedOrigin]]()

    def release_all(mut self):
        """gh #369: unmap EVERY slab, including the first one `reset()` keeps.
        For an allocator whose owner is being destroyed — after this the
        allocator must not be used again."""
        for i in range(len(self.slabs)):
            self._mmap_free(self.slabs[i], self.slab_sizes[i])
        self.slabs.clear()
        self.slab_sizes.clear()
        self.free_list.clear()
        self.current_slab = null_ptr[UInt8, MutUntrackedOrigin]()
        self.current_offset = 0
        self.items_per_slab = 0

    def allocate(mut self) -> Pointer[Self.T, MutUntrackedOrigin]:
        if len(self.free_list) > 0:
            return self.free_list.pop().unsafe_bitcast[Self.T]()
        
        if self.current_offset >= self.items_per_slab:
            # Adaptive Slab Resizing: double the items per slab up to a sane limit to avoid excessive virtual memory reservations
            if self.items_per_slab < 10000000:
                self.items_per_slab *= 2
            var total_size = self.item_size * self.items_per_slab
            var new_slab = self._mmap_alloc(total_size)
            self.slabs.append(new_slab)
            self.slab_sizes.append(total_size)
            self.current_slab = new_slab
            self.current_offset = 0

        var ptr = self.current_slab.unsafe_offset((self.current_offset * self.item_size))
        self.current_offset += 1
        return ptr.unsafe_bitcast[Self.T]()

    def deallocate(mut self, ptr: Pointer[Self.T, MutUntrackedOrigin]):
        self.free_list.append(ptr.unsafe_bitcast[UInt8]())

    def free_all(var self):
        for i in range(len(self.slabs)):
            self._mmap_free(self.slabs[i], self.slab_sizes[i])
