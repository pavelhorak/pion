from std.collections import Dict, List
from .slab_allocator import SlabAllocator
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.sys import size_of
from std.math import min, max

@fieldwise_init
struct PageMetadata(Copyable, Movable, ImplicitlyCopyable):
    var id: Int
    var is_dirty: Bool
    var ref_count: Int
    var buffer: Pointer[UInt8, MutUntrackedOrigin]

struct BufferPool:
    var meta_allocator: SlabAllocator[PageMetadata]
    var page_table: Dict[Int, Pointer[PageMetadata, MutUntrackedOrigin]]
    var capacity: Int
    # ARC Lists
    var t1: List[Int] # Recent hits
    var t2: List[Int] # Frequent hits
    var b1: List[Int] # Recent evictions (ghost)
    var b2: List[Int] # Frequent evictions (ghost)
    var p: Int # Target size for t1

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.meta_allocator = SlabAllocator[PageMetadata](capacity)
        self.page_table = Dict[Int, Pointer[PageMetadata, MutUntrackedOrigin]]()
        self.t1 = List[Int]()
        self.t2 = List[Int]()
        self.b1 = List[Int]()
        self.b2 = List[Int]()
        self.p = 0

    def get_page(mut self, page_id: Int) raises -> Pointer[UInt8, MutUntrackedOrigin]:
        # Case 1: Page is in T1 or T2 (Cache Hit)
        if self._in_list(self.t1, page_id):
            self._move_t1_to_t2(page_id)
            return self.page_table[page_id][].buffer
        
        if self._in_list(self.t2, page_id):
            self._move_t2_to_end(page_id)
            return self.page_table[page_id][].buffer

        # Case 2: Page is in B1 (Ghost Hit - Recent)
        if self._in_list(self.b1, page_id):
            var delta = 1
            if len(self.b2) > len(self.b1) and len(self.b1) > 0:
                delta = len(self.b2) // len(self.b1)
            self.p = min(self.capacity, self.p + delta)
            self.replace(page_id)
            self._move_b1_to_t2(page_id)
            return self._fetch_page(page_id)

        # Case 3: Page is in B2 (Ghost Hit - Frequent)
        if self._in_list(self.b2, page_id):
            var delta = 1
            if len(self.b1) > len(self.b2) and len(self.b2) > 0:
                delta = len(self.b1) // len(self.b2)
            self.p = max(0, self.p - delta)
            self.replace(page_id)
            self._move_b2_to_t2(page_id)
            return self._fetch_page(page_id)

        # Case 4: Cache Miss
        if len(self.t1) + len(self.b1) == self.capacity:
            if len(self.t1) < self.capacity:
                _ = self.b1.pop(0)
                self.replace(page_id)
            else:
                var victim = self.t1.pop(0)
                self._evict_from_memory(victim)
        else:
            var total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.capacity:
                if total == 2 * self.capacity:
                    _ = self.b2.pop(0)
                self.replace(page_id)

        self.t1.append(page_id)
        return self._fetch_page(page_id)

    def replace(mut self, page_id: Int):
        if len(self.t1) > 0 and (len(self.t1) > self.p or (self._in_list(self.b2, page_id) and len(self.t1) == self.p)):
            var victim = self.t1.pop(0)
            self.b1.append(victim)
            self._evict_from_memory(victim)
        else:
            if len(self.t2) > 0:
                var victim = self.t2.pop(0)
                self.b2.append(victim)
                self._evict_from_memory(victim)

    def _fetch_page(mut self, page_id: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        var buffer = alloc[UInt8](4096)
        var meta_ptr = self.meta_allocator.allocate()
        meta_ptr.unsafe_write(PageMetadata(page_id, False, 0, buffer))
        self.page_table[page_id] = meta_ptr
        return buffer

    def _evict_from_memory(mut self, page_id: Int):
        try:
            var meta_ptr = self.page_table[page_id]
            if meta_ptr[].is_dirty:
                print("BufferPool ARC: Flushing dirty page " + String(page_id) + " to SSD...")
            meta_ptr[].buffer.unsafe_free()
            meta_ptr.unsafe_deinit_pointee()
            self.meta_allocator.deallocate(meta_ptr)
            _ = self.page_table.pop(page_id)
        except:
            pass

    def _in_list(self, l: List[Int], val: Int) -> Bool:
        for i in range(len(l)):
            if l[i] == val: return True
        return False

    def _move_t1_to_t2(mut self, val: Int):
        for i in range(len(self.t1)):
            if self.t1[i] == val:
                _ = self.t1.pop(i)
                break
        self.t2.append(val)

    def _move_t2_to_end(mut self, val: Int):
        for i in range(len(self.t2)):
            if self.t2[i] == val:
                _ = self.t2.pop(i)
                break
        self.t2.append(val)

    def _move_b1_to_t2(mut self, val: Int):
        for i in range(len(self.b1)):
            if self.b1[i] == val:
                _ = self.b1.pop(i)
                break
        self.t2.append(val)

    def _move_b2_to_t2(mut self, val: Int):
        for i in range(len(self.b2)):
            if self.b2[i] == val:
                _ = self.b2.pop(i)
                break
        self.t2.append(val)

    def mark_dirty(mut self, page_id: Int):
        try:
            var meta = self.page_table[page_id]
            meta[].is_dirty = True
        except:
            pass

    def free_all(var self):
        for entry in self.page_table.items():
            try:
                if entry.value[].is_dirty:
                    print("BufferPool: Shutdown flush of page " + String(entry.key))
                entry.value[].buffer.unsafe_free()
                entry.value[].unsafe_deinit_pointee()
            except:
                pass
