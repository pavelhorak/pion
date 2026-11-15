from src.memory.slab_allocator import SlabAllocator
from std.testing import assert_true, assert_equal

def test_slab_allocator() raises:
    # Use UInt8 as T for a 16-byte chunk size if possible?
    # No, it's generic now. Let's use SIMD[DType.uint8, 16] for 16-byte chunks.
    var allocator = SlabAllocator[UInt8](4) # 1 byte per chunk, 4 chunks per slab
    
    # Allocate 4 chunks (fills first slab)
    var p1 = allocator.allocate()
    var p2 = allocator.allocate()
    var p3 = allocator.allocate()
    var p4 = allocator.allocate()
    
    # Check that they are distinct (simple check)
    assert_true(p1 != p2)
    assert_true(p2 != p3)
    assert_true(p3 != p4)

    # Allocate 5th chunk (should trigger new slab allocation)
    var p5 = allocator.allocate()
    assert_true(p5 != p4)
    
    # Deallocate p2
    allocator.deallocate(p2)
    
    # Allocate again, should get p2 back (LIFO behavior of free_list)
    var p6 = allocator.allocate()
    assert_equal(p6, p2)

    print("Generic Slab Allocator Test Passed!")

def test_mmap_failure_is_null() raises:
    # A mapping the kernel must refuse (2^62 bytes) comes back null, never
    # MAP_FAILED (-1), which the allocator used to hand out as a slab.
    var allocator = SlabAllocator[UInt8](4)
    var bad = allocator._mmap_alloc(1 << 62)
    assert_equal(Int(bad), 0)
    var good = allocator._mmap_alloc(4096)
    assert_true(Int(good) != 0 and Int(good) != -1)
    allocator._mmap_free(good, 4096)
    # A huge-page request on a machine without them falls back to normal
    # pages (Apple Silicon has no superpages; most Linux boxes have none set up).
    var huge = SlabAllocator[UInt8](4096, use_huge_pages=True)
    var hp = huge.allocate()
    hp[] = 7
    assert_equal(Int(hp[]), 7)
    print("mmap failure test passed")

def main() raises:
    test_slab_allocator()
    test_mmap_failure_is_null()
