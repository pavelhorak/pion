from src.memory.slab_allocator import SlabAllocator
from std.memory.unsafe_pointer import UnsafePointer
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

def main() raises:
    test_slab_allocator()
