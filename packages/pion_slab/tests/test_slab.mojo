# pion_slab: the allocator and the pool Pion's keyspace and lists are built on.
from pion_slab import SlabAllocator, ObjectPool, is_not_null
from std.testing import assert_equal, assert_true


def test_slab_reuses_freed_chunks() raises:
    var allocator = SlabAllocator[UInt64](4)
    var p1 = allocator.allocate()
    var p2 = allocator.allocate()
    var p3 = allocator.allocate()
    var p4 = allocator.allocate()
    assert_true(p1 != p2 and p2 != p3 and p3 != p4)
    var p5 = allocator.allocate()          # past the first slab
    assert_true(p5 != p4)
    p1[] = 11
    p5[] = 55
    assert_equal(p1[], 11)
    assert_equal(p5[], 55)
    allocator.deallocate(p2)
    var p6 = allocator.allocate()
    assert_equal(p6, p2)                   # LIFO free list


def test_slab_reset_keeps_working() raises:
    var allocator = SlabAllocator[UInt64](16)
    for _ in range(100):                   # grows past the first slab
        var p = allocator.allocate()
        p[] = 7
    allocator.reset()
    for i in range(100):
        var p = allocator.allocate()
        p[] = UInt64(i)
        assert_equal(p[], UInt64(i))


def test_object_pool() raises:
    var pool = ObjectPool[UInt64](8)
    var a = pool.acquire()
    var b = pool.acquire()
    assert_true(is_not_null(a) and is_not_null(b) and a != b)
    pool.release(a)
    var c = pool.acquire()
    assert_equal(c, a)


def main() raises:
    test_slab_reuses_freed_chunks()
    test_slab_reset_keeps_working()
    test_object_pool()
    print("pion_slab: all tests passed")
