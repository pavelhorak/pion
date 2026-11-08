from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap
from src.common.skip_list import SlabSkipList
from std.testing import assert_equal, assert_true

def test_generic_value() raises:
    print("Testing GenericValue...")
    var v1 = GenericValue.from_int(42)
    assert_equal(v1.type.value, ValueType.INT)
    assert_equal(v1.as_int(), 42)
    
    var v2 = GenericValue.from_float(3.14)
    assert_equal(v2.type.value, ValueType.FLOAT)
    assert_true(v2.as_float() > 3.1 and v2.as_float() < 3.2)
    print("GenericValue tests passed!")

def test_slab_hash_map() raises:
    print("Testing SlabHashMap...")
    var hm = SlabHashMap(16, 100)
    
    hm.set("key1", GenericValue.from_int(100))
    hm.set("key2", GenericValue.from_int(200))
    
    var v1 = hm.get("key1")
    assert_equal(v1.as_int(), 100)
    
    var v2 = hm.get("key2")
    assert_equal(v2.as_int(), 200)
    
    var v3 = hm.get("unknown")
    assert_equal(v3.type.value, ValueType.NONE)
    print("SlabHashMap tests passed!")

def test_table_growth() raises:
    """The table grows past 7/8 full, and a big table's arrays are mapped
    (TABLE_MAP_MIN): they start zero instead of being written, and go back
    with munmap on a rebuild and when the map is dropped."""
    print("Testing table growth...")
    for _ in range(3):              # build and drop: a bad unmap shows on reuse
        var hm = SlabHashMap(16)
        # 220,000 keys: under 7/8 of 262,144 (229,376), over 70% of it, so
        # the old trigger would have doubled to 524,288.
        var n = 220000
        for i in range(n):
            hm.set("k" + String(i), GenericValue.from_int(Int64(i)))
        assert_equal(hm.capacity, 262144)
        assert_equal(hm.size, n)
        for i in range(n):
            assert_equal(Int(hm.get("k" + String(i)).as_int()), i)
        assert_equal(hm.get("missing").type.value, ValueType.NONE)

        # Churn: half the keys go and as many new ones come, into tombstones.
        for i in range(0, n, 2):
            assert_true(hm.remove("k" + String(i)))
        for i in range(n, n + n // 2):
            hm.set("k" + String(i), GenericValue.from_int(Int64(i)))
        assert_equal(hm.size, n)
        for i in range(n + n // 2):
            var v = hm.get("k" + String(i))
            if i < n and i % 2 == 0:
                assert_equal(v.type.value, ValueType.NONE)
            else:
                assert_equal(Int(v.as_int()), i)

        # An access table on a mapped table: it starts zero, carries its
        # may-grow word across a doubling, and is unmapped.
        hm.enable_access_table()
        var t = hm.access_table()
        assert_equal(Int(t[unsafe_offset=hm.capacity]), 1)
        assert_equal(Int(t[unsafe_offset=0]) + Int(t[unsafe_offset=hm.capacity - 1]), 0)
        # 340,000 live keys: past 7/8 of 262,144, under 7/8 of 524,288.
        for i in range(n + n // 2, n + n // 2 + 120000):
            hm.set("k" + String(i), GenericValue.from_int(Int64(i)))
        assert_equal(hm.capacity, 524288)
        assert_equal(Int(hm.access_table()[unsafe_offset=hm.capacity]), 1)
        assert_equal(Int(hm.get("k" + String(n + n // 2 + 119999)).as_int()), n + n // 2 + 119999)
        hm.disable_access_table()
        assert_equal(Int(hm.access_table()), 0)
    print("table growth tests passed!")

def test_slab_skip_list() raises:
    print("Testing SlabSkipList...")
    var sl = SlabSkipList(100)
    
    sl.insert(10.5, GenericValue.from_int(1))
    sl.insert(5.0, GenericValue.from_int(2))
    sl.insert(20.0, GenericValue.from_int(3))
    
    assert_equal(sl.length, 3)
    
    var range1 = sl.get_range(0.0, 15.0)
    assert_equal(len(range1), 2)
    # Scores 5.0 and 10.5 are in range
    
    var range2 = sl.get_range(15.0, 25.0)
    assert_equal(len(range2), 1)
    assert_equal(range2[0].as_int(), 3)
    print("SlabSkipList tests passed!")

def main() raises:
    # Assertions PROPAGATE: the old main caught every failure, printed it and
    # exited 0, so this test could not fail even before it stopped compiling.
    test_generic_value()
    test_slab_hash_map()
    test_table_growth()
    test_slab_skip_list()
    print("--- ALL PHASE 5 TESTS PASSED ---")
