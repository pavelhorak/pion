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
    test_slab_skip_list()
    print("--- ALL PHASE 5 TESTS PASSED ---")
