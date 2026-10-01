from src.common.list import SlabList
from src.common.value import GenericValue, ValueType
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc
from std.testing import assert_true, assert_equal


def make_value(s: String) -> GenericValue:
    """Create a GenericValue from a String via from_string."""
    return GenericValue.from_string(s)


def values_match(val: GenericValue, expected: String) -> Bool:
    """Check if a GenericValue matches an expected string by comparing string_len
    and byte content. Works for both SSO and STRING (from_ptr_unsafe) types."""
    if not val.is_string():
        return False
    var expected_val = GenericValue.from_string(expected)
    var l1 = val.string_len()
    var l2 = expected_val.string_len()
    if l1 != l2:
        return False
    # For SSO values we can compare directly
    if val.type.value == ValueType.STRING_SSO and expected_val.type.value == ValueType.STRING_SSO:
        return val == expected_val
    # For STRING (from_ptr_unsafe in lrange), compare bytes
    var buf1 = alloc[UInt8](l1)
    var buf2 = alloc[UInt8](l2)
    val.copy_to(buf1)
    expected_val.copy_to(buf2)
    var same = True
    for i in range(l1):
        if buf1[i] != buf2[i]:
            same = False
            break
    buf1.free()
    buf2.free()
    return same


def assert_value_str(val: GenericValue, expected: String) raises:
    """Assert a GenericValue's string content matches expected."""
    assert_true(values_match(val, expected), "Value mismatch: expected '" + expected + "'")


def test_lpush_basic() raises:
    """Lpush 3 items, llen = 3, lrange returns correct LIFO order."""
    print("  test_lpush_basic...")
    var lst = SlabList()
    lst.lpush(make_value("a"))
    lst.lpush(make_value("b"))
    lst.lpush(make_value("c"))
    assert_equal(lst.llen(), 3)
    # LPUSH order: last pushed = index 0, so: c, b, a
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 3)
    assert_value_str(items[0], "c")
    assert_value_str(items[1], "b")
    assert_value_str(items[2], "a")
    print("    PASS")


def test_rpush_basic() raises:
    """Rpush 3 items, verify FIFO order."""
    print("  test_rpush_basic...")
    var lst = SlabList()
    lst.rpush(make_value("x"))
    lst.rpush(make_value("y"))
    lst.rpush(make_value("z"))
    assert_equal(lst.llen(), 3)
    # RPUSH order: first pushed = index 0, so: x, y, z
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 3)
    assert_value_str(items[0], "x")
    assert_value_str(items[1], "y")
    assert_value_str(items[2], "z")
    print("    PASS")


def test_lpop() raises:
    """Lpop returns head element (most recently lpushed)."""
    print("  test_lpop...")
    var lst = SlabList()
    lst.lpush(make_value("first"))
    lst.lpush(make_value("second"))
    lst.lpush(make_value("third"))
    # Order: third, second, first
    var popped = lst.lpop()
    assert_value_str(popped, "third")
    assert_equal(lst.llen(), 2)
    print("    PASS")


def test_rpop() raises:
    """Rpop returns tail element."""
    print("  test_rpop...")
    var lst = SlabList()
    lst.rpush(make_value("a"))
    lst.rpush(make_value("b"))
    lst.rpush(make_value("c"))
    # Order: a, b, c -- rpop returns c
    var popped = lst.rpop()
    assert_value_str(popped, "c")
    assert_equal(lst.llen(), 2)
    print("    PASS")


def test_ziplist_100_items() raises:
    """Push 100 items (under 1024 threshold), lrange all, verify count and order."""
    print("  test_ziplist_100_items...")
    var lst = SlabList()
    for i in range(100):
        lst.rpush(make_value(String(i)))
    assert_equal(lst.llen(), 100)
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 100)
    # Verify first and last elements
    assert_value_str(items[0], "0")
    assert_value_str(items[99], "99")
    print("    PASS")


def test_quicklist_transition() raises:
    """Push 1025 items (crosses 1024 threshold), verify llen = 1025."""
    print("  test_quicklist_transition...")
    var lst = SlabList()
    for i in range(1025):
        lst.rpush(make_value(String(i)))
    assert_equal(lst.llen(), 1025)
    print("    PASS")


def test_quicklist_lpop_rpop() raises:
    """After quicklist transition, lpop and rpop still return correct elements."""
    print("  test_quicklist_lpop_rpop...")
    var lst = SlabList()
    for i in range(1025):
        lst.rpush(make_value(String(i)))
    # lpop should return first element ("0")
    var head = lst.lpop()
    assert_value_str(head, "0")
    assert_equal(lst.llen(), 1024)
    # rpop should return last element ("1024")
    var tail = lst.rpop()
    assert_value_str(tail, "1024")
    assert_equal(lst.llen(), 1023)
    print("    PASS")


def test_quicklist_lrange() raises:
    """After quicklist transition, lrange returns correct subset."""
    print("  test_quicklist_lrange...")
    var lst = SlabList()
    for i in range(1100):
        lst.rpush(make_value(String(i)))
    # Read a small range near the beginning
    var items = lst.lrange(0, 4)
    assert_equal(len(items), 5)
    assert_value_str(items[0], "0")
    assert_value_str(items[4], "4")
    # Read a small range near the end
    var end_items = lst.lrange(1095, 1099)
    assert_equal(len(end_items), 5)
    assert_value_str(end_items[0], "1095")
    assert_value_str(end_items[4], "1099")
    print("    PASS")


def test_mixed_lpush_rpush() raises:
    """Alternate lpush and rpush, verify lrange ordering."""
    print("  test_mixed_lpush_rpush...")
    var lst = SlabList()
    # rpush "a", lpush "b", rpush "c", lpush "d"
    # After rpush "a": [a]
    # After lpush "b": [b, a]
    # After rpush "c": [b, a, c]
    # After lpush "d": [d, b, a, c]
    lst.rpush(make_value("a"))
    lst.lpush(make_value("b"))
    lst.rpush(make_value("c"))
    lst.lpush(make_value("d"))
    assert_equal(lst.llen(), 4)
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 4)
    assert_value_str(items[0], "d")
    assert_value_str(items[1], "b")
    assert_value_str(items[2], "a")
    assert_value_str(items[3], "c")
    print("    PASS")


def test_lpop_empty() raises:
    """Lpop on empty list returns NONE."""
    print("  test_lpop_empty...")
    var lst = SlabList()
    var val = lst.lpop()
    assert_true(val.is_none(), "lpop on empty must return NONE")
    print("    PASS")


def test_rpop_empty() raises:
    """Rpop on empty list returns NONE."""
    print("  test_rpop_empty...")
    var lst = SlabList()
    var val = lst.rpop()
    assert_true(val.is_none(), "rpop on empty must return NONE")
    print("    PASS")


def test_pop_all_then_empty() raises:
    """Push items, pop all, verify empty behavior."""
    print("  test_pop_all_then_empty...")
    var lst = SlabList()
    lst.rpush(make_value("x"))
    lst.rpush(make_value("y"))
    _ = lst.lpop()
    _ = lst.lpop()
    assert_equal(lst.llen(), 0)
    var val = lst.lpop()
    assert_true(val.is_none(), "lpop after draining must return NONE")
    print("    PASS")


def test_sso_boundary_values() raises:
    """Push values at SSO boundary (23B, 24B) and verify round-trip."""
    print("  test_sso_boundary_values...")
    var lst = SlabList()
    # 23 bytes -- SSO
    var s23 = "abcdefghijklmnopqrstuvw"  # 23 chars
    lst.rpush(make_value(s23))
    # 24 bytes -- heap (but in ziplist, stored as raw bytes)
    var s24 = "abcdefghijklmnopqrstuvwx"  # 24 chars
    lst.rpush(make_value(s24))
    assert_equal(lst.llen(), 2)
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 2)
    assert_equal(items[0].string_len(), 23)
    assert_equal(items[1].string_len(), 24)
    print("    PASS")


def test_lrange_negative_indices() raises:
    """Lrange with negative indices works like Redis."""
    print("  test_lrange_negative_indices...")
    var lst = SlabList()
    for i in range(5):
        lst.rpush(make_value(String(i)))
    # lrange(-2, -1) = last 2 elements
    var items = lst.lrange(-2, -1)
    assert_equal(len(items), 2)
    assert_value_str(items[0], "3")
    assert_value_str(items[1], "4")
    print("    PASS")


def test_lrange_out_of_bounds() raises:
    """Lrange with out-of-bounds stop is clamped to list end."""
    print("  test_lrange_out_of_bounds...")
    var lst = SlabList()
    lst.rpush(make_value("a"))
    lst.rpush(make_value("b"))
    var items = lst.lrange(0, 100)
    assert_equal(len(items), 2)
    print("    PASS")


def test_lrange_empty() raises:
    """Lrange on empty list returns empty."""
    print("  test_lrange_empty...")
    var lst = SlabList()
    var items = lst.lrange(0, -1)
    assert_equal(len(items), 0)
    print("    PASS")


def test_lpush_quicklist_large() raises:
    """Lpush 1100 items into quicklist, verify order with lrange."""
    print("  test_lpush_quicklist_large...")
    var lst = SlabList()
    for i in range(1100):
        lst.lpush(make_value(String(i)))
    assert_equal(lst.llen(), 1100)
    # LPUSH: last pushed = index 0, so index 0 = "1099"
    var head_items = lst.lrange(0, 0)
    assert_equal(len(head_items), 1)
    assert_value_str(head_items[0], "1099")
    # Last item should be "0"
    var tail_items = lst.lrange(1099, 1099)
    assert_equal(len(tail_items), 1)
    assert_value_str(tail_items[0], "0")
    print("    PASS")


def main() raises:
    print("=== SlabList Tests ===")
    test_lpush_basic()
    test_rpush_basic()
    test_lpop()
    test_rpop()
    test_ziplist_100_items()
    test_quicklist_transition()
    test_quicklist_lpop_rpop()
    test_quicklist_lrange()
    test_mixed_lpush_rpush()
    test_lpop_empty()
    test_rpop_empty()
    test_pop_all_then_empty()
    test_sso_boundary_values()
    test_lrange_negative_indices()
    test_lrange_out_of_bounds()
    test_lrange_empty()
    test_lpush_quicklist_large()
    print("=== ALL SlabList TESTS PASSED ===")
