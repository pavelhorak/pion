from src.common.value import GenericValue, ValueType
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc
from std.testing import assert_true, assert_equal


def fill_buf(buf: UnsafePointer[UInt8, MutUntrackedOrigin], s: String, length: Int):
    """Copy string bytes into a pre-allocated UInt8 buffer."""
    var b = s.as_bytes()
    for i in range(length):
        buf[i] = b[i]


def test_sso_from_ptr() raises:
    """Verify from_ptr with <=23 bytes produces STRING_SSO."""
    print("  test_sso_from_ptr...")
    var buf = alloc[UInt8](8)
    fill_buf(buf, "hello", 5)
    var val = GenericValue.from_ptr(buf, 5)
    assert_equal(val.type.value, ValueType.STRING_SSO)
    assert_equal(val.string_len(), 5)
    buf.free()
    print("    PASS")


def test_heap_from_ptr() raises:
    """Verify from_ptr with >23 bytes produces heap STRING."""
    print("  test_heap_from_ptr...")
    var s = "abcdefghijklmnopqrstuvwxyz"  # 26 bytes
    var buf = alloc[UInt8](26)
    fill_buf(buf, s, 26)
    var val = GenericValue.from_ptr(buf, 26)
    assert_equal(val.type.value, ValueType.STRING)
    assert_equal(val.string_len(), 26)
    buf.free()
    print("    PASS")


def test_from_ptr_unsafe_always_string() raises:
    """Verify from_ptr_unsafe always produces STRING type, never SSO."""
    print("  test_from_ptr_unsafe_always_string...")
    var buf = alloc[UInt8](4)
    fill_buf(buf, "hi", 2)
    var val = GenericValue.from_ptr_unsafe(buf, 2)
    assert_equal(val.type.value, ValueType.STRING)
    assert_equal(val.string_len(), 2)
    buf.free()
    print("    PASS")


def test_sso_vs_unsafe_not_equal() raises:
    """CRITICAL: from_ptr (SSO) and from_ptr_unsafe (STRING) with same bytes
    are NOT equal. __eq__ returns False on type mismatch. This is why
    from_ptr_unsafe must never be used for hash map key lookups (T3.3 bug)."""
    print("  test_sso_vs_unsafe_not_equal...")
    var buf = alloc[UInt8](8)
    fill_buf(buf, "testkey", 7)
    var sso_val = GenericValue.from_ptr(buf, 7)
    var str_val = GenericValue.from_ptr_unsafe(buf, 7)
    assert_equal(sso_val.type.value, ValueType.STRING_SSO)
    assert_equal(str_val.type.value, ValueType.STRING)
    # They must NOT be equal despite having the same bytes
    assert_true(not (sso_val == str_val), "SSO and STRING with same bytes must NOT be equal")
    buf.free()
    print("    PASS")


def test_two_sso_from_ptr_equal() raises:
    """CRITICAL: Two from_ptr keys with same bytes <=23B should be equal (both SSO)."""
    print("  test_two_sso_from_ptr_equal...")
    var buf1 = alloc[UInt8](8)
    var buf2 = alloc[UInt8](8)
    fill_buf(buf1, "mykey", 5)
    fill_buf(buf2, "mykey", 5)
    var val1 = GenericValue.from_ptr(buf1, 5)
    var val2 = GenericValue.from_ptr(buf2, 5)
    assert_equal(val1.type.value, ValueType.STRING_SSO)
    assert_equal(val2.type.value, ValueType.STRING_SSO)
    assert_true(val1 == val2, "Two SSO values with same bytes must be equal")
    buf1.free()
    buf2.free()
    print("    PASS")


def test_two_heap_from_ptr_equal() raises:
    """CRITICAL: Two from_ptr keys with same bytes >23B should be equal (both heap STRING)."""
    print("  test_two_heap_from_ptr_equal...")
    var s = "abcdefghijklmnopqrstuvwxyz01"  # 28 bytes
    var buf1 = alloc[UInt8](28)
    var buf2 = alloc[UInt8](28)
    fill_buf(buf1, s, 28)
    fill_buf(buf2, s, 28)
    var val1 = GenericValue.from_ptr(buf1, 28)
    var val2 = GenericValue.from_ptr(buf2, 28)
    assert_equal(val1.type.value, ValueType.STRING)
    assert_equal(val2.type.value, ValueType.STRING)
    assert_true(val1 == val2, "Two heap STRING values with same bytes must be equal")
    buf1.free()
    buf2.free()
    print("    PASS")


def test_hash_consistency_sso() raises:
    """Two equal SSO values should have the same __hash__."""
    print("  test_hash_consistency_sso...")
    var buf1 = alloc[UInt8](8)
    var buf2 = alloc[UInt8](8)
    fill_buf(buf1, "hashme", 6)
    fill_buf(buf2, "hashme", 6)
    var val1 = GenericValue.from_ptr(buf1, 6)
    var val2 = GenericValue.from_ptr(buf2, 6)
    assert_equal(val1.__hash__(), val2.__hash__())
    buf1.free()
    buf2.free()
    print("    PASS")


def test_hash_consistency_heap() raises:
    """Two equal heap STRING values should have the same __hash__."""
    print("  test_hash_consistency_heap...")
    var s = "abcdefghijklmnopqrstuvwxyz1234"  # 30 bytes
    var buf1 = alloc[UInt8](30)
    var buf2 = alloc[UInt8](30)
    fill_buf(buf1, s, 30)
    fill_buf(buf2, s, 30)
    var val1 = GenericValue.from_ptr(buf1, 30)
    var val2 = GenericValue.from_ptr(buf2, 30)
    assert_equal(val1.__hash__(), val2.__hash__())
    buf1.free()
    buf2.free()
    print("    PASS")


def test_sso_boundary_22_bytes() raises:
    """Test at exactly 22 bytes -- should be SSO."""
    print("  test_sso_boundary_22_bytes...")
    var s = "abcdefghijklmnopqrstuv"  # 22 bytes
    var buf = alloc[UInt8](22)
    fill_buf(buf, s, 22)
    var val = GenericValue.from_ptr(buf, 22)
    assert_equal(val.type.value, ValueType.STRING_SSO)
    assert_equal(val.string_len(), 22)
    buf.free()
    print("    PASS")


def test_sso_boundary_23_bytes() raises:
    """Test at exactly 23 bytes -- should be SSO (boundary)."""
    print("  test_sso_boundary_23_bytes...")
    var s = "abcdefghijklmnopqrstuvw"  # 23 bytes
    var buf = alloc[UInt8](23)
    fill_buf(buf, s, 23)
    var val = GenericValue.from_ptr(buf, 23)
    assert_equal(val.type.value, ValueType.STRING_SSO)
    assert_equal(val.string_len(), 23)
    buf.free()
    print("    PASS")


def test_sso_boundary_24_bytes() raises:
    """Test at exactly 24 bytes -- should be heap STRING."""
    print("  test_sso_boundary_24_bytes...")
    var s = "abcdefghijklmnopqrstuvwx"  # 24 bytes
    var buf = alloc[UInt8](24)
    fill_buf(buf, s, 24)
    var val = GenericValue.from_ptr(buf, 24)
    assert_equal(val.type.value, ValueType.STRING)
    assert_equal(val.string_len(), 24)
    buf.free()
    print("    PASS")


def test_int_type() raises:
    """GenericValue.from_int produces INT type with correct value."""
    print("  test_int_type...")
    var val = GenericValue.from_int(42)
    assert_equal(val.type.value, ValueType.INT)
    assert_equal(Int(val.as_int()), 42)
    print("    PASS")


def test_int_negative() raises:
    """INT type with negative value."""
    print("  test_int_negative...")
    var val = GenericValue.from_int(-100)
    assert_equal(val.type.value, ValueType.INT)
    assert_equal(Int(val.as_int()), -100)
    print("    PASS")


def test_none_type() raises:
    """Default GenericValue is NONE."""
    print("  test_none_type...")
    var val = GenericValue()
    assert_equal(val.type.value, ValueType.NONE)
    assert_true(val.is_none())
    print("    PASS")


def test_none_equality() raises:
    """Two NONE values are equal."""
    print("  test_none_equality...")
    var v1 = GenericValue()
    var v2 = GenericValue()
    assert_true(v1 == v2, "Two NONE values must be equal")
    print("    PASS")


def test_different_strings_not_equal() raises:
    """Two SSO values with different bytes should NOT be equal."""
    print("  test_different_strings_not_equal...")
    var v1 = GenericValue.from_string("abc")
    var v2 = GenericValue.from_string("xyz")
    assert_true(not (v1 == v2), "Different strings must not be equal")
    print("    PASS")


def test_different_lengths_not_equal() raises:
    """SSO values with different lengths should NOT be equal."""
    print("  test_different_lengths_not_equal...")
    var v1 = GenericValue.from_string("abc")
    var v2 = GenericValue.from_string("abcd")
    assert_true(not (v1 == v2), "Different length strings must not be equal")
    print("    PASS")


def test_from_string_sso() raises:
    """Verify from_string with short string produces SSO that matches from_ptr."""
    print("  test_from_string_sso...")
    var val1 = GenericValue.from_string("test")
    var buf = alloc[UInt8](4)
    fill_buf(buf, "test", 4)
    var val2 = GenericValue.from_ptr(buf, 4)
    assert_equal(val1.type.value, ValueType.STRING_SSO)
    assert_true(val1 == val2, "from_string and from_ptr should produce equal values")
    assert_equal(val1.__hash__(), val2.__hash__())
    buf.free()
    print("    PASS")


def test_int_vs_string_not_equal() raises:
    """INT and STRING types should never be equal."""
    print("  test_int_vs_string_not_equal...")
    var v_int = GenericValue.from_int(42)
    var v_str = GenericValue.from_string("42")
    assert_true(not (v_int == v_str), "INT and STRING must not be equal")
    print("    PASS")


def test_empty_string() raises:
    """Empty string (0 bytes) should be SSO with length 0."""
    print("  test_empty_string...")
    var val = GenericValue.from_string("")
    assert_equal(val.type.value, ValueType.STRING_SSO)
    assert_equal(val.string_len(), 0)
    print("    PASS")


def test_hash_and_pack_sso() raises:
    """Verify hash_and_pack_sso produces same hash as from_ptr for SSO keys."""
    print("  test_hash_and_pack_sso...")
    var buf = alloc[UInt8](10)
    fill_buf(buf, "lookup_key", 10)
    var val = GenericValue.from_ptr(buf, 10)
    var packed = GenericValue.hash_and_pack_sso(buf, 10)
    assert_equal(Int(packed[0]), val.__hash__())
    buf.free()
    print("    PASS")


def main() raises:
    print("=== GenericValue Tests ===")
    test_sso_from_ptr()
    test_heap_from_ptr()
    test_from_ptr_unsafe_always_string()
    test_sso_vs_unsafe_not_equal()
    test_two_sso_from_ptr_equal()
    test_two_heap_from_ptr_equal()
    test_hash_consistency_sso()
    test_hash_consistency_heap()
    test_sso_boundary_22_bytes()
    test_sso_boundary_23_bytes()
    test_sso_boundary_24_bytes()
    test_int_type()
    test_int_negative()
    test_none_type()
    test_none_equality()
    test_different_strings_not_equal()
    test_different_lengths_not_equal()
    test_from_string_sso()
    test_int_vs_string_not_equal()
    test_empty_string()
    test_hash_and_pack_sso()
    print("=== ALL GenericValue TESTS PASSED ===")
