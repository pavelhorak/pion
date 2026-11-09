"""The keyspace's table (KeyShard, through StripedHashMap): 8-byte slots that
point to one allocation per key, with a string value stored in it.

Key and value lengths straddle every boundary the entry layout has: 7/8,
15/16 and 23/24 (the three words a short key compares as; 23 is the longest
STRING_SSO), 254/255 (the one-byte key length), and the 16-byte size steps an
entry is rewritten in place within. Both kinds of entry, every change between
them, a value that borrows its own entry, and rebuilds under churn.

    pixi run mojo run -I . tests/test_key_shard.mojo
"""
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from std.testing import assert_equal, assert_true, assert_false
from std.collections import List
from src.common.value import GenericValue, ValueType, BORROW_TAG
from src.common.hash_map import StripedHashMap, KeyShard, KS_G


def lens() -> List[Int]:
    return [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 22, 23, 24, 25, 31, 32, 100, 254, 255, 256, 300]


def fill(seed: Int, n: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Deterministic bytes on the heap, `n` of them (one more is allocated,
    so that n == 0 allocates too)."""
    var p = unsafe_alloc[UInt8](n + 1)
    var x = UInt64(seed) * 0x9E3779B97F4A7C15 + 1
    for i in range(n):
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        p[unsafe_offset=i] = UInt8(x & 0xFF)
    return p


def kfill(ki: Int, vi: Int, n: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """A key for the (ki, vi) pair: unique, its first bytes being vi and ki
    (there is one key length per ki, so one byte is enough at length 1)."""
    var p = fill(1000 + ki * 64 + vi, n)
    if n >= 1:
        p[unsafe_offset=0] = UInt8(vi)
    if n >= 2:
        p[unsafe_offset=1] = UInt8(ki)
    return p


def heap_copy(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var n = s.byte_length()
    var p = unsafe_alloc[UInt8](n + 1)
    var b = s.as_bytes()
    for i in range(n):
        p[unsafe_offset=i] = b[i]
    return p


def same_bytes(v: GenericValue, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    if not v.is_string() or v.string_len() != n:
        return False
    if n == 0:
        return True
    var tmp = unsafe_alloc[UInt8](24)
    var q = v.as_string_safe(tmp)
    var ok = external_call["memcmp", Int32](q, p, n) == 0
    tmp.unsafe_free()
    return ok


def slot_of(ks: StripedHashMap, key: GenericValue) -> Tuple[Int, Int]:
    var h = UInt64(key.__hash__())
    var s = Int(h & 7)
    return (s, ks.shards[unsafe_offset=s].find_slot(key, h))


def test_lengths() raises:
    """Every key length against every value length: set, get, key_at,
    value_at, overwrite, remove."""
    print("Testing key and value lengths...")
    var LENS = lens()
    var ks = StripedHashMap(16)
    var n = 0
    for ki in range(len(LENS)):
        var kl = LENS[ki]
        if kl == 0:
            continue
        for vi in range(len(LENS)):
            var vl = LENS[vi]
            var kb = kfill(ki, vi, kl)
            var vb = fill(5000 + ki * 64 + vi, vl)
            var key = GenericValue.borrow(kb, kl)
            ks.set(key, GenericValue.borrow(vb, vl))
            n += 1
            assert_true(same_bytes(ks.get(key), vb, vl), "get after set")
            var at = slot_of(ks, key)
            assert_true(at[1] >= 0)
            var sh = ks.shards.unsafe_offset(at[0])
            assert_equal(sh[].entry_at(at[1]) & KS_G, 0)        # an S entry
            assert_true(same_bytes(sh[].key_at(at[1]), kb, kl), "key_at")
            assert_true(same_bytes(sh[].value_at(at[1]), vb, vl), "value_at")
            var kv = sh[].key_at(at[1])
            if kl <= 23:
                assert_equal(kv.type.value, ValueType.STRING_SSO)
            else:
                assert_equal(kv.type.value, ValueType.STRING)
                assert_equal(kv._data2, BORROW_TAG)
            # The key as stored hashes and compares as the lookup key does.
            assert_true(kv == key)
            assert_equal(kv.__hash__(), key.__hash__())
            kb.unsafe_free()
            vb.unsafe_free()
    var total = 0
    for s in range(8):
        total += ks.shards[unsafe_offset=s].size
    assert_equal(total, n)
    # Overwrite every key with every other length, then remove them all.
    for ki in range(len(LENS)):
        var kl = LENS[ki]
        if kl == 0:
            continue
        for vi in range(len(LENS)):
            var kb = kfill(ki, vi, kl)
            var key = GenericValue.borrow(kb, kl)
            var vl2 = LENS[(vi + 7) % len(LENS)]
            var vb2 = fill(9000 + ki * 64 + vi, vl2)
            ks.set(key, GenericValue.borrow(vb2, vl2))
            assert_true(same_bytes(ks.get(key), vb2, vl2), "overwrite")
            assert_true(ks.remove_generic(key))
            assert_false(ks.remove_generic(key))
            assert_equal(ks.get(key).type.value, ValueType.NONE)
            kb.unsafe_free()
            vb2.unsafe_free()
    total = 0
    for s in range(8):
        total += ks.shards[unsafe_offset=s].size
    assert_equal(total, 0)


def test_in_place() raises:
    """An overwrite whose entry size does not change keeps the entry; one
    that crosses a 16-byte step gets a new one, and the old bytes are gone."""
    print("Testing in-place overwrites...")
    var ks = StripedHashMap(16)
    var kb = fill(1, 16)
    var key = GenericValue.borrow(kb, 16)
    var vb = fill(2, 200)
    ks.set_str_reuse(key, vb, 100)
    var at = slot_of(ks, key)
    var e0 = ks.shards[unsafe_offset=at[0]].entry_at(at[1])
    # The fast path's SET, same length: same entry.
    ks.set_str_reuse(key, vb.unsafe_offset(50), 100)
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]), e0)
    assert_true(same_bytes(ks.get(key), vb.unsafe_offset(50), 100))
    # 1 + 16 + 4 + 100 = 121 -> 128; 107 bytes is 128 too, 108 is 144.
    ks.set_str_reuse(key, vb, 107)
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]), e0)
    assert_true(same_bytes(ks.get(key), vb, 107))
    ks.set_str_reuse(key, vb, 108)
    assert_true(same_bytes(ks.get(key), vb, 108))
    # set() with a borrowed value, same size: in place too.
    var e1 = ks.shards[unsafe_offset=at[0]].entry_at(at[1])
    ks.set(key, GenericValue.borrow(vb.unsafe_offset(3), 108))
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]), e1)
    assert_true(same_bytes(ks.get(key), vb.unsafe_offset(3), 108))
    kb.unsafe_free()
    vb.unsafe_free()


def test_self_alias() raises:
    """A value that borrows the entry it is stored into: the bytes are read
    before the entry is rewritten or freed."""
    print("Testing a value that borrows its own entry...")
    var ks = StripedHashMap(16)
    var kb = fill(3, 30)
    var key = GenericValue.borrow(kb, 30)
    var vb = fill(4, 300)
    ks.set(key, GenericValue.borrow(vb, 300))
    # The same bytes back: in place (overlap is exact).
    var v = ks.get(key)
    ks.set(key, v)
    assert_true(same_bytes(ks.get(key), vb, 300))
    # Its own tail, shorter by 5: still in place, an overlapping move.
    v = ks.get(key)
    ks.set(key, GenericValue.borrow(v.as_string().unsafe_offset(5), 295))
    assert_true(same_bytes(ks.get(key), vb.unsafe_offset(5), 295))
    # Its own head, 100 bytes shorter: a new entry, copied from the old one.
    v = ks.get(key)
    ks.set(key, GenericValue.borrow(v.as_string(), 150))
    assert_true(same_bytes(ks.get(key), vb.unsafe_offset(5), 150))
    # The stored key used to store under itself (a caller walking the slots).
    var at = slot_of(ks, key)
    var own = ks.shards[unsafe_offset=at[0]].key_at(at[1])
    ks.set(own, GenericValue.borrow(vb, 40))
    assert_true(same_bytes(ks.get(key), vb, 40))
    # And removed through its own stored key.
    at = slot_of(ks, key)
    own = ks.shards[unsafe_offset=at[0]].key_at(at[1])
    assert_true(ks.remove_generic(own))
    assert_equal(ks.get(key).type.value, ValueType.NONE)
    kb.unsafe_free()
    vb.unsafe_free()


def test_kinds() raises:
    """S to G and back, G to G; an owned heap STRING is kept as given; INCR's
    pointer turns an S entry into a G entry holding the same string."""
    print("Testing entry kinds...")
    var ks = StripedHashMap(16)
    var kb = fill(5, 12)
    var key = GenericValue.borrow(kb, 12)
    var vb = fill(6, 64)
    ks.set(key, GenericValue.borrow(vb, 64))                 # S
    ks.set(key, GenericValue.from_int(41))                   # S -> G
    var at = slot_of(ks, key)
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]) & KS_G, KS_G)
    assert_equal(Int(ks.get(key).as_int()), 41)
    var p = ks.get_value_ptr(key)
    p[]._data0 = UInt64(42)                                   # INCR in place
    assert_equal(Int(ks.get(key).as_int()), 42)
    ks.set(key, GenericValue.from_int(43))                   # G -> G
    assert_equal(Int(ks.get(key).as_int()), 43)
    ks.set(key, GenericValue.borrow(vb, 10))                 # G -> S
    at = slot_of(ks, key)
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]) & KS_G, 0)
    assert_true(same_bytes(ks.get(key), vb, 10))
    # get_value_ptr on an S entry: a G entry with the same string.
    var q = ks.get_value_ptr(key)
    assert_equal(q[].type.value, ValueType.STRING_SSO)
    assert_true(same_bytes(q[], vb, 10))
    at = slot_of(ks, key)
    assert_equal(ks.shards[unsafe_offset=at[0]].entry_at(at[1]) & KS_G, KS_G)
    assert_true(same_bytes(ks.get(key), vb, 10))
    ks.set(key, GenericValue.borrow(vb, 64))                 # long S
    var r = ks.get_value_ptr(key)
    assert_equal(r[].type.value, ValueType.STRING)
    assert_true(r[]._data2 != BORROW_TAG)                     # owned: copied out
    assert_true(same_bytes(ks.get(key), vb, 64))
    # An owned heap STRING is stored as it is: get returns that pointer.
    var owned = GenericValue.from_ptr(vb, 64)
    ks.set(key, owned)
    assert_equal(Int(ks.get(key)._data0), Int(owned._data0))
    # A fake aggregate, overwritten: parked in the graveyard, not freed.
    var agg = GenericValue()
    agg.type = ValueType(ValueType.LIST)
    agg._data0 = UInt64(0xDEAD0)
    ks.set(key, agg)
    ks.set(key, GenericValue.borrow(vb, 5))
    assert_equal(len(ks.graveyard[]), 1)
    assert_equal(Int(ks.graveyard[][0]._data0), 0xDEAD0)
    ks.graveyard[].clear()
    # Removed with _taking: handed back, not freed.
    ks.set(key, agg)
    var taken = GenericValue()
    assert_true(ks.remove_generic_taking(key, taken))
    assert_equal(Int(taken._data0), 0xDEAD0)
    assert_equal(ks.get(key).type.value, ValueType.NONE)
    # set_value_at stores without dropping the old value.
    ks.set(key, GenericValue.borrow(vb, 64))
    at = slot_of(ks, key)
    ks.shards[unsafe_offset=at[0]].set_value_at(at[1], GenericValue.from_int(7))
    assert_equal(Int(ks.get(key).as_int()), 7)
    ks.shards[unsafe_offset=at[0]].set_value_at(at[1], GenericValue.borrow(vb, 33))
    assert_true(same_bytes(ks.get(key), vb, 33))
    kb.unsafe_free()
    vb.unsafe_free()


def test_many() raises:
    """300,000 keys: the shards rebuild as they grow, every key is found and
    walked once, and churn through tombstones loses nothing."""
    print("Testing 300,000 keys with churn...")
    for round in range(2):
        var ks = StripedHashMap(65536)
        var n = 300000
        for i in range(n):
            ks.set("key:" + String(i), GenericValue.from_string("v" + String(i * 7)))
        var total = 0
        var walked = 0
        for s in range(8):
            var sh = ks.shards.unsafe_offset(s)
            total += sh[].size
            var slot = sh[].next_live(0, sh[].capacity)
            while slot < sh[].capacity:
                var k = sh[].key_at(slot)
                assert_true(k.is_string())
                assert_true(not ks.get(k).is_none())
                walked += 1
                slot = sh[].next_live(slot + 1, sh[].capacity)
            assert_true(sh[].size * 8 <= sh[].capacity * 7)
        assert_equal(total, n)
        assert_equal(walked, n)
        for i in range(n):
            var v = ks.get("key:" + String(i))
            assert_true(v.is_string())
            assert_equal(String(v), "v" + String(i * 7))
        for i in range(0, n, 2):
            assert_true(ks.remove("key:" + String(i)))
        for i in range(n, n + n // 2):
            ks.set("key:" + String(i), GenericValue.from_string("w" + String(i)))
        for i in range(n + n // 2):
            var v = ks.get("key:" + String(i))
            if i < n and i % 2 == 0:
                assert_true(v.is_none())
            elif i < n:
                assert_equal(String(v), "v" + String(i * 7))
            else:
                assert_equal(String(v), "w" + String(i))
        if round == 1:
            ks.reset()
            for s in range(8):
                assert_equal(ks.shards[unsafe_offset=s].size, 0)
            assert_true(ks.get("key:1").is_none())
            ks.set("key:1", GenericValue.from_string("back"))
            assert_equal(String(ks.get("key:1")), "back")


def test_access_table() raises:
    """The access stamps move with their keys when a shard rebuilds."""
    print("Testing the access table across a rebuild...")
    var ks = StripedHashMap(128)
    for s in range(8):
        ks.shards[unsafe_offset=s].enable_access_table()
    ks.set("stamped", GenericValue.from_string("x"))
    var sb = heap_copy("stamped")
    var key = GenericValue.borrow(sb, 7)
    var at = slot_of(ks, key)
    ks.shards[unsafe_offset=at[0]].access_table()[unsafe_offset=at[1]] = UInt32(0x5A123456)
    for i in range(20000):
        ks.set("fill:" + String(i), GenericValue.from_int(Int64(i)))
    at = slot_of(ks, key)
    assert_true(ks.shards[unsafe_offset=at[0]].rehashes > 0)
    assert_equal(Int(ks.shards[unsafe_offset=at[0]].access_table()[unsafe_offset=at[1]]), 0x5A123456)
    sb.unsafe_free()


def main() raises:
    test_lengths()
    test_in_place()
    test_self_alias()
    test_kinds()
    test_many()
    test_access_table()
    print("All key shard tests passed.")
