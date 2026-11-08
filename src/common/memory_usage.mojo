"""MEMORY USAGE's estimate of what one key holds, which eviction also steers by."""
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.hll import HLL_REGISTERS


def memory_usage(v: GenericValue, key_len: Int, samples: Int) -> Int64:
    """MEMORY USAGE's estimate: the slot (metadata byte, 32-byte key and
    value), the key's bytes when they do not fit in it, and the value's own
    allocations: a string's bytes; a hash's, set's or sorted set's table plus
    its elements' bytes, sampled as Redis samples them (`samples` elements, 0
    all) and scaled to the whole; a list's elements."""
    var total = Int64(65)
    if key_len > 23:
        total += Int64(key_len)
    var t = v.type.value
    if t == ValueType.STRING or t == ValueType.BITMAP:
        total += Int64(v.string_len())
    elif t == ValueType.HLL:
        total += Int64(HLL_REGISTERS)
    elif t == ValueType.HASH:
        var m = v.as_hash().unsafe_bitcast[SlabHashMap]()
        total += Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    elif t == ValueType.SET:
        var m = v.as_set().unsafe_bitcast[SlabHashMap]()
        total += Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    elif t == ValueType.LIST:
        var l = v.as_list().unsafe_bitcast[SlabList]()
        total += Int64(l[].size) * 34 + 64
    elif t == ValueType.ZSET or t == ValueType.GEO:
        var z = v.as_zset().unsafe_bitcast[SlabSkipList]()
        var m = Pointer[SlabHashMap, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=z[].members)))
        total += Int64(z[].length) * 96 + Int64(m[].capacity) * 65 + _sampled_payload(m, samples)
    return total


def _sampled_payload(m: Pointer[SlabHashMap, MutUntrackedOrigin], samples: Int) -> Int64:
    """The heap bytes of a hash's or set's fields and values: every element
    when samples is 0, else the first `samples` found, scaled to the size."""
    var seen = 0
    var bytes = Int64(0)
    for slot in range(m[].capacity):
        var md = m[].metadata[unsafe_offset=slot]
        if md == 0x80 or md == 0xFF:
            continue
        if m[].keys[unsafe_offset=slot].type.value == ValueType.STRING:
            bytes += Int64(m[].keys[unsafe_offset=slot].string_len())
        if m[].values[unsafe_offset=slot].type.value == ValueType.STRING:
            bytes += Int64(m[].values[unsafe_offset=slot].string_len())
        seen += 1
        if samples > 0 and seen >= samples:
            break
    if seen == 0:
        return 0
    return bytes * Int64(m[].size) // Int64(seen)
