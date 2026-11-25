"""MEMORY USAGE's estimate of what one key holds, which eviction also steers by."""
from std.sys import size_of
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap, ks_key_bytes
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList, SkipListNode
from src.common.hll import HLL_REGISTERS
from src.common.stream_data import StreamData, StreamEntry, StreamGroup, StreamNack, StreamConsumer
from src.common.vector_set import VectorSet


def memory_usage(v: GenericValue, key_len: Int, samples: Int) -> Int64:
    """MEMORY USAGE's estimate: the key's slot and entry (ks_key_bytes: a
    string stored in the entry is counted there), and what the value has
    allocated. Element payloads are sampled as Redis samples them (`samples`
    elements, 0 all) and scaled to the whole.

    Containers count their struct and every buffer they hold (2026-11-18).
    Before, a list counted 34 bytes an element and none of its buffers, a
    stream and a vector set their key only, a sorted set 96 bytes a node
    whatever its slabs held: 1/67 of a small list's RSS, 1/42 of a stream's.
    Eviction's size estimate is this function, so it was steering by those."""
    var total = Int64(ks_key_bytes(key_len, v))
    var t = v.type.value
    if (t == ValueType.STRING and not v.is_borrowed()) or t == ValueType.BITMAP:
        total += Int64(v.string_len())
    elif t == ValueType.HLL:
        total += Int64(HLL_REGISTERS)
    elif t == ValueType.HASH:
        var m = v.as_hash().unsafe_bitcast[SlabHashMap]()
        total += Int64(size_of[SlabHashMap]()) + _table_bytes(m) + _sampled_payload(m, samples)
    elif t == ValueType.SET:
        var m = v.as_set().unsafe_bitcast[SlabHashMap]()
        total += Int64(size_of[SlabHashMap]()) + _table_bytes(m) + _sampled_payload(m, samples)
    elif t == ValueType.LIST:
        var l = v.as_list().unsafe_bitcast[SlabList]()
        total += Int64(size_of[SlabList]()) + Int64(l[].alloc_bytes(samples))
    elif t == ValueType.ZSET or t == ValueType.GEO:
        var z = v.as_zset().unsafe_bitcast[SlabSkipList]()
        var m = Pointer[SlabHashMap, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=z[].members)))
        var slabs = 0
        for i in range(len(z[].allocator.slab_sizes)):
            slabs += z[].allocator.slab_sizes[i]
        # A member's heap payload is held twice: by its node and by the dict.
        total += Int64(size_of[SlabSkipList]()) + Int64(slabs) + _table_bytes(m) + 2 * _sampled_payload(m, samples)
    elif t == ValueType.STREAM:
        var sd = v.as_hash().unsafe_bitcast[StreamData]()
        total += Int64(size_of[StreamData]()) + Int64(sd[].capacity * size_of[StreamEntry]())
        total += _stream_data_bytes(sd, samples)
        for g in range(len(sd[].groups)):
            total += Int64(size_of[StreamGroup]() + len(sd[].groups[g].name))
            total += Int64(sd[].groups[g].pel.capacity() * size_of[StreamNack]())
            total += Int64(sd[].groups[g].consumers.capacity() * size_of[StreamConsumer]())
            total += Int64(sd[].groups[g].by_name.capacity() * 8)
            for c in range(len(sd[].groups[g].consumers)):
                total += Int64(len(sd[].groups[g].consumers[c].name))
    elif t == ValueType.VSET:
        var vs = v.as_hash().unsafe_bitcast[VectorSet]()
        var m = Pointer[SlabHashMap, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=vs[].index)))
        total += Int64(size_of[VectorSet]()) + Int64(vs[].cap * (vs[].dim * 4 + 4 + 1))
        total += Int64((vs[].names.capacity() + vs[].attrs.capacity()) * size_of[String]())
        total += _table_bytes(m) + _vset_string_bytes(vs, samples)
    return total


@always_inline
def _table_bytes(m: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int64:
    """A SlabHashMap's arrays: a key and a value (32 bytes each) and a
    metadata byte a slot, plus the 16 mirrored metadata bytes."""
    return Int64(m[].capacity) * 65 + 16


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


def _stream_data_bytes(sd: Pointer[StreamData, MutUntrackedOrigin], samples: Int) -> Int64:
    """The live entries' packed field data, sampled like a hash's payload."""
    var seen = 0
    var bytes = Int64(0)
    for i in range(sd[].count):
        if sd[].entries[unsafe_offset=i].deleted:
            continue
        bytes += Int64(sd[].entries[unsafe_offset=i].data_len)
        seen += 1
        if samples > 0 and seen >= samples:
            break
    if seen == 0:
        return 0
    return bytes * Int64(sd[].alive) // Int64(seen)


def _vset_string_bytes(vs: Pointer[VectorSet, MutUntrackedOrigin], samples: Int) -> Int64:
    """Element names and attributes past a String's inline bytes, and the
    names the index holds, sampled."""
    var n = len(vs[].names)
    if n == 0:
        return 0
    var want = n if samples <= 0 or samples > n else samples
    var bytes = Int64(0)
    for i in range(want):
        var nl = vs[].names[i].byte_length()
        if nl > 23:
            bytes += Int64(2 * nl)   # the name and the index's copy of it
        if i < len(vs[].attrs):
            var al = vs[].attrs[i].byte_length()
            if al > 23:
                bytes += Int64(al)
    return bytes * Int64(n) // Int64(want)
