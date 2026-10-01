"""gh #369: free the container behind a keyspace value when its key goes away.

Aggregates (LIST / HASH / SET / ZSET / GEO) live behind a pointer in
`GenericValue`, and before this nothing ever freed one: a key removed by DEL,
or dropped because its last element was popped (gh #234), left its container
allocated. A ZADD/ZPOPMIN loop on ONE key therefore allocated a fresh sorted
set every cycle — a 16 KB node slab (one page) plus a member dict — and 60K
cycles grew RSS by ~1 GB that never came back.

Every container is allocated individually (`alloc[T](1)`, including the pooled
ones — ObjectPool pre-allocates each object on its own), so it is freed with
`unsafe_deinit_pointee()` + `free()`. Ownership is the caller's problem: call this
only on a value that no other key aliases. COPY deep-copies containers for
exactly that reason (key_mgmt.mojo); RENAME moves the pointer and removes the
source WITHOUT freeing it.
"""
from src.common.ptr import is_null, is_not_null
from std.memory import alloc, stack_allocation
from std.ffi import external_call
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap, StripedHashMap
from std.memory.unsafe_pointer import Pointer
from src.common.list import SlabList
from src.common.skip_list import SlabSkipList
from src.common.vector_set import VectorSet, free_vset
from src.common.stream_data import StreamData
from src.common.hll import HLL_REGISTERS
from std.memory import unsafe_memcpy


def free_container(val: GenericValue):
    """Release the aggregate `val` points to. Strings and every other type are
    left alone (heap string payloads are freed by the map on remove)."""
    var t = val.type.value
    if t == ValueType.ZSET or t == ValueType.GEO:
        var z = val.as_zset().bitcast[SlabSkipList]()
        if is_null(z): return
        z[].release()
        z.unsafe_deinit_pointee()
        z.free()
    elif t == ValueType.HASH or t == ValueType.SET:
        var h = val.as_hash().bitcast[SlabHashMap]()
        if is_null(h): return
        h.unsafe_deinit_pointee()   # SlabHashMap.__del__ frees slots + payloads
        h.free()
    elif t == ValueType.VSET:
        free_vset(val.as_hash().bitcast[VectorSet]())
    elif t == ValueType.LIST:
        var l = val.as_list().bitcast[SlabList]()
        if is_null(l): return
        l[].release()
        l.free()              # release() freed every buffer; no destructor to run
    # gh #394: the three types below were never freed at all — DEL of an HLL
    # leaked its 16 KB register array, of a stream ~3 KB plus every entry,
    # of a bitmap its bytes.
    elif t == ValueType.STREAM:
        var sd = val.as_hash().bitcast[StreamData]()
        if is_null(sd): return
        sd[].release()
        sd.free()
    elif t == ValueType.HLL or t == ValueType.BITMAP:
        if val._data0 != 0:
            val.as_bitmap().unsafe_free()


def free_graveyard(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]):
    """Free every aggregate the keyspace OVERWROTE since the last call
    (SlabHashMap._retire parks them — gh #394). The engine calls this once per
    dispatch batch, after its replies are written: a reply may still borrow
    from a value the batch replaced, so the free is deferred, never early."""
    var g = keyspace[].graveyard
    for j in range(len(g[])):
        free_container(g[][j])
    g[].clear()


@always_inline
def owns_heap(val: GenericValue) -> Bool:
    """True when dropping `val` must go through free_container: every
    aggregate, plus HLL and BITMAP (a bare heap block each). Heap STRING
    payloads are the map's job (free_str_payload) and are excluded."""
    var t = val.type.value
    return val.is_container() or t == ValueType.HLL or t == ValueType.BITMAP


def remove_and_free(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key: GenericValue) -> Bool:
    """Drop `key` AND free the container it held — for the gh #234 sites that
    remove an aggregate because it just became empty, and for DEL/UNLINK.
    Must run after any reply that borrows from the container."""
    var val = keyspace[].get(key)
    if val.is_none():
        return False
    free_container(val)
    return keyspace[].remove_generic(key)


def deep_clone(val: GenericValue) -> GenericValue:
    """A copy of `val` that shares nothing with it — for COPY.

    `GenericValue.clone()` deep-copies only heap STRING payloads; for an
    aggregate it copies the 32-byte handle, so `COPY src dst` left both keys
    pointing at ONE container: a write to either showed up in both, and once
    containers are freed on DEL (gh #369) a DEL of either would free the
    other's data. Aggregates are rebuilt element by element here."""
    var t = val.type.value
    if t == ValueType.HASH or t == ValueType.SET:
        var src = val.as_hash().bitcast[SlabHashMap]()
        var dst = alloc[SlabHashMap](1)
        dst.unsafe_write(SlabHashMap(16))
        for i in range(src[].capacity):
            if (src[].metadata[unsafe_offset=i] & 0x80) == 0:   # occupied
                dst[].set(src[].keys[unsafe_offset=i].clone(), src[].values[unsafe_offset=i].clone())
        # gh #392: the copy keeps the field TTLs (Redis COPY does), each field
        # key cloned — sharing the source map's key would alias its payload.
        if Int(src[].field_ttl) != 0:
            var ft = src[].field_ttl
            for i in range(ft[].capacity):
                if (ft[].metadata[unsafe_offset=i] & 0x80) == 0:
                    dst[].set_field_deadline(ft[].keys[unsafe_offset=i].clone(),
                                             ft[].values[unsafe_offset=i].as_int())
        var out = GenericValue()
        out.type = val.type
        out.set_ptr(dst.bitcast[NoneType]())
        return out
    if t == ValueType.ZSET or t == ValueType.GEO:
        var zsrc = val.as_zset().bitcast[SlabSkipList]()
        var zdst = alloc[SlabSkipList](1)
        zdst.unsafe_write(SlabSkipList(16))
        var node = zsrc[].head[].forward[0]
        while is_not_null(node):
            zdst[].insert(node[].score, node[].obj.clone())
            node = node[].forward[0]
        var zout = GenericValue()
        zout.type = val.type
        zout.set_ptr(zdst.bitcast[NoneType]())
        return zout
    if t == ValueType.LIST:
        var lsrc = val.as_list().bitcast[SlabList]()
        var ldst = alloc[SlabList](1)
        ldst.unsafe_write(SlabList())
        ldst[].replace_all(lsrc[].owned_elems())
        var lout = GenericValue()
        lout.type = val.type
        lout.set_ptr(ldst.bitcast[NoneType]())
        return lout
    if t == ValueType.VSET:
        var vsrc = val.as_hash().bitcast[VectorSet]()
        var vdst = alloc[VectorSet](1)
        vdst.unsafe_write(VectorSet(vsrc[].dim))
        var tmp = alloc[Float32](vsrc[].dim)
        for s in range(vsrc[].n):
            if vsrc[].alive[unsafe_offset=s] == 0:
                continue
            var nrm = vsrc[].norms[unsafe_offset=s]
            for d in range(vsrc[].dim):
                tmp[unsafe_offset=d] = vsrc[].vecs[unsafe_offset=s * vsrc[].dim + d] * nrm
            # Borrow the name in place: a `var` copy of a <= 23-byte String
            # keeps its bytes inline on THIS stack frame, and passing them to
            # the out-of-line `add` is the gh #349 / #384 `tail call` hazard.
            ref nm = vsrc[].names[s]
            var np = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(nm.unsafe_ptr()))
            _ = vdst[].add(np, nm.byte_length(), tmp)
            vdst[].attrs[vdst[].find(np, nm.byte_length())] = vsrc[].attrs[s]
        tmp.free()
        var vout = GenericValue()
        vout.type = val.type
        vout.set_ptr(vdst.bitcast[NoneType]())
        return vout
    # gh #394: these fell through to clone(), a SHALLOW copy of the handle —
    # harmless while nothing ever freed them, a use-after-free the moment DEL
    # does (COPY k k2; DEL k left k2 on freed registers / entries / bytes).
    if t == ValueType.STREAM:
        var ssrc = val.as_hash().bitcast[StreamData]()
        var sdst = alloc[StreamData](1)
        sdst.unsafe_write(ssrc[].deep_copy())
        var sout = GenericValue()
        sout.type = val.type
        sout.set_ptr(sdst.bitcast[NoneType]())
        return sout
    if t == ValueType.HLL:
        var hdst = alloc[UInt8](HLL_REGISTERS)
        unsafe_memcpy(dest=hdst, src=val.as_hll(), count=HLL_REGISTERS)
        var hout = val
        hout.set_ptr(hdst.bitcast[NoneType]())
        return hout
    if t == ValueType.BITMAP:
        var n = val.bitmap_len()
        var bdst = alloc[UInt8](max(n, 1))
        if n > 0:
            unsafe_memcpy(dest=bdst, src=val.as_bitmap(), count=n)
        var bout = val
        bout.set_ptr(bdst.bitcast[NoneType]())
        return bout
    return val.clone()


@always_inline
def _now_ns() -> Int64:
    """CLOCK_REALTIME in ns — fast_path._get_now_ns, which this module cannot
    import (fast_path imports it). Same clock, so deadlines compare."""
    var ts = stack_allocation[2, Int64]()
    _ = external_call["clock_gettime", Int32](Int32(0), ts)
    return ts[0] * Int64(1_000_000_000) + ts[1]


def hash_get_live(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key: GenericValue) -> GenericValue:
    """`keyspace.get(key)`, with the hash's expired fields deleted first — and
    the key itself when that empties it (Redis drops the key with its last
    field). Every hash read goes through this so an expired field is never
    visible; the sweep only reclaims memory sooner."""
    var v = keyspace[].get(key)
    if v.type.value == ValueType.HASH:
        var hp = v.as_hash().unsafe_bitcast[SlabHashMap]()
        if hp[].has_field_ttls():
            _ = hp[].purge_expired_fields(_now_ns())
            if hp[].size == 0:
                _ = remove_and_free(keyspace, key)
                return GenericValue()
    return v


def index_field_ttls(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key: GenericValue,
                     val: GenericValue):
    """gh #392: after a hash lands under `key` (RENAME, COPY), put it on the
    active-expiry index if it carries field TTLs."""
    if val.type.value == ValueType.HASH:
        var hp = val.as_hash().unsafe_bitcast[SlabHashMap]()
        if hp[].has_field_ttls():
            keyspace[].note_field_ttl(key)
