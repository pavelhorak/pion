"""#43, #46: a hash's vector field reaches the vector index from every command
that sets hash fields, and leaves it with the hash.

#43. The fast path's HSET arms used to route a field to the index inline, and
every other way to set a hash field (HSET inside MULTI/EXEC, from a script,
pipelined behind a slow-path command; HMSET and HSETNX always) stored the
field and indexed nothing, so FT.SEARCH missed documents HGET showed were
there. Every route now calls ingest_hash_vector after the field is stored:

  1. the index is defined (`pre_index_ready`), the value is `dim` float32s and
     the field is the index's vector field (its name folded, as before);
  2. the vector goes to the shared ingest buffer, which returns its slot;
  3. `__hk__<slot>` → key is set in the keyspace and logged to the WAL (the
     only durable slot → key map for keys over 31 bytes, gh #211), and keys of
     up to 31 bytes also go to the shared map every worker reads;
  4. #46: the hash records the slot (SlabHashMap.index_vector), so the slot
     dies with the hash or with its vector field (src/common/vec_tomb.mojo).

#46. Dead slots are recorded as members `<build id>:<slot>` of the worker's
`__hk_dead__` set (log_dead_slots, after each batch), which the WAL,
snapshots and replication carry like any set; restore_index_state applies
the ones of the index that loaded and relinks the hashes to their slots
after a restart (replay rebuilds hashes without the link). A slot that dies
before FT.OPTIMIZE has no build yet: it is recorded when the build is
(record_all_dead, from FT.OPTIMIZE).
"""

from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import UnsafePointer
from std.collections import List
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import format_int_to_buf
from src.common.vec_tomb import VecTomb
from src.vector.hnsw import SharedHNSWView
from src.io.wal import WAL
from src.network.dispatcher import CommandDispatcher

comptime HK_DEAD_KEY = "__hk_dead__"


def _field_hash(field: UnsafePointer[UInt8, MutUntrackedOrigin], flen: Int) -> UInt64:
    """The hash SlabHashMap gives the field `field` (what index_vector keeps)."""
    return UInt64(GenericValue.borrow(field, flen).__hash__())


def _is_vector_field(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                     field: UnsafePointer[UInt8, MutUntrackedOrigin], flen: Int) -> Bool:
    """The index's vector field, its name folded as the fast path folds it."""
    if flen != shared[].pre_vector_field_len:
        return False
    var name = shared[].pre_vector_field_name.unsafe_ptr()
    for b in range(flen):
        if (field[b] | 0x20) != name[b]:
            return False
    return True


def ingest_hash_vector(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                       keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                       wal: UnsafePointer[WAL, MutUntrackedOrigin],
                       tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                       key: UnsafePointer[UInt8, MutUntrackedOrigin], klen: Int,
                       field: UnsafePointer[UInt8, MutUntrackedOrigin], flen: Int,
                       val: UnsafePointer[UInt8, MutUntrackedOrigin], vlen: Int) -> Bool:
    """Called after `field` = `val` was stored in the hash at `key`: add the
    field's vector to the index when it is the index's vector field, and link
    the hash to its slot. True when it was added. (Storing the field already
    killed the slot of the vector it replaced: SlabHashMap.set.)"""
    if is_null(shared) or not shared[].pre_index_ready:
        return False
    if vlen != shared[].pre_dim * 4 or not _is_vector_field(shared, field, flen):
        return False
    var hv = keyspace[].get(GenericValue.borrow(key, klen))
    if hv.type.value != ValueType.HASH:
        return False
    var slot = shared[].add_ingest_vector(0, val.bitcast[Float32]())
    if slot < 0:
        return False
    # heap, not stack_allocation: it goes to the WAL appender, an out-of-line
    # call (a stack buffer may cross only into C or @always_inline, gh #349)
    var hk = alloc[UInt8](32)
    hk[0] = 95
    hk[1] = 95
    hk[2] = 104
    hk[3] = 107
    hk[4] = 95
    hk[5] = 95                              # "__hk__"
    var hk_end = format_int_to_buf(hk, 6, Int64(slot))
    keyspace[].set(GenericValue.borrow(hk, hk_end), GenericValue.borrow_buf(key, klen))
    if is_not_null(wal):
        _ = wal[].append_kv(1, hk, hk_end, key, klen)
    if is_not_null(shared[].hk_keys_buf) and slot < shared[].hk_max_elements and klen <= 31:
        var dst = shared[].hk_keys_buf + slot * 32
        dst[0] = UInt8(klen)
        unsafe_memcpy(dest=dst + 1, src=key, count=klen)
    hk.unsafe_free()
    if is_not_null(tomb):
        hv.as_hash().bitcast[SlabHashMap]()[].index_vector(slot, _field_hash(field, flen), tomb)
    return True


def ingest_whole_hash(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                      keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                      wal: UnsafePointer[WAL, MutUntrackedOrigin],
                      tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                      key: UnsafePointer[UInt8, MutUntrackedOrigin], klen: Int) -> Bool:
    """#46: the hash now at `key` arrived whole (COPY, RENAME, RESTORE): ingest
    its vector field as HSET would have."""
    if is_null(shared) or not shared[].pre_index_ready:
        return False
    var hv = keyspace[].get(GenericValue.borrow(key, klen))
    if hv.type.value != ValueType.HASH:
        return False
    var h = hv.as_hash().bitcast[SlabHashMap]()
    var fbuf = alloc[UInt8](24)
    var vbuf = alloc[UInt8](24)
    var done = False
    for i in range(h[].capacity):
        if (h[].metadata[i] & 0x80) != 0:
            continue
        var f = h[].keys[i]
        var fl = f.string_len()
        var fp = f.as_string_safe(fbuf)
        if not _is_vector_field(shared, fp, fl):
            continue
        var v = h[].values[i]
        if not v.is_string():
            break
        done = ingest_hash_vector(shared, keyspace, wal, tomb, key, klen, fp, fl,
                                  v.as_string_safe(vbuf), v.string_len())
        break
    fbuf.unsafe_free()
    vbuf.unsafe_free()
    return done


def hash_addr(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
              key: UnsafePointer[UInt8, MutUntrackedOrigin], klen: Int) -> Int:
    """The address of the hash at `key` (0 when it is not a hash): to see,
    after RENAME, whether this very hash moved."""
    var hv = keyspace[].get(GenericValue.borrow(key, klen))
    if hv.type.value != ValueType.HASH:
        return 0
    return Int(hv.as_hash())


def after_rename(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                 keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                 wal: UnsafePointer[WAL, MutUntrackedOrigin],
                 tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                 moved: Int, src: UnsafePointer[UInt8, MutUntrackedOrigin], slen: Int,
                 dst: UnsafePointer[UInt8, MutUntrackedOrigin], dlen: Int):
    """#46: the hash at address `moved` was renamed, if `dst` now holds it.
    Its slot names the old key, so the slot dies; before a build, its vector
    is ingested again under the new name (as COPY's is). `RENAME k k` moves
    nothing."""
    if moved == 0 or hash_addr(keyspace, dst, dlen) != moved:
        return
    if slen == dlen:
        var same = True
        for b in range(slen):
            if src[b] != dst[b]:
                same = False
                break
        if same:
            return
    var hv = keyspace[].get(GenericValue.borrow(dst, dlen))
    hv.as_hash().bitcast[SlabHashMap]()[].drop_vector()
    _ = ingest_whole_hash(shared, keyspace, wal, tomb, dst, dlen)


def _dead_member(build_id: UInt64, slot: Int) -> String:
    return String(build_id) + ":" + String(slot)


def log_dead_slots(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                   tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                   mut dispatcher: CommandDispatcher):
    """#46, after each batch: record the slots this worker killed, so a restart
    does not revive them. Before the first build there is nothing to record
    against (FT.OPTIMIZE records every dead slot under its new build)."""
    if is_null(tomb) or len(tomb[].pending) == 0:
        return
    if is_null(shared) or is_null(shared[].build_id) or is_null(shared[].ready_atomic) \
            or shared[].ready_atomic[] == 0:
        tomb[].pending.clear()
        return
    var bid = shared[].build_id[]
    for k in range(len(tomb[].pending)):
        _ = dispatcher.execute_sadd(String(HK_DEAD_KEY), _dead_member(bid, tomb[].pending[k]))
    tomb[].pending.clear()


def record_all_dead(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                    tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                    mut dispatcher: CommandDispatcher):
    """#46, after FT.OPTIMIZE gave the index a new build id: the slots already
    dead (including the ones that died before the build) under that id."""
    _ = dispatcher.execute_del(String(HK_DEAD_KEY))
    if is_not_null(tomb):
        tomb[].pending.clear()
    if is_null(shared) or is_null(shared[].vec_dead) or is_null(shared[].build_id) or not shared[].any_dead():
        return
    var bid = shared[].build_id[]
    for s in range(shared[].hk_max_elements):
        if shared[].vec_dead[s] != 0:
            _ = dispatcher.execute_sadd(String(HK_DEAD_KEY), _dead_member(bid, s))


def restore_index_state(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                        keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                        tomb: UnsafePointer[VecTomb, MutUntrackedOrigin]):
    """#46, at startup once the index has loaded: mark the slots this worker
    recorded dead for this build, then link each live slot's hash (this
    worker's `__hk__<slot>` entries) back to its slot, which replay cannot."""
    if is_null(shared) or is_null(shared[].vec_dead) or is_null(shared[].ready_atomic) \
            or shared[].ready_atomic[] == 0:
        return
    var bid = shared[].build_id[] if is_not_null(shared[].build_id) else UInt64(0)
    var n = shared[].hk_max_elements
    # 1. this worker's dead slots for this build
    var dv = keyspace[].get(GenericValue.borrow(HK_DEAD_KEY.unsafe_ptr(), HK_DEAD_KEY.byte_length()))
    if dv.type.value == ValueType.SET:
        var ds = dv.as_set().bitcast[SlabHashMap]()
        var mbuf = alloc[UInt8](24)
        for i in range(ds[].capacity):
            if (ds[].metadata[i] & 0x80) != 0:
                continue
            var m = ds[].keys[i]
            var ml = m.string_len()
            var mp = m.as_string_safe(mbuf)
            var b = UInt64(0)
            var j = 0
            while j < ml and mp[j] >= 48 and mp[j] <= 57:
                b = b * 10 + UInt64(Int(mp[j]) - 48)
                j += 1
            if j == 0 or j >= ml or mp[j] != 58 or b != bid:   # ':'
                continue
            j += 1
            var s = 0
            var k = j
            while k < ml and mp[k] >= 48 and mp[k] <= 57:
                s = s * 10 + Int(mp[k]) - 48
                k += 1
            if k == j or k != ml or s >= n:
                continue
            shared[].mark_dead(s)
        mbuf.unsafe_free()
    # 2. relink: every live slot whose __hk__ entry this worker holds
    if is_null(tomb) or shared[].pre_vector_field_len <= 0:
        return
    var field_h = _field_hash(shared[].pre_vector_field_name.unsafe_ptr(), shared[].pre_vector_field_len)
    var hk = alloc[UInt8](32)
    hk[0] = 95
    hk[1] = 95
    hk[2] = 104
    hk[3] = 107
    hk[4] = 95
    hk[5] = 95
    var kbuf = alloc[UInt8](24)
    var nodes = shared[].num_nodes
    for s in range(min(nodes, n)):
        if shared[].vec_dead[s] != 0:
            continue
        var hk_end = format_int_to_buf(hk, 6, Int64(s))
        var kv = keyspace[].get(GenericValue.borrow(hk, hk_end))
        if kv.is_none() or not kv.is_string():
            continue
        var hv = keyspace[].get(GenericValue.borrow(kv.as_string_safe(kbuf), kv.string_len()))
        if hv.type.value == ValueType.HASH:
            hv.as_hash().bitcast[SlabHashMap]()[].index_vector(s, field_h, tomb)
    hk.unsafe_free()
    kbuf.unsafe_free()
