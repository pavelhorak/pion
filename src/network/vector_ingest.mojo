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
  3. slot → key goes to this worker's slot map (StripedHashMap.vec_maps) and
     to the WAL as record 46 (what a restart reads it back from), and to the
     shared map every worker reads (SharedHNSWView.hk_keys_buf, hk_put). Until
     1.0.6 that one held keys of up to 31 bytes only, so under -w N a longer
     key was the slot number on every worker but the one that ingested it;
  4. #46: the hash records the slot (SlabHashMap.index_vector), so the slot
     dies with the hash or with its vector field (src/common/vec_tomb.mojo).

#46. Dead slots are recorded as `<build id>:<slot>` in the worker's dead-slot
set (StripedHashMap.vec_dead_map; log_dead_slots, after each batch), logged as
record 47 and carried by snapshots and replication; restore_index_state
applies the ones of the index that loaded and relinks the hashes to their
slots after a restart (replay rebuilds hashes without the link). A slot that
dies before FT.OPTIMIZE has no build yet: it is recorded when the build is
(record_all_dead, from FT.OPTIMIZE).

The slot map and the dead slots were the keyspace keys `__hk__<slot>` and
`__hk_dead__` until 1.0.5. DBSIZE, KEYS, SCAN and RANDOMKEY counted and
listed them, a client's key of that name was overwritten by an ingest, and
deleting them (SCAN MATCH + DEL, any cleanup script) left a document that
was deleted after the next restart in the results, because nothing linked
its hash to its slot any more. A data directory an older server wrote is
read into the maps (wal.wal_apply_legacy_vec).

The slot map describes one slot numbering. FT.DROPINDEX, or an FT.CREATE that
replaces the index, starts the numbering again on one worker; every worker
drops its own records of the old one before it next records anything
(sync_vector_records), and logs that it did (record 48). Before 1.0.6 only
the worker that ran the command did, and a restart relinked the others'
hashes to slots of the new index.
"""

from std.memory import alloc, unsafe_memcpy
from std.atomic import Atomic, Ordering
from std.memory.unsafe_pointer import UnsafePointer
from std.collections import List, Set
from src.common.ptr import is_not_null, is_null, null_ptr
from src.common.hash_map import StripedHashMap, SlabHashMap
from src.common.value import GenericValue, ValueType
from src.common.utils import format_int_to_buf
from src.common.vec_tomb import VecTomb
from src.vector.hnsw import SharedHNSWView
from src.vector.hnsw_types import hk_put, hk_claim
from src.io.wal import WAL, VEC_KEY_REC, VEC_DEAD_REC, VEC_CLEAR_REC, VEC_CLEAR_DEAD, VEC_CLEAR_KEYS, legacy_hk_slot, is_legacy_hk_dead
from src.common.container_free import remove_and_free
from src.network.dispatcher import CommandDispatcher


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
    # FT.CREATE ... PREFIX: a key outside the list is not this index's
    # document, so it is neither indexed nor counted as unindexed. Every route
    # in comes through here (HSET, COPY/RENAME/RESTORE, FT.CREATE's backfill,
    # a restart's refill).
    if not shared[].key_in_prefixes(key, klen):
        return False
    var hv = keyspace[].get(GenericValue.borrow(key, klen))
    if hv.type.value != ValueType.HASH:
        return False
    # #46 keeps an identical rewrite's slot alive (SlabHashMap._vector_field_set),
    # so that slot still holds this vector: a second slot would list the
    # document twice (an ingest script run twice, before or after a build).
    var hm = hv.as_hash().bitcast[SlabHashMap]()
    if hm[].vec_slot >= 0 and is_not_null(tomb) and Int(hm[].vec_tomb) == Int(tomb) \
            and hm[].vec_gen == tomb[].generation() and not shared[].slot_dead(hm[].vec_slot):
        return False
    var slot = shared[].add_ingest_vector(0, val.bitcast[Float32]())
    if slot < 0:
        # Drop-in: after an explicit FT.OPTIMIZE the ingest buffer is gone and
        # this vector is stored but not indexed. Count it (INFO
        # vector_unindexed_writes) rather than lose it without a trace.
        if is_null(shared[].ingest_fp32) and is_not_null(shared[].vec_stats):
            _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](shared[].vec_stats, UInt64(1))
        return False
    # The slot may be of a numbering another worker started: this worker's
    # records of the old one go first, or dropping them later takes this one.
    _ = sync_vector_records(keyspace, wal, tomb)
    keyspace[].set_vec_key(slot, key, klen)
    if is_not_null(wal):
        # heap, not stack_allocation: it goes to the WAL appender, an out-of-line
        # call (a stack buffer may cross only into C or @always_inline, gh #349)
        var sd = alloc[UInt8](24)
        var sd_end = format_int_to_buf(sd, 0, Int64(slot))
        _ = wal[].append_kv(VEC_KEY_REC, sd, sd_end, key, klen)
        sd.unsafe_free()
    # Every worker names the document by this entry (any key length)
    hk_put(shared[].hk_keys_buf, shared[].hk_max_elements, slot, key, klen)
    if is_not_null(tomb):
        hv.as_hash().bitcast[SlabHashMap]()[].index_vector(slot, _field_hash(field, flen), tomb)
    shared[].commit_ingest(slot)          # drop-in: the exact scan may read it now
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


def backfill_index(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                   keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                   wal: UnsafePointer[WAL, MutUntrackedOrigin],
                   tomb: UnsafePointer[VecTomb, MutUntrackedOrigin]) -> Int:
    """FT.CREATE indexes the hashes this worker's keyspace already holds, as
    RediSearch does (`SKIPINITIALSCAN` opts out). Without it, an index created
    after its documents were written stayed empty: FT.SEARCH answered `[0]`
    with no error, and the documents had to be written again. That is also the
    restart of an index whose DIM differs from the server's --dim: its file is
    refused, and re-issuing FT.CREATE was the only way back.

    A hash already linked to a live slot of this index is skipped by
    ingest_hash_vector, so a defensive re-create adds nothing. Returns the
    number of hashes indexed."""
    if is_null(shared) or is_null(keyspace) or not shared[].pre_index_ready \
            or is_null(shared[].ingest_fp32):
        return 0
    # Collect first: an ingest stores into the hash it indexes, and the walk
    # must not depend on what a store does to the shard it is reading.
    var keys = List[GenericValue]()
    for shard_i in range(8):
        var sp = keyspace[].shards.unsafe_offset(shard_i)
        for slot in range(sp[].capacity):
            var m = sp[].metadata[unsafe_offset=slot]
            if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                continue
            if sp[].values[unsafe_offset=slot].type.value != ValueType.HASH:
                continue
            if keyspace[].is_expired(sp[].keys[unsafe_offset=slot]):
                continue
            keys.append(sp[].keys[unsafe_offset=slot].clone())
    var kbuf = alloc[UInt8](24)
    var n = 0
    for k in range(len(keys)):
        if ingest_whole_hash(shared, keyspace, wal, tomb, keys[k].as_string_safe(kbuf), keys[k].string_len()):
            n += 1
        keys[k].free_str_payload()
    kbuf.unsafe_free()
    return n


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


def _dead_member(buf: UnsafePointer[UInt8, MutUntrackedOrigin], build_id: UInt64, slot: Int) -> Int:
    """`<build id>:<slot>` into `buf` (at least 41 bytes); its length."""
    var digits = 0
    var v = build_id
    while True:
        digits += 1
        v //= 10
        if v == 0:
            break
    v = build_id
    for d in range(digits):
        buf[digits - 1 - d] = UInt8(48 + Int(v % 10))
        v //= 10
    buf[digits] = 58                     # ':'
    return format_int_to_buf(buf, digits + 1, Int64(slot))


def _record_dead(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                 wal: UnsafePointer[WAL, MutUntrackedOrigin],
                 buf: UnsafePointer[UInt8, MutUntrackedOrigin], build_id: UInt64, slot: Int):
    var n = _dead_member(buf, build_id, slot)
    if keyspace[].add_vec_dead(buf, n) and is_not_null(wal):
        _ = wal[].append(VEC_DEAD_REC, buf, n)


def sync_vector_records(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                        wal: UnsafePointer[WAL, MutUntrackedOrigin],
                        tomb: UnsafePointer[VecTomb, MutUntrackedOrigin]) -> Bool:
    """The slot numbering moved on (FT.DROPINDEX, or an FT.CREATE that replaced
    the index, on any worker): this worker's slot map and dead slots describe
    the old one, so they go, logged as record 48, and so do the slots it
    killed but has not logged. Called before this worker records anything
    (an ingest, its dead slots) and every housekeeping tick. Before 1.0.6 only
    the worker that ran the command dropped its records: the others kept
    theirs, and after a restart relinked their hashes to slots the new index
    had given other documents. True when it dropped them."""
    if is_null(tomb) or Int(tomb[].gen) == 0:
        return False
    var g = Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](tomb[].gen)
    if g == tomb[].seen_gen:
        return False
    tomb[].seen_gen = g
    tomb[].pending.clear()
    if is_not_null(keyspace) and keyspace[].has_vec():
        clear_vector_records(keyspace, wal)
    return True


def log_dead_slots(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                   tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                   mut dispatcher: CommandDispatcher):
    """#46, after each batch: record the slots this worker killed, so a restart
    does not revive them. Before the first build there is nothing to record
    against (FT.OPTIMIZE records every dead slot under its new build)."""
    if is_null(tomb) or len(tomb[].pending) == 0:
        return
    _ = sync_vector_records(dispatcher.keyspace, dispatcher.wal, tomb)
    if len(tomb[].pending) == 0:
        return
    if is_null(shared) or is_null(shared[].build_id) or is_null(shared[].ready_atomic) \
            or shared[].ready_atomic[] == 0:
        tomb[].pending.clear()
        return
    var bid = shared[].build_id[]
    var buf = alloc[UInt8](48)            # heap: it goes to the WAL appender (gh #349)
    for k in range(len(tomb[].pending)):
        _record_dead(dispatcher.keyspace, dispatcher.wal, buf, bid, tomb[].pending[k])
    buf.unsafe_free()
    tomb[].pending.clear()


def record_all_dead(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                    tomb: UnsafePointer[VecTomb, MutUntrackedOrigin],
                    mut dispatcher: CommandDispatcher):
    """#46, after FT.OPTIMIZE gave the index a new build id: the slots already
    dead (including the ones that died before the build) under that id."""
    _ = sync_vector_records(dispatcher.keyspace, dispatcher.wal, tomb)
    var buf = alloc[UInt8](48)            # heap: it goes to the WAL appender (gh #349)
    dispatcher.keyspace[].clear_vec_dead()
    if is_not_null(dispatcher.wal):
        buf[0] = UInt8(VEC_CLEAR_DEAD)
        _ = dispatcher.wal[].append(VEC_CLEAR_REC, buf, 1)
    if is_not_null(tomb):
        tomb[].pending.clear()
    if is_not_null(shared) and is_not_null(shared[].vec_dead) and is_not_null(shared[].build_id) \
            and shared[].any_dead():
        var bid = shared[].build_id[]
        for s in range(shared[].hk_max_elements):
            if shared[].vec_dead[s] != 0:
                _record_dead(dispatcher.keyspace, dispatcher.wal, buf, bid, s)
    buf.unsafe_free()


def clear_vector_records(keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                         wal: UnsafePointer[WAL, MutUntrackedOrigin]):
    """FT.DROPINDEX: this worker's slot map and dead slots describe the index
    that went. A new index numbers its slots from 0 again."""
    keyspace[].clear_vec_keys()
    keyspace[].clear_vec_dead()
    if is_not_null(wal):
        var buf = alloc[UInt8](8)         # heap: it goes to the WAL appender (gh #349)
        buf[0] = UInt8(VEC_CLEAR_DEAD | VEC_CLEAR_KEYS)
        _ = wal[].append(VEC_CLEAR_REC, buf, 1)
        buf.unsafe_free()


def _dead_member_shape(p: UnsafePointer[UInt8, MutUntrackedOrigin], l: Int) -> Bool:
    """`<digits>:<digits>`, a dead-slot record (at most 20 + 1 + 19 bytes)."""
    if l > 40:
        return False
    var colon = -1
    for i in range(l):
        if p[i] == 58:
            if colon >= 0:
                return False
            colon = i
        elif p[i] < 48 or p[i] > 57:
            return False
    return colon > 0 and colon < l - 1


def adopt_legacy_slot_keys(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                           keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                           wal: UnsafePointer[WAL, MutUntrackedOrigin]) -> Int:
    """At startup, once an index has loaded: `__hk__<slot>` keys and a set
    `__hk_dead__` that replay left in this worker's keyspace are an older
    server's slot map. Replay reads an older server's records into the maps
    itself, but only those below the offset where this server's records start
    (wal.mojo); these came after it: the data went back to an older server,
    which indexed it again, and then forward. Their slots are that server's
    numbering, so the map is replaced by them, all logged, and the keys go.

    A client's own key of the same name is left alone: the set counts only
    when every member is a `<build id>:<slot>`, and a slot key only when its
    value names a hash or that set records its slot dead in the loaded build
    (a document deleted there). Returns how many keys were taken."""
    if is_null(shared) or is_null(shared[].ready_atomic) or shared[].ready_atomic[] == 0:
        return 0
    var slots = List[Int]()
    var names = List[GenericValue]()      # the keys, owned copies
    var maybe_slots = List[Int]()         # value names no hash: taken if dead
    var maybe_names = List[GenericValue]()
    var dead_key = GenericValue()
    var kbuf = alloc[UInt8](24)
    var vbuf = alloc[UInt8](24)
    for shard_i in range(8):
        var sp = keyspace[].shards.unsafe_offset(shard_i)
        for i in range(sp[].capacity):
            var m = sp[].metadata[unsafe_offset=i]
            if m == SlabHashMap.EMPTY or m == SlabHashMap.DELETED:
                continue
            var k = sp[].keys[unsafe_offset=i]
            if not k.is_string():
                continue
            var kp = k.as_string_safe(kbuf)
            var kl = k.string_len()
            var v = sp[].values[unsafe_offset=i]
            var slot = legacy_hk_slot(kp, kl)
            if slot >= 0:
                if v.is_string() and keyspace[].get(GenericValue.borrow(v.as_string_safe(vbuf), v.string_len())).type.value == ValueType.HASH:
                    slots.append(slot)
                    names.append(k.clone())
                elif v.is_string():
                    maybe_slots.append(slot)
                    maybe_names.append(k.clone())
            elif v.type.value == ValueType.SET and is_legacy_hk_dead(kp, kl):
                var ds = v.as_set().bitcast[SlabHashMap]()
                var all_dead = ds[].size > 0
                for j in range(ds[].capacity):
                    if (ds[].metadata[j] & 0x80) != 0:
                        continue
                    var mk = ds[].keys[j]
                    if not mk.is_string() or not _dead_member_shape(mk.as_string_safe(vbuf), mk.string_len()):
                        all_dead = False
                        break
                if all_dead:
                    dead_key = k.clone()
    if not dead_key.is_none() and len(maybe_names) > 0 and is_not_null(shared[].build_id):
        # `<build id>:<slot>`: a slot of the loaded build recorded dead
        var dead = Set[Int]()
        var bid = String(shared[].build_id[]) + ":"
        var ds = keyspace[].get(dead_key).as_set().bitcast[SlabHashMap]()
        for j in range(ds[].capacity):
            if (ds[].metadata[j] & 0x80) != 0:
                continue
            var mk = ds[].keys[j]
            var mp = mk.as_string_safe(vbuf)
            var ml = mk.string_len()
            var bl = bid.byte_length()
            if ml <= bl:
                continue
            var same = True
            var bp = bid.unsafe_ptr()
            for b in range(bl):
                if mp[b] != bp[b]:
                    same = False
                    break
            if not same:
                continue
            var sv = 0
            for b in range(bl, ml):
                sv = sv * 10 + Int(mp[b]) - 48
            dead.add(sv)
        for t in range(len(maybe_names)):
            if maybe_slots[t] in dead:
                slots.append(maybe_slots[t])
                names.append(maybe_names[t])
            else:
                maybe_names[t].free_str_payload()
    else:
        for t in range(len(maybe_names)):
            maybe_names[t].free_str_payload()
    var n = len(names) + (0 if dead_key.is_none() else 1)
    if n == 0:
        kbuf.unsafe_free()
        vbuf.unsafe_free()
        return 0
    print("Vector index: " + String(n) + " slot-map key(s) an older server wrote moved out of the keyspace")
    var buf = alloc[UInt8](48)            # heap: it goes to the WAL appender (gh #349)
    if len(names) > 0:
        keyspace[].clear_vec_keys()
        if is_not_null(wal):
            buf[0] = UInt8(VEC_CLEAR_KEYS)
            _ = wal[].append(VEC_CLEAR_REC, buf, 1)
    for t in range(len(names)):
        var kp = names[t].as_string_safe(kbuf)
        var kl = names[t].string_len()
        var v = keyspace[].get(GenericValue.borrow(kp, kl))
        var vl = v.string_len()
        var vp = v.as_string_safe(vbuf)
        # into the map before the key goes: `v` is the key's own value
        keyspace[].set_vec_key(slots[t], vp, vl)
        if is_not_null(wal):
            var sl = format_int_to_buf(buf, 0, Int64(slots[t]))
            _ = wal[].append_kv(VEC_KEY_REC, buf, sl, vp, vl)
        _ = remove_and_free(keyspace, GenericValue.borrow(kp, kl))
        if is_not_null(wal):
            _ = wal[].append(2, kp, kl)
        names[t].free_str_payload()
    if not dead_key.is_none():
        var kp = dead_key.as_string_safe(kbuf)
        var kl = dead_key.string_len()
        keyspace[].clear_vec_dead()
        if is_not_null(wal):
            buf[0] = UInt8(VEC_CLEAR_DEAD)
            _ = wal[].append(VEC_CLEAR_REC, buf, 1)
        var ds = keyspace[].get(GenericValue.borrow(kp, kl)).as_set().bitcast[SlabHashMap]()
        for j in range(ds[].capacity):
            if (ds[].metadata[j] & 0x80) != 0:
                continue
            var mk = ds[].keys[j]
            var ml = mk.string_len()
            unsafe_memcpy(dest=buf, src=mk.as_string_safe(vbuf), count=ml)
            if keyspace[].add_vec_dead(buf, ml) and is_not_null(wal):
                _ = wal[].append(VEC_DEAD_REC, buf, ml)
        _ = remove_and_free(keyspace, GenericValue.borrow(kp, kl))
        if is_not_null(wal):
            _ = wal[].append(2, kp, kl)
    buf.unsafe_free()
    kbuf.unsafe_free()
    vbuf.unsafe_free()
    return n


def restore_index_state(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                        keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                        tomb: UnsafePointer[VecTomb, MutUntrackedOrigin]):
    """#46, at startup once the index has loaded: mark the slots this worker
    recorded dead for this build, then link each live slot's hash (this
    worker's slot map entries) back to its slot, which replay cannot.

    Each entry also goes to the shared map when the index file left it empty
    (a file from before 1.0.6 holds no key over 31 bytes, so the other workers
    could not name the document), and a hash is linked only when the shared
    map names its key (hk_claim): a record of an older numbering would link
    a hash to a slot the new index gave another document.

    When the load reopened ingest (an automatic build's index,
    vector_auto.reopen_after_load), each relinked hash also puts its vector
    back in the slot's row, which the index file does not hold."""
    if is_null(shared) or is_null(shared[].vec_dead) or is_null(shared[].ready_atomic) \
            or shared[].ready_atomic[] == 0:
        return
    var bid = shared[].build_id[] if is_not_null(shared[].build_id) else UInt64(0)
    # Only the graph's slots: a slot past it was written after the build (the
    # delta), and a reopened ingest numbers the delta again from the graph's
    # end, so a tombstone of the old numbering would kill a new document.
    var n = min(shared[].num_nodes, shared[].hk_max_elements)
    # 1. this worker's dead slots for this build
    if keyspace[].has_vec():
        var ds = keyspace[].vec_dead_map()
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
    # 2. relink: every live slot this worker's slot map holds
    if is_null(tomb) or shared[].pre_vector_field_len <= 0:
        return
    var field_h = _field_hash(shared[].pre_vector_field_name.unsafe_ptr(), shared[].pre_vector_field_len)
    var kbuf = alloc[UInt8](24)
    var refill = is_not_null(shared[].restart_refill) and shared[].restart_refill[1] != 0 \
        and is_not_null(shared[].ingest_fp32)
    for s in range(n):
        if shared[].vec_dead[s] != 0:
            continue
        var kv = keyspace[].vec_key(s)
        if kv.is_none() or not kv.is_string():
            continue
        var kvp = kv.as_string_safe(kbuf)
        if not hk_claim(shared[].hk_keys_buf, shared[].hk_max_elements, s, kvp, kv.string_len()):
            continue
        var hv = keyspace[].get(GenericValue.borrow(kvp, kv.string_len()))
        if hv.type.value == ValueType.HASH:
            var hm = hv.as_hash().bitcast[SlabHashMap]()
            if refill:
                # A row whose hash has no vector of this DIM stays unwritten:
                # the last worker to refill declares it dead.
                if not _refill_row(shared, hm, s):
                    continue
                shared[].commit_ingest(s)
            hm[].index_vector(s, field_h, tomb)
    kbuf.unsafe_free()


def _refill_row(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                h: UnsafePointer[SlabHashMap, MutUntrackedOrigin], slot: Int) -> Bool:
    """Row `slot` from the index's vector field of hash `h`. False when the
    hash has no such field of `dim` float32s."""
    var fbuf = alloc[UInt8](24)
    var vbuf = alloc[UInt8](24)
    var ok = False
    for i in range(h[].capacity):
        if (h[].metadata[i] & 0x80) != 0:
            continue
        var f = h[].keys[i]
        if not _is_vector_field(shared, f.as_string_safe(fbuf), f.string_len()):
            continue
        var v = h[].values[i]
        if v.is_string() and v.string_len() == shared[].pre_dim * 4:
            shared[].fill_ingest_row(slot, v.as_string_safe(vbuf).bitcast[Float32]())
            ok = True
        break
    fbuf.unsafe_free()
    vbuf.unsafe_free()
    return ok
