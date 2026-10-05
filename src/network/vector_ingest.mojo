"""#43: a hash's vector field reaches the vector index from every command that
sets hash fields.

The fast path's HSET arms (src/network/fast_path.mojo, single- and multi-field)
route a field to the index inline: it is the VectorDBBench ingest path, and the
HSET gate row is measured on it. Every other way to set a hash field went
through the slow path, which stored the field and never indexed it: HSET inside
MULTI/EXEC, from a script, pipelined behind a slow-path command or while a
client monitors; HMSET and HSETNX always. FT.SEARCH then missed documents that
HGET showed were there. This is the slow path's copy of the same steps, and the
two must agree:

  1. the index is defined (`pre_index_ready`), the value is `dim` float32s and
     the field is the index's vector field (folded as the fast path folds it);
  2. the vector goes to the shared ingest buffer, which returns its slot;
  3. `__hk__<slot>` → key is set in the keyspace and logged to the WAL (the
     only durable slot → key map for keys over 31 bytes, gh #211), and keys of
     up to 31 bytes also go to the shared map every worker reads.
"""

from std.memory import alloc, unsafe_memcpy
from std.memory.unsafe_pointer import UnsafePointer
from src.common.ptr import is_not_null, is_null
from src.common.hash_map import StripedHashMap
from src.common.value import GenericValue
from src.common.utils import format_int_to_buf
from src.vector.hnsw import SharedHNSWView
from src.io.wal import WAL


def ingest_hash_vector(shared: UnsafePointer[SharedHNSWView, MutUntrackedOrigin],
                       keyspace: UnsafePointer[StripedHashMap, MutUntrackedOrigin],
                       wal: UnsafePointer[WAL, MutUntrackedOrigin],
                       key: UnsafePointer[UInt8, MutUntrackedOrigin], klen: Int,
                       field: UnsafePointer[UInt8, MutUntrackedOrigin], flen: Int,
                       val: UnsafePointer[UInt8, MutUntrackedOrigin], vlen: Int) -> Bool:
    """Add the field's vector to the index when it is the index's vector field.
    True when it was added. The caller stores the field in the hash as well,
    as for any field (gh #360)."""
    if is_null(shared) or not shared[].pre_index_ready:
        return False
    if vlen != shared[].pre_dim * 4 or flen != shared[].pre_vector_field_len:
        return False
    var name = shared[].pre_vector_field_name.unsafe_ptr()
    for b in range(flen):
        if (field[b] | 0x20) != name[b]:
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
    return True
