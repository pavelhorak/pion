"""Pion iOS C API bridge.

Exports C-callable functions for the PionMesh iOS app.
Build: mojo build --target-triple arm64-apple-ios17.0 --emit object -I . src/lib/pion_ios.mojo -o pion_ios.o
"""

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from std.memory import unsafe_memcpy, unsafe_memset

from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue
from src.vector.hnsw import HNSWGraph
from src.memory.slab_allocator import SlabAllocator
from src.common.list import SlabList, ListNode


# Global state via Pointer (accessible from @export functions)
struct PionState:
    var keyspace: Pointer[StripedHashMap, MutUntrackedOrigin]
    var hnsw: Pointer[HNSWGraph, MutUntrackedOrigin]
    var initialized: Bool

    def __init__(out self):
        self.keyspace = null_ptr[StripedHashMap, MutUntrackedOrigin]()
        self.hnsw = null_ptr[HNSWGraph, MutUntrackedOrigin]()
        self.initialized = False


# Single global instance via alloc
var _state = alloc[PionState](1)


def _ensure_state() -> Pointer[PionState, MutUntrackedOrigin]:
    return _state


@export
def pion_init(max_elements: Int32, vector_dim: Int32) -> Int32:
    """Initialize Pion engine."""
    var s = _ensure_state()
    if s[].initialized:
        return Int32(0)

    s[].keyspace = alloc[StripedHashMap](1)
    (s[].keyspace).unsafe_write(StripedHashMap())

    s[].hnsw = alloc[HNSWGraph](1)
    try:
        (s[].hnsw).unsafe_write(
            HNSWGraph(max_elements=Int(max_elements), dim=Int(vector_dim), M=16, ef_construction=128)
        )
    except:
        return Int32(-1)

    s[].initialized = True
    return Int32(0)


@export
def pion_shutdown():
    var s = _ensure_state()
    s[].initialized = False


@export
def pion_set(key: Pointer[UInt8, MutUntrackedOrigin], key_len: Int,
             value: Pointer[UInt8, MutUntrackedOrigin], value_len: Int) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    var k = GenericValue.borrow(key, key_len)
    var v = GenericValue.borrow(value, value_len)
    s[].keyspace[].set(k, v)
    return Int32(0)


@export
def pion_get(key: Pointer[UInt8, MutUntrackedOrigin], key_len: Int,
             buf: Pointer[UInt8, MutUntrackedOrigin], buf_len: Int) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    var k = GenericValue.borrow(key, key_len)
    var result = s[].keyspace[].get(k)
    if result.is_none():
        return Int32(-1)
    var val_ptr = result.string_ptr()
    var val_len = result.string_len()
    if val_len > buf_len:
        return Int32(val_len)
    unsafe_memcpy(dest=buf, src=val_ptr, count=val_len)
    buf[val_len] = 0
    return Int32(val_len)


@export
def pion_del(key: Pointer[UInt8, MutUntrackedOrigin], key_len: Int) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(0)
    var k = GenericValue.from_ptr(key, key_len)
    var existed = s[].keyspace[].remove(k)
    return Int32(1) if existed else Int32(0)


@export
def pion_dbsize() -> Int64:
    var s = _ensure_state()
    if not s[].initialized: return Int64(0)
    var total = 0
    for i in range(8):
        total += s[].keyspace[].shards[i].size
    return Int64(total)


@export
def pion_vector_create_index(name: Pointer[UInt8, MutUntrackedOrigin],
                              dim: Int32, M: Int32, ef_construction: Int32) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    return Int32(0)


@export
def pion_vector_add(key: Pointer[UInt8, MutUntrackedOrigin], key_len: Int,
                     vector: Pointer[Float32, MutUntrackedOrigin], dim: Int32) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    var total = 0
    for i in range(8):
        total += s[].keyspace[].shards[i].size
    var id = total
    var k = GenericValue.borrow(key, key_len)
    var v = GenericValue.from_int(Int64(id))
    s[].keyspace[].set(k, v)
    try:
        s[].hnsw[].add_vector(id, vector)
    except:
        return Int32(-1)
    return Int32(0)


@export
def pion_vector_optimize() -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    try:
        s[].hnsw[].build_index()
    except:
        return Int32(-1)
    return Int32(0)


@export
def pion_save(path: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    var path_str = String("")
    var i = 0
    while path[i] != 0:
        path_str += String(chr(Int(path[i])))
        i += 1
    s[].hnsw[].save_to_disk(path_str)
    return Int32(0)


@export
def pion_load(path: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    var s = _ensure_state()
    if not s[].initialized: return Int32(-1)
    var path_str = String("")
    var i = 0
    while path[i] != 0:
        path_str += String(chr(Int(path[i])))
        i += 1
    var ok = s[].hnsw[].load_from_disk(path_str)
    return Int32(0) if ok else Int32(-1)
