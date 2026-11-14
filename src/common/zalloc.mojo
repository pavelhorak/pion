"""Zeroed tables that are never freed: anonymous mmap.

`zalloc[T](n)` is `unsafe_alloc[T](n)` followed by zeroing it, without the
zeroing: an anonymous mapping reads as zero, and a page becomes resident only
when something writes it. Pion keeps ~20 per-connection tables of 65,536
entries (CLIENT_MAX_FDS, MAX_TX_FDS, the writer's per-fd blocks), and writing
zeros into all of them at startup made ~8 MB resident that an idle server
never reads (idle RSS, and embedded Pion's footprint). Init-time only.

NEVER pass the pointer to `unsafe_free` / `free()`: it is not heap memory.
Use it for tables that live as long as the process.
"""
from std.ffi import external_call
from std.memory import unsafe_memset
from std.memory.alloc import unsafe_alloc
from std.sys import size_of, CompilationTarget
from src.common.ptr import null_ptr


def zalloc[T: AnyType](n: Int) -> Pointer[T, MutUntrackedOrigin]:
    var nbytes = n * size_of[T]()
    if nbytes <= 0:
        nbytes = 1
    var flags = 0x1002                # macOS: MAP_ANON | MAP_PRIVATE
    comptime if CompilationTarget.is_linux():
        flags = 0x22                  # Linux: MAP_ANONYMOUS | MAP_PRIVATE
    var p = external_call["mmap", Pointer[UInt8, MutUntrackedOrigin]](
        null_ptr[UInt8, MutUntrackedOrigin](), nbytes, 0x3, flags, -1, 0)
    if Int(p) == -1:                  # MAP_FAILED: fall back to the heap, zeroed
        p = unsafe_alloc[UInt8](nbytes)
        unsafe_memset(p, 0, nbytes)
    return p.unsafe_bitcast[T]()
