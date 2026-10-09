"""Transparent huge pages for the big, randomly probed regions (gh #204).

`advise_hugepage(addr, nbytes)` asks Linux to back a region with 2 MiB pages:
madvise(MADV_HUGEPAGE) over the 2 MiB-aligned interior of the region. Call it
right after the allocation and BEFORE the first write, because a page that is
already faulted in as 4 KiB stays that way until khugepaged collapses it.

The call sites are the regions a lookup reaches by random index, where a 4 KiB
TLB entry covers almost nothing:
  - the SlabHashMap metadata, keys and values (10M slots per worker);
  - the SlabAllocator mmap arenas;
  - the HNSW neighbor pool, visit epochs, compact vector buffer and level-0
    lists (66 MB at 500K vectors).

It does something only with `PION_MADV_HUGEPAGE=1` in the environment while
#204 is measured; without it nothing calls madvise, which is the behaviour
before #204. It is a no-op on macOS, which has no superpages for user memory
(16 KiB base pages only), and for regions under 2 MiB.
"""
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.sys import CompilationTarget
from src.common.ptr import is_not_null

comptime MADV_HUGEPAGE = 14
comptime HUGE_PAGE = 2 * 1024 * 1024


def hugepage_advice_enabled() -> Bool:
    """True when PION_MADV_HUGEPAGE=1 (Linux only). Read at each call: the call
    sites are allocations of megabytes, never the request path."""
    comptime if CompilationTarget.is_linux():
        var p = external_call["getenv", Pointer[UInt8, MutUntrackedOrigin]](
            "PION_MADV_HUGEPAGE\0".unsafe_ptr())
        return is_not_null(p) and p[] == UInt8(ord("1"))
    else:
        return False


def advise_hugepage(addr: Int, nbytes: Int):
    """madvise(MADV_HUGEPAGE) over the 2 MiB-aligned interior of [addr, addr+nbytes)."""
    comptime if CompilationTarget.is_linux():
        if nbytes < HUGE_PAGE or not hugepage_advice_enabled():
            return
        var start = (addr + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1)
        var end = (addr + nbytes) & ~(HUGE_PAGE - 1)
        if end <= start:
            return
        _ = external_call["madvise", Int32](start, end - start, Int32(MADV_HUGEPAGE))
