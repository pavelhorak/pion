# gh #199 spike artifact — proves Mojo defs run on foreign pthreads (the
# parallel-FT.OPTIMIZE architecture). Build + run:
#   pixi run mojo build -I . tests/smoke_gh199_build_pool.mojo \
#     -Xlinker src/ffi/build_pool_wrap.o -Xlinker -export_dynamic -o /tmp/smoke && /tmp/smoke
# Re-run on any Mojo toolchain bump (the @export/dlsym contract is the risk).

# gh #199 spike: can a Mojo def run on foreign pthreads?
# Path: @export the lane as a C symbol; the C shim dlsym-resolves it and runs
# it on 3 spawned pthreads + the caller. Exercises heap allocation (List),
# atomics, and pointer arithmetic — the runtime surface the real build worker
# needs.
from std.memory.unsafe_pointer import UnsafePointer
from std.memory import alloc
from std.atomic import Atomic, Ordering
from std.ffi import external_call


@export
def pion_gh199_smoke_lane(ctx: UnsafePointer[UInt64, MutUntrackedOrigin], idx: Int64):
    var l = List[Int]()
    for i in range(1000):
        l.append(i)
    var acc = 0
    for i in range(len(l)):
        acc += l[i]
    _ = Atomic[Scalar[DType.uint64]].fetch_add[ordering=Ordering.RELAXED](
        ctx, UInt64(len(l)) + UInt64(Int(idx)) + UInt64(acc % 2))  # acc even -> +0


def main():
    var counter = alloc[UInt64](1)
    counter[0] = 0
    var sym = String("pion_gh199_smoke_lane") + "\0"
    var rc = external_call["pion_build_pool_run_sym", Int32](
        sym.unsafe_ptr(), counter, Int64(4))
    # expect: 4 lanes × 1000 (acc is even) + (0+1+2+3) = 4006
    print("rc=", rc, " counter=", counter[0], " expect=4006")
    if rc == 0 and counter[0] == 4006:
        print("SMOKE OK — Mojo defs run on foreign pthreads")
    else:
        print("SMOKE FAILED")
