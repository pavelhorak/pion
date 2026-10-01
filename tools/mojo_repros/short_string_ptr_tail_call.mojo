# Mojo 1.0.0 (ed45d567), macOS arm64. `mojo build -O3 <this> && ./<bin>`
# Same `tail call` defect reached through String's inline storage: lengths
# 9-23 (stored in the String itself, i.e. on the stack) print WRONG at -O3;
# lengths >= 24 (heap buffer) are correct; -O0 is correct everywhere. The
# trailing `_ = d` keeps `d` alive, so this is not ASAP destruction.
from std.memory import UnsafePointer

@no_inline
def sum_bytes(p: UnsafePointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var t = 0
    for i in range(n):
        t += Int(p[i])
    return t

@no_inline
def mk(n: Int, reps: Int) -> String:
    var s = String("")
    for _ in range(reps):
        s += "a"
    return s + String(n)

def main():
    for reps in [1, 8, 20, 22, 23, 24, 30, 64]:
        var d = mk(7, reps)
        var want = 97 * reps + 55
        var got = sum_bytes(rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](d.unsafe_ptr()), reps + 1)
        print("len", reps + 1, "ok" if got == want else "WRONG", got, want)
        _ = d
