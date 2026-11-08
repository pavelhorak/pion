# Mojo 1.0.0 (ed45d567), macOS arm64. `mojo build -O3 <this> && ./<bin>`
# Expected: "stack ints: 42". -O0 prints 42; -O3 prints garbage (6128380289).
# `--emit llvm` shows `tail call @consume(ptr %alloca)` after the two stores:
# the `tail` marker tells LLVM the callee does not read the caller's allocas,
# so the stores are dead. Passing a heap pointer, or making the callee generic
# over the origin instead of MutUntrackedOrigin, produces a plain `call`.
from std.memory import stack_allocation, alloc

@no_inline
def consume(v: Pointer[Int, MutUntrackedOrigin]) -> Int:
    return v[unsafe_offset=0] + v[unsafe_offset=1]

def main():
    var s = stack_allocation[2, Int]()
    s[unsafe_offset=0] = 40
    s[unsafe_offset=1] = 2
    print("stack ints:", consume(rebind[Pointer[Int, MutUntrackedOrigin]](s)))
