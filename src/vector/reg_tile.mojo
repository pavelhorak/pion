"""Register tiles for the batch distance kernels (gh #127).

A batch kernel scores one query against R vectors at once: one query load per
chunk, then R independent accumulator chains, so the R vectors' loads and
multiply-adds overlap. Each family used to spell the R chains out by hand
(sum0..sum7, d0..d7, r0..r7), once for R = 4 and again for R = 8.

`RegTile[dtype, rows, width]` holds the R accumulators. Every access is by a
compile-time row index inside a `comptime for`, so after inlining LLVM sees
the same straight-line code the hand-unrolled kernels had and keeps each
accumulator in a register (SROA). A family is written once, for any R, and
its batch-4 and batch-8 entry points are thin wrappers.

Pion-local, after MAX's `_Accumulator` (linalg/accumulate.mojo), without the
layout and tiling machinery importing it would pull in.

Rules for a kernel built on it:
- index rows only with `comptime for` values, never a runtime index, or the
  tile is spilled to the stack;
- do per row exactly the operations the unrolled form did, in the same order
  within the row: an FP32 kernel is bit-exact only if each row's expression
  tree is unchanged (rows are independent, so their interleaving is free);
- keep every function that touches a tile `@always_inline`. A tile is a stack
  local, and a pointer to a stack local must never reach an out-of-line call
  (the tail-call alloca rule).
"""
from std.collections import Array
from std.memory.unsafe_pointer import UnsafePointer


struct RegTile[dtype: DType, rows: Int, width: Int](Copyable, Movable):
    """`rows` accumulators of SIMD[dtype, width], zeroed at construction."""
    var acc: Array[SIMD[Self.dtype, Self.width], Self.rows]

    @always_inline
    def __init__(out self):
        self.acc = Array[SIMD[Self.dtype, Self.width], Self.rows](
            fill=SIMD[Self.dtype, Self.width](0))

    @always_inline
    def __getitem__(self, r: Int) -> SIMD[Self.dtype, Self.width]:
        return self.acc[r]

    @always_inline
    def __setitem__(mut self, r: Int, v: SIMD[Self.dtype, Self.width]):
        self.acc[r] = v

    @always_inline
    def reduce_f32(self) -> Array[Float32, Self.rows]:
        """Each row's horizontal sum, as Float32: `reduce_add` in the
        accumulator's own type, then one cast, as the unrolled kernels did."""
        var out = Array[Float32, Self.rows](uninitialized=True)
        comptime for r in range(Self.rows):
            out[r] = self.acc[r].reduce_add().cast[DType.float32]()
        return out^


@always_inline
def pack_rows[rows: Int](vals: Array[Float32, rows]) -> SIMD[DType.float32, rows]:
    """Per-row scalars into the SIMD the batch kernels return (rows must be a
    power of two)."""
    var out = SIMD[DType.float32, rows](0)
    comptime for r in range(rows):
        out[r] = vals[r]
    return out


comptime I8Ptr = UnsafePointer[Int8, MutUntrackedOrigin]


@always_inline
def ptr_rows4(v0: I8Ptr, v1: I8Ptr, v2: I8Ptr, v3: I8Ptr) -> Array[I8Ptr, 4]:
    """The row pointers of a batch-4 call, for a `rows=4` microkernel."""
    var out = Array[I8Ptr, 4](uninitialized=True)
    out[0] = v0; out[1] = v1; out[2] = v2; out[3] = v3
    return out^


@always_inline
def ptr_rows8(v0: I8Ptr, v1: I8Ptr, v2: I8Ptr, v3: I8Ptr,
              v4: I8Ptr, v5: I8Ptr, v6: I8Ptr, v7: I8Ptr) -> Array[I8Ptr, 8]:
    """The row pointers of a batch-8 call, for a `rows=8` microkernel."""
    var out = Array[I8Ptr, 8](uninitialized=True)
    out[0] = v0; out[1] = v1; out[2] = v2; out[3] = v3
    out[4] = v4; out[5] = v5; out[6] = v6; out[7] = v7
    return out^
