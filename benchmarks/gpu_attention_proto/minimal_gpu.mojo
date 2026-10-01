# Minimal GPU test to check if DeviceContext works at all

from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import memcpy
from std.sys.info import CompilationTarget
from std.math import ceildiv


def main() raises:
    print("Testing GPU DeviceContext...")

    comptime if CompilationTarget.is_macos():
        from std.gpu.host import DeviceContext, DeviceBuffer
        from std.gpu import block_dim, block_idx, thread_idx

        var ctx = DeviceContext()
        print("DeviceContext created")

        # Simple kernel: out[i] = in[i] * 2.0
        var n = 1024
        var in_buf = ctx.enqueue_create_buffer[DType.float32](n)
        var out_buf = ctx.enqueue_create_buffer[DType.float32](n)

        with in_buf.map_to_host() as h:
            for i in range(n):
                h[i] = Float32(i)

        def double_kernel(
            inp: UnsafePointer[Float32, MutAnyOrigin],
            dst: UnsafePointer[Float32, MutAnyOrigin],
        ):
            var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if gid >= 1024:
                return
            dst[gid] = inp[gid] * 2.0

        ctx.enqueue_function[double_kernel, double_kernel](
            in_buf, out_buf,
            grid_dim=ceildiv(n, 256), block_dim=256,
        )
        ctx.synchronize()

        with out_buf.map_to_host() as h:
            print("out[0] =", h[0], "(expected 0.0)")
            print("out[1] =", h[1], "(expected 2.0)")
            print("out[100] =", h[100], "(expected 200.0)")
            print("out[1023] =", h[1023], "(expected 2046.0)")

        print("GPU test passed!")
    else:
        print("Not macOS — skipping GPU test")
