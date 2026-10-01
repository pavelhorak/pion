from std.sys.info import CompilationTarget
from std.memory.unsafe_pointer import UnsafePointer, alloc
from std.memory import unsafe_memcpy

comptime BS: Int = 256


def main() raises:
    comptime if CompilationTarget.is_macos():
        from std.gpu import block_dim, block_idx, thread_idx
        from std.gpu.host import DeviceContext, DeviceBuffer
        from std.math import ceildiv

        def trivial_kernel(s_ptr: UnsafePointer[Float32, MutAnyOrigin]):
            var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
            if gid >= 256:
                return
            s_ptr[gid] = 1.0

        var ctx_ptr = alloc[DeviceContext](1)
        ctx_ptr.unsafe_write(DeviceContext())
        var ctx = ctx_ptr

        var s_buf = ctx[].enqueue_create_buffer[DType.float32](256)

        var blocks = ceildiv(256, BS)
        ctx[].enqueue_function[trivial_kernel, trivial_kernel](
            s_buf, grid_dim=blocks, block_dim=BS)
        ctx[].synchronize()

        var host_arr = alloc[Float32](256)
        with s_buf.map_to_host() as h:
            unsafe_memcpy(dest=host_arr.bitcast[UInt8](), src=h.unsafe_ptr().bitcast[UInt8](), count=256 * 4)

        var ok = True
        for i in range(256):
            if host_arr[i] != 1.0:
                ok = False
                break
        if ok:
            print("TRIVIAL KERNEL OK")
        else:
            print("TRIVIAL KERNEL FAIL")
        host_arr.free()
