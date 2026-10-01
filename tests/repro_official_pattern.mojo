from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime float_dtype = DType.float32
comptime VECTOR_WIDTH = 10
comptime BLOCK_SIZE = 5
comptime layout = row_major[VECTOR_WIDTH]()


def main() raises:
    comptime assert has_accelerator(), "This example requires a supported GPU"

    var ctx = DeviceContext()
    var lhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var rhs_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)
    var out_buffer = ctx.enqueue_create_buffer[float_dtype](VECTOR_WIDTH)

    lhs_buffer.enqueue_fill(1.25)
    rhs_buffer.enqueue_fill(2.5)

    var lhs_tensor = TileTensor(lhs_buffer, layout)
    var rhs_tensor = TileTensor(rhs_buffer, layout)
    var out_tensor = TileTensor(out_buffer, layout)

    var grid_dim = ceildiv(VECTOR_WIDTH, BLOCK_SIZE)

    ctx.enqueue_function[vector_addition, vector_addition](
        lhs_tensor,
        rhs_tensor,
        out_tensor,
        VECTOR_WIDTH,
        grid_dim=grid_dim,
        block_dim=BLOCK_SIZE,
    )

    with out_buffer.map_to_host() as host_buffer:
        var host_tensor = TileTensor(host_buffer, layout)
        print("Resulting vector:", host_tensor)


def vector_addition(
    lhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    rhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    out_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    size: Int,
):
    var global_tid = global_idx.x
    if global_tid < size:
        out_tensor[global_tid] = lhs_tensor[global_tid] + rhs_tensor[global_tid]
