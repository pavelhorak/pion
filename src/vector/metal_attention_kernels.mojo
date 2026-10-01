# Mojo+Metal attention kernels for Pion's V-store hot path.
#
# PoC scope (this file):
#   - Single-head FP32 attention: O[d_head] = softmax(Q @ K^T / sqrt(d_head)) @ V
#   - One GPU kernel (scores Q@K^T) + CPU softmax + CPU V-product
#   - d_head pinned to comptime D_HEAD=128 (Mojo Metal codegen requires comptime loops)
#
# CURRENT STATUS (2026-04-30 PoC, build 0.562):
#   - cpu_attend_reference correct and deterministic (tests/bench_attend_cpu_only.mojo)
#   - GPU path validated: cosine(cpu, gpu) ≥ 0.99999994 across N ∈ {16, 256, 2048}
#     (tests/bench_metal_attention.mojo ALL CASES PASSED)
#   - d_head pinned to comptime 128 (matches gpu_search.mojo's DIM_1536 pattern)
#   - Single GPU kernel (Q @ K^T scores) + CPU softmax + CPU V-product
#
# Toolchain note: the original build failure ("Metal Compiler failed to compile
# metallib") was a missing Apple Metal toolchain — Xcode-select pointed at
# CommandLineTools instead of full Xcode. Fix recorded in
# memory/mac_metal_toolchain_required.md.
#
# Out of scope (follow-ups, in roughly increasing difficulty):
#   - Resolve Mojo+Metal codegen blocker (likely Mojo nightly issue;
#     `pion_release_gated_by_mojo_ga.md` notes a Pointer migration
#     in 042305+ nightlies that may interact)
#   - FP16 K/V (drop the FP32 cast)
#   - Multi-head batched dispatch (one threadgroup per head)
#   - Fused single-kernel softmax via threadgroup reductions
#   - turbo4/turbo3/turbo2/int8 V-store quant decode lane-wise inside the kernel
#   - Wire into ATTEND.PREFIX.QUERY when --mlx-attention is off
#
# On Linux: this struct is a no-op stub (ready=False; attend returns 0).

from std.sys.info import CompilationTarget
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.math import sqrt, exp

comptime ATTN_BLOCK_SIZE: Int = 256
# PoC pins d_head to a single value because Mojo's Metal codegen requires
# comptime loop bounds (same constraint gpu_search.mojo follows with DIM_1536).
# Production: compile {64, 96, 128, 256} variants and dispatch by runtime d.
comptime D_HEAD: Int = 128

# FlashAttention tile size for K/V along the sequence dim. Threadgroup memory
# usage: Q + K_tile + V_tile + S_tile + scalars.
#   FP32 storage:  BC=24 → 25 KB,   BC=32 → 33 KB (over 32 KB)
#   FP16 storage:  BC=48 → 25 KB,   BC=64 → 33 KB (over)
# FP32 path uses BC=24; FP16 path uses BC_FP16=48 (twice the work per tile).
comptime BC: Int = 24
comptime BC_FP16: Int = 48
# Split-K: how many threadgroups handle each head's K dimension. Total
# threadgroups = H * K_SPLIT. With H=8 K_SPLIT=8 = 64 threadgroups → saturates
# any Apple Silicon GPU (M1=8 EUs, M-Pro=16, M-Max=40 typically).
comptime K_SPLIT: Int = 8
# Sentinel for "no max yet" in the online softmax. exp(NEG_INF - x) → 0 cleanly
# and avoids a special-case branch on the first tile.
comptime NEG_INF: Float32 = -3.4e38


struct MetalAttentionContext(Movable):
    var ready: Bool
    var d_head: Int
    var max_n: Int
    # Raw pointers (UInt64) to DeviceContext / DeviceBuffer to avoid platform-conditional types.
    # On Linux these stay 0 and `attend()` short-circuits.
    var _ctx: UInt64
    var _q_buf: UInt64       # [d_head] FP32
    var _k_buf: UInt64       # [max_n × d_head] FP32
    var _v_buf: UInt64       # [max_n × d_head] FP32
    var _scores_buf: UInt64  # [max_n] FP32 — scores after Q@K^T / sqrt(d), then softmax
    var _out_buf: UInt64     # [d_head] FP32

    def __init__(out self):
        self.ready = False
        self.d_head = 0
        self.max_n = 0
        self._ctx = 0
        self._q_buf = 0
        self._k_buf = 0
        self._v_buf = 0
        self._scores_buf = 0
        self._out_buf = 0

    def init_device(mut self, d_head: Int, max_n: Int) raises:
        self.d_head = d_head
        self.max_n = max_n

        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceContext, DeviceBuffer

            var ctx_ptr = alloc[DeviceContext](1)
            ctx_ptr.unsafe_write(DeviceContext())
            self._ctx = UInt64(Int(ctx_ptr))
            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))

            var q_ptr = alloc[DeviceBuffer[DType.float32]](1)
            q_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](d_head))
            self._q_buf = UInt64(Int(q_ptr))

            var k_ptr = alloc[DeviceBuffer[DType.float32]](1)
            k_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](max_n * d_head))
            self._k_buf = UInt64(Int(k_ptr))

            var v_ptr = alloc[DeviceBuffer[DType.float32]](1)
            v_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](max_n * d_head))
            self._v_buf = UInt64(Int(v_ptr))

            var s_ptr = alloc[DeviceBuffer[DType.float32]](1)
            s_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](max_n))
            self._scores_buf = UInt64(Int(s_ptr))

            var o_ptr = alloc[DeviceBuffer[DType.float32]](1)
            o_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](d_head))
            self._out_buf = UInt64(Int(o_ptr))

            self.ready = True

    def attend(mut self,
              q: Pointer[Float32, MutAnyOrigin],
              k: Pointer[Float32, MutAnyOrigin],
              v: Pointer[Float32, MutAnyOrigin],
              n: Int,
              out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_dim, block_idx, thread_idx
            from std.gpu.host import DeviceContext, DeviceBuffer
            from std.math import ceildiv

            var d = self.d_head
            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(d))

            # ── Kernel 1: scores[i] = (Q · K[i]) * inv_sqrt_d  ────────────────
            # One thread per token i. d_head is comptime (D_HEAD).
            def scores_kernel(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float32, MutAnyOrigin],
                s_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var gid = Int(block_idx.x * block_dim.x + thread_idx.x)
                var nv = Int(n_ptr[0])
                if gid >= nv:
                    return
                var k_row = k_ptr + gid * D_HEAD
                var acc: Float32 = 0.0
                for j in range(D_HEAD):
                    acc += q_ptr[j] * k_row[j]
                s_ptr[gid] = acc * inv_sqrt_d_ptr[0]

            # Note on PoC scope: only the scores kernel runs on GPU. Softmax and
            # the V product run on CPU below. Promoting V product to GPU is the
            # next step (a 2D-indexed kernel hit a Metal compiler bug here, worth
            # a follow-up to bisect — likely runtime-stride pattern in the inner
            # loop). Once that's fixed, the natural endpoint is one fused kernel
            # with a threadgroup-reduction softmax.

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var s_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._scores_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._out_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=d * 4)
            with k_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=k.unsafe_bitcast[UInt8](), count=n * d * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            var blocks_n = ceildiv(n, ATTN_BLOCK_SIZE)
            ctx[].enqueue_function[scores_kernel, scores_kernel](
                q_buf[], k_buf[], s_buf[], n_const, inv_const,
                grid_dim=blocks_n, block_dim=ATTN_BLOCK_SIZE)
            ctx[].synchronize()

            var scores_cpu = alloc[Float32](n)
            with s_buf[].map_to_host() as h:
                unsafe_memcpy(dest=scores_cpu.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=n * 4)

            var smax: Float32 = scores_cpu[0]
            for i in range(1, n):
                if scores_cpu[i] > smax:
                    smax = scores_cpu[i]
            var ssum: Float32 = 0.0
            for i in range(n):
                scores_cpu[i] = exp(scores_cpu[i] - smax)
                ssum += scores_cpu[i]
            var inv_sum: Float32 = 1.0 / ssum
            for i in range(n):
                scores_cpu[i] *= inv_sum

            for j in range(d):
                var acc: Float32 = 0.0
                for i in range(n):
                    acc += scores_cpu[i] * v[i * d + j]
                out_buf[j] = acc

            scores_cpu.unsafe_free()
            return d

        return 0


struct MetalMHAttentionContext(Movable):
    """Multi-head batched attention. Layout: [H, N, D_HEAD] for K/V; [H, D_HEAD] for Q/O; [H, N] for scores."""
    var ready: Bool
    var d_head: Int
    var h_heads: Int
    var max_n: Int
    var _ctx: UInt64
    var _q_buf: UInt64
    var _k_buf: UInt64
    var _v_buf: UInt64
    var _scores_buf: UInt64
    var _out_buf: UInt64

    def __init__(out self):
        self.ready = False
        self.d_head = 0
        self.h_heads = 0
        self.max_n = 0
        self._ctx = 0
        self._q_buf = 0
        self._k_buf = 0
        self._v_buf = 0
        self._scores_buf = 0
        self._out_buf = 0

    def init_device(mut self, d_head: Int, h_heads: Int, max_n: Int) raises:
        self.d_head = d_head
        self.h_heads = h_heads
        self.max_n = max_n

        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceContext, DeviceBuffer

            var ctx_ptr = alloc[DeviceContext](1)
            ctx_ptr.unsafe_write(DeviceContext())
            self._ctx = UInt64(Int(ctx_ptr))
            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))

            var q_ptr = alloc[DeviceBuffer[DType.float32]](1)
            q_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._q_buf = UInt64(Int(q_ptr))

            var k_ptr = alloc[DeviceBuffer[DType.float32]](1)
            k_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._k_buf = UInt64(Int(k_ptr))

            var v_ptr = alloc[DeviceBuffer[DType.float32]](1)
            v_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._v_buf = UInt64(Int(v_ptr))

            var s_ptr = alloc[DeviceBuffer[DType.float32]](1)
            s_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n))
            self._scores_buf = UInt64(Int(s_ptr))

            var o_ptr = alloc[DeviceBuffer[DType.float32]](1)
            o_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._out_buf = UInt64(Int(o_ptr))

            self.ready = True

    def attend(mut self,
               q: Pointer[Float32, MutAnyOrigin],   # [H, D_HEAD]
               k: Pointer[Float32, MutAnyOrigin],   # [H, N, D_HEAD]
               v: Pointer[Float32, MutAnyOrigin],   # [H, N, D_HEAD]
               n: Int,
               out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_dim, block_idx, thread_idx
            from std.gpu.host import DeviceContext, DeviceBuffer
            from std.math import ceildiv

            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(D_HEAD))
            var H = self.h_heads

            # 2D grid: (token-block, head). Each thread = one (head, token) pair.
            def mh_scores_kernel(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float32, MutAnyOrigin],
                s_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var i = Int(block_idx.x * block_dim.x + thread_idx.x)
                var h = Int(block_idx.y)
                var nv = Int(n_ptr[0])
                if i >= nv:
                    return
                var q_row = q_ptr + h * D_HEAD
                var k_row = k_ptr + h * nv * D_HEAD + i * D_HEAD
                var acc: Float32 = 0.0
                for j in range(D_HEAD):
                    acc += q_row[j] * k_row[j]
                s_ptr[h * nv + i] = acc * inv_sqrt_d_ptr[0]

            # 2D grid: (d-block, head). Each thread = one (head, output_dim) pair.
            def mh_matvec_kernel(
                s_ptr: Pointer[Float32, MutAnyOrigin],
                v_ptr: Pointer[Float32, MutAnyOrigin],
                out_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
            ):
                var d_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
                var h = Int(block_idx.y)
                if d_idx >= D_HEAD:
                    return
                var nv = Int(n_ptr[0])
                var s_row = s_ptr + h * nv
                var v_base = v_ptr + h * nv * D_HEAD
                var acc: Float32 = 0.0
                for i in range(nv):
                    acc += s_row[i] * v_base[i * D_HEAD + d_idx]
                out_ptr[h * D_HEAD + d_idx] = acc

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var s_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._scores_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._out_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            with k_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=k.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)
            with v_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=v.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            var blocks_n = ceildiv(n, ATTN_BLOCK_SIZE)
            ctx[].enqueue_function[mh_scores_kernel, mh_scores_kernel](
                q_buf[], k_buf[], s_buf[], n_const, inv_const,
                grid_dim=(blocks_n, H, 1), block_dim=(ATTN_BLOCK_SIZE, 1, 1))
            ctx[].synchronize()

            # Per-head softmax on CPU (independent reductions per row of scores[H, N]).
            var scores_cpu = alloc[Float32](H * n)
            with s_buf[].map_to_host() as h:
                unsafe_memcpy(dest=scores_cpu.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * n * 4)

            for h_idx in range(H):
                var row = scores_cpu + h_idx * n
                var smax: Float32 = row[0]
                for i in range(1, n):
                    if row[i] > smax:
                        smax = row[i]
                var ssum: Float32 = 0.0
                for i in range(n):
                    row[i] = exp(row[i] - smax)
                    ssum += row[i]
                var inv_sum: Float32 = 1.0 / ssum
                for i in range(n):
                    row[i] *= inv_sum

            with s_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=scores_cpu.unsafe_bitcast[UInt8](), count=H * n * 4)
            scores_cpu.unsafe_free()

            var blocks_d = ceildiv(D_HEAD, ATTN_BLOCK_SIZE)
            ctx[].enqueue_function[mh_matvec_kernel, mh_matvec_kernel](
                s_buf[], v_buf[], o_buf[], n_const,
                grid_dim=(blocks_d, H, 1), block_dim=(ATTN_BLOCK_SIZE, 1, 1))
            ctx[].synchronize()

            with o_buf[].map_to_host() as h:
                unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            return H * D_HEAD

        return 0


struct MetalFlashAttentionContext(Movable):
    """FlashAttention-style fused multi-head attention.

    One threadgroup per head, D_HEAD threads per group. Tiles K/V along the
    sequence dim into BC-token chunks held in threadgroup-shared memory; uses
    online softmax (running m, l) so the full N×N scores matrix is never
    materialized. Single GPU dispatch (no separate scores/softmax/matvec).
    """
    var ready: Bool
    var d_head: Int
    var h_heads: Int
    var max_n: Int
    var _ctx: UInt64
    var _q_buf: UInt64
    var _k_buf: UInt64
    var _v_buf: UInt64
    var _o_buf: UInt64

    def __init__(out self):
        self.ready = False
        self.d_head = 0
        self.h_heads = 0
        self.max_n = 0
        self._ctx = 0
        self._q_buf = 0
        self._k_buf = 0
        self._v_buf = 0
        self._o_buf = 0

    def init_device(mut self, d_head: Int, h_heads: Int, max_n: Int) raises:
        self.d_head = d_head
        self.h_heads = h_heads
        self.max_n = max_n

        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceContext, DeviceBuffer

            var ctx_ptr = alloc[DeviceContext](1)
            ctx_ptr.unsafe_write(DeviceContext())
            self._ctx = UInt64(Int(ctx_ptr))
            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))

            var q_ptr = alloc[DeviceBuffer[DType.float32]](1)
            q_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._q_buf = UInt64(Int(q_ptr))

            var k_ptr = alloc[DeviceBuffer[DType.float32]](1)
            k_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._k_buf = UInt64(Int(k_ptr))

            var v_ptr = alloc[DeviceBuffer[DType.float32]](1)
            v_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._v_buf = UInt64(Int(v_ptr))

            var o_ptr = alloc[DeviceBuffer[DType.float32]](1)
            o_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._o_buf = UInt64(Int(o_ptr))

            self.ready = True

    def attend(mut self,
               q: Pointer[Float32, MutAnyOrigin],   # [H, D_HEAD]
               k: Pointer[Float32, MutAnyOrigin],   # [H, N, D_HEAD]
               v: Pointer[Float32, MutAnyOrigin],   # [H, N, D_HEAD]
               n: Int,
               out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_idx, thread_idx
            from std.gpu.sync import barrier
            from std.gpu.memory import AddressSpace
            from std.gpu.host import DeviceContext, DeviceBuffer
            from layout import stack_allocation
            from layout.tile_layout import row_major

            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(D_HEAD))
            var H = self.h_heads

            alias q_layout = row_major[D_HEAD]()
            alias k_tile_layout = row_major[BC, D_HEAD]()
            alias v_tile_layout = row_major[BC, D_HEAD]()
            alias s_tile_layout = row_major[BC]()
            alias scalar_layout = row_major[1]()

            def fa_kernel(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float32, MutAnyOrigin],
                v_ptr: Pointer[Float32, MutAnyOrigin],
                o_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var h = Int(block_idx.y)
                var tid = Int(thread_idx.x)
                var nv = Int(n_ptr[0])
                var inv_sd: Float32 = inv_sqrt_d_ptr[0]

                var Q_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](q_layout)
                var K_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](k_tile_layout)
                var V_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](v_tile_layout)
                var S_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](s_tile_layout)
                var m_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](scalar_layout)

                Q_shared[tid] = q_ptr[h * D_HEAD + tid]
                barrier()

                var m_run: Float32 = NEG_INF
                var l_run: Float32 = 0.0
                var O_acc: Float32 = 0.0

                var num_tiles = (nv + BC - 1) // BC
                for tile_idx in range(num_tiles):
                    var tile_start = tile_idx * BC

                    # Cooperative load: thread tid loads dim tid for all BC tokens.
                    for j in range(BC):
                        var gtok = tile_start + j
                        if gtok < nv:
                            K_tile[j, tid] = k_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                            V_tile[j, tid] = v_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                        else:
                            K_tile[j, tid] = 0.0
                            V_tile[j, tid] = 0.0
                    barrier()

                    # First BC threads compute one score each.
                    if tid < BC:
                        var score: Float32 = 0.0
                        for d in range(D_HEAD):
                            score += Q_shared[d] * K_tile[tid, d]
                        S_tile[tid] = score * inv_sd
                    barrier()

                    # Thread 0 reduces tile max (BC=16 small enough for serial).
                    if tid == 0:
                        var mx: Float32 = NEG_INF
                        for j in range(BC):
                            if (tile_start + j) < nv:
                                if S_tile[j] > mx:
                                    mx = S_tile[j]
                        m_shared[0] = mx
                    barrier()

                    var m_tile = m_shared[0]
                    var m_new = m_run
                    if m_tile > m_new:
                        m_new = m_tile
                    var alpha: Float32 = exp(m_run - m_new)

                    O_acc *= alpha
                    var l_tile: Float32 = 0.0
                    for j in range(BC):
                        if (tile_start + j) < nv:
                            var w = exp(S_tile[j] - m_new)
                            l_tile += w
                            O_acc += w * V_tile[j, tid]
                    l_run = l_run * alpha + l_tile
                    m_run = m_new
                    barrier()

                o_ptr[h * D_HEAD + tid] = O_acc / l_run

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._o_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            with k_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=k.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)
            with v_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=v.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            ctx[].enqueue_function[fa_kernel, fa_kernel](
                q_buf[], k_buf[], v_buf[], o_buf[], n_const, inv_const,
                grid_dim=(1, H, 1), block_dim=(D_HEAD, 1, 1))
            ctx[].synchronize()

            with o_buf[].map_to_host() as h:
                unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            return H * D_HEAD

        return 0

    def upload_kv(mut self,
                  k: Pointer[Float32, MutAnyOrigin],
                  v: Pointer[Float32, MutAnyOrigin],
                  n: Int) raises:
        """Upload K/V once; subsequent attend_q_resident() calls reuse them.
        Models the production case where Pion's V-store keeps K/V on the GPU
        and only Q changes per query.
        """
        if not self.ready:
            return
        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceBuffer
            var H = self.h_heads
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            with k_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=k.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)
            with v_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=v.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)

    def attend_q_resident(mut self,
                          q: Pointer[Float32, MutAnyOrigin],
                          n: Int,
                          out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        """Like attend(), but assumes K/V already in _k_buf/_v_buf via upload_kv()."""
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_idx, thread_idx
            from std.gpu.sync import barrier
            from std.gpu.memory import AddressSpace
            from std.gpu.primitives.warp import sum as warp_sum
            from std.gpu.host import DeviceContext, DeviceBuffer
            from layout import stack_allocation
            from layout.tile_layout import row_major

            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(D_HEAD))
            var H = self.h_heads

            alias q_layout = row_major[D_HEAD]()
            alias k_tile_layout = row_major[BC, D_HEAD]()
            alias v_tile_layout = row_major[BC, D_HEAD]()
            alias s_tile_layout = row_major[BC]()
            alias scalar_layout = row_major[1]()

            def fa_kernel_r(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float32, MutAnyOrigin],
                v_ptr: Pointer[Float32, MutAnyOrigin],
                o_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var h = Int(block_idx.y)
                var tid = Int(thread_idx.x)
                var nv = Int(n_ptr[0])
                var inv_sd: Float32 = inv_sqrt_d_ptr[0]

                var Q_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](q_layout)
                var K_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](k_tile_layout)
                var V_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](v_tile_layout)
                var S_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](s_tile_layout)
                var m_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](scalar_layout)

                Q_shared[tid] = q_ptr[h * D_HEAD + tid]
                barrier()

                var m_run: Float32 = NEG_INF
                var l_run: Float32 = 0.0
                var O_acc: Float32 = 0.0

                var num_tiles = (nv + BC - 1) // BC
                for tile_idx in range(num_tiles):
                    var tile_start = tile_idx * BC
                    for j in range(BC):
                        var gtok = tile_start + j
                        if gtok < nv:
                            K_tile[j, tid] = k_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                            V_tile[j, tid] = v_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                        else:
                            K_tile[j, tid] = 0.0
                            V_tile[j, tid] = 0.0
                    barrier()

                    # Simdgroup-parallel score: 4 simdgroups (32 threads each)
                    # cooperatively compute 4 scores per outer iter. Each lane
                    # handles D_HEAD/32 = 4 dim elements; warp_sum reduces 32-way.
                    # 6 outer iters cover BC=24 scores. All 128 threads stay busy.
                    var sg_id = tid // 32
                    var lane = tid % 32
                    for outer in range(0, BC, 4):
                        var score_idx = outer + sg_id
                        var partial: Float32 = 0.0
                        if score_idx < BC:
                            for d_chunk in range(D_HEAD // 32):
                                var d = lane + d_chunk * 32
                                partial += Q_shared[d] * K_tile[score_idx, d]
                        var total = warp_sum(partial)
                        if score_idx < BC and lane == 0:
                            S_tile[score_idx] = total * inv_sd
                    barrier()

                    if tid == 0:
                        var mx: Float32 = NEG_INF
                        for j in range(BC):
                            if (tile_start + j) < nv:
                                if S_tile[j] > mx:
                                    mx = S_tile[j]
                        m_shared[0] = mx
                    barrier()

                    var m_tile = m_shared[0]
                    var m_new = m_run
                    if m_tile > m_new:
                        m_new = m_tile
                    var alpha: Float32 = exp(m_run - m_new)

                    O_acc *= alpha
                    var l_tile: Float32 = 0.0
                    for j in range(BC):
                        if (tile_start + j) < nv:
                            var w = exp(S_tile[j] - m_new)
                            l_tile += w
                            O_acc += w * V_tile[j, tid]
                    l_run = l_run * alpha + l_tile
                    m_run = m_new
                    barrier()

                o_ptr[h * D_HEAD + tid] = O_acc / l_run

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._o_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            ctx[].enqueue_function[fa_kernel_r, fa_kernel_r](
                q_buf[], k_buf[], v_buf[], o_buf[], n_const, inv_const,
                grid_dim=(1, H, 1), block_dim=(D_HEAD, 1, 1))
            ctx[].synchronize()

            with o_buf[].map_to_host() as h:
                unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            return H * D_HEAD

        return 0


struct MetalFlashAttentionFP16Context(Movable):
    """FP16 FlashAttention. K/V stored as FP16 (halves memory bandwidth, doubles
    compute throughput on Apple Silicon FP16 SIMD). BC=48 (twice FP32 path).
    Q stays FP32 in threadgroup memory; accumulators FP32 (mixed precision).
    """
    var ready: Bool
    var d_head: Int
    var h_heads: Int
    var max_n: Int
    var _ctx: UInt64
    var _q_buf: UInt64        # [H, D_HEAD] FP32
    var _k_buf: UInt64        # [H, max_n, D_HEAD] FP16
    var _v_buf: UInt64        # [H, max_n, D_HEAD] FP16
    var _o_buf: UInt64        # [H, D_HEAD] FP32

    def __init__(out self):
        self.ready = False
        self.d_head = 0
        self.h_heads = 0
        self.max_n = 0
        self._ctx = 0
        self._q_buf = 0
        self._k_buf = 0
        self._v_buf = 0
        self._o_buf = 0

    def init_device(mut self, d_head: Int, h_heads: Int, max_n: Int) raises:
        self.d_head = d_head
        self.h_heads = h_heads
        self.max_n = max_n

        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceContext, DeviceBuffer

            var ctx_ptr = alloc[DeviceContext](1)
            ctx_ptr.unsafe_write(DeviceContext())
            self._ctx = UInt64(Int(ctx_ptr))
            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))

            var q_ptr = alloc[DeviceBuffer[DType.float32]](1)
            q_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._q_buf = UInt64(Int(q_ptr))

            var k_ptr = alloc[DeviceBuffer[DType.float16]](1)
            k_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float16](h_heads * max_n * d_head))
            self._k_buf = UInt64(Int(k_ptr))

            var v_ptr = alloc[DeviceBuffer[DType.float16]](1)
            v_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float16](h_heads * max_n * d_head))
            self._v_buf = UInt64(Int(v_ptr))

            var o_ptr = alloc[DeviceBuffer[DType.float32]](1)
            o_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._o_buf = UInt64(Int(o_ptr))

            self.ready = True

    def upload_kv(mut self,
                  k: Pointer[Float32, MutAnyOrigin],
                  v: Pointer[Float32, MutAnyOrigin],
                  n: Int) raises:
        """Upload K/V from FP32 host into FP16 device buffers."""
        if not self.ready:
            return
        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceBuffer
            var H = self.h_heads
            var k_buf = Pointer[DeviceBuffer[DType.float16], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float16], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var total = H * n * D_HEAD
            with k_buf[].map_to_host() as h_k:
                var dst = h_k.unsafe_ptr()
                for i in range(total):
                    dst[i] = k[i].cast[DType.float16]()
            with v_buf[].map_to_host() as h_v:
                var dst = h_v.unsafe_ptr()
                for i in range(total):
                    dst[i] = v[i].cast[DType.float16]()

    def attend_q_resident(mut self,
                          q: Pointer[Float32, MutAnyOrigin],
                          n: Int,
                          out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        """Run FP16 FlashAttention assuming K/V already uploaded via upload_kv()."""
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_idx, thread_idx
            from std.gpu.sync import barrier
            from std.gpu.memory import AddressSpace
            from std.gpu.primitives.warp import sum as warp_sum
            from std.gpu.host import DeviceContext, DeviceBuffer
            from layout import stack_allocation
            from layout.tile_layout import row_major

            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(D_HEAD))
            var H = self.h_heads

            alias q_layout = row_major[D_HEAD]()
            alias k_tile_layout = row_major[BC_FP16, D_HEAD]()
            alias v_tile_layout = row_major[BC_FP16, D_HEAD]()
            alias s_tile_layout = row_major[BC_FP16]()
            alias scalar_layout = row_major[1]()

            def fa_kernel_fp16(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float16, MutAnyOrigin],
                v_ptr: Pointer[Float16, MutAnyOrigin],
                o_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var h = Int(block_idx.y)
                var tid = Int(thread_idx.x)
                var nv = Int(n_ptr[0])
                var inv_sd: Float32 = inv_sqrt_d_ptr[0]

                var Q_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](q_layout)
                var K_tile = stack_allocation[
                    DType.float16, address_space=AddressSpace.SHARED
                ](k_tile_layout)
                var V_tile = stack_allocation[
                    DType.float16, address_space=AddressSpace.SHARED
                ](v_tile_layout)
                var S_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](s_tile_layout)
                var m_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](scalar_layout)

                Q_shared[tid] = q_ptr[h * D_HEAD + tid]
                barrier()

                var m_run: Float32 = NEG_INF
                var l_run: Float32 = 0.0
                var O_acc: Float32 = 0.0

                var num_tiles = (nv + BC_FP16 - 1) // BC_FP16
                for tile_idx in range(num_tiles):
                    var tile_start = tile_idx * BC_FP16
                    for j in range(BC_FP16):
                        var gtok = tile_start + j
                        if gtok < nv:
                            K_tile[j, tid] = k_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                            V_tile[j, tid] = v_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                        else:
                            K_tile[j, tid] = Float16(0.0)
                            V_tile[j, tid] = Float16(0.0)
                    barrier()

                    # Simdgroup-parallel scores: 4 simdgroups × 12 outer iters = 48 scores.
                    var sg_id = tid // 32
                    var lane = tid % 32
                    for outer in range(0, BC_FP16, 4):
                        var score_idx = outer + sg_id
                        var partial: Float32 = 0.0
                        if score_idx < BC_FP16:
                            for d_chunk in range(D_HEAD // 32):
                                var d = lane + d_chunk * 32
                                partial += Q_shared[d] * Float32(K_tile[score_idx, d])
                        var total = warp_sum(partial)
                        if score_idx < BC_FP16 and lane == 0:
                            S_tile[score_idx] = total * inv_sd
                    barrier()

                    if tid == 0:
                        var mx: Float32 = NEG_INF
                        for j in range(BC_FP16):
                            if (tile_start + j) < nv:
                                if S_tile[j] > mx:
                                    mx = S_tile[j]
                        m_shared[0] = mx
                    barrier()

                    var m_tile = m_shared[0]
                    var m_new = m_run
                    if m_tile > m_new:
                        m_new = m_tile
                    var alpha: Float32 = exp(m_run - m_new)

                    O_acc *= alpha
                    var l_tile: Float32 = 0.0
                    for j in range(BC_FP16):
                        if (tile_start + j) < nv:
                            var w = exp(S_tile[j] - m_new)
                            l_tile += w
                            O_acc += w * Float32(V_tile[j, tid])
                    l_run = l_run * alpha + l_tile
                    m_run = m_new
                    barrier()

                o_ptr[h * D_HEAD + tid] = O_acc / l_run

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float16], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float16], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._o_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            ctx[].enqueue_function[fa_kernel_fp16, fa_kernel_fp16](
                q_buf[], k_buf[], v_buf[], o_buf[], n_const, inv_const,
                grid_dim=(1, H, 1), block_dim=(D_HEAD, 1, 1))
            ctx[].synchronize()

            with o_buf[].map_to_host() as h:
                unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            return H * D_HEAD

        return 0


struct MetalFlashAttentionSplitKContext(Movable):
    """Split-K FlashAttention. K_SPLIT threadgroups per head, each over a
    chunk of N tokens, produce partial (m, l, O); a second reduce kernel
    combines them into final O. Goal: fill the GPU (H * K_SPLIT threadgroups
    instead of H) so dispatch-floor amortizes over more useful work.
    """
    var ready: Bool
    var d_head: Int
    var h_heads: Int
    var max_n: Int
    var _ctx: UInt64
    var _q_buf: UInt64       # [H, D_HEAD]
    var _k_buf: UInt64       # [H, max_n, D_HEAD]
    var _v_buf: UInt64       # [H, max_n, D_HEAD]
    var _o_buf: UInt64       # [H, D_HEAD]
    var _pm_buf: UInt64      # [H, K_SPLIT] — partial m
    var _pl_buf: UInt64      # [H, K_SPLIT] — partial l
    var _po_buf: UInt64      # [H, K_SPLIT, D_HEAD] — partial O (un-normalized)

    def __init__(out self):
        self.ready = False
        self.d_head = 0
        self.h_heads = 0
        self.max_n = 0
        self._ctx = 0
        self._q_buf = 0
        self._k_buf = 0
        self._v_buf = 0
        self._o_buf = 0
        self._pm_buf = 0
        self._pl_buf = 0
        self._po_buf = 0

    def init_device(mut self, d_head: Int, h_heads: Int, max_n: Int) raises:
        self.d_head = d_head
        self.h_heads = h_heads
        self.max_n = max_n

        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceContext, DeviceBuffer

            var ctx_ptr = alloc[DeviceContext](1)
            ctx_ptr.unsafe_write(DeviceContext())
            self._ctx = UInt64(Int(ctx_ptr))
            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))

            var q_ptr = alloc[DeviceBuffer[DType.float32]](1)
            q_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._q_buf = UInt64(Int(q_ptr))

            var k_ptr = alloc[DeviceBuffer[DType.float32]](1)
            k_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._k_buf = UInt64(Int(k_ptr))

            var v_ptr = alloc[DeviceBuffer[DType.float32]](1)
            v_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * max_n * d_head))
            self._v_buf = UInt64(Int(v_ptr))

            var o_ptr = alloc[DeviceBuffer[DType.float32]](1)
            o_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * d_head))
            self._o_buf = UInt64(Int(o_ptr))

            var pm_ptr = alloc[DeviceBuffer[DType.float32]](1)
            pm_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * K_SPLIT))
            self._pm_buf = UInt64(Int(pm_ptr))

            var pl_ptr = alloc[DeviceBuffer[DType.float32]](1)
            pl_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * K_SPLIT))
            self._pl_buf = UInt64(Int(pl_ptr))

            var po_ptr = alloc[DeviceBuffer[DType.float32]](1)
            po_ptr.unsafe_write(ctx[].enqueue_create_buffer[DType.float32](h_heads * K_SPLIT * d_head))
            self._po_buf = UInt64(Int(po_ptr))

            self.ready = True

    def upload_kv(mut self,
                  k: Pointer[Float32, MutAnyOrigin],
                  v: Pointer[Float32, MutAnyOrigin],
                  n: Int) raises:
        if not self.ready:
            return
        comptime if CompilationTarget.is_macos():
            from std.gpu.host import DeviceBuffer
            var H = self.h_heads
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            with k_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=k.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)
            with v_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=v.unsafe_bitcast[UInt8](), count=H * n * D_HEAD * 4)

    def attend_q_resident(mut self,
                          q: Pointer[Float32, MutAnyOrigin],
                          n: Int,
                          out_buf: Pointer[Float32, MutAnyOrigin]) raises -> Int:
        if not self.ready or n == 0 or n > self.max_n:
            return 0
        if self.d_head != D_HEAD:
            return 0

        comptime if CompilationTarget.is_macos():
            from std.gpu import block_idx, thread_idx
            from std.gpu.sync import barrier
            from std.gpu.memory import AddressSpace
            from std.gpu.host import DeviceContext, DeviceBuffer
            from layout import stack_allocation
            from layout.tile_layout import row_major

            var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(D_HEAD))
            var H = self.h_heads
            # Each chunk handles ceil(n / K_SPLIT) tokens. Chunk k spans
            # [k * chunk_size, min((k+1) * chunk_size, n)).
            var chunk_size = (n + K_SPLIT - 1) // K_SPLIT

            alias q_layout = row_major[D_HEAD]()
            alias k_tile_layout = row_major[BC, D_HEAD]()
            alias v_tile_layout = row_major[BC, D_HEAD]()
            alias s_tile_layout = row_major[BC]()
            alias scalar_layout = row_major[1]()

            def fa_partial_kernel(
                q_ptr: Pointer[Float32, MutAnyOrigin],
                k_ptr: Pointer[Float32, MutAnyOrigin],
                v_ptr: Pointer[Float32, MutAnyOrigin],
                pm_ptr: Pointer[Float32, MutAnyOrigin],
                pl_ptr: Pointer[Float32, MutAnyOrigin],
                po_ptr: Pointer[Float32, MutAnyOrigin],
                n_ptr: Pointer[Int32, MutAnyOrigin],
                chunk_size_ptr: Pointer[Int32, MutAnyOrigin],
                inv_sqrt_d_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var h = Int(block_idx.y)
                var k_idx = Int(block_idx.x)
                var tid = Int(thread_idx.x)
                var nv = Int(n_ptr[0])
                var cs = Int(chunk_size_ptr[0])
                var inv_sd: Float32 = inv_sqrt_d_ptr[0]

                var chunk_start = k_idx * cs
                var chunk_end = chunk_start + cs
                if chunk_end > nv:
                    chunk_end = nv

                # Empty chunk → write sentinels and exit.
                if chunk_start >= chunk_end:
                    po_ptr[h * K_SPLIT * D_HEAD + k_idx * D_HEAD + tid] = 0.0
                    if tid == 0:
                        pm_ptr[h * K_SPLIT + k_idx] = NEG_INF
                        pl_ptr[h * K_SPLIT + k_idx] = 0.0
                    return

                var Q_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](q_layout)
                var K_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](k_tile_layout)
                var V_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](v_tile_layout)
                var S_tile = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](s_tile_layout)
                var m_shared = stack_allocation[
                    DType.float32, address_space=AddressSpace.SHARED
                ](scalar_layout)

                Q_shared[tid] = q_ptr[h * D_HEAD + tid]
                barrier()

                var m_run: Float32 = NEG_INF
                var l_run: Float32 = 0.0
                var O_acc: Float32 = 0.0

                var num_tiles = (chunk_end - chunk_start + BC - 1) // BC
                for tile_idx in range(num_tiles):
                    var tile_start = chunk_start + tile_idx * BC
                    for j in range(BC):
                        var gtok = tile_start + j
                        if gtok < chunk_end:
                            K_tile[j, tid] = k_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                            V_tile[j, tid] = v_ptr[h * nv * D_HEAD + gtok * D_HEAD + tid]
                        else:
                            K_tile[j, tid] = 0.0
                            V_tile[j, tid] = 0.0
                    barrier()

                    if tid < BC:
                        var score: Float32 = 0.0
                        for d in range(D_HEAD):
                            score += Q_shared[d] * K_tile[tid, d]
                        S_tile[tid] = score * inv_sd
                    barrier()

                    if tid == 0:
                        var mx: Float32 = NEG_INF
                        for j in range(BC):
                            if (tile_start + j) < chunk_end:
                                if S_tile[j] > mx:
                                    mx = S_tile[j]
                        m_shared[0] = mx
                    barrier()

                    var m_tile = m_shared[0]
                    var m_new = m_run
                    if m_tile > m_new:
                        m_new = m_tile
                    var alpha: Float32 = exp(m_run - m_new)

                    O_acc *= alpha
                    var l_tile: Float32 = 0.0
                    for j in range(BC):
                        if (tile_start + j) < chunk_end:
                            var w = exp(S_tile[j] - m_new)
                            l_tile += w
                            O_acc += w * V_tile[j, tid]
                    l_run = l_run * alpha + l_tile
                    m_run = m_new
                    barrier()

                # Write partials (un-normalized O, raw m, raw l).
                po_ptr[h * K_SPLIT * D_HEAD + k_idx * D_HEAD + tid] = O_acc
                if tid == 0:
                    pm_ptr[h * K_SPLIT + k_idx] = m_run
                    pl_ptr[h * K_SPLIT + k_idx] = l_run

            # Reduce kernel: one threadgroup per head; D_HEAD threads;
            # combines K_SPLIT partials via FA's online-softmax merge.
            def fa_reduce_kernel(
                pm_ptr: Pointer[Float32, MutAnyOrigin],
                pl_ptr: Pointer[Float32, MutAnyOrigin],
                po_ptr: Pointer[Float32, MutAnyOrigin],
                o_ptr: Pointer[Float32, MutAnyOrigin],
            ):
                var h = Int(block_idx.y)
                var tid = Int(thread_idx.x)

                var m_global: Float32 = NEG_INF
                for k in range(K_SPLIT):
                    var mm = pm_ptr[h * K_SPLIT + k]
                    if mm > m_global:
                        m_global = mm

                var l_global: Float32 = 0.0
                var O_global: Float32 = 0.0
                for k in range(K_SPLIT):
                    var mm = pm_ptr[h * K_SPLIT + k]
                    var alpha: Float32 = exp(mm - m_global)
                    l_global += pl_ptr[h * K_SPLIT + k] * alpha
                    O_global += po_ptr[h * K_SPLIT * D_HEAD + k * D_HEAD + tid] * alpha

                o_ptr[h * D_HEAD + tid] = O_global / l_global

            var ctx = Pointer[DeviceContext, MutAnyOrigin](unsafe_from_address=Int(self._ctx))
            var q_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._q_buf))
            var k_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._k_buf))
            var v_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._v_buf))
            var o_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._o_buf))
            var pm_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._pm_buf))
            var pl_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._pl_buf))
            var po_buf = Pointer[DeviceBuffer[DType.float32], MutAnyOrigin](unsafe_from_address=Int(self._po_buf))

            with q_buf[].map_to_host() as h:
                unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=q.unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)

            var n_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with n_const.map_to_host() as h:
                h[0] = Int32(n)
            var cs_const = ctx[].enqueue_create_buffer[DType.int32](1)
            with cs_const.map_to_host() as h:
                h[0] = Int32(chunk_size)
            var inv_const = ctx[].enqueue_create_buffer[DType.float32](1)
            with inv_const.map_to_host() as h:
                h[0] = inv_sqrt_d

            ctx[].enqueue_function[fa_partial_kernel, fa_partial_kernel](
                q_buf[], k_buf[], v_buf[],
                pm_buf[], pl_buf[], po_buf[],
                n_const, cs_const, inv_const,
                grid_dim=(K_SPLIT, H, 1), block_dim=(D_HEAD, 1, 1))
            ctx[].enqueue_function[fa_reduce_kernel, fa_reduce_kernel](
                pm_buf[], pl_buf[], po_buf[], o_buf[],
                grid_dim=(1, H, 1), block_dim=(D_HEAD, 1, 1))
            ctx[].synchronize()

            with o_buf[].map_to_host() as h:
                unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=h.unsafe_ptr().unsafe_bitcast[UInt8](), count=H * D_HEAD * 4)
            return H * D_HEAD

        return 0


def cpu_mh_attend_reference(
    q: Pointer[Float32, MutAnyOrigin],   # [H, d]
    k: Pointer[Float32, MutAnyOrigin],   # [H, n, d]
    v: Pointer[Float32, MutAnyOrigin],   # [H, n, d]
    h_heads: Int,
    n: Int,
    d: Int,
    out_buf: Pointer[Float32, MutAnyOrigin],   # [H, d]
):
    var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(d))
    var scores = alloc[Float32](n)
    for h_idx in range(h_heads):
        var q_row = q + h_idx * d
        var k_base = k + h_idx * n * d
        var v_base = v + h_idx * n * d
        var o_row = out_buf + h_idx * d

        for i in range(n):
            var acc: Float32 = 0.0
            var k_row = k_base + i * d
            for j in range(d):
                acc += q_row[j] * k_row[j]
            scores[i] = acc * inv_sqrt_d

        var smax: Float32 = scores[0]
        for i in range(1, n):
            if scores[i] > smax:
                smax = scores[i]
        var ssum: Float32 = 0.0
        for i in range(n):
            scores[i] = exp(scores[i] - smax)
            ssum += scores[i]
        var inv_sum: Float32 = 1.0 / ssum
        for i in range(n):
            scores[i] *= inv_sum

        for j in range(d):
            var acc: Float32 = 0.0
            for i in range(n):
                acc += scores[i] * v_base[i * d + j]
            o_row[j] = acc

    scores.unsafe_free()


def cpu_attend_reference(
    q: Pointer[Float32, MutAnyOrigin],
    k: Pointer[Float32, MutAnyOrigin],
    v: Pointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    out_buf: Pointer[Float32, MutAnyOrigin],
):
    var inv_sqrt_d: Float32 = 1.0 / sqrt(Float32(d))
    var scores = alloc[Float32](n)
    for i in range(n):
        var acc: Float32 = 0.0
        var k_row = k + i * d
        for j in range(d):
            acc += q[j] * k_row[j]
        scores[i] = acc * inv_sqrt_d

    var smax: Float32 = scores[0]
    for i in range(1, n):
        if scores[i] > smax:
            smax = scores[i]
    var ssum: Float32 = 0.0
    for i in range(n):
        scores[i] = exp(scores[i] - smax)
        ssum += scores[i]
    var inv_sum: Float32 = 1.0 / ssum
    for i in range(n):
        scores[i] *= inv_sum

    for j in range(d):
        var acc: Float32 = 0.0
        for i in range(n):
            acc += scores[i] * v[i * d + j]
        out_buf[j] = acc

    scores.unsafe_free()
