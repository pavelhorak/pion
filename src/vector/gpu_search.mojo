# §7v2 GPU Vector Engine — STUBBED at the Mojo 1.0 migration.
#
# The native std.gpu kernel path this file used to hold moved to the `max`
# package in Mojo 1.0, which the server build must not depend on. Per
# memory/max_stdgpu_spike_2026_05_15 the native path also LOSES to the
# shipping Metal FFI path (src/ffi/metal_wrap.m — pion_metal_* calls), and
# nothing ever called init_device() in production, so `ready` was always
# False and every caller took its fallback. The API is preserved so
# hnsw.mojo's call sites compile unchanged; the pre-1.0 implementation is in
# git history if the experiment is ever revived (it would then live behind a
# max-gated build variant, not the default server build).

from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc

comptime GPU_BLOCK_SIZE: Int = 256
comptime DIM_1536: Int = 1536
# FP32 rerank: minimum candidate count to dispatch the Metal gather kernel.
# Below this, the CPU SIMD loop wins on dispatch overhead. Tuned for M4 — at
# K=64 with dim=1536, GPU dispatch (~50µs) breaks even with CPU SIMD (~10µs/64).
comptime GPU_RERANK_THRESHOLD: Int = 64
# Matches PION_RERANK_K_MAX in src/ffi/metal_wrap.m
comptime GPU_RERANK_K_MAX: Int = 2048


struct GPUSearchContext(RegisterPassable, Movable):
    var num_vectors: Int
    var stride: Int           # bytes per compact slot (set at register_vectors)
    var ready: Bool           # always False in the stub — callers take their fallback

    def __init__(out self):
        self.num_vectors = 0
        self.stride = 0
        self.ready = False

    def init_device(mut self) raises:
        pass

    def register_vectors(mut self, compact_buffer: Pointer[Int8, MutUntrackedOrigin],
                        num_vectors: Int, stride: Int) raises:
        pass

    def search(mut self, query_int8: Pointer[Int8, MutUntrackedOrigin],
              query_norm_sq: Float32, k: Int,
              out_ids: Pointer[Int32, MutUntrackedOrigin],
              out_dists: Pointer[Float32, MutUntrackedOrigin]) raises -> Int:
        return 0
