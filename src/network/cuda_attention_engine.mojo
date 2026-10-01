"""CudaAttentionEngine — in-process Mojo wrapper for the CUDA SDPA kernels.

Mirror of `src/network/metal_attention_engine.mojo` for the Linux/CUDA port
(gh #9). Same struct shape, same method signatures, same fall-back behavior;
only differences are:
  - external_call symbols are `pion_cuda_sdpa_*` instead of `pion_metal_sdpa_*`
  - guard switches from `is_macos()` to `is_linux()`
  - C side lives in `src/ffi/cuda_wrap.c`, kernels move to
    `src/ffi/cuda_kernels.cu` when fully integrated.

End-to-end correctness gate: 32/32 dense + 36/36 sparse cases pass through
the C ABI vs PyTorch CPU SDPA reference (max abs err ≤ 9e-8).

Shape coverage matches the kernel PSO grid:
  - M = 1 single-query SDPA (decode-step path)
  - D ∈ {32, 64, 96, 128, 160, 192, 256, 512}
  - H_q / H_kv: GQA via head_map[H_q] (NULL = identity)
  - W_window: sliding-window clamp to last W tokens; 0 = full attention

Integration is not yet wired into the pion-server runtime; it needs
config.mojo + main.mojo + state.mojo + attend.mojo patches.
"""

from src.common.ptr import null_ptr
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.sys.info import CompilationTarget
from std.sys import is_defined
from src.common.config import PionConfig


# gh #83: CUDA support is opt-in at build time via `-D PION_CUDA`. A stock
# Linux build (no GPU, no nvcc/cudart — e.g. CI runners, OSS newcomers)
# omits the define, so every `pion_cuda_sdpa_*` external_call below is
# comptime-elided and the binary links without libcudart. The production
# Linux build in pixi.toml passes `-D PION_CUDA` and links the CUDA objects.
comptime CUDA_BUILD = CompilationTarget.is_linux() and is_defined["PION_CUDA"]()


@always_inline
def _supported_d(D: Int) -> Bool:
    return D == 32 or D == 64 or D == 96 or D == 128 or D == 160 or D == 192 or D == 256 or D == 512


struct CudaAttentionEngine(Movable):
    """Native CUDA SDPA engine, scoped to a single worker.

    Falls back gracefully (`available == False`) when:
      - cudaGetDeviceCount returns 0 (no GPU on this host)
      - libpion_cuda_attention.so isn't loaded
      - the Mojo binary was built without CUDA support (Mac, aarch64 default)
    """

    var enabled: Bool        # config flag — caller-requested
    var available: Bool      # FFI init succeeded (CUDA present + libs loaded)
    var worker_id: UInt32    # index into the per-worker context array (0..15)
    var fa_window: UInt32    # sliding-window size; 0 = full attention

    def __init__(out self, enabled: Bool, worker_id: Int, fa_window: Int = 0):
        self.enabled = enabled
        self.available = False
        self.worker_id = UInt32(worker_id)
        self.fa_window = UInt32(fa_window) if fa_window > 0 else UInt32(0)
        # CUDA exists only on Linux; on macOS/aarch64 the FFI symbols are
        # not linked, so the external_call must be guarded at comptime.
        comptime if CUDA_BUILD:
            if enabled:
                # init is idempotent + thread-safe (mutex-guarded inside C).
                var rc = external_call["pion_cuda_sdpa_init", Int32]()
                self.available = rc == 0
                if not self.available:
                    print("[CudaAttn] init failed (rc=", rc, "); ATTEND.PREFIX.* will fall back")

    def __moveinit__(out self, deinit take: Self):
        self.enabled = take.enabled
        self.available = take.available
        self.worker_id = take.worker_id
        self.fa_window = take.fa_window

    @always_inline
    def store_kv(mut self,
                 session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                 session_id_len: Int,
                 layer_id: Int, H: Int, N: Int, D: Int,
                 K_ptr: Pointer[Float32, MutUntrackedOrigin],
                 V_ptr: Pointer[Float32, MutUntrackedOrigin]) -> Bool:
        """Cache K/V for (session_id, layer_id) in this worker's session cache.
        K/V layout: `[H, N, D]` row-major float32.
        """
        if not self.available:
            return False
        if not _supported_d(D):
            return False
        comptime if CUDA_BUILD:
            var rc = external_call["pion_cuda_sdpa_store_kv", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H), UInt32(N), UInt32(D),
                K_ptr, V_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query(mut self,
              session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
              session_id_len: Int,
              layer_id: Int, H: Int, D: Int,
              Q_ptr: Pointer[Float32, MutUntrackedOrigin],
              out_ptr: Pointer[Float32, MutUntrackedOrigin],
              fa_window_override: Int = -1) -> Bool:
        """Run M=1 dense attention with cached K/V. Writes H*D floats to `out_ptr`.
        Returns False on miss / shape mismatch / kernel error.

        `fa_window_override >= 0` overrides the engine-state `self.fa_window`
        for this single call (matches the gh #60 Step 1 per-call window
        override on the Metal side).
        """
        if not self.available:
            return False
        if not _supported_d(D):
            return False
        comptime if CUDA_BUILD:
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_cuda_sdpa_query", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H), UInt32(D),
                Q_ptr, out_ptr, window,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_sparse(mut self,
                     session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                     session_id_len: Int,
                     layer_id: Int, H: Int, D: Int,
                     Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                     out_ptr: Pointer[Float32, MutUntrackedOrigin],
                     indices_ptr: Pointer[Int32, MutUntrackedOrigin],
                     counts_ptr: Pointer[UInt32, MutUntrackedOrigin],
                     K_sparse_max: Int,
                     head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                     fa_window_override: Int = -1) -> Bool:
        """Run M=1 sparse-mask attention with caller-supplied per-head index list.
        Writes H*D floats to `out_ptr`. Returns False on miss / shape mismatch /
        kernel error.

        Layouts (host-side, row-major):
          indices: [H, K_sparse_max] Int32
          counts:  [H] UInt32
          head_map: [H] UInt8 (NULL = identity / non-GQA)

        This is the kernel that flips the gh #15 cloud-GPU TTFT verdict.
        Bench-level: 56× faster than the dense kernel at H=8/D=128/N=26K
        with K_blocks=4 (256 sparse tokens). Same family as gh #60's
        Mac result (326× warm-TTFT on Gemma-4-E2B at 64K, 0.78% prefix).
        """
        if not self.available:
            return False
        if not _supported_d(D):
            return False
        comptime if CUDA_BUILD:
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_cuda_sdpa_query_sparse", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H), UInt32(D),
                Q_ptr, out_ptr, window,
                indices_ptr, counts_ptr, UInt32(K_sparse_max),
                head_map_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def drop(mut self,
             session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
             session_id_len: Int) -> Int:
        """Release all cached layers for this session. Returns dropped slot count.

        First-pass implementation tombstones every non-empty slot in the worker's
        table (the C side doesn't yet remember per-slot sid for selective drop).
        Caller (ATTEND.PREFIX.DROP) typically drops at end-of-conversation, so
        per-worker bulk-drop is acceptable; per-session selectivity is a v2
        follow-up tracked in INTEGRATION_SCOPE.md.
        """
        if not self.available:
            return 0
        comptime if CUDA_BUILD:
            var rc = external_call["pion_cuda_sdpa_drop", Int32](
                self.worker_id, session_id_ptr, UInt32(session_id_len))
            return Int(rc)
        else:
            return 0

    @always_inline
    def session_exists(mut self,
                       session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                       session_id_len: Int) -> Bool:
        if not self.available:
            return False
        comptime if CUDA_BUILD:
            var rc = external_call["pion_cuda_sdpa_session_exists", Int32](
                self.worker_id, session_id_ptr, UInt32(session_id_len))
            return rc != 0
        else:
            return False

    @always_inline
    def query_sparse_auto(mut self,
                          session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          session_id_len: Int,
                          layer_id: Int, H_q: Int, D: Int,
                          Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                          out_ptr: Pointer[Float32, MutUntrackedOrigin],
                          K_block: Int, K_blocks: Int,
                          head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          fa_window_override: Int = -1) -> Bool:
        """Server-side block-mean K + top-K block selection + sparse SDPA.

        gh #9 item 6: full ATTEND.PREFIX.QUERY_SPARSE_AUTO end-to-end. The
        block-mean-K + dot(Q) selection happens entirely on-device (no
        host roundtrip for indices), then the picked indices feed straight
        into sdpa_q1_sparse_fp32 via the same per-worker stream. Caller
        only supplies Q + K_block + K_blocks; selection algorithm matches
        Mac-side `make_pion_prompt_cache(sparse_full_layers={...})`.

        Writes H_q*D floats to `out_ptr`. Returns False on miss / error.
        """
        if not self.available:
            return False
        if not _supported_d(D):
            return False
        comptime if CUDA_BUILD:
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var null_idx = null_ptr[Int32, MutUntrackedOrigin]()
            var null_cnt = null_ptr[UInt32, MutUntrackedOrigin]()
            var rc = external_call["pion_cuda_sdpa_query_sparse_auto", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H_q), UInt32(D),
                Q_ptr, out_ptr, window,
                UInt32(K_block), UInt32(K_blocks),
                head_map_ptr,
                null_idx,   # don't return picked indices to caller
                null_cnt,
            )
            return rc == 0
        else:
            return False
