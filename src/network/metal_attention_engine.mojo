"""MetalAttentionEngine — in-process Mojo wrapper for the Metal SDPA kernels.

Backs `ATTEND.PREFIX.STORE` / `ATTEND.PREFIX.QUERY` directly from Mojo via
the C FFI in `src/ffi/metal_wrap.m` (kernels `sdpa_q1_fp32` and
`sdpa_batched_q_fp32` in `src/ffi/metal_compute.metal`). No Python sidecar,
no Unix socket, no `--mlx-attention` flag.

End-to-end measured 1.34-1.55× faster than `mlx.fast.scaled_dot_product_attention`
at H=8/N=2048/d=128 (proof: tests/bench_msl_sdpa_q1.m,
tests/bench_pion_metal_attention.py).

Multi-worker (Phase 2 of the retirement roadmap): each worker owns its own
SDPA_SLOTS-entry session cache and Q/O staging buffers; the device, command
queue, PSOs, and shared event are process-global. This struct holds the
caller's worker_id and threads it through every FFI call so the C side can
look up the right per-worker context.

Shape coverage (caller falls back to MLX bridge for everything outside):
  - M ∈ [1, ∞) — M=1 uses the fast `sdpa_q1_fp32`; M>1 uses the batched kernel.
  - D ∈ {64, 96, 128, 160, 192, 256} — kernel function constant per-PSO.
"""

from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.sys.info import CompilationTarget
from src.common.config import PionConfig


@always_inline
def _supported_d(D: Int) -> Bool:
    return D == 32 or D == 64 or D == 96 or D == 128 or D == 160 or D == 192 or D == 256 or D == 512


struct MetalAttentionEngine(Movable):
    """Native Metal SDPA engine, scoped to a single worker.

    Falls back gracefully (`available == False`) when the Metal SDPA
    pipeline can't be initialized (no Metal device, missing metallib).
    """

    var enabled: Bool         # Config flag — caller-requested.
    var available: Bool       # FFI init succeeded.
    var worker_id: UInt32     # Index into the per-worker context array (0..SDPA_MAX_WORKERS-1).
    var fp16: Bool            # Use fp16 kernels (matches vanilla mlx-lm precision).
    var fa_window: UInt32     # Sliding-window size; 0 = full attention.

    def __init__(out self, enabled: Bool, worker_id: Int, fp16: Bool = False, fa_window: Int = 0):
        self.enabled = enabled
        self.available = False
        self.worker_id = UInt32(worker_id)
        self.fp16 = fp16
        self.fa_window = UInt32(fa_window) if fa_window > 0 else UInt32(0)
        # Metal exists only on Apple platforms; on Linux the FFI symbols are
        # not linked at all, so the external_call must be guarded at comptime.
        comptime if CompilationTarget.is_macos():
            if enabled:
                # init is idempotent + thread-safe (mutex-guarded inside C).
                var rc = external_call["pion_metal_sdpa_init", Int32]()
                self.available = rc == 0
                # gh #398: fp16 kernels read half K/V, so the store keeps half.
                # Process-wide and set before any STORE; every worker agrees.
                if self.available and fp16:
                    external_call["pion_metal_sdpa_set_kv_half", NoneType](Int32(1))
                if not self.available:
                    print("[MetalAttn] init failed (rc=", rc, "); ATTEND.PREFIX.* will fall back to MLX bridge")

    def __moveinit__(out self, deinit take: Self):
        self.enabled = take.enabled
        self.available = take.available
        self.worker_id = take.worker_id
        self.fp16 = take.fp16
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
        comptime if CompilationTarget.is_macos():
            var rc = external_call["pion_metal_sdpa_store_kv", Int32](
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
        """Run M=1 attention with cached K/V. Writes H*D floats to `out_ptr`.
        Returns False on miss or error — caller falls back to MLX bridge.

        gh #60 Step 1: `fa_window_override >= 0` overrides the engine-state
        `self.fa_window` for this single call. Lets the consumer pass a
        per-layer window (e.g. 512 for Gemma 4 sliding layers, 0 for full
        layers) without changing the server-wide flag. -1 = use default.
        """
        if not self.available:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H), UInt32(D), prec, window,
                Q_ptr, out_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_sparse_auto(mut self,
                          session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          session_id_len: Int,
                          layer_id: Int, H_q: Int, D: Int,
                          B: Int, K_top: Int,
                          H_kv: Int,
                          Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                          head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          out_ptr: Pointer[Float32, MutUntrackedOrigin],
                          fa_window_override: Int = -1,
                          selector_id: Int = 0) -> Bool:
        """gh #63 Phase 3b: server-side top-K + sparse attention in one FFI call.

        Caller supplies Q + B + K_top; server picks the top-K blocks via the
        selector (block-mean QK scoring v1, Quest upper-bound for selector_id=1
        — W11 Phase 2 / gh #9), runs the sparse kernel against resident K/V,
        returns attention output. Single round-trip, no client-side scoring.

        selector_id:
          0 = block-mean Q·mean(K) (gh #60 / gh #63 default — NIAH-class only)
          1 = Quest upper-bound Σ_d max(Q·K_min, Q·K_max) (W11 Phase 2 — also
              recovers factual-QA on dense corpora)

        Returns False on miss/error.
        """
        if not self.available:
            return False
        if B < 1 or K_top < 1:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query_sparse_auto", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H_q), UInt32(D), prec, window,
                UInt32(B), UInt32(K_top),
                UInt32(H_kv),
                Q_ptr, head_map_ptr, out_ptr,
                UInt32(selector_id),
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_sparse_auto_fused(mut self,
                                session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                                session_id_len: Int,
                                layer_id: Int, H_q: Int, D: Int,
                                B: Int, K_top: Int,
                                H_kv: Int, S_suf: Int,
                                Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                                head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                                K_suf_ptr: Pointer[Float32, MutUntrackedOrigin],
                                V_suf_ptr: Pointer[Float32, MutUntrackedOrigin],
                                out_ptr: Pointer[Float32, MutUntrackedOrigin],
                                fa_window_override: Int = -1,
                                selector_id: Int = 0) -> Bool:
        """gh #63 follow-on: sparse-AUTO + dense-suffix in one fused kernel call.

        Same selector dispatch as query_sparse_auto (selector_id=0 block-mean,
        selector_id=1 Quest UB — W11 Phase 2), plus a dense suffix loop over
        K_suf/V_suf with online-softmax merge across both phases. Returns
        merged attention output — no client-side suffix-merge needed.

        S_suf can be 0 (degenerates to pure sparse_auto).
        """
        if not self.available:
            return False
        if B < 1 or K_top < 1:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query_sparse_auto_fused", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H_q), UInt32(D), prec, window,
                UInt32(B), UInt32(K_top),
                UInt32(H_kv), UInt32(S_suf),
                Q_ptr, head_map_ptr, K_suf_ptr, V_suf_ptr, out_ptr,
                UInt32(selector_id),
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_sparse(mut self,
                     session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                     session_id_len: Int,
                     layer_id: Int, H_q: Int, D: Int,
                     K_sparse_max: Int,
                     H_kv: Int,
                     Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                     indices_ptr: Pointer[Int32, MutUntrackedOrigin],
                     counts_ptr: Pointer[UInt32, MutUntrackedOrigin],
                     head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                     out_ptr: Pointer[Float32, MutUntrackedOrigin],
                     fa_window_override: Int = -1) -> Bool:
        """gh #60 Phase 2: M=1 sparse-mask SDPA — attend only to caller-supplied
        token indices.

        indices_ptr points at an H*K_sparse_max int32 buffer; counts_ptr at an
        H uint32 buffer with the actual per-head K_sparse_h ≤ K_sparse_max.
        Returns False on miss/error (caller can fall back to dense `.query()`).

        Mask selection is the caller's responsibility — this kernel just
        consumes the indices. Combine with `fa_window_override` to layer a
        dense local window on top of sparse global picks.
        """
        if not self.available:
            return False
        if K_sparse_max < 1:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query_sparse", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H_q), UInt32(D), prec, window,
                UInt32(K_sparse_max), UInt32(H_kv),
                Q_ptr, indices_ptr, counts_ptr, head_map_ptr, out_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_batched(mut self,
                      session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      session_id_len: Int,
                      layer_id: Int, H: Int, M: Int, D: Int,
                      Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                      out_ptr: Pointer[Float32, MutUntrackedOrigin],
                      lse_ptr: Pointer[Float32, MutUntrackedOrigin],
                      fa_window_override: Int = -1) -> Bool:
        """Run M>1 batched-Q attention with cached K/V. Writes H*M*D floats to
        `out_ptr` and H*M floats (rowwise log-sum-exp) to `lse_ptr`.

        gh #60 Step 1: per-call `fa_window_override` — see `.query()` docstring.
        """
        if not self.available:
            return False
        if M < 1:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query_batched", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H), UInt32(M), UInt32(D), prec, window,
                Q_ptr, out_ptr, lse_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def query_batched_fused(mut self,
                            session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                            session_id_len: Int,
                            layer_id: Int,
                            H_q: Int, M: Int, D: Int,
                            H_kv: Int, S_suf: Int,
                            Q_ptr: Pointer[Float32, MutUntrackedOrigin],
                            K_suf_ptr: Pointer[Float32, MutUntrackedOrigin],
                            V_suf_ptr: Pointer[Float32, MutUntrackedOrigin],
                            head_map_ptr: Pointer[UInt8, MutUntrackedOrigin],
                            out_ptr: Pointer[Float32, MutUntrackedOrigin],
                            fa_window_override: Int = -1) -> Bool:
        """gh #49: server-side fused suffix-SDPA + prefix merge.

        Runs one online softmax over (cached prefix K/V) ∪ (caller-supplied
        suffix K/V), eliminating the host-side merge in pion-vllm-mlx.
        Writes H_q*M*D floats to `out_ptr` (no LSE — merge already happened).

        H_q ≥ H_kv (GQA); head_map[H_q] = h_kv selects which kv-head each
        q-head reads. S_suf=0 is identical to the legacy batched path
        (no suffix work). K_suf_ptr / V_suf_ptr are unused when S_suf=0.
        """
        if not self.available:
            return False
        if M < 1 or H_q < 1 or H_kv < 1:
            return False
        if (H_q % H_kv) != 0:
            return False
        if not _supported_d(D):
            return False
        comptime if CompilationTarget.is_macos():
            var prec = UInt32(1) if self.fp16 else UInt32(0)
            # gh #60 Step 1: per-call window override (see `.query()` docstring).
            var window = UInt32(fa_window_override) if fa_window_override >= 0 else self.fa_window
            var rc = external_call["pion_metal_sdpa_query_batched_fused", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
                UInt32(H_q), UInt32(M), UInt32(D), prec, window,
                UInt32(H_kv), UInt32(S_suf),
                Q_ptr, K_suf_ptr, V_suf_ptr, head_map_ptr, out_ptr,
            )
            return rc == 0
        else:
            return False

    @always_inline
    def drop(mut self,
             session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
             session_id_len: Int,
             layer_id: Int) -> Bool:
        if not self.available:
            return False
        comptime if CompilationTarget.is_macos():
            var rc = external_call["pion_metal_sdpa_drop", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
            )
            return rc == 0
        else:
            return False

    @always_inline
    def session_exists(mut self,
                       session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                       session_id_len: Int,
                       layer_id: Int) -> Bool:
        """gh #65 follow-on: client-visible existence check on Metal session
        cache. Used by `ATTEND.PREFIX.LOOKUP` so wire-mode consumers can tell
        whether ATTEND.PREFIX.* state is hot independently of the V-store
        (KV.PREFIX.*) registration state.

        gh #67: "exists" means specifically WARM — a session demoted to the
        cold tier reports MISS here. Use `session_state` for tri-state
        (missing / warm / cold) probes.
        """
        if not self.available:
            return False
        comptime if CompilationTarget.is_macos():
            var rc = external_call["pion_metal_sdpa_session_exists", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
            )
            return rc != 0
        else:
            return False

    @always_inline
    def session_state(mut self,
                      session_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      session_id_len: Int,
                      layer_id: Int) -> Int:
        """gh #67: tri-state probe — 0 = missing, 1 = WARM (resident in
        Metal cache), 2 = COLD (demoted to the cold registry; KV.PREFIX.WARM
        can rehydrate from V-store). Used by ATTEND.PREFIX.LOOKUP and the
        auto-rehydrate path.
        """
        if not self.available:
            return 0
        comptime if CompilationTarget.is_macos():
            var rc = external_call["pion_metal_sdpa_session_state", Int32](
                self.worker_id,
                session_id_ptr, UInt32(session_id_len), UInt32(layer_id),
            )
            return Int(rc)
        else:
            return 0

    @always_inline
    def cold_stats(mut self) -> Array[UInt64, 4]:
        """gh #67: cold-tier telemetry — returns [warm_count, cold_count,
        demotions, rehydrates] for this worker. Demotions = lifetime
        WARM→COLD transitions; rehydrates = lifetime COLD→WARM transitions.
        Used by KV.PREFIX.INFO and `tests/test_kv_prefix_cold_tier.py`.
        """
        var out = Array[UInt64, 4](fill=UInt64(0))
        if not self.available:
            return out^
        comptime if CompilationTarget.is_macos():
            var base = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(Pointer(to=out)))
            _ = external_call["pion_metal_sdpa_cold_stats", Int32](
                self.worker_id,
                base,                # warm_count
                base.unsafe_offset(1),            # cold_count
                base.unsafe_offset(2),            # demotions
                base.unsafe_offset(3),            # rehydrates
            )
            return out^
        else:
            return out^
