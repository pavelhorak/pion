// UMMA Phase 1R — GPU-Batched Multi-Head Sparse Attention
// Single Metal dispatch for H attention heads
//
// Layout: all matrices are [H, N, D] or [H, D] with H as the batch dimension.
// Each threadgroup handles one head. Within a threadgroup, threads cooperate
// on GEMV, top-k, softmax, and V gather+multiply.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant uint HEAD_DIM [[function_constant(0)]];    // d (e.g. 64, 128)
constant uint TOP_K    [[function_constant(1)]];     // k (e.g. 32, 64, 128)

// ---------------------------------------------------------------------------
// Kernel 1: Batched GEMV — scores[h][n] = dot(Q[h], K[h][n])
// Grid: (N, H, 1)  Threadgroup: (TG_SIZE, 1, 1)
// Each thread computes one score element (one dot product).
// ---------------------------------------------------------------------------
kernel void batched_gemv(
    device const float*  Q       [[buffer(0)]],   // [H, D]
    device const float*  K       [[buffer(1)]],   // [H, N, D]
    device float*        scores  [[buffer(2)]],   // [H, N]
    constant uint&       N       [[buffer(3)]],
    constant uint&       D       [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]          // (n, h)
) {
    uint h = tid.y;
    uint n = tid.x;
    if (n >= N) return;

    device const float* q_h = Q + h * D;
    device const float* k_hn = K + (h * N + n) * D;

    float acc = 0.0f;
    for (uint j = 0; j < D; j += 4) {
        acc += q_h[j]   * k_hn[j];
        acc += q_h[j+1] * k_hn[j+1];
        acc += q_h[j+2] * k_hn[j+2];
        acc += q_h[j+3] * k_hn[j+3];
    }
    float scale = rsqrt((float)D);
    scores[h * N + n] = acc * scale;
}

// Vectorized float4 variant
kernel void batched_gemv_vec4(
    device const float4* Q       [[buffer(0)]],   // [H, D/4]
    device const float4* K       [[buffer(1)]],   // [H, N, D/4]
    device float*        scores  [[buffer(2)]],   // [H, N]
    constant uint&       N       [[buffer(3)]],
    constant uint&       D4      [[buffer(4)]],   // D / 4
    uint2 tid [[thread_position_in_grid]]
) {
    uint h = tid.y;
    uint n = tid.x;
    if (n >= N) return;

    device const float4* q_h = Q + h * D4;
    device const float4* k_hn = K + (h * N + n) * D4;

    float acc = 0.0f;
    for (uint j = 0; j < D4; j++) {
        float4 q = q_h[j];
        float4 k = k_hn[j];
        acc += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
    }
    float scale = rsqrt((float)(D4 * 4));
    scores[h * N + n] = acc * scale;
}

// ---------------------------------------------------------------------------
// Kernel 2: Batched top-k extraction
// Grid: (H, 1, 1)  Threadgroup: (1, 1, 1)
// Single thread per head — top-k selection is inherently sequential.
// For k <= 64, a simple partial insertion sort beats parallel approaches.
// ---------------------------------------------------------------------------
kernel void batched_topk(
    device const float*  scores     [[buffer(0)]],   // [H, N]
    device uint*         indices    [[buffer(1)]],   // [H, K]
    device float*        topk_vals  [[buffer(2)]],   // [H, K]
    constant uint&       N          [[buffer(3)]],
    constant uint&       K          [[buffer(4)]],   // top-k
    uint tid [[thread_position_in_grid]]              // h
) {
    uint h = tid;
    device const float* s = scores + h * N;
    device uint*  out_idx = indices + h * K;
    device float* out_val = topk_vals + h * K;

    // Initialize with first K elements
    for (uint i = 0; i < K; i++) {
        out_idx[i] = i;
        out_val[i] = s[i];
    }

    // Find minimum in current top-k
    float min_val = out_val[0];
    uint  min_pos = 0;
    for (uint i = 1; i < K; i++) {
        if (out_val[i] < min_val) {
            min_val = out_val[i];
            min_pos = i;
        }
    }

    // Scan remaining elements
    for (uint n = K; n < N; n++) {
        float v = s[n];
        if (v > min_val) {
            out_idx[min_pos] = n;
            out_val[min_pos] = v;
            // Find new minimum
            min_val = out_val[0];
            min_pos = 0;
            for (uint i = 1; i < K; i++) {
                if (out_val[i] < min_val) {
                    min_val = out_val[i];
                    min_pos = i;
                }
            }
        }
    }
}

// Parallel top-k: each thread in the threadgroup scans a stripe of N,
// then we merge per-thread results. Better for large N.
kernel void batched_topk_parallel(
    device const float*  scores     [[buffer(0)]],   // [H, N]
    device uint*         indices    [[buffer(1)]],   // [H, K]
    device float*        topk_vals  [[buffer(2)]],   // [H, K]
    constant uint&       N          [[buffer(3)]],
    constant uint&       K          [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 tpg3 [[threads_per_threadgroup]]
) {
    uint h = gid.x;
    uint lid = lid3.x;
    uint tpg = tpg3.x;
    device const float* s = scores + h * N;

    // Each thread finds local top-K from its stripe
    // Use threadgroup memory for merge
    threadgroup float tg_vals[32 * 128];   // up to 128 threads × 32 top-k
    threadgroup uint  tg_idxs[32 * 128];

    uint local_k = min(K, 32u);  // cap per-thread top-k
    uint offset = lid * local_k;

    // Initialize local top-k
    for (uint i = 0; i < local_k; i++) {
        tg_vals[offset + i] = -1e30f;
        tg_idxs[offset + i] = 0;
    }

    float local_min = -1e30f;
    uint  local_min_pos = 0;

    // Scan stripe
    for (uint n = lid; n < N; n += tpg) {
        float v = s[n];
        if (v > local_min) {
            tg_vals[offset + local_min_pos] = v;
            tg_idxs[offset + local_min_pos] = n;
            // Find new local min
            local_min = tg_vals[offset];
            local_min_pos = 0;
            for (uint i = 1; i < local_k; i++) {
                if (tg_vals[offset + i] < local_min) {
                    local_min = tg_vals[offset + i];
                    local_min_pos = i;
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 merges all local results into global top-K
    if (lid == 0) {
        device uint*  out_idx = indices + h * K;
        device float* out_val = topk_vals + h * K;

        // Initialize output with -inf
        for (uint i = 0; i < K; i++) {
            out_val[i] = -1e30f;
            out_idx[i] = 0;
        }
        float merge_min = -1e30f;
        uint  merge_min_pos = 0;

        // Scan all thread-local top-k results
        uint total_candidates = min(tpg, (uint)128) * local_k;
        for (uint c = 0; c < total_candidates; c++) {
            float v = tg_vals[c];
            if (v > merge_min) {
                out_val[merge_min_pos] = v;
                out_idx[merge_min_pos] = tg_idxs[c];
                merge_min = out_val[0];
                merge_min_pos = 0;
                for (uint i = 1; i < K; i++) {
                    if (out_val[i] < merge_min) {
                        merge_min = out_val[i];
                        merge_min_pos = i;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Batched softmax over top-k scores (in-place)
// Grid: (H, 1, 1)  Threadgroup: (1, 1, 1)
// ---------------------------------------------------------------------------
kernel void batched_softmax(
    device float*        topk_vals [[buffer(0)]],    // [H, K] — modified in place
    constant uint&       K         [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint h = tid;
    device float* v = topk_vals + h * K;

    // Find max
    float mx = v[0];
    for (uint i = 1; i < K; i++) {
        mx = max(mx, v[i]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        float e = exp(v[i] - mx);
        v[i] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < K; i++) {
        v[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Batched V gather + weighted sum
// Grid: (D, H, 1)  Threadgroup: (D_or_TG, 1, 1)
// Each thread computes one dimension of one head's output.
// ---------------------------------------------------------------------------
kernel void batched_gather_multiply(
    device const uint*   indices   [[buffer(0)]],   // [H, K]
    device const float*  weights   [[buffer(1)]],   // [H, K] (softmax output)
    device const float*  V         [[buffer(2)]],   // [H, N, D]
    device float*        output    [[buffer(3)]],   // [H, D]
    constant uint&       N         [[buffer(4)]],
    constant uint&       D         [[buffer(5)]],
    constant uint&       K         [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]            // (d, h)
) {
    uint h = tid.y;
    uint d = tid.x;
    if (d >= D) return;

    device const uint*  idx = indices + h * K;
    device const float* wt  = weights + h * K;
    device const float* v_h = V + h * N * D;

    float acc = 0.0f;
    for (uint i = 0; i < K; i++) {
        acc += wt[i] * v_h[idx[i] * D + d];
    }
    output[h * D + d] = acc;
}

// With threadgroup-cached weights and indices
kernel void batched_gather_multiply_cached(
    device const uint*   indices   [[buffer(0)]],
    device const float*  weights   [[buffer(1)]],
    device const float*  V         [[buffer(2)]],
    device float*        output    [[buffer(3)]],
    constant uint&       N         [[buffer(4)]],
    constant uint&       D         [[buffer(5)]],
    constant uint&       K         [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 tpg3 [[threads_per_threadgroup]]
) {
    uint h = gid.y;
    uint lid = lid3.x;
    uint tpg = tpg3.x;
    uint d = lid;
    if (d >= D) return;

    // Cache weights and indices in threadgroup memory
    threadgroup float tg_wt[512];
    threadgroup uint  tg_idx[512];

    device const uint*  idx = indices + h * K;
    device const float* wt  = weights + h * K;

    // Cooperative load
    for (uint i = lid; i < K; i += tpg) {
        tg_wt[i] = wt[i];
        tg_idx[i] = idx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float* v_h = V + h * N * D;

    float acc = 0.0f;
    for (uint i = 0; i < K; i++) {
        acc += tg_wt[i] * v_h[tg_idx[i] * D + d];
    }
    output[h * D + d] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 5: FUSED batched sparse attention (single dispatch)
// Grid: (1, H, 1)  Threadgroup: (TG_SIZE, 1, 1)
//
// Each threadgroup handles one head completely:
//   Phase A: Cooperative GEMV (scores = Q @ K^T / sqrt(d))
//   Phase B: Thread-0 top-k extraction
//   Phase C: Thread-0 softmax
//   Phase D: Cooperative V gather + weighted sum
//
// This avoids H intermediate buffer round-trips (scores, indices, weights).
// ---------------------------------------------------------------------------
kernel void fused_multihead_sparse_attention(
    device const float*  Q       [[buffer(0)]],   // [H, D]
    device const float*  K       [[buffer(1)]],   // [H, N, D]
    device const float*  V       [[buffer(2)]],   // [H, N, D]
    device float*        output  [[buffer(3)]],   // [H, D]
    constant uint&       N       [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    constant uint&       top_k   [[buffer(6)]],
    uint  gid_h [[threadgroup_position_in_grid]],   // head index
    uint  lid   [[thread_position_in_threadgroup]],
    uint  tpg   [[threads_per_threadgroup]]
) {
    uint h = gid_h;

    device const float* q_h = Q + h * D;
    device const float* k_h = K + h * N * D;
    device const float* v_h = V + h * N * D;

    // M4 threadgroup memory limit: 32KB
    // tg_scores: 4096 * 4B = 16KB, tg_topk: 2 * 512 * 4B = 4KB, tg_weights: 2KB
    // Total: ~22KB — fits in 32KB limit
    threadgroup float tg_scores[4096];
    threadgroup uint  tg_topk_idx[512];
    threadgroup float tg_topk_val[512];
    threadgroup float tg_weights[512];

    uint N_local = min(N, 4096u);

    // --- Phase A: Cooperative GEMV ---
    // Each thread computes scores for a stripe of N
    for (uint n = lid; n < N_local; n += tpg) {
        device const float* k_n = k_h + n * D;
        float acc = 0.0f;
        for (uint j = 0; j < D; j += 4) {
            acc += q_h[j]   * k_n[j];
            acc += q_h[j+1] * k_n[j+1];
            acc += q_h[j+2] * k_n[j+2];
            acc += q_h[j+3] * k_n[j+3];
        }
        tg_scores[n] = acc * rsqrt((float)D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase B: Top-k (thread 0 only — sequential, fast for k<=128) ---
    if (lid == 0) {
        uint K_val = min(top_k, 512u);

        for (uint i = 0; i < K_val; i++) {
            tg_topk_idx[i] = i;
            tg_topk_val[i] = tg_scores[i];
        }

        float min_val = tg_topk_val[0];
        uint  min_pos = 0;
        for (uint i = 1; i < K_val; i++) {
            if (tg_topk_val[i] < min_val) {
                min_val = tg_topk_val[i];
                min_pos = i;
            }
        }

        for (uint n = K_val; n < N_local; n++) {
            float v = tg_scores[n];
            if (v > min_val) {
                tg_topk_idx[min_pos] = n;
                tg_topk_val[min_pos] = v;
                min_val = tg_topk_val[0];
                min_pos = 0;
                for (uint i = 1; i < K_val; i++) {
                    if (tg_topk_val[i] < min_val) {
                        min_val = tg_topk_val[i];
                        min_pos = i;
                    }
                }
            }
        }

        // --- Phase C: Softmax ---
        float mx = tg_topk_val[0];
        for (uint i = 1; i < K_val; i++) mx = max(mx, tg_topk_val[i]);

        float sum = 0.0f;
        for (uint i = 0; i < K_val; i++) {
            float e = exp(tg_topk_val[i] - mx);
            tg_weights[i] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint i = 0; i < K_val; i++) tg_weights[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase D: Cooperative V gather + weighted sum ---
    // Each thread handles a stripe of D dimensions
    uint K_val = min(top_k, 512u);
    for (uint d = lid; d < D; d += tpg) {
        float acc = 0.0f;
        for (uint i = 0; i < K_val; i++) {
            acc += tg_weights[i] * v_h[tg_topk_idx[i] * D + d];
        }
        output[h * D + d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: Fused variant for large N (N > 16K)
// Uses device memory for scores instead of threadgroup shared memory.
// Grid: (1, H, 1)  Threadgroup: (TG_SIZE, 1, 1)
// ---------------------------------------------------------------------------
kernel void fused_multihead_sparse_attention_large(
    device const float*  Q       [[buffer(0)]],   // [H, D]
    device const float*  K       [[buffer(1)]],   // [H, N, D]
    device const float*  V       [[buffer(2)]],   // [H, N, D]
    device float*        output  [[buffer(3)]],   // [H, D]
    device float*        scores  [[buffer(4)]],   // [H, N] scratch
    constant uint&       N       [[buffer(5)]],
    constant uint&       D       [[buffer(6)]],
    constant uint&       top_k   [[buffer(7)]],
    uint  gid_h [[threadgroup_position_in_grid]],
    uint  lid   [[thread_position_in_threadgroup]],
    uint  tpg   [[threads_per_threadgroup]]
) {
    uint h = gid_h;

    device const float* q_h = Q + h * D;
    device const float* k_h = K + h * N * D;
    device const float* v_h = V + h * N * D;
    device float* s_h = scores + h * N;

    threadgroup uint  tg_topk_idx[512];
    threadgroup float tg_topk_val[512];
    threadgroup float tg_weights[512];

    // --- Phase A: Cooperative GEMV to device memory ---
    for (uint n = lid; n < N; n += tpg) {
        device const float* k_n = k_h + n * D;
        float acc = 0.0f;
        for (uint j = 0; j < D; j += 4) {
            acc += q_h[j]   * k_n[j];
            acc += q_h[j+1] * k_n[j+1];
            acc += q_h[j+2] * k_n[j+2];
            acc += q_h[j+3] * k_n[j+3];
        }
        s_h[n] = acc * rsqrt((float)D);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // --- Phase B+C: Top-k + softmax (thread 0) ---
    if (lid == 0) {
        uint K_val = min(top_k, 512u);

        for (uint i = 0; i < K_val; i++) {
            tg_topk_idx[i] = i;
            tg_topk_val[i] = s_h[i];
        }
        float min_val = tg_topk_val[0];
        uint  min_pos = 0;
        for (uint i = 1; i < K_val; i++) {
            if (tg_topk_val[i] < min_val) { min_val = tg_topk_val[i]; min_pos = i; }
        }
        for (uint n = K_val; n < N; n++) {
            float v = s_h[n];
            if (v > min_val) {
                tg_topk_idx[min_pos] = n;
                tg_topk_val[min_pos] = v;
                min_val = tg_topk_val[0]; min_pos = 0;
                for (uint i = 1; i < K_val; i++) {
                    if (tg_topk_val[i] < min_val) { min_val = tg_topk_val[i]; min_pos = i; }
                }
            }
        }

        float mx = tg_topk_val[0];
        for (uint i = 1; i < K_val; i++) mx = max(mx, tg_topk_val[i]);
        float sum = 0.0f;
        for (uint i = 0; i < K_val; i++) {
            float e = exp(tg_topk_val[i] - mx);
            tg_weights[i] = e;
            sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (uint i = 0; i < K_val; i++) tg_weights[i] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase D: Cooperative V gather ---
    uint K_val = min(top_k, 512u);
    for (uint d = lid; d < D; d += tpg) {
        float acc = 0.0f;
        for (uint i = 0; i < K_val; i++) {
            acc += tg_weights[i] * v_h[tg_topk_idx[i] * D + d];
        }
        output[h * D + d] = acc;
    }
}
