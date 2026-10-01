// gh #9 — first kernel port: sdpa_q1_fp32 (Metal MSL → CUDA).
//
// Single-query (M=1) scaled-dot-product attention for the
// ATTEND.PREFIX.QUERY decode-step path. CUDA equivalent of
// `sdpa_q1_fp32` in src/ffi/metal_compute.metal.
//
// Layout (float4-packed, stride D4 = D/4):
//   Q [H,    D4], K [H, N, D4], V [H, N, D4], O [H, D4]
//
// Translation map MSL → CUDA:
//   simdgroup (32 threads)            → warp (32 threads)
//   simd_sum                          → __shfl_xor_sync butterfly reduction
//   fast::exp                         → __expf
//   threadgroup memory                → __shared__ (dynamic via extern smem)
//   threadgroup_position_in_grid      → blockIdx.x
//   simdgroup_index_in_threadgroup    → threadIdx.x / 32
//   thread_index_in_simdgroup         → threadIdx.x % 32
//   function_constant D_HEAD          → runtime kernel arg (no PSO specialization
//                                       in this prototype; CHUNK computed at runtime)
//   threadgroup_barrier(mem_threadgroup) → __syncthreads
//
// Grid:   H blocks
// Block:  SG_PER_TG * 32 = 256 threads (8 warps)
//
// Numerical equivalence vs Metal: the order of the per-token softmax updates
// is identical (online softmax over the same per-warp slice of t in [t_begin,
// t_end)) and the per-warp reduction across 32 lanes uses the same butterfly
// structure as Metal's simd_sum. fp32 reductions are not bit-equivalent across
// vendors due to ULP differences in __expf vs Metal fast::exp, but max-abs
// error against a CPU fp32 reference should land within ~1e-3 in single
// precision.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define SG_PER_TG    8u
#define WARP_SIZE    32u
#define CHUNK_MAX    4u   // bounds register arrays at compile time; supports D ≤ 512

// Warp-level sum reduction across 32 lanes via butterfly.
// Equivalent to Metal's `simd_sum`.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// Shared memory layout (set via kernel-launch smem-bytes argument):
//   float  s_m[SG_PER_TG]                              // per-warp running max
//   float  s_l[SG_PER_TG]                              // per-warp running denominator
//   float4 s_o[SG_PER_TG * CHUNK * WARP_SIZE]          // per-warp output rows
// Total bytes = 8 + 8 + 8*CHUNK*32*16 + alignment slack.
//
// One block per head h. D_HEAD passed at runtime; CHUNK = ceil(D4/32).
extern "C" __global__
void sdpa_q1_fp32(
    const float4* __restrict__ Q,        // [H, D4]
    const float4* __restrict__ K,        // [H, N, D4]
    const float4* __restrict__ V,        // [H, N, D4]
    float4*       __restrict__ O,        // [H, D4]
    uint32_t                   N_tokens,
    float                      scale,
    uint32_t                   W_window,  // 0 = full attention; >0 = scan only last W tokens
    uint32_t                   D_HEAD)
{
    extern __shared__ unsigned char smem_raw[];
    float*  s_m = reinterpret_cast<float*>(smem_raw);
    float*  s_l = s_m + SG_PER_TG;
    // Align s_o to 16B boundary after the floats.
    uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_l + SG_PER_TG);
    s_o_addr = (s_o_addr + 15) & ~uintptr_t(15);
    float4* s_o = reinterpret_cast<float4*>(s_o_addr);

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    const uint32_t h    = blockIdx.x;
    const uint32_t sg   = threadIdx.x / WARP_SIZE;     // simdgroup index in threadgroup
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    // Load this lane's slice of Q (CHUNK float4 chunks; idle lanes contribute 0).
    float4 q[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) q[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        uint32_t idx = c * WARP_SIZE + lane;
        if (idx < D4) q[c] = Q[h * D4 + idx];
    }

    // Sliding-window: scan only the last W_window tokens. W=0 → full attention.
    uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint32_t eff_n     = N_tokens - eff_start;
    uint32_t chunk_n   = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint32_t t_begin   = eff_start + sg * chunk_n;
    uint32_t t_end     = min(t_begin + chunk_n, N_tokens);

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) o[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4* k_base = K + (size_t)h * N_tokens * D4;
    const float4* v_base = V + (size_t)h * N_tokens * D4;

    for (uint32_t t = t_begin; t < t_end; ++t) {
        // QK dot — each lane contributes its share, warp_sum reduces across the 32 lanes.
        float partial = 0.0f;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 k = k_base[(size_t)t * D4 + idx];
                partial += q[c].x * k.x + q[c].y * k.y + q[c].z * k.z + q[c].w * k.w;
            }
        }
        float score = warp_sum(partial) * scale;

        float new_m  = fmaxf(m, score);
        float factor = __expf(m - new_m);
        float exp_s  = __expf(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        // V update — each lane handles its own chunks.
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 v = v_base[(size_t)t * D4 + idx];
                o[c].x = o[c].x * factor + exp_s * v.x;
                o[c].y = o[c].y * factor + exp_s * v.y;
                o[c].z = o[c].z * factor + exp_s * v.z;
                o[c].w = o[c].w * factor + exp_s * v.w;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    // Stash this warp's per-lane O into shared mem so warp 0 can merge.
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        s_o[sg * CHUNK * WARP_SIZE + c * WARP_SIZE + lane] = o[c];
    }
    __syncthreads();

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = fmaxf(M, m_i);
            float a = __expf(M   - new_M);
            float b = __expf(m_i - new_M);
            for (uint32_t c = 0u; c < CHUNK; ++c) {
                float4 part = s_o[i * CHUNK * WARP_SIZE + c * WARP_SIZE + lane];
                O_acc[c].x = O_acc[c].x * a + part.x * b;
                O_acc[c].y = O_acc[c].y * a + part.y * b;
                O_acc[c].z = O_acc[c].z * a + part.z * b;
                O_acc[c].w = O_acc[c].w * a + part.w * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 out;
                float inv_L = 1.0f / L;
                out.x = O_acc[c].x * inv_L;
                out.y = O_acc[c].y * inv_L;
                out.z = O_acc[c].z * inv_L;
                out.w = O_acc[c].w * inv_L;
                O[h * D4 + idx] = out;
            }
        }
    }
}

// C ABI host launcher — call signature mirrors what a future Mojo bridge
// would invoke. Returns 0 on success, negative on cuda error.
//
// Q, K, V, O are device pointers (caller manages H2D / D2H).
extern "C" int pion_sdpa_q1_fp32_cuda(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    uint32_t     H,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    float        scale,
    uint32_t     W_window,
    cudaStream_t stream)
{
    if (D_HEAD == 0 || (D_HEAD & 3u) != 0u) return -1;            // must be multiple of 4
    if (D_HEAD > 4u * CHUNK_MAX * WARP_SIZE) return -2;           // D > 512 unsupported
    if (H == 0u || N_tokens == 0u) return -3;

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    // Shared memory: 2 * SG_PER_TG floats (s_m, s_l) + alignment slack
    // + SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4) for s_o.
    size_t smem_bytes = 2 * SG_PER_TG * sizeof(float)
                      + 16   // alignment slack to next 16B boundary
                      + (size_t)SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4);

    dim3 grid(H);
    dim3 block(SG_PER_TG * WARP_SIZE);

    sdpa_q1_fp32<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float4*>(d_Q),
        reinterpret_cast<const float4*>(d_K),
        reinterpret_cast<const float4*>(d_V),
        reinterpret_cast<float4*>(d_O),
        N_tokens, scale, W_window, D_HEAD);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sdpa_q1_fp32 launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// PSO-specialized variant: D_HEAD is a template parameter so CHUNK becomes
// constexpr; nvcc fully unrolls the register arrays and CHUNK loops, and at
// D ≤ 128 the `idx < D4` bounds check is dead-code-eliminated entirely
// (idle lanes were already loading 0). Mirrors the MSL function-constant
// PSO grid in src/ffi/metal_compute.metal: D ∈ {32, 64, 96, 128, 160, 192,
// 256, 512}.
// ─────────────────────────────────────────────────────────────────────────────

template<unsigned int D_HEAD_T>
__global__
void sdpa_q1_fp32_pso(
    const float4* __restrict__ Q,
    const float4* __restrict__ K,
    const float4* __restrict__ V,
    float4*       __restrict__ O,
    uint32_t                   N_tokens,
    float                      scale,
    uint32_t                   W_window)
{
    extern __shared__ unsigned char smem_raw[];
    float*  s_m = reinterpret_cast<float*>(smem_raw);
    float*  s_l = s_m + SG_PER_TG;
    uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_l + SG_PER_TG);
    s_o_addr = (s_o_addr + 15) & ~uintptr_t(15);
    float4* s_o = reinterpret_cast<float4*>(s_o_addr);

    constexpr uint32_t D4_T    = D_HEAD_T / 4u;
    constexpr uint32_t CHUNK_T = (D4_T + 31u) / 32u;

    const uint32_t h    = blockIdx.x;
    const uint32_t sg   = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    float4 q[CHUNK_T];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_T; ++c) {
        uint32_t idx = c * WARP_SIZE + lane;
        if (idx < D4_T) q[c] = Q[h * D4_T + idx];
        else            q[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint32_t eff_n     = N_tokens - eff_start;
    uint32_t chunk_n   = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint32_t t_begin   = eff_start + sg * chunk_n;
    uint32_t t_end     = min(t_begin + chunk_n, N_tokens);

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_T];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_T; ++c) o[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4* k_base = K + (size_t)h * N_tokens * D4_T;
    const float4* v_base = V + (size_t)h * N_tokens * D4_T;

    for (uint32_t t = t_begin; t < t_end; ++t) {
        float partial = 0.0f;
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_T; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4_T) {
                float4 k = k_base[(size_t)t * D4_T + idx];
                partial += q[c].x * k.x + q[c].y * k.y + q[c].z * k.z + q[c].w * k.w;
            }
        }
        float score = warp_sum(partial) * scale;

        float new_m  = fmaxf(m, score);
        float factor = __expf(m - new_m);
        float exp_s  = __expf(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_T; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4_T) {
                float4 v = v_base[(size_t)t * D4_T + idx];
                o[c].x = o[c].x * factor + exp_s * v.x;
                o[c].y = o[c].y * factor + exp_s * v.y;
                o[c].z = o[c].z * factor + exp_s * v.z;
                o[c].w = o[c].w * factor + exp_s * v.w;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_T; ++c) {
        s_o[sg * CHUNK_T * WARP_SIZE + c * WARP_SIZE + lane] = o[c];
    }
    __syncthreads();

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_T];
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_T; ++c) O_acc[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = fmaxf(M, m_i);
            float a = __expf(M   - new_M);
            float b = __expf(m_i - new_M);
            #pragma unroll
            for (uint32_t c = 0u; c < CHUNK_T; ++c) {
                float4 part = s_o[i * CHUNK_T * WARP_SIZE + c * WARP_SIZE + lane];
                O_acc[c].x = O_acc[c].x * a + part.x * b;
                O_acc[c].y = O_acc[c].y * a + part.y * b;
                O_acc[c].z = O_acc[c].z * a + part.z * b;
                O_acc[c].w = O_acc[c].w * a + part.w * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        float inv_L = 1.0f / L;
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_T; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4_T) {
                float4 out;
                out.x = O_acc[c].x * inv_L;
                out.y = O_acc[c].y * inv_L;
                out.z = O_acc[c].z * inv_L;
                out.w = O_acc[c].w * inv_L;
                O[h * D4_T + idx] = out;
            }
        }
    }
}

// Explicit instantiation of the PSO grid (mirror of MSL function-constant set).
template __global__ void sdpa_q1_fp32_pso< 32>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso< 64>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso< 96>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso<128>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso<160>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso<192>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso<256>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);
template __global__ void sdpa_q1_fp32_pso<512>(const float4*, const float4*, const float4*, float4*, uint32_t, float, uint32_t);

// PSO dispatcher — same signature as the runtime-D launcher except that the
// kernel selected has D_HEAD baked in at compile time. Returns -4 if D_HEAD
// is not in the supported set (compile-time bake = no other choice).
extern "C" int pion_sdpa_q1_fp32_cuda_pso(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    uint32_t     H,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    float        scale,
    uint32_t     W_window,
    cudaStream_t stream)
{
    if (H == 0u || N_tokens == 0u) return -3;

    auto launch = [&](auto kernel, uint32_t D_compile) -> int {
        const uint32_t CHUNK = (D_compile / 4u + 31u) / 32u;
        size_t smem_bytes = 2 * SG_PER_TG * sizeof(float)
                          + 16
                          + (size_t)SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4);
        dim3 grid(H);
        dim3 block(SG_PER_TG * WARP_SIZE);
        kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const float4*>(d_Q),
            reinterpret_cast<const float4*>(d_K),
            reinterpret_cast<const float4*>(d_V),
            reinterpret_cast<float4*>(d_O),
            N_tokens, scale, W_window);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "sdpa_q1_fp32_pso<%u> launch failed: %s\n",
                    D_compile, cudaGetErrorString(err));
            return -10;
        }
        return 0;
    };

    switch (D_HEAD) {
        case  32: return launch(sdpa_q1_fp32_pso< 32>,  32);
        case  64: return launch(sdpa_q1_fp32_pso< 64>,  64);
        case  96: return launch(sdpa_q1_fp32_pso< 96>,  96);
        case 128: return launch(sdpa_q1_fp32_pso<128>, 128);
        case 160: return launch(sdpa_q1_fp32_pso<160>, 160);
        case 192: return launch(sdpa_q1_fp32_pso<192>, 192);
        case 256: return launch(sdpa_q1_fp32_pso<256>, 256);
        case 512: return launch(sdpa_q1_fp32_pso<512>, 512);
        default:  return -4;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse-mask SDPA Q=1 — port of `sdpa_q1_sparse_fp32` from
// src/ffi/metal_compute.metal:439. Same algorithmic shape as the dense kernel
// above; deltas are:
//   - inner loop iterates over caller-supplied indices[h, 0..counts[h]-1]
//     instead of [eff_start..N_tokens)
//   - GQA: query head h reads K/V from KV head head_map[h]
//   - sliding-window combines: indices below (N_tokens - W_window) are skipped
//     so caller can dispatch (sparse global picks + dense local window) in one call
//
// Compute saving when K_sparse << N: kernel does ~K_sparse/N of the FLOPs/DRAM
// of the dense path. This is the kernel that, on Mac, hit 326x warm-TTFT vs
// vanilla on Gemma-4-E2B at 64K with 0.78% prefix budget. Bringing it to CUDA
// is the load-bearing move for the cloud-GPU TTFT pitch (per gh #15
// corpus-sweep follow-up: dense CAG TTFT scales linearly with N; sparse
// breaks that scaling).
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __global__
void sdpa_q1_sparse_fp32(
    const float4* __restrict__ Q,           // [H_q, D4]
    const float4* __restrict__ K,           // [H_kv, N, D4]
    const float4* __restrict__ V,           // [H_kv, N, D4]
    float4*       __restrict__ O,           // [H_q, D4]
    uint32_t                   N_tokens,
    float                      scale,
    uint32_t                   W_window,    // 0 = no clamp; >0 = mask indices < (N - W)
    const int* __restrict__    indices,     // [H_q, K_sparse_max]
    const uint32_t* __restrict__ counts,    // [H_q]
    uint32_t                   K_sparse_max,
    const unsigned char* __restrict__ head_map, // [H_q] = h_kv per query head; nullable
    uint32_t                   D_HEAD)
{
    extern __shared__ unsigned char smem_raw[];
    float*  s_m = reinterpret_cast<float*>(smem_raw);
    float*  s_l = s_m + SG_PER_TG;
    uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_l + SG_PER_TG);
    s_o_addr = (s_o_addr + 15) & ~uintptr_t(15);
    float4* s_o = reinterpret_cast<float4*>(s_o_addr);

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    const uint32_t h    = blockIdx.x;        // query head index 0..H_q-1
    const uint32_t sg   = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    // GQA mapping: head_map[h] = h_kv. NULL → identity (h_kv = h).
    uint32_t h_kv = head_map ? (uint32_t)head_map[h] : h;

    float4 q[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) q[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        uint32_t idx = c * WARP_SIZE + lane;
        if (idx < D4) q[c] = Q[h * D4 + idx];
    }

    // Distribute this head's index slice across SG_PER_TG simdgroups.
    uint32_t K_sparse_h = counts[h];
    uint32_t per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint32_t i_begin    = sg * per_sg;
    uint32_t i_end      = min(i_begin + per_sg, K_sparse_h);

    uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) o[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float4* k_base   = K + (size_t)h_kv * N_tokens * D4;
    const float4* v_base   = V + (size_t)h_kv * N_tokens * D4;
    const int*    idx_base = indices + (size_t)h * K_sparse_max;

    for (uint32_t i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint32_t)idx_i >= N_tokens) continue;
        uint32_t t = (uint32_t)idx_i;
        if (t < eff_start) continue;

        float partial = 0.0f;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 k = k_base[(size_t)t * D4 + idx];
                partial += q[c].x * k.x + q[c].y * k.y + q[c].z * k.z + q[c].w * k.w;
            }
        }
        float score = warp_sum(partial) * scale;

        float new_m  = fmaxf(m, score);
        float factor = __expf(m - new_m);
        float exp_s  = __expf(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 v = v_base[(size_t)t * D4 + idx];
                o[c].x = o[c].x * factor + exp_s * v.x;
                o[c].y = o[c].y * factor + exp_s * v.y;
                o[c].z = o[c].z * factor + exp_s * v.z;
                o[c].w = o[c].w * factor + exp_s * v.w;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        s_o[sg * CHUNK * WARP_SIZE + c * WARP_SIZE + lane] = o[c];
    }
    __syncthreads();

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = fmaxf(M, m_i);
            float a = __expf(M   - new_M);
            float b = __expf(m_i - new_M);
            for (uint32_t c = 0u; c < CHUNK; ++c) {
                float4 part = s_o[i * CHUNK * WARP_SIZE + c * WARP_SIZE + lane];
                O_acc[c].x = O_acc[c].x * a + part.x * b;
                O_acc[c].y = O_acc[c].y * a + part.y * b;
                O_acc[c].z = O_acc[c].z * a + part.z * b;
                O_acc[c].w = O_acc[c].w * a + part.w * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        float inv_L = 1.0f / L;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                float4 out;
                out.x = O_acc[c].x * inv_L;
                out.y = O_acc[c].y * inv_L;
                out.z = O_acc[c].z * inv_L;
                out.w = O_acc[c].w * inv_L;
                O[h * D4 + idx] = out;
            }
        }
    }
}

extern "C" int pion_sdpa_q1_sparse_fp32_cuda(
    const float* d_Q,                       // [H_q, D]
    const float* d_K,                       // [H_kv, N, D]
    const float* d_V,                       // [H_kv, N, D]
    float*       d_O,                       // [H_q, D]
    uint32_t     H_q,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    float        scale,
    uint32_t     W_window,
    const int*   d_indices,                 // [H_q, K_sparse_max]
    const uint32_t* d_counts,               // [H_q]
    uint32_t     K_sparse_max,
    const unsigned char* d_head_map,        // [H_q]; NULL → identity
    cudaStream_t stream)
{
    if (D_HEAD == 0 || (D_HEAD & 3u) != 0u) return -1;
    if (D_HEAD > 4u * CHUNK_MAX * WARP_SIZE) return -2;
    if (H_q == 0u || N_tokens == 0u || K_sparse_max == 0u) return -3;

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    size_t smem_bytes = 2 * SG_PER_TG * sizeof(float)
                      + 16
                      + (size_t)SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4);

    dim3 grid(H_q);
    dim3 block(SG_PER_TG * WARP_SIZE);

    sdpa_q1_sparse_fp32<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const float4*>(d_Q),
        reinterpret_cast<const float4*>(d_K),
        reinterpret_cast<const float4*>(d_V),
        reinterpret_cast<float4*>(d_O),
        N_tokens, scale, W_window,
        d_indices, d_counts, K_sparse_max, d_head_map, D_HEAD);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sdpa_q1_sparse_fp32 launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse-mask SDPA Q=1 — FP16 storage variant. Port of `sdpa_q1_sparse_fp16`
// from src/ffi/metal_compute.metal:557. Same online-softmax + indexing as
// the fp32 sibling above; storage is fp16 (half2-packed equivalent of MSL's
// half4) but accumulators stay fp32 for numerical parity with vanilla mlx-lm
// SDPA. Cuts K/V DRAM bandwidth ~2x at large N.
// Layout — all device buffers are __half-typed instead of float; Q/K/V/O
// strides are still D8 = D/8 half2s per row (since 1 half2 = 2 halves = 4 B,
// matching float4's 16 B with 4 floats / 8 halves per element... actually no,
// for symmetry with the MSL kernel that uses half4 (= 8 bytes), we use
// "half-pair-of-half2 = 4 halves = 8 bytes" via __half[4] structs). For
// simplicity in this prototype, work in __half2 (8 bytes = 4 halves... wait)
// Standardizing: D_HALF8 = D/8 packs of 8 halves; each pack stored as 2 ×
// __half4 = 2 × ushort4. We just go through D as a half pointer and convert
// to float on load.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __global__
void sdpa_q1_sparse_fp16(
    const __half* __restrict__ Q,           // [H_q,  D]      fp16
    const __half* __restrict__ K,           // [H_kv, N, D]   fp16
    const __half* __restrict__ V,           // [H_kv, N, D]   fp16
    __half*       __restrict__ O,           // [H_q,  D]      fp16
    uint32_t                   N_tokens,
    float                      scale,
    uint32_t                   W_window,
    const int* __restrict__    indices,     // [H_q, K_sparse_max]
    const uint32_t* __restrict__ counts,    // [H_q]
    uint32_t                   K_sparse_max,
    const unsigned char* __restrict__ head_map,
    uint32_t                   D_HEAD)
{
    extern __shared__ unsigned char smem_raw[];
    float*  s_m = reinterpret_cast<float*>(smem_raw);
    float*  s_l = s_m + SG_PER_TG;
    uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_l + SG_PER_TG);
    s_o_addr = (s_o_addr + 15) & ~uintptr_t(15);
    // s_o stays fp32 (per-warp partial output, summed before final fp16 cast).
    float4* s_o = reinterpret_cast<float4*>(s_o_addr);

    const uint32_t D4    = D_HEAD / 4u;     // groups of 4 halves loaded together
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    const uint32_t h    = blockIdx.x;
    const uint32_t sg   = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    uint32_t h_kv = head_map ? (uint32_t)head_map[h] : h;

    // Load Q as int2 (8 bytes = 4 halves = 1 chunk slot) → bit-cast to 2 __half2
    // → unpack to float4. Single coalesced 8-byte load per chunk slot, vs the
    // 4 scalar __half2float calls the prototype was using. Matches the
    // bandwidth profile of the fp32 path's float4 (16-byte) loads.
    float4 q[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) q[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        uint32_t idx = c * WARP_SIZE + lane;
        if (idx < D4) {
            const int2* qp = reinterpret_cast<const int2*>(Q + h * D_HEAD + idx * 4u);
            int2 raw = __ldg(qp);
            __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
            __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
            q[c].x = __low2float(h0);
            q[c].y = __high2float(h0);
            q[c].z = __low2float(h1);
            q[c].w = __high2float(h1);
        }
    }

    uint32_t K_sparse_h = counts[h];
    uint32_t per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint32_t i_begin    = sg * per_sg;
    uint32_t i_end      = min(i_begin + per_sg, K_sparse_h);

    uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) o[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const __half* k_base   = K + (size_t)h_kv * N_tokens * D_HEAD;
    const __half* v_base   = V + (size_t)h_kv * N_tokens * D_HEAD;
    const int*    idx_base = indices + (size_t)h * K_sparse_max;

    for (uint32_t i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint32_t)idx_i >= N_tokens) continue;
        uint32_t t = (uint32_t)idx_i;
        if (t < eff_start) continue;

        float partial = 0.0f;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                const int2* kp = reinterpret_cast<const int2*>(k_base + (size_t)t * D_HEAD + idx * 4u);
                int2 raw = __ldg(kp);
                __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
                __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
                float kx = __low2float(h0);
                float ky = __high2float(h0);
                float kz = __low2float(h1);
                float kw = __high2float(h1);
                partial += q[c].x * kx + q[c].y * ky + q[c].z * kz + q[c].w * kw;
            }
        }
        float score = warp_sum(partial) * scale;

        float new_m  = fmaxf(m, score);
        float factor = __expf(m - new_m);
        float exp_s  = __expf(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                const int2* vp = reinterpret_cast<const int2*>(v_base + (size_t)t * D_HEAD + idx * 4u);
                int2 raw = __ldg(vp);
                __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
                __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
                float vx = __low2float(h0);
                float vy = __high2float(h0);
                float vz = __low2float(h1);
                float vw = __high2float(h1);
                o[c].x = o[c].x * factor + exp_s * vx;
                o[c].y = o[c].y * factor + exp_s * vy;
                o[c].z = o[c].z * factor + exp_s * vz;
                o[c].w = o[c].w * factor + exp_s * vw;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        s_o[sg * CHUNK * WARP_SIZE + c * WARP_SIZE + lane] = o[c];
    }
    __syncthreads();

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = fmaxf(M, m_i);
            float a = __expf(M   - new_M);
            float b = __expf(m_i - new_M);
            for (uint32_t c = 0u; c < CHUNK; ++c) {
                float4 part = s_o[i * CHUNK * WARP_SIZE + c * WARP_SIZE + lane];
                O_acc[c].x = O_acc[c].x * a + part.x * b;
                O_acc[c].y = O_acc[c].y * a + part.y * b;
                O_acc[c].z = O_acc[c].z * a + part.z * b;
                O_acc[c].w = O_acc[c].w * a + part.w * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        float inv_L = 1.0f / L;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                // Pack 4 halves as 2 __half2 -> 1 int2 -> single 8B store.
                __half2 h0 = __floats2half2_rn(O_acc[c].x * inv_L, O_acc[c].y * inv_L);
                __half2 h1 = __floats2half2_rn(O_acc[c].z * inv_L, O_acc[c].w * inv_L);
                int2 raw;
                raw.x = *reinterpret_cast<int*>(&h0);
                raw.y = *reinterpret_cast<int*>(&h1);
                int2* op = reinterpret_cast<int2*>(O + h * D_HEAD + idx * 4u);
                *op = raw;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse-mask SDPA Q=M (batched prefill) — FP16 storage variant.
//
// Generalizes sdpa_q1_sparse_fp16 to M>1 query positions per (head, batch).
// For each (h, q):
//   - Effective K positions = caller-supplied prefix indices (K_sparse_h count)
//     UNION causal suffix [N_prefix, N_prefix + q] inclusive
//   - Online-softmax accumulator (FlashAttention-1) over those positions
//
// Caller layout convention: K and V are CONCATENATED prefix+suffix in the
// same [H_kv, N_total, D] tensor where N_total = N_prefix + M. Indices
// returned by the block-mean selector are into [0, N_prefix) only; the
// suffix is reached by walking past N_prefix in this kernel.
//
// Grid: (H_q, M); Block: SG_PER_TG * 32 = 256 threads (8 warps).
// One CUDA block per (head, query position). Warps split positions; lanes
// within a warp split D4 chunks. fp16 storage with int2 vectorized loads
// (matches sdpa_q1_sparse_fp16's pattern).
//
// Use case: gh #15 suffix prefill — closes the M>1 path so Pion can bite
// into the actual TTFT metric (which times the suffix prefill in the
// CAG measurement). The decode-step path (M=1) still uses sdpa_q1_sparse_fp16.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __global__
void sdpa_qm_sparse_fp16(
    const __half* __restrict__ Q,           // [H_q, M, D]
    const __half* __restrict__ K,           // [H_kv, N_total, D]
    const __half* __restrict__ V,           // [H_kv, N_total, D]
    __half*       __restrict__ O,           // [H_q, M, D]
    uint32_t                   M,
    uint32_t                   N_total,
    uint32_t                   N_prefix,    // suffix starts at index N_prefix
    float                      scale,
    uint32_t                   W_window,    // sliding-window over prefix only; 0 = full
    const int* __restrict__    indices,     // [H_q, K_sparse_max] — prefix indices
    const uint32_t* __restrict__ counts,    // [H_q]
    uint32_t                   K_sparse_max,
    const unsigned char* __restrict__ head_map,
    uint32_t                   D_HEAD)
{
    extern __shared__ unsigned char smem_raw[];
    float*  s_m = reinterpret_cast<float*>(smem_raw);
    float*  s_l = s_m + SG_PER_TG;
    uintptr_t s_o_addr = reinterpret_cast<uintptr_t>(s_l + SG_PER_TG);
    s_o_addr = (s_o_addr + 15) & ~uintptr_t(15);
    float4* s_o = reinterpret_cast<float4*>(s_o_addr);

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    const uint32_t h    = blockIdx.x;       // query head
    const uint32_t q    = blockIdx.y;       // query position in [0, M)
    const uint32_t sg   = threadIdx.x / WARP_SIZE;
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1);

    const uint32_t h_kv = head_map ? (uint32_t)head_map[h] : h;

    // Load Q[h, q, :] into per-lane float4 chunks via int2 vectorized fp16 loads.
    float4 qv[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) qv[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const __half* Q_base = Q + ((size_t)h * M + q) * D_HEAD;
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        uint32_t idx = c * WARP_SIZE + lane;
        if (idx < D4) {
            const int2* qp = reinterpret_cast<const int2*>(Q_base + idx * 4u);
            int2 raw = __ldg(qp);
            __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
            __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
            qv[c].x = __low2float(h0);
            qv[c].y = __high2float(h0);
            qv[c].z = __low2float(h1);
            qv[c].w = __high2float(h1);
        }
    }

    // Effective positions: [0, K_sparse_h) into prefix indices, then
    // [K_sparse_h, K_sparse_h + q + 1) into suffix [N_prefix, N_prefix+q].
    uint32_t K_sparse_h = counts[h];
    uint32_t total_positions = K_sparse_h + q + 1u;
    uint32_t per_sg = (total_positions + SG_PER_TG - 1u) / SG_PER_TG;
    uint32_t i_begin = sg * per_sg;
    uint32_t i_end   = min(i_begin + per_sg, total_positions);

    uint32_t eff_start = (W_window > 0u && N_prefix > W_window) ?
                         (N_prefix - W_window) : 0u;

    float  m_acc = -INFINITY;
    float  l_acc = 0.0f;
    float4 o[CHUNK_MAX];
    #pragma unroll
    for (uint32_t c = 0u; c < CHUNK_MAX; ++c) o[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const __half* k_base   = K + (size_t)h_kv * N_total * D_HEAD;
    const __half* v_base   = V + (size_t)h_kv * N_total * D_HEAD;
    const int*    idx_base = indices + (size_t)h * K_sparse_max;

    for (uint32_t i = i_begin; i < i_end; ++i) {
        uint32_t n;
        if (i < K_sparse_h) {
            int idx_i = idx_base[i];
            if (idx_i < 0 || (uint32_t)idx_i >= N_prefix) continue;
            uint32_t pn = (uint32_t)idx_i;
            if (pn < eff_start) continue;
            n = pn;
        } else {
            // Suffix position: causal slot (i - K_sparse_h) in [0, q].
            n = N_prefix + (i - K_sparse_h);
        }

        // QK dot — int2 vectorized fp16 load
        float partial = 0.0f;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                const int2* kp = reinterpret_cast<const int2*>(k_base + (size_t)n * D_HEAD + idx * 4u);
                int2 raw = __ldg(kp);
                __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
                __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
                float kx = __low2float(h0), ky = __high2float(h0);
                float kz = __low2float(h1), kw = __high2float(h1);
                partial += qv[c].x * kx + qv[c].y * ky + qv[c].z * kz + qv[c].w * kw;
            }
        }
        float score = warp_sum(partial) * scale;

        float new_m  = fmaxf(m_acc, score);
        float factor = __expf(m_acc - new_m);
        float exp_s  = __expf(score - new_m);
        m_acc = new_m;
        l_acc = l_acc * factor + exp_s;

        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                const int2* vp = reinterpret_cast<const int2*>(v_base + (size_t)n * D_HEAD + idx * 4u);
                int2 raw = __ldg(vp);
                __half2 h0 = *reinterpret_cast<__half2*>(&raw.x);
                __half2 h1 = *reinterpret_cast<__half2*>(&raw.y);
                float vx = __low2float(h0), vy = __high2float(h0);
                float vz = __low2float(h1), vw = __high2float(h1);
                o[c].x = o[c].x * factor + exp_s * vx;
                o[c].y = o[c].y * factor + exp_s * vy;
                o[c].z = o[c].z * factor + exp_s * vz;
                o[c].w = o[c].w * factor + exp_s * vw;
            }
        }
    }

    // Per-warp partials → shared mem, then warp 0 merges via online softmax.
    if (lane == 0u) {
        s_m[sg] = m_acc;
        s_l[sg] = l_acc;
    }
    for (uint32_t c = 0u; c < CHUNK; ++c) {
        s_o[sg * CHUNK * WARP_SIZE + c * WARP_SIZE + lane] = o[c];
    }
    __syncthreads();

    if (sg == 0u) {
        float M_acc = -INFINITY;
        float L_acc = 0.0f;
        float4 O_acc[CHUNK_MAX];
        #pragma unroll
        for (uint32_t c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = fmaxf(M_acc, m_i);
            float a = __expf(M_acc - new_M);
            float b = __expf(m_i   - new_M);
            for (uint32_t c = 0u; c < CHUNK; ++c) {
                float4 part = s_o[i * CHUNK * WARP_SIZE + c * WARP_SIZE + lane];
                O_acc[c].x = O_acc[c].x * a + part.x * b;
                O_acc[c].y = O_acc[c].y * a + part.y * b;
                O_acc[c].z = O_acc[c].z * a + part.z * b;
                O_acc[c].w = O_acc[c].w * a + part.w * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        // Empty-mask guard: l=0 → would produce NaN. Write zeros (caller bug;
        // mirrors the M=1 sparse kernel's contract).
        if (L_acc <= 0.0f) {
            for (uint32_t c = 0u; c < CHUNK; ++c) {
                uint32_t idx = c * WARP_SIZE + lane;
                if (idx < D4) {
                    int2 raw = make_int2(0, 0);
                    int2* op = reinterpret_cast<int2*>(O + ((size_t)h * M + q) * D_HEAD + idx * 4u);
                    *op = raw;
                }
            }
            return;
        }
        float inv_L = 1.0f / L_acc;
        for (uint32_t c = 0u; c < CHUNK; ++c) {
            uint32_t idx = c * WARP_SIZE + lane;
            if (idx < D4) {
                __half2 hp0 = __floats2half2_rn(O_acc[c].x * inv_L, O_acc[c].y * inv_L);
                __half2 hp1 = __floats2half2_rn(O_acc[c].z * inv_L, O_acc[c].w * inv_L);
                int2 raw;
                raw.x = *reinterpret_cast<int*>(&hp0);
                raw.y = *reinterpret_cast<int*>(&hp1);
                int2* op = reinterpret_cast<int2*>(O + ((size_t)h * M + q) * D_HEAD + idx * 4u);
                *op = raw;
            }
        }
    }
}

extern "C" int pion_sdpa_qm_sparse_fp16_cuda(
    const __half* d_Q,           // [H_q, M, D]
    const __half* d_K,           // [H_kv, N_total, D]
    const __half* d_V,           // [H_kv, N_total, D]
    __half*       d_O,           // [H_q, M, D]
    uint32_t      H_q,
    uint32_t      M,
    uint32_t      N_total,
    uint32_t      N_prefix,
    uint32_t      D_HEAD,
    float         scale,
    uint32_t      W_window,
    const int*    d_indices,
    const uint32_t* d_counts,
    uint32_t      K_sparse_max,
    const unsigned char* d_head_map,
    cudaStream_t  stream)
{
    if (D_HEAD == 0 || (D_HEAD & 3u) != 0u) return -1;
    if (D_HEAD > 4u * CHUNK_MAX * WARP_SIZE) return -2;
    if (H_q == 0u || M == 0u || N_total == 0u || K_sparse_max == 0u) return -3;
    if (N_prefix > N_total) return -4;

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;
    size_t smem_bytes = 2 * SG_PER_TG * sizeof(float)
                      + 16
                      + (size_t)SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4);

    dim3 grid(H_q, M);
    dim3 block(SG_PER_TG * WARP_SIZE);

    sdpa_qm_sparse_fp16<<<grid, block, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O,
        M, N_total, N_prefix, scale, W_window,
        d_indices, d_counts, K_sparse_max, d_head_map, D_HEAD);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sdpa_qm_sparse_fp16 launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

extern "C" int pion_sdpa_q1_sparse_fp16_cuda(
    const __half* d_Q,
    const __half* d_K,
    const __half* d_V,
    __half*       d_O,
    uint32_t      H_q,
    uint32_t      N_tokens,
    uint32_t      D_HEAD,
    float         scale,
    uint32_t      W_window,
    const int*    d_indices,
    const uint32_t* d_counts,
    uint32_t      K_sparse_max,
    const unsigned char* d_head_map,
    cudaStream_t  stream)
{
    if (D_HEAD == 0 || (D_HEAD & 3u) != 0u) return -1;
    if (D_HEAD > 4u * CHUNK_MAX * WARP_SIZE) return -2;
    if (H_q == 0u || N_tokens == 0u || K_sparse_max == 0u) return -3;

    const uint32_t D4    = D_HEAD / 4u;
    const uint32_t CHUNK = (D4 + 31u) / 32u;

    size_t smem_bytes = 2 * SG_PER_TG * sizeof(float)
                      + 16
                      + (size_t)SG_PER_TG * CHUNK * WARP_SIZE * sizeof(float4);

    dim3 grid(H_q);
    dim3 block(SG_PER_TG * WARP_SIZE);

    sdpa_q1_sparse_fp16<<<grid, block, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O,
        N_tokens, scale, W_window,
        d_indices, d_counts, K_sparse_max, d_head_map, D_HEAD);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "sdpa_q1_sparse_fp16 launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-mean K-selector — gh #9 item 6.
//
// Server-side top-K block picker for ATTEND.PREFIX.QUERY_SPARSE_AUTO.
// Mirrors the Mac-side `make_pion_prompt_cache(sparse_full_layers={K_block,
// K_blocks})` algorithm: for each query head h, pick the K_blocks blocks
// (each K_block tokens wide) whose mean K vector has the highest dot-product
// with Q[h]. Returns indices[H_q, K_blocks * K_block] suitable for direct
// dispatch to sdpa_q1_sparse_fp32.
//
// One block per query head; threads in the block cooperate on:
//   1. Compute per-block mean K (across K_block tokens, all D dimensions)
//   2. Compute score = mean_K · Q for each block
//   3. Top-K_blocks selection (selection sort, K_blocks small)
//   4. Expand selected block IDs into per-token indices
//
// W_window: blocks whose end-token is below (N - W_window) are excluded.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __global__
void block_mean_topk_select(
    const float* __restrict__ Q,         // [H_q, D]
    const float* __restrict__ K,         // [H_kv, N, D]
    int* __restrict__         indices,   // [H_q, K_blocks * K_block]
    uint32_t* __restrict__    counts,    // [H_q]
    uint32_t                  N_tokens,
    uint32_t                  D_HEAD,
    uint32_t                  K_block,
    uint32_t                  K_blocks,
    const unsigned char* __restrict__ head_map,
    uint32_t                  W_window)
{
    // Shared memory layout (set via kernel-launch smem_bytes):
    //   float s_Q[D_HEAD]                  ← cached Q[h] once per block
    //   float s_scores[n_blocks]           ← per-block selector score
    extern __shared__ unsigned char smem_raw[];
    float* s_Q      = reinterpret_cast<float*>(smem_raw);
    float* s_scores = s_Q + D_HEAD;

    const uint32_t h    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t nthr = blockDim.x;

    const uint32_t h_kv = head_map ? (uint32_t)head_map[h] : h;
    const uint32_t n_blocks = (N_tokens + K_block - 1u) / K_block;
    const uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ?
                                (N_tokens - W_window) : 0u;

    const float* Q_h    = Q + (size_t)h    * D_HEAD;
    const float* K_base = K + (size_t)h_kv * N_tokens * D_HEAD;

    // Cooperatively load Q[h] into shared memory ONCE — every warp re-uses
    // it across all blocks it handles. Replaces per-block re-fetches from
    // global. D_HEAD ≤ 512, threads = 128 → at most 4 strided loads per
    // thread.
    for (uint32_t d = tid; d < D_HEAD; d += nthr) {
        s_Q[d] = Q_h[d];
    }
    __syncthreads();

    // Phase 1 (warp-shuffle reduction): each WARP (32 lanes) handles ONE
    // block at a time. Lanes co-process the per-block (span × D4) work
    // with adjacent lanes reading adjacent tokens → all 32 K loads in a
    // step land on a single 512-byte cache line. Per-lane partial summed
    // across lanes via __shfl_xor_sync butterfly, lane 0 writes the score.
    //
    // Volume per CUDA-block: same as before. Win: HBM bandwidth utilization
    // — old version had each thread handle its own (far-apart) tokens →
    // 32 separate 16-byte requests per cycle. New: one 512-byte request
    // serves the whole warp.
    const uint32_t D4   = D_HEAD >> 2u;   // D must be a multiple of 4
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t warps_per_cublock = nthr >> 5u;   // 128 / 32 = 4
    const float4* Q4 = reinterpret_cast<const float4*>(s_Q);

    for (uint32_t b = warp; b < n_blocks; b += warps_per_cublock) {
        uint32_t t0 = b * K_block;
        uint32_t t1 = (t0 + K_block) < N_tokens ? (t0 + K_block) : N_tokens;
        if (t1 <= eff_start) {
            if (lane == 0u) s_scores[b] = -INFINITY;
            continue;
        }
        uint32_t e0 = t0 > eff_start ? t0 : eff_start;
        uint32_t e1 = t1;
        uint32_t span = e1 - e0;
        if (span == 0u) {
            if (lane == 0u) s_scores[b] = -INFINITY;
            continue;
        }
        // Lane i handles tokens (e0 + i, e0 + i + 32, e0 + i + 64, ...).
        // For K_block ≤ 32: each lane gets 0 or 1 token (idle lanes contribute 0).
        // For K_block = 64: each lane gets exactly 2 tokens.
        float partial = 0.0f;
        for (uint32_t local_t = lane; local_t < span; local_t += 32u) {
            uint32_t t = e0 + local_t;
            const float4* K4 = reinterpret_cast<const float4*>(
                K_base + (size_t)t * D_HEAD);
            float dot_t = 0.0f;
            #pragma unroll 4
            for (uint32_t d4 = 0; d4 < D4; ++d4) {
                float4 k = __ldg(K4 + d4);
                float4 q = Q4[d4];
                dot_t += k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w;
            }
            partial += dot_t;
        }
        // Butterfly reduction: sum partial across all 32 lanes.
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0u) {
            s_scores[b] = partial / (float)span;
        }
    }
    __syncthreads();

    // Phase 2: thread 0 does a top-K_blocks selection sort.
    if (tid == 0u) {
        uint32_t K = K_blocks < n_blocks ? K_blocks : n_blocks;
        for (uint32_t k = 0; k < K; ++k) {
            float best = -INFINITY;
            int best_i = -1;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                float s = s_scores[b];
                if (s > best) { best = s; best_i = (int)b; }
            }
            if (best_i < 0) break;
            uint32_t t0 = (uint32_t)best_i * K_block;
            uint32_t t1 = (t0 + K_block) < N_tokens ? (t0 + K_block) : N_tokens;
            for (uint32_t t = t0; t < t1; ++t) {
                indices[(size_t)h * (K_blocks * K_block) + k * K_block + (t - t0)] = (int)t;
            }
            for (uint32_t pad = (t1 - t0); pad < K_block; ++pad) {
                indices[(size_t)h * (K_blocks * K_block) + k * K_block + pad] = -1;
            }
            s_scores[best_i] = -INFINITY;
        }
        counts[h] = K * K_block;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Precompute kernel: fills block_means[H_kv, n_blocks, D] = mean over
// K_block tokens per block. One CUDA block per (h_kv, b). Threads in the
// CUDA block cooperate on the per-D-chunk average. Coalesced float4 reads.
//
// Called from store_kv after K is on device. Memory cost per slot:
// H_kv * n_blocks * D * 4 bytes. At Llama-3-8B GQA H_kv=8, K_block=64,
// D=128, N=26K: 8 * 412 * 128 * 4 = 1.7 MB per layer. × 32 layers ≈ 54 MB
// per session — negligible vs the K/V cache (3.4 GB at the same shape).
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__
void compute_block_means(
    const float* __restrict__ K,            // [H_kv, N, D]
    float*       __restrict__ block_means,  // [H_kv, n_blocks, D]
    uint32_t                  N_tokens,
    uint32_t                  D_HEAD,
    uint32_t                  K_block)
{
    const uint32_t h_kv  = blockIdx.x;
    const uint32_t b     = blockIdx.y;
    const uint32_t tid   = threadIdx.x;
    const uint32_t nthr  = blockDim.x;
    const uint32_t D4    = D_HEAD >> 2u;

    const uint32_t t0    = b * K_block;
    const uint32_t t1    = (t0 + K_block) < N_tokens ? (t0 + K_block) : N_tokens;
    const uint32_t span  = t1 - t0;
    if (span == 0u) return;
    const float    inv_span = 1.0f / (float)span;

    const float* K_row = K + (size_t)h_kv * N_tokens * D_HEAD;
    const uint32_t n_blocks = (N_tokens + K_block - 1u) / K_block;
    float* bm_row = block_means + ((size_t)h_kv * n_blocks + b) * D_HEAD;

    // Each thread handles a strided slice of D4 chunks.
    for (uint32_t d4 = tid; d4 < D4; d4 += nthr) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint32_t t = t0; t < t1; ++t) {
            const float4* row4 = reinterpret_cast<const float4*>(K_row + (size_t)t * D_HEAD);
            float4 k = __ldg(&row4[d4]);
            sum.x += k.x; sum.y += k.y; sum.z += k.z; sum.w += k.w;
        }
        sum.x *= inv_span; sum.y *= inv_span; sum.z *= inv_span; sum.w *= inv_span;
        reinterpret_cast<float4*>(bm_row)[d4] = sum;
    }
}

extern "C" int pion_compute_block_means_cuda(
    const float* d_K,
    float*       d_block_means,
    uint32_t     H_kv,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    uint32_t     K_block,
    cudaStream_t stream)
{
    if (H_kv == 0 || N_tokens == 0 || D_HEAD == 0 || K_block == 0) return -3;
    if ((D_HEAD & 3u) != 0u) return -1;
    uint32_t n_blocks = (N_tokens + K_block - 1u) / K_block;
    dim3 grid(H_kv, n_blocks);
    dim3 block(64);
    compute_block_means<<<grid, block, 0, stream>>>(
        d_K, d_block_means, N_tokens, D_HEAD, K_block);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "compute_block_means launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-K selector that reads PRECOMPUTED block_means instead of scanning K.
// O(H_q × n_blocks × D) per call — drops the dominant K-scan cost from
// the hot path entirely. Selection sort same as block_mean_topk_select.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" __global__
void select_topk_from_precomputed(
    const float* __restrict__ Q,             // [H_q, D]
    const float* __restrict__ block_means,   // [H_kv, n_blocks, D]
    int* __restrict__         indices,       // [H_q, K_blocks * K_block]
    uint32_t* __restrict__    counts,        // [H_q]
    uint32_t                  N_tokens,
    uint32_t                  D_HEAD,
    uint32_t                  K_block,
    uint32_t                  K_blocks,
    uint32_t                  n_blocks,
    const unsigned char* __restrict__ head_map,
    uint32_t                  W_window)
{
    extern __shared__ unsigned char smem_raw[];
    float* s_Q      = reinterpret_cast<float*>(smem_raw);
    float* s_scores = s_Q + D_HEAD;

    const uint32_t h    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t nthr = blockDim.x;
    const uint32_t h_kv = head_map ? (uint32_t)head_map[h] : h;
    const uint32_t eff_start = (W_window > 0u && N_tokens > W_window) ?
                                (N_tokens - W_window) : 0u;
    const uint32_t D4   = D_HEAD >> 2u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t warps_per_cublock = nthr >> 5u;

    // Cache Q[h] in shared once.
    const float* Q_h = Q + (size_t)h * D_HEAD;
    for (uint32_t d = tid; d < D_HEAD; d += nthr) s_Q[d] = Q_h[d];
    __syncthreads();
    const float4* Q4 = reinterpret_cast<const float4*>(s_Q);

    // Per-warp processing of one block at a time. Each warp dots Q[h] with
    // block_means[h_kv, b, :]. Lanes split D4 work.
    const float* bm_base = block_means + (size_t)h_kv * n_blocks * D_HEAD;

    for (uint32_t b = warp; b < n_blocks; b += warps_per_cublock) {
        uint32_t t0 = b * K_block;
        uint32_t t1 = (t0 + K_block) < N_tokens ? (t0 + K_block) : N_tokens;
        if (t1 <= eff_start) {
            if (lane == 0u) s_scores[b] = -INFINITY;
            continue;
        }
        const float4* bm4 = reinterpret_cast<const float4*>(bm_base + (size_t)b * D_HEAD);
        float partial = 0.0f;
        for (uint32_t d4 = lane; d4 < D4; d4 += 32u) {
            float4 m = __ldg(&bm4[d4]);
            float4 q = Q4[d4];
            partial += m.x * q.x + m.y * q.y + m.z * q.z + m.w * q.w;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0u) s_scores[b] = partial;
    }
    __syncthreads();

    // Same top-K selection as the full-scan kernel.
    if (tid == 0u) {
        uint32_t K = K_blocks < n_blocks ? K_blocks : n_blocks;
        for (uint32_t k = 0; k < K; ++k) {
            float best = -INFINITY;
            int best_i = -1;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                float s = s_scores[b];
                if (s > best) { best = s; best_i = (int)b; }
            }
            if (best_i < 0) break;
            uint32_t t0 = (uint32_t)best_i * K_block;
            uint32_t t1 = (t0 + K_block) < N_tokens ? (t0 + K_block) : N_tokens;
            for (uint32_t t = t0; t < t1; ++t) {
                indices[(size_t)h * (K_blocks * K_block) + k * K_block + (t - t0)] = (int)t;
            }
            for (uint32_t pad = (t1 - t0); pad < K_block; ++pad) {
                indices[(size_t)h * (K_blocks * K_block) + k * K_block + pad] = -1;
            }
            s_scores[best_i] = -INFINITY;
        }
        counts[h] = K * K_block;
    }
}

extern "C" int pion_select_topk_from_precomputed_cuda(
    const float* d_Q,
    const float* d_block_means,
    int*         d_indices,
    uint32_t*    d_counts,
    uint32_t     H_q,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    uint32_t     K_block,
    uint32_t     K_blocks,
    uint32_t     n_blocks,
    const unsigned char* d_head_map,
    uint32_t     W_window,
    cudaStream_t stream)
{
    if (H_q == 0 || N_tokens == 0 || D_HEAD == 0 || K_block == 0 || K_blocks == 0) return -3;
    if ((D_HEAD & 3u) != 0u) return -1;
    size_t smem_bytes = (size_t)D_HEAD * sizeof(float)
                      + (size_t)n_blocks * sizeof(float);
    dim3 grid(H_q);
    dim3 block(128);
    select_topk_from_precomputed<<<grid, block, smem_bytes, stream>>>(
        d_Q, d_block_means, d_indices, d_counts,
        N_tokens, D_HEAD, K_block, K_blocks, n_blocks, d_head_map, W_window);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "select_topk_from_precomputed launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}

extern "C" int pion_block_mean_topk_select_cuda(
    const float* d_Q,
    const float* d_K,
    int*         d_indices,
    uint32_t*    d_counts,
    uint32_t     H_q,
    uint32_t     N_tokens,
    uint32_t     D_HEAD,
    uint32_t     K_block,
    uint32_t     K_blocks,
    const unsigned char* d_head_map,
    uint32_t     W_window,
    cudaStream_t stream)
{
    if (H_q == 0 || N_tokens == 0 || D_HEAD == 0 || K_block == 0 || K_blocks == 0) return -3;
    if ((D_HEAD & 3u) != 0u) return -1;  // selector requires D % 4 == 0 (matches the SDPA kernels)

    uint32_t n_blocks = (N_tokens + K_block - 1u) / K_block;
    // Shared memory: D_HEAD floats for Q + n_blocks floats for scores.
    size_t smem_bytes = (size_t)D_HEAD * sizeof(float)
                      + (size_t)n_blocks * sizeof(float);

    dim3 grid(H_q);
    dim3 block(128);

    block_mean_topk_select<<<grid, block, smem_bytes, stream>>>(
        d_Q, d_K, d_indices, d_counts,
        N_tokens, D_HEAD, K_block, K_blocks, d_head_map, W_window);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "block_mean_topk_select launch failed: %s\n", cudaGetErrorString(err));
        return -10;
    }
    return 0;
}
