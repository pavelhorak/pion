#include <metal_stdlib>
using namespace metal;

// ─── INT8 L2 Distance: 1 query vs N candidates ─────────────────────────
// Each thread computes distance for one candidate vector.
// Uses threadgroup shared memory to cache query vector for reuse across threads.
//
// Layout of candidates buffer: N vectors × dim bytes (INT8), contiguous.
// Layout of norms buffer: N floats (precomputed INT8 L2 norm² per candidate).
// Output: N floats = query_norm + cand_norm - 2*dot(query, cand)

// compact_buffer layout per vector: [Float32 full_norm_sq (4B)][Float32 prefix_norm_sq (4B)][Int8 × dim]
// stride = 8 + dim bytes per vector

// V2: char16 vectorized loads (96 iterations vs 384) + threadgroup shared query cache.
// Gemini analysis: char4 loop requires 384 iterations; char16 cuts to 96.
// Threadgroup memory prevents query eviction from L1 during 77MB candidate streaming.

kernel void int8_l2_distance_batch(
    device const char     *query       [[buffer(0)]],   // [dim] INT8 query
    device const char     *candidates  [[buffer(1)]],   // compact_buffer: N × (8 + dim) bytes
    device       float    *distances   [[buffer(2)]],   // [N] output L2 distances
    constant     uint     &dim         [[buffer(3)]],   // vector dimension (e.g. 1536)
    constant     float    &query_norm  [[buffer(4)]],   // precomputed query INT8 norm²
    constant     uint     &stride      [[buffer(5)]],   // bytes per vector slot (8 + dim)
    constant     uint     &num_vectors [[buffer(6)]],   // N (bounds check for dispatchThreadgroups)
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Bounds check (required when using dispatchThreadgroups — last group may be partial)
    if (gid >= num_vectors) return;

    // Cache query in threadgroup shared memory (1536 bytes, shared across 256 threads).
    // Prevents L1 eviction when streaming the 77MB candidate buffer.
    threadgroup char shared_query[1536];
    // Cooperative load: each thread loads a chunk of the query
    for (uint i = lid; i < dim; i += tg_size) {
        shared_query[i] = query[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread: one candidate vector
    device const char *slot = candidates + gid * stride;
    device const float *norm_ptr = (device const float *)slot;
    float cand_norm = norm_ptr[0];  // full_norm_sq is first 4 bytes
    device const char *vec = slot + 8;

    // char4 × 4 unrolled: process 16 bytes per loop body (96 iterations for 1536 dims).
    // Metal doesn't support char16 (reserved name). Use char4 with 4× unroll instead.
    // short4 widening for multiply avoids int8 overflow, accumulate into int.
    int dot_acc = 0;
    uint dim16 = dim / 16;
    threadgroup const char4 *q4 = (threadgroup const char4 *)shared_query;
    device const char4 *v4 = (device const char4 *)vec;

    for (uint i = 0; i < dim16; i++) {
        uint base = i * 4;
        // 4 × char4 = 16 bytes per iteration
        short4 p0 = short4(q4[base])     * short4(v4[base]);
        short4 p1 = short4(q4[base + 1]) * short4(v4[base + 1]);
        short4 p2 = short4(q4[base + 2]) * short4(v4[base + 2]);
        short4 p3 = short4(q4[base + 3]) * short4(v4[base + 3]);
        // Pairwise horizontal sum: short4 → 2 ints via dot(short4.lo, short4.hi)
        dot_acc += int(p0.x) + int(p0.y) + int(p0.z) + int(p0.w)
                 + int(p1.x) + int(p1.y) + int(p1.z) + int(p1.w)
                 + int(p2.x) + int(p2.y) + int(p2.z) + int(p2.w)
                 + int(p3.x) + int(p3.y) + int(p3.z) + int(p3.w);
    }

    // Handle remaining elements (dim % 16) — for 1536 this is 0
    for (uint i = dim16 * 16; i < dim; i++) {
        dot_acc += int(shared_query[i]) * int(vec[i]);
    }

    // L2 distance = query_norm² + cand_norm² - 2 * dot(query, candidate)
    float dist = query_norm + cand_norm - 2.0f * float(dot_acc);
    distances[gid] = dist;
}

// ─── Batch multi-query: Q queries × N candidates ───────────────────────
// §7v2: Optimized with threadgroup shared query cache + char4×4 unrolling.
// 2D grid: (N, Q). Each threadgroup handles candidates for ONE query.
// Query loaded cooperatively into shared memory → prevents L1 eviction.

kernel void int8_l2_batch_multiquery(
    device const char     *queries      [[buffer(0)]],   // [Q × dim] INT8 queries
    device const char     *candidates   [[buffer(1)]],   // compact_buffer: N × stride bytes
    device const float    *query_norms  [[buffer(2)]],   // [Q] query norms
    device       float    *distances    [[buffer(3)]],   // [Q × N] output distances
    constant     uint     &dim          [[buffer(4)]],
    constant     uint     &N            [[buffer(5)]],
    constant     uint     &stride       [[buffer(6)]],   // 8 + dim
    uint2 gid [[thread_position_in_grid]],               // .x = candidate idx, .y = query idx
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint qidx = gid.y;
    uint cidx = gid.x;
    uint lid = lid2.x;
    uint tg_size = tg_size2.x;

    // Cache query in threadgroup shared memory (1536 bytes, shared across 256 threads).
    threadgroup char shared_query[1536];
    device const char *q_src = queries + qidx * dim;
    for (uint i = lid; i < dim; i += tg_size) {
        shared_query[i] = q_src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (cidx >= N) return;

    // Each thread: one candidate vector
    device const char *slot = candidates + cidx * stride;
    device const float *norm_ptr = (device const float *)slot;
    float cand_norm = norm_ptr[0];
    device const char *vec = slot + 8;

    // char4 × 4 unrolled: 16 bytes per iteration (96 iterations for 1536 dims)
    int dot_acc = 0;
    uint dim16 = dim / 16;
    threadgroup const char4 *q4 = (threadgroup const char4 *)shared_query;
    device const char4 *v4 = (device const char4 *)vec;

    for (uint i = 0; i < dim16; i++) {
        uint base = i * 4;
        short4 p0 = short4(q4[base])     * short4(v4[base]);
        short4 p1 = short4(q4[base + 1]) * short4(v4[base + 1]);
        short4 p2 = short4(q4[base + 2]) * short4(v4[base + 2]);
        short4 p3 = short4(q4[base + 3]) * short4(v4[base + 3]);
        dot_acc += int(p0.x) + int(p0.y) + int(p0.z) + int(p0.w)
                 + int(p1.x) + int(p1.y) + int(p1.z) + int(p1.w)
                 + int(p2.x) + int(p2.y) + int(p2.z) + int(p2.w)
                 + int(p3.x) + int(p3.y) + int(p3.z) + int(p3.w);
    }
    for (uint i = dim16 * 16; i < dim; i++) {
        dot_acc += int(shared_query[i]) * int(vec[i]);
    }

    float dist = query_norms[qidx] + cand_norm - 2.0f * float(dot_acc);
    distances[qidx * N + cidx] = dist;
}

// ─── FP32 L2 Gather Re-rank ────────────────────────────────────────────
// Exact FP32 L2² over a small set of K candidate slot indices, used as the
// final rerank step for the four HNSW quant variants (PolarQuant INT4,
// TurboQuant INT3, NanoQuant INT2, §7v2 INT8). Replaces a CPU SIMD loop
// when K ≥ GPU_RERANK_THRESHOLD; CPU stays as the fallback below threshold.
//
// Layout: rerank buffer is BFS-ordered FP32 [N_total × dim]; ids[K] are
// slot indices into that buffer. Threadgroup-shared query cache keeps the
// 1536×4=6KB query in tg memory (avoids L1 eviction across K dispatches).
//
// Dispatch: K threadgroups × TG_SIZE threads. Each TG handles one ID,
// strides over dim with float4 loads, reduces (q-v)² into shared mem,
// thread 0 emits the sum to distances[gid].
kernel void fp32_l2_gather_rerank(
    device const float  *query     [[buffer(0)]],   // [dim] fp32 query
    device const float  *rerank    [[buffer(1)]],   // [N_total × dim] BFS-ordered fp32
    device const int    *ids       [[buffer(2)]],   // [K] slot indices
    device       float  *distances [[buffer(3)]],   // [K] output L2² distances
    constant     uint   &dim       [[buffer(4)]],
    constant     uint   &K         [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (tgid >= K) return;

    // Cache query in threadgroup shared memory (max 1536 floats = 6 KB).
    // Reused across all K candidates dispatched concurrently; prevents L1
    // eviction during the rerank scan.
    threadgroup float shared_query[1536];
    for (uint i = lid; i < dim; i += tg_size) {
        shared_query[i] = query[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int slot = ids[tgid];
    if (slot < 0) {
        if (lid == 0) distances[tgid] = INFINITY;
        return;
    }

    device const float *vec = rerank + uint(slot) * dim;

    // float4 vectorised reduction: each thread handles dim/(4·tg_size) chunks.
    // For dim=1536, tg_size=384 → each thread sums exactly 1 float4 (4 elems).
    // For dim=1536, tg_size=256 → each thread sums 1.5 float4s (handled by stride loop).
    float acc = 0.0f;
    uint dim4 = dim >> 2;
    threadgroup const float4 *q4 = (threadgroup const float4 *)shared_query;
    device const float4 *v4 = (device const float4 *)vec;
    for (uint i = lid; i < dim4; i += tg_size) {
        float4 d = q4[i] - v4[i];
        acc += d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w;
    }
    // Tail (dim % 4 != 0)
    for (uint i = dim4 * 4 + lid; i < dim; i += tg_size) {
        float d = shared_query[i] - vec[i];
        acc += d * d;
    }

    // Reduce acc across threadgroup → distances[tgid]
    threadgroup float scratch[1024];   // ≥ max threadgroup size on Apple GPUs
    scratch[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) distances[tgid] = scratch[0];
}

// ─── SDPA Q=1 (Decoder-step Multi-Head Attention, FP32) ────────────────
//
// Single-query scaled-dot-product attention for the `ATTEND.PREFIX.QUERY`
// decode-step path. Beats `mlx.fast.scaled_dot_product_attention` at
// H=8 N=2048 d_head=128 by 1.34-1.55× end-to-end (tests/bench_msl_sdpa_q1.m).
//
// Layout (float4-packed, stride D4 = D/4):
//   Q [H,    D4], K [H, N, D4], V [H, N, D4], O [H, D4]
//
// D parameterization:
//   D_HEAD is set per-PSO via [[function_constant(0)]]. One source kernel
//   specializes to PSOs for D ∈ {64, 96, 128, 160, 192, 256}.
//   D4 = D/4 float4s per row; CHUNK = ⌈D4/32⌉ float4s per lane.
//   D ≤ 128: CHUNK=1, only first D4 lanes active. D > 128: CHUNK=2.
//   Bounds-check `idx < D4` on each chunk; lanes beyond D4 contribute 0.
//
// Dispatch: H threadgroups × (32 * SG_PER_TG) threads.
//   SG_PER_TG = 8 → 64 simdgroups dispatched at H=8 (saturates M-series GPU).
#define SG_PER_TG  8u
#define CHUNK_MAX  4u   // bounds register arrays at compile time; supports D ≤ 512
                        // (D ∈ {32..256} use CHUNK ≤ 2; D=512 uses CHUNK=4. Trailing
                        // chunks are zero-initialized and bounds-checked via `idx < D4`,
                        // so the compiler can dead-code-eliminate them at D≤256 PSOs
                        // since CHUNK is a function constant.)

constant uint D_HEAD [[function_constant(0)]];
constant uint D4    = D_HEAD / 4u;
constant uint CHUNK = (D4 + 31u) / 32u;

// gh #130 §4.2: batched-Q kernels stage K/V tiles in threadgroup memory so all
// SG_PER_TG query rows in a threadgroup reuse each loaded tile (was: 1 query/TG,
// K/V re-streamed from DRAM per TG). TILE_KV_F4 = float4 budget for K+V combined
// (1024 float4 = 16 KB fp32 / 8 KB fp16); TILE_N tokens derived per-PSO to fit.
#define TILE_KV_F4 1024u
constant uint TILE_N = (TILE_KV_F4 / (2u * D4)) > 0u ? (TILE_KV_F4 / (2u * D4)) : 1u;

kernel void sdpa_q1_fp32(
    const device float4* Q [[buffer(0)]],
    const device float4* K [[buffer(1)]],
    const device float4* V [[buffer(2)]],
    device       float4* O [[buffer(3)]],
    constant     uint&   N_tokens [[buffer(4)]],
    constant     float&  scale    [[buffer(5)]],
    constant     uint&   W_window [[buffer(6)]],   // 0 = full attention; >0 = scan only last W tokens
    threadgroup  float4* s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]; host sets length
    uint h     [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup float  s_m[SG_PER_TG];
    threadgroup float  s_l[SG_PER_TG];

    // Load this lane's slice of Q (CHUNK float4 chunks; idle lanes contribute 0).
    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = Q[h * D4 + idx];
    }

    // Sliding-window: scan only the last W_window tokens. W=0 → full attention.
    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin = eff_start + sg * chunk_n;
    uint t_end   = min(t_begin + chunk_n, N_tokens);

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base = K + h * N_tokens * D4;
    const device float4* v_base = V + h * N_tokens * D4;

    for (uint t = t_begin; t < t_end; ++t) {
        // QK dot — each lane contributes its share, simd_sum reduces across the 32 lanes.
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 k = k_base[t * D4 + idx];
                partial += dot(q[c], k);
            }
        }
        float score = simd_sum(partial) * scale;

        float new_m  = max(m, score);
        float factor = fast::exp(m - new_m);
        float exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        // V update — each lane handles its own chunks.
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    // Dynamic tg-memory layout: s_o[sg, c, lane] = s_o[sg*CHUNK*32 + c*32 + lane].
    // CHUNK is a per-PSO function constant, so the host sets length to exactly
    // SG_PER_TG * CHUNK * 32 * sizeof(float4) — no wasted tg memory.
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = float4(0.0f);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = max(M, m_i);
            float a = fast::exp(M   - new_M);
            float b = fast::exp(m_i - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[h * D4 + idx] = O_acc[c] / L;
            }
        }
    }
}

// ─── SDPA Q=1 SPARSE-MASK (gh #60 Phase 2, FP32) ────────────────────────
//
// Sibling to `sdpa_q1_fp32`: same online-softmax shape, same dispatch grid,
// but the inner loop iterates over a caller-supplied per-head index list
// instead of the dense `[eff_start..N_tokens)` range. Each (head, query)
// attends to `counts[h] ≤ K_sparse_max` cached K/V tokens picked by an
// upstream selector (block-mean top-K v1, learned router v2). Mask selection
// is the caller's responsibility — this kernel just consumes the indices.
//
// Compute saving: when K_sparse_max << N_tokens (e.g. 512 picked out of
// 65536), the kernel does ~K_sparse_max/N_tokens of the FLOPs and DRAM reads
// of the dense path. Pion's defensible angle over standalone sparse-attention
// kernels (MInference, Quest): the same `indices[H, K_sparse_max]` array
// also drives V-store I/O reduction via `V.FETCH BATCH <indices>`.
//
// Layout — identical to sdpa_q1_fp32 except for two new buffers:
//   indices [H, K_sparse_max]  int32  — token IDs per head (head-divergent OK)
//   counts  [H]                uint32 — actual K_sparse_h ≤ K_sparse_max per head
//
// Sliding-window combination: if W_window > 0, indices below
// (N_tokens - W_window) are skipped (treated as masked out). This lets a
// caller dispatch one kernel with `dense local window` + `sparse global picks`
// by passing global picks in `indices` and a local window via `W_window`.
//
// Edge cases (caller's responsibility):
//   - counts[h] must be > 0 for all h. Empty mask → l=0 → divide-by-zero NaN.
//   - indices[h, i] must be in [0, N_tokens). Out-of-range → silently skipped
//     so the kernel doesn't crash, but it's a caller bug.
kernel void sdpa_q1_sparse_fp32(
    const device float4* Q             [[buffer(0)]],
    const device float4* K             [[buffer(1)]],
    const device float4* V             [[buffer(2)]],
    device       float4* O             [[buffer(3)]],
    constant     uint&   N_tokens      [[buffer(4)]],
    constant     float&  scale         [[buffer(5)]],
    constant     uint&   W_window      [[buffer(6)]],   // 0 = no clamp; >0 = mask indices < (N-W)
    const device int*    indices       [[buffer(7)]],   // [H_q, K_sparse_max]
    const device uint*   counts        [[buffer(8)]],   // [H_q]
    constant     uint&   K_sparse_max  [[buffer(9)]],
    const device uchar*  head_map      [[buffer(10)]],  // [H_q] — head_map[h_q] = h_kv. NULL/all-zero on non-GQA.
    constant     uint&   H_kv          [[buffer(11)]],  // = H_q on non-GQA paths
    threadgroup  float4* s_o           [[threadgroup(0)]],
    uint h     [[threadgroup_position_in_grid]],        // query head index (0..H_q-1)
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup float  s_m[SG_PER_TG];
    threadgroup float  s_l[SG_PER_TG];

    // GQA: query head `h` reads K/V from KV head head_map[h] (= h on non-GQA).
    uint h_kv = (uint)head_map[h];

    // Load this lane's slice of Q.
    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = Q[h * D4 + idx];
    }

    // Distribute this head's index slice across the SG_PER_TG simdgroups.
    uint K_sparse_h = counts[h];
    uint per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint i_begin    = sg * per_sg;
    uint i_end      = min(i_begin + per_sg, K_sparse_h);

    uint eff_start  = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base   = K + h_kv * N_tokens * D4;
    const device float4* v_base   = V + h_kv * N_tokens * D4;
    const device int*    idx_base = indices + h * K_sparse_max;

    for (uint i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint)idx_i >= N_tokens) continue;
        uint t = (uint)idx_i;
        if (t < eff_start) continue;

        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 k = k_base[t * D4 + idx];
                partial += dot(q[c], k);
            }
        }
        float score = simd_sum(partial) * scale;

        float new_m  = max(m, score);
        float factor = fast::exp(m - new_m);
        float exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = float4(0.0f);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = max(M, m_i);
            float a = fast::exp(M   - new_M);
            float b = fast::exp(m_i - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[h * D4 + idx] = O_acc[c] / L;
            }
        }
    }
}

// ─── SDPA Q=1 SPARSE-MASK (gh #60 Phase 2, FP16-emulation) ──────────────
//
// FP16 sibling to sdpa_q1_sparse_fp32. Same online-softmax structure as
// sdpa_q1_fp16 with indices/counts replacing the dense token range. Used by
// the mlx-lm patch path (vanilla mlx-lm precision parity).
kernel void sdpa_q1_sparse_fp16(
    const device float4* Q             [[buffer(0)]],
    const device half4*  K             [[buffer(1)]],
    const device half4*  V             [[buffer(2)]],
    device       float4* O             [[buffer(3)]],
    constant     uint&   N_tokens      [[buffer(4)]],
    constant     float&  scale         [[buffer(5)]],
    constant     uint&   W_window      [[buffer(6)]],
    const device int*    indices       [[buffer(7)]],
    const device uint*   counts        [[buffer(8)]],
    constant     uint&   K_sparse_max  [[buffer(9)]],
    const device uchar*  head_map      [[buffer(10)]],   // [H_q]; head_map[h_q] = h_kv
    constant     uint&   H_kv          [[buffer(11)]],
    threadgroup  half4*  s_o           [[threadgroup(0)]],
    uint h     [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup half  s_m[SG_PER_TG];
    threadgroup half  s_l[SG_PER_TG];

    uint h_kv = (uint)head_map[h];

    half4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = half4(Q[h * D4 + idx]);
    }

    uint K_sparse_h = counts[h];
    uint per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint i_begin    = sg * per_sg;
    uint i_end      = min(i_begin + per_sg, K_sparse_h);

    uint eff_start  = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    half  m = -HALF_MAX;
    half  l = 0.0h;
    half4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);

    const device half4*  k_base   = K + h_kv * N_tokens * D4;
    const device half4*  v_base   = V + h_kv * N_tokens * D4;
    const device int*    idx_base = indices + h * K_sparse_max;

    for (uint i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint)idx_i >= N_tokens) continue;
        uint t = (uint)idx_i;
        if (t < eff_start) continue;

        // Dot in float (headroom), cast to half post-simd_sum (matches sdpa_q1_fp16).
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 k = k_base[t * D4 + idx];
                partial += float(dot(q[c], k));
            }
        }
        half score = half(simd_sum(partial) * scale);

        half new_m  = max(m, score);
        half factor = fast::exp(m - new_m);
        half exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        half M_acc = -HALF_MAX;
        half L_acc = 0.0h;
        half4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            half m_i = s_m[i];
            half l_i = s_l[i];
            half new_M = max(M_acc, m_i);
            half a = fast::exp(M_acc - new_M);
            half b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[h * D4 + idx] = float4(O_acc[c] / L_acc);
            }
        }
    }
}

// ─── SDPA Q=1 SPARSE-MASK + DENSE SUFFIX FUSED (gh #63 follow-on) ──────
//
// Closes the "wire-mode sparse ignores local suffix" caveat from gh #63
// commit 2be3fc5. Combines:
//   1. Sparse-prefix attention over server-resident K/V at indices[h, *]
//      (same inner loop as sdpa_q1_sparse_fp32).
//   2. Dense-suffix attention over caller-supplied K_suf, V_suf (the
//      consumer's locally-decoded suffix K/V).
//   3. Online-softmax merge across the two phases via the running (m, l, o)
//      accumulator — single pass, no separate LSE writeback.
//
// For M=1 decode all S_suf suffix tokens are visible (no causal mask
// needed; the suffix here is the consumer's growing decode buffer, which
// the query already saw all of). For M>1 the caller would need a different
// kernel; this fused variant is M=1-only.
//
// Grid: H_q threadgroups × (32 * SG_PER_TG) threads. K_suf/V_suf at H_kv
// heads (GQA-aware via head_map). Suffix loop distributes S_suf across
// the SG_PER_TG simdgroups (each sg processes S_suf/SG_PER_TG suffix
// tokens) — same split pattern as the prefix index loop.
//
// Replaces the wire-mode sparse path's "skip suffix, accept ~quality"
// hack: now suffix attention is fully accounted for via the same softmax
// distribution as prefix.
kernel void sdpa_q1_sparse_fused_fp32(
    const device float4* Q             [[buffer(0)]],
    const device float4* K             [[buffer(1)]],   // [H_kv, N, D4]
    const device float4* V             [[buffer(2)]],
    device       float4* O             [[buffer(3)]],
    constant     uint&   N_tokens      [[buffer(4)]],
    constant     float&  scale         [[buffer(5)]],
    constant     uint&   W_window      [[buffer(6)]],
    const device int*    indices       [[buffer(7)]],
    const device uint*   counts        [[buffer(8)]],
    constant     uint&   K_sparse_max  [[buffer(9)]],
    const device uchar*  head_map      [[buffer(10)]],
    constant     uint&   H_kv          [[buffer(11)]],
    const device float4* K_suf         [[buffer(12)]],   // [H_kv, S_suf, D4]
    const device float4* V_suf         [[buffer(13)]],
    constant     uint&   S_suf         [[buffer(14)]],
    threadgroup  float4* s_o           [[threadgroup(0)]],
    uint h     [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup float  s_m[SG_PER_TG];
    threadgroup float  s_l[SG_PER_TG];

    uint h_kv = (uint)head_map[h];

    // Load this lane's slice of Q.
    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = Q[h * D4 + idx];
    }

    // ── Sparse prefix loop ──
    uint K_sparse_h = counts[h];
    uint per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint i_begin    = sg * per_sg;
    uint i_end      = min(i_begin + per_sg, K_sparse_h);

    uint eff_start  = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base   = K + h_kv * N_tokens * D4;
    const device float4* v_base   = V + h_kv * N_tokens * D4;
    const device int*    idx_base = indices + h * K_sparse_max;

    for (uint i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint)idx_i >= N_tokens) continue;
        uint t = (uint)idx_i;
        if (t < eff_start) continue;

        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 k = k_base[t * D4 + idx];
                partial += dot(q[c], k);
            }
        }
        float score = simd_sum(partial) * scale;

        float new_m  = max(m, score);
        float factor = fast::exp(m - new_m);
        float exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    // ── Dense suffix loop ──
    if (S_suf > 0u) {
        uint chunk_s   = (S_suf + SG_PER_TG - 1u) / SG_PER_TG;
        uint ts_begin  = sg * chunk_s;
        uint ts_end    = min(ts_begin + chunk_s, S_suf);
        const device float4* ks_base = K_suf + h_kv * S_suf * D4;
        const device float4* vs_base = V_suf + h_kv * S_suf * D4;
        for (uint t = ts_begin; t < ts_end; ++t) {
            float partial = 0.0f;
            for (uint c = 0u; c < CHUNK; ++c) {
                uint idx = c * 32u + lane;
                if (idx < D4) {
                    float4 k = ks_base[t * D4 + idx];
                    partial += dot(q[c], k);
                }
            }
            float score = simd_sum(partial) * scale;

            float new_m  = max(m, score);
            float factor = fast::exp(m - new_m);
            float exp_s  = fast::exp(score - new_m);
            m = new_m;
            l = l * factor + exp_s;

            for (uint c = 0u; c < CHUNK; ++c) {
                uint idx = c * 32u + lane;
                if (idx < D4) {
                    float4 v = vs_base[t * D4 + idx];
                    o[c] = o[c] * factor + exp_s * v;
                }
            }
        }
    }

    // ── Merge across simdgroups ──
    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float M = -INFINITY;
        float L = 0.0f;
        float4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = float4(0.0f);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = max(M, m_i);
            float a = fast::exp(M   - new_M);
            float b = fast::exp(m_i - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L = L * a + l_i * b;
            M = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[h * D4 + idx] = O_acc[c] / L;
            }
        }
    }
}

// ─── SDPA Q=1 SPARSE-MASK + DENSE SUFFIX FUSED (FP16) ─────────────────
kernel void sdpa_q1_sparse_fused_fp16(
    const device float4* Q             [[buffer(0)]],
    const device half4*  K             [[buffer(1)]],
    const device half4*  V             [[buffer(2)]],
    device       float4* O             [[buffer(3)]],
    constant     uint&   N_tokens      [[buffer(4)]],
    constant     float&  scale         [[buffer(5)]],
    constant     uint&   W_window      [[buffer(6)]],
    const device int*    indices       [[buffer(7)]],
    const device uint*   counts        [[buffer(8)]],
    constant     uint&   K_sparse_max  [[buffer(9)]],
    const device uchar*  head_map      [[buffer(10)]],
    constant     uint&   H_kv          [[buffer(11)]],
    const device float4* K_suf         [[buffer(12)]],
    const device float4* V_suf         [[buffer(13)]],
    constant     uint&   S_suf         [[buffer(14)]],
    threadgroup  half4*  s_o           [[threadgroup(0)]],
    uint h     [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup half  s_m[SG_PER_TG];
    threadgroup half  s_l[SG_PER_TG];

    uint h_kv = (uint)head_map[h];

    half4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = half4(Q[h * D4 + idx]);
    }

    uint K_sparse_h = counts[h];
    uint per_sg     = (K_sparse_h + SG_PER_TG - 1u) / SG_PER_TG;
    uint i_begin    = sg * per_sg;
    uint i_end      = min(i_begin + per_sg, K_sparse_h);

    uint eff_start  = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    half  m = -HALF_MAX;
    half  l = 0.0h;
    half4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);

    const device half4*  k_base   = K + h_kv * N_tokens * D4;
    const device half4*  v_base   = V + h_kv * N_tokens * D4;
    const device int*    idx_base = indices + h * K_sparse_max;

    // Sparse prefix loop
    for (uint i = i_begin; i < i_end; ++i) {
        int idx_i = idx_base[i];
        if (idx_i < 0 || (uint)idx_i >= N_tokens) continue;
        uint t = (uint)idx_i;
        if (t < eff_start) continue;

        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 k = k_base[t * D4 + idx];
                partial += float(dot(q[c], k));
            }
        }
        half score = half(simd_sum(partial) * scale);

        half new_m  = max(m, score);
        half factor = fast::exp(m - new_m);
        half exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    // Dense suffix loop
    if (S_suf > 0u) {
        uint chunk_s   = (S_suf + SG_PER_TG - 1u) / SG_PER_TG;
        uint ts_begin  = sg * chunk_s;
        uint ts_end    = min(ts_begin + chunk_s, S_suf);
        const device float4* ks_base = K_suf + h_kv * S_suf * D4;
        const device float4* vs_base = V_suf + h_kv * S_suf * D4;
        for (uint t = ts_begin; t < ts_end; ++t) {
            float partial = 0.0f;
            for (uint c = 0u; c < CHUNK; ++c) {
                uint idx = c * 32u + lane;
                if (idx < D4) {
                    half4 k = half4(ks_base[t * D4 + idx]);
                    partial += float(dot(q[c], k));
                }
            }
            half score = half(simd_sum(partial) * scale);

            half new_m  = max(m, score);
            half factor = fast::exp(m - new_m);
            half exp_s  = fast::exp(score - new_m);
            m = new_m;
            l = l * factor + exp_s;

            for (uint c = 0u; c < CHUNK; ++c) {
                uint idx = c * 32u + lane;
                if (idx < D4) {
                    half4 v = half4(vs_base[t * D4 + idx]);
                    o[c] = o[c] * factor + exp_s * v;
                }
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        half M_acc = -HALF_MAX;
        half L_acc = 0.0h;
        half4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            half m_i = s_m[i];
            half l_i = s_l[i];
            half new_M = max(M_acc, m_i);
            half a = fast::exp(M_acc - new_M);
            half b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[h * D4 + idx] = float4(O_acc[c] / L_acc);
            }
        }
    }
}

// ─── SDPA Batched-Q (Prefill / TTFT path, FP32) ─────────────────────────
//
// M>1 batched-query attention for `ATTEND.PREFIX.QUERY` .
// Reuses the same online-softmax structure as `sdpa_q1_fp32`.
//
// Layout:
//   Q   [H, M, D4]     K/V [H, N, D4]     O [H, M, D4]     LSE [H, M]
//   LSE = rowwise log-sum-exp = max + log(sum_exp), required for the online-
//   softmax merge between cached prefix and locally computed suffix attention.
//
// gh #130 §4.2: TWO kernels with an M crossover picked by the host:
//   • sdpa_batched_q_fp32       — small M: one query row per TG, N tokens split
//     across SG_PER_TG simdgroups (max token parallelism when there aren't
//     enough query rows to fill the simdgroups). Grid H × M.
//   • sdpa_batched_q_tiled_fp32 — large M: SG_PER_TG query rows per TG, K/V
//     staged in threadgroup memory and reused across all rows in the TG (each
//     K/V tile read from DRAM once per TG, not once per row). Grid H × ⌈M/8⌉.
//   Measured crossover ~M=16 (D=128,N=2048): tiled is 1.7× at M=128, but ~0.7×
//   at M≤8 due to idle simdgroups — hence the split. Both hold cosine ≈ 1.0.
kernel void sdpa_batched_q_fp32(
    const device float4* Q   [[buffer(0)]],    // [H, M, D4]
    const device float4* K   [[buffer(1)]],    // [H, N, D4]
    const device float4* V   [[buffer(2)]],    // [H, N, D4]
    device       float4* O   [[buffer(3)]],    // [H, M, D4]
    device       float*  LSE [[buffer(4)]],    // [H, M]
    constant     uint&   N_tokens [[buffer(5)]],
    constant     float&  scale    [[buffer(6)]],
    constant     uint&   M_rows   [[buffer(7)]],
    constant     uint&   W_window [[buffer(8)]],   // 0 = full attention; >0 = scan only last W tokens
    threadgroup  float4* s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h  = tg_pos.x;
    uint mq = tg_pos.y;

    threadgroup float  s_m[SG_PER_TG];
    threadgroup float  s_l[SG_PER_TG];

    // Q row for this (head, query_row).
    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    uint q_base = (h * M_rows + mq) * D4;
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = Q[q_base + idx];
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin = eff_start + sg * chunk_n;
    uint t_end   = min(t_begin + chunk_n, N_tokens);

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base = K + h * N_tokens * D4;
    const device float4* v_base = V + h * N_tokens * D4;

    for (uint t = t_begin; t < t_end; ++t) {
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 k = k_base[t * D4 + idx];
                partial += dot(q[c], k);
            }
        }
        float score = simd_sum(partial) * scale;

        float new_m  = max(m, score);
        float factor = fast::exp(m - new_m);
        float exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float M_acc = -INFINITY;
        float L_acc = 0.0f;
        float4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = float4(0.0f);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = max(M_acc, m_i);
            float a = fast::exp(M_acc - new_M);
            float b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        uint o_base = (h * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[o_base + idx] = O_acc[c] / L_acc;
            }
        }
        if (lane == 0u) {
            LSE[h * M_rows + mq] = M_acc + log(L_acc);
        }
    }
}

// gh #130 §4.2: large-M tiled variant (see the crossover note above).
kernel void sdpa_batched_q_tiled_fp32(
    const device float4* Q   [[buffer(0)]],    // [H, M, D4]
    const device float4* K   [[buffer(1)]],    // [H, N, D4]
    const device float4* V   [[buffer(2)]],    // [H, N, D4]
    device       float4* O   [[buffer(3)]],    // [H, M, D4]
    device       float*  LSE [[buffer(4)]],    // [H, M]
    constant     uint&   N_tokens [[buffer(5)]],
    constant     float&  scale    [[buffer(6)]],
    constant     uint&   M_rows   [[buffer(7)]],
    constant     uint&   W_window [[buffer(8)]],   // 0 = full attention; >0 = scan only last W tokens
    threadgroup  float4* kv_tile  [[threadgroup(0)]],  // [2 * TILE_N * D4]: K tile then V tile
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h  = tg_pos.x;
    uint mq = tg_pos.y * SG_PER_TG + sg;   // this simdgroup's query row
    bool active = (mq < M_rows);

    threadgroup float4* K_tile = kv_tile;                 // [TILE_N * D4]
    threadgroup float4* V_tile = kv_tile + TILE_N * D4;   // [TILE_N * D4]
    uint tid = sg * 32u + lane;                           // 0..(SG_PER_TG*32-1)

    // Q row for this simdgroup's query (idle lanes / inactive rows contribute 0).
    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    if (active) {
        uint q_base = (h * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) q[c] = Q[q_base + idx];
        }
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base = K + h * N_tokens * D4;
    const device float4* v_base = V + h * N_tokens * D4;

    // Stream K/V in TILE_N-token tiles resident in threadgroup memory. All
    // SG_PER_TG*32 threads cooperate on the load + barriers regardless of
    // `active`; only the softmax and writeback are per-query.
    for (uint tile0 = eff_start; tile0 < N_tokens; tile0 += TILE_N) {
        uint tile_n = min(TILE_N, N_tokens - tile0);
        uint total  = tile_n * D4;   // float4 count for K (and again for V)
        for (uint i = tid; i < total; i += SG_PER_TG * 32u) {
            K_tile[i] = k_base[tile0 * D4 + i];
            V_tile[i] = v_base[tile0 * D4 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint tl = 0u; tl < tile_n; ++tl) {
                float partial = 0.0f;
                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) partial += dot(q[c], K_tile[tl * D4 + idx]);
                }
                float score = simd_sum(partial) * scale;

                float new_m  = max(m, score);
                float factor = fast::exp(m - new_m);
                float exp_s  = fast::exp(score - new_m);
                m = new_m;
                l = l * factor + exp_s;

                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) o[c] = o[c] * factor + exp_s * V_tile[tl * D4 + idx];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // before the tile is overwritten
    }

    if (active) {
        uint o_base = (h * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) O[o_base + idx] = o[c] / l;
        }
        if (lane == 0u) LSE[h * M_rows + mq] = m + log(l);
    }
}

// ─── SDPA Q=1 (FP16-emulation, matches vanilla mlx-lm precision) ────────
//
// Same structure as `sdpa_q1_fp32` but every internal value is `half`.
// Wire format stays fp32 (no protocol change) — Q/K/V/O cast at load/store.
// Diagnosis: Pion's fp32
// kernel is more accurate than vanilla mlx-lm's fp16 attention by ~0.1%
// relative, which compounds across decode steps and flips argmax at token
// 8 (45% token agreement vs 50% threshold). This kernel matches vanilla
// fp16 numerically.
//
// Stability: K range observed ±12, dot D=64 → max |partial| ≤ ~9000;
// simd_sum across 32 lanes ≤ ~290000 — slightly above fp16 max (65504).
// Therefore: dot products accumulate in float; cast to half AFTER simd_sum.
// Online softmax stats (m, l) in half — matches what vanilla mlx-lm does.
//
// The kernel reuses the same D_HEAD function constant — one source, one
// PSO per supported D, just like the fp32 version.

// gh #398: under --metal-attention-fp16 the session K/V are STORED as half
// (pion_metal_sdpa_store_kv converts once on the host, round-to-nearest-even),
// so all five fp16 kernels read half4 prefix K/V: half the bytes per token.
// The kernels used to convert float4 to half4 on every load, and that cast
// differs from the host's in the last bit of some elements, so the output is
// not bit-identical to before: cosine against the fp32 CPU reference moved
// 0.9999965 -> 0.9999966 at N=2048 and 0.9996163 -> 0.9996138 at N=28672 (H=8,
// D=128), and Llama-3.2-1B decode still matches vanilla mlx-lm 40/40 tokens.
// The fused kernels' suffix K/V arrive per call and stay float4.
kernel void sdpa_q1_fp16(
    const device float4* Q [[buffer(0)]],
    const device half4*  K [[buffer(1)]],
    const device half4*  V [[buffer(2)]],
    device       float4* O [[buffer(3)]],
    constant     uint&   N_tokens [[buffer(4)]],
    constant     float&  scale    [[buffer(5)]],
    constant     uint&   W_window [[buffer(6)]],   // 0 = full attention; >0 = scan only last W tokens
    threadgroup  half4*  s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]; host sets length
    uint h     [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    threadgroup half  s_m[SG_PER_TG];
    threadgroup half  s_l[SG_PER_TG];

    half4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = half4(Q[h * D4 + idx]);
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin = eff_start + sg * chunk_n;
    uint t_end   = min(t_begin + chunk_n, N_tokens);

    half  m = -HALF_MAX;
    half  l = 0.0h;
    half4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);

    const device half4*  k_base = K + h * N_tokens * D4;
    const device half4*  v_base = V + h * N_tokens * D4;

    for (uint t = t_begin; t < t_end; ++t) {
        // Dot accumulates in float for headroom (D=64 × |q*k| up to 9000 per-lane,
        // simd_sum across 32 lanes up to ~290000 — exceeds half max). Cast to
        // half after simd_sum + scale; from this point on, math is half.
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 k = k_base[t * D4 + idx];
                partial += float(dot(q[c], k));
            }
        }
        half score = half(simd_sum(partial) * scale);

        half new_m  = max(m, score);
        half factor = fast::exp(m - new_m);
        half exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    // Dynamic tg-memory layout: s_o[sg, c, lane] = s_o[sg*CHUNK*32 + c*32 + lane].
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        half M_acc = -HALF_MAX;
        half L_acc = 0.0h;
        half4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            half  m_i = s_m[i];
            half  l_i = s_l[i];
            half  new_M = max(M_acc, m_i);
            half  a = fast::exp(M_acc - new_M);
            half  b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                // Output: cast back to fp32 for wire compatibility. The values
                // carry only fp16 precision; the upcast is information-preserving.
                O[h * D4 + idx] = float4(O_acc[c] / L_acc);
            }
        }
    }
}

// ─── SDPA Batched-Q (FP16-emulation) ────────────────────────────────────
//
// Same precision contract as `sdpa_q1_fp16` but for the M>1 prefill path.
// Wire-format K/V/Q stay fp32; cast to half on load. LSE output stays fp32
// for online softmax merge — vanilla mlx-lm computes the merge in fp16 too,
// but the wire trailer shape is fixed at fp32 across both kernels. Caller
// can downcast LSE on receive if bit-exact merge is required.
kernel void sdpa_batched_q_fp16(
    const device float4* Q   [[buffer(0)]],
    const device half4*  K   [[buffer(1)]],
    const device half4*  V   [[buffer(2)]],
    device       float4* O   [[buffer(3)]],
    device       float*  LSE [[buffer(4)]],
    constant     uint&   N_tokens [[buffer(5)]],
    constant     float&  scale    [[buffer(6)]],
    constant     uint&   M_rows   [[buffer(7)]],
    constant     uint&   W_window [[buffer(8)]],   // 0 = full attention; >0 = scan only last W tokens
    threadgroup  half4*  s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h  = tg_pos.x;
    uint mq = tg_pos.y;

    threadgroup half  s_m[SG_PER_TG];
    threadgroup half  s_l[SG_PER_TG];

    half4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);
    uint q_base = (h * M_rows + mq) * D4;
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = half4(Q[q_base + idx]);
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin = eff_start + sg * chunk_n;
    uint t_end   = min(t_begin + chunk_n, N_tokens);

    half  m = -HALF_MAX;
    half  l = 0.0h;
    half4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);

    const device half4*  k_base = K + h * N_tokens * D4;
    const device half4*  v_base = V + h * N_tokens * D4;

    for (uint t = t_begin; t < t_end; ++t) {
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 k = k_base[t * D4 + idx];
                partial += float(dot(q[c], k));
            }
        }
        half score = half(simd_sum(partial) * scale);

        half new_m  = max(m, score);
        half factor = fast::exp(m - new_m);
        half exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        half M_acc = -HALF_MAX;
        half L_acc = 0.0h;
        half4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            half m_i = s_m[i];
            half l_i = s_l[i];
            half new_M = max(M_acc, m_i);
            half a = fast::exp(M_acc - new_M);
            half b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        uint o_base = (h * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[o_base + idx] = float4(O_acc[c] / L_acc);
            }
        }
        if (lane == 0u) {
            LSE[h * M_rows + mq] = float(M_acc) + log(float(L_acc));
        }
    }
}

// ─── SDPA Batched-Q FUSED (Stage-2 TTFT, gh #49) ────────────────────────
//
// Server-side fusion of suffix SDPA + online-softmax merge with the
// resident prefix K/V. One online softmax over (prefix ∪ suffix) per
// query row — mathematically identical to "compute prefix attention →
// compute suffix attention with LSE → online-merge two-step", but the
// running (m, l, o) accumulator never leaves registers between the
// prefix and suffix passes. Replaces the host-side path in
// pion-vllm-mlx/pion_vllm_mlx/mlx_lm_patch.py (_suffix_sdpa_with_lse +
// _online_softmax_merge + numpy↔MLX conversions) which was measured at
// ~50 ms / 16 layers.
//
// GQA: head_map[h_q] = h_kv. Q is indexed by query head; K/V (and
// K_suf/V_suf) by kv head. Caller no longer needs to reshape Q for GQA.
//
// Causal mask within suffix: query row mq sees suffix positions
// [0 .. (S_suf - M_rows + mq)] inclusive. Prefix is fully visible.
//
// Sliding window applies to PREFIX ONLY (suffix is always small and
// fully attended). Combining --fa-window with fused suffix is
// supported but treats W_window as the prefix tail bound only.
kernel void sdpa_batched_q_fused_fp32(
    const device float4* Q       [[buffer(0)]],   // [H_q, M, D4]
    const device float4* K       [[buffer(1)]],   // [H_kv, N, D4]
    const device float4* V       [[buffer(2)]],   // [H_kv, N, D4]
    const device float4* K_suf   [[buffer(3)]],   // [H_kv, S_suf, D4]
    const device float4* V_suf   [[buffer(4)]],   // [H_kv, S_suf, D4]
    device       float4* O       [[buffer(5)]],   // [H_q, M, D4]
    const device uchar*  head_map [[buffer(6)]],  // [H_q]
    constant     uint&   N_tokens [[buffer(7)]],
    constant     uint&   S_suf    [[buffer(8)]],
    constant     float&  scale    [[buffer(9)]],
    constant     uint&   M_rows   [[buffer(10)]],
    constant     uint&   W_window [[buffer(11)]],
    threadgroup  float4* s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h_q = tg_pos.x;
    uint mq  = tg_pos.y;
    uint h_kv = (uint)head_map[h_q];

    threadgroup float  s_m[SG_PER_TG];
    threadgroup float  s_l[SG_PER_TG];

    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    uint q_base = (h_q * M_rows + mq) * D4;
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = Q[q_base + idx];
    }

    // Prefix range (sliding-window tail, applied only to prefix).
    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n   = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin   = eff_start + sg * chunk_n;
    uint t_end     = min(t_begin + chunk_n, N_tokens);

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    const device float4* k_base = K + h_kv * N_tokens * D4;
    const device float4* v_base = V + h_kv * N_tokens * D4;

    // Prefix loop — no mask.
    for (uint t = t_begin; t < t_end; ++t) {
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 k = k_base[t * D4 + idx];
                partial += dot(q[c], k);
            }
        }
        float score = simd_sum(partial) * scale;

        float new_m  = max(m, score);
        float factor = fast::exp(m - new_m);
        float exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                float4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    // Suffix loop — causal-within-suffix.
    if (S_suf > 0u) {
        uint chunk_s   = (S_suf + SG_PER_TG - 1u) / SG_PER_TG;
        uint ts_begin  = sg * chunk_s;
        uint ts_end    = min(ts_begin + chunk_s, S_suf);
        uint suf_off   = (S_suf > M_rows) ? (S_suf - M_rows) : 0u;
        uint causal_lo = suf_off + mq;          // last visible suffix index (inclusive)
        if (ts_begin <= causal_lo && ts_begin < ts_end) {
            uint ts_stop = min(ts_end, causal_lo + 1u);
            const device float4* ks_base = K_suf + h_kv * S_suf * D4;
            const device float4* vs_base = V_suf + h_kv * S_suf * D4;
            for (uint t = ts_begin; t < ts_stop; ++t) {
                float partial = 0.0f;
                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) {
                        float4 k = ks_base[t * D4 + idx];
                        partial += dot(q[c], k);
                    }
                }
                float score = simd_sum(partial) * scale;

                float new_m  = max(m, score);
                float factor = fast::exp(m - new_m);
                float exp_s  = fast::exp(score - new_m);
                m = new_m;
                l = l * factor + exp_s;

                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) {
                        float4 v = vs_base[t * D4 + idx];
                        o[c] = o[c] * factor + exp_s * v;
                    }
                }
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float M_acc = -INFINITY;
        float L_acc = 0.0f;
        float4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = float4(0.0f);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            float m_i = s_m[i];
            float l_i = s_l[i];
            float new_M = max(M_acc, m_i);
            float a = fast::exp(M_acc - new_M);
            float b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        uint o_base = (h_q * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[o_base + idx] = O_acc[c] / L_acc;
            }
        }
    }
}

// gh #130 §4.2: large-M tiled variant of the fused kernel. SG_PER_TG query rows
// per TG all share the same h_q → same h_kv, so the prefix K/V and suffix
// K_suf/V_suf tiles are reused across all rows (each read from DRAM once per TG).
// Each simdgroup runs a single-pass online softmax over its query — prefix
// unmasked, suffix causal per row — no cross-simdgroup merge and no LSE (the
// prefix∪suffix merge is internal). Host picks this only for large M.
kernel void sdpa_batched_q_tiled_fused_fp32(
    const device float4* Q       [[buffer(0)]],   // [H_q, M, D4]
    const device float4* K       [[buffer(1)]],   // [H_kv, N, D4]
    const device float4* V       [[buffer(2)]],   // [H_kv, N, D4]
    const device float4* K_suf   [[buffer(3)]],   // [H_kv, S_suf, D4]
    const device float4* V_suf   [[buffer(4)]],   // [H_kv, S_suf, D4]
    device       float4* O       [[buffer(5)]],   // [H_q, M, D4]
    const device uchar*  head_map [[buffer(6)]],  // [H_q]
    constant     uint&   N_tokens [[buffer(7)]],
    constant     uint&   S_suf    [[buffer(8)]],
    constant     float&  scale    [[buffer(9)]],
    constant     uint&   M_rows   [[buffer(10)]],
    constant     uint&   W_window [[buffer(11)]],
    threadgroup  float4* kv_tile  [[threadgroup(0)]],  // [2 * TILE_N * D4]: K tile then V tile
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h_q  = tg_pos.x;
    uint mq   = tg_pos.y * SG_PER_TG + sg;
    bool active = (mq < M_rows);
    uint h_kv = (uint)head_map[h_q];

    threadgroup float4* K_tile = kv_tile;
    threadgroup float4* V_tile = kv_tile + TILE_N * D4;
    uint tid = sg * 32u + lane;

    float4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = float4(0.0f);
    if (active) {
        uint q_base = (h_q * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) q[c] = Q[q_base + idx];
        }
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;

    float  m = -INFINITY;
    float  l = 0.0f;
    float4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = float4(0.0f);

    // Prefix — unmasked; every row attends to [eff_start, N).
    const device float4* k_base = K + h_kv * N_tokens * D4;
    const device float4* v_base = V + h_kv * N_tokens * D4;
    for (uint tile0 = eff_start; tile0 < N_tokens; tile0 += TILE_N) {
        uint tile_n = min(TILE_N, N_tokens - tile0);
        uint total  = tile_n * D4;
        for (uint i = tid; i < total; i += SG_PER_TG * 32u) {
            K_tile[i] = k_base[tile0 * D4 + i];
            V_tile[i] = v_base[tile0 * D4 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint tl = 0u; tl < tile_n; ++tl) {
                float partial = 0.0f;
                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) partial += dot(q[c], K_tile[tl * D4 + idx]);
                }
                float score = simd_sum(partial) * scale;
                float new_m  = max(m, score);
                float factor = fast::exp(m - new_m);
                float exp_s  = fast::exp(score - new_m);
                m = new_m;
                l = l * factor + exp_s;
                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) o[c] = o[c] * factor + exp_s * V_tile[tl * D4 + idx];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Suffix — causal per row: row mq sees suffix indices [0, suf_off + mq].
    if (S_suf > 0u) {
        uint suf_off   = (S_suf > M_rows) ? (S_suf - M_rows) : 0u;
        uint causal_lo = suf_off + mq;   // inclusive; simdgroup-uniform (all lanes share mq)
        const device float4* ks_base = K_suf + h_kv * S_suf * D4;
        const device float4* vs_base = V_suf + h_kv * S_suf * D4;
        for (uint tile0 = 0u; tile0 < S_suf; tile0 += TILE_N) {
            uint tile_n = min(TILE_N, S_suf - tile0);
            uint total  = tile_n * D4;
            for (uint i = tid; i < total; i += SG_PER_TG * 32u) {
                K_tile[i] = ks_base[tile0 * D4 + i];
                V_tile[i] = vs_base[tile0 * D4 + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (uint tl = 0u; tl < tile_n; ++tl) {
                    if (tile0 + tl > causal_lo) break;   // uniform across the simdgroup
                    float partial = 0.0f;
                    for (uint c = 0u; c < CHUNK; ++c) {
                        uint idx = c * 32u + lane;
                        if (idx < D4) partial += dot(q[c], K_tile[tl * D4 + idx]);
                    }
                    float score = simd_sum(partial) * scale;
                    float new_m  = max(m, score);
                    float factor = fast::exp(m - new_m);
                    float exp_s  = fast::exp(score - new_m);
                    m = new_m;
                    l = l * factor + exp_s;
                    for (uint c = 0u; c < CHUNK; ++c) {
                        uint idx = c * 32u + lane;
                        if (idx < D4) o[c] = o[c] * factor + exp_s * V_tile[tl * D4 + idx];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (active) {
        uint o_base = (h_q * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) O[o_base + idx] = o[c] / l;
        }
    }
}

// FP16 sibling — same structure, half-precision math (matches vanilla
// mlx-lm precision; numerical headroom matches sdpa_q1_fp16).
kernel void sdpa_batched_q_fused_fp16(
    const device float4* Q       [[buffer(0)]],
    const device half4*  K       [[buffer(1)]],
    const device half4*  V       [[buffer(2)]],
    const device float4* K_suf   [[buffer(3)]],
    const device float4* V_suf   [[buffer(4)]],
    device       float4* O       [[buffer(5)]],
    const device uchar*  head_map [[buffer(6)]],
    constant     uint&   N_tokens [[buffer(7)]],
    constant     uint&   S_suf    [[buffer(8)]],
    constant     float&  scale    [[buffer(9)]],
    constant     uint&   M_rows   [[buffer(10)]],
    constant     uint&   W_window [[buffer(11)]],
    threadgroup  half4*  s_o      [[threadgroup(0)]],  // [SG_PER_TG * CHUNK * 32]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    uint h_q = tg_pos.x;
    uint mq  = tg_pos.y;
    uint h_kv = (uint)head_map[h_q];

    threadgroup half  s_m[SG_PER_TG];
    threadgroup half  s_l[SG_PER_TG];

    half4 q[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);
    uint q_base = (h_q * M_rows + mq) * D4;
    for (uint c = 0u; c < CHUNK; ++c) {
        uint idx = c * 32u + lane;
        if (idx < D4) q[c] = half4(Q[q_base + idx]);
    }

    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u;
    uint eff_n     = N_tokens - eff_start;
    uint chunk_n   = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;
    uint t_begin   = eff_start + sg * chunk_n;
    uint t_end     = min(t_begin + chunk_n, N_tokens);

    half  m = -HALF_MAX;
    half  l = 0.0h;
    half4 o[CHUNK_MAX];
    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);

    const device half4*  k_base = K + h_kv * N_tokens * D4;
    const device half4*  v_base = V + h_kv * N_tokens * D4;

    for (uint t = t_begin; t < t_end; ++t) {
        float partial = 0.0f;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 k = k_base[t * D4 + idx];
                partial += float(dot(q[c], k));
            }
        }
        half score = half(simd_sum(partial) * scale);

        half new_m  = max(m, score);
        half factor = fast::exp(m - new_m);
        half exp_s  = fast::exp(score - new_m);
        m = new_m;
        l = l * factor + exp_s;

        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                half4 v = v_base[t * D4 + idx];
                o[c] = o[c] * factor + exp_s * v;
            }
        }
    }

    if (S_suf > 0u) {
        uint chunk_s   = (S_suf + SG_PER_TG - 1u) / SG_PER_TG;
        uint ts_begin  = sg * chunk_s;
        uint ts_end    = min(ts_begin + chunk_s, S_suf);
        uint suf_off   = (S_suf > M_rows) ? (S_suf - M_rows) : 0u;
        uint causal_lo = suf_off + mq;
        if (ts_begin <= causal_lo && ts_begin < ts_end) {
            uint ts_stop = min(ts_end, causal_lo + 1u);
            const device float4* ks_base = K_suf + h_kv * S_suf * D4;
            const device float4* vs_base = V_suf + h_kv * S_suf * D4;
            for (uint t = ts_begin; t < ts_stop; ++t) {
                float partial = 0.0f;
                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) {
                        half4 k = half4(ks_base[t * D4 + idx]);
                        partial += float(dot(q[c], k));
                    }
                }
                half score = half(simd_sum(partial) * scale);

                half new_m  = max(m, score);
                half factor = fast::exp(m - new_m);
                half exp_s  = fast::exp(score - new_m);
                m = new_m;
                l = l * factor + exp_s;

                for (uint c = 0u; c < CHUNK; ++c) {
                    uint idx = c * 32u + lane;
                    if (idx < D4) {
                        half4 v = half4(vs_base[t * D4 + idx]);
                        o[c] = o[c] * factor + exp_s * v;
                    }
                }
            }
        }
    }

    if (lane == 0u) {
        s_m[sg] = m;
        s_l[sg] = l;
    }
    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        half M_acc = -HALF_MAX;
        half L_acc = 0.0h;
        half4 O_acc[CHUNK_MAX];
        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);
        for (uint i = 0u; i < SG_PER_TG; ++i) {
            half m_i = s_m[i];
            half l_i = s_l[i];
            half new_M = max(M_acc, m_i);
            half a = fast::exp(M_acc - new_M);
            half b = fast::exp(m_i   - new_M);
            for (uint c = 0u; c < CHUNK; ++c) {
                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b;
            }
            L_acc = L_acc * a + l_i * b;
            M_acc = new_M;
        }
        uint o_base = (h_q * M_rows + mq) * D4;
        for (uint c = 0u; c < CHUNK; ++c) {
            uint idx = c * 32u + lane;
            if (idx < D4) {
                O[o_base + idx] = float4(O_acc[c] / L_acc);
            }
        }
    }
}
