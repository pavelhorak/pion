// UMMA Phase 0 — Metal Compute Kernels for Hardware Characterization
// Experiments 0.1 (bandwidth), 0.2 (dispatch latency), 0.3 (SLC probing)

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Exp 0.1: GPU bandwidth — simple copy kernel
// ---------------------------------------------------------------------------
kernel void copy_bandwidth(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    dst[tid] = src[tid];
}

// Triad: dst[i] = src_a[i] + scalar * src_b[i]
kernel void triad_bandwidth(
    device const float* src_a  [[buffer(0)]],
    device const float* src_b  [[buffer(1)]],
    device float*       dst    [[buffer(2)]],
    constant float&     scalar [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    dst[tid] = src_a[tid] + scalar * src_b[tid];
}

// Read-only bandwidth: sum reduction (prevents compiler from eliding reads)
kernel void read_bandwidth(
    device const float4* src   [[buffer(0)]],
    device float*        out   [[buffer(1)]],
    uint tid  [[thread_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]]
) {
    float4 acc = src[tid];
    // Write one value per threadgroup to prevent elision
    if (tid % tpg == 0) {
        out[gid] = acc.x + acc.y + acc.z + acc.w;
    }
}

// ---------------------------------------------------------------------------
// Exp 0.2: Dispatch latency kernels
// ---------------------------------------------------------------------------

// Empty kernel — measures pure dispatch overhead
kernel void empty_kernel() {
    // intentionally empty
}

// Trivial kernel — 1 write
kernel void trivial_kernel(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = 1.0f;
}

// GEMV: output[row] = dot(matrix[row], vec)  — (N rows × D cols)
// Each thread computes one output element
kernel void gemv_kernel(
    device const float*  matrix [[buffer(0)]],  // N × D row-major
    device const float*  vec    [[buffer(1)]],   // D
    device float*        output [[buffer(2)]],   // N
    constant uint&       D      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    device const float* row = matrix + tid * D;
    for (uint j = 0; j < D; j += 4) {
        acc += row[j]   * vec[j];
        acc += row[j+1] * vec[j+1];
        acc += row[j+2] * vec[j+2];
        acc += row[j+3] * vec[j+3];
    }
    output[tid] = acc;
}

// GEMV with float4 vectorized loads
kernel void gemv_vec4_kernel(
    device const float4* matrix [[buffer(0)]],  // N × (D/4) packed
    device const float4* vec    [[buffer(1)]],   // D/4
    device float*        output [[buffer(2)]],   // N
    constant uint&       D4     [[buffer(3)]],   // D / 4
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    device const float4* row = matrix + tid * D4;
    for (uint j = 0; j < D4; j++) {
        float4 m = row[j];
        float4 v = vec[j];
        acc += m.x * v.x + m.y * v.y + m.z * v.z + m.w * v.w;
    }
    output[tid] = acc;
}

// Gather + weighted sum: output[d] = sum_i(weights[i] * V[indices[i]*D + d])
// Each thread computes one dimension of the output vector
kernel void gather_multiply_kernel(
    device const uint*   indices [[buffer(0)]],  // k
    device const float*  weights [[buffer(1)]],  // k
    device const float*  V       [[buffer(2)]],  // N × D
    device float*        output  [[buffer(3)]],  // D
    constant uint&       k       [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint i = 0; i < k; i++) {
        acc += weights[i] * V[indices[i] * D + tid];
    }
    output[tid] = acc;
}

// Gather with threadgroup-cached weights
kernel void gather_multiply_cached_kernel(
    device const uint*   indices [[buffer(0)]],
    device const float*  weights [[buffer(1)]],
    device const float*  V       [[buffer(2)]],
    device float*        output  [[buffer(3)]],
    constant uint&       k       [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    // Cache weights and indices in threadgroup memory
    threadgroup float tg_weights[512];
    threadgroup uint  tg_indices[512];

    // Cooperative load
    for (uint i = lid; i < k; i += tpg) {
        tg_weights[i] = weights[i];
        tg_indices[i] = indices[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = 0; i < k; i++) {
        acc += tg_weights[i] * V[tg_indices[i] * D + tid];
    }
    output[tid] = acc;
}

// ---------------------------------------------------------------------------
// Exp 0.3: SLC probing — read K and V from separate or same buffers
// GPU side reads buffer B while CPU reads buffer A concurrently
// ---------------------------------------------------------------------------

// Sequential scan of a buffer — measures effective read bandwidth
// which degrades if SLC is being evicted by concurrent CPU access
kernel void slc_probe_scan(
    device const float4* data [[buffer(0)]],
    device float*        out  [[buffer(1)]],
    constant uint&       N4   [[buffer(2)]],  // number of float4 elements
    uint tid  [[thread_position_in_grid]],
    uint tpg  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]]
) {
    float4 acc = float4(0.0f);
    // Stride across the buffer to touch all cache lines
    for (uint i = tid; i < N4; i += tpg * 256) {
        acc += data[i];
    }
    if (tid % tpg == 0) {
        out[gid] = acc.x + acc.y + acc.z + acc.w;
    }
}

// Random access pattern — worst case for SLC
kernel void slc_probe_random(
    device const float*  data    [[buffer(0)]],
    device const uint*   offsets [[buffer(1)]],
    device float*        out     [[buffer(2)]],
    constant uint&       count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint i = 0; i < count; i++) {
        acc += data[offsets[(tid * count + i) % count]];
    }
    out[tid] = acc;
}
