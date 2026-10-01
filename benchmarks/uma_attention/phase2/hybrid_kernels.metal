// UMMA Phase 2 — Hybrid Pipeline Metal Kernels
// GPU side of the CPU→GPU handoff via MTLSharedEvent

#include <metal_stdlib>
using namespace metal;

// V gather + weighted sum: output[d] = sum_i(weights[i] * V[indices[i]*D + d])
// Used after CPU completes top-k extraction and signals via MTLSharedEvent
kernel void v_gather_multiply(
    device const uint*   indices [[buffer(0)]],  // [k]
    device const float*  weights [[buffer(1)]],  // [k]
    device const float*  V       [[buffer(2)]],  // [N, D]
    device float*        output  [[buffer(3)]],  // [D]
    constant uint&       k       [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= D) return;
    float acc = 0.0f;
    for (uint i = 0; i < k; i++) {
        acc += weights[i] * V[indices[i] * D + tid];
    }
    output[tid] = acc;
}

// Batched variant: H heads, each with own indices/weights
// Grid: (D, H, 1)
kernel void v_gather_multiply_batched(
    device const uint*   indices [[buffer(0)]],  // [H, k]
    device const float*  weights [[buffer(1)]],  // [H, k]
    device const float*  V       [[buffer(2)]],  // [H, N, D]
    device float*        output  [[buffer(3)]],  // [H, D]
    constant uint&       N       [[buffer(4)]],
    constant uint&       D       [[buffer(5)]],
    constant uint&       k       [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]         // (d, h)
) {
    uint d = tid.x;
    uint h = tid.y;
    if (d >= D) return;

    device const uint*  idx = indices + h * k;
    device const float* wt  = weights + h * k;
    device const float* v_h = V + h * N * D;

    float acc = 0.0f;
    for (uint i = 0; i < k; i++) {
        acc += wt[i] * v_h[idx[i] * D + d];
    }
    output[h * D + d] = acc;
}
