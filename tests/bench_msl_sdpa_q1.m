// bench_msl_sdpa_q1.m — Hand-written MSL Q=1 SDPA kernel benchmark.
//
// Goal: Beat MLX `fast.scaled_dot_product_attention` at H=8 N=2048 d_head=128.
//   MLX SDPA reference (per tests/bench_mlx_vs_mojo.py): 0.495 ms median
//   This kernel target: <0.495 ms median
//
// Design (Metal-kernel research notes, 2026-05-01):
//   - 1 threadgroup per head (8 threadgroups)
//   - 1 SIMD group per threadgroup (32 threads — Apple GPU simdgroup size)
//   - Each thread holds 1 float4 of Q (d_head=128 / 32 lanes = 4 floats per thread)
//   - QK dot product: per-thread dot(q,k) → simd_sum reduction
//   - Online softmax: running max + sum, no second pass
//   - V update inline: o = o * factor + exp_score * v
//   - Float4 vectorized loads throughout
//
// Same idea as MLX's sdpa_vector kernel; the question is whether persistent
// command buffer + tighter dispatch can match or beat MLX's Python-driven
// path on this specific shape.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation \
//          -O3 tests/bench_msl_sdpa_q1.m -o /tmp/bench_msl_sdpa_q1
// Run:   /tmp/bench_msl_sdpa_q1

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach_time.h>

#define H 8
#define N 2048
#define D 128
#define WARMUP 20
#define ITERS  100

// Tunable: number of SIMD groups per threadgroup (each owns N/SG_PER_TG tokens).
// Higher = more occupancy. Limit: threadgroup memory + register pressure.
// 8 → 8*4=32 partial outputs in TG memory: 32 * 32 * float4 = 16 KB (fits 32KB cap).
#define SG_PER_TG 8

static const char *kKernelSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"#define SG_PER_TG 8u\n"
"\n"
"// Q=1 single-query decode SDPA, multi-simdgroup over N for occupancy.\n"
"//   Threadgroups: H            (one per head)\n"
"//   Threads/TG:   32 * 4 = 128 (4 simdgroups, each owns N/4 tokens)\n"
"//   Each lane within a SG owns 1 float4 of D (D/32 = 4 floats).\n"
"// Pass: each simdgroup runs online softmax over its N-chunk → partial (m,l,o).\n"
"// Merge: simdgroup 0 reads 4 partials from threadgroup memory and merges.\n"
"kernel void sdpa_q1_fp32(\n"
"    const device float4* Q [[buffer(0)]],\n"
"    const device float4* K [[buffer(1)]],\n"
"    const device float4* V [[buffer(2)]],\n"
"    device       float4* O [[buffer(3)]],\n"
"    constant     uint&   N_tokens [[buffer(4)]],\n"
"    uint h     [[threadgroup_position_in_grid]],\n"
"    uint sg    [[simdgroup_index_in_threadgroup]],\n"
"    uint lane  [[thread_index_in_simdgroup]])\n"
"{\n"
"    constexpr float SCALE = 0.0883883476f; // 1/sqrt(128)\n"
"    constexpr uint  D4    = 32;            // D/4\n"
"\n"
"    // Threadgroup-shared partial state across simdgroups.\n"
"    threadgroup float  s_m[8];\n"
"    threadgroup float  s_l[8];\n"
"    threadgroup float4 s_o[8][32];   // [sg][lane] partial output\n"
"\n"
"    // Each lane reads its 1 float4 of Q (broadcast across simdgroups).\n"
"    float4 q = Q[h * D4 + lane];\n"
"\n"
"    // Determine this simdgroup's [t_begin, t_end) chunk of N.\n"
"    uint chunk = (N_tokens + SG_PER_TG - 1u) / SG_PER_TG;\n"
"    uint t_begin = sg * chunk;\n"
"    uint t_end   = min(t_begin + chunk, N_tokens);\n"
"\n"
"    float  m = -INFINITY;\n"
"    float  l = 0.0f;\n"
"    float4 o = float4(0.0f);\n"
"\n"
"    const device float4* k_base = K + h * N_tokens * D4;\n"
"    const device float4* v_base = V + h * N_tokens * D4;\n"
"\n"
"    for (uint t = t_begin; t < t_end; ++t) {\n"
"        float4 k = k_base[t * D4 + lane];\n"
"        float partial = dot(q, k);\n"
"        float score = simd_sum(partial) * SCALE;\n"
"\n"
"        float new_m  = max(m, score);\n"
"        float factor = fast::exp(m - new_m);\n"
"        float exp_s  = fast::exp(score - new_m);\n"
"        m = new_m;\n"
"        l = l * factor + exp_s;\n"
"\n"
"        float4 v = v_base[t * D4 + lane];\n"
"        o = o * factor + exp_s * v;\n"
"    }\n"
"\n"
"    // Publish partial (m, l, o) to threadgroup memory.\n"
"    if (lane == 0) {\n"
"        s_m[sg] = m;\n"
"        s_l[sg] = l;\n"
"    }\n"
"    s_o[sg][lane] = o;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    // Merge: simdgroup 0 reduces SG_PER_TG partials into final (M, L, O).\n"
"    if (sg == 0) {\n"
"        float  M = -INFINITY;\n"
"        float  L = 0.0f;\n"
"        float4 O_acc = float4(0.0f);\n"
"        for (uint i = 0; i < SG_PER_TG; ++i) {\n"
"            float  m_i = s_m[i];\n"
"            float  l_i = s_l[i];\n"
"            float4 o_i = s_o[i][lane];\n"
"            float new_M = max(M, m_i);\n"
"            float a = fast::exp(M   - new_M);\n"
"            float b = fast::exp(m_i - new_M);\n"
"            O_acc = O_acc * a + o_i * b;\n"
"            L     = L     * a + l_i * b;\n"
"            M     = new_M;\n"
"        }\n"
"        O[h * D4 + lane] = O_acc / L;\n"
"    }\n"
"}\n";

static double mach_to_ms(uint64_t mach) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach * (double)tb.numer / ((double)tb.denom * 1e6);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static void cpu_reference(const float *Q, const float *K, const float *V, float *O) {
    const float scale = 1.0f / sqrtf((float)D);
    float scores[N];
    for (int h = 0; h < H; ++h) {
        const float *qh = Q + h * D;
        const float *kh = K + h * N * D;
        const float *vh = V + h * N * D;
        float maxs = -INFINITY;
        for (int t = 0; t < N; ++t) {
            float s = 0.0f;
            for (int j = 0; j < D; ++j) s += qh[j] * kh[t * D + j];
            s *= scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; ++t) {
            scores[t] = expf(scores[t] - maxs);
            sum += scores[t];
        }
        float *oh = O + h * D;
        for (int j = 0; j < D; ++j) oh[j] = 0.0f;
        for (int t = 0; t < N; ++t) {
            float w = scores[t] / sum;
            for (int j = 0; j < D; ++j) oh[j] += w * vh[t * D + j];
        }
    }
}

static float cosine(const float *a, const float *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb));
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        printf("device: %s\n", [[dev name] UTF8String]);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kKernelSrc) options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile error: %s\n", [[err localizedDescription] UTF8String]); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"sdpa_q1_fp32"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fprintf(stderr, "PSO error: %s\n", [[err localizedDescription] UTF8String]); return 3; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        // MTLSharedEvent + spin-wait replaces waitUntilCompleted (cuts ~100µs latency).
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        __block uint64_t evCounter = 0;

        // Allocate shared buffers (Apple unified memory — no upload, just shared).
        const size_t Q_bytes = H * D * sizeof(float);
        const size_t K_bytes = H * N * D * sizeof(float);
        const size_t V_bytes = H * N * D * sizeof(float);
        const size_t O_bytes = H * D * sizeof(float);

        id<MTLBuffer> Qb = [dev newBufferWithLength:Q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Kb = [dev newBufferWithLength:K_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Vb = [dev newBufferWithLength:V_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Ob = [dev newBufferWithLength:O_bytes options:MTLResourceStorageModeShared];
        uint32_t Ntok = N;
        id<MTLBuffer> Nb = [dev newBufferWithBytes:&Ntok length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Fill with random data (same seed as Python bench for reproducibility).
        srand(0);
        float *Qp = (float *)[Qb contents];
        float *Kp = (float *)[Kb contents];
        float *Vp = (float *)[Vb contents];
        float *Op = (float *)[Ob contents];
        for (size_t i = 0; i < H * D;       ++i) Qp[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        for (size_t i = 0; i < H * N * D;   ++i) Kp[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        for (size_t i = 0; i < H * N * D;   ++i) Vp[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        MTLSize grid = MTLSizeMake(H, 1, 1);                // 8 threadgroups (one per head)
        MTLSize tgsize = MTLSizeMake(32 * SG_PER_TG, 1, 1);  // SG_PER_TG simdgroups per threadgroup

        // Single dispatch helper using MTLSharedEvent + busy spin (low-latency wait).
        void (^dispatch_once)(void) = ^{
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:pso];
            [ce setBuffer:Qb offset:0 atIndex:0];
            [ce setBuffer:Kb offset:0 atIndex:1];
            [ce setBuffer:Vb offset:0 atIndex:2];
            [ce setBuffer:Ob offset:0 atIndex:3];
            [ce setBuffer:Nb offset:0 atIndex:4];
            [ce dispatchThreadgroups:grid threadsPerThreadgroup:tgsize];
            [ce endEncoding];
            uint64_t target = ++evCounter;
            [cb encodeSignalEvent:ev value:target];
            [cb commit];
            // Busy-spin on the shared event — no kernel transition, no condition variable.
            while ([ev signaledValue] < target) { /* spin */ }
        };

        // Correctness check vs CPU reference.
        dispatch_once();
        float *cpu_O = (float *)malloc(O_bytes);
        cpu_reference(Qp, Kp, Vp, cpu_O);
        float c = cosine(Op, cpu_O, H * D);
        printf("correctness: cosine(cpu, gpu) = %.7f\n", c);
        if (c < 0.999f) {
            fprintf(stderr, "FAIL: cosine below 0.999 — kernel is incorrect.\n");
            free(cpu_O);
            return 4;
        }
        free(cpu_O);

        // Warmup
        for (int i = 0; i < WARMUP; ++i) dispatch_once();

        // Timed iterations
        double *t_ms = (double *)malloc(ITERS * sizeof(double));
        for (int i = 0; i < ITERS; ++i) {
            uint64_t t0 = mach_absolute_time();
            dispatch_once();
            uint64_t t1 = mach_absolute_time();
            t_ms[i] = mach_to_ms(t1 - t0);
        }
        qsort(t_ms, ITERS, sizeof(double), cmp_double);
        double median = t_ms[ITERS / 2];
        double p95    = t_ms[(int)(ITERS * 0.95)];
        double minv   = t_ms[0];
        printf("\nshape: H=%d N=%d d_head=%d  (Q=1 decoder)\n", H, N, D);
        printf("MSL sdpa_q1_fp32   median=%.3f ms  p95=%.3f ms  min=%.3f ms\n", median, p95, minv);
        printf("\nMLX reference (from tests/bench_mlx_vs_mojo.py):\n");
        printf("  fast.scaled_dot_product_attention   median=0.495 ms\n");
        printf("  naive matmul + softmax              median=0.576 ms\n");
        printf("\nVerdict:\n");
        printf("  vs MLX SDPA: %.2f×  (%s)\n",
               0.495 / median, median < 0.495 ? "MSL faster — beats MLX" : "MLX still faster");
        free(t_ms);
    }
    return 0;
}
