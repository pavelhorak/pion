// bench_msl_sdpa_half_kv.m — gh #195 ceiling test.
//
// Question the issue asks: `--metal-attention-fp16` stores session K/V as
// fp32 (`pion_metal_sdpa_store_kv`, metal_wrap.m:1403) and every fp16 kernel
// declares `const device float4* K/V`, casting `half4(k_base[...])` per
// element. So the kernel streams 2× the bytes it uses. How much of the kernel
// time is that?
//
// Method: two kernels, byte-identical except for the K/V element type.
//   A = sdpa_q1_fp16         const device float4* K/V   (what ships today)
//   B = sdpa_q1_fp16_halfkv  const device half4*  K/V   (pre-converted)
// Same math, same online-softmax order, same threadgroup shape, same PSO
// function constant. B's buffers are pre-converted on the host, so this is the
// hard upper bound on the store-as-half change — no conversion cost is charged
// to either side.
//
// Decision rule (from the issue): delta < 25% → close #195.
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation -O3 \
//          tests/bench_msl_sdpa_half_kv.m -o /tmp/bench_msl_sdpa_half_kv
// Run:   /tmp/bench_msl_sdpa_half_kv            # N=2048 and N=65536
//        /tmp/bench_msl_sdpa_half_kv 4096       # single N

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#define H 8
#define D 128
#define WARMUP 20
#define ITERS  60
#define SG_PER_TG 8

// Both kernels are verbatim src/ffi/metal_compute.metal `sdpa_q1_fp16`, with
// the K/V pointer type as the only difference in B. Keeping them in one source
// string guarantees the same compiler settings for both.
static const char *kSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#define SG_PER_TG  8u\n"
"#define CHUNK_MAX  4u\n"
"constant uint D_HEAD [[function_constant(0)]];\n"
"constant uint D4    = D_HEAD / 4u;\n"
"constant uint CHUNK = (D4 + 31u) / 32u;\n"
"\n"
"#define SDPA_Q1_FP16_BODY(KVTYPE, KVCAST)                                     \\\n"
"    threadgroup half  s_m[SG_PER_TG];                                         \\\n"
"    threadgroup half  s_l[SG_PER_TG];                                         \\\n"
"    half4 q[CHUNK_MAX];                                                       \\\n"
"    for (uint c = 0u; c < CHUNK_MAX; ++c) q[c] = half4(0.0h);                 \\\n"
"    for (uint c = 0u; c < CHUNK; ++c) {                                       \\\n"
"        uint idx = c * 32u + lane;                                            \\\n"
"        if (idx < D4) q[c] = half4(Q[h * D4 + idx]);                          \\\n"
"    }                                                                         \\\n"
"    uint eff_start = (W_window > 0u && N_tokens > W_window) ? (N_tokens - W_window) : 0u; \\\n"
"    uint eff_n     = N_tokens - eff_start;                                    \\\n"
"    uint chunk_n = (eff_n + SG_PER_TG - 1u) / SG_PER_TG;                      \\\n"
"    uint t_begin = eff_start + sg * chunk_n;                                  \\\n"
"    uint t_end   = min(t_begin + chunk_n, N_tokens);                          \\\n"
"    half  m = -HALF_MAX;                                                      \\\n"
"    half  l = 0.0h;                                                           \\\n"
"    half4 o[CHUNK_MAX];                                                       \\\n"
"    for (uint c = 0u; c < CHUNK_MAX; ++c) o[c] = half4(0.0h);                 \\\n"
"    const device KVTYPE* k_base = K + h * N_tokens * D4;                      \\\n"
"    const device KVTYPE* v_base = V + h * N_tokens * D4;                      \\\n"
"    for (uint t = t_begin; t < t_end; ++t) {                                  \\\n"
"        float partial = 0.0f;                                                 \\\n"
"        for (uint c = 0u; c < CHUNK; ++c) {                                   \\\n"
"            uint idx = c * 32u + lane;                                        \\\n"
"            if (idx < D4) {                                                   \\\n"
"                half4 k = KVCAST(k_base[t * D4 + idx]);                       \\\n"
"                partial += float(dot(q[c], k));                               \\\n"
"            }                                                                 \\\n"
"        }                                                                     \\\n"
"        half score = half(simd_sum(partial) * scale);                         \\\n"
"        half new_m  = max(m, score);                                          \\\n"
"        half factor = fast::exp(m - new_m);                                   \\\n"
"        half exp_s  = fast::exp(score - new_m);                               \\\n"
"        m = new_m;                                                            \\\n"
"        l = l * factor + exp_s;                                               \\\n"
"        for (uint c = 0u; c < CHUNK; ++c) {                                   \\\n"
"            uint idx = c * 32u + lane;                                        \\\n"
"            if (idx < D4) {                                                   \\\n"
"                half4 v = KVCAST(v_base[t * D4 + idx]);                       \\\n"
"                o[c] = o[c] * factor + exp_s * v;                             \\\n"
"            }                                                                 \\\n"
"        }                                                                     \\\n"
"    }                                                                         \\\n"
"    if (lane == 0u) { s_m[sg] = m; s_l[sg] = l; }                             \\\n"
"    for (uint c = 0u; c < CHUNK; ++c) s_o[sg * CHUNK * 32u + c * 32u + lane] = o[c]; \\\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);                          \\\n"
"    if (sg == 0u) {                                                           \\\n"
"        half M_acc = -HALF_MAX;                                               \\\n"
"        half L_acc = 0.0h;                                                    \\\n"
"        half4 O_acc[CHUNK_MAX];                                               \\\n"
"        for (uint c = 0u; c < CHUNK_MAX; ++c) O_acc[c] = half4(0.0h);         \\\n"
"        for (uint i = 0u; i < SG_PER_TG; ++i) {                               \\\n"
"            half  m_i = s_m[i];                                               \\\n"
"            half  l_i = s_l[i];                                               \\\n"
"            half  new_M = max(M_acc, m_i);                                    \\\n"
"            half  a = fast::exp(M_acc - new_M);                               \\\n"
"            half  b = fast::exp(m_i   - new_M);                               \\\n"
"            for (uint c = 0u; c < CHUNK; ++c) {                               \\\n"
"                O_acc[c] = O_acc[c] * a + s_o[i * CHUNK * 32u + c * 32u + lane] * b; \\\n"
"            }                                                                 \\\n"
"            L_acc = L_acc * a + l_i * b;                                      \\\n"
"            M_acc = new_M;                                                    \\\n"
"        }                                                                     \\\n"
"        for (uint c = 0u; c < CHUNK; ++c) {                                   \\\n"
"            uint idx = c * 32u + lane;                                        \\\n"
"            if (idx < D4) O[h * D4 + idx] = float4(O_acc[c] / L_acc);         \\\n"
"        }                                                                     \\\n"
"    }\n"
"\n"
"// A — ships today: fp32 K/V streamed, cast to half per element on load.\n"
"kernel void sdpa_q1_fp16(\n"
"    const device float4* Q [[buffer(0)]],\n"
"    const device float4* K [[buffer(1)]],\n"
"    const device float4* V [[buffer(2)]],\n"
"    device       float4* O [[buffer(3)]],\n"
"    constant     uint&   N_tokens [[buffer(4)]],\n"
"    constant     float&  scale    [[buffer(5)]],\n"
"    constant     uint&   W_window [[buffer(6)]],\n"
"    threadgroup  half4*  s_o      [[threadgroup(0)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]])\n"
"{\n"
"    SDPA_Q1_FP16_BODY(float4, half4)\n"
"}\n"
"\n"
"// B — the proposal: K/V already half in memory, loaded directly.\n"
"kernel void sdpa_q1_fp16_halfkv(\n"
"    const device float4* Q [[buffer(0)]],\n"
"    const device half4*  K [[buffer(1)]],\n"
"    const device half4*  V [[buffer(2)]],\n"
"    device       float4* O [[buffer(3)]],\n"
"    constant     uint&   N_tokens [[buffer(4)]],\n"
"    constant     float&  scale    [[buffer(5)]],\n"
"    constant     uint&   W_window [[buffer(6)]],\n"
"    threadgroup  half4*  s_o      [[threadgroup(0)]],\n"
"    uint h [[threadgroup_position_in_grid]],\n"
"    uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]])\n"
"{\n"
"    SDPA_Q1_FP16_BODY(half4, half4)\n"
"}\n";

static double mach_to_ms(uint64_t d) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)d * tb.numer / tb.denom / 1e6;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static float cosine(const float *a, const float *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb));
}
// IEEE fp32 → fp16 (round-to-nearest-even), matching what `half4(float4)` does
// on the GPU, so B's pre-converted buffer holds exactly the values A computes.
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;               // subnormal/zero — inputs are ~U(-1,1)
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);  // inf/nan
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    if ((man & 0x1FFFu) > 0x1000u || ((man & 0x1FFFu) == 0x1000u && (h & 1u))) h++;  // RNE
    return h;
}

static void run_one_N(id<MTLDevice> dev, id<MTLLibrary> lib, id<MTLCommandQueue> q,
                      id<MTLComputePipelineState> psoA, id<MTLComputePipelineState> psoB,
                      uint32_t N) {
    @autoreleasepool {
        const size_t Q_bytes  = (size_t)H * D * sizeof(float);
        const size_t KV32     = (size_t)H * N * D * sizeof(float);
        const size_t KV16     = (size_t)H * N * D * sizeof(uint16_t);
        const size_t O_bytes  = (size_t)H * D * sizeof(float);

        id<MTLBuffer> Qb  = [dev newBufferWithLength:Q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> K32 = [dev newBufferWithLength:KV32 options:MTLResourceStorageModeShared];
        id<MTLBuffer> V32 = [dev newBufferWithLength:KV32 options:MTLResourceStorageModeShared];
        id<MTLBuffer> K16 = [dev newBufferWithLength:KV16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> V16 = [dev newBufferWithLength:KV16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> OA  = [dev newBufferWithLength:O_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> OB  = [dev newBufferWithLength:O_bytes options:MTLResourceStorageModeShared];

        uint32_t Ntok = N, W = 0; float scale = 1.0f / sqrtf((float)D);
        id<MTLBuffer> Nb = [dev newBufferWithBytes:&Ntok  length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Sb = [dev newBufferWithBytes:&scale length:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Wb = [dev newBufferWithBytes:&W     length:4 options:MTLResourceStorageModeShared];

        srand(0);
        float *Qp = (float *)[Qb contents];
        float *Kp = (float *)[K32 contents];
        float *Vp = (float *)[V32 contents];
        uint16_t *Kh = (uint16_t *)[K16 contents];
        uint16_t *Vh = (uint16_t *)[V16 contents];
        for (size_t i = 0; i < (size_t)H * D; ++i) Qp[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        for (size_t i = 0; i < (size_t)H*N*D; ++i) { Kp[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f; Kh[i] = f32_to_f16(Kp[i]); }
        for (size_t i = 0; i < (size_t)H*N*D; ++i) { Vp[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f; Vh[i] = f32_to_f16(Vp[i]); }

        MTLSize grid   = MTLSizeMake(H, 1, 1);
        MTLSize tgsize = MTLSizeMake(32 * SG_PER_TG, 1, 1);
        // s_o = [SG_PER_TG * CHUNK * 32] half4; CHUNK = ceil((D/4)/32).
        NSUInteger chunk = (((D / 4) + 31) / 32);
        NSUInteger tgmem = SG_PER_TG * chunk * 32 * 8;

        id<MTLSharedEvent> ev = [dev newSharedEvent];
        __block uint64_t ctr = 0;

        void (^dispatch_one)(id<MTLComputePipelineState>, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>) =
            ^(id<MTLComputePipelineState> pso, id<MTLBuffer> Kb, id<MTLBuffer> Vb, id<MTLBuffer> Ob) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:pso];
                [ce setBuffer:Qb offset:0 atIndex:0];
                [ce setBuffer:Kb offset:0 atIndex:1];
                [ce setBuffer:Vb offset:0 atIndex:2];
                [ce setBuffer:Ob offset:0 atIndex:3];
                [ce setBuffer:Nb offset:0 atIndex:4];
                [ce setBuffer:Sb offset:0 atIndex:5];
                [ce setBuffer:Wb offset:0 atIndex:6];
                [ce setThreadgroupMemoryLength:tgmem atIndex:0];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tgsize];
                [ce endEncoding];
                uint64_t target = ++ctr;
                [cb encodeSignalEvent:ev value:target];
                [cb commit];
                while ([ev signaledValue] < target) { /* spin */ }
            };

        // Correctness: B must agree with A — same values, same accumulation order.
        dispatch_one(psoA, K32, V32, OA);
        dispatch_one(psoB, K16, V16, OB);
        float c = cosine((float *)[OA contents], (float *)[OB contents], H * D);

        for (int i = 0; i < WARMUP; ++i) { dispatch_one(psoA, K32, V32, OA); dispatch_one(psoB, K16, V16, OB); }

        // Interleaved A,B per iteration — thermal drift hits both equally.
        double *ta = malloc(ITERS * sizeof(double));
        double *tb = malloc(ITERS * sizeof(double));
        for (int i = 0; i < ITERS; ++i) {
            uint64_t t0 = mach_absolute_time(); dispatch_one(psoA, K32, V32, OA);
            uint64_t t1 = mach_absolute_time(); dispatch_one(psoB, K16, V16, OB);
            uint64_t t2 = mach_absolute_time();
            ta[i] = mach_to_ms(t1 - t0);
            tb[i] = mach_to_ms(t2 - t1);
        }
        qsort(ta, ITERS, sizeof(double), cmp_double);
        qsort(tb, ITERS, sizeof(double), cmp_double);
        double ma = ta[ITERS/2], mb = tb[ITERS/2];
        double gb_a = (double)(2.0 * H * N * D * 4) / 1e9 / (ma / 1e3);
        double gb_b = (double)(2.0 * H * N * D * 2) / 1e9 / (mb / 1e3);

        printf("N=%-6u  A fp32-K/V median=%.4f ms (%.1f GB/s)   B half-K/V median=%.4f ms (%.1f GB/s)\n",
               N, ma, gb_a, mb, gb_b);
        printf("          delta = %+.1f%%   (speedup %.2fx)   cosine(A,B) = %.7f\n",
               (ma - mb) / ma * 100.0, ma / mb, c);
        free(ta); free(tb);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        printf("device: %s\nshape: H=%d d_head=%d, Q=1 decode, %d iters (A/B interleaved)\n\n",
               [[dev name] UTF8String], H, D, ITERS);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile error: %s\n", [[err localizedDescription] UTF8String]); return 2; }

        MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
        uint32_t dhead = D;
        [fc setConstantValue:&dhead type:MTLDataTypeUInt atIndex:0];
        id<MTLFunction> fa = [lib newFunctionWithName:@"sdpa_q1_fp16" constantValues:fc error:&err];
        id<MTLFunction> fb = [lib newFunctionWithName:@"sdpa_q1_fp16_halfkv" constantValues:fc error:&err];
        if (!fa || !fb) { fprintf(stderr, "fn error: %s\n", [[err localizedDescription] UTF8String]); return 3; }
        id<MTLComputePipelineState> psoA = [dev newComputePipelineStateWithFunction:fa error:&err];
        id<MTLComputePipelineState> psoB = [dev newComputePipelineStateWithFunction:fb error:&err];
        if (!psoA || !psoB) { fprintf(stderr, "PSO error: %s\n", [[err localizedDescription] UTF8String]); return 4; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        if (argc > 1) {
            run_one_N(dev, lib, q, psoA, psoB, (uint32_t)atoi(argv[1]));
        } else {
            run_one_N(dev, lib, q, psoA, psoB, 2048);
            run_one_N(dev, lib, q, psoA, psoB, 65536);
        }
        printf("\ngh #195 decision rule: delta < 25%% at both N → close the issue.\n");
    }
    return 0;
}
