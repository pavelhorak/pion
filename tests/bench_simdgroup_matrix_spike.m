// Spike bench — does Apple's simdgroup_matrix (the 8x8x8 hardware MAC) deliver
// enough speedup over naive thread-per-element matmul to justify the multi-week
// implementation cost of a FlashAttention-2-style attention kernel built on it?
//
// Workload: 64x64 @ 64x64 = 64x64 matmul, FP32. Repeated K_REPS=10000 times in a
// kernel to amortize dispatch overhead. The 64x64 shape factors as 8 × 8x8 tiles
// in M, K, N — clean fit for simdgroup_matrix; comparable to one inner-loop
// iteration of attention (M=8 query rows × D=64 head dim per token tile).
//
// Build: clang -fobjc-arc -framework Metal -framework Foundation -O3 \
//          tests/bench_simdgroup_matrix_spike.m -o /tmp/bench_simdgroup_matrix_spike
// Run:   /tmp/bench_simdgroup_matrix_spike
//
// Decision rule:
//   simdgroup_matrix beats naive by ≥ 4× → implement the kernel
//   speedup < 4× → document and stay with the current Phase C linear-M kernel

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>

#define MAT_DIM   64
#define K_REPS    10000
#define WARMUP    10
#define ITERS     50

static const char *kSrc =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"\n"
"// Naive: one thread per output element. tg = 8x8 = 64 threads = 2 simdgroups.\n"
"kernel void matmul_naive(\n"
"    const device float* A [[buffer(0)]],\n"
"    const device float* B [[buffer(1)]],\n"
"    device       float* C [[buffer(2)]],\n"
"    constant uint& reps   [[buffer(3)]],\n"
"    uint2 tid [[thread_position_in_grid]])\n"
"{\n"
"    uint row = tid.y, col = tid.x;\n"
"    if (row >= 64u || col >= 64u) return;\n"
"    float acc = 0.0f;\n"
"    for (uint r = 0; r < reps; ++r) {\n"
"        float local = 0.0f;\n"
"        for (uint k = 0; k < 64u; ++k) local += A[row*64 + k] * B[k*64 + col];\n"
"        acc += local;\n"
"    }\n"
"    C[row*64 + col] = acc;\n"
"}\n"
"\n"
"// simdgroup_matrix: one threadgroup of 32 threads (= 1 simdgroup) computes\n"
"// the entire 64x64 matmul as 8x8 of 8x8 tiles, each via simdgroup_matrix MAC.\n"
"kernel void matmul_sgm(\n"
"    const device float* A [[buffer(0)]],\n"
"    const device float* B [[buffer(1)]],\n"
"    device       float* C [[buffer(2)]],\n"
"    constant uint& reps   [[buffer(3)]])\n"
"{\n"
"    // Output: 8x8 tiles, indexed by (mt, nt) ∈ [0..8) × [0..8).\n"
"    // Each tile is 8x8, computed as sum over kt ∈ [0..8) of A[8*mt:8*mt+8, 8*kt:8*kt+8] @ B[8*kt:8*kt+8, 8*nt:8*nt+8].\n"
"    simdgroup_matrix<float, 8, 8> A_tile;\n"
"    simdgroup_matrix<float, 8, 8> B_tile;\n"
"    simdgroup_matrix<float, 8, 8> C_acc[64]; // 8x8 of 8x8 = 64 tiles\n"
"    for (uint i = 0; i < 64; ++i) C_acc[i] = simdgroup_matrix<float, 8, 8>(0);\n"
"    for (uint r = 0; r < reps; ++r) {\n"
"        for (uint mt = 0; mt < 8; ++mt) {\n"
"            for (uint nt = 0; nt < 8; ++nt) {\n"
"                simdgroup_matrix<float, 8, 8> acc = simdgroup_matrix<float, 8, 8>(0);\n"
"                for (uint kt = 0; kt < 8; ++kt) {\n"
"                    simdgroup_load(A_tile, A, 64, ulong2(8*kt, 8*mt));\n"
"                    simdgroup_load(B_tile, B, 64, ulong2(8*nt, 8*kt));\n"
"                    simdgroup_multiply_accumulate(acc, A_tile, B_tile, acc);\n"
"                }\n"
"                C_acc[mt*8 + nt] = acc;\n"
"            }\n"
"        }\n"
"    }\n"
"    // Store back\n"
"    for (uint mt = 0; mt < 8; ++mt) {\n"
"        for (uint nt = 0; nt < 8; ++nt) {\n"
"            simdgroup_store(C_acc[mt*8 + nt], C, 64, ulong2(8*nt, 8*mt));\n"
"        }\n"
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

static double bench(id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                    id<MTLBuffer> Ab, id<MTLBuffer> Bb, id<MTLBuffer> Cb,
                    id<MTLBuffer> Rb, MTLSize grid, MTLSize tg,
                    id<MTLSharedEvent> ev, uint64_t *evCounter,
                    int iters)
{
    double *t = (double *)malloc(iters * sizeof(double));
    for (int i = 0; i < iters; ++i) {
        uint64_t t0 = mach_absolute_time();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:pso];
        [ce setBuffer:Ab offset:0 atIndex:0];
        [ce setBuffer:Bb offset:0 atIndex:1];
        [ce setBuffer:Cb offset:0 atIndex:2];
        [ce setBuffer:Rb offset:0 atIndex:3];
        [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [ce endEncoding];
        uint64_t target = ++*evCounter;
        [cb encodeSignalEvent:ev value:target];
        [cb commit];
        while ([ev signaledValue] < target) { /* spin */ }
        uint64_t t1 = mach_absolute_time();
        t[i] = mach_to_ms(t1 - t0);
    }
    qsort(t, iters, sizeof(double), cmp_double);
    double median = t[iters / 2];
    free(t);
    return median;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", [[dev name] UTF8String]);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&err];
        if (!lib) { fprintf(stderr, "compile: %s\n", [[err localizedDescription] UTF8String]); return 1; }

        id<MTLFunction> fnNaive = [lib newFunctionWithName:@"matmul_naive"];
        id<MTLFunction> fnSGM   = [lib newFunctionWithName:@"matmul_sgm"];
        id<MTLComputePipelineState> psoNaive = [dev newComputePipelineStateWithFunction:fnNaive error:&err];
        id<MTLComputePipelineState> psoSGM   = [dev newComputePipelineStateWithFunction:fnSGM   error:&err];
        if (!psoNaive || !psoSGM) { fprintf(stderr, "PSO: %s\n", [[err localizedDescription] UTF8String]); return 2; }

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        uint64_t evCounter = 0;

        size_t bytes = MAT_DIM * MAT_DIM * sizeof(float);
        id<MTLBuffer> Ab = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Cb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        uint32_t reps = K_REPS;
        id<MTLBuffer> Rb = [dev newBufferWithBytes:&reps length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        srand(0);
        float *A = (float *)[Ab contents];
        float *B = (float *)[Bb contents];
        for (int i = 0; i < MAT_DIM * MAT_DIM; ++i) {
            A[i] = ((float)rand() / RAND_MAX) - 0.5f;
            B[i] = ((float)rand() / RAND_MAX) - 0.5f;
        }

        // Naive: 8x8 grid of threadgroups, each 8x8 threads (2 simdgroups).
        // Total threads = 64x64 = 4096 across the GPU.
        MTLSize gridNaive = MTLSizeMake(64, 64, 1);   // dispatchThreads (will be auto-divided)
        MTLSize tgNaive   = MTLSizeMake(8, 8, 1);

        // SGM: 1 threadgroup of 32 threads (1 simdgroup). 1 simdgroup does the whole thing.
        MTLSize gridSGM = MTLSizeMake(1, 1, 1);
        MTLSize tgSGM   = MTLSizeMake(32, 1, 1);

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            (void)bench(q, psoNaive, Ab, Bb, Cb, Rb, gridNaive, tgNaive, ev, &evCounter, 1);
            (void)bench(q, psoSGM,   Ab, Bb, Cb, Rb, gridSGM,   tgSGM,   ev, &evCounter, 1);
        }

        printf("\nworkload: %dx%d FP32 matmul × %d reps inside one kernel\n", MAT_DIM, MAT_DIM, K_REPS);
        printf("(picks the inner-loop hot path of an attention kernel)\n\n");
        double mn = bench(q, psoNaive, Ab, Bb, Cb, Rb, gridNaive, tgNaive, ev, &evCounter, ITERS);
        double sm = bench(q, psoSGM,   Ab, Bb, Cb, Rb, gridSGM,   tgSGM,   ev, &evCounter, ITERS);
        printf("matmul_naive (thread-per-element, %d threads): median=%.3f ms\n",
               MAT_DIM*MAT_DIM, mn);
        printf("matmul_sgm   (1 simdgroup, simdgroup_matrix):  median=%.3f ms\n", sm);

        double speedup = mn / sm;
        printf("\nspeedup: %.2f×\n", speedup);
        printf("decision rule (≥4× → implement step 5 kernel; <4× → stay with Phase C):\n");
        printf("  → %s\n", speedup >= 4.0 ? "JUSTIFIED — implement simdgroup_matrix kernel"
                                          : "NOT justified — Phase C linear-M is the right tradeoff");
    }
    return 0;
}
