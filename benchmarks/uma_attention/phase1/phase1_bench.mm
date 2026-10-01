// UMMA Phase 1R — GPU-Batched Multi-Head Sparse Attention Benchmark
//
// Measures native Metal batched multi-head attention vs CPU sequential (cblas)
// at various H (heads) and N (sequence length) configurations.
//
// Build: make
// Run:   ./phase1_bench [--markdown] [--heads H] [--topk K]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/sysctl.h>

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
static mach_timebase_info_data_t g_timebase;
static void init_timing() { mach_timebase_info(&g_timebase); }
static inline uint64_t now_ns() {
    return mach_absolute_time() * g_timebase.numer / g_timebase.denom;
}
static inline double ns_to_us(uint64_t ns) { return (double)ns / 1000.0; }

struct Stats {
    double mean_us, std_us, min_us, max_us, median_us, p95_us;
};

static Stats compute_stats(const std::vector<double>& t) {
    Stats s{};
    if (t.empty()) return s;
    auto sorted = t;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (auto v : sorted) sum += v;
    s.mean_us = sum / sorted.size();
    s.min_us = sorted.front();
    s.max_us = sorted.back();
    s.median_us = sorted[sorted.size() / 2];
    s.p95_us = sorted[(size_t)(sorted.size() * 0.95)];
    double var = 0;
    for (auto v : sorted) var += (v - s.mean_us) * (v - s.mean_us);
    s.std_us = sqrt(var / sorted.size());
    return s;
}

// ---------------------------------------------------------------------------
// Metal state
// ---------------------------------------------------------------------------
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;

static bool init_metal() {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) { fprintf(stderr, "ERROR: No Metal device\n"); return false; }
    g_queue = [g_device newCommandQueue];

    NSError* error = nil;
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* srcPath = [cwd stringByAppendingPathComponent:@"multihead_attention.metal"];
    NSString* source = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&error];
    if (!source) {
        fprintf(stderr, "ERROR: Cannot read multihead_attention.metal: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    g_library = [g_device newLibraryWithSource:source options:opts error:&error];
    if (!g_library) {
        fprintf(stderr, "ERROR: Shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }
    fprintf(stderr, "INFO: Shaders compiled successfully.\n");
    return true;
}

static id<MTLComputePipelineState> make_pipeline(const char* name) {
    NSString* fn = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [g_library newFunctionWithName:fn];
    if (!func) { fprintf(stderr, "ERROR: Kernel '%s' not found\n", name); return nil; }
    NSError* error = nil;
    auto pso = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) { fprintf(stderr, "ERROR: Pipeline '%s': %s\n", name, [[error localizedDescription] UTF8String]); }
    return pso;
}

// ---------------------------------------------------------------------------
// CPU baseline: sequential multi-head sparse attention via Accelerate BLAS
// ---------------------------------------------------------------------------
struct AttentionResult {
    std::vector<float> output;  // [H * D]
};

static AttentionResult cpu_multihead_attention(
    const float* Q, const float* K, const float* V,
    uint32_t H, uint32_t N, uint32_t D, uint32_t top_k
) {
    AttentionResult result;
    result.output.resize(H * D);

    std::vector<float> scores(N);
    std::vector<float> topk_vals(top_k);
    std::vector<uint32_t> topk_idx(top_k);

    float scale = 1.0f / sqrtf((float)D);

    for (uint32_t h = 0; h < H; h++) {
        const float* q_h = Q + h * D;
        const float* k_h = K + (size_t)h * N * D;
        const float* v_h = V + (size_t)h * N * D;
        float* out_h = result.output.data() + h * D;

        // GEMV: scores = K @ q * scale
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, scale,
                    k_h, D, q_h, 1, 0.0f, scores.data(), 1);

        // Top-k: partial sort
        // Initialize with first k
        for (uint32_t i = 0; i < top_k; i++) {
            topk_idx[i] = i;
            topk_vals[i] = scores[i];
        }
        float min_val = topk_vals[0];
        uint32_t min_pos = 0;
        for (uint32_t i = 1; i < top_k; i++) {
            if (topk_vals[i] < min_val) { min_val = topk_vals[i]; min_pos = i; }
        }
        for (uint32_t n = top_k; n < N; n++) {
            float v = scores[n];
            if (v > min_val) {
                topk_idx[min_pos] = n;
                topk_vals[min_pos] = v;
                min_val = topk_vals[0]; min_pos = 0;
                for (uint32_t i = 1; i < top_k; i++) {
                    if (topk_vals[i] < min_val) { min_val = topk_vals[i]; min_pos = i; }
                }
            }
        }

        // Softmax
        float mx = topk_vals[0];
        for (uint32_t i = 1; i < top_k; i++) mx = fmaxf(mx, topk_vals[i]);
        float sum = 0;
        for (uint32_t i = 0; i < top_k; i++) {
            topk_vals[i] = expf(topk_vals[i] - mx);
            sum += topk_vals[i];
        }
        float inv_sum = 1.0f / sum;
        for (uint32_t i = 0; i < top_k; i++) topk_vals[i] *= inv_sum;

        // V gather + weighted sum
        memset(out_h, 0, D * sizeof(float));
        for (uint32_t i = 0; i < top_k; i++) {
            const float* v_row = v_h + topk_idx[i] * D;
            float w = topk_vals[i];
            for (uint32_t d = 0; d < D; d++) {
                out_h[d] += w * v_row[d];
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// GPU: Multi-kernel pipeline (GEMV → top-k → softmax → gather)
// ---------------------------------------------------------------------------
static double gpu_pipeline_attention(
    id<MTLBuffer> Q_buf, id<MTLBuffer> K_buf, id<MTLBuffer> V_buf,
    id<MTLBuffer> out_buf, id<MTLBuffer> scores_buf,
    id<MTLBuffer> idx_buf, id<MTLBuffer> topk_buf,
    uint32_t H, uint32_t N, uint32_t D, uint32_t top_k,
    id<MTLComputePipelineState> gemv_pso,
    id<MTLComputePipelineState> topk_pso,
    id<MTLComputePipelineState> softmax_pso,
    id<MTLComputePipelineState> gather_pso,
    id<MTLBuffer> N_buf, id<MTLBuffer> D_buf, id<MTLBuffer> D4_buf,
    id<MTLBuffer> K_topk_buf
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];

        // 1. Batched GEMV
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:gemv_pso];
            [enc setBuffer:Q_buf offset:0 atIndex:0];
            [enc setBuffer:K_buf offset:0 atIndex:1];
            [enc setBuffer:scores_buf offset:0 atIndex:2];
            [enc setBuffer:N_buf offset:0 atIndex:3];
            [enc setBuffer:D4_buf offset:0 atIndex:4];
            NSUInteger tgW = MIN((NSUInteger)gemv_pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
            [enc dispatchThreads:MTLSizeMake(N, H, 1)
           threadsPerThreadgroup:MTLSizeMake(tgW, 1, 1)];
            [enc endEncoding];
        }

        // 2. Batched top-k (parallel variant for large N)
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (N > 4096) {
                [enc setComputePipelineState:topk_pso];  // parallel
                [enc setBuffer:scores_buf offset:0 atIndex:0];
                [enc setBuffer:idx_buf offset:0 atIndex:1];
                [enc setBuffer:topk_buf offset:0 atIndex:2];
                [enc setBuffer:N_buf offset:0 atIndex:3];
                [enc setBuffer:K_topk_buf offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            } else {
                // Sequential top-k for small N
                auto seq_pso = make_pipeline("batched_topk");
                [enc setComputePipelineState:seq_pso];
                [enc setBuffer:scores_buf offset:0 atIndex:0];
                [enc setBuffer:idx_buf offset:0 atIndex:1];
                [enc setBuffer:topk_buf offset:0 atIndex:2];
                [enc setBuffer:N_buf offset:0 atIndex:3];
                [enc setBuffer:K_topk_buf offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(H, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
            [enc endEncoding];
        }

        // 3. Batched softmax
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:softmax_pso];
            [enc setBuffer:topk_buf offset:0 atIndex:0];
            [enc setBuffer:K_topk_buf offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(H, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        // 4. Batched V gather + multiply
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:gather_pso];
            [enc setBuffer:idx_buf offset:0 atIndex:0];
            [enc setBuffer:topk_buf offset:0 atIndex:1];
            [enc setBuffer:V_buf offset:0 atIndex:2];
            [enc setBuffer:out_buf offset:0 atIndex:3];
            [enc setBuffer:N_buf offset:0 atIndex:4];
            [enc setBuffer:D_buf offset:0 atIndex:5];
            [enc setBuffer:K_topk_buf offset:0 atIndex:6];
            NSUInteger tgW = MIN((NSUInteger)D, (NSUInteger)gather_pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(D, H, 1)
           threadsPerThreadgroup:MTLSizeMake(tgW, 1, 1)];
            [enc endEncoding];
        }

        uint64_t t0 = now_ns();
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = now_ns();
        return ns_to_us(t1 - t0);
    }
}

// ---------------------------------------------------------------------------
// GPU: Fused single-dispatch
// ---------------------------------------------------------------------------
static double gpu_fused_attention(
    id<MTLBuffer> Q_buf, id<MTLBuffer> K_buf, id<MTLBuffer> V_buf,
    id<MTLBuffer> out_buf,
    uint32_t H, uint32_t N, uint32_t D, uint32_t top_k,
    id<MTLComputePipelineState> fused_pso,
    id<MTLBuffer> N_buf, id<MTLBuffer> D_buf, id<MTLBuffer> K_topk_buf,
    id<MTLBuffer> scores_scratch  // only for large-N variant
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:fused_pso];
        [enc setBuffer:Q_buf offset:0 atIndex:0];
        [enc setBuffer:K_buf offset:0 atIndex:1];
        [enc setBuffer:V_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];

        if (scores_scratch) {
            // Large-N variant: needs scratch buffer
            [enc setBuffer:scores_scratch offset:0 atIndex:4];
            [enc setBuffer:N_buf offset:0 atIndex:5];
            [enc setBuffer:D_buf offset:0 atIndex:6];
            [enc setBuffer:K_topk_buf offset:0 atIndex:7];
        } else {
            [enc setBuffer:N_buf offset:0 atIndex:4];
            [enc setBuffer:D_buf offset:0 atIndex:5];
            [enc setBuffer:K_topk_buf offset:0 atIndex:6];
        }

        NSUInteger tgSize = MIN((NSUInteger)256, (NSUInteger)fused_pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreadgroups:MTLSizeMake(H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];

        uint64_t t0 = now_ns();
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t1 = now_ns();
        return ns_to_us(t1 - t0);
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity for correctness check
// ---------------------------------------------------------------------------
static float cosine_sim(const float* a, const float* b, size_t n) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------
static void print_system_info() {
    printf("================================================================\n");
    printf("  UMMA Phase 1R — GPU-Batched Multi-Head Sparse Attention\n");
    printf("================================================================\n\n");
    char chip[64] = "Unknown";
    size_t chip_len = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, NULL, 0);
    uint64_t memsize = 0; size_t ms = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &ms, NULL, 0);
    printf("  CPU: %s\n", chip);
    printf("  RAM: %.0f GB\n", (double)memsize / 1073741824.0);
    if (g_device) printf("  GPU: %s\n", [[g_device name] UTF8String]);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Main benchmark loop
// ---------------------------------------------------------------------------
static void run_benchmark(uint32_t H, uint32_t top_k, bool markdown) {
    const uint32_t D = 128;
    const uint32_t D4 = D / 4;
    uint32_t Ns[] = {1024, 4096, 16384};
    int nNs = 3;

    printf("  Config: H=%u heads, D=%u, top_k=%u\n\n", H, D, top_k);

    // Build pipelines
    auto gemv_pso     = make_pipeline("batched_gemv_vec4");
    auto topk_par_pso = make_pipeline("batched_topk_parallel");
    auto softmax_pso  = make_pipeline("batched_softmax");
    auto gather_pso   = make_pipeline("batched_gather_multiply_cached");
    auto fused_pso    = make_pipeline("fused_multihead_sparse_attention");
    auto fused_lg_pso = make_pipeline("fused_multihead_sparse_attention_large");

    if (!gemv_pso || !topk_par_pso || !softmax_pso || !gather_pso || !fused_pso) {
        fprintf(stderr, "ERROR: Pipeline creation failed\n");
        return;
    }

    if (markdown) {
        printf("| N | Heads | CPU µs | GPU-Pipeline µs | GPU-Fused µs | CPU/Fused | Cosine |\n");
        printf("|---:|---:|---:|---:|---:|---:|---:|\n");
    }

    for (int ni = 0; ni < nNs; ni++) {
        uint32_t N = Ns[ni];

        size_t KV_bytes = (size_t)H * N * D * sizeof(float);
        size_t Q_bytes  = (size_t)H * D * sizeof(float);
        size_t out_bytes = Q_bytes;

        // Check memory (K + V + Q + out + scores + buffers < 14GB)
        double total_mb = (2.0 * KV_bytes + Q_bytes + out_bytes + H * N * 4) / 1e6;
        if (total_mb > 12000) {
            printf("  SKIP N=%u H=%u (%.0f MB exceeds safe limit)\n", N, H, total_mb);
            continue;
        }

        // Generate random data
        srand(42);
        std::vector<float> Q_data(H * D);
        std::vector<float> K_data((size_t)H * N * D);
        std::vector<float> V_data((size_t)H * N * D);
        for (auto& v : Q_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;
        for (auto& v : K_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;
        for (auto& v : V_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;

        // --- CPU baseline ---
        const int WARMUP = 3;
        const int ITERS = 10;

        for (int i = 0; i < WARMUP; i++) {
            cpu_multihead_attention(Q_data.data(), K_data.data(), V_data.data(), H, N, D, top_k);
        }

        std::vector<double> cpu_times;
        AttentionResult cpu_ref;
        for (int i = 0; i < ITERS; i++) {
            uint64_t t0 = now_ns();
            cpu_ref = cpu_multihead_attention(Q_data.data(), K_data.data(), V_data.data(), H, N, D, top_k);
            uint64_t t1 = now_ns();
            cpu_times.push_back(ns_to_us(t1 - t0));
        }
        Stats cpu_stats = compute_stats(cpu_times);

        // --- GPU buffers ---
        id<MTLBuffer> Q_buf = [g_device newBufferWithBytes:Q_data.data() length:Q_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> K_buf = [g_device newBufferWithBytes:K_data.data() length:KV_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> V_buf = [g_device newBufferWithBytes:V_data.data() length:KV_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buf = [g_device newBufferWithLength:(size_t)H * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> idx_buf = [g_device newBufferWithLength:(size_t)H * top_k * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> topk_buf = [g_device newBufferWithLength:(size_t)H * top_k * sizeof(float) options:MTLResourceStorageModeShared];

        // Constant buffers
        id<MTLBuffer> N_buf = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> D_buf = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> D4_buf = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> K_topk_buf = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        *(uint32_t*)N_buf.contents = N;
        *(uint32_t*)D_buf.contents = D;
        *(uint32_t*)D4_buf.contents = D4;
        *(uint32_t*)K_topk_buf.contents = top_k;

        // --- GPU pipeline (4 dispatches) ---
        for (int i = 0; i < WARMUP; i++) {
            gpu_pipeline_attention(Q_buf, K_buf, V_buf, out_buf, scores_buf,
                                   idx_buf, topk_buf,
                                   H, N, D, top_k,
                                   gemv_pso, topk_par_pso, softmax_pso, gather_pso,
                                   N_buf, D_buf, D4_buf, K_topk_buf);
        }
        std::vector<double> pipe_times;
        for (int i = 0; i < ITERS; i++) {
            double t = gpu_pipeline_attention(Q_buf, K_buf, V_buf, out_buf, scores_buf,
                                              idx_buf, topk_buf,
                                              H, N, D, top_k,
                                              gemv_pso, topk_par_pso, softmax_pso, gather_pso,
                                              N_buf, D_buf, D4_buf, K_topk_buf);
            pipe_times.push_back(t);
        }
        Stats pipe_stats = compute_stats(pipe_times);

        // --- GPU fused (1 dispatch) ---
        bool use_large = (N > 4096);
        auto fused_use = use_large ? fused_lg_pso : fused_pso;
        id<MTLBuffer> scratch = use_large ? scores_buf : nil;

        for (int i = 0; i < WARMUP; i++) {
            gpu_fused_attention(Q_buf, K_buf, V_buf, out_buf,
                                H, N, D, top_k, fused_use,
                                N_buf, D_buf, K_topk_buf, scratch);
        }
        std::vector<double> fused_times;
        for (int i = 0; i < ITERS; i++) {
            double t = gpu_fused_attention(Q_buf, K_buf, V_buf, out_buf,
                                            H, N, D, top_k, fused_use,
                                            N_buf, D_buf, K_topk_buf, scratch);
            fused_times.push_back(t);
        }
        Stats fused_stats = compute_stats(fused_times);

        // --- Correctness: compare GPU fused output to CPU ---
        float* gpu_out = (float*)out_buf.contents;
        float cos = cosine_sim(cpu_ref.output.data(), gpu_out, H * D);

        double speedup_fused = cpu_stats.median_us / fused_stats.median_us;

        if (markdown) {
            printf("| %u | %u | %.0f | %.0f | %.0f | **%.2fx** | %.4f |\n",
                   N, H, cpu_stats.median_us, pipe_stats.median_us,
                   fused_stats.median_us, speedup_fused, cos);
        } else {
            printf("  N=%6u H=%2u:  CPU=%8.0f µs  Pipeline=%8.0f µs  Fused=%8.0f µs  "
                   "Speedup=%.2fx  cos=%.4f%s\n",
                   N, H, cpu_stats.median_us, pipe_stats.median_us,
                   fused_stats.median_us, speedup_fused, cos,
                   speedup_fused > 1.0 ? " <<<" : "");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    @autoreleasepool {
        init_timing();
        if (!init_metal()) return 1;

        bool markdown = false;
        uint32_t heads_override = 0;
        uint32_t topk_override = 0;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--markdown") == 0) markdown = true;
            else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads_override = atoi(argv[++i]);
            else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) topk_override = atoi(argv[++i]);
            else if (strcmp(argv[i], "--help") == 0) {
                printf("Usage: %s [--markdown] [--heads H] [--topk K]\n", argv[0]);
                return 0;
            }
        }

        print_system_info();

        uint32_t top_k = topk_override ? topk_override : 32;

        if (heads_override) {
            run_benchmark(heads_override, top_k, markdown);
        } else {
            // Sweep heads: 1, 8, 16, 32
            uint32_t head_configs[] = {1, 8, 16, 32};
            for (auto H : head_configs) {
                printf("----------------------------------------------------------------\n");
                run_benchmark(H, top_k, markdown);
                printf("\n");
            }
        }

        printf("================================================================\n");
        printf("  Phase 1R complete.\n");
        printf("================================================================\n");
        return 0;
    }
}
