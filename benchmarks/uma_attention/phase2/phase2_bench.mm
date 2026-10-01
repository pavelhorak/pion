// UMMA Phase 2 — Hybrid Pipeline Benchmark
//
// Measures the actual CPU→GPU hybrid attention pipeline:
//   Mode 1: Sequential  — CPU GEMV + top-k → MTLSharedEvent → GPU V-multiply → wait
//   Mode 2: Pipelined   — CPU(Q_i+1) overlaps GPU(Q_i) via double-buffered events
//   Mode 3: CPU-only    — CPU does everything (baseline)
//
// This produces the measured data the paper needs (Phase 0 only predicted).
//
// Build: make
// Run:   ./phase2_bench [--markdown]

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
#include <thread>
#include <atomic>
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
    if (!g_device) return false;
    g_queue = [g_device newCommandQueue];
    NSError* error = nil;
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* src = [NSString stringWithContentsOfFile:[cwd stringByAppendingPathComponent:@"hybrid_kernels.metal"]
                                             encoding:NSUTF8StringEncoding error:&error];
    if (!src) { fprintf(stderr, "ERROR: Cannot read hybrid_kernels.metal\n"); return false; }
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    g_library = [g_device newLibraryWithSource:src options:opts error:&error];
    if (!g_library) { fprintf(stderr, "ERROR: Shader compilation: %s\n", [[error localizedDescription] UTF8String]); return false; }
    return true;
}

static id<MTLComputePipelineState> make_pipeline(const char* name) {
    id<MTLFunction> func = [g_library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!func) return nil;
    NSError* err = nil;
    return [g_device newComputePipelineStateWithFunction:func error:&err];
}

// ---------------------------------------------------------------------------
// CPU attention (single query)
// ---------------------------------------------------------------------------
static void cpu_attention(
    const float* Q, const float* K, const float* V,
    float* output, uint32_t N, uint32_t D, uint32_t top_k,
    float* scores_buf, uint32_t* idx_buf, float* wt_buf
) {
    float scale = 1.0f / sqrtf((float)D);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, scale,
                K, D, Q, 1, 0.0f, scores_buf, 1);

    // Top-k
    for (uint32_t i = 0; i < top_k; i++) { idx_buf[i] = i; wt_buf[i] = scores_buf[i]; }
    float mn = wt_buf[0]; uint32_t mp = 0;
    for (uint32_t i = 1; i < top_k; i++) { if (wt_buf[i] < mn) { mn = wt_buf[i]; mp = i; } }
    for (uint32_t n = top_k; n < N; n++) {
        float v = scores_buf[n];
        if (v > mn) {
            idx_buf[mp] = n; wt_buf[mp] = v;
            mn = wt_buf[0]; mp = 0;
            for (uint32_t i = 1; i < top_k; i++) { if (wt_buf[i] < mn) { mn = wt_buf[i]; mp = i; } }
        }
    }

    // Softmax
    float mx = wt_buf[0];
    for (uint32_t i = 1; i < top_k; i++) mx = fmaxf(mx, wt_buf[i]);
    float sm = 0;
    for (uint32_t i = 0; i < top_k; i++) { wt_buf[i] = expf(wt_buf[i] - mx); sm += wt_buf[i]; }
    float inv = 1.0f / sm;
    for (uint32_t i = 0; i < top_k; i++) wt_buf[i] *= inv;

    // V gather
    memset(output, 0, D * sizeof(float));
    for (uint32_t i = 0; i < top_k; i++) {
        const float* row = V + idx_buf[i] * D;
        float w = wt_buf[i];
        for (uint32_t d = 0; d < D; d++) output[d] += w * row[d];
    }
}

// ---------------------------------------------------------------------------
// Mode 1: Sequential Hybrid
//   CPU: GEMV + top-k + softmax → write indices+weights to shared buffer
//   Signal MTLSharedEvent
//   GPU: gather V + weighted sum
//   Wait for GPU completion
// ---------------------------------------------------------------------------
struct HybridResult {
    double total_us;
    double cpu_phase_us;
    double signal_us;
    double gpu_phase_us;
};

static HybridResult sequential_hybrid(
    const float* Q, const float* K,
    id<MTLBuffer> V_buf,
    id<MTLBuffer> idx_buf_mtl, id<MTLBuffer> wt_buf_mtl,
    id<MTLBuffer> out_buf_mtl,
    id<MTLBuffer> k_const, id<MTLBuffer> d_const,
    id<MTLComputePipelineState> gather_pso,
    uint32_t N, uint32_t D, uint32_t top_k
) {
    float* scores = (float*)malloc(N * sizeof(float));
    uint32_t* idx_local = (uint32_t*)malloc(top_k * sizeof(uint32_t));
    float* wt_local = (float*)malloc(top_k * sizeof(float));

    float scale = 1.0f / sqrtf((float)D);

    // --- CPU Phase ---
    uint64_t t0 = now_ns();

    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, scale,
                K, D, Q, 1, 0.0f, scores, 1);

    // Top-k
    for (uint32_t i = 0; i < top_k; i++) { idx_local[i] = i; wt_local[i] = scores[i]; }
    float mn = wt_local[0]; uint32_t mp = 0;
    for (uint32_t i = 1; i < top_k; i++) { if (wt_local[i] < mn) { mn = wt_local[i]; mp = i; } }
    for (uint32_t n = top_k; n < N; n++) {
        float v = scores[n];
        if (v > mn) {
            idx_local[mp] = n; wt_local[mp] = v;
            mn = wt_local[0]; mp = 0;
            for (uint32_t i = 1; i < top_k; i++) { if (wt_local[i] < mn) { mn = wt_local[i]; mp = i; } }
        }
    }

    // Softmax
    float mx = wt_local[0];
    for (uint32_t i = 1; i < top_k; i++) mx = fmaxf(mx, wt_local[i]);
    float sm = 0;
    for (uint32_t i = 0; i < top_k; i++) { wt_local[i] = expf(wt_local[i] - mx); sm += wt_local[i]; }
    float inv = 1.0f / sm;
    for (uint32_t i = 0; i < top_k; i++) wt_local[i] *= inv;

    // Write to shared Metal buffers (zero-copy on UMA)
    memcpy(idx_buf_mtl.contents, idx_local, top_k * sizeof(uint32_t));
    memcpy(wt_buf_mtl.contents, wt_local, top_k * sizeof(float));

    uint64_t t1 = now_ns();

    // --- GPU Phase: encode + commit + wait ---
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gather_pso];
        [enc setBuffer:idx_buf_mtl offset:0 atIndex:0];
        [enc setBuffer:wt_buf_mtl offset:0 atIndex:1];
        [enc setBuffer:V_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf_mtl offset:0 atIndex:3];
        [enc setBuffer:k_const offset:0 atIndex:4];
        [enc setBuffer:d_const offset:0 atIndex:5];
        NSUInteger tg = MIN((NSUInteger)D, (NSUInteger)gather_pso.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        uint64_t t2 = now_ns();
        [cb commit];
        [cb waitUntilCompleted];
        uint64_t t3 = now_ns();

        free(scores); free(idx_local); free(wt_local);

        HybridResult r;
        r.cpu_phase_us = ns_to_us(t1 - t0);
        r.signal_us = ns_to_us(t2 - t1);
        r.gpu_phase_us = ns_to_us(t3 - t2);
        r.total_us = ns_to_us(t3 - t0);
        return r;
    }
}

// ---------------------------------------------------------------------------
// Mode 2: Pipelined Hybrid (multi-query)
//   Process Q queries. CPU works on Q(i+1) while GPU works on Q(i).
//   Double-buffered shared buffers + MTLSharedEvent.
//   Measure steady-state throughput (us/query).
// ---------------------------------------------------------------------------
static double pipelined_hybrid(
    const float* Qs,       // [num_queries, D]
    const float* K,        // [N, D]
    id<MTLBuffer> V_buf,
    uint32_t N, uint32_t D, uint32_t top_k,
    uint32_t num_queries,
    id<MTLComputePipelineState> gather_pso
) {
    // Double-buffered Metal shared buffers
    id<MTLBuffer> idx_bufs[2], wt_bufs[2], out_bufs[2];
    for (int b = 0; b < 2; b++) {
        idx_bufs[b] = [g_device newBufferWithLength:top_k * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        wt_bufs[b]  = [g_device newBufferWithLength:top_k * sizeof(float) options:MTLResourceStorageModeShared];
        out_bufs[b] = [g_device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> k_const = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> d_const = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    *(uint32_t*)k_const.contents = top_k;
    *(uint32_t*)d_const.contents = D;

    id<MTLSharedEvent> event = [g_device newSharedEvent];

    float* scores = (float*)malloc(N * sizeof(float));
    uint32_t* idx_local = (uint32_t*)malloc(top_k * sizeof(uint32_t));
    float* wt_local = (float*)malloc(top_k * sizeof(float));
    float scale = 1.0f / sqrtf((float)D);

    uint64_t event_val = 0;

    // Lambda: CPU phase for one query
    auto cpu_phase = [&](uint32_t qi, int buf) {
        const float* Q = Qs + qi * D;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, scale,
                    K, D, Q, 1, 0.0f, scores, 1);

        for (uint32_t i = 0; i < top_k; i++) { idx_local[i] = i; wt_local[i] = scores[i]; }
        float mn = wt_local[0]; uint32_t mp = 0;
        for (uint32_t i = 1; i < top_k; i++) { if (wt_local[i] < mn) { mn = wt_local[i]; mp = i; } }
        for (uint32_t n = top_k; n < N; n++) {
            float v = scores[n];
            if (v > mn) {
                idx_local[mp] = n; wt_local[mp] = v;
                mn = wt_local[0]; mp = 0;
                for (uint32_t i = 1; i < top_k; i++) { if (wt_local[i] < mn) { mn = wt_local[i]; mp = i; } }
            }
        }

        float mx = wt_local[0];
        for (uint32_t i = 1; i < top_k; i++) mx = fmaxf(mx, wt_local[i]);
        float sm = 0;
        for (uint32_t i = 0; i < top_k; i++) { wt_local[i] = expf(wt_local[i] - mx); sm += wt_local[i]; }
        float inv = 1.0f / sm;
        for (uint32_t i = 0; i < top_k; i++) wt_local[i] *= inv;

        memcpy(idx_bufs[buf].contents, idx_local, top_k * sizeof(uint32_t));
        memcpy(wt_bufs[buf].contents, wt_local, top_k * sizeof(float));
    };

    // Lambda: GPU dispatch for one query
    auto gpu_dispatch = [&](int buf) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];

            // Wait for CPU to finish writing this buffer
            [cb encodeWaitForEvent:event value:event_val];

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:gather_pso];
            [enc setBuffer:idx_bufs[buf] offset:0 atIndex:0];
            [enc setBuffer:wt_bufs[buf] offset:0 atIndex:1];
            [enc setBuffer:V_buf offset:0 atIndex:2];
            [enc setBuffer:out_bufs[buf] offset:0 atIndex:3];
            [enc setBuffer:k_const offset:0 atIndex:4];
            [enc setBuffer:d_const offset:0 atIndex:5];
            NSUInteger tg = MIN((NSUInteger)D, (NSUInteger)gather_pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(D, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];

            event_val++;
            [cb encodeSignalEvent:event value:event_val];
            [cb commit];
        }
    };

    // Warmup: run 3 queries sequentially
    for (uint32_t qi = 0; qi < MIN(3u, num_queries); qi++) {
        int buf = qi % 2;
        cpu_phase(qi % num_queries, buf);
        event_val++;
        event.signaledValue = event_val;  // manual signal for warmup
        gpu_dispatch(buf);
        while (event.signaledValue < event_val) {}
    }

    // --- Pipelined execution ---
    // Query 0: CPU phase first
    uint64_t t_start = now_ns();

    cpu_phase(0, 0);
    event_val++;
    event.signaledValue = event_val;
    gpu_dispatch(0);

    // Steady-state: CPU(i+1) overlaps GPU(i)
    for (uint32_t qi = 1; qi < num_queries; qi++) {
        int buf = qi % 2;
        int prev_buf = (qi - 1) % 2;

        // CPU works on next query while GPU processes previous
        cpu_phase(qi % num_queries, buf);

        // Wait for previous GPU to finish (need the buffer back)
        // This is the pipeline stall point
        while (event.signaledValue < event_val) {}

        // Now dispatch GPU for current query
        event_val++;
        event.signaledValue = event_val;
        gpu_dispatch(buf);
    }

    // Wait for last GPU
    while (event.signaledValue < event_val) {}
    uint64_t t_end = now_ns();

    free(scores); free(idx_local); free(wt_local);

    double total_us = ns_to_us(t_end - t_start);
    return total_us / num_queries;  // us per query
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    @autoreleasepool {
        init_timing();
        if (!init_metal()) return 1;

        bool markdown = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--markdown") == 0) markdown = true;
        }

        auto gather_pso = make_pipeline("v_gather_multiply");
        if (!gather_pso) { fprintf(stderr, "ERROR: Pipeline creation failed\n"); return 1; }

        printf("================================================================\n");
        printf("  UMMA Phase 2 — Hybrid Pipeline Benchmark\n");
        printf("================================================================\n\n");

        char chip[64] = "Unknown"; size_t cl = sizeof(chip);
        sysctlbyname("machdep.cpu.brand_string", chip, &cl, NULL, 0);
        printf("  CPU: %s\n  GPU: %s\n\n", chip, [[g_device name] UTF8String]);

        const uint32_t D = 128;
        const uint32_t top_k = 32;
        uint32_t Ns[] = {1024, 4096, 16384, 65536, 131072};
        int nNs = 5;

        const int WARMUP = 3;
        const int ITERS = 20;
        const uint32_t PIPELINE_QUERIES = 64;

        if (markdown) {
            printf("| N | CPU-only µs | Seq-Hybrid µs | (CPU/Signal/GPU) | Pipelined µs/q | CPU/SeqHybrid | CPU/Pipeline |\n");
            printf("|---:|---:|---:|---|---:|---:|---:|\n");
        }

        for (int ni = 0; ni < nNs; ni++) {
            uint32_t N = Ns[ni];
            size_t kv_bytes = (size_t)N * D * sizeof(float);
            if (kv_bytes > 4ULL * 1024 * 1024 * 1024) { printf("  SKIP N=%u\n", N); continue; }

            // Generate data
            srand(42);
            std::vector<float> K_data(N * D), V_data(N * D), Q_data(D);
            for (auto& v : K_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;
            for (auto& v : V_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;
            for (auto& v : Q_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;

            // Multi-query data for pipeline
            std::vector<float> Qs_data(PIPELINE_QUERIES * D);
            for (auto& v : Qs_data) v = ((float)(rand() % 2000) - 1000) * 0.001f;

            // Metal buffers
            id<MTLBuffer> V_buf = [g_device newBufferWithBytes:V_data.data() length:kv_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> idx_buf = [g_device newBufferWithLength:top_k * sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> wt_buf = [g_device newBufferWithLength:top_k * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> out_buf = [g_device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> k_const = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> d_const = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
            *(uint32_t*)k_const.contents = top_k;
            *(uint32_t*)d_const.contents = D;

            // --- CPU-only baseline ---
            std::vector<float> cpu_out(D);
            std::vector<float> scores(N);
            std::vector<uint32_t> idx(top_k);
            std::vector<float> wt(top_k);

            for (int i = 0; i < WARMUP; i++)
                cpu_attention(Q_data.data(), K_data.data(), V_data.data(),
                              cpu_out.data(), N, D, top_k, scores.data(), idx.data(), wt.data());

            std::vector<double> cpu_times;
            for (int i = 0; i < ITERS; i++) {
                uint64_t t0 = now_ns();
                cpu_attention(Q_data.data(), K_data.data(), V_data.data(),
                              cpu_out.data(), N, D, top_k, scores.data(), idx.data(), wt.data());
                uint64_t t1 = now_ns();
                cpu_times.push_back(ns_to_us(t1 - t0));
            }
            Stats cpu_stats = compute_stats(cpu_times);

            // --- Sequential Hybrid ---
            for (int i = 0; i < WARMUP; i++)
                sequential_hybrid(Q_data.data(), K_data.data(), V_buf,
                                  idx_buf, wt_buf, out_buf, k_const, d_const,
                                  gather_pso, N, D, top_k);

            std::vector<double> seq_times;
            double avg_cpu_ph = 0, avg_sig = 0, avg_gpu_ph = 0;
            for (int i = 0; i < ITERS; i++) {
                auto r = sequential_hybrid(Q_data.data(), K_data.data(), V_buf,
                                           idx_buf, wt_buf, out_buf, k_const, d_const,
                                           gather_pso, N, D, top_k);
                seq_times.push_back(r.total_us);
                avg_cpu_ph += r.cpu_phase_us;
                avg_sig += r.signal_us;
                avg_gpu_ph += r.gpu_phase_us;
            }
            Stats seq_stats = compute_stats(seq_times);
            avg_cpu_ph /= ITERS; avg_sig /= ITERS; avg_gpu_ph /= ITERS;

            // --- Pipelined Hybrid ---
            // Warmup
            pipelined_hybrid(Qs_data.data(), K_data.data(), V_buf,
                             N, D, top_k, PIPELINE_QUERIES, gather_pso);

            std::vector<double> pipe_times;
            for (int i = 0; i < ITERS; i++) {
                double us_per_q = pipelined_hybrid(Qs_data.data(), K_data.data(), V_buf,
                                                    N, D, top_k, PIPELINE_QUERIES, gather_pso);
                pipe_times.push_back(us_per_q);
            }
            Stats pipe_stats = compute_stats(pipe_times);

            double seq_ratio = cpu_stats.median_us / seq_stats.median_us;
            double pipe_ratio = cpu_stats.median_us / pipe_stats.median_us;

            if (markdown) {
                printf("| %u | %.0f | %.0f | (%.0f/%.0f/%.0f) | %.0f | %.2fx | %.2fx |\n",
                       N, cpu_stats.median_us, seq_stats.median_us,
                       avg_cpu_ph, avg_sig, avg_gpu_ph,
                       pipe_stats.median_us, seq_ratio, pipe_ratio);
            } else {
                printf("  N=%6u:\n", N);
                printf("    CPU-only:    %8.0f µs\n", cpu_stats.median_us);
                printf("    Seq-Hybrid:  %8.0f µs  (cpu=%.0f + sig=%.0f + gpu=%.0f)  ratio=%.2fx\n",
                       seq_stats.median_us, avg_cpu_ph, avg_sig, avg_gpu_ph, seq_ratio);
                printf("    Pipelined:   %8.0f µs/query (%.0f total / %u queries)  ratio=%.2fx\n",
                       pipe_stats.median_us, pipe_stats.median_us * PIPELINE_QUERIES,
                       PIPELINE_QUERIES, pipe_ratio);
                printf("    Winner: %s\n\n",
                       cpu_stats.median_us <= seq_stats.median_us && cpu_stats.median_us <= pipe_stats.median_us
                       ? "CPU-only" : pipe_stats.median_us < seq_stats.median_us ? "Pipelined" : "Seq-Hybrid");
            }
        }

        printf("================================================================\n");
        printf("  Phase 2 complete.\n");
        printf("================================================================\n");
        return 0;
    }
}
