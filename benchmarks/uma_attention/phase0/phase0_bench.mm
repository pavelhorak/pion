// UMMA Phase 0 — Hardware Characterization for Apple Silicon UMA
// Experiments 0.1 (bandwidth contention), 0.2 (dispatch latency), 0.3 (SLC)
//
// Build:
//   make phase0
// Run:
//   ./phase0_bench              # all experiments
//   ./phase0_bench --bw         # bandwidth only
//   ./phase0_bench --dispatch   # dispatch latency only
//   ./phase0_bench --slc        # SLC probing only
//   ./phase0_bench --markdown   # output markdown tables

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#import <pthread.h>
#import <sys/sysctl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <functional>

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------
static mach_timebase_info_data_t g_timebase;

static void init_timing() {
    mach_timebase_info(&g_timebase);
}

// Returns nanoseconds
static inline uint64_t now_ns() {
    return mach_absolute_time() * g_timebase.numer / g_timebase.denom;
}

static inline double ns_to_us(uint64_t ns) { return (double)ns / 1000.0; }
static inline double ns_to_ms(uint64_t ns) { return (double)ns / 1e6; }

struct Stats {
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
    double median_us;
    double p95_us;
};

static Stats compute_stats(const std::vector<double>& times_us) {
    Stats s{};
    if (times_us.empty()) return s;

    std::vector<double> sorted = times_us;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0;
    for (auto t : sorted) sum += t;
    s.mean_us = sum / sorted.size();
    s.min_us = sorted.front();
    s.max_us = sorted.back();
    s.median_us = sorted[sorted.size() / 2];
    s.p95_us = sorted[(size_t)(sorted.size() * 0.95)];

    double var = 0;
    for (auto t : sorted) var += (t - s.mean_us) * (t - s.mean_us);
    s.std_us = sqrt(var / sorted.size());

    return s;
}

// ---------------------------------------------------------------------------
// Global Metal state
// ---------------------------------------------------------------------------
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;

static bool init_metal() {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "ERROR: No Metal device found\n");
        return false;
    }
    g_queue = [g_device newCommandQueue];

    // Try pre-compiled metallib first, fall back to runtime compilation
    NSError* error = nil;
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* libPath = [cwd stringByAppendingPathComponent:@"uma_kernels.metallib"];
    NSURL* libURL = [NSURL fileURLWithPath:libPath];
    g_library = [g_device newLibraryWithURL:libURL error:&error];

    if (!g_library) {
        // Runtime compile from .metal source
        fprintf(stderr, "INFO: No metallib found, compiling shaders at runtime...\n");
        NSString* srcPath = [cwd stringByAppendingPathComponent:@"uma_kernels.metal"];
        NSString* source = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            fprintf(stderr, "ERROR: Cannot read uma_kernels.metal: %s\n",
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
    }
    return true;
}

static id<MTLComputePipelineState> make_pipeline(const char* name) {
    NSString* fn = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [g_library newFunctionWithName:fn];
    if (!func) {
        fprintf(stderr, "ERROR: Kernel '%s' not found in library\n", name);
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        fprintf(stderr, "ERROR: Pipeline creation failed for '%s': %s\n",
                name, [[error localizedDescription] UTF8String]);
    }
    return pso;
}

// ---------------------------------------------------------------------------
// Experiment 0.1: Memory Bandwidth Under Contention
// ---------------------------------------------------------------------------

// CPU STREAM-style bandwidth test
struct BandwidthResult {
    const char* label;
    double gbps;
    double latency_ms;
};

static BandwidthResult cpu_stream_read(size_t bytes) {
    size_t count = bytes / sizeof(float);
    float* buf = (float*)malloc(bytes);
    // Initialize to prevent lazy allocation
    memset(buf, 0x42, bytes);

    const int ITERS = 20;
    std::vector<double> times;
    for (int iter = 0; iter < ITERS; iter++) {
        uint64_t t0 = now_ns();
        // Use vDSP for vectorized read (auto-selects NEON/AMX)
        float sum = 0;
        vDSP_sve(buf, 1, &sum, count);
        uint64_t t1 = now_ns();
        (void)sum;
        times.push_back(ns_to_us(t1 - t0));
    }

    Stats s = compute_stats(times);
    double gbps = ((double)bytes / (s.median_us / 1e6)) / 1e9;
    free(buf);
    return {"CPU-Read (vDSP)", gbps, s.median_us / 1000.0};
}

static BandwidthResult cpu_stream_copy(size_t bytes) {
    float* src = (float*)malloc(bytes);
    float* dst = (float*)malloc(bytes);
    memset(src, 0x42, bytes);
    memset(dst, 0, bytes);

    const int ITERS = 30;
    std::vector<double> times;
    volatile float* vdst = (volatile float*)dst;
    for (int iter = 0; iter < ITERS; iter++) {
        uint64_t t0 = now_ns();
        memcpy(dst, src, bytes);
        // Prevent compiler from eliding the memcpy
        float sink = vdst[0] + vdst[bytes/sizeof(float) - 1];
        (void)sink;
        uint64_t t1 = now_ns();
        times.push_back(ns_to_us(t1 - t0));
    }

    Stats s = compute_stats(times);
    double gbps = (2.0 * bytes / (s.median_us / 1e6)) / 1e9;  // read + write
    free(src);
    free(dst);
    return {"CPU-Copy (memcpy)", gbps, s.median_us / 1000.0};
}

static BandwidthResult cpu_stream_triad(size_t bytes) {
    size_t count = bytes / sizeof(float);
    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* c = (float*)malloc(bytes);
    for (size_t i = 0; i < count; i++) {
        a[i] = (float)(i % 1000) * 0.001f;
        b[i] = (float)((i + 500) % 1000) * 0.001f;
    }
    memset(c, 0, bytes);

    const int ITERS = 20;
    std::vector<double> times;
    float scalar = 3.14f;
    for (int iter = 0; iter < ITERS; iter++) {
        uint64_t t0 = now_ns();
        // c = a + scalar * b  (via Accelerate)
        vDSP_vsma(b, 1, &scalar, a, 1, c, 1, count);
        uint64_t t1 = now_ns();
        times.push_back(ns_to_us(t1 - t0));
    }

    Stats s = compute_stats(times);
    // 2 reads + 1 write = 3 memory streams
    double gbps = (3.0 * bytes / (s.median_us / 1e6)) / 1e9;
    free(a); free(b); free(c);
    return {"CPU-Triad (vDSP)", gbps, s.median_us / 1000.0};
}

static BandwidthResult gpu_copy_bandwidth(size_t bytes) {
    auto pso = make_pipeline("copy_bandwidth");
    if (!pso) return {"GPU-Copy", 0, 0};

    size_t count = bytes / sizeof(float);
    id<MTLBuffer> src = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(src.contents, 0x42, bytes);

    const int ITERS = 30;
    const int WARMUP = 5;

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            MTLSize grid = MTLSizeMake(count, 1, 1);
            MTLSize tg = MTLSizeMake(MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    std::vector<double> times;
    for (int iter = 0; iter < ITERS; iter++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            MTLSize grid = MTLSizeMake(count, 1, 1);
            MTLSize tg = MTLSizeMake(MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            uint64_t t0 = now_ns();
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t t1 = now_ns();
            times.push_back(ns_to_us(t1 - t0));
        }
    }

    Stats s = compute_stats(times);
    // read + write
    double gbps = (2.0 * bytes / (s.median_us / 1e6)) / 1e9;
    return {"GPU-Copy (Metal)", gbps, s.median_us / 1000.0};
}

static BandwidthResult gpu_read_bandwidth(size_t bytes) {
    auto pso = make_pipeline("read_bandwidth");
    if (!pso) return {"GPU-Read", 0, 0};

    size_t count4 = bytes / sizeof(float) / 4;  // float4 elements
    id<MTLBuffer> src = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    memset(src.contents, 0x42, bytes);

    const int ITERS = 30;
    const int WARMUP = 5;
    NSUInteger tgSize = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
    MTLSize grid = MTLSizeMake(count4, 1, 1);
    MTLSize tg = MTLSizeMake(tgSize, 1, 1);

    for (int i = 0; i < WARMUP; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    std::vector<double> times;
    for (int iter = 0; iter < ITERS; iter++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            uint64_t t0 = now_ns();
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t t1 = now_ns();
            times.push_back(ns_to_us(t1 - t0));
        }
    }

    Stats s = compute_stats(times);
    double gbps = ((double)bytes / (s.median_us / 1e6)) / 1e9;
    return {"GPU-Read (Metal)", gbps, s.median_us / 1000.0};
}

// Simultaneous CPU + GPU bandwidth
struct SimultaneousResult {
    double cpu_gbps;
    double gpu_gbps;
    double total_gbps;
    double cpu_solo_gbps;
    double gpu_solo_gbps;
    double cpu_retention;  // cpu_gbps / cpu_solo_gbps
    double gpu_retention;
    double total_ratio;    // total / max(cpu_solo, gpu_solo)
};

static SimultaneousResult simultaneous_bandwidth(size_t bytes) {
    // Allocate separate buffers for CPU and GPU to minimize contention
    size_t count = bytes / sizeof(float);

    // CPU buffers (plain malloc — not MTLBuffer to avoid Metal overhead on CPU side)
    float* cpu_src = (float*)malloc(bytes);
    float* cpu_dst = (float*)malloc(bytes);
    memset(cpu_src, 0x42, bytes);
    memset(cpu_dst, 0, bytes);

    // GPU buffers
    auto pso = make_pipeline("copy_bandwidth");
    id<MTLBuffer> gpu_src = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> gpu_dst = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(gpu_src.contents, 0x42, bytes);

    // Solo measurements first
    auto cpu_solo = cpu_stream_copy(bytes);
    auto gpu_solo = gpu_copy_bandwidth(bytes);

    // Simultaneous: CPU copies in a thread while GPU copies via Metal
    const int ITERS = 15;
    std::vector<double> cpu_times, gpu_times;

    NSUInteger tgSize = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);

    for (int iter = 0; iter < ITERS; iter++) {
        @autoreleasepool {
            std::atomic<bool> go{false};
            double cpu_us = 0, gpu_us = 0;

            // Prepare GPU command buffer
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:gpu_src offset:0 atIndex:0];
            [enc setBuffer:gpu_dst offset:0 atIndex:1];
            MTLSize grid = MTLSizeMake(count, 1, 1);
            MTLSize tg = MTLSizeMake(tgSize, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            // CPU thread
            std::thread cpu_thread([&]() {
                while (!go.load(std::memory_order_acquire)) {}
                uint64_t t0 = now_ns();
                memcpy(cpu_dst, cpu_src, bytes);
                uint64_t t1 = now_ns();
                cpu_us = ns_to_us(t1 - t0);
            });

            // Launch both as close together as possible
            go.store(true, std::memory_order_release);
            uint64_t gt0 = now_ns();
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t gt1 = now_ns();
            gpu_us = ns_to_us(gt1 - gt0);

            cpu_thread.join();

            cpu_times.push_back(cpu_us);
            gpu_times.push_back(gpu_us);
        }
    }

    // Compute medians
    std::sort(cpu_times.begin(), cpu_times.end());
    std::sort(gpu_times.begin(), gpu_times.end());
    double cpu_median = cpu_times[cpu_times.size() / 2];
    double gpu_median = gpu_times[gpu_times.size() / 2];

    double cpu_gbps = (2.0 * bytes / (cpu_median / 1e6)) / 1e9;
    double gpu_gbps = (2.0 * bytes / (gpu_median / 1e6)) / 1e9;

    free(cpu_src); free(cpu_dst);

    SimultaneousResult r;
    r.cpu_gbps = cpu_gbps;
    r.gpu_gbps = gpu_gbps;
    r.total_gbps = cpu_gbps + gpu_gbps;
    r.cpu_solo_gbps = cpu_solo.gbps;
    r.gpu_solo_gbps = gpu_solo.gbps;
    r.cpu_retention = cpu_gbps / cpu_solo.gbps;
    r.gpu_retention = gpu_gbps / gpu_solo.gbps;
    r.total_ratio = r.total_gbps / fmax(cpu_solo.gbps, gpu_solo.gbps);
    return r;
}

static void run_experiment_01(bool markdown) {
    printf("\n");
    printf("================================================================\n");
    printf("  Experiment 0.1: Memory Bandwidth Under Contention\n");
    printf("================================================================\n\n");

    size_t sizes[] = {
        4   * 1024 * 1024,   // 4 MB (fits in SLC ~16MB)
        16  * 1024 * 1024,   // 16 MB (SLC boundary)
        64  * 1024 * 1024,   // 64 MB (DRAM)
        256 * 1024 * 1024,   // 256 MB (large DRAM)
    };
    const char* size_labels[] = {"4MB", "16MB", "64MB", "256MB"};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    // --- Solo bandwidth ---
    printf("  Solo bandwidth (no contention):\n");
    if (markdown) {
        printf("\n| Size | CPU-Read GB/s | CPU-Copy GB/s | CPU-Triad GB/s | GPU-Read GB/s | GPU-Copy GB/s |\n");
        printf("|------|---:|---:|---:|---:|---:|\n");
    }

    for (int i = 0; i < nsizes; i++) {
        auto cr = cpu_stream_read(sizes[i]);
        auto cc = cpu_stream_copy(sizes[i]);
        auto ct = cpu_stream_triad(sizes[i]);
        auto gr = gpu_read_bandwidth(sizes[i]);
        auto gc = gpu_copy_bandwidth(sizes[i]);

        if (markdown) {
            printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                   size_labels[i], cr.gbps, cc.gbps, ct.gbps, gr.gbps, gc.gbps);
        } else {
            printf("    %5s: CPU-Read=%6.1f  CPU-Copy=%6.1f  CPU-Triad=%6.1f  GPU-Read=%6.1f  GPU-Copy=%6.1f GB/s\n",
                   size_labels[i], cr.gbps, cc.gbps, ct.gbps, gr.gbps, gc.gbps);
        }
    }

    // --- Simultaneous bandwidth ---
    printf("\n  Simultaneous CPU+GPU bandwidth:\n");
    if (markdown) {
        printf("\n| Size | CPU solo | GPU solo | CPU simul | GPU simul | Total | CPU retain | GPU retain | Total/Max |\n");
        printf("|------|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    }

    for (int i = 0; i < nsizes; i++) {
        auto r = simultaneous_bandwidth(sizes[i]);
        if (markdown) {
            printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.0f%% | %.0f%% | %.2fx |\n",
                   size_labels[i], r.cpu_solo_gbps, r.gpu_solo_gbps,
                   r.cpu_gbps, r.gpu_gbps, r.total_gbps,
                   r.cpu_retention * 100, r.gpu_retention * 100, r.total_ratio);
        } else {
            printf("    %5s: CPU=%.1f/%.1f(%.0f%%)  GPU=%.1f/%.1f(%.0f%%)  Total=%.1f  Ratio=%.2fx\n",
                   size_labels[i],
                   r.cpu_gbps, r.cpu_solo_gbps, r.cpu_retention * 100,
                   r.gpu_gbps, r.gpu_solo_gbps, r.gpu_retention * 100,
                   r.total_gbps, r.total_ratio);
        }
    }

    printf("\n  GATE G0: Simultaneous > 110%% of single-agent? → check Total/Max column\n");
}

// ---------------------------------------------------------------------------
// Experiment 0.2: Metal Dispatch Latency Profiling
// ---------------------------------------------------------------------------

struct DispatchResult {
    const char* label;
    Stats stats;
};

static DispatchResult measure_dispatch(const char* label, const char* kernel_name,
                                        std::function<void(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>)> setup,
                                        MTLSize grid, MTLSize tg,
                                        int warmup = 10, int iters = 100) {
    auto pso = make_pipeline(kernel_name);
    if (!pso) return {label, {}};

    // Warmup
    for (int i = 0; i < warmup; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            setup(enc, pso);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    // Measure
    std::vector<double> times;
    for (int i = 0; i < iters; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            setup(enc, pso);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];

            uint64_t t0 = now_ns();
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t t1 = now_ns();
            times.push_back(ns_to_us(t1 - t0));
        }
    }

    return {label, compute_stats(times)};
}

// Measure encode + commit + wait as separate phases
static void measure_dispatch_phases(int iters = 100) {
    auto pso = make_pipeline("trivial_kernel");
    if (!pso) return;

    id<MTLBuffer> out = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];

    // Warmup
    for (int i = 0; i < 10; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    std::vector<double> encode_times, commit_times, wait_times, total_times;

    for (int i = 0; i < iters; i++) {
        @autoreleasepool {
            uint64_t t0 = now_ns();
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            uint64_t t1 = now_ns();

            [cb commit];
            uint64_t t2 = now_ns();

            [cb waitUntilCompleted];
            uint64_t t3 = now_ns();

            encode_times.push_back(ns_to_us(t1 - t0));
            commit_times.push_back(ns_to_us(t2 - t1));
            wait_times.push_back(ns_to_us(t3 - t2));
            total_times.push_back(ns_to_us(t3 - t0));
        }
    }

    auto es = compute_stats(encode_times);
    auto cs = compute_stats(commit_times);
    auto ws = compute_stats(wait_times);
    auto ts = compute_stats(total_times);

    printf("\n  Dispatch phase breakdown (trivial 1-thread kernel, %d iters):\n", iters);
    printf("    Encode:  median=%.1f µs  mean=%.1f µs  p95=%.1f µs\n", es.median_us, es.mean_us, es.p95_us);
    printf("    Commit:  median=%.1f µs  mean=%.1f µs  p95=%.1f µs\n", cs.median_us, cs.mean_us, cs.p95_us);
    printf("    Wait:    median=%.1f µs  mean=%.1f µs  p95=%.1f µs\n", ws.median_us, ws.mean_us, ws.p95_us);
    printf("    Total:   median=%.1f µs  mean=%.1f µs  p95=%.1f µs\n", ts.median_us, ts.mean_us, ts.p95_us);
}

// Measure using MTLSharedEvent for signaling latency
static void measure_shared_event_latency(int iters = 100) {
    auto pso = make_pipeline("trivial_kernel");
    if (!pso) return;

    id<MTLBuffer> out = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLSharedEvent> event = [g_device newSharedEvent];

    // Warmup
    for (int i = 0; i < 10; i++) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb encodeSignalEvent:event value:i + 1];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    std::vector<double> signal_times;
    uint64_t event_val = 100;

    for (int i = 0; i < iters; i++) {
        @autoreleasepool {
            event_val++;
            id<MTLCommandBuffer> cb = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:out offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb encodeSignalEvent:event value:event_val];
            [cb commit];

            // Spin-wait on the event (measures signal-to-CPU-notify latency)
            uint64_t t0 = now_ns();
            while (event.signaledValue < event_val) {
                // Spin
            }
            uint64_t t1 = now_ns();
            signal_times.push_back(ns_to_us(t1 - t0));
        }
    }

    auto ss = compute_stats(signal_times);
    printf("\n  MTLSharedEvent signal latency (GPU→CPU, spin-wait, %d iters):\n", iters);
    printf("    median=%.1f µs  mean=%.1f µs  min=%.1f µs  p95=%.1f µs\n",
           ss.median_us, ss.mean_us, ss.min_us, ss.p95_us);
}

static void run_experiment_02(bool markdown) {
    printf("\n");
    printf("================================================================\n");
    printf("  Experiment 0.2: Metal Dispatch Latency Profiling\n");
    printf("================================================================\n");

    // Pre-allocate buffers for reuse
    const uint32_t D = 128;
    const uint32_t D4 = D / 4;
    uint32_t Ns[] = {512, 1024, 4096, 16384, 32768, 65536, 131072};
    int nNs = sizeof(Ns) / sizeof(Ns[0]);

    id<MTLBuffer> outBuf = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];

    // --- Empty kernel ---
    auto empty = measure_dispatch("Empty kernel (0 threads)", "empty_kernel",
        [](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {},
        MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));

    // --- Trivial kernel (1 thread) ---
    auto trivial = measure_dispatch("Trivial kernel (1 thread)", "trivial_kernel",
        [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso) {
            [enc setBuffer:outBuf offset:0 atIndex:0];
        },
        MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));

    printf("\n  Fixed-cost dispatch latency:\n");
    printf("    %-35s  median=%6.1f µs  mean=%6.1f µs  min=%6.1f µs  p95=%6.1f µs\n",
           empty.label, empty.stats.median_us, empty.stats.mean_us, empty.stats.min_us, empty.stats.p95_us);
    printf("    %-35s  median=%6.1f µs  mean=%6.1f µs  min=%6.1f µs  p95=%6.1f µs\n",
           trivial.label, trivial.stats.median_us, trivial.stats.mean_us, trivial.stats.min_us, trivial.stats.p95_us);

    // --- Phase breakdown ---
    measure_dispatch_phases(200);

    // --- MTLSharedEvent latency ---
    measure_shared_event_latency(200);

    // --- GEMV at various N ---
    printf("\n  GEMV kernel (1×%u @ %u×N) dispatch+compute:\n", D, D);
    if (markdown) {
        printf("\n| N | Median µs | Mean µs | Min µs | P95 µs | Eff BW GB/s |\n");
        printf("|---:|---:|---:|---:|---:|---:|\n");
    }

    for (int ni = 0; ni < nNs; ni++) {
        uint32_t N = Ns[ni];
        size_t mat_bytes = (size_t)N * D * sizeof(float);
        id<MTLBuffer> mat = [g_device newBufferWithLength:mat_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vec = [g_device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out = [g_device newBufferWithLength:N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> dBuf = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Fill with random data
        float* mp = (float*)mat.contents;
        float* vp = (float*)vec.contents;
        for (size_t j = 0; j < (size_t)N * D; j++) mp[j] = (float)(rand() % 1000) * 0.001f;
        for (uint32_t j = 0; j < D; j++) vp[j] = (float)(rand() % 1000) * 0.001f;
        *(uint32_t*)dBuf.contents = D4;

        auto pso = make_pipeline("gemv_vec4_kernel");
        NSUInteger tgSize = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);

        auto result = measure_dispatch("GEMV", "gemv_vec4_kernel",
            [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p) {
                [enc setBuffer:mat offset:0 atIndex:0];
                [enc setBuffer:vec offset:0 atIndex:1];
                [enc setBuffer:out offset:0 atIndex:2];
                [enc setBuffer:dBuf offset:0 atIndex:3];
            },
            MTLSizeMake(N, 1, 1), MTLSizeMake(tgSize, 1, 1),
            10, 50);

        double bytes_read = (double)N * D * 4 + D * 4;
        double eff_bw = (bytes_read / (result.stats.median_us / 1e6)) / 1e9;

        if (markdown) {
            printf("| %u | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                   N, result.stats.median_us, result.stats.mean_us, result.stats.min_us,
                   result.stats.p95_us, eff_bw);
        } else {
            printf("    N=%6u:  median=%7.1f µs  mean=%7.1f µs  min=%7.1f µs  p95=%7.1f µs  BW=%5.1f GB/s\n",
                   N, result.stats.median_us, result.stats.mean_us,
                   result.stats.min_us, result.stats.p95_us, eff_bw);
        }
    }

    // --- Gather+multiply at various k ---
    printf("\n  Gather+multiply kernel (k gathered rows × D=%u):\n", D);
    uint32_t ks[] = {8, 16, 32, 64, 128, 256, 512};
    int nks = sizeof(ks) / sizeof(ks[0]);
    uint32_t N_gather = 65536;

    id<MTLBuffer> V_buf = [g_device newBufferWithLength:(size_t)N_gather * D * sizeof(float) options:MTLResourceStorageModeShared];
    float* vp = (float*)V_buf.contents;
    for (size_t j = 0; j < (size_t)N_gather * D; j++) vp[j] = (float)(rand() % 1000) * 0.001f;

    if (markdown) {
        printf("\n| k | Median µs | Mean µs | Min µs | P95 µs |\n");
        printf("|---:|---:|---:|---:|---:|\n");
    }

    for (int ki = 0; ki < nks; ki++) {
        uint32_t k = ks[ki];
        id<MTLBuffer> idx_buf = [g_device newBufferWithLength:k * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> wt_buf  = [g_device newBufferWithLength:k * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [g_device newBufferWithLength:D * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buf   = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> d_buf   = [g_device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        uint32_t* ip = (uint32_t*)idx_buf.contents;
        float* wp = (float*)wt_buf.contents;
        for (uint32_t j = 0; j < k; j++) {
            ip[j] = rand() % N_gather;
            wp[j] = 1.0f / k;
        }
        *(uint32_t*)k_buf.contents = k;
        *(uint32_t*)d_buf.contents = D;

        auto pso = make_pipeline("gather_multiply_cached_kernel");
        NSUInteger tgSize = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        // D threads — one per output dimension
        MTLSize grid = MTLSizeMake(D, 1, 1);
        MTLSize tg = MTLSizeMake(MIN((NSUInteger)D, tgSize), 1, 1);

        auto result = measure_dispatch("Gather+Mul", "gather_multiply_cached_kernel",
            [&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p) {
                [enc setBuffer:idx_buf offset:0 atIndex:0];
                [enc setBuffer:wt_buf offset:0 atIndex:1];
                [enc setBuffer:V_buf offset:0 atIndex:2];
                [enc setBuffer:out_buf offset:0 atIndex:3];
                [enc setBuffer:k_buf offset:0 atIndex:4];
                [enc setBuffer:d_buf offset:0 atIndex:5];
            },
            grid, tg, 10, 100);

        if (markdown) {
            printf("| %u | %.1f | %.1f | %.1f | %.1f |\n",
                   k, result.stats.median_us, result.stats.mean_us,
                   result.stats.min_us, result.stats.p95_us);
        } else {
            printf("    k=%4u:  median=%6.1f µs  mean=%6.1f µs  min=%6.1f µs  p95=%6.1f µs\n",
                   k, result.stats.median_us, result.stats.mean_us,
                   result.stats.min_us, result.stats.p95_us);
        }
    }

    // --- CPU GEMV comparison (Accelerate BLAS) ---
    printf("\n  CPU GEMV comparison (cblas_sgemv, Accelerate):\n");
    if (markdown) {
        printf("\n| N | Median µs | Mean µs | Min µs | Eff BW GB/s |\n");
        printf("|---:|---:|---:|---:|---:|\n");
    }

    for (int ni = 0; ni < nNs; ni++) {
        uint32_t N = Ns[ni];
        size_t mat_bytes = (size_t)N * D * sizeof(float);
        float* mat = (float*)malloc(mat_bytes);
        float* vec = (float*)malloc(D * sizeof(float));
        float* out = (float*)malloc(N * sizeof(float));

        for (size_t j = 0; j < (size_t)N * D; j++) mat[j] = (float)(rand() % 1000) * 0.001f;
        for (uint32_t j = 0; j < D; j++) vec[j] = (float)(rand() % 1000) * 0.001f;

        // Warmup
        for (int i = 0; i < 10; i++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, 1.0f, mat, D, vec, 1, 0.0f, out, 1);
        }

        std::vector<double> times;
        for (int i = 0; i < 100; i++) {
            uint64_t t0 = now_ns();
            cblas_sgemv(CblasRowMajor, CblasNoTrans, N, D, 1.0f, mat, D, vec, 1, 0.0f, out, 1);
            uint64_t t1 = now_ns();
            times.push_back(ns_to_us(t1 - t0));
        }

        Stats s = compute_stats(times);
        double bytes_read = (double)N * D * 4 + D * 4;
        double eff_bw = (bytes_read / (s.median_us / 1e6)) / 1e9;

        if (markdown) {
            printf("| %u | %.1f | %.1f | %.1f | %.1f |\n", N, s.median_us, s.mean_us, s.min_us, eff_bw);
        } else {
            printf("    N=%6u:  median=%7.1f µs  mean=%7.1f µs  min=%7.1f µs  BW=%5.1f GB/s\n",
                   N, s.median_us, s.mean_us, s.min_us, eff_bw);
        }
        free(mat); free(vec); free(out);
    }

    printf("\n  GATE G1: Metal dispatch < 20µs (pre-compiled pipeline)? → check empty/trivial kernel median\n");
}

// ---------------------------------------------------------------------------
// Experiment 0.3: SLC Behavior Under Split Access
// ---------------------------------------------------------------------------

static void run_experiment_03(bool markdown) {
    printf("\n");
    printf("================================================================\n");
    printf("  Experiment 0.3: SLC Behavior Under Split Access\n");
    printf("================================================================\n");

    // Test: GPU reads buffer while CPU reads a different buffer simultaneously
    // Compare GPU read bandwidth when CPU is idle vs active
    // The delta tells us how much SLC contention costs

    auto gpu_read_pso = make_pipeline("read_bandwidth");
    if (!gpu_read_pso) return;

    // Sizes that straddle the SLC boundary (~16MB on M4)
    size_t sizes[] = {
        1   * 1024 * 1024,   // 1 MB — definitely fits SLC
        4   * 1024 * 1024,   // 4 MB — fits SLC
        8   * 1024 * 1024,   // 8 MB — fits SLC
        16  * 1024 * 1024,   // 16 MB — SLC boundary
        32  * 1024 * 1024,   // 32 MB — exceeds SLC
        64  * 1024 * 1024,   // 64 MB — DRAM
        128 * 1024 * 1024,   // 128 MB — large DRAM
    };
    const char* size_labels[] = {"1MB", "4MB", "8MB", "16MB", "32MB", "64MB", "128MB"};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n  GPU read bandwidth: solo vs CPU-contended (separate buffers)\n");
    printf("  Tests whether CPU reading buffer A degrades GPU reading buffer B via SLC eviction\n\n");

    if (markdown) {
        printf("| Buffer Size | GPU Solo GB/s | GPU+CPU GB/s | Degradation | CPU buffer |\n");
        printf("|---:|---:|---:|---:|---:|\n");
    } else {
        printf("    %7s  %12s  %12s  %11s  %s\n", "Size", "GPU Solo", "GPU+CPU", "Degradation", "CPU buf");
        printf("    %7s  %12s  %12s  %11s  %s\n", "-------", "--------", "-------", "-----------", "-------");
    }

    for (int si = 0; si < nsizes; si++) {
        size_t buf_size = sizes[si];
        size_t count4 = buf_size / sizeof(float) / 4;

        // GPU buffer (Metal shared)
        id<MTLBuffer> gpu_buf = [g_device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> gpu_out = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        memset(gpu_buf.contents, 0x42, buf_size);

        // CPU buffer (separate allocation — different physical pages)
        float* cpu_buf = (float*)malloc(buf_size);
        memset(cpu_buf, 0x42, buf_size);

        NSUInteger tgSize = MIN((NSUInteger)gpu_read_pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
        MTLSize grid = MTLSizeMake(count4, 1, 1);
        MTLSize tg = MTLSizeMake(tgSize, 1, 1);

        // Warmup
        for (int i = 0; i < 5; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:gpu_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        // --- GPU solo ---
        const int ITERS = 20;
        std::vector<double> solo_times;
        for (int i = 0; i < ITERS; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:gpu_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                uint64_t t0 = now_ns();
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t t1 = now_ns();
                solo_times.push_back(ns_to_us(t1 - t0));
            }
        }

        // --- GPU + CPU concurrent (CPU reads different buffer) ---
        std::vector<double> contended_times;
        for (int i = 0; i < ITERS; i++) {
            @autoreleasepool {
                std::atomic<bool> go{false};
                std::atomic<bool> done{false};

                // CPU thread: continuously read cpu_buf to pollute SLC
                std::thread cpu_thread([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    size_t count = buf_size / sizeof(float);
                    while (!done.load(std::memory_order_acquire)) {
                        float sink = 0;
                        vDSP_sve(cpu_buf, 1, &sink, count);
                        (void)sink;
                    }
                });

                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:gpu_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];

                go.store(true, std::memory_order_release);
                uint64_t t0 = now_ns();
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t t1 = now_ns();
                done.store(true, std::memory_order_release);

                cpu_thread.join();
                contended_times.push_back(ns_to_us(t1 - t0));
            }
        }

        std::sort(solo_times.begin(), solo_times.end());
        std::sort(contended_times.begin(), contended_times.end());
        double solo_median = solo_times[solo_times.size() / 2];
        double contended_median = contended_times[contended_times.size() / 2];

        double solo_gbps = ((double)buf_size / (solo_median / 1e6)) / 1e9;
        double contended_gbps = ((double)buf_size / (contended_median / 1e6)) / 1e9;
        double degradation = 1.0 - (contended_gbps / solo_gbps);

        if (markdown) {
            printf("| %s | %.1f | %.1f | %.1f%% | same size |\n",
                   size_labels[si], solo_gbps, contended_gbps, degradation * 100);
        } else {
            printf("    %7s:  solo=%6.1f GB/s  contended=%6.1f GB/s  degradation=%5.1f%%\n",
                   size_labels[si], solo_gbps, contended_gbps, degradation * 100);
        }

        free(cpu_buf);
    }

    // --- Same buffer test: GPU and CPU both read the SAME Metal shared buffer ---
    printf("\n  GPU read bandwidth: solo vs CPU reading SAME buffer (worst case)\n\n");

    if (markdown) {
        printf("| Buffer Size | GPU Solo GB/s | GPU+CPU Same GB/s | Degradation |\n");
        printf("|---:|---:|---:|---:|\n");
    }

    // Use smaller subset for same-buffer test
    size_t same_sizes[] = {4*1024*1024, 16*1024*1024, 64*1024*1024};
    const char* same_labels[] = {"4MB", "16MB", "64MB"};

    for (int si = 0; si < 3; si++) {
        size_t buf_size = same_sizes[si];
        size_t count4 = buf_size / sizeof(float) / 4;

        // SAME buffer: MTLShared, both CPU and GPU read it
        id<MTLBuffer> shared_buf = [g_device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> gpu_out = [g_device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        memset(shared_buf.contents, 0x42, buf_size);

        NSUInteger tgSize = MIN((NSUInteger)gpu_read_pso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
        MTLSize grid = MTLSizeMake(count4, 1, 1);
        MTLSize tg = MTLSizeMake(tgSize, 1, 1);

        // Warmup
        for (int i = 0; i < 5; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:shared_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }
        }

        // Solo
        const int ITERS = 20;
        std::vector<double> solo_times;
        for (int i = 0; i < ITERS; i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:shared_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                uint64_t t0 = now_ns();
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t t1 = now_ns();
                solo_times.push_back(ns_to_us(t1 - t0));
            }
        }

        // Contended: CPU reads the SAME buffer
        std::vector<double> contended_times;
        float* cpu_ptr = (float*)shared_buf.contents;
        size_t count = buf_size / sizeof(float);

        for (int i = 0; i < ITERS; i++) {
            @autoreleasepool {
                std::atomic<bool> go{false};
                std::atomic<bool> done{false};

                std::thread cpu_thread([&]() {
                    while (!go.load(std::memory_order_acquire)) {}
                    while (!done.load(std::memory_order_acquire)) {
                        float sink = 0;
                        vDSP_sve(cpu_ptr, 1, &sink, count);
                        (void)sink;
                    }
                });

                id<MTLCommandBuffer> cb = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:gpu_read_pso];
                [enc setBuffer:shared_buf offset:0 atIndex:0];
                [enc setBuffer:gpu_out offset:0 atIndex:1];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];

                go.store(true, std::memory_order_release);
                uint64_t t0 = now_ns();
                [cb commit];
                [cb waitUntilCompleted];
                uint64_t t1 = now_ns();
                done.store(true, std::memory_order_release);

                cpu_thread.join();
                contended_times.push_back(ns_to_us(t1 - t0));
            }
        }

        std::sort(solo_times.begin(), solo_times.end());
        std::sort(contended_times.begin(), contended_times.end());
        double solo_median = solo_times[solo_times.size() / 2];
        double contended_median = contended_times[contended_times.size() / 2];

        double solo_gbps = ((double)buf_size / (solo_median / 1e6)) / 1e9;
        double contended_gbps = ((double)buf_size / (contended_median / 1e6)) / 1e9;
        double degradation = 1.0 - (contended_gbps / solo_gbps);

        if (markdown) {
            printf("| %s | %.1f | %.1f | %.1f%% |\n",
                   same_labels[si], solo_gbps, contended_gbps, degradation * 100);
        } else {
            printf("    %7s:  solo=%6.1f GB/s  same-buf=%6.1f GB/s  degradation=%5.1f%%\n",
                   same_labels[si], solo_gbps, contended_gbps, degradation * 100);
        }
    }

    printf("\n  Interpretation: degradation < 5%% at buffer < SLC size → SLC partitions well\n");
    printf("  Degradation > 20%% at buffer > SLC → DRAM contention is real\n");
}

// ---------------------------------------------------------------------------
// System info
// ---------------------------------------------------------------------------
static void print_system_info() {
    printf("================================================================\n");
    printf("  UMMA Phase 0 — Hardware Characterization\n");
    printf("================================================================\n\n");

    // Chip name
    char chip[64] = "Unknown";
    size_t chip_len = sizeof(chip);
    sysctlbyname("machdep.cpu.brand_string", chip, &chip_len, NULL, 0);
    printf("  CPU: %s\n", chip);

    // Physical memory
    uint64_t memsize = 0;
    size_t ms = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &ms, NULL, 0);
    printf("  RAM: %.0f GB unified\n", (double)memsize / (1024.0 * 1024.0 * 1024.0));

    // CPU cores
    int pcores = 0, ecores = 0, total = 0;
    size_t s = sizeof(int);
    sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &s, NULL, 0);
    sysctlbyname("hw.perflevel1.physicalcpu", &ecores, &s, NULL, 0);
    sysctlbyname("hw.physicalcpu", &total, &s, NULL, 0);
    printf("  CPU cores: %d (%dP + %dE)\n", total, pcores, ecores);

    // Metal device
    if (g_device) {
        printf("  GPU: %s\n", [[g_device name] UTF8String]);
        printf("  Metal: %s\n", [g_device supportsFamily:MTLGPUFamilyApple9] ? "Apple9 (M4)" :
                               [g_device supportsFamily:MTLGPUFamilyApple8] ? "Apple8 (M3)" :
                               [g_device supportsFamily:MTLGPUFamilyApple7] ? "Apple7 (M1/M2)" : "Unknown");
        printf("  Max threadgroup size: %lu\n", (unsigned long)g_device.maxThreadsPerThreadgroup.width);
        printf("  Max buffer length: %.0f MB\n", (double)g_device.maxBufferLength / (1024.0 * 1024.0));
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    @autoreleasepool {
        init_timing();

        if (!init_metal()) {
            return 1;
        }

        bool run_bw = false, run_dispatch = false, run_slc = false;
        bool markdown = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bw") == 0) run_bw = true;
            else if (strcmp(argv[i], "--dispatch") == 0) run_dispatch = true;
            else if (strcmp(argv[i], "--slc") == 0) run_slc = true;
            else if (strcmp(argv[i], "--markdown") == 0) markdown = true;
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [--bw] [--dispatch] [--slc] [--markdown]\n", argv[0]);
                printf("  No flags = run all experiments\n");
                return 0;
            }
        }

        // Default: run all
        if (!run_bw && !run_dispatch && !run_slc) {
            run_bw = run_dispatch = run_slc = true;
        }

        print_system_info();

        if (run_bw) run_experiment_01(markdown);
        if (run_dispatch) run_experiment_02(markdown);
        if (run_slc) run_experiment_03(markdown);

        printf("\n================================================================\n");
        printf("  Phase 0 complete. See results above for gate decisions.\n");
        printf("================================================================\n");

        return 0;
    }
}
