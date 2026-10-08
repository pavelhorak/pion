// metal_wrap.m — Apple Metal GPU compute wrapper for Pion vector search
// Provides C FFI functions callable from Mojo via external_call.
// Compiles with: clang -c -fobjc-arc src/ffi/metal_wrap.m -o src/ffi/metal_wrap.o -framework Metal -framework Foundation

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>   /* gh #281: _NSGetExecutablePath */
#include <limits.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>   // gh #191: multi-accumulator block-score selector
#endif

/* ─── Shader-library discovery (gh #281) ────────────────────────────────────
 *
 * Both loaders used to search two CWD-RELATIVE paths and nothing else:
 * "src/ffi/metal_compute.metallib" and "metal_compute.metallib". That works
 * exactly once — when the server is launched from a source checkout. It fails
 * for every shipped artifact:
 *
 *   - the release tarball contains bin/, lib/, LICENSE, README.txt and the
 *     wrapper, and no metallib at all;
 *   - even with one bundled, README.txt tells the reader state lands in the
 *     working directory, so people run the server from a data directory,
 *     where a CWD-relative lookup cannot find a file that sits next to the
 *     binary.
 *
 * So --metal-attention could not work from a release, and the failure was a
 * quiet fall back to the MLX bridge (gh #281). The search is now anchored to
 * the EXECUTABLE as well as the CWD, and lives in one place instead of being
 * written out twice — the two copies had already been kept in sync by a
 * comment, which is the state just before they drift.
 *
 * Order is deliberate: an explicit override beats everything, then paths next
 * to the binary (what a shipped artifact looks like), then the source-tree
 * layout (what a developer's CWD looks like).
 */
static NSArray<NSString *> *pion_metal_dirs(void) {
    NSMutableArray<NSString *> *dirs = [NSMutableArray array];

    char exe[PATH_MAX];
    uint32_t sz = (uint32_t)sizeof(exe);
    if (_NSGetExecutablePath(exe, &sz) == 0) {
        char resolved[PATH_MAX];
        const char *full = realpath(exe, resolved) ? resolved : exe;
        NSString *bin = [[NSString stringWithUTF8String:full] stringByDeletingLastPathComponent];
        [dirs addObject:bin];                                          /* <exe dir>/            */
        [dirs addObject:[bin stringByAppendingPathComponent:@"../lib"]];   /* tarball bin/ + lib/ */
        [dirs addObject:[bin stringByAppendingPathComponent:@"../share/pion"]];
        [dirs addObject:[bin stringByAppendingPathComponent:@"../src/ffi"]];
        /* A source build is ./pion-server at the repo root, with the library
         * in src/ffi beside it. Without this entry, a checkout binary started
         * from any other directory found nothing and fell back to the MLX
         * bridge: the CWD-relative entries below only work from the root. */
        [dirs addObject:[bin stringByAppendingPathComponent:@"src/ffi"]];
    }
    [dirs addObject:@"src/ffi"];   /* source checkout, CWD-relative — unchanged */
    [dirs addObject:@"."];
    return dirs;
}

/* Candidate paths for `name`, most specific first. PION_METAL_LIB names a file
 * outright and is checked before anything else, so a packager or an operator
 * can always be explicit. */
static NSArray<NSString *> *pion_metal_candidates(NSString *name) {
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    const char *env = getenv("PION_METAL_LIB");
    if (env && *env && [name hasSuffix:@".metallib"]) {
        [out addObject:[NSString stringWithUTF8String:env]];
    }
    for (NSString *dir in pion_metal_dirs()) {
        [out addObject:[dir stringByAppendingPathComponent:name]];
    }
    return out;
}

/* The first existing candidate, or nil. */
static NSString *pion_metal_find(NSString *name) {
    NSFileManager *fm = [NSFileManager defaultManager];
    for (NSString *p in pion_metal_candidates(name)) {
        if ([fm fileExistsAtPath:p]) return p;
    }
    return nil;
}

/* Every place we looked, for an error message that can be acted on. A "not
 * found" that does not say where it looked just moves the puzzle. */
static NSString *pion_metal_searched(NSString *name) {
    return [pion_metal_candidates(name) componentsJoinedByString:@"\n    "];
}

/* Options for compiling metal_compute.metal at run time, when no metallib was
 * found. Both compile sites take them from here.
 *
 * `mathMode` exists only from the macOS 15 SDK, which also deprecates
 * `fastMathEnabled`. Release binaries are built on macos-14 runners (SDK 14.x),
 * so naming `mathMode` unconditionally did not compile there: the first CI run
 * on that runner (2026-10-01) died in this file, and the v0.9.0 release build
 * would have too. Both spellings ask for fast math. */
static MTLCompileOptions *pion_metal_compile_options(void) {
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
        return opts;
    }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
    return opts;
}

// ─── Persistent Metal Context ──────────────────────────────────────────
// Created once at startup, shared across all searches.
// Holds device, command queue, compiled pipeline, and reusable buffers.

// FP32 rerank: max candidate IDs per dispatch. Per-worker buffers sized for
// this cap (4 × 2048 = 8 KB ids + 8 KB dists per worker).
#define PION_RERANK_K_MAX 2048

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> l2_pipeline;       // int8_l2_distance_batch
    id<MTLComputePipelineState> multiquery_pipeline; // int8_l2_batch_multiquery
    id<MTLComputePipelineState> rerank_pipeline;   // fp32_l2_gather_rerank
    NSUInteger                  max_threadgroup_size;
    NSUInteger                  rerank_tg_size;    // tuned threadgroup size for rerank kernel
    // (old single query_buf removed — per-worker buffers in query_bufs[])
    id<MTLBuffer>               candidates_buf;    // compact_buffer: [max_n × stride] (wraps mmap)
    id<MTLBuffer>               query_norm_buf;    // single Float32
    id<MTLBuffer>               dim_buf;           // single uint32
    id<MTLBuffer>               stride_buf;        // single uint32
    // Per-worker buffers: each worker gets its own query + output + distances
    // to eliminate mutex contention. 16 workers max.
    id<MTLBuffer>               query_bufs[16];    // [dim] INT8 per worker
    id<MTLBuffer>               distances_bufs[16]; // [max_n] Float32 per worker
    // Per-worker FP32 rerank buffers (separate from INT8 search buffers — sizes differ
    // and a worker may issue a full-scan + rerank concurrently in §7v2).
    id<MTLBuffer>               rerank_query_bufs[16];   // [dim] Float32 per worker
    id<MTLBuffer>               rerank_ids_bufs[16];     // [PION_RERANK_K_MAX] Int32 per worker
    id<MTLBuffer>               rerank_dists_bufs[16];   // [PION_RERANK_K_MAX] Float32 per worker
    id<MTLBuffer>               rerank_buf;        // wraps gpu_rerank_fp32: [rerank_n × dim]
    void                       *rerank_buf_ptr;    // raw pointer registered (for idempotent re-register)
    uint32_t                    rerank_n;          // num_vectors in rerank_buf (0 if unregistered)
    uint32_t                    dim;
    uint32_t                    stride;            // 8 + dim
    uint32_t                    max_n;
    // MTLSharedEvent for spin-wait completion (replaces waitUntilCompleted)
    id<MTLSharedEvent>          shared_event;
    uint64_t                    event_counter;     // monotonically increasing per dispatch
    int                         ready;
    // §7v2: Per-worker dispatch semaphores for async GPU pipelining
    dispatch_semaphore_t        worker_sema[16];   // signaled on GPU completion
    volatile int                worker_gpu_pending[16]; // 1 = GPU work in flight
    // FP32 rerank: separate per-worker semaphores so a worker may have both
    // a full-scan and a rerank dispatch in flight simultaneously.
    dispatch_semaphore_t        rerank_sema[16];
} PionMetalContext;

// Global context pointer (one per process)
static PionMetalContext* g_ctx = NULL;

// Forward declarations
int32_t pion_metal_search_topk_w(int8_t *query_int8, float query_norm_sq,
    uint32_t num_vectors, uint32_t k, int32_t *out_ids, float *out_dists, uint32_t worker_id);
int32_t pion_metal_register_rerank_buffer(float *fp32_buf, uint32_t num_vectors, uint32_t dim);
int32_t pion_metal_unregister_rerank_buffer(void);
int32_t pion_metal_rerank(float *query_fp32, int32_t *ids, uint32_t k,
    float *out_distances, uint32_t worker_id);

// ─── Adaptive GPU/CPU Load Tracking ────────────────────────────────────
// Per-worker rolling average of GPU dispatch latency (nanoseconds).
// When latency exceeds threshold (GPU busy with LLM/MAX/other), pion_metal_should_use_gpu()
// returns 0 → caller falls back to CPU HNSW.
#include <mach/mach_time.h>

#define GPU_LOAD_SLOTS 16
#define GPU_LATENCY_THRESHOLD_NS 3000000  // 3ms — GPU bypassed when LLM/MAX causes >3ms dispatch latency
#define GPU_COOLDOWN_TICKS 50             // after going CPU, stay CPU for N queries before retrying

static struct {
    uint64_t latencies[GPU_LOAD_SLOTS];   // rolling buffer of recent GPU dispatch latencies (ns)
    uint32_t idx;                          // current write position
    uint32_t count;                        // total dispatches (saturates at UINT32_MAX)
    uint64_t avg_latency_ns;               // cached rolling average
    uint32_t cooldown;                     // ticks remaining in CPU-only mode
    mach_timebase_info_data_t timebase;    // for mach_absolute_time → nanoseconds
} g_gpu_load = { .idx = 0, .count = 0, .avg_latency_ns = 0, .cooldown = 0 };

static void _gpu_load_init(void) {
    mach_timebase_info(&g_gpu_load.timebase);
    for (int i = 0; i < GPU_LOAD_SLOTS; i++) g_gpu_load.latencies[i] = 0;
}

static uint64_t _mach_to_ns(uint64_t mach_ticks) {
    return mach_ticks * g_gpu_load.timebase.numer / g_gpu_load.timebase.denom;
}

static void _gpu_load_record(uint64_t latency_ns) {
    g_gpu_load.latencies[g_gpu_load.idx % GPU_LOAD_SLOTS] = latency_ns;
    g_gpu_load.idx++;
    if (g_gpu_load.count < UINT32_MAX) g_gpu_load.count++;

    // Compute rolling average
    uint64_t sum = 0;
    uint32_t n = (g_gpu_load.count < GPU_LOAD_SLOTS) ? g_gpu_load.count : GPU_LOAD_SLOTS;
    for (uint32_t i = 0; i < n; i++) sum += g_gpu_load.latencies[i];
    g_gpu_load.avg_latency_ns = sum / n;
}

// Returns 1 if GPU should be used, 0 if CPU fallback is recommended.
// Called by Mojo via FFI before each FT.SEARCH dispatch.
int32_t pion_metal_should_use_gpu(void) {
    if (!g_ctx || !g_ctx->ready) return 0;

    // Cooldown: after detecting high GPU load, stay on CPU for N queries
    if (g_gpu_load.cooldown > 0) {
        g_gpu_load.cooldown--;
        return 0;
    }

    // Not enough samples yet — use GPU to collect baseline
    if (g_gpu_load.count < 4) return 1;

    // If average latency exceeds threshold, GPU is busy (LLM/MAX/other compute)
    if (g_gpu_load.avg_latency_ns > GPU_LATENCY_THRESHOLD_NS) {
        g_gpu_load.cooldown = GPU_COOLDOWN_TICKS;
        return 0;
    }

    return 1;
}

// Query the current GPU load stats (for INFO/diagnostics)
const char* pion_metal_device_name(void) {
    if (!g_ctx || !g_ctx->device) return "None";
    return [[g_ctx->device name] UTF8String];
}

uint32_t pion_metal_max_threadgroup_size(void) {
    if (!g_ctx) return 0;
    return (uint32_t)g_ctx->max_threadgroup_size;
}

uint64_t pion_metal_gpu_latency_ns(void) {
    return g_gpu_load.avg_latency_ns;
}

uint32_t pion_metal_gpu_dispatch_count(void) {
    return g_gpu_load.count;
}

// ─── Initialization ────────────────────────────────────────────────────

int32_t pion_metal_init(uint32_t dim, uint32_t max_n) {
    if (g_ctx != NULL) return 1; // already initialized

    @autoreleasepool {
        g_ctx = (PionMetalContext*)calloc(1, sizeof(PionMetalContext));
        if (!g_ctx) return -1;

        g_ctx->dim = dim;
        g_ctx->stride = 8 + dim;  // compact_buffer layout: [4B norm][4B prefix_norm][dim B vector]
        g_ctx->max_n = max_n;

        // Get default Metal device (Apple Silicon GPU)
        g_ctx->device = MTLCreateSystemDefaultDevice();
        if (!g_ctx->device) {
            fprintf(stderr, "[Metal] No Metal device found\n");
            free(g_ctx); g_ctx = NULL;
            return -2;
        }

        fprintf(stderr, "[Metal] Device: %s\n", [[g_ctx->device name] UTF8String]);

        // Create command queue
        g_ctx->queue = [g_ctx->device newCommandQueue];
        if (!g_ctx->queue) {
            fprintf(stderr, "[Metal] Failed to create command queue\n");
            free(g_ctx); g_ctx = NULL;
            return -3;
        }

        // Load Metal shader library — try pre-compiled metallib first, then runtime compile
        NSError *error = nil;
        id<MTLLibrary> library = nil;

        NSString *libPath = pion_metal_find(@"metal_compute.metallib");
        if (libPath) {
            library = [g_ctx->device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
            if (library) {
                fprintf(stderr, "[Metal] Loaded pre-compiled metallib: %s\n", [libPath UTF8String]);
            }
        }

        if (!library) {
            // Runtime compile from .metal source
            NSString *srcPath = pion_metal_find(@"metal_compute.metal");
            if (srcPath) {
                NSString *source = [NSString stringWithContentsOfFile:srcPath
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];
                if (source) {
                    MTLCompileOptions *opts = pion_metal_compile_options();
                    library = [g_ctx->device newLibraryWithSource:source options:opts error:&error];
                    if (library) {
                        fprintf(stderr, "[Metal] Compiled shader from source at runtime\n");
                    } else {
                        fprintf(stderr, "[Metal] Runtime compilation failed: %s\n",
                                error ? [[error localizedDescription] UTF8String] : "unknown");
                    }
                }
            }
        }

        if (!library) {
            NSString *tried = pion_metal_searched(@"metal_compute.metallib");
            fprintf(stderr,
                "[Metal] No metallib or metal source found. Looked in:\n    %s\n"
                "        Set PION_METAL_LIB=/path/to/metal_compute.metallib to be explicit.\n",
                [tried UTF8String]);
            free(g_ctx); g_ctx = NULL;
            return -4;
        }

        // Create compute pipelines
        id<MTLFunction> l2Func = [library newFunctionWithName:@"int8_l2_distance_batch"];
        if (!l2Func) {
            fprintf(stderr, "[Metal] Function 'int8_l2_distance_batch' not found\n");
            free(g_ctx); g_ctx = NULL;
            return -5;
        }
        g_ctx->l2_pipeline = [g_ctx->device newComputePipelineStateWithFunction:l2Func error:&error];
        if (!g_ctx->l2_pipeline) {
            fprintf(stderr, "[Metal] Pipeline creation failed: %s\n", [[error localizedDescription] UTF8String]);
            free(g_ctx); g_ctx = NULL;
            return -6;
        }

        id<MTLFunction> mqFunc = [library newFunctionWithName:@"int8_l2_batch_multiquery"];
        if (mqFunc) {
            g_ctx->multiquery_pipeline = [g_ctx->device newComputePipelineStateWithFunction:mqFunc error:&error];
        }

        // FP32 rerank pipeline (optional — older metallibs without the symbol stay
        // CPU-only; pion_metal_rerank() returns -1 in that case).
        id<MTLFunction> rerankFunc = [library newFunctionWithName:@"fp32_l2_gather_rerank"];
        if (rerankFunc) {
            g_ctx->rerank_pipeline = [g_ctx->device newComputePipelineStateWithFunction:rerankFunc error:&error];
            if (!g_ctx->rerank_pipeline) {
                fprintf(stderr, "[Metal] Rerank pipeline creation failed: %s\n",
                        error ? [[error localizedDescription] UTF8String] : "unknown");
            } else {
                // 256 threads/TG with float4 loads on 1536 dims → each thread handles
                // dim/(4·256) = 1.5 float4s. Reduction tree halves at each step.
                NSUInteger r_max = g_ctx->rerank_pipeline.maxTotalThreadsPerThreadgroup;
                g_ctx->rerank_tg_size = (r_max < 256) ? r_max : 256;
            }
        } else {
            fprintf(stderr, "[Metal] fp32_l2_gather_rerank not in library — CPU rerank only\n");
        }

        g_ctx->max_threadgroup_size = g_ctx->l2_pipeline.maxTotalThreadsPerThreadgroup;
        fprintf(stderr, "[Metal] Max threadgroup size: %lu\n", (unsigned long)g_ctx->max_threadgroup_size);

        // MTLSharedEvent for spin-wait completion (replaces waitUntilCompleted)
        // Spin-wait on signaledValue bypasses OS scheduler → saves ~0.8ms per dispatch
        g_ctx->shared_event = [g_ctx->device newSharedEvent];
        g_ctx->event_counter = 0;

        // §7v2: Per-worker dispatch semaphores for async GPU pipelining
        for (int w = 0; w < 16; w++) {
            g_ctx->worker_sema[w] = dispatch_semaphore_create(0);
            g_ctx->worker_gpu_pending[w] = 0;
        }

        // Pre-allocate per-worker GPU buffers (16 workers max)
        // Each worker gets its own query + distances buffer → zero contention
        for (int w = 0; w < 16; w++) {
            g_ctx->query_bufs[w] = [g_ctx->device newBufferWithLength:dim
                                                              options:MTLResourceStorageModeShared];
            g_ctx->distances_bufs[w] = [g_ctx->device newBufferWithLength:max_n * sizeof(float)
                                                                  options:MTLResourceStorageModeShared];
            // FP32 rerank: separate per-worker buffers + semaphore. Sized for K_MAX
            // candidates so beam-search results never overflow.
            g_ctx->rerank_query_bufs[w] = [g_ctx->device newBufferWithLength:dim * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
            g_ctx->rerank_ids_bufs[w] = [g_ctx->device newBufferWithLength:PION_RERANK_K_MAX * sizeof(int32_t)
                                                                   options:MTLResourceStorageModeShared];
            g_ctx->rerank_dists_bufs[w] = [g_ctx->device newBufferWithLength:PION_RERANK_K_MAX * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];
            g_ctx->rerank_sema[w] = dispatch_semaphore_create(0);
        }
        g_ctx->rerank_buf = nil;
        g_ctx->rerank_buf_ptr = NULL;
        g_ctx->rerank_n = 0;

        // Query norm: single float
        g_ctx->query_norm_buf = [g_ctx->device newBufferWithLength:sizeof(float)
                                                           options:MTLResourceStorageModeShared];

        // Dim constant
        g_ctx->dim_buf = [g_ctx->device newBufferWithLength:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        *(uint32_t*)[g_ctx->dim_buf contents] = dim;

        // Stride constant
        g_ctx->stride_buf = [g_ctx->device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        *(uint32_t*)[g_ctx->stride_buf contents] = 8 + dim;

        g_ctx->ready = 1;
        _gpu_load_init();
        fprintf(stderr, "[Metal] GPU vector engine initialized (dim=%u, max_n=%u, adaptive=on)\n", dim, max_n);
        return 0;
    }
}

// ─── Register compact_buffer (zero-copy) ───────────────────────────────
// Called after FT.OPTIMIZE to wrap the existing mmap'd compact_buffer.
// The buffer has 8 bytes of norm header per vector, then dim bytes of INT8.
// We need to create a contiguous view of just the INT8 vectors for the GPU.

int32_t pion_metal_register_vectors(
    int8_t *compact_buffer,     // raw pointer to compact_buffer (owned by HNSW)
    uint32_t num_vectors,
    uint32_t stride)            // gh #196.2: actual slot stride (may be 64B-padded past 8+dim)
{
    if (!g_ctx || !g_ctx->ready) return -1;

    @autoreleasepool {
        if (stride != 0) {
            g_ctx->stride = stride;
            if (g_ctx->stride_buf)
                *(uint32_t*)[g_ctx->stride_buf contents] = stride;
        }
        uint32_t total_bytes = num_vectors * g_ctx->stride;

        // Zero-copy wrap of compact_buffer (unified memory on Apple Silicon)
        // compact_buffer must be page-aligned for true zero-copy.
        g_ctx->candidates_buf = [g_ctx->device newBufferWithBytesNoCopy:compact_buffer
                                                                 length:total_bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!g_ctx->candidates_buf) {
            // Fallback: Metal copies internally (still fast)
            g_ctx->candidates_buf = [g_ctx->device newBufferWithBytes:compact_buffer
                                                               length:total_bytes
                                                              options:MTLResourceStorageModeShared];
            fprintf(stderr, "[Metal] Warning: compact_buffer not page-aligned, using copy\n");
        }

        g_ctx->max_n = num_vectors;
        fprintf(stderr, "[Metal] Registered %u vectors (%.1f MB compact_buffer, stride=%u)\n",
                num_vectors, (float)total_bytes / (1024.0f * 1024.0f), g_ctx->stride);
        return 0;
    }
}

// ─── GPU Brute-Force Search ────────────────────────────────────────────
// Computes L2 distance from one INT8 query to all N registered vectors.
// Returns distances in the shared output buffer.
// Caller reads distances and does CPU top-K (fast for K=150).

// Fused search + top-K: GPU computes distances, CPU immediately scans output.
// Eliminates the separate memcpy of 200KB distance buffer.
// Uses setBytes for small constants (no buffer tracking overhead).
// Output distances written directly to caller's buffer (zero-copy via unified memory).

int32_t pion_metal_search_topk(
    int8_t   *query_int8,       // [dim] INT8 quantized query
    float     query_norm_sq,    // precomputed query norm²
    uint32_t  num_vectors,      // how many vectors to search
    uint32_t  k,                // top-K to select
    int32_t  *out_ids,          // [K] output: slot indices
    float    *out_dists)        // [K] output: distances
{
    return pion_metal_search_topk_w(query_int8, query_norm_sq, num_vectors, k, out_ids, out_dists, 0);
}

// §7v2: Per-worker fused search with async GPU pipelining.
// Encode + commit is non-blocking. Semaphore-wait replaces spin-wait,
// allowing the Metal command queue to pipeline dispatches from multiple workers.
// Each worker has dedicated buffers → zero contention on buffer access.
int32_t pion_metal_search_topk_w(
    int8_t   *query_int8,
    float     query_norm_sq,
    uint32_t  num_vectors,
    uint32_t  k,
    int32_t  *out_ids,
    float    *out_dists,
    uint32_t  worker_id)
{
    if (!g_ctx || !g_ctx->ready || !g_ctx->candidates_buf) return -1;
    if (num_vectors > g_ctx->max_n) num_vectors = g_ctx->max_n;
    if (worker_id >= 16) worker_id = 0;

    @autoreleasepool {
        uint64_t t_start = mach_absolute_time();

        // Per-worker query buffer (no contention)
        id<MTLBuffer> wquery = g_ctx->query_bufs[worker_id];
        id<MTLBuffer> wdists = g_ctx->distances_bufs[worker_id];
        memcpy([wquery contents], query_int8, g_ctx->dim);

        id<MTLCommandBuffer> cmdBuf = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:g_ctx->l2_pipeline];
        [encoder setBuffer:wquery                  offset:0 atIndex:0];
        [encoder setBuffer:g_ctx->candidates_buf   offset:0 atIndex:1];
        [encoder setBuffer:wdists                  offset:0 atIndex:2];
        [encoder setBytes:&(g_ctx->dim)    length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&query_norm_sq   length:sizeof(float)    atIndex:4];
        [encoder setBytes:&(g_ctx->stride) length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_vectors     length:sizeof(uint32_t) atIndex:6];

        // §7v2: dispatchThreadgroups avoids non-uniform threadgroup overhead
        NSUInteger tg_size = MIN(g_ctx->max_threadgroup_size, 256);
        NSUInteger num_groups = (num_vectors + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [encoder endEncoding];

        // §7v2: Async completion — signal per-worker semaphore instead of spin-wait.
        // Metal pipelines command buffers from multiple workers concurrently.
        dispatch_semaphore_t sema = g_ctx->worker_sema[worker_id];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buf) {
            dispatch_semaphore_signal(sema);
        }];
        [cmdBuf commit];

        // Wait for GPU completion (semaphore — yields to OS scheduler, allows GPU pipelining)
        dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

        // Record GPU dispatch latency for adaptive routing
        uint64_t t_gpu_done = mach_absolute_time();
        _gpu_load_record(_mach_to_ns(t_gpu_done - t_start));

        // Fused top-K scan directly from per-worker GPU output (unified memory)
        float *distances = (float *)[wdists contents];
        if (k > num_vectors) k = num_vectors;

        for (uint32_t i = 0; i < k; i++) {
            out_dists[i] = INFINITY;
            out_ids[i] = -1;
        }

        float worst = INFINITY;
        uint32_t worst_idx = 0;

        for (uint32_t i = 0; i < num_vectors; i++) {
            float d = distances[i];
            if (d < worst) {
                out_dists[worst_idx] = d;
                out_ids[worst_idx] = (int32_t)i;
                worst = -INFINITY;
                for (uint32_t j = 0; j < k; j++) {
                    if (out_dists[j] > worst) {
                        worst = out_dists[j];
                        worst_idx = j;
                    }
                }
            }
        }

        return (int32_t)k;
    }
}

// Legacy API — keep for compatibility (uses worker 0 buffers)
int32_t pion_metal_search(
    int8_t   *query_int8,
    float     query_norm_sq,
    uint32_t  num_vectors,
    float    *out_distances)
{
    if (!g_ctx || !g_ctx->ready || !g_ctx->candidates_buf) return -1;
    if (num_vectors > g_ctx->max_n) num_vectors = g_ctx->max_n;

    @autoreleasepool {
        memcpy([g_ctx->query_bufs[0] contents], query_int8, g_ctx->dim);

        id<MTLCommandBuffer> cmdBuf = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:g_ctx->l2_pipeline];
        [encoder setBuffer:g_ctx->query_bufs[0]   offset:0 atIndex:0];
        [encoder setBuffer:g_ctx->candidates_buf   offset:0 atIndex:1];
        [encoder setBuffer:g_ctx->distances_bufs[0] offset:0 atIndex:2];
        [encoder setBytes:&(g_ctx->dim)    length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&query_norm_sq   length:sizeof(float)    atIndex:4];
        [encoder setBytes:&(g_ctx->stride) length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&num_vectors     length:sizeof(uint32_t) atIndex:6];

        // §7v2: dispatchThreadgroups avoids non-uniform threadgroup overhead
        NSUInteger tg_size = MIN(g_ctx->max_threadgroup_size, 256);
        NSUInteger num_groups = (num_vectors + tg_size - 1) / tg_size;
        [encoder dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(out_distances, [g_ctx->distances_bufs[0] contents], num_vectors * sizeof(float));
        return 0;
    }
}

// ─── GPU Top-K extraction (CPU-side, after GPU distance computation) ───
// Simple partial selection: find K smallest distances from N.
// For K=150, N=50K: ~50K comparisons = ~10µs on M4 CPU. Not worth GPU dispatch.

int32_t pion_metal_topk(
    float    *distances,        // [N] input distances
    uint32_t  N,
    uint32_t  K,
    int32_t  *out_ids,          // [K] output indices
    float    *out_dists)        // [K] output distances
{
    if (K > N) K = N;

    // Simple partial sort: maintain a max-heap of size K
    // For K=150, N=50K, this is O(N log K) ≈ 50K × 7 ≈ 350K ops — ~10µs
    // Using a simple array + linear scan (K=150 is small enough)

    for (uint32_t i = 0; i < K; i++) {
        out_dists[i] = INFINITY;
        out_ids[i] = -1;
    }

    // Track the worst (maximum) distance in our top-K
    float worst = INFINITY;
    uint32_t worst_idx = 0;

    for (uint32_t i = 0; i < N; i++) {
        float d = distances[i];
        if (d < worst) {
            // Replace the worst entry
            out_dists[worst_idx] = d;
            out_ids[worst_idx] = (int32_t)i;

            // Find new worst
            worst = -INFINITY;
            for (uint32_t j = 0; j < K; j++) {
                if (out_dists[j] > worst) {
                    worst = out_dists[j];
                    worst_idx = j;
                }
            }
        }
    }

    return (int32_t)K;
}

// ─── FP32 Rerank (gather kernel) ───────────────────────────────────────
// Replaces the CPU SIMD rerank in HNSW quant variants when K ≥ threshold.
// Caller passes the BFS-ordered FP32 buffer (gpu_rerank_fp32) once via
// pion_metal_register_rerank_buffer; subsequent rerank calls reuse the
// zero-copy wrap until pion_metal_unregister_rerank_buffer.

int32_t pion_metal_register_rerank_buffer(float *fp32_buf, uint32_t num_vectors, uint32_t dim) {
    if (!g_ctx || !g_ctx->ready || !g_ctx->rerank_pipeline) return -1;
    if (!fp32_buf || num_vectors == 0 || dim == 0 || dim != g_ctx->dim) return -2;

    @autoreleasepool {
        // Idempotent: if same pointer is already registered, no-op (rebuild path
        // may call this redundantly).
        if (g_ctx->rerank_buf && g_ctx->rerank_buf_ptr == (void *)fp32_buf
            && g_ctx->rerank_n == num_vectors) {
            return 0;
        }

        // Drop any prior wrap before installing the new one.
        g_ctx->rerank_buf = nil;
        g_ctx->rerank_buf_ptr = NULL;
        g_ctx->rerank_n = 0;

        size_t total_bytes = (size_t)num_vectors * dim * sizeof(float);
        id<MTLBuffer> buf = [g_ctx->device newBufferWithBytesNoCopy:fp32_buf
                                                            length:total_bytes
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
        if (!buf) {
            // Fallback: copy. Caller's pointer is unchanged.
            buf = [g_ctx->device newBufferWithBytes:fp32_buf
                                             length:total_bytes
                                            options:MTLResourceStorageModeShared];
            fprintf(stderr, "[Metal] FP32 rerank buffer not page-aligned, using copy\n");
        }
        if (!buf) return -3;
        g_ctx->rerank_buf = buf;
        g_ctx->rerank_buf_ptr = (void *)fp32_buf;
        g_ctx->rerank_n = num_vectors;
        fprintf(stderr, "[Metal] FP32 rerank registered: %u vectors × %u dim (%.1f MB)\n",
                num_vectors, dim, (float)total_bytes / (1024.0f * 1024.0f));
        return 0;
    }
}

int32_t pion_metal_unregister_rerank_buffer(void) {
    if (!g_ctx) return 0;
    g_ctx->rerank_buf = nil;
    g_ctx->rerank_buf_ptr = NULL;
    g_ctx->rerank_n = 0;
    return 0;
}

// Counter for rerank dispatches (diagnostic — exposed via pion_metal_rerank_count).
static _Atomic uint64_t g_rerank_dispatch_count = 0;

uint64_t pion_metal_rerank_count(void) {
    return atomic_load(&g_rerank_dispatch_count);
}

int32_t pion_metal_rerank(
    float    *query_fp32,       // [dim] FP32 query
    int32_t  *ids,              // [k] candidate slot indices into rerank_buf
    uint32_t  k,
    float    *out_distances,    // [k] output L2² distances
    uint32_t  worker_id)
{
    if (!g_ctx || !g_ctx->ready || !g_ctx->rerank_pipeline || !g_ctx->rerank_buf) return -1;
    if (k == 0) return 0;
    if (k > PION_RERANK_K_MAX) k = PION_RERANK_K_MAX;
    if (worker_id >= 16) worker_id = 0;
    atomic_fetch_add(&g_rerank_dispatch_count, 1);

    @autoreleasepool {
        id<MTLBuffer> wquery = g_ctx->rerank_query_bufs[worker_id];
        id<MTLBuffer> wids   = g_ctx->rerank_ids_bufs[worker_id];
        id<MTLBuffer> wdists = g_ctx->rerank_dists_bufs[worker_id];

        memcpy([wquery contents], query_fp32, g_ctx->dim * sizeof(float));
        memcpy([wids contents],   ids,        k * sizeof(int32_t));

        id<MTLCommandBuffer> cmdBuf = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:g_ctx->rerank_pipeline];
        [encoder setBuffer:wquery               offset:0 atIndex:0];
        [encoder setBuffer:g_ctx->rerank_buf    offset:0 atIndex:1];
        [encoder setBuffer:wids                 offset:0 atIndex:2];
        [encoder setBuffer:wdists               offset:0 atIndex:3];
        [encoder setBytes:&(g_ctx->dim) length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&k            length:sizeof(uint32_t) atIndex:5];

        // One threadgroup per candidate ID; threads cooperatively reduce dim.
        NSUInteger tg_size = g_ctx->rerank_tg_size > 0 ? g_ctx->rerank_tg_size : 256;
        [encoder dispatchThreadgroups:MTLSizeMake(k, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [encoder endEncoding];

        dispatch_semaphore_t sema = g_ctx->rerank_sema[worker_id];
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buf) {
            dispatch_semaphore_signal(sema);
        }];
        [cmdBuf commit];
        dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);

        memcpy(out_distances, [wdists contents], k * sizeof(float));
        return (int32_t)k;
    }
}

// ─── Check GPU availability ────────────────────────────────────────────

int32_t pion_metal_available(void) {
    return (g_ctx && g_ctx->ready) ? 1 : 0;
}

// ─── Cleanup ───────────────────────────────────────────────────────────

void pion_metal_shutdown(void) {
    if (g_ctx) {
        g_ctx->ready = 0;
        // ARC handles Obj-C object cleanup
        free(g_ctx);
        g_ctx = NULL;
        fprintf(stderr, "[Metal] GPU engine shut down\n");
    }
}

// ─── SDPA Q=1 (Decoder-step Multi-Head Attention) ──────────────────────
// Independent from the INT8 vector search context above. Lazily inits its
// own Metal device + queue + pipeline + per-session K/V buffers on first
// pion_metal_sdpa_init() call.
//
// Beats `mlx.fast.scaled_dot_product_attention` by 1.34-1.55× at H=8/N=2048/d=128
// (proof: tests/bench_msl_sdpa_q1.m).

// 256 slots per worker × 16 workers = 4096 cached sessions max. Linear probing
// with tombstones handles collisions safely (different sessions on the same
// worker hashing to the same bucket → probe forward, no eviction race).
#define SDPA_SLOTS 256
#define SDPA_KEY_EMPTY      0ULL
#define SDPA_KEY_TOMBSTONE  0xFFFFFFFFFFFFFFFFULL

// gh #67 cold tier: when the slot table is full, the LRU live slot's Metal
// K_buf/V_buf are released and its (key, dims, last_access) move to the cold
// registry — V-store still has the K/V on disk, so a later KV.PREFIX.WARM
// can rehydrate. The cold registry holds metadata only (no Metal memory).
// 4× the slot table so a worker can carry up to ~5K observed sessions before
// dropping the oldest cold record. Linear scan; at 1024 entries that's a
// sub-µs sweep on M-series unified memory and the cold path is not hot.
#define SDPA_COLD_SLOTS 1024

// gh #67: per-slot lifecycle state. EMPTY/TOMBSTONE remain encoded via the
// `key` field so existing find/alloc logic keeps working; WARM vs COLD is
// distinguished by the new `state` field below (only meaningful when
// `key != EMPTY && key != TOMBSTONE`). Slots in the WARM table only ever
// hold state=WARM (or state=0 in a transient half-allocated window between
// _sdpa_alloc_slot and the store_kv memcpy). COLD entries live in the
// per-worker cold[] registry, not in the slot table.
#define SDPA_SLOT_WARM  1u
#define SDPA_SLOT_COLD  2u

// gh #67: return codes from `_sdpa_resolve_slot`. Negative values mirror the
// existing -1/-2/... semantics used by the public sdpa_query* entries; -10
// is new and means "the session was stored once but has been evicted to the
// cold registry — caller can return -COLDMISS to the wire".
#define SDPA_RC_OK         0
#define SDPA_RC_HARD_MISS  (-1)
#define SDPA_RC_COLD_MISS  (-10)

// Supported d_head values (Phase B: D-templatization). Each value gets its
// own PSO compiled with the D_HEAD function constant. Runtime D outside the
// table → query returns -2 and caller falls back to MLX bridge.
// D=512 (gh #60 Phase 1, 2026-05-11): added for Gemma 4 full-attention layers
// (global_head_dim=512). Requires CHUNK_MAX=4 in metal_compute.metal and
// dynamic threadgroup memory for s_o (s_o size scales with CHUNK).
#define SDPA_PSOS_COUNT 8
static const uint32_t SDPA_D_VALUES[SDPA_PSOS_COUNT] = { 32, 64, 96, 128, 160, 192, 256, 512 };
#define SDPA_D_MAX 512u

#define SDPA_MAX_WORKERS 16   // mirrors PionMetalContext.query_bufs[16]; bounds per-worker arrays.

typedef struct {
    uint64_t      key;          // hash(session_id) ^ (layer_id << 56); 0 = empty
    uint32_t      H, N, D;
    uint8_t       state;        // gh #67: SDPA_SLOT_WARM | SDPA_SLOT_COLD (only valid when key not EMPTY/TOMBSTONE)
    uint64_t      last_access_ns; // gh #67: mach_absolute_time() at most recent STORE/QUERY — drives LRU pick.
    id<MTLBuffer> K_buf;        // H*N*D float32, or half when kv_half; MTLResourceStorageModeShared
    id<MTLBuffer> V_buf;
    uint8_t       kv_half;      // gh #398: 1 = K_buf/V_buf hold half (--metal-attention-fp16)
    // gh #63 Phase 3b: server-side selector cache. K_mean is computed at
    // STORE time over the just-stored K tensor, in fixed-B blocks (B=64 v1).
    // Stored in host memory (not a Metal buffer) because the selector scoring
    // path is CPU — H * n_blocks * D = ~2M ops/query at H=4 N=64K D=512,
    // sub-millisecond on the host without a kernel dispatch tax. The kernel
    // dispatch tax (event signal + spin) is ~300 µs which is more than the
    // CPU compute; CPU wins below ~1B ops.
    //
    // Layout: [H, n_blocks, D] row-major float32. n_blocks = (N + B - 1) / B.
    // The last block may be partial; we record its actual fill in last_blk_n
    // for the mean denominator. Lazily-zero K_mean (computed lazily on first
    // QUERY_SPARSE_AUTO if not pre-computed at STORE).
    float*        K_mean;       // [H * n_blocks * D] host-allocated; freed on slot reset
    uint32_t      K_mean_B;     // block size used (0 = K_mean not computed yet)
    uint32_t      K_mean_n_blocks;
    // W11 / gh #9 Phase 2: server-side Quest precompute. Same shape as
    // K_mean (H × n_blocks × D each) — per-(head, block, dim) min and max
    // of K. Lazily computed by _sdpa_ensure_kminmax on first sparse_auto
    // call with selector_id=1; cached identically to K_mean.
    //   UB(block) = Σ_d max(Q[d]·K_min[h,b,d], Q[d]·K_max[h,b,d])
    // This is provably ≥ max_t (Q·K_t) over the block — the upper bound
    // the softmax weight is dominated by. v1 (W11): host CPU compute,
    // same dispatch tax math as K_mean.
    float*        K_min;        // [H * n_blocks * D]
    float*        K_max;        // [H * n_blocks * D]
    uint32_t      K_minmax_B;   // block size; 0 = not computed
    uint32_t      K_minmax_n_blocks;
} PionSDPASlot;

// gh #67: cold registry entry. One row per session that's been evicted from
// the WARM slot table but still has K/V backed by V-store on disk. Layout
// matches `PionSDPASlot` for the metadata fields so a rehydrate (cold→warm)
// can restore H/N/D without the consumer re-specifying them.
typedef struct {
    uint64_t key;
    uint64_t last_access_ns;
    uint32_t H, N, D;
} PionSDPAColdEntry;

// Phase 2 (multi-worker): per-worker session cache + staging buffers.
// Each worker has its own SDPA_SLOTS-entry hash bucket array (shared-nothing
// — STORE on worker A goes to A's cache; QUERY on A reads A's cache).
// Staging buffers are also per-worker so concurrent dispatches from different
// workers don't race on the Q/O memcpys. The Metal command queue, PSOs,
// and MTLSharedEvent stay shared (Apple's queue is thread-safe; the event
// counter is atomic).
typedef struct {
    PionSDPASlot   slots[SDPA_SLOTS];
    // gh #67: cold tier. Backing-store-resident sessions whose Metal K_buf/V_buf
    // were freed under LRU pressure. Linear-scan lookup; capacity-bounded.
    PionSDPAColdEntry cold[SDPA_COLD_SLOTS];
    uint32_t       cold_count;
    uint64_t       cold_demotions;      // monotonic counter (telemetry)
    uint64_t       cold_rehydrates;     // monotonic counter
    uint64_t       cold_drops;          // monotonic counter (registry full → drop oldest)
    id<MTLBuffer>  Q_buf;       // H_max * SDPA_D_MAX * 4 (M=1)
    id<MTLBuffer>  O_buf;
    id<MTLBuffer>  N_buf;
    id<MTLBuffer>  scale_buf;
    id<MTLBuffer>  M_buf;
    id<MTLBuffer>  W_buf;       // sliding-window size; 0 = full attention
    uint32_t       Q_capacity;
    // Batched-Q staging — lazily allocated, grown on overflow.
    id<MTLBuffer>  Qb_buf;
    id<MTLBuffer>  Ob_buf;
    id<MTLBuffer>  LSE_buf;
    uint64_t       Qb_capacity;
    uint64_t       Ob_capacity;
    uint64_t       LSE_capacity;
    // gh #49 fused-suffix staging: K_suf, V_suf, head_map, S_suf scalar.
    id<MTLBuffer>  Ks_buf;
    id<MTLBuffer>  Vs_buf;
    id<MTLBuffer>  HM_buf;
    id<MTLBuffer>  Ssuf_buf;
    uint64_t       Ks_capacity;
    uint64_t       Vs_capacity;
    uint64_t       HM_capacity;
    // gh #60 Phase 2 sparse-mask staging: indices[H * K_sparse_max] int32,
    // counts[H] uint32, K_sparse_max scalar. Lazy-allocated, grow on demand.
    id<MTLBuffer>  Idx_buf;     // indices
    id<MTLBuffer>  Cnt_buf;     // counts per head
    id<MTLBuffer>  Ksp_buf;     // K_sparse_max scalar
    uint64_t       Idx_capacity;
    uint64_t       Cnt_capacity;
} PionSDPAWorker;

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;        // shared, thread-safe per Apple docs
    id<MTLComputePipelineState> psos[SDPA_PSOS_COUNT];               // sdpa_q1_fp32  (M=1 fast path)
    id<MTLComputePipelineState> psos_batched[SDPA_PSOS_COUNT];       // sdpa_batched_q_fp32 (M>1, small-M split)
    id<MTLComputePipelineState> psos_batched_tiled[SDPA_PSOS_COUNT]; // sdpa_batched_q_tiled_fp32 (gh #130, large-M K/V tiling)
    id<MTLComputePipelineState> psos_fp16[SDPA_PSOS_COUNT];          // sdpa_q1_fp16   (vanilla mlx-lm parity)
    id<MTLComputePipelineState> psos_batched_fp16[SDPA_PSOS_COUNT];  // sdpa_batched_q_fp16
    id<MTLComputePipelineState> psos_fused[SDPA_PSOS_COUNT];         // sdpa_batched_q_fused_fp32 (gh #49)
    id<MTLComputePipelineState> psos_fused_tiled[SDPA_PSOS_COUNT];   // sdpa_batched_q_tiled_fused_fp32 (gh #130, large-M K/V tiling)
    id<MTLComputePipelineState> psos_fused_fp16[SDPA_PSOS_COUNT];    // sdpa_batched_q_fused_fp16 (gh #49)
    id<MTLComputePipelineState> psos_sparse[SDPA_PSOS_COUNT];        // sdpa_q1_sparse_fp32 (gh #60 Phase 2)
    id<MTLComputePipelineState> psos_sparse_fp16[SDPA_PSOS_COUNT];   // sdpa_q1_sparse_fp16 (gh #60 Phase 2)
    id<MTLComputePipelineState> psos_sparse_fused[SDPA_PSOS_COUNT];  // sdpa_q1_sparse_fused_fp32 (gh #63 follow-on)
    id<MTLComputePipelineState> psos_sparse_fused_fp16[SDPA_PSOS_COUNT]; // sdpa_q1_sparse_fused_fp16 (gh #63 follow-on)
    id<MTLSharedEvent>          event;        // shared event; counter is atomic.
    _Atomic uint64_t            event_counter;
    PionSDPAWorker              workers[SDPA_MAX_WORKERS];
    int                         ready;
} PionMetalSDPAContext;

static PionMetalSDPAContext g_sdpa = {0};
static pthread_mutex_t g_sdpa_init_mutex = PTHREAD_MUTEX_INITIALIZER;

static int _sdpa_pso_index(uint32_t D) {
    for (int i = 0; i < SDPA_PSOS_COUNT; i++) {
        if (SDPA_D_VALUES[i] == D) return i;
    }
    return -1;
}

// Per-PSO threadgroup memory size for `s_o` in the SDPA kernels (sdpa_q1_*,
// sdpa_batched_q_*, sdpa_batched_q_fused_*). All six SDPA kernels use the same
// online-softmax shape: s_o[SG_PER_TG][CHUNK][32] of float4 (fp32) or half4
// (fp16). CHUNK = ceil(D/4 / 32). Host must call setThreadgroupMemoryLength
// before dispatchThreadgroups so the kernel's [[threadgroup(0)]] parameter is
// backed by the right amount of memory. fp32: 8*CHUNK*32*16 bytes; fp16:
// 8*CHUNK*32*8 bytes.
#define SDPA_SG_PER_TG 8u
static inline uint32_t _sdpa_chunk(uint32_t D) {
    uint32_t D4 = D / 4u;
    return (D4 + 31u) / 32u;
}
static inline size_t _sdpa_so_bytes(uint32_t D, uint32_t precision) {
    uint32_t chunk = _sdpa_chunk(D);
    // float4 = 16 bytes, half4 = 8 bytes
    size_t elem = (precision == 1u) ? 8u : 16u;
    return (size_t)SDPA_SG_PER_TG * chunk * 32u * elem;
}

// gh #130 §4.2: threadgroup K/V tile bytes for the tiled batched-Q kernel.
// Must match the kernel's `TILE_N = (TILE_KV_F4/(2*D4)) or 1` and the
// [2 * TILE_N * D4] float4 layout (K tile followed by V tile). TILE_KV_F4=1024.
#define SDPA_TILE_KV_F4 1024u
// gh #130 §4.2: M at/above which the tiled kernel beats the split kernel.
// Measured crossover ~16 (D=128, N=2048): tiled ≥1.06× at M=16, 1.7× at M=128;
// below this the split kernel's per-query token parallelism wins.
#define SDPA_BATCHED_TILE_MIN_M 16u
static inline uint32_t _sdpa_tile_n(uint32_t D) {
    uint32_t D4 = D / 4u;
    uint32_t t = SDPA_TILE_KV_F4 / (2u * D4);
    return t > 0u ? t : 1u;
}
static inline size_t _sdpa_batched_tile_bytes(uint32_t D, uint32_t precision) {
    uint32_t D4 = D / 4u;
    size_t elem = (precision == 1u) ? 8u : 16u;   // half4 : float4
    return (size_t)2u * _sdpa_tile_n(D) * D4 * elem;
}

static uint64_t _sdpa_key(const char *sid, uint32_t sid_len, uint32_t layer) {
    // FNV-1a 64-bit, with reserved sentinels mapped to 1 to avoid collision
    // with SDPA_KEY_EMPTY (0) / SDPA_KEY_TOMBSTONE (UINT64_MAX).
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < sid_len; i++) {
        h ^= (uint64_t)(uint8_t)sid[i];
        h *= 0x100000001b3ULL;
    }
    h ^= ((uint64_t)layer << 56);
    if (h == SDPA_KEY_EMPTY || h == SDPA_KEY_TOMBSTONE) h = 1ULL;
    return h;
}

// gh #67: forward decls — `_sdpa_resolve_slot` below calls `_sdpa_find_slot`
// which is defined later in this file (kept where it was to minimize churn).
static int _sdpa_find_slot(PionSDPAWorker *w, uint64_t key);

// gh #67: monotonic per-slot timestamp for LRU. mach_absolute_time() is the
// cheapest monotonic clock on macOS — sub-nanosecond per call, no syscall.
// Units are mach ticks; we never convert to wall-clock seconds (only used
// for ordering comparisons), so the conversion factor is irrelevant.
static inline uint64_t _sdpa_now_ns(void) {
    return mach_absolute_time();
}

// gh #67: linear scan over the per-worker cold registry. Returns index or -1.
// Cold lookups happen on QUERY miss + KV.PREFIX.WARM, both off the hot path.
static int _sdpa_cold_find(const PionSDPAWorker *w, uint64_t key) {
    for (uint32_t i = 0u; i < w->cold_count; i++) {
        if (w->cold[i].key == key) return (int)i;
    }
    return -1;
}

// gh #67: remove a cold entry by key (no-op if not present). Used on cold→warm
// promotion in `pion_metal_sdpa_store_kv` so a rehydrated session doesn't show
// up in both tables.
static void _sdpa_cold_remove(PionSDPAWorker *w, uint64_t key) {
    int idx = _sdpa_cold_find(w, key);
    if (idx < 0) return;
    if ((uint32_t)idx != w->cold_count - 1u) {
        w->cold[idx] = w->cold[w->cold_count - 1u];
    }
    w->cold_count--;
}

// gh #67: insert a cold entry. When the registry is full, drop the oldest by
// last_access_ns. The dropped session is gone for good — but V-store still has
// it on disk, so a future KV.PREFIX.LOOKUP + KV.PREFIX.WARM still succeeds
// (the WARM path goes V-store → ATTEND.PREFIX.STORE, not registry → cache).
static void _sdpa_cold_insert(PionSDPAWorker *w, uint64_t key, uint64_t last_access_ns,
                              uint32_t H, uint32_t N, uint32_t D) {
    if (w->cold_count >= SDPA_COLD_SLOTS) {
        // Find oldest by last_access_ns.
        uint32_t oldest = 0u;
        uint64_t oldest_t = w->cold[0].last_access_ns;
        for (uint32_t i = 1u; i < w->cold_count; i++) {
            if (w->cold[i].last_access_ns < oldest_t) {
                oldest_t = w->cold[i].last_access_ns;
                oldest = i;
            }
        }
        if (oldest != w->cold_count - 1u) {
            w->cold[oldest] = w->cold[w->cold_count - 1u];
        }
        w->cold_count--;
        w->cold_drops++;
    }
    w->cold[w->cold_count].key = key;
    w->cold[w->cold_count].last_access_ns = last_access_ns;
    w->cold[w->cold_count].H = H;
    w->cold[w->cold_count].N = N;
    w->cold[w->cold_count].D = D;
    w->cold_count++;
}

// gh #67: pick the LRU live slot in the probe chain rooted at `start`. Returns
// the slot index whose `last_access_ns` is smallest among slots with non-empty
// non-tombstone keys (i.e. WARM slots — COLD slots don't live in the slot
// table). Caller has already verified the chain has no empty/tombstone slot.
static int _sdpa_pick_lru_slot(const PionSDPAWorker *w, int start) {
    int victim = start;
    uint64_t victim_t = w->slots[start].last_access_ns;
    for (int i = 1; i < SDPA_SLOTS; i++) {
        int slot = (start + i) % SDPA_SLOTS;
        uint64_t k = w->slots[slot].key;
        if (k == SDPA_KEY_EMPTY || k == SDPA_KEY_TOMBSTONE) continue;
        if (w->slots[slot].last_access_ns < victim_t) {
            victim_t = w->slots[slot].last_access_ns;
            victim = slot;
        }
    }
    return victim;
}

// gh #67: free Metal buffers + K_mean on a slot in-place. Used both by
// drop() and by the WARM→COLD demotion path. Does NOT touch `key` or
// `last_access_ns` — the caller decides whether the slot becomes EMPTY,
// TOMBSTONE, or COLD with metadata retained.
static void _sdpa_release_slot_payload(PionSDPASlot *s) {
    s->K_buf = nil;
    s->V_buf = nil;
    if (s->K_mean) {
        free(s->K_mean);
        s->K_mean = NULL;
        s->K_mean_B = 0;
        s->K_mean_n_blocks = 0;
    }
    // W11 Phase 2: also drop Quest precompute on slot release.
    if (s->K_min) { free(s->K_min); s->K_min = NULL; }
    if (s->K_max) { free(s->K_max); s->K_max = NULL; }
    s->K_minmax_B = 0;
    s->K_minmax_n_blocks = 0;
}

// gh #67: combined slot + cold-registry probe. Used by every public query
// entry point so behaviour is identical across kernels. On WARM hit, the
// slot's last_access_ns is stamped (LRU advances on read, not just write).
// Returns SDPA_RC_OK + sets *out_slot, or SDPA_RC_COLD_MISS / SDPA_RC_HARD_MISS.
static int _sdpa_resolve_slot(PionSDPAWorker *w, uint64_t key, int *out_slot) {
    int slot = _sdpa_find_slot(w, key);
    if (slot >= 0) {
        PionSDPASlot *s = &w->slots[slot];
        if (s->state == SDPA_SLOT_WARM && s->K_buf) {
            s->last_access_ns = _sdpa_now_ns();
            *out_slot = slot;
            return SDPA_RC_OK;
        }
        // Half-allocated transient (alloc_slot ran but store_kv hasn't
        // committed yet) — treat as miss; the caller will fail H/D check
        // or fall back to MLX.
    }
    if (_sdpa_cold_find(w, key) >= 0) return SDPA_RC_COLD_MISS;
    return SDPA_RC_HARD_MISS;
}

// Linear probe from hash bucket, stopping on key match or empty (tombstones
// don't terminate the search). Returns -1 on miss.
static int _sdpa_find_slot(PionSDPAWorker *w, uint64_t key) {
    int start = (int)(key % SDPA_SLOTS);
    for (int i = 0; i < SDPA_SLOTS; i++) {
        int slot = (start + i) % SDPA_SLOTS;
        uint64_t k = w->slots[slot].key;
        if (k == key) return slot;
        if (k == SDPA_KEY_EMPTY) return -1;
        // tombstone → keep probing.
    }
    return -1;
}

// Reserve a slot for `key`. Returns existing slot if `key` already present
// (re-store same session), else first tombstone or empty slot via linear
// probe from the hash bucket.
//
// gh #67: when the table is full of live (WARM) entries, pick the LRU live
// slot in the probe chain, demote it to the cold registry (frees Metal
// K_buf/V_buf but retains key + H/N/D + last_access_ns), and reuse the
// slot for `key`. The demoted session is still rehydratable from the V-store
// via `KV.PREFIX.WARM` — old behaviour was a silent drop with no recovery.
static int _sdpa_alloc_slot(PionSDPAWorker *w, uint64_t key, int *out_slot) {
    int start = (int)(key % SDPA_SLOTS);
    int first_tombstone = -1;
    for (int i = 0; i < SDPA_SLOTS; i++) {
        int slot = (start + i) % SDPA_SLOTS;
        uint64_t k = w->slots[slot].key;
        if (k == key) { *out_slot = slot; return 0; }
        if (k == SDPA_KEY_EMPTY) {
            int target = (first_tombstone >= 0) ? first_tombstone : slot;
            w->slots[target].key = key;
            *out_slot = target;
            return 0;
        }
        if (k == SDPA_KEY_TOMBSTONE && first_tombstone < 0) first_tombstone = slot;
    }
    // Table fully populated with non-matching live entries — pick LRU victim,
    // demote to cold registry, then reuse its slot for the new key.
    int victim = _sdpa_pick_lru_slot(w, start);
    PionSDPASlot *vs = &w->slots[victim];
    _sdpa_cold_insert(w, vs->key, vs->last_access_ns, vs->H, vs->N, vs->D);
    w->cold_demotions++;
    _sdpa_release_slot_payload(vs);
    vs->key = key;
    vs->state = 0;            // not WARM yet — store_kv flips to WARM after memcpy.
    vs->last_access_ns = 0;
    *out_slot = victim;
    return 0;
}

// gh #398: under --metal-attention-fp16 every fp16 kernel converted each fp32
// K/V element to half on load. Storing half once at STORE halves the bytes the
// kernels stream and the slot's memory. The host's conversion and the kernel's
// cast can differ in an element's last bit, so outputs move in the 6th-7th
// significant digit (metal_compute.metal has the measurements). Process-wide:
// the engine sets it from its fp16 flag at startup, before any STORE.
static int g_sdpa_kv_half = 0;
void pion_metal_sdpa_set_kv_half(int32_t on) { g_sdpa_kv_half = on ? 1 : 0; }

// Bounds-check + return per-worker context, or NULL if invalid.
static PionSDPAWorker* _sdpa_worker(uint32_t worker_id) {
    if (worker_id >= SDPA_MAX_WORKERS) return NULL;
    return &g_sdpa.workers[worker_id];
}

int32_t pion_metal_sdpa_init(void) {
    // Idempotent and thread-safe: multiple workers may call this concurrently
    // at startup. The double-checked-locking dance avoids the mutex on the hot
    // path (already-initialized case) while serializing the initial setup.
    if (g_sdpa.ready) return 0;
    pthread_mutex_lock(&g_sdpa_init_mutex);
    if (g_sdpa.ready) { pthread_mutex_unlock(&g_sdpa_init_mutex); return 0; }
    @autoreleasepool {
        g_sdpa.device = MTLCreateSystemDefaultDevice();
        if (!g_sdpa.device) {
            fprintf(stderr, "[Metal SDPA] no Metal device\n");
            pthread_mutex_unlock(&g_sdpa_init_mutex);
            return -1;
        }
        g_sdpa.queue = [g_sdpa.device newCommandQueue];

        // Prefer pre-compiled metallib (matches pion_metal_init search paths).
        NSError *err = nil;
        id<MTLLibrary> lib = nil;
        NSString *sdpaLibPath = pion_metal_find(@"metal_compute.metallib");
        if (sdpaLibPath) {
            lib = [g_sdpa.device newLibraryWithURL:[NSURL fileURLWithPath:sdpaLibPath] error:&err];
        }
        if (!lib) {
            NSString *sdpaSrcPath = pion_metal_find(@"metal_compute.metal");
            NSString *src = sdpaSrcPath
                ? [NSString stringWithContentsOfFile:sdpaSrcPath
                                            encoding:NSUTF8StringEncoding error:&err]
                : nil;
            if (src) {
                MTLCompileOptions *opts = pion_metal_compile_options();
                lib = [g_sdpa.device newLibraryWithSource:src options:opts error:&err];
            }
        }
        if (!lib) {
            /* Hold the string: an NSString temporary has no strong reference at
             * the call site under ARC, so its UTF8String buffer can be gone
             * before fprintf reads it — which printed an EMPTY path list the
             * first time this ran. */
            NSString *tried = pion_metal_searched(@"metal_compute.metallib");
            NSString *why = err ? [err localizedDescription]
                                : (sdpaLibPath ? @"(no error reported)"
                                               : @"no metal_compute.metallib on any search path");
            fprintf(stderr,
                "[Metal SDPA] failed to load library: %s\n"
                "[Metal SDPA] looked in:\n    %s\n"
                "[Metal SDPA] set PION_METAL_LIB=/path/to/metal_compute.metallib to be explicit.\n"
                "Metal Attn: NOT ACTIVE — falling back to the MLX bridge.\n",
                [why UTF8String], [tried UTF8String]);
            pthread_mutex_unlock(&g_sdpa_init_mutex);
            return -2;
        }
        // Specialize one PSO per supported D_HEAD via function constants —
        // for both kernels (sdpa_q1_fp32 and sdpa_batched_q_fp32).
        // Same source kernel for each; the constant propagates through D4, CHUNK, etc.
        for (int i = 0; i < SDPA_PSOS_COUNT; i++) {
            uint32_t d = SDPA_D_VALUES[i];
            MTLFunctionConstantValues *fcv = [[MTLFunctionConstantValues alloc] init];
            [fcv setConstantValue:&d type:MTLDataTypeUInt atIndex:0];

            id<MTLFunction> fn = [lib newFunctionWithName:@"sdpa_q1_fp32"
                                            constantValues:fcv error:&err];
            if (!fn) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -3;
            }
            g_sdpa.psos[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn error:&err];
            if (!g_sdpa.psos[i]) {
                fprintf(stderr, "[Metal SDPA] PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -4;
            }

            id<MTLFunction> fn_b = [lib newFunctionWithName:@"sdpa_batched_q_fp32"
                                              constantValues:fcv error:&err];
            if (!fn_b) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -5;
            }
            g_sdpa.psos_batched[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_b error:&err];
            if (!g_sdpa.psos_batched[i]) {
                fprintf(stderr, "[Metal SDPA] batched PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -6;
            }

            // gh #130 §4.2: large-M tiled batched kernel (K/V staged in threadgroup mem).
            id<MTLFunction> fn_bt = [lib newFunctionWithName:@"sdpa_batched_q_tiled_fp32"
                                               constantValues:fcv error:&err];
            if (!fn_bt) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_tiled_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -6;
            }
            g_sdpa.psos_batched_tiled[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_bt error:&err];
            if (!g_sdpa.psos_batched_tiled[i]) {
                fprintf(stderr, "[Metal SDPA] tiled batched PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -6;
            }

            // FP16 variants (vanilla mlx-lm parity).
            id<MTLFunction> fn_h = [lib newFunctionWithName:@"sdpa_q1_fp16"
                                              constantValues:fcv error:&err];
            if (!fn_h) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_fp16 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -7;
            }
            g_sdpa.psos_fp16[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_h error:&err];
            if (!g_sdpa.psos_fp16[i]) {
                fprintf(stderr, "[Metal SDPA] fp16 PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -8;
            }

            id<MTLFunction> fn_bh = [lib newFunctionWithName:@"sdpa_batched_q_fp16"
                                               constantValues:fcv error:&err];
            if (!fn_bh) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_fp16 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -9;
            }
            g_sdpa.psos_batched_fp16[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_bh error:&err];
            if (!g_sdpa.psos_batched_fp16[i]) {
                fprintf(stderr, "[Metal SDPA] batched fp16 PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -10;
            }

            // gh #49: fused suffix kernels (fp32 + fp16). Same FCV (D_HEAD).
            id<MTLFunction> fn_fu = [lib newFunctionWithName:@"sdpa_batched_q_fused_fp32"
                                              constantValues:fcv error:&err];
            if (!fn_fu) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_fused_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -11;
            }
            g_sdpa.psos_fused[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_fu error:&err];
            if (!g_sdpa.psos_fused[i]) {
                fprintf(stderr, "[Metal SDPA] fused PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -12;
            }
            // gh #130 §4.2: large-M tiled fused kernel (K/V staged in threadgroup mem).
            id<MTLFunction> fn_fut = [lib newFunctionWithName:@"sdpa_batched_q_tiled_fused_fp32"
                                               constantValues:fcv error:&err];
            if (!fn_fut) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_tiled_fused_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -12;
            }
            g_sdpa.psos_fused_tiled[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_fut error:&err];
            if (!g_sdpa.psos_fused_tiled[i]) {
                fprintf(stderr, "[Metal SDPA] tiled fused PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -12;
            }
            id<MTLFunction> fn_fuh = [lib newFunctionWithName:@"sdpa_batched_q_fused_fp16"
                                               constantValues:fcv error:&err];
            if (!fn_fuh) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_batched_q_fused_fp16 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -13;
            }
            g_sdpa.psos_fused_fp16[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_fuh error:&err];
            if (!g_sdpa.psos_fused_fp16[i]) {
                fprintf(stderr, "[Metal SDPA] fused fp16 PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -14;
            }

            // gh #60 Phase 2: sparse-mask kernels (fp32 + fp16). Same FCV (D_HEAD).
            id<MTLFunction> fn_sp = [lib newFunctionWithName:@"sdpa_q1_sparse_fp32"
                                               constantValues:fcv error:&err];
            if (!fn_sp) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_sparse_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -15;
            }
            g_sdpa.psos_sparse[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_sp error:&err];
            if (!g_sdpa.psos_sparse[i]) {
                fprintf(stderr, "[Metal SDPA] sparse PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -16;
            }
            id<MTLFunction> fn_sph = [lib newFunctionWithName:@"sdpa_q1_sparse_fp16"
                                                constantValues:fcv error:&err];
            if (!fn_sph) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_sparse_fp16 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -17;
            }
            g_sdpa.psos_sparse_fp16[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_sph error:&err];
            if (!g_sdpa.psos_sparse_fp16[i]) {
                fprintf(stderr, "[Metal SDPA] sparse fp16 PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -18;
            }

            // gh #63 follow-on: sparse-fused (prefix-sparse + dense-suffix + merge).
            id<MTLFunction> fn_spf = [lib newFunctionWithName:@"sdpa_q1_sparse_fused_fp32"
                                                constantValues:fcv error:&err];
            if (!fn_spf) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_sparse_fused_fp32 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -19;
            }
            g_sdpa.psos_sparse_fused[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_spf error:&err];
            if (!g_sdpa.psos_sparse_fused[i]) {
                fprintf(stderr, "[Metal SDPA] sparse_fused PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -20;
            }
            id<MTLFunction> fn_spfh = [lib newFunctionWithName:@"sdpa_q1_sparse_fused_fp16"
                                                 constantValues:fcv error:&err];
            if (!fn_spfh) {
                fprintf(stderr, "[Metal SDPA] specialize sdpa_q1_sparse_fused_fp16 for D=%u failed: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -21;
            }
            g_sdpa.psos_sparse_fused_fp16[i] = [g_sdpa.device newComputePipelineStateWithFunction:fn_spfh error:&err];
            if (!g_sdpa.psos_sparse_fused_fp16[i]) {
                fprintf(stderr, "[Metal SDPA] sparse_fused fp16 PSO for D=%u error: %s\n",
                        d, err ? [[err localizedDescription] UTF8String] : "(no error)");
                pthread_mutex_unlock(&g_sdpa_init_mutex);
                return -22;
            }
        }
        g_sdpa.event = [g_sdpa.device newSharedEvent];
        atomic_store_explicit(&g_sdpa.event_counter, 0, memory_order_relaxed);

        // Per-worker: Q/O staging sized for H=32 d=SDPA_D_MAX worst case (32 KB each)
        // plus N/scale/M constant buffers. Batched Qb/Ob/LSE allocated lazily on
        // first M>1 query per worker.
        const uint32_t qcap = 32u * SDPA_D_MAX * 4u;
        for (int w = 0; w < SDPA_MAX_WORKERS; w++) {
            PionSDPAWorker *pw = &g_sdpa.workers[w];
            pw->Q_buf = [g_sdpa.device newBufferWithLength:qcap options:MTLResourceStorageModeShared];
            pw->O_buf = [g_sdpa.device newBufferWithLength:qcap options:MTLResourceStorageModeShared];
            pw->Q_capacity = qcap;
            pw->N_buf     = [g_sdpa.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            pw->scale_buf = [g_sdpa.device newBufferWithLength:sizeof(float)    options:MTLResourceStorageModeShared];
            pw->M_buf     = [g_sdpa.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            pw->W_buf     = [g_sdpa.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            *(uint32_t*)[pw->W_buf contents] = 0u;  // default: full attention
            pw->Qb_buf = nil; pw->Ob_buf = nil; pw->LSE_buf = nil;
            pw->Qb_capacity = pw->Ob_capacity = pw->LSE_capacity = 0;
            pw->Ks_buf = nil; pw->Vs_buf = nil; pw->HM_buf = nil;
            pw->Ks_capacity = pw->Vs_capacity = pw->HM_capacity = 0;
            pw->Ssuf_buf = [g_sdpa.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            // gh #60 Phase 2 sparse staging — indices/counts allocated lazily; Ksp scalar eager.
            pw->Idx_buf = nil; pw->Cnt_buf = nil;
            pw->Idx_capacity = pw->Cnt_capacity = 0;
            pw->Ksp_buf = [g_sdpa.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            memset(pw->slots, 0, sizeof(pw->slots));
        }
        g_sdpa.ready = 1;
        fprintf(stderr, "[Metal SDPA] initialized on %s (%d workers × %d slots, %d D-PSOs: ",
                [[g_sdpa.device name] UTF8String], SDPA_MAX_WORKERS, SDPA_SLOTS, SDPA_PSOS_COUNT);
        for (int i = 0; i < SDPA_PSOS_COUNT; i++) {
            fprintf(stderr, "%u%s", SDPA_D_VALUES[i], i == SDPA_PSOS_COUNT - 1 ? ")\n" : ",");
        }
        /* gh #281: the startup banner says "Metal Attn: requested". This is the
         * line that says whether the request was honoured — same key, so one
         * grep answers the question either way. */
        fprintf(stderr, "Metal Attn: ACTIVE (%s)\n",
                sdpaLibPath ? [sdpaLibPath UTF8String] : "compiled from source at runtime");
    }
    pthread_mutex_unlock(&g_sdpa_init_mutex);
    return 0;
}

// Store K/V for (session_id, layer_id) on a specific worker's cache. Each
// worker has its own SDPA_SLOTS-entry hash bucket array (shared-nothing).
// Forward decls for the selector-cache populators called from STORE
// (definitions live below). Keeps the STORE-time precompute call
// resolvable without reshuffling the file order.
static int _sdpa_ensure_kmean(PionSDPASlot *s, uint32_t B, const float *K_src);
static int _sdpa_ensure_kminmax(PionSDPASlot *s, uint32_t B, const float *K_src);

int32_t pion_metal_sdpa_store_kv(uint32_t worker_id,
                                 const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                 uint32_t H, uint32_t N, uint32_t D,
                                 const float *K, const float *V) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    if (_sdpa_pso_index(D) < 0) {
        fprintf(stderr, "[Metal SDPA] unsupported d_head=%u (supported: 64,96,128,160,192,256)\n", D);
        return -2;
    }
    @autoreleasepool {
        uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
        int slot = -1;
        if (_sdpa_alloc_slot(w, key, &slot) != 0) return -3;

        uint8_t want_half = (uint8_t)g_sdpa_kv_half;
        size_t n_elems = (size_t)H * N * D;
        size_t kv_bytes = n_elems * (want_half ? sizeof(_Float16) : sizeof(float));
        PionSDPASlot *s = &w->slots[slot];

        // Reuse existing buffers if same shape and element type; else reallocate.
        if (!s->K_buf || s->H != H || s->N < N || s->D != D || s->kv_half != want_half) {
            s->K_buf = [g_sdpa.device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            s->V_buf = [g_sdpa.device newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
        }
        s->H = H; s->N = N; s->D = D;
        s->kv_half = want_half;
        if (want_half) {
            // Round-to-nearest-even, once, instead of a cast on every load.
            _Float16 *kh = (_Float16 *)[s->K_buf contents];
            _Float16 *vh = (_Float16 *)[s->V_buf contents];
            for (size_t i = 0; i < n_elems; i++) { kh[i] = (_Float16)K[i]; vh[i] = (_Float16)V[i]; }
        } else {
            memcpy([s->K_buf contents], K, kv_bytes);
            memcpy([s->V_buf contents], V, kv_bytes);
        }
        // gh #63 Phase 3b: drop any stale K_mean — it's recomputed lazily on
        // the next QUERY_SPARSE_AUTO call. The selector caches with B=64 by
        // default; a future caller passing a different B reallocates here.
        if (s->K_mean) {
            free(s->K_mean);
            s->K_mean = NULL;
            s->K_mean_B = 0;
            s->K_mean_n_blocks = 0;
        }
        // W11 / gh #9 Phase 2: also drop Quest precompute on STORE — same
        // staleness reason (K changed → K_min/K_max no longer correct).
        if (s->K_min) { free(s->K_min); s->K_min = NULL; }
        if (s->K_max) { free(s->K_max); s->K_max = NULL; }
        s->K_minmax_B = 0;
        s->K_minmax_n_blocks = 0;

        // W11 Phase 2 follow-up: STORE-time selector precompute. The lazy
        // populate at first QUERY_SPARSE_AUTO call adds noticeable latency
        // to the user-facing first sparse query (measured at H=4 N=16K D=128:
        // block-mean +11 ms, Quest +19 ms). Doing the precompute here moves
        // that latency into the STORE call (already on the cold prefill
        // path) so the first query is cache-warm. B=64 is the canonical
        // block size used by both selectors; non-64 B values fall back to
        // the existing lazy path on first query.
        //
        // **Default policy**: K_mean eager (block-mean is the most-used
        // path); K_min/K_max lazy (Quest is opt-in, most STOREs won't be
        // followed by Quest queries).
        //
        // Env overrides:
        //   PION_SDPA_NO_STORE_PRECOMPUTE=1     → all eager precompute off
        //                                          (pure lazy on first query)
        //   PION_SDPA_STORE_PRECOMPUTE_QUEST=1  → also eagerly compute
        //                                          K_min/K_max at STORE
        //                                          (for Quest-heavy workloads
        //                                          willing to trade STORE
        //                                          latency for first-query)
        static int precompute_block_mean = -1;
        static int precompute_quest      = -1;
        if (precompute_block_mean < 0) {
            const char *env_off = getenv("PION_SDPA_NO_STORE_PRECOMPUTE");
            precompute_block_mean = (env_off && env_off[0] == '1') ? 0 : 1;
            const char *env_q   = getenv("PION_SDPA_STORE_PRECOMPUTE_QUEST");
            precompute_quest    = (env_off && env_off[0] == '1') ? 0
                                : ((env_q && env_q[0] == '1') ? 1 : 0);
        }
        // Selector statistics come from the fp32 input, so a half slot's
        // eager precompute is the same as an fp32 slot's.
        if (precompute_block_mean) {
            (void)_sdpa_ensure_kmean(s, 64u, K);
        }
        if (precompute_quest) {
            (void)_sdpa_ensure_kminmax(s, 64u, K);
        }
        // Failures tolerated silently — the lazy path retries on first query.

        // gh #67: slot is now WARM. Stamp LRU clock and clear any matching
        // cold-registry entry (cold→warm promotion) so a later QUERY doesn't
        // false-positive into COLD_MISS while the slot is hot.
        s->state = SDPA_SLOT_WARM;
        s->last_access_ns = _sdpa_now_ns();
        if (_sdpa_cold_find(w, key) >= 0) {
            _sdpa_cold_remove(w, key);
            w->cold_rehydrates++;
        }
    }
    return 0;
}

// gh #63 Phase 3b: ensure the slot's K_mean is computed at block size B.
// Lazily allocates + populates K_mean over the slot's resident K tensor.
// CPU loop — at H=4 N=64K D=512 B=64 this is ~130 MB of reads + 2M float
// writes, ~30-50 ms one-shot. Cached for all subsequent sparse-auto queries
// against this slot.
//
// Returns 0 on success, -1 on alloc failure.
// gh #398: element i of a slot's K, whichever type the slot stores. K_src,
// when given, is the fp32 tensor being stored (STORE-time precompute).
static inline float _sdpa_k_elem(const void *base, int half, size_t i) {
    return half ? (float)((const _Float16 *)base)[i] : ((const float *)base)[i];
}

static int _sdpa_ensure_kmean(PionSDPASlot *s, uint32_t B, const float *K_src) {
    if (s->K_mean && s->K_mean_B == B) return 0;
    if (B == 0u) return -1;
    if (s->K_mean) { free(s->K_mean); s->K_mean = NULL; }
    uint32_t n_blocks = (s->N + B - 1u) / B;
    size_t bytes = (size_t)s->H * (size_t)n_blocks * (size_t)s->D * sizeof(float);
    float *buf = (float*)malloc(bytes);
    if (!buf) return -1;
    const void *K = K_src ? (const void *)K_src : [s->K_buf contents];
    int kh = K_src ? 0 : s->kv_half;
    // K layout [H, N, D]; K_mean layout [H, n_blocks, D].
    for (uint32_t h = 0u; h < s->H; h++) {
        size_t Kh = (size_t)h * s->N * s->D;
        float       *Mh = buf + (size_t)h * n_blocks * s->D;
        for (uint32_t b = 0u; b < n_blocks; b++) {
            uint32_t t_begin = b * B;
            uint32_t t_end   = t_begin + B;
            if (t_end > s->N) t_end = s->N;
            uint32_t cnt = t_end - t_begin;
            float *Mb = Mh + (size_t)b * s->D;
            // Sum across `cnt` tokens at this (head, block).
            for (uint32_t d = 0u; d < s->D; d++) Mb[d] = 0.0f;
            for (uint32_t t = t_begin; t < t_end; t++) {
                size_t Kt = Kh + (size_t)t * s->D;
                for (uint32_t d = 0u; d < s->D; d++) Mb[d] += _sdpa_k_elem(K, kh, Kt + d);
            }
            float inv = 1.0f / (float)cnt;
            for (uint32_t d = 0u; d < s->D; d++) Mb[d] *= inv;
        }
    }
    s->K_mean = buf;
    s->K_mean_B = B;
    s->K_mean_n_blocks = n_blocks;
    return 0;
}

// W11 / gh #9 Phase 2: ensure the slot's Quest precompute (K_min + K_max) is
// computed at block size B. Mirrors _sdpa_ensure_kmean. Each block's per-dim
// min and max enable the Quest upper bound:
//   UB(block) = Σ_d max(Q[d]·K_min[h,b,d], Q[d]·K_max[h,b,d])
// which is provably ≥ max_t (Q·K_t) over the block. Used when the wire
// caller passes selector_id=1.
//
// Cost: same memory bandwidth as ensure_kmean (one pass over K per block),
// 2× the writes (two output buffers vs one). ~60-100 ms one-shot at H=4
// N=64K D=512 B=64; cached for lifetime of the slot.
//
// Returns 0 on success, -1 on alloc failure.
static int _sdpa_ensure_kminmax(PionSDPASlot *s, uint32_t B, const float *K_src) {
    if (s->K_min && s->K_max && s->K_minmax_B == B) return 0;
    if (B == 0u) return -1;
    if (s->K_min) { free(s->K_min); s->K_min = NULL; }
    if (s->K_max) { free(s->K_max); s->K_max = NULL; }
    uint32_t n_blocks = (s->N + B - 1u) / B;
    size_t bytes = (size_t)s->H * (size_t)n_blocks * (size_t)s->D * sizeof(float);
    float *bmin = (float*)malloc(bytes);
    float *bmax = (float*)malloc(bytes);
    if (!bmin || !bmax) { if (bmin) free(bmin); if (bmax) free(bmax); return -1; }
    const void *K = K_src ? (const void *)K_src : [s->K_buf contents];
    int kh = K_src ? 0 : s->kv_half;
    for (uint32_t h = 0u; h < s->H; h++) {
        size_t Kh = (size_t)h * s->N * s->D;
        float *MNh = bmin + (size_t)h * n_blocks * s->D;
        float *MXh = bmax + (size_t)h * n_blocks * s->D;
        for (uint32_t b = 0u; b < n_blocks; b++) {
            uint32_t t_begin = b * B;
            uint32_t t_end   = t_begin + B;
            if (t_end > s->N) t_end = s->N;
            float *Mnb = MNh + (size_t)b * s->D;
            float *Mxb = MXh + (size_t)b * s->D;
            // Initialise from the first token in the block.
            size_t Kt0 = Kh + (size_t)t_begin * s->D;
            for (uint32_t d = 0u; d < s->D; d++) {
                float v0 = _sdpa_k_elem(K, kh, Kt0 + d);
                Mnb[d] = v0; Mxb[d] = v0;
            }
            for (uint32_t t = t_begin + 1u; t < t_end; t++) {
                size_t Kt = Kh + (size_t)t * s->D;
                for (uint32_t d = 0u; d < s->D; d++) {
                    float v = _sdpa_k_elem(K, kh, Kt + d);
                    if (v < Mnb[d]) Mnb[d] = v;
                    if (v > Mxb[d]) Mxb[d] = v;
                }
            }
        }
    }
    s->K_min = bmin;
    s->K_max = bmax;
    s->K_minmax_B = B;
    s->K_minmax_n_blocks = n_blocks;
    return 0;
}

// Top-K argsort over a per-block score vector (descending). Returns indices
// in `out_idx` (size K_top). Caller must ensure K_top ≤ n_blocks.
// Small-N partial-sort; n_blocks is typically ≤ 1024 (64K context / B=64).
static void _topk_argsort_desc(const float *scores, uint32_t n_blocks, uint32_t K_top, uint32_t *out_idx) {
    // Initialize out_idx as a min-heap of the top K_top blocks (by score).
    // Simple O(n*K) partial sort; for K_top=8 and n_blocks=1024 that's ~8000
    // comparisons — sub-microsecond.
    for (uint32_t i = 0u; i < K_top; i++) out_idx[i] = i;
    // Find the position of the current minimum-score selected block.
    uint32_t worst_pos = 0u;
    float    worst_val = scores[out_idx[0]];
    for (uint32_t i = 1u; i < K_top; i++) {
        if (scores[out_idx[i]] < worst_val) { worst_val = scores[out_idx[i]]; worst_pos = i; }
    }
    for (uint32_t i = K_top; i < n_blocks; i++) {
        if (scores[i] > worst_val) {
            out_idx[worst_pos] = i;
            // Rescan for new minimum.
            worst_pos = 0u;
            worst_val = scores[out_idx[0]];
            for (uint32_t j = 1u; j < K_top; j++) {
                if (scores[out_idx[j]] < worst_val) { worst_val = scores[out_idx[j]]; worst_pos = j; }
            }
        }
    }
    // Sort selected indices ascending (preserves rotary position order, matches
    // the in-proc selector convention in pion-vllm-mlx/mlx_lm_patch.py).
    for (uint32_t i = 0u; i < K_top; i++) {
        for (uint32_t j = i + 1u; j < K_top; j++) {
            if (out_idx[j] < out_idx[i]) {
                uint32_t t = out_idx[i]; out_idx[i] = out_idx[j]; out_idx[j] = t;
            }
        }
    }
}

// Drop stored K/V for (session_id, layer_id) on a specific worker.
// gh #65 follow-on: client-visible existence check on the Metal SDPA session
// cache. KV.PREFIX.* state and ATTEND.PREFIX.* state are tracked separately
// in Pion — V-store registration via KV.PREFIX.REGISTER vs Metal-resident
// K/V via ATTEND.PREFIX.STORE. A consumer that only registers V-store can't
// rely on KV.PREFIX.LOOKUP to know whether Metal cache is hot. This entry
// exposes the per-slot existence check that ATTEND.PREFIX.QUERY uses
// internally. Returns 1 if slot exists with non-nil K/V, 0 if missing.
//
// gh #67: existence means WARM specifically — a session that's been evicted
// to the cold registry reports MISS here. Use pion_metal_sdpa_session_state
// for warm/cold/missing discrimination.
int32_t pion_metal_sdpa_session_exists(uint32_t worker_id,
                                       const char *session_id, uint32_t sid_len, uint32_t layer_id) {
    if (!g_sdpa.ready) return 0;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return 0;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = _sdpa_find_slot(w, key);
    if (slot < 0) return 0;
    if (w->slots[slot].state != SDPA_SLOT_WARM) return 0;
    return w->slots[slot].K_buf ? 1 : 0;
}

// gh #67: tri-state probe used by ATTEND.PREFIX.LOOKUP and the auto-rehydrate
// path. Returns 0 = MISSING, 1 = WARM (slot has live K_buf), 2 = COLD (slot
// was evicted but metadata is still in the cold registry — KV.PREFIX.WARM
// can rehydrate).
int32_t pion_metal_sdpa_session_state(uint32_t worker_id,
                                      const char *session_id, uint32_t sid_len, uint32_t layer_id) {
    if (!g_sdpa.ready) return 0;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return 0;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = _sdpa_find_slot(w, key);
    if (slot >= 0 && w->slots[slot].state == SDPA_SLOT_WARM && w->slots[slot].K_buf) return 1;
    if (_sdpa_cold_find(w, key) >= 0) return 2;
    return 0;
}

// gh #67: cold-tier telemetry for tests + KV.PREFIX.INFO. Caller passes four
// uint64_t out-pointers (any may be NULL to skip). Returns -1 if not ready,
// -8 if invalid worker, 0 on success.
//   warm_count:   number of slots in WARM state (live K_buf)
//   cold_count:   number of entries in the cold registry
//   demotions:    cumulative WARM→COLD transitions on this worker
//   rehydrates:   cumulative COLD→WARM transitions on this worker
int32_t pion_metal_sdpa_cold_stats(uint32_t worker_id,
                                   uint64_t *warm_count,
                                   uint64_t *cold_count,
                                   uint64_t *demotions,
                                   uint64_t *rehydrates) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    if (warm_count) {
        uint64_t c = 0;
        for (int i = 0; i < SDPA_SLOTS; i++) {
            if (w->slots[i].state == SDPA_SLOT_WARM && w->slots[i].K_buf) c++;
        }
        *warm_count = c;
    }
    if (cold_count) *cold_count = (uint64_t)w->cold_count;
    if (demotions) *demotions = w->cold_demotions;
    if (rehydrates) *rehydrates = w->cold_rehydrates;
    return 0;
}

int32_t pion_metal_sdpa_drop(uint32_t worker_id,
                             const char *session_id, uint32_t sid_len, uint32_t layer_id) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    // gh #67: also clear any cold-registry entry for this key so a future
    // QUERY doesn't report COLD_MISS for an explicitly-dropped session.
    _sdpa_cold_remove(w, key);
    int slot = _sdpa_find_slot(w, key);
    if (slot < 0) return 0;  // not found is OK
    // Leave a tombstone (not EMPTY) so linear-probe finds for keys that were
    // displaced past this slot still succeed.
    w->slots[slot].key = SDPA_KEY_TOMBSTONE;
    w->slots[slot].state = 0;
    _sdpa_release_slot_payload(&w->slots[slot]);
    return 0;
}

// Returns 0 on success and writes H*D float32 to `out`. -1 if no stored K/V
// for (worker_id, session_id, layer_id) — caller can fall back to MLX bridge.
// precision: 0 = fp32 (default, more precise), 1 = fp16 (matches vanilla mlx-lm).
int32_t pion_metal_sdpa_query(uint32_t worker_id,
                              const char *session_id, uint32_t sid_len, uint32_t layer_id,
                              uint32_t H, uint32_t D, uint32_t precision, uint32_t window,
                              const float *Q, float *out) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    int rc = _sdpa_resolve_slot(w, key, &slot);
    if (rc != SDPA_RC_OK) return rc;  // -1 hard miss, -10 cold miss (gh #67)

    PionSDPASlot *s = &w->slots[slot];
    if (s->H != H || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V

    @autoreleasepool {
        size_t q_bytes = (size_t)H * D * sizeof(float);
        if (q_bytes > w->Q_capacity) return -4;
        memcpy([w->Q_buf contents], Q, q_bytes);
        *(uint32_t*)[w->N_buf contents] = s->N;
        *(float*)[w->scale_buf contents] = 1.0f / sqrtf((float)D);
        *(uint32_t*)[w->W_buf contents] = window;

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        id<MTLComputePipelineState> pso = (precision == 1u)
            ? g_sdpa.psos_fp16[pso_idx]
            : g_sdpa.psos[pso_idx];
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Q_buf offset:0 atIndex:0];
        [ce setBuffer:s->K_buf offset:0 atIndex:1];
        [ce setBuffer:s->V_buf offset:0 atIndex:2];
        [ce setBuffer:w->O_buf offset:0 atIndex:3];
        [ce setBuffer:w->N_buf offset:0 atIndex:4];
        [ce setBuffer:w->scale_buf offset:0 atIndex:5];
        [ce setBuffer:w->W_buf offset:0 atIndex:6];
        [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
        [ce dispatchThreadgroups:MTLSizeMake(H, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];  // 8 simdgroups (matches kernel SG_PER_TG)
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        // Spin-wait — same trick as the standalone bench, ~300µs faster than
        // waitUntilCompleted (no kernel-mode transition).
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->O_buf contents], q_bytes);
    }
    return 0;
}

// gh #60 Phase 2: sparse-mask single-Q. Same K/V cache slot lookup as
// pion_metal_sdpa_query, but the kernel iterates over caller-supplied
// per-head index lists instead of [eff_start..N). H*K_sparse_max ints +
// H uints get staged into per-worker buffers. Returns -2 if D PSO missing,
// -3 on H/D mismatch, -4 if Q exceeds capacity, -5 if K_sparse_max == 0.
int32_t pion_metal_sdpa_query_sparse(uint32_t worker_id,
                                     const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                     uint32_t H_q, uint32_t D, uint32_t precision, uint32_t window,
                                     uint32_t K_sparse_max,
                                     uint32_t H_kv,
                                     const float *Q,
                                     const int32_t *indices,
                                     const uint32_t *counts,
                                     const uint8_t *head_map,    // [H_q]; head_map[h_q] = h_kv. May be NULL on non-GQA.
                                     float *out) {
    if (!g_sdpa.ready) return -1;
    if (K_sparse_max == 0u) return -5;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    {
        int _rc = _sdpa_resolve_slot(w, key, &slot);
        if (_rc != SDPA_RC_OK) return _rc;  // gh #67: -1 hard miss, -10 cold miss
    }

    PionSDPASlot *s = &w->slots[slot];
    if (H_kv == 0u) H_kv = H_q;             // non-GQA shortcut
    if (s->H != H_kv || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V
    if (H_q < H_kv || (H_q % H_kv) != 0u) return -7;

    @autoreleasepool {
        size_t q_bytes   = (size_t)H_q * D * sizeof(float);
        size_t idx_bytes = (size_t)H_q * K_sparse_max * sizeof(int32_t);
        size_t cnt_bytes = (size_t)H_q * sizeof(uint32_t);
        size_t hm_bytes  = (size_t)H_q;
        if (q_bytes > w->Q_capacity) return -4;

        if (w->Idx_capacity < idx_bytes) {
            size_t cap = (idx_bytes + 65535u) & ~(size_t)65535u;
            w->Idx_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Idx_capacity = cap;
        }
        if (w->Cnt_capacity < cnt_bytes) {
            size_t cap = (cnt_bytes + 4095u) & ~(size_t)4095u;
            w->Cnt_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Cnt_capacity = cap;
        }
        if (w->HM_capacity < hm_bytes) {
            size_t cap = (hm_bytes + 4095u) & ~(size_t)4095u;
            w->HM_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->HM_capacity = cap;
        }

        memcpy([w->Q_buf   contents], Q,       q_bytes);
        memcpy([w->Idx_buf contents], indices, idx_bytes);
        memcpy([w->Cnt_buf contents], counts,  cnt_bytes);
        // Synthesize identity head_map when caller passes NULL (non-GQA).
        uint8_t *hm_dst = (uint8_t*)[w->HM_buf contents];
        if (head_map) {
            memcpy(hm_dst, head_map, hm_bytes);
        } else {
            uint32_t rep = H_q / H_kv;
            for (uint32_t h = 0u; h < H_q; h++) hm_dst[h] = (uint8_t)(h / rep);
        }
        *(uint32_t*)[w->N_buf  contents]  = s->N;
        *(float*)   [w->scale_buf contents] = 1.0f / sqrtf((float)D);
        *(uint32_t*)[w->W_buf  contents]  = window;
        *(uint32_t*)[w->Ksp_buf contents] = K_sparse_max;

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        id<MTLComputePipelineState> pso = (precision == 1u)
            ? g_sdpa.psos_sparse_fp16[pso_idx]
            : g_sdpa.psos_sparse[pso_idx];
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Q_buf     offset:0 atIndex:0];
        [ce setBuffer:s->K_buf     offset:0 atIndex:1];
        [ce setBuffer:s->V_buf     offset:0 atIndex:2];
        [ce setBuffer:w->O_buf     offset:0 atIndex:3];
        [ce setBuffer:w->N_buf     offset:0 atIndex:4];
        [ce setBuffer:w->scale_buf offset:0 atIndex:5];
        [ce setBuffer:w->W_buf     offset:0 atIndex:6];
        [ce setBuffer:w->Idx_buf   offset:0 atIndex:7];
        [ce setBuffer:w->Cnt_buf   offset:0 atIndex:8];
        [ce setBuffer:w->Ksp_buf   offset:0 atIndex:9];
        [ce setBuffer:w->HM_buf    offset:0 atIndex:10];
        // H_kv as a uint constant buffer — reuse N_buf? No, distinct. Use M_buf as scratch since it's not used in M=1 kernel.
        *(uint32_t*)[w->M_buf contents] = H_kv;
        [ce setBuffer:w->M_buf     offset:0 atIndex:11];
        [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
        [ce dispatchThreadgroups:MTLSizeMake(H_q, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->O_buf contents], q_bytes);
    }
    return 0;
}

// gh #63 Phase 3b: server-side block-mean top-K selector + sparse attention
// in ONE wire call. Caller passes Q + B + K_top; server picks the top-K
// blocks via block-mean QK scoring (averaged across query heads → shared
// mask v1), expands to token indices, and dispatches the existing sparse
// kernel. No client-side scoring code, no K_mean download, single
// round-trip per decode.
//
// The "moat realized" path: combined with the resident K/V in this slot,
// the warm decode wire cost is `H*D*4 (Q) + H*D*4 (out)` bytes regardless
// of prefix length. K/V never crosses the wire post-cold-prefill.
//
// Returns 0 on success. Negative codes mirror pion_metal_sdpa_query_sparse:
//   -1 = not ready / cache miss, -2 = unsupported D, -3 = H/D mismatch,
//   -4 = Q overflows Q_capacity, -5 = K_top == 0 or B == 0, -8 = bad worker_id.
int32_t pion_metal_sdpa_query_sparse_auto(uint32_t worker_id,
                                          const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                          uint32_t H_q, uint32_t D, uint32_t precision, uint32_t window,
                                          uint32_t B, uint32_t K_top,
                                          uint32_t H_kv,
                                          const float *Q,
                                          const uint8_t *head_map,    // [H_q] for GQA; NULL on non-GQA (H_q == H_kv).
                                          float *out,
                                          uint32_t selector_id) {        // W11 Phase 2: 0=block-mean (default), 1=quest
    if (!g_sdpa.ready) return -1;
    if (B == 0u || K_top == 0u) return -5;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    {
        int _rc = _sdpa_resolve_slot(w, key, &slot);
        if (_rc != SDPA_RC_OK) return _rc;  // gh #67: -1 hard miss, -10 cold miss
    }

    PionSDPASlot *s = &w->slots[slot];
    if (H_kv == 0u) H_kv = H_q;             // non-GQA shortcut
    if (s->H != H_kv || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V
    if (H_q < H_kv || (H_q % H_kv) != 0u) return -7;
    uint32_t rep = H_q / H_kv;

    // Lazily compute the selector's precompute for this block size. block-mean
    // (selector_id=0, default) uses K_mean; Quest (selector_id=1) uses
    // K_min/K_max. One-shot cost ~30-100 ms at 64K; cached for the slot's
    // lifetime. selector_id values > 1 are treated as block-mean for forward
    // compatibility.
    uint32_t n_blocks = 0u;
    if (selector_id == 1u) {
        if (_sdpa_ensure_kminmax(s, B, NULL) != 0) return -4;
        n_blocks = s->K_minmax_n_blocks;
    } else {
        if (_sdpa_ensure_kmean(s, B, NULL) != 0) return -4;
        n_blocks = s->K_mean_n_blocks;
    }
    if (K_top > n_blocks) K_top = n_blocks;

    @autoreleasepool {
        size_t q_bytes   = (size_t)H_q * D * sizeof(float);
        size_t hm_bytes  = (size_t)H_q;
        if (q_bytes > w->Q_capacity) return -4;

        if (w->HM_capacity < hm_bytes) {
            size_t cap = (hm_bytes + 4095u) & ~(size_t)4095u;
            w->HM_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->HM_capacity = cap;
        }
        uint8_t *hm_dst = (uint8_t*)[w->HM_buf contents];
        if (head_map) {
            memcpy(hm_dst, head_map, hm_bytes);
        } else {
            for (uint32_t h = 0u; h < H_q; h++) hm_dst[h] = (uint8_t)(h / rep);
        }

        // Compute block scores. v1: shared mask across query heads (mean of
        // per-head scores). GQA: query heads in the same kv-group dot their
        // distinct Q with the SAME precompute row.
        //
        // selector_id=0 (block-mean): score[b] = mean_h( Q[h] · K_mean[h_kv,b] ) * inv_sqrt_D
        // selector_id=1 (Quest UB):   score[b] = mean_h( Σ_d max(Q[h,d]·K_min[h_kv,b,d],
        //                                                        Q[h,d]·K_max[h_kv,b,d]) ) * inv_sqrt_D
        float *scores = (float*)alloca((size_t)n_blocks * sizeof(float));
        const float inv_sqrt_D = 1.0f / sqrtf((float)D);
        for (uint32_t b = 0u; b < n_blocks; b++) scores[b] = 0.0f;
        //
        // gh #191: both inner loops are 4-way unrolled with INDEPENDENT
        // accumulators. The bottleneck was never the multiplies — it was the
        // single serial FADD chain, which C semantics forbid the compiler from
        // reassociating. FMA latency on M-series is ~4 cycles against 1-cycle
        // throughput, so one chain leaves roughly three quarters of the FP
        // pipeline idle no matter how wide the vector is.
        //
        // Measured on a 64K session (n_blocks=1024, D=128, H_q=8, H_kv=2):
        //   block-mean  0.368 -> 0.042 ms  (8.8x)
        //   Quest UB    0.264 -> 0.086 ms  (3.1x)
        // Both at or past the measured -ffast-math ceiling, WITHOUT the flag: the reassociation
        // here is explicit and fixed, so results are deterministic run to run
        // rather than at the optimizer's discretion.
        //
        // The correctness surface is the selected block SET, not the scores.
        // Reassociation moves the last ulps, which can reorder near-tied
        // blocks in _topk_argsort_desc below. Measured over the whole score
        // vector: max relative delta 1.0e-4 (block-mean) / 2.8e-7 (Quest), and
        // the top-400 set and its order were both IDENTICAL. K_top >= 400 is
        // the gh #9 correctness floor, so a set change is what would matter.
        if (selector_id == 1u) {
            // Quest upper bound.
            for (uint32_t h = 0u; h < H_q; h++) {
                uint32_t h_kv = hm_dst[h];
                const float *Qh  = Q + (size_t)h * D;
                const float *Nh  = s->K_min + (size_t)h_kv * n_blocks * D;
                const float *Xh  = s->K_max + (size_t)h_kv * n_blocks * D;
                for (uint32_t b = 0u; b < n_blocks; b++) {
                    const float *Nb = Nh + (size_t)b * D;
                    const float *Xb = Xh + (size_t)b * D;
                    float ub = 0.0f;
                    uint32_t d = 0u;
#if defined(__ARM_NEON)
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
                    for (; d + 16u <= D; d += 16u) {
                        float32x4_t q0 = vld1q_f32(Qh + d);
                        float32x4_t q1 = vld1q_f32(Qh + d + 4u);
                        float32x4_t q2 = vld1q_f32(Qh + d + 8u);
                        float32x4_t q3 = vld1q_f32(Qh + d + 12u);
                        a0 = vaddq_f32(a0, vmaxq_f32(vmulq_f32(q0, vld1q_f32(Nb + d)),
                                                     vmulq_f32(q0, vld1q_f32(Xb + d))));
                        a1 = vaddq_f32(a1, vmaxq_f32(vmulq_f32(q1, vld1q_f32(Nb + d + 4u)),
                                                     vmulq_f32(q1, vld1q_f32(Xb + d + 4u))));
                        a2 = vaddq_f32(a2, vmaxq_f32(vmulq_f32(q2, vld1q_f32(Nb + d + 8u)),
                                                     vmulq_f32(q2, vld1q_f32(Xb + d + 8u))));
                        a3 = vaddq_f32(a3, vmaxq_f32(vmulq_f32(q3, vld1q_f32(Nb + d + 12u)),
                                                     vmulq_f32(q3, vld1q_f32(Xb + d + 12u))));
                    }
                    for (; d + 4u <= D; d += 4u) {
                        float32x4_t q0 = vld1q_f32(Qh + d);
                        a0 = vaddq_f32(a0, vmaxq_f32(vmulq_f32(q0, vld1q_f32(Nb + d)),
                                                     vmulq_f32(q0, vld1q_f32(Xb + d))));
                    }
                    ub = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#endif
                    for (; d < D; d++) {   // D not a multiple of 4, or no NEON
                        float qd = Qh[d];
                        float a  = qd * Nb[d];
                        float c  = qd * Xb[d];
                        ub += (a > c) ? a : c;
                    }
                    scores[b] += ub * inv_sqrt_D;
                }
            }
        } else {
            // Block-mean (default / forward-compat).
            for (uint32_t h = 0u; h < H_q; h++) {
                uint32_t h_kv = hm_dst[h];
                const float *Qh = Q + (size_t)h * D;
                const float *Mh = s->K_mean + (size_t)h_kv * n_blocks * D;
                for (uint32_t b = 0u; b < n_blocks; b++) {
                    const float *Mb = Mh + (size_t)b * D;
                    float dot = 0.0f;
                    uint32_t d = 0u;
#if defined(__ARM_NEON)
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
                    for (; d + 16u <= D; d += 16u) {
                        a0 = vfmaq_f32(a0, vld1q_f32(Qh + d),       vld1q_f32(Mb + d));
                        a1 = vfmaq_f32(a1, vld1q_f32(Qh + d + 4u),  vld1q_f32(Mb + d + 4u));
                        a2 = vfmaq_f32(a2, vld1q_f32(Qh + d + 8u),  vld1q_f32(Mb + d + 8u));
                        a3 = vfmaq_f32(a3, vld1q_f32(Qh + d + 12u), vld1q_f32(Mb + d + 12u));
                    }
                    for (; d + 4u <= D; d += 4u)
                        a0 = vfmaq_f32(a0, vld1q_f32(Qh + d), vld1q_f32(Mb + d));
                    dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
#endif
                    for (; d < D; d++) dot += Qh[d] * Mb[d];
                    scores[b] += dot * inv_sqrt_D;
                }
            }
        }
        const float inv_H = 1.0f / (float)H_q;
        for (uint32_t b = 0u; b < n_blocks; b++) scores[b] *= inv_H;

        // Top-K block argsort (ascending order in output, recency append below).
        uint32_t *top_blocks = (uint32_t*)alloca((size_t)K_top * sizeof(uint32_t));
        _topk_argsort_desc(scores, n_blocks, K_top, top_blocks);

        // Build per-head indices array. Shared mask → same indices for every head.
        // Always include the last partial block (recency keep) if it wasn't picked.
        uint32_t K_full = K_top * B;
        uint32_t last_blk_start = (n_blocks > 0u) ? (n_blocks - 1u) * B : 0u;
        uint32_t last_blk_n     = (s->N > last_blk_start) ? (s->N - last_blk_start) : 0u;
        int last_already = 0;
        if (n_blocks > 0u) {
            for (uint32_t i = 0u; i < K_top; i++) {
                if (top_blocks[i] == n_blocks - 1u) { last_already = 1; break; }
            }
        }
        uint32_t recency_n = (last_already || last_blk_n == 0u) ? 0u : last_blk_n;
        uint32_t K_sparse_max = K_full + recency_n;
        if (K_sparse_max == 0u) return -5;

        // Indices/counts buffers sized by H_q (the kernel dispatches H_q TGs).
        size_t idx_bytes = (size_t)H_q * K_sparse_max * sizeof(int32_t);
        size_t cnt_bytes = (size_t)H_q * sizeof(uint32_t);
        if (w->Idx_capacity < idx_bytes) {
            size_t cap = (idx_bytes + 65535u) & ~(size_t)65535u;
            w->Idx_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Idx_capacity = cap;
        }
        if (w->Cnt_capacity < cnt_bytes) {
            size_t cap = (cnt_bytes + 4095u) & ~(size_t)4095u;
            w->Cnt_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Cnt_capacity = cap;
        }

        int32_t *idx_dst = (int32_t*)[w->Idx_buf contents];
        int32_t *row = idx_dst;
        uint32_t actual_count = 0u;
        for (uint32_t i = 0u; i < K_top; i++) {
            uint32_t blk = top_blocks[i];
            uint32_t t_begin = blk * B;
            uint32_t t_end   = t_begin + B;
            if (t_end > s->N) t_end = s->N;
            for (uint32_t t = t_begin; t < t_end; t++) {
                row[actual_count++] = (int32_t)t;
            }
        }
        if (recency_n > 0u) {
            for (uint32_t t = last_blk_start; t < last_blk_start + recency_n; t++) {
                row[actual_count++] = (int32_t)t;
            }
        }
        // Replicate row across H_q heads.
        for (uint32_t h = 1u; h < H_q; h++) {
            memcpy(row + (size_t)h * K_sparse_max, row, (size_t)K_sparse_max * sizeof(int32_t));
        }
        uint32_t *cnt_dst = (uint32_t*)[w->Cnt_buf contents];
        for (uint32_t h = 0u; h < H_q; h++) cnt_dst[h] = actual_count;

        // Stage Q + scalars and dispatch the sparse kernel.
        memcpy([w->Q_buf contents], Q, q_bytes);
        *(uint32_t*)[w->N_buf  contents]  = s->N;
        *(float*)   [w->scale_buf contents] = inv_sqrt_D;
        *(uint32_t*)[w->W_buf  contents]  = window;
        *(uint32_t*)[w->Ksp_buf contents] = K_sparse_max;
        *(uint32_t*)[w->M_buf contents]   = H_kv;   // reuse M_buf for H_kv scalar

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        id<MTLComputePipelineState> pso = (precision == 1u)
            ? g_sdpa.psos_sparse_fp16[pso_idx]
            : g_sdpa.psos_sparse[pso_idx];
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Q_buf     offset:0 atIndex:0];
        [ce setBuffer:s->K_buf     offset:0 atIndex:1];
        [ce setBuffer:s->V_buf     offset:0 atIndex:2];
        [ce setBuffer:w->O_buf     offset:0 atIndex:3];
        [ce setBuffer:w->N_buf     offset:0 atIndex:4];
        [ce setBuffer:w->scale_buf offset:0 atIndex:5];
        [ce setBuffer:w->W_buf     offset:0 atIndex:6];
        [ce setBuffer:w->Idx_buf   offset:0 atIndex:7];
        [ce setBuffer:w->Cnt_buf   offset:0 atIndex:8];
        [ce setBuffer:w->Ksp_buf   offset:0 atIndex:9];
        [ce setBuffer:w->HM_buf    offset:0 atIndex:10];
        [ce setBuffer:w->M_buf     offset:0 atIndex:11];
        [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
        [ce dispatchThreadgroups:MTLSizeMake(H_q, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->O_buf contents], q_bytes);
    }
    return 0;
}

// gh #63 follow-on: sparse-AUTO + dense-suffix in a single FUSED kernel call.
// Same selector as pion_metal_sdpa_query_sparse_auto (server picks top-K via
// block-mean Q·K_mean scoring), plus a dense suffix loop over K_suf/V_suf
// with online-softmax merge across both phases. Single round-trip; no
// client-side suffix merge needed; the wire-mode sparse consumer (gh #63
// mlx_lm_patch wire-sparse branch) no longer has to choose between
// "ignore suffix" (v1 caveat) and "extra wire RTT for the merge".
//
// K_suf, V_suf layout: [H_kv, S_suf, D] float32. S_suf can be 0 (no
// suffix); in that case the kernel reduces to sparse_auto.
//
// Returns 0 on success. Same error codes as sparse_auto, plus the FFI
// staging buffer growths for K_suf/V_suf.
int32_t pion_metal_sdpa_query_sparse_auto_fused(uint32_t worker_id,
                                                const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                                uint32_t H_q, uint32_t D, uint32_t precision, uint32_t window,
                                                uint32_t B, uint32_t K_top,
                                                uint32_t H_kv, uint32_t S_suf,
                                                const float *Q,
                                                const uint8_t *head_map,
                                                const float *K_suf,
                                                const float *V_suf,
                                                float *out,
                                                uint32_t selector_id) {       // W11 Phase 2: 0=block-mean, 1=quest
    if (!g_sdpa.ready) return -1;
    if (B == 0u || K_top == 0u) return -5;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    {
        int _rc = _sdpa_resolve_slot(w, key, &slot);
        if (_rc != SDPA_RC_OK) return _rc;  // gh #67: -1 hard miss, -10 cold miss
    }

    PionSDPASlot *s = &w->slots[slot];
    if (H_kv == 0u) H_kv = H_q;
    if (s->H != H_kv || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V
    if (H_q < H_kv || (H_q % H_kv) != 0u) return -7;
    uint32_t rep = H_q / H_kv;

    // W11 Phase 2: selector dispatch — block-mean (0) uses K_mean,
    // Quest (1) uses K_min/K_max. Cached identically.
    uint32_t n_blocks = 0u;
    if (selector_id == 1u) {
        if (_sdpa_ensure_kminmax(s, B, NULL) != 0) return -4;
        n_blocks = s->K_minmax_n_blocks;
    } else {
        if (_sdpa_ensure_kmean(s, B, NULL) != 0) return -4;
        n_blocks = s->K_mean_n_blocks;
    }
    if (K_top > n_blocks) K_top = n_blocks;

    @autoreleasepool {
        size_t q_bytes   = (size_t)H_q * D * sizeof(float);
        size_t hm_bytes  = (size_t)H_q;
        size_t ks_bytes  = (size_t)H_kv * S_suf * D * sizeof(float);
        if (q_bytes > w->Q_capacity) return -4;

        if (w->HM_capacity < hm_bytes) {
            size_t cap = (hm_bytes + 4095u) & ~(size_t)4095u;
            w->HM_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->HM_capacity = cap;
        }
        if (S_suf > 0u) {
            if (w->Ks_capacity < ks_bytes) {
                size_t cap = (ks_bytes + 65535u) & ~(size_t)65535u;
                w->Ks_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
                w->Ks_capacity = cap;
            }
            if (w->Vs_capacity < ks_bytes) {
                size_t cap = (ks_bytes + 65535u) & ~(size_t)65535u;
                w->Vs_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
                w->Vs_capacity = cap;
            }
        }

        uint8_t *hm_dst = (uint8_t*)[w->HM_buf contents];
        if (head_map) {
            memcpy(hm_dst, head_map, hm_bytes);
        } else {
            for (uint32_t h = 0u; h < H_q; h++) hm_dst[h] = (uint8_t)(h / rep);
        }

        // Same top-K selection logic as pion_metal_sdpa_query_sparse_auto;
        // selector_id picks block-mean (0) or Quest UB (1).
        float *scores = (float*)alloca((size_t)n_blocks * sizeof(float));
        const float inv_sqrt_D = 1.0f / sqrtf((float)D);
        for (uint32_t b = 0u; b < n_blocks; b++) scores[b] = 0.0f;
        if (selector_id == 1u) {
            for (uint32_t h = 0u; h < H_q; h++) {
                uint32_t h_kv2 = hm_dst[h];
                const float *Qh = Q + (size_t)h * D;
                const float *Nh = s->K_min + (size_t)h_kv2 * n_blocks * D;
                const float *Xh = s->K_max + (size_t)h_kv2 * n_blocks * D;
                for (uint32_t b = 0u; b < n_blocks; b++) {
                    const float *Nb = Nh + (size_t)b * D;
                    const float *Xb = Xh + (size_t)b * D;
                    float ub = 0.0f;
                    for (uint32_t d = 0u; d < D; d++) {
                        float qd = Qh[d];
                        float a  = qd * Nb[d];
                        float c  = qd * Xb[d];
                        ub += (a > c) ? a : c;
                    }
                    scores[b] += ub * inv_sqrt_D;
                }
            }
        } else {
            for (uint32_t h = 0u; h < H_q; h++) {
                uint32_t h_kv2 = hm_dst[h];
                const float *Qh = Q + (size_t)h * D;
                const float *Mh = s->K_mean + (size_t)h_kv2 * n_blocks * D;
                for (uint32_t b = 0u; b < n_blocks; b++) {
                    const float *Mb = Mh + (size_t)b * D;
                    float dot = 0.0f;
                    for (uint32_t d = 0u; d < D; d++) dot += Qh[d] * Mb[d];
                    scores[b] += dot * inv_sqrt_D;
                }
            }
        }
        const float inv_H = 1.0f / (float)H_q;
        for (uint32_t b = 0u; b < n_blocks; b++) scores[b] *= inv_H;

        uint32_t *top_blocks = (uint32_t*)alloca((size_t)K_top * sizeof(uint32_t));
        _topk_argsort_desc(scores, n_blocks, K_top, top_blocks);

        uint32_t K_full = K_top * B;
        uint32_t last_blk_start = (n_blocks > 0u) ? (n_blocks - 1u) * B : 0u;
        uint32_t last_blk_n     = (s->N > last_blk_start) ? (s->N - last_blk_start) : 0u;
        int last_already = 0;
        if (n_blocks > 0u) {
            for (uint32_t i = 0u; i < K_top; i++) {
                if (top_blocks[i] == n_blocks - 1u) { last_already = 1; break; }
            }
        }
        uint32_t recency_n = (last_already || last_blk_n == 0u) ? 0u : last_blk_n;
        uint32_t K_sparse_max = K_full + recency_n;
        if (K_sparse_max == 0u) return -5;

        size_t idx_bytes = (size_t)H_q * K_sparse_max * sizeof(int32_t);
        size_t cnt_bytes = (size_t)H_q * sizeof(uint32_t);
        if (w->Idx_capacity < idx_bytes) {
            size_t cap = (idx_bytes + 65535u) & ~(size_t)65535u;
            w->Idx_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Idx_capacity = cap;
        }
        if (w->Cnt_capacity < cnt_bytes) {
            size_t cap = (cnt_bytes + 4095u) & ~(size_t)4095u;
            w->Cnt_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Cnt_capacity = cap;
        }

        int32_t *idx_dst = (int32_t*)[w->Idx_buf contents];
        int32_t *row = idx_dst;
        uint32_t actual_count = 0u;
        for (uint32_t i = 0u; i < K_top; i++) {
            uint32_t blk = top_blocks[i];
            uint32_t t_begin = blk * B;
            uint32_t t_end   = t_begin + B;
            if (t_end > s->N) t_end = s->N;
            for (uint32_t t = t_begin; t < t_end; t++) {
                row[actual_count++] = (int32_t)t;
            }
        }
        if (recency_n > 0u) {
            for (uint32_t t = last_blk_start; t < last_blk_start + recency_n; t++) {
                row[actual_count++] = (int32_t)t;
            }
        }
        for (uint32_t h = 1u; h < H_q; h++) {
            memcpy(row + (size_t)h * K_sparse_max, row, (size_t)K_sparse_max * sizeof(int32_t));
        }
        uint32_t *cnt_dst = (uint32_t*)[w->Cnt_buf contents];
        for (uint32_t h = 0u; h < H_q; h++) cnt_dst[h] = actual_count;

        memcpy([w->Q_buf contents], Q, q_bytes);
        if (S_suf > 0u) {
            memcpy([w->Ks_buf contents], K_suf, ks_bytes);
            memcpy([w->Vs_buf contents], V_suf, ks_bytes);
        }
        *(uint32_t*)[w->N_buf  contents]  = s->N;
        *(float*)   [w->scale_buf contents] = inv_sqrt_D;
        *(uint32_t*)[w->W_buf  contents]  = window;
        *(uint32_t*)[w->Ksp_buf contents] = K_sparse_max;
        *(uint32_t*)[w->M_buf  contents]  = H_kv;
        *(uint32_t*)[w->Ssuf_buf contents] = S_suf;

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        id<MTLComputePipelineState> pso = (precision == 1u)
            ? g_sdpa.psos_sparse_fused_fp16[pso_idx]
            : g_sdpa.psos_sparse_fused[pso_idx];
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Q_buf     offset:0 atIndex:0];
        [ce setBuffer:s->K_buf     offset:0 atIndex:1];
        [ce setBuffer:s->V_buf     offset:0 atIndex:2];
        [ce setBuffer:w->O_buf     offset:0 atIndex:3];
        [ce setBuffer:w->N_buf     offset:0 atIndex:4];
        [ce setBuffer:w->scale_buf offset:0 atIndex:5];
        [ce setBuffer:w->W_buf     offset:0 atIndex:6];
        [ce setBuffer:w->Idx_buf   offset:0 atIndex:7];
        [ce setBuffer:w->Cnt_buf   offset:0 atIndex:8];
        [ce setBuffer:w->Ksp_buf   offset:0 atIndex:9];
        [ce setBuffer:w->HM_buf    offset:0 atIndex:10];
        [ce setBuffer:w->M_buf     offset:0 atIndex:11];
        // K_suf / V_suf — if S_suf=0, bind any non-null buffer (kernel guards
        // on S_suf==0 and never reads). Reuse Ks/Vs (allocated above if S>0)
        // or the Q buffer as a stand-in when S_suf==0 (kernel only references
        // the bind, doesn't actually touch the bytes).
        id<MTLBuffer> ks_bind = (S_suf > 0u && w->Ks_buf) ? w->Ks_buf : w->Q_buf;
        id<MTLBuffer> vs_bind = (S_suf > 0u && w->Vs_buf) ? w->Vs_buf : w->Q_buf;
        [ce setBuffer:ks_bind      offset:0 atIndex:12];
        [ce setBuffer:vs_bind      offset:0 atIndex:13];
        [ce setBuffer:w->Ssuf_buf  offset:0 atIndex:14];
        [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
        [ce dispatchThreadgroups:MTLSizeMake(H_q, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->O_buf contents], q_bytes);
    }
    return 0;
}

// Phase C: M>1 batched-Q. Same K/V cache slot lookup as single-Q; uses
// sdpa_batched_q_fp32 kernel with H × M threadgroups. Per-worker staging.
//
// Returns H*M*D float32 in `out` and H*M float32 in `lse` (= max + log(sum_exp))
// for online-softmax merge on the client side. Caller must pass non-null lse
// for M>1 — the wire format documented in src/commands/attend_prefix.mojo
// always carries the LSE trailer.
int32_t pion_metal_sdpa_query_batched(uint32_t worker_id,
                                      const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                      uint32_t H, uint32_t M, uint32_t D, uint32_t precision, uint32_t window,
                                      const float *Q, float *out, float *lse) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    if (M < 1u) return -7;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    {
        int _rc = _sdpa_resolve_slot(w, key, &slot);
        if (_rc != SDPA_RC_OK) return _rc;  // gh #67: -1 hard miss, -10 cold miss
    }
    PionSDPASlot *s = &w->slots[slot];
    if (s->H != H || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V

    @autoreleasepool {
        size_t q_bytes  = (size_t)H * M * D * sizeof(float);
        size_t o_bytes  = q_bytes;
        size_t lse_bytes = (size_t)H * M * sizeof(float);
        // Grow staging buffers per-worker as needed (rounded up to a 64KB stride).
        if (w->Qb_capacity < q_bytes) {
            size_t cap = (q_bytes + 65535u) & ~(size_t)65535u;
            w->Qb_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Qb_capacity = cap;
        }
        if (w->Ob_capacity < o_bytes) {
            size_t cap = (o_bytes + 65535u) & ~(size_t)65535u;
            w->Ob_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Ob_capacity = cap;
        }
        if (w->LSE_capacity < lse_bytes) {
            size_t cap = (lse_bytes + 4095u) & ~(size_t)4095u;
            w->LSE_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->LSE_capacity = cap;
        }

        memcpy([w->Qb_buf contents], Q, q_bytes);
        *(uint32_t*)[w->N_buf contents]     = s->N;
        *(float*)   [w->scale_buf contents] = 1.0f / sqrtf((float)D);
        *(uint32_t*)[w->M_buf contents]     = M;
        *(uint32_t*)[w->W_buf contents]     = window;

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        // gh #130 §4.2: pick the tiled (K/V-staged) fp32 kernel only at large M,
        // where its K/V reuse outweighs the reduced token parallelism; below the
        // crossover the split kernel (max token parallelism per query) is faster.
        // fp16 stays on the split kernel until it is ported (increment 1b).
        bool use_tiled = (precision != 1u) && (M >= SDPA_BATCHED_TILE_MIN_M);
        id<MTLComputePipelineState> pso = use_tiled
            ? g_sdpa.psos_batched_tiled[pso_idx]
            : ((precision == 1u) ? g_sdpa.psos_batched_fp16[pso_idx]
                                 : g_sdpa.psos_batched[pso_idx]);
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Qb_buf  offset:0 atIndex:0];
        [ce setBuffer:s->K_buf   offset:0 atIndex:1];
        [ce setBuffer:s->V_buf   offset:0 atIndex:2];
        [ce setBuffer:w->Ob_buf  offset:0 atIndex:3];
        [ce setBuffer:w->LSE_buf offset:0 atIndex:4];
        [ce setBuffer:w->N_buf   offset:0 atIndex:5];
        [ce setBuffer:w->scale_buf offset:0 atIndex:6];
        [ce setBuffer:w->M_buf   offset:0 atIndex:7];
        [ce setBuffer:w->W_buf   offset:0 atIndex:8];
        if (use_tiled) {
            [ce setThreadgroupMemoryLength:_sdpa_batched_tile_bytes(D, precision) atIndex:0];
            [ce dispatchThreadgroups:MTLSizeMake(H, (M + 7u) / 8u, 1)   // SG_PER_TG=8 query rows/TG
                threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        } else {
            [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
            [ce dispatchThreadgroups:MTLSizeMake(H, M, 1)              // 1 query row/TG
                threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        }
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->Ob_buf  contents], o_bytes);
        memcpy(lse, [w->LSE_buf contents], lse_bytes);
    }
    return 0;
}

// gh #49: server-side fused suffix-SDPA + prefix-merge. One online softmax
// over (resident prefix K/V) ∪ (caller-supplied suffix K/V), eliminating
// the host-side _suffix_sdpa_with_lse + _online_softmax_merge path in
// pion-vllm-mlx/pion_vllm_mlx/mlx_lm_patch.py.
//
// Returns H_q*M*D fp32 in `out` (merged attention output). No LSE on the
// wire (merge already happened). H_q can differ from cached H (GQA): the
// `head_map[H_q] = h_kv` table tells the kernel which kv-head each q-head
// reads from.
int32_t pion_metal_sdpa_query_batched_fused(uint32_t worker_id,
                                            const char *session_id, uint32_t sid_len, uint32_t layer_id,
                                            uint32_t H_q, uint32_t M, uint32_t D, uint32_t precision, uint32_t window,
                                            uint32_t H_kv, uint32_t S_suf,
                                            const float *Q,
                                            const float *K_suf, const float *V_suf,
                                            const uint8_t *head_map,
                                            float *out) {
    if (!g_sdpa.ready) return -1;
    PionSDPAWorker *w = _sdpa_worker(worker_id);
    if (!w) return -8;
    if (M < 1u || H_q == 0u || H_kv == 0u || (H_q % H_kv) != 0u) return -7;
    int pso_idx = _sdpa_pso_index(D);
    if (pso_idx < 0) return -2;
    uint64_t key = _sdpa_key(session_id, sid_len, layer_id);
    int slot = -1;
    {
        int _rc = _sdpa_resolve_slot(w, key, &slot);
        if (_rc != SDPA_RC_OK) return _rc;  // gh #67: -1 hard miss, -10 cold miss
    }
    PionSDPASlot *s = &w->slots[slot];
    if (s->H != H_kv || s->D != D) return -3;
    if ((precision == 1u) != (s->kv_half != 0)) return -3;  // gh #398: kernel type must match the stored K/V

    @autoreleasepool {
        size_t q_bytes  = (size_t)H_q  * M * D * sizeof(float);
        size_t o_bytes  = q_bytes;
        size_t ks_bytes = (size_t)H_kv * S_suf * D * sizeof(float);
        size_t vs_bytes = ks_bytes;
        size_t hm_bytes = (size_t)H_q;
        // Grow staging buffers per-worker. Q/O reuse the existing batched paths' staging.
        if (w->Qb_capacity < q_bytes) {
            size_t cap = (q_bytes + 65535u) & ~(size_t)65535u;
            w->Qb_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Qb_capacity = cap;
        }
        if (w->Ob_capacity < o_bytes) {
            size_t cap = (o_bytes + 65535u) & ~(size_t)65535u;
            w->Ob_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->Ob_capacity = cap;
        }
        if (S_suf > 0u) {
            if (w->Ks_capacity < ks_bytes) {
                size_t cap = (ks_bytes + 65535u) & ~(size_t)65535u;
                w->Ks_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
                w->Ks_capacity = cap;
            }
            if (w->Vs_capacity < vs_bytes) {
                size_t cap = (vs_bytes + 65535u) & ~(size_t)65535u;
                w->Vs_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
                w->Vs_capacity = cap;
            }
        } else {
            // Kernel still requires bound buffers (no MTL_NULL). Use a 4-byte
            // placeholder if we've never allocated (the kernel won't read it
            // because S_suf=0 short-circuits the suffix loop).
            if (!w->Ks_buf) {
                w->Ks_buf = [g_sdpa.device newBufferWithLength:16u options:MTLResourceStorageModeShared];
                w->Ks_capacity = 16u;
            }
            if (!w->Vs_buf) {
                w->Vs_buf = [g_sdpa.device newBufferWithLength:16u options:MTLResourceStorageModeShared];
                w->Vs_capacity = 16u;
            }
        }
        if (w->HM_capacity < hm_bytes) {
            size_t cap = (hm_bytes + 63u) & ~(size_t)63u;
            if (cap < 64u) cap = 64u;
            w->HM_buf = [g_sdpa.device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            w->HM_capacity = cap;
        }

        memcpy([w->Qb_buf contents], Q, q_bytes);
        if (S_suf > 0u) {
            memcpy([w->Ks_buf contents], K_suf, ks_bytes);
            memcpy([w->Vs_buf contents], V_suf, vs_bytes);
        }
        memcpy([w->HM_buf contents], head_map, hm_bytes);
        *(uint32_t*)[w->N_buf contents]    = s->N;
        *(uint32_t*)[w->Ssuf_buf contents] = S_suf;
        *(float*)   [w->scale_buf contents] = 1.0f / sqrtf((float)D);
        *(uint32_t*)[w->M_buf contents]    = M;
        *(uint32_t*)[w->W_buf contents]    = window;

        id<MTLCommandBuffer> cb = [g_sdpa.queue commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        // gh #130 §4.2: tiled fused kernel (K/V staged in threadgroup mem) at large
        // M; split fused kernel below the crossover. fp16 stays split (port pending).
        bool use_tiled = (precision != 1u) && (M >= SDPA_BATCHED_TILE_MIN_M);
        id<MTLComputePipelineState> pso = use_tiled
            ? g_sdpa.psos_fused_tiled[pso_idx]
            : ((precision == 1u) ? g_sdpa.psos_fused_fp16[pso_idx]
                                 : g_sdpa.psos_fused[pso_idx]);
        [ce setComputePipelineState:pso];
        [ce setBuffer:w->Qb_buf    offset:0 atIndex:0];
        [ce setBuffer:s->K_buf     offset:0 atIndex:1];
        [ce setBuffer:s->V_buf     offset:0 atIndex:2];
        [ce setBuffer:w->Ks_buf    offset:0 atIndex:3];
        [ce setBuffer:w->Vs_buf    offset:0 atIndex:4];
        [ce setBuffer:w->Ob_buf    offset:0 atIndex:5];
        [ce setBuffer:w->HM_buf    offset:0 atIndex:6];
        [ce setBuffer:w->N_buf     offset:0 atIndex:7];
        [ce setBuffer:w->Ssuf_buf  offset:0 atIndex:8];
        [ce setBuffer:w->scale_buf offset:0 atIndex:9];
        [ce setBuffer:w->M_buf     offset:0 atIndex:10];
        [ce setBuffer:w->W_buf     offset:0 atIndex:11];
        if (use_tiled) {
            [ce setThreadgroupMemoryLength:_sdpa_batched_tile_bytes(D, precision) atIndex:0];
            [ce dispatchThreadgroups:MTLSizeMake(H_q, (M + 7u) / 8u, 1)   // SG_PER_TG=8 rows/TG
                threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        } else {
            [ce setThreadgroupMemoryLength:_sdpa_so_bytes(D, precision) atIndex:0];
            [ce dispatchThreadgroups:MTLSizeMake(H_q, M, 1)
                threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        }
        [ce endEncoding];

        uint64_t target = atomic_fetch_add_explicit(&g_sdpa.event_counter, 1, memory_order_relaxed) + 1;
        [cb encodeSignalEvent:g_sdpa.event value:target];
        [cb commit];
        while ([g_sdpa.event signaledValue] < target) { /* spin */ }

        memcpy(out, [w->Ob_buf contents], o_bytes);
    }
    return 0;
}

int32_t pion_metal_sdpa_available(void) {
    return g_sdpa.ready ? 1 : 0;
}

void pion_metal_sdpa_shutdown(void) {
    if (!g_sdpa.ready) return;
    g_sdpa.ready = 0;
    for (int wi = 0; wi < SDPA_MAX_WORKERS; wi++) {
        memset(g_sdpa.workers[wi].slots, 0, sizeof(g_sdpa.workers[wi].slots));
    }
    // ARC reaps the rest.
    fprintf(stderr, "[Metal SDPA] shut down\n");
}
