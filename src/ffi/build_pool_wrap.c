/* gh #199: generic N-thread fork/join pool for parallel HNSW builds.
 *
 * Why this exists: Mojo's `parallelize` pool is permanently occupied by the
 * server's event-loop workers (one per worker, capped at 4 threads on Apple
 * Silicon), so a nested `parallelize` inside an FT.OPTIMIZE handler deadlocks
 * at -w N (measured 2026-08-07, gh #199 probe). Dedicated pthreads sidestep
 * the pool entirely — same pattern as moe_warm_pool.c / uring_wrap.c threads,
 * except the workers call BACK INTO a Mojo function.
 *
 * Contract: pion_build_pool_run(fn, ctx, n_threads) spawns n_threads-1
 * pthreads and runs one chunk on the calling thread (so a 4-way build costs 3
 * spawns), each invoking fn(ctx, thread_idx) exactly once with thread_idx in
 * [0, n_threads). Joins all before returning. Returns 0 on success, -1 if any
 * pthread_create failed (in which case the failed lanes are run serially on
 * the caller so every thread_idx still executes — the build must not silently
 * skip a stride).
 *
 * The Mojo callback must be a non-raising `fn` with C ABI, must not call
 * parallelize/print, and must confine itself to thread-safe runtime surface
 * (malloc/free are). That is exactly the shape of the gh #199 link-phase
 * worker.
 */
#include <pthread.h>
#include <stdint.h>
#include <dlfcn.h>

typedef void (*pion_build_fn)(void *ctx, int64_t thread_idx);

typedef struct {
    pion_build_fn fn;
    void *ctx;
    int64_t idx;
} _pion_build_lane;

static void *_pion_build_trampoline(void *arg) {
    _pion_build_lane *lane = (_pion_build_lane *)arg;
    lane->fn(lane->ctx, lane->idx);
    return NULL;
}

int pion_build_pool_run(pion_build_fn fn, void *ctx, int64_t n_threads) {
    if (n_threads <= 1) {
        fn(ctx, 0);
        return 0;
    }
    enum { MAX_LANES = 64 };
    if (n_threads > MAX_LANES) n_threads = MAX_LANES;
    _pion_build_lane lanes[MAX_LANES];
    pthread_t tids[MAX_LANES];
    unsigned char spawned[MAX_LANES] = {0};
    int rc = 0;
    for (int64_t i = 1; i < n_threads; i++) {
        lanes[i].fn = fn;
        lanes[i].ctx = ctx;
        lanes[i].idx = i;
        if (pthread_create(&tids[i], NULL, _pion_build_trampoline, &lanes[i]) == 0) {
            spawned[i] = 1;
        } else {
            rc = -1; /* run this lane serially below — never skip a stride */
        }
    }
    fn(ctx, 0); /* caller runs lane 0 */
    for (int64_t i = 1; i < n_threads; i++) {
        if (spawned[i]) {
            pthread_join(tids[i], NULL);
        } else {
            fn(ctx, i);
        }
    }
    return rc;
}

/* Mojo 1.0.0b2 removed `fn` (and with it clean function-pointer passing to C),
 * so the Mojo side @exports its lane worker as a C symbol and we resolve it by
 * name here. Returns -2 if the symbol is missing. */
int pion_build_pool_run_sym(const char *sym, void *ctx, int64_t n_threads) {
    pion_build_fn fn = (pion_build_fn)dlsym(RTLD_DEFAULT, sym);
    if (!fn) return -2;
    return pion_build_pool_run(fn, ctx, n_threads);
}

/* gh #400: y = A·x for a row-major [m × n] FP32 matrix, the exact scan behind
 * VSIM. On macOS this is Accelerate's cblas_sgemv, found once with dlopen so
 * that no build line has to link the framework; measured ~5× the shipped
 * one-row-at-a-time loop on a contiguous set. Returns 0 when it ran, -1 when
 * no BLAS is available (Linux, or a dlopen failure); the caller then runs its
 * portable kernel. The resolve is idempotent, so a race between two workers
 * on the first call stores the same pointer twice. */
typedef void (*_pion_sgemv_t)(int order, int trans, int m, int n, float alpha,
                              const float *a, int lda, const float *x, int incx,
                              float beta, float *y, int incy);
static _pion_sgemv_t _pion_sgemv_fn = 0;
static int _pion_sgemv_state = 0; /* 0 unresolved, 1 resolved, -1 unavailable */

int pion_sgemv_f32(int64_t m, int64_t n, const float *a, const float *x, float *y) {
#if defined(__APPLE__)
    if (_pion_sgemv_state == 0) {
        void *h = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                         RTLD_LAZY | RTLD_LOCAL);
        _pion_sgemv_t f = h ? (_pion_sgemv_t)dlsym(h, "cblas_sgemv") : 0;
        _pion_sgemv_fn = f;
        _pion_sgemv_state = f ? 1 : -1;
    }
    if (_pion_sgemv_state != 1 || m <= 0 || n <= 0 || m > 0x7fffffff || n > 0x7fffffff)
        return -1;
    /* 101 = CblasRowMajor, 111 = CblasNoTrans */
    _pion_sgemv_fn(101, 111, (int)m, (int)n, 1.0f, a, (int)n, x, 1, 0.0f, y, 1);
    return 0;
#else
    (void)m; (void)n; (void)a; (void)x; (void)y;
    return -1;
#endif
}
