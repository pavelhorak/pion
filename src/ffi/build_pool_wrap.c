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
