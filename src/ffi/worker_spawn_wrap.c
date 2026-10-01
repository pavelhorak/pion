/* worker_spawn_wrap.c — Mojo 1.0 migration: std.algorithm.parallelize moved to
 * the `max` package, which the server build must not depend on. Workers are
 * spawned as raw pthreads instead, calling back into the Mojo @export
 * pion_worker_entry(ctx, idx) resolved via dlsym (the gh #199 build-lane
 * pattern — see build_pool_wrap.c; the executable links -export_dynamic and
 * pins the symbol with -u so a regression is a loud link error).
 *
 * ctx is an Int64 slot array packed by main() (see main.mojo for the layout).
 * main() blocks in here on pthread_join — the workers run their event loops
 * forever, matching the old parallelize[worker_task](n, n) never-returns
 * behavior — so everything ctx points at outlives the workers.
 */
#include <pthread.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*pion_worker_fn)(int64_t* ctx, int64_t idx);

struct _pion_worker_arg {
    pion_worker_fn fn;
    int64_t* ctx;
    int64_t idx;
};

static void* _pion_worker_trampoline(void* p) {
    struct _pion_worker_arg* a = (struct _pion_worker_arg*)p;
    a->fn(a->ctx, a->idx);
    /* Workers never return in normal operation — this line firing means the
     * worker died (raised out of its body or exited early). Unbuffered C
     * stderr on purpose: a Mojo print from a dying pthread can be lost in a
     * buffered stdout, which made worker deaths invisible during the Mojo
     * 1.0 migration (only the gh #138 status file hinted). */
    fprintf(stderr, "[pion] worker %lld DIED (event loop returned)\n", (long long)a->idx);
    return NULL;
}

int32_t pion_spawn_workers(int32_t n, int64_t* ctx) {
    pion_worker_fn fn = (pion_worker_fn)dlsym(RTLD_DEFAULT, "pion_worker_entry");
    if (!fn) {
        fprintf(stderr, "pion_spawn_workers: dlsym(pion_worker_entry) failed\n");
        return -1;
    }
    pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)n);
    struct _pion_worker_arg* args =
        (struct _pion_worker_arg*)malloc(sizeof(struct _pion_worker_arg) * (size_t)n);
    if (!tids || !args) {
        fprintf(stderr, "pion_spawn_workers: malloc failed\n");
        free(tids); free(args);
        return -1;
    }
    /* 16 MB stacks: the old parallelize ran worker 0 on the main thread
     * (8 MB) and pool workers on ~512 KB stacks — the direct cause of the
     * historic dev-build (-O0 fat frames) SIGBUS at -w >= 2. Explicit stacks
     * fix all workers, including -O0 ones. */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
    for (int32_t i = 0; i < n; i++) {
        args[i].fn = fn;
        args[i].ctx = ctx;
        args[i].idx = (int64_t)i;
        if (pthread_create(&tids[i], &attr, _pion_worker_trampoline, &args[i]) != 0) {
            fprintf(stderr, "pion_spawn_workers: pthread_create(%d) failed\n", i);
            args[i].idx = -1; /* mark: do not join */
        }
    }
    pthread_attr_destroy(&attr);
    for (int32_t i = 0; i < n; i++) {
        if (args[i].idx >= 0) pthread_join(tids[i], NULL);
    }
    free(args);
    free(tids);
    return 0;
}
