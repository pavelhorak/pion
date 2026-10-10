/* worker_spawn_wrap.c — Mojo 1.0 migration: std.algorithm.parallelize moved to
 * the `max` package, which the server build must not depend on. Workers are
 * spawned as raw pthreads instead, calling back into the Mojo @export
 * pion_worker_entry(ctx, idx) resolved via dlsym (the gh #199 build-lane
 * pattern — see build_pool_wrap.c; the executable links -export_dynamic and
 * pins the symbol with -u so a regression is a loud link error).
 *
 * ctx is an Int64 slot array packed by main() (see main.mojo for the layout).
 * main() blocks in here on pthread_join — the workers run their event loops
 * until a shutdown drain (gh #259) returns them — so everything ctx points at
 * outlives the workers.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* cpu_set_t, pthread_setaffinity_np */
#endif
#include <pthread.h>
#include <sched.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

typedef void (*pion_worker_fn)(int64_t* ctx, int64_t idx);

/* crash_wrap.c: every link line that takes this file takes that one too. */
int pion_shutdown_requested(void);

struct _pion_worker_arg {
    pion_worker_fn fn;
    int64_t* ctx;
    int64_t idx;
};

static void* _pion_worker_trampoline(void* p) {
    struct _pion_worker_arg* a = (struct _pion_worker_arg*)p;
    a->fn(a->ctx, a->idx);
    /* A worker returns on purpose only after SIGTERM, SIGINT or SHUTDOWN: its
     * event loop flushes the WAL and returns (gh #259). That is how every
     * supervisor stops the server, so it is reported as a stop. Before this
     * check, each `brew services stop pion` wrote a worker death into the
     * service log. Any other return means the worker died (raised out of its
     * body or exited early). Unbuffered C stderr on purpose: a Mojo print from
     * a dying pthread can be lost in a buffered stdout, which made worker
     * deaths invisible during the Mojo 1.0 migration (only the gh #138 status
     * file hinted). */
    if (pion_shutdown_requested())
        fprintf(stderr, "[pion] worker %lld stopped (shutdown)\n", (long long)a->idx);
    else
        fprintf(stderr, "[pion] worker %lld DIED (event loop returned)\n", (long long)a->idx);
    return NULL;
}

/* `--affinity` on Linux: pin the calling worker thread to one CPU. It picks
 * the (cpu mod N)-th CPU of the thread's CURRENT mask, not CPU number `cpu`,
 * so a container or taskset that hands the process CPUs 4-7 still pins inside
 * them. Returns the CPU pinned to, or -1 when nothing was applied. Before
 * #20 the flag was a no-op on Linux that still logged "pinned to CPU i". */
int32_t pion_pin_current_thread(int32_t cpu) {
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return -1;
    int n = CPU_COUNT(&allowed);
    if (n <= 0 || cpu < 0) return -1;
    int want = cpu % n;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &allowed)) continue;
        if (want-- > 0) continue;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(c, &one);
        return pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0 ? c : -1;
    }
    return -1;
#else
    (void)cpu;
    return -1;
#endif
}

/* Raise the open-file soft limit toward `want`, as Redis does at startup. A
 * stock Linux login has a soft limit of 1024, which capped a server at about a
 * thousand clients while every per-fd table holds 65536. Bounded by the hard
 * limit, and on macOS by kern.maxfilesperproc (setrlimit answers EINVAL above
 * it), so it steps down until one value is accepted. Returns the soft limit in
 * force afterwards; *before receives the one found. */
int64_t pion_raise_nofile(int64_t want, int64_t* before) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return -1;
    if (before) *before = (int64_t)rl.rlim_cur;
    if (rl.rlim_cur != RLIM_INFINITY && (int64_t)rl.rlim_cur >= want) return (int64_t)rl.rlim_cur;
    if (rl.rlim_cur == RLIM_INFINITY) return want;
    rlim_t target = (rlim_t)want;
    if (rl.rlim_max != RLIM_INFINITY && target > rl.rlim_max) target = rl.rlim_max;
    while (target > rl.rlim_cur) {
        struct rlimit nr = { target, rl.rlim_max };
        if (setrlimit(RLIMIT_NOFILE, &nr) == 0) return (int64_t)target;
        target -= target / 4 + 1;
    }
    return (int64_t)rl.rlim_cur;
}

/* #465: the I/O threads of `--io-threads`. Detached: they serve until the
 * process exits. Each runs pion_worker_entry(ctx, base + i); the entry tells
 * an I/O thread from a worker by base (IO_THREAD_BASE in io_threads.mojo), so
 * no second exported symbol (and no new -u link flag) is needed. */
static void* _pion_io_trampoline(void* p) {
    struct _pion_worker_arg* a = (struct _pion_worker_arg*)p;
    a->fn(a->ctx, a->idx);
    fprintf(stderr, "[pion] I/O thread %lld returned\n", (long long)a->idx);
    return NULL;
}

/* #465: a connection's output queue is written by the executor and drained by
 * the I/O thread that owns the socket; this word guards it. Held for an append
 * or for one send loop, never across anything that can block for long. */
void pion_spin_lock(uint32_t* w) {
    while (__atomic_exchange_n(w, 1u, __ATOMIC_ACQUIRE) != 0u) {
        while (__atomic_load_n(w, __ATOMIC_RELAXED) != 0u) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield");
#endif
        }
    }
}

void pion_spin_unlock(uint32_t* w) {
    __atomic_store_n(w, 0u, __ATOMIC_RELEASE);
}

/* #465: a full fence, for the sleeper's side of the wake-up handshake: store
 * "I am sleeping", FENCE, then look at the ring once more. A sequentially
 * consistent store followed by an acquire load is not enough on ARM64, where
 * LLVM lowers the load to LDAPR (RCpc), which may be satisfied before the
 * earlier STLR is visible: the waker then reads "not sleeping" while the
 * sleeper read "ring empty", and the wake-up is lost until the poll times out. */
void pion_full_fence(void) {
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

int32_t pion_spawn_detached(int32_t n, int64_t* ctx, int64_t base) {
    pion_worker_fn fn = (pion_worker_fn)dlsym(RTLD_DEFAULT, "pion_worker_entry");
    if (!fn || n <= 0) return -1;
    struct _pion_worker_arg* args =
        (struct _pion_worker_arg*)malloc(sizeof(struct _pion_worker_arg) * (size_t)n);
    if (!args) return -1;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int32_t rc = 0;
    for (int32_t i = 0; i < n; i++) {
        pthread_t tid;
        args[i].fn = fn;
        args[i].ctx = ctx;
        args[i].idx = base + (int64_t)i;
        if (pthread_create(&tid, &attr, _pion_io_trampoline, &args[i]) != 0) rc = -1;
    }
    pthread_attr_destroy(&attr);
    return rc;   /* args stays allocated: the threads read it for their lifetime */
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
