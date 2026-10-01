/* moe_warm_pool.c — Stage 4b background-warming I/O scaffold.
 *
 * Status: SCAFFOLD ONLY — single warming pthread + bounded SPSC request
 * and completion rings + worker function skeleton. Build is gated out
 * of pixi until Stage 4b-1 wires the Mojo side.
 *
 * Why a scaffold first: the threading primitives are stable across
 * platforms (pthread + condvar) and the API surface is small enough
 * to lock down before Mojo integration. The pread / shard-open / blob-
 * assembly inner loop is intentionally left as a TODO: it should reuse
 * the same per-(proj, comp) iteration that
 * `handle_moe_expert_fetch` in src/commands/moe_expert.mojo runs today,
 * just on the warming thread. Factoring that out into a C helper is
 * Stage 4b-2.
 */

#include "moe_warm_pool.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Stage 4c: Linux io_uring batched-pread variant. Opt-in at compile time
   via -DPION_MOE_WARM_USE_IO_URING -luring. macOS and default-Linux builds
   keep the synchronous pread loop.

   The opt-in build collapses ~11 syscalls per warm cycle (1 open + 9 pread
   + 1 close per expert) to ~2 (one io_uring_submit + one cqe_wait) by
   submitting all 9 preads as a single batched ring entry. Most useful on
   high-FETCH workloads where syscall overhead becomes the bottleneck. */
#if defined(__linux__) && defined(PION_MOE_WARM_USE_IO_URING)
#  include <liburing.h>
#  define PION_MOE_WARM_HAVE_IO_URING 1
#else
#  define PION_MOE_WARM_HAVE_IO_URING 0
#endif

/* Must mirror MAX_SHARD_PATH_INLINE on the Mojo side
   (src/network/moe_expert_tier.mojo). */
#define MAX_SHARD_PATH_INLINE_C 256

/* ── Request ring ─────────────────────────────────────────────────────
   Bounded SPSC ring: enqueued by main thread, dequeued by worker.
   Capacity is power-of-two so head/tail mask cheaply.

   Important: the enqueue side copies shard_dir + offset table into these
   inline buffers, so callers can stack-allocate or pass non-persistent
   pointers. The original API took pointers (and required caller-side
   lifetime guarantees); copying simplifies the contract at the cost of
   ~340 bytes/slot × 64 slots = 22 KiB of upfront memory. */
#define MOE_WARM_OFFSET_BYTES 180
#define MOE_WARM_SHARD_PATH_MAX 256
typedef struct {
    int32_t  slot;
    int32_t  layer;
    int32_t  expert;
    int      shard_count;
    char     shard_dir[MOE_WARM_SHARD_PATH_MAX];
    uint8_t  offset_table[MOE_WARM_OFFSET_BYTES];
} moe_warm_request_t;

/* Lock + condvar guarded request ring (main producer, worker consumer).
   Using a mutex (not lock-free) for the request side is fine: enqueue
   happens once per PREFETCH (rare relative to FETCH) and the worker's
   blocking wait is well-served by condvar. */
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
    size_t capacity;       /* power of two */
    size_t mask;
    size_t head, tail;     /* head=write, tail=read */
    int    shutting_down;
    moe_warm_request_t* slots;
} moe_warm_request_ring_t;

/* Lock-free SPSC completion ring (worker producer, main consumer).
   atomic head + atomic tail; the consumer reads up to head, writes tail. */
typedef struct {
    size_t capacity;
    size_t mask;
    _Atomic size_t head;   /* writer position */
    _Atomic size_t tail;   /* reader position */
    moe_warm_completion_t* slots;
} moe_warm_completion_ring_t;

struct moe_warm_pool {
    moe_warm_request_ring_t    req;
    moe_warm_completion_ring_t comp;
    pthread_t worker_tid;
    int       worker_started;
    /* Counters */
    _Atomic uint64_t cnt_enqueued;
    _Atomic uint64_t cnt_dropped_full;
    _Atomic uint64_t cnt_completed;
    _Atomic uint64_t cnt_errored;
    _Atomic uint64_t cnt_bytes_warmed;
};

/* ── Helpers ──────────────────────────────────────────────────────── */

static size_t next_pow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static int req_ring_init(moe_warm_request_ring_t* r, size_t cap) {
    cap = next_pow2(cap);
    r->slots = (moe_warm_request_t*)calloc(cap, sizeof(*r->slots));
    if (!r->slots) return -1;
    r->capacity = cap;
    r->mask = cap - 1;
    r->head = 0; r->tail = 0;
    r->shutting_down = 0;
    if (pthread_mutex_init(&r->lock, NULL) != 0) { free(r->slots); return -1; }
    if (pthread_cond_init(&r->not_empty, NULL) != 0) { pthread_mutex_destroy(&r->lock); free(r->slots); return -1; }
    if (pthread_cond_init(&r->not_full, NULL) != 0) { pthread_cond_destroy(&r->not_empty); pthread_mutex_destroy(&r->lock); free(r->slots); return -1; }
    return 0;
}

static void req_ring_destroy(moe_warm_request_ring_t* r) {
    pthread_cond_destroy(&r->not_full);
    pthread_cond_destroy(&r->not_empty);
    pthread_mutex_destroy(&r->lock);
    free(r->slots);
}

static int comp_ring_init(moe_warm_completion_ring_t* r, size_t cap) {
    cap = next_pow2(cap);
    r->slots = (moe_warm_completion_t*)calloc(cap, sizeof(*r->slots));
    if (!r->slots) return -1;
    r->capacity = cap;
    r->mask = cap - 1;
    atomic_store(&r->head, 0);
    atomic_store(&r->tail, 0);
    return 0;
}

static void comp_ring_destroy(moe_warm_completion_ring_t* r) {
    free(r->slots);
}

/* ── Worker thread ──────────────────────────────────────────────────── */

/* Little-endian byte readers — the Mojo enqueue path serialises (shard_id,
   data_start, per_expert_bytes) as a packed 20-byte struct per (proj, comp).
   Layout is fixed regardless of Mojo's native alignment so the C side
   doesn't have to match struct padding. */
static int32_t read_i32le(const uint8_t* p) {
    return (int32_t)((uint32_t)p[0]
                     | ((uint32_t)p[1] << 8)
                     | ((uint32_t)p[2] << 16)
                     | ((uint32_t)p[3] << 24));
}

static uint64_t read_u64le(const uint8_t* p) {
    return  (uint64_t)p[0]
         | ((uint64_t)p[1] <<  8)
         | ((uint64_t)p[2] << 16)
         | ((uint64_t)p[3] << 24)
         | ((uint64_t)p[4] << 32)
         | ((uint64_t)p[5] << 40)
         | ((uint64_t)p[6] << 48)
         | ((uint64_t)p[7] << 56);
}

/* Push a completion onto the SPSC ring. Returns 0 on success, -1 if full. */
static int push_completion(moe_warm_pool_t* p, const moe_warm_completion_t* c) {
    size_t head = atomic_load_explicit(&p->comp.head, memory_order_relaxed);
    size_t tail = atomic_load_explicit(&p->comp.tail, memory_order_acquire);
    if (head - tail >= p->comp.capacity) return -1;
    p->comp.slots[head & p->comp.mask] = *c;
    atomic_store_explicit(&p->comp.head, head + 1, memory_order_release);
    return 0;
}

static void emit_error_completion(moe_warm_pool_t* p,
                                    const moe_warm_request_t* req, int err) {
    moe_warm_completion_t c = {
        .slot     = req->slot,
        .layer    = req->layer,
        .expert   = req->expert,
        .addr     = 0,
        .data_len = 0,
        .status   = err,
    };
    (void)push_completion(p, &c);
    atomic_fetch_add_explicit(&p->cnt_errored, 1, memory_order_relaxed);
}

/* Layout note (must match the Mojo enqueue serialiser):
     offset_table_for_layer is exactly 9 entries × 20 bytes = 180 bytes.
     Entry i corresponds to (proj = i/3, comp = i%3). Each entry is:
       [0..4)   int32_t  shard_id (LE; 0 = absent)
       [4..12)  uint64_t data_start (absolute byte offset in shard)
       [12..20) uint64_t per_expert_bytes
   The multi-blob output buffer matches the synchronous FETCH layout:
       byte 0   : u8 n_blobs
       then n_blobs × { u8 proj, u8 comp, u32 LE data_len, raw bytes }
*/

#if PION_MOE_WARM_HAVE_IO_URING
/* Stage 4c batched-pread path. Linux + liburing only. Submits all 9 preads
   as one batched ring entry; collects all completions in one wait call. */
static int io_uring_read_blobs(struct io_uring* ring, int fd,
                                 uint8_t* out_buf,
                                 const uint8_t* ot,
                                 int32_t expert,
                                 size_t* write_off_inout) {
    /* Build up to 9 SQEs in one submit. */
    int n_subs = 0;
    size_t blob_writes[9];   /* offsets in out_buf where each blob lands */
    size_t blob_sizes[9];
    int blob_idxs[9];

    for (int i = 0; i < 9; i++) {
        int32_t shard_id = read_i32le(ot + i * 20);
        if (shard_id <= 0) continue;
        uint64_t data_start = read_u64le(ot + i * 20 + 4);
        uint64_t per_e      = read_u64le(ot + i * 20 + 12);
        int proj = i / 3;
        int comp = i % 3;

        /* Blob header — written immediately, no I/O needed. */
        out_buf[(*write_off_inout)++] = (uint8_t)proj;
        out_buf[(*write_off_inout)++] = (uint8_t)comp;
        uint32_t dl = (uint32_t)per_e;
        out_buf[(*write_off_inout)++] = (uint8_t)(dl & 0xff);
        out_buf[(*write_off_inout)++] = (uint8_t)((dl >> 8) & 0xff);
        out_buf[(*write_off_inout)++] = (uint8_t)((dl >> 16) & 0xff);
        out_buf[(*write_off_inout)++] = (uint8_t)((dl >> 24) & 0xff);

        blob_writes[n_subs] = *write_off_inout;
        blob_sizes[n_subs] = (size_t)per_e;
        blob_idxs[n_subs] = i;
        *write_off_inout += (size_t)per_e;

        struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (!sqe) return -EBUSY;
        io_uring_prep_read(sqe, fd, out_buf + blob_writes[n_subs],
                            (unsigned)per_e,
                            (off_t)(data_start + (uint64_t)expert * per_e));
        sqe->user_data = (uint64_t)n_subs;
        n_subs++;
    }

    if (n_subs == 0) return 0;

    int submitted = io_uring_submit(ring);
    if (submitted < 0) return submitted;

    /* Wait for all completions; check each. */
    for (int i = 0; i < n_subs; i++) {
        struct io_uring_cqe* cqe = NULL;
        int rc = io_uring_wait_cqe(ring, &cqe);
        if (rc < 0) return rc;
        ssize_t res = cqe->res;
        int slot = (int)cqe->user_data;
        io_uring_cqe_seen(ring, cqe);
        if (res < 0) return (int)res;
        if ((size_t)res != blob_sizes[slot]) return -EIO;
    }
    return 0;
}
#endif  /* PION_MOE_WARM_HAVE_IO_URING */

static void worker_one_request(moe_warm_pool_t* p, const moe_warm_request_t* req) {
    /* Offset table is copied inline at enqueue time, so always present. */
    const uint8_t* ot = req->offset_table;

    /* Pass 1: count blobs + compute total buffer size. */
    int n_blobs = 0;
    size_t total_bytes = 1;   /* leading n_blobs byte */
    for (int i = 0; i < 9; i++) {
        int32_t shard_id = read_i32le(ot + i * 20);
        if (shard_id > 0) {
            uint64_t per_e = read_u64le(ot + i * 20 + 12);
            n_blobs++;
            /* 6-byte blob header + payload */
            total_bytes += 6 + (size_t)per_e;
        }
    }
    if (n_blobs == 0) {
        emit_error_completion(p, req, -ENODATA);
        return;
    }
    if (total_bytes > (size_t)0x40000000) {  /* sanity cap at 1 GiB */
        emit_error_completion(p, req, -EFBIG);
        return;
    }

    uint8_t* out_buf = (uint8_t*)malloc(total_bytes);
    if (!out_buf) {
        emit_error_completion(p, req, -ENOMEM);
        return;
    }
    out_buf[0] = (uint8_t)n_blobs;
    size_t write_off = 1;

#if PION_MOE_WARM_HAVE_IO_URING
    /* Stage 4c — Linux io_uring batched path. Single shard fd, single
       submit + drain of all blobs. Falls through to sync pread loop on
       ring init failure (init failure indicates a kernel/config issue
       — sync is still correct). */
    {
        struct io_uring ring;
        if (io_uring_queue_init(16, &ring, 0) == 0) {
            /* Each warm req hits at most ONE shard for all 9 blobs in
               OLMoE/Phi-3.5/Gemma 4 today. Open it once, batched-read. */
            int32_t shard_id_first = 0;
            for (int i = 0; i < 9; i++) {
                int32_t sid = read_i32le(ot + i * 20);
                if (sid > 0) { shard_id_first = sid; break; }
            }
            char shard_path_u[MAX_SHARD_PATH_INLINE_C + 64];
            snprintf(shard_path_u, sizeof(shard_path_u),
                     "%s/model-%05d-of-%05d.safetensors",
                     req->shard_dir, shard_id_first, req->shard_count);
            int fd_u = open(shard_path_u, O_RDONLY);
            int err_u = 0;
            if (fd_u < 0) {
                err_u = -errno;
            } else {
                err_u = io_uring_read_blobs(&ring, fd_u, out_buf, ot, req->expert, &write_off);
                close(fd_u);
            }
            io_uring_queue_exit(&ring);
            if (err_u < 0) {
                free(out_buf);
                emit_error_completion(p, req, err_u);
                return;
            }
            /* Push completion + return — done. */
            moe_warm_completion_t c = {
                .slot = req->slot, .layer = req->layer, .expert = req->expert,
                .addr = (uint64_t)(uintptr_t)out_buf, .data_len = (int32_t)write_off,
                .status = 0,
            };
            if (push_completion(p, &c) < 0) {
                free(out_buf);
                atomic_fetch_add_explicit(&p->cnt_errored, 1, memory_order_relaxed);
                return;
            }
            atomic_fetch_add_explicit(&p->cnt_completed, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&p->cnt_bytes_warmed, (uint64_t)write_off, memory_order_relaxed);
            return;
        }
        /* io_uring_queue_init failed — fall through to sync pread loop. */
    }
#endif

    /* Pass 2: open shards, pread each (proj, comp) blob, append to buffer.
       Same shard fd is reused across consecutive entries — typical for
       stacked-tensor MoEs where one shard holds many layers' tensors. */
    int32_t last_shard_id = -1;
    int fd = -1;
    int err = 0;
    char shard_path[MAX_SHARD_PATH_INLINE_C + 64];
    for (int i = 0; i < 9; i++) {
        int32_t shard_id = read_i32le(ot + i * 20);
        if (shard_id <= 0) continue;
        uint64_t data_start = read_u64le(ot + i * 20 + 4);
        uint64_t per_e      = read_u64le(ot + i * 20 + 12);
        int proj = i / 3;
        int comp = i % 3;

        if (shard_id != last_shard_id) {
            if (fd >= 0) { close(fd); fd = -1; }
            snprintf(shard_path, sizeof(shard_path),
                     "%s/model-%05d-of-%05d.safetensors",
                     req->shard_dir, shard_id, req->shard_count);
            fd = open(shard_path, O_RDONLY);
            if (fd < 0) { err = -errno; break; }
            last_shard_id = shard_id;
        }

        /* Blob header */
        out_buf[write_off++] = (uint8_t)proj;
        out_buf[write_off++] = (uint8_t)comp;
        uint32_t dl = (uint32_t)per_e;
        out_buf[write_off++] = (uint8_t)(dl & 0xff);
        out_buf[write_off++] = (uint8_t)((dl >> 8) & 0xff);
        out_buf[write_off++] = (uint8_t)((dl >> 16) & 0xff);
        out_buf[write_off++] = (uint8_t)((dl >> 24) & 0xff);

        /* Read the per-expert slice */
        off_t target = (off_t)(data_start + (uint64_t)req->expert * per_e);
        size_t remaining = (size_t)per_e;
        uint8_t* dst = out_buf + write_off;
        while (remaining > 0) {
            ssize_t got = pread(fd, dst, remaining, target);
            if (got <= 0) { err = got == 0 ? -EIO : -errno; break; }
            dst += got;
            target += got;
            remaining -= (size_t)got;
        }
        if (err < 0) break;
        write_off += (size_t)per_e;
    }
    if (fd >= 0) close(fd);

    if (err < 0) {
        free(out_buf);
        emit_error_completion(p, req, err);
        return;
    }

    /* Push completion. If completion ring is full, free the buffer — the
       consumer didn't drain in time. Increment cnt_errored so it's visible. */
    moe_warm_completion_t c = {
        .slot     = req->slot,
        .layer    = req->layer,
        .expert   = req->expert,
        .addr     = (uint64_t)(uintptr_t)out_buf,
        .data_len = (int32_t)write_off,
        .status   = 0,
    };
    if (push_completion(p, &c) < 0) {
        free(out_buf);
        atomic_fetch_add_explicit(&p->cnt_errored, 1, memory_order_relaxed);
        return;
    }
    atomic_fetch_add_explicit(&p->cnt_completed, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->cnt_bytes_warmed, (uint64_t)write_off, memory_order_relaxed);
}

static void* worker_main(void* arg) {
    moe_warm_pool_t* p = (moe_warm_pool_t*)arg;
    while (1) {
        moe_warm_request_t req;
        pthread_mutex_lock(&p->req.lock);
        while (p->req.head == p->req.tail && !p->req.shutting_down) {
            pthread_cond_wait(&p->req.not_empty, &p->req.lock);
        }
        if (p->req.shutting_down && p->req.head == p->req.tail) {
            pthread_mutex_unlock(&p->req.lock);
            return NULL;
        }
        req = p->req.slots[p->req.tail & p->req.mask];
        p->req.tail++;
        pthread_cond_signal(&p->req.not_full);
        pthread_mutex_unlock(&p->req.lock);

        worker_one_request(p, &req);
    }
}

/* ── Public API ─────────────────────────────────────────────────────── */

moe_warm_pool_t* moe_warm_pool_init(size_t request_capacity, size_t completion_capacity) {
    moe_warm_pool_t* p = (moe_warm_pool_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    if (req_ring_init(&p->req, request_capacity) != 0) { free(p); return NULL; }
    if (comp_ring_init(&p->comp, completion_capacity) != 0) {
        req_ring_destroy(&p->req); free(p); return NULL;
    }
    if (pthread_create(&p->worker_tid, NULL, worker_main, p) != 0) {
        comp_ring_destroy(&p->comp); req_ring_destroy(&p->req); free(p); return NULL;
    }
    p->worker_started = 1;
    atomic_store(&p->cnt_enqueued, 0);
    atomic_store(&p->cnt_dropped_full, 0);
    atomic_store(&p->cnt_completed, 0);
    atomic_store(&p->cnt_errored, 0);
    atomic_store(&p->cnt_bytes_warmed, 0);
    return p;
}

void moe_warm_pool_shutdown(moe_warm_pool_t* p) {
    if (!p) return;
    if (p->worker_started) {
        pthread_mutex_lock(&p->req.lock);
        p->req.shutting_down = 1;
        pthread_cond_broadcast(&p->req.not_empty);
        pthread_mutex_unlock(&p->req.lock);
        pthread_join(p->worker_tid, NULL);
    }
    comp_ring_destroy(&p->comp);
    req_ring_destroy(&p->req);
    free(p);
}

int moe_warm_pool_enqueue(moe_warm_pool_t* p,
                           int32_t slot, int32_t layer, int32_t expert,
                           const char* shard_dir, int shard_count,
                           const uint8_t* offset_table_for_layer,
                           size_t offset_table_bytes) {
    if (!p || !shard_dir || !offset_table_for_layer) return -1;
    if (offset_table_bytes != MOE_WARM_OFFSET_BYTES) return -1;
    pthread_mutex_lock(&p->req.lock);
    if (p->req.head - p->req.tail >= p->req.capacity) {
        /* Full — drop the enqueue rather than block the main thread. */
        pthread_mutex_unlock(&p->req.lock);
        atomic_fetch_add_explicit(&p->cnt_dropped_full, 1, memory_order_relaxed);
        return -1;
    }
    moe_warm_request_t* slot_ptr = &p->req.slots[p->req.head & p->req.mask];
    slot_ptr->slot = slot;
    slot_ptr->layer = layer;
    slot_ptr->expert = expert;
    slot_ptr->shard_count = shard_count;
    /* Copy shard_dir (truncate if oversized — uncommon, paths < 200 chars) */
    strncpy(slot_ptr->shard_dir, shard_dir, MOE_WARM_SHARD_PATH_MAX - 1);
    slot_ptr->shard_dir[MOE_WARM_SHARD_PATH_MAX - 1] = '\0';
    memcpy(slot_ptr->offset_table, offset_table_for_layer, MOE_WARM_OFFSET_BYTES);
    p->req.head++;
    pthread_cond_signal(&p->req.not_empty);
    pthread_mutex_unlock(&p->req.lock);
    atomic_fetch_add_explicit(&p->cnt_enqueued, 1, memory_order_relaxed);
    return 0;
}

size_t moe_warm_pool_drain(moe_warm_pool_t* p,
                            moe_warm_completion_t* out, size_t max) {
    if (!p || !out || max == 0) return 0;
    size_t head = atomic_load_explicit(&p->comp.head, memory_order_acquire);
    size_t tail = atomic_load_explicit(&p->comp.tail, memory_order_relaxed);
    size_t available = head - tail;
    size_t n = available < max ? available : max;
    for (size_t i = 0; i < n; ++i) {
        out[i] = p->comp.slots[(tail + i) & p->comp.mask];
    }
    atomic_store_explicit(&p->comp.tail, tail + n, memory_order_release);
    return n;
}

/* Free a buffer that was malloc'd by the worker. Called from the Mojo
   drain path after the contents have been memcpy'd into a Mojo-allocated
   destination. Critical: Mojo's runtime uses tcmalloc; the worker's
   malloc here goes through libc on macOS, so attempting to `p.free()`
   the libc pointer from Mojo trips "Attempt to free invalid pointer"
   in tcmalloc. This shim ensures the matching free runs. */
void moe_warm_free_buffer(uint64_t addr) {
    if (addr == 0) return;
    free((void*)(uintptr_t)addr);
}

void moe_warm_pool_stats(moe_warm_pool_t* p, moe_warm_stats_t* out) {
    if (!p || !out) return;
    out->requests_enqueued     = atomic_load_explicit(&p->cnt_enqueued,     memory_order_relaxed);
    out->requests_dropped_full = atomic_load_explicit(&p->cnt_dropped_full, memory_order_relaxed);
    out->requests_completed    = atomic_load_explicit(&p->cnt_completed,    memory_order_relaxed);
    out->requests_errored      = atomic_load_explicit(&p->cnt_errored,      memory_order_relaxed);
    out->bytes_warmed          = atomic_load_explicit(&p->cnt_bytes_warmed, memory_order_relaxed);
}
