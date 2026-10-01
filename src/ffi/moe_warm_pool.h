/* moe_warm_pool — MOE.EXPERT.* Stage 4b background-warming I/O scaffold.
 *
 * Status: scaffold only — not yet wired into the build or referenced by
 *         Mojo. Future commit (Stage 4b-1/2/3) integrates with
 *         MoEExpertTier + engine event loop.
 */
#ifndef PION_MOE_WARM_POOL_H
#define PION_MOE_WARM_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. One pool per server worker (one warming thread).
   The pool reads bytes from safetensors shards on disk and pushes
   (slot, layer, expert, addr, len) tuples onto a completion ring.
   The main thread drains completions in its event loop tick and
   transfers ownership to Pion's LRU cache via cache_store. */
typedef struct moe_warm_pool moe_warm_pool_t;

/* Completion record — what the main thread reads from the completion
   ring on each drain. The `addr` is a malloc'd buffer holding the
   multi-blob FETCH payload (same layout as the synchronous FETCH
   handler emits). Main thread owns `addr` after dequeue; pass it to
   cache_store, which then owns the free. */
typedef struct {
    int32_t  slot;
    int32_t  layer;
    int32_t  expert;
    uint64_t addr;    /* malloc'd buffer with multi-blob payload */
    int32_t  data_len;
    int32_t  status;  /* 0=ok, <0=error code (errno-style) */
} moe_warm_completion_t;

/* Initialise a pool with one warming thread and ring buffers of the
   requested capacity. Returns NULL on error. The completion ring is
   single-producer (warming thread) / single-consumer (main thread);
   the request ring is the opposite. */
moe_warm_pool_t* moe_warm_pool_init(size_t request_capacity,
                                     size_t completion_capacity);

/* Shut down the pool: signals the worker thread to exit, joins,
   frees ring buffers + struct. Pending completions are NOT auto-
   drained — caller drains first if it cares about losing them. */
void moe_warm_pool_shutdown(moe_warm_pool_t* pool);

/* Enqueue a request to warm (slot, layer, expert). The shard path
   is passed through; the worker thread opens, preads, builds the
   multi-blob buffer, and pushes a completion. shard_path may be
   referenced asynchronously, so it must outlive the request — pass
   a stable pointer (e.g. owned by MoEExpertTier's shard_dirs).
   shard_count is the total number of shards (for the per-shard
   path-component math).
   Returns 0 on success, -1 if the request ring is full. */
int moe_warm_pool_enqueue(moe_warm_pool_t* pool,
                           int32_t slot, int32_t layer, int32_t expert,
                           const char* shard_dir, int shard_count,
                           /* offset table is u64-packed: 9 entries per (proj=3, comp=3);
                              each entry is (shard_id:i32, data_start:u64, per_expert_bytes:u64). */
                           const uint8_t* offset_table_for_layer,
                           size_t offset_table_bytes);

/* Drain up to `max` completions into the caller's buffer. Returns
   the actual count drained. Safe to call from the main thread only.
   The caller takes ownership of each completion's `addr` and is
   responsible for either passing to cache_store (which frees on
   eviction) or freeing directly. */
size_t moe_warm_pool_drain(moe_warm_pool_t* pool,
                            moe_warm_completion_t* out,
                            size_t max);

/* Counters for observability — fed into MOE.EXPERT.STATS at Stage 4b-3. */
typedef struct {
    uint64_t requests_enqueued;
    uint64_t requests_dropped_full;
    uint64_t requests_completed;
    uint64_t requests_errored;
    uint64_t bytes_warmed;
} moe_warm_stats_t;

void moe_warm_pool_stats(moe_warm_pool_t* pool, moe_warm_stats_t* out);

/* Free a completion's payload buffer (libc free). Use this when handing
   ownership back to the libc allocator instead of relying on Mojo's
   tcmalloc-backed free — the worker mallocs via libc, so the matching
   free must also be libc-side. */
void moe_warm_free_buffer(uint64_t addr);

#ifdef __cplusplus
}
#endif

#endif /* PION_MOE_WARM_POOL_H */
