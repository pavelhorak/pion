// gh #9 — CUDA bridge between Pion-Mojo and the SDPA kernels.
//
// Mirrors the C ABI surface of src/ffi/metal_wrap.m (`pion_metal_sdpa_*`):
//
//   pion_cuda_sdpa_init()                                        -> int
//   pion_cuda_sdpa_store_kv(worker, sid, sid_len, layer, H, N, D, K, V) -> int
//   pion_cuda_sdpa_query   (worker, sid, sid_len, layer, H, D, Q, out, fa_window) -> int
//   pion_cuda_sdpa_drop    (worker, sid, sid_len)                -> int
//   pion_cuda_sdpa_session_exists(worker, sid, sid_len)          -> int
//
// Per-worker session table (linear-probe + tombstones), per-(session, layer)
// device K/V buffers, dispatched via the sdpa_q1_fp32 kernels.
//
// Build: nvcc -O3 -Xcompiler -fPIC -arch=sm_89 -shared. The .so is loaded
// alongside the Mojo binary by the Linux build (pixi.toml).
//
// Compile under nvcc so we can call the kernel launchers directly via C ABI;
// the launchers (`pion_sdpa_q1_fp32_cuda` etc.) are extern "C" in the .cu.

#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External kernel launchers from sdpa_q1_fp32.cu.
extern int pion_sdpa_q1_fp32_cuda(const float*, const float*, const float*,
                                  float*, uint32_t H, uint32_t N, uint32_t D,
                                  float scale, uint32_t W, cudaStream_t stream);
extern int pion_sdpa_q1_fp32_cuda_pso(const float*, const float*, const float*,
                                      float*, uint32_t H, uint32_t N, uint32_t D,
                                      float scale, uint32_t W, cudaStream_t stream);
extern int pion_sdpa_q1_sparse_fp32_cuda(
    const float*, const float*, const float*, float*,
    uint32_t H_q, uint32_t N, uint32_t D, float scale, uint32_t W,
    const int* indices, const uint32_t* counts, uint32_t K_sparse_max,
    const unsigned char* head_map, cudaStream_t stream);
extern int pion_block_mean_topk_select_cuda(
    const float* d_Q, const float* d_K, int* d_indices, uint32_t* d_counts,
    uint32_t H_q, uint32_t N, uint32_t D, uint32_t K_block, uint32_t K_blocks,
    const unsigned char* d_head_map, uint32_t W_window, cudaStream_t stream);
extern int pion_compute_block_means_cuda(
    const float* d_K, float* d_block_means,
    uint32_t H_kv, uint32_t N_tokens, uint32_t D, uint32_t K_block,
    cudaStream_t stream);
extern int pion_select_topk_from_precomputed_cuda(
    const float* d_Q, const float* d_block_means,
    int* d_indices, uint32_t* d_counts,
    uint32_t H_q, uint32_t N_tokens, uint32_t D, uint32_t K_block,
    uint32_t K_blocks, uint32_t n_blocks,
    const unsigned char* d_head_map, uint32_t W_window, cudaStream_t stream);

// Same dimensions as Metal side: 256 slots per worker × 16 workers max.
#define CUDA_SDPA_SLOTS       256
#define CUDA_SDPA_MAX_WORKERS 16

#define CUDA_SLOT_EMPTY       0ULL
#define CUDA_SLOT_TOMBSTONE   0xFFFFFFFFFFFFFFFFULL

// Staging buffer caps — sized for the expected attention shapes:
//   MAX_STAGE_H = 32    (Llama-3-70B has 64 query heads; we cap at 32 since
//                        the 24 GB VRAM bound never lets that fit anyway)
//   MAX_STAGE_D = 512   (matches the CHUNK_MAX bound in the kernels)
//   MAX_STAGE_K_SPARSE  (K_block * K_blocks worst case)
// Per worker total ~1 MB; 16 workers ~ 16 MB — negligible vs the K/V cache.
#define MAX_STAGE_H         32u
#define MAX_STAGE_D         512u
#define MAX_STAGE_K_SPARSE  4096u

// Pre-computed block-means cache (gh #9 perf #1): selector reads this instead
// of scanning the full K cache. Default K_block=64 matches gh #60. We cache
// for ONE K_block size per slot (regenerated lazily if a query uses a
// different K_block). 0 = not yet computed.
#define BM_DEFAULT_K_BLOCK 64u

typedef struct {
    uint64_t key;        // hash(session_id || layer_id), 0 = empty, ~0 = tombstone
    float* d_K;          // device pointer, [H, N, D] float32
    float* d_V;          // device pointer
    uint32_t H, N, D;
    size_t bytes;        // = H * N * D * sizeof(float)
    // Block-means cache for QUERY_SPARSE_AUTO precomputed-selector path.
    float* d_block_means;     // [H, n_blocks, D] float32; NULL until precomputed
    uint32_t bm_K_block;      // K_block this cache was built for; 0 = not built
    uint32_t bm_n_blocks;     // = ceil(N / bm_K_block) at build time
    size_t   bm_bytes;        // = H * bm_n_blocks * D * sizeof(float)
} PionCudaSDPASlot;

typedef struct {
    PionCudaSDPASlot slots[CUDA_SDPA_SLOTS];
    cudaStream_t stream;
    int initialized;
    // gh #9 perf: pre-allocated device staging buffers, reused across calls
    // — drops the per-call cudaMalloc/cudaFree pair (each ~50-100 µs) plus
    // the 5 separate alloc/free pairs for sparse_auto. Sized at worker init.
    float*         d_Q_stage;     // [MAX_STAGE_H * MAX_STAGE_D]
    float*         d_O_stage;     // [MAX_STAGE_H * MAX_STAGE_D]
    int*           d_idx_stage;   // [MAX_STAGE_H * MAX_STAGE_K_SPARSE]
    uint32_t*      d_cnt_stage;   // [MAX_STAGE_H]
    unsigned char* d_hm_stage;    // [MAX_STAGE_H]
} PionCudaSDPAWorker;

static PionCudaSDPAWorker g_workers[CUDA_SDPA_MAX_WORKERS] = {0};
static pthread_mutex_t    g_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static int                g_global_ready = 0;
static int                g_device_id = 0;

// FNV-1a 64-bit, mirroring `_sdpa_key` in metal_wrap.m + ssm_state_wrap.c.
// Reserved sentinels (0, ~0) get bumped to 1 so they never collide.
static uint64_t cuda_slot_key(const char* sid, uint32_t sid_len, uint32_t layer) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < sid_len; i++) {
        h ^= (uint64_t)(uint8_t)sid[i];
        h *= 0x100000001b3ULL;
    }
    h ^= (uint64_t)layer;
    h *= 0x100000001b3ULL;
    if (h == CUDA_SLOT_EMPTY || h == CUDA_SLOT_TOMBSTONE) h = 1;
    return h;
}

// Find slot for `key`; returns index >= 0 if present, -1 if not. Linear probe.
static int slot_find(const PionCudaSDPAWorker* w, uint64_t key) {
    uint32_t h = (uint32_t)(key & (CUDA_SDPA_SLOTS - 1));
    for (uint32_t step = 0; step < CUDA_SDPA_SLOTS; step++) {
        uint32_t i = (h + step) & (CUDA_SDPA_SLOTS - 1);
        uint64_t sk = w->slots[i].key;
        if (sk == CUDA_SLOT_EMPTY) return -1;
        if (sk == key) return (int)i;
        // tombstones: keep probing
    }
    return -1;
}

// Find an insert slot for `key` (replacing an EMPTY/TOMBSTONE/duplicate);
// returns index >= 0 on success, -1 if table is full.
static int slot_insert_pos(PionCudaSDPAWorker* w, uint64_t key) {
    uint32_t h = (uint32_t)(key & (CUDA_SDPA_SLOTS - 1));
    int first_tomb = -1;
    for (uint32_t step = 0; step < CUDA_SDPA_SLOTS; step++) {
        uint32_t i = (h + step) & (CUDA_SDPA_SLOTS - 1);
        uint64_t sk = w->slots[i].key;
        if (sk == CUDA_SLOT_EMPTY) return (first_tomb >= 0) ? first_tomb : (int)i;
        if (sk == key) return (int)i;  // overwrite same key
        if (sk == CUDA_SLOT_TOMBSTONE && first_tomb < 0) first_tomb = (int)i;
    }
    return first_tomb;  // -1 if no room at all
}

// Ensure CUDA is initialized. Idempotent + thread-safe.
int pion_cuda_sdpa_init(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_global_ready) {
        pthread_mutex_unlock(&g_init_mutex);
        return 0;
    }
    int n_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&n_devices);
    if (err != cudaSuccess || n_devices == 0) {
        fprintf(stderr, "[cuda_wrap] cudaGetDeviceCount failed: %s (n=%d)\n",
                cudaGetErrorString(err), n_devices);
        pthread_mutex_unlock(&g_init_mutex);
        return -1;
    }
    cudaSetDevice(g_device_id);
    // Touch the runtime so subsequent calls have a context already created.
    void* dummy = NULL;
    cudaMalloc(&dummy, 16);
    cudaFree(dummy);
    g_global_ready = 1;
    pthread_mutex_unlock(&g_init_mutex);
    return 0;
}

// Per-worker init: idempotent, lazy. Called from store_kv / query.
// Pre-allocates the staging buffers so query/sparse paths don't pay
// cudaMalloc latency per call.
static int worker_ensure(uint32_t worker_id) {
    if (worker_id >= CUDA_SDPA_MAX_WORKERS) return -2;
    PionCudaSDPAWorker* w = &g_workers[worker_id];
    if (w->initialized) return 0;
    pthread_mutex_lock(&g_init_mutex);
    if (!w->initialized) {
        if (cudaStreamCreate(&w->stream) != cudaSuccess) {
            pthread_mutex_unlock(&g_init_mutex);
            return -3;
        }
        size_t qo_sz  = (size_t)MAX_STAGE_H * MAX_STAGE_D * sizeof(float);
        size_t idx_sz = (size_t)MAX_STAGE_H * MAX_STAGE_K_SPARSE * sizeof(int);
        size_t cnt_sz = (size_t)MAX_STAGE_H * sizeof(uint32_t);
        size_t hm_sz  = (size_t)MAX_STAGE_H * sizeof(unsigned char);
        if (cudaMalloc(&w->d_Q_stage,   qo_sz)  != cudaSuccess ||
            cudaMalloc(&w->d_O_stage,   qo_sz)  != cudaSuccess ||
            cudaMalloc(&w->d_idx_stage, idx_sz) != cudaSuccess ||
            cudaMalloc(&w->d_cnt_stage, cnt_sz) != cudaSuccess ||
            cudaMalloc(&w->d_hm_stage,  hm_sz)  != cudaSuccess) {
            fprintf(stderr, "[cuda_wrap] worker %u staging cudaMalloc failed\n", worker_id);
            // partial — leak whatever did allocate; init flag stays 0 so future
            // attempts re-try (and likely fail again, which is the right signal)
            pthread_mutex_unlock(&g_init_mutex);
            return -4;
        }
        w->initialized = 1;
    }
    pthread_mutex_unlock(&g_init_mutex);
    return 0;
}

// Store K/V for (session_id, layer_id) under this worker's table.
// K/V layout host-side: [H, N, D] row-major float32, contiguous.
// Returns 0 on success, negative on error.
int pion_cuda_sdpa_store_kv(
    uint32_t      worker_id,
    const char*   sid,
    uint32_t      sid_len,
    uint32_t      layer_id,
    uint32_t      H,
    uint32_t      N,
    uint32_t      D,
    const float*  K_host,
    const float*  V_host)
{
    if (!g_global_ready) return -1;
    int rc = worker_ensure(worker_id);
    if (rc) return rc;
    if (H == 0 || N == 0 || D == 0) return -4;

    PionCudaSDPAWorker* w = &g_workers[worker_id];
    uint64_t key = cuda_slot_key(sid, sid_len, layer_id);
    int idx = slot_insert_pos(w, key);
    if (idx < 0) return -5;  // table full

    PionCudaSDPASlot* slot = &w->slots[idx];
    size_t need = (size_t)H * N * D * sizeof(float);

    // If overwriting same key with different bytes, free + realloc.
    if (slot->key == key && slot->bytes != need) {
        if (slot->d_K) cudaFree(slot->d_K);
        if (slot->d_V) cudaFree(slot->d_V);
        if (slot->d_block_means) cudaFree(slot->d_block_means);
        slot->d_K = NULL; slot->d_V = NULL; slot->bytes = 0;
        slot->d_block_means = NULL; slot->bm_K_block = 0; slot->bm_n_blocks = 0; slot->bm_bytes = 0;
    }
    if (slot->d_K == NULL) {
        if (cudaMalloc(&slot->d_K, need) != cudaSuccess) return -10;
        if (cudaMalloc(&slot->d_V, need) != cudaSuccess) {
            cudaFree(slot->d_K); slot->d_K = NULL; return -11;
        }
        slot->bytes = need;
    }
    if (cudaMemcpyAsync(slot->d_K, K_host, need, cudaMemcpyHostToDevice, w->stream) != cudaSuccess) return -12;
    if (cudaMemcpyAsync(slot->d_V, V_host, need, cudaMemcpyHostToDevice, w->stream) != cudaSuccess) return -13;
    slot->key = key;
    slot->H = H; slot->N = N; slot->D = D;

    // gh #9 perf #1: precompute block_means at default K_block=64 so the
    // hot path (QUERY_SPARSE_AUTO) skips the full K-scan. Free old cache
    // if present (different N or shape changed). Memory cost is small
    // (~1.7 MB per layer at H=8 N=26K D=128).
    if (slot->d_block_means) {
        cudaFree(slot->d_block_means);
        slot->d_block_means = NULL;
    }
    uint32_t bm_K_block = BM_DEFAULT_K_BLOCK;
    uint32_t bm_n_blocks = (N + bm_K_block - 1u) / bm_K_block;
    size_t bm_need = (size_t)H * bm_n_blocks * D * sizeof(float);
    if (cudaMalloc(&slot->d_block_means, bm_need) != cudaSuccess) {
        // Non-fatal: clear and let the slow-path selector handle it.
        slot->d_block_means = NULL;
        slot->bm_K_block = 0; slot->bm_n_blocks = 0; slot->bm_bytes = 0;
    } else {
        slot->bm_K_block = bm_K_block;
        slot->bm_n_blocks = bm_n_blocks;
        slot->bm_bytes = bm_need;
        // Launch precompute (uses same stream as the K/V Memcpy → ordered).
        int rb = pion_compute_block_means_cuda(
            slot->d_K, slot->d_block_means, H, N, D, bm_K_block, w->stream);
        if (rb != 0) {
            cudaFree(slot->d_block_means);
            slot->d_block_means = NULL;
            slot->bm_K_block = 0; slot->bm_n_blocks = 0; slot->bm_bytes = 0;
        }
    }
    cudaStreamSynchronize(w->stream);
    return 0;
}

// Query (M=1) using cached K/V for (session_id, layer_id).
// Q layout: [H, D] row-major float32. out: [H, D] float32.
// fa_window > 0 → sliding-window restricted to last fa_window tokens.
int pion_cuda_sdpa_query(
    uint32_t      worker_id,
    const char*   sid,
    uint32_t      sid_len,
    uint32_t      layer_id,
    uint32_t      H,
    uint32_t      D,
    const float*  Q_host,
    float*        out_host,
    uint32_t      fa_window)
{
    if (!g_global_ready) return -1;
    int rc = worker_ensure(worker_id);
    if (rc) return rc;

    PionCudaSDPAWorker* w = &g_workers[worker_id];
    uint64_t key = cuda_slot_key(sid, sid_len, layer_id);
    int idx = slot_find(w, key);
    if (idx < 0) return -6;  // miss
    PionCudaSDPASlot* slot = &w->slots[idx];
    if (slot->H != H || slot->D != D) return -7;  // shape mismatch
    if (H > MAX_STAGE_H || D > MAX_STAGE_D) return -8;  // exceeds staging cap

    // gh #9: pooled staging buffers — no cudaMalloc per call. Reuse the
    // worker's pre-allocated d_Q_stage / d_O_stage (sized for max H * D).
    size_t qsz = (size_t)H * D * sizeof(float);
    if (cudaMemcpyAsync(w->d_Q_stage, Q_host, qsz, cudaMemcpyHostToDevice, w->stream) != cudaSuccess)
        return -12;

    float scale = 1.0f / sqrtf((float)D);
    int rk = pion_sdpa_q1_fp32_cuda_pso(w->d_Q_stage, slot->d_K, slot->d_V, w->d_O_stage,
                                        H, slot->N, D, scale, fa_window, w->stream);
    if (rk != 0) return rk;
    if (cudaMemcpyAsync(out_host, w->d_O_stage, qsz, cudaMemcpyDeviceToHost, w->stream) != cudaSuccess)
        return -13;
    cudaStreamSynchronize(w->stream);
    return 0;
}

int pion_cuda_sdpa_drop(uint32_t worker_id, const char* sid, uint32_t sid_len) {
    if (!g_global_ready) return -1;
    if (worker_id >= CUDA_SDPA_MAX_WORKERS) return -2;
    PionCudaSDPAWorker* w = &g_workers[worker_id];
    if (!w->initialized) return 0;
    // Drop ALL layers for this session (any slot whose key starts with sid).
    // Linear scan since the per-layer keys aren't grouped.
    int dropped = 0;
    for (uint32_t i = 0; i < CUDA_SDPA_SLOTS; i++) {
        PionCudaSDPASlot* s = &w->slots[i];
        if (s->key == CUDA_SLOT_EMPTY || s->key == CUDA_SLOT_TOMBSTONE) continue;
        // Recover sid from key isn't possible; in practice store_kv could
        // remember the sid string. For first-pass, "drop" tombstones every
        // entry in the worker's table. Caller (Pion ATTEND.PREFIX.DROP)
        // typically calls this on a session basis at end-of-conversation.
        // TODO(v2): store sid string per-slot for selective drop.
        if (s->d_K) cudaFree(s->d_K);
        if (s->d_V) cudaFree(s->d_V);
        if (s->d_block_means) cudaFree(s->d_block_means);
        s->d_K = NULL; s->d_V = NULL; s->bytes = 0;
        s->d_block_means = NULL; s->bm_K_block = 0; s->bm_n_blocks = 0; s->bm_bytes = 0;
        s->key = CUDA_SLOT_TOMBSTONE;
        dropped++;
    }
    return dropped;
}

int pion_cuda_sdpa_session_exists(uint32_t worker_id, const char* sid,
                                  uint32_t sid_len) {
    if (!g_global_ready) return 0;
    if (worker_id >= CUDA_SDPA_MAX_WORKERS) return 0;
    PionCudaSDPAWorker* w = &g_workers[worker_id];
    if (!w->initialized) return 0;
    // Probe layer 0 — convention from Metal side: "session exists" iff at
    // least one layer is cached.
    uint64_t key = cuda_slot_key(sid, sid_len, 0);
    return slot_find(w, key) >= 0 ? 1 : 0;
}

// Sparse-mask query path: caller supplies indices + counts.
// Mirrors `pion_metal_sdpa_query_sparse` but with caller-provided index list.
// Index/counts layout host-side: indices[H, K_sparse_max] int32, counts[H] uint32.
int pion_cuda_sdpa_query_sparse(
    uint32_t      worker_id,
    const char*   sid,
    uint32_t      sid_len,
    uint32_t      layer_id,
    uint32_t      H,
    uint32_t      D,
    const float*  Q_host,
    float*        out_host,
    uint32_t      fa_window,
    const int*    indices_host,
    const uint32_t* counts_host,
    uint32_t      K_sparse_max,
    const unsigned char* head_map_host)  // nullable (NULL → identity)
{
    if (!g_global_ready) return -1;
    int rc = worker_ensure(worker_id);
    if (rc) return rc;

    PionCudaSDPAWorker* w = &g_workers[worker_id];
    uint64_t key = cuda_slot_key(sid, sid_len, layer_id);
    int idx = slot_find(w, key);
    if (idx < 0) return -6;
    PionCudaSDPASlot* slot = &w->slots[idx];
    if (slot->D != D) return -7;
    if (H > MAX_STAGE_H || D > MAX_STAGE_D || K_sparse_max > MAX_STAGE_K_SPARSE) return -8;

    // gh #9: pooled staging — reuse the worker's pre-allocated d_Q/O/idx/cnt/hm.
    size_t qsz   = (size_t)H * D * sizeof(float);
    size_t isz   = (size_t)H * K_sparse_max * sizeof(int);
    size_t csz   = (size_t)H * sizeof(uint32_t);
    size_t hmsz  = (size_t)H * sizeof(unsigned char);

    cudaMemcpyAsync(w->d_Q_stage,   Q_host,        qsz,  cudaMemcpyHostToDevice, w->stream);
    cudaMemcpyAsync(w->d_idx_stage, indices_host,  isz,  cudaMemcpyHostToDevice, w->stream);
    cudaMemcpyAsync(w->d_cnt_stage, counts_host,   csz,  cudaMemcpyHostToDevice, w->stream);
    if (head_map_host) {
        cudaMemcpyAsync(w->d_hm_stage, head_map_host, hmsz, cudaMemcpyHostToDevice, w->stream);
    }

    float scale = 1.0f / sqrtf((float)D);
    int rk = pion_sdpa_q1_sparse_fp32_cuda(w->d_Q_stage, slot->d_K, slot->d_V, w->d_O_stage,
                                            H, slot->N, D, scale, fa_window,
                                            w->d_idx_stage, w->d_cnt_stage, K_sparse_max,
                                            head_map_host ? w->d_hm_stage : NULL,
                                            w->stream);
    if (rk == 0) {
        cudaMemcpyAsync(out_host, w->d_O_stage, qsz, cudaMemcpyDeviceToHost, w->stream);
        cudaStreamSynchronize(w->stream);
    }
    return rk;
}

// ATTEND.PREFIX.QUERY_SPARSE_AUTO server-side path: select K_blocks via the
// block-mean-K + dot(Q) top-K kernel, then dispatch the sparse SDPA kernel
// over the chosen indices. Mirrors the Mac-side
// `make_pion_prompt_cache(sparse_full_layers={"K_block": B, "K_blocks": K_top})`
// algorithm.
//
// Caller (Mojo / wire handler) supplies only Q + K_block + K_blocks; selection
// happens entirely on-device. Output: H*D fp32 floats (attention output) +
// the picked indices written back to indices_out (so the consumer can audit
// or reuse them).
int pion_cuda_sdpa_query_sparse_auto(
    uint32_t      worker_id,
    const char*   sid,
    uint32_t      sid_len,
    uint32_t      layer_id,
    uint32_t      H_q,
    uint32_t      D,
    const float*  Q_host,
    float*        out_host,
    uint32_t      fa_window,
    uint32_t      K_block,
    uint32_t      K_blocks,
    const unsigned char* head_map_host,    // nullable
    int*          indices_out_host,        // optional [H_q * K_blocks * K_block] int32; NULL = don't return
    uint32_t*     counts_out_host)         // optional [H_q] uint32; NULL = don't return
{
    if (!g_global_ready) return -1;
    int rc = worker_ensure(worker_id);
    if (rc) return rc;
    if (K_block == 0 || K_blocks == 0) return -3;

    PionCudaSDPAWorker* w = &g_workers[worker_id];
    uint64_t key = cuda_slot_key(sid, sid_len, layer_id);
    int idx = slot_find(w, key);
    if (idx < 0) return -6;
    PionCudaSDPASlot* slot = &w->slots[idx];
    if (slot->D != D) return -7;

    uint32_t K_sparse_max = K_blocks * K_block;
    if (H_q > MAX_STAGE_H || D > MAX_STAGE_D || K_sparse_max > MAX_STAGE_K_SPARSE) return -8;

    // gh #9: pooled staging — reuse worker's d_Q/O/idx/cnt/hm.
    size_t qsz   = (size_t)H_q * D * sizeof(float);
    size_t isz   = (size_t)H_q * K_sparse_max * sizeof(int);
    size_t csz   = (size_t)H_q * sizeof(uint32_t);
    size_t hmsz  = (size_t)H_q * sizeof(unsigned char);

    cudaMemcpyAsync(w->d_Q_stage, Q_host, qsz, cudaMemcpyHostToDevice, w->stream);
    if (head_map_host) {
        cudaMemcpyAsync(w->d_hm_stage, head_map_host, hmsz, cudaMemcpyHostToDevice, w->stream);
    }

    // gh #9 perf #1: prefer the precomputed-block-means selector when the
    // slot has a cache for this K_block. Drops selector from O(H*N*D) to
    // O(H*n_blocks*D) — at N=26K D=128 K_block=64, ~64x less memory work.
    int rs;
    if (slot->d_block_means != NULL && slot->bm_K_block == K_block) {
        rs = pion_select_topk_from_precomputed_cuda(
            w->d_Q_stage, slot->d_block_means, w->d_idx_stage, w->d_cnt_stage,
            H_q, slot->N, D, K_block, K_blocks, slot->bm_n_blocks,
            head_map_host ? w->d_hm_stage : NULL, fa_window, w->stream);
    } else {
        // Fallback: full-K-scan selector (used when K_block doesn't match
        // the cached one, or the cache failed to build).
        rs = pion_block_mean_topk_select_cuda(
            w->d_Q_stage, slot->d_K, w->d_idx_stage, w->d_cnt_stage,
            H_q, slot->N, D, K_block, K_blocks,
            head_map_host ? w->d_hm_stage : NULL, fa_window, w->stream);
    }
    if (rs != 0) return -20 + rs;

    // Step 2: sparse SDPA over the picked indices.
    float scale = 1.0f / sqrtf((float)D);
    int rk = pion_sdpa_q1_sparse_fp32_cuda(w->d_Q_stage, slot->d_K, slot->d_V, w->d_O_stage,
                                            H_q, slot->N, D, scale, fa_window,
                                            w->d_idx_stage, w->d_cnt_stage, K_sparse_max,
                                            head_map_host ? w->d_hm_stage : NULL,
                                            w->stream);
    if (rk == 0) {
        cudaMemcpyAsync(out_host, w->d_O_stage, qsz, cudaMemcpyDeviceToHost, w->stream);
        if (indices_out_host) cudaMemcpyAsync(indices_out_host, w->d_idx_stage, isz, cudaMemcpyDeviceToHost, w->stream);
        if (counts_out_host)  cudaMemcpyAsync(counts_out_host,  w->d_cnt_stage, csz, cudaMemcpyDeviceToHost, w->stream);
        cudaStreamSynchronize(w->stream);
    }
    return rk;
}
