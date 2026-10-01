// gh #65 — SSM.PREFIX.STORE/FETCH/DROP server-side byte-blob storage.
// gh #94 — WAL + snapshot durability (mirrors src/network/v_store.mojo).
//
// Pure host-side; no Metal. The state-space-model recurrent state (Mamba's
// [conv_state, ssm_state] tuple, or any future linear-attention family's
// per-layer state) is serialized client-side into an opaque byte blob; the
// server just stores and returns it.
//
// Per-worker hash table of (sid, layer_id) → malloc'd byte blob. Shared-
// nothing: STORE on worker A goes to A's table; FETCH on A reads A's table.
// (gh #65 v1: callers SHOULD pin sid → worker via `-w 1` for now. Multi-
// worker routing of SSM state is a follow-on, same shape as the gh #11
// KV.PREFIX cross-worker directory.)
//
// Blob format is OPAQUE to the server: the consumer chooses serialization.
// Recommended layout (gh #65 issue body):
//   uint32 version
//   uint32 n_arrays
//   for each array: uint32 ndim, uint32[ndim] shape, uint32 dtype_code, raw bytes
// — but the server never parses it. This keeps the substrate model-family-
// agnostic (Mamba, RWKV, RetNet, Hedgehog, GLA all serialize differently).
//
// ── Durability (gh #94) ──────────────────────────────────────────────────
// WAL (append-only, per-worker, mirrors V-store WAL):
//   [1B op] [1B sid_len] [4B layer_id] [4B blob_len] [sid] [blob]
//   op=1 STORE / op=2 DROP (blob_len always 0; layer_id=0xFFFFFFFF = drop all)
//
// Snapshot (pion.ssm.<worker_id>, mirrors V-store snapshot):
//   [8B magic "PIONSS01"][4B version=1][4B slot_count]
//   for each populated slot:
//     [1B sid_len][4B layer_id][4B blob_len][sid][blob]
//
// Recovery: load snapshot → replay WAL → open WAL for append.
// KV.PREFIX.SAVE compacts: write fresh snapshot, then truncate WAL.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// Match the SDPA shape — keeps everything inside Pion consistent. 256 slots
// per worker × 16 workers = 4096 sessions × n_layers total slots. At Mamba-
// 130M with 24 layers, ~170 distinct sessions; at Llama-3-70B-class hybrid
// with 80 layers, ~50 sessions. Bumping is a constant-define change if
// production needs it.
#define SSM_SLOTS 256
#define SSM_MAX_WORKERS 16
#define SSM_MAX_SID_LEN 255   // 1-byte sid_len in WAL/snapshot record

#define SSM_KEY_EMPTY     0ULL
#define SSM_KEY_TOMBSTONE 0xFFFFFFFFFFFFFFFFULL

// WAL op codes
#define SSM_WAL_OP_STORE 1
#define SSM_WAL_OP_DROP  2
// layer_id sentinel for DROP-all-layers (matches the -1 path in pion_ssm_drop).
#define SSM_WAL_DROP_ALL_LAYER 0xFFFFFFFFu

typedef struct {
    uint64_t key;
    void *blob;
    size_t blob_len;
    size_t blob_cap;
    // gh #94: sid + layer copy so snapshot/replay can recover the original
    // (sid, layer_id) tuple without consulting the WAL. Negligible memory
    // overhead — sids are small (<32B typical, capped at SSM_MAX_SID_LEN).
    uint32_t layer_id;
    uint8_t  sid_len;
    uint8_t  sid[SSM_MAX_SID_LEN];
} PionSSMSlot;

typedef struct {
    PionSSMSlot slots[SSM_SLOTS];
    // gh #94: per-worker WAL state. wal_fd < 0 disables append (cold path /
    // replay in progress). wal_path is kept so wal_truncate() can reopen
    // a freshly-truncated file in append mode.
    int wal_fd;
    char wal_path[1024];
    uint64_t wal_appended;
    uint64_t wal_replayed;
} PionSSMWorker;

static PionSSMWorker g_ssm_workers[SSM_MAX_WORKERS] = {0};
static pthread_mutex_t g_ssm_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_ssm_ready = 0;

// FNV-1a 64-bit, mirroring _sdpa_key in metal_wrap.m. Reserved sentinels
// (EMPTY=0, TOMBSTONE=UINT64_MAX) get bumped to 1 so they never collide
// with a real key.
static uint64_t _ssm_key(const char *sid, uint32_t sid_len, uint32_t layer) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < sid_len; i++) {
        h ^= (uint64_t)(uint8_t)sid[i];
        h *= 0x100000001b3ULL;
    }
    h ^= ((uint64_t)layer << 56);
    if (h == SSM_KEY_EMPTY || h == SSM_KEY_TOMBSTONE) h = 1ULL;
    return h;
}

static PionSSMWorker *_ssm_worker(uint32_t worker_id) {
    if (worker_id >= SSM_MAX_WORKERS) return NULL;
    return &g_ssm_workers[worker_id];
}

// Linear probe over SSM_SLOTS. Returns slot index. Out-param `out_slot` is
// set to the slot index; return value 0 = found existing, 1 = allocated new,
// negative = error (table full).
static int _ssm_find_or_alloc(PionSSMWorker *w, uint64_t key, int *out_slot) {
    size_t start = (size_t)(key & (SSM_SLOTS - 1));
    int first_tombstone = -1;
    for (size_t i = 0; i < SSM_SLOTS; i++) {
        size_t slot = (start + i) & (SSM_SLOTS - 1);
        uint64_t k = w->slots[slot].key;
        if (k == key) { *out_slot = (int)slot; return 0; }
        if (k == SSM_KEY_EMPTY) {
            int target = (first_tombstone >= 0) ? first_tombstone : (int)slot;
            w->slots[target].key = key;
            *out_slot = target;
            return 1;
        }
        if (k == SSM_KEY_TOMBSTONE && first_tombstone < 0)
            first_tombstone = (int)slot;
    }
    // Table fully populated. v1 behavior: evict the slot we hashed to.
    // (Match the SDPA cache's eviction-on-full policy. Production with high
    // session counts will need an LRU or explicit DROP. Bump SSM_SLOTS if
    // hitting this in practice.)
    if (w->slots[start].blob) {
        free(w->slots[start].blob);
        w->slots[start].blob = NULL;
        w->slots[start].blob_cap = 0;
        w->slots[start].blob_len = 0;
    }
    w->slots[start].sid_len = 0;
    w->slots[start].layer_id = 0;
    w->slots[start].key = key;
    *out_slot = (int)start;
    return 1;
}

static int _ssm_find(PionSSMWorker *w, uint64_t key) {
    size_t start = (size_t)(key & (SSM_SLOTS - 1));
    for (size_t i = 0; i < SSM_SLOTS; i++) {
        size_t slot = (start + i) & (SSM_SLOTS - 1);
        uint64_t k = w->slots[slot].key;
        if (k == key) return (int)slot;
        if (k == SSM_KEY_EMPTY) return -1;
        // tombstone: keep probing
    }
    return -1;
}

// ── gh #94: WAL helpers ─────────────────────────────────────────────────

// Write the full buffer or fail. Matches V-store's _wal_write_all semantics.
static int _ssm_write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t w = write(fd, p, remaining);
        if (w <= 0) return -1;
        p += w;
        remaining -= (size_t)w;
    }
    return 0;
}

static int _ssm_read_all(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = read(fd, p, remaining);
        if (r <= 0) return -1;
        p += r;
        remaining -= (size_t)r;
    }
    return 0;
}

// Append one WAL record. Best-effort: append failures are non-fatal — the
// in-memory state is correct, only durability is degraded for this op.
static void _ssm_wal_append_record(PionSSMWorker *w,
                                   uint8_t op,
                                   const char *sid, uint32_t sid_len,
                                   uint32_t layer_id_wire,
                                   const void *blob, uint32_t blob_len) {
    if (!w || w->wal_fd < 0) return;
    if (sid_len == 0 || sid_len > SSM_MAX_SID_LEN) return;  // refuse oversized sid
    // Header [1B op][1B sid_len][4B layer_id][4B blob_len] + sid + blob.
    uint8_t hdr[10];
    hdr[0] = op;
    hdr[1] = (uint8_t)sid_len;
    hdr[2] = (uint8_t)(layer_id_wire        & 0xFF);
    hdr[3] = (uint8_t)((layer_id_wire >>  8) & 0xFF);
    hdr[4] = (uint8_t)((layer_id_wire >> 16) & 0xFF);
    hdr[5] = (uint8_t)((layer_id_wire >> 24) & 0xFF);
    hdr[6] = (uint8_t)(blob_len        & 0xFF);
    hdr[7] = (uint8_t)((blob_len >>  8) & 0xFF);
    hdr[8] = (uint8_t)((blob_len >> 16) & 0xFF);
    hdr[9] = (uint8_t)((blob_len >> 24) & 0xFF);
    if (_ssm_write_all(w->wal_fd, hdr, sizeof(hdr)) < 0) return;
    if (_ssm_write_all(w->wal_fd, sid, sid_len) < 0) return;
    if (blob_len > 0 && _ssm_write_all(w->wal_fd, blob, blob_len) < 0) return;
    w->wal_appended++;
}

// ── FFI entry points ────────────────────────────────────────────────────

int32_t pion_ssm_init(void) {
    if (g_ssm_ready) return 0;
    pthread_mutex_lock(&g_ssm_init_mutex);
    if (g_ssm_ready) { pthread_mutex_unlock(&g_ssm_init_mutex); return 0; }
    // Slots are zero-initialized via the static decl above. We also need
    // wal_fd=-1 to signal "WAL not yet opened" on every worker.
    for (int w = 0; w < SSM_MAX_WORKERS; w++) {
        g_ssm_workers[w].wal_fd = -1;
        g_ssm_workers[w].wal_path[0] = '\0';
        g_ssm_workers[w].wal_appended = 0;
        g_ssm_workers[w].wal_replayed = 0;
    }
    g_ssm_ready = 1;
    pthread_mutex_unlock(&g_ssm_init_mutex);
    return 0;
}

// STORE: idempotent overwrite. Reallocates the blob buffer if the new size
// exceeds the slot's current capacity; else reuses (the common case for
// per-token snapshots that hit the same slot with same-size state).
//
// Returns 0 on success, -1 on bad worker, -2 on alloc failure.
int32_t pion_ssm_store(uint32_t worker_id,
                       const char *sid, uint32_t sid_len, uint32_t layer_id,
                       const void *blob, uint32_t blob_len) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    uint64_t key = _ssm_key(sid, sid_len, layer_id);
    int slot;
    if (_ssm_find_or_alloc(w, key, &slot) < 0) return -2;
    PionSSMSlot *s = &w->slots[slot];
    if (s->blob_cap < blob_len) {
        // Grow with a 64KB-aligned overhead so frequent same-size updates
        // don't realloc.
        size_t cap = (blob_len + 65535u) & ~(size_t)65535u;
        void *nb = realloc(s->blob, cap);
        if (!nb) return -2;
        s->blob = nb;
        s->blob_cap = cap;
    }
    if (blob_len > 0 && blob) memcpy(s->blob, blob, blob_len);
    s->blob_len = blob_len;
    // gh #94: record sid+layer for snapshot serialization (skip if sid is
    // longer than the WAL's 1-byte len field — durability is degraded for
    // that key but in-memory STORE still succeeds, matching V-store's
    // best-effort policy).
    if (sid_len > 0 && sid_len <= SSM_MAX_SID_LEN) {
        s->sid_len = (uint8_t)sid_len;
        memcpy(s->sid, sid, sid_len);
    } else {
        s->sid_len = 0;
    }
    s->layer_id = layer_id;
    // gh #94: WAL append. Best-effort.
    _ssm_wal_append_record(w, SSM_WAL_OP_STORE, sid, sid_len, layer_id,
                           blob, blob_len);
    return 0;
}

// FETCH: caller-allocated `out_buf` (capacity `out_buf_cap`). Always writes
// the actual stored size to `*out_len`. Returns:
//   0  → ok, blob copied
//  -1  → not found (or bad worker)
//  -2  → out_buf_cap < blob_len (out_len is set; caller can retry with bigger)
int32_t pion_ssm_fetch(uint32_t worker_id,
                       const char *sid, uint32_t sid_len, uint32_t layer_id,
                       void *out_buf, uint32_t out_buf_cap, uint32_t *out_len) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    uint64_t key = _ssm_key(sid, sid_len, layer_id);
    int slot = _ssm_find(w, key);
    if (slot < 0) return -1;
    PionSSMSlot *s = &w->slots[slot];
    if (out_len) *out_len = (uint32_t)s->blob_len;
    if (out_buf_cap < s->blob_len) return -2;
    if (s->blob_len > 0 && out_buf) memcpy(out_buf, s->blob, s->blob_len);
    return 0;
}

// Internal: drop one (sid, layer) slot. Returns 1 if removed, 0 if missing.
static int _ssm_drop_one(PionSSMWorker *w, const char *sid, uint32_t sid_len, uint32_t layer_id) {
    uint64_t key = _ssm_key(sid, sid_len, layer_id);
    int slot = _ssm_find(w, key);
    if (slot < 0) return 0;
    if (w->slots[slot].blob) {
        free(w->slots[slot].blob);
        w->slots[slot].blob = NULL;
        w->slots[slot].blob_cap = 0;
        w->slots[slot].blob_len = 0;
    }
    w->slots[slot].sid_len = 0;
    w->slots[slot].layer_id = 0;
    w->slots[slot].key = SSM_KEY_TOMBSTONE;
    return 1;
}

// DROP: layer_id >= 0 drops a specific (sid, layer); layer_id == -1 drops
// ALL layers for this sid (scans the table).
//
// Returns 0 on success (including not-found — DROP is idempotent), -1 on
// bad worker.
int32_t pion_ssm_drop(uint32_t worker_id,
                      const char *sid, uint32_t sid_len, int32_t layer_id) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    if (layer_id >= 0) {
        (void)_ssm_drop_one(w, sid, sid_len, (uint32_t)layer_id);
        // gh #94: WAL the DROP (single-layer).
        _ssm_wal_append_record(w, SSM_WAL_OP_DROP, sid, sid_len,
                               (uint32_t)layer_id, NULL, 0);
        return 0;
    }
    // layer_id == -1: drop ALL layers matching this sid. Linear scan — slow
    // O(SSM_SLOTS) but DROP_ALL is per-session-end, not per-decode-step.
    // We recompute the (sid, layer) key for each common layer count up to
    // the bound (256 layers is more than any LLM today).
    for (uint32_t layer = 0u; layer < 256u; layer++) {
        (void)_ssm_drop_one(w, sid, sid_len, layer);
    }
    // gh #94: single DROP_ALL record so replay reconstructs the all-layers
    // semantics without having to enumerate every layer.
    _ssm_wal_append_record(w, SSM_WAL_OP_DROP, sid, sid_len,
                           SSM_WAL_DROP_ALL_LAYER, NULL, 0);
    return 0;
}

// SIZE: peek at the blob size without copying. Used by RESP handlers to
// allocate the response buffer before FETCH. Returns 0 on success, -1 on
// not-found.
int32_t pion_ssm_size(uint32_t worker_id,
                      const char *sid, uint32_t sid_len, uint32_t layer_id,
                      uint32_t *out_len) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    uint64_t key = _ssm_key(sid, sid_len, layer_id);
    int slot = _ssm_find(w, key);
    if (slot < 0) return -1;
    if (out_len) *out_len = (uint32_t)w->slots[slot].blob_len;
    return 0;
}

// ── gh #94: WAL append-mode lifecycle ───────────────────────────────────

// Open the WAL in append mode for ongoing mutations. Call AFTER
// pion_ssm_wal_replay() so replay doesn't see records we're about to write.
// Idempotent — closes any previously-open fd. Returns 0 on success, -1 on
// failure (caller logs; engine continues — durability is degraded for this
// worker but in-memory store still works).
int32_t pion_ssm_wal_open(uint32_t worker_id, const char *path) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w || !path) return -1;
    if (w->wal_fd >= 0) {
        close(w->wal_fd);
        w->wal_fd = -1;
    }
    size_t pn = strlen(path);
    if (pn >= sizeof(w->wal_path)) return -1;
    memcpy(w->wal_path, path, pn + 1);
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return -1;
    w->wal_fd = fd;
    return 0;
}

int32_t pion_ssm_wal_close(uint32_t worker_id) {
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    if (w->wal_fd >= 0) {
        close(w->wal_fd);
        w->wal_fd = -1;
    }
    return 0;
}

// Truncate the WAL — call after a successful snapshot is durable. Reopens
// the fd in append mode against a freshly truncated file. No-op if no path
// is registered (i.e. wal_open was never called).
int32_t pion_ssm_wal_truncate(uint32_t worker_id) {
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w) return -1;
    if (w->wal_path[0] == '\0') return 0;
    if (w->wal_fd >= 0) {
        close(w->wal_fd);
        w->wal_fd = -1;
    }
    int trunc_fd = open(w->wal_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (trunc_fd >= 0) close(trunc_fd);
    int append_fd = open(w->wal_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    w->wal_fd = append_fd;
    w->wal_appended = 0;
    return (append_fd >= 0) ? 0 : -1;
}

// Diagnostics (used by INFO).
uint64_t pion_ssm_wal_appended(uint32_t worker_id) {
    PionSSMWorker *w = _ssm_worker(worker_id);
    return w ? w->wal_appended : 0;
}

uint64_t pion_ssm_wal_replayed(uint32_t worker_id) {
    PionSSMWorker *w = _ssm_worker(worker_id);
    return w ? w->wal_replayed : 0;
}

// ── gh #94: snapshot save / load ────────────────────────────────────────
//
// Snapshot format (little-endian, mirrors V-store's PIONVS01):
//   [8B magic "PIONSS01"]
//   [4B version = 1]
//   [4B slot_count]
//   for each populated slot:
//     [1B sid_len][4B layer_id][4B blob_len][sid][blob]
//
// Walks the per-worker slot table; writes every slot whose sid_len > 0 (so
// pre-gh#94 slots created without a recorded sid get skipped — durability
// is only for sessions written through the new STORE path).

int32_t pion_ssm_save_snapshot(uint32_t worker_id, const char *path) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w || !path) return -1;

    // Count populated slots first so we can write a fixed header.
    uint32_t saved = 0;
    for (int i = 0; i < SSM_SLOTS; i++) {
        uint64_t k = w->slots[i].key;
        if (k == SSM_KEY_EMPTY || k == SSM_KEY_TOMBSTONE) continue;
        if (w->slots[i].sid_len == 0) continue;  // no recoverable sid
        saved++;
    }

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;

    // Header: 8B magic + 4B version + 4B slot_count = 16 bytes.
    uint8_t hdr[16];
    static const char magic[8] = {'P','I','O','N','S','S','0','1'};
    memcpy(hdr, magic, 8);
    uint32_t version = 1u;
    memcpy(hdr + 8, &version, 4);
    memcpy(hdr + 12, &saved, 4);
    if (_ssm_write_all(fd, hdr, sizeof(hdr)) < 0) {
        close(fd);
        return -1;
    }

    for (int i = 0; i < SSM_SLOTS; i++) {
        uint64_t k = w->slots[i].key;
        if (k == SSM_KEY_EMPTY || k == SSM_KEY_TOMBSTONE) continue;
        PionSSMSlot *s = &w->slots[i];
        if (s->sid_len == 0) continue;
        // Per-slot header: [1B sid_len][4B layer_id][4B blob_len] = 9 bytes.
        uint8_t shdr[9];
        shdr[0] = s->sid_len;
        memcpy(shdr + 1, &s->layer_id, 4);
        uint32_t blen = (uint32_t)s->blob_len;
        memcpy(shdr + 5, &blen, 4);
        if (_ssm_write_all(fd, shdr, sizeof(shdr)) < 0) { close(fd); return -1; }
        if (_ssm_write_all(fd, s->sid, s->sid_len) < 0) { close(fd); return -1; }
        if (blen > 0 && _ssm_write_all(fd, s->blob, blen) < 0) { close(fd); return -1; }
    }
    close(fd);
    return (int32_t)saved;
}

int32_t pion_ssm_load_snapshot(uint32_t worker_id, const char *path) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w || !path) return -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;  // Cold start — no snapshot is OK.

    uint8_t hdr[16];
    if (_ssm_read_all(fd, hdr, sizeof(hdr)) < 0) {
        close(fd);
        fprintf(stderr, "SSM snapshot: short header read\n");
        return -1;
    }
    if (hdr[0] != 'P' || hdr[1] != 'I' || hdr[2] != 'O' || hdr[3] != 'N'
     || hdr[4] != 'S' || hdr[5] != 'S' || hdr[6] != '0' || hdr[7] != '1') {
        close(fd);
        fprintf(stderr, "SSM snapshot: bad magic\n");
        return -1;
    }
    uint32_t version, slot_count;
    memcpy(&version, hdr + 8, 4);
    memcpy(&slot_count, hdr + 12, 4);
    if (version != 1u) {
        close(fd);
        fprintf(stderr, "SSM snapshot: unsupported version %u\n", version);
        return -1;
    }
    if (slot_count > SSM_SLOTS) {
        close(fd);
        fprintf(stderr, "SSM snapshot: invalid slot_count %u\n", slot_count);
        return -1;
    }

    uint32_t loaded = 0;
    for (uint32_t i = 0; i < slot_count; i++) {
        uint8_t shdr[9];
        if (_ssm_read_all(fd, shdr, sizeof(shdr)) < 0) {
            close(fd);
            fprintf(stderr, "SSM snapshot: truncated slot header at %u\n", i);
            return -1;
        }
        uint8_t sid_len = shdr[0];
        uint32_t layer_id, blob_len;
        memcpy(&layer_id, shdr + 1, 4);
        memcpy(&blob_len, shdr + 5, 4);
        if (sid_len == 0 || sid_len > SSM_MAX_SID_LEN) {
            close(fd);
            fprintf(stderr, "SSM snapshot: invalid sid_len %u at slot %u\n",
                    (unsigned)sid_len, i);
            return -1;
        }
        char sid_buf[SSM_MAX_SID_LEN];
        if (_ssm_read_all(fd, sid_buf, sid_len) < 0) { close(fd); return -1; }
        // Allocate the blob buffer and stream-read into it (avoids a second
        // copy through a temporary). Skipping the WAL path entirely — load is
        // pre-replay and the WAL is what brought us back to the snapshot state
        // in the first place; re-writing here would just double the WAL.
        uint64_t key = _ssm_key(sid_buf, sid_len, layer_id);
        int slot;
        if (_ssm_find_or_alloc(w, key, &slot) < 0) { close(fd); return -1; }
        PionSSMSlot *s = &w->slots[slot];
        if (blob_len > 0) {
            size_t cap = (blob_len + 65535u) & ~(size_t)65535u;
            void *nb = realloc(s->blob, cap);
            if (!nb) { close(fd); return -1; }
            s->blob = nb;
            s->blob_cap = cap;
            if (_ssm_read_all(fd, s->blob, blob_len) < 0) { close(fd); return -1; }
        }
        s->blob_len = blob_len;
        s->sid_len = sid_len;
        memcpy(s->sid, sid_buf, sid_len);
        s->layer_id = layer_id;
        loaded++;
    }
    close(fd);
    return (int32_t)loaded;
}

// ── gh #94: WAL replay ──────────────────────────────────────────────────
//
// Walks the WAL applying STORE / DROP records onto whatever state the snapshot
// load left in place. Stops at the first short read (truncated tail = crash
// mid-write; safe to drop). Replay calls into the *underlying* slot table
// directly — it MUST NOT re-emit WAL records (would double the log), so the
// wal_fd is closed before this runs.

int32_t pion_ssm_wal_replay(uint32_t worker_id, const char *path) {
    if (!g_ssm_ready) pion_ssm_init();
    PionSSMWorker *w = _ssm_worker(worker_id);
    if (!w || !path) return -1;

    // Sanity: ensure WAL append is off during replay.
    if (w->wal_fd >= 0) { close(w->wal_fd); w->wal_fd = -1; }

    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;  // No WAL — fresh start.

    uint32_t n_store = 0, n_drop = 0;
    while (1) {
        uint8_t hdr[10];
        if (_ssm_read_all(fd, hdr, sizeof(hdr)) < 0) break;  // EOF or short tail
        uint8_t op = hdr[0];
        uint8_t sid_len = hdr[1];
        uint32_t layer_id_wire, blob_len;
        memcpy(&layer_id_wire, hdr + 2, 4);
        memcpy(&blob_len, hdr + 6, 4);
        if (sid_len == 0 || sid_len > SSM_MAX_SID_LEN) {
            fprintf(stderr, "SSM WAL replay: invalid sid_len %u\n", (unsigned)sid_len);
            break;
        }
        if (blob_len > (1u << 30)) {
            fprintf(stderr, "SSM WAL replay: blob_len %u exceeds 1GiB cap\n", blob_len);
            break;
        }
        char sid_buf[SSM_MAX_SID_LEN];
        if (_ssm_read_all(fd, sid_buf, sid_len) < 0) break;  // truncated tail
        void *blob_buf = NULL;
        if (blob_len > 0) {
            blob_buf = malloc(blob_len);
            if (!blob_buf) { fprintf(stderr, "SSM WAL replay: OOM on %u-byte blob\n", blob_len); break; }
            if (_ssm_read_all(fd, blob_buf, blob_len) < 0) { free(blob_buf); break; }
        }
        if (op == SSM_WAL_OP_STORE) {
            // Apply STORE in-place — bypass the WAL append path.
            uint64_t key = _ssm_key(sid_buf, sid_len, layer_id_wire);
            int slot;
            if (_ssm_find_or_alloc(w, key, &slot) >= 0) {
                PionSSMSlot *s = &w->slots[slot];
                if (s->blob_cap < blob_len) {
                    size_t cap = (blob_len + 65535u) & ~(size_t)65535u;
                    void *nb = realloc(s->blob, cap);
                    if (nb) { s->blob = nb; s->blob_cap = cap; }
                }
                if (blob_len > 0 && s->blob) memcpy(s->blob, blob_buf, blob_len);
                s->blob_len = blob_len;
                s->sid_len = sid_len;
                memcpy(s->sid, sid_buf, sid_len);
                s->layer_id = layer_id_wire;
                n_store++;
            }
        } else if (op == SSM_WAL_OP_DROP) {
            if (layer_id_wire == SSM_WAL_DROP_ALL_LAYER) {
                for (uint32_t layer = 0u; layer < 256u; layer++) {
                    (void)_ssm_drop_one(w, sid_buf, sid_len, layer);
                }
            } else {
                (void)_ssm_drop_one(w, sid_buf, sid_len, layer_id_wire);
            }
            n_drop++;
        } else {
            fprintf(stderr, "SSM WAL replay: unknown op %u, stopping\n", (unsigned)op);
            if (blob_buf) free(blob_buf);
            break;
        }
        if (blob_buf) free(blob_buf);
    }
    close(fd);
    w->wal_replayed = (uint64_t)(n_store + n_drop);
    if (w->wal_replayed > 0) {
        // stderr is unbuffered when redirected to a file, so the line lands
        // before the next syscall — operators see replay activity in real time
        // instead of "at process exit when stdout finally flushes".
        fprintf(stderr, "SSM WAL replayed: %u store, %u drop (worker %u)\n",
                n_store, n_drop, worker_id);
    }
    return (int32_t)w->wal_replayed;
}
