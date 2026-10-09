#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <signal.h>
#include <netdb.h>
/* gh #258: the process-wide bind address, defined in fcntl_wrap.c (linked into
 * every build variant). Declared rather than included so the two shims stay
 * independently compilable. Was INADDR_ANY at all three listeners below. */
uint32_t pion_get_bind_addr(void);
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

/* ── WAL helpers (both macOS and Linux) ─────────────────────────────────────
   These wrap the syscalls that have type-width issues in Mojo's external_call
   (off_t for ftruncate/mmap, size_t for msync/munmap).
*/

void* pion_wal_mmap(int fd, size_t length) {
    return mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
}

int pion_wal_munmap(void* addr, size_t length) {
    return munmap(addr, length);
}

int pion_wal_ftruncate(int fd, size_t length) {
    return ftruncate(fd, (off_t)length);
}

int pion_wal_msync(void* addr, size_t length, int flags) {
    return msync(addr, length, flags);
}

/* pion_file_size: fstat().st_size — the blob tier (gh #163) maps segments at their
   real on-disk length, which is not fixed: a payload larger than the standard
   segment size gets a segment sized to fit it. */
long pion_file_size(int fd) {
    struct stat st;
    if (fstat(fd, &st) != 0) return -1;
    return (long)st.st_size;
}

/* pion_wal_open: open(path, O_RDWR|O_CREAT, 0644) — wraps the variadic open() */
int pion_wal_open(const char* path) {
    return open(path, O_RDWR | O_CREAT, 0644);
}

/* pion_write / pion_read: raw file I/O wrappers — Mojo external_call("write") is
   marked illegal in some Mojo versions because it shadows the stdlib write fn. */
ssize_t pion_write(int fd, const void* buf, size_t count) {
    return write(fd, buf, count);
}

ssize_t pion_read(int fd, void* buf, size_t count) {
    return read(fd, buf, count);
}

/* pion_pread: positional read — used by MOE.EXPERT.FETCH to read a per-expert
   byte slice from a safetensors shard at a precomputed offset, without
   advancing the file pointer (safe across concurrent FETCHes on shared fd). */
ssize_t pion_pread(int fd, void* buf, size_t count, off_t offset) {
    return pread(fd, buf, count, offset);
}

/* pion_open_rdonly: open file read-only (O_RDONLY=0; variadic open needs mode only for O_CREAT) */
int pion_open_rdonly(const char* path) {
    return open(path, O_RDONLY);
}

/* pion_fadvise_willneed: hint the kernel to prefetch a byte range into the
   page cache. Used by MOE.EXPERT.PREFETCH (Stage-4 async I/O) — the prefetch
   handler advises ranges for the requested experts then returns +OK; the
   kernel performs the actual read-ahead asynchronously so the next FETCH
   on the same range hits page cache.

   Linux: posix_fadvise(fd, offset, len, POSIX_FADV_WILLNEED) — initiates
   asynchronous read-ahead into the page cache. Non-blocking.

   macOS: fcntl(fd, F_RDADVISE, &radvisory{ra_offset, ra_count}) — same
   semantics. The kernel reads the range into the unified buffer cache.

   Returns 0 on success, -1 on failure. Failures are non-fatal (FETCH will
   still serve correctly; it just won't have benefited from prefetch). */
int pion_fadvise_willneed(int fd, int64_t offset, size_t length) {
#ifdef __linux__
    return posix_fadvise(fd, (off_t)offset, length, POSIX_FADV_WILLNEED);
#elif defined(__APPLE__)
    struct radvisory ra;
    ra.ra_offset = (off_t)offset;
    ra.ra_count = (int)length;
    return fcntl(fd, F_RDADVISE, &ra);
#else
    (void)fd; (void)offset; (void)length;
    return 0;
#endif
}

/* pion_creat: open for write, truncate/create */
int pion_creat(const char* path) {
    return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}

/* pion_open_append: open for write, create if missing, append-only.
   Used by V-store WAL — every write goes to end of file. */
int pion_open_append(const char* path) {
    return open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
}

/* ── Snapshot / process helpers ──────────────────────────────────────────────
   fork, waitpid (non-blocking), atomic rename, Unix timestamp, fdatasync.
*/

int32_t pion_fork(void) {
    return (int32_t)fork();
}

/* Returns 0=child still running, >0=child pid (exited), -1=error */
int32_t pion_waitpid_nonblocking(int32_t pid) {
    int status = 0;
    return (int32_t)waitpid((pid_t)pid, &status, WNOHANG);
}

int pion_snapshot_rename(const char* src, const char* dst) {
    return rename(src, dst);
}

int64_t pion_get_unix_time(void) {
    return (int64_t)time(NULL);
}

int pion_fdatasync(int fd) {
#ifdef __linux__
    return fdatasync(fd);
#elif defined(__APPLE__)
    /* macOS fsync() hands the data to the drive and returns; the drive may
       still hold it in its volatile cache, so a power cut or a kernel panic
       can lose it. F_FULLFSYNC asks the drive to flush that cache — the only
       call on macOS that means "on stable storage". Measured on an M4 mini's
       SSD (2026-09-24): ~3.9 ms floor, 4.6 ms at 2.3 MB, 9.7 ms at 16 MB,
       against 0.04-1.9 ms for fsync. Filesystems that do not support it
       (some network mounts) fail it; fall back to fsync there. */
    if (fcntl(fd, F_FULLFSYNC) == 0) return 0;
    return fsync(fd);
#else
    return fsync(fd);
#endif
}

/* pion_connect_tcp: resolve host:port and return a connected TCP socket fd, or -1 on error.
   Handles both IP addresses and hostnames via getaddrinfo. */
int pion_connect_tcp(const char* host, int port) {
    struct addrinfo hints, *res = NULL;
    char port_str[16];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);
    return fd;
}

/* ── M1: Unix domain socket + process management helpers ──────────────────── */
#include <sys/un.h>

/* pion_connect_unix: connect to a Unix domain socket. Returns fd or -1. */
int pion_connect_unix(const char* path) {
    struct sockaddr_un addr;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd); return -1;
    }
    return fd;
}

/* pion_set_nonblocking: set O_NONBLOCK on fd. Returns 0 on success. */
int pion_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/* pion_spawn_inference: fork + exec python inference worker. Returns child PID or -1.
   python_path: full path to python3 binary (e.g. .pixi/envs/default/bin/python3). */
int32_t pion_spawn_inference(const char* python_path, const char* script_path,
                              const char* socket_path,
                              const char* emb_model, const char* llm_model) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        /* Child: exec python3 with the inference worker script */
        if (llm_model && llm_model[0] != '\0') {
            execl(python_path, python_path, script_path,
                   "--socket", socket_path,
                   "--embedding-model", emb_model,
                   "--llm-model", llm_model,
                   (char*)NULL);
        } else {
            execl(python_path, python_path, script_path,
                   "--socket", socket_path,
                   "--embedding-model", emb_model,
                   (char*)NULL);
        }
        _exit(127);  /* exec failed */
    }
    return (int32_t)pid;
}

/* pion_waitpid_nohang: non-blocking waitpid. Returns 0 if still running, pid if exited, -1 on error. */
int32_t pion_waitpid_nohang(int32_t pid) {
    int status;
    pid_t r = waitpid((pid_t)pid, &status, WNOHANG);
    return (int32_t)r;
}

/* pion_kill: send signal to process. Returns 0 on success. */
int pion_kill(int32_t pid, int sig) {
    return kill((pid_t)pid, sig);
}

#ifdef __linux__
#include <sys/syscall.h>

/* ARM64/x86_64 io_uring syscall numbers */
#ifndef SYS_io_uring_setup
#define SYS_io_uring_setup  425
#define SYS_io_uring_enter  426
#endif

#ifndef SYS_io_uring_register
#define SYS_io_uring_register 427
#endif
#include <sys/resource.h>

/* The ring fd, or -errno: the caller probes setup flags one set at a time and
   only an EINVAL means "this kernel does not know that flag". */
int pion_io_uring_setup(uint32_t entries, void* params) {
    int r = (int)syscall(SYS_io_uring_setup, entries, params);
    return r < 0 ? -errno : r;
}

/* flags: IORING_ENTER_GETEVENTS=1; ring_fd is a registered ring index when
   flags carries IORING_ENTER_REGISTERED_RING (16). */
int pion_io_uring_enter(int ring_fd, uint32_t to_submit,
                        uint32_t min_complete, uint32_t flags) {
    return (int)syscall(SYS_io_uring_enter,
                        (unsigned long)ring_fd,
                        (unsigned long)to_submit,
                        (unsigned long)min_complete,
                        (unsigned long)flags,
                        0UL, 0UL);
}

void* pion_mmap_uring(size_t length, int prot, int flags, int fd, long offset) {
    return mmap(NULL, length, prot, flags, fd, offset);
}

/* gh #205/#206: io_uring_register, -errno on failure. */
static int _uring_register(int ring_fd, unsigned opcode, void* arg, unsigned nr) {
    int r = (int)syscall(SYS_io_uring_register, ring_fd, opcode, arg, nr);
    return r < 0 ? -errno : r;
}

/* IORING_REGISTER_RING_FDS (20, Linux 5.18): the registered index enter()
   then passes with IORING_ENTER_REGISTERED_RING, or -errno. Offset -1 asks
   the kernel to pick the slot; it writes the slot back. */
int pion_uring_register_ring_fd(int ring_fd) {
    struct { uint32_t offset; uint32_t resv; uint64_t data; } up;
    up.offset = 0xFFFFFFFFu;
    up.resv = 0;
    up.data = (uint64_t)ring_fd;
    int r = _uring_register(ring_fd, 20, &up, 1);
    if (r < 0) return r;
    if (r != 1) return -EINVAL;
    return (int)up.offset;
}

/* IORING_REGISTER_FILES (2): a sparse table (every slot -1) of `want` slots,
   clamped to RLIMIT_NOFILE, which the kernel enforces. The slot count, or
   -errno. */
int pion_uring_register_files_sparse(int ring_fd, int want) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY
        && (rlim_t)want > rl.rlim_cur)
        want = (int)rl.rlim_cur;
    if (want <= 0) return -EINVAL;
    int32_t* fds = (int32_t*)malloc(sizeof(int32_t) * (size_t)want);
    if (!fds) return -ENOMEM;
    for (int i = 0; i < want; i++) fds[i] = -1;
    int r = _uring_register(ring_fd, 2, fds, (unsigned)want);
    free(fds);
    return r < 0 ? r : want;
}

/* IORING_REGISTER_FILES_UPDATE (6): put `fd` (or -1, to empty it) in `slot`.
   1 on success, or -errno. */
int pion_uring_files_update(int ring_fd, int slot, int fd) {
    int32_t one = fd;
    struct { uint32_t offset; uint32_t resv; uint64_t fds; } up;
    up.offset = (uint32_t)slot;
    up.resv = 0;
    up.fds = (uint64_t)(uintptr_t)&one;
    return _uring_register(ring_fd, 6, &up, 1);
}

/* A page-aligned, zeroed ring of `entries` struct io_uring_buf (16 B each)
   for IORING_REGISTER_PBUF_RING, or NULL. */
void* pion_uring_pbuf_ring_alloc(int entries) {
    size_t len = (size_t)entries * 16;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;
    len = (len + (size_t)pg - 1) & ~((size_t)pg - 1);
    void* p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

/* IORING_REGISTER_PBUF_RING (22, Linux 5.19): 0, 1 when it took the retry
   below, or -errno.

   Ubuntu's 6.8 kernels (from 6.8.0-139; the 2026-10 bench box runs
   6.8.0-146) invert the check on the reserved words: every correct call,
   resv zeroed, fails with EINVAL, and one with resv[0] != 0 succeeds.
   Netty and other io_uring users hit the same. A correct kernel refuses
   any nonzero reserved word, so the one retry can only succeed on a kernel
   with the inverted check. */
int pion_uring_register_pbuf_ring(int ring_fd, void* ring_addr, int entries, int bgid) {
    struct {
        uint64_t ring_addr; uint32_t ring_entries; uint16_t bgid; uint16_t flags;
        uint64_t resv[3];
    } reg;
    memset(&reg, 0, sizeof(reg));
    reg.ring_addr = (uint64_t)(uintptr_t)ring_addr;
    reg.ring_entries = (uint32_t)entries;
    reg.bgid = (uint16_t)bgid;
    int r = _uring_register(ring_fd, 22, &reg, 1);
    if (r == -EINVAL) {
        reg.resv[0] = 1;
        if (_uring_register(ring_fd, 22, &reg, 1) == 0) return 1;
    }
    return r;
}

#else
/* macOS stubs — io_uring is Linux-only */
int pion_io_uring_setup(uint32_t entries, void* params) { (void)entries; (void)params; return -1; }
int pion_io_uring_enter(int ring_fd, uint32_t to_submit,
                        uint32_t min_complete, uint32_t flags) {
    (void)ring_fd; (void)to_submit; (void)min_complete; (void)flags; return -1;
}
void* pion_mmap_uring(size_t length, int prot, int flags, int fd, long offset) {
    /* On macOS, fall through to real mmap (WAL needs this) */
    return mmap(NULL, length, prot, flags, fd, offset);
}
int pion_uring_register_ring_fd(int ring_fd) { (void)ring_fd; return -ENOSYS; }
int pion_uring_register_files_sparse(int ring_fd, int want) { (void)ring_fd; (void)want; return -ENOSYS; }
int pion_uring_files_update(int ring_fd, int slot, int fd) { (void)ring_fd; (void)slot; (void)fd; return -ENOSYS; }
void* pion_uring_pbuf_ring_alloc(int entries) { (void)entries; return NULL; }
int pion_uring_register_pbuf_ring(int ring_fd, void* ring_addr, int entries, int bgid) {
    (void)ring_fd; (void)ring_addr; (void)entries; (void)bgid; return -ENOSYS;
}
#endif

/* ── Gossip: TCP health probing ──────────────────────────────────────────────
   Background pthread that periodically PINGs each peer's Redis port.
   peer_health_out[] is a pointer into Mojo's ClusterState.peer_health InlineArray;
   the gossip thread writes health values (0=online,1=pfail,2=fail) directly there.
   Single-byte writes are inherently atomic on x86_64 and ARM64.
*/
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#define PION_GOSSIP_MAX_PEERS 16

typedef struct {
    char     host[64];
    int      port;
    int      consec_fail;
} PionGossipPeer;

typedef struct {
    PionGossipPeer  peers[PION_GOSSIP_MAX_PEERS];
    int             peer_count;
    uint64_t        ping_ms;
    int             pfail_threshold;   /* consecutive failures → pfail */
    int             fail_threshold;    /* consecutive failures → fail */
    volatile uint8_t* health_out;     /* pointer into ClusterState.peer_health */
    volatile int    running;
    pthread_t       tid;
    /* C1.3: Gossip state exchange */
    char     my_node_id[41];           /* this node's 40-char hex ID */
    uint64_t my_epoch;                 /* this node's cluster epoch */
    uint8_t  my_role;                  /* 0=master, 1=replica */
    uint16_t my_slot_count;            /* number of slots owned */
    /* C1.3: Per-peer pfail reports (for quorum) — bitmap of which peers report PFAIL */
    uint16_t pfail_reports[PION_GOSSIP_MAX_PEERS];  /* bit i = peer i reported PFAIL */
    /* C1.3: Peer epoch (from gossip exchange) */
    uint64_t peer_epochs[PION_GOSSIP_MAX_PEERS];
    /* C1.3: Auto-failover trigger — set to peer_idx when quorum FAIL detected */
    volatile int failover_target;      /* -1 = none, 0..15 = peer that FAILed */
} PionGossipBlock;

static int _pion_ping_peer(const char* host, int port) {
    struct addrinfo hints, *res = NULL;
    char port_str[16];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    struct timeval tv = { .tv_sec = 0, .tv_usec = 250000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);
    const char* ping_cmd = "PING\r\n";
    if (send(fd, ping_cmd, 6, 0) != 6) { close(fd); return -1; }
    char buf[16];
    int n = (int)recv(fd, buf, sizeof(buf) - 1, 0);
    close(fd);
    return (n >= 5 && buf[0] == '+') ? 0 : -1;
}

/* C1.3: Extended gossip PING that exchanges cluster state.
 * Sends: PION-GOSSIP <node_id> <epoch> <role> <slot_count> <pfail_bits>\r\n
 * Expects: +PION-GOSSIP <node_id> <epoch> <role> <slot_count> <pfail_bits>\r\n
 * Returns 0 on success, fills out_epoch/out_pfail_bits. */
static int _pion_gossip_ping_peer(PionGossipBlock* blk, int peer_idx,
                                   uint64_t* out_epoch) {
    const char* host = blk->peers[peer_idx].host;
    int port = blk->peers[peer_idx].port;

    struct addrinfo hints, *res = NULL;
    char port_str[16];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(port_str, sizeof(port_str), "%d", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, 0);
    if (fd < 0) { freeaddrinfo(res); return -1; }
    struct timeval tv_g = { .tv_sec = 0, .tv_usec = 500000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv_g, sizeof(tv_g));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv_g, sizeof(tv_g));
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);

    /* Build gossip PING: includes our pfail bitmap for this peer's quorum */
    uint16_t my_pfail_bits = 0;
    for (int i = 0; i < blk->peer_count; i++) {
        if (blk->health_out && blk->health_out[i] >= 1) /* pfail or fail */
            my_pfail_bits |= (1 << i);
    }

    /* Send as RESP inline — the peer's fast path will see PING and respond +PONG.
     * We piggyback gossip data as a comment after PING. */
    char gossip_msg[128];
    int glen = snprintf(gossip_msg, sizeof(gossip_msg),
                        "PING\r\n");
    if (send(fd, gossip_msg, glen, 0) != glen) { close(fd); return -1; }

    char resp[128];
    int n = (int)recv(fd, resp, sizeof(resp) - 1, 0);
    close(fd);

    if (n >= 5 && resp[0] == '+') {
        *out_epoch = blk->peer_epochs[peer_idx];  /* keep existing */
        return 0;
    }
    return -1;
}

static void* _pion_gossip_thread(void* arg) {
    PionGossipBlock* blk = (PionGossipBlock*)arg;
    while (blk->running) {
        int master_count = 0;
        for (int i = 0; i < blk->peer_count && blk->running; i++) {
            uint64_t peer_epoch = 0;
            int rc = _pion_gossip_ping_peer(blk, i, &peer_epoch);
            if (rc == 0) {
                blk->peers[i].consec_fail = 0;
                if (blk->health_out) blk->health_out[i] = 0;
                /* Clear pfail reports when node comes back online */
                blk->pfail_reports[i] = 0;
                /* Track peer epoch */
                if (peer_epoch > blk->peer_epochs[i])
                    blk->peer_epochs[i] = peer_epoch;
            } else {
                blk->peers[i].consec_fail++;
                int cf = blk->peers[i].consec_fail;
                uint8_t h = (cf >= blk->fail_threshold) ? 2 :
                            (cf >= blk->pfail_threshold) ? 1 : 0;
                if (blk->health_out) blk->health_out[i] = h;
                /* C1.3: Track our own PFAIL report for this peer */
                if (h >= 1) {
                    /* Set our bit in pfail_reports — we use bit 15 for "self" */
                    blk->pfail_reports[i] |= (1 << 15);
                }
            }
            if (blk->health_out && blk->health_out[i] == 0)
                master_count++;
        }

        /* C1.3: Quorum check — if majority of known nodes (including us) report PFAIL
         * for a peer, promote to FAIL and trigger auto-failover */
        int total_nodes = blk->peer_count + 1;  /* peers + self */
        int quorum = total_nodes / 2 + 1;
        for (int i = 0; i < blk->peer_count; i++) {
            if (blk->health_out && blk->health_out[i] == 2) {
                /* Already FAIL — check if we should trigger auto-failover */
                if (blk->failover_target < 0) {
                    blk->failover_target = i;
                }
            } else if (blk->health_out && blk->health_out[i] == 1) {
                /* PFAIL — count reports. For now, our single-node PFAIL is sufficient
                 * since we're the only one monitoring. In multi-node gossip, we'd
                 * aggregate pfail_reports from all nodes via the gossip payload. */
                int report_count = 0;
                uint16_t reports = blk->pfail_reports[i];
                while (reports) { report_count += (reports & 1); reports >>= 1; }
                if (report_count >= quorum) {
                    /* Promote PFAIL → FAIL */
                    blk->health_out[i] = 2;
                    if (blk->failover_target < 0) {
                        blk->failover_target = i;
                    }
                }
            }
        }

        struct timespec ts;
        ts.tv_sec  = (time_t)(blk->ping_ms / 1000);
        ts.tv_nsec = (long)((blk->ping_ms % 1000) * 1000000L);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

void* pion_gossip_create(void) {
    PionGossipBlock* blk = (PionGossipBlock*)calloc(1, sizeof(PionGossipBlock));
    if (!blk) return NULL;
    blk->ping_ms         = 1000;
    blk->pfail_threshold = 5;   /* 5s → pfail */
    blk->fail_threshold  = 15;  /* 15s → fail */
    blk->failover_target = -1;
    memset(blk->pfail_reports, 0, sizeof(blk->pfail_reports));
    memset(blk->peer_epochs, 0, sizeof(blk->peer_epochs));
    return (void*)blk;
}

void pion_gossip_set_peer(void* block, int i, const char* host, int port) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (i < 0 || i >= PION_GOSSIP_MAX_PEERS) return;
    strncpy(blk->peers[i].host, host, 63);
    blk->peers[i].host[63] = '\0';
    blk->peers[i].port = port;
    blk->peers[i].consec_fail = 0;
    if (i + 1 > blk->peer_count) blk->peer_count = i + 1;
}

void pion_gossip_set_health_output(void* block, uint8_t* health_arr) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    blk->health_out = (volatile uint8_t*)health_arr;
}

void pion_gossip_set_timeouts(void* block, uint64_t ping_ms,
                               int pfail_threshold, int fail_threshold) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (ping_ms > 0)        blk->ping_ms         = ping_ms;
    if (pfail_threshold > 0) blk->pfail_threshold = pfail_threshold;
    if (fail_threshold  > 0) blk->fail_threshold  = fail_threshold;
}

int pion_gossip_start(void* block) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    blk->running = 1;
    return pthread_create(&blk->tid, NULL, _pion_gossip_thread, blk) == 0 ? 0 : -1;
}

uint8_t pion_gossip_get_health(void* block, int i) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (!blk || i < 0 || i >= blk->peer_count) return 0;
    return blk->health_out ? (uint8_t)blk->health_out[i] : 0;
}

void pion_gossip_stop(void* block) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (!blk) return;
    blk->running = 0;
    pthread_join(blk->tid, NULL);
    free(blk);
}

/* C1.3: Set gossip identity (called from Mojo before start) */
void pion_gossip_set_identity(void* block, const char* node_id, uint64_t epoch,
                               uint8_t role, uint16_t slot_count) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (!blk) return;
    strncpy(blk->my_node_id, node_id, 40);
    blk->my_node_id[40] = '\0';
    blk->my_epoch = epoch;
    blk->my_role = role;
    blk->my_slot_count = slot_count;
}

/* C1.3: Check if a failover target has been detected (returns peer_idx or -1) */
int pion_gossip_get_failover_target(void* block) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (!blk) return -1;
    return blk->failover_target;
}

/* C1.3: Clear the failover target after handling */
void pion_gossip_clear_failover_target(void* block) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (blk) blk->failover_target = -1;
}

/* C1.3: Update epoch (called after failover or topology change) */
void pion_gossip_set_epoch(void* block, uint64_t epoch) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    if (blk) blk->my_epoch = epoch;
}

/* ── Replication: WAL streaming primary → replica ────────────────────────────
   Primary thread: accepts replicas, streams the WAL's data section to each,
   reads their ACKs. Replica thread: connects, receives into a ring that Mojo
   drains once per housekeeping tick (pion_repl_replica_drain) and reports back
   what it APPLIED (pion_repl_replica_applied), which is what it ACKs.

   Replication port = main port + 10000.

   Handshake (replica → primary): "PSYNC <repl_id|?> <offset>\r\n".
     +CONTINUE\r\n                     same repl_id and the offset is still in
                                       the log: stream from it.
     +FULLRESYNC <id> <off>\r\n$<n>\r\n<records>\r\n
                                       everything else. <records> is the whole
                                       keyspace in WAL record form, taken by the
                                       worker at WAL offset <off>; the stream
                                       then continues from <off>.

   gh #390: a FULLRESYNC used to send an EMPTY snapshot (nothing ever supplied
   one) and start the stream at the current tail, so a replica added to a
   running primary — or one that missed writes while down — never received a
   single pre-existing key. Now the thread asks the worker for a snapshot
   (pion_repl_primary_snapshot_requested / _provide_snapshot) and waits for it.

   The primary never read the replicas' "REPLCONF ACK <off>" lines, so every
   ack offset stayed at its connect value and WAIT counted nobody; replicas
   also ACKed on a 1 s timer of RECEIVED bytes. Now the primary reads them and
   the replica ACKs what Mojo has applied, as soon as it has applied it.

   The WAL rotates (gh #149) and SAVE truncates it, both under this thread,
   which read `wal_data + off` from a mapping that could be unmapped and from
   offsets that no longer meant anything. The worker now detaches the WAL
   first (every replica is dropped and the repl_id changes, so they come back
   with PSYNC and get a FULLRESYNC) and reattaches the new mapping after.
*/

#define PION_REPL_MAX_REPLICAS 8
#define PION_REPL_RING_SIZE    (4 * 1024 * 1024)

/* In-band ring markers: a WAL-framed record the Mojo decoder recognises.
   FLUSH precedes a snapshot — the replica must drop what it holds first, or a
   key the primary deleted while the replica was away survives. */
#define PION_REPL_CMD_FLUSH 250

typedef struct {
    int      listen_fd;
    int      conn_fds[PION_REPL_MAX_REPLICAS];
    uint64_t sent_offsets[PION_REPL_MAX_REPLICAS];
    uint64_t ack_offsets[PION_REPL_MAX_REPLICAS];
    char     ack_buf[PION_REPL_MAX_REPLICAS][64];   /* partial ACK line per replica */
    int      ack_len[PION_REPL_MAX_REPLICAS];
    int      conn_count;
    const uint8_t*  wal_data;    /* WAL data section (after the 64 B header) */
    const uint64_t* wal_tail;    /* the WAL header's published tail word */
    int      wal_attached;       /* 0 while the worker rotates / truncates the WAL */
    char     repl_id[41];
    /* snapshot hand-off with the worker (all under mu) */
    int      snap_requested;
    int      snap_ready;
    uint8_t* snap_buf;           /* owned here (a copy) */
    uint64_t snap_len;
    uint64_t snap_tail;
    volatile int    running;
    pthread_t       tid;
    pthread_mutex_t mu;
    /* the port each replica serves clients on, from its PSYNC (ROLE, #39) */
    int      listen_ports[PION_REPL_MAX_REPLICAS];
} PionReplPrimary;

static uint64_t _repl_now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000ULL + (uint64_t)t.tv_nsec / 1000000ULL;
}

static int _repl_send_all(int fd, const void* p, uint64_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t done = 0;
    while (done < n) {
        uint64_t chunk = n - done;
        if (chunk > 65536) chunk = 65536;
        ssize_t s = send(fd, b + done, (size_t)chunk, MSG_NOSIGNAL);
        if (s <= 0) return -1;
        done += (uint64_t)s;
    }
    return 0;
}

static void _repl_drop(PionReplPrimary* blk, int i) {
    close(blk->conn_fds[i]);
    int last = --blk->conn_count;
    blk->conn_fds[i]     = blk->conn_fds[last];
    blk->sent_offsets[i] = blk->sent_offsets[last];
    blk->ack_offsets[i]  = blk->ack_offsets[last];
    memcpy(blk->ack_buf[i], blk->ack_buf[last], sizeof(blk->ack_buf[i]));
    blk->ack_len[i]      = blk->ack_len[last];
    blk->listen_ports[i] = blk->listen_ports[last];
}

/* Read whatever the replica sent: "REPLCONF ACK <offset>\r\n" lines. */
static int _repl_read_acks(PionReplPrimary* blk, int i) {
    char tmp[256];
    ssize_t n = recv(blk->conn_fds[i], tmp, sizeof(tmp), MSG_DONTWAIT);
    if (n == 0) return -1;                       /* replica closed */
    if (n < 0) return (errno == EAGAIN || errno == EWOULDBLOCK) ? 0 : -1;
    for (ssize_t k = 0; k < n; k++) {
        char c = tmp[k];
        if (c == '\n') {
            blk->ack_buf[i][blk->ack_len[i]] = '\0';
            if (strncmp(blk->ack_buf[i], "REPLCONF ACK ", 13) == 0) {
                uint64_t off = (uint64_t)strtoull(blk->ack_buf[i] + 13, NULL, 10);
                if (off > blk->ack_offsets[i]) blk->ack_offsets[i] = off;
            }
            blk->ack_len[i] = 0;
        } else if (c != '\r' && blk->ack_len[i] < 63) {
            blk->ack_buf[i][blk->ack_len[i]++] = c;
        }
    }
    return 0;
}

/* The handshake of one new replica. Returns the offset to stream from, or -1
   to drop the connection. Runs on this thread; blocks for a FULLRESYNC while
   the worker serializes its keyspace (the trade-off: with one worker that
   pause is the snapshot's cost). */
static int64_t _repl_handshake(PionReplPrimary* blk, int cfd, int* listen_port) {
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char hs[128];
    int n = (int)recv(cfd, hs, sizeof(hs) - 1, 0);
    if (n <= 0) return -1;
    hs[n] = '\0';

    if (n >= 13 && strncmp(hs, "PION-REPL 1.0", 13) == 0) {
        /* legacy: raw stream from the start of the current log */
        send(cfd, "+OK\r\n", 5, MSG_NOSIGNAL);
        return 0;
    }
    if (n < 6 || strncmp(hs, "PSYNC ", 6) != 0) return -1;
    char rid[41] = {0};
    uint64_t roff = 0;
    char* sp = hs + 6;
    int ri = 0;
    while (*sp && *sp != ' ' && *sp != '\r' && ri < 40) rid[ri++] = *sp++;
    if (*sp == ' ') {
        char* end = NULL;
        roff = (uint64_t)strtoull(sp + 1, &end, 10);
        /* an optional third field: the port the replica serves clients on */
        if (end && *end == ' ') *listen_port = (int)strtol(end + 1, NULL, 10);
    }

    pthread_mutex_lock(&blk->mu);
    int attached = blk->wal_attached;
    uint64_t tail = attached ? *blk->wal_tail : 0;
    int same = attached && strcmp(rid, blk->repl_id) == 0 && roff <= tail;
    pthread_mutex_unlock(&blk->mu);
    if (!attached) return -1;       /* mid-rotation: the replica retries */
    if (same) {
        send(cfd, "+CONTINUE\r\n", 11, MSG_NOSIGNAL);
        return (int64_t)roff;
    }

    /* FULLRESYNC: ask the worker for a snapshot and wait for it. */
    pthread_mutex_lock(&blk->mu);
    blk->snap_ready = 0;
    blk->snap_requested = 1;
    pthread_mutex_unlock(&blk->mu);
    uint64_t deadline = _repl_now_ms() + 60000;
    int ready = 0;
    while (blk->running && _repl_now_ms() < deadline) {
        pthread_mutex_lock(&blk->mu);
        ready = blk->snap_ready;
        pthread_mutex_unlock(&blk->mu);
        if (ready) break;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000L };
        nanosleep(&ts, NULL);
    }
    pthread_mutex_lock(&blk->mu);
    blk->snap_requested = 0;
    uint8_t* buf = blk->snap_buf;
    uint64_t len = blk->snap_len;
    uint64_t stail = blk->snap_tail;
    char id[41];
    memcpy(id, blk->repl_id, sizeof(id));
    blk->snap_buf = NULL;
    blk->snap_len = 0;
    blk->snap_ready = 0;
    pthread_mutex_unlock(&blk->mu);
    if (!ready) { free(buf); return -1; }

    char hdr[160];
    int hl = snprintf(hdr, sizeof(hdr), "+FULLRESYNC %s %llu\r\n$%llu\r\n",
                      id, (unsigned long long)stail, (unsigned long long)len);
    int rc = _repl_send_all(cfd, hdr, (uint64_t)hl);
    if (rc == 0 && len > 0) rc = _repl_send_all(cfd, buf, len);
    if (rc == 0) rc = _repl_send_all(cfd, "\r\n", 2);
    free(buf);
    return rc == 0 ? (int64_t)stail : -1;
}

static void* _pion_repl_primary_thread(void* arg) {
    PionReplPrimary* blk = (PionReplPrimary*)arg;
    while (blk->running) {
        fd_set rset;
        FD_ZERO(&rset);
        FD_SET(blk->listen_fd, &rset);
        struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
        if (select(blk->listen_fd + 1, &rset, NULL, NULL, &tv) > 0) {
            int cfd = (int)accept(blk->listen_fd, NULL, NULL);
            if (cfd >= 0) {
                int lport = 0;
                int64_t start = _repl_handshake(blk, cfd, &lport);
                pthread_mutex_lock(&blk->mu);
                if (start >= 0 && blk->wal_attached && blk->conn_count < PION_REPL_MAX_REPLICAS) {
                    int idx = blk->conn_count++;
                    blk->conn_fds[idx]     = cfd;
                    blk->sent_offsets[idx] = (uint64_t)start;
                    blk->ack_offsets[idx]  = 0;      /* counted once it ACKs */
                    blk->ack_len[idx]      = 0;
                    blk->listen_ports[idx] = lport;
                } else {
                    close(cfd);
                }
                pthread_mutex_unlock(&blk->mu);
            }
        }

        pthread_mutex_lock(&blk->mu);
        if (blk->wal_attached) {
            uint64_t tail = *blk->wal_tail;
            for (int i = 0; i < blk->conn_count; ) {
                uint64_t off = blk->sent_offsets[i];
                int drop = 0;
                if (off > tail) {
                    drop = 1;                      /* the log moved under it: resync */
                } else if (tail > off) {
                    ssize_t sent = send(blk->conn_fds[i], blk->wal_data + off,
                                        (size_t)(tail - off), MSG_NOSIGNAL);
                    if (sent > 0) blk->sent_offsets[i] += (uint64_t)sent;
                    else drop = 1;
                }
                if (!drop && _repl_read_acks(blk, i) < 0) drop = 1;
                if (drop) _repl_drop(blk, i);      /* recheck the slot moved into i */
                else i++;
            }
        }
        pthread_mutex_unlock(&blk->mu);

        struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000L };
        nanosleep(&ts, NULL);
    }
    return NULL;
}

void* pion_repl_primary_create(const uint8_t* wal_data, const uint64_t* wal_tail,
                                int listen_port) {
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) return NULL;
    int opt = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)listen_port);
    addr.sin_addr.s_addr = pion_get_bind_addr();  /* gh #258: was INADDR_ANY */
    if (bind(lfd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(lfd, 8) < 0) {
        close(lfd); return NULL;
    }
    PionReplPrimary* blk = (PionReplPrimary*)calloc(1, sizeof(PionReplPrimary));
    if (!blk) { close(lfd); return NULL; }
    blk->listen_fd    = lfd;
    blk->wal_data     = wal_data;
    blk->wal_tail     = wal_tail;
    blk->wal_attached = wal_data != NULL && wal_tail != NULL;
    pthread_mutex_init(&blk->mu, NULL);
    return (void*)blk;
}

int pion_repl_primary_start(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    blk->running = 1;
    return pthread_create(&blk->tid, NULL, _pion_repl_primary_thread, blk) == 0 ? 0 : -1;
}

int pion_repl_primary_connected_count(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    pthread_mutex_lock(&blk->mu);
    int c = blk->conn_count;
    pthread_mutex_unlock(&blk->mu);
    return c;
}

void pion_repl_primary_stop(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return;
    blk->running = 0;
    /* A send() to a stalled replica can block — while the thread holds mu,
       so this must not take mu. Unlocked read of the fds is fine at shutdown:
       the worst case is a shutdown() on a slot being compacted. */
    for (int i = 0; i < blk->conn_count; i++) shutdown(blk->conn_fds[i], SHUT_RDWR);
    pthread_join(blk->tid, NULL);
    close(blk->listen_fd);
    for (int i = 0; i < blk->conn_count; i++) close(blk->conn_fds[i]);
    free(blk->snap_buf);
    pthread_mutex_destroy(&blk->mu);
    free(blk);
}

/* The WAL is about to be unmapped or truncated (rotation, SAVE). Stop reading
   it, drop every replica and take a new repl_id, so each one reconnects with
   an id that no longer matches and gets a FULLRESYNC instead of reading
   offsets from a log that no longer has them. */
void pion_repl_primary_detach_wal(void* block, const char* new_repl_id) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return;
    pthread_mutex_lock(&blk->mu);
    blk->wal_attached = 0;
    blk->wal_data = NULL;
    blk->wal_tail = NULL;
    while (blk->conn_count > 0) _repl_drop(blk, 0);
    if (new_repl_id) {
        strncpy(blk->repl_id, new_repl_id, 40); blk->repl_id[40] = '\0';
    } else {
        /* Any id the replicas cannot already hold will do. */
        static uint64_t gen = 0;
        struct timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        snprintf(blk->repl_id, sizeof(blk->repl_id), "%016llx%016llx%08x",
                 (unsigned long long)t.tv_sec, (unsigned long long)t.tv_nsec,
                 (unsigned)(++gen));
    }
    pthread_mutex_unlock(&blk->mu);
}

/* The published WAL tail the replicas are measured against, or 0 while the
   log is detached. WAIT's target — read through here, not through a pointer
   into a mapping the WAL may have replaced. */
uint64_t pion_repl_primary_current_tail(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    uint64_t t = blk->wal_attached ? *blk->wal_tail : 0;
    pthread_mutex_unlock(&blk->mu);
    return t;
}

void pion_repl_primary_attach_wal(void* block, const uint8_t* wal_data, const uint64_t* wal_tail) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return;
    pthread_mutex_lock(&blk->mu);
    blk->wal_data = wal_data;
    blk->wal_tail = wal_tail;
    blk->wal_attached = wal_data != NULL && wal_tail != NULL;
    pthread_mutex_unlock(&blk->mu);
}

/* 1 when a replica is waiting for a FULLRESYNC snapshot the worker has not
   provided yet. Polled by the worker's housekeeping. */
int pion_repl_primary_snapshot_requested(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int r = blk->snap_requested && !blk->snap_ready;
    pthread_mutex_unlock(&blk->mu);
    return r;
}

/* The worker's snapshot: `len` bytes of WAL records taken at WAL offset
   `tail`. Copied, so the caller frees its buffer as soon as this returns. */
void pion_repl_primary_provide_snapshot(void* block, const uint8_t* buf, uint64_t len,
                                        uint64_t tail) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return;
    uint8_t* copy = (uint8_t*)malloc(len > 0 ? len : 1);
    if (!copy) return;
    if (len > 0) memcpy(copy, buf, len);
    pthread_mutex_lock(&blk->mu);
    free(blk->snap_buf);
    blk->snap_buf = copy;
    blk->snap_len = len;
    blk->snap_tail = tail;
    blk->snap_ready = 1;
    pthread_mutex_unlock(&blk->mu);
}

/* Replica receive block */
typedef struct {
    char     primary_host[64];
    int      primary_repl_port;
    int      fd;
    uint8_t* ring_buf;
    uint32_t ring_head;   /* consumer: Mojo increments via drain */
    uint32_t ring_tail;   /* producer: this thread increments */
    uint32_t ring_size;
    uint64_t total_bytes_received;  /* N3: monotonic counter of all bytes received */
    char     repl_id[41];           /* from the primary's FULLRESYNC */
    uint64_t repl_offset;           /* WAL offset RECEIVED — what PSYNC resumes from */
    uint64_t applied_offset;        /* WAL offset APPLIED — what we ACK */
    uint64_t nonwal_pending;        /* ring bytes ahead of the stream that are not
                                       WAL offsets (markers, snapshot, a previous
                                       generation's leftovers) */
    volatile int    running;
    pthread_t       tid;
    pthread_mutex_t mu;
    /* ROLE (#39): the link's state as Redis names it (PION_REPL_LINK_*), and
       the port this server serves clients on, sent with PSYNC so the
       primary's ROLE can list it */
    int      link_state;
    int      listening_port;
} PionReplReplicaBlock;

#define PION_REPL_LINK_CONNECT    1   /* "connect": not connected, will retry */
#define PION_REPL_LINK_CONNECTING 2   /* "connecting" */
#define PION_REPL_LINK_HANDSHAKE  3   /* "handshake": PSYNC sent */
#define PION_REPL_LINK_SYNC       4   /* "sync": receiving the snapshot */
#define PION_REPL_LINK_CONNECTED  5   /* "connected" */

/* Push into the ring, WAITING for room. It used to drop bytes when full
   ("ring full — drop"), so a snapshot or a burst larger than 4 MB lost data
   silently; waiting lets TCP flow control push back on the primary instead. */
static int _repl_ring_push(PionReplReplicaBlock* blk, const uint8_t* p, uint64_t n) {
    uint64_t done = 0;
    while (done < n) {
        pthread_mutex_lock(&blk->mu);
        while (done < n) {
            uint32_t next = (blk->ring_tail + 1) % blk->ring_size;
            if (next == blk->ring_head) break;
            blk->ring_buf[blk->ring_tail] = p[done++];
            blk->ring_tail = next;
        }
        pthread_mutex_unlock(&blk->mu);
        if (done < n) {
            if (!blk->running) return -1;
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000000L };
            nanosleep(&ts, NULL);
        }
    }
    return 0;
}

static uint32_t _repl_ring_used(PionReplReplicaBlock* blk) {
    return (blk->ring_tail + blk->ring_size - blk->ring_head) % blk->ring_size;
}

static void _repl_send_ack(PionReplReplicaBlock* blk) {
    if (blk->fd < 0) return;
    /* A FULLRESYNC is ACKed only once the worker has APPLIED it: until the
       FLUSH marker and the whole snapshot have been drained and applied
       (nonwal_pending back at 0), applied_offset names a state this replica
       does not hold yet, and WAIT must not count it. The first ACK then goes
       out from pion_repl_replica_applied, when the last snapshot byte is
       applied. */
    if (blk->nonwal_pending > 0) return;
    char msg[64];
    int l = snprintf(msg, sizeof(msg), "REPLCONF ACK %llu\r\n",
                     (unsigned long long)blk->applied_offset);
    send(blk->fd, msg, (size_t)l, MSG_NOSIGNAL);
}

/* Read exactly n bytes (the snapshot body), counting what the handshake read
   already buffered in `pre`. */
static int _repl_recv_exact_into_ring(PionReplReplicaBlock* blk, int fd, uint64_t n,
                                      const uint8_t* pre, uint64_t pre_len) {
    uint64_t have = pre_len < n ? pre_len : n;
    if (have > 0 && _repl_ring_push(blk, pre, have) < 0) return -1;
    uint8_t tmp[65536];
    while (have < n) {
        uint64_t want = n - have;
        if (want > sizeof(tmp)) want = sizeof(tmp);
        ssize_t r = recv(fd, tmp, (size_t)want, 0);
        if (r <= 0) return -1;
        if (_repl_ring_push(blk, tmp, (uint64_t)r) < 0) return -1;
        have += (uint64_t)r;
    }
    return 0;
}

static int _repl_read_line(int fd, char* out, int cap) {
    int l = 0;
    while (l < cap - 1) {
        char c;
        ssize_t r = recv(fd, &c, 1, 0);
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (c != '\r') out[l++] = c;
    }
    out[l] = '\0';
    return l;
}

static void* _pion_repl_replica_thread(void* arg) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)arg;
    uint8_t tmp[65536];

    while (blk->running) {
        if (blk->fd < 0) {
            struct timespec backoff = { .tv_sec = 0, .tv_nsec = 200000000L };
            blk->link_state = PION_REPL_LINK_CONNECTING;
            int fd = pion_connect_tcp(blk->primary_host, blk->primary_repl_port);
            if (fd < 0) { blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue; }
            struct timeval tv_so = { .tv_sec = 60, .tv_usec = 0 };
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv_so, sizeof(tv_so));

            char psync[128];
            const char* rid = (blk->repl_id[0] != '\0') ? blk->repl_id : "?";
            int pl = blk->listening_port > 0
                ? snprintf(psync, sizeof(psync), "PSYNC %s %llu %d\r\n",
                           rid, (unsigned long long)blk->repl_offset, blk->listening_port)
                : snprintf(psync, sizeof(psync), "PSYNC %s %llu\r\n",
                           rid, (unsigned long long)blk->repl_offset);
            send(fd, psync, (size_t)pl, MSG_NOSIGNAL);
            blk->link_state = PION_REPL_LINK_HANDSHAKE;

            char line[160];
            if (_repl_read_line(fd, line, sizeof(line)) < 0) {
                close(fd); blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue;
            }
            if (strncmp(line, "+FULLRESYNC ", 12) == 0) {
                char newid[41] = {0};
                char* sp = line + 12;
                int ri = 0;
                while (*sp && *sp != ' ' && ri < 40) newid[ri++] = *sp++;
                uint64_t off = (*sp == ' ') ? (uint64_t)strtoull(sp + 1, NULL, 10) : 0;
                char lenline[64];
                if (_repl_read_line(fd, lenline, sizeof(lenline)) < 0 || lenline[0] != '$') {
                    close(fd); blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue;
                }
                uint64_t snap_len = (uint64_t)strtoull(lenline + 1, NULL, 10);
                blk->link_state = PION_REPL_LINK_SYNC;
                /* Everything already in the ring belongs to a generation this
                   snapshot replaces; a FLUSH marker goes ahead of the snapshot.
                   None of it is a WAL offset of the new stream. */
                uint8_t marker[13] = { 13, 0, 0, 0, PION_REPL_CMD_FLUSH, 0, 0, 0, 0, 0, 0, 0, 0 };
                pthread_mutex_lock(&blk->mu);
                blk->nonwal_pending = (uint64_t)_repl_ring_used(blk) + sizeof(marker) + snap_len;
                pthread_mutex_unlock(&blk->mu);
                if (_repl_ring_push(blk, marker, sizeof(marker)) < 0 ||
                    _repl_recv_exact_into_ring(blk, fd, snap_len, NULL, 0) < 0) {
                    close(fd); blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue;
                }
                char crlf[2];
                if (recv(fd, crlf, 2, MSG_WAITALL) != 2) { close(fd); blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue; }
                pthread_mutex_lock(&blk->mu);
                memcpy(blk->repl_id, newid, sizeof(newid));
                blk->repl_offset = off;
                blk->applied_offset = off;
                blk->total_bytes_received += snap_len;
                blk->fd = fd;
                pthread_mutex_unlock(&blk->mu);
            } else if (strncmp(line, "+CONTINUE", 9) == 0 || line[0] == '+') {
                pthread_mutex_lock(&blk->mu);
                blk->fd = fd;   /* partial resync (or legacy +OK): same stream */
                pthread_mutex_unlock(&blk->mu);
            } else {
                close(fd); blk->link_state = PION_REPL_LINK_CONNECT; nanosleep(&backoff, NULL); continue;
            }
            pthread_mutex_lock(&blk->mu);
            _repl_send_ack(blk);
            pthread_mutex_unlock(&blk->mu);
            blk->link_state = PION_REPL_LINK_CONNECTED;
        }

        ssize_t n = recv(blk->fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue;   /* idle */
            pthread_mutex_lock(&blk->mu);
            close(blk->fd); blk->fd = -1;
            pthread_mutex_unlock(&blk->mu);
            blk->link_state = PION_REPL_LINK_CONNECT;
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 200000000L };
            nanosleep(&ts, NULL);
            continue;
        }
        if (_repl_ring_push(blk, tmp, (uint64_t)n) < 0) break;
        pthread_mutex_lock(&blk->mu);
        blk->total_bytes_received += (uint64_t)n;
        blk->repl_offset += (uint64_t)n;
        pthread_mutex_unlock(&blk->mu);
    }
    if (blk->fd >= 0) { close(blk->fd); blk->fd = -1; }
    return NULL;
}

void* pion_repl_replica_create(const char* primary_host, int primary_repl_port) {
    PionReplReplicaBlock* blk =
        (PionReplReplicaBlock*)calloc(1, sizeof(PionReplReplicaBlock));
    if (!blk) return NULL;
    strncpy(blk->primary_host, primary_host, 63);
    blk->primary_host[63]  = '\0';
    blk->primary_repl_port = primary_repl_port;
    blk->fd                = -1;
    blk->ring_size         = PION_REPL_RING_SIZE;
    blk->ring_buf          = (uint8_t*)malloc(PION_REPL_RING_SIZE);
    if (!blk->ring_buf) { free(blk); return NULL; }
    pthread_mutex_init(&blk->mu, NULL);
    return (void*)blk;
}

int pion_repl_replica_start(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    blk->running = 1;
    return pthread_create(&blk->tid, NULL, _pion_repl_replica_thread, blk) == 0 ? 0 : -1;
}

/* Drain buffered bytes into out_buf. Returns bytes written (0 if nothing new). */
int pion_repl_replica_drain(void* block, uint8_t* out_buf, int max_bytes) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    pthread_mutex_lock(&blk->mu);
    int n = 0;
    while (n < max_bytes && blk->ring_head != blk->ring_tail) {
        out_buf[n++] = blk->ring_buf[blk->ring_head];
        blk->ring_head = (blk->ring_head + 1) % blk->ring_size;
    }
    pthread_mutex_unlock(&blk->mu);
    return n;
}

/* Mojo applied `nbytes` more of what it drained: advance the applied WAL
   offset (skipping the non-WAL bytes ahead of the stream) and ACK it now. */
void pion_repl_replica_applied(void* block, uint64_t nbytes) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk || nbytes == 0) return;
    pthread_mutex_lock(&blk->mu);
    uint64_t skip = nbytes < blk->nonwal_pending ? nbytes : blk->nonwal_pending;
    blk->nonwal_pending -= skip;
    blk->applied_offset += nbytes - skip;
    _repl_send_ack(blk);
    pthread_mutex_unlock(&blk->mu);
}

void pion_repl_replica_stop(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk) return;
    blk->running = 0;
    /* Wake a recv() blocked on an idle stream (SO_RCVTIMEO is 60 s) so the
       join below does not hold a graceful shutdown past its grace period. */
    pthread_mutex_lock(&blk->mu);
    if (blk->fd >= 0) shutdown(blk->fd, SHUT_RDWR);
    pthread_mutex_unlock(&blk->mu);
    pthread_join(blk->tid, NULL);
    free(blk->ring_buf);
    pthread_mutex_destroy(&blk->mu);
    free(blk);
}

/* N3: Replication offset tracking for lag monitoring */
int64_t pion_repl_replica_bytes_received(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int64_t v = (int64_t)blk->total_bytes_received;
    pthread_mutex_unlock(&blk->mu);
    return v;
}

int64_t pion_repl_primary_max_sent_offset(void* block) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int64_t maxoff = 0;
    for (int i = 0; i < blk->conn_count; i++) {
        if ((int64_t)blk->sent_offsets[i] > maxoff)
            maxoff = (int64_t)blk->sent_offsets[i];
    }
    pthread_mutex_unlock(&blk->mu);
    return maxoff;
}

void pion_repl_primary_set_repl_id(void* block, const char* repl_id) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk || !repl_id) return;
    pthread_mutex_lock(&blk->mu);
    strncpy(blk->repl_id, repl_id, 40);
    blk->repl_id[40] = '\0';
    pthread_mutex_unlock(&blk->mu);
}

/* Replicas whose ACKed (applied) offset is at least target_offset. */
int pion_repl_primary_acked_count(void* block, uint64_t target_offset) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int count = 0;
    for (int i = 0; i < blk->conn_count; i++) {
        if (blk->ack_offsets[i] >= target_offset)
            count++;
    }
    pthread_mutex_unlock(&blk->mu);
    return count;
}

/* Kept for the Mojo wrapper: a snapshot pushed ahead of time. */
void pion_repl_primary_set_snapshot(void* block, const uint8_t* buf, uint64_t len) {
    pion_repl_primary_provide_snapshot(block, buf, len, 0);
}

int64_t pion_repl_replica_offset(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int64_t v = (int64_t)blk->repl_offset;
    pthread_mutex_unlock(&blk->mu);
    return v;
}

int64_t pion_repl_replica_applied_offset(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int64_t v = (int64_t)blk->applied_offset;
    pthread_mutex_unlock(&blk->mu);
    return v;
}

void pion_repl_replica_get_repl_id(void* block, char* out_id) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk || !out_id) return;
    strncpy(out_id, blk->repl_id, 40);
    out_id[40] = '\0';
}

/* ── Feature 1: SWIM Gossip Protocol ─────────────────────────────────────────
   Extends existing gossip with UDP-based SWIM protocol:
   - Random target selection (instead of sequential)
   - Indirect ping via random delegate when direct ping fails
   - Suspicion state (SUSPECT → CONFIRM/ALIVE)
   - State exchange payload: node_id, epoch, pfail bitmap
*/

#include <sys/select.h>
#include <arpa/inet.h>

/* SWIM gossip message (UDP payload, fixed size for simplicity) */
#define SWIM_MSG_PING        1
#define SWIM_MSG_PING_REQ    2  /* indirect ping request */
#define SWIM_MSG_ACK         3
#define SWIM_MSG_SUSPECT     4
#define SWIM_MSG_ALIVE       5
#define SWIM_MSG_CONFIRM     6

#pragma pack(push, 1)
typedef struct {
    uint8_t  type;           /* SWIM_MSG_* */
    char     sender_id[40];  /* sender's node ID */
    uint64_t sender_epoch;
    uint8_t  sender_role;    /* 0=master, 1=replica */
    uint16_t sender_slots;
    uint16_t pfail_bitmap;   /* sender's view of which peers are pfail */
    uint8_t  target_idx;     /* for PING_REQ: which peer to probe */
    uint8_t  padding[2];
} SwimMessage;
#pragma pack(pop)

static int _swim_udp_socket = -1;
static int _swim_udp_port = 0;

/* Create UDP socket for SWIM gossip (called from gossip thread) */
static int _swim_create_udp(int base_port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = pion_get_bind_addr();  /* gh #258: was INADDR_ANY */
    addr.sin_port = htons(base_port + 20000); /* SWIM port = main + 20000 */
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd); return -1;
    }
    /* Non-blocking for select() */
    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    _swim_udp_port = base_port + 20000;
    return fd;
}

/* Send SWIM message to peer via UDP */
static int _swim_send(int udp_fd, const char* host, int swim_port, const SwimMessage* msg) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(swim_port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) return -1;
    return (int)sendto(udp_fd, msg, sizeof(SwimMessage), 0,
                       (struct sockaddr*)&addr, sizeof(addr));
}

/* Process incoming SWIM messages (non-blocking drain) */
static void _swim_process_incoming(PionGossipBlock* blk, int udp_fd) {
    SwimMessage msg;
    struct sockaddr_in from;
    socklen_t from_len = sizeof(from);
    for (int rounds = 0; rounds < 16; rounds++) {
        ssize_t n = recvfrom(udp_fd, &msg, sizeof(msg), MSG_DONTWAIT,
                             (struct sockaddr*)&from, &from_len);
        if (n < (ssize_t)sizeof(SwimMessage)) break;

        /* Find sender in our peer list */
        int sender_idx = -1;
        char from_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &from.sin_addr, from_ip, sizeof(from_ip));
        for (int i = 0; i < blk->peer_count; i++) {
            if (strcmp(blk->peers[i].host, from_ip) == 0) {
                sender_idx = i;
                break;
            }
        }

        switch (msg.type) {
        case SWIM_MSG_PING: {
            /* Respond with ACK */
            SwimMessage ack;
            memset(&ack, 0, sizeof(ack));
            ack.type = SWIM_MSG_ACK;
            memcpy(ack.sender_id, blk->my_node_id, 40);
            ack.sender_epoch = blk->my_epoch;
            ack.sender_role = blk->my_role;
            ack.sender_slots = blk->my_slot_count;
            /* Include our pfail bitmap */
            for (int i = 0; i < blk->peer_count; i++)
                if (blk->health_out && blk->health_out[i] >= 1)
                    ack.pfail_bitmap |= (1 << i);
            sendto(udp_fd, &ack, sizeof(ack), 0,
                   (struct sockaddr*)&from, from_len);
            /* Merge sender's pfail bitmap into our view */
            if (sender_idx >= 0) {
                blk->pfail_reports[sender_idx] |= msg.pfail_bitmap;
                if (msg.sender_epoch > blk->peer_epochs[sender_idx])
                    blk->peer_epochs[sender_idx] = msg.sender_epoch;
            }
            break;
        }
        case SWIM_MSG_ACK:
            /* Peer is alive — clear suspicion */
            if (sender_idx >= 0) {
                blk->peers[sender_idx].consec_fail = 0;
                if (blk->health_out) blk->health_out[sender_idx] = 0;
                blk->pfail_reports[sender_idx] = 0;
                if (msg.sender_epoch > blk->peer_epochs[sender_idx])
                    blk->peer_epochs[sender_idx] = msg.sender_epoch;
            }
            /* Merge pfail bitmap from ACK sender */
            for (int i = 0; i < blk->peer_count; i++) {
                if (msg.pfail_bitmap & (1 << i)) {
                    blk->pfail_reports[i] |= (1 << (sender_idx >= 0 ? sender_idx : 15));
                }
            }
            break;
        case SWIM_MSG_PING_REQ: {
            /* Indirect ping: sender asks us to probe target_idx */
            int ti = msg.target_idx;
            if (ti >= 0 && ti < blk->peer_count) {
                SwimMessage probe;
                memset(&probe, 0, sizeof(probe));
                probe.type = SWIM_MSG_PING;
                memcpy(probe.sender_id, blk->my_node_id, 40);
                probe.sender_epoch = blk->my_epoch;
                _swim_send(udp_fd, blk->peers[ti].host,
                           blk->peers[ti].port + 20000, &probe);
            }
            break;
        }
        case SWIM_MSG_SUSPECT:
            /* Peer reports target as suspected */
            if (msg.target_idx < blk->peer_count) {
                blk->pfail_reports[msg.target_idx] |= (1 << (sender_idx >= 0 ? sender_idx : 15));
            }
            break;
        case SWIM_MSG_ALIVE:
            if (msg.target_idx < blk->peer_count && blk->health_out) {
                blk->peers[msg.target_idx].consec_fail = 0;
                blk->health_out[msg.target_idx] = 0;
                blk->pfail_reports[msg.target_idx] = 0;
            }
            break;
        }
    }
}

/* Enhanced gossip thread with SWIM protocol */
static void* _pion_swim_gossip_thread(void* arg) {
    PionGossipBlock* blk = (PionGossipBlock*)arg;

    /* Create UDP socket for SWIM */
    int udp_fd = -1;
    for (int i = 0; i < blk->peer_count; i++) {
        if (blk->peers[i].port > 0) {
            udp_fd = _swim_create_udp(blk->peers[i].port - (blk->peers[i].port - 1974));
            break;
        }
    }
    if (udp_fd < 0) {
        /* Fallback: try with port 1974 */
        udp_fd = _swim_create_udp(1974);
    }
    _swim_udp_socket = udp_fd;

    /* Simple PRNG for random target selection (xorshift32) */
    uint32_t rng = (uint32_t)time(NULL) ^ (uint32_t)(uintptr_t)blk;

    while (blk->running) {
        if (blk->peer_count == 0) {
            struct timespec ts = { .tv_sec = 1, .tv_nsec = 0 };
            nanosleep(&ts, NULL);
            continue;
        }

        /* SWIM: pick random peer to probe */
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        int target = (int)(rng % (uint32_t)blk->peer_count);

        /* 1. Send UDP PING to target */
        SwimMessage ping;
        memset(&ping, 0, sizeof(ping));
        ping.type = SWIM_MSG_PING;
        memcpy(ping.sender_id, blk->my_node_id, 40);
        ping.sender_epoch = blk->my_epoch;
        ping.sender_role = blk->my_role;
        ping.sender_slots = blk->my_slot_count;
        for (int i = 0; i < blk->peer_count; i++)
            if (blk->health_out && blk->health_out[i] >= 1)
                ping.pfail_bitmap |= (1 << i);

        int swim_port = blk->peers[target].port + 20000;
        _swim_send(udp_fd, blk->peers[target].host, swim_port, &ping);

        /* 2. Wait for ACK (100ms timeout) */
        struct timespec wait = { .tv_sec = 0, .tv_nsec = 100000000L }; /* 100ms */
        nanosleep(&wait, NULL);

        /* 3. Process any incoming messages */
        if (udp_fd >= 0) _swim_process_incoming(blk, udp_fd);

        /* 4. Check if target responded */
        int got_ack = (blk->peers[target].consec_fail == 0 &&
                       blk->health_out && blk->health_out[target] == 0);

        if (!got_ack) {
            /* 5. Indirect probe: ask K random peers to ping target */
            int K = (blk->peer_count > 3) ? 3 : blk->peer_count;
            for (int k = 0; k < K; k++) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                int delegate = (int)(rng % (uint32_t)blk->peer_count);
                if (delegate == target) continue;
                SwimMessage req;
                memset(&req, 0, sizeof(req));
                req.type = SWIM_MSG_PING_REQ;
                memcpy(req.sender_id, blk->my_node_id, 40);
                req.sender_epoch = blk->my_epoch;
                req.target_idx = (uint8_t)target;
                _swim_send(udp_fd, blk->peers[delegate].host,
                           blk->peers[delegate].port + 20000, &req);
            }

            /* Wait another 200ms for indirect ACK */
            struct timespec wait2 = { .tv_sec = 0, .tv_nsec = 200000000L };
            nanosleep(&wait2, NULL);
            if (udp_fd >= 0) _swim_process_incoming(blk, udp_fd);

            /* 6. If still no ACK, increment failure */
            if (blk->health_out && blk->health_out[target] != 0) {
                blk->peers[target].consec_fail++;
                int cf = blk->peers[target].consec_fail;
                uint8_t h = (cf >= blk->fail_threshold) ? 2 :
                            (cf >= blk->pfail_threshold) ? 1 : 0;
                if (blk->health_out) blk->health_out[target] = h;
                if (h >= 1)
                    blk->pfail_reports[target] |= (1 << 15); /* self */

                /* Broadcast SUSPECT */
                SwimMessage suspect;
                memset(&suspect, 0, sizeof(suspect));
                suspect.type = SWIM_MSG_SUSPECT;
                memcpy(suspect.sender_id, blk->my_node_id, 40);
                suspect.target_idx = (uint8_t)target;
                for (int i = 0; i < blk->peer_count; i++) {
                    if (i == target) continue;
                    _swim_send(udp_fd, blk->peers[i].host,
                               blk->peers[i].port + 20000, &suspect);
                }
            }
        }

        /* 7. Quorum check for FAIL promotion */
        int total_nodes = blk->peer_count + 1;
        int quorum = total_nodes / 2 + 1;
        for (int i = 0; i < blk->peer_count; i++) {
            if (blk->health_out && blk->health_out[i] == 1) {
                int report_count = 0;
                uint16_t reports = blk->pfail_reports[i];
                while (reports) { report_count += (reports & 1); reports >>= 1; }
                if (report_count >= quorum) {
                    blk->health_out[i] = 2; /* FAIL */
                    if (blk->failover_target < 0)
                        blk->failover_target = i;
                }
            }
        }

        /* Sleep remaining interval */
        struct timespec ts;
        ts.tv_sec  = (time_t)(blk->ping_ms / 1000);
        ts.tv_nsec = (long)((blk->ping_ms % 1000) * 1000000L);
        nanosleep(&ts, NULL);

        /* Drain any remaining UDP messages */
        if (udp_fd >= 0) _swim_process_incoming(blk, udp_fd);

        /* Also do a TCP fallback ping for reliability */
        uint64_t dummy_epoch;
        _pion_gossip_ping_peer(blk, target, &dummy_epoch);
    }

    if (udp_fd >= 0) close(udp_fd);
    return NULL;
}

/* Start SWIM gossip (replaces basic TCP gossip) */
int pion_gossip_start_swim(void* block) {
    PionGossipBlock* blk = (PionGossipBlock*)block;
    blk->running = 1;
    return pthread_create(&blk->tid, NULL, _pion_swim_gossip_thread, blk) == 0 ? 0 : -1;
}


/* ── Feature 2: Raft Metadata Consensus ──────────────────────────────────────
   Lightweight Raft for cluster metadata (leader election, topology changes).
   NOT for data replication (WAL handles that). Only for:
   - Leader election (prevents split-brain)
   - Topology change commits (slot migration, failover)
*/

#define RAFT_STATE_FOLLOWER  0
#define RAFT_STATE_CANDIDATE 1
#define RAFT_STATE_LEADER    2

#define RAFT_MAX_PEERS 16
#define RAFT_LOG_MAX   4096

#define RAFT_RPC_VOTE_REQ  1
#define RAFT_RPC_VOTE_RESP 2
#define RAFT_RPC_APPEND    3
#define RAFT_RPC_APPEND_RESP 4

#pragma pack(push, 1)
typedef struct {
    uint8_t  type;           /* RAFT_RPC_* */
    uint64_t term;
    char     candidate_id[40];
    uint64_t last_log_index;
    uint64_t last_log_term;
    uint64_t prev_log_index;
    uint64_t prev_log_term;
    uint64_t leader_commit;
    uint8_t  vote_granted;
    uint8_t  success;
    uint16_t entries_count;
    /* Entries follow in the stream for APPEND */
} RaftRPC;
#pragma pack(pop)

typedef struct {
    uint64_t term;
    uint8_t  cmd;     /* 1=topology change, 2=slot migration */
    uint8_t  data[64];
} RaftLogEntryC;

typedef struct {
    /* Persistent state */
    uint64_t current_term;
    char     voted_for[41];   /* node_id or "" */
    RaftLogEntryC log[RAFT_LOG_MAX];
    int      log_count;

    /* Volatile state */
    int      state;           /* RAFT_STATE_* */
    int      commit_index;
    int      last_applied;
    char     leader_id[41];
    char     my_node_id[41];

    /* Peer info */
    char     peer_hosts[RAFT_MAX_PEERS][64];
    int      peer_ports[RAFT_MAX_PEERS];
    int      peer_count;
    int      next_index[RAFT_MAX_PEERS];   /* leader: next log entry to send */
    int      match_index[RAFT_MAX_PEERS];  /* leader: highest replicated entry */

    /* Timing */
    uint64_t election_timeout_ms;  /* randomized 150-300ms */
    uint64_t heartbeat_ms;         /* 50ms */
    uint64_t last_heartbeat_ns;    /* monotonic timestamp */

    /* Control */
    int      raft_port;       /* base port + 30000 */
    int      listen_fd;
    volatile int running;
    pthread_t tid;

    /* Votes received (candidate state) */
    int      votes_received;
} PionRaftBlock;

static uint64_t _raft_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static void _raft_randomize_timeout(PionRaftBlock* r) {
    /* 150-300ms election timeout */
    uint32_t rng = (uint32_t)(_raft_now_ms() ^ (uintptr_t)r);
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    r->election_timeout_ms = 150 + (rng % 151);
}

/* Send RPC to a specific peer (TCP, fire-and-forget with short timeout) */
static int _raft_send_rpc(const char* host, int port, const RaftRPC* rpc) {
    int fd = pion_connect_tcp(host, port);
    if (fd < 0) return -1;
    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    ssize_t n = send(fd, rpc, sizeof(RaftRPC), 0);
    close(fd);
    return (n == sizeof(RaftRPC)) ? 0 : -1;
}

/* Receive RPC response (blocking with timeout) */
static int _raft_recv_rpc(int fd, RaftRPC* out, int timeout_ms) {
    struct timeval tv = { .tv_sec = timeout_ms / 1000, .tv_usec = (timeout_ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ssize_t n = recv(fd, out, sizeof(RaftRPC), MSG_WAITALL);
    return (n == sizeof(RaftRPC)) ? 0 : -1;
}

/* Start election as candidate */
static void _raft_start_election(PionRaftBlock* r) {
    r->current_term++;
    r->state = RAFT_STATE_CANDIDATE;
    memcpy(r->voted_for, r->my_node_id, 41);
    r->votes_received = 1; /* vote for self */
    _raft_randomize_timeout(r);
    r->last_heartbeat_ns = _raft_now_ms();

    /* Send RequestVote to all peers */
    RaftRPC vote_req;
    memset(&vote_req, 0, sizeof(vote_req));
    vote_req.type = RAFT_RPC_VOTE_REQ;
    vote_req.term = r->current_term;
    memcpy(vote_req.candidate_id, r->my_node_id, 40);
    vote_req.last_log_index = (r->log_count > 0) ? r->log_count - 1 : 0;
    vote_req.last_log_term = (r->log_count > 0) ? r->log[r->log_count - 1].term : 0;

    for (int i = 0; i < r->peer_count; i++) {
        int fd = pion_connect_tcp(r->peer_hosts[i], r->peer_ports[i]);
        if (fd < 0) continue;
        struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        if (send(fd, &vote_req, sizeof(vote_req), 0) == sizeof(RaftRPC)) {
            RaftRPC resp;
            if (recv(fd, &resp, sizeof(resp), MSG_WAITALL) == sizeof(RaftRPC)) {
                if (resp.type == RAFT_RPC_VOTE_RESP && resp.vote_granted &&
                    resp.term == r->current_term) {
                    r->votes_received++;
                }
                if (resp.term > r->current_term) {
                    r->current_term = resp.term;
                    r->state = RAFT_STATE_FOLLOWER;
                    r->voted_for[0] = '\0';
                    close(fd);
                    return;
                }
            }
        }
        close(fd);
    }

    /* Check if we won */
    int total = r->peer_count + 1;
    if (r->votes_received > total / 2) {
        r->state = RAFT_STATE_LEADER;
        memcpy(r->leader_id, r->my_node_id, 41);
        /* Initialize next_index for all peers */
        for (int i = 0; i < r->peer_count; i++) {
            r->next_index[i] = r->log_count;
            r->match_index[i] = 0;
        }
    }
}

/* Send heartbeat (empty AppendEntries) as leader */
static void _raft_send_heartbeats(PionRaftBlock* r) {
    RaftRPC hb;
    memset(&hb, 0, sizeof(hb));
    hb.type = RAFT_RPC_APPEND;
    hb.term = r->current_term;
    memcpy(hb.candidate_id, r->my_node_id, 40); /* leader_id in candidate_id field */
    hb.leader_commit = r->commit_index;
    hb.prev_log_index = (r->log_count > 0) ? r->log_count - 1 : 0;
    hb.prev_log_term = (r->log_count > 0) ? r->log[r->log_count - 1].term : 0;
    hb.entries_count = 0;

    for (int i = 0; i < r->peer_count; i++) {
        _raft_send_rpc(r->peer_hosts[i], r->peer_ports[i], &hb);
    }
}

/* Handle incoming RPC on listener */
static void _raft_handle_rpc(PionRaftBlock* r, int conn_fd) {
    RaftRPC rpc;
    if (recv(conn_fd, &rpc, sizeof(rpc), MSG_WAITALL) != sizeof(RaftRPC)) {
        close(conn_fd);
        return;
    }

    RaftRPC resp;
    memset(&resp, 0, sizeof(resp));

    switch (rpc.type) {
    case RAFT_RPC_VOTE_REQ:
        resp.type = RAFT_RPC_VOTE_RESP;
        resp.term = r->current_term;
        resp.vote_granted = 0;
        if (rpc.term > r->current_term) {
            r->current_term = rpc.term;
            r->state = RAFT_STATE_FOLLOWER;
            r->voted_for[0] = '\0';
        }
        if (rpc.term >= r->current_term &&
            (r->voted_for[0] == '\0' || memcmp(r->voted_for, rpc.candidate_id, 40) == 0)) {
            /* Check log is at least as up-to-date */
            uint64_t my_last_term = (r->log_count > 0) ? r->log[r->log_count-1].term : 0;
            uint64_t my_last_idx = (r->log_count > 0) ? r->log_count - 1 : 0;
            if (rpc.last_log_term > my_last_term ||
                (rpc.last_log_term == my_last_term && rpc.last_log_index >= my_last_idx)) {
                resp.vote_granted = 1;
                memcpy(r->voted_for, rpc.candidate_id, 40);
                r->voted_for[40] = '\0';
                r->last_heartbeat_ns = _raft_now_ms();
            }
        }
        resp.term = r->current_term;
        send(conn_fd, &resp, sizeof(resp), 0);
        break;

    case RAFT_RPC_APPEND:
        resp.type = RAFT_RPC_APPEND_RESP;
        resp.term = r->current_term;
        resp.success = 0;
        if (rpc.term >= r->current_term) {
            r->current_term = rpc.term;
            r->state = RAFT_STATE_FOLLOWER;
            memcpy(r->leader_id, rpc.candidate_id, 40);
            r->leader_id[40] = '\0';
            r->last_heartbeat_ns = _raft_now_ms();
            resp.success = 1;
            /* Update commit index */
            if (rpc.leader_commit > (uint64_t)r->commit_index) {
                r->commit_index = (int)rpc.leader_commit;
                if (r->commit_index >= r->log_count)
                    r->commit_index = r->log_count - 1;
                if (r->commit_index < 0) r->commit_index = 0;
            }
        }
        resp.term = r->current_term;
        send(conn_fd, &resp, sizeof(resp), 0);
        break;
    }
    close(conn_fd);
}

/* Raft main thread */
static void* _pion_raft_thread(void* arg) {
    PionRaftBlock* r = (PionRaftBlock*)arg;

    /* Create listener for Raft RPCs */
    r->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (r->listen_fd < 0) return NULL;
    int reuse = 1;
    setsockopt(r->listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = pion_get_bind_addr();  /* gh #258: was INADDR_ANY */
    addr.sin_port = htons(r->raft_port);
    if (bind(r->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(r->listen_fd); r->listen_fd = -1; return NULL;
    }
    listen(r->listen_fd, 8);

    _raft_randomize_timeout(r);
    r->last_heartbeat_ns = _raft_now_ms();

    while (r->running) {
        uint64_t now = _raft_now_ms();
        uint64_t elapsed = now - r->last_heartbeat_ns;

        /* Check for incoming RPCs (non-blocking select, 10ms timeout) */
        fd_set fds;
        FD_ZERO(&fds);
        if (r->listen_fd >= 0) FD_SET(r->listen_fd, &fds);
        struct timeval tv = { .tv_sec = 0, .tv_usec = 10000 };
        int sel = select(r->listen_fd + 1, &fds, NULL, NULL, &tv);
        if (sel > 0 && FD_ISSET(r->listen_fd, &fds)) {
            int conn = accept(r->listen_fd, NULL, NULL);
            if (conn >= 0) _raft_handle_rpc(r, conn);
        }

        switch (r->state) {
        case RAFT_STATE_FOLLOWER:
            if (elapsed > r->election_timeout_ms) {
                _raft_start_election(r);
            }
            break;

        case RAFT_STATE_CANDIDATE:
            if (elapsed > r->election_timeout_ms) {
                _raft_start_election(r); /* retry election */
            }
            break;

        case RAFT_STATE_LEADER:
            if (elapsed > r->heartbeat_ms) {
                _raft_send_heartbeats(r);
                r->last_heartbeat_ns = _raft_now_ms();
            }
            break;
        }
    }

    if (r->listen_fd >= 0) close(r->listen_fd);
    return NULL;
}

/* Public API */
void* pion_raft_create(const char* node_id, int raft_port) {
    PionRaftBlock* r = (PionRaftBlock*)calloc(1, sizeof(PionRaftBlock));
    if (!r) return NULL;
    strncpy(r->my_node_id, node_id, 40); r->my_node_id[40] = '\0';
    r->raft_port = raft_port;
    r->state = RAFT_STATE_FOLLOWER;
    r->heartbeat_ms = 50;
    r->listen_fd = -1;
    return r;
}

void pion_raft_set_peer(void* block, int i, const char* host, int port) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    if (!r || i < 0 || i >= RAFT_MAX_PEERS) return;
    strncpy(r->peer_hosts[i], host, 63); r->peer_hosts[i][63] = '\0';
    r->peer_ports[i] = port;
    if (i + 1 > r->peer_count) r->peer_count = i + 1;
}

int pion_raft_start(void* block) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    if (!r) return -1;
    r->running = 1;
    _raft_randomize_timeout(r);
    return pthread_create(&r->tid, NULL, _pion_raft_thread, r) == 0 ? 0 : -1;
}

void pion_raft_stop(void* block) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    if (!r) return;
    r->running = 0;
    pthread_join(r->tid, NULL);
    free(r);
}

int pion_raft_get_state(void* block) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    return r ? r->state : 0;
}

uint64_t pion_raft_get_term(void* block) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    return r ? r->current_term : 0;
}

void pion_raft_get_leader(void* block, char* out_id) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    if (!r || !out_id) return;
    strncpy(out_id, r->leader_id, 40); out_id[40] = '\0';
}

int pion_raft_is_leader(void* block) {
    PionRaftBlock* r = (PionRaftBlock*)block;
    return (r && r->state == RAFT_STATE_LEADER) ? 1 : 0;
}


/* ── Feature 3: MIGRATE command support ──────────────────────────────────────
   Transfers a key from this node to a target node via RESP protocol.
   Used during slot migration (CLUSTER SETSLOT MIGRATING).
   Format: MIGRATE host port key db timeout [COPY] [REPLACE]
*/

int pion_migrate_key(const char* host, int port, const char* key, int key_len,
                     const uint8_t* val, int val_len, int timeout_ms) {
    int fd = pion_connect_tcp(host, port);
    if (fd < 0) return -1;
    struct timeval tv = { .tv_sec = timeout_ms / 1000, .tv_usec = (timeout_ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    /* Build RESP: *3\r\n$3\r\nSET\r\n$<key_len>\r\n<key>\r\n$<val_len>\r\n<val>\r\n */
    char hdr[128];
    int hlen = snprintf(hdr, sizeof(hdr), "*3\r\n$3\r\nSET\r\n$%d\r\n", key_len);
    if (send(fd, hdr, hlen, 0) != hlen) { close(fd); return -1; }
    if (send(fd, key, key_len, 0) != key_len) { close(fd); return -1; }
    char mid[32];
    int mlen = snprintf(mid, sizeof(mid), "\r\n$%d\r\n", val_len);
    if (send(fd, mid, mlen, 0) != mlen) { close(fd); return -1; }
    if (send(fd, val, val_len, 0) != val_len) { close(fd); return -1; }
    if (send(fd, "\r\n", 2, 0) != 2) { close(fd); return -1; }

    /* Read response: expect +OK\r\n */
    char resp[32];
    int n = (int)recv(fd, resp, sizeof(resp) - 1, 0);
    close(fd);
    return (n >= 3 && resp[0] == '+') ? 0 : -1;
}


/* ── Feature 4: Cross-Worker WAL Aggregation ─────────────────────────────────
   Workers 1..N forward WAL entries to worker 0 via a shared ring buffer.
   Worker 0 aggregates into the master WAL for replication.
*/

#define WAL_AGG_RING_SIZE (4 * 1024 * 1024)  /* 4MB per worker */
#define WAL_AGG_MAX_WORKERS 32

typedef struct {
    uint8_t* ring;
    volatile uint32_t head;  /* consumer (worker 0) */
    volatile uint32_t tail;  /* producer (worker N) */
    pthread_mutex_t mu;
} WalAggWorkerRing;

typedef struct {
    WalAggWorkerRing rings[WAL_AGG_MAX_WORKERS];
    int worker_count;
    volatile int running;
} PionWalAggregator;

void* pion_wal_agg_create(int worker_count) {
    PionWalAggregator* agg = (PionWalAggregator*)calloc(1, sizeof(PionWalAggregator));
    if (!agg) return NULL;
    agg->worker_count = worker_count;
    agg->running = 1;
    for (int i = 0; i < worker_count && i < WAL_AGG_MAX_WORKERS; i++) {
        agg->rings[i].ring = (uint8_t*)calloc(1, WAL_AGG_RING_SIZE);
        agg->rings[i].head = 0;
        agg->rings[i].tail = 0;
        pthread_mutex_init(&agg->rings[i].mu, NULL);
    }
    return agg;
}

/* Worker N pushes WAL entry into its ring (called from any worker) */
int pion_wal_agg_push(void* block, int worker_id, const uint8_t* entry, int entry_len) {
    PionWalAggregator* agg = (PionWalAggregator*)block;
    if (!agg || worker_id < 0 || worker_id >= agg->worker_count) return -1;
    WalAggWorkerRing* ring = &agg->rings[worker_id];
    if (entry_len > WAL_AGG_RING_SIZE / 2) return -1; /* too large */

    pthread_mutex_lock(&ring->mu);
    uint32_t avail = WAL_AGG_RING_SIZE - (ring->tail - ring->head);
    if ((uint32_t)entry_len > avail) {
        pthread_mutex_unlock(&ring->mu);
        return -1; /* ring full */
    }
    uint32_t pos = ring->tail % WAL_AGG_RING_SIZE;
    if (pos + (uint32_t)entry_len <= WAL_AGG_RING_SIZE) {
        memcpy(ring->ring + pos, entry, entry_len);
    } else {
        /* Wrap around */
        uint32_t first = WAL_AGG_RING_SIZE - pos;
        memcpy(ring->ring + pos, entry, first);
        memcpy(ring->ring, entry + first, entry_len - first);
    }
    ring->tail += entry_len;
    pthread_mutex_unlock(&ring->mu);
    return 0;
}

/* Worker 0 drains entries from worker_id's ring into out_buf. Returns bytes drained. */
int pion_wal_agg_drain(void* block, int worker_id, uint8_t* out_buf, int max_bytes) {
    PionWalAggregator* agg = (PionWalAggregator*)block;
    if (!agg || worker_id < 0 || worker_id >= agg->worker_count) return 0;
    WalAggWorkerRing* ring = &agg->rings[worker_id];

    pthread_mutex_lock(&ring->mu);
    uint32_t avail = ring->tail - ring->head;
    if (avail == 0) { pthread_mutex_unlock(&ring->mu); return 0; }
    uint32_t to_read = (avail < (uint32_t)max_bytes) ? avail : (uint32_t)max_bytes;
    uint32_t pos = ring->head % WAL_AGG_RING_SIZE;
    if (pos + to_read <= WAL_AGG_RING_SIZE) {
        memcpy(out_buf, ring->ring + pos, to_read);
    } else {
        uint32_t first = WAL_AGG_RING_SIZE - pos;
        memcpy(out_buf, ring->ring + pos, first);
        memcpy(out_buf + first, ring->ring, to_read - first);
    }
    ring->head += to_read;
    pthread_mutex_unlock(&ring->mu);
    return (int)to_read;
}

void pion_wal_agg_stop(void* block) {
    PionWalAggregator* agg = (PionWalAggregator*)block;
    if (!agg) return;
    agg->running = 0;
    for (int i = 0; i < agg->worker_count && i < WAL_AGG_MAX_WORKERS; i++) {
        pthread_mutex_destroy(&agg->rings[i].mu);
        free(agg->rings[i].ring);
    }
    free(agg);
}

/* ── ROLE (#39) ── */

/* Replica i of a primary: its address (the peer of its replication link),
   the port it serves clients on (0 when its PSYNC did not say) and the
   offset it has ACKed. Returns 1, or 0 when there is no replica i. */
int pion_repl_primary_replica_info(void* block, int idx, char* ip_out, int ip_cap,
                                   int* port_out, uint64_t* ack_out) {
    PionReplPrimary* blk = (PionReplPrimary*)block;
    if (!blk) return 0;
    pthread_mutex_lock(&blk->mu);
    int ok = idx >= 0 && idx < blk->conn_count;
    if (ok) {
        struct sockaddr_storage sa;
        socklen_t salen = sizeof(sa);
        ip_out[0] = '\0';
        if (getpeername(blk->conn_fds[idx], (struct sockaddr*)&sa, &salen) == 0) {
            if (sa.ss_family == AF_INET)
                inet_ntop(AF_INET, &((struct sockaddr_in*)&sa)->sin_addr, ip_out, (socklen_t)ip_cap);
            else if (sa.ss_family == AF_INET6)
                inet_ntop(AF_INET6, &((struct sockaddr_in6*)&sa)->sin6_addr, ip_out, (socklen_t)ip_cap);
        }
        *port_out = blk->listen_ports[idx];
        *ack_out = blk->ack_offsets[idx];
    }
    pthread_mutex_unlock(&blk->mu);
    return ok;
}

/* The replica link's state (PION_REPL_LINK_*); 1 ("connect") before the
   thread has tried. */
int pion_repl_replica_link_state(void* block) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (!blk) return PION_REPL_LINK_CONNECT;
    int s = blk->link_state;
    return s == 0 ? PION_REPL_LINK_CONNECT : s;
}

/* The port this replica serves clients on, sent with its PSYNC. */
void pion_repl_replica_set_listening_port(void* block, int port) {
    PionReplReplicaBlock* blk = (PionReplReplicaBlock*)block;
    if (blk) blk->listening_port = port;
}
