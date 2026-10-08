#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <poll.h>
#include <time.h>

int set_nonblock_c(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

/* C1: Connect to a remote host:port with timeout (ms). Returns fd or -1. */
int pion_tcp_connect(const char *host, int port, int timeout_ms) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        close(fd);
        return -1;
    }

    /* Non-blocking connect with timeout */
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
        close(fd);
        return -1;
    }
    if (rc < 0) {
        /* Wait for connection with poll() */
        struct pollfd pfd = { .fd = fd, .events = POLLOUT };
        rc = poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : 5000);
        if (rc <= 0) {
            close(fd);
            return -1;
        }
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
            close(fd);
            return -1;
        }
    }

    /* Set blocking + TCP_NODELAY for MIGRATE */
    fcntl(fd, F_SETFL, flags);  /* restore blocking */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    return fd;
}

/* ── gh #258: process-wide bind address ───────────────────────────────────────
 *
 * Every listener used to bind INADDR_ANY because the sockaddr was memset to
 * zero and nobody filled bytes 4-7 in. That is five listeners on all
 * interfaces: RESP, port+1 (binary lane), port+10000 (WAL replication, which
 * has no auth at all), and the gossip/Raft pair.
 *
 * The address lives here rather than in Mojo because BOTH sides need it: the
 * three Mojo bind sites in server.mojo and the three C listeners in
 * uring_wrap.c. One global read by all six is the only way to be sure a new
 * listener cannot silently keep the old default.
 *
 * Stored in NETWORK byte order, ready to memcpy into sin_addr. Default is
 * loopback — main.mojo overrides it to INADDR_ANY only when the operator has
 * asked for that (--bind) or has set a password (Redis's protected-mode
 * precedent). A zero here would mean "all interfaces", so the default must be
 * set explicitly, never inherited from a memset. */
static uint32_t g_pion_bind_addr = 0x0100007f;  /* 127.0.0.1, network order */

/* Returns 0 on success, -1 if the address does not parse. Rejecting is
 * important: silently falling back would land on INADDR_ANY, i.e. a typo in
 * --bind would expose the server rather than fail to start. */
int pion_set_bind_addr(const char *addr) {
    if (!addr || !*addr) return -1;
    struct in_addr a;
    if (inet_pton(AF_INET, addr, &a) != 1) return -1;
    g_pion_bind_addr = (uint32_t)a.s_addr;
    return 0;
}

uint32_t pion_get_bind_addr(void) { return g_pion_bind_addr; }

/* Human-readable form for the startup banner and for error messages. Returns a
 * pointer to a static buffer — single-threaded startup use only. */
const char *pion_get_bind_addr_str(void) {
    static char buf[INET_ADDRSTRLEN];
    struct in_addr a;
    a.s_addr = g_pion_bind_addr;
    if (!inet_ntop(AF_INET, &a, buf, sizeof(buf))) return "?";
    return buf;
}

/* ── gh #258: read a secret from a file instead of argv ───────────────────────
 *
 * `--requirepass <pw>` puts the password in the process command line, where
 * `ps` and /proc/<pid>/cmdline expose it to every local user. This reads it
 * from a file instead.
 *
 * Trailing newline and CR are stripped, because `echo pw > file` is how the
 * file will actually be produced and a trailing \n would silently become part
 * of the password — a failure that looks like "the password is wrong" with no
 * indication why.
 *
 * Returns the length on success, -1 if the file cannot be read, -2 if it is
 * empty after stripping, -3 if it does not fit. The caller distinguishes these
 * because "file missing" and "file empty" are different operator errors.
 *
 * A world- or group-readable file is reported (once, to stderr) but NOT
 * refused: failing to start over a permission bit would be worse than the
 * exposure on a single-user box, and the operator may have a deliberate setup.
 */
int pion_read_secret_file(const char *path, char *out, int cap) {
    if (!path || !*path || !out || cap <= 1) return -1;
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & (S_IRGRP | S_IROTH))) {
        const char *w = "WARNING: secret file is group- or world-readable; "
                        "chmod 600 it\n";
        ssize_t ignored = write(2, w, strlen(w)); (void)ignored;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, out, (size_t)(cap - 1));
    close(fd);
    if (n < 0) return -1;
    /* Strip ALL trailing newline/CR/space — an editor may leave more than one. */
    while (n > 0 && (out[n-1] == '\n' || out[n-1] == '\r' ||
                     out[n-1] == ' '  || out[n-1] == '\t')) n--;
    out[n] = '\0';
    if (n == 0) return -2;
    return (int)n;
}

/* ── gh #393: numbers parsed exactly as Redis parses them ─────────────────────
   Redis reads a double with strtod(3) plus a few checks, and which checks
   depends on the entry point. Pion's hand-rolled parser rejected what strtod
   accepts ("inf", "0x10", "+.5") and accepted what Redis refuses ("1e400",
   "nan", "1.5 "), so the same argument was a number on one server and an error
   on the other. This is the one place those rules live.

   mode 0 — a value argument (ZADD score, ZINCRBY, …): Redis string2d. Empty,
            leading whitespace, trailing bytes, NaN, or out of range (ERANGE
            to ±HUGE_VAL or to 0) are errors. "inf" is fine.
   mode 1 — a range bound (ZCOUNT/ZRANGEBYSCORE after any "(" is stripped):
            zslParseRange runs strtod on the NUL-terminated argument and only
            rejects trailing bytes and NaN — so "" is 0 and "1e400" is inf.
   mode 2 — INCRBYFLOAT / HINCRBYFLOAT: string2ld (strtold) with the same
            refusals as mode 0; the value is returned as a double.
   Returns 1 and stores *out on success, 0 on a refusal. */
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>

int pion_parse_double(const char* p, int64_t n, int mode, double* out) {
    char buf[256];
    if (n < 0 || n >= (int64_t)sizeof(buf)) return 0;   /* Redis caps long-double text too */
    memcpy(buf, p, (size_t)n);
    buf[n] = '\0';
    if (memchr(buf, '\0', (size_t)n) != NULL) return 0;  /* an embedded NUL ends strtod early */
    if (mode != 1 && (n == 0 || isspace((unsigned char)buf[0]))) return 0;
    char* end = NULL;
    errno = 0;
    double v;
    if (mode == 2) {
        long double lv = strtold(buf, &end);
        if (*end != '\0' || isnan(lv)) return 0;
        if (errno == ERANGE && (lv == HUGE_VALL || lv == -HUGE_VALL || fpclassify(lv) == FP_ZERO)) return 0;
        v = (double)lv;
        if (isinf(v) && !isinf(lv)) return 0;              /* fits a long double, not a double */
    } else {
        v = strtod(buf, &end);
        if (*end != '\0' || isnan(v)) return 0;
        if (mode == 0 && errno == ERANGE &&
            (v == HUGE_VAL || v == -HUGE_VAL || fpclassify(v) == FP_ZERO)) return 0;
    }
    *out = v;
    return 1;
}

/* INCRBYFLOAT / HINCRBYFLOAT in Redis's own arithmetic: long double.

   Redis reads the current value and the increment with string2ld (strtold),
   adds them in long double, and prints the sum with ld2string(LD_STR_HUMAN):
   "%.17Lf", trailing zeros and a bare '.' removed, "-0" as "0". long double
   is the platform's — 80-bit on x86-64 Linux, 128-bit on AArch64 Linux, a
   double on Apple silicon — so doing the same here gives the same reply as
   the Redis built for that platform, range included: on Linux `1e400` is a
   number. Pion added Float64s, which differed in range on Linux and in the
   digits wherever long double is wider than a double.

   pion_ld_kind: 0 not a float (string2ld's refusals), 1 finite, 2 infinite.
   pion_ld_incr: cur_kind 0 = no current value (0), 1 = text in cur/curlen,
   2 = the double cur_d. Returns the length written to out, or -1 (current
   value not a float), -2 (increment not a float), -3 (NaN or infinite
   result), -4 (out too small). */
#include <stdio.h>

static int pion_string2ld(const char* p, int64_t n, long double* out) {
    char buf[5 * 1024];                  /* MAX_LONG_DOUBLE_CHARS */
    if (n <= 0 || n >= (int64_t)sizeof(buf)) return 0;
    memcpy(buf, p, (size_t)n);
    buf[n] = '\0';
    char* end = NULL;
    errno = 0;
    long double v = strtold(buf, &end);
    if (isspace((unsigned char)buf[0]) || *end != '\0' || isnan(v)) return 0;
    if (errno == ERANGE && (v == HUGE_VALL || v == -HUGE_VALL || fpclassify(v) == FP_ZERO)) return 0;
    if (errno == EINVAL) return 0;
    *out = v;
    return 1;
}

/* A blocking command's timeout, as Redis's getTimeoutFromObjectOrReply reads
   seconds (#38): string2ld, times 1000, ceil. 0 with *out = 0 (block forever)
   or the absolute deadline in ms from now_ms; 1 "timeout is not a float or out
   of range", 2 "timeout is out of range", 3 "timeout is negative". */
int64_t pion_parse_block_timeout(const char* p, int64_t n, int64_t now_ms, int64_t* out) {
    long double v;
    if (!pion_string2ld(p, n, &v)) return 1;
    v *= 1000.0L;
    if (v > (long double)LLONG_MAX) return 2;
    long long t = (long long)ceill(v);
    if (t < 0) return 3;
    if (t > 0) {
        if (t > LLONG_MAX - now_ms) return 2;
        t += now_ms;
    }
    *out = t;
    return 0;
}

int pion_ld_kind(const char* p, int64_t n) {
    long double v;
    if (!pion_string2ld(p, n, &v)) return 0;
    return isinf(v) ? 2 : 1;
}

int64_t pion_ld_incr(int cur_kind, const char* cur, int64_t curlen, double cur_d,
                     const char* inc, int64_t inclen, char* out, int64_t cap) {
    long double a = 0, b;
    if (cur_kind == 1 && !pion_string2ld(cur, curlen, &a)) return -1;
    if (cur_kind == 2) a = (long double)cur_d;
    if (!pion_string2ld(inc, inclen, &b)) return -2;
    long double v = a + b;
    if (isnan(v) || isinf(v)) return -3;
    int l = snprintf(out, (size_t)cap, "%.17Lf", v);
    if (l < 0 || (int64_t)l + 1 > cap) return -4;
    if (strchr(out, '.') != NULL) {
        char* q = out + l - 1;
        while (*q == '0') { q--; l--; }
        if (*q == '.') l--;
    }
    if (l == 2 && out[0] == '-' && out[1] == '0') { out[0] = '0'; l = 1; }
    return l;
}

/* snprintf("%.*f") — Redis's "%f" in error messages (GEO's "invalid
   longitude,latitude pair %f,%f"). Returns the length or -1. */
int64_t pion_fmt_fixed(double v, int decimals, char* out, int64_t cap) {
    int l = snprintf(out, (size_t)cap, "%.*f", decimals, v);
    return (l < 0 || (int64_t)l + 1 > cap) ? -1 : l;
}

/* LCS and LOLWUT are ports of Valkey's (BSD-3-Clause); they live in their own
   file under that licence, compiled as part of this one so that every build
   line that links fcntl_wrap.o links them too. */
#include "redis_ports.c"

/* ── MONITOR (#39) ──
   pion_peer_id: the connection's peer as Redis prints it ("ip:port", or
   "[ip]:port" for IPv6), the client address in a MONITOR line and in CLIENT
   LIST. Returns its length, or 0 when the fd has no peer.

   pion_monitor_line: one MONITOR line, as Redis's replicationFeedMonitors
   builds it: "+<sec>.<usec> [<db> <peer>] "arg" "arg"...\r\n", each argument
   quoted and escaped as sdscatrepr does. fd < 0 means a script's command
   ("[0 lua]"). Returns the length and the malloc'd line in *out
   (pion_lcs_free). */
#include <sys/time.h>
#include <netdb.h>

/* #45: true when the peer is this machine (loopback, or a Unix socket), as
   Redis's islocalClient decides for enable-debug-command "local". */
int pion_peer_is_local(int fd) {
    struct sockaddr_storage sa;
    socklen_t salen = sizeof(sa);
    if (getpeername(fd, (struct sockaddr *)&sa, &salen) != 0) return 0;
    if (sa.ss_family == AF_UNIX) return 1;
    if (sa.ss_family == AF_INET) {
        struct sockaddr_in *s = (struct sockaddr_in *)&sa;
        return ntohl(s->sin_addr.s_addr) == 0x7f000001u;
    }
    if (sa.ss_family == AF_INET6) {
        struct sockaddr_in6 *s = (struct sockaddr_in6 *)&sa;
        if (IN6_IS_ADDR_LOOPBACK(&s->sin6_addr)) return 1;
        if (IN6_IS_ADDR_V4MAPPED(&s->sin6_addr))
            return s->sin6_addr.s6_addr[12] == 127 && s->sin6_addr.s6_addr[13] == 0
                && s->sin6_addr.s6_addr[14] == 0 && s->sin6_addr.s6_addr[15] == 1;
    }
    return 0;
}

int64_t pion_peer_id(int fd, char *buf, int64_t cap) {
    struct sockaddr_storage sa;
    socklen_t salen = sizeof(sa);
    if (getpeername(fd, (struct sockaddr *)&sa, &salen) != 0) return 0;
    char ip[INET6_ADDRSTRLEN];
    int port = 0;
    int v6 = 0;
    if (sa.ss_family == AF_INET) {
        struct sockaddr_in *s = (struct sockaddr_in *)&sa;
        if (!inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip))) return 0;
        port = ntohs(s->sin_port);
    } else if (sa.ss_family == AF_INET6) {
        struct sockaddr_in6 *s = (struct sockaddr_in6 *)&sa;
        if (!inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip))) return 0;
        port = ntohs(s->sin6_port);
        v6 = 1;
    } else {
        return 0;
    }
    int l = snprintf(buf, (size_t)cap, v6 ? "[%s]:%d" : "%s:%d", ip, port);
    return (l < 0 || l >= cap) ? 0 : l;
}

/* #47: this end of the connection ("ip:port"), CLIENT LIST's laddr. */
int64_t pion_local_id(int fd, char *buf, int64_t cap) {
    struct sockaddr_storage sa;
    socklen_t salen = sizeof(sa);
    if (getsockname(fd, (struct sockaddr *)&sa, &salen) != 0) return 0;
    char ip[INET6_ADDRSTRLEN];
    int port = 0;
    int v6 = 0;
    if (sa.ss_family == AF_INET) {
        struct sockaddr_in *s = (struct sockaddr_in *)&sa;
        if (!inet_ntop(AF_INET, &s->sin_addr, ip, sizeof(ip))) return 0;
        port = ntohs(s->sin_port);
    } else if (sa.ss_family == AF_INET6) {
        struct sockaddr_in6 *s = (struct sockaddr_in6 *)&sa;
        if (!inet_ntop(AF_INET6, &s->sin6_addr, ip, sizeof(ip))) return 0;
        port = ntohs(s->sin6_port);
        v6 = 1;
    } else {
        return 0;
    }
    int l = snprintf(buf, (size_t)cap, v6 ? "[%s]:%d" : "%s:%d", ip, port);
    return (l < 0 || l >= cap) ? 0 : l;
}

/* #47: CLIENT IDs, never reused, unique across the workers. A connection's
   ID used to be its fd, so the next connection to get the fd had a killed
   client's ID. */
static unsigned long long pion_client_ids = 0;

unsigned long long pion_next_client_id(void) {
    return __atomic_add_fetch(&pion_client_ids, 1, __ATOMIC_RELAXED);
}

/* #47: CLIENT KILL. Shutting the socket down, rather than closing it, leaves
   the fd to the worker that serves it: its event loop reads EOF and closes
   the connection the way it closes any other, so no other worker frees what
   that worker uses, and the fd is not reused while that worker still has it.
   A client that kills itself is shut down only once its reply is out (the
   engine's close_after). */
int pion_kill_fd(int fd) {
    return shutdown(fd, SHUT_RDWR);
}

/* #47: a cheap monotonic tick counter, for SLOWLOG's per-command timing and
   CLIENT LIST's idle. On arm64 and x86-64 it is the CPU's counter register (a
   fraction of a nanosecond to read on Apple silicon), elsewhere
   CLOCK_MONOTONIC in ns. pion_ticks_per_us converts. */
uint64_t pion_ticks(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#elif defined(__x86_64__)
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

double pion_ticks_per_us(void) {
#if defined(__aarch64__)
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return (double)f / 1e6;
#elif defined(__x86_64__)
    /* The TSC's rate is not architectural: measure it once against
       CLOCK_MONOTONIC (2 ms; a second worker racing here measures the
       same rate). */
    static double rate = 0;
    if (rate == 0) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        uint64_t c0 = pion_ticks();
        do { clock_gettime(CLOCK_MONOTONIC, &b); }
        while ((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec) < 2000000);
        uint64_t c1 = pion_ticks();
        double us = ((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)) / 1000.0;
        rate = (double)(c1 - c0) / us;
    }
    return rate;
#else
    return 1000.0;
#endif
}

/* #47: wall-clock milliseconds (CLIENT LIST age, SLOWLOG timestamps). */
int64_t pion_unix_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* #47: slowlog-log-slower-than (us; -1 off, 0 every command) and
   slowlog-max-len, process-wide so a CONFIG SET on one worker reaches all. */
static int64_t pion_slowlog_slower_than = 10000;
static int64_t pion_slowlog_max_len = 128;

int64_t pion_slowlog_get_slower_than(void) { return __atomic_load_n(&pion_slowlog_slower_than, __ATOMIC_RELAXED); }
void pion_slowlog_set_slower_than(int64_t v) { __atomic_store_n(&pion_slowlog_slower_than, v, __ATOMIC_RELAXED); }
int64_t pion_slowlog_get_max_len(void) { return __atomic_load_n(&pion_slowlog_max_len, __ATOMIC_RELAXED); }
void pion_slowlog_set_max_len(int64_t v) { __atomic_store_n(&pion_slowlog_max_len, v, __ATOMIC_RELAXED); }

static void mon_add(LcsBuf *b, const char *s, size_t n) { lcs_add(b, s, n); }

static void mon_repr(LcsBuf *b, const unsigned char *p, int64_t len) {
    mon_add(b, "\"", 1);
    for (int64_t k = 0; k < len; k++) {
        unsigned char c = p[k];
        char esc[8];
        switch (c) {
        case '\\': mon_add(b, "\\\\", 2); break;
        case '"': mon_add(b, "\\\"", 2); break;
        case '\n': mon_add(b, "\\n", 2); break;
        case '\r': mon_add(b, "\\r", 2); break;
        case '\t': mon_add(b, "\\t", 2); break;
        case '\a': mon_add(b, "\\a", 2); break;
        case '\b': mon_add(b, "\\b", 2); break;
        default:
            if (isprint(c)) {
                mon_add(b, (const char *)&c, 1);
            } else {
                snprintf(esc, sizeof(esc), "\\x%02x", c);
                mon_add(b, esc, 4);
            }
        }
    }
    mon_add(b, "\"", 1);
}

int64_t pion_monitor_line(int fd, int64_t argc, const char **argv, const int64_t *lens, char **out) {
    LcsBuf b = {NULL, 0, 0, 0};
    struct timeval tv;
    gettimeofday(&tv, NULL);
    char head[128];
    int hl = snprintf(head, sizeof(head), "+%ld.%06ld ", (long)tv.tv_sec, (long)tv.tv_usec);
    mon_add(&b, head, (size_t)hl);
    if (fd < 0) {
        mon_add(&b, "[0 lua] ", 8);
    } else {
        char peer[80];
        int64_t pl = pion_peer_id(fd, peer, sizeof(peer));
        mon_add(&b, "[0 ", 3);
        mon_add(&b, peer, (size_t)pl);
        mon_add(&b, "] ", 2);
    }
    for (int64_t j = 0; j < argc; j++) {
        mon_repr(&b, (const unsigned char *)argv[j], lens[j]);
        if (j != argc - 1) mon_add(&b, " ", 1);
    }
    mon_add(&b, "\r\n", 2);
    *out = b.p;
    return b.oom ? -1 : (int64_t)b.n;
}

/* ── Pub/sub between workers (#42) ──
   Each worker has an inbox. PUBLISH and SPUBLISH on one worker append the
   whole message to every other worker's inbox, and each worker takes its
   inbox on its tick and delivers to its own subscribers. A record is
   [u8 kind][u32 channel length][u32 message length][channel][message].
   This replaced a ring of fixed 2 KB slots, which dropped longer messages
   and published a slot's index before writing the slot. */
#include <pthread.h>

typedef struct {
    pthread_mutex_t mu;
    uint8_t *buf;
    size_t len, cap;
    int has;                 /* nonzero while buf holds records; read without the lock */
} PionPubsubInbox;

static PionPubsubInbox *g_pubsub_inbox = NULL;
static int g_pubsub_workers = 0;

void pion_pubsub_init(int nworkers) {
    if (g_pubsub_inbox || nworkers <= 1) return;
    g_pubsub_inbox = (PionPubsubInbox *)calloc((size_t)nworkers, sizeof(PionPubsubInbox));
    if (!g_pubsub_inbox) return;
    for (int w = 0; w < nworkers; w++) pthread_mutex_init(&g_pubsub_inbox[w].mu, NULL);
    g_pubsub_workers = nworkers;
}

/* Post to every worker but `from`. Returns 0, or -1 when a copy could not be
   allocated (that worker misses the message). */
int pion_pubsub_post(int from, int kind, const uint8_t *ch, int64_t cl, const uint8_t *msg, int64_t ml) {
    int rc = 0;
    for (int w = 0; w < g_pubsub_workers; w++) {
        if (w == from) continue;
        PionPubsubInbox *in = &g_pubsub_inbox[w];
        size_t need = 9 + (size_t)cl + (size_t)ml;
        pthread_mutex_lock(&in->mu);
        if (in->len + need > in->cap) {
            size_t c = in->cap ? in->cap : 4096;
            while (c < in->len + need) c *= 2;
            uint8_t *nb = (uint8_t *)realloc(in->buf, c);
            if (!nb) { pthread_mutex_unlock(&in->mu); rc = -1; continue; }
            in->buf = nb;
            in->cap = c;
        }
        uint8_t *p = in->buf + in->len;
        uint32_t c32 = (uint32_t)cl, m32 = (uint32_t)ml;
        p[0] = (uint8_t)kind;
        memcpy(p + 1, &c32, 4);
        memcpy(p + 5, &m32, 4);
        memcpy(p + 9, ch, (size_t)cl);
        memcpy(p + 9 + cl, msg, (size_t)ml);
        in->len += need;
        __atomic_store_n(&in->has, 1, __ATOMIC_RELEASE);
        pthread_mutex_unlock(&in->mu);
    }
    return rc;
}

/* Take this worker's inbox: returns its length and the buffer in *out (the
   caller frees it with pion_lcs_free), or 0. The check before the lock is a
   hint only: a record that lands just after it is taken on the next tick. */
int64_t pion_pubsub_take(int worker, uint8_t **out) {
    *out = NULL;
    if (!g_pubsub_inbox || worker < 0 || worker >= g_pubsub_workers) return 0;
    PionPubsubInbox *in = &g_pubsub_inbox[worker];
    if (!__atomic_load_n(&in->has, __ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&in->mu);
    int64_t n = (int64_t)in->len;
    *out = in->buf;
    in->buf = NULL;
    in->len = in->cap = 0;
    __atomic_store_n(&in->has, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&in->mu);
    return n;
}

/* ── DUMP payloads (#41) ──
   CRC-64/Jones (reflected polynomial 0x95AC9329AC4BC9B5, init 0, no final
   xor): the checksum Redis puts in a DUMP payload's footer. Pion's payload
   carries its own format version, so a payload either server did not write
   is refused by the other with "DUMP payload version or checksum are wrong". */
static uint64_t g_crc64_table[256];
static int g_crc64_ready = 0;

static void crc64_init(void) {
    for (int i = 0; i < 256; i++) {
        uint64_t c = (uint64_t)i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (c >> 1) ^ 0x95AC9329AC4BC9B5ULL : (c >> 1);
        g_crc64_table[i] = c;
    }
    g_crc64_ready = 1;
}

uint64_t pion_crc64(uint64_t crc, const uint8_t *p, int64_t n) {
    if (!g_crc64_ready) crc64_init();
    for (int64_t i = 0; i < n; i++)
        crc = g_crc64_table[(uint8_t)(crc ^ p[i])] ^ (crc >> 8);
    return crc;
}

/* ── MIGRATE's connection (#41), as Redis's syncio.c ──
   pion_tcp_connect_host resolves the host (a name, IPv4 or IPv6), connects
   within the timeout and returns a blocking socket with TCP_NODELAY.
   pion_sync_write writes everything or fails; pion_sync_readline reads one
   CRLF-terminated line. Each waits at most `timeout_ms` for progress. */
#include <netdb.h>

int pion_tcp_connect_host(const char *host, int port, int timeout_ms) {
    char ports[16];
    snprintf(ports, sizeof(ports), "%d", port);
    struct addrinfo hints, *res = NULL, *ai;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, ports, &hints, &res) != 0) return -1;
    int fd = -1;
    for (ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc < 0 && errno == EINPROGRESS) {
            struct pollfd pfd = { .fd = fd, .events = POLLOUT };
            rc = poll(&pfd, 1, timeout_ms > 0 ? timeout_ms : 1000) > 0 ? 0 : -1;
            if (rc == 0) {
                int err = 0;
                socklen_t len = sizeof(err);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
                if (err != 0) rc = -1;
            }
        }
        if (rc == 0) {
            fcntl(fd, F_SETFL, flags);
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

int pion_sync_write(int fd, const uint8_t *p, int64_t n, int timeout_ms) {
    int64_t done = 0;
    while (done < n) {
        struct pollfd pfd = { .fd = fd, .events = POLLOUT };
        if (poll(&pfd, 1, timeout_ms) <= 0) return -1;
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, p + done, (size_t)(n - done), MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, p + done, (size_t)(n - done), 0);
#endif
        if (w <= 0) {
            if (w < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            return -1;
        }
        done += w;
    }
    return 0;
}

int64_t pion_sync_readline(int fd, uint8_t *buf, int64_t cap, int timeout_ms) {
    int64_t len = 0;
    for (;;) {
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        if (poll(&pfd, 1, timeout_ms) <= 0) return -1;
        char c;
        ssize_t r = recv(fd, &c, 1, 0);
        if (r <= 0) {
            if (r < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            return -1;
        }
        if (c == '\n') {
            if (len > 0 && len <= cap && buf[len - 1] == '\r') len--;
            return len > cap ? cap : len;
        }
        /* a line longer than the buffer keeps its first `cap` bytes and is
           read to its end, so the next read starts at the next reply */
        if (len < cap) buf[len] = (uint8_t)c;
        len++;
    }
}

/* gh #468: the export lane. V.EXPORT writes a prompt prefix's K/V as a
 * safetensors file that a client on the same machine maps with mx.load,
 * instead of copying the bytes over loopback TCP. The server picks every
 * name (no client-supplied path), the directory is 0700 and the files 0600,
 * and a file is written under a temporary name and renamed, so a reader never
 * maps a half-written one. */
#include <dirent.h>

static int pion_xf_join(char *out, size_t cap, const char *dir, const char *name) {
    int n = snprintf(out, cap, "%s/%s", dir, name);
    return (n > 0 && (size_t)n < cap) ? 0 : -1;
}

/* Create `dir` (0700) if missing, then open `dir/tmp_name` for writing (0600,
 * truncated). Returns the fd, or -1. */
int pion_xf_open(const char *dir, const char *tmp_name) {
    char path[PATH_MAX];
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return -1;
    if (pion_xf_join(path, sizeof path, dir, tmp_name) != 0) return -1;
    return open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
}

/* Write all `n` bytes. Returns 0, or -1 on any error. */
int pion_xf_write(int fd, const void *buf, int64_t n) {
    const char *p = (const char *)buf;
    while (n > 0) {
        ssize_t w = write(fd, p, (size_t)(n > (1 << 30) ? (1 << 30) : n));
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += w;
        n -= w;
    }
    return 0;
}

/* Close fd and rename dir/tmp_name to dir/final_name (atomic on one volume).
 * ok == 0 discards the temporary file instead. Returns 0 on success. */
int pion_xf_commit(int fd, const char *dir, const char *tmp_name, const char *final_name, int ok) {
    char tmp[PATH_MAX], fin[PATH_MAX];
    int rc = close(fd);
    if (pion_xf_join(tmp, sizeof tmp, dir, tmp_name) != 0) return -1;
    if (!ok || rc != 0) { unlink(tmp); return -1; }
    if (pion_xf_join(fin, sizeof fin, dir, final_name) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, fin) != 0) { unlink(tmp); return -1; }
    return 0;
}

/* 1 when dir/name is a regular file, else 0. */
int pion_xf_exists(const char *dir, const char *name) {
    char path[PATH_MAX];
    struct stat st;
    if (pion_xf_join(path, sizeof path, dir, name) != 0) return 0;
    return (stat(path, &st) == 0 && S_ISREG(st.st_mode)) ? 1 : 0;
}

/* Delete the regular files in `dir` whose names start with `prefix` and do
 * not start with `keep`. Returns how many were deleted (0 if dir is absent). */
int pion_xf_prune(const char *dir, const char *prefix, const char *keep) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t lp = strlen(prefix), lk = strlen(keep);
    int n = 0;
    struct dirent *e;
    char path[PATH_MAX];
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, prefix, lp) != 0) continue;
        if (lk > 0 && strncmp(e->d_name, keep, lk) == 0) continue;
        if (pion_xf_join(path, sizeof path, dir, e->d_name) != 0) continue;
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode) && unlink(path) == 0) n++;
    }
    closedir(d);
    return n;
}

/* Absolute form of dir/name into out (cap bytes). Returns its length, or -1. */
int64_t pion_xf_realpath(const char *dir, const char *name, char *out, int64_t cap) {
    char path[PATH_MAX], full[PATH_MAX];
    if (pion_xf_join(path, sizeof path, dir, name) != 0) return -1;
    if (!realpath(path, full)) return -1;
    size_t n = strlen(full);
    if ((int64_t)n + 1 > cap) return -1;
    memcpy(out, full, n + 1);
    return (int64_t)n;
}
