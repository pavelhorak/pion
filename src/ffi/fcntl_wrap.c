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

#include <stdio.h>

/* snprintf("%.*f") — Redis's "%f" in error messages (GEO's "invalid
   longitude,latitude pair %f,%f"). Returns the length or -1. */
int64_t pion_fmt_fixed(double v, int decimals, char* out, int64_t cap) {
    int l = snprintf(out, (size_t)cap, "%.*f", decimals, v);
    return (l < 0 || (int64_t)l + 1 > cap) ? -1 : l;
}
