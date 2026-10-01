/* crash_wrap.c — gh #138: crash/exit breadcrumbs + liveness heartbeat
 *
 * Problem this solves: a `pion-server` process can disappear mid-session with
 * no trace. On a memory-constrained host the most likely killer is the OS
 * (macOS jetsam / Linux OOM killer), which sends an *uncatchable* SIGKILL —
 * no signal handler can ever report it. So this shim provides two layers:
 *
 *   1. Signal breadcrumb (catchable deaths). sigaction handlers for
 *      SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT/SIGTERM/SIGINT/SIGQUIT/SIGXCPU
 *      append one line to the crash log and stderr, then re-raise the default
 *      disposition so the exit status and core dump stay correct.
 *
 *   2. Heartbeat status file (uncatchable deaths). The event loop calls
 *      pion_crash_heartbeat() from its per-tick housekeeping; once a second
 *      one worker samples RSS and rewrites a fixed-size 512-byte record with
 *      pwrite(). After a SIGKILL the file is the last thing the process said
 *      about itself — RSS at ~1s before death is exactly what confirms or
 *      denies jetsam/OOM, and the per-worker tick counters distinguish
 *      "was serving" from "was idle" from "worker N was wedged".
 *
 * A high-water warning fires once when RSS crosses the warn threshold
 * (default 70% of physical RAM), so an operator gets a heads-up *before*
 * the kernel reaps the process rather than a silent gap afterwards.
 *
 * Everything executed from the signal handler is async-signal-safe: no
 * malloc, no stdio, no snprintf, no RSS sampling (cached values only) —
 * just hand-rolled integer formatting, write() and pwrite().
 *
 * Build: gcc -c src/ffi/crash_wrap.c -o src/ffi/crash_wrap.o
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <errno.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

/* Backtrace on fault. execinfo lives in libSystem (macOS) and glibc; musl has
 * no such header, so a musl build simply gets the breadcrumb without frames. */
#if defined(__APPLE__) || defined(__GLIBC__)
#include <execinfo.h>
#define PION_HAVE_BACKTRACE 1
#endif

#define PION_CRASH_MAX_WORKERS 64
#define PION_STATUS_RECORD_LEN 512

/* One 64-byte cache line per worker's tick counter. Eight UInt64 counters would
 * otherwise share a line and ping-pong it between cores on every beat — the
 * same reason SharedHNSWView strides shard_ready by 64 bytes. */
#define PION_TICK_STRIDE 8   /* uint64 units == 64 bytes */

/* ── State ────────────────────────────────────────────────────────────────── */

static int      g_log_fd       = -1;
static int      g_status_fd    = -1;
static int      g_initialized  = 0;
static int      g_pid          = 0;
static int      g_port         = 0;
static int      g_workers      = 0;
static char     g_version[48];

static uint64_t g_start_mono   = 0;   /* CLOCK_MONOTONIC seconds at init */
static uint64_t g_start_unix   = 0;
static uint64_t g_total_ram    = 0;
static uint64_t g_rss_warn     = 0;   /* bytes; 0 = warning disabled */

static volatile uint64_t g_ticks[PION_CRASH_MAX_WORKERS * PION_TICK_STRIDE];
static volatile uint64_t g_rss_bytes      = 0;
static volatile uint64_t g_rss_peak_bytes = 0;
static volatile uint64_t g_hb_mono        = 0;  /* monotonic secs of last sample */
static volatile uint64_t g_hb_unix        = 0;
static volatile int      g_rss_warned     = 0;
static volatile int      g_exiting        = 0;

/* ── Async-signal-safe helpers ────────────────────────────────────────────── */

static uint64_t _mono_secs(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) return (uint64_t)ts.tv_sec;
#endif
    return (uint64_t)time(NULL);
}

static int _lit(char *out, const char *s) {
    int n = 0;
    while (s[n]) { out[n] = s[n]; n++; }
    return n;
}

static int _u64(char *out, uint64_t v) {
    char tmp[24];
    int n = 0;
    if (v == 0) { out[0] = '0'; return 1; }
    while (v && n < 24) { tmp[n++] = (char)('0' + (v % 10)); v /= 10; }
    for (int i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    return n;
}

/* "key=value\n" with an unsigned decimal value */
static int _kv_u64(char *out, const char *key, uint64_t v) {
    int o = _lit(out, key);
    out[o++] = '=';
    o += _u64(out + o, v);
    out[o++] = '\n';
    return o;
}

static int _kv_str(char *out, const char *key, const char *v) {
    int o = _lit(out, key);
    out[o++] = '=';
    o += _lit(out + o, v);
    out[o++] = '\n';
    return o;
}

static const char *_signame(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV";
        case SIGBUS:  return "SIGBUS";
        case SIGILL:  return "SIGILL";
        case SIGFPE:  return "SIGFPE";
        case SIGABRT: return "SIGABRT";
        case SIGTERM: return "SIGTERM";
        case SIGINT:  return "SIGINT";
        case SIGQUIT: return "SIGQUIT";
        case SIGXCPU: return "SIGXCPU";
        case SIGPIPE: return "SIGPIPE";
        case SIGHUP:  return "SIGHUP";
        case SIGALRM: return "SIGALRM";
        default:      return "SIG?";
    }
}

/* Sum of per-worker event-loop ticks — proves the loop was (or was not) alive. */
static uint64_t _tick_total(void) {
    uint64_t t = 0;
    int n = g_workers > 0 && g_workers < PION_CRASH_MAX_WORKERS
            ? g_workers : PION_CRASH_MAX_WORKERS;
    for (int i = 0; i < n; i++) t += g_ticks[i * PION_TICK_STRIDE];
    return t;
}

/* ── RSS / RAM sampling (NEVER called from a signal handler) ──────────────── */

static uint64_t _sample_rss(void) {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return (uint64_t)info.resident_size;
    return 0;
#elif defined(__linux__)
    /* /proc/self/statm: "size resident shared text lib data dt" in pages */
    char buf[128];
    int fd = open("/proc/self/statm", O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    const char *p = buf;
    while (*p && *p != ' ') p++;      /* skip total size */
    while (*p == ' ') p++;
    uint64_t pages = 0;
    while (*p >= '0' && *p <= '9') { pages = pages * 10 + (uint64_t)(*p - '0'); p++; }
    long ps = sysconf(_SC_PAGESIZE);
    return pages * (uint64_t)(ps > 0 ? ps : 4096);
#else
    return 0;
#endif
}

static uint64_t _total_ram(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) == 0) return mem;
    return 0;
#elif defined(__linux__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long ps    = sysconf(_SC_PAGESIZE);
    if (pages > 0 && ps > 0) return (uint64_t)pages * (uint64_t)ps;
    return 0;
#else
    return 0;
#endif
}

/* ── Status file ──────────────────────────────────────────────────────────── */

/* Format the fixed-size status record from CACHED values only, then pwrite it
 * at offset 0. Fixed size means a concurrent reader can never see a truncated
 * record, and pwrite()+cached-values-only makes this callable from a handler. */
static void _write_status(const char *state, int sig) {
    if (g_status_fd < 0) return;

    char rec[PION_STATUS_RECORD_LEN];
    int o = 0;
    o += _kv_u64(rec + o, "pion_status_version", 1);
    o += _kv_str(rec + o, "state", state);
    o += _kv_u64(rec + o, "signal", (uint64_t)(sig < 0 ? 0 : sig));
    o += _kv_str(rec + o, "signal_name", sig > 0 ? _signame(sig) : "-");
    o += _kv_u64(rec + o, "pid", (uint64_t)g_pid);
    o += _kv_str(rec + o, "version", g_version);
    o += _kv_u64(rec + o, "port", (uint64_t)g_port);
    o += _kv_u64(rec + o, "workers", (uint64_t)g_workers);
    o += _kv_u64(rec + o, "start_unix_s", g_start_unix);
    o += _kv_u64(rec + o, "uptime_s", _mono_secs() - g_start_mono);
    o += _kv_u64(rec + o, "heartbeat_unix_s", g_hb_unix);
    o += _kv_u64(rec + o, "rss_bytes", g_rss_bytes);
    o += _kv_u64(rec + o, "rss_peak_bytes", g_rss_peak_bytes);
    o += _kv_u64(rec + o, "total_ram_bytes", g_total_ram);
    o += _kv_u64(rec + o, "rss_pct", g_total_ram ? (g_rss_bytes * 100) / g_total_ram : 0);
    o += _kv_u64(rec + o, "ticks", _tick_total());

    /* Pad with newlines so the record is always exactly PION_STATUS_RECORD_LEN
     * bytes and still parses as (empty) shell lines. */
    while (o < PION_STATUS_RECORD_LEN) rec[o++] = '\n';

    ssize_t w = pwrite(g_status_fd, rec, PION_STATUS_RECORD_LEN, 0);
    (void)w;
}

/* ── Fault backtrace ──────────────────────────────────────────────────────── */

/* Frames are captured into a preallocated static array and printed with
 * backtrace_symbols_fd(), which writes straight to an fd and — unlike
 * backtrace_symbols() — never mallocs. This runs AFTER the breadcrumb line and
 * the status record are already on disk, so if the unwind itself faults we lose
 * only the frames, never the evidence that mattered before this existed.
 *
 * g_in_backtrace is the recursion guard for exactly that case: a fault inside
 * the unwinder re-enters this handler, sees the flag, and goes straight to
 * re-raise instead of looping.
 *
 * Why this exists: on 2026-08-18 a 0.913 server took a SIGSEGV that Gate 1
 * never noticed (the suite reported 114/114 while the process died), and it did
 * not reproduce in ~17 attempts. There was nothing to debug afterwards — macOS
 * wrote no .ips, core dumps need sudo here (`ulimit -c` 0, /cores root-owned),
 * and lldb refuses non-interactively without DevToolsSecurity. A rare crash you
 * cannot reproduce is only diagnosable if the FIRST occurrence records where it
 * was, so the breadcrumb now carries frames.
 */
#if PION_HAVE_BACKTRACE
static void *g_bt_frames[64];
static volatile sig_atomic_t g_in_backtrace = 0;

static int _is_fault_signal(int sig) {
    return sig == SIGSEGV || sig == SIGBUS || sig == SIGILL
        || sig == SIGFPE  || sig == SIGABRT;
}

static void _write_backtrace(int sig) {
    if (!_is_fault_signal(sig) || g_in_backtrace) return;
    g_in_backtrace = 1;
    static const char hdr[] = "PION EXIT: backtrace (innermost first)\n";
    int n = backtrace(g_bt_frames, (int)(sizeof(g_bt_frames) / sizeof(g_bt_frames[0])));
    if (n <= 0) return;
    if (g_log_fd >= 0) {
        ssize_t w = write(g_log_fd, hdr, sizeof(hdr) - 1); (void)w;
        backtrace_symbols_fd(g_bt_frames, n, g_log_fd);
    }
    { ssize_t w = write(STDERR_FILENO, hdr, sizeof(hdr) - 1); (void)w; }
    backtrace_symbols_fd(g_bt_frames, n, STDERR_FILENO);
}
#else
static void _write_backtrace(int sig) { (void)sig; }
#endif

/* ── Signal handler ───────────────────────────────────────────────────────── */

/* ── gh #259: graceful shutdown ────────────────────────────────────────────
 *
 * SIGTERM/SIGINT are the NORMAL way every supervisor, Docker and systemd stops
 * a service, and the old handler re-raised immediately: no WAL msync, so the
 * last tick's acknowledged writes were lost on a routine `kill`. The crash
 * path (SIGKILL, jetsam, faults) was well tested; the graceful one was not.
 *
 * These two signals now LATCH instead of killing. The event loop polls
 * `pion_shutdown_requested()`, flushes its WAL durably, and returns, which ends
 * the worker thread; main() falls out of pthread_join and exits normally, so
 * the atexit breadcrumb still runs.
 *
 * The alarm is the safety net and it is not optional: a latch with no deadline
 * turns any bug in the drain path into a process that ignores SIGTERM, which is
 * a worse failure than the data loss it replaces. When it fires, SIGALRM lands
 * on the ordinary crash handler and dies the old way, breadcrumb and all.
 *
 * Faults (SEGV/BUS/ILL/FPE/ABRT) and SIGQUIT/SIGXCPU/SIGHUP are unchanged: they
 * mean the process is already broken or is being asked to dump, so draining is
 * neither safe nor wanted. */
static volatile sig_atomic_t g_shutdown_requested = 0;
static int g_shutdown_grace_s = 10;

/* gh #424: a child process (the inference sidecar) to SIGTERM when this process
 * exits, so it does not outlive the server. The sidecar also polls getppid() as
 * the catch-all (covers an uncatchable SIGKILL of this process, which runs no
 * handler here); this makes the clean-exit and FATAL-bind paths immediate. */
static volatile sig_atomic_t g_child_pid = 0;

void pion_register_child_pid(int pid) { g_child_pid = (pid > 0) ? pid : 0; }

static void _reap_child(void) {
    if (g_child_pid > 0) {
        kill((pid_t)g_child_pid, SIGTERM);
        g_child_pid = 0;
    }
}

int pion_shutdown_requested(void) { return (int)g_shutdown_requested; }

/* Test/embedding hook: request the same drain without a signal (SHUTDOWN). */
void pion_request_shutdown(void) {
    if (!g_shutdown_requested) {
        g_shutdown_requested = 1;
        alarm((unsigned)g_shutdown_grace_s);
    }
}

static void _pion_crash_handler(int sig) {
    char line[PION_STATUS_RECORD_LEN];
    int o = 0;
    uint64_t now = _mono_secs();

    /* Latch and return — the drain happens on the event loop, not here. Only
     * the FIRST such signal latches; a second one falls through and kills, so
     * an impatient operator pressing Ctrl-C twice still gets an immediate exit
     * rather than being told to wait. */
    if ((sig == SIGTERM || sig == SIGINT) && !g_shutdown_requested) {
        g_shutdown_requested = 1;
        alarm((unsigned)g_shutdown_grace_s);
        static const char msg[] =
            "PION: shutdown requested, draining (WAL flush) — repeat the signal to force\n";
        if (g_log_fd >= 0) { ssize_t w = write(g_log_fd, msg, sizeof(msg) - 1); (void)w; }
        { ssize_t w = write(STDERR_FILENO, msg, sizeof(msg) - 1); (void)w; }
        return;
    }

    o += _lit(line + o, "PION EXIT: signal ");
    o += _lit(line + o, _signame(sig));
    o += _lit(line + o, " (");
    o += _u64(line + o, (uint64_t)sig);
    o += _lit(line + o, ") pid=");
    o += _u64(line + o, (uint64_t)g_pid);
    o += _lit(line + o, " port=");
    o += _u64(line + o, (uint64_t)g_port);
    o += _lit(line + o, " uptime_s=");
    o += _u64(line + o, now - g_start_mono);
    o += _lit(line + o, " rss_mb=");
    o += _u64(line + o, g_rss_bytes >> 20);
    o += _lit(line + o, " rss_peak_mb=");
    o += _u64(line + o, g_rss_peak_bytes >> 20);
    o += _lit(line + o, " rss_pct=");
    o += _u64(line + o, g_total_ram ? (g_rss_bytes * 100) / g_total_ram : 0);
    o += _lit(line + o, " hb_age_s=");
    o += _u64(line + o, g_hb_mono && now >= g_hb_mono ? now - g_hb_mono : 0);
    o += _lit(line + o, " ticks=");
    o += _u64(line + o, _tick_total());
    line[o++] = '\n';

    if (g_log_fd >= 0) { ssize_t w = write(g_log_fd, line, (size_t)o); (void)w; }
    { ssize_t w = write(STDERR_FILENO, line, (size_t)o); (void)w; }

    _write_status("signalled", sig);

    /* gh #424: take the sidecar down with us before re-raising. */
    _reap_child();

    /* Last, so a fault inside the unwinder cannot cost us the breadcrumb. */
    _write_backtrace(sig);

    /* Restore the default disposition and re-raise so the exit status (and any
     * core dump) is exactly what it would have been without this handler. */
    signal(sig, SIG_DFL);
    raise(sig);
}

static void _pion_crash_atexit(void) {
    if (!g_initialized || g_exiting) return;
    g_exiting = 1;
    _reap_child();  /* gh #424: no orphan sidecar on a clean or FATAL-bind exit */
    char line[256];
    int o = 0;
    o += _lit(line + o, "PION EXIT: clean shutdown pid=");
    o += _u64(line + o, (uint64_t)g_pid);
    o += _lit(line + o, " uptime_s=");
    o += _u64(line + o, _mono_secs() - g_start_mono);
    o += _lit(line + o, " rss_peak_mb=");
    o += _u64(line + o, g_rss_peak_bytes >> 20);
    line[o++] = '\n';
    if (g_log_fd >= 0) { ssize_t w = write(g_log_fd, line, (size_t)o); (void)w; }
    _write_status("exited", 0);
}

/* ── Public API (called from Mojo) ────────────────────────────────────────── */

/* Install handlers, open the crash log and status file, snapshot RAM size.
 * Call ONCE from the main thread before parallelize(). Returns 0 on success,
 * -1 if the crash log could not be opened (status file is best-effort).
 *
 * rss_warn_pct: 1..100 → warn threshold as a percentage of physical RAM.
 *               0 → default (70%). >100 → warning disabled. */
int pion_crash_init(const char *log_path, const char *status_path,
                    const char *version, int port, int workers,
                    int rss_warn_pct) {
    if (g_initialized) return 0;

    g_pid        = (int)getpid();
    g_port       = port;
    g_workers    = workers > 0 ? workers : 1;
    g_start_mono = _mono_secs();
    g_start_unix = (uint64_t)time(NULL);
    g_total_ram  = _total_ram();

    g_version[0] = '\0';
    if (version) {
        size_t n = strlen(version);
        if (n > sizeof(g_version) - 1) n = sizeof(g_version) - 1;
        memcpy(g_version, version, n);
        g_version[n] = '\0';
    }

    if (rss_warn_pct > 100) g_rss_warn = 0;
    else {
        int pct = rss_warn_pct > 0 ? rss_warn_pct : 70;
        g_rss_warn = g_total_ram ? (g_total_ram / 100) * (uint64_t)pct : 0;
    }

    for (int i = 0; i < PION_CRASH_MAX_WORKERS * PION_TICK_STRIDE; i++) g_ticks[i] = 0;

    if (log_path && log_path[0])
        g_log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (status_path && status_path[0])
        g_status_fd = open(status_path, O_WRONLY | O_CREAT, 0644);

    /* Startup line: pairs with the exit line so a log reader always sees a
     * start/stop pair — a missing stop IS the diagnostic. */
    if (g_log_fd >= 0) {
        char line[320];
        int o = 0;
        o += _lit(line + o, "PION START: pid=");
        o += _u64(line + o, (uint64_t)g_pid);
        o += _lit(line + o, " version=");
        o += _lit(line + o, g_version[0] ? g_version : "?");
        o += _lit(line + o, " port=");
        o += _u64(line + o, (uint64_t)g_port);
        o += _lit(line + o, " workers=");
        o += _u64(line + o, (uint64_t)g_workers);
        o += _lit(line + o, " unix_s=");
        o += _u64(line + o, g_start_unix);
        o += _lit(line + o, " total_ram_mb=");
        o += _u64(line + o, g_total_ram >> 20);
        line[o++] = '\n';
        ssize_t w = write(g_log_fd, line, (size_t)o);
        (void)w;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = _pion_crash_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_ONSTACK;   /* survive a stack-overflow SIGSEGV */

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS,  &sa, NULL);
    sigaction(SIGILL,  &sa, NULL);
    sigaction(SIGFPE,  &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    sigaction(SIGXCPU, &sa, NULL);
    /* gh #259: the graceful-shutdown deadline. Registered on the SAME handler,
     * and deliberately NOT latched above, so when the drain overruns its grace
     * period this dies exactly the way the pre-#259 code did — breadcrumb,
     * status record, backtrace — rather than hanging. */
    sigaction(SIGALRM, &sa, NULL);
    /* SIGHUP: the controlling terminal or session leader went away. Its default
     * action is terminate, and with no handler the process vanishes leaving NO
     * breadcrumb — a START line, no EXIT line, and a status file still saying
     * state=running. That is indistinguishable from an uncatchable jetsam/OOM
     * SIGKILL, which is precisely the distinction this shim exists to make. A
     * backgrounded server losing its parent shell is an ordinary way to die,
     * so record it rather than leaving a hole in the log. */
    sigaction(SIGHUP,  &sa, NULL);
    /* Note for the next person reading a "no EXIT line" death: SIGHUP is one
     * cause, but a Mojo-level ABORT (failed `Optional.value()`, an uncaught
     * raise in a -O0 build) is another — the Mojo runtime prints its own stack
     * trace to stderr and exits without routing through these handlers. Check
     * the server's stdout/stderr log before concluding the process was killed
     * from outside; that trace is the fastest attribution available. */

    atexit(_pion_crash_atexit);

    g_rss_bytes = _sample_rss();
    g_rss_peak_bytes = g_rss_bytes;
    g_hb_mono = g_start_mono;
    g_hb_unix = g_start_unix;
    g_initialized = 1;
    _write_status("running", 0);
    return g_log_fd >= 0 ? 0 : -1;
}

/* Per-tick liveness beat. Cheap: one non-atomic increment on the common path.
 * Once a second, whichever worker wins the CAS samples RSS and rewrites the
 * status record. Safe to call when uninitialised (no-op). */
/* gh #259: how long the drain gets before SIGALRM kills us anyway. Docker's
 * default SIGTERM grace is 10 s, so the default matches it — a drain that
 * outlives the supervisor's patience buys nothing. */
void pion_set_shutdown_grace(int seconds) {
    if (seconds > 0) g_shutdown_grace_s = seconds;
}

void pion_crash_heartbeat(int worker_id) {
    if (!g_initialized) return;
    if (worker_id >= 0 && worker_id < PION_CRASH_MAX_WORKERS)
        g_ticks[worker_id * PION_TICK_STRIDE]++;

    uint64_t now  = _mono_secs();
    uint64_t last = g_hb_mono;
    if (now == last) return;
    /* Exactly one worker per second performs the sample + write. */
    if (!__sync_bool_compare_and_swap(&g_hb_mono, last, now)) return;

    uint64_t rss = _sample_rss();
    g_rss_bytes = rss;
    if (rss > g_rss_peak_bytes) g_rss_peak_bytes = rss;
    g_hb_unix = (uint64_t)time(NULL);

    _write_status("running", 0);

    /* One-shot high-water warning: an operator gets told the host is about to
     * reap us, instead of finding a silent gap afterwards. Re-arms once RSS
     * falls back below 90% of the threshold. */
    if (g_rss_warn) {
        if (!g_rss_warned && rss >= g_rss_warn) {
            g_rss_warned = 1;
            char line[256];
            int o = 0;
            o += _lit(line + o, "PION WARN: rss_mb=");
            o += _u64(line + o, rss >> 20);
            o += _lit(line + o, " is ");
            o += _u64(line + o, g_total_ram ? (rss * 100) / g_total_ram : 0);
            o += _lit(line + o, "% of total_ram_mb=");
            o += _u64(line + o, g_total_ram >> 20);
            o += _lit(line + o, " — OS may kill this process (jetsam/OOM) without a catchable signal\n");
            if (g_log_fd >= 0) { ssize_t w = write(g_log_fd, line, (size_t)o); (void)w; }
            { ssize_t w = write(STDERR_FILENO, line, (size_t)o); (void)w; }
        } else if (g_rss_warned && rss < (g_rss_warn / 10) * 9) {
            g_rss_warned = 0;
        }
    }
}

/* Last sampled RSS in bytes (0 when uninitialised). Used by INFO. */
uint64_t pion_crash_rss_bytes(void) {
    return g_rss_bytes;
}

uint64_t pion_crash_rss_peak_bytes(void) {
    return g_rss_peak_bytes;
}

uint64_t pion_crash_total_ram_bytes(void) {
    return g_total_ram;
}

/* gh #262: RSS sampled NOW, for INFO. The heartbeat's cached g_rss_bytes is
 * 0 under --no-crash-log and up to 1 s stale otherwise. INFO is a slow-path
 * command, so one task_info()/statm read per call is fine. NEVER call this
 * from a signal handler. */
uint64_t pion_rss_sample_now(void) {
    uint64_t rss = _sample_rss();
    if (rss > g_rss_peak_bytes) g_rss_peak_bytes = rss;
    return rss;
}

/* ── gh #261: --maxmemory ────────────────────────────────────────────────────
 * One process-wide limit on RESIDENT memory, the number jetsam and the Linux
 * OOM killer act on. Redis limits its allocator's used_memory instead; Pion's
 * slab allocators and mmap arenas make that figure a poor proxy for what gets
 * the process killed, and INFO already reports RSS as used_memory (gh #262).
 *
 * Workers poll pion_maxmemory_check() from their housekeeping tick. At most one
 * of them samples RSS per 100 ms (CAS on the timestamp, same scheme as the
 * heartbeat); the rest read the cached verdict. With no limit set the call is
 * one load and a return, and the dispatch path never calls it at all. */
static volatile uint64_t g_maxmemory      = 0;   /* bytes; 0 = unlimited */
static volatile uint64_t g_mm_last_ms     = 0;
static volatile int      g_over_maxmemory = 0;

static uint64_t _mono_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
    return 0;
}

void pion_set_maxmemory(uint64_t bytes) {
    g_maxmemory = bytes;
    g_mm_last_ms = 0;        /* re-sample on the next check */
    if (bytes == 0) g_over_maxmemory = 0;
}

uint64_t pion_get_maxmemory(void) { return g_maxmemory; }

uint64_t pion_physical_ram_bytes(void) { return _total_ram(); }

int pion_maxmemory_check(void) {
    uint64_t lim = g_maxmemory;
    if (lim == 0) return 0;
    uint64_t now = _mono_ms();
    uint64_t last = g_mm_last_ms;
    if (last != 0 && now - last < 100) return g_over_maxmemory;
    if (__sync_bool_compare_and_swap(&g_mm_last_ms, last, now ? now : 1)) {
        uint64_t rss = _sample_rss();
        if (rss > g_rss_peak_bytes) g_rss_peak_bytes = rss;
        int over = rss > lim;
        if (over != g_over_maxmemory) {
            /* Say so on every crossing: an operator seeing -OOM needs to know
             * when it started and what the process measured, not guess. */
            char line[200];
            int o = 0;
            o += _lit(line + o, over ? "PION WARN: rss_mb=" : "PION: rss_mb=");
            o += _u64(line + o, rss >> 20);
            o += _lit(line + o, over ? " > maxmemory_mb=" : " <= maxmemory_mb=");
            o += _u64(line + o, lim >> 20);
            o += _lit(line + o, over ? " — refusing memory-growing writes with -OOM\n"
                                     : " — writes accepted again\n");
            if (g_log_fd >= 0) { ssize_t w = write(g_log_fd, line, (size_t)o); (void)w; }
            { ssize_t w = write(STDERR_FILENO, line, (size_t)o); (void)w; }
        }
        g_over_maxmemory = over;
    }
    return g_over_maxmemory;
}
