#!/usr/bin/env python3
"""RSS must stay bounded under churn — with HEAP-sized members, and through the
store/clone paths.

WHY
test_gh369_container_free.py proved containers are freed on drain and DEL, but
every member it used was 1 byte, i.e. inline (SSO): a heap payload that is
never freed — or freed twice — is invisible to it. The ownership fixes of
this change (SUNION/ZUNION temporaries no longer free borrowed members; *STORE
clones into its destination; SORT STORE builds a list) each add or remove a
free, so each needs a churn loop that would show a leak.

Method (as gh #369): a warm-up of the same workload, then RSS growth over
CYCLES cycles must stay under BOUND_KB. 8 MiB over 20,000 cycles is 0.4 KB a
cycle — below one leaked 48-byte member per cycle times the members each
cycle touches, so a per-cycle leak of any member fails.

Also: a store command whose DESTINATION already exists must free the
container it replaces, or re-running the same `SUNIONSTORE d a b` leaks one
container per call.

    python3 tests/test_soak_rss_bounded.py [./pion-server]
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready, wait_port_free  # noqa: E402

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BINARY = os.path.abspath(_ARGS[0] if _ARGS else os.environ.get("PION_BIN", "./pion-server"))
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None   # substring of a workload name
PORT = 6476
CYCLES = 20000
# 512 KB over 20K cycles = 0.025 KB a cycle. The clean workloads measure 0;
# gh #369's 8 MiB bound let every leak below 0.4 KB a cycle pass, and most
# heap-member paths leak at 0.05-0.7 KB a cycle (all LINEAR in CYCLES: 3x the
# cycles gives 3x the growth, so it is not allocator caching).
BOUND_KB = 512
M = [b"heap-member-%02d-" % i + b"m" * 32 for i in range(6)]      # 48 bytes: heap, not SSO
V = b"v" * 64
B2K = b"q" * 2048


def rss_kb(pid):
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())


def run(c: Conn, cmds):
    out = []
    for k in range(0, len(cmds), 500):
        out += c.pipeline(cmds[k:k + 500])
    return out


def growth(proc, c, make_cycle):
    run(c, [cmd for i in range(2000) for cmd in make_cycle(i)])
    base = rss_kb(proc.pid)
    # i starts past the warm-up: a workload's i == 0 setup step (build the
    # source set, the list to COPY) must not run again inside the window.
    replies = run(c, [cmd for i in range(2000, 2000 + CYCLES) for cmd in make_cycle(i)])
    errors = [r for r in replies if isinstance(r, RespError)]
    return rss_kb(proc.pid) - base, errors


WORKLOADS = {
    "ZADD/ZPOPMIN, heap member": lambda i: [("ZADD", "z", "1", M[0]), ("ZPOPMIN", "z")],
    "RPUSH/LPOP, heap element": lambda i: [("RPUSH", "l", M[0]), ("LPOP", "l")],
    "SADD/SPOP, heap member": lambda i: [("SADD", "s", M[0]), ("SPOP", "s")],
    # Elements over 64 bytes make the list segmented, where a pop hands back
    # the stored element itself: never freed, a 100 KB-message queue grew by
    # ~92 KB per message.
    "RPUSH/LPOP + RPUSH/RPOP, 2 KB element": lambda i: [("RPUSH", "lb", B2K), ("LPOP", "lb"),
                                                          ("RPUSH", "lb", B2K), ("RPOP", "lb")],
    "multi-value RPUSH + LPOP/RPOP count, 2 KB (slow path)": lambda i: [
        ("RPUSH", "lc", B2K, B2K, B2K), ("LPOP", "lc", "2"), ("RPOP", "lc", "1")],
    "HSET/HDEL, heap field+value": lambda i: [("HSET", "h", M[0], V), ("HDEL", "h", M[0])],
    "SET heap value EX / DEL (TTL entry)": lambda i: [("SET", "t", V, "EX", "100"), ("DEL", "t")],
    "EXPIRE churn on a heap key": lambda i: [("SET", M[1], V), ("EXPIRE", M[1], "100"), ("DEL", M[1])],
    "MSET/DEL heap values": lambda i: [("MSET", "a1", V, "a2", V), ("DEL", "a1", "a2")],
    "APPEND grow / DEL": lambda i: [("APPEND", "ap", V), ("APPEND", "ap", V), ("DEL", "ap")],
    "segmented list build / DEL": lambda i: ([("RPUSH", "seg", *[b"%04d" % j for j in range(1100)]), ("DEL", "seg")]
                                             if i % 50 == 0 else [("PING",)]),
    # read-only set operations over heap members: must neither leak nor free
    "SUNION / ZUNION / ZINTER / ZDIFF (read-only)": lambda i: (
        [("SADD", "ra", *M[:4]), ("SADD", "rb", *M[2:]), ("ZADD", "rz", "1", M[0], "2", M[1]),
         ("ZADD", "rz2", "1", M[1], "3", M[2])] if i == 0 else
        [("SUNION", "ra", "rb"), ("ZUNION", "2", "rz", "rz2"), ("ZINTER", "2", "rz", "rz2"),
         ("ZDIFF", "2", "rz", "rz2")]),
    # stores into a destination that is DELeted each cycle
    "S*STORE into a fresh destination": lambda i: (
        [("SADD", "sa", *M[:4]), ("SADD", "sb", *M[2:])] if i == 0 else
        [("SUNIONSTORE", "sd", "sa", "sb"), ("DEL", "sd"), ("SINTERSTORE", "sd", "sa", "sb"), ("DEL", "sd"),
         ("SDIFFSTORE", "sd", "sa", "sb"), ("DEL", "sd")]),
    "Z*STORE into a fresh destination": lambda i: (
        [("ZADD", "za", "1", M[0], "2", M[1]), ("ZADD", "zb", "2", M[1], "3", M[2])] if i == 0 else
        [("ZUNIONSTORE", "zd", "2", "za", "zb"), ("DEL", "zd"), ("ZINTERSTORE", "zd", "2", "za", "zb"), ("DEL", "zd"),
         ("ZDIFFSTORE", "zd", "2", "za", "zb"), ("DEL", "zd"), ("ZRANGESTORE", "zd", "za", "0", "-1"), ("DEL", "zd")]),
    # stores OVER an existing destination: the replaced container must be freed
    "SUNIONSTORE over an existing destination": lambda i: (
        [("SADD", "oa", *M[:4]), ("SADD", "ob", *M[2:])] if i == 0 else [("SUNIONSTORE", "od", "oa", "ob")]),
    "ZUNIONSTORE over an existing destination": lambda i: (
        [("ZADD", "oza", "1", M[0], "2", M[1]), ("ZADD", "ozb", "2", M[1])] if i == 0 else
        [("ZUNIONSTORE", "ozd", "2", "oza", "ozb")]),
    "SORT STORE over an existing destination": lambda i: (
        [("RPUSH", "so", "3", "1", "2")] if i == 0 else [("SORT", "so", "STORE", "sod")]),
    "COPY REPLACE over an existing destination": lambda i: (
        [("RPUSH", "cl", *M[:3])] if i == 0 else [("COPY", "cl", "cld", "REPLACE")]),
    # gh #394, found while fixing the list above — each leaked linearly:
    # DEL never freed these three types at all.
    "HLL create / DEL": lambda i: [("PFADD", "hl", "a", "b"), ("DEL", "hl")],
    "stream create / DEL": lambda i: [("XADD", "st", "*", "f", V), ("DEL", "st")],
    "bitmap create / DEL": lambda i: [("SETBIT", "bm", "8191", "1"), ("DEL", "bm")],   # 1 KB
    # XADD MAXLEN only FLAGGED trimmed entries: their data was never freed and
    # the entry array never shrank, so a capped log grew without bound.
    "capped stream (XADD MAXLEN 10)": lambda i: [("XADD", "cs", "MAXLEN", "10", "*", "f", V)],
    # ZPOPMAX and ZREM rebuilt the whole set per call and freed nothing they
    # removed; pop_min left one dict entry per distinct member ever popped.
    "ZADD/ZPOPMAX, heap member": lambda i: [("ZADD", "zx", "1", M[0]), ("ZPOPMAX", "zx")],
    "ZADD/ZREM, heap member": lambda i: [("ZADD", "zr", "1", M[0]), ("ZREM", "zr", M[0])],
    "ZADD/ZPOPMIN, a DISTINCT member each cycle": lambda i: [
        ("ZADD", "zu", "1", b"job-%08d-" % i + b"j" * 32), ("ZPOPMIN", "zu")],
    # A slow-path SET appended to an in-memory Raft log nothing ever read or
    # trimmed. STRLEN (slow path only) first, so the SETs behind it in the
    # batch run on the slow path too.
    "slow-path SET (Raft log)": lambda i: [("STRLEN", "x")] + [("SET", "rs", V)] * 5,
    # A plain read of a key longer than 23 bytes (SSO ends at 23).
    # 20 calls a cycle: at 2 the ~48 B/call leak (~2 MB) was absorbed by
    # memory earlier workloads had freed, and read as fixed.
    "GET / EXISTS on a 48-byte key": lambda i: (
        [("SET", LONG_KEY, "v")] if i == 0 else [("GET", LONG_KEY), ("EXISTS", LONG_KEY)] * 10),
}

LONG_KEY = b"user:session:" + b"x" * 35    # 48 bytes

# Known leaks (each grows linearly with CYCLES — not fragmentation). A known
# workload that fails is XFAIL; one that PASSES is XPASS and fails the run, so
# the entry must be removed on purpose once the leak is fixed. Empty since
# gh #394: all 15 listed there were fixed.
KNOWN = {}


def start(d):
    # --no-wal: the log grows with every command and would swamp the RSS signal.
    proc = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-crash-log",
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=d, stdout=open(os.path.join(d, "log"), "a"), stderr=subprocess.STDOUT)
    try:
        wait_ready(PORT, 30, proc=proc)
    except RuntimeError:
        print("server failed to start; log tail:\n" + open(os.path.join(d, "log")).read()[-1500:])
        raise
    return proc


def main():
    d = tempfile.mkdtemp(prefix="pion_soak_")
    fails = []
    print(f"[soak] {BINARY}: {CYCLES} cycles per workload, bound {BOUND_KB} KB, fresh server each")
    try:
        for name, mk in WORKLOADS.items():
            if ONLY and ONLY not in name:
                continue
            # A FRESH server per workload: on a shared one, memory an earlier
            # workload freed is reused first and absorbs a later leak — the
            # SPOP heap-member leak (~1 MB here) read as 0 KB that way.
            proc = start(d)
            try:
                c = Conn(PORT, timeout=60)
                try:
                    grown, errors = growth(proc, c, mk)
                except (ConnectionError, TimeoutError) as e:
                    fails.append(f"server died or hung during: {name} ({type(e).__name__})")
                    print(f"  FAIL  {name:48} server died or hung ({type(e).__name__})")
                    continue
                ok = grown < BOUND_KB and not errors
                known = KNOWN.get(name)
                # A known leak counts as FIXED only when clearly below the bound
                # (a quarter of it): some sit near the line and would otherwise flap.
                fixed = grown < BOUND_KB // 4 and not errors
                verdict = ("XPASS" if fixed else "XFAIL") if known else ("PASS" if ok else "FAIL")
                print(f"  {verdict:5} {name:48} grew {grown:7d} KB"
                      + (f"  ({known})" if known else "")
                      + (f"  errors: {errors[:2]}" if errors else ""))
                if verdict in ("FAIL", "XPASS"):
                    fails.append(name + (f" — XPASS, remove it from KNOWN ({known})" if verdict == "XPASS" else ""))
                if proc.poll() is not None:
                    fails.append(f"server died during: {name}")
                else:
                    c.assert_in_sync()
            finally:
                proc.kill(); proc.wait()
                wait_port_free(PORT)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
