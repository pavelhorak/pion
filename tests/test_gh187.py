#!/usr/bin/env python3
"""gh #187: SlabSkipList member dedup + pop_min node deallocation.

Pre-fix defects:
  1. ZADD of an existing member appended a duplicate node and replied :1 —
     the zset grew without bound on member overwrite and ZSCORE/ZRANGE
     semantics depended on traversal order.
  2. pop_min never returned the removed node to the SlabAllocator — every
     ZPOPMIN leaked one 176 B node.

Covers: fast-path single-pair ZADD, slow-path multi-pair ZADD, ZSCORE after
overwrite, ZRANGE order after score move, ZPOPMIN dedup interaction, GEOADD
member update, WAL-replay dedup (restart), and an RSS-bounded ZADD/ZPOPMIN
churn loop for the node leak.
"""
import os, socket, subprocess, sys, time, shutil, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader, wait_ready_pid, wait_port_free  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1979
WORKDIR = f"/tmp/pion_gh187_test_{PORT}"

def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str): a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)

def send(sock, *args):
    # Exactly one parsed reply: a single recv() with a swallowed timeout
    # used to return "" and let the late reply answer the NEXT command.
    sock.sendall(encode(args))
    return reader(sock).read_raw().decode(errors="replace")

def _await_ready(s, deadline_s=30.0):
    """TCP accept is not readiness: the server listens BEFORE it initialises
    (so clients queue instead of being refused), and after a restart its first
    reply waits for the 10M-slot map and the WAL replay. send() gives up after
    2 s, so a slow first reply read as '' and every later check read the reply
    of the command before it. Block until PING answers, then assert."""
    s.settimeout(deadline_s)
    s.sendall(b"*1\r\n$4\r\nPING\r\n")
    buf = b""
    while b"+PONG\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            raise RuntimeError("server closed the connection before it was ready")
        buf += chunk
    s.settimeout(None)
    return s


def connect(proc):
    # Ready = THIS process answering, not a killed server's lingering listener (#27).
    wait_ready_pid(PORT, proc, 60)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            s = socket.socket(); s.settimeout(1.0)
            s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            return _await_ready(s)
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")

def start():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def rss_kb(pid):
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
    return int(out.strip())

passed = failed = 0
def check(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS {name}")
    else:
        failed += 1; print(f"  FAIL {name} {detail}")

def xfail(cond, name, issue, detail=""):
    """A known bug, pinned: XFAIL while `cond` is false; XPASS fails the run."""
    global failed
    if cond:
        failed += 1; print(f"  XPASS {name} — {issue} looks fixed: make this a check()")
    else:
        print(f"  XFAIL {name} ({issue}) {detail}")

def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = start()
    try:
        s = connect(proc)

        # ── 1. fast-path single-pair ZADD dedup ──
        r = send(s, "ZADD", "z", "1", "m")
        check(":1" in r, "first ZADD returns 1", f"got {r!r}")
        r = send(s, "ZADD", "z", "2", "m")
        check(":0" in r, "overwrite ZADD returns 0", f"got {r!r}")
        r = send(s, "ZCARD", "z")
        check(":1" in r, "ZCARD is 1 after overwrite", f"got {r!r}")
        r = send(s, "ZSCORE", "z", "m")
        check("2" in r, "ZSCORE reflects new score", f"got {r!r}")
        r = send(s, "ZADD", "z", "2", "m")
        check(":0" in r, "equal-score re-add returns 0", f"got {r!r}")
        r = send(s, "ZCARD", "z")
        check(":1" in r, "ZCARD still 1 after equal-score re-add", f"got {r!r}")

        # ── 2. slow-path multi-pair ZADD dedup ──
        r = send(s, "ZADD", "zm", "1", "a", "2", "b", "3", "a")
        check(":2" in r, "multi-pair ZADD counts new members only", f"got {r!r}")
        r = send(s, "ZCARD", "zm")
        check(":2" in r, "ZCARD is 2 after multi-pair dup", f"got {r!r}")
        r = send(s, "ZSCORE", "zm", "a")
        check("3" in r, "last score wins within one multi-pair ZADD", f"got {r!r}")

        # ── 3. ZRANGE order after a score move ──
        send(s, "ZADD", "zo", "1", "x")
        send(s, "ZADD", "zo", "2", "y")
        send(s, "ZADD", "zo", "9", "x")   # x moves behind y
        r = send(s, "ZRANGE", "zo", "0", "-1")
        ix, iy = r.find("x"), r.find("y")
        check(ix > iy >= 0, "ZRANGE reorders moved member", f"got {r!r}")
        check(r.count("x") == 1, "no duplicate node in ZRANGE", f"got {r!r}")

        # ── 4. ZPOPMIN pops the updated member exactly once ──
        send(s, "ZADD", "zp", "5", "solo")
        send(s, "ZADD", "zp", "1", "solo")
        r = send(s, "ZPOPMIN", "zp")
        check("solo" in r and "1" in r, "ZPOPMIN returns updated score", f"got {r!r}")
        r = send(s, "ZPOPMIN", "zp")
        check("solo" not in r, "member fully gone after pop", f"got {r!r}")
        # member can be re-added after pop (dict entry was removed)
        r = send(s, "ZADD", "zp", "7", "solo")
        check(":1" in r, "re-add after pop counts as new", f"got {r!r}")

        # ── 5. GEOADD member update ──
        r = send(s, "GEOADD", "geo", "13.361389", "38.115556", "Palermo")
        check(":1" in r, "first GEOADD returns 1", f"got {r!r}")
        r = send(s, "GEOADD", "geo", "15.087269", "37.502669", "Palermo")
        check(":0" in r, "GEOADD update returns 0", f"got {r!r}")

        # ── 6. WAL replay dedups (ZADD m s1; ZADD m s2 replays to one member) ──
        proc.kill(); proc.wait(timeout=5)
        wait_port_free(PORT)
        proc = start()
        s = connect(proc)
        r = send(s, "ZCARD", "z")
        check(":1" in r, "ZCARD is 1 after WAL replay", f"got {r!r}")
        r = send(s, "ZSCORE", "z", "m")
        check("2" in r, "replayed score is the last write", f"got {r!r}")

        # ── 7. node-leak churn: ZADD/ZPOPMIN loop, RSS must stay bounded ──
        # Pre-fix each cycle leaked one 176 B node → 60K cycles ≈ 10.6 MB.
        for i in range(2000):   # warm-up: allocator slabs, buffers
            send(s, "ZADD", "leak", "1", "n")
            send(s, "ZPOPMIN", "leak")
        base = rss_kb(proc.pid)
        for i in range(60000):
            send(s, "ZADD", "leak", "1", "n")
            send(s, "ZPOPMIN", "leak")
        grown = rss_kb(proc.pid) - base
        check(grown < 5 * 1024, "RSS bounded over 60K ZADD/ZPOPMIN cycles (gh #369)",
              f"grew {grown} KB")

        s.close()
    finally:
        proc.kill()
        proc.wait(timeout=5)
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
