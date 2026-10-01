#!/usr/bin/env python3
"""gh #253 regression test — multi-worker mode must be an explicit opt-in.

Pion workers are shared-nothing: each owns a PRIVATE keyspace, and a connection
is bound to whichever worker won the accept() race. A `SET` acknowledged `+OK`
on one connection is therefore invisible to a `GET` on another. That is the
design (the cross-worker bus was deleted in gh #85) — but the default was `-w 8`
on desktop / `-w 16` on cloud, and every default connection pool (redis-py,
Jedis, go-redis, ioredis) opens more than one connection. The result was silent
read-your-writes violation with no error anywhere.

Measured on 0.979 before the fence, `-w 4`, 16 CONCURRENTLY-opened connections,
6 trials of one SET on conn 0 then GET on conns 1-15: **41/90 nil reads**.

The measurement trap this test is built around: connections opened SERIALLY all
land on one worker and mask the problem completely (0/12 nils in the same
session). Phase 1 therefore opens its connections concurrently — a serial probe
here would pass on the broken build and prove nothing.

What is asserted:
  1. The DEFAULT server (no -w) is one coherent keyspace under a concurrent pool.
  2. `-w N > 1` without --independent-workers REFUSES to start, exit 1, and
     leaves nothing listening.
  3. `-w N > 1` WITH --independent-workers starts, serves, and prints the banner.
  4. A cap that collapses the count back to 1 is not fenced (the fence reads the
     FINAL worker count, not the requested one).

Usage: python3 tests/test_gh253_multiworker_fence.py [./pion-server]
"""
import os, socket, subprocess, sys, time, shutil
from concurrent.futures import ThreadPoolExecutor

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1991
WORKDIR = f"/tmp/pion_gh253_test_{PORT}"

CONNS = 16       # concurrent connections per trial
TRIALS = 6

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        a = str(a).encode()
        parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
    return b"".join(parts)


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        self.sock.sendall(encode(args)); return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        t = line[:1]
        if t in b"+-:":
            return line[:-2]
        if t == b"$":
            n = int(line[1:]); return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(line[1:]); return None if n == -1 else [self._read() for _ in range(n)]
        raise RuntimeError("unparseable reply " + repr(line))

    def close(self):
        try: self.f.close(); self.sock.close()
        except OSError: pass


def spawn(extra):
    return subprocess.Popen([BINARY, "-p", str(PORT), "--no-auto-detect",
                             "--no-auto-embed"] + extra,
                            cwd=WORKDIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def connect(timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: return Client(PORT)
        except OSError: time.sleep(0.25)
    raise RuntimeError("server did not come up")


def port_is_open():
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=1).close(); return True
    except OSError:
        return False


def nil_reads():
    """One SET on conn 0, then GET the same key on every other connection.

    The connections are opened CONCURRENTLY on purpose — serially-opened ones
    all land on a single worker and would hide a split keyspace entirely.
    Returns (nils, reads).
    """
    nils = reads = 0
    for t in range(TRIALS):
        with ThreadPoolExecutor(max_workers=CONNS) as ex:
            clients = list(ex.map(lambda _: Client(PORT), range(CONNS)))
        try:
            key, val = f"gh253:{t}", f"value-{t}"
            if clients[0]("SET", key, val) != b"+OK":
                raise RuntimeError("SET was not acknowledged")
            for c in clients[1:]:
                reads += 1
                if c("GET", key) != val.encode():
                    nils += 1
        finally:
            for c in clients:
                c.close()
    return nils, reads


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)

    # ── Phase 1: the DEFAULT server is one coherent keyspace ────────────────
    proc = spawn([])
    try:
        connect().close()
        nils, reads = nil_reads()
        check("default server: a concurrent pool reads its own writes",
              nils == 0, f"{nils}/{reads} nil reads — the default is multi-worker again")
    finally:
        proc.kill(); proc.wait(timeout=10)
        time.sleep(0.5)

    # ── Phase 2: -w > 1 without the flag REFUSES ────────────────────────────
    proc = spawn(["-w", "4"])
    out, _ = proc.communicate(timeout=60)
    check("-w 4 without --independent-workers exits non-zero",
          proc.returncode != 0, f"returncode={proc.returncode}")
    check("-w 4 without --independent-workers names the flag",
          "--independent-workers" in out and "FATAL" in out,
          repr(out[-300:]))
    check("-w 4 without --independent-workers leaves nothing listening",
          not port_is_open())

    # ── Phase 3: -w > 1 WITH the flag starts, serves, and warns ─────────────
    proc = spawn(["-w", "4", "--independent-workers"])
    try:
        c = connect()
        check("-w 4 --independent-workers serves", c("PING") == b"+PONG")
        c.close()
        # Evidence, not an assertion: with 4 independent keyspaces some fraction
        # of a concurrent pool's reads MUST miss, but the exact count depends on
        # how the kernel spread the accepts, so asserting a number here would
        # flake. Printed so a reader can see what the flag actually buys.
        nils, reads = nil_reads()
        print(f"  INFO  --independent-workers: {nils}/{reads} nil reads "
              f"across {CONNS} concurrent connections (this is the documented "
              f"semantics, not a failure)")
    finally:
        proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        proc.wait(timeout=10)
        check("-w 4 --independent-workers prints the keyspace banner",
              "INDEPENDENT WORKERS" in out and "SEPARATE KEYSPACES" in out,
              repr(out[:400]))
        time.sleep(0.5)

    # ── Phase 4: a cap back to 1 is not fenced ──────────────────────────────
    # --inference collapses any -w N to 1. The fence reads the FINAL count, so
    # this must start normally rather than refusing on the requested 4.
    # (Checked without spawning the sidecar: --profile ai does the same via
    # apply_profile, and needs no external process.)
    proc = spawn(["-w", "4", "--profile", "ai"])
    try:
        c = connect()
        check("a profile that caps workers back to 1 is NOT fenced",
              c("PING") == b"+PONG")
        c.close()
    except Exception as e:
        check("a profile that caps workers back to 1 is NOT fenced", False, str(e))
    finally:
        proc.kill(); proc.wait(timeout=10)

    shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
