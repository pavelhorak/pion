#!/usr/bin/env python3
"""gh #259 regression test — SIGTERM/SIGINT must flush the WAL before exiting.

The handler in `crash_wrap.c` wrote its breadcrumb and then did
`signal(sig, SIG_DFL); raise(sig)`: no msync, no drain, and no `SHUTDOWN`
command existed at all, so the only way to stop a server was a signal.

SCOPE — READ THIS BEFORE STRENGTHENING THE ASSERTIONS.
The issue says a plain `kill` "can lose the last tick's acknowledged writes".
Measured against the pre-fix binary, it does NOT, and the reason is structural:
the WAL is a **file-backed mmap**, so its dirty pages belong to the kernel and
are written back after the process dies. Process death — SIGTERM, SIGINT, even
SIGKILL — does not discard them. (gh #170/#174 already rely on this: their
SIGKILL recovery tests pass.) The exposure is real only for MACHINE-level
failure — power loss or a kernel panic — in the seconds after exit, which this
test cannot stage.

So `all N acked writes survive` below is a genuine invariant and is asserted,
but it is NOT what distinguishes fixed from broken; it passes either way. The
discriminating assertions are the exit status, the breadcrumb, and SHUTDOWN
existing at all. What the fix buys is an explicit durability barrier instead of
reliance on kernel writeback, plus a stop path a supervisor can tell apart from
a crash.

What is asserted:

  [1] SIGTERM  — process exits 0 with a `PION EXIT` breadcrumb rather than dying
      by signal (pre-fix: rc == -15), and every acked write is back.
  [2] SIGINT   — same, since Ctrl-C is the interactive spelling of the same thing
      (pre-fix: rc == -2).
  [3] SHUTDOWN — the command exists at all (it was in admin.mojo's docstring and
      implemented nowhere; pre-fix it answers `-ERR unknown command`), routes to
      the same drain, and sends NO reply, which is what real Redis does.
  [4] SHUTDOWN NOSAVE — also exits, and does not error.
  [5] a SECOND signal forces immediate exit rather than waiting on the drain, so
      an impatient operator is never stuck. This is the guard on the latch: a
      shutdown request that could hang would be worse than the lossy re-raise
      it replaced.

Measured: 19/19 on the fix, 9 failures on 0.980+3a49edf.

Usage: python3 tests/test_gh259_graceful_shutdown.py [./pion-server]
"""
import os, signal, socket, subprocess, sys, time, shutil

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1993
WORKDIR = f"/tmp/pion_gh259_test_{PORT}"
KEYS = 300

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
    def __init__(self, port, timeout=15):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        self.sock.sendall(encode(args)); return self._read()

    def send_only(self, *args):
        self.sock.sendall(encode(args))

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


def spawn():
    return subprocess.Popen(
        [BINARY, "-p", str(PORT), "-w", "1", "--no-auto-detect", "--no-auto-embed"],
        cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect(timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: return Client(PORT)
        except OSError: time.sleep(0.25)
    raise RuntimeError("server did not come up")


def write_keys(c, prefix):
    for i in range(KEYS):
        r = c("SET", f"{prefix}:{i}", f"val-{i}")
        if r != b"+OK":
            raise RuntimeError(f"SET refused: {r!r}")
    return [f"{prefix}:{i}" for i in range(KEYS)]


def count_present(c, keys):
    return sum(1 for k in keys if c("GET", k) is not None)


def crash_log_text():
    for name in os.listdir(WORKDIR):
        if name.endswith(".crash.log"):
            return open(os.path.join(WORKDIR, name)).read()
    return ""


def cycle(stop_fn, label, expect_exit_zero=True):
    """Write KEYS, stop the server via stop_fn, restart, count survivors."""
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn()
    c = connect()
    keys = write_keys(c, "g")
    rc = stop_fn(proc, c)
    try: c.close()
    except OSError: pass

    check(f"{label}: process exited", rc is not None, "still running after grace period")
    if expect_exit_zero:
        # A drained exit returns through main(), so status is 0 — not -SIGTERM.
        check(f"{label}: exit status is clean (0), not death-by-signal",
              rc == 0, f"got {rc}")
        log = crash_log_text()
        check(f"{label}: breadcrumb records a clean exit",
              "PION EXIT" in log, "no PION EXIT line in the crash log")

    proc2 = spawn()
    try:
        c2 = connect()
        present = count_present(c2, keys)
        check(f"{label}: all {KEYS} acked writes survive", present == KEYS,
              f"only {present}/{KEYS} came back — the drain did not happen")
        c2.close()
    finally:
        proc2.terminate()
        try: proc2.wait(timeout=20)
        except subprocess.TimeoutExpired: proc2.kill(); proc2.wait(timeout=10)


def stop_signal(sig):
    def go(proc, _c):
        proc.send_signal(sig)
        try: return proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10); return None
    return go


def stop_command(*args):
    def go(proc, c):
        # Redis sends no reply to a successful SHUTDOWN; the client just sees
        # the close. Anything arriving here that is not a close is a bug.
        c.send_only(*args)
        try: return proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10); return None
    return go


def test_shutdown_sends_no_reply():
    print("\n[3b] SHUTDOWN sends no reply (matches Redis)")
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn()
    try:
        c = connect()
        c("SET", "x", "1")
        c.send_only("SHUTDOWN")
        c.sock.settimeout(25)
        try:
            data = c.sock.recv(256)
            check("no payload before close", data == b"",
                  f"server replied {data[:60]!r} — a client would desync")
        except socket.timeout:
            check("no payload before close", False, "connection neither replied nor closed")
        except OSError:
            check("no payload before close", True)
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=10)


def test_second_signal_forces():
    print("\n[5] A second signal forces immediate exit")
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn()
    try:
        c = connect(); write_keys(c, "f"); c.close()
        proc.send_signal(signal.SIGTERM)
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=25)
            check("exits after a repeated signal", True)
            # Either path is acceptable: it may have drained before the second
            # signal landed (0) or been killed by it (negative). What must NOT
            # happen is hanging.
            check("second signal did not hang the process", rc is not None, f"rc={rc}")
        except subprocess.TimeoutExpired:
            check("exits after a repeated signal", False, "still running after 25s")
            proc.kill(); proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=10)


def main():
    try:
        print("[1] SIGTERM drains the WAL")
        cycle(stop_signal(signal.SIGTERM), "SIGTERM")
        print("\n[2] SIGINT drains the WAL")
        cycle(stop_signal(signal.SIGINT), "SIGINT")
        print("\n[3] SHUTDOWN drains the WAL")
        cycle(stop_command("SHUTDOWN"), "SHUTDOWN")
        test_shutdown_sends_no_reply()
        print("\n[4] SHUTDOWN NOSAVE drains the WAL")
        cycle(stop_command("SHUTDOWN", "NOSAVE"), "SHUTDOWN NOSAVE")
        test_second_signal_forces()
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
