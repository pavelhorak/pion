#!/usr/bin/env python3
"""gh #258 regression test — bind defaults and keeping the password out of argv.

Before this, every listener bound INADDR_ANY, because the sockaddr was memset
to zero and nobody filled in `sin_addr`. The comment in `server.mojo` even said
so: "no need to set addr[4-7] as they are already 0 from memset". That is five
listeners on all interfaces — the RESP port, port+1 (binary lane under
--kvcache), port+10000 (WAL replication, which has NO authentication), and the
gossip/Raft pair. On a laptop or an unfirewalled cloud box that is the whole
keyspace, reachable from the network, by default.

And `--requirepass` was argv-only, so the password showed up in `ps` and
/proc/<pid>/cmdline for every local user.

What is asserted:

  [1] default with no password  -> loopback only
  [2] default WITH a password   -> all interfaces (Redis protected-mode
                                   precedent: a password signals intent to
                                   serve remotely)
  [3] explicit --bind wins over both
  [4] an INVALID --bind exits non-zero and listens on NOTHING. This is the one
      that matters most: a typo silently falling back would EXPOSE the server,
      the exact opposite of the flag's purpose.
  [5] --requirepass-file works, strips the trailing newline `echo` leaves, and
      keeps the password out of argv
  [6] PION_REQUIREPASS works and keeps the password out of argv
  [7] an empty password file is refused rather than silently disabling auth

The bind assertions read the ACTUAL listening sockets via `lsof`, not the log
line — the server announcing "Binding to 127.0.0.1" while a listener sits on
0.0.0.0 is precisely the failure being guarded against. (During development it
did exactly that on the worker-side listeners.)

Usage: python3 tests/test_gh258_bind_and_secrets.py [./pion-server]
"""
import os, re, socket, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_port_free  # noqa: E402

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1995
WORKDIR = f"/tmp/pion_gh258_{PORT}"

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def spawn(extra=None, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.Popen(
        [BINARY, "-p", str(PORT), "-w", "1", "--no-auto-detect", "--no-auto-embed"]
        + (extra or []),
        cwd=WORKDIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=e)


def wait_up(timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", PORT), timeout=2); s.close()
            return True
        except OSError:
            time.sleep(0.25)
    return False


def listen_addrs(pid):
    """The addresses this pid is ACTUALLY listening on, per lsof."""
    try:
        # -a is REQUIRED: without it lsof ORs its selectors, so `-p PID -iTCP`
        # lists every OTHER process's sockets too. That made this test report
        # Pion as listening on `*` when the wildcard belonged to an unrelated
        # process, and made "leaves nothing listening" fail for a dead pid.
        out = subprocess.run(["lsof", "-a", "-nP", "-p", str(pid),
                              "-iTCP", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    addrs = set()
    for line in out.splitlines()[1:]:
        m = re.search(r"(\S+):(\d+) \(LISTEN\)", line)
        if m:
            addrs.add(m.group(1))
    return addrs


def stop(p):
    if p.poll() is None:
        p.terminate()
        try: p.wait(timeout=20)
        except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=10)
    # The next case reuses the port, and wait_up() only connects: a killed
    # server's lingering listener would pass it (#27). wait_ready_pid cannot
    # be used there, as most cases set a password and a bare PING gets NOAUTH.
    wait_port_free(PORT)


def cmd(port, *args):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            a = str(a).encode()
            parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
        s.sendall(b"".join(parts))
        time.sleep(0.3)
        return s.recv(400)
    finally:
        s.close()


def argv_of(pid):
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def bind_case(label, extra, env, want_loopback_only):
    p = spawn(extra, env)
    try:
        if not wait_up():
            check(f"{label}: server started", False, "did not come up")
            return None
        addrs = listen_addrs(p.pid)
        check(f"{label}: has listeners", bool(addrs), "lsof saw none")
        if want_loopback_only:
            check(f"{label}: every listener is loopback",
                  addrs and all(a == "127.0.0.1" for a in addrs),
                  f"listening on {sorted(addrs)}")
        else:
            check(f"{label}: listeners are on all interfaces",
                  any(a in ("*", "0.0.0.0") for a in addrs),
                  f"listening on {sorted(addrs)}")
        return p
    except Exception:
        stop(p); raise


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)

    print("[1] Default, no password -> loopback only")
    p = bind_case("no-password", None, None, True); stop(p)

    print("\n[2] Default, WITH a password -> all interfaces (protected-mode precedent)")
    p = bind_case("with-password", ["--requirepass", "pw1"], None, False); stop(p)

    print("\n[3] Explicit --bind wins even when a password is set")
    p = bind_case("explicit-bind", ["--requirepass", "pw1", "--bind", "127.0.0.1"],
                  None, True); stop(p)

    print("\n[4] An INVALID --bind refuses to start (never falls back to 0.0.0.0)")
    p = spawn(["--bind", "999.999.1.1"])
    try:
        rc = p.wait(timeout=40)
        out = p.stdout.read() if p.stdout else ""
        check("invalid --bind exits non-zero", rc != 0, f"exit={rc}")
        check("invalid --bind says why", "not a valid IPv4" in out, f"output: {out[-200:]!r}")
        check("invalid --bind leaves nothing listening", not listen_addrs(p.pid))
    except subprocess.TimeoutExpired:
        check("invalid --bind exits non-zero", False, "still running — it started anyway")
        stop(p)

    print("\n[5] --requirepass-file: works, strips the newline, stays out of argv")
    pwfile = os.path.join(WORKDIR, "pw.txt")
    with open(pwfile, "w") as f:
        f.write("filepw123\n")          # the trailing \n `echo` would leave
    os.chmod(pwfile, 0o600)
    p = spawn(["--requirepass-file", pwfile])
    try:
        if wait_up():
            check("file password: wrong password rejected",
                  b"WRONGPASS" in cmd(PORT, "AUTH", "nope"))
            check("file password: correct password accepted (newline stripped)",
                  cmd(PORT, "AUTH", "filepw123").startswith(b"+OK"))
            check("file password: not visible in argv",
                  "filepw123" not in argv_of(p.pid))
        else:
            check("file password: server started", False)
    finally:
        stop(p)

    print("\n[6] PION_REQUIREPASS: works and stays out of argv")
    p = spawn(None, {"PION_REQUIREPASS": "envpw456"})
    try:
        if wait_up():
            check("env password: wrong password rejected",
                  b"WRONGPASS" in cmd(PORT, "AUTH", "nope"))
            check("env password: correct password accepted",
                  cmd(PORT, "AUTH", "envpw456").startswith(b"+OK"))
            check("env password: not visible in argv",
                  "envpw456" not in argv_of(p.pid))
        else:
            check("env password: server started", False)
    finally:
        stop(p)

    print("\n[7] An EMPTY password file is refused, not silently treated as no auth")
    empty = os.path.join(WORKDIR, "empty.txt")
    open(empty, "w").close()
    p = spawn(["--requirepass-file", empty])
    try:
        rc = p.wait(timeout=40)
        out = p.stdout.read() if p.stdout else ""
        check("empty password file exits non-zero", rc != 0, f"exit={rc}")
        check("empty password file says why", "empty" in out.lower(),
              f"output: {out[-200:]!r}")
    except subprocess.TimeoutExpired:
        check("empty password file exits non-zero", False,
              "started anyway — auth would be silently OFF")
        stop(p)

    shutil.rmtree(WORKDIR, ignore_errors=True)
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
