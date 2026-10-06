#!/usr/bin/env python3
"""A server restarted on the same port comes up, and a stopped one frees it (#22).

WHY
On Linux the kernel tears an io_uring instance down after the process has
exited, and a pending ACCEPT holds a reference to the listening socket until
it has. So a killed server's port kept listening, and accepting, for a moment
after the process was gone. A server started right away failed with "cannot
bind port"; a client that connected in that window reached the dead server's
backlog and got refused or reset.

Fixed on two sides: a graceful stop cancels its ACCEPTs before the workers
return, so the port is free when the process is; and a starting server waits
up to 3 s for a port that is still held. A SIGKILLed server cannot cancel
anything, which is what the retry is for.

CHECKS
  1. SIGKILL, then start a new server on the same port at once: it serves, and
     the server that answers is the NEW one (INFO process_id).
  2. SIGTERM: once the process has exited, the port binds immediately.

    python3 tests/test_restart_rebind.py [--binary pion-server]
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 6433
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def start(binary, work, tag):
    log = open(os.path.join(work, f"{tag}.log"), "w")
    return subprocess.Popen([binary, "-p", str(PORT), "-w", "1", "--no-auto-detect",
                             "--no-auto-embed", "--no-crash-log", "--no-wal"],
                            cwd=work, stdout=log, stderr=subprocess.STDOUT)


def info_pid(port):
    with Conn(port, timeout=5) as c:
        info = c.cmd("INFO", "server")
    for line in info.decode().splitlines():
        if line.startswith("process_id:"):
            return int(line.split(":", 1)[1])
    return None


def busy(c, rounds=50):
    """Traffic on a connection, so the ring has completions in flight."""
    for i in range(rounds):
        c.cmd("SET", f"k{i}", "v" * 64)


def main() -> int:
    binary = os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server"))
    work = tempfile.mkdtemp(prefix="rebind_")
    procs = []
    try:
        print("[1] SIGKILL, then restart on the same port at once")
        a = start(binary, work, "a")
        procs.append(a)
        wait_ready(PORT, 60, proc=a)
        held = [Conn(PORT, timeout=5) for _ in range(4)]   # connections still open at the kill
        for c in held:
            busy(c)
        a.send_signal(signal.SIGKILL)
        a.wait()
        b = start(binary, work, "b")
        procs.append(b)
        try:
            wait_ready(PORT, 60, proc=b)
            up = True
        except RuntimeError as e:
            up = False
            print(f"  {e}")
        blog = open(os.path.join(work, "b.log")).read()
        check("the new server started (no 'cannot bind port')", up and "cannot bind" not in blog,
              blog[-300:])
        if up:
            pid = info_pid(PORT)
            check("the server answering is the new process", pid == b.pid,
                  f"INFO process_id={pid}, new pid={b.pid}, killed pid={a.pid}")
        for c in held:
            c.close()

        print("[2] SIGTERM frees the port with the process")
        if up:
            with Conn(PORT, timeout=5) as c:
                busy(c)
                b.send_signal(signal.SIGTERM)
                b.wait(timeout=30)
            t0 = time.monotonic()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bound, err = False, None
            while time.monotonic() - t0 < 2.0:
                try:
                    s.bind(("127.0.0.1", PORT))
                    bound = True
                    break
                except OSError as e:
                    err = e
                    time.sleep(0.01)
            waited = time.monotonic() - t0
            s.close()
            check("exit status 0", b.returncode == 0, f"exit {b.returncode}")
            check("the port binds as soon as the stopped server has exited",
                  bound and waited < 0.2, f"bound={bound} after {waited:.2f} s ({err})")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
