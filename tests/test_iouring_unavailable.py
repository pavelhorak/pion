#!/usr/bin/env python3
"""Without io_uring, a Linux server must still serve — on epoll (#21).

WHY
io_uring is the default event loop on Linux, and it is missing in common
places: Docker's default seccomp profile blocks its syscalls, old kernels lack
them, sandboxes forbid them. When `io_uring_setup` failed, the worker fell back
to the kqueue loop, which returns at once on Linux. The worker thread ended
while the listening socket stayed open: clients connected and never got an
answer, and nothing said why.

HOW
An LD_PRELOAD shim makes `syscall(SYS_io_uring_setup, ...)` fail with ENOSYS,
exactly what seccomp does, without needing a container. The server must log
the epoll fallback and answer commands on every worker.

    python3 tests/test_iouring_unavailable.py [--binary pion-server]
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 6431

SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <sys/syscall.h>
#ifndef SYS_io_uring_setup
#define SYS_io_uring_setup 425
#endif
/* Linux passes every syscall argument in a register, variadic or not, so
   forwarding six longs is exact for the calls that reach the real one. */
long syscall(long n, ...) {
    static long (*real)(long, ...) = 0;
    va_list ap;
    va_start(ap, n);
    long a1 = va_arg(ap, long), a2 = va_arg(ap, long), a3 = va_arg(ap, long);
    long a4 = va_arg(ap, long), a5 = va_arg(ap, long), a6 = va_arg(ap, long);
    va_end(ap);
    if (n == SYS_io_uring_setup) { errno = ENOSYS; return -1; }
    if (!real) real = (long (*)(long, ...))dlsym(RTLD_NEXT, "syscall");
    return real(n, a1, a2, a3, a4, a5, a6);
}
"""

failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def main() -> int:
    if platform.system() != "Linux":
        print("SKIP: io_uring is Linux-only")
        return 0
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("SKIP: no C compiler to build the shim")
        return 0
    binary = os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server"))
    work = tempfile.mkdtemp(prefix="nouring_")
    src, lib = os.path.join(work, "nouring.c"), os.path.join(work, "libnouring.so")
    with open(src, "w") as f:
        f.write(SHIM)
    subprocess.run([cc, "-O2", "-shared", "-fPIC", src, "-o", lib, "-ldl"], check=True)
    log_path = os.path.join(work, "server.log")
    env = dict(os.environ, LD_PRELOAD=lib)
    proc = subprocess.Popen([binary, "-p", str(PORT), "-w", "2", "--independent-workers",
                             "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                            cwd=work, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    try:
        try:
            wait_ready(PORT, 60, proc=proc)
            up = True
        except RuntimeError as e:
            up = False
            print(f"  {e}")
        log = open(log_path).read()
        print("[1] the shim took: io_uring_setup failed")
        if not check("log says io_uring is unavailable", "io_uring unavailable" in log,
                     "the shim did not load, or the message changed — every later check "
                     "would be vacuous"):
            return 1
        print("[2] the workers fell back to epoll and serve")
        check("both workers run the epoll loop", log.count("EPOLL Engine Active") == 2,
              f"{log.count('EPOLL Engine Active')} epoll loops in the log")
        check("no worker died", "DIED" not in log, log[-400:])
        check("server answers PING", up)
        if up:
            # Several connections, so with -w 2 both workers are exercised.
            ok = True
            for i in range(8):
                c = Conn(PORT, timeout=10)
                ok &= c.cmd("SET", f"k{i}", f"v{i}") == "OK"
                ok &= c.cmd("GET", f"k{i}") == f"v{i}".encode()
                c.close()
            check("SET/GET on 8 connections", ok)
        time.sleep(0.5)
        check("server still running", proc.poll() is None, f"exit {proc.returncode}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        try:
            wait_port_free(PORT)
        except RuntimeError:
            pass
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
