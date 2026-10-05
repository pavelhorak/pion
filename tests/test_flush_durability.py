#!/usr/bin/env python3
"""FLUSHALL and FLUSHDB survive a restart, and a removed key's TTL goes with it.

CHECKS
  1. Keys written, then FLUSHALL, then new keys, then SIGKILL and restart: only
     the new keys come back (WAL replay). Same for FLUSHDB, and after a SAVE.
  2. A key created after a FLUSHALL under a name that had a TTL has no TTL, and
     is still there after that old deadline.
  3. A key re-created after its list was emptied by a pop has no TTL and
     survives the old deadline (the active expiry sweep included).
  4. FLUSHALL / FLUSHDB take only ASYNC or SYNC.

    python3 tests/test_flush_durability.py [--port 6447]
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready_pid, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.path.abspath(os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


class Server:
    def __init__(self, work, port):
        self.port, self.dir, self.proc = port, work, None

    def start(self):
        self.proc = subprocess.Popen([BIN, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed"],
                                     cwd=self.dir, stdout=open(os.path.join(self.dir, "log"), "a"),
                                     stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)
        return Conn(self.port, timeout=30)

    def kill(self):
        if self.proc:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait()
            self.proc = None
            wait_port_free(self.port)


def restart_flush(srv, flush, save_first):
    c = srv.start()
    c.cmd("FLUSHALL")
    for i in range(20):
        c.cmd("SET", f"old:{i}", "v")
    c.cmd("RPUSH", "old:list", "a", "b")
    c.cmd("SET", "old:ttl", "v", "EX", "1000")
    if save_first:
        check(f"{flush}: SAVE", c.cmd("SAVE") == "OK")
    check(f"{flush} answers OK", c.cmd(flush) == "OK")
    c.cmd("SET", "new:1", "v")
    c.cmd("SET", "old:ttl", "fresh")
    c.close()
    srv.kill()
    c = srv.start()
    keys = sorted(k.decode() for k in c.cmd("KEYS", "*"))
    check(f"{flush}{' after SAVE' if save_first else ''}: only the new keys come back",
          keys == ["new:1", "old:ttl"], repr(keys[:8]))
    check(f"{flush}: the re-created key has no TTL", c.cmd("TTL", "old:ttl") == -1,
          repr(c.cmd("TTL", "old:ttl")))
    c.close()
    srv.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6447)
    a = ap.parse_args()
    work = tempfile.mkdtemp(prefix="flush_dur_")
    srv = Server(work, a.port)
    try:
        print("[1] a flush is durable")
        restart_flush(srv, "FLUSHALL", False)
        restart_flush(srv, "FLUSHDB", False)
        restart_flush(srv, "FLUSHALL", True)

        print("[2] a flushed key's TTL is gone")
        c = srv.start()
        c.cmd("SET", "t", "v", "PX", "1500")
        c.cmd("FLUSHALL")
        c.cmd("RPUSH", "t", "a")
        check("TTL after FLUSHALL + re-create", c.cmd("TTL", "t") == -1, repr(c.cmd("TTL", "t")))

        print("[3] an emptied key's TTL is gone")
        c.cmd("RPUSH", "q", "a")
        c.cmd("PEXPIRE", "q", "1500")
        c.cmd("LPOP", "q")
        c.cmd("RPUSH", "q", "b")
        check("TTL after pop-to-empty + re-create", c.cmd("TTL", "q") == -1, repr(c.cmd("TTL", "q")))
        time.sleep(2.5)                          # past both old deadlines; the sweep has run
        check("the re-created keys outlive the old deadlines",
              c.cmd("EXISTS", "t") == 1 and c.cmd("EXISTS", "q") == 1,
              f"t={c.cmd('EXISTS', 't')} q={c.cmd('EXISTS', 'q')}")

        print("[4] FLUSHALL / FLUSHDB arguments")
        c.cmd("SET", "keep", "v")
        for cmd in ("FLUSHALL", "FLUSHDB"):
            check(f"{cmd} NOPE is a syntax error", c.cmd(cmd, "NOPE") == "ERR syntax error",
                  repr(c.cmd(cmd, "NOPE")))
            check(f"{cmd} SYNC ASYNC is a syntax error", c.cmd(cmd, "SYNC", "ASYNC") == "ERR syntax error")
        check("a refused flush flushed nothing", c.cmd("EXISTS", "keep") == 1)
        check("FLUSHDB ASYNC works", c.cmd("FLUSHDB", "ASYNC") == "OK" and c.cmd("DBSIZE") == 0)
        c.close()
    finally:
        srv.kill()
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
