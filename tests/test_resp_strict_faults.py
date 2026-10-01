#!/usr/bin/env python3
"""The test suite's reply reader must SEE a damaged reply stream (step 6 of
the 2026-09-29 test audit).

WHY
Every gate test now reads replies through tests/resp_strict.py. It replaced a
helper that did one `recv()` per command and swallowed the timeout, and the
claim was that the old helper passed three kinds of broken server. A claim
about a reader is only worth something if the reader is shown a fault and
reacts — so this puts tests/fault_proxy.py between a real server and both
readers, for each fault:

  dup     a reply sent twice (a reply-count desync). The legacy helper reads
          the copy as the NEXT command's answer and a lenient check passes;
          resp_strict must fail the sync check.
  split   replies cut into pieces across reads. The legacy helper returns the
          first piece; resp_strict must return the whole reply, unchanged.
  delay   the first reply held past the reader's timeout (a server replaying
          its WAL). The legacy helper returns "" and the late reply then
          answers the next command; resp_strict must raise, never return "".

The legacy half is the canary: if the old helper is NOT fooled, the proxy did
not inject the fault and the resp_strict half proves nothing, so that fails
the test too.

    python3 tests/test_resp_strict_faults.py [./pion-server]
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fault_proxy import FaultProxy  # noqa: E402
from resp_strict import Conn, RespProtocolError, wait_ready, wait_port_free  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BINARY = os.path.abspath(ARGS[0] if ARGS else os.environ.get("PION_BIN", "./pion-server"))
PORT, PROXY = 6650, 6651
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def legacy_cmd(sock, *args, timeout=1.0):
    """The helper resp_strict replaced: one recv per command, timeout swallowed."""
    sock.sendall(b"".join([b"*%d\r\n" % len(args)] +
                          [b"$%d\r\n%s\r\n" % (len(a), a) for a in (x.encode() for x in args)]))
    sock.settimeout(timeout)
    try:
        return sock.recv(65536).decode(errors="replace")
    except socket.timeout:
        return ""


def main():
    d = tempfile.mkdtemp(prefix="pion_faults_")
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-crash-log",
                          "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "a"), stderr=subprocess.STDOUT)
    try:
        wait_ready(PORT, 30, proc=p)
        big = "x" * 70000
        c = Conn(PORT)
        c.cmd("SET", "big", big)
        c.cmd("RPUSH", "l", *[str(i) for i in range(500)])
        c.close()

        print("=== dup: the 3rd reply is sent twice ===")
        with FaultProxy(PROXY, PORT, "dup:3") as fp:
            s = socket.create_connection(("127.0.0.1", PROXY))
            r = [legacy_cmd(s, "SET", "a", "1"), legacy_cmd(s, "SET", "b", "2"),
                 legacy_cmd(s, "SET", "c", "3"), legacy_cmd(s, "INCR", "n")]
            s.close()
            # A lenient check of the kind the old tests made: every write "OK",
            # and INCR "answered". The INCR slot actually holds SET c's copy.
            fooled = all("OK" in x for x in r[:3]) and r[3] != "" and ":1" not in r[3]
            check("canary: the legacy helper is fooled (INCR read SET c's copy)", fooled, repr(r))
            c = Conn(PROXY, timeout=3)
            got = [c.cmd("SET", "a", "1"), c.cmd("SET", "b", "2"), c.cmd("SET", "c", "3")]
            try:
                c.assert_in_sync()
                caught = False
            except RespProtocolError:
                caught = True
            c.close()
            check("resp_strict: assert_in_sync catches the surplus reply", caught, repr(got))
            c = Conn(PROXY, timeout=3)
            c.cmd("PING"); c.cmd("PING")
            try:
                c.cmd_synced("SET", "d", "4")
                caught = False
            except RespProtocolError:
                caught = True
            c.close()
            check("resp_strict: cmd_synced catches it too", caught)
            check("the proxy injected the fault", fp.injected >= 3, str(fp.injected))

        print("=== split: every reply cut into pieces across reads ===")
        with FaultProxy(PROXY, PORT, "split") as fp:
            s = socket.create_connection(("127.0.0.1", PROXY))
            g = legacy_cmd(s, "GET", "big")
            s.close()
            check("canary: the legacy helper returns a truncated reply",
                  0 < len(g) < len(big), f"{len(g)} bytes")
            c = Conn(PROXY, timeout=10)
            got_big = c.cmd("GET", "big")
            got_l = c.cmd("LRANGE", "l", "0", "-1")
            got_p = c.pipeline([("GET", "big"), ("PING",), ("LLEN", "l")])
            c.assert_in_sync()
            c.close()
            check("resp_strict: GET of 70 KB comes back whole", got_big == big.encode())
            check("resp_strict: LRANGE of 500 comes back whole",
                  got_l == [str(i).encode() for i in range(500)])
            check("resp_strict: a pipeline keeps its pairing", got_p == [big.encode(), "PONG", 500])
            check("the proxy injected the fault", fp.injected >= 5, str(fp.injected))

        print("=== delay: the first reply arrives after the reader's timeout ===")
        with FaultProxy(PROXY, PORT, "delay:1.5") as fp:
            s = socket.create_connection(("127.0.0.1", PROXY))
            r1 = legacy_cmd(s, "GET", "k1", timeout=0.5)
            r2 = legacy_cmd(s, "INCR", "k2", timeout=3)
            s.close()
            check("canary: the legacy helper returns '' and pairs the late reply with INCR",
                  r1 == "" and r2.startswith("$-1"), f"{r1!r} / {r2!r}")
            c = Conn(PROXY, timeout=0.5)
            try:
                v = c.cmd("GET", "k1")
                raised = False
            except TimeoutError:
                raised, v = True, None
            c.close()
            check("resp_strict: raises instead of returning an empty answer", raised, repr(v))
            check("the proxy injected the fault", fp.injected >= 2, str(fp.injected))
    finally:
        p.kill()
        p.wait()
        wait_port_free(PORT)
        shutil.rmtree(d, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
