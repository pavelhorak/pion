#!/usr/bin/env python3
"""gh #138 — crash diagnostics, supervised serving, client-visible liveness.

Covers the three asks on the issue:

  1. Crash/exit diagnostic — a `pion-server` that dies leaves a breadcrumb.
     Catchable deaths (SIGTERM/SIGSEGV/...) append a line to the crash log with
     signal, uptime and RSS. Uncatchable deaths (jetsam / OOM-killer SIGKILL)
     leave the 1 Hz status file frozen at `state=running` with the last RSS
     sample — which is exactly the evidence needed to confirm or deny an OS
     memory kill.

  2. Supervised serving — scripts/pion-supervise.sh restarts a dead server and
     writes a postmortem (exit status, last heartbeat, OS memory-kill lookup).

  3. Client-visible liveness — FT.SEARCH against an index that does not exist
     returns an error instead of an empty array, so "the store lost its index"
     is no longer indistinguishable from "the query matched nothing".

Usage:
    python3 tests/test_gh138_robustness.py [--binary ./pion-server] [--port 7861]
"""

import argparse
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERVISOR = os.path.join(REPO, "scripts", "pion-supervise.sh")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


# ── tiny RESP client ─────────────────────────────────────────────────────────
class Client:
    def __init__(self, port, timeout=10):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # makefile('rb'), never manual slicing — see python_resp_array_slicing_trap
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.sock.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self._read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            if n < 0:
                return None
            data = self.f.read(n + 2)
            return data[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def wait_ready(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = Client(port, timeout=2)
            r = c.cmd("PING")
            c.close()
            if r == b"PONG":
                return True
        except (OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def read_status(path):
    """Parse the fixed-size key=value status record."""
    try:
        with open(path, "r") as fh:
            raw = fh.read()
    except OSError:
        return {}
    out = {}
    for line in raw.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ── Test 1: heartbeat + catchable-signal breadcrumb ──────────────────────────
def test_breadcrumbs(binary, port, workdir):
    print("\n[1] crash/exit diagnostic")
    status_path = os.path.join(workdir, f"pion-{port}.status")
    crash_path = os.path.join(workdir, f"pion-{port}.crash.log")

    proc = subprocess.Popen(
        [binary, "--profile", "kv", "-w", "1", "-p", str(port),
         "--no-auto-detect", "--no-auto-embed"],
        cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        if not check("server becomes ready", wait_ready(port)):
            return

        check("crash log created at startup", os.path.exists(crash_path))
        with open(crash_path) as fh:
            start_line = fh.read()
        check("crash log has a PION START line", "PION START:" in start_line,
              start_line.strip().splitlines()[:1])

        # The heartbeat runs at 1 Hz from the event loop's 64-tick housekeeping.
        time.sleep(3)
        st = read_status(status_path)
        check("status file exists and parses", bool(st), f"{len(st)} fields")
        check("status state=running", st.get("state") == "running", st.get("state"))
        check("status pid matches the process", st.get("pid") == str(proc.pid),
              f"{st.get('pid')} vs {proc.pid}")
        check("status reports a plausible RSS", int(st.get("rss_bytes", 0)) > 1_000_000,
              f"rss_bytes={st.get('rss_bytes')}")
        check("status reports total RAM", int(st.get("total_ram_bytes", 0)) > 0,
              f"total_ram_bytes={st.get('total_ram_bytes')}")

        ticks_1 = int(st.get("ticks", 0))
        time.sleep(2)
        ticks_2 = int(read_status(status_path).get("ticks", 0))
        check("event-loop tick counter advances (liveness)", ticks_2 > ticks_1,
              f"{ticks_1} -> {ticks_2}")

        # Catchable death: the handler must leave a line AND preserve the
        # signal exit status (128 + signum). SIGQUIT, not SIGTERM: since
        # gh #259 SIGTERM is a graceful drain (exit 0, "clean shutdown" —
        # tests/test_gh259_graceful_shutdown.py), and this test kept expecting
        # a signal death for months without being run. SIGQUIT still dies.
        proc.send_signal(signal.SIGQUIT)
        proc.wait(timeout=15)
        with open(crash_path) as fh:
            log = fh.read()
        check("SIGQUIT leaves an exit breadcrumb", "PION EXIT: signal SIGQUIT" in log,
              log.strip().splitlines()[-1] if log.strip() else "<empty>")
        check("breadcrumb carries RSS at exit", "rss_mb=" in log)
        check("signal exit status preserved", proc.returncode in (-signal.SIGQUIT, 128 + signal.SIGQUIT),
              f"returncode={proc.returncode}")
        st = read_status(status_path)
        check("status marked state=signalled", st.get("state") == "signalled", st.get("state"))
        check("status names the signal", st.get("signal_name") == "SIGQUIT",
              st.get("signal_name"))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ── Test 2: uncatchable death leaves usable evidence ─────────────────────────
def test_sigkill_evidence(binary, port, workdir):
    print("\n[2] uncatchable death (SIGKILL — the jetsam/OOM shape)")
    status_path = os.path.join(workdir, f"pion-{port}.status")
    crash_path = os.path.join(workdir, f"pion-{port}.crash.log")
    for p in (status_path, crash_path):
        if os.path.exists(p):
            os.remove(p)

    proc = subprocess.Popen(
        [binary, "--profile", "kv", "-w", "1", "-p", str(port),
         "--no-auto-detect", "--no-auto-embed"],
        cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        if not check("server becomes ready", wait_ready(port)):
            return
        time.sleep(3)
        before = read_status(status_path)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=15)
        time.sleep(1)

        after = read_status(status_path)
        with open(crash_path) as fh:
            log = fh.read()
        # This is the whole point: no in-process trace is possible, so the
        # frozen heartbeat has to carry the evidence.
        check("no exit line in the crash log (death was uncatchable)",
              "PION EXIT:" not in log)
        check("status frozen at state=running (proves external kill)",
              after.get("state") == "running", after.get("state"))
        check("last RSS sample survived the kill",
              int(after.get("rss_bytes", 0)) > 1_000_000,
              f"rss_bytes={after.get('rss_bytes')} rss_pct={after.get('rss_pct')}")
        check("heartbeat timestamp present for age computation",
              int(after.get("heartbeat_unix_s", 0)) > 0)
        check("uptime recorded", int(after.get("uptime_s", -1)) >= 0,
              f"uptime_s={after.get('uptime_s')}")
        check("heartbeat kept advancing until death",
              int(after.get("ticks", 0)) >= int(before.get("ticks", 0)))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


# ── Test 3: FT.SEARCH on a missing index is an error, not [] ─────────────────
def test_ft_search_liveness(binary, port, workdir):
    print("\n[3] client-visible liveness (FT.SEARCH on a missing index)")
    proc = subprocess.Popen(
        [binary, "--profile", "vector", "-w", "1", "-p", str(port),
         "--no-auto-detect", "--no-auto-embed"],
        cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        if not check("server becomes ready", wait_ready(port)):
            return
        c = Client(port)

        qvec = struct.pack("<4f", 0.5, 0.5, 0.5, 0.5)
        r = c.cmd("FT.SEARCH", "ghost_idx", "*=>[KNN 3 @vector $vec]",
                  "PARAMS", "2", "vec", qvec)
        check("KNN on a never-created index errors",
              isinstance(r, Exception) and "no such index" in str(r), repr(r)[:110])

        r = c.cmd("FT.SEARCH", "ghost_idx", "BM25", "hello world", "K", "3")
        check("BM25 on a never-created index errors",
              isinstance(r, Exception) and "no such index" in str(r), repr(r)[:110])

        # An error must not desync the RESP frame for pipelined clients.
        check("connection still usable after the error", c.cmd("PING") == b"PONG")

        # Positive path: a real index must still answer normally, and an empty
        # result set on a REAL index must stay an empty array (not an error).
        check("FT.CREATE", c.cmd(
            "FT.CREATE", "live_idx", "ON", "HASH", "PREFIX", "1", "doc:",
            "SCHEMA", "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32",
            "DIM", "4", "DISTANCE_METRIC", "L2") == b"OK")
        r = c.cmd("FT.SEARCH", "live_idx", "*=>[KNN 3 @vector $vec]",
                  "PARAMS", "2", "vec", qvec)
        check("created-but-empty index returns an empty array, not an error",
              not isinstance(r, Exception), repr(r)[:110])

        for i in range(16):
            v = struct.pack("<4f", i / 16.0, 0.5, 0.5, 0.5)
            c.cmd("HSET", f"doc:{i}", "vector", v)
        c.cmd("FT.OPTIMIZE", "live_idx")
        r = c.cmd("FT.SEARCH", "live_idx", "*=>[KNN 3 @vector $vec]",
                  "PARAMS", "2", "vec", qvec)
        check("populated index returns hits",
              isinstance(r, list) and len(r) > 1, f"{len(r) if isinstance(r, list) else r} elems")
        c.close()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


# ── Test 4: supervisor restarts a killed server and files a postmortem ───────
def test_supervisor(binary, port, workdir):
    print("\n[4] supervised serving (scripts/pion-supervise.sh)")
    sup_log = os.path.join(workdir, f"pion-supervisor-{port}.log")
    status_path = os.path.join(workdir, f"pion-{port}.status")
    for p in (sup_log, status_path):
        if os.path.exists(p):
            os.remove(p)

    sup = subprocess.Popen(
        [SUPERVISOR, "--binary", os.path.abspath(binary), "--interval", "3",
         "--max-restarts", "2", "--",
         "--profile", "kv", "-w", "1", "-p", str(port),
         "--no-auto-detect", "--no-auto-embed"],
        cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        if not check("supervisor brings the server up", wait_ready(port)):
            return
        time.sleep(3)
        first_pid = int(read_status(status_path).get("pid", 0))
        check("supervisor recorded readiness",
              "is serving on port" in open(sup_log).read())

        # Simulate the reported failure: the process vanishes with no warning.
        os.kill(first_pid, signal.SIGKILL)

        # Restart = new pid answering PING. Backoff starts at 1s; the server
        # needs a few seconds to init its slabs.
        deadline = time.time() + 120
        second_pid = first_pid
        while time.time() < deadline:
            time.sleep(2)
            st = read_status(status_path)
            pid = int(st.get("pid", 0))
            if pid and pid != first_pid and st.get("state") == "running":
                second_pid = pid
                break
        check("server was restarted after the kill", second_pid != first_pid,
              f"{first_pid} -> {second_pid}")
        check("restarted server answers PING", wait_ready(port, timeout=60))

        log = open(sup_log).read()
        check("postmortem written", "POSTMORTEM" in log)
        check("postmortem reports the signal", "signal=9" in log)
        check("postmortem reports RSS at death", "rss_mb=" in log)
        check("postmortem checked the OS memory-kill log",
              "jetsam" in log or "OOM" in log,
              [l for l in log.splitlines() if "jetsam" in l or "OOM" in l][:1])
    finally:
        # Kill the supervisor's process group so it cannot restart the server
        # out from under the teardown.
        try:
            os.killpg(os.getpgid(sup.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            sup.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(sup.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        st = read_status(status_path)
        pid = int(st.get("pid", 0) or 0)
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args()

    # Servers run with cwd=<tmp workdir> so their breadcrumb files land there —
    # so the binary path has to be absolute.
    args.binary = os.path.abspath(args.binary)
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary} (build with: pixi run build)")
        return 2

    workdir = tempfile.mkdtemp(prefix="pion-gh138-")
    print(f"gh #138 robustness gate — binary={args.binary} port={args.port}")
    print(f"workdir={workdir}")
    try:
        test_breadcrumbs(args.binary, args.port, workdir)
        test_sigkill_evidence(args.binary, args.port, workdir)
        test_ft_search_liveness(args.binary, args.port + 1, workdir)
        test_supervisor(args.binary, args.port + 2, workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
