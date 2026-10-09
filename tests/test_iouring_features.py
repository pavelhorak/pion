#!/usr/bin/env python3
"""The optional io_uring features serve exactly what the default loop serves
(gh #205, gh #206).

WHY
--iouring-defer (SINGLE_ISSUER + DEFER_TASKRUN), --iouring-regfiles
(registered files + registered ring fd) and --iouring-pbuf (a provided-buffer
ring, with received bytes parsed in the provided buffer itself) each change
how the io_uring loop talks to the kernel, and each has a known way to break:

  * DEFER_TASKRUN posts completions only inside io_uring_enter(GETEVENTS). A
    loop that waits for them anywhere else, or stops entering while idle,
    never sees a timeout fire: a BLPOP with a timeout, an XREAD BLOCK and the
    graceful-shutdown drain all hang.
  * A registered-file slot holds its own reference to the socket. A slot not
    emptied at close keeps the old socket alive, and a request that names it
    after the fd number went to a new connection reaches the old socket.
  * Parsing in the provided buffer is only safe while nothing outlives the
    drain: a frame split across buffers, a command that parks with commands
    pipelined behind it, and MULTI must all keep their bytes.

HOW
One server per configuration (off, each flag alone, all three by environment
variable, all three on 4 workers). The log must say which features the kernel
granted, and each configuration runs the same workloads, every reply parsed:
many concurrent pipelined connections with frames straddling the 16 KB
buffers, a 1 MB value, a frame split across writes, a parked BLPOP and XREAD
with commands behind them, MULTI/EXEC in one write, connection churn with
resets (fd reuse), and a graceful stop.

    python3 tests/test_iouring_features.py [--binary pion-server]
"""
from __future__ import annotations

import os
import platform
import random
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode, wait_ready, wait_port_free  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 6452
ENV_SWITCHES = ("PION_IOURING_DEFER", "PION_IOURING_REGFILES", "PION_IOURING_PBUF")
ALL_FLAGS = ["--iouring-defer", "--iouring-regfiles", "--iouring-pbuf"]
ALL_ENV = {k: "1" for k in ENV_SWITCHES}

# (name, extra flags, extra environment, features asked for)
CONFIGS = [
    ("off", [], {}, set()),
    ("defer", ["--iouring-defer"], {}, {"defer"}),
    ("regfiles", ["--iouring-regfiles"], {}, {"regfiles", "ring_fd"}),
    ("pbuf", ["--iouring-pbuf"], {}, {"pbuf_ring", "zero_copy_recv"}),
    ("all-by-env", [], ALL_ENV, {"defer", "regfiles", "ring_fd", "pbuf_ring", "zero_copy_recv"}),
    ("all-w4", ALL_FLAGS + ["-w", "4", "--independent-workers"], {},
     {"defer", "regfiles", "ring_fd", "pbuf_ring", "zero_copy_recv"}),
]
# The kernel each feature first appeared in. A kernel at least this new that
# does not grant the feature is a failure; an older one is reported only.
MIN_KERNEL = {"defer": (6, 1), "regfiles": (5, 5), "ring_fd": (5, 18),
              "pbuf_ring": (5, 19), "zero_copy_recv": (6, 0)}

failures: list = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)
    return cond


def kernel_version():
    m = re.match(r"(\d+)\.(\d+)", platform.release())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# ── workloads ────────────────────────────────────────────────────────────────

def w_concurrent_pipelines(tag):
    """32 connections opened at once, each sending pipelines whose frames
    straddle the 16 KB provided buffers; every reply checked."""
    errors = []
    barrier = threading.Barrier(32)

    def client(cid):
        rnd = random.Random(cid)
        try:
            barrier.wait(timeout=30)
            with Conn(PORT, timeout=30) as c:
                for rnd_i in range(6):
                    cmds, want = [], []
                    for j in range(120):
                        key = f"{tag}:c{cid}:k{j}"
                        val = bytes([97 + (cid + j) % 26]) * rnd.choice((1, 7, 23, 24, 900, 3000, 9000))
                        cmds += [("SET", key, val), ("GET", key), ("INCR", f"{tag}:c{cid}:n")]
                        want += ["OK", val, rnd_i * 120 + j + 1]
                    got = c.pipeline(cmds)
                    if got != want:
                        bad = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
                        errors.append(f"conn {cid} round {rnd_i} reply {bad}: "
                                      f"{str(got[bad])[:60]} != {str(want[bad])[:60]}")
                        return
                c.assert_in_sync()
        except Exception as e:  # noqa: BLE001 — any failure is the finding
            errors.append(f"conn {cid}: {type(e).__name__}: {e}")

    ts = [threading.Thread(target=client, args=(i,)) for i in range(32)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    check(f"[{tag}] 32 concurrent pipelined connections, every reply right",
          not errors, "; ".join(errors[:3]))


def w_large_and_split(tag):
    with Conn(PORT, timeout=30) as c:
        big = os.urandom(1 << 20)
        check(f"[{tag}] 1 MB SET", c.cmd("SET", f"{tag}:big", big) == "OK")
        check(f"[{tag}] 1 MB GET round-trips", c.cmd("GET", f"{tag}:big") == big)
        # One frame in three writes, cut inside the bulk header and the value.
        frame = encode(("SET", f"{tag}:split", "x" * 5000)) + encode(("GET", f"{tag}:split"))
        for cut in (frame[:9], frame[9:2000], frame[2000:]):
            c.sock.sendall(cut)
            time.sleep(0.05)
        check(f"[{tag}] frame split across writes: SET", c.read() == "OK")
        check(f"[{tag}] frame split across writes: GET", c.read() == b"x" * 5000)
        c.assert_in_sync()


def w_parked_then_pipelined(tag):
    """A command that parks, with commands behind it in the same write. The
    timeouts fire from the loop's tick, with no other traffic: under
    DEFER_TASKRUN that needs enter() while idle."""
    with Conn(PORT, timeout=10) as c:
        t0 = time.monotonic()
        got = c.pipeline([("BLPOP", f"{tag}:nolist", "0.3"), ("PING",),
                          ("SET", f"{tag}:after", "v"), ("GET", f"{tag}:after")])
        dt = time.monotonic() - t0
        check(f"[{tag}] BLPOP timeout then the pipelined commands, in order",
              got == [None, "PONG", "OK", b"v"], repr(got))
        check(f"[{tag}] BLPOP 0.3 answered in time while idle", 0.25 <= dt < 3.0, f"{dt:.2f} s")
        c.cmd("XADD", f"{tag}:s", "*", "f", "v")
        t0 = time.monotonic()
        got = c.pipeline([("XREAD", "BLOCK", "300", "STREAMS", f"{tag}:s", "$"), ("PING",)])
        dt = time.monotonic() - t0
        check(f"[{tag}] XREAD BLOCK timeout then PING", got == [None, "PONG"], repr(got))
        check(f"[{tag}] XREAD BLOCK 300 answered in time while idle", 0.25 <= dt < 3.0, f"{dt:.2f} s")
        c.assert_in_sync()
    # A parked BLPOP served by a push from another connection. Both go to
    # worker 0's own port (port + 2), so they share a keyspace at -w 4 too.
    with Conn(PORT + 2, timeout=10) as c:
        c.sock.sendall(encode(("BLPOP", f"{tag}:wake", "5")) + encode(("PING",)))
        time.sleep(0.2)
        with Conn(PORT + 2) as d:
            d.cmd("RPUSH", f"{tag}:wake", "x")
        check(f"[{tag}] BLPOP woken by a push", c.read() == [f"{tag}:wake".encode(), b"x"])
        check(f"[{tag}] the PING behind it", c.read() == "PONG")
        c.assert_in_sync()


def w_multi_exec(tag):
    with Conn(PORT) as c:
        got = c.pipeline([("MULTI",), ("SET", f"{tag}:x", "1"), ("INCR", f"{tag}:x"),
                          ("GET", f"{tag}:x"), ("EXEC",), ("GET", f"{tag}:x")])
        check(f"[{tag}] MULTI/EXEC in one write",
              got == ["OK", "QUEUED", "QUEUED", "QUEUED", ["OK", 2, b"2"], b"2"], repr(got))


def w_churn(tag):
    """Connections opened and closed (half of them with a reset, some with a
    reply unread) while one long-lived connection keeps talking. Every new
    connection on a reused fd number must get its own replies, and the
    long-lived one only its own."""
    errors = []
    stop = threading.Event()

    def steady():
        try:
            with Conn(PORT, timeout=10) as c:
                i = 0
                while not stop.is_set():
                    if c.cmd("ECHO", f"steady-{i}") != f"steady-{i}".encode():
                        errors.append(f"steady connection got a wrong reply at {i}")
                        return
                    i += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"steady: {type(e).__name__}: {e}")

    th = threading.Thread(target=steady)
    th.start()
    try:
        for i in range(400):
            c = Conn(PORT, timeout=10)
            try:
                if c.cmd("ECHO", f"{tag}-{i}") != f"{tag}-{i}".encode():
                    errors.append(f"churn connection {i} got another connection's reply")
                    break
                if i % 3 == 0:
                    c.sock.sendall(encode(("SET", f"{tag}:churn:{i}", "x" * 20000)))   # reply unread
                if i % 2 == 0:
                    c.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            finally:
                c.close()
    except Exception as e:  # noqa: BLE001
        errors.append(f"churn: {type(e).__name__}: {e}")
    stop.set()
    th.join(30)
    check(f"[{tag}] 400 connections churned beside a live one", not errors, "; ".join(errors[:3]))
    with Conn(PORT) as c:
        check(f"[{tag}] server answers after the churn", c.cmd("PING") == "PONG")


def features_line(log):
    m = re.search(r"io_uring features \(worker 0\):(.*)", log)
    if not m:
        return None
    return dict(re.findall(r"(\w+)=([A-Z_]+)", m.group(1)))


def run_config(binary, work, name, flags, extra_env, asked):
    print(f"\n[{name}] {' '.join(flags) or '(no flags)'} {' '.join(f'{k}={v}' for k, v in extra_env.items())}")
    env = {k: v for k, v in os.environ.items() if k not in ENV_SWITCHES}
    env.update(extra_env)
    log_path = os.path.join(work, f"server_{name}.log")
    cwd = os.path.join(work, name)
    os.makedirs(cwd, exist_ok=True)
    proc = subprocess.Popen([binary, "-p", str(PORT), "--iouring", "--no-auto-detect",
                             "--no-auto-embed", "--no-crash-log", "--no-wal"] + flags,
                            cwd=cwd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    try:
        try:
            wait_ready(PORT, 60, proc=proc)
        except RuntimeError as e:
            check(f"[{name}] server answers PING", False, str(e))
            return
        log = open(log_path).read()
        if not check(f"[{name}] the io_uring loop runs", "IO_URING Engine Active" in log,
                     "the server picked another loop; every later check would be vacuous"):
            return
        feats = features_line(log)
        if not asked:
            check(f"[{name}] no feature line when nothing is asked for", feats is None,
                  str(feats))
        else:
            if check(f"[{name}] the log names what the kernel granted", feats is not None,
                     log[-600:]):
                kv = kernel_version()
                for f in sorted(asked):
                    state = feats.get(f, "MISSING")
                    print(f"  INFO  {f}={state} (kernel {kv[0]}.{kv[1]})")
                    if kv >= MIN_KERNEL[f]:
                        check(f"[{name}] {f} ACTIVE on this kernel", state == "ACTIVE", state)
        tag = name.replace("-", "")
        w_concurrent_pipelines(tag)
        w_large_and_split(tag)
        w_parked_then_pipelined(tag)
        w_multi_exec(tag)
        w_churn(tag)
        check(f"[{name}] server still running", proc.poll() is None, f"exit {proc.returncode}")
        # Graceful stop: the drain runs from the loop's tick.
        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=20)
            check(f"[{name}] SIGTERM exits cleanly", rc == 0, f"exit {rc}")
        except subprocess.TimeoutExpired:
            check(f"[{name}] SIGTERM exits cleanly", False, "still running 20 s after SIGTERM")
        log = open(log_path).read()
        check(f"[{name}] no worker died", "DIED" not in log, log[-400:])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            wait_port_free(PORT)
        except RuntimeError:
            pass


def main() -> int:
    if platform.system() != "Linux":
        print("SKIP: io_uring is Linux-only")
        return 0
    binary = os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server"))
    if "--binary" in sys.argv:
        binary = sys.argv[sys.argv.index("--binary") + 1]
    binary = os.path.abspath(binary)
    work = tempfile.mkdtemp(prefix="uringfeat_")
    try:
        for name, flags, extra_env, asked in CONFIGS:
            run_config(binary, work, name, flags, extra_env, asked)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
