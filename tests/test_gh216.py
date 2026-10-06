#!/usr/bin/env python3
"""gh #216 / gh #217 — fast-path mutations that never reached the WAL.

Both bugs are the same shape: a command handled entirely on the fast path
mutates the keyspace without writing an effect record, so the write is gone
after a crash. gh #216 is INCR/DECR, gh #217 is MSET.

## gh #216 — INCR/DECR durability

The fast path used to log INCR/DECR as a cmd-1 SET with a NULL value pointer
and length 0, so replay reconstructed nothing:

  * a counter created by INCR came back as an EMPTY STRING (and the next INCR
    against it then failed, because an empty string is not an integer);
  * a counter incremented over an existing SET rewound to the pre-INCR value.

Why the pre-existing coverage missed it: tests/test_gh170.py writes
`INCR int:a` followed by `INCRBY int:a 41`. INCRBY runs on the slow path and
logs a full-value record, which overwrites the broken one — so the assertion on
the final value passed while INCR alone was silently lossy. Every counter here
is therefore written with INCR/DECR *only*, with no later value-carrying write
to the same key.

## gh #217 — MSET durability

The fast-path MSET arm wrote each pair straight into the keyspace with no
effect record, so a cleanly-parsed MSET vanished on replay. The slow-path arm
routes through `execute_set` and always logged, which is why MSET durability
was path-dependent rather than uniformly broken.

Two crash modes, matching test_gh170.py:
  Mode A: SAVE -> SIGKILL -> restart   (snapshot serializer path)
  Mode B: no SAVE -> SIGKILL -> restart (WAL effect-record replay path)

Only Mode B can catch either bug — a SAVE serializes live memory, which is
correct regardless of what the WAL holds. Mode A is here to prove the fixes do
not break the snapshot path.

Usage: python3 tests/test_gh216.py [./pion-server-dev]
"""
import os, socket, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader, wait_ready_pid  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1979
WORKDIR = f"/tmp/pion_gh216_test_{PORT}"


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
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
        s = socket.socket()
        try:
            s.settimeout(1.0)
            s.connect(("127.0.0.1", PORT))
            s.settimeout(None)
            return _await_ready(s)
        except OSError:
            s.close()
            time.sleep(0.2)
    raise RuntimeError("server did not come up")


def start():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def kill(proc):
    proc.kill()
    proc.wait(timeout=5)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(0.3)
            s.connect(("127.0.0.1", PORT))
            s.close()
            time.sleep(0.2)
        except OSError:
            s.close()
            return
    raise RuntimeError("port still busy")


def write_workload(s):
    # Counters written with INCR/DECR ONLY — nothing later rewrites the value.
    send(s, "INCR", "c:fresh")                  # 1
    send(s, "INCR", "c:fresh")                  # 2
    send(s, "INCR", "c:fresh")                  # 3
    send(s, "DECR", "c:dec")                    # -1
    send(s, "DECR", "c:dec")                    # -2
    send(s, "SET", "c:over", "41")
    send(s, "INCR", "c:over")                   # 42, over an existing SET
    send(s, "SET", "c:neg", "-5")
    send(s, "DECR", "c:neg")                    # -6
    # The shape gh170 already covered: a value-carrying write after the INCR.
    # It masked the bug, so it must keep passing rather than be dropped.
    send(s, "INCR", "c:masked")
    send(s, "INCRBY", "c:masked", "41")         # 42
    # gh #217: a single fast-path MSET, nothing else touching these keys.
    send(s, "MSET", *[t for kv in MSET_KEYS for t in kv])


COUNTERS = [
    ("c:fresh", "3"),
    ("c:dec", "-2"),
    ("c:over", "42"),
    ("c:neg", "-6"),
    ("c:masked", "42"),
]

# gh #217, found by the same scan: the fast-path MSET arm wrote to the keyspace
# without an effect record, so an MSET that parsed cleanly was lost entirely.
# The slow-path arm routes through execute_set and was always durable.
MSET_KEYS = [("ms:a", "1"), ("ms:b", "2"), ("ms:c", "3")]


def parse_bulk(reply):
    """Return the payload of a bulk-string reply, or None for a nil reply."""
    if reply.startswith("$-1"):
        return None
    if not reply.startswith("$"):
        return reply.strip()
    return reply.split("\r\n", 1)[1].rstrip("\r\n")


def verify(s, mode):
    failures = []
    for key, want in COUNTERS + MSET_KEYS:
        got = parse_bulk(send(s, "GET", key))
        if got != want:
            shown = "<nil>" if got is None else f"'{got}'"
            failures.append(f"{mode} GET {key}: got {shown}, want '{want}'")

    # A recovered counter must still BE a counter. The empty-string failure mode
    # restores the key but not its semantics, so a value check alone can pass
    # while the next INCR errors out.
    for key, want in COUNTERS:
        reply = send(s, "INCR", key).strip()
        if not reply.startswith(":"):
            failures.append(f"{mode} INCR {key} after recovery: {reply!r}")
        elif reply != f":{int(want) + 1}":
            failures.append(f"{mode} INCR {key} after recovery: {reply!r}, "
                            f"want ':{int(want) + 1}'")
    return failures


def run_mode(save_first):
    mode = "MODE A (snapshot)" if save_first else "MODE B (WAL replay)"
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    proc = start()
    s = connect(proc)
    write_workload(s)
    if save_first:
        send(s, "SAVE")
    s.close()
    kill(proc)

    proc = start()
    s = connect(proc)
    failures = verify(s, mode)
    s.close()
    kill(proc)
    shutil.rmtree(WORKDIR, ignore_errors=True)

    if failures:
        for f in failures:
            print(f"  FAIL {f}")
    else:
        print(f"  PASS {mode}: {len(COUNTERS)} counters survived with value and "
              f"type intact, {len(MSET_KEYS)} MSET keys survived")
    return failures


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found: {BINARY}")
        return 1
    print(f"gh #216 INCR/DECR + gh #217 MSET durability — {BINARY}, port {PORT}")
    failures = run_mode(save_first=False) + run_mode(save_first=True)
    print()
    if failures:
        print(f"FAILED — {len(failures)} check(s)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
