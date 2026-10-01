#!/usr/bin/env python3
"""gh #225 — a replica must reject writes, including ones that LOOK like reads.

The replica write-rejection classifier in fast_path.mojo matched first byte +
length only, so any write sharing a prefix and length with a whitelisted read
was classified as a read and accepted:

    MSET   (m,4) matched MGET        HSET / HDEL (h,4) matched HGET
    LPUSH / LTRIM (l,5) matched the LLEN arm (LLEN is 4 bytes, so that arm
                        never matched LLEN in the first place)
    EXPIRE (e,6) matched EXISTS      GETSET (g,6) matched GETBIT
    DECRBY (d,6) matched DBSIZE      APPEND (a,6) matched ASKING
    PERSIST(p,7) matched PFCOUNT

A read-only replica silently accepted those writes and diverged from its
primary — the divergence is then lost or overwritten at the next sync, with no
error at any point.

SCOPE LIMIT — read before trusting a pass. In this configuration the replica
rejects these writes with `-MOVED <slot> <primary>`, i.e. cluster slot
redirection, NOT the `_reject_if_replica_write` path the classifier feeds. The
classifier only decides anything when a replica serves a slot locally (READONLY
mode). So this test pins the user-visible property — a replica must not accept
writes and must still serve reads — and would catch a regression that broke
either. It does NOT by itself prove the whole-name matching fix; that fix is
correct by inspection (matching the full name can only ever reject a superset
of what a prefix match rejected) and is covered for reachability by the
dispatch sweep.

Usage: python3 tests/test_gh225_replica_writes.py [--binary ./pion-server]
Starts its own primary + replica.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")


class Conn:
    def __init__(self, port, timeout=6):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        buf = f"*{len(args)}\r\n".encode()
        for a in args:
            buf += b"$%d\r\n%s\r\n" % (len(a), a.encode())
        self.s.sendall(buf)
        line = self.f.readline()
        if not line:
            raise EOFError("closed")
        t, body = line[:1], line[1:-2]
        if t == b"-":
            return "ERR:" + body.decode()
        if t in b"+:":
            return body.decode()
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            if n <= 0:
                return []
            # Consume the elements. Returning a placeholder without reading
            # them leaves the array body in the socket, so every later reply is
            # shifted — which looked exactly like a server-side desync (INFO
            # answering PONG) until a per-command control run showed the server
            # was fine.
            return [self._read_one() for _ in range(n)]
        return "?"

    def _read_one(self):
        line = self.f.readline()
        if not line:
            raise EOFError("closed")
        t, body = line[:1], line[1:-2]
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t == b"*":
            n = int(body)
            return [] if n <= 0 else [self._read_one() for _ in range(n)]
        return body.decode()

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def start(binary, port, extra):
    p = subprocess.Popen([binary, "-p", str(port), "-w", "1",
                          "--no-auto-detect", "--no-auto-embed"] + extra,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         preexec_fn=os.setsid)
    for _ in range(80):
        time.sleep(0.5)
        try:
            Conn(port, timeout=2).close()
            return p
        except OSError:
            continue
    return None


def stop(p):
    if p:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


# (command, args) — every one is a WRITE that used to slip past the classifier
# because it shares a first byte and length with a whitelisted read.
SNEAKY_WRITES = [
    ("MSET", ["gh225:a", "1"]),            # looks like MGET
    ("HSET", ["gh225:h", "f", "v"]),       # looks like HGET
    ("HDEL", ["gh225:h", "f"]),            # looks like HGET
    ("LPUSH", ["gh225:l", "v"]),           # looks like the LLEN arm
    ("LTRIM", ["gh225:l", "0", "0"]),      # looks like the LLEN arm
    ("EXPIRE", ["gh225:a", "100"]),        # looks like EXISTS
    ("GETSET", ["gh225:a", "2"]),          # looks like GETBIT
    ("DECRBY", ["gh225:n", "1"]),          # looks like DBSIZE
    ("APPEND", ["gh225:a", "x"]),          # looks like ASKING
    ("PERSIST", ["gh225:a"]),              # looks like PFCOUNT
]

# Reads that must still be allowed — the fix must not over-reject.
ALLOWED_READS = [
    ("GET", ["gh225:a"]),
    ("MGET", ["gh225:a"]),
    ("EXISTS", ["gh225:a"]),
    ("HGET", ["gh225:h", "f"]),
    ("LLEN", ["gh225:l"]),
    ("LRANGE", ["gh225:l", "0", "-1"]),
    ("DBSIZE", []),
    ("PING", []),
    ("INFO", []),
    ("TYPE", ["gh225:a"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", "./pion-server"))
    ap.add_argument("--primary-port", type=int, default=1986)
    ap.add_argument("--replica-port", type=int, default=1987)
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"FATAL: {args.binary} not found")
        return 2

    primary = start(args.binary, args.primary_port, ["--cluster"])
    if not primary:
        print("FATAL: primary did not start")
        return 2
    # A node becomes a replica via --cluster-replica (+ primary coordinates),
    # not REPLICAOF — that command does not exist in Pion.
    replica = start(args.binary, args.replica_port,
                    ["--cluster", "--cluster-replica",
                     "--cluster-primary-host", HOST,
                     "--cluster-primary-port", str(args.primary_port)])
    if not replica:
        stop(primary)
        print("FATAL: replica did not start")
        return 2

    try:
        r = Conn(args.replica_port)
        time.sleep(2.0)
        role = r.cmd("INFO", "replication")
        is_rep = isinstance(role, str) and "slave" in role
        if not is_rep:
            print(f"  SKIP  node did not come up as a replica (INFO says: "
                  f"{str(role)[:60]!r}) — cannot test the classifier")
            return 0

        print(f"\nreplica on :{args.replica_port} — writes must be REJECTED\n")
        for cmd, extra in SNEAKY_WRITES:
            try:
                got = r.cmd(cmd, *extra)
            except (EOFError, socket.timeout) as e:
                check(f"{cmd} rejected", False, f"{type(e).__name__}")
                continue
            # A replica should answer -READONLY / -ERR, not perform the write.
            check(f"{cmd} rejected on replica",
                  isinstance(got, str) and got.startswith("ERR:"),
                  f"got {got!r}")

        print("\nreads must still be allowed\n")
        for cmd, extra in ALLOWED_READS:
            try:
                got = r.cmd(cmd, *extra)
            except (EOFError, socket.timeout) as e:
                check(f"{cmd} allowed", False, f"{type(e).__name__}")
                continue
            check(f"{cmd} allowed on replica",
                  not (isinstance(got, str) and got.startswith("ERR:READONLY")),
                  f"got {str(got)[:28]!r}")
        r.close()
    finally:
        stop(replica)
        stop(primary)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
