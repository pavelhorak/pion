#!/usr/bin/env python3
"""gh #166 — RESP token table: `VADD ... VALUES` at high dimension.

`VADD key VALUES 1536 <scalars> elem` is ~1539 RESP tokens. At the old 64-token
bound it could only ever be the gh #153 error. Raising the bound alone is not
enough and is actively unsafe: handlers took the token array by value, so every
handler call copied the whole table into its frame — 1.5 KB at 64 tokens, 48 KB
at 2048, against a ~512 KB parallelize worker stack. The array now lives on the
heap, one set per SlowPathHandler, and handlers take a pointer.

The `-w 8` case is the one that matters, because the failure mode is
multi-worker only. It is skipped for -O0 dev binaries: those SIGBUS at `-w >= 2`
on any command, unmodified code included (A/B verified against a stashed tree),
because their frames do not fit a worker stack.

Usage: python3 tests/test_gh166_high_dim_values.py [--binary ./pion-server] [--port 1989]
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


# ── RESP plumbing ───────────────────────────────────────────────────────────

def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        out += b"$%d\r\n%s\r\n" % (len(a), a)
    return out


class Client:
    def __init__(self, port, timeout=120):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # A multi-MB bulk reply must be read through a buffered file object —
        # slicing a growing bytes object is O(N^2) and turns a 6 MB GET into a
        # visible stall (see the python_resp_array_slicing_trap note).
        self.f = self.s.makefile("rb")

    def call(self, *args):
        self.s.sendall(cmd(*args))
        return self.read()

    def read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("connection closed")
        t, rest = line[0:1], line[1:-2]
        if t in b"+-:":
            return rest
        if t == b"$":
            n = int(rest)
            if n == -1:
                return None
            data = self.f.read(n + 2)
            return data[:-2]
        if t == b"*":
            return [self.read() for _ in range(int(rest))]
        raise RuntimeError(f"unexpected reply: {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


# ── server lifecycle ────────────────────────────────────────────────────────

class Server:
    def __init__(self, binary, port, datadir, extra=(), profile="kv", workers=1):
        self.binary, self.port, self.datadir = binary, port, datadir
        self.extra = list(extra)
        self.profile, self.workers = profile, workers
        self.proc = None

    def start(self):
        cmd_ = [self.binary, "-p", str(self.port), "-w", str(self.workers),
                "--profile", self.profile,
                "--no-auto-detect", "--no-auto-embed"] + self.extra
        if self.workers > 1:
            cmd_.append("--independent-workers")   # gh #253
        self.log = os.path.join(self.datadir, f"server.{int(time.time()*1000)%100000}.log")
        fp = open(self.log, "w")
        self.proc = subprocess.Popen(cmd_, cwd=self.datadir, stdout=fp, stderr=fp,
                                     preexec_fn=os.setsid)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                c = Client(self.port, timeout=2)
                if c.call("PING") == b"PONG":
                    c.close()
                    return self
                c.close()
            except (OSError, EOFError):
                time.sleep(0.3)
        raise RuntimeError(f"server did not come up on {self.port}; see {self.log}")

    def sigkill(self):
        """The jetsam shape: uncatchable, nothing flushed on the way out."""
        if self.proc:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()
            self.proc = None

    def log_text(self):
        try:
            with open(self.log) as fh:
                return fh.read()
        except OSError:
            return ""


def payload(i, mb):
    return bytes(((i * 31 + j * 17) & 0xFF) for j in range(256)) * (mb * 4096)


def free_dir():
    return tempfile.mkdtemp(prefix="pion_gh149_")


def test_high_dim_vadd_values(binary, port):
    """gh #166: `VADD key VALUES 1536 ...` is ~1539 RESP tokens. At the old
    64-token bound it could only ever be the gh #153 error."""
    d = free_dir()
    srv = Server(binary, port, d, profile="vector").start()
    c = Client(port)
    vals = [f"{(i % 97) * 0.01:.4f}" for i in range(1536)]
    vals2 = [f"{((i + 7) % 97) * 0.01:.4f}" for i in range(1536)]
    r1 = c.call("VADD", "vs", "VALUES", "1536", *vals, "elem-a")
    r2 = c.call("VADD", "vs", "VALUES", "1536", *vals2, "elem-b")
    card = c.call("VCARD", "vs")
    dim = c.call("VDIM", "vs")
    sim = c.call("VSIM", "vs", "VALUES", "1536", *vals, "COUNT", "2")
    # reply pairing must hold with a 1539-token command inside a pipeline
    c.s.sendall(cmd("INCR", "pc")
                + cmd("VADD", "vs", "VALUES", "1536", *vals, "elem-c")
                + cmd("INCR", "pc"))
    p1, p2, p3 = c.read(), c.read(), c.read()
    pong = c.call("PING")
    c.close()
    srv.sigkill()

    check("gh #166: VADD VALUES at dim 1536 is accepted", r1 == b"1",
          f"reply {r1!r}")
    check("gh #166: a second high-dim VALUES insert works", r2 == b"1")
    check("gh #166: VCARD sees both elements", card == b"2", f"reply {card!r}")
    check("gh #166: VDIM reports 1536", dim == b"1536", f"reply {dim!r}")
    check("gh #166: VSIM VALUES at dim 1536 returns the nearest element",
          isinstance(sim, list) and len(sim) >= 1 and sim[0] == b"elem-a",
          f"reply {sim!r}")
    check("gh #166: pipelined around a 1539-token command, replies stay paired",
          (p1, p2, p3) == (b"1", b"1", b"2"), f"replies {p1!r} {p2!r} {p3!r}")
    check("gh #166: connection is usable afterwards", pong == b"PONG")

    # The stack-overflow failure mode this fix exists to avoid is multi-worker
    # only, so the bound has to be exercised at -w 8. Skipped for -O0 dev
    # binaries: those SIGBUS at -w >= 2 on any command, unmodified code included
    # (verified by A/B), because their frames do not fit a parallelize worker
    # stack. Release builds are the ones that mean anything here.
    if "-dev" in os.path.basename(binary):
        print("  SKIP  gh #166: -w 8 check (dev/-O0 binary — see comment)")
    else:
        d8 = free_dir()
        srv8 = Server(binary, port, d8, profile="vector", workers=8).start()
        c8 = Client(port)
        r8 = c8.call("VADD", "vs8", "VALUES", "1536", *vals, "elem-a")
        pong8 = c8.call("PING")
        c8.close()
        srv8.sigkill()
        check("gh #166: VADD VALUES 1536 at -w 8 (worker-stack bound)",
              r8 == b"1" and pong8 == b"PONG", f"reply {r8!r}, ping {pong8!r}")
        shutil.rmtree(d8, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
    ap.add_argument("--port", type=int, default=1989)
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2
    print(f"gh #166 high-dimension VADD VALUES — {args.binary} port {args.port}\n")
    try:
        test_high_dim_vadd_values(args.binary, args.port)
    except Exception as exc:
        check("test_high_dim_vadd_values", False, f"{type(exc).__name__}: {exc}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


sys.exit(main())
