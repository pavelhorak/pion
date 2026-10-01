#!/usr/bin/env python3
"""Vector-set commands (VADD/VSIM/…) differential: Pion vs a real Redis 8.

Redis 8 ships the vectorset module, so for this surface "is Pion's reply
right?" has an oracle, the way tests/test_redis_differential.py has one for
the KV commands (the tool that found most of 0.x's wire bugs). Every step is
sent to both servers and the replies compared under a declared comparison:

  exact  — byte-for-byte after RESP decoding (counts, membership, attributes,
           return codes, error-vs-value, nil-vs-empty)
  set    — same elements, any order (VSIM over a set smaller than COUNT:
           every element must come back, so ranking noise cannot hide a
           missing or extra element)
  first  — same first element (VSIM for an element against itself)
  shape  — same RESP type and length (VINFO, VEMB: values differ by
           quantization, but a nil where Redis has an array is a bug)

Errors are compared as "both error" / "neither error", not by message text.

Starts its own redis-server and pion-server. Needs `redis-server` on PATH
with the vectorset module (Redis 8+); otherwise it SKIPS with exit 2 — and
tests/run_all.py lists the skip with its reason.

Usage: python3 tests/test_vset_differential.py [--binary ./pion-server] [-v]
"""
import argparse
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PION_PORT, REDIS_PORT = 6420, 6421


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.f = self.s.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            a = a if isinstance(a, bytes) else str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("closed")
        t, body = line[:1], line[1:-2]
        if t in (b"*", b">", b"~"):
            n = int(body)
            return None if n < 0 else [self._read() for _ in range(n)]
        if t == b"%":
            return [self._read() for _ in range(2 * int(body))]
        if t == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"-":
            return ("ERR", body)
        if t == b":":
            return int(body)
        if t == b",":
            return float(body)
        if t == b"_":
            return None
        if t == b"#":
            return body == b"t"
        return body   # simple string


def is_err(v):
    return isinstance(v, tuple) and v and v[0] == "ERR"


def f32(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


def rvec(dim, seed):
    rnd = random.Random(seed)
    return [rnd.gauss(0, 1) for _ in range(dim)]


def compare(kind, a, b):
    """True if Pion reply `a` matches Redis reply `b` under `kind`."""
    if is_err(a) or is_err(b):
        return is_err(a) == is_err(b)
    if kind == "exact":
        return a == b
    if kind == "set":
        return isinstance(a, list) and isinstance(b, list) and sorted(a) == sorted(b)
    if kind == "first":
        return isinstance(a, list) and isinstance(b, list) and a[:1] == b[:1]
    if kind == "shape":
        if a is None or b is None:
            return a is b
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b)
        return type(a) is type(b)
    raise ValueError(kind)


def script(dim=16, n=12):
    """(kind, args) steps. Each block is a self-contained scenario on its own key."""
    S = []
    add = lambda k, e, seed, *extra: S.append(("exact", ["VADD", k, "FP32", f32(rvec(dim, seed)), e, *extra]))
    # --- basics on a fresh key
    for i in range(n):
        add("vs", f"e{i}", i)
    S += [("exact", ["VCARD", "vs"]), ("exact", ["VDIM", "vs"]),
          ("exact", ["VISMEMBER", "vs", "e3"]), ("exact", ["VISMEMBER", "vs", "nope"])]
    add("vs", "e3", 3)                      # re-add same element: 0, not 1
    S += [("exact", ["VCARD", "vs"])]
    # --- VSIM: every element back when COUNT >= card; self first
    S += [("set", ["VSIM", "vs", "ELE", "e5", "COUNT", 50]),
          ("first", ["VSIM", "vs", "ELE", "e5", "COUNT", 3]),
          ("first", ["VSIM", "vs", "FP32", f32(rvec(dim, 7)), "COUNT", 3]),
          ("set", ["VSIM", "vs", "VALUES", dim, *[f"{x:.6f}" for x in rvec(dim, 2)], "COUNT", 50])]
    # --- VALUES ingest form
    one_hot = lambda i: ["1" if j == i else "0" for j in range(dim)]
    S += [("exact", ["VADD", "vv", "VALUES", dim, *one_hot(0), "a"]),
          ("exact", ["VADD", "vv", "VALUES", dim, *one_hot(1), "b"]),
          ("first", ["VSIM", "vv", "VALUES", dim, *(["0.9", "0.1"] + ["0"] * (dim - 2)), "COUNT", 2])]
    # --- attributes
    S += [("exact", ["VSETATTR", "vs", "e1", '{"color":"red"}']),
          ("exact", ["VGETATTR", "vs", "e1"]), ("exact", ["VGETATTR", "vs", "e2"]),
          ("exact", ["VSETATTR", "vs", "missing", '{"a":1}'])]
    # --- removal
    S += [("exact", ["VREM", "vs", "e0"]), ("exact", ["VREM", "vs", "e0"]),
          ("exact", ["VCARD", "vs"]), ("exact", ["VISMEMBER", "vs", "e0"]),
          ("set", ["VSIM", "vs", "ELE", "e5", "COUNT", 50])]
    # --- shapes where values differ by quantization
    S += [("shape", ["VEMB", "vs", "e4"]), ("shape", ["VEMB", "vs", "nope"]), ("shape", ["VINFO", "vs"])]
    # --- missing key replies
    S += [("exact", ["VCARD", "nokey"]), ("exact", ["VDIM", "nokey"]),
          ("exact", ["VISMEMBER", "nokey", "x"]), ("exact", ["VREM", "nokey", "x"]),
          ("shape", ["VSIM", "nokey", "ELE", "x"]), ("exact", ["VGETATTR", "nokey", "x"])]
    # --- wrong type
    S += [("exact", ["SET", "str", "hello"]), ("exact", ["VCARD", "str"]),
          ("exact", ["VADD", "str", "VALUES", 2, "1", "2", "x"]), ("exact", ["GET", "str"])]
    # --- dimension mismatch on an existing set
    S += [("exact", ["VADD", "vs", "VALUES", 3, "1", "2", "3", "wrongdim"]), ("exact", ["VCARD", "vs"])]
    # --- removing the last element deletes the key (aggregate rule)
    S += [("exact", ["VADD", "one", "VALUES", dim, *one_hot(0), "only"]), ("exact", ["VREM", "one", "only"]),
          ("exact", ["EXISTS", "one"])]
    return S


def start(cmd, port, cwd):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(150):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return p
        except OSError:
            time.sleep(0.1)
    p.kill()
    raise SystemExit(f"server on {port} did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--dim", type=int, default=16, help="vector dim for the main script")
    args = ap.parse_args()
    rs = shutil.which("redis-server")
    if not rs:
        print("SKIP: redis-server not on PATH")
        return 2
    tmp = tempfile.mkdtemp(prefix="vset_diff_")
    redis = start([rs, "--port", str(REDIS_PORT), "--save", "", "--appendonly", "no"], REDIS_PORT, tmp)
    pion = start([os.path.abspath(args.binary), "-p", str(PION_PORT), "-w", "1", "--no-auto-detect",
                  "--no-auto-embed", "--no-crash-log"], PION_PORT, tmp)
    try:
        r = Client(REDIS_PORT)
        probe = r.cmd("VCARD", "__probe__")
        if is_err(probe) and b"unknown command" in probe[1].lower():
            print("SKIP: this redis-server has no vectorset module (needs Redis 8+)")
            return 2
        p = Client(PION_PORT)
        diffs = 0
        steps = script(dim=args.dim)
        for kind, cmd in steps:
            try:
                a = p.cmd(*cmd)
            except ConnectionError:
                print(f"FAIL: Pion closed the connection on {cmd[0]} {cmd[1:3]}")
                return 1
            b = r.cmd(*cmd)
            ok = compare(kind, a, b)
            shown = [c if len(str(c)) < 24 else f"<{len(c)}B>" for c in cmd[:6]]
            if not ok:
                diffs += 1
                print(f"  DIFF [{kind}] {shown}\n        pion : {a!r:.160}\n        redis: {b!r:.160}")
            elif args.verbose:
                print(f"  ok   [{kind}] {shown}")
        print(f"\n{len(steps) - diffs}/{len(steps)} steps agree with Redis {r.cmd('INFO', 'server').split(b'redis_version:')[1][:8].decode(errors='replace').strip()}")
        return 1 if diffs else 0
    finally:
        for proc in (pion, redis):
            proc.kill()
            proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
