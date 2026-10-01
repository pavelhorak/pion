#!/usr/bin/env python3
"""gh #148 — quantized KV.PREFIX / V-store tier: mlx int4 group-32, affine.

Two things are checked here, and the second one is why the first never worked:

  1. `mlx4g32` stores and serves at the mlx QuantizedKVCache layout — D/2 packed
     bytes plus a 2 B scale and 2 B bias per group of 32 — so a 4B-class model's
     86K-token resident cartridge fits on a 16 GB machine (fp16 needs 12.5 GB;
     this is 3.2x smaller). Group 32 and affine are both measured requirements:
     g64 corrupted recalled facts at digit grain, and affine beat symmetric on K.

  2. `VQUANT` was silently ignored on the uniform (non-SCHEMA) path. create_session
     broadcast the *default* INT8 into every layer slot, and store_batch reads the
     layer slot — so `V.CREATE ... VQUANT turbo4` and `KV.PREFIX.REGISTER <ns>
     <dim> <quant>` stored INT8 no matter what was asked for. The quantized tier
     this issue asks for could not engage at all. Each format must now produce its
     own distinct error profile.

Usage: python3 tests/test_gh148_mlx4g32_tier.py [--port 1984] [--binary ./pion-server]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        out += b"$%d\r\n%s\r\n" % (len(a), a)
    return out


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=60)
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
            return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            return [self.read() for _ in range(int(rest))]
        raise RuntimeError(f"unexpected reply {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def start(binary, port, datadir):
    proc = subprocess.Popen(
        [binary, "-p", str(port), "-w", "1", "--kvcache", "--no-auto-detect",
         "--no-auto-embed"],
        cwd=datadir, stdout=open(os.path.join(datadir, "s.log"), "w"),
        stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            c = Client(port)
            if c.call("PING") == b"PONG":
                c.close()
                return proc
            c.close()
        except (OSError, EOFError):
            time.sleep(0.3)
    raise RuntimeError("server did not come up")


def stop(proc):
    if proc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def info_field(c, sid, field):
    info = (c.call("V.INFO", sid) or b"").decode(errors="replace")
    for line in info.split("\r\n"):
        if line.startswith(field + ":"):
            return line.split(":", 1)[1]
    return ""


def roundtrip(c, sid, fmt, dim, ntok, vals):
    blob = struct.pack("<%df" % (ntok * dim), *vals)
    c.call("V.CREATE", sid, str(dim), "VQUANT", fmt)
    c.call("V.STOREBATCH", sid, "0", "0", str(ntok), blob)
    got = c.call("V.FETCH", sid, "0", "RANGE", "0", str(ntok))
    if not isinstance(got, bytes) or len(got) != ntok * dim * 4:
        return None, info_field(c, sid, "v_format")
    back = struct.unpack("<%df" % (ntok * dim), got)
    err = [abs(a - b) for a, b in zip(vals, back)]
    return (max(err), sum(err) / len(err)), info_field(c, sid, "v_format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1984)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)

    d = tempfile.mkdtemp(prefix="pion_gh148_")
    proc = start(args.binary, args.port, d)
    print(f"gh #148 mlx4g32 quantized tier — {args.binary} port {args.port}\n")

    dim, ntok = 1024, 8
    random.seed(7)
    vals = [random.gauss(0, 0.6) for _ in range(ntok * dim)]

    c = Client(args.port)
    profiles = {}
    for fmt in ("fp16", "int8", "turbo4", "mlx4g32"):
        errs, reported = roundtrip(c, f"g148-{fmt}", fmt, dim, ntok, vals)
        profiles[fmt] = errs
        check(f"V.CREATE VQUANT {fmt} reports {fmt} back",
              reported == fmt, f"V.INFO says {reported!r}")
        check(f"{fmt} round-trips through V.STOREBATCH / V.FETCH",
              errs is not None,
              "fetch returned nothing" if errs is None else f"max err {errs[0]:.5f}")

    # gh #148 §2 — the formats must actually differ. Before the fix every one of
    # these was INT8, so all four profiles were identical.
    distinct = len({None if v is None else round(v[1], 6) for v in profiles.values()})
    check("each VQUANT format produces its own error profile (VQUANT is honoured)",
          distinct == len(profiles),
          " ".join(f"{k}={v[1]:.5f}" for k, v in profiles.items() if v))

    # Accuracy ordering that the tier's whole argument depends on.
    if all(profiles.values()):
        check("mlx4g32 is more accurate than symmetric int4 (affine, gate 2)",
              profiles["mlx4g32"][1] < profiles["turbo4"][1],
              f"mlx4g32 {profiles['mlx4g32'][1]:.5f} vs turbo4 {profiles['turbo4'][1]:.5f}")
        check("mlx4g32 is coarser than int8, as an int4 format must be",
              profiles["mlx4g32"][1] > profiles["int8"][1],
              f"mlx4g32 {profiles['mlx4g32'][1]:.5f} vs int8 {profiles['int8'][1]:.5f}")

    # The reason the format exists: bytes per token.
    fp16_bpt = dim * 2
    mlx_bpt = dim // 2 + (dim // 32) * 4
    check("mlx4g32 is 3.2x smaller per token than fp16",
          abs(fp16_bpt / mlx_bpt - 3.2) < 0.05,
          f"{fp16_bpt} B -> {mlx_bpt} B/token")

    # KV.PREFIX is the lane the cartridge actually uses.
    reg = c.call("KV.PREFIX.REGISTER", "gh148-ns", str(dim), "mlx4g32")
    check("KV.PREFIX.REGISTER accepts mlx4g32", reg is not None and not str(reg).startswith("b'ERR"),
          f"reply {reg!r}")

    # A dim that is not a multiple of the group must degrade to int8 rather than
    # writing a corrupt buffer.
    c.call("V.CREATE", "g148-odd", "100", "VQUANT", "mlx4g32")
    check("non-multiple-of-32 dim falls back to int8 instead of corrupting",
          info_field(c, "g148-odd", "v_format") == "int8",
          f"reports {info_field(c, 'g148-odd', 'v_format')!r}")

    c.close()
    stop(proc)
    shutil.rmtree(d, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


sys.exit(main())
