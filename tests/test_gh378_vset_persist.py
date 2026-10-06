#!/usr/bin/env python3
"""gh #378 — vector sets survive a restart.

Vector sets (gh #366) were written to neither the WAL nor the snapshot, so any
restart emptied them. Now VADD / VREM / VSETATTR append effect records (WAL
28/29/30) and SAVE / BGREWRITEAOF serialize a set as the same records.

Every mode builds the same workload, captures the full observable state of
every key — TYPE, VCARD, VDIM, VINFO, VRANGE, VEMB of every element, every
attribute, VSIM with scores — then SIGKILLs the server, restarts it in the same
directory and requires the state to come back BYTE-IDENTICAL:

  A  SAVE -> SIGKILL                      snapshot writer + loader
  B  no SAVE -> SIGKILL                   WAL effect-record replay
  C  SAVE, more writes -> SIGKILL         snapshot base + WAL delta on top
  D  BGREWRITEAOF -> SIGKILL              the WAL rewriter

A, C and D compact: they write live elements only, so VINFO's
hnsw-max-node-uid (a slot counter that includes VREM tombstones) restarts at
the live count, as it does across a Redis RDB load. That one field is checked
for exactly that instead of for equality; B (pure replay) must match it too.

The workload covers what a replay can get wrong: a re-VADD that replaces a
vector, attributes set, changed and cleared, VREM of a middle element, a set
emptied by VREM (must stay gone), a high-dimensional set, a key of another type
that VADD refused (must be untouched), and a vector whose norm is not 1 (VEMB
returns unit x norm, so a replay that re-normalized would drift in the last
digit and the byte compare would catch it).

Usage: python3 tests/test_gh378_vset_persist.py [./pion-server]
"""
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid  # noqa: E402

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6478
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


class Client:
    def __init__(self):
        self.s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
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
        t, rest = line[:1], line[1:-2]
        if t in (b"+", b":"):
            return rest
        if t == b"-":
            return RuntimeError(rest.decode(errors="replace"))
        if t == b"$":
            n = int(rest)
            if n < 0:
                return None
            data = self.f.read(n + 2)[:-2]
            return data
        if t == b"*":
            n = int(rest)
            return None if n < 0 else [self._read() for _ in range(n)]
        if t == b"_":
            return None
        raise RuntimeError(f"unparsed reply {line!r}")

    def close(self):
        self.s.close()


def port_open():
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=0.3).close()
        return True
    except OSError:
        return False


def start(workdir):
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-auto-detect", "--no-auto-embed"],
                         cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready_pid(PORT, p, 30)   # this process, not a lingering listener (#27)
    except RuntimeError:
        p.kill()
        raise
    return p, Client()


def sigkill(p):
    p.kill()
    p.wait(timeout=10)
    deadline = time.monotonic() + 10
    while port_open() and time.monotonic() < deadline:
        time.sleep(0.1)


def fp32(v):
    return struct.pack(f"<{len(v)}f", *v)


def workload(c, rng):
    """Returns the keys to capture."""
    # small set: replace, attributes, VREM of a middle element
    for i, name in enumerate(["a", "b", "c", "d", "e"]):
        c.cmd("VADD", "vs:small", "VALUES", 3, 1.0 + i, 2.0 - i * 0.5, 0.25 * i, name)
    c.cmd("VADD", "vs:small", "VALUES", 3, -3.5, 0.125, 9.0, "b")          # replaces b
    c.cmd("VADD", "vs:small", "VALUES", 3, 0.1, 0.2, 0.3, "f", "SETATTR", '{"k":1}')
    c.cmd("VSETATTR", "vs:small", "a", '{"color":"red"}')
    c.cmd("VSETATTR", "vs:small", "a", '{"color":"blue"}')                 # changed
    c.cmd("VSETATTR", "vs:small", "c", '{"x":2}')
    c.cmd("VSETATTR", "vs:small", "c", "")                                 # cleared
    c.cmd("VREM", "vs:small", "d")
    # FP32-blob set at a real embedding dimension, non-unit norms
    for i in range(40):
        v = [rng.gauss(0, 3) for _ in range(1536)]
        c.cmd("VADD", "vs:big", "FP32", fp32(v), f"doc:{i}")
    c.cmd("VREM", "vs:big", "doc:7")
    # a set emptied by VREM: the key must be gone and stay gone
    c.cmd("VADD", "vs:gone", "VALUES", 2, 1, 1, "x")
    c.cmd("VADD", "vs:gone", "VALUES", 2, 1, 2, "y")
    c.cmd("VREM", "vs:gone", "x")
    c.cmd("VREM", "vs:gone", "y")
    # a key of another type that VADD must have refused, untouched
    c.cmd("SET", "vs:str", "plain")
    r = c.cmd("VADD", "vs:str", "VALUES", 2, 1, 1, "x")
    check("VADD on a string key is refused", isinstance(r, RuntimeError), repr(r))
    # a set deleted with DEL, then re-created at a different dimension
    c.cmd("VADD", "vs:redim", "VALUES", 2, 1, 0, "old")
    c.cmd("DEL", "vs:redim")
    c.cmd("VADD", "vs:redim", "VALUES", 4, 0, 1, 0, 1, "new")
    return ["vs:small", "vs:big", "vs:gone", "vs:str", "vs:redim"]


def more_writes(c, rng):
    """Mode C: mutations after the SAVE, which only the WAL delta can carry."""
    c.cmd("VADD", "vs:small", "VALUES", 3, 7, 7, 7, "g")
    c.cmd("VREM", "vs:small", "e")
    c.cmd("VSETATTR", "vs:small", "b", '{"after":"save"}')
    c.cmd("VADD", "vs:big", "FP32", fp32([rng.gauss(0, 1) for _ in range(1536)]), "doc:40")
    c.cmd("VADD", "vs:late", "VALUES", 2, 0.5, -0.5, "only")


def capture(c, keys):
    state = {}
    for k in keys:
        st = {"type": c.cmd("TYPE", k), "exists": c.cmd("EXISTS", k)}
        if st["type"] == b"vectorset":
            st["card"] = c.cmd("VCARD", k)
            st["dim"] = c.cmd("VDIM", k)
            st["info"] = c.cmd("VINFO", k)
            names = c.cmd("VRANGE", k, "-", "+")
            st["names"] = names
            st["emb"] = {n: c.cmd("VEMB", k, n) for n in names}
            st["attr"] = {n: c.cmd("VGETATTR", k, n) for n in names}
            st["sim"] = {n: c.cmd("VSIM", k, "ELE", n, "WITHSCORES", "COUNT", 5) for n in names[:6]}
        elif st["type"] == b"string":
            st["get"] = c.cmd("GET", k)
        state[k] = st
    return state


UID = b"hnsw-max-node-uid"


def strip_uid(info):
    """VINFO without hnsw-max-node-uid: the slot counter counts tombstones,
    and a snapshot or rewrite writes live elements only, so a compacting
    restore restarts it at the live count (as a Redis RDB load does)."""
    if not isinstance(info, list):
        return info
    out = []
    for f, v in zip(info[0::2], info[1::2]):
        if f != UID:
            out += [f, v]
    return out


def uid(info):
    return dict(zip(info[0::2], info[1::2])).get(UID) if isinstance(info, list) else None


def diff(a, b, compacting):
    out = []
    for k in a:
        for f in a[k]:
            x, y = a[k][f], b.get(k, {}).get(f)
            if f == "info" and compacting:
                x, y = strip_uid(x), strip_uid(y)
            if x != y:
                out.append(f"{k}.{f}")
    return out


def run_mode(label, save, after_save, rewrite):
    workdir = tempfile.mkdtemp(prefix="pion_gh378_")
    rng = random.Random(378)
    p, c = start(workdir)
    try:
        keys = workload(c, rng)
        if save:
            check(f"{label}: SAVE ok", c.cmd("SAVE") == b"OK")
        if after_save:
            more_writes(c, rng)
            keys = keys + ["vs:late"]
        if rewrite:
            r = c.cmd("BGREWRITEAOF")
            check(f"{label}: BGREWRITEAOF accepted", not isinstance(r, RuntimeError), repr(r))
            time.sleep(0.5)
        before = capture(c, keys)
        c.close()
        sigkill(p)
        p, c = start(workdir)
        after = capture(c, keys)
        compacting = save or rewrite
        bad = diff(before, after, compacting)
        check(f"{label}: every key's state is byte-identical after restart", not bad, ", ".join(bad[:8]))
        if compacting and not after_save:   # C's WAL delta VREMs after the SAVE
            check(f"{label}: tombstones compacted (max-node-uid == live count)",
                  uid(after["vs:small"].get("info")) == after["vs:small"].get("card"),
                  f'{uid(after["vs:small"].get("info"))} vs {after["vs:small"].get("card")}')
        check(f"{label}: sets came back non-empty",
              after["vs:small"]["type"] == b"vectorset" and after["vs:big"].get("card") == before["vs:big"].get("card"),
              repr(after["vs:small"].get("type")))
        check(f"{label}: emptied set stays gone", after["vs:gone"]["exists"] == b"0")
        check(f"{label}: refused key keeps its string", after["vs:str"].get("get") == b"plain")
        check(f"{label}: re-created set has its new dimension", after["vs:redim"].get("dim") == b"4",
              repr(after["vs:redim"].get("dim")))
        # the restored set is live, not a read-only image
        c.cmd("VADD", "vs:small", "VALUES", 3, 1, 1, 1, "post")
        check(f"{label}: restored set accepts writes",
              c.cmd("VISMEMBER", "vs:small", "post") == b"1")
    finally:
        try:
            c.close()
        except Exception:
            pass
        sigkill(p)
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    if port_open():
        print(f"port {PORT} is busy")
        return 1
    print(f"[gh #378] {BINARY}")
    run_mode("A snapshot", save=True, after_save=False, rewrite=False)
    run_mode("B WAL", save=False, after_save=False, rewrite=False)
    run_mode("C snapshot+WAL", save=True, after_save=True, rewrite=False)
    run_mode("D BGREWRITEAOF", save=False, after_save=False, rewrite=True)
    print(f"\n{'FAIL' if FAIL else 'PASS'}: {len(FAIL)} failure(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
