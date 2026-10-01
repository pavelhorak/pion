#!/usr/bin/env python3
"""A pipelined frame split across two reads must be answered exactly as if it
had arrived whole — at EVERY byte offset.

WHY
The fast path parses the recv buffer in place. When a frame is not complete
yet, an arm returns and the engine re-runs it once more bytes arrive. Anything
the arm did before noticing the frame was short is then done TWICE: MGET wrote
its `*N` array header first, so a split MGET sent a stray header, the client
read the NEXT replies as that array's elements, and every reply after it on
the connection was paired with the wrong request (found by
tests/test_long_key_leaks.py as a "hang" at one particular pipeline size, on
every binary back to the audit's base). No test split a frame: they all send
small pipelines that arrive in one read.

METHOD
One pipeline of every fast-path command shape (plus wrong-type and slow-path
neighbours), against short AND long (> 23 byte) keys. Reference replies come
from sending it whole. Then for each byte offset: FLUSHALL, send the first
part, pause so the server reads it alone, send the rest, and require the
same replies, byte for byte, with nothing extra behind them.

    python3 tests/test_split_frames.py [./pion-server] [--step N]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode, wait_ready  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BINARY = os.path.abspath(ARGS[0] if ARGS else os.environ.get("PION_BIN", "./pion-server"))
STEP = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 1
PORT = 6570
L = "long-key:" + "x" * 30          # 39 bytes: heap, not SSO


def commands():
    out = []
    for k in ("k", L):
        out += [
            ("SET", k + "s", "v1"), ("GET", k + "s"), ("SET", k + "s", "v" * 40),
            ("MSET", k + "m1", "a", k + "m2", "b"), ("MGET", k + "m1", k + "m2", k + "zz"),
            ("INCR", k + "n"), ("DECR", k + "n"), ("INCRBY", k + "n", "5"),
            ("HSET", k + "h", "f", "v"), ("HSET", k + "h", "f", "w", "g", "x"), ("HGET", k + "h", "g"),
            ("LPUSH", k + "l", "a"), ("RPUSH", k + "l", "b"), ("LRANGE", k + "l", "0", "-1"),
            ("LLEN", k + "l"), ("LPOP", k + "l"), ("RPOP", k + "l"),
            ("SADD", k + "S", "m"), ("SADD", k + "S", "n", "o"), ("SCARD", k + "S"),
            ("ZADD", k + "z", "1", "a"), ("ZADD", k + "z", "2", "b"), ("ZPOPMIN", k + "z"),
            ("EXISTS", k + "s", k + "zz"), ("SETBIT", k + "b", "7", "1"), ("GETBIT", k + "b", "7"),
            ("BITCOUNT", k + "b"), ("PFADD", k + "p", "a", "b"), ("PFADD", k + "p", "a"),
            ("PFCOUNT", k + "p"),
            # wrong type on arms that answer an error part-way through the frame
            ("PFADD", k + "s", "a", "b"), ("HSET", k + "s", "f", "v"), ("LPUSH", k + "s", "a"),
            ("SADD", k + "s", "a"), ("ZADD", k + "s", "1", "a"), ("INCR", k + "h"),
            ("DEL", k + "m1", k + "m2"), ("PING",), ("PING", "hello"), ("ECHO", "e"),
            ("MSET", k + "a", "1", k + "b", "2", k + "c", "3"), ("HSET", k + "s", "f", "v", "g", "w"),
            ("TYPE", k + "h"), ("SADD", k + "one", "x"), ("SPOP", k + "one", "1"), ("SCARD", k + "S"),
            # no-argument arms that must skip a surplus argument — only once it is all here
            ("DBSIZE", "x"), ("SELECT", "0"), ("CLIENT", "SETNAME", "n1"), ("CLIENT", "GETNAME"),
            ("COMMAND", "COUNT"), ("RESET", "x"),
        ]
    return out


def start(d):
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-crash-log",
                          "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "a"), stderr=subprocess.STDOUT)
    wait_ready(PORT, 30, proc=p)
    return p


def replies(c, n):
    return [c.read_raw() for _ in range(n)]


def main():
    cmds = commands()
    frames = [encode(x) for x in cmds]
    blob = b"".join(frames)
    starts, o = [], 0
    for f in frames:
        starts.append(o)
        o += len(f)
    d = tempfile.mkdtemp(prefix="pion_split_")
    p = start(d)
    fails = []
    try:
        c = Conn(PORT, timeout=5)
        c.cmd("FLUSHALL")
        c.sock.sendall(blob)
        ref = replies(c, len(cmds))
        c.close()
        print(f"[split] {BINARY}: {len(cmds)} commands, {len(blob)} bytes, every {STEP} byte(s)")
        failed_frames = set()
        for s in range(1, len(blob), STEP):
            if max(i for i in range(len(starts)) if starts[i] <= s) in failed_frames:
                continue      # one report per frame; keep looking at the others
            c = Conn(PORT, timeout=3)
            c.cmd("FLUSHALL")
            c.sock.sendall(blob[:s])
            time.sleep(0.002)
            c.sock.sendall(blob[s:])
            k = max(i for i in range(len(starts)) if starts[i] <= s)
            where = f"split at byte {s}, +{s - starts[k]} into {cmds[k][0]} {cmds[k][1] if len(cmds[k]) > 1 else ''}"
            try:
                got = replies(c, len(cmds))
                c.assert_in_sync()
            except Exception as e:  # noqa: BLE001 — any failure to frame is the finding
                fails.append(f"{where}: {type(e).__name__}: {e}")
                failed_frames.add(k)
                c.close()
                if p.poll() is not None:
                    fails.append("server died")
                    break
                continue
            c.close()
            bad = [i for i in range(len(cmds)) if got[i] != ref[i]]
            if bad:
                i = bad[0]
                fails.append(f"{where}: reply {i} ({cmds[i][0]}) {got[i][:60]!r} != {ref[i][:60]!r}")
                failed_frames.add(k)
    finally:
        p.kill()
        p.wait()
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print("  FAIL", f)
    print(f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
