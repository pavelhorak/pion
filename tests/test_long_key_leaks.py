#!/usr/bin/env python3
"""No command may leak memory on a key longer than 23 bytes (gh #394).

WHY
`GenericValue.from_ptr` / `from_string` copy a key over 23 bytes (the SSO
limit) to the heap, and a key built only to look something up was almost
never freed: ~48-64 B per call on EVERY command, plain fast-path GET included.
At gate rates that is ~100 MB/s of growth on any server whose keys look like
`user:1234:session:...`. Every existing test used short keys, so none of them
could see it. The fix borrows lookup keys instead of copying them, converted
by tools/audit_borrowed_keys.py across ~300 sites — so the check has to cover
the same surface: every command in the generated command table.

METHOD
Per command: a cycle creates one 48-byte key of every type (string, list,
hash, set, zset, stream, geo, hll, bitmap, plus a missing one), runs the
command against each in several argument shapes (0-2 plain arguments, a
second 48-byte key as destination/source), then DELs every key. After a
warm-up of the same cycle, RSS growth over CYCLES cycles must stay under
BOUND_KB. A 48-byte leak in any ONE shape is CYCLES * 48 B = ~280 KB; a
clean command measures 0, or a one-off allocator step of up to ~80 KB that
does NOT grow with the cycle count (checked: 2000 and 6000 cycles both read
80 KB) — the bound sits between the two.

A command that errors on every shape still has to look its key up first
(that is where the copy was made), so errors are fine and expected here —
this is a leak check, not a semantics check.

    python3 tests/test_long_key_leaks.py [./pion-server] [--only CMD ...]
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, wait_ready, wait_port_free  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BINARY = os.path.abspath(ARGS[0] if ARGS and "/" in ARGS[0] else os.environ.get("PION_BIN", "./pion-server"))
ONLY = [a.lower() for a in sys.argv[sys.argv.index("--only") + 1:]] if "--only" in sys.argv else None
PORT = 6560
CYCLES = 6000
WARMUP = 300
BOUND_KB = 160
TABLE = Path(__file__).resolve().parents[1] / "src" / "commands" / "command_table.mojo"


def key(tag):
    k = b"leakcheck:long-key:" + tag.encode() + b":"
    return k + b"x" * (48 - len(k))


K = {t: key(t) for t in ("str", "list", "hash", "set", "zset", "stream", "geo", "hll", "bitmap", "missing")}
DST = key("dst")
SETUP = [
    ("SET", K["str"], "12345"),
    ("RPUSH", K["list"], "a", "b", "c"),
    ("HSET", K["hash"], "f", "v", "g", "1"),
    ("SADD", K["set"], "a", "b", "c"),
    ("ZADD", K["zset"], "1", "a", "2", "b"),
    ("XADD", K["stream"], "*", "f", "v"),
    ("GEOADD", K["geo"], "13.361389", "38.115556", "a"),
    ("PFADD", K["hll"], "a", "b"),
    ("SETBIT", K["bitmap"], "7", "1"),
]
TEARDOWN = [("DEL", *K.values(), DST)]
SHAPES = [(), ("f",), ("f", "v"), ("0", "-1"), ("1",), (DST,), (DST, "a")]

# Not probed, each for a reason: they end the connection or the server, change
# connection mode, block, or grow server-global state by design (a stored
# script, a pub/sub subscription, a user, an index), which is not a leak.
SKIP = {
    "shutdown", "quit", "reset", "hello", "auth", "select", "swapdb", "monitor",
    "subscribe", "psubscribe", "ssubscribe", "unsubscribe", "punsubscribe", "sunsubscribe",
    "multi", "exec", "discard", "watch", "unwatch",
    "sync", "psync", "replconf", "replicaof", "slaveof", "failover", "migrate",
    "blpop", "brpop", "blmove", "blmpop", "bzmpop", "bzpopmin", "bzpopmax", "brpoplpush",
    "wait", "waitaof", "debug", "client",
    "flushall", "flushdb", "save", "bgsave", "bgrewriteaof",
    "config", "acl", "script", "function", "eval", "evalsha", "eval_ro", "evalsha_ro",
    "fcall", "fcall_ro", "publish", "spublish", "readonly", "readwrite", "asking",
}


def rss_kb(pid):
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip())


def commands():
    names = sorted(set(re.findall(r'_cmd_eq_ci\(tp, tl, "([^"]+)"\)', TABLE.read_text())))
    names = [n for n in names if n not in SKIP and not n.startswith(("ft.", "kv.", "attend", "ai.", "rag.",
                                                                     "v.", "neuron.", "moe.", "route"))]
    return [n for n in names if ONLY is None or n in ONLY]


def cycle(cmd):
    out = list(SETUP)
    for k in K.values():
        for shape in SHAPES:
            out.append((cmd, k, *shape))
    return out + TEARDOWN


def run(c, cmds):
    for i in range(0, len(cmds), 400):
        c.pipeline(cmds[i:i + 400])


def start(d):
    # --no-wal: the log grows with every write and would swamp the signal.
    proc = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-crash-log",
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=d, stdout=open(os.path.join(d, "log"), "a"), stderr=subprocess.STDOUT)
    try:
        wait_ready(PORT, 30, proc=proc)
    except RuntimeError:
        print("server failed to start; log tail:\n" + open(os.path.join(d, "log")).read()[-1500:])
        raise
    return proc


def ft_search_leak(d):
    """FT.* is not in the per-command sweep (it needs an index), and its reply
    path built a heap copy of every result key of 24-31 bytes — the ones the
    shared slot->key map holds — for a lookup and never freed it: ~22 B per
    RESULT, 2,144 KB over 10,000 searches of 10 results on the pre-fix binary."""
    proc = start(d)
    try:
        c = Conn(PORT, timeout=60)
        dim = 16
        c.cmd("FT.CREATE", "lk", "SCHEMA", "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32",
              "DIM", str(dim))
        vec = lambda i: struct.pack(f"{dim}f", *[((i * 7919 + j * 104729) % 1000) / 1000.0
                                                 for j in range(dim)])
        run(c, [("HSET", f"document:longkey:{i:08d}", "vec", vec(i), "id", str(i))
                for i in range(2000)])     # 25-byte keys: SSO is <= 23, the map holds <= 31
        c.cmd("FT.OPTIMIZE", "lk")
        q = ("FT.SEARCH", "lk", "*=>[KNN 10 @vec $B]", "PARAMS", "2", "B", vec(5), "DIALECT", "2")
        r = c.cmd(*q)
        if not (isinstance(r, list) and r and r[0] == 10 and len(r[1]) == 25):
            return f"FT.SEARCH: probe did not return 10 results on 25-byte keys: {r!r:.120}"
        run(c, [q] * 2000)
        base = rss_kb(proc.pid)
        run(c, [q] * 10000)
        grown = rss_kb(proc.pid) - base
        print(f"  {'PASS' if grown < BOUND_KB else 'FAIL'}  FT.SEARCH x 10,000 (10 results, 25-byte keys)"
              f"  grew {grown} KB")
        return None if grown < BOUND_KB else f"FT.SEARCH: grew {grown} KB over 10,000 searches"
    finally:
        proc.kill()
        proc.wait()
        wait_port_free(PORT)


def main():
    d = tempfile.mkdtemp(prefix="pion_longkey_")
    fails, clean = [], 0
    cmds = commands()
    print(f"[long-key] {BINARY}: {len(cmds)} commands x {len(K)} key types x {len(SHAPES)} shapes, "
          f"{CYCLES} cycles, bound {BOUND_KB} KB, fresh server each")
    try:
        for name in cmds:
            # A fresh server per command: memory an earlier command freed is
            # reused first and would absorb this one's leak.
            proc = start(d)
            try:
                c = Conn(PORT, timeout=60)
                one = cycle(name)
                try:
                    run(c, one * WARMUP)
                    base = rss_kb(proc.pid)
                    run(c, one * CYCLES)
                    grown = rss_kb(proc.pid) - base
                except (ConnectionError, TimeoutError, OSError) as e:
                    fails.append(f"{name}: server died or hung ({type(e).__name__})")
                    print(f"  FAIL  {name:24} server died or hung")
                    continue
                if grown >= BOUND_KB:
                    fails.append(f"{name}: grew {grown} KB ({grown * 1024 / CYCLES:.0f} B/cycle)")
                    print(f"  FAIL  {name:24} grew {grown:6d} KB  ({grown * 1024 / CYCLES:.0f} B/cycle)")
                else:
                    clean += 1
                if proc.poll() is not None:
                    fails.append(f"server died after: {name}")
                else:
                    c.assert_in_sync()
            finally:
                proc.kill()
                proc.wait()
                wait_port_free(PORT)
        if ONLY is None or "ft.search" in ONLY:
            f = ft_search_leak(d)
            if f:
                fails.append(f)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"{clean} clean, {len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
