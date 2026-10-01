#!/usr/bin/env python3
"""Hash field TTLs belong to their hash (gh #392).

WHY
A field's expiry was an entry in the GLOBAL TTL table under the string
`key + "::" + field`. That string is ambiguous, so two different fields shared
one TTL; the entry outlived DEL (a new hash under the same key inherited the
old field's deadline); it did not follow RENAME or COPY; it was never written
to the WAL or the snapshot, so every field TTL vanished on restart; and the
active-expiry sweep split EVERY key containing `::` as if it were a field, so
a plain key like `user::1` with a TTL was never expired (its TTL entry was
dropped and the key served forever).

Each case below is checked against what Redis 8.10 answers (probed live when
this was written; the replies are spelled out so the test needs no Redis).

    python3 tests/test_hash_field_ttl.py [./pion-server]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready, wait_port_free  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BINARY = os.path.abspath(ARGS[0] if ARGS else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6640
fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  — got {got!r}, want {want!r}"))
    if not ok:
        fails.append(name)


def start(d):
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-crash-log",
                          "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "a"), stderr=subprocess.STDOUT)
    wait_ready(PORT, 30, proc=p)
    return p, Conn(PORT, timeout=10)


def stop(p, hard=True):
    (p.kill if hard else p.terminate)()
    p.wait()
    wait_port_free(PORT)


def ttl_set(c, key, field):
    """1 if the field has a TTL, else HTTL's own code (-1 no TTL / -2 no field)."""
    t = c.cmd("HTTL", key, "FIELDS", "1", field)[0]
    return 1 if t > 0 else t


def main():
    d = tempfile.mkdtemp(prefix="pion_hfttl_")
    p, c = start(d)
    try:
        print("=== collisions: two different fields, two different TTLs ===")
        c.cmd("HSET", "a::b", "c", "v"); c.cmd("HSET", "a", "b::c", "v")
        check("HEXPIRE a::b c", c.cmd("HEXPIRE", "a::b", "100", "FIELDS", "1", "c"), [1])
        check("HTTL a b::c has NO TTL (it used to report a::b c's)", ttl_set(c, "a", "b::c"), -1)
        check("HPERSIST a b::c answers -1, not 1", c.cmd("HPERSIST", "a", "FIELDS", "1", "b::c"), [-1])
        check("... and a::b c still has its TTL", ttl_set(c, "a::b", "c"), 1)

        print("=== the TTL belongs to the hash ===")
        c.cmd("HSET", "h", "f", "v"); c.cmd("HEXPIRE", "h", "1000", "FIELDS", "1", "f")
        c.cmd("DEL", "h"); c.cmd("HSET", "h", "f", "v")
        check("DEL then a new hash: the field has no TTL", ttl_set(c, "h", "f"), -1)
        c.cmd("HEXPIRE", "h", "1000", "FIELDS", "1", "f")
        c.cmd("RENAME", "h", "h2")
        check("RENAME carries it", ttl_set(c, "h2", "f"), 1)
        c.cmd("COPY", "h2", "h3")
        check("COPY carries it", ttl_set(c, "h3", "f"), 1)
        c.cmd("HPERSIST", "h3", "FIELDS", "1", "f")
        check("... as the copy's own (HPERSIST on it leaves the original)", ttl_set(c, "h2", "f"), 1)

        print("=== what clears it and what keeps it (Redis 8.10) ===")
        c.cmd("HSET", "k", "a", "1", "b", "2", "c", "3")
        c.cmd("HEXPIRE", "k", "100", "FIELDS", "3", "a", "b", "c")
        c.cmd("HSET", "k", "a", "9")
        check("HSET over a field clears its TTL", ttl_set(c, "k", "a"), -1)
        c.cmd("HINCRBY", "k", "b", "5")
        check("HINCRBY keeps it", ttl_set(c, "k", "b"), 1)
        c.cmd("HSETNX", "k", "c", "x")
        check("HSETNX (no-op on an existing field) keeps it", ttl_set(c, "k", "c"), 1)
        check("NX on a field without a TTL", c.cmd("HEXPIRE", "k", "50", "NX", "FIELDS", "1", "a"), [1])
        check("XX on a field without a TTL", c.cmd("HEXPIRE", "k", "50", "XX", "FIELDS", "1", "zz"), [-2])
        check("a past deadline deletes the field (2)", c.cmd("HEXPIRE", "k", "0", "FIELDS", "1", "c"), [2])
        check("... it is gone", c.cmd("HEXISTS", "k", "c"), 0)
        r = c.cmd("HEXPIRE", "k", "-1", "FIELDS", "1", "a")
        check("a negative TTL is an error", isinstance(r, RespError) and "invalid expire time" in r, True)
        check("missing key: -2 per field", c.cmd("HTTL", "nokey", "FIELDS", "2", "a", "b"), [-2, -2])
        c.cmd("SET", "s", "x")
        r = c.cmd("HTTL", "s", "FIELDS", "1", "a")
        check("wrong type: WRONGTYPE", isinstance(r, RespError) and r.startswith("WRONGTYPE"), True)

        print("=== expiry: lazy on every read, and active ===")
        c.cmd("HSET", "e", "f", "v", "g", "w"); c.cmd("HPEXPIRE", "e", "50", "FIELDS", "1", "f")
        time.sleep(0.15)
        check("HGET of an expired field", c.cmd("HGET", "e", "f"), None)
        check("HLEN skips it", c.cmd("HLEN", "e"), 1)
        check("HGETALL skips it", c.cmd("HGETALL", "e"), [b"g", b"w"])
        c.cmd("HSET", "e1", "f", "v"); c.cmd("HPEXPIRE", "e1", "50", "FIELDS", "1", "f")
        time.sleep(0.15)
        check("the key goes with its last field", c.cmd("EXISTS", "e1"), 0)
        c.cmd("FLUSHALL")
        c.cmd("HSET", "act", "f", "v"); c.cmd("HPEXPIRE", "act", "50", "FIELDS", "1", "f")
        deadline = time.time() + 5
        while time.time() < deadline and c.cmd("DBSIZE") != 0:   # DBSIZE never touches the hash
            time.sleep(0.1)
        check("active expiry removes a hash nobody reads", c.cmd("DBSIZE"), 0)
        c.cmd("SET", "user::1", "v", "PX", "100")
        deadline = time.time() + 5
        while time.time() < deadline and c.cmd("DBSIZE") != 0:
            time.sleep(0.1)
        check("a plain key containing '::' still expires (the sweep split it)", c.cmd("DBSIZE"), 0)
        check("... and is not served afterwards", c.cmd("GET", "user::1"), None)

        print("=== durability (the WAL and the snapshot carry field TTLs) ===")
        c.cmd("FLUSHALL")
        c.cmd("HSET", "d", "f", "v", "g", "w"); c.cmd("HEXPIRE", "d", "1000", "FIELDS", "1", "f")
        stop(p); p, c = start(d)
        check("after SIGKILL (WAL replay)", (ttl_set(c, "d", "f"), ttl_set(c, "d", "g")), (1, -1))
        c.cmd("SAVE"); stop(p); p, c = start(d)
        check("after SAVE + SIGKILL (snapshot)", (ttl_set(c, "d", "f"), ttl_set(c, "d", "g")), (1, -1))
        c.cmd("HPERSIST", "d", "FIELDS", "1", "f"); c.cmd("HEXPIRE", "d", "1000", "FIELDS", "1", "g")
        stop(p); p, c = start(d)
        check("HPERSIST and a new HEXPIRE after the snapshot replay too",
              (ttl_set(c, "d", "f"), ttl_set(c, "d", "g")), (-1, 1))
        c.cmd("BGREWRITEAOF"); stop(p); p, c = start(d)
        check("after BGREWRITEAOF + SIGKILL", (ttl_set(c, "d", "f"), ttl_set(c, "d", "g")), (-1, 1))
        c.cmd("HPEXPIRE", "d", "100", "FIELDS", "1", "g")
        stop(p); time.sleep(0.3); p, c = start(d)
        check("a field that expired while the server was down is gone", c.cmd("HEXISTS", "d", "g"), 0)
    finally:
        stop(p)
        shutil.rmtree(d, ignore_errors=True)
    print(f"{len(fails)} failure(s)" + "".join(f"\n  {f}" for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
