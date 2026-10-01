#!/usr/bin/env python3
"""Lua `redis.call('RENAME', …)` must hand the new key its own string payload.

Found 2026-09-27 (alongside gh #369): the Lua RENAME stored the old key's
value under the new key and then `remove_generic`-ed the old key — which frees
a heap STRING payload (> 23 bytes). The new key was left pointing at freed
memory; later allocations reuse it and `GET newkey` returns someone else's
bytes. key_mgmt.mojo's RENAME already cloned (gh #123); the Lua copy did not.

The check forces reuse: rename, then allocate many same-sized values, then
read the renamed key back.

Usage: python3 tests/test_lua_rename_payload.py [./pion-server]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6476


def main():
    d = tempfile.mkdtemp(prefix="pion_luaren_")
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
    fails = 0
    try:
        import redis
        r = None
        for _ in range(200):
            try:
                r = redis.Redis(port=PORT)
                r.ping()
                break
            except Exception:
                time.sleep(0.1)
        assert r is not None
        value = "payload-" + "x" * 90          # heap STRING (> 23 bytes)
        script = "redis.call('SET', KEYS[1], ARGV[1]); return redis.call('RENAME', KEYS[1], KEYS[2])"
        for trial in range(20):
            a, b = f"lr:a:{trial}", f"lr:b:{trial}"
            r.eval(script, 2, a, b, value)
            pipe = r.pipeline(transaction=False)
            for j in range(200):                # reuse the freed block if it was freed
                pipe.set(f"lr:fill:{trial}:{j}", "F" * 98)
            pipe.execute()
            got = r.get(b)
            ok = got == value.encode() and r.exists(a) == 0
            if not ok:
                fails += 1
                print(f"  FAIL  trial {trial}: GET {b} = {got!r:.60}")
        print(f"  {'PASS' if fails == 0 else 'FAIL'}  Lua RENAME keeps the value intact (20 trials)")
        alive = p.poll() is None
        print(f"  {'PASS' if alive else 'FAIL'}  server alive")
        fails += 0 if alive else 1
    finally:
        p.kill()
        p.wait()
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
