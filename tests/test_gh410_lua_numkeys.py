#!/usr/bin/env python3
"""gh #410: EVAL/FCALL must validate numkeys, and a script must never abort
the process.

Two families were fatal on the default server, unauthenticated, one packet:

  1. `EVAL "return 1" 1000000` — `_set_keys_argv` sized a numkeys-count array
     before checking numkeys against the frame; the alloc failed and Mojo
     aborted the whole process. `FCALL <fn> 1000000` shared the path.

  2. A script that hit the 1 MB Lua memory cap (or returned a table whose
     metamethod errors) left the failure to surface OUTSIDE any protected
     call — copying KEYS/ARGV in, or serializing the reply — where a failed
     allocation or a raised error is a Lua panic, not a catchable error, and
     ends the process. A memory-heavy script followed by a deep-recursion
     script on the SAME connection reproduced it.

The check: the boundary is accepted (numkeys == #args is legal), the hostile
values get Redis's error text, and after every route a FRESH connection still
answers — the worker is alive.

Usage: python3 tests/test_gh410_lua_numkeys.py [./pion-server]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6410

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


def main():
    import redis
    d = tempfile.mkdtemp(prefix="pion_gh410_")
    p = subprocess.Popen(
        # A 1 MB Lua heap (the old fixed cap; --lua-memory-limit since #36), so
        # the memory-cap routes below are reachable with small scripts.
        [BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-auto-detect", "--no-auto-embed",
         "--lua-memory-limit", "1mb"],
        cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)

    def fresh():
        """A brand-new connection PINGs — the real liveness invariant."""
        try:
            c = redis.Redis(port=PORT, socket_timeout=4)
            return c.ping() is True
        except Exception:
            return False

    def err_of(fn):
        try:
            return ("ok", fn())
        except redis.ResponseError as e:
            return ("err", str(e))
        except Exception as e:
            return ("exc", repr(e))

    try:
        r = None
        for _ in range(200):
            try:
                r = redis.Redis(port=PORT, socket_timeout=8)
                r.ping()
                break
            except Exception:
                time.sleep(0.1)
        if r is None:
            print("FATAL: server did not start")
            return 2

        # ── 1. numkeys validation ──────────────────────────────────────────
        # Redis: too-large numkeys is an error, negative is an error, and the
        # process must survive both. execute_command bypasses redis-py's own
        # numkeys bookkeeping so the raw value reaches the server.
        for label, val in [("1000000", 1000000), ("int64 max", 9223372036854775807)]:
            k, v = err_of(lambda: r.execute_command("EVAL", "return 1", val))
            ck(f"EVAL numkeys {label} -> error", k == "err" and "number of" in v.lower(), v[:48])
            ck(f"  server alive after EVAL numkeys {label}", fresh())

        # Redis looks the function up before it reads numkeys (#36), so the
        # numkeys guard needs a function that exists.
        r.execute_command("FUNCTION", "LOAD", "REPLACE",
                          "#!lua name=gh410\nredis.register_function('gh410f', function() return 1 end)")
        k, v = err_of(lambda: r.execute_command("FCALL", "gh410f", 1000000))
        ck("FCALL numkeys 1e6 -> error", k == "err" and "number of" in v.lower(), v[:48])
        ck("  server alive after FCALL numkeys 1e6", fresh())

        k, v = err_of(lambda: r.execute_command("EVAL", "return #ARGV", -1000, "a", "b"))
        ck("EVAL numkeys -1000 -> negative error", k == "err" and "negative" in v.lower(), v[:48])
        ck("  server alive after negative numkeys", fresh())

        # ── boundary: numkeys == #args is LEGAL, and one-past is not ────────
        k, v = err_of(lambda: r.execute_command("EVAL", "return #KEYS", 2, "a", "b"))
        ck("EVAL numkeys == #args accepted", k == "ok" and v == 2, repr(v))
        k, v = err_of(lambda: r.execute_command("EVAL", "return #KEYS", 3, "a", "b"))
        ck("EVAL numkeys == #args+1 rejected", k == "err" and "number of" in v.lower(), v[:48])
        ck("  server alive after boundary probes", fresh())

        # ── 2. panic-outside-protected-call routes ─────────────────────────
        # A returned table whose metamethod errors: serialized outside any
        # protected call. Must not abort; returns an (empty) reply.
        k, v = err_of(lambda: r.execute_command(
            "EVAL", "return setmetatable({},{__index=function() error(1) end})", 0))
        ck("EVAL erroring __index return survives", k in ("ok", "err"), f"{k}:{str(v)[:32]}")
        ck("  server alive after __index return", fresh())

        # Memory-cap hit, THEN deep recursion on the SAME connection: the cap
        # must be enforceable again (garbage from the first script is collected)
        # and neither script may abort the worker.
        for script in [
            "local s = string.rep('x', 2*1024*1024); return #s",
            "local function f(n) return f(n+1)+1 end; return f(0)",
        ]:
            k, v = err_of(lambda s=script: r.execute_command("EVAL", s, 0))
            ck(f"EVAL mem/recursion survives ({script[:24]}…)",
               k == "err" and "memory" in v.lower(), v[:40])
        ck("  same connection still works after mem+recursion",
           err_of(lambda: r.execute_command("EVAL", "return 42", 0)) == ("ok", 42))
        ck("  server alive after mem+recursion sequence", fresh())

    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
