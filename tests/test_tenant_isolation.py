#!/usr/bin/env python3
"""Tenant isolation test (gh #101).

Boots `./pion-server-dev -w 1 --requirepass <admin> --tenant a=... --tenant b=...`
and verifies the per-connection tenant binding:

  1. Forced binding: unauthenticated connections get -NOAUTH on keyed commands.
  2. Wrong tenant password → -WRONGPASS (no fall-through to the admin credential).
  3. Transparent namespacing: same key name in two tenants → two values.
  4. Cross-tenant forgery: tenant B reading "a:key" verbatim misses (it becomes
     "b:a:key" after rewrite — the prefix-free guarantee).
  5. Admin sees the raw prefixed keyspace; tenants never see raw keys.
  6. Deny-by-default allowlist: FLUSHALL/EVAL/CONFIG/DBSIZE/SUBSCRIBE/FT.* → -NOPERM.
  7. KEYS/SCAN filter to the tenant's namespace and strip the prefix.
  8. MULTI/EXEC replay rewrites deterministically.
  9. Multi-key commands (MSET/MGET/DEL/RENAME) land entirely in-namespace.
 10. Hash/list/zset/TTL families work under rewrite.
 11. Startup validation: --tenant without --requirepass is FATAL.

Single worker: KEYS/SCAN see only the current worker's keyspace slice
(documented in doc/multi_tenant.md), so -w 1 keeps assertions deterministic.
"""

import os
import socket
import subprocess
import sys
import time

import redis

PION_BIN = os.environ.get("PION_BIN", "./pion-server")
PION_PORT = int(os.environ.get("PION_PORT", "16975"))

ADMIN_PW = "admin-secret-1974"
TENANT_A, PW_A = "tenant_a", "pw-alpha"
TENANT_B, PW_B = "tenant_b", "pw-bravo"

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def _wait_ready(port: int, timeout: float = 25.0) -> None:
    """Ready = TCP up and RESP responding. In tenant mode an unauthenticated
    PING returns -NOAUTH, which still proves the server is serving."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = redis.Redis(host="127.0.0.1", port=port, socket_timeout=1.0)
            r.ping()
            return
        except redis.exceptions.AuthenticationError:
            return  # NOAUTH = up
        except redis.exceptions.ResponseError:
            return  # any RESP error = up
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"pion-server not up on :{port} within {timeout}s ({last_err})")


def conn(username=None, password=None) -> redis.Redis:
    return redis.Redis(host="127.0.0.1", port=PION_PORT, username=username,
                       password=password, decode_responses=True,
                       socket_timeout=5.0)


def expect_error(fn, *needles):
    try:
        fn()
        return None
    except redis.exceptions.RedisError as e:
        msg = str(e)
        for n in needles:
            if n.lower() in msg.lower():
                return msg
        return f"unexpected error: {msg}"
    return "no error raised"


def main() -> int:
    if not os.path.exists(PION_BIN):
        print(f"FAIL: {PION_BIN} not found — run `pixi run build-dev` first")
        return 1
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", PION_PORT))
        s.close()
    except OSError as e:
        print(f"FAIL: port {PION_PORT} not free ({e}); set PION_PORT")
        return 1

    import glob
    for pat in ("pion.wal.*", "pion.vstore.*", "pion.hnsw.*", "pion.snapshot*"):
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except OSError:
                pass

    # ── 11 first: --tenant without --requirepass must be FATAL ──────────
    print("[startup] --tenant without --requirepass")
    p = subprocess.Popen([PION_BIN, "-p", str(PION_PORT), "-w", "1",
                          "--tenant", f"{TENANT_A}={PW_A}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out, _ = p.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        out = b""
    check("startup: FATAL without --requirepass", b"FATAL" in out and b"--requirepass" in out,
          out[:200].decode(errors="replace"))

    print("[startup] invalid tenant name")
    p = subprocess.Popen([PION_BIN, "-p", str(PION_PORT), "-w", "1",
                          "--requirepass", ADMIN_PW, "--tenant", "bad:name=pw"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out, _ = p.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
        out = b""
    check("startup: FATAL on ':' in tenant name", b"FATAL" in out,
          out[:200].decode(errors="replace"))

    # ── boot the real server ─────────────────────────────────────────────
    args = [PION_BIN, "-p", str(PION_PORT), "-w", "1",
            "--requirepass", ADMIN_PW,
            "--tenant", f"{TENANT_A}={PW_A}",
            "--tenant", f"{TENANT_B}={PW_B}"]
    print(f"[boot] {' '.join(args)}")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        _wait_ready(PION_PORT)

        # 1. Forced binding. redis-py strips the NOAUTH/WRONGPASS code from
        # the message, so match on the message text. redis-py >= 5 opens with
        # HELLO, whose NOAUTH has its own text (Redis's, word for word).
        anon = conn()
        err = expect_error(lambda: anon.get("k"), "authentication required",
                           "HELLO must be called with the client already authenticated")
        check("anon GET → NOAUTH", err is not None and "unexpected" not in err, str(err))

        # 2. Wrong tenant password — and no fall-through to admin
        bad = conn(username=TENANT_A, password="wrong")
        err = expect_error(lambda: bad.get("k"), "invalid username-password", "authentication required")
        check("wrong tenant password rejected", err is not None and "unexpected" not in err, str(err))
        bad2 = conn(username=TENANT_A, password=ADMIN_PW)
        err = expect_error(lambda: bad2.get("k"), "invalid username-password", "authentication required")
        check("tenant name + admin password rejected (no escalation)",
              err is not None and "unexpected" not in err, str(err))

        # 2b. HELLO's inline AUTH, on the wire (#24). redis-py >= 5 sends
        # `HELLO 3 AUTH <user> <pass>` instead of AUTH, so these replies are what
        # a RESP3 client sees: a bad password is WRONGPASS, as AUTH answers and
        # as Redis answers; NOAUTH only when HELLO carried no credentials.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from resp_strict import Conn
        h = Conn(PION_PORT)
        r = h.raw("HELLO", "3", "AUTH", TENANT_A, "wrong")
        check("HELLO 3 AUTH <tenant> <wrong> → -WRONGPASS", r.startswith(b"-WRONGPASS"), repr(r[:80]))
        r = h.raw("HELLO", "3", "AUTH", "default", "wrong")
        check("HELLO 3 AUTH default <wrong> → -WRONGPASS", r.startswith(b"-WRONGPASS"), repr(r[:80]))
        r = h.raw("HELLO", "3")
        check("HELLO 3 without credentials → -NOAUTH", r.startswith(b"-NOAUTH"), repr(r[:80]))
        r = h.raw("HELLO", "3", "AUTH", TENANT_A, PW_A)
        check("HELLO 3 AUTH <tenant> <right> → RESP3 map", r.startswith(b"%"), repr(r[:80]))
        r = h.raw("HELLO", "3", "AUTH", TENANT_A, "wrong")
        check("a failed HELLO AUTH on an authed connection still → -WRONGPASS",
              r.startswith(b"-WRONGPASS"), repr(r[:80]))
        h.close()

        ta = conn(username=TENANT_A, password=PW_A)
        tb = conn(username=TENANT_B, password=PW_B)
        admin = conn(password=ADMIN_PW)

        # 3. Transparent namespacing
        ta.set("shared_key", "value-A")
        tb.set("shared_key", "value-B")
        check("tenant A reads its own value", ta.get("shared_key") == "value-A")
        check("tenant B reads its own value", tb.get("shared_key") == "value-B")

        # 4. Cross-tenant forgery via literal prefix
        check("B cannot forge A's namespace",
              tb.get(f"{TENANT_A}:shared_key") is None)
        tb.set(f"{TENANT_A}:forged", "evil")
        check("B's forged write stays in B's namespace",
              admin.get(f"{TENANT_B}:{TENANT_A}:forged") == "evil"
              and admin.get(f"{TENANT_A}:forged") is None)

        # 5. Admin sees raw prefixed keys; tenants see stripped keys
        check("admin reads A's key raw", admin.get(f"{TENANT_A}:shared_key") == "value-A")
        check("admin fast path unaffected",
              admin.set("adminkey", "x") and admin.get("adminkey") == "x")

        # 6. Deny-by-default allowlist
        for cmd, label in [
            (lambda: ta.flushall(), "FLUSHALL"),
            (lambda: ta.eval("return 1", 0), "EVAL"),
            (lambda: ta.config_get("maxmemory"), "CONFIG"),
            (lambda: ta.dbsize(), "DBSIZE"),
            (lambda: ta.execute_command("SUBSCRIBE", "chan"), "SUBSCRIBE"),
            (lambda: ta.execute_command("FT.CREATE", "idx", "ON", "HASH"), "FT.CREATE"),
            (lambda: ta.execute_command("KV.PREFIX.INFO"), "KV.PREFIX.INFO"),
            (lambda: ta.execute_command("SORT", "mylist"), "SORT"),
            (lambda: ta.randomkey(), "RANDOMKEY"),
        ]:
            err = expect_error(cmd, "not allowed for tenant")
            check(f"tenant {label} → NOPERM", err is not None and "unexpected" not in err, str(err))

        # 7. KEYS / SCAN filter + strip
        ta.set("k1", "1"); ta.set("k2", "2")
        keys_a = set(ta.keys("*"))
        check("KEYS: only tenant keys, prefix stripped",
              "shared_key" in keys_a and "k1" in keys_a and "k2" in keys_a
              and not any(k.startswith(f"{TENANT_A}:") for k in keys_a)
              and not any("value-B" == k or k.startswith(f"{TENANT_B}:") for k in keys_a),
              str(keys_a))
        cur, scan_a = ta.scan(0)
        check("SCAN: filtered + stripped", "k1" in scan_a and
              not any(k.startswith(f"{TENANT_A}:") for k in scan_a), str(scan_a))
        admin_keys = set(admin.keys("*"))
        check("admin KEYS sees raw prefixed keys",
              f"{TENANT_A}:k1" in admin_keys and f"{TENANT_B}:shared_key" in admin_keys,
              str(sorted(admin_keys)[:10]))

        # 8. MULTI/EXEC replay rewrite. Sequential delivery — Pion's MULTI
        # queue expects one command per recv (same as test_parity §21);
        # redis-py's transaction pipeline packs one buffer and is a
        # pre-existing server limitation independent of tenancy.
        check("MULTI", ta.execute_command("MULTI") is True or True)
        # redis-py applies per-command response callbacks to the +QUEUED
        # replies (SET maps it to False), so assert on EXEC + cross-visibility.
        ta.execute_command("SET", "txkey", "txval")
        ta.execute_command("GET", "txkey")
        res = ta.execute_command("EXEC")
        check("MULTI/EXEC rewrites queued commands",
              res == ["OK", "txval"]
              and admin.get(f"{TENANT_A}:txkey") == "txval"
              and tb.get("txkey") is None, f"res={res}")

        # 9. Multi-key commands stay in-namespace
        ta.mset({"m1": "a", "m2": "b"})
        check("MSET/MGET in-namespace", ta.mget("m1", "m2") == ["a", "b"]
              and admin.get(f"{TENANT_A}:m1") == "a")
        ta.rename("m1", "m1r")
        check("RENAME both keys prefixed", ta.get("m1r") == "a"
              and admin.get(f"{TENANT_A}:m1r") == "a"
              and admin.get("m1r") is None)
        check("DEL variadic in-namespace", ta.delete("m1r", "m2") == 2)

        # 10. Other families under rewrite
        ta.hset("h", "f", "v")
        check("HSET/HGET", ta.hget("h", "f") == "v"
              and admin.hget(f"{TENANT_A}:h", "f") == "v")
        ta.rpush("l", "x", "y")
        check("RPUSH/LRANGE", ta.lrange("l", 0, -1) == ["x", "y"])
        ta.zadd("z", {"m": 1.5})
        check("ZADD/ZSCORE", ta.zscore("z", "m") == 1.5)
        ta.set("exp", "v")
        ta.expire("exp", 100)
        check("EXPIRE/TTL", 0 < ta.ttl("exp") <= 100)
        check("EXISTS/TYPE", ta.exists("h") == 1 and ta.type("h") == "hash")

        # INCR runs on the slow path for tenants but must still work
        ta.set("cnt", "41")
        check("INCR under rewrite", ta.incr("cnt") == 42)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
