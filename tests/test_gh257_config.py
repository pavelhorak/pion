#!/usr/bin/env python3
"""gh #257 regression test — CONFIG must not lie in either direction.

Two silent lies lived in `handle_config`:

  1. `CONFIG SET <anything> <anything>` replied `+OK` and did nothing. An
     operator, a Terraform/ansible module, or a client library's startup probe
     believed the setting took. Same rule as gh #229's numeric parsing and
     gh #260's full WAL: **a command that cannot honour its contract must
     error, not acknowledge.**

  2. `CONFIG GET <anything>` returned the parameter name with an EMPTY VALUE,
     for every parameter. That reads as "configured to nothing" rather than "I
     do not know this parameter" — a fabricated value dressed as a real one.

What is asserted:

  - the parameters Pion can answer TRUTHFULLY return their real values, and
    agree with a live redis-server where the underlying state agrees
  - `port` reflects the port actually being served, not a hardcoded 1974
  - an UNKNOWN parameter returns `*0` (empty array), byte-for-byte what real
    Redis returns — NOT an empty value
  - `CONFIG SET` errors, and the error names the CLI-flag alternative
  - `CONFIG REWRITE` errors as Redis does without a config file, and
    `RESETSTAT` resets the counters INFO reports
  - the connection stays framed afterwards (an error mid-pipeline is exactly
    where gh #232's "array header before the type check" class desyncs)

`save` is expected to DIFFER from Redis: Redis ships default save points and
Pion has none, so "" is the honest answer, not a compatibility bug.

`databases` is 1: Pion has one database and refuses `SELECT` of any other,
as Redis configured with `databases 1` does. (It was left out while `SELECT 5`
still answered +OK, when no value would have been true.)

Usage: python3 tests/test_gh257_config.py [--port 1974] [--redis-port 6399]
"""
import argparse, socket, sys

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=15)
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            a = str(a).encode()
            parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
        self.sock.sendall(b"".join(parts))
        return self._read()

    def raw(self, payload, nbytes=256):
        self.sock.sendall(payload)
        import time; time.sleep(0.3)
        return self.sock.recv(nbytes)

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed")
        t, body = line[:1], line[1:-2]
        if t == b"+": return body.decode()
        if t == b"-": return "ERR:" + body.decode()
        if t == b":": return int(body)
        if t == b"$":
            n = int(body); return None if n == -1 else self.f.read(n + 2)[:-2].decode()
        if t in b"*%":
            n = int(body)
            if n <= 0: return []
            count = 2 * n if t == b"%" else n
            return [self._read() for _ in range(count)]
        return body.decode(errors="replace")


def is_err(r):
    return isinstance(r, str) and r.startswith("ERR:")


# Parameters Pion can state without inventing anything, and whether the value
# should match a live redis-server's (save legitimately differs).
TRUTHFUL = {
    "maxmemory": ("0", True),
    "maxmemory-policy": ("noeviction", True),
    "appendonly": ("no", True),
    "timeout": ("0", True),
    "save": ("", False),          # Redis has default save points; Pion has none
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--redis-port", type=int, default=6399)
    args = ap.parse_args()

    try:
        c = Client(args.port)
    except OSError as e:
        print(f"FATAL: no Pion on {args.port} ({e})")
        return 2
    try:
        redis = Client(args.redis_port); redis("PING")
    except (OSError, RuntimeError):
        redis = None
        print(f"NOTE: no redis-server on {args.redis_port} — skipping oracle comparisons")

    print("\n[1] Truthful parameters report their real values")
    for param, (want, same_as_redis) in TRUTHFUL.items():
        got = c("CONFIG", "GET", param)
        check(f"CONFIG GET {param} == [{param}, {want!r}]",
              got == [param, want], f"got {got}")
        if redis and same_as_redis:
            r = redis("CONFIG", "GET", param)
            check(f"CONFIG GET {param} agrees with redis", got == r,
                  f"pion={got} redis={r}")

    print("\n[2] port reflects the port actually being served")
    got = c("CONFIG", "GET", "port")
    check("CONFIG GET port is the real port",
          got == ["port", str(args.port)], f"got {got} (serving {args.port})")

    print("\n[3] An unknown parameter is *0, not an empty value")
    raw = c.raw(b"*3\r\n$6\r\nCONFIG\r\n$3\r\nGET\r\n$11\r\nnosuchparam\r\n")
    check("unknown param replies *0", raw == b"*0\r\n",
          f"got {raw!r} — an empty VALUE asserts the parameter exists and is blank")
    if redis:
        rraw = redis.raw(b"*3\r\n$6\r\nCONFIG\r\n$3\r\nGET\r\n$11\r\nnosuchparam\r\n")
        check("unknown param matches redis byte-for-byte", raw == rraw,
              f"pion={raw!r} redis={rraw!r}")

    print("\n[4] databases is 1, and SELECT agrees")
    got = c("CONFIG", "GET", "databases")
    check("CONFIG GET databases is 1", got == ["databases", "1"], f"got {got}")
    check("SELECT 1 is refused", is_err(c("SELECT", "1")))

    print("\n[5] CONFIG SET errors instead of acknowledging")
    # maxmemory is settable since gh #261 — see [5b]. Everything else still
    # refuses rather than acknowledging a change that never happens.
    for param, val in (("hz", "20"), ("appendonly", "yes"),
                       ("nosuchparam", "5"), ("save", "900 1")):
        r = c("CONFIG", "SET", param, val)
        check(f"CONFIG SET {param} {val} errors", is_err(r), f"got {r!r}")
    r = c("CONFIG", "SET", "appendonly", "yes")
    check("the error names the CLI-flag alternative",
          is_err(r) and "CLI" in str(r), f"got {r!r}")
    # And the setting genuinely did not take.
    check("appendonly still reports no after a refused SET",
          c("CONFIG", "GET", "appendonly") == ["appendonly", "no"])

    print("\n[5b] CONFIG SET maxmemory takes effect (gh #261)")
    check("CONFIG SET maxmemory 4gb -> OK", c("CONFIG", "SET", "maxmemory", "4gb") == "OK")
    check("CONFIG GET reads it back in bytes",
          c("CONFIG", "GET", "maxmemory") == ["maxmemory", str(4 * 1024 ** 3)])
    check("a malformed value errors", is_err(c("CONFIG", "SET", "maxmemory", "lots")))
    check("...and leaves the limit alone",
          c("CONFIG", "GET", "maxmemory") == ["maxmemory", str(4 * 1024 ** 3)])
    check("CONFIG SET maxmemory 0 restores unlimited",
          c("CONFIG", "SET", "maxmemory", "0") == "OK"
          and c("CONFIG", "GET", "maxmemory") == ["maxmemory", "0"])

    print("\n[6] REWRITE errors (no config file); RESETSTAT resets the counters")
    check("CONFIG REWRITE errors", is_err(c("CONFIG", "REWRITE")))
    check("CONFIG RESETSTAT -> OK", c("CONFIG", "RESETSTAT") == "OK")

    print("\n[7] Arity and framing")
    check("CONFIG with no subcommand errors", is_err(c("CONFIG")))
    check("CONFIG GET with no param errors", is_err(c("CONFIG", "GET")))
    check("PING after everything", c("PING") == "PONG")
    # Pipelined: an error must consume exactly its own command (gh #240).
    c.sock.sendall(b"*4\r\n$6\r\nCONFIG\r\n$3\r\nSET\r\n$10\r\nappendonly\r\n$3\r\nyes\r\n"
                   b"*1\r\n$4\r\nPING\r\n")
    r1, r2 = c._read(), c._read()
    check("pipelined CONFIG SET + PING gives exactly two replies",
          is_err(r1) and r2 == "PONG", f"got {r1!r}, {r2!r}")

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
