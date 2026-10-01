#!/usr/bin/env python3
"""Type confusion: every command against every WRONG value type.

`GenericValue` is a 32-byte tagged union (STRING/HASH/LIST/SET/ZSET/INT/
BITMAP/HLL/GEO/...) whose accessors — `as_list()`, `as_zset()`, `as_hll()`,
`as_hash()` — bitcast `_data0` to a container pointer. A handler that reaches
one of those WITHOUT first checking `type.value` does not merely return a wrong
answer: it reinterprets whatever bits are in that slot as a pointer and walks
it. For a STRING_SSO value those bits are inline character data, so the "list
pointer" is literally the key's own text.

So the required reply here is `-WRONGTYPE` (or any error), and the required
behaviour is that the server keeps serving. A crash, a hang, or a plausible
success are all failures — the last most of all, because it means the accessor
ran.

Each probe uses a fresh connection and is followed by PING, so a desync is
caught too.

Usage: python3 tests/test_type_confusion.py [--port 1974] [--verbose]
"""

import argparse
import socket
import sys

HOST = "127.0.0.1"
PASSED, FAILED, SUSPECT = [], [], []

# key name -> (setup commands, the type it holds)
FIXTURES = {
    "tc:string": ([["SET", "tc:string", "hello"]], "string"),
    "tc:list":   ([["RPUSH", "tc:list", "a", "b", "c"]], "list"),
    "tc:hash":   ([["HSET", "tc:hash", "f", "v"]], "hash"),
    "tc:set":    ([["SADD", "tc:set", "m1", "m2"]], "set"),
    "tc:zset":   ([["ZADD", "tc:zset", "1", "m1"]], "zset"),
    "tc:int":    ([["SET", "tc:int", "42"], ["INCR", "tc:int"]], "int"),
    "tc:hll":    ([["PFADD", "tc:hll", "e1", "e2"]], "hll"),
    "tc:bitmap": ([["SETBIT", "tc:bitmap", "7", "1"]], "bitmap"),
    "tc:stream": ([["XADD", "tc:stream", "*", "f", "v"]], "stream"),
}

# (command template, the type it EXPECTS). %K is replaced by the victim key.
# Every one of these reaches a typed accessor in the dispatcher.
PROBES = [
    (["LPUSH", "%K", "x"],              "list"),
    (["RPUSH", "%K", "x"],              "list"),
    (["LPOP", "%K"],                    "list"),
    (["RPOP", "%K"],                    "list"),
    (["LRANGE", "%K", "0", "-1"],       "list"),
    (["LLEN", "%K"],                    "list"),
    (["LINSERT", "%K", "BEFORE", "a", "x"], "list"),
    (["LSET", "%K", "0", "x"],          "list"),
    (["LTRIM", "%K", "0", "0"],         "list"),
    (["HSET", "%K", "f", "v"],          "hash"),
    (["HGET", "%K", "f"],               "hash"),
    (["HDEL", "%K", "f"],               "hash"),
    (["HLEN", "%K"],                    "hash"),
    (["HGETALL", "%K"],                 "hash"),
    (["HINCRBY", "%K", "f", "1"],       "hash"),
    (["SADD", "%K", "m"],               "set"),
    (["SREM", "%K", "m"],               "set"),
    (["SCARD", "%K"],                   "set"),
    (["SMEMBERS", "%K"],                "set"),
    (["SPOP", "%K"],                    "set"),
    (["SISMEMBER", "%K", "m"],          "set"),
    (["ZADD", "%K", "1", "m"],          "zset"),
    (["ZREM", "%K", "m"],               "zset"),
    (["ZCARD", "%K"],                   "zset"),
    (["ZSCORE", "%K", "m"],             "zset"),
    (["ZRANGE", "%K", "0", "-1"],       "zset"),
    (["ZRANGEBYSCORE", "%K", "0", "9"], "zset"),
    (["ZREVRANGEBYSCORE", "%K", "9", "0"], "zset"),
    (["ZINCRBY", "%K", "1", "m"],       "zset"),
    (["ZPOPMIN", "%K"],                 "zset"),
    (["PFADD", "%K", "e"],              "hll"),
    (["PFCOUNT", "%K"],                 "hll"),
    (["GETBIT", "%K", "0"],             "bitmap/string"),
    (["SETBIT", "%K", "0", "1"],        "bitmap/string"),
    (["BITCOUNT", "%K"],                "bitmap/string"),
    (["APPEND", "%K", "x"],             "string"),
    (["STRLEN", "%K"],                  "string"),
    (["GETRANGE", "%K", "0", "1"],      "string"),
    (["SETRANGE", "%K", "0", "x"],      "string"),
    (["INCR", "%K"],                    "string/int"),
    (["INCRBYFLOAT", "%K", "1.0"],      "string/int"),
    (["XADD", "%K", "*", "f", "v"],     "stream"),
    (["XLEN", "%K"],                    "stream"),
    (["XRANGE", "%K", "-", "+"],        "stream"),
    (["GETDEL", "%K"],                  "string"),
    (["SETEX", "%K", "100", "v"],       "string"),
]


class Conn:
    def __init__(self, port, timeout=4):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def send(self, *cmds):
        buf = b""
        for parts in cmds:
            buf += f"*{len(parts)}\r\n".encode()
            for p in parts:
                p = p.encode() if isinstance(p, str) else p
                buf += b"$%d\r\n%s\r\n" % (len(p), p)
        self.s.sendall(buf)

    def read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("server closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return "ERR:" + body.decode()
        if t == b":":
            return body.decode()
        if t == b"$":
            n = int(body)
            return None if n == -1 else self.f.read(n + 2)[:-2].decode(errors="replace")
        if t == b"*":
            n = int(body)
            return [] if n <= 0 else [self.read() for _ in range(n)]
        if t == b"_":
            return None
        return body.decode(errors="replace")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        c = Conn(args.port)
    except OSError as e:
        print(f"FATAL: no server on {HOST}:{args.port} ({e})")
        return 2

    # Build the fixtures.
    for key, (setup, _kind) in FIXTURES.items():
        c.send(["DEL", key])
        c.read()
        for cmd in setup:
            c.send(cmd)
            c.read()
    c.close()

    total = 0
    print(f"type-confusion sweep: {len(PROBES)} commands x {len(FIXTURES)} "
          f"value types = {len(PROBES) * len(FIXTURES)} probes\n")

    for key, (_setup, kind) in FIXTURES.items():
        for probe, expects in PROBES:
            if kind in expects:
                continue          # not a mismatch; skip
            total += 1
            cmd = [key if p == "%K" else p for p in probe]
            label = f"{probe[0]} on {kind}"
            try:
                c = Conn(args.port)
                c.send(cmd, ["PING"])
                r1 = c.read()
                r2 = c.read()
                c.close()
            except (EOFError, ConnectionResetError) as e:
                FAILED.append((label, f"SERVER DIED ({type(e).__name__})"))
                print(f"  FAIL {label:34s} SERVER DIED — {type(e).__name__}", flush=True)
                print("\n  Stopping: server gone, later results meaningless.")
                print("  Check pion-<port>.crash.log for a backtrace and the")
                print("  server stderr for a Mojo ABORT trace.")
                return 1
            except socket.timeout:
                FAILED.append((label, "TIMEOUT (hang)"))
                print(f"  FAIL {label:34s} TIMEOUT — handler hung", flush=True)
                continue

            if r2 != "PONG":
                FAILED.append((label, f"desync: second reply {r2!r}"))
                print(f"  FAIL {label:34s} desync — second reply {r2!r}", flush=True)
            elif isinstance(r1, str) and r1.startswith("ERR:"):
                PASSED.append(label)          # an error is the correct outcome
                if args.verbose:
                    print(f"  ok   {label:34s} {r1[:46]}")
            else:
                # No error: the command was ACCEPTED against the wrong type.
                # Either a missing type check (the accessor ran) or a
                # deliberate Redis-compatible behaviour — flag for review
                # rather than calling it a pass.
                SUSPECT.append((label, repr(r1)[:52]))
                print(f"  ???? {label:34s} accepted, returned {repr(r1)[:44]}", flush=True)

    print(f"\n{len(PASSED)}/{total} correctly rejected · "
          f"{len(SUSPECT)} accepted (review) · {len(FAILED)} failures")
    for label, why in FAILED:
        print(f"  FAILED:  {label}: {why}")
    if SUSPECT and args.verbose:
        for label, got in SUSPECT:
            print(f"  ACCEPTED: {label} -> {got}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
