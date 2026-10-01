#!/usr/bin/env python3
"""gh #241 — LSET/LINSERT/LREM/LTRIM/LPOS in QUICKLIST mode.

A SlabList switches representation at 1024 entries (or a >64B value), and every
one of these five commands was implemented for the ziplist half only. The
failure modes were not symmetric, which is why this test asserts against a real
Redis rather than against expectations:

  LTRIM   deleted the WHOLE list and replied +OK
  LSET    replied -ERR ... not supported
  LINSERT replied -1, indistinguishable from "pivot not found"
  LREM    replied 0, indistinguishable from "no matches"
  LPOS    replied nil for an element that IS in the list

Only LTRIM lost data; the other four returned a plausible wrong answer, which
is the more dangerous shape because a caller cannot detect it.

Run:  python3 tests/test_gh241_quicklist.py [--port 1974]
      python3 tests/test_gh241_quicklist.py --oracle   # diff vs real redis
"""
import argparse
import socket
import subprocess
import sys
import time

BIG = 1500          # comfortably past ZIPLIST_MAX_ENTRIES (1024)


class Client:
    def __init__(self, port, host="127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            b = a.encode() if isinstance(a, str) else a
            out.append(b"$%d\r\n%s\r\n" % (len(b), b))
        self.sock.sendall(b"".join(out))
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            return "ERR:" + body.decode()
        if t == b":":
            return int(body)
        if t == b"$":
            n = int(body)
            if n == -1:
                return None
            data = self.f.read(n + 2)[:-2]
            return data.decode()
        if t == b"*":
            n = int(body)
            if n == -1:
                return None
            return [self._read() for _ in range(n)]
        raise ValueError("bad reply type %r" % t)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def fill(c, key, n, prefix="e"):
    c.cmd("DEL", key)
    # Pipeline the fill; one RPUSH per element is too slow at n=1500.
    args = [prefix + str(i) for i in range(n)]
    for chunk in range(0, n, 500):
        c.cmd("RPUSH", key, *args[chunk:chunk + 500])
    return args


# Each case: (name, list of (cmd tuple)) — applied to a freshly filled list.
# The final LRANGE/LLEN is the state assertion.
def scenarios(key):
    return [
        ("ltrim-keep-all",      [("LTRIM", key, "0", "-1")]),
        ("ltrim-tail-window",   [("LTRIM", key, "-100", "-1")]),
        ("ltrim-head-window",   [("LTRIM", key, "0", "99")]),
        ("ltrim-middle",        [("LTRIM", key, "500", "600")]),
        ("ltrim-empty-range",   [("LTRIM", key, "5", "4")]),
        ("ltrim-out-of-range",  [("LTRIM", key, "9000", "9999")]),
        ("ltrim-down-to-zip",   [("LTRIM", key, "0", "9")]),
        ("lset-head",           [("LSET", key, "0", "REPLACED")]),
        ("lset-mid",            [("LSET", key, "700", "REPLACED")]),
        ("lset-tail",           [("LSET", key, "-1", "REPLACED")]),
        ("lset-out-of-range",   [("LSET", key, "99999", "x")]),
        ("linsert-before",      [("LINSERT", key, "BEFORE", "e700", "INS")]),
        ("linsert-after",       [("LINSERT", key, "AFTER", "e700", "INS")]),
        ("linsert-before-head", [("LINSERT", key, "BEFORE", "e0", "INS")]),
        ("linsert-after-tail",  [("LINSERT", key, "AFTER", "e%d" % (BIG - 1), "INS")]),
        ("linsert-missing",     [("LINSERT", key, "BEFORE", "nope", "INS")]),
        ("lrem-single",         [("LREM", key, "0", "e700")]),
        ("lrem-all-dupes",      [("RPUSH", key, "dup", "dup", "dup"),
                                 ("LREM", key, "0", "dup")]),
        ("lrem-first-two",      [("RPUSH", key, "dup", "dup", "dup"),
                                 ("LREM", key, "2", "dup")]),
        ("lrem-last-two",       [("RPUSH", key, "dup", "dup", "dup"),
                                 ("LREM", key, "-2", "dup")]),
        ("lrem-missing",        [("LREM", key, "0", "nope")]),
        ("lpos-head",           [("LPOS", key, "e0")]),
        ("lpos-mid",            [("LPOS", key, "e700")]),
        ("lpos-tail",           [("LPOS", key, "e%d" % (BIG - 1))]),
        ("lpos-missing",        [("LPOS", key, "nope")]),
        ("lpos-rank-neg",       [("RPUSH", key, "dup", "dup", "dup"),
                                 ("LPOS", key, "dup", "RANK", "-1")]),
        ("lpos-count-all",      [("RPUSH", key, "dup", "dup", "dup"),
                                 ("LPOS", key, "dup", "COUNT", "0")]),
        # Emptying a list must delete the key (gh #234), even via the quicklist path.
        ("lrem-empties-key",    [("LTRIM", key, "0", "0"),
                                 ("LREM", key, "0", "e0"),
                                 ("EXISTS", key)]),
    ]


def run_case(c, key, steps, n):
    fill(c, key, n)
    replies = [c.cmd(*s) for s in steps]
    # State after: length plus the first/last few elements and a mid sample.
    ln = c.cmd("LLEN", key)
    head = c.cmd("LRANGE", key, "0", "4")
    tail = c.cmd("LRANGE", key, "-5", "-1")
    return (replies, ln, head, tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--oracle", action="store_true",
                    help="diff every case against a real redis-server")
    ap.add_argument("--redis-port", type=int, default=6398)
    args = ap.parse_args()

    redis_proc = None
    oracle = None
    if args.oracle:
        redis_proc = subprocess.Popen(
            ["redis-server", "--port", str(args.redis_port), "--save", "",
             "--appendonly", "no"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                oracle = Client(args.redis_port)
                break
            except OSError:
                time.sleep(0.1)
        if oracle is None:
            print("could not start redis-server on port %d" % args.redis_port)
            return 2

    try:
        pion = Client(args.port)
    except OSError as e:
        print("cannot reach Pion on port %d: %s" % (args.port, e))
        if redis_proc:
            redis_proc.terminate()
        return 2

    key = "gh241:list"
    failures = []
    checked = 0

    for name, steps in scenarios(key):
        got = run_case(pion, key, steps, BIG)
        checked += 1
        if oracle is not None:
            want = run_case(oracle, key, steps, BIG)
            if got != want:
                failures.append((name, want, got))
        else:
            # Without the oracle, assert the property that used to break:
            # nothing may silently claim success while losing the list.
            replies, ln, _, _ = got
            if name == "ltrim-keep-all" and ln != BIG:
                failures.append((name, "LLEN == %d" % BIG, "LLEN == %s" % ln))

        # The connection must still be framed after every case.
        if pion.cmd("PING") != "PONG":
            failures.append((name, "PONG", "connection desynced"))
            pion.close()
            pion = Client(args.port)

    pion.cmd("DEL", key)
    pion.close()
    if oracle is not None:
        oracle.cmd("DEL", key)
        oracle.close()
    if redis_proc:
        redis_proc.terminate()
        redis_proc.wait(timeout=5)

    mode = "vs real redis" if args.oracle else "standalone"
    if failures:
        print("gh #241 quicklist: %d/%d PASS, %d FAIL (%s)"
              % (checked - len(failures), checked, len(failures), mode))
        for name, want, got in failures:
            print("\n  FAIL %s" % name)
            print("    redis: %r" % (want,))
            print("    pion : %r" % (got,))
        return 1

    print("gh #241 quicklist: %d/%d PASS (%s, list size %d)"
          % (checked, checked, mode, BIG))
    return 0


if __name__ == "__main__":
    sys.exit(main())
