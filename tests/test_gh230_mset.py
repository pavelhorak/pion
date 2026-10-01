#!/usr/bin/env python3
"""gh #230 regression test — MSET must be a SET.

The MSET fast-path arm wrote the value into the keyspace and stopped there.
SET, three lines away, also clears the key's TTL and bumps its WATCH version.
MSET did neither, and both gaps are silent:

  - `SET k v EX 100; MSET k v2` left the old deadline in place, so the NEW
    value expired on the OLD value's clock. Nothing reports this; the key is
    simply gone later.
  - `WATCH k` did not observe an `MSET k ...`, so EXEC ran a transaction whose
    precondition had already been overwritten — the lost update WATCH exists to
    prevent. Verified against a real redis-server, which aborts.

The perf work in the same commit is what makes the correctness work affordable
(the version slot now falls out of the hash the store already computes, instead
of a byte-at-a-time loop), and it rewrote MSET's WAL append into a batched form:
one capacity check and one accounting for the whole command. So this file also
pins durability — a batching bug would drop keys only on a crash, which no
throughput number would ever show.

Usage: python3 tests/test_gh230_mset.py [./pion-server-dev]
"""
import os, socket, subprocess, sys, time, shutil, signal

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1985
WORKDIR = f"/tmp/pion_gh230_test_{PORT}"

failures = []
passes = []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


class Client:
    """Framing RESP client — one reply per command, byte-counted."""

    def __init__(self, sock):
        self.sock = sock
        self.f = sock.makefile("rb")

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        t = line[:1]
        if t in b"+-:":
            return line[:-2]
        if t == b"$":
            n = int(line[1:])
            # $-1 and *-1 both mean "nil" here. Pion answers an aborted EXEC
            # with $-1 where Redis answers *-1; a reader that accepts only the
            # array form reports every abort as a transaction that RAN, which
            # is how this bug hid from an earlier probe of mine.
            return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(line[1:])
            return None if n == -1 else [self._read() for _ in range(n)]
        raise RuntimeError("unparseable reply " + repr(line))

    def __call__(self, *args):
        self.sock.sendall(encode(args))
        return self._read()

    def close(self):
        try:
            self.f.close(); self.sock.close()
        except OSError:
            pass


def connect(port=PORT):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(1.0); s.connect(("127.0.0.1", port)); s.settimeout(None)
            return Client(s)
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")


def spawn():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1",
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


# Keys are exercised at both GenericValue representations, because the version
# slot is derived from that value's hash and the two halves hash differently:
# <= 23 bytes is inline SSO, longer is a heap copy. A derivation converted for
# only one of them would pass an SSO-only test and lose WATCH on long keys.
SSO_KEY = "k:sso"
HEAP_KEY = "k:" + "x" * 40


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn()
    try:
        c = connect()

        # ── MSET clears the TTL of every key it writes ───────────────────
        print("\ngh #230 — MSET clears TTL (Redis: an MSET is a SET)")
        for label, key in (("sso", SSO_KEY), ("heap", HEAP_KEY)):
            c("SET", key, "v1", "EX", "100")
            check(f"{label}: TTL is set before MSET", c("TTL", key) == b":100")
            c("MSET", key, "v2", "other", "x")
            check(f"{label}: MSET cleared the TTL", c("TTL", key) == b":-1",
                  f"got {c('TTL', key)!r}")
            check(f"{label}: MSET wrote the new value", c("GET", key) == b"v2")

        # Over-clearing is the opposite bug and just as bad: a key MSET never
        # named must keep its deadline.
        c("SET", "ttl:keep", "v", "EX", "100")
        c("MSET", "ttl:unrelated", "z")
        check("TTL of an untouched key survives MSET", c("TTL", "ttl:keep") == b":100",
              f"got {c('TTL', 'ttl:keep')!r}")

        # ── MSET bumps the WATCH version ─────────────────────────────────
        print("\ngh #230 — WATCH observes MSET")

        def watch_trial(mutator_args, watched, expect_abort):
            a, b = connect(), connect()
            try:
                a("SET", watched, "v0")
                a("WATCH", watched)
                a("MULTI")
                a("SET", "tx:sentinel", "1")
                b(*mutator_args)                      # other connection mutates
                aborted = a("EXEC") is None
                return aborted == expect_abort, aborted
            finally:
                a.close(); b.close()

        for label, key in (("sso", SSO_KEY), ("heap", HEAP_KEY)):
            ok, aborted = watch_trial(("MSET", key, "changed", "zz", "1"), key, True)
            check(f"{label}: MSET of a watched key aborts EXEC", ok,
                  f"EXEC {'aborted' if aborted else 'RAN'}")

        # The same derivation feeds SET and DEL. They worked before this change
        # and must still work: converting the bump sites without the recording
        # site (or vice versa) breaks WATCH silently in exactly this way.
        ok, aborted = watch_trial(("SET", SSO_KEY, "changed"), SSO_KEY, True)
        check("SET of a watched key still aborts EXEC", ok,
              f"EXEC {'aborted' if aborted else 'RAN'}")
        ok, aborted = watch_trial(("DEL", SSO_KEY), SSO_KEY, True)
        check("DEL of a watched key still aborts EXEC", ok,
              f"EXEC {'aborted' if aborted else 'RAN'}")

        # An MSET that never names the watched key must NOT abort it, or the
        # fix has simply become "bump everything" and every transaction fails.
        ok, aborted = watch_trial(("MSET", "wholly:other", "1", "second:other", "2"),
                                  "w:untouched", False)
        check("MSET of unrelated keys does not abort EXEC", ok,
              f"EXEC {'aborted' if aborted else 'RAN'}")

        # ── MSET semantics still hold ────────────────────────────────────
        print("\ngh #230 — MSET semantics unchanged")
        check("MSET returns +OK", c("MSET", "m:1", "a", "m:2", "b") == b"+OK")
        check("MSET wrote both pairs",
              (c("GET", "m:1"), c("GET", "m:2")) == (b"a", b"b"))
        odd = c("MSET", "m:3", "v", "m:4")
        check("odd argument count is an error",
              isinstance(odd, bytes) and odd.startswith(b"-ERR"), repr(odd))
        check("server aligned after the error", c("PING") == b"+PONG")

        # A wide MSET drives the batched WAL loop over many records in one
        # command, which is where a single-check/single-accounting scheme is
        # most likely to mis-account.
        wide = []
        for i in range(64):
            wide += [f"w:{i}", f"val{i}"]
        check("64-pair MSET returns +OK", c("MSET", *wide) == b"+OK")
        check("64-pair MSET wrote every key",
              all(c("GET", f"w:{i}") == f"val{i}".encode() for i in range(64)))

        # ── Durability across a crash (batched WAL append) ───────────────
        print("\ngh #230 — batched MSET records replay after SIGKILL")
        c("FLUSHALL")
        for i in range(3):
            pairs = []
            for j in range(10):
                pairs += [f"d:{i}:{j}", f"v{i}{j}"]
            c("MSET", *pairs)                  # 10 pairs: the batched path
        c("MSET", "d:short:1", "p", "d:short:2", "q")
        c("SET", "d:solo", "standalone")
        c("MSET", HEAP_KEY, "heapvalue", "d:tail", "t")
        expected = {f"d:{i}:{j}": f"v{i}{j}" for i in range(3) for j in range(10)}
        expected.update({"d:short:1": "p", "d:short:2": "q",
                         "d:solo": "standalone", HEAP_KEY: "heapvalue",
                         "d:tail": "t"})
        before = c("DBSIZE")
        check("all keys present before the kill", before == b":%d" % len(expected),
              f"got {before!r}, want {len(expected)}")
        c.close()
        # NB: DBSIZE after replay is deliberately not asserted equal to `before`.
        # The FLUSHALL above is not a WAL record, so replay rebuilds every key
        # this file ever wrote, not just the post-flush set. That resurrection
        # is its own question (Redis propagates FLUSHALL to the AOF and stays
        # empty); it is out of scope here, and what this section pins is that
        # nothing MSET wrote is LOST.

        # SIGKILL: no clean shutdown, no snapshot — WAL replay is the only
        # path by which any of this can come back.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        proc = spawn()
        c = connect()
        after = c("DBSIZE")
        check("keyspace is non-empty after WAL replay",
              isinstance(after, bytes) and int(after[1:]) >= len(expected),
              f"before {before!r}, after {after!r}")
        missing = [k for k, v in expected.items() if c("GET", k) != v.encode()]
        check("every MSET key replayed byte-exact", not missing,
              f"{len(missing)} wrong/missing, e.g. {missing[:5]}")
        c.close()
    finally:
        try:
            proc.kill(); proc.wait(timeout=5)
        except Exception:
            pass
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
