#!/usr/bin/env python3
"""gh #202 + gh #203 regression test.

Both changes are perf work on paths where the failure mode is silent, so this
covers behavior the benchmarks would never notice:

gh #202
  - `RESP3Token.value()` no longer builds its String one char at a time. It now
    bulk-constructs when the token is all-ASCII and keeps the old per-character
    spelling (bytes >= 128 -> '?') only when a high byte is present. Every
    slow-path handler that names a token goes through this, so the test drives
    a handler that echoes a token back (XREAD's stream-name reply, ECHO) with
    ASCII, empty, exactly-16-byte (the SIMD stride), and high-byte payloads.
  - The stream family now keys through `GenericValue.from_ptr` on the token
    instead of `value()` + `from_string`. That makes stream keys binary-exact
    rather than '?'-mangled — the point of the test below is that the whole
    family agrees, i.e. a key written by XADD is findable by XLEN / XRANGE /
    XDEL / XTRIM / XINFO / XREAD. Converting one site and not the rest would
    make a binary key writable but unreadable.

gh #203
  - `SkipListNode.__init__` nulls only `forward[0..level)` and both `update`
    arrays are left uninitialized. Sound only because every reader is bounded
    by the level discipline; if that is ever violated, the symptom is a garbage
    pointer walk, so the test pushes enough members to force level promotion
    well past the E[level]=1.33 average and then exercises every traversal
    (range, rank order, pop, remove, upsert-move).
"""
import os, socket, subprocess, sys, time, shutil, random

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1981
WORKDIR = f"/tmp/pion_gh202_test_{PORT}"

failures = []
passes = []
known = []   # pre-existing issues this file documents but does not gate on


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
    """Framing RESP client.

    Deliberately NOT one-recv-per-command: replies coalesce under load, so a
    naive `recv()` per command silently pairs reply N with request N+k and
    reports phantom timeouts. Read exactly one reply, byte-counted.
    """

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
            if n == -1:
                return None
            return self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(line[1:])
            if n == -1:
                return None
            return [self._read() for _ in range(n)]
        raise RuntimeError("unparseable reply " + repr(line))

    def raw(self, *args):
        """Reply as its RESP wire bytes (for shape assertions)."""
        self.sock.sendall(encode(args))
        return self._raw_read()

    def _raw_read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        t = line[:1]
        if t in b"+-:":
            return line
        if t == b"$":
            n = int(line[1:])
            if n == -1:
                return line
            return line + self.f.read(n + 2)
        if t == b"*":
            n = int(line[1:])
            out = line
            if n > 0:
                for _ in range(n):
                    out += self._raw_read()
            return out
        raise RuntimeError("unparseable reply " + repr(line))

    def __call__(self, *args):
        self.sock.sendall(encode(args))
        return self._read()

    def close(self):
        try:
            self.f.close(); self.sock.close()
        except OSError:
            pass


def flatten(v):
    """Flatten a nested array reply to a list of leaf byte strings."""
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(flatten(x))
        return out
    return [v]


def connect():
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(1.0); s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            return Client(s)
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1",
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        c = connect()

        # ── gh #202: RESP3Token.value() spelling ─────────────────────────
        print("\ngh #202 — token value() spelling")
        for label, payload in [
            ("ascii-short", b"hello"),
            ("ascii-15", b"a" * 15),     # just under the SIMD stride
            ("ascii-16", b"b" * 16),     # exactly one stride
            ("ascii-17", b"c" * 17),     # stride + tail
            ("ascii-64", b"d" * 64),     # several strides
            ("empty", b""),
        ]:
            check(f"ECHO {label} exact", c.raw("ECHO", payload)
                  == b"$%d\r\n%s\r\n" % (len(payload), payload))
        # A high byte must still be answered (spelled '?'), never crash.
        check("ECHO high-byte answered", c.raw("ECHO", b"caf\xc3\xa9").startswith(b"$"))
        check("server alive after high-byte token", c("PING") == b"+PONG")

        # ── hand-counted RESP literals ───────────────────────────────────
        # Three handlers shipped wrong literal lengths, all invisible to a
        # lenient client and to substring-matching tests: XINFO STREAM wrote 18
        # bytes of a 16-byte header (two NULs before the integer) and declared
        # $15 for the 17-byte `last-generated-id`; ACL LIST declared $31 for a
        # 34-byte payload and wrote 41 of 45 bytes; ACL USERS wrote 18 of 17.
        # The framing reader above turns any of these into a hard parse error.
        print("\nRESP literal lengths (XINFO STREAM / ACL)")
        c("DEL", "st:shape")
        c("XADD", "st:shape", "1-1", "f", "v1")
        c("XADD", "st:shape", "2-1", "f", "v2")
        def array_reply(name, reply, size):
            ok = isinstance(reply, list) and len(reply) == size
            check(f"{name} is a {size}-element array", ok, repr(reply))
            return reply if ok else [None] * size

        def no_desync(name, *args):
            """Pipeline a PING behind the command and require its own reply.

            A wrong bulk length shows up as the next reply being consumed, so
            this is the assertion that actually catches the literal-length bugs
            this file exists for.

            This was a KNOWN ISSUE until gh #214: these commands ALSO had a
            separate, pre-existing token-skip desync — `handle_acl` /
            `handle_xinfo` and 17 others (XRANGE, XREVRANGE, XDEL, XTRIM, ZREM,
            DEBUG, MEMORY, SLOWLOG, LATENCY, MODULE, SINTERCARD, FUNCTION LIST,
            SCRIPT EXISTS) returned `num_tokens - i - 1` from a dispatch site
            doing `i += handler(...)`, skipping to the end of the whole recv
            buffer and swallowing everything pipelined behind them. gh #214
            converted all 15 dispatch sites to `i = cmd_end_tok - 1` with
            `cmd_end_tok` as the handler bound, and
            `tests/diag_pipeline_desync_sweep.py` now reports 0 of 57 affected.

            So a swallow here is a REAL FAILURE again, and it is the assertion
            that catches a wrong hand-counted bulk length — the thing this file
            exists for.

            Runs on its own connection: a swallow leaves the reader waiting,
            and a timed-out makefile stays permanently unusable, so sharing the
            main client would poison every later assertion.
            """
            probe = socket.create_connection(("127.0.0.1", PORT), timeout=2.0)
            try:
                probe.sendall(encode(args) + encode(("PING",)))
                deadline = time.monotonic() + 2.0
                buf = b""
                while time.monotonic() < deadline and b"+PONG" not in buf:
                    try:
                        chunk = probe.recv(65536)
                    except (socket.timeout, OSError):
                        break
                    if not chunk:
                        break
                    buf += chunk
            finally:
                probe.close()
            if b"+PONG" in buf:
                passes.append(f"no desync after {name}")
                print(f"  PASS  no desync after {name}")
            else:
                failures.append((f"no desync after {name}",
                                 "pipelined PING swallowed — either a wrong bulk "
                                 "length or a token-skip regression (gh #214)"))
                print(f"  FAIL  pipelined command swallowed after {name}")

        info = array_reply("XINFO STREAM", c("XINFO", "STREAM", "st:shape"), 6)
        check("XINFO field names intact",
              [info[0], info[2], info[4]] == [b"length", b"last-generated-id", b"entries"],
              repr(info))
        check("XINFO values correct",
              [info[1], info[3], info[5]] == [b":2", b"2-1", b":2"], repr(info))
        no_desync("XINFO STREAM", "XINFO", "STREAM", "st:shape")

        acl_list = array_reply("ACL LIST", c("ACL", "LIST"), 1)
        check("ACL LIST entry intact",
              acl_list[0] == b"user default on nopass ~* &* +@all", repr(acl_list))
        no_desync("ACL LIST", "ACL", "LIST")

        acl_users = array_reply("ACL USERS", c("ACL", "USERS"), 1)
        check("ACL USERS entry intact", acl_users[0] == b"default", repr(acl_users))
        no_desync("ACL USERS", "ACL", "USERS")

        # ── gh #202: stream family keys agree ────────────────────────────
        print("\ngh #202 — stream family key agreement")
        for label, key in [("ascii", b"st:ascii"), ("binary", b"st:\xff\xfe\x80bin")]:
            c("DEL", key)
            c("XADD", key, "1-1", "f", "v1")
            c("XADD", key, "2-1", "f", "v2")
            check(f"XLEN sees XADD ({label})", c("XLEN", key) == b":2")
            check(f"XRANGE sees XADD ({label})", b"v1" in flatten(c("XRANGE", key, "-", "+")))
            check(f"XREVRANGE sees XADD ({label})",
                  b"v2" in flatten(c("XREVRANGE", key, "+", "-")))
            check(f"XINFO sees XADD ({label})", b"length" in flatten(c("XINFO", "STREAM", key)))
            # XREAD echoes the stream name back on the reply path.
            check(f"XREAD echoes key verbatim ({label})",
                  key in flatten(c("XREAD", "COUNT", "10", "STREAMS", key, "0")))
            check(f"XDEL finds key ({label})", c("XDEL", key, "1-1") == b":1")
            check(f"XLEN after XDEL ({label})", c("XLEN", key) == b":1")
            c("XADD", key, "3-1", "f", "v3")
            c("XTRIM", key, "MAXLEN", "1")
            check(f"XLEN after XTRIM ({label})", c("XLEN", key) == b":1")

        # Auto-ID XADD: well formed, strictly increasing, stack id buffer.
        c("DEL", "st:auto")
        ids = [c("XADD", "st:auto", "*", "k", "v") for _ in range(500)]
        parsed = [(int(x.split(b"-")[0]), int(x.split(b"-")[1])) for x in ids]
        check("XADD auto-IDs well formed", all(len(x.split(b"-")) == 2 for x in ids))
        check("XADD auto-IDs strictly increasing",
              all(parsed[i] < parsed[i + 1] for i in range(len(parsed) - 1)))
        check("XLEN after 500 auto appends", c("XLEN", "st:auto") == b":500")

        # ── gh #203 / allocator: skip list under level promotion ─────────
        print("\ngh #203 — skip list traversal under level promotion")
        N = 5000
        c("DEL", "z:big")
        for i in range(N):
            c("ZADD", "z:big", str(i), f"m{i:05d}")
        check("ZCARD after bulk insert", c("ZCARD", "z:big") == b":%d" % N)
        members = flatten(c("ZRANGE", "z:big", "0", "-1"))
        check("ZRANGE returns all members", len(members) == N, f"got {len(members)}")
        check("ZRANGE is score-ordered", members == sorted(members))
        check("ZRANGEBYSCORE window", len(flatten(c("ZRANGEBYSCORE", "z:big", "100", "199"))) == 100)

        # Upserts move nodes: remove + reinsert at every level.
        rnd = random.Random(1974)
        moved = [rnd.randrange(N) for _ in range(500)]
        for i in moved:
            c("ZADD", "z:big", str(N + i), f"m{i:05d}")
        check("ZCARD unchanged after upserts", c("ZCARD", "z:big") == b":%d" % N)
        check("ZSCORE reflects moved score",
              c("ZSCORE", "z:big", f"m{moved[0]:05d}").startswith(str(N + moved[0]).encode()))

        # ── the ZREM rebuild path (SlabAllocator.reset bound) ────────────
        # ZREM collects every node, calls SlabSkipList.reset(), then reinserts
        # the survivors. reset() frees all slabs but the first while allocate()
        # kept the grown items_per_slab, so the refill walked off the end of the
        # first mapping — ZREM on a ~200-member zset SIGSEGV'd the server.
        print("\nZREM rebuild — allocator bound after reset()")
        for size in (200, 1000, 5000):
            key = f"z:rebuild{size}"
            c("DEL", key)
            for i in range(size):
                c("ZADD", key, str(i), f"m{i:05d}")
            check(f"ZREM on {size}-member zset replies", c("ZREM", key, "m00000") == b":1")
            check(f"ZCARD after ZREM ({size})", c("ZCARD", key) == b":%d" % (size - 1))
            survivors = flatten(c("ZRANGE", key, "0", "-1"))
            check(f"ZRANGE intact after rebuild ({size})",
                  len(survivors) == size - 1 and survivors == sorted(survivors),
                  f"got {len(survivors)}")
            check(f"server alive after ZREM ({size})", c("PING") == b"+PONG")

        # Strided ZREM: many rebuilds back to back.
        expect = N - len(range(0, N, 7))
        for i in range(0, N, 7):
            c("ZREM", "z:big", f"m{i:05d}")
        check("ZCARD after strided ZREM", c("ZCARD", "z:big") == b":%d" % expect)
        check("ZRANGE consistent after strided ZREM",
              len(flatten(c("ZRANGE", "z:big", "0", "-1"))) == expect)

        # ZREMRANGEBYRANK/SCORE use the same collect-reset-reinsert shape.
        c("DEL", "z:rr")
        for i in range(1500):
            c("ZADD", "z:rr", str(i), f"r{i:05d}")
        check("ZREMRANGEBYSCORE replies", c("ZREMRANGEBYSCORE", "z:rr", "0", "499") == b":500")
        check("ZCARD after ZREMRANGEBYSCORE", c("ZCARD", "z:rr") == b":1000")
        check("ZREMRANGEBYRANK replies", c("ZREMRANGEBYRANK", "z:rr", "0", "99") == b":100")
        check("ZCARD after ZREMRANGEBYRANK", c("ZCARD", "z:rr") == b":900")
        check("ZRANGE intact after range removals",
              len(flatten(c("ZRANGE", "z:rr", "0", "-1"))) == 900)

        # pop_min drains through head[].forward[i].
        c("DEL", "z:pop")
        M = 2000
        for i in range(M):
            c("ZADD", "z:pop", str(i), f"p{i:05d}")
        popped = []
        for _ in range(M):
            r = flatten(c("ZPOPMIN", "z:pop"))
            if r:
                popped.append(r[0])
        check("ZPOPMIN drained everything", len(popped) == M, f"got {len(popped)}")
        check("ZPOPMIN returned ascending", popped == sorted(popped))
        check("ZCARD 0 after drain", c("ZCARD", "z:pop") == b":0")

        # Mixed churn — insert/pop interleaved across many rebuilds.
        c("DEL", "z:churn")
        for round_ in range(20):
            for i in range(200):
                c("ZADD", "z:churn", str(rnd.randrange(10000)), f"c{round_}:{i}")
            for _ in range(100):
                c("ZPOPMIN", "z:churn")
        check("ZCARD after churn", c("ZCARD", "z:churn") == b":2000")
        check("ZRANGE after churn", len(flatten(c("ZRANGE", "z:churn", "0", "-1"))) == 2000)
        check("server alive at end", c("PING") == b"+PONG")
        c.close()
    finally:
        proc.kill(); proc.wait(timeout=5)
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed, {len(known)} known-issue")
    for k in known:
        print(f"  KNOWN: {k}")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
