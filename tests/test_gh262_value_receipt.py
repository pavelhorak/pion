#!/usr/bin/env python3
"""gh #262 — the value receipt: PION.STATS, a truthful INFO, and the ledger.

The server used to measure nothing about the value it delivered. INFO
hardcoded `tcp_port:1974`, `used_memory:1048576`, an empty `# Keyspace`;
`MetricsRegistry` was instantiated once and called from nowhere. The one
number the wedge user retains on — "how much prefill did Pion skip?" — was
computed client-side by pion-serve and never by the server.

What this file pins:
  [1] INFO reports the REAL port, RSS, uptime and per-worker key counts, and
      no longer contains the old hardcoded values.
  [2] PION.STATS is a 16-pair map, zero on a fresh server, RESP2 flat array.
  [3] KV.PREFIX.LOOKUP records misses and hits; a hit on a prefix registered
      with PREFILL_MS is credited exactly that MEASURED time.
  [4] A hit on a prefix registered WITHOUT PREFILL_MS is credited a per-token
      ESTIMATE from the stored token count; the measured twin does not move.
  [5] V.FETCH RANGE bytes are counted.
  [6] HELLO 3 turns the reply into a RESP3 map (%16).
  [7] PION.STATS RESET zeroes the counters; PREFILL_MS is range-checked.
  [8] Pipeline integrity: PION.STATS + PING → exactly two replies.
  [9] MetricsRegistry is gone from the tree.

Runs against its own server on port 19620 (binary lane on 19621).
Usage: python3 tests/test_gh262_value_receipt.py [--binary ./pion-server-dev]
"""
import os, re, socket, struct, subprocess, sys, tempfile, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 19620
failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        a = a if isinstance(a, bytes) else str(a).encode()
        parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
    return b"".join(parts)


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        self.sock.sendall(encode(args)); return self._read()

    def raw(self, payload, n_replies):
        self.sock.sendall(payload); return [self._read() for _ in range(n_replies)]

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        t = line[:1]
        if t in b"+-:":
            return line[:-2]
        if t == b"$":
            n = int(line[1:]); return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(line[1:]); return None if n == -1 else [self._read() for _ in range(n)]
        if t == b"%":
            n = int(line[1:]); return {"__map__": [self._read() for _ in range(2 * n)]}
        if t == b"_":
            return None
        raise RuntimeError("unparseable reply " + repr(line))

    def close(self):
        try: self.f.close(); self.sock.close()
        except OSError: pass


def stats(c):
    r = c("PION.STATS")
    if isinstance(r, dict): r = r["__map__"]
    assert isinstance(r, list) and len(r) % 2 == 0, r
    out = {}
    for k, v in zip(r[::2], r[1::2]):
        v = v.decode() if isinstance(v, bytes) else v
        out[k.decode()] = int(v[1:]) if isinstance(v, str) and v[:1] == ":" else v
    return out


def info(c):
    body = c("INFO").decode()
    return {ln.split(":", 1)[0]: ln.split(":", 1)[1] for ln in body.split("\r\n") if ":" in ln and not ln.startswith("#")}


def main():
    binary = os.environ.get("PION_BIN", "./pion-server")
    if "--binary" in sys.argv:
        binary = sys.argv[sys.argv.index("--binary") + 1]
    if not os.path.exists(os.path.join(REPO, binary)) and os.path.exists(os.path.join(REPO, "pion-server-dev")):
        binary = "./pion-server-dev"
    work = tempfile.mkdtemp(prefix="gh262_")
    proc = subprocess.Popen([os.path.join(REPO, binary), "-p", str(PORT), "-w", "1", "--kvcache",
                             "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                            cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 60
        c = None
        while time.monotonic() < deadline:
            try:
                c = Client(PORT); break
            except OSError:
                if proc.poll() is not None:
                    print(proc.stdout.read()); raise SystemExit("server died at startup")
                time.sleep(0.25)
        if c is None:
            raise SystemExit("server did not come up")
        c("PING")

        print("\n[1] INFO reports resolved values, not hardcoded ones")
        i = info(c)
        check("tcp_port is the real port", i.get("tcp_port") == str(PORT), i.get("tcp_port"))
        check("pion_version present", re.match(r"^\d+\.\d+\.\d+\+", i.get("pion_version", "")) is not None, i.get("pion_version"))
        um = int(i.get("used_memory", "0"))
        check("used_memory is a real RSS (> 4 MiB, != 1048576)", um > 4 * 1024 * 1024 and um != 1048576, um)
        check("used_memory_peak >= used_memory", int(i.get("used_memory_peak", "0")) >= um, i.get("used_memory_peak"))
        check("uptime_in_seconds is an integer", i.get("uptime_in_seconds", "x").isdigit(), i.get("uptime_in_seconds"))
        body0 = c("INFO").decode()
        check("empty keyspace has no db0 line (Redis shape)", "db0:" not in body0)
        for k in ("k1", "k2", "k3"):
            c("SET", k, "v")
        c("EXPIRE", "k3", "1000")
        i = info(c)
        check("db0 keys=3 after three SETs", i.get("db0", "").startswith("keys=3,"), i.get("db0"))
        check("db0 expires=1 after one EXPIRE", ",expires=1," in i.get("db0", ""), i.get("db0"))
        c("DEL", "k1")
        check("db0 keys=2 after DEL", info(c).get("db0", "").startswith("keys=2,"), info(c).get("db0"))
        check("# Pion section present", "# Pion\r\n" in body0)
        check("stats_scope:worker declared", "stats_scope:worker" in body0)

        print("\n[2] PION.STATS on a fresh server")
        s = stats(c)
        expected = ["worker_id", "uptime_in_seconds", "kvprefix_hits", "kvprefix_misses",
                    "kvprefix_hits_measured", "kvprefix_tokens_served", "kvprefix_bytes_served",
                    "kvprefix_prefill_us_saved", "kvprefix_prefill_us_measured",
                    "prefill_seconds_avoided", "prefill_seconds_avoided_measured",
                    "semantic_hits", "semantic_misses", "moe_hits", "moe_misses", "vector_queries"]
        check("16 fields, in order", list(s.keys()) == expected, list(s.keys()))
        zero = [k for k in expected if k not in ("worker_id", "uptime_in_seconds", "prefill_seconds_avoided", "prefill_seconds_avoided_measured")]
        check("all counters zero", all(s.get(k) == 0 for k in zero), {k: s.get(k) for k in zero})
        check("seconds text is 0.000", s.get("prefill_seconds_avoided") == "0.000" and s.get("prefill_seconds_avoided_measured") == "0.000", s)

        print("\n[3] KV.PREFIX.LOOKUP — miss, then a MEASURED hit")
        check("lookup of unknown prefix is MISS", c("KV.PREFIX.LOOKUP", "app|m|p1") == b"+MISS")
        check("miss counted", stats(c)["kvprefix_misses"] == 1, stats(c))
        check("REGISTER with PREFILL_MS", c("KV.PREFIX.REGISTER", "app|m|p1", "64", "fp16", "PREFILL_MS", "1200") == b"+OK")
        check("lookup is HIT", c("KV.PREFIX.LOOKUP", "app|m|p1") == b"+HIT")
        s = stats(c)
        check("hit counted", s["kvprefix_hits"] == 1, s)
        check("hit is a measured hit", s["kvprefix_hits_measured"] == 1, s)
        check("credited exactly 1,200,000 us", s["kvprefix_prefill_us_saved"] == 1_200_000 and s["kvprefix_prefill_us_measured"] == 1_200_000, s)
        check("seconds text 1.200 / 1.200", s["prefill_seconds_avoided"] == "1.200" and s["prefill_seconds_avoided_measured"] == "1.200", s)
        c("KV.PREFIX.LOOKUP", "app|m|p1"); c("KV.PREFIX.LOOKUP", "app|m|p1")
        s = stats(c)
        check("three hits credit 3.600 s", s["kvprefix_hits"] == 3 and s["prefill_seconds_avoided"] == "3.600", s)
        check("INFO carries the same number", "prefill_seconds_avoided:3.600" in c("INFO").decode())

        print("\n[4] A prefix registered WITHOUT PREFILL_MS is credited an ESTIMATE from its token count")
        check("REGISTER without PREFILL_MS", c("KV.PREFIX.REGISTER", "app|m|p2", "8", "fp16") == b"+OK")
        n_tok, dim = 100, 8
        blob = struct.pack(f"<{n_tok * dim}f", *([0.5] * (n_tok * dim)))
        r = c("V.STOREBATCH", "app|m|p2_pk", "0", "0", str(n_tok), blob)
        check("V.STOREBATCH on the K side", r[:1] in (b"+", b":"), r)
        before = stats(c)
        check("lookup HIT on p2", c("KV.PREFIX.LOOKUP", "app|m|p2") == b"+HIT")
        s = stats(c)
        check("tokens_served += 100", s["kvprefix_tokens_served"] - before["kvprefix_tokens_served"] == n_tok, s)
        est = s["kvprefix_prefill_us_saved"] - before["kvprefix_prefill_us_saved"]
        check("estimate = 100 tokens x 555 us", est == n_tok * 555, est)
        check("measured twin did NOT move", s["kvprefix_prefill_us_measured"] == before["kvprefix_prefill_us_measured"], s)
        check("measured hits did NOT move", s["kvprefix_hits_measured"] == before["kvprefix_hits_measured"], s)
        check("seconds text 3.655", s["prefill_seconds_avoided"] == "3.655", s["prefill_seconds_avoided"])

        print("\n[5] V.FETCH RANGE bytes are counted")
        before = stats(c)
        r = c("V.FETCH", "app|m|p2_pk", "0", "RANGE", "0", str(n_tok))
        check("fetch returned a payload", isinstance(r, bytes) and len(r) > 0, type(r))
        s = stats(c)
        check("bytes_served grew by the payload size", s["kvprefix_bytes_served"] - before["kvprefix_bytes_served"] == len(r),
              (s["kvprefix_bytes_served"], before["kvprefix_bytes_served"], len(r)))

        print("\n[6] RESP3: HELLO 3 makes PION.STATS a %-map")
        c3 = Client(PORT)
        h = c3("HELLO", "3")
        check("HELLO 3 accepted", h is not None)
        c3.sock.sendall(encode(["PION.STATS"]))
        first = c3.f.readline()
        check("reply starts with %16", first == b"%16\r\n", first)
        for _ in range(32): c3._read()
        c3.close()

        print("\n[7] RESET and argument validation")
        check("PION.STATS RESET → +OK", c("PION.STATS", "RESET") == b"+OK")
        s = stats(c)
        check("counters zero after RESET", all(s[k] == 0 for k in zero), s)
        check("uptime survives RESET", isinstance(s["uptime_in_seconds"], int) and s["uptime_in_seconds"] >= 0, s)
        check("PION.STATS junk-arg is not an error", isinstance(c("PION.STATS", "bogus"), (list, dict)))
        r = c("KV.PREFIX.REGISTER", "app|m|p3", "8", "fp16", "PREFILL_MS", "-5")
        check("PREFILL_MS -5 refused", r.startswith(b"-ERR"), r)
        check("...and the refusal was a no-op", c("KV.PREFIX.LOOKUP", "app|m|p3") == b"+MISS")
        r = c("KV.PREFIX.REGISTER", "app|m|p3", "8", "fp16", "prefill_ms", "7")
        check("PREFILL_MS keyword is case-insensitive", r == b"+OK", r)

        print("\n[8] Pipeline integrity")
        rs = c.raw(encode(["PION.STATS"]) + encode(["PING"]), 2)
        check("PION.STATS + PING → map then PONG", isinstance(rs[0], list) and rs[1] == b"+PONG", rs[1])
        rs = c.raw(encode(["PION.STATS", "RESET"]) + encode(["PING"]), 2)
        check("PION.STATS RESET + PING → +OK then PONG", rs[0] == b"+OK" and rs[1] == b"+PONG", rs)
        rs = c.raw(encode(["KV.PREFIX.REGISTER", "app|m|p4", "8", "fp16", "PREFILL_MS", "3"]) + encode(["PING"]), 2)
        check("REGISTER ... PREFILL_MS + PING → +OK then PONG", rs[0] == b"+OK" and rs[1] == b"+PONG", rs)
        rs = c.raw(encode(["KV.PREFIX.REGISTER", "app|m|p5", "8", "fp16"]) + encode(["PREFILL_MS"]), 2)
        check("a pipelined command NAMED prefill_ms is not eaten as an option", rs[0] == b"+OK" and rs[1].startswith(b"-ERR unknown command"), rs)
        c.close()

        print("\n[9] MetricsRegistry is gone")
        out = subprocess.run(["grep", "-rl", "struct MetricsRegistry", os.path.join(REPO, "src")], capture_output=True, text=True).stdout
        check("no `struct MetricsRegistry` under src/", out.strip() == "", out)
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for n, d in failures:
        print(f"  - {n}: {d}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
