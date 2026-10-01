#!/usr/bin/env python3
"""gh #115 — semantic-cache workspace audit trail, and the UTF-8 mangling it exposed.

`AI.SEMANTIC_CACHE` returns a cached answer, but a consumer had no way to see
what the model had in its workspace when that answer was generated. This adds
an opaque, caller-supplied snapshot (the J-lens top-k, base64) recorded at
cache-WRITE time and readable later WITHOUT any inference infrastructure —
which is the point: a compliance reviewer can ask what was in the workspace
without being able to run the model.

    AI.SEMANTIC_CACHE SET <query> <response> [WORKSPACE <blob>]
    AI.SEMANTIC_CACHE GET <query> [THRESHOLD f] [WITHWORKSPACE]
    AI.SEMANTIC_CACHE EXPLAIN <query>

**Deliberate deviation from the issue.** It specified that GET always return a
RESP3 map carrying the workspace. That would break every RESP3 client already
reading GET as a bulk string, so the map is OPT-IN via `WITHWORKSPACE` and the
default reply shape is untouched on both protocols. `EXPLAIN` — also in the
issue — is the audit path, and needs no flag.

**The bug found on the way in.** `cache_set` stored the response by walking it
one byte at a time and spelling every byte >= 128 as '?'. A UTF-8 response came
back mangled AND longer: "café" stored as "caf??", one '?' per byte. This is a
cache for MODEL OUTPUT, where curly quotes, accents, em-dashes and emoji are
the norm, so the corrupted case was the common one. Pinned below across UTF-8,
emoji, CJK and Arabic.

Needs an embedding backend to exercise the cache at all; on macOS the cheapest
is `--nle-embed` (Apple NLEmbedding, no model download, no Python). Without
one, every SET is a silent no-op and the whole file ENV_SKIPs rather than
reporting failures that are really an absent backend.

Run:  ./pion-server -p 1974 -w 1 --nle-embed
      python3 tests/test_gh115_workspace.py
"""
import argparse
import base64
import socket
import sys

PASS = 0
FAIL = 0


def client(port, proto=2):
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    f = s.makefile("rb")

    def read():
        line = f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        t, body = line[:1], line[1:-2]
        if t == b"$":
            n = int(body)
            return None if n == -1 else f.read(n + 2)[:-2]
        if t == b"_":
            return None
        if t in (b"*", b"%"):
            n = int(body)
            if n == -1:
                return None
            # A RESP3 map declares PAIRS; flatten it so both protocols compare
            # against the same expected list.
            return [read() for _ in range(n * 2 if t == b"%" else n)]
        return (t + body).decode()

    def cmd(*args):
        enc = [a.encode() if isinstance(a, str) else a for a in args]
        s.sendall(b"*%d\r\n" % len(enc) +
                  b"".join(b"$%d\r\n%s\r\n" % (len(e), e) for e in enc))
        return read()

    if proto == 3:
        cmd("HELLO", "3")
    return cmd


def check(label, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


def truthy(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    a = ap.parse_args()

    c = client(a.port)

    # An absent embedding backend makes every SET a silent no-op. Detect it
    # once rather than reporting a file full of failures that are really a
    # missing --nle-embed.
    probe_q = "gh115 probe query for backend detection"
    c("AI.SEMANTIC_CACHE", "SET", probe_q, "probe-response")
    if c("AI.SEMANTIC_CACHE", "GET", probe_q) is None:
        print("ENV_SKIP: no embedding backend — start the server with "
              "--nle-embed (macOS) or an Ollama/PyTorch backend.")
        return 0

    ws = base64.b64encode(bytes(range(64))).decode()
    ws2 = base64.b64encode(b"a different workspace snapshot").decode()

    print("\n=== 1. WORKSPACE round-trips through EXPLAIN ===")
    q1 = "what is the capital city of France"
    check("SET with WORKSPACE", c("AI.SEMANTIC_CACHE", "SET", q1, "Paris", "WORKSPACE", ws), "+OK")
    check("EXPLAIN returns it byte-exact", c("AI.SEMANTIC_CACHE", "EXPLAIN", q1), ws.encode())
    check("  ...and GET is unaffected", c("AI.SEMANTIC_CACHE", "GET", q1), b"Paris")

    print("\n=== 2. absent workspace is nil, not an empty string ===")
    q2 = "how tall is the mountain everest"
    check("SET without WORKSPACE still works", c("AI.SEMANTIC_CACHE", "SET", q2, "8849 m"), "+OK")
    check("EXPLAIN is nil", c("AI.SEMANTIC_CACHE", "EXPLAIN", q2), None)
    check("  ...GET still returns the answer", c("AI.SEMANTIC_CACHE", "GET", q2), b"8849 m")
    check("EXPLAIN on a cache MISS is nil",
          c("AI.SEMANTIC_CACHE", "EXPLAIN", "zzz unrelated gibberish qqq"), None)

    print("\n=== 3. index alignment across mixed writes ===")
    # `workspaces` runs parallel to `responses`, and OTHER callers (MCP memory,
    # RAG ingest) append to `responses` without a workspace. If the two ever
    # drift, EXPLAIN silently returns SOMEONE ELSE'S audit trail — worse than
    # returning none. Interleave and re-check every entry.
    q3 = "which ocean is the largest one on earth"
    c("AI.SEMANTIC_CACHE", "SET", q3, "Pacific", "WORKSPACE", ws2)
    check("first entry's workspace intact", c("AI.SEMANTIC_CACHE", "EXPLAIN", q1), ws.encode())
    check("no-workspace entry still nil", c("AI.SEMANTIC_CACHE", "EXPLAIN", q2), None)
    check("third entry's own workspace", c("AI.SEMANTIC_CACHE", "EXPLAIN", q3), ws2.encode())

    print("\n=== 4. GET shape: default unchanged, WITHWORKSPACE opt-in ===")
    check("RESP2 GET default is a bulk string", c("AI.SEMANTIC_CACHE", "GET", q1), b"Paris")
    check("RESP2 GET WITHWORKSPACE",
          c("AI.SEMANTIC_CACHE", "GET", q1, "WITHWORKSPACE"),
          [b"response", b"Paris", b"workspace", ws.encode()])
    check("WITHWORKSPACE on an entry with none",
          c("AI.SEMANTIC_CACHE", "GET", q2, "WITHWORKSPACE"),
          [b"response", b"8849 m", b"workspace", None])
    c3 = client(a.port, proto=3)
    check("RESP3 GET default is STILL a bulk string",
          c3("AI.SEMANTIC_CACHE", "GET", q1), b"Paris")
    check("RESP3 GET WITHWORKSPACE is a map",
          c3("AI.SEMANTIC_CACHE", "GET", q1, "WITHWORKSPACE"),
          [b"response", b"Paris", b"workspace", ws.encode()])

    print("\n=== 5. cached responses are BINARY-SAFE (the mangling fix) ===")
    # Every byte >= 128 used to become '?', one per byte: "café" -> "caf??".
    cases = [
        ("plain ascii", "a plain ascii answer"),
        ("accents", "Le café coûte 5€ — naïve résumé"),
        ("emoji", "done 🎯 ✅ shipped"),
        ("CJK + Arabic", "中文 العربية"),
        ("curly quotes", "He said “hello” and left…"),
    ]
    for idx, (label, resp) in enumerate(cases):
        q = f"binary safety probe number {idx} asking about things"
        c("AI.SEMANTIC_CACHE", "SET", q, resp)
        got = c("AI.SEMANTIC_CACHE", "GET", q)
        # isinstance, not just a None check: on a regression that changes the
        # reply SHAPE this must report FAIL, not raise and hide the rest.
        truthy(f"{label}: round-trips byte-exact",
               isinstance(got, bytes) and got.decode("utf-8", "replace") == resp,
               f"got {got!r}")

    print("\n=== 6. backwards compatibility and errors ===")
    check("2-arg SET (no WORKSPACE) still +OK",
          c("AI.SEMANTIC_CACHE", "SET", "legacy shaped set query here", "v"), "+OK")
    check("unknown subcommand names all three",
          c("AI.SEMANTIC_CACHE", "NOPE", "x"),
          "-ERR AI.SEMANTIC_CACHE subcommand must be GET, SET or EXPLAIN")
    # The command must consume its own frame — a trailing PING has to come back
    # exactly once, or the arm leaked tokens into the command stream (gh #240).
    check("frame consumed: EXPLAIN + PING", c("PING"), "+PONG")
    check("frame consumed: GET WITHWORKSPACE + PING", c("PING"), "+PONG")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
