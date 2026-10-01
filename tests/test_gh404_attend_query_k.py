#!/usr/bin/env python3
"""gh #404 — ATTEND.QUERY returns k value rows, not one.

`ATTEND.QUERY <sid> <layer> <k> <query>` ran a top-k search and then replied
with the FIRST result's value row only, on the RESP arm and on the 0xCA5E
binary lane alike: k=1, k=5 and k=32 all came back as one `value_dim * 4` byte
row, with no error and no count. A caller attending over "the top-k tokens"
attended over one.

The reply is now every result, best first, as one blob of
`num_results * value_dim` FP32s (num_results = min(k, tokens stored)), which is
what `vllm_pion.attention_plugin` already parses (`k_actual = floats //
value_dim`). k=1 replies are byte-identical to before.

Each stored token's value row is a unique marker (its token id, repeated), so a
row identifies the token it came from; the key set is small enough that the
graph search is exact, so the rows must be the brute-force L2 top-k, in order.

The binary lane gets the RESP arm's argument rules on the way: a query that is
not exactly key_dim floats used to be read past its end, k was unbounded, and
the token-id scratch leaked per query.

Usage:
    python3 tests/test_gh404_attend_query_k.py [--binary ./pion-server] [--port 7404]
"""

import argparse
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KD, VD, N = 16, 8, 100
PASS, FAIL = [], []


def check(name, ok, detail: object = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


class Client:
    def __init__(self, port, timeout=30):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.f = self.sock.makefile("rb")

    def cmd(self, *args):
        out = b"*%d\r\n" % len(args)
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.sock.sendall(out)
        return self.read()

    def read(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("server closed the connection")
        tag, body = line[:1], line[1:-2]
        if tag == b"*":
            n = int(body)
            return [] if n < 0 else [self.read() for _ in range(n)]
        if tag == b"$":
            n = int(body)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if tag == b"-":
            return Exception(body.decode(errors="replace"))
        if tag == b":":
            return int(body)
        return body

    def close(self):
        self.sock.close()


def binary_query(port, sid, layer, k, query_bytes):
    """One 0xCA5E CMD_ATTEND_QUERY (0x23) frame; returns (status, body)."""
    body = struct.pack("<H", len(sid)) + sid + struct.pack("<HH", layer, k) + query_bytes
    frame = struct.pack("<HBI", 0xCA5E, 0x23, len(body)) + body
    s = socket.create_connection(("127.0.0.1", port + 1), timeout=10)
    try:
        s.sendall(frame)
        hdr = b""
        while len(hdr) < 7:
            chunk = s.recv(7 - len(hdr))
            if not chunk:
                raise ConnectionError("binary lane closed")
            hdr += chunk
        magic, status, blen = struct.unpack("<HBI", hdr)
        assert magic == 0xCA5E, hex(magic)
        out = b""
        while len(out) < blen:
            chunk = s.recv(blen - len(out))
            if not chunk:
                raise ConnectionError("binary lane closed mid-body")
            out += chunk
        return status, out
    finally:
        s.close()


def rows_of(blob):
    """Token ids encoded in the value rows (row t is t/100 repeated)."""
    v = np.frombuffer(blob, dtype=np.float32).reshape(-1, VD)
    return [int(round(float(r[0]) * 100)) for r in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--port", type=int, default=7404)
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)  # the server runs in a temp cwd
    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2
    d = tempfile.mkdtemp(prefix="pion-gh404-")
    log = open(os.path.join(d, "server.log"), "w")
    proc = subprocess.Popen([args.binary, "--kvcache", "-w", "1", "-p", str(args.port),
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=d, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        deadline = time.time() + 60
        c = None
        while time.time() < deadline:
            try:
                c = Client(args.port, timeout=2)
                if c.cmd("PING") == b"PONG":
                    break
            except OSError:
                time.sleep(0.3)
        if not check("server ready", c is not None):
            return 1
        c = Client(args.port)

        rng = np.random.default_rng(404)
        keys = rng.standard_normal((N, KD)).astype(np.float32)
        # Value row t = t/100 in every lane. Values are INT8-quantized with one
        # scale per layer over [0, 0.99]: step ~0.0039, so t/100 rounds back to
        # t exactly (neighbours are 0.01 apart).
        vals = np.repeat((np.arange(N, dtype=np.float32) / 100.0)[:, None], VD, axis=1)
        check("ATTEND.CREATE", isinstance(c.cmd("ATTEND.CREATE", "s404", str(KD), str(VD)), int))
        check("ATTEND.STORE 100 tokens",
              c.cmd("ATTEND.STORE", "s404", "0", str(N), keys.tobytes(), vals.tobytes()) == b"OK")
        check("ATTEND.FINALIZE", c.cmd("ATTEND.FINALIZE", "s404", "0") == b"OK")

        q = keys[17] + 0.05 * rng.standard_normal(KD).astype(np.float32)
        dist = ((keys - q) ** 2).sum(axis=1)
        order = [int(t) for t in np.argsort(dist)]
        # The re-rank is exact in the norms but takes the angle from the INT8
        # graph (gh #391): a cosine error of ~0.01 moves a distance by about
        # 2|q||k| * 0.01, i.e. 1-3% at these ranks (measured worst 3.4%), so
        # neighbours that close may swap. Order is checked up to TIE. That
        # still fails a repeated row, a wrong row offset or an unsorted reply.
        TIE = 0.05

        def ranked_ok(got, k):
            if len(got) != k or len(set(got)) != k:
                return False
            # Best-first: no row may be clearly farther than a later one.
            for a in range(k - 1):
                if dist[got[a]] > dist[got[a + 1]] * (1 + TIE):
                    return False
            # The top-k SET: anything missing must tie the k-th distance.
            kth = dist[order[k - 1]]
            return all(dist[t] <= kth * (1 + TIE) for t in got)

        print("\n[RESP] reply length and content follow k")
        prev: bytes = b""
        for k in (1, 5, 32, 100):
            r = c.cmd("ATTEND.QUERY", "s404", "0", str(k), q.tobytes())
            ok_len = isinstance(r, bytes) and len(r) == k * VD * 4
            check(f"k={k}: {k} value rows ({k * VD * 4} bytes)", ok_len,
                  f"got {len(r) if isinstance(r, bytes) else r!r}")
            if not ok_len or not isinstance(r, bytes):
                continue
            got = rows_of(r)
            check(f"k={k}: rows are the L2 top-{k}, best first", got[0] == 17 and ranked_ok(got, k),
                  f"{got[:8]} vs {order[:8]}")
            if prev:
                check(f"k={k}: extends the k={len(prev) // (VD * 4)} reply", r[:len(prev)] == prev)
            prev = r
        r = c.cmd("ATTEND.QUERY", "s404", "0", "150", q.tobytes())
        check("k=150 > 100 stored: all 100 rows", isinstance(r, bytes) and len(r) == N * VD * 4,
              f"got {len(r) if isinstance(r, bytes) else r!r}")
        r = c.cmd("ATTEND.QUERY", "s404", "0", "1", keys[42].tobytes())
        check("k=1 self-query: one row, the key's own value", isinstance(r, bytes)
              and len(r) == VD * 4 and rows_of(r) == [42])

        print("\n[binary lane] same body, and the RESP arm's argument rules")
        resp5 = c.cmd("ATTEND.QUERY", "s404", "0", "5", q.tobytes())
        st, body = binary_query(args.port, b"s404", 0, 5, q.tobytes())
        check("k=5: STATUS_OK, identical to the RESP reply", st == 0 and body == resp5,
              f"status {st}, {len(body)} bytes")
        st, body = binary_query(args.port, b"s404", 0, 1, keys[42].tobytes())
        check("k=1 self-query: one row", st == 0 and rows_of(body) == [42])
        st, body = binary_query(args.port, b"s404", 0, 5, q.tobytes()[:-4])
        check("query one float short: STATUS_ERROR (was read past the frame)", st == 2,
              f"status {st} {body[:60]!r}")
        st, body = binary_query(args.port, b"s404", 0, 5, q.tobytes() + b"\0\0\0\0")
        check("query one float long: STATUS_ERROR", st == 2, f"status {st}")
        st, body = binary_query(args.port, b"s404", 0, 0, q.tobytes())
        check("k=0: STATUS_ERROR", st == 2, f"status {st}")
        st, body = binary_query(args.port, b"s404", 0, 5000, q.tobytes())
        check("k=5000 > 4096: STATUS_ERROR", st == 2, f"status {st}")
        st, body = binary_query(args.port, b"nosuch", 0, 5, q.tobytes())
        check("unknown session: STATUS_MISS", st == 1, f"status {st}")
        check("server still answers", c.cmd("PING") == b"PONG")

        # A reply past the 4 MB binary buffer is streamed with a blocking
        # send-all. It used to retry EVERY error — send() has SIGPIPE
        # suppressed, so a client that left mid-body turned into EPIPE /
        # ECONNRESET forever and hung the worker. 16 MB reply, client gone.
        BVD, BN = 1024, 4096
        bkeys = rng.standard_normal((BN, KD)).astype(np.float32)
        bvals = rng.standard_normal((BN, BVD)).astype(np.float32)
        check("ATTEND.CREATE big (value_dim 1024)",
              isinstance(c.cmd("ATTEND.CREATE", "big404", str(KD), str(BVD)), int))
        check("ATTEND.STORE 4096 tokens",
              c.cmd("ATTEND.STORE", "big404", "0", str(BN), bkeys.tobytes(), bvals.tobytes()) == b"OK")
        check("ATTEND.FINALIZE big", c.cmd("ATTEND.FINALIZE", "big404", "0") == b"OK")
        body = struct.pack("<H", 6) + b"big404" + struct.pack("<HH", 0, 4096) + bkeys[0].tobytes()
        s2 = socket.create_connection(("127.0.0.1", args.port + 1), timeout=10)
        s2.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s2.sendall(struct.pack("<HBI", 0xCA5E, 0x23, len(body)) + body)
        _ = s2.recv(7)                     # the header arrived: the body is streaming
        s2.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s2.close()                         # RST, with ~16 MB still unsent
        time.sleep(0.5)
        try:
            c.sock.settimeout(10)
            ok = c.cmd("PING") == b"PONG"
        except (OSError, ConnectionError):
            ok = False
        check("a client that leaves mid-16 MB-reply does not hang the worker", ok)
        c.close()
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
