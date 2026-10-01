#!/usr/bin/env python3
"""W9 — AI.KNN_LM.* wire-surface smoke test.

Exercises the substrate end-to-end:
  CREATE → STORE one + STOREBATCH bulk → INFO → QUERY top-k → DROP

Verifies:
  - CREATE rejects duplicates and bad params
  - STORE / STOREBATCH counts are tracked correctly
  - QUERY returns the actual nearest neighbour first (small synthetic dataset)
  - QUERY response wire format: k × (Int32 token_id LE, Float32 distance LE)
  - DROP frees the datastore and subsequent ops fail cleanly
  - Error responses for: unknown datastore, dim mismatch, k=0, blob length mismatch

Requires: ./pion-server --kvcache -w 1 (or --inference)
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import numpy as np


HOST = "127.0.0.1"
PORT = 1974


def encode(parts):
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


class Conn:
    def __init__(self, host=HOST, port=PORT):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(15)
        self.s.connect((host, port))
        self.buf = b""

    def call(self, *parts):
        self.s.sendall(encode(parts))
        return self._read_one()

    def _read_one(self):
        while True:
            done = self._first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]; self.buf = self.buf[done:]
                return msg
            chunk = self.s.recv(4 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    @staticmethod
    def _first_complete(d):
        if len(d) < 3: return None
        nl = d.find(b"\r\n")
        if nl < 0: return None
        if d[0:1] in (b"+", b"-", b":"):
            return nl + 2
        if d[0:1] == b"$":
            ls = d[1:nl].decode()
            if ls == "-1": return nl + 2
            n = int(ls); need = nl + 2 + n + 2
            return need if len(d) >= need else None
        return nl + 2

    @staticmethod
    def parse_bulk(r):
        assert r.startswith(b"$"), f"expected bulk string, got {r[:40]!r}"
        nl = r.find(b"\r\n"); n = int(r[1:nl])
        if n < 0: return None
        return r[nl + 2:nl + 2 + n]


PASSED = 0
FAILED = 0


def check(name, ok, detail=""):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    try:
        c = Conn(port=args.port)
    except OSError as e:
        print(f"FAIL: pion not reachable on port {args.port} ({e})")
        return 2

    DS = "test_knnlm_ds_" + str(int(time.time()))
    DIM = 16

    print(f"AI.KNN_LM.* smoke against pion :{args.port}  ds={DS}  dim={DIM}")

    # ── CREATE ─────────────────────────────────────────────────────────────
    r = c.call("AI.KNN_LM.CREATE", DS, str(DIM), "1024")
    check("CREATE OK", r == b"+OK\r\n", repr(r))

    r = c.call("AI.KNN_LM.CREATE", DS, str(DIM))
    check("CREATE rejects duplicate", r.startswith(b"-ERR"), repr(r))

    r = c.call("AI.KNN_LM.CREATE", "bad_dim_ds", "0")
    check("CREATE rejects dim=0", r.startswith(b"-ERR"), repr(r))

    # ── STORE single ───────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    e0 = rng.standard_normal(DIM).astype(np.float32)
    r = c.call("AI.KNN_LM.STORE", DS, "42", e0.tobytes())
    check("STORE single OK", r == b"+OK\r\n", repr(r))

    # ── STOREBATCH ─────────────────────────────────────────────────────────
    N_BULK = 99
    bulk_ids = np.arange(100, 100 + N_BULK, dtype=np.int32)
    bulk_embs = rng.standard_normal((N_BULK, DIM)).astype(np.float32)
    r = c.call("AI.KNN_LM.STOREBATCH", DS, str(N_BULK),
                bulk_ids.tobytes(), bulk_embs.tobytes())
    check("STOREBATCH returns count", r == f":{N_BULK}\r\n".encode(), repr(r))

    # ── INFO ──────────────────────────────────────────────────────────────
    r = c.call("AI.KNN_LM.INFO", DS)
    body = c.parse_bulk(r) or b""
    check("INFO contains count", f"count={1+N_BULK}".encode() in body, repr(body))
    check("INFO contains dim",   f"dim={DIM}".encode() in body, repr(body))

    # ── QUERY: nearest neighbour is the exact-match query ────────────────
    target_idx = 7  # arbitrary
    target_token = int(bulk_ids[target_idx])
    q_emb = bulk_embs[target_idx]  # exact match → distance 0
    K = 5
    r = c.call("AI.KNN_LM.QUERY", DS, str(K), q_emb.tobytes())
    body = c.parse_bulk(r) or b""
    check("QUERY response is k*8 bytes", len(body) == K * 8, f"got {len(body)} bytes")
    if len(body) == K * 8:
        rows = []
        for i in range(K):
            tok, dist = struct.unpack("<if", body[i * 8:(i + 1) * 8])
            rows.append((tok, dist))
        check("QUERY top-1 is exact-match token", rows[0][0] == target_token,
              f"got {rows[0]} (expected token {target_token})")
        check("QUERY top-1 distance ≈ 0", rows[0][1] < 1e-4,
              f"got distance {rows[0][1]}")
        check("QUERY distances ascending",
              all(rows[i][1] <= rows[i + 1][1] for i in range(K - 1)),
              str(rows))

    # ── QUERY error paths ─────────────────────────────────────────────────
    r = c.call("AI.KNN_LM.QUERY", DS, "0", q_emb.tobytes())
    check("QUERY rejects k=0", r.startswith(b"-ERR"), repr(r))

    bad_emb = rng.standard_normal(DIM + 1).astype(np.float32).tobytes()
    r = c.call("AI.KNN_LM.QUERY", DS, "3", bad_emb)
    check("QUERY rejects dim mismatch", r.startswith(b"-ERR"), repr(r))

    r = c.call("AI.KNN_LM.QUERY", "nonexistent_ds", "3", q_emb.tobytes())
    check("QUERY rejects unknown ds", r.startswith(b"-ERR"), repr(r))

    # ── QUERY with k > count returns sentinels in trailing slots ─────────
    LARGE_K = 1 + N_BULK + 50  # exceeds count by 50
    r = c.call("AI.KNN_LM.QUERY", DS, str(LARGE_K), q_emb.tobytes())
    body = c.parse_bulk(r) or b""
    check("QUERY oversize k packs k entries", len(body) == LARGE_K * 8,
          f"got {len(body)} bytes")
    if len(body) == LARGE_K * 8:
        last_tok, last_dist = struct.unpack("<if", body[-8:])
        check("QUERY tail entry is sentinel (token=-1)", last_tok == -1,
              f"got token {last_tok}")

    # ── DROP ──────────────────────────────────────────────────────────────
    r = c.call("AI.KNN_LM.DROP", DS)
    check("DROP returns 1", r == b":1\r\n", repr(r))

    r = c.call("AI.KNN_LM.QUERY", DS, "5", q_emb.tobytes())
    check("QUERY after DROP fails", r.startswith(b"-ERR"), repr(r))

    r = c.call("AI.KNN_LM.DROP", DS)
    check("DROP idempotent (returns 0)", r == b":0\r\n", repr(r))

    print(f"\nAI.KNN_LM.* smoke: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
