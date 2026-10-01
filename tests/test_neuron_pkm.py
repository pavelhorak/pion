#!/usr/bin/env python3
"""gh #146 — NEURON.PKM.* wire + correctness + latency test.

Exercises the product-key memory surface end-to-end:
  CREATE → SETKEYS(0,1) → QUERY / QUERY FAST → SETVALS → FFN → INFO → DROP

Correctness is checked against a numpy brute force over the *full materialized*
N-slot score vector — the point of product-key decomposition is that it is
EXACT, so top-k ids and scores must match a dense argsort, not merely overlap.

Also measures QUERY p50 at the scales gh #146 priced against AI.KNN_LM.* HNSW
(5K / 50K / 250K / 1M slots, dim 896, k 32) so the two are directly comparable.

Requires: ./pion-server --kvcache -w 1     (or --inference)
Usage:    python3 tests/test_neuron_pkm.py [--bench] [--port 1974]
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

PASSED = 0
FAILED = 0


def encode(parts):
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        b = p if isinstance(p, bytes) else str(p).encode()
        out.append(f"${len(b)}\r\n".encode())
        out.append(b)
        out.append(b"\r\n")
    return b"".join(out)


class Conn:
    def __init__(self, host=HOST, port=PORT):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(60)
        self.s.connect((host, port))
        # Buffered reader — slicing a bytes accumulator is O(N^2) on big blobs.
        self.f = self.s.makefile("rb")

    def call(self, *parts):
        self.s.sendall(encode(parts))
        return self._read_one()

    def _read_one(self):
        line = self.f.readline()
        if not line:
            raise ConnectionError("pion closed")
        if line[0:1] in (b"+", b"-", b":"):
            return line[:-2]
        if line[0:1] == b"$":
            n = int(line[1:-2])
            if n < 0:
                return None
            body = self.f.read(n + 2)
            return body[:n]
        raise AssertionError(f"unexpected reply {line[:40]!r}")


def ok(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def unpack_pairs(blob, nq, k):
    a = np.frombuffer(blob, dtype=np.dtype([("id", "<i4"), ("score", "<f4")]))
    assert a.shape[0] == nq * k, f"got {a.shape[0]} pairs, want {nq*k}"
    return a["id"].reshape(nq, k), a["score"].reshape(nq, k)


def dense_scores(c0, c1, q):
    """Materialize the full S² score vector: slot i*S+j = <q0,c0[i]> + <q1,c1[j]>."""
    S, half = c0.shape
    s0 = c0 @ q[:half]
    s1 = c1 @ q[half:]
    return (s0[:, None] + s1[None, :]).reshape(S * S)


# ─────────────────────────────────────────────────────────────────────────
# correctness
# ─────────────────────────────────────────────────────────────────────────

def test_exactness(c, rng):
    print("\n[1] exact top-k vs dense numpy argsort")
    dim, S, k = 64, 16, 5
    n_slots = S * S
    c0 = rng.standard_normal((S, dim // 2), dtype=np.float32)
    c1 = rng.standard_normal((S, dim // 2), dtype=np.float32)

    r = c.call("NEURON.PKM.CREATE", "t1", dim, n_slots)
    ok("CREATE", r == b"+OK", r)
    ok("SETKEYS half 0", c.call("NEURON.PKM.SETKEYS", "t1", 0, c0.tobytes()) == b"+OK")
    ok("SETKEYS half 1", c.call("NEURON.PKM.SETKEYS", "t1", 1, c1.tobytes()) == b"+OK")

    q = rng.standard_normal(dim, dtype=np.float32)
    ids, scores = unpack_pairs(c.call("NEURON.PKM.QUERY", "t1", k, q.tobytes()), 1, k)

    ref = dense_scores(c0, c1, q)
    want = np.argsort(-ref, kind="stable")[:k]
    ok("QUERY ids == dense argsort", list(ids[0]) == list(want), f"{ids[0]} vs {want}")
    ok("QUERY scores match", np.allclose(scores[0], ref[want], atol=2e-4),
       f"{scores[0]} vs {ref[want]}")
    ok("QUERY scores descending", bool(np.all(np.diff(scores[0]) <= 1e-6)))

    # FAST selects on the INT8 mirror, then re-scores in FP32 — ids should still
    # match exactly at this spread, and the scores must be the exact FP32 dots.
    fids, fscores = unpack_pairs(c.call("NEURON.PKM.QUERY", "t1", k, q.tobytes(), "FAST"), 1, k)
    ok("QUERY FAST ids == exact ids", list(fids[0]) == list(want), f"{fids[0]} vs {want}")
    ok("QUERY FAST scores exact", np.allclose(fscores[0], ref[want], atol=2e-4))


def test_multihead(c, rng):
    print("\n[2] multi-head: H queries share one codebook pass")
    dim, S, k, H = 32, 12, 4, 8
    c0 = rng.standard_normal((S, dim // 2), dtype=np.float32)
    c1 = rng.standard_normal((S, dim // 2), dtype=np.float32)
    c.call("NEURON.PKM.CREATE", "t2", dim, S * S)
    c.call("NEURON.PKM.SETKEYS", "t2", 0, c0.tobytes())
    c.call("NEURON.PKM.SETKEYS", "t2", 1, c1.tobytes())

    qs = rng.standard_normal((H, dim), dtype=np.float32)
    ids, scores = unpack_pairs(c.call("NEURON.PKM.QUERY", "t2", k, qs.tobytes()), H, k)

    all_match = True
    for h in range(H):
        ref = dense_scores(c0, c1, qs[h])
        want = np.argsort(-ref, kind="stable")[:k]
        if list(ids[h]) != list(want) or not np.allclose(scores[h], ref[want], atol=2e-4):
            all_match = False
            print(f"        head {h}: {ids[h]} vs {want}")
    ok(f"all {H} heads exact in one call", all_match)

    # nq is inferred from the blob, so odd counts must peel correctly (8→4→2→1).
    peel_ok = True
    for nqi in (1, 2, 3, 5, 7, 9, 16):
        sub = qs[: min(nqi, H)]
        if nqi > H:
            sub = np.concatenate([qs, rng.standard_normal((nqi - H, dim), dtype=np.float32)])
        i2, s2 = unpack_pairs(c.call("NEURON.PKM.QUERY", "t2", k, sub.tobytes()), nqi, k)
        for h in range(nqi):
            ref = dense_scores(c0, c1, sub[h])
            if list(i2[h]) != list(np.argsort(-ref, kind="stable")[:k]):
                peel_ok = False
                print(f"        nq={nqi} head {h} mismatch")
    ok("nq peeling 1/2/3/5/7/9/16 exact", peel_ok)


def test_ffn(c, rng, valtype, name):
    print(f"\n[3] fused FFN, VALTYPE {valtype}")
    dim, S, k, vdim = 32, 10, 6, 24
    n_slots = S * S
    c0 = rng.standard_normal((S, dim // 2), dtype=np.float32)
    c1 = rng.standard_normal((S, dim // 2), dtype=np.float32)
    c.call("NEURON.PKM.CREATE", name, dim, n_slots, "VDIM", vdim, "VALTYPE", valtype)
    c.call("NEURON.PKM.SETKEYS", name, 0, c0.tobytes())
    c.call("NEURON.PKM.SETKEYS", name, 1, c1.tobytes())

    dt = np.float16 if valtype == "F16" else np.float32
    vals = rng.standard_normal((n_slots, vdim)).astype(dt)
    r = c.call("NEURON.PKM.SETVALS", name, 0, n_slots, vals.tobytes())
    ok("SETVALS", r == b":" + str(n_slots).encode(), r)

    q = rng.standard_normal(dim, dtype=np.float32)
    got = np.frombuffer(c.call("NEURON.PKM.FFN", name, k, q.tobytes()), dtype="<f4")

    ref = dense_scores(c0, c1, q)
    top = np.argsort(-ref, kind="stable")[:k]
    sc = ref[top].astype(np.float64)
    w = np.exp(sc - sc.max())
    w /= w.sum()
    want = (w[:, None] * vals[top].astype(np.float64)).sum(0)
    ok(f"FFN {valtype} == softmax-weighted gather", np.allclose(got, want, atol=2e-3),
       f"max|d|={np.abs(got - want).max():.2e}")

    # Temperature sharpens the distribution; TEMP→0 must approach the argmax row.
    sharp = np.frombuffer(c.call("NEURON.PKM.FFN", name, k, q.tobytes(), "TEMP", "0.001"),
                          dtype="<f4")
    ok("FFN TEMP→0 approaches argmax row",
       np.allclose(sharp, vals[top[0]].astype(np.float64), atol=2e-3))
    c.call("NEURON.PKM.DROP", name)


def test_errors(c, rng):
    print("\n[4] validation and lifecycle")
    ok("non-square n_slots rejected",
       c.call("NEURON.PKM.CREATE", "bad", 32, 1000).startswith(b"-ERR"))
    ok("odd dim rejected",
       c.call("NEURON.PKM.CREATE", "bad", 33, 256).startswith(b"-ERR"))
    ok("duplicate name rejected",
       c.call("NEURON.PKM.CREATE", "t1", 64, 256).startswith(b"-ERR"))
    ok("unknown table on QUERY",
       c.call("NEURON.PKM.QUERY", "nope", 4, b"\x00" * 64).startswith(b"-ERR"))

    c.call("NEURON.PKM.CREATE", "half", 32, 64)
    q = np.zeros(32, dtype=np.float32)
    ok("QUERY before both codebooks errors",
       c.call("NEURON.PKM.QUERY", "half", 2, q.tobytes()).startswith(b"-ERR"))
    c.call("NEURON.PKM.SETKEYS", "half", 0, np.zeros((8, 16), dtype=np.float32).tobytes())
    ok("QUERY with one codebook still errors",
       c.call("NEURON.PKM.QUERY", "half", 2, q.tobytes()).startswith(b"-ERR"))
    ok("SETKEYS wrong blob length rejected",
       c.call("NEURON.PKM.SETKEYS", "half", 1, b"\x00" * 7).startswith(b"-ERR"))
    ok("SETKEYS bad half rejected",
       c.call("NEURON.PKM.SETKEYS", "half", 2, np.zeros((8, 16), dtype=np.float32).tobytes())
       .startswith(b"-ERR"))
    c.call("NEURON.PKM.SETKEYS", "half", 1, np.zeros((8, 16), dtype=np.float32).tobytes())
    ok("ragged query blob rejected",
       c.call("NEURON.PKM.QUERY", "half", 2, b"\x00" * 130).startswith(b"-ERR"))
    ok("k=0 rejected", c.call("NEURON.PKM.QUERY", "half", 0, q.tobytes()).startswith(b"-ERR"))
    ok("k over cap rejected",
       c.call("NEURON.PKM.QUERY", "half", 100000, q.tobytes()).startswith(b"-ERR"))
    ok("FFN without VDIM errors",
       c.call("NEURON.PKM.FFN", "half", 2, q.tobytes()).startswith(b"-ERR"))
    ok("SETVALS without VDIM errors",
       c.call("NEURON.PKM.SETVALS", "half", 0, 1, b"\x00" * 4).startswith(b"-ERR"))
    ok("unknown option rejected",
       c.call("NEURON.PKM.QUERY", "half", 2, q.tobytes(), "TURBO").startswith(b"-ERR"))

    # k > n_slots must pad rather than overrun.
    ids, scores = unpack_pairs(c.call("NEURON.PKM.QUERY", "half", 100, q.tobytes()), 1, 100)
    ok("k > n_slots pads with (-1, -inf)",
       int((ids[0] >= 0).sum()) == 64 and bool(np.all(ids[0][64:] == -1)),
       f"live={(ids[0]>=0).sum()}")

    info = c.call("NEURON.PKM.INFO", "half")
    ok("INFO reports geometry", b"s_rows=8" in info and b"n_slots=64" in info, info)
    ok("DROP returns 1", c.call("NEURON.PKM.DROP", "half") == b":1")
    ok("DROP again returns 0", c.call("NEURON.PKM.DROP", "half") == b":0")
    ok("QUERY after DROP errors",
       c.call("NEURON.PKM.QUERY", "half", 2, q.tobytes()).startswith(b"-ERR"))

    print("\n[5] pipelining — the token-skip must not desync the parser")
    c.s.sendall(encode(["NEURON.PKM.INFO", "t1"]) + encode(["PING"])
                + encode(["NEURON.PKM.DROP", "zzz"]) + encode(["PING"]))
    r1, r2, r3, r4 = c._read_one(), c._read_one(), c._read_one(), c._read_one()
    ok("4 pipelined replies stay framed",
       r1.startswith(b"dim=") and r2 == b"+PONG" and r3 == b":0" and r4 == b"+PONG",
       f"{r1[:20]!r} {r2!r} {r3!r} {r4!r}")

    # CREATE/QUERY/FFN scan trailing options in a loop. If that scan ran to the
    # batch-wide token count instead of this command's end, the next pipelined
    # command's name would be read as an unknown option and rejected.
    c.call("NEURON.PKM.DROP", "pipe")
    c.s.sendall(encode(["NEURON.PKM.CREATE", "pipe", 32, 64]) + encode(["PING"]))
    r1, r2 = c._read_one(), c._read_one()
    ok("pipelined CREATE ignores the next command's tokens",
       r1 == b"+OK" and r2 == b"+PONG", f"{r1!r} {r2!r}")

    for h in (0, 1):
        c.call("NEURON.PKM.SETKEYS", "pipe", h, np.zeros((8, 16), dtype=np.float32).tobytes())
    qz = np.zeros(32, dtype=np.float32).tobytes()
    c.s.sendall(encode(["NEURON.PKM.QUERY", "pipe", 2, qz]) + encode(["PING"]))
    r1, r2 = c._read_one(), c._read_one()
    ok("pipelined QUERY ignores the next command's tokens",
       r1 is not None and len(r1) == 16 and r2 == b"+PONG", f"{r1!r} {r2!r}")
    c.call("NEURON.PKM.DROP", "pipe")


# ─────────────────────────────────────────────────────────────────────────
# latency — directly comparable to the gh #146 AI.KNN_LM.* table
# ─────────────────────────────────────────────────────────────────────────

def _p50(c, args, reps):
    for _ in range(10):
        c.call(*args)
    t = []
    for _ in range(reps):
        a = time.perf_counter()
        c.call(*args)
        t.append((time.perf_counter() - a) * 1000)
    return float(np.percentile(t, 50))


def bench(c, rng, reps=200):
    dim, k = 896, 32
    # All timings are wall round-trip from the client, so PING is the floor:
    # anything near it is transport, not search. gh #146 measured its HNSW
    # baseline the same way on the same machine.
    ping = _p50(c, ["PING"], reps)
    print(f"\n[bench] QUERY p50 ms, dim={dim}, k={k}, {reps} reps, single worker."
          f"\n        PING round-trip floor = {ping:.3f} ms — subtract it for server compute.")

    # gh #146 priced 5K/50K/250K/1M; product keys need a perfect square, so use
    # the nearest S² at or above each (71²=5041, 224²=50176, 500²=250000, 1024²).
    print(f"\n  {'slots':>9} {'S':>5} {'exact':>8} {'FAST':>8} {'8-head':>8} "
          f"{'per head':>9} {'HNSW':>7} {'setup':>8}")
    baseline = {5041: "0.47", 50176: "3.27", 250000: "4.04", 1048576: "5.13"}

    for S in (71, 224, 500, 1024):
        n = S * S
        name = f"b{S}"
        c.call("NEURON.PKM.DROP", name)
        r = c.call("NEURON.PKM.CREATE", name, dim, n)
        if not r.startswith(b"+OK"):
            print(f"  CREATE {n} failed: {r}")
            continue
        t0 = time.perf_counter()
        for h in (0, 1):
            cb = rng.standard_normal((S, dim // 2), dtype=np.float32)
            r = c.call("NEURON.PKM.SETKEYS", name, h, cb.tobytes())
            if not r.startswith(b"+OK"):
                print(f"  SETKEYS failed: {r}")
                return
        setup = time.perf_counter() - t0

        def p50(nq, fast):
            qs = rng.standard_normal((nq, dim), dtype=np.float32).tobytes()
            return _p50(c, ["NEURON.PKM.QUERY", name, k, qs] + (["FAST"] if fast else []), reps)

        e, f, m = p50(1, False), p50(1, True), p50(8, False)
        print(f"  {n:>9} {S:>5} {e:>8.3f} {f:>8.3f} {m:>8.3f} {m/8:>9.3f} "
              f"{baseline.get(n,'-'):>7} {setup:>7.2f}s")
        c.call("NEURON.PKM.DROP", name)

    # Fused FFN at a size whose value matrix fits comfortably in RAM
    # (50176 × 2560 FP16 = 257 MB; 1M × 2560 would be 5.1 GB).
    S, vdim = 224, 2560
    n = S * S
    c.call("NEURON.PKM.DROP", "bffn")
    c.call("NEURON.PKM.CREATE", "bffn", dim, n, "VDIM", vdim, "VALTYPE", "F16")
    for h in (0, 1):
        c.call("NEURON.PKM.SETKEYS", "bffn", h,
               rng.standard_normal((S, dim // 2), dtype=np.float32).tobytes())
    chunk = 8192
    for off in range(0, n, chunk):
        m = min(chunk, n - off)
        c.call("NEURON.PKM.SETVALS", "bffn", off, m,
               rng.standard_normal((m, vdim)).astype(np.float16).tobytes())
    q1 = rng.standard_normal((1, dim), dtype=np.float32).tobytes()
    qq = _p50(c, ["NEURON.PKM.QUERY", "bffn", k, q1], reps)
    ff = _p50(c, ["NEURON.PKM.FFN", "bffn", k, q1], reps)
    print(f"\n  fused FFN, {n} slots × vdim {vdim} FP16 (257 MB of value rows):"
          f"\n    QUERY {qq:.3f} ms → client then fetches {k}×{vdim*2//1024} KB itself"
          f"\n    FFN   {ff:.3f} ms → {vdim*4//1024} KB of activations on the wire, values never leave")
    c.call("NEURON.PKM.DROP", "bffn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--seed", type=int, default=1974)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    try:
        c = Conn(port=a.port)
    except OSError as e:
        print(f"cannot reach pion on :{a.port} ({e}) — start ./pion-server --kvcache -w 1")
        return 2

    probe = c.call("NEURON.PKM.INFO", "__probe__")
    if b"requires --kvcache" in probe:
        print("NEURON.PKM.* is gated off — start the server with --kvcache or --inference")
        return 2

    for t in ("t1", "t2", "half", "ffn32", "ffn16"):
        c.call("NEURON.PKM.DROP", t)

    test_exactness(c, rng)
    test_multihead(c, rng)
    test_ffn(c, rng, "F32", "ffn32")
    test_ffn(c, rng, "F16", "ffn16")
    test_errors(c, rng)
    for t in ("t1", "t2"):
        c.call("NEURON.PKM.DROP", t)

    if a.bench:
        bench(c, rng)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
