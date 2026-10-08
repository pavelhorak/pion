#!/usr/bin/env python3
"""gh #400: the exact FP32 scans behind VSIM and kNN-LM's brute force, checked
against numpy at dimensions that exercise every part of the kernel.

VSIM scores a whole set with one GEMV (Accelerate on macOS) or the
four-chain FMA kernel; kNN-LM's brute force uses the four-chain L2 kernel.
The kernels walk 16 floats at a time, then 4, then one, so the dims here
include multiples of 16, a 4-remainder and a scalar tail. The scores move
only by reassociation, so each check is a tolerance, and the rankings must
match numpy's wherever numpy's own gaps are wider than that tolerance.

Requires: ./pion-server --kvcache -w 1
"""
import argparse
import struct
import sys

import numpy as np
import redis

DIMS = (16, 20, 23, 384, 1537)
FAILS = []


def check(cond, what):
    print(("  PASS  " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def vsim(r, rng, dim):
    n = 300
    key = f"fp32scan:vs:{dim}"
    r.delete(key)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    p = r.pipeline(transaction=False)
    for i in range(n):
        p.execute_command("VADD", key, "FP32", X[i].tobytes(), f"e{i}")
    p.execute()
    # A tombstone: VREM must keep the removed slot out of the answer.
    r.execute_command("VREM", key, "e0")
    q = rng.standard_normal(dim).astype(np.float32)
    reply = r.execute_command("VSIM", key, "FP32", q.tobytes(), "COUNT", 10, "WITHSCORES")
    if isinstance(reply, dict):
        pairs = list(reply.items())
    else:
        pairs = [(reply[i], reply[i + 1]) for i in range(0, len(reply), 2)]
    got = [(k.decode() if isinstance(k, bytes) else k, float(v)) for k, v in pairs]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    want = (1.0 + Xn.astype(np.float64) @ qn.astype(np.float64)) / 2.0
    want[0] = -1.0                                   # removed
    order = np.argsort(-want)
    check(len(got) == 10, f"VSIM dim={dim}: 10 results")
    check(all(name != "e0" for name, _ in got), f"VSIM dim={dim}: the removed element is not returned")
    worst = max(abs(s - want[int(name[1:])]) for name, s in got)
    check(worst < 1e-5, f"VSIM dim={dim}: scores within 1e-5 of numpy (worst {worst:.2e})")
    # Rank agreement wherever numpy separates neighbours by more than the tolerance.
    ok = True
    for i, (name, _) in enumerate(got):
        exp = int(order[i])
        if int(name[1:]) != exp and abs(want[int(name[1:])] - want[exp]) > 1e-5:
            ok = False
    check(ok, f"VSIM dim={dim}: same top-10 order as numpy")
    r.delete(key)


def knn(r, rng, dim):
    n = 400
    ds = f"fp32scan_knn_{dim}"
    try:
        r.execute_command("AI.KNN_LM.DROP", ds)
    except redis.ResponseError:
        pass
    r.execute_command("AI.KNN_LM.CREATE", ds, str(dim), str(n + 10))
    X = rng.standard_normal((n, dim)).astype(np.float32)
    tok = np.arange(n, dtype=np.int32).tobytes()
    r.execute_command("AI.KNN_LM.STOREBATCH", ds, str(n), tok, X.tobytes())
    q = rng.standard_normal(dim).astype(np.float32)
    k = 8
    blob = r.execute_command("AI.KNN_LM.QUERY", ds, str(k), q.tobytes())
    rows = [struct.unpack_from("<if", blob, 8 * i) for i in range(len(blob) // 8)]
    d2 = ((X.astype(np.float64) - q.astype(np.float64)) ** 2).sum(axis=1)
    order = np.argsort(d2)
    check(len(rows) == k, f"KNN_LM dim={dim}: {k} results")
    worst = max(abs(dist - d2[tid]) / max(1.0, d2[tid]) for tid, dist in rows)
    check(worst < 1e-5, f"KNN_LM dim={dim}: distances within 1e-5 (relative) of numpy (worst {worst:.2e})")
    check([tid for tid, _ in rows] == [int(t) for t in order[:k]],
          f"KNN_LM dim={dim}: same top-{k} as numpy")
    r.execute_command("AI.KNN_LM.DROP", ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    a = ap.parse_args()
    r = redis.Redis(port=a.port)
    rng = np.random.default_rng(400)
    for dim in DIMS:
        vsim(r, rng, dim)
        knn(r, rng, dim)
    print(f"{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
