#!/usr/bin/env python3
"""V.EXPORT (gh #468): a prompt prefix's K/V as a file a local client maps.

The export lane replaces a loopback-TCP copy of the prefix with a safetensors
file the client opens with mx.load. What a client depends on, checked here
without MLX (the file is parsed with numpy):

  - every tensor equals what V.FETCH BATCH returns, transposed to
    [1, n_kv_heads, N, head_dim] and rounded to fp16, for an fp16 session
    (bit-exact) and an int8 one (its dequantized values, rounded);
  - a sub-range [start, end) exports exactly those tokens;
  - asking twice returns the same file; storing more into either session
    gives a new file and deletes the old one, so a hit can never map a stale
    prefix;
  - the server picks the path: it lies in the export directory, the
    directory is 0700 and the file 0600;
  - bad requests are refused with an error and leave the connection usable.

Requires: ./pion-server --kvcache -w 1 (the runner starts it).
"""
import argparse
import json
import os
import stat
import struct
import sys

import numpy as np
import redis

FAILS = []
H, D, LAYERS, N = 4, 32, 3, 37          # kv_dim 128; an odd token count


def check(cond, what):
    print(("  PASS  " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def read_safetensors(path):
    with open(path, "rb") as f:
        raw = f.read()
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        assert meta["dtype"] == "F16", meta
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=np.float16).reshape(meta["shape"])
    return out


def store(r, sid, fmt, rng):
    r.execute_command("V.CREATE", sid, str(H * D), "VQUANT", fmt)
    rows = []
    for li in range(LAYERS):
        x = rng.standard_normal((N, H * D)).astype(np.float32)
        r.execute_command("V.STOREBATCH", sid, str(li), "0", str(N), x.tobytes())
        rows.append(x)
    return rows


def fetched(r, sid, start, end):
    """V.FETCH BATCH as fp32 rows per layer (the TCP lane's view)."""
    reply = r.execute_command("V.FETCH", sid, "BATCH", str(start), str(end), str(LAYERS))
    return [np.frombuffer(b, dtype=np.float32).reshape(end - start, H * D) for b in reply]


def as_heads(rows):
    """[N, H*D] -> [1, H, N, D], rounded to fp16."""
    return rows.reshape(rows.shape[0], H, D).transpose(1, 0, 2)[None].astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    a = ap.parse_args()
    r = redis.Redis(port=a.port)
    rng = np.random.default_rng(468)
    sfx = str(os.getpid())

    for fmt in ("fp16", "int8"):
        k, v = f"exp_{fmt}_{sfx}_pk", f"exp_{fmt}_{sfx}_pv"
        store(r, k, fmt, rng)
        store(r, v, fmt, rng)
        path = r.execute_command("V.EXPORT", k, v, "0", str(N), str(LAYERS), str(H), str(D)).decode()
        check(os.path.isabs(path) and os.path.isfile(path), f"{fmt}: V.EXPORT returns an absolute path to a file")
        t = read_safetensors(path)
        check(sorted(t) == sorted([f"k.{i}" for i in range(LAYERS)] + [f"v.{i}" for i in range(LAYERS)]),
              f"{fmt}: one k and one v tensor per layer")
        fk, fv = fetched(r, k, 0, N), fetched(r, v, 0, N)
        same = all(np.array_equal(t[f"k.{i}"], as_heads(fk[i])) and np.array_equal(t[f"v.{i}"], as_heads(fv[i]))
                   for i in range(LAYERS))
        check(same, f"{fmt}: every tensor equals V.FETCH BATCH, transposed to [1,H,N,D] and rounded to fp16")
        check(all(t[f"k.{i}"].shape == (1, H, N, D) for i in range(LAYERS)), f"{fmt}: shape [1, {H}, {N}, {D}]")

        # A sub-range exports exactly those tokens.
        sub = r.execute_command("V.EXPORT", k, v, "5", "20", str(LAYERS), str(H), str(D)).decode()
        ts = read_safetensors(sub)
        fk_sub = fetched(r, k, 5, 20)
        check(all(np.array_equal(ts[f"k.{i}"], as_heads(fk_sub[i])) for i in range(LAYERS)),
              f"{fmt}: tokens [5, 20) export as exactly those tokens")

        # Same request, unchanged sessions: the same file.
        again = r.execute_command("V.EXPORT", k, v, "0", str(N), str(LAYERS), str(H), str(D)).decode()
        check(again == path, f"{fmt}: asking again returns the same file")

        # A store into the V side: a new file, and the old one is gone.
        x = rng.standard_normal((N, H * D)).astype(np.float32)
        r.execute_command("V.STOREBATCH", v, "1", "0", str(N), x.tobytes())
        newer = r.execute_command("V.EXPORT", k, v, "0", str(N), str(LAYERS), str(H), str(D)).decode()
        check(newer != path, f"{fmt}: after a store the export is a new file")
        check(not os.path.exists(path), f"{fmt}: the stale file is deleted")
        tn = read_safetensors(newer)
        check(np.array_equal(tn["v.1"], as_heads(fetched(r, v, 0, N)[1])), f"{fmt}: the new file holds the new values")

        d = os.path.dirname(newer)
        check(stat.S_IMODE(os.stat(d).st_mode) == 0o700, f"{fmt}: export directory is 0700")
        check(stat.S_IMODE(os.stat(newer).st_mode) == 0o600, f"{fmt}: export file is 0600")

    # Refusals, each leaving the connection usable.
    k, v = f"exp_fp16_{sfx}_pk", f"exp_fp16_{sfx}_pv"
    bad = [
        (("V.EXPORT", k, v, "0", str(N), str(LAYERS), str(H)), "missing an argument"),
        (("V.EXPORT", "nope_pk", v, "0", str(N), str(LAYERS), str(H), str(D)), "unknown session"),
        (("V.EXPORT", k, v, "9", "9", str(LAYERS), str(H), str(D)), "empty range"),
        (("V.EXPORT", k, v, "0", str(N), str(LAYERS + 5), str(H), str(D)), "more layers than stored"),
        (("V.EXPORT", k, v, "0", str(N), str(LAYERS), "3", str(D)), "heads * head_dim != stored width"),
        (("V.EXPORT", k, v, "0", str(N), "0", str(H), str(D)), "zero layers"),
    ]
    for cmd, what in bad:
        try:
            r.execute_command(*cmd)
            ok = False
        except redis.ResponseError:
            ok = True
        check(ok and r.ping(), f"refused: {what}")

    print(f"{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
