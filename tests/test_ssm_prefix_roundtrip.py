#!/usr/bin/env python3
"""gh #65 — SSM.PREFIX.STORE/FETCH/DROP round-trip correctness gate.

Server-side substrate for SSM-family prefix sharing (Mamba, RWKV, etc.).
The wire just stores opaque byte blobs keyed by (session_id, layer_id);
the consumer chooses serialization.

Two checks:
  [1] Lossless byte round-trip: STORE a random blob, FETCH it, compare
      byte-for-byte. Covers the wire framing + server-side malloc/memcpy.
  [2] End-to-end SSM state preservation: snapshot a Mamba-130M layer cache,
      serialize, STORE, drop local, FETCH, deserialize, restore — decode
      tokens must match a no-snapshot baseline. This is the same gate the
      gh #64 in-process drift check ran, now passing through the wire.

Requires: ./pion-server --kvcache -w 1 (no Metal needed; SSM.PREFIX.* is
pure host-side byte storage).
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


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


def _first_complete(d: bytes):
    if len(d) < 3:
        return None
    p = d[0:1]
    nl = d.find(b"\r\n")
    if nl < 0:
        return None
    if p in (b"+", b"-", b":"):
        return nl + 2
    if p == b"$":
        ls = d[1:nl].decode()
        if ls == "-1":
            return nl + 2
        n = int(ls)
        need = nl + 2 + n + 2
        return need if len(d) >= need else None
    return nl + 2


class Conn:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(60)
        self.s.connect((HOST, PORT))
        self.buf = b""

    def _read(self) -> bytes:
        while True:
            done = _first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.s.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    def call(self, *parts):
        self.s.sendall(_encode(parts))
        return self._read()


def parse_bulk(r: bytes) -> bytes | None:
    if r.startswith(b"$-1"):
        return None
    if not r.startswith(b"$"):
        raise RuntimeError(f"expected bulk reply, got: {r[:80]!r}")
    nl = r.find(b"\r\n")
    blen = int(r[1:nl])
    return r[nl + 2:nl + 2 + blen]


# ── Optional Mamba [2] gate (skip cleanly if MLX unavailable) ───────────


def serialize_arrays_cache(cache_entries):
    """[arr1, arr2, ...] of mx.arrays → bytes blob.

    Layout (matches the format suggested in the gh #65 issue body):
        uint32 version = 1
        uint32 n_arrays
        for each array:
            uint32 ndim
            uint32[ndim] shape
            uint32 dtype_code (0=fp32, 1=fp16)
            raw bytes
    """
    import mlx.core as mx
    # bfloat16 isn't in numpy. We upcast to fp32 for serialization (lossless
    # for bf16 values since fp32 has strictly more precision in both mantissa
    # and exponent range). On deserialize we cast back to bf16 to preserve
    # the array dtype, which is identity for the bf16 value set. Wire pays
    # 2× the bytes for bf16 layers; acceptable for v1 since SSM.PREFIX.* is
    # a per-session-cold-prefill cost, not per-decode.
    dt_code = {mx.float32: 0, mx.float16: 1, mx.bfloat16: 2}
    out = [struct.pack("<II", 1, len(cache_entries))]
    for arr in cache_entries:
        if arr is None:
            out.append(struct.pack("<IIII", 0, 0, 0, 0))
            continue
        mx.eval(arr)
        shape = tuple(arr.shape)
        ndim = len(shape)
        if arr.dtype not in dt_code:
            raise NotImplementedError(f"dtype {arr.dtype} not in serializer dict")
        out.append(struct.pack("<I", ndim))
        out.append(struct.pack("<" + "I" * ndim, *shape))
        out.append(struct.pack("<I", dt_code[arr.dtype]))
        # mx.array → bytes. bfloat16 round-trips via fp32 (numpy lacks bf16).
        if arr.dtype == mx.bfloat16:
            arr_np = np.array(arr.astype(mx.float32))
        else:
            arr_np = np.array(arr)
        out.append(arr_np.tobytes())
    return b"".join(out)


def deserialize_arrays_cache(blob: bytes):
    import mlx.core as mx
    # dt_code 2 (bfloat16) is stored on the wire as fp32 bytes; we deserialize
    # to numpy fp32 and cast back to bf16 in MLX. Round-trip is value-exact
    # since fp32 has strict superset of bf16 representable values.
    dt_decode = {
        0: (mx.float32, np.float32, False),   # (mx_dtype, np_dtype, cast_back_to_bf16)
        1: (mx.float16, np.float16, False),
        2: (mx.bfloat16, np.float32, True),   # serialized as fp32, restore as bf16
    }
    version, n = struct.unpack_from("<II", blob, 0)
    if version != 1:
        raise ValueError(f"unsupported SSM-blob version {version}")
    off = 8
    entries = []
    for _ in range(n):
        (ndim,) = struct.unpack_from("<I", blob, off); off += 4
        if ndim == 0:
            off += 12
            entries.append(None)
            continue
        shape = struct.unpack_from("<" + "I" * ndim, blob, off); off += 4 * ndim
        (dt_code,) = struct.unpack_from("<I", blob, off); off += 4
        mx_dtype, np_dtype, cast_back = dt_decode[dt_code]
        nbytes = int(np.prod(shape)) * np_dtype().itemsize
        arr_np = np.frombuffer(blob, dtype=np_dtype, count=int(np.prod(shape)), offset=off)
        off += nbytes
        arr_mx = mx.array(arr_np.reshape(shape))
        if cast_back:
            arr_mx = arr_mx.astype(mx.bfloat16)
        entries.append(arr_mx)
    return entries


def main(args) -> int:
    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError:
        print("FAIL: pion not reachable. Start with: ./pion-server --kvcache -w 1")
        return 2

    c = Conn()
    fail = False

    # ── [1] Lossless byte round-trip ──
    print("[1] Lossless byte round-trip (random 1 MB blob)")
    sid = "ssm_roundtrip_v1"
    blob_in = np.random.bytes(1024 * 1024)
    r = c.call("SSM.PREFIX.STORE", sid, "0", blob_in)
    if not r.startswith(b"+OK"):
        print(f"   FAIL: STORE rejected: {r[:80]!r}")
        return 1
    r = c.call("SSM.PREFIX.FETCH", sid, "0")
    blob_out = parse_bulk(r)
    if blob_out is None:
        print(f"   FAIL: FETCH returned null")
        return 1
    if blob_out != blob_in:
        print(f"   FAIL: round-trip mismatch (in {len(blob_in)} B, out {len(blob_out)} B)")
        diffs = sum(1 for a, b in zip(blob_in, blob_out) if a != b)
        print(f"   first 64 bytes diff count: {diffs}")
        fail = True
    else:
        print(f"   OK: {len(blob_in)} bytes round-tripped exactly")

    # Repeat STORE+FETCH with a different blob — verifies overwrite
    blob_in2 = np.random.bytes(2048)
    c.call("SSM.PREFIX.STORE", sid, "0", blob_in2)
    r = c.call("SSM.PREFIX.FETCH", sid, "0")
    blob_out2 = parse_bulk(r)
    if blob_out2 != blob_in2:
        print(f"   FAIL: overwrite round-trip mismatch")
        fail = True
    else:
        print(f"   OK: overwrite + shrink ({len(blob_in)} B → {len(blob_in2)} B) round-tripped exactly")

    # DROP and verify missing
    c.call("SSM.PREFIX.DROP", sid, "0")
    r = c.call("SSM.PREFIX.FETCH", sid, "0")
    if not r.startswith(b"$-1"):
        print(f"   FAIL: DROP didn't drop — got {r[:80]!r}")
        fail = True
    else:
        print("   OK: DROP works (FETCH after DROP returns null)")

    # Multi-layer DROP ALL
    c.call("SSM.PREFIX.STORE", sid, "0", b"layer0")
    c.call("SSM.PREFIX.STORE", sid, "5", b"layer5")
    c.call("SSM.PREFIX.STORE", sid, "23", b"layer23")
    c.call("SSM.PREFIX.DROP", sid)   # no layer_id = drop all
    r0 = parse_bulk(c.call("SSM.PREFIX.FETCH", sid, "0"))
    r5 = parse_bulk(c.call("SSM.PREFIX.FETCH", sid, "5"))
    r23 = parse_bulk(c.call("SSM.PREFIX.FETCH", sid, "23"))
    if r0 is None and r5 is None and r23 is None:
        print("   OK: DROP without layer_id removed all layers for the session")
    else:
        print(f"   FAIL: DROP-all left some layers behind: r0={r0}, r5={r5}, r23={r23}")
        fail = True

    # ── [2] End-to-end Mamba state round-trip ──
    print("\n[2] End-to-end Mamba-130M state round-trip via wire")
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        print("   SKIP: mlx_lm not available")
        print(f"\n{'PASS' if not fail else 'FAIL'} — gh #65 SSM.PREFIX round-trip ([2] skipped)")
        return 1 if fail else 0

    print(f"   loading {args.model}...")
    model, tok = load(args.model)
    prefix_ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 16)[:args.prefix_len]
    print(f"   prefix len = {len(prefix_ids)}")

    # Baseline: encode prefix[:-1] in fresh cache, feed prefix[-1] + decode 64.
    sid2 = "ssm_roundtrip_v2_mamba"
    encode_ids = prefix_ids[:-1]
    first_dec_in = prefix_ids[-1]

    cache_base = model.make_cache()
    _ = model(mx.array([encode_ids]), cache=cache_base); mx.eval(_)
    x = mx.array([[first_dec_in]])
    out = model(x, cache=cache_base); mx.eval(out)
    tok_base = int(mx.argmax(out[0, -1]).item())
    tokens_base = [tok_base]
    for _ in range(args.decode_tokens - 1):
        nxt = mx.array([[tok_base]])
        out = model(nxt, cache=cache_base); mx.eval(out)
        tok_base = int(mx.argmax(out[0, -1]).item())
        tokens_base.append(tok_base)
    print(f"   baseline first 6: {tokens_base[:6]}")

    # Wire round-trip: encode → snapshot → STORE per layer → DROP local →
    # FETCH per layer → restore → feed first_dec_in + decode 64.
    cache_snap = model.make_cache()
    _ = model(mx.array([encode_ids]), cache=cache_snap); mx.eval(_)
    print(f"   snapshotting + STORE-ing {len(cache_snap)} layers...")
    for li, c_layer in enumerate(cache_snap):
        if c_layer is None: continue
        blob = serialize_arrays_cache(c_layer.cache)
        r = c.call("SSM.PREFIX.STORE", sid2, str(li), blob)
        if not r.startswith(b"+OK"):
            print(f"   FAIL: STORE layer {li}: {r[:80]!r}")
            return 1
    del cache_snap

    # Restore from wire
    cache_reloaded = model.make_cache()
    for li, c_layer in enumerate(cache_reloaded):
        if c_layer is None: continue
        r = c.call("SSM.PREFIX.FETCH", sid2, str(li))
        blob = parse_bulk(r)
        if blob is None:
            print(f"   FAIL: FETCH layer {li} returned null")
            return 1
        c_layer.cache = deserialize_arrays_cache(blob)

    x = mx.array([[first_dec_in]])
    out = model(x, cache=cache_reloaded); mx.eval(out)
    tok_rel = int(mx.argmax(out[0, -1]).item())
    tokens_reload = [tok_rel]
    for _ in range(args.decode_tokens - 1):
        nxt = mx.array([[tok_rel]])
        out = model(nxt, cache=cache_reloaded); mx.eval(out)
        tok_rel = int(mx.argmax(out[0, -1]).item())
        tokens_reload.append(tok_rel)
    print(f"   reload   first 6: {tokens_reload[:6]}")

    n_match = sum(1 for a, b in zip(tokens_base, tokens_reload) if a == b)
    n_total = len(tokens_base)
    print(f"   token agreement: {n_match}/{n_total} ({n_match/n_total*100:.1f}%)")
    if n_match != n_total:
        first_div = next(i for i, (a, b) in enumerate(zip(tokens_base, tokens_reload)) if a != b)
        print(f"   FAIL: first divergence at step {first_div}")
        fail = True
    else:
        print(f"   OK: {n_total}/{n_total} bit-perfect decode trajectory through the wire (same as in-process)")

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #65 SSM.PREFIX round-trip on {args.model}")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/mamba-130m-hf-f32",
                    help="Family-agnostic — works with any mlx-lm model whose cache is ArraysCache.")
    ap.add_argument("--prefix-len", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    args = ap.parse_args()
    sys.exit(main(args))
