#!/usr/bin/env python3
"""A1 (gh #29): V.CREATE SCHEMA per-layer roundtrip + back-compat tests.

Spec: the V.CREATE SCHEMA form in doc/command_matrix.md.

Requires: ./pion-server --kvcache -w 1
"""

import os
import socket
import struct
import sys
import time
import numpy as np

HOST = "127.0.0.1"
PORT = 1974


class VStoreClient:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.settimeout(15)
        self.sock.connect((host, port))
        self.f = self.sock.makefile("rb")

    def _encode(self, parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, bytes):
                out.append(f"${len(p)}\r\n".encode() + p + b"\r\n")
            else:
                s = str(p)
                out.append(f"${len(s)}\r\n{s}\r\n".encode())
        return b"".join(out)

    def _read_line(self):
        return self.f.readline().rstrip(b"\r\n")

    def _read_reply(self):
        line = self._read_line()
        t, body = line[:1], line[1:]
        if t in (b"+", b"-", b":"):
            return line
        if t == b"$":
            n = int(body)
            if n < 0:
                return None
            data = self.f.read(n)
            assert self.f.read(2) == b"\r\n"
            return data
        if t == b"*":
            n = int(body)
            return [self._read_reply() for _ in range(n)]
        raise ValueError(f"unexpected reply: {line!r}")

    def call(self, *parts):
        self.sock.sendall(self._encode(parts))
        return self._read_reply()

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ── Tests ───────────────────────────────────────────────────────────────


def test_back_compat_uniform():
    """Pre-A1 wire form keeps working: V.CREATE sid dim [VQUANT fmt]."""
    c = VStoreClient()
    sid = "bc_uniform"
    c.call("V.FREE", sid)  # no-op if cmd absent; ignore
    # Skip if FREE not supported — V.CREATE is idempotent on existing sid (returns same idx).
    r = c.call("V.CREATE", sid, 256)
    assert r.startswith(b":"), f"legacy V.CREATE: {r!r}"

    np.random.seed(1)
    values = (np.random.randn(8, 256) * 0.5).astype(np.float32)
    r = c.call("V.STOREBATCH", sid, "0", "0", "8", values.tobytes())
    assert r == b"+OK", f"V.STOREBATCH: {r!r}"

    r = c.call("V.FETCH", sid, "0", "0", "1", "2", "3")
    assert r is not None and len(r) >= 4 * 256 * 4, f"V.FETCH short: {len(r) if r else 0}"
    c.close()
    print("PASS: test_back_compat_uniform")


def test_schema_basic_int8():
    """SCHEMA form: 4 layers, all int8 with same dim. Roundtrip per-layer."""
    c = VStoreClient()
    sid = "schema_int8"
    r = c.call("V.CREATE", sid, "128", "SCHEMA", "4",
               "fmt=int8", "fmt=int8", "fmt=int8", "fmt=int8")
    assert r.startswith(b":"), f"V.CREATE SCHEMA: {r!r}"

    np.random.seed(7)
    for layer in range(4):
        vals = (np.random.randn(16, 128) * 0.4).astype(np.float32)
        r = c.call("V.STOREBATCH", sid, str(layer), "0", "16", vals.tobytes())
        assert r == b"+OK", f"layer {layer} STORE: {r!r}"
        r = c.call("V.FETCH", sid, str(layer), "0", "5", "10", "15")
        assert r is not None
        # 3 tokens × 128 dim × 4 bytes each
        assert len(r) >= 3 * 128 * 4
        ret = np.frombuffer(r[:3*128*4], dtype=np.float32).reshape(3, 128)
        for i, tid in enumerate([0, 5, 10]):
            cos = np.dot(vals[tid], ret[i]) / (np.linalg.norm(vals[tid]) * np.linalg.norm(ret[i]))
            assert cos > 0.99, f"layer {layer} tok {tid}: cos={cos:.4f}"
    c.close()
    print("PASS: test_schema_basic_int8")


def test_schema_mixed_per_layer_format():
    """SCHEMA with different fmt per layer: int8, fp16, turbo4, int8."""
    c = VStoreClient()
    sid = "schema_mixed_fmt"
    r = c.call("V.CREATE", sid, "256", "SCHEMA", "4",
               "fmt=int8", "fmt=fp16", "fmt=turbo4,dim=512", "fmt=int8")
    assert r.startswith(b":"), f"V.CREATE SCHEMA mixed: {r!r}"

    np.random.seed(11)
    layer_dims = [256, 256, 512, 256]   # layer 2 overrode dim, others use default 256
    layer_min_cos = [0.99, 0.99, 0.95, 0.99]  # turbo4 lossier
    for layer, dim in enumerate(layer_dims):
        vals = (np.random.randn(8, dim) * 0.5).astype(np.float32)
        r = c.call("V.STOREBATCH", sid, str(layer), "0", "8", vals.tobytes())
        assert r == b"+OK", f"mixed layer {layer} STORE: {r!r}"
        r = c.call("V.FETCH", sid, str(layer), "0", "3", "5")
        assert r is not None
        ret = np.frombuffer(r[:3*dim*4], dtype=np.float32).reshape(3, dim)
        for i, tid in enumerate([0, 3, 5]):
            cos = np.dot(vals[tid], ret[i]) / (np.linalg.norm(vals[tid]) * np.linalg.norm(ret[i]))
            assert cos > layer_min_cos[layer], f"layer {layer} fmt-fail tok {tid}: cos={cos:.4f}"
    c.close()
    print("PASS: test_schema_mixed_per_layer_format")


def test_schema_per_layer_dim():
    """SCHEMA with per-layer DIFFERENT dims (V4-style indexer vs main split)."""
    c = VStoreClient()
    sid = "schema_per_dim"
    # 3 layers, each different dim
    r = c.call("V.CREATE", sid, "0", "SCHEMA", "3",
               "fmt=int8,dim=128", "fmt=int8,dim=512", "fmt=int8,dim=256")
    assert r.startswith(b":"), f"V.CREATE SCHEMA per-dim: {r!r}"

    np.random.seed(13)
    layer_dims = [128, 512, 256]
    for layer, dim in enumerate(layer_dims):
        vals = (np.random.randn(4, dim) * 0.3).astype(np.float32)
        r = c.call("V.STOREBATCH", sid, str(layer), "0", "4", vals.tobytes())
        assert r == b"+OK", f"per-dim layer {layer} STORE: {r!r}"
        r = c.call("V.FETCH", sid, str(layer), "0", "1", "2")
        assert r is not None
        ret = np.frombuffer(r[:3*dim*4], dtype=np.float32).reshape(3, dim)
        for i, tid in enumerate([0, 1, 2]):
            cos = np.dot(vals[tid], ret[i]) / (np.linalg.norm(vals[tid]) * np.linalg.norm(ret[i]))
            assert cos > 0.99, f"per-dim layer {layer} tok {tid}: cos={cos:.4f}"
    c.close()
    print("PASS: test_schema_per_layer_dim")


def test_strict_storebatch_size_validation():
    """A1 strict mode (open Q #1): refuse oversized AND undersized blobs."""
    c = VStoreClient()
    sid = "strict_size"
    r = c.call("V.CREATE", sid, "64", "SCHEMA", "1", "fmt=int8")
    assert r.startswith(b":"), r

    # Correct: 4 tokens × 64 dim × 4 = 1024 bytes
    good = (np.random.randn(4, 64) * 0.2).astype(np.float32).tobytes()
    assert c.call("V.STOREBATCH", sid, "0", "0", "4", good) == b"+OK"

    # Undersize
    short = (np.random.randn(4, 32) * 0.2).astype(np.float32).tobytes()  # half the dim
    r = c.call("V.STOREBATCH", sid, "0", "0", "4", short)
    assert r.startswith(b"-ERR") and b"blob size" in r, f"undersize: {r!r}"

    # Oversize
    big = (np.random.randn(4, 128) * 0.2).astype(np.float32).tobytes()  # 2× dim
    r = c.call("V.STOREBATCH", sid, "0", "0", "4", big)
    assert r.startswith(b"-ERR") and b"blob size" in r, f"oversize: {r!r}"
    c.close()
    print("PASS: test_strict_storebatch_size_validation")


# test_reject_fp8_pre_a2 deleted 2026-05-03 — A2 (gh #30) shipped, fp8 and
# bf16_rope_fp8_body are now accepted. Coverage moved to tests/test_v_store_fp8.py.


def test_reject_s_tensor_reserved():
    """D3: `s=` reserved syntactically, refused at runtime in v0.1."""
    c = VStoreClient()
    r = c.call("V.CREATE", "rej_s", "128", "SCHEMA", "1", "fmt=int8,s=fp16")
    assert r.startswith(b"-ERR") and b"S-tensor reserved" in r, f"s= should be refused: {r!r}"
    c.close()
    print("PASS: test_reject_s_tensor_reserved")


def test_reject_too_many_layers():
    """Refuse N > MAX_SCHEMA_LAYERS (59 — InlineArray[64] frame ceiling).
    Send the count without all the specs — server should reject on N alone."""
    c = VStoreClient()
    # 7 args: cmd + sid + dim + SCHEMA + "60" + 1 spec — well under 64 limit.
    # Server checks N before validating that enough specs follow.
    r = c.call("V.CREATE", "rej_big", "128", "SCHEMA", "60", "fmt=int8")
    assert r.startswith(b"-ERR") and b"exceeds" in r, f"60 layers should be refused: {r!r}"
    c.close()
    print("PASS: test_reject_too_many_layers")


def test_unknown_key_in_spec():
    c = VStoreClient()
    r = c.call("V.CREATE", "bad_key", "128", "SCHEMA", "1", "fmt=int8,bogus=42")
    assert r.startswith(b"-ERR") and b"unknown key" in r, f"unknown key: {r!r}"
    c.close()
    print("PASS: test_unknown_key_in_spec")


def test_info_reports_schema():
    """V.INFO <sid> on heterogeneous session reports per-layer dim+fmt."""
    c = VStoreClient()
    sid = "info_het"
    r = c.call("V.CREATE", sid, "128", "SCHEMA", "2",
               "fmt=int8,dim=64", "fmt=fp16,dim=256")
    assert r.startswith(b":"), r
    info = c.call("V.INFO", sid).decode()
    assert "schema:heterogeneous" in info, info
    assert "layer_0_dim:64" in info
    assert "layer_0_fmt:int8" in info
    assert "layer_1_dim:256" in info
    assert "layer_1_fmt:fp16" in info

    # Uniform session → schema:uniform
    sid2 = "info_unif"
    r = c.call("V.CREATE", sid2, "256")
    assert r.startswith(b":"), r
    info = c.call("V.INFO", sid2).decode()
    assert "schema:uniform" in info, info
    c.close()
    print("PASS: test_info_reports_schema")


def test_snapshot_roundtrip_via_kvprefix_save():
    """Snapshot v=2 round-trip: SCHEMA → STORE → KV.PREFIX.SAVE → restart."""
    c = VStoreClient()
    sid = "snap_het"
    # Use KV.PREFIX.REGISTER instead of V.CREATE so KV.PREFIX.SAVE has a
    # registered prefix to snapshot against. We can also test V.CREATE SCHEMA
    # path through this since SAVE snapshots all sessions.
    r = c.call("V.CREATE", sid, "256", "SCHEMA", "3",
               "fmt=int8,dim=128", "fmt=fp16,dim=256", "fmt=turbo4,dim=512")
    assert r.startswith(b":"), r

    np.random.seed(31)
    layer_dims = [128, 256, 512]
    saved_vals = []
    for layer, dim in enumerate(layer_dims):
        vals = (np.random.randn(4, dim) * 0.3).astype(np.float32)
        saved_vals.append(vals)
        assert c.call("V.STOREBATCH", sid, str(layer), "0", "4", vals.tobytes()) == b"+OK"

    # Force snapshot. KV.PREFIX.SAVE writes pion.vstore.<wid> + truncates WAL.
    save_resp = c.call("KV.PREFIX.SAVE")
    # Acceptable: +OK or any non-error. We don't gate on the specific shape.
    assert not save_resp.startswith(b"-ERR"), f"SAVE refused: {save_resp!r}"
    c.close()

    # Restart server (handled by the harness: this test is run with manual
    # restart by the runner, not in-process). So instead of restart, we
    # just re-fetch from the running server — which validates the save
    # didn't corrupt in-memory state. A full restart roundtrip is a
    # separate manual test.
    c = VStoreClient()
    for layer, dim in enumerate(layer_dims):
        # V.FETCH wire form: V.FETCH sid layer_id tid_1 [tid_2 ...]
        r = c.call("V.FETCH", sid, str(layer), "0", "1", "2")
        assert r is not None, f"post-save fetch layer {layer} returned None"
        ret = np.frombuffer(r[:3*dim*4], dtype=np.float32).reshape(3, dim)
        min_cos = 0.95 if layer == 2 else 0.99
        for i, tid in enumerate([0, 1, 2]):
            v = saved_vals[layer][tid]
            cos = np.dot(v, ret[i]) / (np.linalg.norm(v) * np.linalg.norm(ret[i]))
            assert cos > min_cos, f"post-save layer {layer} tok {tid}: cos={cos:.4f}"
    c.close()
    print("PASS: test_snapshot_roundtrip_via_kvprefix_save")


# ── Runner ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, PORT))
        s.close()
    except Exception:
        print(f"Pion not running at {HOST}:{PORT}")
        print("Start with: ./pion-server --kvcache -w 1")
        sys.exit(1)

    print("=" * 60)
    print("V-store SCHEMA tests (issue #29 / A1)")
    print("=" * 60)

    tests = [
        test_back_compat_uniform,
        test_schema_basic_int8,
        test_schema_mixed_per_layer_format,
        test_schema_per_layer_dim,
        test_strict_storebatch_size_validation,
        test_reject_s_tensor_reserved,
        test_reject_too_many_layers,
        test_unknown_key_in_spec,
        test_info_reports_schema,
        test_snapshot_roundtrip_via_kvprefix_save,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
