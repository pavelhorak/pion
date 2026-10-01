#!/usr/bin/env python3
"""Test V.CREATE / V.STOREBATCH / V.FETCH / V.INFO commands.

Requires: ./pion-server --kvcache -w 1
"""

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
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4*1024*1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024*1024)
        self.sock.settimeout(10)
        self.sock.connect((host, port))

    def _encode(self, parts):
        header = f"*{len(parts)}\r\n".encode()
        body = b""
        for p in parts:
            if isinstance(p, bytes):
                body += f"${len(p)}\r\n".encode() + p + b"\r\n"
            else:
                s = str(p)
                body += f"${len(s)}\r\n{s}\r\n".encode()
        return header + body

    def _send(self, parts):
        self.sock.sendall(self._encode(parts))
        return self._recv()

    def _recv(self):
        buf = b""
        while True:
            chunk = self.sock.recv(4*1024*1024)
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
            if self._complete(buf):
                break
        # drain
        self.sock.settimeout(0.002)
        try:
            while True:
                extra = self.sock.recv(1024*1024)
                if not extra:
                    break
        except (socket.timeout, BlockingIOError):
            pass
        self.sock.settimeout(10)
        return buf

    def _complete(self, d):
        if len(d) < 3:
            return False
        p = d[0:1]
        if p in (b"+", b"-", b":"):
            return b"\r\n" in d
        if p == b"$":
            nl = d.find(b"\r\n")
            if nl < 0:
                return False
            ls = d[1:nl].decode()
            if ls == "-1":
                return True
            return len(d) >= nl + 2 + int(ls) + 2
        return b"\r\n" in d

    def create(self, sid, dim, vquant="int8"):
        parts = ["V.CREATE", sid, str(dim)]
        if vquant != "int8":
            parts += ["VQUANT", vquant]
        resp = self._send(parts)
        return b":" in resp and not resp.startswith(b"-")

    def storebatch(self, sid, layer, start_id, values_fp32):
        n = values_fp32.shape[0]
        resp = self._send([
            "V.STOREBATCH", sid, str(layer), str(start_id), str(n),
            values_fp32.astype(np.float32).tobytes(),
        ])
        return b"+OK" in resp

    def fetch(self, sid, layer, token_ids):
        parts = ["V.FETCH", sid, str(layer)] + [str(t) for t in token_ids]
        resp = self._send(parts)
        if resp.startswith(b"$") and not resp.startswith(b"$-1"):
            nl = resp.find(b"\r\n")
            blen = int(resp[1:nl])
            blob = resp[nl+2:nl+2+blen]
            return np.frombuffer(blob, dtype=np.float32)
        return None

    def info(self, sid=None):
        parts = ["V.INFO"]
        if sid:
            parts.append(sid)
        resp = self._send(parts)
        return resp.decode("utf-8", errors="replace")


def test_basic_roundtrip():
    """Store and fetch INT8 values, verify cosine similarity."""
    c = VStoreClient()
    dim = 768
    n = 64

    assert c.create("test_rt", dim), "V.CREATE failed"

    np.random.seed(42)
    values = np.random.randn(n, dim).astype(np.float32) * 0.5

    assert c.storebatch("test_rt", 0, 0, values), "V.STOREBATCH failed"

    # Fetch individual tokens
    for tid in [0, 10, 32, 63]:
        ret = c.fetch("test_rt", 0, [tid])
        assert ret is not None, f"V.FETCH returned None for token {tid}"
        assert len(ret) >= dim, f"V.FETCH returned {len(ret)} floats, expected {dim}"
        v_orig = values[tid]
        v_ret = ret[:dim]
        cos = np.dot(v_orig, v_ret) / (np.linalg.norm(v_orig) * np.linalg.norm(v_ret))
        assert cos > 0.99, f"Token {tid}: cosine {cos:.4f} < 0.99"
        print(f"  Token {tid}: cosine={cos:.6f} OK")

    # Batch fetch
    ids = [0, 5, 10, 15, 20]
    ret = c.fetch("test_rt", 0, ids)
    assert ret is not None, "Batch V.FETCH returned None"
    assert len(ret) >= len(ids) * dim, f"Batch returned {len(ret)}, expected {len(ids)*dim}"
    for i, tid in enumerate(ids):
        v_ret = ret[i*dim:(i+1)*dim]
        cos = np.dot(values[tid], v_ret) / (np.linalg.norm(values[tid]) * np.linalg.norm(v_ret))
        assert cos > 0.99, f"Batch token {tid}: cosine {cos:.4f}"
    print(f"  Batch fetch ({len(ids)} tokens): OK")

    print("PASS: test_basic_roundtrip")


def test_turbo4_roundtrip():
    """Store turbo4 values, verify cosine within expected range."""
    c = VStoreClient()
    dim = 1024  # must be divisible by 32
    n = 32

    assert c.create("test_t4", dim, vquant="turbo4"), "V.CREATE turbo4 failed"

    np.random.seed(123)
    values = np.random.randn(n, dim).astype(np.float32) * 0.5

    assert c.storebatch("test_t4", 0, 0, values), "V.STOREBATCH failed"

    for tid in [0, 15, 31]:
        ret = c.fetch("test_t4", 0, [tid])
        assert ret is not None, f"V.FETCH returned None for token {tid}"
        v_orig = values[tid]
        v_ret = ret[:dim]
        cos = np.dot(v_orig, v_ret) / (np.linalg.norm(v_orig) * np.linalg.norm(v_ret))
        assert cos > 0.95, f"turbo4 token {tid}: cosine {cos:.4f} < 0.95"
        print(f"  turbo4 token {tid}: cosine={cos:.6f}")

    print("PASS: test_turbo4_roundtrip")


def test_multi_layer():
    """Store across multiple layers, fetch from each."""
    c = VStoreClient()
    dim = 512
    n = 16

    assert c.create("test_ml", dim), "V.CREATE failed"

    np.random.seed(77)
    for layer in range(4):
        values = np.random.randn(n, dim).astype(np.float32) * 0.5
        assert c.storebatch("test_ml", layer, 0, values), f"V.STOREBATCH layer {layer} failed"

    # Fetch from each layer
    for layer in range(4):
        ret = c.fetch("test_ml", layer, [0])
        assert ret is not None, f"V.FETCH layer {layer} returned None"
        assert len(ret) >= dim, f"V.FETCH layer {layer} wrong size"

    print("PASS: test_multi_layer")


def test_info():
    """V.INFO returns session stats."""
    c = VStoreClient()
    info = c.info()
    assert "enabled:1" in info, f"V.INFO missing enabled: {info}"
    print(f"  Global info: {info.strip()[:80]}...")
    print("PASS: test_info")


def test_large_batch():
    """Store 1K tokens, fetch random subset."""
    c = VStoreClient()
    dim = 1024
    n = 1024

    assert c.create("test_1k", dim), "V.CREATE failed"

    np.random.seed(999)
    values = np.random.randn(n, dim).astype(np.float32) * 0.3

    t0 = time.perf_counter()
    assert c.storebatch("test_1k", 0, 0, values), "V.STOREBATCH 1K failed"
    store_ms = (time.perf_counter() - t0) * 1000

    # Fetch 32 random tokens
    ids = np.random.choice(n, 32, replace=False).tolist()
    t0 = time.perf_counter()
    ret = c.fetch("test_1k", 0, ids)
    fetch_ms = (time.perf_counter() - t0) * 1000

    assert ret is not None, "V.FETCH returned None"
    cosines = []
    for i, tid in enumerate(ids):
        v_ret = ret[i*dim:(i+1)*dim]
        cos = np.dot(values[tid], v_ret) / (np.linalg.norm(values[tid]) * np.linalg.norm(v_ret))
        cosines.append(cos)

    mean_cos = np.mean(cosines)
    print(f"  1K store: {store_ms:.0f}ms, 32-token fetch: {fetch_ms:.1f}ms, "
          f"mean cosine: {mean_cos:.4f}")
    assert mean_cos > 0.99, f"Mean cosine {mean_cos:.4f} < 0.99"

    print("PASS: test_large_batch")


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
    print("V-Store Tests")
    print("=" * 60)

    tests = [test_basic_roundtrip, test_turbo4_roundtrip, test_multi_layer, test_info, test_large_batch]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
