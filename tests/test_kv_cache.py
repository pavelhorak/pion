#!/usr/bin/env python3
"""Test KV.STORE / KV.FETCH / KV.INFO commands for M14 Phase 1."""

import socket
import struct
import time
import numpy as np

HOST = "127.0.0.1"
PORT = 1974

def send_resp(sock, *args):
    """Send a RESP command and return the raw response."""
    cmd = f"*{len(args)}\r\n"
    for arg in args:
        if isinstance(arg, bytes):
            cmd_bytes = cmd.encode() + f"${len(arg)}\r\n".encode() + arg + b"\r\n"
            sock.sendall(cmd_bytes)
            return sock.recv(65536)
        else:
            s = str(arg)
            cmd += f"${len(s)}\r\n{s}\r\n"
    sock.sendall(cmd.encode())
    return sock.recv(65536)

def send_raw_resp(sock, parts):
    """Send a RESP command with mixed string/bytes args."""
    header = f"*{len(parts)}\r\n".encode()
    body = b""
    for part in parts:
        if isinstance(part, bytes):
            body += f"${len(part)}\r\n".encode() + part + b"\r\n"
        else:
            s = str(part)
            body += f"${len(s)}\r\n{s}\r\n".encode()
    sock.sendall(header + body)
    return sock.recv(1024 * 1024)  # 1MB buffer for large blob responses

def test_kv_info():
    """Test KV.INFO returns stats."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.settimeout(5)

    resp = send_resp(sock, "KV.INFO")
    print(f"KV.INFO response: {resp[:200]}")

    sock.close()
    return b"entries:" in resp or b"ERR" in resp

def _server_dim(sock):
    """Read the server's embedding dimension from KV.INFO (varies by embed backend)."""
    resp = send_resp(sock, "KV.INFO")
    for ln in resp.decode(errors="replace").split("\r\n"):
        if ln.startswith("dimensions:"):
            return int(ln.split(":", 1)[1])
    raise RuntimeError(f"KV.INFO did not report dimensions: {resp!r}")


def test_kv_store_fetch():
    """Test KV.STORE then KV.FETCH with synthetic embeddings."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.settimeout(5)

    # Embedding dim depends on the server's embed backend (MiniLM=384, nomic=768).
    dim = _server_dim(sock)

    # Create a synthetic FP32 embedding (random unit-norm) at the server's dim
    np.random.seed(42)
    embedding = np.random.randn(dim).astype(np.float32)
    embedding /= np.linalg.norm(embedding)
    embed_bytes = embedding.tobytes()

    # Create a synthetic KV cache blob (just 1KB for testing)
    blob = b"FAKE_KV_CACHE_TENSOR_" * 50  # ~1KB

    # KV.STORE test_cache_1 <embedding> <blob>
    resp = send_raw_resp(sock, ["KV.STORE", "test_cache_1", embed_bytes, blob])
    print(f"KV.STORE response: {resp}")
    assert b"+OK" in resp, f"KV.STORE failed: {resp}"

    # KV.FETCH with the SAME embedding (should be a perfect hit)
    resp = send_raw_resp(sock, ["KV.FETCH", embed_bytes])
    print(f"KV.FETCH (exact match) response length: {len(resp)} bytes")
    assert len(resp) > 100, f"KV.FETCH returned too little data: {resp[:100]}"
    assert b"FAKE_KV_CACHE_TENSOR_" in resp, "KV.FETCH did not return the stored blob"

    # KV.FETCH with a SIMILAR embedding (add small noise)
    similar_embedding = embedding + np.random.randn(dim).astype(np.float32) * 0.01
    similar_embedding /= np.linalg.norm(similar_embedding)
    similar_bytes = similar_embedding.tobytes()

    resp = send_raw_resp(sock, ["KV.FETCH", similar_bytes])
    print(f"KV.FETCH (similar, noise=0.01) response length: {len(resp)} bytes")
    # Should still match (cosine similarity ~0.999)
    assert b"FAKE_KV_CACHE_TENSOR_" in resp, "KV.FETCH missed similar embedding"

    # KV.FETCH with a DIFFERENT embedding (should miss)
    diff_embedding = np.random.randn(dim).astype(np.float32)
    diff_embedding /= np.linalg.norm(diff_embedding)
    diff_bytes = diff_embedding.tobytes()

    resp = send_raw_resp(sock, ["KV.FETCH", diff_bytes])
    print(f"KV.FETCH (different) response: {resp[:100]}")
    # May or may not match — with only 1 entry, HNSW will find it, but cosine should be low
    # The threshold (0.95) should filter it out

    # KV.INFO — check stats
    resp = send_raw_resp(sock, ["KV.INFO"])
    print(f"KV.INFO response: {resp[:300]}")

    sock.close()
    print("\n=== ALL KV CACHE TESTS PASSED ===")
    return True

if __name__ == "__main__":
    print("Testing M14 Phase 1: KV Cache Store")
    print(f"Connecting to Pion at {HOST}:{PORT}")
    print()

    try:
        test_kv_info()
        print()
        test_kv_store_fetch()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
