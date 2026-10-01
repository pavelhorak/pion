#!/usr/bin/env python3
"""A3 Auto-Embedding Integration Test — verifies semantic cache works with the inference sidecar.

Tests:
1. AI.SEMANTIC_CACHE SET stores a query-response pair via auto-embedding
2. AI.SEMANTIC_CACHE GET retrieves it for semantically similar queries
3. AI.SEMANTIC_CACHE GET returns nil for unrelated queries
4. AI.EMBED returns a 384-dim vector (MiniLM-L6-v2)

Prerequisites:
    - Pion server running with auto-embed (default: no --no-auto-embed, no --profile kv)
    - OR: Pion server running with --inference flag
    - The inference sidecar must be ready (loads MiniLM-L6-v2 on startup)

Usage:
    # Start server (auto-embed is the default):
    ./pion-server -w 1

    # Wait for sidecar to load model (~5-15s), then:
    python3 tests/test_auto_embed.py
    python3 tests/test_auto_embed.py --port 1974
"""

import argparse
import socket
import struct
import sys
import time


def encode_cmd(args):
    """Encode a list of strings/bytes as a RESP array command."""
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def recv_line(s):
    """Read a single RESP line (up to \\r\\n)."""
    buf = b""
    while not buf.endswith(b"\r\n"):
        ch = s.recv(1)
        if not ch:
            raise ConnectionError("socket closed")
        buf += ch
    return buf[:-2]  # strip \r\n


def recv_resp(s):
    """Parse one RESP value from the socket."""
    line = recv_line(s)
    prefix = chr(line[0])
    data = line[1:]
    if prefix == "+":
        return data.decode()
    elif prefix == "-":
        return Exception(data.decode())
    elif prefix == ":":
        return int(data)
    elif prefix == "$":
        length = int(data)
        if length == -1:
            return None
        payload = b""
        while len(payload) < length + 2:
            payload += s.recv(length + 2 - len(payload))
        return payload[:length]  # strip trailing \r\n
    elif prefix == "*":
        count = int(data)
        if count == -1:
            return None
        return [recv_resp(s) for _ in range(count)]
    else:
        return data.decode()


def send_cmd(s, *args):
    """Send a RESP command and return the parsed response."""
    s.sendall(encode_cmd(args))
    return recv_resp(s)


def connect(host, port):
    """Connect to Pion with TCP_NODELAY."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(30)  # embedding can be slow first time
    s.connect((host, port))
    return s


def main():
    parser = argparse.ArgumentParser(description="A3 Auto-Embedding Integration Test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1974)
    args = parser.parse_args()

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}: {detail}")
            failed += 1

    print(f"\n=== A3 Auto-Embedding Test (port {args.port}) ===\n")

    s = connect(args.host, args.port)

    # The sidecar loads MiniLM AFTER the server answers; the test used to race
    # it and report the load as three failures. Wait for AI.EMBED to answer
    # (a real readiness signal), and fail loudly if it never does.
    import time
    deadline, r = time.monotonic() + 90, None
    while time.monotonic() < deadline:
        r = send_cmd(s, "AI.EMBED", "readiness probe")
        if not (isinstance(r, str) and r.startswith("ERR")):
            break
        time.sleep(1)
    else:
        print(f"  FAIL  embedding backend never became ready in 90 s: {r}")
        sys.exit(1)

    # Test 1: AI.SEMANTIC_CACHE SET
    print("[1] AI.SEMANTIC_CACHE SET — store query-response pair")
    r = send_cmd(s, "AI.SEMANTIC_CACHE", "SET", "What is Pion?", "Pion is a high-performance vector database written in Mojo")
    check("SET returns OK", r == "OK", f"got: {r}")

    # Store a few more pairs for richer testing
    send_cmd(s, "AI.SEMANTIC_CACHE", "SET", "How fast is Pion?", "Pion achieves 14M ops/sec on Linux with io_uring")
    send_cmd(s, "AI.SEMANTIC_CACHE", "SET", "What language is Pion written in?", "Pion is written in Mojo, a systems programming language")

    # Test 2: AI.SEMANTIC_CACHE GET — exact match
    print("[2] AI.SEMANTIC_CACHE GET — exact query retrieval")
    r = send_cmd(s, "AI.SEMANTIC_CACHE", "GET", "What is Pion?")
    check("GET exact match returns response", r is not None and isinstance(r, bytes), f"got: {r}")
    if r is not None and isinstance(r, bytes):
        check("GET response content matches", b"Pion" in r, f"got: {r[:80]}")

    # Test 3: AI.SEMANTIC_CACHE GET — similar query
    print("[3] AI.SEMANTIC_CACHE GET — similar query retrieval")
    r = send_cmd(s, "AI.SEMANTIC_CACHE", "GET", "Tell me about Pion", "THRESHOLD", "0.80")
    check("GET similar query returns response", r is not None and isinstance(r, bytes), f"got: {type(r).__name__}: {r}")

    # Test 4: AI.SEMANTIC_CACHE GET — unrelated query
    print("[4] AI.SEMANTIC_CACHE GET — unrelated query returns nil")
    r = send_cmd(s, "AI.SEMANTIC_CACHE", "GET", "recipe for chocolate cake", "THRESHOLD", "0.95")
    check("GET unrelated returns nil", r is None, f"got: {r}")

    # Test 5: AI.EMBED — get raw embedding vector
    print("[5] AI.EMBED — returns 384-dim FP32 vector")
    r = send_cmd(s, "AI.EMBED", "test embedding query")
    if isinstance(r, Exception):
        check("AI.EMBED succeeds", False, str(r))
    elif r is not None and isinstance(r, bytes):
        # Expected: 384 floats × 4 bytes = 1536 bytes
        expected_bytes = 384 * 4
        check("AI.EMBED returns correct byte length", len(r) == expected_bytes,
              f"got {len(r)} bytes, expected {expected_bytes}")
        if len(r) == expected_bytes:
            # Verify it's valid FP32 data (not all zeros)
            floats = struct.unpack(f"<{384}f", r)
            non_zero = sum(1 for f in floats if abs(f) > 1e-6)
            check("AI.EMBED vector is non-trivial", non_zero > 100,
                  f"only {non_zero}/384 non-zero values")
    else:
        check("AI.EMBED returns bytes", False, f"got: {type(r).__name__}")

    s.close()

    # Summary
    print(f"\n{'='*50}")
    total = passed + failed
    if failed == 0:
        print(f"A3 Auto-Embedding: ALL {total} TESTS PASSED")
    else:
        print(f"A3 Auto-Embedding: {passed}/{total} passed, {failed} FAILED")
    print(f"{'='*50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
