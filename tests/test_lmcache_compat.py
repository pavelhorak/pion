#!/usr/bin/env python3
"""A6 LMCache Wire Compatibility Test — verifies Pion handles the exact RESP2 protocol
that LMCache's C++ RedisConnector sends.

LMCache uses 4 commands: GET, SET, EXISTS, DEL
- Keys: SHA256-style format "model@world_size@worker_id@chunk_hash@dtype"
- Values: Raw binary blobs (KV cache tensors, typically 1-4MB per chunk)
- Protocol: RESP2 (arrays of bulk strings)

Tests:
1. AUTH (optional, LMCache sends when configured)
2. SET large binary blob (~2MB, simulating KV cache tensor)
3. GET returns exact binary blob
4. EXISTS returns 1 for stored key
5. EXISTS returns 0 for missing key
6. DEL removes key, GET returns nil
7. Multiple SET/GET roundtrips (8 workers simulation)
8. Very large blob (8MB, 70B model chunk)

Prerequisites:
    ./pion-server -w 1

Usage:
    python3 tests/test_lmcache_compat.py
    python3 tests/test_lmcache_compat.py --port 1974
"""

import argparse
import hashlib
import os
import socket
import struct
import sys
import time


# ─── Raw RESP2 protocol (matching LMCache's C++ connector) ────────────────────

def resp_bulk_string(data: bytes) -> bytes:
    """Encode as RESP bulk string: $<len>\r\n<data>\r\n"""
    return f"${len(data)}\r\n".encode() + data + b"\r\n"


def resp_array(*args: bytes) -> bytes:
    """Encode as RESP array of bulk strings."""
    header = f"*{len(args)}\r\n".encode()
    body = b"".join(resp_bulk_string(a) for a in args)
    return header + body


def recv_line(s: socket.socket) -> bytes:
    """Read until \r\n."""
    buf = b""
    while not buf.endswith(b"\r\n"):
        ch = s.recv(1)
        if not ch:
            raise ConnectionError("socket closed")
        buf += ch
    return buf[:-2]


def recv_resp(s: socket.socket):
    """Parse one RESP response."""
    line = recv_line(s)
    prefix = chr(line[0])
    data = line[1:]
    if prefix == "+":
        return ("ok", data.decode())
    elif prefix == "-":
        return ("err", data.decode())
    elif prefix == ":":
        return ("int", int(data))
    elif prefix == "$":
        length = int(data)
        if length == -1:
            return ("nil", None)
        payload = b""
        while len(payload) < length + 2:
            payload += s.recv(length + 2 - len(payload))
        return ("bulk", payload[:length])
    elif prefix == "*":
        count = int(data)
        if count == -1:
            return ("nil", None)
        return ("array", [recv_resp(s) for _ in range(count)])
    return ("unknown", data)


def lmcache_key(model: str = "meta-llama/Llama-3.1-8B-Instruct",
                world_size: int = 1, worker_id: int = 0,
                chunk_hash: str = None, dtype: str = "bfloat16") -> str:
    """Generate an LMCache-style key."""
    if chunk_hash is None:
        chunk_hash = hashlib.sha256(os.urandom(32)).hexdigest()
    return f"vllm@{model}@{world_size}@{worker_id}@{chunk_hash}@{dtype}"


def make_kv_blob(size: int) -> bytes:
    """Generate a deterministic binary blob simulating KV cache tensor data."""
    # Use a repeating pattern that's verifiable
    pattern = struct.pack("<" + "f" * 256, *[float(i % 256) for i in range(256)])
    repeats = size // len(pattern) + 1
    return (pattern * repeats)[:size]


def connect(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(30)
    s.connect((host, port))
    return s


def main():
    parser = argparse.ArgumentParser(description="A6 LMCache Wire Compatibility Test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1974)
    parser.add_argument("--large", action="store_true", help="Include 8MB blob test")
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

    print(f"\n=== A6 LMCache Wire Compatibility Test (port {args.port}) ===\n")

    s = connect(args.host, args.port)

    # ── Test 1: AUTH ──
    print("[1] AUTH — LMCache optional authentication")
    s.sendall(resp_array(b"AUTH", b"password123"))
    r = recv_resp(s)
    check("AUTH returns +OK", r[0] == "ok", f"got: {r}")

    # LMCache sends AUTH <username> <password> when a username is configured.
    # Redis accepts that form only for the `default` user (or an ACL user);
    # expecting +OK for an arbitrary name ("user") was not Redis behaviour.
    s.sendall(resp_array(b"AUTH", b"default", b"password123"))
    r = recv_resp(s)
    check("AUTH default <password> returns +OK", r[0] == "ok", f"got: {r}")

    # ── Test 2: SET large binary blob ──
    print("[2] SET — store 2MB KV cache tensor blob")
    key1 = lmcache_key(chunk_hash="a" * 64)
    blob1 = make_kv_blob(2 * 1024 * 1024)  # 2MB
    s.sendall(resp_array(b"SET", key1.encode(), blob1))
    r = recv_resp(s)
    check("SET 2MB blob returns +OK", r[0] == "ok", f"got: {r}")

    # ── Test 3: GET returns exact blob ──
    print("[3] GET — retrieve exact binary blob")
    s.sendall(resp_array(b"GET", key1.encode()))
    r = recv_resp(s)
    check("GET returns bulk response", r[0] == "bulk", f"got type: {r[0]}")
    if r[0] == "bulk":
        check("GET blob size matches", len(r[1]) == len(blob1),
              f"got {len(r[1])} bytes, expected {len(blob1)}")
        check("GET blob content matches", r[1] == blob1,
              f"first 16 bytes differ" if r[1][:16] != blob1[:16] else "mismatch elsewhere")

    # ── Test 4: EXISTS returns 1 ──
    print("[4] EXISTS — key exists")
    s.sendall(resp_array(b"EXISTS", key1.encode()))
    r = recv_resp(s)
    check("EXISTS returns :1", r == ("int", 1), f"got: {r}")

    # ── Test 5: EXISTS returns 0 for missing key ──
    print("[5] EXISTS — key does not exist")
    missing_key = lmcache_key(chunk_hash="f" * 64)
    s.sendall(resp_array(b"EXISTS", missing_key.encode()))
    r = recv_resp(s)
    check("EXISTS returns :0", r == ("int", 0), f"got: {r}")

    # ── Test 6: DEL removes key ──
    print("[6] DEL — delete key, then GET returns nil")
    s.sendall(resp_array(b"DEL", key1.encode()))
    r = recv_resp(s)
    check("DEL returns :1", r == ("int", 1), f"got: {r}")

    s.sendall(resp_array(b"GET", key1.encode()))
    r = recv_resp(s)
    check("GET after DEL returns nil", r[0] == "nil" or (r[0] == "bulk" and r[1] is None),
          f"got: {r}")

    # ── Test 7: Multiple SET/GET roundtrips (8-worker simulation) ──
    print("[7] Multi-worker simulation — 8 keys SET then GET")
    keys = []
    blobs = []
    for i in range(8):
        k = lmcache_key(worker_id=i, chunk_hash=hashlib.sha256(f"chunk{i}".encode()).hexdigest())
        b = make_kv_blob(512 * 1024)  # 512KB each
        keys.append(k)
        blobs.append(b)
        s.sendall(resp_array(b"SET", k.encode(), b))
        r = recv_resp(s)
        if r[0] != "ok":
            check(f"SET worker {i}", False, f"got: {r}")

    all_match = True
    for i in range(8):
        s.sendall(resp_array(b"GET", keys[i].encode()))
        r = recv_resp(s)
        if r[0] != "bulk" or r[1] != blobs[i]:
            all_match = False
            break
    check("All 8 worker blobs match on GET", all_match)

    # ── Test 8: Large blob (8MB, 70B model) ──
    if args.large:
        print("[8] SET/GET — 8MB blob (70B model simulation)")
        key8m = lmcache_key(model="meta-llama/Llama-3-70B-Instruct", chunk_hash="b" * 64)
        blob8m = make_kv_blob(8 * 1024 * 1024)  # 8MB
        s.sendall(resp_array(b"SET", key8m.encode(), blob8m))
        r = recv_resp(s)
        check("SET 8MB blob returns +OK", r[0] == "ok", f"got: {r}")

        s.sendall(resp_array(b"GET", key8m.encode()))
        r = recv_resp(s)
        check("GET 8MB blob size matches", r[0] == "bulk" and len(r[1]) == len(blob8m),
              f"got {len(r[1]) if r[0] == 'bulk' else 'non-bulk'}")
        if r[0] == "bulk":
            check("GET 8MB blob content matches", r[1] == blob8m)
    else:
        print("[8] SKIP — 8MB blob test (use --large to enable)")

    s.close()

    # ── Summary ──
    print(f"\n{'='*50}")
    total = passed + failed
    if failed == 0:
        print(f"A6 LMCache Compat: ALL {total} TESTS PASSED")
    else:
        print(f"A6 LMCache Compat: {passed}/{total} passed, {failed} FAILED")
    print(f"{'='*50}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
