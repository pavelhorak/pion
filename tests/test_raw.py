#!/usr/bin/env python3
"""Gate 1: Correctness invariants — read-after-write, type safety, SSO boundary, pipeline stress.

This is the highest-value, lowest-cost test in the Pion test suite. It catches the exact
class of bugs that have shipped multiple times (from_ptr_unsafe key mismatches, type
confusion, SSO boundary errors).

Usage:
    python3 tests/test_raw.py                  # default port 1974
    python3 tests/test_raw.py --port 1975      # custom port

Exit code 0 = all passed, 1 = failures detected.
"""

import argparse
import os
import socket
import sys
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader  # noqa: E402

# ─── RESP helpers ──────────────────────────────────────────────────────────────

def encode_cmd(args):
    """Encode a list of strings/bytes as a RESP array command."""
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def recv_resp(sock, timeout=10.0):
    """Receive exactly ONE complete RESP reply (raw bytes).

    This used to be a single recv() with a swallowed 2 s timeout: a slow reply
    returned b"" and was then read as the answer to the NEXT command, and a
    reply split across TCP segments was truncated. The strict reader parses
    the frame and raises instead of guessing."""
    return reader(sock, timeout).read_raw()


def send_recv(sock, *args):
    """Send a single command and receive the response."""
    sock.sendall(encode_cmd(args))
    return recv_resp(sock, timeout=2.0).decode(errors="replace")


def send_recv_multi(sock, commands):
    """Send commands pipelined; return exactly one reply per command,
    concatenated. Framing is parsed, so a truncated batch raises and a
    command that answers twice shows up as a surplus reply in the next read."""
    payload = b"".join(encode_cmd(cmd) for cmd in commands)
    sock.sendall(payload)
    r = reader(sock)
    return b"".join(r.read_raw() for _ in commands).decode(errors="replace")


def flushall(sock):
    """Flush the keyspace."""
    send_recv(sock, "FLUSHALL")


# ─── Test infrastructure ──────────────────────────────────────────────────────

passed = 0
failed = 0
errors = []


def check(condition, name, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        errors.append(msg)
        print(f"  ✗ {name}: {detail}")


# ─── Section 1: Read-After-Write Consistency ──────────────────────────────────

def test_read_after_write(sock):
    """Every write followed by its corresponding read must return the written value."""
    print("\n=== Section 1: Read-After-Write Consistency ===")
    flushall(sock)

    # SET / GET
    send_recv(sock, "SET", "raw:k1", "hello")
    r = send_recv(sock, "GET", "raw:k1")
    check("hello" in r, "SET→GET", f"got {r!r}")

    # SET overwrite
    send_recv(sock, "SET", "raw:k1", "world")
    r = send_recv(sock, "GET", "raw:k1")
    check("world" in r, "SET overwrite→GET", f"got {r!r}")

    # MSET / MGET
    send_recv(sock, "MSET", "raw:m1", "a", "raw:m2", "b", "raw:m3", "c")
    r = send_recv(sock, "MGET", "raw:m1", "raw:m2", "raw:m3")
    check("a" in r and "b" in r and "c" in r, "MSET→MGET", f"got {r!r}")

    # INCR / GET
    send_recv(sock, "SET", "raw:counter", "10")
    send_recv(sock, "INCR", "raw:counter")
    r = send_recv(sock, "GET", "raw:counter")
    check("11" in r, "INCR→GET", f"got {r!r}")

    # DECR
    send_recv(sock, "DECR", "raw:counter")
    r = send_recv(sock, "GET", "raw:counter")
    check("10" in r, "DECR→GET", f"got {r!r}")

    # HSET / HGET
    send_recv(sock, "HSET", "raw:h1", "field1", "val1")
    r = send_recv(sock, "HGET", "raw:h1", "field1")
    check("val1" in r, "HSET→HGET", f"got {r!r}")

    # HSET multi-field / HGETALL
    send_recv(sock, "HSET", "raw:h2", "f1", "v1", "f2", "v2")
    r = send_recv(sock, "HGETALL", "raw:h2")
    check("f1" in r and "v1" in r and "f2" in r and "v2" in r, "HSET multi→HGETALL", f"got {r!r}")

    # LPUSH / LRANGE
    send_recv(sock, "LPUSH", "raw:l1", "a")
    send_recv(sock, "LPUSH", "raw:l1", "b")
    r = send_recv(sock, "LRANGE", "raw:l1", "0", "-1")
    check("a" in r and "b" in r, "LPUSH→LRANGE", f"got {r!r}")

    # RPUSH / LRANGE
    send_recv(sock, "RPUSH", "raw:l2", "x")
    send_recv(sock, "RPUSH", "raw:l2", "y")
    r = send_recv(sock, "LRANGE", "raw:l2", "0", "-1")
    check("x" in r and "y" in r, "RPUSH→LRANGE", f"got {r!r}")

    # LPOP / RPOP
    send_recv(sock, "RPUSH", "raw:l3", "first")
    send_recv(sock, "RPUSH", "raw:l3", "second")
    send_recv(sock, "RPUSH", "raw:l3", "third")
    r = send_recv(sock, "LPOP", "raw:l3")
    check("first" in r, "LPOP returns head", f"got {r!r}")
    r = send_recv(sock, "RPOP", "raw:l3")
    check("third" in r, "RPOP returns tail", f"got {r!r}")

    # LLEN
    r = send_recv(sock, "LLEN", "raw:l3")
    check(":1" in r, "LLEN after pops", f"got {r!r}")

    # SADD / SMEMBERS (one at a time — fast path handles single-arg SADD)
    send_recv(sock, "SADD", "raw:s1", "a")
    send_recv(sock, "SADD", "raw:s1", "b")
    send_recv(sock, "SADD", "raw:s1", "c")
    r = send_recv(sock, "SMEMBERS", "raw:s1")
    check("a" in r and "b" in r and "c" in r, "SADD→SMEMBERS", f"got {r!r}")

    # SCARD
    r = send_recv(sock, "SCARD", "raw:s1")
    check(":3" in r, "SCARD", f"got {r!r}")

    # ZADD / ZPOPMIN (ZSCORE/ZRANGE not in fast path; use ZPOPMIN to verify data)
    send_recv(sock, "ZADD", "raw:z1", "1.5", "alice")
    send_recv(sock, "ZADD", "raw:z1", "2.5", "bob")
    r = send_recv(sock, "ZPOPMIN", "raw:z1")
    check("alice" in r, "ZADD→ZPOPMIN returns lowest", f"got {r!r}")

    # DEL / EXISTS
    send_recv(sock, "SET", "raw:del1", "val")
    r = send_recv(sock, "EXISTS", "raw:del1")
    check(":1" in r, "EXISTS before DEL", f"got {r!r}")
    send_recv(sock, "DEL", "raw:del1")
    r = send_recv(sock, "EXISTS", "raw:del1")
    check(":0" in r, "EXISTS after DEL", f"got {r!r}")
    r = send_recv(sock, "GET", "raw:del1")
    check("$-1" in r, "GET after DEL returns nil", f"got {r!r}")

    # SETBIT / GETBIT / BITCOUNT
    send_recv(sock, "SETBIT", "raw:bm1", "7", "1")
    r = send_recv(sock, "GETBIT", "raw:bm1", "7")
    check(":1" in r, "SETBIT→GETBIT", f"got {r!r}")
    r = send_recv(sock, "GETBIT", "raw:bm1", "0")
    check(":0" in r, "GETBIT unset bit", f"got {r!r}")
    r = send_recv(sock, "BITCOUNT", "raw:bm1")
    check(":1" in r, "BITCOUNT", f"got {r!r}")

    # PFADD / PFCOUNT
    send_recv(sock, "PFADD", "raw:hll1", "a", "b", "c")
    r = send_recv(sock, "PFCOUNT", "raw:hll1")
    # HLL is probabilistic but for 3 items it should be exactly 3
    check(":3" in r or ":2" in r or ":4" in r, "PFADD→PFCOUNT", f"got {r!r}")


# ─── Section 2: SSO Boundary Tests ───────────────────────────────────────────

def test_sso_boundary(sock):
    """Test keys and values at exactly the SSO boundary (23 bytes).

    GenericValue uses STRING_SSO for <=23 bytes and heap STRING for >23 bytes.
    The from_ptr_unsafe bug caused type mismatches at this boundary.
    """
    print("\n=== Section 2: SSO Boundary (23 bytes) ===")
    flushall(sock)

    # Keys at boundary
    key_22 = "k" * 22   # 22 bytes — SSO
    key_23 = "k" * 23   # 23 bytes — SSO (max)
    key_24 = "k" * 24   # 24 bytes — heap STRING

    for key, label in [(key_22, "22B SSO"), (key_23, "23B SSO max"), (key_24, "24B heap")]:
        send_recv(sock, "SET", key, "v")
        r = send_recv(sock, "GET", key)
        check("v" in r and "$-1" not in r, f"SET→GET key {label}", f"got {r!r}")

        send_recv(sock, "DEL", key)
        r = send_recv(sock, "GET", key)
        check("$-1" in r, f"DEL→GET key {label}", f"got {r!r}")

    # Values at boundary
    val_22 = "v" * 22
    val_23 = "v" * 23
    val_24 = "v" * 24

    for val, label in [(val_22, "22B val"), (val_23, "23B val"), (val_24, "24B val")]:
        send_recv(sock, "SET", "raw:sso_val", val)
        r = send_recv(sock, "GET", "raw:sso_val")
        check(val in r, f"SET→GET {label}", f"got {r!r}")

    # HSET keys at boundary
    for key, label in [(key_22, "22B"), (key_23, "23B"), (key_24, "24B")]:
        hkey = "raw:h_" + key[:8]
        send_recv(sock, "HSET", hkey, key, "hval")
        r = send_recv(sock, "HGET", hkey, key)
        check("hval" in r, f"HSET→HGET field {label}", f"got {r!r}")

    # Cross-boundary: write with 23B key, read with same key (exact match required)
    send_recv(sock, "SET", key_23, "boundary_value")
    r = send_recv(sock, "GET", key_23)
    check("boundary_value" in r, "23B key exact match", f"got {r!r}")

    # Verify 23B and 24B keys are distinct
    send_recv(sock, "SET", key_23, "twenty_three")
    send_recv(sock, "SET", key_24, "twenty_four")
    r23 = send_recv(sock, "GET", key_23)
    r24 = send_recv(sock, "GET", key_24)
    check("twenty_three" in r23, "23B key distinct from 24B", f"got {r23!r}")
    check("twenty_four" in r24, "24B key distinct from 23B", f"got {r24!r}")

    # Large values crossing writev threshold (512B)
    val_511 = "A" * 511
    val_512 = "B" * 512
    val_513 = "C" * 513
    for val, label in [(val_511, "511B"), (val_512, "512B writev threshold"), (val_513, "513B")]:
        send_recv(sock, "SET", "raw:big", val)
        r = send_recv(sock, "GET", "raw:big")
        check(val in r, f"SET→GET value {label}", f"response length {len(r)}")


# ─── Section 3: Type Safety ──────────────────────────────────────────────────

def test_type_safety(sock):
    """Operating on the wrong type must return an error, not nil or crash."""
    print("\n=== Section 3: Type Safety ===")
    flushall(sock)

    # Create keys of different types
    send_recv(sock, "SET", "raw:str", "hello")
    send_recv(sock, "LPUSH", "raw:list", "a")
    send_recv(sock, "SADD", "raw:set", "x")
    send_recv(sock, "ZADD", "raw:zset", "1", "m")
    send_recv(sock, "HSET", "raw:hash", "f", "v")

    # GET on non-string types — should return WRONGTYPE error
    r = send_recv(sock, "LPUSH", "raw:str", "oops")
    check("WRONGTYPE" in r or "ERR" in r, "LPUSH on string key → error", f"got {r!r}")

    r = send_recv(sock, "GET", "raw:list")
    check("WRONGTYPE" in r or "ERR" in r, "GET on list key → error", f"got {r!r}")

    r = send_recv(sock, "SADD", "raw:list", "oops")
    check("WRONGTYPE" in r or "ERR" in r, "SADD on list key → error", f"got {r!r}")

    # INCR on non-numeric string
    send_recv(sock, "SET", "raw:text", "hello")
    r = send_recv(sock, "INCR", "raw:text")
    check("ERR" in r, "INCR on non-numeric → error", f"got {r!r}")

    # Verify original keys are unchanged after wrong-type operations
    r = send_recv(sock, "GET", "raw:str")
    check("hello" in r, "string key unchanged after failed LPUSH", f"got {r!r}")
    r = send_recv(sock, "LRANGE", "raw:list", "0", "-1")
    check("a" in r, "list key unchanged after failed GET", f"got {r!r}")


# ─── Section 4: Edge Cases ───────────────────────────────────────────────────

def test_edge_cases(sock):
    """Empty values, missing keys, multi-key operations with gaps."""
    print("\n=== Section 4: Edge Cases ===")
    flushall(sock)

    # Empty string value
    send_recv(sock, "SET", "raw:empty", "")
    r = send_recv(sock, "GET", "raw:empty")
    check("$0\r\n\r\n" in r, "SET→GET empty string", f"got {r!r}")

    # GET missing key
    r = send_recv(sock, "GET", "raw:nonexistent")
    check("$-1" in r, "GET missing key → nil", f"got {r!r}")

    # MGET with mix of existing and missing keys
    send_recv(sock, "SET", "raw:e1", "yes")
    r = send_recv(sock, "MGET", "raw:e1", "raw:missing1", "raw:missing2")
    check("yes" in r and "$-1" in r, "MGET mixed existing/missing", f"got {r!r}")

    # DEL non-existent key
    r = send_recv(sock, "DEL", "raw:nonexistent")
    check(":0" in r, "DEL missing key → 0", f"got {r!r}")

    # EXISTS non-existent key
    r = send_recv(sock, "EXISTS", "raw:nonexistent")
    check(":0" in r, "EXISTS missing key → 0", f"got {r!r}")

    # LRANGE on empty/missing list
    r = send_recv(sock, "LRANGE", "raw:no_list", "0", "-1")
    check("*0" in r, "LRANGE missing key → empty array", f"got {r!r}")

    # LPOP / RPOP on empty/missing list
    r = send_recv(sock, "LPOP", "raw:no_list")
    check("$-1" in r, "LPOP missing key → nil", f"got {r!r}")

    # Numeric key names (common in benchmarks)
    send_recv(sock, "SET", "12345", "numeric_key")
    r = send_recv(sock, "GET", "12345")
    check("numeric_key" in r, "numeric key name", f"got {r!r}")

    # Single-char key
    send_recv(sock, "SET", "x", "single")
    r = send_recv(sock, "GET", "x")
    check("single" in r, "single-char key", f"got {r!r}")


# ─── Section 5: Pipeline Stress ──────────────────────────────────────────────

def test_pipeline_stress(sock):
    """Send commands in bulk pipelines and verify every response matches."""
    print("\n=== Section 5: Pipeline Stress ===")
    flushall(sock)

    N = 200  # number of keys

    # Pipeline N SETs
    cmds = [("SET", f"raw:pipe:{i}", f"val_{i}") for i in range(N)]
    resp = send_recv_multi(sock, cmds)
    ok_count = resp.count("+OK")
    check(ok_count == N, f"Pipeline {N} SETs", f"got {ok_count} OKs, expected {N}")

    # Pipeline N GETs and verify all values
    cmds = [("GET", f"raw:pipe:{i}") for i in range(N)]
    resp = send_recv_multi(sock, cmds)
    get_ok = sum(1 for i in range(N) if f"val_{i}" in resp)
    check(get_ok == N, f"Pipeline {N} GETs", f"got {get_ok}/{N} correct values")

    # Pipeline mixed commands
    cmds = []
    for i in range(50):
        cmds.append(("SET", f"raw:pmix:{i}", f"m{i}"))
        cmds.append(("GET", f"raw:pmix:{i}"))
    resp = send_recv_multi(sock, cmds)
    ok_count = resp.count("+OK")
    check(ok_count >= 50, "Pipeline mixed SET+GET (50 pairs)", f"got {ok_count} OKs")

    # Pipeline INCR
    send_recv(sock, "SET", "raw:pctr", "0")
    cmds = [("INCR", "raw:pctr") for _ in range(100)]
    send_recv_multi(sock, cmds)
    r = send_recv(sock, "GET", "raw:pctr")
    check("100" in r, "Pipeline 100 INCRs", f"got {r!r}")

    # Pipeline LPUSH then LLEN
    cmds = [("LPUSH", "raw:plist", f"item_{i}") for i in range(100)]
    send_recv_multi(sock, cmds)
    r = send_recv(sock, "LLEN", "raw:plist")
    check(":100" in r, "Pipeline 100 LPUSHs → LLEN", f"got {r!r}")

    # Pipeline HSET then HGETALL
    cmds = [("HSET", "raw:phash", f"f{i}", f"v{i}") for i in range(50)]
    send_recv_multi(sock, cmds)
    r = send_recv(sock, "HGETALL", "raw:phash")
    field_ok = sum(1 for i in range(50) if f"f{i}" in r and f"v{i}" in r)
    check(field_ok == 50, "Pipeline 50 HSETs → HGETALL", f"got {field_ok}/50 fields")


# ─── Section 6: Concurrent Connections ────────────────────────────────────────

def test_concurrent_connections(host, port):
    """Multiple connections writing/reading unique keys must not interfere."""
    print("\n=== Section 6: Concurrent Connections ===")

    NUM_CONNS = 10
    KEYS_PER_CONN = 20

    socks = []
    for i in range(NUM_CONNS):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        socks.append(s)

    # Each connection writes its own keys
    for i, s in enumerate(socks):
        cmds = [("SET", f"raw:conn{i}:k{j}", f"c{i}v{j}") for j in range(KEYS_PER_CONN)]
        payload = b"".join(encode_cmd(cmd) for cmd in cmds)
        s.sendall(payload)

    # Drain all SET responses — exactly one +OK per SET
    set_ok = True
    for s in socks:
        rd = reader(s)
        set_ok &= all(rd.read_raw() == b"+OK\r\n" for _ in range(KEYS_PER_CONN))
    check(set_ok, f"{NUM_CONNS}×{KEYS_PER_CONN} pipelined SETs each answered +OK", "a SET reply was not +OK")

    # Each connection reads its own keys and verifies each VALUE, in order.
    # (A substring test over the whole batch passed on wrong order and on
    # prefixes: "c0v1" is inside "c0v10".)
    all_ok, bad = True, ""
    for i, s in enumerate(socks):
        cmds = [("GET", f"raw:conn{i}:k{j}") for j in range(KEYS_PER_CONN)]
        payload = b"".join(encode_cmd(cmd) for cmd in cmds)
        s.sendall(payload)
        rd = reader(s)
        for j in range(KEYS_PER_CONN):
            got = rd.read()
            if got != f"c{i}v{j}".encode():
                all_ok, bad = False, f"conn {i} key {j}: {got!r}"
                break
        if not all_ok:
            break

    check(all_ok, f"{NUM_CONNS} connections × {KEYS_PER_CONN} keys isolated", bad)

    # Cross-read: connection 0 reads connection 1's keys
    cmds = [("GET", f"raw:conn1:k{j}") for j in range(KEYS_PER_CONN)]
    payload = b"".join(encode_cmd(cmd) for cmd in cmds)
    socks[0].sendall(payload)
    rd = reader(socks[0])
    got = [rd.read() for _ in range(KEYS_PER_CONN)]
    cross_ok = got == [f"c1v{j}".encode() for j in range(KEYS_PER_CONN)]
    check(cross_ok, "Cross-connection reads work", f"got {got[:3]!r}…")

    for s in socks:
        s.close()


# ─── Section 7: Binary Safety ────────────────────────────────────────────────

def test_binary_safety(sock):
    """Binary-safe keys and values: null bytes, embedded CRLF, 0xFF bytes."""
    print("\n=== Section 7: Binary Safety ===")
    flushall(sock)

    # Value with null bytes: \x00\x01\x02\xff (4 bytes)
    raw_cmd = b"*3\r\n$3\r\nSET\r\n$7\r\nraw:bin\r\n$4\r\n\x00\x01\x02\xff\r\n"
    sock.sendall(raw_cmd)
    r = recv_resp(sock)
    check(b"+OK" in r, "SET binary value with null bytes", f"got {r!r}")

    sock.sendall(encode_cmd(["GET", "raw:bin"]))
    r = recv_resp(sock)
    check(b"$4\r\n" in r, "GET binary value length is 4", f"got {r!r}")

    # Value with embedded CRLF: "hello\r\nworld" (12 bytes)
    crlf_val = b"hello\r\nworld"
    sock.sendall(encode_cmd(["SET", "raw:crlf", crlf_val]))
    r = recv_resp(sock)
    check(b"+OK" in r, "SET value with embedded CRLF", f"got {r!r}")

    sock.sendall(encode_cmd(["GET", "raw:crlf"]))
    r = recv_resp(sock)
    check(b"$12\r\n" in r, "GET CRLF value length is 12", f"got {r!r}")

    # Value with all-0xFF bytes (8 bytes)
    ff_val = b"\xff" * 8
    sock.sendall(encode_cmd(["SET", "raw:ff", ff_val]))
    r = recv_resp(sock)
    check(b"+OK" in r, "SET all-0xFF value", f"got {r!r}")

    sock.sendall(encode_cmd(["GET", "raw:ff"]))
    r = recv_resp(sock)
    check(b"$8\r\n" in r, "GET all-0xFF value length is 8", f"got {r!r}")
    check(b"\xff" * 8 in r, "GET all-0xFF value content", f"got {r!r}")


# ─── Section 8: TTL Basic ───────────────────────────────────────────────────

def test_ttl_basic(sock):
    """TTL, EXPIRE, PEXPIRE, PERSIST — key expiry and persistence."""
    print("\n=== Section 8: TTL Basic ===")
    flushall(sock)

    # EXPIRE 1 second, verify alive then expired
    send_recv(sock, "SET", "raw:ttl", "val")
    send_recv(sock, "EXPIRE", "raw:ttl", "1")
    r = send_recv(sock, "TTL", "raw:ttl")
    check(":1" in r or ":0" in r, "TTL after EXPIRE 1s -> positive", f"got {r!r}")

    r = send_recv(sock, "GET", "raw:ttl")
    check("val" in r, "GET before expiry returns value", f"got {r!r}")

    time.sleep(1.5)
    r = send_recv(sock, "GET", "raw:ttl")
    check("$-1" in r, "GET after 1.5s sleep -> expired", f"got {r!r}")

    # PEXPIRE 500ms
    send_recv(sock, "SET", "raw:pttl", "val")
    send_recv(sock, "PEXPIRE", "raw:pttl", "500")
    r = send_recv(sock, "PTTL", "raw:pttl")
    # PTTL should return a positive number (remaining ms)
    check(":-1" not in r and ":-2" not in r, "PTTL after PEXPIRE 500ms -> positive", f"got {r!r}")

    time.sleep(0.8)
    r = send_recv(sock, "GET", "raw:pttl")
    check("$-1" in r, "GET after 0.8s sleep (PEXPIRE 500ms) -> expired", f"got {r!r}")

    # PERSIST removes TTL
    send_recv(sock, "SET", "raw:persist", "val")
    send_recv(sock, "EXPIRE", "raw:persist", "10")
    r = send_recv(sock, "PERSIST", "raw:persist")
    check(":1" in r, "PERSIST returns 1", f"got {r!r}")
    r = send_recv(sock, "TTL", "raw:persist")
    check(":-1" in r, "TTL after PERSIST -> -1 (no expiry)", f"got {r!r}")


# ─── Section 9: Integer Overflow ─────────────────────────────────────────────

def test_integer_overflow(sock):
    """INCR/DECR at INT64 boundaries and on non-numeric values."""
    print("\n=== Section 9: Integer Overflow ===")
    flushall(sock)

    # INCR at INT64_MAX
    send_recv(sock, "SET", "raw:maxint", "9223372036854775807")
    r = send_recv(sock, "INCR", "raw:maxint")
    check("ERR" in r, "INCR at INT64_MAX -> error", f"got {r!r}")

    # DECR at INT64_MIN
    send_recv(sock, "SET", "raw:minint", "-9223372036854775808")
    r = send_recv(sock, "DECR", "raw:minint")
    check("ERR" in r, "DECR at INT64_MIN -> error", f"got {r!r}")

    # INCR on non-numeric string
    send_recv(sock, "SET", "raw:str", "hello")
    r = send_recv(sock, "INCR", "raw:str")
    check("ERR" in r, "INCR on non-numeric string -> error", f"got {r!r}")


# ─── Section 10: Ziplist/Quicklist Transition ────────────────────────────────

def test_ziplist_quicklist_transition(sock):
    """Push 1025 items to cross the 1024-entry ziplist threshold."""
    print("\n=== Section 10: Ziplist/Quicklist Transition ===")
    flushall(sock)

    # Pipeline 1025 LPUSHes in batches to avoid overwhelming the socket
    TOTAL = 1025
    BATCH = 200
    for start in range(0, TOTAL, BATCH):
        end = min(start + BATCH, TOTAL)
        cmds = [("LPUSH", "raw:zq", f"item_{i}") for i in range(start, end)]
        send_recv_multi(sock, cmds)

    r = send_recv(sock, "LLEN", "raw:zq")
    check(":1025" in r, "LLEN after 1025 LPUSHes", f"got {r!r}")

    # LPOP -> should return item_1024 (last pushed via LPUSH goes to head)
    r = send_recv(sock, "LPOP", "raw:zq")
    check("item_1024" in r, "LPOP returns last-pushed item (head)", f"got {r!r}")

    # RPOP -> should return item_0 (first pushed via LPUSH is at tail)
    r = send_recv(sock, "RPOP", "raw:zq")
    check("item_0" in r, "RPOP returns first-pushed item (tail)", f"got {r!r}")

    r = send_recv(sock, "LLEN", "raw:zq")
    check(":1023" in r, "LLEN after LPOP+RPOP", f"got {r!r}")


# ─── Section 10b: Response Buffer Overflow (gh #82) ──────────────────────────

def _drain_all(sock, total_timeout=10.0, idle_timeout=0.5):
    """Read everything the server sends until the socket goes idle.
    Larger timeouts than send_recv_multi because we expect ~3 MB."""
    chunks = []
    deadline = time.time() + total_timeout
    sock.settimeout(2.0)
    try:
        while time.time() < deadline:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                break
            if not data:
                break
            chunks.append(data)
            sock.settimeout(idle_timeout)
    except socket.timeout:
        pass
    sock.settimeout(None)
    return b"".join(chunks)


def test_response_buffer_overflow(sock):
    """gh #82, #49: a pipelined batch whose replies overflow the 4 MB response
    buffer is answered whole. gh #82 replaced a silent drop with one
    `-ERR response exceeds buffer` and dropped the rest of the batch; since
    #49 a full buffer is handed to the connection and every reply arrives."""
    print("\n=== Section 10b: Response Buffer Overflow (gh #82, #49) ===")
    flushall(sock)

    # Store a 1 MiB value. Each GET response is ~1,048,591 bytes
    # ($1048576\r\n + 1 MiB + \r\n). Three full responses fit in the 4 MB
    # buffer; the fourth crosses into the 194 KB safety margin.
    big = b"o" * (1024 * 1024)
    sock.sendall(encode_cmd(["SET", "raw:big", big]))
    r = recv_resp(sock)
    check(b"+OK" in r, "SET 1 MiB value", f"got {r!r}")

    cmds = [encode_cmd(["GET", "raw:big"]) for _ in range(5)]
    sock.sendall(b"".join(cmds))
    raw = _drain_all(sock)

    # 1. All five replies arrive, byte for byte, and no overflow error.
    one = b"$1048576\r\n" + big + b"\r\n"
    check(raw == one * 5, "all 5 replies past the 4 MB buffer arrive whole (#49)",
          f"len={len(raw)} of {5 * len(one)} tail={raw[-64:]!r}")
    check(b"-ERR response exceeds buffer" not in raw, "no overflow error frame (#49)")

    # 2. The connection must remain frame-synced: a subsequent small command
    #    on the same socket must succeed.
    r = send_recv(sock, "PING")
    check("+PONG" in r, "socket usable after the batch (frame-synced)",
          f"got {r!r}")
    r = send_recv(sock, "SET", "raw:tiny", "v")
    check("+OK" in r, "subsequent SET works after the batch",
          f"got {r!r}")


# ─── Section 11: Error Mid-Pipeline ─────────────────────────────────────────

def test_error_mid_pipeline(sock):
    """Server must continue processing after a mid-pipeline error."""
    print("\n=== Section 11: Error Mid-Pipeline ===")
    flushall(sock)

    # Create raw:errlist as a list
    send_recv(sock, "LPUSH", "raw:errlist", "a")

    # Pipeline: SET ok1, INCR on list (WRONGTYPE), SET ok2
    cmds = [
        ("SET", "raw:ok1", "v1"),
        ("INCR", "raw:errlist"),
        ("SET", "raw:ok2", "v2"),
    ]
    resp = send_recv_multi(sock, cmds)

    # Verify the pipeline response contains both OK and an error
    check("+OK" in resp, "First SET in pipeline succeeded", f"got {resp!r}")
    check("WRONGTYPE" in resp or "ERR" in resp, "Mid-pipeline INCR on list -> error", f"got {resp!r}")

    # Verify ok2 was set (server continued after error)
    r = send_recv(sock, "GET", "raw:ok2")
    check("v2" in r, "SET after mid-pipeline error succeeded", f"got {r!r}")

    # Verify ok1 was also set
    r = send_recv(sock, "GET", "raw:ok1")
    check("v1" in r, "SET before mid-pipeline error succeeded", f"got {r!r}")


# ─── Section 12: WATCH Cross-Connection ──────────────────────────────────────

def test_watch_cross_connection(host, port):
    """WATCH + MULTI/EXEC must abort if another connection modifies the key."""
    print("\n=== Section 12: WATCH Cross-Connection ===")

    sock_a = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_b = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock_a.connect((host, port))
        sock_b.connect((host, port))

        # Setup: SET the key via sock_a
        send_recv(sock_a, "SET", "raw:watch", "val1")

        # sock_a: WATCH the key
        r = send_recv(sock_a, "WATCH", "raw:watch")
        check("+OK" in r, "WATCH returns OK", f"got {r!r}")

        # sock_b: modify the watched key from another connection
        r = send_recv(sock_b, "SET", "raw:watch", "val2")
        check("+OK" in r, "SET from sock_b returns OK", f"got {r!r}")

        # sock_a: MULTI + SET + EXEC -> should abort (nil)
        r = send_recv(sock_a, "MULTI")
        check("+OK" in r, "MULTI returns OK", f"got {r!r}")

        r = send_recv(sock_a, "SET", "raw:watch", "val3")
        check("+QUEUED" in r or "QUEUED" in r, "SET inside MULTI returns QUEUED", f"got {r!r}")

        r = send_recv(sock_a, "EXEC")
        check("$-1" in r or "*-1" in r or "nil" in r.lower(), "EXEC aborted (nil) due to WATCH conflict", f"got {r!r}")

        # Verify the value is val2 (sock_b's write, not sock_a's transaction)
        r = send_recv(sock_a, "GET", "raw:watch")
        check("val2" in r, "GET returns sock_b value after aborted EXEC", f"got {r!r}")
    finally:
        sock_a.close()
        sock_b.close()


# ─── Section 13: Stream Basic ────────────────────────────────────────────────

def test_stream_basic(sock):
    """XADD, XLEN, XRANGE basic operations."""
    print("\n=== Section 13: Stream Basic ===")
    flushall(sock)

    # XADD with auto-generated ID
    r = send_recv(sock, "XADD", "raw:stream", "*", "field1", "val1")
    check("-" in r and "$-1" not in r, "XADD returns stream ID (contains '-')", f"got {r!r}")

    r = send_recv(sock, "XADD", "raw:stream", "*", "field2", "val2")
    check("-" in r and "$-1" not in r, "XADD second entry returns ID", f"got {r!r}")

    r = send_recv(sock, "XLEN", "raw:stream")
    check(":2" in r, "XLEN returns 2", f"got {r!r}")

    # XRANGE - get all entries
    r = send_recv(sock, "XRANGE", "raw:stream", "-", "+")
    check("field1" in r and "val1" in r, "XRANGE contains field1/val1", f"got {r!r}")
    check("field2" in r and "val2" in r, "XRANGE contains field2/val2", f"got {r!r}")


# ─── Section 14: Type Cross-Errors ───────────────────────────────────────────

def test_type_cross_errors(sock):
    """Expanded type safety: every type combination returns WRONGTYPE, values unchanged."""
    print("\n=== Section 14: Type Cross-Errors ===")
    flushall(sock)

    # Create keys of each type
    send_recv(sock, "SET", "raw:xstr", "hello")
    send_recv(sock, "LPUSH", "raw:xlist", "a")
    send_recv(sock, "HSET", "raw:xhash", "f", "v")
    send_recv(sock, "SADD", "raw:xset", "x")
    send_recv(sock, "ZADD", "raw:xzset", "1", "m")

    # LPUSH on a string key -> WRONGTYPE
    r = send_recv(sock, "LPUSH", "raw:xstr", "oops")
    check("WRONGTYPE" in r or "ERR" in r, "LPUSH on string -> WRONGTYPE", f"got {r!r}")
    r = send_recv(sock, "GET", "raw:xstr")
    check("hello" in r, "string unchanged after failed LPUSH", f"got {r!r}")

    # HSET on a list key -> WRONGTYPE
    r = send_recv(sock, "HSET", "raw:xlist", "f", "v")
    check("WRONGTYPE" in r or "ERR" in r, "HSET on list -> WRONGTYPE", f"got {r!r}")
    r = send_recv(sock, "LRANGE", "raw:xlist", "0", "-1")
    check("a" in r, "list unchanged after failed HSET", f"got {r!r}")

    # SADD on a hash key -> WRONGTYPE
    r = send_recv(sock, "SADD", "raw:xhash", "oops")
    check("WRONGTYPE" in r or "ERR" in r, "SADD on hash -> WRONGTYPE", f"got {r!r}")
    r = send_recv(sock, "HGET", "raw:xhash", "f")
    check("v" in r, "hash unchanged after failed SADD", f"got {r!r}")

    # ZADD on a set key -> WRONGTYPE
    r = send_recv(sock, "ZADD", "raw:xset", "1", "oops")
    check("WRONGTYPE" in r or "ERR" in r, "ZADD on set -> WRONGTYPE", f"got {r!r}")
    r = send_recv(sock, "SMEMBERS", "raw:xset")
    check("x" in r, "set unchanged after failed ZADD", f"got {r!r}")

    # INCR on a list key -> WRONGTYPE
    r = send_recv(sock, "INCR", "raw:xlist")
    check("WRONGTYPE" in r or "ERR" in r, "INCR on list -> WRONGTYPE", f"got {r!r}")
    r = send_recv(sock, "LRANGE", "raw:xlist", "0", "-1")
    check("a" in r, "list unchanged after failed INCR", f"got {r!r}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gate 1: Pion correctness invariants")
    parser.add_argument("--port", type=int, default=1974, help="Pion port (default: 1974)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Pion host")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
    except ConnectionRefusedError:
        print(f"ERROR: Cannot connect to Pion at {args.host}:{args.port}")
        print("Start the server first: ./pion-server -w 1")
        sys.exit(1)

    test_read_after_write(sock)
    test_sso_boundary(sock)
    test_type_safety(sock)
    test_edge_cases(sock)
    test_pipeline_stress(sock)
    test_binary_safety(sock)
    test_ttl_basic(sock)
    test_integer_overflow(sock)
    test_ziplist_quicklist_transition(sock)
    test_response_buffer_overflow(sock)
    test_error_mid_pipeline(sock)
    test_stream_basic(sock)
    test_type_cross_errors(sock)

    sock.close()

    # Tests that need fresh sockets / multiple connections
    test_concurrent_connections(args.host, args.port)
    test_watch_cross_connection(args.host, args.port)

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Gate 1 Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for e in errors:
            print(f"  {e}")
    print(f"{'='*60}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
