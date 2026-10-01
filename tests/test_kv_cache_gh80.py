#!/usr/bin/env python3
"""gh #80 — KV cache substrate lifecycle gate: LRU eviction, TTL expiry, MODEL filter.

Validates the fixes for the silently-degrading KV.STORE/KV.FETCH path:

  [1] LRU eviction: filling KVCACHE_MAX_ENTRIES+N entries never returns
      "cache full" — KV.INFO reports entries <= capacity and evictions >= N.
  [2] LRU ordering: after filling to capacity, touching entry #0 via FETCH
      protects it; the next STORE evicts the least-recently-accessed entry
      instead (entry #0 still fetchable).
  [3] TTL lazy expiry: STORE with TTL 1 hits immediately, misses after
      expiry, and KV.INFO expired counter increments.
  [4] MODEL filter: FETCH MODEL <tag> only returns entries stored with a
      byte-identical tag; FETCH without MODEL matches any.

Uses small dims/blobs to keep the 1000-entry fill fast. The script manages
the server lifecycle itself (boot, probe, kill).

Requires: ./pion-server-dev (or ./pion-server) built with the gh #80 fix.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import numpy as np

PORT = 1979  # non-1974 so the gate doesn't trip the PionMesh iOS app conflict
HOST = "127.0.0.1"
DIM = 768  # overwritten from KV.INFO `dimensions:` at boot
CAPACITY = 1000  # KVCACHE_MAX_ENTRIES in src/network/kv_cache_store.mojo
OVERFLOW = 1005  # crosses KVCACHE_HNSW_CAPACITY (2000 inserts) → rebuild fires


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode())
            out.append(p)
            out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


class Client:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((HOST, PORT))
        self.sock.settimeout(10)
        self.rf = self.sock.makefile("rb")

    def cmd(self, *parts):
        self.sock.sendall(_encode(parts))
        return self._read_reply()

    def _read_reply(self):
        line = self.rf.readline()
        if not line:
            raise ConnectionError("server closed connection")
        prefix, rest = line[:1], line[1:-2]
        if prefix in (b"+", b"-"):
            return line[:-2]
        if prefix == b":":
            return int(rest)
        if prefix == b"$":
            n = int(rest)
            if n < 0:
                return None
            data = self.rf.read(n + 2)
            return data[:-2]
        raise ValueError(f"unexpected reply prefix: {line!r}")

    def close(self):
        self.rf.close()
        self.sock.close()


def make_embedding(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


def kv_info(c: Client) -> dict:
    raw = c.cmd("KV.INFO")
    assert raw is not None, "KV.INFO returned nil"
    stats = {}
    for ln in raw.decode().strip().split("\r\n"):
        if ":" in ln:
            k, v = ln.split(":", 1)
            stats[k] = v
    return stats


def boot_server() -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or ("./pion-server-dev" if os.path.exists("./pion-server-dev") else "./pion-server")
    # CI injects --epoll --no-auto-embed (hosted runners seccomp-block io_uring
    # and lack the embed sidecar's `transformers`); local Mac runs leave it unset.
    # KV.STORE carries its own embedding blob, so no embed backend is needed —
    # dimensions default to 768 and the test reads the value from KV.INFO.
    extra = os.environ.get("PION_SERVER_EXTRA_ARGS", "").split()
    proc = subprocess.Popen(
        [binary, "--kvcache", "-w", "1", "-p", str(PORT), *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            c = Client()
            c.cmd("PING")
            c.close()
            return proc
        except (ConnectionRefusedError, OSError):
            if proc.poll() is not None:
                raise RuntimeError(f"{binary} exited during boot (rc={proc.returncode})")
            time.sleep(0.3)
    proc.kill()
    raise RuntimeError("server did not come up in 30s")


def test_lru_eviction(c: Client):
    print("[1] LRU eviction: filling past capacity (incl. HNSW rebuild at 2x) ...")
    blob = b"GH80_BLOB_" * 10
    t0 = time.time()
    # OVERFLOW > CAPACITY so total inserts cross KVCACHE_HNSW_CAPACITY (2x)
    # and the internal-node rebuild path fires mid-fill.
    for i in range(CAPACITY + OVERFLOW):
        resp = c.cmd("KV.STORE", f"gh80_{i}", make_embedding(i), blob)
        assert resp == b"+OK", f"STORE #{i} rejected: {resp!r} — eviction not working"
        if i % 250 == 0:
            print(f"    stored {i} ({time.time()-t0:.1f}s)")
    stats = kv_info(c)
    entries, evictions = int(stats["entries"]), int(stats["evictions"])
    assert entries <= CAPACITY, f"entries {entries} > capacity {CAPACITY}"
    assert evictions >= OVERFLOW, f"evictions {evictions} < {OVERFLOW}"
    # A freshly stored entry must be fetchable (also proves post-rebuild search works)
    last = c.cmd("KV.FETCH", make_embedding(CAPACITY + OVERFLOW - 1))
    assert last is not None and blob in last, "freshly stored entry not fetchable"
    print(f"    OK — entries={entries} evictions={evictions}")


def test_lru_ordering(c: Client):
    print("[2] LRU ordering: touched entry survives the next eviction ...")
    # State from [1]: slots hold entries (OVERFLOW..CAPACITY+OVERFLOW-1) by LRU
    # of store time. Touch the oldest surviving entry, then force one eviction.
    oldest = OVERFLOW  # first entry not yet evicted after test [1]
    touched = c.cmd("KV.FETCH", make_embedding(oldest))
    assert touched is not None, f"entry #{oldest} unexpectedly already evicted"
    resp = c.cmd("KV.STORE", "gh80_extra", make_embedding(999_001), b"X" * 64)
    assert resp == b"+OK"
    survived = c.cmd("KV.FETCH", make_embedding(oldest))
    assert survived is not None, "recently-touched entry was evicted — not LRU"
    # The second-oldest (untouched) entry should be the one that went
    gone = c.cmd("KV.FETCH", make_embedding(oldest + 1))
    assert gone is None, "untouched LRU candidate still present — eviction order wrong"
    print("    OK — touched entry survived, untouched LRU victim evicted")


def test_ttl_expiry(c: Client):
    print("[3] TTL lazy expiry ...")
    emb = make_embedding(7_000_001)
    resp = c.cmd("KV.STORE", "gh80_ttl", emb, b"TTL_BLOB" * 8, "TTL", "1")
    assert resp == b"+OK"
    hit = c.cmd("KV.FETCH", emb)
    assert hit is not None and b"TTL_BLOB" in hit, "fresh TTL entry should hit"
    expired_before = int(kv_info(c)["expired"])
    time.sleep(1.5)
    miss = c.cmd("KV.FETCH", emb)
    assert miss is None, f"expired entry still served: {miss[:40]!r}"
    expired_after = int(kv_info(c)["expired"])
    assert expired_after > expired_before, "expired counter did not increment"
    print(f"    OK — hit before expiry, nil after, expired={expired_after}")


def test_model_filter(c: Client):
    print("[4] MODEL filter ...")
    emb = make_embedding(8_000_001)
    resp = c.cmd("KV.STORE", "gh80_model", emb, b"LLAMA_BLOB" * 8, "MODEL", "llama-3.2-1b")
    assert resp == b"+OK"
    wrong = c.cmd("KV.FETCH", emb, "MODEL", "mistral-7b")
    assert wrong is None, f"MODEL mismatch returned a blob: {wrong[:40]!r}"
    right = c.cmd("KV.FETCH", emb, "MODEL", "llama-3.2-1b")
    assert right is not None and b"LLAMA_BLOB" in right, "matching MODEL missed"
    anymodel = c.cmd("KV.FETCH", emb)
    assert anymodel is not None and b"LLAMA_BLOB" in anymodel, "no-MODEL fetch missed"
    print("    OK — mismatch nil, match hit, no-filter hit")


def main() -> int:
    print(f"gh #80 gate — booting server on :{PORT}")
    proc = boot_server()
    try:
        c = Client()
        global DIM
        DIM = int(kv_info(c)["dimensions"])
        print(f"server embedding dimensions: {DIM}")
        test_lru_eviction(c)
        test_lru_ordering(c)
        test_ttl_expiry(c)
        test_model_filter(c)
        c.close()
        print("\n=== ALL gh #80 TESTS PASSED ===")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
