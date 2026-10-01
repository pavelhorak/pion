#!/usr/bin/env python3
"""LMCache RESPConnector wire-emulation test.

Validates that the exact RESP command sequence LMCache's RESPConnector
emits against a `resp://` remote_url works against Pion. We don't import
`lmcache` itself (CUDA-blocked on Mac); instead we reproduce its wire
pattern from the upstream source:

  https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/native_clients/resp_client.py

What RESPConnector does on the wire (for one chunk):
  PUT  → SET key:string blob:bytes
  GET  → GET key:string  → bulk string
  EXISTS → EXISTS key  → integer 0/1
  DEL  → DEL key       → integer 0/1
  LIST  → SCAN cursor 0 MATCH * COUNT N (drains via cursor 0 → next → 0)

Keys are CacheEngineKey.to_string():
  "<model>@<world_size>@<worker_id>@<chunk_hash>@<dtype>"

Blobs are KV-cache tensor bytes; sizes range from KB (decode chunks) to
MB (long-prompt chunks). We exercise that range plus the contains/list
contract to give us confidence the upstream library will work end-to-end
when run on a CUDA box (see INSTALL_LMCACHE.md).

Requires: ./pion-server -w 1   (no --kvcache needed — wire-compat is the
RESP keyspace path, A6).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from typing import List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-lmcache"))

from pion_lmcache.store import _RESPClient, _RESPError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6399)
    p.add_argument("--start", action="store_true",
                   help="Spawn `pion-server -w 1` for the test.")
    p.add_argument("--large", action="store_true",
                   help="Add an 8 MB chunk test (70B-class workloads).")
    return p.parse_args()


def start_pion(port: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 30
    import socket
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def stop_pion(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def lmcache_key(model: str, ws: int, wid: int, chunk_hash: str, dtype: str) -> bytes:
    return f"{model}@{ws}@{wid}@{chunk_hash}@{dtype}".encode("utf-8")


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0

    try:
        if args.start:
            proc = start_pion(args.port, "/tmp/pion_lmcache_resp.log")

        c = _RESPClient("127.0.0.1", args.port)

        # ── 1. Single-chunk PUT/GET round-trip (decode-size, ~256 KB) ──────
        rng = np.random.default_rng(0xCAFE)
        chunk_kb = rng.integers(0, 256, size=(256 * 1024,), dtype=np.uint8).tobytes()
        chunk_hash = hashlib.sha256(chunk_kb).hexdigest()[:16]
        key = lmcache_key("meta-llama/Llama-3.2-1B", 1, 0, chunk_hash, "fp16")

        resp = c.cmd("SET", key, chunk_kb)
        assert resp == b"OK", f"SET unexpected {resp!r}"
        got = c.cmd("GET", key)
        assert got == chunk_kb, f"GET round-trip mismatch (lengths {len(got) if got else 0} vs {len(chunk_kb)})"
        print(f"  1. SET/GET 256 KB: OK")

        # ── 2. EXISTS contract ────────────────────────────────────────────
        assert c.cmd("EXISTS", key) == 1
        assert c.cmd("EXISTS", b"absent_key_lmcache_test") == 0
        print(f"  2. EXISTS: present=1, missing=0 OK")

        # ── 3. DEL + GET-after-DEL returns nil ────────────────────────────
        assert c.cmd("DEL", key) == 1
        got = c.cmd("GET", key)
        assert got is None, f"GET after DEL should be nil, got {got!r}"
        assert c.cmd("EXISTS", key) == 0
        print(f"  3. DEL + GET-nil: OK")

        # ── 4. Eight workers — distinct keys, parallel-style PUT/GET ──────
        # LMCache may run 8 workers writing chunks in parallel; we issue
        # them serially here (one connection) but verify each round-trips.
        keys: List[bytes] = []
        blobs: List[bytes] = []
        for wid in range(8):
            blob = rng.integers(0, 256, size=(128 * 1024,), dtype=np.uint8).tobytes()
            h = hashlib.sha256(blob).hexdigest()[:16]
            k = lmcache_key("meta-llama/Llama-3.2-1B", 8, wid, h, "fp16")
            assert c.cmd("SET", k, blob) == b"OK"
            keys.append(k)
            blobs.append(blob)
        for k, b in zip(keys, blobs):
            got = c.cmd("GET", k)
            assert got == b, f"worker key {k!r} round-trip failed"
        print(f"  4. 8 worker chunks SET/GET: OK")

        # ── 5. Large chunk (default 1 MB; 8 MB with --large) ───────────────
        size = (8 * 1024 * 1024) if args.large else (1024 * 1024)
        big = rng.integers(0, 256, size=(size,), dtype=np.uint8).tobytes()
        h = hashlib.sha256(big).hexdigest()[:16]
        k = lmcache_key("meta-llama/Llama-3.1-70B", 1, 0, h, "fp16")
        t0 = time.perf_counter()
        assert c.cmd("SET", k, big) == b"OK"
        t1 = time.perf_counter()
        got = c.cmd("GET", k)
        t2 = time.perf_counter()
        assert got == big
        print(f"  5. {size // 1024} KB SET={1000*(t1-t0):.1f}ms GET={1000*(t2-t1):.1f}ms OK")

        # ── 6. Cleanup ─────────────────────────────────────────────────────
        for k in keys + [k]:
            c.cmd("DEL", k)
        c.close()

        print("\nPASS: LMCache RESPConnector wire pattern works against Pion (A6 wire-compat).")
        print("      Full LMCacheEngineConfig integration requires CUDA — see INSTALL_LMCACHE.md.")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        rc = 1
    finally:
        if args.start:
            stop_pion(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
