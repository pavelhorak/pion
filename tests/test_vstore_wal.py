#!/usr/bin/env python3
"""V-store WAL persistence test.

Asserts that V.STOREBATCH writes survive a restart even without an explicit
KV.PREFIX.SAVE. The WAL replay path on startup should reconstruct the
session content bit-for-bit (deterministic re-quantization).

Test plan:
  1. Start pion-server with --kvcache.
  2. KV.PREFIX.REGISTER + V.STOREBATCH for one session.
  3. V.FETCH RANGE → capture round-tripped FP32 values (post-quantization).
  4. SIGKILL the server (no KV.PREFIX.SAVE — only WAL records).
  5. Restart server. WAL replay runs. KV.PREFIX.LOOKUP should HIT.
  6. V.FETCH RANGE → values must equal the pre-restart values bit-for-bit.

Also verifies KV.PREFIX.SAVE truncates the WAL: after SAVE, wal_appended=0
in KV.PREFIX.INFO, and the on-disk pion.vstore.wal.0 is empty/short.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np

try:
    import redis
except ImportError:
    print("redis-py not installed; pip install 'redis<5.0'")
    sys.exit(2)


# redis-py >= 8 defaults to RESP3 (HELLO 3) which Pion answers with
# -NOPROTO (gh #172); pin RESP2 where the kwarg exists (>= 5.0).
_RESP2_KW = {}
try:
    if int(redis.__version__.split(".")[0]) >= 5:
        _RESP2_KW = {"protocol": 2}
except Exception:
    pass

HOST = "127.0.0.1"
DEFAULT_PORT = 1974
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--keep-running", action="store_true",
                   help="Don't kill the server at the end (useful for follow-up debugging).")
    return p.parse_args()


def start_server(port: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log_fp, stderr=log_fp,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = redis.Redis(host=HOST, port=port, socket_connect_timeout=1, **_RESP2_KW)
            r.ping()
            r.close()
            return proc
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not start on port {port}")


def stop_server(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def fetch_range(r: redis.Redis, sid: bytes, layer: int, start: int, end: int, dim: int) -> np.ndarray:
    """V.FETCH <sid> <layer> RANGE <start> <end> → np.ndarray of shape (end-start, dim)."""
    res = r.execute_command(b"V.FETCH", sid, str(layer).encode(), b"RANGE",
                            str(start).encode(), str(end).encode())
    if res is None:
        raise RuntimeError(f"V.FETCH returned nil for {sid!r} layer={layer}")
    expected = (end - start) * dim * 4
    if len(res) != expected:
        raise RuntimeError(f"V.FETCH returned {len(res)} bytes, expected {expected}")
    return np.frombuffer(res, dtype=np.float32).reshape(end - start, dim)


def info_dict(r: redis.Redis) -> dict:
    raw = r.execute_command("KV.PREFIX.INFO")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    out = {}
    for line in raw.split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    args = parse_args()
    port = args.port

    log1 = "/tmp/pion_wal_test_1.log"
    log2 = "/tmp/pion_wal_test_2.log"

    # Clean prior state.
    for f in ("pion.vstore.0", "pion.vstore.wal.0",
              "pion.wal.0", "pion.snapshot.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p):
            os.remove(p)

    # Use a deterministic V tensor — replay must round-trip bit-for-bit.
    rng = np.random.default_rng(0xBEEF)
    DIM = 64
    N_TOKENS = 16
    LAYER = 0
    NS_KEY = b"wal-test-ns"
    SID_K = NS_KEY + b"_pk"   # key-side session
    SID_V = NS_KEY + b"_pv"   # value-side session
    V = rng.standard_normal((N_TOKENS, DIM)).astype(np.float32)

    proc = None
    rc = 0
    try:
        # ── Phase 1: write some V via KV.PREFIX.REGISTER + V.STOREBATCH ───────
        proc = start_server(port, log1)
        r = redis.Redis(host=HOST, port=port, decode_responses=False, **_RESP2_KW)

        r.execute_command("KV.PREFIX.REGISTER", NS_KEY, str(DIM), "fp16")
        # Store on the value-side session (writes go through V.STOREBATCH path
        # and are logged to the WAL).
        r.execute_command(b"V.STOREBATCH", SID_V, str(LAYER).encode(),
                          b"0", str(N_TOKENS).encode(), V.tobytes())

        # Fetch back; this is the "source of truth" we'll compare against post-restart.
        before = fetch_range(r, SID_V, LAYER, 0, N_TOKENS, DIM)

        info_pre = info_dict(r)
        print(f"phase1 KV.PREFIX.INFO: {info_pre}")
        wal_appended_pre = int(info_pre.get("wal_appended", "0"))
        if wal_appended_pre < 1:
            print(f"FAIL: wal_appended={wal_appended_pre} after a STOREBATCH; WAL isn't logging.")
            rc = 1
            return rc
        r.close()

        # ── Phase 2: SIGKILL — no graceful shutdown, no KV.PREFIX.SAVE.  ──────
        # If the WAL is doing its job, the records are already on disk.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
        proc = None

        # ── Phase 3: restart, expect WAL replay to restore the session. ──────
        proc = start_server(port, log2)

        # Look for the replay log line — sanity check the path ran.
        with open(log2) as fp:
            log = fp.read()
        if "WAL replayed" not in log:
            print("FAIL: server log does not show 'WAL replayed' after restart. Path didn't run.")
            print("---server log---")
            print(log)
            rc = 1
            return rc
        print("phase3 server log mentions WAL replay ✓")

        r = redis.Redis(host=HOST, port=port, decode_responses=False, **_RESP2_KW)

        # KV.PREFIX.LOOKUP must HIT (replay re-created both _pk and _pv sessions).
        lookup = r.execute_command(b"KV.PREFIX.LOOKUP", NS_KEY)
        if isinstance(lookup, (bytes, bytearray)):
            lookup = lookup.decode()
        if lookup != "HIT":
            print(f"FAIL: KV.PREFIX.LOOKUP returned {lookup!r}, expected 'HIT' after WAL replay.")
            rc = 1
            return rc
        print("phase3 KV.PREFIX.LOOKUP → HIT ✓")

        info_post = info_dict(r)
        print(f"phase3 KV.PREFIX.INFO: {info_post}")
        replayed = int(info_post.get("wal_replayed", "0"))
        if replayed < 3:
            # Expect at least 3 ops: 2 CREATEs (k_pk + k_pv) + 1 STOREBATCH.
            print(f"FAIL: wal_replayed={replayed}, expected ≥3.")
            rc = 1
            return rc

        # Bit-for-bit equality of fetched values pre- vs post-restart.
        after = fetch_range(r, SID_V, LAYER, 0, N_TOKENS, DIM)
        if not np.array_equal(before, after):
            max_abs = float(np.max(np.abs(before - after)))
            print(f"FAIL: V.FETCH after restart differs from before. max abs diff = {max_abs:.6e}")
            rc = 1
            return rc
        print(f"phase3 V.FETCH bit-equal pre/post restart ✓ (max |Δ| = 0.0)")

        # ── Phase 4: KV.PREFIX.SAVE truncates the WAL. ──────────────────────
        r.execute_command("KV.PREFIX.SAVE")
        info_after_save = info_dict(r)
        print(f"phase4 after SAVE: {info_after_save}")
        if int(info_after_save.get("wal_appended", "1")) != 0:
            print(f"FAIL: KV.PREFIX.SAVE didn't reset wal_appended to 0.")
            rc = 1
            return rc
        # On-disk WAL file should also be empty after truncate.
        wal_path = os.path.join(PROJECT_ROOT, "pion.vstore.wal.0")
        wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else -1
        if wal_size != 0:
            print(f"FAIL: pion.vstore.wal.0 size = {wal_size} bytes after KV.PREFIX.SAVE; should be 0.")
            rc = 1
            return rc
        print(f"phase4 WAL truncated to 0 bytes on disk ✓")
        r.close()

        print("\nPASS: V-store WAL persists STOREBATCH across SIGKILL; SAVE compacts WAL.")
    finally:
        if not args.keep_running:
            stop_server(proc)
        # Clean state files only if we ran to completion successfully.
        if rc == 0 and not args.keep_running:
            for f in ("pion.vstore.0", "pion.vstore.wal.0",
                      "pion.wal.0", "pion.snapshot.0"):
                p = os.path.join(PROJECT_ROOT, f)
                if os.path.exists(p):
                    os.remove(p)

    return rc


if __name__ == "__main__":
    sys.exit(main())
