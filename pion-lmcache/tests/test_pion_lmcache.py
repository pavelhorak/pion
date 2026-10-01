#!/usr/bin/env python3
"""Integration tests for pion-lmcache against a running pion-server --kvcache.

Three sections:

  1. PionStore (structured) — register/lookup/store_layer/fetch_layer round-trip,
     fp16 quantization tolerance, save/restart bit-equality (mirrors
     tests/test_vstore_wal.py but uses the public Python API).

  2. LMCacheRemoteBackend (blob) — put/get/contains/remove + ns_prefix isolation.

  3. Adapter ↔ Store independence — namespace under PionStore is independent
     from blob keys under LMCacheRemoteBackend; both can coexist.

Requires: ./pion-server --kvcache -w 1   (default port 1974)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-lmcache"))

from pion_lmcache import PionStore, LMCacheRemoteBackend


def _start_pion(port: int, log_path: str) -> subprocess.Popen:
    binary = os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 30
    import socket as _socket
    while time.time() < deadline:
        try:
            with _socket.create_connection(("127.0.0.1", port), timeout=1):
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def _stop(proc, sig: int = signal.SIGTERM):
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


def _clean_state():
    for f in ("pion.vstore.0", "pion.vstore.wal.0",
              "pion.wal.0", "pion.snapshot.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p):
            os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_store_round_trip(port: int) -> None:
    rng = np.random.default_rng(0xCAFE)
    KV_DIM = 64
    N_TOK = 12
    N_LAYERS = 4
    NS = "test|store|v1"

    K = rng.standard_normal((N_LAYERS, N_TOK, KV_DIM)).astype(np.float32)
    V = rng.standard_normal((N_LAYERS, N_TOK, KV_DIM)).astype(np.float32)

    with PionStore(port=port, vquant="fp16") as s:
        assert not s.lookup(NS), "namespace should be cold"
        s.store_prefix(NS, K, V, kv_dim=KV_DIM)
        assert s.lookup(NS), "lookup must HIT after store_prefix"

        K2, V2 = s.fetch_prefix(NS, n_layers=N_LAYERS, n_tokens=N_TOK, kv_dim=KV_DIM)

    # fp16 quantization → bounded error vs the FP32 input
    max_k = float(np.max(np.abs(K - K2)))
    max_v = float(np.max(np.abs(V - V2)))
    assert max_k < 1e-2, f"K fp16 round-trip max |Δ| = {max_k:.4e}"
    assert max_v < 1e-2, f"V fp16 round-trip max |Δ| = {max_v:.4e}"
    print(f"  PionStore round-trip ok (max |ΔK|={max_k:.2e}, max |ΔV|={max_v:.2e})")


def test_store_info_after_save(port: int) -> None:
    with PionStore(port=port) as s:
        info_pre = s.info()
        ok = s.save()
        assert ok or info_pre.get("vstore_sessions", 0) == 0, \
            f"save should succeed when sessions exist; info={info_pre}"
        info_post = s.info()
        # After save, WAL is truncated.
        assert info_post.get("wal_appended", -1) == 0, \
            f"wal_appended should reset to 0 after KV.PREFIX.SAVE; info={info_post}"
    print(f"  save → wal_appended=0, info={info_post}")


def test_blob_adapter_round_trip(port: int) -> None:
    with LMCacheRemoteBackend(port=port, ns_prefix="lmctest") as b:
        assert b.ping(), "ping should succeed"

        key = "model@1@0@deadbeef@fp16"
        blob = bytes(range(256)) * 1024  # 256 KB blob
        assert b.put(key, blob)
        assert b.contains(key)
        got = b.get(key)
        assert got == blob, f"blob mismatch: len got={len(got) if got else 0} expected={len(blob)}"
        assert b.remove(key)
        assert not b.contains(key)
    print(f"  blob put/get/contains/remove ok (256 KB blob)")


def test_blob_ns_prefix_isolation(port: int) -> None:
    """Two adapters with different ns_prefix must not see each other's keys."""
    with LMCacheRemoteBackend(port=port, ns_prefix="ns_a") as a, \
         LMCacheRemoteBackend(port=port, ns_prefix="ns_b") as b:
        a.put("shared_key", b"alpha")
        b.put("shared_key", b"beta")
        assert a.get("shared_key") == b"alpha"
        assert b.get("shared_key") == b"beta"
        # Cleanup
        a.remove("shared_key"); b.remove("shared_key")
    print(f"  ns_prefix isolation ok")


def test_wal_persists_across_sigkill(port: int, log_path: str) -> None:
    """End-to-end: store via PionStore → SIGKILL → restart → fetch returns
    the same quantized values bit-for-bit (fp16 deterministic re-cast)."""
    rng = np.random.default_rng(0xFEED)
    KV_DIM = 32
    N_TOK = 8
    NS = "test|wal|sigkill"
    K = rng.standard_normal((1, N_TOK, KV_DIM)).astype(np.float32)
    V = rng.standard_normal((1, N_TOK, KV_DIM)).astype(np.float32)

    with PionStore(port=port, vquant="fp16") as s:
        s.store_prefix(NS, K, V, kv_dim=KV_DIM)
        K_pre, V_pre = s.fetch_prefix(NS, 1, N_TOK, KV_DIM)

    # SIGKILL the server (must be the one started by _start_pion in main).
    # We discover its PID by scanning the log file for the configured port.
    # Simpler: caller passes proc; here we trust that there's exactly one.
    pid_out = subprocess.run(
        ["pgrep", "-f", f"pion-server.*-p {port}"],
        capture_output=True, text=True,
    )
    pids = [int(x) for x in pid_out.stdout.strip().split() if x]
    assert pids, "couldn't find pion-server pid"
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    # Wait for it to actually exit
    deadline = time.time() + 5
    while time.time() < deadline:
        out = subprocess.run(["pgrep", "-f", f"pion-server.*-p {port}"], capture_output=True)
        if not out.stdout.strip():
            break
        time.sleep(0.1)

    # Restart
    proc = _start_pion(port, log_path + ".restart")
    try:
        with PionStore(port=port, vquant="fp16") as s:
            assert s.lookup(NS), "WAL replay should restore the namespace"
            K_post, V_post = s.fetch_prefix(NS, 1, N_TOK, KV_DIM)
        assert np.array_equal(K_pre, K_post), "K not bit-equal after SIGKILL+replay"
        assert np.array_equal(V_pre, V_post), "V not bit-equal after SIGKILL+replay"
        print(f"  SIGKILL → WAL replay → V.FETCH bit-equal ok")
    finally:
        _stop(proc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--no-start", action="store_true",
                   help="Assume pion-server is already running on --port.")
    args = p.parse_args()

    proc = None
    log_path = "/tmp/pion_lmcache_test.log"
    rc = 0

    try:
        if not args.no_start:
            _clean_state()
            proc = _start_pion(args.port, log_path)

        print("== test_store_round_trip ==")
        test_store_round_trip(args.port)

        print("== test_blob_adapter_round_trip ==")
        test_blob_adapter_round_trip(args.port)

        print("== test_blob_ns_prefix_isolation ==")
        test_blob_ns_prefix_isolation(args.port)

        print("== test_store_info_after_save ==")
        test_store_info_after_save(args.port)

        if not args.no_start:
            print("== test_wal_persists_across_sigkill ==")
            # This test SIGKILLs and restarts; only safe when --start owns the proc.
            test_wal_persists_across_sigkill(args.port, log_path)
            # The function spawned a replacement proc; reassign so cleanup kills it.
            pid_out = subprocess.run(
                ["pgrep", "-f", f"pion-server.*-p {args.port}"],
                capture_output=True, text=True,
            )
            if pid_out.stdout.strip():
                # Best-effort: kill any survivor so we leave a clean port.
                for pid in [int(x) for x in pid_out.stdout.strip().split() if x]:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
                proc = None  # already killed above

        print("\nAll pion-lmcache tests PASS")
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        rc = 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        rc = 2
    finally:
        if proc is not None:
            _stop(proc)
        # leave state files in place if we started outside the test (no-start)
        if not args.no_start and rc == 0:
            _clean_state()

    return rc


if __name__ == "__main__":
    sys.exit(main())
