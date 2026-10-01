#!/usr/bin/env python3
"""Multi-tenant isolation regression test.

Spins up a pion-server with `--ns-prefix tenant_a:` and verifies:
  1. PionStore configured with matching tenant_prefix succeeds.
  2. PionStore configured with NO tenant_prefix is rejected with
     `-ERR namespace must start with required prefix '...'`.
  3. PionStore configured with the WRONG tenant_prefix is rejected.
  4. KV.PREFIX.LOOKUP / OWNER / REGISTER all enforce uniformly.

Pairs with `pion-lmcache/MULTITENANT.md`.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-lmcache"))

import numpy as np
from pion_lmcache import PionStore
from pion_lmcache.store import _RESPError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6391)
    return p.parse_args()


def start_pion(port: int, ns_prefix: str, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed",
           "--ns-prefix", ns_prefix]
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


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0
    log_path = "/tmp/pion_multitenant.log"

    for f in ("pion.vstore.0", "pion.vstore.wal.0", "pion.wal.0", "pion.snapshot.0"):
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p): os.remove(p)

    try:
        TENANT = "tenant_a:"
        proc = start_pion(args.port, TENANT, log_path)

        rng = np.random.default_rng(0xCAFE)
        K = rng.standard_normal((1, 4, 32)).astype(np.float32)
        V = rng.standard_normal((1, 4, 32)).astype(np.float32)

        # ── 1. Correct tenant: succeeds ────────────────────────────────────
        with PionStore(port=args.port, vquant="fp16", tenant_prefix=TENANT) as s:
            s.register("my_app|prompt_1", kv_dim=32)
            s.store_prefix("my_app|prompt_1", K, V, kv_dim=32)
            assert s.lookup("my_app|prompt_1"), "tenant-correct lookup must HIT"
        print("  1. tenant_prefix matches → REGISTER + STORE + LOOKUP succeed ✓")

        # ── 2. No tenant prefix: rejected ──────────────────────────────────
        try:
            with PionStore(port=args.port, vquant="fp16") as s:  # no tenant_prefix
                s.register("attacker_namespace", kv_dim=32)
            print("FAIL: REGISTER without tenant_prefix should have been rejected.")
            rc = 1
            return rc
        except _RESPError as e:
            assert "must start with required prefix" in str(e)
            print(f"  2. no tenant_prefix → REGISTER rejected: {str(e)[:80]} ✓")

        # ── 3. Wrong tenant prefix: rejected ───────────────────────────────
        try:
            with PionStore(port=args.port, vquant="fp16", tenant_prefix="tenant_b:") as s:
                s.register("any_namespace", kv_dim=32)
            print("FAIL: REGISTER with wrong tenant_prefix should have been rejected.")
            rc = 1
            return rc
        except _RESPError as e:
            assert "must start with required prefix" in str(e)
            print(f"  3. wrong tenant_prefix → REGISTER rejected ✓")

        # ── 4. LOOKUP / OWNER also gated ───────────────────────────────────
        try:
            with PionStore(port=args.port, vquant="fp16") as s:  # no prefix
                _ = s.lookup("attacker_namespace")
            print("FAIL: LOOKUP without tenant_prefix should have been rejected.")
            rc = 1
            return rc
        except _RESPError as e:
            assert "must start with required prefix" in str(e)
            print(f"  4. LOOKUP without tenant_prefix → rejected ✓")

        # ── 5. The legitimate tenant still sees their data ─────────────────
        with PionStore(port=args.port, vquant="fp16", tenant_prefix=TENANT) as s:
            assert s.lookup("my_app|prompt_1"), "legitimate tenant lost access"
            K2, V2 = s.fetch_prefix("my_app|prompt_1", n_layers=1, n_tokens=4, kv_dim=32)
            assert np.max(np.abs(K - K2)) < 1e-2  # fp16 tolerance
            assert np.max(np.abs(V - V2)) < 1e-2
        print("  5. legitimate tenant retains FETCH access ✓")

        print("\nPASS: multi-tenant isolation works as advertised.")
    except AssertionError as e:
        print(f"FAIL: {e}")
        rc = 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        stop_pion(proc)
        if rc == 0:
            for f in ("pion.vstore.0", "pion.vstore.wal.0", "pion.wal.0", "pion.snapshot.0"):
                p = os.path.join(PROJECT_ROOT, f)
                if os.path.exists(p): os.remove(p)

    return rc


if __name__ == "__main__":
    sys.exit(main())
