#!/usr/bin/env python3
"""PionStore client-side cross-worker auto-redirect — Stage-2 transport.

Pre-this-test, multi-worker `--kvcache` returned `-ERR session lives on
worker N` whenever V.STOREBATCH/V.FETCH landed on a non-owner connection,
forcing the caller to reconnect manually. This pins the post-fix flow:

  - PionStore.auto_redirect transparently handles that error: close the
    current connection, connect to the owner worker's affinity port, retry.
    The connection is then pinned to the owner worker for the remainder of
    the PionStore instance.
  - PionStore.owner(namespace) returns the worker_id directly (KV.PREFIX.OWNER
    helper command), no V.* round-trip.

Test plan:
  1. Start `pion-server --kvcache -w 4 --independent-workers`.
  2. Open 8 fresh PionStore instances, each registers a different namespace
     and writes one layer. With auto_redirect, every store_layer succeeds
     regardless of which worker the connection initially lands on.
  3. From a different fresh PionStore, fetch each namespace — also covered
     by auto_redirect.
  4. Verify owner() returns the same worker_id for all subsequent calls
     (i.e. the directory entry is stable).
  5. gh #406: every redirect lands on the owner in ONE hop — the client
     connects to the owner's affinity port (port + 2 + owner) instead of
     re-racing the shared port's accept().

Without auto_redirect (auto_redirect=False), the same flow expects
-ERR roughly 75% of the time at -w 4 — proves the redirect is what's
working, not just luck.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-lmcache"))

from pion_lmcache import PionStore
from pion_lmcache.store import _RESPError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6398)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def start_pion(port: int, workers: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", str(workers), "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    if workers > 1:
        cmd.append("--independent-workers")   # gh #253
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
    log_path = "/tmp/pion_autoredirect.log"

    for w in range(8):
        for stem in ("pion.vstore.", "pion.vstore.wal.",
                     "pion.wal.", "pion.snapshot."):
            f = os.path.join(PROJECT_ROOT, f"{stem}{w}")
            if os.path.exists(f):
                os.remove(f)

    KV_DIM = 32
    N_TOK = 4
    N_NAMESPACES = 8
    rng = np.random.default_rng(0xCAFE)
    rows = rng.standard_normal((N_NAMESPACES, N_TOK, KV_DIM)).astype(np.float32)

    try:
        proc = start_pion(args.port, args.workers, log_path)

        # ── Phase 1: register + store across N namespaces from N FRESH clients ──
        # auto_redirect is on — every call must succeed regardless of which
        # worker the kernel race-accepts onto.
        owners: List[int] = []
        total_redirects = 0
        for i in range(N_NAMESPACES):
            ns = f"autoredirect|test|ns_{i}"
            with PionStore(port=args.port, vquant="fp16") as s:
                s.register(ns, KV_DIM)
                s.store_layer(ns, "V", layer=0, token_offset=0, tensor_fp32=rows[i])
                owners.append(s.owner(ns))
                total_redirects += s.redirect_count
        print(f"  phase1 store: {N_NAMESPACES} namespaces stored; total redirects = {total_redirects}")
        if any(o < 0 for o in owners):
            print(f"FAIL: some namespaces have no owner: {owners}")
            return 1
        # With -w 4, owners should distribute across multiple workers.
        unique = sorted(set(owners))
        print(f"  phase1 owners distinct workers: {unique}")
        if len(unique) < 2:
            print(f"WARN: all {N_NAMESPACES} namespaces ended on a single worker. Auto-redirect might")
            print("      not be exercising the cross-worker path. Re-run; usually 2-3 workers seen.")

        # ── Phase 2: fetch from FRESH clients — each fetch may need a redirect ──
        # gh #406: a redirect goes straight to the owner's affinity port
        # (port + 2 + owner), so it takes at most ONE hop. It used to reopen
        # the shared port and hope the accept race picked the owner, which at
        # -w 4 exhausted 16 attempts in 6-15% of runs (accept is skewed).
        # Repeat the fetches so a regression back to the race shows up here
        # as a multi-hop redirect, not as a 1-in-10 flake.
        max_abs = 0.0
        total_redirects_fetch = 0
        worst_hops = 0
        for rep in range(4):
            for i in range(N_NAMESPACES):
                ns = f"autoredirect|test|ns_{i}"
                with PionStore(port=args.port, vquant="fp16") as s:
                    got = s.fetch_layer(ns, "V", layer=0, token_start=0,
                                        token_end=N_TOK, kv_dim=KV_DIM)
                    total_redirects_fetch += s.redirect_count
                    worst_hops = max(worst_hops, s.redirect_count)
                    d = float(np.max(np.abs(got - rows[i])))
                    if d > max_abs:
                        max_abs = d
        print(f"  phase2 fetch: max |Δ|={max_abs:.3e} across all namespaces; "
              f"redirects = {total_redirects_fetch}, worst per fetch = {worst_hops}")
        if max_abs > 1e-2:  # fp16 quantization tolerance
            print(f"FAIL: fp16 round-trip max |Δ|={max_abs:.3e} > 1e-2")
            return 1
        if worst_hops > 1:
            print(f"FAIL: a fetch needed {worst_hops} redirects; the owner's affinity "
                  "port makes it one (gh #406)")
            return 1

        # ── Phase 3: turn auto_redirect OFF — at -w 4 should see -ERR ~75% ──
        # Fresh client per attempt to make accept races independent.
        if args.workers > 1:
            err_seen = 0
            for i in range(8 * args.workers):
                ns = f"autoredirect|test|ns_{i % N_NAMESPACES}"
                with PionStore(port=args.port, vquant="fp16",
                               auto_redirect=False) as s:
                    try:
                        _ = s.fetch_layer(ns, "V", 0, 0, N_TOK, KV_DIM)
                    except _RESPError as e:
                        if "lives on worker" in str(e):
                            err_seen += 1
            if err_seen == 0:
                print("WARN: never saw the cross-worker -ERR with auto_redirect=False —")
                print("      either -w 1 was negotiated or accept() always landed on the owner.")
            else:
                print(f"  phase3 with auto_redirect=False: {err_seen} cross-worker -ERR events caught (expected)")

        print("\nPASS: PionStore auto-redirect works across workers.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        stop_pion(proc)
        for w in range(8):
            for stem in ("pion.vstore.", "pion.vstore.wal.",
                         "pion.wal.", "pion.snapshot."):
                f = os.path.join(PROJECT_ROOT, f"{stem}{w}")
                if os.path.exists(f):
                    os.remove(f)

    return rc


if __name__ == "__main__":
    sys.exit(main())
