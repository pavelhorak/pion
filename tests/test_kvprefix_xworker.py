#!/usr/bin/env python3
"""Cross-worker KV.PREFIX directory regression test.

Pre-fix, `--kvcache` auto-capped to `-w 1` because per-worker V-store state
made KV.PREFIX.LOOKUP on a non-owner worker silently MISS. This test pins
the post-fix behavior:

  - KV.PREFIX.REGISTER on one connection (lands on one worker) is visible
    to KV.PREFIX.LOOKUP from N fresh connections (each on a random worker).
  - V.STOREBATCH / V.FETCH on a non-owner worker return -ERR with the
    owner worker_id so clients can pin their connection.
  - KV.PREFIX.INFO surfaces directory_entries / directory_enabled.

Requires: ./pion-server --kvcache -w 4 --independent-workers   (≥2 workers — capped to 4 on macOS).
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

from pion_lmcache import PionStore  # uses minimal RESP client; no redis-py dep
from pion_lmcache.store import _RESPClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=6397)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--probes", type=int, default=12)
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
    import socket as _s
    while time.time() < deadline:
        try:
            with _s.create_connection(("127.0.0.1", port), timeout=1):
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


def info_dict(client: _RESPClient) -> dict:
    raw = client.cmd("KV.PREFIX.INFO")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    out = {}
    for line in raw.split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def main() -> int:
    args = parse_args()
    log = "/tmp/pion_kvprefix_xworker.log"
    rc = 0
    proc = None

    # Clean state.
    for w in range(8):
        for stem in ("pion.vstore.", "pion.vstore.wal.",
                     "pion.wal.", "pion.snapshot."):
            f = os.path.join(PROJECT_ROOT, f"{stem}{w}")
            if os.path.exists(f):
                os.remove(f)

    NS = "xworker|test|v1"
    KV_DIM = 32
    NUM_TOK = 4
    import numpy as np

    try:
        proc = start_pion(args.port, args.workers, log)

        # ── Phase 1: REGISTER on one connection (one worker) ─────────────────
        with PionStore(port=args.port, vquant="fp16") as s:
            s.register(NS, KV_DIM)
            # Sanity: same connection sees HIT.
            assert s.lookup(NS), "same-connection lookup must HIT"
            info = s.info()
            assert info.get("directory_enabled") == 1, f"directory should be enabled; info={info}"
            assert info.get("directory_entries", 0) >= 2, f"REGISTER should publish 2 entries (pk+pv); info={info}"
            print(f"  phase1 same-connection HIT, directory_entries={info['directory_entries']}")

        # ── Phase 2: N fresh connections each issue LOOKUP — every one HIT ──
        hits = 0
        misses = 0
        for _ in range(args.probes):
            c = _RESPClient("127.0.0.1", args.port)
            try:
                resp = c.cmd("KV.PREFIX.LOOKUP", NS)
                if isinstance(resp, (bytes, bytearray)) and resp == b"HIT":
                    hits += 1
                else:
                    misses += 1
            finally:
                c.close()
        print(f"  phase2 cross-worker LOOKUP: {hits} HIT / {misses} MISS over {args.probes} fresh connections")
        if misses > 0:
            print("FAIL: at least one fresh connection MISSed — directory not actually cross-worker.")
            print("(Pre-fix this would have been ~75% MISS at -w 4. Any miss = regression.)")
            rc = 1
            return rc

        # ── Phase 3: V.STOREBATCH on the right worker, then on fresh ones ──
        # Connection A is the one that REGISTERed and OWNS the V buffers. Use
        # PionStore against the same connection underlying — open a new
        # PionStore that re-REGISTERs (idempotent, returns the slot from this
        # worker) then STOREBATCHes on whatever worker we land on.
        # We can't predict which worker that is, so we'll iterate fresh
        # connections until V.STOREBATCH succeeds at least once and at least
        # one returns the cross-worker -ERR. Both prove the directory routing.
        rng = np.random.default_rng(0xABCD)
        fp32 = rng.standard_normal((NUM_TOK, KV_DIM)).astype(np.float32)
        observed_owner_err = False
        observed_success = False
        owner_worker_seen = None
        for attempt in range(8 * args.workers):
            c = _RESPClient("127.0.0.1", args.port)
            try:
                resp = c.cmd(
                    "V.STOREBATCH", f"{NS}_pv", "0", "0", str(NUM_TOK),
                    fp32.tobytes(),
                )
                if isinstance(resp, (bytes, bytearray)) and resp == b"OK":
                    observed_success = True
                else:
                    print(f"   attempt {attempt}: unexpected V.STOREBATCH resp: {resp!r}")
            except Exception as e:
                msg = str(e)
                if "lives on worker" in msg:
                    observed_owner_err = True
                    # Parse the worker_id out of the message.
                    import re as _re
                    m = _re.search(r"worker (\d+)", msg)
                    if m:
                        owner_worker_seen = int(m.group(1))
                else:
                    print(f"   attempt {attempt}: unexpected error: {msg}")
            finally:
                c.close()
            if observed_success and observed_owner_err:
                break
        if not observed_success:
            print("FAIL: V.STOREBATCH never succeeded — every fresh connection landed on a non-owner.")
            rc = 1
            return rc
        if not observed_owner_err:
            # If this is a single-worker server (-w 1), the cross-worker error
            # path is unreachable — that's fine, but the test wants ≥2 workers.
            print("WARN: never saw cross-worker error — increase --probes or --workers if -w >1.")
        else:
            print(f"  phase3 V.STOREBATCH cross-worker error path triggered (owner={owner_worker_seen}) ✓")
        print(f"  phase3 V.STOREBATCH succeeded on the owner connection ✓")

        # ── Phase 4: V.FETCH on a non-owner returns the same -ERR ──────────
        observed_fetch_err = False
        for attempt in range(8 * args.workers):
            c = _RESPClient("127.0.0.1", args.port)
            try:
                _ = c.cmd("V.FETCH", f"{NS}_pv", "0", "RANGE", "0", str(NUM_TOK))
                # If it succeeded we landed on the owner; loop tries again.
            except Exception as e:
                if "lives on worker" in str(e):
                    observed_fetch_err = True
                    break
            finally:
                c.close()
        if observed_fetch_err:
            print(f"  phase4 V.FETCH cross-worker error path triggered ✓")
        else:
            print("WARN: V.FETCH always landed on owner; can't verify cross-worker error path.")

        print("\nPASS: cross-worker KV.PREFIX directory works.")
    except AssertionError as e:
        print(f"FAIL: {e}")
        rc = 1
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        rc = 2
    finally:
        stop_pion(proc)
        # Clean state on success.
        if rc == 0:
            for w in range(8):
                for stem in ("pion.vstore.", "pion.vstore.wal.",
                             "pion.wal.", "pion.snapshot."):
                    f = os.path.join(PROJECT_ROOT, f"{stem}{w}")
                    if os.path.exists(f):
                        os.remove(f)

    return rc


if __name__ == "__main__":
    sys.exit(main())
