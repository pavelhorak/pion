#!/usr/bin/env python3
"""Cross-worker KV.PREFIX directory regression test.

Pre-fix, `--kvcache` auto-capped to `-w 1` because per-worker V-store state
made KV.PREFIX.LOOKUP on a non-owner worker silently MISS. This test pins
the post-fix behavior:

  - KV.PREFIX.REGISTER on one connection (lands on one worker) is visible
    to KV.PREFIX.LOOKUP from N fresh connections (each on a random worker).
  - V.STOREBATCH / V.FETCH on a non-owner worker return -ERR with the
    owner worker_id so clients can pin their connection.

Every probe targets one worker through its affinity port (port + 2 + id), so
each phase is deterministic; fresh connections on the shared port only add
coverage of the accept() path in phase 2.
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

from pion_store import PionStore  # uses minimal RESP client; no redis-py dep
from pion_store import _RESPClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7581)  # +1 binary lane, +2.. affinity
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


def affinity_port(port: int, worker: int) -> int:
    """Only worker `worker` accepts on this port (main.mojo: port + 2 + id)."""
    return port + 2 + worker


def live_workers(port: int, wanted: int) -> list[int]:
    """Workers whose affinity port answers. macOS caps -w at 4, so the server
    may run fewer workers than asked for; ask the ports, not the flag."""
    import socket as _s
    out = []
    for w in range(wanted):
        try:
            with _s.create_connection(("127.0.0.1", affinity_port(port, w)), timeout=1):
                out.append(w)
        except OSError:
            pass
    return out


def owner_error(fn) -> str | None:
    """Run fn(); return the error text if it raised, None if it succeeded."""
    try:
        fn()
        return None
    except Exception as e:  # _RESPClient raises on -ERR
        return str(e)


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
        workers = live_workers(args.port, args.workers)
        assert len(workers) >= 2, f"need >= 2 workers, affinity ports answered for {workers}"
        # Register on the LAST worker, so a reply that names worker 0 by
        # default cannot pass for the right answer.
        owner = workers[-1]
        others = [w for w in workers if w != owner]
        print(f"  workers {workers}, registering on worker {owner}")

        # Every connection below goes to one worker's affinity port. The old
        # version opened fresh connections on the shared port and hoped the
        # accept() race would spread them; serially-opened connections tend
        # to land on ONE worker, so 32 of them once all missed the owner.

        # ── Phase 1: REGISTER on the owner ────────────────────────────────────
        with PionStore(port=affinity_port(args.port, owner), vquant="fp16",
                       auto_redirect=False) as s:
            s.register(NS, KV_DIM)
            assert s.lookup(NS), "same-connection lookup must HIT"
            info = s.info()
            assert info.get("directory_enabled") == 1, f"directory should be enabled; info={info}"
            assert info.get("directory_entries", 0) >= 2, f"REGISTER should publish 2 entries (pk+pv); info={info}"
            print(f"  phase1 same-connection HIT, directory_entries={info['directory_entries']}")

        # ── Phase 2: LOOKUP HITs on EVERY worker, and on the shared port ─────
        for w in workers:
            c = _RESPClient("127.0.0.1", affinity_port(args.port, w))
            try:
                resp = c.cmd("KV.PREFIX.LOOKUP", NS)
            finally:
                c.close()
            assert resp == b"HIT", f"LOOKUP on worker {w} answered {resp!r}, want HIT"
        for _ in range(args.probes):
            c = _RESPClient("127.0.0.1", args.port)
            try:
                resp = c.cmd("KV.PREFIX.LOOKUP", NS)
            finally:
                c.close()
            assert resp == b"HIT", f"LOOKUP on the shared port answered {resp!r}, want HIT"
        print(f"  phase2 LOOKUP HIT on all {len(workers)} workers + {args.probes} shared-port connections")

        # ── Phase 3: V.STOREBATCH works on the owner, names it everywhere else
        rng = np.random.default_rng(0xABCD)
        fp32 = rng.standard_normal((NUM_TOK, KV_DIM)).astype(np.float32)

        def storebatch(w):
            c = _RESPClient("127.0.0.1", affinity_port(args.port, w))
            try:
                resp = c.cmd("V.STOREBATCH", f"{NS}_pv", "0", "0", str(NUM_TOK), fp32.tobytes())
                assert resp == b"OK", f"V.STOREBATCH on worker {w} answered {resp!r}"
            finally:
                c.close()

        err = owner_error(lambda: storebatch(owner))
        assert err is None, f"V.STOREBATCH on the owner (worker {owner}) failed: {err}"
        for w in others:
            err = owner_error(lambda: storebatch(w))
            assert err is not None and f"lives on worker {owner}" in err, \
                f"V.STOREBATCH on worker {w} must name owner {owner}; got {err!r}"
        print(f"  phase3 V.STOREBATCH OK on worker {owner}, owner named by {others} ✓")

        # ── Phase 4: V.FETCH likewise ────────────────────────────────────────
        def fetch(w):
            c = _RESPClient("127.0.0.1", affinity_port(args.port, w))
            try:
                return c.cmd("V.FETCH", f"{NS}_pv", "0", "RANGE", "0", str(NUM_TOK))
            finally:
                c.close()

        err = owner_error(lambda: fetch(owner))
        assert err is None, f"V.FETCH on the owner (worker {owner}) failed: {err}"
        for w in others:
            err = owner_error(lambda: fetch(w))
            assert err is not None and f"lives on worker {owner}" in err, \
                f"V.FETCH on worker {w} must name owner {owner}; got {err!r}"
        print(f"  phase4 V.FETCH OK on worker {owner}, owner named by {others} ✓")

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
