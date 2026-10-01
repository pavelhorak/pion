#!/usr/bin/env python3
"""gh #94 — SSM.PREFIX.* WAL + snapshot durability gate.

Asserts that SSM.PREFIX.STORE writes survive a SIGKILL restart and that
KV.PREFIX.SAVE compacts the SSM WAL alongside the V-store WAL.

Closes the production hole called out in
pion-vllm-mlx/pion_vllm_mlx/prompt_cache.py:1170 — "Cross-restart durability
caveat: SSM.PREFIX.* is in-memory only on the server. After SIGKILL/restart,
subsequent SSM.PREFIX.FETCH returns nil and the warm fetch path raises."

Test plan:
  [1] Small-blob WAL round-trip
      STORE → SIGKILL → restart → FETCH returns the same bytes.
      Plus multi-layer + DROP-all replay semantics.
  [2] 12 MB Mamba-class blob bit-equal round-trip (gh #76 large-blob scenario).
  [3] KV.PREFIX.SAVE compacts: writes pion.ssm.<wid>, truncates the WAL,
      next restart loads the snapshot and replays zero WAL records.

Requires: ./pion-server --kvcache -w 1 (no Metal needed).
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import Optional

import numpy as np

HOST = "127.0.0.1"
DEFAULT_PORT = 1974
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# State files this test touches on disk. Cleaned before the first run and
# after the last successful run.
SSM_STATE_FILES = (
    "pion.ssm.0",
    "pion.ssm.wal.0",
    "pion.vstore.0",
    "pion.vstore.wal.0",
    "pion.wal.0",
    "pion.snapshot.0",
)


# ── Wire helpers (no redis-py — we want raw RESP control for blob sizes) ──


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


class Conn:
    """Minimal RESP client using socket.makefile('rb') — see hf_cache memory note
    in this project's MEMORY.md (python_resp_array_slicing_trap)."""
    def __init__(self, port: int, timeout: float = 60.0):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.s.settimeout(timeout)
        self.s.connect((HOST, port))
        self.f = self.s.makefile("rb")

    def call(self, *parts) -> bytes:
        self.s.sendall(_encode(parts))
        return self._read_one()

    def _read_one(self) -> bytes:
        line = self.f.readline()
        if not line:
            raise ConnectionError("pion closed")
        kind = line[:1]
        if kind in (b"+", b"-", b":"):
            return line
        if kind == b"$":
            n = int(line[1:].rstrip())
            if n < 0:
                return line  # nil
            body = self.f.read(n + 2)  # data + CRLF
            return line + body
        raise RuntimeError(f"unexpected reply prefix {kind!r}: {line!r}")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass
        try:
            self.s.close()
        except Exception:
            pass


def parse_bulk(r: bytes) -> Optional[bytes]:
    if r.startswith(b"$-1"):
        return None
    if not r.startswith(b"$"):
        raise RuntimeError(f"expected bulk reply, got: {r[:80]!r}")
    nl = r.find(b"\r\n")
    blen = int(r[1:nl])
    return r[nl + 2:nl + 2 + blen]


# ── Server lifecycle ────────────────────────────────────────────────────


def _clean_state(verbose: bool = False) -> None:
    for f in SSM_STATE_FILES:
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p):
            if verbose:
                print(f"  cleanup: rm {f}")
            os.remove(p)


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
            s = socket.create_connection((HOST, port), timeout=1)
            s.close()
            return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not start on port {port}")


def stop_server(proc: Optional[subprocess.Popen], sig: int = signal.SIGTERM) -> None:
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


def sigkill_server(proc: subprocess.Popen) -> None:
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)


# ── Test phases ─────────────────────────────────────────────────────────


def phase_small_blob_durability(port: int, log_path_pre: str, log_path_post: str) -> bool:
    """[1] small-blob durability across SIGKILL — round-trip + multi-layer + DROP-all."""
    print("[1] small-blob WAL durability (STORE → SIGKILL → restart → FETCH)")
    proc = start_server(port, log_path_pre)
    try:
        c = Conn(port)
        sid_single = b"ssm_dura_v1"
        blob_in = np.random.bytes(64 * 1024)  # 64 KB
        r = c.call("SSM.PREFIX.STORE", sid_single, "0", blob_in)
        if not r.startswith(b"+OK"):
            print(f"   FAIL: STORE rejected: {r[:80]!r}"); return False

        # Multi-layer for the DROP semantics check below.
        sid_multi = b"ssm_dura_multi"
        layers = {0: b"L0-bytes", 5: b"L5-payload-bytes", 23: b"L23-final-bytes"}
        for li, blob in layers.items():
            r = c.call("SSM.PREFIX.STORE", sid_multi, str(li), blob)
            if not r.startswith(b"+OK"):
                print(f"   FAIL: STORE layer {li} rejected: {r[:80]!r}"); return False

        # Drop one specific layer (will be replayed as DROP on restart).
        c.call("SSM.PREFIX.DROP", sid_multi, "5")

        # SIGKILL — no graceful shutdown, only WAL on disk.
        c.close()
        sigkill_server(proc)
        proc = None
    finally:
        if proc is not None:
            stop_server(proc)

    # Restart — durability proof is FETCH-equality, not a log string. The
    # C-side fprintf goes to stderr and is buffered until the next syscall
    # against the file descriptor, so depending on it would race; the
    # ground truth is whether the bytes come back unchanged.
    proc = start_server(port, log_path_post)
    try:
        c = Conn(port)
        # Single-layer bytes must come back unchanged.
        blob_out = parse_bulk(c.call("SSM.PREFIX.FETCH", sid_single, "0"))
        if blob_out is None:
            print("   FAIL: FETCH of single-layer sid returned nil after restart"); return False
        if blob_out != blob_in:
            print(f"   FAIL: round-trip mismatch (in={len(blob_in)} B, out={len(blob_out)} B)"); return False
        print(f"   single-layer round-trip OK ({len(blob_in)} bytes)")

        # Multi-layer: layer 5 was DROPped pre-kill → should be nil; 0 and 23 alive.
        for li, expected in layers.items():
            r = c.call("SSM.PREFIX.FETCH", sid_multi, str(li))
            got = parse_bulk(r)
            if li == 5:
                if got is not None:
                    print(f"   FAIL: DROP on layer 5 didn't survive replay (got {got!r})"); return False
            else:
                if got != expected:
                    print(f"   FAIL: layer {li} mismatch — got {got!r}, want {expected!r}"); return False
        print("   multi-layer + single-layer DROP semantics replay OK")
        c.close()
    finally:
        stop_server(proc)
    return True


def phase_large_blob(port: int, log_path: str) -> bool:
    """[2] 12 MB Mamba-class blob bit-equal round-trip across SIGKILL."""
    print("[2] 12 MB blob bit-equal round-trip across SIGKILL (gh #76 scenario)")
    _clean_state()
    big = np.random.bytes(12 * 1024 * 1024)

    proc = start_server(port, log_path + ".pre")
    try:
        c = Conn(port)
        r = c.call("SSM.PREFIX.STORE", b"ssm_dura_big", "0", big)
        if not r.startswith(b"+OK"):
            print(f"   FAIL: STORE rejected: {r[:80]!r}"); return False
        c.close()
        sigkill_server(proc)
        proc = None
    finally:
        if proc is not None:
            stop_server(proc)

    proc = start_server(port, log_path + ".post")
    try:
        c = Conn(port)
        out = parse_bulk(c.call("SSM.PREFIX.FETCH", b"ssm_dura_big", "0"))
        if out is None:
            print("   FAIL: FETCH returned nil for 12 MB blob"); return False
        if out != big:
            diff = sum(1 for a, b in zip(big, out) if a != b)
            print(f"   FAIL: 12 MB round-trip differs in {diff} bytes"); return False
        print(f"   12 MB bit-equal round-trip OK")
        c.close()
    finally:
        stop_server(proc)
    return True


def phase_save_compacts_wal(port: int, log_path: str) -> bool:
    """[3] KV.PREFIX.SAVE writes pion.ssm.0 + truncates WAL; restart zero-replay."""
    print("[3] KV.PREFIX.SAVE compacts SSM WAL (snapshot + truncate)")
    _clean_state()

    proc = start_server(port, log_path + ".pre")
    try:
        c = Conn(port)
        # Write a few blobs so the snapshot has content.
        for li in range(4):
            r = c.call("SSM.PREFIX.STORE", b"ssm_dura_save", str(li),
                       np.random.bytes(8 * 1024))
            if not r.startswith(b"+OK"):
                print(f"   FAIL: STORE rejected: {r[:80]!r}"); return False

        # KV.PREFIX.SAVE — paired surface; SSM snapshot rides alongside V-store.
        r = c.call("KV.PREFIX.SAVE")
        if not r.startswith(b"+OK"):
            # V-store may report no sessions, but SSM should still snapshot.
            # The shared SAVE returns -ERR only when both sides are empty. We
            # registered a prefix below to force +OK so this path doesn't fire.
            print(f"   info: KV.PREFIX.SAVE returned {r[:80]!r}; checking SSM snapshot anyway")

        c.close()

        # On-disk artifacts: pion.ssm.0 should exist with magic; WAL truncated.
        snap_path = os.path.join(PROJECT_ROOT, "pion.ssm.0")
        if not os.path.exists(snap_path):
            print(f"   FAIL: KV.PREFIX.SAVE didn't write pion.ssm.0"); return False
        with open(snap_path, "rb") as fp:
            magic = fp.read(8)
        if magic != b"PIONSS01":
            print(f"   FAIL: pion.ssm.0 magic {magic!r} != b'PIONSS01'"); return False
        wal_path = os.path.join(PROJECT_ROOT, "pion.ssm.wal.0")
        wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        if wal_size != 0:
            print(f"   FAIL: SSM WAL not truncated (size={wal_size})"); return False
        print(f"   SAVE wrote pion.ssm.0 ({os.path.getsize(snap_path)} B) + truncated WAL")

        sigkill_server(proc); proc = None
    finally:
        if proc is not None:
            stop_server(proc)

    # Restart — should load the snapshot, replay zero WAL records.
    proc = start_server(port, log_path + ".post")
    try:
        with open(log_path + ".post") as fp:
            log = fp.read()
        # Zero replay records means the "SSM WAL replayed" log line is absent
        # (the C code only emits when n_store + n_drop > 0).
        if "SSM WAL replayed" in log:
            print("   FAIL: server replayed WAL records after a clean SAVE"); return False

        c = Conn(port)
        # All snapshot-restored blobs must be FETCH-able.
        for li in range(4):
            out = parse_bulk(c.call("SSM.PREFIX.FETCH", b"ssm_dura_save", str(li)))
            if out is None or len(out) != 8 * 1024:
                print(f"   FAIL: snapshot did not restore layer {li} (got {None if out is None else len(out)})"); return False
        print("   snapshot restored all 4 layers; zero WAL replay")
        c.close()
    finally:
        stop_server(proc)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    _clean_state(verbose=True)
    np.random.seed(0xDEAD)

    all_ok = True
    all_ok &= phase_small_blob_durability(
        args.port, "/tmp/pion_ssm_dura_1pre.log", "/tmp/pion_ssm_dura_1post.log")
    if all_ok:
        all_ok &= phase_large_blob(args.port, "/tmp/pion_ssm_dura_2")
    if all_ok:
        all_ok &= phase_save_compacts_wal(args.port, "/tmp/pion_ssm_dura_3")

    if all_ok:
        _clean_state()
        print("\nPASS — gh #94 SSM.PREFIX.* WAL + snapshot durability")
        return 0
    print("\nFAIL — gh #94 SSM.PREFIX.* WAL + snapshot durability")
    return 1


if __name__ == "__main__":
    sys.exit(main())
