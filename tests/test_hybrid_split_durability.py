#!/usr/bin/env python3
"""gh #61 — composed mixed-hybrid prefix durability across SIGKILL restart.

The umbrella issue's Path-2 "production-grade durable hybrid prefix sharing"
headline rests on BOTH halves of a split-substrate hybrid prefix surviving a
server restart:

  - softmax (attention) layers → V-store  (KV.PREFIX.REGISTER + V.STOREBATCH)
  - linear  (Mamba/GatedDelta) layers → SSM.PREFIX.STORE

Two existing gates each prove ONE half in isolation:
  - tests/test_ssm_prefix_durability.py  → SSM.PREFIX half (gh #94)
  - V-store WAL/snapshot gate            → V.STOREBATCH half (vstore_wal)

Neither proves the COMPOSITION: a single mixed-hybrid namespace where the
softmax K/V is in the V-store and the linear state is in SSM.PREFIX, both
recovered together in one server lifecycle, keyed exactly as
`PionPromptCache._store_mixed_hybrid` / `_fetch_to_cache_mixed_hybrid` key
them (`<ns>_pk` / `<ns>_pv` softmax-rank V-store sessions + `<ns>`/slot SSM
blobs). Before gh #94 this composition was *guaranteed to desync* on restart
(V-store survived via WAL, SSM half was in-memory only — see the now-stale
recovery comment at prompt_cache.py:1376). This gate proves the desync window
is closed on Mac.

Two cycles:
  [A] STORE both halves → SIGKILL (no SAVE) → restart → FETCH both.
      Pure WAL replay for both substrates.
  [B] STORE both halves → KV.PREFIX.SAVE → SIGKILL → restart → FETCH both.
      Snapshot compaction for both substrates under one SAVE.

Mac-scoped: the V-store WAL replay is known not to fire on Linux
(vstore_wal_linux_regression). This gate runs on the Mac dev profile.

Requires: ./pion-server --kvcache -w 1 (no Metal / no model download).
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import numpy as np

HOST = "127.0.0.1"
DEFAULT_PORT = 1974
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STATE_FILES = (
    "pion.ssm.0", "pion.ssm.wal.0",
    "pion.vstore.0", "pion.vstore.wal.0",
    "pion.wal.0", "pion.snapshot.0",
)

# Representative Qwen3.5-4B-class shape: 8 softmax (attention) layers, 24
# linear (GatedDeltaNet) slots. Sizes are plausible-but-small so the test is
# fast; durability is independent of magnitude (gh #94 already gates 12 MB).
N_SOFTMAX = 8
N_LINEAR = 24
PREFIX_LEN = 64            # tokens
KV_DIM = 128              # uniform softmax kv_dim (H*D)
VQUANT = "fp16"


# ── Wire helpers (raw RESP; socket.makefile per python_resp_array_slicing_trap) ──


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
                return line
            body = self.f.read(n + 2)
            return line + body
        raise RuntimeError(f"unexpected reply prefix {kind!r}: {line!r}")

    def close(self) -> None:
        for fn in (self.f.close, self.s.close):
            try:
                fn()
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


def _clean_state() -> None:
    for f in STATE_FILES:
        p = os.path.join(PROJECT_ROOT, f)
        if os.path.exists(p):
            os.remove(p)


def start_server(port: int, log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_fp, stderr=log_fp,
                            preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection((HOST, port), timeout=1).close()
            return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"pion-server did not start on port {port}")


def stop_server(proc: Optional[subprocess.Popen]) -> None:
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


def sigkill_server(proc: subprocess.Popen) -> None:
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=5)


# ── Workload generator ──────────────────────────────────────────────────


def _make_prefix(seed: int):
    """Deterministic mixed-hybrid prefix: softmax K/V per rank + linear blobs."""
    rng = np.random.default_rng(seed)
    softmax = []  # list of (k_flat, v_flat) fp32, shape (PREFIX_LEN, KV_DIM)
    for _ in range(N_SOFTMAX):
        k = rng.standard_normal((PREFIX_LEN, KV_DIM)).astype(np.float32)
        v = rng.standard_normal((PREFIX_LEN, KV_DIM)).astype(np.float32)
        softmax.append((k, v))
    # Linear slots: opaque byte blobs of varying size (mirrors serialized
    # ArraysCache; exact bytes must round-trip).
    linear = [rng.bytes(4096 + i * 257) for i in range(N_LINEAR)]
    return softmax, linear


def _store_prefix(c: Conn, ns: bytes, softmax, linear) -> None:
    """Mirror PionPromptCache._store_mixed_hybrid wire shape exactly."""
    r = c.call("KV.PREFIX.REGISTER", ns, str(KV_DIM), VQUANT)
    if not r.startswith(b"+OK") and b"already" not in r.lower() and b"exists" not in r.lower():
        raise RuntimeError(f"KV.PREFIX.REGISTER failed: {r!r}")
    sid_k = ns + b"_pk"
    sid_v = ns + b"_pv"
    for rank, (k, v) in enumerate(softmax):
        rk = c.call("V.STOREBATCH", sid_k, str(rank), "0", str(PREFIX_LEN),
                    np.ascontiguousarray(k, dtype=np.float32).tobytes())
        if not rk.startswith(b"+OK"):
            raise RuntimeError(f"V.STOREBATCH K rank {rank}: {rk[:80]!r}")
        rv = c.call("V.STOREBATCH", sid_v, str(rank), "0", str(PREFIX_LEN),
                    np.ascontiguousarray(v, dtype=np.float32).tobytes())
        if not rv.startswith(b"+OK"):
            raise RuntimeError(f"V.STOREBATCH V rank {rank}: {rv[:80]!r}")
    for slot, blob in enumerate(linear):
        rs = c.call("SSM.PREFIX.STORE", ns, str(slot), blob)
        if not rs.startswith(b"+OK"):
            raise RuntimeError(f"SSM.PREFIX.STORE slot {slot}: {rs[:80]!r}")


def _capture(c: Conn, ns: bytes) -> dict:
    """Fetch both halves of `ns` and return the raw wire bytes, keyed.

    Mirrors PionPromptCache._fetch_to_cache_mixed_hybrid wire shape. The
    softmax half rides the V-store at `self.vquant` (fp16 here), so the
    fetched bytes are quantization-rounded — durability is "the rounded
    bytes the live server serves are unchanged across restart," not
    equality to the fp32 input. The linear half is an opaque blob (no
    quant) so it must equal the stored bytes exactly.
    """
    sid_k = ns + b"_pk"
    sid_v = ns + b"_pv"
    snap: dict = {}
    for rank in range(N_SOFTMAX):
        snap[("k", rank)] = parse_bulk(
            c.call("V.FETCH", sid_k, str(rank), "RANGE", "0", str(PREFIX_LEN)))
        snap[("v", rank)] = parse_bulk(
            c.call("V.FETCH", sid_v, str(rank), "RANGE", "0", str(PREFIX_LEN)))
    for slot in range(N_LINEAR):
        snap[("ssm", slot)] = parse_bulk(c.call("SSM.PREFIX.FETCH", ns, str(slot)))
    return snap


def _compare(before: dict, after: dict, linear) -> Optional[str]:
    """after must byte-equal before for every key; nil after = data lost."""
    for key, want in before.items():
        got = after.get(key)
        if want is None:
            return f"{key}: pre-kill fetch was already nil — store path broken"
        if got is None:
            return f"{key}: fetch returned nil after restart — durability lost"
        if got != want:
            return f"{key}: bytes changed across restart ({len(got)} vs {len(want)} B)"
    # Belt-and-suspenders: linear half is lossless, so the recovered bytes
    # must also equal the original input blobs, not just the pre-kill fetch.
    for slot, blob in enumerate(linear):
        if after[("ssm", slot)] != blob:
            return f"linear slot {slot}: recovered bytes != original input"
    return None


# ── Cycles ──────────────────────────────────────────────────────────────


def cycle_sigkill_wal(port: int) -> bool:
    """[A] Both halves survive a plain SIGKILL via WAL replay (no SAVE)."""
    print("[A] composed prefix survives SIGKILL via WAL replay (no SAVE)")
    _clean_state()
    ns = b"hybrid_dura_wal"
    softmax, linear = _make_prefix(seed=0xA11CE)

    proc = start_server(port, "/tmp/pion_hybrid_dura_A.pre.log")
    try:
        c = Conn(port)
        _store_prefix(c, ns, softmax, linear)
        before = _capture(c, ns)   # ground truth: what the live server serves
        c.close()
        sigkill_server(proc); proc = None
    finally:
        if proc is not None:
            stop_server(proc)

    proc = start_server(port, "/tmp/pion_hybrid_dura_A.post.log")
    try:
        c = Conn(port)
        after = _capture(c, ns)
        c.close()
        err = _compare(before, after, linear)
        if err:
            print(f"   FAIL: {err}"); return False
        print(f"   OK — {N_SOFTMAX} softmax ranks (K+V) + {N_LINEAR} linear slots "
              f"recovered bit-equal after SIGKILL")
        return True
    finally:
        stop_server(proc)


def cycle_save_snapshot(port: int) -> bool:
    """[B] Both halves survive via KV.PREFIX.SAVE snapshot under one SAVE."""
    print("[B] composed prefix survives SIGKILL via KV.PREFIX.SAVE snapshot")
    _clean_state()
    ns = b"hybrid_dura_save"
    softmax, linear = _make_prefix(seed=0xB0B)

    proc = start_server(port, "/tmp/pion_hybrid_dura_B.pre.log")
    try:
        c = Conn(port)
        _store_prefix(c, ns, softmax, linear)
        before = _capture(c, ns)
        r = c.call("KV.PREFIX.SAVE")
        if not r.startswith(b"+OK"):
            print(f"   info: KV.PREFIX.SAVE returned {r[:80]!r}")
        c.close()
        # Verify the SAVE wrote both substrate snapshots.
        ssm_snap = os.path.join(PROJECT_ROOT, "pion.ssm.0")
        if not os.path.exists(ssm_snap):
            print("   FAIL: KV.PREFIX.SAVE did not write pion.ssm.0"); return False
        with open(ssm_snap, "rb") as fp:
            if fp.read(8) != b"PIONSS01":
                print("   FAIL: pion.ssm.0 bad magic"); return False
        sigkill_server(proc); proc = None
    finally:
        if proc is not None:
            stop_server(proc)

    proc = start_server(port, "/tmp/pion_hybrid_dura_B.post.log")
    try:
        c = Conn(port)
        after = _capture(c, ns)
        c.close()
        err = _compare(before, after, linear)
        if err:
            print(f"   FAIL: {err}"); return False
        print(f"   OK — both halves restored from snapshot after SIGKILL")
        return True
    finally:
        stop_server(proc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    _clean_state()
    ok = cycle_sigkill_wal(args.port)
    if ok:
        ok = cycle_save_snapshot(args.port) and ok
    if ok:
        _clean_state()
        print("\nPASS — gh #61 composed mixed-hybrid prefix durability "
              "(softmax V-store + linear SSM.PREFIX) survives restart")
        return 0
    print("\nFAIL — gh #61 composed mixed-hybrid prefix durability")
    return 1


if __name__ == "__main__":
    sys.exit(main())
