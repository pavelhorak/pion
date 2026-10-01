#!/usr/bin/env python3
"""ATTEND.PREFIX.QUERY LSE wire path integration test.

Validates that the §31 protocol extension reaches end-to-end:
  Mojo handle_attend_prefix_query → bridge.query_cached_blocking → MLX sidecar
  query_cached → wire body [output ‖ LSE] → Python attend_query(with_lse=True).

Then proves the LSE numbers are correct by online-merging two adjacent
prefix queries (same K/V split into halves) and comparing against a single
query over the full K/V — must be bit-close.

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "pion-lmcache"))

# Pion-lmcache's _RESPClient has no mlx dependency, so it imports cleanly even
# on hosts without mlx installed (e.g. CI runners that just probe the wire).
from pion_lmcache.store import _RESPClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1974)
    p.add_argument("--start", action="store_true",
                   help="Start pion-server with --kvcache -w 1.")
    return p.parse_args()


def start_server(port: int) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "--kvcache", "-w", "1", "-p", str(port),
           "--no-auto-detect", "--no-auto-embed"]
    log = open("/tmp/pion_attend_lse.log", "w")
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=log,
                             preexec_fn=os.setsid)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            import socket
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                # Allow the MLX sidecar to come up too — first ATTEND.PREFIX
                # call waits on it.
                time.sleep(2)
                return proc
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("pion-server didn't start")


def stop_server(proc):
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


def softmax_lse_np(scores, axis=-1):
    m = np.max(scores, axis=axis, keepdims=True)
    s = np.sum(np.exp(scores - m), axis=axis, keepdims=True)
    return (m + np.log(s)).squeeze(axis)


def main() -> int:
    args = parse_args()
    proc = None
    rc = 0
    try:
        if args.start:
            proc = start_server(args.port)

        rng = np.random.default_rng(0xBEEF)
        H, N, D, M = 8, 64, 32, 4
        K = rng.standard_normal((H, N, D)).astype(np.float32)
        V = rng.standard_normal((H, N, D)).astype(np.float32)
        Q = rng.standard_normal((H, M, D)).astype(np.float32)

        # Two halves: [0..N/2) is "prefix", [N/2..N) is "suffix" — both stored
        # as separate Pion sessions so we can issue two attend_query calls and
        # online-merge.
        half = N // 2
        K_p, V_p = K[:, :half], V[:, :half]
        K_s, V_s = K[:, half:], V[:, half:]

        client = _RESPClient("127.0.0.1", args.port)

        # STORE both halves as separate sessions on the MLX sidecar.
        # ATTEND.PREFIX.STORE <session_id> <layer_id> <H> <N> <D> <K_blob> <V_blob>
        for ns_label, K_part, V_part, Nh in (("p_test_pref", K_p, V_p, half),
                                              ("p_test_suff", K_s, V_s, N - half)):
            r = client.cmd(
                "ATTEND.PREFIX.STORE", ns_label, "0",
                str(H), str(Nh), str(D),
                np.ascontiguousarray(K_part).tobytes(),
                np.ascontiguousarray(V_part).tobytes(),
            )
            if r != b"OK":
                raise RuntimeError(f"ATTEND.PREFIX.STORE {ns_label!r} failed: {r!r}")

        # Query both with the same Q, parse output ‖ LSE.
        Q_bytes = np.ascontiguousarray(Q).tobytes()
        out_floats = H * M * D
        lse_floats = H * M

        def query(sid):
            # top_k must be > 0 even though it's ignored on the M>1 dense path.
            blob = client.cmd("ATTEND.PREFIX.QUERY", sid, "0",
                              str(H), str(D), str(1), Q_bytes)
            if blob is None:
                raise RuntimeError(f"ATTEND.PREFIX.QUERY {sid} returned nil")
            need = (out_floats + lse_floats) * 4
            if len(blob) < need:
                raise RuntimeError(
                    f"short ATTEND.PREFIX.QUERY response: {len(blob)} bytes, expected {need}")
            arr = np.frombuffer(blob[:out_floats * 4], dtype=np.float32).copy()
            lse = np.frombuffer(blob[out_floats * 4:out_floats * 4 + lse_floats * 4],
                                 dtype=np.float32).copy()
            return arr.reshape(H, M, D), lse.reshape(H, M)

        p_out, p_lse = query("p_test_pref")
        s_out, s_lse = query("p_test_suff")
        print(f"  attend_query (prefix): output {p_out.shape} LSE {p_lse.shape}")
        print(f"  attend_query (suffix): output {s_out.shape} LSE {s_lse.shape}")

        # Online merge.
        m = np.maximum(p_lse, s_lse)
        p_w = np.exp(p_lse - m)[..., None]
        s_w = np.exp(s_lse - m)[..., None]
        merged = (p_w * p_out + s_w * s_out) / (p_w + s_w)

        # Reference: full attention over the unsplit K/V (numpy, fp32).
        scale = 1.0 / np.sqrt(D)
        scores = (Q @ np.swapaxes(K, -1, -2)) * scale
        m_ref = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - m_ref)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        ref = weights @ V

        max_abs = float(np.max(np.abs(ref - merged)))
        max_rel = float(np.max(np.abs(ref - merged) / (np.abs(ref) + 1e-9)))
        print(f"  reference vs merged: max|Δ|={max_abs:.2e}  max rel={max_rel:.2e}")
        # Tolerance: MLX runs fp32 too, so sidecar↔reference should agree to fp32 precision.
        if max_abs > 1e-3:
            print(f"FAIL: merged attention diverges from reference by {max_abs:.2e}")
            rc = 1
            return rc

        # Also verify LSE numbers themselves are sane (compare against direct numpy).
        p_scores = (Q @ np.swapaxes(K_p, -1, -2)) * scale
        p_lse_ref = softmax_lse_np(p_scores)
        lse_diff = float(np.max(np.abs(p_lse_ref - p_lse)))
        print(f"  prefix LSE vs numpy reference: max|Δ|={lse_diff:.2e}")
        if lse_diff > 1e-3:
            print(f"FAIL: prefix LSE wrong: max|Δ|={lse_diff:.2e}")
            rc = 1
            return rc

        print("\nPASS: ATTEND.PREFIX.QUERY LSE trailer works end-to-end; online merge is correct.")
        client.close()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        rc = 2
    finally:
        if args.start:
            stop_server(proc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
