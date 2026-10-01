#!/usr/bin/env python3
"""Gate 2c — AI substrate wire tests (gh #109).

Runs the pure-wire subset of the attend/kv_prefix/ssm/moe/wal test files —
no MLX, no model downloads, no GPU — so the flagship substrate surface has an
automated guard next to the KV/parity gates.

Two phases:
  1. WIRE  — tests that talk to a shared pion-server this runner starts
             (--kvcache + --moe-cache, single worker).
  2. SELF  — durability tests that manage their own server lifecycle
             (SIGKILL + restart); run after the shared server is stopped.

Tests importing MLX or loading models are excluded BY DESIGN and listed in
the output (no silent caps) — they stay manual / Linux-session material.

Usage:
    python3 tests/run_substrate_gate.py            # full run (starts server)
    python3 tests/run_substrate_gate.py --skip-self
    python3 tests/run_substrate_gate.py --only test_attend_prefix.py
"""

import argparse
import functools
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

print = functools.partial(print, flush=True)

HOST = "127.0.0.1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))

# Tests that assume a running --kvcache server and speak pure RESP/binary wire.
WIRE_TESTS = [
    # ATTEND.* — externalized attention index
    "test_attend_prefix.py",
    "test_attend_prefix_batched.py",
    "test_attend_prefix_lookup.py",
    "test_attend_prefix_lse.py",
    "test_attend_prefix_merge.py",
    "test_attend_prefix_realistic.py",
    "test_attend_quality.py",   # gh #391: stored keys must retrieve themselves
    "test_attend_128k.py",
    "test_attend_scale.py",
    "test_attend_sparse_auto.py",
    "test_attend_sparse_auto_fused.py",
    "test_attend_sparse_kernel.py",
    "test_attend_per_call_window.py",
    "test_attend_d512.py",
    "test_attend_asymmetric.py",
    "test_attend_fused_parity.py",
    "test_attend_fused_binary_parity.py",
    # KV.PREFIX.* — shared KV cache (wire-only subset)
    "test_kv_prefix_lru.py",
    "test_kv_prefix_blocks.py",
    "test_kv_prefix_cold_tier.py",
    # SSM.PREFIX.*
    "test_ssm_prefix_roundtrip.py",
    "test_ssm_prefix_large_blob.py",
    # MOE.EXPERT.*
    "test_moe_expert_wire.py",
    "test_moe_expert_load_dedup.py",
    "test_moe_hist_namespaces.py",
    "test_moe_ns_prune_selection.py",
    "test_moe_wire_tensor_parity.py",
]

# Tests that spawn/kill their own pion-server (durability / multi-worker).
SELF_TESTS = [
    "test_vstore_wal.py",
    "test_ssm_prefix_durability.py",
    "test_kv_prefix_persistence.py",
    "test_kvprefix_autoredirect.py",
    "test_kvprefix_xworker.py",
]

# Excluded by design — need MLX / model weights / manual setup. Printed, never
# silently dropped (vision rule: no silent caps).
EXCLUDED = {
    "test_kv_prefix_prototype.py": "loads HF gpt2; G1 thresholds await recalibration (gh #171)",
    "test_kv_prefix_mlx.py": "loads MLX Llama; G1 thresholds await recalibration (gh #171)",
    "test_kv_prefix_admission.py": "imports mlx (model K/V)",
    "test_kv_prefix_bleu.py": "imports mlx + loads Llama (BLEU cross-instance)",
    "test_kv_prefix_cross_instance.py": "imports mlx + loads model",
    "test_kv_prefix_workload.py": "imports mlx + 100-request model workload",
    "test_attend_fused_client.py": "imports mlx (fused client parity)",
}

ENV_SKIP_MARKERS = (
    "ModuleNotFoundError",
    "No module named",
    "mlx not available",
    "SKIP",
    "requires --inference",
    "requires Metal",
    "attention engine not available",
)


def port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        s = socket.create_connection((HOST, port), timeout=timeout)
        s.close()
        return True
    except OSError:
        return False


def ping(port: int) -> bool:
    try:
        s = socket.create_connection((HOST, port), timeout=2)
        s.sendall(b"*1\r\n$4\r\nPING\r\n")
        data = s.recv(64)
        s.close()
        return data.startswith(b"+PONG")
    except OSError:
        return False


def start_server(binary: str, port: int, moe_dir: str, no_engine: bool = False) -> subprocess.Popen:
    log = open(os.path.join(REPO, "pion_gate2c.log"), "w")
    proc = subprocess.Popen(
        [
            binary,
            "--kvcache",
            "-w", "1",
            "-p", str(port),
            "--no-auto-embed",
            "--moe-cache", moe_dir,
            "--moe-cache-mib", "256",
        ] + ([] if no_engine else
             (["--metal-attention"] if sys.platform == "darwin" else ["--cuda-attention"])),
        cwd=REPO,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 45
    while time.time() < deadline:
        if ping(port):
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(
        f"pion-server did not become ready on port {port} — see pion_gate2c.log"
    )


def run_test(name: str, port: int, timeout: int) -> tuple[str, float, str]:
    """Returns (status, seconds, detail); status in PASS/FAIL/ENV_SKIP/TIMEOUT."""
    env = dict(os.environ, PION_PORT=str(port))
    t0 = time.time()
    # start_new_session: on timeout the WHOLE process group dies — a bare
    # child-kill orphans the test's own spawned servers/subprocesses, and the
    # orphans deadlock every later run (observed 2026-08-01).
    proc = subprocess.Popen(
        [sys.executable, os.path.join(TESTS, name)],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out_text, _ = proc.communicate(timeout=timeout)
        r = subprocess.CompletedProcess(proc.args, proc.returncode, out_text, "")
    except subprocess.TimeoutExpired:
        import signal as _sig
        try:
            os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        return "TIMEOUT", time.time() - t0, f"exceeded {timeout}s (process group killed)"
    dt = time.time() - t0
    if r.returncode == 0:
        return "PASS", dt, ""
    out = r.stdout or ""
    for marker in ENV_SKIP_MARKERS:
        if marker in out:
            return "ENV_SKIP", dt, marker
    tail = out.strip().splitlines()[-3:]
    return "FAIL", dt, " | ".join(tail)


def clean_state() -> None:
    """Remove per-worker state files so one phase's leftovers can't poison the
    next (the crashed-server vstore files broke autoredirect on the first run)."""
    # Orphaned embedding sidecars hold inherited stdout pipes and stall
    # any test that drains a dead server's output.
    subprocess.run(["pkill", "-9", "-f", "inference/worker"],
                   capture_output=True)
    for w in range(8):
        for stem in ("pion.vstore.", "pion.vstore.wal.", "pion.wal.",
                     "pion.snapshot.", "pion.ssm.", "pion.hnsw."):
            f = os.path.join(REPO, f"{stem}{w}")
            if os.path.exists(f):
                os.remove(f)
        for seg in range(64):
            f = os.path.join(REPO, f"pion.blob.{w}.{seg}")
            if os.path.exists(f):
                os.remove(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", "1974")))
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("--timeout", type=int, default=90, help="per-test timeout, wire phase")
    ap.add_argument("--self-timeout", type=int, default=240, help="per-test timeout, self-managed phase")
    ap.add_argument("--skip-self", action="store_true")
    ap.add_argument("--no-attention-engine", action="store_true",
                    help="GPU-less host: omit --metal/--cuda-attention; "
                         "engine-dependent ATTEND tests classify as ENV_SKIP")
    ap.add_argument("--only", help="run a single named test from either list")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"FATAL: {args.binary} not found — run `pixi run build` first")
        return 2

    wire = [t for t in WIRE_TESTS if os.path.exists(os.path.join(TESTS, t))]
    selfm = [t for t in SELF_TESTS if os.path.exists(os.path.join(TESTS, t))]
    if args.only:
        wire = [t for t in wire if t == args.only]
        selfm = [t for t in selfm if t == args.only]

    results: list[tuple[str, str, float, str]] = []

    # Phase 1 — shared-server wire tests
    if wire:
        if port_open(args.port):
            print(f"FATAL: port {args.port} already in use — stop the other server first")
            return 2
        clean_state()
        moe_dir = tempfile.mkdtemp(prefix="pion_gate2c_moe_")
        print(f"[gate2c] starting {args.binary} --kvcache -w 1 -p {args.port} (moe-cache {moe_dir})")
        proc = start_server(args.binary, args.port, moe_dir, args.no_attention_engine)
        try:
            for t in wire:
                status, dt, detail = run_test(t, args.port, args.timeout)
                if status != "PASS" and not ping(args.port):
                    status, detail = "CRASHED_SERVER", "server died during this test"
                results.append((t, status, dt, detail))
                print(f"  {status:14s} {dt:6.1f}s  {t}" + (f"  — {detail}" if detail else ""))
                if status == "CRASHED_SERVER":
                    break
        finally:
            proc.kill()
            proc.wait()
            shutil.rmtree(moe_dir, ignore_errors=True)

    # Phase 2 — self-managed durability tests (need the port free)
    if selfm and not args.skip_self:
        time.sleep(1)
        for t in selfm:
            clean_state()
            status, dt, detail = run_test(t, args.port, args.self_timeout)
            results.append((t, status, dt, detail))
            print(f"  {status:14s} {dt:6.1f}s  {t}" + (f"  — {detail}" if detail else ""))

    # Summary
    n = {"PASS": 0, "FAIL": 0, "ENV_SKIP": 0, "TIMEOUT": 0, "CRASHED_SERVER": 0}
    for _, status, _, _ in results:
        n[status] += 1
    print()
    print(f"[gate2c] excluded by design ({len(EXCLUDED)}):")
    for name, why in EXCLUDED.items():
        print(f"    {name} — {why}")
    total = sum(n.values())
    bad = n["FAIL"] + n["TIMEOUT"] + n["CRASHED_SERVER"]
    print(
        f"[gate2c] {total} run: {n['PASS']} pass, {n['FAIL']} fail, "
        f"{n['TIMEOUT']} timeout, {n['CRASHED_SERVER']} crash, {n['ENV_SKIP']} env-skip"
    )
    # Name the failures in the SUMMARY, not only in the per-test line far above.
    # The per-test lines were always there; the problem is that everyone reads
    # the tail, so an intermittent failure gets recorded as "31 pass, 1 fail"
    # with the name scrolled off — which is what happened twice on 2026-08-20
    # and left a flake unattributable across eight runs.
    if bad:
        print("[gate2c] FAILING:")
        for name, status, dt, detail in results:
            if status in ("FAIL", "TIMEOUT", "CRASHED_SERVER"):
                print(f"    {status:14s} {dt:6.1f}s  {name}"
                      + (f"  — {detail}" if detail else ""))
    print(f"GATE 2c — AI Substrate: {'PASSED' if bad == 0 else 'FAILED'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
