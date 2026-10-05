#!/usr/bin/env python3
"""WAL Recovery Test -- validates data durability across server restart.

Requires: pion-server binary at ./pion-server (built via pixi run build)
Usage: python3 tests/test_wal_recovery.py
       python3 tests/test_wal_recovery.py --binary /path/to/pion-server
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time


# ─── Configuration ────────────────────────────────────────────────────────────

PORT = 1976
WORKERS = 1
WORKDIR = f"/tmp/pion_wal_test_{PORT}"
NUM_STRING_KEYS = 100
NUM_HASH_KEYS = 10
NUM_LIST_KEYS = 5


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader, wait_ready_pid  # noqa: E402

# ─── RESP helpers (self-contained, no external deps) ─────────────────────────

def encode_cmd(args):
    """Encode a list of strings as a RESP array command."""
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def recv_resp(sock, timeout=10.0):
    """Receive exactly ONE complete RESP reply. The old single recv() with a
    swallowed timeout returned "" for a slow first reply after restart (the
    server replays its WAL first) and let that reply answer the NEXT check."""
    return reader(sock, timeout).read_raw().decode(errors="replace")


def send_recv(sock, *args):
    """Send a RESP command and return the response string."""
    sock.sendall(encode_cmd(args))
    return recv_resp(sock)


def connect(port, timeout=5.0):
    """Open a TCP connection to the server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", port))
    s.settimeout(None)
    return s


# ─── Server lifecycle ────────────────────────────────────────────────────────

def wait_for_port(port, proc, timeout=15):
    """Poll until the server ANSWERS, not merely accepts.

    It listens before it initialises (clients queue rather than being
    refused), and after a restart the first reply waits for the hash map and
    the WAL replay. Returning on accept let the first GET time out at 2 s and
    read as '' — the one "lost" key (wal:0) this test reported was that.
    The answer must come from `proc` itself, not whatever else holds the
    port (#27)."""
    try:
        wait_ready_pid(port, proc, timeout)
        return True
    except RuntimeError:
        return False


def wait_for_port_free(port, timeout=10):
    """Poll until the port stops accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
            s.close()
            time.sleep(0.2)
        except (ConnectionRefusedError, OSError):
            s.close()
            return True
    return False


def start_server(binary, port, workers, workdir):
    """Start pion-server and wait for it to be ready. Returns Popen handle."""
    cmd = [
        os.path.abspath(binary),
        "-p", str(port),
        "-w", str(workers),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def kill_server(proc, use_sigkill=False):
    """Kill the server process."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if use_sigkill:
            proc.kill()  # SIGKILL
        else:
            proc.terminate()  # SIGTERM
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def clean_wal_files(workdir):
    """Remove persistence files from the working directory. gh #170: snapshots
    now carry hashes/lists too, so a stale pion.snapshot.* leaks the previous
    run's aggregates into this one."""
    if not os.path.isdir(workdir):
        return
    for f in os.listdir(workdir):
        if f.startswith("pion.wal.") or f.startswith("pion.snapshot.") or f.startswith("pion.blob."):
            try:
                os.remove(os.path.join(workdir, f))
            except OSError:
                pass


# ─── Test logic ───────────────────────────────────────────────────────────────

def phase_write(port):
    """Phase 1: Write test data to the server."""
    sock = connect(port)
    results = {"ok": 0, "fail": 0}

    # 100 string keys
    for i in range(NUM_STRING_KEYS):
        resp = send_recv(sock, "SET", f"wal:{i}", f"value:{i}")
        if "+OK" in resp:
            results["ok"] += 1
        else:
            results["fail"] += 1
            print(f"  FAIL SET wal:{i} -> {resp.strip()}")

    # 10 hash keys
    for i in range(NUM_HASH_KEYS):
        resp = send_recv(sock, "HSET", f"wal:h:{i}", "f1", "v1", "f2", "v2")
        if ":2" in resp or ":0" in resp:  # :2 for new, :0 for overwrite
            results["ok"] += 1
        else:
            results["fail"] += 1
            print(f"  FAIL HSET wal:h:{i} -> {resp.strip()}")

    # 5 list keys
    for i in range(NUM_LIST_KEYS):
        resp = send_recv(sock, "LPUSH", f"wal:l:{i}", "a", "b", "c")
        if ":3" in resp:
            results["ok"] += 1
        else:
            results["fail"] += 1
            print(f"  FAIL LPUSH wal:l:{i} -> {resp.strip()}")

    # Trigger SAVE to sync WAL
    resp = send_recv(sock, "SAVE")
    if "+OK" in resp:
        results["ok"] += 1
    else:
        print(f"  WARN SAVE response: {resp.strip()}")

    sock.close()
    return results


def phase_verify(port):
    """Phase 2: Verify all data survived the restart."""
    sock = connect(port)
    passed = 0
    failed = 0
    errors = []

    # Verify 100 string keys
    for i in range(NUM_STRING_KEYS):
        resp = send_recv(sock, "GET", f"wal:{i}")
        expected = f"value:{i}"
        if expected in resp:
            passed += 1
        else:
            failed += 1
            errors.append(f"GET wal:{i}: expected '{expected}', got '{resp.strip()}'")

    # Verify 10 hash keys
    for i in range(NUM_HASH_KEYS):
        resp = send_recv(sock, "HGET", f"wal:h:{i}", "f1")
        if "v1" in resp:
            passed += 1
        else:
            failed += 1
            errors.append(f"HGET wal:h:{i} f1: expected 'v1', got '{resp.strip()}'")

        resp = send_recv(sock, "HGET", f"wal:h:{i}", "f2")
        if "v2" in resp:
            passed += 1
        else:
            failed += 1
            errors.append(f"HGET wal:h:{i} f2: expected 'v2', got '{resp.strip()}'")

    # Verify 5 list keys
    for i in range(NUM_LIST_KEYS):
        resp = send_recv(sock, "LLEN", f"wal:l:{i}")
        if ":3" in resp:
            passed += 1
        else:
            failed += 1
            errors.append(f"LLEN wal:l:{i}: expected :3, got '{resp.strip()}'")

    sock.close()
    return passed, failed, errors


def main():
    parser = argparse.ArgumentParser(description="WAL Recovery Test")
    parser.add_argument(
        "--binary", default=os.environ.get("PION_BIN", "./pion-server"),
        help="Path to pion-server binary (default: ./pion-server)"
    )
    args = parser.parse_args()

    binary = args.binary
    if not os.path.isfile(binary):
        print(f"FATAL: Server binary not found: {binary}")
        print("Build it first: pixi run build")
        sys.exit(1)

    # Prepare working directory
    os.makedirs(WORKDIR, exist_ok=True)
    clean_wal_files(WORKDIR)

    proc = None
    try:
        # ── Phase 1: Start server, write data ────────────────────────────
        print(f"[1/6] Starting pion-server on port {PORT} (w={WORKERS})...")
        proc = start_server(binary, PORT, WORKERS, WORKDIR)

        if not wait_for_port(PORT, proc, timeout=15):
            print("FATAL: Server did not start within 15s")
            kill_server(proc)
            sys.exit(1)
        print("      Server ready.")

        print(f"[2/6] Writing test data ({NUM_STRING_KEYS} strings, "
              f"{NUM_HASH_KEYS} hashes, {NUM_LIST_KEYS} lists)...")
        write_results = phase_write(PORT)
        print(f"      Writes: {write_results['ok']} ok, {write_results['fail']} fail")

        if write_results["fail"] > 0:
            print("FATAL: Write phase had failures, cannot test recovery.")
            kill_server(proc)
            sys.exit(1)

        # ── Phase 2: SAVE + crash ────────────────────────────────────────
        print("[3/6] SAVE sent. Waiting 1s for WAL sync...")
        time.sleep(1)

        # Verify WAL files exist
        wal_files = [f for f in os.listdir(WORKDIR) if f.startswith("pion.wal.")]
        if wal_files:
            total_size = sum(
                os.path.getsize(os.path.join(WORKDIR, f)) for f in wal_files
            )
            print(f"      WAL files: {wal_files} (total {total_size:,} bytes)")
        else:
            print("      WARN: No WAL files found in working directory")

        print("[4/6] Killing server with SIGKILL (simulating crash)...")
        kill_server(proc, use_sigkill=True)
        proc = None

        if not wait_for_port_free(PORT, timeout=10):
            print("FATAL: Port did not become free after killing server")
            sys.exit(1)
        print("      Server stopped.")

        # ── Phase 3: Restart and verify ──────────────────────────────────
        print(f"[5/6] Restarting pion-server on port {PORT}...")
        proc = start_server(binary, PORT, WORKERS, WORKDIR)

        if not wait_for_port(PORT, proc, timeout=20):
            print("FATAL: Server did not restart within 20s")
            kill_server(proc)
            sys.exit(1)
        print("      Server ready (restarted).")

        print("[6/6] Verifying data survived restart...")
        passed, failed, errors = phase_verify(PORT)

        # ── Results ──────────────────────────────────────────────────────
        total = passed + failed
        print()
        print("=" * 60)
        if failed == 0:
            print(f"WAL RECOVERY TEST: ALL PASSED ({passed}/{total} checks)")
            print("=" * 60)
        else:
            print(f"WAL RECOVERY TEST: FAILED ({failed}/{total} checks failed)")
            print("=" * 60)
            for err in errors[:20]:  # Print at most 20 errors
                print(f"  FAIL: {err}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    finally:
        # Always clean up the server
        if proc is not None:
            kill_server(proc)
        # Do NOT clean WAL files here — leave them for debugging on failure


if __name__ == "__main__":
    main()
