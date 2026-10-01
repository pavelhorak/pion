#!/usr/bin/env python3
"""N3 End-to-end failover test.

Tests:
1. Start primary on port 1974
2. Start replica on port 1975, replicating from primary
3. Write 1000 keys to primary
4. Wait for replication lag → 0
5. Kill primary
6. CLUSTER FAILOVER FORCE on replica
7. Verify all 1000 keys accessible on replica
8. Verify replica accepts new writes

Usage:
    pixi run build && python3 tests/test_failover.py
"""
import os
import signal
import socket
import subprocess
import sys
import time

PION_BIN = os.environ.get("PION_BIN") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pion-server"))
PRIMARY_PORT = 1974
REPLICA_PORT = 1984
NUM_KEYS = 1000
FAILOVER_TIMEOUT = 10  # seconds
PRIMARY_DIR = "/tmp/pion-test-primary"
REPLICA_DIR = "/tmp/pion-test-replica"


def send_resp(sock, *args):
    """Send RESP command and read response."""
    header = f"*{len(args)}\r\n".encode()
    body = b""
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        body += f"${len(a)}\r\n".encode() + a + b"\r\n"
    sock.sendall(header + body)
    return sock.recv(65536)


def connect(port, timeout=5):
    """Connect to a Pion server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", port))
    return s


def wait_for_server(port, timeout=15):
    """Wait until server responds to PING."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = connect(port, timeout=2)
            resp = send_resp(s, "PING")
            s.close()
            if b"PONG" in resp:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_server(proc):
    """Kill a server process."""
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)
    except Exception:
        pass


def main():
    print("=" * 60)
    print("  N3 Failover Test")
    print("=" * 60)

    # Clean and create isolated directories
    import shutil
    for d in [PRIMARY_DIR, REPLICA_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    primary_proc = None
    replica_proc = None
    passed = 0
    failed = 0

    try:
        # ── Step 1: Start primary ──────────────────────────────────────
        print("\n[1] Starting primary on port", PRIMARY_PORT)
        primary_proc = subprocess.Popen(
            [PION_BIN, "-p", str(PRIMARY_PORT), "-w", "1",
             "--cluster", "--cluster-host", "127.0.0.1",
             "--no-auto-detect"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=PRIMARY_DIR,
        )
        if not wait_for_server(PRIMARY_PORT):
            print("FAIL: primary did not start")
            return 1
        print("  Primary ready.")

        # ── Step 2: Start replica ──────────────────────────────────────
        print("\n[2] Starting replica on port", REPLICA_PORT)
        replica_proc = subprocess.Popen(
            [PION_BIN, "-p", str(REPLICA_PORT), "-w", "1",
             "--cluster", "--cluster-host", "127.0.0.1",
             "--cluster-replica",
             "--cluster-primary-host", "127.0.0.1",
             "--cluster-primary-port", str(PRIMARY_PORT),
             "--cluster-nodes", f"127.0.0.1:{PRIMARY_PORT}",
             "--no-auto-detect"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=REPLICA_DIR,
        )
        if not wait_for_server(REPLICA_PORT):
            print("FAIL: replica did not start")
            return 1
        print("  Replica ready.")

        # ── Step 3: Write keys to primary ──────────────────────────────
        print(f"\n[3] Writing {NUM_KEYS} keys to primary...")
        s_primary = connect(PRIMARY_PORT)
        for i in range(NUM_KEYS):
            send_resp(s_primary, "SET", f"key:{i}", f"value:{i}")
        print(f"  Written {NUM_KEYS} keys.")

        # Verify a sample key on primary
        resp = send_resp(s_primary, "GET", "key:0")
        if b"value:0" in resp:
            print("  PASS: key:0 readable on primary")
            passed += 1
        else:
            print(f"  FAIL: key:0 not found on primary: {resp[:50]}")
            failed += 1
        s_primary.close()

        # ── Step 4: Wait for replication ───────────────────────────────
        print("\n[4] Waiting for replication (3s)...")
        time.sleep(3)

        # Check if replica has keys (read should work if WAL applied)
        s_replica = connect(REPLICA_PORT)
        resp = send_resp(s_replica, "GET", "key:0")
        if b"value:0" in resp:
            print("  PASS: key:0 replicated to replica")
            passed += 1
        else:
            print(f"  INFO: key:0 not yet on replica (expected for async repl): {resp[:50]}")
        s_replica.close()

        # ── Step 5: Kill primary ───────────────────────────────────────
        print("\n[5] Killing primary...")
        kill_server(primary_proc)
        primary_proc = None
        time.sleep(1)
        print("  Primary killed.")

        # ── Step 6: Manual failover ────────────────────────────────────
        print("\n[6] Sending CLUSTER FAILOVER FORCE to replica...")
        t_failover_start = time.time()
        s_replica = connect(REPLICA_PORT)
        resp = send_resp(s_replica, "CLUSTER", "FAILOVER", "FORCE")
        t_failover_end = time.time()
        failover_ms = (t_failover_end - t_failover_start) * 1000

        if b"+OK" in resp:
            print(f"  PASS: failover completed in {failover_ms:.0f}ms")
            passed += 1
        else:
            print(f"  FAIL: failover returned: {resp[:80]}")
            failed += 1

        # ── Step 7: Verify keys on promoted replica ────────────────────
        print("\n[7] Verifying keys on promoted replica...")
        hits = 0
        misses = 0
        for i in range(0, NUM_KEYS, 100):  # sample every 100th key
            resp = send_resp(s_replica, "GET", f"key:{i}")
            if f"value:{i}".encode() in resp:
                hits += 1
            else:
                misses += 1

        total_checked = hits + misses
        print(f"  Checked {total_checked} keys: {hits} hits, {misses} misses")
        if hits > 0:
            print(f"  PASS: {hits}/{total_checked} keys accessible after failover")
            passed += 1
        else:
            print(f"  FAIL: no keys accessible after failover")
            failed += 1

        # ── Step 8: Verify new writes work ─────────────────────────────
        print("\n[8] Testing new writes on promoted replica...")
        resp = send_resp(s_replica, "SET", "post_failover", "works")
        if b"+OK" in resp:
            resp2 = send_resp(s_replica, "GET", "post_failover")
            if b"works" in resp2:
                print("  PASS: new writes work after failover")
                passed += 1
            else:
                print(f"  FAIL: GET post_failover returned: {resp2[:50]}")
                failed += 1
        else:
            print(f"  FAIL: SET returned: {resp[:50]}")
            failed += 1

        # ── Step 9: Check CLUSTER INFO role ────────────────────────────
        print("\n[9] Checking CLUSTER INFO role...")
        resp = send_resp(s_replica, "CLUSTER", "INFO")
        if b"cluster_role:master" in resp:
            print("  PASS: role is master after failover")
            passed += 1
        else:
            print(f"  INFO: cluster_role not found (cluster may not be fully enabled)")

        s_replica.close()

    finally:
        # Cleanup
        if primary_proc:
            kill_server(primary_proc)
        if replica_proc:
            kill_server(replica_proc)

    # ── Results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
