#!/usr/bin/env python3
"""C1.2 Replication Hardening Test

Tests PSYNC handshake, REPLCONF ACK, WAIT, READONLY/READWRITE,
and replica recovery (partial sync).

Usage:
    python3 tests/test_replication.py [--port 1974]
"""

import socket
import sys
import time
import redis
import subprocess
import os

PORT = 1974
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}: {detail}")


def test_single_node_commands():
    """Test new commands on a single node."""
    global passed, failed
    r = redis.Redis(port=PORT, decode_responses=False)

    print("=== Section 1: READONLY / READWRITE ===")
    res = r.execute_command("READONLY")
    test("READONLY returns OK", res == b"OK" or res is True, repr(res))

    res = r.execute_command("READWRITE")
    test("READWRITE returns OK", res == b"OK" or res is True, repr(res))

    print("=== Section 2: REPLCONF ACK ===")
    # A replica's ACK gets no reply, in Redis and (since #39) in Pion, so a
    # client that waits for one times out. Send it with a PING behind it on a
    # raw socket: the PONG must be the very next and only reply.
    s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    s.sendall(b"*3\r\n$8\r\nREPLCONF\r\n$3\r\nACK\r\n$1\r\n0\r\n*1\r\n$4\r\nPING\r\n")
    got = b""
    deadline = time.time() + 5
    while not got.endswith(b"\r\n") and time.time() < deadline:
        got += s.recv(4096)
    time.sleep(0.2)                     # anything extra would have arrived by now
    s.setblocking(False)
    try:
        got += s.recv(4096)
    except BlockingIOError:
        pass
    s.close()
    test("REPLCONF ACK answers nothing (PONG is the only reply)", got == b"+PONG\r\n", repr(got))

    print("=== Section 3: WAIT (no replicas) ===")
    res = r.execute_command("WAIT", "1", "100")  # 100ms timeout, 1 replica needed
    test("WAIT returns 0 (no replicas)", res == 0, repr(res))

    print("=== Section 4: INFO REPLICATION ===")
    info = r.execute_command("INFO", "replication")
    # Should return some replication info
    test("INFO REPLICATION returns data", info is not None and len(info) > 0, f"len={len(info) if info else 0}")

    r.flushall()


def test_primary_replica():
    """Test replication between primary and replica."""
    global passed, failed
    print("=== Section 5: Primary-Replica Replication ===")

    PORT_A = PORT
    PORT_B = PORT + 20

    # Start replica
    proc_b = subprocess.Popen(
        [os.environ.get("PION_BIN", "./pion-server"), "-p", str(PORT_B), "-w", "1",
         "--no-auto-detect", "--no-auto-embed",
         "--cluster", "--cluster-host", "127.0.0.1",
         "--cluster-replica",
         "--cluster-primary-host", "127.0.0.1",
         "--cluster-primary-port", str(PORT_A)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(4)

    try:
        ra = redis.Redis(port=PORT_A, decode_responses=False)
        rb = redis.Redis(port=PORT_B, decode_responses=False)

        test("Primary PING", ra.ping())
        test("Replica PING", rb.ping())

        # Write to primary
        for i in range(50):
            ra.set(f"repl:{i}", f"value:{i}")

        # Every key, with its exact value. "found > 0" passed with 1 of 50,
        # and never compared a value.
        deadline = time.time() + 10
        wrong = list(range(50))
        while wrong and time.time() < deadline:
            time.sleep(0.2)
            wrong = [i for i in range(50) if rb.get(f"repl:{i}") != f"value:{i}".encode()]
        test(f"Replica received every key exactly ({50 - len(wrong)}/50)", not wrong,
             f"missing/wrong: {wrong[:5]}")

        # MSET is logged as ONE cmd-31 record ([varint kl][key][varint vl][val]
        # per pair), and the primary streams its raw WAL. So a replica that
        # cannot decode cmd 31 silently misses every MSET. Hash-tagged keys keep
        # the command in one cluster slot, and every value must arrive exactly.
        pairs = {f"{{m}}:{i}".encode(): (b"mv%d-" % i) * (i + 1) for i in range(20)}
        flat = [x for kv in pairs.items() for x in kv]
        test("Primary MSET OK", ra.execute_command("MSET", *flat) in (True, b"OK"))
        deadline = time.time() + 10
        missing = list(pairs)
        while missing and time.time() < deadline:
            time.sleep(0.2)
            missing = [k for k, v in pairs.items() if rb.get(k) != v]
        test(f"Replica received every MSET pair exactly ({20 - len(missing)}/20)",
             not missing, f"missing/wrong: {missing[:3]}")

        # READONLY on replica should work
        res = rb.execute_command("READONLY")
        test("Replica READONLY OK", res == b"OK" or res is True)

        # Read from replica after READONLY: the value, not "did not crash".
        val = rb.get("repl:0")
        test("Replica read after READONLY returns the value", val == b"value:0", repr(val))

        # READWRITE restores default
        res = rb.execute_command("READWRITE")
        test("Replica READWRITE OK", res == b"OK" or res is True)

        # WAIT on primary (replica should have ACKed by now)
        # WAIT must honour its timeout (it blocks the worker while it waits)
        # and count the replica once it has ACKed. Measured: WAIT 1 1000 took
        # ~4.6 s (each 1 ms poll step costs ~4.7 ms) and usually answered 0,
        # because replica ACKs arrive ~5 s after a write (gh #390).
        t_wait = time.time()
        res = ra.execute_command("WAIT", "1", "1000")
        waited = time.time() - t_wait
        test("WAIT 1 1000 returns within 1.5 s", waited < 1.5, f"took {waited:.2f} s")
        test("WAIT counts the one caught-up replica", res == 1, repr(res))

        ra.flushall()

    except Exception as e:
        test("Primary-Replica test", False, str(e))

    finally:
        proc_b.terminate()
        try:
            proc_b.wait(timeout=5)
        except:
            proc_b.kill()


def test_replica_recovery():
    """Test replica reconnection after restart (PSYNC partial sync)."""
    global passed, failed
    print("=== Section 6: Replica Recovery (PSYNC) ===")

    PORT_A = PORT
    PORT_B = PORT + 30

    # Start replica
    proc_b = subprocess.Popen(
        [os.environ.get("PION_BIN", "./pion-server"), "-p", str(PORT_B), "-w", "1",
         "--no-auto-detect", "--no-auto-embed",
         "--cluster", "--cluster-host", "127.0.0.1",
         "--cluster-replica",
         "--cluster-primary-host", "127.0.0.1",
         "--cluster-primary-port", str(PORT_A)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(4)

    try:
        ra = redis.Redis(port=PORT_A, decode_responses=False)
        rb = redis.Redis(port=PORT_B, decode_responses=False)

        # Write initial data
        for i in range(20):
            ra.set(f"psync:{i}", f"initial:{i}")
        time.sleep(2)

        # Kill replica
        proc_b.terminate()
        proc_b.wait(timeout=5)
        time.sleep(1)

        # Write more data while replica is down
        for i in range(20, 40):
            ra.set(f"psync:{i}", f"new:{i}")

        # Restart replica
        proc_b = subprocess.Popen(
            [os.environ.get("PION_BIN", "./pion-server"), "-p", str(PORT_B), "-w", "1",
             "--no-auto-detect", "--no-auto-embed",
             "--cluster", "--cluster-host", "127.0.0.1",
             "--cluster-replica",
             "--cluster-primary-host", "127.0.0.1",
             "--cluster-primary-port", str(PORT_A)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(5)

        rb = redis.Redis(port=PORT_B, decode_responses=False)
        test("Replica restarted", rb.ping())

        # Check if replica caught up (at least some new keys should be there)
        found_new = 0
        for i in range(20, 40):
            val = rb.get(f"psync:{i}")
            if val is not None:
                found_new += 1

        test(f"Replica caught up after restart ({found_new}/20 new keys)", found_new > 0,
             f"found={found_new}")

        ra.flushall()

    except Exception as e:
        test("Replica recovery test", False, str(e))

    finally:
        proc_b.terminate()
        try:
            proc_b.wait(timeout=5)
        except:
            proc_b.kill()


if __name__ == "__main__":
    print("=" * 60)
    print("C1.2 Replication Hardening Tests")
    print(f"Primary: 127.0.0.1:{PORT}")
    print("=" * 60)
    print()

    test_single_node_commands()
    test_primary_replica()
    test_replica_recovery()

    print()
    print("=" * 60)
    print(f"C1.2 Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
