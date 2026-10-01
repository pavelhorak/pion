#!/usr/bin/env python3
"""C1.1 Cluster Slot Migration Test

Tests ASKING, ADDSLOTS/DELSLOTS, GETKEYSINSLOT, COUNTKEYSINSLOT,
DUMP, RESTORE, and MIGRATE commands.

Usage:
    python3 tests/test_cluster_migration.py [--port 1974]
"""

import sys
import time
import redis
import struct
import subprocess
import signal
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


def test_single_node():
    """Test cluster commands on a single node."""
    global passed, failed
    r = redis.Redis(port=PORT, decode_responses=False)
    r.flushall()

    print("=== Section 1: ASKING command ===")
    res = r.execute_command("ASKING")
    test("ASKING returns OK", res == b"OK" or res is True, repr(res))

    print("=== Section 2: CLUSTER ADDSLOTS / DELSLOTS ===")
    # In non-cluster mode, these still work on the ClusterState struct
    res = r.execute_command("CLUSTER", "ADDSLOTS", "100", "200", "300")
    test("ADDSLOTS returns OK", res == b"OK" or res is True, repr(res))

    res = r.execute_command("CLUSTER", "DELSLOTS", "200")
    test("DELSLOTS returns OK", res == b"OK" or res is True, repr(res))

    print("=== Section 3: CLUSTER KEYSLOT ===")
    slot = r.execute_command("CLUSTER", "KEYSLOT", "mykey")
    test("KEYSLOT returns int", isinstance(slot, int), repr(slot))
    test("KEYSLOT consistent", slot == r.execute_command("CLUSTER", "KEYSLOT", "mykey"))

    # Hash tag support
    slot1 = r.execute_command("CLUSTER", "KEYSLOT", "{user}.name")
    slot2 = r.execute_command("CLUSTER", "KEYSLOT", "{user}.email")
    test("Hash tags: same slot", slot1 == slot2, f"{slot1} vs {slot2}")

    print("=== Section 4: CLUSTER GETKEYSINSLOT / COUNTKEYSINSLOT ===")
    r.flushall()
    # Insert keys and find their slots
    r.set("key1", "val1")
    r.set("key2", "val2")
    r.set("key3", "val3")

    # Find slot for key1
    slot1 = r.execute_command("CLUSTER", "KEYSLOT", "key1")

    count = r.execute_command("CLUSTER", "COUNTKEYSINSLOT", str(slot1))
    test("COUNTKEYSINSLOT >= 1", count >= 1, f"slot={slot1}, count={count}")

    keys = r.execute_command("CLUSTER", "GETKEYSINSLOT", str(slot1), "10")
    test("GETKEYSINSLOT returns list", isinstance(keys, list), repr(type(keys)))
    test("GETKEYSINSLOT has key1", b"key1" in keys, repr(keys))

    print("=== Section 5: DUMP / RESTORE ===")
    r.flushall()
    r.set("dump_test", "hello world")
    dumped = r.execute_command("DUMP", "dump_test")
    test("DUMP returns bytes", isinstance(dumped, bytes) and len(dumped) > 0, f"len={len(dumped) if dumped else 0}")

    # DUMP of non-existent key returns null
    null_dump = r.execute_command("DUMP", "nonexistent")
    test("DUMP nonexistent returns None", null_dump is None, repr(null_dump))

    # RESTORE into new key
    res = r.execute_command("RESTORE", "restored_key", "0", dumped, "REPLACE")
    test("RESTORE returns OK", res == b"OK" or res is True, repr(res))

    # Verify restored value
    val = r.get("restored_key")
    test("RESTORE value matches", val == b"hello world", repr(val))

    # DUMP/RESTORE for different types
    r.lpush("list_key", "c", "b", "a")
    list_dump = r.execute_command("DUMP", "list_key")
    test("DUMP list returns bytes", isinstance(list_dump, bytes) and len(list_dump) > 0)

    r.sadd("set_key", "x", "y", "z")
    set_dump = r.execute_command("DUMP", "set_key")
    test("DUMP set returns bytes", isinstance(set_dump, bytes) and len(set_dump) > 0)

    r.hset("hash_key", mapping={"f1": "v1", "f2": "v2"})
    hash_dump = r.execute_command("DUMP", "hash_key")
    test("DUMP hash returns bytes", isinstance(hash_dump, bytes) and len(hash_dump) > 0)

    # Restore set into a new key
    res = r.execute_command("RESTORE", "set_restored", "0", set_dump, "REPLACE")
    test("RESTORE set OK", res == b"OK" or res is True)

    # Restore hash into a new key
    res = r.execute_command("RESTORE", "hash_restored", "0", hash_dump, "REPLACE")
    test("RESTORE hash OK", res == b"OK" or res is True)

    print("=== Section 6: MIGRATE (skipped — self-connect deadlocks with w=1) ===")

    r.flushall()
    print()


def test_two_node_migration():
    """Test slot migration between two Pion nodes on different ports."""
    global passed, failed
    print("=== Section 7: Two-Node Slot Migration ===")

    PORT_A = PORT
    PORT_B = PORT + 10

    # Start second Pion node
    proc_b = subprocess.Popen(
        [os.environ.get("PION_BIN", "./pion-server"), "-p", str(PORT_B), "-w", "1", "--no-auto-detect", "--no-auto-embed",
         "--cluster", "--cluster-host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(3)

    try:
        ra = redis.Redis(port=PORT_A, decode_responses=False)
        rb = redis.Redis(port=PORT_B, decode_responses=False)

        # Verify both nodes are up
        test("Node A PING", ra.ping())
        test("Node B PING", rb.ping())

        # Insert keys on node A
        for i in range(20):
            ra.set(f"mig:{i}", f"value:{i}")

        # Pick a slot to migrate
        target_slot = ra.execute_command("CLUSTER", "KEYSLOT", "mig:0")

        # Get keys in this slot
        keys_in_slot = ra.execute_command("CLUSTER", "GETKEYSINSLOT", str(target_slot), "100")
        count_in_slot = ra.execute_command("CLUSTER", "COUNTKEYSINSLOT", str(target_slot))
        test(f"Keys in slot {target_slot}", len(keys_in_slot) > 0, f"count={count_in_slot}")

        # Migrate each key to node B
        migrated = 0
        for key in keys_in_slot:
            try:
                res = ra.execute_command("MIGRATE", "127.0.0.1", str(PORT_B),
                                         key.decode(), "0", "5000", "REPLACE")
                if res == b"OK" or res is True:
                    migrated += 1
            except Exception as e:
                print(f"  MIGRATE {key}: {e}")

        test(f"Migrated {migrated}/{len(keys_in_slot)} keys", migrated == len(keys_in_slot),
             f"migrated={migrated}")

        # Verify keys exist on node B
        verified = 0
        for key in keys_in_slot:
            val = rb.get(key)
            if val is not None:
                verified += 1
        test(f"Verified {verified}/{len(keys_in_slot)} on node B", verified == len(keys_in_slot))

        # Verify keys deleted from node A
        deleted = 0
        for key in keys_in_slot:
            val = ra.get(key)
            if val is None:
                deleted += 1
        test(f"Deleted {deleted}/{len(keys_in_slot)} from node A", deleted == len(keys_in_slot))

        ra.flushall()
        rb.flushall()

    finally:
        proc_b.terminate()
        proc_b.wait(timeout=5)


if __name__ == "__main__":
    print("=" * 60)
    print("C1.1 Cluster Slot Migration Tests")
    print(f"Target: 127.0.0.1:{PORT}")
    print("=" * 60)
    print()

    test_single_node()
    test_two_node_migration()

    print("=" * 60)
    print(f"C1.1 Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
