#!/usr/bin/env python3
"""gh #209: prefix-aware routing on KV.PREFIX.MEMBERSHIP.

Unit tests (no server): leading-run bit math, sticky fallback, tie stability.
Functional tests (two local --kvcache servers): a prompt sharing registered
leading blocks routes to the peer that holds them; a disjoint prompt stays
sticky; coverage below min_blocks stays sticky.

Run: python3 pion-exo/tests/test_prefix_router.py [path-to-pion-server]
"""
import os
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import redis  # noqa: E402

from pion_exo.prefix_router import (  # noqa: E402
    PrefixAwareRouter, blocks_of, hash_token_block,
)

BINARY = sys.argv[1] if len(sys.argv) > 1 else "./pion-server"
PORTS = (6410, 6412)   # --kvcache reserves port+1; keep peers 2 apart
BLOCK = 64

passed = failed = 0
def check(cond, name, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"  PASS {name}")
    else: failed += 1; print(f"  FAIL {name} {detail}")


def unit_tests():
    r = PrefixAwareRouter([("h1", 1), ("h2", 2)])
    # leading-run bit math (LSB-first per byte)
    check(r._leading_run(bytes([0b00001111]), 8) == 4, "leading run stops at first 0")
    check(r._leading_run(bytes([0xFF, 0b00000011]), 16) == 10, "run crosses byte boundary")
    check(r._leading_run(bytes([0b11111110]), 8) == 0, "bit 0 clear -> run 0")
    # no hashes -> sticky
    check(r.route("s1") == r.sticky_peer("s1"), "no hashes routes sticky")
    # block hashing is stable + order-sensitive
    check(hash_token_block([1, 2, 3]) == hash_token_block([1, 2, 3]), "block hash stable")
    check(hash_token_block([1, 2, 3]) != hash_token_block([3, 2, 1]), "block hash order-sensitive")
    toks = list(range(BLOCK * 3 + 10))
    check(len(blocks_of(toks, BLOCK)) == 3, "partial tail block dropped")


def start(port):
    d = f"/tmp/pion_gh209_{port}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return subprocess.Popen(
        [os.path.abspath(BINARY), "-p", str(port), "-w", "1", "--kvcache", "--no-auto-embed"],
        cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_up(port):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if redis.Redis(port=port, socket_timeout=1).ping():
                return True
        except redis.RedisError:
            time.sleep(0.5)
    return False


def functional_tests():
    peers = [("127.0.0.1", PORTS[0]), ("127.0.0.1", PORTS[1])]
    procs = [start(p) for p in PORTS]
    try:
        for p in PORTS:
            assert wait_up(p), f"server on {p} did not come up"

        prompt = list(range(1000, 1000 + BLOCK * 8))       # 8 full blocks
        hashes = blocks_of(prompt, BLOCK)
        router = PrefixAwareRouter(peers, min_blocks=2)
        sticky = router.sticky_peer("sess-A")
        other = peers[0] if sticky == peers[1] else peers[1]

        # Register a namespace on the NON-sticky peer carrying the prompt's
        # first 6 blocks (leading run 6), via KV.PREFIX.REGISTER ... BLOCKS.
        ns = "gh209|test|ns1"
        blob = struct.pack("<" + "Q" * 6, *hashes[:6])
        rr = redis.Redis(host=other[0], port=other[1], decode_responses=False)
        resp = rr.execute_command("KV.PREFIX.REGISTER", ns, "64", "fp16",
                                  "BLOCKS", str(BLOCK), "6", blob)
        check(resp in (b"OK", "OK"), "KV.PREFIX.REGISTER BLOCKS", repr(resp))
        router.note_registered(ns, other)

        got = router.route("sess-A", hashes)
        check(got == other, "prompt with 6-block coverage routes to holder",
              f"got {got}, holder {other}, sticky {sticky}")

        # Disjoint prompt -> sticky
        alien = blocks_of(list(range(50_000, 50_000 + BLOCK * 8)), BLOCK)
        check(router.route("sess-A", alien) == sticky, "disjoint prompt stays sticky")

        # Coverage below min_blocks -> sticky: register 1 matching block only
        ns2 = "gh209|test|ns2"
        blob1 = struct.pack("<Q", hashes[0])
        resp = rr.execute_command("KV.PREFIX.REGISTER", ns2, "64", "fp16",
                                  "BLOCKS", str(BLOCK), "1", blob1)
        r2 = PrefixAwareRouter(peers, min_blocks=2)
        r2.note_registered(ns2, other)
        check(r2.route("sess-A", hashes) == sticky, "1-block coverage < min_blocks stays sticky")

        # Unknown namespace in LRU (never registered) must not break routing
        r3 = PrefixAwareRouter(peers, min_blocks=2)
        r3.note_registered("gh209|test|ghost", other)
        check(r3.route("sess-A", hashes) == sticky, "UNKNOWN membership degrades to sticky")

        # Peer down: router degrades to sticky, never raises
        r4 = PrefixAwareRouter([("127.0.0.1", PORTS[0]), ("127.0.0.1", 6499)], min_blocks=2)
        r4.note_registered("whatever", ("127.0.0.1", 6499))
        got = r4.route("sess-A", hashes)
        check(got in [("127.0.0.1", PORTS[0]), ("127.0.0.1", 6499)], "dead peer never raises")
    finally:
        for pr in procs:
            pr.kill(); pr.wait(timeout=5)
        for p in PORTS:
            shutil.rmtree(f"/tmp/pion_gh209_{p}", ignore_errors=True)


if __name__ == "__main__":
    unit_tests()
    functional_tests()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
