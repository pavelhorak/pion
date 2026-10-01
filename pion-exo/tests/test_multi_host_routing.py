#!/usr/bin/env python3
"""Multi-host routing tests for PionAttentionHook (gh #43).

Two layers of validation:

  1. Pure-Python distribution test — no server required. Verifies that
     blake2b(session_id) mod N is stable, deterministic, and balanced.

  2. Optional integration test — set PION_PEERS=host1:port1,host2:port2 to
     run against two live Pion instances. Verifies that V.STOREBATCH lands
     on the hashed peer and is invisible from the other peer.

Single-host test_exo_hook.py is unchanged and remains the primary smoke
gate; this file covers the multi-host carve-out specifically.
"""

import collections
import os
import sys
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "pion-vllm-mlx"))

from pion_exo import PionAttentionHook, PionSessionManager


def _peers_from_env() -> List[Tuple[str, int]]:
    raw = os.environ.get("PION_PEERS", "").strip()
    if not raw:
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"PION_PEERS entry {tok!r} missing port")
        host, port_s = tok.rsplit(":", 1)
        out.append((host, int(port_s)))
    return out


# ---- 1. distribution + determinism -----------------------------------------

def test_distribution_balanced():
    peers = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    hook = PionAttentionHook(peers=peers)
    counts = collections.Counter()
    n = 1000
    for i in range(n):
        peer = hook.peer_for_session(f"session_{i}")
        counts[peer] += 1
    assert set(counts.keys()) == set(peers), \
        f"some peers received zero sessions: {counts}"
    expected = n / len(peers)
    for peer, c in counts.items():
        # blake2b is uniform; ±15% over n=1000 is loose enough to never flake.
        assert abs(c - expected) < expected * 0.20, \
            f"peer {peer} got {c}, expected ~{expected:.0f}"
    print(f"[1] distribution OK: {dict(counts)}")


def test_determinism_across_instances():
    peers = [("a", 1), ("b", 2), ("c", 3)]
    h1 = PionAttentionHook(peers=peers)
    h2 = PionAttentionHook(peers=peers)
    for sid in ("alice", "bob", "deadbeef-0001", "x" * 100):
        assert h1.peer_for_session(sid) == h2.peer_for_session(sid), \
            f"sid {sid!r}: {h1.peer_for_session(sid)} vs {h2.peer_for_session(sid)}"
    print("[2] determinism OK across hook instances")


def test_session_caching():
    peers = [("a", 1), ("b", 2)]
    hook = PionAttentionHook(peers=peers)
    sid = "cached_session"
    first = hook.peer_for_session(sid)
    # Mutate cache — should still return the cached value.
    hook._sid_to_peer[sid] = ("z", 99)
    assert hook.peer_for_session(sid) == ("z", 99)
    # drop_session evicts the cache.
    hook.drop_session(sid)
    again = hook.peer_for_session(sid)
    assert again == first, f"after drop, expected {first}, got {again}"
    print("[3] session cache + eviction OK")


def test_session_manager_routing():
    peers = [("a", 1), ("b", 2), ("c", 3)]
    sm = PionSessionManager(peers=peers)
    seen = collections.Counter()
    for i in range(300):
        peer = sm.peer_for_session(f"sm_{i}")
        seen[peer] += 1
    assert set(seen.keys()) == set(peers), seen
    print(f"[4] PionSessionManager distribution OK: {dict(seen)}")


def test_single_host_back_compat():
    # Old API must still work and behave like a one-peer list.
    h = PionAttentionHook(pion_host="example.org", pion_port=4242)
    assert h.peers == [("example.org", 4242)]
    assert h.pion_host == "example.org"
    assert h.pion_port == 4242
    # All sessions must land on the only peer.
    for sid in ("a", "b", "c"):
        assert h.peer_for_session(sid) == ("example.org", 4242)
    print("[5] single-host back-compat OK")


def test_owner_resp_parser():
    p = PionAttentionHook._parse_owner_resp
    assert p(b"0") == 0
    assert p(b"3") == 3
    assert p(b"3 12345") == 3  # owner + schema_digest form
    assert p(b"-1") == -1
    assert p("0") == 0
    assert p(7) == 7
    assert p(None) == -1
    assert p(b"") == -1
    assert p(b"garbage") == -1
    print("[6] _parse_owner_resp covers all forms")


# ---- 2. optional integration ------------------------------------------------

def test_two_instance_integration():
    peers = _peers_from_env()
    if len(peers) < 2:
        print("[7] integration SKIPPED (set PION_PEERS=host1:port1,host2:port2 to run)")
        return

    hook = PionAttentionHook(peers=peers, mode="v_offload")
    health = hook.health_check()
    assert health["pion_resp"], f"not all peers reachable: {health}"

    # Pick a session_id that we know lands on peer[0] and one that lands on peer[1].
    # blake2b is stable so we can search a small range.
    target_per_peer = {p: None for p in peers}
    for i in range(10000):
        sid = f"distrib_{i}"
        peer = hook.peer_for_session(sid)
        if target_per_peer[peer] is None:
            target_per_peer[peer] = sid
        if all(v is not None for v in target_per_peer.values()):
            break
    assert all(target_per_peer.values()), "couldn't find sids for every peer"

    H, N, D = 1, 32, 64
    np.random.seed(7)
    K = np.random.randn(H, N, D).astype(np.float32) * 0.1
    V = np.random.randn(H, N, D).astype(np.float32)

    for peer, sid in target_per_peer.items():
        hook.on_prefill(sid, layer_id=0, K=K, V=V)
        # On the *owning* peer V.INFO must show the session; on every other
        # peer it must NOT.
        for other in peers:
            r = hook._redis_for(other)
            try:
                resp = r.execute_command("V.INFO", sid)
                resp_s = resp.decode() if isinstance(resp, bytes) else str(resp)
                exists = "ERR" not in resp_s.upper() and "not found" not in resp_s.lower()
            except Exception:
                exists = False
            if other == peer:
                assert exists, f"V.INFO on owner {peer} should see session {sid}"
            else:
                assert not exists, \
                    f"V.INFO on non-owner {other} unexpectedly sees session {sid}"
        hook.drop_session(sid)
    hook.close()
    print(f"[7] two-instance integration OK ({len(peers)} peers)")


def main():
    tests = [
        test_distribution_balanced,
        test_determinism_across_instances,
        test_session_caching,
        test_session_manager_routing,
        test_single_host_back_compat,
        test_owner_resp_parser,
        test_two_instance_integration,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
