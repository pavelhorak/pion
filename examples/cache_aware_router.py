#!/usr/bin/env python3
"""
Cache-aware router stub: route by KV.PREFIX.LOOKUP / MEMBERSHIP hits.

Composes three Pion primitives into a single routing decision:

  1. KV.PREFIX.LOOKUP <ns_key>                          (fast path — exact-prefix HIT)
  2. KV.PREFIX.MEMBERSHIP <ns_key> <count> <hash_blob>  (per-pod block-residency bitmap)
  3. AI.ROUTE <query_embedding_fp32>                    (semantic fallback when nothing is warm)

Tier 1 is microseconds, tier 2 is ~125 µs p50 e2e on a 1,562-block probe (Apple Silicon,
loopback), tier 3 is ~0.14 ms. All three answer "where should this request go?" against
data that lives in Pion, not in the inference servers.

Usage:
    # 1. Start Pion with the kv-cache surface
    ./pion-server --kvcache -w 1

    # 2. Run the demo (registers 3 fake pods + 1 warm prefix, routes 3 queries)
    python3 examples/cache_aware_router.py

This is a stub, not production: a real router would (a) compute pod block tables from
live inference-server telemetry, (b) score per-pod coverage against pod load, and
(c) cache LOOKUP results to avoid the 0.5 ms per-query roundtrip.
"""
from __future__ import annotations

import argparse
import hashlib
import socket
import struct
import sys
from typing import Optional

import redis  # pip install redis


def _pack_hashes(hashes: list[int]) -> bytes:
    """Pack u64 block hashes as little-endian for KV.PREFIX.REGISTER/MEMBERSHIP."""
    return struct.pack(f"<{len(hashes)}Q", *hashes)


def _pack_fp32(vec: list[float]) -> bytes:
    """Pack an fp32 embedding for AI.ROUTE / AI.ROUTE.REGISTER."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _block_hash(token_block: bytes) -> int:
    """Demo block-hash: low 64 bits of blake2b. Production routers should match
    whatever hash the inference server uses to identify cached blocks."""
    return int.from_bytes(hashlib.blake2b(token_block, digest_size=8).digest(), "little")


def _popcount_bytes(b: bytes) -> int:
    return sum(bin(x).count("1") for x in b)


class CacheAwareRouter:
    """Three-tier router: exact prefix → block coverage → semantic fallback."""

    def __init__(self, pion_host: str, pion_port: int, coverage_threshold: float = 0.5):
        # decode_responses=False keeps MEMBERSHIP bitmaps and bulk endpoints as bytes.
        self.r = redis.Redis(host=pion_host, port=pion_port, decode_responses=False)
        self.coverage_threshold = coverage_threshold

    def lookup(self, ns_key: str) -> bool:
        """Tier 1 — exact-prefix residency. Returns True on +HIT."""
        reply = self.r.execute_command("KV.PREFIX.LOOKUP", ns_key)
        return reply == b"HIT"

    def membership(self, ns_key: str, probe_hashes: list[int]) -> Optional[bytes]:
        """Tier 2 — within-prefix block residency. Returns a bitmap of len ceil(K/8),
        or None if the namespace has no block table registered (+UNKNOWN)."""
        blob = _pack_hashes(probe_hashes)
        reply = self.r.execute_command("KV.PREFIX.MEMBERSHIP", ns_key, len(probe_hashes), blob)
        if reply == b"UNKNOWN":
            return None
        return reply

    def coverage(self, ns_key: str, probe_hashes: list[int]) -> float:
        """Tier 2 scored — fraction of probe_hashes resident on this pod's namespace."""
        bitmap = self.membership(ns_key, probe_hashes)
        if bitmap is None or not probe_hashes:
            return 0.0
        return _popcount_bytes(bitmap) / len(probe_hashes)

    def semantic_route(self, query_embedding: list[float]) -> Optional[str]:
        """Tier 3 — embedding-keyed routing. Returns the endpoint or None."""
        reply = self.r.execute_command("AI.ROUTE", _pack_fp32(query_embedding))
        return reply.decode() if reply else None

    def route_dim(self) -> int:
        """Read the semantic router's centroid dimension from AI.ROUTE.INFO.
        Inference sidecar defaults to MiniLM-L6-v2 (384); standalone defaults
        to ROUTE_EMBED_DIM (768). Either way, we want the live value."""
        info = self.r.execute_command("AI.ROUTE.INFO")
        text = info.decode() if isinstance(info, bytes) else info
        for line in text.split("\r\n"):
            if line.startswith("dimensions:"):
                return int(line.split(":", 1)[1])
        return 768

    def route(self, pods: dict[str, dict], prompt_hash: str,
              probe_hashes: list[int], query_embedding: list[float]) -> tuple[str, str]:
        """
        Pick a pod. Returns (endpoint, tier_used).

        pods             — {endpoint: {"exact_prefix": "pod-X/exact", "blocks_ns": "pod-X/blocks"}}
        prompt_hash      — stable string identifier for the incoming prompt
        probe_hashes     — token-block hashes for the incoming prompt
        query_embedding  — fp32 vector (matches the router's centroid dim)

        Tier-1 probes "{pod_exact_prefix}/{prompt_hash}" per pod — a router-maintained
        registry of which pods have memoized which exact prompts. Tier-2 probes the
        stable per-pod blocks namespace for partial-block residency. Tier-3 is the
        semantic-similarity fallback when nothing is warm.
        """
        # Tier 1: exact-prefix per pod, keyed on (pod, prompt_hash)
        for endpoint, ns in pods.items():
            if self.lookup(f"{ns['exact_prefix']}/{prompt_hash}"):
                return (endpoint, "exact-prefix")

        # Tier 2: per-pod block coverage; pick the best if it clears threshold
        best_pod, best_cov = None, 0.0
        for endpoint, ns in pods.items():
            cov = self.coverage(ns["blocks_ns"], probe_hashes)
            if cov > best_cov:
                best_pod, best_cov = endpoint, cov
        if best_pod and best_cov >= self.coverage_threshold:
            return (f"{best_pod} (coverage={best_cov:.0%})", "block-coverage")

        # Tier 3: semantic fallback
        endpoint = self.semantic_route(query_embedding)
        if endpoint:
            return (endpoint, "semantic")

        # Last resort: caller picks (least-loaded, round-robin, etc.)
        return ("(no route — fall through to least-loaded)", "fallback")


def _demo(router: CacheAwareRouter) -> None:
    """Register 3 fake pods + 1 warm prefix, then route 3 queries showing each tier."""
    # AI.ROUTE centroid dim is discovered live (varies with --inference flag).
    # The demo uses sparse fake centroids that still give clear cosine winners.
    dim = router.route_dim()
    def _onehot(slot: int) -> list[float]:
        v = [0.0] * dim
        v[slot] = 1.0
        return v

    pods = {
        "http://pod-a:8000": {"exact_prefix": "pod-a/exact", "blocks_ns": "pod-a/blocks", "centroid": _onehot(0)},
        "http://pod-b:8000": {"exact_prefix": "pod-b/exact", "blocks_ns": "pod-b/blocks", "centroid": _onehot(1)},
        "http://pod-c:8000": {"exact_prefix": "pod-c/exact", "blocks_ns": "pod-c/blocks", "centroid": _onehot(2)},
    }
    for endpoint, ns in pods.items():
        # Idempotent: REMOVE may fail with "node not found" on first run, that's fine.
        try:
            router.r.execute_command("AI.ROUTE.REMOVE", endpoint)
        except redis.ResponseError:
            pass
        router.r.execute_command("AI.ROUTE.REGISTER", endpoint, endpoint, _pack_fp32(ns["centroid"]))

    # pod-a's block table — stable, registered once
    warm_blocks = [_block_hash(f"block-{i}".encode()) for i in range(8)]
    router.r.execute_command(
        "KV.PREFIX.REGISTER", "pod-a/blocks", "512", "fp16",
        "BLOCKS", "64", str(len(warm_blocks)), _pack_hashes(warm_blocks),
    )
    # pod-a memoized a specific prompt — separate exact-prefix entry
    Q1_HASH = "abc123"
    router.r.execute_command("KV.PREFIX.REGISTER", f"pod-a/exact/{Q1_HASH}", "512", "fp16")

    q3_emb = [0.0] * dim
    q3_emb[2] = 0.9  # leans toward pod-c

    cases = [
        ("Q1 exact warm prompt on pod-a", Q1_HASH, warm_blocks, pods["http://pod-a:8000"]["centroid"]),
        ("Q2 new prompt, 4/8 blocks on pod-a", "newQ2",
            warm_blocks[:4] + [_block_hash(f"new-{i}".encode()) for i in range(4)],
            pods["http://pod-b:8000"]["centroid"]),
        ("Q3 cold prompt, semantic fallback", "coldQ3",
            [_block_hash(f"cold-{i}".encode()) for i in range(8)], q3_emb),
    ]
    for label, prompt_hash, blocks, emb in cases:
        endpoint, tier = router.route(pods, prompt_hash, blocks, emb)
        print(f"  {label:42s} → {endpoint:46s} [{tier}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    args = ap.parse_args()

    try:
        router = CacheAwareRouter(args.pion_host, args.pion_port)
        router.r.ping()
    except (redis.ConnectionError, socket.error) as e:
        print(f"cannot reach Pion at {args.pion_host}:{args.pion_port} — {e}", file=sys.stderr)
        print("start it with:  ./pion-server --kvcache -w 1", file=sys.stderr)
        return 1

    print(f"connected to pion://{args.pion_host}:{args.pion_port}")
    print("routing 3 queries through tier-1 (exact) → tier-2 (block coverage) → tier-3 (semantic):")
    _demo(router)
    return 0


if __name__ == "__main__":
    sys.exit(main())
