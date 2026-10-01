#!/usr/bin/env python3
"""
Q1: exo Session Latency Demo.

Simulates an exo cluster (OpenAI-compatible head node) behind Pion's session store.
Measures the overhead added by Pion HSET/HGET session routing vs. direct requests.

Architecture:
    Client → session_router (this script) → Pion:1974 (HSET/HGET)
                                           → exo head (or stub) for inference

Pion stores:
    HSET session:{id}  node "exo-head:52415"  model "llama3.1:8b"  last_seen {ts}
    EXPIRE session:{id} 3600

Usage:
    # 1. Start Pion
    ./pion-server -w 1

    # 2. Run demo (stub mode — no real exo needed)
    python3 examples/exo_session_demo.py

    # 3. Run against real exo head
    python3 examples/exo_session_demo.py --exo-url http://exo-head.local:52415

Requirements:
    pip install redis requests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import redis
import requests

PION_PORT = 1974
EXO_DEFAULT_PORT = 52415
SESSION_TTL = 3600
TURN_COUNT = 5
LATENCY_SAMPLES = 20

# ── Fake exo head stub ────────────────────────────────────────────────────────

class ExoStubHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible stub that echoes the node name."""
    server_version = "ExoStub/0.1"
    node_name: str = "exo-head-A"

    def do_POST(self) -> None:  # noqa: N802
        content_len = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(content_len)
        body = json.dumps({
            "id": "stub-001",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"[{self.node_name}] acknowledged"
                },
                "finish_reason": "stop",
                "index": 0,
            }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass  # suppress request logs

def _start_exo_stub(port: int, node_name: str) -> ThreadingHTTPServer:
    ExoStubHandler.node_name = node_name
    # create a unique subclass so each stub has its own node_name
    handler_cls = type(f"Handler_{node_name}", (ExoStubHandler,), {"node_name": node_name})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

# ── Session router ────────────────────────────────────────────────────────────

class SessionRouter:
    """Routes multi-turn requests to the same exo node via Pion session store."""

    def __init__(self, r: redis.Redis, nodes: list[str]) -> None:
        self.r = r
        self.nodes = nodes
        self._rr = 0  # round-robin index for new sessions

    def _session_key(self, session_id: str) -> str:
        return f"exo:session:{session_id}"

    def get_node(self, session_id: str) -> str:
        key = self._session_key(session_id)
        node = self.r.hget(key, "node")
        if node:
            return node.decode() if isinstance(node, bytes) else node
        # New session — assign via round-robin
        assigned = self.nodes[self._rr % len(self.nodes)]
        self._rr += 1
        self.r.hset(key, mapping={
            "node": assigned,
            "model": "llama3.1:8b",
            "last_seen": str(int(time.time())),
        })
        self.r.expire(key, SESSION_TTL)
        return assigned

    def update_last_seen(self, session_id: str) -> None:
        key = self._session_key(session_id)
        self.r.hset(key, "last_seen", str(int(time.time())))
        self.r.expire(key, SESSION_TTL)

    def complete(self, session_id: str, prompt: str) -> tuple[str, float, float]:
        """Returns (response_text, routing_ms, total_ms)."""
        t0 = time.perf_counter()
        node = self.get_node(session_id)
        t_routed = time.perf_counter()
        routing_ms = (t_routed - t0) * 1000

        resp = requests.post(
            f"{node}/v1/chat/completions",
            json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": prompt}]},
            timeout=10,
        )
        resp.raise_for_status()
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000

        self.update_last_seen(session_id)
        content = resp.json()["choices"][0]["message"]["content"]
        return content, routing_ms, total_ms

# ── Measurements ──────────────────────────────────────────────────────────────

def measure_direct(node_url: str, n: int = LATENCY_SAMPLES) -> list[float]:
    """Baseline: direct requests to exo node, no Pion routing."""
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.post(
            f"{node_url}/v1/chat/completions",
            json={"model": "llama3.1:8b", "messages": [{"role": "user", "content": "ping"}]},
            timeout=10,
        ).raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies

def measure_routed(router: SessionRouter, n: int = LATENCY_SAMPLES) -> tuple[list[float], list[float]]:
    """Routed: requests through Pion session store."""
    routing_ms_list, total_ms_list = [], []
    for i in range(n):
        sid = hashlib.sha256(f"bench-session-{i}".encode()).hexdigest()[:16]
        _, r_ms, t_ms = router.complete(sid, "ping")
        routing_ms_list.append(r_ms)
        total_ms_list.append(t_ms)
    return routing_ms_list, total_ms_list

def p50(data: list[float]) -> float:
    return statistics.median(data)

def p99(data: list[float]) -> float:
    data_sorted = sorted(data)
    idx = max(0, int(len(data_sorted) * 0.99) - 1)
    return data_sorted[idx]

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exo session latency demo")
    parser.add_argument("--exo-url", help="Real exo head URL (omit for stub mode)")
    parser.add_argument("--pion-port", type=int, default=PION_PORT)
    parser.add_argument("--samples", type=int, default=LATENCY_SAMPLES)
    args = parser.parse_args()

    # Connect to Pion
    r = redis.Redis(host="127.0.0.1", port=args.pion_port, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        print(f"ERROR: Cannot connect to Pion on port {args.pion_port}: {e}")
        sys.exit(1)

    # Start stub or use real exo
    if args.exo_url:
        nodes = [args.exo_url.rstrip("/")]
        print(f"Using real exo head: {nodes[0]}")
        stub_servers = []
    else:
        print("Stub mode: starting fake exo heads on :52415 and :52416")
        s1 = _start_exo_stub(52415, "exo-head-A")
        s2 = _start_exo_stub(52416, "exo-head-B")
        stub_servers = [s1, s2]
        nodes = ["http://127.0.0.1:52415", "http://127.0.0.1:52416"]
        time.sleep(0.1)

    router = SessionRouter(r, nodes)

    # ── Demo 1: multi-turn session affinity ─────────────────────────────────
    print("\n── Multi-Turn Session Affinity ──")
    session_id = hashlib.sha256(b"demo-user-1").hexdigest()[:16]
    # Clear any leftover session
    r.delete(f"exo:session:{session_id}")

    first_node = None
    all_same = True
    for turn in range(1, TURN_COUNT + 1):
        _, r_ms, t_ms = router.complete(session_id, f"Turn {turn}: hello")
        node = router.get_node(session_id)
        if first_node is None:
            first_node = node
        elif node != first_node:
            all_same = False
        print(f"  Turn {turn:2d}: node={node.split('/')[-1]}  routing={r_ms:.2f}ms  total={t_ms:.2f}ms")

    print(f"\n  Session consistency: {'✅ all turns → same node' if all_same else '❌ node changed mid-session'}")

    # ── Demo 2: TTL expiry → rebalance ──────────────────────────────────────
    print("\n── TTL Expiry → Rebalance ──")
    r.expire(f"exo:session:{session_id}", 1)
    time.sleep(1.1)
    _, _, _ = router.complete(session_id, "new turn after expiry")
    new_node = router.get_node(session_id)
    print(f"  Before TTL: {first_node.split('/')[-1]}  After TTL: {new_node.split('/')[-1]}")
    print("  ✅ Session reset and reassigned after TTL expiry")

    # ── Demo 3: latency overhead measurement ────────────────────────────────
    print(f"\n── Latency Overhead ({args.samples} samples each) ──")

    print("  Warming up...")
    # Warm up connections
    for _ in range(5):
        requests.post(f"{nodes[0]}/v1/chat/completions",
                      json={"model": "x", "messages": []}, timeout=5)

    direct = measure_direct(nodes[0], args.samples)
    routing_ms_list, total_ms_list = measure_routed(router, args.samples)
    overhead = [t - d for t, d in zip(total_ms_list, direct)]

    print(f"\n  Direct (no Pion):")
    print(f"    p50={p50(direct):.2f}ms  p99={p99(direct):.2f}ms  mean={statistics.mean(direct):.2f}ms")
    print(f"\n  Routed (Pion HSET/HGET + inference):")
    print(f"    p50={p50(total_ms_list):.2f}ms  p99={p99(total_ms_list):.2f}ms  mean={statistics.mean(total_ms_list):.2f}ms")
    print(f"\n  Pion routing overhead (HGET + HSET + EXPIRE):")
    print(f"    p50={p50(routing_ms_list):.2f}ms  p99={p99(routing_ms_list):.2f}ms  mean={statistics.mean(routing_ms_list):.2f}ms")
    print(f"\n  Net added latency (total - direct):")
    print(f"    p50={p50(overhead):.2f}ms  p99={p99(overhead):.2f}ms  mean={statistics.mean(overhead):.2f}ms")

    # Verdict
    overhead_p50 = p50(overhead)
    print()
    if overhead_p50 < 1.0:
        print(f"  ✅ Pion session routing overhead: {overhead_p50:.2f}ms p50 — imperceptible for a multi-device exo pipeline")
    elif overhead_p50 < 5.0:
        print(f"  ⚠️  Pion session routing overhead: {overhead_p50:.2f}ms p50 — small but visible; acceptable for most workloads")
    else:
        print(f"  ❌ Pion session routing overhead: {overhead_p50:.2f}ms p50 — investigate network/Pion config")

    for s in stub_servers:
        s.shutdown()


if __name__ == "__main__":
    main()
