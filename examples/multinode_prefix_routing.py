#!/usr/bin/env python3
"""
Q5: Multi-Node Prefix Cache Routing Demo.

Shows cross-node prefix affinity routing for a cluster of inference backends
(exo / llama.cpp / vllm-mlx / Ollama) behind Pion.

Each backend registers its warmed system-prompt prefixes in Pion:
    HSET prefix:{sha256(system_prompt)[:16]}  node "{url}"  warmed_at {ts}

The router reads Pion before forwarding, sending prefix-matched requests to
the warm node instead of a cold one. Measures TTFT difference.

Works with any two OpenAI-compatible HTTP endpoints. Stub mode requires no
real inference backend.

Usage:
    # Stub mode (no inference backends needed)
    ./pion-server -w 1
    python3 examples/multinode_prefix_routing.py

    # Real two-node setup
    python3 examples/multinode_prefix_routing.py \\
        --node-a http://host-a:11434 \\
        --node-b http://host-b:11434

    # Two-machine test: run Pion on a shared host
    python3 examples/multinode_prefix_routing.py \\
        --pion-host 192.168.1.10 \\
        --node-a http://host-a:11434 \\
        --node-b http://host-b:11434

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
WARM_TTL = 3600  # seconds — how long a warmed prefix stays registered
COLD_PENALTY_MS = 80  # simulated cold-start overhead in stub mode

# ── Stub inference backend ────────────────────────────────────────────────────

class InferenceStubHandler(BaseHTTPRequestHandler):
    """Simulates an exo/llama.cpp/vllm-mlx node. Warm requests are faster."""
    node_name: str = "node-A"
    warm_prefixes: set[str] = set()

    def do_POST(self) -> None:  # noqa: N802
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len)
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        messages = body.get("messages", [])
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        key = hashlib.sha256(system.encode()).hexdigest()[:16]
        warm = key in self.__class__.warm_prefixes
        latency_ms = 10 if warm else COLD_PENALTY_MS

        # Register this prefix as warmed after first request
        self.__class__.warm_prefixes.add(key)

        time.sleep(latency_ms / 1000)

        resp = json.dumps({
            "id": "stub-001",
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": f"[{self.__class__.node_name}:{'warm' if warm else 'cold'}] ok"},
                "finish_reason": "stop",
                "index": 0,
            }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass

def _start_stub(port: int, name: str) -> ThreadingHTTPServer:
    handler = type(f"Stub_{name}", (InferenceStubHandler,), {
        "node_name": name,
        "warm_prefixes": set(),
    })
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

# ── Prefix router ─────────────────────────────────────────────────────────────

class PrefixRouter:
    """Routes requests by system-prompt prefix hash, using Pion as the store."""

    def __init__(self, r: redis.Redis, nodes: list[str]) -> None:
        self.r = r
        self.nodes = nodes
        self._rr = 0

    @staticmethod
    def _prefix_key(system_prompt: str) -> str:
        return "prefix:" + hashlib.sha256(system_prompt.encode()).hexdigest()[:16]

    def register_warm(self, node_url: str, system_prompt: str) -> None:
        """Node calls this after it has processed a request with this system prompt."""
        key = self._prefix_key(system_prompt)
        self.r.hset(key, mapping={"node": node_url, "warmed_at": str(int(time.time()))})
        self.r.expire(key, WARM_TTL)

    def route(self, system_prompt: str) -> str:
        """Returns the URL of the warm node, or a least-loaded node."""
        key = self._prefix_key(system_prompt)
        warm_node = self.r.hget(key, "node")
        if warm_node:
            url = warm_node.decode() if isinstance(warm_node, bytes) else warm_node
            if url in self.nodes:
                return url
        # Fallback: round-robin
        node = self.nodes[self._rr % len(self.nodes)]
        self._rr += 1
        return node

    def complete(self, system_prompt: str, user_msg: str) -> tuple[str, float, str]:
        """Returns (content, ttft_ms, node_url)."""
        node = self.route(system_prompt)
        t0 = time.perf_counter()
        resp = requests.post(
            f"{node}/v1/chat/completions",
            json={
                "model": "llama3.2:1b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        ttft_ms = (time.perf_counter() - t0) * 1000
        content = resp.json()["choices"][0]["message"]["content"]
        # Register this prefix as warmed on the node that served it
        self.register_warm(node, system_prompt)
        return content, ttft_ms, node

# ── Helpers ───────────────────────────────────────────────────────────────────

def p50(data: list[float]) -> float:
    return statistics.median(data)

def mean(data: list[float]) -> float:
    return statistics.mean(data)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-node prefix cache routing demo")
    parser.add_argument("--node-a", default=None, help="URL of inference node A")
    parser.add_argument("--node-b", default=None, help="URL of inference node B")
    parser.add_argument("--pion-host", default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=PION_PORT)
    parser.add_argument("--rounds", type=int, default=5, help="Requests per system prompt per condition")
    args = parser.parse_args()

    r = redis.Redis(host=args.pion_host, port=args.pion_port, decode_responses=True)
    try:
        r.ping()
    except Exception as e:
        print(f"ERROR: Cannot connect to Pion at {args.pion_host}:{args.pion_port}: {e}")
        sys.exit(1)

    # Start stubs or use real nodes
    stubs = []
    if args.node_a and args.node_b:
        nodes = [args.node_a.rstrip("/"), args.node_b.rstrip("/")]
        print(f"Real nodes: {nodes[0]}  {nodes[1]}")
    else:
        print("Stub mode: starting fake inference nodes on :52420 and :52421")
        stubs.append(_start_stub(52420, "node-A"))
        stubs.append(_start_stub(52421, "node-B"))
        nodes = ["http://127.0.0.1:52420", "http://127.0.0.1:52421"]
        time.sleep(0.1)

    router = PrefixRouter(r, nodes)

    # System prompts — two distinct personas
    SYSTEM_A = "You are a helpful coding assistant. Always respond in Python."
    SYSTEM_B = "You are a customer support agent for a cloud storage product. Be concise."

    print(f"\nPion:    {args.pion_host}:{args.pion_port}")
    print(f"Node A:  {nodes[0]}")
    print(f"Node B:  {nodes[1]}")

    # ── Phase 1: warm both prefixes ──────────────────────────────────────────
    print("\n── Phase 1: Warming prefixes (first request each) ──")
    _, t_a, n_a = router.complete(SYSTEM_A, "How do I read a file?")
    print(f"  System A → {n_a.split('/')[-1]}  TTFT={t_a:.1f}ms  [cold: prefix registered in Pion]")
    _, t_b, n_b = router.complete(SYSTEM_B, "How do I upgrade my plan?")
    print(f"  System B → {n_b.split('/')[-1]}  TTFT={t_b:.1f}ms  [cold: prefix registered in Pion]")

    # ── Phase 2: routed warm requests ────────────────────────────────────────
    print(f"\n── Phase 2: Warm routing ({args.rounds} requests each) ──")
    warm_a, warm_b = [], []
    for i in range(args.rounds):
        _, t, n = router.complete(SYSTEM_A, f"Q{i+1}: how do I sort a list?")
        warm_a.append(t)
        warm_b_node = router.route(SYSTEM_B)
        print(f"  [{i+1}] A → {n.split('/')[-1]}  {t:.1f}ms", end="  ")
        _, t, n2 = router.complete(SYSTEM_B, f"Q{i+1}: how do I cancel my subscription?")
        warm_b.append(t)
        print(f"B → {n2.split('/')[-1]}  {t:.1f}ms")

    # ── Phase 3: cold-node comparison ────────────────────────────────────────
    print(f"\n── Phase 3: Cold node penalty (force wrong node) ──")
    cold_a = []
    wrong_node = nodes[1] if n_a == nodes[0] else nodes[0]
    for i in range(min(args.rounds, 3)):
        t0 = time.perf_counter()
        requests.post(
            f"{wrong_node}/v1/chat/completions",
            json={"model": "x", "messages": [
                {"role": "system", "content": SYSTEM_A},
                {"role": "user", "content": f"Q{i+1}: define a class"},
            ]},
            timeout=30,
        ).raise_for_status()
        cold_a.append((time.perf_counter() - t0) * 1000)
        print(f"  [{i+1}] A → WRONG node {wrong_node.split('/')[-1]}  {cold_a[-1]:.1f}ms  [cold]")

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────")
    print(f"  System A (coding):    warm p50={p50(warm_a):.1f}ms  cold p50={p50(cold_a):.1f}ms")
    print(f"  System B (support):   warm p50={p50(warm_b):.1f}ms")
    if cold_a and warm_a:
        improvement = (p50(cold_a) - p50(warm_a)) / p50(cold_a) * 100
        print(f"\n  TTFT improvement (warm vs cold): {improvement:.1f}%")
        if improvement > 5:
            print(f"  ✅ Prefix routing reduces TTFT by {improvement:.1f}% for repeated system prompts")
        else:
            print("  ⚠️  Small improvement — stub latency gap may be smaller than inference backend overhead")

    # ── Routing correctness ──────────────────────────────────────────────────
    print("\n── Routing Correctness ──────────────────────────")
    key_a = PrefixRouter._prefix_key(SYSTEM_A)
    key_b = PrefixRouter._prefix_key(SYSTEM_B)
    stored_a = r.hget(key_a, "node") or "?"
    stored_b = r.hget(key_b, "node") or "?"
    correct = (stored_a == n_a and stored_b == n_b)
    print(f"  prefix_A → {stored_a.split('/')[-1] if stored_a != '?' else '?'}")
    print(f"  prefix_B → {stored_b.split('/')[-1] if stored_b != '?' else '?'}")
    print(f"  {'✅ both prefixes correctly registered in Pion' if correct else '⚠️  prefix mismatch — check routing logic'}")

    # ── Multi-machine note ────────────────────────────────────────────────────
    print("""
── Multi-Machine Setup Notes ──────────────────────────────────────────────────
  To test cross-machine prefix routing:
  1. Run Pion on a shared host reachable by both machines.
  2. On machine A: register warmed prefixes via HSET prefix:{hash} node "http://machine-a:port"
  3. On machine B: this router reads the same Pion instance → routes to machine A
  4. Run this script with:
       --pion-host <shared-host> --node-a http://machine-a:port --node-b http://machine-b:port
  Pion acts as the cross-machine coordination store; no changes to inference backends needed.
""")

    for s in stubs:
        s.shutdown()


if __name__ == "__main__":
    main()
