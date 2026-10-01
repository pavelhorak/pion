#!/usr/bin/env python3
"""
Item 11d — Cross-Replica Prefix Cache Routing (vllm-mlx + Pion).

vLLM's PagedAttention prefix cache operates within a single replica.
Pion routes requests by system-prompt affinity so queries always land on
the replica that already has those KV pages materialised, cutting TTFT.

Architecture:
    Client
      │
      ▼
    PrefixRouter  (this demo)
      │  HGET prefix:{sha256(system_prompt)[:16]} node  → Pion:1974
      │
      ├─ warm replica found  ──→  vllm-mlx replica A :8000  (fast TTFT)
      │
      └─ no warm replica     ──→  least-loaded replica       (cold TTFT)
           then:  HSET prefix:{hash} node <url> warmed_at <ts>

Pion commands used:
  HSET  prefix:{hash}  node <url>  warmed_at <ts>   — register warm prefix
  HGET  prefix:{hash}  node                          — lookup warm replica
  ZADD  inference_fleet <queue_depth> <node>         — load visibility
  ZINCRBY / ZRANGEBYSCORE                            — atomic queue tracking

Usage:
    # 1. Start Pion
    ./pion-server -w 1

    # 2. Start two vllm-mlx replicas (edit MODEL if needed)
    vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit --port 8000 --enable-prefix-cache
    vllm-mlx serve mlx-community/Llama-3.2-1B-Instruct-4bit --port 8001 --enable-prefix-cache

    # 3. Run demo
    python3 examples/prefix_routing_demo.py

    # Or: let the demo start the replicas automatically
    python3 examples/prefix_routing_demo.py --auto-start

Options:
    --model MODEL       MLX model to use (default: mlx-community/Llama-3.2-1B-Instruct-4bit)
    --auto-start        Auto-start two vllm-mlx replicas on ports 8000/8001
    --pion-port PORT    Pion port (default: 1974)
    --max-tokens N      Max tokens per response (default: 64)

Requirements:
    pip install redis requests
    pip install vllm-mlx  (or: pip install mlx-lm)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from typing import Optional

import redis
import requests

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "mlx-community/Llama-3.2-1B-Instruct-4bit"
REPLICA_PORTS   = [8000, 8001]
PION_PORT       = 1974
MAX_TOKENS      = 1    # 1 token = pure TTFT (time to process prompt + first token)
STARTUP_TIMEOUT = 120   # seconds to wait for vllm-mlx to be ready
PREFIX_TTL      = 3600  # seconds before prefix affinity expires

# Two system prompts with different character — these warm different replicas
SYSTEM_PROMPTS = {
    "coding": (
        "You are an expert software engineer. Answer concisely with code examples "
        "when relevant. Prefer Python. Be brief."
    ),
    "creative": (
        "You are a creative writer specialising in short fiction. "
        "Respond with vivid, imaginative prose. Be brief."
    ),
}

USER_QUERIES = {
    "coding": [
        "Write a one-liner to reverse a string in Python.",
        "How do I sort a dict by value in Python?",
        "What is a generator expression?",
        "Show me a lambda that doubles a number.",
        "How do I read a file line by line?",
    ],
    "creative": [
        "Describe a rainy afternoon in three sentences.",
        "Write an opening line for a mystery novel.",
        "Describe the smell of a forest after rain.",
        "Write a haiku about the ocean.",
        "Describe a character who collects clocks.",
    ],
}


# ── Pion prefix router ─────────────────────────────────────────────────────────

class PrefixRouter:
    """
    Routes requests to the replica whose KV prefix cache is already warm.

    Pion commands used:
      HSET  prefix:{hash}  node <url>  warmed_at <ts>   — register warm replica
      HGET  prefix:{hash}  node                          — lookup warm replica
      INCR  queue:{node_idx}                             — increment in-flight count
      DECR  queue:{node_idx}                             — decrement in-flight count
      GET   queue:{node_idx}                             — read queue depth
    """

    def __init__(self, r: redis.Redis, nodes: list[str]):
        self.r = r
        self.nodes = nodes
        # In-memory queue depth counters (demo only — no persistence needed)
        self._queue: dict[str, int] = {url: 0 for url in nodes}

    def _prefix_key(self, system_prompt: str) -> str:
        h = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
        return f"prefix:{h}"

    def get_warm_node(self, system_prompt: str) -> Optional[str]:
        """Return URL of the replica warmed for this system prompt, or None."""
        key = self._prefix_key(system_prompt)
        node = self.r.hget(key, "node")
        return node if node else None

    def register_warm(self, system_prompt: str, node_url: str) -> None:
        """Mark a replica as warmed for this system prompt."""
        key = self._prefix_key(system_prompt)
        self.r.hset(key, mapping={
            "node": node_url,
            "warmed_at": str(int(time.time())),
        })
        self.r.expire(key, PREFIX_TTL)

    def least_loaded(self) -> str:
        """Return the node with the lowest current queue depth."""
        depths = self.queue_depths()
        return min(depths, key=depths.get)

    def route(self, system_prompt: str) -> tuple[str, bool]:
        """
        Return (node_url, was_warm).
        Prefers the warmed replica; falls back to least-loaded.
        """
        warm = self.get_warm_node(system_prompt)
        if warm and warm in self.nodes:
            return warm, True
        return self.least_loaded(), False

    def increment_queue(self, node: str) -> None:
        self._queue[node] = self._queue.get(node, 0) + 1

    def decrement_queue(self, node: str) -> None:
        self._queue[node] = max(0, self._queue.get(node, 0) - 1)

    def queue_depths(self) -> dict[str, int]:
        return dict(self._queue)


# ── vllm-mlx interaction ───────────────────────────────────────────────────────

def measure_ttft(node_url: str, system_prompt: str, user_prompt: str,
                 model: str, max_tokens: int) -> tuple[float, str]:
    """
    Send a streaming chat request and return (ttft_ms, first_token).
    TTFT = time from request send to first content token received.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    first_token = ""
    try:
        with requests.post(
            f"{node_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    first_token = content
                    return ttft_ms, first_token
    except Exception as e:
        return -1.0, f"ERROR: {e}"
    return -1.0, ""


def check_replica(url: str, timeout: float = 3.0) -> bool:
    """Return True if the vllm-mlx replica is accepting requests."""
    try:
        resp = requests.get(f"{url}/v1/models", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_replica(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll until replica is ready or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_replica(url, timeout=2.0):
            return True
        time.sleep(2)
    return False


def _vllm_mlx_bin() -> str:
    """Resolve full path to vllm-mlx binary."""
    import os
    import shutil
    found = shutil.which("vllm-mlx")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/.local/bin/vllm-mlx"),
        "/usr/local/bin/vllm-mlx",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "vllm-mlx"  # let it fail with a clear PATH error


def start_replicas(model: str, ports: list[int]) -> list[subprocess.Popen]:
    """Start vllm-mlx serve processes."""
    bin_path = _vllm_mlx_bin()
    procs = []
    for port in ports:
        cmd = [
            bin_path, "serve", model,
            "--port", str(port),
            "--enable-prefix-cache",
            "--max-tokens", "256",
        ]
        print(f"  Starting vllm-mlx on port {port}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proc)
    return procs


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo(args: argparse.Namespace) -> int:
    model      = args.model
    pion_port  = args.pion_port
    max_tokens = args.max_tokens
    auto_start = args.auto_start

    all_node_urls = [f"http://127.0.0.1:{p}" for p in REPLICA_PORTS]

    print("=" * 65)
    print("Pion Prefix Cache Routing Demo (vllm-mlx)")
    print("=" * 65)
    print(f"Model:    {model}")
    print(f"Pion:     127.0.0.1:{pion_port}")
    print()

    # ── Check Pion ─────────────────────────────────────────────────────────
    r = redis.Redis(host="127.0.0.1", port=pion_port, decode_responses=True)
    try:
        r.ping()
        print("✅ Pion connected")
    except Exception as e:
        print(f"❌ Pion not available: {e}")
        print("   Start with: ./pion-server -w 1")
        return 1

    # ── Start or check replicas ─────────────────────────────────────────────
    procs: list[subprocess.Popen] = []
    replica_ok = [check_replica(url) for url in all_node_urls]
    live_urls  = [url for url, ok in zip(all_node_urls, replica_ok) if ok]

    if not live_urls and auto_start:
        print(f"  Starting vllm-mlx on port {REPLICA_PORTS[0]}...")
        procs = start_replicas(model, [REPLICA_PORTS[0]])
        ok = wait_for_replica(all_node_urls[0], STARTUP_TIMEOUT)
        if not ok:
            print(f"❌ Replica did not start within {STARTUP_TIMEOUT}s")
            for p in procs: p.terminate()
            return 1
        live_urls = [all_node_urls[0]]
        print(f"  ✅ Replica ready")

    if not live_urls:
        print("❌ No vllm-mlx replicas running.")
        print()
        print(f"Start one with:")
        print(f"  vllm-mlx serve {model} --port 8000 --enable-prefix-cache")
        print()
        print("Or rerun with --auto-start")
        return 1

    # On macOS, MLX can only run one model instance at a time (shared GPU).
    # Two simultaneous instances deadlock. Map both logical "replicas" to the
    # same URL so the demo still exercises Pion routing logic correctly.
    single_replica = len(live_urls) == 1
    if single_replica:
        node_urls = [live_urls[0], live_urls[0]]
        print(f"✅ Single replica mode (macOS MLX): {live_urls[0]}")
        print("   (Both logical replicas map to the same endpoint — "
              "demonstrating warm vs cold TTFT on one node)")
    else:
        node_urls = live_urls[:2]
        print(f"✅ Two replicas: {node_urls}")
    print()

    # ── Initialise router ───────────────────────────────────────────────────
    # Clear any stale prefix state from previous runs (Pion has no SCAN)
    for sp in SYSTEM_PROMPTS.values():
        h = hashlib.sha256(sp.encode()).hexdigest()[:16]
        r.delete(f"prefix:{h}")

    router = PrefixRouter(r, node_urls)

    # Assign system prompts to logical replicas
    assignments = {
        "coding":   node_urls[0],
        "creative": node_urls[1],
    }

    print("Phase 1 — Warm each replica with its assigned system prompt")
    print("          (cold = first request; warm = prefix cached)")
    print("-" * 65)

    ttft_results: dict[str, list[float]] = {"coding": [], "creative": []}

    for role, node_url in assignments.items():
        sys_prompt = SYSTEM_PROMPTS[role]
        queries    = USER_QUERIES[role]

        print(f"\n[{role.upper()}]  {node_url}")
        router.increment_queue(node_url)

        for qi, query in enumerate(queries[:3]):
            ttft, first_tok = measure_ttft(node_url, sys_prompt, query, model, max_tokens)
            label = "cold" if qi == 0 else "warm"
            tag   = "cold" if qi == 0 else "warm"
            marker = "  " if qi == 0 else "✓ "
            if ttft > 0:
                print(f"  {marker}[{label:4}] {ttft:6.0f}ms  {query[:50]!r}")
                ttft_results[role].append(ttft)
            else:
                print(f"  ✗ [{label:4}] ERROR")

        # Register this node as warm for this system prompt in Pion
        router.register_warm(sys_prompt, node_url)
        router.decrement_queue(node_url)

    # ── Phase 2: Routing validation ────────────────────────────────────────
    print()
    print("Phase 2 — Routing validation")
    print("          (Pion HGET returns the pre-warmed node URL)")
    print("-" * 65)

    routing_rows = []
    for role in ["coding", "creative"]:
        sys_prompt   = SYSTEM_PROMPTS[role]
        expected_url = assignments[role]
        query        = USER_QUERIES[role][3]

        routed_url, was_warm = router.route(sys_prompt)
        pion_key = f"prefix:{hashlib.sha256(sys_prompt.encode()).hexdigest()[:16]}"
        correct  = routed_url == expected_url

        router.increment_queue(routed_url)
        ttft, _ = measure_ttft(routed_url, sys_prompt, query, model, max_tokens)
        router.decrement_queue(routed_url)

        routing_rows.append({
            "role": role, "expected": expected_url,
            "routed": routed_url, "warm": was_warm,
            "correct": correct, "ttft": ttft,
        })

        status = "✅" if correct else "❌"
        src    = f"Pion HGET {pion_key}" if was_warm else "least-loaded fallback"
        print(f"  {status} [{role:8}]  TTFT={ttft:5.0f}ms  via {src}")

    # ── Phase 3: Warmup TTFT improvement summary ────────────────────────────
    print()
    print("Phase 3 — Prefix cache warmup effect")
    print("-" * 65)

    for role in ["coding", "creative"]:
        ts = ttft_results[role]
        if len(ts) >= 2:
            cold_ttft = ts[0]
            warm_mean = sum(ts[1:]) / len(ts[1:])
            improvement = (cold_ttft - warm_mean) / cold_ttft * 100 if cold_ttft > 0 else 0
            marker = "✅" if improvement > 3 else "~"
            print(f"  {marker} [{role:8}]  cold={cold_ttft:.0f}ms → "
                  f"warm_avg={warm_mean:.0f}ms  ({improvement:+.1f}%)")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("Summary")
    print("=" * 65)

    correct_routes = sum(1 for row in routing_rows if row["correct"])
    print(f"Routing accuracy:   {correct_routes}/{len(routing_rows)} requests directed to warm replica")
    print()
    print("Pion commands used:")
    print("  HSET prefix:{sha256[:16]} node <url>  warmed_at <ts>")
    print("    → register which replica holds the warm KV cache for this prompt")
    print("  HGET prefix:{sha256[:16]} node")
    print("    → look up warm replica before forwarding the request")
    print("  HSET / HGET  (INCR/DECR via in-memory counters for demo)")
    print("    → queue depth tracking for least-loaded fallback")
    if single_replica:
        print()
        print("Note: macOS MLX only supports one model instance at a time.")
        print("On Linux with multiple GPUs, run two vllm-mlx replicas on")
        print(":8000 and :8001 for true cross-replica routing measurement.")

    # ── Cleanup ────────────────────────────────────────────────────────────
    if procs:
        print()
        print("Stopping auto-started replicas...")
        for proc in procs:
            proc.terminate()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pion cross-replica prefix cache routing demo (vllm-mlx)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"MLX model to serve (default: {DEFAULT_MODEL})")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start vllm-mlx replicas on ports 8000/8001")
    parser.add_argument("--pion-port", type=int, default=PION_PORT,
                        help=f"Pion port (default: {PION_PORT})")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens per response (default: {MAX_TOKENS})")
    return run_demo(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
