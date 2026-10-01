#!/usr/bin/env python3
"""
Session Affinity Demo for Pion.

Starts two fake llama-server stubs on ports 8080 and 8081, then routes
multi-turn chat requests via Pion using HSETNX + EXPIRE for session affinity.

Usage:
    # 1. Start Pion
    ./pion-server -w 1

    # 2. Run demo
    python3 examples/session_affinity_demo.py

Requirements:
    pip install redis requests
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import redis
import requests

PION_PORT = 1974
NODE_PORTS = [8080, 8081]
NODE_URLS = [f"http://127.0.0.1:{p}" for p in NODE_PORTS]
SESSION_TTL_SECONDS = 300


class LlamaStubHandler(BaseHTTPRequestHandler):
    server_version = "FakeLlamaServer/0.1"

    def do_POST(self) -> None:  # noqa: N802 (matches BaseHTTPRequestHandler API)
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""

        port = self.server.server_port
        payload = {
            "id": f"stub-{port}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "fake-llama",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Hello from :{port}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "port": port,
        }
        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        # Silence default request logging to keep demo output clean.
        return


class PionSessionRouter:
    def __init__(self, client: redis.Redis, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self.r = client
        self.ttl_seconds = ttl_seconds

    def assign_session(self, session_id: str, node_url: str) -> bool:
        """Assign session to node if not already present. Returns True if new."""
        key = f"session:{session_id}"
        assigned_at = str(int(time.time()))
        pipe = self.r.pipeline()
        pipe.hsetnx(key, "node", node_url)
        pipe.hsetnx(key, "assigned_at", assigned_at)
        pipe.expire(key, self.ttl_seconds)
        node_set, _ts_set, _ = pipe.execute()
        return bool(node_set)

    def get_session_node(self, session_id: str) -> str | None:
        key = f"session:{session_id}"
        return self.r.hget(key, "node")

    def clear_session(self, session_id: str) -> int:
        key = f"session:{session_id}"
        return int(self.r.delete(key))


def start_stub_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), LlamaStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _get_inflight_count(r: redis.Redis, node_url: str) -> int:
    val = r.get(f"inflight:{node_url}")
    return int(val) if val else 0


def _pick_least_busy_node(r: redis.Redis, node_urls: list[str]) -> str:
    counts = {node: _get_inflight_count(r, node) for node in node_urls}
    min_count = min(counts.values())
    candidates = [node for node, count in counts.items() if count == min_count]

    if not hasattr(_pick_least_busy_node, "_rr_index"):
        _pick_least_busy_node._rr_index = 0  # type: ignore[attr-defined]

    candidates.sort()
    idx = _pick_least_busy_node._rr_index % len(candidates)  # type: ignore[attr-defined]
    _pick_least_busy_node._rr_index += 1  # type: ignore[attr-defined]
    return candidates[idx]


def route_request(
    router: PionSessionRouter,
    session_id: str,
    payload: dict,
    node_urls: list[str],
) -> tuple[str, bool, dict]:
    node_url = router.get_session_node(session_id)
    from_pion = True

    if not node_url:
        from_pion = False
        node_url = _pick_least_busy_node(router.r, node_urls)
        router.assign_session(session_id, node_url)

    inflight_key = f"inflight:{node_url}"
    router.r.incr(inflight_key)
    try:
        resp = requests.post(
            f"{node_url}/v1/chat/completions",
            json=payload,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        router.r.decr(inflight_key)

    return node_url, from_pion, data


def _print_summary(rows: list[dict]) -> None:
    print()
    print("Turn  Session   Node         From Pion?    Response")
    for row in rows:
        print(
            f"{row['turn']:<5}"
            f"{row['session']:<10}"
            f"{row['node']:<13}"
            f"{row['from_pion']:<13}"
            f"{row['response']}"
        )


def main() -> int:
    print("=" * 60)
    print("Pion Session Affinity Demo")
    print("=" * 60)

    r = redis.Redis(host="127.0.0.1", port=PION_PORT, decode_responses=True)
    try:
        r.ping()
    except Exception as exc:
        print(f"ERROR: Pion not available on port {PION_PORT}: {exc}")
        print("Start with: ./pion-server -w 1")
        return 1

    try:
        servers = [start_stub_server(p) for p in NODE_PORTS]
    except OSError as exc:
        print(f"ERROR: Failed to start stub servers: {exc}")
        print("Ensure ports 8080 and 8081 are free.")
        return 1
    time.sleep(0.1)

    router = PionSessionRouter(r)

    payload = {
        "model": "fake-llama",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }

    rows: list[dict] = []

    try:
        # Turn 1: New session
        session_id = "abc123"
        node_url, from_pion, response = route_request(router, session_id, payload, NODE_URLS)
        rkey = f"session:{session_id}"
        print(f"Turn 1 routed to {node_url} (from_pion={from_pion})")
        print(f"Pion HGETALL {rkey} -> {r.hgetall(rkey)}")
        rows.append(
            {
                "turn": 1,
                "session": session_id,
                "node": node_url.replace("http://127.0.0.1", ""),
                "from_pion": "no (new)",
                "response": json.dumps({"port": response.get("port")}),
            }
        )

        # Turn 2: Same session should hit Pion
        node_url2, from_pion2, response2 = route_request(router, session_id, payload, NODE_URLS)
        print(f"Turn 2 routed to {node_url2} (from_pion={from_pion2})")
        rows.append(
            {
                "turn": 2,
                "session": session_id,
                "node": node_url2.replace("http://127.0.0.1", ""),
                "from_pion": "yes (hit)",
                "response": json.dumps({"port": response2.get("port")}),
            }
        )

        # Turn 3: Different session, least busy node
        session_id3 = "xyz789"
        node_url3, from_pion3, response3 = route_request(
            router, session_id3, payload, NODE_URLS
        )
        print(f"Turn 3 routed to {node_url3} (from_pion={from_pion3})")
        rows.append(
            {
                "turn": 3,
                "session": session_id3,
                "node": node_url3.replace("http://127.0.0.1", ""),
                "from_pion": "no (new)",
                "response": json.dumps({"port": response3.get("port")}),
            }
        )

        # TTL expiry simulation: manually clear session
        router.clear_session("abc123")
        print("Cleared session abc123 in Pion (simulating TTL expiry)")

        # Turn 4: Reassign after delete
        node_url4, from_pion4, response4 = route_request(router, "abc123", payload, NODE_URLS)
        print(f"Turn 4 routed to {node_url4} (from_pion={from_pion4})")
        rows.append(
            {
                "turn": 4,
                "session": "abc123",
                "node": node_url4.replace("http://127.0.0.1", ""),
                "from_pion": "no (re-asgn)",
                "response": json.dumps({"port": response4.get("port")}),
            }
        )

        _print_summary(rows)
        return 0
    finally:
        for server in servers:
            server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
