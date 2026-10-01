"""PionRouterClient — RESP client for M13 AI.ROUTE.* semantic routing commands.

Wraps Pion's semantic load balancer for GPU fleet routing:
  - AI.ROUTE.REGISTER: register GPU node with semantic centroid
  - AI.ROUTE.UPDATE: update node centroid (e.g., as KV cache evolves)
  - AI.ROUTE: route query to best-matching GPU node
  - AI.ROUTE.REMOVE: remove node from routing table
  - AI.ROUTE.INFO: get routing statistics

Requires Pion server with --kvcache flag.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RouteResult:
    """Result of a routing decision."""
    hit: bool
    endpoint: str = ""
    node_id: str = ""  # populated by FleetManager, not by raw client


@dataclass
class NodeInfo:
    """Per-node statistics from AI.ROUTE.INFO."""
    node_id: str
    endpoint: str
    routed: int
    capacity: int


@dataclass
class RouterInfo:
    """Aggregate routing statistics."""
    node_count: int
    total_queries: int
    total_hits: int
    hit_rate: float
    dimensions: int
    nodes: list[NodeInfo]


class PionRouterClient:
    """RESP client for Pion's M13 semantic router."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1974, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self):
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _ensure_connected(self):
        if self._sock is None:
            self.connect()

    def _send_command(self, *args: bytes) -> bytes:
        self._ensure_connected()
        header = f"*{len(args)}\r\n".encode()
        body = b""
        for arg in args:
            body += f"${len(arg)}\r\n".encode() + arg + b"\r\n"
        self._sock.sendall(header + body)
        return self._read_resp()

    def _read_resp(self) -> bytes:
        buf = b""
        while True:
            chunk = self._sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            if self._is_complete(buf):
                break
        return buf

    def _is_complete(self, data: bytes) -> bool:
        if len(data) < 3:
            return False
        p = data[0:1]
        if p in (b"+", b"-", b":"):
            return b"\r\n" in data
        if p == b"$":
            nl = data.find(b"\r\n")
            if nl < 0:
                return False
            length_str = data[1:nl].decode()
            if length_str == "-1":
                return True
            length = int(length_str)
            return len(data) >= nl + 2 + length + 2
        return True

    def _parse_simple(self, resp: bytes) -> str:
        """Parse +OK or -ERR response."""
        if resp.startswith(b"+"):
            return resp[1:resp.find(b"\r\n")].decode()
        if resp.startswith(b"-"):
            raise RuntimeError(resp[1:resp.find(b"\r\n")].decode())
        return resp.decode(errors="replace")

    def _parse_bulk(self, resp: bytes) -> Optional[bytes]:
        """Parse bulk string or null."""
        if resp.startswith(b"$-1"):
            return None
        if resp.startswith(b"$"):
            nl = resp.find(b"\r\n")
            length = int(resp[1:nl].decode())
            return resp[nl + 2:nl + 2 + length]
        return None

    # ── Commands ─────────────────────────────────────────────────────────

    def register(
        self,
        node_id: str,
        endpoint: str,
        centroid: np.ndarray,
        capacity: int = 0,
    ) -> bool:
        """Register a GPU node with its semantic centroid.

        Args:
            node_id: Unique node identifier (e.g., "gpu-0").
            endpoint: URL to forward routed requests to.
            centroid: FP32 embedding representing this node's cached content.
            capacity: Max concurrent requests (0 = unlimited).
        """
        assert centroid.dtype == np.float32, f"Centroid must be float32, got {centroid.dtype}"

        args = [
            b"AI.ROUTE.REGISTER",
            node_id.encode(),
            endpoint.encode(),
            centroid.tobytes(),
        ]
        if capacity > 0:
            args.extend([b"CAPACITY", str(capacity).encode()])

        resp = self._send_command(*args)
        self._parse_simple(resp)  # raises on error
        return True

    def update(self, node_id: str, centroid: np.ndarray) -> bool:
        """Update a node's semantic centroid.

        Call when the node's KV cache state changes significantly.
        """
        assert centroid.dtype == np.float32
        resp = self._send_command(
            b"AI.ROUTE.UPDATE",
            node_id.encode(),
            centroid.tobytes(),
        )
        self._parse_simple(resp)
        return True

    def route(
        self,
        query_embedding: np.ndarray,
        exclude: str = "",
    ) -> RouteResult:
        """Route a query to the best-matching GPU node.

        Args:
            query_embedding: FP32 prompt embedding.
            exclude: Node ID to skip (e.g., current node).

        Returns:
            RouteResult with hit=True and endpoint on success.
        """
        assert query_embedding.dtype == np.float32

        args = [b"AI.ROUTE", query_embedding.tobytes()]
        if exclude:
            args.extend([b"EXCLUDE", exclude.encode()])

        resp = self._send_command(*args)
        bulk = self._parse_bulk(resp)

        if bulk is not None:
            return RouteResult(hit=True, endpoint=bulk.decode())
        return RouteResult(hit=False)

    def remove(self, node_id: str) -> bool:
        """Remove a node from the routing table."""
        resp = self._send_command(b"AI.ROUTE.REMOVE", node_id.encode())
        self._parse_simple(resp)
        return True

    def info(self) -> RouterInfo:
        """Get routing statistics."""
        resp = self._send_command(b"AI.ROUTE.INFO")
        bulk = self._parse_bulk(resp)

        if bulk is None:
            return RouterInfo(0, 0, 0, 0.0, 0, [])

        text = bulk.decode()
        result = RouterInfo(0, 0, 0, 0.0, 768, [])

        for line in text.strip().split("\r\n"):
            if not line:
                continue
            if line.startswith("nodes:"):
                result.node_count = int(line.split(":", 1)[1])
            elif line.startswith("total_queries:"):
                result.total_queries = int(line.split(":", 1)[1])
            elif line.startswith("total_hits:"):
                result.total_hits = int(line.split(":", 1)[1])
            elif line.startswith("hit_rate:"):
                result.hit_rate = float(line.split(":", 1)[1])
            elif line.startswith("dimensions:"):
                result.dimensions = int(line.split(":", 1)[1])
            elif line.startswith("node:"):
                # node:<id>:endpoint=<ep>,routed=<n>,capacity=<c>
                parts = line[5:]  # after "node:"
                colon = parts.find(":")
                if colon > 0:
                    nid = parts[:colon]
                    kvs = {}
                    for kv in parts[colon + 1:].split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            kvs[k] = v
                    result.nodes.append(NodeInfo(
                        node_id=nid,
                        endpoint=kvs.get("endpoint", ""),
                        routed=int(kvs.get("routed", 0)),
                        capacity=int(kvs.get("capacity", 0)),
                    ))

        return result

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
