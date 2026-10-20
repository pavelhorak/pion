"""Pion Attention Client — externalized attention via HNSW-indexed token KV pairs.

Phase 3 of M14. Provides the Python interface for storing and querying
per-layer token KV pairs in Pion's attention index.

Uses a persistent TCP connection (not per-request) to avoid fd exhaustion
and connection overhead at scale (128K+ tokens).
"""

import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class PionAttentionClient:
    """Client for Pion's externalized attention HNSW index.

    Uses a single persistent TCP connection for all operations.
    Reconnects automatically on failure.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1974, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def _connect(self):
        """Establish or re-establish the persistent connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def _ensure_connected(self):
        if self._sock is None:
            self._connect()

    def _send(self, parts: list) -> bytes:
        """Send RESP command over persistent connection and read response."""
        self._ensure_connected()

        header = f"*{len(parts)}\r\n".encode()
        body = b""
        for p in parts:
            if isinstance(p, bytes):
                body += f"${len(p)}\r\n".encode() + p + b"\r\n"
            else:
                s = str(p)
                body += f"${len(s)}\r\n{s}\r\n".encode()

        try:
            self._sock.sendall(header + body)
            return self._read_resp()
        except (ConnectionError, socket.timeout, OSError):
            # Reconnect and retry once
            self._connect()
            self._sock.sendall(header + body)
            return self._read_resp()

    def _read_resp(self) -> bytes:
        """Read a complete RESP response, draining any trailing error noise from binary blobs."""
        buf = self._recv_exact_resp()
        # Drain any trailing "-ERR unknown command" noise from RESP parsing binary blobs.
        # These phantom errors arrive because binary blob bytes look like extra RESP tokens.
        # Set a very short timeout to non-blocking-drain without waiting.
        orig_timeout = self._sock.gettimeout()
        self._sock.settimeout(0.0001)  # 0.1ms — drain queued phantom errors without blocking
        try:
            while True:
                extra = self._sock.recv(1024 * 1024)
                if not extra:
                    break
        except (socket.timeout, BlockingIOError):
            pass
        self._sock.settimeout(orig_timeout)
        return buf

    def _recv_exact_resp(self) -> bytes:
        """Read exactly one complete RESP response."""
        buf = b""
        while True:
            chunk = self._sock.recv(4 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
            if self._is_complete_resp(buf):
                break
        return buf

    def _is_complete_resp(self, data: bytes) -> bool:
        """Check if we have a complete RESP response."""
        if len(data) < 3:
            return False
        prefix = data[0:1]
        if prefix == b"+":
            return b"\r\n" in data
        elif prefix == b"-":
            return b"\r\n" in data
        elif prefix == b":":
            return b"\r\n" in data
        elif prefix == b"$":
            nl = data.find(b"\r\n")
            if nl < 0:
                return False
            length_str = data[1:nl].decode()
            if length_str == "-1":
                return True
            length = int(length_str)
            return len(data) >= nl + 2 + length + 2
        elif prefix == b"*":
            return b"\r\n" in data
        return True

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def create_session(
        self,
        session_id: str,
        key_dim: int = 1024,
        value_dim: int = 1024,
        k_format: str = "int8",
        v_format: str = "int8",
        boundary_layers: int = 0,
        boundary_v_format: str = "int8",
        rope_dim: int = 0,
    ) -> int:
        """Create an attention session. Returns session index or -1.

        Supported formats:
          - "int8"               : 1.0 B/val (default, backward compatible)
          - "turbo4|3|2"         : Block-INT{4,3,2} (val_dim must be multiple of 32)
          - "fp16"               : high-precision, reserved for boundary layers
          - "fp8"                : E4M3 per-block-of-32 (~1.06 B/val) — A2 (gh #30)
          - "bf16_rope_fp8_body" : V4 §2.3.4 hybrid; rope_dim required (gh #39)
        boundary_layers: protect first N + last N layers with boundary_v_format.
        rope_dim: BF16 prefix length when v_format is bf16_rope_fp8_body. Must
            satisfy 0 < rope_dim < value_dim and (value_dim - rope_dim) % 32 == 0.
        """
        parts = ["ATTEND.CREATE", session_id, str(key_dim), str(value_dim)]
        # Only emit new args if they differ from defaults, to stay compatible with
        # older Pion servers that don't know these keywords (they'd ignore them anyway,
        # but minimizing the over-the-wire diff keeps logs clean).
        if k_format != "int8":
            parts += ["KQUANT", k_format]
        if v_format != "int8":
            parts += ["VQUANT", v_format]
        if boundary_layers > 0:
            parts += ["BOUNDARY", str(boundary_layers)]
            if boundary_v_format != "int8":
                parts += ["BOUNDARY_VQUANT", boundary_v_format]
        if rope_dim > 0:
            parts += ["ROPE", str(rope_dim)]
        resp = self._send(parts)
        if resp.startswith(b":"):
            nl = resp.find(b"\r\n")
            return int(resp[1:nl])
        return -1

    def store_tokens(
        self,
        session_id: str,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> bool:
        """Store token KV pairs for a layer. Keys and values are sent as-is.

        The server calibrates its key quantizer when the layer is finalized
        (gh #391). This client used to rescale every call's keys, and every
        query, by that call's own max-abs to fit a fixed [-0.2, 0.2] range, so
        a stored key and the same key sent as a query landed at different
        scales and a key found itself 4 times in 10.
        """
        assert keys.dtype == np.float32
        assert values.dtype == np.float32
        num_tokens = keys.shape[0]
        resp = self._send([
            "ATTEND.STORE", session_id, str(layer_id), str(num_tokens),
            np.ascontiguousarray(keys).tobytes(), np.ascontiguousarray(values).tobytes()
        ])
        return b"+OK" in resp

    def finalize_layer(self, session_id: str, layer_id: int) -> bool:
        """Compact the HNSW index after all tokens for a layer are stored.
        Must be called before querying. Triggers BFS reorder + INT8 quantization."""
        resp = self._send(["ATTEND.FINALIZE", session_id, str(layer_id)])
        return b"+OK" in resp

    def query_topk(
        self,
        session_id: str,
        layer_id: int,
        query: np.ndarray,
        k: int = 128,
    ) -> Optional[bytes]:
        """Query for top-k most relevant token values (by L2 to the stored
        keys). The query is sent as-is, like the keys.

        Returns the raw reply: ``min(k, tokens stored) * value_dim`` FP32s,
        one value row per result, best match first (gh #404 — before, the
        server sent the first row only, whatever k was)."""
        assert query.dtype == np.float32
        resp = self._send([
            "ATTEND.QUERY", session_id, str(layer_id), str(k),
            np.ascontiguousarray(query.reshape(-1)).tobytes()
        ])
        if resp.startswith(b"$") and not resp.startswith(b"$-1"):
            nl = resp.find(b"\r\n")
            if nl > 0:
                blob_len = int(resp[1:nl])
                return resp[nl + 2:nl + 2 + blob_len]
        return None

    def info(self) -> dict:
        """Get attention index statistics."""
        resp = self._send(["ATTEND.INFO"])
        result = {}
        if resp.startswith(b"$"):
            nl = resp.find(b"\r\n")
            if nl > 0:
                length = int(resp[1:nl])
                body = resp[nl + 2:nl + 2 + length].decode()
                for line in body.strip().split("\r\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        try:
                            result[key] = int(val)
                        except ValueError:
                            result[key] = val
        return result

    # --- MLX GPU-accelerated batched attention ---

    def query_batch_gpu(
        self,
        H: int,
        N: int,
        D: int,
        top_k: int,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> Optional[np.ndarray]:
        """GPU-accelerated batched multi-head sparse attention via MLX sidecar.

        Sends Q+K+V in one request. Use for one-shot queries where K/V aren't
        reused. For repeated queries on the same K/V, use store_kv_gpu +
        query_cached_gpu instead.

        Args:
            H: number of heads
            N: sequence length (tokens)
            D: head dimension
            top_k: number of top-scoring tokens to attend to
            Q: [H, D] or [H, 1, D] float32 query vectors
            K: [H, N, D] float32 key matrix
            V: [H, N, D] float32 value matrix

        Returns: [H, D] float32 output, or None on error.
        Requires: server started with --mlx-attention
        """
        Q_flat = Q.reshape(H, D).astype(np.float32)
        K_flat = K.reshape(H, N, D).astype(np.float32)
        V_flat = V.reshape(H, N, D).astype(np.float32)
        resp = self._send([
            "ATTEND.QUERYBATCH",
            str(H), str(N), str(D), str(top_k),
            Q_flat.tobytes(), K_flat.tobytes(), V_flat.tobytes(),
        ])
        if resp.startswith(b"$") and not resp.startswith(b"$-1"):
            nl = resp.find(b"\r\n")
            if nl > 0:
                blob_len = int(resp[1:nl])
                data = resp[nl + 2:nl + 2 + blob_len]
                return np.frombuffer(data, dtype=np.float32).reshape(H, D)
        return None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args):
        self.close()
