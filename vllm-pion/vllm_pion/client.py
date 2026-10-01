"""Pion binary client for KV cache operations.

Communicates with Pion's KV.STORE / KV.FETCH / KV.INFO commands via RESP protocol.
Uses raw binary embedding/blob transfer (RESP bulk strings).
"""

import socket
import struct
import numpy as np
from typing import Optional, Tuple


class PionKVClient:
    """RESP client for Pion KV cache store."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1974, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self):
        """Establish TCP connection to Pion."""
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def close(self):
        """Close the connection."""
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
        """Send a RESP command with binary arguments and read response."""
        self._ensure_connected()

        # Build RESP array
        header = f"*{len(args)}\r\n".encode()
        body = b""
        for arg in args:
            body += f"${len(arg)}\r\n".encode() + arg + b"\r\n"

        self._sock.sendall(header + body)

        # Read response (up to 64 MB for large blobs)
        return self._read_resp_response()

    def _read_resp_response(self) -> bytes:
        """Read a complete RESP response."""
        buf = b""
        while True:
            chunk = self._sock.recv(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buf += chunk
            # Check if we have a complete response
            if self._is_complete_resp(buf):
                break
        return buf

    def _is_complete_resp(self, data: bytes) -> bool:
        """Check if we have a complete RESP response."""
        if len(data) < 3:
            return False
        prefix = data[0:1]
        if prefix == b"+":  # Simple string
            return b"\r\n" in data
        elif prefix == b"-":  # Error
            return b"\r\n" in data
        elif prefix == b":":  # Integer
            return b"\r\n" in data
        elif prefix == b"$":  # Bulk string
            newline_pos = data.find(b"\r\n")
            if newline_pos < 0:
                return False
            length_str = data[1:newline_pos].decode()
            if length_str == "-1":
                return True  # Null bulk string
            length = int(length_str)
            expected_total = newline_pos + 2 + length + 2
            return len(data) >= expected_total
        return True  # Unknown — treat as complete

    def kv_store(
        self,
        cache_id: str,
        embedding: np.ndarray,
        blob: bytes,
        ttl: int = 0,
        model: str = "",
    ) -> bool:
        """Store a KV cache blob with its prompt embedding.

        Args:
            cache_id: Unique identifier for this cache entry.
            embedding: FP32 numpy array (prompt embedding, e.g., 768d).
            blob: Raw KV cache tensor bytes.
            ttl: Time-to-live in seconds (0 = no expiry).
            model: Model family tag (e.g., "llama-3-8b").

        Returns:
            True if stored successfully.
        """
        assert embedding.dtype == np.float32, f"Embedding must be float32, got {embedding.dtype}"

        args = [
            b"KV.STORE",
            cache_id.encode(),
            embedding.tobytes(),
            blob,
        ]

        if ttl > 0:
            args.extend([b"TTL", str(ttl).encode()])
        if model:
            args.extend([b"MODEL", model.encode()])

        resp = self._send_command(*args)
        return b"+OK" in resp

    def kv_fetch(
        self,
        embedding: np.ndarray,
        threshold: float = 0.0,
        model: str = "",
    ) -> Optional[bytes]:
        """Fetch the closest cached KV blob by embedding similarity.

        Args:
            embedding: FP32 numpy array (query prompt embedding).
            threshold: Cosine similarity threshold (0 = use server default 0.95).
            model: Filter by model family tag.

        Returns:
            Raw KV cache tensor bytes on hit, None on miss.
        """
        assert embedding.dtype == np.float32

        args = [b"KV.FETCH", embedding.tobytes()]

        if threshold > 0:
            args.extend([b"THRESHOLD", f"{threshold:.4f}".encode()])
        if model:
            args.extend([b"MODEL", model.encode()])

        resp = self._send_command(*args)

        # Parse bulk string response
        if resp.startswith(b"$-1"):
            return None  # Cache miss

        if resp.startswith(b"$"):
            newline_pos = resp.find(b"\r\n")
            if newline_pos < 0:
                return None
            length = int(resp[1:newline_pos].decode())
            blob_start = newline_pos + 2
            blob_end = blob_start + length
            if blob_end <= len(resp):
                return resp[blob_start:blob_end]

        return None

    def kv_info(self) -> dict:
        """Get KV cache store statistics.

        Returns:
            Dictionary with keys: entries, total_blob_bytes, hits, misses,
            hit_rate, dimensions, enabled, capacity.
        """
        resp = self._send_command(b"KV.INFO")

        result = {}
        if resp.startswith(b"$"):
            newline_pos = resp.find(b"\r\n")
            if newline_pos < 0:
                return result
            length = int(resp[1:newline_pos].decode())
            body = resp[newline_pos + 2:newline_pos + 2 + length].decode()
            for line in body.strip().split("\r\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    try:
                        if "." in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value
        return result

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
