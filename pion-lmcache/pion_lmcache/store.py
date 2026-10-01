"""PionStore — structured wrapper over Pion's KV.PREFIX.* / V.STOREBATCH /
V.FETCH RANGE / KV.PREFIX.SAVE / KV.PREFIX.INFO commands.

Why structured (not blob)? LMCache's stock remote tier is opaque blob storage:
the engine pickles a tensor + metadata, hashes it, calls remote.put(key, blob).
That works against Pion via wire-compat (A6) but doesn't get you any of Pion's
real value:

- per-block quantization (turbo4=4-bit, turbo3=3-bit, turbo2=2-bit, fp16, int8)
- prefix sharing — every consumer of the same namespace key gets the same
  cached K/V without re-marshaling
- WAL-durable persistence (`KV.PREFIX.SAVE` compacts; SIGKILL is recovered)
- `V.FETCH RANGE` slices a token range without round-tripping the whole prefix

PionStore exposes Pion's structured API directly so callers can use it as a
KV cache backend instead of a generic byte store.
"""
from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RESP2 client (just what PionStore needs)
# ─────────────────────────────────────────────────────────────────────────────

class _RESPError(RuntimeError):
    pass


class _RESPClient:
    """Tiny RESP2 client. Avoids a hard redis-py dependency at runtime —
    callers who need full features can ignore this and pass an external
    redis.Redis to PionStore.from_redis(...)."""

    def __init__(self, host: str, port: int, sock_buf: int = 64 * 1024 * 1024,
                 timeout: float = 60.0, connect_timeout: float | None = None) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sock_buf)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, sock_buf)
        try:
            self.sock.settimeout(connect_timeout if connect_timeout is not None else timeout)
            self.sock.connect((host, port))
        except OSError:
            self.sock.close()
            raise
        self.sock.settimeout(timeout)
        self.buf = b""

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    @staticmethod
    def _encode(parts: Iterable) -> bytes:
        ps = list(parts)
        out = [f"*{len(ps)}\r\n".encode()]
        for p in ps:
            if isinstance(p, (bytes, bytearray, memoryview)):
                b = bytes(p)
                out.append(f"${len(b)}\r\n".encode())
                out.append(b)
                out.append(b"\r\n")
            else:
                s = str(p).encode()
                out.append(f"${len(s)}\r\n".encode())
                out.append(s)
                out.append(b"\r\n")
        return b"".join(out)

    def _read_line(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("Pion connection closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _read_n(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("Pion connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read_response(self):
        line = self._read_line()
        if not line:
            raise _RESPError("empty response")
        tag, payload = line[:1], line[1:]
        if tag == b"+":
            return payload  # simple string
        if tag == b"-":
            raise _RESPError(payload.decode("utf-8", "replace"))
        if tag == b":":
            return int(payload)
        if tag == b"$":
            n = int(payload)
            if n < 0:
                return None
            data = self._read_n(n)
            self._read_n(2)  # trailing CRLF
            return data
        if tag == b"*":
            n = int(payload)
            if n < 0:
                return None
            return [self._read_response() for _ in range(n)]
        raise _RESPError(f"unknown RESP tag {tag!r}")

    def cmd(self, *parts):
        self.sock.sendall(self._encode(parts))
        return self._read_response()


# ─────────────────────────────────────────────────────────────────────────────
# PionStore
# ─────────────────────────────────────────────────────────────────────────────

VQUANT_VALUES = {"int8", "turbo4", "turbo3", "turbo2", "fp16"}


@dataclass
class PrefixInfo:
    namespace: str
    kv_dim: int
    vquant: str
    layers_seen: int = 0
    tokens_per_layer: dict = field(default_factory=dict)


class PionStore:
    """Structured KV-cache store over Pion.

    All sessions are identified by a *namespace key* — a string the caller
    chooses to bucket prefix sharing. The recommended scheme encodes every
    load-bearing piece of execution context that would invalidate a cache
    hit:  ``model | tokenizer | rope_theta | quant | adapter | prompt_hash``.
    PionStore does not hash for you — give it the exact namespace you want
    sharing under.

    Internally each namespace becomes two V-store sessions:
      ``<ns>_pk`` for the K side, ``<ns>_pv`` for the V side. This mirrors
    the shared-KV-cache design, §13.3.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1974,
        vquant: str = "fp16",
        client=None,
        auto_redirect: bool = True,
        max_redirects: int = 16,
        tenant_prefix: str = "",
    ) -> None:
        if vquant not in VQUANT_VALUES:
            raise ValueError(f"vquant must be one of {VQUANT_VALUES}, got {vquant!r}")
        self.host = host
        self.port = port
        self.vquant = vquant
        # Multi-tenant: every namespace is silently prefixed with this string.
        # Pair with `pion-server --ns-prefix <tenant_prefix>` on the server
        # side to enforce isolation. Empty disables (single-tenant default).
        self.tenant_prefix = tenant_prefix
        self._client = client if client is not None else _RESPClient(host, port)
        # Cross-worker auto-redirect: when a V.STOREBATCH or V.FETCH on the
        # current connection returns "ERR session lives on worker N" (the
        # connection landed on a non-owner worker), reconnect transparently to
        # worker N's affinity port (port + 2 + N), which only worker N
        # accepts on — one redirect, deterministic. Disable by passing False
        # to surface the underlying error verbatim.
        self.auto_redirect = auto_redirect
        self.max_redirects = max_redirects
        self.redirect_count = 0  # diagnostic counter

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ── Prefix lifecycle ──────────────────────────────────────────────────

    def register(self, namespace: str, kv_dim: int, vquant: str | None = None) -> None:
        """Register a new prefix namespace. Idempotent — re-registering the
        same namespace with the same kv_dim/vquant is a no-op (Pion-side
        ``create_session`` returns the existing slot and bumps LRU).

        Side effect: WAL appends two CREATE records (pk + pv). On replay,
        the namespace is reconstituted before any STORE_LAYER records replay.
        """
        q = vquant or self.vquant
        if q not in VQUANT_VALUES:
            raise ValueError(f"vquant must be one of {VQUANT_VALUES}, got {q!r}")
        resp = self._client.cmd("KV.PREFIX.REGISTER", self._ns(namespace), str(kv_dim), q)
        if resp != b"OK":
            raise _RESPError(f"KV.PREFIX.REGISTER unexpected: {resp!r}")

    def lookup(self, namespace: str) -> bool:
        """True iff Pion has both ``<ns>_pk`` and ``<ns>_pv`` sessions live.
        Touches LRU on hit so a hot prefix isn't evicted by other REGISTER calls.
        """
        resp = self._client.cmd("KV.PREFIX.LOOKUP", self._ns(namespace))
        # Simple-string '+HIT' / '+MISS' — payload comes back as bytes.
        if isinstance(resp, (bytes, bytearray)):
            return resp == b"HIT"
        return resp == "HIT"

    # ── Per-layer K/V I/O ─────────────────────────────────────────────────

    def store_layer(
        self,
        namespace: str,
        side: str,
        layer: int,
        token_offset: int,
        tensor_fp32: np.ndarray,
    ) -> None:
        """Store a (num_tokens, kv_dim) FP32 layer slice. ``side`` is "K" or "V".

        Token IDs in [token_offset, token_offset + num_tokens) are written.
        Data is logged to the WAL before quantization, so the store survives
        SIGKILL with bit-equal round-trip after replay.
        """
        sid = self._sid(self._ns(namespace), side)
        if tensor_fp32.dtype != np.float32:
            tensor_fp32 = tensor_fp32.astype(np.float32, copy=False)
        if tensor_fp32.ndim != 2:
            raise ValueError(f"tensor_fp32 must be 2-D (num_tokens, kv_dim); got shape {tensor_fp32.shape}")
        n_tok = tensor_fp32.shape[0]
        if n_tok == 0:
            return
        blob = tensor_fp32.tobytes()
        resp = self._cmd_with_redirect(
            "V.STOREBATCH", sid, str(layer), str(token_offset), str(n_tok), blob,
        )
        if resp != b"OK":
            raise _RESPError(f"V.STOREBATCH unexpected: {resp!r}")

    def fetch_layer(
        self,
        namespace: str,
        side: str,
        layer: int,
        token_start: int,
        token_end: int,
        kv_dim: int,
    ) -> np.ndarray:
        """Fetch tokens [start, end) from a layer slice. Returns FP32 array of
        shape (token_end - token_start, kv_dim). Out-of-range tokens come back
        zero-filled (Pion semantics)."""
        if token_end <= token_start:
            raise ValueError(f"token_end ({token_end}) must exceed token_start ({token_start})")
        sid = self._sid(self._ns(namespace), side)
        blob = self._cmd_with_redirect(
            "V.FETCH", sid, str(layer), "RANGE", str(token_start), str(token_end),
        )
        if blob is None:
            raise _RESPError(f"V.FETCH returned nil for {sid}/{layer}")
        n_tok = token_end - token_start
        expected = n_tok * kv_dim * 4
        if len(blob) != expected:
            raise _RESPError(
                f"V.FETCH returned {len(blob)} bytes; expected {expected} "
                f"(n_tok={n_tok}, kv_dim={kv_dim}, *4 for fp32)"
            )
        return np.frombuffer(blob, dtype=np.float32).reshape(n_tok, kv_dim)

    # ── Multi-layer convenience for per-prefix prefill ────────────────────

    def store_prefix(
        self,
        namespace: str,
        K: np.ndarray,
        V: np.ndarray,
        token_offset: int = 0,
        kv_dim: int | None = None,
        vquant: str | None = None,
    ) -> None:
        """Store every layer of a (num_layers, num_tokens, kv_dim) K/V tensor pair
        for one prefix. Calls ``register`` for you on first use of the namespace."""
        if K.shape != V.shape:
            raise ValueError(f"K {K.shape} and V {V.shape} must have identical shapes")
        if K.ndim != 3:
            raise ValueError(f"K/V must be 3-D (num_layers, num_tokens, kv_dim); got {K.shape}")
        n_layers, n_tok, dim = K.shape
        if kv_dim is None:
            kv_dim = dim
        elif kv_dim != dim:
            raise ValueError(f"kv_dim={kv_dim} doesn't match tensor dim {dim}")
        self.register(namespace, kv_dim, vquant)
        for li in range(n_layers):
            self.store_layer(namespace, "K", li, token_offset, K[li])
            self.store_layer(namespace, "V", li, token_offset, V[li])

    def fetch_prefix(
        self,
        namespace: str,
        n_layers: int,
        n_tokens: int,
        kv_dim: int,
        token_offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of ``store_prefix``. Returns (K, V) each of shape
        (n_layers, n_tokens, kv_dim) as FP32 arrays."""
        K = np.empty((n_layers, n_tokens, kv_dim), dtype=np.float32)
        V = np.empty((n_layers, n_tokens, kv_dim), dtype=np.float32)
        for li in range(n_layers):
            K[li] = self.fetch_layer(namespace, "K", li, token_offset, token_offset + n_tokens, kv_dim)
            V[li] = self.fetch_layer(namespace, "V", li, token_offset, token_offset + n_tokens, kv_dim)
        return K, V

    # ── Persistence + introspection ───────────────────────────────────────

    def save(self) -> bool:
        """KV.PREFIX.SAVE: snapshot to ``pion.vstore.<wid>`` then truncate the
        WAL. Idempotent. Returns True on success; False if Pion has no active
        sessions or rejected the save (e.g., I/O error)."""
        try:
            resp = self._client.cmd("KV.PREFIX.SAVE")
            return resp == b"OK"
        except _RESPError:
            return False

    def info(self) -> dict:
        """Return KV.PREFIX.INFO as a parsed dict."""
        raw = self._client.cmd("KV.PREFIX.INFO")
        if raw is None:
            return {}
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "replace")
        out = {}
        for line in raw.split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.isdigit():
                    out[k] = int(v)
                else:
                    out[k] = v
        return out

    def owner(self, namespace: str) -> int:
        """KV.PREFIX.OWNER — returns the worker_id that physically holds this
        namespace's V buffers, or -1 if no worker owns it (cold cache).

        This is a metadata-only query; it doesn't read any V tensor and never
        triggers a redirect. Useful for ops dashboards or for clients that
        want to pin a connection deterministically before doing a large
        write batch (rather than relying on auto_redirect to find the worker
        on the first STOREBATCH)."""
        resp = self._client.cmd("KV.PREFIX.OWNER", self._ns(namespace))
        # Wire form is +<int>; the simple-string body comes back as bytes.
        if isinstance(resp, (bytes, bytearray)):
            try:
                return int(resp)
            except ValueError:
                return -1
        if isinstance(resp, int):
            return resp
        return -1

    # ── Internals ─────────────────────────────────────────────────────────

    def _ns(self, namespace: str) -> str:
        """Apply the configured tenant_prefix transparently."""
        if self.tenant_prefix:
            return self.tenant_prefix + namespace
        return namespace

    _OWNER_RE = re.compile(r"lives on worker (\d+)")

    def _connect_owner(self, owner: int):
        """A connection on worker `owner`, via its affinity port.

        gh #406: the redirect used to reopen the SHARED port and hope the
        accept race picked the owner. Accept is not uniform across workers,
        so 16 attempts at -w 4 missed 6-15% of the time instead of the ~1%
        a fair race gives. Pion listens on `port + 2 + worker_id` for every
        worker (only that worker accepts there), so the owner is one connect
        away. Falls back to the shared port when the affinity port is not
        reachable (a proxy or firewall exposing only the main port)."""
        try:
            return _RESPClient(self.host, self.port + 2 + owner, connect_timeout=2.0)
        except OSError:
            return _RESPClient(self.host, self.port)

    def _cmd_with_redirect(self, *parts):
        """Wrap self._client.cmd with cross-worker auto-redirect.

        On `-w >1` Pion, V.STOREBATCH/V.FETCH against a non-owner worker
        returns -ERR with "lives on worker N". With auto_redirect on, we
        close the current connection and connect to worker N's affinity port
        (see _connect_owner), so the retry lands on the owner. Subsequent
        operations on the same PionStore instance bypass the redirect path
        entirely (the connection is already on the right worker).
        """
        if not self.auto_redirect:
            return self._client.cmd(*parts)
        for attempt in range(self.max_redirects + 1):
            try:
                return self._client.cmd(*parts)
            except _RESPError as e:
                m = self._OWNER_RE.search(str(e))
                if m is None:
                    raise
                self.redirect_count += 1  # diagnostics
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = self._connect_owner(int(m.group(1)))
        raise _RESPError(
            f"PionStore: redirect retries exhausted ({self.max_redirects}); "
            f"last attempt still on a non-owner worker. "
            f"Check that --kvcache is enabled and the namespace is registered."
        )

    @staticmethod
    def _sid(namespace: str, side: str) -> str:
        s = side.upper()
        if s == "K":
            return f"{namespace}_pk"
        if s == "V":
            return f"{namespace}_pv"
        raise ValueError(f"side must be 'K' or 'V'; got {side!r}")
