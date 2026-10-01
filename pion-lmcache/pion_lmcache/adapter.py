"""LMCacheRemoteBackend — the blob-store façade.

LMCache's standard remote-tier interface treats storage as opaque
``key -> blob`` map (see ``lmcache.v1.storage_backend.RemoteBackendInterface``).
The native way to plug Pion in is the wire-compat path A6 already provides:
configure ``remote_url: resp://localhost:1974`` and LMCache's stock
``RedisRemoteBackend`` talks to Pion via standard GET/SET/EXISTS/DEL.

This module exists for the cases where:

  - You don't want a TCP/RESP hop (e.g. in-process Python integration).
  - You want explicit control over the Pion connection (timeouts, retry,
    namespace prefixes, Pion-specific stats hooks).
  - LMCache is bundled with another framework that wires its own backend.

The class deliberately mirrors what LMCache calls on a remote backend so it
can be passed where LMCache expects one. It does NOT subclass any LMCache
class — that would force a hard import dependency on a fast-moving upstream.
Where LMCache's exact interface signature differs across versions, this
adapter sticks to the smallest stable subset (`put / get / contains / remove`).
"""
from __future__ import annotations

from typing import Optional

from pion_lmcache.store import _RESPClient, _RESPError


class LMCacheRemoteBackend:
    """Pion-backed LMCache remote-tier adapter (blob API).

    Methods:
      put(key, blob)            -> bool
      get(key)                  -> Optional[bytes]
      contains(key)             -> bool
      remove(key)               -> bool
      ping()                    -> bool
      close()                   -> None

    Optional namespace prefix applied to every key — useful when multiple
    LMCache deployments share one Pion instance.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1974,
        ns_prefix: str = "",
        timeout: float = 60.0,
        client=None,
    ) -> None:
        self.host = host
        self.port = port
        self.ns_prefix = ns_prefix
        self._client = client if client is not None else _RESPClient(host, port, timeout=timeout)

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ── Internal: namespace ─────────────────────────────────────────────

    def _key(self, key) -> bytes:
        if isinstance(key, str):
            kb = key.encode("utf-8")
        elif isinstance(key, (bytes, bytearray, memoryview)):
            kb = bytes(key)
        else:
            kb = str(key).encode("utf-8")
        if self.ns_prefix:
            return self.ns_prefix.encode("utf-8") + b":" + kb
        return kb

    # ── LMCache RemoteBackendInterface (subset) ─────────────────────────

    def put(self, key, blob) -> bool:
        """Store ``blob`` under ``key``. Returns True on success.

        Maps to RESP ``SET key blob`` over Pion's wire-compat path. Pion's
        CLIENT_BUF_SIZE is 64 MB so single-blob KV cache chunks up to several
        MB are fine without chunking.
        """
        if isinstance(blob, (bytes, bytearray, memoryview)):
            value = bytes(blob)
        else:
            value = bytes(blob)
        try:
            resp = self._client.cmd("SET", self._key(key), value)
            # Redis SET returns +OK; if Pion is in some edge config we tolerate
            # an integer 1 as well.
            return resp == b"OK" or resp == 1
        except _RESPError:
            return False

    def get(self, key) -> Optional[bytes]:
        """Fetch the blob; None if not found or remote error."""
        try:
            return self._client.cmd("GET", self._key(key))
        except _RESPError:
            return None

    def contains(self, key) -> bool:
        try:
            r = self._client.cmd("EXISTS", self._key(key))
            return bool(r)
        except _RESPError:
            return False

    def remove(self, key) -> bool:
        """Delete the key. Returns True iff something was removed."""
        try:
            r = self._client.cmd("DEL", self._key(key))
            return bool(r)
        except _RESPError:
            return False

    # Pluralized helpers — LMCache versions sometimes call these.
    def mput(self, items) -> int:
        """Best-effort batch put. Iterates one-by-one (Pion already
        pipelines under the hood when commands ride one connection).
        Returns the number of successful puts."""
        ok = 0
        for k, v in items:
            if self.put(k, v):
                ok += 1
        return ok

    def mget(self, keys):
        return [self.get(k) for k in keys]

    # ── Health ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            r = self._client.cmd("PING")
            return r == b"PONG" or r == "PONG"
        except _RESPError:
            return False
