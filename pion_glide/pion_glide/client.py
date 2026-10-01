"""PionClient — async Pion client built on Valkey GLIDE.

Valkey GLIDE (https://github.com/valkey-io/valkey-glide) is the official
multi-language client with built-in cluster topology discovery, AZ-affinity
routing, and OpenTelemetry tracing.  Pion speaks RESP3, so GLIDE works
out of the box — this thin wrapper adds typed FT.* and AI.* helpers via
GLIDE's ``custom_command()`` escape hatch.

Standalone:
    client = await PionClient.connect("127.0.0.1", 1974)

Cluster (requires ``./pion-server --cluster``):
    client = await PionClient.connect_cluster([("node-a", 1974), ("node-b", 1974)])
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .ft import FTIndex
from .ai import AIGateway

try:
    from glide import (  # valkey-glide package exposes as 'glide'
        GlideClient,
        GlideClientConfiguration,
        GlideClusterClient,
        GlideClusterClientConfiguration,
        NodeAddress,
    )
    _GLIDE_AVAILABLE = True
except ImportError:
    _GLIDE_AVAILABLE = False


class PionClient:
    """Async Pion client wrapping Valkey GLIDE.

    All methods are coroutines — use ``await`` or run inside an async context.

    Parameters
    ----------
    _client:
        Internal GLIDE client (GlideClient or GlideClusterClient).  Obtained
        via the factory class-methods, not by direct construction.
    """

    def __init__(self, _client: Any) -> None:
        self._client = _client
        self.ft = FTIndex(self)
        self.ai = AIGateway(self)

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    async def connect(
        cls,
        host: str = "127.0.0.1",
        port: int = 1974,
    ) -> "PionClient":
        """Connect to a standalone Pion node.

        Parameters
        ----------
        host:   Pion server host (default ``127.0.0.1``)
        port:   Pion server port (default ``1974``)
        """
        if not _GLIDE_AVAILABLE:
            raise ImportError(
                "valkey-glide is required: pip install valkey-glide\n"
                "See https://github.com/valkey-io/valkey-glide"
            )
        config = GlideClientConfiguration(
            addresses=[NodeAddress(host=host, port=port)]
        )
        client = await GlideClient.create(config)
        return cls(client)

    @classmethod
    async def connect_cluster(
        cls,
        addresses: List[Tuple[str, int]],
    ) -> "PionClient":
        """Connect to a Pion cluster.

        Requires each node to be started with ``--cluster --cluster-host <ip>``.
        GLIDE discovers the full topology from the initial CLUSTER NODES response.

        Parameters
        ----------
        addresses:
            List of ``(host, port)`` seed nodes.  One is enough — GLIDE fills
            in the rest via cluster discovery.

        Example
        -------
        ::

            client = await PionClient.connect_cluster([
                ("192.168.1.10", 1974),
                ("192.168.1.11", 1974),
            ])
        """
        if not _GLIDE_AVAILABLE:
            raise ImportError(
                "valkey-glide is required: pip install valkey-glide\n"
                "See https://github.com/valkey-io/valkey-glide"
            )
        config = GlideClusterClientConfiguration(
            addresses=[NodeAddress(host=h, port=p) for h, p in addresses]
        )
        client = await GlideClusterClient.create(config)
        return cls(client)

    # ── Raw command execution ─────────────────────────────────────────────────

    async def execute(self, *args: Any) -> Any:
        """Execute any command via GLIDE's ``custom_command()`` escape hatch.

        Use this for Pion-specific commands not in the standard Redis spec
        (``FT.*``, ``AI.*``, ``CLUSTER KEYSLOT``, …).

        Example
        -------
        ::

            raw = await client.execute("FT.OPTIMIZE", "products")
            info = await client.execute("CLUSTER", "INFO")
        """
        normalized: List[Any] = []
        for a in args:
            if isinstance(a, (bytes, bytearray)):
                normalized.append(a)
            else:
                normalized.append(str(a))
        return await self._client.custom_command(normalized)

    # ── Standard KV commands ──────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        return await self._client.get(key)

    async def set(self, key: str, value: str, **kwargs: Any) -> str:
        return await self._client.set(key, value)

    async def delete(self, *keys: str) -> int:
        return await self._client.delete(list(keys))

    async def exists(self, *keys: str) -> int:
        return await self._client.exists(list(keys))

    async def expire(self, key: str, seconds: int) -> bool:
        return await self._client.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        return await self._client.ttl(key)

    async def incr(self, key: str) -> int:
        return await self._client.incr(key)

    # ── Hash commands ─────────────────────────────────────────────────────────

    async def hset(self, key: str, mapping: Dict[str, Any]) -> int:
        return await self._client.hset(key, mapping)

    async def hget(self, key: str, field: str) -> Optional[str]:
        return await self._client.hget(key, field)

    async def hgetall(self, key: str) -> Dict[str, str]:
        return await self._client.hgetall(key)

    # ── Utility ───────────────────────────────────────────────────────────────

    async def ping(self) -> str:
        return await self._client.ping()

    async def close(self) -> None:
        """Close the GLIDE connection."""
        await self._client.close()

    async def __aenter__(self) -> "PionClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
