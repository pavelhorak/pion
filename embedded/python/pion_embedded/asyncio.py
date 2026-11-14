"""pion_embedded.asyncio.Redis: redis.asyncio.Redis on the engine in this process.

Unlike a client bound to its event loop, the engine holds no loop state, so
clients on different loops (threads) share one engine."""
from __future__ import annotations

import redis.asyncio
from redis.asyncio.connection import AbstractConnection
from redis._parsers import _AsyncRESP2Parser

from ._engine import Engine, EngineReader, EngineWriter, get_engine


class PionAsyncConnection(AbstractConnection):
    def __init__(self, pion_engine: Engine, **kwargs):
        for k in ("host", "port", "path", "socket_keepalive", "socket_keepalive_options",
                  "socket_type", "unix_socket_path"):
            kwargs.pop(k, None)
        kwargs.setdefault("parser_class", _AsyncRESP2Parser)
        self.pion_engine = pion_engine
        super().__init__(**kwargs)

    def repr_pieces(self):
        return [("dir", self.pion_engine.dir), ("db", self.db)]

    async def _connect(self):
        c = self.pion_engine.conn_new()
        self._reader = EngineReader(self.pion_engine, c)
        self._writer = EngineWriter(self.pion_engine, self._reader, c)

    def _host_error(self):
        return "pion:" + self.pion_engine.dir


class Redis(redis.asyncio.Redis):
    def __init__(self, dir: str | None = None, **kwargs):
        from . import split_kwargs
        cfg, kwargs = split_kwargs(kwargs)
        engine = get_engine(dir, cfg)
        single = kwargs.pop("single_connection_client", False)
        given = kwargs.pop("connection_pool", None)
        pool = given or redis.asyncio.ConnectionPool(
            connection_class=PionAsyncConnection, pion_engine=engine, **kwargs)
        super().__init__(connection_pool=pool, single_connection_client=single)
        # A pool this client made is this client's to close, as from_url's is:
        # aclose() otherwise leaves every connection open.
        self.auto_close_connection_pool = given is None
        self.pion_engine = engine


def from_url(url: str, **kwargs) -> Redis:
    from . import _parse_url
    d, kw = _parse_url(url)
    kw.update(kwargs)
    return Redis(dir=d, **kw)
