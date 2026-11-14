"""Pion without the server: the engine in this process, behind redis-py.

    import pion_embedded as pion
    r = pion.Redis(dir="./data")          # a redis.Redis: every redis-py method works
    ar = pion.asyncio.Redis(dir="./data") # a redis.asyncio.Redis
    r = pion.from_url("pion:///abs/path/to/data")

Writes go to the server's write-ahead log in `dir` as each command runs, so an
acknowledged write survives the process being killed. One engine and one data
directory per process; a second process opening the directory is refused (run
`pion-server` on it to share it). KV only: no vector search.

Working name: `pion_embedded`. The PyPI name is undecided.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import redis
from redis._parsers import _RESP2Parser
from redis.connection import AbstractConnection

from ._engine import Engine, PionError, close, get_engine, EngineSocket
from . import asyncio  # noqa: F401  (pion_embedded.asyncio.Redis)

__all__ = ["Redis", "from_url", "close", "PionError", "PionConnection"]

_ENGINE_KEYS = ("appendonly", "wal_size_mb", "maxmemory", "maxmemory_policy")


def engine_config(appendonly=True, wal_size_mb=None, maxmemory=None, maxmemory_policy=None) -> dict:
    cfg = {"appendonly": "yes" if appendonly else "no"}
    if wal_size_mb is not None:
        cfg["wal-size"] = str(int(wal_size_mb))
    if maxmemory is not None:
        cfg["maxmemory"] = str(int(maxmemory))
    if maxmemory_policy is not None:
        cfg["maxmemory-policy"] = str(maxmemory_policy)
    return cfg


def split_kwargs(kwargs: dict):
    eng = {k: kwargs.pop(k) for k in _ENGINE_KEYS if k in kwargs}
    return engine_config(**eng), kwargs


class PionConnection(AbstractConnection):
    """A redis-py connection to the engine in this process."""

    def __init__(self, pion_engine: Engine, **kwargs):
        for k in ("host", "port", "path", "socket_keepalive", "socket_keepalive_options",
                  "socket_type", "unix_socket_path"):
            kwargs.pop(k, None)
        kwargs.setdefault("parser_class", _RESP2Parser)   # RESP3 when protocol=3 (redis-py swaps it)
        self.pion_engine = pion_engine
        super().__init__(**kwargs)

    def repr_pieces(self):
        return [("dir", self.pion_engine.dir), ("db", self.db)]

    def _connect(self):
        return EngineSocket(self.pion_engine, self.socket_timeout)

    def _host_error(self):
        return "pion:" + self.pion_engine.dir


_REDIS_ONLY = ("single_connection_client",)


class Redis(redis.Redis):
    """redis.Redis on the engine in this process."""

    def __init__(self, dir: str | None = None, **kwargs):
        cfg, kwargs = split_kwargs(kwargs)
        engine = get_engine(dir, cfg)
        top = {k: kwargs.pop(k) for k in _REDIS_ONLY if k in kwargs}
        given = kwargs.pop("connection_pool", None)
        pool = given or redis.ConnectionPool(
            connection_class=PionConnection, pion_engine=engine, **kwargs)
        super().__init__(connection_pool=pool, **top)
        self.auto_close_connection_pool = given is None   # close() closes the pool it made
        self.pion_engine = engine

    def pion_info(self) -> dict:
        e = self.pion_engine
        return {"version": e.version(), "dir": e.dir, "appendonly": e.config.get("appendonly")}


def _parse_url(url: str):
    u = urlparse(url)
    if u.scheme != "pion":
        raise ValueError("a pion:// URL: pion:///abs/path/to/data")
    q = {k: v[-1] for k, v in parse_qs(u.query).items()}
    kw = {}
    if "appendonly" in q:
        kw["appendonly"] = q.pop("appendonly") not in ("no", "0", "false")
    for k in ("wal_size_mb", "maxmemory"):
        if k in q:
            kw[k] = int(q.pop(k))
    if "maxmemory_policy" in q:
        kw["maxmemory_policy"] = q.pop("maxmemory_policy")
    if "protocol" in q:
        kw["protocol"] = int(q.pop("protocol"))
    if "decode_responses" in q:
        kw["decode_responses"] = q.pop("decode_responses") in ("1", "yes", "true")
    return (u.netloc + u.path) or None, kw


def from_url(url: str, **kwargs) -> Redis:
    d, kw = _parse_url(url)
    kw.update(kwargs)
    return Redis(dir=d, **kw)
