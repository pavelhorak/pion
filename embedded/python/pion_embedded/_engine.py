"""The ctypes binding to libpion (src/lib/pion_embed.mojo) and the
socket-like objects redis-py's connections talk to.

One engine per process. Every call releases the GIL (ctypes does) and takes
the engine's own mutex, so threads may share it.
"""
from __future__ import annotations

import atexit
import ctypes
import errno
import os
import socket
import sys
import threading
import time

DONE, PARKED, PARTIAL, CLOSED = 0, 1, 2, 3
ERRORS = {-1: "engine error", -2: "the engine is closed", -3: "request too large",
          -4: "bad connection", -5: "a forked child cannot use its parent's engine"}

_lock = threading.Lock()
_engine: "Engine | None" = None


class PionError(RuntimeError):
    pass


def _find_lib() -> str:
    env = os.environ.get("PION_EMBED_LIB")
    if env:
        return env
    name = "libpion.dylib" if sys.platform == "darwin" else "libpion.so"
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "_lib", name),
                 os.path.join(here, "..", "..", "..", name)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise PionError(f"{name} not found: set PION_EMBED_LIB, or build it (pixi run build-embed)")


class Engine:
    """An open data directory."""

    def __init__(self, dir: str, config: dict[str, str]):
        self.dir = os.path.abspath(dir)
        os.makedirs(self.dir, exist_ok=True)
        cfg = {"dir": self.dir, **config}
        text = "".join(f"{k} {v}\n" for k, v in cfg.items()).encode()
        path = _find_lib()
        self.binding = "ctypes"
        ext = None
        if os.environ.get("PION_EMBED_CTYPES") != "1":
            try:
                from . import _pion as ext          # the stable-ABI extension
            except ImportError:
                ext = None
        if ext is not None:
            ext.load(path)                          # RTLD_GLOBAL, like the ctypes path
            h, err = ext.open(text)
            if not h:
                raise PionError(err)
            self.binding = "abi3"
            self.lib = None
            self.h = h
            self.exec = lambda c, data, _e=ext.exec, _h=h: _e(_h, c, data)
            self.poll = lambda c, _p=ext.poll, _h=h: _p(_h, c)
            self._ext = ext
        else:
            # RTLD_GLOBAL: lua_wrap.c finds pion_script_dispatch with dlsym(RTLD_DEFAULT)
            self.lib = lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            I64, P = ctypes.c_int64, ctypes.c_void_p
            lib.pion_embed_open.restype = I64
            lib.pion_embed_open.argtypes = [ctypes.c_char_p, I64, ctypes.c_char_p, I64]
            lib.pion_embed_conn_new.restype = I64
            lib.pion_embed_conn_new.argtypes = [I64]
            lib.pion_embed_conn_free.restype = None
            lib.pion_embed_conn_free.argtypes = [I64, I64]
            lib.pion_embed_exec.restype = I64
            lib.pion_embed_exec.argtypes = [I64, I64, ctypes.c_char_p, I64, ctypes.POINTER(P), ctypes.POINTER(I64)]
            lib.pion_embed_poll.restype = I64
            lib.pion_embed_poll.argtypes = [I64, I64, ctypes.POINTER(P), ctypes.POINTER(I64)]
            lib.pion_embed_tick.restype = I64
            lib.pion_embed_tick.argtypes = [I64]
            lib.pion_embed_close.restype = I64
            lib.pion_embed_close.argtypes = [I64]
            lib.pion_embed_version.restype = I64
            lib.pion_embed_version.argtypes = [ctypes.c_char_p, I64]
            err = ctypes.create_string_buffer(1024)
            h = lib.pion_embed_open(text, len(text), err, 1024)
            if not h:
                raise PionError(err.value.decode(errors="replace"))
            self.h = h
            self._exec, self._poll = lib.pion_embed_exec, lib.pion_embed_poll
            self._string_at = ctypes.string_at
            self.exec, self.poll = self._exec_ctypes, self._poll_ctypes
        self.config = cfg
        self.closed = False
        self.pid = os.getpid()
        self._ticker = threading.Thread(target=self._tick_loop, name="pion-tick", daemon=True)
        self._ticker.start()
        atexit.register(self.close)

    def _tick_loop(self):
        tick = self._ext.tick if self.lib is None else self.lib.pion_embed_tick
        while not self.closed:
            if tick(self.h) < 0:
                return
            time.sleep(0.005)

    def version(self) -> str:
        if self.lib is None:
            return self._ext.version()
        b = ctypes.create_string_buffer(128)
        self.lib.pion_embed_version(b, 128)
        return b.value.decode()

    def conn_new(self) -> int:
        c = self._ext.conn_new(self.h) if self.lib is None else self.lib.pion_embed_conn_new(self.h)
        if c < 0:
            raise PionError(ERRORS.get(c, f"error {c}"))
        return c

    def conn_free(self, c: int):
        if not self.closed and os.getpid() == self.pid:
            if self.lib is None:
                self._ext.conn_free(self.h, c)
            else:
                self.lib.pion_embed_conn_free(self.h, c)

    def _exec_ctypes(self, c: int, data: bytes):
        p, n = ctypes.c_void_p(), ctypes.c_int64()
        st = self._exec(self.h, c, data, len(data), ctypes.byref(p), ctypes.byref(n))
        return st, (self._string_at(p.value, n.value) if n.value else b"")

    def _poll_ctypes(self, c: int):
        p, n = ctypes.c_void_p(), ctypes.c_int64()
        st = self._poll(self.h, c, ctypes.byref(p), ctypes.byref(n))
        return st, (self._string_at(p.value, n.value) if n.value else b"")

    def close(self):
        if self.closed or os.getpid() != self.pid:
            return
        self.closed = True
        if self.lib is None:
            self._ext.close(self.h)
        else:
            self.lib.pion_embed_close(self.h)


def get_engine(dir: str | None, config: dict[str, str]) -> Engine:
    """The process's engine. A second directory is refused: one engine per
    process: the engine's C state is process-global."""
    global _engine
    with _lock:
        if _engine is not None:
            if dir is not None and os.path.abspath(dir) != _engine.dir:
                raise PionError(f"this process already has {_engine.dir} open; one data directory per process")
            if _engine.closed:
                raise PionError("the engine was closed; one engine per process")
            return _engine
        if dir is None:
            raise PionError("pass dir=: the data directory")
        _engine = Engine(dir, config)
        return _engine


def close(dir: str | None = None):
    """Flush the log to disk and release the directory."""
    if _engine is not None:
        _engine.close()


_never_readable = os.pipe()[0]


def _status_error(st: int):
    raise ConnectionError(ERRORS.get(st, f"engine status {st}"))


class EngineSocket:
    """What redis-py's sync Connection reads and writes. A request runs
    inside sendall(); its reply waits here for recv()."""

    def __init__(self, engine: Engine, timeout):
        self._e = engine
        self._c = engine.conn_new()
        self._buf = bytearray()
        self._timeout = timeout
        self._eof = False

    # socket API redis-py uses
    def settimeout(self, t):
        self._timeout = t

    def gettimeout(self):
        return self._timeout

    def setblocking(self, flag):
        self._timeout = None if flag else 0.0

    def setsockopt(self, *a):
        pass

    def fileno(self):
        return _never_readable

    def getpeername(self):
        return ("pion", 0)

    getsockname = getpeername

    def shutdown(self, how):
        pass

    def close(self):
        if self._c is not None:
            self._e.conn_free(self._c)
            self._c = None

    def sendall(self, data):
        if self._c is None:
            raise OSError(errno.EBADF, "closed")
        st, out = self._e.exec(self._c, bytes(data))
        if out:
            self._buf += out
        if st == CLOSED:
            self._eof = True
        elif st < 0:
            _status_error(st)

    def _fill(self):
        if self._buf or self._eof:
            return
        t = self._timeout
        deadline = None if t is None else time.monotonic() + t
        delay = 0.0001
        while True:
            st, out = self._e.poll(self._c)
            if out:
                self._buf += out
                return
            if st == CLOSED:
                self._eof = True
                return
            if st < 0:
                _status_error(st)
            if t == 0:
                raise BlockingIOError(errno.EWOULDBLOCK, "no reply yet")
            if deadline is not None and time.monotonic() >= deadline:
                raise socket.timeout("timed out")
            time.sleep(delay)
            delay = min(delay * 2, 0.002)

    def recv(self, n):
        self._fill()
        d = bytes(self._buf[:n])
        del self._buf[:n]
        return d

    def recv_into(self, buf, nbytes=0):
        self._fill()
        n = min(len(self._buf), nbytes or len(buf))
        buf[:n] = self._buf[:n]
        del self._buf[:n]
        return n


try:
    import asyncio

    class EngineReader(asyncio.StreamReader):
        """redis-py's async parsers read an asyncio.StreamReader. When one
        would wait for data, this one polls the engine instead of a socket."""

        def __init__(self, engine: Engine, c: int):
            super().__init__()
            self._e = engine
            self._pc = c

        async def _wait_for_data(self, func_name):
            delay = 0.0001
            while not self._buffer and not self._eof:
                st, out = self._e.poll(self._pc)
                if out:
                    self.feed_data(out)
                    return
                if st == CLOSED:
                    self.feed_eof()
                    return
                if st < 0:
                    _status_error(st)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.002)

    class EngineWriter:
        def __init__(self, engine: Engine, reader: EngineReader, c: int):
            self._e = engine
            self._r = reader
            self._c = c

        def write(self, data):
            st, out = self._e.exec(self._c, bytes(data))
            if out:
                self._r.feed_data(out)
            if st == CLOSED:
                self._r.feed_eof()
            elif st < 0:
                _status_error(st)

        def writelines(self, chunks):
            self.write(b"".join(chunks))

        async def drain(self):
            pass

        def close(self):
            if self._c is not None:
                self._e.conn_free(self._c)
                self._c = None

        def is_closing(self):
            return self._c is None

        async def wait_closed(self):
            pass

        def get_extra_info(self, name, default=None):
            return default
except ImportError:  # pragma: no cover
    pass
