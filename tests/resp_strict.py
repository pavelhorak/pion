"""Strict RESP2/RESP3 client for tests — one command in, exactly one reply out.

WHY
Test helpers used to do `sock.recv(65536)` once, with a timeout that was
swallowed. Three failure modes followed, all of which pass a buggy server:

  * a slow reply (a server replaying its WAL after a restart) timed out, the
    helper returned "" and moved on, and the late reply was then read as the
    answer to the NEXT command — every later check compared the wrong reply;
  * a reply split across TCP segments was truncated at the first segment;
  * a command that emitted TWO replies (a framing desync) looked fine, because
    the extra reply was either read as part of this one or as the next one.

This reader parses the frame, so it knows where a reply ends. It raises on a
timeout, a malformed frame, or a closed socket; it never returns "".

    c = Conn(port)
    c.cmd("SET", "k", "v")          -> "OK"          (simple string -> str)
    c.cmd("GET", "k")               -> b"v"          (bulk -> bytes, nil -> None)
    c.cmd("LPUSH", "x", "a")        -> 1             (integer -> int)
    c.cmd("GET", "x")               -> RespError("WRONGTYPE ...")  (returned, not raised)
    c.raw("GET", "k")               -> b"$1\\r\\nv\\r\\n"  (the exact bytes of ONE reply)
    c.pipeline([...])               -> [reply, ...]  (exactly one per command)
    c.assert_in_sync()              -> PING must be the very next reply, nothing extra
"""
from __future__ import annotations

import socket
import time


class RespError(str):
    """An error reply (`-ERR ...`). A str subclass so `"WRONGTYPE" in e` works,
    but distinguishable: `isinstance(r, RespError)`."""


class RespProtocolError(Exception):
    """The server sent something that is not one well-formed RESP reply."""


def encode(args) -> bytes:
    out = [b"*%d\r\n" % len(args)]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        elif isinstance(a, (int, float)):
            a = repr(a).encode()
        out.append(b"$%d\r\n%s\r\n" % (len(a), a))
    return b"".join(out)


class Conn:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 10.0,
                 connect_timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        self.sock.settimeout(timeout)
        self.buf = b""
        self.timeout = timeout

    @classmethod
    def wrap(cls, sock: socket.socket, timeout: float = 10.0) -> "Conn":
        """A reader over a socket the test already owns. Keep the returned
        object for the socket's lifetime: it holds bytes read past a reply."""
        c = cls.__new__(cls)
        c.sock, c.buf, c.timeout = sock, b"", timeout
        sock.settimeout(timeout)
        return c

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # ── reading ────────────────────────────────────────────────────────────
    def _fill(self):
        # Re-arm every time: callers sharing the socket may have changed it.
        self.sock.settimeout(self.timeout)
        try:
            chunk = self.sock.recv(1 << 16)
        except socket.timeout:
            raise TimeoutError(f"no complete reply within {self.timeout} s "
                               f"(have {len(self.buf)} bytes: {self.buf[:80]!r})")
        if not chunk:
            raise ConnectionError(f"server closed the connection mid-reply "
                                  f"(have {len(self.buf)} bytes: {self.buf[:80]!r})")
        self.buf += chunk

    def _line(self, pos: int) -> tuple[bytes, int]:
        while True:
            end = self.buf.find(b"\r\n", pos)
            if end >= 0:
                return self.buf[pos:end], end + 2
            self._fill()

    def _need(self, n: int):
        while len(self.buf) < n:
            self._fill()

    def _parse(self, pos: int):
        """Returns (value, end_pos). Reads more from the socket as needed."""
        self._need(pos + 1)
        t = self.buf[pos:pos + 1]
        line, nxt = self._line(pos + 1)
        if t == b"+":
            return line.decode("utf-8", "surrogateescape"), nxt
        if t == b"-":
            return RespError(line.decode("utf-8", "surrogateescape")), nxt
        if t == b":":
            try:
                return int(line), nxt
            except ValueError:
                raise RespProtocolError(f"bad integer reply {line!r}")
        if t in (b"$", b"="):          # bulk / verbatim
            try:
                n = int(line)
            except ValueError:
                raise RespProtocolError(f"bad bulk length {line!r}")
            if n < 0:
                return None, nxt
            self._need(nxt + n + 2)
            if self.buf[nxt + n:nxt + n + 2] != b"\r\n":
                raise RespProtocolError(
                    f"bulk of declared length {n} not followed by CRLF: "
                    f"{self.buf[nxt:nxt + n + 8]!r}")
            return self.buf[nxt:nxt + n], nxt + n + 2
        if t in (b"*", b"~", b">", b"%", b"|"):
            try:
                n = int(line)
            except ValueError:
                raise RespProtocolError(f"bad aggregate length {line!r}")
            if n < 0:
                return None, nxt
            if t in (b"%", b"|"):
                n *= 2
            items = []
            for _ in range(n):
                v, nxt = self._parse(nxt)
                items.append(v)
            if t == b"%":
                return dict(zip(items[0::2], items[1::2])), nxt
            return items, nxt
        if t == b"_":
            return None, nxt
        if t == b"#":
            return line == b"t", nxt
        if t == b",":
            return float(line), nxt
        if t == b"(":
            return int(line), nxt
        raise RespProtocolError(f"unknown RESP type byte {t!r} in {self.buf[pos:pos + 40]!r}")

    def read(self):
        v, end = self._parse(0)
        self.buf = self.buf[end:]
        return v

    def read_raw(self) -> bytes:
        _, end = self._parse(0)
        raw, self.buf = self.buf[:end], self.buf[end:]
        return raw

    # ── commands ───────────────────────────────────────────────────────────
    def cmd(self, *args):
        self.sock.sendall(encode(args))
        return self.read()

    def raw(self, *args) -> bytes:
        self.sock.sendall(encode(args))
        return self.read_raw()

    def cmd_synced(self, *args):
        """Send the command and a PING in ONE write; return the command's reply
        and require the very next reply to be PONG. A command that answers
        twice (or not at all) puts something else where PONG must be — and it
        costs no waiting, unlike probing for surplus bytes with a timeout."""
        self.sock.sendall(encode(args) + encode(("PING",)))
        r = self.read()
        p = self.read()
        if p != "PONG":
            raise RespProtocolError(f"{args[0]!r}: reply count desync — after its reply "
                                    f"{r!r} the next reply was {p!r}, not PONG")
        return r

    def pipeline(self, commands):
        self.sock.sendall(b"".join(encode(c) for c in commands))
        return [self.read() for _ in commands]

    def pipeline_raw(self, commands) -> list[bytes]:
        self.sock.sendall(b"".join(encode(c) for c in commands))
        return [self.read_raw() for _ in commands]

    def assert_in_sync(self):
        """The next reply must be the PONG we send now, and nothing may be
        buffered ahead of it. A command that answered twice fails here."""
        if self.buf:
            raise RespProtocolError(f"unread bytes before sync PING: {self.buf[:120]!r}")
        r = self.cmd("PING")
        if r != "PONG":
            raise RespProtocolError(f"expected PONG in sync check, got {r!r}")
        # Anything else arriving shortly after is a surplus reply.
        self.sock.settimeout(0.05)
        try:
            extra = self.sock.recv(4096)
        except (socket.timeout, BlockingIOError):
            extra = b""
        finally:
            self.sock.settimeout(self.timeout)
        if extra:
            raise RespProtocolError(f"surplus bytes after sync PING: {extra[:120]!r}")


def wait_ready(port: int, timeout: float = 30.0, host: str = "127.0.0.1", proc=None) -> None:
    """Block until the server ANSWERS PING. Accepting a connection is not
    enough: the server listens before it initialises and replays its WAL."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before answering PING")
        try:
            with Conn(port, host, timeout=2.0, connect_timeout=0.5) as c:
                if c.cmd("PING") == "PONG":
                    return
        except (OSError, TimeoutError, ConnectionError, RespProtocolError) as e:
            last = e
        time.sleep(0.1)
    raise RuntimeError(f"server on :{port} did not answer PING within {timeout} s ({last})")


def wait_ready_pid(port: int, proc, timeout: float = 60.0, host: str = "127.0.0.1") -> None:
    """Block until THIS process answers PING on `port` (#27).

    After a stop, a dead server's listening socket can outlive it for a moment
    (on Linux, the kernel tears an io_uring instance down after exit), and a
    connect then succeeds against nothing. Restart harnesses that took a
    successful connect, or any PONG, as "ready" sometimes talked to the old
    server. INFO's process_id must be the new pid; a binary that predates the
    field is accepted on PONG alone."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before answering PING")
        try:
            with Conn(port, host, timeout=5.0, connect_timeout=0.5) as c:
                if c.cmd("PING") == "PONG":
                    info = c.cmd("INFO", "server")
                    text = info.decode(errors="replace") if isinstance(info, bytes) else ""
                    pid = next((int(l.split(":", 1)[1]) for l in text.splitlines()
                                if l.startswith("process_id:")), None)
                    if pid is None or pid == proc.pid:
                        return
                    last = f"answered by pid {pid}, not {proc.pid}"
        except (OSError, TimeoutError, ConnectionError, RespProtocolError, ValueError) as e:
            last = e
        time.sleep(0.1)
    raise RuntimeError(f"server pid {proc.pid} did not answer on :{port} within {timeout} s ({last})")


def wait_port_free(port: int, timeout: float = 10.0, host: str = "127.0.0.1") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.3).close()
        except OSError:
            return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} still accepting after {timeout} s")


def parse_bytes(raw: bytes):
    """Parse ONE complete reply held in memory. Raises if `raw` is not exactly
    one well-formed reply (truncated, or with bytes left over)."""
    c = Conn.__new__(Conn)
    c.sock, c.buf, c.timeout = None, raw, 0.0

    def _no_more():
        raise RespProtocolError(f"truncated reply: {raw[:120]!r}")
    c._fill = _no_more
    v = c.read()
    if c.buf:
        raise RespProtocolError(f"{len(c.buf)} bytes after one reply: {c.buf[:80]!r}")
    return v


_WRAPPED: dict = {}


def reader(sock: socket.socket, timeout: float = 10.0) -> Conn:
    """The one Conn wrapping `sock` (created on first use). Helpers that pass
    bare sockets around use this so bytes read past a reply are not lost."""
    c = _WRAPPED.get(id(sock))
    if c is None or c.sock is not sock:
        c = _WRAPPED[id(sock)] = Conn.wrap(sock, timeout)
    return c
