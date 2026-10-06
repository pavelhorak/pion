#!/usr/bin/env python3
"""A TCP proxy that damages the server's replies on purpose — to prove a test's
reply reader notices (step 6 of the 2026-09-29 test audit).

A test can only fail on a fault its reader can see. The old helpers read one
`recv()` per command, so three faults went straight past them; this proxy
injects each one between a test and a real server:

  split          every reply is forwarded in tiny pieces with pauses, so no
                 reply arrives in one read (a reader must parse, not recv once)
  delay:S        the FIRST reply on each connection is held S seconds — a
                 server replaying its WAL after a restart looks like this
  dup:N          the N-th reply on each connection is sent twice — exactly
                 what a reply-count desync (one command in, two replies out)
                 looks like on the wire

Client -> server bytes are forwarded untouched.

    python3 tests/fault_proxy.py LISTEN_PORT TARGET_PORT MODE
    with FaultProxy(listen, target, "dup:3"): ...
"""
from __future__ import annotations

import socket
import sys
import threading
import time


def frame_end(buf: bytes, pos: int = 0):
    """End offset of the complete RESP reply starting at `pos`, or None if
    `buf` does not hold all of it yet."""
    if pos >= len(buf):
        return None
    t = buf[pos:pos + 1]
    eol = buf.find(b"\r\n", pos + 1)
    if eol < 0:
        return None
    line, nxt = buf[pos + 1:eol], eol + 2
    if t in (b"+", b"-", b":", b"_", b"#", b",", b"("):
        return nxt
    if t in (b"$", b"=", b"!"):
        n = int(line)
        if n < 0:
            return nxt
        return nxt + n + 2 if len(buf) >= nxt + n + 2 else None
    if t in (b"*", b"~", b">", b"%", b"|"):
        n = int(line)
        if n < 0:
            return nxt
        if t in (b"%", b"|"):
            n *= 2
        for _ in range(n):
            nxt = frame_end(buf, nxt)
            if nxt is None:
                return None
        return nxt
    raise ValueError(f"unknown RESP type byte {t!r}")


class FaultProxy:
    def __init__(self, listen_port: int, target_port: int, mode: str, host: str = "127.0.0.1"):
        self.listen_port, self.target_port, self.mode, self.host = listen_port, target_port, mode, host
        self.injected = 0          # faults actually applied, so a caller can tell it happened
        self._stop = threading.Event()
        self._lsock: socket.socket = None  # type: ignore[assignment]
        self._threads = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.listen_port))
        s.listen(64)
        s.settimeout(0.2)
        self._lsock = s
        t = threading.Thread(target=self._accept, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        """Release the port before returning (#27). Closing the listening
        socket alone did not: on Linux an accept() blocked in another thread
        keeps the socket alive until it returns, so the next FaultProxy on the
        same port got EADDRINUSE. shutdown() wakes that accept; then join the
        thread, then close."""
        self._stop.set()
        try:
            self._lsock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        for t in self._threads:
            t.join(timeout=5)
        try:
            self._lsock.close()
        except OSError:
            pass

    def _accept(self):
        while not self._stop.is_set():
            try:
                c, _ = self._lsock.accept()
            except (socket.timeout, OSError):
                continue
            try:
                u = socket.create_connection((self.host, self.target_port), timeout=5)
            except OSError:
                c.close()
                continue
            for a in (c, u):
                a.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                a.settimeout(None)
            threading.Thread(target=self._up, args=(c, u), daemon=True).start()
            threading.Thread(target=self._down, args=(u, c), daemon=True).start()

    @staticmethod
    def _up(c, u):
        try:
            while True:
                b = c.recv(1 << 16)
                if not b:
                    break
                u.sendall(b)
        except OSError:
            pass
        finally:
            for a in (u,):
                try:
                    a.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

    def _down(self, u, c):
        kind, _, arg = self.mode.partition(":")
        buf, n = b"", 0
        try:
            while True:
                b = u.recv(1 << 16)
                if not b:
                    break
                buf += b
                while True:
                    end = frame_end(buf)
                    if end is None:
                        break
                    frame, buf = buf[:end], buf[end:]
                    n += 1
                    if kind == "split":
                        # Cuts inside the type byte, the length line, the
                        # trailing CRLF and the middle, each read on its own.
                        cuts = sorted({x for x in (1, 2, 5, len(frame) // 2, len(frame) - 2,
                                                   len(frame) - 1) if 0 < x < len(frame)})
                        prev = 0
                        for x in cuts + [len(frame)]:
                            c.sendall(frame[prev:x])
                            prev = x
                            time.sleep(0.002)
                        self.injected += 1
                    elif kind == "delay" and n == 1:
                        self.injected += 1
                        time.sleep(float(arg))
                        c.sendall(frame)
                    elif kind == "dup" and n == int(arg):
                        # Separate writes with gaps, so the copy is read as the
                        # NEXT reply rather than in the same read as the first.
                        self.injected += 1
                        c.sendall(frame)
                        time.sleep(0.05)
                        c.sendall(frame)
                        time.sleep(0.05)
                    else:
                        c.sendall(frame)
        except OSError:
            pass
        finally:
            try:
                c.close()
            except OSError:
                pass


if __name__ == "__main__":
    lp, tp, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    p = FaultProxy(lp, tp, mode)
    p.start()
    print(f"[fault_proxy] :{lp} -> :{tp} mode={mode}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.stop()
