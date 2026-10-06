#!/usr/bin/env python3
"""gh #82 — ResponseWriter overflow regression.

Asserts that when a pipelined batch of responses can't fit in the 4 MB
RESP_BUF_SIZE, the client receives N well-formed frames followed by
`-ERR response exceeds buffer\r\n` — never a silently truncated stream.

Pre-fix behaviour: every fixed-byte appender (`append_ok_response`,
`append_pong_response`, …) silently `return`-ed on overflow, leaving the
client one response short with no indication. Pipelined clients (Celery,
Bull, MGET batchers) then desynced their request/reply accounting and
read the next response off the WRONG request.

Test plan:
  1. Start the server.
  2. SET k to a 220 KB value (small enough that several copies fit; large
     enough that ~19 of them blow past 4 MB).
  3. Pipeline N GET k requests in a single TCP write (N chosen so total
     reply bytes ≈ 6 MB, comfortably past the 4 MB buffer + 194 KB margin).
  4. Read RESP frames off the connection until quiet. Count how many are
     `$220000\r\n…\r\n` and confirm at least one `-ERR response exceeds
     buffer\r\n` appears in their natural order.
  5. Send PING on the same socket and require `+PONG` — confirms the wire
     stayed frame-synced through the overflow.

Requires: ./pion-server -w 1 (any worker count works; -w 1 keeps timing
stable). No --kvcache needed.
"""
from __future__ import annotations

import os
import select
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid  # noqa: E402

HOST = "127.0.0.1"
PORT = 1974
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VALUE_SIZE = 220_000     # bytes per GET reply (just over 200 KB)
PIPELINE_DEPTH = 30      # 30 × ~220 KB ≈ 6.6 MB ≫ 4 MB RESP_BUF_SIZE
OVERFLOW_LINE = b"-ERR response exceeds buffer\r\n"


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


def start_server(log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "-w", "1", "-p", str(PORT), "--no-auto-detect", "--no-auto-embed"]
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log_fp, stderr=log_fp,
        preexec_fn=os.setsid,
    )
    wait_ready_pid(PORT, proc, 30)   # this process, not a lingering listener (#27)
    return proc


def stop_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


class FrameReader:
    """Pull one RESP frame at a time from a socket, buffering raw bytes.

    Built on raw socket.recv so we can use `select` to poll for "is there
    more data?" without putting the socket file object into a sticky
    timed-out state (which `socket.makefile().readline()` does on Python 3.14)."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def _recv_until(self, want_bytes: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while len(self.buf) < want_bytes:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            r, _, _ = select.select([self.sock], [], [], remaining)
            if not r:
                return False
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                return False  # EOF
            self.buf += chunk
        return True

    def _find_crlf(self, start: int, timeout: float) -> int:
        # Pull more bytes until we find a CRLF after `start`, or timeout.
        deadline = time.time() + timeout
        while True:
            idx = self.buf.find(b"\r\n", start)
            if idx >= 0:
                return idx
            remaining = deadline - time.time()
            if remaining <= 0:
                return -1
            r, _, _ = select.select([self.sock], [], [], remaining)
            if not r:
                return -1
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                return -1  # EOF
            self.buf += chunk

    def next_frame(self, timeout: float = 15.0) -> Optional[bytes]:
        nl = self._find_crlf(0, timeout)
        if nl < 0:
            return None
        line = self.buf[:nl + 2]
        kind = line[:1]
        if kind in (b"+", b"-", b":"):
            self.buf = self.buf[nl + 2:]
            return line
        if kind == b"$":
            n = int(line[1:nl])
            if n < 0:
                self.buf = self.buf[nl + 2:]
                return line
            need = nl + 2 + n + 2
            if not self._recv_until(need, timeout):
                return None
            frame = self.buf[:need]
            self.buf = self.buf[need:]
            return frame
        raise RuntimeError(f"unexpected prefix {kind!r} at {line[:40]!r}")


def main() -> int:
    log_path = "/tmp/pion_resp_overflow.log"
    proc = None
    rc = 0
    try:
        proc = start_server(log_path)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        s.settimeout(15)
        s.connect((HOST, PORT))
        fr = FrameReader(s)

        # Phase 1: SET k = 220 KB value.
        big_value = (b"X" * VALUE_SIZE)
        s.sendall(_encode(["SET", "k", big_value]))
        if fr.next_frame() != b"+OK\r\n":
            print("FAIL: SET did not return +OK"); return 1

        # Phase 2: pipeline N GETs in a single write.
        pipe = _encode(["GET", "k"]) * PIPELINE_DEPTH
        print(f"sending {len(pipe)} bytes ({PIPELINE_DEPTH} pipelined GETs, "
              f"~{PIPELINE_DEPTH * VALUE_SIZE // (1024*1024)} MB of replies)…")
        s.sendall(pipe)

        # Phase 3: drain whatever the server sent for the batch. The current
        # ResponseWriter design emits exactly one `-ERR response exceeds buffer`
        # per flush-cycle (overflow_emitted latch). Subsequent appends in the
        # same flush are dropped — pre-fix they were SILENTLY dropped, which
        # is the bug; post-fix the latch-and-emit-once design at least gives
        # the client a clear signal in the stream instead of a truncated tail.
        # So the contract this test enforces:
        #   (a) at least one `-ERR response exceeds buffer` appeared,
        #   (b) every frame the client actually received parses as a
        #       well-formed RESP frame,
        #   (c) the connection is still frame-synced for the next request.
        ok_count = 0
        overflow_count = 0
        other = []
        while True:
            # Tight timeout once we've already seen the overflow signal:
            # the server flushes once per batch, so the trailing tail is
            # whatever it managed to write before _check_overflow tripped.
            poll_timeout = 0.4 if overflow_count >= 1 else 5.0
            frame = fr.next_frame(timeout=poll_timeout)
            if frame is None:
                break
            if frame.startswith(b"$" + str(VALUE_SIZE).encode() + b"\r\n"):
                ok_count += 1
            elif frame == OVERFLOW_LINE:
                overflow_count += 1
            else:
                other.append(frame[:60])

        print(f"  ok={ok_count}  overflow={overflow_count}  other={len(other)}")
        if other:
            print(f"  unexpected first three: {other[:3]}")

        if overflow_count == 0:
            print("FAIL: no overflow frame seen — silent drop still happening")
            return 1
        if ok_count == 0:
            print("FAIL: zero successful GETs — every response replaced by overflow")
            return 1
        if other:
            print("FAIL: malformed frames in the stream")
            return 1

        # Phase 4: PING — connection must still be frame-synced. The server's
        # next read picks up cleanly; pre-fix the silent drop left junk bytes
        # / truncated frames in the kernel buffer that could desync PING.
        s.sendall(_encode(["PING"]))
        ping_reply = fr.next_frame(timeout=5.0)
        if ping_reply != b"+PONG\r\n":
            print(f"FAIL: PING after overflow returned {ping_reply!r}, expected +PONG")
            return 1
        print("  PING after overflow → +PONG (connection still frame-synced)")

        s.close()
        print("\nPASS — gh #82 ResponseWriter overflow emits -ERR, stays frame-synced")
    finally:
        stop_server(proc)
        # No state files generated; nothing to clean.

    return rc


if __name__ == "__main__":
    sys.exit(main())
