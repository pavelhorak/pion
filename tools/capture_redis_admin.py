#!/usr/bin/env python3
"""Capture what Pion's admin and introspection commands are compared with
(#47) from a live redis-server, into tools/redis_command_info.json:

  * COMMAND INFO of every command Pion implements that Redis has, as Redis
    encodes it in RESP2 and in RESP3 (raw reply bytes, base64);
  * the ACL categories, and the commands in each.

The command metadata (arity, flags, key positions, key specs, ACL
categories) describes the protocol interface clients route by; it is
committed, like tools/redis_arity.txt, so the build needs no redis-server.

    REDIS_PORT=6399 python3 tools/capture_redis_admin.py   # against a running server
"""
import base64
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("REDIS_PORT", "6379"))


def encode(*args) -> bytes:
    out = [b"*%d\r\n" % len(args)]
    for a in args:
        a = a if isinstance(a, bytes) else str(a).encode()
        out.append(b"$%d\r\n%s\r\n" % (len(a), a))
    return b"".join(out)


class Raw:
    """Reads one whole RESP reply and returns its raw bytes."""
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port))
        self.buf = b""

    def _need(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 16)
            if not chunk:
                raise EOFError
            self.buf += chunk

    def _line_end(self, pos):
        while True:
            e = self.buf.find(b"\r\n", pos)
            if e >= 0:
                return e
            self._need(len(self.buf) + 1)

    def _skip(self, pos):
        self._need(pos + 1)
        t = self.buf[pos:pos + 1]
        e = self._line_end(pos)
        head = self.buf[pos + 1:e]
        nxt = e + 2
        if t in b"+-:_,#(":
            return nxt
        if t in b"$=!":
            n = int(head)
            if n < 0:
                return nxt
            self._need(nxt + n + 2)
            return nxt + n + 2
        if t in b"*~>":
            n = int(head)
            for _ in range(max(n, 0)):
                nxt = self._skip(nxt)
            return nxt
        if t == b"%" or t == b"|":
            n = int(head)
            for _ in range(2 * n):
                nxt = self._skip(nxt)
            return nxt
        raise ValueError(t)

    def cmd(self, *args) -> bytes:
        self.s.sendall(encode(*args))
        end = self._skip(0)
        raw, self.buf = self.buf[:end], self.buf[end:]
        return raw


def pion_command_names():
    """The names in Pion's generated command table."""
    text = (ROOT / "src" / "commands" / "command_table.mojo").read_text()
    return sorted(set(re.findall(r'_cmd_eq_ci\(tp, tl, "([^"]+)"\)', text)))


def main():
    r2 = Raw(PORT)
    r3 = Raw(PORT)
    r3.cmd("HELLO", "3")
    names = pion_command_names()
    info = {}
    for n in names:
        a = r2.cmd("COMMAND", "INFO", n)
        if a[4:] in (b"$-1\r\n", b"*-1\r\n", b"_\r\n"):
            continue                                   # Redis has no such command
        b = r3.cmd("COMMAND", "INFO", n)
        info[n] = {"resp2": base64.b64encode(a[4:]).decode(), "resp3": base64.b64encode(b[4:]).decode()}
    cats_raw = r2.cmd("ACL", "CAT")
    cats = re.findall(rb"\$\d+\r\n([^\r]+)\r\n", cats_raw)
    cat_members = {}
    for c in cats:
        raw = r2.cmd("ACL", "CAT", c)
        cat_members[c.decode()] = sorted(m.decode() for m in re.findall(rb"\$\d+\r\n([^\r]+)\r\n", raw))
    out = {"redis_version": re.search(rb"redis_version:([0-9.]+)", r2.cmd("INFO", "server")).group(1).decode(),
           "command_info": info, "acl_categories": [c.decode() for c in cats], "acl_category_members": cat_members}
    dst = ROOT / "tools" / "redis_command_info.json"
    dst.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{len(info)} of {len(names)} Pion commands have a Redis entry; "
          f"{len(cats)} ACL categories -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
