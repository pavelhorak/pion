"""A TCP front for the engine in this process, for TESTS: it lets the server's
own test tiers (raw sockets on a port) run through library mode, as
PION_IO_THREADS runs them through the I/O threads. The socket defeats the
point for performance; it exercises the outboxes, parked commands, pub/sub
and the tick.

    python -m pion_embedded.proxy --dir /tmp/data --port 1974
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import threading

from ._engine import CLOSED, PARKED, get_engine

TRACE = os.environ.get("PION_PROXY_TRACE") == "1"


def serve_conn(engine, s: socket.socket):
    c = engine.conn_new()
    wake = engine.wake_fd(c)          # readable while output waits for this connection
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    poller = select.poll()             # not select(): descriptors past 1024 (FD_SETSIZE)
    poller.register(s.fileno(), select.POLLIN)
    poller.register(wake, select.POLLIN)
    try:
        while True:
            ready = {fd for fd, _ in poller.poll(250)}
            if s.fileno() in ready:
                data = s.recv(1 << 20)
                if not data:
                    return
                if TRACE:
                    os.write(2, b"REQ " + data[:300].replace(b"\r\n", b" ") + b"\n")
                st, out = engine.exec(c, data)
            else:
                st, out = engine.poll(c)      # a parked reply, a pub/sub message
            if out:
                s.sendall(out)
            if st == CLOSED:
                return
            if st < 0:
                return
    except OSError:
        return
    finally:
        engine.conn_free(c)
        try:
            s.close()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--appendonly", default="yes")
    ap.add_argument("--maxmemory")
    ap.add_argument("--maxmemory-policy")
    ap.add_argument("--stack-mb", type=int, default=0, help="thread stack size (an -O0 library needs ~64)")
    a = ap.parse_args()
    if a.stack_mb:
        threading.stack_size(a.stack_mb << 20)
    cfg = {"appendonly": a.appendonly}
    if a.maxmemory:
        cfg["maxmemory"] = a.maxmemory
    if a.maxmemory_policy:
        cfg["maxmemory-policy"] = a.maxmemory_policy
    serve(get_engine(a.dir, cfg), a.port)


def serve(engine, port: int):
    """Accept on 127.0.0.1:port forever, one thread per connection."""
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        ls.bind(("127.0.0.1", port))
    except OSError as e:
        print(f"FATAL: cannot bind port {port}: {e}", flush=True)
        engine.close()
        raise SystemExit(1)
    ls.listen(65535)   # as pion-server: a connection burst queues, never resets
    print(f"pion_embedded proxy: {engine.version()} dir={engine.dir} port={port}", flush=True)
    while True:
        s, _ = ls.accept()
        threading.Thread(target=serve_conn, args=(engine, s), daemon=True).start()


if __name__ == "__main__":
    main()
