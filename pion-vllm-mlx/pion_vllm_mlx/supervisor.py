"""`serve --unload-after SECONDS`: give the model's RAM back when idle.

On a 16–32 GB Mac the model competes with Xcode, a Docker build and the
browser. Every in-process prompt cache dies with the model; Pion's does not.
So the unload is real — the model process exits — and the next request pays
a model load plus a restore from Pion instead of a full re-prefill.

A small front process owns the public port. It starts the model process
(`serve` on an internal port) on the first request, forwards every request
to it, and stops it after SECONDS without traffic. If the model process
dies mid-answer, the client's connection drops; its retry starts a fresh one.
"""
from __future__ import annotations

import http.client
import os
import signal
import subprocess
import sys
import threading
import time
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FastBindHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS lookup in server_bind.

    http.server.HTTPServer.server_bind calls socket.getfqdn(host) only to fill
    `server_name`. On a Mac whose reverse lookup times out it costs ~33 s per
    bind (measured on the 16 GB mini, 2026-09-24) — every model reload paid it,
    and so does every stock `mlx_lm.server` start on such a machine."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
       "proxy-authorization", "proxy-authenticate", "host", "content-length"}


class Supervisor:
    def __init__(self, child_argv, internal_port, unload_after, log=print):
        self.child_argv = child_argv
        self.port = internal_port
        self.unload_after = unload_after
        self.log = log
        self.proc = None
        self.lock = threading.Lock()
        self.in_flight = 0
        self.last = time.monotonic()
        self.loads = 0
        self.unloads = 0

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _healthy(self) -> bool:
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
            c.request("GET", "/health")
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except OSError:
            return False

    def ensure(self) -> None:
        with self.lock:
            if self._alive() and self._healthy():
                return
            if self._alive():
                self._stop()
            t0 = time.monotonic()
            self.proc = subprocess.Popen(self.child_argv, start_new_session=True)
            while not self._healthy():
                if self.proc.poll() is not None:
                    raise RuntimeError(f"model process exited with {self.proc.returncode}")
                if time.monotonic() - t0 > 300:
                    raise RuntimeError("model process did not become healthy in 300 s")
                time.sleep(0.2)
            self.loads += 1
            self.log(f"supervisor: model loaded in {time.monotonic() - t0:.1f}s (load #{self.loads})")

    def _stop(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc = None

    def reaper(self) -> None:
        while True:
            time.sleep(1.0)
            with self.lock:
                idle = time.monotonic() - self.last
                if self._alive() and self.in_flight == 0 and idle >= self.unload_after:
                    self._stop()
                    self.unloads += 1
                    self.log(f"supervisor: idle {idle:.0f}s — model unloaded (unload #{self.unloads}); "
                             f"its prompt cache stays in Pion")

    def begin(self) -> None:
        with self.lock:
            self.in_flight += 1
            self.last = time.monotonic()

    def end(self) -> None:
        with self.lock:
            self.in_flight -= 1
            self.last = time.monotonic()


def make_handler(sup: Supervisor):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward(self):
            if self.command == "GET" and self.path.split("?")[0] == "/health":
                return self._json(200, b'{"status": "ok"}')
            if self.command == "GET" and self.path.split("?")[0] == "/v1/pion/supervisor":
                pid = sup.proc.pid if sup._alive() else None
                body = (f'{{"loaded": {str(sup._alive()).lower()}, "pid": {"null" if pid is None else pid}, '
                        f'"loads": {sup.loads}, "unloads": {sup.unloads}, "in_flight": {sup.in_flight}}}').encode()
                return self._json(200, body)
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            sup.begin()
            try:
                try:
                    sup.ensure()
                except RuntimeError as e:
                    return self._json(503, f'{{"error": "{e}"}}'.encode())
                conn = http.client.HTTPConnection("127.0.0.1", sup.port, timeout=3600)
                hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                conn.request(self.command, self.path, body=body, headers=hdrs)
                r = conn.getresponse()
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                except (http.client.IncompleteRead, ConnectionError, OSError):
                    pass            # model process died mid-answer: drop; the client retries
                self.close_connection = True
                conn.close()
            finally:
                sup.end()

        do_GET = do_POST = do_PUT = do_DELETE = _forward

    return H


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(host, port, child_argv, internal_port, unload_after):
    sup = Supervisor(child_argv, internal_port, unload_after,
                     log=lambda m: print(m, file=sys.stderr, flush=True))
    threading.Thread(target=sup.reaper, daemon=True).start()

    # The model process runs in its own session (so a signal to it cannot hit
    # us); that also means it outlives us unless we stop it on the way out.
    def _exit(signum, _frame):
        with sup.lock:
            if sup._alive():
                sup._stop()
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _exit)
    print(f"pion-vllm-mlx serve: http://{host}:{port} — model loads on demand, "
          f"unloads after {unload_after:.0f}s idle (model process on :{internal_port})", flush=True)
    srv = FastBindHTTPServer((host, port), make_handler(sup))
    try:
        srv.serve_forever()
    finally:
        with sup.lock:
            if sup._alive():
                sup._stop()
