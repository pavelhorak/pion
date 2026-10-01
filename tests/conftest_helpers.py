"""Server lifecycle helpers for Pion integration tests.

Usage:
    from conftest_helpers import PionServer

    with PionServer(port=1976, workers=1, flags=["--epoll"]) as server:
        sock = server.connect()
        resp = server.send_recv(sock, "PING")
        assert "+PONG" in resp
        sock.close()
"""

import os
import signal
import socket
import subprocess
import time


class PionServerError(Exception):
    """Raised when the Pion server fails to start or behaves unexpectedly."""
    pass


class PionServer:
    """Context manager for starting/stopping a pion-server instance.

    Handles:
    - Starting the binary with the given port, worker count, and extra flags
    - Waiting for the port to become ready (polling connect)
    - Cleanup on exit: SIGTERM -> wait -> SIGKILL if needed -> WAL file removal
    - RESP command encoding/decoding via raw sockets
    """

    def __init__(
        self,
        port=1976,
        workers=1,
        binary="./pion-server",
        flags=None,
        startup_timeout=20,
        workdir=None,
    ):
        self.port = port
        self.workers = workers
        self.binary = binary
        self.flags = flags or []
        self.startup_timeout = startup_timeout
        self.workdir = workdir or os.path.join("/tmp", f"pion_test_{port}")
        self.process = None
        self._prev_sigint = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        return False

    # ─── Public API ───────────────────────────────────────────────────────────

    def connect(self, timeout=5.0):
        """Return a connected TCP socket to the server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(("127.0.0.1", self.port))
        sock.settimeout(None)
        return sock

    def send_recv(self, sock, *args, timeout=2.0):
        """Encode a RESP command from *args, send it, and return the decoded response."""
        sock.sendall(self._encode_cmd(args))
        return self._recv_resp(sock, timeout=timeout)

    def flushall(self, sock):
        """Send FLUSHALL and return the response."""
        return self.send_recv(sock, "FLUSHALL")

    def wait_for_port(self, timeout=None):
        """Block until the server port accepts connections, or raise on timeout."""
        timeout = timeout or self.startup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect(("127.0.0.1", self.port))
                s.close()
                return
            except (ConnectionRefusedError, OSError):
                s.close()
                time.sleep(0.2)
        # Timed out — kill the server and raise
        self._force_kill()
        raise PionServerError(
            f"Pion server did not become ready on port {self.port} "
            f"within {timeout}s"
        )

    def wait_for_port_free(self, timeout=10):
        """Block until the port is no longer accepting connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", self.port))
                s.close()
                time.sleep(0.2)
            except (ConnectionRefusedError, OSError):
                s.close()
                return
        raise PionServerError(
            f"Port {self.port} still in use after {timeout}s"
        )

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _start(self):
        # Verify binary exists
        if not os.path.isfile(self.binary):
            raise PionServerError(
                f"Server binary not found: {self.binary}\n"
                f"Build it first: pixi run build"
            )

        # Check port is free before starting
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(("127.0.0.1", self.port))
            s.close()
            raise PionServerError(
                f"Port {self.port} is already in use. "
                f"Kill existing processes first."
            )
        except (ConnectionRefusedError, OSError):
            try:
                s.close()
            except Exception:
                pass

        # Create working directory and clean up old WAL files
        os.makedirs(self.workdir, exist_ok=True)
        self._clean_wal_files()

        # Build command
        cmd = [
            os.path.abspath(self.binary),
            "-p", str(self.port),
            "-w", str(self.workers),
        ] + self.flags

        # gh #253: -w N > 1 refuses to start without an explicit acknowledgement
        # that each worker owns a private keyspace. A test that deliberately
        # asks for multiple workers has already made that choice.
        if self.workers > 1 and "--independent-workers" not in cmd:
            cmd.append("--independent-workers")

        # Start the server
        self.process = subprocess.Popen(
            cmd,
            cwd=self.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Install SIGINT handler to ensure cleanup
        self._prev_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Wait for port to be ready
        self.wait_for_port()

    def _stop(self):
        # Restore SIGINT handler
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
            self._prev_sigint = None

        if self.process is None:
            return

        # SIGTERM first (graceful)
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._force_kill()
        except ProcessLookupError:
            pass  # Already dead

        self.process = None
        self._clean_wal_files()

    def _force_kill(self):
        """Send SIGKILL and wait."""
        if self.process is None:
            return
        try:
            self.process.kill()
            self.process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def _clean_wal_files(self):
        """Remove pion.wal.* files from the working directory."""
        if not os.path.isdir(self.workdir):
            return
        for f in os.listdir(self.workdir):
            if f.startswith("pion.wal."):
                try:
                    os.remove(os.path.join(self.workdir, f))
                except OSError:
                    pass

    def _sigint_handler(self, signum, frame):
        """Ensure server is killed on Ctrl-C."""
        self._stop()
        if self._prev_sigint and callable(self._prev_sigint):
            self._prev_sigint(signum, frame)
        else:
            raise KeyboardInterrupt

    # ─── RESP helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _encode_cmd(args):
        """Encode a list/tuple of strings as a RESP array command."""
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
        return b"".join(parts)

    @staticmethod
    def _recv_resp(sock, timeout=2.0):
        """Receive one RESP response (best-effort, short timeout)."""
        sock.settimeout(timeout)
        chunks = []
        try:
            data = sock.recv(65536)
            if data:
                chunks.append(data)
        except socket.timeout:
            pass
        sock.settimeout(None)
        return b"".join(chunks).decode(errors="replace")
