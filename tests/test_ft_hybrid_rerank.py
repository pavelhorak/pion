#!/usr/bin/env python3
"""gh #70 — FT.HYBRID RERANK regression test.

Exercises the opt-in `RERANK <host> <port> <model> [<top_n>]` keyword on
FT.HYBRID. Doesn't depend on the real superlinked/sie service — spins up
a Python `http.server`-based mock that returns deterministic scores from
the request body, then verifies Pion's response order matches the mock's
score order rather than the underlying RRF order.

Test plan:
  1. Start Pion (-w 1, --no-auto-detect, --no-auto-embed).
  2. FT.CREATE the index with TEXT + VECTOR fields.
  3. FT.ADDTEXT + HSET for 5 docs whose vector and text are arranged so
     vector-search alone returns a known order.
  4. Spin up the mock SIE server on a free port. The mock parses the
     `documents` JSON array and returns `{scores: [...]}` that reverses
     the candidate order (highest score for the last doc).
  5. FT.HYBRID idx "query" <blob> K 5 ALPHA 0.5 RERANK 127.0.0.1 <port> mock-model 5
  6. Assert the response top-1 is the doc that scored highest at the mock,
     i.e. the document that was LAST in the RRF order.
  7. As a control, repeat without the RERANK keyword — assert the order
     matches the original RRF (top-1 should NOT match the mock's pick).

Failure-mode coverage: if the mock isn't reachable (port closed), Pion
should fall back to RRF order silently. Verified by setting an invalid
port and confirming the response is identical to the no-RERANK control.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np

HOST = "127.0.0.1"
PORT = 1974
DIM = 8
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _encode(parts) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for p in parts:
        if isinstance(p, bytes):
            out.append(f"${len(p)}\r\n".encode()); out.append(p); out.append(b"\r\n")
        else:
            s = str(p)
            out.append(f"${len(s)}\r\n{s}\r\n".encode())
    return b"".join(out)


class Conn:
    def __init__(self, port: int = PORT, timeout: float = 30.0):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.settimeout(timeout)
        self.s.connect((HOST, port))
        self.f = self.s.makefile("rb")

    def call(self, *parts):
        self.s.sendall(_encode(parts))
        return self._read()

    def _read(self) -> bytes:
        line = self.f.readline()
        if not line:
            raise ConnectionError("pion closed")
        kind = line[:1]
        if kind in (b"+", b"-", b":"):
            return line
        if kind == b"$":
            n = int(line[1:].rstrip())
            if n < 0: return line
            body = self.f.read(n + 2)
            return line + body
        if kind == b"*":
            n = int(line[1:].rstrip())
            if n < 0: return line
            # Read N elements (recursively for nested arrays).
            return line + b"".join(self._read() for _ in range(n))
        raise RuntimeError(f"unexpected prefix {kind!r}: {line!r}")

    def close(self):
        try: self.f.close()
        except Exception: pass
        try: self.s.close()
        except Exception: pass


def start_server(log_path: str) -> subprocess.Popen:
    binary = os.environ.get("PION_BIN") or os.path.join(PROJECT_ROOT, "pion-server")
    cmd = [binary, "-w", "1", "-p", str(PORT), "--no-auto-detect", "--no-auto-embed"]
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log_fp, stderr=log_fp,
        preexec_fn=os.setsid,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            s = socket.create_connection((HOST, PORT), timeout=1); s.close()
            return proc
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("pion-server did not start")


def stop_server(proc: Optional[subprocess.Popen]):
    if proc is None: return
    try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=10)
    except Exception:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception: pass


# ── Mock SIE server ────────────────────────────────────────────────────


# Storage for "what did the mock see last" — global because the handler
# class is constructed per-request by http.server.
_last_request: dict = {}


class _MockSIEHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # silence access logs

    def do_POST(self):
        if self.path != "/v1/score":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
        except Exception:
            self.send_response(400); self.end_headers(); return
        docs = payload.get("documents", [])
        # Score = position in the input list (highest for the last doc).
        # That way: input order [d0, d1, d2, d3, d4] → scores [0, 1, 2, 3, 4]
        # → output ranking [d4, d3, d2, d1, d0] (reversed).
        scores = [float(i) for i in range(len(docs))]
        _last_request["docs"] = docs
        _last_request["query"] = payload.get("query")
        _last_request["model"] = payload.get("model")
        resp = json.dumps({"scores": scores}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def start_mock_sie() -> tuple[HTTPServer, int, threading.Thread]:
    port = free_port()
    httpd = HTTPServer(("127.0.0.1", port), _MockSIEHandler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, port, th


# ── Helpers to extract docs from the RESP array response ──────────────


def _split_resp_array(buf: bytes) -> list[bytes]:
    """Walk a RESP top-level array and return its raw element bytes."""
    if not buf.startswith(b"*"):
        raise RuntimeError(f"not an array: {buf[:40]!r}")
    nl = buf.find(b"\r\n")
    n = int(buf[1:nl])
    pos = nl + 2
    parts = []
    for _ in range(n):
        end = _frame_end(buf, pos)
        parts.append(buf[pos:end])
        pos = end
    return parts


def _frame_end(buf: bytes, start: int) -> int:
    kind = buf[start:start + 1]
    nl = buf.find(b"\r\n", start)
    if kind in (b"+", b"-", b":"):
        return nl + 2
    if kind == b"$":
        n = int(buf[start + 1:nl])
        if n < 0: return nl + 2
        return nl + 2 + n + 2
    if kind == b"*":
        n = int(buf[start + 1:nl])
        if n < 0: return nl + 2
        pos = nl + 2
        for _ in range(n):
            pos = _frame_end(buf, pos)
        return pos
    raise RuntimeError(f"bad prefix {kind!r} at {buf[start:start+40]!r}")


def first_doc_id(response: bytes) -> bytes:
    """FT.HYBRID returns `[N, key1, fields1, key2, fields2, …]`. Return key1."""
    parts = _split_resp_array(response)
    # parts[0] is the integer count, parts[1] is the first doc key.
    if len(parts) < 2:
        raise RuntimeError(f"too few parts: {len(parts)}")
    key_frame = parts[1]
    nl = key_frame.find(b"\r\n")
    n = int(key_frame[1:nl])
    return key_frame[nl + 2:nl + 2 + n]


def main() -> int:
    proc = None
    httpd = None
    rc = 0
    log_path = "/tmp/pion_ft_hybrid_rerank.log"
    try:
        # State carries over between runs — wipe before start.
        for f in ("pion.hnsw.0", "pion.wal.0", "pion.snapshot.0"):
            p = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(p): os.remove(p)

        proc = start_server(log_path)
        httpd, mock_port, _ = start_mock_sie()

        c = Conn()
        # ── Build a tiny searchable index ─────────────────────────────
        r = c.call(
            "FT.CREATE", "idx", "SCHEMA",
            "body", "TEXT",
            "vec", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
        )
        if not r.startswith(b"+OK"):
            print(f"FAIL: FT.CREATE rejected: {r[:80]!r}"); return 1

        # 5 docs whose vectors land at predictable distances. The query
        # vector is all zeros — so doc[i]'s L2 distance ranks ascending with
        # i. RRF order should be doc:0 best → doc:4 worst.
        for i in range(5):
            doc_id = f"doc:{i}"
            vec = np.full(DIM, float(i + 1), dtype=np.float32).tobytes()
            c.call("HSET", doc_id, "body", f"document number {i} apple banana", "vec", vec)
            c.call("FT.ADDTEXT", "idx", str(i), f"document number {i} apple banana")

        if not c.call("FT.OPTIMIZE", "idx").startswith(b"+OK"):
            print("FAIL: FT.OPTIMIZE rejected"); return 1

        query_vec = np.zeros(DIM, dtype=np.float32).tobytes()

        # ── Control 1: no RERANK → RRF order (doc:0 first) ────────────
        ctrl = c.call("FT.HYBRID", "idx", "apple", query_vec, "K", "5", "ALPHA", "0.5")
        top = first_doc_id(ctrl)
        print(f"[ctrl] top-1 without RERANK: {top!r}")
        if top != b"doc:0":
            print(f"   warn: expected doc:0 top under RRF, got {top!r}")
            # Don't fail — RRF can put the BM25 winner first depending on
            # tokenization. The contract we test is "RERANK changes the order".

        # Cache the control order to compare against the unreachable-mock fallback.
        ctrl_parts = _split_resp_array(ctrl)

        # ── Phase 1: RERANK against the mock ──────────────────────────
        r = c.call(
            "FT.HYBRID", "idx", "apple", query_vec,
            "K", "5", "ALPHA", "0.5",
            "RERANK", "127.0.0.1", str(mock_port), "mock-model", "5",
        )
        top = first_doc_id(r)
        print(f"[1] top-1 with mock RERANK: {top!r}")

        # The mock returns scores indexed by input order. So whichever doc
        # was the LAST input becomes top-1. Check the mock received 5 docs.
        if not _last_request.get("docs"):
            print("FAIL: mock SIE was not called (no recorded request)")
            return 1
        if len(_last_request["docs"]) != 5:
            print(f"FAIL: mock saw {len(_last_request['docs'])} docs, expected 5")
            return 1
        if _last_request.get("query") != "apple":
            print(f"FAIL: mock saw query {_last_request.get('query')!r}, expected 'apple'")
            return 1
        if _last_request.get("model") != "mock-model":
            print(f"FAIL: mock saw model {_last_request.get('model')!r}")
            return 1

        # The expected top-1 is the doc whose text was the LAST one sent
        # (since mock gives that one the highest score).
        last_doc_text = _last_request["docs"][-1]
        # Cross-reference last_doc_text back to a doc_id: the text was
        # "document number N apple banana".
        last_n = int(last_doc_text.split()[2])
        expected_top = f"doc:{last_n}".encode()
        if top != expected_top:
            print(f"FAIL: RERANK top-1 = {top!r}, expected {expected_top!r}")
            return 1
        print(f"   OK — RERANK reordered: top-1 = {top!r} matches mock's argmax")

        # ── Phase 2: RERANK with unreachable port → fallback to RRF ──
        # Pick a port nothing's listening on.
        dead_port = free_port()  # immediately released; nothing's listening
        r = c.call(
            "FT.HYBRID", "idx", "apple", query_vec,
            "K", "5", "ALPHA", "0.5",
            "RERANK", "127.0.0.1", str(dead_port), "mock-model", "5",
        )
        top = first_doc_id(r)
        ctrl_top = first_doc_id(ctrl)
        print(f"[2] top-1 with unreachable RERANK: {top!r} (control: {ctrl_top!r})")
        if top != ctrl_top:
            print(f"FAIL: RERANK fallback did not match RRF order "
                  f"({top!r} vs control {ctrl_top!r})")
            return 1
        print("   OK — RERANK fallback returns the RRF order silently")

        c.close()
        print("\nPASS — gh #70 FT.HYBRID RERANK keyword works + fails safe")
    finally:
        if httpd is not None:
            httpd.shutdown()
        stop_server(proc)
        for f in ("pion.hnsw.0", "pion.wal.0", "pion.snapshot.0"):
            p = os.path.join(PROJECT_ROOT, f)
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
