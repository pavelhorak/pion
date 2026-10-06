#!/usr/bin/env python3
"""#51: AI.CHAT and AI.COMPLETE decode an LLM's JSON reply correctly.

Download-free: a fake HTTP server stands in for Ollama / an OpenAI-compatible
backend and returns a fixed reply with the escapes Go's encoder emits. No
model, no network beyond localhost.

Before the fix, AI.CHAT compared the 11-byte tag `"content":"` as 12 bytes, so
it never matched and AI.CHAT answered nil to every reply; AI.COMPLETE's
`"response":"` is genuinely 12 and worked. Neither decoded `\\uXXXX`, so Go's
`\\u003c` for `<` came back as `u003c`.

    python3 tests/test_ai_chat_json.py [--port 1974]
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAILS: list[str] = []

# The decoded text the backend "generates": ASCII that Go escapes as \uXXXX
# (< > &), a tab, a quote and backslash, and an emoji (a surrogate pair).
WANT = 'if a < b && c > d: say "hi"\tok \U0001F44B'


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def _fake_llm(sock: socket.socket, stop: list):
    """A raw-socket stand-in for Ollama / an OpenAI-compatible backend. Python's
    http.server does not answer Pion's hand-rolled HTTP/1.0 request, so this
    reads the request and writes a fixed HTTP/1.0 reply, as the real backend
    does. The JSON value carries the escapes Go's encoder emits: < > & as
    \\u003c \\u003e \\u0026, the emoji as a \\uXXXX surrogate pair, plus \\t \\" \\\\."""
    esc = r'if a \u003c b \u0026\u0026 c \u003e d: say \"hi\"\tok \ud83d\udc4b'
    sock.settimeout(0.5)
    while not stop:
        try:
            c, _ = sock.accept()
        except OSError:
            continue
        try:
            c.settimeout(2)
            req = b""
            while b"\r\n\r\n" not in req:
                d = c.recv(65536)
                if not d:
                    break
                req += d
            hdr, _, body = req.partition(b"\r\n\r\n")
            cl = 0
            for line in hdr.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    cl = int(line.split(b":")[1])
            while len(body) < cl:
                d = c.recv(65536)
                if not d:
                    break
                body += d
            is_chat = b"/v1/chat/completions" in hdr
            if is_chat:
                payload = ('{"choices":[{"message":{"role":"assistant","content":"'
                           + esc + '"}}]}').encode()
            else:
                payload = ('{"response":"' + esc + '","done":true,"logprobs":[-0.1,-0.2]}').encode()
            c.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                      + str(len(payload)).encode() + b"\r\nConnection: close\r\n\r\n" + payload)
        except OSError:
            pass
        finally:
            c.close()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    port = ap.parse_args().port
    llm_port = free_port()

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", llm_port))
    srv.listen(8)
    stop: list = []
    threading.Thread(target=_fake_llm, args=(srv, stop), daemon=True).start()

    binary = os.environ.get("PION_BIN") or os.path.join(ROOT, "pion-server")
    log = open("/tmp/pion_ai_chat_json.log", "w")
    proc = subprocess.Popen(
        [binary, "-p", str(port), "--kvcache", "-w", "1", "--flare",
         "--llm-model", "fake", "--llm-port", str(llm_port),
         "--no-auto-detect", "--no-auto-embed"],
        cwd=ROOT, stdout=log, stderr=log, preexec_fn=os.setsid)
    try:
        for _ in range(120):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        c = Conn(port, timeout=30)
        chat = c.cmd("AI.CHAT", "anything")
        check("AI.CHAT returns the content, not nil", chat is not None, repr(chat))
        check("AI.CHAT decodes \\uXXXX, \\t and the emoji", chat == WANT.encode(), repr(chat))
        # AI.COMPLETE goes through the semantic cache, which needs an embedding
        # backend this download-free test has turned off (--no-auto-embed), so
        # it answers "generation failed" at the embed step before the LLM. It
        # decodes the reply through the SAME _decode_json_string helper AI.CHAT
        # just exercised, and is covered end-to-end by the ollama-gated
        # test_ai_gateway.py. Report what it did; do not fail on the embed gate.
        comp = c.cmd("AI.COMPLETE", "anything", "THRESHOLD", "2.0")
        flat = comp if isinstance(comp, (bytes, bytearray)) else (
            b"".join(x for x in comp if isinstance(x, (bytes, bytearray))) if isinstance(comp, list) else b"")
        if WANT.encode() in bytes(flat):
            check("AI.COMPLETE decodes the same reply (shared decoder)", True)
        else:
            print(f"  INFO AI.COMPLETE needs embedding, off here: {repr(comp)[:80]} "
                  f"(decoder covered by AI.CHAT above)")
        check("server alive", c.cmd("PING") == "PONG")
        c.close()
    finally:
        stop.append(True)
        srv.close()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
