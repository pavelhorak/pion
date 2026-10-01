"""Pion Serve — MAX Serve proxy backend (`--backend max`).

MAX Serve exposes an OpenAI-compatible endpoint, so the `max` backend flows
through the same generic forward path as vllm/openai/gemini/llamacpp — there is
no special-casing in `_forward_to_backend` / `_forward_stream`. These tests
stand up a minimal OpenAI-compatible mock in place of `max serve` and assert
both the non-streaming and streaming paths route correctly.

This is the request-layer Pion×MAX surface (semantic cache / routing). The KV
datapath is out of scope: MAX 26.4 removed the LMCache connector and its native
KVConnector factory is closed (gh #95).

Run: python pion-serve/tests/test_max_backend.py
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _MockMax(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible stand-in for `max serve`."""

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        assert req["model"] == "gemma-3-4b", req.get("model")
        stream = req.get("stream", False)
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in ["pong", "-from", "-max"]:
                chunk = {"choices": [{"index": 0, "delta": {"content": tok},
                                     "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "pong-from-max"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve():
    srv = HTTPServer(("127.0.0.1", 8000), _MockMax)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    srv = _serve()
    try:
        import serve  # noqa: E402
    except Exception as e:  # heavy deps (flask, numpy) missing → environment skip
        print(f"ENV_SKIP: cannot import serve ({type(e).__name__}: {e})")
        srv.shutdown()
        return 0

    assert "max" in serve.BACKENDS, "max backend not registered"
    assert serve.BACKENDS["max"]["url"] == "http://127.0.0.1:8000"

    serve._config.update({
        "backend_type": "max",
        "backend_url": serve.BACKENDS["max"]["url"],
        "model": "gemma-3-4b",
        "cache_enabled": False,
    })

    body = {"model": "gemma-3-4b", "max_tokens": 8}
    out = serve._forward_to_backend([{"role": "user", "content": "ping"}], body)
    assert out["content"] == "pong-from-max", out
    assert out["prompt_tokens"] == 3 and out["completion_tokens"] == 2, out
    print("PASS: non-streaming --backend max routes through OpenAI-compatible path")

    chunks = list(serve._forward_stream([{"role": "user", "content": "ping"}], body))
    joined = ""
    for c in chunks:
        if c.startswith("data: ") and "[DONE]" not in c:
            d = json.loads(c[6:])
            joined += d["choices"][0].get("delta", {}).get("content", "")
    assert joined == "pong-from-max", joined
    assert any("[DONE]" in c for c in chunks), "stream missing [DONE]"
    print("PASS: streaming --backend max relays SSE deltas")

    srv.shutdown()
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
