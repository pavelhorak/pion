#!/usr/bin/env python3
"""gh #69 — pion-serve SIE embedding backend smoke test.

Validates:
  [1] Cascade `_init_embedder` honors `--sie-url`: when the URL is reachable
      and returns a valid OpenAI-compat embeddings response, the cascade
      selects SIE (not Ollama). Tested via a tiny mock server in-process.
  [2] SIE-unreachable / non-200 / bad-shape: cascade falls through to the
      next backend (Ollama probe → fallback). No crash.
  [3] `_embed_via_sie(text)` correctly L2-normalizes the returned vector.

No live SIE instance required.
"""
from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
from contextlib import contextmanager
from typing import Optional

import numpy as np


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def mock_sie_server(*, return_embedding: Optional[list] = None, status: int = 200,
                     bad_shape: bool = False):
    """Tiny HTTP server that mimics SIE's /v1/embeddings. Returns the chosen
    embedding (default: 8-dim random) for any request, or a non-200, or
    a bad-shape response, depending on args."""
    port = _free_port()
    embedding = return_embedding if return_embedding is not None else \
        list(np.random.RandomState(0).standard_normal(8).astype(float))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):  # silence
            pass

        def do_POST(self):
            # Drain the request body so the connection stays clean for follow-up.
            ln = int(self.headers.get("Content-Length", 0))
            if ln:
                self.rfile.read(ln)
            if status != 200:
                body = b'{"error": "mock SIE failure"}'
            elif bad_shape:
                body = json.dumps({"object": "list"}).encode()
            else:
                body = json.dumps({
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
                    "model": "BAAI/bge-small-en-v1.5",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", embedding
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    # Import after mocks are constructible (top-level imports of pion-serve
    # don't require a live Pion server beyond a connection attempt — we
    # neutralize that by clearing `_pion` after import).
    sys.path.insert(0, "pion-serve")
    import serve as ps

    fail = False

    # ── [1] SIE reachable + valid response → cascade picks SIE ──
    print("[1] SIE reachable + valid response → cascade picks _embed_via_sie")
    with mock_sie_server() as (url, planted_embedding):
        ps._config = {"sie_url": url, "sie_model": "BAAI/bge-small-en-v1.5"}
        ps._pion = None  # skip Pion probe
        ps._embed_fn = None
        ps._init_embedder(provider="auto")
        if ps._embed_fn is ps._embed_via_sie:
            print("   OK: _embed_fn is _embed_via_sie")
        else:
            print(f"   FAIL: _embed_fn is {ps._embed_fn}")
            fail = True

        # Embed something and verify the vector matches the planted one (L2 norm of original).
        vec = ps._embed_via_sie("hello world")
        if vec is None:
            print("   FAIL: _embed_via_sie returned None")
            fail = True
        else:
            expected = np.array(planted_embedding, dtype=np.float32)
            expected /= np.linalg.norm(expected) + 1e-10
            max_diff = float(np.max(np.abs(vec - expected)))
            if max_diff < 1e-6:
                print(f"   OK: returned vector matches L2-normalized planted embedding "
                      f"(max|Δ| = {max_diff:.2e})")
            else:
                print(f"   FAIL: max|Δ| = {max_diff:.3e}")
                fail = True

    # ── [2] SIE unreachable → cascade does not pick _embed_via_sie ──
    print("\n[2] SIE URL unreachable → cascade falls through (does not pick SIE)")
    ps._config = {"sie_url": "http://127.0.0.1:1", "sie_model": "x"}  # closed port
    ps._pion = None
    ps._embed_fn = None
    ps._init_embedder(provider="auto")
    if ps._embed_fn is ps._embed_via_sie:
        print(f"   FAIL: cascade picked SIE despite unreachable URL")
        fail = True
    else:
        print(f"   OK: cascade did not pick SIE (_embed_fn={ps._embed_fn})")

    # ── [3] SIE returns bad shape → cascade falls through ──
    print("\n[3] SIE returns malformed body → cascade does not pick SIE")
    with mock_sie_server(bad_shape=True) as (url, _):
        ps._config = {"sie_url": url, "sie_model": "x"}
        ps._pion = None
        ps._embed_fn = None
        ps._init_embedder(provider="auto")
        if ps._embed_fn is ps._embed_via_sie:
            print(f"   FAIL: cascade picked SIE despite malformed response")
            fail = True
        else:
            print(f"   OK: cascade rejected SIE on malformed body")

    # ── [4] SIE returns non-200 → cascade falls through ──
    print("\n[4] SIE returns HTTP 500 → cascade does not pick SIE")
    with mock_sie_server(status=500) as (url, _):
        ps._config = {"sie_url": url, "sie_model": "x"}
        ps._pion = None
        ps._embed_fn = None
        ps._init_embedder(provider="auto")
        if ps._embed_fn is ps._embed_via_sie:
            print(f"   FAIL: cascade picked SIE despite HTTP 500")
            fail = True
        else:
            print(f"   OK: cascade rejected SIE on HTTP 500")

    # ── [5] --embed-backend sie with no sie_url → embedder disabled (strict) ──
    print("\n[5] provider='sie' with no sie_url → embedder disabled (strict mode)")
    ps._config = {"sie_url": None, "sie_model": "x"}
    ps._pion = None
    ps._embed_fn = lambda t: None  # poison; must be cleared
    ps._init_embedder(provider="sie")
    if ps._embed_fn is None and ps._config.get("embed_backend") == "none":
        print("   OK: _embed_fn=None, embed_backend='none'")
    else:
        print(f"   FAIL: _embed_fn={ps._embed_fn}, embed_backend={ps._config.get('embed_backend')}")
        fail = True

    # ── [6] --embed-backend sie unreachable → strict (no fall-through) ──
    print("\n[6] provider='sie' with unreachable URL → strict (no fall-through to ollama)")
    ps._config = {"sie_url": "http://127.0.0.1:1", "sie_model": "x"}
    ps._pion = None
    ps._embed_fn = None
    ps._init_embedder(provider="sie")
    if ps._embed_fn is ps._embed_via_ollama:
        print("   FAIL: provider='sie' fell through to ollama")
        fail = True
    elif ps._embed_fn is None and ps._config.get("embed_backend") == "none":
        print("   OK: strict mode held; embed_backend='none'")
    else:
        print(f"   FAIL: unexpected state _embed_fn={ps._embed_fn} "
              f"embed_backend={ps._config.get('embed_backend')}")
        fail = True

    # ── [7] embed_calls_sie counter increments on successful embeds ──
    print("\n[7] embed_calls_sie increments per successful _embed_via_sie call")
    with mock_sie_server() as (url, _):
        ps._config = {"sie_url": url, "sie_model": "BAAI/bge-small-en-v1.5"}
        ps._pion = None
        ps._embed_fn = None
        ps._stats["embed_calls_sie"] = 0
        ps._stats["embed_calls_ollama"] = 0
        ps._stats["embed_calls_pion"] = 0
        ps._init_embedder(provider="sie")
        for _ in range(3):
            ps._embed_via_sie("hello")
        if ps._stats["embed_calls_sie"] == 3 and ps._stats["embed_calls_ollama"] == 0:
            print(f"   OK: embed_calls_sie=3, embed_calls_ollama=0")
        else:
            print(f"   FAIL: embed_calls_sie={ps._stats['embed_calls_sie']}, "
                  f"embed_calls_ollama={ps._stats['embed_calls_ollama']}")
            fail = True

    # ── [8] embed_backend is exposed in _config after _init_embedder ──
    print("\n[8] _config['embed_backend'] set to 'sie' after successful selection")
    if ps._config.get("embed_backend") == "sie":
        print("   OK: embed_backend='sie'")
    else:
        print(f"   FAIL: embed_backend={ps._config.get('embed_backend')}")
        fail = True

    print(f"\n{'PASS' if not fail else 'FAIL'} — gh #69 SIE embedding backend cascade")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
