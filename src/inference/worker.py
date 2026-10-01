"""Pion M1 Inference Worker — in-process model serving via Unix domain socket.

Spawned by Pion at startup when --inference is passed.
Loads embedding and LLM models via HuggingFace transformers + PyTorch.
Communicates with Pion over a binary protocol on a Unix domain socket.

Standalone usage:
    python3 src/inference/worker.py --socket /tmp/pion_inference.sock

Protocol (little-endian):
    Request:  [type:1B][req_id:4B][body_len:4B][body]
    Response: [type:1B][req_id:4B][status:1B][body_len:4B][body]

Types: EMBED=1, GENERATE=2, LOAD_MODEL=3, HEALTH=4, SHUTDOWN=5
Status: OK=0, ERROR=1
"""

import socket
import struct
import os
import sys
import signal
import argparse
import time
import threading

# Message types
MSG_EMBED = 1
MSG_GENERATE = 2
MSG_LOAD_MODEL = 3
MSG_HEALTH = 4
MSG_SHUTDOWN = 5

# Model types
MODEL_EMBEDDING = 0
MODEL_LLM = 1

# Status codes
STATUS_OK = 0
STATUS_ERROR = 1

HEADER_SIZE = 9  # type(1) + req_id(4) + body_len(4)
RESP_HEADER_SIZE = 10  # type(1) + req_id(4) + status(1) + body_len(4)


def _watch_parent():
    """Exit when the parent (Pion) dies (gh #424).

    The sidecar is fork+exec'd and outlived every server exit — a clean
    SIGTERM, an uncatchable SIGKILL, and the FATAL-bind exit all left a
    ~450 MB torch process and its socket behind, one per restart. Polling
    getppid() catches all of those, including the SIGKILL the server-side
    kill cannot cover: when the parent is gone the process is reparented to
    launchd/init (PID 1)."""
    orig = os.getppid()
    while True:
        time.sleep(2.0)
        if os.getppid() != orig or os.getppid() == 1:
            os._exit(0)


def _maybe_offline(model_id: str):
    """Set HF/transformers offline when the model is already cached (gh #424).

    On a default start with the model cached, transformers still opened HTTPS to
    the Hugging Face CDN (and printed a rate-limit warning) — an unconfigured
    third-party contact on every start, against the 'connects out only to what
    you enable' posture. Going offline when the snapshot is on disk keeps the
    first-use download working while making the steady state silent. An explicit
    HF_HUB_OFFLINE in the environment is respected either way. Must run BEFORE
    transformers is imported for the env vars to take effect."""
    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(model_id, "config.json")
        if isinstance(hit, str) and os.path.exists(hit):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
    except Exception:
        pass


class ModelRegistry:
    """Thread-safe model registry for embedding and LLM models."""

    def __init__(self):
        self.embedding_model = None  # (tokenizer, model, dim)
        self.llm_model = None  # (tokenizer, model)
        self.lock = threading.Lock()

    def load_embedding(self, model_id: str) -> tuple:
        """Download from HF if needed, load with transformers. Returns (ok, message)."""
        try:
            _maybe_offline(model_id)  # gh #424: no phone-home when cached
            from transformers import AutoTokenizer, AutoModel
            import torch

            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            model.eval()

            # Determine dim from a dummy forward pass
            dummy = tok("test", return_tensors="pt", truncation=True, max_length=32)
            with torch.no_grad():
                out = model(**dummy)
            dim = out.last_hidden_state.shape[-1]

            with self.lock:
                self.embedding_model = (tok, model, dim)

            elapsed = time.time() - t0
            return True, f"loaded {model_id} (dim={dim}) in {elapsed:.1f}s"
        except Exception as e:
            return False, str(e)

    def load_llm(self, model_id: str) -> tuple:
        """Load a causal LM for text generation. Returns (ok, message)."""
        try:
            _maybe_offline(model_id)  # gh #424: no phone-home when cached
            from transformers import AutoTokenizer, AutoModelForCausalLM

            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype="auto"
            )
            model.eval()

            with self.lock:
                self.llm_model = (tok, model)

            elapsed = time.time() - t0
            return True, f"loaded {model_id} in {elapsed:.1f}s"
        except Exception as e:
            return False, str(e)

    def embed(self, text: str) -> bytes | None:
        """Mean-pool last_hidden_state, L2-normalize, return [dim:u32][f32 * dim]."""
        import torch

        with self.lock:
            if not self.embedding_model:
                return None
            tok, model, dim = self.embedding_model

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**inputs)
        # Mean pooling over token positions
        token_emb = out.last_hidden_state  # [1, seq_len, dim]
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        # L2 normalize
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.squeeze(0)  # [dim]

        return struct.pack("<I", dim) + emb.numpy().tobytes()

    def generate(self, prompt: str, context: str, max_tokens: int) -> str | None:
        """Generate text with optional context prefix. Returns generated text."""
        import torch

        with self.lock:
            if not self.llm_model:
                return None
            tok, model = self.llm_model

        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        inputs = tok(
            full_prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        # Decode only the generated tokens (skip the input)
        generated = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated


def _read_exact(conn, n: int) -> bytes | None:
    """Read exactly n bytes from socket. Returns None on disconnect."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_lenprefix_str(body: bytes, offset: int) -> tuple:
    """Read [len:u32][bytes] from body at offset. Returns (string, new_offset)."""
    slen = struct.unpack_from("<I", body, offset)[0]
    offset += 4
    s = body[offset : offset + slen].decode("utf-8", errors="replace")
    return s, offset + slen


def _build_response(msg_type: int, req_id: int, status: int, body: bytes) -> bytes:
    """Build a response frame."""
    return struct.pack("<BIbI", msg_type, req_id, status, len(body)) + body


def _build_error(msg_type: int, req_id: int, err_msg: str) -> bytes:
    """Build an error response."""
    err_bytes = err_msg.encode("utf-8")
    body = struct.pack("<I", len(err_bytes)) + err_bytes
    return _build_response(msg_type, req_id, STATUS_ERROR, body)


def _build_ok(msg_type: int, req_id: int, body: bytes = b"") -> bytes:
    """Build a success response."""
    return _build_response(msg_type, req_id, STATUS_OK, body)


class InferenceServer:
    """Unix socket server for inference requests."""

    def __init__(self, socket_path: str, default_emb_model: str = "", default_llm_model: str = ""):
        self.socket_path = socket_path
        self.registry = ModelRegistry()
        self.running = True
        self.default_emb_model = default_emb_model
        self.default_llm_model = default_llm_model

    def start(self):
        """Start the server: load default models, listen for connections."""
        # Clean up stale socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Auto-load default embedding model
        if self.default_emb_model:
            print(
                f"[inference] Loading default embedding model: {self.default_emb_model}",
                flush=True,
            )
            ok, msg = self.registry.load_embedding(self.default_emb_model)
            print(f"[inference] Embedding: {msg}", flush=True)

        # Auto-load default LLM if specified
        if self.default_llm_model:
            print(
                f"[inference] Loading default LLM: {self.default_llm_model}",
                flush=True,
            )
            ok, msg = self.registry.load_llm(self.default_llm_model)
            print(f"[inference] LLM: {msg}", flush=True)

        # Create and bind Unix socket. gh #424: bind under umask 077 so the
        # socket is owner-only (0700) — it used to be 0755, world-connectable.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _old_umask = os.umask(0o077)
        try:
            srv.bind(self.socket_path)
        finally:
            os.umask(_old_umask)
        try:
            os.chmod(self.socket_path, 0o700)
        except OSError:
            pass
        srv.listen(4)
        srv.settimeout(1.0)

        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "running", False))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "running", False))

        print(f"[inference] Listening on {self.socket_path}", flush=True)

        while self.running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle_connection(conn)
            except Exception as e:
                print(f"[inference] Connection error: {e}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        srv.close()
        try:
            os.unlink(self.socket_path)
        except Exception:
            pass
        print("[inference] Shutdown complete", flush=True)

    def _handle_connection(self, conn):
        """Process requests on a persistent connection."""
        while self.running:
            hdr = _read_exact(conn, HEADER_SIZE)
            if not hdr:
                break

            msg_type = hdr[0]
            req_id = struct.unpack_from("<I", hdr, 1)[0]
            body_len = struct.unpack_from("<I", hdr, 5)[0]

            body = _read_exact(conn, body_len) if body_len > 0 else b""
            if body is None:
                break

            resp = self._dispatch(msg_type, req_id, body)
            conn.sendall(resp)

            if msg_type == MSG_SHUTDOWN:
                self.running = False
                break

    def _dispatch(self, msg_type: int, req_id: int, body: bytes) -> bytes:
        """Route a request to the appropriate handler."""
        try:
            if msg_type == MSG_EMBED:
                return self._handle_embed(req_id, body)
            elif msg_type == MSG_GENERATE:
                return self._handle_generate(req_id, body)
            elif msg_type == MSG_LOAD_MODEL:
                return self._handle_load_model(req_id, body)
            elif msg_type == MSG_HEALTH:
                return self._handle_health(req_id)
            elif msg_type == MSG_SHUTDOWN:
                return _build_ok(MSG_SHUTDOWN, req_id, b"bye")
            else:
                return _build_error(msg_type, req_id, f"unknown msg_type={msg_type}")
        except Exception as e:
            return _build_error(msg_type, req_id, str(e))

    def _handle_embed(self, req_id: int, body: bytes) -> bytes:
        """Handle EMBED request: [text_len:u32][text_bytes]."""
        text, _ = _read_lenprefix_str(body, 0)
        result = self.registry.embed(text)
        if result is None:
            return _build_error(MSG_EMBED, req_id, "no embedding model loaded")
        return _build_ok(MSG_EMBED, req_id, result)

    def _handle_generate(self, req_id: int, body: bytes) -> bytes:
        """Handle GENERATE request: [prompt_len:u32][prompt][ctx_len:u32][ctx][max_tokens:u32]."""
        off = 0
        prompt, off = _read_lenprefix_str(body, off)
        context, off = _read_lenprefix_str(body, off)
        max_tokens = struct.unpack_from("<I", body, off)[0] if off < len(body) else 256

        result = self.registry.generate(prompt, context, max_tokens)
        if result is None:
            return _build_error(MSG_GENERATE, req_id, "no LLM loaded")
        text_bytes = result.encode("utf-8")
        resp_body = struct.pack("<I", len(text_bytes)) + text_bytes
        return _build_ok(MSG_GENERATE, req_id, resp_body)

    def _handle_load_model(self, req_id: int, body: bytes) -> bytes:
        """Handle LOAD_MODEL: [model_id_len:u32][model_id][model_type:u8]."""
        model_id, off = _read_lenprefix_str(body, 0)
        model_type = body[off] if off < len(body) else MODEL_EMBEDDING

        if model_type == MODEL_EMBEDDING:
            ok, msg = self.registry.load_embedding(model_id)
        elif model_type == MODEL_LLM:
            ok, msg = self.registry.load_llm(model_id)
        else:
            return _build_error(MSG_LOAD_MODEL, req_id, f"unknown model_type={model_type}")

        if ok:
            msg_bytes = msg.encode("utf-8")
            resp_body = struct.pack("<I", len(msg_bytes)) + msg_bytes
            return _build_ok(MSG_LOAD_MODEL, req_id, resp_body)
        else:
            return _build_error(MSG_LOAD_MODEL, req_id, msg)

    def _handle_health(self, req_id: int) -> bytes:
        """Health check — return model status."""
        with self.registry.lock:
            has_emb = self.registry.embedding_model is not None
            has_llm = self.registry.llm_model is not None
        status = f"emb={'yes' if has_emb else 'no'},llm={'yes' if has_llm else 'no'}"
        body = status.encode("utf-8")
        return _build_ok(MSG_HEALTH, req_id, body)


def main():
    parser = argparse.ArgumentParser(description="Pion M1 Inference Worker")
    parser.add_argument(
        "--socket",
        default="/tmp/pion_inference.sock",
        help="Unix socket path (default: /tmp/pion_inference.sock)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Default embedding model to load at startup",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        help="Default LLM model to load at startup (empty = none)",
    )
    args = parser.parse_args()

    # gh #424: exit if the parent (Pion) dies, so no orphan torch process
    # survives a server exit. Daemon thread; polls getppid().
    threading.Thread(target=_watch_parent, daemon=True).start()

    server = InferenceServer(
        socket_path=args.socket,
        default_emb_model=args.embedding_model,
        default_llm_model=args.llm_model,
    )
    server.start()


if __name__ == "__main__":
    main()
