#!/usr/bin/env python3
"""Replay a recorded coding-agent session against local model servers and
measure how much of each prompt each server recomputes.

A coding agent's prompt only grows, so a server's prompt cache decides how
much of every request is prefilled again. This replays the SAME recorded
requests (a Claude Code session: Anthropic /v1/messages bodies; or a Codex
session: OpenAI /v1/responses bodies; as the client sent them) against each
server, greedy, and records per request:

  prompt    prompt tokens, as the server counts them
  cached    tokens the server did not recompute, from the streamed usage
            (cache_read_input_tokens); oMLX's own log is recorded beside it as
            a cross-check
  ttft_s    seconds from sending the request to the first content event of
            the streamed reply, measured on the client
  total_s   seconds to the end of the stream

The session runs in three phases:

  session   the first --restart-after requests, server started fresh
  restart   the server process is stopped and started again (model reloaded,
            in-memory cache gone), then the rest of the session's requests.
            restart_s is the time from the start command until the server
            answers; servers that load the model on the first request (oMLX,
            Ollama) pay that load inside the next request's first token, so
            the table adds the two
  second    one request from a SECOND session of the same client on the same
            repository: same system prompt and tools, a new first message
            (and, for Claude Code, a different per-session billing header)

Servers (each started and stopped by this script):

  pion      `pion-vllm-mlx serve` with a pion-server --kvcache it starts
  mlx       the same serve module with --no-pion --no-normalize: mlx-lm's
            own server and prompt cache behind serve's Anthropic translation,
            which is how Claude Code reaches stock mlx_lm.server at all
  omlx      `omlx serve` with its paged SSD cache in a scratch directory
  ollama    `ollama serve` (the model must be pulled already)
  lmstudio  `lms server start` + `lms load` (LM Studio installed)

Every request is sent as recorded, except: stream on, temperature 0, a fixed
max_tokens (--max-tokens), Anthropic-only fields a local server does not
implement (thinking, context_management, output_config, metadata) removed,
and Claude Code's mid-conversation `role: "system"` messages folded into the
user turn before them (--no-fold sends them as recorded). The fold makes every
server render the same prompt: as recorded, Ollama renders a trailing system
message in a way that drops the tool definitions (a 16K-token request becomes
3.6K), while pion-vllm-mlx serve folds it the same way itself.

    python3 agent_session_ab.py --server pion --model mlx-community/Llama-3.2-1B-Instruct-4bit \\
        --trace claude.jsonl.gz --second second.jsonl.gz --n 20 --restart-after 10 --out pion.json
    python3 agent_session_ab.py --summarize pion.json mlx.json omlx.json ...
    python3 agent_session_ab.py --public pion.json --public-out results/pion.json

The trace. A recorded session is not shipped with this script: its requests
carry the client's own system prompt and the recording machine's paths. Record
one with the proxy mode, in front of any server that answers /v1/messages:

    python3 agent_session_ab.py --record 8090 127.0.0.1:8080 claude.jsonl
    ANTHROPIC_BASE_URL=http://127.0.0.1:8090 claude -p "..."     # then --continue, one task at a time

--public writes a copy of a result without reply text (replies echo the
recording machine's paths): token counts, timings, and a SHA-256 of each reply,
so identical replies across servers can still be checked.

It is a timing measurement: run it on an idle machine, one server at a time,
with nothing downloading (LM Studio resumes queued downloads when its service
starts; clear them first).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PION_REPO", HERE.parents[1] if (HERE.parents[1] / "pion-vllm-mlx").exists()
                           else Path.home() / "Projects/pion"))
DROP = ("thinking", "context_management", "output_config", "metadata")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(port: int, path: str, timeout: float, proc=None) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            c.request("GET", path)
            r = c.getresponse()
            r.read()
            if r.status < 500:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def load_trace(path: str, n: int) -> tuple[str, list[dict]]:
    """("anthropic" | "responses", the first n agent requests) of a recorded
    session: Claude Code's /v1/messages calls that carry tools (its side calls,
    such as title generation, carry none), or Codex's /v1/responses calls."""
    op = gzip.open if path.endswith(".gz") else open
    out, api = [], None
    for line in op(path, "rt"):
        r = json.loads(line)
        b = r.get("body")
        if r.get("method") != "POST" or not isinstance(b, dict):
            continue
        p = r.get("path", "")
        if "/v1/messages" in p and "count_tokens" not in p and b.get("tools"):
            api = api or "anthropic"
            out.append(b)
        elif p.split("?")[0].endswith("/responses"):
            api = api or "responses"
            out.append(b)
        if len(out) >= n:
            break
    return api or "anthropic", out


# ── servers ────────────────────────────────────────────────────────────────

class Server:
    name = "?"
    log_path: Path

    def __init__(self, a, work: Path):
        self.a, self.work = a, work
        self.port = free_port()
        self.proc = None
        self.log_path = work / f"{self.name}.log"

    def model_id(self) -> str:
        return self.a.model

    def start(self):
        raise NotImplementedError

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.proc = None

    def close(self):
        self.stop()

    def _spawn(self, cmd, env=None, cwd=None):
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, cwd=cwd or self.work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def log_mark(self) -> int:
        return self.log_path.stat().st_size if self.log_path.exists() else 0

    def cached(self, usage: dict, mark: int) -> tuple[int | None, str]:
        c = usage.get("cache_read_input_tokens")
        return (c, "usage.cache_read_input_tokens") if c is not None else (None, "not reported")

    def prompt_of(self, usage: dict) -> int | None:
        """Anthropic semantics: the prompt is input_tokens + cache reads + cache
        writes. oMLX streams the new tokens as cache_creation_input_tokens with
        input_tokens 0; the others leave cache writes at 0."""
        inp = usage.get("input_tokens")
        if inp is None:
            return None
        return inp + (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)


class PionServe(Server):
    name = "pion"
    extra: list[str] = []

    def __init__(self, a, work):
        super().__init__(a, work)
        self.pion = None
        self.pion_port = None

    def start(self):
        if self.name == "pion" and self.pion is None:
            self.pion_port = free_port()
            pdir = self.work / "pion-data"
            pdir.mkdir(exist_ok=True)
            plog = open(self.work / "pion-server.log", "ab")
            self.pion = subprocess.Popen([self.a.pion_binary, "--kvcache", "-w", "1", "-p", str(self.pion_port),
                                          "--no-auto-detect", "--no-auto-embed"], cwd=pdir, stdout=plog,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(120):
                try:
                    socket.create_connection(("127.0.0.1", self.pion_port), timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.25)
        env = dict(os.environ, PYTHONPATH=str(REPO / "pion-vllm-mlx"))
        cmd = [sys.executable, "-m", "pion_vllm_mlx.serve", "--model", self.a.model, "--host", "127.0.0.1",
               "--port", str(self.port), "--max-tokens", str(self.a.max_tokens), *self.extra]
        if self.name == "pion":
            cmd += ["--pion-port", str(self.pion_port), "--pion-budget-gb", "6"]
        cmd += self.a.serve_arg or []
        if self.name == "mlx" and self.a.no_cache:
            cmd += ["--prompt-cache-size", "0"]
        self._spawn(cmd, env=env)
        return wait_http(self.port, "/v1/models", 300, self.proc)

    def close(self):
        self.stop()
        if self.pion is not None and self.pion.poll() is None:
            os.killpg(self.pion.pid, signal.SIGTERM)
            self.pion.wait(timeout=30)


class MlxStock(PionServe):
    name = "mlx"
    extra = ["--no-pion", "--no-normalize"]


class OMLX(Server):
    name = "omlx"

    def model_id(self):
        return Path(self.a.model).name

    def start(self):
        mdir = self.work / "omlx-models"
        mdir.mkdir(exist_ok=True)
        link = mdir / self.model_id()
        if not link.exists():
            snap = sorted((Path.home() / ".cache/huggingface/hub" / ("models--" + self.a.model.replace("/", "--"))
                           / "snapshots").iterdir())[0]
            link.symlink_to(snap)
        ssd = self.work / "omlx-ssd"
        ssd.mkdir(exist_ok=True)
        cmd = ["omlx", "serve", "--model-dir", str(mdir), "--port", str(self.port), "--paged-ssd-cache-dir", str(ssd),
               "--paged-ssd-cache-max-size", "2GB", "--base-path", str(self.work / "omlx-base"), "--no-hf-cache",
               "--log-level", "info", "--max-concurrent-requests", "1"]
        if self.a.no_cache:
            cmd = [c for c in cmd if c not in ("--paged-ssd-cache-dir", str(ssd), "--paged-ssd-cache-max-size", "2GB")]
            cmd.append("--no-cache")
        self._spawn(cmd)
        return wait_http(self.port, "/v1/models", 300, self.proc)

    RESTORE = re.compile(r"Prefix cache restore for \S+: source=(\S+) cached=(\d+) suffix=(\d+)")

    def cached(self, usage, mark):
        # Streamed usage carries the cache read; the scheduler's log line is a
        # cross-check (its non-streamed /v1/messages usage reported 0 on a hit
        # in 0.7.0).
        time.sleep(0.3)
        with open(self.log_path, "rb") as f:
            f.seek(mark)
            text = f.read().decode(errors="replace")
        hits = self.RESTORE.findall(text)
        logged = int(hits[-1][1]) if hits else 0
        where = f"log: restore source={hits[-1][0]}" if hits else "log: no restore"
        c = usage.get("cache_read_input_tokens")
        if c is None:
            return logged, where
        return c, f"usage.cache_read_input_tokens ({where} cached={logged})"


class Ollama(Server):
    name = "ollama"

    def start(self):
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{self.port}", OLLAMA_KEEP_ALIVE="30m",
                   OLLAMA_CONTEXT_LENGTH=str(self.a.ctx))
        self._spawn(["ollama", "serve"], env=env)
        ok = wait_http(self.port, "/api/version", 120, self.proc)
        return ok


class LMStudio(Server):
    name = "lmstudio"
    LMS = str(Path.home() / ".lmstudio/bin/lms")

    def _lms(self, *args, timeout=600):
        with open(self.log_path, "ab") as log:
            return subprocess.run([self.LMS, *args], stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode

    def model_id(self):
        return self.a.lms_model or self.a.model

    def start(self):
        self._lms("server", "start", "--port", str(self.port))
        self._lms("load", self.model_id(), "--context-length", str(self.a.ctx), "--yes")
        return wait_http(self.port, "/v1/models", 300)

    def stop(self):
        self._lms("unload", "--all")
        self._lms("server", "stop")

    def close(self):
        self.stop()

    def prompt_of(self, usage):
        # LM Studio's input_tokens is the whole prompt, the cached part included.
        return usage.get("input_tokens")


SERVERS = {c.name: c for c in (PionServe, MlxStock, OMLX, Ollama, LMStudio)}


# ── one request ───────────────────────────────────────────────────────────

def fold_system(messages: list[dict]) -> list[dict]:
    """Mid-conversation system messages as user text, merged into the user turn
    before them, so roles still alternate."""
    def blocks(c):
        return [{"type": "text", "text": c}] if isinstance(c, str) else list(c or [])
    out: list[dict] = []
    for m in messages:
        role = "user" if m.get("role") == "system" else m.get("role")
        if out and out[-1]["role"] == role == "user":
            out[-1] = {"role": "user", "content": blocks(out[-1]["content"]) + blocks(m.get("content"))}
        else:
            out.append({"role": role, "content": m.get("content")})
    return out


FOLD = True


def send(server: Server, body: dict, max_tokens: int) -> dict:
    b = {k: v for k, v in body.items() if k not in DROP}
    if FOLD:
        b["messages"] = fold_system(b.get("messages") or [])
    b.update(model=server.model_id(), max_tokens=max_tokens, temperature=0, stream=True)
    mark = server.log_mark()
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3600)
    c.request("POST", "/v1/messages", body=json.dumps(b),
              headers={"Content-Type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
    r = c.getresponse()
    rec = {"status": r.status}
    if r.status != 200:
        rec["error"] = r.read().decode(errors="replace")[:1500]
        c.close()
        return rec
    ttft = None
    usage: dict = {}
    text, thinking, tools = [], [], []
    stream_error = None
    event = None
    head: list[str] = []
    for raw in r:
        line = raw.decode(errors="replace").rstrip("\r\n")
        if len(head) < 40:
            head.append(line[:300])
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        typ = d.get("type") or event
        if typ == "message_start":
            usage.update((d.get("message") or {}).get("usage") or {})
        elif typ == "content_block_start":
            cb = d.get("content_block") or {}
            if cb.get("type") == "tool_use":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tools.append({"name": cb.get("name"), "input": ""})
        elif typ == "content_block_delta":
            dl = d.get("delta") or {}
            piece = dl.get("text") or dl.get("partial_json") or dl.get("thinking") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            if dl.get("type") == "text_delta":
                text.append(dl.get("text", ""))
            elif dl.get("type") == "thinking_delta":
                thinking.append(dl.get("thinking", ""))
            elif dl.get("type") == "input_json_delta" and tools:
                tools[-1]["input"] += dl.get("partial_json", "")
        elif typ == "message_delta":
            usage.update(d.get("usage") or {})
        elif typ == "message_stop":
            break
        elif typ == "error":
            stream_error = (d.get("error") or {}).get("message") or str(d)[:300]
            break
    c.close()
    total = time.perf_counter() - t0
    cached, src = server.cached(usage, mark)
    prompt = server.prompt_of(usage)
    if ttft is None:
        rec["raw_head"] = head          # no content event found: keep the stream's start to see why
    rec.update(prompt=prompt, cached=cached, cached_source=src, ttft_s=None if ttft is None else round(ttft, 3),
               total_s=round(total, 3), text="".join(text), thinking="".join(thinking), tools=tools, usage=usage)
    if stream_error is not None:
        rec["stream_error"] = stream_error
    if prompt is not None and cached is not None:
        rec["processed"] = max(0, prompt - cached)
    return rec


# ── recording proxy ───────────────────────────────────────────────────────

def record(listen_port: int, upstream: str, out: str) -> None:
    """Log every request body a client sends, forward it unchanged, and stream
    the reply back unchanged (one JSON line per request)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    up_host, up_port = upstream.rsplit(":", 1)
    lock = threading.Lock()
    hop = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
           "proxy-authorization", "proxy-authenticate", "host", "content-length"}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _fwd(self):
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n) if n else b""
            rec = {"t": time.time(), "method": self.command, "path": self.path}
            try:
                rec["body"] = json.loads(body) if body else None
            except json.JSONDecodeError:
                rec["body_raw"] = body.decode("utf-8", "replace")
            c = http.client.HTTPConnection(up_host, int(up_port), timeout=3600)
            c.request(self.command, self.path, body=body,
                      headers={k: v for k, v in self.headers.items() if k.lower() not in hop})
            r = c.getresponse()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in hop:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.close_connection = True
            rec["status"] = r.status
            with lock, open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")

        do_POST = do_GET = _fwd

    print(f"recording {listen_port} -> {upstream} into {out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", listen_port), H).serve_forever()


# ── summary and the public copy ───────────────────────────────────────────

LABEL = {"mlx": "mlx-lm server, stock", "ollama": "Ollama", "lmstudio": "LM Studio",
         "omlx": "oMLX", "pion": "Pion serve"}


def _ttft(r) -> float:
    # An empty reply has no content event; its stream end bounds the first token.
    return r["ttft_s"] if r.get("ttft_s") is not None else r["total_s"]


def summarize(paths: list[str]) -> None:
    import statistics

    def share(rs):
        p = sum(r.get("prompt") or 0 for r in rs)
        return sum(r.get("cached") or 0 for r in rs) / p if p else float("nan")

    runs = {}
    for path in paths:
        d = json.load(open(path))
        runs[d["server"]] = d
    print("| Server | first request, cold | within the session: reused, TTFT "
          "| **after a restart**: recomputed; server start + first token "
          "| **a second session**: reused, TTFT |")
    print("|---|---:|---:|---:|---:|")
    for name in [n for n in ("mlx", "ollama", "lmstudio", "omlx", "pion") if n in runs]:
        d = runs[name]
        rq, k = d["requests"], d["restart_after"]
        if any(r["status"] != 200 for r in rq):
            print(f"| {LABEL[name]} | {sum(r['status'] != 200 for r in rq)} requests failed | | | |")
            continue
        sess = [r for r in rq if r["phase"] in ("session", "restart") and r["i"] not in (0, k)]
        rs = next(r for r in rq if r["i"] == k)
        sec = next((r for r in rq if r["phase"] == "second"), None)
        errs = sum(1 for r in rq if r.get("stream_error"))
        sess_ok = [r for r in sess if r.get("prompt") is not None]
        recomputed = (f"{rs['processed']:,} of {rs['prompt']:,}" if rs.get("processed") is not None
                      else "not reported (stream error)")
        row = (f"| {LABEL[name]}{f' ({errs} of {len(rq)} replies ended in a stream error)' if errs else ''} "
               f"| {rq[0]['prompt']:,} tokens, {_ttft(rq[0]):.2f} s "
               f"| {share(sess_ok):.1%}, {statistics.median(_ttft(r) for r in sess):.2f} s "
               f"| {recomputed}; {d.get('restart_s', 0):.1f} + {_ttft(rs):.2f} s "
               f"= **{d.get('restart_s', 0) + _ttft(rs):.1f} s** ")
        row += f"| {sec['cached']:,} of {sec['prompt']:,}, {_ttft(sec):.2f} s |" if sec else "| — |"
        print(row)


def public_copy(path: str, out: str) -> None:
    d = json.load(open(path))
    for r in d["requests"]:
        reply = json.dumps({"text": r.pop("text", ""), "thinking": r.pop("thinking", ""),
                            "tools": r.pop("tools", [])}, sort_keys=True)
        r["reply_sha256"] = hashlib.sha256(reply.encode()).hexdigest()
        r["empty_reply"] = r.get("ttft_s") is None
        r.pop("error", None)
        r.pop("raw_head", None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(d, indent=1) + "\n")


def versions(server: str) -> dict:
    out = {}
    try:
        import mlx.core as mx
        import mlx_lm
        out.update(mlx=mx.__version__, mlx_lm=mlx_lm.__version__)
    except ImportError:
        pass
    probes = {"omlx": ["omlx", "--version"], "ollama": ["ollama", "--version"],
              "lmstudio": [LMStudio.LMS, "version"]}
    if server in probes:
        try:
            r = subprocess.run(probes[server], capture_output=True, text=True, timeout=30)
            out[server] = (r.stdout + r.stderr).strip().splitlines()[-1][:120]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return out


DROP_RESPONSES = ("reasoning", "include", "store", "prompt_cache_key", "client_metadata")


def send_responses(server: Server, body: dict, max_tokens: int) -> dict:
    """One Codex request (OpenAI Responses API), streamed. OpenAI usage
    semantics: input_tokens is the whole prompt, cached_tokens the part reused."""
    b = {k: v for k, v in body.items() if k not in DROP_RESPONSES}
    b.update(model=server.model_id(), max_output_tokens=max_tokens, temperature=0, stream=True)
    mark = server.log_mark()
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3600)
    c.request("POST", "/v1/responses", body=json.dumps(b),
              headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    r = c.getresponse()
    rec = {"status": r.status}
    if r.status != 200:
        rec["error"] = r.read().decode(errors="replace")[:1500]
        c.close()
        return rec
    ttft = None
    usage: dict = {}
    text, thinking, tools, head = [], [], [], []
    stream_error = None
    for raw in r:
        line = raw.decode(errors="replace").rstrip("\r\n")
        if len(head) < 40:
            head.append(line[:300])
        if not line.startswith("data:"):
            continue
        try:
            d = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        typ = d.get("type", "")
        if typ in ("response.output_text.delta", "response.function_call_arguments.delta",
                   "response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            if ttft is None and d.get("delta"):
                ttft = time.perf_counter() - t0
            if typ == "response.output_text.delta":
                text.append(d.get("delta", ""))
            elif typ == "response.function_call_arguments.delta":
                if tools:
                    tools[-1]["input"] += d.get("delta", "")
            else:
                thinking.append(d.get("delta", ""))
        elif typ == "response.output_item.added":
            it = d.get("item") or {}
            if it.get("type") == "function_call":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tools.append({"name": it.get("name"), "input": ""})
        elif typ in ("response.completed", "response.incomplete"):
            usage = (d.get("response") or {}).get("usage") or {}
            break
        elif typ in ("response.failed", "error"):
            stream_error = str(d.get("error") or (d.get("response") or {}).get("error") or d)[:300]
            break
    c.close()
    total = time.perf_counter() - t0
    reported = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    cached, src = server.cached({"cache_read_input_tokens": reported} if reported is not None else {}, mark)
    prompt = usage.get("input_tokens")
    rec.update(prompt=prompt, cached=cached, cached_source=src.replace("cache_read_input_tokens", "input_tokens_details.cached_tokens"),
               ttft_s=None if ttft is None else round(ttft, 3), total_s=round(total, 3), text="".join(text),
               thinking="".join(thinking), tools=tools, usage=usage)
    if ttft is None:
        rec["raw_head"] = head
    if stream_error is not None:
        rec["stream_error"] = stream_error
    if prompt is not None and cached is not None:
        rec["processed"] = max(0, prompt - cached)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", choices=sorted(SERVERS))
    ap.add_argument("--model", help="HF repo id (MLX servers) or the server's model name")
    ap.add_argument("--record", nargs=3, metavar=("PORT", "UPSTREAM", "OUT"), help="run the recording proxy")
    ap.add_argument("--summarize", nargs="+", metavar="RESULT", help="print the table for these results")
    ap.add_argument("--public", metavar="RESULT", help="write a copy without reply text to --public-out")
    ap.add_argument("--public-out")
    ap.add_argument("--lms-model", help="LM Studio model key, when it differs from --model")
    ap.add_argument("--trace", help="recorded Claude Code requests (JSONL, optionally .gz)")
    ap.add_argument("--second", help="a second session's recorded requests; its first is the probe")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--restart-after", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=65536, help="context length for servers that take one")
    ap.add_argument("--pion-binary", default=str(REPO / "pion-server"))
    ap.add_argument("--out")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-fold", action="store_true", help="send mid-conversation system messages as recorded")
    ap.add_argument("--min-free-gb", type=float, default=2.5, help="abort below this much free disk")
    ap.add_argument("--serve-arg", action="append", help="pion and mlx: one more flag for pion-vllm-mlx serve (repeatable)")
    ap.add_argument("--no-cache", action="store_true",
                    help="oMLX and mlx: run with the prompt cache off (every request cold), the reference "
                         "for checking that cached replies are identical")
    a = ap.parse_args()
    global FOLD
    FOLD = not a.no_fold
    if a.record:
        record(int(a.record[0]), a.record[1], a.record[2])
        return 0
    if a.summarize:
        summarize(a.summarize)
        return 0
    if a.public:
        public_copy(a.public, a.public_out or a.public.replace(".json", ".public.json"))
        return 0
    if not (a.server and a.model and a.trace and a.out):
        ap.error("--server, --model, --trace and --out are required to run a session")

    api, reqs = load_trace(a.trace, a.n)
    probe = load_trace(a.second, 1)[1][0] if a.second else None
    sender = send_responses if api == "responses" else send
    work = Path(tempfile.mkdtemp(prefix=f"agent_ab_{a.server}_"))
    srv = SERVERS[a.server](a, work)
    report = {"server": a.server, "model": a.model, "measured": time.strftime("%Y-%m-%d %H:%M"),
              "n": len(reqs), "restart_after": a.restart_after, "max_tokens": a.max_tokens,
              "api": api, "fold_system_messages": FOLD, "versions": versions(a.server), "requests": []}

    def run(i, body, phase):
        free = shutil.disk_usage("/").free / 2**30
        if free < a.min_free_gb:
            raise SystemExit(f"ABORT: {free:.1f} GiB free on / (< {a.min_free_gb}); a full disk makes every number wrong")
        rec = sender(srv, body, a.max_tokens)
        rec.update(i=i, phase=phase)
        report["requests"].append(rec)
        print(f"{a.server:8s} {phase:8s} {i:2d}  status {rec['status']}  prompt {rec.get('prompt')}  "
              f"cached {rec.get('cached')}  processed {rec.get('processed')}  ttft {rec.get('ttft_s')}s  "
              f"total {rec.get('total_s')}s  {rec.get('cached_source', '')}"
              + (f"  ERROR {rec.get('error', '')[:200]}" if rec["status"] != 200 else ""), flush=True)
        return rec

    try:
        if not srv.start():
            print(open(srv.log_path, errors="replace").read()[-3000:])
            raise SystemExit(f"{a.server} did not start")
        for i, body in enumerate(reqs):
            if i == a.restart_after:
                srv.stop()
                t0 = time.perf_counter()
                if not srv.start():
                    raise SystemExit(f"{a.server} did not restart")
                report["restart_s"] = round(time.perf_counter() - t0, 1)
            run(i, body, "session" if i < a.restart_after else "restart")
        if probe is not None:
            run(len(reqs), probe, "second")
    finally:
        srv.close()
        if a.keep:
            print("work dir:", work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
