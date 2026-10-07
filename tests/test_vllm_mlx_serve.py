#!/usr/bin/env python3
"""`pion-vllm-mlx serve` end to end: the command exists, and a restarted serve
answers from Pion.

serve is the one Pion surface with a measured user-level win (a restarted
local model behind Claude Code answers its first request from the stored
prefix instead of re-prefilling it), and for its first weeks it had no
runnable command and no gate test. This test pins both.

  1. The command. pyproject's [project.scripts] names a callable that
     imports; `pion-vllm-mlx --help`, `--version` and an unknown command
     behave; `serve --help` lists serve's own flags before mlx-lm's.
  2. Translation, no model: an Anthropic request loses Claude Code's
     per-session billing-header line, keeps the system prompt, and maps
     tool_use / tool_result onto OpenAI tool calls; a Responses request maps
     instructions and function calls the same way.
  3. Live, Llama-3.2-1B-Instruct-4bit against the runner's
     `pion-server --kvcache -w 1` on 1974:
       a. first request (a fresh system prompt) is a miss;
       b. a second "session" whose billing header differs reuses the prompt
          from the process cache (normalization is what makes that possible);
       c. serve is SIGKILLed and restarted: the same request is now served
          from Pion (cache_read_input_tokens covers the prompt, /v1/pion/stats
          counts a restore) and the greedy reply is identical to the miss;
       d. Codex's /v1/responses answers over the same cache.
  4. Live, a hybrid model (Gemma-4-E2B: sliding-window layers), when it is in
     the Hugging Face cache: a request that extends the previous one reuses
     the previous prompt in-process. Pion does not store hybrid caches, so
     this is mlx-lm's own segment-boundary reuse, which serve's resume path
     used to switch off (every Gemma request was a full prefill).

    python3 tests/test_vllm_mlx_serve.py [--pion-port 1974] [--offline]

--offline runs parts 1 and 2 only (no model, no server).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "pion-vllm-mlx")
sys.path.insert(0, PKG)

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
HYBRID = "mlx-community/gemma-4-e2b-it-4bit"
fails = 0


def check(name, ok, detail: object = ""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail != "" else ""), flush=True)
    fails += 0 if ok else 1


# ── 1. the command ─────────────────────────────────────────────────────────

def part_command():
    import contextlib
    import importlib
    import io

    meta = tomllib.loads(open(os.path.join(PKG, "pyproject.toml"), encoding="utf-8").read())
    target = meta.get("project", {}).get("scripts", {}).get("pion-vllm-mlx")
    check("pyproject declares the pion-vllm-mlx console script", target == "pion_vllm_mlx.cli:main", target)
    if not target:
        return
    mod, _, fn = target.partition(":")
    main = getattr(importlib.import_module(mod), fn, None)
    check("the console script's target imports and is callable", callable(main))
    if not callable(main):
        return

    def run(*args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main(list(args))
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 0
        return rc, out.getvalue(), err.getvalue()

    rc, out, _ = run("--help")
    check("`pion-vllm-mlx --help` names the serve command", rc == 0 and "serve" in out, rc)
    rc, out, _ = run("--version")
    check("`pion-vllm-mlx --version` prints a version", rc == 0 and out.startswith("pion-vllm-mlx "), out.strip())
    rc, _, err = run("no-such-command")
    check("an unknown command exits 2 with the usage", rc == 2 and "usage:" in err, rc)

    # `serve --help` in a child: mlx-lm's parser exits the process.
    r = subprocess.run([sys.executable, "-m", "pion_vllm_mlx", "serve", "--help"], cwd=PKG,
                       env=dict(os.environ, PYTHONPATH=PKG), capture_output=True, text=True, timeout=120)
    ok = r.returncode == 0 and "--pion-budget-gb" in r.stdout and "--model" in r.stdout
    check("`serve --help` lists serve's flags and mlx-lm's", ok, f"rc={r.returncode} {r.stderr[-300:]}")


# ── 2. translation ─────────────────────────────────────────────────────────

def part_translation():
    from pion_vllm_mlx import serve as S

    sysblocks = [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0; cch=abcd1234;"},
                 {"type": "text", "text": "You are a coding agent."}]
    body = {"model": "claude-sonnet", "system": sysblocks, "max_tokens": 16,
            "tools": [{"name": "Read", "description": "read a file",
                       "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                      {"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {"role": "user", "content": "read a.py"},
                {"role": "assistant", "content": [{"type": "text", "text": "Reading."},
                                                  {"type": "tool_use", "id": "toolu_1", "name": "Read",
                                                   "input": {"path": "a.py"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                              "content": "print(1)"}]}]}
    req = S.anthropic_to_openai(body)
    msgs = req["messages"]
    check("the billing-header line is removed from the system prompt",
          msgs[0]["role"] == "system" and msgs[0]["content"] == "You are a coding agent.", msgs[0])
    check("the request's model is not passed through", req["model"] == "default_model", req["model"])
    calls = msgs[2].get("tool_calls") or []
    check("tool_use becomes an OpenAI tool call",
          msgs[2]["role"] == "assistant" and calls and calls[0]["function"]["name"] == "Read"
          and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}, msgs[2])
    check("tool_result becomes a tool message",
          msgs[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "print(1)"}, msgs[3])
    check("server-side tools (no input_schema) are dropped",
          [t["function"]["name"] for t in req.get("tools", [])] == ["Read"], req.get("tools"))

    rbody = {"model": "gpt-5-codex", "instructions": "You are Codex.", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "a.py"}]}
    r = S.responses_to_openai(rbody)
    roles = [m["role"] for m in r["messages"]]
    check("Responses: instructions become the system prompt",
          r["messages"][0] == {"role": "system", "content": "You are Codex."}, r["messages"][0])
    check("Responses: a function call and its output map to assistant + tool",
          roles[:4] == ["system", "user", "assistant", "tool"]
          and r["messages"][2]["tool_calls"][0]["function"]["name"] == "shell", roles)


# ── 3. live ────────────────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(port, method, path, body=None, timeout=600) -> tuple[int, dict]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path, body=None if body is None else json.dumps(body),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, {"raw": data.decode(errors="replace")[:500]}


class Serve:
    def __init__(self, port, pion_port, workdir, log_path, model=MODEL):
        self.port = port
        self.log = open(log_path, "ab")
        cmd = [sys.executable, "-m", "pion_vllm_mlx", "serve", "--model", model, "--host", "127.0.0.1",
               "--port", str(port), "--pion-port", str(pion_port), "--max-tokens", "64"]
        self.p = subprocess.Popen(cmd, cwd=workdir, env=dict(os.environ, PYTHONPATH=PKG),
                                  stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return False
            try:
                st, _ = http_json(self.port, "GET", "/v1/pion/stats", timeout=5)
                if st == 200:
                    return True
            except OSError:
                pass
            time.sleep(0.5)
        return False

    def kill(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGKILL)
            self.p.wait(timeout=30)
        self.log.close()


def system_text(nonce: str) -> str:
    # ~1,300 tokens of stable instructions; the nonce makes every run's prefix
    # new, so the first request is a miss even against a long-lived server.
    rules = [f"Rule {i}: when a task touches module m{i}, read its tests first, keep the public API "
             f"stable, and explain each change in one sentence." for i in range(48)]
    return f"Session nonce {nonce}. You are a careful coding agent working in a small repository.\n" + "\n".join(rules)


def messages_request(nonce, header):
    return {"model": "claude-sonnet-4-5", "max_tokens": 12, "temperature": 0, "stream": False,
            "system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.0; cch={header};"},
                       {"type": "text", "text": system_text(nonce)}],
            "messages": [{"role": "user", "content": "Which rule number mentions module m7? Answer with the number."}]}


def wait_stored(port, at_least, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, s = http_json(port, "GET", "/v1/pion/stats", timeout=10)
        if st == 200 and s.get("tokens_stored", 0) >= at_least:
            return s
        time.sleep(0.25)
    return None


def text_of(msg):
    return "".join(b.get("text", "") for b in msg.get("content") or [] if b.get("type") == "text")


def part_live(pion_port):
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("SKIP live part: mlx_lm is not installed")
        return
    try:
        socket.create_connection(("127.0.0.1", pion_port), timeout=2).close()
    except OSError:
        check(f"a Pion server answers on {pion_port}", False, "start pion-server --kvcache -w 1")
        return

    nonce = uuid.uuid4().hex[:12]
    work = tempfile.mkdtemp(prefix="pion_serve_test_")
    log_path = os.path.join(work, "serve.log")
    port = free_port()
    s = Serve(port, pion_port, work, log_path)
    try:
        ready = s.wait_ready()
        check("serve starts and answers /v1/pion/stats", ready, log_path)
        if not ready:
            print(open(log_path, errors="replace").read()[-3000:])
            return

        st, r1 = http_json(port, "POST", "/v1/messages", messages_request(nonce, "aaaa1111"))
        u1 = r1.get("usage", {})
        prompt = u1.get("input_tokens", 0) + u1.get("cache_read_input_tokens", 0)
        check("first request answers", st == 200 and text_of(r1) != "", (st, str(r1)[:300]))
        check("first request is a miss", u1.get("cache_read_input_tokens", -1) == 0, u1)
        check("the prompt is long enough to measure", prompt > 1000, prompt)

        stored = wait_stored(port, prompt - 64)
        check("serve writes the prompt through to Pion", stored is not None,
              stored if stored else "tokens_stored never reached the prompt length")

        st, r2 = http_json(port, "POST", "/v1/messages", messages_request(nonce, "bbbb2222"))
        u2 = r2.get("usage", {})
        check("a second session with a different billing header reuses the prompt",
              st == 200 and u2.get("cache_read_input_tokens", 0) >= prompt - 64, u2)
        check("... and gets the same greedy reply", text_of(r2) == text_of(r1), (text_of(r1), text_of(r2)))
    finally:
        s.kill()

    s = Serve(port, pion_port, work, log_path)
    try:
        ready = s.wait_ready()
        check("serve restarts after SIGKILL", ready, log_path)
        if not ready:
            print(open(log_path, errors="replace").read()[-3000:])
            return
        t0 = time.perf_counter()
        st, r3 = http_json(port, "POST", "/v1/messages", messages_request(nonce, "cccc3333"))
        dt = time.perf_counter() - t0
        u3 = r3.get("usage", {})
        check("after the restart the prompt is served from Pion",
              st == 200 and u3.get("cache_read_input_tokens", 0) >= prompt - 64, u3)
        check("the reply after the restart is identical to the miss's", text_of(r3) == text_of(r1),
              (text_of(r1), text_of(r3)))
        st, stats = http_json(port, "GET", "/v1/pion/stats")
        check("/v1/pion/stats counts the restore", stats.get("restores", 0) >= 1 and stats.get("hits", 0) >= 1, stats)
        print(f"     restored request: {dt * 1000:.0f} ms for {prompt} prompt tokens, "
              f"{u3.get('cache_read_input_tokens')} from the cache")

        rbody = {"model": "gpt-5-codex", "instructions": system_text(nonce), "stream": False,
                 "max_output_tokens": 12, "temperature": 0,
                 "input": "Which rule number mentions module m7? Answer with the number."}
        st, r4 = http_json(port, "POST", "/v1/responses", rbody)
        out = "".join(p.get("text", "") for it in r4.get("output") or [] if it.get("type") == "message"
                      for p in it.get("content") or [])
        cached = ((r4.get("usage") or {}).get("input_tokens_details") or {}).get("cached_tokens", 0)
        check("/v1/responses answers over the same cache", st == 200 and out != "" and cached > 1000,
              (st, cached, out[:80]))
    finally:
        s.kill()


def hf_cached(repo: str) -> bool:
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    return os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots"))


def part_hybrid(pion_port):
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("SKIP hybrid part: mlx_lm is not installed")
        return
    if not hf_cached(HYBRID):
        print(f"SKIP hybrid part: {HYBRID} is not in the Hugging Face cache")
        return
    nonce = uuid.uuid4().hex[:12]
    work = tempfile.mkdtemp(prefix="pion_serve_hybrid_")
    log_path = os.path.join(work, "serve.log")
    port = free_port()
    s = Serve(port, pion_port, work, log_path, model=HYBRID)
    try:
        ready = s.wait_ready()
        check("serve starts on a hybrid model", ready, log_path)
        if not ready:
            print(open(log_path, errors="replace").read()[-3000:])
            return
        first = messages_request(nonce, "aaaa1111")
        first["model"] = "gemma"
        st, r1 = http_json(port, "POST", "/v1/messages", first)
        u1 = r1.get("usage", {})
        p1 = u1.get("input_tokens", 0) + u1.get("cache_read_input_tokens", 0)
        check("hybrid: first request answers", st == 200 and p1 > 1000, (st, u1))
        second = dict(first)
        second["messages"] = first["messages"] + [
            {"role": "assistant", "content": "Rule 7."},
            {"role": "user", "content": "And which rule mentions module m9?"}]
        st, r2 = http_json(port, "POST", "/v1/messages", second)
        u2 = r2.get("usage", {})
        check("hybrid: the next turn reuses the previous prompt in-process",
              st == 200 and u2.get("cache_read_input_tokens", 0) >= 0.9 * p1, (u2, p1))
    finally:
        s.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--offline", action="store_true", help="the command and translation checks only")
    a = ap.parse_args()
    part_command()
    part_translation()
    if not a.offline:
        part_live(a.pion_port)
        part_hybrid(a.pion_port)
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
