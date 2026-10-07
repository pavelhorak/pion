"""pion-vllm-mlx serve — one local endpoint for every coding agent on the Mac.

    pion-server --kvcache -w 1 &
    pion-vllm-mlx serve --model mlx-community/Qwen3-4B-4bit --port 8080
    # same thing without the console script: python -m pion_vllm_mlx serve ...
    export ANTHROPIC_BASE_URL=http://localhost:8080      # Claude Code
    export OPENAI_BASE_URL=http://localhost:8080/v1      # Codex CLI, Aider, Continue, Zed

This is mlx-lm's own server (OpenAI chat/completions, tool calling, streaming,
batching) with three changes:

1. Its prompt cache is backed by Pion (`PionLRUPromptCache`). The in-process
   LRU stays exactly as stock, so a hot hit behaves like vanilla mlx-lm; Pion
   is consulted when it holds a LONGER prefix than the process does (a
   restarted or reloaded server, a second serve process, another tool's
   session), and every insert is written through to Pion as a delta.
2. Requests are normalized before rendering: per-session client metadata
   that would end the shared prefix at token ~15 is removed (Claude Code's
   `x-anthropic-billing-header`), and the request's `model` is pinned to the served model (stock
   mlx-lm would try to LOAD `claude-...` or `qwen3-4b` as a repo).
3. An Anthropic `/v1/messages` endpoint (and `/v1/messages/count_tokens`)
   and an OpenAI Responses `/v1/responses` endpoint (Codex CLI speaks only
   that), both translated onto the chat path over a loopback connection,
   with usage reporting what the cache actually served
   (`cache_read_input_tokens` / `input_tokens_details.cached_tokens`).

What Pion holds is capped by `--pion-budget-gb` (default 4): least-recently-
used lineage leaves are evicted past it, and the V-store WAL is compacted at
idle once it outgrows it. Every restore is credited in the server's value
receipt (`PION.STATS`).

`GET /v1/pion/stats` returns the store's counters. Every other flag is
mlx-lm's (`--model`, `--port`, `--max-tokens`, `--prompt-cache-size`, ...).
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import logging
import re
import sys
import uuid

log = logging.getLogger("pion.serve")

# Lines a client puts in its system prompt that change per session or per
# client version and mean nothing to a local model. Each one, left in place,
# ends the prefix two sessions can share at that line.
VOLATILE_LINES = re.compile(r"^x-anthropic-billing-header:[^\n]*(\n|$)", re.M)


NORMALIZE = True     # --no-normalize turns it off, for the A/B that shows what it buys


def normalize_text(s: str) -> str:
    return VOLATILE_LINES.sub("", s) if (s and NORMALIZE) else s


# ── Pion-backed prompt cache ───────────────────────────────────────────────

_PLAIN: dict = {}


def plain_attention(model) -> bool:
    """True when every layer of the model's prompt cache is a plain KVCache:
    the only kind Pion stores (prefix_store.layout_of). A hybrid model's
    sliding-window or recurrent layers are not stored, so for it the Pion tier
    and the resume journal have nothing to work with."""
    k = id(model)
    if k not in _PLAIN:
        from mlx_lm.models.cache import make_prompt_cache
        _PLAIN[k] = model is not None and all(type(c).__name__ == "KVCache" for c in make_prompt_cache(model))
        if not _PLAIN[k]:
            log.warning("pion: this model has sliding-window or recurrent cache layers; Pion stores "
                        "only plain attention caches, so serve runs on mlx-lm's in-process cache alone "
                        "(no restore after a restart, no resume)")
    return _PLAIN[k]

def make_cache_class():
    import hashlib

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

    from pion_vllm_mlx.prefix_store import layout_of, to_mx

    class _Req:
        """The request mlx-lm is serving on its single-request path."""

        def __init__(self, model_key, prompt, jkey):
            self.model_key, self.prompt, self.jkey = model_key, prompt, jkey
            self.replay: list[int] = []      # tokens generated before a crash
            self.match = None                # lineage end, extended by the journal

    class PionLRUPromptCache(LRUPromptCache):
        """mlx-lm's LRUPromptCache with Pion as the tier below it."""

        def __init__(self, max_size, store, model_provider, min_gain=16, resume=True, durable_every=0):
            super().__init__(max_size)
            self.store = store
            # --durable: a KV.PREFIX.COMMIT barrier after every `durable_every`
            # generated tokens and after each turn's prompt store (0 = off).
            self.durable_every = durable_every
            self.model_provider = model_provider
            self.min_gain = min_gain
            self.resume = resume and store is not None
            self._prompt = None     # prompt tokens of the latest fetch; inserts are cut at it
            self.req_args = None    # GenerationArguments of the request (set by _serve_single)
            self.ctx = None         # _Req for the request stream_generate is about to serve
            self.restores = 0
            self.resumes = 0

        def _jkey(self, model, tokens) -> str:
            # Same prompt AND same sampling: only then is "continue where it
            # stopped" the answer the client asked for again.
            a = self.req_args
            sig = repr(tuple(getattr(a, f, None) for f in
                             ("sampling", "logits", "seed", "max_tokens", "stop_words")))
            h = hashlib.blake2b(digest_size=12)
            h.update(repr(model).encode())
            h.update(np.asarray(tokens, dtype=np.uint32).tobytes())
            h.update(sig.encode())
            return h.hexdigest()

        def _extend(self, model, cache, local, match, end):
            """The process's cache (rows [0, local)) extended with Pion's rows
            up to `end`, or a fresh cache filled from row 0; None on failure."""
            if cache is not None and local > 0 and layout_of(cache) is not None:
                base, start = cache, local
            else:
                base, start = make_prompt_cache(self.model_provider.model), 0
            try:
                rows = self.store.fetch_rows(model, match, start, end)
            except Exception as e:
                log.warning("pion fetch failed: %s", e)
                self.store._reset_conn()
                return None
            if rows is None:
                return None
            lay = self.store._layout(self.store.model_id(model))
            for c, (k, v) in zip(base, rows):
                c.update_and_fetch(to_mx(k, lay)[None], to_mx(v, lay)[None])
            mx.eval(*[c.keys for c in base], *[c.values for c in base])
            self.restores += 1
            return base

        def fetch_nearest_cache(self, model, tokens):
            cache, rest = super().fetch_nearest_cache(model, tokens)
            local = len(tokens) - len(rest)
            self._prompt = list(tokens)
            self.ctx = None
            if (self.store is None or self.model_provider.draft_model is not None
                    or not plain_attention(self.model_provider.model)):
                return cache, rest
            try:
                jkey = self._jkey(model, tokens) if self.resume else None
                self.ctx = _Req(model, list(tokens), jkey)
                replay = self.store.journal_get(model, jkey) if jkey else None
                log.debug("pion: journal %s -> %s", jkey, None if replay is None else len(replay))
                if replay:
                    # An earlier attempt at this exact request died mid-answer.
                    # Its rows for prompt + replay[:-1] are in the lineage: feed
                    # replay[-1] next and the model takes the same step it would
                    # have taken, over the same rows.
                    target = list(tokens) + replay[:-1]
                    tm = self.store.lookup(model, target)
                    log.debug("pion: resume target %d, lineage holds %d", len(target), tm.length)
                    if tm.length >= len(target):
                        base = self._extend(model, cache, local, tm, len(target))
                        if base is not None:
                            self.ctx.replay, self.ctx.match = replay, tm
                            self.resumes += 1
                            log.info("pion: resuming an interrupted answer at token %d "
                                     "(prompt %d, restored %d rows)", len(replay), len(tokens), len(target) - local)
                            return base, [replay[-1]]
                    self.store.journal_drop(model, jkey)
                match = self.store.lookup(model, tokens)
            except Exception as e:               # Pion down must never take serving down
                log.warning("pion lookup failed: %s", e)
                self.store._reset_conn()
                return cache, rest
            m = min(match.length, len(tokens) - 1)
            if m - local < self.min_gain:
                return cache, rest
            # Extend what the process already holds rather than replace it, so
            # the rows fetched are only the ones it is missing.
            base = self._extend(model, cache, local, match, m)
            if base is None:
                return cache, rest
            log.info("pion: restored %d tokens (process had %d of %d)", m - local, local, len(tokens))
            return base, tokens[m:]

        def journal(self, ctx, prompt_cache, generated) -> None:
            """Rows for prompt + generated[:-1] into the lineage, then the ids.
            Rows first: a journal naming rows that are not there cannot resume."""
            try:
                n = self.store.store(ctx.model_key, ctx.prompt + generated[:-1], prompt_cache, match=ctx.match)
                ctx.match = self.store.last_match
                log.debug("pion: journal k=%d stored %d rows, lineage now %s", len(generated), n,
                          None if ctx.match is None else ctx.match.length)
                self.store.journal_put(ctx.model_key, ctx.jkey, generated)
                if self.durable_every and (len(generated) == 1 or len(generated) % self.durable_every == 0):
                    self.store.commit()
            except Exception as e:
                log.warning("pion journal failed: %s", e)
                self.store._reset_conn()

        def journal_done(self, ctx) -> None:
            try:
                self.store.journal_drop(ctx.model_key, ctx.jkey)
            except Exception as e:
                log.warning("pion journal drop failed: %s", e)
                self.store._reset_conn()

        def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
            super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
            if self.store is None or not plain_attention(self.model_provider.model):
                return
            # Store the prompt, not the reply. An agent's next prompt re-renders
            # the assistant turn from its own copy (reasoning stripped, output
            # truncated, tool calls re-serialized), so generated rows rarely
            # match it; keeping them would fork the lineage every turn instead
            # of extending it in place. mlx-lm inserts several caches per
            # request (segment ends, then prompt + reply); every one of them is
            # cut at the prompt of the latest fetch when it runs past it.
            prompt = self._prompt
            if prompt is not None and len(tokens) > len(prompt) and list(tokens[:len(prompt)]) == prompt:
                tokens = tokens[:len(prompt)]
            try:
                n = self.store.store(model, tokens, prompt_cache)
                if n:
                    log.info("pion: stored %d new tokens", n)
                if n and self.durable_every:
                    self.store.commit()
            except Exception as e:
                log.warning("pion store failed: %s", e)
                self.store._reset_conn()

    return PionLRUPromptCache


JOURNAL_EVERY = 16


def make_stream_generate(orig, cache):
    """mlx-lm's stream_generate, journaled, and able to continue an answer.

    A resumed request first replays the tokens the interrupted attempt had
    produced, through one detokenizer shared with what follows, so every
    consumer downstream (text, reasoning and tool-call parsing, SSE) sees one
    uninterrupted stream; then it generates from the step after them. The
    journal is written at the first new token (which stores the prompt rows)
    and every JOURNAL_EVERY tokens."""
    import mlx.core as mx
    from dataclasses import replace
    from mlx_lm.generate import GenerationResponse

    no_logprobs = None

    def pion_stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None, **kw):
        nonlocal no_logprobs
        ctx, cache.ctx = cache.ctx, None
        if ctx is None or ctx.jkey is None or draft_model is not None:
            yield from orig(model, tokenizer, prompt, max_tokens=max_tokens, draft_model=draft_model, **kw)
            return
        pc = kw.get("prompt_cache")
        detok = tokenizer.detokenizer
        detok.reset()
        gen: list[int] = []
        if no_logprobs is None:
            no_logprobs = mx.zeros((1 << 18,))      # replayed tokens have no logprobs to report
        for t in ctx.replay:
            detok.add_token(t)
            gen.append(t)
            yield GenerationResponse(text=detok.last_segment, token=t, logprobs=no_logprobs, from_draft=False,
                                     prompt_tokens=len(ctx.prompt), prompt_tps=0.0, generation_tokens=len(gen),
                                     generation_tps=0.0, peak_memory=0.0, finish_reason=None)
        first_new = len(gen) + 1
        for r in orig(model, tokenizer, prompt, max_tokens=max_tokens - len(gen), **kw):
            if r.finish_reason is None:
                detok.add_token(r.token)
                gen.append(r.token)
                yield replace(r, text=detok.last_segment, generation_tokens=len(gen))
                if len(gen) == first_new or len(gen) % JOURNAL_EVERY == 0:
                    cache.journal(ctx, pc, gen)
            else:
                if r.token not in tokenizer.eos_token_ids:
                    detok.add_token(r.token)
                    gen.append(r.token)
                detok.finalize()
                # Drop the journal BEFORE the final yield: mlx-lm stops iterating
                # at the response that carries finish_reason, so nothing after
                # this yield ever runs.
                cache.journal_done(ctx)
                yield replace(r, text=detok.last_segment, generation_tokens=len(gen))

    return pion_stream_generate


# ── Anthropic <-> OpenAI translation ───────────────────────────────────────

def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
        elif isinstance(b, str):
            out.append(b)
    return "".join(out)


def anthropic_to_openai(body: dict) -> dict:
    """An Anthropic Messages request as an OpenAI chat request mlx-lm renders
    through the model's own chat template."""
    msgs: list[dict] = []
    sys_blocks = body.get("system")
    sys_blocks = [sys_blocks] if isinstance(sys_blocks, str) else [_text_of([b]) for b in sys_blocks or []]
    # Normalize per block: Claude Code sends the billing header as a block of
    # its own, and joining first would fuse it with the prompt that follows.
    system = "\n\n".join(t for t in (normalize_text(b) for b in sys_blocks) if t.strip())
    if system:
        msgs.append({"role": "system", "content": system})

    def add_user_text(text: str) -> None:
        if not text:
            return
        if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], str):
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": "user", "content": text})

    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        if role == "assistant":
            text, calls = [], []
            for b in blocks:
                t = b.get("type")
                if t == "text":
                    text.append(b.get("text", ""))
                elif t == "tool_use":
                    calls.append({"id": b.get("id") or f"toolu_{uuid.uuid4().hex[:24]}", "type": "function",
                                  "function": {"name": b.get("name", ""),
                                               "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
                # thinking / redacted_thinking: chat templates drop earlier reasoning
            msg = {"role": "assistant", "content": "".join(text)}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
            continue
        # user, and Claude Code's mid-conversation "system" messages, which
        # most chat templates only accept first: both become user text.
        pending = []
        for b in blocks:
            t = b.get("type")
            if t == "tool_result":
                c = b.get("content")
                text = _text_of(c) if not isinstance(c, str) else c
                if b.get("is_error"):
                    text = "Error: " + text
                msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": text})
            elif t == "text":
                pending.append(b.get("text", ""))
            elif t == "image":
                pending.append("[image omitted: this model reads text only]")
        text = "".join(pending)
        add_user_text(normalize_text(text) if role == "system" else text)

    tools = []
    for t in body.get("tools") or []:
        if "input_schema" not in t:      # server-side tools (web search, ...) have no local meaning
            continue
        tools.append({"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}})
    out = {"model": "default_model", "messages": msgs, "stream": True,
           "stream_options": {"include_usage": True}}
    if tools:
        out["tools"] = tools
    for a, b in (("max_tokens", "max_tokens"), ("temperature", "temperature"),
                 ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop")):
        if body.get(a) is not None:
            out[b] = body[a]
    return out


class _AnthropicStream:
    """Turns mlx-lm's OpenAI SSE chunks into Anthropic Messages events."""

    def __init__(self, write, model: str):
        self.write, self.model = write, model
        self.id = f"msg_{uuid.uuid4().hex[:24]}"
        self.idx = -1
        self.open = None            # "text" | "thinking" | None
        self.finish = None
        self.saw_tool = False
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        self.content = []           # for the non-streaming reply

    def event(self, name, data):
        self.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

    def start(self):
        self.event("message_start", {"type": "message_start", "message": {
            "id": self.id, "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None, "usage": dict(self.usage)}})

    def _open(self, kind, block):
        if self.open == kind:
            return
        self._close()
        self.idx += 1
        self.open = kind
        self.event("content_block_start", {"type": "content_block_start", "index": self.idx, "content_block": block})
        self.content.append(dict(block))

    def _close(self):
        if self.open is not None:
            if self.open == "thinking":
                self.event("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "signature_delta", "signature": ""}})
            self.event("content_block_stop", {"type": "content_block_stop", "index": self.idx})
            self.open = None

    def chunk(self, obj):
        if obj.get("usage"):
            u = obj["usage"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            self.usage.update(input_tokens=u.get("prompt_tokens", 0) - cached,
                              output_tokens=u.get("completion_tokens", 0),
                              cache_read_input_tokens=cached)
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or ch.get("message") or {}
            if d.get("reasoning"):
                self._open("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                self.content[-1]["thinking"] += d["reasoning"]
                self.event("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "thinking_delta", "thinking": d["reasoning"]}})
            if d.get("content"):
                self._open("text", {"type": "text", "text": ""})
                self.content[-1]["text"] += d["content"]
                self.event("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "text_delta", "text": d["content"]}})
            for tc in d.get("tool_calls") or []:
                self._close()
                self.idx += 1
                f = tc.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                block = {"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                         "name": f.get("name", ""), "input": {}}
                self.event("content_block_start", {"type": "content_block_start", "index": self.idx,
                                                   "content_block": block})
                self.event("content_block_delta", {"type": "content_block_delta", "index": self.idx,
                                                   "delta": {"type": "input_json_delta",
                                                             "partial_json": json.dumps(args, ensure_ascii=False)}})
                self.event("content_block_stop", {"type": "content_block_stop", "index": self.idx})
                self.content.append(dict(block, input=args))
                self.saw_tool = True
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]

    def stop_reason(self):
        if self.saw_tool:
            return "tool_use"
        return {"length": "max_tokens"}.get(self.finish, "end_turn")

    def end(self):
        self._close()
        self.event("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": self.stop_reason(), "stop_sequence": None},
                                     "usage": dict(self.usage)})
        self.event("message_stop", {"type": "message_stop"})

    def message(self):
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": self.content, "stop_reason": self.stop_reason(), "stop_sequence": None,
                "usage": dict(self.usage)}


# ── OpenAI Responses (Codex CLI) <-> chat ──────────────────────────────────

def _parts_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content
                   if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text"))


def responses_to_openai(body: dict) -> dict:
    """An OpenAI Responses request (Codex CLI's only wire format) as a chat
    request. Reasoning items are dropped (chat templates drop earlier
    reasoning); consecutive function calls become one assistant turn."""
    msgs: list[dict] = []
    instr = normalize_text(body.get("instructions") or "")
    if instr:
        msgs.append({"role": "system", "content": instr})

    def add_user(text):
        if not text:
            return
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": "user", "content": text})

    inp = body.get("input")
    items = [{"type": "message", "role": "user", "content": inp}] if isinstance(inp, str) else (inp or [])
    for it in items:
        t = it.get("type", "message")
        if t == "message":
            role, text = it.get("role"), _parts_text(it.get("content"))
            if role in ("system", "developer"):
                if len(msgs) == 1 and msgs[0]["role"] == "system" or not msgs:
                    if msgs:
                        msgs[0]["content"] += "\n\n" + normalize_text(text)
                    else:
                        msgs.append({"role": "system", "content": normalize_text(text)})
                else:
                    add_user(text)
            elif role == "assistant":
                msgs.append({"role": "assistant", "content": text})
            else:
                add_user(text)
        elif t == "function_call":
            call = {"id": it.get("call_id") or it.get("id") or f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                    "function": {"name": it.get("name", ""), "arguments": it.get("arguments") or "{}"}}
            if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("_closed"):
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif t == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id", ""),
                         "content": out if isinstance(out, str) else _parts_text(out)})
        # reasoning and anything else: nothing a local chat template renders
    for m in msgs:
        m.pop("_closed", None)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("parameters") or {"type": "object"}}}
             for t in body.get("tools") or [] if t.get("type") == "function" and t.get("name")]
    out = {"model": "default_model", "messages": msgs, "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        out["tools"] = tools
    for a, b in (("max_output_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(a) is not None:
            out[b] = body[a]
    return out


class _ResponsesStream:
    """mlx-lm's OpenAI chat chunks as OpenAI Responses events."""

    def __init__(self, write, model: str):
        self.write, self.model = write, model
        self.id = f"resp_{uuid.uuid4().hex[:24]}"
        self.seq = 0
        self.output: list[dict] = []
        self.cur = None          # the message item being streamed
        self.reasoning = None
        self.usage = {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}

    def event(self, typ, **data):
        self.seq += 1
        data = {"type": typ, "sequence_number": self.seq, **data}
        self.write(f"event: {typ}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())

    def _resp(self, status):
        r = {"id": self.id, "object": "response", "created_at": 0, "status": status, "model": self.model,
             "output": self.output}
        if status == "completed":
            r["usage"] = self.usage
        return r

    def start(self):
        self.event("response.created", response=self._resp("in_progress"))
        self.event("response.in_progress", response=self._resp("in_progress"))

    def _close_message(self):
        if self.cur is None:
            return
        idx = self.output.index(self.cur)
        text = self.cur["content"][0]["text"]
        self.event("response.output_text.done", item_id=self.cur["id"], output_index=idx, content_index=0, text=text)
        self.event("response.content_part.done", item_id=self.cur["id"], output_index=idx, content_index=0,
                   part=self.cur["content"][0])
        self.cur["status"] = "completed"
        self.event("response.output_item.done", output_index=idx, item=self.cur)
        self.cur = None

    def _close_reasoning(self):
        if self.reasoning is None:
            return
        idx = self.output.index(self.reasoning)
        self.event("response.output_item.done", output_index=idx, item=self.reasoning)
        self.reasoning = None

    def chunk(self, obj):
        if obj.get("usage"):
            u = obj["usage"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            self.usage.update(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                              total_tokens=u.get("total_tokens", 0), input_tokens_details={"cached_tokens": cached})
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or ch.get("message") or {}
            if d.get("reasoning"):
                if self.reasoning is None:
                    self.reasoning = {"id": f"rs_{uuid.uuid4().hex[:24]}", "type": "reasoning", "summary": [],
                                      "content": [{"type": "reasoning_text", "text": ""}]}
                    self.output.append(self.reasoning)
                    self.event("response.output_item.added", output_index=len(self.output) - 1, item=self.reasoning)
                self.reasoning["content"][0]["text"] += d["reasoning"]
                self.event("response.reasoning_text.delta", item_id=self.reasoning["id"],
                           output_index=self.output.index(self.reasoning), content_index=0, delta=d["reasoning"])
            if d.get("content"):
                self._close_reasoning()
                if self.cur is None:
                    self.cur = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
                                "status": "in_progress", "content": [{"type": "output_text", "text": "", "annotations": []}]}
                    self.output.append(self.cur)
                    idx = len(self.output) - 1
                    self.event("response.output_item.added", output_index=idx, item=self.cur)
                    self.event("response.content_part.added", item_id=self.cur["id"], output_index=idx,
                               content_index=0, part={"type": "output_text", "text": "", "annotations": []})
                self.cur["content"][0]["text"] += d["content"]
                self.event("response.output_text.delta", item_id=self.cur["id"],
                           output_index=self.output.index(self.cur), content_index=0, delta=d["content"])
            for tc in d.get("tool_calls") or []:
                self._close_reasoning()
                self._close_message()
                f = tc.get("function") or {}
                args = f.get("arguments") or "{}"
                item = {"id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call", "status": "in_progress",
                        "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}", "name": f.get("name", ""),
                        "arguments": ""}
                self.output.append(item)
                idx = len(self.output) - 1
                self.event("response.output_item.added", output_index=idx, item=dict(item))
                self.event("response.function_call_arguments.delta", item_id=item["id"], output_index=idx, delta=args)
                self.event("response.function_call_arguments.done", item_id=item["id"], output_index=idx, arguments=args)
                item.update(arguments=args, status="completed")
                self.event("response.output_item.done", output_index=idx, item=item)

    def end(self):
        self._close_reasoning()
        self._close_message()
        self.event("response.completed", response=self._resp("completed"))


# ── HTTP handler ───────────────────────────────────────────────────────────

def make_handler_class():
    from mlx_lm.server import APIHandler

    class PionAPIHandler(APIHandler):
        loopback = ("127.0.0.1", 8080)
        store = None
        cache = None

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _reply_json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/pion/stats":
                st = self.store.stats.as_dict() if self.store is not None else {"pion": "disabled"}
                if self.cache is not None:
                    st["restores"] = getattr(self.cache, "restores", 0)
                    st["resumes"] = getattr(self.cache, "resumes", 0)
                return self._reply_json(200, st)
            return super().do_GET()

        def do_POST(self):
            path = self.path.split("?")[0]
            if path in ("/v1/messages", "/v1/messages/count_tokens"):
                return self._anthropic(path)
            if path in ("/v1/responses", "/responses"):
                return self._responses()
            # OpenAI shapes: normalize, then hand the rewritten body to mlx-lm.
            raw = self._read_body()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = None
            if isinstance(body, dict):
                body["model"] = "default_model"
                for m in body.get("messages") or []:
                    if m.get("role") == "system" and isinstance(m.get("content"), str):
                        m["content"] = normalize_text(m["content"])
                raw = json.dumps(body).encode()
            self.rfile = io.BytesIO(raw)
            del self.headers["Content-Length"]
            self.headers["Content-Length"] = str(len(raw))
            return super().do_POST()

        def _anthropic(self, path):
            try:
                body = json.loads(self._read_body())
                req = anthropic_to_openai(body)
            except Exception as e:
                return self._reply_json(400, {"type": "error", "error": {
                    "type": "invalid_request_error", "message": f"{type(e).__name__}: {e}"}})
            if path.endswith("count_tokens"):
                return self._count_tokens(req)
            streaming = bool(body.get("stream"))
            model_name = body.get("model") or "local"
            conn = http.client.HTTPConnection(*self.loopback, timeout=3600)
            conn.request("POST", "/v1/chat/completions", body=json.dumps(req),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                return self._reply_json(r.status, {"type": "error", "error": {
                    "type": "api_error", "message": r.read().decode(errors="replace")[:2000]}})
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def write(b):
                    self.wfile.write(b)
                    self.wfile.flush()
            else:
                def write(b):
                    pass
            out = _AnthropicStream(write, model_name)
            out.start()
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    out.chunk(json.loads(data))
                except json.JSONDecodeError:
                    continue
            out.end()
            conn.close()
            if not streaming:
                return self._reply_json(200, out.message())
            self.close_connection = True

        def _responses(self):
            try:
                body = json.loads(self._read_body())
                req = responses_to_openai(body)
            except Exception as e:
                return self._reply_json(400, {"error": {"type": "invalid_request_error",
                                                        "message": f"{type(e).__name__}: {e}"}})
            streaming = bool(body.get("stream"))
            conn = http.client.HTTPConnection(*self.loopback, timeout=3600)
            conn.request("POST", "/v1/chat/completions", body=json.dumps(req),
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                return self._reply_json(r.status, {"error": {"type": "server_error",
                                                             "message": r.read().decode(errors="replace")[:2000]}})
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                def write(b):
                    self.wfile.write(b)
                    self.wfile.flush()
            else:
                def write(b):
                    pass
            out = _ResponsesStream(write, body.get("model") or "local")
            out.start()
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    out.chunk(json.loads(data))
                except json.JSONDecodeError:
                    continue
            out.end()
            conn.close()
            if not streaming:
                return self._reply_json(200, out._resp("completed"))
            self.close_connection = True

        def _count_tokens(self, req):
            tok = self.response_generator.model_provider.tokenizer
            from mlx_lm.server import process_message_content
            msgs = json.loads(json.dumps(req["messages"]))
            process_message_content(msgs)
            ids = tok.apply_chat_template(msgs, tools=req.get("tools"), add_generation_prompt=True, tokenize=True)
            return self._reply_json(200, {"input_tokens": len(ids)})

    return PionAPIHandler


# ── entry point ────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pion-vllm-mlx serve", add_help=False,
        description="mlx-lm's server with a Pion-backed prompt cache, plus Anthropic /v1/messages "
                    "(Claude Code) and OpenAI /v1/responses (Codex). Every flag not listed here "
                    "is mlx-lm's (--model, --port, --host, --max-tokens, --prompt-cache-size, ...).")
    ap.add_argument("--pion-host", default="127.0.0.1")
    ap.add_argument("--pion-port", type=int, default=1974)
    ap.add_argument("--pion-vquant", default="fp16",
                    help="V-store format for stored K/V (fp16 is lossless for fp16 models)")
    ap.add_argument("--pion-budget-gb", type=float, default=4.0, metavar="GB",
                    help="cap on the K/V rows kept in Pion (V-store RAM; its disk stays within "
                         "~2-3x): least-recently-used lineage leaves are evicted past it, and the "
                         "V-store WAL is compacted at idle once it outgrows it. 0 = no cap (default 4)")
    ap.add_argument("--pion-min-gain", type=int, default=16,
                    help="restore from Pion only when it adds at least this many tokens")
    ap.add_argument("--no-pion", action="store_true",
                    help="plain mlx-lm prompt cache (the A/B control), keeping normalization and /v1/messages")
    ap.add_argument("--durable", type=int, nargs="?", const=64, default=0, metavar="TOKENS",
                    help="survive a power cut or kernel panic, not just a crash: KV.PREFIX.COMMIT "
                         "(F_FULLFSYNC on macOS) after every TOKENS generated tokens (default 64) and "
                         "after each prompt store. ~4 ms per barrier on an M4 SSD")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not journal answers (every 16 tokens) nor continue an interrupted one")
    ap.add_argument("--no-normalize", action="store_true",
                    help="keep per-session client metadata in the prompt (the A/B control for normalization)")
    ap.add_argument("--unload-after", type=float, default=None, metavar="SECONDS",
                    help="free the model's RAM after SECONDS idle; the next request reloads it "
                         "and restores its prompt cache from Pion")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in raw_argv):
        # mlx-lm's parser owns --help (it parses everything we do not), so
        # print ours first and then let it print its own and exit.
        print(ap.format_help())
        print("mlx-lm server options (passed through unchanged):\n", flush=True)
        import mlx_lm.server as S
        sys.argv = ["pion-vllm-mlx serve", "--help"]
        S.main()
        return 0
    ours, rest = ap.parse_known_args(argv)
    global NORMALIZE
    NORMALIZE = not ours.no_normalize

    if ours.unload_after is not None:
        from pion_vllm_mlx import supervisor
        raw = list(sys.argv[1:] if argv is None else argv)
        host, port = _opt(raw, "--host", "127.0.0.1"), int(_opt(raw, "--port", "8080"))
        internal = supervisor.free_port()
        child = [a for a in _drop_opt(_drop_opt(raw, "--unload-after"), "--port")]
        child_argv = [sys.executable, "-m", "pion_vllm_mlx.serve", *child, "--port", str(internal)]
        return supervisor.run(host, port, child_argv, internal, ours.unload_after)

    import mlx.core as mx
    import mlx_lm.server as S
    from http.server import ThreadingHTTPServer

    from pion_vllm_mlx.prefix_store import PionPrefixStore

    store = None if ours.no_pion else PionPrefixStore(ours.pion_host, ours.pion_port, ours.pion_vquant,
                                                     budget_bytes=int(ours.pion_budget_gb * (1 << 30)))
    Cache = make_cache_class()
    Handler = make_handler_class()

    def run(host, port, model_provider, server_class=ThreadingHTTPServer, handler_class=None):
        mx.distributed.init()
        size = model_provider.cli_args.prompt_cache_size
        cache = (Cache(size, store, model_provider, ours.pion_min_gain, resume=not ours.no_resume,
                       durable_every=ours.durable)
                 if store is not None else S.LRUPromptCache(size))
        if store is not None and not ours.no_resume:
            # Resume needs the single-request path, where generation is one
            # stream_generate call per request (the batched path interleaves
            # requests inside mlx-lm). One local user loses nothing by it,
            # except on a hybrid model: there Pion stores nothing to resume
            # from, and the single-request path also skips the cache snapshots
            # mlx-lm's batched path takes at segment ends, which are the only
            # reuse a sliding-window or recurrent cache gets. Forcing it there
            # cost every request a full prefill (Gemma-4-E2B: 0% of a Claude
            # Code session reused, against 98.4% on stock mlx-lm).
            orig_batchable = S.ResponseGenerator._is_batchable

            def _is_batchable(self, args):
                if plain_attention(self.model_provider.model):
                    return False
                return orig_batchable(self, args)

            S.ResponseGenerator._is_batchable = _is_batchable
            orig_single = S.ResponseGenerator._serve_single

            # mlx-lm 0.32 added a second argument (the generation stream);
            # pass through whatever the installed version sends.
            def _serve_single(self, request, *rest):
                self.prompt_cache.req_args = request[2]
                return orig_single(self, request, *rest)

            S.ResponseGenerator._serve_single = _serve_single
            S.stream_generate = make_stream_generate(S.stream_generate, cache)
        Handler.loopback = ("127.0.0.1" if host in ("0.0.0.0", "::", "") else host, port)
        Handler.store, Handler.cache = store, cache
        rg = S.ResponseGenerator(model_provider, cache)
        where = ("off (--no-pion)" if store is None else
                 f"{ours.pion_host}:{ours.pion_port} vquant={ours.pion_vquant} "
                 f"budget={'none' if not ours.pion_budget_gb else f'{ours.pion_budget_gb:g} GB'}")
        print(f"pion-vllm-mlx serve: http://{host}:{port}  (OpenAI /v1/chat/completions, Anthropic /v1/messages)"
              f"  pion={where}", flush=True)
        from pion_vllm_mlx.supervisor import FastBindHTTPServer
        S._run_http_server(host, port, rg, FastBindHTTPServer, Handler)

    S.run = run
    sys.argv = ["pion-vllm-mlx serve"] + rest
    S.main()


def _opt(argv, name, default):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _drop_opt(argv, name):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == name:
            skip = True
            continue
        if a.startswith(name + "="):
            continue
        out.append(a)
    return out


if __name__ == "__main__":
    main()
