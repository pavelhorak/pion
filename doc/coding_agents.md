# Coding agents: one local model behind Claude Code and Codex

`pion-vllm-mlx serve` runs a local MLX model behind the three APIs coding
agents speak, with the model's prompt cache kept in Pion:

| Client | API it speaks | Point it at |
|---|---|---|
| Claude Code | Anthropic `/v1/messages` (+ `count_tokens`) | `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| Codex CLI | OpenAI `/v1/responses` | a `model_providers` entry with `wire_api = "responses"` |
| Aider, Continue, Zed, any OpenAI client | OpenAI `/v1/chat/completions` | `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` |

It is mlx-lm's own server (`mlx_lm.server`: chat templates, tool calling,
streaming) with three changes: its prompt cache is backed by Pion, requests
are normalized so two sessions can share a prefix, and the Anthropic and
Responses endpoints are translated onto mlx-lm's chat path.

## Quick start

```bash
# 1. Pion, with the prompt cache on (the Homebrew service already runs with --kvcache)
brew install pavelhorak/tap/pion && brew services start pion
#    or, from a release tarball or a source build:  ./pion-server --kvcache -w 1

# 2. serve
pip install 'pion-vllm-mlx[mlx]'
pion-vllm-mlx serve --model mlx-community/Qwen3-4B-4bit --port 8080
#    pion-vllm-mlx 0.1.5 and earlier have no console script; run the module instead:
#    python -m pion_vllm_mlx.serve --model mlx-community/Qwen3-4B-4bit --port 8080
```

**Claude Code:**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=local      # any value; serve does not check it
claude
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
model = "local"
model_provider = "local"

[model_providers.local]
name = "local"
base_url = "http://127.0.0.1:8080/v1"
env_key = "OPENAI_API_KEY"          # export OPENAI_API_KEY=local
wire_api = "responses"
```

`pion-vllm-mlx serve --help` lists serve's own flags and then mlx-lm's
(`--model`, `--host`, `--port`, `--max-tokens`, `--prompt-cache-size`, …),
which pass through unchanged.

## What the Pion tier adds

A stock local server keeps its prompt cache in the process. Within one
running session that is enough: an agent's next prompt extends its last one,
and the server reuses the shared part. serve keeps exactly that in-process
tier (mlx-lm's `LRUPromptCache`) and puts Pion underneath it, which changes
what happens when the process does not already hold the prefix:

- **A restart.** serve writes every prompt's K/V through to Pion as it goes.
  A restarted serve, or one whose model was unloaded with `--unload-after`
  and reloaded, finds the conversation's longest stored prefix and fetches
  only the rows it is missing, instead of prefilling the whole prompt again.
  Restored rows are the stored bits: fp16 K/V as fp16, bf16 K/V as its bit
  patterns, so a restored prefix produces the same greedy reply as the
  cold one.
- **A second session, a second tool, a second serve process.** Claude Code
  opens its system prompt with an `x-anthropic-billing-header` line that
  differs per session, which ends the prefix two sessions could share a few
  tokens in. serve removes that line before rendering (`--no-normalize`
  keeps it), so a new session starts from the stored system prompt and tool
  definitions. Several serve processes on one machine share one store.
- **An interrupted answer.** With resume on (the default), serve journals
  each answer every 16 tokens; a request repeated after a crash continues
  where the first attempt stopped. `--no-resume` turns it off.
- **A budget.** `--pion-budget-gb` (default 4) caps what Pion holds; the
  least-recently-used conversation branch goes first, and the shared root
  last. `--durable [TOKENS]` adds an fsync barrier for power-loss safety.

`usage` reports what the cache actually served: `cache_read_input_tokens`
on `/v1/messages`, `input_tokens_details.cached_tokens` on `/v1/responses`,
`prompt_tokens_details.cached_tokens` on chat completions. `GET
/v1/pion/stats` returns serve's own counters (lookups, restores, tokens
restored and stored), and Pion's `PION.STATS` credits each restore.

## Measured against the alternatives

Recorded coding-agent sessions, replayed against five local servers, one at a
time, on an M4 Mac mini (16 GB). From Claude Code, 21 requests went to each
server: 20 turns of one session on a 5-file Python repository, then the first
request of a second session on the same repository. From Codex, 20 turns of
the same tasks. The server process was stopped and started again after turn
10. Requests are sent greedy and streamed, with replies capped at 64 tokens.
Claude Code's prompts run from 14K to 21K tokens and Codex's from 7K to 14K,
most of them each client's system prompt and tool definitions.

**Llama-3.2-1B-Instruct-4bit** (plain attention):

| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 16,020 tokens, 15.24 s | 98.9%, 0.49 s | 17,478 of 17,478; 3.0 + 16.21 s = 19.2 s | 45 of 16,065, 15.36 s |
| Ollama | 13,971 tokens, 18.42 s | 35.3%, 14.66 s | 15,376 of 15,376; 0.6 + 18.34 s = 18.9 s | 30 of 14,016, 16.98 s |
| LM Studio | 15,921 tokens, 15.01 s | 98.6%, 1.33 s | 18,263 of 18,263; 3.4 + 18.25 s = 21.6 s | 46 of 15,966, 15.80 s |
| oMLX | 16,009 tokens, 15.04 s | 97.9%, 0.85 s | 469 of 18,389; 1.2 + 1.63 s = 2.8 s | 13,824 of 16,054, 2.67 s |
| Pion serve | 15,994 tokens, 14.07 s | 98.9%, 0.42 s | 89 of 17,452; 3.5 + 0.66 s = 4.2 s | 14,056 of 16,039, 2.33 s |

**Gemma-4-E2B-it-4bit** (hybrid: sliding-window and global attention layers):

| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 14,426 tokens, 9.89 s | 98.4%, 0.41 s | 16,129 of 16,129; 2.4 + 10.47 s = 12.9 s | 0 of 14,486, 8.02 s |
| LM Studio | 14,422 tokens, 9.14 s | 15.5%, 10.27 s | 16,121 of 16,121; 7.1 + 11.95 s = 19.1 s | 0 of 14,482, 8.17 s |
| oMLX | 14,397 tokens, 11.34 s | 98.4%, 0.40 s | 116 of 16,102; 2.7 + 5.46 s = 8.2 s | 12,288 of 14,457, 1.33 s |
| Pion serve | 14,391 tokens, 9.99 s | 98.4%, 0.40 s | 16,094 of 16,094; 2.4 + 10.22 s = 12.6 s | 12,087 of 14,451, 1.67 s |

**Codex CLI, Llama-3.2-1B-Instruct-4bit** (OpenAI Responses API; the same 20 tasks recorded from Codex, no second session recorded):

| Server | first request, cold | within the session: reused, TTFT | **after a restart**: recomputed; server start + first token | **a second session**: reused, TTFT |
|---|---:|---:|---:|---:|
| mlx-lm server, stock | 7,121 tokens, 5.14 s | 97.5%, 0.36 s | 9,560 of 9,560; 3.0 + 7.33 s = 10.3 s | — |
| Ollama | 7,540 tokens, 9.96 s | 82.7%, 0.40 s | 9,872 of 9,872; 0.7 + 9.94 s = 10.6 s | — |
| LM Studio (9 of 20 replies ended in a stream error) | 7,249 tokens, 5.37 s | 97.0%, 0.54 s | not reported (stream error); 3.3 + 8.22 s = 11.5 s | — |
| oMLX | 9,703 tokens, 8.05 s | 97.1%, 0.51 s | 110 of 12,142; 1.2 + 1.13 s = 2.3 s | — |
| Pion serve | 7,121 tokens, 5.59 s | 97.5%, 0.38 s | 77 of 9,560; 2.3 + 1.13 s = 3.4 s | — |

What the tables say:

- **Within one session every server reuses almost the whole prompt, with two
  exceptions.** Ollama re-prefills about two thirds of every Claude Code turn:
  its Llama 3.2 template places the tool definitions before the latest user
  message, so the prompt changes early each turn (on Codex it reuses 82.7%).
  LM Studio reuses little on the hybrid model.
- **After a restart only oMLX and serve skip the re-prefill.** oMLX reads its
  SSD cache; Pion holds the prefix in RAM. The first token after the restart
  came sooner from serve on Claude Code (0.66 s against 1.63 s) and at the
  same time on Codex (1.13 s each). oMLX starts faster, because it loads the
  model on the first request, so from the start command to the first token it
  is ahead on both: 2.8 s against 4.2 s, and 2.3 s against 3.4 s.
- **A second session reuses the shared system prompt and tools only on oMLX
  and serve.** Both drop Claude Code's per-session billing-header line; the
  stock servers reuse 30 to 46 tokens.
- **On the hybrid model oMLX is ahead of serve.** Pion stores plain attention
  caches only, so after a restart serve recomputes the whole prompt (12.6 s)
  where oMLX's SSD cache holds the sliding-window layers too (8.2 s). Within
  the session and for a second session the two match. Before this version
  serve reused nothing on hybrid models, not even within a session
  (`gemma_pion_before_fix.json`): its resume journal forced mlx-lm's
  single-request path, which skips the snapshots mlx-lm takes at segment ends.
- **oMLX 0.7 does everything serve does here, and on hybrid models more.**
  serve's lead is the first token after a restart on Claude Code with a
  plain-attention model, about 1 s, and a crash-consistent store that a
  different program can read over the Redis wire.

How faithful the reuse is: on Gemma, mlx-lm's own prompt cache gives the same
greedy reply as a cold prefill in 11 of the 21 requests, and oMLX's in 14 of
21. The rest share their opening and then diverge, which is what a different
prefill chunking does to floating-point sums. On Llama, serve's restored
prefix gives the identical reply (`tests/test_vllm_mlx_serve.py`), because
Pion stores the fp16 rows bit for bit.

Method notes. Every server got the same request bodies; Claude Code's
mid-conversation `system` messages were folded into the user turn before them
for all of them, because Ollama, given them as recorded, drops the tool
definitions from the prompt. Ollama ran its
`llama3.2:1b` build (GGUF Q8_0), the others the MLX 4-bit weights, so its
prompt counts differ; on Codex they alternate between about 4K and 11K tokens
from one turn to the next, so Ollama renders some of those requests without
part of their content, and its Codex reuse is measured on those prompts. "Reused" is what each server reports as cached
(`cache_read_input_tokens`); oMLX's own log agreed with its usage on every
request. A reply with no streamed content is timed to the end of its stream:
LM Studio streamed none for 13 of its 21 Claude Code replies on Llama, ended 7
of its 21 on Gemma with "Failed to generate a valid tool call", and ended 9 of
its 20 Codex replies with "Failed to parse tool call", which the 64-token cap
provokes by cutting tool calls short; a request without usage is reported as
such.
Ollama was not run on Gemma for lack of disk. Versions: mlx-lm 0.31.3 on mlx
0.31.2, oMLX 0.7.0, Ollama 0.34.4, LM Studio 0.4.25 (MLX engine 1.13.1), Pion
0.9.7. The session was recorded from Claude Code 2.1.280 on 2026-09-23.

Harness and raw results:
[`agent_session_ab.py`](../benchmarks/reproducers/agent_session_ab.py),
[`results/agent_session_ab_2026_10_07/`](../benchmarks/reproducers/results/agent_session_ab_2026_10_07/).
The recorded session itself is not published (its requests carry Claude Code's
system prompt and the recording machine's paths); the harness has a recording
proxy to capture your own.

## What it does not do

- **Within one live session it ties a stock server.** The in-process tier is
  mlx-lm's own; Pion is consulted only when it holds a longer prefix than
  the process does.
- **Hybrid models are not stored.** Pion stores plain attention layers
  (mlx-lm `KVCache`). Sliding-window layers (Gemma 3 and 4) and recurrent
  ones (Qwen3.5, Mamba-style SSM layers) are skipped rather than
  approximated, so on those models serve behaves like stock mlx-lm: no
  reuse across a restart or between processes. oMLX 0.7 does keep them
  across a restart (see the table above).
- **One request at a time** while resume is on, on plain-attention models:
  the journal needs mlx-lm's single-request path, so batching is off there.
  One local user loses nothing by it. On a hybrid model there is nothing to
  resume from, so serve keeps mlx-lm's batched path.
- **It does not make a small model smart.** A 16 GB Mac runs 1B–8B models
  at 4 bits. serve changes how much of each prompt is recomputed, not the
  quality of the answer.

## How it stores a conversation

An agent's prompt only grows, and two sessions on one repository share their
opening, so the unit serve stores is a **lineage**: a tree of segments, each
holding the K/V rows for a run of token positions on top of its parent. A
segment grows in place while the conversation keeps extending it, and
branches only where a prompt diverges. Lookup is one round trip: the prompt
is cut into 64-token blocks with chained hashes, one `HMGET` finds the
longest stored block-aligned prefix, and the match is refined token by token.
serve stores the **prompt**, not the reply: agents re-render the assistant
turn from their own copy (reasoning stripped, tool calls re-serialized), so
storing generated rows would fork the lineage every turn instead of
extending it. The design notes are in the module docstrings of
[`serve.py`](https://github.com/pavelhorak/pion/blob/main/pion-vllm-mlx/pion_vllm_mlx/serve.py)
and
[`prefix_store.py`](https://github.com/pavelhorak/pion/blob/main/pion-vllm-mlx/pion_vllm_mlx/prefix_store.py).

## Tests

- `tests/test_vllm_mlx_serve.py` (gate tier, every push): the console
  script; Anthropic and Responses translation; and live on
  Llama-3.2-1B-Instruct-4bit, a miss, a second session reusing the prompt
  despite a different billing header, then serve SIGKILLed and restarted and
  the same request served from Pion with an identical greedy reply.
- `pion-vllm-mlx/tests/test_prefix_store.py`: restored-then-continued logits
  bit-identical to a locally built cache, fp16 and bf16.
- `tests/test_serve_prefix_budget.py`: the budget, eviction, durability of
  eviction across a SIGKILL, and WAL compaction.
