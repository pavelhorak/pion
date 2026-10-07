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

## What it does not do

- **Within one live session it ties a stock server.** The in-process tier is
  mlx-lm's own; Pion is consulted only when it holds a longer prefix than
  the process does.
- **Hybrid models are not stored yet.** Pion stores plain attention layers
  (mlx-lm `KVCache`). Sliding-window layers (Gemma 3 and 4) and recurrent
  ones (Qwen3.5, Mamba-style SSM layers) are skipped rather than
  approximated, so on those models serve behaves like stock mlx-lm: no
  reuse across a restart or between processes.
- **One request at a time** while resume is on: the journal needs mlx-lm's
  single-request path, so batching is off. One local user loses nothing by it.
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
