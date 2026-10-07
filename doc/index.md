# Pion — the memory engine for AI inference

**Pion remembers what your model already read:** prefill a long prompt once,
and every later request, from any process and even after a crash, starts at
the first new token. It is a memory engine for AI inference — one small Mojo
binary that holds the prompt's K/V cache, and the rest of a model's working
memory, and serves it over the Redis wire.

On Apple Silicon it is four lines around `mlx_lm`:

```python
from pion_vllm_mlx import PionPromptCache

pc = PionPromptCache(model, host="127.0.0.1", port=1974)                       # once per loaded model
cache = pc.get_or_prefill(prefix_ids, namespace="app|v1|llama-1b|system_v1")  # MISS: prefill once + store · HIT: fetch
text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)             # decode as usual, at native speed
```

| Time to first token, Apple Silicon: Llama-3.2-1B-4bit, 2,049-token prefix, 16-token question | first token | vs cold |
|---|---:|---:|
| Cold prefill, vanilla mlx-lm | 1,193 ms | |
| mlx-lm's own prompt-cache file, read by a fresh process (67 MB) | **37.0 ms** | 32× |
| Pion, **same process** | 46.2 ms | 26× |
| Pion, **from a separate process**, over the wire | 69.0 ms | 17× |

The two Pion rows differ only by *where the cache comes from*: the first is the
process that computed it, the second fetches what a separate process wrote and
reproduces the vanilla output at BLEU 1.000. Both come from
[`cross_process_ttft.py`](../benchmarks/reproducers/cross_process_ttft.py)
(`--same` adds the first); the file row from
[`file_cache_ttft.py`](../benchmarks/reproducers/file_cache_ttft.py), timed the
same way. **A file is faster:** if one program reuses one fixed prefix, use
mlx-lm's `save_prompt_cache` / `load_prompt_cache`. Pion is for what a file
does not do: a cache every process reads over one wire, crash-consistent, and
through [`pion-vllm-mlx serve`](coding_agents.md) a longest-prefix match that a
restarted agent or a second session finds by itself.
Read the limits before installing: it caches prefill, not decode, and it
only pays off when a long prefix is really reused.

## What Pion is, and is not

Pion is a memory engine: the prompt's K/V, recurrent (SSM) state, MoE expert
weights, vectors and embeddings, and agent memory — behind one wire protocol,
with one durability story. Two properties are Pion's own: an acked write is
still there after `SIGKILL`, and the cache is read directly by a *different
program* over a protocol every language already has a client for.

It is **not** an inference engine: it feeds forward passes, it does not run
them. The vector engine exists so that recall needs no second database, and
[Pion Serve](pion_serve.md) is an example proxy, not a requirement.

## Where to start

- **Get started** — [Shared KV cache](shared_kv_cache.md): namespaces, Stage 1
  vs Stage 2, the wire protocol, what is measured.
- **Reference** — [Command matrix](command_matrix.md) for every Redis-compatible
  command and its known divergences; [Configuration](configuration.md) for
  profiles; [Networking](networking.md) for the event-loop tiers and the
  binary lane on `port+1`; [Client APIs](client_apis.md).
- **Operations** — [Running in production](operations.md): crash breadcrumbs,
  the status file, supervised serving, durability of every type;
  [Persistence](persistence.md); [Multi-tenant](multi_tenant.md);
  [Distributed systems](distributed_systems.md) (a preview — read its caveats).
- **Internals** — [Architecture](architecture.md), [Memory
  management](memory_management.md), [Vector engine](vector_engine.md).
- **Licensing** — [`licensing.md`](licensing.md): Pion is Apache-2.0,
  with one free closed binary library (`libpion_vector`) for the tuned vector
  kernels.

## Also here

[AI gateway](ai_gateway.md) · [Embeddings](embeddings.md) ·
[Pion Serve](pion_serve.md) · [Data types](data_types.md) ·
[Benchmarking guide](benchmarking_guide.md) · [Development guide](development_guide.md)
