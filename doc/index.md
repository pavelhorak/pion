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

| Time to first token, Apple Silicon | vanilla mlx-lm | Pion warm | |
|---|---:|---:|:---:|
| Llama-3.2-1B-4bit, 2,048-token prefix, **same process** | 1,530 ms | **30.2 ms** | **50.6×** |
| Same model and prefix, **from a separate process**, over the wire | 1,558 ms | 64.7 ms | **24×** |

The two rows differ by *where the attention runs*; the second pays a wire hop
for a cache a separate process wrote, and reproduces the vanilla output at
BLEU 1.000. Each row has a reproducer:
[`tests/bench_ttft.py`](../tests/bench_ttft.py) and
[`cross_process_ttft.py`](../benchmarks/reproducers/cross_process_ttft.py).
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
[Benchmarking guide](benchmarking_guide.md) · [Development guide](development_guide.md) ·
[Codebase search](codebase_search.md)
