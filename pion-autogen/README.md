# pion-autogen

> **Experimental.** pion-autogen is covered by one gate-tier test, `tests/test_framework_integrations.py`, which exercises its basic calls. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](../README.md#experimental)).

AutoGen memory backend powered by Pion's HNSW vector index — semantic recall for multi-agent systems.

`PionMemoryStore` implements AutoGen Core's `Memory` interface against a running Pion server, so any AutoGen agent can persist and recall facts with sub-millisecond cosine-similarity lookup. No RedisJSON dependency; uses the wire-compatible RESP2 path.

## Install

```bash
pip install -e pion-autogen/
# optional: dev tools + AutoGen for the test suite
pip install -e pion-autogen/[dev]
```

Requires a running Pion server on the host/port you pass:

```bash
./pion-server --kvcache -w 1   # binds 127.0.0.1:1974
```

## Usage

```python
from pion_autogen import PionMemoryStore

mem = PionMemoryStore(host="127.0.0.1", port=1974, index_name="agent_a",
                       dimensions=384, embed_provider="ollama")
await mem.add("the user lives in Bratislava and prefers metric units")
hits = await mem.query("what city is the user in?")
print(hits[0].content)   # → "the user lives in Bratislava ..."
```

Supports `embed_provider` = `"mock"` (deterministic, for tests), `"openai"`, or `"ollama"`. Embedding dim defaults to 384 (MiniLM-L6-v2) when paired with Pion's `--auto-embed` sidecar.

## Docs

- Main project: [`../README.md`](../README.md)
- Vector engine internals: [`../doc/vector_engine.md`](../doc/vector_engine.md)
- Substrate this backend uses: `FT.CREATE` / `FT.SEARCH` over `HSET` documents (see [`../doc/vector_engine.md`](../doc/vector_engine.md))

## License

Apache-2.0 — see [`LICENSE`](LICENSE) in this directory. Pion's client packages are
deliberately permissive so they can be vendored into any stack; the Pion **server**
is Apache-2.0 too, with one closed binary library for its tuned vector kernels —
see the top-level [`LICENSE`](../LICENSE).

Which parts of Pion are under which licence: [`doc/licensing.md`](../doc/licensing.md).
