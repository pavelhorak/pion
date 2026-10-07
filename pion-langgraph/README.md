# pion-langgraph

> **Experimental.** pion-langgraph is covered by one gate-tier test, `tests/test_framework_integrations.py`, which exercises its basic calls. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](../README.md#experimental)).

LangGraph checkpoint saver backed by Pion — deterministic low-latency agent-state persistence.

`PionSaver` implements LangGraph's `BaseCheckpointSaver` against a running Pion server. Uses only RESP2-stable commands (HSET / HGET / HGETALL, single-key ZADD / ZREVRANGE) — no RedisJSON, no Lua scripting, no Sentinel. Works against any Pion deployment, single-worker or sharded.

## Install

```bash
pip install -e pion-langgraph/
# optional: dev tools + LangGraph for the test suite
pip install -e pion-langgraph/[dev]
```

Requires a running Pion server:

```bash
./pion-server --kvcache -w 1   # binds 127.0.0.1:1974
```

## Usage

```python
from pion_langgraph import PionSaver

saver = PionSaver(host="127.0.0.1", port=1974, key_prefix="lgcp")
graph = workflow.compile(checkpointer=saver)
result = graph.invoke({"input": "hello"})
# Saver persists every superstep to Pion; resume on next invocation
# is a single HGETALL round-trip (~30 µs hot path).
```

## Docs

- Main project: [`../README.md`](../README.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) in this directory. Pion's client packages are
deliberately permissive so they can be vendored into any stack; the Pion **server**
is Apache-2.0 too, with one closed binary library for its tuned vector kernels —
see the top-level [`LICENSE`](../LICENSE).

Which parts of Pion are under which licence: [`doc/licensing.md`](../doc/licensing.md).
