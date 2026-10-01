# `KV.PREFIX.*` — the prefix registry

A *namespace key* names one exact token prefix for one model and quantization;
the convention is `app|version|model|vquant|prompt-id`. Register it once, look
it up from any process, and fetch the K/V through the [V-store](v-store.md).
All of these need `pion-server --kvcache`. From Python,
[`PionPromptCache`](/reference/python/pion-vllm-mlx.md) issues every command on
this page for you.

<!-- include-section: doc/shared_kv_cache.md | ## Wire Protocol | shift:-1 -->

## Related

- [Shared KV cache](/concepts/shared-kv-cache.md) — the concept, Stage 1 vs Stage 2, namespaces, durability.
- [`V.*`](v-store.md) — `V.STOREBATCH` and `V.FETCH … BATCH`, the commands that move the bytes.
- [`ATTEND.*`](attend.md) — when Pion runs the attention itself.
