# Pion-native commands

The commands that have no Redis equivalent, by family. Each page is assembled
at build time from the document that specifies the family, so the wire forms
here are the wire forms the server implements. The
[command index](/reference/command-index.md) lists every command the engine
dispatches, generated from the same table `MULTI` validates against.

| Family | Needs | What it is |
|---|---|---|
| [`KV.PREFIX.*`](kv-prefix.md) | `--kvcache` | The prefix registry: register a prompt prefix once, look it up from any process, block-hash membership for cache-aware routers |
| [`V.*`](v-store.md) | `--kvcache` | The V-store the prefix cache is built on: token-indexed K/V storage, `V.STOREBATCH` / `V.FETCH … RANGE` / `BATCH`, `V.EXPORT` (the same-machine file lane), quantization tiers |
| [`ATTEND.*`](attend.md) | `--kvcache` (+ `--metal-attention`) | Stage 2: Pion computes the attention over cached K/V itself, including the sparse long-context selector |
| [`SSM.PREFIX.*`](ssm-prefix.md) | `--kvcache` | Recurrent-state companion for Mamba / GDN / hybrid models |
| [`MOE.EXPERT.*`](moe-expert.md) | `--moe-cache DIR` | MoE expert paging: tiered expert weights, access histograms, HIST-guided pruning |
| [`FT.*` and VSET](vector.md) | any profile with HNSW | Vector search, BM25 and hybrid retrieval; the Redis 8 VSET commands (one vector set per key) |
| [`AI.*`](ai.md) | `--nle-embed` / an embedding backend | Semantic cache, embeddings, routing, the gateway commands |
| [`RAG.*` · `AI.KNN_LM.*` · `NEURON.PKM.*`](rag-knn-pkm.md) | `--kvcache` | Speculative RAG, the kNN-LM datastore, product-key memory |
| [`PION.STATS` and `INFO`](stats.md) | — | The value receipt and the server's telemetry |
