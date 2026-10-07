# The memory planes

Pion is one process that holds the memory an inference stack otherwise
spreads over four services: the prompt's K/V tensors, recurrent (SSM) state,
MoE expert weights, vectors and embeddings, and the agent's own memory —
behind one wire protocol, with one durability story.

## The planes

| Plane | Commands | Status | Reference |
|---|---|---|---|
| Prefix K/V cache | `KV.PREFIX.*`, `V.*` | supported | [KV.PREFIX.*](/reference/commands/kv-prefix.md) · [V-store](/reference/commands/v-store.md) |
| Attention over cached K/V | `ATTEND.*`, `ATTEND.PREFIX.*` | supported | [ATTEND.*](/reference/commands/attend.md) |
| Recurrent state | `SSM.PREFIX.*` | supported | [SSM.PREFIX.*](/reference/commands/ssm-prefix.md) |
| Vectors and retrieval | `FT.*`, `VADD`/`VSIM` and the rest of VSET, BM25, hybrid | supported | [FT.* and VSET](/reference/commands/vector.md) |
| Semantic cache | `AI.SEMANTIC_CACHE`, `AI.EMBED` | supported | [AI.*](/reference/commands/ai.md) |
| Key-value | the Redis command surface | supported | [Redis-compatible commands](/command_matrix.md) |
| MoE expert paging | `MOE.EXPERT.*` | **experimental** | [MOE.EXPERT.*](/reference/commands/moe-expert.md) |
| Routing, RAG, kNN-LM, product keys | `AI.ROUTE.*`, `RAG.*`, `AI.KNN_LM.*`, `NEURON.PKM.*`, `AI.COMPLETE`, `AI.CHAT` | **experimental** | [AI.*](/reference/commands/ai.md) · [RAG and friends](/reference/commands/rag-knn-pkm.md) |

An experimental plane is in the tree and tested as far as its page says, with
no published measurement of what it buys and no user outside this project; it
may change or be removed.

Every plane speaks RESP2/RESP3, so any Redis client can reach it. The K/V
planes also have a binary lane on port+1 for large tensor transfers.

## What Pion is not

- **Not an inference engine.** Pion does not run forward passes. mlx-lm, MLX,
  vLLM and similar runtimes run the model and use Pion as their memory.
- **Not a general vector database.** The vector engine exists so that recall
  does not need a second database. It is one HNSW index per server
  ([vector engine](/vector_engine.md)).
- **Not an AI gateway.** [Pion Serve](/pion_serve.md) is an experimental
  example proxy built on the planes above. It is not required to use them.

The integrations are separate packages that talk to the server over the wire.
`pion-vllm-mlx` (the prompt cache and `pion-vllm-mlx serve`) is supported; the
MCP server, the LangGraph, AutoGen and LlamaIndex adapters and `pion-exo` are
experimental.
