# The memory planes

Pion is one process that holds the memory an inference stack otherwise
spreads over four services: the prompt's K/V tensors, recurrent (SSM) state,
MoE expert weights, vectors and embeddings, and the agent's own memory —
behind one wire protocol, with one durability story.

## The planes

| Plane | Commands | Reference |
|---|---|---|
| Prefix K/V cache | `KV.PREFIX.*`, `V.*` | [KV.PREFIX.*](/reference/commands/kv-prefix.md) · [V-store](/reference/commands/v-store.md) |
| Attention over cached K/V | `ATTEND.*`, `ATTEND.PREFIX.*` | [ATTEND.*](/reference/commands/attend.md) |
| Recurrent state | `SSM.PREFIX.*` | [SSM.PREFIX.*](/reference/commands/ssm-prefix.md) |
| MoE expert paging | `MOE.EXPERT.*` | [MOE.EXPERT.*](/reference/commands/moe-expert.md) |
| Vectors and retrieval | `FT.*`, `VADD`/`VSIM` and the rest of VSET, BM25, hybrid | [FT.* and VSET](/reference/commands/vector.md) |
| Semantic cache and routing | `AI.*`, `RAG.*`, `AI.KNN_LM.*`, `NEURON.PKM.*` | [AI.*](/reference/commands/ai.md) · [RAG and friends](/reference/commands/rag-knn-pkm.md) |
| Key-value | the Redis command surface | [Redis-compatible commands](/command_matrix.md) |

Every plane speaks RESP2/RESP3, so any Redis client can reach it. The K/V
planes also have a binary lane on port+1 for large tensor transfers.

## What Pion is not

- **Not an inference engine.** Pion does not run forward passes. mlx-lm, MLX,
  vLLM and similar runtimes run the model and use Pion as their memory.
- **Not a general vector database.** The vector engine exists so that recall
  does not need a second database. It is one HNSW index per server
  ([vector engine](/vector_engine.md)).
- **Not an AI gateway.** [Pion Serve](/pion_serve.md) is an example proxy
  built on the planes above. It is not required to use them.

The integrations (`pion-vllm-mlx`, the MCP server, the LangGraph, AutoGen and
LlamaIndex adapters, `pion-exo`) are separate packages that talk to the server
over the wire.
