# pion-llamaindex

> **Experimental.** pion-llamaindex is covered by one gate-tier test, `tests/test_framework_integrations.py`, which exercises its basic calls. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](../README.md#experimental)).

LlamaIndex `VectorStore` backed by Pion's HNSW index — sub-millisecond RAG retrieval.

`PionVectorStore` plugs into LlamaIndex's vector-store interface so any LlamaIndex pipeline (RAG, agents, query engines) can use Pion's quantization-aware HNSW (INT4 / INT3 / INT2 variants) for retrieval without any LlamaIndex protocol changes.

## Install

```bash
pip install -e pion-llamaindex/
# optional: dev tools + LlamaIndex for the test suite
pip install -e pion-llamaindex/[dev]
```

Requires a running Pion server:

```bash
./pion-server --profile vector -w 1   # binds 127.0.0.1:1974
```

## Usage

```python
from pion_llamaindex import PionVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document

store = PionVectorStore(host="127.0.0.1", port=1974,
                        index_name="docs", dimensions=1536)
storage = StorageContext.from_defaults(vector_store=store)
idx = VectorStoreIndex.from_documents([Document(text="...")], storage_context=storage)
print(idx.as_query_engine().query("what is X?"))
```

Pion handles the embedding-similarity search (cosine, INT8 SIMD); LlamaIndex still owns chunking, prompt assembly, and the LLM call. Default `dimensions=1536` matches OpenAI's `text-embedding-3-small`; override for other embedders.

## Docs

- Main project: [`../README.md`](../README.md)
- Vector engine internals + quantization variants: [`../doc/vector_engine.md`](../doc/vector_engine.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) in this directory. Pion's client packages are
deliberately permissive so they can be vendored into any stack; the Pion **server**
is Apache-2.0 too, with one closed binary library for its tuned vector kernels —
see the top-level [`LICENSE`](../LICENSE).

Which parts of Pion are under which licence: [`doc/licensing.md`](../doc/licensing.md).
