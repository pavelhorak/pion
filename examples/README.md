# Pion Examples

Quick index. Start with the **5-minute demo** at the top; the rest are research reproducers.

---

## Start here

| Script | What it shows | Prerequisites |
|---|---|---|
| [`prompt_cache_demo.py`](prompt_cache_demo.py) | **Shared KV Cache** — the four `PionPromptCache` lines, timed: vanilla mlx-lm's cold prefill against five Pion requests (each a fetch and a full answer), TTFT for all of them, the median's ratio, outputs compared. Default `--prefix-tokens 2048`; `--prefix-tokens 33` shows the small end, where there is little prefill to save. | `pip install 'pion-vllm-mlx[mlx]'` · Pion running with `--kvcache` (the Homebrew service is, or `./pion-server --kvcache --metal-attention -w 1`); `--backend inproc` needs no server but re-prefills, so it shows no speedup |

---

## Feature demos

| Script | Feature | Notes |
|---|---|---|
| [`hybrid_retrieval_demo.py`](hybrid_retrieval_demo.py) | Hybrid Retrieval Cache | Chunk-id-keyed K/V, inproc + pion backends |
| [`sparse_mask_64k_niah.py`](sparse_mask_64k_niah.py) | Sparse-mask SDPA at 64K context | Single-needle NIAH with a sparse prefix budget; use `--skip-vanilla` on a 16 GB Mac |
| [`agent_memory_demo.py`](agent_memory_demo.py) | Agent memory across a restart | `pion_memory`'s `remember` / `recall`; no model needed |
| [`ai_platform_scenario.py`](ai_platform_scenario.py) | Multi-command AI gateway scenario | Semantic cache + RAG + routing |
| [`prefix_routing_demo.py`](prefix_routing_demo.py) | `KV.PREFIX.*` routing | Cross-worker prefix directory |
| [`multinode_prefix_routing.py`](multinode_prefix_routing.py) | Cross-host prefix routing | two OpenAI-compatible endpoints; stub mode needs none |
| [`session_affinity_demo.py`](session_affinity_demo.py) | Session affinity routing | blake2b session-id hashing |
| [`pion_vs_ollama_demo.py`](pion_vs_ollama_demo.py) | `AI.COMPLETE`'s semantic cache in front of Ollama, against raw Ollama | Semantic cache hit rate |
| [`coalescing_proxy.py`](coalescing_proxy.py) | Request coalescing proxy | Batch identical prompts |
| [`exo_session_demo.py`](exo_session_demo.py) | exo head-node session routing | an exo head node, or the built-in stub |

---

## Research reproducers (step-series)

The `step*.py` files are an incremental research series exploring speculative/kNN-LM inference. They are not maintained as standalone demos.

| Script | Topic |
|---|---|
| `step6_flare.py`, `step6b_flare_fewshot.py` | FLARE mid-generation retrieval |
| `step7_rest_pion_drafter.py` | REST-based speculative draft |
| `step8_rest_13b.py`, `step9_knnlm.py` | kNN-LM baseline |
| `step10_vocab_bias.py`, `step11_rest_probabilistic.py` | Vocabulary bias; REST probabilistic draft |
| `step12_kv_cache_aligned.py`, `step13_flare_mojo.py` | KV-cache-aligned draft; FLARE in Mojo |
