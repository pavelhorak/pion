# Pion Examples

Quick index. Start with the **5-minute demo** at the top; the rest are research reproducers.

---

## Start here

| Script | What it shows | Prerequisites |
|---|---|---|
| [`prompt_cache_demo.py`](prompt_cache_demo.py) | **Shared KV Cache** — the four `PionPromptCache` lines, timed: vanilla mlx-lm's cold prefill against five Pion requests (each a fetch and a full answer), TTFT for all of them, the median's ratio, outputs compared. Default `--prefix-tokens 2048`; `--prefix-tokens 33` shows the small end, where there is little prefill to save. | `pip install 'pion-vllm-mlx[mlx]'` · Pion running with `--kvcache` (the Homebrew service is, or `./pion-server --kvcache --metal-attention -w 1`); `--backend inproc` needs no server but re-prefills, so it prints ~1× |
| [`cag_legal_demo/`](cag_legal_demo/README.md) | **CAG-hybrid pion-serve** — 4 SCOTUS opinions, calibration, head-to-head vs Vector-RAG. Self-contained with corpus + calibration state. | pion-serve, Ollama |

---

## Feature demos

| Script | Feature | Notes |
|---|---|---|
| [`hybrid_retrieval_demo.py`](hybrid_retrieval_demo.py) | Hybrid Retrieval Cache | Chunk-id-keyed K/V, inproc + pion backends |
| [`sparse_mask_64k_niah.py`](sparse_mask_64k_niah.py) | Sparse-mask SDPA at 64K context | 100% NIAH, 0.78% prefix budget; use `--skip-vanilla` on 16 GB Mac |
| [`moe_expert_substrate_demo.py`](moe_expert_substrate_demo.py) | MoE expert paging | `MOE.EXPERT.*` — models beyond RAM |
| [`agent_memory_demo.py`](agent_memory_demo.py) | AI agent memory via MCP | `agent_remember` / `agent_recall` / `agent_forget` |
| [`ai_platform_scenario.py`](ai_platform_scenario.py) | Multi-command AI gateway scenario | Semantic cache + RAG + routing |
| [`cache_aware_router.py`](cache_aware_router.py) | `AI.ROUTE.*` semantic load balancer | 88% accuracy, 0.14 ms |
| [`prefix_routing_demo.py`](prefix_routing_demo.py) | `KV.PREFIX.*` routing | Cross-worker prefix directory |
| [`multinode_prefix_routing.py`](multinode_prefix_routing.py) | Cross-host prefix routing | pion-exo cluster |
| [`session_affinity_demo.py`](session_affinity_demo.py) | Session affinity routing | blake2b session-id hashing |
| [`pion_vs_ollama_demo.py`](pion_vs_ollama_demo.py) | Pion Serve vs raw Ollama latency | Semantic cache hit rate |
| [`coalescing_proxy.py`](coalescing_proxy.py) | Request coalescing proxy | Batch identical prompts |
| [`exo_session_demo.py`](exo_session_demo.py) | exo distributed inference hook | pion-exo + KV.PREFIX |
| [`pion_glide_demo.py`](pion_glide_demo.py) | Valkey GLIDE client | GLIDE cluster client compat |

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

## Modules, not demos

Two files here are imported by the demos rather than run directly:

- `pion_moe_tier.py` — the MoE expert substrate, used by
  [`moe_expert_substrate_demo.py`](moe_expert_substrate_demo.py).
- `pion_moe_tier_client.py` — its wire-backed client, also used by
  pion-serve's `pion-moe` backend.

Both were moved here from the private research tree so the demo and the
backend actually run outside the development repo.
