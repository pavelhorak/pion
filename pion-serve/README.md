# pion-serve

OpenAI-compatible inference proxy with semantic cache, fragment-level distillation, RAG, and semantic intent routing.

```
Client → pion-serve (:8080) → Backend (ollama / vLLM / OpenAI / Claude / Gemini / llama.cpp / MAX / pion-moe / pion-cag-hybrid)
              │
              ├── L1: semantic cache (cosine on query embedding → cached response)
              ├── L3: inference distillation (concept synthesis, --distill)
              ├── L3b: fragment cache (sentence-level paraphrase reuse)
              ├── RAG: FT.SEARCH vector index → prepend context
              └── Intent routing (--route): cheap / mid / top-tier model selection
```

How much traffic each layer catches depends on how repetitive the traffic is. No hit rate or cost saving is published with a harness yet, so this README states none; `/v1/stats` reports both for your own traffic.

## Status

- **Threading correctness** ✅ landed (`_stats_lock`, `_l3_response_lock`, `_once_under_init_lock`); see [`tests/`](tests/).
- **Packaging** (`pyproject.toml` + console entry point) — **not done yet**. The package is currently used via `python pion-serve/serve.py …` and the in-tree `from concept_store import …` imports. Folded into the OSS launch readiness work; when it lands, you'll be able to `pip install pion-serve` and run `pion-serve …`.

In the meantime, run it directly:

## Usage

```bash
# Ensure Pion is up:
./pion-server --kvcache -w 1

# Basic proxy with Ollama backend + semantic cache (L1):
python pion-serve/serve.py --backend ollama --model gemma4:e4b

# + L3 distillation (concept synthesis):
python pion-serve/serve.py --backend ollama --model gemma4:e4b --distill

# + RAG injection from a pre-built Pion vector index:
python pion-serve/serve.py --backend ollama --rag-index my_docs

# + Semantic intent routing (Anthropic three-tier default):
python pion-serve/serve.py --route --route-config pion-serve/routing.json

# + Pion-MoE in-process backend (cross-arch MoE via MOE.EXPERT.* substrate):
python pion-serve/serve.py --backend pion-moe \
    --pion-moe-model-path /path/to/snapshot --pion-moe-model-id model_v1

# + Pion CAG-hybrid (cascade-gated CAG + RAG):
python pion-serve/serve.py --backend pion-cag-hybrid \
    --cag-foundation-corpus ./corpus.txt \
    --cag-calibration-state ./calibration.json

# + Modular MAX Serve behind the semantic cache / router (request layer):
max serve --model-path modularai/Llama-3.1-8B-Instruct-GGUF   # :8000, OpenAI-compatible
python pion-serve/serve.py --backend max --model llama-3.1-8b-instruct
```

Backends supported: `ollama`, `vllm`, `openai`, `gemini`, `claude`, `llamacpp`, `max`, `pion-moe`, `pion-cag-hybrid`. The `max` backend proxies Modular MAX Serve at the request layer (semantic cache + routing); it is **not** a KV-datapath integration — MAX 26.4 removed the LMCache connector and its native KVConnector factory is closed. Embedding backend cascade: Pion auto-embed (default) → SIE (`--sie-url`) → Ollama; pin explicitly with `--embed-backend {pion,sie,ollama,none}`.

Stats at `/v1/stats` (per-backend embed call counters, L1/L3 hit rates, RAG injection counts, CAG-hybrid gate kept/fallback ratios, MLX sidecar telemetry if `--mlx-tcp-port` is set).

## Tests

```bash
python pion-serve/tests/test_intent_router.py        # 15 routing-logic tests
python pion-serve/tests/test_thread_safety.py        # 6 lock-correctness tests
```

Both run offline (no backend required).

## Docs

- Main project: [`../README.md`](../README.md)
- Pion Serve design + cache cascade: [`../doc/pion_serve.md`](../doc/pion_serve.md)
- SIE embed integration: [`tests/test_pion_serve_sie_embed.py`](../tests/test_pion_serve_sie_embed.py)

## License

Apache-2.0 — the repository licence. See [`../LICENSE`](../LICENSE) and the map in [`../doc/licensing.md`](../doc/licensing.md).
