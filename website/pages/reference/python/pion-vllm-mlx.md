# `pion-vllm-mlx`

The client for the prompt-cache path on Apple Silicon: `PionPromptCache` is
the four-line drop-in for `mlx_lm.make_prompt_cache`, `HybridRetrievalCache`
does the same for retrieved RAG chunks, and the Stage-2 attention patch lets
Pion run the attention over the cached prefix itself. Apache-2.0; the
signatures below are read from the source at build time.

```
pip install 'pion-vllm-mlx[mlx]'
```

## `PionPromptCache`

::: pion_vllm_mlx.prompt_cache.PionPromptCache
    options:
      members:
        - get_or_prefill
        - lookup
        - stats

## `HybridRetrievalCache`

::: pion_vllm_mlx.hybrid_retrieval.HybridRetrievalCache
    options:
      members:
        - ingest
        - prepare

## Stage 2 — the mlx-lm attention patch

::: pion_vllm_mlx.mlx_lm_patch.install_pion_attention_patch

::: pion_vllm_mlx.mlx_lm_patch.make_pion_prompt_cache

::: pion_vllm_mlx.mlx_lm_patch.PionPrefixCache
    options:
      members:
        - state

## The seam with mlx-lm

::: pion_vllm_mlx._compat.PionMlxCompatError

## Package README

<!-- include-file: pion-vllm-mlx/README.md | strip-h1 | shift:2 -->
