# `RAG.*` · `AI.KNN_LM.*` · `NEURON.PKM.*`

> **Experimental.** `RAG.*`, `AI.KNN_LM.*` and `NEURON.PKM.*` are wire-tested: `tests/test_ai_gateway.py` (gate tier) sends `RAG.*`, and `tests/test_ai_knn_lm.py` and `tests/test_neuron_pkm.py` (full tier) cover the other two. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](https://github.com/pavelhorak/pion#experimental)).

Three substrate families that share one property: they are datastores an
inference loop reads at every step, so latency is the whole spec.

## `RAG.*` — speculative RAG

<!-- include-section: doc/command_matrix.md | ## 19. Speculative RAG Commands (`--kvcache`) -->

## `AI.KNN_LM.*` and `NEURON.PKM.*`

`AI.KNN_LM.*` is a token-id-tagged kNN datastore for client-side kNN-LM
augmentation (`CREATE`, `STORE`, `STOREBATCH`, `QUERY`, `INFO`, `DROP`), up to
16 named datastores per worker. Below 5K entries it scans by brute force;
above, it searches a per-vector SQ8 HNSW with asymmetric INT8 SIMD distances
(M=32). The intended uses are code completion, log generation and in-domain
text infill.

Product-key memory (`NEURON.PKM.*`) gives exact top-k over N = S² slots at
2·√N·(dim/2) MACs — 1M slots, dim 896, k=32 answers in 0.075 ms (p50) where the
kNN-LM HNSW takes 5.13 ms (`tests/test_neuron_pkm.py --bench`, M4 Mac mini,
[raw output](https://github.com/pavelhorak/pion/blob/main/benchmarks/results/2026-10-06-mac-m4/neuron_pkm_bench.txt)).
