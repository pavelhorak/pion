# `RAG.*` · `AI.KNN_LM.*` · `NEURON.PKM.*`

Three substrate families that share one property: they are datastores an
inference loop reads at every step, so latency is the whole spec.

## `RAG.*` — speculative RAG

<!-- include-section: doc/command_matrix.md | ## 19. Speculative RAG Commands (`--kvcache`) -->

## `AI.KNN_LM.*` and `NEURON.PKM.*`

<!-- include-section: README.md | ### AI Gateway (wire-native) -->

Product-key memory (`NEURON.PKM.*`) gives exact top-k over N = S² slots at
2·√N·(dim/2) MACs — 1M slots, dim 896, k=32 answers in 0.075 ms (p50) where the
kNN-LM HNSW takes 5.13 ms (`tests/test_neuron_pkm.py --bench`, M4 Mac mini,
[raw output](https://github.com/pavelhorak/pion/blob/main/benchmarks/results/2026-10-06-mac-m4/neuron_pkm_bench.txt)).
