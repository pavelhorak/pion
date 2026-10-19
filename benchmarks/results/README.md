# Published benchmark results

Every measured number in this repository's docs points at raw output that a
harness in this repository produced. `tools/check_doc_claims.py` enforces it:
it finds each number with a performance unit in the published docs and fails
unless `benchmarks/claims.toml` names its evidence (a raw result here, a test
that asserts it, or the source line that defines it).

| Directory | What | Machine |
|---|---|---|
| [`2026-10-19-mac-m4-drop-in/`](2026-10-19-mac-m4-drop-in/README.md) | Vector search without `FT.OPTIMIZE`: a stock client against today's path, the streaming case, INT8 score error | M4 Mac mini |
| [`2026-10-19-mac-m4-drop-in-b/`](2026-10-19-mac-m4-drop-in-b/README.md) | Rebuilding a served vector index: the INT8 delta's cost, the streaming case and a bulk load against phase A | M4 Mac mini |
| [`2026-10-18-linux-epyc-8124p/vector_field/`](2026-10-18-linux-epyc-8124p/vector_field/README.md) | Vector search: Pion against RediSearch (HNSW FP32, SVS 8-bit) and valkey-search (HNSW FP16, FP32), 50K and 500K, matched recall, time until searchable, streaming | AMD EPYC 8124P |
| [`2026-10-12-linux-epyc-8124p/`](2026-10-12-linux-epyc-8124p/README.md) | One keyspace: Pion `--io-threads` against Redis `io-threads`, throughput and durable | AMD EPYC 8124P |
| [`2026-10-06-mac-m4/`](2026-10-06-mac-m4/README.md) | Prompt-cache lanes and TTFT, Stage-1 workload, BLEU, Metal SDPA, PKM, `ATTEND.*`, failover, per-profile memory | M4 Mac mini |
| [`../reproducers/results/`](../reproducers/results/README.md) | The reproducers' runs: cross-process TTFT, the Qwen3.5 prefix sweep, hybrid retrieval | M4 Mac mini |

If a number does not reproduce on your machine, open an issue with your
hardware and the raw output.
