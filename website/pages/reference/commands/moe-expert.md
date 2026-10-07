# `MOE.EXPERT.*` — expert paging

> **Experimental.** `MOE.EXPERT.*` is wire-tested by `tests/run_substrate_gate.py` (gate tier) and `tests/test_moe_*.py` (full tier); decode through it is research-grade, and no model-size, hit-latency or pruning measurement is published with a harness. No measurement of what it buys is published and nobody outside this project is known to use it, so it is outside Pion's supported surface and may change or be removed. Supported: the prompt cache and `pion-vllm-mlx serve`, the Redis-compatible KV with its WAL, and vector search with the semantic cache ([README](https://github.com/pavelhorak/pion#experimental)).

Expert weights for MoE models live in a tiered cache (per-worker RAM LRU →
SSD → network) and are served over the wire, so that a model larger than
physical memory can run. Enable with `--moe-cache DIR --moe-cache-mib N`.

- **HIST-guided pruning**: per-distribution access histograms (up to 8
  namespaces per model) drive the prune set. Quality depends on the traffic:
  collect histograms on traffic like yours and check perplexity on it before
  pruning.
- **Multi-model**: 4 concurrent models per worker over a shared LRU; stacked
  bf16, per-expert INT4 and stacked INT4 expert layouts.

## Commands

<!-- include-section: doc/command_matrix.md | ## 16b. MoE Expert Paging — `MOE.EXPERT.*` (`--moe-cache <DIR> --moe-cache-mib N`) -->

