# Published benchmark results

Every measured number in this repository's docs points at raw output that a
harness in this repository produced. `tools/check_doc_claims.py` enforces it:
it finds each number with a performance unit in the published docs and fails
unless `benchmarks/claims.toml` names its evidence (a raw result here, a test
that asserts it, or the source line that defines it).

| Directory | What | Machine |
|---|---|---|
| [`2026-10-06-mac-m4/`](2026-10-06-mac-m4/README.md) | Prompt-cache lanes and TTFT, Stage-1 workload, BLEU, Metal SDPA, PKM, `ATTEND.*`, failover, per-profile memory | M4 Mac mini |
| [`../reproducers/results/`](../reproducers/results/README.md) | The reproducers' runs: cross-process TTFT, the Qwen3.5 prefix sweep, hybrid retrieval | M4 Mac mini |

If a number does not reproduce on your machine, open an issue with your
hardware and the raw output.
