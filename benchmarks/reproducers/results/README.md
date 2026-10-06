# Reproducer results

The raw output behind the numbers the docs quote from these reproducers, one
file per run, as the script wrote it (the hybrid run's `config.out` field had a
local scratch path and now holds the file's own name).

| File | Script | Machine | Backs |
|---|---|---|---|
| [`cross_process_ttft_2026_10_02.json`](cross_process_ttft_2026_10_02.json) | [`../cross_process_ttft.py`](../cross_process_ttft.py) `--same` | M4 Mac mini, Pion 0.9.1 | TTFT 20× (same process) and 17× (separate process) at a 2,049-token prefix; 11× / 4.5× / 1.5× at 1,035 / 268 / 34 tokens |
| [`stage1_qwen3_5_prefix_sweep_2026_10_02.json`](stage1_qwen3_5_prefix_sweep_2026_10_02.json) | [`../sweep_qwen3_5_warm_ttft.py`](../sweep_qwen3_5_warm_ttft.py) | M4 Mac mini, Pion 0.9.1 | Qwen3.5-4B warm TTFT 14.0× / 25.3× / 29.0× at 2K / 4K / 8K |
| [`stage1_hybrid_results_2026_10_02.json`](stage1_hybrid_results_2026_10_02.json) | [`../stage1_hybrid_recall_bench.py`](../stage1_hybrid_recall_bench.py) | M4 Mac mini, Pion 0.9.1 | `HybridRetrievalCache` 3.0× (inproc) and 2.7× (pion) p50 TTFT, 98.3% token agreement, 100 SQuAD v2 queries |
