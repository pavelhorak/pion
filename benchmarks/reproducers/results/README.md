# Reproducer results

The raw output behind the numbers the docs quote from these reproducers, one
file per run, as the script wrote it (the hybrid run's `config.out` field had a
local scratch path and now holds the file's own name).

| File | Script | Machine | Backs |
|---|---|---|---|
| [`cross_process_ttft_2026_10_07.json`](cross_process_ttft_2026_10_07.json) | [`../cross_process_ttft.py`](../cross_process_ttft.py) `--prefix-tokens 34 268 1035 2049 --pairs 5 --same` | M4 Mac mini, Pion 0.9.6 | TTFT 26× (same process) and 17× (separate process) at a 2,049-token prefix; 12× / 4.6× / 1.4× at 1,035 / 268 / 34 tokens |
| [`file_cache_ttft_llama_2049_2026_10_07.json`](file_cache_ttft_llama_2049_2026_10_07.json) | [`../file_cache_ttft.py`](../file_cache_ttft.py) `--workload llama --pairs 3` | M4 Mac mini, mlx-lm 0.31.3 | mlx-lm's own prompt-cache file at the same 2,049-token prefix and question: 37.0 ms from a fresh process (67.1 MB file), first token equal to the cold path's |
| [`file_cache_ttft_gemma_64000_2026_10_07.json`](file_cache_ttft_gemma_64000_2026_10_07.json) | [`../file_cache_ttft.py`](../file_cache_ttft.py) `--workload gemma --pairs 3` | M4 Mac mini, mlx-lm 0.31.3 | the 64K Gemma-4-E2B prefix through a 405.1 MB file (sliding-window layers included), dense: 82.6 ms median of three fresh processes (the first read took 137.7 ms) |
| [`file_cache_ttft_ssm_2048_2026_10_07.json`](file_cache_ttft_ssm_2048_2026_10_07.json) | [`../file_cache_ttft.py`](../file_cache_ttft.py) `--workload ssm --pairs 3` | M4 Mac mini, mlx-lm 0.31.3 | a Mamba model's recurrent state (`ArraysCache`) round-trips through the file: first token equal to the cold path's |
| [`agent_session_ab_2026_10_07/`](agent_session_ab_2026_10_07/) | [`../agent_session_ab.py`](../agent_session_ab.py) `--n 20 --restart-after 10`, one file per server and model, written with `--public` (reply text replaced by its SHA-256) | M4 Mac mini 16 GB; versions in `environment.json` | the serve A/B tables in `doc/coding_agents.md`; `*_nocache.json` are the cold references for the reply comparison |
| [`cross_process_ttft_2026_10_02.json`](cross_process_ttft_2026_10_02.json) | the same, before 2026-10-07 | M4 Mac mini, Pion 0.9.1 | Superseded: its prefix had no `<bos>`. Read 20× / 17×; a same-day rerun without the fix (`../../results/2026-10-07-mac-m4/before_fix/`) lands within run-to-run variation of the fixed one (17.4× and 29.3×), so the difference is the day, not the prompt |
| [`stage1_qwen3_5_prefix_sweep_2026_10_02.json`](stage1_qwen3_5_prefix_sweep_2026_10_02.json) | [`../sweep_qwen3_5_warm_ttft.py`](../sweep_qwen3_5_warm_ttft.py) | M4 Mac mini, Pion 0.9.1 | Qwen3.5-4B warm TTFT 14.0× / 25.3× / 29.0× at 2K / 4K / 8K |
| [`stage1_hybrid_results_2026_10_07.json`](stage1_hybrid_results_2026_10_07.json) | [`../stage1_hybrid_recall_bench.py`](../stage1_hybrid_recall_bench.py) `--n 100 --with-pion` | M4 Mac mini, Pion 0.9.6 | `HybridRetrievalCache` 3.5× (inproc) and 3.2× (pion) p50 TTFT, 96.1% token agreement, answer found 0.73 against text-RAG's 0.72, 100 SQuAD v2 queries |
| [`stage1_hybrid_results_2026_10_02.json`](stage1_hybrid_results_2026_10_02.json) | the same, before 2026-10-07 | M4 Mac mini, Pion 0.9.1 | Superseded: each question began with a second `<bos>`. Read 3.0× / 2.7×, 98.3% agreement and 0.68 answers on every path |
