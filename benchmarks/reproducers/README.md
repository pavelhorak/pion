# Reproducers

The scripts behind Pion's headline numbers, so you can check them rather than
take them on trust.

**Read the prerequisites before running.** Most of the cost is a model
download, and five of these are timing measurements — a run on a loaded machine
produces a *wrong number*, not an error. Close everything else first.

| Script | Backs | Needs |
|---|---|---|
| [`cross_process_ttft.py`](cross_process_ttft.py) | Llama-3.2-1B time to first token from a **separate process**: **24×** at a 2,049-token prefix (1,558 → 64.7 ms), and the prefix-length regime 1.6× → 24× | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit, a Pion server with `--kvcache` |
| [`sweep_qwen3_5_warm_ttft.py`](sweep_qwen3_5_warm_ttft.py) | Qwen3.5-4B hybrid warm TTFT: **24.77× / 36.0× / 29.5×** at 2K / 4K / 8K prefix | Apple Silicon, MLX, Qwen3.5-4B-MLX-4bit, a Pion server with `--kvcache` |
| [`bench_qwen3_5_warm_ttft.py`](bench_qwen3_5_warm_ttft.py) | single-point version of the above | same |
| [`stage1_hybrid_recall_bench.py`](stage1_hybrid_recall_bench.py) | hybrid retrieval cache recall | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit, a Pion server |
| [`stage0_hybrid_kv_injection.py`](stage0_hybrid_kv_injection.py) | the chunk-keyed cache's first evidence | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit |

## If a number does not reproduce

Open an issue with your hardware, the model build and the raw output. A number
that only reproduces on our machine is a number we want to know about —
several of these were wrong the first time.
