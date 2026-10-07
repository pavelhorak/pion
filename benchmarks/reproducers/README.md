# Reproducers

The scripts behind Pion's headline numbers, so you can check them rather than
take them on trust.

**Read the prerequisites before running.** Most of the cost is a model
download, and most of these are timing measurements — a run on a loaded machine
produces a *wrong number*, not an error. Close everything else first.

| Script | Backs | Needs |
|---|---|---|
| [`cross_process_ttft.py`](cross_process_ttft.py) | Llama-3.2-1B time to first token from a **separate process**: **17×** at a 2,049-token prefix (1,193 → 69.0 ms), the prefix-length regime 1.4× → 17×, and with `--same` the same-process row (46.2 ms, **26×**) | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit, a Pion server with `--kvcache --metal-attention` (`--same` needs the attention engine) |
| [`file_cache_ttft.py`](file_cache_ttft.py) | the **baseline** for every TTFT row above: mlx-lm's own prompt-cache file (`save_prompt_cache` / `load_prompt_cache`) read by a fresh process and timed like a Pion hit — 37.0 ms at the same 2,049-token prefix (`--workload llama`), 82.6 ms at the 64K Gemma prefix with dense attention (`--workload gemma`), and a recurrent-state round trip (`--workload ssm`) | Apple Silicon, MLX, the workload's model (Llama-3.2-1B-Instruct-4bit, gemma-4-e2b-it-4bit or mamba-130m-hf-f32); no Pion server |
| [`agent_session_ab.py`](agent_session_ab.py) | `pion-vllm-mlx serve` against stock mlx-lm, Ollama, LM Studio and oMLX on one recorded Claude Code session: tokens reused within the session, after a server restart and in a second session, and time to first token (the tables in [`doc/coding_agents.md`](../../doc/coding_agents.md)). Also records your own session (`--record`) and writes copies without reply text (`--public`) | Apple Silicon, MLX, the model, each server installed, a recorded session; a Pion server for `--server pion` |
| [`sweep_qwen3_5_warm_ttft.py`](sweep_qwen3_5_warm_ttft.py) | Qwen3.5-4B hybrid warm TTFT: **14.0× / 25.3× / 29.0×** at 2K / 4K / 8K prefix (`--target-tokens 2048` runs one length) | Apple Silicon, MLX, Qwen3.5-4B-MLX-4bit, a Pion server with `--kvcache` |
| [`stage1_hybrid_recall_bench.py`](stage1_hybrid_recall_bench.py) | hybrid retrieval cache recall | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit, a Pion server |
| [`stage0_hybrid_kv_injection.py`](stage0_hybrid_kv_injection.py) | the chunk-keyed cache's first evidence | Apple Silicon, MLX, Llama-3.2-1B-Instruct-4bit |

## If a number does not reproduce

Open an issue with your hardware, the model build and the raw output. A number
that only reproduces on our machine is a number we want to know about —
several of these were wrong the first time.
