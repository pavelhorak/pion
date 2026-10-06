# 2026-10-06 — M4 Mac mini re-measurements

Raw output of the public harnesses behind the numbers the docs quote for
Apple Silicon. Every file is the harness's own stdout, unedited.

- Machine: Apple M4, 16 GB, macOS 26.6.2 ([`machine.txt`](machine.txt)).
- Server: `pion-server 0.9.5+403f689`, the release build with the closed vector
  library ([`pion_version.txt`](pion_version.txt)); the engine source is the same
  as `main` at the time of this commit.
- Python: mlx-lm 0.31.3, mlx 0.31.2 ([`python_versions.txt`](python_versions.txt)).
- Model: `mlx-community/Llama-3.2-1B-Instruct-4bit` unless a row says otherwise.

Each harness ran against a fresh `pion-server` started with the flags shown,
from an empty working directory, and the server was stopped afterwards. Timing
runs are only as good as the machine was quiet; the vector runs waited on
`benchmarks/preflight.py` before each run.

| File | Server flags | Command (from the repo root) | Backs |
|---|---|---|---|
| `kv_prefix_workload_q10.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_workload.py` | README "mixed workload", Stage 1 |
| `kv_prefix_workload_q30.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_workload.py --queries 30` | Stage 1 at 30 queries a prompt |
| `w1_stage2.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/bench_w1_stage2.py` | in-process lane |
| `w1_stage2_binary_lane.txt` | `--kvcache --metal-attention -w 1` | `PION_PROMPT_CACHE_NO_INPROC=1 python3 tests/bench_w1_stage2.py` | binary lane |
| `w1_stage2_resp_lane.txt` | `--kvcache --metal-attention -w 1` | `PION_PROMPT_CACHE_NO_INPROC=1 PION_PROMPT_CACHE_NO_BINARY=1 python3 tests/bench_w1_stage2.py` | RESP lane |
| `ttft_256.txt`, `ttft_1024.txt`, `ttft_2048.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/bench_ttft.py --prompt-tokens N` | TTFT paths A–D, 5 runs |
| `ttft_r11_256.txt`, `ttft_r11_1024.txt`, `ttft_r11_2048.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/bench_ttft.py --prompt-tokens N --runs 11` | the TTFT table in `doc/shared_kv_cache.md` (median of 10 warm runs) |
| `kv_prefix_workload_q30_r8.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_workload.py --queries 30 --prompt-repeats 8` | Stage-1 workload table |
| `prompt_cache_workload_q30_r8.txt` | `--kvcache -w 1` | `python3 pion-vllm-mlx/tests/test_prompt_cache_workload.py --vquant fp16 --prompts 5 --queries 30 --prompt-repeats 8` | the same workload through the public `PionPromptCache` API |
| `attend_prefix.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/test_attend_prefix.py` | `ATTEND.PREFIX.STORE`/`QUERY` at H=8 N=2048 D=64 |
| `kv_prefix_blocks.txt` | (the test starts its own server) | `python3 tests/test_kv_prefix_blocks.py` | `KV.PREFIX.MEMBERSHIP` latency |
| `kv_prefix_bleu_int8.txt`, `kv_prefix_bleu_turbo4.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_bleu.py --vquant Q` | BLEU per quantized tier (the test's 0.95 gate fails for both, as the doc says) |
| `mlx_lm_patch.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/test_mlx_lm_patch.py` | the patched path's tokens against vanilla mlx-lm |
| `metal_attention.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/bench_pion_metal_attention.py` | native Metal SDPA vs MLX |
| `neuron_pkm_bench.txt` | `--kvcache -w 1` | `python3 tests/test_neuron_pkm.py --bench` | product-key memory vs kNN-LM HNSW |
| `attend_128k.txt` | `--kvcache -w 1` | `python3 tests/test_attend_128k.py` | `ATTEND.*` at 128K tokens over RESP |
| `failover.txt` | `--cluster --cluster-host 127.0.0.1 -w 1` | `python3 tests/test_failover.py` | forced failover time |
| `profile_rss.txt` | `--profile P --no-auto-embed --no-auto-detect -w 1` | `ps -o rss=` 20 s after PING answers | idle resident memory per profile |
| `kv_prefix_bleu.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_bleu.py` | fp16 BLEU over 20 questions |
| `kv_prefix_cross_instance.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_cross_instance.py` | a second client, same output |
| `sparse_mask_64k_niah.txt` | `--kvcache --metal-attention -w 1` | `python3 examples/sparse_mask_64k_niah.py --skip-vanilla` | 64K sparse-mask NIAH (Gemma-4-E2B-it-4bit), Pion only: **the needle was not found** |
| `sparse_mask_64k_niah_full.txt` | `--kvcache --metal-attention -w 1` | `python3 examples/sparse_mask_64k_niah.py` | the same with the vanilla baseline: **neither path found the needle**, so the reproducer is broken here, not just Pion's path |
| `vec/vec_<mode>_<n>_<arm>.txt` | started by the harness (`-w 10 --independent-workers`, plus the mode's flag) | `python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10 [--polarquant\|--turboquant\|--nanoquant]`, with `pion-server` swapped between the `pixi run build` binary (`closed`) and the `pixi run build-open` one (`open`) | the closed library against the open build in four quant modes, arms in the order closed, open, open, closed, closed, open |

The vector logs had the local checkout path in them; it is replaced with `<repo>`
(and the home directory with `~`), nothing else is changed. `vec/vec_version_*.txt`
is each binary's `--version`.
