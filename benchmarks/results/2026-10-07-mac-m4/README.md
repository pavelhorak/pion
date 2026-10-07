# 2026-10-07 — the 64K sparse-mask NIAH, after its prompt fix

Raw output of [`examples/sparse_mask_64k_niah.py`](../../../examples/sparse_mask_64k_niah.py),
unedited, on the machine that recorded the 2026-10-06 miss
([`../2026-10-06-mac-m4/sparse_mask_64k_niah_full.txt`](../2026-10-06-mac-m4/sparse_mask_64k_niah_full.txt)).

- Machine: Apple M4, 16 GB, macOS 26.6.2 ([`machine.txt`](machine.txt)).
- Python: mlx-lm 0.31.3, mlx 0.31.2 ([`python_versions.txt`](python_versions.txt)).
- Model: `mlx-community/gemma-4-e2b-it-4bit`, 35 layers (7 full attention, 28
  sliding with a 512-token window).

**What changed.** The example built its prompt from separate `tok.encode()`
calls, and the loader turns on `add_bos_token` for Gemma 4, so every call began
with `<bos>`: the 64K prompt held 397 of them, one every 162 tokens of filler,
one before the needle and one before the question. Neither vanilla mlx-lm nor
Pion found the needle. The prompt now has one `<bos>`, at position 0.

Each run started a fresh `pion-server --kvcache --metal-attention -w 1 --no-wal`
from an empty working directory and ran the example from the repository root
with no arguments (64,000 tokens, needle at depth 0.5, seed 0).

| File | Server | Vanilla cold TTFT | Pion warm, steady state | Needle |
|---|---|---:|---:|---|
| `sparse_mask_64k_niah.txt` | `pion-server 0.9.6+d724a08`, the v0.9.6 release tarball ([`pion_version.txt`](pion_version.txt)) | 54,378.6 ms | 124.4 ms | found by both |
| `sparse_mask_64k_niah_prerelease.txt` | a release build of `main` at a8c88da, whose engine source is v0.9.6's | 54,407.4 ms | 121.5 ms | found by both |

Vanilla is a cold prefill of all 63,984 tokens. Pion's number is a warm call:
the 63,961-token prefix was prefilled into the cache first (54.8 s, not
counted), and the timed call runs the 23-token question with a sparse mask on
the full-attention layers, 512 prefix tokens per layer (0.80%). The sliding
layers stay dense. This is one needle at one depth, not a measure of
long-context quality in general.

## The harnesses, after the same fix

Six more harnesses assembled prompts from separately encoded pieces. Llama
3's tokenizer prepends `<|begin_of_text|>` on every `encode()` too, so the
Llama workloads carried a second `<bos>` before every question, and the Gemma
ones carried one at every seam. They now build prompts through
`tests/_prompt_ids.py`, and `tests/test_prompt_bos.py` checks each builder.
Every file here is from the fixed harness; [`before_fix/`](before_fix/) holds
same-day runs of the unfixed harnesses (main at fb943a4) for comparison.

Server: the v0.9.6 release tarball, fresh per run, `--no-wal`, with the flags
shown. The timing runs waited on `benchmarks/preflight.py` first.

| File | Server flags | Command (from the repository root) | Result |
|---|---|---|---|
| `kv_prefix_workload_q10.txt`, `_repeat.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_workload.py` | Stage 1: 3.61× and 3.56× mean TTFT, 90% hits, first-token agreement 49/50 in both runs (correctness verdict FAIL, see below) |
| `w1_stage2.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/bench_w1_stage2.py` | in-process lane: p50 34.4 ms, mean 42.2 ms, 4.74×, 50/50 |
| `w1_stage2_binary_lane.txt` | same | `PION_PROMPT_CACHE_NO_INPROC=1 python3 tests/bench_w1_stage2.py` | binary lane: p50 98.2 ms, 0.63 ms a call, 2.00×, 50/50 |
| `w1_stage2_resp_lane.txt`, `_repeat.txt` | same | `PION_PROMPT_CACHE_NO_INPROC=1 PION_PROMPT_CACHE_NO_BINARY=1 python3 tests/bench_w1_stage2.py` | RESP lane: p50 110.3 and 128.1 ms, 0.65 and 0.72 ms a call, 50/50 |
| `hybrid_per_layer_agreement.txt` | `-w 1` | `python3 pion-vllm-mlx/tests/test_hybrid_per_layer_agreement.py` | 16/16 tokens of real text agree (the unfixed harness: 5/16; see the next section for the prompt) |
| `long_context_niah.txt` | `--kvcache --metal-attention -w 1` | `python3 tests/test_long_context_niah.py` | 4K/8K/16K: vanilla and Pion 100% |
| `ruler_vt_4k.txt` | same | `python3 tests/test_ruler_subset.py --include-vt --lengths 4096` | multi-value F1 1.000 (unfixed: 0.800); variable tracking 3/3 on every path |
| `ruler_subset.txt` | same | `python3 tests/test_ruler_subset.py` | multi-value at 32K/64K: F1 1.000 on every path |
| `long_context_niah_multi.txt` | same | `python3 tests/test_long_context_niah_multi.py` | 3 needles at 32K/64K: vanilla, dense and sparse 100% |

**Stage 1's one disagreement is a near-tie.** In both runs the same request
differs: prompt 0, "how should I version a binary …". Vanilla mlx-lm's top two
first tokens are 0.016 logits apart, and the fp16-stored cache picks the other
one. The test requires 99% agreement, so 49 of 50 fails it. The unfixed
harness reported 50/50 on the same day; its stray `<bos>` changed the prompt.

The TTFT columns of the long-context tests are not comparable with the
example's: their vanilla side prefills unchunked by default, which swaps on a
16 GB machine.

## The rest of the class: reproducers, BLEU, hybrid retrieval

`tools/audit_prompt_bos.py` (gate and CI: `tests/test_audit_prompt_bos.py`)
reads every tracked .py and reports a plain `encode()` piece placed after
another piece. It found 47 sites in 21 files on main before the first round (#63) and
finds 0 after. The second round fixed the harnesses below, and the hybrid
retrieval docstring, demo and tests. The Qwen3.5 and Qwen2.5 tokenizers add no
`<bos>`, so the Qwen3.5 prefix sweep and the two Qwen hybrid tests were never
affected; their spelling changed, not their tokens. `tests/bench_ttft.py`'s ids
are identical before and after.

The second round ran while another project's self-hosted CI runner was busy on
this machine (load about 2–3, 0.7–1.2 GB free disk; every server ran with
`--no-wal`). Only correctness ran then; the timing runs below waited for the
runner to finish.

| File | Server flags | Command | Result |
|---|---|---|---|
| `kv_prefix_bleu.txt` | `--kvcache -w 1` | `python3 tests/test_kv_prefix_bleu.py` | fp16: mean BLEU 0.979, first token 20/20, bit-identical on 19/20 |
| `kv_prefix_bleu_int8.txt`, `_repeat.txt` | same | `… --vquant int8` | 0.499 in both runs, first token 19/20 |
| `kv_prefix_bleu_turbo4.txt`, `_repeat.txt` | same | `… --vquant turbo4` | 0.378 in both runs, first token 18/20 |
| `kv_prefix_cross_instance.txt` | same | `python3 tests/test_kv_prefix_cross_instance.py` | BLEU 1.0000, the same 50 tokens |
| `hybrid_retrieval_demo.txt`, `_pion.txt` | `--kvcache --metal-attention -w 1` | `python3 examples/hybrid_retrieval_demo.py [--backend pion]` | 12/12 tokens agree, both answer 330 m |
| `test_hybrid_retrieval.txt` | same | `python3 pion-vllm-mlx/tests/test_hybrid_retrieval.py` | 3/3 agree 1.000 |
| `wire_sparse_consumer.txt` | same | `python3 tests/test_wire_sparse_consumer.py` | vanilla and wire-sparse find the needle |
| `boundary_protect.txt` | `--kvcache -w 1` | `python3 pion-vllm-mlx/tests/test_boundary_protect.py` | first tokens match |
| `chunked_prefill_correctness.txt` | `-w 1` | `python3 tests/test_chunked_prefill_correctness.py` | 16/16 tokens of real text agree |

**The quantized BLEU figures were inflated by the stray `<bos>`.** The unfixed
harness reproduces the published 0.9688 / 0.6774 / 0.5376 exactly
([`before_fix/`](before_fix/)), and the fixed one repeats 0.4989 / 0.3779
exactly. A `<bos>` before each question drew attention away from the stored
prefix and hid part of the quantization error; fp16 is unaffected (0.969 to
0.979).

**Two agreement tests compared noise after `<eos>`.** With one `<bos>`,
Gemma 4 (an instruct model) answers bare filler by ending its turn, so
`test_chunked_prefill_correctness` and `test_hybrid_per_layer_agreement`
compared tokens decoded after `<eos>` (the former failed at 13/16). Unfixed,
the stray `<bos>` made the model continue the filler, and they passed on that.
Both now build a one-turn chat request (`tests/_prompt_ids.py:chat_prompt`,
thinking off) and compare 16 tokens of real text.

### The timing reproducers

These waited until the other project's runner was idle and the 1-minute load
average had stayed under 4 for a minute. Free disk was 0.6 GB, under
`benchmarks/preflight.py`'s 2 GB floor; that floor guards the WAL, and every
server ran with `--no-wal`. `--no-wal` does not stop the V-store WAL
(`pion.vstore.wal.*`), though, and the hybrid run's 100 stored chunks filled the
disk twice, so each server's working directory was on a RAM disk. The WAL is
written at ingest, outside every timed span.

| File | Server flags | Command | Result |
|---|---|---|---|
| `ttft_r11_file_256.txt`, `ttft_r11_file_1024.txt`, `ttft_r11_file_2048.txt` | `--kvcache --metal-attention -w 1`, started by the harness with its WAL on (16 GB free, not the RAM disk above), late on 2026-10-07 | `PION_BIN=./pion-server python3 tests/bench_ttft.py --start --prompt-tokens N --runs 11` | five paths with a one-token suffix, path E being mlx-lm's own prompt-cache file; at 2,048 tokens: cold 1,176.9 ms, Stage 1 40.5 ms, wire lane 24.3 ms, in-process 11.2 ms, file 14.6 ms |
| `cross_process_ttft.txt` ([JSON](../../reproducers/results/cross_process_ttft_2026_10_07.json)) | `--kvcache --metal-attention -w 1` | `python3 benchmarks/reproducers/cross_process_ttft.py --prefix-tokens 34 268 1035 2049 --pairs 5 --same` | 2,049 tokens: cold 1,193 ms, separate process 69.0 ms (17.3×), same process 46.2 ms (25.8×); 12.35× / 4.62× / 1.44× at 1,035 / 268 / 34 |
| `stage1_hybrid.txt` ([JSON](../../reproducers/results/stage1_hybrid_results_2026_10_07.json)) | same | `python3 benchmarks/reproducers/stage1_hybrid_recall_bench.py --n 100 --with-pion` | p50 TTFT 113.8 / 32.5 / 35.8 ms (text-RAG / inproc / pion), 3.5× and 3.2×; answer found 0.72 / 0.73 / 0.73; token agreement 96.1% |
| `prompt_cache_workload_q30_r8.txt` | `--kvcache -w 1` | `python3 pion-vllm-mlx/tests/test_prompt_cache_workload.py --vquant fp16 --prompts 5 --queries 30 --prompt-repeats 8` | 8.92× mean TTFT, 96.7% hits, 50/50 first tokens |
| `stage0_hybrid.txt`, `stage0_hybrid_results.json` | `-w 1` | `python3 benchmarks/reproducers/stage0_hybrid_kv_injection.py` | token agreement 1.0, TTFT saving 71.8% mean |

[`before_fix/`](before_fix/) holds the same three timing runs from main's
unfixed harnesses, the same afternoon: 17.4× / 29.3× at 2,049 tokens, 3.55× /
3.24× hybrid with 0.68 answers and 99.3% agreement, and 8.98×. **The speedups do
not move with the fix**; against the published 2026-10-02 figures (17× / 20×,
3.0× / 2.7×) the difference is the day, and the same-process row varies by
several milliseconds between runs. What the fix changes is quality: with the
stray `<bos>` before each question, the hybrid run answered 0.68 on every path
and its paths agreed 99.3%; with one `<bos>`, 0.72–0.73 and 96.1%.
