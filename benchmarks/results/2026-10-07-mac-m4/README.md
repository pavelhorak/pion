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
| `hybrid_per_layer_agreement.txt` | `-w 1` | `python3 pion-vllm-mlx/tests/test_hybrid_per_layer_agreement.py` | 16/16 tokens agree (the unfixed harness: 5/16) |
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
