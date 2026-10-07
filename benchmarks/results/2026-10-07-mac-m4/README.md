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
