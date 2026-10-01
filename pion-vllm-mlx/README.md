# pion-vllm-mlx

The Pion prompt cache for mlx-lm on Apple Silicon: a prefix's K/V that a
different process, a different model object, or a restarted server can reuse
(Stage 1), plus an attention patch that lets Pion compute the attention over
that prefix itself (Stage 2).

Time to first token on Llama-3.2-1B-4bit with a 2,048-token prefix: 1,530 ms
cold in vanilla mlx-lm, **30.2 ms** warm in the same process (50.6×), and
**64.7 ms** from a separate process over the wire (24×, against 1,558 ms). With
a prefix under about a thousand tokens there is little to save; the
[Pion README](https://github.com/pavelhorak/pion#readme) has the sweep.

Stage 2 runs on three lanes, auto-selected by `PionPromptCache`:

| Lane | Where it runs | TTFT p50, short system-prompt prefixes (Llama-3.2-1B-4bit, 100 requests) |
|---|---|---|
| 1. **In-process** (default, same-process consumer) | MLX in the calling process; zero wire roundtrips | **28.2 ms** (6.51× vs vanilla cold) |
| 2. **Binary fast lane** (port+1, `0xCA5E` frames) | Cross-process via `sendmsg` scatter-gather + single RTT | 92.1 ms |
| 3. **RESP fallback** | Plain RESP, for older Pion servers without the binary listener | 108.6 ms |

Plus `HybridRetrievalCache` for RAG: chunk-id-keyed K/V hydration (4.5× p50 TTFT and 99.3% token agreement on a 100-query SQuAD v2 run, [`stage1_hybrid_recall_bench.py`](https://github.com/pavelhorak/pion/blob/main/benchmarks/reproducers/stage1_hybrid_recall_bench.py)).

## Install

```bash
pip install 'pion-vllm-mlx[mlx]'
```

It talks to a Pion server started with the prompt cache enabled:

```bash
brew install pavelhorak/tap/pion && brew services start pion   # macOS; the service runs with these flags
./pion-server --kvcache --metal-attention                       # or from a release tarball / source build
```

The `mlx` extra pins **`mlx-lm>=0.20.1,<0.32`**. That ceiling is not decoration:
`install_pion_attention_patch()` replaces a *private* mlx-lm function and rebinds
the snapshot-bound name inside every imported `mlx_lm.models.*` module. The
signature is re-checked at patch time, so an incompatible mlx-lm raises
`PionMlxCompatError` naming the observed signature and patches nothing, rather
than failing as a `TypeError` inside your generation loop. To see the seam:

```bash
python -c "import pion_vllm_mlx as p; print(p.mlx_lm_seam_report())"
```

From a checkout of the [Pion repository](https://github.com/pavelhorak/pion),
`pip install -e 'pion-vllm-mlx/[dev,mlx]'` adds the dev tools, then
`python pion-vllm-mlx/tests/test_mlx_lm_seam.py` checks one mlx-lm version and
`pion-vllm-mlx/tests/run_mlx_version_matrix.sh` the whole matrix.

The MLX dependency is optional because the package can act as a wire-compatibility shim on machines that don't have MLX (e.g. a Linux test runner). Lanes 2 and 3 work without MLX in the consumer; lane 1 requires it.

Lanes 2 and 3 need the server started with `--kvcache --metal-attention`, as above.

## Usage

### The four lines (Stage 1 — cross-process, native decode)

```python
from mlx_lm import load, generate
from pion_vllm_mlx import PionPromptCache

model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
prefix_ids = tok.encode(SYSTEM_PROMPT, add_special_tokens=False)   # the part every request shares
suffix_ids = tok.encode(USER_TURN, add_special_tokens=False)       # the part that changes

pc = PionPromptCache(model, host="127.0.0.1", port=1974)                       # once per loaded model
cache = pc.get_or_prefill(prefix_ids, namespace="app|v1|llama-1b|system_v1")  # MISS: prefill once + store · HIT: fetch
text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)             # decode as usual, at native speed
```

`get_or_prefill` returns what `mlx_lm.make_prompt_cache(model)` would, with the
prefix's K/V already in it, so mlx-lm decodes at native speed. The namespace
names one exact token sequence for one model and quantization — key it on the
tokens, never the text, and change it when either changes. The first process to
ask pays the prefill once; every later call, from any process, fetches it.
Runnable with timings: [`examples/prompt_cache_demo.py`](https://github.com/pavelhorak/pion/blob/main/examples/prompt_cache_demo.py).

### Stage 2 — Pion computes the attention over the prefix

```python
from mlx_lm import load, generate
from pion_vllm_mlx import PionPromptCache, install_pion_attention_patch, make_pion_prompt_cache

install_pion_attention_patch()                                          # once per process
model, tok = load("mlx-community/gemma-4-e2b-it-4bit")
pc = PionPromptCache(model, stage2=True, host="127.0.0.1", port=1974)
ns = "rag/corpus_v1"

if not pc.lookup(ns):                       # cold: prefill once, push per-layer K/V to Pion
    pc.get_or_prefill(prefix_ids, namespace=ns)

# warm: no prefill anywhere; attention over the prefix runs in Pion
cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc, prefix_len=len(prefix_ids))
text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)
```

When the cold prefill ran in this same process the prefix stays resident as MLX
arrays (lane 1, zero wire round trips — the 30.2 ms / 50.6× row in the main
README). From any other process the patch uses lane 2 or 3, and then **every
decode step pays one round trip per layer**: on a 1B model that roughly triples
the per-token decode cost. So for a consumer that only generates text from
another process, the four Stage-1 lines above are the faster end-to-end path;
Stage 2 is for consumers that want the attention itself in Pion — the sparse
long-context selectors (`sparse_full_layers=`), `pion-exo`, a custom vLLM
CacheEngine. `stage2=True` without `make_pion_prompt_cache` changes nothing
except a slightly slower cold path.

Hybrid models supported (per-layer routing): **Llama-3.2-1B-Instruct-4bit**, **Qwen3.5-4B**, **Gemma-4-E2B-it-4bit**, **Gemma-4-12B-it-4bit**. Add a new hybrid arch by exposing `model.args.layer_types` + `sliding_window` — the patch reads both directly.

## Docs

- Main project: [github.com/pavelhorak/pion](https://github.com/pavelhorak/pion) · website: [pion.pavelhorak.com](https://pion.pavelhorak.com/)
- Shared KV cache design: [`doc/shared_kv_cache.md`](https://github.com/pavelhorak/pion/blob/main/doc/shared_kv_cache.md)
- 64K NIAH reproducer (Gemma 4): [`examples/sparse_mask_64k_niah.py`](https://github.com/pavelhorak/pion/blob/main/examples/sparse_mask_64k_niah.py)

## License

Apache-2.0 — see [`LICENSE`](https://github.com/pavelhorak/pion/blob/main/pion-vllm-mlx/LICENSE). Pion's client
packages are deliberately permissive so they can be vendored into any stack;
the Pion **server** this package talks to is Apache-2.0 too, with one closed
binary library for its tuned vector kernels. The full map of what is under
which licence is [`doc/licensing.md`](https://github.com/pavelhorak/pion/blob/main/doc/licensing.md).
