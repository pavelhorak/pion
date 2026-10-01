#!/usr/bin/env python3
"""
moe_expert_substrate_demo.py — runnable demonstration of Pion's
MoE-expert tier substrate on Apple Silicon + MLX (Phase 0 work, gh #61).

What this shows:
  An MoE model whose disk footprint exceeds local unified memory runs to
  coherent generation with bounded resident memory — instead of OOMing /
  GPU-command-buffer-timing-out the way vanilla mlx-lm does.

  Concretely on a 16 GB Mac mini M4:
    - Gemma-4-26B-A4B 8-bit on disk = 28 GB. Vanilla mlx-lm → OOM.
      Substrate → 0.15 tok/s, peak resident 3.9 GB, coherent reasoning.
    - Gemma-4-26B-A4B bf16 (lossless) = 52 GB. Vanilla → OOM. Substrate
      → ~0.07 tok/s, peak resident ~5 GB.

  The substrate primitive:
    1. Parse safetensors headers at startup → precomputed per-expert
       byte offsets.
    2. Per-expert fetch via os.pread (NOT mmap; mmap fails GPU command-
       buffer wall-clock deadlines on memory-constrained systems).
    3. ThreadPoolExecutor for parallel pread (releases GIL during syscall).
    4. mx.array construction on main MLX thread (MLX not thread-safe).
    5. Per-op mx.eval forces RAM residence BEFORE GPU dispatch.
    6. LRU cache with optional PIN for hot-set protection.

  The same os.pread + main-thread mx.array + async prefetch primitive
  validates across three MoE architectures (Phi-3.5-MoE, Gemma 4 26B-A4B,
  OLMoE). The Pion server's MOE.EXPERT.{STORE,FETCH,PREFETCH,PIN,UNPIN,
  INFO,STATS} wire surface (design in
  `WIRE_FORMAT_DESIGN.md`, private research tree) makes the same
  primitive available cross-process via 0xCA5E binary framing.

Hardware:
  - Apple Silicon Mac with MLX installed.
  - Disk space ≥ model size (28 GB for 8-bit, 52 GB for bf16 demo).
  - 16 GB unified memory is the test envelope; 8/24/32+ GB Macs work
    even better (more headroom).

Usage:
  # Smallest demo (Phi-3.5-MoE-4bit, ~23 GB) — first time downloads ~25 min:
  python3 examples/moe_expert_substrate_demo.py --model phi35-moe-4bit

  # 8-bit Gemma 4 demo (~28 GB):
  python3 examples/moe_expert_substrate_demo.py --model gemma4-26b-8bit

  # bf16 lossless Gemma 4 demo (~52 GB):
  python3 examples/moe_expert_substrate_demo.py --model gemma4-26b-bf16

Run-time: model download (~25-90 min one-time per model) + ~30-60 s per
generation. Substrate proves operability; tok/s in 0.07-0.16 range is
expected on a memory-constrained system.

Measured on OLMoE, Phi-3.5, Gemma 4 8-bit and Gemma 4 bf16.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The substrate module ships beside this demo (examples/pion_moe_tier.py).
# It used to be imported from a research directory that does not ship, so this
# demo raised ImportError for every reader outside the
# development repo.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


MODEL_PRESETS = {
    'phi35-moe-4bit': {
        'repo_id': 'mlx-community/Phi-3.5-MoE-instruct-4bit',
        'disk_gb': 23.6,
        'description': '41.9B total / 6.6B active, top-2 of 16 experts, ~23 GB on disk',
    },
    'gemma4-26b-8bit': {
        'repo_id': 'lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit',
        'disk_gb': 28.0,
        'description': '26B total / 4B active, top-8 of 128 + shared MLP, ~28 GB at 8-bit',
        'local_path_hint': Path.home() / '.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit',
    },
    'gemma4-26b-bf16': {
        'repo_id': 'mlx-community/gemma-4-26b-a4b-it-bf16',
        'disk_gb': 51.6,
        'description': '26B / 4B at bf16 LOSSLESS, ~52 GB on disk',
    },
}


def resolve_model_path(name: str) -> Path:
    """Locate the model in HF cache or LM Studio's model dir.

    Returns the snapshot/model directory; raises FileNotFoundError otherwise.
    """
    preset = MODEL_PRESETS[name]
    candidates = []
    # LM Studio explicit hint
    if 'local_path_hint' in preset and preset['local_path_hint'].exists():
        candidates.append(preset['local_path_hint'])
    # HF cache snapshot
    cache_name = 'models--' + preset['repo_id'].replace('/', '--')
    snap_dir = Path.home() / '.cache/huggingface/hub' / cache_name / 'snapshots'
    if snap_dir.exists():
        for s in snap_dir.iterdir():
            if (s / 'config.json').exists():
                candidates.append(s)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f'{name}: not found. Download via:\n'
        f'  ~/.lmstudio/bin/lms get "{preset["repo_id"]}" -y\n'
        f'  OR python3 -c "from huggingface_hub import snapshot_download; '
        f'snapshot_download(\'{preset["repo_id"]}\')"\n'
        f'  ETA: ~{int(preset["disk_gb"] * 60 / 10)} min at HF unauthed ~10 MB/s.'
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', choices=list(MODEL_PRESETS), default='gemma4-26b-8bit',
                    help='Model preset (default: gemma4-26b-8bit)')
    ap.add_argument('--prompt', default='What is the capital of France? Answer in one word.',
                    help='Generation prompt (chat-templated for instruction-tuned models)')
    ap.add_argument('--max-tokens', type=int, default=12)
    ap.add_argument('--cache-mib', type=int, default=1024,
                    help='LRU cache budget. Sweet spot ~1 GB on 16 GB Mac; '
                         'larger caches starve Metal working pool.')
    ap.add_argument('--workers', type=int, default=4,
                    help='ThreadPoolExecutor workers for parallel pread.')
    args = ap.parse_args()

    print(f'\n== MoE-expert substrate demo: {args.model} ==')
    preset = MODEL_PRESETS[args.model]
    print(f'  {preset["description"]}')
    try:
        model_dir = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f'\nERROR: {e}')
        return 1

    print(f'  resolved: {model_dir}')

    # Lazy imports — module-level imports kept light for --help speed
    import mlx.core as mx
    from mlx_lm import load
    from pion_moe_tier import MoEExpertTier, install_moe_substrate

    print('\nloading mlx-lm shell (lazy)...')
    t0 = time.time()
    model, tokenizer = load(str(model_dir), lazy=True)
    print(f'  loaded in {time.time()-t0:.1f}s')

    tier = MoEExpertTier(model_dir, cache_mib=args.cache_mib, workers=args.workers)
    info = tier.info()
    print(f'\nsubstrate tier:')
    print(f'  layout: {info["layout"]} ({"quantized" if info["quantized"] else "lossless"})')
    print(f'  layers={info["num_layers"]}  experts={info["num_experts"]}  top_k={info["top_k"]}')
    print(f'  cache budget: {args.cache_mib} MiB,  pread workers: {args.workers}')

    n_moe = install_moe_substrate(model, tier)
    print(f'  installed intercept on {n_moe} MoE layer(s)')

    # Chat-format the prompt for IT models
    try:
        messages = [{'role': 'user', 'content': args.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = args.prompt

    ids = tokenizer.encode(formatted)
    x = mx.array(ids)[None]
    print(f'\nprompt: {args.prompt!r}  ({len(ids)} prompt tokens after chat template)')

    print('\n-- prefill --')
    t = time.time()
    out = model(x); mx.eval(out)
    dt_pf = time.time() - t
    pf = tier.stats()
    print(f'  {dt_pf:.1f}s; substrate fetches={pf["misses"]}, '
          f'prefetch_hits={pf["prefetched_hits"]} (overlapped with GPU)')

    print('\n-- decode --')
    decoded = []
    t_dec = time.time()
    for i in range(args.max_tokens):
        ts = time.time()
        out = model(x); mx.eval(out)
        tok = int(mx.argmax(out[0, -1]).item())
        dt = time.time() - ts
        decoded.append(tok)
        x = mx.concatenate([x, mx.array([[tok]])], axis=1)
        snippet = tokenizer.decode([tok])
        print(f'  step {i}: {dt:.2f}s  -> {snippet!r}')
        if tok == tokenizer.eos_token_id:
            break
    dt_dec = time.time() - t_dec
    text = tokenizer.decode(decoded)
    rate = len(decoded) / dt_dec if dt_dec > 0 else 0
    final = tier.stats()

    print(f'\n== output ==')
    print(f'  {text!r}')
    print(f'\n== stats ==')
    print(f'  decode: {dt_dec:.1f}s,  rate: {rate:.3f} tok/s')
    print(f'  substrate fetches: {final["misses"]} (prefetch_hits: {final["prefetched_hits"]})')
    print(f'  fetch p50: {final["fetch_p50_ms"]:.1f} ms (cold SSD), p99: {final["fetch_p99_ms"]:.1f} ms')
    print(f'  cache: {final["cache_bytes_used"] // (1<<20)} MiB / {final["cache_max_bytes"] // (1<<20)} MiB')
    print(f'  evictions: {final["evictions"]}')

    tier.shutdown()
    print(f'\nSubstrate successfully ran {args.model} on this hardware.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
