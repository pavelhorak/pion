#!/usr/bin/env python3
"""MOE.EXPERT.FETCH bit-parity regression — wire vs model-loaded tensors.

Catches the class of bug that wasted hours on 2026-05-19: the wire client's
`_reconstruct_array` reinterpreted OLMoE's fp16 quantized scales/biases as
bf16, producing ~1e-21 garbage values. The model then ran with effectively-
zero MoE outputs, but the eval metrics LOOKED correct (next-token argmax was
stable across all prune fractions because the lm_head produced reasonable
output from the bug-broken residual stream).

The cheap, definitive check: for one (layer, expert) per model architecture
we have cached locally, fetch via wire and compare BIT-FOR-BIT against the
mlx_lm model's own loaded tensor. If `mx.all(wire == model)` is True for
every (proj, comp), the wire path is verified end-to-end.

Skipped silently when no candidate models are cached locally.

Coverage:
  - Gemma-4-26B-A4B-bf16        — stacked layout, unquantized (bits=0)
  - OLMoE-1B-7B-Instruct-4bit   — per-expert layout, 4-bit quant, fp16 scales

Run:
  python3 tests/test_moe_wire_tensor_parity.py
"""
import os, socket, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The binary under test: PION_BIN (set by tests/run_all.py), else the -O3
# release build. Never the dev build: correctness must be shown at -O3.
PION = Path(os.environ.get('PION_BIN') or REPO / 'pion-server')
if not PION.exists():
    raise SystemExit(f'No server binary at {PION}. Run `pixi run build` first.')


def free_port() -> int:
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close()
    return p


def spawn(port: int, moe_cache: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(PION), '-p', str(port), '-w', '1', '--no-auto-embed',
         '--moe-cache', moe_cache, '--moe-cache-mib', '512'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                time.sleep(0.4)
                return proc
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f'pion-server-dev did not come up on port {port}')


def find_snapshot(repo_name_substring: str) -> Path | None:
    """Locate a HuggingFace cache snapshot dir for a model matching the substring."""
    hub = Path.home() / '.cache/huggingface/hub'
    if not hub.exists():
        return None
    for d in hub.iterdir():
        if not d.is_dir() or not d.name.startswith('models--'):
            continue
        if repo_name_substring.lower() not in d.name.lower():
            continue
        snap_root = d / 'snapshots'
        if not snap_root.exists():
            continue
        for s in snap_root.iterdir():
            if (s / 'config.json').exists():
                return s
    return None


def check_model(snap_dir: Path, arch: str) -> list[str]:
    """Spawn pion-server with this model, then compare every (proj, comp)
    tensor of layer=0/expert=0 between wire fetch and mlx-lm model load.

    Returns a list of failure strings (empty if all match)."""
    failures: list[str] = []
    snap_id = snap_dir.name
    port = free_port()
    print(f'\n=== {snap_dir.parent.parent.name}  (arch={arch}, port={port}) ===')
    proc = spawn(port, str(snap_dir))
    try:
        sys.path.insert(0, str(REPO / 'experiments' / 'moe_expert_phase0'))
        try:
            from pion_moe_tier_client import PionMoEExpertTierClient   # type: ignore
        except ModuleNotFoundError:
            print("SKIP: pion_moe_tier_client is in the private research tree "
                  "(experiments/), stripped from the public export.")
            return []
        from mlx_lm import load
        import mlx.core as mx

        model, _ = load(str(snap_dir), lazy=True)
        tier = PionMoEExpertTierClient(
            model_id=snap_id, host='127.0.0.1', port=port, architecture=arch)
        wire = tier.fetch_expert(0, 0)
        # Arch-specific path to a MoE layer + the SwitchGLU/SwitchMLP module:
        #   Gemma 4 26B-A4B: model.language_model.model.layers[L].experts
        #     — but only some layers are MoE; pick first with enable_moe=True.
        #     Pion's MOE.EXPERT.FETCH server-side enumerates ALL MoE layers
        #     contiguously (it doesn't expose Gemma's hybrid layer 0 mapping
        #     directly), so we fetch wire layer 0 and compare it to the
        #     model's FIRST MoE layer.
        #   OLMoE / Phi-3.5: model.model.layers[0].mlp (every layer is MoE).
        if arch == 'gemma4_moe':
            lm_root = model.language_model.model
            moe_layer_idx = None
            for lid, lyr in enumerate(lm_root.layers):
                if getattr(lyr, 'enable_moe', False):
                    moe_layer_idx = lid; break
            if moe_layer_idx is None:
                failures.append('no Gemma 4 layer with enable_moe=True')
                return failures
            print(f'  (model first MoE layer = {moe_layer_idx}; wire layer 0 maps to this)')
            layer = lm_root.layers[moe_layer_idx]
            # Gemma 4: layer.experts.switch_glu is the SwitchGLU with .gate_proj etc.
            sw = layer.experts.switch_glu
        else:
            lm_root = model.model
            layer = lm_root.layers[0]
            block = layer.mlp if hasattr(layer, 'mlp') else layer
            # OLMoE: block.switch_mlp; Phi-3.5: block.block_sparse_moe.switch_mlp
            if hasattr(block, 'switch_mlp'):
                sw = block.switch_mlp
            elif hasattr(block, 'block_sparse_moe'):
                sw = block.block_sparse_moe.switch_mlp
            else:
                failures.append(f'unrecognised MoE block layout: {type(block).__name__}')
                return failures

        # For each (proj, comp) Pion knows about, locate the model's tensor for
        # expert 0 and bit-compare. Components vary by quantization.
        components = ('weight', 'scales', 'biases') if tier.quantized else ('weight',)
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            proj_mod = getattr(sw, proj)
            for comp in components:
                key = (proj, comp)
                if key not in wire:
                    failures.append(f'wire missing ({proj},{comp})')
                    continue
                wire_t = wire[key]
                model_t = getattr(proj_mod, comp)[0]   # expert 0 along axis 0
                if wire_t.shape != model_t.shape:
                    failures.append(f'shape mismatch {key}: wire={wire_t.shape} model={model_t.shape}')
                    continue
                if wire_t.dtype != model_t.dtype:
                    failures.append(f'dtype mismatch {key}: wire={wire_t.dtype} model={model_t.dtype}')
                    continue
                # Bit-compare (cast to a common dtype first if needed; for
                # equal-dtype tensors == is bit-equal for non-NaN values).
                if not bool(mx.all(wire_t == model_t).item()):
                    failures.append(f'value mismatch {key}: wire norm={float(mx.linalg.norm(wire_t.astype(mx.float32)).item()):.4f} '
                                      f'model norm={float(mx.linalg.norm(model_t.astype(mx.float32)).item()):.4f}')
                else:
                    print(f'  ✓ {proj}.{comp}: shape={wire_t.shape}, dtype={wire_t.dtype}, BIT-EQUAL')
        tier.shutdown()
    finally:
        proc.terminate()
        try: proc.wait(5)
        except subprocess.TimeoutExpired: proc.kill()
    return failures


def main() -> int:
    candidates = [
        ('gemma-4-26b-a4b-it-bf16',   'gemma4_moe'),
        ('OLMoE-1B-7B-0125-Instruct-4bit', 'olmoe'),
        ('Phi-3.5-MoE-instruct-4bit', 'phi35_moe'),
    ]
    total_failures: list[str] = []
    n_checked = 0
    for substring, arch in candidates:
        snap = find_snapshot(substring)
        if snap is None:
            print(f'[skip] no cached snapshot matching "{substring}"')
            continue
        n_checked += 1
        failures = check_model(snap, arch)
        if failures:
            print(f'  FAIL ({len(failures)}):')
            for f in failures:
                print(f'    {f}')
            total_failures.extend(failures)

    if n_checked == 0:
        print('\n[skip] No candidate models cached locally; nothing to verify.')
        return 0
    if total_failures:
        print(f'\nFAIL — {len(total_failures)} bit-parity mismatch(es) across {n_checked} model(s)')
        return 1
    print(f'\nPASS — all wire-fetched tensors bit-equal to model-loaded values across {n_checked} model(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
