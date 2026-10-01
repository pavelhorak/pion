"""pion_moe_tier — reusable MoE-expert substrate for MLX models on Apple Silicon.

SHIPPED HERE 2026-09-19. `examples/moe_expert_substrate_demo.py` imported this
by inserting a research directory onto sys.path — a directory that does not
ship. The demo therefore failed at import for everyone outside
this repo, which for the capstone example is worse than a dead documentation
link: it is a traceback on the thing the pitch points at. Moved next to its
only consumer so the import is local and the demo runs from a clean checkout.

Extracts the validated substrate primitive from the Phase-0 spikes into a
single importable module. The wire-surface productization in the Pion server
(MOE.EXPERT.{STORE,FETCH,PREFETCH,PIN,UNPIN,INFO,STATS} commands) will mirror
this Python API; for now, in-process consumers can use this directly.

Key design choices (proven empirically across Phi-3.5-MoE-4bit, Gemma-4-26B-
A4B-8bit, Gemma-4-26B-A4B-bf16):

  - `os.pread` against safetensors with PRECOMPUTED per-expert byte offsets
    (NOT mmap; mmap-backed lazy slices fail GPU command-buffer deadlines)
  - All `mx.array` construction stays on the main MLX thread (MLX is not
    thread-safe). Worker threads only do the byte-level read (GIL releases
    during the syscall — real parallel SSD progress).
  - `mx.eval` on each fetched expert forces unified-memory residence BEFORE
    the GPU op. Skipping this re-introduces the page-fault GPU timeout.
  - fp32 accumulator across experts inside a layer's intercept (sequential
    bf16 adds compound past tolerance over 30 layers).
  - Smaller LRU cache beats larger on memory-constrained systems (large
    cache starves Metal's unified-memory working pool; observed p99 fetch
    explodes from 30 ms → 700+ ms at 8 GB cache on 16 GB Mac).

Usage:

    from pion_moe_tier import MoEExpertTier, install_moe_substrate
    from mlx_lm import load

    # Skip mlx-lm's eager materialization
    model, tokenizer = load(model_path, lazy=True)

    tier = MoEExpertTier(model_path, cache_mib=1024, workers=4)
    install_moe_substrate(model, tier)

    out = model(x)   # forward goes through substrate; only ~3 GB resident

See spike_*.py in this folder for full end-to-end examples.
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nnx
import numpy as np


# ─── safetensors layout discovery ─────────────────────────────────────────────

@dataclass
class ExpertSlot:
    """Per-(layer, expert, projection, component) location in a safetensors shard."""
    layer_id: int
    expert_id: int
    proj: str               # gate_proj | up_proj | down_proj
    component: str          # weight | scales | biases  (scales/biases only for quantized)
    shard_path: str
    abs_offset: int
    per_expert_bytes: int
    per_expert_shape: Tuple[int, ...]
    dtype: str              # 'BF16' | 'F16' | 'F32' | 'U32'


# Recognized model layouts. Each entry describes how to parse safetensors
# keys for that architecture into per-expert slot positions.
@dataclass
class LayoutSpec:
    """How to find per-expert tensors in safetensors keys for one model family."""
    name: str
    # Predicate: does this key represent an expert tensor? Returns
    # (layer_id, expert_axis_dim_idx, proj, component) or None.
    key_parser: Callable[[str], Optional[Tuple[int, int, str, str]]]
    quantized: bool         # True if scales/biases keys are present
    components: Tuple[str, ...]  # ('weight',) or ('weight','scales','biases')


def _phi35_key_parser(k: str):
    """Phi-3.5-MoE has two layouts:
      stacked:    model.layers.{L}.block_sparse_moe.switch_mlp.{proj}.{comp}
      per-expert: model.layers.{L}.block_sparse_moe.experts.{E}.{w1|w2|w3}.{comp}
    """
    parts = k.split('.')
    if (len(parts) < 7 or parts[0] != 'model' or parts[1] != 'layers'
            or parts[3] != 'block_sparse_moe'):
        return None
    lid = int(parts[2])
    if parts[4] == 'switch_mlp':
        # stacked: ['model','layers',L,'block_sparse_moe','switch_mlp','{proj}','{comp}']
        proj = parts[5]
        comp = parts[6] if len(parts) > 6 else 'weight'
        return (lid, -1, proj, comp)
    if parts[4] == 'experts' and len(parts) >= 8 and parts[5].isdigit():
        # per-expert: ['model','layers',L,'block_sparse_moe','experts',E,'{w1|w2|w3}','{comp}']
        eid = int(parts[5])
        w = parts[6]
        comp = parts[7]
        proj = {'w1': 'gate_proj', 'w2': 'down_proj', 'w3': 'up_proj'}.get(w)
        if proj is None:
            return None
        return (lid, eid, proj, comp)
    return None


def _gemma4_key_parser(k: str):
    """Gemma-4: language_model.model.layers.{L}.experts.switch_glu.{proj}.{comp}"""
    parts = k.split('.')
    if (len(parts) < 8 or parts[0] != 'language_model'
            or parts[1] != 'model' or parts[2] != 'layers'
            or parts[4] != 'experts' or parts[5] != 'switch_glu'):
        return None
    lid = int(parts[3])
    proj = parts[6]
    comp = parts[7] if len(parts) > 7 else 'weight'
    return (lid, -1, proj, comp)        # always stacked


def _olmoe_key_parser(k: str):
    """OLMoE: model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.{weight|scales|biases}"""
    parts = k.split('.')
    if (len(parts) < 7 or parts[0] != 'model' or parts[1] != 'layers'
            or parts[3] != 'mlp' or parts[4] != 'experts'):
        return None
    if not parts[5].isdigit():
        return None
    lid = int(parts[2])
    eid = int(parts[5])
    proj = parts[6]
    comp = parts[7] if len(parts) > 7 else 'weight'
    return (lid, eid, proj, comp)


def _detect_layout(weight_keys: List[str]) -> LayoutSpec:
    """Auto-detect which model family from a sample of safetensors keys."""
    sample = weight_keys[:200]
    if any('block_sparse_moe' in k for k in sample):
        # Phi-3.5: probe for stacked vs per-expert
        stacked_seen = any('block_sparse_moe.switch_mlp' in k for k in sample)
        quantized = any('.scales' in k for k in sample)
        components = ('weight','scales','biases') if quantized else ('weight',)
        return LayoutSpec('phi35_moe', _phi35_key_parser, quantized, components)
    if any('experts.switch_glu' in k for k in sample):
        quantized = any('.scales' in k for k in sample)
        components = ('weight','scales','biases') if quantized else ('weight',)
        return LayoutSpec('gemma4_moe', _gemma4_key_parser, quantized, components)
    if any('mlp.experts.' in k for k in sample):
        quantized = any('.scales' in k for k in sample)
        components = ('weight','scales','biases') if quantized else ('weight',)
        return LayoutSpec('olmoe', _olmoe_key_parser, quantized, components)
    raise RuntimeError('Could not detect MoE layout from safetensors keys.')


def build_layout(model_dir: Path, num_experts: int) -> Tuple[Dict[Tuple[int, int, str, str], ExpertSlot], LayoutSpec]:
    """Walk all safetensors shards, build per-expert slot map.

    Returns (slots, layout_spec). `slots[(layer, expert, proj, comp)] -> ExpertSlot`.
    """
    shards = sorted(model_dir.glob('*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'no .safetensors shards in {model_dir}')

    # Sample keys from the first shard to detect layout
    fd0 = os.open(str(shards[0]), os.O_RDONLY)
    hdr_len = struct.unpack('<Q', os.pread(fd0, 8, 0))[0]
    hdr0 = json.loads(os.pread(fd0, hdr_len, 8))
    os.close(fd0)
    layout_spec = _detect_layout([k for k in hdr0 if k != '__metadata__'])

    slots: Dict[Tuple[int, int, str, str], ExpertSlot] = {}
    for shard in shards:
        fd = os.open(str(shard), os.O_RDONLY)
        try:
            hl = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            hdr = json.loads(os.pread(fd, hl, 8))
            data_base = 8 + hl
            for k, info in hdr.items():
                if k == '__metadata__':
                    continue
                parsed = layout_spec.key_parser(k)
                if parsed is None:
                    continue
                lid, eid_or_neg, proj, comp = parsed
                if comp not in layout_spec.components:
                    continue
                shape = info['shape']
                if eid_or_neg == -1:
                    # stacked: leading dim is expert axis
                    if not shape or shape[0] != num_experts:
                        continue
                    per_expert_shape = tuple(shape[1:])
                    elems = 1
                    for s in per_expert_shape:
                        elems *= s
                    bpe = {'U32': 4, 'F16': 2, 'BF16': 2, 'F32': 4}.get(info['dtype'])
                    if bpe is None:
                        continue
                    per_expert_bytes = elems * bpe
                    base = data_base + info['data_offsets'][0]
                    for eid in range(num_experts):
                        slots[(lid, eid, proj, comp)] = ExpertSlot(
                            layer_id=lid, expert_id=eid, proj=proj, component=comp,
                            shard_path=str(shard),
                            abs_offset=base + eid * per_expert_bytes,
                            per_expert_bytes=per_expert_bytes,
                            per_expert_shape=per_expert_shape,
                            dtype=info['dtype'],
                        )
                else:
                    # per-expert key directly
                    per_expert_shape = tuple(shape)
                    elems = 1
                    for s in per_expert_shape:
                        elems *= s
                    bpe = {'U32': 4, 'F16': 2, 'BF16': 2, 'F32': 4}.get(info['dtype'])
                    if bpe is None:
                        continue
                    slots[(lid, eid_or_neg, proj, comp)] = ExpertSlot(
                        layer_id=lid, expert_id=eid_or_neg, proj=proj, component=comp,
                        shard_path=str(shard),
                        abs_offset=data_base + info['data_offsets'][0],
                        per_expert_bytes=elems * bpe,
                        per_expert_shape=per_expert_shape,
                        dtype=info['dtype'],
                    )
        finally:
            os.close(fd)
    return slots, layout_spec


# ─── the tier ─────────────────────────────────────────────────────────────────

class MoEExpertTier:
    """Reusable substrate primitive for MoE expert weights on Apple Silicon + MLX.

    Per-expert fetch via os.pread + LRU cache + ThreadPoolExecutor parallel I/O.
    Designed for memory-constrained systems where the model is too big for
    unified memory but each forward's working set fits comfortably.
    """

    def __init__(self, model_dir: str | Path, *, cache_mib: int = 1024,
                 workers: int = 4, pinned: Optional[set] = None):
        self.model_dir = Path(model_dir)
        self.cfg = self._load_config()
        tc = self.cfg.get('text_config', self.cfg)
        self.num_layers = tc['num_hidden_layers']
        self.num_experts = (tc.get('num_experts')
                            or tc.get('num_local_experts'))
        if self.num_experts is None:
            raise RuntimeError('No num_experts / num_local_experts in config')
        self.top_k = (tc.get('top_k_experts')
                      or tc.get('num_experts_per_tok'))
        self.layout, self.spec = build_layout(self.model_dir, self.num_experts)
        self.quantized = self.spec.quantized
        if self.quantized:
            q = self.cfg.get('quantization') or {}
            self.bits = q.get('bits', 8)
            self.group_size = q.get('group_size', 64)
        else:
            self.bits = self.group_size = None

        self.cache_max_bytes = cache_mib * (1 << 20)
        self.cache_bytes = 0
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.pinned = set(pinned or set())
        self.fds: Dict[str, int] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix='pion-moe-tier')
        self.pending: Dict[Tuple[int, int], concurrent.futures.Future] = {}

        self.hits = 0
        self.misses = 0
        self.prefetched_hits = 0
        self.evictions = 0
        self.pinned_evict_skipped = 0
        self.fetch_ms: List[float] = []
        self.wait_ms: List[float] = []
        # Per-(layer, expert) selection counters — observability for
        # predictor training data + capacity planning. Derived per-layer
        # totals come from summing access_counts over expert axis.
        self.access_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)

    def _load_config(self) -> dict:
        return json.loads((self.model_dir / 'config.json').read_text())

    def _fd(self, shard: str) -> int:
        fd = self.fds.get(shard)
        if fd is None:
            fd = os.open(shard, os.O_RDONLY)
            self.fds[shard] = fd
        return fd

    def _pread_raw(self, lid: int, eid: int) -> dict:
        """Worker-thread byte reads for one expert's projections."""
        out = {}
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            for comp in self.spec.components:
                slot = self.layout.get((lid, eid, proj, comp))
                if slot is None:
                    continue
                buf = os.pread(self._fd(slot.shard_path),
                               slot.per_expert_bytes, slot.abs_offset)
                out[(proj, comp)] = (buf, slot)
        return out

    @staticmethod
    def _bytes_to_mxarray(buf: bytes, slot: ExpertSlot) -> mx.array:
        """Main-thread mx.array construction (MLX is not thread-safe)."""
        if slot.dtype == 'BF16':
            arr_u16 = np.frombuffer(buf, dtype=np.uint16).reshape(slot.per_expert_shape)
            return mx.array(arr_u16).view(mx.bfloat16)
        if slot.dtype == 'F16':
            arr = np.frombuffer(buf, dtype=np.float16).reshape(slot.per_expert_shape)
            return mx.array(arr)
        if slot.dtype == 'U32':
            arr = np.frombuffer(buf, dtype=np.uint32).reshape(slot.per_expert_shape)
            return mx.array(arr)
        if slot.dtype == 'F32':
            arr = np.frombuffer(buf, dtype=np.float32).reshape(slot.per_expert_shape)
            return mx.array(arr)
        raise ValueError(f'unsupported dtype: {slot.dtype}')

    def prefetch(self, lid: int, eids: List[int]) -> None:
        """Submit pread futures for the given experts. Non-blocking."""
        for eid in eids:
            key = (lid, eid)
            if key in self.cache or key in self.pending:
                continue
            self.pending[key] = self.executor.submit(self._pread_raw, lid, eid)

    def fetch_expert(self, lid: int, eid: int) -> Dict[Tuple[str, str], mx.array]:
        """Blocking fetch. Returns dict of (proj, comp) -> mx.array."""
        key = (lid, eid)
        self.access_counts[key] += 1
        t0 = time.perf_counter()
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        if key in self.pending:
            fut = self.pending.pop(key)
            tw = time.perf_counter()
            raw = fut.result()
            self.wait_ms.append((time.perf_counter() - tw) * 1000)
            self.prefetched_hits += 1
        else:
            raw = self._pread_raw(lid, eid)
            self.wait_ms.append(0.0)
        tensors = {}
        size_total = 0
        for (proj, comp), (buf, slot) in raw.items():
            arr = self._bytes_to_mxarray(buf, slot)
            mx.eval(arr)
            tensors[(proj, comp)] = arr
            size_total += slot.per_expert_bytes
        self._evict_to_fit(size_total)
        self.cache[key] = tensors
        self.cache_bytes += size_total
        self.misses += 1
        self.fetch_ms.append((time.perf_counter() - t0) * 1000)
        return tensors

    def _evict_to_fit(self, incoming_bytes: int) -> None:
        """LRU eviction respecting pinned set."""
        while self.cache_bytes + incoming_bytes > self.cache_max_bytes and self.cache:
            # Find oldest non-pinned entry
            evict_key = None
            for k in self.cache:
                if k not in self.pinned:
                    evict_key = k
                    break
            if evict_key is None:
                # All resident entries are pinned; nothing to evict
                self.pinned_evict_skipped += 1
                break
            removed = self.cache.pop(evict_key)
            for v in removed.values():
                self.cache_bytes -= v.nbytes
            self.evictions += 1

    def pin(self, lid: int, eid: int) -> None:
        self.pinned.add((lid, eid))

    def unpin(self, lid: int, eid: int) -> None:
        self.pinned.discard((lid, eid))

    def info(self) -> dict:
        return {
            'model_dir': str(self.model_dir),
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'layout': self.spec.name,
            'quantized': self.quantized,
            'bits': self.bits,
            'group_size': self.group_size,
            'cache_max_bytes': self.cache_max_bytes,
            'workers': self.executor._max_workers,
        }

    def stats(self) -> dict:
        n = self.hits + self.misses
        fms = sorted(self.fetch_ms) if self.fetch_ms else [0]
        wms = sorted(self.wait_ms) if self.wait_ms else [0]
        return {
            'hits': self.hits, 'misses': self.misses,
            'prefetched_hits': self.prefetched_hits,
            'hit_rate': self.hits / n if n > 0 else 0.0,
            'evictions': self.evictions,
            'pinned_evict_skipped': self.pinned_evict_skipped,
            'cache_bytes_used': self.cache_bytes,
            'cache_max_bytes': self.cache_max_bytes,
            'fetch_p50_ms': fms[len(fms) // 2],
            'fetch_p99_ms': fms[max(0, int(len(fms) * 0.99) - 1)],
            'wait_p50_ms': wms[len(wms) // 2],
            'wait_p99_ms': wms[max(0, int(len(wms) * 0.99) - 1)],
            'pinned_count': len(self.pinned),
        }

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()

    def routing_summary(self, top_n: int = 16) -> dict:
        """Per-layer routing histogram. The N most-accessed experts per layer,
        with their fraction of that layer's total accesses. Useful for:
          - identifying always-hot experts (PIN candidates)
          - estimating predictor potential (concentration of routing mass)
          - capacity planning (cache sizing if you can hold the top-N hot set)
        """
        # Group accesses by layer
        by_layer: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        for (lid, eid), n in self.access_counts.items():
            by_layer[lid].append((eid, n))
        out = {}
        for lid in sorted(by_layer):
            entries = sorted(by_layer[lid], key=lambda kv: -kv[1])
            total = sum(n for _, n in entries)
            top = []
            for eid, n in entries[:top_n]:
                top.append({'expert_id': eid, 'count': n,
                             'fraction': n / total if total > 0 else 0.0})
            out[lid] = {'total_accesses': total,
                        'unique_experts': len(entries),
                        'top_experts': top,
                        # Concentration ratio: top-K / total. Higher = more
                        # predictable routing → bigger predictor win available.
                        'concentration_top_k': (sum(n for _, n in entries[:self.top_k or 8]) / total
                                                  if total > 0 else 0.0)}
        return out

    def routing_csv(self) -> str:
        """Same data as routing_summary but flat CSV for export.
        Format: layer,expert,access_count,fraction_of_layer
        """
        by_layer: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        for (lid, eid), n in self.access_counts.items():
            by_layer[lid].append((eid, n))
        lines = ['layer,expert,access_count,fraction_of_layer']
        for lid in sorted(by_layer):
            entries = sorted(by_layer[lid], key=lambda kv: -kv[1])
            total = sum(n for _, n in entries)
            for eid, n in entries:
                frac = n / total if total > 0 else 0.0
                lines.append(f'{lid},{eid},{n},{frac:.4f}')
        return '\n'.join(lines)


# ─── intercepts per architecture ─────────────────────────────────────────────

def _install_gemma4(model, tier: MoEExpertTier):
    """Replace gemma4_text.Experts.__call__. Routing already done by Router."""
    lm = model.language_model.model
    layer_for_id: Dict[int, int] = {}
    n_moe = 0
    for lid, block in enumerate(lm.layers):
        if getattr(block, 'enable_moe', False):
            layer_for_id[id(block.experts)] = lid
            n_moe += 1
    if not layer_for_id:
        raise RuntimeError('no MoE layers found on Gemma 4 model')
    ExpertsCls = type(lm.layers[0].experts)
    quantized = tier.quantized
    bits, gs = tier.bits, tier.group_size

    def gemma4_experts(self, x, top_k_indices, top_k_weights):
        B, L, D = x.shape
        K = top_k_indices.shape[-1]
        x_flat = x.reshape(-1, D)
        inds = top_k_indices.reshape(-1, K)
        scores = top_k_weights.reshape(-1, K)
        N = x_flat.shape[0]
        lid = layer_for_id[id(self)]
        flat = inds.flatten().tolist()
        unique = sorted(set(flat))
        tier.prefetch(lid, unique)

        output = mx.zeros((N, D), dtype=mx.float32)
        dropped = mx.zeros((N,), dtype=mx.float32)   # per-token mass lost to PRUNE
        scores_fp32 = scores.astype(mx.float32)
        for eid in unique:
            try:
                w = tier.fetch_expert(lid, eid)
            except RuntimeError as e:
                # Stage-3 PRUNE: server returned -PRUNED. Record the per-token
                # weight that would have gone to this expert so the surviving
                # experts can be renormalized after the loop. Other RuntimeErrors
                # are real failures — propagate.
                if 'PRUNED' not in str(e):
                    raise
                mask = (inds == eid).astype(mx.float32)
                dropped = dropped + (scores_fp32 * mask).sum(axis=-1)
                continue
            mask = (inds == eid).astype(mx.float32)
            tok_s = (scores_fp32 * mask).sum(axis=-1)
            if quantized:
                gw, gsc, gbi = w[('gate_proj','weight')], w[('gate_proj','scales')], w[('gate_proj','biases')]
                uw, usc, ubi = w[('up_proj','weight')], w[('up_proj','scales')], w[('up_proj','biases')]
                dw, dsc, dbi = w[('down_proj','weight')], w[('down_proj','scales')], w[('down_proj','biases')]
                go = mx.quantized_matmul(x_flat, gw, scales=gsc, biases=gbi,
                                          transpose=True, group_size=gs, bits=bits)
                uo = mx.quantized_matmul(x_flat, uw, scales=usc, biases=ubi,
                                          transpose=True, group_size=gs, bits=bits)
                inter = nnx.gelu_approx(go) * uo
                eo = mx.quantized_matmul(inter, dw, scales=dsc, biases=dbi,
                                          transpose=True, group_size=gs, bits=bits)
            else:
                # bf16/f16 lossless path
                gw = w[('gate_proj','weight')]
                uw = w[('up_proj','weight')]
                dw = w[('down_proj','weight')]
                go = x_flat @ gw.T
                uo = x_flat @ uw.T
                inter = nnx.gelu_approx(go) * uo
                eo = inter @ dw.T
            output = output + tok_s[:, None] * eo.astype(mx.float32)
            mx.eval(output)
        # Renormalize survivors: each surviving expert's contribution scales by
        # 1 / (1 - dropped_per_token). If ALL of a token's top-K are pruned,
        # cap at 1e-6 so the math doesn't explode (token gets zero MoE output
        # rather than NaN — better than a crash).
        keep_factor = 1.0 / mx.maximum(1.0 - dropped, mx.array(1e-6, dtype=mx.float32))
        output = output * keep_factor[:, None]
        return output.astype(x_flat.dtype).reshape(B, L, D)

    ExpertsCls.__call__ = gemma4_experts
    return n_moe


def _install_phi35(model, tier: MoEExpertTier):
    """Replace PhiMoESparseMoeBlock.__call__. Phi-3.5 does routing inside the
    block (no external Router class). top-2 of 16, softmax-after-top-K.

    The MoE block lives at either `layer.mlp` (older mlx-lm) or
    `layer.block_sparse_moe` (current mlx-lm). We probe both.
    """
    def _moe_block(lyr):
        # Returns (block, attr_name) or (None, None) if layer is non-MoE.
        for name in ('mlp', 'block_sparse_moe'):
            sub = getattr(lyr, name, None)
            if sub is not None and hasattr(sub, 'switch_mlp'):
                return sub, name
        return None, None

    layer_for_id: Dict[int, int] = {}
    n_moe = 0
    first_block = None
    for lid, block in enumerate(model.model.layers):
        moe_blk, _attr = _moe_block(block)
        if moe_blk is not None:
            layer_for_id[id(moe_blk)] = lid
            n_moe += 1
            if first_block is None:
                first_block = moe_blk
    if not layer_for_id or first_block is None:
        raise RuntimeError('no MoE layers found on Phi-3.5 model')
    MoEBlockCls = type(first_block)
    quantized = tier.quantized
    bits, gs = tier.bits, tier.group_size

    def phi35_block(self, x):
        # Phi-3.5 routing: gate -> top-k -> softmax over k.
        # Same routing for Mixtral; Mixtral's block names the attribute
        # `num_experts_per_tok` while Phi-3.5 names it `top_k` — probe both.
        gates = self.gate(x)
        k = getattr(self, 'top_k', None)
        if k is None:
            k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)

        if x.ndim == 3:
            B, L, D = x.shape
            x_flat = x.reshape(-1, D)
            inds_flat = inds.reshape(-1, k)
            scores_flat = scores.reshape(-1, k)
            unflatten = True
        else:
            B, L = 1, 1
            x_flat = x
            D = x.shape[-1]
            inds_flat = inds
            scores_flat = scores
            unflatten = False
        N = x_flat.shape[0]

        lid = layer_for_id[id(self)]
        flat = inds_flat.flatten().tolist()
        unique = sorted(set(flat))
        tier.prefetch(lid, unique)

        output = mx.zeros((N, D), dtype=mx.float32)
        dropped = mx.zeros((N,), dtype=mx.float32)   # Stage-3 PRUNE renorm
        scores_fp32 = scores_flat.astype(mx.float32)
        for eid in unique:
            try:
                w = tier.fetch_expert(lid, eid)
            except RuntimeError as e:
                if 'PRUNED' not in str(e):
                    raise
                mask = (inds_flat == eid).astype(mx.float32)
                dropped = dropped + (scores_fp32 * mask).sum(axis=-1)
                continue
            mask = (inds_flat == eid).astype(mx.float32)
            tok_s = (scores_fp32 * mask).sum(axis=-1)
            if quantized:
                gw, gsc, gbi = w[('gate_proj','weight')], w[('gate_proj','scales')], w[('gate_proj','biases')]
                uw, usc, ubi = w[('up_proj','weight')], w[('up_proj','scales')], w[('up_proj','biases')]
                dw, dsc, dbi = w[('down_proj','weight')], w[('down_proj','scales')], w[('down_proj','biases')]
                go = mx.quantized_matmul(x_flat, gw, scales=gsc, biases=gbi,
                                          transpose=True, group_size=gs, bits=bits)
                uo = mx.quantized_matmul(x_flat, uw, scales=usc, biases=ubi,
                                          transpose=True, group_size=gs, bits=bits)
                # Phi-3.5 uses SiLU (Llama-style SwiGLU), not GeGLU
                inter = nnx.silu(go) * uo
                eo = mx.quantized_matmul(inter, dw, scales=dsc, biases=dbi,
                                          transpose=True, group_size=gs, bits=bits)
            else:
                gw = w[('gate_proj','weight')]
                uw = w[('up_proj','weight')]
                dw = w[('down_proj','weight')]
                go = x_flat @ gw.T
                uo = x_flat @ uw.T
                inter = nnx.silu(go) * uo
                eo = inter @ dw.T
            output = output + tok_s[:, None] * eo.astype(mx.float32)
            mx.eval(output)
        keep_factor = 1.0 / mx.maximum(1.0 - dropped, mx.array(1e-6, dtype=mx.float32))
        output = output * keep_factor[:, None]
        y = output.astype(x_flat.dtype)
        return y.reshape(B, L, D) if unflatten else y

    MoEBlockCls.__call__ = phi35_block
    return n_moe


def _install_olmoe(model, tier: MoEExpertTier):
    """Replace OlmoeSparseMoeBlock.__call__. OLMoE: top-8 of 64, optional
    norm_topk_prob.
    """
    layer_for_id: Dict[int, int] = {}
    n_moe = 0
    for lid, block in enumerate(model.model.layers):
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'switch_mlp'):
            layer_for_id[id(block.mlp)] = lid
            n_moe += 1
    if not layer_for_id:
        raise RuntimeError('no MoE layers found on OLMoE model')
    MoEBlockCls = type(model.model.layers[0].mlp)
    quantized = tier.quantized
    bits, gs = tier.bits, tier.group_size

    def olmoe_block(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        routing_weights = mx.softmax(router_logits, axis=1, precise=True)
        k = self.top_k
        inds_flat = mx.stop_gradient(
            mx.argpartition(-routing_weights, kth=k - 1, axis=-1)[..., :k]
        )
        scores_flat = mx.take_along_axis(routing_weights, inds_flat, axis=-1)
        if getattr(self, 'norm_topk_prob', False):
            scores_flat = scores_flat / scores_flat.sum(axis=-1, keepdims=True)
        N = x_flat.shape[0]

        lid = layer_for_id[id(self)]
        flat = inds_flat.flatten().tolist()
        unique = sorted(set(flat))
        tier.prefetch(lid, unique)

        output = mx.zeros((N, D), dtype=mx.float32)
        dropped = mx.zeros((N,), dtype=mx.float32)   # Stage-3 PRUNE renorm
        scores_fp32 = scores_flat.astype(mx.float32)
        for eid in unique:
            try:
                w = tier.fetch_expert(lid, eid)
            except RuntimeError as e:
                if 'PRUNED' not in str(e):
                    raise
                mask = (inds_flat == eid).astype(mx.float32)
                dropped = dropped + (scores_fp32 * mask).sum(axis=-1)
                continue
            mask = (inds_flat == eid).astype(mx.float32)
            tok_s = (scores_fp32 * mask).sum(axis=-1)
            if quantized:
                gw, gsc, gbi = w[('gate_proj','weight')], w[('gate_proj','scales')], w[('gate_proj','biases')]
                uw, usc, ubi = w[('up_proj','weight')], w[('up_proj','scales')], w[('up_proj','biases')]
                dw, dsc, dbi = w[('down_proj','weight')], w[('down_proj','scales')], w[('down_proj','biases')]
                go = mx.quantized_matmul(x_flat, gw, scales=gsc, biases=gbi,
                                          transpose=True, group_size=gs, bits=bits)
                uo = mx.quantized_matmul(x_flat, uw, scales=usc, biases=ubi,
                                          transpose=True, group_size=gs, bits=bits)
                inter = nnx.silu(go) * uo
                eo = mx.quantized_matmul(inter, dw, scales=dsc, biases=dbi,
                                          transpose=True, group_size=gs, bits=bits)
            else:
                gw = w[('gate_proj','weight')]
                uw = w[('up_proj','weight')]
                dw = w[('down_proj','weight')]
                go = x_flat @ gw.T
                uo = x_flat @ uw.T
                inter = nnx.silu(go) * uo
                eo = inter @ dw.T
            output = output + tok_s[:, None] * eo.astype(mx.float32)
            mx.eval(output)
        keep_factor = 1.0 / mx.maximum(1.0 - dropped, mx.array(1e-6, dtype=mx.float32))
        output = output * keep_factor[:, None]
        return output.astype(x_flat.dtype).reshape(B, L, D)

    MoEBlockCls.__call__ = olmoe_block
    return n_moe


def install_moe_substrate(model, tier: MoEExpertTier) -> int:
    """Auto-detect the model family and install the substrate intercept.

    Returns the number of MoE layers intercepted. Supports:
      - gemma4_moe (Gemma 4 26B-A4B; shared MLP + sparse experts hybrid)
      - phi35_moe  (Phi-3.5-MoE; pure sparse, routing inside the block)
      - olmoe      (OLMoE; pure sparse with optional norm_topk_prob)
    """
    if tier.spec.name == 'gemma4_moe':
        return _install_gemma4(model, tier)
    if tier.spec.name == 'phi35_moe':
        return _install_phi35(model, tier)
    if tier.spec.name == 'olmoe':
        return _install_olmoe(model, tier)
    raise NotImplementedError(
        f'install_moe_substrate: unknown layout {tier.spec.name!r}')
