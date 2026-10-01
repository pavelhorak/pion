"""pion_moe_tier_client — wire-backed MoE expert tier client.

SHIPPED HERE 2026-09-19, for the same reason as pion_moe_tier.py beside it:
pion-serve's `pion-moe` backend imported this from
a research directory that does not ship, so that
backend raised ImportError for every reader outside the development repo. It
is a lazy init, so serve.py still started — the failure only appeared when
someone selected the backend, which is worse than failing loudly.
substrate. Drop-in replacement for the in-process MoEExpertTier.

The in-process tier (`pion_moe_tier.py`) is per-consumer: each process loads
its own offset table, opens its own shards, holds its own LRU cache. That
duplicates the substrate state across N consumers.

The client variant talks to a SHARED Pion server over the RESP protocol on
TCP/IP. N consumers point at the same server; the server's LRU cache is
amortized across all of them; expert bytes are read from disk ONCE per
(model, layer, expert) and re-served from RAM thereafter.

API matches the in-process MoEExpertTier so `install_moe_substrate(model, tier)`
in `pion_moe_tier.py` works against either. No changes to the consumer code.

Usage (after `pion-server --moe-cache <model_dir>` is running):

    from pion_moe_tier_client import PionMoEExpertTierClient
    from pion_moe_tier import install_moe_substrate
    from mlx_lm import load

    model, tok = load(model_dir, lazy=True)
    tier = PionMoEExpertTierClient(model_id='<snapshot-hash>',
                                    host='127.0.0.1', port=1974,
                                    num_experts=128, num_layers=30,
                                    quantized=False)
    install_moe_substrate(model, tier)
    # forward through model — fetches now go through Pion server
"""
from __future__ import annotations

import collections
import socket
import struct
import time
from typing import Dict, Optional, Tuple

import mlx.core as mx
import numpy as np


# Match MoEExpertTier's auto-detection result names so install_moe_substrate
# can dispatch the right intercept.
class _LayoutSpec:
    def __init__(self, name: str, quantized: bool, components: tuple):
        self.name = name
        self.quantized = quantized
        self.components = components


class PionMoEExpertTierClient:
    """Drop-in client for the Pion server's MOE.EXPERT.* tier.

    Mirrors the public surface of `pion_moe_tier.MoEExpertTier`:
      - `info()`, `stats()`, `routing_summary()`
      - `prefetch(lid, eids)`, `fetch_expert(lid, eid)`
      - `pin(lid, eid)`, `unpin(lid, eid)`
      - `top_k`, `num_layers`, `num_experts`, `quantized`, `bits`, `group_size`
      - `spec` (a _LayoutSpec object matching what _detect_layout returns)

    Reconstructs mx.array tensors from the multi-blob FETCH response. The
    per-expert shape is derived from the model manifest (received via INFO).
    """

    def __init__(self, *, model_id: str, host: str = '127.0.0.1', port: int = 1974,
                 architecture: str = 'gemma4_moe',
                 timeout: float = 30.0):
        """architecture: one of 'gemma4_moe' | 'phi35_moe' | 'olmoe' — used to
        select the right intercept in pion_moe_tier.install_moe_substrate.

        host:port is the Pion RESP listener (default 1974).
        """
        self.model_id = model_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

        # Query the server's INFO for manifest
        info = self.info()
        self.num_layers = info['layers']
        self.num_experts = info['experts']
        self.top_k = info['top_k']
        self.bits = info.get('bits', 0)
        self.group_size = info.get('group_size', 0)
        self.hidden_size = info.get('hidden_size', 0)
        self.moe_intermediate = info.get('moe_intermediate', 0)
        # bits=0 means lossless (bf16/f16/f32); >0 means quantized
        self.quantized = self.bits > 0
        components = ('weight', 'scales', 'biases') if self.quantized else ('weight',)
        self.spec = _LayoutSpec(architecture, self.quantized, components)

        # Quantized arch scales/biases dtype convention (mlx-lm 4-bit quant):
        #   OLMoE   → fp16 scales/biases
        #   Phi-3.5 → fp16 scales/biases
        #   (Gemma-4-26B-A4B-bf16 has bits=0, no quantized scales path)
        # Without this distinction we'd reinterpret fp16 bytes as bf16 →
        # weights matmul to ~1e-21 garbage → MoE returns numerical-zero
        # output → substrate looks "lossless at every fraction" trivially.
        # Override via set_scales_dtype() if a future arch uses bf16 quant.
        if self.quantized:
            if architecture in ('olmoe', 'phi35_moe'):
                self._scales_dtype = 'fp16'
            else:
                self._scales_dtype = 'bf16'
        else:
            self._scales_dtype = 'bf16'   # unused when not quantized

        # Tier-side counters (server's view is canonical; mirror locally for
        # consumer-side telemetry / install_moe_substrate compatibility)
        self.hits = 0
        self.misses = 0
        self.prefetched_hits = 0
        self.evictions = 0
        self.access_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)

        # Per-(proj, comp) → shape registry. Populated by _auto_register_shapes()
        # when INFO exposes hidden_size + moe_intermediate; caller can override
        # via set_tensor_shape().
        self._tensor_shapes: Dict[Tuple[str, str], tuple] = {}
        if self.hidden_size > 0 and self.moe_intermediate > 0:
            self._auto_register_shapes()

    # ─── Protocol helpers ─────────────────────────────────────────────────────

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        sk = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sk.settimeout(self.timeout)
        self._sock = sk
        return sk

    def _close(self):
        if self._sock is not None:
            try: self._sock.close()
            except OSError: pass
            self._sock = None

    def _send_resp(self, *args: bytes | str) -> bytes:
        """Send a RESP-array command + return the raw reply line(s) + payload.

        Returns the response BODY (after the type-prefix line). For a
        bulk-string reply ($N\r\n<N bytes>\r\n), returns the <N bytes>.
        For a simple string (+OK\r\n), returns b'OK'. For an error (-ERR...),
        raises RuntimeError.
        """
        sk = self._connect()
        # Encode the command
        parts = [b'*' + str(len(args)).encode() + b'\r\n']
        for a in args:
            if isinstance(a, str):
                a = a.encode()
            parts.append(b'$' + str(len(a)).encode() + b'\r\n' + a + b'\r\n')
        sk.sendall(b''.join(parts))

        # Parse one reply
        return self._read_reply(sk)

    def _read_line(self, sk: socket.socket) -> bytes:
        """Read until CRLF."""
        buf = bytearray()
        while True:
            c = sk.recv(1)
            if not c:
                raise RuntimeError('connection closed')
            buf.extend(c)
            if buf.endswith(b'\r\n'):
                return bytes(buf[:-2])

    def _read_exact(self, sk: socket.socket, n: int) -> bytes:
        """Read exactly n bytes."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sk.recv(min(65536, n - len(buf)))
            if not chunk:
                raise RuntimeError('connection closed mid-payload')
            buf.extend(chunk)
        return bytes(buf)

    def _read_reply(self, sk: socket.socket) -> bytes:
        line = self._read_line(sk)
        if not line:
            raise RuntimeError('empty reply')
        prefix, rest = chr(line[0]), line[1:]
        if prefix == '+':
            return rest
        if prefix == '-':
            raise RuntimeError(rest.decode('utf-8', errors='replace'))
        if prefix == '$':
            n = int(rest)
            if n < 0:
                return b''
            payload = self._read_exact(sk, n)
            sk.recv(2)   # trailing \r\n
            return payload
        if prefix == ':':
            return rest
        raise RuntimeError(f'unexpected reply type: {prefix}')

    # ─── Shape registry ───────────────────────────────────────────────────────

    def _auto_register_shapes(self) -> None:
        """Derive per-expert (proj, comp) → shape from manifest fields.

        Convention (matches in-process MoEExpertTier):
          gate_proj / up_proj per-expert weight: (moe_intermediate, hidden)
          down_proj      per-expert weight:      (hidden, moe_intermediate)

        For quantized layouts, MLX stores `weight` as (out, in_packed) uint32
        where in_packed = in * bits / 32 elements. `scales`/`biases` are
        (out, in / group_size) in bf16/f16.
        """
        H = int(self.hidden_size)
        I = int(self.moe_intermediate)
        if self.quantized:
            bits = int(self.bits)
            gs = int(self.group_size) if self.group_size > 0 else 1
            in_packed_gate_up = (H * bits) // 32   # in_dim = hidden, packed
            in_packed_down = (I * bits) // 32       # in_dim = intermediate, packed
            sb_cols_gate_up = H // gs
            sb_cols_down = I // gs
            self._tensor_shapes[('gate_proj', 'weight')] = (I, in_packed_gate_up)
            self._tensor_shapes[('up_proj',   'weight')] = (I, in_packed_gate_up)
            self._tensor_shapes[('down_proj', 'weight')] = (H, in_packed_down)
            self._tensor_shapes[('gate_proj', 'scales')] = (I, sb_cols_gate_up)
            self._tensor_shapes[('up_proj',   'scales')] = (I, sb_cols_gate_up)
            self._tensor_shapes[('down_proj', 'scales')] = (H, sb_cols_down)
            self._tensor_shapes[('gate_proj', 'biases')] = (I, sb_cols_gate_up)
            self._tensor_shapes[('up_proj',   'biases')] = (I, sb_cols_gate_up)
            self._tensor_shapes[('down_proj', 'biases')] = (H, sb_cols_down)
        else:
            self._tensor_shapes[('gate_proj', 'weight')] = (I, H)
            self._tensor_shapes[('up_proj',   'weight')] = (I, H)
            self._tensor_shapes[('down_proj', 'weight')] = (H, I)

    # ─── Public API (matches in-process MoEExpertTier) ────────────────────────

    def info(self) -> dict:
        """MOE.EXPERT.INFO <model_id> → manifest dict."""
        import json
        body = self._send_resp('MOE.EXPERT.INFO', self.model_id)
        if not body:
            raise RuntimeError('MOE.EXPERT.INFO returned empty body')
        return json.loads(body.decode('utf-8'))

    def stats(self) -> dict:
        """MOE.EXPERT.STATS → tier-wide counters (server's view)."""
        import json
        body = self._send_resp('MOE.EXPERT.STATS')
        # Merge server counters into local hits/misses for consumer-side telemetry
        d = json.loads(body.decode('utf-8'))
        return d

    def routing_summary(self, top_n: int = 16) -> dict:
        """Routing histogram from THIS CLIENT's recorded accesses (not the
        server's — server doesn't track per-consumer routing). Same data
        shape as `pion_moe_tier.MoEExpertTier.routing_summary`.
        """
        import collections as _c
        by_layer: Dict[int, list] = _c.defaultdict(list)
        for (lid, eid), n in self.access_counts.items():
            by_layer[lid].append((eid, n))
        out = {}
        for lid in sorted(by_layer):
            entries = sorted(by_layer[lid], key=lambda kv: -kv[1])
            total = sum(n for _, n in entries)
            top = [{'expert_id': e, 'count': n, 'fraction': n / total if total > 0 else 0.0}
                   for e, n in entries[:top_n]]
            out[lid] = {'total_accesses': total, 'unique_experts': len(entries),
                        'top_experts': top}
        return out

    def set_active_ns(self, ns: str | None):
        """Set the per-distribution namespace (gh #61) tagged onto every
        subsequent FETCH. Pass None to revert to the default (ns 0) band."""
        self.active_ns = ns

    def fetch_expert(self, lid: int, eid: int) -> Dict[Tuple[str, str], mx.array]:
        """MOE.EXPERT.FETCH <model> <layer> <expert> → dict of (proj, comp) tensors.

        Parses the multi-blob response format:
          [u8 n_blobs] per blob: [u8 proj][u8 comp][u32 LE data_len][data]
        """
        self.access_counts[(lid, eid)] += 1
        # gh #61: tag the FETCH with the active per-distribution namespace so
        # the server banks the access count separately per traffic class.
        # `active_ns` is set by the caller (see set_active_ns); None = default.
        active_ns = getattr(self, 'active_ns', None)
        if active_ns:
            body = self._send_resp('MOE.EXPERT.FETCH', self.model_id,
                                    str(lid).encode(), str(eid).encode(),
                                    b'NS', active_ns.encode())
        else:
            body = self._send_resp('MOE.EXPERT.FETCH', self.model_id,
                                    str(lid).encode(), str(eid).encode())
        if not body:
            raise RuntimeError(f'MOE.EXPERT.FETCH returned empty body for ({lid}, {eid})')

        proj_names = ('gate_proj', 'up_proj', 'down_proj')
        comp_names = ('weight', 'scales', 'biases')
        out: Dict[Tuple[str, str], mx.array] = {}
        off = 0
        n_blobs = body[off]; off += 1
        for _ in range(n_blobs):
            proj_id = body[off]; off += 1
            comp_id = body[off]; off += 1
            data_len = struct.unpack('<I', body[off:off+4])[0]; off += 4
            payload = body[off:off+data_len]; off += data_len
            proj = proj_names[proj_id]
            comp = comp_names[comp_id]
            arr = self._reconstruct_array(proj, comp, payload, lid)
            out[(proj, comp)] = arr
        self.misses += 1   # mirror in-process tier semantics: every consumer call counts
        return out

    def _reconstruct_array(self, proj: str, comp: str, payload: bytes, lid: int) -> mx.array:
        """Reconstruct an mx.array from raw bytes. Shape derivation uses the
        manifest (num_experts, num_layers) + observed per-expert byte size.

        For the canonical Gemma 4 / Phi-3.5 stacked layout, per-expert
        weight shapes are derived from the model's hidden_size + moe_intermediate
        once known. Until then, the array is returned as a 1-D uint8/uint16
        buffer that the consumer reshapes downstream — sufficient for the
        first end-to-end demo.

        For Gemma 4 bf16 weight: returns shape (intermediate, hidden) bf16
        via numpy uint16 buffer reinterpreted as bf16.
        """
        # First-pass: infer dtype from comp + bits
        if comp == 'weight':
            if self.quantized:
                arr_np = np.frombuffer(payload, dtype=np.uint32)
                arr = mx.array(arr_np)
            else:
                # bf16: numpy has no native; load as uint16 then view as bf16
                arr_np = np.frombuffer(payload, dtype=np.uint16)
                arr = mx.array(arr_np).view(mx.bfloat16)
        else:
            # scales / biases — arch-specific dtype. OLMoE/Phi-3.5 4-bit quant
            # uses fp16; Gemma 4 quantized variants (if any) would use bf16.
            # See __init__ for the policy.
            if self._scales_dtype == 'fp16':
                arr_np = np.frombuffer(payload, dtype=np.float16)
                arr = mx.array(arr_np)
            else:
                arr_np = np.frombuffer(payload, dtype=np.uint16)
                arr = mx.array(arr_np).view(mx.bfloat16)

        # Reshape — needs hidden_size + moe_intermediate from the model. The
        # consumer-side substrate (install_moe_substrate intercept) reshapes
        # via `weight.T @ x` operations which work on flat shapes too, but
        # quantized_matmul / matmul typically expect the right shape.
        # Cache the derived per-expert shape on first observation.
        shape_key = (proj, comp)
        if shape_key in self._tensor_shapes:
            try:
                arr = arr.reshape(self._tensor_shapes[shape_key])
            except Exception:
                pass
        else:
            # Best-effort: for Gemma 4 bf16 weight tensors we know:
            #   gate_proj.weight, up_proj.weight: (moe_intermediate, hidden)
            #   down_proj.weight: (hidden, moe_intermediate)
            # We don't know intermediate/hidden from server INFO yet (no
            # field exposed). Caller can attach _tensor_shapes externally
            # if reshape is needed.
            pass
        return arr

    def prefetch(self, lid: int, eids: list) -> None:
        """MOE.EXPERT.PREFETCH — fire-and-forget hint.

        Chunk to fit Pion's 64-token RESP-array parser cap. Each call carries
        `MOE.EXPERT.PREFETCH model_id layer_id` (3 tokens) + N expert IDs;
        cap N at 50 to leave headroom.
        """
        if not eids:
            return
        CHUNK = 50
        for i in range(0, len(eids), CHUNK):
            batch = eids[i:i + CHUNK]
            args = ['MOE.EXPERT.PREFETCH', self.model_id, str(lid)]
            for e in batch:
                args.append(str(e))
            try:
                _ = self._send_resp(*args)
            except RuntimeError:
                pass   # prefetch failure is non-fatal

    def pin(self, lid: int, eid: int) -> None:
        try:
            _ = self._send_resp('MOE.EXPERT.PIN', self.model_id, str(lid), str(eid))
        except RuntimeError:
            pass

    def unpin(self, lid: int, eid: int) -> None:
        try:
            _ = self._send_resp('MOE.EXPERT.UNPIN', self.model_id, str(lid), str(eid))
        except RuntimeError:
            pass

    def shutdown(self) -> None:
        self._close()

    # Hook for install_moe_substrate: consumer-side ops use this dict
    # convention. The Python in-process MoEExpertTier exposes the same field.
    @property
    def layout(self) -> dict:
        return {}   # unused by current install paths

    def set_tensor_shape(self, proj: str, comp: str, shape: tuple) -> None:
        """Allow caller to register the canonical per-expert shape for a
        (proj, comp), so subsequent _reconstruct_array calls reshape correctly.
        """
        self._tensor_shapes[(proj, comp)] = shape


# ─── Quick standalone test ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True, help='snapshot hash or model_id')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=1974)
    ap.add_argument('--n', type=int, default=10, help='FETCH iterations for warmup test')
    args = ap.parse_args()

    tier = PionMoEExpertTierClient(model_id=args.model_id, host=args.host, port=args.port)
    print(f'info: {tier.info()}')
    print(f'\nstats (pre-fetch): {tier.stats()}')

    # Time N cold + warm fetches
    t0 = time.time()
    for i in range(args.n):
        _ = tier.fetch_expert(0, i % tier.num_experts)
    dt_cold = time.time() - t0
    print(f'\n{args.n} fetches (mix of cold + warm): {dt_cold*1000:.1f} ms total, '
          f'{dt_cold*1000/args.n:.1f} ms avg')

    # Repeat — should now be all hits
    t1 = time.time()
    for i in range(args.n):
        _ = tier.fetch_expert(0, i % tier.num_experts)
    dt_warm = time.time() - t1
    print(f'{args.n} fetches (all warm): {dt_warm*1000:.1f} ms total, '
          f'{dt_warm*1000/args.n:.1f} ms avg')

    s = tier.stats()
    print(f'\nstats (post-fetch): {s}')
    tier.shutdown()
