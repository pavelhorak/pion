# pion-exo

Pion attention hook for [exo](https://github.com/exo-explore/exo) distributed inference — V offloading and Stage-2 Metal SDPA on Apple Silicon clusters.

Two operating modes:

| Mode | What it does | When to use |
|---|---|---|
| `v_offload` | Offloads the V tensor of every attention layer to Pion's V-store (INT8 / turbo4 / fp16). Frees Mac unified memory for activations + KVQ. | Long-context generation on a single Mac, or 4-tier shared corpora across a small cluster. |
| `gpu_attention` | All of `v_offload`, plus the warm forward runs Metal SDPA directly via [pion-vllm-mlx](../pion-vllm-mlx/)'s `PionPromptCache` Stage-2 path. | Production cluster serving with cross-host prefix sharing and the 326×-warm-TTFT (Gemma-4-E2B 64K NIAH) substrate. |

## Install

```bash
pip install -e pion-exo/
# optional: gpu_attention mode pulls pion-vllm-mlx
pip install -e ../pion-vllm-mlx/
```

Requires a running Pion server with V-store + KV-prefix enabled:

```bash
./pion-server --kvcache --metal-attention -w 1
```

For Mac-cluster topologies (Stage-3 `peers=[...]`) every node must run its own Pion server on port 1974.

## Usage

```python
from pion_exo import PionAttentionHook

# Single-host V offload — drops V cache to Pion, keeps K + activations in Mac RAM.
hook = PionAttentionHook(pion_host="127.0.0.1", pion_port=1974, mode="v_offload")

# Mac cluster — sticky-route sessions across nodes by blake2b(session_id) mod N.
hook = PionAttentionHook(
    peers=[("mac-mini-1", 1974), ("mac-mini-2", 1974), ("mac-mini-3", 1974)],
    mode="gpu_attention",
)
```

The hook is driven directly — `on_prefill` per layer during prefill, then
`on_decode_attention` per layer per step:

```python
hook.on_prefill(session_id, layer_id, K=K, V=V)          # push K/V to Pion
out = hook.on_decode_attention(session_id, layer_id,      # attention output
                               Q=Q, K_local=K, top_k=64)
hook.drop_session(session_id)                             # release when done
```

### Wiring it into exo

**exo does not currently expose an attention-hook API** — there is no
`register_attention_hook` or equivalent extension point upstream (checked
2026-08-21 against `exo-explore/exo`). Two routes, neither of which lives in
this package yet:

1. **Monkey-patch exo's attention path** from outside, the way
   [`pion-vllm-mlx`](../pion-vllm-mlx/) does for mlx-lm in
   `mlx_lm_patch.py`. Same shape: intercept the attention call, delegate to
   the hook. This needs no upstream change.
2. **Land a hook point upstream.** Cleaner and durable, but it is a PR to
   someone else's project on their timeline.

Until one of those exists, the hook is usable — and tested — by a caller that
drives it directly, which is what the tests in [`tests/`](tests/) do.

## Docs

- Main project: [`../README.md`](../README.md)
- Stage-2 lane design: [`../doc/shared_kv_cache.md`](../doc/shared_kv_cache.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) in this directory. Pion's client packages are
deliberately permissive so they can be vendored into any stack; the Pion **server**
is Apache-2.0 too, with one closed binary library for its tuned vector kernels —
see the top-level [`LICENSE`](../LICENSE).

Which parts of Pion are under which licence: [`doc/licensing.md`](../doc/licensing.md).
