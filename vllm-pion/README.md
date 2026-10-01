# vllm-pion

Python client for Pion's externalized-attention and KV-cache surface — the
`ATTEND.*` and `KV.STORE`/`KV.FETCH` commands (`--kvcache`). It lets an
inference runtime store and reuse per-layer K/V state in Pion instead of
recomputing prefill.

> **Status:** the v1 KVConnector integration is not yet merged upstream into
> vLLM. This package ships the client (`PionKVClient`, `PionAttentionClient`)
> and the serialization/RoPE-rerotation helpers; wire it into your runtime
> yourself. For the maintained Apple-Silicon path, see
> [`pion-vllm-mlx`](https://github.com/pavelhorak/pion/tree/main/pion-vllm-mlx).

## Install

```bash
pip install -e vllm-pion/            # from a Pion checkout
pip install -e 'vllm-pion/[serve]'   # + torch/transformers for the serve demos
```

The CLI runs as a module (no console script, to avoid colliding with the
`pion-serve` proxy):

```bash
python -m vllm_pion.cli --help
```

## Contents

- `vllm_pion/client.py` — RESP client for `KV.STORE` / `KV.FETCH`.
- `vllm_pion/attention_client.py` — `ATTEND.*` externalized attention.
- `vllm_pion/kv_serializer.py`, `rope_rerotation.py` — K/V (de)serialization.
- `vllm_pion/routing_policy.py`, `attention_plugin.py` — integration glue.

## License

Apache-2.0 (see `LICENSE`). Part of [Pion](https://github.com/pavelhorak/pion).
