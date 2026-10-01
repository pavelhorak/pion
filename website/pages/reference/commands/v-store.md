# `V.*` — the V-store

Token-ID-indexed storage for K and V tensors, one session per `<ns>_pk` /
`<ns>_pv` pair created by `KV.PREFIX.REGISTER`. The wire forms of
`V.STOREBATCH` and `V.FETCH … RANGE` / `BATCH` are in the
[M14 externalized-attention table](attend.md#m14-externalized-attention-commands)
and the [`KV.PREFIX.*` page](kv-prefix.md); this page covers the storage
formats.

## Quantization tiers

<!-- include-section: doc/shared_kv_cache.md | ## Quantization Formats -->

## The `mlx4g32` tier

<!-- include-section: doc/shared_kv_cache.md | ## Quantized tier — `mlx4g32` | shift:1 -->

## Heterogeneous K/V (per-layer formats)

<!-- include-section: doc/shared_kv_cache.md | ### Heterogeneous KV cache -->

