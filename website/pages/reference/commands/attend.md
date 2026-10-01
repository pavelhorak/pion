# `ATTEND.*` and `ATTEND.PREFIX.*`

Stage 2 of the shared KV cache: instead of shipping K/V back to the model,
Pion computes the attention over the cached prefix itself — natively on Metal
with `--metal-attention`, or through the MLX bridge — and the consumer
attaches only the suffix. This is the path behind the same-process TTFT row,
the sparse long-context selector and the `pion-exo` hook.

## Wire forms

<!-- include-section: doc/shared_kv_cache.md | ### Wire forms -->

<!-- include-section: doc/shared_kv_cache.md | #### Wire-protocol details (for clients implementing their own consumer) -->

## When to use Stage 1 vs Stage 2

<!-- include-section: doc/shared_kv_cache.md | ### When to use Stage 1 vs Stage 2 -->

## Externalized attention commands

<!-- include-section: doc/command_matrix.md | ## 17. Externalized Attention Commands (`--kvcache`) -->

## Fixed-size state cache — `STATE.*`

<!-- include-section: doc/command_matrix.md | ## 16c. Fixed-Size State Cache — `STATE.*` (`--kvcache`) -->

