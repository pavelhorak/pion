# `MOE.EXPERT.*` — expert paging

Expert weights for MoE models live in a tiered cache (per-worker RAM LRU →
SSD → network) and are served over the wire, so a model larger than physical
memory runs at cache-hit latency. Enable with `--moe-cache DIR --moe-cache-mib N`.

<!-- include-section: README.md | ### MoE Expert Paging — `MOE.EXPERT.*` (models beyond RAM) -->

## Commands

<!-- include-section: doc/command_matrix.md | ## 16b. MoE Expert Paging — `MOE.EXPERT.*` (`--moe-cache <DIR> --moe-cache-mib N`) -->

