# `PION.STATS` and `INFO` — the value receipt

The server keeps a per-worker ledger of what the prefix cache actually did
for you, and `INFO` reports resolved values — the real port, RSS, uptime and
key counts — rather than placeholders.

<!-- include-section: doc/shared_kv_cache.md | ### The value receipt — `PION.STATS` -->

## Liveness, crash and status files

<!-- include-section: doc/operations.md | ## 1. Crash / exit diagnostics | shift:1 -->

<!-- include-section: doc/operations.md | ## 3. Client-visible liveness | shift:1 -->

Everything else an operator needs: [Running in production](/operations.md).
