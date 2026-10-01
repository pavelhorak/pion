# First five minutes

## The four Pion lines

On Apple Silicon, `PionPromptCache` sits where mlx-lm's own prompt cache does.
The first run warms it; every later run — this process or another — hits.

<!-- include-region: README.md | four-lines -->

<!-- include-region: README.md | four-lines-explained -->

<!-- include-region: README.md | pip-install -->

## The Redis wire, vector search, the semantic cache

<!-- include-region: README.md | first-five-minutes -->

## Server profiles

<!-- include-section: README.md | ### Server Profiles -->

## Where next

- [Where Pion does not help](where-it-does-not-help.md) — read this before you
  build on it.
- [Shared KV cache](/concepts/shared-kv-cache.md) — namespaces, lanes, Stage 1 vs Stage 2.
- [Command index](/reference/command-index.md) and the [CLI flags](/reference/cli-flags.md).
