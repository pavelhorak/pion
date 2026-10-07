# `AI.*` — semantic cache, embeddings, routing

`AI.SEMANTIC_CACHE` and `AI.EMBED` are part of Pion's supported surface. The
rest of the `AI.*` family (`AI.COMPLETE`, `AI.CHAT`, `AI.ROUTE.*`,
`AI.FLARE.*`, `AI.KNN_LM.*`, `AI.MEMORY`) is **experimental**: wired and
wire-tested, with no published measurement of what it buys and no user outside
this project, so it may change or be removed
([README](https://github.com/pavelhorak/pion#experimental)).

<!-- include-section: doc/command_matrix.md | ## 16. Pion-Native AI Commands (no Redis / Valkey equivalent) -->

## The gateway commands

<!-- include-section: doc/ai_gateway.md | ## Implemented Commands -->

## `AI.SEMANTIC_CACHE`

<!-- include-section: doc/ai_gateway.md | ## Related: AI.SEMANTIC_CACHE -->

## `AI.ROUTE.*` — semantic load balancer

<!-- include-section: doc/command_matrix.md | ## 18. Semantic Router Commands (`--kvcache`) -->

Embedding backends and how the first request just works: [Embeddings](/embeddings.md).
