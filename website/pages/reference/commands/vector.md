# `FT.*` and VSET — vector search

HNSW with SIMD INT8 kernels, the RediSearch-shaped `FT.*` protocol, BM25 and
hybrid retrieval, and Redis 8 vector sets (one set per key — see below). One index per server; `L2` and
`COSINE`, anything else refused at `FT.CREATE`.

## `FT.*`

<!-- include-section: doc/vector_engine.md | ## Wire Protocol — FT.* Commands -->

## BM25 and hybrid retrieval

<!-- include-section: doc/vector_engine.md | ## BM25 Full-Text Search -->

## Metadata filters

<!-- include-section: doc/vector_engine.md | ## Metadata Filters -->

## Vector sets (Redis 8 VSET commands)

One vector set per key, created by its first `VADD`, with its own dimension. Search is exact — every element is scored against the query — so there is no recall loss and nothing to tune, at a cost linear in the set's size. Scores are Redis's: `(1 + cos) / 2`, where 1 is identical. Options that only steer Redis's HNSW graph (`Q8`, `NOQUANT`, `BIN`, `M`, `EF`) are accepted and change nothing; `REDUCE`, `FILTER`, `VEMB … RAW` and `VLINKS` are refused with an error rather than ignored. Vector sets are WAL-persisted and snapshotted: they survive a restart.

<!-- include-section: doc/command_matrix.md | ## 20. VSET Commands (Redis 8 vector sets — one set per key) -->

Engine internals and benchmarks: [Vector engine](/vector_engine.md).
