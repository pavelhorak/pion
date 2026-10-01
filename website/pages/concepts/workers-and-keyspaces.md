# Workers and keyspaces

The single most important operational fact about Pion: **`-w N` is N
independent keyspaces**, not one keyspace served by N threads. The default is
`-w 1`, and the server refuses `-w N > 1` unless you acknowledge the semantics
with `--independent-workers`.

<!-- include-section: README.md | ### Scaling past one worker -->

## What the server does about it

<!-- include-section: doc/operations.md | ## 2b. Worker count — `-w N` is N keyspaces -->
