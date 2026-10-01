# libpion_vector — the closed vector library

`pixi run build` links `vendor/pion-vector/<platform>/libpion_vector.a`: the
tuned 1536-dim beam searches (INT8, and the PolarQuant / TurboQuant /
NanoQuant variants) and the tuned product-key-memory kernels. Everything else
in Pion — the KV engine, graph build, persistence, every other dimension, the
quantizers, the Metal shaders — is open source in this repository.

**Nothing here is needed to build or run Pion.** `pixi run build-open` builds
the same engine against the open reference implementations in
`src/vector/reference/`: same algorithms with the tuning removed, bit-identical
results, slower vector search (numbers: `doc/vector_engine.md` § Open build).
`pion-server --version` and `INFO` (`pion_vector:`) say which one a binary uses.

Per platform directory:

| File | What |
|---|---|
| `libpion_vector.a` | static archive; C-ABI exports `pion_v_*` only, local symbols stripped |
| `MANIFEST` | ABI version, target CPU, source fingerprint, exported and undefined symbols |
| `SHA256SUMS` | checksum of the archive |

The interface is open: `src/vector/vector_abi.mojo` (the calls),
`src/vector/beam_view.mojo` and `src/vector/quant_beam_view.mojo` (the
argument structs; field order is ABI). The engine refuses to start if the
library reports a different ABI version than it was built for.

Every vendor bump is checked by `pixi run test-vector-differential`, which runs
each closed routine and its open reference on the same inputs and requires
identical results.

## Licence

The library is distributed under the **Pion Vector Binary Licence**
([`LICENSE`](LICENSE) in this directory), not under the repository's
Apache-2.0 licence. In summary — the text is the licence, this is not: free
for any use including commercial, as part of Pion (modified or not);
redistribution only unmodified and together with Pion; may not be used to
offer Pion as a managed database, key-value, vector-search or KV-cache
service; no warranty; no decompilation except as permitted by applicable law.
A Pion built with `pixi run build-open` contains none of this code and is
Apache-2.0 alone.

Platforms vendored today: `macos-arm64`. Linux builds use the open reference
until the Linux archives are vendored and linked.
