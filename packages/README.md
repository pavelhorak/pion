# Mojo packages from Pion

Three parts of Pion's server that stand on their own, packaged for other Mojo
programs. Each is built from the files under `src/` that Pion itself compiles,
so a package and the server can never disagree: there is no second copy.

| Package | What it is | Built from |
|---|---|---|
| `pion_resp` | A zero-copy RESP2/RESP3 **request** parser: pipelined commands in one buffer, inline commands (quotes and escapes included), partial frames left for the next read, and Redis's protocol errors with their codes. Tokens point into your buffer. | `src/network/resp3.mojo` |
| `pion_slab` | An mmap-backed slab allocator with O(1) allocate and free, and a fixed-capacity object pool. | `src/memory/slab_allocator.mojo`, `src/memory/object_pool.mojo` |
| `pion_simd` | Vector-distance kernels: FP32 dot product, INT8 and INT4 squared L2, Hamming, FP32-to-INT8 quantization, with ARM NEON SDOT and x86 VNNI paths. These are the open kernels in `src/vector/`, not the closed tuned beams. | `src/vector/kernels.mojo`, `src/vector/fma_mad.mojo` |

Each package also carries `src/common/ptr.mojo` where its sources use it.

## Build and test

```bash
pixi install                       # once: the Mojo toolchain Pion uses
python3 packages/build.py          # assemble, precompile and test all three
python3 packages/build.py pion_resp
```

`build.py` copies a package's files into `build/mojo-packages/src/<name>/`,
rewrites Pion's absolute imports (`from src.common.ptr import ...`) to
package-relative ones, writes an `__init__.mojo` from the manifest's exports,
runs `mojo precompile`, and runs every `packages/<name>/tests/test_*.mojo`
against the result. A source that imports a `src/` module its manifest does
not list fails the build. `tests/test_mojo_packages.py` runs the same build in
Pion's gate tier, so a change to `src/` that breaks a package fails there.

To use one from your own project, precompile it and point `mojo` at it:

```bash
python3 packages/build.py pion_resp --no-tests    # -> build/mojo-packages/pion_resp.mojoc
mojo run -I build/mojo-packages your_program.mojo
```

```mojo
from pion_resp import RESP3Parser, RESP3Token
```

A precompiled package is tied to the compiler that built it (Mojo 1.2.0 here).

## Install from the modular-community channel

All four are on Modular's community channel for macOS arm64, Linux x86-64 and
Linux arm64, from Pion 1.1.0:

```bash
pixi init my-project -c conda-forge -c https://conda.modular.com/max \
  -c https://prefix.dev/modular-community
cd my-project
pixi add pion-resp        # or pion-slab, pion-simd
pixi add pion             # the server: pixi run pion-server
```

A library brings the Mojo compiler it was built with (1.2.0) and installs as
`$PREFIX/lib/mojo/<name>.mojoc`, where `pixi run mojo` finds it without an
`-I`. The `pion` package is the server, built from source with the open
reference vector kernels, because the channel carries no closed binaries:
`pion-server --version` prints `vector: reference (open)` and `INFO` reports
`pion_vector:reference (open)`. Vector search at 1536 dims is slower with those
kernels than in a release tarball or the Homebrew formula, by the amounts the
[top-level README](../README.md) gives; the rest of the server is the same code.

Checked on 2026-11-09, in fresh pixi projects on macOS arm64: `pion` 1.1.0
installed, reported `1.1.0+8221878` and answered SET and GET, and a program
importing `pion_resp` ran under `pixi run mojo run` with no `-I`. On Linux the
check is the channel's CI, which built and tested all four on x86-64 and arm64
(and macOS arm64) before they were published.

## The recipes

`packages/<name>/recipe.yaml` (and `packages/pion/build.sh` and
`test_server.sh`) are copies of what the channel builds, in
[modular/modular-community](https://github.com/modular/modular-community/tree/main/recipes):
recipes `pion`, `pion-resp`, `pion-simd` and `pion-slab`, added by
modular/modular-community#450 and modular/modular-community#451. The copies
there are the ones that build; these are kept identical so the next version
starts from them.

- **The libraries** take their source from a Pion release tag, assemble the
  package with `build.py --assemble-only`, precompile it into
  `$PREFIX/lib/mojo/<name>.mojoc`, and test it with the package's own tests
  against the installed `.mojoc` (`mojo run -I $PREFIX/lib/mojo …`).
- **The server** takes the release's tag archive (`commit` and `sha256` pinned
  in the recipe) and builds `pion-server` with the open kernels.
  `test_server.sh` checks that the version matches and that no closed library
  is linked, starts the server, and runs the wire basics (`tests/test_raw.py`)
  and a 1536-dim recall check (`tests/test_vector_recall_smoke.py`).

To move them to a new release: set `version` in all four (and `commit` and
`sha256` in `pion`), check them locally, and open one PR upstream with the
four recipe changes. To check a recipe locally, copy it, point its `source` at
your checkout (`- path: <your pion checkout>` with `use_gitignore: true`)
if the tag does not exist yet, and run:

```bash
pixi exec --spec "rattler-build>=0.30.0,<0.31" rattler-build build \
  -c conda-forge -c https://conda.modular.com/max -c https://prefix.dev/modular-community \
  --recipe <copy>/recipe.yaml --output-dir out
```

The channel's CI runs only after a maintainer labels the PR "OK to test".

## Licence

Apache-2.0, like the rest of Pion's `src/`.
