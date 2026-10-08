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

A precompiled package is tied to the compiler that built it (Mojo 1.1.0 here).

## Packaging for the modular-community channel

`packages/<name>/recipe.yaml` is a draft rattler-build recipe in the
channel's format: the source is a Pion release tag, the build assembles the
package with `build.py --assemble-only` and precompiles it into
`$PREFIX/lib/mojo/<name>.mojoc`, and the test runs the package's tests against
the installed `.mojoc` (`mojo run -I $PREFIX/lib/mojo …`). None has been
submitted yet; a submission needs a Pion tag that contains `packages/`.

To check a recipe locally before a tag exists, copy it, replace its `source`
with `- path: <your pion checkout>` plus `use_gitignore: true`, and run:

```bash
pixi exec --spec "rattler-build>=0.30.0,<0.31" rattler-build build \
  -c conda-forge -c https://conda.modular.com/max -c https://prefix.dev/modular-community \
  --recipe <copy>/recipe.yaml --output-dir out
```

All three built and passed their tests that way on macOS arm64 (2026-10-08).
On Linux x86-64 and arm64, the gate tier's `tests/test_mojo_packages.py` ran
the same assemble, precompile and test steps green on GitHub's runners the same
day; rattler-build itself was not run there.

## Licence

Apache-2.0, like the rest of Pion's `src/`.
