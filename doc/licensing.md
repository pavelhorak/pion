# Licensing — what is under which licence, and what is not here at all

Pion is open source under **Apache-2.0** — the engine, the clients, the tests,
the tools and the docs. One component is not: **`libpion_vector`**, the tuned
1536-dim vector search kernels, which ships as a free, closed binary library
under its own licence. That is open core, and this page says exactly where the
line is. The legal texts are [`LICENSE`](../LICENSE) at the repository root,
[`vendor/pion-vector/LICENSE`](../vendor/pion-vector/LICENSE) for the library,
and the `LICENSE` file inside each client package; where this page and those
files disagree, the files win.

## The map

| What | Licence | Where the text is |
|---|---|---|
| **The engine** — everything under `src/`, including the open reference vector kernels in `src/vector/reference/` | **Apache-2.0** | [`LICENSE`](../LICENSE) |
| Everything else not listed below — `tests/`, `tools/`, `benchmarks/`, `scripts/`, `docker/`, `examples/`, `doc/`, `pion-serve/`, `flare_gateway/`, `vllm-pion/`, `pion_memory.py` | Apache-2.0 | [`LICENSE`](../LICENSE) |
| **`libpion_vector`** — the object code under `vendor/pion-vector/`: tuned 1536-dim beam searches (INT8, PolarQuant, TurboQuant, NanoQuant) and product-key-memory kernels | **Pion Vector Binary Licence** — free, closed | [`vendor/pion-vector/LICENSE`](../vendor/pion-vector/LICENSE) |
| **Release tarballs and the Docker image**, built with `pixi run build` (`build-portable` on Linux x86-64), which link `libpion_vector` on macOS arm64, Linux x86-64 and Linux arm64 | Apache-2.0 for Pion, plus the binary licence for the library inside it; both texts ship with the binary (`LICENSE`, `LICENSE-pion-vector`) | both of the above |
| Any build made with `pixi run build-open`, and the Linux tarballs and images of v0.9.2 and earlier, which predate the Linux archives | Apache-2.0 only — no closed code is linked | [`LICENSE`](../LICENSE) |
| **`pion-vllm-mlx`** — the mlx-lm prompt cache and attention patch (PyPI: `pion-vllm-mlx`) | Apache-2.0 | [`pion-vllm-mlx/LICENSE`](../pion-vllm-mlx/LICENSE) |
| **`pion-mcp`** (`mcp/`) — the MCP agent-memory server | Apache-2.0 | [`mcp/LICENSE`](../mcp/LICENSE) |
| **`pion-lmcache`** — the LMCache connector | Apache-2.0 | [`pion-lmcache/LICENSE`](../pion-lmcache/LICENSE) |
| **`pion-context`** (`pion_context/`) — the semantic codebase search client | Apache-2.0 | [`pion_context/LICENSE`](../pion_context/LICENSE) |
| **`pion-glide`** (`pion_glide/`) — typed wrappers over Valkey GLIDE | Apache-2.0 | [`pion_glide/LICENSE`](../pion_glide/LICENSE) |
| **`pion-langgraph`** — LangGraph checkpoint saver | Apache-2.0 | [`pion-langgraph/LICENSE`](../pion-langgraph/LICENSE) |
| **`pion-autogen`** — AutoGen memory store | Apache-2.0 | [`pion-autogen/LICENSE`](../pion-autogen/LICENSE) |
| **`pion-llamaindex`** — LlamaIndex vector store | Apache-2.0 | [`pion-llamaindex/LICENSE`](../pion-llamaindex/LICENSE) |
| **`pion-exo`** — exo attention hook | Apache-2.0 | [`pion-exo/LICENSE`](../pion-exo/LICENSE) |
| Vendored third-party code under `src/ffi/lua/`: **Lua 5.1.5** (patched for read-only tables, as Redis does), the **lua-cjson** sources (`fpconv.c`, `strbuf.c`, `lua_cjson.c`), **lua-cmsgpack** (`lua_cmsgpack.c`), the Lua **struct** library (`lua_struct.c`) and **LuaBitOp** (`lua_bit.c`) | MIT, their own | headers in those files; [`NOTICE`](../NOTICE) |

Each client package carries the full Apache-2.0 text inside its own
directory, and therefore inside its sdist and wheel — a build from that
directory cannot reach the repository root, and Apache-2.0 requires the text
to travel with the code.

## Why one piece is closed

Pion is written by one developer, and the tuned vector kernels are the work
that cannot be given away yet. Everything you need to *trust* Pion is open:

- **The algorithms are open.** `src/vector/reference/` implements the same
  traversal, heaps and distance formulas with the tuning removed.
- **Equivalence is checked, not claimed.** `pixi run test-vector-differential`
  runs every closed routine and its open reference on the same inputs and
  requires identical results; it runs before every release is packaged and
  in the gate. The interface (`src/vector/vector_abi.mojo` and the view
  structs) is open, and the engine refuses to start on an ABI mismatch.
- **You can build without it.** `pixi run build-open` builds Pion entirely
  from source. The cost is vector search speed at 1536 dims only — 22–34%
  lower QPS depending on the quantization mode, measured and published in
  [`vector_engine.md` § Open build](vector_engine.md#open-build-and-the-closed-vector-library).
  The KV engine, persistence, the prompt cache, every quantizer, every other
  vector dimension and the Metal shaders are the same code in both builds.
- **Which one you are running is reported.** `pion-server --version` and
  `INFO` (`pion_vector:`) name the backend a binary links.

When a successor replaces a generation of the tuned kernels, the superseded
generation's source is published under Apache-2.0. That is a stated
intention of the project, not a term of either licence.

## The binary licence in plain words

**You may:** use Pion with the library for anything, including commercially
and in production; embed it in a product you sell; run it on your own
machines or in your own cloud account; run it inside an application you
operate for your customers; link the library into a version of Pion you have
modified; and redistribute it unmodified together with Pion.

**You may not:** offer Pion itself to third parties as a hosted or managed
**database, key-value store, vector-search or KV-cache service** using the
library; modify the library; or decompile it, except where the law allows
that regardless of the licence (in the EU, Directive 2009/24/EC).

A hosting provider that wants to offer Pion as a service has two options: run
the Apache-2.0 open build, which carries no restriction at all, or ask the
licensor for a managed-service licence to the library.

**Trademarks.** Neither licence grants trademark rights. The name "Pion" and
the logo identify this project; a fork should pick its own name.

**Contributions.** Contributions are accepted under the Contributor Licence
Agreement in [`CLA.md`](../CLA.md). It keeps copyright in the project in one
place, which is what makes any future relicensing possible without chasing
every contributor. You keep the copyright in what you wrote.

## What is not in this repository

- **The tuned source of `libpion_vector`.** The library ships as object code
  under its own licence (above); the same algorithms are open in
  `src/vector/reference/`, and the differential proves the two agree.
- **Research, planning and design notes.** The documentation here is
  reference: what the server does and how to run it.

Nothing withheld is needed to build, run or test what is published.

## Questions this page should answer

- *Is Pion open source?* The engine and everything around it is, under
  Apache-2.0. The tuned vector kernels are a free, closed binary library.
  Calling the whole thing open source without that qualifier would be wrong;
  calling it open core is right.
- *Can I run Pion in production without paying?* Yes, with or without the
  library.
- *Can I ship Pion inside my product?* Yes, including commercially.
- *Can I offer "hosted Pion" to my customers?* Yes with the open build
  (`pixi run build-open`), which is Apache-2.0 alone. With the library, only
  under a managed-service licence from the licensor.
- *Can I fork Pion?* Yes — it is Apache-2.0. Your fork may link the library
  unmodified under its licence, or use the open reference.
- *Can I vendor `pion-vllm-mlx` (or any other client package: `pion-mcp`,
  `pion-lmcache`, `pion-context`, `pion-glide`, `pion-langgraph`,
  `pion-autogen`, `pion-llamaindex`, `pion-exo`) into an Apache/MIT project?*
  Yes — they are Apache-2.0 and self-contained, and none of them contains the
  library.
- *Is the Docker image under the same licence as the binary?* Yes, from
  v0.9.3 on: its binary links `libpion_vector` exactly as a release tarball
  does, and both licence texts are in the image at `/opt/pion/LICENSE` and
  `/opt/pion/LICENSE-pion-vector`. Images up to v0.9.2 predate the Linux
  archives and are Apache-2.0 only. For a binary with no closed code, run
  `pixi run build-open`; it exists on macOS and on both Linux targets.
