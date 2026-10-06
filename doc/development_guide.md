# Development Guide: Building & Contributing

**Pion** is a unified, shared-nothing database engine built in Mojo. This guide provides the necessary information for setting up, building, and contributing to the project.

## Prerequisites

Pion is built using the [Mojo](https://modular.com/mojo) programming language and the [Pixi](https://pixi.sh) package manager.

- **Mojo 1.1**: pinned exactly (`==1.1.0`) in `pixi.toml`. Performance baselines depend on the compiler, so the toolchain must not float.
- **Pixi**: Used for dependency management and environment isolation.

## Project Structure

The codebase is organized into logical components:

- `src/main.mojo`: Entry point — arg parsing, shared listen socket, worker spawning via pthreads (`worker_spawn_wrap.c` → `@export pion_worker_entry`).
- `src/engine/state.mojo`: `Pion` struct — dependency injector and state container per worker.
- `src/network/`:
  - `engine.mojo` — `NetworkEngine`: kqueue event loop, recv buffer management
  - `fast_path.mojo` — `FastPathHandler`: zero-alloc dispatch for 28 hot commands
  - `slow_path.mojo` — `SlowPathHandler`: RESP3 fallback, all `FT.*` commands
  - `response_writer.mojo` — `ResponseWriter`: RESP formatting + socket send
  - `dispatcher.mojo`, `resp3.mojo`, `server.mojo`, `raft.mojo`, `ai_gateway.mojo`
- `src/common/`: Data structures (`SlabHashMap`, `SlabList`, `SlabSkipList`, `GenericValue`, `HLL`, `bitmap`, `geohash`) and utilities.
- `src/io/`: Write-Ahead Log (`wal.mojo`), `io_uring` wrappers (`io_uring.mojo`).
- `src/memory/`: `SlabAllocator[T]`, `ObjectPool[T]`.
- `src/vector/`: `HNSWGraph` (`hnsw.mojo`), SIMD/JIT distance kernels (`kernels.mojo`).
- `tests/`: Protocol parity test (`test_parity.py`), HNSW unit tests, micro-benchmarks.

## Building and Running

Two build profiles. Use `build-dev` for iteration; reserve `build` for benchmarks, gates, and pushes.

| task | flags | time | binary | when to use |
|---|---|---:|---|---|
| `pixi run build` | `-O3` (default), full LLVM optimization, monomorphization, SIMD codegen | minutes | `pion-server` | benchmarks, the perf gate, before push |
| `pixi run build-dev` | `-O0`, no optimization, no DCE | seconds | `pion-server-dev` | "does it compile + raw tests pass" iteration |

`build-dev` compiles in a fraction of the time, but the binary skips vectorization, inlining, loop unrolling, and SIMD codegen — never benchmark a `pion-server-dev` build, the perf numbers depend on `-O3`. The dev binary passes the correctness tests (`test_raw.py`), so it's safe for "did my change break anything?" loops; for the actual perf gates run a release build.

The dev path links `src/ffi/dev_macos_stubs.c` (no-op stubs for Linux-only `__errno_location` + `epoll_*`) so `-O0` doesn't fail to link from un-DCE'd Linux paths. `xdp_wrap.c` is also compiled on macOS now (its `#else /* !__linux__ */` block emits matching stubs); both stubs are linked into the release build too but get DCE'd at `-O3` — same final behavior.

### Building (macOS)
```bash
pixi run build           # release
pixi run build-dev       # dev iteration (-O0, compiles fast, runs slow)
```

### Building (Linux)
```bash
# Build the io_uring C wrapper first, then the Mojo binary
gcc -c src/ffi/uring_wrap.c -o src/ffi/uring_wrap.o
pixi run build
# `build-dev` is currently macOS-only; the Linux build path doesn't
# need the macOS stub shims (epoll lives in glibc; xdp_wrap.c is
# already compiled), but a parallel Linux dev task hasn't been wired
# up yet — open an issue if you want one.
```

### Starting the Server
```bash
./pion-server                           # port 1974, 1 worker (-w N>1 needs --independent-workers)
./pion-server -p 6379 -w 10 --independent-workers   # 10 INDEPENDENT keyspaces — for VectorDBBench
./pion-server -p 6379 -w 1             # single worker (for debugging)
./pion-server --flare                   # enable AI features (auto-detects Ollama)
./pion-server --cluster                 # enable cluster mode (GLIDE-compatible)
./pion-server --emb-enabled --llm-enabled  # manually enable embedding + LLM
```

### Running Tests
```bash
# Protocol parity test (Redis compatibility)
python3 tests/test_parity.py

# HNSW unit test
mojo run -I . tests/test_hnsw.mojo

# Slab allocator test
mojo run -I . tests/test_slab_allocator.mojo
```

## Coding Standards

When contributing to Pion, please adhere to these core mandates:

1.  **Explicit Memory**: ALWAYS use `SlabAllocator`. NEVER use standard collections that perform hidden `malloc` calls in the hot path.
2.  **Pointer Rigor**: All `UnsafePointer` usage must include explicit initialization (`unsafe_write`) and destruction.
3.  **Non-Blocking I/O**: Use the `TCPServer` or `IORing` to ensure all operations are asynchronous.
4.  **Zero-Copy Parsing**: Avoid `memcpy` in protocol and data parsers. Index directly into raw buffers.
5.  **Ownership**: Use the Mojo move operator `^` rigorously to transfer non-copyable resources.

## Performance Profiling & Benchmarking

**Mandatory**: You must run the benchmark suite before submitting a Pull Request to ensure no performance regressions have been introduced.

### Running Benchmarks

```bash
# KV benchmark — Pion only (pipeline=10, 50 clients, 100K requests)
python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only

# Vector benchmark — Pion only (Performance1536D50K)
python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10

# Protocol parity test
python3 tests/test_parity.py
```

The per-command floors are in `benchmarks/gate_baselines.json` (Mac and Linux
profiles differ); pass `--gate` to either benchmark script for a pass/fail
verdict. Floors are never lowered to make a run pass.
