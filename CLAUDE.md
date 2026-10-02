# Pion: notes for coding agents

Pion is a Redis-compatible (RESP2/RESP3) key-value and vector engine written in
Mojo. Its main job is memory for AI inference: a prompt's K/V cache that other
processes, and a restarted server, can reuse. [README.md](README.md) says what
it does, [CONTRIBUTING.md](CONTRIBUTING.md) what a mergeable PR needs, and
[doc/index.md](doc/index.md) indexes the reference docs.

Where the code lives: `src/network/` (event loops, `fast_path.mojo`,
`slow_path.mojo`, `response_writer.mojo`), `src/commands/` (handlers by family),
`src/common/` (`GenericValue`, the Swiss-table hash map, lists),
`src/vector/` (HNSW, kernels, `reference/`), `src/io/` (WAL, snapshots).
[doc/architecture.md](doc/architecture.md) walks a request through them.

## Build and run

```bash
pixi run build            # release (-O3) -> ./pion-server; links the closed vector library
pixi run build-dev        # -O0, ~15 s -> ./pion-server-dev; never benchmark it
pixi run build-open       # everything from source, with the open reference vector kernels
pixi run build-portable   # Linux without a GPU: x86-64-v2 -> ./pion-server-dev
./pion-server             # port 1974, one worker
./pion-server --kvcache --metal-attention -w 1    # the prompt cache (KV.PREFIX.*)
```

`-w N` runs N independent keyspaces: a key written on one connection is
invisible to a connection that landed on another worker. The server refuses
`-w N > 1` unless `--independent-workers` is passed.

## Tests

```bash
python3 tests/run_all.py                    # gate tier, before every push
python3 tests/run_all.py --tier full        # everything runnable on this machine
python3 tests/run_all.py --only test_raw test_parity    # a quick check
```

`tests/manifest.toml` classifies every test file (an unclassified file fails
the run) and names the server flags each one needs. The runner starts a fresh
server per test, and fails a test whose server crashed or stopped answering
even when the test itself passed. `tests/test_dispatch_sweep.py` and
`tests/test_redis_differential.py` (needs `redis-server`) cover the whole
command surface.

## Performance gate

Run it on a quiet machine whenever a change can touch the network engine, the
KV path or the vector kernels. The floors are in
[benchmarks/gate_baselines.json](benchmarks/gate_baselines.json). A floor never
moves to make a run pass.

```bash
python3 benchmarks/preflight.py
python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only --gate
python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pion-only --profiles throughput,pipeline -w 1 --gate
python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10 --gate
```

- A leaked server costs about 30% on the write rows. Check `pgrep -x pion-server`
  and `pgrep -f 'redis-serve[r]'`; Redis rewrites its process title, so
  `pgrep -x redis-server` never matches.
- LRANGE_300 and LRANGE_600 sit within a few percent of their floors, MSET
  depends on warm-up, and vector QPS is noisy. Re-run a miss, then A/B it
  interleaved against the previous build on the same machine before calling
  it a regression.
- If a change cannot plausibly alter machine code, prove it: build both sides
  and compare their `__TEXT,__text` sections.
- The vector harness needs VectorDBBench installed; see
  [doc/benchmarking_guide.md](doc/benchmarking_guide.md).

## The closed vector library

The tuned 1536-dim beam kernels ship as
`vendor/pion-vector/<platform>/libpion_vector.a`, which `pixi run build` links
with `-D PION_HELD_VECTOR`. The same algorithms are open in
`src/vector/reference/`, and `pixi run test-vector-differential` must stay
bit-exact between the two. Never edit `vendor/` by hand. A change to a view
struct or an exported signature bumps `VECTOR_ABI_VERSION`; the server refuses
to start on a mismatch. [doc/licensing.md](doc/licensing.md) has the terms.

## Hot-path rules

Each of these has produced a shipped bug.

- No allocation, `String` construction or `print()` in the event loop. Replies
  go through `ResponseWriter` into its preallocated buffer, and integers are
  formatted with `format_int_to_buf`.
- Use `GenericValue.borrow(ptr, len)` for every key lookup, and `from_ptr` only
  for a value the caller will own. Never use `from_ptr_unsafe` for a map key:
  its hash and equality differ from the stored inline form.
- A hot loop over many items gets its own `@no_inline` function, as
  `_get_burst` and `_mset_frame` do. `process_data_plane` is one large
  function, and a branch added to any arm can make another arm's loop spill
  registers.

## Dispatch and replies

- Match the whole command name: `cmd_eq` in the slow path, `cmd_matches_N` in
  the fast path.
- Every arm consumes its whole frame, error paths included: set
  `i = cmd_end_tok - 1`, and bound any optional-argument scan by `cmd_end_tok`.
  One command in, exactly one reply out.
- Check the type before writing an array header. A missing key and a
  wrong-type key get different replies, and WRONGTYPE is checked first.
- A command that refuses must not have changed anything: validate every key
  before doing any work.
- Parse numeric arguments with `parse_int64_strict` or `parse_redis_double`,
  never with a hand-rolled digit loop.
- Never hand-count a RESP literal's length. Tests parse replies; they never
  substring-match them.
- A new mutating command appends a WAL effect record, and its record type
  must be decoded by WAL replay, the snapshot loader and replication. Log the
  resolved effect (the popped member, the generated ID, the absolute
  deadline), never a random choice or a relative time.

## Mojo pitfalls

- A pointer into a stack local (`stack_allocation`, an `InlineArray`, a short
  `String`'s bytes) may cross only into an `@always_inline` Mojo function or
  into C. An out-of-line call is emitted as an LLVM `tail call`, and at -O3 the
  caller's stores disappear. `pixi run audit-tail-alloca` must report 0;
  [tools/mojo_repros/](tools/mojo_repros/) has minimal repros.
- No load may read past the end of its allocation, even within one page.
- A shift within one buffer uses `memmove`, never `unsafe_memcpy`.
- Null pointers go through `src/common/ptr.mojo` (`null_ptr`, `is_null`).
- `String.unsafe_ptr()` is not NUL-terminated: append `"\0"` before passing a
  path to C.
- Negating an `Int64` to take its magnitude breaks at the minimum value.

## Conventions

- `gh #N` in code comments refers to the tracker Pion was developed in before
  publication, not to this repository's issues (see CONTRIBUTING).
- Test both sides of every representation switch: lists change layout at
  1,024 entries, strings at 23 bytes.
- A probe for a suspected mechanism has to vary the input that mechanism
  depends on: scramble insert orders, and open cross-worker connections
  concurrently.
- A startup line reports what the server obtained, not what a flag requested.
