# Contributing to Pion

Thanks for considering a contribution. Pion is Apache-2.0, with one closed binary library (see [License](#license) below). This file covers what you need to know to land a PR.

## What "ready to merge" looks like

Every PR that touches Mojo source or wire-protocol surface must show:

1. **Green build** — `pixi run build` exits clean (release `-O3` binary, ~4 MB on Mac, comparable on Linux).
2. **Green correctness gate** — `python3 tests/test_raw.py` (114 invariants) + `python3 tests/test_parity.py` (RESP parity vs Redis).
3. **Green perf gate when the change can affect the hot path** — KV memtier + VectorDBBench above the documented baselines. The floors are in [`benchmarks/gate_baselines.json`](benchmarks/gate_baselines.json); run via:

   ```bash
   python3 benchmarks/preflight.py    # is the machine quiet enough to measure?
   python3 benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only --gate
   python3 benchmarks/memtier-benchmark/memtier-benchmark.py --pion-only --profiles throughput,pipeline -w 1 --gate
   python3 benchmarks/VectorDBBench/vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 10 --gate
   ```

   On bare-metal Linux, add `--gate-profile linux-epyc-8124p` to the memtier and
   vector commands. The vector harness needs VectorDBBench installed; see
   [`doc/benchmarking_guide.md`](doc/benchmarking_guide.md).

   PRs that don't touch the network engine / KV path / vector kernels can skip Gates 3 + 4.

**CI runs the correctness gates; the perf gates are yours to run.**
`.github/workflows/ci.yml` builds on Linux and macOS and runs Gate 1
(`test_raw.py`), Gate 2 (`test_parity.py`) and the whole-surface sweeps on
every pull request.

**Perf gates deliberately do not run in CI and never will.** They are
position-dependent, warm-up-dependent and contention-sensitive — a hosted
runner cannot produce a number worth reading, and a green badge over a
meaningless measurement is worse than no badge. Run them yourself on a quiet
machine and paste the numbers into the PR description when you touch the
network engine, the KV path or the vector kernels. Several rows sit close enough to
their floor that ordinary run-to-run noise looks exactly like a regression:
warm the box up, and compare against the previous commit on the same machine,
not against the floor alone.

## Build prerequisites

```bash
# Toolchain — Pion pins an exact Mojo release:
pixi install                           # reads pixi.toml; installs Mojo + Python

# C FFI shims (Lua, io_uring, XDP, SSM-state, MoE warm-pool):
pixi run build-ffi

# Release binary:
pixi run build                          # ~290 s on Mac M4

# Dev binary (no optimization, ~15 s, ~28 MB):
pixi run build-dev                      # NEVER use for benchmarks
```

The Mojo version is pinned at **`==1.1.0`** (Mojo 1.1, 2026-09-17). Don't bump it in a PR: the gate thresholds are perf baselines, so a floating toolchain makes every number incomparable. See [`pixi.toml`](pixi.toml) for the rationale. A bump is its own session, with the gates re-run on both sides.

## Adding a new command

The short version:

1. Add an `execute_*` method on `CommandDispatcher` in `src/network/dispatcher.mojo`. State mutations write to WAL + Raft log here.
2. Promote to the fast path if it's a top-30-frequency command: add the dispatch branch in `src/network/fast_path.mojo::FastPathHandler.process_data_plane()` using `cmd_matches_N` byte comparisons.
3. Slow-path fallback in `src/network/slow_path.mojo::SlowPathHandler::process_slow_path()`.
4. Cover it with a Section in `tests/test_raw.py` and `tests/test_parity.py`.

**Anti-patterns** (rejected at review):

- `alloc[T]` on the event loop — use `response_buffer`, `SlabAllocator`, `ObjectPool`, or `stack_allocation`.
- `String(...)` construction in the fast path.
- `GenericValue.from_ptr_unsafe(...)` for hash-map key lookups (breaks STRING / STRING_SSO equality).
- `.upper()` or `.value()` on tokens in the fast path (allocates a heap String).
- Integer `/` or `%` for response serialisation — use `format_int_to_buf` (division-free).


## Commit + PR style

- One logical change per commit. The commit message's first line is "imperative present, ≤72 chars" (`fix(wal): ...`, `refactor(engine): ...`, `add: ...`). The body explains why, not what.
- Reference the GitHub issue in the subject line whenever there is one.
- Comments in the source cite issues as `gh #N`. Those numbers come from the
  tracker Pion was developed in before it was published, which is not public;
  they do not refer to issues in this repository. Cite this repository's
  issues as `#N`.
- Don't bypass commit hooks (`--no-verify` / `--no-gpg-sign`). If a hook fails, fix the underlying issue.
- Don't push to `main` directly. Open a PR even for trivial fixes. CI runs the correctness gates on it; for a change that can touch the hot path, the perf-gate output you paste into the description is the only performance evidence a reviewer has.

## Test conventions

- New tests go in `tests/` (Mojo + Python) or `pion-*/tests/` (per-package).
- Wire-protocol tests use `tests/test_raw.py` style (raw socket + RESP).
- Behavior parity vs Redis goes in `tests/test_parity.py` (uses `redis-py < 5.0`).
- The MoE / KV-prefix substrate has its own perf gates in `tests/bench_*.py`. If you're touching those, run the gates and quote numbers.

## License

**Apache-2.0** — see [`LICENSE`](LICENSE). Your contributions to this
repository are released under it.

The one exception is `vendor/pion-vector/`: the tuned 1536-dim vector kernels,
vendored as object code under the
[Pion Vector Binary Licence](vendor/pion-vector/LICENSE). Their source is not
in this repository and pull requests cannot change them. Their open
counterparts in `src/vector/reference/` are Apache-2.0 like everything else,
and improvements there are welcome — `pixi run test-vector-differential` must
stay at zero differences. The full map is [`doc/licensing.md`](doc/licensing.md).

### Contributor Licence Agreement

Pull requests need a signed CLA — [`CLA.md`](CLA.md), adapted from the Apache
ICLA. A bot asks on your first PR; you reply with one line, once, and it covers
everything you contribute afterwards.

**You keep your copyright.** It is a licence, not an assignment: your
contribution stays yours to use, publish or relicense anywhere else. What the
agreement adds is the right to distribute it under *different* licence terms
later. Without it, any relicensing of future versions would mean tracking
down every contributor who ever landed a patch.

That is also why this is a CLA and not a DCO. A DCO certifies that you had the
right to submit the code; it does not grant the project anything, so it cannot
support a relicence. One mechanism, not both.

**Nothing else requires it.** Bug reports, reproductions, benchmark results,
design discussion and documentation feedback are all welcome with no agreement
of any kind. If you have a fix you would rather not sign for, describe it in an
issue — that is a perfectly good contribution and it will get implemented.

## Reporting bugs

Open an issue at https://github.com/pavelhorak/pion/issues with:

- the commit SHA you reproduced on (`git rev-parse HEAD`)
- the exact `./pion-server …` invocation
- the failing test or wire trace
- platform (Mac / Linux + CPU class — gate baselines differ by CPU class)

For a security issue, email pion@pavelhorak.com directly and don't open a public issue until disclosed.
