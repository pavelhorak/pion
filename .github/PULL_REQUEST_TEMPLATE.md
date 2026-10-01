## What this changes

<!-- One paragraph. What was wrong, or what is new. -->

## Gates

CI runs the correctness gates on Linux and macOS automatically. **Perf gates do
not run in CI and cannot** — they need a quiet machine. If this touches the
network engine, the KV path or the vector kernels, run them locally and paste
the numbers:

```
<!-- valkey-benchmark / memtier / VectorDBBench output -->
```

Before reading a number as a regression, remember that several rows sit close
enough to their floor that ordinary run-to-run noise looks like one, and that a
leaked server from an earlier run costs ~30% on writes.

**Never move a gate threshold to make a run pass.** A row below its floor is
something to diagnose.

## Checklist

- [ ] Correctness gates pass locally (`test_raw.py`, `test_parity.py`)
- [ ] Perf gates run and quoted, or not applicable
- [ ] New mutating command appends a WAL effect record (or routes through a
      dispatcher `execute_*` that does)
- [ ] New dispatch arm consumes its own frame (`test_dispatch_sweep.py`)
- [ ] Docs updated if behaviour or a documented number changed
