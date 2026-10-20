# 2026-10-20: vector search without `FT.OPTIMIZE`, measured on the EPYC box

The [2026-10-18 run](../../2026-10-18-linux-epyc-8124p/vector_field/README.md) measured
Pion's index as it was then: built once, by `FT.OPTIMIZE`. Since then Pion searches
without it (`doc/vector_engine.md`, "Searching without FT.OPTIMIZE"):
- before the graph exists, `FT.SEARCH` is exact;
- the graph then builds itself in the background;
- vectors written after the build are searched exactly and merged in.

This run repeats the streaming case on the same type of machine, with the client
that never sends `FT.OPTIMIZE`, next to RediSearch. It also measures what writes
after the build cost at 500K vectors.

Two Pion builds ran:
- **Pion A**, the change described above;
- **Pion B**, a follow-up that is not merged (PR #92). B rebuilds into a second graph while
  the first keeps answering, and ranks the delta from an 8-bit copy of each vector.

## Machine and software

- **Machine:** AMD EPYC 8124P, 16 cores / 32 threads, 124 GiB, NVMe RAID1, Ubuntu
  24.04, bare metal ([`setup.txt`](setup.txt)).
- **Pion:** both builds with `pixi run build-portable` on the machine, the
  release's linux-x86_64 build (x86-64-v2 plus the closed vector library, which
  picked its VNNI build: `libpion_vector abi=1 (closed, dims=1536) isa=vnni`).
  - Pion A: `pion-server 0.9.9+4c735f8`.
  - Pion B: `pion-server 0.9.9+f638318`.
- **Redis:** 8.10.2 with RediSearch, HNSW FLOAT32, `search-workers 16`. Built and
  configured as on 10-18 ([`engines/`](engines/)).
- **Harness:** VectorDBBench 1.0.22 with redis-py 4.6.0.
  - Pion used the **stock** install. Its client file is byte-identical to the one
    1.0.22 ships, and its `optimize()` does nothing, so no `FT.OPTIMIZE` is sent.
  - RediSearch used the 10-18 competitor install, whose `optimize()` waits until
    RediSearch reports the index built.
- **Pinning:** server on physical cores 0-7 and their SMT siblings, client on 8-15
  and theirs, as on 10-18. Datasets: VectorDBBench's S3 copies
  ([`datasets.sha256.txt`](datasets.sha256.txt)).

## 1. The streaming case at 50K

This is VectorDBBench's `StreamingPerformanceCase` with the 10-18 settings
([`scripts/vdbb_stream.py`](scripts/vdbb_stream.py)):
- OpenAI 50K, inserted at 200 rows/s;
- searches at 50% and 80% inserted, at concurrency 5 and 10;
- 30 s of searches after the last write.

Pion ran at `-w 16 --independent-workers` with the WAL on. After each Pion run,
[`scripts/s50_post_check.py`](scripts/s50_post_check.py) read recall@100 on each
of the 16 workers through its own port, before the server stopped. Per block:
[`blocks/S50_*`](blocks/), summarized in [`summary.md`](summary.md).

| | recall@100 at 50% / 80% / write done / after `optimize()` | max QPS at 50% / 80% | max QPS after the stream |
|---|---|---:|---:|
| Pion A, run 1 | 0.5138 / 0.7993 / 0.9669 / 0.9599 | 2,155 / 2,928 | 6,709 / 7,077 |
| Pion A, run 2 | 0.5132 / 0.8001 / 0.9671 / 0.9602 | 2,126 / 2,905 | 6,954 / 6,662 |
| Pion B, run 1 | 0.5097 / 0.7998 / 0.9672 / 0.9597 | 3,125 / 3,128 | 6,694 / 6,721 |
| Pion B, run 2 | 0.5086 / 0.7993 / 0.9678 / 0.9601 | 3,049 / 3,276 | 6,856 / 6,935 |
| RediSearch HNSW FP32 | 0.5084 / 0.7974 / 0.9629 / 0.9629 | 1,797 / 1,752 | 1,852 / 1,836 |

- **During the stream, Pion now answers at RediSearch's recall.** On 10-18 its
  recall was 0 until `FT.OPTIMIZE`.
- **Pion A, the lower of its two runs, served 1.18× and 1.66× RediSearch's QPS at
  the two stages** (2,126 against 1,797, and 2,905 against 1,752).
- **After the stream it served about 3.6× as much** (6,662 against 1,836).
- **Pion B served 1.70× and 1.79×.**
- **Every worker returned the same recall after each Pion run.** The end of the
  stream read normally in all four. On the Mac, 1 run in 7 of A had ended at 0.76.

## 2. Writes after the build, at 500K

[`scripts/delta_steps.py`](scripts/delta_steps.py), with one client against Pion
at `-w 1 --no-wal`:
- loads the 500K OpenAI training vectors through one pipelined connection;
- waits for the automatic build;
- times 1,000 queries (KNN 100, `EF_RUNTIME 150`, the first 50 dropped);
- does that with no delta, and after 10,000, 25,000 and 45,000 vectors written
  after the build;
- then writes past a tenth of the index, which starts a rebuild, and times
  queries while it runs.

Latency is p50 and includes the client's round trip. Pion B ran with
`PION_VECTOR_DELTA_MAX=1000000` and `PION_VECTOR_SETTLE_MS=30000`, its rebuild
knobs, so that the steps held still; A has neither.

| | Pion A | Pion B |
|---|---:|---:|
| Search, the graph alone | 1.616 ms | 1.613 ms |
| with 10,000 vectors written after the build | 3.820 ms | 3.444 ms |
| with 25,000 | 6.481 ms | 5.470 ms |
| with 45,000 | 9.969 ms | 8.103 ms |
| Added per 1,000 such vectors (fit, delta points) | 0.176 ms (+0.457 ms) | 0.133 ms (+0.512 ms) |
| First build of 500,000 (server log) | 66.7 s | 67.3 s |
| Rebuild of ~550,000 (server log) | 84.1 s | 78.8 s |
| **Search while that rebuild runs** | **190–195 ms** | **31 ms** |
| Search after it | 1.631 ms | 2.021 ms |

- **A large index that keeps taking writes is where Pion is slow.** Vectors
  written since the last build are searched exactly. At 500K, 10,000 of them
  more than double a query's time.
- **In Pion A every search is exact during a rebuild.** Here a rebuild took
  84 s, and a query took about 190 ms instead of 1.6.
- **A load-once index pays none of this:** after its build it answers from the
  graph alone.
- **Pion B keeps answering from the old graph during a rebuild.** That costs
  31 ms here: its 51,000-vector delta, plus the build's threads, which share the
  server's 16 CPUs with the search.
- **B's rebuild left 1,000 vectors in its delta.** The rebuild started while the
  last vectors were still arriving, and nothing folded them in afterwards. So its
  after-rebuild search reads 2.02 ms, not 1.6.
- **On this CPU B's 8-bit copy is 1.3× cheaper per vector than A's FP32 scan.**
  On the Mac it was about 4×. Why is not measured.

Per block: [`blocks/D500_1_pion_b`](blocks/D500_1_pion_b/) and
[`blocks/D500_2_pion_a`](blocks/D500_2_pion_a/), with the server's build lines in
`builds.txt`.

## Not measured

- A streaming case at 500K. At 200 rows/s one run takes about 42 minutes.
- The bulk-load case (`Performance1536D50K` / `500K`) with the stock client:
  time until searchable at 500K, as a stock client sees it.
- Where B's 8-bit scan spends its time on x86.

## Files

- [`summary.md`](summary.md): every table above, generated from the blocks.
- `blocks/<block>/`, one folder per block:
  - `r1.json`: VectorDBBench's result, for the S50 blocks;
  - `r1_cmd.txt` and `r1_run.txt`: the command and its output;
  - `server_log.txt`, `engine.txt` (versions and flags), `info.txt` and
    `post_check.txt`;
  - for the 500K blocks, `delta_steps.txt` and `cmd.txt`.
- [`engines/`](engines/), [`smoke/`](smoke/) (the build check),
  [`chain_progress.txt`](chain_progress.txt) (the run's timeline, UTC),
  [`scripts/`](scripts/).
