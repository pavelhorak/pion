# 2026-10-19: rebuilding a served vector index (Mac)

The checks for phase B of searching without `FT.OPTIMIZE` (`doc/vector_engine.md`, "Searching
without FT.OPTIMIZE"; phase A is [`../2026-10-19-mac-m4-drop-in/`](../2026-10-19-mac-m4-drop-in/README.md)).
B changes three things:

- **A rebuild no longer stops the graph serving.** It builds into a second graph while the first,
  with its exact delta, keeps answering. Phase A unpublished the graph for each rebuild, so every
  search during one was an exact scan.
- **The delta is scanned through an INT8 copy of each vector**, and the best candidates are then
  scored from FP32, so scores stay exact.
- **An automatically built index keeps indexing after a restart.**

Setup, as for phase A:
- Machine: Apple M4 Mac mini, 10 cores. Client and server on the same machine, over loopback.
- Pion builds:
  - **A:** the phase A branch at `4c735f8`.
  - **B:** release builds of the phase B branch. The final one is the last commit.
- Server flags: the vector gate's (`-w 10 --independent-workers`, 4 workers on macOS), plus `--no-wal`.
- **Ollama was resident throughout.** Every comparison is interleaved, so both sides ran with it.
  The absolute numbers are not gate numbers.
- The disk had 1.8–2.0 GB free, under the benchmark preflight's 2 GB. Every run used `--no-wal`,
  and no build ran during a measurement.

## 1. What the delta costs a query

[`delta_cost_run.sh`](delta_cost_run.sh) runs phase A's [`delta_cost.py`](delta_cost.py):
- `-w 1`, a 50K graph of OpenAI embeddings (1536 dims, COSINE);
- 1,000 queries, KNN 100, one client;
- first with no delta, then with 1,000 or 4,900 vectors written after the build.

The settle rebuild is off (`PION_VECTOR_SETTLE_MS=0`), so the delta stays in place.
Raw: [`raw/delta_cost/`](raw/delta_cost/).

| build | +1,000 delta rows | +4,900 delta rows | slope, per 1,000 rows |
|---|---:|---:|---:|
| FP32 delta (phase A's scan) | +0.112 ms | +0.572 ms | 0.118 ms |
| INT8 ranking, delta and graph rescored apart | +0.155 ms | +0.294 ms | 0.036 ms |
| INT8 ranking of both, one FP32 rescore (final) | +0.115 ms | +0.233 ms | 0.030 ms |

The query alone (graph, no delta) took 0.429–0.439 ms at the median. The slope is the difference
between the two delta sizes per 3,900 rows. (The script's own "per 1,000 rows" line divides by all
the delta rows, so it includes the fixed part.)
- The INT8 scan costs about a quarter of the FP32 one per row.
- What remains, about 0.085 ms a query (0.115 ms at 1,000 rows, less the slope), is scoring ~141
  scattered rows from FP32. That cost does not grow with the delta.

## 2. Streaming: VectorDBBench's StreamingPerformanceCase

The case and its settings are phase A's, with the stock client. Each run is
[`s50_run.sh`](s50_run.sh), a fresh server per run. Two interleaved series ran,
[`s50_series.sh`](s50_series.sh). After each run, [`post_check.py`](post_check.py) read recall on
every worker through its affinity port. Produced by [`s50_summary.py`](s50_summary.py) from
[`raw/s50/`](raw/s50/).

The second series is the final build ([`summary_s50_ab2.md`](summary_s50_ab2.md)):

| run | 50% | 80% | write finished | after optimize() | max QPS by stage | builds |
|---|---:|---:|---:|---:|---|---:|
| A | 0.4983 | 0.7947 | 0.9676 | 0.9596 | 4,251 / 4,080 / 8,528 / 8,781 | 40 |
| B #1 | 0.5030 | 0.7945 | 0.9690 | 0.9588 | 5,459 / 5,810 / 8,866 / 8,471 | 35 |
| B #2 | 0.5031 | 0.7946 | 0.9674 | 0.9598 | 5,463 / 5,692 / 8,681 / 7,955 | 35 |

- **In-stream QPS is about 1.35× A's, at the same recall.** Means: 5,606 against 4,166.
- **After the stream, QPS is level with A's.**
- **Every worker read the same recall after each run.**

The first series ([`summary_s50_ab1.md`](summary_s50_ab1.md), 2 A and 2 B runs) used an earlier
build. That build settled the leftover delta after 1 s without writes. VectorDBBench writes one
burst a second, so the 1 s settle fired mid-stream, 53–54 builds a stream. One run's last rebuilds
overlapped the after-stream stages. The settle is 3 s now.

Neither series showed phase A's intermittent end-of-stream dip (0.80 / 0.76 in 1 run of 7).

## 3. A stock client on a bulk load (Performance1536D50K)

[`k50_series.sh`](k50_series.sh) uses phase A's arms, 2 pairs, order rotated:
- A with the stock client;
- B with the stock client;
- B with the patched client, which sends `FT.OPTIMIZE`.

Final build: [`summary_k50.md`](summary_k50.md), from [`raw/k50_b7/`](raw/k50_b7/).

| | A, stock | B, stock | B, patched |
|---|---:|---:|---:|
| recall@100 | 0.9595 / 0.9600 | 0.9598 / 0.9602 | 0.9601 / 0.9604 |
| load | 3.3–3.7 s | 3.2–3.3 s | 3.4 s |
| builds, the last | 1, 3.6 s | 1, 3.5 s | 1, `FT.OPTIMIZE` 2.8–3.8 s |
| search-only max QPS, median of 4 | 8,302 | 8,642 | 8,800 |

An earlier B build ([`summary_k50_first_trigger.md`](summary_k50_first_trigger.md), from
[`raw/k50_b6/`](raw/k50_b6/)) started building at 16,384 vectors without waiting for a pause. On
this load that meant builds at about 16K, 23K and 35K vectors:
- the load took 5.1–5.2 s;
- the graph covering all 50K vectors came later than A's.

Building without a pause now needs searches that are paying for exact scans (an exact scan in
the last second). A bulk load that nobody queries waits for its pause, as in phase A.

## 4. Tests

`tests/test_vector_auto_build.py` has 148 checks.
- **Streaming, `-w 1` and `-w 4`:** writes in 300-row batches every 50 ms, with searches on every
  worker. No empty replies; `vector_index` never exact-only; rebuilds with no pause; each stored
  vector finds itself.
- **Restart with the WAL, `-w 1` and `-w 4`:**
  - the delta, overwrites and deletes come back;
  - a new `HSET` is indexed at once;
  - a rebuild after the restart keeps recall;
  - an `FT.OPTIMIZE` index stays closed.

## Not measured here

Two things need the Linux box, on which the 2026-10-18 comparison ran:
- the streaming case against RediSearch on the same machine;
- the cost of the delta and of a rebuild at 500K vectors.
