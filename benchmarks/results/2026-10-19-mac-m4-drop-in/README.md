# 2026-10-19: vector search without `FT.OPTIMIZE` (Mac)

The checks for the change that lets a stock Redis client search a Pion vector
index: before the graph exists `FT.SEARCH` answers exactly, the graph then
builds itself in the background, and vectors written after it are searched
exactly and merged in (`doc/vector_engine.md`, "Searching without
FT.OPTIMIZE").

- Machine: Apple M4 Mac mini, 10 cores. Client and server on the same machine,
  over loopback.
- Pion: release builds (`pixi run build`) of `main` at `619aea2` (the 0.9.9
  engine) and of the branch at `d5cebeb`, `6ca3181` and `1ba93b9`.
- Server flags: the vector gate's (`-w 10 --independent-workers`, 4 workers on
  macOS), plus `--no-wal`.
- VectorDBBench 1.0.22, two installs:
  - the **stock** client, which never sends `FT.OPTIMIZE`;
  - the **patched** client the vector gate uses (`pixi run install-vdbbench`),
    whose `optimize()` sends it.
- **Ollama was resident throughout**, which costs vector QPS. Each comparison
  below is interleaved, so both sides ran with it; the absolute numbers are not
  gate numbers.

## 1. A stock client, against today's path

[`ab_dropin.sh`](ab_dropin.sh) runs three arms in each of 3 pairs, in an order
rotated per pair:
- **old:** `619aea2` with the patched client;
- **new:** the branch with the stock client;
- **newp:** the branch with the patched client.

Each arm is a fresh server: one full run (load, optimize, concurrent search at
1, 5 and 10 clients for 5 s each, serial recall), then two search-only runs on
the built index. Case: Performance1536D50K (50K OpenAI embeddings, 1536 dims,
COSINE), k=100, ef=150. [`ab_summary.py`](ab_summary.py) produces
[`summary_ab.md`](summary_ab.md) from [`raw/ab/`](raw/ab/).

| | old: `619aea2`, patched client | new: branch, **stock** client | newp: branch, patched client |
|---|---:|---:|---:|
| recall@100, the 3 full runs | 0.9599 / 0.9602 / 0.9598 | **0.9592 / 0.9600 / 0.9604** | 0.9597 / 0.9596 / 0.9597 |
| max QPS, the 6 search-only runs | 8,169–9,035, median **8,675** | 7,780–8,783, median **8,660** (0.998×) | 8,276–8,889, median 8,771 (1.011×) |
| index build | `FT.OPTIMIZE`: 4.0–4.1 s | in the background: 3.6–3.8 s (`build_log.txt`) | `FT.OPTIMIZE`: 2.7–2.8 s |

- **The stock client never sent `FT.OPTIMIZE` and got the same recall.** Its first
  concurrency level ran while the graph was being built, so it was answered by
  the exact scan (1,139–1,366 QPS at 1 client, against 1,585–1,690 for old).
- **newp's `FT.OPTIMIZE` took less time than old's.** The patched client sends it
  more than 200 ms after its last insert, by which time the background build had
  started. `FT.OPTIMIZE` waits for that build and does not build again.
- **None of these runs had vectors written after a build.** So the delta merge
  never ran, and `6ca3181` (which changes only the merge) does not affect them.

## 2. Streaming: VectorDBBench's StreamingPerformanceCase

[`vdbb_stream.py`](vdbb_stream.py) runs it with the 2026-10-18 settings:
- OpenAISmall (50K), 200 rows/s, the stock client;
- search stages at 50% and 80% inserted, then when the write finishes, then
  after `optimize()`;
- concurrency 5 and 10.

Produced by [`s50_summary.py`](s50_summary.py) into
[`summary_s50.md`](summary_s50.md) from [`raw/s50/`](raw/s50/):

| run | 50% | 80% | write finished | after optimize() |
|---|---:|---:|---:|---:|
| `d5cebeb` | 0.0732 | 0.0813 | 0.9671 | 0.9595 |
| `6ca3181`, run 1 | 0.5047 | 0.7947 | **0.8018** | **0.7622** |
| `6ca3181`, runs 2–7 | 0.4986–0.5049 | 0.7900–0.7953 | 0.9663–0.9694 | 0.9595–0.9603 |

For reference, the 2026-10-18 box run: RediSearch HNSW FP32 0.5106 / 0.7974 / 0.9628 / 0.9628, and Pion
0.0000 / 0.0000 / 0.0000 / 0.9600
([`../2026-10-18-linux-epyc-8124p/vector_field/`](../2026-10-18-linux-epyc-8124p/vector_field/README.md)).
At 50% and 80% inserted, no engine can return more than about half or four
fifths of the true neighbours, so 0.50 and 0.79 are the ceiling.

- **`d5cebeb`: new vectors were merged on unlike scores.** The graph scored its
  rows from INT8 codes, and those scores read high (section 3). The vectors
  written after the last build were scored exactly, so they outranked the true
  neighbours. `6ca3181` scores the graph's rows from the FP32 vectors before
  merging.
- **`6ca3181` run 1 ended low, and the cause is not known.** Its last two stages
  read about what a graph missing the last fifth of the vectors gives. Runs 2–7
  were checked afterwards on every worker through its own port
  ([`post_check.py`](post_check.py),
  [`raw/s50/post_check_runs2-7.txt`](raw/s50/post_check_runs2-7.txt)): 0.956–0.958
  on all four, every time. Run 1 had no such check. Eight back-to-back builds of
  the same 50K vectors all measured 0.9577–0.9593 ([`build_repeat.py`](build_repeat.py),
  [`build_repeat.txt`](build_repeat.txt)), so a bad build is unlikely.
- **Every stream ran 40 automatic builds.** The inserter sends one 200-row burst a
  second; each pause is longer than the 200 ms quiet window, and a rebuild
  starts whenever the vectors written since the last build reach a tenth of it.
  During each rebuild every search is exact.

## 3. FT.SEARCH scores from INT8 codes

[`score_error.py`](score_error.py) indexes 20K OpenAI-50K vectors and compares
each returned score with the exact `1 - cos`
([`score_error.txt`](score_error.txt)):

| | mean error | range |
|---|---:|---|
| `619aea2` (0.9.9) and `6ca3181`, explicit `FT.OPTIMIZE` | **+0.0941** | +0.0759 to +0.1043 |
| `1ba93b9`, explicit `FT.OPTIMIZE` | −0.0001 | −0.0017 to +0.0008 |
| `1ba93b9`, automatic build | −0.0001 | −0.0017 to +0.0008 |

The INT8 calibration clips every component to mean ± 8 sigma of the data, which
leaves the few near-constant outlier dimensions of these embeddings outside the
range. That
costs search nothing, because the clipped part cancels between documents. The
score, though, compared the unclipped query with the clipped codes, and counted
the difference. `1ba93b9` clamps the query to the same range. Recall is
unchanged; only the reported score moves.
