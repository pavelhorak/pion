<!-- Public copy for the landing page and the docs site's "Where Pion does not
help" page. website/landing/build_landing.py and website/gen_pages.py read the
two sections below by heading, so keep the headings exactly as written. -->

## Two numbers, with their denominators

| | vanilla mlx-lm | Pion warm | |
|---|---:|---:|:---:|
| Llama-3.2-1B-4bit, 2,048-token prefix, **same process** | 1,530 ms | **30.2 ms** | **50.6×** |
| Same model and prefix, **from a separate process**, over the wire | 1,558 ms | 64.7 ms | **24×** |

The two rows differ by *where the attention runs*, not by how much faster the
same thing got. The first keeps model and cache in one process. The second pays
a wire hop for a cache a separate process wrote, and produces **BLEU 1.000** on
a 50-token greedy completion against the vanilla output — a different socket
and a different model object, same text.

A third number, because it is the one that surprised us: at 64K context with a
sparse mask, 100% needle recall while attending **0.78%** of the prefix, with
bit-identical greedy decode. That is needle-in-a-haystack retrieval, not a
claim about every long-context task — see the list below.

Each row has a script: [`tests/bench_ttft.py`](../../tests/bench_ttft.py) for
the first, [`cross_process_ttft.py`](../../benchmarks/reproducers/cross_process_ttft.py)
for the second. The [reproducers](../../benchmarks/reproducers/README.md) list
what each one needs.

## Where this does not help

This section exists because the demo that ships with this project once printed
**0.86×** — slower than doing nothing — and we shipped it that way for a while.

- **It caches prefill, not decode.** If your bottleneck is tokens-per-second
  once generation starts, this changes nothing.
- **The reuse has to be real, and the prefix has to be long.** Time to first
  token from a separate process, only the prefix length varying: 34 tokens →
  1.6×. 268 → 6.0×. 1,035 → 16.4×. 2,049 → **24×**. The ratio holds up at short
  prefixes; the saving does not — at 34 tokens it is about 19 ms, not worth a
  second process. The demo defaults to 2,048, and `--prefix-tokens 33` shows how
  little there is to save.
- **The sparse selector is NIAH-class.** Factual QA over dense paragraphs needs
  a query-aware selector and collapses to 0.38×.
- **`-w N` is N independent keyspaces**, not one shared keyspace. The server
  refuses to start with `-w N > 1` unless you pass `--independent-workers`.
- **No TLS.** Loopback by default; terminate TLS at a proxy.
- **No eviction.** `--maxmemory` refuses writes with `-OOM` once the process
  reaches the limit, but nothing is ever evicted to make room.
- **Cluster mode is single-node.** Slot migration and replication exist;
  production multi-node does not.
- **One maintainer**, working evenings.

