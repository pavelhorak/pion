<!-- Public copy for the landing page and the docs site's "Where Pion does not
help" page. website/landing/build_landing.py and website/gen_pages.py read the
two sections below by heading, so keep the headings exactly as written. -->

## Two numbers, with their denominators

| | vanilla mlx-lm | Pion warm | |
|---|---:|---:|:---:|
| Llama-3.2-1B-4bit, 2,049-token prefix, **same process** | 1,242 ms | **61.9 ms** | **20×** |
| Same model and prefix, **from a separate process**, over the wire | 1,242 ms | 73.9 ms | **17×** |

Both rows answer the same 16-token question after the same prefix, and differ
only in *where the cache comes from*. The first is the process that computed
it, which an in-process prompt cache also gives you. The second pays a wire hop
for a cache a separate process wrote, and produces **BLEU 1.000** on a 50-token
greedy completion against the reference output: a different socket and a
different model object, same text
([test output](../../benchmarks/results/2026-10-06-mac-m4/kv_prefix_cross_instance.txt)). Vanilla here prefills the way mlx-lm's own
`generate_step` does; the ratios this table showed until 2026-10-02 did not,
and were too high (the changelog lists them).

A third number used to sit here: 64K context with a sparse mask, needle found
while attending under 1% of the prefix. It is withdrawn until its reproducer
works again: on 2026-10-06, with mlx-lm 0.31.3, neither vanilla mlx-lm nor Pion
found the needle
([raw output](../../benchmarks/results/2026-10-06-mac-m4/sparse_mask_64k_niah_full.txt)).

Both rows come from one script,
[`cross_process_ttft.py`](../../benchmarks/reproducers/cross_process_ttft.py)
(`--same` adds the first), and its raw output for every run is
[published](../../benchmarks/reproducers/results/cross_process_ttft_2026_10_02.json).
The [reproducers](../../benchmarks/reproducers/README.md) list what each one needs.

## Where this does not help

A 34-token prefix saves **1.5×** — about 16 ms. This section exists because
the demo that ships with this project once printed a slowdown at a short prefix,
slower than doing nothing, and we shipped it that way for a while.

- **It caches prefill, not decode.** If your bottleneck is tokens-per-second
  once generation starts, this changes nothing.
- **The reuse has to be real, and the prefix has to be long.** Time to first
  token from a separate process, only the prefix length varying: 34 tokens →
  1.5×. 268 → 4.5×. 1,035 → 11×. 2,049 → **17×**. At short prefixes the saving
  is small — about 16 ms at 34 tokens, not worth a second process. The demo
  defaults to 2,048, and `--prefix-tokens 33` shows how little there is to save.
- **The sparse selector is NIAH-class.** Factual QA over dense paragraphs needs
  a query-aware selector; the default block-mean selector is built for finding a
  needle, not for reading every paragraph.
- **`-w N` is N independent keyspaces**, not one shared keyspace. The server
  refuses to start with `-w N > 1` unless you pass `--independent-workers`.
- **No TLS.** Loopback by default; terminate TLS at a proxy.
- **No eviction.** `--maxmemory` refuses writes with `-OOM` once the process
  reaches the limit, but nothing is ever evicted to make room.
- **Cluster mode is single-node.** Slot migration and replication exist;
  production multi-node does not.
- **One maintainer**, working evenings.

