<!-- Public copy for the landing page and the docs site's "Where Pion does not
help" page. website/landing/build_landing.py and website/gen_pages.py read the
two sections below by heading, so keep the headings exactly as written. -->

## The numbers, with their denominators

| Llama-3.2-1B-4bit, 2,049-token prefix | first token | what it reads | vs cold |
|---|---:|---:|:---:|
| Cold prefill, vanilla mlx-lm | 1,193 ms | nothing | |
| mlx-lm's own prompt-cache file, a fresh process | **37.0 ms** | a 67 MB file (mmap) | 32× |
| Pion, **same process** | 46.2 ms | its own MLX arrays | 26× |
| Pion, **from a separate process**, over the wire | 69.0 ms | 67 MB over loopback TCP | 17× |
| Pion, **from a separate process** on the same machine | **37.0 ms** | a file Pion wrote (`V.EXPORT`, mmap) | 32× |

**On the same machine, Pion hands over a file too.** mlx-lm ships
`save_prompt_cache` / `load_prompt_cache`, and mapping a file beats fetching the
same rows over TCP, so a hit on the same machine maps a file Pion wrote when
the prefix was stored: 37.0 ms, the file's own time. If one program reuses one
fixed prefix, mlx-lm's file is the simpler tool
([`file_cache_ttft.py`](../../benchmarks/reproducers/file_cache_ttft.py),
[raw output](../../benchmarks/reproducers/results/file_cache_ttft_llama_2049_2026_10_07.json)).
Pion is for what a file does not do: one cache that any process, model object
or tool reads over the Redis wire, an acked write that survives a crash, and
in `pion-vllm-mlx serve` a longest-prefix match that a restarted agent or a
second session finds without naming a file.

The Pion rows answer the same 16-token question after the same prefix, and differ
only in *where the cache comes from*. The first is the process that computed
it, which an in-process prompt cache also gives you. The second pays a wire hop
for a cache a separate process wrote, and produces **BLEU 1.000** on a 50-token
greedy completion against the reference output: a different socket and a
different model object, same text
([test output](../../benchmarks/results/2026-10-07-mac-m4/kv_prefix_cross_instance.txt)). Vanilla here prefills the way mlx-lm's own
`generate_step` does; the ratios this table showed until 2026-10-02 did not,
and were too high (the changelog lists them).

Both rows come from one script,
[`cross_process_ttft.py`](../../benchmarks/reproducers/cross_process_ttft.py)
(`--same` adds the first), and its raw output for every run is
[published](../../benchmarks/reproducers/results/cross_process_ttft_2026_10_07.json).
The [reproducers](../../benchmarks/reproducers/README.md) list what each one needs.

A third number: at 64K context, with a sparse mask that attends 0.80% of the
prefix on each full-attention layer, Gemma-4-E2B-it-4bit still finds a single
needle, and the warm call takes 124.4 ms against vanilla's 54.4 s cold prefill
(437×). It is one needle at one depth, warm against cold
([raw output](../../benchmarks/results/2026-10-07-mac-m4/sparse_mask_64k_niah.txt)).
mlx-lm's prompt-cache file of the same prefix (405 MB) answers in 82.6 ms with
dense attention over all of it
([raw output](../../benchmarks/reproducers/results/file_cache_ttft_gemma_64000_2026_10_07.json)).
The example missed on 2026-10-06 because its prompt carried 397 `<bos>`
tokens; it now carries one.

## Where this does not help

Across all 16 cores of one Linux server, both sides as sixteen independent
keyspaces, sixteen Redis processes beat Pion's sixteen workers by **13–18%**.
Pion's lead over Redis is on one keyspace with I/O threads.

- **`-w N` is N independent keyspaces**, not one shared keyspace. The server
  refuses to start with `-w N > 1` unless you pass `--independent-workers`.
- **The default is one thread.** Without `--io-threads`, Pion trails Redis with
  `io-threads 8` at pipeline depths 1 and 10, and leads single-threaded Redis
  at every depth. I/O threads pay on a server with cores to spare; on a Mac
  with the client on the same machine they lower throughput, for Redis too.
- **No TLS.** Loopback by default; terminate TLS at a proxy.
- **Eviction holds RSS, not allocator bytes.** `--maxmemory-policy` evicts as
  Redis's policies do, but freed memory stays with the allocator, so RSS can sit
  somewhat over the limit after a burst.
- **Cluster mode is single-node.** Slot migration and replication exist;
  production multi-node does not.
- **One program, one fixed prefix: use mlx-lm's own prompt-cache file.**
  `save_prompt_cache` / `load_prompt_cache` ship with mlx-lm and answer faster
  than a Pion fetch: 37.0 ms against 69.0 ms at a 2,049-token prefix. Pion's
  prompt cache earns its place when several processes, tools or restarts share
  the cache.
- **The prompt cache caches prefill, not decode.** If your bottleneck is
  tokens-per-second once generation starts, it changes nothing.
- **The reuse has to be real, and the prefix has to be long.** Time to first
  token from a separate process, only the prefix length varying: 34 tokens →
  1.4×. 268 → 4.6×. 1,035 → 12×. 2,049 → **17×**. At short prefixes the saving
  is small — about 14 ms at 34 tokens, not worth a second process. The demo
  once printed a slowdown at a short prefix, and shipped that way for a while;
  it defaults to 2,048 now, and `--prefix-tokens 33` shows how little there is
  to save.
- **The sparse selector is NIAH-class.** Factual QA over dense paragraphs needs
  a query-aware selector; the default block-mean selector is built for finding a
  needle, not for reading every paragraph.
- **A large vector index that keeps taking writes.** Vectors written since the
  last build are searched exactly, and during a rebuild every search is. At
  500K vectors on an EPYC 8124P a query took 1.6 ms from the graph alone,
  3.8 ms with 10,000 vectors written since, and 190 ms during the 84 s rebuild.
  An index loaded once and then queried pays none of this.
- **One maintainer**, working evenings.
