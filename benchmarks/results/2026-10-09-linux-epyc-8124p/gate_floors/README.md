# Linux gate floors, re-derived (#28)

EPYC 8124P (Zen 4c, 16 cores / 32 threads, 2.45 GHz), Ubuntu 24.04, kernel 6.8.
Native build of public main at 4b672bb (`pixi run build`; the vector library
picks its VNNI build by default; `binary.txt`). The three gate harnesses, in the
gate's own order, with `--gate-profile linux-epyc-8124p`:

- KV floors: `valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only`
- `memtier-benchmark.py --profiles throughput,pipeline -w 32`
- `vectordb-benchmark.py --pion-only --ef-runtime 150 --workers 16`

2 warm-up rounds (not kept), then 5 recorded rounds (`rec1`..`rec5`). The rule,
fixed before the data: known_good = the median of the 5 recorded rounds, floor =
95% of it. A floor is only ever raised; a row whose new floor would be lower
keeps its current floor and is listed below for a separate decision.
`analysis.txt` applies the rule to every row.

## Raised

| Row | Old floor | New floor | Median |
|---|---:|---:|---:|
| HSET | 561,450 | 562,130 | 591,716 |
| LPUSH | 668,800 | 708,955 | 746,268 |
| RPUSH | 582,350 | 646,258 | 680,272 |
| MSET (10 keys) | 470,250 | 641,891 | 675,675 |
| PING_INLINE | 542,450 | 669,014 | 704,225 |
| PING_MBULK | 571,900 | 669,014 | 704,225 |
| ZADD | 551,950 | 703,703 | 740,740 |
| ZPOPMIN | 549,100 | 572,289 | 602,409 |
| XADD | (none) | 637,583 | 671,140 |
| Vector peak QPS | 4,280 | 7,312 | 7,697 |
| Vector FT.OPTIMIZE (ceiling) | 28 s | 7.7 s | 7.32 s |

Vector recall: median 0.960; its floor stays 0.940.

## Not raised (the rule would lower them)

GET, SET, INCR, LPOP, RPOP, SADD, SPOP, LRANGE_100/300/500/600, and memtier at
P=1 and P=10 keep their floors. Four of them sit below their current floor at
the median on this box: LRANGE_100 (160,256 vs 167,200), LRANGE_300 (41,963 vs
53,200), LRANGE_500 (25,549 vs 31,350), LRANGE_600 (21,450 vs 26,600). The
LRANGE rows measure the benchmark client's reply parsing as much as the server.

## A caution about the single-worker KV rows

On this box each single-worker KV run lands near either ~590K or ~740K ops/sec
(`analysis.txt`, per-round column): the server's worker and the benchmark
client share 32 hardware threads unpinned, and the two modes look like where
they land. A floor taken from the high mode fails the runs that land in the low
one: in these 5 rounds, 1 of 5 runs of LPUSH, RPUSH, ZADD, MSET, PING_INLINE and XADD,
and 2 of 5 of PING_MBULK, are below their new floor.
