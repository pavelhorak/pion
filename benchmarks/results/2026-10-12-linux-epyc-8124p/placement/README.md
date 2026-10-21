# 2026-10-12: where the client runs decides the single-worker KV numbers

**The question:** why the single-worker floor runs on an AMD EPYC 8124P were bimodal.
Unpinned, each row came out near either ~740K or ~590K ops/sec.

**Setup:**
- **Harness:** `benchmarks/valkey-benchmark/valkey-benchmark.py -c 50 -n 100000 -P 10 -w 1 --pion-only --gate --gate-profile linux-epyc-8124p`.
- **Binary:** `pion-server 0.9.9+b74b82a`, the native build (`pixi run build`), which is what the Linux floors were measured with.
- **Client:** Redis 8.10.2's `redis-benchmark`.
- **Topology:** the machine has 4 L3 domains of 4 cores: {0-3,16-19}, {4-7,20-23}, {8-11,24-27} and {12-15,28-31}. CPU n's SMT sibling is n+16.

## 1. Server pinned, client anywhere else: [`pinned_any_cpu/`](pinned_any_cpu/)

- **Placement:** the server on CPUs 2,18; the harness and client on the other 30 CPUs ([`setup.txt`](pinned_any_cpu/setup.txt)).
- **Runs:** 5 recorded rounds.
- **Result:** every row of every round landed in the low mode.
- **Where the client ran:** the `*_placement.txt` samples show it on CPUs 5 and 23, which are in another L3 domain.

## 2. Three placements, interleaved: [`arms/`](arms/)

The arms and the prediction were written down before any of this data
([`run_arms.sh`](arms/run_arms.sh)). The server is always pinned. Each arm ran 3 rounds:

| Row (ops/sec) | client in the server's L3 domain | client in another L3 domain | client on the server's SMT sibling |
|---|---|---|---|
| SET | 740,740 · 735,294 · 729,927 | 588,235 · 595,238 · 595,238 | 588,235 · 588,235 · 591,716 |
| LPUSH | 740,740 · 740,740 · 740,740 | 606,060 · 598,802 · 602,409 | 588,235 · 595,238 · 598,802 |
| ZADD | 740,740 · 740,740 · 735,294 | 598,802 · 598,802 · 595,238 | 591,716 · 595,238 · 591,716 |

- **The high mode** is the client sharing the server's L3 domain. Every other placement gives the low mode, the SMT sibling included.
- **The rest of the rows,** in the logs, split the same way.
- **The exception is LRANGE.** It does not move, because it is limited by the client.
- **A logging bug:** the `rc=` the script logged is wrong (it reads `$?` after a `$(date)`). The tables in the logs are the data.

**What changed:** `valkey-benchmark.py` now pins this placement on Linux. The server goes on
the second physical core of the L3 domain holding the lowest CPU, and the client on that
domain's other cores. `--pin off` restores scheduler placement.
