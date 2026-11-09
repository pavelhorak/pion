# What a key costs: pointer slots against main and Redis

`benchmarks/reproducers/memory_per_key.py`, 2026-11-09, M4 Mac mini (macOS 26.6),
Pion `-w 1 --no-wal`, Redis 8.10.3 from Homebrew. The probe loads `k:<n>` keys
with values of a fixed size, 1,000 per pipeline, and reads RSS idle and loaded.
"Data" is loaded less idle.

- `pion_pointer_slots_*`: the keyspace's 8-byte slots (`KeyShard`), this change.
- `main_*`: main at `d71d888` (the 7/8 load factor and mapped tables, 1.0.9 + PR #116 and #117).
- `redis_*`, and the second half of each `main_and_redis_*`: Redis, same keys and values.

200-byte values:

| keys | Pion, pointer slots | main | Redis |
|---:|---:|---:|---:|
| 100,000 | 23.3 MB (244 B/key) | 28.8 MB | 27.2 MB |
| 150,000 | 35.7 MB (249 B/key) | | 41.8 MB |
| 200,000 | 46.4 MB (243 B/key) | 57.2 MB | 54.2 MB |
| 230,000 | 54.5 MB (248 B/key) | 73.5 MB | 61.8 MB |
| 300,000 | 70.5 MB (246 B/key) | | 83.4 MB |
| 400,000 | 92.0 MB (241 B/key) | | 108.3 MB |
| 460,000 | 109.2 MB (249 B/key) | | 123.5 MB |

3-byte values, 200,000 keys: 6.4 MB (33 B/key); main 16.8 MB; Redis 14.2 MB.

Loaded RSS at 200,000 × 200 B: 103.1 MB, main 113.7 MB, Redis 60.8 MB. Pion's
idle RSS (56.7 MB, Redis 6.6 MB) is the other half of what a `--maxmemory`
limit sees, and this change does not address it. Pion's idle reads either 56.7
or 60.7 MB from one start to the next, with the same data delta
(`pion_pointer_slots_100000x200*`).

## Instructions and cycles per command

The server's retired instructions and cycles, read with `proc_pid_rusage`
around a fixed number of requests from `redis-benchmark` (`-c 50`), less the
idle rate over the same time; `-w 1 --no-wal`, keys `key:%012d` with 3-byte
values. In these logs `pion-server-ps2` is this change, `pion-server-seed2`
is main's keyspace code (built from #117's branch) and `pion-server.sh` is the
1.0.9 release tarball.

- `instructions_one_key.txt`: one loaded key, `-r 1`, 1M requests, median of
  3. GET (a hit): 1.0.9 3,157, main 3,108, this change 3,170. SET 3,606 /
  3,589 / 3,451; MSET 6,794 / 6,755 / 6,563.
- `get_5m_keys_*.txt`: 5,000,000 keys, GET `-r 5000000`. At `-P 100` with 4
  client threads (20M requests, 3 runs a start), cycles per GET: 1.0.9
  264–266, main 261–266, this change 284–288. Requests a second read ~10.0M
  for all three: the client's ceiling, not the server's. At `-P 10` and
  `-P 50` (one client thread) cycles per GET are 1,316–1,342 and 359–367 on
  main, 1,362–1,388 and 387–389 on this change.
