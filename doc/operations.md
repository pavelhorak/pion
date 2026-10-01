# Operating Pion — crash diagnostics, supervised serving, liveness

| Question | Answer |
|---|---|
| Why did it die? | **Breadcrumbs** — crash log + 1 Hz status file (§1) |
| How do I keep it running? | **Supervisor** — `scripts/pion-supervise.sh` (§2) |
| How many workers? | **`-w 1`** unless you know you want N keyspaces (§2b) |
| How do I tell a dead store from an empty result? | **FT.SEARCH errors on a missing index** (§3) |
| What survives a restart? | **Every type, TTLs, streams, the vector index** (§3b–§4) |
| How do I install it? | **Tarball or Docker** (§5) |

---

## 1. Crash / exit diagnostics

On by default. Every server writes two files into its working directory:

| File | Written | Contents |
|---|---|---|
| `pion-<port>.crash.log` | append, on start and on any catchable death | one line per event |
| `pion-<port>.status` | rewritten in place at 1 Hz | fixed 512-byte `key=value` record |

```bash
./pion-server --profile vector -w 1 -p 1974        # both files, default paths
./pion-server --crash-log /var/log/pion.log ...    # explicit crash-log path
./pion-server --status-file /run/pion.status ...   # explicit status path
./pion-server --no-crash-log ...                   # disable both
./pion-server --rss-warn-pct 60 ...                # warn earlier (default 70; >100 = off)
```

### The two death shapes

**Catchable** (SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT/SIGTERM/SIGINT/SIGQUIT/SIGXCPU/SIGHUP)
— the handler appends a line, plus a backtrace for a fault, and then re-raises
the default disposition, so the exit status and any core dump are exactly what
they would have been:

```
PION START: pid=9659 version=0.815+7f42d4d port=1974 workers=1 unix_s=1785086858 total_ram_mb=16384
PION EXIT: signal SIGTERM (15) pid=9659 port=1974 uptime_s=17 rss_mb=67 rss_peak_mb=67 rss_pct=0 hb_age_s=0 ticks=283
```

A clean shutdown, a bind failure included, writes `PION EXIT: clean shutdown`.

**Uncatchable** — macOS jetsam and the Linux OOM killer send **SIGKILL**, which
no handler in any language can observe. There is by construction no in-process
trace. That is what the status file is for: it is the last thing the process
said about itself, and it is written *before* the kill rather than after.

```
$ cat pion-1974.status
pion_status_version=1
state=running            ← still "running" though the process is gone == killed from outside
signal=0
signal_name=-
pid=9659
version=0.815+7f42d4d
port=1974
workers=1
start_unix_s=1785086858
uptime_s=48
heartbeat_unix_s=1785086906
rss_bytes=423706624      ← RSS as of ≤1 s before death — this is what confirms/denies jetsam
rss_peak_bytes=423706624
total_ram_bytes=17179869184
rss_pct=2
ticks=749                ← event-loop ticks; frozen value distinguishes "was serving" from "was wedged"
```

**Reading it:**

| Evidence | Verdict |
|---|---|
| `PION EXIT: signal …` present | in-process death; the signal names the cause |
| No exit line + `state=running` + high `rss_pct` | OS memory kill (jetsam / OOM). Confirm via §2's postmortem |
| No exit line + `state=running` + low `rss_pct` | external `kill -9`, or the host went down |
| No exit line + a Mojo stack trace on stderr | a runtime abort; read the server's stderr log |
| `ticks` unchanged across several seconds while the process lives | wedged event loop, not a memory problem |

A one-shot warning fires when RSS crosses `--rss-warn-pct` of physical RAM, so
the operator hears about pressure *before* the kernel reaps the process:

```
PION WARN: rss_mb=11468 is 70% of total_ram_mb=16384 — OS may kill this process (jetsam/OOM) without a catchable signal
```

To refuse writes before that point instead of dying, set `--maxmemory` (see
`doc/configuration.md`).

### Cost

The heartbeat rides the existing 64-tick housekeeping block of every event
loop — one non-atomic counter bump plus a `clock_gettime`; the RSS sample,
status write and CAS happen once per second, performed by whichever worker wins
the CAS. Nothing is added to the request path. Because the counter is per
worker, a status file whose `ticks` stop advancing while the process is alive
localises a hung worker.

The last command executed is **not** tracked: recording it would put a store
on the zero-allocation fast path for every request.

---

## 2. Supervised serving

`scripts/pion-supervise.sh` starts a server, waits for a real `+PONG` (not just
a successful `connect()` — Pion listens *before* it initialises its 10M-slot
hash map, so a connect can succeed seconds before the server can serve), then
polls liveness and restarts on death with exponential backoff.

```bash
scripts/pion-supervise.sh -- --profile vector -w 1 --nle-embed -p 1974
scripts/pion-supervise.sh --interval 2 --max-restarts 10 -- --profile kv -w 4 --independent-workers
scripts/pion-supervise.sh --once -- --profile kv -w 1     # postmortem, no restart
```

| Option | Default | Meaning |
|---|---|---|
| `--binary PATH` | `./pion-server` | server binary |
| `--port N` | inferred from `-p`/`--port`, else 1974 | health-probe port |
| `--interval S` | 5 | health poll period |
| `--timeout S` | 3 | per-probe deadline |
| `--start-timeout S` | 120 | readiness wait after a start |
| `--max-restarts N` | 0 (unlimited) | give up after N |
| `--backoff S` | 1 | initial backoff, doubles to 60 s; reset by a successful start |
| `--log PATH` | `pion-supervisor-<port>.log` | supervisor log |
| `--once` | off | do not restart; exit with the server's status |

Everything after `--` is passed to `pion-server` verbatim.

Detected failures: process exit (any status), **and** a live process that stops
answering PING within `--timeout` (wedged event loop) — the latter is killed and
restarted. A single dropped probe is retried once before being called a failure.

Each death produces a postmortem in the supervisor log:

```
[…] pion-server pid=9659 exited unexpectedly
[…] POSTMORTEM pid=9659 exit_status=137 signal=9 last_state=running uptime_s=15 ticks=233
[…] POSTMORTEM rss_mb=67 peak_mb=67 of total_mb=16384 (0%) last_heartbeat_unix_s=1785086873
[…] POSTMORTEM no jetsam record for pid=9659 in the last 10m — killed by something else (manual kill -9, or a crash before the handler ran)
[…] POSTMORTEM crash-log tail:
[…] POSTMORTEM   PION START: pid=9659 version=0.815+7f42d4d port=1974 workers=1 …
[…] restart #1 in 1s
```

The jetsam/OOM lookup queries `/usr/bin/log show --last 10m` on macOS for
jetsam / memorystatus / lowswap records naming the dead pid or binary; on Linux
it greps `dmesg` / `/var/log/kern.log` for OOM-killer records. A confirmed kill
prints `POSTMORTEM CONFIRMED OS memory kill`.

Only bash, `awk` and `/dev/tcp` are required — no `redis-cli`, and no
`timeout(1)` (absent by default on macOS).

Under a real init system, keep the supervisor for its postmortem or hand
restarts to `launchd`/`systemd` and read `pion-<port>.status` from your own
monitoring — the status file is the machine-readable contract either way.

---

## 2b. Worker count — `-w N` is N keyspaces

**The default is `-w 1`, and `-w N > 1` refuses to start without
`--independent-workers`.** This is an operational fence, not a tuning knob.

Workers are shared-nothing: each owns a private hash map, WAL, HNSW graph and
allocators, there is no cross-worker request bus, and connections are assigned by
`accept()` race on one shared listen fd. So a write acknowledged `+OK` on one
connection is invisible to a read on another. Every default connection pool
(redis-py, Jedis, go-redis, ioredis) opens more than one connection, which makes
the violation silent — nils, no error, nothing in any log. Measured at `-w 4`
with 16 concurrently-opened connections: **41 of 90** GETs of a just-acked key
returned nil.

```bash
./pion-server -w 1                          # default — one coherent keyspace
./pion-server -w 16 --independent-workers   # 16 keyspaces; prints the semantics at startup
./pion-server -w 16                         # FATAL, exit 1, names the flag
```

Deploy `-w N > 1` only when:

- every client **pins a single connection** for its whole lifetime, or shards
  keys across per-worker ports itself; **or**
- the workload is read-mostly vector search — the HNSW graph *is* published
  across workers via `SharedHNSWView`, so `FT.SEARCH` stays coherent at `-w N`
  (same ranking, same keys on every worker) even though the KV keyspace does
  not. Hash fields are per worker: a result answered by a worker that did not
  ingest the doc omits its `id` field rather than inventing one.

Otherwise run N single-worker Pions on N ports and shard client-side.

Three related splits follow the same line and are not fixed by the flag:
cross-worker `PUBLISH` delivers to zero subscribers, `FLUSHALL` clears one
worker's slice, and replication covers worker 0 only.

**Diagnostic note.** Connections opened *serially* all land on one worker and
mask the split completely — a serial probe reports zero nils on a `-w 16` server.
Any check of cross-worker behaviour must open its connections concurrently.
`tests/test_gh253_multiworker_fence.py` is the regression test.

Caps interact with the fence in the safe direction: `--flare`/`--inference`/
`--profile ai` collapse any `-w N` back to 1, and macOS caps to 4. The fence reads
the **final** count, so a `-w 8` that a cap already reduced to 1 starts normally.

---

## 3. Client-visible liveness

`FT.SEARCH` against an index that does not exist returns an error, not an empty
array:

```
-ERR no such index 'my_docs' - not created on this server (FT.CREATE first);
    if the server restarted, its index was not persisted
```

A client cannot distinguish "the store lost its index" from "the query matched
nothing" when both are `*0`. This matters most after a restart: the WAL replays
keys, but `FT.CREATE` is not a WAL entry, so a restarted server without a saved
index file serves KV traffic while every vector query would otherwise return
nothing.

Scope:

- Fires only when **no index exists at all** — this worker has never seen
  `FT.CREATE`, has no nodes, and the shared cross-worker view is empty too.
- A **created-but-empty** index still returns `*0`. An empty result on a live
  index is a legitimate answer, not an error.
- A server holds one index; querying an index another `FT.OPTIMIZE` displaced
  errors and names both indexes (see `doc/vector_engine.md` § One index per
  server).
- Both the KNN and `BM25` sub-commands are covered, and pipelined clients stay
  frame-synchronised.

**Client-side corollary:** a retriever that catches `ConnectionRefusedError` and
returns `[]` will still hide a dead server. Distinguish "no results" from "cannot
reach the store" at your call site.

---

## 3b. Durability of every type

`SAVE` / `BGSAVE` write every keyspace type, and every mutating command logs
its effect to the WAL. Removes are logged too, so a crash replay cannot
resurrect deleted members. Coverage is machine-readable in `INFO`
(Persistence section) — check these before trusting a restart:

| Field | Meaning |
|---|---|
| `snapshot_version:2` | SAVE covers every value type, streams included |
| `aggregate_wal_records:1` | mutations effect-log to the WAL |
| `hll_wal_logged:1` | PFADD logs each element; PFMERGE logs the merged register image, because a register-wise max cannot be reconstructed from elements |
| `streams_persisted:1` | XADD logs the *resolved* entry ID and XDEL/MAXLEN trims log tombstones |
| `ttls_persisted:1` | EXPIRE-family logs the absolute deadline, PERSIST logs its removal |

Replay reproduces **resolved effects, never arguments.** `XADD key *` logs the
ID the server generated, so recovery cannot renumber the stream against a
different wall clock. `EXPIRE key 60` logs the absolute deadline, so an outage
does not silently extend every TTL by its own length. `SPOP` logs an `SREM` of
the member it actually removed.

Verify with `python3 tests/test_gh170.py` and `python3 tests/test_gh174.py` —
both run every shape through both crash modes (SAVE + SIGKILL, and SIGKILL
only). The two modes exercise different code (snapshot serializer vs WAL
replay), so check both.

## 3c. The vector index file

`FT.OPTIMIZE` writes the index to `pion.hnsw.N` so a restart skips the rebuild.
The file records the slot stride, header offset, compact format and distance
metric, the slot → key map, and the INT8 calibration, and the loading worker
publishes the index to every worker.

Quantized indexes (`--polarquant`, `--turboquant`, `--nanoquant`) carry two
optional sections, flagged in header word 25: the FP32 re-rank buffer
(`num_nodes × dim × 4` bytes — ~300 MB at 50K × 1536, so plan disk for it) and
TurboQuant's QJL signs + residual norms. Plain INT8 files do not carry them.

A file the server cannot use — an older format, or a quant format different
from the server's current flags — is **refused with a log line** and the index
is rebuilt cold (re-ingest + `FT.OPTIMIZE`).

Verify: `python3 tests/test_gh211_warm_restart.py --quant {int8,polarquant,turboquant,nanoquant}`
asserts the quantized build really ran, then compares recall against brute
force before and after SIGTERM.

## 4. Durability of large values

| Concern | Behaviour |
|---|---|
| Log capacity | The active WAL file is sealed to `pion.wal.{worker}.{n}` and a fresh one opened. Recovery replays sealed segments oldest-first, then the active file. |
| Genuinely full | Only when `--wal-max-segments` is exhausted. The entry is counted (`wal_dropped_entries`), a one-shot `WAL: FULL` line is printed, and the log names the fix. Never silent. |
| Large values | Values ≥ `--blob-threshold` (1 MiB) are copied into mmap'd arena files `pion.blob.{worker}.{seg}` (1 GiB each). Pages are file-backed — reclaimable under memory pressure — and on disk when written. |
| Ordering | The WAL stays the single ordered index: a large `SET` logs a 24-byte pointer record, not the payload. Large-over-small and small-over-large overwrites, and `DEL`, all replay correctly. |
| `SAVE` | The snapshot stores the same pointer record, so a checkpoint does not copy a multi-GB arena into a second file. |

```bash
./pion-server --wal-size 512 --wal-max-segments 64   # 32 GB of unsnapshotted delta
./pion-server --blob-threshold 262144                # arena from 256 KiB up
./pion-server --no-blob-tier                         # large values on the heap
redis-cli -p 1974 INFO | grep -E 'wal_|blob_'
```

### Operating notes

- **A full log is an operator signal, not a policy.** `wal_dropped_entries > 0`
  means acknowledged writes were refused. Run `SAVE`/`BGSAVE` (which checkpoints
  the WAL and unlinks sealed segments) or raise `--wal-max-segments`.
- **The arena reclaims at startup, not in place.** Overwriting or deleting a
  blob key strands its bytes; `blob_bytes_used` counts arena bytes handed out,
  including stranded ones. On the next start, after replay and before the
  socket serves, Pion counts what the keyspace can still reach and — when the
  arena is over 256 MB and more than half of it is stranded — rewrites the live
  payloads into fresh segments, re-points every value, and unlinks the old
  files. A compaction is immediately followed by a snapshot + checkpoint; watch
  for `Blob tier: compacted N value(s), reclaimed M MB`. `blob_compactions` and
  `blob_bytes_live_at_last_scan` are in `INFO`.
- **A long-running write-heavy process still grows** between restarts — the
  ceiling is 64 GiB of arena per worker, after which large values fall back to
  the heap path with a warning. Restart to reclaim.
- **Segment files are sparse.** `pion.blob.0.0` reports 1 GiB in `ls` while
  occupying only what has been written; use `du` for real usage.
- **`--no-blob-tier` is the escape hatch** if you need heap residency. It is
  still durable — values ride the rotating WAL instead.

---

## Verification

```bash
python3 tests/test_gh138_robustness.py                    # 36 checks, ~3 min
python3 tests/test_gh138_robustness.py --binary ./pion-server-dev --port 7861
python3 tests/test_gh149_gh163_blob_durability.py         # 19 checks, ~2 min
```

Covers: heartbeat freshness and tick advance, SIGTERM breadcrumb with preserved
exit status, SIGKILL leaving a frozen `state=running` record with the last RSS,
`FT.SEARCH` error/empty/hit trichotomy with the connection staying usable, and a
supervisor restart with a filed postmortem.

---

## 5. Distribution — prebuilt tarball & Docker image

Official binaries are built by CI from a `v*` tag
(`.github/workflows/release.yml`): macOS arm64, Linux x86_64 and Linux arm64,
each tested before upload, with the versioned tarballs, byte-identical
`pion-<platform>.tar.gz` aliases and `SHA256SUMS`.

**Prebuilt tarball:**

```bash
tar -xzf pion-<version>-macos-arm64.tar.gz && cd pion-<version>-macos-arm64
./pion-server.sh                    # wrapper sets the dyld/ld library path
```

Run the wrapper, not `bin/pion-server`: the tarball bundles the Mojo runtime
libraries the binary needs, and the wrapper points the loader at them.

**Metal shader library (macOS).** `--metal-attention` loads
`metal_compute.metallib`, searched relative to the **executable**: `<exe>/`,
`<exe>/../lib`, `<exe>/../share/pion`, `<exe>/../src/ffi`, then the working
directory for a source checkout. `PION_METAL_LIB` overrides the search. The
banner prints `Metal Attn: requested`, and the engine then prints
`Metal Attn: ACTIVE (<path>)` or `Metal Attn: NOT ACTIVE` under the same key, so
one grep answers whether it is running.

**Inference sidecar.** When the PyTorch sidecar (`--inference`, or auto-embed)
cannot be found — a release tarball ships none — the server says so and starts
without it rather than waiting. When it can be found, the server listens first
and the sidecar starts concurrently, so clients queue instead of being refused.

**Building a tarball yourself** — `scripts/package_release.sh` (run on the
target platform; add `--build` to build first):

```bash
scripts/package_release.sh          # → dist/pion-<version>-<os>-<arch>.tar.gz
```

It ships `./pion-server` as it finds it, so build at HEAD first; it refuses a
working tree with uncommitted changes (`--allow-dirty` overrides for local use).
On macOS it refuses when `src/ffi/metal_compute.metallib` is missing or older
than `metal_compute.metal` — building the shaders needs full Xcode, not only
the Command Line Tools. To test a tarball the way a user will run it, hide
`.pixi/envs/default/lib` first: the binary's baked rpath points there, so an
unpacked tarball resolves its libraries on the build machine whether or not the
bundle is complete.

**Docker** — root `Dockerfile`, multi-arch, GPU-free:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t pion .
docker run -p 1974:1974 -v pion-data:/data pion
```

- amd64 builds with `build-portable` (`--target-cpu x86-64-v2`) so the image
  runs on any x86-64 host; arm64 uses the default GPU-free build.
- Default CMD is `--epoll --no-auto-embed`: Docker's default seccomp
  profile blocks io_uring (Docker ≥ 25), and the slim image ships no
  Python/torch for the embedding sidecar. For io_uring:
  `docker run --security-opt seccomp=unconfined pion --no-auto-embed`.
- State (WAL, snapshots, blob arenas) lands in the `/data` volume.
- `docker build --target runtime-prebuilt -t pion:dev .` wraps an
  already-built host `./pion-server` and its runtime libraries in seconds, for
  local use. The default target builds from source (~20 min cold on 4 cores,
  toolchain layer cached; budget ~5 GB of build cache,
  `docker builder prune -f` reclaims it).
- `Dockerfile.linux` + `docker/run-linux.sh` are an interactive Linux dev
  shell, not the shippable image.

CUDA images are a separate lane (the linux-64 default build requires nvcc;
see `pixi.toml`).
