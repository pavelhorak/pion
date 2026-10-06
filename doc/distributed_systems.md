# Distributed Systems: Cluster Mode

Pion implements Redis Cluster protocol compatibility, enabling use with cluster-aware clients such as
[Valkey GLIDE](https://github.com/valkey-io/valkey-glide), `redis-py` cluster mode, Lettuce, and Jedis.
Two cluster modes are available: **single-node cluster** (one Pion instance owns all 16,384 hash slots)
and **multi-node cluster** (multiple Pion instances with gossip health monitoring and WAL replication).

> **Multi-node is a preview.** Slot migration, gossip, failover and replication exist and are
> tested, but production multi-node deployment is not supported yet. Replication covers worker 0's
> keyspace only, so run replicated servers with `-w 1`.

---

## Implemented Commands

### CLUSTER INFO

Returns cluster health metadata in `key:value\r\n` format (Redis 7 compatible):

```
CLUSTER INFO
→ cluster_enabled:1
  cluster_state:ok
  cluster_slots_assigned:16384
  cluster_slots_ok:16384
  cluster_slots_pfail:<live pfail count>
  cluster_slots_fail:<live fail count>
  cluster_known_nodes:<1 + peer_count>
  cluster_size:<1 + peer_count>
  ...
```

`cluster_slots_pfail` and `cluster_slots_fail` reflect live gossip health state updated by
the background PING thread.

### CLUSTER MYID

Returns the node's 40-character hex node ID (derived from FNV-1a of `host:port`):

```
CLUSTER MYID
→ "9856db3c0b62d1f44b58db3c0b63d1f400000000"
```

### CLUSTER KEYSLOT \<key\>

Returns the CRC16 hash slot for a key (0–16383):

```
CLUSTER KEYSLOT foo    → 12356
CLUSTER KEYSLOT {user}.profile → slot({user}) = 5474
```

Hash tags (`{...}`) are supported: only the substring inside `{}` is hashed, matching Redis Cluster spec.

### CLUSTER NODES

Returns one line per node with live health status:

```
CLUSTER NODES
→ "<id> <host>:<port>@<bus-port> myself,master - 0 0 1 connected 0-16383"
   "<id> <peer-host>:<peer-port>@<bus-port> master - 0 0 1 connected|pfail|fail <slots>"
```

Peer health values: `connected` (online), `pfail` (possible failure — gossip PING timeout),
`fail` (confirmed failure — repeated PING failures).

### CLUSTER SLOTS / CLUSTER SHARDS

Returns slot-to-node mapping. Same format as previous; health reflected via peer entries.

### CLUSTER MEET \<host\> \<port\>

Registers a new peer at runtime (no restart required):

```
CLUSTER MEET 10.0.0.2 1974  → +OK
```

- Adds peer to `ClusterState.peers[]` with evenly distributed slot range
- Gossip starts health-probing the new peer immediately
- `cluster_epoch` is incremented

### CLUSTER FORGET \<node-id\>

Removes a peer by its 40-char hex node ID:

```
CLUSTER FORGET 9856db3c...  → +OK
```

### CLUSTER REPLICATE \<node-id\>

Marks this node as a replica of the given peer. Used in conjunction with
`--cluster-replica` / `--cluster-primary-host` at startup, or applied dynamically:

```
CLUSTER REPLICATE <primary-node-id>  → +OK
```

### CLUSTER FAILOVER \[FORCE\]

Promotes a replica to primary (clears `is_replica` flag, increments `cluster_epoch`):

```
CLUSTER FAILOVER        → +OK   (graceful — waits for replication sync)
CLUSTER FAILOVER FORCE  → +OK   (forced — skips health checks, immediate promotion)
```

The `FORCE` variant bypasses gossip health verification and promotes immediately. Use when the primary is unreachable and graceful failover would hang.

### CLUSTER SETSLOT \<slot\> \<subcommand\> \[node-id\]

Manages slot ownership during migration:

```
CLUSTER SETSLOT 1234 IMPORTING <source-node-id>   → +OK
CLUSTER SETSLOT 1234 MIGRATING <target-node-id>   → +OK
CLUSTER SETSLOT 1234 STABLE                        → +OK
CLUSTER SETSLOT 1234 NODE <node-id>                → +OK
```

Used for zero-downtime slot migration between nodes. IMPORTING/MIGRATING set transitional state; NODE finalizes ownership; STABLE cancels migration.

### CLUSTER RESET

Resets replica state (promotes to standalone primary):

```
CLUSTER RESET  → +OK
```

### HELLO \[protover\]

RESP2/RESP3 negotiation (required by Valkey GLIDE). Pion always responds with `proto:2`
to prevent GLIDE from enabling RESP3 features:

```
HELLO 2
→ server: pion, version: 1.0.0, proto: 2, mode: cluster, role: master, modules: []
```

---

## CRC16 Hash Slot Computation (`src/network/cluster.mojo`)

`ClusterState` implements the full Redis CRC16 polynomial (`0x1021`) using a 256-entry
precomputed lookup table. The `keyslot()` method handles hash tags per spec:

```
if '{' found before '}' and '}' comes after '{':
    hash only the bytes between '{' and '}'
else:
    hash the entire key
```

`owns_slot(slot)` returns `True` for all slots in single-node mode (owns all 16,384).

---

## MOVED Redirect

When a cluster-aware client sends a command for a key that maps to a slot owned by a peer, Pion
returns a `MOVED` redirect:

```
-MOVED <slot> <host>:<port>\r\n
```

In single-node mode all slots are owned locally so MOVED is never generated for a correctly configured
client. MOVED responses are generated when `peer_nodes` is configured with multiple hosts and the
target slot maps to a peer.

---

## Gossip: Peer Health Monitoring

Pion uses lightweight TCP-based health probing (not SWIM UDP) via a background C pthread
(`PionGossipBlock` in `src/ffi/uring_wrap.c`).

**Mechanism:**

1. Every `gossip_ping_ms` (default 1000ms), the gossip thread iterates all peers
2. For each peer (SWIM, the default): sends a direct probe and waits 100 ms for the
   ack; without one, asks other peers to probe it and waits up to 200 ms more
3. On success: `peer_health[i] = 0` (online), resets consecutive fail counter
4. On failure: increments consecutive fail counter
   - `>= pfail_threshold` consecutive failures (default 5) → `peer_health[i] = 1` (pfail)
   - `>= fail_threshold` consecutive failures (default 15) → `peer_health[i] = 2` (fail)

**Implementation:** `peer_health` is an `InlineArray[UInt8, 16]` inside the heap-allocated
`ClusterState`. The gossip thread writes directly through a pointer, eliminating any
per-tick C call from Mojo. Single-byte writes are inherently atomic on x86_64 and ARM64.

**Architecture:** Only worker 0 starts the gossip pthread. All workers read the same
`ClusterState` health array.

---

## Auto-Failover

Worker 0 monitors gossip health state and automatically promotes a replica to primary when the
current primary is detected as `FAIL` (confirmed failure after `fail_threshold` consecutive PING
failures, default 15 = ~15 seconds of unreachability).

**Mechanism:**

1. Worker 0 checks `peer_health[]` each event-loop tick
2. When a peer transitions to `FAIL` state, worker 0 triggers automatic promotion
3. The replica clears `is_replica`, increments `cluster_epoch`, and begins accepting writes
4. Gossip continues monitoring — if the old primary recovers, it must be manually reconfigured

**Performance:** `tests/test_failover.py` kills the primary and sends `CLUSTER FAILOVER FORCE`
to the replica; the promotion completed in 350 ms on an M4 Mac mini (2026-10-06,
[raw output](../benchmarks/results/2026-10-06-mac-m4/failover.txt)). That is the forced path.
Automatic promotion first waits for gossip to mark the primary `FAIL` (`fail_threshold`
consecutive failed PINGs at the gossip interval, about 15 s at the defaults), and no test
times it yet.

**Testing:**

```bash
python3 tests/test_failover.py
```

This test starts a primary and replica, kills the primary, and verifies the replica auto-promotes
and begins serving reads/writes within the expected timeframe.

**Manual override:** `CLUSTER FAILOVER` (graceful) or `CLUSTER FAILOVER FORCE` (immediate, skips
health checks) can be issued at any time to trigger manual promotion without waiting for auto-detection.

---

## INFO REPLICATION

Returns replication status in `key:value\r\n` format (Redis-compatible):

```
INFO REPLICATION
→ # Replication
  role:master              (or role:slave if --cluster-replica)
  connected_slaves:0
  master_repl_offset:0
```

In replica mode, additional fields report the primary host/port and replication lag. The `role`
field reflects the current state — after `CLUSTER FAILOVER`, a former replica reports `role:master`.

---

## WAL Replication

Primary→replica streaming of Write-Ahead Log entries via a background C pthread.

**Primary side (`PionReplPrimary`):**
- Listens on `port + 10000` (e.g., port 1974 → replication port 11974)
- Accepts replica connections, validates handshake (`PION-REPL 1.0\r\n` → `+OK\r\n`)
- Background thread polls `wal.tail_offset` every 1ms and streams new bytes to each replica
- Handles replica reconnects transparently; restarts from offset 0 on reconnect

**Replica side (`PionReplReplicaBlock`):**
- Connects to `primary_host:primary_repl_port` (primary's port + 10000)
- Receives WAL bytes into a 4MB ring buffer (C-side, protected by mutex)
- Mojo drains the ring each event-loop tick (kqueue and io_uring paths) via
  `pion_repl_replica_drain()` — zero-copy into a pre-allocated drain buffer
- `apply_wal_entries()` parses and applies entries: SET → `keyspace.set()`, DEL → `keyspace.remove_generic()`

**WAL entry format** (same as local WAL, see `src/io/wal.mojo`):
```
[4B entry_len LE][1B cmd_id][4B key_len LE][key bytes][4B val_len LE][val bytes]
cmd_id: 1=SET, 2=DEL
```

Since the WAL does not wrap in Phase 1, replication is a linear byte stream from offset 0.

---

## Enabling Cluster Mode

### Single-node cluster (GLIDE-compatible)

```bash
./pion-server --cluster --cluster-host 127.0.0.1 --cluster-nodes 127.0.0.1:1974

# Verify
redis-cli -p 1974 CLUSTER INFO | head -3
redis-cli -p 1974 CLUSTER MYID
redis-cli -p 1974 CLUSTER KEYSLOT mykey
redis-cli -p 1974 HELLO 2
```

### Multi-node cluster (primary + replica)

```bash
# Primary node (port 1974, replication listener on 11974)
./pion-server --cluster --cluster-host 10.0.0.1 \
              --cluster-nodes "10.0.0.1:1974,10.0.0.2:1975"

# Replica node (connects to primary for WAL streaming)
./pion-server --cluster --cluster-host 10.0.0.2 -p 1975 \
              --cluster-nodes "10.0.0.1:1974,10.0.0.2:1975" \
              --cluster-replica \
              --cluster-primary-host 10.0.0.1 \
              --cluster-primary-port 1974
```

### Runtime peer management

```bash
# Add peer at runtime (no restart)
redis-cli -p 1974 CLUSTER MEET 10.0.0.3 1976

# Remove peer
redis-cli -p 1974 CLUSTER FORGET <node-id>

# Promote replica to primary
redis-cli -p 1975 CLUSTER FAILOVER
```

### Configuration flags

```
--cluster                  Enable cluster mode
--cluster-host <ip>        Advertised IP in CLUSTER NODES output
--cluster-nodes <list>     Comma-separated "host:port,..." including self
--cluster-replica          This node replicates from primary
--cluster-primary-host <h> Primary's advertised host
--cluster-primary-port <p> Primary's main port (replication port = p + 10000)
--gossip-ping-ms <ms>      Gossip PING interval (default: 1000ms)
```

ClusterConfig (`src/common/config.mojo`):
```
ClusterConfig:
  enabled:         false   (opt-in via --cluster)
  my_host:         "127.0.0.1"
  peer_nodes:      ""
  is_replica:      false
  primary_host:    ""
  primary_port:    0
  gossip_ping_ms:  1000
  pfail_threshold: 5       (consecutive PING failures → pfail)
  fail_threshold:  15      (consecutive PING failures → fail)
```

---

## Implementation (`src/ffi/uring_wrap.c`)

All gossip and replication logic runs as C pthreads to avoid blocking Mojo's event loop
(macOS Mojo green threads don't yield at OS syscalls; pthreads are full OS threads).

| C struct | Role |
|---|---|
| `PionGossipBlock` | Gossip pthread state; writes `peer_health[]` directly |
| `PionReplPrimary` | Primary listener: accept + stream thread |
| `PionReplReplicaBlock` | Replica receiver thread + 4MB ring buffer |

Mojo calls per event-loop tick (both kqueue and io_uring paths):
- `pion_repl_replica_drain(handle, buf, max)` → drain ring into apply buffer

---

## Client Compatibility

| Client | Status |
|---|---|
| `redis-cli` (single-node) | ✅ All 9 CLUSTER subcommands verified |
| `redis-py` cluster mode | ✅ CLUSTER SLOTS/SHARDS format correct |
| Valkey GLIDE | ✅ HELLO 2 + single-node mode |
| Lettuce / Jedis | ✅ CLUSTER NODES format with live health |

---

## Architecture Notes

**Worker 0 exclusivity:** Only worker 0 starts gossip and replication pthreads. This is
necessary because Pion caps workers at 4 P-cores on macOS (main.mojo policy) — all threads are needed
for the event loop. pthreads bypass this limit (they are scheduled by the OS independently).

**Shared-nothing consistency:** In shared-nothing mode, each worker has its own keyspace.
Replication only applies to worker 0's keyspace. Use `-w 1` (the default) for fully replicated
workloads; `-w N > 1` refuses to start without `--independent-workers`.

**macOS cluster testing:** Use `redis-cli` in cluster mode or single-node GLIDE.
Full multi-node testing requires two separate Pion instances on separate ports/IPs.

---

## Cluster Protocols

### 1. SWIM Gossip Protocol

Replaced basic TCP PING with UDP SWIM protocol (`pion_gossip_start_swim()` in `uring_wrap.c`).

**Features:**
- **Random target selection** — each round probes a random peer (not sequential)
- **Indirect ping** — if direct ping fails, asks K=3 random delegates to probe the target
- **State exchange** — SWIM messages carry node_id, epoch, role, slot_count, pfail bitmap
- **Suspicion protocol** — SUSPECT/ALIVE/CONFIRM messages for graceful failure detection
- **TCP fallback** — each round also does a TCP PING for reliability
- **Quorum FAIL promotion** — pfail→fail requires majority of nodes reporting PFAIL

**Ports:** SWIM UDP = main port + 20000 (e.g., 21974 for port 1974).

**Usage:** Enabled by default when `--cluster` is set. The `GossipManager.start(use_swim=True)` parameter
controls SWIM vs legacy TCP. Legacy TCP is still available with `start(use_swim=False)`.

### 2. Raft Metadata Consensus

Lightweight Raft for cluster metadata — leader election and topology change commits.
NOT for data replication (WAL handles that).

**Features:**
- **Leader election** — randomized timeout (150-300ms), RequestVote RPC, majority vote
- **Heartbeat** — leader sends AppendEntries every 50ms to maintain authority
- **Term tracking** — monotonic term counter prevents stale leaders
- **Log replication** — topology changes committed via Raft log (slot migration, failover)
- **Split-brain prevention** — only Raft leader can commit topology changes

**Ports:** Raft TCP = main port + 30000 (e.g., 31974 for port 1974).

**C API:**
```c
void* pion_raft_create(const char* node_id, int raft_port);
void  pion_raft_set_peer(void* block, int i, const char* host, int port);
int   pion_raft_start(void* block);
int   pion_raft_is_leader(void* block);
int   pion_raft_get_state(void* block);  // 0=Follower, 1=Candidate, 2=Leader
```

### 3. vNode Migration (Slot Migration)

Full slot migration with CLUSTER SETSLOT + MIGRATE + ASK/ASKING protocol.

**Commands:**
- `CLUSTER SETSLOT <slot> MIGRATING <node-id>` — mark slot for export
- `CLUSTER SETSLOT <slot> IMPORTING <node-id>` — mark slot for import
- `CLUSTER SETSLOT <slot> NODE <node-id>` — finalize slot ownership
- `CLUSTER SETSLOT <slot> STABLE` — cancel migration
- `MIGRATE host port key|"" db timeout [COPY] [REPLACE] [KEYS key ...]` — push keys to target
- `ASKING` — per-fd flag for importing slot acceptance
- `DUMP key` — serialize key to binary format
- `RESTORE key ttl payload [REPLACE]` — deserialize and store
- `CLUSTER GETKEYSINSLOT slot count` — find keys in a slot
- `CLUSTER COUNTKEYSINSLOT slot` — count keys in a slot

**ASK/MOVED redirect flow:**
1. Client sends command for migrating slot
2. If key exists locally → serve it
3. If key not found → `-ASK <slot> <target>` redirect
4. Client sends `ASKING` to target, then retries command
5. Target serves the command (ASKING flag clears after one use)

**MIGRATE internals:** Uses `dump_key()` to serialize, TCP to target, `RESTORE` to deserialize.
Supports COPY (keep local), REPLACE (overwrite existing), multi-key via `KEYS` keyword.

### 4. Cross-Worker WAL Aggregation

Workers 1..N forward WAL entries to worker 0 via per-worker ring buffers (4MB each).
Worker 0 drains all rings each event-loop tick and applies entries to the master WAL.

**C API:**
```c
void* pion_wal_agg_create(int worker_count);
int   pion_wal_agg_push(void* block, int worker_id, const uint8_t* entry, int entry_len);
int   pion_wal_agg_drain(void* block, int worker_id, uint8_t* out_buf, int max_bytes);
void  pion_wal_agg_stop(void* block);
```

**Ring buffer:** 4MB per worker, mutex-protected, supports wrap-around.
Worker 0 can drain entries from any worker's ring for replication.

### 5. Replica-Reads

Per-connection READONLY/READWRITE enforcement on replica nodes.

**Commands:**
- `READONLY` — enable read-only mode on this connection (allows reads on replicas)
- `READWRITE` — restore default mode (reject reads on replicas)

**Write rejection:**
- Non-READONLY connections on replicas: `-MOVED` to primary
- READONLY connections on replicas: reads pass through, writes get `-READONLY` error
- Write commands identified by exclusion: GET/MGET/PING/EXISTS/HGET/LLEN/LRANGE/DBSIZE/
  GETBIT/BITCOUNT/PFCOUNT/CLUSTER/CONFIG/INFO/HELLO/READONLY/READWRITE/ASKING/QUIT/ECHO
  are reads; everything else is a write.

**CLUSTER NODES output:** Replicas now show `myself,slave` instead of `myself,master`.
