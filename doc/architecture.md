# Architecture: Shared-Nothing, Thread-per-Core

**Pion** is a shared-nothing, thread-per-core database engine written in Mojo. Each worker thread owns all of its resources — no locks, no shared state on the hot path.

---

## Thread Model

```
main()
  ├── create_listen_socket(port)        # One shared listen fd (no SO_REUSEPORT)
  └── pion_spawn_workers(N) → pthreads  # N workers (16 MB stacks), each pinned to a CPU core (capped at 4 on macOS Apple Silicon)
        └── worker_task(i)
              ├── set_thread_affinity(i)
              ├── Pion.__init__()
              │     ├── Step 1: NetworkEngine.__init__() + server.listen(shared_fd)
              │     │            ← Socket is LIVE before heavy init
              │     └── Step 2: StripedHashMap(65,536 slots, grows) + HNSWGraph + WAL + RaftNode
              └── Pion.run_server()
                    └── NetworkEngine.run_server_kqueue()   # macOS: kqueue event loop
                        NetworkEngine.run_server_epoll()    # Linux --epoll: best for P=1 w=1
                        NetworkEngine.run_server_uring()    # Linux default: io_uring event loop
                        NetworkEngine.run_server_xdp()      # Linux --xdp: AF_XDP kernel bypass
```

Each worker owns **privately**:
- `StripedHashMap` — the keyspace: 8 `SlabHashMap` shards, 65,536 slots to start, each shard doubling as it fills
- `HNSWGraph` — the vector index (or borrows from `SharedHNSWView` after FT.OPTIMIZE)
- `ObjectPool[SlabHashMap/SlabSkipList/SlabList]` — recycled data structure instances
- `WAL`, `RaftNode` — persistence and replication state
- `KVCacheStore`, `LayerStore`, `AttentionIndex` — externalized attention state (when `--kvcache`)
- `SemanticRouter` — semantic router state (when `--kvcache`)
- `SpeculativeRAG` — per-session trajectory tracking + speculative cache (when `--kvcache`)
- Network socket + kqueue fd + recv buffer + response buffer

**Shared across all workers:**
- One listen socket fd (created before worker spawn; workers call `accept()` independently — one wins per connection)
- `SharedHNSWView` — read-only pointers to the index built by the first worker to run FT.OPTIMIZE; other workers lazy-borrow these pointers on the first FT.SEARCH

---

## Event Loop (macOS: kqueue, Linux: epoll / io_uring / XDP)

```
while True:
    nevents = kevent_batch(kq, pending_changes, pending_count, events, 1024)
    for event in events[0..nevents]:
        if event.filter == EVFILT_WRITE:        # Back-pressure drain
            writer.flush_response(fd, server, kq)
        elif fd == server.fd:                   # New connection
            new_fd = server.accept()
            server.set_nonblocking(new_fd)
            server.set_tcp_nodelay(new_fd)
            server.kevent_add_read(kq, new_fd)  # Immediate (not batched)
        else:                                   # Client data
            while True:
                n = server.recv(fd, buffer + stored, avail)
                consumed = fast_path.process_data_plane(...)
                if consumed == 0:
                    consumed = slow_path.process_slow_path(...)
                # memmove leftover; break when recv returns < buffer_size
```

`EVFILT_READ` is registered immediately on accept (not batched) to avoid missing data that arrives before the next `kevent_batch()` call.

### Linux backends

Four backends are available on Linux, selected by CLI flag:

| Backend | Flag | Syscalls per batch (K fds) | Best for |
|---|---|---|---|
| **epoll** | `--epoll` | 2K+1 | P=1 per-command benchmarks (w=1) |
| **io_uring** | `--iouring` (default) | 2 | Multi-connection production |
| **io_uring SQPOLL** | `--sqpoll` | 0-1 | Experimental (hangs under load) |
| **XDP/AF_XDP** | `--xdp` | 0 (kernel bypass) | Tail latency at P=1 over a real NIC (needs flow steering); multi-worker + P>1 ready |

### One event loop per worker

Each worker runs `NetworkEngine.run_server_*()` directly on its own pthread;
there is no inner task scheduler. Mojo green threads do not yield at OS
syscalls, so a second task per worker would block the event loop.

### Throughput

Measured throughput against Redis, on one machine with the raw runs published, is in the
[README](../README.md#the-engine-underneath) and
[`benchmarks/results/`](../benchmarks/results/README.md).

---

## Per-Request Data Flow

```
TCP recv
    │
    ├──► FastPathHandler.process_data_plane()     # zero-alloc, 28 hot commands
    │         returns consumed > 0 on success, 0 to fall through
    │
    └──► SlowPathHandler.process_slow_path()      # RESP3 fallback, FT.* commands
              │
              ▼
    ResponseWriter.flush_response(fd, server, kq)
              │
              ├── send() succeeds → done
              └── EAGAIN → copy to pending_buffers[fd], register EVFILT_WRITE
```

---

## Shared-Nothing Data Model

Each worker's keyspace is completely independent. `SO_REUSEPORT` is **not** used — instead a single shared listen fd distributes connections via kernel `accept()` scheduling. A client connected to worker 0's fd stays on worker 0 for its entire lifetime.

**This is observable as a correctness property, so it is fenced.** `-w N` is N independent keyspaces, not one keyspace served by N threads: a `SET` acknowledged `+OK` on a connection that landed on worker 1 is invisible to a `GET` on a connection that landed on worker 3. Every default connection pool (redis-py, Jedis, go-redis, ioredis) opens more than one connection, so the violation is silent — nils, no error. Measured at `-w 4` with 16 concurrently-opened connections: **41 of 90** GETs of a just-acked key returned nil.

Two consequences for anyone measuring this:

- **The default is `-w 1`**, and `-w N > 1` refuses to start without `--independent-workers`, which prints the semantics at startup. The flag is an acknowledgement, not a feature switch — it changes no behaviour beyond letting the server boot.
- **Connections opened SERIALLY all land on one worker** and hide the split completely (0/12 nils in the same session that measured 41/90 concurrently). Any probe of cross-worker behaviour must open its connections concurrently, or it proves nothing.

The same split runs through the rest of the surface: `FLUSHALL` clears one worker's slice, and replication covers worker 0 only. Pub/sub is the exception: a message reaches subscribers on every worker through per-worker inboxes, while PUBLISH counts its own worker's subscribers (#42).

Vector data is an exception: after `FT.OPTIMIZE`, the index-building worker publishes read-only pointers via `SharedHNSWView`. Other workers borrow these pointers lazily on the first `FT.SEARCH`. All mutable per-search state (`visited_map`, `query_int8`, `cur_num`) remains per-worker.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Single shared listen fd | Avoids `SO_REUSEPORT` kernel hash imbalance; all workers compete fairly for connections |
| EVFILT_READ registered immediately on accept | Prevents data loss if client sends before next `kevent_batch()` |
| 4 MB shared response buffer per worker | Pre-allocated; no `alloc` on hot path |
| Ziplist (≤1024 entries) + Quicklist (>1024) for lists | Contiguous memory for small lists; two-sided 256-element segmented arrays for large lists |
| Generation-counter visited-set in HNSW | O(1) mark/check without clearing the `max_elements`-entry array between queries |
| INT8-INT8 batch-8 search kernel | Graph build and search use the same metric; no dequantization per neighbor |
| Binary protocol on port+1 | 0xCA5E framing avoids RESP parsing overhead for bulk tensor ops |

---

## Externalized Attention Modules

When `--kvcache` is enabled, the following modules are activated:

| Module | File | Role |
|---|---|---|
| KVCacheStore | `src/network/kv_cache_store.mojo` | Per-session, per-layer KV cache tensor storage |
| LayerStore | `src/network/layer_store.mojo` | Binary blob storage for per-layer tensors (LAYER.STORE/FETCH) |
| AttentionIndex | `src/network/attention_index.mojo` | Per-layer HNSW index for O(N log k) attention approximation |
| BinaryProtocol | `src/network/binary_protocol.mojo` | 0xCA5E framed wire protocol on port 1975 (main_port + 1) |
| kv_cache commands | `src/commands/kv_cache.mojo` | KV.STORE, KV.FETCH, KV.INFO handlers |
| attend commands | `src/commands/attend.mojo` | ATTEND.CREATE, ATTEND.STORE, ATTEND.FINALIZE, ATTEND.QUERY, ATTEND.INFO handlers |

**Data flow (externalized attention):**
```
Inference engine (vLLM/MLX) → PionAttentionClient (vllm-pion/)
    │
    ├── ATTEND.CREATE session_id key_dim value_dim
    ├── ATTEND.STORE session_id layer_id num_tokens keys_fp32 values_fp32
    ├── ATTEND.FINALIZE session_id layer_id  (builds the layer's HNSW)
    └── ATTEND.QUERY session_id layer_id k query_fp32
         └── Returns the top-k value rows
```

**Performance** (`tests/test_attend_128k.py` over RESP, one layer, 128K tokens, 128-d keys, M4 Mac mini, 2026-10-06,
[raw output](../benchmarks/results/2026-10-06-mac-m4/attend_128k.txt)):
- Store: 222,473 tok/s
- Query: 1.54 ms client round trip per layer
- Build: 11.3 s for 128K tokens (finalize)
- Memory: ~125 MB per layer at 128K

---

## Semantic Router Modules

When `--kvcache` is enabled, the semantic routing modules are activated:

| Module | File | Role |
|---|---|---|
| SemanticRouter | `src/network/semantic_router.mojo` | HNSW/FP32 routing table indexed by node centroid embeddings |
| route commands | `src/commands/route.mojo` | AI.ROUTE.REGISTER, AI.ROUTE.UPDATE, AI.ROUTE, AI.ROUTE.REMOVE, AI.ROUTE.INFO handlers |

**Routing strategy:** FP32 brute-force cosine similarity for <=16 nodes (zero recall loss); HNSW O(log N) for >16 nodes. Per-node capacity limits, exclude filters. No accuracy or latency result is published with a harness yet.

---

## Speculative RAG Modules

When `--kvcache` is enabled, the speculative RAG modules are activated:

| Module | File | Role |
|---|---|---|
| SpeculativeRAG | `src/network/speculative_rag.mojo` | Per-session trajectory ring buffer (last 10 embeddings), momentum prediction, speculative HNSW cache |
| speculative commands | `src/commands/speculative.mojo` | RAG.SPECULATE.ENABLE, RAG.QUERY, RAG.SPECULATE.INFO handlers |

**Data flow (speculative RAG):**
```
RAG.SPECULATE.ENABLE session_id
    └── Creates per-session trajectory ring buffer

RAG.QUERY session_id query_embedding
    ├── Append embedding to trajectory ring buffer
    ├── Check speculative cache (cosine > 0.9 match against predictions)
    │     └── HIT: return pre-computed HNSW results
    ├── MISS: execute live HNSW search
    ├── Generate 3 predictions: current + alpha * (current - previous), alpha=[0.5, 1.0, 1.5]
    └── Pre-execute HNSW search for each prediction → store in speculative cache
```

