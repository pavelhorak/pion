# Networking: Event Loops & Protocol

Pion's networking layer provides five event loop configurations plus a zero-copy RESP parser. The backend is selected by CLI flag at startup.

---

## Event Loop Tiers

| Tier | Flag | Syscall model | Best for | Platforms |
|---|---|---|---|---|
| **XDP/AF_XDP** | `--xdp --xdp-iface eth0` | Kernel bypass: NIC→BPF→AF_XDP→UMEM | Tail latency at P=1 over a real NIC (needs flow steering), multi-worker ready | Linux 5.4+, CAP_NET_ADMIN |
| **io_uring SQPOLL** | `--sqpoll` | Kernel SQ polling thread (no enter() on hot path) | Experimental — hangs under load | Linux 5.11+, root/CAP_SYS_NICE |
| **io_uring** | `--iouring` (default on Linux) | Batched enter(): 2 syscalls per batch regardless of K fds | Multi-connection production (memtier, P>=10) | Linux 5.4+ |
| **epoll** | `--epoll` | epoll_wait + read + send per fd: 2K+1 syscalls per batch of K ready fds | Per-command P=1 benchmarks (w=1) | Linux 2.6+ |
| **kqueue** | (default on macOS) | kevent per tick | macOS development | macOS |

**Backend selection by CLI flag:** `--xdp` → XDP, `--sqpoll` → SQPOLL, `--iouring` → io_uring (Linux default), `--epoll` → epoll, no flag on macOS → kqueue. This is NOT a fallback chain — each backend is explicitly selected.

---

## kqueue (macOS)

`run_server_kqueue()` in `engine.mojo` — default on macOS. Uses `kevent()` per tick with `EVFILT_READ` for accept/recv and `EVFILT_WRITE` for EAGAIN flush. Stack-allocated `KEvent` via `stack_allocation[1, KEvent]()`.


---

## epoll (Linux, `--epoll`)

`run_server_epoll()` in `engine.mojo` — opt-in on Linux via `--epoll`. Uses `epoll_wait()` + per-fd `read()` + `send()`.

- Per-fd syscall model: 2K+1 syscalls per batch of K ready fds (1 epoll_wait + K reads + K sends)
- Best for P=1 w=1 benchmarks: 91-96K RPS, matching Redis/Valkey parity
- **Limitation at w>1:** connection drops at >=800 concurrent connections (inherent per-fd syscall ceiling). Use io_uring for high-connection multi-worker.

---

## io_uring (Linux, default)

`run_server_uring()` in `engine.mojo` — default on Linux. Wraps `io_uring_setup()`, SQE submission, CQE harvesting via C FFI (`src/ffi/uring_wrap.c`).

- SQE batch: submit accept + recv in one `io_uring_enter()` call — 2 syscalls per batch regardless of K fds
- CQE harvesting: drain completion ring each tick
- Best for multi-connection production workloads (memtier, P>=10)

### SQPOLL mode (`--sqpoll`)

**Experimental — hangs under load.** Kernel spawns a dedicated SQ polling thread that consumes SQEs from the submission ring without requiring `io_uring_enter()`. The Mojo event loop only calls `enter()` when:
1. The kernel thread goes idle (`IORING_SQ_NEED_WAKEUP` flag in `sq_flags`)
2. Waiting for CQEs (`min_complete > 0`)

Eliminates one syscall per event loop iteration on the hot path. Graceful fallback: if SQPOLL setup fails (requires root or CAP_SYS_NICE), retries without SQPOLL.

### Optional ring features (off by default)

Three switches change how the io_uring loop talks to the kernel. Each is also
turned on by an environment variable, so a test run can enable it without
editing anything. They are off until measurements decide otherwise.

| Flag | Environment | What it does | Kernel |
|---|---|---|---|
| `--iouring-defer` | `PION_IOURING_DEFER=1` | Creates the ring with `SINGLE_ISSUER` + `DEFER_TASKRUN`: completion work runs only inside `io_uring_enter`, not as task-work interrupts. Each worker owns and drives its ring alone, which is the precondition. | 6.1 (6.0 for `SINGLE_ISSUER` alone) |
| `--iouring-regfiles` | `PION_IOURING_REGFILES=1` | Registers the ring fd (`enter` names it by index) and a sparse file table: slot *n* holds the socket on fd *n* from accept until close, and SENDs name the slot (`IOSQE_FIXED_FILE`). | 5.18 for the ring fd |
| `--iouring-pbuf` | `PION_IOURING_PBUF=1` | Gives the multishot RECV's 256 × 16 KB buffers to the kernel through a provided-buffer ring (a buffer goes back with a few stores, not a `PROVIDE_BUFFERS` SQE), and parses received bytes in the provided buffer itself. | 5.19 |

Each feature is probed and falls back on its own, so an older kernel runs the
plain loop and never fails to start. The server logs what the kernel granted,
from worker 0 and from any worker that got less than it asked for:

```
io_uring features (worker 0): defer=ACTIVE(SINGLE_ISSUER+DEFER_TASKRUN) regfiles=ACTIVE(65536 slots) ring_fd=ACTIVE pbuf_ring=ACTIVE zero_copy_recv=ACTIVE
```

A feature the kernel refused says so with its errno, for example
`pbuf_ring=NOT_ACTIVE(errno 22; PROVIDE_BUFFERS)`.

Ubuntu's 6.8 kernels (from 6.8.0-139) invert the reserved-word check of
`IORING_REGISTER_PBUF_RING`: a correct call fails with `EINVAL`, and one with
a nonzero reserved word succeeds. The registration retries once that way after
an `EINVAL`. A correct kernel refuses a nonzero reserved word, so the retry
can only succeed on an affected one, and the line then reads
`pbuf_ring=ACTIVE(inverted-resv kernel)`.

The rules that keep them correct:

- **The loop enters on every pass, with `GETEVENTS`.** Under `DEFER_TASKRUN`
  that is the only place the kernel posts completions. A 1 ms timeout is
  always in flight, so the loop also ticks while no client sends anything.
  `DEFER_TASKRUN` and `--sqpoll` exclude each other; the probe then keeps
  `SINGLE_ISSUER` alone.
- **A registered slot is emptied before its fd is closed.** The file table
  holds its own reference to the socket. Left in place, the socket would
  outlive `close()`, and a SEND naming the slot would reach it after the fd
  number had gone to a new connection. Accept fills the slot before the first
  SEND.
- **Only SENDs use a registered slot.** A multishot RECV takes its file
  reference once, when it is armed, so a slot saves it nothing. On kernels
  before 6.13, a long-lived fixed-file request also holds back the release of
  every file removed from the table after it was issued.
- **Bytes are parsed in the provided buffer only when the connection holds
  nothing unfinished.** Whatever the drain leaves (a frame cut at the buffer's
  end, or commands behind one that parked) is copied into the connection's own
  buffer, and from there on the copy-and-accumulate path runs as before. The
  provided buffer goes back to the kernel once the drain returns, because
  nothing a command keeps outlives it: MULTI and blocking commands copy their
  frames, and replies are copied into the writer. The binary protocol lane
  always copies.

`tests/test_iouring_features.py` runs the same workloads with each switch and
all of them: concurrent pipelined connections whose frames straddle the 16 KB
buffers, a 1 MB value, a frame split across writes, parked BLPOP and XREAD
with commands behind them, MULTI/EXEC, connection churn with resets, and a
graceful stop.

---

## io_uring vs epoll Tradeoff

This is Pion's key networking design tradeoff on Linux:

- **io_uring** batches syscalls: 2 per batch (1 enter for submit + 1 enter for CQE wait) regardless of how many fds are ready. Wins at high concurrency.
- **epoll** uses per-fd syscalls: 2K+1 per batch of K ready fds. Simpler path per fd wins when K=1.

Which one wins at P=1 depends on the CPU, so measure both on yours. At pipeline depth,
io_uring's batching advantage grows with the number of ready connections.

### Measured throughput

The published head-to-head against Redis (io_uring, one machine, raw memtier output) is in
[`benchmarks/results/`](../benchmarks/results/README.md) and summarized in the
[README](../README.md#the-engine-underneath). No epoll-vs-io_uring comparison is published yet.

---

## XDP/eBPF Kernel Bypass (`--xdp`)

Zero-copy packet processing: BPF program filters at NIC driver level, redirects to AF_XDP socket in userspace.

### Architecture

```
NIC driver → XDP BPF filter (port match) → AF_XDP socket → UMEM shared mmap
  → Mojo poll_rx() → frame_ptr() into UMEM (zero-copy) → extract TCP payload
  → fast_path.process_data_plane() → ResponseWriter.buffer
  → send_data_ack() → build TCP response in UMEM TX frame → AF_XDP TX ring → NIC
```

### Components

| Component | File | Role |
|---|---|---|
| BPF filter | `src/ffi/xdp_kern.c` | Compiled to BPF bytecode, attaches to NIC driver. Matches TCP port → redirects to AF_XDP via XSKMAP |
| AF_XDP socket | `src/ffi/xdp_wrap.c` | UMEM setup (16MB shared mmap, 4096×4KB frames), fill/completion/RX/TX ring management |
| TCP-Lite | `src/io/xdp.mojo` | `XDPEngine` with `TCPConnection` table (65536 entries, indexed by source port) |
| Event loop | `src/network/engine.mojo` | `run_server_xdp()` — polls AF_XDP RX ring in batches of 64 frames |

### TCP-Lite state machine

Minimal TCP for controlled internal links: SYN→SYN+ACK, ACK, DATA+PSH→ACK+response, FIN→FIN+ACK, RST. Connection table: 65536 entries indexed by client source port. No retransmission, congestion control, or OOO reassembly (assumes reliable internal network). IP/TCP checksums computed in C helpers.

### Implementation notes

1. **Multi-frame TX fragmentation:** `pion_xdp_send_fragmented()` in `xdp_wrap.c` splits responses >4042B into multiple TCP segments with proper seq/ack continuation. Handles TX ring exhaustion with kick+drain+retry.
2. **Shared XSKMAP for multi-worker:** BPF program + XSKMAP created once before worker spawn via `pion_xdp_create_shared_xskmap()` and `pion_xdp_attach_bpf()`. Each worker calls `pion_xdp_create_worker()` to create its own AF_XDP socket + UMEM and register in the shared map.
3. **Automatic flow steering:** `pion_xdp_setup_flow_steering()` runs `ethtool -N` at startup to direct target port traffic to the correct RX queue. Rules cleaned up on exit.
4. **SIGINT/SIGTERM signal handlers:** `pion_xdp_register_signal_handlers()` installs `sigaction` handlers that detach BPF from NIC via netlink (IFLA_XDP_FD=-1) and remove flow steering rules before exit.
5. **Loopback TCP fallback:** TCP listener is polled every tick in the XDP event loop (non-blocking accept + recv/send). Supports up to 256 concurrent loopback clients (redis-cli, health checks). Loopback traffic bypasses the NIC so XDP can't see it.

### Requirements

Linux 5.4+, CAP_NET_ADMIN (or root), AF_XDP-capable NIC driver, `--security-opt seccomp=unconfined` in Docker.

### CLI

```bash
./pion-server --xdp --xdp-iface eth0 -w 1        # Single-worker XDP
./pion-server --xdp --xdp-iface eth0 -w 10 --independent-workers   # Multi-worker XDP (shared XSKMAP)
./pion-server --sqpoll -w 10 --independent-workers   # io_uring SQPOLL
```

---

## RESP Parser

### Fast Path (`fast_path.mojo`)

Zero-allocation dispatch for 34 hot commands. `b0_lower = b0 | 0x20` for case-insensitive matching, ordered by frequency (GET first). Returns `consumed > 0` on success, `0` for slow-path fallback.

### Slow Path (`slow_path.mojo`)

RESP3 tokenizer over a per-handler, heap-resident token table (up to 2048 tokens per command; longer commands get a clean `-ERR`). `cmd_matches_N()` byte dispatch (no `.upper()`). Handles FT.*, AI.*, CONFIG, INFO, and all commands not yet promoted to fast path.

### Response Writer (`response_writer.mojo`)

Pre-allocated 4 MB response buffer per worker. `append_*_response()` methods write directly to the buffer, and `flush_response()` sends it. When the socket cannot take everything, the rest waits in the connection's 4 MB pending block (allocated on first EAGAIN) and, past that, in an overflow queue with no size limit (#49); kqueue and epoll drain it on the write event, io_uring on each SEND completion. A reply that outgrows the response buffer is handed to the connection as the buffer fills, so no reply is ever cut short, and a slow reader never blocks the worker. A subscriber or MONITOR connection owed more than 32 MB is disconnected, as Redis's pubsub limit does.

---

## Binary Protocol (externalized attention, port 1975)

When `--kvcache` is enabled, Pion listens on a second port (main_port + 1, default 1975) for a binary wire protocol optimized for bulk tensor operations.

### Frame format

```
[magic:2B = 0xCA5E][cmd:1B][body_len:4B LE][body...]
```

### Commands

Opcodes (from the `comptime CMD_*` block in `src/network/binary_protocol.mojo`):

| Byte | Command | Description |
|:---:|---|---|
| 0x01 | KV.STORE | Store a KV-prefix blob |
| 0x02 | KV.FETCH | Fetch a KV-prefix blob |
| 0x10 | LAYER.STORE | Store per-layer tensor blob |
| 0x11 | LAYER.FETCH | Fetch per-layer tensor blob |
| 0x12 | LAYER.FETCH_BATCH | Fetch several layers in one frame |
| 0x13 | LAYER.EXTEND | Append tokens to a stored layer |
| 0x20 | ATTEND.CREATE | Create attention session (key_dim, value_dim) |
| 0x21 | ATTEND.STORE | Stage token KV pairs |
| 0x22 | ATTEND.FINALIZE | Batch build HNSW index from staged keys |
| 0x23 | ATTEND.QUERY | Top-k HNSW search |
| 0x24 | ATTEND.PREFIX.QUERY_FUSED | Fused sparse-mask prefix query |
| 0x31–0x36 | MOE.EXPERT.{FETCH,PREFETCH,PIN,UNPIN,INFO,STATS} | MoE expert paging |
| 0x37 | AUTH | Authenticate the binary connection (under `--requirepass`) |
| 0xFF | PING | Binary keepalive |

Connections on port 1975 are routed to the binary handler via `local_affinity[fd] == 2`. The binary protocol avoids RESP parsing overhead entirely for bulk attention data.

RESP-based equivalents (KV.STORE, KV.FETCH, KV.INFO, ATTEND.*) are also available on port 1974 for compatibility.

### Python client

`vllm-pion/` package provides `PionKVClient`, `PionAttentionClient`, and `ExternalizedAttentionLayer` for integration with vLLM/MLX inference engines.

---

## Socket Configuration

- `TCP_NODELAY` on accepted sockets (not listening socket)
- Shared listen fd across workers (no `SO_REUSEPORT`); workers compete for `accept()`
- Kernel backlog: 65535 (queues connections during startup hash map initialization)
- Non-blocking sockets (`O_NONBLOCK`) on all client connections
