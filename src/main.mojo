import std.sys as sys
from std.sys.info import CompilationTarget
from std.collections import List
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from src.common.ptr import is_not_null

from src.common.config import PionConfig
from src.common.version import PION_VERSION, PION_BUILD_SHA, PION_BUILD_DATE
from src.vector.vector_abi import vector_backend_line, vector_abi_ok, vector_lib_abi, VECTOR_ABI_VERSION
from src.common.utils import set_thread_affinity, set_thread_qos_user_interactive, parse_memory_value
from src.memory.slab_allocator import SlabAllocator

from src.engine.state import Pion
from src.vector.hnsw import SharedHNSWView, HNSWGraph
from src.network.server import create_listen_socket
from src.common.lock_free import ShardQueryBus
from src.network.cluster import ClusterState
from src.network.slow_path import SlowPathHandler
from src.network.v_store import VStoreDirectory
from src.commands.tenant import tenant_arg_error
from std.memory import unsafe_memset

def _path_exists(path: String) -> Bool:
    """access(path, F_OK) — probe without opening (gh #176 sidecar path resolve)."""
    var p = path
    return external_call["access", Int32](p.as_c_string_slice(), Int32(0)) == 0


def _exe_dir() -> String:
    """Directory of argv[0], with trailing slash, or "" for a bare PATH lookup."""
    var a0 = String(sys.argv()[0])
    var slash = -1
    for ci in range(a0.byte_length()):
        if a0.as_bytes()[ci] == 47:  # '/'
            slash = ci
    if slash < 0:
        return String("")
    # Startup-only cold path: byte-append beats relying on String slice syntax
    # that b2 does not guarantee.
    var out = String("")
    for cj in range(slash + 1):
        out += chr(Int(a0.as_bytes()[cj]))
    return out


def _resolve_sidecar(mut py: String, mut script: String) -> Bool:
    """Can the PyTorch inference sidecar actually be started? (gh #176, gh #282)

    Both paths are repo-relative, so a server started from anywhere else — a
    test's /tmp workdir, or the release tarball, which contains no
    src/inference at all — cannot exec them. gh #176 added the
    executable-relative retry; gh #282 is the case where NEITHER resolves.

    That case used to fork anyway: exec died on ENOENT, and the connect-poll
    then burned its full 30 SECONDS before disabling inference. On the release
    tarball that is 30 s spent on something that cannot succeed, with the
    listen socket not yet open — so the README's next command answers
    "Connection refused" and the reader concludes the binary is broken.

    Returns True and rewrites both paths on success; False means do not spawn.
    """
    if _path_exists(py) and _path_exists(script):
        return True
    var d = _exe_dir()
    if d.byte_length() > 0:
        var cand_py = d + py
        var cand_sc = d + script
        if _path_exists(cand_py) and _path_exists(cand_sc):
            py = cand_py
            script = cand_sc
            return True
    return False


def _try_auto_detect_ollama(mut config: PionConfig):
    """Probe Ollama at 127.0.0.1:11434/api/tags.
    If nomic-embed-text is available, auto-enable semantic cache + AI.COMPLETE.
    Silent no-op if Ollama is not running."""
    var chost = String("127.0.0.1")
    var fd = external_call["pion_connect_tcp", Int32](chost.as_c_string_slice(), Int32(11434))
    if fd < 0:
        return  # Ollama not running — skip silently

    var req = String("GET /api/tags HTTP/1.0\r\nHost: 127.0.0.1:11434\r\nConnection: close\r\n\r\n")
    _ = external_call["pion_write", Int64](fd, req.unsafe_ptr(), req.byte_length())

    var buf = alloc[UInt8](65536)
    var total_n = 0
    var nr: Int64 = 1
    while nr > 0 and total_n < 65535:
        nr = external_call["pion_read", Int64](fd, buf.unsafe_offset(total_n), 65535 - total_n)
        if nr > 0: total_n += Int(nr)
    _ = external_call["close", Int32](fd)

    if total_n == 0:
        buf.unsafe_free()
        return

    # Scan for "nomic-embed-text" in response body
    var embed_tag = String("nomic-embed-text")
    var etag_ptr = embed_tag.unsafe_ptr()
    var etag_len = embed_tag.byte_length()
    var found_embed = False
    for si in range(total_n - etag_len):
        var match_ok = True
        for ti in range(etag_len):
            if buf[unsafe_offset=si + ti] != etag_ptr[unsafe_offset=ti]: match_ok = False; break
        if match_ok: found_embed = True; break

    buf.unsafe_free()

    if found_embed:
        config.embedding.enabled = True
        config.llm.enabled = True
        config.llm.port = 11434
        config.llm.model = "llama3.2:1b"   # Ollama default; override with --llm-model
        print("Ollama auto-detected: nomic-embed-text found")
        print("  Semantic cache + AI.COMPLETE enabled (--no-auto-detect to disable)")


def _parse_peer_nodes(cluster: Pointer[ClusterState, MutUntrackedOrigin], peer_str: String, my_host: String, my_port: Int):
    """Parse comma-separated 'host:port,...' peer list and compute slot ranges.
    Identifies self in list, assigns evenly-split slot ranges to all nodes,
    fills cluster peers[] for non-self entries."""
    # Count nodes
    var n_nodes = 1
    for i in range(peer_str.byte_length()):
        if peer_str.unsafe_ptr()[unsafe_offset=i] == UInt8(44):  # ','
            n_nodes += 1

    var slots_per_node = 16384 // n_nodes
    var self_idx = -1

    # Parse each "host:port" entry
    var node_idx = 0
    var seg_start = 0
    var peer_fill = 0

    for ci in range(peer_str.byte_length() + 1):
        var is_end = (ci == peer_str.byte_length())
        var is_comma = (not is_end) and peer_str.unsafe_ptr()[unsafe_offset=ci] == UInt8(44)
        if is_end or is_comma:
            var seg_len = ci - seg_start
            if seg_len > 0:
                # Find ':' in this segment
                var colon_pos = -1
                for si in range(seg_len):
                    if peer_str.unsafe_ptr()[unsafe_offset=seg_start + si] == UInt8(58):  # ':'
                        colon_pos = si
                        break
                if colon_pos > 0:
                    var h_len = colon_pos
                    var p_start = seg_start + colon_pos + 1
                    var p_len = seg_len - colon_pos - 1
                    # Parse port
                    var port_val = 0
                    for pi in range(p_len):
                        var b = peer_str.unsafe_ptr()[unsafe_offset=p_start + pi]
                        if b >= 48 and b <= 57:
                            port_val = port_val * 10 + Int(b - 48)

                    # Compute slot range for this node
                    var slot_start = node_idx * slots_per_node
                    var slot_end = (node_idx + 1) * slots_per_node - 1
                    if node_idx == n_nodes - 1:
                        slot_end = 16383

                    # Check if this is self
                    var is_self = (port_val == my_port)
                    if is_self and h_len == my_host.byte_length():
                        # Verify host matches
                        var host_match = True
                        for hi in range(h_len):
                            if peer_str.unsafe_ptr()[unsafe_offset=seg_start + hi] != my_host.unsafe_ptr()[unsafe_offset=hi]:
                                host_match = False
                                break
                        if host_match:
                            self_idx = node_idx

                    if is_self and self_idx == node_idx:
                        cluster[].my_slot_start = slot_start
                        cluster[].my_slot_end = slot_end
                    else:
                        # Add as peer using flat array API
                        if peer_fill < 16:
                            var tmp_host_buf = alloc[UInt8](h_len + 1)
                            for hi in range(h_len):
                                tmp_host_buf[unsafe_offset=hi] = peer_str.unsafe_ptr()[unsafe_offset=seg_start + hi]
                            tmp_host_buf[unsafe_offset=h_len] = 0
                            cluster[].set_peer(peer_fill, tmp_host_buf, h_len, port_val, slot_start, slot_end)
                            tmp_host_buf.unsafe_free()
                            peer_fill += 1
            node_idx += 1
            seg_start = ci + 1

    cluster[].peer_count = peer_fill


# gh #372: every flag the parser in main() accepts. The parser, not this list,
# decides what a flag DOES — the list only lets a refusal say "did you mean"
# and tell a value flag given no value apart from an unknown one.
# tests/test_gh372_unknown_flags.py checks both lists against the parser.
def _known_flags() -> List[String]:
    return [
        "-p", "--port", "-w", "--workers", "--independent-workers", "-G", "--ai-gateway",
        "-M", "--max-engine", "--huge-pages", "--no-huge-pages", "--affinity", "--no-affinity",
        "--cluster", "--cluster-host", "--cluster-nodes", "--cluster-replica",
        "--cluster-primary-host", "--cluster-primary-port", "--gossip-ping-ms", "--flare",
        "--emb-enabled", "--emb-model", "--emb-host", "--emb-port", "--emb-query-prefix",
        "--emb-doc-prefix", "--emb-dim", "--llm-enabled", "--llm-port", "--llm-model",
        "--inference", "--inference-socket", "--inference-emb-model", "--inference-llm-model",
        "--metal-attention", "--metal-attention-fp16", "--fa-window", "--cuda-attention",
        "--no-auto-detect", "--auto-embed", "--no-auto-embed", "--nle-embed", "--profile",
        "--gpu", "--sqpoll", "--polarquant", "--turboquant", "--nanoquant", "--xdp",
        "--xdp-interface", "--xdp-iface", "--kvcache", "--no-wal", "--wal-size",
        "--wal-max-segments", "--wal-full-policy", "--blob-threshold", "--no-blob-tier",
        "--iouring", "--epoll", "--ns-prefix", "--requirepass", "--requirepass-file", "--bind",
        "--tenant", "--moe-cache", "--moe-cache-mib", "--dim", "--max-elements", "--crash-log",
        "--status-file", "--no-crash-log", "--rss-warn-pct", "--maxmemory",
        "--lua-time-limit", "--lua-memory-limit",
        "--help", "-h", "--version", "-v",
    ]


def _value_flags() -> List[String]:
    return [
        "-p", "--port", "-w", "--workers", "--cluster-host", "--cluster-nodes",
        "--cluster-primary-host", "--cluster-primary-port", "--gossip-ping-ms", "--emb-model",
        "--emb-host", "--emb-port", "--emb-query-prefix", "--emb-doc-prefix", "--emb-dim",
        "--llm-port", "--llm-model", "--inference-socket", "--inference-emb-model",
        "--inference-llm-model", "--fa-window", "--profile", "--xdp-interface", "--xdp-iface",
        "--wal-size", "--wal-max-segments", "--wal-full-policy", "--blob-threshold", "--ns-prefix",
        "--requirepass", "--requirepass-file", "--bind", "--tenant", "--moe-cache",
        "--moe-cache-mib", "--dim", "--max-elements", "--crash-log", "--status-file",
        "--rss-warn-pct", "--maxmemory", "--lua-time-limit", "--lua-memory-limit",
    ]


def _edit_distance(a: String, b: String) -> Int:
    """Levenshtein distance over bytes — flags are ASCII."""
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var n = a.byte_length()
    var m = b.byte_length()
    var prev = List[Int](capacity=m + 1)
    for j in range(m + 1):
        prev.append(j)
    for i in range(1, n + 1):
        var cur = List[Int](capacity=m + 1)
        cur.append(i)
        for j in range(1, m + 1):
            var cost = 0 if ap[unsafe_offset=i - 1] == bp[unsafe_offset=j - 1] else 1
            cur.append(min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost))
        prev = cur^
    return prev[m]


def _refuse_arg(arg: String, why: String):
    """gh #372: a command line the parser did not understand must not start a
    server. Every unrecognised argument used to be skipped in silence, so a
    typo on --requirepass started an UNAUTHENTICATED server and a typo on
    --wal-size ran the default — both looking perfectly healthy."""
    var msg = "FATAL: " + why + " '" + arg + "'"
    if arg.startswith("-"):
        var best = String("")
        var best_d = 1000
        var known = _known_flags()
        for k in range(len(known)):
            var d = _edit_distance(arg, known[k])
            if d < best_d:
                best_d = d
                best = known[k]
        # Suggest only a near miss: a third of the flag's length, at least 2.
        if best_d <= max(2, arg.byte_length() // 3):
            msg += " (did you mean " + best + "?)"
    print(msg)
    print("       run with --help for the list of flags")
    external_call["exit", NoneType](Int32(1))


def _is_value_flag(arg: String) -> Bool:
    var vf = _value_flags()
    for k in range(len(vf)):
        if arg == vf[k]:
            return True
    return False


def _print_help():
    print("pion-server " + PION_VERSION + "+" + PION_BUILD_SHA + " (" + PION_BUILD_DATE + ")")
    print("Wire-compatible Redis/Valkey + native vector + AI substrate, in Mojo.")
    print("")
    print("Usage: pion-server [flags]")
    print("")
    print("Server")
    print("  -p, --port N              listen port (default 1974; binary protocol on N+1 with --kvcache)")
    print("  -w, --workers N           OS-thread workers (default 1; capped at 4 on Apple Silicon).")
    print("                            N > 1 gives each worker a PRIVATE keyspace and requires")
    print("                            --independent-workers — see below.")
    print("      --independent-workers acknowledge that -w N > 1 runs N independent keyspaces:")
    print("                            a connection is bound to whichever worker won accept(), so a")
    print("                            write acked on one connection is invisible to a read on")
    print("                            another. Safe only when every client pins one connection")
    print("                            (or shards keys itself). Pooled clients will read stale nils.")
    print("      --profile PROF        kv | vector | ai | full — KV-only / +HNSW / +AI / everything")
    print("      --ns-prefix STR       multi-tenant namespace gate for KV.PREFIX.* / V.* (single-tenant if empty)")
    print("      --requirepass STR     require AUTH <password> before serving any command (no auth if empty)")
    print("                            NOTE: visible in `ps`; prefer the two below")
    print("      --requirepass-file P  read the password from file P (trailing newline stripped)")
    print("                            or set PION_REQUIREPASS in the environment")
    print("      --tenant NAME=PW      per-tenant password; key-prefixes NAME's keyspace")
    print("                            (repeatable; requires --requirepass) [gh #101]")
    print("      --bind ADDR           interface to listen on (default 127.0.0.1;")
    print("                            0.0.0.0 when a password is set). Applies to the")
    print("                            RESP port, port+1, replication and gossip.")
    print("      --no-wal              disable WAL append (benchmark only; SIGKILL loses writes)")
    print("      --wal-size MB         WAL segment size (default 256); rotates instead of dropping")
    print("      --wal-max-segments N  sealed WAL segments to keep (default 32); 0 = never rotate")
    print("      --wal-full-policy P   refuse (default) | drop — what to do with keyspace")
    print("                            writes once the WAL is full. refuse replies -MISCONF")
    print("                            rather than ACKing writes that cannot be persisted.")
    print("      --blob-threshold N    values >= N bytes go to the file-backed blob tier (default 1048576)")
    print("      --no-blob-tier        keep large values on the anonymous heap (the pre-blob-tier behaviour)")
    print("      --huge-pages / --no-huge-pages    request 2 MB pages (Linux)")
    print("      --affinity / --no-affinity        pin workers to cores")
    print("  -v, --version             print version and exit")
    print("  -h, --help                this help")
    print("")
    print("Diagnostics")
    print("      --crash-log PATH      start/exit breadcrumb log (default pion-<port>.crash.log)")
    print("      --status-file PATH    1 Hz liveness record: pid/uptime/RSS/ticks (default pion-<port>.status)")
    print("      --no-crash-log        disable both breadcrumb files")
    print("      --rss-warn-pct N      warn once when RSS crosses N% of physical RAM (default 70; >100 off)")
    print("      --maxmemory SIZE      refuse memory-growing writes (-OOM) above this RSS; bytes,")
    print("                            k/kb/m/mb/g/gb or N% of RAM (default 0 = off; no eviction)")
    print("      --lua-time-limit MS   stop a script that runs longer without writing (default 5000;")
    print("                            0 = never). A worker cannot answer SCRIPT KILL mid-script.")
    print("      --lua-memory-limit SIZE  Lua heap cap per worker state (default 1gb; 0 = none)")
    print("      (supervised serving with auto-restart: scripts/pion-supervise.sh -- <server args>)")
    print("")
    print("I/O backend")
    print("      --epoll               epoll (Linux; best at P=1 / w=1 benchmarks)")
    print("      --iouring             io_uring (Linux default, best for production multi-connection)")
    print("      --sqpoll              io_uring SQPOLL — kernel-side polling, eliminates enter() syscall")
    print("      --xdp                 AF_XDP zero-copy kernel bypass (Linux 5.4+, P=1 only)")
    print("      --xdp-iface IF        NIC for XDP attach (default eth0; alias: --xdp-interface)")
    print("")
    print("Vector")
    print("      --dim N               embedding dimension (default 1536)")
    print("      --max-elements N      HNSW capacity per worker (default 600,000)")
    print("      --polarquant          block-INT4 PolarQuant (recall ~0.96, -17% QPS vs INT8)")
    print("      --turboquant          block-INT3 + QJL TurboQuant (recall ~0.95, -39% QPS vs INT8)")
    print("      --nanoquant           block-INT2 NanoQuant (EXPERIMENTAL: recall ~0.46)")
    print("      --gpu                 enable GPU search path (Metal on macOS)")
    print("")
    print("Externalized attention / KV cache (Mac)")
    print("      --kvcache             enable KV.PREFIX.* / ATTEND.* / V.* / AI.ROUTE.* / RAG.* + binary protocol on port+1")
    print("      --moe-cache PATH      enable MOE.EXPERT.* tier (path to MoE safetensors directory)")
    print("      --moe-cache-mib N     MoE tier LRU cache budget in MiB (default 1024)")
    print("      --metal-attention     native Metal SDPA, fp32 (Apple Silicon, no Python sidecar)")
    print("      --metal-attention-fp16   fp16 kernel — matches vanilla mlx-lm precision")
    print("      --cuda-attention      native CUDA SDPA (Linux GPU). Routes ATTEND.PREFIX.{STORE,QUERY,QUERY_SPARSE,QUERY_SPARSE_AUTO} through cuda_kernels.cu")
    print("      --fa-window N         sliding-window SDPA: scan only last N tokens; 0 = full attention. Safe on Mistral SWA / Longformer; lossy on dense Llama / Gemma / GPT.")
    print("")
    print("AI gateway / inference / embedding")
    print("  -G, --ai-gateway          enable AI gateway (semantic cache + AI.* surface)")
    print("  -M, --max-engine          enable MAX engine path")
    print("      --flare               FLARE mid-generation retrieval (caps to -w 1)")
    print("      --emb-enabled         enable external embedding backend (default Ollama at 11434)")
    print("      --emb-model NAME      external embedding model (default nomic-embed-text)")
    print("      --emb-host HOST       external embedding host (default 127.0.0.1)")
    print("      --emb-port N          external embedding port (default 11434)")
    print("      --emb-dim N           external embedding width (default 768; MUST match the model)")
    print("      --emb-query-prefix S  instruction prefix for query embeds (asymmetric retrievers)")
    print("      --emb-doc-prefix S    instruction prefix for FT.ADDTEXT document embeds")
    print("      --llm-enabled         enable external LLM backend (MAX Serve / OpenAI-compat)")
    print("      --llm-port N          LLM backend port (default 8000)")
    print("      --llm-model NAME      LLM model id")
    print("      --inference           start in-process inference sidecar (PyTorch MiniLM-L6-v2)")
    print("      --inference-socket P  socket path (default /tmp/pion-<uid>-<port>.inference.sock)")
    print("      --inference-emb-model NAME    embedding model for sidecar")
    print("      --inference-llm-model NAME    LLM model for sidecar")
    print("      --auto-embed          auto-launch the embedding sidecar (default on)")
    print("      --no-auto-embed       disable the auto sidecar (use Ollama or --emb-enabled)")
    print("      --no-auto-detect      skip Ollama auto-probe at startup")
    print("      --nle-embed           macOS Apple NLEmbedding (512-dim, no Python)")
    print("")
    print("Cluster / replication")
    print("      --cluster                     enable cluster mode (gossip + CLUSTER NODES)")
    print("      --cluster-host HOST           advertised host for this node")
    print("      --cluster-nodes LIST          comma-separated host:port,... (including self)")
    print("      --cluster-replica             run as replica of --cluster-primary-*")
    print("      --cluster-primary-host HOST   primary's advertised host")
    print("      --cluster-primary-port N      primary's main port (replication uses primary_port + 10000)")
    print("      --gossip-ping-ms N            gossip PING interval (default 1000)")
    print("")
    print("More: README.md, doc/configuration.md, doc/operations.md.")


def main():
    var args = sys.argv()

    # --help: print and exit immediately
    for i in range(1, len(args)):
        if args[i] == "--help" or args[i] == "-h":
            _print_help()
            return

    # --version: print and exit immediately
    for i in range(1, len(args)):
        if args[i] == "--version" or args[i] == "-v":
            print("pion-server " + PION_VERSION + "+" + PION_BUILD_SHA + " (" + PION_BUILD_DATE + ")")
            print("vector: " + vector_backend_line())
            return

    # D11: a vendored library built against a different view layout would read
    # every field at the wrong offset — refuse to start rather than serve that.
    if not vector_abi_ok():
        print("FATAL: libpion_vector ABI mismatch (library reports "
              + String(vector_lib_abi()) + ", engine expects "
              + String(VECTOR_ABI_VERSION) + ") — rebuild or re-vendor the library")
        external_call["exit", NoneType](Int32(1))

    var config = PionConfig()
    var lua_time_limit_ms = 5000         # #36: --lua-time-limit
    var lua_memory_limit = 1 << 30       # #36: --lua-memory-limit

    var auto_detect = True   # auto-probe Ollama unless --no-auto-detect is passed
    var auto_embed = True    # A3: auto-launch embedding sidecar when no external server available
    var nle_embed = False    # macOS Apple NLEmbedding (no PyTorch sidecar, 512-dim)
    var inference_socket_explicit = False  # gh #424: keep the per-port default unless overridden
    var i = 1
    while i < len(args):
        if (args[i] == "-p" or args[i] == "--port") and i + 1 < len(args):
            try:
                config.server.port = atol(args[i+1])
                i += 2
            except:
                _refuse_arg(String(args[i+1]), "invalid port")   # gh #372
        elif (args[i] == "-w" or args[i] == "--workers") and i + 1 < len(args):
            try:
                config.server.workers = atol(args[i+1])
                i += 2
            except:
                _refuse_arg(String(args[i+1]), "invalid worker count")   # gh #372
        elif args[i] == "--independent-workers":
            config.server.independent_workers = True
            i += 1
        elif args[i] == "-G" or args[i] == "--ai-gateway":
            config.ai.enable_gateway = True
            i += 1
        elif args[i] == "-M" or args[i] == "--max-engine":
            config.ai.enable_max_engine = True
            i += 1
        elif args[i] == "--huge-pages":
            config.server.use_huge_pages = True
            i += 1
        elif args[i] == "--no-huge-pages":
            config.server.use_huge_pages = False
            i += 1
        elif args[i] == "--affinity":
            config.server.strict_affinity = True
            i += 1
        elif args[i] == "--no-affinity":
            config.server.strict_affinity = False
            i += 1
        elif args[i] == "--cluster":
            config.cluster.enabled = True
            i += 1
        elif (args[i] == "--cluster-host") and i + 1 < len(args):
            config.cluster.my_host = args[i+1]
            i += 2
        elif (args[i] == "--cluster-nodes") and i + 1 < len(args):
            config.cluster.peer_nodes = args[i+1]
            i += 2
        elif args[i] == "--cluster-replica":
            config.cluster.is_replica = True
            i += 1
        elif (args[i] == "--cluster-primary-host") and i + 1 < len(args):
            config.cluster.primary_host = args[i+1]
            i += 2
        elif (args[i] == "--cluster-primary-port") and i + 1 < len(args):
            try:
                config.cluster.primary_port = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif (args[i] == "--gossip-ping-ms") and i + 1 < len(args):
            try:
                config.cluster.gossip_ping_ms = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--flare":
            # Enable FLARE: embedding + LLM both via Ollama on port 11434
            config.embedding.enabled = True
            config.llm.enabled = True
            config.llm.port = 11434
            config.llm.model = "llama3.1:8b"
            i += 1
        elif args[i] == "--emb-enabled":
            config.embedding.enabled = True
            i += 1
        elif (args[i] == "--emb-model") and i + 1 < len(args):
            # gh #140: the external embed backend was pinned to Ollama's
            # nomic-embed-text at 768-dim with no way to point it elsewhere, so
            # a stronger retriever could not be selected without a rebuild.
            config.embedding.model = args[i+1]
            config.embedding.enabled = True
            i += 2
        elif (args[i] == "--emb-host") and i + 1 < len(args):
            config.embedding.host = args[i+1]
            config.embedding.enabled = True
            i += 2
        elif (args[i] == "--emb-port") and i + 1 < len(args):
            try:
                config.embedding.port = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            config.embedding.enabled = True
            i += 2
        elif (args[i] == "--emb-query-prefix") and i + 1 < len(args):
            config.embedding.query_prefix = args[i+1]
            config.embedding.enabled = True
            i += 2
        elif (args[i] == "--emb-doc-prefix") and i + 1 < len(args):
            config.embedding.doc_prefix = args[i+1]
            config.embedding.enabled = True
            i += 2
        elif (args[i] == "--emb-dim") and i + 1 < len(args):
            # Must match the backend exactly: SemanticCache.embed_into rejects a
            # vector whose width differs and falls through the cascade silently.
            try:
                config.embedding.dimensions = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            config.embedding.enabled = True
            i += 2
        elif args[i] == "--llm-enabled":
            config.llm.enabled = True
            i += 1
        elif (args[i] == "--llm-port") and i + 1 < len(args):
            try:
                config.llm.port = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            config.llm.enabled = True
            i += 2
        elif (args[i] == "--llm-model") and i + 1 < len(args):
            config.llm.model = args[i+1]
            i += 2
        elif args[i] == "--inference":
            config.inference.enabled = True
            i += 1
        elif (args[i] == "--inference-socket") and i + 1 < len(args):
            config.inference.socket_path = args[i+1]
            config.inference.enabled = True
            inference_socket_explicit = True
            i += 2
        elif (args[i] == "--inference-emb-model") and i + 1 < len(args):
            config.inference.default_embedding_model = args[i+1]
            config.inference.enabled = True
            i += 2
        elif (args[i] == "--inference-llm-model") and i + 1 < len(args):
            config.inference.default_llm_model = args[i+1]
            config.inference.enabled = True
            i += 2
        elif args[i] == "--metal-attention":
            config.metal_attention.enabled = True
            i += 1
        elif args[i] == "--metal-attention-fp16":
            config.metal_attention.enabled = True
            config.metal_attention.fp16 = True
            i += 1
        elif (args[i] == "--fa-window") and i + 1 < len(args):
            try:
                var w = atol(args[i+1])
                config.metal_attention.fa_window = w
                # gh #9: --fa-window also flows to the CUDA engine; either side
                # can be enabled (mac vs linux build) and they share the flag.
                config.cuda_attention.fa_window = w
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--cuda-attention":
            config.cuda_attention.enabled = True
            i += 1
        elif args[i] == "--no-auto-detect":
            auto_detect = False
            i += 1
        elif args[i] == "--auto-embed":
            auto_embed = True
            i += 1
        elif args[i] == "--no-auto-embed":
            auto_embed = False
            i += 1
        elif args[i] == "--nle-embed":
            nle_embed = True
            auto_embed = False  # NLE replaces the PyTorch sidecar; don't spawn it
            i += 1
        elif (args[i] == "--profile") and i + 1 < len(args):
            # gh #372: apply_profile ignores a name it does not know, so a typo
            # ran the smart default profile instead of the one asked for.
            if not (args[i+1] == "kv" or args[i+1] == "vector" or args[i+1] == "ai" or args[i+1] == "full"):
                _refuse_arg(String(args[i+1]), "invalid --profile (expected kv|vector|ai|full):")
            config.apply_profile(args[i+1])
            i += 2
        elif args[i] == "--gpu":
            config.vector.has_gpu = True
            i += 1
        elif args[i] == "--sqpoll":
            config.server.use_sqpoll = True
            i += 1
        elif args[i] == "--polarquant":
            config.vector.polarquant = True
            i += 1
        elif args[i] == "--turboquant":
            config.vector.turboquant = True
            i += 1
        elif args[i] == "--nanoquant":
            config.vector.nanoquant = True
            i += 1
        elif args[i] == "--xdp":
            config.server.use_xdp = True
            i += 1
        elif (args[i] == "--xdp-interface" or args[i] == "--xdp-iface") and i + 1 < len(args):
            config.server.use_xdp = True
            config.server.xdp_interface = args[i+1]
            i += 2
        elif args[i] == "--kvcache":
            config.server.kvcache_enabled = True
            i += 1
        elif args[i] == "--no-wal":
            config.server.no_wal = True
            i += 1
        elif args[i] == "--wal-size" and i + 1 < len(args):
            try:
                config.server.wal_size_mb = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--wal-max-segments" and i + 1 < len(args):
            try:
                config.server.wal_max_segments = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--wal-full-policy" and i + 1 < len(args):
            # gh #260. Anything other than an exact "drop" keeps the safe
            # default: a typo must not silently disable durability enforcement.
            if args[i+1] == "drop":
                config.server.wal_refuse_when_full = False
            elif args[i+1] == "refuse":
                config.server.wal_refuse_when_full = True
            else:
                _refuse_arg(String(args[i+1]), "invalid --wal-full-policy (expected refuse|drop):")   # gh #372
            i += 2
        elif args[i] == "--blob-threshold" and i + 1 < len(args):
            try:
                config.server.blob_threshold = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--no-blob-tier":
            config.server.no_blob_tier = True
            i += 1
        elif args[i] == "--iouring":
            config.server.use_iouring = True
            i += 1
        elif args[i] == "--epoll":
            config.server.use_epoll = True
            i += 1
        elif args[i] == "--ns-prefix" and i + 1 < len(args):
            # Multi-tenant isolation: KV.PREFIX.* must use this prefix.
            config.server.ns_prefix = args[i + 1]
            i += 2
        elif args[i] == "--requirepass" and i + 1 < len(args):
            # gh #100 (C2): require AUTH <password> before serving any command.
            # gh #258: this spelling puts the password in argv, where `ps` and
            # /proc/<pid>/cmdline expose it to every local user. Warn and point
            # at the alternatives rather than silently accepting it.
            config.server.requirepass = args[i + 1]
            print("WARNING: --requirepass puts the password in this process's")
            print("  command line, visible to any local user via `ps`. Prefer")
            print("  --requirepass-file <path> or the PION_REQUIREPASS env var.")
            i += 2
        elif args[i] == "--requirepass-file" and i + 1 < len(args):
            # gh #258: read the password from a file instead of argv.
            var _sbuf = alloc[UInt8](512)
            var _pathz = args[i + 1] + "\0"
            var _slen = Int(external_call["pion_read_secret_file", Int32](
                _pathz.unsafe_ptr(), _sbuf, Int32(512)))
            if _slen == -1:
                print("FATAL: --requirepass-file: cannot read " + args[i + 1])
                _sbuf.unsafe_free()
                external_call["exit", NoneType](Int32(1))
            elif _slen == -2:
                print("FATAL: --requirepass-file: " + args[i + 1] + " is empty.")
                print("  An empty password file would silently disable auth.")
                _sbuf.unsafe_free()
                external_call["exit", NoneType](Int32(1))
            elif _slen < 0:
                print("FATAL: --requirepass-file: password too long (max 511).")
                _sbuf.unsafe_free()
                external_call["exit", NoneType](Int32(1))
            else:
                config.server.requirepass = String(StringSpan[MutUntrackedOrigin](
                    unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=_sbuf, length=_slen)))
            _sbuf.unsafe_free()
            i += 2
        elif args[i] == "--bind" and i + 1 < len(args):
            # gh #258: validated after the loop, because an invalid address must
            # be a hard failure rather than a silent fall back to INADDR_ANY.
            config.server.bind_addr = args[i + 1]
            i += 2
        elif args[i] == "--tenant" and i + 1 < len(args):
            # gh #101: repeatable NAME=PASSWORD tenant credential. Any --tenant
            # enables per-connection tenant namespacing and requires
            # --requirepass (the admin credential) — validated after the loop.
            var terr = tenant_arg_error(args[i + 1])
            if terr.byte_length() > 0:
                print("FATAL: invalid --tenant argument: " + terr)
                return
            if config.server.tenants.byte_length() > 0:
                config.server.tenants += "\n"
            config.server.tenants += args[i + 1]
            i += 2
        elif args[i] == "--moe-cache" and i + 1 < len(args):
            # gh #61 Stage 2: MoE-expert tier. Path to a directory of MoE
            # safetensors shards (e.g. an HF cache snapshot dir). When set,
            # MoEExpertTier loads the manifest at startup; MOE.EXPERT.*
            # handlers serve real INFO/FETCH/STATS from disk.
            config.server.moe_cache_path = args[i + 1]
            i += 2
        elif args[i] == "--moe-cache-mib" and i + 1 < len(args):
            # gh #61 Stage 2: LRU cache budget in MiB. Default 1024
            # (Phase-0 sweet spot on 16 GB Mac).
            try:
                config.server.moe_cache_mib = atol(args[i + 1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif (args[i] == "--dim") and i + 1 < len(args):
            try:
                config.vector.dimensions = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif (args[i] == "--max-elements") and i + 1 < len(args):
            try:
                config.vector.max_elements = atol(args[i+1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        # gh #138: crash/exit diagnostics.
        elif args[i] == "--crash-log" and i + 1 < len(args):
            config.server.crash_log = args[i + 1]
            i += 2
        elif args[i] == "--status-file" and i + 1 < len(args):
            config.server.status_file = args[i + 1]
            i += 2
        elif args[i] == "--no-crash-log":
            # Sentinel: suppress the port-stamped defaults applied below.
            config.server.crash_log = "-"
            config.server.status_file = "-"
            i += 1
        elif args[i] == "--maxmemory" and i + 1 < len(args):
            # gh #261: Redis units (1k = 1000, 1kb = 1024 ...) or N% of physical RAM.
            # Parsed from a HEAP copy: a short String keeps its bytes inline, on
            # the stack, and a stack pointer handed to an out-of-line Mojo
            # function as MutUntrackedOrigin is miscompiled at -O3 (gh #349).
            var _mm = String(args[i + 1])
            var _mm_n = _mm.byte_length()
            var _mm_h = alloc[UInt8](_mm_n + 1)
            for _k in range(_mm_n):
                _mm_h[unsafe_offset=_k] = _mm.as_bytes()[_k]
            var _mm_v = parse_memory_value(_mm_h, _mm_n,
                Int64(external_call["pion_physical_ram_bytes", UInt64]()))
            _mm_h.unsafe_free()
            if not _mm_v.ok:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i])
                            + " (bytes, or with k/kb/m/mb/g/gb, or 1-100%):")
            config.server.maxmemory = Int(_mm_v.value)
            i += 2
        elif args[i] == "--rss-warn-pct" and i + 1 < len(args):
            try:
                config.server.rss_warn_pct = atol(args[i + 1])
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]))   # gh #372
            i += 2
        elif args[i] == "--lua-time-limit" and i + 1 < len(args):
            # #36: milliseconds; 0 = never stop a script
            try:
                var _ltl = atol(args[i + 1])
                if _ltl < 0:
                    raise Error("negative")
                lua_time_limit_ms = _ltl
            except:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i]) + " (milliseconds, 0 = never):")
            i += 2
        elif args[i] == "--lua-memory-limit" and i + 1 < len(args):
            # #36: Redis units, as --maxmemory; 0 = no cap. Parsed from a heap
            # copy for the reason --maxmemory's comment gives (gh #349).
            var _lm = String(args[i + 1])
            var _lm_n = _lm.byte_length()
            var _lm_h = alloc[UInt8](_lm_n + 1)
            for _k in range(_lm_n):
                _lm_h[unsafe_offset=_k] = _lm.as_bytes()[_k]
            var _lm_v = parse_memory_value(_lm_h, _lm_n,
                Int64(external_call["pion_physical_ram_bytes", UInt64]()))
            _lm_h.unsafe_free()
            if not _lm_v.ok:
                _refuse_arg(String(args[i + 1]), "invalid value for " + String(args[i])
                            + " (bytes, or with k/kb/m/mb/g/gb, or 1-100%):")
            lua_memory_limit = Int(_lm_v.value)
            i += 2
        else:
            # gh #372: nothing matched. A known value flag here means its value
            # is missing (every value arm guards `i + 1 < len(args)`).
            var _bad = String(args[i])
            if _is_value_flag(_bad):
                _refuse_arg(_bad, "flag requires a value:")
            elif _bad.startswith("-"):
                _refuse_arg(_bad, "unknown flag")
            else:
                _refuse_arg(_bad, "unexpected argument")
            i += 1

    # gh #138: default breadcrumb paths are port-stamped so two servers on the
    # same host never interleave lines in one file. "-" = explicitly disabled.
    if config.server.crash_log.byte_length() == 0:
        config.server.crash_log = "pion-" + String(config.server.port) + ".crash.log"
    elif config.server.crash_log == "-":
        config.server.crash_log = ""
    if config.server.status_file.byte_length() == 0:
        config.server.status_file = "pion-" + String(config.server.port) + ".status"
    elif config.server.status_file == "-":
        config.server.status_file = ""

    # --nle-embed claims the embedding backend before auto-detect/auto-embed
    # would otherwise win it. Apple NLEmbedding via the system NaturalLanguage
    # framework — 512-dim sentence embeddings, no Python.
    if nle_embed and config.server.profile != "kv":
        config.embedding.enabled = True
        config.embedding.nle = True
        config.embedding.dimensions = 512
        config.embedding.threshold = 0.80   # NLE uses different vector space than MiniLM; calibrate later
        config.embedding.model = "apple-nle-sentence-en"
        config.embedding.host = ""          # disables HTTP fallback path
        print("--nle-embed: Apple NLEmbedding (512-dim, no Python)")
        print("    AI.SEMANTIC_CACHE SET/GET will use the system NaturalLanguage framework")

    # Auto-detect Ollama if neither --emb-enabled nor --flare nor --nle-embed was set
    if auto_detect and not config.embedding.enabled:
        _try_auto_detect_ollama(config)

    # A3: Auto-embed fallback — when no Ollama found and no explicit embedding config,
    # auto-launch the inference sidecar with MiniLM-L6-v2 (384-dim, ~22MB, zero-dependency).
    # This makes AI.SEMANTIC_CACHE SET/GET work out of the box without any external server.
    if auto_embed and not config.embedding.enabled and not config.inference.enabled and config.server.profile != "kv":
        # gh #282: ASK whether the sidecar can start before promising it.
        # The release tarball ships bin/ + lib/ and no src/inference, so this
        # branch used to enable inference, print two lines saying semantic cache
        # would work, fork something that could not exec, and burn 30 s in the
        # connect-poll — all before the listen socket opened.
        var probe_py = String(".pixi/envs/default/bin/python3")
        var probe_sc = String("src/inference/worker.py")
        if _resolve_sidecar(probe_py, probe_sc):
            config.inference.enabled = True
            config.inference.default_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            config.embedding.enabled = True
            config.embedding.dimensions = 384   # MiniLM-L6-v2 output dimension
            config.embedding.threshold = 0.85   # lower for 384-dim auto-embed: catches rephrased queries
            print("A3: Auto-embed enabled — launching MiniLM-L6-v2 sidecar (384-dim, zero config)")
            print("    AI.SEMANTIC_CACHE SET/GET will work without external Ollama/OpenAI")
            print("    Disable with --no-auto-embed or --profile kv (or use --nle-embed on macOS)")
        else:
            print("A3: Auto-embed unavailable — no inference sidecar next to this binary.")
            print("    (Expected src/inference/worker.py; a prebuilt release does not ship it.)")
            print("    Embeddings: --nle-embed on macOS (no Python, no download), an Ollama")
            print("    instance, or a source checkout. Starting without them.")

    # gh #258: PION_REQUIREPASS is the last resort, so an explicit flag always
    # wins. An env var is not as good as a file (it is inherited by children and
    # readable via /proc/<pid>/environ on some systems) but it is strictly better
    # than argv, and it is what container orchestrators hand you.
    if config.server.requirepass.byte_length() == 0:
        var _envp = external_call["getenv", Pointer[UInt8, MutUntrackedOrigin]](
            String("PION_REQUIREPASS\0").unsafe_ptr())
        if is_not_null(_envp):
            var _n = 0
            while _n < 512 and _envp[unsafe_offset=_n] != 0:
                _n += 1
            if _n > 0:
                config.server.requirepass = String(StringSpan[MutUntrackedOrigin](
                    unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                        unsafe_ptr=_envp, length=_n)))

    # gh #258: choose and APPLY the bind address before any listener is created.
    #
    # Every listener used to bind INADDR_ANY — RESP, the port+1 binary lane, the
    # port+10000 WAL replication stream (which has no authentication at all),
    # and the gossip/Raft pair. On a laptop or a cloud box without a firewall
    # that is the entire keyspace, reachable from the network, by default.
    #
    # The default follows Redis's protected mode: loopback unless the operator
    # has set a password, since a password is the signal that remote access is
    # intended. An explicit --bind always wins over both.
    var _bind_choice = config.server.bind_addr
    var _bind_reason = String("--bind")
    if _bind_choice.byte_length() == 0:
        if config.server.requirepass.byte_length() > 0:
            _bind_choice = "0.0.0.0"
            _bind_reason = "default (a password is set)"
        else:
            _bind_choice = "127.0.0.1"
            _bind_reason = "default (no password set)"
    var _bindz = _bind_choice + "\0"
    if Int(external_call["pion_set_bind_addr", Int32](_bindz.unsafe_ptr())) != 0:
        # Refuse rather than fall back: a typo'd --bind that silently became
        # INADDR_ANY would EXPOSE the server, which is the opposite of the ask.
        print("FATAL: --bind '" + _bind_choice + "' is not a valid IPv4 address.")
        external_call["exit", NoneType](Int32(1))
    print("Binding to " + _bind_choice + " [" + _bind_reason + "]")
    if _bind_choice == "0.0.0.0" and config.server.requirepass.byte_length() == 0:
        print("  WARNING: listening on ALL interfaces with NO password. Anyone")
        print("  who can reach this host can read and write the keyspace, and")
        print("  the replication port (" + String(config.server.port + 10000)
              + ") streams the WAL unauthenticated.")

    # gh #101: tenant mode has no anonymous path — --requirepass is the admin
    # credential (someone has to run BGSAVE/INFO/FLUSHALL) and its presence
    # arms the gh #100 NOAUTH gate that forces every connection to bind.
    if config.server.tenants.byte_length() > 0 and config.server.requirepass.byte_length() == 0:
        print("FATAL: --tenant requires --requirepass (the admin credential).")
        print("  Tenant isolation is fail-closed: without --requirepass an")
        print("  unauthenticated connection could read prefixed keys verbatim.")
        return

    # Ignore SIGPIPE (signal 13) so writes to closed sockets return EPIPE instead of killing the process
    _ = external_call["signal", Int32](13, 1)

    # gh #282: last chance to make the banner true. The auto-embed path already
    # pre-flights, but an explicit --inference reaches here unchecked, and the
    # banner would then announce "Inference: enabled (socket=...)" for a sidecar
    # that cannot start — the same announce-before-you-have-it shape as gh #281's
    # Metal line and gh #257's CONFIG SET. Resolve it here so dump() reports what
    # will actually happen.
    if config.inference.enabled:
        var pre_py = String(".pixi/envs/default/bin/python3")
        var pre_sc = String("src/inference/worker.py")
        if not _resolve_sidecar(pre_py, pre_sc):
            print("Notice: --inference requested but no sidecar found beside the working")
            print("        directory or the binary (src/inference/worker.py). Disabling it")
            print("        now rather than spending 30s discovering it after a failed exec.")
            config.inference.enabled = False

    # gh #373: `--inference` alone started the sidecar and left EMBEDDING off —
    # the A3 auto-embed above only fires when inference is not already on — so
    # AI.EMBED answered "requires --inference" to a server started with it. The
    # sidecar serves the MiniLM default, so route embeddings through it. A
    # non-default --inference-emb-model has an unknown dimension; leave that to
    # an explicit --emb-dim/--emb-enabled rather than guess.
    if (config.inference.enabled and not config.embedding.enabled and auto_embed
            and config.server.profile != "kv"
            and config.inference.default_embedding_model == "sentence-transformers/all-MiniLM-L6-v2"):
        config.embedding.enabled = True
        config.embedding.dimensions = 384
        config.embedding.threshold = 0.85
        print("--inference: embeddings served by the sidecar (MiniLM-L6-v2, 384-dim)")

    # gh #424: the sidecar socket defaulted to a fixed machine-wide path
    # (/tmp/pion_inference.sock, mode 0755): two servers on one box collided
    # (the second unlinked the first's socket), and any local user could squat
    # it. Derive a per-uid, per-port default so instances never collide; the
    # sidecar binds it under umask 077 (0700) so another user cannot connect.
    # An explicit --inference-socket is left untouched.
    if config.inference.enabled and not inference_socket_explicit:
        var _uid = Int(external_call["getuid", UInt32]())
        config.inference.socket_path = String("/tmp/pion-") + String(_uid) \
            + String("-") + String(config.server.port) + String(".inference.sock")

    var nodes = List[String]()
    nodes.append("127.0.0.1:" + String(config.server.port))

    # V21: Apple M4 E-core trap mitigation
    # On Apple Silicon, Mojo's parallelize thread pool uses only P-cores (4 on M4).
    # Requesting more workers causes them to silently not start, breaking sharding.
    # Cap unconditionally to 4 on macOS regardless of -w flag.
    if CompilationTarget.is_macos() and config.server.workers > 4:
        config.server.workers = 4
        print("Notice: macOS P-core limit — capped to 4 workers (Mojo thread pool constraint)")

    # gh #422: the `ai` profile's single-worker guarantee is enforced HERE, not
    # inside apply_profile. apply_profile runs at the position of --profile in
    # argv, so a later `-w N` overrode the workers=1 it set: `--profile ai -w 4`
    # started 4 private semantic caches — exactly the split the profile exists to
    # prevent — while `-w 4 --profile ai` started 1. Every cap in this section is
    # order-independent, so applying it here makes the profile's promise true
    # regardless of flag order.
    if config.server.profile == "ai" and config.server.workers > 1:
        config.server.workers = 1
        print("Notice: --profile ai — capped to 1 worker (semantic cache is per-worker).")

    # Q4: AI features (embedding / semantic cache / AI.COMPLETE) use a per-worker SemanticCache.
    # With multiple workers, cache hits stored on worker 0 are invisible to worker 1-N
    # (no cross-worker HNSW sharing for the semantic cache path, unlike FT.SEARCH).
    # Enforce -w 1 when AI features are enabled to guarantee a single shared cache.
    if config.embedding.enabled and config.server.workers > 1:
        config.server.workers = 1
        print("Notice: AI features enabled (--flare/--emb-enabled) — capped to 1 worker.")
        print("        Semantic cache is per-worker; multi-worker mode would split cache across")
        print("        workers, causing cache misses for queries landing on different workers.")
        print("        Use -w 1 explicitly to suppress this notice.")

    # M1: Inference sidecar requires single-worker mode
    if config.inference.enabled and config.server.workers > 1:
        config.server.workers = 1
        print("Notice: --inference enabled — capped to 1 worker (inference sidecar is single-connection).")

    # gh #422: dump the banner AFTER every worker cap, so "Workers: N" reports the
    # count the server will actually run — it used to print before the caps, so
    # `--profile ai -w 4` announced "Workers: 4" and then silently ran 1.
    config.dump()

    # gh #253: THE multi-worker fence. Every cap above can only *reduce* the
    # worker count, so this is the first point where the number is final — a
    # `-w 8` that AI features already collapsed to 1 is coherent and must not be
    # refused. Past here, workers > 1 means N independent keyspaces, and that is
    # only ever reached by explicit acknowledgement.
    #
    # The failure it fences is silent, which is why it is fatal rather than a
    # warning: connections are assigned by accept() race, so a pooled client
    # (redis-py's default ConnectionPool included) writes on one worker and reads
    # from another. Measured at -w 4 with 16 concurrently-opened connections:
    # 41 of 90 GETs of a just-acked key returned nil. Serially-opened
    # connections all land on one worker and hide it completely.
    if config.server.workers > 1 and not config.server.independent_workers:
        print("")
        print("FATAL: -w " + String(config.server.workers) +
              " requires --independent-workers.")
        print("")
        print("  Pion workers are shared-nothing: each owns a PRIVATE keyspace, and a")
        print("  connection is bound to whichever worker won the accept() race. A write")
        print("  acknowledged +OK on one connection is INVISIBLE to a read on another.")
        print("")
        print("  Any client that opens more than one connection — which includes every")
        print("  default connection pool (redis-py, Jedis, go-redis, ioredis) — will read")
        print("  nils for keys it just wrote, with no error anywhere.")
        print("")
        print("  Run one of:")
        print("    ./pion-server -w 1                          coherent single keyspace (default)")
        print("    ./pion-server -w " + String(config.server.workers) +
              " --independent-workers   N keyspaces, client pins one connection")
        print("")
        print("  See doc/architecture.md.")
        external_call["exit", NoneType](Int32(1))

    if config.server.workers > 1:
        print("")
        print("=== INDEPENDENT WORKERS: " + String(config.server.workers) +
              " SEPARATE KEYSPACES ===")
        print("  Each worker owns a private keyspace, hash map, WAL and HNSW graph.")
        print("  Connections are assigned by accept() race — a write acked on one")
        print("  connection is NOT readable from another. Cross-worker PUBLISH delivers")
        print("  to zero subscribers, FLUSHALL clears one slice, and replication covers")
        print("  worker 0 only.")
        print("  Safe only if every client pins a single connection, or shards keys")
        print("  across per-worker ports itself. Use -w 1 for shared-keyspace semantics.")
        print("=======================================================")
        print("")

    print("--- Pion Master: Spawning " + String(config.server.workers) + " Persistent Mojo Workers ---")

    # KV.PREFIX.* / V.STOREBATCH / V.FETCH — V buffers themselves stay
    # per-worker (no shared mmap), but the cross-worker session directory
    # (allocated in `vstore_directory_ptr` below) makes KV.PREFIX.LOOKUP
    # accurate across workers. V.STOREBATCH / V.FETCH still need to land
    # on the owner worker that physically holds the buffers — non-owner
    # V.* commands return "ERR session not on this worker (owner=N)" so
    # clients can pin a connection per namespace.
    if config.server.kvcache_enabled and config.server.workers > 1:
        print("Notice: --kvcache with -w " + String(config.server.workers) +
              " — KV.PREFIX.LOOKUP is cross-worker via shared directory.")
        print("        V.STOREBATCH / V.FETCH must run on the owner worker; non-owner")
        print("        V.* commands return -ERR with the owner worker_id so the client")
        print("        can pin its connection per namespace.")

    # M1: Spawn inference sidecar process
    #
    # gh #282: the readiness POLL is deliberately NOT here. It used to run in
    # this block, ~250 lines before create_listen_socket(), so a reader who ran
    # the README's next command during it got "Connection refused" for up to
    # 30 s — the listen-before-init property (doc/architecture.md) inverted for
    # exactly the configuration a first-time user starts with. The spawn itself
    # is non-blocking, so it stays; the wait moved below the listen sockets,
    # where a client queues in the 65535-deep kernel backlog instead of being
    # refused. The sidecar also imports torch concurrently with the rest of
    # init now rather than serially before it, so the wait is shorter too.
    var sidecar_pid = Int32(-1)
    if config.inference.enabled:
        var c_python = String(".pixi/envs/default/bin/python3")
        var c_script = String("src/inference/worker.py")
        # gh #176 resolved the foreign-cwd case; gh #282 handles "resolves
        # nowhere". Forking a doomed exec costs the connect-poll its full 30 s
        # and ends in exactly the same state, so refuse up front and say so.
        # This block runs BEFORE the listen socket opens, which is why the wasted
        # time shows up to a user as "Connection refused".
        if not _resolve_sidecar(c_python, c_script):
            print("M1: no inference sidecar found — inference disabled.")
            print("    Looked for src/inference/worker.py beside the working directory")
            print("    and beside the binary. A prebuilt release does not ship it.")
            print("    Embeddings: --nle-embed on macOS, an Ollama instance, or a")
            print("    source checkout. Pass --no-auto-embed to skip this check.")
            config.inference.enabled = False
        else:
            print("M1: Spawning inference sidecar...")
            var c_socket = config.inference.socket_path
            var c_emb = config.inference.default_embedding_model
            var c_llm = config.inference.default_llm_model
            var inference_sidecar_pid = external_call["pion_spawn_inference", Int32](
                c_python.as_c_string_slice(),
                c_script.as_c_string_slice(),
                c_socket.as_c_string_slice(),
                c_emb.as_c_string_slice(),
                c_llm.as_c_string_slice(),
            )
            if inference_sidecar_pid > 0:
                print("M1: Inference sidecar PID=" + String(Int(inference_sidecar_pid)))
                # Readiness is polled after the listen sockets open — see below.
                sidecar_pid = inference_sidecar_pid
                # gh #424: reap the sidecar on our exit (clean shutdown or the
                # FATAL-bind path, which is reached AFTER this spawn). The
                # sidecar's own getppid() watchdog is the catch-all for SIGKILL.
                external_call["pion_register_child_pid", NoneType](inference_sidecar_pid)
            else:
                print("M1: ERROR — failed to spawn inference sidecar")
                config.inference.enabled = False

    # MLX sidecar removed 2026-05-01 — ATTEND.PREFIX.* is served end-to-end
    # by the in-process Metal SDPA kernels (--metal-attention / --metal-attention-fp16).

    # V2.6: single shared HNSW view — after FT.OPTIMIZE on any worker, all workers can search
    var shared_hnsw_ptr = alloc[SharedHNSWView](1)
    shared_hnsw_ptr.unsafe_write(SharedHNSWView())

    # ingest_count: separately allocated so pointer value is preserved in SharedHNSWView copies.
    # Embedded Atomic[Scalar[DType.uint64]] was snapshotted in __moveinit__, causing workers 1-N to always
    # see count=0 and skip build. Pointer pattern (same as optimize_trigger) fixes this.
    var ingest_count_buf = alloc[UInt64](1)
    ingest_count_buf[unsafe_offset=0] = 0
    shared_hnsw_ptr[].ingest_count = ingest_count_buf

    # Sharding: each worker builds 1/N of the HNSW graph so each shard's compact_buffer
    # (50K/N × 1536 bytes) fits better in per-core L2/L3 and reduces cross-worker DRAM pressure.
    # macOS: disabled (num_shards=1). Blocking cooperative-wait cuts QPS 8000→3097 because
    #   kqueue timeout=0 costs 5–10µs/iteration → coordinator spin-wait ≈1.4ms.
    # Linux (io_uring): enabled. io_uring enter() with min_complete=0 costs ~100–200ns/iteration
    #   → coordination overhead drops 50–100×. Each shard (50K/N × 1536B) is smaller:
    #   8 shards → 9.6MB each → fits Linux server L3 (typically 16–64MB) → ~10× fewer DRAM
    #   misses per beam search → expected 15–30K QPS on bare-metal vs 9.6K unsharded.
    var n_workers = config.server.workers  # already capped to ≤4 on macOS above

    # gh #138: crash/exit breadcrumbs. Installed here — before parallelize() and
    # before the XDP handler registration below, so XDP keeps SIGINT/SIGTERM for
    # its BPF detach while the fatal signals (SEGV/BUS/ILL/FPE/ABRT) stay ours.
    # The heartbeat half (status file) is what survives an *uncatchable* jetsam
    # or OOM-killer SIGKILL: it holds RSS as of ~1s before death.
    #
    # Installed even with --no-crash-log (it then opens no file): the same call
    # installs the gh #259 SIGTERM/SIGINT latch, and skipping it made a server
    # started with that flag die on the signal without its WAL flush.
    var _crash_log_p = config.server.crash_log + "\0"
    var _status_p = config.server.status_file + "\0"
    var _ver = PION_VERSION + "+" + PION_BUILD_SHA + "\0"
    _ = external_call["pion_crash_init", Int32](
        _crash_log_p.unsafe_ptr(),
        _status_p.unsafe_ptr(),
        _ver.unsafe_ptr(),
        Int32(config.server.port),
        Int32(n_workers),
        Int32(config.server.rss_warn_pct),
    )

    # gh #261: the limit is process-wide and lives in C, where every worker's
    # housekeeping tick reads it. The C side logs each crossing (with the RSS it
    # measured), so this line states only what was configured.
    if config.server.maxmemory > 0:
        external_call["pion_set_maxmemory", NoneType](UInt64(config.server.maxmemory))
        print("Maxmemory: " + String(config.server.maxmemory >> 20)
              + " MB (noeviction: above it, memory-growing writes get -OOM)")
    # #36: every worker's Lua states take these when they are created.
    external_call["pion_lua_set_defaults", NoneType](Int64(lua_memory_limit), Int64(lua_time_limit_ms))

    # Sharding: DISABLED on all platforms (2026-04-27).
    #
    # Previously Linux enabled sharding (num_shards = n_workers) for L3-cache-fit perf
    # while macOS kept num_shards=1 because kqueue timeout=0 made coordinator spin-wait
    # costly. But T3.4 sharded-search returns recall≈0.0002 at 500K w=4/16 on Linux
    # (drain_deferred_shard_responses merge path returns effectively-random IDs), and at
    # 5M w=16 the same path SIGSEGVs (all workers crash at the same IP). Mac was always
    # num_shards=1, so it never exercised the buggy code path — that's why the Mac gate
    # shows recall ≥ 0.94 while Linux multi-worker silently returned random results.
    # Until the shard merge correctness bug is properly fixed, force num_shards=1 on
    # all platforms — sacrifices the "8 shards in 32MB L3" perf claim but recovers
    # correctness.
    shared_hnsw_ptr[].num_shards = 1
    var optimize_trigger_buf = alloc[UInt64](1)
    optimize_trigger_buf[unsafe_offset=0] = 0
    shared_hnsw_ptr[].optimize_trigger = optimize_trigger_buf

    # ready_atomic: separately allocated UInt64. publish_to_shared writes 1 with
    # RELEASE ordering after all field writes; FT.SEARCH reads with ACQUIRE
    # ordering before borrowing pointers. Without this, ARM weak memory order
    # can let a borrower observe ready=True before observing the field writes,
    # producing recall≈0 when borrowed pointers are stale.
    var ready_atomic_buf = alloc[UInt64](1)
    ready_atomic_buf[unsafe_offset=0] = 0
    shared_hnsw_ptr[].ready_atomic = ready_atomic_buf

    # #19: count the persisted indexes the workers are about to load. Each is
    # pion.hnsw.<worker that built it>, loaded by that worker alone; the others
    # wait for it before serving (see Pion.__init__ phase 6).
    # [0] = loaders still loading; [1 + w] = 1 when worker w is one. Decided
    # here, once, so a worker never re-derives it from the filesystem.
    var warm_pending_buf = alloc[UInt64](1 + n_workers)
    warm_pending_buf[unsafe_offset=0] = 0
    for _wl in range(n_workers):
        var _wl_path = "pion.hnsw." + String(_wl) + "\0"
        var _loads = external_call["access", Int32](_wl_path.unsafe_ptr(), Int32(4)) == 0   # R_OK
        warm_pending_buf[unsafe_offset=1 + _wl] = 1 if _loads else 0
        if _loads:
            warm_pending_buf[unsafe_offset=0] += 1
    shared_hnsw_ptr[].warm_load_pending = warm_pending_buf

    # gh #14: phase-2 epoch RCU state. `reclaim_epoch` is bumped by
    # FT.DROPINDEX; `worker_epoch` holds one 64-byte-strided slot per worker so
    # the per-batch relaxed store on the dispatch path never shares a cache
    # line with another core. Sized from n_workers, which is already capped.
    var reclaim_epoch_buf = alloc[UInt64](1)
    reclaim_epoch_buf[unsafe_offset=0] = 1
    shared_hnsw_ptr[].reclaim_epoch = reclaim_epoch_buf
    var worker_epoch_buf = alloc[UInt64](8 * n_workers)
    unsafe_memset(worker_epoch_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    shared_hnsw_ptr[].worker_epoch = worker_epoch_buf
    shared_hnsw_ptr[].worker_epoch_slots = n_workers

    # hk_keys_buf: cross-worker slot → original-hash-key map. Each slot is 32 bytes
    # (byte 0 = length, bytes 1..31 = key data). Without this, multi-worker FT.SEARCH
    # recall collapses to ~0 when the search worker isn't the same as the HSET worker
    # (the local __hk__<slot> keyspace lookup misses for slots written by another
    # worker, and the response falls back to the slot number — which doesn't match
    # the bench's hash key).
    var hk_max = config.vector.max_elements
    var hk_keys_buf = alloc[UInt8](hk_max * 32)
    unsafe_memset(hk_keys_buf, 0, hk_max * 32)
    shared_hnsw_ptr[].hk_keys_buf = hk_keys_buf
    shared_hnsw_ptr[].hk_max_elements = hk_max

    # 64-byte stride per worker: shard_ready[worker_id * 8] (UInt64 units × 8 bytes each = 64 bytes).
    # Each worker's ready flag occupies its own cache line — eliminates LDXR/STXR livelock on ARM64.
    var shard_ready_buf = alloc[UInt64](8 * n_workers)
    unsafe_memset(shard_ready_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    shared_hnsw_ptr[].shard_ready = shard_ready_buf

    var shard_bus_ptr = alloc[ShardQueryBus](1)
    shard_bus_ptr.unsafe_write(ShardQueryBus(n_workers))
    shared_hnsw_ptr[].shard_bus = shard_bus_ptr

    # dbg_counters: 8 UInt64 per worker (64-byte stride), slots: [poll, saw_trig, build_done, build_fail, ...]
    var dbg_counters_buf = alloc[UInt64](8 * n_workers)
    unsafe_memset(dbg_counters_buf.unsafe_bitcast[UInt8](), 0, 8 * n_workers * 8)
    shared_hnsw_ptr[].dbg_counters = dbg_counters_buf

    # Cross-worker V-store directory — shared metadata segment so KV.PREFIX.LOOKUP
    # answers correctly regardless of which worker owns the V buffers. NULL
    # when --kvcache is off (single-worker fast path; no overhead). Stored in
    # SharedHNSWView as an opaque pointer to dodge an import cycle.
    if config.server.kvcache_enabled:
        var _vd = alloc[VStoreDirectory](1)
        _vd.unsafe_write(VStoreDirectory())
        shared_hnsw_ptr[].vstore_directory = Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(_vd))

    # Phase 5: Cluster state — shared across all workers (read-only after init)
    var cluster_ptr = alloc[ClusterState](1)
    cluster_ptr.unsafe_write(ClusterState())
    if config.cluster.enabled:
        cluster_ptr[].enabled = True
        cluster_ptr[].my_port = config.server.port
        cluster_ptr[].set_my_host(config.cluster.my_host)
        cluster_ptr[].generate_node_id(
            cluster_ptr[].my_host.unsafe_ptr(),
            cluster_ptr[].my_host_len,
            config.server.port,
        )
        # Replica mode
        cluster_ptr[].is_replica = config.cluster.is_replica
        # Parse peer_nodes: "host1:port1,host2:port2,..."
        if config.cluster.peer_nodes.byte_length() > 0:
            _parse_peer_nodes(cluster_ptr, config.cluster.peer_nodes, config.cluster.my_host, config.server.port)
        # For replicas: register primary as peer[0] if not already in peer_nodes
        if config.cluster.is_replica and config.cluster.primary_host != "" and cluster_ptr[].peer_count == 0:
            var ph = config.cluster.primary_host
            var tmp = alloc[UInt8](ph.byte_length() + 1)
            for ci in range(ph.byte_length()):
                tmp[unsafe_offset=ci] = ph.unsafe_ptr()[unsafe_offset=ci]
            tmp[unsafe_offset=ph.byte_length()] = 0
            cluster_ptr[].set_peer(0, tmp, ph.byte_length(), config.cluster.primary_port, 0, 16383)
            tmp.unsafe_free()
            cluster_ptr[].peer_count = 1
            cluster_ptr[].primary_peer_idx = 0

    # §7: GPU Vector Engine — Metal compute initialization (macOS only, --gpu flag)
    if config.vector.has_gpu:
        comptime if CompilationTarget.is_macos():
            var gpu_rc = external_call["pion_metal_init", Int32](
                UInt32(config.vector.dimensions), UInt32(config.vector.max_elements))
            if gpu_rc == 0:
                print("GPU: Metal compute engine initialized (dim=" + String(config.vector.dimensions) + ")")
            else:
                print("GPU: Metal init failed (rc=" + String(gpu_rc) + "), falling back to CPU")
                config.vector.has_gpu = False

    # XDP multi-worker: create shared XSKMAP + BPF program ONCE before parallelize.
    # Each worker registers its own AF_XDP socket in the shared XSKMAP.
    # Also set up flow steering so NIC directs target port traffic to worker queues.
    var xdp_shared_xskmap_fd = Int32(-1)
    var xdp_shared_bpf_fd = Int32(-1)
    if config.server.use_xdp and CompilationTarget.is_linux():
        xdp_shared_xskmap_fd = external_call["pion_xdp_create_shared_xskmap", Int32](Int32(config.server.workers))
        if xdp_shared_xskmap_fd >= 0:
            xdp_shared_bpf_fd = external_call["pion_xdp_load_shared_bpf", Int32](
                UInt16(config.server.port), xdp_shared_xskmap_fd
            )
            if xdp_shared_bpf_fd >= 0:
                var iface_cstr = config.server.xdp_interface
                var attach_rc = external_call["pion_xdp_attach_bpf", Int32](
                    iface_cstr.as_c_string_slice(), xdp_shared_bpf_fd
                )
                if attach_rc < 0:
                    print("XDP: BPF attach failed — falling back to io_uring")
                    config.server.use_xdp = False
                else:
                    # Set up flow steering: direct target port traffic to RX queue 0
                    # For single worker: all traffic → queue 0
                    # For multi-worker: need per-queue flow rules (RSS doesn't work with AF_XDP)
                    var fs_iface = config.server.xdp_interface
                    _ = external_call["pion_xdp_setup_flow_steering", Int32](
                        fs_iface.as_c_string_slice(), UInt16(config.server.port), Int32(0)
                    )
                    print("XDP: shared XSKMAP fd=" + String(Int(xdp_shared_xskmap_fd))
                          + " BPF fd=" + String(Int(xdp_shared_bpf_fd))
                          + " workers=" + String(config.server.workers))
            else:
                print("XDP: BPF load failed — falling back to io_uring")
                config.server.use_xdp = False
        else:
            print("XDP: XSKMAP creation failed — falling back to io_uring")
            config.server.use_xdp = False

    # Signal handler for graceful XDP cleanup: detach BPF from NIC on SIGINT/SIGTERM.
    # Without this, the BPF program stays attached after exit and blocks all port traffic.
    if config.server.use_xdp and CompilationTarget.is_linux():
        var sig_iface = config.server.xdp_interface
        external_call["pion_xdp_register_signal_handlers", NoneType](
            sig_iface.as_c_string_slice(), UInt16(config.server.port)
        )

    # Each connection is one fd, and every per-fd table holds 65536 entries. A
    # stock Linux login allows 1024 open files, which capped the server at about
    # a thousand clients; raise the soft limit as Redis does.
    var _nofile_before = alloc[Int64](1)
    _nofile_before[unsafe_offset=0] = 0
    var _nofile = external_call["pion_raise_nofile", Int64](Int64(65536), _nofile_before)
    if _nofile > _nofile_before[unsafe_offset=0] and _nofile_before[unsafe_offset=0] > 0:
        print("Open-file limit raised from " + String(_nofile_before[unsafe_offset=0])
              + " to " + String(_nofile))
    _nofile_before.unsafe_free()

    # V3.1: single shared listen socket — all workers register with their own kqueue and race
    # to accept(). macOS delivers connections round-robin across workers → N× QPS scaling.
    var shared_listen_fd = create_listen_socket(config.server.port)
    if shared_listen_fd < 0:
        print("FATAL: cannot bind port " + String(config.server.port) + " (already in use?)")
        print("  Check with: lsof -i :" + String(config.server.port))
        # `return` here exited with status 0, so every supervisor, CI step and
        # launch script read a port collision as a SUCCESSFUL start — and then
        # benchmarked or tested against whatever process already held the port.
        # On this Mac that is a live failure mode: the PionMesh iOS app binds
        # 1974. Exit non-zero so the caller can actually tell.
        external_call["exit", NoneType](Int32(1))
        return

    # Binary protocol listen socket (shared across all workers, like the main listen fd).
    # Port = config.server.port + 1 (e.g. 1975 for -p 1974). Workers race to accept().
    # Only created when --kvcache is enabled (avoids extra kqueue fd + icache pressure).
    var binary_listen_fd = create_listen_socket(config.server.port + 1) if config.server.kvcache_enabled else Int32(-1)

    # P4: per-worker secondary listen sockets for local-affinity connections.
    # Port = config.server.port + 2 + worker_id (e.g. 1976, 1977, 1978, 1979 for -p 1974 -w 4).
    # Clients connecting to these ports bypass cross-worker routing entirely.
    var secondary_listen_fds = alloc[Int32](n_workers)
    for wi in range(n_workers):
        secondary_listen_fds[unsafe_offset=wi] = create_listen_socket(config.server.port + 2 + wi)

    # gh #282: NOW wait for the sidecar, with every listen socket already open.
    # A client connecting during this wait is queued by the kernel and served
    # the moment the workers start, rather than refused. That is also why the
    # 3 s message reads differently from the one it replaced: it can truthfully
    # say connections are being accepted, because they are.
    if sidecar_pid > 0:
        var connected = False
        var waited_ds = 0
        for _ in range(300):
            var c_sp2 = config.inference.socket_path
            var test_fd = external_call["pion_connect_unix", Int32](c_sp2.as_c_string_slice())
            if test_fd >= 0:
                _ = external_call["close", Int32](test_fd)
                connected = True
                break
            if waited_ds == 30:
                print("M1: still waiting for the sidecar to listen (up to 30s; it is")
                print("    importing torch). Connections are already accepted and will be")
                print("    served as soon as it is ready. --no-auto-embed skips this.")
            # Sleep 100ms
            _ = external_call["usleep", Int32](Int32(100000))
            waited_ds += 1
        if connected:
            print("M1: Inference sidecar ready (socket=" + config.inference.socket_path +
                  ", " + String(waited_ds // 10) + "." + String(waited_ds % 10) + "s)")
        else:
            print("M1: WARNING — sidecar not ready after 30s, disabling inference")
            config.inference.enabled = False

    # XDP shared state: store fds in allocated memory so workers can access them
    var xdp_shared_fds = alloc[Int32](2)  # [0] = xskmap_fd, [1] = bpf_prog_fd
    xdp_shared_fds[unsafe_offset=0] = xdp_shared_xskmap_fd
    xdp_shared_fds[unsafe_offset=1] = xdp_shared_bpf_fd

    # #42: the workers' pub/sub inboxes, for PUBLISH across workers (no-op at -w 1).
    external_call["pion_pubsub_init", NoneType](Int32(n_workers))

    # Mojo 1.0: spawn workers via pthreads (see pion_worker_entry above the
    # heap import below). Blocks forever — workers never exit in normal
    # operation, same contract as the old parallelize[worker_task](n, n).
    var _boot_ctx = alloc[Int64](8)
    _boot_ctx[unsafe_offset=0] = Int64(Int(shared_hnsw_ptr))
    _boot_ctx[unsafe_offset=1] = Int64(Int(cluster_ptr))
    _boot_ctx[unsafe_offset=2] = Int64(0)    # was the pub/sub ring (#42: per-worker inboxes in C)
    _boot_ctx[unsafe_offset=3] = Int64(Int(secondary_listen_fds))
    _boot_ctx[unsafe_offset=4] = Int64(Int(xdp_shared_fds))
    _boot_ctx[unsafe_offset=5] = Int64(Int(shared_listen_fd))
    _boot_ctx[unsafe_offset=6] = Int64(Int(binary_listen_fd))
    _boot_ctx[unsafe_offset=7] = Int64(n_workers)
    _ = external_call["pion_spawn_workers", Int32](
        Int32(config.server.workers), _boot_ctx)



# ── #36: redis.call() from a running script ──────────────────────────────────
# lua_wrap.c resolves this with dlsym and calls it, synchronously, for each
# redis.call() / redis.pcall(): the command runs through the worker's own slow
# path (SlowPathHandler.script_dispatch), re-entrantly. `ctx` is the
# SlowPathHandler the running EVAL/FCALL set as the Lua state's host. Lives in
# main.mojo for the reason the next export gives (root-module exports only);
# the -u link flags keep it. Must not raise at the ABI boundary.
@export
def pion_script_dispatch(ctx: Pointer[NoneType, MutUntrackedOrigin], argc: Int64,
                         argv: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                         lens: Pointer[Int64, MutUntrackedOrigin], flags: Int64, resp: Int64,
                         reply: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
                         wrote: Pointer[Int64, MutUntrackedOrigin]) -> Int64:
    var sp = Pointer[SlowPathHandler, MutUntrackedOrigin](unsafe_from_address=Int(ctx))
    return Int64(sp[].script_dispatch(Int(argc), argv, lens, Int(flags), Int(resp), reply, wrote))


# ── Mojo 1.0 migration: worker spawn ─────────────────────────────────────────
# std.algorithm.parallelize moved to the `max` package, which the server build
# must not depend on. Workers are raw pthreads (worker_spawn_wrap.c) calling
# back into this @export via dlsym — the gh #199 build-lane pattern. Lives in
# main.mojo because only the root module's exports reach the dynamic table,
# and the -u link flag makes a regression a loud link error.
# ctx layout (Int64 slots, packed in main(), outlives workers — main() blocks
# in pion_spawn_workers): [0]=SharedHNSWView* [1]=ClusterState*
# [2]=unused (was the pub/sub ring) [3]=secondary_listen_fds(Int32*) [4]=xdp_shared_fds
# (Int32*) [5]=shared_listen_fd [6]=binary_listen_fd [7]=n_workers.
@export
def pion_worker_entry(ctx: Pointer[Int64, MutUntrackedOrigin], worker_idx: Int64):
    # Re-derive the names the old closure captured; the body below is the old
    # worker_task body, unchanged.
    var shared_hnsw_ptr = Pointer[SharedHNSWView, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=0]))
    var cluster_ptr = Pointer[ClusterState, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=1]))
    var secondary_listen_fds = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=3]))
    var xdp_shared_fds = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=4]))
    var shared_listen_fd = Int32(ctx[unsafe_offset=5])
    var binary_listen_fd = Int32(ctx[unsafe_offset=6])
    var n_workers = Int(ctx[unsafe_offset=7])
    var i = Int(worker_idx)
    try:
        var worker_config = PionConfig()
        var args = sys.argv()
        for j in range(len(args)):
            if (args[j] == "-p" or args[j] == "--port") and j+1 < len(args):
                worker_config.server.port = atol(args[j+1])
            elif args[j] == "-G" or args[j] == "--ai-gateway":
                worker_config.ai.enable_gateway = True
            elif args[j] == "-M" or args[j] == "--max-engine":
                worker_config.ai.enable_max_engine = True
            elif args[j] == "--huge-pages":
                worker_config.server.use_huge_pages = True
            elif args[j] == "--no-huge-pages":
                worker_config.server.use_huge_pages = False
            elif args[j] == "--affinity":
                worker_config.server.strict_affinity = True
            elif args[j] == "--no-affinity":
                worker_config.server.strict_affinity = False
            elif args[j] == "--cluster":
                worker_config.cluster.enabled = True
            elif (args[j] == "--cluster-host") and j + 1 < len(args):
                worker_config.cluster.my_host = args[j + 1]
            elif (args[j] == "--cluster-nodes") and j + 1 < len(args):
                worker_config.cluster.peer_nodes = args[j + 1]
            elif args[j] == "--cluster-replica":
                worker_config.cluster.is_replica = True
            elif (args[j] == "--cluster-primary-host") and j + 1 < len(args):
                worker_config.cluster.primary_host = args[j + 1]
            elif (args[j] == "--cluster-primary-port") and j + 1 < len(args):
                try:
                    worker_config.cluster.primary_port = atol(args[j+1])
                except:
                    pass
            elif (args[j] == "--gossip-ping-ms") and j + 1 < len(args):
                try:
                    worker_config.cluster.gossip_ping_ms = atol(args[j+1])
                except:
                    pass
            elif args[j] == "--flare":
                worker_config.embedding.enabled = True
                worker_config.llm.enabled = True
                worker_config.llm.port = 11434
                worker_config.llm.model = "llama3.1:8b"
            elif args[j] == "--emb-enabled":
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-model") and j + 1 < len(args):
                worker_config.embedding.model = args[j+1]
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-host") and j + 1 < len(args):
                worker_config.embedding.host = args[j+1]
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-port") and j + 1 < len(args):
                worker_config.embedding.port = atol(args[j+1])
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-dim") and j + 1 < len(args):
                worker_config.embedding.dimensions = atol(args[j+1])
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-query-prefix") and j + 1 < len(args):
                worker_config.embedding.query_prefix = args[j+1]
                worker_config.embedding.enabled = True
            elif (args[j] == "--emb-doc-prefix") and j + 1 < len(args):
                worker_config.embedding.doc_prefix = args[j+1]
                worker_config.embedding.enabled = True
            elif args[j] == "--llm-enabled":
                worker_config.llm.enabled = True
            elif (args[j] == "--llm-port") and j + 1 < len(args):
                worker_config.llm.port = atol(args[j+1])
                worker_config.llm.enabled = True
            elif (args[j] == "--llm-model") and j + 1 < len(args):
                worker_config.llm.model = args[j+1]
            elif (args[j] == "--profile") and j + 1 < len(args):
                worker_config.apply_profile(args[j+1])
            elif args[j] == "--gpu":
                worker_config.vector.has_gpu = True
            elif args[j] == "--polarquant":
                worker_config.vector.polarquant = True
            elif args[j] == "--turboquant":
                worker_config.vector.turboquant = True
            elif args[j] == "--nanoquant":
                worker_config.vector.nanoquant = True
            elif args[j] == "--sqpoll":
                worker_config.server.use_sqpoll = True
            elif args[j] == "--xdp":
                worker_config.server.use_xdp = True
            elif (args[j] == "--xdp-interface" or args[j] == "--xdp-iface") and j + 1 < len(args):
                worker_config.server.use_xdp = True
                worker_config.server.xdp_interface = args[j+1]
            elif args[j] == "--kvcache":
                worker_config.server.kvcache_enabled = True
            elif args[j] == "--no-wal":
                worker_config.server.no_wal = True
            elif args[j] == "--wal-size" and j + 1 < len(args):
                try:
                    worker_config.server.wal_size_mb = atol(args[j+1])
                except:
                    pass
            elif args[j] == "--wal-max-segments" and j + 1 < len(args):
                try:
                    worker_config.server.wal_max_segments = atol(args[j+1])
                except:
                    pass
            elif args[j] == "--wal-full-policy" and j + 1 < len(args):
                if args[j+1] == "drop":
                    worker_config.server.wal_refuse_when_full = False
                elif args[j+1] == "refuse":
                    worker_config.server.wal_refuse_when_full = True
            elif args[j] == "--blob-threshold" and j + 1 < len(args):
                try:
                    worker_config.server.blob_threshold = atol(args[j+1])
                except:
                    pass
            elif args[j] == "--no-blob-tier":
                worker_config.server.no_blob_tier = True
            elif args[j] == "--iouring":
                worker_config.server.use_iouring = True
            elif args[j] == "--epoll":
                worker_config.server.use_epoll = True
            elif args[j] == "--ns-prefix" and j + 1 < len(args):
                worker_config.server.ns_prefix = args[j + 1]
            elif args[j] == "--requirepass" and j + 1 < len(args):
                worker_config.server.requirepass = args[j + 1]  # gh #100 (C2)
            elif args[j] == "--requirepass-file" and j + 1 < len(args):
                # gh #258: this parse loop is SEPARATE from main()'s and builds
                # each worker's own config from argv. Handling the flag only in
                # main() set the password for the bind decision and left every
                # worker with an empty one — the server announced "a password is
                # set", bound 0.0.0.0, and then answered AUTH with "no password
                # is set". Same two-parse-site trap as gh #253's worker cap.
                var _wbuf = alloc[UInt8](512)
                var _wpathz = args[j + 1] + "\0"
                var _wlen = Int(external_call["pion_read_secret_file", Int32](
                    _wpathz.unsafe_ptr(), _wbuf, Int32(512)))
                if _wlen > 0:
                    worker_config.server.requirepass = String(
                        StringSpan[MutUntrackedOrigin](
                            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                                unsafe_ptr=_wbuf, length=_wlen)))
                _wbuf.unsafe_free()
            elif args[j] == "--tenant" and j + 1 < len(args):
                # gh #101: already validated in the main parse loop.
                if worker_config.server.tenants.byte_length() > 0:
                    worker_config.server.tenants += "\n"
                worker_config.server.tenants += args[j + 1]
            elif args[j] == "--moe-cache" and j + 1 < len(args):
                worker_config.server.moe_cache_path = args[j + 1]
            elif args[j] == "--moe-cache-mib" and j + 1 < len(args):
                try:
                    worker_config.server.moe_cache_mib = atol(args[j + 1])
                except:
                    pass
            elif (args[j] == "--dim") and j + 1 < len(args):
                try:
                    worker_config.vector.dimensions = atol(args[j+1])
                except:
                    pass
            elif (args[j] == "--max-elements") and j + 1 < len(args):
                try:
                    worker_config.vector.max_elements = atol(args[j+1])
                except:
                    pass
            elif args[j] == "--inference":
                worker_config.inference.enabled = True
            elif (args[j] == "--inference-socket") and j + 1 < len(args):
                worker_config.inference.socket_path = args[j+1]
                worker_config.inference.enabled = True
            elif (args[j] == "--inference-emb-model") and j + 1 < len(args):
                worker_config.inference.default_embedding_model = args[j+1]
                worker_config.inference.enabled = True
            elif (args[j] == "--inference-llm-model") and j + 1 < len(args):
                worker_config.inference.default_llm_model = args[j+1]
                worker_config.inference.enabled = True
            elif args[j] == "--metal-attention":
                worker_config.metal_attention.enabled = True
            elif args[j] == "--metal-attention-fp16":
                worker_config.metal_attention.enabled = True
                worker_config.metal_attention.fp16 = True
            elif args[j] == "--cuda-attention":
                worker_config.cuda_attention.enabled = True
            elif (args[j] == "--fa-window") and j + 1 < len(args):
                try:
                    var w = atol(args[j+1])
                    worker_config.metal_attention.fa_window = w
                    worker_config.cuda_attention.fa_window = w
                except:
                    worker_config.metal_attention.fa_window = 0
                    worker_config.cuda_attention.fa_window = 0

        # Pass shared XDP fds to worker config (created before parallelize)
        worker_config.server.xdp_shared_xskmap_fd = xdp_shared_fds[unsafe_offset=0]
        worker_config.server.xdp_shared_bpf_fd = xdp_shared_fds[unsafe_offset=1]

        # Auto-detect Ollama in the worker too (same logic as main thread)
        var worker_no_auto = False
        var worker_no_auto_embed = False
        var worker_nle_embed = False
        var worker_inf_socket_explicit = False  # gh #424
        for j in range(len(args)):
            if args[j] == "--no-auto-detect": worker_no_auto = True
            if args[j] == "--no-auto-embed": worker_no_auto_embed = True
            if args[j] == "--inference-socket": worker_inf_socket_explicit = True
            if args[j] == "--nle-embed":
                worker_nle_embed = True
                worker_no_auto_embed = True  # NLE replaces PyTorch sidecar
        if not worker_no_auto and not worker_config.embedding.enabled and not worker_nle_embed:
            _try_auto_detect_ollama(worker_config)

        # --nle-embed: Apple NLEmbedding instead of PyTorch sidecar (mirror main thread).
        if worker_nle_embed and not worker_config.embedding.enabled and worker_config.server.profile != "kv":
            worker_config.embedding.enabled = True
            worker_config.embedding.nle = True
            worker_config.embedding.dimensions = 512
            worker_config.embedding.threshold = 0.80
            worker_config.embedding.model = "apple-nle-sentence-en"
        # A3: Auto-embed fallback in worker (mirrors main thread logic)
        elif not worker_no_auto_embed and not worker_config.embedding.enabled and not worker_config.inference.enabled and worker_config.server.profile != "kv":
            worker_config.inference.enabled = True
            worker_config.inference.default_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            worker_config.embedding.enabled = True
            worker_config.embedding.dimensions = 384
            worker_config.embedding.threshold = 0.85
        # gh #373: explicit --inference with the default model — mirror main().
        elif (not worker_no_auto_embed and not worker_config.embedding.enabled
              and worker_config.inference.enabled and worker_config.server.profile != "kv"
              and worker_config.inference.default_embedding_model == "sentence-transformers/all-MiniLM-L6-v2"):
            worker_config.embedding.enabled = True
            worker_config.embedding.dimensions = 384
            worker_config.embedding.threshold = 0.85

        # gh #424: match the master's per-uid/per-port sidecar socket default so
        # the worker's InferenceBridge connects to the socket the master spawned
        # (the master applied the identical rule before spawning). Explicit
        # --inference-socket is honored on both sides.
        if worker_config.inference.enabled and not worker_inf_socket_explicit:
            var _wuid = Int(external_call["getuid", UInt32]())
            worker_config.inference.socket_path = String("/tmp/pion-") + String(_wuid) \
                + String("-") + String(worker_config.server.port) + String(".inference.sock")

        # --- Huge Pages Fallback ---
        if CompilationTarget.is_macos() and worker_config.server.use_huge_pages:
            print("Warning: Huge Pages are not supported on macOS. Falling back to standard pages.")
            worker_config.server.use_huge_pages = False
        # Linux: the profiles leave huge pages off (a stock box reserves none),
        # so this is on only when `--huge-pages` asked for it. It used to be
        # forced off here, which made that flag a no-op the banner still
        # reported as on. SlabAllocator falls back to normal pages per mmap.

        # Step 5: Pin to P-cores (always on macOS; QoS class must be set before affinity).
        if CompilationTarget.is_macos():
            set_thread_qos_user_interactive()
        if worker_config.server.strict_affinity:
            var _aff = set_thread_affinity(i)
            if _aff.byte_length() > 0:
                print("Worker " + String(i) + " " + _aff)
            else:
                print("Worker " + String(i) + ": --affinity requested but not applied")

        # gh #258: same env-var fallback as main()'s parse. This loop rebuilds
        # each worker's config from argv, so anything main() resolved from the
        # environment rather than a flag has to be resolved again here.
        if worker_config.server.requirepass.byte_length() == 0:
            var _wenv = external_call["getenv", Pointer[UInt8, MutUntrackedOrigin]](
                String("PION_REQUIREPASS\0").unsafe_ptr())
            if is_not_null(_wenv):
                var _wn = 0
                while _wn < 512 and _wenv[unsafe_offset=_wn] != 0:
                    _wn += 1
                if _wn > 0:
                    worker_config.server.requirepass = String(
                        StringSpan[MutUntrackedOrigin](
                            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                                unsafe_ptr=_wenv, length=_wn)))

        var worker_nodes = List[String]()
        worker_nodes.append("127.0.0.1:" + String(worker_config.server.port))
        
        var db = Pion(worker_nodes^, worker_config, shared_hnsw_ptr, shared_listen_fd, worker_id=i,
                      num_workers=n_workers,
                      secondary_listen_fd=secondary_listen_fds[unsafe_offset=i],
                      binary_listen_fd=binary_listen_fd,
                      cluster=cluster_ptr)
        db.run_server()
    except e:
        # stderr + explicit message: worker deaths were invisible when this
        # printed a bare line into a buffered stdout from a raw pthread.
        print("Worker", i, "failed:", String(e))


from src.common.heap import MinHeap as _GH199MinHeap, MaxHeap as _GH199MaxHeap


# ── gh #199: pthread lane worker for the parallel FT.OPTIMIZE link phase ──────
# Lives in main.mojo, NOT hnsw.mojo: Mojo b2 elides unreferenced @export defs
# in imported modules (verified: symbol absent from the binary, recall 0.0);
# only the root module's exports reach the dynamic table. The -u link flags
# make any regression of this a loud link error instead of a silent dead graph.
# Called from build_pool_wrap.c threads via dlsym("pion_gh199_build_lane") —
# Mojo b2 removed `fn`, so @export + dlsym is the only C-callback path (see
# memory: the executable links with -export_dynamic for this). Must stay
# non-raising at the ABI boundary, no print, no parallelize.
# ctx layout (Int64 slots): [0]=HNSWGraph*, [1]=ingest_fp32*, [2]=n,
# [3]=entry_idx (pre-pass max-level node; linked via backlinks only),
# [4]=n_threads (stride).
@export
def pion_gh199_build_lane(ctx: Pointer[Int64, MutUntrackedOrigin], lane_idx: Int64):
    var g = Pointer[HNSWGraph, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=0]))
    var fp32 = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(ctx[unsafe_offset=1]))
    var n = Int(ctx[unsafe_offset=2])
    var entry_idx = Int(ctx[unsafe_offset=3])
    var n_threads = Int(ctx[unsafe_offset=4])
    var cand = _GH199MinHeap()
    var res = _GH199MaxHeap()
    var visited = alloc[UInt16](g[].max_elements)
    unsafe_memset(visited.unsafe_bitcast[UInt8](), 0, g[].max_elements * 2)
    var epoch_cell = alloc[UInt16](1)
    epoch_cell[unsafe_offset=0] = 0
    var bi = Int(lane_idx)
    while bi < n:
        if bi != entry_idx:
            try:
                g[]._insert_to_graph_mt(bi, g[].nodes[unsafe_offset=bi].max_level, fp32.unsafe_offset(bi * g[].dim),
                                        cand, res, visited, epoch_cell)
            except:
                pass  # a failed insert loses one node's links, never the build
        bi += n_threads
    visited.unsafe_free()
    epoch_cell.unsafe_free()
