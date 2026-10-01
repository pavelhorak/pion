from src.common.env import Environment

@fieldwise_init
struct VectorConfig(Copyable, Movable, ImplicitlyCopyable):
    var dimensions: Int
    var max_elements: Int
    var M: Int
    var ef_construction: Int
    var use_int4: Bool
    var use_bq: Bool
    var has_gpu: Bool
    var polarquant: Bool  # M6: WHT rotation + INT4 quantization (replaces naive INT4)
    var turboquant: Bool  # M7: TurboQuant 3-bit + QJL error correction
    var nanoquant: Bool   # N4: NanoQuant 2-bit block quantization

    def __init__(out self):
        self.dimensions = 1536
        self.max_elements = 600000
        self.M = 16
        self.ef_construction = 100
        self.use_int4 = False
        self.use_bq = False
        self.has_gpu = False
        self.polarquant = False
        self.turboquant = False
        self.nanoquant = False

@fieldwise_init
struct ServerConfig(Copyable, Movable, ImplicitlyCopyable):
    var port: Int
    var workers: Int
    var profile: String
    var use_huge_pages: Bool
    var strict_affinity: Bool
    var enable_sharding: Bool
    var wal_sync_mode: String   # "async" (default, MS_ASYNC) | "sync" (fdatasync on every SAVE)
    var use_sqpoll: Bool        # io_uring SQPOLL: kernel-side SQ polling (eliminates enter() syscall)
    var use_xdp: Bool           # XDP/AF_XDP kernel bypass: zero-copy packet processing
    var xdp_interface: String   # NIC interface for XDP attach (e.g. "eth0", "lo")
    var xdp_shared_xskmap_fd: Int32  # Shared XSKMAP fd (multi-worker XDP, set before parallelize)
    var xdp_shared_bpf_fd: Int32     # Shared BPF prog fd (multi-worker XDP, set before parallelize)
    var kvcache_enabled: Bool   # M14: KV cache store for externalized attention
    var no_wal: Bool             # --no-wal: skip WAL append/sync (benchmark mode, proves WAL overhead)
    # gh #149: WAL segment sizing. The log rotates at wal_size_mb and keeps at most
    # wal_max_segments sealed segments; past that appends are refused *loudly*
    # instead of silently dropped.
    var wal_size_mb: Int
    var wal_max_segments: Int
    # gh #260: what to do once the log is full and appends are being refused.
    # True (default) => reply -MISCONF to keyspace writes, because acknowledging
    # a write the server knows it cannot persist is a lie the client has no way
    # to detect. False => --wal-full-policy drop, the pre-gh #260 behaviour, for
    # cache-only deployments that have explicitly opted out of durability.
    var wal_refuse_when_full: Bool
    # gh #163: values >= blob_threshold bytes are stored in the file-backed blob
    # tier instead of anonymous heap — durable by construction, reclaimable under
    # memory pressure, and never memcpy'd into the WAL ring.
    var blob_threshold: Int
    var no_blob_tier: Bool
    var use_iouring: Bool        # --iouring: force io_uring (default on Linux)
    var use_epoll: Bool          # --epoll: force epoll (wins at low concurrency P=1)
    # Multi-tenant isolation: when set, KV.PREFIX.REGISTER/LOOKUP/OWNER/SAVE
    # require the supplied namespace to start with this exact byte string.
    # Empty = no enforcement (single-tenant). The canonical multi-tenant
    # deployment runs one pion-server per tenant with `--ns-prefix tenant_<id>:`.
    # See doc/multi_tenant.md.
    var ns_prefix: String
    # gh #61 Stage 2: MoE-expert tier. Empty = disabled (Stage 1 stub
    # handlers active). Non-empty = path to a directory containing MoE
    # safetensors shards; loaded at startup, manifest populated.
    var moe_cache_path: String
    # Cache budget in MiB. Default 1024 mirrors the Phase-0 empirical
    # sweet-spot on 16 GB Mac mini.
    var moe_cache_mib: Int
    # gh #100 (C2): server password. Empty = no auth (default). When non-empty,
    # every RESP connection must issue `AUTH <password>` before any other command;
    # the binary port (port+1) requires the same via its 0x30 AUTH frame.
    var requirepass: String
    # gh #101: tenant credentials — newline-joined "NAME=PASSWORD" entries from
    # repeatable `--tenant` flags (validated at arg-parse). Empty = tenant mode
    # off. When non-empty, --requirepass is required (it becomes the admin
    # credential) and AUTH NAME PASSWORD binds a connection to tenant NAME's
    # namespace. Parsed into a TenantTable per worker (src/commands/tenant.mojo).
    var tenants: String
    # gh #138: crash/exit diagnostics. `crash_log` gets one appended line per
    # process start / catchable death; `status_file` is a fixed-size record
    # rewritten once a second (pid, uptime, RSS, peak RSS, event-loop ticks) so
    # an *uncatchable* death (jetsam / OOM-killer SIGKILL) still leaves the last
    # thing the process knew about itself. Empty path = that half is disabled;
    # `--no-crash-log` disables both. rss_warn_pct: one-shot warning when RSS
    # crosses this percentage of physical RAM (>100 disables).
    var crash_log: String
    var status_file: String
    var rss_warn_pct: Int
    # gh #253: workers are shared-nothing — each owns a PRIVATE keyspace, and a
    # connection is bound to whichever worker won the accept() race. A `SET`
    # acknowledged on one connection is therefore invisible to a `GET` on
    # another, which is exactly the shape of every pooled client (redis-py's
    # default ConnectionPool included). `-w N` for N > 1 is refused unless this
    # flag is also passed, so nobody reaches that semantics by accident.
    var independent_workers: Bool
    # gh #258: the interface to bind every listener to (RESP, port+1 binary lane,
    # port+10000 replication, gossip/Raft). Empty means "decide from the security
    # posture at parse time": loopback when no password is set, all interfaces
    # when one is — Redis's protected-mode precedent. Every listener used to bind
    # INADDR_ANY unconditionally, because the sockaddr was memset to zero and
    # nobody filled in sin_addr.
    var bind_addr: String
    # gh #261: refuse memory-growing writes (Redis's `denyoom` set, plus the
    # substrate ingest commands) while process RSS is above this many bytes,
    # with Redis's own -OOM error. 0 = unlimited. No eviction: refusal is the
    # whole policy (`maxmemory-policy noeviction`).
    var maxmemory: Int

    def __init__(out self):
        self.port = 1974
        # gh #253: 1, not 8 — see `independent_workers`. Multi-worker is an
        # explicit opt-in, never a default.
        self.workers = 1
        self.profile = "auto"
        self.use_huge_pages = False
        self.strict_affinity = False
        self.enable_sharding = False
        self.wal_sync_mode = "async"
        self.use_sqpoll = False
        self.use_xdp = False
        self.xdp_interface = "eth0"
        self.xdp_shared_xskmap_fd = Int32(-1)
        self.xdp_shared_bpf_fd = Int32(-1)
        self.kvcache_enabled = False
        self.no_wal = False
        self.wal_size_mb = 256
        self.wal_max_segments = 32
        self.wal_refuse_when_full = True
        self.blob_threshold = 1024 * 1024
        self.no_blob_tier = False
        self.use_iouring = False
        self.use_epoll = False
        self.ns_prefix = ""
        self.moe_cache_path = ""
        self.moe_cache_mib = 1024
        self.requirepass = ""
        self.tenants = ""
        # Defaults are port-stamped at parse time in main.mojo so concurrent
        # servers on different ports never share a breadcrumb file.
        self.crash_log = ""
        self.status_file = ""
        self.rss_warn_pct = 70
        self.independent_workers = False
        self.bind_addr = ""
        self.maxmemory = 0

@fieldwise_init
struct AIConfig(Copyable, Movable, ImplicitlyCopyable):
    var enable_gateway: Bool
    var enable_max_engine: Bool
    var model_path: String

    def __init__(out self):
        self.enable_gateway = False
        self.enable_max_engine = False
        self.model_path = "models/all-MiniLM-L6-v2.onnx"

@fieldwise_init
struct EmbeddingConfig(Copyable, Movable, ImplicitlyCopyable):
    var host: String
    var port: Int
    var model: String
    var dimensions: Int
    var threshold: Float32  # cosine similarity threshold (0–1); 0.95 = very high similarity
    var enabled: Bool
    var nle: Bool           # Apple NLEmbedding (macOS, ANE-accelerated, 512-dim)
    # gh #140: asymmetric retrievers (EmbeddingGemma, E5, BGE) are trained with
    # distinct query/document instruction prefixes and lose accuracy without
    # them. Empty by default — symmetric models must NOT get a prefix.
    var query_prefix: String
    var doc_prefix: String

    def __init__(out self):
        self.host = "127.0.0.1"
        self.port = 11434      # Ollama default
        self.model = "nomic-embed-text"
        self.dimensions = 768
        self.threshold = 0.95
        self.enabled = False   # requires explicit opt-in
        self.nle = False
        self.query_prefix = ""
        self.doc_prefix = ""

@fieldwise_init
struct LLMConfig(Copyable, Movable, ImplicitlyCopyable):
    var host: String
    var port: Int
    var model: String
    var enabled: Bool

    def __init__(out self):
        self.host = "127.0.0.1"
        self.port = 8000          # MAX Serve / OpenAI-compatible server default
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.enabled = False      # requires explicit opt-in (start max serve first)

@fieldwise_init
struct ClusterConfig(Copyable, Movable, ImplicitlyCopyable):
    var enabled: Bool
    var my_host: String      # advertised IP for CLUSTER NODES
    var peer_nodes: String   # comma-separated "host:port,host:port,..." including self
    # Replica mode: set --cluster-replica to replicate from a primary
    var is_replica: Bool
    var primary_host: String # primary's advertised host (for replication connect)
    var primary_port: Int    # primary's main port (replication port = primary_port + 10000)
    # Gossip health timeouts
    var gossip_ping_ms: Int  # interval between PING rounds (ms)
    var pfail_threshold: Int # consecutive ping failures → pfail
    var fail_threshold: Int  # consecutive ping failures → fail

    def __init__(out self):
        self.enabled = False
        self.my_host = "127.0.0.1"
        self.peer_nodes = ""
        self.is_replica = False
        self.primary_host = ""
        self.primary_port = 0
        self.gossip_ping_ms = 1000
        self.pfail_threshold = 5
        self.fail_threshold = 15


@fieldwise_init
struct InferenceConfig(Copyable, Movable, ImplicitlyCopyable):
    var enabled: Bool
    var socket_path: String
    var default_embedding_model: String
    var default_llm_model: String

    def __init__(out self):
        self.enabled = False
        self.socket_path = "/tmp/pion_inference.sock"
        self.default_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.default_llm_model = ""

@fieldwise_init
struct MetalAttentionConfig(Copyable, Movable, ImplicitlyCopyable):
    """In-process Metal SDPA engine (no Python sidecar). M=1 / D=128 fast path
    for ATTEND.PREFIX.QUERY. Beats the MLX sidecar by 1.34-1.55× end-to-end on
    Apple Silicon (proof: tests/bench_msl_sdpa_q1.m)."""
    var enabled: Bool
    var fp16: Bool   # Use FP16 kernel (matches vanilla mlx-lm precision; --metal-attention-fp16).
    # --fa-window N: sliding-window flash-attention. Kernel scans only the last
    # N tokens of K/V instead of the full prefix. 0 = full attention (default).
    # Lossless for hybrid-attention models (Qwen3.5, layers with full softmax
    # every Nth slice) — the unrelated full-attention layers catch anything the
    # windowed layers miss. Lossy on plain dense transformers; document in the
    # caller's docs before enabling.
    var fa_window: Int

    def __init__(out self):
        self.enabled = False
        self.fp16 = False
        self.fa_window = 0


@fieldwise_init
struct CudaAttentionConfig(Copyable, Movable, ImplicitlyCopyable):
    """gh #9: native in-process CUDA SDPA. --cuda-attention enables.
    Linux + CUDA-equipped host required. Falls back gracefully (engine
    .available = False) on non-CUDA hosts."""
    var enabled: Bool
    var fa_window: Int

    def __init__(out self):
        self.enabled = False
        self.fa_window = 0


@fieldwise_init
struct PionConfig(Copyable, Movable, ImplicitlyCopyable):
    var server: ServerConfig
    var vector: VectorConfig
    var ai: AIConfig
    var embedding: EmbeddingConfig
    var llm: LLMConfig
    var cluster: ClusterConfig
    var inference: InferenceConfig
    var metal_attention: MetalAttentionConfig
    var cuda_attention: CudaAttentionConfig

    def __init__(out self):
        self.server = ServerConfig()
        self.vector = VectorConfig()
        self.ai = AIConfig()
        self.embedding = EmbeddingConfig()
        self.llm = LLMConfig()
        self.cluster = ClusterConfig()
        self.inference = InferenceConfig()
        self.metal_attention = MetalAttentionConfig()
        self.cuda_attention = CudaAttentionConfig()
        self._apply_smart_profile()

    def apply_profile(mut self, profile: String):
        """Apply a deployment profile. Called after CLI parsing to override smart defaults.
        Profiles: kv (KV-only, no vector), vector (KV + HNSW), full (everything), ai (vector + AI)."""
        if profile == "kv":
            self.server.profile = "kv"
            self.vector.max_elements = 1          # minimal HNSW stub (~few KB vs ~650MB)
            self.embedding.enabled = False
            self.llm.enabled = False
            self.inference.enabled = False
        elif profile == "vector":
            self.server.profile = "vector"
            # Keep vector defaults, disable AI
            self.embedding.enabled = False
            self.llm.enabled = False
            self.inference.enabled = False
        elif profile == "ai":
            self.server.profile = "ai"
            # Vector + AI features, enforce single worker (semantic cache is per-worker)
            self.server.workers = 1
        elif profile == "full":
            self.server.profile = "full"
            # Everything enabled — no overrides needed

    def _apply_smart_profile(mut self):
        var env = Environment()
        self.vector.has_gpu = env.has_gpu

        if env.is_embedded:
            self.server.profile = "embedded"
            self.server.workers = 1
            self.server.use_huge_pages = False
            self.server.strict_affinity = False
            self.vector.use_bq = False  # BQ disabled: recall=0.65 on OpenAI embeddings (Hamming too coarse)
            self.vector.M = 16
        elif env.is_cloud:
            self.server.profile = "cloud"
            # gh #253: this used to default to 16. A cloud box is exactly where a
            # pooled client connects, and 16 independent keyspaces silently break
            # read-your-writes — so the throughput default is now opt-in
            # (`-w 16 --independent-workers`), not inherited from the environment.
            self.server.workers = 1
            self.server.use_huge_pages = True
            self.server.strict_affinity = True
            # Was `use_int4 = True` from the first smart-config commit; V13
            # abandoned naive INT4 (recall 0.74) and fixed only the desktop
            # branch below. Worse than low recall: INT4 stores dim/2-byte rows
            # and _beam_search_1536 reads dim-byte INT8 rows, so on any box
            # with >= 16 online cores the first FT.SEARCH after FT.OPTIMIZE
            # SIGSEGV'd the worker. Quantized variants are opt-in flags
            # (--polarquant / --turboquant / --nanoquant), never a core count.
            self.vector.use_int4 = False
            self.vector.ef_construction = 200 # Higher accuracy for cloud
        else:
            self.server.profile = "desktop"
            self.server.workers = 1   # gh #253: was 8 — see ServerConfig.independent_workers
            self.server.use_huge_pages = True
            self.server.strict_affinity = False
            self.vector.M = 16  # V10: restored to M=16 (M=12 tested V38C: ef=150 recall=0.9235 fails gate; ef=175 recall ok but QPS regresses vs M=16 ef=150)
            self.vector.use_int4 = False  # V13 INT4 ABANDONED: recall=0.74 (16-level quant too coarse for 1536-dim cosine)
            
    def dump(self):
        print("--- Pion Configuration ---")
        print("Profile:    " + self.server.profile)
        print("Port:       " + String(self.server.port))
        print("Workers:    " + String(self.server.workers) +
              (" (INDEPENDENT KEYSPACES)" if self.server.workers > 1 else ""))
        print("Huge Pages: " + String(self.server.use_huge_pages))
        print("Affinity:   " + String(self.server.strict_affinity))
        print("Vector Dim: " + String(self.vector.dimensions))
        print("INT4:       " + String(self.vector.use_int4))
        print("PolarQuant: " + String(self.vector.polarquant))
        print("TurboQuant: " + String(self.vector.turboquant))
        if self.vector.nanoquant:
            print("NanoQuant:  True (EXPERIMENTAL: recall@100 ~0.46 on the gate dataset)")
        else:
            print("NanoQuant:  False")
        print("BQ:         " + String(self.vector.use_bq))
        print("GPU:        " + String(self.vector.has_gpu))
        print("AI Gateway: " + String(self.ai.enable_gateway))
        print("MAX Engine: " + String(self.ai.enable_max_engine))
        if self.embedding.enabled:
            if self.embedding.nle:
                print("Embedding:  Apple NLEmbedding (model=" + self.embedding.model + ", dim=" + String(self.embedding.dimensions) + ", in-process)")
            else:
                print("Embedding:  " + self.embedding.host + ":" + String(self.embedding.port) + " model=" + self.embedding.model)
        else:
            print("Embedding:  disabled")
        if self.llm.enabled:
            print("LLM:        " + self.llm.host + ":" + String(self.llm.port) + " model=" + self.llm.model)
        else:
            print("LLM:        disabled")
        if self.server.no_wal:
            print("WAL:        disabled (--no-wal benchmark mode)")
        if self.server.use_epoll:
            print("EPOLL:      forced (--epoll)")
        elif self.server.use_iouring:
            print("IO_URING:   forced (--iouring)")
        if self.server.use_sqpoll:
            print("SQPOLL:     enabled (kernel-side SQ polling)")
        if self.server.use_xdp:
            print("XDP:        enabled (AF_XDP on " + self.server.xdp_interface + ")")
        if self.inference.enabled:
            print("Inference:  enabled (socket=" + self.inference.socket_path + ")")
            print("  Emb Model: " + self.inference.default_embedding_model)
            if self.inference.default_llm_model:
                print("  LLM Model: " + self.inference.default_llm_model)
        else:
            print("Inference:  disabled")
        if self.metal_attention.enabled:
            # gh #281: this line used to say "enabled", printed straight from the
            # flag, ~20 lines before the engine actually tried to load its shader
            # library — so a server that then fell back to the MLX bridge had
            # already announced success. It reports the REQUEST; the engine
            # prints "Metal Attn: ACTIVE" or "Metal Attn: NOT ACTIVE" when it
            # knows. Same class as gh #257: never announce a capability you have
            # not yet obtained.
            var prec = "fp16" if self.metal_attention.fp16 else "fp32"
            if self.metal_attention.fa_window > 0:
                print("Metal Attn: requested (" + prec + ", fa-window=" + String(self.metal_attention.fa_window) + ")")
            else:
                print("Metal Attn: requested (" + prec + ")")
        if self.cuda_attention.enabled:
            if self.cuda_attention.fa_window > 0:
                print("CUDA Attn:  enabled (fa-window=" + String(self.cuda_attention.fa_window) + ")")
            else:
                print("CUDA Attn:  enabled")
        print("KV Cache:   " + String(self.server.kvcache_enabled) + " (--kvcache enables ATTEND.*/M13 router/M9 RAG)")
        if self.server.moe_cache_path.byte_length() > 0:
            print("MoE Cache:  enabled (path=" + self.server.moe_cache_path + ", " + String(self.server.moe_cache_mib) + " MiB)")
        print("Cluster:    " + String(self.cluster.enabled))
        if self.cluster.enabled:
            print("  Host:     " + self.cluster.my_host)
            print("  Nodes:    " + self.cluster.peer_nodes)
        print("--------------------------")
