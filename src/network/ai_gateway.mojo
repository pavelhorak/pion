"""FLAREGateway — mid-generation retrieval (FLARE) implemented natively in Mojo.

Architecture (in-process, no Python):
  AI.FLARE LOAD <text>           → embed + index text in per-worker HNSW
  AI.FLARE RUN  <query>          → FLARE generation loop (logprob monitoring + retrieval)
  AI.FLARE INFO                  → stats (doc count, tau, etc.)

The FLARE loop:
  1. Build prompt = system + context_facts + "Question: " + query + "\\nAnswer: " + generated
  2. Call upstream LLM (Ollama /api/generate) for CHUNK_TOKENS tokens with logprobs
  3. If min_logprob < log(tau): embed (generated + chunk), search HNSW,
     prepend new facts to context, discard uncertain chunk, retry
  4. Otherwise: accept chunk, append to generated, repeat until done or max_tokens

All retrieval is in-process (< 2ms). HTTP calls only to the upstream LLM.

Configuration (via LLMConfig and EmbeddingConfig in PionConfig):
  LLM:       llm.host, llm.port (Ollama: 11434), llm.model ("llama3.1:8b")
  Embedding: embedding.host, embedding.port, embedding.model, embedding.dimensions

Enable: config.llm.enabled = True and config.embedding.enabled = True
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.collections import List, Span
from std.math import log

from src.vector.hnsw import HNSWGraph
from src.network.embedding_client import EmbeddingClient
from src.network.llm_client import LLMClient
from src.network.nle_embedding_engine import NLEmbeddingEngine

# Max documents in the FLARE knowledge base (per worker)
comptime FLARE_KB_CAPACITY = 5000
# Default FLARE generation parameters
comptime FLARE_DEFAULT_TAU         = Float32(0.3)   # probability threshold → logprob = ln(0.3) ≈ -1.2
comptime FLARE_DEFAULT_CHUNK_N     = 10             # tokens per generation chunk
comptime FLARE_DEFAULT_MAX_TOKENS  = 200            # total tokens budget
comptime FLARE_DEFAULT_MAX_RETRIES = 2              # max retrieval attempts per position
comptime FLARE_DEFAULT_K           = 3              # documents to retrieve per trigger
comptime FLARE_DEFAULT_EF          = 32             # HNSW search ef_runtime

# Pre-allocated buffer sizes
comptime FLARE_GENERATED_MAX = 4 * 1024 * 1024     # 4MB  — accumulated answer
comptime FLARE_CONTEXT_MAX   = 512 * 1024           # 512KB — retrieved facts
comptime FLARE_PROMPT_MAX    = 2 * 1024 * 1024     # 2MB  — full assembled prompt
comptime FLARE_CHUNK_MAX     = 64 * 1024            # 64KB  — single LLM chunk
comptime FLARE_SEARCH_MAX    = 64 * 1024            # 64KB  — search query scratch


struct FLAREGateway(Movable):
    """Per-worker FLARE gateway: in-process HNSW KB + FLARE generation loop."""

    var hnsw:         HNSWGraph
    var emb_client:   EmbeddingClient
    var nle:          NLEmbeddingEngine   # macOS native embed (--nle-embed)
    var llm:          LLMClient
    var embed_buf:    Pointer[Float32, MutUntrackedOrigin]
    var doc_texts:    List[String]
    var doc_count:    Int
    var dimensions:   Int

    # Default FLARE params (overridable per-command)
    var tau:          Float32
    var chunk_tokens: Int
    var max_tokens:   Int

    # Pre-allocated generation buffers (zero heap alloc in the hot loop)
    var generated_buf: Pointer[UInt8, MutUntrackedOrigin]
    var context_buf:   Pointer[UInt8, MutUntrackedOrigin]
    var prompt_buf:    Pointer[UInt8, MutUntrackedOrigin]
    var chunk_buf:     Pointer[UInt8, MutUntrackedOrigin]
    var search_buf:    Pointer[UInt8, MutUntrackedOrigin]
    # Scratch for LLM output values (allocated once in __init__)
    var min_lp_buf:    Pointer[Float32, MutUntrackedOrigin]
    var done_buf:      Pointer[UInt8, MutUntrackedOrigin]

    var enabled: Bool

    def __init__(out self,
                emb_host: String, emb_port: Int, emb_model: String,
                dimensions: Int,
                llm_host: String, llm_port: Int, llm_model: String,
                llm_enabled: Bool,
                enabled: Bool,
                nle_enabled: Bool = False):
        self.dimensions = dimensions
        self.enabled = enabled
        self.doc_count = 0
        self.tau = FLARE_DEFAULT_TAU
        self.chunk_tokens = FLARE_DEFAULT_CHUNK_N
        self.max_tokens = FLARE_DEFAULT_MAX_TOKENS
        self.doc_texts = List[String]()
        self.hnsw = HNSWGraph(FLARE_KB_CAPACITY, dimensions, M=16, ef_construction=32)
        self.emb_client = EmbeddingClient(emb_host, emb_port, emb_model, dimensions)
        self.nle = NLEmbeddingEngine(nle_enabled)
        self.llm = LLMClient(llm_host, llm_port, llm_model, llm_enabled)
        self.embed_buf    = alloc[Float32](dimensions)
        self.generated_buf = alloc[UInt8](FLARE_GENERATED_MAX)
        self.context_buf   = alloc[UInt8](FLARE_CONTEXT_MAX)
        self.prompt_buf    = alloc[UInt8](FLARE_PROMPT_MAX)
        self.chunk_buf     = alloc[UInt8](FLARE_CHUNK_MAX)
        self.search_buf    = alloc[UInt8](FLARE_SEARCH_MAX)
        self.min_lp_buf    = alloc[Float32](1)
        self.done_buf      = alloc[UInt8](1)

    def __moveinit__(out self, deinit take: Self):
        self.hnsw         = take.hnsw^
        self.emb_client   = take.emb_client^
        self.nle          = take.nle^
        self.llm          = take.llm^
        self.embed_buf    = take.embed_buf
        self.doc_texts    = take.doc_texts^
        self.doc_count    = take.doc_count
        self.dimensions   = take.dimensions
        self.tau          = take.tau
        self.chunk_tokens = take.chunk_tokens
        self.max_tokens   = take.max_tokens
        self.generated_buf = take.generated_buf
        self.context_buf   = take.context_buf
        self.prompt_buf    = take.prompt_buf
        self.chunk_buf     = take.chunk_buf
        self.search_buf    = take.search_buf
        self.min_lp_buf    = take.min_lp_buf
        self.done_buf      = take.done_buf
        self.enabled       = take.enabled

    @always_inline
    def _embed(mut self, text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int) -> Bool:
        """NLE → HTTP cascade. Same shape as SemanticCache.embed_into so
        --nle-embed flips on for FLARE too."""
        if self.nle.available and Int(self.nle.dimension) == self.dimensions:
            var dim = self.nle.embed(text_ptr, text_len, self.embed_buf, self.dimensions)
            if dim == self.dimensions:
                return True
        return self.emb_client.embed(text_ptr, text_len, self.embed_buf)

    def load(mut self, text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int) -> Bool:
        """Embed text and add to FLARE knowledge base. Returns True on success."""
        if not self.enabled or self.doc_count >= FLARE_KB_CAPACITY: return False
        if not self._embed(text_ptr, text_len): return False
        try:
            self.hnsw.add_and_insert(self.doc_count, self.embed_buf)
        except:
            return False
        # gh #115: was the same non-ASCII mangling the semantic cache had —
        # every byte >= 128 spelled '?', ONE PER BYTE, so an ingested UTF-8
        # document came back corrupted and longer. These are RAG source
        # documents; mangling them corrupts what the model is grounded on.
        var s = String(StringSpan[MutUntrackedOrigin](
            unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=text_ptr, length=text_len)))
        self.doc_texts.append(s)
        self.doc_count += 1
        return True

    def _retrieve_into_context(mut self,
                              query_ptr: Pointer[UInt8, MutUntrackedOrigin],
                              query_len: Int, k: Int) -> Int:
        """Embed query, search HNSW, write top-k doc texts into context_buf[0..].
        Returns bytes written (0 if nothing found)."""
        if self.doc_count < 1: return 0
        if not self._embed(query_ptr, query_len): return 0
        var scores = List[Float32]()
        var results: List[Int]
        try:
            results = self.hnsw.search_fp32_scored(self.embed_buf, k, scores, FLARE_DEFAULT_EF)
        except:
            return 0
        if len(results) == 0: return 0
        var ctx_len = 0
        for ri in range(len(results)):
            var doc_id = results[ri]
            if doc_id < 0 or doc_id >= self.doc_count: continue
            var doc = self.doc_texts[doc_id]
            var doc_bytes = doc.unsafe_ptr()
            var doc_len = doc.byte_length()
            if ctx_len + 2 + doc_len + 1 >= FLARE_CONTEXT_MAX: break
            self.context_buf[unsafe_offset=ctx_len] = 45; ctx_len += 1   # '-'
            self.context_buf[unsafe_offset=ctx_len] = 32; ctx_len += 1   # ' '
            unsafe_memcpy(dest=self.context_buf.unsafe_offset(ctx_len),
                   src=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(doc_bytes)), count=doc_len)
            ctx_len += doc_len
            self.context_buf[unsafe_offset=ctx_len] = 10; ctx_len += 1   # '\n'
        return ctx_len

    def run(mut self,
           query_ptr: Pointer[UInt8, MutUntrackedOrigin], query_len: Int,
           system_ptr: Pointer[UInt8, MutUntrackedOrigin], system_len: Int,
           tau: Float32, chunk_n: Int, max_tok: Int,
           out_buf: Pointer[UInt8, MutUntrackedOrigin], out_max: Int) -> Int:
        """Run the FLARE generation loop. Returns bytes written to out_buf."""
        if not self.enabled or not self.llm.enabled: return 0

        var effective_tau   = tau if tau > Float32(0.0) else self.tau
        var effective_chunk = chunk_n if chunk_n > 0 else self.chunk_tokens
        var effective_max   = max_tok if max_tok > 0 else self.max_tokens

        # Logprob threshold = ln(tau). Higher uncertainty = more negative logprob.
        # exp(logprob) < tau  ↔  logprob < ln(tau)
        var tau_logprob = Float32(log(Float64(effective_tau)))

        var generated_len = 0
        var context_len   = 0   # bytes currently in context_buf
        var tokens_done   = 0   # rough generated token count
        var retries       = 0

        while tokens_done < effective_max:
            # ── Build full prompt ────────────────────────────────────────────
            var pp = 0

            if system_len > 0 and pp + system_len + 2 < FLARE_PROMPT_MAX:
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp),
                       src=system_ptr, count=system_len)
                pp += system_len
                self.prompt_buf[unsafe_offset=pp] = 10; pp += 1
                self.prompt_buf[unsafe_offset=pp] = 10; pp += 1

            if context_len > 0 and pp + 9 + context_len + 2 < FLARE_PROMPT_MAX:
                var ctx_hdr = "Context:\n"
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp),
                       src=ctx_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=9); pp += 9
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp), src=self.context_buf, count=context_len)
                pp += context_len
                self.prompt_buf[unsafe_offset=pp] = 10; pp += 1

            var q_hdr = "Question: "
            if pp + 10 + query_len + 1 < FLARE_PROMPT_MAX:
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp),
                       src=q_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=10); pp += 10
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp),
                       src=query_ptr, count=query_len)
                pp += query_len
                self.prompt_buf[unsafe_offset=pp] = 10; pp += 1

            var a_hdr = "Answer: "
            if pp + 8 + generated_len < FLARE_PROMPT_MAX:
                unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp),
                       src=a_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=8); pp += 8
                if generated_len > 0:
                    unsafe_memcpy(dest=self.prompt_buf.unsafe_offset(pp), src=self.generated_buf, count=generated_len)
                    pp += generated_len

            # ── Call LLM for one chunk ───────────────────────────────────────
            self.min_lp_buf[unsafe_offset=0] = Float32(0.0)
            self.done_buf[unsafe_offset=0]   = 0

            var chunk_len = self.llm.complete_ollama(
                self.prompt_buf, pp,
                effective_chunk,
                self.chunk_buf, FLARE_CHUNK_MAX,
                self.min_lp_buf, self.done_buf)

            if chunk_len == 0:
                break   # LLM error or empty response

            # ── Check confidence ─────────────────────────────────────────────
            # min_lp_buf[0] is the most negative logprob in this chunk.
            # Uncertain if it's below the logprob threshold.
            var is_uncertain = self.min_lp_buf[unsafe_offset=0] < tau_logprob

            if is_uncertain and retries < FLARE_DEFAULT_MAX_RETRIES:
                # Build search query: generated_so_far + " " + chunk
                var sq_len = 0
                if generated_len > 0 and sq_len + generated_len < FLARE_SEARCH_MAX:
                    unsafe_memcpy(dest=self.search_buf, src=self.generated_buf, count=generated_len)
                    sq_len = generated_len
                    self.search_buf[unsafe_offset=sq_len] = 32; sq_len += 1
                if sq_len + chunk_len < FLARE_SEARCH_MAX:
                    unsafe_memcpy(dest=self.search_buf.unsafe_offset(sq_len), src=self.chunk_buf, count=chunk_len)
                    sq_len += chunk_len

                # Retrieve into context_buf (overwrites from offset 0)
                var new_facts_len = self._retrieve_into_context(
                    self.search_buf, sq_len, FLARE_DEFAULT_K)

                if new_facts_len > 0:
                    context_len = new_facts_len
                    retries += 1
                    continue   # discard uncertain chunk, loop with new context
                # No new facts — fall through and accept the chunk anyway

            # ── Accept chunk ─────────────────────────────────────────────────
            if generated_len + chunk_len < FLARE_GENERATED_MAX:
                unsafe_memcpy(dest=self.generated_buf.unsafe_offset(generated_len), src=self.chunk_buf, count=chunk_len)
                generated_len += chunk_len

            tokens_done += (chunk_len + 3) // 4   # bytes → rough token count
            retries = 0

            if self.done_buf[unsafe_offset=0] == 1:
                break   # LLM signalled done

        var write_len = generated_len if generated_len < out_max else out_max
        if write_len > 0:
            unsafe_memcpy(dest=out_buf, src=self.generated_buf, count=write_len)
        return write_len
