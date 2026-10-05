"""AI commands: AI.CHAT, AI.FLARE, AI.COMPLETE, AI.SEMANTIC_CACHE, AI.MEMORY."""
from src.common.ptr import null_ptr
from std.memory.unsafe_pointer import Pointer
from std.collections import Array, List
from std.memory import alloc, unsafe_memcpy, stack_allocation
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.network.server import TCPServer
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.list import SlabList
from src.common.utils import format_int_to_buf, parse_filter_float, arg_eq, is_valid_float_arg, parse_float64, THRESHOLD_UNSET
from src.common.config import PionConfig
from src.network.semantic_cache import SemanticCache, CACHE_MAX_ENTRIES, bytes_name
from src.network.llm_client import LLMClient
from src.network.ai_gateway import FLAREGateway
from src.network.inference_bridge import InferenceBridge, InferenceResponse, INFER_MSG_EMBED, INFER_MSG_GENERATE, INFER_MSG_LOAD_MODEL, INFER_STATUS_OK, INFER_STATUS_ERROR, INFER_RECV_BUF_SIZE
from src.vector.hnsw import HNSWGraph
from src.memory.object_pool import ObjectPool


# gh #87.3: _parse_filter_float moved to src.common.utils as parse_filter_float.


@always_inline
def handle_ai_chat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut llm_client: LLMClient, llm_out_buf: Pointer[UInt8, MutUntrackedOrigin], mut scache: SemanticCache) raises -> Int:
    """AI.CHAT <prompt> [CONTEXT <index> <text> K <k>] → LLM completion with optional RAG context."""
    var ii = i
    if ii + 1 < num_tokens:
        var _prompt_tok = tokens[unsafe_offset=ii + 1]
        ii += 1
        var _ctx_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        var _ctx_len = 0
        # Parse optional CONTEXT <index> <text> K <k>
        if ii + 3 < num_tokens:
            var _maybe_ctx = tokens[unsafe_offset=ii + 1]
            # "CONTEXT" = 7 bytes c(99),o(111),n(110),t(116),e(101),x(120),t(116)
            if _maybe_ctx.length == 7 and (_maybe_ctx.ptr[unsafe_offset=0]|0x20)==99 and (_maybe_ctx.ptr[unsafe_offset=1]|0x20)==111 and (_maybe_ctx.ptr[unsafe_offset=2]|0x20)==110:
                ii += 1  # skip CONTEXT keyword
                ii += 1
                var _ctx_idx_tok = tokens[unsafe_offset=ii]   # #29: this index's documents only
                var _ctx_text_tok = tokens[unsafe_offset=ii + 1]
                ii += 1  # consume text token
                # Parse optional K
                var _ctx_k = 3
                if ii + 1 < num_tokens and tokens[unsafe_offset=ii + 1].length == 1 and (tokens[unsafe_offset=ii + 1].ptr[unsafe_offset=0]|0x20)==107:
                    if ii + 2 < num_tokens:
                        var _kv = 0
                        for _ki in range(tokens[unsafe_offset=ii + 2].length):
                            var _kc = Int(tokens[unsafe_offset=ii + 2].ptr[unsafe_offset=_ki])
                            if _kc >= 48 and _kc <= 57: _kv = _kv * 10 + (_kc - 48)
                        if _kv > 0: _ctx_k = _kv
                        ii += 2
                # Embed context text and retrieve doc_ids for context building
                var _ctx_owner = scache.owner_id(bytes_name("t:", _ctx_idx_tok.ptr, _ctx_idx_tok.length), False)
                if scache.enabled and scache.count > 0 and _ctx_owner >= 0:
                    var _ctx_ok = scache.embed_into(
                        _ctx_text_tok.ptr,
                        _ctx_text_tok.length, scache.embed_buf)
                    if _ctx_ok:
                        var _ctx_scores = List[Float32]()
                        var _ctx_results = scache.search_owner(
                            scache.embed_buf, _ctx_owner, _ctx_k, _ctx_scores)
                        # Build context string from doc_ids (space-separated)
                        if len(_ctx_results) > 0:
                            var _ctx_sb = String("")
                            for _cri in range(len(_ctx_results)):
                                var _crid = _ctx_results[_cri]
                                if _crid >= 0 and _crid < scache.count:
                                    if _ctx_sb.byte_length() > 0: _ctx_sb += " "
                                    _ctx_sb += scache.responses[_crid]
                            # Store context in llm_out_buf temporarily (reuse for input)
                            var _clen = _ctx_sb.byte_length()
                            if _clen > 0 and _clen < 1024 * 1024:
                                unsafe_memcpy(dest=llm_out_buf, src=_ctx_sb.unsafe_ptr(), count=_clen)
                                _ctx_ptr = llm_out_buf
                                _ctx_len = _clen
        if not llm_client.enabled:
            writer.append_error_response("ERR LLM not enabled (set LLMConfig.enabled=True and start max serve)")
        else:
            var _out = alloc[UInt8](1024 * 1024)
            var _out_mb = _out
            var _out_len = llm_client.complete(
                _prompt_tok.ptr,
                _prompt_tok.length,
                _ctx_ptr, _ctx_len,
                _out_mb, 1024 * 1024 - 1)
            if _out_len > 0:
                writer.append_bulk_string_response(_out_mb, _out_len)
            else:
                writer.append_null_response()
            _out.unsafe_free()
    else:
        writer.append_error_response("ERR AI.CHAT <prompt> [CONTEXT <index> <text> [K k]]")
    return ii


@always_inline
def handle_ai_flare(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut flare: FLAREGateway) raises -> Int:
    """AI.FLARE LOAD <text> | RUN <query> [SYSTEM <s>] [TAU <f>] [CHUNKS <n>] [MAXTOKENS <n>] | INFO."""
    var ii = i
    if ii + 1 < num_tokens:
        var _fl_sub = tokens[unsafe_offset=ii + 1]
        var _fl_sp = _fl_sub.ptr; var _fl_sl = _fl_sub.length
        ii += 1
        if _fl_sl == 4 and (_fl_sp[unsafe_offset=0]|0x20)==108 and (_fl_sp[unsafe_offset=1]|0x20)==111 and (_fl_sp[unsafe_offset=2]|0x20)==97 and (_fl_sp[unsafe_offset=3]|0x20)==100:
            # LOAD <text>
            if ii + 1 < num_tokens:
                var _fl_text_tok = tokens[unsafe_offset=ii + 1]
                ii += 1
                if not flare.enabled:
                    writer.append_error_response("ERR AI.FLARE requires embedding.enabled=True and llm.enabled=True")
                elif flare.load(
                        _fl_text_tok.ptr,
                        _fl_text_tok.length):
                    writer.append_ok_response()
                else:
                    writer.append_error_response("ERR AI.FLARE LOAD: embed failed or KB full (max 5000 docs)")
            else:
                writer.append_error_response("ERR syntax: AI.FLARE LOAD <text>")
        elif _fl_sl == 3 and (_fl_sp[unsafe_offset=0]|0x20)==114 and (_fl_sp[unsafe_offset=1]|0x20)==117 and (_fl_sp[unsafe_offset=2]|0x20)==110:
            # RUN <query> [SYSTEM <s>] [TAU <f>] [CHUNKS <n>] [MAXTOKENS <n>]
            if ii + 1 < num_tokens:
                var _fl_query_tok = tokens[unsafe_offset=ii + 1]
                ii += 1
                # Parse optional keyword args
                var _fl_sys_ptr  = null_ptr[UInt8, MutUntrackedOrigin]()
                var _fl_sys_len  = 0
                var _fl_tau      = Float32(0.0)   # 0 = use gateway default
                var _fl_chunks   = 0              # 0 = use gateway default
                var _fl_maxtok   = 0              # 0 = use gateway default
                while ii + 1 < num_tokens:
                    var _kwt = tokens[unsafe_offset=ii + 1]
                    var _kwp = _kwt.ptr; var _kwl = _kwt.length
                    # SYSTEM (6 bytes)
                    if _kwl == 6 and (_kwp[unsafe_offset=0]|0x20)==115 and (_kwp[unsafe_offset=1]|0x20)==121 and (_kwp[unsafe_offset=2]|0x20)==115:
                        if ii + 2 < num_tokens:
                            var _st = tokens[unsafe_offset=ii + 2]
                            _fl_sys_ptr = _st.ptr
                            _fl_sys_len = _st.length
                            ii += 2
                        else: break
                    # TAU (3 bytes)
                    elif _kwl == 3 and (_kwp[unsafe_offset=0]|0x20)==116 and (_kwp[unsafe_offset=1]|0x20)==97 and (_kwp[unsafe_offset=2]|0x20)==117:
                        if ii + 2 < num_tokens:
                            var _tv = tokens[unsafe_offset=ii + 2]
                            _fl_tau = parse_filter_float(
                                _tv.ptr, _tv.length)
                            ii += 2
                        else: break
                    # CHUNKS (6 bytes)
                    elif _kwl == 6 and (_kwp[unsafe_offset=0]|0x20)==99 and (_kwp[unsafe_offset=1]|0x20)==104 and (_kwp[unsafe_offset=2]|0x20)==117:
                        if ii + 2 < num_tokens:
                            var _nv = tokens[unsafe_offset=ii + 2]
                            for _ni in range(_nv.length):
                                var _nc = Int(_nv.ptr[unsafe_offset=_ni])
                                if _nc >= 48 and _nc <= 57: _fl_chunks = _fl_chunks * 10 + (_nc - 48)
                            ii += 2
                        else: break
                    # MAXTOKENS (9 bytes)
                    elif _kwl == 9 and (_kwp[unsafe_offset=0]|0x20)==109 and (_kwp[unsafe_offset=1]|0x20)==97 and (_kwp[unsafe_offset=2]|0x20)==120:
                        if ii + 2 < num_tokens:
                            var _mv = tokens[unsafe_offset=ii + 2]
                            for _mi in range(_mv.length):
                                var _mc = Int(_mv.ptr[unsafe_offset=_mi])
                                if _mc >= 48 and _mc <= 57: _fl_maxtok = _fl_maxtok * 10 + (_mc - 48)
                            ii += 2
                        else: break
                    else:
                        break   # unknown keyword — stop parsing
                if not flare.enabled:
                    writer.append_error_response("ERR AI.FLARE requires embedding.enabled=True and llm.enabled=True")
                else:
                    var _fl_out = alloc[UInt8](4 * 1024 * 1024)
                    var _fl_out_mb = _fl_out
                    var _fl_out_len = flare.run(
                        _fl_query_tok.ptr,
                        _fl_query_tok.length,
                        _fl_sys_ptr, _fl_sys_len,
                        _fl_tau, _fl_chunks, _fl_maxtok,
                        _fl_out_mb, 4 * 1024 * 1024 - 1)
                    if _fl_out_len > 0:
                        writer.append_bulk_string_response(_fl_out_mb, _fl_out_len)
                    else:
                        writer.append_null_response()
                    _fl_out.unsafe_free()
            else:
                writer.append_error_response("ERR syntax: AI.FLARE RUN <query> [SYSTEM <s>] [TAU <f>] [CHUNKS <n>] [MAXTOKENS <n>]")
        elif _fl_sl == 4 and (_fl_sp[unsafe_offset=0]|0x20)==105 and (_fl_sp[unsafe_offset=1]|0x20)==110 and (_fl_sp[unsafe_offset=2]|0x20)==102 and (_fl_sp[unsafe_offset=3]|0x20)==111:
            # INFO → return stats as bulk string
            var _info = String("FLARE KB: ")
            _info += String(flare.doc_count)
            _info += " docs | tau="
            _info += String(flare.tau)
            _info += " | chunk_tokens="
            _info += String(flare.chunk_tokens)
            _info += " | max_tokens="
            _info += String(flare.max_tokens)
            _info += " | enabled="
            _info += "true" if flare.enabled else "false"
            writer.append_bulk_string_response(_info.unsafe_ptr(), _info.byte_length())
        else:
            writer.append_error_response("ERR AI.FLARE subcommand must be LOAD | RUN | INFO")
    else:
        writer.append_error_response("ERR syntax: AI.FLARE LOAD|RUN|INFO ...")
    return ii


@always_inline
def handle_ai_complete(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut scache: SemanticCache, mut llm_client: LLMClient, server: TCPServer, fd: Int32, kq: Int32) raises -> Int:
    """AI.COMPLETE <prompt> [MODEL <model>] [TOKENS <n>] [THRESHOLD <t>] → cached LLM completion."""
    var ii = i
    if ii + 1 < num_tokens:
        var _ac_prompt_tok = tokens[unsafe_offset=ii + 1]
        ii += 1
        # Parse optional MODEL / TOKENS / THRESHOLD
        var _ac_model = String("")        # empty = use embedding client's model host
        var _ac_tokens = 256
        var _ac_threshold = THRESHOLD_UNSET  # gh #373: unset = scache default; 0 is a real threshold
        var _ac_bad_thr = False
        var _ac_j = ii + 1
        while _ac_j < num_tokens - 1:
            var _ac_kw = tokens[unsafe_offset=_ac_j]
            var _ac_kwp = _ac_kw.ptr
            var _ac_kwl = _ac_kw.length
            # MODEL keyword (5 bytes: m,o,d,e,l)
            if _ac_kwl == 5 and (_ac_kwp[unsafe_offset=0]|0x20)==109 and (_ac_kwp[unsafe_offset=1]|0x20)==111 and (_ac_kwp[unsafe_offset=2]|0x20)==100:
                _ac_model = tokens[unsafe_offset=_ac_j + 1].value()
                ii += 2; _ac_j += 2
            # TOKENS keyword (6 bytes: t,o,k,e,n,s)
            elif _ac_kwl == 6 and (_ac_kwp[unsafe_offset=0]|0x20)==116 and (_ac_kwp[unsafe_offset=1]|0x20)==111 and (_ac_kwp[unsafe_offset=2]|0x20)==107:
                var _tv = tokens[unsafe_offset=_ac_j + 1].value()
                try: _ac_tokens = atol(_tv)
                except: pass
                ii += 2; _ac_j += 2
            # THRESHOLD keyword (9 bytes: t,h,r,e,s,h,o,l,d)
            elif _ac_kwl == 9 and (_ac_kwp[unsafe_offset=0]|0x20)==116 and (_ac_kwp[unsafe_offset=1]|0x20)==104 and (_ac_kwp[unsafe_offset=2]|0x20)==114:
                var _tv2 = tokens[unsafe_offset=_ac_j + 1]
                # gh #373: strict parse — the digit loop read "abc" as 0.0.
                if is_valid_float_arg(_tv2.ptr, _tv2.length):
                    _ac_threshold = Float32(parse_float64(_tv2.ptr, _tv2.length))
                else:
                    _ac_bad_thr = True
                ii += 2; _ac_j += 2
            else:
                break
        # 1. Check semantic cache
        if _ac_bad_thr:
            writer.append_error_response("ERR THRESHOLD value is not a valid float")
        elif scache.enabled:
            var _ac_hit = scache.cache_get(
                _ac_prompt_tok.ptr, _ac_prompt_tok.length,
                _ac_threshold, writer, server, fd, kq)
            if _ac_hit:
                pass  # response already written by cache_get
            else:
                # 2. Cache miss → call Ollama /api/generate
                var _ac_host = scache.client.host
                var _ac_port = scache.client.port
                var _ac_use_model = _ac_model if _ac_model.byte_length() > 0 else llm_client.model
                var _ac_out = alloc[UInt8](1024 * 1024)
                var _ac_out_mb = _ac_out
                var _ac_logprob = stack_allocation[1, Float32]()
                var _ac_done = stack_allocation[1, UInt8]()
                var _ac_tmp_client = LLMClient(
                    _ac_host, _ac_port, _ac_use_model, True)
                var _ac_len = _ac_tmp_client.complete_ollama(
                    _ac_prompt_tok.ptr,
                    _ac_prompt_tok.length,
                    _ac_tokens,
                    _ac_out_mb, 1024 * 1024 - 1,
                    _ac_logprob,
                    _ac_done)
                if _ac_len > 0:
                    # 3. Store in semantic cache (best-effort; the generation
                    # already succeeded, so a failed store must not fail AI.COMPLETE)
                    _ = scache.cache_set(
                        _ac_prompt_tok.ptr, _ac_prompt_tok.length,
                        _ac_out_mb, _ac_len)
                    writer.append_bulk_string_response(_ac_out_mb, _ac_len)
                else:
                    writer.append_error_response("ERR AI.COMPLETE: Ollama generation failed")
                _ac_out.unsafe_free()
        else:
            writer.append_error_response("ERR AI.COMPLETE requires --emb-enabled (Ollama embedding server)")
    else:
        writer.append_error_response("ERR syntax: AI.COMPLETE <prompt> [MODEL <model>] [TOKENS <n>] [THRESHOLD <t>]")
    return ii


@always_inline
def handle_ai_semantic_cache(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut scache: SemanticCache, server: TCPServer, fd: Int32, kq: Int32) raises -> Int:
    """AI.SEMANTIC_CACHE GET|SET <query> ... → semantic cache operations."""
    var ii = i
    if ii + 2 < num_tokens:
        var sub = tokens[unsafe_offset=ii + 1]
        var sub_p = sub.ptr; var sub_l = sub.length
        var query_tok = tokens[unsafe_offset=ii + 2]
        var query_ptr2 = query_tok.ptr
        var query_len2 = query_tok.length
        # GET subcommand (3 bytes: g,e,t)
        if sub_l == 3 and (sub_p[unsafe_offset=0]|0x20)==103 and (sub_p[unsafe_offset=1]|0x20)==101 and (sub_p[unsafe_offset=2]|0x20)==116:
            # Parse optional THRESHOLD argument
            var thr_override = THRESHOLD_UNSET   # gh #373: unset = default; 0 means match anything
            var thr_bad = False
            if ii + 4 < num_tokens:
                var maybe_thr = tokens[unsafe_offset=ii + 3]
                if arg_eq(maybe_thr.ptr, maybe_thr.length, "threshold"):
                    var tv = tokens[unsafe_offset=ii + 4]
                    # gh #373: strict — the digit loop read "abc" as 0.0,
                    # which then also fell back to the default.
                    if is_valid_float_arg(tv.ptr, tv.length):
                        thr_override = Float32(parse_float64(tv.ptr, tv.length))
                    else:
                        thr_bad = True
                    ii += 2   # consume THRESHOLD + value tokens
            # gh #115: optional WITHWORKSPACE (13 bytes) — opt-in, so the
            # default bulk-string reply is unchanged for existing clients.
            var with_ws = False
            var _wsi = ii + 3
            while _wsi < num_tokens:
                var _wt = tokens[unsafe_offset=_wsi]
                # arg_eq, not a hand-rolled byte test: my first attempt here
                # checked byte 4 for 'o' when WITHWORKSPACE has 'W' there
                # (W-I-T-H-W-O), so the flag silently never matched. Exactly
                # the gh #225/#251 failure — match the WHOLE keyword.
                if arg_eq(_wt.ptr, _wt.length, "withworkspace"):
                    with_ws = True
                    ii += 1
                    break
                _wsi += 1
            ii += 2   # consume GET + query tokens
            if thr_bad:
                writer.append_error_response("ERR THRESHOLD value is not a valid float")
            else:
                var got_hit = scache.cache_get(
                    query_ptr2, query_len2, thr_override, writer, server, fd, kq, with_ws)
                if not got_hit: writer.append_null_response()
        # SET subcommand (3 bytes: s,e,t)
        elif sub_l == 3 and (sub_p[unsafe_offset=0]|0x20)==115 and (sub_p[unsafe_offset=1]|0x20)==101 and (sub_p[unsafe_offset=2]|0x20)==116:
            if ii + 3 < num_tokens:
                var resp_tok = tokens[unsafe_offset=ii + 3]
                # gh #115: optional `WORKSPACE <blob>` — the caller's J-lens
                # top-k snapshot at generation time. Opaque bytes to the
                # server: it stores and returns them, and never interprets
                # them, so the lens format can change without a wire change.
                var ws_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
                var ws_len = 0
                var _consumed = 3
                if ii + 5 < num_tokens:
                    var _kw = tokens[unsafe_offset=ii + 4]
                    if arg_eq(_kw.ptr, _kw.length, "workspace"):
                        var _wv = tokens[unsafe_offset=ii + 5]
                        ws_ptr = _wv.ptr; ws_len = _wv.length
                        _consumed = 5
                var _set_st = scache.cache_set(
                    query_ptr2, query_len2,
                    resp_tok.ptr, resp_tok.length, ws_ptr, ws_len)
                ii += _consumed
                # gh #421: reply +OK only when the response was actually stored.
                # It used to answer +OK unconditionally, so on a fresh install
                # with no embedding backend, SET said OK and GET always missed —
                # "a command that reports success without applying".
                if _set_st == 0:
                    writer.append_ok_response()
                elif _set_st == 2:
                    writer.append_error_response("ERR semantic cache is full")
                else:
                    writer.append_error_response("ERR embedding backend unavailable (start with --nle-embed on macOS, --emb-enabled for an Ollama/OpenAI endpoint, or run pixi run install-inference for the in-process sidecar)")
            else:
                ii += 2
                writer.append_error_response("ERR syntax: AI.SEMANTIC_CACHE SET <query> <response> [WORKSPACE <blob>]")
        # gh #115: EXPLAIN <query> — the workspace recorded when this answer was
        # cached, with no inference infrastructure required. That is the point:
        # a compliance reviewer can ask what the model had in its workspace
        # without being able to run the model.
        elif arg_eq(sub_p, sub_l, "explain"):
            ii += 2
            var _hit = scache.cache_lookup(query_ptr2, query_len2, THRESHOLD_UNSET)
            if _hit < 0:
                writer.append_null_response()
            else:
                var _ws = scache.workspace_at(_hit)
                if _ws.byte_length() > 0:
                    writer.append_bulk_string_response(_ws.unsafe_ptr(), _ws.byte_length())
                else:
                    writer.append_null_response()
        else:
            ii += 2
            writer.append_error_response("ERR AI.SEMANTIC_CACHE subcommand must be GET, SET or EXPLAIN")
    else:
        writer.append_error_response("ERR syntax: AI.SEMANTIC_CACHE GET|SET <query> ...")
    return ii


@always_inline
def handle_ai_embed(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut inference_bridge: InferenceBridge, mut scache: SemanticCache) raises -> Int:
    """AI.EMBED <text> → raw FP32 vector as RESP bulk string (dim x 4 bytes).

    Goes through `scache.embed_into` so the full cascade fires:
    NLEmbedding (Mac, --nle-embed) → InferenceBridge (PyTorch sidecar)
    → HTTP EmbeddingClient (Ollama/OpenAI/MAX). Whichever backend is
    enabled at server start handles the request transparently."""
    var ii = i
    if scache.enabled:
        if ii + 1 < num_tokens:
            var text_tok = tokens[unsafe_offset=ii + 1]
            ii += 1
            var dims = scache.dimensions
            var emb_buf = alloc[Float32](dims)
            var ok = scache.embed_into(
                text_tok.ptr.unsafe_bitcast[UInt8](), text_tok.length, emb_buf)
            if ok:
                writer.append_bulk_string_response(emb_buf.unsafe_bitcast[UInt8](), dims * 4)
            else:
                writer.append_error_response("ERR AI.EMBED: embedding failed")
            emb_buf.unsafe_free()
        else:
            writer.append_error_response("ERR syntax: AI.EMBED <text>")
    else:
        # gh #373: name the real condition. The old text told a server started with
        # --inference to pass --inference.
        writer.append_error_response("ERR AI.EMBED: no embedding backend is enabled (start with --nle-embed on macOS, --emb-enabled for an Ollama/OpenAI endpoint, or --inference with its default model; --no-auto-embed and --profile kv disable it)")
    return ii


@always_inline
def handle_ai_generate(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut inference_bridge: InferenceBridge) raises -> Int:
    """AI.GENERATE <prompt> [KEYS k1 k2 ...] [MAX_TOKENS n] → LLM generation via inference sidecar."""
    var ii = i
    if not inference_bridge.connected:
        writer.append_error_response("ERR AI.GENERATE requires --inference")
    elif ii + 1 < num_tokens:
        var prompt_tok = tokens[unsafe_offset=ii + 1]
        ii += 1
        # Parse optional KEYS, MAX_TOKENS
        var context_buf = alloc[UInt8](65536)
        var ctx_len = 0
        var max_tokens = 256
        while ii + 1 < num_tokens:
            var opt = tokens[unsafe_offset=ii + 1]
            var op = opt.ptr; var ol = opt.length
            # KEYS
            if ol == 4 and (op[unsafe_offset=0]|0x20)==107 and (op[unsafe_offset=1]|0x20)==101 and (op[unsafe_offset=2]|0x20)==121 and (op[unsafe_offset=3]|0x20)==115:
                ii += 1
                # Read subsequent tokens as key names until we hit another keyword or end
                while ii + 1 < num_tokens:
                    var next_t = tokens[unsafe_offset=ii + 1]
                    var np2 = next_t.ptr; var nl2 = next_t.length
                    # Check if next token is MAX_TOKENS keyword
                    if nl2 == 10 and (np2[unsafe_offset=0]|0x20)==109 and (np2[unsafe_offset=1]|0x20)==97 and (np2[unsafe_offset=2]|0x20)==120:
                        break
                    ii += 1
                    # Look up key in keyspace
                    var key = GenericValue.from_ptr(next_t.ptr.unsafe_bitcast[UInt8](), next_t.length)
                    var hash = key.__hash__()
                    var val = keyspace[].get_with_hash(key, UInt64(hash))
                    if not val.is_none():
                        # Append "key: value\n" to context
                        if ctx_len + next_t.length + 3 < 65536:
                            unsafe_memcpy(dest=context_buf.unsafe_offset(ctx_len), src=next_t.ptr.unsafe_bitcast[UInt8](), count=next_t.length)
                            ctx_len += next_t.length
                            context_buf[unsafe_offset=ctx_len] = 58; ctx_len += 1  # ':'
                            context_buf[unsafe_offset=ctx_len] = 32; ctx_len += 1  # ' '
                        # Extract value as string
                        if val.is_string():
                            var vlen = val.string_len()
                            if ctx_len + vlen + 1 < 65536:
                                val.copy_to(context_buf.unsafe_offset(ctx_len))
                                ctx_len += vlen
                        elif val.type.value == ValueType.INT:
                            var int_buf = alloc[UInt8](24)
                            var il = format_int_to_buf(int_buf, 0, Int64(val._data0))
                            if ctx_len + il + 1 < 65536:
                                unsafe_memcpy(dest=context_buf.unsafe_offset(ctx_len), src=int_buf, count=il)
                                ctx_len += il
                            int_buf.unsafe_free()
                        context_buf[unsafe_offset=ctx_len] = 10; ctx_len += 1  # '\n'
            # MAX_TOKENS
            elif ol == 10 and (op[unsafe_offset=0]|0x20)==109 and (op[unsafe_offset=1]|0x20)==97 and (op[unsafe_offset=2]|0x20)==120:
                ii += 1
                if ii + 1 < num_tokens:
                    var mt_tok = tokens[unsafe_offset=ii + 1]
                    ii += 1
                    # Parse integer
                    var mt_val = 0
                    for mi in range(mt_tok.length):
                        var c = mt_tok.ptr[unsafe_offset=mi]
                        if c >= 48 and c <= 57:
                            mt_val = mt_val * 10 + Int(c - 48)
                    if mt_val > 0:
                        max_tokens = mt_val
            else:
                break
        # Send to inference sidecar
        var req_id = inference_bridge.send_generate_request(
            prompt_tok.ptr.unsafe_bitcast[UInt8](), prompt_tok.length,
            context_buf, ctx_len, max_tokens)
        if req_id > 0:
            # Blocking receive (single-worker mode, acceptable latency)
            var resp_body = alloc[UInt8](INFER_RECV_BUF_SIZE)
            var resp = inference_bridge.recv_response_blocking(
                resp_body, INFER_RECV_BUF_SIZE)
            if resp.status == INFER_STATUS_OK and resp.body_len >= 4:
                # Body: [text_len:u32][text_bytes]
                var text_len2 = Int(resp_body[unsafe_offset=0]) | (Int(resp_body[unsafe_offset=1]) << 8) | (Int(resp_body[unsafe_offset=2]) << 16) | (Int(resp_body[unsafe_offset=3]) << 24)
                if text_len2 > 0 and text_len2 + 4 <= resp.body_len:
                    writer.append_bulk_string_response(resp_body.unsafe_offset(4), text_len2)
                else:
                    writer.append_null_response()
            else:
                writer.append_error_response("ERR AI.GENERATE: inference failed")
            resp_body.unsafe_free()
        else:
            writer.append_error_response("ERR AI.GENERATE: send failed")
        context_buf.unsafe_free()
    else:
        writer.append_error_response("ERR syntax: AI.GENERATE <prompt> [KEYS k1 ...] [MAX_TOKENS n]")
    return ii


@always_inline
def handle_ai_loadmodel(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, mut inference_bridge: InferenceBridge) raises -> Int:
    """AI.LOADMODEL <model_id> [EMBEDDING|LLM] → load model in inference sidecar."""
    var ii = i
    if not inference_bridge.connected:
        writer.append_error_response("ERR AI.LOADMODEL requires --inference")
    elif ii + 1 < num_tokens:
        var model_tok = tokens[unsafe_offset=ii + 1]
        ii += 1
        var model_type = UInt8(0)  # default: embedding
        if ii + 1 < num_tokens:
            var type_tok = tokens[unsafe_offset=ii + 1]
            var ttp = type_tok.ptr; var ttl2 = type_tok.length
            # LLM (3 bytes)
            if ttl2 == 3 and (ttp[unsafe_offset=0]|0x20)==108 and (ttp[unsafe_offset=1]|0x20)==108 and (ttp[unsafe_offset=2]|0x20)==109:
                model_type = UInt8(1)
                ii += 1
            # EMBEDDING (9 bytes)
            elif ttl2 == 9 and (ttp[unsafe_offset=0]|0x20)==101:
                model_type = UInt8(0)
                ii += 1
        var req_id = inference_bridge.send_load_model_request(
            model_tok.ptr.unsafe_bitcast[UInt8](), model_tok.length, model_type)
        if req_id > 0:
            var resp_body = alloc[UInt8](4096)
            var resp = inference_bridge.recv_response_blocking(resp_body, 4096)
            if resp.status == INFER_STATUS_OK:
                writer.append_ok_response()
            else:
                # Extract error message
                writer.append_error_response("ERR AI.LOADMODEL: load failed")
            resp_body.unsafe_free()
        else:
            writer.append_error_response("ERR AI.LOADMODEL: send failed")
    else:
        writer.append_error_response("ERR syntax: AI.LOADMODEL <model_id> [EMBEDDING|LLM]")
    return ii


@always_inline
def handle_ai_memory(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut scache: SemanticCache, list_pool: Pointer[ObjectPool[SlabList], MutUntrackedOrigin], server: TCPServer, fd: Int32, kq: Int32) raises -> Int:
    """AI.MEMORY ADD|RECALL|CONTEXT → dual-write LIST + HNSW memory management."""
    var ii = i
    if ii + 2 < num_tokens:
        var sub = tokens[unsafe_offset=ii + 1]
        var sub_p = sub.ptr; var sub_l = sub.length
        # ADD subcommand (3 bytes: a,d,d)
        if sub_l == 3 and (sub_p[unsafe_offset=0]|0x20)==97 and (sub_p[unsafe_offset=1]|0x20)==100 and (sub_p[unsafe_offset=2]|0x20)==100:
            # AI.MEMORY ADD <session_id> <role> <content>
            if ii + 4 < num_tokens:
                var sess_tok = tokens[unsafe_offset=ii + 2]
                var role_tok = tokens[unsafe_offset=ii + 3]
                var content_tok = tokens[unsafe_offset=ii + 4]
                var sess_str = sess_tok.value()
                var role_str = role_tok.value()
                var content_str = content_tok.value()
                ii += 4
                # Dual-write: 1) LPUSH to session list (short-term)
                var list_key = "mem:list:" + sess_str
                var entry = role_str + ": " + content_str
                var list_key_v = GenericValue.from_string(list_key)
                var val = keyspace[].get(list_key_v)
                if val.is_none():
                    var list_ptr = list_pool[].acquire(); list_ptr[].reset()
                    var new_val = GenericValue()
                    new_val.type = ValueType(ValueType.LIST)
                    new_val.set_ptr(list_ptr.unsafe_bitcast[NoneType]())
                    var entry_v = GenericValue.from_string(entry)
                    list_ptr[].lpush(entry_v)
                    keyspace[].set(list_key_v, new_val)
                elif val.type.value == ValueType.LIST:
                    var list_ptr = val.as_list().unsafe_bitcast[SlabList]()
                    var entry_v = GenericValue.from_string(entry)
                    list_ptr[].lpush(entry_v)
                # 2) If embeddings enabled, add to semantic index (long-term),
                # as this session's entry (#29).
                if scache.enabled and scache.count < CACHE_MAX_ENTRIES:
                    var ok = scache.embed_into(
                        content_tok.ptr.unsafe_bitcast[UInt8](), content_tok.length,
                        scache.embed_buf)
                    if ok:
                        var _mem_owner = scache.owner_id(
                            bytes_name("m:", sess_tok.ptr, sess_tok.length), True)
                        try:
                            _ = scache.add_entry(_mem_owner, entry)
                        except:
                            pass
                writer.append_ok_response()
            else:
                ii += 1
                writer.append_error_response("ERR syntax: AI.MEMORY ADD <session_id> <role> <content>")
        # RECALL subcommand (6 bytes: r,e,c,a,l,l)
        elif sub_l == 6 and (sub_p[unsafe_offset=0]|0x20)==114 and (sub_p[unsafe_offset=1]|0x20)==101 and (sub_p[unsafe_offset=2]|0x20)==99 and (sub_p[unsafe_offset=3]|0x20)==97 and (sub_p[unsafe_offset=4]|0x20)==108 and (sub_p[unsafe_offset=5]|0x20)==108:
            # AI.MEMORY RECALL <session_id> <query> [K k]
            if ii + 3 < num_tokens:
                var rsess_tok = tokens[unsafe_offset=ii + 2]
                var query_tok = tokens[unsafe_offset=ii + 3]
                ii += 3
                # #29: this session's memories only. It used to search every
                # entry in the store, so it could answer with another session's
                # memory, a cached response or a document id.
                var _rc_owner = scache.owner_id(
                    bytes_name("m:", rsess_tok.ptr, rsess_tok.length), False)
                if not scache.enabled:
                    writer.append_error_response("ERR AI.MEMORY RECALL requires --emb-enabled")
                elif scache.count == 0 or _rc_owner < 0:
                    writer.append_empty_array_response()
                else:
                    var got_hit = scache.cache_get(
                        query_tok.ptr, query_tok.length,
                        THRESHOLD_UNSET, writer, server, fd, kq, owner=_rc_owner)
                    if not got_hit:
                        writer.append_empty_array_response()
            else:
                ii += 1
                writer.append_error_response("ERR syntax: AI.MEMORY RECALL <session_id> <query> [K k]")
        # CONTEXT subcommand (7 bytes: c,o,n,t,e,x,t)
        elif sub_l == 7 and (sub_p[unsafe_offset=0]|0x20)==99 and (sub_p[unsafe_offset=1]|0x20)==111 and (sub_p[unsafe_offset=2]|0x20)==110 and (sub_p[unsafe_offset=3]|0x20)==116 and (sub_p[unsafe_offset=4]|0x20)==101 and (sub_p[unsafe_offset=5]|0x20)==120 and (sub_p[unsafe_offset=6]|0x20)==116:
            # AI.MEMORY CONTEXT <session_id> <n> → LRANGE last N turns
            if ii + 3 < num_tokens:
                var sess_tok = tokens[unsafe_offset=ii + 2]
                var n_tok = tokens[unsafe_offset=ii + 3]
                var sess_str = sess_tok.value()
                var n_str = n_tok.value()
                ii += 3
                var n_val: Int = 0
                var np2 = n_str.unsafe_ptr().unsafe_bitcast[UInt8]()
                for ni in range(n_str.byte_length()):
                    if np2[unsafe_offset=ni] >= 48 and np2[unsafe_offset=ni] <= 57:
                        n_val = n_val * 10 + Int(np2[unsafe_offset=ni] - 48)
                var list_key = "mem:list:" + sess_str
                var list_key_v = GenericValue.from_string(list_key)
                var ctx_val = keyspace[].get(list_key_v)
                if ctx_val.is_none():
                    writer.append_empty_array_response()
                else:
                    var list_ptr = ctx_val.as_list().unsafe_bitcast[SlabList]()
                    var count = list_ptr[].llen()
                    var end_idx = min(n_val, count)
                    # Build RESP array: *N\r\n followed by N bulk strings
                    writer.buffer[unsafe_offset=writer.offset] = 42  # '*'
                    writer.offset += 1
                    writer.offset = format_int_to_buf(writer.buffer, writer.offset, Int64(end_idx))
                    writer.buffer[unsafe_offset=writer.offset] = 13; writer.buffer[unsafe_offset=writer.offset + 1] = 10
                    writer.offset += 2
                    var lrange_items = list_ptr[].lrange(0, end_idx - 1)
                    for li in range(len(lrange_items)):
                        writer.append_bulk_value_response(lrange_items[li])
            else:
                ii += 1
                writer.append_error_response("ERR syntax: AI.MEMORY CONTEXT <session_id> <n>")
        else:
            ii += 1
            writer.append_error_response("ERR AI.MEMORY subcommand must be ADD | RECALL | CONTEXT")
    else:
        writer.append_error_response("ERR syntax: AI.MEMORY ADD|RECALL|CONTEXT ...")
    return ii
