"""LLMClient — blocking HTTP/1.0 POST to an OpenAI-compatible /v1/chat/completions endpoint.

Protocol:
  POST /v1/chat/completions HTTP/1.0
  Content-Type: application/json
  {"model":"MODEL","messages":[{"role":"user","content":"<context>\\n<prompt>"}]}

Response: JSON with choices[0].message.content text extracted as a plain string.

Thread-safety: each worker creates its own LLMClient — no sharing required.
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.ffi import external_call


@always_inline
def _parse_float_lp(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float32:
    """Parse ASCII float for logprob values (negative floats like -1.204)."""
    var i = 0
    var sign: Float64 = 1.0
    if i < plen and p[unsafe_offset=i] == 45: sign = -1.0; i += 1
    elif i < plen and p[unsafe_offset=i] == 43: i += 1
    var integer: Float64 = 0.0
    while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
        integer = integer * 10.0 + Float64(p[unsafe_offset=i] - 48); i += 1
    var frac: Float64 = 0.0; var frac_mul: Float64 = 0.1
    if i < plen and p[unsafe_offset=i] == 46:
        i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            frac = frac + frac_mul * Float64(p[unsafe_offset=i] - 48); frac_mul *= 0.1; i += 1
    var exp_val: Int = 0; var exp_sign: Int = 1
    if i < plen and (p[unsafe_offset=i] == 101 or p[unsafe_offset=i] == 69):
        i += 1
        if i < plen and p[unsafe_offset=i] == 45: exp_sign = -1; i += 1
        elif i < plen and p[unsafe_offset=i] == 43: i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            exp_val = exp_val * 10 + Int(p[unsafe_offset=i] - 48); i += 1
        exp_val *= exp_sign
    var result = Float32(sign * (integer + frac))
    if exp_val > 0:
        var scale: Float64 = 1.0
        for _ in range(exp_val): scale *= 10.0
        result = Float32(Float64(result) * scale)
    elif exp_val < 0:
        var scale: Float64 = 1.0
        for _ in range(-exp_val): scale /= 10.0
        result = Float32(Float64(result) * scale)
    return result


@always_inline
def _llm_escape_json(src: Pointer[UInt8, MutUntrackedOrigin], slen: Int,
                    dst: Pointer[UInt8, MutUntrackedOrigin], mut dp: Int):
    """JSON-escape src[0..slen-1] into dst starting at dp; advances dp."""
    for ci in range(slen):
        var b = src[unsafe_offset=ci]
        if b == 34:       # '"' → \"
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 34; dp += 1
        elif b == 92:     # '\' → \\
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 92; dp += 1
        elif b == 10:     # '\n' → \n
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 110; dp += 1
        elif b == 13:     # '\r' → \r
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 114; dp += 1
        else:
            dst[unsafe_offset=dp] = b; dp += 1


struct LLMClient(Movable):
    """Blocking HTTP/1.0 client for OpenAI-compatible /v1/chat/completions."""
    var host: String
    var port: Int
    var model: String
    var enabled: Bool

    def __init__(out self, host: String, port: Int, model: String, enabled: Bool = False):
        self.host = host
        self.port = port
        self.model = model
        self.enabled = enabled

    def __moveinit__(out self, deinit take: Self):
        self.host = take.host^
        self.port = take.port
        self.model = take.model^
        self.enabled = take.enabled

    def complete(self,
                prompt_ptr: Pointer[UInt8, MutUntrackedOrigin], prompt_len: Int,
                context_ptr: Pointer[UInt8, MutUntrackedOrigin], context_len: Int,
                out_buf: Pointer[UInt8, MutUntrackedOrigin], out_max: Int) -> Int:
        """POST prompt (with optional context) to chat completions; fills out_buf.
        Returns bytes written to out_buf, or 0 on failure."""
        if not self.enabled: return 0
        var chost = self.host
        var fd = external_call["pion_connect_tcp", Int32](chost.as_c_string_slice(), Int32(self.port))
        if fd < 0: return 0

        # ── Build JSON-escaped content string ────────────────────────────────
        var content_max = context_len * 2 + prompt_len * 2 + 64
        var content_buf = alloc[UInt8](content_max)
        var content_mb = content_buf
        var cp = 0
        if context_len > 0:
            var ctx_hdr = "Context:\\n"
            unsafe_memcpy(dest=content_mb.unsafe_offset(cp), src=ctx_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=ctx_hdr.byte_length()); cp += ctx_hdr.byte_length()
            _llm_escape_json(context_ptr, context_len, content_mb, cp)
            content_mb[unsafe_offset=cp] = 92; cp += 1; content_mb[unsafe_offset=cp] = 110; cp += 1  # \n
            content_mb[unsafe_offset=cp] = 92; cp += 1; content_mb[unsafe_offset=cp] = 110; cp += 1  # \n
            var usr_hdr = "User: "
            unsafe_memcpy(dest=content_mb.unsafe_offset(cp), src=usr_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=usr_hdr.byte_length()); cp += usr_hdr.byte_length()
        _llm_escape_json(prompt_ptr, prompt_len, content_mb, cp)
        var content_len = cp

        # ── Build JSON body ───────────────────────────────────────────────────
        var cmodel = self.model
        var model_len = cmodel.byte_length()
        var prefix1 = "{\"model\":\""
        var mid1    = "\",\"messages\":[{\"role\":\"user\",\"content\":\""
        var suffix1 = "\"}],\"stream\":false}"
        var p1l = prefix1.byte_length(); var m1l = mid1.byte_length(); var s1l = suffix1.byte_length()
        var max_body = p1l + model_len + m1l + content_len + s1l + 4
        var body_buf = alloc[UInt8](max_body)
        var body_mb = body_buf
        var bp = 0
        unsafe_memcpy(dest=body_mb, src=prefix1.unsafe_ptr().unsafe_bitcast[UInt8](), count=p1l); bp += p1l
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=cmodel.unsafe_ptr().unsafe_bitcast[UInt8](), count=model_len); bp += model_len
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=mid1.unsafe_ptr().unsafe_bitcast[UInt8](), count=m1l); bp += m1l
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=content_mb, count=content_len); bp += content_len
        content_buf.unsafe_free()
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=suffix1.unsafe_ptr().unsafe_bitcast[UInt8](), count=s1l); bp += s1l
        var body_len = bp

        # ── Build HTTP request ────────────────────────────────────────────────
        var body_len_str = String(body_len)
        var chost2 = self.host
        var port_str = String(self.port)
        var req_hdr = String("POST /v1/chat/completions HTTP/1.0\r\nHost: ")
        req_hdr += chost2; req_hdr += ":"; req_hdr += port_str
        req_hdr += "\r\nContent-Type: application/json\r\nContent-Length: "
        req_hdr += body_len_str; req_hdr += "\r\nConnection: close\r\n\r\n"
        var hdr_len = req_hdr.byte_length()
        var req_buf = alloc[UInt8](hdr_len + body_len + 1)
        var req_mb = req_buf
        unsafe_memcpy(dest=req_mb, src=req_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=hdr_len)
        unsafe_memcpy(dest=req_mb.unsafe_offset(hdr_len), src=body_mb, count=body_len)
        body_buf.unsafe_free()

        # ── Send ─────────────────────────────────────────────────────────────
        var total_len = hdr_len + body_len
        var sent = external_call["pion_write", Int64](fd, req_buf, total_len)
        req_buf.unsafe_free()
        if sent < Int64(total_len):
            _ = external_call["close", Int32](fd); return 0

        # ── Receive (read until connection close) ─────────────────────────────
        var buf_size = 1024 * 1024  # 1MB
        var buf = alloc[UInt8](buf_size)
        var buf_mb = buf
        var total_read = 0
        var nr: Int64 = 1
        while nr > 0 and total_read < buf_size - 1:
            nr = external_call["pion_read", Int64](fd, buf_mb.unsafe_offset(total_read), buf_size - total_read - 1)
            if nr > 0: total_read += Int(nr)
        _ = external_call["close", Int32](fd)

        # ── Parse JSON: find "content":" ──────────────────────────────────────
        # Searches for the first occurrence of "content":" in the response body
        comptime TAG = "\"content\":\""
        comptime TAG_LEN = 12
        var tag_ptr = TAG.unsafe_ptr()
        var found = -1
        for si in range(total_read - TAG_LEN):
            var ok = True
            for ti in range(TAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != tag_ptr[unsafe_offset=ti]: ok = False; break
            if ok: found = si + TAG_LEN; break

        if found < 0:
            buf.unsafe_free(); return 0

        # ── Extract until closing '"' (handle backslash escapes) ──────────────
        var op = found; var dp2 = 0
        while op < total_read and dp2 < out_max - 1:
            var b2 = buf_mb[unsafe_offset=op]; op += 1
            if b2 == 92 and op < total_read:  # backslash escape
                var esc = buf_mb[unsafe_offset=op]; op += 1
                if esc == 110:   out_buf[unsafe_offset=dp2] = 10; dp2 += 1  # \n
                elif esc == 116: out_buf[unsafe_offset=dp2] = 9;  dp2 += 1  # \t
                elif esc == 114: out_buf[unsafe_offset=dp2] = 13; dp2 += 1  # \r
                elif esc == 34:  out_buf[unsafe_offset=dp2] = 34; dp2 += 1  # \"
                elif esc == 92:  out_buf[unsafe_offset=dp2] = 92; dp2 += 1  # \\
                else: out_buf[unsafe_offset=dp2] = esc; dp2 += 1
            elif b2 == 34:  # closing '"'
                break
            else:
                out_buf[unsafe_offset=dp2] = b2; dp2 += 1
        buf.unsafe_free()
        return dp2

    def complete_ollama(self,
                       prompt_ptr: Pointer[UInt8, MutUntrackedOrigin], prompt_len: Int,
                       num_predict: Int,
                       out_text: Pointer[UInt8, MutUntrackedOrigin], out_text_max: Int,
                       out_min_logprob: Pointer[Float32, MutUntrackedOrigin],
                       out_done: Pointer[UInt8, MutUntrackedOrigin]) -> Int:
        """POST to Ollama /api/generate with logprobs.

        Fills out_text[0..returned_len-1], sets out_min_logprob, out_done (0/1).
        Returns text byte count, or 0 on failure.

        Request JSON:
          {"model":"...","prompt":"...","stream":false,"logprobs":true,
           "options":{"num_predict":N,"temperature":0}}

        Response JSON (Ollama ≥0.5):
          {"response":"...","logprobs":[-0.5,-1.2,...],"done":false,...}
        """
        if not self.enabled: return 0
        out_min_logprob[unsafe_offset=0] = Float32(0.0)   # 0.0 = no uncertainty (exp(0)=1.0)
        out_done[unsafe_offset=0] = 0

        var chost = self.host
        var fd = external_call["pion_connect_tcp", Int32](chost.as_c_string_slice(), Int32(self.port))
        if fd < 0: return 0

        # ── Build JSON body ───────────────────────────────────────────────────
        var cmodel = self.model
        var model_len = cmodel.byte_length()
        var num_predict_str = String(num_predict)
        var np_len = num_predict_str.byte_length()

        # prefix1: {"model":"
        # model
        # mid1: ","prompt":"
        # escaped prompt
        # suffix1: ","stream":false,"logprobs":true,"options":{"num_predict":
        # num_predict digits
        # suffix2: ,"temperature":0}}
        var prefix1 = "{\"model\":\""
        var mid1    = "\",\"prompt\":\""
        var suffix1 = "\",\"stream\":false,\"logprobs\":true,\"options\":{\"num_predict\":"
        var suffix2 = ",\"temperature\":0}}"
        var p1l = prefix1.byte_length(); var m1l = mid1.byte_length()
        var s1l = suffix1.byte_length(); var s2l = suffix2.byte_length()

        var max_body = p1l + model_len + m1l + prompt_len * 2 + s1l + np_len + s2l + 4
        var body_buf = alloc[UInt8](max_body)
        var body_mb = body_buf
        var bp = 0
        unsafe_memcpy(dest=body_mb, src=prefix1.unsafe_ptr().unsafe_bitcast[UInt8](), count=p1l); bp += p1l
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=cmodel.unsafe_ptr().unsafe_bitcast[UInt8](), count=model_len); bp += model_len
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=mid1.unsafe_ptr().unsafe_bitcast[UInt8](), count=m1l); bp += m1l
        _llm_escape_json(prompt_ptr, prompt_len, body_mb, bp)
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=suffix1.unsafe_ptr().unsafe_bitcast[UInt8](), count=s1l); bp += s1l
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=num_predict_str.unsafe_ptr().unsafe_bitcast[UInt8](), count=np_len); bp += np_len
        unsafe_memcpy(dest=body_mb.unsafe_offset(bp), src=suffix2.unsafe_ptr().unsafe_bitcast[UInt8](), count=s2l); bp += s2l
        var body_len = bp

        # ── Build HTTP request ────────────────────────────────────────────────
        var body_len_str = String(body_len)
        var chost2 = self.host
        var port_str = String(self.port)
        var req_hdr = String("POST /api/generate HTTP/1.0\r\nHost: ")
        req_hdr += chost2; req_hdr += ":"; req_hdr += port_str
        req_hdr += "\r\nContent-Type: application/json\r\nContent-Length: "
        req_hdr += body_len_str; req_hdr += "\r\nConnection: close\r\n\r\n"
        var hdr_len = req_hdr.byte_length()
        var req_buf = alloc[UInt8](hdr_len + body_len + 1)
        var req_mb = req_buf
        unsafe_memcpy(dest=req_mb, src=req_hdr.unsafe_ptr().unsafe_bitcast[UInt8](), count=hdr_len)
        unsafe_memcpy(dest=req_mb.unsafe_offset(hdr_len), src=body_mb, count=body_len)
        body_buf.unsafe_free()

        var total_len = hdr_len + body_len
        var sent = external_call["pion_write", Int64](fd, req_buf, total_len)
        req_buf.unsafe_free()
        if sent < Int64(total_len):
            _ = external_call["close", Int32](fd); return 0

        # ── Receive ───────────────────────────────────────────────────────────
        var buf_size = 1024 * 1024   # 1MB
        var buf = alloc[UInt8](buf_size)
        var buf_mb = buf
        var total_read = 0
        var nr: Int64 = 1
        while nr > 0 and total_read < buf_size - 1:
            nr = external_call["pion_read", Int64](fd, buf_mb.unsafe_offset(total_read), buf_size - total_read - 1)
            if nr > 0: total_read += Int(nr)
        _ = external_call["close", Int32](fd)
        buf_mb[unsafe_offset=total_read] = 0

        # ── Parse "response":"..." ────────────────────────────────────────────
        comptime RTAG = "\"response\":\""
        comptime RTAG_LEN = 12
        var rtag_ptr = RTAG.unsafe_ptr()
        var text_len = 0
        var rfound = -1
        for si in range(total_read - RTAG_LEN):
            var ok = True
            for ti in range(RTAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != rtag_ptr[unsafe_offset=ti]: ok = False; break
            if ok: rfound = si + RTAG_LEN; break
        if rfound >= 0:
            var op = rfound
            while op < total_read and text_len < out_text_max - 1:
                var b2 = buf_mb[unsafe_offset=op]; op += 1
                if b2 == 92 and op < total_read:
                    var esc = buf_mb[unsafe_offset=op]; op += 1
                    if esc == 110:   out_text[unsafe_offset=text_len] = 10; text_len += 1
                    elif esc == 116: out_text[unsafe_offset=text_len] = 9;  text_len += 1
                    elif esc == 114: out_text[unsafe_offset=text_len] = 13; text_len += 1
                    elif esc == 34:  out_text[unsafe_offset=text_len] = 34; text_len += 1
                    elif esc == 92:  out_text[unsafe_offset=text_len] = 92; text_len += 1
                    else: out_text[unsafe_offset=text_len] = esc; text_len += 1
                elif b2 == 34:  # closing '"'
                    break
                else:
                    out_text[unsafe_offset=text_len] = b2; text_len += 1

        # ── Parse "logprobs":[...] → find minimum logprob ────────────────────
        comptime LTAG = "\"logprobs\":["
        comptime LTAG_LEN = 12
        var ltag_ptr = LTAG.unsafe_ptr()
        var lfound = -1
        for si in range(total_read - LTAG_LEN):
            var ok = True
            for ti in range(LTAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != ltag_ptr[unsafe_offset=ti]: ok = False; break
            if ok: lfound = si + LTAG_LEN; break
        if lfound >= 0:
            var lp = buf_mb.unsafe_offset(lfound)
            var min_lp = Float32(0.0)   # 0.0 = certain (exp(0)=1.0)
            var first = True
            while lp[unsafe_offset=0] != 93 and lp[unsafe_offset=0] != 0:   # ']' or null
                while lp[unsafe_offset=0] == 32 or lp[unsafe_offset=0] == 10 or lp[unsafe_offset=0] == 9 or lp[unsafe_offset=0] == 13:
                    lp = lp.unsafe_offset(1)
                if lp[unsafe_offset=0] == 93 or lp[unsafe_offset=0] == 0: break
                var start = lp
                while lp[unsafe_offset=0] != 44 and lp[unsafe_offset=0] != 93 and lp[unsafe_offset=0] != 0:
                    lp = lp.unsafe_offset(1)
                var flen = Int(lp) - Int(start)
                if flen > 0:
                    var val = Float32(_parse_float_lp(start, flen))
                    if first or val < min_lp:
                        min_lp = val; first = False
                if lp[unsafe_offset=0] == 44: lp = lp.unsafe_offset(1)
            out_min_logprob[unsafe_offset=0] = min_lp

        # ── Parse "done":true/false ───────────────────────────────────────────
        comptime DTAG = "\"done\":"
        comptime DTAG_LEN = 7
        var dtag_ptr = DTAG.unsafe_ptr()
        for si in range(total_read - DTAG_LEN - 4):
            var ok = True
            for ti in range(DTAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != dtag_ptr[unsafe_offset=ti]: ok = False; break
            if ok:
                # skip whitespace after ':'
                var dv = si + DTAG_LEN
                while dv < total_read and (buf_mb[unsafe_offset=dv] == 32 or buf_mb[unsafe_offset=dv] == 9):
                    dv += 1
                if dv < total_read and buf_mb[unsafe_offset=dv] == 116:  # 't' = true
                    out_done[unsafe_offset=0] = 1
                break

        buf.unsafe_free()
        return text_len
