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
    """JSON-escape src[0..slen-1] into dst starting at dp; advances dp. Every
    control byte below 0x20 must be escaped or the body is invalid JSON — a
    lone tab or form-feed in the prompt used to go through raw and the server
    rejected the request (#51). dst must have room for 6 bytes per input byte."""
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
        elif b == 9:      # '\t' → \t
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 116; dp += 1
        elif b < 32:      # other C0 control → \u00XX
            dst[unsafe_offset=dp] = 92; dp += 1; dst[unsafe_offset=dp] = 117; dp += 1   # \u
            dst[unsafe_offset=dp] = 48; dp += 1; dst[unsafe_offset=dp] = 48; dp += 1    # 00
            var hi = Int(b) >> 4
            var lo = Int(b) & 0xF
            dst[unsafe_offset=dp] = UInt8(48 + hi if hi < 10 else 87 + hi); dp += 1
            dst[unsafe_offset=dp] = UInt8(48 + lo if lo < 10 else 87 + lo); dp += 1
        else:
            dst[unsafe_offset=dp] = b; dp += 1


@always_inline
def _hexval(b: UInt8) -> Int:
    """A hex digit's value, or -1."""
    var c = Int(b)
    if c >= 48 and c <= 57: return c - 48
    if c >= 97 and c <= 102: return c - 87
    if c >= 65 and c <= 70: return c - 55
    return -1


@always_inline
def _put_utf8(cp: Int, dst: Pointer[UInt8, MutUntrackedOrigin], mut o: Int, out_max: Int):
    """Encode code point `cp` as UTF-8 into out, advancing o, within out_max-1."""
    if cp < 0x80:
        if o < out_max - 1: dst[unsafe_offset=o] = UInt8(cp); o += 1
    elif cp < 0x800:
        if o < out_max - 2:
            dst[unsafe_offset=o] = UInt8(0xC0 | (cp >> 6)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | (cp & 0x3F)); o += 1
    elif cp < 0x10000:
        if o < out_max - 3:
            dst[unsafe_offset=o] = UInt8(0xE0 | (cp >> 12)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | ((cp >> 6) & 0x3F)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | (cp & 0x3F)); o += 1
    else:
        if o < out_max - 4:
            dst[unsafe_offset=o] = UInt8(0xF0 | (cp >> 18)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | ((cp >> 12) & 0x3F)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | ((cp >> 6) & 0x3F)); o += 1
            dst[unsafe_offset=o] = UInt8(0x80 | (cp & 0x3F)); o += 1


def _decode_json_string(buf: Pointer[UInt8, MutUntrackedOrigin], start: Int, total: Int,
                        dst: Pointer[UInt8, MutUntrackedOrigin], out_max: Int) -> Int:
    """Decode a JSON string body from `start` until its unescaped closing `"`
    (or end of buffer), writing the bytes into out. Handles \\n \\t \\r \\" \\\\
    \\/ \\b \\f and \\uXXXX, including a surrogate pair → one 4-byte UTF-8
    char (#51: Ollama and OpenAI write `<` `>` `&` as \\u003c etc, and emoji
    as surrogate pairs; the old loop copied `\\u003c` through as `u003c`).
    Returns bytes written."""
    var op = start
    var o = 0
    while op < total and o < out_max - 1:
        var b = buf[unsafe_offset=op]
        if b == 34:            # closing "
            break
        if b == 92 and op + 1 < total:   # backslash
            var e = buf[unsafe_offset=op + 1]
            if e == 117 and op + 5 < total:    # \u XXXX
                var h0 = _hexval(buf[unsafe_offset=op + 2]); var h1 = _hexval(buf[unsafe_offset=op + 3])
                var h2 = _hexval(buf[unsafe_offset=op + 4]); var h3 = _hexval(buf[unsafe_offset=op + 5])
                if h0 < 0 or h1 < 0 or h2 < 0 or h3 < 0:
                    dst[unsafe_offset=o] = b; o += 1; op += 1
                    continue
                var cp = (h0 << 12) | (h1 << 8) | (h2 << 4) | h3
                op += 6
                # a high surrogate followed by \uXXXX low surrogate → one char
                if cp >= 0xD800 and cp <= 0xDBFF and op + 5 < total \
                   and buf[unsafe_offset=op] == 92 and buf[unsafe_offset=op + 1] == 117:
                    var l0 = _hexval(buf[unsafe_offset=op + 2]); var l1 = _hexval(buf[unsafe_offset=op + 3])
                    var l2 = _hexval(buf[unsafe_offset=op + 4]); var l3 = _hexval(buf[unsafe_offset=op + 5])
                    if l0 >= 0 and l1 >= 0 and l2 >= 0 and l3 >= 0:
                        var lo = (l0 << 12) | (l1 << 8) | (l2 << 4) | l3
                        if lo >= 0xDC00 and lo <= 0xDFFF:
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                            op += 6
                _put_utf8(cp, dst, o, out_max)
                continue
            op += 2
            if e == 110:   dst[unsafe_offset=o] = 10; o += 1          # \n
            elif e == 116: dst[unsafe_offset=o] = 9;  o += 1          # \t
            elif e == 114: dst[unsafe_offset=o] = 13; o += 1          # \r
            elif e == 98:  dst[unsafe_offset=o] = 8;  o += 1          # \b
            elif e == 102: dst[unsafe_offset=o] = 12; o += 1          # \f
            elif e == 47:  dst[unsafe_offset=o] = 47; o += 1          # \/
            elif e == 34:  dst[unsafe_offset=o] = 34; o += 1          # \"
            elif e == 92:  dst[unsafe_offset=o] = 92; o += 1          # \\
            else:          dst[unsafe_offset=o] = e;  o += 1
        else:
            dst[unsafe_offset=o] = b; o += 1; op += 1
    return o


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
        var content_max = context_len * 6 + prompt_len * 6 + 64   # #51: \u00XX is 6 bytes/char
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
        # #51: TAG is 11 bytes (`"content":"`). It was compared as 12, so the
        # 12th byte — the first character of the reply, never the tag's — never
        # matched and AI.CHAT answered nil to every reply. AI.COMPLETE's
        # `"response":"` is genuinely 12, which is why it worked and this did
        # not. The length comes from the literal now, so it cannot drift again.
        comptime TAG = "\"content\":\""
        comptime TAG_LEN = TAG.byte_length()
        var tag_ptr = TAG.unsafe_ptr()
        var found = -1
        for si in range(total_read - TAG_LEN):
            var ok = True
            for ti in range(TAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != tag_ptr[unsafe_offset=ti]: ok = False; break
            if ok: found = si + TAG_LEN; break

        if found < 0:
            buf.unsafe_free(); return 0

        # Decode the JSON string value, including \uXXXX (Ollama writes < > &
        # as < etc), through the shared decoder (#51).
        var dp2 = _decode_json_string(buf_mb, found, total_read, out_buf, out_max)
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

        var max_body = p1l + model_len + m1l + prompt_len * 6 + s1l + np_len + s2l + 4   # #51: 6 bytes/char
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
        comptime RTAG_LEN = RTAG.byte_length()   # #51: from the literal, not hand-counted
        var rtag_ptr = RTAG.unsafe_ptr()
        var text_len = 0
        var rfound = -1
        for si in range(total_read - RTAG_LEN):
            var ok = True
            for ti in range(RTAG_LEN):
                if buf_mb[unsafe_offset=si + ti] != rtag_ptr[unsafe_offset=ti]: ok = False; break
            if ok: rfound = si + RTAG_LEN; break
        if rfound >= 0:
            # #51: same shared decoder as AI.CHAT, so \uXXXX and surrogate
            # pairs come back decoded here too.
            text_len = _decode_json_string(buf_mb, rfound, total_read, out_text, out_text_max)

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
