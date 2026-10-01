"""RerankClient — blocking HTTP/1.0 POST to a cross-encoder rerank endpoint.

gh #70: optional rerank stage on FT.HYBRID. The server-side rerank pattern
matches the EmbeddingClient design (per-call new TCP connection, blocking
read-until-close, ASCII JSON scanner) — kept deliberately separate from
EmbeddingClient because the wire shape differs:

  POST /v1/score HTTP/1.0
  Content-Type: application/json
  {"model":"<MODEL>","query":"<QUERY>","documents":["<DOC1>","<DOC2>",...]}

Response shape (SIE convention, matching most OSS cross-encoder servers):
  {"scores":[0.83, 0.21, 0.95, ...]}

Scope: opt-in only (FT.HYBRID `RERANK <host> <port> [top_k]`), so we don't
pay the connection overhead on the hot path. The 4 MB receive buffer
matches EmbeddingClient — a score array of 50 floats is far below that.

Thread-safety: each worker creates its own client, matching EmbeddingClient.
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.ffi import external_call


@always_inline
def _r_parse_float32(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float32:
    """Parse a single ASCII float from buffer[0..plen-1]. Shape matches the
    EmbeddingClient `_parse_float32` (sign, integer, fraction, optional
    decimal exponent). Reused identically so a future refactor can lift one
    helper to `src/common/utils.mojo` and delete the duplicate."""
    var i = 0
    var sign: Float64 = 1.0
    if i < plen and p[unsafe_offset=i] == 45: sign = -1.0; i += 1   # '-'
    elif i < plen and p[unsafe_offset=i] == 43: i += 1               # '+'

    var integer: Float64 = 0.0
    while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
        integer = integer * 10.0 + Float64(p[unsafe_offset=i] - 48)
        i += 1

    var frac: Float64 = 0.0
    var frac_mul: Float64 = 0.1
    if i < plen and p[unsafe_offset=i] == 46:   # '.'
        i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            frac = frac + frac_mul * Float64(p[unsafe_offset=i] - 48)
            frac_mul *= 0.1
            i += 1

    var exp_val: Int = 0
    var exp_sign: Int = 1
    if i < plen and (p[unsafe_offset=i] == 101 or p[unsafe_offset=i] == 69):  # 'e' or 'E'
        i += 1
        if i < plen and p[unsafe_offset=i] == 45: exp_sign = -1; i += 1
        elif i < plen and p[unsafe_offset=i] == 43: i += 1
        while i < plen and p[unsafe_offset=i] >= 48 and p[unsafe_offset=i] <= 57:
            exp_val = exp_val * 10 + Int(p[unsafe_offset=i] - 48)
            i += 1
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
def _r_escape_into(
    src: Pointer[UInt8, MutUntrackedOrigin], src_len: Int,
    mut dst: Pointer[UInt8, MutUntrackedOrigin], mut dp: Int,
):
    """Append src[0..src_len] into dst[dp..] with JSON escapes for the
    double-quote and backslash bytes, plus the three common whitespace
    controls (CR/LF/TAB). Other bytes pass through. `dp` is bumped in place.
    Caller must size `dst` for worst case (2× src)."""
    for bi in range(src_len):
        var b = src[unsafe_offset=bi]
        if b == 34:        # '"'
            dst[unsafe_offset=dp] = 92; dp += 1
            dst[unsafe_offset=dp] = 34; dp += 1
        elif b == 92:      # '\'
            dst[unsafe_offset=dp] = 92; dp += 1
            dst[unsafe_offset=dp] = 92; dp += 1
        elif b == 10:      # '\n'
            dst[unsafe_offset=dp] = 92; dp += 1
            dst[unsafe_offset=dp] = 110; dp += 1
        elif b == 13:      # '\r'
            dst[unsafe_offset=dp] = 92; dp += 1
            dst[unsafe_offset=dp] = 114; dp += 1
        elif b == 9:       # '\t'
            dst[unsafe_offset=dp] = 92; dp += 1
            dst[unsafe_offset=dp] = 116; dp += 1
        else:
            dst[unsafe_offset=dp] = b; dp += 1


struct RerankClient(Movable):
    """Blocking HTTP/1.0 client for a cross-encoder rerank endpoint.

    Construct per query/RERANK invocation (host/port come from the wire
    keyword, not from server config). `score()` is a one-shot blocking call —
    fine for opt-in FT.HYBRID; if hot-path rerank ever lands, swap in a
    pooled connection.
    """
    var host: String
    var port: Int
    var model: String

    def __init__(out self, host: String, port: Int, model: String):
        self.host = host
        self.port = port
        self.model = model

    def __moveinit__(out self, deinit take: Self):
        self.host = take.host^
        self.port = take.port
        self.model = take.model^

    def score(
        self,
        query_ptr: Pointer[UInt8, MutUntrackedOrigin], query_len: Int,
        doc_ptrs: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
        doc_lens: Pointer[Int, MutUntrackedOrigin],
        n_docs: Int,
        out_scores: Pointer[Float32, MutUntrackedOrigin],
    ) -> Bool:
        """Blocking POST to `<host>:<port>/v1/score`. Fills `out_scores[0..n_docs)`
        with the parsed `scores[]` array. Returns True on success.

        Failure modes (all return False, leave `out_scores` untouched):
        - TCP connect / send / recv error
        - HTTP non-200 (we don't parse the status line; absence of the
          `"scores":[` tag in the body is treated as failure)
        - JSON parse error (fewer than `n_docs` floats before the closing `]`)
        """
        if n_docs <= 0:
            return True

        # ── Open TCP ──────────────────────────────────────────────────────
        var chost = self.host
        var fd = external_call["pion_connect_tcp", Int32](chost.as_c_string_slice(), Int32(self.port))
        if fd < 0:
            return False

        # ── Compute upper-bound body size ─────────────────────────────────
        # {"model":"M","query":"Q","documents":["D1","D2",...]}
        # Worst case: every doc + query byte escapes to 2 bytes. Plus JSON
        # structural overhead.
        var total_text_bytes = query_len
        for i in range(n_docs):
            total_text_bytes += doc_lens[unsafe_offset=i]
        var max_body = (
            64                                  # JSON structure literals
            + self.model.byte_length()
            + 2 * query_len
            + 2 * total_text_bytes
            + n_docs * 4                        # "", + commas + brackets per doc
        )
        var body_buf = alloc[UInt8](max_body)
        var bp = 0

        # ── Build JSON body ───────────────────────────────────────────────
        # {"model":"
        var p1 = "{\"model\":\""
        unsafe_memcpy(dest=body_buf, src=p1.unsafe_ptr(), count=p1.byte_length()); bp += p1.byte_length()
        var cmodel = self.model
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=cmodel.unsafe_ptr(), count=cmodel.byte_length()); bp += cmodel.byte_length()
        # ","query":"
        var p2 = "\",\"query\":\""
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=p2.unsafe_ptr(), count=p2.byte_length()); bp += p2.byte_length()
        _r_escape_into(query_ptr, query_len, body_buf, bp)
        # ","documents":[
        var p3 = "\",\"documents\":["
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=p3.unsafe_ptr(), count=p3.byte_length()); bp += p3.byte_length()
        for i in range(n_docs):
            if i > 0:
                body_buf[unsafe_offset=bp] = 44; bp += 1   # ','
            body_buf[unsafe_offset=bp] = 34; bp += 1       # '"'
            _r_escape_into(doc_ptrs[unsafe_offset=i], doc_lens[unsafe_offset=i], body_buf, bp)
            body_buf[unsafe_offset=bp] = 34; bp += 1       # '"'
        # ]}
        body_buf[unsafe_offset=bp] = 93; bp += 1
        body_buf[unsafe_offset=bp] = 125; bp += 1
        var body_len = bp

        # ── Build HTTP request ────────────────────────────────────────────
        var body_len_str = String(body_len)
        var port_str = String(self.port)
        var req_hdr = String("POST /v1/score HTTP/1.0\r\nHost: ")
        req_hdr += self.host
        req_hdr += String(":")
        req_hdr += port_str
        req_hdr += String("\r\nContent-Type: application/json\r\nContent-Length: ")
        req_hdr += body_len_str
        req_hdr += String("\r\nConnection: close\r\n\r\n")
        var hdr_len = req_hdr.byte_length()

        var req_buf = alloc[UInt8](hdr_len + body_len + 1)
        unsafe_memcpy(dest=req_buf, src=req_hdr.unsafe_ptr(), count=hdr_len)
        unsafe_memcpy(dest=req_buf.unsafe_offset(hdr_len), src=body_buf, count=body_len)
        body_buf.unsafe_free()

        var sent = external_call["pion_write", Int64](fd, req_buf, Int(hdr_len + body_len))
        req_buf.unsafe_free()
        if sent < Int64(hdr_len + body_len):
            _ = external_call["close", Int32](fd)
            return False

        # ── Receive (read until close) ────────────────────────────────────
        var buf_size = 4 * 1024 * 1024
        var buf = alloc[UInt8](buf_size)
        var total_read = 0
        var nr: Int64 = 1
        while nr > 0 and total_read < buf_size - 1:
            nr = external_call["pion_read", Int64](fd, buf.unsafe_offset(total_read), buf_size - total_read - 1)
            if nr > 0: total_read += Int(nr)
        _ = external_call["close", Int32](fd)
        buf[unsafe_offset=total_read] = 0

        # ── Find "scores" then a `:` then a `[`, tolerating whitespace ────
        # Python's stdlib `json.dumps` and most SIE responses ship
        # `"scores": [...]` (with a space); avoid a hard exact-match scan.
        comptime KEY_LEN = 8  # bytes of `"scores"`
        var key = "\"scores\""
        var key_ptr = key.unsafe_ptr()
        var key_at = -1
        for si in range(total_read - KEY_LEN):
            var ok = True
            for ti in range(KEY_LEN):
                if buf[unsafe_offset=si + ti] != key_ptr[unsafe_offset=ti]: ok = False; break
            if ok: key_at = si + KEY_LEN; break
        if key_at < 0:
            buf.unsafe_free()
            return False
        # Skip whitespace + colon + whitespace + `[`.
        var ki = key_at
        while ki < total_read and (buf[unsafe_offset=ki] == 32 or buf[unsafe_offset=ki] == 10 or buf[unsafe_offset=ki] == 13 or buf[unsafe_offset=ki] == 9):
            ki += 1
        if ki >= total_read or buf[unsafe_offset=ki] != 58:  # ':'
            buf.unsafe_free(); return False
        ki += 1
        while ki < total_read and (buf[unsafe_offset=ki] == 32 or buf[unsafe_offset=ki] == 10 or buf[unsafe_offset=ki] == 13 or buf[unsafe_offset=ki] == 9):
            ki += 1
        if ki >= total_read or buf[unsafe_offset=ki] != 91:  # '['
            buf.unsafe_free(); return False
        var found = ki + 1

        # ── Parse N floats until ']' ──────────────────────────────────────
        var p = buf.unsafe_offset(found)
        var parsed = 0
        while parsed < n_docs:
            while p[unsafe_offset=0] == 32 or p[unsafe_offset=0] == 10 or p[unsafe_offset=0] == 13 or p[unsafe_offset=0] == 9:
                p = p.unsafe_offset(1)
            if p[unsafe_offset=0] == 93 or p[unsafe_offset=0] == 0: break
            var start = p
            while p[unsafe_offset=0] != 44 and p[unsafe_offset=0] != 93 and p[unsafe_offset=0] != 0:
                p = p.unsafe_offset(1)
            var flen = Int(p) - Int(start)
            if flen > 0:
                out_scores[unsafe_offset=parsed] = _r_parse_float32(start, flen)
                parsed += 1
            if p[unsafe_offset=0] == 44: p = p.unsafe_offset(1)

        buf.unsafe_free()
        return parsed == n_docs
