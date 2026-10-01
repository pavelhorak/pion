"""EmbeddingClient — blocking HTTP/1.0 POST to an OpenAI-compatible /v1/embeddings endpoint.

Protocol:
  POST /v1/embeddings HTTP/1.0
  Content-Type: application/json
  {"model":"MODEL","input":"QUERY"}

Response: JSON with `data[0].embedding` float array.

Thread-safety: each worker creates its own EmbeddingClient — no sharing required.
Hot-path note: embed() opens a new TCP connection per call.  For the semantic cache
this is acceptable (called only for AI.SEMANTIC_CACHE commands, not per KV op).
"""

from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.ffi import external_call


@always_inline
def _parse_float32(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float32:
    """Parse a single ASCII float from buffer[0..plen-1].
    Handles optional sign, integer part, fraction, and decimal exponent (e/E±N)."""
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


struct EmbeddingClient(Movable):
    """Blocking HTTP/1.0 client for OpenAI-compatible /v1/embeddings."""
    var host: String
    var port: Int
    var model: String
    var dimensions: Int

    def __init__(out self, host: String, port: Int, model: String, dimensions: Int):
        self.host = host
        self.port = port
        self.model = model
        self.dimensions = dimensions

    def __moveinit__(out self, deinit take: Self):
        self.host = take.host^
        self.port = take.port
        self.model = take.model^
        self.dimensions = take.dimensions

    def embed(self, text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int,
             out_vec: Pointer[Float32, MutUntrackedOrigin]) -> Bool:
        """Blocking HTTP POST; fills out_vec[0..dimensions-1]. Returns True on success."""
        var chost = self.host
        var fd = external_call["pion_connect_tcp", Int32](chost.as_c_string_slice(), Int32(self.port))
        if fd < 0: return False

        # ── Build JSON body ─────────────────────────────────────────────────
        # {"model":"MODEL","input":"TEXT"} — escape " and \ in TEXT
        var prefix = "{\"model\":\""
        var mid    = "\",\"input\":\""
        var suffix = "\"}"
        var prefix_len = prefix.byte_length()
        var mid_len    = mid.byte_length()
        var suffix_len = suffix.byte_length()
        var cmodel = self.model
        var model_len = cmodel.byte_length()

        # Upper-bound body size: prefix + model + mid + text*2 (escaping) + suffix
        var max_body = prefix_len + model_len + mid_len + text_len * 2 + suffix_len + 4
        var body_buf = alloc[UInt8](max_body)
        var bp = 0

        unsafe_memcpy(dest=body_buf, src=prefix.unsafe_ptr(), count=prefix_len); bp += prefix_len
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=cmodel.unsafe_ptr(), count=model_len); bp += model_len
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=mid.unsafe_ptr(), count=mid_len); bp += mid_len
        for bi in range(text_len):
            var b = text_ptr[unsafe_offset=bi]
            if b == 34:         # '"' → \"
                body_buf[unsafe_offset=bp] = 92; bp += 1
                body_buf[unsafe_offset=bp] = 34; bp += 1
            elif b == 92:       # '\' → \\
                body_buf[unsafe_offset=bp] = 92; bp += 1
                body_buf[unsafe_offset=bp] = 92; bp += 1
            else:
                body_buf[unsafe_offset=bp] = b; bp += 1
        unsafe_memcpy(dest=body_buf.unsafe_offset(bp), src=suffix.unsafe_ptr(), count=suffix_len); bp += suffix_len
        var body_len = bp

        # ── Build HTTP request ───────────────────────────────────────────────
        var body_len_str = String(body_len)
        var chost2  = self.host
        var port_str = String(self.port)
        var req_hdr = String("POST /v1/embeddings HTTP/1.0\r\nHost: ")
        req_hdr += chost2
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

        # ── Send ─────────────────────────────────────────────────────────────
        var total_len = hdr_len + body_len
        var sent = external_call["pion_write", Int64](fd, req_buf, Int(total_len))
        req_buf.unsafe_free()
        if sent < Int64(total_len):
            _ = external_call["close", Int32](fd)
            return False

        # ── Receive (read until close) ────────────────────────────────────────
        var buf_size = 4 * 1024 * 1024   # 4MB: plenty for 1536-float responses
        var buf = alloc[UInt8](buf_size)
        var total_read = 0
        var nr: Int64 = 1
        while nr > 0 and total_read < buf_size - 1:
            nr = external_call["pion_read", Int64](fd, buf.unsafe_offset(total_read), buf_size - total_read - 1)
            if nr > 0: total_read += Int(nr)
        _ = external_call["close", Int32](fd)
        buf[unsafe_offset=total_read] = 0   # null-terminate for scanner

        # ── Parse JSON: find "embedding":[ ───────────────────────────────────
        # Tag bytes: " e m b e d d i n g " : [
        # (34) 101 109 98 101 100 100 105 110 103 (34) 58 91
        comptime TAG_LEN = 13
        var tag = "\"embedding\":["
        var tag_ptr = tag.unsafe_ptr()
        var found = -1
        for si in range(total_read - TAG_LEN):
            var tag_ok = True
            for ti in range(TAG_LEN):
                if buf[unsafe_offset=si + ti] != tag_ptr[unsafe_offset=ti]: tag_ok = False; break
            if tag_ok: found = si + TAG_LEN; break

        if found < 0:
            buf.unsafe_free()
            return False

        # ── Parse float array until ']' ───────────────────────────────────────
        var p = buf.unsafe_offset(found)
        var parsed = 0
        while parsed < self.dimensions:
            while p[unsafe_offset=0] == 32 or p[unsafe_offset=0] == 10 or p[unsafe_offset=0] == 13 or p[unsafe_offset=0] == 9:
                p = p.unsafe_offset(1)
            if p[unsafe_offset=0] == 93 or p[unsafe_offset=0] == 0: break   # ']' or null

            var start = p
            while p[unsafe_offset=0] != 44 and p[unsafe_offset=0] != 93 and p[unsafe_offset=0] != 0:
                p = p.unsafe_offset(1)
            var flen = Int(p) - Int(start)
            if flen > 0:
                out_vec[unsafe_offset=parsed] = _parse_float32(start, flen)
                parsed += 1
            if p[unsafe_offset=0] == 44: p = p.unsafe_offset(1)   # skip ','

        buf.unsafe_free()
        return parsed == self.dimensions
