"""MAXEngine — scaffold for Modular MAX in-process inference.

Current state (pre-Mojo 1.0):
  All methods proxy via HTTP/1.0 to an external MAX Serve / OpenAI-compatible server.

Post-Mojo-1.0 plan (H1 2026):
  Replace HTTP proxy calls with InferenceSession API (shown as POST-MOJO-1.0 comments).

  from max.engine import InferenceSession, InputMap
  var session = InferenceSession()
  var model   = session.load("sentence-transformers/all-MiniLM-L6-v2")
  var inputs  = InputMap()
  inputs.set("input_ids",      token_ids_tensor)
  inputs.set("attention_mask", mask_tensor)
  var outputs = model.execute(inputs)
  var emb = outputs.get[DType.float32]("pooler_output")
  # emb: NDBuffer[DType.float32] — zero-copy, no serialization, no HTTP overhead
"""

from src.common.ptr import is_not_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.ffi import external_call
from std.collections import List


@always_inline
def _me_parse_float32(p: Pointer[UInt8, MutUntrackedOrigin], plen: Int) -> Float32:
    var i = 0
    var sign: Float64 = 1.0
    if i < plen and p[i] == 45: sign = -1.0; i += 1
    elif i < plen and p[i] == 43: i += 1
    var integer: Float64 = 0.0
    while i < plen and p[i] >= 48 and p[i] <= 57:
        integer = integer * 10.0 + Float64(Int(p[i] - 48)); i += 1
    var frac: Float64 = 0.0; var frac_div: Float64 = 1.0
    if i < plen and p[i] == 46:
        i += 1
        while i < plen and p[i] >= 48 and p[i] <= 57:
            frac = frac * 10.0 + Float64(Int(p[i] - 48)); frac_div *= 10.0; i += 1
    var exp_val: Int = 0; var exp_sign: Int = 1
    if i < plen and (p[i] == 101 or p[i] == 69):
        i += 1
        if i < plen and p[i] == 45: exp_sign = -1; i += 1
        elif i < plen and p[i] == 43: i += 1
        while i < plen and p[i] >= 48 and p[i] <= 57:
            exp_val = exp_val * 10 + Int(p[i] - 48); i += 1
    var mantissa = sign * (integer + frac / frac_div)
    if exp_val != 0:
        var e: Float64 = 1.0
        for _ in range(exp_val): e *= 10.0
        if exp_sign > 0: mantissa *= e
        else:            mantissa /= e
    return Float32(mantissa)


@always_inline
def _me_escape_json(
    src: Pointer[UInt8, MutUntrackedOrigin],
    slen: Int,
    dst: Pointer[UInt8, MutUntrackedOrigin],
    mut dp: Int,
):
    for i in range(slen):
        var c = src[i]
        if c == 34:   dst[dp] = 92; dp += 1; dst[dp] = 34;  dp += 1
        elif c == 92: dst[dp] = 92; dp += 1; dst[dp] = 92;  dp += 1
        elif c == 10: dst[dp] = 92; dp += 1; dst[dp] = 110; dp += 1
        elif c == 13: dst[dp] = 92; dp += 1; dst[dp] = 114; dp += 1
        else:         dst[dp] = c;  dp += 1


struct MAXEngine(Movable):
    """Proxy to MAX Serve for embedding + LLM inference (pre-Mojo-1.0 HTTP path).

    Post-Mojo-1.0: InferenceSession replaces HTTP calls — same method signatures.
    """
    var host: String
    var port: Int
    var model: String
    var llm_host: String
    var llm_port: Int
    var llm_model: String
    var dim: Int
    var enabled: Bool
    var _recv_buf: Pointer[UInt8, MutUntrackedOrigin]
    var _req_buf:  Pointer[UInt8, MutUntrackedOrigin]

    def __init__(
        out self,
        host: String = "127.0.0.1",
        port: Int = 11434,
        model: String = "nomic-embed-text",
        llm_host: String = "127.0.0.1",
        llm_port: Int = 8000,
        llm_model: String = "meta-llama/Llama-3.1-8B-Instruct",
        dim: Int = 768,
        enabled: Bool = False,
    ):
        self.host      = host
        self.port      = port
        self.model     = model
        self.llm_host  = llm_host
        self.llm_port  = llm_port
        self.llm_model = llm_model
        self.dim       = dim
        self.enabled   = enabled
        self._recv_buf = alloc[UInt8](2 * 1024 * 1024)
        self._req_buf  = alloc[UInt8](65536)

    def __moveinit__(out self, owned take: Self):
        self.host      = take.host^
        self.port      = take.port
        self.model     = take.model^
        self.llm_host  = take.llm_host^
        self.llm_port  = take.llm_port
        self.llm_model = take.llm_model^
        self.dim       = take.dim
        self.enabled   = take.enabled
        self._recv_buf = take._recv_buf
        self._req_buf  = take._req_buf
        take._recv_buf = null_ptr[UInt8, MutUntrackedOrigin]()
        take._req_buf  = null_ptr[UInt8, MutUntrackedOrigin]()

    def __del__(owned self):
        if is_not_null(self._recv_buf): self._recv_buf.unsafe_free()
        if is_not_null(self._req_buf):  self._req_buf.unsafe_free()

    def tokenize(self, text_ptr: Pointer[UInt8, MutUntrackedOrigin], text_len: Int) -> List[Int32]:
        """Tokenize text to token IDs.
        POST-MOJO-1.0:
            from max.tokenizer import Tokenizer
            var tok = Tokenizer.from_pretrained(self.model)
            return tok.encode(text_ptr, text_len)
        """
        return List[Int32]()  # stub — HTTP proxy path does not need tokenization

    def embed_text(
        self,
        text_ptr: Pointer[UInt8, MutUntrackedOrigin],
        text_len: Int,
        out_buf: Pointer[Float32, MutUntrackedOrigin],
    ) -> Int:
        """Embed text → write `dim` Float32 values to out_buf. Returns dim on success, 0 on error.

        POST-MOJO-1.0:
            var tokens = self.tokenize(text_ptr, text_len)
            var inputs = InputMap()
            inputs.set("input_ids", tokens_tensor)
            inputs.set("attention_mask", ones_tensor)
            var outputs = self._session.execute(inputs)
            var emb = outputs.get[DType.float32]("pooler_output")
            unsafe_memcpy(out_buf.unsafe_bitcast[UInt8](), emb.unsafe_ptr().unsafe_bitcast[UInt8](), self.dim * 4)
            return self.dim
        """
        if not self.enabled:
            return 0
        var rp = self._req_buf; var rlen = 0
        var p1 = String('{"model":"')
        unsafe_memcpy(rp + rlen, p1.unsafe_ptr(), len(p1)); rlen += len(p1)
        unsafe_memcpy(rp + rlen, self.model.unsafe_ptr(), len(self.model)); rlen += len(self.model)
        var p2 = String('","input":"')
        unsafe_memcpy(rp + rlen, p2.unsafe_ptr(), len(p2)); rlen += len(p2)
        _me_escape_json(text_ptr, text_len, rp, rlen)
        var p3 = String('"}')
        unsafe_memcpy(rp + rlen, p3.unsafe_ptr(), len(p3)); rlen += len(p3)
        var chost = self.host
        var fd = external_call["pion_connect_tcp", Int32](chost.unsafe_cstr_ptr(), Int32(self.port))
        if fd < 0: return 0
        var hdr  = String("POST /v1/embeddings HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: ")
        var hdr2 = String(Int(rlen)) + "\r\n\r\n"
        _ = external_call["write", Int](fd, hdr.unsafe_ptr(), hdr.byte_length())
        _ = external_call["write", Int](fd, hdr2.unsafe_ptr(), hdr2.byte_length())
        _ = external_call["write", Int](fd, rp, rlen)
        var total = 0
        var chunk = external_call["read", Int](fd, self._recv_buf + total, 2*1024*1024 - total)
        while chunk > 0:
            total += chunk
            chunk = external_call["read", Int](fd, self._recv_buf + total, 2*1024*1024 - total)
        _ = external_call["close", Int](fd)
        # Parse float array from JSON response
        var body = self._recv_buf; var n_written = 0; var i = 0
        while i + 3 < total:  # skip HTTP headers
            if body[i] == 13 and body[i+1] == 10 and body[i+2] == 13 and body[i+3] == 10:
                i += 4; break
            i += 1
        while i < total - 1:
            if body[i] == 91:  # '[' start of embedding array
                i += 1
                while i < total and n_written < self.dim:
                    while i < total and (body[i] == 44 or body[i] == 32 or body[i] == 10 or body[i] == 13):
                        i += 1
                    if i >= total or body[i] == 93: break
                    var start = i
                    while i < total and body[i] != 44 and body[i] != 93 and body[i] != 32 and body[i] != 10:
                        i += 1
                    if i > start:
                        out_buf[n_written] = _me_parse_float32(body + start, i - start)
                        n_written += 1
                break
            i += 1
        return n_written if n_written == self.dim else 0

    def complete(
        self,
        prompt_ptr: Pointer[UInt8, MutUntrackedOrigin],
        prompt_len: Int,
        ctx_ptr: Pointer[UInt8, MutUntrackedOrigin],
        ctx_len: Int,
        out_buf: Pointer[UInt8, MutUntrackedOrigin],
        out_max: Int,
    ) -> Int:
        """LLM completion — write response text to out_buf. Returns byte count.

        POST-MOJO-1.0:
            var llm = InferenceSession().load(self.llm_model)
            var inputs = InputMap()
            inputs.set("input_ids", prompt_tokens_tensor)
            var outputs = llm.execute(inputs)
            var tok_ids = outputs.get[DType.int32]("output_ids")
            return self._detokenize(tok_ids, out_buf, out_max)
        """
        if not self.enabled: return 0
        var rp = self._req_buf; var rlen = 0
        var p1 = String('{"model":"')
        unsafe_memcpy(rp + rlen, p1.unsafe_ptr(), len(p1)); rlen += len(p1)
        unsafe_memcpy(rp + rlen, self.llm_model.unsafe_ptr(), len(self.llm_model)); rlen += len(self.llm_model)
        var p2 = String('","messages":[{"role":"user","content":"')
        unsafe_memcpy(rp + rlen, p2.unsafe_ptr(), len(p2)); rlen += len(p2)
        if ctx_len > 0:
            var cpfx = String("Context:\\n")
            unsafe_memcpy(rp + rlen, cpfx.unsafe_ptr(), len(cpfx)); rlen += len(cpfx)
            _me_escape_json(ctx_ptr, ctx_len, rp, rlen)
            var csep = String("\\n\\nUser: ")
            unsafe_memcpy(rp + rlen, csep.unsafe_ptr(), len(csep)); rlen += len(csep)
        _me_escape_json(prompt_ptr, prompt_len, rp, rlen)
        var p3 = String('"}],"stream":false}')
        unsafe_memcpy(rp + rlen, p3.unsafe_ptr(), len(p3)); rlen += len(p3)
        var chost = self.llm_host
        var fd = external_call["pion_connect_tcp", Int32](chost.unsafe_cstr_ptr(), Int32(self.llm_port))
        if fd < 0: return 0
        var hdr  = String("POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: ")
        var hdr2 = String(Int(rlen)) + "\r\n\r\n"
        _ = external_call["write", Int](fd, hdr.unsafe_ptr(), hdr.byte_length())
        _ = external_call["write", Int](fd, hdr2.unsafe_ptr(), hdr2.byte_length())
        _ = external_call["write", Int](fd, rp, rlen)
        var total = 0
        var chunk = external_call["read", Int](fd, self._recv_buf + total, 2*1024*1024 - total)
        while chunk > 0:
            total += chunk
            chunk = external_call["read", Int](fd, self._recv_buf + total, 2*1024*1024 - total)
        _ = external_call["close", Int](fd)
        # Find "content":"<text>" in JSON
        var body = self._recv_buf; var i = 0
        while i + 11 < total:
            if body[i]==34 and body[i+1]==99 and body[i+2]==111 and body[i+3]==110 and body[i+4]==116 and body[i+5]==101 and body[i+6]==110 and body[i+7]==116 and body[i+8]==34 and body[i+9]==58 and body[i+10]==34:
                i += 11
                var out_len = 0
                while i < total and body[i] != 34 and out_len < out_max - 1:
                    if body[i] == 92 and i + 1 < total:
                        i += 1
                        if   body[i] == 110: out_buf[out_len] = 10;        out_len += 1
                        elif body[i] == 116: out_buf[out_len] = 9;         out_len += 1
                        elif body[i] == 34:  out_buf[out_len] = 34;        out_len += 1
                        elif body[i] == 92:  out_buf[out_len] = 92;        out_len += 1
                        else:                out_buf[out_len] = body[i];   out_len += 1
                    else:
                        out_buf[out_len] = body[i]; out_len += 1
                    i += 1
                return out_len
            i += 1
        return 0

    def run_inference(self, tokens: List[Int32]) -> Pointer[Float32, MutUntrackedOrigin]:
        """Legacy token-based stub. Use embed_text() for the HTTP proxy path.
        POST-MOJO-1.0: replace with InferenceSession.execute() + tensor extraction."""
        var tensor = alloc[Float32](self.dim)
        for i in range(self.dim): tensor[i] = 0.0
        return tensor
