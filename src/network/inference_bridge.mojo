"""InferenceBridge — non-blocking IPC client for the M1 Python inference sidecar.

Communicates over a Unix domain socket using a binary protocol:
    Request:  [type:1B][req_id:4B LE][body_len:4B LE][body]
    Response: [type:1B][req_id:4B LE][status:1B][body_len:4B LE][body]

Types: EMBED=1, GENERATE=2, LOAD_MODEL=3, HEALTH=4, SHUTDOWN=5
Status: OK=0, ERROR=1
"""

from src.common.ptr import is_not_null
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.memory import unsafe_memcpy
from std.ffi import external_call

# Protocol constants
comptime INFER_MSG_EMBED = UInt8(1)
comptime INFER_MSG_GENERATE = UInt8(2)
comptime INFER_MSG_LOAD_MODEL = UInt8(3)
comptime INFER_MSG_HEALTH = UInt8(4)
comptime INFER_MSG_SHUTDOWN = UInt8(5)

comptime INFER_STATUS_OK = UInt8(0)
comptime INFER_STATUS_ERROR = UInt8(1)

comptime INFER_MODEL_EMBEDDING = UInt8(0)
comptime INFER_MODEL_LLM = UInt8(1)

comptime INFER_REQ_HEADER_SIZE = 9    # type(1) + req_id(4) + body_len(4)
comptime INFER_RESP_HEADER_SIZE = 10  # type(1) + req_id(4) + status(1) + body_len(4)

comptime INFER_SEND_BUF_SIZE = 65536
comptime INFER_RECV_BUF_SIZE = 2 * 1024 * 1024  # 2MB for embedding vectors + LLM output


struct InferenceBridge(Movable):
    """Non-blocking IPC client for the Python inference sidecar."""

    var socket_fd: Int32
    var sidecar_pid: Int32
    var next_req_id: UInt32
    var send_buf: Pointer[UInt8, MutUntrackedOrigin]
    var recv_buf: Pointer[UInt8, MutUntrackedOrigin]
    var recv_offset: Int       # bytes accumulated in recv_buf
    var connected: Bool
    var enabled: Bool
    var socket_path: String

    def __init__(out self, enabled: Bool, socket_path: String):
        self.socket_fd = Int32(-1)
        self.sidecar_pid = Int32(-1)
        self.next_req_id = UInt32(1)
        self.send_buf = alloc[UInt8](INFER_SEND_BUF_SIZE)
        self.recv_buf = alloc[UInt8](INFER_RECV_BUF_SIZE)
        self.recv_offset = 0
        self.connected = False
        self.enabled = enabled
        self.socket_path = socket_path

    def __moveinit__(out self, deinit take: Self):
        self.socket_fd = take.socket_fd
        self.sidecar_pid = take.sidecar_pid
        self.next_req_id = take.next_req_id
        self.send_buf = take.send_buf
        self.recv_buf = take.recv_buf
        self.recv_offset = take.recv_offset
        self.connected = take.connected
        self.enabled = take.enabled
        self.socket_path = take.socket_path^

    def __del__(deinit self):
        if is_not_null(self.send_buf):
            self.send_buf.unsafe_free()
        if is_not_null(self.recv_buf):
            self.recv_buf.unsafe_free()
        if self.connected and self.socket_fd >= 0:
            _ = external_call["close", Int32](self.socket_fd)

    @always_inline
    def connect(mut self) -> Bool:
        """Connect to the inference sidecar Unix socket. Returns True on success."""
        if not self.enabled:
            return False
        var c_path = self.socket_path
        var fd = external_call["pion_connect_unix", Int32](c_path.as_c_string_slice())
        if fd < 0:
            return False
        self.socket_fd = fd
        self.connected = True
        self.recv_offset = 0
        return True

    @always_inline
    def _alloc_req_id(mut self) -> UInt32:
        """Allocate a monotonically increasing request ID."""
        var rid = self.next_req_id
        self.next_req_id += 1
        return rid

    @always_inline
    def _write_header(self, buf: Pointer[UInt8, MutUntrackedOrigin],
                     msg_type: UInt8, req_id: UInt32, body_len: UInt32):
        """Write a 9-byte request header into buf."""
        buf[unsafe_offset=0] = msg_type
        # req_id LE
        buf[unsafe_offset=1] = UInt8(req_id & 0xFF)
        buf[unsafe_offset=2] = UInt8((req_id >> 8) & 0xFF)
        buf[unsafe_offset=3] = UInt8((req_id >> 16) & 0xFF)
        buf[unsafe_offset=4] = UInt8((req_id >> 24) & 0xFF)
        # body_len LE
        buf[unsafe_offset=5] = UInt8(body_len & 0xFF)
        buf[unsafe_offset=6] = UInt8((body_len >> 8) & 0xFF)
        buf[unsafe_offset=7] = UInt8((body_len >> 16) & 0xFF)
        buf[unsafe_offset=8] = UInt8((body_len >> 24) & 0xFF)

    @always_inline
    def _write_u32_le(self, buf: Pointer[UInt8, MutUntrackedOrigin],
                     offset: Int, val: UInt32):
        """Write a little-endian uint32 at offset."""
        buf[unsafe_offset=offset] = UInt8(val & 0xFF)
        buf[unsafe_offset=offset + 1] = UInt8((val >> 8) & 0xFF)
        buf[unsafe_offset=offset + 2] = UInt8((val >> 16) & 0xFF)
        buf[unsafe_offset=offset + 3] = UInt8((val >> 24) & 0xFF)

    @always_inline
    def _read_u32_le(self, buf: Pointer[UInt8, MutUntrackedOrigin],
                    offset: Int) -> UInt32:
        """Read a little-endian uint32 from offset."""
        return UInt32(buf[unsafe_offset=offset]) | (UInt32(buf[unsafe_offset=offset+1]) << 8) | (UInt32(buf[unsafe_offset=offset+2]) << 16) | (UInt32(buf[unsafe_offset=offset+3]) << 24)

    @always_inline
    def _send_all(self, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Bool:
        """Blocking write of all bytes. Returns True on success."""
        var sent = 0
        while sent < length:
            var n = external_call["pion_write", Int](self.socket_fd, buf.unsafe_offset(sent), length - sent)
            if n <= 0:
                return False
            sent += n
        return True

    @always_inline
    def _recv_all(self, buf: Pointer[UInt8, MutUntrackedOrigin], length: Int) -> Bool:
        """Blocking read of exactly `length` bytes. Returns True on success."""
        var received = 0
        while received < length:
            var n = external_call["pion_read", Int](self.socket_fd, buf.unsafe_offset(received), length - received)
            if n <= 0:
                return False
            received += n
        return True

    def send_embed_request(mut self,
                          text_ptr: Pointer[UInt8, MutUntrackedOrigin],
                          text_len: Int) -> UInt32:
        """Send an EMBED request. Returns req_id (0 on failure)."""
        if not self.connected:
            return 0
        var req_id = self._alloc_req_id()
        var body_len = UInt32(4 + text_len)  # [text_len:4][text]
        var total = INFER_REQ_HEADER_SIZE + Int(body_len)
        if total > INFER_SEND_BUF_SIZE:
            return 0
        self._write_header(self.send_buf, INFER_MSG_EMBED, req_id, body_len)
        self._write_u32_le(self.send_buf, INFER_REQ_HEADER_SIZE, UInt32(text_len))
        unsafe_memcpy(dest=self.send_buf.unsafe_offset(INFER_REQ_HEADER_SIZE).unsafe_offset(4), src=text_ptr, count=text_len)
        if not self._send_all(self.send_buf, total):
            return 0
        return req_id

    def send_generate_request(mut self,
                             prompt_ptr: Pointer[UInt8, MutUntrackedOrigin],
                             prompt_len: Int,
                             context_ptr: Pointer[UInt8, MutUntrackedOrigin],
                             context_len: Int,
                             max_tokens: Int) -> UInt32:
        """Send a GENERATE request. Returns req_id (0 on failure)."""
        if not self.connected:
            return 0
        var req_id = self._alloc_req_id()
        # body: [prompt_len:4][prompt][ctx_len:4][ctx][max_tokens:4]
        var body_len = UInt32(4 + prompt_len + 4 + context_len + 4)
        var total = INFER_REQ_HEADER_SIZE + Int(body_len)
        if total > INFER_SEND_BUF_SIZE:
            return 0
        self._write_header(self.send_buf, INFER_MSG_GENERATE, req_id, body_len)
        var off = INFER_REQ_HEADER_SIZE
        self._write_u32_le(self.send_buf, off, UInt32(prompt_len)); off += 4
        unsafe_memcpy(dest=self.send_buf.unsafe_offset(off), src=prompt_ptr, count=prompt_len); off += prompt_len
        self._write_u32_le(self.send_buf, off, UInt32(context_len)); off += 4
        if context_len > 0:
            unsafe_memcpy(dest=self.send_buf.unsafe_offset(off), src=context_ptr, count=context_len); off += context_len
        self._write_u32_le(self.send_buf, off, UInt32(max_tokens))
        if not self._send_all(self.send_buf, total):
            return 0
        return req_id

    def send_load_model_request(mut self,
                               model_id_ptr: Pointer[UInt8, MutUntrackedOrigin],
                               model_id_len: Int,
                               model_type: UInt8) -> UInt32:
        """Send a LOAD_MODEL request. Returns req_id (0 on failure)."""
        if not self.connected:
            return 0
        var req_id = self._alloc_req_id()
        # body: [model_id_len:4][model_id][model_type:1]
        var body_len = UInt32(4 + model_id_len + 1)
        var total = INFER_REQ_HEADER_SIZE + Int(body_len)
        if total > INFER_SEND_BUF_SIZE:
            return 0
        self._write_header(self.send_buf, INFER_MSG_LOAD_MODEL, req_id, body_len)
        var off = INFER_REQ_HEADER_SIZE
        self._write_u32_le(self.send_buf, off, UInt32(model_id_len)); off += 4
        unsafe_memcpy(dest=self.send_buf.unsafe_offset(off), src=model_id_ptr, count=model_id_len); off += model_id_len
        self.send_buf[unsafe_offset=off] = model_type
        if not self._send_all(self.send_buf, total):
            return 0
        return req_id

    def recv_response_blocking(mut self,
                              out_body: Pointer[UInt8, MutUntrackedOrigin],
                              out_body_max: Int) -> InferenceResponse:
        """Blocking receive of a response. Returns InferenceResponse with parsed fields."""
        # Read header: type(1) + req_id(4) + status(1) + body_len(4) = 10 bytes
        var hdr = self.recv_buf
        if not self._recv_all(hdr, INFER_RESP_HEADER_SIZE):
            return InferenceResponse(UInt8(0), UInt32(0), INFER_STATUS_ERROR, 0)
        var msg_type = hdr[unsafe_offset=0]
        var req_id = self._read_u32_le(hdr, 1)
        var status = hdr[unsafe_offset=5]
        var body_len = Int(self._read_u32_le(hdr, 6))
        # Read body
        var actual_read = body_len if body_len <= out_body_max else out_body_max
        if actual_read > 0:
            if not self._recv_all(out_body, actual_read):
                return InferenceResponse(msg_type, req_id, INFER_STATUS_ERROR, 0)
        # Drain any remaining bytes if body was larger than out_body_max
        if body_len > out_body_max:
            var skip = body_len - out_body_max
            var skip_buf = alloc[UInt8](skip)
            _ = self._recv_all(skip_buf, skip)
            skip_buf.unsafe_free()
        return InferenceResponse(msg_type, req_id, status, actual_read)

    def embed_blocking(mut self,
                      text_ptr: Pointer[UInt8, MutUntrackedOrigin],
                      text_len: Int,
                      out_buf: Pointer[Float32, MutUntrackedOrigin]) -> Int:
        """Synchronous embed: send request, wait for response, write floats to out_buf.
        Returns dimension count on success, 0 on failure.
        Used by SemanticCache integration where blocking is acceptable (single-worker mode)."""
        var req_id = self.send_embed_request(text_ptr, text_len)
        if req_id == 0:
            return 0
        # Response body: [dim:u32][f32 * dim]
        var body_buf = alloc[UInt8](INFER_RECV_BUF_SIZE)
        var resp = self.recv_response_blocking(body_buf, INFER_RECV_BUF_SIZE)
        if resp.status != INFER_STATUS_OK or resp.body_len < 4:
            body_buf.unsafe_free()
            return 0
        var dim = Int(self._read_u32_le(body_buf, 0))
        var float_bytes = dim * 4
        if resp.body_len < 4 + float_bytes:
            body_buf.unsafe_free()
            return 0
        unsafe_memcpy(dest=out_buf.unsafe_bitcast[UInt8](), src=body_buf.unsafe_offset(4), count=float_bytes)
        body_buf.unsafe_free()
        return dim

    def shutdown(mut self):
        """Send SHUTDOWN to sidecar, close socket."""
        if self.connected and self.socket_fd >= 0:
            # Send shutdown message (header only, no body)
            self._write_header(self.send_buf, INFER_MSG_SHUTDOWN, UInt32(0), UInt32(0))
            _ = self._send_all(self.send_buf, INFER_REQ_HEADER_SIZE)
            _ = external_call["close", Int32](self.socket_fd)
            self.socket_fd = Int32(-1)
            self.connected = False
        # Wait for sidecar to exit
        if self.sidecar_pid > 0:
            for _ in range(50):  # 5 seconds
                var r = external_call["pion_waitpid_nohang", Int32](self.sidecar_pid)
                if r != 0:
                    break
                _ = external_call["usleep", Int32](Int32(100000))  # 100ms
            # Force kill if still running
            var r2 = external_call["pion_waitpid_nohang", Int32](self.sidecar_pid)
            if r2 == 0:
                _ = external_call["pion_kill", Int32](self.sidecar_pid, Int32(9))  # SIGKILL
            self.sidecar_pid = Int32(-1)


struct InferenceResponse(Copyable, Movable, ImplicitlyCopyable):
    """Parsed response from the inference sidecar."""
    var msg_type: UInt8
    var req_id: UInt32
    var status: UInt8
    var body_len: Int

    def __init__(out self, msg_type: UInt8, req_id: UInt32, status: UInt8, body_len: Int):
        self.msg_type = msg_type
        self.req_id = req_id
        self.status = status
        self.body_len = body_len
