"""XDP/AF_XDP kernel bypass module for Pion.

Provides zero-copy packet I/O by bypassing the Linux kernel TCP/IP stack entirely.
Packets are intercepted at the NIC driver level by an XDP BPF program and delivered
directly to userspace via AF_XDP sockets and shared UMEM memory regions.

Architecture:
    NIC → XDP BPF filter → AF_XDP socket → UMEM (shared mmap) → Mojo event loop
    Mojo event loop → TCP-Lite response → UMEM → AF_XDP TX ring → NIC

Performance target: 5M+ QPS single worker (vs 3.3M with io_uring).
The improvement comes from eliminating:
  - Kernel TCP/IP stack processing (~400ns per packet)
  - Socket buffer copies (~200ns per packet)
  - Context switches between kernel and userspace

Requirements: Linux 5.4+, CAP_NET_ADMIN, AF_XDP-capable NIC driver.
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc
from std.ffi import external_call
from std.memory import unsafe_memset, unsafe_memcpy


# TCP flags
comptime TCP_FIN  = UInt8(0x01)
comptime TCP_SYN  = UInt8(0x02)
comptime TCP_RST  = UInt8(0x04)
comptime TCP_PSH  = UInt8(0x08)
comptime TCP_ACK  = UInt8(0x10)
comptime TCP_URG  = UInt8(0x20)
comptime TCP_SYN_ACK = UInt8(0x12)
comptime TCP_FIN_ACK = UInt8(0x11)

# XDP batch size — max frames to process per poll
comptime XDP_BATCH_SIZE = 64

# Invalid frame address sentinel
comptime XDP_FRAME_INVALID = UInt64(0xFFFFFFFFFFFFFFFF)


struct XDPFrameInfo(Copyable, Movable, ImplicitlyCopyable):
    """Parsed TCP frame metadata extracted from a raw Ethernet frame."""
    var payload_ptr: Pointer[UInt8, MutUntrackedOrigin]
    var payload_len: Int
    var seq_num:     UInt32
    var ack_num:     UInt32
    var flags:       UInt8
    var sport:       UInt16
    var dport:       UInt16

    def __init__(out self):
        self.payload_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        self.payload_len = 0
        self.seq_num = 0
        self.ack_num = 0
        self.flags = 0
        self.sport = 0
        self.dport = 0


struct TCPConnection(Copyable, Movable, ImplicitlyCopyable):
    """Minimal TCP connection state for the TCP-Lite state machine.
    Tracks sequence numbers and connection state for a single client.

    States: CLOSED=0, SYN_RECEIVED=1, ESTABLISHED=2, FIN_WAIT=3
    """
    var state:        UInt8   # 0=CLOSED, 1=SYN_RECEIVED, 2=ESTABLISHED, 3=FIN_WAIT
    var our_seq:      UInt32  # our current sequence number
    var their_seq:    UInt32  # expected next sequence from peer
    var their_ack:    UInt32  # last ack we received from peer
    var sport:        UInt16  # client's source port (our identifier)
    var dport:        UInt16  # our port

    def __init__(out self):
        self.state = 0
        self.our_seq = 0
        self.their_seq = 0
        self.their_ack = 0
        self.sport = 0
        self.dport = 0


struct XDPEngine:
    """XDP/AF_XDP engine for zero-copy packet I/O.

    Manages AF_XDP socket lifecycle, UMEM frame allocation, and the TCP-Lite
    state machine for handling connections without the kernel TCP/IP stack.

    Two setup modes:
    1. Single-worker: setup() — creates everything (BPF, XSKMAP, socket)
    2. Multi-worker:  setup_worker() — uses shared BPF/XSKMAP from parent

    Usage from NetworkEngine:
        var xdp = XDPEngine()
        if xdp.setup("eth0", 0, 1974):
            xdp.run_event_loop(...)  # called from run_server_xdp()
    """
    var handle: Pointer[NoneType, MutUntrackedOrigin]
    var umem_base: Pointer[UInt8, MutUntrackedOrigin]
    var active: Bool
    var target_port: UInt16
    var interface_name: String
    var queue_id: Int
    var is_shared_mode: Bool  # True if using shared BPF/XSKMAP (multi-worker)

    # TCP-Lite connection table: indexed by client source port (0..65535)
    # Each entry tracks the TCP state machine for one connection.
    var connections: Pointer[TCPConnection, MutUntrackedOrigin]

    # Batch buffers for poll_rx (pre-allocated to avoid per-poll allocation)
    var rx_addrs: Pointer[UInt64, MutUntrackedOrigin]
    var rx_lens:  Pointer[UInt32, MutUntrackedOrigin]

    def __init__(out self):
        self.handle = null_ptr[NoneType, MutUntrackedOrigin]()
        self.umem_base = null_ptr[UInt8, MutUntrackedOrigin]()
        self.active = False
        self.target_port = 1974
        self.interface_name = "eth0"
        self.queue_id = 0
        self.is_shared_mode = False
        self.connections = null_ptr[TCPConnection, MutUntrackedOrigin]()
        self.rx_addrs = null_ptr[UInt64, MutUntrackedOrigin]()
        self.rx_lens = null_ptr[UInt32, MutUntrackedOrigin]()

    def setup(mut self, interface: String, queue_id: Int, port: UInt16) -> Bool:
        """Initialize the XDP engine: create AF_XDP socket, load BPF program, allocate UMEM.
        Single-worker mode: creates its own BPF + XSKMAP.
        Returns True on success, False if XDP is not available (fallback to io_uring)."""
        self.target_port = port
        self.interface_name = interface
        self.queue_id = queue_id
        self.is_shared_mode = False

        var iface_cstr = interface
        self.handle = external_call["pion_xdp_create", Pointer[NoneType, MutUntrackedOrigin]](
            iface_cstr.as_c_string_slice(), Int32(queue_id), port
        )
        if is_null(self.handle):
            return False

        self.umem_base = external_call["pion_xdp_get_umem", Pointer[UInt8, MutUntrackedOrigin]](
            self.handle
        )
        if is_null(self.umem_base):
            external_call["pion_xdp_destroy", NoneType](self.handle)
            self.handle = null_ptr[NoneType, MutUntrackedOrigin]()
            return False

        # Allocate connection table (65536 entries, one per source port)
        self.connections = alloc[TCPConnection](65536)
        unsafe_memset(self.connections.unsafe_bitcast[UInt8](), 0, 65536 * 16)  # sizeof(TCPConnection) ≈ 16

        # Allocate batch buffers
        self.rx_addrs = alloc[UInt64](XDP_BATCH_SIZE)
        self.rx_lens = alloc[UInt32](XDP_BATCH_SIZE)

        self.active = True
        print("XDP engine active: " + interface + " queue=" + String(queue_id) + " port=" + String(Int(port)))
        return True

    def setup_worker(mut self, interface: String, queue_id: Int, port: UInt16,
                     shared_xskmap_fd: Int32, shared_bpf_fd: Int32) -> Bool:
        """Initialize XDP engine in multi-worker mode using shared BPF/XSKMAP.
        The BPF program and XSKMAP are created once before parallelize and shared
        across all workers. Each worker only creates its own AF_XDP socket + UMEM."""
        self.target_port = port
        self.interface_name = interface
        self.queue_id = queue_id
        self.is_shared_mode = True

        var iface_cstr = interface
        self.handle = external_call["pion_xdp_create_worker", Pointer[NoneType, MutUntrackedOrigin]](
            iface_cstr.as_c_string_slice(), Int32(queue_id), port,
            shared_xskmap_fd, shared_bpf_fd
        )
        if is_null(self.handle):
            return False

        self.umem_base = external_call["pion_xdp_get_umem", Pointer[UInt8, MutUntrackedOrigin]](
            self.handle
        )
        if is_null(self.umem_base):
            external_call["pion_xdp_destroy_worker", NoneType](self.handle)
            self.handle = null_ptr[NoneType, MutUntrackedOrigin]()
            return False

        self.connections = alloc[TCPConnection](65536)
        unsafe_memset(self.connections.unsafe_bitcast[UInt8](), 0, 65536 * 16)

        self.rx_addrs = alloc[UInt64](XDP_BATCH_SIZE)
        self.rx_lens = alloc[UInt32](XDP_BATCH_SIZE)

        self.active = True
        print("XDP worker active: " + interface + " queue=" + String(queue_id) + " port=" + String(Int(port)))
        return True

    @always_inline
    def poll_rx(mut self) -> Int:
        """Poll the RX ring for received frames. Returns number of frames available.
        Frame addresses and lengths are stored in self.rx_addrs/rx_lens."""
        if not self.active:
            return 0
        return Int(external_call["pion_xdp_poll_rx", Int32](
            self.handle, self.rx_addrs, self.rx_lens, Int32(XDP_BATCH_SIZE)
        ))

    @always_inline
    def frame_ptr(self, addr: UInt64) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Get a direct pointer into UMEM for the given frame address. Zero-copy."""
        return self.umem_base.unsafe_offset(Int(addr))

    @always_inline
    def release_rx_frame(self, addr: UInt64):
        """Return a received frame back to the free pool after processing."""
        external_call["pion_xdp_rx_release", NoneType](self.handle, addr)

    @always_inline
    def extract_tcp(self, frame: Pointer[UInt8, MutUntrackedOrigin], frame_len: Int) -> XDPFrameInfo:
        """Parse TCP header fields from a raw Ethernet frame. Zero-copy: payload_ptr
        points directly into the UMEM frame data."""
        var info = XDPFrameInfo()
        var payload_ptr = null_ptr[UInt8, MutUntrackedOrigin]()
        var seq = UInt32(0)
        var ack = UInt32(0)
        var flags = UInt8(0)
        var sport = UInt16(0)
        _ = payload_ptr; _ = seq; _ = ack; _ = flags; _ = sport

        # Stack variables for the out-params
        var pp = alloc[UInt64](1)  # pointer to payload pointer
        var sq = alloc[UInt32](1)
        var ak = alloc[UInt32](1)
        var fl = alloc[UInt8](1)
        var sp = alloc[UInt16](1)
        var dp = alloc[UInt16](1)

        var payload_len = external_call["pion_xdp_extract_tcp_payload", Int32](
            frame.unsafe_bitcast[NoneType](), Int32(frame_len),
            pp.unsafe_bitcast[NoneType](),
            sq.unsafe_bitcast[NoneType](), ak.unsafe_bitcast[NoneType](),
            fl.unsafe_bitcast[NoneType](),
            sp.unsafe_bitcast[NoneType](), dp.unsafe_bitcast[NoneType]()
        )

        if payload_len >= 0:
            info.payload_ptr = pp.unsafe_bitcast[Pointer[UInt8, MutUntrackedOrigin]]()[]
            info.payload_len = Int(payload_len)
            info.seq_num = sq[]
            info.ack_num = ak[]
            info.flags = fl[]
            info.sport = sp[]
            info.dport = dp[]

        pp.unsafe_free()
        sq.unsafe_free()
        ak.unsafe_free()
        fl.unsafe_free()
        sp.unsafe_free()
        dp.unsafe_free()
        return info

    @always_inline
    def alloc_tx_frame(self) -> UInt64:
        """Allocate a frame from the UMEM free pool for transmission.
        Returns UMEM address, or XDP_FRAME_INVALID if pool exhausted."""
        return external_call["pion_xdp_alloc_tx_frame", UInt64](self.handle)

    @always_inline
    def send_tcp_response(mut self, src_frame: Pointer[UInt8, MutUntrackedOrigin],
                          src_frame_len: Int, tcp_flags: UInt8,
                          seq_num: UInt32, ack_num: UInt32,
                          payload: Pointer[UInt8, MutUntrackedOrigin],
                          payload_len: Int) -> Bool:
        """Build and transmit a TCP response frame. Zero-copy: the response is built
        directly in a UMEM TX frame and submitted to the TX ring.
        Returns True on success."""
        var tx_addr = self.alloc_tx_frame()
        if tx_addr == XDP_FRAME_INVALID:
            return False

        var tx_buf = self.frame_ptr(tx_addr)
        var frame_len = external_call["pion_xdp_build_tcp_response", Int32](
            tx_buf.unsafe_bitcast[NoneType](), src_frame.unsafe_bitcast[NoneType](),
            Int32(src_frame_len), tcp_flags, seq_num, ack_num,
            payload.unsafe_bitcast[NoneType](), Int32(payload_len)
        )
        if frame_len <= 0:
            self.release_rx_frame(tx_addr)
            return False

        var rc = external_call["pion_xdp_submit_tx", Int32](
            self.handle, tx_addr, UInt32(frame_len)
        )
        return rc == 0

    @always_inline
    def tx_kick(self):
        """Notify the kernel to process pending TX submissions."""
        _ = external_call["pion_xdp_tx_kick", Int32](self.handle)

    @always_inline
    def drain_completion(self) -> Int:
        """Drain completed TX frames back to the free pool."""
        return Int(external_call["pion_xdp_drain_completion", Int32](self.handle))

    @always_inline
    def get_fd(self) -> Int32:
        """Get the AF_XDP socket fd for poll()/epoll() integration."""
        return external_call["pion_xdp_get_fd", Int32](self.handle)

    @always_inline
    def handle_syn(mut self, frame: Pointer[UInt8, MutUntrackedOrigin],
                   frame_len: Int, info: XDPFrameInfo) -> Bool:
        """Handle TCP SYN: transition to SYN_RECEIVED, send SYN+ACK."""
        var conn_idx = Int(info.sport)
        var conn = self.connections.unsafe_offset(conn_idx)
        conn[].state = 1  # SYN_RECEIVED
        conn[].our_seq = UInt32(0x12345678)  # ISN — simplified; production would use random
        conn[].their_seq = info.seq_num + 1  # expect their seq + 1 after SYN
        conn[].sport = info.sport
        conn[].dport = info.dport

        # Send SYN+ACK: our_seq, ack their seq+1
        return self.send_tcp_response(
            frame, frame_len, TCP_SYN_ACK,
            conn[].our_seq, conn[].their_seq,
            null_ptr[UInt8, MutUntrackedOrigin](), 0
        )

    @always_inline
    def handle_ack(mut self, info: XDPFrameInfo):
        """Handle TCP ACK: transition SYN_RECEIVED → ESTABLISHED."""
        var conn_idx = Int(info.sport)
        var conn = self.connections.unsafe_offset(conn_idx)
        if conn[].state == 1:  # SYN_RECEIVED → ESTABLISHED
            conn[].our_seq = info.ack_num  # they acked our SYN+ACK
            conn[].their_seq = info.seq_num
            conn[].state = 2  # ESTABLISHED
        elif conn[].state == 2:
            conn[].their_ack = info.ack_num

    @always_inline
    def handle_fin(mut self, frame: Pointer[UInt8, MutUntrackedOrigin],
                   frame_len: Int, info: XDPFrameInfo) -> Bool:
        """Handle TCP FIN: send FIN+ACK, close connection."""
        var conn_idx = Int(info.sport)
        var conn = self.connections.unsafe_offset(conn_idx)

        # Send FIN+ACK
        var ok = self.send_tcp_response(
            frame, frame_len, TCP_FIN_ACK,
            conn[].our_seq, info.seq_num + 1,
            null_ptr[UInt8, MutUntrackedOrigin](), 0
        )
        conn[].state = 0  # CLOSED
        return ok

    @always_inline
    def handle_rst(mut self, info: XDPFrameInfo):
        """Handle TCP RST: immediately close connection."""
        var conn_idx = Int(info.sport)
        self.connections[unsafe_offset=conn_idx].state = 0  # CLOSED

    @always_inline
    def send_data_ack(mut self, frame: Pointer[UInt8, MutUntrackedOrigin],
                      frame_len: Int, info: XDPFrameInfo,
                      response_data: Pointer[UInt8, MutUntrackedOrigin],
                      response_len: Int) -> Bool:
        """Send a data response with ACK, fragmenting across multiple TX frames if needed.
        Responses > 4042 bytes are split into multiple TCP segments with proper seq/ack.
        Updates connection sequence numbers."""
        var conn_idx = Int(info.sport)
        var conn = self.connections.unsafe_offset(conn_idx)

        # ACK their data
        var new_their_seq = info.seq_num + UInt32(info.payload_len)

        # Use fragmented send for all responses — handles both small (1 frame)
        # and large (multi-frame) responses correctly.
        var sent = Int(external_call["pion_xdp_send_fragmented", Int32](
            self.handle, frame.unsafe_bitcast[NoneType](), Int32(frame_len),
            TCP_ACK | TCP_PSH,
            conn[].our_seq, new_their_seq,
            response_data.unsafe_bitcast[NoneType](), Int32(response_len)
        ))
        if sent > 0:
            conn[].our_seq += UInt32(sent)
            conn[].their_seq = new_their_seq
            return True
        return False

    def destroy(mut self):
        """Clean up all resources. Detaches BPF from NIC if in single-worker mode."""
        if is_not_null(self.handle):
            if self.is_shared_mode:
                external_call["pion_xdp_destroy_worker", NoneType](self.handle)
            else:
                external_call["pion_xdp_destroy", NoneType](self.handle)
            self.handle = null_ptr[NoneType, MutUntrackedOrigin]()
        if is_not_null(self.connections):
            self.connections.unsafe_free()
            self.connections = null_ptr[TCPConnection, MutUntrackedOrigin]()
        if is_not_null(self.rx_addrs):
            self.rx_addrs.unsafe_free()
            self.rx_addrs = null_ptr[UInt64, MutUntrackedOrigin]()
        if is_not_null(self.rx_lens):
            self.rx_lens.unsafe_free()
            self.rx_lens = null_ptr[UInt32, MutUntrackedOrigin]()
        self.active = False
