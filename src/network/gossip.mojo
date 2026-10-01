"""GossipManager — SWIM-based health monitoring for cluster peers.

Uses UDP SWIM protocol (random target, indirect ping, suspicion/alive/confirm)
with TCP fallback. Background C pthread probes peers and exchanges state
(node_id, epoch, pfail bitmap). Health state (0=online, 1=pfail, 2=fail)
is written directly into ClusterState.peer_health via a pointer.
Worker 0 calls setup() + start(); all workers read ClusterState.peer_health.

SWIM port = main port + 20000 (UDP). Raft port = main port + 30000 (TCP).
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.network.cluster import ClusterState


struct GossipManager(Movable):
    """Wraps the C PionGossipBlock. Worker 0 owns this; handle=0 means not started."""
    var handle: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self):
        self.handle = null_ptr[NoneType, MutUntrackedOrigin]()

    def __moveinit__(out self, deinit take: Self):
        self.handle = take.handle

    def __del__(deinit self):
        if is_not_null(self.handle):
            external_call["pion_gossip_stop", NoneType](self.handle)

    def setup(
        mut self,
        cluster: Pointer[ClusterState, MutUntrackedOrigin],
        ping_ms: Int = 1000,
        pfail_threshold: Int = 5,
        fail_threshold: Int = 15,
    ):
        """Create gossip block and register all peers from ClusterState."""
        if is_null(cluster) or not cluster[].enabled or cluster[].peer_count == 0:
            return
        var blk = external_call["pion_gossip_create", Pointer[NoneType, MutUntrackedOrigin]]()
        if is_null(blk):
            return
        self.handle = blk
        # Set timeouts
        external_call["pion_gossip_set_timeouts", NoneType](
            blk, UInt64(ping_ms), Int32(pfail_threshold), Int32(fail_threshold)
        )
        # Register peers
        for pi in range(cluster[].peer_count):
            var host_ptr = cluster[].peer_host_ptr(pi)
            external_call["pion_gossip_set_peer", NoneType](
                blk, Int32(pi),
                host_ptr.unsafe_bitcast[Int8](),
                Int32(cluster[].peer_ports[pi]),
            )
        # Wire health output directly into ClusterState.peer_health
        var health_ptr = cluster[].peer_health.unsafe_ptr()
        external_call["pion_gossip_set_health_output", NoneType](blk, health_ptr)

    def start(mut self, use_swim: Bool = True) -> Bool:
        """Start the gossip background thread. Returns True on success.
        use_swim=True (default): UDP SWIM protocol with indirect ping + state exchange.
        use_swim=False: legacy TCP-only PING."""
        if is_null(self.handle):
            return False
        var rc: Int32
        if use_swim:
            rc = external_call["pion_gossip_start_swim", Int32](self.handle)
        else:
            rc = external_call["pion_gossip_start", Int32](self.handle)
        return rc == 0

    def add_peer(
        mut self,
        peer_idx: Int,
        host_ptr: Pointer[UInt8, MutUntrackedOrigin],
        port: Int,
    ):
        """Register a new peer (called by CLUSTER MEET). Thread-safe: gossip thread
        re-reads peer_count each round; safe to add before incrementing count."""
        if is_null(self.handle):
            return
        external_call["pion_gossip_set_peer", NoneType](
            self.handle, Int32(peer_idx),
            host_ptr.unsafe_bitcast[Int8](),
            Int32(port),
        )

    def set_identity(self, node_id: String, epoch: UInt64, role: UInt8, slot_count: UInt16):
        """C1.3: Set this node's identity for gossip state exchange."""
        if is_null(self.handle):
            return
        var nid = node_id + "\0"
        external_call["pion_gossip_set_identity", NoneType](
            self.handle, nid.unsafe_ptr().unsafe_bitcast[UInt8](), epoch, role, slot_count
        )

    def get_failover_target(self) -> Int:
        """C1.3: Check if a quorum-based FAIL was detected. Returns peer_idx or -1."""
        if is_null(self.handle):
            return -1
        return Int(external_call["pion_gossip_get_failover_target", Int32](self.handle))

    def clear_failover_target(self):
        """C1.3: Clear the failover target after handling."""
        if is_null(self.handle):
            return
        external_call["pion_gossip_clear_failover_target", NoneType](self.handle)

    def set_epoch(self, epoch: UInt64):
        """C1.3: Update epoch after topology change."""
        if is_null(self.handle):
            return
        external_call["pion_gossip_set_epoch", NoneType](self.handle, epoch)
