"""Raft consensus — leader election and metadata replication for cluster topology.

NOT for data replication (WAL handles that). Only for:
- Leader election (prevents split-brain during network partitions)
- Topology change commits (slot migration, failover decisions)

Uses a C pthread for the Raft state machine (election timeout, heartbeat,
RequestVote, AppendEntries). Mojo polls state each event-loop tick.

Raft port = main port + 30000 (e.g., 31974 for port 1974).
"""

from src.common.ptr import is_not_null, is_null, null_ptr
from std.collections import List
from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer

comptime CMD_ID_SET   = 1
comptime CMD_ID_HSET  = 2
comptime CMD_ID_LPUSH = 3

comptime RAFT_FOLLOWER  = 0
comptime RAFT_CANDIDATE = 1
comptime RAFT_LEADER    = 2

@fieldwise_init
struct RaftLogEntry(Copyable, Movable, ImplicitlyCopyable):
    var term: Int
    var command_id: UInt8
    var key_offset: Int
    var key_len: UInt16

@fieldwise_init
struct RaftNode(Movable):
    var current_term: Int
    var voted_for: String
    var log: List[RaftLogEntry]
    var commit_index: Int
    var last_applied: Int
    var state: Int # 0: Follower, 1: Candidate, 2: Leader
    var leader_id: String
    var node_id: String
    # C handle for the Raft background thread
    var handle: Pointer[NoneType, MutUntrackedOrigin]

    def __init__(out self, node_id: String):
        self.current_term = 0
        self.voted_for = ""
        self.log = List[RaftLogEntry]()
        self.commit_index = 0
        self.last_applied = 0
        self.state = 0
        self.leader_id = ""
        self.node_id = node_id
        self.handle = null_ptr[NoneType, MutUntrackedOrigin]()

    def __moveinit__(out self, deinit take: Self):
        self.current_term = take.current_term
        self.voted_for = take.voted_for^
        self.log = take.log^
        self.commit_index = take.commit_index
        self.last_applied = take.last_applied
        self.state = take.state
        self.leader_id = take.leader_id^
        self.node_id = take.node_id^
        self.handle = take.handle

    def __del__(deinit self):
        if is_not_null(self.handle):
            external_call["pion_raft_stop", NoneType](self.handle)

    def setup(mut self, raft_port: Int) -> Bool:
        """Create Raft block and set node identity. Call before start()."""
        var nid = self.node_id + "\0"
        var blk = external_call["pion_raft_create",
                                 Pointer[NoneType, MutUntrackedOrigin]](
            nid.unsafe_ptr().unsafe_bitcast[UInt8](),
            Int32(raft_port),
        )
        if not blk:
            return False
        self.handle = blk
        return True

    def add_peer(self, host: String, raft_port: Int, peer_idx: Int):
        """Register a Raft peer."""
        if is_null(self.handle):
            return
        var h = host + "\0"
        external_call["pion_raft_set_peer", NoneType](
            self.handle, Int32(peer_idx),
            h.unsafe_ptr().unsafe_bitcast[UInt8](),
            Int32(raft_port),
        )

    def start(mut self) -> Bool:
        """Start the Raft background thread."""
        if is_null(self.handle):
            return False
        var rc = external_call["pion_raft_start", Int32](self.handle)
        return rc == 0

    def is_leader(self) -> Bool:
        """Check if this node is the Raft leader."""
        if is_null(self.handle):
            return False
        return external_call["pion_raft_is_leader", Int32](self.handle) == 1

    def get_state(self) -> Int:
        """Return Raft state: 0=Follower, 1=Candidate, 2=Leader."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_raft_get_state", Int32](self.handle))

    def get_term(self) -> Int:
        """Return current Raft term."""
        if is_null(self.handle):
            return 0
        return Int(external_call["pion_raft_get_term", Int64](self.handle))

    def request_vote(mut self, term: Int, candidate_id: String, last_log_index: Int, last_log_term: Int) -> Bool:
        if term < self.current_term:
            return False
        if term > self.current_term:
            self.current_term = term
            self.state = 0
            self.voted_for = ""
        if (self.voted_for == "" or self.voted_for == candidate_id):
            var my_last_log_term = 0
            var my_last_log_index = len(self.log) - 1
            if my_last_log_index >= 0:
                my_last_log_term = self.log[my_last_log_index].term
            if last_log_term > my_last_log_term or (last_log_term == my_last_log_term and last_log_index >= my_last_log_index):
                self.voted_for = candidate_id
                return True
        return False

    def append_entries(mut self, term: Int, leader_id: String, prev_log_index: Int, prev_log_term: Int, entries: List[RaftLogEntry], leader_commit: Int) -> Bool:
        if term < self.current_term:
            return False
        if term > self.current_term:
            self.current_term = term
            self.state = 0
            self.voted_for = ""
        self.leader_id = leader_id
        if prev_log_index >= 0:
            if prev_log_index >= len(self.log):
                return False
            if self.log[prev_log_index].term != prev_log_term:
                while len(self.log) > prev_log_index:
                    _ = self.log.pop()
                return False
        for i in range(len(entries)):
            self.log.append(entries[i])
        if leader_commit > self.commit_index:
            var last_new_index = len(self.log) - 1
            self.commit_index = leader_commit if leader_commit < last_new_index else last_new_index
        return True
