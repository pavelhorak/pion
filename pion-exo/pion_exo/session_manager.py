"""Session lifecycle management for Pion attention offloading."""

import hashlib
from typing import TYPE_CHECKING, List, Optional, Tuple

import redis

if TYPE_CHECKING:  # gh #209
    from .prefix_router import PrefixAwareRouter


PeerAddr = Tuple[str, int]


class PionSessionManager:
    """Manages V-Store session lifecycle in Pion.

    Used by exo's v_offload mode to create/drop V.STOREBATCH sessions. The
    Stage-2 (gpu_attention) path manages its own session state through
    PionPromptCache — no separate session manager needed.

    Multi-host (gh #43): pass `peers=[(host, port), ...]` to spread session
    creation across nodes. Sessions are sticky-hashed by blake2b(session_id)
    mod len(peers), matching `PionAttentionHook.peer_for_session`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1974,
                 peers: Optional[List[PeerAddr]] = None,
                 router: Optional["PrefixAwareRouter"] = None):
        if peers is not None:
            if not peers:
                raise ValueError("peers must be a non-empty list of (host, port)")
            self.peers: List[PeerAddr] = [(str(h), int(p)) for (h, p) in peers]
        else:
            self.peers = [(host, int(port))]
        self._redis_by_peer: dict[PeerAddr, redis.Redis] = {}
        # gh #209: optional prefix-aware router. When set, peer_for_prompt()
        # routes by KV.PREFIX.MEMBERSHIP block coverage instead of the sticky
        # hash; peer_for_session() keeps the sticky behavior for back-compat.
        self.router = router

    # --- Back-compat single-peer accessors ---

    @property
    def host(self) -> str:
        return self.peers[0][0]

    @property
    def port(self) -> int:
        return self.peers[0][1]

    @property
    def redis(self) -> redis.Redis:
        return self._redis_for(self.peers[0])

    # --- Routing ---

    @staticmethod
    def _sid_hash(sid: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(sid.encode("utf-8"), digest_size=8).digest(),
            "big",
        )

    def peer_for_session(self, session_id: str) -> PeerAddr:
        return self.peers[self._sid_hash(session_id) % len(self.peers)]

    def peer_for_prompt(self, session_id: str,
                        block_hashes: Optional[List[int]] = None) -> PeerAddr:
        """gh #209: cache-aware peer choice. With a router configured and the
        prompt's block hashes supplied (see prefix_router.blocks_of), routes to
        the peer holding the longest leading run of matching prefix blocks;
        otherwise identical to peer_for_session."""
        if self.router is not None:
            return self.router.route(session_id, block_hashes)
        return self.peer_for_session(session_id)

    def _redis_for(self, peer: PeerAddr) -> redis.Redis:
        r = self._redis_by_peer.get(peer)
        if r is None:
            r = redis.Redis(host=peer[0], port=peer[1], decode_responses=False)
            self._redis_by_peer[peer] = r
        return r

    # --- API ---

    def create_session(self, session_id: str, value_dim: int = 128,
                       v_format: str = "turbo4") -> bool:
        """Create a V-Store session for V offloading on the owning peer.

        Args:
            session_id: unique session identifier
            value_dim: dimension of V vectors (typically head_dim)
            v_format: quantization format (turbo4/turbo3/turbo2/int8/fp16)
        """
        peer = self.peer_for_session(session_id)
        try:
            args = ["V.CREATE", session_id, str(value_dim)]
            if v_format != "int8":
                args.extend(["VQUANT", v_format])
            self._redis_for(peer).execute_command(*args)
            return True
        except redis.RedisError as e:
            print(f"[pion-exo] Session create failed on {peer[0]}:{peer[1]}: {e}")
            return False

    def drop_session(self, session_id: str) -> bool:
        """Drop hint — V-Store sessions are LRU-evicted server-side.

        Stage-2 (ATTEND.PREFIX.*) sessions are managed via
        `PionAttentionHook.drop_session` → `PionPromptCache.attend_drop`,
        which marks the namespace as locally re-pushable; server-side
        eviction is LRU.
        """
        return True

    def close(self):
        for r in self._redis_by_peer.values():
            try:
                r.close()
            except Exception:
                pass
        self._redis_by_peer.clear()
