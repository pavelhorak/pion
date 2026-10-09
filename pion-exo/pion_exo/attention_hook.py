"""Pion attention hook for exo distributed inference.

Drop-in hook that offloads V storage to Pion and optionally runs attention
on Pion's native Metal SDPA engine. Integrates with exo's layer dispatch
to intercept prefill and decode attention.

Two modes:
    v_offload:      CPU attention routing + Pion V fetch (any network)
    gpu_attention:  Full attention on Pion's native Metal SDPA, mediated by
                    PionPromptCache(stage2=True) — best when Pion has cached K/V

The gpu_attention path is the high-level integration described in
gh #13: it routes ATTEND.PREFIX.STORE / ATTEND.PREFIX.QUERY through
`pion_vllm_mlx.PionPromptCache` so callers don't replicate the wire
protocol. Requires `pion-server --kvcache --metal-attention -w 1`.

Multi-host routing (gh #43):
    For Mac-cluster deployments (4× M4 Mini etc.), pass `peers=[(host, port), ...]`
    instead of a single host. Sessions are sticky-hashed across peers via
    blake2b(session_id) mod N, so that a given session's V-Store / ATTEND.PREFIX
    state always lands on the same node. Single-host construction
    (`PionAttentionHook(pion_host=...)`) keeps working unchanged.
"""

# Annotations stay strings: the class below defines a `redis` property, and on
# Python 3.10-3.13 an eagerly evaluated `-> redis.Redis` in a later method's
# signature resolves to that property and the import fails (3.14 defers them).
from __future__ import annotations

import hashlib
import socket
from typing import List, Optional, Tuple

import numpy as np
import redis


PeerAddr = Tuple[str, int]


class PionAttentionHook:
    """Attention hook for exo that offloads V to Pion.

    Single-host usage:
        hook = PionAttentionHook(pion_host="192.168.1.100", mode="v_offload")

    Multi-host (Mac cluster) usage:
        hook = PionAttentionHook(
            peers=[("mac-mini-1", 1974), ("mac-mini-2", 1974),
                   ("mac-mini-3", 1974), ("mac-mini-4", 1974)],
            mode="gpu_attention",
        )

    Per-session API is unchanged. The hook picks the owning peer by
    blake2b(session_id) mod len(peers) on first contact and caches the mapping.

        hook.on_prefill(session_id, layer_id, K, V)
        output = hook.on_decode_attention(session_id, layer_id, Q, K_local, top_k=32)
        hook.drop_session(session_id)

    For gpu_attention mode, start each Pion node with:
        ./pion-server --kvcache --metal-attention -w 1
    """

    def __init__(self, pion_host: str = "127.0.0.1", pion_port: int = 1974,
                 mode: str = "v_offload", v_format: str = "turbo4",
                 peers: Optional[List[PeerAddr]] = None,
                 lookup_existing: bool = False):
        """
        Args:
            pion_host / pion_port: single-peer fallback. Used only when
                `peers` is not supplied.
            mode: "v_offload" (CPU routing + Pion V fetch) or
                  "gpu_attention" (Stage-2 via PionPromptCache + Metal SDPA)
            v_format: V quantization format for V.CREATE in v_offload mode
                       (turbo4/turbo3/int8). Ignored in gpu_attention mode —
                       PionPromptCache controls the Stage-2 vquant.
            peers: optional list of (host, port) for multi-host routing. When
                   set, session_ids are sticky-hashed across peers (gh #43).
            lookup_existing: when True and len(peers) > 1, on first contact
                   for a session the hook probes `KV.PREFIX.OWNER` on every
                   peer and pins the session to the peer that already holds
                   it. Useful when sessions can outlive a hook instance and
                   the hash assignment may have changed (e.g. peer added).
                   Adds one round-trip per peer per cold session, so opt-in.
        """
        if mode not in {"v_offload", "gpu_attention"}:
            raise ValueError(f"unknown mode {mode!r}; expected v_offload or gpu_attention")

        if peers is not None:
            if not peers:
                raise ValueError("peers must be a non-empty list of (host, port)")
            self.peers: List[PeerAddr] = [(str(h), int(p)) for (h, p) in peers]
        else:
            self.peers = [(pion_host, int(pion_port))]

        self.mode = mode
        self.v_format = v_format
        self.lookup_existing = bool(lookup_existing)

        self._redis_by_peer: dict[PeerAddr, redis.Redis] = {}
        self._cache_by_peer: dict[PeerAddr, "object"] = {}
        self._sid_to_peer: dict[str, PeerAddr] = {}
        self._created_sessions: set = set()

    # --- Backward-compat single-peer accessors ---

    @property
    def pion_host(self) -> str:
        return self.peers[0][0]

    @property
    def pion_port(self) -> int:
        return self.peers[0][1]

    @property
    def redis(self) -> redis.Redis:
        """Single-peer accessor for back-compat. Returns the client for
        peers[0]. Multi-host callers should rely on the per-session methods
        instead — they route through `_peer_for_session(sid)`."""
        return self._redis_for(self.peers[0])

    @property
    def prompt_cache(self):
        """Single-peer accessor for back-compat. See `redis` above."""
        return self._cache_for(self.peers[0])

    # --- Routing ---

    @staticmethod
    def _sid_hash(sid: str) -> int:
        """Stable hash for session-to-peer assignment. Python's built-in
        hash() is randomised per-process, so sessions would land on
        different peers across restarts. blake2b is stable and cheap."""
        return int.from_bytes(
            hashlib.blake2b(sid.encode("utf-8"), digest_size=8).digest(),
            "big",
        )

    def peer_for_session(self, session_id: str) -> PeerAddr:
        """Return the (host, port) that owns this session_id.

        Resolution order:
            1. cached mapping from a prior call
            2. if lookup_existing and len(peers) > 1: KV.PREFIX.OWNER probe
               across peers; first hit wins
            3. fallback: blake2b(session_id) mod len(peers)
        """
        cached = self._sid_to_peer.get(session_id)
        if cached is not None:
            return cached

        if len(self.peers) == 1:
            peer = self.peers[0]
            self._sid_to_peer[session_id] = peer
            return peer

        if self.lookup_existing:
            for p in self.peers:
                try:
                    r = self._redis_for(p)
                    resp = r.execute_command("KV.PREFIX.OWNER", session_id)
                    owner = self._parse_owner_resp(resp)
                    if owner >= 0:
                        self._sid_to_peer[session_id] = p
                        return p
                except (redis.RedisError, ConnectionError):
                    continue

        peer = self.peers[self._sid_hash(session_id) % len(self.peers)]
        self._sid_to_peer[session_id] = peer
        return peer

    @staticmethod
    def _parse_owner_resp(resp) -> int:
        """KV.PREFIX.OWNER returns simple-string `+<owner>\r\n` or
        `+<owner> <schema_digest>\r\n`. redis-py decodes that as bytes
        like b"0" or b"0 12345"; sometimes returns int directly. -1 means
        not registered."""
        if resp is None:
            return -1
        if isinstance(resp, int):
            return resp
        if isinstance(resp, (bytes, bytearray)):
            try:
                return int(resp.split()[0])
            except (ValueError, IndexError):
                return -1
        if isinstance(resp, str):
            try:
                return int(resp.split()[0])
            except (ValueError, IndexError):
                return -1
        return -1

    # --- Per-peer client construction (lazy) ---

    def _redis_for(self, peer: PeerAddr) -> redis.Redis:
        r = self._redis_by_peer.get(peer)
        if r is None:
            r = redis.Redis(
                host=peer[0], port=peer[1],
                decode_responses=False,
                socket_timeout=30.0,
            )
            self._redis_by_peer[peer] = r
        return r

    def _cache_for(self, peer: PeerAddr):
        """Lazy-instantiated PionPromptCache(stage2=True) per peer.

        Constructed with model=None — the wire-only Stage-2 surface
        (attend_store_layer / attend_query / attend_drop / lookup) doesn't
        need an mlx-lm model, since exo computes K/V externally per layer.
        """
        pc = self._cache_by_peer.get(peer)
        if pc is None:
            try:
                from pion_vllm_mlx import PionPromptCache
            except ImportError as e:
                raise ImportError(
                    "gpu_attention mode requires `pion-vllm-mlx` "
                    "(install with `pip install -e pion-vllm-mlx/`)"
                ) from e
            pc = PionPromptCache(
                model=None, stage2=True,
                host=peer[0], port=peer[1],
            )
            self._cache_by_peer[peer] = pc
        return pc

    # --- Session management ---

    def _ensure_session(self, session_id: str, value_dim: int):
        """Create V-Store session on the owning peer (v_offload only)."""
        if session_id in self._created_sessions:
            return
        peer = self.peer_for_session(session_id)
        args = ["V.CREATE", session_id, str(value_dim)]
        if self.v_format != "int8":
            args.extend(["VQUANT", self.v_format])
        try:
            self._redis_for(peer).execute_command(*args)
        except redis.RedisError:
            pass  # May already exist
        self._created_sessions.add(session_id)

    # --- Prefill hook ---

    def on_prefill(self, session_id: str, layer_id: int,
                   K: np.ndarray, V: np.ndarray):
        """Called after computing K, V for a layer during prefill.

        v_offload mode: stores V in Pion V-Store (turbo4 compressed) on the
            session's owning peer. K stays local on the compute node for
            decode-time attention routing.

        gpu_attention mode: pushes both K and V to the owning peer's Metal-
            side memory via ATTEND.PREFIX.STORE (PionPromptCache.attend_store_layer).
            Future decode-time attend_query() calls run on that peer's Metal SDPA
            with Q-only on the wire.

        Args:
            session_id: unique session identifier (becomes the namespace)
            layer_id: transformer layer index
            K: [H, N, D] or [N, D] float32 key matrix
            V: [H, N, D] or [N, D] float32 value matrix
        """
        # Flatten heads if 2D → treat as single-head batch.
        if V.ndim == 2:
            N, D = V.shape
            H = 1
            K_3d = K.reshape(H, N, D).astype(np.float32, copy=False)
            V_3d = V.reshape(H, N, D).astype(np.float32, copy=False)
        else:
            H, N, D = V.shape
            K_3d = K.astype(np.float32, copy=False)
            V_3d = V.astype(np.float32, copy=False)

        peer = self.peer_for_session(session_id)

        if self.mode == "v_offload":
            self._ensure_session(session_id, D)
            V_flat = V_3d.reshape(-1)
            try:
                self._redis_for(peer).execute_command(
                    "V.STOREBATCH", session_id, str(layer_id),
                    "0", str(H * N), V_flat.tobytes()
                )
            except redis.RedisError as e:
                print(f"[pion-exo] V.STOREBATCH failed: {e}")
            return

        # gpu_attention: route through PionPromptCache → ATTEND.PREFIX.STORE.
        try:
            self._cache_for(peer).attend_store_layer(session_id, layer_id, K_3d, V_3d)
        except Exception as e:
            print(f"[pion-exo] ATTEND.PREFIX.STORE failed: {e}")

    # --- Decode hook ---

    def on_decode_attention(self, session_id: str, layer_id: int,
                            Q: np.ndarray, K_local: np.ndarray,
                            top_k: int = 32) -> np.ndarray:
        """Called during decode to compute attention for one token.

        Mode "v_offload":
            1. CPU: Q @ K_local^T → top-k token IDs
            2. Pion (owning peer): V.FETCH → dequantized V for top-k tokens
            3. CPU: softmax(scores) @ V_fetched → output

        Mode "gpu_attention":
            1. Send Q to the session's owning peer via ATTEND.PREFIX.QUERY
               (K+V already cached on that peer's Metal side via on_prefill)
            2. That peer runs full attention on Metal → returns output

        Args:
            session_id: session identifier (namespace passed to PionPromptCache)
            layer_id: transformer layer index
            Q: [H, D] or [H, 1, D] float32 query vectors
            K_local: [H, N, D] float32 local key matrix (unused in
                      gpu_attention; kept in the signature for API parity)
            top_k: number of top-scoring tokens to attend to. v_offload uses
                   it to size the V.FETCH; gpu_attention forwards it to
                   ATTEND.PREFIX.QUERY (0 = sidecar-default).

        Returns: [H, D] float32 attention output
        """
        if self.mode == "gpu_attention":
            return self._gpu_attention(session_id, layer_id, Q, top_k)
        return self._v_offload_attention(session_id, layer_id, Q, K_local, top_k)

    def _v_offload_attention(self, session_id: str, layer_id: int,
                             Q: np.ndarray, K_local: np.ndarray,
                             top_k: int) -> np.ndarray:
        """CPU attention routing + Pion V fetch from the owning peer."""
        if Q.ndim == 3:
            Q = Q[:, 0, :]  # [H, 1, D] → [H, D]
        H, D = Q.shape
        scale = D ** -0.5

        peer = self.peer_for_session(session_id)
        r = self._redis_for(peer)

        outputs = np.zeros((H, D), dtype=np.float32)
        for h in range(H):
            K_h = K_local[h] if K_local.ndim == 3 else K_local
            scores = (Q[h] @ K_h.T) * scale

            actual_k = min(top_k, scores.shape[0])
            topk_idx = np.argpartition(scores, -actual_k)[-actual_k:]
            topk_scores = scores[topk_idx]

            try:
                token_ids = [str(int(i)) for i in topk_idx]
                resp = r.execute_command(
                    "V.FETCH", session_id, str(layer_id), *token_ids
                )
                if resp and isinstance(resp, bytes):
                    V_fetched = np.frombuffer(resp, dtype=np.float32).reshape(actual_k, D)
                else:
                    continue
            except redis.RedisError:
                continue

            topk_scores -= topk_scores.max()
            weights = np.exp(topk_scores)
            weights /= weights.sum()
            outputs[h] = weights @ V_fetched

        return outputs

    def _gpu_attention(self, session_id: str, layer_id: int,
                       Q: np.ndarray, top_k: int) -> np.ndarray:
        """Stage-2 attention on the owning peer's Metal SDPA via PionPromptCache.

        Q shape:
            [H, D] or [H, 1, D]  → single decode-step → returns (H, D)
            [H, M, D]            → batched suffix     → returns (H, M, D)
        """
        if Q.ndim == 3 and Q.shape[1] == 1:
            Q_send = Q[:, 0, :]
        else:
            Q_send = Q
        peer = self.peer_for_session(session_id)
        return self._cache_for(peer).attend_query(
            session_id, layer_id, Q_send.astype(np.float32, copy=False), top_k=top_k
        )

    # --- Cleanup ---

    def drop_session(self, session_id: str):
        """Drop session from V-Store and Pion's Metal cache on the owning peer.

        v_offload: clears the local "created" cache (V-Store eviction is LRU).
        gpu_attention: also calls PionPromptCache.attend_drop(), which marks
            the namespace as no longer pushed so a future on_prefill will
            re-store. (Server-side eviction is LRU.)

        Either way the cached sid→peer mapping is dropped so a future
        on_prefill picks the peer fresh.
        """
        self._created_sessions.discard(session_id)
        peer = self._sid_to_peer.pop(session_id, None)
        if peer is None:
            return
        pc = self._cache_by_peer.get(peer)
        if pc is not None:
            try:
                pc.attend_drop(session_id)
            except Exception:
                pass

    def health_check(self) -> dict:
        """Probe Pion connectivity per peer.

        Single-peer mode (back-compat): returns
            {"pion_resp": bool, "metal_attention": bool}

        Multi-peer mode: returns
            {
              "peers": {"host:port": {"pion_resp": bool, "metal_attention": bool}, ...},
              "pion_resp": all_peers_responded,
              "metal_attention": all_peers_have_metal,
            }
        """
        per_peer = {}
        all_resp = True
        all_metal = True
        for peer in self.peers:
            r = self._probe_peer(peer)
            per_peer[f"{peer[0]}:{peer[1]}"] = r
            all_resp = all_resp and r["pion_resp"]
            all_metal = all_metal and r["metal_attention"]

        if len(self.peers) == 1:
            # Back-compat: top-level keys only.
            return per_peer[f"{self.peers[0][0]}:{self.peers[0][1]}"]
        return {
            "peers": per_peer,
            "pion_resp": all_resp,
            "metal_attention": all_metal,
        }

    def _probe_peer(self, peer: PeerAddr) -> dict:
        result = {"pion_resp": False, "metal_attention": False}
        try:
            self._redis_for(peer).ping()
            result["pion_resp"] = True
        except Exception:
            return result
        # Probe ATTEND.PREFIX.QUERY: send a well-formed query against a
        # sentinel session that doesn't exist. Server response patterns:
        #   -ERR ATTEND.PREFIX.QUERY failed (session not found ...)
        #     → command is registered, --metal-attention is on
        #   -ERR unknown command 'ATTEND.PREFIX.QUERY'
        #     → server compiled without the kvcache/metal-attention path
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(peer)
            H, D = 1, 128
            zero_q = np.zeros((H, D), dtype=np.float32).tobytes()
            sid = b"__pion_exo_health_probe"
            parts = [b"ATTEND.PREFIX.QUERY", sid, b"0",
                     str(H).encode(), str(D).encode(), b"1", zero_q]
            req = f"*{len(parts)}\r\n".encode()
            for p in parts:
                req += f"${len(p)}\r\n".encode() + p + b"\r\n"
            sock.sendall(req)
            buf = b""
            while b"\r\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            sock.close()
            if buf.startswith(b"-ERR"):
                line = buf.split(b"\r\n", 1)[0].lower()
                # Any -ERR that doesn't say "unknown command" means the
                # handler exists (likely "session not found").
                result["metal_attention"] = b"unknown command" not in line
            else:
                # +OK / $... → command produced a real response → exists.
                result["metal_attention"] = True
        except Exception:
            pass
        return result

    def close(self):
        """Close all per-peer connections."""
        for r in self._redis_by_peer.values():
            try:
                r.close()
            except Exception:
                pass
        self._redis_by_peer.clear()
        for pc in self._cache_by_peer.values():
            try:
                pc.resp.sock.close()
            except Exception:
                pass
        self._cache_by_peer.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
