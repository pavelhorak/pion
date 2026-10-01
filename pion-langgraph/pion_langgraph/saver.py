"""PionSaver — LangGraph BaseCheckpointSaver backed by Pion.

Storage layout (all via RESP2 commands Pion supports):
  - checkpoint:{thread_id}:{ns}:{checkpoint_id}  → HASH with checkpoint, metadata, parent_id, write_keys
  - checkpoint_versions:{thread_id}:{ns}          → SORTED SET (score = version int, member = checkpoint_id)
  - writes:{thread_id}:{ns}:{checkpoint_id}:{task_id}:{idx} → HASH with channel, value

Designed for Pion's supported command subset:
  - ZADD (single member only), ZREVRANGE (may return duplicates → deduped)
  - HSET/HGET/HGETALL, SET/GET/INCR, DEL
  - No ZSCORE, no ZREVRANGEBYSCORE, no SCAN pattern filtering
"""
from __future__ import annotations

import pickle
from typing import Any, Iterator, Optional, Sequence

import redis as redis_lib

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)


def _dedup_ordered(items: list[bytes]) -> list[bytes]:
    """Remove duplicates from a list while preserving order."""
    seen: set[bytes] = set()
    result: list[bytes] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class PionSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver using Pion as backend.

    Uses HSET/HGET/HGETALL, ZADD (single), ZREVRANGE — no RedisJSON needed.

    Args:
        host: Pion server host (default: 127.0.0.1)
        port: Pion server port (default: 1974)
        key_prefix: Namespace prefix for all keys (default: "lgcp")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1974,
        key_prefix: str = "lgcp",
    ):
        super().__init__()
        self._r = redis_lib.Redis(
            host=host, port=port,
            socket_timeout=30,
            decode_responses=False,
        )
        self._prefix = key_prefix

    # ── Key helpers ──────────────────────────────────────────────────────

    def _cp_key(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:cp:{thread_id}:{ns}:{checkpoint_id}"

    def _ver_key(self, thread_id: str, ns: str) -> str:
        return f"{self._prefix}:ver:{thread_id}:{ns}"

    def _wr_key(self, thread_id: str, ns: str, checkpoint_id: str,
                task_id: str, idx: int) -> str:
        return f"{self._prefix}:wr:{thread_id}:{ns}:{checkpoint_id}:{task_id}:{idx}"

    # ── BaseCheckpointSaver interface ────────────────────────────────────

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict[str, Any]:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        key = self._cp_key(thread_id, ns, checkpoint_id)
        self._r.hset(key, mapping={
            b"checkpoint": pickle.dumps(checkpoint),
            b"metadata": pickle.dumps(metadata),
            b"parent_id": (parent_id or "").encode(),
            b"new_versions": pickle.dumps(new_versions),
            b"write_keys": b"",  # will be updated by put_writes
        })

        # version index — monotonic counter as score (single-member ZADD)
        ver_key = self._ver_key(thread_id, ns)
        count = self._r.zcard(ver_key)
        self._r.zadd(ver_key, {checkpoint_id.encode(): float(count + 1)})

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        write_key_list: list[str] = []
        for idx, (channel, value) in enumerate(writes):
            wkey = self._wr_key(thread_id, ns, checkpoint_id, task_id, idx)
            self._r.hset(wkey, mapping={
                b"channel": channel.encode(),
                b"value": pickle.dumps(value),
                b"task_id": task_id.encode(),
            })
            write_key_list.append(wkey)

        # Store write keys in checkpoint hash for retrieval (avoids SCAN)
        cp_key = self._cp_key(thread_id, ns, checkpoint_id)
        existing_raw = self._r.hget(cp_key, "write_keys")
        existing = existing_raw.decode() if existing_raw else ""
        all_keys = existing + ("," if existing else "") + ",".join(write_key_list)
        self._r.hset(cp_key, b"write_keys", all_keys.encode())

    def get_tuple(self, config: dict[str, Any]) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            # get latest — ZREVRANGE returns newest first; dedup Pion duplicates
            ver_key = self._ver_key(thread_id, ns)
            raw_members = self._r.zrevrange(ver_key, 0, 0)
            members = _dedup_ordered(raw_members)
            if not members:
                return None
            checkpoint_id = members[0].decode() if isinstance(members[0], bytes) else members[0]

        key = self._cp_key(thread_id, ns, checkpoint_id)
        raw = self._r.hgetall(key)
        if not raw:
            return None

        checkpoint = pickle.loads(raw[b"checkpoint"])
        metadata = pickle.loads(raw[b"metadata"])
        parent_id_raw = raw.get(b"parent_id", b"")
        parent_id = parent_id_raw.decode() if parent_id_raw else None

        parent_config = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": parent_id,
                }
            }

        # collect pending writes from stored key list
        pending_writes = self._get_writes(raw)

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[dict[str, Any]],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")

        ver_key = self._ver_key(thread_id, ns)
        # Get all members newest-first, dedup
        raw_members = self._r.zrevrange(ver_key, 0, -1)
        members = _dedup_ordered(raw_members)

        # Apply 'before' filter — find position and skip
        if before:
            before_id = before["configurable"]["checkpoint_id"].encode()
            try:
                idx = members.index(before_id)
                members = members[idx + 1:]
            except ValueError:
                return

        count = 0
        for member in members:
            cp_id = member.decode() if isinstance(member, bytes) else member
            tup = self.get_tuple({
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            })
            if tup is None:
                continue

            if filter:
                meta = tup.metadata or {}
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue

            yield tup
            count += 1
            if limit and count >= limit:
                break

    # ── Writes collection ────────────────────────────────────────────────

    def _get_writes(self, cp_hash: dict[bytes, bytes]) -> list[tuple[str, str, Any]]:
        """Collect pending writes from the write_keys stored in the checkpoint hash."""
        write_keys_raw = cp_hash.get(b"write_keys", b"")
        if not write_keys_raw:
            return []

        write_keys_str = write_keys_raw.decode()
        if not write_keys_str:
            return []

        writes: list[tuple[str, str, Any]] = []
        for wkey in write_keys_str.split(","):
            wkey = wkey.strip()
            if not wkey:
                continue
            raw = self._r.hgetall(wkey)
            if not raw:
                continue
            channel = raw[b"channel"].decode()
            value = pickle.loads(raw[b"value"])
            task_id = raw.get(b"task_id", b"").decode()
            writes.append((task_id, channel, value))

        return writes

    # ── Cleanup ──────────────────────────────────────────────────────────

    def delete_thread(self, thread_id: str, ns: str = "") -> None:
        """Delete all checkpoints and writes for a thread."""
        ver_key = self._ver_key(thread_id, ns)
        raw_members = self._r.zrange(ver_key, 0, -1)
        members = _dedup_ordered(raw_members)

        for member in members:
            cp_id = member.decode() if isinstance(member, bytes) else member
            cp_key = self._cp_key(thread_id, ns, cp_id)

            # delete write keys stored in checkpoint
            write_keys_raw = self._r.hget(cp_key, "write_keys")
            if write_keys_raw:
                for wkey in write_keys_raw.decode().split(","):
                    wkey = wkey.strip()
                    if wkey:
                        self._r.delete(wkey)

            self._r.delete(cp_key)

        self._r.delete(ver_key)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass

    def __enter__(self) -> "PionSaver":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
