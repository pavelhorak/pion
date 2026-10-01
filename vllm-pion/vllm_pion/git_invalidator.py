"""GitCacheInvalidator — git-aware KV cache invalidation.

Detects file changes from git events (push, commit, webhook) and marks
affected cache entries as stale. Stale entries are skipped during lookup,
preventing the model from generating completions based on outdated code.

Invalidation strategy:
  1. Receive changed file paths (from webhook, git diff, or manual API)
  2. Embed each changed file path + content snippet
  3. Find cached prompts whose embeddings are close to the changed content
  4. Mark those entries as stale (soft delete — evict on next access)

This is conservative by design: it's better to invalidate too many entries
(causing cache misses → full prefill) than to serve stale completions.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .prompt_embedder import Embedder, create_embedder


@dataclass
class InvalidationEvent:
    """A single invalidation event."""
    timestamp: float
    source: str  # "webhook", "git_diff", "manual", "polling"
    files_changed: list[str]
    entries_invalidated: int
    duration_ms: float


@dataclass
class InvalidationStats:
    """Cumulative invalidation statistics."""
    total_events: int = 0
    total_files_processed: int = 0
    total_entries_invalidated: int = 0
    total_false_invalidations: int = 0  # entries that were stale but not actually affected
    events: list[InvalidationEvent] = field(default_factory=list)

    @property
    def avg_entries_per_event(self) -> float:
        return self.total_entries_invalidated / self.total_events if self.total_events > 0 else 0.0


@dataclass
class StaleCacheEntry:
    """Tracks a stale cache entry."""
    cache_id: str
    prompt_preview: str
    stale_since: float
    reason: str  # file path that caused invalidation
    similarity: float  # how close the changed file was to this cache entry


class GitCacheInvalidator:
    """Git-aware cache invalidation engine.

    Works in two modes:
    1. Push mode: receives file change events via webhook or API call
    2. Poll mode: periodically runs `git diff` to detect changes
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        embed_provider: str = "ngram",
        embed_dim: int = 1536,
        similarity_threshold: float = 0.60,
        repo_root: str = "",
    ):
        """
        Args:
            embedder: Pre-configured embedder (shared with cache manager).
            embed_provider: Fallback if embedder not provided.
            embed_dim: Embedding dimension.
            similarity_threshold: How close a changed file must be to a cached
                prompt to trigger invalidation. Lower = more aggressive.
            repo_root: Git repository root (for polling mode).
        """
        self._embedder = embedder or create_embedder(embed_provider, dim=embed_dim)
        self._threshold = similarity_threshold
        self._repo_root = repo_root or os.getcwd()
        self._stats = InvalidationStats()

        # Cache entry tracking: cache_id -> (embedding, prompt_preview, stored_at)
        self._tracked_entries: dict[str, tuple[np.ndarray, str, float]] = {}
        # Stale set: cache_ids marked as stale
        self._stale_ids: set[str] = set()

        # Last known git HEAD for polling
        self._last_head: str = ""

    @property
    def stats(self) -> InvalidationStats:
        return self._stats

    @property
    def stale_ids(self) -> set[str]:
        return self._stale_ids

    def track_entry(self, cache_id: str, prompt_embedding: np.ndarray, prompt_preview: str = ""):
        """Register a cache entry for invalidation tracking."""
        self._tracked_entries[cache_id] = (prompt_embedding, prompt_preview[:100], time.time())

    def untrack_entry(self, cache_id: str):
        """Remove a cache entry from tracking."""
        self._tracked_entries.pop(cache_id, None)
        self._stale_ids.discard(cache_id)

    def is_stale(self, cache_id: str) -> bool:
        """Check if a cache entry is marked stale."""
        return cache_id in self._stale_ids

    def invalidate_for_files(
        self,
        changed_files: list[str],
        source: str = "manual",
    ) -> list[StaleCacheEntry]:
        """Invalidate cache entries affected by changed files.

        Args:
            changed_files: List of file paths (relative to repo root).
            source: Event source for tracking.

        Returns:
            List of newly stale cache entries.
        """
        t0 = time.perf_counter()
        self._stats.total_events += 1
        self._stats.total_files_processed += len(changed_files)

        if not changed_files or not self._tracked_entries:
            event = InvalidationEvent(
                timestamp=time.time(), source=source,
                files_changed=changed_files, entries_invalidated=0,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
            self._stats.events.append(event)
            return []

        # Embed changed files (path + content snippet for context)
        file_embeddings = []
        for fpath in changed_files:
            text = self._build_file_text(fpath)
            emb = self._embedder.embed(text)
            file_embeddings.append((fpath, emb))

        # Find affected cache entries
        newly_stale = []
        for cache_id, (prompt_emb, preview, _stored_at) in self._tracked_entries.items():
            if cache_id in self._stale_ids:
                continue  # already stale

            for fpath, file_emb in file_embeddings:
                sim = float(np.dot(prompt_emb, file_emb))
                if sim >= self._threshold:
                    self._stale_ids.add(cache_id)
                    newly_stale.append(StaleCacheEntry(
                        cache_id=cache_id,
                        prompt_preview=preview,
                        stale_since=time.time(),
                        reason=fpath,
                        similarity=sim,
                    ))
                    break  # one match is enough

        self._stats.total_entries_invalidated += len(newly_stale)

        event = InvalidationEvent(
            timestamp=time.time(), source=source,
            files_changed=changed_files, entries_invalidated=len(newly_stale),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        self._stats.events.append(event)

        return newly_stale

    def invalidate_from_webhook(self, payload: dict) -> list[StaleCacheEntry]:
        """Process a GitHub/GitLab push webhook payload.

        Supports:
          - GitHub push event: payload["commits"][*]["added"|"modified"|"removed"]
          - GitLab push event: payload["commits"][*]["added"|"modified"|"removed"]
          - Simple format: payload["files"] (list of paths)
        """
        files = set()

        # Simple format
        if "files" in payload:
            files.update(payload["files"])

        # GitHub/GitLab push event
        for commit in payload.get("commits", []):
            files.update(commit.get("added", []))
            files.update(commit.get("modified", []))
            files.update(commit.get("removed", []))

        return self.invalidate_for_files(list(files), source="webhook")

    def poll_git_changes(self) -> list[StaleCacheEntry]:
        """Poll git for changes since last check.

        Uses `git diff --name-only` against the last known HEAD.
        """
        try:
            current_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repo_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        if not self._last_head:
            self._last_head = current_head
            return []

        if current_head == self._last_head:
            return []

        try:
            diff_output = subprocess.check_output(
                ["git", "diff", "--name-only", self._last_head, current_head],
                cwd=self._repo_root,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except subprocess.CalledProcessError:
            self._last_head = current_head
            return []

        self._last_head = current_head
        changed_files = [f for f in diff_output.split("\n") if f.strip()]

        if not changed_files:
            return []

        return self.invalidate_for_files(changed_files, source="polling")

    def get_stale_entries(self) -> list[StaleCacheEntry]:
        """Get all currently stale cache entries."""
        result = []
        for cache_id in self._stale_ids:
            if cache_id in self._tracked_entries:
                _emb, preview, _stored = self._tracked_entries[cache_id]
                result.append(StaleCacheEntry(
                    cache_id=cache_id,
                    prompt_preview=preview,
                    stale_since=0.0,
                    reason="(multiple or unknown)",
                    similarity=0.0,
                ))
        return result

    def clear_stale(self):
        """Clear all stale markers (e.g., after a full cache rebuild)."""
        self._stale_ids.clear()

    def _build_file_text(self, fpath: str) -> str:
        """Build embedding text for a changed file.

        Combines path components with a content snippet for better matching.
        Path components are repeated to boost their weight in n-gram embeddings.
        """
        parts = [fpath]

        # Add path components as semantic signal (repeated for weight)
        path_parts = fpath.replace("\\", "/").split("/")
        for p in path_parts:
            if p and not p.startswith("."):
                # Strip extension, split on underscores/dots for semantic tokens
                name = p.rsplit(".", 1)[0] if "." in p else p
                parts.append(name)
                parts.extend(name.replace("_", " ").replace("-", " ").split())

        # Repeat the filename for emphasis (most discriminative part)
        filename = path_parts[-1] if path_parts else ""
        parts.append(filename)
        parts.append(filename)

        # Try to read file content snippet
        full_path = os.path.join(self._repo_root, fpath)
        if os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(2000)  # first 2KB
                parts.append(content)
            except OSError:
                pass

        return " ".join(parts)
