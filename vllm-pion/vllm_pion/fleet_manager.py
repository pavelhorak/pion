"""FleetManager — multi-GPU orchestrator with semantic centroid tracking.

Manages a fleet of GPU inference workers:
  - Registers/removes workers with Pion's semantic router (M13)
  - Tracks per-worker centroid embeddings via exponential moving average
  - Routes incoming requests to the worker with warmest KV cache
  - Monitors worker health and capacity
  - Supports local-only mode (no Pion server) for development/testing

Architecture:
    Client request
         → FleetManager.route(prompt)
              → embed prompt
              → AI.ROUTE query (Pion) or local brute-force
              → returns best worker endpoint
         → forward to worker
         → on completion: FleetManager.report_completion(worker_id, prompt_embedding)
              → EMA centroid update
              → AI.ROUTE.UPDATE (Pion)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .prompt_embedder import Embedder, create_embedder
from .router_client import PionRouterClient, RouteResult, RouterInfo


@dataclass
class WorkerConfig:
    """Configuration for a GPU worker."""
    worker_id: str
    endpoint: str
    capacity: int = 0  # 0 = unlimited
    initial_centroid: Optional[np.ndarray] = None  # set from first request or domain description
    tags: dict = field(default_factory=dict)  # e.g., {"model": "llama-3.2", "gpu": "H100"}


@dataclass
class WorkerState:
    """Runtime state for a GPU worker."""
    config: WorkerConfig
    centroid: Optional[np.ndarray] = None  # current EMA centroid, FP32
    active_requests: int = 0
    total_routed: int = 0
    total_completed: int = 0
    registered: bool = False
    last_health_check: float = 0.0
    healthy: bool = True


@dataclass
class FleetConfig:
    """Configuration for FleetManager."""
    # Pion router connection
    pion_host: str = "127.0.0.1"
    pion_port: int = 1974
    use_pion: bool = True  # False = local-only mode

    # Embedding
    embed_provider: str = "auto"
    embed_dim: int = 768  # M13 default

    # Centroid tracking
    ema_alpha: float = 0.05  # centroid EMA decay: new = (1-alpha)*old + alpha*latest
    centroid_update_interval: int = 10  # update Pion every N completions

    # Health
    health_check_interval_s: float = 30.0
    unhealthy_after_s: float = 60.0  # mark unhealthy if no response for this long


@dataclass
class FleetStats:
    """Aggregate fleet statistics."""
    total_routed: int = 0
    total_completed: int = 0
    route_hits: int = 0
    route_misses: int = 0
    centroid_updates: int = 0
    pion_errors: int = 0
    local_routes: int = 0

    @property
    def hit_rate(self) -> float:
        return self.route_hits / self.total_routed if self.total_routed > 0 else 0.0


class FleetManager:
    """Multi-GPU fleet orchestrator with semantic routing."""

    def __init__(self, config: FleetConfig | None = None):
        self.config = config or FleetConfig()
        self._workers: dict[str, WorkerState] = {}
        self._embedder: Optional[Embedder] = None
        self._router: Optional[PionRouterClient] = None
        self._stats = FleetStats()
        self._pion_connected = False

    @property
    def stats(self) -> FleetStats:
        return self._stats

    @property
    def workers(self) -> dict[str, WorkerState]:
        return self._workers

    def start(self):
        """Initialize embedder and connect to Pion router."""
        self._embedder = create_embedder(
            self.config.embed_provider,
            dim=self.config.embed_dim,
        )

        if self.config.use_pion:
            try:
                self._router = PionRouterClient(
                    host=self.config.pion_host,
                    port=self.config.pion_port,
                )
                self._router.connect()
                self._pion_connected = True
            except Exception as e:
                print(f"[FleetManager] Pion router unavailable ({e}), using local routing")
                self._pion_connected = False
        else:
            self._pion_connected = False

    def stop(self):
        """Shut down fleet manager."""
        # Deregister all workers from Pion
        if self._pion_connected and self._router:
            for wid, state in self._workers.items():
                if state.registered:
                    try:
                        self._router.remove(wid)
                    except Exception:
                        pass
            self._router.close()
        self._workers.clear()

    def add_worker(self, config: WorkerConfig) -> bool:
        """Add a GPU worker to the fleet.

        If initial_centroid is None, the worker starts with a zero centroid
        and will be updated as requests are completed.
        """
        if config.worker_id in self._workers:
            return False

        centroid = config.initial_centroid
        if centroid is None:
            # Generate centroid from worker description if available
            desc = config.tags.get("description", "")
            if desc and self._embedder:
                centroid = self._embedder.embed(desc)
            else:
                centroid = np.zeros(self.config.embed_dim, dtype=np.float32)

        state = WorkerState(config=config, centroid=centroid)

        # Register with Pion
        if self._pion_connected:
            try:
                self._router.register(
                    node_id=config.worker_id,
                    endpoint=config.endpoint,
                    centroid=centroid,
                    capacity=config.capacity,
                )
                state.registered = True
            except Exception as e:
                self._stats.pion_errors += 1
                print(f"[FleetManager] Failed to register {config.worker_id}: {e}")

        self._workers[config.worker_id] = state
        return True

    def remove_worker(self, worker_id: str) -> bool:
        """Remove a GPU worker from the fleet."""
        if worker_id not in self._workers:
            return False

        state = self._workers[worker_id]
        if self._pion_connected and state.registered:
            try:
                self._router.remove(worker_id)
            except Exception:
                self._stats.pion_errors += 1

        del self._workers[worker_id]
        return True

    def route(
        self,
        prompt: str,
        exclude: str = "",
    ) -> RouteResult:
        """Route a request to the best GPU worker.

        Args:
            prompt: The prompt text (will be embedded).
            exclude: Worker ID to skip.

        Returns:
            RouteResult with endpoint of chosen worker.
        """
        self._stats.total_routed += 1

        embedding = self._embedder.embed(prompt)

        # Try Pion first
        if self._pion_connected:
            try:
                result = self._router.route(embedding, exclude=exclude)
                if result.hit:
                    self._stats.route_hits += 1
                    # Identify which worker was chosen
                    for wid, state in self._workers.items():
                        if state.config.endpoint == result.endpoint:
                            result.node_id = wid
                            state.active_requests += 1
                            break
                    return result
            except Exception:
                self._stats.pion_errors += 1

        # Local routing fallback
        return self._local_route(embedding, exclude)

    def route_with_embedding(
        self,
        embedding: np.ndarray,
        exclude: str = "",
    ) -> RouteResult:
        """Route using a pre-computed embedding (avoids re-embedding)."""
        self._stats.total_routed += 1

        if self._pion_connected:
            try:
                result = self._router.route(embedding, exclude=exclude)
                if result.hit:
                    self._stats.route_hits += 1
                    for wid, state in self._workers.items():
                        if state.config.endpoint == result.endpoint:
                            result.node_id = wid
                            state.active_requests += 1
                            break
                    return result
            except Exception:
                self._stats.pion_errors += 1

        return self._local_route(embedding, exclude)

    def report_completion(
        self,
        worker_id: str,
        prompt_embedding: Optional[np.ndarray] = None,
    ):
        """Report that a request completed on a worker.

        Updates the worker's centroid via EMA and optionally pushes to Pion.
        """
        state = self._workers.get(worker_id)
        if not state:
            return

        state.active_requests = max(0, state.active_requests - 1)
        state.total_completed += 1
        self._stats.total_completed += 1

        # Update centroid via EMA
        if prompt_embedding is not None and state.centroid is not None:
            alpha = self.config.ema_alpha
            state.centroid = (1 - alpha) * state.centroid + alpha * prompt_embedding
            # Re-normalize to unit length
            norm = np.linalg.norm(state.centroid)
            if norm > 0:
                state.centroid /= norm

            # Push updated centroid to Pion periodically
            if (state.total_completed % self.config.centroid_update_interval == 0
                    and self._pion_connected and state.registered):
                try:
                    self._router.update(worker_id, state.centroid)
                    self._stats.centroid_updates += 1
                except Exception:
                    self._stats.pion_errors += 1

    def get_info(self) -> Optional[RouterInfo]:
        """Get routing info from Pion server."""
        if self._pion_connected:
            try:
                return self._router.info()
            except Exception:
                return None
        return None

    def get_worker_centroids(self) -> dict[str, np.ndarray]:
        """Get current centroids for all workers."""
        return {
            wid: state.centroid.copy()
            for wid, state in self._workers.items()
            if state.centroid is not None
        }

    # ── Local routing ────────────────────────────────────────────────────

    def _local_route(self, embedding: np.ndarray, exclude: str) -> RouteResult:
        """Brute-force cosine routing over local worker centroids."""
        self._stats.local_routes += 1

        best_score = -1.0
        best_wid = ""
        best_endpoint = ""

        for wid, state in self._workers.items():
            if wid == exclude:
                continue
            if not state.healthy:
                continue
            if state.config.capacity > 0 and state.active_requests >= state.config.capacity:
                continue
            if state.centroid is None:
                continue

            score = float(np.dot(embedding, state.centroid))
            if score > best_score:
                best_score = score
                best_wid = wid
                best_endpoint = state.config.endpoint

        if best_wid:
            self._stats.route_hits += 1
            state = self._workers[best_wid]
            state.active_requests += 1
            state.total_routed += 1
            return RouteResult(hit=True, endpoint=best_endpoint, node_id=best_wid)

        self._stats.route_misses += 1
        return RouteResult(hit=False)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
