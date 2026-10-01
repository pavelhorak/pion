"""Routing policies for GPU fleet request distribution.

Strategies:
  - semantic_affinity: Route to worker with warmest KV cache (M13 HNSW)
  - round_robin: Simple rotation across healthy workers
  - least_loaded: Route to worker with fewest active requests
  - random: Random healthy worker selection

All strategies respect capacity limits and health status.
The semantic_affinity policy falls back to round_robin if routing fails.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .fleet_manager import FleetManager, WorkerState
from .router_client import RouteResult


class Strategy(Enum):
    SEMANTIC_AFFINITY = "semantic_affinity"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


@dataclass
class RoutingDecision:
    """Extended routing decision with policy metadata."""
    result: RouteResult
    strategy_used: str
    fallback: bool = False  # True if primary strategy failed and fallback was used
    scores: dict[str, float] = None  # worker_id -> score (for semantic/load)

    @property
    def hit(self) -> bool:
        return self.result.hit

    @property
    def endpoint(self) -> str:
        return self.result.endpoint

    @property
    def worker_id(self) -> str:
        return self.result.node_id


class RoutingPolicy:
    """Configurable routing policy wrapping FleetManager."""

    def __init__(
        self,
        fleet: FleetManager,
        strategy: Strategy | str = Strategy.SEMANTIC_AFFINITY,
        fallback_strategy: Strategy | str = Strategy.ROUND_ROBIN,
    ):
        self._fleet = fleet
        self._strategy = Strategy(strategy) if isinstance(strategy, str) else strategy
        self._fallback = Strategy(fallback_strategy) if isinstance(fallback_strategy, str) else fallback_strategy
        self._rr_index = 0  # round-robin counter
        self._rng = random.Random(42)

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    def route(
        self,
        prompt: str = "",
        embedding: Optional[np.ndarray] = None,
        exclude: str = "",
    ) -> RoutingDecision:
        """Route a request using the configured strategy.

        Either prompt or embedding must be provided. If both are given,
        embedding is preferred (avoids re-embedding).
        """
        if embedding is None and prompt:
            embedding = self._fleet._embedder.embed(prompt)

        # Primary strategy
        decision = self._apply_strategy(self._strategy, embedding, exclude)

        # Fallback if primary fails
        if not decision.hit and self._fallback != self._strategy:
            fallback_decision = self._apply_strategy(self._fallback, embedding, exclude)
            if fallback_decision.hit:
                fallback_decision.fallback = True
                return fallback_decision

        return decision

    def _apply_strategy(
        self,
        strategy: Strategy,
        embedding: Optional[np.ndarray],
        exclude: str,
    ) -> RoutingDecision:
        if strategy == Strategy.SEMANTIC_AFFINITY:
            return self._route_semantic(embedding, exclude)
        elif strategy == Strategy.ROUND_ROBIN:
            return self._route_round_robin(exclude)
        elif strategy == Strategy.LEAST_LOADED:
            return self._route_least_loaded(exclude)
        elif strategy == Strategy.RANDOM:
            return self._route_random(exclude)
        else:
            return RoutingDecision(
                result=RouteResult(hit=False),
                strategy_used=strategy.value,
            )

    def _route_semantic(self, embedding: Optional[np.ndarray], exclude: str) -> RoutingDecision:
        """Route by semantic affinity (KV cache warmth)."""
        if embedding is None:
            return RoutingDecision(result=RouteResult(hit=False), strategy_used="semantic_affinity")

        result = self._fleet.route_with_embedding(embedding, exclude=exclude)

        # Compute scores for all workers (for observability)
        scores = {}
        for wid, state in self._fleet.workers.items():
            if state.centroid is not None:
                scores[wid] = float(np.dot(embedding, state.centroid))

        return RoutingDecision(
            result=result,
            strategy_used="semantic_affinity",
            scores=scores,
        )

    def _route_round_robin(self, exclude: str) -> RoutingDecision:
        """Simple round-robin across healthy workers."""
        eligible = self._get_eligible(exclude)
        if not eligible:
            return RoutingDecision(result=RouteResult(hit=False), strategy_used="round_robin")

        idx = self._rr_index % len(eligible)
        self._rr_index += 1
        wid, state = eligible[idx]

        state.active_requests += 1
        state.total_routed += 1
        self._fleet._stats.total_routed += 1
        self._fleet._stats.route_hits += 1
        self._fleet._stats.local_routes += 1

        return RoutingDecision(
            result=RouteResult(hit=True, endpoint=state.config.endpoint, node_id=wid),
            strategy_used="round_robin",
        )

    def _route_least_loaded(self, exclude: str) -> RoutingDecision:
        """Route to worker with fewest active requests."""
        eligible = self._get_eligible(exclude)
        if not eligible:
            return RoutingDecision(result=RouteResult(hit=False), strategy_used="least_loaded")

        best_wid, best_state = min(eligible, key=lambda x: x[1].active_requests)
        best_state.active_requests += 1
        best_state.total_routed += 1
        self._fleet._stats.total_routed += 1
        self._fleet._stats.route_hits += 1
        self._fleet._stats.local_routes += 1

        scores = {wid: float(-state.active_requests) for wid, state in eligible}

        return RoutingDecision(
            result=RouteResult(hit=True, endpoint=best_state.config.endpoint, node_id=best_wid),
            strategy_used="least_loaded",
            scores=scores,
        )

    def _route_random(self, exclude: str) -> RoutingDecision:
        """Random healthy worker."""
        eligible = self._get_eligible(exclude)
        if not eligible:
            return RoutingDecision(result=RouteResult(hit=False), strategy_used="random")

        wid, state = self._rng.choice(eligible)
        state.active_requests += 1
        state.total_routed += 1
        self._fleet._stats.total_routed += 1
        self._fleet._stats.route_hits += 1
        self._fleet._stats.local_routes += 1

        return RoutingDecision(
            result=RouteResult(hit=True, endpoint=state.config.endpoint, node_id=wid),
            strategy_used="random",
        )

    def _get_eligible(self, exclude: str) -> list[tuple[str, WorkerState]]:
        """Get eligible workers (healthy, not excluded, not at capacity)."""
        eligible = []
        for wid, state in self._fleet.workers.items():
            if wid == exclude:
                continue
            if not state.healthy:
                continue
            if state.config.capacity > 0 and state.active_requests >= state.config.capacity:
                continue
            eligible.append((wid, state))
        return eligible
