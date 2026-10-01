#!/usr/bin/env python3
"""Pion Serve CLI — launch the semantic-cached LLM inference engine.

Usage:
    # Dry-run (test cache layer without model/GPU):
    pion-serve

    # With model:
    pion-serve --model meta-llama/Llama-3.2-1B-Instruct

    # With Pion server for distributed cache:
    pion-serve --model meta-llama/Llama-3.2-1B-Instruct --pion-host 10.0.0.1

    # OpenAI embeddings (production):
    OPENAI_API_KEY=sk-... pion-serve --model ... --embed-provider openai
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .serve_engine import PionServeEngine, ServeConfig
from .semantic_cache import CacheConfig
from .fleet_manager import FleetConfig, WorkerConfig


class OpenAICompatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /v1/chat/completions endpoint."""

    engine: PionServeEngine = None  # set by serve()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_completion()
        elif self.path == "/v1/invalidate":
            self._handle_invalidate_webhook()
        elif self.path == "/v1/invalidate/files":
            self._handle_invalidate_files()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/stats":
            stats = self.engine.stats
            cache_stats = self.engine._cache_manager.stats
            result = {
                "requests": stats.total_requests,
                "cache_hit_rate": round(stats.cache_hit_rate, 4),
                "avg_ttft_ms": round(stats.avg_ttft_ms, 2),
                "avg_tokens_per_second": round(stats.avg_tokens_per_second, 1),
                "cache": {
                    "lookups": cache_stats.lookups,
                    "hits": cache_stats.hits,
                    "misses": cache_stats.misses,
                    "stores": cache_stats.stores,
                    "avg_lookup_ms": round(cache_stats.avg_lookup_ms, 2),
                },
            }
            if cache_stats.invalidation_events > 0:
                result["invalidation"] = {
                    "events": cache_stats.invalidation_events,
                    "entries_invalidated": cache_stats.entries_invalidated,
                    "stale_skips": cache_stats.stale_skips,
                }
            if self.engine.fleet:
                fs = self.engine.fleet.stats
                result["fleet"] = {
                    "total_routed": fs.total_routed,
                    "route_hit_rate": round(fs.hit_rate, 4),
                    "centroid_updates": fs.centroid_updates,
                    "local_routes": fs.local_routes,
                    "workers": {
                        wid: {
                            "endpoint": s.config.endpoint,
                            "active_requests": s.active_requests,
                            "total_routed": s.total_routed,
                            "healthy": s.healthy,
                        }
                        for wid, s in self.engine.fleet.workers.items()
                    },
                }
            self._send_json(result)
        elif self.path == "/v1/models":
            model_id = self.engine.config.model_id or "dry-run"
            self._send_json({
                "data": [{"id": model_id, "object": "model", "owned_by": "pion-serve"}]
            })
        else:
            self.send_error(404)

    def _handle_completion(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        messages = request.get("messages", [])
        if not messages:
            self.send_error(400, "No messages provided")
            return

        # Build prompt from messages
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        )

        max_tokens = request.get("max_tokens", self.engine.config.max_tokens)
        temperature = request.get("temperature", self.engine.config.temperature)

        result = self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = {
            "id": f"chatcmpl-{result.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.engine.config.model_id or "dry-run",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.tokens_generated,
                "total_tokens": result.prompt_tokens + result.tokens_generated,
            },
            "pion_serve": {
                "cache_hit": result.cache_hit,
                "cache_similarity": round(result.cache_similarity, 4),
                "ttft_ms": round(result.ttft_ms, 2),
                "prefill_skipped": result.prefill_skipped,
            },
        }

        self._send_json(response)

    def _handle_invalidate_webhook(self):
        """Handle GitHub/GitLab push webhook → invalidate affected cache entries."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        stale = self.engine._cache_manager.invalidate_from_webhook(payload)
        self._send_json({
            "invalidated": len(stale),
            "entries": [
                {"cache_id": e.cache_id, "reason": e.reason, "similarity": round(e.similarity, 4)}
                for e in stale
            ],
        })

    def _handle_invalidate_files(self):
        """Handle manual file invalidation: POST {"files": ["path/to/file.py"]}"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        files = payload.get("files", [])
        if not files:
            self.send_error(400, "No files provided")
            return

        stale = self.engine._cache_manager.invalidate_for_files(files, source="api")
        self._send_json({
            "invalidated": len(stale),
            "entries": [
                {"cache_id": e.cache_id, "reason": e.reason, "similarity": round(e.similarity, 4)}
                for e in stale
            ],
        })

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def serve(engine: PionServeEngine, host: str, port: int):
    """Start the HTTP server."""
    OpenAICompatHandler.engine = engine
    server = HTTPServer((host, port), OpenAICompatHandler)
    print(f"[pion-serve] Listening on {host}:{port}")
    print(f"[pion-serve] Endpoints:")
    print(f"  POST /v1/chat/completions  — OpenAI-compatible chat")
    print(f"  POST /v1/invalidate        — GitHub/GitLab push webhook")
    print(f"  POST /v1/invalidate/files  — Manual file invalidation")
    print(f"  GET  /v1/stats             — Cache + invalidation statistics")
    print(f"  GET  /v1/models            — Model info")
    print(f"  GET  /health               — Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pion-serve] Shutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Pion Serve — Semantic-Cached LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pion-serve                                     # dry-run mode
  pion-serve --model meta-llama/Llama-3.2-1B     # with model
  pion-serve --pion-host 10.0.0.1 --pion-port 1974  # remote Pion
  OPENAI_API_KEY=sk-... pion-serve --embed openai    # production embeddings
""",
    )

    # Model
    parser.add_argument("--model", type=str, default="",
                        help="HuggingFace model ID (empty = dry-run)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, mps, cpu, auto")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype: float16, bfloat16, float32")

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # Generation
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Cache
    parser.add_argument("--pion-host", type=str, default="127.0.0.1")
    parser.add_argument("--pion-port", type=int, default=1974)
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Cosine similarity threshold for cache hit")
    parser.add_argument("--embed", type=str, default="auto",
                        help="Embedding provider: ngram, openai, ollama, auto")
    parser.add_argument("--ttl", type=int, default=3600,
                        help="Cache entry TTL in seconds")
    parser.add_argument("--no-local-fallback", action="store_true",
                        help="Disable local cache fallback (require Pion)")

    # Git-aware invalidation
    parser.add_argument("--git-invalidation", action="store_true",
                        help="Enable git-aware cache invalidation")
    parser.add_argument("--git-invalidation-threshold", type=float, default=0.60,
                        help="Similarity threshold for file→cache invalidation (lower = more aggressive)")

    # Fleet routing
    parser.add_argument("--fleet", action="store_true",
                        help="Enable fleet routing (multi-GPU)")
    parser.add_argument("--workers", type=str, default="",
                        help="Comma-separated worker endpoints (e.g., 'http://gpu0:8080,http://gpu1:8080')")
    parser.add_argument("--routing-strategy", type=str, default="semantic_affinity",
                        help="Routing strategy: semantic_affinity, round_robin, least_loaded, random")

    args = parser.parse_args()

    fleet_config = None
    if args.fleet or args.workers:
        fleet_config = FleetConfig(
            pion_host=args.pion_host,
            pion_port=args.pion_port,
            use_pion=True,
            embed_provider=args.embed,
        )

    config = ServeConfig(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        host=args.host,
        port=args.port,
        cache=CacheConfig(
            pion_host=args.pion_host,
            pion_port=args.pion_port,
            cosine_threshold=args.threshold,
            embed_provider=args.embed,
            ttl=args.ttl,
            fallback_to_local=not args.no_local_fallback,
            git_invalidation=args.git_invalidation,
            git_invalidation_threshold=args.git_invalidation_threshold,
        ),
        fleet=fleet_config,
        routing_strategy=args.routing_strategy,
    )

    print("=" * 50)
    print("Pion Serve — Semantic-Cached LLM Inference")
    print("=" * 50)
    print(f"Model:     {args.model or '(dry-run)'}")
    print(f"Embedding: {args.embed}")
    print(f"Threshold: {args.threshold}")
    print(f"Pion:      {args.pion_host}:{args.pion_port}")
    print(f"Routing:   {args.routing_strategy}" + (" (fleet enabled)" if fleet_config else " (single-node)"))
    print(f"Server:    {args.host}:{args.port}")
    print()

    engine = PionServeEngine(config)
    engine.start()

    # Register workers if specified
    if args.workers and engine.fleet:
        for i, endpoint in enumerate(args.workers.split(",")):
            endpoint = endpoint.strip()
            if endpoint:
                wid = f"gpu-{i}"
                engine.fleet.add_worker(WorkerConfig(
                    worker_id=wid,
                    endpoint=endpoint,
                    tags={"description": f"GPU worker {i}"},
                ))
                print(f"  Registered worker {wid} → {endpoint}")

    engine = PionServeEngine(config)
    engine.start()

    try:
        serve(engine, args.host, args.port)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
