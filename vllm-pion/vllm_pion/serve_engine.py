"""PionServeEngine — LLM inference engine with semantic KV cache.

Integrates with HuggingFace transformers for model loading and generation,
using SemanticCacheManager for prompt-level KV cache reuse.

The engine:
  1. Receives a prompt (text or chat messages)
  2. Embeds the prompt and checks semantic cache
  3. On cache hit: deserializes KV tensors, applies RoPE re-rotation, injects into model
  4. On cache miss: runs full prefill, stores KV cache for future reuse
  5. Runs autoregressive decode and streams tokens

Supports both HuggingFace transformers (PyTorch) and dry-run mode (no GPU).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np

from .semantic_cache import SemanticCacheManager, CacheConfig, CacheLookupResult
from .kv_serializer import serialize_kv_cache, deserialize_kv_cache
from .fleet_manager import FleetManager, FleetConfig, WorkerConfig
from .routing_policy import RoutingPolicy, Strategy


@dataclass
class ServeConfig:
    """Configuration for PionServeEngine."""
    # Model
    model_id: str = ""
    device: str = "auto"  # "cuda", "mps", "cpu", "auto"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # Generation
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Cache
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Fleet routing
    fleet: Optional[FleetConfig] = None  # None = no fleet routing (single-node)
    routing_strategy: str = "semantic_affinity"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class GenerationResult:
    """Result of a single generation request."""
    request_id: str
    text: str
    tokens_generated: int
    prompt_tokens: int
    cache_hit: bool
    cache_similarity: float
    ttft_ms: float  # time to first token
    total_ms: float
    tokens_per_second: float
    routed_to: str = ""  # worker_id if fleet routing was used
    routing_strategy: str = ""

    @property
    def prefill_skipped(self) -> bool:
        return self.cache_hit


@dataclass
class EngineStats:
    """Cumulative engine statistics."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_ttft_ms: float = 0.0
    total_generation_ms: float = 0.0
    prefill_flops_saved: int = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        return self.total_ttft_ms / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_generation_ms <= 0:
            return 0.0
        return self.total_tokens_generated / (self.total_generation_ms / 1000)


class PionServeEngine:
    """LLM inference engine with semantic KV cache acceleration."""

    def __init__(self, config: ServeConfig):
        self.config = config
        self._cache_manager: Optional[SemanticCacheManager] = None
        self._fleet: Optional[FleetManager] = None
        self._routing_policy: Optional[RoutingPolicy] = None
        self._model = None
        self._tokenizer = None
        self._stats = EngineStats()
        self._dry_run = not config.model_id  # no model = dry-run mode

    @property
    def stats(self) -> EngineStats:
        return self._stats

    @property
    def fleet(self) -> Optional[FleetManager]:
        return self._fleet

    @property
    def routing_policy(self) -> Optional[RoutingPolicy]:
        return self._routing_policy

    def start(self):
        """Initialize model, cache manager, and optional fleet routing."""
        # Initialize cache manager
        self._cache_manager = SemanticCacheManager(self.config.cache)
        self._cache_manager.connect()

        # Initialize fleet routing if configured
        if self.config.fleet is not None:
            self._fleet = FleetManager(self.config.fleet)
            self._fleet.start()
            self._routing_policy = RoutingPolicy(
                self._fleet,
                strategy=self.config.routing_strategy,
            )
            print(f"[PionServeEngine] Fleet routing enabled (strategy={self.config.routing_strategy})")

        if self._dry_run:
            print("[PionServeEngine] Dry-run mode (no model loaded)")
            return

        # Load model via transformers
        self._load_model()

    def stop(self):
        """Shut down engine."""
        if self._fleet:
            self._fleet.stop()
        if self._cache_manager:
            self._cache_manager.close()

    def _load_model(self):
        """Load HuggingFace model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[PionServeEngine] Loading {self.config.model_id}...")
        t0 = time.time()

        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self._model = self._model.to(device)

        # Extract RoPE config from model for re-rotation
        self._extract_rope_config()

        elapsed = time.time() - t0
        print(f"[PionServeEngine] Model loaded in {elapsed:.1f}s on {device}")

    def _extract_rope_config(self):
        """Extract RoPE parameters from loaded model."""
        try:
            model_config = self._model.config
            head_dim = getattr(model_config, "head_dim", 0)
            if head_dim == 0:
                hidden_size = getattr(model_config, "hidden_size", 0)
                num_heads = getattr(model_config, "num_attention_heads", 0)
                if hidden_size and num_heads:
                    head_dim = hidden_size // num_heads

            if head_dim > 0:
                rope_theta = getattr(model_config, "rope_theta", 10000.0)
                self.config.cache.rope_head_dim = head_dim
                self.config.cache.rope_base_theta = rope_theta
                print(f"[PionServeEngine] RoPE: head_dim={head_dim}, theta={rope_theta}")
        except Exception as e:
            print(f"[PionServeEngine] Could not extract RoPE config: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        request_id: str = "",
    ) -> GenerationResult:
        """Generate completion for a prompt.

        Checks semantic cache first. On hit, injects cached KV tensors
        and skips prefill. On miss, runs full prefill and stores result.
        """
        if not request_id:
            request_id = uuid.uuid4().hex[:12]

        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        t_start = time.perf_counter()
        self._stats.total_requests += 1

        # Step 0: Fleet routing (if configured)
        routed_to = ""
        routing_strategy = ""
        if self._routing_policy:
            decision = self._routing_policy.route(prompt=prompt)
            routed_to = decision.worker_id
            routing_strategy = decision.strategy_used

        # Step 1: Check semantic cache
        cache_result = self._cache_manager.lookup(prompt)

        if self._dry_run:
            result = self._dry_run_generate(request_id, prompt, cache_result, t_start)
            result.routed_to = routed_to
            result.routing_strategy = routing_strategy
            # Report completion to fleet manager for centroid update
            if self._fleet and routed_to:
                embedding = self._cache_manager._embedder.embed(prompt) if self._cache_manager._embedder else None
                self._fleet.report_completion(routed_to, embedding)
            return result

        # Step 2: Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        prompt_tokens = input_ids.shape[1]
        self._stats.total_prompt_tokens += prompt_tokens

        if cache_result.hit:
            return self._generate_with_cache(
                request_id, input_ids, prompt_tokens, cache_result,
                max_tok, temp, t_start, prompt,
            )
        else:
            return self._generate_full_prefill(
                request_id, input_ids, prompt_tokens,
                max_tok, temp, t_start, prompt,
            )

    def _generate_with_cache(
        self, request_id, input_ids, prompt_tokens, cache_result,
        max_tokens, temperature, t_start, prompt,
    ):
        """Generate using cached KV tensors (skip prefill)."""
        import torch

        self._stats.cache_hits += 1
        # Estimate saved FLOPs: ~2 * prompt_tokens^2 * hidden_size
        self._stats.prefill_flops_saved += prompt_tokens * prompt_tokens

        # Apply RoPE re-rotation if configured
        kv_layers = cache_result.kv_layers
        if self.config.cache.rope_head_dim > 0 and kv_layers:
            kv_layers = self._cache_manager.rerotate_cached_keys(
                kv_layers, original_offset=0, target_offset=0,
            )

        # Convert cached KV to model's past_key_values format
        past_kv = self._build_past_key_values(kv_layers)

        # Forward pass with cached KV (only process new/query tokens)
        with torch.no_grad():
            outputs = self._model(
                input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

        t_first_token = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000
        self._stats.total_ttft_ms += ttft_ms

        # Autoregressive decode
        generated_text, tokens_generated = self._decode_loop(
            outputs, max_tokens, temperature,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        self._stats.total_tokens_generated += tokens_generated
        self._stats.total_generation_ms += total_ms

        return GenerationResult(
            request_id=request_id,
            text=generated_text,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            cache_hit=True,
            cache_similarity=cache_result.similarity,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_second=tokens_generated / (total_ms / 1000) if total_ms > 0 else 0,
        )

    def _generate_full_prefill(
        self, request_id, input_ids, prompt_tokens,
        max_tokens, temperature, t_start, prompt,
    ):
        """Generate with full prefill, then store KV cache."""
        import torch

        self._stats.cache_misses += 1

        with torch.no_grad():
            outputs = self._model(
                input_ids,
                use_cache=True,
            )

        t_first_token = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000
        self._stats.total_ttft_ms += ttft_ms

        # Store KV cache for future reuse
        past_kv = outputs.past_key_values
        if past_kv:
            kv_layers = self._extract_kv_layers(past_kv)
            self._cache_manager.store(prompt, kv_layers)

        # Autoregressive decode
        generated_text, tokens_generated = self._decode_loop(
            outputs, max_tokens, temperature,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        self._stats.total_tokens_generated += tokens_generated
        self._stats.total_generation_ms += total_ms

        return GenerationResult(
            request_id=request_id,
            text=generated_text,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            cache_hit=False,
            cache_similarity=0.0,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_second=tokens_generated / (total_ms / 1000) if total_ms > 0 else 0,
        )

    def _decode_loop(self, outputs, max_tokens, temperature):
        """Autoregressive token generation loop."""
        import torch

        generated_ids = []
        next_logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values
        eos_id = self._tokenizer.eos_token_id or 2

        for _ in range(max_tokens):
            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id == eos_id:
                break
            generated_ids.append(token_id)

            with torch.no_grad():
                outputs = self._model(
                    next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            next_logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, len(generated_ids)

    def _build_past_key_values(self, kv_layers):
        """Convert numpy KV layers to PyTorch past_key_values tuple."""
        import torch
        device = self._model.device
        dtype = next(self._model.parameters()).dtype

        past_kv = []
        for keys, values in kv_layers:
            k = torch.from_numpy(keys.astype(np.float32)).to(device=device, dtype=dtype)
            v = torch.from_numpy(values.astype(np.float32)).to(device=device, dtype=dtype)
            past_kv.append((k, v))

        return tuple(past_kv)

    def _extract_kv_layers(self, past_key_values) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract KV layers from PyTorch past_key_values to numpy."""
        layers = []
        for k, v in past_key_values:
            k_np = k.detach().cpu().float().numpy()
            v_np = v.detach().cpu().float().numpy()
            layers.append((k_np, v_np))
        return layers

    def _dry_run_generate(
        self, request_id, prompt, cache_result, t_start,
    ) -> GenerationResult:
        """Simulate generation without a model (for benchmarking cache layer)."""
        ttft_ms = (time.perf_counter() - t_start) * 1000

        if cache_result.hit:
            self._stats.cache_hits += 1
            num_tokens = cache_result.num_tokens
        else:
            self._stats.cache_misses += 1
            num_tokens = len(prompt.split())

        # Simulate storing KV cache on miss
        if not cache_result.hit:
            # Create synthetic KV tensors for cache population
            num_layers = 32
            num_kv_heads = 8
            head_dim = 128
            seq_len = max(1, len(prompt.split()))
            fake_layers = [
                (
                    np.random.randn(1, num_kv_heads, seq_len, head_dim).astype(np.float16),
                    np.random.randn(1, num_kv_heads, seq_len, head_dim).astype(np.float16),
                )
                for _ in range(num_layers)
            ]
            self._cache_manager.store(prompt, fake_layers)

        total_ms = (time.perf_counter() - t_start) * 1000
        self._stats.total_ttft_ms += ttft_ms
        self._stats.total_generation_ms += total_ms
        self._stats.total_requests += 0  # already counted
        prompt_tokens = len(prompt.split())
        self._stats.total_prompt_tokens += prompt_tokens

        return GenerationResult(
            request_id=request_id,
            text="[dry-run]",
            tokens_generated=0,
            prompt_tokens=prompt_tokens,
            cache_hit=cache_result.hit,
            cache_similarity=cache_result.similarity,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_second=0,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
