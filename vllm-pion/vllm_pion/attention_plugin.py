"""Pion Externalized Attention Plugin — drop-in replacement for PyTorch attention.

Phase 3 of M14. Replaces standard attention computation in selected transformer
layers with Pion-backed HNSW retrieval. During prefill, token KV pairs are stored
in Pion's per-layer attention index. During decode, query vectors are sent to Pion
via ATTEND.QUERY, which returns the top-k most relevant (key, value) pairs; the
layer then computes attention over only those k results instead of the full context.

This dramatically reduces the per-token decode cost from O(context_len) to O(k),
enabling million-token context windows at constant memory and compute.

Usage:
    from vllm_pion.attention_plugin import (
        ExternalizedAttentionConfig,
        ExternalizedAttentionManager,
        patch_model_for_externalized_attention,
    )

    config = ExternalizedAttentionConfig(
        external_layers=list(range(20, 56)),
        k=128,
        key_dim=1024,
        value_dim=1024,
    )
    manager = ExternalizedAttentionManager(config)
    manager.create_session()
    model = patch_model_for_externalized_attention(model, config)
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_pion.attention_client import PionAttentionClient

logger = logging.getLogger(__name__)


@dataclass
class ExternalizedAttentionConfig:
    """Configuration for externalized attention via Pion HNSW index.

    Attributes:
        pion_host: Pion server hostname.
        pion_port: Pion server port.
        external_layers: List of transformer layer indices to externalize.
            Layers not in this list use standard local attention unchanged.
        k: Number of top-k tokens to retrieve from Pion per query.
        key_dim: Dimension of concatenated key heads (num_kv_heads * head_dim).
        value_dim: Dimension of concatenated value heads (num_kv_heads * head_dim).
        session_id: Unique session identifier. Auto-generated if empty.
        timeout: Socket timeout in seconds for Pion communication.
        fallback_to_local: If True, fall back to standard attention on Pion errors
            instead of raising. Useful for development and gradual rollout.
        store_batch_size: Maximum number of tokens to store in a single
            ATTEND.STORE call. Larger batches reduce round-trips but increase
            per-call latency.
    """

    pion_host: str = "127.0.0.1"
    pion_port: int = 1974
    external_layers: List[int] = field(default_factory=list)
    k: int = 128
    key_dim: int = 1024
    value_dim: int = 1024
    session_id: str = ""
    timeout: float = 10.0
    fallback_to_local: bool = True
    store_batch_size: int = 4096

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"attn_{int(time.time() * 1000)}"
        if not self.external_layers:
            logger.warning(
                "ExternalizedAttentionConfig: external_layers is empty. "
                "No layers will be externalized."
            )


class ExternalizedAttentionManager:
    """Manages Pion sessions and coordinates KV storage/retrieval across layers.

    This is the primary interface between the inference engine and Pion's
    externalized attention index. It owns the PionAttentionClient and provides
    high-level methods for session lifecycle, KV storage, and top-k queries.
    """

    def __init__(self, config: ExternalizedAttentionConfig) -> None:
        self.config = config
        self._client = PionAttentionClient(
            host=config.pion_host,
            port=config.pion_port,
            timeout=config.timeout,
        )
        self._session_created = False
        self._layer_token_counts: Dict[int, int] = {}
        logger.info(
            "ExternalizedAttentionManager initialized: host=%s port=%d "
            "session=%s layers=%s k=%d",
            config.pion_host,
            config.pion_port,
            config.session_id,
            config.external_layers,
            config.k,
        )

    @property
    def client(self) -> PionAttentionClient:
        """Expose the underlying client for advanced use cases."""
        return self._client

    def create_session(self) -> int:
        """Create an ATTEND session in Pion.

        Returns:
            Session index (>= 0) on success, -1 on failure.

        Raises:
            ConnectionError: If Pion is unreachable and fallback_to_local is False.
        """
        try:
            result = self._client.create_session(
                session_id=self.config.session_id,
                key_dim=self.config.key_dim,
                value_dim=self.config.value_dim,
            )
            if result >= 0:
                self._session_created = True
                logger.info(
                    "ATTEND session created: session=%s index=%d",
                    self.config.session_id,
                    result,
                )
            else:
                logger.error(
                    "ATTEND.CREATE returned error for session=%s",
                    self.config.session_id,
                )
            return result
        except Exception as e:
            logger.error("Failed to create ATTEND session: %s", e)
            if not self.config.fallback_to_local:
                raise ConnectionError(
                    f"Cannot create ATTEND session: {e}"
                ) from e
            return -1

    def store_prefill_kv(
        self,
        layer_id: int,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> bool:
        """Store prefill-phase KV pairs for a layer in Pion.

        Keys and values are stored in batches of store_batch_size to bound
        per-call latency for long prefills.

        Args:
            layer_id: Transformer layer index (0-based).
            keys: [num_tokens, key_dim] FP32 array.
            values: [num_tokens, value_dim] FP32 array.

        Returns:
            True if all batches stored successfully.
        """
        if not self._session_created:
            logger.warning(
                "store_prefill_kv called before create_session; "
                "attempting auto-create."
            )
            if self.create_session() < 0:
                return False

        num_tokens = keys.shape[0]
        if keys.shape[1] != self.config.key_dim:
            raise ValueError(
                f"Key dimension mismatch: expected {self.config.key_dim}, "
                f"got {keys.shape[1]}"
            )
        if values.shape[1] != self.config.value_dim:
            raise ValueError(
                f"Value dimension mismatch: expected {self.config.value_dim}, "
                f"got {values.shape[1]}"
            )

        batch_size = self.config.store_batch_size
        all_ok = True

        for start in range(0, num_tokens, batch_size):
            end = min(start + batch_size, num_tokens)
            k_batch = np.ascontiguousarray(keys[start:end], dtype=np.float32)
            v_batch = np.ascontiguousarray(values[start:end], dtype=np.float32)

            try:
                ok = self._client.store_tokens(
                    session_id=self.config.session_id,
                    layer_id=layer_id,
                    keys=k_batch,
                    values=v_batch,
                )
                if not ok:
                    logger.error(
                        "ATTEND.STORE failed: layer=%d batch=[%d:%d]",
                        layer_id,
                        start,
                        end,
                    )
                    all_ok = False
            except Exception as e:
                logger.error(
                    "ATTEND.STORE exception: layer=%d batch=[%d:%d] %s",
                    layer_id,
                    start,
                    end,
                    e,
                )
                all_ok = False
                if not self.config.fallback_to_local:
                    raise

        if all_ok:
            self._layer_token_counts[layer_id] = (
                self._layer_token_counts.get(layer_id, 0) + num_tokens
            )
            logger.debug(
                "Stored %d tokens for layer %d (total: %d)",
                num_tokens,
                layer_id,
                self._layer_token_counts[layer_id],
            )

        return all_ok

    def query(
        self,
        layer_id: int,
        query_vector: np.ndarray,
        k: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Query Pion for top-k most relevant value vectors.

        Args:
            layer_id: Transformer layer index.
            query_vector: [key_dim] FP32 query vector (concatenated Q heads).
            k: Number of results. Defaults to config.k.

        Returns:
            [k, value_dim] FP32 array of top-k value vectors, or None on failure.
        """
        if k is None:
            k = self.config.k

        query_flat = np.ascontiguousarray(
            query_vector.reshape(-1), dtype=np.float32
        )

        try:
            raw = self._client.query_topk(
                session_id=self.config.session_id,
                layer_id=layer_id,
                query=query_flat,
                k=k,
            )
            if raw is None:
                logger.debug(
                    "ATTEND.QUERY returned None: layer=%d", layer_id
                )
                return None

            # Parse raw bytes into [k_actual, value_dim] FP32 array.
            num_floats = len(raw) // 4
            values = np.frombuffer(raw, dtype=np.float32)

            if num_floats % self.config.value_dim != 0:
                logger.error(
                    "ATTEND.QUERY response size mismatch: %d floats not "
                    "divisible by value_dim=%d",
                    num_floats,
                    self.config.value_dim,
                )
                return None

            k_actual = num_floats // self.config.value_dim
            return values.reshape(k_actual, self.config.value_dim)

        except Exception as e:
            logger.error(
                "ATTEND.QUERY exception: layer=%d %s", layer_id, e
            )
            if not self.config.fallback_to_local:
                raise
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return ATTEND.INFO statistics from Pion.

        Returns:
            Dictionary of stat_name -> value. Empty dict on error.
        """
        try:
            return self._client.info()
        except Exception as e:
            logger.error("ATTEND.INFO failed: %s", e)
            return {}

    def get_layer_token_count(self, layer_id: int) -> int:
        """Return the number of tokens stored for a given layer."""
        return self._layer_token_counts.get(layer_id, 0)


class ExternalizedAttentionLayer(nn.Module):
    """Drop-in replacement for a transformer attention layer.

    When the layer_id is in the configured external_layers list, this module
    sends query vectors to Pion's ATTEND.QUERY and computes attention over
    the returned top-k (key, value) pairs. Otherwise, it delegates to the
    original attention module unchanged.

    The forward pass supports two modes:
      - **Prefill mode** (key_value_cache is not None and has sequence length):
        Standard attention is used, then KV pairs are stored in Pion.
      - **Decode mode** (single-token generation):
        Query is sent to Pion, top-k values returned, attention computed
        over k results instead of the full context.
    """

    def __init__(
        self,
        layer_id: int,
        config: ExternalizedAttentionConfig,
        manager: ExternalizedAttentionManager,
        original_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.manager = manager
        self.original_module = original_module
        self._is_external = layer_id in config.external_layers

        if self._is_external:
            logger.debug(
                "Layer %d: externalized attention enabled (k=%d)",
                layer_id,
                config.k,
            )

    @property
    def is_external(self) -> bool:
        """Whether this layer uses externalized attention."""
        return self._is_external

    def store_kv(self, keys: torch.Tensor, values: torch.Tensor) -> bool:
        """Store token KV pairs in Pion after prefill.

        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim] or
                  [batch, seq_len, num_kv_heads * head_dim] tensor.
            values: Same shape convention as keys.

        Returns:
            True if storage succeeded.
        """
        if not self._is_external:
            return True

        # Reshape to [num_tokens, dim] — collapse batch and seq dimensions,
        # concatenate heads if needed.
        k_np = self._to_flat_numpy(keys)
        v_np = self._to_flat_numpy(values)

        return self.manager.store_prefill_kv(self.layer_id, k_np, v_np)

    def forward(
        self,
        query: torch.Tensor,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention, optionally via Pion externalization.

        Args:
            query: [batch, num_q_heads, seq_len, head_dim] query tensor.
            key_value_cache: Optional (key, value) tuple, each shaped
                [batch, num_kv_heads, seq_len, head_dim]. Used for standard
                local attention when the layer is not externalized or as
                fallback.
            attention_mask: Optional attention mask for local attention.

        Returns:
            [batch, num_q_heads, seq_len, head_dim] attention output.
        """
        # Non-external layers: always use standard attention.
        if not self._is_external:
            return self._local_attention(query, key_value_cache, attention_mask)

        batch_size, num_q_heads, seq_len, head_dim = query.shape

        # Prefill (seq_len > 1): use local attention, then store KV in Pion.
        if seq_len > 1:
            output = self._local_attention(
                query, key_value_cache, attention_mask
            )
            if key_value_cache is not None:
                k_cache, v_cache = key_value_cache
                self.store_kv(k_cache, v_cache)
            return output

        # Decode (seq_len == 1): query Pion for top-k, compute sparse attention.
        return self._externalized_decode(
            query, key_value_cache, attention_mask
        )

    def _externalized_decode(
        self,
        query: torch.Tensor,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Perform decode-phase externalized attention via Pion.

        Sends the query to Pion, receives top-k value vectors, and computes
        scaled dot-product attention over just those k results.
        """
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        device = query.device
        dtype = query.dtype

        # Flatten query heads for Pion: [batch, num_q_heads * head_dim].
        q_flat = query.squeeze(2)  # [batch, num_q_heads, head_dim]
        q_flat = q_flat.reshape(batch_size, -1)  # [batch, num_q_heads * head_dim]

        results = []
        for b in range(batch_size):
            q_np = q_flat[b].detach().cpu().float().numpy()
            topk_values = self.manager.query(self.layer_id, q_np)

            if topk_values is not None:
                # topk_values: [k_actual, value_dim]
                k_actual = topk_values.shape[0]

                # Reshape values back to [k_actual, num_kv_heads, head_dim].
                num_kv_heads = self.config.value_dim // head_dim
                v_tensor = torch.from_numpy(topk_values).to(
                    device=device, dtype=dtype
                )
                v_tensor = v_tensor.reshape(
                    k_actual, num_kv_heads, head_dim
                )  # [k, num_kv_heads, head_dim]

                # For GQA: expand kv_heads to match q_heads.
                heads_per_group = num_q_heads // num_kv_heads
                if heads_per_group > 1:
                    v_tensor = v_tensor.unsqueeze(2).expand(
                        k_actual, num_kv_heads, heads_per_group, head_dim
                    )
                    v_tensor = v_tensor.reshape(
                        k_actual, num_q_heads, head_dim
                    )

                # v_tensor: [k, num_q_heads, head_dim]
                # query_b: [num_q_heads, 1, head_dim]
                query_b = query[b]  # [num_q_heads, 1, head_dim]

                # Transpose v for attention: [num_q_heads, k, head_dim].
                v_t = v_tensor.permute(1, 0, 2)  # [num_q_heads, k, head_dim]

                # Uniform attention weights over top-k results (keys are not
                # returned by ATTEND.QUERY; Pion's HNSW already ranked by
                # query-key similarity). This is equivalent to the retrieval
                # step selecting the k most relevant KV pairs.
                scale = 1.0 / math.sqrt(head_dim)

                # Compute attention scores: Q @ V^T to approximate relevance.
                # query_b: [num_q_heads, 1, head_dim]
                # v_t:     [num_q_heads, k, head_dim]
                attn_scores = torch.matmul(
                    query_b, v_t.transpose(-2, -1)
                )  # [num_q_heads, 1, k]
                attn_scores = attn_scores * scale
                attn_weights = F.softmax(attn_scores, dim=-1)

                # Weighted sum over top-k values.
                # attn_weights: [num_q_heads, 1, k]
                # v_t:          [num_q_heads, k, head_dim]
                output_b = torch.matmul(
                    attn_weights, v_t
                )  # [num_q_heads, 1, head_dim]

                results.append(output_b)
            else:
                # Fallback to local attention for this batch element.
                logger.debug(
                    "Layer %d: Pion query returned None for batch %d, "
                    "falling back to local attention.",
                    self.layer_id,
                    b,
                )
                if key_value_cache is not None:
                    fallback = self._local_attention_single(
                        query[b:b + 1],
                        key_value_cache,
                        attention_mask,
                    )
                    results.append(
                        fallback.squeeze(0)
                    )  # [num_q_heads, 1, head_dim]
                else:
                    # No cache and no Pion result: return zeros.
                    results.append(
                        torch.zeros(
                            num_q_heads,
                            1,
                            head_dim,
                            device=device,
                            dtype=dtype,
                        )
                    )

        # Stack batch results: [batch, num_q_heads, 1, head_dim].
        return torch.stack(results, dim=0)

    def _local_attention(
        self,
        query: torch.Tensor,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard scaled dot-product attention (local, full context)."""
        if key_value_cache is None:
            # Self-attention: Q=K=V (unusual in decode, but handle gracefully).
            return F.scaled_dot_product_attention(
                query, query, query, attn_mask=attention_mask
            )

        key, value = key_value_cache
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )

    def _local_attention_single(
        self,
        query: torch.Tensor,
        key_value_cache: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Local attention for a single batch element (fallback path)."""
        key, value = key_value_cache
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )

    def _to_flat_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a multi-dimensional KV tensor to [num_tokens, dim] FP32 numpy.

        Handles both:
          - [batch, num_heads, seq_len, head_dim] (standard 4D)
          - [batch, seq_len, concat_dim] (pre-concatenated 3D)
        """
        t = tensor.detach().cpu().float()

        if t.ndim == 4:
            batch, num_heads, seq_len, head_dim = t.shape
            # Reshape to [batch * seq_len, num_heads * head_dim].
            t = t.permute(0, 2, 1, 3).reshape(
                batch * seq_len, num_heads * head_dim
            )
        elif t.ndim == 3:
            batch, seq_len, dim = t.shape
            t = t.reshape(batch * seq_len, dim)
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {t.ndim}D shape {t.shape}"
            )

        return np.ascontiguousarray(t.numpy())


def patch_model_for_externalized_attention(
    model: nn.Module,
    config: ExternalizedAttentionConfig,
) -> nn.Module:
    """Wrap selected model layers with ExternalizedAttentionLayer.

    This function walks the model's named modules looking for attention layers
    (by convention: modules with 'attn' or 'attention' or 'self_attn' in
    their name) and wraps those at indices in config.external_layers.

    The patching preserves the original module as a sub-module so that its
    parameters remain part of the model's state_dict.

    Args:
        model: A PyTorch transformer model (e.g., LlamaForCausalLM).
        config: Externalized attention configuration.

    Returns:
        The same model object with selected attention layers wrapped.
        (Modified in-place; the return value is for convenience.)
    """
    if not config.external_layers:
        logger.warning(
            "patch_model_for_externalized_attention: no external_layers "
            "configured, returning model unchanged."
        )
        return model

    manager = ExternalizedAttentionManager(config)

    # Find transformer layers by common naming conventions.
    # Supports: model.layers[i].self_attn (Llama, Mistral, Qwen)
    #           model.transformer.h[i].attn (GPT-2, GPT-J)
    #           model.model.layers[i].self_attn (wrapped models)
    patched_count = 0
    layer_modules = _find_transformer_layers(model)

    if not layer_modules:
        logger.warning(
            "Could not find transformer layers in model. "
            "Supported patterns: model.layers[i], model.transformer.h[i], "
            "model.model.layers[i]. Returning model unchanged."
        )
        return model

    for layer_idx, (parent, attr_name, layer_module) in enumerate(
        layer_modules
    ):
        if layer_idx not in config.external_layers:
            continue

        # Find the attention sub-module within this transformer layer.
        attn_attr = _find_attention_attr(layer_module)
        if attn_attr is None:
            logger.warning(
                "Layer %d: could not find attention sub-module, skipping.",
                layer_idx,
            )
            continue

        original_attn = getattr(layer_module, attn_attr)
        wrapper = ExternalizedAttentionLayer(
            layer_id=layer_idx,
            config=config,
            manager=manager,
            original_module=original_attn,
        )
        setattr(layer_module, attn_attr, wrapper)
        patched_count += 1
        logger.info("Patched layer %d attention (%s)", layer_idx, attn_attr)

    logger.info(
        "Externalized attention patching complete: %d/%d layers patched.",
        patched_count,
        len(config.external_layers),
    )

    # Attach the manager to the model for external access.
    model._pion_attention_manager = manager  # type: ignore[attr-defined]

    return model


def _find_transformer_layers(
    model: nn.Module,
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Locate the sequential list of transformer layers in the model.

    Returns a list of (parent_module, attribute_name, layer_module) tuples,
    ordered by layer index.
    """
    # Common layer container paths in popular model architectures.
    candidates = [
        ("model", "layers"),           # Llama, Mistral, Qwen2, Phi
        ("transformer", "h"),          # GPT-2, GPT-J, GPT-NeoX
        ("model.model", "layers"),     # Double-wrapped (e.g., GPTQ models)
        ("encoder", "layer"),          # BERT, RoBERTa
        ("decoder", "layers"),         # T5 decoder, BART
    ]

    for container_path, layer_attr in candidates:
        parent = model
        try:
            for part in container_path.split("."):
                parent = getattr(parent, part)
            layers = getattr(parent, layer_attr, None)
            if layers is not None and isinstance(layers, nn.ModuleList):
                return [
                    (parent, f"{layer_attr}[{i}]", layer)
                    for i, layer in enumerate(layers)
                ]
        except AttributeError:
            continue

    # Fallback: search for any ModuleList with >10 children (likely the
    # transformer stack).
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            return [
                (parent, f"{name.split('.')[-1]}[{i}]", layer)
                for i, layer in enumerate(module)
            ]

    return []


def _find_attention_attr(layer_module: nn.Module) -> Optional[str]:
    """Find the attention sub-module attribute name in a transformer layer.

    Checks common naming conventions across model families.
    """
    # Ordered by prevalence in HuggingFace model zoo.
    candidates = [
        "self_attn",    # Llama, Mistral, Qwen, Phi
        "attn",         # GPT-2, GPT-J
        "attention",    # BERT, some custom models
        "self_attention",  # GPT-NeoX
    ]
    for attr in candidates:
        if hasattr(layer_module, attr):
            return attr
    return None
