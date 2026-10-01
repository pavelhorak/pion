"""vllm-pion: Pion KV Cache Connector and Semantic Inference Engine.

Provides:
  - PionKVClient: RESP client for KV.STORE/FETCH
  - PionAttentionClient: ATTEND.* commands for externalized attention
  - SemanticCacheManager: Prompt-level semantic KV cache matching
  - PionServeEngine: Full inference engine with cache-accelerated generation
  - RoPE re-rotation: Position-shifted KV cache injection
  - KV serializer: Standardized multi-layer tensor serialization

Usage (vLLM connector):
    VLLM_KV_CONNECTOR=vllm_pion.connector.PionKVConnector \\
    python -m vllm.entrypoints.openai.api_server --model ...

Usage (standalone Pion Serve):
    pion-serve --model meta-llama/Llama-3.2-1B-Instruct --embed auto

Usage (cache layer only, dry-run):
    pion-serve
"""

__version__ = "0.2.0"
