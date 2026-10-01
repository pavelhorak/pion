"""Pion attention integration for mlx-lm on Apple Silicon.

Pion's native Metal SDPA kernel (src/ffi/metal_compute.metal,
sdpa_q1_fp32 / sdpa_batched_q_fp32 / fp16 variants) is used by the
mlx-lm monkey-patch via ATTEND.PREFIX.QUERY. Start pion-server with:

    ./pion-server --kvcache --metal-attention -w 1

Usage — Stage 1, the default: a cross-process prompt cache mlx-lm decodes from
at native speed.

    from pion_vllm_mlx import PionPromptCache
    pc = PionPromptCache(model, host="127.0.0.1", port=1974)
    cache = pc.get_or_prefill(prefix_ids, namespace=ns)   # MISS: prefill + store · HIT: fetch
    text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)

Stage 2 — Pion computes the attention over the prefix (sparse selectors,
pion-exo, custom CacheEngines):

    install_pion_attention_patch()
    pc = PionPromptCache(model, stage2=True, host="127.0.0.1", port=1974)
    if not pc.lookup(ns):
        pc.get_or_prefill(prefix_ids, namespace=ns)
    cache = make_pion_prompt_cache(model, namespace=ns, prompt_cache=pc,
                                   prefix_len=len(prefix_ids))
    text = generate(model, tok, prompt=suffix_ids, prompt_cache=cache)
"""

from pion_vllm_mlx.prompt_cache import PionPromptCache
from pion_vllm_mlx.hybrid_retrieval import HybridRetrievalCache

from pion_vllm_mlx._compat import PionMlxCompatError, describe as mlx_lm_seam_report

# Stage-2 mlx-lm Attention monkey-patch — optional, only when mlx_lm is installed.
#
# gh #263: the names are exported EITHER WAY. Letting them simply not exist
# turned "mlx-lm is missing or incompatible" into `ImportError: cannot import
# name 'install_pion_attention_patch'`, which names the wrong thing entirely —
# the reader goes looking for a typo in our package. The stubs below carry the
# real reason to the call site.
_PATCH_IMPORT_ERROR = None
try:
    from pion_vllm_mlx.mlx_lm_patch import (
        install_pion_attention_patch,
        uninstall_pion_attention_patch,
        make_pion_prompt_cache,
        PionPrefixCache,
    )
    _PATCH_AVAILABLE = True
except ImportError as _e:
    _PATCH_AVAILABLE = False
    _PATCH_IMPORT_ERROR = _e

    def _unavailable(*_args, **_kwargs):
        raise PionMlxCompatError(
            "the Pion attention patch could not be imported: %s\n"
            "It needs Apple MLX and mlx-lm, which are Apple-Silicon only:\n"
            "    pip install 'pion-vllm-mlx[mlx]'\n"
            "PionPromptCache itself does not need them and is unaffected."
            % (_PATCH_IMPORT_ERROR,)
        )

    def install_pion_attention_patch(*a, **k):    # type: ignore[misc]
        _unavailable(*a, **k)

    def uninstall_pion_attention_patch(*a, **k):  # type: ignore[misc]
        _unavailable(*a, **k)

    def make_pion_prompt_cache(*a, **k):          # type: ignore[misc]
        _unavailable(*a, **k)

    PionPrefixCache = None                        # type: ignore[assignment]

__all__ = [
    "PionPromptCache", "HybridRetrievalCache",
    "install_pion_attention_patch", "uninstall_pion_attention_patch",
    "make_pion_prompt_cache", "PionPrefixCache",
    "PionMlxCompatError", "mlx_lm_seam_report",
]
