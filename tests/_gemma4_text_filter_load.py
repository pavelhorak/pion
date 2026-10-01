"""In-memory loader: build a gemma4_text Model from a cached multimodal Gemma4 4-bit quant.

Filters language_model.* weights from `mlx-community/gemma-4-e2b-it-4bit` (multimodal,
already cached) and constructs a `gemma4_text.Model` with them. Avoids downloading the
separate text-only quant (which we don't have disk space for right now).

Usage:
    from _gemma4_text_filter_load import load_text_only_from_cached
    model, tok = load_text_only_from_cached("mlx-community/gemma-4-e2b-it-4bit")
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gemma4_text import Model as Gemma4TextModel, ModelArgs as Gemma4TextArgs
from mlx_lm.tokenizer_utils import load as load_tokenizer


def _resolve_snapshot(repo: str) -> Path:
    """Look for a cached snapshot of `repo` in HF_HUB_CACHE first (shared bulk
    cache, e.g. /Volumes/192.168.0.1/hf_cache when the SMB share is mounted),
    then fall back to the local HF_HOME convention (~/.cache/huggingface/hub).

    HF_HUB_CACHE layout (set explicitly) holds `models--*` directly at the
    root; the default HF_HOME layout puts them under a `hub/` subdir.
    """
    import os
    repo_dir = f"models--{repo.replace('/', '--')}"
    candidates = []
    if os.environ.get("HF_HUB_CACHE"):
        candidates.append(Path(os.environ["HF_HUB_CACHE"]) / repo_dir)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / repo_dir)
    for cache in candidates:
        snaps_dir = cache / "snapshots"
        if snaps_dir.is_dir():
            snaps = list(snaps_dir.iterdir())
            if snaps:
                return snaps[0]
    raise FileNotFoundError(f"No cached snapshot for {repo} in {[str(c) for c in candidates]}")


def load_text_only_from_cached(repo: str = "mlx-community/gemma-4-e2b-it-4bit"):
    snapshot = _resolve_snapshot(repo)
    cfg_full = json.loads((snapshot / "config.json").read_text())

    text_cfg = dict(cfg_full["text_config"])
    text_cfg.setdefault("model_type", "gemma4_text")
    text_cfg["quantization"] = cfg_full.get("quantization")
    if cfg_full.get("quantization_config"):
        text_cfg["quantization_config"] = cfg_full["quantization_config"]
    if "vocab_size" not in text_cfg:
        text_cfg["vocab_size"] = cfg_full.get("vocab_size", 262208)
    text_cfg["tie_word_embeddings"] = text_cfg.get("tie_word_embeddings", True)

    weights_files = glob.glob(str(snapshot / "model*.safetensors"))
    raw = {}
    for wf in weights_files:
        raw.update(mx.load(wf))

    text_weights = {}
    for k, v in raw.items():
        if k.startswith("language_model."):
            text_weights[k[len("language_model."):]] = v
    if not text_weights:
        raise RuntimeError("no language_model.* weights found")

    args = Gemma4TextArgs.from_dict(text_cfg)
    model = Gemma4TextModel(args)

    if hasattr(model, "sanitize"):
        text_weights = model.sanitize(text_weights)

    qcfg = text_cfg.get("quantization")
    if qcfg is not None:
        def class_predicate(p, m):
            if isinstance(qcfg, dict) and p in qcfg and isinstance(qcfg[p], dict):
                return qcfg[p]
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in text_weights

        nn.quantize(
            model,
            group_size=qcfg["group_size"],
            bits=qcfg["bits"],
            mode=qcfg.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.eval()
    model.load_weights(list(text_weights.items()), strict=False)
    mx.eval(model.parameters())

    tok = load_tokenizer(snapshot, eos_token_ids=text_cfg.get("eos_token_id"))
    # gh #93: Gemma 4's `GemmaTokenizer` ships with `add_bos_token=False`, so
    # `tok.encode("…")` does NOT prepend `<bos>` (id=2). On Gemma-4-12B-it-4bit
    # this collapses the next-token distribution onto punctuation/control
    # tokens — the very-degenerate-vanilla-output bug. (E2B-it-4bit happened to
    # produce coherent output without BOS, masking the issue.) Force BOS
    # prepending on both the outer TokenizerWrapper and the underlying
    # `GemmaTokenizer` so every consumer of this helper gets a faithful vanilla
    # baseline.
    try:
        tok.add_bos_token = True
    except Exception:
        pass
    underlying = getattr(tok, "_tokenizer", None)
    if underlying is not None:
        try:
            underlying.add_bos_token = True
        except Exception:
            pass
    return model, tok


if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/gemma-4-e2b-it-4bit"
    print(f"loading text-only filter view of {repo}...")
    model, tok = load_text_only_from_cached(repo)
    print(f"OK: model_type={model.model_type}  layers={model.args.num_hidden_layers}  hidden={model.args.hidden_size}")
    test = "Hello, world!"
    ids = tok.encode(test)
    print(f"tokenizer round-trip: {test!r} -> {ids[:10]}...  decoded={tok.decode(ids)!r}")
    out = model(mx.array([ids]))
    print(f"forward output shape: {out.shape}")
