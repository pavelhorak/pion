"""HybridRetrievalCache — embedding-recall + stored K/V injection (gh #54).

The retrieval is done by the consumer's existing embedding stack (any stable
embedder: bge-small, MiniLM, OpenAI, etc.). Pion's role is to cache the
already-prefilled K/V tensors of each *retrieved chunk* keyed by chunk_id, so
the consumer's next forward pass skips the prefill of the retrieved chunk.

Stage 0 spike for gh #54 measured (Llama-3.2-1B-Instruct-4bit, Mac M-series):
    in-process backend: bit-perfect, ~70% TTFT savings, no Pion server needed
    wire backend (Pion fp16): ~0.97 BLEU functional parity, cross-process

Two backends:
  - "inproc" (default, single-process consumers): K/V held as MLX arrays in
    a process-local dict. Bit-perfect generation parity vs combined encoding,
    matched only by the consumer running the model itself. No server needed.
  - "pion" (cross-process / cross-host): K/V serialized via the existing
    KV.PREFIX.* + V.STOREBATCH/V.FETCH path. fp16 precision (BLEU ~0.97 vs
    text-RAG, established in `kv_prefix_cache_ship.md`). Requires
    `pion-server --kvcache --metal-attention -w 1`.

The two modes share the same chunk_id ↔ prefix_len manifest. The wire backend
upgrades / downgrades automatically: ingest publishes to Pion, prepare on a
chunk_id missing locally falls back to a Pion fetch.

Usage:
    from mlx_lm import load
    from pion_vllm_mlx import HybridRetrievalCache

    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    hr = HybridRetrievalCache(model)             # inproc, bit-perfect

    chunk_ids = tok.encode("The Eiffel Tower is...")
    hr.ingest("eiffel_passage", chunk_ids)

    # The query follows the chunk, so encode it without special tokens: a plain
    # tok.encode() prepends <bos> again (Llama 3 does by default), and a second
    # <bos> mid-prompt changes what the model answers.
    query_ids = tok.encode("Question: How tall is it?\\nAnswer:", add_special_tokens=False)
    cache, suffix_ids = hr.prepare("eiffel_passage", query_ids)
    # mlx-lm generate from this point — chunk K/V is already in `cache`.

For cross-process / cross-host:
    hr = HybridRetrievalCache(model, backend="pion", port=1984)

v2 adds multi-chunk composition (gh #150) — `set_shared_stub` + `prepare_multi`:

  * **Delta-rotation** moves a chunk's K from the positions it was encoded at to
    the positions it will occupy, using R(delta)·R(p) = R(p+delta) with the
    model's own inv_freq. Exact, not an approximation: the rotation-identity
    check lands at ~2e-05 rel err vs prefilling at the target offset, and holds
    out to delta=84K. V needs no rotation. RoPE is not the composition problem.
  * **Shared-stub restoration is mandatory.** Serving chunk K/V whose stored
    instruction-stub prefix was sliced off, without re-prepending the stub's own
    K/V, degenerates immediately ("QL MB CONS CONS..."). The stub is encoded
    once and always spliced first.

**Composition grain must be coarse, and that is a measured bound, not caution.**
At product scale (Qwen3-4B, 20 per-doc packs = an 86K corpus, shared stub +
delta-rotation, composer verified bit-identical) direct recall was 15/27 against
26/27 for a single mono cartridge of the same corpus. The failure is not sinks
and not RoPE (big-delta rotation was exonerated to delta=84K): it is
cross-pack distractor collision — mono ingest encodes each document having
attended the previous ones, which differentiates near-duplicate facts, and packs
built in isolation forfeit that. Same question passes at 2 packs and abstains at
20. `prepare_multi` therefore refuses more than `max_packs` (default 8, the
largest count measured at mono parity) unless you pass `allow_fine_grain=True`.

The update economics are the reason to use it anyway: rebuilding one document's
pack and recomposing is ~10.4 s against ~1,700 s for a full re-ingest (~170x).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pion_vllm_mlx._compat import set_slot_arrays


def _require_mlx():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    return mx, make_prompt_cache


def _layout_from(model):
    a = model.args
    n_kv = getattr(a, "num_key_value_heads", a.num_attention_heads)
    head_dim = getattr(a, "head_dim", a.hidden_size // a.num_attention_heads)
    return a.num_hidden_layers, n_kv, head_dim


class HybridRetrievalCache:
    """Chunk-id-keyed K/V cache for RAG pipelines.

    See module docstring for the two-backend design.
    """

    def __init__(
        self,
        model,
        backend: Literal["inproc", "pion"] = "inproc",
        host: str = "127.0.0.1",
        port: int = 1974,
        vquant: str = "fp16",
        manifest_path: str | os.PathLike | None = None,
    ) -> None:
        if backend not in ("inproc", "pion"):
            raise ValueError(f"backend must be 'inproc' or 'pion', got {backend!r}")
        self.model = model
        self.backend = backend
        self.n_layers, self.n_kv_heads, self.head_dim = _layout_from(model)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        # chunk_id -> {"namespace": str, "tokens": int}
        self._manifest: dict[str, dict] = {}
        if self.manifest_path and self.manifest_path.exists():
            with self.manifest_path.open() as f:
                self._manifest = json.load(f)
        # In-process K/V store: chunk_id -> [(K_mx, V_mx), ...] one tuple per layer.
        # Each K_mx / V_mx has shape (1, n_kv_heads, prefix_len, head_dim), fp16 mlx.
        # Memory cost: ~8 KB per token × prefix_len. Caller manages eviction.
        self._inproc_kv: dict[str, list] = {}
        # Pion wire backend (lazy — only constructed if backend="pion" or fallback used)
        self._pion = None
        if backend == "pion":
            from .prompt_cache import PionPromptCache
            self._pion = PionPromptCache(model, vquant=vquant, host=host, port=port)
        # gh #150 v2: the shared instruction stub, encoded once. Chunks are
        # ingested behind it and always served behind it.
        self._stub_kv: list | None = None
        self._stub_len: int = 0
        self._stub_ids: list[int] = []
        # stats
        self.ingest_count = 0
        self.hydrate_count = 0
        self.hydrate_miss_count = 0

    @staticmethod
    def chunk_namespace(chunk_id: str) -> str:
        """Stable namespace derived from chunk_id. Hashed so chunk_ids that
        contain ':' / spaces / unicode do not break the wire-level KV.PREFIX
        key. Mirrors PionPromptCache.make_namespace under a chunk-scoped prefix
        so chunk caches do not collide with prompt prefix caches."""
        return "chunk_" + hashlib.sha256(chunk_id.encode()).hexdigest()[:24]

    def _persist(self) -> None:
        if self.manifest_path is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(self._manifest, f)
        tmp.replace(self.manifest_path)

    def _encode_to_mlx_kv(self, chunk_token_ids: list[int]):
        """Run forward pass on the chunk and snapshot per-layer K/V as MLX arrays.

        Returned shape per layer: (1, n_kv_heads, prefix_len, head_dim) fp16,
        ready to be loaded with _compat.set_slot_arrays for bit-perfect reuse.
        """
        mx, make_prompt_cache = _require_mlx()
        cache = make_prompt_cache(self.model)
        _ = self.model(mx.array([chunk_token_ids]), cache=cache)
        mx.eval([c.state for c in cache])
        prefix_len = len(chunk_token_ids)
        per_layer = []
        for li in range(self.n_layers):
            k = cache[li].keys[:, :, :prefix_len, :]
            v = cache[li].values[:, :, :prefix_len, :]
            mx.eval(k, v)  # materialize so the lazy graph doesn't grow per chunk
            per_layer.append((k, v))
        return per_layer

    def ingest(self, chunk_id: str, chunk_token_ids: list[int]) -> dict:
        """Encode the chunk through the model, store K/V. Idempotent.

        For backend="inproc": held as MLX arrays in this process.
        For backend="pion": shipped via KV.PREFIX.REGISTER + V.STOREBATCH.

        If chunk_id is already present with the same token count, this is a
        no-op (avoids re-encoding). Pass a fresh chunk_id to rotate content.
        """
        existing = self._manifest.get(chunk_id)
        if existing and existing["tokens"] == len(chunk_token_ids):
            if self.backend == "inproc" and chunk_id in self._inproc_kv:
                return existing
            # Fall through to re-populate (e.g. pion-only path)

        if self.backend == "inproc":
            self._inproc_kv[chunk_id] = self._encode_to_mlx_kv(chunk_token_ids)
            ns = self.chunk_namespace(chunk_id)
        else:
            ns = self.chunk_namespace(chunk_id)
            _ = self._pion.get_or_prefill(chunk_token_ids, namespace=ns)

        entry = {"namespace": ns, "tokens": len(chunk_token_ids), "backend": self.backend}
        self._manifest[chunk_id] = entry
        self._persist()
        self.ingest_count += 1
        return entry

    def has(self, chunk_id: str) -> bool:
        """True iff chunk_id is known (locally or, for Pion backend, on server)."""
        if chunk_id in self._manifest:
            return True
        if self.backend == "pion":
            return self._pion.lookup(self.chunk_namespace(chunk_id))
        return False

    def prepare(
        self,
        chunk_id: str,
        suffix_token_ids: list[int],
        chunk_token_count: int | None = None,
    ):
        """Hydrate the chunk's K/V into a fresh mlx-lm cache and return
        (cache, suffix_token_ids). The cache is positioned at offset =
        chunk_token_count, so the next model forward will RoPE-rotate the
        suffix tokens at positions chunk_token_count..chunk_token_count+S-1
        — identical to encoding (chunk + suffix) as one continuous text.

        chunk_token_count is read from the local manifest if not provided;
        pass explicitly for cross-process consumers that maintain their own
        mapping alongside the embedding index.

        Returns:
            tuple: `(cache, suffix_token_ids)` — pass to mlx-lm's generate loop. The
            cache already contains the chunk K/V; the consumer feeds suffix
            via model(suffix, cache=cache) directly, no extra prefill.
        """
        mx, make_prompt_cache = _require_mlx()
        if chunk_token_count is None:
            entry = self._manifest.get(chunk_id)
            if entry is None:
                raise KeyError(
                    f"chunk_id {chunk_id!r} not in local manifest. Ingest "
                    f"in this process first, or pass chunk_token_count=N."
                )
            chunk_token_count = entry["tokens"]

        if self.backend == "inproc":
            per_layer = self._inproc_kv.get(chunk_id)
            if per_layer is None:
                self.hydrate_miss_count += 1
                raise KeyError(
                    f"chunk_id {chunk_id!r} not in in-process K/V dict. "
                    f"Re-ingest after a restart, or use backend='pion' for "
                    f"cross-process persistence."
                )
            cache = make_prompt_cache(self.model)
            for li in range(self.n_layers):
                set_slot_arrays(cache[li], per_layer[li], offset=chunk_token_count)
            self.hydrate_count += 1
            return cache, suffix_token_ids

        # pion backend
        ns = self.chunk_namespace(chunk_id)
        if not self._pion.lookup(ns):
            self.hydrate_miss_count += 1
            raise KeyError(
                f"chunk_id {chunk_id!r} (namespace {ns!r}) not registered "
                f"on Pion. Cache evicted, or wrong host/port?"
            )
        cache = self._pion._fetch_to_cache(ns, chunk_token_count)
        self.hydrate_count += 1
        return cache, suffix_token_ids

    # ── v2: multi-chunk composition (gh #150) ──────────────────────────────

    def _rope_params(self):
        """(head_dim, theta) for the rotation, refusing layouts the identity
        does not hold for rather than silently emitting rotated garbage."""
        a = self.model.args
        if getattr(a, "rope_traditional", False):
            raise NotImplementedError(
                "delta-rotation implements the split-half RoPE layout; this "
                "model sets rope_traditional=True (interleaved). Encode the "
                "chunk set as one mono pack instead."
            )
        if getattr(a, "rope_scaling", None):
            raise NotImplementedError(
                "delta-rotation assumes plain RoPE; this model uses "
                "rope_scaling, where R(delta)·R(p) != R(p+delta). Encode the "
                "chunk set as one mono pack instead."
            )
        return self.head_dim, float(getattr(a, "rope_theta", 10000.0))

    def _rotate_k(self, k, delta: int):
        """Move K from the positions it was encoded at to those + delta.

        k' = k·cos(delta·theta) + rotate_half(k)·sin(delta·theta). The shift is
        constant across the chunk, so this is one cos/sin vector broadcast over
        the token axis — not a per-position table."""
        mx, _ = _require_mlx()
        if delta == 0:
            return k
        head_dim, theta = self._rope_params()
        half = head_dim // 2
        inv_freq = mx.exp(
            -mx.log(mx.array(theta)) * (mx.arange(0, half, dtype=mx.float32) * 2.0 / head_dim)
        )
        angle = inv_freq * float(delta)                     # (half,)
        cos = mx.concatenate([mx.cos(angle), mx.cos(angle)])  # (head_dim,)
        sin = mx.concatenate([mx.sin(angle), mx.sin(angle)])
        kf = k.astype(mx.float32)
        x1 = kf[..., :half]
        x2 = kf[..., half:]
        rot_half = mx.concatenate([-x2, x1], axis=-1)
        out = kf * cos + rot_half * sin
        return out.astype(k.dtype)

    def set_shared_stub(self, stub_token_ids: list[int]) -> dict:
        """Encode the shared instruction stub once. Required before ingesting
        chunks for multi-chunk use, and spliced ahead of every composition.

        Omitting it is not a quality trade — chunk K/V served without the stub
        it was encoded behind degenerates into repetition immediately."""
        mx, _ = _require_mlx()
        self._stub_ids = list(stub_token_ids)
        self._stub_len = len(self._stub_ids)
        self._stub_kv = self._encode_to_mlx_kv(self._stub_ids) if self._stub_len else None
        return {"stub_tokens": self._stub_len}

    def ingest_pack(self, chunk_id: str, chunk_token_ids: list[int]) -> dict:
        """Ingest a chunk for multi-chunk composition: encode it *behind* the
        shared stub, then keep only the chunk's own K/V slice.

        The stub has to be present at encode time (the chunk attends it) and
        re-prepended at serve time (see set_shared_stub) — storing the chunk
        alone and hoping is exactly the failure mode this API exists to avoid.

        The stub opens the prompt and carries the tokenizer's <bos>; encode the
        chunk with `add_special_tokens=False`, or it adds a second one.
        """
        mx, _ = _require_mlx()
        if self._stub_kv is None:
            raise RuntimeError(
                "call set_shared_stub(...) before ingest_pack(...) — a pack "
                "encoded without the stub cannot be composed with one."
            )
        full = self._stub_ids + list(chunk_token_ids)
        per_layer_full = self._encode_to_mlx_kv(full)
        s = self._stub_len
        per_layer = []
        for li in range(self.n_layers):
            k, v = per_layer_full[li]
            per_layer.append((k[:, :, s:, :], v[:, :, s:, :]))
        self._inproc_kv[chunk_id] = per_layer
        entry = {
            "namespace": self.chunk_namespace(chunk_id),
            "tokens": len(chunk_token_ids),
            "backend": self.backend,
            "encoded_at": s,
        }
        self._manifest[chunk_id] = entry
        self._persist()
        self.ingest_count += 1
        return entry

    def prepare_multi(
        self,
        chunk_ids: list[str],
        suffix_token_ids: list[int],
        max_packs: int = 8,
        allow_fine_grain: bool = False,
    ):
        """Compose several packs into one cache: stub, then each pack
        delta-rotated to its slot, then the suffix at the end.

        Returns (cache, suffix_token_ids), same contract as prepare().
        """
        mx, make_prompt_cache = _require_mlx()
        if self._stub_kv is None:
            raise RuntimeError("call set_shared_stub(...) first")
        if not chunk_ids:
            raise ValueError("chunk_ids is empty")
        if len(chunk_ids) > max_packs and not allow_fine_grain:
            raise ValueError(
                f"{len(chunk_ids)} packs requested, max_packs={max_packs}. "
                "Composition quality falls off with pack count and distractor "
                "density: measured 15/27 direct recall at 20 packs vs 26/27 for "
                "one mono cartridge of the same corpus, from cross-pack "
                "distractor collisions that the shared stub does not address. "
                "Use fewer, larger packs, or a mono build. Pass "
                "allow_fine_grain=True to override deliberately."
            )

        cache = make_prompt_cache(self.model)
        per_layer_k: list[list] = [[] for _ in range(self.n_layers)]
        per_layer_v: list[list] = [[] for _ in range(self.n_layers)]

        for li in range(self.n_layers):
            k, v = self._stub_kv[li]
            per_layer_k[li].append(k)
            per_layer_v[li].append(v)

        pos = self._stub_len
        for cid in chunk_ids:
            packed = self._inproc_kv.get(cid)
            if packed is None:
                self.hydrate_miss_count += 1
                raise KeyError(
                    f"pack {cid!r} not in this process. ingest_pack() it first."
                )
            entry = self._manifest.get(cid, {})
            encoded_at = entry.get("encoded_at", self._stub_len)
            delta = pos - encoded_at
            n_tok = entry.get("tokens") or packed[0][0].shape[2]
            for li in range(self.n_layers):
                k, v = packed[li]
                per_layer_k[li].append(self._rotate_k(k, delta))
                per_layer_v[li].append(v)          # V carries no positional term
            pos += n_tok

        for li in range(self.n_layers):
            k_all = mx.concatenate(per_layer_k[li], axis=2)
            v_all = mx.concatenate(per_layer_v[li], axis=2)
            mx.eval(k_all, v_all)
            set_slot_arrays(cache[li], (k_all, v_all), offset=pos)
        self.hydrate_count += 1
        return cache, suffix_token_ids

    def evict(self, chunk_id: str) -> bool:
        """Drop a chunk from the in-process dict (no-op for pion backend; use
        Pion's own eviction). Returns True if anything was evicted."""
        out = False
        if chunk_id in self._inproc_kv:
            del self._inproc_kv[chunk_id]
            out = True
        if chunk_id in self._manifest:
            del self._manifest[chunk_id]
            self._persist()
            out = True
        return out

    def stats(self) -> dict:
        s = {
            "backend": self.backend,
            "ingest_count": self.ingest_count,
            "hydrate_count": self.hydrate_count,
            "hydrate_miss_count": self.hydrate_miss_count,
            "chunks_in_manifest": len(self._manifest),
            "chunks_in_inproc": len(self._inproc_kv),
        }
        if self._pion is not None:
            inner = self._pion.stats()
            s["pion_fetch_ms_total"] = inner["fetch_ms_total"]
            s["pion_store_ms_total"] = inner["store_ms_total"]
        return s
