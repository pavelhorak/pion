"""PionPromptCache — drop-in replacement for mlx_lm.models.cache.make_prompt_cache.

For each fresh session, behaves like a regular MLX prompt cache. Once a prefix
has been seen and registered with Pion, the next session for the same namespace
fetches K/V from Pion instead of recomputing prefill.

Per the shared-KV-cache design §13.3 (Stage 1 ship) and §15.4 (PASS verdict
on fp16 with this exact wire pattern).

Usage:
    from pion_vllm_mlx.prompt_cache import PionPromptCache
    from mlx_lm import load

    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    pc = PionPromptCache(model, vquant="fp16")

    # First request: prefill + register + store
    cache = pc.get_or_prefill(prompt_ids, namespace="my_app|v1|llama321b|fp16|prompt_a")
    out = model(suffix_ids, cache=cache)

    # Second request, same prefix: fetched from Pion
    cache = pc.get_or_prefill(prompt_ids, namespace="my_app|v1|llama321b|fp16|prompt_a")
    out = model(suffix_ids, cache=cache)  # same fast path

The namespace key is the contract — the caller must encode every load-bearing
piece of execution context (model | tokenizer | rope_theta | quant | adapter |
prompt). PionPromptCache hashes it before sending to Pion.
"""
from __future__ import annotations

import hashlib
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pion_vllm_mlx._compat import set_slot_arrays, slot_arrays


# gh #50: Binary protocol opcodes mirrored from src/network/binary_protocol.mojo.
# Used by the Stage-2 fast lane in attend_query_fused — skips RESP framing on
# send (no $N\r\n per blob) and the RESP _read_one() bytes alloc on receive.
_BINARY_MAGIC = 0xCA5E
_CMD_PING = 0xFF
_CMD_ATTEND_PREFIX_QUERY_FUSED = 0x24
_BSTATUS_OK = 0x00
_BSTATUS_MISS = 0x01
_BSTATUS_ERROR = 0x02
# gh #67: distinct status for cold-tier demotion. Maps to the RESP lane's
# `-COLDMISS` reply — client warms + retries on the RESP path.
_BSTATUS_COLDMISS = 0x03


class _BinarySock:
    """0xCA5E-framed binary client to Pion's port+1 listener (--kvcache only).

    Single connection, one inflight request at a time. Used by
    PionPromptCache.attend_query_fused() to avoid RESP framing tax on the
    Stage-2 hot path (gh #50). Falls back transparently to the RESP path if
    the binary listener is unreachable or returns a non-OK status.
    """

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.sock.settimeout(60)
        self.sock.connect((host, port))
        # Sanity-check the listener — server returns OK with empty body for
        # CMD_PING. If we connected to a non-binary listener (e.g. user
        # passed the RESP port by mistake), the magic mismatch surfaces here.
        status, body = self._send_recv(_CMD_PING, ())
        if status != _BSTATUS_OK:
            raise ConnectionError(f"binary PING returned status={status}")

    def _recv_exactly(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            r = self.sock.recv_into(view[got:], n - got)
            if r == 0:
                raise ConnectionError("pion binary closed")
            got += r
        return bytes(buf)

    def _send_recv(self, cmd: int, body_parts) -> tuple[int, bytes]:
        # Each part may be `bytes` (sid, packed header, sentinel empty bytes)
        # or a numpy ndarray. Use memoryview to extract the actual byte count
        # uniformly — `len(ndarray)` returns the first-axis size, not nbytes.
        mvs = [memoryview(p).cast("B") for p in body_parts]
        body_len = sum(m.nbytes for m in mvs)
        header = struct.pack("<HBI", _BINARY_MAGIC, cmd, body_len)
        # Single-syscall scatter-gather send when the platform supports it
        # (Linux + macOS do). Falls back to one concatenated sendall — note
        # this allocates a temporary bytes object the size of the request,
        # so on the fallback path the per-call cost moves with body_len.
        try:
            self.sock.sendmsg([header, *mvs])
        except (AttributeError, OSError):
            joined = bytearray(7 + body_len)
            joined[:7] = header
            off = 7
            for m in mvs:
                joined[off:off + m.nbytes] = m
                off += m.nbytes
            self.sock.sendall(bytes(joined))
        resp_hdr = self._recv_exactly(7)
        magic, status, blen = struct.unpack("<HBI", resp_hdr)
        if magic != _BINARY_MAGIC:
            raise RuntimeError(f"binary frame magic mismatch: 0x{magic:04X}")
        body = self._recv_exactly(blen) if blen > 0 else b""
        return status, body

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def _require_mlx():
    """Import MLX lazily so `from pion_vllm_mlx import PionPromptCache` works
    on Linux where mlx is unavailable. Wire-protocol surface (lookup,
    attend_query, raw KV.PREFIX.*) does not need MLX; only tensor-packaging
    methods (get_or_prefill, _cache_to_arrays, _arrays_to_cache,
    _stage2_push_cold) do. Those call this on entry.

    Returns (mlx.core module, make_prompt_cache callable).
    """
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as e:
        raise ImportError(
            "this method requires Apple MLX (mlx + mlx-lm). "
            "Install with `pip install mlx mlx-lm` on Apple Silicon. "
            "On Linux, use raw KV.PREFIX.* / V.* commands via redis-py "
            "instead of PionPromptCache's tensor helpers."
        ) from e
    return mx, make_prompt_cache


@dataclass
class CacheLayout:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    # gh #59: hybrid (Gemma 4 / Mistral SWA / Qwen3.5) architectures have
    # per-layer kv_dim variation, KV-shared layers (no own cache), and/or
    # sliding-window cache layers. is_hybrid=True forces the in-process lane
    # only and disables the wire path (V.STOREBATCH / V.FETCH assume uniform
    # kv_dim). cache_len is the model's actual cache list length, which can
    # be < n_layers when KV-shared layers are present.
    is_hybrid: bool = False
    cache_len: int = 0

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def _layout_from(model) -> CacheLayout:
    # Hybrid models like Qwen3.5-4B nest the transformer config one level
    # down (`model.args.text_config` is a dict). Fall through to it when the
    # top-level args is missing the standard attributes. Same pattern works
    # for any future model that wraps a text config.
    a = model.args
    text_cfg = getattr(a, "text_config", None)
    def _g(key: str, default=None):
        v = getattr(a, key, None)
        if v is not None:
            return v
        if isinstance(text_cfg, dict):
            return text_cfg.get(key, default)
        return default
    n_attn = _g("num_attention_heads")
    n_kv = _g("num_key_value_heads", n_attn)
    head_dim = _g("head_dim")
    if head_dim is None:
        # Some hybrids (e.g. Nemotron-H) specify the attention head dim under
        # `attention_head_dim` and it is NOT hidden_size // num_attention_heads
        # (Nemotron-H-4B: 128, not 3072//32=96). Read it before the fallback,
        # else the softmax-half K/V reshape on fetch is wrong for GQA.
        head_dim = _g("attention_head_dim")
    if head_dim is None:
        head_dim = _g("hidden_size") // n_attn if n_attn else 0
    n_layers = _g("num_hidden_layers", 0)
    # gh #59: detect hybrid architectures — non-empty mixed `layer_types`,
    # KV-shared layers, or per-layer-varying head_dim (`global_head_dim`
    # differing from `head_dim`). When hybrid, capture the model's actual
    # cache list length so the in-process lane iterates correctly.
    layer_types = _g("layer_types") or []
    n_kv_shared = _g("num_kv_shared_layers", 0) or 0
    global_head_dim = _g("global_head_dim")
    has_mixed_types = len(set(layer_types)) > 1 if layer_types else False
    has_split_head_dim = (
        global_head_dim is not None and global_head_dim != head_dim
    )
    is_hybrid = bool(has_mixed_types or n_kv_shared > 0 or has_split_head_dim)
    cache_len = n_layers
    if is_hybrid:
        # The outer mlx-lm Model class exposes make_cache() for hybrid archs
        # (Gemma 4 / Mistral SWA / Qwen3.5) — it's the source of truth for
        # cache list length, accounting for KV-shared layers that have no own
        # cache slot. Fall back to (n_layers - n_kv_shared) if make_cache
        # isn't reachable for some reason.
        try:
            if hasattr(model, "make_cache"):
                cache_len = len(model.make_cache())
            else:
                cache_len = n_layers - n_kv_shared if n_kv_shared > 0 else n_layers
        except Exception:
            cache_len = n_layers - n_kv_shared if n_kv_shared > 0 else n_layers
    return CacheLayout(
        n_layers=n_layers,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        is_hybrid=is_hybrid,
        cache_len=cache_len,
    )


def _require_layout(layout: "CacheLayout | None"):
    if layout is None:
        raise RuntimeError(
            "this method requires a model — PionPromptCache was constructed "
            "with model=None for wire-only Stage-2 use (attend_store_layer / "
            "attend_query / attend_drop / lookup). Pass `model=<mlx-lm model>` "
            "to enable get_or_prefill / cache reconstruction."
        )
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# Minimal RESP client (only the commands PionPromptCache needs)
# ─────────────────────────────────────────────────────────────────────────────


class _RESP:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
        self.sock.settimeout(60)
        self.sock.connect((host, port))
        self.buf = b""

    @staticmethod
    def _encode(parts) -> bytes:
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, bytes):
                out.append(f"${len(p)}\r\n".encode())
                out.append(p)
                out.append(b"\r\n")
            else:
                s = str(p)
                out.append(f"${len(s)}\r\n{s}\r\n".encode())
        return b"".join(out)

    def _read_one(self) -> bytes:
        while True:
            done = self._first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.sock.recv(64 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed")
            self.buf += chunk

    @staticmethod
    def _first_complete(d: bytes):
        if len(d) < 3:
            return None
        p = d[0:1]
        nl = d.find(b"\r\n")
        if nl < 0:
            return None
        if p in (b"+", b"-", b":"):
            return nl + 2
        if p == b"$":
            ls = d[1:nl].decode()
            if ls == "-1":
                return nl + 2
            n = int(ls)
            need = nl + 2 + n + 2
            return need if len(d) >= need else None
        return nl + 2

    def call(self, *parts):
        self.sock.sendall(self._encode(parts))
        return self._read_one()

    def call_array(self, *parts):
        """Send a command and parse a RESP array reply: returns a list whose
        elements are either `bytes` (bulk string) or `None` ($-1 nil).
        Raises on `-ERR` and on malformed replies. Used by V.FETCH BATCH and
        any other multi-element response we add later.

        Uses sock.makefile('rb') for efficient incremental reads — naive
        bytes-slicing into self.buf was O(N²) on multi-MB array replies and
        cost more than the wire round-trips it was meant to save (V.FETCH
        BATCH at prefix=512: 32MB reply, observed +26% slower than the
        per-layer V.FETCH RANGE legacy until this fix).

        Pre-condition: self.buf must be empty (no half-consumed bytes from
        a prior `call`). PionPromptCache only mixes `call` and `call_array`
        across distinct fetches, never within one, so this is safe today.
        Asserted to catch future misuse.
        """
        if self.buf:
            # Drain pre-existing single-frame data before switching to the
            # buffered-reader model so the two paths don't fight over bytes.
            raise RuntimeError(
                f"call_array invoked with {len(self.buf)} bytes pending in buf — "
                "interleaved call/call_array would corrupt the wire stream"
            )
        self.sock.sendall(self._encode(parts))
        rd = self.sock.makefile("rb", buffering=64 * 1024 * 1024)
        try:
            hdr = rd.readline()
            if not hdr:
                raise RuntimeError("empty reply")
            if hdr.endswith(b"\r\n"):
                hdr = hdr[:-2]
            t = hdr[:1]
            if t == b"-":
                raise RuntimeError(hdr[1:].decode("utf-8", errors="replace"))
            if t != b"*":
                raise RuntimeError(f"expected RESP array, got {hdr[:80]!r}")
            n = int(hdr[1:])
            if n < 0:
                return None
            out: list = []
            for _ in range(n):
                elem_hdr = rd.readline()
                if elem_hdr.endswith(b"\r\n"):
                    elem_hdr = elem_hdr[:-2]
                if not elem_hdr or elem_hdr[:1] != b"$":
                    raise RuntimeError(f"unexpected array element: {elem_hdr[:80]!r}")
                blen = int(elem_hdr[1:])
                if blen < 0:
                    out.append(None)
                    continue
                data = rd.read(blen)
                if len(data) != blen:
                    raise ConnectionError(f"short read: got {len(data)} of {blen} bytes")
                trailer = rd.read(2)  # \r\n
                if trailer != b"\r\n":
                    raise RuntimeError(f"missing trailing CRLF after bulk: {trailer!r}")
                out.append(data)
            return out
        finally:
            # Must not detach() — that would steal the underlying socket.
            # Just let `rd` go out of scope; any further bytes from the
            # server land in self.buf via the next sock.recv().
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Prompt cache
# ─────────────────────────────────────────────────────────────────────────────


class PionPromptCache:
    """Drop-in replacement for make_prompt_cache that backs onto Pion's V-store.

    The first time a namespace is seen, the prefix is prefilled locally and
    pushed to Pion via KV.PREFIX.REGISTER + V.STOREBATCH per layer. Subsequent
    sessions with the same namespace fetch K/V via V.FETCH RANGE and skip
    prefill entirely.

    Multi-worker deployments
    ------------------------
    Pion's V-store is per-worker (no shared mmap), so a multi-worker
    pion-server fragments the cache. The canonical pattern is to run two
    pion-server instances:

        ./pion-server -w N                      # primary KV/vector, e.g. :1974
        ./pion-server --kvcache -w 1 -p 1984    # shared KV cache only

    Then construct PionPromptCache pointing at the KV cache instance:

        pc = PionPromptCache(model, vquant="fp16", port=1984)

    pion-server with --kvcache automatically caps itself at -w 1 since 0.501.
    """

    def __init__(
        self,
        model=None,
        vquant: str = "fp16",
        host: str = "127.0.0.1",
        port: int = 1974,
        stage2: bool = False,
        admission_threshold: int = 1,
        boundary_protect: int = 0,
        prefill_chunk_size: int | None = None,
        softmax_bitexact: bool = False,
    ) -> None:
        """If stage2=True, the cold path also pushes K/V to the MLX sidecar
        via ATTEND.PREFIX.STORE so subsequent attention queries can run on
        Pion (Q-only on the wire, no K/V re-fetch). Stage 1 path is unchanged
        — mlx-lm still consumes the returned MLX cache. Stage 2 is reached
        through .attend_query() / .attend_drop().

        IMPORTANT: stage2=True is bridge plumbing, not an mlx-lm speedup.
        mlx-lm's model(suffix, cache=cache) computes attention itself on the
        K/V tensors in the MLX cache — it does not consume external attention
        output. For a standalone Mac + mlx-lm user calling only get_or_prefill():
            cold path:  slightly slower (extra ATTEND.PREFIX.STORE per layer)
            warm path:  unchanged
            net latency change: zero, slightly negative on first request

        Stage 2 produces a measurable end-to-end win only when something
        downstream of get_or_prefill() actually calls attend_query():
            - pion-exo's gpu_attention mode (Mac cluster inference)
            - a custom vLLM CacheEngine (Linux GPU)
            - monkey-patched mlx-lm attention layer (single-Mac, measurable)

        See doc/shared_kv_cache.md "Honest caveat" for the full accounting.

        Requires `pion-server --kvcache --metal-attention -w 1`.

        admission_threshold:
          Number of times a namespace must be observed before its K/V is
          actually pushed to Pion. Default 1 means "cache on first miss"
          (current behaviour, matches §22 measurements). Set to 2 to skip
          one-off prefixes — prevents single-shot prompts from evicting
          long-lived hot prefixes under LRU pressure. Recommended for
          multi-tenant workloads with a long tail of unique prompts.

        boundary_protect:
          When > 0, the K-side and V-side V-store sessions are created via
          `V.CREATE … SCHEMA` (per-layer formats) instead of the legacy
          uniform `KV.PREFIX.REGISTER`. K stays fp16 across all layers (K
          drives softmax routing — keep it conservative). V uses fp16 for the first `boundary_protect`
          + last `boundary_protect` layers and the user's `vquant` for the
          middle. Reduces V-side storage when `vquant` is fp8/turbo4 while
          keeping the boundary layers' precision intact.

          Trade-off: skips KV.PREFIX.REGISTER's cross-worker directory
          publish — the SCHEMA path is single-worker visibility only. With
          --kvcache forcing `-w 1` today, that's a no-op. A future
          `KV.PREFIX.REGISTER.SCHEMA` server command would lift this.

          Default 0 = uniform vquant across all layers (current behavior,
          backward compatible).
        """
        if boundary_protect > 0 and vquant in {"fp8", "turbo4", "turbo3", "turbo2"}:
            # These formats require dim % 32 == 0; validated upfront so the
            # caller doesn't see a confusing -ERR mid-flight. boundary_protect
            # is only meaningful when a model is provided (it routes per-layer
            # vquants), so the check requires model.
            if model is None:
                raise ValueError(
                    "boundary_protect>0 requires a model — pass `model=<mlx-lm model>`"
                )
            cl = _layout_from(model)
            if cl.kv_dim % 32 != 0:
                raise ValueError(
                    f"boundary_protect={boundary_protect} with vquant={vquant!r} "
                    f"requires kv_dim ({cl.kv_dim}) divisible by 32"
                )
        self.model = model
        self.vquant = vquant
        # Wire-only Stage-2 consumers (e.g. pion-exo gpu_attention) construct
        # without a model — layout-dependent paths raise via _require_layout.
        self.layout = _layout_from(model) if model is not None else None
        self.host = host
        self.port = port
        self.stage2 = stage2
        self.admission_threshold = max(1, int(admission_threshold))
        self.boundary_protect = max(0, int(boundary_protect))
        # softmax_bitexact: route the softmax (attention) anchor slots through the
        # opaque SSM.PREFIX safetensors path (bit-exact, native dtype) instead of
        # the fp16 V-store. Needed for extreme hybrids (e.g. Nemotron-H: 4 softmax
        # anchors carry ALL prefix retrieval, so the fp16 V-store ceiling breaks
        # decode). Default False keeps the structured V-store path (Qwen3.5).
        self.softmax_bitexact = bool(softmax_bitexact)
        # gh #60 Step 2 unblock: chunked cold prefill. When set, the cold path
        # in get_or_prefill() forwards the prefix in slices of this many tokens
        # with mx.eval between slices, so the lazy graph + peak attention buffer
        # stay O(chunk·N_kv) instead of O(N²). Needed to profile/decode at 64K+
        # on M4 16GB; smaller machines fit dense at 16K-24K. None = single-shot.
        self.prefill_chunk_size: int | None = (
            int(prefill_chunk_size) if prefill_chunk_size else None
        )
        self.resp = _RESP(host, port)
        # gh #50: Stage-2 binary fast lane on port+1. Lazy-opened on first
        # attend_query_fused() call; if the connect/PING fails the binary
        # path is permanently disabled for this PionPromptCache and the RESP
        # path stays in service. PION_PROMPT_CACHE_NO_BINARY=1 forces RESP
        # for clean A/B benchmarking.
        import os as _os
        self._binary: _BinarySock | None = None
        self._binary_disabled = (_os.environ.get("PION_PROMPT_CACHE_NO_BINARY", "0") == "1")
        self._binary_attempted = False
        # gh #50 in-process fast lane. When the cold prefill happens in the
        # same Python process that will later serve the warm forwards (the
        # bench harness, single-process inference servers), keeping the
        # prefix K/V as MLX arrays here lets the patched SDPA run attention
        # locally via mx.fast.scaled_dot_product_attention — no wire
        # roundtrip, no mx.eval barrier, no numpy↔MLX hop. Cross-process
        # consumers (different Python interpreter, container boundary)
        # continue using the wire path because they have no reference to
        # this dict. Disabled with PION_PROMPT_CACHE_NO_INPROC=1 for
        # benchmarking the wire-only path.
        # Layout: dict[namespace] -> list[(K_mx, V_mx) per layer]; each
        # entry has shape (1, n_kv_heads, prefix_len, head_dim).
        self._mlx_prefix_kv: dict = {}
        self._inproc_disabled = (_os.environ.get("PION_PROMPT_CACHE_NO_INPROC", "0") == "1")
        # gh #468 export lane: a server on this machine writes the prefix as a
        # safetensors file and the hit maps it (mx.load), instead of copying
        # it over loopback TCP. Off for a remote host, when the server is too
        # old to know V.EXPORT, and with PION_PROMPT_CACHE_NO_EXPORT=1.
        self._export_disabled = (
            _os.environ.get("PION_PROMPT_CACHE_NO_EXPORT", "0") == "1"
            or host not in ("127.0.0.1", "localhost", "::1"))
        # local manifest cache: ns_hash -> (prefix_len,)
        self._local: dict[str, int] = {}
        self._stage2_pushed: set[str] = set()
        # admission tracking — counts observed misses per namespace before push
        self._observed: dict[str, int] = {}
        # stats
        self.hits = 0
        self.misses = 0
        self.admission_skips = 0  # misses that returned a cache without pushing to Pion
        self.fetch_ms_total = 0.0
        self.store_ms_total = 0.0
        # gh #262: wall time of the most recent local cold prefill, reported
        # to the server on KV.PREFIX.REGISTER as PREFILL_MS so PION.STATS /
        # INFO credit later hits with a MEASURED number, not an estimate.
        self.last_prefill_ms = 0.0
        self.attend_query_ms_total = 0.0
        self.attend_query_calls = 0
        # V.FETCH BATCH adoption (W1.4) — tries the multi-layer primitive on
        # every fetch_to_cache; falls back to the per-layer V.FETCH RANGE loop
        # one-shot if the server doesn't support it (older Pion build) or if
        # a parse error abandons unread bytes in the connection.
        # Env override `PION_PROMPT_CACHE_NO_BATCH=1` forces the legacy path —
        # used by tests/bench_ttft.py for clean A/B measurement.
        import os as _os
        self._batch_disabled = (_os.environ.get("PION_PROMPT_CACHE_NO_BATCH", "0") == "1")

    @staticmethod
    def make_namespace(*parts: str) -> str:
        """Build a canonical namespace key from execution-context fragments."""
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]

    def lookup(self, namespace: str) -> bool:
        """Check whether the namespace's K/V is hot on the server.

        Two states need to agree for the cache to be HIT-usable:
          1. V-store registration (KV.PREFIX.LOOKUP) — used by Stage-1 fetch path.
          2. ATTEND.PREFIX.* Metal session cache (ATTEND.PREFIX.LOOKUP) —
             used by Stage-2 wire-mode consumers (wire-lane sparse
             auto), which read from the Metal cache, NOT V-store.

        Pre-gh-#65-follow-on this checked only (1); a stale V-store hit while
        the Metal cache was cold caused `_stage2_push_cold` to skip on
        re-instantiated PionPromptCache objects (test runs, server restarts).
        Now: when `stage2=True`, BOTH must hit. The probe on (2) checks
        layer 0 of the attend session; full-layer audit would over-spend
        wire bandwidth and is unnecessary since `_stage2_push_cold` writes
        all layers atomically.
        """
        r = self.resp.call("KV.PREFIX.LOOKUP", namespace)
        kv_hit = r.startswith(b"+HIT")
        if not kv_hit:
            return False
        if not self.stage2:
            return True
        sid = self._attend_session(namespace)
        r2 = self.resp.call("ATTEND.PREFIX.LOOKUP", sid, "0")
        return r2.startswith(b"+HIT")

    def _register(self, namespace: str) -> None:
        layout = _require_layout(self.layout)
        if self.boundary_protect > 0:
            self._register_schema(namespace)
            return
        args = ["KV.PREFIX.REGISTER", namespace, str(layout.kv_dim), self.vquant]
        if self.last_prefill_ms > 0:
            # gh #262: the server's value receipt credits every later hit on
            # this prefix with exactly this measured time. Servers predating
            # PREFILL_MS ignore trailing tokens, so this is safe to send always.
            args += ["PREFILL_MS", str(int(round(self.last_prefill_ms)))]
        r = self.resp.call(*args)
        if not r.startswith(b"+OK"):
            raise RuntimeError(f"KV.PREFIX.REGISTER failed: {r!r}")

    def _register_schema(self, namespace: str) -> None:
        """boundary-protect path: emit two raw V.CREATE … SCHEMA calls instead
        of KV.PREFIX.REGISTER. K stays fp16 across all layers (K drives
        softmax routing). V uses fp16 for the
        boundary layers (first/last `boundary_protect` each) and self.vquant
        for the middle layers."""
        layout = _require_layout(self.layout)
        N = layout.n_layers
        bp = self.boundary_protect
        kv_dim = layout.kv_dim
        # K-side specs: all fp16.
        k_specs = [f"fmt=fp16,dim={kv_dim}"] * N
        # V-side specs: fp16 boundary, vquant middle.
        v_specs = []
        for li in range(N):
            is_boundary = li < bp or li >= (N - bp)
            fmt = "fp16" if is_boundary else self.vquant
            v_specs.append(f"fmt={fmt},dim={kv_dim}")
        # InlineArray[RESP3Token, 64] limit on the server: cmd + sid + dim
        # + SCHEMA + N + N specs = 5 + N. With Llama-1B's 16 layers we use
        # 21 tokens; ample headroom up to N=59.
        if N + 5 > 64:
            raise ValueError(
                f"boundary_protect requires n_layers ({N}) ≤ 59 — "
                f"split into multiple sessions or wait for V.CREATE.SCHEMA.APPEND"
            )
        sid_k = f"{namespace}_pk"
        sid_v = f"{namespace}_pv"
        # default_dim = 0 sentinel; every layer carries explicit dim.
        r = self.resp.call("V.CREATE", sid_k, "0", "SCHEMA", str(N), *k_specs)
        if not r.startswith(b":"):
            raise RuntimeError(f"V.CREATE SCHEMA (K-side) failed: {r!r}")
        r = self.resp.call("V.CREATE", sid_v, "0", "SCHEMA", str(N), *v_specs)
        if not r.startswith(b":"):
            raise RuntimeError(f"V.CREATE SCHEMA (V-side) failed: {r!r}")

    def _storebatch(self, sid: str, layer: int, values_fp32: np.ndarray) -> None:
        n = values_fp32.shape[0]
        r = self.resp.call(
            "V.STOREBATCH", sid, str(layer), "0", str(n),
            np.ascontiguousarray(values_fp32, dtype=np.float32).tobytes(),
        )
        if not r.startswith(b"+OK"):
            raise RuntimeError(f"V.STOREBATCH failed: {r!r}")

    def _fetch_range(self, sid: str, layer: int, prefix_len: int,
                     native_dim: int | None = None) -> np.ndarray:
        """gh #193: with native_dim set, request FMT NATIVE — fp16-stored
        layers come back as raw fp16 (half the wire bytes). The reply dtype is
        decided by payload length, so quantized layers and pre-#193 servers
        (which ignore the trailing tokens and reply fp32) decode correctly
        with no capability negotiation."""
        args = ["V.FETCH", sid, str(layer), "RANGE", "0", str(prefix_len)]
        if native_dim:
            args += ["FMT", "NATIVE"]
        r = self.resp.call(*args)
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            raise RuntimeError(f"V.FETCH RANGE failed: {r[:80]!r}")
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        payload = r[nl + 2:nl + 2 + blen]
        if native_dim and blen == prefix_len * native_dim * 2:
            return np.frombuffer(payload, dtype=np.float16).copy()
        return np.frombuffer(payload, dtype=np.float32).copy()

    def _fetch_batch_all_layers(self, sid: str, prefix_len: int, num_layers: int,
                                native_dim: int | None = None):
        """Fetch all `num_layers` layers' K (or V) tensors for `sid` in ONE
        wire round-trip, replacing N×_fetch_range loop. Server: V.FETCH BATCH
        in src/commands/v_store.mojo. Returns list[np.ndarray] of length
        `num_layers`, each shape (prefix_len, kv_dim) flat. Raises on any
        unsupported-command / malformed-array reply so the caller can fall
        back to the per-layer path on older servers."""
        args = ["V.FETCH", sid, "BATCH", "0", str(prefix_len), str(num_layers)]
        if native_dim:
            # gh #193: fp16-stored layers reply raw fp16 (half the wire).
            # Decode dtype per payload by length — quantized layers and
            # pre-#193 servers reply fp32 and decode correctly with no
            # capability negotiation.
            args += ["FMT", "NATIVE"]
        replies = self.resp.call_array(*args)
        if replies is None or len(replies) != num_layers:
            raise RuntimeError(f"V.FETCH BATCH count mismatch: got {len(replies) if replies else None}, expected {num_layers}")
        out: list[np.ndarray] = []
        for li, payload in enumerate(replies):
            if payload is None:
                raise RuntimeError(f"V.FETCH BATCH layer {li} nil")
            if native_dim and len(payload) == prefix_len * native_dim * 2:
                out.append(np.frombuffer(payload, dtype=np.float16).copy())
            else:
                out.append(np.frombuffer(payload, dtype=np.float32).copy())
        return out

    # ── Cache-extraction helpers ────────────────────────────────────────────

    def _cache_to_arrays(self, cache, prefix_len: int):
        mx, _ = _require_mlx()
        L = _require_layout(self.layout)
        out = []
        for li in range(L.n_layers):
            k = cache[li].keys[:, :, :prefix_len, :]
            v = cache[li].values[:, :, :prefix_len, :]
            k_np = np.array(k.transpose(0, 2, 1, 3).reshape(prefix_len, L.kv_dim).astype(mx.float32))
            v_np = np.array(v.transpose(0, 2, 1, 3).reshape(prefix_len, L.kv_dim).astype(mx.float32))
            out.append((k_np.copy(), v_np.copy()))
        return out

    def _arrays_to_cache(self, per_layer, prefix_len: int):
        mx, make_prompt_cache = _require_mlx()
        L = _require_layout(self.layout)
        cache = make_prompt_cache(self.model)
        for li in range(L.n_layers):
            k_np, v_np = per_layer[li]
            k_mx = mx.array(
                k_np.reshape(prefix_len, L.n_kv_heads, L.head_dim).transpose(1, 0, 2)[None, ...],
                dtype=mx.float16,
            )
            v_mx = mx.array(
                v_np.reshape(prefix_len, L.n_kv_heads, L.head_dim).transpose(1, 0, 2)[None, ...],
                dtype=mx.float16,
            )
            cache[li].update_and_fetch(k_mx, v_mx)
        return cache

    # ── Stage 2 helpers (ATTEND.PREFIX.*) ───────────────────────────────────

    @staticmethod
    def _attend_session(namespace: str) -> str:
        """Distinct namespace for the MLX-resident K/V (so Stage 2 storage
        doesn't collide with Stage 1 V-store sids `<ns>_pk` / `<ns>_pv`)."""
        return f"{namespace}_attn"

    def _warm_namespace(self, namespace: str, H: int, D: int) -> int:
        """gh #67: trigger server-side cold-tier rehydrate.

        Issues `KV.PREFIX.WARM <ns> H D <ns>_attn` so the server walks the
        V-store sessions for `<ns>_pk` / `<ns>_pv`, dequantizes per layer,
        transposes into [H,N,D], and re-pushes each layer into the Metal
        SDPA cache under sid `<ns>_attn` (matching `_attend_session`).
        Returns the number of layers rehydrated (parsed from `+N`). On
        error (no V-store data, no Metal engine) returns 0 — caller can
        decide whether that's fatal or a stale-cache no-op.
        """
        sid_attn = self._attend_session(namespace)
        r = self.resp.call("KV.PREFIX.WARM", namespace, str(int(H)), str(int(D)), sid_attn)
        if not r.startswith(b"+"):
            return 0
        # `+N\r\n` or `+N skipped_dim=K\r\n`.
        try:
            tail = r[1: r.find(b"\r\n")].split(b" ", 1)[0]
            return int(tail)
        except (ValueError, IndexError):
            return 0

    def _attend_call_with_warm_retry(self, resp_args, namespace: str, H: int, D: int):
        """gh #67: wire-call wrapper that detects `-COLDMISS`, calls
        KV.PREFIX.WARM, and retries the original call once.

        The retry is bounded to one attempt — if a second COLDMISS happens
        (e.g. the table is so hot that rehydrated entries are demoted again
        between WARM and the retry), the caller sees the second response
        and decides what to do (typically: fall back to a fresh cold prefill).
        """
        r = self.resp.call(*resp_args)
        if r.startswith(b"-COLDMISS"):
            self.cold_rehydrates_observed = getattr(self, "cold_rehydrates_observed", 0) + 1
            n = self._warm_namespace(namespace, H, D)
            if n > 0:
                r = self.resp.call(*resp_args)
        return r

    def attend_store_layer(self, namespace: str, layer_id: int,
                            K: np.ndarray, V: np.ndarray) -> None:
        """ATTEND.PREFIX.STORE — push one layer's K/V to Pion's Metal SDPA.

        K, V shape: (H, N, D) float32. Pion holds these resident in
        MLX-side memory until DROP_SESSION or LRU eviction. This is the
        wire-level Stage-2 push; layout-free and model-free, suitable for
        consumers like pion-exo that compute K/V externally per layer.

        Requires `pion-server --kvcache --metal-attention -w 1`.
        """
        H, N, D = K.shape
        sid = self._attend_session(namespace)
        r = self.resp.call(
            "ATTEND.PREFIX.STORE", sid, str(layer_id), str(H), str(N), str(D),
            np.ascontiguousarray(K, dtype=np.float32).tobytes(),
            np.ascontiguousarray(V, dtype=np.float32).tobytes(),
        )
        if not r.startswith(b"+OK"):
            raise RuntimeError(f"ATTEND.PREFIX.STORE failed: {r[:80]!r}")

    # Backward-compatibility alias for the previously-private name.
    _attend_store_layer = attend_store_layer

    def attend_query(self, namespace: str, layer_id: int,
                      Q: np.ndarray, top_k: int = 0,
                      with_lse: bool = False,
                      fa_window: Optional[int] = None):
        """ATTEND.PREFIX.QUERY — Q-only on the wire, attention runs on Pion's MLX.

        Q shape:
          (H, D)    — single decode-step query, sparse top_k attention,
                       returns (H, D).
          (H, M, D) — batched query over M suffix tokens (Stage-2 monkey-patch
                       at TTFT time), dense softmax (top_k ignored sidecar-side),
                       returns (H, M, D).

        top_k=0 → use cached prefix length (or 1024 fallback). Ignored when M>1.

        with_lse=True (only valid for M>1): returns ((H, M, D) output, (H, M)
        rowwise log-sum-exp). The LSE is required by the mlx-lm monkey-patch's
        online-softmax merge between cached prefix attention and locally
        computed suffix attention. The wire trailer is always sent for M>1, but
        old callers that pass with_lse=False (default) get the legacy single
        ndarray and the trailer is parsed-and-dropped.

        gh #60 Step 1: `fa_window` (optional) overrides the server's
        --fa-window for this call only. Per-layer routing for hybrid SWA;
        see `attend_query_fused` for full notes.
        """
        if Q.ndim == 2:
            H, D = Q.shape
            M = 1
            out_shape = (H, D)
            if with_lse:
                raise ValueError("with_lse=True requires M>1 (Q.ndim==3)")
        elif Q.ndim == 3:
            H, M, D = Q.shape
            out_shape = (H, M, D)
        else:
            raise ValueError(f"attend_query: Q must be 2-D or 3-D, got {Q.shape}")
        if top_k <= 0:
            top_k = self._local.get(namespace, 0) or 1024
        sid = self._attend_session(namespace)
        # gh #60 Step 1: optional trailing positional fa_window (RESP).
        resp_args = ["ATTEND.PREFIX.QUERY", sid, str(layer_id), str(H), str(D), str(top_k),
                     np.ascontiguousarray(Q, dtype=np.float32).tobytes()]
        if fa_window is not None:
            resp_args.append(str(int(fa_window)))
        t0 = time.perf_counter()
        # gh #67: detect COLDMISS, warm + retry once.
        r = self._attend_call_with_warm_retry(resp_args, namespace, H, D)
        self.attend_query_ms_total += (time.perf_counter() - t0) * 1000
        self.attend_query_calls += 1
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            raise RuntimeError(f"ATTEND.PREFIX.QUERY failed: {r[:80]!r}")
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        body = r[nl + 2:nl + 2 + blen]
        out_bytes = H * M * D * 4
        out = np.frombuffer(body[:out_bytes], dtype=np.float32).copy().reshape(out_shape)
        if M > 1 and len(body) >= out_bytes + H * M * 4:
            lse = np.frombuffer(body[out_bytes:out_bytes + H * M * 4],
                                 dtype=np.float32).copy().reshape(H, M)
            if with_lse:
                return out, lse
        elif with_lse:
            raise RuntimeError("ATTEND.PREFIX.QUERY response missing LSE trailer")
        return out

    def attend_query_sparse_auto(self, namespace: str, layer_id: int,
                                  Q: np.ndarray, B: int, K_top: int,
                                  H_kv: Optional[int] = None,
                                  head_map: Optional[np.ndarray] = None,
                                  fa_window: Optional[int] = None,
                                  selector: str = "block_mean") -> np.ndarray:
        """gh #63 Phase 3b consumer: ATTEND.PREFIX.QUERY_SPARSE_AUTO.

        Server picks the top-K blocks via the chosen selector + runs the
        sparse SDPA kernel; only Q + B + K_top + head_map cross the wire
        (plus the H_q*D*4 attention output back). K/V never leaves the server.

        Q shape: (H_q, D) — M=1 only in v1.
        H_kv: kv head count (defaults to H_q on non-GQA).
        head_map: shape (H_q,) uint8 with head_map[h_q] = h_kv. Defaults to
                  identity when H_kv == H_q; required when H_kv < H_q.
        selector: "block_mean" (default, NIAH-class, gh #60 / gh #63) or
                  "quest" (Quest upper-bound, W11 Phase 2 — also recovers
                  factual QA; Metal-only as of this commit, CUDA support
                  is the W11 Phase 3 follow-up).
        Returns (H_q, D).
        """
        if Q.ndim != 2:
            raise ValueError(f"attend_query_sparse_auto: Q must be 2-D (H_q, D); got {Q.shape}")
        H_q, D = Q.shape
        if H_kv is None:
            H_kv = H_q
        if H_q % H_kv != 0:
            raise ValueError(f"H_q={H_q} must be a multiple of H_kv={H_kv}")
        # Build head_map: identity if H_kv == H_q and not given; else required.
        if head_map is None:
            if H_kv == H_q:
                hm_bytes = b""    # server synthesizes identity
            else:
                rep = H_q // H_kv
                hm = np.repeat(np.arange(H_kv, dtype=np.uint8), rep)
                hm_bytes = hm.tobytes()
        else:
            if head_map.shape != (H_q,):
                raise ValueError(f"head_map shape must be (H_q={H_q},); got {head_map.shape}")
            hm_bytes = np.ascontiguousarray(head_map, dtype=np.uint8).tobytes()
        selector_id = 0
        if selector == "quest":
            selector_id = 1
        elif selector not in ("block_mean", "block-mean", ""):
            raise ValueError(f"unknown selector {selector!r}; expected 'block_mean' or 'quest'")
        sid = self._attend_session(namespace)
        resp_args = ["ATTEND.PREFIX.QUERY_SPARSE_AUTO", sid, str(layer_id),
                     str(H_q), str(D), str(int(B)), str(int(K_top)), str(int(H_kv)),
                     np.ascontiguousarray(Q, dtype=np.float32).tobytes(),
                     hm_bytes]
        # W11 Phase 2: selector_id is the 11th positional, after fa_window.
        # When selector != block_mean but fa_window is unset, we must still
        # emit a fa_window placeholder (-1 = use engine default) so the
        # selector_id lands in the right slot.
        if fa_window is not None or selector_id != 0:
            resp_args.append(str(int(fa_window) if fa_window is not None else -1))
        if selector_id != 0:
            resp_args.append(str(selector_id))
        t0 = time.perf_counter()
        # gh #67: detect COLDMISS, warm + retry once.
        r = self._attend_call_with_warm_retry(resp_args, namespace, H_q, D)
        self.attend_query_ms_total += (time.perf_counter() - t0) * 1000
        self.attend_query_calls += 1
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            import os as _osd
            if _osd.environ.get("PION_DEBUG_SPARSE_AUTO"):
                print(f"[debug] sparse_auto failed: ns={namespace!r} sid={sid!r} layer={layer_id} H_q={H_q} D={D} B={B} K_top={K_top} H_kv={H_kv} hm_len={len(hm_bytes)} fa={fa_window} r={r[:160]!r}")
            raise RuntimeError(f"ATTEND.PREFIX.QUERY_SPARSE_AUTO failed: {r[:80]!r}")
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        body = r[nl + 2:nl + 2 + blen]
        return np.frombuffer(body[: H_q * D * 4], dtype=np.float32).copy().reshape(H_q, D)

    def attend_query_sparse_auto_fused(self, namespace: str, layer_id: int,
                                        Q: np.ndarray, B: int, K_top: int,
                                        K_suf: np.ndarray, V_suf: np.ndarray,
                                        H_kv: Optional[int] = None,
                                        head_map: Optional[np.ndarray] = None,
                                        fa_window: Optional[int] = None,
                                        selector: str = "block_mean") -> np.ndarray:
        """gh #63 follow-on: sparse-AUTO + dense-suffix in one wire call.

        Server picks the top-K blocks via the selector (block-mean default;
        Quest UB on W11 Phase 2 via selector="quest") on the resident K/V,
        then runs sparse-prefix attention AND dense-suffix attention over
        the caller's K_suf/V_suf, merging via online softmax in the kernel.
        Returns merged (H_q, D) attention output. No client-side suffix-
        merge needed — closes the "ignore suffix in M=1 wire sparse" caveat
        from gh #63 commit 2be3fc5.

        Q:     (H_q, D) — M=1 only
        K_suf: (H_kv, S_suf, D) — caller's locally-decoded suffix K
        V_suf: (H_kv, S_suf, D) — caller's locally-decoded suffix V
        H_kv:  defaults to H_q on non-GQA
        head_map: identity if not given and H_kv == H_q
        selector: "block_mean" (default) or "quest" (W11 Phase 2)
        """
        if Q.ndim != 2:
            raise ValueError(f"attend_query_sparse_auto_fused: Q must be (H_q, D); got {Q.shape}")
        H_q, D = Q.shape
        if H_kv is None:
            H_kv = H_q
        if H_q % H_kv != 0:
            raise ValueError(f"H_q={H_q} must be a multiple of H_kv={H_kv}")
        if K_suf.shape != V_suf.shape:
            raise ValueError(f"K_suf and V_suf shapes mismatch: {K_suf.shape} vs {V_suf.shape}")
        if K_suf.ndim != 3 or K_suf.shape[0] != H_kv or K_suf.shape[2] != D:
            raise ValueError(f"K_suf must be (H_kv={H_kv}, S_suf, D={D}); got {K_suf.shape}")
        S_suf = K_suf.shape[1]
        if head_map is None:
            hm_bytes = b"" if H_kv == H_q else \
                np.repeat(np.arange(H_kv, dtype=np.uint8), H_q // H_kv).tobytes()
        else:
            hm_bytes = np.ascontiguousarray(head_map, dtype=np.uint8).tobytes()
        selector_id = 0
        if selector == "quest":
            selector_id = 1
        elif selector not in ("block_mean", "block-mean", ""):
            raise ValueError(f"unknown selector {selector!r}; expected 'block_mean' or 'quest'")
        sid = self._attend_session(namespace)
        resp_args = ["ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED", sid, str(layer_id),
                     str(H_q), str(D), str(int(B)), str(int(K_top)),
                     str(int(H_kv)), str(int(S_suf)),
                     np.ascontiguousarray(Q, dtype=np.float32).tobytes(),
                     np.ascontiguousarray(K_suf, dtype=np.float32).tobytes(),
                     np.ascontiguousarray(V_suf, dtype=np.float32).tobytes(),
                     hm_bytes]
        # W11 Phase 2: selector_id is positional after fa_window. Emit
        # fa_window=-1 placeholder if needed so selector_id lands correctly.
        if fa_window is not None or selector_id != 0:
            resp_args.append(str(int(fa_window) if fa_window is not None else -1))
        if selector_id != 0:
            resp_args.append(str(selector_id))
        t0 = time.perf_counter()
        # gh #67: detect COLDMISS, warm + retry once.
        r = self._attend_call_with_warm_retry(resp_args, namespace, H_q, D)
        self.attend_query_ms_total += (time.perf_counter() - t0) * 1000
        self.attend_query_calls += 1
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            raise RuntimeError(f"ATTEND.PREFIX.QUERY_SPARSE_AUTO_FUSED failed: {r[:80]!r}")
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        body = r[nl + 2:nl + 2 + blen]
        return np.frombuffer(body[: H_q * D * 4], dtype=np.float32).copy().reshape(H_q, D)

    def attend_query_fused(self, namespace: str, layer_id: int,
                           Q: np.ndarray, K_suf: np.ndarray, V_suf: np.ndarray,
                           head_map: np.ndarray,
                           fa_window: Optional[int] = None) -> np.ndarray:
        """gh #49: ATTEND.PREFIX.QUERY_FUSED — server-side suffix-SDPA + merge.

        Replaces the host-side _suffix_sdpa_with_lse + _online_softmax_merge
        path in mlx_lm_patch.py. One wire round-trip per layer; merged
        output returned directly (no LSE merge on the client).

        Shapes:
          Q:        (H_q, M, D)        float32
          K_suf:    (H_kv, S_suf, D)   float32 — suffix K accumulated locally
          V_suf:    (H_kv, S_suf, D)   float32
          head_map: (H_q,)             uint8   — head_map[h_q] = h_kv

        Returns: (H_q, M, D) float32 — merged attention over (prefix ∪ suffix).

        gh #60 Step 1: `fa_window` (optional) overrides the server's
        --fa-window flag for THIS call only. Per-layer routing for hybrid SWA
        models (Gemma 4 / Mistral SWA / Qwen3.5): pass `fa_window=512` on
        sliding layers, `fa_window=0` on full layers. Both wire lanes (binary
        + RESP) carry the override; older servers ignore the trailing field
        on the binary frame and reject the extra positional on RESP — the
        binary lane will succeed on any 2026-05-11+ build.
        """
        if Q.ndim != 3:
            raise ValueError(f"attend_query_fused: Q must be 3-D (H_q, M, D), got {Q.shape}")
        H_q, M, D = Q.shape
        if K_suf.ndim != 3 or V_suf.ndim != 3:
            raise ValueError("attend_query_fused: K_suf/V_suf must be 3-D (H_kv, S_suf, D)")
        H_kv = K_suf.shape[0]
        S_suf = K_suf.shape[1]
        if V_suf.shape != K_suf.shape:
            raise ValueError("attend_query_fused: K_suf/V_suf shape mismatch")
        if K_suf.shape[2] != D:
            raise ValueError("attend_query_fused: D mismatch between Q and K_suf/V_suf")
        if H_q % H_kv != 0:
            raise ValueError(f"attend_query_fused: H_q ({H_q}) must be a multiple of H_kv ({H_kv})")
        if head_map.shape != (H_q,):
            raise ValueError(f"attend_query_fused: head_map must have shape ({H_q},), got {head_map.shape}")
        if head_map.dtype != np.uint8:
            head_map = head_map.astype(np.uint8)
        sid = self._attend_session(namespace)
        # Hot path is called once per layer per forward — keep allocations
        # to a minimum. `np.ascontiguousarray` is a no-op when the caller
        # already passes contiguous fp32 (the gh #49 zero-copy patch does);
        # `.tobytes()` is unavoidable for the RESP path because `_RESP._encode`
        # joins all parts via `b"".join`. The binary fast lane below uses
        # sendmsg scatter-gather and feeds the underlying buffers directly
        # via the buffer protocol — no per-call bytes alloc on send.
        if Q.dtype != np.float32 or not Q.flags["C_CONTIGUOUS"]:
            Q = np.ascontiguousarray(Q, dtype=np.float32)
        if S_suf > 0:
            if K_suf.dtype != np.float32 or not K_suf.flags["C_CONTIGUOUS"]:
                K_suf = np.ascontiguousarray(K_suf, dtype=np.float32)
            if V_suf.dtype != np.float32 or not V_suf.flags["C_CONTIGUOUS"]:
                V_suf = np.ascontiguousarray(V_suf, dtype=np.float32)
        if head_map.dtype != np.uint8 or not head_map.flags["C_CONTIGUOUS"]:
            head_map = np.ascontiguousarray(head_map, dtype=np.uint8)

        out_bytes = H_q * M * D * 4

        # gh #50: try the binary fast lane first. The first call lazy-opens
        # the connection on port+1; on any failure the binary path is
        # disabled for this PromptCache and the RESP fallback below runs.
        if not self._binary_disabled and self._binary is None and not self._binary_attempted:
            self._binary_attempted = True
            try:
                self._binary = _BinarySock(self.host, self.port + 1)
            except (ConnectionError, OSError, RuntimeError):
                # Server has no --kvcache (no binary listener), or older
                # build, or a different process is on port+1. Stay on RESP.
                self._binary = None
                self._binary_disabled = True
        if self._binary is not None:
            try:
                sid_b = sid.encode()
                # Body layout — must match parse_attend_prefix_query_fused in
                # src/network/binary_protocol.mojo:
                #   [sid_len:u16][sid_bytes][layer_id:u16][H_q:u16][D:u16]
                #   [H_kv:u16][M:u32][S_suf:u32][Q][K_suf][V_suf][head_map]
                sid_len_b = struct.pack("<H", len(sid_b))
                fixed_b = struct.pack(
                    "<HHHHII",
                    int(layer_id), int(H_q), int(D), int(H_kv),
                    int(M), int(S_suf),
                )
                # gh #60 Step 1: optional trailing u32 fa_window override.
                # Old servers ignore (read body up to data_start+need); new
                # servers parse the 4 trailing bytes. Send only when caller
                # specified — keeps wire bytes constant on the common path.
                fa_window_b = (
                    struct.pack("<I", int(fa_window)) if fa_window is not None else b""
                )
                t0 = time.perf_counter()
                # Pass numpy arrays via the buffer protocol — sendmsg accepts
                # any bytes-like (including ndarrays' underlying buffer), so
                # no .tobytes() copy on the send side.
                status, body = self._binary._send_recv(
                    _CMD_ATTEND_PREFIX_QUERY_FUSED,
                    (sid_len_b, sid_b, fixed_b, Q,
                     K_suf if S_suf > 0 else b"",
                     V_suf if S_suf > 0 else b"", head_map,
                     fa_window_b),
                )
                self.attend_query_ms_total += (time.perf_counter() - t0) * 1000
                self.attend_query_calls += 1
                if status == _BSTATUS_COLDMISS:
                    # gh #67: cold-tier rehydrate, then retry once on the
                    # binary lane. The warm command itself only exists on
                    # RESP — the binary protocol is intentionally narrow
                    # (no admin commands).
                    self.cold_rehydrates_observed = getattr(self, "cold_rehydrates_observed", 0) + 1
                    n = self._warm_namespace(namespace, H_q, D)
                    if n > 0:
                        status, body = self._binary._send_recv(
                            _CMD_ATTEND_PREFIX_QUERY_FUSED,
                            (sid_len_b, sid_b, fixed_b, Q,
                             K_suf if S_suf > 0 else b"",
                             V_suf if S_suf > 0 else b"", head_map,
                             fa_window_b),
                        )
                if status != _BSTATUS_OK or len(body) != out_bytes:
                    raise RuntimeError(
                        f"binary QUERY_FUSED status={status} body={len(body)}/{out_bytes}")
                return np.frombuffer(body, dtype=np.float32).reshape(H_q, M, D)
            except (RuntimeError, ConnectionError, OSError) as _e:
                # Permanently disable the binary lane on first error: the
                # connection state may be unrecoverable, and silently
                # alternating between paths for the same PromptCache hides
                # bugs. Subsequent calls flow through RESP.
                try:
                    self._binary.close()
                finally:
                    self._binary = None
                    self._binary_disabled = True
                # Fall through to RESP.

        # RESP fallback — kept verbatim from pre-#50 so older Pion builds
        # without the binary opcode still work.
        Q_b = Q.tobytes()
        if S_suf > 0:
            Ks_b = K_suf.tobytes()
            Vs_b = V_suf.tobytes()
        else:
            Ks_b = b""
            Vs_b = b""
        HM_b = head_map.tobytes()
        # gh #60 Step 1: optional trailing positional fa_window. Older Pion
        # builds don't expect an extra arg → only append when set.
        resp_args = ["ATTEND.PREFIX.QUERY_FUSED", sid, str(layer_id),
                     str(H_q), str(D), str(S_suf), str(H_kv),
                     Q_b, Ks_b, Vs_b, HM_b]
        if fa_window is not None:
            resp_args.append(str(int(fa_window)))
        t0 = time.perf_counter()
        # gh #67: detect COLDMISS, warm + retry once.
        r = self._attend_call_with_warm_retry(resp_args, namespace, H_q, D)
        self.attend_query_ms_total += (time.perf_counter() - t0) * 1000
        self.attend_query_calls += 1
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            raise RuntimeError(f"ATTEND.PREFIX.QUERY_FUSED failed: {r[:160]!r}")
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        body = r[nl + 2:nl + 2 + blen]
        if len(body) != out_bytes:
            raise RuntimeError(
                f"ATTEND.PREFIX.QUERY_FUSED reply size mismatch: got {len(body)}, expected {out_bytes}")
        return np.frombuffer(body, dtype=np.float32).reshape(H_q, M, D)

    def attend_drop(self, namespace: str) -> None:
        """Hint the MLX sidecar to drop this namespace's resident K/V.

        Best-effort — relies on the sidecar's existing DROP_SESSION binary
        message; not surfaced via RESP yet, so this is a no-op until the
        sidecar's RESP shim is added. Marker-only for now.
        """
        self._stage2_pushed.discard(self._attend_session(namespace))

    def _stage2_push_cold(self, namespace: str, prefix_len: int, cache) -> None:
        """Push the just-prefilled MLX cache to the sidecar via ATTEND.PREFIX.STORE.

        Called from get_or_prefill() on the MISS path when stage2=True. The
        K/V tensors are already in MLX memory; we slice, reshape to (H, N, D),
        and ship per layer. Total wire bytes ≈ same as the V.STOREBATCH path
        but the data lands in MLX-side memory rather than V-store, so future
        attend_query() calls don't re-marshal K/V.
        """
        mx, _ = _require_mlx()
        sid = self._attend_session(namespace)
        if sid in self._stage2_pushed:
            return
        L = _require_layout(self.layout)
        for li in range(L.n_layers):
            k = cache[li].keys[:, :, :prefix_len, :]   # (1, n_kv, S, D)
            v = cache[li].values[:, :, :prefix_len, :]
            # Pion expects (H, N, D); flatten the batch dim.
            K_np = np.array(k[0].astype(mx.float32))  # (n_kv, S, D)
            V_np = np.array(v[0].astype(mx.float32))
            self.attend_store_layer(namespace, li, K_np, V_np)
        self._stage2_pushed.add(sid)

    # ── Mixed-cache hybrid wire path ────────────────────────────────────────
    # Qwen3.5-4B-style hybrids carry two cache types in the same list:
    # ArraysCache (linear / SSM / GatedDeltaNet layers — fixed-size recurrent
    # state) and KVCache (softmax layers — per-token K/V). The legacy wire
    # path (V.STOREBATCH for all layers) doesn't fit; this path routes per
    # cache slot.
    #
    #   ArraysCache slot → SSM.PREFIX.STORE / FETCH (opaque safetensors blob)
    #   KVCache slot     → V.CREATE-via-KV.PREFIX.REGISTER + V.STOREBATCH /
    #                       V.FETCH RANGE on softmax-rank 0..N_softmax-1
    #
    # Cross-restart durability (gh #94, Mac): BOTH halves now survive a server
    # restart. The softmax half rides the V-store WAL/snapshot; the linear half
    # rides the SSM.PREFIX WAL/snapshot (gh #94). A plain SIGKILL replays both
    # WALs; KV.PREFIX.SAVE compacts both under one call. Composed round-trip is
    # gated by tests/test_hybrid_split_durability.py (8 softmax ranks + 24
    # linear slots, bit-equal across SIGKILL via WAL replay AND via snapshot).
    # Note: V-store WAL replay is Mac-only today (vstore_wal_linux_regression),
    # so the cross-restart guarantee is Mac-scoped until that lands on Linux.

    @staticmethod
    def _classify_cache(cache):
        """Walk a fresh post-prefill cache; return (linear_idxs, softmax_idxs, kv_dim).

        Routes based on `type(c).__name__`:
          KVCache / RotatingKVCache → softmax slot (V-store)
          everything else (ArraysCache, MambaCache, …) → linear slot (SSM.PREFIX)

        kv_dim is read from the first softmax slot's K shape `[B, H, N, D]`
        → `H * D`. Returns kv_dim=None if no softmax slots exist (pure-
        recurrent model — wire path falls back to all-SSM).
        """
        linear: list[int] = []
        softmax: list[int] = []
        kv_dim: int | None = None
        for i, c in enumerate(cache):
            if c is None:
                continue
            ctype = type(c).__name__
            if ctype in ("KVCache", "RotatingKVCache"):
                softmax.append(i)
                if kv_dim is None and getattr(c, "keys", None) is not None:
                    shp = c.keys.shape
                    if len(shp) == 4:
                        kv_dim = int(shp[1]) * int(shp[3])
            else:
                linear.append(i)
        return linear, softmax, kv_dim

    def _serialize_arrays_cache(self, c) -> bytes:
        """ArraysCache / MambaCache → safetensors bytes (opaque blob).

        Same wire-shape as `spike_qwen3_5_cross_process_wire.py`
        — type name + array count in metadata; arrays keyed "a0", "a1", …
        """
        mx, _ = _require_mlx()
        import os as _os
        import tempfile as _tempfile
        # Never `c.state`: mlx-lm 0.32 widened it with scalars and padded
        # buffers, which would change (or break) the blob. See _compat.
        contents = slot_arrays(c)
        arrays: dict = {}
        meta_bits = [f"type={type(c).__name__}", f"narr={len(contents)}"]
        for i, a in enumerate(contents):
            if a is not None:
                mx.eval(a)
                arrays[f"a{i}"] = a
        meta = "|".join(meta_bits)
        with _tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        mx.save_safetensors(path, arrays, metadata={"meta": meta})
        with open(path, "rb") as f:
            blob = f.read()
        _os.unlink(path)
        return blob

    def _restore_arrays_cache(self, c, blob: bytes) -> None:
        """safetensors bytes → the slot's contents (see _compat.set_slot_arrays)."""
        mx, _ = _require_mlx()
        import os as _os
        import tempfile as _tempfile
        with _tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(blob)
            path = f.name
        arrays = mx.load(path)
        _, md = mx.load(path, return_metadata=True)
        _os.unlink(path)
        meta = dict(p.split("=", 1) for p in md.get("meta", "").split("|") if "=" in p)
        narr = int(meta.get("narr", 0))
        set_slot_arrays(c, [arrays.get(f"a{i}") for i in range(narr)])

    def _kv_to_flat(self, c, prefix_len: int):
        """KVCache → (k_fp32_flat, v_fp32_flat, H, D) for V.STOREBATCH.

        cache.keys is [B=1, H, N, D]; V.STOREBATCH expects rows of
        per-token vectors of length H*D, fp32. Identical to the spike
        `softmax_cache_to_wire`.
        """
        mx, _ = _require_mlx()
        k = c.keys[:, :, :prefix_len, :]
        v = c.values[:, :, :prefix_len, :]
        H = int(k.shape[1])
        D = int(k.shape[3])
        k_np = np.array(k.transpose(0, 2, 1, 3).reshape(prefix_len, H * D).astype(mx.float32))
        v_np = np.array(v.transpose(0, 2, 1, 3).reshape(prefix_len, H * D).astype(mx.float32))
        return k_np.copy(), v_np.copy(), H, D

    def _flat_to_kv(self, c, k_flat: np.ndarray, v_flat: np.ndarray,
                    prefix_len: int, H: int, D: int) -> None:
        """(k_fp32_flat, v_fp32_flat) → c.update_and_fetch in fp16."""
        mx, _ = _require_mlx()
        k_mx = mx.array(
            k_flat.reshape(prefix_len, H, D).transpose(1, 0, 2)[None, ...],
            dtype=mx.float16,
        )
        v_mx = mx.array(
            v_flat.reshape(prefix_len, H, D).transpose(1, 0, 2)[None, ...],
            dtype=mx.float16,
        )
        c.update_and_fetch(k_mx, v_mx)

    def _store_mixed_hybrid(self, namespace: str, cache, prefix_len: int,
                             linear_idxs: list[int], softmax_idxs: list[int],
                             kv_dim: int) -> None:
        """Push the prefix to Pion via the split-substrate path.

        - Linear cache slots → SSM.PREFIX.STORE keyed by `namespace, slot_idx`.
        - Softmax cache slots → V.STOREBATCH on `<ns>_pk` / `<ns>_pv` indexed
          by softmax-rank 0..N_softmax-1, after KV.PREFIX.REGISTER creates
          the K and V sessions with uniform kv_dim.
        """
        # softmax_bitexact: store softmax anchors opaquely too (bit-exact). A
        # KVCache slot's contents are (keys, values) — the same opaque safetensors
        # serializer used for linear ArraysCache slots round-trips them in the
        # native dtype, with no fp16 V-store loss. Slot indices are unique across
        # the cache list, so linear and softmax never collide on SSM.PREFIX keys.
        # KV.PREFIX.REGISTER always runs — it is what makes KV.PREFIX.LOOKUP
        # return +HIT on the next request (the server-side hit signal), even in
        # bitexact mode where the softmax K/V rides SSM.PREFIX instead of V-store.
        r = self.resp.call("KV.PREFIX.REGISTER", namespace, str(kv_dim), self.vquant)
        if not r.startswith(b"+OK") and b"already" not in r.lower() and b"exists" not in r.lower():
            raise RuntimeError(f"KV.PREFIX.REGISTER (hybrid) failed: {r!r}")

        opaque_slots = list(linear_idxs)
        if self.softmax_bitexact:
            opaque_slots += list(softmax_idxs)
        else:
            # Structured V-store path for the softmax half (default; Qwen3.5).
            sid_k = f"{namespace}_pk"
            sid_v = f"{namespace}_pv"
            for sm_rank, slot in enumerate(softmax_idxs):
                k_flat, v_flat, _H, _D = self._kv_to_flat(cache[slot], prefix_len)
                self._storebatch(sid_k, sm_rank, k_flat)
                self._storebatch(sid_v, sm_rank, v_flat)

        # Ship opaque cache slots via SSM.PREFIX.STORE, keyed by raw slot.
        for slot in opaque_slots:
            blob = self._serialize_arrays_cache(cache[slot])
            rep = self.resp.call("SSM.PREFIX.STORE", namespace, str(slot), blob)
            if not rep.startswith(b"+OK"):
                raise RuntimeError(f"SSM.PREFIX.STORE slot {slot}: {rep!r}")

    def _fetch_to_cache_mixed_hybrid(self, namespace: str, prefix_len: int):
        """Build a fresh cache and rehydrate it from the split substrate.

        Walks `make_prompt_cache(model)` to discover cache-slot types
        (deterministic for a given model), then:
          - Linear slot → SSM.PREFIX.FETCH(namespace, slot) → restore.
          - Softmax slot → V.FETCH RANGE on `<ns>_pk` / `<ns>_pv` at
                            softmax-rank → restore via update_and_fetch.
        """
        _mx, make_prompt_cache = _require_mlx()
        cache = make_prompt_cache(self.model)
        linear_idxs, softmax_idxs, kv_dim = self._classify_cache(cache)

        # Softmax half — opaque SSM.PREFIX (bit-exact) or structured V-store.
        opaque_slots = list(linear_idxs)
        if self.softmax_bitexact:
            opaque_slots += list(softmax_idxs)
        else:
            sid_k = f"{namespace}_pk"
            sid_v = f"{namespace}_pv"
            for sm_rank, slot in enumerate(softmax_idxs):
                k_flat = self._fetch_range(sid_k, sm_rank, prefix_len)
                v_flat = self._fetch_range(sid_v, sm_rank, prefix_len)
                # H, D derived from kv_dim and model.args (uniform across softmax
                # layers in hybrids today).
                layout = _require_layout(self.layout)
                H = layout.n_kv_heads
                D = layout.head_dim
                self._flat_to_kv(cache[slot], k_flat, v_flat, prefix_len, H, D)

        # Opaque slots — SSM.PREFIX.FETCH per slot.
        for slot in opaque_slots:
            rep = self.resp.call("SSM.PREFIX.FETCH", namespace, str(slot))
            if rep is None or rep.startswith(b"$-1"):
                raise RuntimeError(
                    f"SSM.PREFIX.FETCH slot {slot} nil — linear-half state was lost "
                    f"(SSM.PREFIX is in-memory only on the server; a restart drops it). "
                    f"Drop the namespace and re-prefill."
                )
            # call() returns the full RESP frame `$<len>\r\n<bytes>\r\n`.
            nl = rep.find(b"\r\n")
            blen = int(rep[1:nl])
            blob = rep[nl + 2: nl + 2 + blen]
            self._restore_arrays_cache(cache[slot], blob)

        return cache

    # ── Public API ──────────────────────────────────────────────────────────

    def get_or_prefill(self, prefix_token_ids, namespace: str):
        """Return an MLX prompt cache populated with this prefix's K/V.

        Hits Pion if the namespace was registered before; otherwise prefills
        the prefix locally, registers + stores, returns the populated cache.

        Admission policy (admission_threshold > 1): observed-miss count must
        reach the threshold before Pion is touched. Below threshold the prefill
        runs locally and the cache is returned without registering — keeps
        one-off prompts out of the V-store under LRU pressure.
        """
        prefix_len = len(prefix_token_ids)
        if self.lookup(namespace):
            try:
                cache = self._fetch_to_cache(namespace, prefix_len)
                self.hits += 1
                return cache
            except RuntimeError as e:
                # Mixed-hybrid (Qwen3.5) split-substrate desync fallback.
                # gh #94 made SSM.PREFIX WAL/snapshot-durable, so a clean
                # restart no longer desyncs the two halves (both replay) —
                # see tests/test_hybrid_split_durability.py. This path now
                # guards only the narrow residual windows: an explicit
                # SSM.PREFIX.DROP of the linear half, or a SIGKILL between the
                # V-store WAL flush and the SSM WAL flush of the same store.
                # In those cases lookup still HITs on the V-store while
                # SSM.PREFIX.FETCH returns nil — fall through to cold-prefill
                # + re-store, same result as a true MISS plus a lookup RTT.
                if "linear-half state was lost" not in str(e):
                    raise
                # Treat as a miss; the store path below will repopulate both
                # halves under the same namespace.
        # MISS — local prefill in every case.
        self.misses += 1
        mx, make_prompt_cache = _require_mlx()
        prefix_ids = mx.array([prefix_token_ids])
        _t_prefill = time.perf_counter()
        cache = make_prompt_cache(self.model)
        chunk = self.prefill_chunk_size
        if chunk is None or prefix_len <= chunk:
            _ = self.model(prefix_ids, cache=cache)
            # Eval EVERY layer's state, not one slot. MLX is lazy: evaluating
            # layer 0's K computes layer 0 only, so the timer below used to
            # stop after ~1/n_layers of the prefill and the remaining layers
            # ran later inside _cache_to_arrays — PREFILL_MS (gh #262) then
            # credited Llama-3.2-1B's 2K-token prefill with 23 ms. The chunked
            # branch below already evals all layers per slice; this makes the
            # two branches measure the same work. Same total cost, moved
            # inside the clock. `state` covers KVCache and ArraysCache alike.
            evals = []
            for c in cache:
                if c is None or not getattr(c, "state", None):
                    continue
                evals.extend(a for a in c.state if a is not None)
            if evals:
                mx.eval(*evals)
        else:
            # Chunked cold prefill — forward in slices, eval ALL layer caches
            # between slices so the lazy graph doesn't accumulate across chunks.
            # Evaling cache[0].keys alone is not sufficient: layers >0 keep
            # their concat lazy and the next chunk's forward would re-evaluate
            # the entire history, defeating the memory savings.
            for start in range(0, prefix_len, chunk):
                end = start + chunk if start + chunk < prefix_len else prefix_len
                _ = self.model(prefix_ids[:, start:end], cache=cache)
                evals = []
                for c in cache:
                    if c is None:
                        continue
                    evals.append(c.keys)
                    evals.append(c.values)
                if evals:
                    mx.eval(*evals)

        self.last_prefill_ms = (time.perf_counter() - _t_prefill) * 1000

        # Admission: only push to Pion once we've seen this namespace `threshold`
        # times. Below threshold we still serve a correct local cache, just
        # don't pay the V.STOREBATCH cost or hold a V-store slot.
        observed = self._observed.get(namespace, 0) + 1
        self._observed[namespace] = observed
        if observed < self.admission_threshold:
            self.admission_skips += 1
            return cache

        layout = _require_layout(self.layout)
        # Classify cache shape: a mixed-type cache list (some KVCache, some
        # ArraysCache) is a "mixed hybrid" — Qwen3.5-4B-style Mamba+Transformer.
        # The wire is supported via the split path (softmax → V-store, linear
        # → SSM.PREFIX). Pure-softmax hybrids (Gemma 4 sliding-vs-full) are
        # still wire-blocked here; their in-process lane handles them.
        linear_idxs, softmax_idxs, kv_dim_from_cache = self._classify_cache(cache)
        is_mixed = len(linear_idxs) > 0 and len(softmax_idxs) > 0 and kv_dim_from_cache is not None

        if is_mixed:
            t0 = time.perf_counter()
            self._store_mixed_hybrid(
                namespace, cache, prefix_len,
                linear_idxs, softmax_idxs, kv_dim_from_cache,
            )
            self.store_ms_total += (time.perf_counter() - t0) * 1000
            self._local[namespace] = prefix_len
        elif not layout.is_hybrid:
            per_layer = self._cache_to_arrays(cache, prefix_len)

            t0 = time.perf_counter()
            self._register(namespace)
            sid_k = f"{namespace}_pk"
            sid_v = f"{namespace}_pv"
            for li, (k_np, v_np) in enumerate(per_layer):
                self._storebatch(sid_k, li, k_np)
                self._storebatch(sid_v, li, v_np)
            if not self._export_disabled:
                # gh #468: write the export file now, on the miss that already
                # paid a prefill, so the first hit from another process maps it.
                try:
                    self._export(namespace, prefix_len)
                except Exception:
                    self._export_disabled = True
            self.store_ms_total += (time.perf_counter() - t0) * 1000

            self._local[namespace] = prefix_len

        # Stage 2: also push K/V to MLX sidecar so attend_query() can run
        # with Q-only on the wire (no re-fetch). Stage 1 path is unchanged.
        # On pure-softmax hybrid arch (Gemma 4) the sidecar push is wire-
        # coupled, so skip it. Mixed hybrid (Qwen3.5) doesn't yet have a
        # GPU-attention path; same skip.
        if self.stage2 and not layout.is_hybrid and not is_mixed:
            self._stage2_push_cold(namespace, prefix_len, cache)
        if self.stage2 and not self._inproc_disabled:
            # gh #50 in-process fast lane: also stash the per-layer prefix
            # K/V as MLX arrays. The same-process consumer (mlx_lm_patch.py)
            # picks them up via PionPrefixCache and runs attention locally.
            # Cross-process consumers see an empty dict and stay on the wire.
            #
            # gh #59: iterate over `len(cache)`, not `n_layers`. On hybrid
            # archs (Gemma 4) the cache list is shorter than n_layers because
            # KV-shared layers don't carry their own cache entry — the model
            # routes them via shared_kv from earlier layers. cache[li] for
            # li >= len(cache) would IndexError; the consumer side
            # (mlx_lm_patch.PionPrefixCache) handles missing indices by
            # falling through to the local-cache path.
            kv_pairs = []
            for li in range(len(cache)):
                if cache[li] is None:
                    kv_pairs.append(None)
                    continue
                k = cache[li].keys[:, :, :prefix_len, :]
                v = cache[li].values[:, :, :prefix_len, :]
                # mx.eval to materialize before stashing — otherwise the
                # lazy graph could grow unbounded across requests.
                mx.eval(k, v)
                kv_pairs.append((k, v))
            self._mlx_prefix_kv[namespace] = kv_pairs

        return cache

    def _export(self, namespace: str, prefix_len: int):
        """gh #468: ask the server for the prefix as a file. Returns its path,
        or None (and turns the lane off) when the server cannot."""
        L = _require_layout(self.layout)
        r = self.resp.call("V.EXPORT", f"{namespace}_pk", f"{namespace}_pv", "0",
                           str(prefix_len), str(L.n_layers), str(L.n_kv_heads), str(L.head_dim))
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            self._export_disabled = True
            return None
        nl = r.find(b"\r\n")
        return r[nl + 2:-2].decode()

    def _fetch_via_export(self, namespace: str, prefix_len: int):
        """gh #468: the hit through the export lane — map the file and hand
        its arrays to the cache as they are (mx.load is lazy; the first
        forward reads them from the page cache, as with mlx-lm's own cache
        file). Returns None when a layer's cache is not a plain KVCache."""
        mx, make_prompt_cache = _require_mlx()
        cache = make_prompt_cache(self.model)
        if any(type(c).__name__ != "KVCache" for c in cache):
            return None
        path = self._export(namespace, prefix_len)
        if path is None:
            return None
        arrays = mx.load(path)
        for li, c in enumerate(cache):
            c.state = (arrays[f"k.{li}"], arrays[f"v.{li}"])
        return cache

    def _fetch_to_cache(self, namespace: str, prefix_len: int):
        # Dispatch on cache shape — a mixed-cache hybrid (Qwen3.5-style)
        # takes the split-substrate path. Cache-slot types are deterministic
        # for a given model, so we can probe a fresh cache once to decide.
        _mx, make_prompt_cache = _require_mlx()
        probe = make_prompt_cache(self.model)
        linear_idxs, softmax_idxs, _ = self._classify_cache(probe)
        if len(linear_idxs) > 0 and len(softmax_idxs) > 0:
            t0 = time.perf_counter()
            cache = self._fetch_to_cache_mixed_hybrid(namespace, prefix_len)
            self.fetch_ms_total += (time.perf_counter() - t0) * 1000
            return cache

        sid_k = f"{namespace}_pk"
        sid_v = f"{namespace}_pv"
        L = _require_layout(self.layout)
        per_layer = []
        t0 = time.perf_counter()
        if not self._export_disabled:
            try:
                cache = self._fetch_via_export(namespace, prefix_len)
            except Exception:
                cache = None
                self._export_disabled = True
            if cache is not None:
                self.fetch_ms_total += (time.perf_counter() - t0) * 1000
                return cache
        # Try V.FETCH BATCH (single round-trip per side, 2 round-trips total
        # across all layers). Falls back to per-layer V.FETCH RANGE if the
        # server is older or returns an unexpected reply — keeps the client
        # forward-compatible while we adopt the new primitive.
        if not self._batch_disabled:
            try:
                k_layers = self._fetch_batch_all_layers(sid_k, prefix_len, L.n_layers,
                                                        native_dim=L.kv_dim)
                v_layers = self._fetch_batch_all_layers(sid_v, prefix_len, L.n_layers,
                                                        native_dim=L.kv_dim)
                for k_flat, v_flat in zip(k_layers, v_layers):
                    per_layer.append((
                        k_flat.reshape(prefix_len, L.kv_dim),
                        v_flat.reshape(prefix_len, L.kv_dim),
                    ))
                self.fetch_ms_total += (time.perf_counter() - t0) * 1000
                return self._arrays_to_cache(per_layer, prefix_len)
            except (RuntimeError, ConnectionError) as e:
                # One-shot fallback: disable BATCH for this PionPromptCache,
                # remake the connection (an exception may have left it in
                # a half-read state — call_array reads multi-frame replies
                # so a parse error abandons unread bytes), and fall through
                # to the legacy per-layer path.
                self._batch_disabled = True
                try:
                    self.resp.sock.close()
                except Exception:
                    pass
                self.resp = _RESP(self.host, self.port)
                # Continue below using the per-layer loop.
        for li in range(L.n_layers):
            k_flat = self._fetch_range(sid_k, li, prefix_len, native_dim=L.kv_dim)
            v_flat = self._fetch_range(sid_v, li, prefix_len, native_dim=L.kv_dim)
            per_layer.append((k_flat.reshape(prefix_len, L.kv_dim), v_flat.reshape(prefix_len, L.kv_dim)))
        self.fetch_ms_total += (time.perf_counter() - t0) * 1000
        return self._arrays_to_cache(per_layer, prefix_len)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "admission_skips": self.admission_skips,
            "admission_threshold": self.admission_threshold,
            "fetch_ms_total": self.fetch_ms_total,
            "store_ms_total": self.store_ms_total,
            "last_prefill_ms": self.last_prefill_ms,
        }
