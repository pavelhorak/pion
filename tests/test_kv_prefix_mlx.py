#!/usr/bin/env python3
"""Stage 1 G1 gate — MLX edition. Path 1 from §13.3 / §17 / §18.

Apple Silicon unified memory + GQA + Pion's V-store. The HF gpt2 prototype
(`test_kv_prefix_prototype.py`) proved the wire format end-to-end; this script
runs the same gate against an MLX GQA model — the deployment platform of
record per §13.7 and §14.5.

Default model: mlx-community/Llama-3.2-1B-Instruct-4bit (16 layers, GQA 32q/8kv,
head_dim=64, kv_dim=512 — divisible by 32 so all V-store quant formats work).

Flow per repeat:
  COLD:    forward(system + user_B) end-to-end, time to last logits.
  ORACLE:  forward(system) once, capture native MLX cache; then forward(suffix_B)
           with that cache — in-process upper bound (no Pion).
  WARM:    forward(system) once, push K/V to Pion. New connection, V.FETCH RANGE,
           rebuild MLX cache via update_and_fetch, forward(suffix_B).

Pass criteria from §15.2 G1:
  TTFT(warm)/TTFT(cold)   < 0.30
  First-token agreement   = 100% (cold, oracle, warm)
  Logprob delta next-10   < per-tier ceiling (gh #171 — tests/g1_criteria.py)
  Storage efficiency       > per-tier floor; not asserted for fp16
  Fetch latency            < 5ms target on 2K-token prefix (we use whatever fits)
"""

from __future__ import annotations

import argparse
import hashlib
import socket
import sys
import time
from dataclasses import dataclass

import numpy as np

from g1_criteria import evaluate as g1_evaluate, print_verdict as g1_print_verdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, KVCache
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(1)


def _log_softmax(x, axis: int = -1):
    """mlx.nn doesn't ship a functional log_softmax across versions; do it manually."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


HOST = "127.0.0.1"
PORT = 1974
DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

_SYSTEM_TEMPLATE = (
    "You are a senior software engineer at an established technology company. "
    "Answer concisely, prefer concrete examples, and never speculate beyond what the user asked. "
    "If the user describes a bug, ask for the smallest reproduction first. "
    "If the user describes a design choice, weigh tradeoffs in two sentences before recommending. "
    "Always preserve backward compatibility unless the user explicitly waives it. "
    "Style: tight, imperative, no filler. "
    "House conventions: prefer fewer files over more, name things after what they do, never reach for "
    "a framework when a function will do, write tests before refactors, and treat warnings as bugs. "
    "When you cite documentation, link to the canonical reference rather than a tutorial. "
    "When you produce code, format it for the surrounding file's conventions even if you disagree. "
    "If the user is wrong about a fact, say so plainly and supply the correct fact. "
    "If the user asks for an opinion, supply one and own the tradeoffs. Avoid excessive disclaimers. "
)
USER_A = " Question: how should I structure a tagged-union value type in a low-level engine?"
USER_B = " Question: what is the right way to evict entries from a fixed-capacity pool?"


def system_prompt(n_repeats: int) -> str:
    return _SYSTEM_TEMPLATE * n_repeats


# ─────────────────────────────────────────────────────────────────────────────
# Pion RESP client (V.* commands) — shared with HF prototype
# ─────────────────────────────────────────────────────────────────────────────


class PionVStore:
    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(30)
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

    def _read_complete(self) -> bytes:
        while True:
            done = self._first_complete(self.buf)
            if done is not None:
                msg = self.buf[:done]
                self.buf = self.buf[done:]
                return msg
            chunk = self.sock.recv(8 * 1024 * 1024)
            if not chunk:
                raise ConnectionError("pion closed connection")
            self.buf += chunk

    @staticmethod
    def _first_complete(d: bytes) -> int | None:
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

    def send(self, parts) -> bytes:
        self.sock.sendall(self._encode(parts))
        return self._read_complete()

    def create(self, sid: str, dim: int, vquant: str = "int8") -> bool:
        parts = ["V.CREATE", sid, str(dim)]
        if vquant != "int8":
            parts += ["VQUANT", vquant]
        return self.send(parts).startswith(b":")

    def storebatch(self, sid: str, layer: int, start_id: int, values_fp32: np.ndarray) -> bool:
        n = values_fp32.shape[0]
        return self.send([
            "V.STOREBATCH", sid, str(layer), str(start_id), str(n),
            np.ascontiguousarray(values_fp32, dtype=np.float32).tobytes(),
        ]).startswith(b"+OK")

    def fetch_range(self, sid: str, layer: int, start: int, end: int) -> np.ndarray | None:
        r = self.send(["V.FETCH", sid, str(layer), "RANGE", str(start), str(end)])
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        return np.frombuffer(r[nl + 2:nl + 2 + blen], dtype=np.float32).copy()

    def info(self, sid: str | None = None) -> str:
        parts = ["V.INFO"] + ([sid] if sid else [])
        r = self.send(parts)
        if r.startswith(b"$"):
            nl = r.find(b"\r\n")
            n = int(r[1:nl])
            return r[nl + 2:nl + 2 + n].decode("utf-8", errors="replace")
        return r.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# MLX cache adapter
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheLayout:
    n_layers: int
    n_kv_heads: int
    head_dim: int

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def layout_from(model) -> CacheLayout:
    a = model.args
    n_kv = getattr(a, "num_key_value_heads", a.num_attention_heads)
    head_dim = getattr(a, "head_dim", a.hidden_size // a.num_attention_heads)
    return CacheLayout(n_layers=a.num_hidden_layers, n_kv_heads=n_kv, head_dim=head_dim)


def cache_to_arrays(cache, layout: CacheLayout, prefix_len: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice cached K/V down to the actual prefix tokens, flatten heads, return numpy fp32."""
    out = []
    for li in range(layout.n_layers):
        k = cache[li].keys[:, :, :prefix_len, :]   # (1, n_kv, S, D)
        v = cache[li].values[:, :, :prefix_len, :]
        # (1, n_kv, S, D) -> (S, n_kv * D)
        k_np = np.array(k.transpose(0, 2, 1, 3).reshape(prefix_len, layout.kv_dim).astype(mx.float32))
        v_np = np.array(v.transpose(0, 2, 1, 3).reshape(prefix_len, layout.kv_dim).astype(mx.float32))
        out.append((k_np.copy(), v_np.copy()))
    return out


def arrays_to_cache(model, per_layer: list[tuple[np.ndarray, np.ndarray]], layout: CacheLayout, prefix_len: int):
    """Build a fresh prompt cache and inject pre-computed K/V via update_and_fetch.

    update_and_fetch handles MLX's chunked allocation (step=256) so subsequent
    decode steps append correctly.
    """
    cache = make_prompt_cache(model)
    for li in range(layout.n_layers):
        k_np, v_np = per_layer[li]
        # (S, n_kv * D) -> (S, n_kv, D) -> (1, n_kv, S, D)
        k_mx = mx.array(
            k_np.reshape(prefix_len, layout.n_kv_heads, layout.head_dim).transpose(1, 0, 2)[None, ...],
            dtype=mx.float16,
        )
        v_mx = mx.array(
            v_np.reshape(prefix_len, layout.n_kv_heads, layout.head_dim).transpose(1, 0, 2)[None, ...],
            dtype=mx.float16,
        )
        cache[li].update_and_fetch(k_mx, v_mx)
    return cache


def store_prefix(
    pion: PionVStore,
    prefix_hash: str,
    per_layer: list[tuple[np.ndarray, np.ndarray]],
    layout: CacheLayout,
    vquant: str,
    boundary_layers: int,
):
    """Push K/V into Pion. Optional boundary-layer FP16 protection (M4 Phase 4)."""
    sid_k = f"{prefix_hash}_k_{vquant}"
    sid_v = f"{prefix_hash}_v_{vquant}"
    assert pion.create(sid_k, layout.kv_dim, vquant=vquant)
    assert pion.create(sid_v, layout.kv_dim, vquant=vquant)
    sid_kb = sid_vb = None
    if boundary_layers > 0:
        sid_kb = f"{prefix_hash}_k_fp16"
        sid_vb = f"{prefix_hash}_v_fp16"
        assert pion.create(sid_kb, layout.kv_dim, vquant="fp16")
        assert pion.create(sid_vb, layout.kv_dim, vquant="fp16")

    bytes_sent = 0
    n_b = n_m = 0
    t0 = time.perf_counter()
    for li, (k_np, v_np) in enumerate(per_layer):
        is_boundary = boundary_layers > 0 and (
            li < boundary_layers or li >= layout.n_layers - boundary_layers
        )
        if is_boundary:
            assert pion.storebatch(sid_kb, li, 0, k_np)
            assert pion.storebatch(sid_vb, li, 0, v_np)
            n_b += 1
        else:
            assert pion.storebatch(sid_k, li, 0, k_np)
            assert pion.storebatch(sid_v, li, 0, v_np)
            n_m += 1
        bytes_sent += k_np.nbytes + v_np.nbytes
    summary = f"main({vquant})={n_m} boundary(fp16)={n_b}"
    return sid_k, sid_v, sid_kb, sid_vb, (time.perf_counter() - t0) * 1000, bytes_sent, summary


def fetch_prefix(
    pion: PionVStore,
    layout: CacheLayout,
    prefix_len: int,
    sid_k_main: str,
    sid_v_main: str,
    sid_k_boundary: str | None,
    sid_v_boundary: str | None,
    boundary_layers: int,
):
    bytes_received = 0
    per_layer = []
    t0 = time.perf_counter()
    for li in range(layout.n_layers):
        is_boundary = (
            sid_k_boundary is not None
            and (li < boundary_layers or li >= layout.n_layers - boundary_layers)
        )
        sk = sid_k_boundary if is_boundary else sid_k_main
        sv = sid_v_boundary if is_boundary else sid_v_main
        k_flat = pion.fetch_range(sk, li, 0, prefix_len)
        v_flat = pion.fetch_range(sv, li, 0, prefix_len)
        if k_flat is None or v_flat is None:
            raise RuntimeError(f"V.FETCH RANGE None at layer {li}")
        bytes_received += k_flat.nbytes + v_flat.nbytes
        per_layer.append((k_flat.reshape(prefix_len, layout.kv_dim), v_flat.reshape(prefix_len, layout.kv_dim)))
    return per_layer, (time.perf_counter() - t0) * 1000, bytes_received


# ─────────────────────────────────────────────────────────────────────────────
# Inference + measurement
# ─────────────────────────────────────────────────────────────────────────────


PREFILL_STEP = 2048          # mlx_lm.generate_step's default prefill_step_size


def forward_logits(model, ids: mx.array, cache=None) -> tuple[float, mx.array]:
    """Prefill `ids` the way mlx_lm.generate_step does; return (ttft_ms, last-token logits).

    Every token but the last runs through the model in PREFILL_STEP chunks with
    only the cache state evaluated; the last token alone yields the logits.
    Until 2026-10-02 this evaluated one forward's logits over all of `ids` — a
    vocabulary projection at every position, which no generation computes — so
    every cold baseline timed with it was too slow (~30% at 2K tokens on
    Llama-3.2-1B) and every speedup against it too high.
    """
    if cache is None:
        cache = make_prompt_cache(model)
    mx.eval(ids)
    t0 = time.perf_counter()
    done, n = 0, ids.shape[1]
    while n - done > 1:
        step = min(PREFILL_STEP, n - done - 1)
        model(ids[:, done:done + step], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()       # as generate_step does after each prefill chunk
        done += step
    last = model(ids[:, done:], cache=cache)[0, -1]
    mx.eval(last)
    ttft_ms = (time.perf_counter() - t0) * 1000
    return ttft_ms, last


def quant_storage_bytes(per_layer, fmt: str, kv_dim: int) -> int:
    n_layers = len(per_layer)
    seq = per_layer[0][0].shape[0]
    if fmt == "fp16":
        bpt = kv_dim * 2
    elif fmt == "int8":
        bpt = kv_dim
    elif fmt == "turbo4":
        bpt = 4 + (kv_dim // 32) * 18
    else:
        raise ValueError(fmt)
    return n_layers * seq * bpt * 2  # K + V


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(args) -> int:
    print(f"Stage 1 G1 (MLX)  model={args.model} vquant={args.vquant} "
          f"prompt_repeats={args.prompt_repeats} boundary={args.boundary}")
    print(f"  pion: {HOST}:{PORT}")

    try:
        s = socket.create_connection((HOST, PORT), timeout=2); s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e}). Start with: ./pion-server --kvcache -w 1")
        return 2

    print("loading model...")
    model, tok = load(args.model)
    layout = layout_from(model)
    print(f"  layers={layout.n_layers} n_kv_heads={layout.n_kv_heads} head_dim={layout.head_dim} kv_dim={layout.kv_dim}")
    if layout.kv_dim % 32 != 0:
        print(f"FAIL: kv_dim={layout.kv_dim} not divisible by 32; turbo4 unavailable.")
        return 2

    SYSTEM = system_prompt(args.prompt_repeats)
    prefix_ids_list = tok.encode(SYSTEM)
    full_b_list = tok.encode(SYSTEM + USER_B)
    prefix_len = len(prefix_ids_list)
    suffix_b_list = full_b_list[prefix_len:]
    print(f"  prefix_tokens={prefix_len}  suffix_b_tokens={len(suffix_b_list)}")
    if len(suffix_b_list) <= 0 or full_b_list[:prefix_len] != prefix_ids_list:
        print("FAIL: tokenizer didn't reproduce prefix; pick a different USER_B")
        return 2

    full_b = mx.array([full_b_list])
    prefix_ids = mx.array([prefix_ids_list])
    suffix_b = mx.array([suffix_b_list])

    # Warmup MLX kernels
    for _ in range(args.warmup):
        forward_logits(model, full_b)
        forward_logits(model, prefix_ids)

    # ── COLD: forward(system + B) end-to-end ────────────────────────────────
    print("\n[cold] forward(system + B) end-to-end")
    cold_ttft, cold_first_logits, cold_logp = [], None, None
    for _ in range(args.repeats):
        ttft, last_logits = forward_logits(model, full_b)
        cold_ttft.append(ttft)
        cold_first_logits = last_logits
    cold_logp = _log_softmax(cold_first_logits, axis=-1)
    cold_med = float(np.median(cold_ttft))
    cold_id = int(mx.argmax(cold_first_logits).item())
    print(f"  TTFT median={cold_med:.1f}ms  first_token_id={cold_id}")

    # ── ORACLE: forward(prefix) once, then forward(suffix) with native cache ─
    print("\n[oracle] forward(prefix), then forward(suffix_B) with native cache")
    # Prefill once, capture native cache
    native_cache = make_prompt_cache(model)
    _ = model(prefix_ids, cache=native_cache); mx.eval(native_cache[0].keys)
    # Snapshot keys/values so we can rebuild between repeats (model() mutates cache)
    native_arrays = cache_to_arrays(native_cache, layout, prefix_len)

    oracle_ttft, oracle_first_logits = [], None
    for _ in range(args.repeats):
        rebuilt = arrays_to_cache(model, native_arrays, layout, prefix_len)
        ttft, last_logits = forward_logits(model, suffix_b, cache=rebuilt)
        oracle_ttft.append(ttft)
        oracle_first_logits = last_logits
    oracle_logp = _log_softmax(oracle_first_logits, axis=-1)
    oracle_med = float(np.median(oracle_ttft))
    oracle_id = int(mx.argmax(oracle_first_logits).item())
    print(f"  TTFT median={oracle_med:.1f}ms  first_token_id={oracle_id}")

    # ── STORE: push to Pion ─────────────────────────────────────────────────
    print(f"\n[store] uploading prefix K/V to Pion ({args.vquant}, boundary={args.boundary})")
    pion = PionVStore()
    prefix_hash = hashlib.sha256(
        f"{args.model}|{args.vquant}|b{args.boundary}|{SYSTEM}".encode()
    ).hexdigest()[:16]
    sid_k, sid_v, sid_kb, sid_vb, store_ms, bytes_up, summary = store_prefix(
        pion, prefix_hash, native_arrays, layout, args.vquant, args.boundary
    )
    print(f"  store_ms={store_ms:.1f}  bytes_up={bytes_up/1024:.1f}KB  layout: {summary}")
    print(f"  V.INFO main: {pion.info(sid_k).strip()[:100]}...")

    # ── WARM: V.FETCH RANGE prefix, rebuild cache, forward(suffix) ──────────
    print("\n[warm] V.FETCH RANGE prefix, rebuild cache, forward(suffix_B)")
    warm_total_ttft, warm_fetch_ms, warm_first_logits = [], [], None
    bytes_dn = 0
    for _ in range(args.repeats):
        per_layer_q, fetch_ms, bytes_dn = fetch_prefix(
            pion, layout, prefix_len,
            sid_k_main=sid_k, sid_v_main=sid_v,
            sid_k_boundary=sid_kb, sid_v_boundary=sid_vb,
            boundary_layers=args.boundary,
        )
        rebuilt = arrays_to_cache(model, per_layer_q, layout, prefix_len)
        ttft, last_logits = forward_logits(model, suffix_b, cache=rebuilt)
        warm_fetch_ms.append(fetch_ms)
        warm_total_ttft.append(fetch_ms + ttft)
        warm_first_logits = last_logits
    warm_logp = _log_softmax(warm_first_logits, axis=-1)
    warm_med = float(np.median(warm_total_ttft))
    fetch_med = float(np.median(warm_fetch_ms))
    warm_id = int(mx.argmax(warm_first_logits).item())
    print(f"  TTFT median={warm_med:.1f}ms  (fetch={fetch_med:.1f}ms + forward={warm_med-fetch_med:.1f}ms)")
    print(f"  bytes_dn={bytes_dn/1024:.1f}KB  first_token_id={warm_id}")

    # ── METRICS ─────────────────────────────────────────────────────────────
    ratio = warm_med / cold_med if cold_med > 0 else float("inf")
    cold_top10 = mx.argsort(cold_logp)[-10:]
    warm_logp_at_top10 = warm_logp[cold_top10]
    oracle_logp_at_top10 = oracle_logp[cold_top10]
    cold_logp_at_top10 = cold_logp[cold_top10]
    drift = float(mx.mean(mx.abs(warm_logp_at_top10 - oracle_logp_at_top10)).item())
    drift_oracle_vs_cold = float(mx.mean(mx.abs(oracle_logp_at_top10 - cold_logp_at_top10)).item())

    fp16_bytes = quant_storage_bytes(native_arrays, "fp16", layout.kv_dim)
    if args.boundary > 0:
        seq = native_arrays[0][0].shape[0]
        n_b = 2 * args.boundary
        n_m = layout.n_layers - n_b
        bpt_fp16 = layout.kv_dim * 2
        bpt_q = (
            (4 + (layout.kv_dim // 32) * 18) if args.vquant == "turbo4"
            else layout.kv_dim if args.vquant == "int8"
            else layout.kv_dim * 2
        )
        quant_bytes = (n_b * seq * bpt_fp16 + n_m * seq * bpt_q) * 2
    else:
        quant_bytes = quant_storage_bytes(native_arrays, args.vquant, layout.kv_dim)
    storage_ratio = fp16_bytes / quant_bytes if quant_bytes else float("inf")

    print("\n──────── G1 metrics ────────")
    print(f"  TTFT(cold)         {cold_med:7.1f} ms")
    print(f"  TTFT(oracle)       {oracle_med:7.1f} ms  (in-proc native cache)")
    print(f"  TTFT(warm Pion)    {warm_med:7.1f} ms  ({fetch_med:.1f}ms fetch + {warm_med-fetch_med:.1f}ms forward)")
    print(f"  ratio warm/cold    {ratio:.3f}    (target < 0.30)")
    print(f"  first-token cold   {cold_id}")
    print(f"  first-token oracle {oracle_id}  (vs cold: {'OK' if oracle_id == cold_id else 'MISMATCH'})")
    print(f"  first-token warm   {warm_id}  (vs cold: {'OK' if warm_id == cold_id else 'MISMATCH'})")
    print(f"  logprob drift warm vs oracle (top10 cold) {drift:.6f}")
    print(f"  logprob drift oracle vs cold (top10 cold) {drift_oracle_vs_cold:.6f}  (cache-rebuild noise)")
    print(f"  storage ratio fp16/{args.vquant}{'_bdry'+str(args.boundary) if args.boundary else ''}  {storage_ratio:.2f}x")
    print(f"  fetch latency      {fetch_med:.1f} ms for {prefix_len} tokens × {layout.n_layers} layers")
    print(f"  bytes up/down      {bytes_up/1024:.1f} / {bytes_dn/1024:.1f} KB")

    # gh #171: criteria are per-quantization-tier — see tests/g1_criteria.py.
    overall, results = g1_evaluate(args.vquant, ratio, warm_id, cold_id,
                                   oracle_id, drift, storage_ratio, model=args.model)
    g1_print_verdict(args.vquant, results, overall)
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vquant", default="turbo4", choices=["int8", "turbo4", "fp16"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prompt-repeats", type=int, default=4,
                    help="Repeat the system prompt N times to amplify prefill cost")
    ap.add_argument("--boundary", type=int, default=0,
                    help="M4 Phase 4: keep first/last N layers in FP16, middle layers in --vquant")
    args = ap.parse_args()
    sys.exit(main(args))
