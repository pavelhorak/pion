#!/usr/bin/env python3
"""Stage 1 prototype: shared KV cache across requests via Pion's V.STOREBATCH/V.FETCH.

Implements the gate G1 measurement defined in the shared-KV-cache design
sections 9 and 15.2.

Flow:
  1. COLD: tokenize [system_prompt + user_query_A], full forward, record TTFT + first-token logits.
  2. STORE: forward [system_prompt] only, capture per-layer (K, V) cache, push into Pion via V.STOREBATCH.
  3. WARM: tokenize [system_prompt + user_query_B], fetch the prefix K/V from Pion,
           build past_key_values, forward ONLY the user_query_B tokens with the cached prefix.
  4. ORACLE: same as WARM but with the model-native past_key_values (no quant round-trip)
            to isolate quantization drift from infrastructure overhead.

Pass criteria (from §15.2 G1):
  TTFT(warm)/TTFT(cold)   < 0.30
  First-token agreement   = 100%   (warm vs oracle full forward of [prefix+B])
  Logprob delta next 10   < per-tier ceiling (gh #171 — tests/g1_criteria.py)
  Storage efficiency       > per-tier floor; not asserted for fp16
  Fetch latency            < 5ms for 2K-token prefix (we use shorter for gpt2)

Requires: ./pion-server --kvcache -w 1
Model: gpt2 (12 layers, hidden=768) -- divisible by 32 so turbo4 works.
"""

from __future__ import annotations

import hashlib
import socket
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np

from g1_criteria import evaluate as g1_evaluate, print_verdict as g1_print_verdict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
except ImportError as e:
    print(f"missing dep: {e}")
    sys.exit(1)


HOST = "127.0.0.1"
PORT = 1974
MODEL_NAME = "gpt2"
DEVICE = "cpu"  # MPS adds noise to TTFT; CPU is steadier for small models
DTYPE = torch.float32

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
    """Larger prefix means prefill dominates and the warm-path win shows up.

    gpt2 caps at 1024 positions; n_repeats=4 gives ~600 tokens with headroom for the suffix.
    """
    return _SYSTEM_TEMPLATE * n_repeats


# ─────────────────────────────────────────────────────────────────────────────
# Pion RESP client (V.* commands)
# ─────────────────────────────────────────────────────────────────────────────


class PionVStore:
    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.sock.settimeout(20)
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
        # Read until we have one complete top-level reply.
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
        return nl + 2  # crude — no nested arrays expected

    def send(self, parts) -> bytes:
        self.sock.sendall(self._encode(parts))
        return self._read_complete()

    def create(self, sid: str, dim: int, vquant: str = "int8") -> bool:
        parts = ["V.CREATE", sid, str(dim)]
        if vquant != "int8":
            parts += ["VQUANT", vquant]
        r = self.send(parts)
        return r.startswith(b":")

    def storebatch(self, sid: str, layer: int, start_id: int, values_fp32: np.ndarray) -> bool:
        n = values_fp32.shape[0]
        r = self.send([
            "V.STOREBATCH", sid, str(layer), str(start_id), str(n),
            np.ascontiguousarray(values_fp32, dtype=np.float32).tobytes(),
        ])
        return r.startswith(b"+OK")

    def fetch(self, sid: str, layer: int, token_ids) -> np.ndarray | None:
        parts = ["V.FETCH", sid, str(layer)] + [str(t) for t in token_ids]
        return self._read_blob(self.send(parts))

    def fetch_range(self, sid: str, layer: int, start: int, end: int) -> np.ndarray | None:
        """Single-round-trip prefix fetch via V.FETCH ... RANGE start end."""
        return self._read_blob(self.send(["V.FETCH", sid, str(layer), "RANGE", str(start), str(end)]))

    @staticmethod
    def _read_blob(r: bytes) -> np.ndarray | None:
        if not r.startswith(b"$") or r.startswith(b"$-1"):
            return None
        nl = r.find(b"\r\n")
        blen = int(r[1:nl])
        blob = r[nl + 2:nl + 2 + blen]
        return np.frombuffer(blob, dtype=np.float32).copy()

    def info(self, sid: str | None = None) -> str:
        parts = ["V.INFO"] + ([sid] if sid else [])
        r = self.send(parts)
        if r.startswith(b"$"):
            nl = r.find(b"\r\n")
            n = int(r[1:nl])
            return r[nl + 2:nl + 2 + n].decode("utf-8", errors="replace")
        return r.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Cache <-> Pion bridge
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CacheLayout:
    n_layers: int
    n_heads: int
    head_dim: int

    @property
    def dim(self) -> int:
        return self.n_heads * self.head_dim


def pkv_to_arrays(pkv, layout: CacheLayout):
    """Convert HF cache (legacy tuple OR DynamicCache) -> per-layer (K, V) numpy [seq, dim]."""
    if isinstance(pkv, DynamicCache):
        out = []
        for li in range(len(pkv)):
            k = pkv.layers[li].keys      # [B, H, S, D]
            v = pkv.layers[li].values
            out.append(_flatten_kv(k, v, layout))
        return out
    out = []
    for layer in pkv:
        k, v = layer
        out.append(_flatten_kv(k, v, layout))
    return out


def _flatten_kv(k: torch.Tensor, v: torch.Tensor, layout: CacheLayout):
    assert k.shape[0] == 1, "batch=1 only"
    seq = k.shape[2]
    k_np = k[0].permute(1, 0, 2).reshape(seq, layout.dim).contiguous().to(torch.float32).cpu().numpy()
    v_np = v[0].permute(1, 0, 2).reshape(seq, layout.dim).contiguous().to(torch.float32).cpu().numpy()
    return k_np.copy(), v_np.copy()


def arrays_to_pkv(per_layer, layout: CacheLayout):
    """Convert per-layer (K_np, V_np) -> DynamicCache populated through update()."""
    cache = DynamicCache()
    for li, (k_np, v_np) in enumerate(per_layer):
        seq = k_np.shape[0]
        k_t = torch.from_numpy(k_np).reshape(seq, layout.n_heads, layout.head_dim).permute(1, 0, 2).unsqueeze(0).to(DTYPE)
        v_t = torch.from_numpy(v_np).reshape(seq, layout.n_heads, layout.head_dim).permute(1, 0, 2).unsqueeze(0).to(DTYPE)
        cache.update(k_t, v_t, li)
    return cache


def store_prefix(
    pion: PionVStore,
    prefix_hash: str,
    per_layer,
    layout: CacheLayout,
    vquant: str,
    boundary_layers: int = 0,
):
    """Push K/V tensors into Pion.

    If boundary_layers > 0, layers [0..b) and [N-b..N) go into a FP16-quant
    side-session; middle layers go into the main `vquant` session. This is the
    M4 Phase 4 "boundary-layer protection" pattern.

    Returns (sid_k_main, sid_v_main, sid_k_b|None, sid_v_b|None, store_ms, bytes_sent, layout_summary).
    """
    sid_k = f"{prefix_hash}_k_{vquant}"
    sid_v = f"{prefix_hash}_v_{vquant}"
    assert pion.create(sid_k, layout.dim, vquant=vquant)
    assert pion.create(sid_v, layout.dim, vquant=vquant)
    sid_kb = sid_vb = None
    if boundary_layers > 0:
        sid_kb = f"{prefix_hash}_k_fp16"
        sid_vb = f"{prefix_hash}_v_fp16"
        assert pion.create(sid_kb, layout.dim, vquant="fp16")
        assert pion.create(sid_vb, layout.dim, vquant="fp16")

    bytes_sent = 0
    n_boundary = 0
    n_main = 0
    t0 = time.perf_counter()
    for li, (k_np, v_np) in enumerate(per_layer):
        is_boundary = boundary_layers > 0 and (
            li < boundary_layers or li >= layout.n_layers - boundary_layers
        )
        if is_boundary:
            assert pion.storebatch(sid_kb, li, 0, k_np)
            assert pion.storebatch(sid_vb, li, 0, v_np)
            n_boundary += 1
        else:
            assert pion.storebatch(sid_k, li, 0, k_np)
            assert pion.storebatch(sid_v, li, 0, v_np)
            n_main += 1
        bytes_sent += k_np.nbytes + v_np.nbytes
    summary = f"main({vquant})={n_main} boundary(fp16)={n_boundary}"
    return sid_k, sid_v, sid_kb, sid_vb, (time.perf_counter() - t0) * 1000, bytes_sent, summary


def fetch_prefix(
    pion: PionVStore,
    layout: CacheLayout,
    prefix_len: int,
    sid_k_main: str,
    sid_v_main: str,
    sid_k_boundary: str | None = None,
    sid_v_boundary: str | None = None,
    boundary_layers: int = 0,
):
    """Pull all K/V layers from Pion in a single round-trip per layer side via V.FETCH RANGE.

    If `sid_k_boundary` / `sid_v_boundary` are given, the first and last
    `boundary_layers` layers are fetched from those (FP16) sessions; middle
    layers come from the main (quantized) sessions.
    """
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
            raise RuntimeError(f"V.FETCH RANGE None at layer {li} (boundary={is_boundary})")
        bytes_received += k_flat.nbytes + v_flat.nbytes
        per_layer.append((k_flat.reshape(prefix_len, layout.dim), v_flat.reshape(prefix_len, layout.dim)))
    return per_layer, (time.perf_counter() - t0) * 1000, bytes_received


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────


def time_first_token(model, input_ids, past_key_values=None) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Run one forward, return (ttft_ms, first_token_id, full_log_softmax_over_vocab).

    The full log-softmax is returned so callers can probe it at fixed indices for
    cross-run drift measurement.
    """
    if DEVICE != "cpu":
        torch.mps.synchronize() if DEVICE == "mps" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
    if DEVICE == "mps":
        torch.mps.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000
    last_logits = out.logits[0, -1].float()
    logprobs = torch.log_softmax(last_logits, dim=-1)
    first_id = torch.argmax(last_logits).unsqueeze(0)
    return ttft_ms, first_id, logprobs


def quant_storage_bytes(per_layer, fmt: str, dim: int) -> int:
    """Storage footprint in bytes for K and V combined under a given quant fmt."""
    n_layers = len(per_layer)
    seq = per_layer[0][0].shape[0]
    if fmt == "fp16":
        bpt = dim * 2
    elif fmt == "int8":
        bpt = dim
    elif fmt == "turbo4":
        bpt = 4 + (dim // 32) * 18
    elif fmt == "fp32":
        bpt = dim * 4
    else:
        raise ValueError(fmt)
    return n_layers * seq * bpt * 2  # K + V


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(vquant: str = "turbo4", warmup: int = 2, repeats: int = 5, prompt_repeats: int = 4,
         boundary_layers: int = 0) -> int:
    print(f"Stage 1 prototype  model={MODEL_NAME} device={DEVICE} vquant={vquant} "
          f"prompt_repeats={prompt_repeats} boundary={boundary_layers}")
    print(f"  pion: {HOST}:{PORT}")
    SYSTEM = system_prompt(prompt_repeats)

    # Sanity: pion reachable
    try:
        s = socket.create_connection((HOST, PORT), timeout=2)
        s.close()
    except OSError as e:
        print(f"FAIL: pion not reachable ({e}). Start with: ./pion-server --kvcache -w 1")
        return 2

    print("loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    model.eval()
    cfg = model.config
    layout = CacheLayout(n_layers=cfg.n_layer, n_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head)
    print(f"  layers={layout.n_layers} heads={layout.n_heads} head_dim={layout.head_dim} dim={layout.dim}")

    # Tokenize. The "prefix" is exactly the system prompt. Both queries reuse it.
    prefix_ids = tok(SYSTEM, return_tensors="pt").input_ids.to(DEVICE)
    full_a_ids = tok(SYSTEM + USER_A, return_tensors="pt").input_ids.to(DEVICE)
    full_b_ids = tok(SYSTEM + USER_B, return_tensors="pt").input_ids.to(DEVICE)
    prefix_len = prefix_ids.shape[1]
    suffix_b_len = full_b_ids.shape[1] - prefix_len
    suffix_b_ids = full_b_ids[:, prefix_len:]
    print(f"  prefix_tokens={prefix_len}  suffix_b_tokens={suffix_b_len}")
    if suffix_b_len <= 0:
        print("FAIL: tokenizer produced overlapping prefix/suffix; pick a different USER_B.")
        return 2

    # Sanity: prefix is a true prefix of full_b_ids
    assert torch.equal(full_b_ids[:, :prefix_len], prefix_ids), "tokenizer reproducibility check failed"

    # Warmup torch + python (excluded from timing)
    for _ in range(warmup):
        time_first_token(model, full_a_ids)
        time_first_token(model, full_b_ids)

    # ── COLD PATH: full forward of [system + B], measured ───────────────────
    print("\n[cold] full forward of [system + B]")
    cold_ttft = []
    cold_first = None
    cold_logp = None
    for _ in range(repeats):
        ttft, first, logp = time_first_token(model, full_b_ids)
        cold_ttft.append(ttft)
        cold_first = first
        cold_logp = logp
    cold_med = float(np.median(cold_ttft))
    print(f"  TTFT median={cold_med:.1f}ms  first_token_id={cold_first.item()}")

    # ── ORACLE: in-process prefix cache (no Pion, no quant) ─────────────────
    # Establishes the upper bound: how fast can the warm path possibly be?
    print("\n[oracle] forward(prefix), then forward(B-suffix) with native pkv")
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
    native_pkv = prefix_out.past_key_values
    oracle_ttft = []
    oracle_first = None
    oracle_logp = None
    for _ in range(repeats):
        # Re-build cache each iteration; the model mutates DynamicCache in place.
        per_layer = pkv_to_arrays(native_pkv, layout)
        rebuilt = arrays_to_pkv(per_layer, layout)
        ttft, first, logp = time_first_token(model, suffix_b_ids, past_key_values=rebuilt)
        oracle_ttft.append(ttft)
        oracle_first = first
        oracle_logp = logp
    oracle_med = float(np.median(oracle_ttft))
    print(f"  TTFT median={oracle_med:.1f}ms  first_token_id={oracle_first.item()}")

    if oracle_first.item() != cold_first.item():
        print(f"WARN: oracle first-token != cold first-token "
              f"({oracle_first.item()} vs {cold_first.item()}). "
              f"This is an HF cache-reconstruction artifact, not a Pion issue.")

    # ── STORE: push K/V into Pion ───────────────────────────────────────────
    print(f"\n[store] uploading prefix K/V to Pion ({vquant}, boundary={boundary_layers})")
    pion = PionVStore()
    per_layer_native = pkv_to_arrays(native_pkv, layout)
    # Cache key includes vquant + boundary count (per GPT-5 §11.3: namespace must encode full execution context).
    prefix_hash = hashlib.sha256(
        f"{MODEL_NAME}|{tok.__class__.__name__}|{vquant}|b{boundary_layers}|{SYSTEM}".encode()
    ).hexdigest()[:16]
    sid_k, sid_v, sid_kb, sid_vb, store_ms, bytes_up, store_summary = store_prefix(
        pion, prefix_hash, per_layer_native, layout, vquant, boundary_layers
    )
    print(f"  store_ms={store_ms:.1f}  bytes_up={bytes_up/1024:.1f}KB  layout: {store_summary}")
    print(f"  V.INFO main: {pion.info(sid_k).strip()[:100]}...")

    # ── WARM PATH: fetch prefix from Pion via V.FETCH RANGE ─────────────────
    print(f"\n[warm] V.FETCH RANGE prefix, forward(B-suffix) with reconstructed pkv")
    warm_ttft_full = []
    warm_fetch_ms = []
    warm_first = None
    warm_logp = None
    for _ in range(repeats):
        per_layer_q, fetch_ms, bytes_dn = fetch_prefix(
            pion, layout, prefix_len,
            sid_k_main=sid_k, sid_v_main=sid_v,
            sid_k_boundary=sid_kb, sid_v_boundary=sid_vb,
            boundary_layers=boundary_layers,
        )
        rebuilt = arrays_to_pkv(per_layer_q, layout)
        ttft, first, logp = time_first_token(model, suffix_b_ids, past_key_values=rebuilt)
        warm_fetch_ms.append(fetch_ms)
        warm_ttft_full.append(fetch_ms + ttft)
        warm_first = first
        warm_logp = logp
    warm_med = float(np.median(warm_ttft_full))
    fetch_med = float(np.median(warm_fetch_ms))
    print(f"  TTFT median={warm_med:.1f}ms  (fetch={fetch_med:.1f}ms + forward={warm_med-fetch_med:.1f}ms)")
    print(f"  bytes_dn={bytes_dn/1024:.1f}KB  first_token_id={warm_first.item()}")

    # ── METRICS ─────────────────────────────────────────────────────────────
    # Compare warm vs cold (full system+B forward)
    ratio = warm_med / cold_med if cold_med > 0 else float("inf")
    # Quantization drift: probe fixed indices (cold path's top-10) on both runs so we
    # are comparing the same tokens. Mismatched top-10 sets make a rank-sorted compare
    # meaningless.
    cold_top10 = torch.topk(cold_logp, k=10).indices
    drift = float(torch.mean(torch.abs(warm_logp[cold_top10] - oracle_logp[cold_top10])).item())
    drift_oracle_vs_cold = float(torch.mean(torch.abs(oracle_logp[cold_top10] - cold_logp[cold_top10])).item())
    # First-token agreement (both warm and oracle vs cold; both should match)
    cold_id = cold_first.item()
    oracle_id = oracle_first.item()
    warm_id = warm_first.item()

    fp16_bytes = quant_storage_bytes(per_layer_native, "fp16", layout.dim)
    if boundary_layers > 0:
        # Mixed: 2 * boundary_layers stay fp16, the rest go to vquant
        seq = per_layer_native[0][0].shape[0]
        n_b = 2 * boundary_layers
        n_m = layout.n_layers - n_b
        bpt_fp16 = layout.dim * 2
        bpt_q = (
            (4 + (layout.dim // 32) * 18) if vquant == "turbo4"
            else layout.dim if vquant == "int8"
            else layout.dim * 2
        )
        # K + V both stored, hence ×2
        quant_bytes = (n_b * seq * bpt_fp16 + n_m * seq * bpt_q) * 2
    else:
        quant_bytes = quant_storage_bytes(per_layer_native, vquant, layout.dim)
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
    print(f"  storage ratio fp16/{vquant}  {storage_ratio:.2f}x")
    print(f"  fetch latency       {fetch_med:.1f} ms for {prefix_len} tokens × {layout.n_layers} layers")
    print(f"  bytes up/down       {bytes_up/1024:.1f} / {bytes_dn/1024:.1f} KB")

    # gh #171: criteria are per-quantization-tier — see tests/g1_criteria.py
    # for why one shared threshold could never pass.
    overall, results = g1_evaluate(vquant, ratio, warm_id, cold_id, oracle_id,
                                   drift, storage_ratio, model=MODEL_NAME)
    g1_print_verdict(vquant, results, overall)
    return 0 if overall else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vquant", default="turbo4", choices=["int8", "turbo4", "fp16"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--prompt-repeats", type=int, default=4,
                    help="Repeat the system prompt N times to amplify prefill cost")
    ap.add_argument("--boundary", type=int, default=0,
                    help="M4 Phase 4: keep first/last N layers in FP16, middle layers in --vquant")
    args = ap.parse_args()
    sys.exit(main(vquant=args.vquant, warmup=args.warmup, repeats=args.repeats,
                  prompt_repeats=args.prompt_repeats, boundary_layers=args.boundary))
