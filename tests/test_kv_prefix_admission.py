#!/usr/bin/env python3
"""PionPromptCache admission policy — closes §13.3 day 5 (other half of LRU).

Per the shared-KV-cache admission design (§11.5 / §13.3 day 5):
  "Do not cache every prefix on first sight. Use a simple 2-hit admission rule
   so RAM is spent on reusable prefixes, not one-off prompts."

PionPromptCache(admission_threshold=N) only pushes K/V to Pion after a
namespace has been observed N times. Below threshold, the prefill still runs
locally and a correct cache is returned — admission only gates the V.STOREBATCH
side effect, not the user-facing answer.

What this test checks:
  1. threshold=1 (default): each miss triggers REGISTER + STOREBATCH.
  2. threshold=2: first miss skips Pion, second miss registers, third call hits.
  3. one-off namespace at threshold=2: never pushes (admission_skips counted).

Requires: ./pion-server --kvcache -w 1
"""
from __future__ import annotations

import socket
import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "pion-vllm-mlx")
from pion_vllm_mlx.prompt_cache import PionPromptCache


MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def assert_pion_up() -> None:
    s = socket.create_connection(("127.0.0.1", 1974), timeout=2)
    s.close()


def short_prefix(tok, txt: str):
    return tok.encode(txt)[:40]


def main() -> int:
    assert_pion_up()
    print("loading model...")
    model, tok = load(MODEL)

    # ── threshold=1 (default) ─────────────────────────────────────────────
    pc1 = PionPromptCache(model, vquant="fp16")
    ns1 = PionPromptCache.make_namespace("admission_test", "threshold1", "ns1")
    pc1.get_or_prefill(short_prefix(tok, "alpha beta gamma delta"), ns1)
    pc1.get_or_prefill(short_prefix(tok, "alpha beta gamma delta"), ns1)
    s1 = pc1.stats()
    print(f"[t=1] {s1}")
    assert s1["misses"] == 1 and s1["hits"] == 1, f"t=1 expected 1m+1h, got {s1}"
    assert s1["admission_skips"] == 0, f"t=1 should not skip, got {s1['admission_skips']}"

    # ── threshold=2 (skip one-off) ────────────────────────────────────────
    pc2 = PionPromptCache(model, vquant="fp16", admission_threshold=2)
    ns2 = PionPromptCache.make_namespace("admission_test", "threshold2", "ns2")

    # Call 1: skipped (observed=1 < 2). No REGISTER, no STOREBATCH.
    pc2.get_or_prefill(short_prefix(tok, "epsilon zeta eta theta"), ns2)
    s = pc2.stats()
    assert s["misses"] == 1 and s["hits"] == 0 and s["admission_skips"] == 1, (
        f"t=2 call1 expected 1m+0h+1skip, got {s}"
    )

    # Call 2: now admitted. REGISTER + STOREBATCH happen. Still a miss because
    # LOOKUP returned MISS at call start (we hadn't pushed yet).
    pc2.get_or_prefill(short_prefix(tok, "epsilon zeta eta theta"), ns2)
    s = pc2.stats()
    assert s["misses"] == 2 and s["hits"] == 0 and s["admission_skips"] == 1, (
        f"t=2 call2 expected 2m+0h+1skip, got {s}"
    )

    # Call 3: HIT now that the prefix is in Pion.
    pc2.get_or_prefill(short_prefix(tok, "epsilon zeta eta theta"), ns2)
    s = pc2.stats()
    assert s["misses"] == 2 and s["hits"] == 1 and s["admission_skips"] == 1, (
        f"t=2 call3 expected 2m+1h+1skip, got {s}"
    )
    print(f"[t=2 hot prefix] {s}")

    # ── one-off namespace stays out of Pion ───────────────────────────────
    ns_oneoff = PionPromptCache.make_namespace("admission_test", "threshold2", "oneoff")
    pc2.get_or_prefill(short_prefix(tok, "iota kappa lambda mu"), ns_oneoff)
    s = pc2.stats()
    assert s["admission_skips"] == 2, f"one-off should add 1 skip, got {s['admission_skips']}"
    # Re-checking via lookup: the one-off namespace was never registered.
    assert not pc2.lookup(ns_oneoff), "one-off namespace should not be in Pion"
    print(f"[t=2 one-off blocked] {s}")

    print("[admission] PASS — threshold gates V.STOREBATCH without breaking prefill semantics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
