"""PionPrefixStore against a live `pion-server --kvcache` (PION_PORT, default
1974) and a small MLX model (PION_TEST_MODEL, default Llama-3.2-1B-4bit).

The equality that matters is not "restored == fresh one-pass prefill" (a
one-pass and a two-pass prefill differ numerically on their own) but
"restored K/V then B" == "locally cached K/V then B", which is exactly what
mlx-lm's own prompt cache does. It must hold bit for bit."""
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from pion_vllm_mlx.prefix_store import PionPrefixStore, to_mx

PORT = int(os.environ.get("PION_PORT", "1974"))
MODEL = os.environ.get("PION_TEST_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit")
KEY = (MODEL, f"test-{os.getpid()}")          # fresh namespace per run


def prefill(model, ids, cache=None):
    cache = cache if cache is not None else make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits, *[c.keys for c in cache])
    return logits[0, -1], cache


def restored(model, store, match, m):
    rows = store.fetch_rows(KEY, match, 0, m)
    assert rows is not None, "fetch failed"
    cache = make_prompt_cache(model)
    lay = store._layout(store.model_id(KEY))
    for c, (k, v) in zip(cache, rows):
        c.update_and_fetch(to_mx(k, lay)[None], to_mx(v, lay)[None])
    return cache


def main():
    model, tok = load(MODEL)
    rng = np.random.default_rng(0)
    A = tok.encode("You are a careful coding agent. " * 150)[:1000]
    B = rng.integers(1000, 20000, 300).tolist()
    C = rng.integers(1000, 20000, 200).tolist()
    D = rng.integers(1000, 20000, 150).tolist()
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += 0 if ok else 1

    s1 = PionPrefixStore(port=PORT)
    _, cA = prefill(model, A)
    check("store A writes 1000 rows", s1.store(KEY, A, cA) == 1000)
    check("store A again writes nothing", s1.store(KEY, A, cA) == 0)

    # A fresh store object = a restarted serve process: nothing in memory.
    s2 = PionPrefixStore(port=PORT)
    mt = s2.lookup(KEY, A + B)
    check(f"lookup(A+B) matches exactly len(A) (got {mt.length})", mt.length == len(A))
    ref_logits, ref_cache = prefill(model, B, prefill(model, A)[1])
    got_logits, got_cache = prefill(model, B, restored(model, s2, mt, len(A)))
    check("restored-then-B logits == local-then-B logits, bit-exact",
          bool(mx.array_equal(ref_logits, got_logits).item()))

    # Extension: A+B+C continues the same lineage in place.
    _, cABC = prefill(model, C, got_cache)
    ext0 = s2.stats.segments_extended
    check("store A+B+C writes 500 rows", s2.store(KEY, A + B + C, cABC) == 500)
    check("A+B+C extended the segment in place", s2.stats.segments_extended == ext0 + 1)

    # Branch: A+D diverges at len(A) -> child segment, A stored once.
    _, cAD = prefill(model, D, restored(model, s2, s2.lookup(KEY, A + D), len(A)))
    cr0 = s2.stats.segments_created
    check("store A+D writes 150 rows", s2.store(KEY, A + D, cAD) == 150)
    check("A+D created a child segment", s2.stats.segments_created == cr0 + 1)

    # The reference is the computation that produced the stored rows (the
    # same passes, in the same order), so the comparison is exact. A one-pass
    # prefill of the whole sequence is a different computation and differs in
    # the last bits on its own; on random token ids it can even flip argmax.
    s3 = PionPrefixStore(port=PORT)
    for name, parts in (("A+B+C", [A, B, C]), ("A+D", [A, D])):
        seq = sum(parts, [])
        mt = s3.lookup(KEY, seq + [7, 8, 9])
        check(f"lookup({name}+x) matches {len(seq)} (got {mt.length})", mt.length == len(seq))
        c = None
        for part in parts:
            _, c = prefill(model, part, c)
        ref = prefill(model, [7, 8, 9], c)[0]
        got = prefill(model, [7, 8, 9], restored(model, s3, mt, len(seq)))[0]
        check(f"{name}: restored-then-x logits == locally-built-then-x, bit-exact",
              bool(mx.array_equal(ref, got).item()))

    # Element-wise: every stored row comes back as the model's own values.
    # For bf16 models the store keeps fp16, which is exact only inside fp16's
    # normal range; count what did not survive rather than assume.
    lay = s3._layouts[s3.model_id(KEY)]
    rows = s3.fetch_rows(KEY, s3.lookup(KEY, A + [1]), 0, len(A))
    bad = total = 0
    for li, (k, v) in enumerate(rows):
        for got_np, src in ((k, cA[li].keys), (v, cA[li].values)):
            ref_np = np.array(src[0, :, :len(A), :].astype(mx.float32))
            restored_np = np.array(to_mx(got_np, lay).astype(mx.float32))
            bad += int((ref_np != restored_np).sum())
            total += ref_np.size
    check(f"stored rows == model rows element-wise ({lay.dtype}, enc={lay.enc}: {bad} of {total} differ)", bad == 0)
    # Power-cut divergence: the metadata (keyspace) says the lineage is longer
    # than the rows the V-store holds. The V-store zero-fills missing rows, so
    # trusting the metadata would restore zeros; the store must refuse, and
    # trim the metadata so the next lookup matches only what exists.
    import redis
    rc = redis.Redis(port=PORT)
    mid = s3.model_id(KEY)
    seg_key = f"pps:{mid}:seg:{s3.lookup(KEY, A + [1]).chain[-1].id}"
    toks = np.frombuffer(rc.hget(seg_key, "toks"), dtype=np.uint32)
    extra = np.arange(5000, 5100, dtype=np.uint32)          # 100 rows that were never stored
    rc.hset(seg_key, mapping={"len": len(toks) + 100, "toks": np.concatenate([toks, extra]).tobytes()})
    s4 = PionPrefixStore(port=PORT)
    seq = A + B + C if len(toks) >= len(A + B + C) else A
    mt = s4.lookup(KEY, list(np.concatenate([toks, extra]).tolist()) + [1])
    check(f"lookup trusts the inflated metadata ({mt.length} >= {len(toks) + 100})", mt.length >= len(toks) + 100)
    check("fetch past the stored rows is refused (None), not zero-filled",
          s4.fetch_rows(KEY, mt, 0, mt.length) is None)
    mt2 = PionPrefixStore(port=PORT).lookup(KEY, list(np.concatenate([toks, extra]).tolist()) + [1])
    check(f"metadata healed: the next lookup stops at the stored rows ({mt2.length} <= {len(toks)})",
          mt2.length <= len(toks))

    print(f"{'ALL PASS' if not fails else f'{fails} FAILED'}  stats={s3.stats.as_dict()}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
