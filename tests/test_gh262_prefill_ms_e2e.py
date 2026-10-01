#!/usr/bin/env python3
"""gh #262 end to end: a real PionPromptCache cold prefill sends PREFILL_MS,
a second client's hit is credited with exactly that measured number, and the
hit's cache produces the same next token as a local cold prefill.

Not download-free: needs mlx-lm and mlx-community/Llama-3.2-1B-Instruct-4bit
cached, and a release ./pion-server (it starts its own --kvcache server).

The load-bearing check is "PREFILL_MS is a full prefill". The client's barrier
used to eval layer 0's K only; MLX is lazy, so the clock stopped after ~1/16
of the work and a 2K-token prefill was credited as 23 ms (real: ~1.2 s). Every
other check passed on that code — the wire and the ledger were right, the
number fed into them was not.

    python3 tests/test_gh262_prefill_ms_e2e.py
"""
import os, shutil, socket, subprocess, sys, tempfile, time
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "pion-vllm-mlx"))
sys.path.insert(0, os.path.join(REPO, "examples"))
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from pion_vllm_mlx import PionPromptCache
from prompt_cache_demo import _build_prefix

PORT = int(os.environ.get("PORT", "19262"))
fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    fails += (not ok)

import redis
def rc(): return redis.Redis(port=PORT)

def stats():
    r = rc().execute_command("PION.STATS")
    out = {}
    for k, v in zip(r[::2], r[1::2]):
        out[k.decode()] = str(v.decode() if isinstance(v, bytes) else v)
    return out

work = tempfile.mkdtemp(prefix="gh262e2e_")
proc = subprocess.Popen([os.environ.get("PION_BIN") or os.path.join(REPO, "pion-server"), "-p", str(PORT), "-w", "1", "--kvcache",
                         "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                        cwd=work, stdout=open(os.path.join(work, "server.log"), "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(240):
        try: socket.create_connection(("127.0.0.1", PORT)).close(); break
        except OSError: time.sleep(0.25)
    model, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    ids = tok.encode(_build_prefix(tok, 2048), add_special_tokens=False)
    print(f"prefix tokens: {len(ids)}")
    ns = "gh262e2e|llama321b|2048"

    a = PionPromptCache(model, host="127.0.0.1", port=PORT)
    cache_a = a.get_or_prefill(ids, namespace=ns)
    sent_ms = int(round(a.last_prefill_ms))
    check("cold pass was a miss", a.misses == 1 and a.hits == 0, f"misses={a.misses} hits={a.hits}")
    check("client measured a real prefill", sent_ms > 50, f"{a.last_prefill_ms:.1f} ms")
    s0 = stats()
    check("server recorded the miss", s0.get("kvprefix_misses") == "1", s0.get("kvprefix_misses"))

    b = PionPromptCache(model, host="127.0.0.1", port=PORT)   # fresh client: no local state
    cache_b = b.get_or_prefill(ids, namespace=ns)
    check("second client hit", b.hits == 1 and b.misses == 0, f"hits={b.hits} misses={b.misses}")
    check("the hit did not prefill", b.last_prefill_ms == 0.0, b.last_prefill_ms)
    s1 = stats()
    check("one hit", s1.get("kvprefix_hits") == "1", s1.get("kvprefix_hits"))
    check("it is a MEASURED hit", s1.get("kvprefix_hits_measured") == "1", s1.get("kvprefix_hits_measured"))
    check("credited exactly the client's PREFILL_MS",
          s1.get("kvprefix_prefill_us_measured") == str(sent_ms * 1000),
          f"server={s1.get('kvprefix_prefill_us_measured')} client={sent_ms*1000}")
    check("seconds text matches", s1.get("prefill_seconds_avoided_measured") == f"{sent_ms/1000:.3f}",
          s1.get("prefill_seconds_avoided_measured"))
    info = rc().info("all")
    check("INFO agrees", abs(float(info.get("prefill_seconds_avoided", -1)) - sent_ms/1000) < 1e-9,
          info.get("prefill_seconds_avoided"))

    # Correctness: next token after a suffix, hit cache vs a local cold prefill.
    # Timed too: an independent full prefill is what PREFILL_MS claims to be.
    t0 = time.perf_counter()
    ref = make_prompt_cache(model); model(mx.array(ids)[None], cache=ref); mx.eval([c.state for c in ref])
    ref_ms = (time.perf_counter() - t0) * 1000
    check("PREFILL_MS is a full prefill (0.5x-2x an independent one)", 0.5 <= sent_ms / ref_ms <= 2.0,
          f"sent={sent_ms} ms independent={ref_ms:.1f} ms")
    suf = mx.array(tok.encode(" Question: what is the capital of France? Answer:", add_special_tokens=False))[None]
    t_ref = mx.argmax(model(suf, cache=ref)[0, -1]).item()
    t_hit = mx.argmax(model(suf, cache=cache_b)[0, -1]).item()
    check("hit cache gives the same next token as cold prefill", t_ref == t_hit, f"{t_ref} vs {t_hit}")
    for k in sorted(s1): print(f"  {k}:{s1[k]}")
finally:
    proc.terminate(); proc.wait(timeout=15)
    shutil.rmtree(work, ignore_errors=True)   # 128 MB of WAL per run otherwise
print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
