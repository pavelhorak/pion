#!/usr/bin/env python3
"""mlx-lm's own prompt-cache FILE as the warm baseline for Pion's TTFT rows.

mlx-lm ships a persistent prompt cache: `save_prompt_cache` writes a prompt's
cache to a .safetensors file and `load_prompt_cache` maps it back (the
`mlx_lm.cache_prompt` CLI is the same two calls). It is free, and it is the
first thing a reader should compare a prompt cache against. This harness times
it exactly the way `cross_process_ttft.py` times a Pion hit:

  store  a process loads the model, prefills the prefix in 2,048-token chunks
         (mlx_lm.generate_step's shape), saves the cache file, then answers the
         question's first token from the same cache: the reference token.
  hit    a NEW process loads the model, runs a short untimed warm-up forward
         (so Metal's one-time kernel compile is outside the clock), then times
         load_prompt_cache(file) + the question's first token.

Model load is outside the clock on both sides, as in cross_process_ttft.py.
Every hit must produce the store's first token, or the script fails. The file
sits in the page cache when it is read (the store just wrote it); a read from
a cold disk adds the file's size at the SSD's read speed. Pion's own numbers
are a RAM-resident server, so page-cache-warm is the like comparison.

Workloads:
  llama  Llama-3.2-1B-Instruct-4bit, the 2,049-token prefix and 16-token
         question of cross_process_ttft.py (its builder, imported).
  gemma  Gemma-4-E2B-it-4bit, the 63,961-token prefix of
         examples/sparse_mask_64k_niah.py (its builder and loader, seed 0
         needle). The file holds every layer's full state, sliding-window
         layers included; the hit attends the WHOLE prefix (dense), where
         that example's Pion row attends 512 tokens per full-attention layer.
  ssm    mamba-130m-hf-f32: a recurrent (SSM) model, whose cache is
         ArraysCache. Checks that the file round-trips recurrent state: the
         hit's first token equals the cold path's.

    python3 benchmarks/reproducers/file_cache_ttft.py --workload llama --pairs 3 \\
        --out benchmarks/reproducers/results/file_cache_ttft_llama.json

Needs Apple Silicon, mlx-lm and the workload's model in the Hugging Face
cache. No Pion server. It is a timing measurement: on a loaded machine it
produces a wrong number, not an error. The cache files go to a temporary
directory (--keep keeps them); the 64K Gemma file is about 0.4 GB.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFILL_STEP = 2048
SSM_MODEL = "mlx-community/mamba-130m-hf-f32"
GEMMA_MODEL = "mlx-community/gemma-4-e2b-it-4bit"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prefill(model, mx, cache, ids) -> None:
    """Every token of `ids` into `cache`, in generate_step's chunks, evaluating
    only the cache (no logits)."""
    x = mx.array(ids)[None]
    done, n = 0, x.shape[1]
    while done < n:
        step = min(PREFILL_STEP, n - done)
        model(x[:, done:done + step], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        done += step


def first_token(model, mx, cache, ids) -> int:
    """The first generated token after `ids`, the way generate_step produces
    it: all but the last token as prefill, then the last token alone."""
    if len(ids) > 1:
        prefill(model, mx, cache, ids[:-1])
    tok_id = mx.argmax(model(mx.array(ids[-1:])[None], cache=cache)[0, -1])
    mx.eval(tok_id)
    return int(tok_id.item())


def niah_needle(seed: int = 0) -> tuple[str, int]:
    # The same draw as examples/sparse_mask_64k_niah.py's main() with --seed 0.
    rng = random.Random(seed)
    city = rng.choice(["Petropavlovsk-Kamchatsky", "Ouagadougou", "Antananarivo",
                       "Bratislava", "Wagga Wagga", "Trondheim"])
    return city, rng.randint(10000, 99999)


def workload(name: str, n: int):
    """(model, tokenizer, prefix ids, question ids, model id) for a workload."""
    from mlx_lm import load

    if name == "llama":
        repro = _load_module(REPO_ROOT / "benchmarks/reproducers/cross_process_ttft.py", "cross_process_ttft")
        model, tok = load(repro.MODEL)
        return model, tok, repro.build_prefix_ids(tok, n), tok.encode(repro.QUESTION, add_special_tokens=False), repro.MODEL
    if name == "gemma":
        niah = _load_module(REPO_ROOT / "examples/sparse_mask_64k_niah.py", "sparse_mask_64k_niah")
        model, tok = niah.load_text_only_from_cached(GEMMA_MODEL)
        city, number = niah_needle(0)
        full, question = niah.build_prompt(n, 0.5, city, number, tok)
        return model, tok, full[: len(full) - len(question)], question, GEMMA_MODEL
    if name == "ssm":
        model, tok = load(SSM_MODEL)
        text = ("The Eiffel Tower is 330 metres tall and located in Paris. It was completed in 1889. "
                "The Golden Gate Bridge has a main span of 1280 metres. ")
        # One <bos> (if the tokenizer has one) first, then pieces without
        # special tokens: tests/test_audit_prompt_bos.py's rule.
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        piece = tok.encode(text, add_special_tokens=False)
        ids = list(bos)
        while len(ids) < n:
            ids = ids + piece
        question = tok.encode(" Question: how tall is the Eiffel Tower? Answer:", add_special_tokens=False)
        return model, tok, ids[:n], question, SSM_MODEL
    raise SystemExit(f"unknown workload {name!r}")


def child(mode: str, name: str, n: int, path: str) -> dict:
    import mlx.core as mx
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    model, _tok, prefix, question, model_id = workload(name, n)
    first_token(model, mx, make_prompt_cache(model), prefix[:64] + question)     # warm-up, untimed

    if mode == "store":
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        prefill(model, mx, cache, prefix)
        prefill_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        save_prompt_cache(path, cache, {"prefix_tokens": str(len(prefix)), "model": model_id})
        save_ms = (time.perf_counter() - t1) * 1000
        ref = first_token(model, mx, cache, question)       # the cold path's answer
        return {"mode": "store", "prefix_tokens": len(prefix), "question_tokens": len(question),
                "prefill_ms": round(prefill_ms, 1), "save_ms": round(save_ms, 1),
                "file_bytes": Path(path).stat().st_size, "first_token": ref,
                "cache_classes": sorted({type(c).__name__ for c in cache})}

    t0 = time.perf_counter()
    cache = load_prompt_cache(path)
    load_ms = (time.perf_counter() - t0) * 1000
    tok_id = first_token(model, mx, cache, question)
    ms = (time.perf_counter() - t0) * 1000
    return {"mode": "hit", "prefix_tokens": len(prefix), "ms": round(ms, 2), "load_ms": round(load_ms, 3),
            "first_token": tok_id}


def run_child(mode: str, name: str, n: int, path: str) -> dict:
    r = subprocess.run([sys.executable, __file__, "--child", mode, name, str(n), path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{mode} {name} failed")
    return json.loads(r.stdout.strip().splitlines()[-1])


def environment() -> dict:
    import mlx.core as mx
    import mlx_lm

    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        chip, mem = platform.processor(), 0
    return {"chip": chip, "memory_gb": round(mem / 2**30), "macos": platform.mac_ver()[0],
            "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "python": platform.python_version()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", choices=["llama", "gemma", "ssm"], default="llama")
    ap.add_argument("--prefix-tokens", type=int, default=None,
                    help="prefix length (default: llama 2049, gemma 64000 context, ssm 2048)")
    ap.add_argument("--pairs", type=int, default=3, help="hits, each from a fresh process")
    ap.add_argument("--out", help="write the raw runs as JSON here")
    ap.add_argument("--keep", action="store_true", help="keep the cache file")
    ap.add_argument("--child", nargs=4, metavar=("MODE", "WORKLOAD", "TOKENS", "PATH"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child[0], a.child[1], int(a.child[2]), a.child[3])))
        return 0

    n = a.prefix_tokens or {"llama": 2049, "gemma": 64000, "ssm": 2048}[a.workload]
    work = Path(tempfile.mkdtemp(prefix="pion_file_cache_ttft_"))
    path = str(work / f"{a.workload}_{n}.safetensors")
    try:
        store = run_child("store", a.workload, n, path)
        print(json.dumps(store), flush=True)
        hits = []
        for _ in range(a.pairs):
            h = run_child("hit", a.workload, n, path)
            print(json.dumps(h), flush=True)
            hits.append(h)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)

    tokens = {h["first_token"] for h in hits} | {store["first_token"]}
    same = len(tokens) == 1
    med = statistics.median(h["ms"] for h in hits)
    report = {"workload": a.workload, "measured": time.strftime("%Y-%m-%d"), "environment": environment(),
              "prefix_tokens": store["prefix_tokens"], "file_bytes": store["file_bytes"],
              "file_mb": round(store["file_bytes"] / 1e6, 1), "cache_classes": store["cache_classes"],
              "hit_median_ms": round(med, 1), "first_token_matches_cold": same,
              "store": store, "hits": hits}
    print(f"\n{a.workload}: {store['prefix_tokens']:,}-token prefix, file {report['file_mb']} MB "
          f"({', '.join(store['cache_classes'])}); hit median {med:.1f} ms over {len(hits)} fresh processes; "
          f"first token {'matches' if same else 'DIFFERS FROM'} the cold path")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"raw runs -> {a.out}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
