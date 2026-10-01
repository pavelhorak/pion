#!/usr/bin/env python3
"""Segmented (quicklist) lists pop the right element from either end.

Found 2026-09-27 while freeing containers (gh #369): past 1,024 entries a list
switches to a segmented representation, and its pop paths broke its own
invariant —
  * RPOP, once the tail side was empty, took the LAST element of the oldest
    head segment and freed the whole 256-slot segment: 255 elements silently
    gone while `size` still counted them. `RPUSH 0..1499` then RPOP x1500
    answered 1499…1023, 1022, then 767, 511, 255 … and finally WRONGTYPE
    garbage read out of freed memory;
  * LPOP, once the head side was empty, popped the tail's NEWEST element;
  * the LRANGE / LINDEX / SORT traversals bounded the active head by SEG_SIZE
    instead of its live end and started the tail at `head_segs_start`.
Nothing noticed because the list tests drain with counts, not with values.

Here a Python deque is the model: random push/pop mixes from both ends at
sizes that cross the ziplist→quicklist switch several segments deep, checking
every popped value, and LLEN / LRANGE / LINDEX against the model as it goes.

Usage: python3 tests/test_quicklist_pops.py [./pion-server]
"""
import collections
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6477
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def safe(fn, *a):
    """The reply, or ("ERR", message) — a pop that errors is a wrong answer."""
    try:
        return fn(*a)
    except Exception as e:   # WRONGTYPE garbage, a dropped connection, …
        return ("ERR", str(e)[:60])


def dec(v):
    return v.decode() if isinstance(v, bytes) else v


def main():
    import redis
    d = tempfile.mkdtemp(prefix="pion_qlpops_")
    p = subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-auto-detect", "--no-auto-embed"],
                         cwd=d, stdout=open(os.path.join(d, "log"), "w"), stderr=subprocess.STDOUT)
    try:
        r = None
        for _ in range(200):
            try:
                r = redis.Redis(port=PORT)
                r.ping()
                break
            except Exception:
                time.sleep(0.1)
        assert r is not None
        print(f"[quicklist pops] {BINARY}")

        # 1. the reported shapes: fill one way, drain one way
        for fill, pop in [("rpush", "rpop"), ("rpush", "lpop"), ("lpush", "rpop"), ("lpush", "lpop")]:
            r.delete("q")
            vals = [str(i) for i in range(1500)]
            getattr(r, fill)("q", *vals)
            model = collections.deque(vals if fill == "rpush" else reversed(vals))
            got, want = [], []
            for _ in range(1500):
                got.append(dec(safe(getattr(r, pop), "q")))
                want.append(model.pop() if pop == "rpop" else model.popleft())
            check(f"{fill.upper()} 1500 then {pop.upper()} x1500 returns every element in order",
                  got == want, f"first mismatch at {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)}")
            check(f"{fill.upper()}/{pop.upper()}: the drained key is gone", safe(r.exists, "q") == 0)
            if p.poll() is not None:
                check("server alive", False, "the server died")
                break

        # 2. random mixes against the model
        for seed in range(6):
            rnd = random.Random(seed)
            r.delete("m")
            # Start past the ziplist limit; then 2,000 ops draining the left,
            # 2,000 draining the right, 2,000 mixed — each drain outruns the
            # side it starts on, so pops must cross into the other side.
            init = [f"i{k}" for k in range(1100)]
            r.rpush("m", *init)
            model = collections.deque(init)
            counter = 0
            ok = True
            detail = ""
            for step in range(6000):
                # Phases of 1,000 ops lean on one end, so one side of the
                # structure runs dry and the pops must cross into the other.
                left_bias = [0.97, 0.97, 0.03, 0.03, 0.5, 0.5][step // 1000]
                op = rnd.random()
                if (op < 0.16 and len(model) < 4000) or len(model) < 50:
                    n = rnd.randint(1, 10)
                    vals = [f"v{counter + k}" for k in range(n)]
                    counter += n
                    if rnd.random() < left_bias:    # push on the side NOT being drained
                        r.rpush("m", *vals); model.extend(vals)
                    else:
                        r.lpush("m", *vals); model.extendleft(vals)
                else:
                    if rnd.random() >= left_bias:
                        a = dec(safe(r.rpop, "m")); b = model.pop() if model else None
                    else:
                        a = dec(safe(r.lpop, "m")); b = model.popleft() if model else None
                    if a != b:
                        ok, detail = False, f"step {step}: popped {a!r}, model {b!r}"
                        break
                if step % 250 == 0 and model:
                    lr = safe(r.lrange, "m", 0, -1)
                    if not isinstance(lr, list) or [dec(x) for x in lr] != list(model):
                        ok, detail = False, f"step {step}: LRANGE disagrees with the model ({len(model)} elements)"
                        break
                    idx = rnd.randrange(len(model))
                    li = dec(safe(r.lindex, "m", idx))
                    if li != model[idx]:
                        ok, detail = False, f"step {step}: LINDEX {idx} = {li!r}, model {model[idx]!r}"
                        break
            check(f"random push/pop mix, seed {seed} (6000 ops, {len(model)} left)", ok, detail)
            if p.poll() is not None:
                break
        check("server alive", p.poll() is None and safe(r.ping) is True)
    finally:
        p.kill()
        p.wait()
        shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'FAILED' if FAIL else 'PASSED'}: {len(FAIL)} failure(s)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
