#!/usr/bin/env python3
"""pion-vllm-mlx serve against a live server: the value receipt and the
prefix budget. No model download — the K/V rows are synthetic mlx arrays in
KVCache-shaped objects, which is all PionPrefixStore reads.

  1. Receipt: a restore through V.FETCH shows up in PION.STATS as a hit with
     its token count (it used to count bytes only). LOOKUP's new options are
     validated before anything is recorded.
  2. Budget: past `budget_bytes`, the least-recently-used LEAF goes and the
     shared root stays; the V-store ends under budget; the evicted segment's
     block-index entries are gone; survivors restore bit-exact.
  3. Eviction is durable: after SIGKILL + restart the evicted prefix is not
     replayed back from the WAL.
  4. Compaction: once the WAL outgrows the budget, an idle-time
     KV.PREFIX.SAVE shrinks it, and a restart restores bit-exact from the
     snapshot.
  5. KV.PREFIX.SAVE pipelined with PING answers twice and writes the default
     snapshot (it used to take PING as its path and truncate the WAL).
  6. The V-store's own LRU eviction (session cap) is WAL-logged too.

Servers run in private temp dirs that are removed afterwards.

    python3 tests/test_serve_prefix_budget.py [--port 1995] [--binary ./pion-server]
"""
import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pion-vllm-mlx"))

import mlx.core as mx  # noqa: E402

from pion_vllm_mlx.prefix_store import PionPrefixStore, _Conn  # noqa: E402

LAYERS, HEADS, HDIM = 4, 2, 64
ROW_BYTES = 2 * LAYERS * HEADS * HDIM * 2          # K+V, every layer, fp16
KEY = ("synthetic-model", "budget-test")

fails = 0


def check(name, ok, detail: object = ""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    fails += 0 if ok else 1


class KVCache:
    """What PionPrefixStore reads from an mlx-lm KVCache: keys/values shaped
    (1, heads, tokens, head_dim) and an offset."""

    def __init__(self, keys, values):
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]


def rows_for(tokens):
    """Deterministic K/V per (position, token) — a prefix of a prompt always
    gets the same rows, as it would from a real model."""
    out = []
    t = np.asarray(tokens, dtype=np.int64)
    pos = np.arange(len(t))
    for li in range(LAYERS):
        base = (t[:, None] * 31 + pos[:, None] * 7 + li * 13 + np.arange(HEADS * HDIM)[None, :]) % 997
        k = (base / 997.0 - 0.5).astype(np.float16).reshape(len(t), HEADS, HDIM).transpose(1, 0, 2)[None]
        v = (base / 499.0 - 1.0).astype(np.float16).reshape(len(t), HEADS, HDIM).transpose(1, 0, 2)[None]
        out.append(KVCache(mx.array(k), mx.array(v)))
    return out


def start(binary, port, cwd, log):
    proc = subprocess.Popen([binary, "--kvcache", "-w", "1", "-p", str(port),
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=cwd, stdout=log, stderr=log, preexec_fn=os.setsid)
    for _ in range(150):
        try:
            c = _Conn("127.0.0.1", port, timeout=10)
            if c.cmd("PING") == "PONG":
                c.close()
                return proc
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def stats(conn):
    flat = conn.cmd("PION.STATS")
    return {flat[i].decode(): flat[i + 1] for i in range(0, len(flat), 2)}


def restore_equal(store, tokens, n):
    """Look `tokens` up in a fresh view and compare its first n rows with the
    rows that were stored for them."""
    m = store.lookup(KEY, tokens)
    if m.length < n:
        return False, f"lookup matched {m.length} < {n}"
    rows = store.fetch_rows(KEY, m, 0, n)
    if rows is None:
        return False, "fetch failed"
    ref = rows_for(tokens[:n])
    for li, (k, v) in enumerate(rows):
        if not (np.array_equal(k, np.array(ref[li].keys[0])) and np.array_equal(v, np.array(ref[li].values[0]))):
            return False, f"layer {li} differs"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1995)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
    args = ap.parse_args()
    args.binary = os.path.abspath(args.binary)
    logging.basicConfig(level=logging.WARNING)
    rng = np.random.default_rng(7)
    R = rng.integers(1000, 30000, 256).tolist()                   # shared root (system prompt)
    B = [rng.integers(1000, 30000, 256).tolist() for _ in range(6)]

    tmp = tempfile.mkdtemp(prefix="pion_budget_")
    log = open(os.path.join(tmp, "server.log"), "wb")
    proc = start(args.binary, args.port, tmp, log)
    try:
        # ── 1. receipt ────────────────────────────────────────────────
        s = PionPrefixStore(port=args.port)
        s.store(KEY, R + B[0], rows_for(R + B[0]))
        c = s.conn
        c.cmd("PION.STATS", "RESET")
        v = PionPrefixStore(port=args.port)          # a restarted serve process
        # 300 of the lineage's 512 rows: the server's own count for the
        # namespace (512) and the caller's (300) must not coincide.
        ok, why = restore_equal(v, R + B[0], 300)
        check("restore through V.FETCH is bit-exact", ok, why)
        st = stats(c)
        check("receipt counts the restore as one hit", st.get("kvprefix_hits") == 1, st.get("kvprefix_hits"))
        check("receipt credits the 300 restored tokens", st.get("kvprefix_tokens_served") == 300,
              st.get("kvprefix_tokens_served"))
        check("receipt shows prefill avoided (estimate, 300 x 555 us)",
              st.get("prefill_seconds_avoided") == b"0.166", st.get("prefill_seconds_avoided"))
        ns = s._ns(s.model_id(KEY), 1)
        for bad in (("TOKENS",), ("TOKENS", "-1"), ("BOGUS", "1")):
            try:
                c.cmd("KV.PREFIX.LOOKUP", ns, *bad)
                check(f"LOOKUP {' '.join(bad)} refused", False, "accepted")
            except Exception as e:
                check(f"LOOKUP {' '.join(bad)} refused", "ERR" in str(e), str(e))
        check("refused LOOKUPs recorded nothing", stats(c).get("kvprefix_hits") == 1)
        check("LOOKUP TOKENS on a MISS credits nothing",
              c.cmd("KV.PREFIX.LOOKUP", "no-such-ns", "TOKENS", "999") == "MISS"
              and stats(c).get("kvprefix_tokens_served") == 300)

        # ── 5. SAVE pipelined with PING ───────────────────────────────
        r = c.pipeline([("KV.PREFIX.SAVE",), ("PING",)])
        check("SAVE + PING pipelined -> OK, PONG", r == ["OK", "PONG"], repr(r))
        check("no file named PING was written", not os.path.exists(os.path.join(tmp, "PING")))
        check("default snapshot written, no .tmp left",
              os.path.exists(os.path.join(tmp, "pion.vstore.0"))
              and not os.path.exists(os.path.join(tmp, "pion.vstore.0.tmp")))

        # ── 2. budget: leaf-first eviction ────────────────────────────
        # Segment 1 is R + B0 (512 rows, stored in section 1). Each R + Bi
        # after it branches at row 256 into a 256-row child of segment 1.
        # The budget holds segment 1 plus two children (1024 rows), not three.
        budget = 1100 * ROW_BYTES
        s = PionPrefixStore(port=args.port, budget_bytes=budget)
        s.COMPACT_IDLE_S = 3600          # compaction is section 4's subject
        for i in range(1, 4):
            s.store(KEY, R + B[i], rows_for(R + B[i]))
            time.sleep(0.01)
        info = s.info()
        check("V-store under budget after eviction", info["vstore_bytes"] <= budget,
              f"held {info['vstore_bytes']}, budget {budget}")
        check("exactly one segment evicted", s.stats.evictions == 1, s.stats.evictions)
        fresh = PionPrefixStore(port=args.port)
        check("root survives (R + B0 matches all 512)", fresh.lookup(KEY, R + B[0]).length == 512,
              fresh.lookup(KEY, R + B[0]).length)
        check("oldest leaf B1 is gone (R + B1 matches only R)", fresh.lookup(KEY, R + B[1]).length == 256,
              fresh.lookup(KEY, R + B[1]).length)
        ok, why = restore_equal(fresh, R + B[3], 512)
        check("R + B3 restores bit-exact", ok, why)
        ok, why = restore_equal(fresh, R + B[2], 512)
        check("R + B2 restores bit-exact", ok, why)
        mid = s.model_id(KEY)
        blk = c.cmd("HGETALL", s._k(mid, "blk"))
        pointed = {blk[i + 1].split(b":")[0] for i in range(0, len(blk), 2)}
        segs_left = {k.decode().rsplit(":", 1)[1].encode()
                     for k in c.cmd("KEYS", s._k(mid, "seg", "*"))}
        check("block index points only at live segments", pointed <= segs_left,
              f"{sorted(pointed - segs_left)}")
        # B2 was used last, so B3 is now the oldest leaf: the next new
        # branch must evict B3, not B2.
        time.sleep(0.01)
        s.store(KEY, R + B[1], rows_for(R + B[1]))
        after = PionPrefixStore(port=args.port)
        check("recently used B2 kept, older B3 evicted",
              after.lookup(KEY, R + B[2]).length == 512 and after.lookup(KEY, R + B[3]).length == 256,
              f"B2 {after.lookup(KEY, R + B[2]).length}, B3 {after.lookup(KEY, R + B[3]).length}")

        # ── 3. eviction survives a restart ────────────────────────────
        live = {sid for sid in range(1, 8)
                if c.cmd("KV.PREFIX.LOOKUP", s._ns(mid, sid)) == "HIT"}
        c.close()
        kill(proc)
        proc = start(args.binary, args.port, tmp, log)
        c = _Conn("127.0.0.1", args.port)
        live2 = {sid for sid in range(1, 8)
                 if c.cmd("KV.PREFIX.LOOKUP", s._ns(mid, sid)) == "HIT"}
        check("same prefixes live after SIGKILL + restart (evictions not replayed)",
              live == live2, f"before {sorted(live)}, after {sorted(live2)}")

        # ── 4. compaction ─────────────────────────────────────────────
        s = PionPrefixStore(port=args.port, budget_bytes=budget)
        s.COMPACT_IDLE_S = 0.3
        for i in (4, 5):                    # new branches: each evicts, and the WAL keeps growing
            s.store(KEY, R + B[i], rows_for(R + B[i]))
        wal_before = s.info()["wal_bytes"]
        t0 = time.time()
        while s.stats.compactions == 0 and time.time() - t0 < 10:
            time.sleep(0.1)
        wal_after = s.info()["wal_bytes"]
        check("WAL outgrew the budget before compaction", wal_before > budget, f"{wal_before} vs {budget}")
        check("idle-time compaction ran", s.stats.compactions >= 1, s.stats.compactions)
        check("WAL shrank to under budget", wal_after <= budget, f"{wal_before} -> {wal_after}")
        c.close()
        kill(proc)
        proc = start(args.binary, args.port, tmp, log)
        fresh = PionPrefixStore(port=args.port)
        ok, why = restore_equal(fresh, R + B[5], 512)
        check("after compaction + restart, R + B5 restores bit-exact", ok, why)

        # ── 6. the V-store's own LRU eviction is WAL-logged ───────────
        c = _Conn("127.0.0.1", args.port)
        c.cmd("FLUSHALL")
        cap = fresh.info()["vstore_max_sessions"]
        n = cap // 2 + 4                      # each prefix is two sessions (_pk, _pv)
        for i in range(n):
            c.cmd("KV.PREFIX.REGISTER", f"lru{i}", 64, "fp16")
        first_gone = c.cmd("KV.PREFIX.LOOKUP", "lru0") == "MISS"
        c.close()
        kill(proc)
        proc = start(args.binary, args.port, tmp, log)
        c = _Conn("127.0.0.1", args.port)
        check("session-cap eviction happened", first_gone)
        check("session-cap eviction not replayed after restart",
              c.cmd("KV.PREFIX.LOOKUP", "lru0") == "MISS")
        check(f"newest prefix lru{n - 1} still there", c.cmd("KV.PREFIX.LOOKUP", f"lru{n - 1}") == "HIT")
        c.close()
    finally:
        kill(proc)
        log.close()
        if fails:
            print(f"server log kept: {tmp}/server.log")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
