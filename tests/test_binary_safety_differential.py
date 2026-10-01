#!/usr/bin/env python3
"""Binary-safe keys, members and values — the correct-type matrix, replayed
with invalid UTF-8 in every key and every member/value literal, against Redis 8.

WHY
Redis keys, values, list elements, hash fields, set and zset members are all
binary-safe. Pion's slow path reads arguments through RESP3Token.value(),
which (after #334) keeps valid UTF-8 exact but still spells every byte of an
INVALID sequence as '?'. Handlers that store must use raw_value(); the ones
that don't turn a serialized object, a token-id blob or a compressed value
into question marks — and reply success. #334's own test used "café", which
is valid UTF-8, so it could not see this half.

The existing matrix (test_redis_differential.SEMANTIC_SCRIPTS, 700+ steps)
uses ASCII literals only. This replays it with a deterministic rewrite applied
identically to both servers:

  * every key                           -> prefixed "bs:\\xff\\xfe" + key
  * every member / value / field literal -> literal + "\\xff\\x80"
  * untouched: KEYWORDS (upper-case), numbers / infinities / score ranges,
    glob patterns, stream ids, lex range sentinels "-" "+".

The rewrite does not need to preserve meaning — both servers see the same
bytes — it only must not change an argument's CLASS (a number stays a number).

    python3 tests/test_binary_safety_differential.py --pion-port 1974 --start-redis
Exit 0 = no divergence.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_redis_differential as trd  # noqa: E402

BIN = b"\xff\x80"
NUMERIC = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?$|^[-+]?inf$", re.I)


def rewrite(arg, keys):
    if not isinstance(arg, str):
        return arg
    if arg in keys:
        return b"bs:\xff\xfe" + keys[arg].encode()
    if arg in ("-", "+", "*", "$", ">", "", "0-0") or re.match(r"^\d+-(\d+|\*)$", arg):
        return arg
    if arg.isupper() or (arg.replace("_", "").replace(".", "").isupper()):
        return arg                                       # KEYWORD
    if arg.lower() in ("m", "km", "mi", "ft"):
        return arg                                       # GEO unit keyword (lower-case in the scripts)
    if NUMERIC.match(arg):
        return arg
    if arg[:1] in "([" and (NUMERIC.match(arg[1:]) or arg[1:] in ("-inf", "+inf")):
        return arg                                       # score range bound
    if any(ch in arg for ch in "*?[]"):
        return arg                                       # glob pattern
    if arg[:1] in "([":
        return arg.encode() + BIN                        # lex bound on a member
    return arg.encode() + BIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--redis-port", type=int, default=6413)
    ap.add_argument("--start-redis", action="store_true")
    args = ap.parse_args()

    rproc, rdir = None, None
    if args.start_redis:
        rdir = tempfile.mkdtemp(prefix="bs-redis-")
        rproc = subprocess.Popen(["redis-server", "--port", str(args.redis_port), "--save", "",
                                  "--appendonly", "no", "--dir", rdir],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 15
        while True:
            try:
                pion, redis = trd.Conn(args.pion_port), trd.Conn(args.redis_port)
                if pion.cmd("PING") == redis.cmd("PING") == ("status", "PONG"):
                    break
            except OSError:
                pass
            if time.time() > deadline:
                print("FATAL: need both servers answering PING"); return 2
            time.sleep(0.2)
        for c in (pion, redis):
            if c.cmd("FLUSHALL") != ("status", "OK"):
                print("FATAL: FLUSHALL refused"); return 2

        keymap = {"%K": "dfs:k", "%K2": "dfs:k2", "%K3": "dfs:k3"}
        n, diffs, desync = 0, [], []
        for name, script in trd.SEMANTIC_SCRIPTS:
            for cmd in script:
                c = [rewrite(keymap.get(p, p), {v: v for v in keymap.values()}) for p in cmd]
                n += 1
                try:
                    rp = trd.normalize(pion.cmd(*c), [x if isinstance(x, str) else "" for x in c])
                    rr = trd.normalize(redis.cmd(*c), [x if isinstance(x, str) else "" for x in c])
                    if pion.cmd("PING") != ("status", "PONG"):
                        desync.append((name, c)); pion = trd.Conn(args.pion_port)
                except (EOFError, socket.timeout) as e:
                    diffs.append((name, c, f"TRANSPORT {type(e).__name__}", "-"))
                    pion, redis = trd.Conn(args.pion_port), trd.Conn(args.redis_port)
                    break
                if c[0] == "DEL":
                    continue                               # setup reply depends on history
                if rp != rr:
                    diffs.append((name, c, rp, rr))

        print(f"binary-safety matrix: {n} steps, {len(diffs)} differ, {len(desync)} desynced")
        by_cmd = {}
        for name, c, rp, rr in diffs:
            by_cmd.setdefault(c[0], []).append((name, c, rp, rr))
        for cmdname, rows in sorted(by_cmd.items()):
            name, c, rp, rr = rows[0]
            print(f"  {cmdname:18} x{len(rows):<3} e.g. {[x if isinstance(x, str) else x for x in c][:5]}")
            print(f"      pion : {str(rp)[:150]}")
            print(f"      redis: {str(rr)[:150]}")
        for name, c in desync:
            print(f"  DESYNC {name}: {c[:4]}")
        return 1 if diffs or desync else 0
    finally:
        if rproc:
            rproc.kill(); rproc.wait()
        if rdir:
            shutil.rmtree(rdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
