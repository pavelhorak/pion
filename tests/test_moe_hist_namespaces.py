#!/usr/bin/env python3
"""MOE.EXPERT.HIST per-distribution namespace regression test (gh #61).

Background: a single cumulative FETCH histogram conflates distinct traffic
classes. The gh #61 retraction arc (7b383d4) showed a bottom-25% prune set
chosen from narrow-English HIST raises PPL by +96.67 on a diverse corpus, because
"least-used overall" is wrong for any single class. Per-distribution HIST
namespaces let an operator accumulate the histogram separately per traffic class
(`FETCH ... NS <name>`) and read it back per class (`HIST ... NS <name>`), so the
prune set can be computed safely across the mixture.

This test is fully self-contained: it needs no model download. A config.json-only
synthetic store is enough because MOE.EXPERT.FETCH bumps the namespace counter
*before* it reads any shard bytes (the FETCH then errors with "offset table
empty", but the histogram already recorded the access). load_manifest registers
the slot from config.json alone and returns success before touching shards.

Covers:
  - Always-on wire contract: NSLIST / NS on an unloaded tier → -UNAVAILABLE;
    HIST NS with no <name> → -ERR.
  - Live per-namespace separation: default vs named namespaces don't cross-talk;
    legacy `HIST <model>` == the default (ns 0) band.
  - INFO ns_count reflects registered namespaces.
  - HIST SAVE/LOAD round-trips a single namespace's counts.
  - Namespace registry overflow (>8 per model) → graceful -ERR, no crash.

Run:
  python3 tests/test_moe_hist_namespaces.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The binary under test: PION_BIN (set by tests/run_all.py), else the -O3
# release build. Never the dev build: correctness must be shown at -O3.
PION = Path(os.environ.get('PION_BIN') or REPO / 'pion-server')
if not PION.exists():
    raise SystemExit(f'No server binary at {PION}. Run `pixi run build` first.')

MAX_MOE_NS = 8   # must match src/network/moe_expert_tier.mojo


def free_port() -> int:
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close()
    return p


def spawn(port: int, moe_cache: str | None = None) -> subprocess.Popen:
    cmd = [str(PION), '-p', str(port), '-w', '1', '--no-auto-embed']
    if moe_cache:
        cmd += ['--moe-cache', moe_cache, '--moe-cache-mib', '32']
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                time.sleep(0.3)
                return proc
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f'pion-server-dev did not come up on port {port}')


def run(port: int, *args: str) -> str:
    """redis-cli call; returns stdout+stderr trimmed (accepts RESP error exits)."""
    res = subprocess.run(
        ['redis-cli', '-p', str(port), *args],
        capture_output=True, text=True, timeout=30,
    )
    return (res.stdout + res.stderr).strip()


def make_synthetic_store(root: Path, layers: int = 4, experts: int = 8) -> str:
    """A config.json-only Gemma-4-shaped MoE store. model_id = dir basename."""
    d = root / 'synthmoe'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'config.json').write_text(json.dumps({
        'num_hidden_layers': layers,
        'num_experts': experts,
        'num_experts_per_tok': 1,
    }))
    return str(d)


def main() -> int:
    failures: list[str] = []

    # ── Tier 1: always-on wire contract (no model loaded) ───────────────────
    port = free_port()
    proc = spawn(port)
    try:
        if run(port, 'PING') != 'PONG':
            failures.append('PING did not return PONG')
        # HIST NSLIST / NS on a tier with no model → UNAVAILABLE
        r = run(port, 'MOE.EXPERT.HIST', 'nomodel', 'NSLIST')
        if 'UNAVAILABLE' not in r:
            failures.append(f'HIST NSLIST unloaded: expected UNAVAILABLE, got {r!r}')
        r = run(port, 'MOE.EXPERT.HIST', 'nomodel', 'NS', 'x')
        if 'UNAVAILABLE' not in r:
            failures.append(f'HIST NS unloaded: expected UNAVAILABLE, got {r!r}')
    finally:
        proc.kill(); proc.wait()

    # ── Tier 2: live namespace behavior (synthetic config-only store) ────────
    with tempfile.TemporaryDirectory() as tmp:
        store = make_synthetic_store(Path(tmp), layers=4, experts=8)
        mid = os.path.basename(store)
        port = free_port()
        proc = spawn(port, moe_cache=store)
        try:
            # Arg validation: HIST NS with no name → -ERR
            r = run(port, 'MOE.EXPERT.HIST', mid, 'NS')
            if not r.startswith('ERR') and 'ERR ' not in r:
                failures.append(f'HIST NS (no name): expected ERR, got {r!r}')

            # Fresh model: ns_count == 1 (just "default")
            info = json.loads(run(port, 'MOE.EXPERT.INFO', mid))
            if info.get('ns_count') != 1:
                failures.append(f'fresh ns_count: expected 1, got {info.get("ns_count")}')

            # Bump ns 0 (default, no NS) at (0,1) twice; ns "code" at (1,2),(1,3);
            # ns "zh" at (2,4). FETCH errors "offset table empty" but bumps first.
            run(port, 'MOE.EXPERT.FETCH', mid, '0', '1')
            run(port, 'MOE.EXPERT.FETCH', mid, '0', '1')
            run(port, 'MOE.EXPERT.FETCH', mid, '1', '2', 'NS', 'code')
            run(port, 'MOE.EXPERT.FETCH', mid, '1', '3', 'NS', 'code')
            run(port, 'MOE.EXPERT.FETCH', mid, '2', '4', 'NS', 'zh')

            hist_default = json.loads(run(port, 'MOE.EXPERT.HIST', mid))
            hist_ns0 = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'default'))
            hist_code = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'code'))
            hist_zh = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'zh'))
            hist_unknown = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'ghost'))

            # Legacy HIST == explicit ns "default"
            if hist_default != hist_ns0:
                failures.append(f'legacy HIST != NS default: {hist_default} vs {hist_ns0}')
            # default band: 2 fetches at (0,1), 1 active entry
            if hist_default['total_fetches'] != 2 or hist_default['active'] != [[0, 1, 2]]:
                failures.append(f'default band wrong: {hist_default}')
            # code band: 2 fetches, 2 active, no cross-talk from default
            if hist_code['total_fetches'] != 2 or sorted(hist_code['active']) != [[1, 2, 1], [1, 3, 1]]:
                failures.append(f'code band wrong: {hist_code}')
            # zh band: 1 fetch at (2,4)
            if hist_zh['total_fetches'] != 1 or hist_zh['active'] != [[2, 4, 1]]:
                failures.append(f'zh band wrong: {hist_zh}')
            # unknown namespace: empty, not error
            if hist_unknown['total_fetches'] != 0 or hist_unknown['active'] != []:
                failures.append(f'unknown ns should be empty: {hist_unknown}')

            # ns_count now 3 (default, code, zh)
            info = json.loads(run(port, 'MOE.EXPERT.INFO', mid))
            if info.get('ns_count') != 3:
                failures.append(f'ns_count after 3 namespaces: got {info.get("ns_count")}')

            # NSLIST enumerates all three with correct per-ns totals
            nslist = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NSLIST'))['namespaces']
            byname = {row[1]: (row[0], row[2], row[3]) for row in nslist}
            if set(byname) != {'default', 'code', 'zh'}:
                failures.append(f'NSLIST names: {list(byname)}')
            elif byname['default'][1] != 2 or byname['code'][1] != 2 or byname['zh'][1] != 1:
                failures.append(f'NSLIST totals wrong: {byname}')

            # SAVE the code band, corrupt it, LOAD it back → round-trip.
            snap = os.path.join(tmp, 'code.mhst')
            r = run(port, 'MOE.EXPERT.HIST', mid, 'SAVE', snap, 'NS', 'code')
            if r != 'OK':
                failures.append(f'HIST SAVE NS code: expected OK, got {r!r}')
            # Overwrite the code band with an extra fetch, then LOAD to restore.
            run(port, 'MOE.EXPERT.FETCH', mid, '0', '0', 'NS', 'code')
            after_extra = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'code'))
            if after_extra['total_fetches'] != 3:
                failures.append(f'code band after extra fetch: {after_extra}')
            r = run(port, 'MOE.EXPERT.HIST', mid, 'LOAD', snap, 'NS', 'code')
            if r != 'OK':
                failures.append(f'HIST LOAD NS code: expected OK, got {r!r}')
            restored = json.loads(run(port, 'MOE.EXPERT.HIST', mid, 'NS', 'code'))
            if restored != hist_code:
                failures.append(f'code band did not round-trip: {restored} vs {hist_code}')
            # LOAD must not touch the default band
            if json.loads(run(port, 'MOE.EXPERT.HIST', mid)) != hist_default:
                failures.append('HIST LOAD of code band leaked into default band')

            # Namespace overflow: registry holds MAX_MOE_NS (incl. default + code
            # + zh = 3). Create up to the cap, then the next must -ERR gracefully.
            existing = 3
            for i in range(MAX_MOE_NS - existing):
                r = run(port, 'MOE.EXPERT.FETCH', mid, '0', '0', 'NS', f'fill{i}')
                # each is a distinct new namespace; FETCH still errors on offset
                # table but must NOT be a namespaces-full error yet
                if 'namespaces full' in r:
                    failures.append(f'premature namespaces-full at fill{i}: {r!r}')
            # registry is now full (8); one more distinct namespace → -ERR
            r = run(port, 'MOE.EXPERT.FETCH', mid, '0', '0', 'NS', 'overflow')
            if 'namespaces full' not in r:
                failures.append(f'expected namespaces-full error, got {r!r}')
            info = json.loads(run(port, 'MOE.EXPERT.INFO', mid))
            if info.get('ns_count') != MAX_MOE_NS:
                failures.append(f'ns_count at cap: expected {MAX_MOE_NS}, got {info.get("ns_count")}')
            # server still alive after the overflow
            if run(port, 'PING') != 'PONG':
                failures.append('server not responsive after namespace overflow')
        finally:
            proc.kill(); proc.wait()

    if failures:
        print('FAIL:')
        for f in failures:
            print('  -', f)
        return 1
    print('PASS: MOE.EXPERT.HIST per-distribution namespaces')
    return 0


if __name__ == '__main__':
    sys.exit(main())
