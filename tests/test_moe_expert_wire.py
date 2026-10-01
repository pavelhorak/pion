#!/usr/bin/env python3
"""MOE.EXPERT.* Stage-1 wire-surface regression test.

Stage 1 ships read-only RESP commands registered against the Pion
slow-path. Backend tier lands in Stage 2; Stage 1 verifies the wire
contract:

  MOE.EXPERT.FETCH  → -UNAVAILABLE (backend not loaded)
  MOE.EXPERT.INFO   → -UNAVAILABLE
  MOE.EXPERT.STATS  → bulk-string JSON with stage flag
  MOE.EXPERT.PIN    → +OK (no-op without backend)
  MOE.EXPERT.UNPIN  → +OK
  MOE.EXPERT.PREFETCH → +OK

Args required for each command are validated:
  - FETCH / PIN / UNPIN need (model_id, layer_id, expert_id) = 3 args
  - PREFETCH needs at minimum (model_id, layer_id, expert_id) = 3 args
  - INFO needs (model_id) = 1 arg
  - STATS needs no args

Run:
  python3 tests/test_moe_expert_wire.py

Spawns a fresh `pion-server-dev` on an unused port, runs the wire
checks via redis-py, asserts replies, tears down.
"""
import os, socket, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The binary under test: PION_BIN (set by tests/run_all.py), else the -O3
# release build. Never the dev build: correctness must be shown at -O3.
PION = Path(os.environ.get('PION_BIN') or REPO / 'pion-server')
if not PION.exists():
    raise SystemExit(f'No server binary at {PION}. Run `pixi run build` first.')


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close()
    return p


def spawn(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(PION), '-p', str(port), '-w', '1', '--no-auto-embed'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # wait for the port to accept
    for _ in range(60):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                time.sleep(0.2)
                return proc
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f'pion-server-dev did not come up on port {port}')


def run(port: int, *args: str) -> str:
    """redis-cli -p PORT cmd args... → trimmed stdout.

    redis-cli exits 1 on RESP error replies (-ERR / -UNAVAILABLE); we still
    want the body text for assertion. Accept any exit code; return whatever
    came out.
    """
    res = subprocess.run(
        ['redis-cli', '-p', str(port), *args],
        capture_output=True, text=True, timeout=30,
    )
    return (res.stdout + res.stderr).strip()


def main() -> int:
    port = free_port()
    print(f'spawning pion-server-dev on port {port}...')
    proc = spawn(port)
    failures = []
    try:
        # 1. PING sanity
        r = run(port, 'PING')
        if r != 'PONG':
            failures.append(f'PING: expected PONG, got {r!r}')

        # 2. MOE.EXPERT.STATS — bulk-string JSON with tier counters
        r = run(port, 'MOE.EXPERT.STATS')
        # Stage 2 fields (real tier counters; stub returns stage=1 backend=unloaded).
        # Either Stage-1 stub or Stage-2 tier-backed reply is acceptable here.
        required_keys = ('stage', 'enabled', 'hits', 'misses', 'models_loaded')
        legacy_keys = ('stage', 'backend')
        if not all(k in r for k in legacy_keys) and not all(k in r for k in required_keys):
            failures.append(f'STATS: missing required fields: {r!r}')
        else:
            print(f'  STATS reply: {r}')

        # 3. MOE.EXPERT.INFO model_id
        r = run(port, 'MOE.EXPERT.INFO', 'test_model')
        if 'UNAVAILABLE' not in r:
            failures.append(f'INFO: expected UNAVAILABLE, got {r!r}')

        # 4. MOE.EXPERT.FETCH
        r = run(port, 'MOE.EXPERT.FETCH', 'test_model', '0', '0')
        if 'UNAVAILABLE' not in r:
            failures.append(f'FETCH: expected UNAVAILABLE, got {r!r}')

        # 5. MOE.EXPERT.PIN
        r = run(port, 'MOE.EXPERT.PIN', 'test_model', '0', '0')
        if r != 'OK':
            failures.append(f'PIN: expected OK, got {r!r}')

        # 6. MOE.EXPERT.UNPIN
        r = run(port, 'MOE.EXPERT.UNPIN', 'test_model', '0', '0')
        if r != 'OK':
            failures.append(f'UNPIN: expected OK, got {r!r}')

        # 7. MOE.EXPERT.PREFETCH — multiple expert IDs
        r = run(port, 'MOE.EXPERT.PREFETCH', 'test_model', '0', '0', '1', '2', '3')
        if r != 'OK':
            failures.append(f'PREFETCH: expected OK, got {r!r}')

        # 8. Underflowed FETCH → ERR
        r = run(port, 'MOE.EXPERT.FETCH', 'test_model')   # missing layer + expert
        if 'ERR' not in r and 'requires' not in r:
            failures.append(f'FETCH(underflow): expected ERR, got {r!r}')

        # 9. PING still works after our commands
        r = run(port, 'PING')
        if r != 'PONG':
            failures.append(f'PING(after): expected PONG, got {r!r}')

        # 9b. MOE.EXPERT.HIST without --moe-cache reports empty (no model loaded)
        r = run(port, 'MOE.EXPERT.HIST', 'test_model')
        if 'UNAVAILABLE' not in r and 'ERR' not in r:
            failures.append(f'HIST(no model): expected UNAVAILABLE or ERR, got {r!r}')

        # 9c. MOE.EXPERT.PRUNE without --moe-cache → UNAVAILABLE
        r = run(port, 'MOE.EXPERT.PRUNE', 'test_model', '0', '0')
        if 'UNAVAILABLE' not in r and 'ERR' not in r:
            failures.append(f'PRUNE(no model): expected UNAVAILABLE or ERR, got {r!r}')

        # 9d. STATS exposes pruned_count field
        r = run(port, 'MOE.EXPERT.STATS')
        if 'pruned_count' not in r:
            failures.append(f'STATS: missing pruned_count field: {r!r}')

        # 10. (optional, Stage 2j) End-to-end FETCH against the Gemma-4-26B bf16
        # snapshot if it's cached locally. Skipped if not present.
        import os, pathlib
        snap_root = pathlib.Path.home() / '.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-bf16/snapshots'
        snaps = list(snap_root.iterdir()) if snap_root.exists() else []
        if snaps and any((s / 'config.json').exists() for s in snaps):
            snap = next(s for s in snaps if (s / 'config.json').exists())
            print(f'  Stage-2j FETCH check: {snap.name}')
            proc.terminate(); proc.wait(5)
            proc = subprocess.Popen(
                [str(PION), '-p', str(port), '-w', '1', '--no-auto-embed',
                 '--moe-cache', str(snap), '--moe-cache-mib', '1024'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for _ in range(60):
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                        time.sleep(0.5); break
                except: time.sleep(0.5)
            r = run(port, 'MOE.EXPERT.STATS')
            if 'stage":2' not in r:
                failures.append(f'STATS post --moe-cache: expected stage:2, got {r!r}')
            r = run(port, 'MOE.EXPERT.INFO', snap.name)
            if '"experts":128' not in r or '"layers":30' not in r:
                failures.append(f'INFO mismatch: {r!r}')
            # gh #61 Item 1: INFO exposes hidden_size + moe_intermediate
            if '"hidden_size":2816' not in r or '"moe_intermediate":704' not in r:
                failures.append(f'INFO shape fields missing: {r!r}')

            # Stage-3 PRUNE round-trip
            r = run(port, 'MOE.EXPERT.PRUNE', snap.name, '0', '0')
            if r != '1':
                failures.append(f'PRUNE(0,0): expected integer 1, got {r!r}')
            r = run(port, 'MOE.EXPERT.FETCH', snap.name, '0', '0')
            if 'PRUNED' not in r:
                failures.append(f'FETCH after PRUNE: expected -PRUNED, got {r[:60]!r}')
            r = run(port, 'MOE.EXPERT.PRUNE', snap.name, '0', '0', '0')  # un-prune
            if r != '0':
                failures.append(f'PRUNE(0,0,0) un-prune: expected integer 0, got {r!r}')
            r = run(port, 'MOE.EXPERT.STATS')
            if '"pruned_count":0' not in r:
                failures.append(f'STATS pruned_count after un-prune: {r!r}')

            # MOE.EXPERT.HIST SAVE/LOAD round-trip — generate access counts via
            # a few FETCHes (raw socket: redis-cli's UTF-8 decode chokes on the
            # binary bf16 payload), SAVE to a tmp file, verify the file is
            # non-empty with magic bytes, then LOAD and verify counts restored.
            def _raw_fetch(L, E):
                s = socket.create_connection(('127.0.0.1', port), timeout=30)
                cmd = (f'*4\r\n$16\r\nMOE.EXPERT.FETCH\r\n'
                       f'${len(snap.name)}\r\n{snap.name}\r\n'
                       f'${len(str(L))}\r\n{L}\r\n'
                       f'${len(str(E))}\r\n{E}\r\n').encode()
                s.sendall(cmd)
                s.settimeout(30)
                buf = b''
                # Read header line ($N\r\n), then drain N bytes + 2 trailing
                while b'\r\n' not in buf: buf += s.recv(4096)
                hdr, _, rest = buf.partition(b'\r\n')
                if hdr[:1] == b'$':
                    n = int(hdr[1:])
                    while len(rest) < n + 2:
                        rest += s.recv(min(65536, n + 2 - len(rest)))
                s.close()
            for L in (0, 1):
                for E in (0, 1, 2):
                    _raw_fetch(L, E)
            hist_before = run(port, 'MOE.EXPERT.HIST', snap.name)
            if '"active_count":6' not in hist_before:
                failures.append(f'HIST baseline active_count: {hist_before[:80]!r}')
            tmp_hist = '/tmp/test_moe_hist_snapshot.bin'
            import os as _os
            try: _os.unlink(tmp_hist)
            except FileNotFoundError: pass
            r = run(port, 'MOE.EXPERT.HIST', snap.name, 'SAVE', tmp_hist)
            if r != 'OK':
                failures.append(f'HIST SAVE: expected OK, got {r!r}')
            if not _os.path.exists(tmp_hist) or _os.path.getsize(tmp_hist) < 9:
                failures.append(f'HIST snapshot file missing or too small')
            else:
                with open(tmp_hist, 'rb') as f:
                    hdr = f.read(5)
                    if hdr[:4] != b'MHST' or hdr[4] != 1:
                        failures.append(f'HIST snapshot magic/version: {hdr!r}')
            # Reload (this server already has the counts; LOAD should replace
            # them with the file contents — which are identical, so we just
            # verify LOAD reports OK and HIST returns the same shape).
            r = run(port, 'MOE.EXPERT.HIST', snap.name, 'LOAD', tmp_hist)
            if r != 'OK':
                failures.append(f'HIST LOAD: expected OK, got {r!r}')
            hist_after = run(port, 'MOE.EXPERT.HIST', snap.name)
            if hist_before != hist_after:
                failures.append(f'HIST after LOAD differs from before SAVE')
            # Bad subcommand
            r = run(port, 'MOE.EXPERT.HIST', snap.name, 'BOGUS', '/tmp/x')
            if 'subcommand' not in r:
                failures.append(f'HIST bad subcommand: {r!r}')
            try: _os.unlink(tmp_hist)
            except FileNotFoundError: pass

            # MOE.EXPERT.LOAD multi-model registration. Substrate supports
            # MAX_MOE_MODELS=4. Use a symlink to register the same model
            # under a different basename (different model_id).
            alias_path = '/tmp/test_moe_alias_b'
            try: _os.unlink(alias_path)
            except FileNotFoundError: pass
            _os.symlink(str(snap), alias_path)
            r = run(port, 'MOE.EXPERT.LOAD', alias_path)
            if r != 'OK':
                failures.append(f'MOE.EXPERT.LOAD: expected OK, got {r!r}')
            r = run(port, 'MOE.EXPERT.STATS')
            if '"models_loaded":2' not in r:
                failures.append(f'STATS after LOAD: expected models_loaded:2, got {r!r}')
            r = run(port, 'MOE.EXPERT.INFO', 'test_moe_alias_b')
            if '"experts":128' not in r:
                failures.append(f'INFO on LOADed model: {r!r}')
            _os.unlink(alias_path)

            # Stage 4b: PREFETCH should populate the LRU cache via the
            # background-warming thread. Pre-PREFETCH cache_bytes_used = 0;
            # after PREFETCH + a small delay it should reflect N × per-expert
            # payload size.
            r0 = run(port, 'MOE.EXPERT.STATS')
            import re
            def _stat_int(blob, key):
                m = re.search(rf'"{key}":(\d+)', blob)
                return int(m.group(1)) if m else None
            base_bytes = _stat_int(r0, 'cache_bytes_used') or 0
            base_hits  = _stat_int(r0, 'hits') or 0
            _ = run(port, 'MOE.EXPERT.PREFETCH', snap.name, '4', '0', '1', '2', '3')
            time.sleep(0.6)   # give warming thread time to complete 4 × ~12MB reads
            r1 = run(port, 'MOE.EXPERT.STATS')
            new_bytes = _stat_int(r1, 'cache_bytes_used') or 0
            grown = new_bytes - base_bytes
            # Each Gemma 4 expert payload is 11,894,803 bytes (3 blobs × 3.96 MB).
            # PREFETCH'ed 4 experts → expect ~47,579,212 bytes added.
            if grown < 4 * 11_000_000:
                failures.append(
                    f'PREFETCH warming did not populate cache: '
                    f'cache_bytes_used grew by {grown} (expected ~47.5 MB)')

            # Regression: pipelined PRUNE / PREFETCH must not bleed across
            # commands. PRUNE has an optional 5th arg; PREFETCH is variadic.
            # Pre-fix, both read past cmd_end_tok and consumed the next
            # command's first token → '-ERR Protocol error or incomplete frame'.
            def encode(args):
                parts = [b'*' + str(len(args)).encode() + b'\r\n']
                for a in args:
                    ab = a.encode() if isinstance(a, str) else a
                    parts.append(b'$' + str(len(ab)).encode() + b'\r\n' + ab + b'\r\n')
                return b''.join(parts)
            sk2 = socket.create_connection(('127.0.0.1', port), timeout=10)
            sk2.settimeout(5)
            # 5× PRUNE pipelined in one write
            sk2.sendall(b''.join(encode(['MOE.EXPERT.PRUNE', snap.name, '0', str(e)]) for e in range(5)))
            buf2 = b''
            try:
                while buf2.count(b'\r\n') < 5:
                    chunk = sk2.recv(4096)
                    if not chunk: break
                    buf2 += chunk
            except socket.timeout: pass
            if b'-ERR' in buf2 or buf2.count(b'\r\n') < 5:
                failures.append(f'pipelined 5x PRUNE: got {buf2!r}')
            # Pipelined PREFETCH (variadic args)
            sk2.sendall(
                encode(['MOE.EXPERT.PREFETCH', snap.name, '0', '0', '1'])
                + encode(['MOE.EXPERT.PREFETCH', snap.name, '0', '2'])
            )
            buf3 = b''
            try:
                while buf3.count(b'\r\n') < 2:
                    chunk = sk2.recv(4096)
                    if not chunk: break
                    buf3 += chunk
            except socket.timeout: pass
            if b'-ERR' in buf3 or buf3.count(b'+OK') < 2:
                failures.append(f'pipelined 2x PREFETCH: got {buf3!r}')
            sk2.close()
            # Clean up the 5 prunes from the regression test
            for e in range(5):
                _ = run(port, 'MOE.EXPERT.PRUNE', snap.name, '0', str(e), '0')

            # FETCH returns ~3.96 MB blob; check first byte count via socket directly
            sk = socket.create_connection(('127.0.0.1', port), timeout=30)
            sk.sendall(f'*4\r\n$16\r\nMOE.EXPERT.FETCH\r\n${len(snap.name)}\r\n{snap.name}\r\n$1\r\n0\r\n$1\r\n0\r\n'.encode())
            buf = b''
            sk.settimeout(30)
            while not buf.endswith(b'\r\n') or len(buf) < 12:
                buf += sk.recv(4096)
                if len(buf) > 64: break
            # Stage 2 multi-blob: 1 (n_blobs) + 3 * (6 + 3964928) = 11894803 bytes
            # for Gemma 4 bf16 (3 projs × 1 component each = 3 blobs).
            if not buf.startswith(b'$11894803\r\n'):
                failures.append(f'FETCH header: expected $11894803, got {buf[:32]!r}')
            else:
                # Verify body structure: header "$11894803\r\n" is 11 bytes,
                # then n_blobs:u8 = 3. (buf[10] is '\n', not data.)
                n_blobs = buf[11] if len(buf) > 11 else 0
                if n_blobs != 3:
                    failures.append(f'FETCH n_blobs: expected 3, got {n_blobs}')
                else:
                    print(f'  FETCH multi-blob OK: $11894803, n_blobs=3')
            sk.close()
        else:
            print('  Stage-2j FETCH skipped (Gemma-4-26B-A4B bf16 not cached locally)')

    finally:
        proc.terminate()
        try: proc.wait(5)
        except subprocess.TimeoutExpired: proc.kill()

    if failures:
        print('\nFAIL:')
        for f in failures:
            print(f'  {f}')
        return 1
    print('\nPASS — all 9 wire-surface assertions hold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
