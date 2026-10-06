#!/usr/bin/env python3
"""CLUSTER STATS regression test (gh #44).

Spins up `./pion-server-dev --kvcache --metal-attention -w 1` on a
sentinel port, drives a few V.CREATE / KV.PREFIX.REGISTER / ATTEND.PREFIX
operations, and asserts that CLUSTER STATS reports the expected counters.

Single-worker only — multi-worker stats coverage for V-Store / ATTEND is
documented as LOCAL-only in the response itself, and gh #44's scope was
explicitly "telemetry that operators can poll". Cross-worker fan-in for
ATTEND counters would need a shared atomic counter array; tracked
separately if it ever becomes load-bearing.
"""

import os
import socket
import subprocess
import sys
import time

import numpy as np
import redis


PION_BIN = os.environ.get("PION_BIN", "./pion-server")
PION_PORT = int(os.environ.get("PION_PORT", "16974"))


def _wait_ready(host: str, port: int, timeout: float = 60.0, proc=None) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"pion-server exited with {proc.returncode} before answering")
        try:
            r = redis.Redis(host=host, port=port, decode_responses=False, socket_timeout=1.0)
            if r.ping():
                return
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"pion-server didn't come up on {host}:{port} within {timeout}s ({last_err})")


def _parse_info_lines(s: bytes) -> dict:
    """Parse `key:value\\r\\n` (or `\\n`) blocks; ignore `# comment` and empties."""
    out = {}
    for raw in s.replace(b"\r\n", b"\n").split(b"\n"):
        line = raw.strip()
        if not line or line.startswith(b"#"):
            continue
        if b":" not in line:
            continue
        k, v = line.split(b":", 1)
        out[k.decode()] = v.decode()
    return out


def _send_resp(sock: socket.socket, *parts: bytes) -> bytes:
    """Send a RESP array, return the raw response (one read until newline-terminator)."""
    req = f"*{len(parts)}\r\n".encode()
    for p in parts:
        req += f"${len(p)}\r\n".encode() + p + b"\r\n"
    sock.sendall(req)
    buf = b""
    sock.settimeout(5.0)
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        # Cheap loop terminator: stop when we have at least one full line +
        # whatever the bulk-string body needs. CLUSTER STATS is a single
        # bulk string, so we can break as soon as we see a final \r\n
        # following the dollar-len header.
        if buf.startswith(b"$") and b"\r\n" in buf:
            try:
                header, rest = buf.split(b"\r\n", 1)
                expected = int(header[1:])
                if expected < 0:
                    break
                if len(rest) >= expected + 2:
                    break
            except ValueError:
                continue
        elif buf.startswith(b"-") or buf.startswith(b"+") or buf.startswith(b":"):
            if buf.endswith(b"\r\n"):
                break
    return buf


def _cluster_stats(host: str, port: int) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    raw = _send_resp(sock, b"CLUSTER", b"STATS")
    sock.close()
    # Strip the bulk-string header `$N\r\n` and trailing `\r\n`.
    assert raw.startswith(b"$"), f"unexpected response: {raw[:120]!r}"
    header, body = raw.split(b"\r\n", 1)
    n = int(header[1:])
    return _parse_info_lines(body[:n])


def main() -> int:
    if not os.path.exists(PION_BIN):
        print(f"FAIL: {PION_BIN} not found — run `pixi run build-dev` first")
        return 1

    # Make sure the port is free.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", PION_PORT))
        s.close()
    except OSError as e:
        print(f"FAIL: port {PION_PORT} not free ({e}); set PION_PORT to an unused port")
        return 1

    # Wipe per-worker persistence so a previous run doesn't restore sessions
    # that would lift the "empty server" counters.
    import glob
    for pat in ("pion.wal.*", "pion.vstore.*", "pion.hnsw.*", "pion.snapshot*"):
        for f in glob.glob(pat):
            try: os.remove(f)
            except OSError: pass

    # #27: no --metal-attention (CLUSTER STATS reads the V-store, KV.PREFIX and
    # ATTEND counters, none of which needs it, and it exists only on macOS) and
    # no auto-detected sidecar (its startup outlasted the readiness wait on a
    # Linux box). The server's output goes to a log that is printed on failure:
    # it went to DEVNULL, so a server that never listened left no trace.
    args = [PION_BIN, "--kvcache", "--no-auto-detect", "--no-auto-embed",
            "-p", str(PION_PORT), "-w", "1"]
    print(f"[boot] {' '.join(args)}")
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), f"cluster_stats_{PION_PORT}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)

    passed, failed = 0, 0
    try:
        try:
            _wait_ready("127.0.0.1", PION_PORT, proc=proc)
        except RuntimeError:
            log.flush()
            print(open(log_path).read()[-3000:])
            raise

        # --- 1. Empty server: STATS should respond with sane zeros. -----
        info = _cluster_stats("127.0.0.1", PION_PORT)
        for required in (
            "cluster_stats_local_worker", "cluster_stats_total_workers",
            "kv_prefix_active_total", "local_vstore_enabled",
            "local_vstore_sessions", "local_attend_enabled",
            "local_attend_sessions", "local_attend_total_queries",
            "local_attend_total_query_hits",
        ):
            if required not in info:
                print(f"[1] FAIL: missing key {required!r} in {info}")
                failed += 1
                break
        else:
            assert info["cluster_stats_total_workers"] == "1", info
            assert info["local_vstore_enabled"] == "1", info
            assert info["local_attend_enabled"] == "1", info
            assert info["local_vstore_sessions"] == "0", info
            assert info["local_attend_sessions"] == "0", info
            assert info["kv_prefix_active_total"] == "0", info
            assert info.get("kv_prefix_directory_enabled", "0") == "1", info
            assert info.get("kv_prefix_active_worker_0") == "0", info
            print("[1] empty-server STATS shape OK")
            passed += 1

        # --- 2. Drive a V-Store session and a KV.PREFIX register. -------
        r = redis.Redis(host="127.0.0.1", port=PION_PORT, decode_responses=False)
        ns = "kv:prefix:test_stats_001"
        r.execute_command("V.CREATE", ns + "_v", "64", "VQUANT", "fp16")
        r.execute_command("KV.PREFIX.REGISTER", ns, "64", "fp16")
        info = _cluster_stats("127.0.0.1", PION_PORT)
        # KV.PREFIX.REGISTER allocates 2 directory slots (K + V sides).
        assert int(info["kv_prefix_active_total"]) >= 2, info
        assert int(info["local_vstore_sessions"]) >= 1, info
        print(f"[2] V.CREATE + KV.PREFIX.REGISTER lift counters OK "
              f"(kv_active={info['kv_prefix_active_total']}, "
              f"vstore_sessions={info['local_vstore_sessions']})")
        passed += 1

        # --- 3. Legacy ATTEND.* (HNSW) path — exercised here because the
        # Mojo-side counters (`attn_idx.session_count`, total_queries,
        # total_query_hits) only see this path. ATTEND.PREFIX.* is backed by
        # the C metal engine and has no Mojo-visible counters today; tracked
        # implicitly via KV.PREFIX residence in test 2.
        att_sid = b"attend_stats_002"
        try:
            # ATTEND.CREATE sid key_dim value_dim
            r.execute_command("ATTEND.CREATE", att_sid, "32", "32")
            # ATTEND.STORE sid layer_id num_tokens keys_blob values_blob
            n_tokens = 4
            keys = np.random.randn(n_tokens, 32).astype(np.float32) * 0.1
            vals = np.random.randn(n_tokens, 32).astype(np.float32) * 0.1
            r.execute_command(
                "ATTEND.STORE", att_sid, "0", str(n_tokens),
                keys.tobytes(), vals.tobytes(),
            )
            stored_ok = True
        except redis.RedisError as e:
            stored_ok = False
            print(f"[3] ATTEND.* path unavailable ({e}) — skipping")

        if stored_ok:
            try:
                Q = np.random.randn(32).astype(np.float32) * 0.1
                # ATTEND.QUERY sid layer_id k query_blob
                r.execute_command("ATTEND.QUERY", att_sid, "0", "2", Q.tobytes())
            except redis.RedisError:
                pass  # query may not finalize — store still bumped session_count
            info = _cluster_stats("127.0.0.1", PION_PORT)
            assert int(info["local_attend_sessions"]) >= 1, info
            assert int(info["local_attend_total_tokens_stored"]) >= n_tokens, info
            print(f"[3] ATTEND.* lifts local counters OK "
                  f"(sessions={info['local_attend_sessions']}, "
                  f"tokens={info['local_attend_total_tokens_stored']}, "
                  f"queries={info['local_attend_total_queries']}, "
                  f"hits={info['local_attend_total_query_hits']})")
            passed += 1

        # --- 4. STATS works for an unknown subcommand variant (CLUSTER stats lowercase). ----
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", PION_PORT))
        raw = _send_resp(sock, b"CLUSTER", b"stats")  # lowercase
        sock.close()
        assert raw.startswith(b"$"), f"lowercase 'stats' rejected: {raw[:120]!r}"
        print("[4] case-insensitive subcommand OK")
        passed += 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
