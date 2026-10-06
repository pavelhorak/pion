#!/usr/bin/env python3
"""MOE.EXPERT.LOAD dedup regression.

Calling `MOE.EXPERT.LOAD <path>` twice with the same model path must NOT
consume a second `MAX_MOE_MODELS` slot — the second call is idempotent.
Before the fix landed in 2026-05-22, the multi-model concurrent demo
would push `models_loaded` to 3 or 4 after a single redundant LOAD,
which is wasteful (slots are scarce — only 4 per worker) but not
semantically broken (find_model returns the first matching slot).

Skips if Gemma 4 or OLMoE snapshots are not present in the local HF
cache.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Prefer whichever binary is newer (release vs dev). A stale -dev binary
# under the release one led to a false negative on this exact regression
# during 2026-05-22 development.
# The binary under test: PION_BIN (set by tests/run_all.py), else the -O3
# release build. Picking the newest of release/dev tested whichever was built last.
PION = Path(os.environ.get("PION_BIN") or REPO / "pion-server")
if not PION.exists():
    raise SystemExit(f"No server binary at {PION}. Run `pixi run build` first.")

HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
GEMMA_DIR = HF_HUB / "models--mlx-community--gemma-4-26b-a4b-it-bf16/snapshots"
OLMOE_DIR = HF_HUB / "models--mlx-community--OLMoE-1B-7B-0125-Instruct-4bit/snapshots"


def _first_snapshot(p: Path):
    if not p.exists():
        return None
    subs = [s for s in p.iterdir() if s.is_dir()]
    return subs[0] if subs else None


GEMMA_SNAP = _first_snapshot(GEMMA_DIR)
OLMOE_SNAP = _first_snapshot(OLMOE_DIR)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def spawn(port: int, moe_snap: Path):
    log_path = f"/tmp/pion-dedup-test.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [str(PION), "-p", str(port), "-w", "1", "--no-auto-embed",
         "--moe-cache", str(moe_snap), "--moe-cache-mib", "256"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        wait_ready_pid(port, proc, 40)   # this process, not a lingering listener (#27)
    except RuntimeError:
        proc.kill()
        raise
    return proc


def resp_call(port: int, *args, timeout: float = 60.0):
    """One RESP request. Returns (reply_kind, body) where reply_kind is
    one of "+", "-", "$", ":". Raises only on socket errors. -ERR is
    returned as ('-', message_bytes), letting the caller decide whether
    to assert or swallow."""
    sk = socket.socket()
    sk.settimeout(timeout)
    sk.connect(("127.0.0.1", port))
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(b"$" + str(len(a)).encode() + b"\r\n" + a + b"\r\n")
    sk.sendall(b"".join(parts))
    buf = bytearray()
    while not buf.endswith(b"\r\n"):
        c = sk.recv(1)
        if not c:
            sk.close()
            raise RuntimeError("connection closed mid-reply")
        buf.extend(c)
    line = bytes(buf[:-2])
    kind, rest = chr(line[0]), line[1:]
    if kind in ("+", "-", ":"):
        sk.close()
        return kind, rest
    if kind == "$":
        n = int(rest)
        if n < 0:
            sk.close()
            return kind, b""
        body = bytearray()
        while len(body) < n:
            chunk = sk.recv(min(65536, n - len(body)))
            if not chunk:
                sk.close()
                raise RuntimeError("connection closed mid-payload")
            body.extend(chunk)
        sk.recv(2)
        sk.close()
        return kind, bytes(body)
    sk.close()
    return kind, line


def stats(port: int):
    kind, body = resp_call(port, "MOE.EXPERT.STATS")
    assert kind == "$", f"STATS reply kind {kind!r}"
    return json.loads(body.decode())


def load(port: int, path: str):
    """Send MOE.EXPERT.LOAD <path>; return (kind, body_string)."""
    kind, body = resp_call(port, "MOE.EXPERT.LOAD", path)
    return kind, body.decode("utf-8", errors="replace")


def step(label, ok):
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {label}")
    return ok


def main():
    if GEMMA_SNAP is None or OLMOE_SNAP is None:
        print(
            "SKIP: Gemma 4 26B-A4B-bf16 and/or OLMoE-1B-7B-Instruct-4bit not in HF cache."
        )
        print(f"  GEMMA_SNAP = {GEMMA_SNAP}")
        print(f"  OLMOE_SNAP = {OLMOE_SNAP}")
        return 0

    port = free_port()
    print(f"[setup] spawning pion-server on port {port}; --moe-cache {GEMMA_SNAP.name}")
    proc = spawn(port, GEMMA_SNAP)
    failures = 0
    try:
        # ── 1. baseline ───────────────────────────────────────────────
        s = stats(port)
        print(f"[1] baseline STATS: models_loaded={s['models_loaded']}")
        failures += not step("models_loaded == 1 at startup", s["models_loaded"] == 1)

        # ── 2. redundant LOAD of the --moe-cache target ───────────────
        kind, body = load(port, str(GEMMA_SNAP))
        s = stats(port)
        print(f"[2] redundant LOAD(gemma) → '{kind}{body}'; STATS models_loaded={s['models_loaded']}")
        failures += not step("redundant LOAD returns +OK", kind == "+" and body == "OK")
        failures += not step("redundant LOAD does NOT consume a slot", s["models_loaded"] == 1)

        # ── 3. LOAD of a different model (OLMoE) ──────────────────────
        kind, body = load(port, str(OLMOE_SNAP))
        s = stats(port)
        print(f"[3] LOAD(olmoe) → '{kind}{body}'; STATS models_loaded={s['models_loaded']}")
        failures += not step("OLMoE LOAD returns +OK", kind == "+" and body == "OK")
        failures += not step("OLMoE LOAD consumes a fresh slot", s["models_loaded"] == 2)

        # ── 4. redundant LOAD of OLMoE ────────────────────────────────
        kind, body = load(port, str(OLMOE_SNAP))
        s = stats(port)
        print(f"[4] redundant LOAD(olmoe) → '{kind}{body}'; STATS models_loaded={s['models_loaded']}")
        failures += not step("redundant OLMoE LOAD returns +OK", kind == "+" and body == "OK")
        failures += not step("redundant OLMoE LOAD does NOT consume a slot", s["models_loaded"] == 2)

        # ── 5. trailing-slash variant must dedup too ──────────────────
        kind, body = load(port, str(GEMMA_SNAP) + "/")
        s = stats(port)
        print(f"[5] LOAD(gemma + '/') → '{kind}{body}'; STATS models_loaded={s['models_loaded']}")
        failures += not step("trailing-slash variant returns +OK", kind == "+" and body == "OK")
        failures += not step("trailing-slash variant does NOT consume a slot", s["models_loaded"] == 2)
    finally:
        proc.kill()
        proc.wait(timeout=5)

    if failures:
        print(f"\n{failures} FAILED assertion(s)")
        return 1
    print("\nMOE.EXPERT.LOAD dedup: all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
