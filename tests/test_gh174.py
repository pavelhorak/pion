#!/usr/bin/env python3
"""gh #174 durability test — streams, TTLs, HLL.

The three gaps gh #170 left open, exercised in the same two crash modes that
test_gh170.py uses, because they fail differently:

  Mode A: SAVE -> SIGKILL -> restart   (snapshot serializer path)
  Mode B: no SAVE -> SIGKILL -> restart (WAL effect-record replay path)

Mode B is the one that mattered here: before this change a stream survived
neither mode, an HLL survived only mode A (PFADD was snapshot-only, so every
add since the last SAVE was lost), and a TTL survived neither — keys came back
permanent, which is worse than losing them because it is silent.
"""
import os, socket, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader, wait_ready_pid  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1979
WORKDIR = f"/tmp/pion_gh174_test_{PORT}"

failures = []
passes = []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)


def send(sock, *args):
    # Exactly one parsed reply: a single recv() with a swallowed timeout
    # used to return "" and let the late reply answer the NEXT command.
    sock.sendall(encode(args))
    return reader(sock).read_raw().decode(errors="replace")


def connect(proc):
    # Ready = THIS process answering, not a killed server's lingering listener (#27).
    wait_ready_pid(PORT, proc, 60)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(1.0); s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            # Ready = answering. The server accepts before it has replayed its WAL.
            if reader(s, 30.0).cmd("PING") != "PONG":
                raise RuntimeError("server accepted but did not answer PING")
            return s
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")


def start():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def kill(proc):
    proc.kill(); proc.wait(timeout=5)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        s = socket.socket()
        try:
            s.settimeout(0.3); s.connect(("127.0.0.1", PORT)); s.close(); time.sleep(0.2)
        except OSError:
            s.close(); return


def populate(sock):
    """Write one of each #174-covered shape. Returns facts to re-check later."""
    # --- streams ---
    send(sock, "DEL", "s:log")
    send(sock, "XADD", "s:log", "1-1", "f1", "v1", "f2", "v2")
    send(sock, "XADD", "s:log", "2-1", "f1", "v3")
    send(sock, "XADD", "s:log", "3-1", "f1", "v4")
    auto = send(sock, "XADD", "s:log", "*", "auto", "yes")
    # an entry we delete, to prove tombstones replay too
    send(sock, "XADD", "s:log", "9-9", "doomed", "1")
    send(sock, "XDEL", "s:log", "9-9")

    # --- HLL ---
    send(sock, "DEL", "h:card")
    for i in range(500):
        send(sock, "PFADD", "h:card", f"elem-{i}")
    card = send(sock, "PFCOUNT", "h:card")

    # --- HLL via merge ---
    send(sock, "DEL", "h:a", "h:b", "h:merged")
    for i in range(100):
        send(sock, "PFADD", "h:a", f"a-{i}")
    for i in range(100):
        send(sock, "PFADD", "h:b", f"b-{i}")
    send(sock, "PFMERGE", "h:merged", "h:a", "h:b")
    merged = send(sock, "PFCOUNT", "h:merged")

    # --- TTLs ---
    send(sock, "SET", "t:long", "x")
    send(sock, "EXPIRE", "t:long", "10000")       # far future, must survive
    send(sock, "SET", "t:persist", "y")
    send(sock, "EXPIRE", "t:persist", "10000")
    send(sock, "PERSIST", "t:persist")            # must come back with no TTL
    send(sock, "SET", "t:none", "z")              # never had a TTL

    return {"auto_id": auto.strip().split("\r\n")[-1], "card": card, "merged": merged}


def verify(sock, facts, mode):
    xlen = send(sock, "XLEN", "s:log")
    check(f"[{mode}] stream survives restart with the right length",
          ":4\r\n" in xlen, f"XLEN={xlen!r} (want 4 live entries)")

    rng = send(sock, "XRANGE", "s:log", "-", "+")
    check(f"[{mode}] stream entry IDs preserved",
          "1-1" in rng and "2-1" in rng and "3-1" in rng,
          f"XRANGE missing IDs: {rng[:200]!r}")
    check(f"[{mode}] stream field/value payloads preserved",
          "v1" in rng and "v2" in rng and "v4" in rng, f"{rng[:200]!r}")
    check(f"[{mode}] XDEL tombstone replayed (deleted entry stays deleted)",
          "doomed" not in rng, f"deleted entry resurrected: {rng[:200]!r}")
    check(f"[{mode}] auto-generated ID replayed verbatim, not regenerated",
          facts["auto_id"] in rng,
          f"auto id {facts['auto_id']!r} absent from {rng[:200]!r}")

    card = send(sock, "PFCOUNT", "h:card")
    check(f"[{mode}] HLL cardinality survives restart",
          card == facts["card"], f"before={facts['card']!r} after={card!r}")

    merged = send(sock, "PFCOUNT", "h:merged")
    check(f"[{mode}] PFMERGE result survives restart",
          merged == facts["merged"], f"before={facts['merged']!r} after={merged!r}")

    ttl = send(sock, "TTL", "t:long")
    ok = False
    try:
        ok = ttl.startswith(":") and 0 < int(ttl[1:].strip()) <= 10000
    except ValueError:
        pass
    check(f"[{mode}] TTL survives restart as a live deadline", ok,
          f"TTL t:long = {ttl!r} (want 0 < n <= 10000; -1 means it came back permanent)")

    ttl_p = send(sock, "TTL", "t:persist")
    check(f"[{mode}] PERSIST replayed (key back without a TTL)",
          ":-1\r\n" in ttl_p, f"TTL t:persist = {ttl_p!r}")

    ttl_n = send(sock, "TTL", "t:none")
    check(f"[{mode}] key that never had a TTL is unaffected",
          ":-1\r\n" in ttl_n, f"TTL t:none = {ttl_n!r}")

    info = send(sock, "INFO", "persistence")
    for flag in ("streams_persisted:1", "hll_wal_logged:1", "ttls_persisted:1"):
        check(f"[{mode}] INFO reports {flag}", flag in info,
              f"missing from INFO persistence")


def run_mode(mode):
    print(f"\n=== Mode {mode}: "
          + ("SAVE then SIGKILL (snapshot path)" if mode == "A"
             else "no SAVE, SIGKILL (WAL replay path)") + " ===")
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)

    proc = start()
    sock = connect(proc)
    facts = populate(sock)
    if mode == "A":
        send(sock, "SAVE")
    sock.close()
    kill(proc)

    proc = start()
    sock = connect(proc)
    try:
        verify(sock, facts, mode)
    finally:
        sock.close()
        kill(proc)
        shutil.rmtree(WORKDIR, ignore_errors=True)


def main():
    for mode in ("A", "B"):
        run_mode(mode)
    print()
    print(f"gh #174: {len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
