#!/usr/bin/env python3
"""gh #210: SlabHashMap.set() never reuses DELETED tombstones and its probe
loop has no wrap-around termination.

Pre-fix defects:
  1. All three insert paths (set / set_with_hash / set_str_reuse_with_hash)
     probe for an h2 match or EMPTY only — DELETED (0xFF) matches neither, so
     tombstones are never reclaimed. Once every slot has been occupied at
     least once (all EMPTY exhausted), inserting a new key spins forever in
     `while True` — the worker wedges at 100% CPU mid-command.
  2. The load-factor check counts `size` only, so a set/remove churn workload
     (size stays small, tombstones accumulate) never triggers the rehash that
     would have purged the tombstones.

Reachable from the wire: SADD/SREM on one key (per-key SlabHashMap(16) wedges
at the 17th distinct member churned through), HSET/HDEL, ZADD/ZREM (skip-list
member map), SPOP (pop_random tombstones), and the main keyspace itself —
SET+DEL of ~unique key names wedges a StripedHashMap shard after ~capacity
unique keys pass through it.

Every churn phase runs under a socket timeout: a wedged server never replies,
so a timeout = reproduced hang = hard FAIL (and the server is killed).
"""
import os, socket, subprocess, sys, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1981
WORKDIR = f"/tmp/pion_gh210_test_{PORT}"

def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str): a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)

def send(sock, *args, timeout=3.0):
    """Send one command, return its (line-shaped) reply (raises on timeout)."""
    # Exactly one parsed reply: a single recv() with a swallowed timeout
    # used to return "" and let the late reply answer the NEXT command.
    sock.sendall(encode(args))
    rd = reader(sock)
    rd.timeout = timeout
    return rd.read_raw().decode(errors="replace")

def send_batch(sock, cmds, timeout=15.0):
    """Pipeline commands whose replies are all single-line (+OK / :N / -ERR).
    Returns number of reply lines received; a wedge shows up as a short count."""
    payload = b"".join(encode(c) for c in cmds)
    sock.sendall(payload)
    sock.settimeout(timeout)
    got, need = 0, len(cmds)
    buf = b""
    try:
        while got < need:
            chunk = sock.recv(1 << 20)
            if not chunk: break
            buf += chunk
            got = buf.count(b"\r\n")
    except socket.timeout:
        pass
    sock.settimeout(None)
    return got

def connect():
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            s = socket.socket(); s.settimeout(1.0)
            s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            # Ready = answering. The server accepts before it has replayed its WAL.
            if reader(s, 30.0).cmd("PING") != "PONG":
                raise RuntimeError("server accepted but did not answer PING")
            return s
        except OSError:
            s.close(); time.sleep(0.2)
    raise RuntimeError("server did not come up")

def start():
    return subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1"],
                            cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

passed = failed = 0
def check(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS {name}")
    else:
        failed += 1; print(f"  FAIL {name} {detail}")

def main():
    global failed
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = start()
    try:
        s = connect()

        # ── 1. SADD/SREM churn: >16 distinct members through a cap-16 map ──
        # Pre-fix: the 17th distinct SADD spins forever (all slots DELETED).
        print("[1] SADD/SREM churn (per-key set map)")
        wedged = False
        for i in range(200):
            r = send(s, "SADD", "churn:set", f"member-{i}")
            if ":1" not in r: wedged = True; break
            r = send(s, "SREM", "churn:set", f"member-{i}")
            if ":1" not in r: wedged = True; break
        check(not wedged, "200 add/remove cycles complete", f"wedged at i={i}")
        if wedged: return  # server is spinning; nothing further can run
        r = send(s, "SADD", "churn:set", "survivor")
        check(":1" in r, "insert after churn works", f"got {r!r}")
        r = send(s, "SCARD", "churn:set")
        check(":1" in r, "SCARD correct after churn", f"got {r!r}")

        # ── 2. tombstone reuse must not shadow live keys ──
        print("[2] live members survive interleaved churn")
        for i in range(8):
            send(s, "SADD", "mix:set", f"keep-{i}")
        for i in range(100):
            send(s, "SADD", "mix:set", f"tmp-{i}")
            send(s, "SREM", "mix:set", f"tmp-{i}")
        ok = all(":1" in send(s, "SISMEMBER", "mix:set", f"keep-{i}") for i in range(8))
        check(ok, "all 8 live members still found")
        r = send(s, "SCARD", "mix:set")
        check(":8" in r, "SCARD is 8", f"got {r!r}")

        # ── 3. HSET/HDEL churn (hash field map) ──
        print("[3] HSET/HDEL churn")
        wedged = False
        for i in range(200):
            r = send(s, "HSET", "churn:hash", f"f-{i}", "v")
            if ":1" not in r: wedged = True; break
            r = send(s, "HDEL", "churn:hash", f"f-{i}")
            if ":1" not in r: wedged = True; break
        check(not wedged, "200 HSET/HDEL cycles complete", f"wedged at i={i}")
        if wedged: return
        send(s, "HSET", "churn:hash", "final", "val")
        r = send(s, "HGET", "churn:hash", "final")
        check("val" in r, "HGET after churn", f"got {r!r}")

        # ── 4. ZADD/ZREM churn (skip-list member map) ──
        print("[4] ZADD/ZREM churn")
        wedged = False
        for i in range(200):
            r = send(s, "ZADD", "churn:zset", "1", f"m-{i}")
            if ":1" not in r: wedged = True; break
            r = send(s, "ZREM", "churn:zset", f"m-{i}")
            if ":1" not in r: wedged = True; break
        check(not wedged, "200 ZADD/ZREM cycles complete", f"wedged at i={i}")
        if wedged: return
        send(s, "ZADD", "churn:zset", "5", "zfinal")
        r = send(s, "ZSCORE", "churn:zset", "zfinal")
        check("5" in r, "ZSCORE after churn", f"got {r!r}")

        # ── 5. SPOP churn (pop_random writes tombstones too) ──
        print("[5] SPOP churn")
        for i in range(20):
            send(s, "SADD", "churn:spop", f"p-{i}")
        for i in range(20):
            send(s, "SPOP", "churn:spop")
        wedged = False
        for i in range(40):
            r = send(s, "SADD", "churn:spop", f"q-{i}")
            if ":1" not in r: wedged = True; break
        check(not wedged, "40 inserts after 20 SPOPs", f"wedged at i={i}")
        if wedged: return
        r = send(s, "SCARD", "churn:spop")
        check(":40" in r, "SCARD is 40", f"got {r!r}")

        # ── 6. growth path unaffected: monotonic inserts still rehash-double ──
        print("[6] growth still works")
        got = send_batch(s, [["SADD", "grow:set", f"g-{i}"] for i in range(1000)])
        check(got == 1000, "1000 monotonic SADDs answered", f"got {got}")
        r = send(s, "SCARD", "grow:set")
        check(":1000" in r, "SCARD is 1000", f"got {r!r}")

        # ── 7. main keyspace churn: unique-name SET+DEL through the shards ──
        # 100K unique keys > total EMPTY slots across StripedHashMap shards;
        # pre-fix a shard exhausts and the worker wedges mid-batch.
        print("[7] main keyspace SET+DEL churn (100K unique keys)")
        wedged = False
        B = 2000
        for base in range(0, 100_000, B):
            cmds = []
            for i in range(base, base + B):
                cmds.append(["SET", f"uniq:{i}", "x"])
                cmds.append(["DEL", f"uniq:{i}"])
            got = send_batch(s, cmds, timeout=20.0)
            if got != len(cmds):
                wedged = True
                check(False, "keyspace churn", f"batch at {base}: {got}/{len(cmds)} replies")
                break
        if wedged: return
        check(True, "100K unique-key SET+DEL cycles complete")
        r = send(s, "SET", "alive", "yes")
        check("+OK" in r, "SET after keyspace churn", f"got {r!r}")
        r = send(s, "GET", "alive")
        check("yes" in r, "GET after keyspace churn", f"got {r!r}")

        # ── 8. keys written before churn are still readable after purges ──
        print("[8] pre-churn keys survive tombstone purge rehash")
        r = send(s, "SCARD", "grow:set")
        check(":1000" in r, "grow:set intact", f"got {r!r}")
        ok = all(":1" in send(s, "SISMEMBER", "mix:set", f"keep-{i}") for i in range(8))
        check(ok, "mix:set live members intact")

    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
