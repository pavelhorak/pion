#!/usr/bin/env python3
"""gh #179 / #180 / #181: RESP compat bug trio (found wire-diffing vs Redis 8.10).

Pre-fix defects:
  #179  fast-path single-member ZADD accumulated integer digits and broke at
        '.', silently storing the truncated integer part of a fractional score.
  #180  HINCRBYFLOAT on a missing key answered WRONGTYPE instead of
        auto-creating the hash, and accumulated in Float32.
  #181  GenericValue.__str__ stringified the value's POINTER ("0x1349…")
        instead of its bytes — every GEO member lookup (GEOPOS/GEODIST/GEOHASH)
        compared against a hex address and answered nil. Multi-member GEOADD
        stored only the first (lon, lat, member) triple, and no GEOADD was
        WAL-logged on the slow path.

Covers: fractional/negative/garbage ZADD scores, pipelined bail-to-slow-path,
HINCRBYFLOAT auto-create + Float64 precision + trim + WRONGTYPE, GEO
round-trips (GEOPOS/GEODIST/GEOHASH), multi-member GEOADD, and GEO/zset
survival across a WAL-replay restart.
"""
import os, socket, subprocess, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import reader, wait_ready_pid, wait_port_free  # noqa: E402  (strict one-reply reads)

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1981
WORKDIR = f"/tmp/pion_gh179_test_{PORT}"

def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str): a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)

def send(sock, *args):
    # Exactly one parsed reply: a single recv() with a swallowed timeout
    # used to return "" and let the late reply answer the NEXT command.
    sock.sendall(encode(args))
    return reader(sock).read_raw().decode(errors="replace")

def send_pipeline(sock, *cmds):
    """Send several commands in one write; return exactly one parsed reply per
    command, concatenated, PLUS anything that arrives after them. The old
    version stopped at len(cmds) CRLFs, so a surplus reply (the desync this
    test exists for) was never read."""
    sock.sendall(b"".join(encode(c) for c in cmds))
    rd = reader(sock)
    out = b"".join(rd.read_raw() for _ in cmds)
    extra, rd.buf = rd.buf, b""
    sock.settimeout(0.2)
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            extra += chunk
    except socket.timeout:
        pass
    sock.settimeout(None)
    return (out + extra).decode(errors="replace")

def bulk_payload(reply):
    """Extract the first bulk-string payload from a RESP2 reply, or None."""
    lines = reply.split("\r\n")
    for idx, ln in enumerate(lines):
        if ln.startswith("$") and ln != "$-1" and idx + 1 < len(lines):
            return lines[idx + 1]
    return None

def _await_ready(s, deadline_s=30.0):
    """TCP accept is not readiness: the server listens BEFORE it initialises
    (so clients queue instead of being refused), and after a restart its first
    reply waits for the 10M-slot map and the WAL replay. send() gives up after
    2 s, so a slow first reply read as '' and every later check read the reply
    of the command before it. Block until PING answers, then assert."""
    s.settimeout(deadline_s)
    s.sendall(b"*1\r\n$4\r\nPING\r\n")
    buf = b""
    while b"+PONG\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            raise RuntimeError("server closed the connection before it was ready")
        buf += chunk
    s.settimeout(None)
    return s


def connect(proc):
    # Ready = THIS process answering, not a killed server's lingering listener (#27).
    wait_ready_pid(PORT, proc, 60)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            s = socket.socket(); s.settimeout(1.0)
            s.connect(("127.0.0.1", PORT)); s.settimeout(None)
            return _await_ready(s)
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
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = start()
    try:
        s = connect(proc)

        # ── gh #179: fractional ZADD scores ──
        r = send(s, "ZADD", "z", "1.5", "m")
        check(":1" in r, "179: fractional ZADD returns 1", f"got {r!r}")
        r = send(s, "ZSCORE", "z", "m")
        check(bulk_payload(r) == "1.5", "179: ZSCORE returns 1.5", f"got {r!r}")
        r = send(s, "ZADD", "zn", "-2.5", "n")
        check(":1" in r, "179: negative fractional ZADD returns 1", f"got {r!r}")
        r = send(s, "ZSCORE", "zn", "n")
        check(bulk_payload(r) == "-2.5", "179: ZSCORE returns -2.5", f"got {r!r}")
        r = send(s, "ZADD", "zi", "3", "k")
        check(":1" in r, "179: integer ZADD still works", f"got {r!r}")
        r = send(s, "ZSCORE", "zi", "k")
        check(bulk_payload(r) == "3", "179: integer ZSCORE unchanged", f"got {r!r}")
        r = send(s, "ZADD", "zm", "1.5", "a", "2.5", "b")
        check(":2" in r, "179: multi-member fractional ZADD", f"got {r!r}")
        r = send(s, "ZSCORE", "zm", "b")
        check(bulk_payload(r) == "2.5", "179: multi-member score correct", f"got {r!r}")
        r = send(s, "ZADD", "zg", "abc", "m")
        check(r.startswith("-ERR"), "179: garbage score errors", f"got {r!r}")
        # bail mid-pipeline: the fractional ZADD must not eat the trailing PING
        r = send_pipeline(s, ("ZADD", "zp", "4.25", "p"), ("PING",))
        check(":1" in r and "+PONG" in r, "179: pipelined bail keeps frame sync", f"got {r!r}")
        r = send(s, "ZSCORE", "zp", "p")
        check(bulk_payload(r) == "4.25", "179: pipelined fractional score stored", f"got {r!r}")
        # >23B member forces a heap GenericValue — the bail must not touch it
        big_member = "user:550e8400-e29b-41d4-a716-446655440000"
        r = send(s, "ZADD", "zb", "3.14", big_member)
        check(":1" in r, "179: fractional score with >23B member", f"got {r!r}")
        r = send(s, "ZSCORE", "zb", big_member)
        check(bulk_payload(r) == "3.14", "179: >23B member score correct", f"got {r!r}")

        # ── gh #180: HINCRBYFLOAT ──
        r = send(s, "HINCRBYFLOAT", "h", "f", "4.5")
        check(bulk_payload(r) == "4.5", "180: missing key auto-creates", f"got {r!r}")
        r = send(s, "HINCRBYFLOAT", "h", "f", "0.1")
        check(bulk_payload(r) == "4.6", "180: increment accumulates", f"got {r!r}")
        r = send(s, "HGET", "h", "f")
        check(bulk_payload(r) == "4.6", "180: HGET sees stored value", f"got {r!r}")
        r = send(s, "HINCRBYFLOAT", "h2", "f", "5.0")
        check(bulk_payload(r) == "5", "180: integer result trims .0", f"got {r!r}")
        r = send(s, "HINCRBYFLOAT", "h3", "g", "3006.999999")
        check(bulk_payload(r) == "3006.999999", "180: Float64 precision (Float32 gave 3007)", f"got {r!r}")
        send(s, "HSET", "h4", "f", "10.5")
        r = send(s, "HINCRBYFLOAT", "h4", "f", "0.1")
        check(bulk_payload(r) == "10.6", "180: string-field increment", f"got {r!r}")
        send(s, "HSET", "h5", "other", "1")
        r = send(s, "HINCRBYFLOAT", "h5", "f", "2.5")
        check(bulk_payload(r) == "2.5", "180: missing field on existing hash", f"got {r!r}")
        send(s, "SET", "str1", "v")
        r = send(s, "HINCRBYFLOAT", "str1", "f", "1")
        check(r.startswith("-WRONGTYPE"), "180: non-hash key is WRONGTYPE", f"got {r!r}")
        # short HINCRBYFLOAT must not satisfy its arity with the next command's tokens
        send(s, "SET", "px1", "pipeval")
        r = send_pipeline(s, ("HINCRBYFLOAT", "hshort", "f"), ("GET", "px1"))
        check("-ERR" in r and "pipeval" in r, "180: short cmd doesn't steal pipelined tokens", f"got {r!r}")
        r = send(s, "HGET", "hshort", "f")
        check("$-1" in r, "180: no phantom field from stolen tokens", f"got {r!r}")

        # ── gh #181: GEO member lookups ──
        r = send(s, "GEOADD", "geo", "13.361389", "38.115556", "Palermo")
        check(":1" in r, "181: GEOADD Palermo", f"got {r!r}")
        r = send(s, "GEOADD", "geo", "15.087269", "37.502669", "Catania")
        check(":1" in r, "181: GEOADD Catania", f"got {r!r}")
        r = send(s, "GEODIST", "geo", "Palermo", "Catania", "km")
        d = bulk_payload(r)
        check(d is not None and abs(float(d) - 166.27) < 2.0, "181: GEODIST km ~166.27", f"got {r!r}")
        r = send(s, "GEODIST", "geo", "Palermo", "Catania")
        d = bulk_payload(r)
        check(d is not None and abs(float(d) - 166274.0) < 2000.0, "181: GEODIST meters default", f"got {r!r}")
        r = send(s, "GEODIST", "geo", "Palermo", "nosuch")
        check("$-1" in r, "181: GEODIST missing member is nil", f"got {r!r}")
        r = send(s, "GEOPOS", "geo", "Palermo")
        lines = r.split("\r\n")
        floats = [float(x) for x in lines if x and x[0] in "-0123456789" and "." in x]
        check(len(floats) >= 2 and abs(floats[0] - 13.361389) < 1e-3 and abs(floats[1] - 38.115556) < 1e-3,
              "181: GEOPOS round-trips coords", f"got {r!r}")
        r = send(s, "GEOHASH", "geo", "Palermo")
        gh = bulk_payload(r)
        check(gh is not None and gh.startswith("sqc8b4"), "181: GEOHASH matches Redis prefix", f"got {r!r}")
        # multi-member GEOADD
        r = send(s, "GEOADD", "geo2", "13.361389", "38.115556", "pa", "15.087269", "37.502669", "pb")
        check(":2" in r, "181: multi-member GEOADD counts 2", f"got {r!r}")
        r = send(s, "GEOPOS", "geo2", "pa")
        check("$-1" not in r and "13.36" in r, "181: GEOPOS finds first member", f"got {r!r}")
        r = send(s, "GEOPOS", "geo2", "pb")
        check("$-1" not in r and "15.08" in r, "181: GEOPOS finds second member", f"got {r!r}")
        r = send(s, "GEODIST", "geo2", "pa", "pb", "km")
        d = bulk_payload(r)
        check(d is not None and abs(float(d) - 166.27) < 2.0, "181: GEODIST across multi-add", f"got {r!r}")
        r = send(s, "GEOADD", "geo2", "1", "1", "px", "2")   # not a multiple of 3
        check(r.startswith("-ERR"), "181: dangling GEOADD args error", f"got {r!r}")
        # FROMLONLAT used to be routed into the FROMMEMBER branch (byte-3 tie)
        r = send(s, "GEOSEARCH", "geo", "FROMLONLAT", "13.361389", "38.115556", "BYRADIUS", "200", "km")
        check("Palermo" in r and "Catania" in r, "181: GEOSEARCH FROMLONLAT finds both", f"got {r!r}")
        r = send(s, "GEOSEARCH", "geo", "FROMMEMBER", "Palermo", "BYRADIUS", "200", "km", "ASC")
        check("Palermo" in r and "Catania" in r, "181: GEOSEARCH FROMMEMBER works", f"got {r!r}")
        r = send(s, "GEORADIUS", "geo", "13.361389", "38.115556", "200", "km")
        check("Palermo" in r and "Catania" in r, "181: GEORADIUS finds both", f"got {r!r}")
        r = send(s, "GEOSEARCH", "geo", "FROMNOWHERE", "1", "2", "BYRADIUS", "200", "km")
        check(r.startswith("-ERR"), "181: unknown FROM keyword is a syntax error", f"got {r!r}")
        r = send(s, "PING")
        check("+PONG" in r, "181: server alive after bad GEOSEARCH", f"got {r!r}")

        # ── WAL replay: GEO + fractional zset survive a restart ──
        s.close()
        proc.terminate(); proc.wait(timeout=10)
        wait_port_free(PORT)
        proc = start()
        s = connect(proc)
        r = send(s, "ZSCORE", "z", "m")
        check(bulk_payload(r) == "1.5", "replay: fractional score survives", f"got {r!r}")
        r = send(s, "GEODIST", "geo", "Palermo", "Catania", "km")
        d = bulk_payload(r)
        check(d is not None and abs(float(d) - 166.27) < 2.0, "replay: GEODIST after restart", f"got {r!r}")
        r = send(s, "GEOPOS", "geo2", "pb")
        check("$-1" not in r and "15.08" in r, "replay: multi-added member survives", f"got {r!r}")
        r = send(s, "HGET", "h", "f")
        check(bulk_payload(r) == "4.6", "replay: HINCRBYFLOAT value survives", f"got {r!r}")
        s.close()
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

main()
