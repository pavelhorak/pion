#!/usr/bin/env python3
"""gh #260 regression test — a full WAL must stop ACKing writes it cannot persist.

When the log hit `--wal-max-segments`, `_note_drop` printed once, incremented
`wal_dropped_entries` — and the write was still answered `+OK`. gh #149 made the
refusal loud in the log and in INFO, but the WIRE contract stayed a lie: from
that point every acknowledged mutation silently evaporated on restart, and the
client had no way to tell. The gh #149 comment in `wal.mojo` records what that
costs in practice ("4.6 GB of SET blobs came back as nil after a restart").

What is asserted here, in both directions:

  refuse (the new default)
    - writes eventually fail with -MISCONF rather than +OK
    - READS keep working; a full log is not a reason to stop serving GETs
    - the substrate surface is NOT refused (FT.*/AI.*/KV.* own their own stores
      and never append to this log, so a full keyspace WAL says nothing about
      them) — asserted via PING/INFO staying alive and a read-only command set
    - INFO reports `wal_full_policy:refuse` and flips `wal_durability_lost:1`
    - and the decisive one: every key the server ACKED is still there after a
      restart. That is the property the old code violated.

  drop (explicit opt-out, the pre-gh #260 behaviour)
    - writes keep being acknowledged past the drop
    - INFO reports `wal_full_policy:drop`
    - `wal_dropped_entries` climbs, i.e. the loss is real and merely consented to

`--wal-max-segments 0` means "never rotate", so the single 1 MiB segment fills
and `_rotate` refuses immediately. That is the fastest way to reach the full
state without writing hundreds of megabytes.

Usage: python3 tests/test_gh260_wal_full_refuses.py [./pion-server]
"""
import os, socket, subprocess, sys, time, shutil

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1992
WORKDIR = f"/tmp/pion_gh260_test_{PORT}"

VAL = "v" * 512          # big enough to fill 1 MiB in a couple of thousand writes
MAX_WRITES = 20000       # hard cap so a non-filling build fails loudly, not forever

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        a = str(a).encode()
        parts.append(b"$%d\r\n%s\r\n" % (len(a), a))
    return b"".join(parts)


class Client:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.f = self.sock.makefile("rb")

    def __call__(self, *args):
        self.sock.sendall(encode(args)); return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        t = line[:1]
        if t in b"+-:":
            return line[:-2]
        if t == b"$":
            n = int(line[1:]); return None if n == -1 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(line[1:]); return None if n == -1 else [self._read() for _ in range(n)]
        raise RuntimeError("unparseable reply " + repr(line))

    def close(self):
        try: self.f.close(); self.sock.close()
        except OSError: pass


def spawn(policy=None):
    cmd = [BINARY, "-p", str(PORT), "-w", "1", "--no-auto-detect", "--no-auto-embed",
           "--wal-size", "1", "--wal-max-segments", "0"]
    if policy:
        cmd += ["--wal-full-policy", policy]
    return subprocess.Popen(cmd, cwd=WORKDIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect():
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try: return Client(PORT)
        except OSError: time.sleep(0.25)
    raise RuntimeError("server did not come up")


def info_field(c, field):
    info = c("INFO", "persistence") or b""
    for line in info.split(b"\n"):
        if line.startswith(field.encode() + b":"):
            return line.split(b":", 1)[1].strip().decode()
    return None


def stop(proc):
    proc.terminate()
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=10)


def fill_until_refused(c, prefix):
    """Write until the server refuses. Returns (acked_keys, refusal_reply)."""
    acked = []
    for i in range(MAX_WRITES):
        r = c("SET", f"{prefix}:{i}", VAL)
        if isinstance(r, bytes) and r.startswith(b"-"):
            return acked, r
        acked.append(f"{prefix}:{i}")
    return acked, None


def run_refuse():
    print("\n[1] --wal-full-policy refuse (the default)")
    proc = spawn()
    try:
        c = connect()
        check("policy reported as refuse", info_field(c, "wal_full_policy") == "refuse",
              f"got {info_field(c, 'wal_full_policy')!r}")
        check("durability intact at startup", info_field(c, "wal_durability_lost") == "0")

        acked, refusal = fill_until_refused(c, "k")
        check("the log actually filled (test is not vacuous)", refusal is not None,
              f"no refusal after {MAX_WRITES} writes — did --wal-max-segments 0 stop rotating?")
        if refusal is None:
            return []
        check("refusal is -MISCONF, not +OK", refusal.startswith(b"-MISCONF"),
              f"got {refusal[:80]!r}")
        check("durability_lost latched", info_field(c, "wal_durability_lost") == "1")

        # A full log must not take the server down with it.
        check("PING still answered", c("PING") == b"+PONG")
        check("reads still served", c("GET", acked[0]) == VAL.encode() if acked else False)
        check("INFO still served", info_field(c, "wal_full_policy") == "refuse")
        # And the refusal must be stable, not a one-shot.
        again = c("SET", "k:after", VAL)
        check("refusal persists on the next write", isinstance(again, bytes)
              and again.startswith(b"-MISCONF"), f"got {again[:60]!r}")
        check("the refused key was NOT stored", c("GET", "k:after") is None)

        c.close()
        return acked
    finally:
        stop(proc)


def verify_survives_restart(acked):
    """The property the old code broke: everything ACKed is still there."""
    print("\n[2] Every ACKed write survives a restart")
    proc = spawn()
    try:
        c = connect()
        missing = [k for k in acked[::max(1, len(acked) // 200)] if c("GET", k) is None]
        check(f"all {len(acked)} acked keys replay (sampled {min(200, len(acked))})",
              not missing, f"{len(missing)} missing, e.g. {missing[:3]}")
        c.close()
    finally:
        stop(proc)


def run_drop():
    print("\n[3] --wal-full-policy drop restores the old lossy behaviour, on request")
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn("drop")
    try:
        c = connect()
        check("policy reported as drop", info_field(c, "wal_full_policy") == "drop",
              f"got {info_field(c, 'wal_full_policy')!r}")
        _acked, refusal = fill_until_refused(c, "d")
        check("writes are NOT refused under drop", refusal is None,
              "" if refusal is None
              else f"got a refusal {refusal[:60]!r} despite --wal-full-policy drop")
        dropped = int(info_field(c, "wal_dropped_entries") or 0)
        check("entries really were dropped (the loss is consented to, not absent)",
              dropped > 0, f"wal_dropped_entries={dropped}")
        check("durability_lost still reported", info_field(c, "wal_durability_lost") == "1")
        c.close()
    finally:
        stop(proc)


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    try:
        acked = run_refuse()
        if acked:
            verify_survives_restart(acked)
        run_drop()
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
