#!/usr/bin/env python3
"""gh #149 + gh #163 — durability and residency for multi-MB values.

gh #149: acknowledged writes past the WAL's fixed 256 MB ring were silently
dropped. Pre-fix measurement on the release binary: of 50 x 6 MB SET blobs, 42
survived a SIGKILL and 8 came back nil — while a *small* value written after
them survived, because small entries still fit the tail slack a blob no longer
did. That is exactly the reported field signature (720 blobs gone, 20 small
metas kept).

gh #163: those blob bytes lived on the anonymous heap, so they were jetsam bait
on a 16 GB machine and were memcpy'd into the WAL on every write.

The suite drives a real server, SIGKILLs it (the jetsam shape — no clean
shutdown, no flush), restarts, and checks what came back. Each test uses its own
data dir so a failure leaves evidence behind.

Usage: python3 tests/test_gh149_gh163_blob_durability.py [--binary ./pion-server] [--port 1987]
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


# ── RESP plumbing ───────────────────────────────────────────────────────────

def cmd(*args):
    out = b"*%d\r\n" % len(args)
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        out += b"$%d\r\n%s\r\n" % (len(a), a)
    return out


class Client:
    def __init__(self, port, timeout=120):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        # A multi-MB bulk reply must be read through a buffered file object —
        # slicing a growing bytes object is O(N^2) and turns a 6 MB GET into a
        # visible stall (see the python_resp_array_slicing_trap note).
        self.f = self.s.makefile("rb")

    def call(self, *args):
        self.s.sendall(cmd(*args))
        return self.read()

    def read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("connection closed")
        t, rest = line[0:1], line[1:-2]
        if t in b"+-:":
            return rest
        if t == b"$":
            n = int(rest)
            if n == -1:
                return None
            data = self.f.read(n + 2)
            return data[:-2]
        if t == b"*":
            return [self.read() for _ in range(int(rest))]
        raise RuntimeError(f"unexpected reply: {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


# ── server lifecycle ────────────────────────────────────────────────────────

class Server:
    def __init__(self, binary, port, datadir, extra=(), profile="kv", workers=1):
        self.binary, self.port, self.datadir = binary, port, datadir
        self.extra = list(extra)
        self.profile, self.workers = profile, workers
        self.proc = None

    def start(self):
        cmd_ = [self.binary, "-p", str(self.port), "-w", str(self.workers),
                "--profile", self.profile,
                "--no-auto-detect", "--no-auto-embed"] + self.extra
        if self.workers > 1:
            cmd_.append("--independent-workers")   # gh #253
        self.log = os.path.join(self.datadir, f"server.{int(time.time()*1000)%100000}.log")
        fp = open(self.log, "w")
        self.proc = subprocess.Popen(cmd_, cwd=self.datadir, stdout=fp, stderr=fp,
                                     preexec_fn=os.setsid)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                c = Client(self.port, timeout=2)
                if c.call("PING") == b"PONG":
                    c.close()
                    return self
                c.close()
            except (OSError, EOFError):
                time.sleep(0.3)
        raise RuntimeError(f"server did not come up on {self.port}; see {self.log}")

    def sigkill(self):
        """The jetsam shape: uncatchable, nothing flushed on the way out."""
        if self.proc:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait()
            self.proc = None

    def log_text(self):
        try:
            with open(self.log) as fh:
                return fh.read()
        except OSError:
            return ""


def payload(i, mb):
    return bytes(((i * 31 + j * 17) & 0xFF) for j in range(256)) * (mb * 4096)


def free_dir():
    return tempfile.mkdtemp(prefix="pion_gh149_")


# ── tests ───────────────────────────────────────────────────────────────────

def test_blobs_past_the_old_ring(binary, port):
    """60 x 6 MB = 378 MB, well past the old fixed 256 MB WAL. Pre-fix this lost
    everything after the 42nd blob while keeping a small value written later."""
    d = free_dir()
    n, mb = 60, 6
    srv = Server(binary, port, d).start()
    c = Client(port)
    for i in range(n):
        c.call("SET", f"blob:{i}", payload(i, mb))
    c.call("SET", "meta:small", "kept")
    c.close()
    time.sleep(1.5)          # let one group-commit tick land
    srv.sigkill()

    srv = Server(binary, port, d).start()
    c = Client(port)
    survivors = sum(1 for i in range(n) if c.call("GET", f"blob:{i}") == payload(i, mb))
    small = c.call("GET", "meta:small")
    c.close()
    srv.sigkill()

    check("gh #149: all 60 multi-MB blobs survive SIGKILL (378 MB > old 256 MB ring)",
          survivors == n, f"{survivors}/{n} survived")
    check("gh #149: small values still survive alongside them", small == b"kept")

    # gh #163: the payloads must be in the arena, not the log. If they were still
    # going through the WAL, 378 MB would have forced a rotation and left sealed
    # segments behind.
    files = os.listdir(d)
    sealed = [f for f in files if f.startswith("pion.wal.0.")]
    blobs = [f for f in files if f.startswith("pion.blob.")]
    check("gh #163: payloads went to the blob arena, not the WAL",
          len(blobs) >= 1 and len(sealed) == 0,
          f"blob segments={len(blobs)} sealed WAL segments={len(sealed)}")
    shutil.rmtree(d, ignore_errors=True)


def test_wal_rotation_for_small_values(binary, port):
    """The general form of gh #149: any workload used to stop being durable after
    256 MB. Forced here with a tiny segment size so the test stays quick."""
    d = free_dir()
    # 4 MB segments, 8 KB values -> a few hundred values fills several segments.
    srv = Server(binary, port, d,
                 extra=["--wal-size", "4", "--wal-max-segments", "16"]).start()
    c = Client(port)
    val = b"v" * 8192
    n = 2000                                  # ~16 MB of entries => rotation
    for i in range(n):
        c.call("SET", f"small:{i}", val)
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    sealed = [f for f in os.listdir(d) if f.startswith("pion.wal.0.")]
    srv = Server(binary, port, d,
                 extra=["--wal-size", "4", "--wal-max-segments", "16"]).start()
    c = Client(port)
    survivors = sum(1 for i in range(n) if c.call("GET", f"small:{i}") == val)
    c.close()
    srv.sigkill()

    check("gh #149: WAL rotated instead of silently dropping", len(sealed) > 0,
          f"{len(sealed)} sealed segment(s)")
    check("gh #149: every value replays across sealed segments",
          survivors == n, f"{survivors}/{n} survived")
    shutil.rmtree(d, ignore_errors=True)


def test_wal_full_is_loud(binary, port):
    """When the log genuinely cannot rotate any further the write is refused
    *loudly*. Silence here is the actual bug being fixed."""
    d = free_dir()
    srv = Server(binary, port, d,
                 extra=["--wal-size", "4", "--wal-max-segments", "0",
                        "--no-blob-tier"]).start()
    c = Client(port)
    val = b"v" * 8192
    for i in range(1200):                     # ~10 MB into a 4 MB single segment
        c.call("SET", f"k:{i}", val)
    c.close()
    time.sleep(0.5)
    log = srv.log_text()
    srv.sigkill()
    check("gh #149: a full WAL says so instead of dropping silently",
          "WAL: FULL" in log,
          "no 'WAL: FULL' line in server log" if "WAL: FULL" not in log else "")
    shutil.rmtree(d, ignore_errors=True)


def test_overwrite_and_delete_ordering(binary, port):
    """The blob arena holds bytes; the WAL stays the single ordered index. Both
    directions of large<->small overwrite, plus DEL, must replay correctly."""
    d = free_dir()
    big1, big2 = payload(1, 2), payload(2, 2)
    small = b"i am small"
    srv = Server(binary, port, d).start()
    c = Client(port)
    c.call("SET", "a", big1); c.call("SET", "a", small)      # large -> small
    c.call("SET", "b", small); c.call("SET", "b", big2)      # small -> large
    c.call("SET", "c", big1); c.call("DEL", "c")             # large -> gone
    c.call("SET", "d", big1); c.call("SET", "d", big2)       # large -> large
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    srv = Server(binary, port, d).start()
    c = Client(port)
    a, b, cc, dd = (c.call("GET", k) for k in ("a", "b", "c", "d"))
    c.close()
    srv.sigkill()

    check("gh #163: large value overwritten by a small one replays small", a == small)
    check("gh #163: small value overwritten by a large one replays large", b == big2)
    check("gh #163: deleted blob key stays deleted", cc is None)
    check("gh #163: blob overwritten by another blob replays the newer one", dd == big2)
    shutil.rmtree(d, ignore_errors=True)


def test_save_keeps_blob_pointers(binary, port):
    """SAVE checkpoints the WAL. The snapshot must carry blob pointer records, or
    the checkpoint throws away the only index the arena had."""
    d = free_dir()
    big = payload(7, 3)
    srv = Server(binary, port, d).start()
    c = Client(port)
    c.call("SET", "kept:blob", big)
    c.call("SET", "kept:small", "s")
    c.call("SAVE")
    c.close()
    srv.sigkill()

    srv = Server(binary, port, d).start()
    c = Client(port)
    got_blob = c.call("GET", "kept:blob")
    got_small = c.call("GET", "kept:small")
    c.close()
    srv.sigkill()
    check("gh #163: blob survives SAVE + restart (snapshot stores a pointer record)",
          got_blob == big, "nil after SAVE" if got_blob is None else "")
    check("gh #163: small value survives SAVE + restart", got_small == b"s")

    snap = os.path.join(d, "pion.snapshot.0")
    snap_size = os.path.getsize(snap) if os.path.exists(snap) else -1
    check("gh #163: SAVE does not copy blob bytes into the snapshot",
          0 < snap_size < len(big), f"snapshot is {snap_size} B for a {len(big)} B blob")
    shutil.rmtree(d, ignore_errors=True)


def test_no_blob_tier_still_durable(binary, port):
    """--no-blob-tier is the documented escape hatch; it must still be correct,
    just heap-resident (values then ride the rotating WAL)."""
    d = free_dir()
    big = payload(3, 2)
    srv = Server(binary, port, d, extra=["--no-blob-tier"]).start()
    c = Client(port)
    c.call("SET", "heap:blob", big)
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    srv = Server(binary, port, d, extra=["--no-blob-tier"]).start()
    c = Client(port)
    got = c.call("GET", "heap:blob")
    c.close()
    srv.sigkill()
    check("gh #163: --no-blob-tier keeps values on the heap and still recovers",
          got == big)
    check("gh #163: --no-blob-tier creates no arena files",
          not any(f.startswith("pion.blob.") for f in os.listdir(d)))
    shutil.rmtree(d, ignore_errors=True)


def test_threshold_boundary(binary, port):
    """Values below the threshold must not change tier — that is what keeps the
    KV hot path untouched."""
    d = free_dir()
    srv = Server(binary, port, d, extra=["--blob-threshold", "65536"]).start()
    c = Client(port)
    below = b"b" * 65535
    at = b"a" * 65536
    c.call("SET", "below", below)
    c.call("SET", "at", at)
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    srv = Server(binary, port, d, extra=["--blob-threshold", "65536"]).start()
    c = Client(port)
    ok_below = c.call("GET", "below") == below
    ok_at = c.call("GET", "at") == at
    c.close()
    srv.sigkill()
    check("gh #163: value just below the threshold round-trips", ok_below)
    check("gh #163: value exactly at the threshold round-trips", ok_at)
    shutil.rmtree(d, ignore_errors=True)


def test_arena_compaction(binary, port):
    """gh #167: overwriting a blob key strands its bytes. Compaction runs at
    startup — the one moment the live set is known and no reader can be holding a
    value mid-GET — and must leave every surviving key byte-exact."""
    d = free_dir()
    mb = 12
    keep = payload(42, mb)
    srv = Server(binary, port, d).start()
    c = Client(port)
    # ~360 MB written, ~12 MB of it live: past BLOB_COMPACT_MIN_BYTES with >50%
    # stranded, which is what arms the rewrite.
    for i in range(29):
        c.call("SET", "churn", payload(i, mb))
    c.call("SET", "churn", keep)
    c.call("SET", "small", "s")
    used_before = blob_stat(c, "blob_bytes_used")
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    srv = Server(binary, port, d).start()
    c = Client(port)
    got = c.call("GET", "churn")
    small = c.call("GET", "small")
    used_after = blob_stat(c, "blob_bytes_used")
    compactions = blob_stat(c, "blob_compactions")
    records_after = blob_stat(c, "blob_records")
    c.close()
    srv.sigkill()

    check("gh #167: compaction ran at startup", compactions >= 1,
          f"blob_compactions={compactions}")
    # Verification-pass residual (1): the re-index rewrote exactly one live
    # blob, and the counter must say so — it used to read 0 after a startup
    # compaction, which made the first sanity check an operator reaches for lie.
    check("gh #167: blob_records reflects the live set after compaction",
          records_after == 1, f"blob_records={records_after}")
    check("gh #167: arena shrank to roughly the live set",
          0 < used_after < used_before // 2,
          f"{used_before} -> {used_after} bytes")
    check("gh #167: the surviving blob is byte-exact after the rewrite", got == keep,
          "nil" if got is None else f"{len(got)} B")
    check("gh #167: small values unaffected", small == b"s")

    # The rewrite invalidated every cmd-4 offset in the log; the re-index has to
    # hold across another restart or the next boot resolves stale pointers.
    srv = Server(binary, port, d).start()
    c = Client(port)
    again = c.call("GET", "churn")
    c.close()
    srv.sigkill()
    check("gh #167: re-index survives a second restart", again == keep,
          "nil" if again is None else f"{len(again)} B")
    shutil.rmtree(d, ignore_errors=True)


def test_dead_arena_compacts_under_threshold(binary, port):
    """Verification-pass residual (2): the compaction gate used to compare TOTAL
    arena bytes against the 256 MB floor, so ~200 MB of deleted blobs — a 100%
    dead arena — survived every restart with blob_compactions:0. Dead bytes are
    what the gate must read; a fully dead arena drops its files at any size."""
    d = free_dir()
    mb = 10
    srv = Server(binary, port, d).start()
    c = Client(port)
    # ~200 MB of >1 MB values, all deleted: under BLOB_COMPACT_MIN_BYTES total,
    # 100% dead. This is the exact repro from the gh #167 verification comment.
    for i in range(20):
        c.call("SET", f"dead:{i}", payload(i, mb))
    for i in range(20):
        c.call("DEL", f"dead:{i}")
    c.call("SET", "small", "s")
    used_before = blob_stat(c, "blob_bytes_used")
    c.close()
    time.sleep(1.5)
    srv.sigkill()

    srv = Server(binary, port, d).start()
    c = Client(port)
    compactions = blob_stat(c, "blob_compactions")
    used_after = blob_stat(c, "blob_bytes_used")
    records = blob_stat(c, "blob_records")
    small = c.call("GET", "small")
    dead_readable = sum(1 for i in range(20) if c.call("GET", f"dead:{i}") is not None)
    c.close()
    srv.sigkill()

    check("gh #167: a 100% dead arena compacts below the size floor",
          compactions >= 1 and used_after == 0,
          f"used {used_before} -> {used_after}, blob_compactions={compactions}")
    check("gh #167: blob_records is 0 for an emptied arena", records == 0,
          f"blob_records={records}")
    check("gh #167: deleted keys stay deleted, small survivors intact",
          dead_readable == 0 and small == b"s",
          f"dead_readable={dead_readable}")

    # The drop must actually reclaim disk, not just forget the bytes.
    blob_files = [f for f in os.listdir(d) if f.startswith("pion.blob.")]
    check("gh #167: dead segment files are unlinked", len(blob_files) == 0,
          f"leftover={blob_files}")
    shutil.rmtree(d, ignore_errors=True)


def test_high_dim_vadd_values(binary, port):
    """gh #166: `VADD key VALUES 1536 ...` is ~1539 RESP tokens. At the old
    64-token bound it could only ever be the gh #153 error."""
    d = free_dir()
    srv = Server(binary, port, d, profile="vector").start()
    c = Client(port)
    vals = [f"{(i % 97) * 0.01:.4f}" for i in range(1536)]
    vals2 = [f"{((i + 7) % 97) * 0.01:.4f}" for i in range(1536)]
    r1 = c.call("VADD", "vs", "VALUES", "1536", *vals, "elem-a")
    r2 = c.call("VADD", "vs", "VALUES", "1536", *vals2, "elem-b")
    card = c.call("VCARD", "vs")
    dim = c.call("VDIM", "vs")
    sim = c.call("VSIM", "vs", "VALUES", "1536", *vals, "COUNT", "2")
    # reply pairing must hold with a 1539-token command inside a pipeline
    c.s.sendall(cmd("INCR", "pc")
                + cmd("VADD", "vs", "VALUES", "1536", *vals, "elem-c")
                + cmd("INCR", "pc"))
    p1, p2, p3 = c.read(), c.read(), c.read()
    pong = c.call("PING")
    c.close()
    srv.sigkill()

    check("gh #166: VADD VALUES at dim 1536 is accepted", r1 == b"1",
          f"reply {r1!r}")
    check("gh #166: a second high-dim VALUES insert works", r2 == b"1")
    check("gh #166: VCARD sees both elements", card == b"2", f"reply {card!r}")
    check("gh #166: VDIM reports 1536", dim == b"1536", f"reply {dim!r}")
    check("gh #166: VSIM VALUES at dim 1536 returns the nearest element",
          isinstance(sim, list) and len(sim) >= 1 and sim[0] == b"elem-a",
          f"reply {sim!r}")
    check("gh #166: pipelined around a 1539-token command, replies stay paired",
          (p1, p2, p3) == (b"1", b"1", b"2"), f"replies {p1!r} {p2!r} {p3!r}")
    check("gh #166: connection is usable afterwards", pong == b"PONG")

    # The stack-overflow failure mode this fix exists to avoid is multi-worker
    # only, so the bound has to be exercised at -w 8. Skipped for -O0 dev
    # binaries: those SIGBUS at -w >= 2 on any command, unmodified code included
    # (verified by A/B), because their frames do not fit a parallelize worker
    # stack. Release builds are the ones that mean anything here.
    if "-dev" in os.path.basename(binary):
        print("  SKIP  gh #166: -w 8 check (dev/-O0 binary — see comment)")
    else:
        d8 = free_dir()
        srv8 = Server(binary, port, d8, profile="vector", workers=8).start()
        c8 = Client(port)
        r8 = c8.call("VADD", "vs8", "VALUES", "1536", *vals, "elem-a")
        pong8 = c8.call("PING")
        c8.close()
        srv8.sigkill()
        check("gh #166: VADD VALUES 1536 at -w 8 (worker-stack bound)",
              r8 == b"1" and pong8 == b"PONG", f"reply {r8!r}, ping {pong8!r}")
        shutil.rmtree(d8, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)


def blob_stat(c, field):
    """Pull one integer field out of INFO Persistence."""
    info = (c.call("INFO") or b"").decode(errors="replace")
    for line in info.split("\r\n"):
        if line.startswith(field + ":"):
            return int(line.split(":", 1)[1])
    return -1


def test_info_reports_tiers(binary, port):
    """An operator has to be able to see this. gh #149's failure mode was that
    nothing anywhere said a write had been dropped."""
    d = free_dir()
    srv = Server(binary, port, d).start()
    c = Client(port)
    c.call("SET", "one:blob", payload(9, 2))
    info = c.call("INFO") or b""
    c.close()
    srv.sigkill()
    text = info.decode(errors="replace")
    check("INFO exposes wal_dropped_entries", "wal_dropped_entries:" in text)
    check("INFO exposes blob tier usage", "blob_bytes_used:" in text)
    check("INFO exposes compaction counters (gh #167)",
          "blob_compactions:" in text and "blob_bytes_live_at_last_scan:" in text)
    shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(ROOT, "pion-server")))
    ap.add_argument("--port", type=int, default=1987)
    args = ap.parse_args()
    # Each test runs the server in its own data dir, so a relative binary path
    # would resolve against that dir instead of the repo.
    args.binary = os.path.abspath(args.binary)

    if not os.path.exists(args.binary):
        print(f"binary not found: {args.binary}")
        return 2

    print(f"gh #149 / gh #163 blob durability — {args.binary} port {args.port}\n")
    for fn in (test_blobs_past_the_old_ring,
               test_wal_rotation_for_small_values,
               test_wal_full_is_loud,
               test_overwrite_and_delete_ordering,
               test_save_keeps_blob_pointers,
               test_no_blob_tier_still_durable,
               test_threshold_boundary,
               test_arena_compaction,
               test_dead_arena_compacts_under_threshold,
               test_high_dim_vadd_values,
               test_info_reports_tiers):
        print(f"{fn.__name__}:")
        try:
            fn(args.binary, args.port)
        except Exception as exc:            # a crash is a failure, not an error
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        print()

    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    return 1 if FAIL else 0


sys.exit(main())
