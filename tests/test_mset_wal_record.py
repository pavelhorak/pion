#!/usr/bin/env python3
"""MSET logs ONE cmd-31 WAL record, and it replays exactly like the SETs it replaces.

Durable MSET is bound by WAL bytes: +31% log bytes cost -6% MSET throughput,
and -0.5% with --no-wal. So a fast-path MSET logs one record with
LEB128 lengths ([varint kl][key][varint vl][val] per pair) instead of one
13-byte-header SET record per pair. The gate's 10-key MSET drops from 320 to
223 log bytes.

What this proves, after a SIGKILL (so WAL replay is the only way back):
  * every pair comes back byte-exact, across the varint boundaries
    (0, 127, 128, 16383, 16384 bytes) and both key representations (SSO <= 23
    bytes, heap above);
  * a key repeated inside one MSET ends with its LAST value, as N SETs would;
  * ordering against the surrounding SET / DEL / MSET records holds;
  * each small MSET (the gate's shape among them) added exactly ONE cmd-31
    record. The WAL file is parsed before and after, so this cannot pass
    vacuously on the old per-pair SET records. Large MSETs are only checked
    for replay: one that spans several recv() calls may legitimately take the
    slow path, which logs per-key SETs.

    python3 tests/test_mset_wal_record.py [binary]
"""
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile

import redis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid, wait_port_free  # noqa: E402

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1987
WAL_HEADER_SIZE = 64

failures = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        failures.append(name)


def spawn(workdir):
    p = subprocess.Popen([os.path.abspath(BINARY), "-p", str(PORT), "-w", "1",
                          "--no-auto-detect", "--no-auto-embed", "--no-crash-log"],
                         cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready_pid(PORT, p, 30)   # this process, not a lingering listener (#27)
    except RuntimeError:
        p.kill()
        sys.exit("FAIL: server did not start")
    return p


def wal_records(path):
    """(cmd_id, key_len, val_len) of every record up to the published tail."""
    with open(path, "rb") as f:
        data = f.read()
    _magic, tail = struct.unpack_from("<QQ", data, 0)
    out, off = [], 0
    body = data[WAL_HEADER_SIZE:WAL_HEADER_SIZE + tail]
    while off + 13 <= len(body):
        elen, cmd, kl = struct.unpack_from("<IBI", body, off)
        vl, = struct.unpack_from("<I", body, off + 9 + kl)
        out.append((cmd, kl, vl))
        off += elen
    return out


def main():
    workdir = tempfile.mkdtemp(prefix="pion_mset_wal_")
    proc = spawn(workdir)
    try:
        r = redis.Redis(port=PORT)
        expected = {}

        wal = os.path.join(workdir, "pion.wal.0")

        def mset(pairs, one_record=False):
            flat = []
            for k, v in pairs:
                flat += [k, v]
                expected[k] = v
            before = len(wal_records(wal)) if one_record else 0
            assert r.execute_command("MSET", *flat) in (True, b"OK")
            if one_record:
                added = wal_records(wal)[before:]
                check(f"{len(pairs)}-pair MSET logged one cmd-31 record",
                      [c for c, _, _ in added] == [31], f"added {added}")

        print("writing")
        # varint boundaries on the value, SSO and heap keys
        sizes = [0, 1, 127, 128, 129, 16383, 16384, 70000]
        mset([(b"sso:%d" % n, bytes([65 + n % 26]) * n) for n in sizes])
        mset([(b"heap-key-over-twenty-three-bytes:%06d" % n, b"v" * n) for n in sizes])
        # varint boundaries on the key
        mset([(b"k" * n, b"key%d" % n) for n in (1, 127, 128, 200)], one_record=True)
        # the gate's shape: one key ten times, last wins
        mset([(b"key:__rand_int__", b"x%d" % i) for i in range(10)], one_record=True)
        # interleaving with SET / DEL / another MSET
        r.set(b"sso:1", b"overwritten-by-SET")
        expected[b"sso:1"] = b"overwritten-by-SET"
        r.delete(b"sso:127")
        del expected[b"sso:127"]
        mset([(b"sso:128", b"second-mset"), (b"new", b"n")], one_record=True)
        # binary-safe bytes, including \r\n and 0x80+ that look like varint continuation
        mset([(b"bin\r\n\x80\xff", bytes(range(256)))], one_record=True)

        recs = wal_records(wal)
        check("cmd-31 records carry an empty key", all(kl == 0 for c, kl, _ in recs if c == 31))

        print("SIGKILL + replay")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        wait_port_free(PORT)
        proc = spawn(workdir)
        r = redis.Redis(port=PORT)

        bad = [k for k, v in expected.items() if r.get(k) != v]
        check(f"all {len(expected)} keys byte-exact after replay", not bad,
              f"{len(bad)} wrong, e.g. {bad[:3]}")
        check("deleted key stays deleted", r.get(b"sso:127") is None)
        check("repeated key ends with its LAST value", r.get(b"key:__rand_int__") == b"x9")
        check("DBSIZE matches", r.dbsize() == len(expected), f"{r.dbsize()} vs {len(expected)}")
    finally:
        proc.kill()
        proc.wait()
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'FAIL' if failures else 'PASS'} ({len(failures)} failure(s))")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
