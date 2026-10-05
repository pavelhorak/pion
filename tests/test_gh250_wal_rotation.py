#!/usr/bin/env python3
"""gh #250 regression test — a WAL segment rotation must not lose records.

`_replay_segment` bounds its walk by the sealed file's on-disk header word 1,
and `_rotate` used to munmap and rename the active file without ever stamping
`tail_offset` into it. Since gh #229 the header is published once per COMMAND,
so a multi-record command that rotated mid-way sealed the segment at the end of
the PREVIOUS command — everything it had already written was left past the
recorded tail and became unreachable. The client had its `+OK`,
`wal_dropped_entries` read 0, and the data was gone.

Measured before the fix: 2000 acked 10-pair MSETs -> 19,998 of 20,000 keys back,
the two lost ones being exactly pairs 0 and 1 of the single command that
straddled the rotation.

Nothing else in the suite rotates — every other durability test (gh #170, #174,
#230) fits inside one 256 MiB segment — which is why this survived a
bug-by-bug audit of this file. Hence `--wal-size 1`.

Usage: python3 tests/test_gh250_wal_rotation.py [./pion-server]
"""
import os, socket, subprocess, sys, time, shutil, signal

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1986
WORKDIR = f"/tmp/pion_gh250_test_{PORT}"

# Enough 10-pair MSETs at a 40-byte value to overrun a 1 MiB segment several
# times over. The point is to cross the boundary repeatedly, not just once.
MSETS = 2000
PAIRS = 10
VAL = "v" * 40

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


def spawn():
    # The server's output and its crash log / status file stay in WORKDIR, and
    # are printed if the run fails: a reset connection on Linux x86 left no
    # evidence while they went to /dev/null and WORKDIR was deleted.
    return subprocess.Popen([BINARY, "-p", str(PORT), "-w", "1", "--no-auto-detect",
                             "--no-auto-embed", "--wal-size", "1"],
                            cwd=WORKDIR, stdout=open(os.path.join(WORKDIR, "server.log"), "ab"),
                            stderr=subprocess.STDOUT)


def evidence():
    for name in sorted(os.listdir(WORKDIR)) if os.path.isdir(WORKDIR) else []:
        if name == "server.log" or name.endswith(".crash.log") or name.endswith(".status"):
            with open(os.path.join(WORKDIR, name), "rb") as f:
                tail = f.read()[-4000:].decode("utf-8", "replace")
            print(f"--- {name} (tail) ---\n{tail}")


def connect():
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try: return Client(PORT)
        except OSError: time.sleep(0.25)
    raise RuntimeError("server did not come up")


def sealed_count(c):
    info = c("INFO", "persistence") or b""
    for line in info.split(b"\n"):
        if line.startswith(b"wal_segments_sealed:"):
            return int(line.split(b":")[1])
    return -1


def main():
    shutil.rmtree(WORKDIR, ignore_errors=True); os.makedirs(WORKDIR, exist_ok=True)
    proc = spawn()
    try:
        c = connect()
        expected = {}
        acked = 0
        for i in range(MSETS):
            args = ["MSET"]
            for j in range(PAIRS):
                k, v = f"r:{i:05d}:{j}", f"{VAL}{i}-{j}"
                args += [k, v]; expected[k] = v
            if c(*args) == b"+OK":
                acked += 1
        # Other record shapes must survive a rotation too, not just cmd-1 SETs.
        for i in range(200):
            c("SET", f"s:{i:05d}", f"{VAL}{i}"); expected[f"s:{i:05d}"] = f"{VAL}{i}"
            c("HSET", f"h:{i:05d}", "f", f"{VAL}{i}")
            c("RPUSH", f"l:{i:05d}", f"{VAL}{i}")

        check("every MSET acked", acked == MSETS, f"{acked}/{MSETS}")
        sealed = sealed_count(c)
        # Without this the test can pass vacuously: no rotation, nothing tested.
        check("the log actually rotated (>=1 sealed segment)", sealed >= 1,
              f"wal_segments_sealed={sealed} — raise MSETS or lower --wal-size")
        info = c("INFO", "persistence")
        check("no entry was dropped",
              isinstance(info, bytes) and b"wal_dropped_entries:0" in info)
        c.close()

        proc.send_signal(signal.SIGKILL); proc.wait(timeout=15)
        proc = spawn()
        c = connect()

        try:
            missing = [k for k, v in expected.items() if c("GET", k) != v.encode()]
        except OSError:
            print(f"restarted server: exit code {proc.poll()}")
            evidence()
            raise
        check("ZERO string keys lost across the rotation", not missing,
              f"{len(missing)} lost, e.g. {sorted(missing)[:6]}")
        h_missing = [i for i in range(200)
                     if c("HGET", f"h:{i:05d}", "f") != f"{VAL}{i}".encode()]
        check("ZERO hash fields lost across the rotation", not h_missing,
              f"{len(h_missing)} lost, e.g. {h_missing[:6]}")
        l_missing = [i for i in range(200) if c("LLEN", f"l:{i:05d}") != b":1"]
        check("ZERO list elements lost across the rotation", not l_missing,
              f"{len(l_missing)} lost, e.g. {l_missing[:6]}")
        c.close()
    finally:
        try: proc.kill(); proc.wait(timeout=5)
        except Exception: pass
        if failures:
            evidence()
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"  FAILED: {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
