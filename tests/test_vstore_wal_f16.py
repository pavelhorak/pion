#!/usr/bin/env python3
"""V.STOREBATCH fp16 input + fp16 WAL records (op 5), across a SIGKILL.

An fp16-stored layer used to be logged as the fp32 it was sent in: 4 bytes
per value for 2 bytes of state, so a 16K-token prompt on a 28-layer model
wrote ~3.7 GB of WAL. Now:

  1. V.STOREBATCH accepts an fp16 blob with a trailing `FMT F16` and stores it
     losslessly. The format is declared, never inferred: a half-dim fp32
     blob is exactly as long as a full-dim fp16 one and must stay an error.
  2. An fp16-stored layer logs op 5 with the fp16 values it holds — half the
     bytes — whether the input was fp16 or fp32.
  3. Replay (fp16 -> fp32 -> fp16) restores the layer bit for bit.
  4. Non-fp16 formats (int8 here) still log fp32 (op 2) and replay exactly.

The server runs in a private temp dir that is removed afterwards.

    python3 tests/test_vstore_wal_f16.py [--port 1994]
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

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIM = 256
N = 200


class RESP:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    def call(self, *parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            b = p if isinstance(p, bytes) else str(p).encode()
            out += [f"${len(b)}\r\n".encode(), b, b"\r\n"]
        self.s.sendall(b"".join(out))
        line = self.f.readline()
        t = line[:1]
        if t == b"$":
            n = int(line[1:-2])
            return None if n < 0 else self.f.read(n + 2)[:-2]
        return line[:-2]


def start(port, cwd, log):
    proc = subprocess.Popen([os.environ.get("PION_BIN") or os.path.join(ROOT, "pion-server"), "--kvcache", "-w", "1", "-p", str(port),
                             "--no-auto-detect", "--no-auto-embed"],
                            cwd=cwd, stdout=log, stderr=log, preexec_fn=os.setsid)
    for _ in range(100):
        try:
            r = RESP(port)
            if r.call("PING") == b"+PONG":
                return proc, r
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def fetch(r, sid, layer):
    b = r.call("V.FETCH", sid, layer, "RANGE", 0, N)
    return np.frombuffer(b, dtype=np.float32).reshape(N, DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1994)
    port = ap.parse_args().port
    d = tempfile.mkdtemp(prefix="pion_wal_f16_")
    log = open(os.path.join(d, "server.log"), "w")
    proc = None
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += 0 if ok else 1

    try:
        proc, r = start(port, d, log)
        rng = np.random.default_rng(1)
        a32 = rng.standard_normal((N // 2, DIM)).astype(np.float32)          # fp32 input
        b16 = rng.standard_normal((N // 2, DIM)).astype(np.float16)          # fp16 input
        c16 = rng.standard_normal((N, DIM)).astype(np.float16)
        i32 = rng.standard_normal((N, DIM)).astype(np.float32)

        check("register fp16 ns", r.call("KV.PREFIX.REGISTER", "wf16", DIM, "fp16") == b"+OK")
        check("register int8 ns", r.call("KV.PREFIX.REGISTER", "wi8", DIM, "int8") == b"+OK")
        wal = os.path.join(d, "pion.vstore.wal.0")
        w0 = os.path.getsize(wal)
        check("fp32 blob into fp16 layer", r.call("V.STOREBATCH", "wf16_pk", 0, 0, N // 2, a32.tobytes()) == b"+OK")
        check("fp16 blob into fp16 layer", r.call("V.STOREBATCH", "wf16_pk", 0, N // 2, N // 2, b16.tobytes(), "FMT", "F16") == b"+OK")
        check("fp16 blob, second layer", r.call("V.STOREBATCH", "wf16_pk", 1, 0, N, c16.tobytes(), "fmt", "f16") == b"+OK")
        w1 = os.path.getsize(wal)
        check("fp32 blob into int8 layer", r.call("V.STOREBATCH", "wi8_pk", 0, 0, N, i32.tobytes()) == b"+OK")
        w2 = os.path.getsize(wal)
        half = np.zeros((N, DIM // 2), dtype=np.float32).tobytes()     # fp16-sized, but declared fp32
        check("an fp16-sized blob without FMT F16 is refused",
              r.call("V.STOREBATCH", "wf16_pk", 2, 0, N, half).startswith(b"-ERR V.STOREBATCH: blob size"))
        check("an fp32-sized blob with FMT F16 is refused",
              r.call("V.STOREBATCH", "wf16_pk", 2, 0, N // 2, a32.tobytes(), "FMT", "F16")
              .startswith(b"-ERR V.STOREBATCH: blob size"))
        check("an unknown option is refused",
              r.call("V.STOREBATCH", "wf16_pk", 2, 0, N, c16.tobytes(), "FMT", "BF16").startswith(b"-ERR"))
        # Pipelined STOREBATCH + PING: exactly two replies, in order (the
        # option scan is bounded by the command, not the receive buffer).
        pipe = b"".join([
            b"*6\r\n$12\r\nV.STOREBATCH\r\n$7\r\nwf16_pk\r\n$1\r\n3\r\n$1\r\n0\r\n$1\r\n1\r\n",
            f"${DIM * 4}\r\n".encode(), np.zeros(DIM, dtype=np.float32).tobytes(), b"\r\n",
            b"*1\r\n$4\r\nPING\r\n"])
        r.s.sendall(pipe)
        replies = [r.f.readline(), r.f.readline()]
        check(f"pipelined STOREBATCH + PING -> 2 replies {replies}", replies == [b"+OK\r\n", b"+PONG\r\n"])

        f16_bytes = w1 - w0          # (w2 is taken before the refusal checks)
        i8_bytes = w2 - w1
        check(f"fp16 layers logged at 2 B/value ({f16_bytes} B for {2 * N * DIM} values)",
              2 * N * DIM * 2 <= f16_bytes < 2 * N * DIM * 2 + 1024)
        check(f"int8 layer still logged at 4 B/value ({i8_bytes} B for {N * DIM} values)",
              N * DIM * 4 <= i8_bytes < N * DIM * 4 + 1024)

        before = {(s, l): fetch(r, s, l) for s, l in (("wf16_pk", 0), ("wf16_pk", 1), ("wi8_pk", 0))}
        check("fp16 input round-trips exactly",
              np.array_equal(before[("wf16_pk", 0)][N // 2:], b16.astype(np.float32))
              and np.array_equal(before[("wf16_pk", 1)], c16.astype(np.float32)))
        check("fp32 input into fp16 layer == its fp16 rounding",
              np.array_equal(before[("wf16_pk", 0)][:N // 2], a32.astype(np.float16).astype(np.float32)))

        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        proc, r = start(port, d, log)
        for key, ref in before.items():
            got = fetch(r, *key)
            check(f"{key[0]} layer {key[1]} bit-identical after SIGKILL + replay", np.array_equal(got, ref))
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.close()
        if fails:
            print(f"server log kept in {d}")
        else:
            shutil.rmtree(d, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
