#!/usr/bin/env python3
"""gh #193: V.FETCH FMT NATIVE — raw fp16 wire format.

Checks:
  1. RANGE FMT NATIVE on an fp16 session returns exactly half the bytes and
     the fp16 payload upcasts bit-exactly to the fp32-path reply.
  2. BATCH FMT NATIVE (with and without explicit num_layers) — same identity
     per layer.
  3. Mixed SCHEMA session: fp16 layers reply fp16, int8 layers reply fp32
     (length disambiguates) under one FMT NATIVE BATCH call.
  4. Tail zero-fill past the stored token count matches the fp32 path.
  5. Legacy explicit-ids form + FMT NATIVE → clean -ERR.
  6. Case-insensitive fmt/native tokens.
  7. Frame-sync: V.FETCH ... FMT NATIVE pipelined with PING answers both
     (the cmd_end_tok bound — gh #166/#162 desync class).

Requires: server started with --kvcache. Run: python3 tests/test_gh193.py [binary]
"""
import os, socket, subprocess, sys, time, shutil
import numpy as np

BINARY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server")
PORT = 1983
WORKDIR = f"/tmp/pion_gh193_test_{PORT}"

passed = failed = 0
def check(cond, name, detail=""):
    global passed, failed
    if cond: passed += 1; print(f"  PASS {name}")
    else: failed += 1; print(f"  FAIL {name} {detail}")

def encode(args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, str): a = a.encode()
        parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
    return b"".join(parts)

class Conn:
    def __init__(self):
        deadline = time.monotonic() + 30
        while True:
            try:
                self.s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
                break
            except OSError:
                if time.monotonic() > deadline: raise
                time.sleep(0.2)
        self.f = self.s.makefile("rb")
    def call(self, *args):
        self.s.sendall(encode(args))
        return self.read_reply()
    def read_reply(self):
        line = self.f.readline()
        t = line[:1]
        if t in (b"+", b"-", b":"):
            return line.rstrip(b"\r\n")
        if t == b"$":
            n = int(line[1:])
            if n == -1: return None
            payload = self.f.read(n); self.f.read(2)
            return payload
        if t == b"*":
            n = int(line[1:])
            if n == -1: return None
            return [self.read_reply() for _ in range(n)]
        raise RuntimeError(f"unexpected reply start: {line!r}")

def main():
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    proc = subprocess.Popen(
        [os.path.abspath(BINARY), "-p", str(PORT), "-w", "1", "--kvcache", "--no-auto-embed"],
        cwd=WORKDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        c = Conn()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if c.call("PING") == b"+PONG": break
            except Exception:
                pass
            time.sleep(0.5)

        DIM, N = 64, 100
        rng = np.random.default_rng(3)
        data = rng.standard_normal((N, DIM)).astype(np.float32)
        f16 = data.astype(np.float16)  # what an fp16 session actually stores

        # ── fp16 session ──
        r = c.call("V.CREATE", "s16", str(DIM), "VQUANT", "fp16")
        check(r.startswith(b":"), "V.CREATE fp16", r)
        r = c.call("V.STOREBATCH", "s16", "0", "0", str(N), data.tobytes())
        check(r == b"+OK", "STOREBATCH layer 0", r)
        r = c.call("V.STOREBATCH", "s16", "1", "0", str(N), (data * 2).tobytes())
        check(r == b"+OK", "STOREBATCH layer 1", r)

        # 1. RANGE identity
        p32 = c.call("V.FETCH", "s16", "0", "RANGE", "0", str(N))
        pnat = c.call("V.FETCH", "s16", "0", "RANGE", "0", str(N), "FMT", "NATIVE")
        check(len(p32) == N * DIM * 4, "fp32 RANGE size", len(p32))
        check(len(pnat) == N * DIM * 2, "native RANGE is half the bytes", len(pnat))
        a32 = np.frombuffer(p32, dtype=np.float32)
        a16 = np.frombuffer(pnat, dtype=np.float16)
        check(np.array_equal(a16.astype(np.float32), a32), "native upcast == fp32 reply")
        check(np.array_equal(a16, f16.reshape(-1)), "native bytes == stored fp16")

        # 2. BATCH identity (explicit and implicit layer count)
        for extra in (["2"], []):
            reps = c.call("V.FETCH", "s16", "BATCH", "0", str(N), *extra, "FMT", "NATIVE")
            tag = "explicit-N" if extra else "implicit-N"
            check(isinstance(reps, list) and len(reps) == 2, f"BATCH {tag} layer count", reps if not isinstance(reps, list) else len(reps))
            ok = all(len(p) == N * DIM * 2 for p in reps)
            check(ok, f"BATCH {tag} all-native sizes")
        reps32 = c.call("V.FETCH", "s16", "BATCH", "0", str(N), "2")
        b16 = np.frombuffer(c.call("V.FETCH", "s16", "BATCH", "0", str(N), "2", "FMT", "NATIVE")[1], dtype=np.float16)
        b32 = np.frombuffer(reps32[1], dtype=np.float32)
        check(np.array_equal(b16.astype(np.float32), b32), "BATCH layer 1 native == fp32")

        # 3. Mixed schema: layer0 fp16, layer1 int8
        r = c.call("V.CREATE", "smix", "0", "SCHEMA", "2", f"fmt=fp16,dim={DIM}", f"fmt=int8,dim={DIM}")
        check(r.startswith(b":"), "V.CREATE mixed schema", r)
        c.call("V.STOREBATCH", "smix", "0", "0", str(N), data.tobytes())
        c.call("V.STOREBATCH", "smix", "1", "0", str(N), data.tobytes())
        reps = c.call("V.FETCH", "smix", "BATCH", "0", str(N), "FMT", "NATIVE")
        check(len(reps[0]) == N * DIM * 2, "mixed: fp16 layer native", len(reps[0]))
        check(len(reps[1]) == N * DIM * 4, "mixed: int8 layer stays fp32", len(reps[1]))

        # 4. Tail zero-fill past stored tokens
        M = N + 20
        p32 = c.call("V.FETCH", "s16", "0", "RANGE", "0", str(M))
        pnat = c.call("V.FETCH", "s16", "0", "RANGE", "0", str(M), "FMT", "NATIVE")
        a32 = np.frombuffer(p32, dtype=np.float32).reshape(M, DIM)
        a16 = np.frombuffer(pnat, dtype=np.float16).reshape(M, DIM)
        check(np.array_equal(a16.astype(np.float32), a32), "tail region identical across formats")
        check(not a16[N:].any(), "native tail is zero-filled")

        # 5. Legacy ids + FMT NATIVE → error
        r = c.call("V.FETCH", "s16", "0", "1", "2", "FMT", "NATIVE")
        check(isinstance(r, bytes) and r.startswith(b"-ERR"), "legacy ids + NATIVE errors", r)

        # 6. Case-insensitive
        p = c.call("V.FETCH", "s16", "0", "RANGE", "0", str(N), "fmt", "native")
        check(len(p) == N * DIM * 2, "lowercase fmt native", len(p))

        # 7. Pipelined frame-sync: NATIVE fetch + PING in one send
        c.s.sendall(encode(["V.FETCH", "s16", "0", "RANGE", "0", str(N), "FMT", "NATIVE"]) + encode(["PING"]))
        r1 = c.read_reply(); r2 = c.read_reply()
        check(len(r1) == N * DIM * 2 and r2 == b"+PONG", "pipelined NATIVE + PING stay in sync", (len(r1) if r1 else None, r2))

        # 8. INT8 plain session ignores the flag (falls back fp32)
        r = c.call("V.CREATE", "s8", str(DIM), "VQUANT", "int8")
        c.call("V.STOREBATCH", "s8", "0", "0", str(N), data.tobytes())
        p = c.call("V.FETCH", "s8", "0", "RANGE", "0", str(N), "FMT", "NATIVE")
        check(len(p) == N * DIM * 4, "int8 session replies fp32 under NATIVE", len(p))

    finally:
        proc.kill(); proc.wait(timeout=5)
        shutil.rmtree(WORKDIR, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
