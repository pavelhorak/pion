#!/usr/bin/env python3
"""gh #183 — reproducible KV round-trip evidence harness.

The claim this exists to make legible: Pion's KV tiers round-trip **exactly**,
across restarts and under concurrency. The 2026 ecosystem's live bug class is
*silent* KV corruption, so the evidence has to be a diff anyone can re-run, not
a benchmark number.

WHAT IS AND IS NOT ASSERTED — this distinction is the whole point, and blurring
it would make the pack worthless:

  * `fp16` is the production default and is asserted **byte-exact** on the
    round-trip: what comes back must equal what a correct fp16 encode of the
    input produces, bit for bit. No tolerance.
  * The quantized tiers (`int8`, `turbo4/3/2`, `fp8`, `mlx4g32`) are LOSSY by
    construction. Claiming "bit-exact int4" would be false. What IS asserted
    for them is the property that actually matters for a cache:
      - DETERMINISM: the same input stored twice yields byte-identical fetches.
      - STABILITY ACROSS RESTART: the bytes after a restart equal the bytes
        before it, exactly.
      - BOUNDED ERROR: reconstruction stays within the tier's documented band.
    A cache that is lossy-but-deterministic is safe to share between processes;
    a cache that is lossy-and-drifting is the vLLM #39146 failure mode.

Usage:
  python3 tests/test_kv_roundtrip_evidence.py                 # in-process tiers
  python3 tests/test_kv_roundtrip_evidence.py --restart       # + cross-restart
  python3 tests/test_kv_roundtrip_evidence.py --concurrency 8 # + parallel writers
  python3 tests/test_kv_roundtrip_evidence.py --binary ./pion-server --all

Requires a server started with --kvcache (the harness can start its own).
"""

import argparse
import hashlib
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

HOST = "127.0.0.1"

# (name, is_lossless, max_abs_err_allowed)
# The error bands are properties of the tier's encoding, not tuning knobs: int8
# is a per-group affine code, turbo4/3/2 drop to 4/3/2 bits, mlx4g32 is int4
# with group size 32. A tier that exceeds its band is a real regression.
TIERS = [
    ("fp16", True, 0.0),
    ("int8", False, 0.05),
    ("turbo4", False, 0.35),
    ("turbo3", False, 0.75),
    # turbo2 was 1.5 until 2026-09-25, which sat just above a quantizer bug
    # that decoded every INT2 value at 2x (max_abs_err 1.333 at the defaults).
    # Fixed, it measures 0.333; 0.75 keeps turbo3's headroom and fails the bug.
    ("turbo2", False, 0.75),
    ("mlx4g32", False, 0.35),
]

PASSED, FAILED, SKIPPED = [], [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")


def skip(name, why):
    SKIPPED.append(name)
    print(f"  SKIP  {name:52s} {why}")


class Resp:
    def __init__(self, port, timeout=30):
        self.s = socket.create_connection((HOST, port), timeout=timeout)
        self.s.settimeout(timeout)
        self.f = self.s.makefile("rb")

    def call(self, *args):
        buf = f"*{len(args)}\r\n".encode()
        for a in args:
            a = a.encode() if isinstance(a, str) else a
            buf += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(buf)
        return self._read()

    def _read(self):
        line = self.f.readline()
        if not line:
            raise EOFError("server closed")
        t, body = line[:1], line[1:-2]
        if t == b"+":
            return body.decode()
        if t == b"-":
            raise RuntimeError(body.decode())
        if t == b":":
            return int(body)
        if t == b"$":
            n = int(body)
            if n == -1:
                return None
            return self.f.read(n + 2)[:-2]        # raw bytes, never decoded
        if t == b"*":
            n = int(body)
            return None if n == -1 else [self._read() for _ in range(n)]
        raise ValueError(f"bad RESP {line!r}")

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def make_vectors(n_tokens, dim, seed):
    """Deterministic pseudo-random fp32 payload — reproducible from the seed
    alone, so anyone can regenerate the exact input this harness used."""
    vals = []
    x = seed & 0xFFFFFFFF
    for _ in range(n_tokens * dim):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        vals.append(((x / 0x7FFFFFFF) * 2.0) - 1.0)
    return vals


def pack_f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def unpack_f32(blob):
    return list(struct.unpack("<%df" % (len(blob) // 4), blob))


def fp16_of_wire(vals):
    """Expected fp16 result for values that travelled the wire as fp32.

    The path is float64 -> fp32 (V.STOREBATCH blob) -> fp16 (server store), so
    the expectation MUST round through fp32 too. Computing fp16 straight from
    the float64 double-rounds and disagrees at exact midpoints — that artifact
    produced a bogus "cross-session bleed under concurrency" here (1-2 elements
    of 8192, only for some seeds) until a single-threaded control with the same
    seeds reproduced it identically and exonerated the server. An evidence
    harness that cries corruption is worse than no harness.
    """
    out = []
    for v in vals:
        as_f32 = struct.unpack("<f", struct.pack("<f", v))[0]
        out.append(struct.unpack("<e", struct.pack("<e", as_f32))[0])
    return out


def store_and_fetch(r, sid, tier, n_tokens, dim, vals, layer=0):
    """Create a session in `tier`, store one batch, fetch it back."""
    r.call("V.CREATE", sid, str(dim), "VQUANT", tier)
    r.call("V.STOREBATCH", sid, str(layer), "0", str(n_tokens), pack_f32(vals))
    # RANGE is half-open [start, end): `RANGE 0 n` returns n tokens,
    # and `RANGE 0 0` is rejected outright ("end must exceed start").
    got = r.call("V.FETCH", sid, str(layer), "RANGE", "0", str(n_tokens))
    return got


def max_abs_err(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a and b else float("inf")


def tier_roundtrip(port, n_tokens, dim):
    print("\n[1] Single-process round-trip per tier")
    digests = {}
    for tier, lossless, band in TIERS:
        sid = f"gh183:{tier}:rt"
        vals = make_vectors(n_tokens, dim, seed=42)
        try:
            r = Resp(port)
            got = store_and_fetch(r, sid, tier, n_tokens, dim, vals)
            r.close()
        except RuntimeError as e:
            skip(f"{tier}: round-trip", f"tier unavailable ({e})")
            continue
        if got is None:
            check(f"{tier}: round-trip returned data", False, "got nil")
            continue
        back = unpack_f32(got)
        digests[tier] = hashlib.sha256(got).hexdigest()
        if len(back) != len(vals):
            check(f"{tier}: element count", False,
                  f"sent {len(vals)}, got {len(back)}")
            continue
        err = max_abs_err(vals, back)
        if lossless:
            # fp16 is asserted on the ENCODED value, not the fp32 original:
            # a correct fp16 store loses mantissa bits by design, so the
            # honest bit-exactness claim is "equals fp16(input) exactly".
            expect = fp16_of_wire(vals)
            exact = all(x == y for x, y in zip(expect, back))
            check(f"{tier}: byte-exact vs fp16(input)", exact,
                  f"max_abs_diff={max_abs_err(expect, back):.3e}")
        else:
            check(f"{tier}: reconstruction within band", err <= band,
                  f"max_abs_err={err:.4f} (band {band})")
    return digests


def determinism(port, n_tokens, dim, digests):
    print("\n[2] Determinism — same input stored twice is byte-identical")
    for tier, _, _ in TIERS:
        if tier not in digests:
            continue
        sid = f"gh183:{tier}:det"
        vals = make_vectors(n_tokens, dim, seed=42)
        try:
            r = Resp(port)
            got = store_and_fetch(r, sid, tier, n_tokens, dim, vals)
            r.close()
        except RuntimeError as e:
            skip(f"{tier}: determinism", str(e))
            continue
        d = hashlib.sha256(got).hexdigest()
        check(f"{tier}: identical bytes on re-store", d == digests[tier],
              f"sha256 {d[:16]}… vs {digests[tier][:16]}…")


def concurrency(port, n_tokens, dim, digests, workers):
    print(f"\n[3] Concurrency — {workers} parallel writers on distinct sessions")
    # Concurrent CPU-offload/eviction is where the ecosystem's silent
    # corruptions live (vLLM #31210). Distinct sessions must not bleed.
    results, errors = {}, []
    lock = threading.Lock()

    def worker(k):
        sid = f"gh183:conc:{k}"
        vals = make_vectors(n_tokens, dim, seed=1000 + k)
        try:
            r = Resp(port)
            got = store_and_fetch(r, sid, "fp16", n_tokens, dim, vals)
            r.close()
            with lock:
                results[k] = (hashlib.sha256(got).hexdigest(), vals, got)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            with lock:
                errors.append(f"worker {k}: {type(e).__name__}: {e}")

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    check("all concurrent writers completed", not errors,
          "; ".join(errors[:3]) if errors else f"{len(results)}/{workers}")
    if len(results) == workers:
        # Every session must return ITS OWN payload — a cross-session bleed is
        # exactly the failure a single-threaded test cannot see.
        bleed = []
        for k, (_, vals, got) in results.items():
            back = unpack_f32(got)
            expect = fp16_of_wire(vals)
            if not all(x == y for x, y in zip(expect, back)):
                bleed.append(k)
        check("no cross-session bleed under concurrency", not bleed,
              f"sessions wrong: {bleed}" if bleed else f"{workers} sessions verified")
        check("all digests distinct (no shared buffer)",
              len({d for d, _, _ in results.values()}) == workers,
              f"{len({d for d, _, _ in results.values()})} unique of {workers}")


def cross_restart(binary, port, n_tokens, dim):
    print("\n[4] Cross-restart — bytes after restart must equal bytes before")
    tiers = [t for t in TIERS]
    before = {}
    ssm_blob = bytes((i * 2654435761 >> 13) & 0xFF for i in range(64 * 1024))
    ssm_sid = "gh183:ssm:restart"
    proc = start_server(binary, port)
    if proc is None:
        skip("cross-restart", "could not start server")
        return
    try:
        try:
            r = Resp(port)
            r.call("SSM.PREFIX.STORE", ssm_sid, "0", ssm_blob)
            r.call("SAVE")
            r.close()
        except (RuntimeError, OSError) as e:
            skip("SSM: pre-restart store", str(e))
        for tier, _, _ in tiers:
            sid = f"gh183:{tier}:restart"
            vals = make_vectors(n_tokens, dim, seed=7)
            try:
                r = Resp(port)
                got = store_and_fetch(r, sid, tier, n_tokens, dim, vals)
                r.call("SAVE")
                r.close()
                before[tier] = hashlib.sha256(got).hexdigest()
            except RuntimeError as e:
                skip(f"{tier}: pre-restart store", str(e))
    finally:
        stop_server(proc)

    proc = start_server(binary, port)
    if proc is None:
        skip("cross-restart (second boot)", "could not restart")
        return
    try:
        for tier, _, _ in tiers:
            if tier not in before:
                continue
            sid = f"gh183:{tier}:restart"
            try:
                r = Resp(port)
                got = r.call("V.FETCH", sid, "0", "RANGE", "0", str(n_tokens))
                r.close()
            except RuntimeError as e:
                check(f"{tier}: survives restart", False, str(e))
                continue
            if got is None:
                check(f"{tier}: survives restart", False, "nil after restart")
                continue
            d = hashlib.sha256(got).hexdigest()
            check(f"{tier}: byte-identical across restart", d == before[tier],
                  f"{d[:16]}… vs {before[tier][:16]}…")
        # SSM state is opaque bytes, so the restart claim is unqualified.
        try:
            r = Resp(port)
            got = r.call("SSM.PREFIX.FETCH", ssm_sid, "0")
            r.close()
            check("SSM: byte-identical across restart", got == ssm_blob,
                  "64 KB opaque blob" if got == ssm_blob
                  else f"got {len(got) if got else 0}B of {len(ssm_blob)}B")
        except (RuntimeError, OSError) as e:
            check("SSM: byte-identical across restart", False, str(e))
    finally:
        stop_server(proc)


def ssm_roundtrip(port, sizes):
    """SSM.PREFIX.* stores OPAQUE BYTES — no quantization, no reinterpretation.

    This is the strongest claim in the pack and the only tier where
    "bit-exact" is unqualified: every byte in must be the same byte out. The
    payloads deliberately include all-zero and all-0xFF runs (which a
    length-or-sentinel bug mangles) and high-entropy data (which a truncation
    bug shortens), and the largest size crosses the >3 MB bulk-string boundary
    that needed writev (gh #76).
    """
    print("\n[5] SSM.PREFIX.* — opaque blob, unqualified bit-exactness")
    patterns = [
        ("high-entropy", lambda n: bytes((i * 2654435761 >> 13) & 0xFF for i in range(n))),
        ("all-zero", lambda n: b"\x00" * n),
        ("all-ones", lambda n: b"\xff" * n),
    ]
    for size in sizes:
        for pname, gen in patterns:
            sid = f"gh183:ssm:{pname}:{size}"
            blob = gen(size)
            try:
                r = Resp(port, timeout=60)
                r.call("SSM.PREFIX.STORE", sid, "0", blob)
                got = r.call("SSM.PREFIX.FETCH", sid, "0")
                r.close()
            except (RuntimeError, EOFError, OSError) as e:
                check(f"SSM {pname} {size}B", False, f"{type(e).__name__}: {e}")
                continue
            if got is None:
                check(f"SSM {pname} {size}B", False, "fetch returned nil")
                continue
            if len(got) != len(blob):
                check(f"SSM {pname} {size}B", False,
                      f"length {len(got)} != {len(blob)} (truncated)")
                continue
            check(f"SSM {pname} {size}B: byte-identical", got == blob,
                  f"sha256 {hashlib.sha256(got).hexdigest()[:16]}…")


def cold_tier(port, n_tokens, heads, head_dim):
    """gh #67 cold tier: KV.PREFIX.WARM rehydrates the Metal session cache from
    V-store. The claim tested here is narrow and checkable without a model:

      * WARM reports the layer count it rehydrated, and
      * WARM does NOT disturb the V-store source — the bytes fetched after a
        rehydrate are byte-identical to the bytes fetched before it.

    That second property is the one that matters for a shared cache: a warm
    path that silently rewrites or reorders the durable copy would be the
    LMCache #2535 shape (version roulette between tiers). Skips cleanly when
    the build has no Metal attention.
    """
    print("\n[6] Cold tier (gh #67) — WARM must not disturb the V-store source")
    kv_dim = heads * head_dim
    ns = "gh183:cold:ns"
    vals = make_vectors(n_tokens, kv_dim, seed=99)
    blob = pack_f32(vals)
    try:
        r = Resp(port, timeout=60)
        r.call("KV.PREFIX.REGISTER", ns, str(kv_dim), "fp16")
        # REGISTER derives <ns>_pk (keys) and <ns>_pv (values).
        for side in ("_pk", "_pv"):
            r.call("V.STOREBATCH", ns + side, "0", "0", str(n_tokens), blob)
        before = {}
        for side in ("_pk", "_pv"):
            got = r.call("V.FETCH", ns + side, "0", "RANGE", "0", str(n_tokens))
            before[side] = hashlib.sha256(got).hexdigest()
    except (RuntimeError, EOFError, OSError) as e:
        skip("cold tier: setup", f"{type(e).__name__}: {e}")
        return

    try:
        warm = r.call("KV.PREFIX.WARM", ns, str(heads), str(head_dim))
        check("KV.PREFIX.WARM reports rehydrated layers",
              isinstance(warm, str) and warm.split()[0].isdigit(),
              f"replied {warm!r}")
    except RuntimeError as e:
        skip("cold tier: WARM", f"unavailable ({e})")
        r.close()
        return

    for side in ("_pk", "_pv"):
        try:
            got = r.call("V.FETCH", ns + side, "0", "RANGE", "0", str(n_tokens))
            d = hashlib.sha256(got).hexdigest()
            check(f"cold tier: {side} byte-identical after WARM", d == before[side],
                  f"{d[:16]}… vs {before[side][:16]}…")
        except RuntimeError as e:
            check(f"cold tier: {side} after WARM", False, str(e))
    r.close()


def start_server(binary, port):
    if not binary or not os.path.exists(binary):
        return None
    p = subprocess.Popen(
        [binary, "-p", str(port), "-w", "1", "--kvcache",
         "--no-auto-detect", "--no-auto-embed"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    for _ in range(90):
        time.sleep(0.5)
        try:
            Resp(port, timeout=2).close()
            return p
        except OSError:
            continue
    return None


def stop_server(proc):
    if not proc:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", None),
                    help="start/stop our own server (needed for --restart)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        args.restart = True

    own = None
    if args.binary:
        # NOT `and not args.restart`. `--all` sets restart=True, so that guard
        # meant the documented invocation — `--binary ./pion-server --all` —
        # never started a server at all: the connect below then failed, the
        # `if not args.restart` guard swallowed the FATAL, and phase [1] ran
        # against nothing and died with "server connection lost".
        #
        # The restart phase needs `own` to be a live process anyway: it calls
        # stop_server(own) and then starts a fresh one. With own=None it had
        # nothing to stop.
        own = start_server(args.binary, args.port)
        if own is None:
            print(f"FATAL: could not start {args.binary} on {HOST}:{args.port}")
            return 2

    print(f"gh #183 KV round-trip evidence — {args.tokens} tokens x {args.dim} dim")
    print(f"input is seed-derived: make_vectors(n, dim, seed) is reproducible\n")

    try:
        Resp(args.port, timeout=3).close()
    except OSError as e:
        if not args.restart:
            print(f"FATAL: no --kvcache server on {HOST}:{args.port} ({e})")
            return 2

    if not args.restart or args.all:
        try:
            digests = tier_roundtrip(args.port, args.tokens, args.dim)
            determinism(args.port, args.tokens, args.dim, digests)
            concurrency(args.port, args.tokens, args.dim, digests, args.concurrency)
            # 4 MB crosses the >3 MB bulk-string boundary (gh #76 writev path).
            ssm_roundtrip(args.port, [1024, 64 * 1024, 4 * 1024 * 1024])
            cold_tier(args.port, 32, 8, 64)
        except OSError as e:
            print(f"FATAL: server connection lost ({e})")
            return 2

    if args.restart:
        if not args.binary:
            skip("cross-restart", "--restart needs --binary")
        else:
            stop_server(own)
            own = None
            cross_restart(args.binary, args.port, args.tokens, args.dim)

    stop_server(own)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    for f in FAILED:
        print(f"  FAILED:  {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
