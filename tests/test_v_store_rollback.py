#!/usr/bin/env python3
"""A10 (gh #37 / #41): V.SNAPSHOT / V.RESTORE / V.COMMIT — spec-decode rollback.

Speculative-decoding consumers (dflash DDTree, EAGLE-3, REST) write candidate
K/V before verify, then roll back rejected branches in-place. V-store gets
per-session snapshots that record per-layer length and a restore that
truncates back to it. Snapshots also capture v_scale / v_min for INT8
correctness across multi-batch sessions.

Requires: ./pion-server --kvcache -w 1
"""

import socket
import sys
import numpy as np

HOST = "127.0.0.1"
PORT = 1974


class C:
    def __init__(self):
        self.s = socket.socket()
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.s.connect((HOST, PORT))
        self.s.settimeout(15)
        self.f = self.s.makefile("rb")

    def _enc(self, parts):
        out = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, bytes):
                out.append(f"${len(p)}\r\n".encode() + p + b"\r\n")
            else:
                s = str(p)
                out.append(f"${len(s)}\r\n{s}\r\n".encode())
        return b"".join(out)

    def call(self, *p):
        self.s.sendall(self._enc(p))
        line = self.f.readline().rstrip(b"\r\n")
        t, body = line[:1], line[1:]
        if t in (b"+", b"-", b":"):
            return line
        if t == b"$":
            n = int(body)
            if n < 0:
                return None
            data = self.f.read(n)
            assert self.f.read(2) == b"\r\n"
            return data
        raise ValueError(f"unexpected reply: {line!r}")


def _snap_id(reply):
    assert reply.startswith(b":"), f"expected :<n>, got {reply!r}"
    return int(reply[1:])


def test_snapshot_returns_positive_id():
    c = C()
    sid = "rb_basic"
    c.call("V.CREATE", sid, "128")
    snap1 = _snap_id(c.call("V.SNAPSHOT", sid))
    snap2 = _snap_id(c.call("V.SNAPSHOT", sid))
    assert snap1 > 0
    assert snap2 > snap1, "snap_id must be monotonic"
    print(f"PASS: test_snapshot_returns_positive_id (ids: {snap1}, {snap2})")


def test_snapshot_then_restore_truncates_layer_length():
    """Write 100, snapshot, write 50 more, restore → length back to 100.
    Uses turbo4 (per-block scale embedded) for clean restore semantics."""
    c = C()
    sid = "rb_truncate"
    dim = 128
    c.call("V.CREATE", sid, str(dim), "VQUANT", "turbo4")

    np.random.seed(101)
    vals_a = (np.random.randn(100, dim) * 0.3).astype(np.float32)
    assert c.call("V.STOREBATCH", sid, "0", "0", "100", vals_a.tobytes()) == b"+OK"

    snap = _snap_id(c.call("V.SNAPSHOT", sid))

    vals_b = (np.random.randn(50, dim) * 0.3).astype(np.float32)
    assert c.call("V.STOREBATCH", sid, "0", "100", "50", vals_b.tobytes()) == b"+OK"

    info = c.call("V.INFO", sid).decode()
    assert "layer_0_tokens:150" in info, f"pre-restore: {info}"

    assert c.call("V.RESTORE", sid, str(snap)) == b"+OK"

    info = c.call("V.INFO", sid).decode()
    assert "layer_0_tokens:100" in info, f"post-restore: {info}"

    r = c.call("V.FETCH", sid, "0", "0", "50", "99")
    ret = np.frombuffer(r[:3 * dim * 4], dtype=np.float32).reshape(3, dim)
    for i, tid in enumerate([0, 50, 99]):
        cos = np.dot(vals_a[tid], ret[i]) / (np.linalg.norm(vals_a[tid]) * np.linalg.norm(ret[i]))
        assert cos > 0.85, f"post-restore turbo4 token {tid}: cos={cos:.4f}"

    r = c.call("V.FETCH", sid, "0", "100", "120")
    assert r is None, f"all-out-of-range fetch should return nil"
    print("PASS: test_snapshot_then_restore_truncates_layer_length")


def test_restore_then_overwrite_works():
    """After restore, new STOREBATCH at the truncated position writes fresh
    values."""
    c = C()
    sid = "rb_overwrite"
    dim = 128
    c.call("V.CREATE", sid, str(dim), "VQUANT", "turbo4")

    np.random.seed(202)
    vals_committed = (np.random.randn(50, dim) * 0.3).astype(np.float32)
    assert c.call("V.STOREBATCH", sid, "0", "0", "50", vals_committed.tobytes()) == b"+OK"
    snap = _snap_id(c.call("V.SNAPSHOT", sid))

    vals_reject = (np.random.randn(30, dim) * 1.0).astype(np.float32)
    c.call("V.STOREBATCH", sid, "0", "50", "30", vals_reject.tobytes())
    c.call("V.RESTORE", sid, str(snap))

    vals_accept = (np.random.randn(20, dim) * 0.3).astype(np.float32)
    assert c.call("V.STOREBATCH", sid, "0", "50", "20", vals_accept.tobytes()) == b"+OK"

    r = c.call("V.FETCH", sid, "0", "0", "25", "49")
    ret = np.frombuffer(r[:3 * dim * 4], dtype=np.float32).reshape(3, dim)
    for i, tid in enumerate([0, 25, 49]):
        cos = np.dot(vals_committed[tid], ret[i]) / (np.linalg.norm(vals_committed[tid]) * np.linalg.norm(ret[i]))
        assert cos > 0.85, f"committed prefix turbo4 tok {tid}: cos={cos:.4f}"

    r = c.call("V.FETCH", sid, "0", "50", "60", "69")
    ret = np.frombuffer(r[:3 * dim * 4], dtype=np.float32).reshape(3, dim)
    for i, tid_offset, tid in [(0, 0, 50), (1, 10, 60), (2, 19, 69)]:
        accept_v = vals_accept[tid_offset]
        reject_v = vals_reject[tid - 50] if tid - 50 < 30 else None
        cos_accept = np.dot(accept_v, ret[i]) / (np.linalg.norm(accept_v) * np.linalg.norm(ret[i]))
        assert cos_accept > 0.85, f"accept turbo4 tok {tid}: cos vs accept {cos_accept:.4f}"
        if reject_v is not None:
            cos_reject = np.dot(reject_v, ret[i]) / (np.linalg.norm(reject_v) * np.linalg.norm(ret[i]))
            assert cos_reject < 0.5, f"accept tok {tid}: leaks reject (cos={cos_reject:.4f})"
    print("PASS: test_restore_then_overwrite_works")


def test_int8_multi_batch_with_snapshot_restore():
    """A10 (#41 fix): INT8 sessions with multi-batch write + restore. Without
    snap_scales/snap_mins capture, the prefix dequant would corrupt because
    the second STOREBATCH rescales v_scale globally. With the fix, prefix
    cosine recovers."""
    c = C()
    sid = "rb_int8_multibatch"
    dim = 64
    c.call("V.CREATE", sid, str(dim))  # default int8

    np.random.seed(401)
    vals_a = (np.random.randn(100, dim) * 0.3).astype(np.float32)
    assert c.call("V.STOREBATCH", sid, "0", "0", "100", vals_a.tobytes()) == b"+OK"
    snap = _snap_id(c.call("V.SNAPSHOT", sid))

    # Second batch with very different distribution — would rescale INT8 range
    vals_b = (np.random.randn(50, dim) * 5.0).astype(np.float32)
    c.call("V.STOREBATCH", sid, "0", "100", "50", vals_b.tobytes())

    # Restore — scale + min must roll back to the first-batch range
    assert c.call("V.RESTORE", sid, str(snap)) == b"+OK"

    # Prefix dequant should match original to within INT8 precision
    r = c.call("V.FETCH", sid, "0", "0", "50", "99")
    ret = np.frombuffer(r[:3 * dim * 4], dtype=np.float32).reshape(3, dim)
    for i, tid in enumerate([0, 50, 99]):
        cos = np.dot(vals_a[tid], ret[i]) / (np.linalg.norm(vals_a[tid]) * np.linalg.norm(ret[i]))
        assert cos > 0.99, f"INT8 prefix tok {tid} after rejected-branch rescale: cos={cos:.4f}"
    print("PASS: test_int8_multi_batch_with_snapshot_restore")


def test_commit_releases_slot():
    c = C()
    sid = "rb_commit"
    c.call("V.CREATE", sid, "64")
    snap = _snap_id(c.call("V.SNAPSHOT", sid))
    assert c.call("V.COMMIT", sid, str(snap)) == b":1"
    assert c.call("V.COMMIT", sid, str(snap)) == b":0", "second commit should be idempotent :0"
    r = c.call("V.RESTORE", sid, str(snap))
    assert r.startswith(b"-ERR") and b"unknown snap_id" in r, r
    print("PASS: test_commit_releases_slot")


def test_max_concurrent_snapshots():
    c = C()
    sid = "rb_max"
    c.call("V.CREATE", sid, "64")
    snaps = [_snap_id(c.call("V.SNAPSHOT", sid)) for _ in range(16)]
    r = c.call("V.SNAPSHOT", sid)
    assert r.startswith(b"-ERR") and b"no free snapshot slot" in r, f"17th: {r!r}"
    c.call("V.COMMIT", sid, str(snaps[0]))
    snap_new = _snap_id(c.call("V.SNAPSHOT", sid))
    assert snap_new > snaps[-1], "new snap_id stays monotonic after commit"
    print("PASS: test_max_concurrent_snapshots")


def test_nested_snapshots_independent():
    c = C()
    sid = "rb_nested"
    dim = 64
    c.call("V.CREATE", sid, str(dim))

    np.random.seed(303)
    v1 = (np.random.randn(20, dim) * 0.3).astype(np.float32)
    c.call("V.STOREBATCH", sid, "0", "0", "20", v1.tobytes())
    outer = _snap_id(c.call("V.SNAPSHOT", sid))

    v2 = (np.random.randn(30, dim) * 0.3).astype(np.float32)
    c.call("V.STOREBATCH", sid, "0", "20", "30", v2.tobytes())
    inner = _snap_id(c.call("V.SNAPSHOT", sid))

    v3 = (np.random.randn(15, dim) * 0.3).astype(np.float32)
    c.call("V.STOREBATCH", sid, "0", "50", "15", v3.tobytes())

    c.call("V.RESTORE", sid, str(inner))
    info = c.call("V.INFO", sid).decode()
    assert "layer_0_tokens:50" in info, info

    c.call("V.RESTORE", sid, str(outer))
    info = c.call("V.INFO", sid).decode()
    assert "layer_0_tokens:20" in info, info

    # inner restore after outer is a no-op (snap_len > current)
    c.call("V.RESTORE", sid, str(inner))
    info = c.call("V.INFO", sid).decode()
    assert "layer_0_tokens:20" in info, f"inner restore after outer should NOT grow: {info}"
    print("PASS: test_nested_snapshots_independent")


def test_restore_unknown_snap_id():
    c = C()
    sid = "rb_unk"
    c.call("V.CREATE", sid, "64")
    r = c.call("V.RESTORE", sid, "999999")
    assert r.startswith(b"-ERR") and b"unknown snap_id" in r, r
    r = c.call("V.RESTORE", sid, "0")
    assert r.startswith(b"-ERR"), r
    print("PASS: test_restore_unknown_snap_id")


def test_unknown_session():
    c = C()
    r = c.call("V.RESTORE", "nope_xyz", "1")
    assert r.startswith(b"-ERR") and b"session not found" in r, r
    r = c.call("V.SNAPSHOT", "nope_xyz")
    assert r.startswith(b"-ERR") and b"session not found" in r, r
    r = c.call("V.COMMIT", "nope_xyz", "1")
    assert r.startswith(b"-ERR") and b"session not found" in r, r
    print("PASS: test_unknown_session")


if __name__ == "__main__":
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect((HOST, PORT))
        s.close()
    except Exception:
        print(f"Pion not running at {HOST}:{PORT}")
        print("Start with: ./pion-server --kvcache -w 1")
        sys.exit(1)

    print("=" * 60)
    print("V-store rollback tests (issue #37 / A10)")
    print("=" * 60)

    tests = [
        test_snapshot_returns_positive_id,
        test_snapshot_then_restore_truncates_layer_length,
        test_restore_then_overwrite_works,
        test_int8_multi_batch_with_snapshot_restore,
        test_commit_releases_slot,
        test_max_concurrent_snapshots,
        test_nested_snapshots_independent,
        test_restore_unknown_snap_id,
        test_unknown_session,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
