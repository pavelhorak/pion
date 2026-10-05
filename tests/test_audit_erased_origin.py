#!/usr/bin/env python3
"""No local buffer may be read through a pointer that does not keep it alive:
tools/audit_erased_origin.py reports 0 (#40).

WHY
Mojo destroys a value right after its last use, and a call's arguments are
evaluated before the call runs. `Pointer[...](unsafe_from_address=
Int(x.unsafe_ptr()))` erases x's origin, so in

    f(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(x.unsafe_ptr())), len(x))

`len(x)` is x's last use and x is freed before f reads it. It shipped: WAL
replay restored stream consumers under garbage names, XDEL wrote its metadata
record from a freed buffer, and the same shape sat in the MoE manifest probes,
AI.KNN_LM.INFO, NEURON.PKM.INFO, COMMAND GETKEYSANDFLAGS and ACL LOG.

A 0 from an audit that cannot see the bug is worth nothing, so this first runs
it over a scratch tree holding each shipped shape (it must report every one)
and the safe shapes (it must report none), then over src/ (it must report 0).

    python3 tests/test_audit_erased_origin.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_erased_origin.py"

BAD = '''
def shape_inline(writer: ResponseWriter):
    var meta = encode_meta_rec(sd)
    _ = wal[].append_kv(45, kp, kl,
                        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(meta.unsafe_ptr())),
                        len(meta))


def shape_bound_branchy(cmd_id: UInt8, p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Bool:
    var gname = rec_bytes(p, n, at, ok)
    var gp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(gname.unsafe_ptr()))
    var g = sd[].group_index(gp, len(gname))
    if cmd_id == 38:
        if g < 0:
            sd[].groups.append(StreamGroup(gname^, 0, 0, 0))
        return True
    return False


def shape_pattern_loop(buf: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var pat = String("-of-")
    var pat_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
    var pat_len = pat.byte_length()
    var i = 0
    while i + pat_len < n:
        if buf[i] == pat_ptr[0]:
            return i
        i += 1
    return -1


def shape_span(writer: ResponseWriter):
    var info = String("count=") + String(3)
    var bytes = info.as_bytes()
    var info_ext = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bytes.unsafe_ptr()))
    writer.append_bulk_string_response(info_ext, len(bytes))


def shape_reassign(writer: ResponseWriter):
    var info = client_line()
    info = bytes_to_string(UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(info.unsafe_ptr())), 3)
    log(info^)


def shape_sibling(c: Bool):
    var f = flags()
    if c:
        _write_flags(writer, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.unsafe_ptr())), len(f))
    else:
        print(f)
'''
BAD_COUNT = 6

GOOD = '''
def kept_alive(writer: ResponseWriter):
    var f = flags()
    _write_flags(writer, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.unsafe_ptr())),
                 f.byte_length())
    _ = f^


def static_pattern(buf: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var pat: StaticString = "-of-"
    var pat_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
    return find(buf, n, pat_ptr, pat.byte_length())


def parameter(name: List[UInt8], mut writer: ResponseWriter):
    writer.append_bulk_string_response(
        Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(name.unsafe_ptr())), len(name))


def used_after(c: Bool):
    var pat = pl[k].name.copy()
    var pp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(pat.unsafe_ptr()))
    if glob(pp, len(pat)):
        send(pat)
    print(len(pat))


def origin_kept(mut writer: ResponseWriter):
    var info = String("x") + String(4)
    writer.append_bulk_string_response(info.unsafe_ptr(), info.byte_length())
'''


def run(src):
    r = subprocess.run([sys.executable, str(TOOL), "--src", str(src)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    fails = []
    with tempfile.TemporaryDirectory(prefix="erased_canary_") as d:
        bad, good = Path(d) / "bad", Path(d) / "good"
        bad.mkdir()
        good.mkdir()
        (bad / "canary.mojo").write_text(BAD)
        (good / "safe.mojo").write_text(GOOD)
        rc, out = run(bad)
        found = out.count("ERASED ORIGIN")
        print(f"[canary] planted {BAD_COUNT} shipped shapes, audit found {found}")
        if found != BAD_COUNT or rc == 0:
            fails.append("canary")
            print(out)
        rc, out = run(good)
        print(f"[safe]   safe shapes flagged: {out.count('ERASED ORIGIN')}")
        if rc != 0 or "ERASED ORIGIN" in out:
            fails.append("safe shapes")
            print(out)
    rc, out = run(ROOT / "src")
    print(f"[src]    {out.strip().splitlines()[-1]}")
    if rc != 0:
        fails.append("src")
        print(out)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
