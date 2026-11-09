#!/usr/bin/env python3
"""--maxmemory-policy: keys are evicted to hold RSS to --maxmemory.

Every reply text below was read off a live redis-server 8.10 first. Where the
behaviour differs by design, the section says so: the limit is on RSS, not on
allocator bytes, and Redis 8.10's allkeys-lrm / volatile-lrm do not exist here.

Every limit is set from the server's own idle RSS, because that number differs
by a factor between macOS and Linux.

  1. the surface: CONFIG, INFO, OBJECT, CLI refusals
  2. each allkeys policy, writing several times the limit: RSS holds, no write
     fails, and the policy shows in WHAT survived, checked by value
  3. the volatile policies: keys without a TTL are never evicted; -OOM when no
     candidate is left, while reads keep working; volatile-ttl goes by deadline
  4. a read-only load at the limit evicts nothing
  5. CONFIG SET: switching policy lifts -OOM; lowering the limit evicts now
  6. a crash and restart does not revive evicted keys, and keeps the others
  7. a replica applies its primary's evictions and does not evict by itself
  8. WATCH sees an eviction; a transaction over the limit runs
  9. every value type and long keys, survivors intact
 10. an evicted vector document leaves the index
 11. several workers and I/O threads

Usage: python3 tests/test_eviction.py [./pion-server]
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import wait_ready_pid, wait_port_free  # noqa: E402

BINARY = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 6474
OOM = "OOM command not allowed when used memory > 'maxmemory'."
FAIL = []
MB = 1 << 20


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


class Conn:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.f = self.s.makefile("rb")

    def send(self, *cmds):
        out = []
        for c in cmds:
            parts = [a if isinstance(a, bytes) else str(a).encode() for a in c]
            out.append(b"*%d\r\n" % len(parts))
            for p in parts:
                out.append(b"$%d\r\n%s\r\n" % (len(p), p))
        self.s.sendall(b"".join(out))

    def read(self):
        line = self.f.readline()
        t, b = line[:1], line[1:-2]
        if t == b"+":
            return b.decode()
        if t == b"-":
            return ("ERR", b.decode())
        if t == b":":
            return int(b)
        if t == b"$":
            n = int(b)
            return None if n < 0 else self.f.read(n + 2)[:-2]
        if t == b"*":
            n = int(b)
            return None if n < 0 else [self.read() for _ in range(n)]
        raise RuntimeError(f"bad reply {line!r}")

    def __call__(self, *a):
        self.send(a)
        return self.read()

    def many(self, cmds):
        """Pipelined; one reply per command, or the connection is desynced."""
        self.send(*cmds)
        return [self.read() for _ in cmds]


class Server:
    def __init__(self, *args, port=PORT, wal=False, work=None):
        self.port = port
        self.args = list(args)
        self.own = work is None
        self.d = work or tempfile.mkdtemp(prefix="pion_evict_")
        self.wal = wal
        self.start()

    def start(self):
        cmd = [BINARY, "-p", str(self.port), "--no-auto-detect", "--no-auto-embed"] + self.args
        if not self.wal:
            cmd.append("--no-wal")
        self.log = os.path.join(self.d, "out.log")
        self.p = subprocess.Popen(cmd, cwd=self.d, stdout=open(self.log, "a"), stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.p, 60)

    def conn(self):
        return Conn(self.port)

    def alive(self):
        return self.p.poll() is None

    def stop(self, sig_kill=True):
        if self.p.poll() is None:
            self.p.kill()
        self.p.wait()
        wait_port_free(self.port)
        if self.own and sig_kill:
            shutil.rmtree(self.d, ignore_errors=True)


def info(c, field):
    raw = c("INFO")
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    for line in text.split("\r\n"):
        if line.startswith(field + ":"):
            return line.split(":", 1)[1]
    return None


def rss(c):
    return int(info(c, "used_memory"))


def evicted(c):
    return int(info(c, "evicted_keys"))


def payload(key, size=200):
    """Derived from the key, so a survivor's value can be checked: a count alone
    cannot see a wrong value."""
    return (key.encode() * (size // len(key) + 1))[:size]


def fill(c, prefix, n, start=0, size=200, batch=2000, extra=None):
    """Write n keys. Returns the replies that were not +OK."""
    bad = []
    for i in range(start, start + n, batch):
        cmds = [("SET", f"{prefix}:{j}", payload(f"{prefix}:{j}", size))
                for j in range(i, min(i + batch, start + n))]
        if extra:
            cmds += extra(i)
        replies = c.many(cmds)
        bad += [r for r in replies[:batch] if r != "OK"]
    return bad


def section(t):
    print(f"\n[{t}]")


def surface():
    section("1. the surface")
    for bad in (["--maxmemory-policy", "foo"], ["--maxmemory-samples", "0"], ["--maxmemory-samples", "65"]):
        r = subprocess.run([BINARY, "-p", str(PORT), "--no-wal", *bad], capture_output=True, timeout=30,
                           cwd=tempfile.gettempdir())
        check(f"CLI refuses {' '.join(bad)}", r.returncode == 1, str(r.returncode))

    s = Server()
    c = s.conn()
    check("default policy is noeviction", c("CONFIG", "GET", "maxmemory-policy") == [b"maxmemory-policy", b"noeviction"])
    check("CONFIG GET maxmemory-samples", c("CONFIG", "GET", "maxmemory-samples") == [b"maxmemory-samples", b"5"])
    check("CONFIG GET lfu-*", sorted(c("CONFIG", "GET", "lfu-*")) == sorted([b"lfu-decay-time", b"1", b"lfu-log-factor", b"10"]))
    r = c("CONFIG", "SET", "maxmemory-policy", "foo")
    check("a bad policy is refused, as Redis words it",
          isinstance(r, tuple) and r[1].startswith(
              "ERR CONFIG SET failed (possibly related to argument 'maxmemory-policy') - argument(s) must be one of the following:"), repr(r))
    for name, val, want in (
            ("maxmemory-samples", "0", "argument must be between 1 and 64 inclusive"),
            ("maxmemory-samples", "65", "argument must be between 1 and 64 inclusive"),
            ("maxmemory-samples", "abc", "argument couldn't be parsed into an integer"),
            ("lfu-log-factor", "-1", "argument must be between 0 and 2147483647 inclusive"),
            ("lfu-decay-time", "-1", "argument must be between 0 and 2147483647 inclusive")):
        r = c("CONFIG", "SET", name, val)
        check(f"CONFIG SET {name} {val}",
              r == ("ERR", f"ERR CONFIG SET failed (possibly related to argument '{name}') - {want}"), repr(r))
    check("evicted_keys starts at 0", evicted(c) == 0)
    check("OBJECT FREQ under noeviction refuses",
          c("SET", "k", "v") == "OK" and isinstance(c("OBJECT", "FREQ", "k"), tuple))
    check("a policy name takes any case", c("CONFIG", "SET", "maxmemory-policy", "ALLKEYS-LFU") == "OK")
    check("CONFIG GET reads it back", c("CONFIG", "GET", "maxmemory-policy")[1] == b"allkeys-lfu")
    check("INFO reports it", info(c, "maxmemory_policy") == "allkeys-lfu")
    time.sleep(0.3)         # a worker takes a new policy at its next housekeeping
    c("PING")
    time.sleep(0.2)
    c("SET", "k2", "v")
    f = c("OBJECT", "FREQ", "k2")
    check("OBJECT FREQ under LFU counts", isinstance(f, int) and f >= 5, repr(f))
    r = c("OBJECT", "IDLETIME", "k2")
    check("OBJECT IDLETIME under LFU refuses, as Redis words it",
          r == ("ERR", "ERR An LFU maxmemory policy is selected, idle time not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust."), repr(r))
    check("OBJECT FREQ of a missing key is nil", c("OBJECT", "FREQ", "nope") is None)
    c("CONFIG", "SET", "maxmemory-policy", "allkeys-lru")
    time.sleep(0.3)
    c("PING")
    time.sleep(0.2)
    c("SET", "k3", "v")
    i = c("OBJECT", "IDLETIME", "k3")
    check("OBJECT IDLETIME under LRU counts", isinstance(i, int) and 0 <= i <= 2, repr(i))
    r = c("OBJECT", "FREQ", "k3")
    check("OBJECT FREQ under LRU refuses, as Redis words it",
          r == ("ERR", "ERR An LFU maxmemory policy is not selected, access frequency not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust."), repr(r))
    s.stop()


def allkeys(policy):
    section(f"2. {policy}, writing several times the limit")
    s = Server("--maxmemory-policy", policy)
    c = s.conn()
    base = rss(c)
    limit = base + 100 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    hot = [f"hot:{i}" for i in range(500)]
    fill(c, "hot", len(hot))
    n = 1_000_000
    peak_after_settle = 0
    bad = []
    batch = 2000
    for i in range(0, n, batch):
        # The hot set is read in a pipeline (a GET burst), half of it per batch
        # and the rest by the next one, as a cache's hot keys are.
        bad += fill(c, "k", batch, start=i, batch=batch)
        replies = c.many([("GET", hot[(i // batch + j * 2) % len(hot)]) for j in range(250)])
        bad += [r for r in replies if isinstance(r, tuple)]
        if i > n // 2:
            peak_after_settle = max(peak_after_settle, rss(c)) if i % (batch * 20) == 0 else peak_after_settle
    check("no write failed", not bad, repr(bad[:2]))
    ev = evicted(c)
    check("keys were evicted", ev > n // 4, str(ev))
    check("RSS holds near the limit", peak_after_settle <= limit * 1.20,
          f"peak {peak_after_settle >> 20} MB, limit {limit >> 20} MB, idle {base >> 20} MB")
    keys = c("DBSIZE")
    check("the keyspace is not emptied", keys > 20_000, str(keys))
    check("keys + evicted = keys written", keys + ev == n + len(hot), f"{keys} + {ev} != {n + len(hot)}")

    hot_alive = sum(1 for r in c.many([("EXISTS", h) for h in hot]) if r == 1)
    newest = [f"k:{n - 1 - i}" for i in range(0, 5000, 5)]
    oldest = [f"k:{i}" for i in range(0, 5000, 5)]
    new_alive = sum(1 for r in c.many([("EXISTS", k) for k in newest]) if r == 1)
    old_alive = sum(1 for r in c.many([("EXISTS", k) for k in oldest]) if r == 1)
    if policy == "allkeys-lru":
        check("LRU keeps the hot set", hot_alive >= 495, f"{hot_alive}/500")
        check("LRU keeps the newest keys", new_alive >= 950, f"{new_alive}/1000")
        check("LRU evicts the oldest keys", old_alive <= 20, f"{old_alive}/1000")
    elif policy == "allkeys-lfu":
        check("LFU keeps the hot set", hot_alive >= 450, f"{hot_alive}/500")
        check("LFU favours a key read often over one written once", hot_alive / 500 > old_alive / 1000 + 0.5,
              f"hot {hot_alive}/500, oldest {old_alive}/1000")
    else:
        # Random has no idea what is hot: survival is the keyspace's share.
        check("random does not keep the hot set", hot_alive <= 400, f"{hot_alive}/500")

    # By value: the WAL/LRU bookkeeping must never serve another key's bytes.
    ks = [f"k:{i}" for i in range(n - 2000, n)]
    vals = c.many([("GET", k) for k in ks])
    wrong = [k for k, v in zip(ks, vals) if v is not None and v != payload(k)]
    check("survivors hold their own values", not wrong, repr(wrong[:3]))
    check("server alive", s.alive())
    r = c("CONFIG", "RESETSTAT")
    check("CONFIG RESETSTAT clears evicted_keys", r == "OK" and evicted(c) == 0 or evicted(c) < 100)
    s.stop()


def volatile():
    section("3. volatile policies")
    for policy in ("volatile-lru", "volatile-lfu", "volatile-random"):
        s = Server("--maxmemory-policy", policy)
        c = s.conn()
        limit = rss(c) + 30 * MB
        c("CONFIG", "SET", "maxmemory", limit)
        # Persistent keys first, then keep writing keys with a TTL.
        fill(c, "keep", 20_000)
        bad = []
        for i in range(0, 600_000, 2000):
            cmds = [("SET", f"t:{j}", payload(f"t:{j}"), "EX", 3600) for j in range(i, i + 2000)]
            r = c.many(cmds)
            bad += [x for x in r if x != "OK"]
            if bad:
                break
        check(f"{policy}: no write failed", not bad, repr(bad[:1]))
        check(f"{policy}: keys were evicted", evicted(c) > 0, str(evicted(c)))
        alive = sum(1 for r in c.many([("EXISTS", f"keep:{i}") for i in range(20_000)]) if r == 1)
        check(f"{policy}: a key without a TTL is never evicted", alive == 20_000, f"{alive}/20000")
        # Both counts from one INFO line. Eviction goes on for a moment after
        # the writes stop (several hundred keys between two commands), so a
        # DBSIZE and a later INFO could see different keyspaces.
        db0 = info(c, "db0")
        ev = evicted(c)
        keys = int(db0.split("keys=")[1].split(",")[0]) if db0 else -1
        n_exp = int(db0.split("expires=")[1].split(",")[0]) if db0 else -1
        check(f"{policy}: evicting a key drops its TTL entry", n_exp == keys - 20_000,
              f"expires={n_exp} keys={keys} evicted={ev}")
        s.stop()

    s = Server("--maxmemory-policy", "volatile-ttl")
    c = s.conn()
    limit = rss(c) + 30 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    # The first keys written expire last: a policy that ignored deadlines (or
    # that went by age) would evict them first.
    for i in range(0, 400_000, 2000):
        ttl = 90_000 if i < 10_000 else 3600
        r = c.many([("SET", f"t:{j}", payload(f"t:{j}"), "EX", ttl) for j in range(i, i + 2000)])
        assert all(x == "OK" for x in r), r[:2]
    far = sum(1 for r in c.many([("EXISTS", f"t:{i}") for i in range(0, 10_000, 10)]) if r == 1)
    near = sum(1 for r in c.many([("EXISTS", f"t:{i}") for i in range(10_000, 50_000, 40)]) if r == 1)
    check("volatile-ttl keeps the far deadlines", far >= 900, f"far {far}/1000")
    check("volatile-ttl evicts the near deadlines", near <= 500, f"near {near}/1000")
    s.stop()

    # -OOM when nothing is left, and only then.
    s = Server("--maxmemory-policy", "volatile-lru")
    c = s.conn()
    limit = rss(c) + 20 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    refused = None
    for i in range(0, 1_000_000, 2000):
        r = c.many([("SET", f"p:{j}", payload(f"p:{j}", 1000)) for j in range(i, i + 2000)])
        oom = [x for x in r if isinstance(x, tuple)]
        if oom:
            refused = oom[0]
            break
    check("volatile, no TTL anywhere: -OOM, in Redis's words", refused == ("ERR", OOM), repr(refused))
    check("nothing was evicted", evicted(c) == 0, str(evicted(c)))
    check("reads still work", c("GET", "p:0") == payload("p:0", 1000))
    check("DEL still works", c("DEL", "p:0") == 1)
    s.stop()


def read_only():
    section("4. a read-only load at the limit")
    s = Server("--maxmemory-policy", "allkeys-lru")
    c = s.conn()
    limit = rss(c) + 30 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    fill(c, "k", 400_000)
    time.sleep(0.5)
    before_ev, before_keys = evicted(c), c("DBSIZE")
    check("over the limit, keys were evicted", before_ev > 0, str(before_ev))
    t0 = time.time()
    i = 0
    while time.time() - t0 < 2.0:
        c.many([("GET", f"k:{(i + j) * 7 % 400_000}") for j in range(500)])
        i += 500
    after_ev, after_keys = evicted(c), c("DBSIZE")
    # Freed memory stays with the allocator, so RSS may sit over the limit with
    # nothing being written. Evicting on that alone would empty the keyspace.
    check("reads evict (almost) nothing", after_ev - before_ev <= before_keys // 50,
          f"{after_ev - before_ev} evicted of {before_keys} during {i} GETs")
    check("the keyspace stays", after_keys >= before_keys * 0.98, f"{before_keys} -> {after_keys}")
    s.stop()


def live_config():
    section("5. CONFIG SET at runtime")
    s = Server()
    c = s.conn()
    limit = rss(c) + 20 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    refused = None
    for i in range(0, 1_000_000, 2000):
        r = c.many([("SET", f"k:{j}", payload(f"k:{j}", 1000)) for j in range(i, i + 2000)])
        oom = [x for x in r if isinstance(x, tuple)]
        if oom:
            refused = oom[0]
            break
    check("noeviction refuses", refused == ("ERR", OOM), repr(refused))
    check("noeviction evicts nothing", evicted(c) == 0)
    c("CONFIG", "SET", "maxmemory-policy", "allkeys-lru")
    deadline = time.time() + 5
    ok = False
    while time.time() < deadline and not ok:
        ok = c("SET", "after", "v") == "OK"
        time.sleep(0.02)
    check("a policy lifts -OOM", ok)
    bad = fill(c, "m", 100_000, size=1000)
    check("writes keep working, evicting", not bad and evicted(c) > 0, f"{bad[:1]} ev={evicted(c)}")

    keys = c("DBSIZE")
    r0 = rss(c)
    lower = r0 - (r0 - rss_floor(s)) // 2
    c("CONFIG", "SET", "maxmemory", lower)
    deadline = time.time() + 5
    while time.time() < deadline and c("DBSIZE") > keys * 0.8:
        time.sleep(0.05)
    k2 = c("DBSIZE")
    check("lowering the limit evicts without a write", k2 < keys * 0.9, f"{keys} -> {k2}")
    check("...and not everything", k2 > 0, str(k2))
    c("CONFIG", "SET", "maxmemory-policy", "noeviction")
    time.sleep(0.3)
    c("PING")
    time.sleep(0.2)
    e0 = evicted(c)
    c("SET", "x", "y")
    time.sleep(0.2)
    check("back to noeviction: eviction stops", evicted(c) == e0)
    check("server alive", s.alive())
    s.stop()


def rss_floor(s):
    """About what the process holds with no data: a fresh server's RSS."""
    t = Server(port=PORT + 1)
    v = rss(t.conn())
    t.stop()
    return v


def durability():
    section("6. a crash and restart")
    s = Server("--maxmemory-policy", "allkeys-lru", wal=True)
    c = s.conn()
    limit = rss(c) + 40 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    n = 500_000
    bad = fill(c, "k", n)
    check("no write failed", not bad, repr(bad[:1]))
    ev = evicted(c)
    check("keys were evicted", ev > 0, str(ev))
    c("SET", "last", "v")       # queued DELs reach the log ahead of the next record
    time.sleep(0.3)
    c("SET", "last2", "v")
    time.sleep(0.2)
    alive = c.many([("EXISTS", f"k:{i}") for i in range(0, n, 50)])
    was_alive = {i * 50 for i, r in enumerate(alive) if r == 1}
    check("some keys survive and some do not", 0 < len(was_alive) < len(alive), str(len(was_alive)))
    d = s.d
    s.p.kill()
    s.p.wait()
    wait_port_free(PORT)
    s2 = Server("--maxmemory-policy", "allkeys-lru", wal=True, work=d)
    c = s2.conn()
    now = c.many([("EXISTS", f"k:{i}") for i in range(0, n, 50)])
    now_alive = {i * 50 for i, r in enumerate(now) if r == 1}
    revived = now_alive - was_alive
    lost = was_alive - now_alive
    check("evicted keys stay gone", len(revived) <= len(alive) // 100, f"{len(revived)} revived of {len(alive)}")
    check("the other keys are still there", len(lost) <= len(alive) // 100, f"{len(lost)} lost of {len(alive)}")
    sample = sorted(now_alive)[:200]
    vals = c.many([("GET", f"k:{i}") for i in sample])
    wrong = [i for i, v in zip(sample, vals) if v != payload(f"k:{i}")]
    check("their values came back", not wrong, repr(wrong[:3]))
    s2.stop(False)
    shutil.rmtree(d, ignore_errors=True)


def replica():
    section("7. a replica")
    pa, pb = PORT + 10, PORT + 20
    d = tempfile.mkdtemp(prefix="pion_evict_repl_")
    common = [BINARY, "--no-auto-detect", "--no-auto-embed", "--no-crash-log", "-w", "1", "--cluster", "--cluster-host", "127.0.0.1"]
    da, db = os.path.join(d, "a"), os.path.join(d, "b")
    os.makedirs(da)
    os.makedirs(db)
    pp = subprocess.Popen(common + ["-p", str(pa), "--maxmemory-policy", "allkeys-lru"], cwd=da,
                          stdout=open(os.path.join(da, "out.log"), "w"), stderr=subprocess.STDOUT)
    wait_ready_pid(pa, pp, 60)
    rp = subprocess.Popen(common + ["-p", str(pb), "--maxmemory-policy", "allkeys-lru", "--cluster-replica",
                                    "--cluster-primary-host", "127.0.0.1", "--cluster-primary-port", str(pa)],
                          cwd=db, stdout=open(os.path.join(db, "out.log"), "w"), stderr=subprocess.STDOUT)
    wait_ready_pid(pb, rp, 60)
    try:
        ca, cb = Conn(pa), Conn(pb)
        base = rss(ca)
        ca("CONFIG", "SET", "maxmemory", base + 30 * MB)
        # A limit the replica is well over: it must follow its primary, not evict.
        cb("CONFIG", "SET", "maxmemory", rss(cb) + 1 * MB)
        wrote = 0
        slots_ok = True
        for i in range(0, 300_000, 2000):
            r = ca.many([("SET", f"{{a}}k:{j}", payload(f"k:{j}")) for j in range(i, i + 2000)])
            if any(x != "OK" for x in r):
                # a key this node does not own would answer MOVED
                slots_ok = False
                print("   first non-OK:", [x for x in r if x != "OK"][0])
                break
            wrote += 2000
        check("primary took the writes", slots_ok and evicted(ca) > 0, f"wrote {wrote} ev={evicted(ca)}")
        deadline = time.time() + 20
        ka = kb = -1
        while time.time() < deadline:
            ka, kb = ca("DBSIZE"), cb("DBSIZE")
            if ka == kb:
                break
            time.sleep(0.25)
        check("the replica converges on the primary's keys", ka == kb, f"primary {ka}, replica {kb}")
        check("the replica evicted nothing itself", evicted(cb) == 0, str(evicted(cb)))
    finally:
        for p in (pp, rp):
            if p.poll() is None:
                p.kill()
            p.wait()
        wait_port_free(pa)
        wait_port_free(pb)
        shutil.rmtree(d, ignore_errors=True)


def transactions():
    section("8. transactions")
    s = Server("--maxmemory-policy", "allkeys-random")
    c = s.conn()
    c2 = s.conn()
    limit = rss(c) + 30 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    fill(c, "w", 50_000)
    check("WATCH", c("WATCH", "w:1") == "OK")
    # Evict w:1 from another connection: the limit falls until it is gone.
    c2("CONFIG", "SET", "maxmemory", 1)
    deadline = time.time() + 10
    while time.time() < deadline and c2("EXISTS", "w:1") == 1:
        time.sleep(0.02)
    check("the watched key was evicted", c2("EXISTS", "w:1") == 0)
    # Nothing is left to evict at a 1-byte limit, so a write would be refused
    # when queued, as Redis does. Lift it: this is about what WATCH saw.
    c2("CONFIG", "SET", "maxmemory", 0)
    c("MULTI")
    c("SET", "w:1", "v")
    r = c("EXEC")
    check("EXEC aborts: an eviction is a change of the key", r is None, repr(r))
    s.stop()

    s = Server("--maxmemory-policy", "allkeys-lru")
    c = s.conn()
    limit = rss(c) + 30 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    fill(c, "w", 400_000)
    check("over the limit", evicted(c) > 0)
    r = c.many([("MULTI",), ("SET", "tx", "1"), ("INCR", "txn"), ("EXEC",)])
    check("a transaction runs over the limit, as eviction makes room", r[3] == ["OK", 1], repr(r))
    r = c("EVAL", "return redis.call('SET', KEYS[1], 'v')", 1, "lua")
    check("a script's write runs too", r == "OK", repr(r))
    s.stop()


def value_types():
    section("9. every value type, long keys")
    s = Server("--maxmemory-policy", "allkeys-lru")
    c = s.conn()
    limit = rss(c) + 25 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    long_tail = "x" * 40
    errs = []
    for i in range(30_000):
        k = f"{i}:{long_tail}"
        cmds = [("SET", f"s:{k}", payload(k)),
                ("RPUSH", f"l:{k}", "a", "b", payload(k, 100)),
                ("HSET", f"h:{k}", "f", payload(k, 100), "g", "v"),
                ("SADD", f"S:{k}", "a", "b", payload(k, 100)),
                ("ZADD", f"z:{k}", 1, "a", 2, payload(k, 100)),
                ("XADD", f"x:{k}", "*", "f", payload(k, 100)),
                ("PFADD", f"p:{k}", "a", "b"),
                ("SETBIT", f"b:{k}", 100, 1),
                ("GEOADD", f"g:{k}", 13.36, 38.11, "m"),
                ("VADD", f"v:{k}", "VALUES", 3, 1, 0, 0, "m")]
        replies = c.many(cmds)
        errs += [(x, y[0]) for x, y in zip(replies, cmds) if isinstance(x, tuple)]
        if i % 2000 == 0 and not s.alive():
            break
    check("no write of any type failed", not errs, repr(errs[:3]))
    check("evicted a lot of them", evicted(c) > 10_000, str(evicted(c)))
    check("server alive", s.alive())
    wrong = []
    seen = 0
    for i in range(29_000, 30_000):
        k = f"{i}:{long_tail}"
        replies = c.many([("GET", f"s:{k}"), ("LRANGE", f"l:{k}", 0, -1), ("HGET", f"h:{k}", "f"),
                          ("SISMEMBER", f"S:{k}", payload(k, 100)), ("ZSCORE", f"z:{k}", payload(k, 100))])
        if replies[0] is not None:
            seen += 1
            if replies[0] != payload(k):
                wrong.append(("s", k))
        if replies[1] and replies[1] != [b"a", b"b", payload(k, 100)]:
            wrong.append(("l", k))
        if replies[2] is not None and replies[2] != payload(k, 100):
            wrong.append(("h", k))
    check("newest survivors exist", seen >= 50, f"{seen}/1000 (dbsize {c('DBSIZE')})")
    check("survivors hold their own values", not wrong, repr(wrong[:3]))
    s.stop()


def vector_docs():
    section("10. a vector document")
    s = Server("--maxmemory-policy", "allkeys-random")
    c = s.conn()
    r = c("FT.CREATE", "idx", "ON", "HASH", "PREFIX", 1, "doc:", "SCHEMA",
          "v", "VECTOR", "HNSW", 6, "TYPE", "FLOAT32", "DIM", 4, "DISTANCE_METRIC", "L2")
    check("FT.CREATE", r == "OK", repr(r))
    import struct
    for i in range(0, 3000, 500):
        c.many([("HSET", f"doc:{j}", "v", struct.pack("<4f", j, 0, 0, 1)) for j in range(i, i + 500)])
    time.sleep(0.5)
    c("FT.OPTIMIZE", "idx")
    limit = rss(c) + 20 * MB
    c("CONFIG", "SET", "maxmemory", limit)
    fill(c, "k", 300_000)
    check("evicted", evicted(c) > 0)
    alive = c.many([("EXISTS", f"doc:{j}") for j in range(3000)])
    gone = sum(1 for r in alive if r == 0)
    check("documents were among them", gone > 0, str(gone))
    q = struct.pack("<4f", 0, 0, 0, 1)
    res = c("FT.SEARCH", "idx", "*=>[KNN 200 @v $q]", "PARAMS", 2, "q", q, "DIALECT", 2, "RETURN", 0)
    bad = [k for k in (res[1:] if isinstance(res, list) else [])
           if isinstance(k, bytes) and k.startswith(b"doc:") and alive[int(k[4:])] == 0]
    check("FT.SEARCH answers", isinstance(res, list), repr(res)[:80])
    check("no evicted document in the results", not bad, repr(bad[:3]))
    check("server alive", s.alive())
    s.stop()


def workers():
    section("11. several workers and I/O threads")
    for args in (["-w", "4", "--independent-workers"], ["-w", "1", "--io-threads", "3"]):
        name = " ".join(args)
        s = Server("--maxmemory-policy", "allkeys-lru", *args)
        c = s.conn()
        limit = rss(c) + 40 * MB
        c("CONFIG", "SET", "maxmemory", limit)
        conns = [s.conn() for _ in range(4)]
        n = 400_000
        bad = []
        peak = 0
        for i in range(0, n, 2000):
            cc = conns[(i // 2000) % 4]
            bad += fill(cc, "k", 2000, start=i)
            if i > n // 2 and i % 40_000 == 0:
                peak = max(peak, rss(c))
        check(f"{name}: no write failed", not bad, repr(bad[:1]))
        check(f"{name}: RSS holds near the limit", peak <= limit * 1.15, f"peak {peak >> 20} MB, limit {limit >> 20} MB")
        check(f"{name}: server alive", s.alive())
        s.stop()


def main():
    surface()
    for p in ("allkeys-lru", "allkeys-lfu", "allkeys-random"):
        allkeys(p)
    volatile()
    read_only()
    live_config()
    durability()
    replica()
    transactions()
    value_types()
    vector_docs()
    workers()
    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)}")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
