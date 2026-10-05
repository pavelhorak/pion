#!/usr/bin/env python3
"""#40: stream consumer groups, XSETID and Redis 7's stream metadata.

Part 1 needs no Redis:
  [1] a work queue end to end: XGROUP CREATE, XREADGROUP, XPENDING (both
      forms), XACK, XCLAIM, XAUTOCLAIM, DELCONSUMER, and XINFO's view of it;
  [2] blocking XREADGROUP: woken by XADD, its timeout, a destroyed group, a
      deleted or retyped key, NOACK, a history read and MULTI never block;
  [3] a large pending list: acknowledging it in order, and deleting a
      consumer that owns most of it, stay linear (an acknowledgement used to
      shift every later pending entry);
  [4] durability: every group, consumer, pending entry and the stream's
      metadata survive WAL replay after SIGKILL, SAVE, BGREWRITEAOF,
      DUMP/RESTORE and COPY, as XINFO STREAM FULL shows them, times included;
  [5] a replica: one that joins after the groups exist (FULLRESYNC) and one
      that follows them live (the WAL stream) both read what the primary does.
  Redis 8.2's group-reference strategies (KEEPREF / DELREF / ACKED on XADD
  and XTRIM trimming, XDELEX, XACKDEL) are in [4]-[6] too.

Part 2 sends the same commands to Pion and to a live redis-server (RESP2 and
RESP3) and requires the same replies, except where Pion differs on purpose:
XINFO STREAM leaves out Redis's radix-tree counts and the fields of Redis 8
features Pion lacks (idempotent XADD, XNACK), and idle times are compared
within a few milliseconds. Skipped when redis-server is not on PATH.

    python3 tests/test_stream_groups.py [--binary ./pion-server] [-v]
"""
import argparse
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, encode, wait_ready_pid, wait_port_free  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []
VERBOSE = False


def check(name, ok, detail=""):
    if ok:
        if VERBOSE:
            print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}  {detail}"[:900])


def free_port_block(n=4, repl=False):
    """A port p with p .. p+n-1 free (Pion also binds p+1 and p+2+worker);
    with `repl`, p+10000 .. p+10000+n-1 too (a cluster node's replication
    port is its port + 10000, so p must stay at or below 55535)."""
    for _ in range(400):
        if repl:
            # not an OS-assigned port: macOS hands those out in sequence
            # from 49152 up, so they soon all sit above 55535
            p = random.randrange(20000, 45000)
        else:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
            s.close()
        if p + n > 65000 or (repl and p + n + 10000 > 65535):
            continue
        ok = True
        for q in list(range(p, p + n)) + (list(range(p + 10000, p + 10000 + n)) if repl else []):
            t = socket.socket()
            try:
                t.bind(("127.0.0.1", q))
            except OSError:
                ok = False
            finally:
                t.close()
        if ok:
            return p
    raise SystemExit("no free port block")


class Server:
    def __init__(self, binary, extra=(), workdir=None, port=None):
        self.binary, self.extra = binary, list(extra)
        self.port = port or free_port_block()
        self.own_dir = workdir is None
        self.dir = workdir or tempfile.mkdtemp(prefix="groups40_")
        self.proc = None
        self.start()

    def start(self):
        self.log = open(os.path.join(self.dir, "log"), "a")
        self.proc = subprocess.Popen([self.binary, "-p", str(self.port), "-w", "1", "--no-crash-log",
                                      "--no-auto-detect", "--no-auto-embed", *self.extra],
                                     cwd=self.dir, stdout=self.log, stderr=subprocess.STDOUT)
        wait_ready_pid(self.port, self.proc, 60)

    def conn(self, protocol=2):
        c = Conn(self.port, timeout=30)
        if protocol == 3:
            c.cmd("HELLO", "3")
        return c

    def kill9(self):
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()
        self.log.close()
        wait_port_free(self.port)

    def stop(self, remove=True):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if not self.log.closed:
            self.log.close()
        wait_port_free(self.port)
        if remove and self.own_dir:
            shutil.rmtree(self.dir, ignore_errors=True)


def kv(reply):
    """A flat RESP2 [k, v, k, v] list (or a RESP3 map) as a dict with str keys."""
    if isinstance(reply, dict):
        return {k.decode() if isinstance(k, bytes) else k: v for k, v in reply.items()}
    return {reply[i].decode(): reply[i + 1] for i in range(0, len(reply), 2)}


def recv_some(c, wait=0.3):
    time.sleep(wait)
    c.sock.settimeout(0.5)
    out = b""
    try:
        while True:
            chunk = c.sock.recv(65536)
            if not chunk:
                return out + b"<EOF>"
            out += chunk
            if len(chunk) < 65536:
                break
    except socket.timeout:
        pass
    finally:
        c.sock.settimeout(30)
    return out


# ── [1] a work queue ────────────────────────────────────────────────────────

def part_work_queue(binary):
    print("[1] a work queue")
    s = Server(binary)
    try:
        c = s.conn()
        check("XREADGROUP on a missing group is NOGROUP",
              str(c.cmd("XREADGROUP", "GROUP", "g", "a", "STREAMS", "q", ">")).startswith("NOGROUP"))
        check("XGROUP CREATE without MKSTREAM needs the key",
              "requires the key to exist" in str(c.cmd("XGROUP", "CREATE", "q", "g", "$")))
        check("XGROUP CREATE MKSTREAM", c.cmd("XGROUP", "CREATE", "q", "g", "$", "MKSTREAM") == "OK")
        check("a second CREATE is BUSYGROUP",
              str(c.cmd("XGROUP", "CREATE", "q", "g", "$")).startswith("BUSYGROUP"))
        for i in range(1, 6):
            c.cmd("XADD", "q", f"{i}-0", "job", f"j{i}")
        r = c.cmd("XREADGROUP", "GROUP", "g", "alice", "COUNT", "2", "STREAMS", "q", ">")
        check("alice gets the first two", r == [[b"q", [[b"1-0", [b"job", b"j1"]], [b"2-0", [b"job", b"j2"]]]]], repr(r))
        r = c.cmd("XREADGROUP", "GROUP", "g", "bob", "STREAMS", "q", ">")
        check("bob gets the other three", [e[0] for e in r[0][1]] == [b"3-0", b"4-0", b"5-0"], repr(r))
        check("nothing is left for '>'", c.cmd("XREADGROUP", "GROUP", "g", "carol", "STREAMS", "q", ">") is None)
        p = c.cmd("XPENDING", "q", "g")
        check("XPENDING summary", p == [5, b"1-0", b"5-0", [[b"alice", b"2"], [b"bob", b"3"]]], repr(p))
        check("XACK counts what it acknowledged", c.cmd("XACK", "q", "g", "1-0", "3-0", "1-0", "9-0") == 2)
        p = c.cmd("XPENDING", "q", "g")
        check("XPENDING after XACK", p == [3, b"2-0", b"5-0", [[b"alice", b"1"], [b"bob", b"2"]]], repr(p))
        rows = c.cmd("XPENDING", "q", "g", "-", "+", "10")
        check("XPENDING extended: ids, owners, counts",
              [(r[0], r[1], r[3]) for r in rows] == [(b"2-0", b"alice", 1), (b"4-0", b"bob", 1), (b"5-0", b"bob", 1)],
              repr(rows))
        rows = c.cmd("XPENDING", "q", "g", "-", "+", "10", "bob")
        check("XPENDING for one consumer", [r[0] for r in rows] == [b"4-0", b"5-0"], repr(rows))
        h = c.cmd("XREADGROUP", "GROUP", "g", "bob", "STREAMS", "q", "0")
        check("bob's history", [e[0] for e in h[0][1]] == [b"4-0", b"5-0"], repr(h))
        rows = c.cmd("XPENDING", "q", "g", "-", "+", "10", "bob")
        check("a history read counts as a delivery", [r[3] for r in rows] == [2, 2], repr(rows))
        r = c.cmd("XCLAIM", "q", "g", "carol", "0", "4-0", "JUSTID")
        check("XCLAIM JUSTID", r == [b"4-0"], repr(r))
        rows = c.cmd("XPENDING", "q", "g", "-", "+", "10")
        check("JUSTID moved the owner and kept the count",
              [(r[0], r[1], r[3]) for r in rows] == [(b"2-0", b"alice", 1), (b"4-0", b"carol", 2), (b"5-0", b"bob", 2)],
              repr(rows))
        check("XCLAIM of an entry not idle long enough claims nothing",
              c.cmd("XCLAIM", "q", "g", "carol", "3600000", "5-0") == [])
        c.cmd("XDEL", "q", "5-0")
        r = c.cmd("XAUTOCLAIM", "q", "g", "dave", "0", "0", "COUNT", "10")
        check("XAUTOCLAIM claims the live ones and reports the deleted one",
              r == [b"0-0", [[b"2-0", [b"job", b"j2"]], [b"4-0", [b"job", b"j4"]]], [b"5-0"]], repr(r))
        p = c.cmd("XPENDING", "q", "g")
        check("dave owns both now", p == [2, b"2-0", b"4-0", [[b"dave", b"2"]]], repr(p))
        cons = {kv(x)["name"]: kv(x) for x in c.cmd("XINFO", "CONSUMERS", "q", "g")}
        check("XINFO CONSUMERS lists every consumer in name order",
              [kv(x)["name"] for x in c.cmd("XINFO", "CONSUMERS", "q", "g")] == [b"alice", b"bob", b"carol", b"dave"])
        check("pending counts per consumer", [cons[n]["pending"] for n in (b"alice", b"bob", b"carol", b"dave")]
              == [0, 0, 0, 2], repr(cons))
        check("carol was active (she claimed), never-active is -1",
              cons[b"carol"]["inactive"] >= 0 and cons[b"carol"]["inactive"] < 60000, repr(cons[b"carol"]))
        check("DELCONSUMER returns what it owned", c.cmd("XGROUP", "DELCONSUMER", "q", "g", "dave") == 2)
        check("its entries left the pending list", c.cmd("XPENDING", "q", "g")[0] == 0)
        g = kv(c.cmd("XINFO", "GROUPS", "q")[0])
        check("XINFO GROUPS", (g["name"], g["consumers"], g["pending"], g["last-delivered-id"], g["entries-read"], g["lag"])
              == (b"g", 3, 0, b"5-0", 5, 0), repr(g))
        c.cmd("XADD", "q", "6-0", "job", "j6")
        g = kv(c.cmd("XINFO", "GROUPS", "q")[0])
        # 5-0 was deleted at the group's position: Redis cannot count past a
        # tombstone, and says so with a nil lag
        check("a deletion ahead of the group makes its lag unknown", g["lag"] is None, repr(g))
        c.cmd("XGROUP", "SETID", "q", "g", "$")
        g = kv(c.cmd("XINFO", "GROUPS", "q")[0])
        check("SETID to the end: lag 0, entries-read unknown (no ENTRIESREAD)",
              (g["lag"], g["entries-read"]) == (0, None), repr(g))
        info = kv(c.cmd("XINFO", "STREAM", "q"))
        check("XINFO STREAM metadata",
              (info["length"], info["last-generated-id"], info["max-deleted-entry-id"], info["entries-added"],
               info["recorded-first-entry-id"], info["groups"]) == (5, b"6-0", b"5-0", 6, b"1-0", 1), repr(info))
        check("XSETID below the top entry is refused",
              "smaller than the target stream top item" in str(c.cmd("XSETID", "q", "5-0")))
        check("XSETID with ENTRIESADDED and MAXDELETEDID",
              c.cmd("XSETID", "q", "100-0", "ENTRIESADDED", "50", "MAXDELETEDID", "7-0") == "OK")
        info = kv(c.cmd("XINFO", "STREAM", "q"))
        check("XSETID moved the metadata", (info["last-generated-id"], info["entries-added"], info["max-deleted-entry-id"])
              == (b"100-0", 50, b"7-0"), repr(info))
        check("XGROUP DESTROY", c.cmd("XGROUP", "DESTROY", "q", "g") == 1 and c.cmd("XINFO", "GROUPS", "q") == [])
        c.assert_in_sync()
    finally:
        s.stop()


# ── [2] blocking ────────────────────────────────────────────────────────────

def part_blocking(binary):
    print("[2] blocking XREADGROUP")
    s = Server(binary)
    try:
        c = s.conn()
        c.cmd("XGROUP", "CREATE", "b", "g", "$", "MKSTREAM")
        w = s.conn()
        w.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c1", "BLOCK", "0", "STREAMS", "b", ">"]))
        time.sleep(0.2)
        check("a blocked read has not answered yet", recv_some(w, 0.1) == b"")
        c.cmd("XADD", "b", "1-0", "f", "v")
        r = w.read()
        check("XADD wakes it with the entry", r == [[b"b", [[b"1-0", [b"f", b"v"]]]]], repr(r))
        check("the woken read's entry is pending", c.cmd("XPENDING", "b", "g")[0] == 1)
        w.assert_in_sync()

        t0 = time.time()
        r = w.cmd("XREADGROUP", "GROUP", "g", "c1", "BLOCK", "150", "STREAMS", "b", ">")
        dt = time.time() - t0
        check("a timeout answers nil", r is None, repr(r))
        check("after about its timeout", 0.12 <= dt < 2.0, f"{dt:.3f}s")
        w3 = s.conn(protocol=3)
        check("RESP3: the timeout's nil is a null", w3.raw("XREADGROUP", "GROUP", "g", "c1", "BLOCK", "50", "STREAMS",
                                                          "b", ">") == b"_\r\n")

        # Several waiters on one group: one entry wakes one of them.
        w2 = s.conn()
        w.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c1", "BLOCK", "0", "STREAMS", "b", ">"]))
        time.sleep(0.1)
        w2.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c2", "BLOCK", "0", "STREAMS", "b", ">"]))
        time.sleep(0.2)
        c.cmd("XADD", "b", "2-0", "f", "v2")
        r1 = w.read()
        check("the first waiter gets the entry", r1 == [[b"b", [[b"2-0", [b"f", b"v2"]]]]], repr(r1))
        check("the second keeps waiting", recv_some(w2, 0.3) == b"")
        c.cmd("XADD", "b", "3-0", "f", "v3")
        r2 = w2.read()
        check("and gets the next one", r2 == [[b"b", [[b"3-0", [b"f", b"v3"]]]]], repr(r2))

        # NOACK: delivered, not pending
        before = c.cmd("XPENDING", "b", "g")[0]
        w.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c1", "NOACK", "BLOCK", "0", "STREAMS", "b", ">"]))
        time.sleep(0.2)
        c.cmd("XADD", "b", "4-0", "f", "v4")
        r = w.read()
        check("a blocked NOACK read gets its entry", r == [[b"b", [[b"4-0", [b"f", b"v4"]]]]], repr(r))
        check("and leaves nothing pending", c.cmd("XPENDING", "b", "g")[0] == before)

        # The group, the key, the key's type change under a waiter
        for label, act, want in (("XGROUP DESTROY", ["XGROUP", "DESTROY", "b", "g"], "NOGROUP"),
                                 ("DEL", ["DEL", "b"], "NOGROUP"),
                                 ("SET over the key", ["SET", "b", "x"], "WRONGTYPE")):
            c.cmd("DEL", "b")
            c.cmd("XGROUP", "CREATE", "b", "g", "$", "MKSTREAM")
            w.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c1", "BLOCK", "0", "STREAMS", "b", ">"]))
            time.sleep(0.2)
            c.cmd(*act)
            r = w.read()
            check(f"{label} wakes a waiter with {want}", isinstance(r, RespError) and r.startswith(want), repr(r))
            w.assert_in_sync()

        # A history read and a read inside MULTI never block
        c.cmd("DEL", "b")
        c.cmd("XGROUP", "CREATE", "b", "g", "$", "MKSTREAM")
        t0 = time.time()
        r = c.cmd("XREADGROUP", "GROUP", "g", "c1", "BLOCK", "0", "STREAMS", "b", "0")
        check("a history read with BLOCK answers at once", r == [[b"b", []]] and time.time() - t0 < 1, repr(r))
        c.cmd("MULTI")
        c.cmd("XREADGROUP", "GROUP", "g", "c1", "BLOCK", "0", "STREAMS", "b", ">")
        r = c.cmd("EXEC")
        check("inside MULTI it answers nil at once", r == [None], repr(r))
        c.assert_in_sync()
    finally:
        s.stop()


# ── [3] a large pending list ────────────────────────────────────────────────

def part_large_pel(binary):
    print("[3] a large pending list")
    s = Server(binary)
    try:
        c = s.conn()
        n = 100_000
        c.cmd("XGROUP", "CREATE", "big", "g", "$", "MKSTREAM")
        for base in range(0, n, 5000):
            c.pipeline([("XADD", "big", f"{i + 1}-0", "f", "v") for i in range(base, base + 5000)])
        r = c.cmd("XREADGROUP", "GROUP", "g", "a", "COUNT", str(n), "STREAMS", "big", ">")
        check(f"{n} entries delivered", len(r[0][1]) == n)
        t0 = time.time()
        for base in range(0, n // 2, 5000):
            got = c.pipeline([("XACK", "big", "g", f"{i + 1}-0") for i in range(base, base + 5000)])
            if got != [1] * 5000:
                check("every XACK acknowledges one", False, repr(got[:5]))
                break
        dt = time.time() - t0
        check(f"{n // 2} acknowledgements in order stay fast ({dt:.2f}s)", dt < 30, f"{dt:.2f}s")
        p = c.cmd("XPENDING", "big", "g")
        check("the pending summary after them", p == [n // 2, f"{n // 2 + 1}-0".encode(), f"{n}-0".encode(),
                                                     [[b"a", str(n // 2).encode()]]], repr(p)[:200])
        rows = c.cmd("XPENDING", "big", "g", "-", "+", "3")
        check("the extended form starts at the first live one",
              [r[0] for r in rows] == [f"{i}-0".encode() for i in (n // 2 + 1, n // 2 + 2, n // 2 + 3)], repr(rows))
        r = c.cmd("XAUTOCLAIM", "big", "g", "b", "0", "0", "COUNT", "2", "JUSTID")
        check("XAUTOCLAIM skips the acknowledged ones",
              r == [f"{n // 2 + 3}-0".encode(), [f"{n // 2 + 1}-0".encode(), f"{n // 2 + 2}-0".encode()], []], repr(r))
        t0 = time.time()
        check("DELCONSUMER of the big owner", c.cmd("XGROUP", "DELCONSUMER", "big", "g", "a") == n // 2 - 2)
        check(f"is fast ({time.time() - t0:.2f}s)", time.time() - t0 < 10)
        check("leaves the claimed two", c.cmd("XPENDING", "big", "g")[0] == 2)
        # consumer churn: many created and deleted, the owners still resolve
        for k in range(300):
            c.cmd("XGROUP", "CREATECONSUMER", "big", "g", f"tmp{k:04d}")
        c.cmd("XCLAIM", "big", "g", "tmp0150", "0", f"{n // 2 + 1}-0", "JUSTID")
        for k in range(300):
            if k != 150:
                c.cmd("XGROUP", "DELCONSUMER", "big", "g", f"tmp{k:04d}")
        rows = c.cmd("XPENDING", "big", "g", "-", "+", "10")
        check("owners survive the consumers' compaction",
              [(r[0], r[1]) for r in rows] == [(f"{n // 2 + 1}-0".encode(), b"tmp0150"), (f"{n // 2 + 2}-0".encode(), b"b")],
              repr(rows))
        check("and so do the consumer list and counts",
              [(kv(x)["name"], kv(x)["pending"]) for x in c.cmd("XINFO", "CONSUMERS", "big", "g")]
              == [(b"b", 1), (b"tmp0150", 1)])
        c.assert_in_sync()
    finally:
        s.stop()


# ── [4] durability ──────────────────────────────────────────────────────────

def build_state(c):
    """Streams whose groups exercise every record kind; returns the keys."""
    c.cmd("XGROUP", "CREATE", "{r}d1", "g1", "$", "MKSTREAM")
    for i in range(1, 9):
        c.cmd("XADD", "{r}d1", f"{i}-1", "f", f"v{i}", "n", str(i))
    c.cmd("XGROUP", "CREATE", "{r}d1", "g2", "0", "ENTRIESREAD", "0")
    c.cmd("XGROUP", "CREATE", "{r}d1", "g3", "3-1")
    c.cmd("XREADGROUP", "GROUP", "g1", "alice", "COUNT", "3", "STREAMS", "{r}d1", ">")
    c.cmd("XREADGROUP", "GROUP", "g1", "bob", "COUNT", "2", "STREAMS", "{r}d1", ">")
    c.cmd("XREADGROUP", "GROUP", "g1", "carl", "NOACK", "COUNT", "1", "STREAMS", "{r}d1", ">")
    c.cmd("XREADGROUP", "GROUP", "g2", "zed", "STREAMS", "{r}d1", ">")
    c.cmd("XACK", "{r}d1", "g1", "2-1")
    c.cmd("XREADGROUP", "GROUP", "g1", "alice", "STREAMS", "{r}d1", "0")   # history: counts move
    c.cmd("XCLAIM", "{r}d1", "g1", "bob", "0", "3-1", "RETRYCOUNT", "7")
    c.cmd("XCLAIM", "{r}d1", "g1", "dora", "0", "1-1", "IDLE", "5000")
    c.cmd("XAUTOCLAIM", "{r}d1", "g2", "yan", "0", "5-1", "COUNT", "2")
    c.cmd("XGROUP", "CREATECONSUMER", "{r}d1", "g3", "idle-one")
    c.cmd("XGROUP", "CREATECONSUMER", "{r}d1", "g2", "gone")
    c.cmd("XGROUP", "DELCONSUMER", "{r}d1", "g2", "gone")
    c.cmd("XGROUP", "SETID", "{r}d1", "g3", "6-1", "ENTRIESREAD", "6")
    c.cmd("XDEL", "{r}d1", "4-1")
    c.cmd("XGROUP", "CREATE", "{r}d1", "doomed", "0")
    c.cmd("XGROUP", "DESTROY", "{r}d1", "doomed")
    # an empty stream that has only a group, and one emptied by XDEL
    c.cmd("XGROUP", "CREATE", "{r}d2", "g", "$", "MKSTREAM")
    c.cmd("XADD", "{r}d3", "5-5", "a", "b")
    c.cmd("XDEL", "{r}d3", "5-5")
    # XSETID's metadata
    c.cmd("XADD", "{r}d4", "1-0", "a", "b")
    c.cmd("XSETID", "{r}d4", "900-9", "ENTRIESADDED", "77", "MAXDELETEDID", "800-0")
    # many pending entries, half acknowledged (dead slots must not persist)
    c.cmd("XGROUP", "CREATE", "{r}d5", "g", "$", "MKSTREAM")
    c.pipeline([("XADD", "{r}d5", f"{i}-0", "f", "v") for i in range(1, 401)])
    c.cmd("XREADGROUP", "GROUP", "g", "w", "STREAMS", "{r}d5", ">")
    c.pipeline([("XACK", "{r}d5", "g", f"{i}-0") for i in range(1, 401, 2)])
    # DELREF trims and XACKDEL / XDELEX: deletions that also drop pending
    # entries, each logged
    c.cmd("XGROUP", "CREATE", "{r}d7", "a", "$", "MKSTREAM")
    c.cmd("XGROUP", "CREATE", "{r}d7", "b", "$")
    for i in range(1, 9):
        c.cmd("XADD", "{r}d7", f"{i}-7", "f", "v")
    c.cmd("XREADGROUP", "GROUP", "a", "x", "STREAMS", "{r}d7", ">")
    c.cmd("XREADGROUP", "GROUP", "b", "y", "COUNT", "5", "STREAMS", "{r}d7", ">")
    c.cmd("XTRIM", "{r}d7", "MAXLEN", "7", "DELREF")
    c.cmd("XACKDEL", "{r}d7", "a", "DELREF", "IDS", "1", "2-7")
    c.cmd("XACKDEL", "{r}d7", "a", "ACKED", "IDS", "2", "3-7", "6-7")
    c.cmd("XDELEX", "{r}d7", "DELREF", "IDS", "1", "4-7")
    c.cmd("XDELEX", "{r}d7", "ACKED", "IDS", "2", "5-7", "8-7")
    return ["{r}d1", "{r}d2", "{r}d3", "{r}d4", "{r}d5", "{r}d7"]


def state_of(c, keys):
    """Everything XINFO shows about each stream, with absolute times (which
    must survive exactly), plus the pending lists' delivery counts."""
    out = {}
    for k in keys:
        full = c.cmd("XINFO", "STREAM", k, "FULL", "COUNT", "0")
        groups = c.cmd("XINFO", "GROUPS", k)
        pend = {}
        if isinstance(groups, list):        # (an error while a replica catches up)
            for g in [kv(x)["name"] for x in groups]:
                rows = c.cmd("XPENDING", k, g, "-", "+", "1000")
                pend[g] = [(r[0], r[1], r[3]) for r in rows] if isinstance(rows, list) else rows
        out[k] = (c.cmd("TYPE", k), repr(full), repr(groups) if not isinstance(groups, list) else len(groups), pend)
    return out


def diff_states(a, b):
    for k in a:
        if a[k] != b.get(k):
            return f"{k}:\n        before: {a[k]!r:.700}\n        after : {b.get(k)!r:.700}"
    return ""


def part_durability(binary):
    print("[4] durability")
    modes = ("wal", "save", "rewrite")
    for mode in modes:
        work = tempfile.mkdtemp(prefix=f"groups40_{mode}_")
        s = Server(binary, workdir=work)
        try:
            c = s.conn()
            keys = build_state(c)
            before = state_of(c, keys)
            if mode == "save":
                check("SAVE", c.cmd("SAVE") == "OK")
            elif mode == "rewrite":
                c.cmd("BGREWRITEAOF")
                time.sleep(1.0)
            c.close()
            s.kill9()
            s.start()
            c = s.conn()
            after = state_of(c, keys)
            check(f"{mode}: groups, consumers, pending entries and metadata survive", before == after,
                  diff_states(before, after))
            # it keeps working after the reload
            r = c.cmd("XREADGROUP", "GROUP", "g1", "alice", "STREAMS", "{r}d1", ">")
            check(f"{mode}: the reloaded group serves what is left", r == [[b"{r}d1", [[b"7-1", [b"f", b"v7", b"n", b"7"]],
                                                                                    [b"8-1", [b"f", b"v8", b"n", b"8"]]]]],
                  repr(r))
            c.close()
        finally:
            s.stop(remove=False)
            shutil.rmtree(work, ignore_errors=True)

    # a consumer that only polls after its last delivery: its seen time is
    # logged once it has moved a second, so after a crash it does not look
    # idle since the delivery (a reaper keyed on idle would delete it)
    work = tempfile.mkdtemp(prefix="groups40_seen_")
    s = Server(binary, workdir=work)
    try:
        c = s.conn()
        c.cmd("XGROUP", "CREATE", "sq", "g", "$", "MKSTREAM")
        c.cmd("XADD", "sq", "1-0", "f", "v")
        c.cmd("XREADGROUP", "GROUP", "g", "poller", "STREAMS", "sq", ">")
        time.sleep(1.2)
        c.cmd("XREADGROUP", "GROUP", "g", "poller", "STREAMS", "sq", ">")
        seen = lambda conn: kv(kv(conn.cmd("XINFO", "STREAM", "sq", "FULL"))["groups"][0])["consumers"][0]
        before = kv(seen(c))["seen-time"]
        c.close()
        s.kill9()
        s.start()
        c = s.conn()
        after = kv(seen(c))["seen-time"]
        check("an empty poll's seen time survives a crash", after == before, f"{before} -> {after}")
        c.close()
    finally:
        s.stop(remove=False)
        shutil.rmtree(work, ignore_errors=True)

    s = Server(binary)
    try:
        c = s.conn()
        keys = build_state(c)
        before = state_of(c, keys)
        for k in keys:
            payload = c.cmd("DUMP", k)
            check(f"RESTORE of {k}", c.cmd("RESTORE", k + "-copy", "0", payload) == "OK")
            check(f"COPY of {k}", c.cmd("COPY", k, k + "-cp") == 1)
        r = state_of(c, [k + "-copy" for k in keys])
        cp = state_of(c, [k + "-cp" for k in keys])
        rename = lambda st, sfx: {k[:-len(sfx)]: v for k, v in st.items()}
        check("DUMP/RESTORE carries the groups", rename(r, "-copy") == before, diff_states(before, rename(r, "-copy")))
        check("COPY carries the groups", rename(cp, "-cp") == before, diff_states(before, rename(cp, "-cp")))
        c.cmd("XACK", "{r}d1-cp", "g1", "1-1")
        check("a copy shares nothing", c.cmd("XPENDING", "{r}d1", "g1")[0] != c.cmd("XPENDING", "{r}d1-cp", "g1")[0])
    finally:
        s.stop()


# ── [5] a replica ───────────────────────────────────────────────────────────

def part_replica(binary):
    print("[5] a replica")
    work = tempfile.mkdtemp(prefix="groups40_repl_")
    pa = free_port_block(30, repl=True)
    pb = pa + 20
    cl = ["--cluster", "--cluster-host", "127.0.0.1"]
    os.makedirs(os.path.join(work, "a"))
    os.makedirs(os.path.join(work, "b"))
    a = Server(binary, cl, workdir=os.path.join(work, "a"), port=pa)
    b = None
    try:
        ca = a.conn()
        keys = build_state(ca)
        b = Server(binary, cl + ["--cluster-replica", "--cluster-primary-host", "127.0.0.1",
                                 "--cluster-primary-port", str(pa)], workdir=os.path.join(work, "b"), port=pb)
        cb = b.conn()
        cb.cmd("READONLY")

        def converged(timeout=30):
            want, got = state_of(ca, keys), None
            deadline = time.time() + timeout
            while time.time() < deadline:
                got = state_of(cb, keys)
                if got == want:
                    return ""
                time.sleep(0.3)
            return diff_states(want, got)

        check("a replica that joins after the groups exist has them", not converged(), converged(1))
        # live: the WAL stream carries every change from here on
        ca.cmd("XADD", "{r}d1", "9-1", "f", "v9")
        ca.cmd("XREADGROUP", "GROUP", "g1", "erin", "STREAMS", "{r}d1", ">")
        ca.cmd("XACK", "{r}d1", "g1", "1-1", "3-1")
        ca.cmd("XCLAIM", "{r}d1", "g1", "erin", "0", "5-1", "FORCE")
        ca.cmd("XGROUP", "SETID", "{r}d1", "g2", "$")
        ca.cmd("XGROUP", "DELCONSUMER", "{r}d1", "g1", "bob")
        ca.cmd("XGROUP", "CREATE", "{r}d6", "fresh", "$", "MKSTREAM")
        ca.cmd("XSETID", "{r}d6", "5-5", "ENTRIESADDED", "3", "MAXDELETEDID", "4-4")
        keys.append("{r}d6")
        check("and follows them live", not converged(), converged(1))
    finally:
        if b:
            b.stop(remove=False)
        a.stop(remove=False)
        shutil.rmtree(work, ignore_errors=True)


# ── Part 2: the same replies as Redis ───────────────────────────────────────

def tparse(b, pos=0):
    """One RESP frame as nested (type, value) tuples: the type byte is kept,
    so a RESP3 map and an array do not compare equal."""
    t = b[pos:pos + 1]
    e = b.index(b"\r\n", pos)
    line, nxt = b[pos + 1:e], e + 2
    if t in (b"+", b"-", b":", b",", b"(", b"#", b"_"):
        return (t.decode(), line), nxt
    if t in (b"$", b"="):
        n = int(line)
        if n < 0:
            return (t.decode(), None), nxt
        return (t.decode(), b[nxt:nxt + n]), nxt + n + 2
    n = int(line)
    if n < 0:
        return (t.decode(), None), nxt
    if t in (b"%", b"|"):
        n *= 2
    items = []
    for _ in range(n):
        v, nxt = tparse(b, nxt)
        items.append(v)
    return (t.decode(), items), nxt


# Redis's internal encoding and the fields of Redis 8 features Pion lacks.
REDIS_ONLY = {b"radix-tree-keys", b"radix-tree-nodes", b"idmp-duration", b"idmp-maxsize", b"pids-tracked",
              b"iids-tracked", b"iids-added", b"iids-duplicates", b"nacked-count"}
TIME_KEYS = {b"seen-time", b"active-time", b"idle", b"inactive"}


def normalize(v, xpending_rows=False):
    t, x = v
    if not isinstance(x, list):
        return v
    if len(x) % 2 == 0 and any(k[0] in ("$", "+") and isinstance(k[1], bytes) and k[1] in REDIS_ONLY
                               for k in x[0::2]):
        kept = []
        for k, val in zip(x[0::2], x[1::2]):
            if not (isinstance(k[1], bytes) and k[1] in REDIS_ONLY):
                kept += [k, val]
        x = kept
    out, prev = [], None
    for y in x:
        if prev is not None and isinstance(prev[1], bytes) and prev[1] in TIME_KEYS and y[0] == ":" and y[1] != b"-1":
            out.append((":", "T"))
        elif prev is not None and prev[1] == b"pending" and y[0] == "*" and isinstance(y[1], list):
            # XINFO STREAM FULL's pending rows carry absolute delivery times:
            # [id, consumer, time, count] for a group, [id, time, count] for a consumer
            rows = []
            for row in y[1]:
                cells = list(row[1])
                at = 2 if len(cells) == 4 else 1
                cells[at] = (":", "T")
                rows.append((row[0], cells))
            out.append(("*", rows))
        else:
            out.append(normalize(y, xpending_rows))
        prev = y
    # XPENDING's extended rows: [id, consumer, idle, count]; the idle time
    # is a clock reading
    if xpending_rows and len(out) == 4 and out[0][0] == "$" and out[2][0] == ":":
        out[2] = (":", "T")
    return (t, out)


def probes():
    cmds = []
    cmds += [["XGROUP", "CREATE", "nokey", "g", "$"], ["XGROUP", "CREATE", "s", "g", "$", "MKSTREAM"],
             ["XGROUP", "CREATE", "s", "g", "$"], ["XGROUP", "CREATE", "s", "g2", "0", "ENTRIESREAD", "-2"],
             ["XGROUP", "CREATE", "s", "g2", "0", "ENTRIESREAD", "x"], ["XGROUP", "CREATE", "s", "g2", "0", "NOPE"],
             ["XGROUP", "CREATE", "s", "g2", "bad"], ["XGROUP", "CREATE", "s", "g2", "0-0"], ["XINFO", "GROUPS", "s"]]
    cmds += [["XADD", "s", f"{i + 1}-0", "f", f"v{i}"] for i in range(5)]
    cmds += [["XINFO", "GROUPS", "s"], ["XREADGROUP", "GROUP", "g2", "c1", "COUNT", "2", "STREAMS", "s", ">"],
             ["XINFO", "GROUPS", "s"], ["XINFO", "CONSUMERS", "s", "g2"], ["XPENDING", "s", "g2"],
             ["XPENDING", "s", "g2", "-", "+", "10"], ["XPENDING", "s", "g2", "IDLE", "0", "-", "+", "10", "c1"],
             ["XPENDING", "s", "g2", "-", "+", "10", "nobody"], ["XPENDING", "s", "nog"], ["XPENDING", "s", "g2", "-", "+"],
             ["XPENDING", "s", "g2", "IDLE", "x", "-", "+", "10"], ["XPENDING", "s", "g2", "x", "+", "10"],
             ["XPENDING", "s", "g2", "(1-0", "+", "10"], ["XREADGROUP", "GROUP", "g2", "c1", "STREAMS", "s", "0"],
             ["XREADGROUP", "GROUP", "g2", "c1", "STREAMS", "s", "$"], ["XREADGROUP", "GROUP", "nog", "c1", "STREAMS", "s", ">"],
             ["XREADGROUP", "GROUP", "g2", "c1", "STREAMS", "nokey", ">"], ["XREADGROUP", "GROUP", "g2", "c1", "STREAMS", "s"],
             ["XREADGROUP", "STREAMS", "s", ">"], ["XREADGROUP", "GROUP", "g2", "c1", "COUNT", "x", "STREAMS", "s", ">"],
             ["XREADGROUP", "GROUP", "g2", "c2", "NOACK", "STREAMS", "s", ">"], ["XINFO", "GROUPS", "s"],
             ["XACK", "s", "g2", "1-0", "9-0"], ["XACK", "s", "g2", "bad"], ["XACK", "nokey", "g2", "bad"],
             ["XACK", "s", "nog", "bad"], ["XCLAIM", "s", "g2", "c3", "0", "2-0"], ["XCLAIM", "s", "g2", "c3", "0", "2-0", "JUSTID"],
             ["XCLAIM", "s", "g2", "c3", "x", "2-0"], ["XCLAIM", "s", "g2", "c3", "0", "2-0", "IDLE", "x"],
             ["XCLAIM", "s", "g2", "c3", "0", "2-0", "NOPE"], ["XCLAIM", "s", "nog", "c3", "0", "2-0"],
             ["XAUTOCLAIM", "s", "g2", "c4", "0", "0"], ["XAUTOCLAIM", "s", "g2", "c4", "0", "0", "COUNT", "0"],
             ["XAUTOCLAIM", "s", "g2", "c4", "0", "0", "JUSTID"], ["XAUTOCLAIM", "s", "g2", "c4", "x", "0"],
             ["XAUTOCLAIM", "s", "nog", "c4", "0", "0"], ["XINFO", "CONSUMERS", "s", "g2"],
             ["XGROUP", "CREATECONSUMER", "s", "g2", "c9"], ["XGROUP", "CREATECONSUMER", "s", "g2", "c9"],
             ["XGROUP", "DELCONSUMER", "s", "g2", "c4"], ["XGROUP", "DELCONSUMER", "s", "nog", "c4"],
             ["XGROUP", "SETID", "s", "g2", "0", "ENTRIESREAD", "3"], ["XINFO", "GROUPS", "s"], ["XGROUP", "SETID", "s", "g2", "$"],
             ["XGROUP", "SETID", "s", "nog", "$"], ["XGROUP", "SETID", "nokey", "g", "$"], ["XGROUP", "DESTROY", "s", "g2"],
             ["XGROUP", "DESTROY", "s", "g2"], ["XGROUP", "HELP"], ["XGROUP", "NOPE"], ["XGROUP", "CREATE", "s"], ["XGROUP"],
             ["XDEL", "s", "3-0"], ["XINFO", "STREAM", "s"], ["XINFO", "STREAM", "s", "FULL", "COUNT", "x"],
             ["XINFO", "STREAM", "s", "NOPE"], ["XINFO", "HELP"], ["XSETID", "s", "1-0"],
             ["XSETID", "s", "99-0", "ENTRIESADDED", "1"], ["XSETID", "s", "99-0", "ENTRIESADDED", "-1"],
             ["XSETID", "s", "99-0", "MAXDELETEDID", "100-0"], ["XSETID", "s", "99-0", "ENTRIESADDED", "50", "MAXDELETEDID", "4-0"],
             ["XSETID", "nokey", "1-0"], ["XSETID", "s", "x"], ["XINFO", "STREAM", "s"],
             ["XADD", "s", "18446744073709551615-18446744073709551615", "f", "v"], ["XINFO", "STREAM", "s"],
             ["XGROUP", "CREATE", "t", "g", "$", "MKSTREAM", "ENTRIESREAD", "5"], ["XINFO", "GROUPS", "t"], ["TYPE", "t"],
             ["XINFO", "CONSUMERS", "s", "nog"], ["XINFO", "GROUPS", "nokey"], ["XINFO", "STREAM", "nokey"]]
    # entries-read and lag through deletions and reads
    cmds += [["XADD", "L", f"{i + 1}-0", "f", f"v{i}"] for i in range(5)]
    for args in (["g1", "$"], ["g2", "0"], ["g3", "3-0"], ["g4", "0", "ENTRIESREAD", "2"], ["g5", "$", "ENTRIESREAD", "9"],
                 ["g6", "3-0", "ENTRIESREAD", "-1"], ["g7", "9-0"]):
        cmds.append(["XGROUP", "CREATE", "L", *args])
    cmds += [["XINFO", "GROUPS", "L"], ["XDEL", "L", "2-0"], ["XINFO", "GROUPS", "L"],
             ["XREADGROUP", "GROUP", "g2", "c", "COUNT", "1", "STREAMS", "L", ">"], ["XINFO", "GROUPS", "L"],
             ["XREADGROUP", "GROUP", "g2", "c", "COUNT", "1", "STREAMS", "L", ">"], ["XINFO", "GROUPS", "L"],
             ["XREADGROUP", "GROUP", "g2", "c", "STREAMS", "L", ">"], ["XINFO", "GROUPS", "L"], ["XDEL", "L", "4-0"],
             ["XREADGROUP", "GROUP", "g2", "c", "STREAMS", "L", "0"], ["XCLAIM", "L", "g2", "d", "0", "4-0", "3-0"],
             ["XPENDING", "L", "g2", "-", "+", "10"], ["XAUTOCLAIM", "L", "g2", "e", "0", "0", "COUNT", "1"],
             ["XAUTOCLAIM", "L", "g2", "e", "0", "0", "COUNT", "1"], ["XAUTOCLAIM", "L", "g2", "e", "0", "0", "COUNT", "5"],
             ["XPENDING", "L", "g2"], ["XCLAIM", "L", "g2", "f", "0", "5-0", "RETRYCOUNT", "7", "IDLE", "1000", "JUSTID"],
             ["XCLAIM", "L", "g2", "f", "0", "1-0", "FORCE"], ["XCLAIM", "L", "g2", "f", "0", "5-0", "LASTID", "50-0", "JUSTID"],
             ["XINFO", "GROUPS", "L"]]
    cmds += [["XADD", "T", f"{i + 1}-0", "f", "v"] for i in range(5)]
    cmds += [["XGROUP", "CREATE", "T", "g", "0"], ["XTRIM", "T", "MAXLEN", "2"], ["XINFO", "STREAM", "T"], ["XINFO", "GROUPS", "T"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "T", ">"], ["XINFO", "GROUPS", "T"],
             ["XADD", "U", "MAXLEN", "0", "1-0", "f", "v"], ["XINFO", "STREAM", "U"], ["EXISTS", "U"],
             ["COPY", "L", "L2"], ["XINFO", "GROUPS", "L2"],
             ["XADD", "W", "1-0", "a", "b"], ["XGROUP", "CREATE", "W", "g", "0"],
             ["XREADGROUP", "GROUP", "g", "c", "COUNT", "0", "STREAMS", "W", ">"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", ">"],
             ["XREADGROUP", "GROUP", "g", "c2", "STREAMS", "W", "0"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", "0-0"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", "1-0"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", "1"],
             ["XACK", "W", "g", "1-0"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", "0"],
             ["XREADGROUP", "GROUP", "g", "c", "BLOCK", "10", "STREAMS", "W", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "W", "W", ">"], ["XREADGROUP", "GROUP", "g", "c", "NOACK", "STREAMS", "W", "0"],
             ["XINFO", "STREAM", "W", "FULL"]]
    # arity, options at their edges
    cmds += [["XACK", "a", "g"], ["XCLAIM", "a", "g", "c", "0"], ["XAUTOCLAIM", "a", "g", "c", "0"], ["XPENDING", "a"], ["XSETID", "a"],
             ["XGROUP", "DESTROY", "a"], ["XGROUP", "DESTROY", "a", "g", "x"], ["XINFO", "GROUPS"], ["XINFO", "CONSUMERS", "a"],
             ["XINFO", "STREAM"], ["XGROUP", "HELP", "x"], ["XINFO", "HELP", "x"], ["XGROUP", "SETID", "a", "g"],
             ["XGROUP", "CREATECONSUMER", "a", "g"], ["XINFO"], ["XGROUP", "create", "a", "g", "$", "mkstream"]]
    cmds += [["XADD", "a", f"{i + 1}-1", "f", f"v{i}"] for i in range(6)]
    cmds += [["XREADGROUP", "GROUP", "g", "c", "COUNT", "-5", "STREAMS", "a", ">"], ["XACK", "a", "g", "1-1", "1-1", "2"],
             ["XREADGROUP", "GROUP", "g", "c", "BLOCK", "-1", "STREAMS", "a", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "BLOCK", "x", "STREAMS", "a", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "BLOCK", "0", "STREAMS", "a", "0"],
             ["XPENDING", "a", "g", "-", "+", "-3"], ["XPENDING", "a", "g", "-", "+", "0"],
             ["XPENDING", "a", "g", "IDLE", "999999999", "-", "+", "10"], ["XPENDING", "a", "g", "+", "-", "10"],
             ["XPENDING", "a", "g", "(2-1", "(5-1", "10"], ["XPENDING", "a", "g", "-", "+", "10", "c", "extra"],
             ["XPENDING", "a", "g", "IDLE", "5"], ["XCLAIM", "a", "g", "d", "-5", "3-1", "JUSTID"],
             ["XCLAIM", "a", "g", "d", "0", "3-1", "TIME", "99999999999999", "JUSTID"],
             ["XCLAIM", "a", "g", "d", "0", "3-1", "RETRYCOUNT", "-2", "JUSTID"],
             ["XCLAIM", "a", "g", "d", "0", "3-1", "IDLE", "-50", "JUSTID"], ["XCLAIM", "a", "g", "d", "0", "3-1", "LASTID", "bad"],
             ["XCLAIM", "a", "g", "d", "0", "3-1", "4-1", "nope-id"], ["XCLAIM", "a", "g", "d", "0", "99-1"],
             ["XCLAIM", "a", "g", "d", "0", "99-1", "FORCE"], ["XPENDING", "a", "g"], ["XCLAIM", "a", "g", "d", "100000", "4-1"],
             ["XAUTOCLAIM", "a", "g", "e", "0", "(3-1", "COUNT", "2"], ["XAUTOCLAIM", "a", "g", "e", "0", "x"],
             ["XAUTOCLAIM", "a", "g", "e", "0", "0", "COUNT", "-1"],
             ["XAUTOCLAIM", "a", "g", "e", "0", "0", "COUNT", "999999999999999999"],
             ["XAUTOCLAIM", "a", "g", "e", "0", "0", "JUSTID", "COUNT", "1"], ["XAUTOCLAIM", "a", "g", "e", "0", "0", "NOPE"],
             ["XAUTOCLAIM", "a", "g", "e", "-1", "0"], ["XINFO", "CONSUMERS", "a", "g"], ["XINFO", "GROUPS", "a"],
             ["XSETID", "a", "6-0"], ["XSETID", "a", "6-1", "ENTRIESADDED", "2"], ["XSETID", "a", "6-1", "ENTRIESADDED", "x"],
             ["XSETID", "a", "6-1", "MAXDELETEDID", "7-0"], ["XSETID", "a", "6-1", "MAXDELETEDID", "x"],
             ["XSETID", "a", "6-1", "NOPE", "1"], ["XSETID", "a", "6-1", "ENTRIESADDED"],
             ["XSETID", "a", "10-0", "ENTRIESADDED", "100", "MAXDELETEDID", "9-9"], ["XINFO", "STREAM", "a"],
             ["XINFO", "GROUPS", "a"], ["XADD", "a", "*", "f", "v"], ["XINFO", "GROUPS", "a"]]
    # several streams, NOGROUP before anything is served, wrong types
    cmds += [["SET", "str", "x"], ["XADD", "m1", "1-0", "a", "1"], ["XADD", "m2", "1-0", "b", "2"], ["XGROUP", "CREATE", "m1", "g", "0"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m1", "m2", ">", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m1", "str", ">", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "str", "m1", ">", ">"], ["XGROUP", "CREATE", "m2", "g", "0"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m1", "m2", ">", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m1", "m2", "0", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m1", "m2", ">", "0"], ["XGROUP", "CREATE", "str", "g", "$"],
             ["XACK", "str", "g", "1-0"], ["XPENDING", "str", "g"], ["XCLAIM", "str", "g", "c", "0", "1-0"],
             ["XAUTOCLAIM", "str", "g", "c", "0", "0"], ["XINFO", "STREAM", "str"], ["XINFO", "GROUPS", "str"],
             ["XSETID", "str", "1-0"], ["XGROUP", "SETID", "str", "g", "$"], ["XGROUP", "CREATE", "m1", "g", "0", "ENTRIESREAD"],
             ["XGROUP", "CREATE", "m1", "g9", "$", "ENTRIESREAD", "5", "MKSTREAM"],
             ["XGROUP", "CREATE", "m1", "g9b", "$", "MKSTREAM", "MKSTREAM"], ["XGROUP", "SETID", "m1", "g", "0", "ENTRIESREAD", "-5"],
             ["XGROUP", "SETID", "m1", "g", "0", "MKSTREAM"], ["XGROUP", "CREATECONSUMER", "m1", "nog", "c"],
             ["XGROUP", "DELCONSUMER", "m1", "g", "nobody"], ["XGROUP", "DELCONSUMER", "m1", "g", "c"],
             ["XINFO", "CONSUMERS", "m1", "g"], ["XPENDING", "m1", "g"], ["XINFO", "STREAM", "m1", "FULL", "COUNT", "0"],
             ["XINFO", "STREAM", "m1", "FULL", "COUNT", "-1"], ["XINFO", "STREAM", "m1", "FULL", "COUNT"],
             ["XINFO", "STREAM", "m1", "full", "count", "1"], ["XINFO", "STREAM", "m2", "FULL"],
             ["XGROUP", "CREATE", "E", "g", "$", "MKSTREAM"], ["XINFO", "STREAM", "E", "FULL"], ["XINFO", "GROUPS", "E"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "E", ">"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "E", "0"],
             ["XINFO", "CONSUMERS", "E", "g"], ["XAUTOCLAIM", "E", "g", "c", "0", "0"], ["XPENDING", "E", "g"],
             ["XPENDING", "E", "g", "-", "+", "5"], ["XDEL", "m2", "1-0"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "m2", "0"],
             ["XAUTOCLAIM", "m2", "g", "c", "0", "0"], ["XPENDING", "m2", "g"], ["XINFO", "STREAM", "m2", "FULL"], ["DEL", "m2"],
             ["XGROUP", "CREATE", "m2", "g", "$"], ["XADD", "big", "1-0", "f", "v"], ["XGROUP", "CREATE", "big", "g", "$"],
             ["XGROUP", "SETID", "big", "g", "999-0"], ["XINFO", "GROUPS", "big"], ["XADD", "big", "5-0", "f", "v"],
             ["XINFO", "GROUPS", "big"], ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "big", ">"], ["XADD", "big", "1000-0", "f", "v"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "big", ">"], ["XINFO", "GROUPS", "big"]]
    # Redis 8.2's group-reference strategies: two groups at different points,
    # with pending entries, under trims, XDELEX and XACKDEL
    cmds += [["XADD", "ds", f"{i}-0", "f", f"v{i}"] for i in range(1, 11)]
    cmds += [["XGROUP", "CREATE", "ds", "g1", "0"], ["XGROUP", "CREATE", "ds", "g2", "0"],
             ["XREADGROUP", "GROUP", "g1", "c", "COUNT", "4", "STREAMS", "ds", ">"],
             ["XREADGROUP", "GROUP", "g2", "c", "COUNT", "2", "STREAMS", "ds", ">"],
             ["XACK", "ds", "g1", "1-0", "2-0"], ["XACK", "ds", "g2", "1-0"],
             ["XTRIM", "ds", "MAXLEN", "8", "ACKED"], ["XINFO", "STREAM", "ds"], ["XPENDING", "ds", "g1", "-", "+", "10"],
             ["XPENDING", "ds", "g2", "-", "+", "10"], ["XTRIM", "ds", "MAXLEN", "7", "DELREF"],
             ["XPENDING", "ds", "g1", "-", "+", "10"], ["XPENDING", "ds", "g2", "-", "+", "10"],
             ["XTRIM", "ds", "MAXLEN", "6", "KEEPREF"], ["XPENDING", "ds", "g1", "-", "+", "10"],
             ["XTRIM", "ds", "MINID", "6", "ACKED"], ["XTRIM", "ds", "MINID", "~", "6", "DELREF"],
             ["XADD", "ds", "ACKED", "MAXLEN", "2", "11-0", "f", "v11"], ["XINFO", "STREAM", "ds"],
             ["XADD", "ds", "KEEPREF", "DELREF", "12-0", "f", "v"], ["XTRIM", "ds", "KEEPREF", "ACKED", "MAXLEN", "1"],
             ["XTRIM", "ds", "DELREF", "MAXLEN", "9"], ["XINFO", "GROUPS", "ds"],
             ["XDELEX", "ds", "IDS", "2", "5-0", "6-0"], ["XDELEX", "ds", "ACKED", "IDS", "3", "7-0", "8-0", "99-0"],
             ["XDELEX", "ds", "DELREF", "IDS", "1", "7-0"], ["XPENDING", "ds", "g1", "-", "+", "10"],
             ["XDELEX", "nokey", "IDS", "2", "1-0", "x"], ["XDELEX", "ds", "IDS", "2", "1-0", "x"],
             ["XDELEX", "ds", "IDS", "0", "1-0"], ["XDELEX", "ds", "IDS", "x", "1-0"], ["XDELEX", "ds", "IDS", "3", "1-0"],
             ["XDELEX", "ds", "KEEPREF"], ["XDELEX", "ds", "KEEPREF", "ACKED", "IDS", "1", "1-0"],
             ["XDELEX", "ds", "IDS", "1", "1-0", "KEEPREF"], ["XDELEX", "ds", "IDS", "1", "1-0", "extra"],
             ["XDELEX", "ds", "ACKED", "x", "y"], ["XDELEX", "str", "IDS", "1", "1-0"], ["XDELEX", "ds", "IDS", "1"],
             ["XACKDEL", "ds", "g1", "IDS", "2", "3-0", "4-0"], ["XACKDEL", "ds", "nog", "IDS", "1", "1-0"],
             ["XACKDEL", "nokey", "g", "IDS", "1", "1-0"], ["XACKDEL", "str", "g", "IDS", "1", "1-0"],
             ["XACKDEL", "ds", "g1", "IDS", "1", "x"], ["XACKDEL", "ds", "g1", "ACKED", "IDS", "1"],
             ["XACKDEL", "ds", "g1", "IDS"], ["XADD", "ds", "20-0", "f", "v"], ["XADD", "ds", "21-0", "f", "v"],
             ["XREADGROUP", "GROUP", "g1", "c", "STREAMS", "ds", ">"], ["XREADGROUP", "GROUP", "g2", "c", "STREAMS", "ds", ">"],
             ["XACKDEL", "ds", "g1", "ACKED", "IDS", "2", "20-0", "21-0"], ["XACKDEL", "ds", "g2", "ACKED", "IDS", "1", "20-0"],
             ["XACKDEL", "ds", "g2", "DELREF", "IDS", "1", "21-0"], ["XPENDING", "ds", "g1"], ["XPENDING", "ds", "g2"],
             ["XINFO", "STREAM", "ds", "FULL"]]
    # a key named twice: served once per entry, in order, as Redis
    cmds += [["XADD", "dup", f"{i}-0", "f", "v"] for i in range(1, 4)]
    cmds += [["XGROUP", "CREATE", "dup", "g", "0"],
             ["XREADGROUP", "GROUP", "g", "c", "COUNT", "2", "STREAMS", "dup", "dup", ">", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "dup", "dup", ">", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "dup", "dup", "0", ">"],
             ["XREADGROUP", "GROUP", "g", "c", "STREAMS", "dup", "dup", ">", "0"],
             ["XPENDING", "dup", "g", "-", "+", "10"]]
    # from a script: the commands run, and a BLOCK read answers at once
    cmds += [["XADD", "ev", "1-0", "f", "v"], ["XGROUP", "CREATE", "ev", "g", "0"],
             ["EVAL", "return redis.call('XREADGROUP','GROUP','g','c','COUNT','1','STREAMS',KEYS[1],'>')", "1", "ev"],
             ["EVAL", "return redis.call('XREADGROUP','GROUP','g','c','BLOCK','0','STREAMS',KEYS[1],'>')", "1", "ev"],
             ["EVAL", "return redis.call('XPENDING',KEYS[1],'g')", "1", "ev"],
             ["EVAL", "return redis.call('XACK',KEYS[1],'g','1-0')", "1", "ev"],
             ["EVAL", "return redis.call('XINFO','GROUPS',KEYS[1])", "1", "ev"],
             ["EVAL", "return redis.pcall('XGROUP','CREATE',KEYS[1],'g','$')", "1", "ev"]]
    return cmds


def redis_differential(binary, rs):
    print("[6] the same replies as Redis")
    for proto in (2, 3):
        s = Server(binary)
        rport = free_port_block(2)
        rdir = tempfile.mkdtemp(prefix="groups40_redis_")
        r = subprocess.Popen([rs, "--port", str(rport), "--save", "", "--appendonly", "no", "--dir", rdir],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_ready_pid(rport, r, 30)
            pc, rc = s.conn(proto), Conn(rport)
            if proto == 3:
                rc.cmd("HELLO", "3")
            for cmd in probes():
                a, b = pc.raw(*cmd), rc.raw(*cmd)
                xp = cmd[0] == "XPENDING"
                na, nb = normalize(tparse(a)[0], xp), normalize(tparse(b)[0], xp)
                if cmd[0] == "XADD" and "*" in cmd:     # a clock reading: compare its shape
                    shape = lambda v: (v[0], bool(re.fullmatch(rb"\d+-0", v[1] or b"")))
                    na, nb = shape(na), shape(nb)
                check(f"RESP{proto} {' '.join(cmd)[:70]}", na == nb,
                      f"\n        pion : {a[:400]!r}\n        redis: {b[:400]!r}")
            # blocking: what a waiter gets when its group, key or key type goes
            for act in (["XGROUP", "DESTROY", "bk", "g"], ["DEL", "bk"], ["SET", "bk", "x"],
                        ["XADD", "bk", "7-7", "f", "v"]):
                outs = []
                for port in (s.port, rport):
                    ctl, w = Conn(port), Conn(port)
                    if proto == 3:
                        ctl.cmd("HELLO", "3")
                        w.cmd("HELLO", "3")
                    ctl.cmd("DEL", "bk")
                    ctl.cmd("XGROUP", "CREATE", "bk", "g", "$", "MKSTREAM")
                    w.sock.sendall(encode(["XREADGROUP", "GROUP", "g", "c", "BLOCK", "0", "STREAMS", "bk", ">"]))
                    time.sleep(0.15)
                    ctl.cmd(*act)
                    outs.append(w.read_raw())
                    w.close()
                    ctl.close()
                check(f"RESP{proto} a waiter after {' '.join(act)}", outs[0] == outs[1], repr(outs))
        finally:
            r.terminate()
            r.wait(10)
            shutil.rmtree(rdir, ignore_errors=True)
            s.stop()


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("PION_BIN", os.path.join(REPO, "pion-server")))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose
    binary = os.path.abspath(args.binary)
    part_work_queue(binary)
    part_blocking(binary)
    part_large_pel(binary)
    part_durability(binary)
    part_replica(binary)
    rs = shutil.which("redis-server")
    if rs:
        redis_differential(binary, rs)
    else:
        print("[6] SKIP: redis-server not on PATH")
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
