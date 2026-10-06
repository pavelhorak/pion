#!/usr/bin/env python3
"""#47: admin and introspection commands do what they answer, or refuse.

Part 1 checks Pion's own behavior (no Redis needed): CLIENT ID / LIST / INFO /
KILL / PAUSE / UNPAUSE / REPLY / UNBLOCK, SLOWLOG recording, the ACL users
(--requirepass and --tenant), ACL LOG, SHUTDOWN ABORT, OBJECT ENCODING,
LASTSAVE and the standalone CLUSTER / READONLY refusals.

Part 2 sends the same commands to Pion and to a live redis-server and requires
byte-identical replies, in RESP2 and RESP3, except where Pion differs on
purpose (its users are fixed, it has no latency monitor, no client-side
caching, and its own memory figures). Skipped when redis-server is not on PATH.

    python3 tests/test_admin_commands.py [--binary ./pion-server] [-v]
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, encode, parse_bytes, wait_ready_pid, wait_port_free  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []
VERBOSE = False


def check(name, ok, detail=""):
    if ok:
        if VERBOSE:
            print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}  {detail}"[:600])


def free_port_block(n=4):
    """A port p with p .. p+n-1 free (Pion also binds p+1 and p+2+worker)."""
    for _ in range(200):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if p + n > 65000:
            continue
        ok = True
        for q in range(p, p + n):
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
    def __init__(self, binary, extra=(), password=None):
        self.port = free_port_block()
        self.dir = tempfile.mkdtemp(prefix="admin47_")
        self.log = open(os.path.join(self.dir, "log"), "w")
        self.proc = subprocess.Popen([binary, "-p", str(self.port), "-w", "1", "--no-crash-log", "--no-auto-detect",
                                      "--no-auto-embed", *extra], cwd=self.dir, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.password = password
        wait_ready_pid(self.port, self.proc, 60, password=password)

    def conn(self, auth=True, protocol=2):
        c = Conn(self.port)
        if auth and self.password:
            c.cmd("AUTH", self.password)
        if protocol == 3:
            c.cmd("HELLO", "3")
        return c

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        wait_port_free(self.port)
        shutil.rmtree(self.dir, ignore_errors=True)


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
    except OSError:
        return out + b"<EOF>"
    finally:
        c.sock.settimeout(10)
    return out


def closed(c):
    """Has the server closed this connection?"""
    try:
        c.sock.sendall(encode(["PING"]))
    except OSError:
        return True
    return recv_some(c).endswith(b"<EOF>")


def client_line(c, cid):
    for line in c.cmd("CLIENT", "LIST").decode().strip().split("\n"):
        fields = dict(f.split("=", 1) for f in line.split(" ") if "=" in f)
        if fields.get("id") == str(cid):
            return fields
    return {}


# ── Part 1: Pion's behavior ─────────────────────────────────────────────────

def part_client(binary):
    print("[1] CLIENT")
    s = Server(binary)
    try:
        a, b = s.conn(), s.conn()
        ida, idb = a.cmd("CLIENT", "ID"), b.cmd("CLIENT", "ID")
        check("IDs differ", ida != idb, f"{ida} {idb}")
        fa = client_line(a, ida)
        for k in ("id", "addr", "laddr", "fd", "name", "age", "idle", "flags", "db", "sub", "psub", "ssub",
                  "multi", "watch", "qbuf", "qbuf-free", "events", "user", "redir", "resp", "lib-name", "lib-ver",
                  "io-thread"):
            check(f"CLIENT LIST has {k}", k in fa, repr(fa))
        check("addr is ip:port of the peer", fa.get("addr") == "%s:%d" % a.sock.getsockname(), repr(fa.get("addr")))
        check("laddr is the server's ip:port", fa.get("laddr") == "127.0.0.1:%d" % s.port, repr(fa.get("laddr")))
        check("flags N, multi -1", fa.get("flags") == "N" and fa.get("multi") == "-1", repr(fa))
        info = a.cmd("CLIENT", "INFO").decode()
        check("CLIENT INFO is the caller's line", info.startswith("id=%d " % ida) and info.endswith("\n"), info)
        # an ID is never reused: a new connection on the closed one's fd gets a new ID
        b.close()
        time.sleep(0.2)
        c = s.conn()
        idc = c.cmd("CLIENT", "ID")
        check("an ID is never reused", idc > idb, f"{idb} -> {idc}")
        # flags and counts
        c.cmd("CLIENT", "SETNAME", "worker-1")
        c.cmd("CLIENT", "SETINFO", "LIB-NAME", "redis-py")
        c.cmd("CLIENT", "SETINFO", "LIB-VER", "5.0.1")
        c.cmd("WATCH", "w1", "w2")
        c.cmd("MULTI")
        c.cmd("SET", "k", "v")
        fc = client_line(a, idc)
        check("name, lib-name, lib-ver", (fc.get("name"), fc.get("lib-name"), fc.get("lib-ver"))
              == ("worker-1", "redis-py", "5.0.1"), repr(fc))
        check("MULTI: flags x, multi 1, watch 2", (fc.get("flags"), fc.get("multi"), fc.get("watch")) == ("x", "1", "2"),
              repr(fc))
        c.cmd("DISCARD")
        p = s.conn()
        p.cmd("SUBSCRIBE", "ch1")
        p.sock.sendall(encode(["PSUBSCRIBE", "pat*"]))
        recv_some(p)
        idp = None
        for line in a.cmd("CLIENT", "LIST", "TYPE", "pubsub").decode().strip().split("\n"):
            idp = int(line.split(" ")[0][3:])
        fp = client_line(a, idp) if idp else {}
        check("a subscriber: flags P, sub 1, psub 1, TYPE pubsub", (fp.get("flags"), fp.get("sub"), fp.get("psub"))
              == ("P", "1", "1"), repr(fp))
        bl = s.conn()
        idbl = bl.cmd("CLIENT", "ID")
        bl.sock.sendall(encode(["BLPOP", "nolist", "0"]))
        time.sleep(0.2)
        check("a blocked client: flags b", client_line(a, idbl).get("flags") == "b", repr(client_line(a, idbl)))
        r3 = s.conn(protocol=3)
        check("resp=3 after HELLO 3", client_line(a, r3.cmd("CLIENT", "ID")).get("resp") == "3")
        # UNBLOCK: TIMEOUT answers as the timeout (nil array), ERROR with UNBLOCKED
        check("UNBLOCK a blocked BLPOP -> 1", a.cmd("CLIENT", "UNBLOCK", str(idbl)) == 1)
        check("... which gets the timeout's nil", recv_some(bl) == b"*-1\r\n")
        bl.sock.sendall(encode(["BRPOPLPUSH", "nolist", "dst", "0"]))
        time.sleep(0.2)
        check("UNBLOCK ... ERROR -> 1", a.cmd("CLIENT", "UNBLOCK", str(idbl), "ERROR") == 1)
        check("... which gets -UNBLOCKED", recv_some(bl) == b"-UNBLOCKED client unblocked via CLIENT UNBLOCK\r\n")
        bl.sock.sendall(encode(["BRPOPLPUSH", "nolist", "dst", "0"]) + encode(["PING"]))
        time.sleep(0.2)
        a.cmd("CLIENT", "UNBLOCK", str(idbl))
        check("BRPOPLPUSH times out with a nil bulk, then its pipeline runs", recv_some(bl) == b"$-1\r\n+PONG\r\n")
        bl.sock.sendall(encode(["XREAD", "BLOCK", "0", "STREAMS", "nostream", "$"]))
        time.sleep(0.2)
        check("UNBLOCK an XREAD BLOCK", a.cmd("CLIENT", "UNBLOCK", str(idbl)) == 1 and recv_some(bl) == b"*-1\r\n")
        check("UNBLOCK a client that is not blocked -> 0", a.cmd("CLIENT", "UNBLOCK", str(idc)) == 0)
        # KILL
        k1 = s.conn()
        idk1 = k1.cmd("CLIENT", "ID")
        check("KILL ID -> 1", a.cmd("CLIENT", "KILL", "ID", str(idk1)) == 1)
        check("... and the connection is closed", closed(k1))
        k2 = s.conn()
        addr = "%s:%d" % k2.sock.getsockname()
        check("KILL <ip:port> -> OK", a.cmd("CLIENT", "KILL", addr) == "OK")
        check("... closed", closed(k2))
        check("KILL <unknown ip:port> -> No such client", a.cmd("CLIENT", "KILL", "1.2.3.4:5") == "ERR No such client")
        check("KILL TYPE pubsub kills the subscriber", a.cmd("CLIENT", "KILL", "TYPE", "pubsub") == 1 and closed(p))
        check("KILL SKIPME yes leaves the caller", a.cmd("CLIENT", "KILL", "ID", str(ida)) == 0 and a.cmd("PING") == "PONG")
        k4 = s.conn()
        idk4 = k4.cmd("CLIENT", "ID")
        k4.sock.sendall(encode(["CLIENT", "KILL", "ID", str(idk4), "SKIPME", "no"]) + encode(["PING"]))
        got = recv_some(k4)
        if not got.endswith(b"<EOF>"):
            got += recv_some(k4)
        check("KILL of itself: its reply, then closed, the rest dropped", got == b":1\r\n<EOF>", repr(got))
        check("KILL MAXAGE keeps young connections", a.cmd("CLIENT", "KILL", "MAXAGE", "100") == 0)
    finally:
        s.stop()


def part_pause(binary):
    print("[2] CLIENT PAUSE")
    s = Server(binary)
    try:
        a, b = s.conn(), s.conn()
        b.cmd("SET", "x", "1")
        a.cmd("CLIENT", "PAUSE", "400", "WRITE")
        t0 = time.time()
        check("PAUSE WRITE: a read runs", b.cmd("GET", "x") == b"1" and time.time() - t0 < 0.2)
        t0 = time.time()
        check("PAUSE WRITE: a write waits for the pause", b.cmd("SET", "y", "2") == "OK" and time.time() - t0 >= 0.3,
              f"{time.time() - t0:.3f}s")
        a.cmd("CLIENT", "PAUSE", "10000", "WRITE")
        b.sock.sendall(encode(["SET", "z", "3"]))
        time.sleep(0.2)
        held = [l for l in a.cmd("CLIENT", "LIST").decode().split("\n") if " flags=b " in l]
        check("a held client shows flags b", len(held) == 1, repr(held))
        b.sock.sendall(encode(["GET", "z"]))
        check("nothing runs behind the held write", recv_some(b) == b"")
        check("UNPAUSE -> OK", a.cmd("CLIENT", "UNPAUSE") == "OK")
        check("UNPAUSE releases the held write and what followed it", recv_some(b) == b"+OK\r\n$1\r\n3\r\n")
        a.cmd("CLIENT", "PAUSE", "10000", "WRITE")
        c = s.conn()
        idc = c.cmd("CLIENT", "ID")
        c.sock.sendall(encode(["SET", "q", "1"]))
        time.sleep(0.2)
        check("UNBLOCK does not release a paused client", a.cmd("CLIENT", "UNBLOCK", str(idc)) == 0)
        check("KILL of a held client closes it now", a.cmd("CLIENT", "KILL", "ID", str(idc)) == 1 and closed(c))
        a.cmd("CLIENT", "UNPAUSE")
        a.cmd("CLIENT", "PAUSE", "300", "ALL")
        t0 = time.time()
        check("PAUSE ALL holds even PING", b.cmd("PING") == "PONG" and time.time() - t0 >= 0.2, f"{time.time() - t0:.3f}s")
        check("the server serves after the pause", b.cmd("GET", "q") is None and b.cmd("GET", "y") == b"2")
    finally:
        s.stop()


def part_reply(binary):
    print("[3] CLIENT REPLY")
    s = Server(binary)
    try:
        c = s.conn()
        c.sock.sendall(encode(["CLIENT", "REPLY", "OFF"]) + encode(["SET", "k", "v"]) + encode(["GET", "k"])
                       + encode(["NOPE"]) + encode(["CLIENT", "REPLY", "ON"]) + encode(["GET", "k"]))
        check("REPLY OFF drops every reply until ON", recv_some(c) == b"+OK\r\n$1\r\nv\r\n")
        c.sock.sendall(encode(["CLIENT", "REPLY", "SKIP"]) + encode(["INCR", "n"]) + encode(["INCR", "n"]))
        check("REPLY SKIP drops one reply", recv_some(c) == b":2\r\n")
        c.sock.sendall(encode(["CLIENT", "REPLY", "SKIP"]))
        recv_some(c, 0.1)
        c.sock.sendall(encode(["INCR", "n"]) + encode(["INCR", "n"]))
        check("... also across receive buffers", recv_some(c) == b":4\r\n")
        c.sock.sendall(encode(["CLIENT", "REPLY", "OFF"]))
        recv_some(c, 0.1)
        c.sock.sendall(encode(["SET", "big", "x" * 100000]) + encode(["GET", "big"]) + encode(["CLIENT", "REPLY", "ON"]))
        check("a large reply is dropped too", recv_some(c) == b"+OK\r\n")
        check("the fast path serves it again afterwards", c.cmd("GET", "k") == b"v")
        c.sock.sendall(encode(["CLIENT", "REPLY", "OFF"]) + encode(["RESET"]) + encode(["PING"]))
        got = recv_some(c)
        check("RESET turns REPLY back ON (its own reply goes out)", got == b"+RESET\r\n+PONG\r\n", repr(got))
    finally:
        s.stop()


def part_slowlog(binary):
    print("[4] SLOWLOG")
    s = Server(binary)
    try:
        c = s.conn()
        check("CONFIG GET slowlog-*", c.cmd("CONFIG", "GET", "slowlog-log-slower-than") == [b"slowlog-log-slower-than", b"10000"])
        c.cmd("CONFIG", "SET", "slowlog-log-slower-than", "0")
        c.cmd("SLOWLOG", "RESET")
        c.cmd("CLIENT", "SETNAME", "me")
        c.cmd("RPUSH", "many", *[str(i) for i in range(40)])
        c.cmd("SET", "k" * 130, "v")
        c.cmd("MULTI")
        c.cmd("HSET", "h", "f", "v")
        c.cmd("EXEC")
        try:
            c.cmd("AUTH", "secret")
        except Exception:
            pass
        log = c.cmd("SLOWLOG", "GET", "-1")
        names = [e[3][0] for e in log]
        check("every command run is logged at threshold 0, newest first",
              names[:6] == [b"AUTH", b"HSET", b"MULTI", b"SET", b"RPUSH", b"CLIENT"], repr(names))
        check("EXEC is not logged; the commands it ran are", b"EXEC" not in names)
        e = log[0]
        check("an entry is id, time, duration, args, peer, name, argc",
              len(e) == 7 and isinstance(e[0], int) and isinstance(e[2], int) and e[4] == b"%s:%d" % (
                  c.sock.getsockname()[0].encode(), c.sock.getsockname()[1]) and e[5] == b"me", repr(e))
        check("AUTH's argument is redacted", log[0][3] == [b"AUTH", b"(redacted)"] and log[0][6] == 2, repr(log[0]))
        rp = [x for x in log if x[3][0] == b"RPUSH"][0]
        check("more than 32 arguments: 31 and a count", len(rp[3]) == 32 and rp[3][31] == b"... (11 more arguments)"
              and rp[6] == 42, repr(rp[3][-2:]))
        st = [x for x in log if x[3][0] == b"SET"][0]
        check("a long argument is cut at 128 bytes", st[3][1] == b"k" * 128 + b"... (2 more bytes)", repr(st[3][1][-24:]))
        check("SLOWLOG LEN", c.cmd("SLOWLOG", "LEN") == len(log) + 1)
        c.cmd("CONFIG", "SET", "slowlog-max-len", "2")
        c.cmd("PING")
        check("slowlog-max-len trims", c.cmd("SLOWLOG", "LEN") == 2)
        c.cmd("CONFIG", "SET", "slowlog-log-slower-than", "-1")
        c.cmd("SLOWLOG", "RESET")
        c.cmd("PING")
        check("-1 logs nothing", c.cmd("SLOWLOG", "LEN") == 0)
        c.cmd("CONFIG", "SET", "slowlog-log-slower-than", "10000", "slowlog-max-len", "128")
        check("several CONFIG SET pairs at once", c.cmd("CONFIG", "GET", "slowlog-max-len") == [b"slowlog-max-len", b"128"])
        bad = c.cmd("CONFIG", "SET", "slowlog-max-len", "7", "nosuchparam", "1")
        check("a bad pair sets nothing", isinstance(bad, RespError) and
              c.cmd("CONFIG", "GET", "slowlog-max-len") == [b"slowlog-max-len", b"128"], repr(bad))
    finally:
        s.stop()


def part_acl(binary):
    print("[5] ACL")
    s = Server(binary, extra=("--requirepass", "secret", "--tenant", "acme=apw"), password="secret")
    try:
        a = s.conn()
        check("ACL USERS: default and the tenant", a.cmd("ACL", "USERS") == [b"acme", b"default"])
        check("ACL WHOAMI", a.cmd("ACL", "WHOAMI") == b"default")
        u = a.cmd("ACL", "GETUSER", "default")
        import hashlib
        h = hashlib.sha256(b"secret").hexdigest().encode()
        check("GETUSER default: on, its password's SHA-256, everything",
              u == [b"flags", [b"on", b"sanitize-payload"], b"passwords", [h], b"commands", b"+@all",
                    b"keys", b"~*", b"channels", b"&*", b"selectors", []], repr(u))
        t = a.cmd("ACL", "GETUSER", "acme")
        check("GETUSER of a tenant: its keys, its commands",
              t[7] == b"~acme:*" and t[5].startswith(b"-@all +") and b"+get" in t[5] and b"+flushall" not in t[5],
              repr(t)[:200])
        check("GETUSER of nobody -> nil", a.cmd("ACL", "GETUSER", "nobody") is None)
        lst = a.cmd("ACL", "LIST")
        check("ACL LIST: one line per user", len(lst) == 2 and lst[1] == b"user default on sanitize-payload #" + h
              + b" ~* &* +@all", repr(lst)[:300])
        check("DRYRUN: a tenant may not FLUSHALL",
              a.cmd("ACL", "DRYRUN", "acme", "FLUSHALL") == b"User acme has no permissions to run the 'flushall' command")
        check("DRYRUN: a tenant may GET", a.cmd("ACL", "DRYRUN", "acme", "GET", "k") == "OK")
        check("SETUSER refuses", isinstance(a.cmd("ACL", "SETUSER", "x", "on"), RespError))
        check("DELUSER of a tenant refuses", isinstance(a.cmd("ACL", "DELUSER", "acme"), RespError))
        check("GENPASS: 64 hex", len(a.cmd("ACL", "GENPASS")) == 64 and len(a.cmd("ACL", "GENPASS", "5")) == 2)
        # ACL LOG
        x = s.conn(auth=False)
        check("a wrong password", str(x.cmd("AUTH", "nope")).startswith("WRONGPASS"))
        x.cmd("AUTH", "nope")
        x.cmd("AUTH", "acme", "bad")
        x.cmd("AUTH", "acme", "apw")
        denied = x.cmd("FLUSHALL")
        check("a tenant's denied command: Redis's NOPERM text",
              denied == "NOPERM User acme has no permissions to run the 'flushall' command", repr(denied))
        log = a.cmd("ACL", "LOG")
        rows = [dict(zip(e[::2], e[1::2])) for e in log]
        check("ACL LOG: the denial, the tenant's failed AUTH, the default's two (grouped)",
              [(r[b"reason"], r[b"object"], r[b"username"], r[b"count"]) for r in rows] ==
              [(b"command", b"flushall", b"acme", 1), (b"auth", b"AUTH", b"acme", 1), (b"auth", b"AUTH", b"default", 2)],
              repr(rows)[:400])
        check("ACL LOG entries carry client-info", rows[0][b"client-info"].startswith(b"id=") and
              b" user=acme " in rows[0][b"client-info"])
        check("ACL LOG 1", len(a.cmd("ACL", "LOG", "1")) == 1)
        check("ACL LOG RESET", a.cmd("ACL", "LOG", "RESET") == "OK" and a.cmd("ACL", "LOG") == [])
        check("CLIENT LIST user=acme", any(" user=acme " in l for l in a.cmd("CLIENT", "LIST").decode().split("\n")))
        check("CLIENT KILL USER acme", a.cmd("CLIENT", "KILL", "USER", "acme") == 1 and closed(x))
    finally:
        s.stop()


def part_misc(binary):
    print("[6] SHUTDOWN ABORT, CLUSTER, OBJECT, LASTSAVE, BGREWRITEAOF")
    s = Server(binary)
    try:
        c = s.conn()
        check("SHUTDOWN ABORT refuses and the server stays up",
              c.cmd("SHUTDOWN", "ABORT") == "ERR No shutdown in progress." and c.cmd("PING") == "PONG")
        for sub in (["KEYSLOT", "k"], ["COUNTKEYSINSLOT", "0"], ["GETKEYSINSLOT", "0", "1"], ["REPLICATE", "x"],
                    ["RESET"], ["INFO"], ["HELP"]):
            check(f"CLUSTER {sub[0]} outside cluster mode refuses",
                  c.cmd("CLUSTER", *sub) == "ERR This instance has cluster support disabled")
        check("CLUSTER STATS stays", isinstance(c.cmd("CLUSTER", "STATS"), bytes))
        for cmd in ("READONLY", "READWRITE", "ASKING"):
            check(f"{cmd} outside cluster mode refuses", c.cmd(cmd) == "ERR This instance has cluster support disabled")
        check("LASTSAVE is the startup time", abs(c.cmd("LASTSAVE") - time.time()) < 120)
        check("BGREWRITEAOF answers as Redis", c.cmd("BGREWRITEAOF") == "Background append only file rewriting started")
        check("MEMORY USAGE of a missing key is nil", c.cmd("MEMORY", "USAGE", "nosuch") is None)
        c.cmd("SET", "k", "v")
        check("MEMORY USAGE of a key is a size", isinstance(c.cmd("MEMORY", "USAGE", "k"), int))
        st = c.cmd("MEMORY", "STATS")
        st = dict(zip(st[::2], st[1::2]))
        check("MEMORY STATS: RSS, peak and keys", st.get(b"keys.count") == 1 and st.get(b"total.allocated", 0) > 0,
              repr(st))
        check("OBJECT IDLETIME refuses (Pion keeps no access times)", isinstance(c.cmd("OBJECT", "IDLETIME", "k"),
                                                                               RespError))
        count = c.cmd("COMMAND", "COUNT")
        lst = c.cmd("COMMAND", "LIST")
        check("COMMAND COUNT counts COMMAND LIST's commands (not its subcommands)",
              count == len([x for x in lst if b"|" not in x]) and b"client|kill" in lst)
        check("COMMAND INFO of a Pion-only command", c.cmd("COMMAND", "INFO", "ft.search")[0][0] == b"ft.search")
        check("MODULE LIST names search", c.cmd("MODULE", "LIST")[0][1] == b"search")
    finally:
        s.stop()


# ── Part 2: the same replies as Redis ───────────────────────────────────────

def probes():
    P = []
    P += [["ACL", "WHOAMI"], ["ACL", "CAT"], ["ACL", "CAT", "nosuch"],
          ["ACL", "LOG"], ["ACL", "LOG", "x"], ["ACL", "DRYRUN", "default", "GET", "k"],
          ["ACL", "DRYRUN", "nouser", "GET", "k"], ["ACL", "DRYRUN", "default", "nosuchcmd"],
          ["ACL", "DRYRUN", "default", "GET"], ["ACL", "SAVE"], ["ACL", "LOAD"], ["ACL", "HELP"], ["ACL", "NOPE"],
          ["ACL", "GETUSER", "default"], ["ACL", "GETUSER", "nouser"], ["ACL", "LIST"], ["ACL", "USERS"],
          ["ACL", "GENPASS", "0"], ["ACL", "GENPASS", "4097"], ["ACL", "GENPASS", "x"], ["ACL", "DELUSER", "default"],
          ["ACL", "DELUSER", "nouser"]]
    P += [["MODULE", "HELP"], ["MODULE", "NOPE"]]
    P += [["SLOWLOG", "LEN"], ["SLOWLOG", "HELP"], ["SLOWLOG", "NOPE"], ["SLOWLOG", "GET", "x"], ["SLOWLOG", "GET", "-2"],
          ["SLOWLOG", "GET", "1", "2"], ["SLOWLOG"]]
    P += [["COMMAND", "INFO", "get"], ["COMMAND", "INFO", "set", "nosuch", "zadd"],
          ["COMMAND", "GETKEYS", "SET", "a", "b"], ["COMMAND", "GETKEYSANDFLAGS", "LMOVE", "a", "b", "LEFT", "RIGHT"],
          ["COMMAND", "GETKEYS", "EVAL", "return 1", "2", "a", "b", "c"], ["COMMAND", "GETKEYS", "MSET", "a", "1", "b", "2"],
          ["COMMAND", "GETKEYS", "ZUNIONSTORE", "d", "2", "a", "b", "WEIGHTS", "1", "2"],
          ["COMMAND", "GETKEYS", "SORT", "s", "STORE", "d"], ["COMMAND", "GETKEYS", "XREAD", "STREAMS", "a", "b", "0", "0"],
          ["COMMAND", "GETKEYS", "GET"], ["COMMAND", "GETKEYS", "PING"], ["COMMAND", "GETKEYS", "nosuch", "a"],
          ["COMMAND", "HELP"], ["COMMAND", "NOPE"], ["COMMAND", "LIST", "FILTERBY", "NOPE", "x"],
          ["COMMAND", "LIST", "FILTERBY", "MODULE", "x"]]
    P += [["CLUSTER", "INFO"], ["CLUSTER", "KEYSLOT", "k"], ["CLUSTER", "HELP"], ["READONLY"], ["READWRITE"], ["ASKING"]]
    P += [["CLIENT", "GETNAME"], ["CLIENT", "SETNAME", "n1"], ["CLIENT", "GETNAME"], ["CLIENT", "SETNAME", "a b"],
          ["CLIENT", "NO-EVICT", "on"], ["CLIENT", "NO-EVICT", "maybe"], ["CLIENT", "NO-TOUCH", "off"],
          ["CLIENT", "REPLY", "ON"], ["CLIENT", "REPLY", "NOPE"], ["CLIENT", "KILL", "ID", "x"], ["CLIENT", "KILL", "ID", "0"],
          ["CLIENT", "KILL", "NOPE", "x"], ["CLIENT", "KILL", "1.2.3.4:5"], ["CLIENT", "KILL", "ID", "999999"],
          ["CLIENT", "KILL", "MAXAGE", "x"], ["CLIENT", "KILL", "MAXAGE", "0"], ["CLIENT", "KILL", "USER", "nouser"],
          ["CLIENT", "KILL", "TYPE", "nope"], ["CLIENT", "KILL", "SKIPME", "maybe"], ["CLIENT", "KILL", "ID", "1", "SKIPME"],
          ["CLIENT", "PAUSE", "x"], ["CLIENT", "PAUSE", "-1"], ["CLIENT", "PAUSE", "10", "NOPE"],
          ["CLIENT", "PAUSE", "99999999999999999999"], ["CLIENT", "UNPAUSE"], ["CLIENT", "UNPAUSE", "x"],
          ["CLIENT", "UNBLOCK", "x"], ["CLIENT", "UNBLOCK", "1", "NOPE"], ["CLIENT", "UNBLOCK", "999999"],
          ["CLIENT", "UNBLOCK", "0"], ["CLIENT", "SETINFO", "LIB-NAME", "x"], ["CLIENT", "SETINFO", "NOPE", "x"],
          ["CLIENT", "SETINFO", "lib-ver", "a b"], ["CLIENT", "CACHING", "yes"], ["CLIENT", "GETREDIR"],
          ["CLIENT", "TRACKINGINFO"], ["CLIENT", "TRACKING", "off"], ["CLIENT", "HELP"], ["CLIENT", "NOPE"],
          ["CLIENT", "LIST", "TYPE", "nope"], ["CLIENT", "LIST", "ID", "x"], ["CLIENT", "LIST", "ID", "999999"],
          ["CLIENT", "LIST", "FOO"], ["CLIENT", "ID", "x"], ["CLIENT", "INFO", "x"], ["CLIENT", "TRACKING"], ["CLIENT"]]
    P += [["LATENCY", "LATEST"], ["LATENCY", "HISTORY", "command"], ["LATENCY", "RESET"], ["LATENCY", "GRAPH", "command"],
          ["LATENCY", "DOCTOR"], ["LATENCY", "HELP"], ["LATENCY", "NOPE"], ["LATENCY", "LATEST", "x"], ["LATENCY"]]
    P += [["MEMORY", "USAGE", "nosuch"], ["MEMORY", "USAGE"], ["MEMORY", "USAGE", "k", "SAMPLES"],
          ["MEMORY", "USAGE", "k", "FOO", "1"], ["MEMORY", "USAGE", "k", "SAMPLES", "x"], ["MEMORY", "PURGE"],
          ["MEMORY", "HELP"], ["MEMORY", "NOPE"], ["MEMORY"]]
    P += [["BGREWRITEAOF"], ["SHUTDOWN", "ABORT"],
          ["CONFIG", "SET", "slowlog-log-slower-than", "abc"],
          ["CONFIG", "SET", "slowlog-log-slower-than", "-2"], ["CONFIG", "SET", "slowlog-max-len", "-1"]]
    # OBJECT ENCODING by size (built in one command each, as Redis decides it then)
    P += [["RPUSH", "l1", *[str(i) for i in range(128)]], ["OBJECT", "ENCODING", "l1"],
          ["RPUSH", "l2", *["x" * 64] * 130], ["OBJECT", "ENCODING", "l2"],
          ["RPUSH", "l3", "x" * 65], ["OBJECT", "ENCODING", "l3"],
          ["RPUSH", "l4", *[str(i) for i in range(2000)]], ["OBJECT", "ENCODING", "l4"],
          ["SET", "s1", "123"], ["OBJECT", "ENCODING", "s1"], ["SET", "s2", "x" * 44], ["OBJECT", "ENCODING", "s2"],
          # Redis 8 keeps a value embstr only while key and value fit one cache
          # line with the object (64 bytes on Linux, 128 on Apple silicon)
          ["SET", "k3", "x" * 37], ["OBJECT", "ENCODING", "k3"], ["SET", "k4", "x" * 39], ["OBJECT", "ENCODING", "k4"],
          ["SET", "k5", "x" * 40], ["OBJECT", "ENCODING", "k5"], ["SET", "a" * 40, "hello"],
          ["OBJECT", "ENCODING", "a" * 40], ["SET", "b" * 61, "x" * 44], ["OBJECT", "ENCODING", "b" * 61],
          ["SET", "b" * 62, "x" * 44], ["OBJECT", "ENCODING", "b" * 62],
          ["SET", "s3", "x" * 45], ["OBJECT", "ENCODING", "s3"], ["SET", "s4", "0123"], ["OBJECT", "ENCODING", "s4"],
          ["HSET", "h1", "f", "v"], ["OBJECT", "ENCODING", "h1"],
          ["HSET", "h2", *sum([[f"f{i}", "v"] for i in range(129)], [])], ["OBJECT", "ENCODING", "h2"],
          ["HSET", "h3", "f", "v" * 65], ["OBJECT", "ENCODING", "h3"],
          ["SADD", "t1", "1", "2", "3"], ["OBJECT", "ENCODING", "t1"], ["SADD", "t2", "a", "b"], ["OBJECT", "ENCODING", "t2"],
          ["SADD", "t3", *[str(i) for i in range(600)]], ["OBJECT", "ENCODING", "t3"],
          ["SADD", "t4", *[f"m{i}" for i in range(129)]], ["OBJECT", "ENCODING", "t4"],
          ["ZADD", "z1", "1", "a"], ["OBJECT", "ENCODING", "z1"],
          ["ZADD", "z2", *sum([[str(i), f"m{i}"] for i in range(129)], [])], ["OBJECT", "ENCODING", "z2"],
          ["XADD", "x1", "1-1", "f", "v"], ["OBJECT", "ENCODING", "x1"]]
    return P


def resp3_probes():
    return [["CLIENT", "TRACKINGINFO"], ["ACL", "GETUSER", "default"], ["COMMAND", "INFO", "get"],
            ["LATENCY", "DOCTOR"], ["ACL", "LOG"], ["COMMAND", "GETKEYSANDFLAGS", "SET", "a", "b"],
            ["LATENCY", "LATEST"], ["MEMORY", "USAGE", "nosuch"]]


def redis_differential(binary, rs):
    print("[7] the same replies as Redis")
    s = Server(binary)
    rport = free_port_block(2)
    rdir = tempfile.mkdtemp(prefix="admin47_redis_")
    r = subprocess.Popen([rs, "--port", str(rport), "--save", "", "--appendonly", "no", "--dir", rdir],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready_pid(rport, r, 30)
        pc, rc = s.conn(), Conn(rport)
        for cmd in probes():
            a, b = pc.raw(*cmd), rc.raw(*cmd)
            check(f"RESP2 {' '.join(cmd)[:60]}", a == b, f"\n        pion : {a[:300]!r}\n        redis: {b[:300]!r}")
        # Redis lists these in its dicts' order, which changes from process to
        # process: compared without order (ACL CAT <category> as the Pion
        # commands in Redis's list; Redis 8.10 has a few Pion lacks)
        names = set(x.decode() for x in pc.cmd("COMMAND", "LIST"))
        low = lambda xs: sorted(x.lower() for x in xs if x.lower().decode() in names)
        for cat in ("string", "String", "hyperloglog", "admin", "keyspace"):
            a, b = pc.cmd("ACL", "CAT", cat), rc.cmd("ACL", "CAT", cat)
            check(f"ACL CAT {cat}", isinstance(a, list) and sorted(a) == low(b),
                  f"pion {sorted(a)!r:.200} redis {sorted(b)!r:.200}")
        a, b = pc.cmd("CONFIG", "GET", "slowlog-*"), rc.cmd("CONFIG", "GET", "slowlog-*")
        check("CONFIG GET slowlog-*", dict(zip(a[::2], a[1::2])) == dict(zip(b[::2], b[1::2])), f"{a} {b}")
        for name in ("client", "config", "xgroup"):
            a, b = pc.cmd("COMMAND", "INFO", name), rc.cmd("COMMAND", "INFO", name)
            for e in (a[0], b[0]):
                e[9] = sorted(e[9], key=lambda x: x[0])
            check(f"COMMAND INFO {name}", a == b, f"\n        pion : {a!r:.300}\n        redis: {b!r:.300}")
        for pat in ("ge*", "GE*", "x?dd", "*range*", "client*", "*|get*"):
            a, b = pc.cmd("COMMAND", "LIST", "FILTERBY", "PATTERN", pat), rc.cmd("COMMAND", "LIST", "FILTERBY", "PATTERN", pat)
            check(f"COMMAND LIST FILTERBY PATTERN {pat}", sorted(a) == low(b),
                  f"{sorted(a)} {sorted(b)}")
        p3, r3 = s.conn(protocol=3), Conn(rport)
        r3.cmd("HELLO", "3")
        for cmd in resp3_probes():
            a, b = p3.raw(*cmd), r3.raw(*cmd)
            check(f"RESP3 {' '.join(cmd)[:60]}", a == b, f"\n        pion : {a[:300]!r}\n        redis: {b[:300]!r}")
        # REPLY OFF / SKIP sequences, byte for byte
        for seq in ([["CLIENT", "REPLY", "SKIP"], ["PING"], ["ECHO", "a"]],
                    [["CLIENT", "REPLY", "OFF"], ["PING"], ["CLIENT", "REPLY", "SKIP"], ["NOPE"], ["CLIENT", "REPLY", "ON"],
                     ["PING"]],
                    [["CLIENT", "REPLY", "SKIP"], ["CLIENT", "REPLY", "ON"], ["PING"]],
                    [["CLIENT", "REPLY", "SKIP"], ["MULTI"], ["PING"], ["EXEC"]],
                    [["CLIENT", "REPLY", "OFF"], ["RESET"], ["PING"]]):
            data = b"".join(encode(c) for c in seq)
            x, y = s.conn(), Conn(rport)
            x.sock.sendall(data)
            y.sock.sendall(data)
            a, b = recv_some(x), recv_some(y)
            check(f"REPLY {' / '.join(' '.join(c) for c in seq)}", a == b, f"pion {a!r} redis {b!r}")
        # SLOWLOG entries have Redis's shape
        for c in (pc, rc):
            c.cmd("CONFIG", "SET", "slowlog-log-slower-than", "0")
            c.cmd("SLOWLOG", "RESET")
            c.cmd("RPUSH", "many", *[str(i) for i in range(40)])
            c.cmd("CONFIG", "SET", "slowlog-log-slower-than", "10000")
        ea, eb = pc.cmd("SLOWLOG", "GET", "2")[1], rc.cmd("SLOWLOG", "GET", "2")[1]
        check("SLOWLOG entry args and argc as Redis's", (ea[3], ea[6]) == (eb[3], eb[6]), f"{ea!r}\n{eb!r}")
        # UNBLOCK as Redis
        for kind in ("TIMEOUT", "ERROR"):
            outs = []
            for port in (s.port, rport):
                ctl, bl = Conn(port), Conn(port)
                cid = bl.cmd("CLIENT", "ID")
                bl.sock.sendall(encode(["BLPOP", "nolist", "0"]))
                time.sleep(0.15)
                outs.append((ctl.cmd("CLIENT", "UNBLOCK", str(cid), kind), recv_some(bl)))
            check(f"UNBLOCK {kind} as Redis", outs[0] == outs[1], repr(outs))
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
    part_client(binary)
    part_pause(binary)
    part_reply(binary)
    part_slowlog(binary)
    part_acl(binary)
    part_misc(binary)
    rs = shutil.which("redis-server")
    if rs:
        redis_differential(binary, rs)
    else:
        print("[7] SKIP: redis-server not on PATH")
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
