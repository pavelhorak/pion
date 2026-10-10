#!/usr/bin/env python3
"""#465: `--io-threads N` serves ONE keyspace correctly from many connections.

The executor (the worker thread) runs every command; N-1 I/O threads own the
sockets. What can go wrong is exactly what a single-threaded loop cannot get
wrong, so that is what this test asserts, for --io-threads 1 (today's loop, as
the control) and for 2 and 4:

  1. Per-connection reply order under pipelining, from many connections at once.
     Each connection pipelines SET/GET/INCR of its own keys and checks every
     reply against the request it answers.
  2. Cross-connection visibility of acknowledged writes. A write acked on one
     connection is read on others right away. All connections are opened
     CONCURRENTLY (#253's measurement trap: connections opened one after another
     can all land on one owner and hide a defect).
  3. Replies that do not answer a request on the same connection: pub/sub
     delivery to a subscriber that another connection's PUBLISH reaches, and a
     BLPOP that another connection's LPUSH wakes, plus what the blocked client
     pipelined behind it.
  4. MULTI/EXEC, a reply larger than the 4 MB writer buffer, QUIT (the reply
     arrives, then the close), and a client that disconnects mid-pipeline.
  5. The rest of step 2: MONITOR, CLIENT KILL of another client, CLIENT PAUSE,
     CLIENT REPLY OFF, a protocol error (answered, then closed), WAIT, XREAD
     BLOCK, a client that stops reading while 64 MB is owed to it (the others
     keep being served), pub/sub fan-out in order, RESP3, connection churn, and
     AUTH plus --maxmemory on a second server.

Linux (epoll) and macOS (kqueue).
    python3 tests/test_gh465_io_threads.py [./pion-server]
"""
import os, socket, subprocess, sys, threading, time, shutil
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, encode, wait_ready_pid, wait_port_free  # noqa: E402

BINARY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PION_BIN", "./pion-server"))
PORT = 1993
CONNS = 48
PIPE = 50

failures, passes = [], []


def check(name, cond, detail=""):
    if cond:
        passes.append(name); print(f"  PASS  {name}")
    else:
        failures.append((name, detail)); print(f"  FAIL  {name}   {detail}")


def start(io_threads, extra=(), password=None):
    wd = f"/tmp/pion_gh465_{PORT}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    wait_port_free(PORT)
    args = [BINARY, "-p", str(PORT), "-w", "1", "--no-wal", "--no-auto-detect",
            "--no-auto-embed", "--no-crash-log"]
    if io_threads > 1:
        args += ["--io-threads", str(io_threads)]
    args += list(extra)
    log = open(os.path.join(wd, "server.log"), "w")
    p = subprocess.Popen(args, cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready_pid(PORT, p, password=password)
    except Exception:
        stop(p)
        raise
    return p


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()


def order_worker(i, barrier):
    """One connection: PIPE rounds of a pipelined SET/GET/INCR on its own keys."""
    c = Conn(PORT, timeout=20)
    barrier.wait()
    bad = 0
    for r in range(PIPE // 5):
        cmds = []
        for j in range(5):
            k = f"o:{i}:{r}:{j}"
            cmds += [["SET", k, f"v{i}-{r}-{j}"], ["GET", k], ["INCR", f"ctr:{i}"]]
        replies = c.pipeline(cmds)
        for j in range(5):
            if replies[3 * j] != "OK" or replies[3 * j + 1] != f"v{i}-{r}-{j}".encode() \
               or replies[3 * j + 2] != r * 5 + j + 1:
                bad += 1
    c.close()
    return bad


def visibility_round(conns):
    """conn 0 writes a fresh value; every other conn must read it at once."""
    nils = 0
    for t in range(10):
        key, val = f"vis:{t}", f"x{t}-{time.monotonic_ns()}"
        assert conns[0].cmd("SET", key, val) == "OK"
        with ThreadPoolExecutor(len(conns) - 1) as ex:
            got = list(ex.map(lambda c: c.cmd("GET", key), conns[1:]))
        nils += sum(1 for g in got if g != val.encode())
    return nils


def run_suite(io_threads):
    print(f"\n== --io-threads {io_threads}")
    p = start(io_threads)
    try:
        # 1. order, many connections at once
        barrier = threading.Barrier(CONNS)
        with ThreadPoolExecutor(CONNS) as ex:
            bad = sum(ex.map(lambda i: order_worker(i, barrier), range(CONNS)))
        check(f"io{io_threads}: per-connection order, {CONNS} concurrent pipelined connections", bad == 0,
              f"{bad} wrong replies")

        # 2. visibility, connections opened concurrently
        with ThreadPoolExecutor(16) as ex:
            conns = list(ex.map(lambda _: Conn(PORT, timeout=10), range(16)))
        nils = visibility_round(conns)
        check(f"io{io_threads}: an acked write is visible on 15 other connections", nils == 0,
              f"{nils} stale reads of 150")
        for c in conns:
            c.close()

        # 3a. pub/sub across connections
        sub = Conn(PORT, timeout=10)
        r = sub.cmd("SUBSCRIBE", "ch465")
        pub = Conn(PORT, timeout=10)
        n = pub.cmd("PUBLISH", "ch465", "hello")
        msg = sub.read()
        check(f"io{io_threads}: PUBLISH reaches a subscriber on another connection",
              n == 1 and msg == [b"message", b"ch465", b"hello"], f"publish={n} got={msg!r} sub={r!r}")
        sub.close(); pub.close()

        # 3b. BLPOP woken by another connection; what it pipelined behind runs after
        blk = Conn(PORT, timeout=10)
        blk.sock.sendall(encode(["BLPOP", "q465", "5"]) + encode(["SET", "after465", "1"]) + encode(["GET", "after465"]))
        time.sleep(0.3)
        w = Conn(PORT, timeout=10)
        w.cmd("LPUSH", "q465", "item")
        got = [blk.read(), blk.read(), blk.read()]
        check(f"io{io_threads}: BLPOP woken by another connection, then its pipelined commands",
              got == [[b"q465", b"item"], "OK", b"1"], f"got={got!r}")
        blk.close(); w.close()

        # 4a. MULTI/EXEC
        c = Conn(PORT, timeout=10)
        rep = c.pipeline([["MULTI"], ["SET", "tx465", "a"], ["APPEND", "tx465", "b"], ["EXEC"]])
        check(f"io{io_threads}: MULTI/EXEC", rep == ["OK", "QUEUED", "QUEUED", ["OK", 2]], f"{rep!r}")
        # 4b. a reply past the 4 MB writer buffer
        big = "z" * (6 * 1024 * 1024)
        c.cmd("SET", "big465", big)
        got = c.cmd("GET", "big465")
        check(f"io{io_threads}: a 6 MB reply arrives whole", got == big.encode(), f"len={len(got or b'')}")
        c.close()
        # 4c. QUIT: +OK, then the close
        q = socket.create_connection(("127.0.0.1", PORT), timeout=5)
        q.sendall(encode(["QUIT"]))
        data = b""
        try:
            while True:
                chunk = q.recv(100)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pass
        check(f"io{io_threads}: QUIT answers +OK then closes", data == b"+OK\r\n", f"{data!r}")
        q.close()
        # 4d. a client that vanishes mid-pipeline must not disturb the others
        for _ in range(20):
            s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
            s.sendall(b"".join(encode(["SET", f"gone{i}", "x"]) for i in range(200)))
            s.close()
        c = Conn(PORT, timeout=10)
        check(f"io{io_threads}: server still serves after 20 clients vanished mid-pipeline",
              c.cmd("PING") == "PONG" and p.poll() is None, "")
        c.close()
    finally:
        stop(p)


def read_eof(sock, timeout=5.0):
    """Bytes until the server closes the connection (None: it did not close)."""
    sock.settimeout(timeout)
    data = b""
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                return data
            data += chunk
    except socket.timeout:
        return None


def run_step2(io_threads):
    print(f"\n== --io-threads {io_threads}: step 2")
    p = start(io_threads)
    try:
        # MONITOR sees another connection's command
        mon = Conn(PORT, timeout=10)
        r = mon.cmd("MONITOR")
        c = Conn(PORT, timeout=10)
        c.cmd("SET", "mon465", "v")
        line = mon.read()
        check(f"io{io_threads}: MONITOR shows another connection's command",
              r == "OK" and isinstance(line, str) and '"SET" "mon465" "v"' in line, f"{r!r} {line!r}")
        mon.close()

        # CLIENT KILL of another client: it sees the end of the stream
        victim = Conn(PORT, timeout=10)
        vid = victim.cmd("CLIENT", "ID")
        n = c.cmd("CLIENT", "KILL", "ID", str(vid))
        tail = read_eof(victim.sock)
        check(f"io{io_threads}: CLIENT KILL ID closes another connection",
              n == 1 and tail == b"" and c.cmd("PING") == "PONG", f"kill={n!r} tail={tail!r}")
        victim.close()

        # CLIENT PAUSE WRITE holds another connection's write, then runs it
        a = Conn(PORT, timeout=10)
        r = a.cmd("CLIENT", "PAUSE", "300", "WRITE")
        t0 = time.monotonic()
        w = c.cmd("SET", "pause465", "1")
        held = time.monotonic() - t0
        check(f"io{io_threads}: CLIENT PAUSE WRITE holds a write from another connection",
              r == "OK" and w == "OK" and held >= 0.2, f"{r!r} {w!r} held {held:.3f}s")
        a.close()

        # CLIENT REPLY OFF: no replies until ON
        rep = Conn(PORT, timeout=10)
        rep.sock.sendall(encode(["CLIENT", "REPLY", "OFF"]) + encode(["SET", "reply465", "1"]) +
                         encode(["CLIENT", "REPLY", "ON"]) + encode(["GET", "reply465"]))
        got = [rep.read(), rep.read()]
        rep.assert_in_sync()
        check(f"io{io_threads}: CLIENT REPLY OFF drops replies until ON", got == ["OK", b"1"], f"{got!r}")
        rep.close()

        # a protocol error is answered, then the connection is closed
        pe = socket.create_connection(("127.0.0.1", PORT), timeout=5)
        pe.sendall(b"*2\r\n$3\r\nGET\r\n$xyz\r\n")
        data = read_eof(pe)
        check(f"io{io_threads}: a protocol error is answered, then closed",
              data is not None and data.startswith(b"-ERR Protocol error"), f"{data!r}")
        pe.close()

        # WAIT with no replicas answers at once; the pipeline goes on
        r = c.pipeline([["WAIT", "0", "0"], ["PING"]])
        check(f"io{io_threads}: WAIT 0 0, then a pipelined PING", r == [0, "PONG"], f"{r!r}")

        # XREAD BLOCK woken by another connection's XADD
        rd = Conn(PORT, timeout=10)
        rd.sock.sendall(encode(["XREAD", "BLOCK", "5000", "STREAMS", "s465", "$"]) + encode(["PING"]))
        time.sleep(0.3)
        xid = c.cmd("XADD", "s465", "*", "f", "v")
        got = rd.read()
        pong = rd.read()
        ok = isinstance(got, list) and got and got[0][0] == b"s465" and got[0][1][0][0] == xid \
            and got[0][1][0][1] == [b"f", b"v"] and pong == "PONG"
        check(f"io{io_threads}: XREAD BLOCK woken by another connection", ok, f"{got!r} {pong!r}")
        rd.close()

        # a client that stops reading while 64 MB is owed; the others are served meanwhile
        mb = "x" * (1 << 20)
        c.cmd("SET", "big1m", mb)
        slow = socket.create_connection(("127.0.0.1", PORT), timeout=10)
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
        slow.sendall(b"".join(encode(["GET", "big1m"]) for _ in range(64)) + encode(["PING"]))
        time.sleep(0.3)
        t0 = time.monotonic()
        pings = [c.cmd("PING") for _ in range(200)]
        dt = time.monotonic() - t0
        sc = Conn.wrap(slow, timeout=30)
        replies = [sc.read() for _ in range(65)]
        good = sum(1 for x in replies[:64] if x == mb.encode())
        check(f"io{io_threads}: 64 MB owed to a client that stops reading; others served meanwhile",
              good == 64 and replies[64] == "PONG" and all(x == "PONG" for x in pings) and dt < 5,
              f"{good}/64 whole, others took {dt:.2f}s")
        slow.close()

        # pub/sub fan-out: 20 subscribers, 500 messages each, in order
        subs = [Conn(PORT, timeout=20) for _ in range(20)]
        for sb in subs:
            sb.cmd("SUBSCRIBE", "fan465")
        counts = c.pipeline([["PUBLISH", "fan465", f"m{i}"] for i in range(500)])
        bad = 0
        for sb in subs:
            for i in range(500):
                m = sb.read()
                if m != [b"message", b"fan465", f"m{i}".encode()]:
                    bad += 1
            sb.close()
        check(f"io{io_threads}: pub/sub fan-out, 20 subscribers x 500 messages in order",
              all(x == 20 for x in counts) and bad == 0, f"{bad} wrong, counts {set(counts)}")

        # RESP3 on one connection, RESP2 on the others
        r3 = Conn(PORT, timeout=10)
        h = r3.cmd("HELLO", "3")
        r3.cmd("HSET", "h465", "a", "1")
        m3 = r3.cmd("HGETALL", "h465")
        m2 = c.cmd("HGETALL", "h465")
        check(f"io{io_threads}: RESP3 (HELLO 3) on one connection, RESP2 on another",
              isinstance(h, dict) and m3 == {b"a": b"1"} and m2 == [b"a", b"1"], f"{m3!r} {m2!r}")
        r3.close()

        # churn: 2000 short connections from 32 threads, some leaving mid-command
        def churn(i):
            s = socket.create_connection(("127.0.0.1", PORT), timeout=10)
            try:
                if i % 3 == 0:
                    s.sendall(b"*3\r\n$3\r\nSET\r\n$4\r\nchrn")      # leaves mid-command
                    return True
                s.sendall(encode(["SET", f"churn{i}", str(i)]) + encode(["GET", f"churn{i}"]))
                cc = Conn.wrap(s, timeout=10)
                return cc.read() == "OK" and cc.read() == str(i).encode()
            finally:
                s.close()
        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(churn, range(2000)))
        check(f"io{io_threads}: 2000 short connections from 32 threads",
              all(res) and c.cmd("PING") == "PONG" and p.poll() is None, f"{res.count(False)} failed")
        c.close()
    finally:
        stop(p)


def run_auth_oom(io_threads):
    print(f"\n== --io-threads {io_threads}: AUTH and --maxmemory")
    p = start(io_threads, ["--requirepass", "pw465", "--maxmemory", "1mb"], password="pw465")
    try:
        time.sleep(0.5)            # housekeeping samples RSS
        c = Conn(PORT, timeout=10)
        noauth = c.cmd("GET", "x")
        ok = c.cmd("AUTH", "pw465")
        oom = c.cmd("SET", "x", "1")
        get = c.cmd("GET", "x")
        dele = c.cmd("DEL", "x")
        check(f"io{io_threads}: NOAUTH, AUTH, then -OOM for a write and reads still served",
              isinstance(noauth, RespError) and "NOAUTH" in noauth and ok == "OK"
              and isinstance(oom, RespError) and oom.startswith("OOM") and get is None and dele == 0,
              f"{noauth!r} {ok!r} {oom!r} {get!r} {dele!r}")
        c.close()
    finally:
        stop(p)


def main():
    for io in (1, 2, 4):
        run_suite(io)
        run_step2(io)
        run_auth_oom(io)
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for n, d in failures:
        print(f"  FAILED: {n}: {d}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
