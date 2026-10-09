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

Linux only (the prototype runs on epoll).
    python3 tests/test_gh465_io_threads.py [./pion-server]
"""
import os, socket, subprocess, sys, threading, time, shutil
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, encode, wait_ready_pid, wait_port_free  # noqa: E402

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


def start(io_threads):
    wd = f"/tmp/pion_gh465_{PORT}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    wait_port_free(PORT)
    args = [BINARY, "-p", str(PORT), "-w", "1", "--epoll", "--no-wal", "--no-auto-detect",
            "--no-auto-embed", "--no-crash-log"]
    if io_threads > 1:
        args += ["--io-threads", str(io_threads)]
    log = open(os.path.join(wd, "server.log"), "w")
    p = subprocess.Popen(args, cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    wait_ready_pid(PORT, p)
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


def main():
    if not sys.platform.startswith("linux"):
        print("SKIP: --io-threads is Linux-only (epoll) for now")
        return 0
    for io in (1, 2, 4):
        run_suite(io)
    print(f"\n{len(passes)} passed, {len(failures)} failed")
    for n, d in failures:
        print(f"  FAILED: {n}: {d}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
