"""Graduation: the embedded engine and
pion-server open the same directory, one at a time, never both.

  python embedded/tests/test_graduation.py <pion-server binary>
"""
import os, subprocess, sys, tempfile, time
import redis

SERVER = os.path.abspath(sys.argv[1])
PY = sys.executable
D = tempfile.mkdtemp(prefix="graduate_")
CWD = tempfile.mkdtemp(prefix="graduate_cwd_")
PORT = 16795
fails = []

def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  {detail}"))
    if not ok: fails.append(name)

def embedded(code):
    return subprocess.run([PY, "-c", "import pion_embedded as pion\n" + code], cwd=CWD,
                          capture_output=True, text=True, timeout=120)

def start_server():
    return subprocess.Popen([SERVER, "--dir", D, "-p", str(PORT), "-w", "1", "--no-auto-detect",
                             "--no-auto-embed", "--no-crash-log"], cwd=CWD,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def wait_ping():
    r = redis.Redis(port=PORT, decode_responses=True)
    for _ in range(100):
        try:
            if r.ping(): return r
        except redis.ConnectionError:
            time.sleep(0.1)
    raise SystemExit("server did not start")

# 1. the app writes, and holds the directory
holder = subprocess.Popen([PY, "-c", f"""
import pion_embedded as pion, sys, time
r = pion.Redis(dir={D!r}, decode_responses=True)
for i in range(1000): r.set(f"k:{{i}}", i)
r.rpush("l", *range(100)); r.hset("h", mapping={{"a": 1, "b": 2}}); r.zadd("z", {{"m": 3.5}})
r.xadd("s", {{"f": "v"}}); r.set("ttl", "x", ex=3600)
print("ready", flush=True); sys.stdin.readline()
pion.close()
print("closed", flush=True)
"""], cwd=CWD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
check("embedded app wrote", holder.stdout.readline().strip() == "ready")
p = start_server(); out, _ = p.communicate(timeout=60)
check("pion-server --dir refused while the app holds it", p.returncode == 1 and "in use" in out, out[-300:])
holder.stdin.write("\n"); holder.stdin.flush()
check("embedded app closed", holder.stdout.readline().strip() == "closed"); holder.wait(timeout=30)

# 2. the server serves what the app wrote
p = start_server(); r = wait_ping()
check("1000 strings", all(r.get(f"k:{i}") == str(i) for i in range(1000)))
check("list", r.lrange("l", 0, -1) == [str(i) for i in range(100)])
check("hash", r.hgetall("h") == {"a": "1", "b": "2"})
check("zset", r.zscore("z", "m") == 3.5)
check("stream", r.xlen("s") == 1)
check("ttl kept", 3500 < r.ttl("ttl") <= 3600, r.ttl("ttl"))
# 3. the library is refused while the server runs
e = embedded(f"pion.Redis(dir={D!r}).ping()")
check("embedded open refused while the server runs", e.returncode != 0 and "in use" in e.stderr, e.stderr[-300:])
# 4. the server writes, then shuts down; the app sees it
r.set("from-server", "hello"); r.incr("k:0")
try: r.shutdown()
except redis.ConnectionError: pass
p.wait(timeout=30)
e = embedded(f"r = pion.Redis(dir={D!r}, decode_responses=True); print(r.get('from-server'), r.get('k:0'), r.get('k:999'))")
# (the engine still prints its log lines to stdout, e.g. "Snapshot: loaded":
# a known gap, embedded/README.md; read the last line)
check("the app reads the server's writes", e.stdout.strip().splitlines()[-1].split() == ["hello", "1", "999"],
      e.stdout + e.stderr[-300:])
# 5. files went to --dir, not to either working directory
check("no files in the working directory", os.listdir(CWD) == [], os.listdir(CWD))
check("WAL in the data directory", "pion.wal.0" in os.listdir(D), os.listdir(D))
print(f"{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
