#!/usr/bin/env python3
"""pion-context's codebase index against a live server: what its README says.

Every check here failed on the pion-context that shipped through 0.9.5:

1. Completeness. Files are git's (tracked + untracked-not-ignored), so a
   source directory is never dropped for its name. SKIP_DIRS used to skip
   every directory named `models`, which is where Django keeps its ORM.
2. Re-indexing a built index. Pion builds an index once (ingest ->
   FT.OPTIMIZE -> search); a vector written after FT.OPTIMIZE never enters
   the graph. The PostToolUse hook re-indexed an edited file that way, and the
   file vanished from search. It now rebuilds from the stored vectors. The
   hook gets Claude Code's ABSOLUTE path, which has to replace the chunks
   stored under the indexed root's relative path, not add a second copy.
3. A deleted file's chunks leave the index on the next `index`.
4. An emptied file's chunks leave too.
5. `install-hooks` writes the hooks; they did not exist ("done automatically").
6. Ollama embeddings go in batches (one /api/embed request per 64 chunks, not
   one request per chunk) and are the same vectors as the one-text path.
7. A rebuild that would lose chunks (no stored vector) refuses before it
   drops anything.
8. A search that cannot run raises, naming why. When another FT index
   replaces the codebase index (Pion serves one at a time), search used to
   answer [] — indistinguishable from "nothing relevant".

No model or network: a fake Ollama in this process serves /api/embed and
/api/embeddings with a hashed bag-of-words embedding, so "salted password
hash" finds the password code. Starts its own ./pion-server.

Usage: python3 tests/test_pion_context_index.py [--port 6436]   (exit 0 = pass, 1 = fail, 2 = skipped)
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ── a fake Ollama ────────────────────────────────────────────────────────────

DIM = 768
REQUESTS = {"embed": 0, "embeddings": 0}


def bag_of_words(text, scale=1.0):
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 64) & 1 else -1.0
    return [x * scale for x in v]


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path == "/api/embed":            # batched, unit-length vectors, as Ollama does
            REQUESTS["embed"] += 1
            out = {"embeddings": [unit(bag_of_words(t)) for t in body["input"]]}
        elif self.path == "/api/embeddings":     # one text, raw (unnormalized) vector
            REQUESTS["embeddings"] += 1
            out = {"embedding": bag_of_words(body["prompt"], scale=3.7)}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ── helpers ──────────────────────────────────────────────────────────────────

def fresh_server(port):
    workdir = tempfile.mkdtemp(prefix="pion_ctx_index_")
    binary = os.environ.get("PION_BIN", os.path.join(REPO, "pion-server"))
    proc = subprocess.Popen([binary, "-p", str(port), "-w", "1", "--no-auto-detect", "--no-auto-embed",
                             "--no-crash-log", "--no-wal"],
                            cwd=workdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.workdir = workdir
    import redis
    for _ in range(150):
        try:
            redis.Redis(port=port).ping()
            return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("server did not start")


def git(repo, *args):
    subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
                   check=True, capture_output=True)


def write(repo, rel, text):
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def make_repo():
    repo = tempfile.mkdtemp(prefix="pion_ctx_repo_")
    write(repo, ".gitignore", "build/\n")
    write(repo, "pkg/models/user.py", 'import hashlib\nimport os\n\n\nclass User:\n    def set_password(self, raw):\n'
          '        """Salted password hash stored instead of the raw password."""\n'
          '        salt = os.urandom(16)\n        self.password_hash = salt + hashlib.sha256(salt + raw.encode()).digest()\n')
    write(repo, "pkg/notify.py", 'import smtplib\n\n\ndef send_email(to, subject, body):\n'
          '    """Send an email notification through the local smtp relay."""\n'
          '    with smtplib.SMTP("localhost") as s:\n        s.sendmail("noreply@example.com", [to], body)\n')
    write(repo, "pkg/config.py", 'import json\n\n\ndef parse_config(path):\n'
          '    """Read the settings json file and fill in defaults for missing keys."""\n'
          '    with open(path) as f:\n        data = json.load(f)\n    data.setdefault("timeout", 30)\n    return data\n')
    write(repo, "build/out.py", 'GENERATED = "ignoredgenerated artifact"\n\n\n\n\n')
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    write(repo, "pkg/fresh.py", '"""A fresh untracked module about telescope calibration."""\n\n\n'
          'def calibrate_telescope(mirror):\n    return mirror * 2\n')   # untracked, not ignored
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6436)
    args = ap.parse_args()
    try:
        import redis  # noqa: F401
        import requests  # noqa: F401
    except ImportError as e:
        print(f"SKIP: {e}")
        return 2
    if not shutil.which("git"):
        print("SKIP: git not found")
        return 2

    fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    env = {"PION_EMBED_PROVIDER": "ollama", "PION_OLLAMA_URL": f"http://127.0.0.1:{fake.server_port}",
           "PION_EMBED_DIM": "1536", "PION_PORT": str(args.port)}
    os.environ.update(env)
    sys.path.insert(0, os.path.join(REPO, "pion_context"))
    from pion_context import embeddings as em
    from pion_context import indexer as ix
    from pion_context.engine import ContextEngine
    cli_env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "pion_context"))

    repo = make_repo()
    proc = fresh_server(args.port)
    cwd = os.getcwd()
    try:
        import redis
        r = redis.Redis(port=args.port)
        os.chdir(repo)
        idx = ix.CodebaseIndexer(port=args.port)
        eng = ContextEngine(port=args.port)

        def top(q, k=3):
            return [x.file_path for x in eng.search_codebase(q, k=k)]

        def stored_paths():
            return sorted({r.hget(k, "file_path").decode() for k in r.scan_iter("cb:*")})

        print("[1] completeness: git decides, no directory is skipped for its name")
        stats = idx.index_directory(".")
        paths = stored_paths()
        check("a file under a directory named `models` is indexed", "./pkg/models/user.py" in paths, str(paths))
        check("an untracked, not ignored file is indexed", "./pkg/fresh.py" in paths)
        check("a git-ignored file is not indexed", "./build/out.py" not in paths)
        check("'salted password hash' finds the models file first", top("salted password hash")[:1] == ["./pkg/models/user.py"],
              str(top("salted password hash")))
        check("no indexing errors", not stats.errors, str(stats.errors))

        print("[2] the PostToolUse hook re-indexes an edited file into the BUILT index")
        notify = os.path.join(repo, "pkg/notify.py")
        with open(notify, "a") as f:
            f.write('\n\ndef render_invoice_pdf(order):\n    """Render the order invoice as a pdf document."""\n'
                    '    return b"%PDF" + str(order).encode()\n')
        hook_in = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": notify}})   # absolute, as Claude Code sends
        subprocess.run([sys.executable, "-m", "pion_context.cli", "--port", str(args.port), "hook-reindex"],
                       input=hook_in, text=True, env=cli_env, cwd=repo, check=False, timeout=120)
        check("the edited file's new code is found", top("render the invoice as a pdf")[:1] == ["./pkg/notify.py"],
              str(top("render the invoice as a pdf")))
        check("the edited file's old code is still found", top("send an email notification smtp")[:1] == ["./pkg/notify.py"],
              str(top("send an email notification smtp")))
        check("untouched files are still found", top("settings json defaults missing keys")[:1] == ["./pkg/config.py"],
              str(top("settings json defaults missing keys")))
        paths = stored_paths()
        check("the absolute hook path replaced the chunks, no second copy", notify not in paths and "./pkg/notify.py" in paths, str(paths))
        with open(notify) as f:
            want = len(ix.chunk_file("./pkg/notify.py", f.read()))
        have = sum(1 for k in r.scan_iter("cb:*") if r.hget(k, "file_path") == b"./pkg/notify.py")
        check("the file has exactly its current chunks", have == want, f"{have} stored, {want} expected")

        print("[3] a deleted file leaves the index")
        os.remove(os.path.join(repo, "pkg/config.py"))
        idx.index_directory(".")
        check("its chunks are gone", "./pkg/config.py" not in stored_paths(), str(stored_paths()))
        check("search no longer returns it", "./pkg/config.py" not in top("settings json defaults missing keys", k=10))
        check("the rest is still found", top("salted password hash")[:1] == ["./pkg/models/user.py"])

        print("[4] an emptied file leaves the index")
        open(os.path.join(repo, "pkg/fresh.py"), "w").close()
        idx.index_file("./pkg/fresh.py")
        check("its chunks are gone", "./pkg/fresh.py" not in stored_paths(), str(stored_paths()))

        print("[5] install-hooks writes the Claude Code hooks")
        proj = tempfile.mkdtemp(prefix="pion_ctx_hooks_")
        write(proj, ".claude/settings.json", json.dumps({"model": "keep-me", "hooks": {"PostToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo other-hook"}]}]}}))
        rcs = []
        for _ in range(2):   # twice: must not duplicate
            rcs.append(subprocess.run([sys.executable, "-m", "pion_context.cli", "--port", str(args.port), "install-hooks",
                                       "--project-dir", proj], env=cli_env, capture_output=True, timeout=60).returncode)
        check("`pion-context install-hooks` exists and succeeds", rcs == [0, 0], f"exit codes {rcs}")
        with open(os.path.join(proj, ".claude/settings.json")) as f:
            st = json.load(f)
        st.setdefault("hooks", {})
        ours = lambda ev: [e for e in st["hooks"].get(ev, []) if "pion_context.cli" in json.dumps(e)]
        check("existing settings are kept", st.get("model") == "keep-me" and "echo other-hook" in json.dumps(st))
        check("one SessionStart and one PostToolUse hook, after two runs", len(ours("SessionStart")) == 1 and len(ours("PostToolUse")) == 1,
              json.dumps(st["hooks"])[:200])
        cmd = json.dumps(ours("PostToolUse"))
        check("the hook runs the installing Python, so it can import pion_context",
              sys.executable in cmd and "hook-reindex" in cmd, cmd[:200])
        shutil.rmtree(proj, ignore_errors=True)

        print("[6] batched Ollama embeddings: fewer requests, the same vectors")
        texts = [f"chunk number {i} about topic {i % 7}" for i in range(130)]
        REQUESTS.update(embed=0, embeddings=0)
        batched = em.embed_texts(texts)
        n_batch = REQUESTS["embed"]
        single = [em.embed_text(t) for t in texts[:20]]
        check("130 texts cost 3 /api/embed requests, not 130", n_batch == 3, f"{n_batch} requests")
        cos = []
        import struct
        for a, b in zip(batched[:20], single):
            va, vb = struct.unpack(f"{len(a) // 4}f", a), struct.unpack(f"{len(b) // 4}f", b)
            cos.append(sum(x * y for x, y in zip(va, vb)))
        check("batched vectors equal the one-text path", min(cos) > 0.999999, f"min cosine {min(cos):.7f}")

        print("[7] a rebuild that would lose chunks refuses before dropping anything")
        r.hset("cb:999999", mapping={"text": "no vector here", "file_path": "./x.py"})
        try:
            idx.rebuild()
            refused = False
        except RuntimeError:
            refused = True
        check("rebuild refuses", refused)
        check("the index is still searchable after the refusal", top("salted password hash")[:1] == ["./pkg/models/user.py"])
        r.delete("cb:999999")

        print("[8] a replaced index is an error, not an empty result")
        from pion_context.engine import SearchError
        r.execute_command("FT.CREATE", "other", "SCHEMA", "v", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32",
                          "DIM", "1536", "DISTANCE_METRIC", "COSINE")
        r.hset("o:1", mapping={"v": b"\x00" * 6144})
        r.execute_command("FT.OPTIMIZE", "other")
        try:
            eng.search_codebase("salted password hash")
            err = ""
        except SearchError as e:
            err = str(e)
        check("search raises SearchError naming the index that replaced it", "other" in err, err[:160])
        cli = subprocess.run([sys.executable, "-m", "pion_context.cli", "--port", str(args.port), "search", "password"],
                             env=cli_env, capture_output=True, text=True, timeout=60)
        check("`pion-context search` exits 1 with the error", cli.returncode == 1 and "other" in cli.stderr,
              f"rc={cli.returncode} {cli.stderr[:120]}")
    finally:
        os.chdir(cwd)
        proc.kill(); proc.wait()
        shutil.rmtree(proc.workdir, ignore_errors=True)
        shutil.rmtree(repo, ignore_errors=True)
        fake.shutdown()

    print(f"\n{'FAIL: ' + ', '.join(FAIL) if FAIL else 'ALL PASS'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
