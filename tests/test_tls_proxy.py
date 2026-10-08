#!/usr/bin/env python3
"""TLS in transit through a terminating proxy: doc/operations.md §TLS, run as written.

Pion has no TLS of its own. The documented answer is stunnel in front of a
loopback-bound server with a password. This test runs that recipe:

  1. openssl makes a throwaway CA and a server certificate for 127.0.0.1;
  2. stunnel listens with TLS on a free port and forwards to Pion's port,
     using the same config text the doc prints (filled in);
  3. clients connect over TLS and check what a deployment depends on:
     - redis-py with ssl=True and the CA: AUTH, SET/GET of a binary value,
       a 200-command pipeline, and a reply larger than one TLS record;
     - redis-cli --tls --cacert: PING;
     - a client that does not trust the CA is refused;
     - a plaintext client on the TLS port gets no Redis reply;
     - without AUTH the server answers NOAUTH, through the tunnel too.

Requires: stunnel, openssl, redis-cli. The runner starts
./pion-server -w 1 --requirepass tlstest --bind 127.0.0.1 on port 1974.
"""
import argparse
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time

import redis

FAILS = []


def check(cond, what):
    print(("  PASS  " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_certs(d):
    run = lambda *a: subprocess.run(a, cwd=d, check=True, capture_output=True)
    # The extensions a strict verifier requires (Python 3.13+ sets
    # VERIFY_X509_STRICT): a CA marked as one, and key identifiers that chain
    # the leaf to it. A bare `openssl req -x509` CA fails with "Missing
    # Authority Key Identifier".
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
        "-subj", "/CN=pion-test-ca", "-keyout", "ca.key", "-out", "ca.crt",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
        "-addext", "subjectKeyIdentifier=hash")
    run("openssl", "req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1",
        "-keyout", "server.key", "-out", "server.csr")
    with open(os.path.join(d, "san.ext"), "w") as f:
        f.write("subjectAltName=IP:127.0.0.1,DNS:localhost\n"
                "basicConstraints=critical,CA:FALSE\n"
                "keyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth\n"
                "subjectKeyIdentifier=hash\n"
                "authorityKeyIdentifier=keyid,issuer\n")
    run("openssl", "x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
        "-CAcreateserial", "-days", "2", "-extfile", "san.ext", "-out", "server.crt")
    # stunnel wants the certificate and its key; the doc keeps them in one file.
    with open(os.path.join(d, "pion.pem"), "w") as f:
        f.write(open(os.path.join(d, "server.crt")).read())
        f.write(open(os.path.join(d, "server.key")).read())
    os.chmod(os.path.join(d, "pion.pem"), 0o600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--password", default="tlstest")
    a = ap.parse_args()
    for tool in ("stunnel", "openssl", "redis-cli"):
        if not shutil.which(tool):
            print(f"SKIP: {tool} not installed")
            return 0

    d = tempfile.mkdtemp(prefix="pion-tls-")
    proc = None
    try:
        make_certs(d)
        tls_port = free_port()
        conf = os.path.join(d, "stunnel.conf")
        # The doc's config, with its placeholders filled in.
        with open(conf, "w") as f:
            f.write(f"""foreground = yes
pid =
[pion]
accept = 127.0.0.1:{tls_port}
connect = 127.0.0.1:{a.port}
cert = {d}/pion.pem
sslVersionMin = TLSv1.2
""")
        log = open(os.path.join(d, "stunnel.log"), "w")
        proc = subprocess.Popen(["stunnel", conf], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", tls_port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)

        ca = os.path.join(d, "ca.crt")
        r = redis.Redis(host="127.0.0.1", port=tls_port, password=a.password,
                        ssl=True, ssl_ca_certs=ca, ssl_cert_reqs="required", socket_timeout=10)
        check(r.ping() is True, "PING over TLS with AUTH")
        blob = os.urandom(1 << 20)                       # several TLS records
        check(r.set("tls:blob", blob) is True, "SET of a 1 MiB binary value over TLS")
        check(r.get("tls:blob") == blob, "GET returns the 1 MiB value byte for byte")
        p = r.pipeline(transaction=False)
        for i in range(200):
            p.set(f"tls:k{i}", i)
        for i in range(200):
            p.get(f"tls:k{i}")
        out = p.execute()
        check(out[:200] == [True] * 200 and out[200:] == [str(i).encode() for i in range(200)],
              "a 400-command pipeline keeps every reply paired with its request")
        r.delete("tls:blob", *[f"tls:k{i}" for i in range(200)])

        cli = subprocess.run(["redis-cli", "--tls", "--cacert", ca, "-h", "127.0.0.1",
                              "-p", str(tls_port), "-a", a.password, "--no-auth-warning", "PING"],
                             capture_output=True, text=True, timeout=10)
        check(cli.stdout.strip() == "PONG", f"redis-cli --tls PING (got {cli.stdout.strip()!r} {cli.stderr.strip()!r})")

        # A client that does not trust this CA must fail the handshake.
        refused = False
        try:
            redis.Redis(host="127.0.0.1", port=tls_port, password=a.password, ssl=True,
                        ssl_cert_reqs="required", ssl_ca_certs=None, socket_timeout=5).ping()
        except (redis.ConnectionError, ssl.SSLError):
            refused = True
        check(refused, "a client that does not trust the CA is refused")

        # Plaintext RESP on the TLS port: stunnel answers with a TLS alert or
        # closes; it must never forward the command to Pion.
        s = socket.create_connection(("127.0.0.1", tls_port), timeout=5)
        s.sendall(b"*1\r\n$4\r\nPING\r\n")
        try:
            got = s.recv(64)
        except OSError:
            got = b""
        s.close()
        check(not got.startswith(b"+PONG") and not got.startswith(b"-NOAUTH"),
              "plaintext RESP on the TLS port reaches no Redis reply")

        # The password still applies through the tunnel.
        noauth = False
        try:
            redis.Redis(host="127.0.0.1", port=tls_port, ssl=True, ssl_ca_certs=ca,
                        socket_timeout=5).set("tls:x", 1)
        except redis.AuthenticationError:
            noauth = True
        except redis.ResponseError as e:
            noauth = "NOAUTH" in str(e)
        check(noauth, "without AUTH the server refuses, through the tunnel too")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(d, ignore_errors=True)
    print(f"{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
