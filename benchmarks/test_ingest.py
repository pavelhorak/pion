#!/usr/bin/env python3
"""Minimal ingest pipeline test: FT.CREATE + 100 HSETs + FT.OPTIMIZE + FT.SEARCH"""
import socket
import struct
import numpy as np
import time

HOST = "127.0.0.1"
PORT = 6395
DIM = 1536

def send_cmd(s, *args):
    cmd = f"*{len(args)}\r\n"
    for a in args:
        if isinstance(a, bytes):
            cmd_bytes = cmd.encode() + f"${len(a)}\r\n".encode() + a + b"\r\n"
            s.sendall(cmd_bytes)
            return
        else:
            a_enc = str(a).encode()
            cmd += f"${len(a_enc)}\r\n{a_enc.decode()}\r\n"
    s.sendall(cmd.encode())

def send_cmd_raw(s, *args):
    parts = [f"*{len(args)}\r\n".encode()]
    for a in args:
        if isinstance(a, bytes):
            parts.append(f"${len(a)}\r\n".encode())
            parts.append(a)
            parts.append(b"\r\n")
        else:
            a_enc = str(a).encode()
            parts.append(f"${len(a_enc)}\r\n".encode())
            parts.append(a_enc)
            parts.append(b"\r\n")
    s.sendall(b"".join(parts))

def read_line(s):
    buf = b""
    while not buf.endswith(b"\r\n"):
        c = s.recv(1)
        if not c:
            break
        buf += c
    return buf.decode().strip()

def read_response(s):
    line = read_line(s)
    if not line:
        return None
    if line[0] == '+':
        return line[1:]
    elif line[0] == '-':
        return f"ERR: {line[1:]}"
    elif line[0] == ':':
        return int(line[1:])
    elif line[0] == '$':
        n = int(line[1:])
        if n == -1:
            return None
        data = b""
        while len(data) < n + 2:
            data += s.recv(n + 2 - len(data))
        return data[:n].decode(errors='replace')
    elif line[0] == '*':
        n = int(line[1:])
        if n == -1:
            return None
        return [read_response(s) for _ in range(n)]
    return line

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.settimeout(30)

    print("=== Step 1: FT.CREATE ===")
    send_cmd_raw(s, "FT.CREATE", "index", "ON", "HASH", "SCHEMA",
                 "id", "TAG", "metadata", "NUMERIC",
                 "vector", "VECTOR", "HNSW", "10",
                 "TYPE", "FLOAT32", "DIM", str(DIM),
                 "DISTANCE_METRIC", "COSINE",
                 "M", "16", "EF_CONSTRUCTION", "128")
    r = read_response(s)
    print(f"FT.CREATE response: {r}")

    print("\n=== Step 2: Insert 100 vectors via HSET ===")
    t0 = time.time()
    for i in range(100):
        vec = np.random.rand(DIM).astype(np.float32)
        vec_bytes = vec.tobytes()
        send_cmd_raw(s, "HSET", str(i),
                     "id", str(i),
                     "metadata", str(i),
                     "vector", vec_bytes)
        r = read_response(s)
        if i == 0 or i == 99:
            print(f"  HSET {i}: {r}")
    t1 = time.time()
    print(f"  Inserted 100 vectors in {t1-t0:.2f}s")

    print("\n=== Step 3: FT.OPTIMIZE ===")
    send_cmd_raw(s, "FT.OPTIMIZE", "index")
    r = read_response(s)
    print(f"FT.OPTIMIZE response: {r}")

    print("\n=== Step 4: FT.SEARCH ===")
    query_vec = np.random.rand(DIM).astype(np.float32).tobytes()
    send_cmd_raw(s, "FT.SEARCH", "index",
                 "*=>[KNN 5 @vector $vec EF_RUNTIME 50 as score]",
                 "PARAMS", "2", "vec", query_vec,
                 "RETURN", "1", "id",
                 "SORTBY", "score",
                 "LIMIT", "0", "5",
                 "DIALECT", "2")
    r = read_response(s)
    print(f"FT.SEARCH response: {r}")

    s.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
