#!/usr/bin/env python3
"""FT.SEARCH, FT.HYBRID and FT.INFO answer in RediSearch's RESP3 shape on a RESP3 connection.

redis-py 8 negotiates RESP3 by default, and its search module then reads
FT.SEARCH and FT.INFO replies as RESP3 maps. Pion sent its RESP2 arrays on
every connection, so with redis-py's default settings `r.ft().search()` and
`r.ft().info()` failed inside redis-py (AttributeError). Pion's own tests pin
protocol=2, which is how it went unnoticed.

The expected shapes are RediSearch's own replies (redis-stack 7.4):
  FT.SEARCH  %5 attributes *0 / format +STRING / results *N / total_results :N / warning *0,
             each result %3 id $key / extra_attributes %M (field -> value) / values *0
  FT.INFO    a map; num_docs is an integer

Checks:
  - RESP3 FT.SEARCH (KNN, BM25, an index not built yet) and FT.HYBRID: that map,
    exactly, with one result map per document and its fields under
    extra_attributes (a document without an `id` field carries only `score`);
  - RESP3 FT.INFO: a map, num_docs an integer;
  - RESP2 replies keep their array shape;
  - a RESP3 FT.SEARCH with a PING pipelined behind it: exactly two replies;
  - with redis-py >= 6 installed, its default connection: ft().search() and
    ft().info() work.

    python3 tests/test_resp3_ft_replies.py [--port 1974]
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys

FAILS: list[str] = []
DOCS = {  # key -> (vector, id field or None, body)
    "r3doc:0": ((1, 0, 0, 0), "0", "alpha bravo"),
    "r3doc:1": ((0, 1, 0, 0), "1", "alpha charlie"),
    "r3doc:2": ((0, 0, 1, 0), "2", "delta echo"),
    "r3doc:3": ((1, 1, 0, 0), None, "alpha foxtrot"),
    "r3doc:4": ((0, 1, 1, 0), None, "golf hotel"),
    "r3doc:5": ((1, 0, 1, 0), None, "alpha india"),
}
QUERY = struct.pack("4f", 1, 0.1, 0, 0)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class Conn:
    """A strict RESP2/RESP3 reader: every reply is parsed to its exact end, so a
    header that declares the wrong count desynchronises the next read."""

    def __init__(self, port: int, proto: int):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.buf = b""
        if proto == 3:
            hello = self.cmd("HELLO", 3)
            assert hello[0] == "%", f"HELLO 3 did not answer a map: {hello!r}"

    def write(self, *cmds) -> None:
        out = b""
        for args in cmds:
            out += b"*%d\r\n" % len(args)
            for a in args:
                a = a if isinstance(a, bytes) else str(a).encode()
                out += b"$%d\r\n%s\r\n" % (len(a), a)
        self.s.sendall(out)

    def cmd(self, *args):
        self.write(args)
        return self.read()

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("server closed the connection")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n + 2:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("server closed the connection")
            self.buf += chunk
        if self.buf[n:n + 2] != b"\r\n":
            raise ValueError(f"bulk of {n} bytes not followed by CRLF: {self.buf[:n + 8]!r}")
        data, self.buf = self.buf[:n], self.buf[n + 2:]
        return data

    def read(self):
        """(type, value): '%' -> list of (k, v) pairs, '*' -> list, '$' -> bytes,
        ':' -> int, '+'/'-' -> str, ',' -> str, '_' -> None."""
        line = self._line()
        t, rest = chr(line[0]), line[1:]
        if t in "+-,":
            return (t, rest.decode())
        if t == ":":
            return (":", int(rest))
        if t == "_":
            return ("_", None)
        if t == "$":
            n = int(rest)
            return ("$", None if n < 0 else self._exact(n))
        if t in "*~":
            n = int(rest)
            return ("*", None if n < 0 else [self.read() for _ in range(n)])
        if t == "%":
            return ("%", [(self.read(), self.read()) for _ in range(int(rest))])
        raise ValueError(f"unexpected RESP type in {line!r}")

    def pending(self) -> bytes:
        """Bytes the server sent beyond the replies read so far."""
        self.s.settimeout(0.3)
        try:
            extra = self.s.recv(65536)
        except socket.timeout:
            extra = b""
        self.s.settimeout(10)
        return self.buf + extra


def map_keys(node) -> list:
    return [k[1] for k, _ in node[1]] if node[0] == "%" else []


def map_get(node, key: str):
    for k, v in node[1]:
        if k[1] == key:
            return v
    return None


def check_search_map(name: str, reply, expect_n: int | None = None) -> list:
    """Check one RESP3 FT.SEARCH-shaped reply; return its result maps."""
    if reply[0] == "-":
        check(f"{name}: answers", False, reply[1])
        return []
    check(f"{name}: a map", reply[0] == "%", f"got type {reply[0]!r}")
    if reply[0] != "%":
        return []
    keys = map_keys(reply)
    check(f"{name}: keys are RediSearch's, in its order",
          keys == ["attributes", "format", "results", "total_results", "warning"], str(keys))
    check(f"{name}: keys are simple strings", all(k[0] == "+" for k, _ in reply[1]))
    check(f"{name}: attributes is an empty array", map_get(reply, "attributes") == ("*", []))
    check(f"{name}: format is +STRING", map_get(reply, "format") == ("+", "STRING"))
    check(f"{name}: warning is an empty array", map_get(reply, "warning") == ("*", []))
    results = map_get(reply, "results")
    total = map_get(reply, "total_results")
    ok = results is not None and results[0] == "*"
    check(f"{name}: results is an array", ok, repr(results)[:80])
    if not ok:
        return []
    docs = results[1]
    check(f"{name}: total_results is an integer equal to the result count",
          total == (":", len(docs)), f"total={total!r} results={len(docs)}")
    if expect_n is not None:
        check(f"{name}: {expect_n} results", len(docs) == expect_n, str(len(docs)))
    for d in docs:
        if not (d[0] == "%" and map_keys(d) == ["id", "extra_attributes", "values"]):
            check(f"{name}: each result is %3 id / extra_attributes / values", False, repr(d)[:120])
            return []
        if map_get(d, "values") != ("*", []):
            check(f"{name}: values is an empty array", False, repr(map_get(d, "values")))
            return []
    check(f"{name}: each result is %3 id / extra_attributes / values", True)
    return docs


def doc_fields(d) -> tuple[str, dict]:
    key = map_get(d, "id")[1].decode()
    attrs = map_get(d, "extra_attributes")
    return key, {k[1].decode(): v[1].decode() for k, v in attrs[1]} if attrs[0] == "%" else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    port = ap.parse_args().port

    c3 = Conn(port, 3)
    c2 = Conn(port, 2)
    c3.cmd("FT.DROPINDEX", "idx")
    created = c3.cmd("FT.CREATE", "idx", "ON", "HASH", "SCHEMA", "id", "TAG", "body", "TEXT",
                     "vector", "VECTOR", "HNSW", 6, "TYPE", "FLOAT32", "DIM", 4,
                     "DISTANCE_METRIC", "L2")
    check("FT.CREATE", created == ("+", "OK"), repr(created))
    c3.cmd("DEL", *DOCS)  # a document's fields must be only the ones written here
    for key, (vec, idv, body) in DOCS.items():
        fields = ["vector", struct.pack("4f", *vec), "body", body]
        if idv is not None:
            fields += ["id", idv]
        c3.cmd("HSET", key, *fields)

    knn = ["FT.SEARCH", "idx", "*=>[KNN 6 @vector $vec EF_RUNTIME 10 AS score]",
           "PARAMS", 2, "vec", QUERY, "SORTBY", "score", "RETURN", 2, "id", "score",
           "LIMIT", 0, 6, "DIALECT", 2]

    print("RESP3, index created but not built")
    check_search_map("unbuilt FT.SEARCH", c3.cmd(*knn), expect_n=0)

    optimized = c3.cmd("FT.OPTIMIZE", "idx")
    check("FT.OPTIMIZE", optimized == ("+", "OK"), repr(optimized))

    print("RESP3 FT.SEARCH KNN")
    docs = check_search_map("KNN", c3.cmd(*knn), expect_n=len(DOCS))
    got = dict(doc_fields(d) for d in docs)
    check("KNN: every document key comes back as its id", sorted(got) == sorted(DOCS), str(sorted(got)))
    for key, (_, idv, _) in DOCS.items():
        f = got.get(key)
        want = ["id", "score"] if idv is not None else ["score"]
        if f is None or sorted(f) != want or (idv is not None and f.get("id") != idv):
            check(f"KNN: {key} fields", False, f"got {f!r}, want keys {want}")
            break
        try:
            float(f["score"])
        except ValueError:
            check(f"KNN: {key} score is a number", False, f["score"])
            break
    else:
        check("KNN: fields are the doc's own (id only where the hash has one) plus score", True)

    print("RESP3 FT.SEARCH BM25")
    bm = check_search_map("BM25", c3.cmd("FT.SEARCH", "idx", "BM25", "alpha", "K", 3))
    check("BM25: has results", len(bm) > 0)

    print("RESP3 FT.HYBRID")
    hy = check_search_map("FT.HYBRID", c3.cmd("FT.HYBRID", "idx", "alpha", QUERY, "K", 3))
    check("FT.HYBRID: has results", len(hy) > 0)

    print("RESP3 FT.INFO")
    info = c3.cmd("FT.INFO", "idx")
    check("FT.INFO: a map", info[0] == "%", repr(info)[:120])
    if info[0] == "%":
        check("FT.INFO: index_name", map_get(info, "index_name") in (("$", b"idx"), ("+", "idx")),
              repr(map_get(info, "index_name")))
        check("FT.INFO: num_docs is an integer", map_get(info, "num_docs") == (":", len(DOCS)),
              repr(map_get(info, "num_docs")))
    check("FT.INFO unknown index is an error", c3.cmd("FT.INFO", "nosuch")[0] == "-")

    print("RESP3 pipelining")
    c3.write(knn, ["PING"])
    check_search_map("pipelined KNN", c3.read(), expect_n=len(DOCS))
    pong = c3.read()
    check("PING behind FT.SEARCH answers PONG", pong == ("+", "PONG"), repr(pong))
    extra = c3.pending()
    check("nothing after the two replies", extra == b"", repr(extra[:80]))

    print("RESP2 unchanged")
    r2 = c2.cmd(*knn)
    ok = r2[0] == "*" and r2[1][0] == (":", len(DOCS)) and len(r2[1]) == 1 + 2 * len(DOCS)
    check("RESP2 FT.SEARCH: [count, key, fields, ...]", ok, repr(r2)[:120])
    if ok:
        check("RESP2 FT.SEARCH: fields are flat arrays",
              all(r2[1][i][0] == "$" and r2[1][i + 1][0] == "*" for i in range(1, len(r2[1]), 2)))
    i2 = c2.cmd("FT.INFO", "idx")
    check("RESP2 FT.INFO: flat array of 4", i2[0] == "*" and len(i2[1]) == 4
          and i2[1][2] == ("$", b"num_docs"), repr(i2)[:120])

    try:
        import redis
        major = int(redis.__version__.split(".")[0])
    except ImportError:
        major = 0
    if major >= 6:
        from redis.commands.search.query import Query
        print(f"redis-py {redis.__version__}, default connection")
        r = redis.Redis(port=port)
        try:
            res = r.ft("idx").search(
                Query("*=>[KNN 3 @vector $vec AS score]").sort_by("score")
                .return_fields("id", "score").paging(0, 3).dialect(2), {"vec": QUERY})
            check("redis-py ft().search()", res.total == 3 and len(res.docs) == 3,
                  f"total={res.total} docs={len(res.docs)}")
            info = r.ft("idx").info()
            check("redis-py ft().info()", int(info["num_docs"]) == len(DOCS), repr(info))
        except Exception as e:  # noqa: BLE001 — the failure IS the finding
            check("redis-py search module on its default connection", False, f"{type(e).__name__}: {e}")
    else:
        print("  (redis-py < 6 here: it speaks RESP2 only; the raw RESP3 checks above are the contract)")

    c3.cmd("FT.DROPINDEX", "idx")
    print(f"\n{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failure(s)")
    for f in FAILS:
        print(f"  - {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
