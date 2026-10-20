#!/usr/bin/env python3
"""FT.SEARCH replies as RediSearch's: RESP3 maps, and RETURN / NOCONTENT / WITHSCORES / KNN 0.

redis-py 8 negotiates RESP3 by default, and its search module then reads
FT.SEARCH and FT.INFO replies as RESP3 maps. Pion sent its RESP2 arrays on
every connection, so with redis-py's default settings `r.ft().search()` and
`r.ft().info()` failed inside redis-py (AttributeError). Pion's own tests pin
protocol=2, which is how it went unnoticed.

FT.SEARCH also ignored RETURN, NOCONTENT and WITHSCORES and refused KNN 0, in
both protocols. RETURN body (fetch the text: RAG's basic call) came back
without the text, NOCONTENT turned each document into two "documents" in
redis-py, WITHSCORES crashed redis-py's parser.

The expected replies are RediSearch's own (redis-stack 7.4), for the same
data and queries:
  FT.SEARCH  RESP3 %5 attributes *0 / format +STRING / results *N / total_results :N / warning *0,
             each result a map: id, score (WITHSCORES), extra_attributes (unless NOCONTENT), values *0
  RETURN     the named fields, renamed by AS, the KNN score first when named
             (its AS alias, or __<field>_score), a field the hash lacks left out;
             RETURN 0 is NOCONTENT; NOCONTENT wins over RETURN
  WITHSCORES the query's text score, 0 for `*=>[KNN ...]`
  KNN 0      an empty result
  FT.INFO    RESP3 a map; num_docs an integer
  no RETURN  the KNN score (its AS alias, or __<field>_score), then every field
             of the hash, the embedding included

Checks: every case in RESP2 and RESP3 with a strict parser; RETURN's two
errors, word for word; a field named like an option inside RETURN; the BM25
form, FT.HYBRID and an index not built yet in RESP3; a pipelined PING behind
a RESP3 FT.SEARCH; with redis-py >= 6 installed, its default connection.

    python3 tests/test_ft_search_replies.py [--port 1974]
    python3 tests/test_ft_search_replies.py --oracle --port <RediSearch port>
        runs the option cases against RediSearch itself (redis-stack-server):
        the expected replies must be its own.
        Keep it off any running Pion's port + 2 + worker: Pion listens there
        too (its affinity ports), and the "oracle" is then Pion.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

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



def _is_num(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def norm2(reply, ws: bool, content: bool):
    """A RESP2 FT.SEARCH reply as (total, [(key, text_score, fields)]); the
    KNN score's value is masked, since RediSearch prints more digits."""
    if reply[0] == "-":
        return reply
    items = reply[1]
    total, docs, i = items[0][1], [], 1
    while i < len(items):
        key = items[i][1].decode()
        i += 1
        ts = None
        if ws:
            ts = items[i][1].decode()
            i += 1
        fields = None
        if content:
            arr = items[i][1]
            i += 1
            fields = [(arr[j][1].decode(), arr[j + 1][1].decode("latin-1")) for j in range(0, len(arr), 2)]
        docs.append((key, ts, fields))
    return mask(total, docs)


def norm3(reply, ws: bool, content: bool):
    """The same for a RESP3 reply; also checks each result map's key order."""
    if reply[0] == "-":
        return reply
    docs = []
    want_keys = ["id"] + (["score"] if ws else []) + (["extra_attributes"] if content else []) + ["values"]
    for d in map_get(reply, "results")[1]:
        if map_keys(d) != want_keys:
            return ("keys", map_keys(d))
        s = map_get(d, "score")
        ea = map_get(d, "extra_attributes")
        docs.append((map_get(d, "id")[1].decode(), s[1] if s else None,
                     [(k[1].decode(), v[1].decode("latin-1")) for k, v in ea[1]] if ea else None))
    return mask(map_get(reply, "total_results")[1], docs)


def mask(total, docs):
    out = []
    for key, ts, fields in docs:
        if fields is not None:
            fields = [(k, "<f>" if (k in ("score", "__vector_score") and _is_num(v)) else v)
                      for k, v in fields]
        out.append((key, ts, fields))
    return (total, out)


D0, D3 = "r3doc:0", "r3doc:3"   # the two nearest to QUERY; r3doc:3 has no id field


def _blob(key: str) -> str:
    return struct.pack("4f", *DOCS[key][0]).decode("latin-1")


def hash_order_free(n):
    """Without RETURN the hash's fields follow the score in the hash's own
    order, which differs between the two servers' hash tables: compare them
    as a set."""
    if not (isinstance(n, tuple) and len(n) == 2 and isinstance(n[1], list)):
        return n
    return (n[0], [(k, ts, f if f is None else f[:1] + sorted(f[1:])) for k, ts, f in n[1]])
KNN2 = "*=>[KNN 2 @vector $vec AS score]"
P = ["PARAMS", 2, "vec", QUERY]
OPTION_CASES = [
    # (name, FT.SEARCH arguments after the index, WITHSCORES, content, expected)
    ("RETURN 3 body id score: the score first, a missing id left out",
     [KNN2, *P, "SORTBY", "score", "RETURN", 3, "body", "id", "score", "DIALECT", 2], False, True,
     (2, [(D0, None, [("score", "<f>"), ("body", "alpha bravo"), ("id", "0")]),
          (D3, None, [("score", "<f>"), ("body", "alpha foxtrot")])])),
    ("RETURN 2 id body: requested order",
     [KNN2, *P, "SORTBY", "score", "RETURN", 2, "id", "body", "DIALECT", 2], False, True,
     (2, [(D0, None, [("id", "0"), ("body", "alpha bravo")]), (D3, None, [("body", "alpha foxtrot")])])),
    ("RETURN 1 nosuch: no fields",
     [KNN2, *P, "SORTBY", "score", "RETURN", 1, "nosuch", "DIALECT", 2], False, True,
     (2, [(D0, None, []), (D3, None, [])])),
    ("RETURN 0 is NOCONTENT",
     [KNN2, *P, "SORTBY", "score", "RETURN", 0, "DIALECT", 2], False, False,
     (2, [(D0, None, None), (D3, None, None)])),
    ("RETURN 3 body AS text",
     [KNN2, *P, "SORTBY", "score", "RETURN", 3, "body", "AS", "text", "DIALECT", 2], False, True,
     (2, [(D0, None, [("text", "alpha bravo")]), (D3, None, [("text", "alpha foxtrot")])])),
    ("NOCONTENT WITHSCORES",
     [KNN2, *P, "SORTBY", "score", "NOCONTENT", "WITHSCORES", "DIALECT", 2], True, False,
     (2, [(D0, "0", None), (D3, "0", None)])),
    ("WITHSCORES RETURN 1 id, no SORTBY",
     [KNN2, *P, "WITHSCORES", "RETURN", 1, "id", "DIALECT", 2], True, True,
     (2, [(D0, "0", [("id", "0")]), (D3, "0", [])])),
    ("no AS: the score is __vector_score",
     ["*=>[KNN 2 @vector $vec]", *P, "RETURN", 2, "id", "__vector_score", "DIALECT", 2], False, True,
     (2, [(D0, None, [("__vector_score", "<f>"), ("id", "0")]), (D3, None, [("__vector_score", "<f>")])])),
    ("RETURN 1 withscores names a field, not the option",
     [KNN2, *P, "SORTBY", "score", "RETURN", 1, "withscores", "DIALECT", 2], False, True,
     (2, [(D0, None, []), (D3, None, [])])),
    ("NOCONTENT before RETURN wins",
     [KNN2, *P, "NOCONTENT", "RETURN", 1, "id", "DIALECT", 2], False, False,
     (2, [(D0, None, None), (D3, None, None)])),
    ("NOCONTENT after RETURN wins",
     [KNN2, *P, "RETURN", 1, "id", "NOCONTENT", "DIALECT", 2], False, False,
     (2, [(D0, None, None), (D3, None, None)])),
    ("KNN 0 is an empty result",
     ["*=>[KNN 0 @vector $vec AS score]", *P, "RETURN", 1, "id", "DIALECT", 2], False, True, (0, [])),
    ("RETURN 5 id: too few arguments",
     [KNN2, *P, "RETURN", 5, "id", "DIALECT", 2], False, True,
     ("-", "Bad arguments for RETURN: Expected an argument, but none provided")),
    ("RETURN x id: not a number",
     [KNN2, *P, "RETURN", "x", "id", "DIALECT", 2], False, True,
     ("-", "Bad arguments for RETURN: Could not convert argument to expected type")),
    ("no RETURN: the score, then every field of the hash",
     [KNN2, *P, "SORTBY", "score", "DIALECT", 2], False, True,
     (2, [(D0, None, [("score", "<f>"), ("body", "alpha bravo"), ("id", "0"), ("vector", _blob("r3doc:0"))]),
          (D3, None, [("score", "<f>"), ("body", "alpha foxtrot"), ("vector", _blob("r3doc:3"))])])),
    ("no RETURN, no AS: the score is __vector_score",
     ["*=>[KNN 2 @vector $vec]", *P, "DIALECT", 2], False, True,
     (2, [(D0, None, [("__vector_score", "<f>"), ("body", "alpha bravo"), ("id", "0"), ("vector", _blob("r3doc:0"))]),
          (D3, None, [("__vector_score", "<f>"), ("body", "alpha foxtrot"), ("vector", _blob("r3doc:3"))])])),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1974)
    ap.add_argument("--oracle", action="store_true",
                    help="the server is RediSearch: check only the option cases")
    a = ap.parse_args()
    port = a.port

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

    if a.oracle:
        # A RediSearch that has just started can answer before it has indexed
        # these hashes; wait until it reports all of them.
        for _ in range(50):
            if map_get(c3.cmd("FT.INFO", "idx"), "num_docs") == (":", len(DOCS)):
                break
            time.sleep(0.1)
        for proto, conn, norm in ((2, c2, norm2), (3, c3, norm3)):
            for name, args, ws, content, want in OPTION_CASES:
                got = norm(conn.cmd("FT.SEARCH", "idx", *args), ws, content)
                if name.startswith("no RETURN"):
                    got, want = hash_order_free(got), hash_order_free(want)
                check(f"oracle RESP{proto} {name}", got == want, f"got {got!r}"[:300])
        c3.cmd("FT.DROPINDEX", "idx")
        c3.cmd("DEL", *DOCS)
        print(f"\n{'FAIL' if FAILS else 'PASS'}: {len(FAILS)} failure(s)")
        return 1 if FAILS else 0

    print("RESP3, index created, graph not built yet")
    # Drop-in: before a build FT.SEARCH answers exactly from the ingest buffer
    # (it used to answer 0 results, which a stock client never got past).
    early = check_search_map("unbuilt FT.SEARCH (exact)", c3.cmd(*knn), expect_n=len(DOCS))
    check("unbuilt FT.SEARCH: every document key comes back",
          sorted(doc_fields(d)[0] for d in early) == sorted(DOCS), str(early)[:200])

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

    for proto, conn, norm in ((2, c2, norm2), (3, c3, norm3)):
        print(f"RESP{proto} options, against RediSearch's replies")
        for name, args, ws, content, want in OPTION_CASES:
            reply = conn.cmd("FT.SEARCH", "idx", *args)
            try:
                got = norm(reply, ws, content)
            except (AttributeError, IndexError, TypeError, ValueError):
                got = ("not this shape", reply)   # e.g. fields where NOCONTENT wants none
            if name.startswith("no RETURN"):
                got, want = hash_order_free(got), hash_order_free(want)
            check(f"RESP{proto} {name}", got == want, f"got {got!r}"[:300])

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
