#!/usr/bin/env python3
"""FT.* against RediSearch, option by option: what a client sees must agree.

WHY
1.0.1, 1.0.2 and 1.0.3 each fixed one FT.CREATE option that was accepted and
never applied: the hashes already in the keyspace, a restart, PREFIX. They
were found one at a time because the FT.CREATE and FT.SEARCH parsers read the
options they know and step over every other token, so an option they do not
know answers +OK and changes nothing. test_redis_differential.py closed that
class for the KV commands by asking a real Redis; this does the same for FT.*.

Each case defines an index, writes the same documents to both servers and runs
the same commands. A step is compared on what a client reads: the reply, and
for FT.SEARCH the total, the documents in order, and their fields (KNN scores
to 4 significant digits; field order within a document is ignored, as a
client builds a dict). FT.INFO is compared on num_docs only, since Pion
documents that it sends two fields.

A step can end three ways:
  agree     the same answer;
  refused   Pion answers an error where RediSearch answers: allowed only for a
            step in ft_differential_known.tsv marked `refused`, which is
            something Pion documents as unsupported and refuses loudly;
  differs   anything else, including Pion accepting what RediSearch refuses.
A step that differs and is not listed FAILS. A listed step that now agrees
FAILS too (remove it on purpose, so a fix cannot be lost silently).

The oracle is redis-server with the RediSearch and RedisJSON modules, which is
what Redis 8 ships. Homebrew's redis 8 has both under lib/redis/modules; set
PION_REDISEARCH_SO (and optionally PION_REJSON_SO) elsewhere.

    python3 tests/test_ft_differential.py --pion-port 1974 --start-oracle
    python3 tests/test_ft_differential.py --only limit sortby     # substring filter on case names
    python3 tests/test_ft_differential.py --show                   # print both replies of every step

Keep the oracle off Pion's affinity ports (port + 2 + worker): Pion listens
there too, and the "oracle" would then be Pion.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resp_strict import Conn, RespError, wait_ready  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWN_FILE = os.path.join(HERE, "ft_differential_known.tsv")


# ── the oracle ─────────────────────────────────────────────────────────────
def find_module(name: str, env: str) -> str | None:
    """A RediSearch / RedisJSON module next to a redis-server, or $env."""
    p = os.environ.get(env)
    if p:
        return p if os.path.exists(p) else None
    pats = [f"/opt/homebrew/Cellar/redis/*/lib/redis/modules/{name}",
            f"/usr/local/Cellar/redis/*/lib/redis/modules/{name}",
            f"/usr/lib/redis/modules/{name}", f"/usr/local/lib/redis/modules/{name}"]
    hits = sorted(h for pat in pats for h in glob.glob(pat))
    return hits[-1] if hits else None


def start_oracle(port: int):
    search = find_module("redisearch.so", "PION_REDISEARCH_SO")
    if not search:
        raise SystemExit("no RediSearch module found (set PION_REDISEARCH_SO)")
    rejson = find_module("rejson.so", "PION_REJSON_SO")
    # the redis-server the modules were built for, when they sit in a Cellar
    server = shutil.which("redis-server")
    cellar_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(search)))),
                              "bin", "redis-server")
    if os.path.exists(cellar_bin):
        server = cellar_bin
    d = tempfile.mkdtemp(prefix="ft-oracle-")
    args = [server, "--port", str(port), "--save", "", "--appendonly", "no", "--dir", d,
            "--loadmodule", search]
    if rejson:
        args += ["--loadmodule", rejson]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, d


# ── documents ──────────────────────────────────────────────────────────────
DIM = 8
N = 40


def _rng(seed):
    # a small LCG, so the documents do not depend on numpy being installed
    state = [seed]

    def nxt():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((state[0] >> 11) / float(1 << 53)) * 2.0 - 1.0
    return nxt


_r = _rng(7)
VECS = [[_r() for _ in range(DIM)] for _ in range(N)]
QUERIES = [[_r() for _ in range(DIM)] for _ in range(4)]
# price and ts are a permutation of the document number, so an order by them
# is not the order by key or by distance
_PERM = [(i * 17 + 5) % N for i in range(N)]
CATS = ["a", "b", "c"]
NAMES = ["kiwi", "apple", "fig", "date", "lime", "pear", "plum", "sloe", "yuzu", "ugli"]


def f32(v):
    return struct.pack(f"<{len(v)}f", *v)


def f64(v):
    return struct.pack(f"<{len(v)}d", *v)


def base_doc(i, enc=f32):
    return {"vec": enc(VECS[i]), "cat": CATS[i % 3], "price": str(_PERM[i] * 1.5),
            "ts": str(1_700_000_000 + _PERM[i] * 7), "name": NAMES[i % len(NAMES)] + str(_PERM[i])}


def hset(key, fields):
    out = ["HSET", key]
    for k, v in fields.items():
        out += [k, v]
    return tuple(out)


BASE_DOCS = [hset(f"doc:{i}", base_doc(i)) for i in range(N)]


# ── commands ───────────────────────────────────────────────────────────────
VEC_ARGS = ["vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2"]
FILTER_FIELDS = ["cat", "TAG", "price", "NUMERIC", "ts", "NUMERIC", "name", "TAG"]
BASE = ["ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA"] + VEC_ARGS + FILTER_FIELDS


def create(*args):
    return ("FT.CREATE", "idx") + tuple(args)


def knn(q=0, k=5, pre="*", field="vec", tail="", extra=(), blob=None, dialect=True, limit=True):
    """A KNN query. FT.SEARCH returns 10 rows unless told otherwise (LIMIT 0 10
    is RediSearch's default), so a k over 10 also sends LIMIT 0 k unless the
    case pages itself or limit=False."""
    b = blob if blob is not None else f32(QUERIES[q])
    extra = tuple(extra)
    if limit and k > 10 and "LIMIT" not in extra:
        extra += ("LIMIT", "0", str(k))
    cmd = ("FT.SEARCH", "idx", f"{pre}=>[KNN {k} @{field} $v{tail}]", "PARAMS", "2", "v", b) + extra
    return cmd + (("DIALECT", "2") if dialect else ())


def ids(q=0, k=5, **kw):
    """The same KNN query, documents only (NOCONTENT)."""
    kw["extra"] = tuple(kw.get("extra", ())) + ("NOCONTENT",)
    return knn(q, k, **kw)


INFO = ("FT.INFO", "idx")
WAIT = ("__wait_indexed__",)


class Case:
    def __init__(self, name, create_args=None, docs=None, steps=None, before=(), setup=()):
        self.name = name
        self.create_args = BASE if create_args is None else create_args
        self.docs = BASE_DOCS if docs is None else docs
        self.before = before          # commands before FT.CREATE (documents already there)
        self.setup = setup            # commands after the documents, not compared
        self.steps = steps if steps is not None else [("knn", knn()), ("ids", ids(1, 10)), ("info", INFO)]

    def script(self):
        out = []
        for c in self.before:
            out.append(("_before", c, False))
        out.append(("create", create(*self.create_args), True))
        if self.before:
            out.append(("_wait", WAIT, False))
        for c in self.docs:
            out.append(("_doc", c, False))
        for c in self.setup:
            out.append(("_setup", c, False))
        for label, c in self.steps:
            out.append((label, c, True))
        return out


def schema(*fields, vec=None, opts=("ON", "HASH", "PREFIX", "1", "doc:")):
    return list(opts) + ["SCHEMA"] + list(vec if vec is not None else VEC_ARGS) + list(fields)


def vec_args(algo="HNSW", *attrs):
    return ["vec", "VECTOR", algo, str(len(attrs))] + list(attrs)


V_STD = ["TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2"]

CASES: list[Case] = []


def case(*a, **kw):
    CASES.append(Case(*a, **kw))


# FT.CREATE: options before SCHEMA ------------------------------------------
case("create.base")
case("create.no_on", ["PREFIX", "1", "doc:", "SCHEMA"] + VEC_ARGS + FILTER_FIELDS)
case("create.no_prefix", ["ON", "HASH", "SCHEMA"] + VEC_ARGS + FILTER_FIELDS,
     docs=BASE_DOCS + [hset("other:1", base_doc(3))])
case("create.on_json", schema(*FILTER_FIELDS, opts=("ON", "JSON", "PREFIX", "1", "doc:")))
case("create.filter_numeric", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "FILTER", "@price>20")))
case("create.filter_startswith", schema(*FILTER_FIELDS, opts=("ON", "HASH", "FILTER", 'startswith(@__key, "doc:1")')))
case("create.language", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "LANGUAGE", "english")))
case("create.language_field", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "LANGUAGE_FIELD", "lang")))
case("create.score", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "SCORE", "0.5")))
case("create.score_field", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "SCORE_FIELD", "s")))
case("create.payload_field", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "PAYLOAD_FIELD", "p")))
case("create.maxtextfields", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "MAXTEXTFIELDS")))
case("create.nooffsets", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "NOOFFSETS")))
case("create.nohl", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "NOHL")))
case("create.nofields", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "NOFIELDS")))
case("create.nofreqs", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "NOFREQS")))
case("create.stopwords0", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "STOPWORDS", "0")))
case("create.stopwords2", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "STOPWORDS", "2", "a", "b")))
case("create.temporary", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "TEMPORARY", "3600")))
case("create.skipinitialscan", before=BASE_DOCS[:10], docs=BASE_DOCS[10:],
     create_args=schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "SKIPINITIALSCAN")))
case("create.backfill", before=BASE_DOCS[:10], docs=BASE_DOCS[10:])
case("create.unknown_option", schema(*FILTER_FIELDS, opts=("ON", "HASH", "PREFIX", "1", "doc:", "BOGUS")))
case("create.again_same", steps=[("again", create(*BASE)), ("knn", knn()), ("info", INFO)])
case("create.missing_schema", ["ON", "HASH", "PREFIX", "1", "doc:"])
case("create.empty_schema", ["ON", "HASH", "PREFIX", "1", "doc:", "SCHEMA"])

# SCHEMA: field forms ---------------------------------------------------------
case("field.vec_as", schema(*FILTER_FIELDS, vec=["vec", "AS", "emb", "VECTOR", "HNSW", "6"] + V_STD),
     steps=[("knn_alias", knn(field="emb")), ("knn_name", knn(field="vec")), ("info", INFO)])
case("field.tag_as", schema("cat", "AS", "category", "TAG", "price", "NUMERIC"),
     steps=[("alias", ids(pre="(@category:{a})")), ("name", ids(pre="(@cat:{a})"))])
case("field.numeric_as", schema("cat", "TAG", "price", "AS", "cost", "NUMERIC"),
     steps=[("alias", ids(pre="(@cost:[10 30])")), ("name", ids(pre="(@price:[10 30])"))])
case("field.sortable", schema("cat", "TAG", "SORTABLE", "price", "NUMERIC", "SORTABLE"),
     steps=[("knn", knn()), ("filter", ids(pre="(@price:[10 30])"))])
case("field.sortable_unf", schema("cat", "TAG", "SORTABLE", "UNF", "price", "NUMERIC", "SORTABLE", "UNF"))
case("field.noindex", schema("cat", "TAG", "price", "NUMERIC", "SORTABLE", "NOINDEX"),
     steps=[("filter_noindex", ids(pre="(@price:[10 30])")), ("knn", knn())])
case("field.tag_separator", schema("cat", "TAG", "SEPARATOR", ";", "price", "NUMERIC"),
     setup=[hset("doc:1", {"cat": "x;y"}), hset("doc:2", {"cat": "y; z"})],
     steps=[("x", ids(pre="(@cat:{x})", k=40)), ("y", ids(pre="(@cat:{y})", k=40)),
            ("z", ids(pre="(@cat:{z})", k=40)), ("whole", ids(pre="(@cat:{x;y})", k=40))])
case("field.tag_default_separator",
     setup=[hset("doc:1", {"cat": "x,y"}), hset("doc:2", {"cat": "y , z"}), hset("doc:4", {"cat": "Mixed"})],
     steps=[("x", ids(pre="(@cat:{x})", k=40)), ("y", ids(pre="(@cat:{y})", k=40)),
            ("z", ids(pre="(@cat:{z})", k=40)), ("upper_query", ids(pre="(@cat:{A})", k=40)),
            ("mixed_lower", ids(pre="(@cat:{mixed})", k=40)), ("mixed_exact", ids(pre="(@cat:{Mixed})", k=40))])
case("field.tag_casesensitive", schema("cat", "TAG", "CASESENSITIVE", "price", "NUMERIC"),
     setup=[hset("doc:4", {"cat": "Mixed"})],
     steps=[("upper_query", ids(pre="(@cat:{A})", k=40)), ("mixed_lower", ids(pre="(@cat:{mixed})", k=40)),
            ("mixed_exact", ids(pre="(@cat:{Mixed})", k=40))])
case("field.text", schema("title", "TEXT", "cat", "TAG"),
     setup=[hset(f"doc:{i}", {"title": f"hello world {i}"}) for i in range(5)],
     steps=[("knn", knn()), ("info", INFO)])
case("field.text_options", schema("title", "TEXT", "WEIGHT", "2", "NOSTEM", "cat", "TAG"))
case("field.geo", schema("loc", "GEO", "cat", "TAG"),
     setup=[hset(f"doc:{i}", {"loc": f"{i * 0.01},{i * 0.01}"}) for i in range(N)],
     steps=[("knn", knn()), ("info", INFO)])
case("field.geoshape", schema("shape", "GEOSHAPE", "cat", "TAG"))
case("field.unknown_type", schema("cat", "BOGUS", "price", "NUMERIC"))
case("field.underscore_name", schema("cat", "TAG", vec=["my_vec", "VECTOR", "HNSW", "6"] + V_STD),
     docs=[hset(f"doc:{i}", {"my_vec": f32(VECS[i]), "cat": CATS[i % 3]}) for i in range(N)],
     steps=[("knn", knn(field="my_vec")), ("info", INFO)])
case("field.upper_name", schema("cat", "TAG", vec=["Vec", "VECTOR", "HNSW", "6"] + V_STD),
     docs=[hset(f"doc:{i}", {"Vec": f32(VECS[i]), "cat": CATS[i % 3]}) for i in range(N)],
     steps=[("knn_exact", knn(field="Vec")), ("knn_lower", knn(field="vec")), ("info", INFO)])
case("field.upper_hash_field", docs=[hset(f"doc:{i}", {"VEC": f32(VECS[i]), "cat": CATS[i % 3]}) for i in range(N)],
     steps=[("knn", knn()), ("info", INFO)])
case("field.tag_upper_name", schema("Cat", "TAG", "price", "NUMERIC"),
     docs=[hset(f"doc:{i}", {"vec": f32(VECS[i]), "Cat": CATS[i % 3]}) for i in range(N)],
     steps=[("exact", ids(pre="(@Cat:{a})", k=40)), ("lower", ids(pre="(@cat:{a})", k=40))])

# VECTOR attributes -------------------------------------------------------------
case("vec.flat", schema(*FILTER_FIELDS, vec=vec_args("FLAT", *V_STD)))
case("vec.hnsw_params", schema(*FILTER_FIELDS, vec=vec_args("HNSW", *V_STD, "M", "4", "EF_CONSTRUCTION", "50")))
case("vec.hnsw_runtime", schema(*FILTER_FIELDS, vec=vec_args("HNSW", *V_STD, "EF_RUNTIME", "20", "EPSILON", "0.01")))
case("vec.initial_cap", schema(*FILTER_FIELDS, vec=vec_args("HNSW", *V_STD, "INITIAL_CAP", "100")))
case("vec.flat_block", schema(*FILTER_FIELDS, vec=vec_args("FLAT", *V_STD, "BLOCK_SIZE", "64")))
case("vec.cosine", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE")))
case("vec.ip", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "IP")))
case("vec.metric_lower", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "float32", "DIM", str(DIM), "DISTANCE_METRIC", "l2")))
case("vec.float64", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT64", "DIM", str(DIM), "DISTANCE_METRIC", "L2")),
     docs=[hset(f"doc:{i}", base_doc(i, f64)) for i in range(N)],
     steps=[("knn64", knn(blob=f64(QUERIES[0]))), ("knn32", knn()), ("info", INFO)])
for _t in ("FLOAT16", "BFLOAT16", "INT8", "UINT8"):
    case(f"vec.{_t.lower()}", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", _t, "DIM", str(DIM), "DISTANCE_METRIC", "L2")),
         steps=[("info", INFO)])
case("vec.svs", schema(*FILTER_FIELDS, vec=vec_args("SVS-VAMANA", *V_STD)), steps=[("info", INFO)])
case("vec.unknown_algo", schema(*FILTER_FIELDS, vec=vec_args("BOGUS", *V_STD)), steps=[("info", INFO)])
case("vec.count_short", schema(*FILTER_FIELDS, vec=["vec", "VECTOR", "HNSW", "4"] + V_STD), steps=[("info", INFO)])
case("vec.count_long", schema(*FILTER_FIELDS, vec=["vec", "VECTOR", "HNSW", "8"] + V_STD), steps=[("info", INFO)])
case("vec.count_bad", schema(*FILTER_FIELDS, vec=["vec", "VECTOR", "HNSW", "six"] + V_STD), steps=[("info", INFO)])
case("vec.no_type", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "DIM", str(DIM), "DISTANCE_METRIC", "L2")), steps=[("info", INFO)])
case("vec.no_dim", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DISTANCE_METRIC", "L2")), steps=[("info", INFO)])
case("vec.no_metric", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DIM", str(DIM))), steps=[("info", INFO)])
case("vec.dim_zero", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DIM", "0", "DISTANCE_METRIC", "L2")), steps=[("info", INFO)])
case("vec.dim_bad", schema(*FILTER_FIELDS, vec=vec_args("HNSW", "TYPE", "FLOAT32", "DIM", "8x", "DISTANCE_METRIC", "L2")), steps=[("info", INFO)])
case("vec.ef_bad", schema(*FILTER_FIELDS, vec=vec_args("HNSW", *V_STD, "EF_CONSTRUCTION", "abc")), steps=[("info", INFO)])
case("vec.two_vectors", schema(*FILTER_FIELDS, "vec2", "VECTOR", "HNSW", "6", *V_STD),
     docs=[hset(f"doc:{i}", dict(base_doc(i), vec2=f32(VECS[N - 1 - i]))) for i in range(N)],
     steps=[("knn1", knn()), ("knn2", knn(field="vec2")), ("info", INFO)])

# FT.SEARCH: KNN, paging, ordering ----------------------------------------------
S = []
S += [("k1", knn(k=1)), ("k40", ids(k=40)), ("k100", ids(k=100)), ("k0", knn(k=0))]
SC = ("SORTBY", "__vec_score")
S += [("limit_0_3", ids(k=10, extra=SC + ("LIMIT", "0", "3"))), ("limit_2_3", ids(k=10, extra=SC + ("LIMIT", "2", "3"))),
      ("limit_8_5", ids(k=10, extra=SC + ("LIMIT", "8", "5"))), ("limit_20_5", ids(k=10, extra=SC + ("LIMIT", "20", "5"))),
      ("limit_0_0", ids(k=10, extra=SC + ("LIMIT", "0", "0"))), ("limit_0_20", ids(k=10, extra=SC + ("LIMIT", "0", "20"))),
      ("limit_bad", ids(k=10, extra=SC + ("LIMIT", "x", "3"))), ("limit_neg", ids(k=10, extra=SC + ("LIMIT", "0", "-1"))),
      ("limit_one_arg", ids(k=10, extra=SC + ("LIMIT", "0"))),
      ("limit_unsorted", ids(k=10, extra=("LIMIT", "2", "3"))),
      ("default_limit", ids(k=20, extra=SC, limit=False)), ("default_limit_unsorted", ids(k=20, limit=False)),
      ("sorted_k10", knn(k=10, extra=SC + ("RETURN", "1", "__vec_score")))]
S += [("sortby_price", ids(k=10, extra=("SORTBY", "price"))),
      ("sortby_price_asc", ids(k=10, extra=("SORTBY", "price", "ASC"))),
      ("sortby_price_desc", ids(k=10, extra=("SORTBY", "price", "DESC"))),
      ("sortby_name", ids(k=10, extra=("SORTBY", "name", "ASC"))),
      ("sortby_score", ids(k=10, extra=("SORTBY", "__vec_score"))),
      ("sortby_score_desc", ids(k=10, extra=("SORTBY", "__vec_score", "DESC"))),
      ("sortby_alias", ids(k=10, tail=" AS dist", extra=("SORTBY", "dist", "ASC"))),
      ("sortby_alias_desc", ids(k=10, tail=" AS dist", extra=("SORTBY", "dist", "DESC"))),
      ("sortby_unknown", ids(k=10, extra=("SORTBY", "nosuch"))),
      ("sortby_price_limit", ids(k=10, extra=("SORTBY", "price", "DESC", "LIMIT", "0", "3"))),
      ("sortby_bad_dir", ids(k=10, extra=("SORTBY", "price", "BOGUS"))),
      ("sortby_no_field", ids(k=10, extra=("SORTBY",))),
      ("k0_limit", ids(k=0, extra=("LIMIT", "0", "5")))]
S += [("return_price", knn(k=5, extra=("RETURN", "1", "price"))),
      ("return_as", knn(k=5, extra=("RETURN", "3", "price", "AS", "p"))),
      ("return_score", knn(k=5, extra=("RETURN", "1", "__vec_score"))),
      ("return_alias", knn(k=5, tail=" AS dist", extra=("RETURN", "2", "dist", "cat"))),
      ("withscores", knn(k=3, extra=("WITHSCORES",))), ("nocontent_withscores", knn(k=3, extra=("NOCONTENT", "WITHSCORES"))),
      ("ef_runtime", ids(k=10, tail=" EF_RUNTIME 50")), ("ef_runtime_param", ("FT.SEARCH", "idx", "*=>[KNN 5 @vec $v EF_RUNTIME $ef]", "PARAMS", "4", "v", f32(QUERIES[0]), "ef", "40", "NOCONTENT", "DIALECT", "2"))]
S += [("no_dialect", ids(k=5, dialect=False)), ("dialect3", ids(k=5, extra=("DIALECT", "3"), dialect=False)),
      ("dialect1", ids(k=5, extra=("DIALECT", "1"), dialect=False)), ("timeout", ids(k=5, extra=("TIMEOUT", "1000")))]
S += [("verbatim", ids(k=5, extra=("VERBATIM",))), ("nostopwords", ids(k=5, extra=("NOSTOPWORDS",))),
      ("inorder", ids(k=5, extra=("INORDER",))), ("slop", ids(k=5, extra=("SLOP", "1"))),
      ("language", ids(k=5, extra=("LANGUAGE", "english"))), ("scorer", ids(k=5, extra=("SCORER", "BM25"))),
      ("infields", ids(k=5, extra=("INFIELDS", "1", "cat"))),
      ("inkeys", ids(k=5, extra=("INKEYS", "3", "doc:1", "doc:2", "doc:3"))),
      ("geofilter", ids(k=5, extra=("GEOFILTER", "loc", "0", "0", "10", "km"))),
      ("withpayloads", knn(k=3, extra=("NOCONTENT", "WITHPAYLOADS"))),
      ("withsortkeys", knn(k=3, extra=("NOCONTENT", "WITHSORTKEYS", "SORTBY", "price"))),
      ("explainscore", knn(k=3, extra=("NOCONTENT", "WITHSCORES", "EXPLAINSCORE"))),
      ("summarize", knn(k=3, extra=("RETURN", "1", "cat", "SUMMARIZE"))),
      ("highlight", knn(k=3, extra=("RETURN", "1", "cat", "HIGHLIGHT"))),
      ("payload", ids(k=5, extra=("PAYLOAD", "x"))), ("expander", ids(k=5, extra=("EXPANDER", "SYNONYM"))),
      ("unknown_option", ids(k=5, extra=("BOGUS",)))]
S += [("params_odd", ("FT.SEARCH", "idx", "*=>[KNN 5 @vec $v]", "PARAMS", "1", "v", "DIALECT", "2")),
      ("param_missing", ("FT.SEARCH", "idx", "*=>[KNN 5 @vec $w]", "PARAMS", "2", "v", f32(QUERIES[0]), "DIALECT", "2")),
      ("blob_short", knn(blob=f32(QUERIES[0])[:-4])), ("blob_long", knn(blob=f32(QUERIES[0]) + b"\0\0\0\0")),
      ("wrong_field", knn(field="nosuch")), ("wrong_index", ("FT.SEARCH", "nosuch", "*=>[KNN 5 @vec $v]", "PARAMS", "2", "v", f32(QUERIES[0]), "DIALECT", "2")),
      ("star", ("FT.SEARCH", "idx", "*", "NOCONTENT", "LIMIT", "0", "5")),
      ("tag_query", ("FT.SEARCH", "idx", "@cat:{a}", "NOCONTENT", "LIMIT", "0", "5")),
      ("count_only", ("FT.SEARCH", "idx", "*", "LIMIT", "0", "0"))]
case("search", steps=S)

# FT.SEARCH: prefilters ------------------------------------------------------------
P = [("tag", ids(pre="(@cat:{a})", k=40)), ("tag_noparen", ids(pre="@cat:{a}", k=40)),
     ("tag_or", ids(pre="(@cat:{a|b})", k=40)), ("tag_upper", ids(pre="(@cat:{A})", k=40)),
     ("tag_space", ids(pre="(@cat:{ a })", k=40)), ("tag_missing", ids(pre="(@cat:{zz})", k=40)),
     ("num", ids(pre="(@price:[10 20])", k=40)), ("num_inf", ids(pre="(@price:[-inf 5])", k=40)),
     ("num_excl", ids(pre="(@price:[(10 20])", k=40)), ("num_excl_hi", ids(pre="(@price:[10 (20])", k=40)),
     ("num_neg", ids(pre="(@price:[-5 -1])", k=40)), ("num_one", ids(pre="(@price:[4.5 4.5])", k=40)),
     ("num_bad", ids(pre="(@price:[a b])", k=40)),
     ("ts_tight", ids(pre="(@ts:[1700000014 1700000014])", k=40)),
     ("ts_range", ids(pre="(@ts:[1700000010 1700000030])", k=40)),
     ("and", ids(pre="(@price:[0 30] @cat:{a})", k=40)), ("and3", ids(pre="(@price:[0 30] @cat:{a} @name:{kiwi0})", k=40)),
     ("not", ids(pre="(-@cat:{a})", k=40)), ("or_clauses", ids(pre="(@cat:{a} | @cat:{b})", k=40)),
     ("undeclared", ids(pre="(@other:{a})", k=40)), ("text_term", ids(pre="(hello)", k=40)),
     ("filter_legacy", ids(k=40, extra=("FILTER", "price", "10", "20"))),
     ("filter_legacy_excl", ids(k=40, extra=("FILTER", "price", "(10", "20"))),
     ("filter_legacy_inf", ids(k=40, extra=("FILTER", "price", "-inf", "+inf"))),
     ("filter_clause", ids(k=40, extra=("FILTER", "@price:[10 20]"))),
     ("filter_eq", ids(k=40, extra=("FILTER", "cat=a"))),
     ("filter_bracket", ids(k=40, extra=("FILTER", "price", "[10 20]"))),
     ("filter_two", ids(k=40, extra=("FILTER", "price", "0", "30", "FILTER", "ts", "1700000000", "1700000100"))),
     ("filter_undeclared", ids(k=40, extra=("FILTER", "other", "0", "1"))),
     ("filter_tag_field", ids(k=40, extra=("FILTER", "cat", "0", "1")))]
P += [("tag_on_numeric", ids(pre="(@price:{a})", k=40)), ("num_on_tag", ids(pre="(@cat:[0 1])", k=40)),
      ("filter_bad_lo", ids(k=40, extra=("FILTER", "price", "abc", "1"))),
      ("num_huge", ids(pre="(@price:[-1e400 1e400])", k=40)), ("num_three", ids(pre="(@price:[1 2 3])", k=40)),
      ("tag_empty", ids(pre="(@cat:{})", k=40))]
case("prefilter", steps=P)
case("prefilter.sort_missing", setup=[("HDEL", "doc:20", "price"), ("HDEL", "doc:37", "price")],
     steps=[("asc", ids(k=10, extra=("SORTBY", "price", "ASC"))), ("desc", ids(k=10, extra=("SORTBY", "price", "DESC"))),
            ("page", ids(k=10, extra=("SORTBY", "price", "ASC", "LIMIT", "2", "3")))])
# ties broken by document order: doc:16 and doc:32 written with one price
case("prefilter.sort_ties", docs=[hset(f"doc:{i}", dict(base_doc(i), price="43.5") if i in (16, 32) else base_doc(i))
                                  for i in range(N)],
     steps=[("asc", ids(k=10, extra=("SORTBY", "price", "ASC"))), ("desc", ids(k=10, extra=("SORTBY", "price", "DESC")))])
case("prefilter.sort_tie_update", setup=[hset("doc:16", {"price": "43.5"})],
     steps=[("asc", ids(k=10, extra=("SORTBY", "price", "ASC")))])
case("prefilter.noindex", schema("cat", "TAG", "price", "NUMERIC", "nx", "NUMERIC", "SORTABLE", "NOINDEX", "title", "TEXT"),
     setup=[hset(f"doc:{i}", {"nx": str(i)}) for i in range(N)],
     steps=[("noindex", ids(pre="(@nx:[0 100])", k=40)), ("noindex_legacy", ids(k=40, extra=("FILTER", "nx", "0", "100"))),
            ("sort_noindex", ids(k=10, extra=("SORTBY", "nx", "DESC"))),
            ("text_num", ids(pre="(@title:[0 1])", k=40)), ("text_tag", ids(pre="(@title:{a})", k=40)),
            ("legacy_text", ids(k=40, extra=("FILTER", "title", "0", "1")))])

# documents changing under the index -------------------------------------------------
case("doc.update", setup=[hset("doc:3", {"vec": f32(QUERIES[0])})], steps=[("knn", ids(k=3)), ("info", INFO)])
case("doc.hdel_vec", setup=[("HDEL", "doc:3", "vec"), ("HDEL", "doc:7", "vec")],
     steps=[("knn", ids(k=40)), ("tag", ids(pre="(@cat:{b})", k=40)), ("info", INFO)])
case("doc.hdel_tag", setup=[("HDEL", "doc:3", "cat")], steps=[("tag", ids(pre="(@cat:{a})", k=40)), ("knn", ids(k=40))])
case("doc.del", setup=[("DEL", "doc:3"), ("DEL", "doc:7")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.unlink", setup=[("UNLINK", "doc:3")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.blob_short", setup=[hset("doc:3", {"vec": b"\0" * (DIM * 4 - 4)})], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.blob_long", setup=[hset("doc:3", {"vec": b"\0" * (DIM * 4 + 4)})], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.no_vec", setup=[hset("doc:100", {"cat": "a", "price": "1"})],
     steps=[("knn", ids(k=50)), ("tag", ids(pre="(@cat:{a})", k=50)), ("info", INFO)])
case("doc.bad_number", setup=[hset("doc:3", {"price": "abc"})],
     steps=[("knn", ids(k=40)), ("num", ids(pre="(@price:[-inf +inf])", k=40)), ("info", INFO)])
case("doc.rename", setup=[("RENAME", "doc:3", "doc:300")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.rename_out", setup=[("RENAME", "doc:3", "x:3")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.not_hash", docs=BASE_DOCS + [("SET", "doc:string", "x")], steps=[("knn", ids(k=50)), ("info", INFO)])
case("doc.expire", setup=[("PEXPIRE", "doc:3", "50"), ("__sleep__", "0.3")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("doc.flushall", setup=[("FLUSHALL",)], steps=[("knn", ids(k=40)), ("info", INFO)])

# after Pion's automatic build (200 ms of quiet): the graph path, with the
# per-node filter values captured at build time ------------------------------------
BUILT = [("__sleep__", "1.0")]
# k = the whole index: these compare what a build changes (scores, deletes,
# updates), not the graph's recall, which is not exact at ef 10
case("built.knn", setup=BUILT, steps=[("knn", knn(k=40, extra=SC + ("LIMIT", "0", "5"))),
                                      ("sorted", ids(2, 40, extra=SC + ("LIMIT", "0", "10"))),
                                      ("scores", knn(3, 40, extra=SC + ("RETURN", "1", "__vec_score", "LIMIT", "0", "10"))),
                                      ("info", INFO)])
case("built.prefilter", setup=BUILT, steps=P)
case("built.tag_update", setup=BUILT + [hset("doc:0", {"cat": "zz"}), hset("doc:1", {"cat": "a"})],
     steps=[("new", ids(pre="(@cat:{zz})", k=40)), ("old", ids(pre="(@cat:{a})", k=40))])
case("built.num_update", setup=BUILT + [hset("doc:0", {"price": "1000"})],
     steps=[("new", ids(pre="(@price:[999 1001])", k=40)), ("old", ids(pre="(@price:[0 100])", k=40))])
case("built.vec_update", setup=BUILT + [hset("doc:3", {"vec": f32(QUERIES[0])})],
     steps=[("knn", ids(k=40, extra=SC + ("LIMIT", "0", "3")))])
case("built.del", setup=BUILT + [("DEL", "doc:3"), ("DEL", "doc:32")], steps=[("knn", ids(k=40)), ("info", INFO)])
case("built.hdel_tag", setup=BUILT + [("HDEL", "doc:0", "cat")], steps=[("tag", ids(pre="(@cat:{a})", k=40))])
case("built.add", setup=BUILT + [hset("doc:100", dict(base_doc(5), vec=f32(QUERIES[1])))],
     steps=[("knn", ids(1, 41, extra=SC + ("LIMIT", "0", "3"))), ("info", INFO)])

# FT.DROPINDEX and the rest of the module ---------------------------------------------
case("drop.keepdocs", steps=[("drop", ("FT.DROPINDEX", "idx")), ("exists", ("EXISTS", "doc:1", "doc:2")),
                              ("info", INFO), ("search", ids())])
case("drop.dd", steps=[("list", ("FT._LIST",)), ("drop", ("FT.DROPINDEX", "idx", "DD")),
                        ("exists", ("EXISTS", "doc:1", "doc:2")), ("dbsize", ("DBSIZE",)), ("info", INFO),
                        ("list_after", ("FT._LIST",)), ("search_after", ids())])
case("drop.dd_twice", steps=[("drop", ("FT.DROPINDEX", "idx", "DD", "DD")), ("info", INFO)])
case("drop.recreate", steps=[("drop", ("FT.DROPINDEX", "idx")), ("create", create(*BASE)), ("wait", WAIT),
                             ("knn", ids(k=40)), ("info", INFO)])
case("drop.dd_prefix", docs=BASE_DOCS + [hset("other:1", base_doc(1))],
     steps=[("drop", ("FT.DROPINDEX", "idx", "DD")), ("other", ("EXISTS", "other:1")), ("doc", ("EXISTS", "doc:1"))])
case("drop.unknown", steps=[("drop", ("FT.DROPINDEX", "nosuch")), ("info", INFO)])
case("drop.bogus_arg", steps=[("drop", ("FT.DROPINDEX", "idx", "BOGUS")), ("info", INFO)])
case("module", steps=[
    ("list", ("FT._LIST",)), ("info_unknown", ("FT.INFO", "nosuch")),
    ("alter", ("FT.ALTER", "idx", "SCHEMA", "ADD", "extra", "TAG")),
    ("explain", ("FT.EXPLAIN", "idx", "*")), ("tagvals", ("FT.TAGVALS", "idx", "cat")),
    ("aliasadd", ("FT.ALIASADD", "al", "idx")), ("aggregate", ("FT.AGGREGATE", "idx", "*", "LIMIT", "0", "1")),
    ("profile", ("FT.PROFILE", "idx", "SEARCH", "QUERY", "*")), ("config", ("FT.CONFIG", "GET", "TIMEOUT")),
    ("spellcheck", ("FT.SPELLCHECK", "idx", "helo")), ("syndump", ("FT.SYNDUMP", "idx")),
    ("dictadd", ("FT.DICTADD", "d", "x")), ("sugadd", ("FT.SUGADD", "ac", "hello", "1")),
    ("cursor", ("FT.CURSOR", "READ", "idx", "1"))])


# ── normalization ───────────────────────────────────────────────────────────
class Approx:
    """A KNN score: equal within 0.2% (or 1e-4). Pion prints 4 decimals, RediSearch 12 digits."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, Approx) and abs(self.v - other.v) <= max(1e-4, 2e-3 * abs(other.v))

    def __hash__(self):
        return 0

    def __repr__(self):
        return f"~{self.v:.6g}"


def _round(v):
    try:
        return Approx(float(v))
    except (TypeError, ValueError):
        return v


def _score_names(cmd):
    names = {b"__vec_score", b"__vec2_score", b"__emb_score", b"__my_vec_score", b"__Vec_score"}
    q = cmd[2] if len(cmd) > 2 and isinstance(cmd[2], str) else ""
    m = re.search(r"\bAS\s+(\w+)\s*\]", q)
    if m:
        names.add(m.group(1).encode())
    return names


def _norm_fields(fields, scores):
    if not isinstance(fields, list):
        return fields
    pairs = []
    for a in range(0, len(fields) - 1, 2):
        k, v = fields[a], fields[a + 1]
        pairs.append((k, _round(v) if k in scores else v))
    return tuple(sorted(pairs, key=lambda p: str(p[0])))


def norm_search(cmd, r):
    """[total, key, fields?, ...] -> (total, ((key, extra..., fields?), ...)).
    Elements after a key up to the next key: WITHSCORES' score, a payload, a
    sort key, the fields list. Keys are bytes starting with a known prefix."""
    if not isinstance(r, list) or not r:
        return r
    scores = _score_names(cmd)
    with_scores = any(isinstance(a, str) and a.upper() == "WITHSCORES" for a in cmd)
    rows, cur = [], None
    for item in r[1:]:
        is_key = isinstance(item, bytes) and (item.startswith(b"doc:") or item.startswith(b"other:")
                                               or item.startswith(b"x:"))
        if is_key and (cur is None or len(cur) > 1 or not with_scores):
            if cur is not None:
                rows.append(tuple(cur))
            cur = [item]
        elif cur is not None:
            cur.append(_norm_fields(item, scores) if isinstance(item, list) else item)
        else:
            rows.append(("?", item))
    if cur is not None:
        rows.append(tuple(cur))
    # Without SORTBY, RediSearch returns the k nearest in its internal document
    # order, not by distance (Pion sorts by distance). Which documents come back
    # is the answer then, not their order.
    if not any(isinstance(a, str) and a.upper() == "SORTBY" for a in cmd):
        rows.sort(key=lambda row: repr(row[0]))
    return (r[0], tuple(rows))


def norm(cmd, r):
    if isinstance(r, RespError):
        return ("ERROR",)
    name = cmd[0].upper() if isinstance(cmd[0], str) else cmd[0]
    if name == "FT.SEARCH":
        return norm_search(cmd, r)
    if name == "FT.INFO" and isinstance(r, list):
        d = {(r[a].decode() if isinstance(r[a], bytes) else r[a]): r[a + 1] for a in range(0, len(r) - 1, 2)}
        nd = d.get("num_docs")
        return ("num_docs", int(nd) if nd is not None else None)
    if name == "FT.CREATE" or name == "FT.DROPINDEX" or name == "FT.ALTER" or name == "FT.ALIASADD":
        return r
    return r


# ── running ─────────────────────────────────────────────────────────────────
def reset(c: Conn):
    c.cmd("FT.DROPINDEX", "idx")
    c.cmd("FLUSHALL")


def wait_indexed(c: Conn, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        r = c.cmd("FT.INFO", "idx")
        if not isinstance(r, list):
            return
        d = {(r[a].decode() if isinstance(r[a], bytes) else r[a]): r[a + 1] for a in range(0, len(r) - 1, 2)}
        if d.get("indexing") in (None, 0, b"0", "0"):
            return
        time.sleep(0.05)


def run(c: Conn, cs: Case):
    reset(c)
    out = []
    for label, cmd, compared in cs.script():
        if cmd[0] == "__wait_indexed__":
            wait_indexed(c)
            if compared:
                out.append((label, ("PING",), "PONG"))
            continue
        if cmd[0] == "__sleep__":
            time.sleep(float(cmd[1]))
            continue
        r = c.cmd(*cmd)
        if compared:
            out.append((label, cmd, r))
    if cs.before or cs.setup:
        pass
    return out


def load_known():
    known = {}
    if os.path.exists(KNOWN_FILE):
        for line in open(KNOWN_FILE, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            known[(parts[0], parts[1])] = (parts[2], parts[3] if len(parts) > 3 else "")
    return known


def first_diff(a, b, path=""):
    """Where two normalized replies first part ways, for the report."""
    if type(a) is not type(b):
        return f"{path or 'reply'}: {_short(a, 80)} vs {_short(b, 80)}"
    if isinstance(a, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_diff(x, y, f"{path}[{i}]")
        if len(a) != len(b):
            return f"{path or 'reply'}: {len(a)} items vs {len(b)}"
        return None
    return f"{path or 'reply'}: {_short(a, 80)} vs {_short(b, 80)}" if a != b else None


def _short(x, n=200):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pion-port", type=int, default=int(os.environ.get("PION_PORT", 1974)))
    ap.add_argument("--oracle-port", type=int, default=6419)
    ap.add_argument("--start-oracle", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write-known", action="store_true",
                    help="record the current divergences in ft_differential_known.tsv and exit 0")
    args = ap.parse_args()
    oproc = odir = None
    if args.start_oracle:
        oproc, odir = start_oracle(args.oracle_port)
    try:
        wait_ready(args.oracle_port, 15, proc=oproc)
        wait_ready(args.pion_port, 30)
        p, o = Conn(args.pion_port, timeout=30), Conn(args.oracle_port, timeout=30)
        mods = o.cmd("MODULE", "LIST")
        if "search" not in repr(mods):
            print(f"FAIL the oracle on {args.oracle_port} has no search module: {mods!r}")
            return 1
        known = load_known()
        n_steps, fails, refused, xpass, seen = 0, [], [], [], set()
        for cs in CASES:
            if args.only and not any(s in cs.name for s in args.only):
                continue
            try:
                got_p = run(p, cs)
            except (ConnectionError, TimeoutError, OSError) as e:
                print(f"FAIL pion lost the connection in case {cs.name} ({type(e).__name__}: {e}) - a crash or a hang")
                return 1
            got_o = run(o, cs)
            for (label, cmd, rp), (_, _, ro) in zip(got_p, got_o):
                n_steps += 1
                key = (cs.name, label)
                seen.add(key)
                np_, no_ = norm(cmd, rp), norm(cmd, ro)
                if args.show:
                    print(f"{cs.name}/{label}\n   pion : {_short(rp, 300)}\n   redis: {_short(ro, 300)}")
                kind = None
                if np_ != no_:
                    kind = "refused" if isinstance(rp, RespError) and not isinstance(ro, RespError) else "differs"
                if key in known:
                    want = known[key][0]
                    if kind is None:
                        xpass.append(key)
                    elif kind == "refused" and want == "refused":
                        refused.append(key)
                    elif kind != want:
                        fails.append((key, kind, cmd, rp, ro, first_diff(np_, no_)))
                elif kind is not None:
                    fails.append((key, kind, cmd, rp, ro, first_diff(np_, no_)))
        if not args.only:
            for key in known:
                if key not in seen:
                    print(f"FAIL {key[0]}/{key[1]} is in {os.path.basename(KNOWN_FILE)} but no such step exists")
                    return 1
        if not p.cmd("PING") == "PONG":
            print("FAIL Pion stopped answering")
            return 1
        if args.write_known:
            with open(KNOWN_FILE, "w", encoding="utf-8") as fh:
                fh.write("# FT.* steps where Pion and RediSearch disagree on purpose (tests/test_ft_differential.py).\n")
                fh.write("# case<TAB>step<TAB>refused|differs<TAB>why\n")
                for key, kind, _, _, _, _ in fails:
                    fh.write(f"{key[0]}\t{key[1]}\t{kind}\tTODO\n")
            print(f"wrote {len(fails)} entries to {KNOWN_FILE}")
            return 0
        print(f"{n_steps} FT.* steps in {len(CASES)} cases: {len(fails)} new divergences from RediSearch, "
              f"{len(refused)} refused as documented, {len(xpass)} known ones now agree")
        for key in xpass:
            print(f"  XPASS {key[0]}/{key[1]} now agrees with RediSearch - remove it from {os.path.basename(KNOWN_FILE)}")
        for key, kind, cmd, rp, ro, where in fails:
            print(f"  {kind:8} {key[0]}/{key[1]}: {_short(tuple(a if not isinstance(a, bytes) or len(a) < 20 else b'<blob>' for a in cmd), 160)}")
            print(f"      first: {where}")
            print(f"      pion : {_short(rp)}")
            print(f"      redis: {_short(ro)}")
        return 1 if fails or xpass else 0
    finally:
        if oproc:
            oproc.kill()
            oproc.wait()
        if odir:
            shutil.rmtree(odir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
