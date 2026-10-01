"""Longest-prefix K/V store in Pion, for `pion-vllm-mlx serve`.

An agent's prompt only ever grows, and two agents on one repository share
their opening (system prompt and tool definitions). So the unit stored
is not "a prompt" but a *lineage*: a tree of segments, each holding the K/V
rows for a contiguous run of token positions on top of its parent.

    segment = (parent, base, tokens[base:base+len]) + K/V rows in the V-store

A segment grows in place while the conversation keeps extending it
(V.STOREBATCH at a start row; gh #41 realloc-grows the buffer). A prompt that
diverges part-way through a segment starts a child segment at the divergence
point, so the shared part is stored once.

Lookup is one round trip: the prompt is cut into BLOCK-token blocks, each
block gets a chained hash (hash of everything up to its end), and a single
HMGET over the index hash returns the segment holding the longest stored
block-aligned prefix. The exact match is then refined token by token inside
that segment.

Everything lives in Pion (keyspace for metadata, V-store for K/V), so a
restarted serve process, a second serve process, or a reloaded model finds
the same lineages.

Budget. The V-store holds every row in RAM and its WAL only grows until a
snapshot truncates it, so without a cap a busy agent fills the disk (a 15K-
token Qwen3-1.7B lineage is ~1.7 GB). With `budget_bytes`, segments carry a
last-use time in one sorted set, and every use touches the whole chain at
once, so a parent's time always equals its newest descendant's: the least
recently used segment is always a leaf (ties go to the newer id, which is the
child). Eviction drops leaves until the V-store is under budget; the V-store's
own LRU would instead take the ROOT, which is written once and never touched
while the leaf grows. Once the WAL outgrows the budget, an idle-time
KV.PREFIX.SAVE snapshots what is live and truncates it. Only plain attention caches (mlx-lm `KVCache`) are
stored; rotating / recurrent caches are skipped rather than approximated.
"""
from __future__ import annotations

import hashlib
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("pion.prefix_store")


class PionError(RuntimeError):
    pass


class _Conn:
    """Minimal RESP2 client: buffered reader, blob-friendly writer, pipelining."""

    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.rd = self.sock.makefile("rb", buffering=1 << 20)

    def _send(self, parts) -> None:
        pieces = [f"*{len(parts)}\r\n".encode()]
        for p in parts:
            if isinstance(p, np.ndarray):
                p = memoryview(np.ascontiguousarray(p)).cast("B")
            elif isinstance(p, (bytes, bytearray, memoryview)):
                p = memoryview(p).cast("B")
            else:
                p = str(p).encode()
            pieces.append(f"${p.nbytes if isinstance(p, memoryview) else len(p)}\r\n".encode())
            pieces.append(p)
            pieces.append(b"\r\n")
        small = bytearray()
        for piece in pieces:
            n = piece.nbytes if isinstance(piece, memoryview) else len(piece)
            if n < 65536:
                small += piece
                continue
            if small:
                self.sock.sendall(small)
                small = bytearray()
            self.sock.sendall(piece)
        if small:
            self.sock.sendall(small)

    def _read(self) -> Any:
        line = self.rd.readline()
        if not line:
            raise ConnectionError("pion closed the connection")
        t, rest = line[:1], line[1:-2]
        if t == b"+":
            return rest.decode()
        if t == b"-":
            return PionError(rest.decode(errors="replace"))
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n < 0:
                return None
            data = self.rd.read(n + 2)
            return data[:-2]
        if t == b"*":
            n = int(rest)
            return None if n < 0 else [self._read() for _ in range(n)]
        raise PionError(f"unparseable reply {line[:60]!r}")

    def cmd(self, *parts) -> Any:
        self._send(parts)
        r = self._read()
        if isinstance(r, PionError):
            raise r
        return r

    def pipeline(self, cmds) -> list[Any]:
        for c in cmds:
            self._send(c)
        out = [self._read() for _ in cmds]
        for r in out:
            if isinstance(r, PionError):
                raise r
        return out

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class Layout:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: str          # "float16" | "bfloat16" | "float32"
    # How rows sit in the V-store's 16-bit slots:
    #   "native"   fp16 rows in fp16 slots — exact.
    #   "bf16bits" bf16 bit patterns in fp16 slots — exact. fp16 -> fp32 ->
    #              fp16 inside the server preserves every non-NaN pattern, and
    #              a bf16 value only looks like an fp16 NaN at |v| >= 2**121.
    #   "cast"     values converted to the store's format — lossy for bf16
    #              (fp16's range is narrower) and for any quantized vquant.
    # Recorded with the layout, so rows are always read back the way they
    # were written.
    enc: str = "cast"

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def encode(self) -> str:
        return f"{self.n_layers},{self.n_kv_heads},{self.head_dim},{self.dtype},{self.enc}"

    @staticmethod
    def decode(s: str) -> "Layout":
        f = s.split(",")
        return Layout(int(f[0]), int(f[1]), int(f[2]), f[3], f[4] if len(f) > 4 else "cast")


def layout_of(cache) -> Layout | None:
    """Layout of an mlx-lm prompt cache, or None when it is not made only of
    plain KVCache layers of one shape (rotating, recurrent and quantized caches
    cannot be restored from a prefix slice)."""
    if not cache:
        return None
    shapes = set()
    dtype = None
    for c in cache:
        if type(c).__name__ != "KVCache" or getattr(c, "keys", None) is None:
            return None
        _, h, _, d = c.keys.shape
        shapes.add((h, d))
        dtype = str(c.keys.dtype).rsplit(".", 1)[-1]
    if len(shapes) != 1:
        return None
    h, d = shapes.pop()
    return Layout(len(cache), h, d, dtype)


@dataclass
class Segment:
    id: int
    parent: int          # -1 for a root
    base: int            # absolute position of this segment's first row
    toks: np.ndarray     # uint32 token ids for rows [base, base + len)

    @property
    def end(self) -> int:
        return self.base + len(self.toks)


@dataclass
class Match:
    length: int = 0                              # tokens of the prompt already stored
    chain: list = field(default_factory=list)    # Segments, root -> segment holding row length-1


@dataclass
class Stats:
    lookups: int = 0
    hits: int = 0
    tokens_restored: int = 0
    restore_ms: float = 0.0
    tokens_stored: int = 0
    store_ms: float = 0.0
    segments_created: int = 0
    segments_extended: int = 0
    commits: int = 0
    commit_ms: float = 0.0
    evictions: int = 0
    compactions: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PionPrefixStore:
    BLOCK = 64
    CHUNK_ROWS = 4096        # rows per V.STOREBATCH / V.FETCH (keeps one layer's blob far under 64 MB)
    HMGET_CHUNK = 1000       # RESP frames are capped at 2048 tokens (gh #166)

    LRU_KEY = "pps:lru"          # member "<mid>:<seg id>", score = last use (unix seconds)
    SESSION_HEADROOM = 8         # prefixes kept free under the V-store's session cap
    COMPACT_IDLE_S = 2.0         # quiet time before an idle-time KV.PREFIX.SAVE

    def __init__(self, host: str = "127.0.0.1", port: int = 1974, vquant: str = "fp16",
                 budget_bytes: int = 0) -> None:
        self.host, self.port, self.vquant = host, port, vquant
        self.budget_bytes = budget_bytes     # 0 = no budget
        self._conn: _Conn | None = None
        self._lock = threading.RLock()
        self._layouts: dict[str, Layout] = {}
        self._segs: dict[tuple[str, int], Segment] = {}
        self._fp16_wire = True
        self._backfilled = False
        self._evict_ok = True
        self._last_write = 0.0
        self._compact_timer: threading.Timer | None = None
        self.last_match: Match | None = None
        self.stats = Stats()

    # ── plumbing ─────────────────────────────────────────────────────────

    @property
    def conn(self) -> _Conn:
        if self._conn is None:
            self._conn = _Conn(self.host, self.port)
        return self._conn

    def _reset_conn(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    def model_id(self, model_key) -> str:
        return hashlib.blake2b(f"{model_key!r}|{self.vquant}".encode(), digest_size=6).hexdigest()

    def _k(self, mid: str, *parts) -> str:
        return ":".join(("pps", mid) + tuple(str(p) for p in parts))

    def _ns(self, mid: str, seg_id: int) -> str:
        return f"pps{mid}s{seg_id}"

    def block_hashes(self, tokens) -> list[str]:
        raw = np.asarray(tokens, dtype=np.uint32).tobytes()
        step = self.BLOCK * 4
        h = b""
        out = []
        for k in range(len(tokens) // self.BLOCK):
            h = hashlib.blake2b(h + raw[k * step:(k + 1) * step], digest_size=12).digest()
            out.append(h.hex())
        return out

    def _layout(self, mid: str) -> Layout | None:
        lay = self._layouts.get(mid)
        if lay is None:
            s = self.conn.cmd("GET", self._k(mid, "layout"))
            if s is None:
                return None
            lay = self._layouts[mid] = Layout.decode(s.decode())
        return lay

    def _segment(self, mid: str, seg_id: int) -> Segment | None:
        key = (mid, seg_id)
        seg = self._segs.get(key)
        # Another process may have extended the segment; the stored length is
        # authoritative, so re-read when it moved.
        n = self.conn.cmd("HGET", self._k(mid, "seg", seg_id), "len")
        if n is None:
            self._segs.pop(key, None)
            return None
        n = int(n)
        if seg is not None and len(seg.toks) == n:
            return seg
        parent, base, toks = self.conn.cmd("HMGET", self._k(mid, "seg", seg_id), "parent", "base", "toks")
        seg = Segment(seg_id, int(parent), int(base), np.frombuffer(toks or b"", dtype=np.uint32)[:n].copy())
        self._segs[key] = seg
        return seg

    def _chain(self, mid: str, seg_id: int) -> list[Segment] | None:
        chain = []
        while seg_id >= 0:
            seg = self._segment(mid, seg_id)
            if seg is None:
                return None
            chain.append(seg)
            seg_id = seg.parent
        chain.reverse()
        return chain

    # ── budget ───────────────────────────────────────────────────────────

    def _touch_cmds(self, mid: str, segs) -> list:
        """ZADD of every segment in `segs` at one timestamp. Callers pass a
        whole chain, which is what keeps every parent at least as recent as
        its children (see the module docstring)."""
        if not segs:
            return []
        now = time.time()
        args = []
        for sg in segs:
            args += [f"{now:.6f}", f"{mid}:{sg.id}"]
        return [("ZADD", self.LRU_KEY, *args)]

    def info(self) -> dict:
        """KV.PREFIX.INFO as a dict of ints (non-numeric fields dropped)."""
        raw = self.conn.cmd("KV.PREFIX.INFO")
        out = {}
        for line in (raw or b"").split(b"\r\n"):
            k, _, v = line.partition(b":")
            try:
                out[k.decode()] = int(v)
            except ValueError:
                pass
        return out

    def _row_bytes(self, lay: Layout) -> int:
        """V-store bytes per token row of a lineage: K and V, every layer."""
        per = 2 if self.vquant == "fp16" else 1
        return 2 * lay.n_layers * lay.kv_dim * per

    def _backfill_lru(self) -> None:
        """Segments stored before this process kept an LRU (or by an older
        serve) get last-use 0, so they are the first candidates."""
        if self._backfilled:
            return
        self._backfilled = True
        cursor, members = b"0", []
        while True:
            cursor, keys = self.conn.cmd("SCAN", cursor, "MATCH", "pps:*:seg:*", "COUNT", 1000)
            for k in keys:
                f = k.decode().split(":")
                if len(f) == 4 and f[3].isdigit():
                    members += ["0", f"{f[1]}:{f[3]}"]
            if cursor in (b"0", "0", 0):
                break
        for i in range(0, len(members), 2 * self.HMGET_CHUNK):
            self.conn.cmd("ZADD", self.LRU_KEY, "NX", *members[i:i + 2 * self.HMGET_CHUNK])

    def _drop_segment(self, mid: str, sid: int) -> int:
        """Evict one segment: its V-store rows (WAL-logged by the server), its
        metadata, its LRU entry, and the block-index entries that point at it.
        Returns the V-store bytes it held (estimated from its length)."""
        seg = self._segment(mid, sid)
        lay = self._layout(mid)
        freed = len(seg.toks) * self._row_bytes(lay) if seg is not None and lay is not None else 0
        cmds = [("KV.PREFIX.DROP", self._ns(mid, sid)),
                ("DEL", self._k(mid, "seg", sid)),
                ("ZREM", self.LRU_KEY, f"{mid}:{sid}")]
        chain = self._chain(mid, sid) if seg is not None else None
        if chain:
            # The block index keys are chained hashes of the whole prefix, so
            # rebuild the tokens from the root: each segment up to where its
            # child branched off, then this one in full.
            parts = [c.toks[:chain[i + 1].base - c.base] for i, c in enumerate(chain[:-1])] + [seg.toks]
            toks = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)
            hs = self.block_hashes(toks)
            fields = [(hs[k], f"{sid}:{(k + 1) * self.BLOCK}") for k in range(len(hs))
                      if seg.base < (k + 1) * self.BLOCK <= seg.end]
            for i in range(0, len(fields), self.HMGET_CHUNK):
                part = fields[i:i + self.HMGET_CHUNK]
                cur = self.conn.cmd("HMGET", self._k(mid, "blk"), *[h for h, _ in part])
                # Only entries still pointing here: another segment may have
                # re-stored the same prefix since.
                stale = [h for (h, want), got in zip(part, cur) if got is not None and got.decode() == want]
                if stale:
                    cmds.append(("HDEL", self._k(mid, "blk"), *stale))
        self.conn.pipeline(cmds)
        self._segs.pop((mid, sid), None)
        self.stats.evictions += 1
        return freed

    def _enforce_budget(self, protect: set) -> None:
        """Evict least-recently-used segments (never one in `protect`, the
        chain being written) until the V-store is under budget and clear of
        its session cap; then schedule compaction if the WAL outgrew the
        budget. Called with the lock held, after every store."""
        info = self.info()
        held = info.get("vstore_bytes", 0)
        sessions = info.get("vstore_sessions", 0)
        cap = info.get("vstore_max_sessions", 0) - 2 * self.SESSION_HEADROOM

        def over() -> bool:
            return (self.budget_bytes > 0 and held > self.budget_bytes) or (cap > 0 and sessions > cap)

        if over():
            self._backfill_lru()
            flat = self.conn.cmd("ZRANGE", self.LRU_KEY, 0, 511, "WITHSCORES")
            cands = []
            for i in range(0, len(flat), 2):
                mid, _, sid = flat[i].decode().rpartition(":")
                cands.append((float(flat[i + 1]), -int(sid), mid, int(sid)))
            cands.sort()     # oldest first; equal times -> newest id (the child) first
            for _, _, mid, sid in cands:
                if not over():
                    break
                if (mid, sid) in protect:
                    continue
                held -= self._drop_segment(mid, sid)
                sessions -= 2
            if over():
                log.warning("prefix budget: still over after evicting every unprotected segment "
                            "(held %d bytes, budget %d, sessions %d)", held, self.budget_bytes, sessions)
        if self.budget_bytes and info.get("wal_bytes", 0) > self.budget_bytes:
            self._schedule_compaction()

    def _schedule_compaction(self, delay: float | None = None) -> None:
        if self._compact_timer is not None and self._compact_timer.is_alive():
            return
        t = threading.Timer(self.COMPACT_IDLE_S if delay is None else delay, self._compact_if_idle)
        t.daemon = True
        self._compact_timer = t
        t.start()

    def _compact_if_idle(self) -> None:
        """KV.PREFIX.SAVE once nothing has been stored for COMPACT_IDLE_S: the
        snapshot holds only live rows, and the server truncates the WAL behind
        it. It blocks the server while it writes (~1 s per GB), so it waits
        for a quiet moment rather than running mid-answer."""
        with self._lock:
            quiet = time.monotonic() - self._last_write
            if quiet < self.COMPACT_IDLE_S:
                self._compact_timer = None
                self._schedule_compaction(self.COMPACT_IDLE_S - quiet + 0.05)
                return
            try:
                before = self.info().get("wal_bytes", 0)
                if before <= self.budget_bytes:
                    return
                t0 = time.perf_counter()
                self.conn.cmd("KV.PREFIX.SAVE")
                self.stats.compactions += 1
                log.info("pion: compacted the V-store WAL (%.0f MB) in %.1f s",
                         before / 1e6, time.perf_counter() - t0)
            except Exception as e:
                log.warning("pion compaction failed: %s", e)
                self._reset_conn()

    # ── lookup / restore ─────────────────────────────────────────────────

    def lookup(self, model_key, tokens) -> Match:
        """Longest stored prefix of `tokens` for this model."""
        with self._lock:
            self.stats.lookups += 1
            mid = self.model_id(model_key)
            hs = self.block_hashes(tokens)
            if not hs:
                return Match()
            idx = self._k(mid, "blk")
            vals = []
            for i in range(0, len(hs), self.HMGET_CHUNK):
                vals.extend(self.conn.cmd("HMGET", idx, *hs[i:i + self.HMGET_CHUNK]))
            # Longest indexed block first; if its segment is gone (V-store
            # eviction) fall back to the next-longest one held by another.
            chain, pos, tried = None, 0, set()
            for i in range(len(vals) - 1, -1, -1):
                if vals[i] is None:
                    continue
                seg_id, pos = (int(x) for x in vals[i].split(b":"))
                if seg_id in tried:
                    continue
                tried.add(seg_id)
                chain = self._chain(mid, seg_id)
                if chain is not None or len(tried) >= 4:
                    break
            if chain is None:
                return Match()
            leaf = chain[-1]
            # Refine past the block boundary inside the hit segment.
            m = pos
            own = leaf.toks
            while m < len(tokens) and m < leaf.end and tokens[m] == own[m - leaf.base]:
                m += 1
            return Match(m, chain)

    def fetch_rows(self, model_key, match: Match, start: int, end: int):
        """K and V rows [start, end) of a matched lineage, per layer, as numpy
        arrays shaped (n_kv_heads, end - start, head_dim) in the stored
        precision; None if the lineage is gone (V-store eviction)."""
        with self._lock:
            mid = self.model_id(model_key)
            lay = self._layout(mid)
            if lay is None or end <= start:
                return None
            t0 = time.perf_counter()
            pieces = []   # (ns, local_start, local_end)
            for i, seg in enumerate(match.chain):
                seg_hi = match.chain[i + 1].base if i + 1 < len(match.chain) else seg.end
                a, b = max(seg.base, start), min(seg_hi, end)
                if b > a:
                    pieces.append((self._ns(mid, seg.id), a - seg.base, b - seg.base))
            # The metadata (keyspace) and the rows (V-store) are separate logs.
            # After a power cut either can be durable further than the other,
            # and the V-store zero-fills rows it does not hold — so a restore
            # that trusted the metadata would feed zeros to the model as K/V.
            # Check every layer holds the rows first; heal and refuse if not.
            infos = self.conn.pipeline([("V.INFO", ns + side) for ns, _, _ in pieces for side in ("_pk", "_pv")])
            for idx, (ns, a, b) in enumerate(pieces):
                live = min(_live_rows(infos[2 * idx]), _live_rows(infos[2 * idx + 1]))
                if live < b:
                    seg = next(sg for sg in match.chain if self._ns(mid, sg.id) == ns)
                    if live == 0:
                        # Gone from the V-store altogether (evicted, or never
                        # durable): forget it, so lookups fall back to what
                        # survives instead of finding it again.
                        log.warning("segment %s is no longer in the V-store; forgetting it", ns)
                        try:
                            self._drop_segment(mid, seg.id)
                        except PionError as e:
                            log.debug("pion: forgetting %s: %s", ns, e)
                        return None
                    log.warning("segment %s holds %d rows but its metadata claims %d; "
                                "trimming it and treating this as a miss", ns, live, len(seg.toks))
                    self.stats.errors += 1
                    seg.toks = seg.toks[:max(0, live)]
                    self.conn.cmd("HSET", self._k(mid, "seg", seg.id), "len", len(seg.toks), "toks", seg.toks.tobytes())
                    return None
            cmds, rows_of = [], []
            for ns, a, b in pieces:
                for s in range(a, b, self.CHUNK_ROWS):
                    e = min(b, s + self.CHUNK_ROWS)
                    cmds.append(("V.FETCH", ns + "_pk", "BATCH", s, e, lay.n_layers, "FMT", "NATIVE"))
                    cmds.append(("V.FETCH", ns + "_pv", "BATCH", s, e, lay.n_layers, "FMT", "NATIVE"))
                    rows_of += [e - s, e - s]
            try:
                replies = self.conn.pipeline(cmds)
            except PionError as e:
                # The V-store evicted part of this lineage. Forget the missing
                # segments so the next lookup falls back to what survives.
                log.warning("prefix fetch failed (%s); treating as a miss", e)
                self.stats.errors += 1
                for seg in match.chain:
                    if self.conn.cmd("KV.PREFIX.LOOKUP", self._ns(mid, seg.id)) != "HIT":
                        self.conn.cmd("DEL", self._k(mid, "seg", seg.id))
                        self._segs.pop((mid, seg.id), None)
                return None
            ks = [[] for _ in range(lay.n_layers)]
            vs = [[] for _ in range(lay.n_layers)]
            for j, rep in enumerate(replies):
                dst = ks if j % 2 == 0 else vs
                for li, payload in enumerate(rep):
                    # FMT NATIVE replies raw fp16 for fp16-stored layers and fp32
                    # otherwise (gh #193); the known row count disambiguates.
                    fp16 = len(payload) == rows_of[j] * lay.kv_dim * 2
                    arr = np.frombuffer(payload, dtype=np.float16 if fp16 else np.float32)
                    dst[li].append(arr.reshape(-1, lay.n_kv_heads, lay.head_dim))
            out = []
            for li in range(lay.n_layers):
                k = np.concatenate(ks[li]).transpose(1, 0, 2)
                v = np.concatenate(vs[li]).transpose(1, 0, 2)
                if k.shape[1] != end - start:
                    log.warning("prefix fetch returned %d rows, wanted %d", k.shape[1], end - start)
                    self.stats.errors += 1
                    return None
                out.append((k, v))
            # The value receipt (PION.STATS / INFO): these rows were restored
            # with V.FETCH, which the server's ledger counts as bytes only —
            # name the leaf and the token count so it counts a hit and the
            # prefill it avoided. Same round trip refreshes the chain's LRU.
            try:
                self.conn.pipeline([("KV.PREFIX.LOOKUP", self._ns(mid, match.chain[-1].id), "TOKENS", end - start),
                                    *self._touch_cmds(mid, match.chain)])
            except PionError as e:
                log.debug("pion: receipt/LRU update failed: %s", e)
            self._last_write = time.monotonic()
            self.stats.hits += 1
            self.stats.tokens_restored += end - start
            self.stats.restore_ms += (time.perf_counter() - t0) * 1000
            return out

    # ── store ────────────────────────────────────────────────────────────

    # ── decode journal ───────────────────────────────────────────────────
    # The tokens an answer has produced so far, so a restarted server can
    # continue it instead of starting over. Rows for prompt + tokens[:-1] live
    # in the lineage (written by the same store() calls); the journal holds
    # the token ids and dies with the answer (or after a day).

    JOURNAL_TTL = 86400

    def journal_get(self, model_key, jkey: str) -> list[int] | None:
        with self._lock:
            b = self.conn.cmd("GET", self._k(self.model_id(model_key), "jr", jkey))
            return None if not b else np.frombuffer(b, dtype=np.uint32).tolist()

    def journal_put(self, model_key, jkey: str, tokens) -> None:
        # Plain SET, then EXPIRE — not `SET ... EX`: the server's option-taking
        # SET path spells every byte >= 0x80 as '?' (it round-trips the value
        # through RESP3Token.value()), which turned token 3577 into 3391.
        key = self._k(self.model_id(model_key), "jr", jkey)
        with self._lock:
            self.conn.pipeline([("SET", key, np.asarray(tokens, dtype=np.uint32).tobytes()),
                                ("EXPIRE", key, self.JOURNAL_TTL)])

    def commit(self) -> float:
        """KV.PREFIX.COMMIT: everything written so far is on stable storage
        (power cut and kernel panic included) when this returns. Returns the
        milliseconds the barrier took."""
        t0 = time.perf_counter()
        with self._lock:
            self.conn.cmd("KV.PREFIX.COMMIT")
        ms = (time.perf_counter() - t0) * 1000
        self.stats.commits += 1
        self.stats.commit_ms += ms
        return ms

    def journal_drop(self, model_key, jkey: str) -> None:
        with self._lock:
            self.conn.cmd("DEL", self._k(self.model_id(model_key), "jr", jkey))

    def _encoding_for(self, lay: Layout) -> str:
        if self.vquant != "fp16" or not self._fp16_wire:
            return "cast"
        return {"float16": "native", "bfloat16": "bf16bits"}.get(lay.dtype, "cast")

    def _write_rows(self, cache, ns, m, n, local, lay) -> None:
        # 16-bit rows on the wire when the V-store keeps 16 bits anyway: half
        # the bytes, and the server logs exactly these values (WAL op 5).
        mx = _mx()
        f16 = lay.enc in ("native", "bf16bits")
        opt = ("FMT", "F16") if f16 else ()
        cmds = []
        for li, c in enumerate(cache):
            if lay.enc == "bf16bits":
                k = _bits(c.keys, m, n, lay.kv_dim)
                v = _bits(c.values, m, n, lay.kv_dim)
            else:
                dt = mx.float16 if f16 else mx.float32
                k = _rows(c.keys, m, n, lay.kv_dim, dt)
                v = _rows(c.values, m, n, lay.kv_dim, dt)
            for s in range(0, n - m, self.CHUNK_ROWS):
                e = min(n - m, s + self.CHUNK_ROWS)
                cmds.append(("V.STOREBATCH", ns + "_pk", li, local + s, e - s, k[s:e], *opt))
                cmds.append(("V.STOREBATCH", ns + "_pv", li, local + s, e - s, v[s:e], *opt))
            if len(cmds) >= 64:
                self.conn.pipeline(cmds)
                cmds = []
        if cmds:
            self.conn.pipeline(cmds)

    def store(self, model_key, tokens, cache, match: Match | None = None) -> int:
        """Write the rows of `cache` that Pion does not hold yet. Returns the
        number of token rows written."""
        lay = layout_of(cache)
        if lay is None:
            return 0
        n = min(len(tokens), cache[0].offset)
        with self._lock:
            mid = self.model_id(model_key)
            known = self._layout(mid)
            if known is None:
                lay.enc = self._encoding_for(lay)
                self.conn.cmd("SET", self._k(mid, "layout"), lay.encode())
                self._layouts[mid] = lay
            elif (known.n_layers, known.n_kv_heads, known.head_dim, known.dtype) != \
                    (lay.n_layers, lay.n_kv_heads, lay.head_dim, lay.dtype):
                log.warning("layout changed for model %s (%s -> %s); not storing", mid, known, lay)
                return 0
            else:
                lay = known
            tokens = list(tokens[:n])
            match = match if match is not None and match.length <= n else self.lookup(model_key, tokens)
            m = match.length
            leaf = match.chain[-1] if match.chain else None
            # A supplied match may stop short of what the leaf already holds
            # (a greedy retry regenerates the very tokens an earlier attempt
            # stored). Walk the leaf's own tokens first, or the same rows get
            # stored again in a duplicate branch.
            while leaf is not None and m < n and m < leaf.end and tokens[m] == leaf.toks[m - leaf.base]:
                m += 1
            if m != match.length:
                match = Match(m, match.chain)
            if m >= n:
                self.last_match = match
                return 0
            t0 = time.perf_counter()
            if leaf is not None and m == leaf.end:
                seg, local = leaf, m - leaf.base
                self.stats.segments_extended += 1
            else:
                sid = self.conn.cmd("INCR", self._k(mid, "nextseg"))
                ns = self._ns(mid, sid)
                r = self.conn.cmd("KV.PREFIX.REGISTER", ns, lay.kv_dim, self.vquant)
                if r != "OK":
                    raise PionError(f"KV.PREFIX.REGISTER {ns}: {r}")
                seg = Segment(sid, leaf.id if leaf is not None else -1, m, np.zeros(0, dtype=np.uint32))
                self.conn.cmd("HSET", self._k(mid, "seg", sid), "parent", seg.parent, "base", m, "len", 0, "toks", b"")
                local = 0
                self.stats.segments_created += 1
            ns = self._ns(mid, seg.id)
            try:
                self._write_rows(cache, ns, m, n, local, lay)
            except PionError as e:
                # A server predating `FMT F16` reads the blob as fp32 and refuses
                # its size; resend as fp32 (twice the wire) from then on.
                if not self._fp16_wire or "blob size" not in str(e):
                    raise
                self._fp16_wire = False
                if lay.enc != "cast":
                    # Rows already written 16-bit cannot be mixed with cast
                    # ones under one layout; stop storing for this model.
                    log.warning("server lacks FMT F16; not storing %s", lay.encode())
                    return 0
                self._write_rows(cache, ns, m, n, local, lay)
            new_toks = np.asarray(tokens[m:n], dtype=np.uint32)
            seg.toks = np.concatenate([seg.toks, new_toks])
            hs = self.block_hashes(tokens)
            first_block = m // self.BLOCK        # blocks ending in (m, n]
            fields = []
            for k in range(first_block, n // self.BLOCK):
                end = (k + 1) * self.BLOCK
                if end > m:
                    fields += [hs[k], f"{seg.id}:{end}"]
            chain = list(match.chain)
            if not chain or chain[-1] is not seg:
                chain.append(seg)
            pipe = [("HSET", self._k(mid, "seg", seg.id), "len", len(seg.toks), "toks", seg.toks.tobytes())]
            for i in range(0, len(fields), 2 * self.HMGET_CHUNK):
                pipe.append(("HSET", self._k(mid, "blk"), *fields[i:i + 2 * self.HMGET_CHUNK]))
            pipe += self._touch_cmds(mid, chain)
            self.conn.pipeline(pipe)
            self._segs[(mid, seg.id)] = seg
            self._last_write = time.monotonic()
            if self._evict_ok:
                try:
                    self._enforce_budget({(mid, sg.id) for sg in chain})
                except PionError as e:
                    # A server without KV.PREFIX.DROP / wal_bytes: keep storing,
                    # stop evicting, say so once.
                    log.warning("prefix budget disabled (server too old?): %s", e)
                    self._evict_ok = False
            # The lineage now ends at n in `seg`: a caller extending it again
            # (the decode journal, every 16 tokens) passes this back and skips
            # the lookup.
            self.last_match = Match(n, chain)
            self.stats.tokens_stored += n - m
            self.stats.store_ms += (time.perf_counter() - t0) * 1000
            return n - m


def _rows(arr, m, n, kv_dim, dtype):
    mx = _mx()
    return np.array(arr[0, :, m:n, :].astype(dtype)).transpose(1, 0, 2).reshape(n - m, kv_dim)


def _live_rows(info) -> int:
    """Rows every layer of a V-store session holds, from V.INFO; 0 if the
    session is gone."""
    if not isinstance(info, (bytes, bytearray)):
        return 0
    counts = [int(line.split(b":")[1]) for line in info.split(b"\r\n")
              if line.startswith(b"layer_") and b"_tokens:" in line]
    return min(counts) if counts else 0


def _bits(arr, m, n, kv_dim):
    """bf16 rows as their raw bit patterns, typed fp16 for the wire."""
    mx = _mx()
    u16 = mx.view(arr[0, :, m:n, :], mx.uint16)
    return np.array(u16).transpose(1, 0, 2).reshape(n - m, kv_dim).view(np.float16)


def to_mx(rows: np.ndarray, lay: Layout):
    """Fetched rows (n_kv_heads, n, head_dim) back into the model's dtype."""
    mx = _mx()
    if lay.enc == "bf16bits":
        return mx.view(mx.array(np.ascontiguousarray(rows).view(np.uint16)), mx.bfloat16)
    return mx.array(rows, dtype=getattr(mx, lay.dtype))


def _mx():
    import mlx.core as mx
    return mx
