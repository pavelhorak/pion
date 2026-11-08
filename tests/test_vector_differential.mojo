# D11 differential: every routine libpion_vector closes must return what its
# open reference returns, on the same inputs.
#   INT8 beam, three quantized beams   same heap/pool entries, same float bits,
#                                      same order
#   PKM INT8 scores, top-k, combine,   bit-exact
#   sort
#   PKM FP32 scores / rescore          |diff| <= 1e-4 * (1 + |score|): the
#                                      tuned kernels sum in a different order
#
#   pixi run test-vector-differential
#
# It needs the library (-D PION_HELD_VECTOR + the vendored archive) and refuses
# to run without it, so it cannot pass by comparing the reference to itself.
#
# The graph is synthetic (random adjacency, random deletions, per-node
# amplitudes so the prune phase's early-exit branch fires): the property under
# test is "same routine output for the same state", which does not need a
# well-built HNSW — only one that exercises every lane path.
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from std.memory import alloc, unsafe_memset
from std.random import random_si64, seed
from std.sys import is_defined

from src.common.heap import LinearPool
from src.common.ptr import null_ptr
from src.vector.hnsw_types import HNSWNode
from src.vector.beam_view import BeamView1536
from src.vector.reference.beam_1536 import beam_search_1536_ref
from src.common.heap import HeapNode, MinHeap, MaxHeap
from src.vector.quant_beam_view import QuantBeamView1536
from src.vector.reference.quant_beams_1536 import quant_beam_search_1536_ref
from src.vector.pkm_kernels import pkm_quantize_rows_i8, pkm_pad_dim
from src.vector.reference.pkm_ref import (
    ref_pkm_scores_f32, ref_pkm_rescore_f32, ref_pkm_scores_i8, ref_pkm_topk_max,
    ref_pkm_combine_topk, ref_pkm_sort_desc,
)
from std.random import random_float64, random_ui64

comptime DIM = 1536
comptime STRIDE = 1600
comptime HDR = 8
comptime HELD = is_defined["PION_HELD_VECTOR"]()


struct Graph:
    var n: Int
    var buf: Pointer[Int8, MutUntrackedOrigin]
    var adj: Pointer[UInt32, MutUntrackedOrigin]
    var slots: Pointer[UInt32, MutUntrackedOrigin]
    var deleted: Pointer[UInt8, MutUntrackedOrigin]
    var visited: Pointer[UInt16, MutUntrackedOrigin]
    var query: Pointer[Int8, MutUntrackedOrigin]
    var view: Pointer[BeamView1536, MutUntrackedOrigin]  # heap: see HNSWGraph.beam_view

    def __init__(out self, n: Int):
        self.n = n
        self.buf = unsafe_alloc[Int8](n * STRIDE)
        self.adj = unsafe_alloc[UInt32](n * 33)
        self.slots = unsafe_alloc[UInt32](n * 33)
        self.deleted = unsafe_alloc[UInt8](n // 8 + 1)
        self.visited = unsafe_alloc[UInt16](n)
        self.query = unsafe_alloc[Int8](DIM)
        self.view = unsafe_alloc[BeamView1536](1)
        unsafe_memset(self.deleted, 0, n // 8 + 1)
        unsafe_memset(self.visited, 0, n)
        for i in range(n):
            var amp = random_si64(4, 127)
            var v = self.buf.unsafe_offset(i * STRIDE).unsafe_offset(HDR)
            var nsq = 0
            for d in range(DIM):
                var x = random_si64(-amp, amp)
                v[unsafe_offset=d] = Int8(x)
                nsq += Int(x * x)
            v.unsafe_offset(-8).unsafe_bitcast[Float32]()[unsafe_offset=0] = Float32(nsq)
            var cnt = Int(random_si64(8, 32))
            self.adj[unsafe_offset=i * 33] = UInt32(cnt)
            for k in range(cnt):
                var nb = Int(random_si64(0, Int64(n - 1)))
                self.adj[unsafe_offset=i * 33 + 1 + k] = UInt32(nb)
                self.slots[unsafe_offset=i * 33 + 1 + k] = UInt32(nb)  # slot == node index here
            if random_si64(0, 99) < 2:
                self.deleted[unsafe_offset=i >> 3] |= UInt8(1 << (i & 7))

    def new_query(mut self):
        var amp = random_si64(4, 127)
        for d in range(DIM):
            self.query[unsafe_offset=d] = Int8(random_si64(-amp, amp))


def l2_exact(a: Pointer[Int8, MutUntrackedOrigin], b: Pointer[Int8, MutUntrackedOrigin]) -> Float32:
    var s = 0
    for i in range(DIM):
        var d = Int(a[unsafe_offset=i]) - Int(b[unsafe_offset=i])
        s += d * d
    return Float32(s)


def qnorm(q: Pointer[Int8, MutUntrackedOrigin], n: Int) -> Float32:
    var s = 0
    for i in range(n):
        s += Int(q[unsafe_offset=i]) * Int(q[unsafe_offset=i])
    return Float32(s)


def run(mut g: Graph, mut pool: LinearPool, ef: Int, epoch: UInt16, entry: Int, which: Int):
    """Which: 1 reference, 2 library."""
    unsafe_memset(g.visited, 0, g.n)
    g.visited[unsafe_offset=entry] = epoch
    pool.reset(ef)
    _ = pool.insert(l2_exact(g.query, g.buf.unsafe_offset(entry * STRIDE).unsafe_offset(HDR)), entry)
    var vp = g.view
    vp.unsafe_write(BeamView1536(
        rebind[Pointer[LinearPool, MutUntrackedOrigin]](Pointer(to=pool)),
        ef, qnorm(g.query, DIM), qnorm(g.query, 256), g.query,
        g.adj, g.slots, g.n, g.deleted, g.visited, epoch,
        g.buf, STRIDE, HDR, null_ptr[HNSWNode, MutUntrackedOrigin]()))
    if which == 1:
        beam_search_1536_ref(vp)
    else:
        comptime if HELD:
            _ = external_call["pion_v_beam_1536", Int](vp)


def same(a: LinearPool, b: LinearPool) -> Bool:
    if a.size != b.size or a.cursor != b.cursor: return False
    for i in range(a.size):
        if a.entries[unsafe_offset=i].id != b.entries[unsafe_offset=i].id: return False
        if a.entries[unsafe_offset=i].dist.to_bits() != b.entries[unsafe_offset=i].dist.to_bits(): return False
    return True


def int8_beam() -> Int:
    seed(1974)
    var g = Graph(20000)
    var p_ref = LinearPool()
    var p_lib = LinearPool()
    var bad = 0
    var entries = 0
    var efs = [10, 64, 150, 300]
    for qi in range(400):
        g.new_query()
        var entry = Int(random_si64(0, Int64(g.n - 1)))
        while Bool((g.deleted[unsafe_offset=entry >> 3] >> UInt8(entry & 7)) & 1):
            entry = Int(random_si64(0, Int64(g.n - 1)))
        var ef = efs[qi % 4]
        run(g, p_ref, ef, UInt16(1), entry, 1)
        run(g, p_lib, ef, UInt16(1), entry, 2)
        entries += p_ref.size
        if not same(p_ref, p_lib):
            bad += 1
            if bad <= 3:
                print("  INT8 DIFFER query", qi, "ef", ef, "sizes", p_ref.size, p_lib.size)
    print("int8 beam:  400 queries,", entries, "pool entries, differ", bad)
    return bad


# ── quantized beams ─────────────────────────────────────────────────────────

struct QGraph:
    var n: Int
    var kind: Int
    var stride: Int
    var buf: Pointer[Int8, MutUntrackedOrigin]
    var adj: Pointer[UInt32, MutUntrackedOrigin]
    var deleted: Pointer[UInt8, MutUntrackedOrigin]
    var visited: Pointer[UInt16, MutUntrackedOrigin]
    var q8: Pointer[Int8, MutUntrackedOrigin]
    var qs: Pointer[Float32, MutUntrackedOrigin]
    var qjl: Pointer[UInt64, MutUntrackedOrigin]
    var qjl_norms: Pointer[Float32, MutUntrackedOrigin]
    var qsigns: Pointer[UInt64, MutUntrackedOrigin]
    var view: Pointer[QuantBeamView1536, MutUntrackedOrigin]

    def __init__(out self, n: Int, kind: Int):
        """Kind 0 = INT4 (18 B blocks), 1 = INT2 (10 B), 2 = INT3 + QJL (14 B)."""
        self.n = n
        self.kind = kind
        var blk = 18 if kind == 0 else (10 if kind == 1 else 14)
        self.stride = 4 + 48 * blk
        self.buf = unsafe_alloc[Int8](n * self.stride)
        self.adj = unsafe_alloc[UInt32](n * 33)
        self.deleted = unsafe_alloc[UInt8](n // 8 + 1)
        self.visited = unsafe_alloc[UInt16](n)
        self.q8 = unsafe_alloc[Int8](DIM)
        self.qs = unsafe_alloc[Float32](48)
        self.qjl = unsafe_alloc[UInt64](n * 24)
        self.qjl_norms = unsafe_alloc[Float32](n)
        self.qsigns = unsafe_alloc[UInt64](24)
        self.view = unsafe_alloc[QuantBeamView1536](1)
        unsafe_memset(self.deleted, 0, n // 8 + 1)
        for i in range(n):
            var v = self.buf.unsafe_offset(i * self.stride)
            for b in range(self.stride):
                v[unsafe_offset=b] = Int8(random_si64(-128, 127))
            for b in range(48):
                (v.unsafe_offset(4).unsafe_offset(b * blk)).unsafe_bitcast[Float16]()[unsafe_offset=0] = Float16(random_float64(0.001, 0.5))
            v.unsafe_bitcast[Float32]()[unsafe_offset=0] = Float32(random_float64(0.5, 2.0))
            var cnt = Int(random_si64(8, 32))
            self.adj[unsafe_offset=i * 33] = UInt32(cnt)
            for k in range(cnt):
                self.adj[unsafe_offset=i * 33 + 1 + k] = UInt32(Int(random_si64(0, Int64(n - 1))))
            if random_si64(0, 99) < 2:
                self.deleted[unsafe_offset=i >> 3] |= UInt8(1 << (i & 7))
            for w in range(24):
                self.qjl[unsafe_offset=i * 24 + w] = random_ui64(0, UInt64.MAX)
            self.qjl_norms[unsafe_offset=i] = Float32(random_float64(0.0, 0.01))

    def new_query(mut self):
        for d in range(DIM):
            self.q8[unsafe_offset=d] = Int8(random_si64(-127, 127))
        for b in range(48):
            self.qs[unsafe_offset=b] = Float32(random_float64(0.001, 0.1))
        for w in range(24):
            self.qsigns[unsafe_offset=w] = random_ui64(0, UInt64.MAX)


def qrun(mut g: QGraph, mut cand: MinHeap, mut res: MaxHeap, ef: Int, entry: Int, lib: Bool):
    unsafe_memset(g.visited, 0, g.n)
    g.visited[unsafe_offset=entry] = UInt16(1)
    cand.data.clear()
    res.data.clear()
    res._cached_worst = Float32(-1e30)
    cand.reserve(ef + 32)
    res.reserve(ef + 32)
    cand.push(HeapNode(Float32(0.0), entry))
    res.push(HeapNode(Float32(0.0), entry))
    var vp = g.view
    vp.unsafe_write(QuantBeamView1536(
        g.kind, ef,
        rebind[Pointer[MinHeap, MutUntrackedOrigin]](Pointer(to=cand)),
        rebind[Pointer[MaxHeap, MutUntrackedOrigin]](Pointer(to=res)),
        g.q8, g.qs, Float32(1.0),
        g.adj, g.adj, g.n, g.deleted, g.visited, UInt16(1),
        g.buf, g.stride, 0, null_ptr[HNSWNode, MutUntrackedOrigin](),
        g.qjl if g.kind == 2 else null_ptr[UInt64, MutUntrackedOrigin](),
        g.qjl_norms if g.kind == 2 else null_ptr[Float32, MutUntrackedOrigin](),
        g.qsigns, Float32(0.005), Float32(1.0)))
    if lib:
        comptime if HELD:
            _ = external_call["pion_v_quant_beam_1536", Int](vp)
    else:
        quant_beam_search_1536_ref(vp)


def heaps_same(a: List[HeapNode], b: List[HeapNode]) -> Bool:
    if len(a) != len(b): return False
    for i in range(len(a)):
        if a[i].id != b[i].id or a[i].distance.to_bits() != b[i].distance.to_bits(): return False
    return True


def quant_beams() -> Int:
    var total_bad = 0
    var names = ["polar INT4", "nano INT2", "turbo INT3+QJL"]
    var efs = [10, 64, 150, 300]
    for kind in range(3):
        seed(346 + kind)
        var g = QGraph(8000, kind)
        var c1 = MinHeap(); var r1 = MaxHeap()
        var c2 = MinHeap(); var r2 = MaxHeap()
        var bad = 0
        var entries = 0
        for qi in range(200):
            g.new_query()
            var entry = Int(random_si64(0, Int64(g.n - 1)))
            while Bool((g.deleted[unsafe_offset=entry >> 3] >> UInt8(entry & 7)) & 1):
                entry = Int(random_si64(0, Int64(g.n - 1)))
            var ef = efs[qi % 4]
            qrun(g, c1, r1, ef, entry, False)
            qrun(g, c2, r2, ef, entry, True)
            entries += len(r1.data)
            if not heaps_same(r1.data, r2.data) or not heaps_same(c1.data, c2.data):
                bad += 1
                if bad <= 3:
                    print("  ", names[kind], "DIFFER query", qi, "ef", ef)
        print("quant beam", names[kind], ": 200 queries,", entries, "result entries, differ", bad)
        total_bad += bad
    return total_bad


# ── PKM ─────────────────────────────────────────────────────────────────────

def pkm() -> Int:
    seed(146)
    comptime S = 1000
    comptime HALF = 448
    comptime NQ = 8
    comptime K = 32
    var d_pad = pkm_pad_dim(HALF)
    var keys = unsafe_alloc[Float32](S * d_pad)
    var q = unsafe_alloc[Float32](NQ * d_pad)
    for i in range(S * d_pad):
        keys[unsafe_offset=i] = Float32(random_float64(-1.0, 1.0)) if i % d_pad < HALF else 0.0
    for i in range(NQ * d_pad):
        q[unsafe_offset=i] = Float32(random_float64(-1.0, 1.0)) if i % d_pad < HALF else 0.0
    var a = unsafe_alloc[Float32](NQ * S)
    var b = unsafe_alloc[Float32](NQ * S)
    var bad = 0

    external_call["pion_v_pkm_scores_f32", NoneType](keys, S, d_pad, q, NQ, a)
    ref_pkm_scores_f32(keys, S, d_pad, q, NQ, b)
    for i in range(NQ * S):
        if abs(a[unsafe_offset=i] - b[unsafe_offset=i]) > Float32(1e-4) * (1.0 + abs(a[unsafe_offset=i])): bad += 1

    var idx = unsafe_alloc[Int32](64)
    for i in range(64): idx[unsafe_offset=i] = Int32(random_si64(0, S - 1))
    external_call["pion_v_pkm_rescore_f32", NoneType](keys, d_pad, q, idx, 64, a)
    ref_pkm_rescore_f32(keys, d_pad, q, idx, 64, b)
    for i in range(64):
        if abs(a[unsafe_offset=i] - b[unsafe_offset=i]) > Float32(1e-4) * (1.0 + abs(a[unsafe_offset=i])): bad += 1

    var ki = unsafe_alloc[Int8](S * d_pad)
    var ks = unsafe_alloc[Float32](S)
    var qi = unsafe_alloc[Int8](NQ * d_pad)
    var qsc = unsafe_alloc[Float32](NQ)
    pkm_quantize_rows_i8(keys, S, HALF, d_pad, ki, ks)
    pkm_quantize_rows_i8(q, NQ, HALF, d_pad, qi, qsc)
    external_call["pion_v_pkm_scores_i8", NoneType](ki, S, d_pad, ks, qi, qsc, NQ, a)
    ref_pkm_scores_i8(ki, S, d_pad, ks, qi, qsc, NQ, b)
    for i in range(NQ * S):
        if a[unsafe_offset=i].to_bits() != b[unsafe_offset=i].to_bits(): bad += 1

    var ti = unsafe_alloc[Int32](2 * K); var tv = unsafe_alloc[Float32](2 * K)
    var ri = unsafe_alloc[Int32](2 * K); var rv = unsafe_alloc[Float32](2 * K)
    for h in range(2):
        var n1 = external_call["pion_v_pkm_topk_max", Int](a.unsafe_offset(h * S), S, K, ti.unsafe_offset(h * K), tv.unsafe_offset(h * K))
        var n2 = ref_pkm_topk_max(a.unsafe_offset(h * S), S, K, ri.unsafe_offset(h * K), rv.unsafe_offset(h * K))
        if n1 != n2: bad += 1
        for j in range(n1):
            if ti[unsafe_offset=h * K + j] != ri[unsafe_offset=h * K + j] or tv[unsafe_offset=h * K + j].to_bits() != rv[unsafe_offset=h * K + j].to_bits(): bad += 1
    var ci = unsafe_alloc[Int32](K); var cv = unsafe_alloc[Float32](K)
    var di = unsafe_alloc[Int32](K); var dv = unsafe_alloc[Float32](K)
    var c1 = external_call["pion_v_pkm_combine_topk", Int](tv, ti, K, tv.unsafe_offset(K), ti.unsafe_offset(K), K, S, K, ci, cv)
    var c2 = ref_pkm_combine_topk(tv, ti, K, tv.unsafe_offset(K), ti.unsafe_offset(K), K, S, K, di, dv)
    if c1 != c2: bad += 1
    for j in range(c1):
        if ci[unsafe_offset=j] != di[unsafe_offset=j] or cv[unsafe_offset=j].to_bits() != dv[unsafe_offset=j].to_bits(): bad += 1

    # sort: same shuffled input to both
    var sv1 = unsafe_alloc[Float32](500); var si1 = unsafe_alloc[Int32](500)
    var sv2 = unsafe_alloc[Float32](500); var si2 = unsafe_alloc[Int32](500)
    for i in range(500):
        sv1[unsafe_offset=i] = Float32(random_float64(-1.0, 1.0)); si1[unsafe_offset=i] = Int32(i)
        sv2[unsafe_offset=i] = sv1[unsafe_offset=i]; si2[unsafe_offset=i] = Int32(i)
    external_call["pion_v_pkm_sort_desc", NoneType](sv1, si1, 500)
    ref_pkm_sort_desc(sv2, si2, 500)
    for i in range(500):
        if si1[unsafe_offset=i] != si2[unsafe_offset=i] or sv1[unsafe_offset=i].to_bits() != sv2[unsafe_offset=i].to_bits(): bad += 1
    print("pkm:        scores f32/i8, rescore, top-k, combine, sort — differ", bad)
    return bad


def main() raises:
    comptime if not HELD:
        raise Error("test_vector_differential needs the library: pixi run test-vector-differential")
    var abi = external_call["pion_v_abi_version", Int]()
    print("libpion_vector abi", abi)
    var bad = int8_beam() + quant_beams() + pkm()
    if bad != 0:
        raise Error("libpion_vector disagrees with the open reference")
    print("PASS: every closed routine matches its open reference")
