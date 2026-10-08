"""VSET (Vector Sets) — Redis 8 vectorset-compatible commands, one set PER KEY.

gh #366: the first implementation ignored the key. Every VADD on the server
went into the one FT.* HNSW graph, element names lived in global `__vn__` /
`__vi__` keyspace entries, and so `VSIM vv …` returned elements added to `vs`,
`VCARD nokey` answered the server-wide count, `VDIM` answered the startup
dimension, and a re-VADD stored a duplicate. The differential against a real
Redis 8 agreed on 15 of 52 steps at dim 16.

Now a vector set is a keyspace value of type `ValueType.VSET` holding its own
dimension, elements, vectors and attributes:

  * vectors are stored L2-normalized in FP32 with their original norm (VEMB
    returns `unit * norm`); similarity is cosine, reported as Redis does:
    `(1 + cos) / 2`, 1 = identical;
  * VSIM is an EXACT scan of the set — no graph, so no recall loss and no
    tuning, at O(size x dim) per query. The HNSW/quantization options Redis
    accepts (Q8, NOQUANT, BIN, M, EF, CAS, NOTHREAD, TRUTH) are accepted and
    have nothing to change; REDUCE, FILTER and VEMB RAW are refused with an
    error rather than ignored;
  * VLINKS is refused: there is no graph to report links from;
  * the set is removed with its last element, like every other aggregate.

Persisted (gh #378): VADD, VREM and VSETATTR append WAL records 28/29/30 and
SAVE serializes a set as the same records (src/io/wal.mojo). VADD logs the
STORED unit vector and norm, so a restart restores the set bit for bit.

Token skip (gh #156): the `Int` these handlers return is informational; the
dispatch site sets `i = cmd_end_tok - 1` and passes `cmd_end_tok` as
`num_tokens`, so optional-argument scans stay inside their own command.
"""

from src.vector.fp32_scan import dot_f32_4chain
from src.common.ptr import is_not_null, is_null, null_ptr
from std.memory.unsafe_pointer import Pointer
from std.memory import alloc, stack_allocation
from std.memory import unsafe_memcpy
from std.collections import List, Span
from std.math import sqrt

from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.value import GenericValue, ValueType
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.vector_set import VectorSet, free_vset
from src.io.wal import WAL
from src.common.utils import format_float64_to_buf, is_valid_float_arg, parse_float64, parse_int64_strict, arg_eq

comptime WRONGTYPE_MSG = "WRONGTYPE Operation against a key holding the wrong kind of value"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lookup(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], key: GenericValue,
            mut wrongtype: Bool) -> Pointer[VectorSet, MutUntrackedOrigin]:
    """The set at `key`, or null. `wrongtype` is set when the key holds
    something else — callers answer WRONGTYPE, never "missing"."""
    wrongtype = False
    var v = keyspace[].get(key)
    if v.is_none():
        return null_ptr[VectorSet, MutUntrackedOrigin]()
    if v.type.value != ValueType.VSET:
        wrongtype = True
        return null_ptr[VectorSet, MutUntrackedOrigin]()
    return v.as_hash().bitcast[VectorSet]()


def _key_p(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    return tokens[unsafe_offset=i].ptr


def _write_str(mut writer: ResponseWriter, s: String):
    writer.append_bulk_string_response(s.unsafe_ptr(), s.byte_length())


def _write_float(mut writer: ResponseWriter, x: Float64):
    var buf = stack_allocation[48, UInt8]()
    var n = format_float64_to_buf(buf, 0, x)
    writer.append_bulk_string_response(buf, n)


def _parse_vector(tokens: Pointer[RESP3Token, MutUntrackedOrigin], mut ci: Int, num_tokens: Int,
                  out_buf: Pointer[Float32, MutUntrackedOrigin], cap: Int,
                  mut dim: Int, mut err: String):
    """FP32 <blob> | VALUES <n> <v1..vn> at token `ci`; advances `ci` past it.
    Writes at most `cap` floats into out_buf; `dim` gets the vector's length."""
    var fmt = tokens[unsafe_offset=ci]
    if arg_eq(fmt.ptr, fmt.length, "fp32"):
        if ci + 1 >= num_tokens:
            err = "ERR wrong number of arguments"
            return
        var blob = tokens[unsafe_offset=ci + 1]
        if blob.length == 0 or blob.length % 4 != 0:
            err = "ERR FP32 blob length must be a positive multiple of 4"
            return
        dim = blob.length // 4
        if dim > cap:
            err = "ERR vector dimension too large"
            return
        unsafe_memcpy(dest=out_buf.bitcast[UInt8](), src=blob.ptr, count=blob.length)
        ci += 2
        return
    if arg_eq(fmt.ptr, fmt.length, "values"):
        if ci + 1 >= num_tokens:
            err = "ERR wrong number of arguments"
            return
        var pn = parse_int64_strict(tokens[unsafe_offset=ci + 1].ptr, tokens[unsafe_offset=ci + 1].length)
        if not pn.ok or pn.value <= 0:
            err = "ERR invalid vector dimension"
            return
        dim = Int(pn.value)
        if dim > cap:
            err = "ERR vector dimension too large"
            return
        if ci + 2 + dim > num_tokens:
            err = "ERR wrong number of arguments"
            return
        for d in range(dim):
            var t = tokens[unsafe_offset=ci + 2 + d]
            if not is_valid_float_arg(t.ptr, t.length):
                err = "ERR invalid vector component"
                return
            out_buf[unsafe_offset=d] = Float32(parse_float64(t.ptr, t.length))
        ci += 2 + dim
        return
    err = "ERR expected FP32 or VALUES"


comptime _VEC_CAP = 65536


# ── VADD ──────────────────────────────────────────────────────────────────────

def handle_vadd(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
    wal: Pointer[WAL, MutUntrackedOrigin],
) -> Int:
    """VADD key [REDUCE dim] (FP32 blob | VALUES n v...) element
       [CAS] [NOQUANT | Q8 | BIN] [EF n] [SETATTR json] [M n]"""
    if i + 4 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vadd' command")
        return 0
    var ci = i + 2
    if arg_eq(tokens[unsafe_offset=ci].ptr, tokens[unsafe_offset=ci].length, "reduce"):
        writer.append_error_response("ERR VADD REDUCE (random projection) is not supported")
        return 0
    var vec = alloc[Float32](_VEC_CAP)
    var dim = 0
    var err = String("")
    _parse_vector(tokens, ci, num_tokens, vec, _VEC_CAP, dim, err)
    if err.byte_length() == 0 and ci >= num_tokens:
        err = "ERR wrong number of arguments for 'vadd' command"
    if err.byte_length() > 0:
        vec.free()
        writer.append_error_response(err)
        return 0
    var elem = tokens[unsafe_offset=ci]
    ci += 1
    var attr = String("")
    var have_attr = False
    while ci < num_tokens:
        var t = tokens[unsafe_offset=ci]
        if arg_eq(t.ptr, t.length, "setattr") and ci + 1 < num_tokens:
            var a = tokens[unsafe_offset=ci + 1]
            attr = String(StringSpan[MutUntrackedOrigin](
                unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=a.ptr, length=a.length)))
            have_attr = True
            ci += 2
        elif (arg_eq(t.ptr, t.length, "ef") or arg_eq(t.ptr, t.length, "m")) and ci + 1 < num_tokens:
            ci += 2   # graph parameters: nothing to tune in an exact set
        elif (arg_eq(t.ptr, t.length, "cas") or arg_eq(t.ptr, t.length, "noquant")
              or arg_eq(t.ptr, t.length, "q8") or arg_eq(t.ptr, t.length, "bin")):
            ci += 1
        else:
            vec.free()
            writer.append_error_response("ERR syntax error in VADD options")
            return 0
    var key = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var wrongtype = False
    var vs = _lookup(keyspace, key, wrongtype)
    if wrongtype:
        vec.free()
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    if is_not_null(vs) and vs[].dim != dim:
        vec.free()
        writer.append_error_response("ERR Vector dimension mismatch - got " + String(dim)
                                     + " but set has " + String(vs[].dim))
        return 0
    if is_null(vs):
        vs = alloc[VectorSet](1)
        vs.unsafe_write(VectorSet(dim))
        var nv = GenericValue()
        nv.type = ValueType(ValueType.VSET)
        nv.set_ptr(vs.bitcast[NoneType]())
        keyspace[].set(key, nv)
    var added = vs[].add(elem.ptr, elem.length, vec)
    var slot = vs[].find(elem.ptr, elem.length)
    if have_attr:
        vs[].attrs[slot] = attr
    vec.free()
    writer.append_int_response(Int64(1) if added else Int64(0))
    # gh #378: effect records — the vector as stored, then the attribute.
    if is_not_null(wal):
        var payload = alloc[UInt8](vs[].payload_len())
        var pl = vs[].stored_payload(slot, payload)
        _ = wal[].append_field_kv(28, _key_p(tokens, i + 1), tokens[unsafe_offset=i + 1].length,
                                  elem.ptr, elem.length, payload, pl)
        payload.free()
        if have_attr:
            # Heap copy: a short String's bytes live INLINE on this frame, and a
            # stack address must not reach the out-of-line append (gh #349 —
            # tests/test_audit_tail_alloca.py caught this one as a tail call).
            var al = attr.byte_length()
            var ab = alloc[UInt8](max(al, 1))
            unsafe_memcpy(dest=ab, src=_sp(attr), count=al)
            _ = wal[].append_field_kv(30, _key_p(tokens, i + 1), tokens[unsafe_offset=i + 1].length,
                                      elem.ptr, elem.length, ab, al)
            ab.free()
    return 0


# ── VSIM ──────────────────────────────────────────────────────────────────────

def handle_vsim(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """VSIM key (ELE element | FP32 blob | VALUES n v...) [WITHSCORES]
       [WITHATTRIBS] [COUNT n] [EPSILON d] [EF n] [TRUTH] [NOTHREAD]"""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vsim' command")
        return 0
    var key = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var wrongtype = False
    var vs = _lookup(keyspace, key, wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var ci = i + 2
    var q = alloc[Float32](_VEC_CAP)
    var dim = 0
    var err = String("")
    var ele_slot = -1
    if arg_eq(tokens[unsafe_offset=ci].ptr, tokens[unsafe_offset=ci].length, "ele"):
        if ci + 1 >= num_tokens:
            err = "ERR wrong number of arguments for 'vsim' command"
        elif is_not_null(vs):
            var e = tokens[unsafe_offset=ci + 1]
            ele_slot = vs[].find(e.ptr, e.length)
            if ele_slot < 0:
                err = "ERR element not found in set"
        ci += 2
    else:
        _parse_vector(tokens, ci, num_tokens, q, _VEC_CAP, dim, err)
    var withscores = False
    var withattribs = False
    var count = 10
    var epsilon = Float64(-1.0)
    while err.byte_length() == 0 and ci < num_tokens:
        var t = tokens[unsafe_offset=ci]
        if arg_eq(t.ptr, t.length, "withscores"):
            withscores = True; ci += 1
        elif arg_eq(t.ptr, t.length, "withattribs"):
            withattribs = True; ci += 1
        elif (arg_eq(t.ptr, t.length, "truth") or arg_eq(t.ptr, t.length, "nothread")):
            ci += 1   # every VSIM here is exact and single-threaded
        elif arg_eq(t.ptr, t.length, "count") and ci + 1 < num_tokens:
            var pc = parse_int64_strict(tokens[unsafe_offset=ci + 1].ptr, tokens[unsafe_offset=ci + 1].length)
            if not pc.ok or pc.value <= 0:
                err = "ERR invalid COUNT"
            else:
                count = Int(pc.value)
            ci += 2
        elif arg_eq(t.ptr, t.length, "epsilon") and ci + 1 < num_tokens:
            var et = tokens[unsafe_offset=ci + 1]
            if not is_valid_float_arg(et.ptr, et.length):
                err = "ERR invalid EPSILON"
            else:
                epsilon = parse_float64(et.ptr, et.length)
            ci += 2
        elif (arg_eq(t.ptr, t.length, "ef") or arg_eq(t.ptr, t.length, "filter-ef")) and ci + 1 < num_tokens:
            ci += 2
        elif arg_eq(t.ptr, t.length, "filter"):
            err = "ERR VSIM FILTER expressions are not supported"
        else:
            err = "ERR syntax error in VSIM options"
    if err.byte_length() > 0:
        q.free()
        writer.append_error_response(err)
        return 0
    if is_null(vs):
        q.free()
        writer.append_empty_array_response()   # Redis: a missing key is an empty result
        return 0
    if ele_slot >= 0:
        unsafe_memcpy(dest=q, src=vs[].vecs + ele_slot * vs[].dim, count=vs[].dim)
    else:
        if dim != vs[].dim:
            q.free()
            writer.append_error_response("ERR Vector dimension mismatch - got " + String(dim)
                                         + " but set has " + String(vs[].dim))
            return 0
        # gh #400: the query's norm on the same four-chain kernel as the
        # scan, instead of one scalar FMA chain over all `dim` floats.
        var ss = dot_f32_4chain(q, q, dim)
        if ss > 0.0:
            var inv = Float32(1.0) / sqrt(ss)
            var d = 0
            while d + 8 <= dim:
                (q + d).store((q + d).load[width=8]() * inv)
                d += 8
            while d < dim:
                q[unsafe_offset=d] = q[unsafe_offset=d] * inv
                d += 1
    var slots = List[Int]()
    var scores = List[Float64]()
    vs[].search(q, count, slots, scores)
    q.free()
    var n = len(slots)
    if epsilon >= 0.0:
        var m = 0
        while m < n and 1.0 - scores[m] <= epsilon:
            m += 1
        n = m
    var per = 1 + (1 if withscores else 0) + (1 if withattribs else 0)
    writer.append_array_header(n * per)
    for r in range(n):
        _write_str(writer, vs[].names[slots[r]])
        if withscores:
            _write_float(writer, scores[r])
        if withattribs:
            if vs[].attrs[slots[r]].byte_length() > 0:
                _write_str(writer, vs[].attrs[slots[r]])
            else:
                writer.append_null_response()
    return 0


# ── Simple readers ────────────────────────────────────────────────────────────

def handle_vcard(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vcard' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
    elif is_null(vs):
        writer.append_int_response(Int64(0))
    else:
        writer.append_int_response(Int64(vs[].live))
    return 0


def handle_vdim(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vdim' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
    elif is_null(vs):
        writer.append_error_response("ERR key does not exist")
    else:
        writer.append_int_response(Int64(vs[].dim))
    return 0


def handle_vinfo(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """Redis's nine fields, in Redis's order. quant-type is "f32" and the graph
    fields describe an exact set (no HNSW levels)."""
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vinfo' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    if is_null(vs):
        writer.append_null_response()
        return 0
    var n_attr = 0
    for s in range(vs[].n):
        if vs[].alive[unsafe_offset=s] != 0 and vs[].attrs[s].byte_length() > 0:
            n_attr += 1
    writer.append_array_header(18)
    _write_str(writer, "quant-type"); _write_str(writer, "f32")
    _write_str(writer, "hnsw-m"); writer.append_int_response(Int64(0))
    _write_str(writer, "vector-dim"); writer.append_int_response(Int64(vs[].dim))
    _write_str(writer, "projection-input-dim"); writer.append_int_response(Int64(0))
    _write_str(writer, "size"); writer.append_int_response(Int64(vs[].live))
    _write_str(writer, "max-level"); writer.append_int_response(Int64(0))
    _write_str(writer, "attributes-count"); writer.append_int_response(Int64(n_attr))
    _write_str(writer, "vset-uid"); writer.append_int_response(Int64(0))
    _write_str(writer, "hnsw-max-node-uid"); writer.append_int_response(Int64(vs[].n))
    return 0


def handle_vismember(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vismember' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var e = tokens[unsafe_offset=i + 2]
    var hit = is_not_null(vs) and vs[].find(e.ptr, e.length) >= 0
    writer.append_int_response(Int64(1) if hit else Int64(0))
    return 0


def handle_vemb(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """VEMB key element → the vector (unit direction x stored norm)."""
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vemb' command")
        return 0
    if i + 3 < num_tokens:
        writer.append_error_response("ERR VEMB RAW is not supported (vectors are stored as FP32)")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var e = tokens[unsafe_offset=i + 2]
    var slot = vs[].find(e.ptr, e.length) if is_not_null(vs) else -1
    if slot < 0:
        writer.append_null_response()
        return 0
    var norm = Float64(vs[].norms[unsafe_offset=slot])
    var v = vs[].vecs + slot * vs[].dim
    writer.append_array_header(vs[].dim)
    for d in range(vs[].dim):
        _write_float(writer, Float64(v[unsafe_offset=d]) * norm)
    return 0


def handle_vsetattr(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
    wal: Pointer[WAL, MutUntrackedOrigin],
) -> Int:
    """VSETATTR key element json → 1 when set, 0 when the key or element is
    missing (Redis). An empty string removes the attribute."""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vsetattr' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var e = tokens[unsafe_offset=i + 2]
    var slot = vs[].find(e.ptr, e.length) if is_not_null(vs) else -1
    if slot < 0:
        writer.append_int_response(Int64(0))
        return 0
    var a = tokens[unsafe_offset=i + 3]
    vs[].attrs[slot] = String(StringSpan[MutUntrackedOrigin](
        unsafe_from_utf8=Span[UInt8, MutUntrackedOrigin](unsafe_ptr=a.ptr, length=a.length)))
    writer.append_int_response(Int64(1))
    if is_not_null(wal):
        _ = wal[].append_field_kv(30, _key_p(tokens, i + 1), tokens[unsafe_offset=i + 1].length,
                                  e.ptr, e.length, a.ptr, a.length)
    return 0


def handle_vgetattr(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vgetattr' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var e = tokens[unsafe_offset=i + 2]
    var slot = vs[].find(e.ptr, e.length) if is_not_null(vs) else -1
    if slot < 0 or vs[].attrs[slot].byte_length() == 0:
        writer.append_null_response()
    else:
        _write_str(writer, vs[].attrs[slot])
    return 0


# ── VREM ──────────────────────────────────────────────────────────────────────

def handle_vrem(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
    wal: Pointer[WAL, MutUntrackedOrigin],
) -> Int:
    """VREM key element → 1 removed / 0 absent. The last element takes the key
    with it (gh #234's aggregate rule), and the set is freed (gh #369)."""
    if i + 2 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vrem' command")
        return 0
    var key = GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length)
    var wrongtype = False
    var vs = _lookup(keyspace, key, wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var e = tokens[unsafe_offset=i + 2]
    var removed = is_not_null(vs) and vs[].remove(e.ptr, e.length)
    writer.append_int_response(Int64(1) if removed else Int64(0))
    if removed and is_not_null(wal):
        # gh #378: replay applies the same last-element rule, so the one
        # record reproduces the key's removal too.
        _ = wal[].append_kv(29, _key_p(tokens, i + 1), tokens[unsafe_offset=i + 1].length,
                            e.ptr, e.length)
    if removed and vs[].live == 0:
        _ = keyspace[].remove_generic(key)
        free_vset(vs)
    return 0


# ── VRANDMEMBER ───────────────────────────────────────────────────────────────

def handle_vrandmember(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """VRANDMEMBER key [count]: no count → one element or nil; count > 0 →
    up to count DISTINCT elements; count < 0 → |count| with repetition."""
    if i + 1 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vrandmember' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var has_count = i + 2 < num_tokens
    var count = Int64(1)
    if has_count:
        var pc = parse_int64_strict(tokens[unsafe_offset=i + 2].ptr, tokens[unsafe_offset=i + 2].length)
        if not pc.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return 0
        count = pc.value
    if is_null(vs):
        if has_count:
            writer.append_empty_array_response()
        else:
            writer.append_null_response()
        return 0
    var live = List[Int]()
    for s in range(vs[].n):
        if vs[].alive[unsafe_offset=s] != 0:
            live.append(s)
    if not has_count:
        var pick = live[Int(vs[].rng.next() % UInt64(len(live)))]
        _write_str(writer, vs[].names[pick])
        return 0
    if count >= 0:
        var k = Int(count) if Int(count) < len(live) else len(live)
        # partial Fisher-Yates: the first k of a shuffled copy
        for a in range(k):
            var b = a + Int(vs[].rng.next() % UInt64(len(live) - a))
            var t = live[a]; live[a] = live[b]; live[b] = t
        writer.append_array_header(k)
        for a in range(k):
            _write_str(writer, vs[].names[live[a]])
    else:
        var m = Int(-count)
        writer.append_array_header(m)
        for _ in range(m):
            _write_str(writer, vs[].names[live[Int(vs[].rng.next() % UInt64(len(live)))]])
    return 0


# ── VLINKS / VRANGE ───────────────────────────────────────────────────────────

def handle_vlinks(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """Refused: an exact set has no HNSW graph, so there are no links to report
    — answering an invented neighbour list would be worse than an error."""
    writer.append_error_response("ERR VLINKS is not supported: Pion vector sets are searched exactly, with no HNSW graph")
    return 0


def _bytes_lt(ap: Pointer[UInt8, MutUntrackedOrigin], an: Int,
              bp: Pointer[UInt8, MutUntrackedOrigin], bn: Int) -> Bool:
    """Byte-wise lexicographic order, shorter first on a shared prefix."""
    var n = an if an < bn else bn
    for k in range(n):
        if ap[unsafe_offset=k] != bp[unsafe_offset=k]:
            return ap[unsafe_offset=k] < bp[unsafe_offset=k]
    return an < bn


def _sp(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(s.unsafe_ptr()))


def _name_lt(a: String, b: String) -> Bool:
    return _bytes_lt(_sp(a), a.byte_length(), _sp(b), b.byte_length())


def _in_lo(name: String, bp: Pointer[UInt8, MutUntrackedOrigin], bl: Int) -> Bool:
    """name satisfies the start bound: `-`, `[x` (>= x) or `(x` (> x)."""
    if bl == 1 and bp[unsafe_offset=0] == 45: return True    # -
    if bl == 1 and bp[unsafe_offset=0] == 43: return False   # +
    var np = _sp(name); var nl = name.byte_length()
    if bp[unsafe_offset=0] == 91:                             # [
        return not _bytes_lt(np, nl, bp + 1, bl - 1)
    return _bytes_lt(bp + 1, bl - 1, np, nl)


def _in_hi(name: String, bp: Pointer[UInt8, MutUntrackedOrigin], bl: Int) -> Bool:
    if bl == 1 and bp[unsafe_offset=0] == 43: return True    # +
    if bl == 1 and bp[unsafe_offset=0] == 45: return False   # -
    var np = _sp(name); var nl = name.byte_length()
    if bp[unsafe_offset=0] == 91:
        return not _bytes_lt(bp + 1, bl - 1, np, nl)
    return _bytes_lt(np, nl, bp + 1, bl - 1)


def _valid_bound(bp: Pointer[UInt8, MutUntrackedOrigin], bl: Int) -> Bool:
    if bl == 1 and (bp[unsafe_offset=0] == 45 or bp[unsafe_offset=0] == 43): return True
    return bl >= 1 and (bp[unsafe_offset=0] == 91 or bp[unsafe_offset=0] == 40)


def handle_vrange(
    tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
    keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], mut writer: ResponseWriter,
) -> Int:
    """VRANGE key start end [count] — elements in lexicographic order between
    `start` and `end` (`-`, `+`, `[inclusive`, `(exclusive`), as in Redis 8."""
    if i + 3 >= num_tokens:
        writer.append_error_response("ERR wrong number of arguments for 'vrange' command")
        return 0
    var wrongtype = False
    var vs = _lookup(keyspace, GenericValue.borrow(tokens[unsafe_offset=i + 1].ptr, tokens[unsafe_offset=i + 1].length), wrongtype)
    if wrongtype:
        writer.append_error_response(WRONGTYPE_MSG)
        return 0
    var lp = tokens[unsafe_offset=i + 2].ptr; var ll = tokens[unsafe_offset=i + 2].length
    var hp = tokens[unsafe_offset=i + 3].ptr; var hl = tokens[unsafe_offset=i + 3].length
    if not (_valid_bound(lp, ll) and _valid_bound(hp, hl)):
        writer.append_error_response("ERR range bounds must be -, +, [element or (element")
        return 0
    var count = -1
    if i + 4 < num_tokens:
        var pc = parse_int64_strict(tokens[unsafe_offset=i + 4].ptr, tokens[unsafe_offset=i + 4].length)
        if not pc.ok:
            writer.append_error_response("ERR value is not an integer or out of range")
            return 0
        count = Int(pc.value)
    if is_null(vs):
        writer.append_empty_array_response()
        return 0
    var picked = List[Int]()
    for s in range(vs[].n):
        if vs[].alive[unsafe_offset=s] != 0 and _in_lo(vs[].names[s], lp, ll) and _in_hi(vs[].names[s], hp, hl):
            picked.append(s)
    for a in range(1, len(picked)):
        var b = a
        while b > 0 and _name_lt(vs[].names[picked[b]], vs[].names[picked[b - 1]]):
            var t = picked[b]; picked[b] = picked[b - 1]; picked[b - 1] = t
            b -= 1
    var n = len(picked)
    if count >= 0 and count < n:
        n = count
    writer.append_array_header(n)
    for r in range(n):
        _write_str(writer, vs[].names[picked[r]])
    return 0
