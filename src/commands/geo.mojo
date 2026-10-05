"""Geospatial commands: GEOADD, GEOPOS, GEODIST, GEOHASH, GEORADIUS,
GEORADIUS_RO, GEORADIUSBYMEMBER, GEORADIUSBYMEMBER_RO, GEOSEARCH,
GEOSEARCHSTORE — Redis's geo.c, ported.

A geo key IS a sorted set, as in Redis: the score is the point's 52-bit
geohash (src/common/geohash.mojo), so every Z* command works on a geo key and
the GEO commands work on a sorted set built with ZADD. Pion used to keep geo
keys as their own type, which Z* commands refused; values of that type from an
older WAL or snapshot load as sorted sets.

Each search parses as Redis's georadiusGeneric does — the same options, the
same errors in the same order, nothing ignored — and walks the cell holding the
centre and its eight neighbours as Redis does, so an unsorted reply and ANY
pick the same members. Coordinates print as Redis 8's addReplyDouble prints
them.
"""
from src.common.ptr import is_not_null, null_ptr
from src.common.utils import arg_eq, parse_redis_double, DOUBLE_VALUE, parse_int64_strict, format_float64_to_buf, format_score
from src.common.container_free import remove_and_free
from std.memory import alloc, stack_allocation
from std.memory.unsafe_pointer import Pointer
from std.ffi import external_call
from src.network.resp3 import RESP3Token
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.skip_list import SlabSkipList
from src.common.geohash import (geohash_encode, geohash_encode_wgs84, geohash_decode_score,
                                geohash_search_cells, geohash_align52, geohash_distance,
                                geohash_within, GeoHashBits, GEO_STEP_MAX,
                                GEO_LAT_MIN, GEO_LAT_MAX, GEO_LONG_MIN, GEO_LONG_MAX)
from src.memory.object_pool import ObjectPool
from src.io.wal import WAL


comptime _E_WRONGTYPE = "WRONGTYPE Operation against a key holding the wrong kind of value"


@always_inline
def _is_geo_type(v: GenericValue) -> Bool:
    """A sorted set, or the old separate geo type (WAL/snapshot from before)."""
    return v.type.value == ValueType.ZSET or v.type.value == ValueType.GEO


@always_inline
def _zset_of(v: GenericValue) -> Pointer[SlabSkipList, MutUntrackedOrigin]:
    return v.as_zset().unsafe_bitcast[SlabSkipList]()


def _geo_unit(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Float64:
    """extractUnitOrReply: m / km / ft / mi, any case; -1 for anything else."""
    if arg_eq(p, n, "m"): return 1.0
    if arg_eq(p, n, "km"): return 1000.0
    if arg_eq(p, n, "ft"): return 0.3048
    if arg_eq(p, n, "mi"): return 1609.34
    return -1.0


def _fmt_f6(v: Float64) -> String:
    """C's "%f", for the error message Redis builds with it."""
    var buf = alloc[UInt8](400)
    var n = Int(external_call["pion_fmt_fixed", Int64](v, Int32(6), buf, Int64(400)))
    var s = String("")
    if n > 0:
        for k in range(n):
            s += chr(Int(buf[unsafe_offset=k]))
    buf.unsafe_free()
    return s


def _lonlat(tokens: Pointer[RESP3Token, MutUntrackedOrigin], j: Int, mut writer: ResponseWriter,
            mut lon: Float64, mut lat: Float64) -> Bool:
    """extractLongLatOrReply: two floats (string2d), within the WGS84 range
    Redis indexes. Writes the error on failure."""
    var a = parse_redis_double(tokens[j].ptr, tokens[j].length, DOUBLE_VALUE)
    if not a.ok:
        writer.append_error_response("ERR value is not a valid float")
        return False
    var b = parse_redis_double(tokens[j + 1].ptr, tokens[j + 1].length, DOUBLE_VALUE)
    if not b.ok:
        writer.append_error_response("ERR value is not a valid float")
        return False
    lon = a.value
    lat = b.value
    if lon < GEO_LONG_MIN or lon > GEO_LONG_MAX or lat < GEO_LAT_MIN or lat > GEO_LAT_MAX:
        writer.append_error_response("ERR invalid longitude,latitude pair " + _fmt_f6(lon) + "," + _fmt_f6(lat))
        return False
    return True


def _distance(tokens: Pointer[RESP3Token, MutUntrackedOrigin], j: Int, mut writer: ResponseWriter,
              mut radius: Float64, mut conversion: Float64) -> Bool:
    """extractDistanceOrReply: a non-negative radius and a unit."""
    var r = parse_redis_double(tokens[j].ptr, tokens[j].length, DOUBLE_VALUE)
    if not r.ok:
        writer.append_error_response("ERR need numeric radius")
        return False
    if r.value < 0:
        writer.append_error_response("ERR radius cannot be negative")
        return False
    var u = _geo_unit(tokens[j + 1].ptr, tokens[j + 1].length)
    if u < 0:
        writer.append_error_response("ERR unsupported unit provided. please use M, KM, FT, MI")
        return False
    radius = r.value
    conversion = u
    return True


def _box(tokens: Pointer[RESP3Token, MutUntrackedOrigin], j: Int, mut writer: ResponseWriter,
         mut width: Float64, mut height: Float64, mut conversion: Float64) -> Bool:
    """extractBoxOrReply: non-negative width and height, and a unit."""
    var w = parse_redis_double(tokens[j].ptr, tokens[j].length, DOUBLE_VALUE)
    if not w.ok:
        writer.append_error_response("ERR need numeric width")
        return False
    var h = parse_redis_double(tokens[j + 1].ptr, tokens[j + 1].length, DOUBLE_VALUE)
    if not h.ok:
        writer.append_error_response("ERR need numeric height")
        return False
    if h.value < 0 or w.value < 0:
        writer.append_error_response("ERR height or width cannot be negative")
        return False
    var u = _geo_unit(tokens[j + 2].ptr, tokens[j + 2].length)
    if u < 0:
        writer.append_error_response("ERR unsupported unit provided. please use M, KM, FT, MI")
        return False
    width = w.value
    height = h.value
    conversion = u
    return True


def _member_lonlat(zp: Pointer[SlabSkipList, MutUntrackedOrigin], mp: Pointer[UInt8, MutUntrackedOrigin],
                   ml: Int, mut lon: Float64, mut lat: Float64) -> Bool:
    """longLatFromMember: the member's decoded position, False if absent."""
    var s = zp[].member_score(GenericValue.borrow(mp, ml))
    if s.is_none():
        return False
    geohash_decode_score(s.as_float(), lon, lat)
    return True


def _append_coord(mut writer: ResponseWriter, v: Float64):
    """addReplyDouble, as Redis 8 replies a coordinate: d2string's shortest
    round-trip form (format_score); a RESP3 double, a bulk string under RESP2.
    Pion printed Mojo's own spelling, which differs in exponent form."""
    var t = format_score(v)
    writer.append_double_response(t.unsafe_ptr(), t.byte_length())


def _append_distance(mut writer: ResponseWriter, d: Float64):
    """addReplyDoubleDistance: "%.4f" — trailing zeros are part of it."""
    var buf = stack_allocation[400, UInt8]()
    var n = format_float64_to_buf(buf, 0, d, 4, False)
    writer.append_bulk_string_response(buf, n)


# ── GEOADD / GEOPOS / GEODIST / GEOHASH ──────────────────────────────────────

def handle_geoadd(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                  keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                  skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                  wal: Pointer[WAL, MutUntrackedOrigin]) -> Int:
    """GEOADD key [NX|XX] [CH] longitude latitude member [...] — Redis's
    geoaddCommand: every coordinate is checked before anything is added, then
    it is a ZADD of geohash scores, NX/XX/CH included. It used to read
    coordinates with atof, refuse NX/XX/CH as "not a valid float", and add
    the triples one by one, so a bad later one left the earlier ones added."""
    if num_tokens - i < 5:
        writer.append_error_response("ERR wrong number of arguments for 'geoadd' command")
        return 0
    var nx = False
    var xx = False
    var ch = False
    var j = i + 2
    while j < num_tokens:
        var t = tokens[j]
        if arg_eq(t.ptr, t.length, "nx"): nx = True
        elif arg_eq(t.ptr, t.length, "xx"): xx = True
        elif arg_eq(t.ptr, t.length, "ch"): ch = True
        else: break
        j += 1
    if (num_tokens - j) % 3 != 0 or (xx and nx):
        writer.append_error_response("ERR syntax error")
        return num_tokens - 1 - i
    var n = (num_tokens - j) // 3
    if n == 0:
        writer.append_error_response("ERR wrong number of arguments for 'geoadd' command")
        return num_tokens - 1 - i
    var scores = List[Float64]()
    for k in range(n):
        var lon: Float64 = 0
        var lat: Float64 = 0
        if not _lonlat(tokens, j + k * 3, writer, lon, lat):
            return num_tokens - 1 - i
        scores.append(Float64(geohash_align52(geohash_encode(lat, lon, GEO_STEP_MAX))))
    var kt = tokens[i + 1]
    var key = GenericValue.borrow(kt.ptr, kt.length)
    var v = keyspace[].get(key)
    if not v.is_none() and not _is_geo_type(v):
        writer.append_error_response(_E_WRONGTYPE)
        return num_tokens - 1 - i
    var zp = null_ptr[SlabSkipList, MutUntrackedOrigin]()
    if not v.is_none():
        zp = _zset_of(v)
    var added = 0
    var changed = 0
    for k in range(n):
        var mt = tokens[j + k * 3 + 2]
        var exists = False
        var old: Float64 = 0
        if is_not_null(zp):
            var cur = zp[].member_score(GenericValue.borrow(mt.ptr, mt.length))
            exists = not cur.is_none()
            if exists:
                old = cur.as_float()
        if (nx and exists) or (xx and not exists):
            continue
        if exists and old == scores[k]:
            continue
        if not is_not_null(zp):
            zp = skip_list_pool[].acquire()
            zp.unsafe_write(SlabSkipList(16))
            var nv = GenericValue()
            nv.type = ValueType(ValueType.ZSET)
            nv.set_ptr(zp.unsafe_bitcast[NoneType]())
            keyspace[].set(key, nv)
        _ = zp[].upsert(scores[k], GenericValue.from_ptr(mt.ptr, mt.length))
        if exists:
            changed += 1
        else:
            added += 1
        if is_not_null(wal):
            _ = wal[].append_scored(9, kt.ptr, kt.length, scores[k], mt.ptr, mt.length)
    writer.append_int_response(Int64(added + changed if ch else added))
    return num_tokens - 1 - i


def handle_geopos(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) -> Int:
    """GEOPOS key [member ...] — [longitude, latitude] per member, or a null
    array for one that is not there."""
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'geopos' command")
        return 0
    var v = keyspace[].get(GenericValue.borrow(tokens[i + 1].ptr, tokens[i + 1].length))
    if not v.is_none() and not _is_geo_type(v):
        writer.append_error_response(_E_WRONGTYPE)
        return num_tokens - 1 - i
    writer.append_array_header(num_tokens - i - 2)
    for j in range(i + 2, num_tokens):
        var lon: Float64 = 0
        var lat: Float64 = 0
        if v.is_none() or not _member_lonlat(_zset_of(v), tokens[j].ptr, tokens[j].length, lon, lat):
            writer.append_null_array_response()
            continue
        writer.append_array_header(2)
        _append_coord(writer, lon)
        _append_coord(writer, lat)
    return num_tokens - 1 - i


def handle_geodist(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) -> Int:
    """GEODIST key member1 member2 [M|KM|FT|MI] — the unit first, then the
    key, as Redis; nil if either member is missing."""
    var argc = num_tokens - i
    if argc < 4:
        writer.append_error_response("ERR wrong number of arguments for 'geodist' command")
        return 0
    var to_meter = 1.0
    if argc == 5:
        to_meter = _geo_unit(tokens[i + 4].ptr, tokens[i + 4].length)
        if to_meter < 0:
            writer.append_error_response("ERR unsupported unit provided. please use M, KM, FT, MI")
            return argc - 1
    elif argc > 5:
        writer.append_error_response("ERR syntax error")
        return argc - 1
    var v = keyspace[].get(GenericValue.borrow(tokens[i + 1].ptr, tokens[i + 1].length))
    if v.is_none():
        writer.append_null_response()
        return argc - 1
    if not _is_geo_type(v):
        writer.append_error_response(_E_WRONGTYPE)
        return argc - 1
    var lon1: Float64 = 0
    var lat1: Float64 = 0
    var lon2: Float64 = 0
    var lat2: Float64 = 0
    var zp = _zset_of(v)
    if (not _member_lonlat(zp, tokens[i + 2].ptr, tokens[i + 2].length, lon1, lat1)
            or not _member_lonlat(zp, tokens[i + 3].ptr, tokens[i + 3].length, lon2, lat2)):
        writer.append_null_response()
        return argc - 1
    _append_distance(writer, geohash_distance(lon1, lat1, lon2, lat2) / to_meter)
    return argc - 1


def handle_geohash(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) -> Int:
    """GEOHASH key [member ...] — the standard 11-character geohash per
    member (re-encoded over latitude ±90; the 11th character is always '0',
    52 bits being all there is), or nil."""
    if num_tokens - i < 2:
        writer.append_error_response("ERR wrong number of arguments for 'geohash' command")
        return 0
    var v = keyspace[].get(GenericValue.borrow(tokens[i + 1].ptr, tokens[i + 1].length))
    if not v.is_none() and not _is_geo_type(v):
        writer.append_error_response(_E_WRONGTYPE)
        return num_tokens - 1 - i
    comptime B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    writer.append_array_header(num_tokens - i - 2)
    var buf = stack_allocation[16, UInt8]()
    for j in range(i + 2, num_tokens):
        var lon: Float64 = 0
        var lat: Float64 = 0
        if v.is_none() or not _member_lonlat(_zset_of(v), tokens[j].ptr, tokens[j].length, lon, lat):
            writer.append_null_response()
            continue
        var bits = geohash_encode_wgs84(lat, lon, GEO_STEP_MAX).bits
        for c in range(11):
            var idx = 0
            if c < 10:
                idx = Int((bits >> UInt64(52 - (c + 1) * 5)) & 0x1F)
            buf[unsafe_offset=c] = B32.unsafe_ptr()[unsafe_offset=idx]
        writer.append_bulk_string_response(buf, 11)
    return num_tokens - 1 - i


# ── The search family ────────────────────────────────────────────────────────

comptime GEO_RADIUS = 0             # GEORADIUS[_RO] key lon lat radius unit ...
comptime GEO_RADIUS_MEMBER = 1      # GEORADIUSBYMEMBER[_RO] key member radius unit ...
comptime GEO_SEARCH = 2             # GEOSEARCH key ...
comptime GEO_SEARCHSTORE = 3        # GEOSEARCHSTORE dest key ...


@fieldwise_init
struct GeoPoint(Copyable, Movable, ImplicitlyCopyable):
    var member: GenericValue        # the sorted set's own member, read only
    var score: Float64
    var lon: Float64
    var lat: Float64
    var dist: Float64


def _collect_box(zp: Pointer[SlabSkipList, MutUntrackedOrigin], cell: GeoHashBits,
                 lon: Float64, lat: Float64, circle: Bool, radius_m: Float64,
                 width_m: Float64, height_m: Float64, limit: Int, mut out: List[GeoPoint]):
    """membersOfGeoHashBox: the members whose score falls in the cell's
    range [min, max), kept when inside the shape (geoAppendIfWithinShape)."""
    var lo = Float64(geohash_align52(cell))
    var hi = Float64(geohash_align52(GeoHashBits(cell.step, cell.bits + 1)))
    var x = zp[].head
    var lvl = zp[].level - 1
    while lvl >= 0:
        while is_not_null(x[].forward[lvl]) and x[].forward[lvl][].score < lo:
            x = x[].forward[lvl]
        lvl -= 1
    var node = x[].forward[0]
    while is_not_null(node) and node[].score < hi:
        if limit > 0 and len(out) >= limit:
            return
        var plon: Float64 = 0
        var plat: Float64 = 0
        geohash_decode_score(node[].score, plon, plat)
        var d: Float64 = 0
        var inside = geohash_within(lon, lat, plon, plat, radius_m, -1.0 if circle else width_m, height_m, d)
        if inside:
            out.append(GeoPoint(node[].obj, node[].score, plon, plat, d))
        node = node[].forward[0]


def _geo_search(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int,
                mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                wal: Pointer[WAL, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                kind: Int, readonly: Bool) -> Int:
    """georadiusGeneric."""
    var consumed = num_tokens - 1 - i
    var name = String("georadius")
    var min_args = 6
    if kind == GEO_RADIUS_MEMBER:
        name = String("georadiusbymember"); min_args = 5
    elif kind == GEO_SEARCH:
        name = String("geosearch"); min_args = 7
    elif kind == GEO_SEARCHSTORE:
        name = String("geosearchstore"); min_args = 8
    if readonly:
        name += "_ro"
    if num_tokens - i < min_args:
        writer.append_error_response("ERR wrong number of arguments for '" + name + "' command")
        return 0
    var src = i + 2 if kind == GEO_SEARCHSTORE else i + 1
    var zv = keyspace[].get(GenericValue.borrow(tokens[src].ptr, tokens[src].length))
    if not zv.is_none() and not _is_geo_type(zv):
        writer.append_error_response(_E_WRONGTYPE)
        return consumed
    var has_src = not zv.is_none()
    var zp = _zset_of(zv) if has_src else null_ptr[SlabSkipList, MutUntrackedOrigin]()

    var lon: Float64 = 0
    var lat: Float64 = 0
    var circle = True
    var radius: Float64 = 0
    var width: Float64 = 0
    var height: Float64 = 0
    var conversion: Float64 = 1
    var base = i + 2
    var store = -1                      # token index of the destination key
    var storedist = False
    if kind == GEO_RADIUS:
        base = i + 6
        if not _lonlat(tokens, i + 2, writer, lon, lat):
            return consumed
        if not _distance(tokens, i + 4, writer, radius, conversion):
            return consumed
    elif kind == GEO_RADIUS_MEMBER:
        base = i + 5
        if has_src:
            if not _member_lonlat(zp, tokens[i + 2].ptr, tokens[i + 2].length, lon, lat):
                writer.append_error_response("ERR could not decode requested zset member")
                return consumed
            if not _distance(tokens, i + 3, writer, radius, conversion):
                return consumed
    elif kind == GEO_SEARCHSTORE:
        base = i + 3
        store = i + 1

    var withdist = False
    var withhash = False
    var withcoord = False
    var frommember = False
    var fromloc = False
    var byradius = False
    var bybox = False
    var sort = 0                        # 0 none, 1 asc, 2 desc
    var any = False
    var count = 0
    var search = kind == GEO_SEARCH or kind == GEO_SEARCHSTORE
    var j = base
    while j < num_tokens:
        var t = tokens[j]
        var left = num_tokens - j - 1   # arguments after this one
        if arg_eq(t.ptr, t.length, "withdist"):
            withdist = True
        elif arg_eq(t.ptr, t.length, "withhash"):
            withhash = True
        elif arg_eq(t.ptr, t.length, "withcoord"):
            withcoord = True
        elif arg_eq(t.ptr, t.length, "any"):
            any = True
        elif arg_eq(t.ptr, t.length, "asc"):
            sort = 1
        elif arg_eq(t.ptr, t.length, "desc"):
            sort = 2
        elif arg_eq(t.ptr, t.length, "count") and left >= 1:
            var c = parse_int64_strict(tokens[j + 1].ptr, tokens[j + 1].length)
            if not c.ok:
                writer.append_error_response("ERR value is not an integer or out of range")
                return consumed
            if c.value <= 0:
                writer.append_error_response("ERR COUNT must be > 0")
                return consumed
            count = Int(c.value)
            j += 1
        elif (arg_eq(t.ptr, t.length, "store") or arg_eq(t.ptr, t.length, "storedist")) \
                and left >= 1 and not readonly and not search:
            store = j + 1
            storedist = arg_eq(t.ptr, t.length, "storedist")
            j += 1
        elif arg_eq(t.ptr, t.length, "storedist") and kind == GEO_SEARCHSTORE:
            storedist = True
        elif arg_eq(t.ptr, t.length, "frommember") and left >= 1 and search and not fromloc:
            if has_src and not _member_lonlat(zp, tokens[j + 1].ptr, tokens[j + 1].length, lon, lat):
                writer.append_error_response("ERR could not decode requested zset member")
                return consumed
            frommember = True
            j += 1
        elif arg_eq(t.ptr, t.length, "fromlonlat") and left >= 2 and search and not frommember:
            if not _lonlat(tokens, j + 1, writer, lon, lat):
                return consumed
            fromloc = True
            j += 2
        elif arg_eq(t.ptr, t.length, "byradius") and left >= 2 and search and not bybox:
            if not _distance(tokens, j + 1, writer, radius, conversion):
                return consumed
            circle = True
            byradius = True
            j += 2
        elif arg_eq(t.ptr, t.length, "bybox") and left >= 3 and search and not byradius:
            if not _box(tokens, j + 1, writer, width, height, conversion):
                return consumed
            circle = False
            bybox = True
            j += 3
        else:
            writer.append_error_response("ERR syntax error")
            return consumed
        j += 1

    if store >= 0 and (withdist or withhash or withcoord):
        if kind == GEO_SEARCHSTORE:
            writer.append_error_response("ERR GEOSEARCHSTORE is not compatible with WITHDIST, WITHHASH and WITHCOORD options")
        else:
            writer.append_error_response("ERR STORE option in GEORADIUS is not compatible with WITHDIST, WITHHASH and WITHCOORD options")
        return consumed
    if search and not (frommember or fromloc):
        writer.append_error_response("ERR exactly one of FROMMEMBER or FROMLONLAT can be specified for " + _cmd_text(tokens[i]))
        return consumed
    if search and not (byradius or bybox):
        writer.append_error_response("ERR exactly one of BYRADIUS and BYBOX can be specified for " + _cmd_text(tokens[i]))
        return consumed
    if any and count == 0:
        writer.append_error_response("ERR the ANY argument requires COUNT argument")
        return consumed

    if not has_src:
        if store >= 0:
            _store_drop(keyspace, ttl_map, wal, tokens[store])
            writer.append_int_response(0)
        else:
            writer.append_empty_array_response()
        return consumed

    if count != 0 and sort == 0 and not any:
        sort = 1                        # COUNT means the nearest N

    # membersOfAllNeighbors
    var radius_m = radius * conversion
    var width_m = width * conversion
    var height_m = height * conversion
    var cells = geohash_search_cells(lon, lat, radius_m, -1.0 if circle else width_m, height_m)
    var points = List[GeoPoint]()
    var limit = count if any else 0
    var last = -1
    for c in range(9):
        if cells[c].bits == 0 and cells[c].step == 0:
            continue
        # A huge radius can make neighbours coincide: skip a repeat.
        if last >= 0 and cells[c].bits == cells[last].bits and cells[c].step == cells[last].step:
            continue
        if len(points) > 0 and limit > 0 and len(points) >= limit:
            break
        _collect_box(zp, cells[c], lon, lat, circle, radius_m, width_m, height_m, limit, points)
        last = c

    if len(points) == 0 and store < 0:
        writer.append_empty_array_response()
        return consumed

    var returned = len(points) if count == 0 or len(points) < count else count
    # Order through indices: a stable insertion sort by distance.
    var order = List[Int]()
    for k in range(len(points)):
        order.append(k)
    if sort != 0:
        for a in range(1, len(order)):
            var cur = order[a]
            var b = a - 1
            while b >= 0:
                var pd = points[order[b]].dist
                var cd = points[cur].dist
                if (sort == 1 and pd > cd) or (sort == 2 and pd < cd):
                    order[b + 1] = order[b]
                    b -= 1
                else:
                    break
            order[b + 1] = cur

    if store < 0:
        var opts = (1 if withdist else 0) + (1 if withhash else 0) + (1 if withcoord else 0)
        writer.append_array_header(returned)
        for k in range(returned):
            var p = points[order[k]]
            if opts > 0:
                writer.append_array_header(opts + 1)
            writer.append_bulk_value_response(p.member)
            if withdist:
                _append_distance(writer, p.dist / conversion)
            if withhash:
                writer.append_int_response(Int64(p.score))
            if withcoord:
                writer.append_array_header(2)
                _append_coord(writer, p.lon)
                _append_coord(writer, p.lat)
        return consumed

    # STORE / STOREDIST: the result replaces the destination, its TTL with
    # it; nothing found deletes it.
    var dt = tokens[store]
    if returned == 0:
        _store_drop(keyspace, ttl_map, wal, dt)
        writer.append_int_response(0)
        return consumed
    var dp = skip_list_pool[].acquire()
    dp.unsafe_write(SlabSkipList(16))
    for k in range(returned):
        var p = points[order[k]]
        # The member is COPIED: sharing the source's payload let a DEL of
        # either key free the other's members.
        _ = dp[].upsert(p.dist / conversion if storedist else p.score, p.member.clone())
    var dkey = GenericValue.borrow(dt.ptr, dt.length)
    _ = remove_and_free(keyspace, dkey)         # and its TTL
    var nv = GenericValue()
    nv.type = ValueType(ValueType.ZSET)
    nv.set_ptr(dp.unsafe_bitcast[NoneType]())
    keyspace[].set(dkey, nv)
    if is_not_null(wal):
        wal[].log_key_image(keyspace, ttl_map, dt.ptr, dt.length)
    writer.append_int_response(Int64(returned))
    return consumed


def _store_drop(keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                wal: Pointer[WAL, MutUntrackedOrigin], dt: RESP3Token):
    """An empty search result deletes the STORE destination."""
    if remove_and_free(keyspace, GenericValue.borrow(dt.ptr, dt.length)):
        if is_not_null(wal):
            wal[].log_key_image(keyspace, ttl_map, dt.ptr, dt.length)


def _cmd_text(t: RESP3Token) -> String:
    """The command name as the client spelled it (Redis echoes argv[0])."""
    return t.value()


def handle_georadius(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                     keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                     wal: Pointer[WAL, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                     readonly: Bool = False) -> Int:
    """GEORADIUS[_RO] key longitude latitude radius M|KM|FT|MI [WITHCOORD]
    [WITHDIST] [WITHHASH] [COUNT count [ANY]] [ASC|DESC] [STORE key|STOREDIST key]."""
    return _geo_search(tokens, i, num_tokens, writer, keyspace, skip_list_pool, wal, ttl_map, GEO_RADIUS, readonly)


def handle_georadiusbymember(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                             keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                             skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                             wal: Pointer[WAL, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin],
                             readonly: Bool = False) -> Int:
    """GEORADIUSBYMEMBER[_RO] key member radius M|KM|FT|MI [options as GEORADIUS]."""
    return _geo_search(tokens, i, num_tokens, writer, keyspace, skip_list_pool, wal, ttl_map, GEO_RADIUS_MEMBER, readonly)


def handle_geosearch(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                     keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                     skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                     wal: Pointer[WAL, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
    """GEOSEARCH key FROMMEMBER member|FROMLONLAT lon lat BYRADIUS r unit|BYBOX w h unit
    [ASC|DESC] [COUNT count [ANY]] [WITHCOORD] [WITHDIST] [WITHHASH]."""
    return _geo_search(tokens, i, num_tokens, writer, keyspace, skip_list_pool, wal, ttl_map, GEO_SEARCH, False)


def handle_geosearchstore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter,
                          keyspace: Pointer[StripedHashMap, MutUntrackedOrigin],
                          skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin],
                          wal: Pointer[WAL, MutUntrackedOrigin], ttl_map: Pointer[SlabHashMap, MutUntrackedOrigin]) -> Int:
    """GEOSEARCHSTORE dest key [GEOSEARCH's FROM/BY/ASC|DESC/COUNT [ANY]] [STOREDIST]."""
    return _geo_search(tokens, i, num_tokens, writer, keyspace, skip_list_pool, wal, ttl_map, GEO_SEARCHSTORE, False)
