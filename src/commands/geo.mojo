"""Geospatial commands: GEOPOS, GEODIST, GEOHASH, GEORADIUS, GEORADIUSBYMEMBER, GEOSEARCH, GEOSEARCHSTORE."""
from src.common.ptr import is_not_null
from src.common.utils import arg_eq, strict_atol, format_float64_to_buf
from std.memory import alloc, stack_allocation
from std.memory.unsafe_pointer import Pointer
from std.collections import Array
from src.network.resp3 import RESP3Token, MAX_CMD_TOKENS
from src.network.response_writer import ResponseWriter
from src.common.hash_map import SlabHashMap, StripedHashMap
from src.common.value import GenericValue, ValueType
from src.common.skip_list import SlabSkipList
from src.common.geohash import geohash_encode, geohash_encode_wgs84, geohash_decode, GeoHashBits, GEO_STEP_MAX
from src.memory.object_pool import ObjectPool
from std.math import sin, cos, sqrt, asin, pi



@always_inline
def _geo_unit(p: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Float64:
    """Redis's extractUnitOrReply: exactly m / km / ft / mi (any case), with
    Redis's constants — or raise its error (the slow path's recovery forwards
    it as the reply). The five copies this replaces matched on the first one
    or two bytes, so "kilograms" was km and anything starting with f was ft,
    and four of them used 1609.344 for a mile where Redis (and GEODIST) use
    1609.34."""
    if arg_eq(p, n, "m"): return 1.0
    if arg_eq(p, n, "km"): return 1000.0
    if arg_eq(p, n, "ft"): return 0.3048
    if arg_eq(p, n, "mi"): return 1609.34
    raise Error("ERR unsupported unit provided. please use M, KM, FT, MI")


def _geo_order(dists: List[Float64], asc: Bool, desc: Bool, count: Int) -> List[Int]:
    """Result order for GEORADIUS*: by distance when ASC/DESC is given, and
    ascending when COUNT is given without an order (Redis sorts to choose the
    COUNT nearest). The handlers parsed ASC/DESC and never applied them."""
    var idx = List[Int]()
    for k in range(len(dists)):
        idx.append(k)
    if asc or desc or count > 0:
        for a in range(1, len(idx)):         # insertion sort: result sets are small
            var j = a
            while j > 0 and ((dists[idx[j]] < dists[idx[j - 1]]) != desc) \
                    and dists[idx[j]] != dists[idx[j - 1]]:
                var t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t
                j -= 1
    return idx^


def _geo_refuse_unsupported(tp: Pointer[UInt8, MutUntrackedOrigin], tl: Int) raises:
    """Options Redis supports and Pion's GEORADIUS* do not: refuse them. They
    were silently ignored, which answers a differently-shaped reply."""
    if arg_eq(tp, tl, "withhash") or arg_eq(tp, tl, "store") or arg_eq(tp, tl, "storedist") \
       or arg_eq(tp, tl, "any"):
        raise Error("ERR this GEORADIUS option is not supported by Pion")
    raise Error("ERR syntax error")

@always_inline
def handle_geopos(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEOPOS key member [member ...] -> array of [lon, lat] or nil."""
    if i + 2 < num_tokens:
        var _gpk = tokens[unsafe_offset=i+1].value()
        var _gpv = keyspace[].get(_gpk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        # gh #232: GEOADD stores ValueType.GEO, so this guard must accept it —
        # checking only for ZSET made Pion's own GEO commands reject their
        # own keys. The wrong-type sweep could not see it: it has no GEO
        # fixture, so the LEGITIMATE case was never probed. Testing only the
        # refusal direction is how a guard breaks the command it guards.
        if (not _gpv.is_none() and _gpv.type.value != ValueType.ZSET
                and _gpv.type.value != ValueType.GEO):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _gp_nc = num_tokens - i - 2
        var _gp_h = "*" + String(_gp_nc) + "\r\n"
        writer.append_to_response(_gp_h.unsafe_ptr(), _gp_h.byte_length())
        for _gpi in range(_gp_nc):
            var _gp_mem = tokens[unsafe_offset=i+2+_gpi].value()
            var _gp_found = False
            if not _gpv.is_none() and _gpv.type.value == ValueType.GEO:
                var _gpp = _gpv.as_geo().unsafe_bitcast[SlabSkipList]()
                var _gpc = _gpp[].head[].forward[0]
                while is_not_null(_gpc):
                    if _gpc[].obj.__str__() == _gp_mem:
                        var _gh = GeoHashBits(GEO_STEP_MAX, UInt64(_gpc[].score))
                        var _ga = geohash_decode(_gh)
                        var _glat = (_ga.latitude.min + _ga.latitude.max) / 2.0
                        var _glon = (_ga.longitude.min + _ga.longitude.max) / 2.0
                        writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
                        var _glons = String(_glon); writer.append_bulk_string_response(_glons.unsafe_ptr(), _glons.byte_length())
                        var _glats = String(_glat); writer.append_bulk_string_response(_glats.unsafe_ptr(), _glats.byte_length())
                        _gp_found = True; break
                    _gpc = _gpc[].forward[0]
            if not _gp_found: writer.append_null_response()
        return 1 + _gp_nc
    else:
        writer.append_error_response("ERR wrong number of arguments for 'geopos' command")
        return 0


@always_inline
def handle_geodist(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEODIST key member1 member2 [m|km|mi|ft] -> bulk string distance or nil."""
    if i + 3 < num_tokens:
        var _gdk = tokens[unsafe_offset=i+1].value()
        var _gdm1 = tokens[unsafe_offset=i+2].value()
        var _gdm2 = tokens[unsafe_offset=i+3].value()
        var _gd_unit = 1.0  # meters
        if i + 4 < num_tokens:
            var _gd_u = tokens[unsafe_offset=i+4].ptr; var _gd_ul = tokens[unsafe_offset=i+4].length
            # gh #232: DIVIDE by Redis's exact constants instead of multiplying by
            # rounded reciprocals. 0.000621371 and 3.28084 are truncated forms of
            # 1/1609.34 and 1/0.3048, and the error shows up in the 4 decimals
            # GEODIST prints: mi read 103.3179 where Redis says 103.3182.
            _gd_unit = _geo_unit(_gd_u, _gd_ul)
        var _gdv = keyspace[].get(_gdk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        # gh #232: GEOADD stores ValueType.GEO, so this guard must accept it —
        # checking only for ZSET made Pion's own GEO commands reject their
        # own keys. The wrong-type sweep could not see it: it has no GEO
        # fixture, so the LEGITIMATE case was never probed. Testing only the
        # refusal direction is how a guard breaks the command it guards.
        if (not _gdv.is_none() and _gdv.type.value != ValueType.ZSET
                and _gdv.type.value != ValueType.GEO):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _gd_s1: Float64 = -1.0; var _gd_s2: Float64 = -1.0
        if not _gdv.is_none() and _gdv.type.value == ValueType.GEO:
            var _gdp = _gdv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _gdcur = _gdp[].head[].forward[0]
            while is_not_null(_gdcur):
                var _ms = _gdcur[].obj.__str__()
                # gh #232: this was an `elif`, so when member1 == member2 the
                # second assignment never ran, `_gd_s2` stayed -1 and
                # `GEODIST key m m` answered nil instead of 0.0000. Both tests
                # must run for the same node.
                if _ms == _gdm1: _gd_s1 = _gdcur[].score
                if _ms == _gdm2: _gd_s2 = _gdcur[].score
                if _gd_s1 >= 0.0 and _gd_s2 >= 0.0: break
                _gdcur = _gdcur[].forward[0]
        if _gd_s1 < 0.0 or _gd_s2 < 0.0:
            writer.append_null_response()
        else:
            var _ga1 = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_gd_s1)))
            var _ga2 = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_gd_s2)))
            var _lat1 = (_ga1.latitude.min + _ga1.latitude.max) / 2.0 * pi / 180.0
            var _lon1 = (_ga1.longitude.min + _ga1.longitude.max) / 2.0 * pi / 180.0
            var _lat2 = (_ga2.latitude.min + _ga2.latitude.max) / 2.0 * pi / 180.0
            var _lon2 = (_ga2.longitude.min + _ga2.longitude.max) / 2.0 * pi / 180.0
            var _dlat = _lat2 - _lat1; var _dlon = _lon2 - _lon1
            var _a = sin(_dlat/2.0)*sin(_dlat/2.0) + cos(_lat1)*cos(_lat2)*sin(_dlon/2.0)*sin(_dlon/2.0)
            var _dist = 2.0 * 6372797.560856 * asin(sqrt(_a)) / _gd_unit
            # gh #232: Redis prints GEODIST with `%.4f` — "166.2742", and a
            # zero distance is "0.0000", so the trailing zeros are part of
            # the contract and must NOT be trimmed. Pion emitted full
            # double precision ("166.2741515696002"), which no Redis client
            # expects to parse.
            var _dbuf = stack_allocation[48, UInt8]()
            var _dlen = format_float64_to_buf(_dbuf, 0, _dist, 4, False)
            writer.append_bulk_string_response(_dbuf, _dlen)
        return 3 + (1 if i+4 < num_tokens else 0)
    else:
        writer.append_error_response("ERR wrong number of arguments for 'geodist' command")
        return 0


@always_inline
def handle_geohash(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEOHASH key member [member ...] -> array of geohash strings or nil."""
    if i + 2 < num_tokens:
        var _ghk = tokens[unsafe_offset=i+1].value()
        var _ghv = keyspace[].get(_ghk)
        # gh #232: wrong type answered like an empty container. Every
        # call site ignores this return and sets i = cmd_end_tok - 1
        # itself, so returning 0 here consumes the frame correctly.
        # gh #232: GEOADD stores ValueType.GEO, so this guard must accept it —
        # checking only for ZSET made Pion's own GEO commands reject their
        # own keys. The wrong-type sweep could not see it: it has no GEO
        # fixture, so the LEGITIMATE case was never probed. Testing only the
        # refusal direction is how a guard breaks the command it guards.
        if (not _ghv.is_none() and _ghv.type.value != ValueType.ZSET
                and _ghv.type.value != ValueType.GEO):
            writer.append_error_response("WRONGTYPE Operation against a key holding the wrong kind of value")
            return 0
        var _gh_nc = num_tokens - i - 2
        var _gh_h = "*" + String(_gh_nc) + "\r\n"
        writer.append_to_response(_gh_h.unsafe_ptr(), _gh_h.byte_length())
        comptime GH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        for _ghi in range(_gh_nc):
            var _gh_mem = tokens[unsafe_offset=i+2+_ghi].value()
            var _gh_found = False
            if not _ghv.is_none() and _ghv.type.value == ValueType.GEO:
                var _ghp = _ghv.as_geo().unsafe_bitcast[SlabSkipList]()
                var _ghc = _ghp[].head[].forward[0]
                while is_not_null(_ghc):
                    if _ghc[].obj.__str__() == _gh_mem:
                        # gh #181: the textual form re-encodes the stored point
                        # with the standard ±90 lat range (Redis behaviour);
                        # base32 of the internal ±85.05 bits gives a wrong hash.
                        var _gha = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_ghc[].score)))
                        var _ghstd = geohash_encode_wgs84(
                            (_gha.latitude.min + _gha.latitude.max) / 2.0,
                            (_gha.longitude.min + _gha.longitude.max) / 2.0, GEO_STEP_MAX)
                        var _ghbits = _ghstd.bits << 3  # 52->55 bits for 11 chars
                        var _ghs = String("")
                        for _gi in range(11):
                            # gh #232: the 11th character is ALWAYS '0'. There are
                            # only 52 bits of hash but 11 base32 chars encode 55,
                            # so the final 3 bits do not exist — Redis hardcodes
                            # index 0 there. Deriving it from `bits << 3` instead
                            # fed the low 2 bits of the score into it, so Palermo
                            # hashed as sqc8b49rny*s* where Redis says sqc8b49rny*0*.
                            # The first ten characters were already correct, which
                            # is why it looked like a rounding artefact.
                            var _gidx = 0
                            if _gi < 10:
                                _gidx = Int((_ghbits >> UInt64(5 * (10 - _gi))) & 0x1F)
                            _ghs += chr(Int(GH_B32.unsafe_ptr()[unsafe_offset=_gidx]))
                        writer.append_bulk_string_response(_ghs.unsafe_ptr(), _ghs.byte_length())
                        _gh_found = True; break
                    _ghc = _ghc[].forward[0]
            if not _gh_found: writer.append_null_response()
        return 1 + _gh_nc
    else:
        writer.append_error_response("ERR wrong number of arguments for 'geohash' command")
        return 0


@always_inline
def handle_georadius(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEORADIUS key longitude latitude radius unit [WITHCOORD] [WITHDIST] [COUNT count] [ASC|DESC]."""
    if i + 5 < num_tokens:
        var _grk = tokens[unsafe_offset=i+1].value()
        var _gr_lon0 = atof(tokens[unsafe_offset=i+2].value()) * pi / 180.0
        var _gr_lat0 = atof(tokens[unsafe_offset=i+3].value()) * pi / 180.0
        var _gr_rad = atof(tokens[unsafe_offset=i+4].value())
        var _gr_u = tokens[unsafe_offset=i+5].ptr; var _gr_ul = tokens[unsafe_offset=i+5].length
        var _gr_unit = 1.0
        _gr_unit = _geo_unit(_gr_u, _gr_ul)
        var _gr_rad_m = _gr_rad * _gr_unit
        var _gr_withcoord = False; var _gr_withdist = False; var _gr_asc = False; var _gr_desc = False; var _gr_cnt = -1
        var _gr_ji = i + 6
        while _gr_ji < num_tokens:
            var _oa = tokens[unsafe_offset=_gr_ji].ptr; var _ol = tokens[unsafe_offset=_gr_ji].length
            if arg_eq(_oa, _ol, "withcoord"): _gr_withcoord = True
            elif arg_eq(_oa, _ol, "withdist"): _gr_withdist = True
            elif arg_eq(_oa, _ol, "asc"): _gr_asc = True; _gr_desc = False
            elif arg_eq(_oa, _ol, "desc"): _gr_desc = True; _gr_asc = False
            elif arg_eq(_oa, _ol, "count") and _gr_ji + 1 < num_tokens:
                _gr_ji += 1
                _gr_cnt = strict_atol(tokens[unsafe_offset=_gr_ji].value())
                if _gr_cnt <= 0: raise Error("ERR COUNT must be > 0")
            else:
                _geo_refuse_unsupported(_oa, _ol)
            _gr_ji += 1
        var _grv = keyspace[].get(_grk)
        var _gr_res = List[String](); var _gr_dists = List[Float64](); var _gr_lats = List[Float64](); var _gr_lons = List[Float64]()
        if not _grv.is_none() and _grv.type.value == ValueType.GEO:
            var _grp = _grv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _grc = _grp[].head[].forward[0]
            while is_not_null(_grc):
                var _ga = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_grc[].score)))
                var _mlat = (_ga.latitude.min + _ga.latitude.max) / 2.0
                var _mlon = (_ga.longitude.min + _ga.longitude.max) / 2.0
                var _mlat_r = _mlat * pi / 180.0; var _mlon_r = _mlon * pi / 180.0
                var _dlat = _mlat_r - _gr_lat0; var _dlon = _mlon_r - _gr_lon0
                var _aa = sin(_dlat/2.0)*sin(_dlat/2.0) + cos(_gr_lat0)*cos(_mlat_r)*sin(_dlon/2.0)*sin(_dlon/2.0)
                var _d = 2.0 * 6372797.560856 * asin(sqrt(_aa))
                if _d <= _gr_rad_m:
                    _gr_res.append(_grc[].obj.__str__()); _gr_dists.append(_d); _gr_lats.append(_mlat); _gr_lons.append(_mlon)
                _grc = _grc[].forward[0]
        var _gr_ord = _geo_order(_gr_dists, _gr_asc, _gr_desc, _gr_cnt)
        var _grn = len(_gr_res)
        if _gr_cnt > 0 and _gr_cnt < _grn: _grn = _gr_cnt
        var _gr_oh = "*" + String(_grn) + "\r\n"
        writer.append_to_response(_gr_oh.unsafe_ptr(), _gr_oh.byte_length())
        for _gro in range(_grn):
            var _gri = _gr_ord[_gro]
            if _gr_withdist or _gr_withcoord:
                var _gre_n = 1 + (1 if _gr_withdist else 0) + (1 if _gr_withcoord else 0)
                var _gre_h = "*" + String(_gre_n) + "\r\n"
                writer.append_to_response(_gre_h.unsafe_ptr(), _gre_h.byte_length())
            writer.append_bulk_string_response(_gr_res[_gri].unsafe_ptr(), _gr_res[_gri].byte_length())
            if _gr_withdist:
                var _ddb = alloc[UInt8](400)   # Redis prints distances with %.4f
                var _ddl = format_float64_to_buf(_ddb, 0, _gr_dists[_gri] / _gr_unit, 4, False)
                writer.append_bulk_string_response(_ddb, _ddl)
                _ddb.free()
            if _gr_withcoord:
                writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
                var _clos = String(_gr_lons[_gri]); writer.append_bulk_string_response(_clos.unsafe_ptr(), _clos.byte_length())
                var _clas = String(_gr_lats[_gri]); writer.append_bulk_string_response(_clas.unsafe_ptr(), _clas.byte_length())
        return _gr_ji - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'georadius' command")
        return 0


@always_inline
def handle_geosearch(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEOSEARCH key FROMMEMBER member|FROMLONLAT lon lat BYRADIUS radius unit|BYBOX w h unit [ASC|DESC] [COUNT count] [WITHCOORD] [WITHDIST]."""
    if i + 4 < num_tokens:
        var _gsk = tokens[unsafe_offset=i+1].value()
        var _gsv = keyspace[].get(_gsk)
        var _gs_clon: Float64 = 0.0; var _gs_clat: Float64 = 0.0
        var _gs_rad_m: Float64 = 0.0
        var _gs_ji = i + 2
        # Parse FROMMEMBER/FROMLONLAT
        if _gs_ji < num_tokens:
            var _fop = tokens[unsafe_offset=_gs_ji].ptr; var _fol = tokens[unsafe_offset=_gs_ji].length
            # gh #181: byte 3 is 'm' in BOTH FROMMEMBER and FROMLONLAT — the
            # old test routed every FROMLONLAT into the member branch and the
            # misaligned radius parse raised out of the handler. Per the
            # gh #162 lesson, match the WHOLE keyword and reject unknowns.
            var _is_fm = _fol == 10 and (_fop[unsafe_offset=0]|0x20)==102 and (_fop[unsafe_offset=1]|0x20)==114 and (_fop[unsafe_offset=2]|0x20)==111 and (_fop[unsafe_offset=3]|0x20)==109 and (_fop[unsafe_offset=4]|0x20)==109 and (_fop[unsafe_offset=5]|0x20)==101 and (_fop[unsafe_offset=6]|0x20)==109 and (_fop[unsafe_offset=7]|0x20)==98 and (_fop[unsafe_offset=8]|0x20)==101 and (_fop[unsafe_offset=9]|0x20)==114
            var _is_fl = _fol == 10 and (_fop[unsafe_offset=0]|0x20)==102 and (_fop[unsafe_offset=1]|0x20)==114 and (_fop[unsafe_offset=2]|0x20)==111 and (_fop[unsafe_offset=3]|0x20)==109 and (_fop[unsafe_offset=4]|0x20)==108 and (_fop[unsafe_offset=5]|0x20)==111 and (_fop[unsafe_offset=6]|0x20)==110 and (_fop[unsafe_offset=7]|0x20)==108 and (_fop[unsafe_offset=8]|0x20)==97 and (_fop[unsafe_offset=9]|0x20)==116
            if not _is_fm and not _is_fl:
                writer.append_error_response("ERR syntax error")
                return 1
            if _is_fm:  # FROMMEMBER
                _gs_ji += 1
                if _gs_ji < num_tokens:
                    var _fm = tokens[unsafe_offset=_gs_ji].value(); _gs_ji += 1
                    if not _gsv.is_none() and _gsv.type.value == ValueType.GEO:
                        var _fgp = _gsv.as_geo().unsafe_bitcast[SlabSkipList]()
                        var _fgc = _fgp[].head[].forward[0]
                        while is_not_null(_fgc):
                            if _fgc[].obj.__str__() == _fm:
                                var _fga = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_fgc[].score)))
                                _gs_clat = (_fga.latitude.min + _fga.latitude.max) / 2.0
                                _gs_clon = (_fga.longitude.min + _fga.longitude.max) / 2.0
                                break
                            _fgc = _fgc[].forward[0]
            else:  # FROMLONLAT
                _gs_ji += 1
                if _gs_ji + 1 < num_tokens:
                    _gs_clon = atof(tokens[unsafe_offset=_gs_ji].value()); _gs_ji += 1
                    _gs_clat = atof(tokens[unsafe_offset=_gs_ji].value()); _gs_ji += 1
        # Parse BYRADIUS/BYBOX
        var _gs_unit = 1.0
        if _gs_ji < num_tokens:
            var _bop = tokens[unsafe_offset=_gs_ji].ptr; var _bol = tokens[unsafe_offset=_gs_ji].length
            _gs_ji += 1
            if _gs_ji < num_tokens: _gs_rad_m = atof(tokens[unsafe_offset=_gs_ji].value()); _gs_ji += 1
            if _gs_ji < num_tokens:
                if _bol >= 8 and (_bop[unsafe_offset=2]|0x20)==120:  # BYBOX - also consume height
                    if _gs_ji < num_tokens: _gs_ji += 1  # skip height
                var _gu = tokens[unsafe_offset=_gs_ji].ptr; var _gul = tokens[unsafe_offset=_gs_ji].length; _gs_ji += 1
                _gs_unit = _geo_unit(_gu, _gul)
                _gs_rad_m *= _gs_unit
        var _gs_withcoord = False; var _gs_withdist = False; var _gs_cnt = -1
        # gh #232: ASC/DESC were not parsed AT ALL, so GEOSEARCH returned
        # skip-list order and `ASC` was a no-op — with Palermo ahead of Catania
        # for a point next to Catania. COUNT was worse: it truncated the
        # UNSORTED list, so `COUNT 1` returned an arbitrary member rather than
        # the nearest one, which is the entire purpose of the option.
        var _gs_asc = False; var _gs_desc = False
        while _gs_ji < num_tokens:
            var _oa = tokens[unsafe_offset=_gs_ji].ptr; var _ol = tokens[unsafe_offset=_gs_ji].length; _gs_ji += 1
            if _ol == 9 and (_oa[unsafe_offset=0]|0x20)==119: _gs_withcoord = True
            elif _ol == 8 and (_oa[unsafe_offset=0]|0x20)==119: _gs_withdist = True
            elif _ol == 3 and (_oa[unsafe_offset=0]|0x20)==97: _gs_asc = True
            elif _ol == 4 and (_oa[unsafe_offset=0]|0x20)==100 and (_oa[unsafe_offset=1]|0x20)==101: _gs_desc = True
            elif _ol == 5 and (_oa[unsafe_offset=0]|0x20)==99:
                if _gs_ji < num_tokens: _gs_cnt = strict_atol(tokens[unsafe_offset=_gs_ji].value()); _gs_ji += 1
        var _gs_clat_r = _gs_clat * pi / 180.0; var _gs_clon_r = _gs_clon * pi / 180.0
        var _gs_res = List[String](); var _gs_dists = List[Float64](); var _gs_lats = List[Float64](); var _gs_lons = List[Float64]()
        if not _gsv.is_none() and _gsv.type.value == ValueType.GEO:
            var _gsp = _gsv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _gsc = _gsp[].head[].forward[0]
            while is_not_null(_gsc):
                var _mga = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_gsc[].score)))
                var _mglat = (_mga.latitude.min + _mga.latitude.max) / 2.0
                var _mglon = (_mga.longitude.min + _mga.longitude.max) / 2.0
                var _mglat_r = _mglat * pi / 180.0; var _mglon_r = _mglon * pi / 180.0
                var _dlat = _mglat_r - _gs_clat_r; var _dlon = _mglon_r - _gs_clon_r
                var _aa = sin(_dlat/2.0)*sin(_dlat/2.0) + cos(_gs_clat_r)*cos(_mglat_r)*sin(_dlon/2.0)*sin(_dlon/2.0)
                var _d = 2.0 * 6372797.560856 * asin(sqrt(_aa))
                if _gs_rad_m <= 0.0 or _d <= _gs_rad_m:
                    _gs_res.append(_gsc[].obj.__str__()); _gs_dists.append(_d); _gs_lats.append(_mglat); _gs_lons.append(_mglon)
                _gsc = _gsc[].forward[0]
        # Emit through an index permutation: the four result lists run in
        # parallel and List is not ImplicitlyCopyable, so sorting Ints and
        # indirecting is cheaper and safer than permuting all four.
        var _ord = List[Int]()
        for _oi in range(len(_gs_res)): _ord.append(_oi)
        # Redis sorts for ASC/DESC, and ALSO whenever COUNT is given — COUNT
        # means "the N nearest", not "any N".
        if _gs_asc or _gs_desc or _gs_cnt > 0:
            for _si in range(1, len(_ord)):
                var _cur = _ord[_si]
                var _sj = _si - 1
                while _sj >= 0:
                    var _worse = (_gs_dists[_ord[_sj]] > _gs_dists[_cur]) if not _gs_desc \
                                 else (_gs_dists[_ord[_sj]] < _gs_dists[_cur])
                    if not _worse: break
                    _ord[_sj + 1] = _ord[_sj]; _sj -= 1
                _ord[_sj + 1] = _cur
        var _gsn = len(_gs_res)
        if _gs_cnt > 0 and _gs_cnt < _gsn: _gsn = _gs_cnt
        var _gs_oh = "*" + String(_gsn) + "\r\n"
        writer.append_to_response(_gs_oh.unsafe_ptr(), _gs_oh.byte_length())
        for _gsk2 in range(_gsn):
            var _gsi = _ord[_gsk2]
            if _gs_withdist or _gs_withcoord:
                var _gse_n = 1 + (1 if _gs_withdist else 0) + (1 if _gs_withcoord else 0)
                var _gse_h = "*" + String(_gse_n) + "\r\n"
                writer.append_to_response(_gse_h.unsafe_ptr(), _gse_h.byte_length())
            writer.append_bulk_string_response(_gs_res[_gsi].unsafe_ptr(), _gs_res[_gsi].byte_length())
            if _gs_withdist:
                # gh #232: %.4f, as GEODIST — WITHDIST printed full double
                # precision (56.4412578701582 where Redis says 56.4413).
                var _dd = _gs_dists[_gsi] / _gs_unit
                var _ddbuf = stack_allocation[48, UInt8]()
                var _ddlen = format_float64_to_buf(_ddbuf, 0, _dd, 4, False)
                writer.append_bulk_string_response(_ddbuf, _ddlen)
            if _gs_withcoord:
                writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
                var _clos = String(_gs_lons[_gsi]); writer.append_bulk_string_response(_clos.unsafe_ptr(), _clos.byte_length())
                var _clas = String(_gs_lats[_gsi]); writer.append_bulk_string_response(_clas.unsafe_ptr(), _clas.byte_length())
        return _gs_ji - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'geosearch' command")
        return 0


@always_inline
def handle_geosearchstore(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEOSEARCHSTORE dest source FROMMEMBER|FROMLONLAT ... BYRADIUS|BYBOX ..."""
    if i + 6 < num_tokens:
        var _gss_dst = tokens[unsafe_offset=i+1].value()
        var _gss_src = tokens[unsafe_offset=i+2].value()
        var _gssv = keyspace[].get(_gss_src)
        var _gss_clat: Float64 = 0.0; var _gss_clon: Float64 = 0.0; var _gss_rad_m: Float64 = 0.0
        var _gss_ji = i + 3
        if _gss_ji < num_tokens:
            var _fp = tokens[unsafe_offset=_gss_ji].ptr; var _fl = tokens[unsafe_offset=_gss_ji].length; _gss_ji += 1
            # gh #181: whole-keyword match, reject unknowns — see handle_geosearch.
            var _is_fm2 = _fl == 10 and (_fp[unsafe_offset=0]|0x20)==102 and (_fp[unsafe_offset=1]|0x20)==114 and (_fp[unsafe_offset=2]|0x20)==111 and (_fp[unsafe_offset=3]|0x20)==109 and (_fp[unsafe_offset=4]|0x20)==109 and (_fp[unsafe_offset=5]|0x20)==101 and (_fp[unsafe_offset=6]|0x20)==109 and (_fp[unsafe_offset=7]|0x20)==98 and (_fp[unsafe_offset=8]|0x20)==101 and (_fp[unsafe_offset=9]|0x20)==114
            var _is_fl2 = _fl == 10 and (_fp[unsafe_offset=0]|0x20)==102 and (_fp[unsafe_offset=1]|0x20)==114 and (_fp[unsafe_offset=2]|0x20)==111 and (_fp[unsafe_offset=3]|0x20)==109 and (_fp[unsafe_offset=4]|0x20)==108 and (_fp[unsafe_offset=5]|0x20)==111 and (_fp[unsafe_offset=6]|0x20)==110 and (_fp[unsafe_offset=7]|0x20)==108 and (_fp[unsafe_offset=8]|0x20)==97 and (_fp[unsafe_offset=9]|0x20)==116
            if not _is_fm2 and not _is_fl2:
                writer.append_error_response("ERR syntax error")
                return 1
            if _is_fm2:  # FROMMEMBER
                if _gss_ji < num_tokens:
                    var _fm2 = tokens[unsafe_offset=_gss_ji].value(); _gss_ji += 1
                    if not _gssv.is_none() and _gssv.type.value == ValueType.GEO:
                        var _fgp2 = _gssv.as_geo().unsafe_bitcast[SlabSkipList]()
                        var _fgc2 = _fgp2[].head[].forward[0]
                        while is_not_null(_fgc2):
                            if _fgc2[].obj.__str__() == _fm2:
                                var _fga2 = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_fgc2[].score)))
                                _gss_clat = (_fga2.latitude.min + _fga2.latitude.max) / 2.0
                                _gss_clon = (_fga2.longitude.min + _fga2.longitude.max) / 2.0
                                break
                            _fgc2 = _fgc2[].forward[0]
            else:
                if _gss_ji + 1 < num_tokens:
                    _gss_clon = atof(tokens[unsafe_offset=_gss_ji].value()); _gss_ji += 1
                    _gss_clat = atof(tokens[unsafe_offset=_gss_ji].value()); _gss_ji += 1
        var _gss_unit = 1.0
        if _gss_ji < num_tokens:
            _gss_ji += 1  # skip BYRADIUS/BYBOX keyword
            if _gss_ji < num_tokens: _gss_rad_m = atof(tokens[unsafe_offset=_gss_ji].value()); _gss_ji += 1
            if _gss_ji < num_tokens:
                var _gu2 = tokens[unsafe_offset=_gss_ji].ptr; var _gul2 = tokens[unsafe_offset=_gss_ji].length; _gss_ji += 1
                _gss_unit = _geo_unit(_gu2, _gul2)
                _gss_rad_m *= _gss_unit
        var _gss_clat_r = _gss_clat * pi / 180.0; var _gss_clon_r = _gss_clon * pi / 180.0
        var _gss_cnt = 0
        var _dst_ptr = skip_list_pool[].acquire()
        _dst_ptr.unsafe_write(SlabSkipList(16))
        if not _gssv.is_none() and _gssv.type.value == ValueType.GEO:
            var _gssp = _gssv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _gssc = _gssp[].head[].forward[0]
            while is_not_null(_gssc):
                var _mga2 = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_gssc[].score)))
                var _mglat2 = (_mga2.latitude.min + _mga2.latitude.max) / 2.0
                var _mglon2 = (_mga2.longitude.min + _mga2.longitude.max) / 2.0
                var _mglat2_r = _mglat2 * pi / 180.0; var _mglon2_r = _mglon2 * pi / 180.0
                var _dl = _mglat2_r - _gss_clat_r; var _dlo = _mglon2_r - _gss_clon_r
                var _aaa = sin(_dl/2.0)*sin(_dl/2.0) + cos(_gss_clat_r)*cos(_mglat2_r)*sin(_dlo/2.0)*sin(_dlo/2.0)
                var _dd2 = 2.0 * 6372797.560856 * asin(sqrt(_aaa))
                if _gss_rad_m <= 0.0 or _dd2 <= _gss_rad_m:
                    _dst_ptr[].insert(_gssc[].score, _gssc[].obj); _gss_cnt += 1
                _gssc = _gssc[].forward[0]
        var _gss_new_val = GenericValue(); _gss_new_val.type = ValueType(ValueType.GEO)
        _gss_new_val.set_ptr(_dst_ptr.unsafe_bitcast[NoneType]())
        keyspace[].set(_gss_dst, _gss_new_val)
        writer.append_int_response(Int64(_gss_cnt))
        return _gss_ji - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'geosearchstore' command")
        return 0


@always_inline
def handle_georadiusbymember(tokens: Pointer[RESP3Token, MutUntrackedOrigin], i: Int, num_tokens: Int, mut writer: ResponseWriter, keyspace: Pointer[StripedHashMap, MutUntrackedOrigin], skip_list_pool: Pointer[ObjectPool[SlabSkipList], MutUntrackedOrigin]) raises -> Int:
    """GEORADIUSBYMEMBER key member radius unit [WITHCOORD] [WITHDIST] [COUNT count] [ASC|DESC]."""
    if i + 4 < num_tokens:
        var _grk2 = tokens[unsafe_offset=i+1].value()
        var _grm2 = tokens[unsafe_offset=i+2].value()
        var _grr2 = atof(tokens[unsafe_offset=i+3].value())
        var _gru2 = tokens[unsafe_offset=i+4].ptr; var _grul2 = tokens[unsafe_offset=i+4].length
        var _grunit2 = 1.0
        _grunit2 = _geo_unit(_gru2, _grul2)
        var _grrad2_m = _grr2 * _grunit2
        var _grbv = keyspace[].get(_grk2)
        var _grc_lat: Float64 = 0.0; var _grc_lon: Float64 = 0.0; var _grc_found = False
        if not _grbv.is_none() and _grbv.type.value == ValueType.GEO:
            var _grbp = _grbv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _grbc = _grbp[].head[].forward[0]
            while is_not_null(_grbc):
                if _grbc[].obj.__str__() == _grm2:
                    var _grba = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_grbc[].score)))
                    _grc_lat = (_grba.latitude.min + _grba.latitude.max) / 2.0
                    _grc_lon = (_grba.longitude.min + _grba.longitude.max) / 2.0
                    _grc_found = True; break
                _grbc = _grbc[].forward[0]
        var _grb_cnt = -1; var _grb_ji = i + 5
        var _grb_asc = False; var _grb_desc = False; var _grb_wd = False; var _grb_wc = False
        while _grb_ji < num_tokens:
            var _oa = tokens[unsafe_offset=_grb_ji].ptr; var _ol = tokens[unsafe_offset=_grb_ji].length; _grb_ji += 1
            if arg_eq(_oa, _ol, "count") and _grb_ji < num_tokens:
                _grb_cnt = strict_atol(tokens[unsafe_offset=_grb_ji].value()); _grb_ji += 1
                if _grb_cnt <= 0: raise Error("ERR COUNT must be > 0")
            elif arg_eq(_oa, _ol, "asc"): _grb_asc = True; _grb_desc = False
            elif arg_eq(_oa, _ol, "desc"): _grb_desc = True; _grb_asc = False
            elif arg_eq(_oa, _ol, "withdist"): _grb_wd = True
            elif arg_eq(_oa, _ol, "withcoord"): _grb_wc = True
            else: _geo_refuse_unsupported(_oa, _ol)
        var _grb_res = List[String]()
        var _grb_d = List[Float64](); var _grb_la = List[Float64](); var _grb_lo = List[Float64]()
        if _grc_found and not _grbv.is_none() and _grbv.type.value == ValueType.GEO:
            var _grbp2 = _grbv.as_geo().unsafe_bitcast[SlabSkipList]()
            var _grbc2 = _grbp2[].head[].forward[0]
            var _grc_latr = _grc_lat * pi / 180.0; var _grc_lonr = _grc_lon * pi / 180.0
            while is_not_null(_grbc2):
                var _mga3 = geohash_decode(GeoHashBits(GEO_STEP_MAX, UInt64(_grbc2[].score)))
                var _mgl3 = (_mga3.latitude.min + _mga3.latitude.max) / 2.0 * pi / 180.0
                var _mglo3 = (_mga3.longitude.min + _mga3.longitude.max) / 2.0 * pi / 180.0
                var _dl3 = _mgl3 - _grc_latr; var _dlo3 = _mglo3 - _grc_lonr
                var _aaa3 = sin(_dl3/2.0)*sin(_dl3/2.0) + cos(_grc_latr)*cos(_mgl3)*sin(_dlo3/2.0)*sin(_dlo3/2.0)
                var _d3 = 2.0 * 6372797.560856 * asin(sqrt(_aaa3))
                if _d3 <= _grrad2_m:
                    _grb_res.append(_grbc2[].obj.__str__()); _grb_d.append(_d3)
                    _grb_la.append(_mgl3 * 180.0 / pi); _grb_lo.append(_mglo3 * 180.0 / pi)
                _grbc2 = _grbc2[].forward[0]
        var _grb_n = len(_grb_res)
        if _grb_cnt > 0 and _grb_cnt < _grb_n: _grb_n = _grb_cnt
        var _grb_h = "*" + String(_grb_n) + "\r\n"
        writer.append_to_response(_grb_h.unsafe_ptr(), _grb_h.byte_length())
        var _grb_ord = _geo_order(_grb_d, _grb_asc, _grb_desc, _grb_cnt)
        for _grbo in range(_grb_n):
            var _grbi = _grb_ord[_grbo]
            if _grb_wd or _grb_wc:
                var _gbe_h = "*" + String(1 + (1 if _grb_wd else 0) + (1 if _grb_wc else 0)) + "\r\n"
                writer.append_to_response(_gbe_h.unsafe_ptr(), _gbe_h.byte_length())
            writer.append_bulk_string_response(_grb_res[_grbi].unsafe_ptr(), _grb_res[_grbi].byte_length())
            if _grb_wd:
                var _gbdb = alloc[UInt8](400)
                var _gbdl = format_float64_to_buf(_gbdb, 0, _grb_d[_grbi] / _grunit2, 4, False)
                writer.append_bulk_string_response(_gbdb, _gbdl)
                _gbdb.free()
            if _grb_wc:
                writer.append_to_response("*2\r\n".unsafe_ptr(), 4)
                var _gblo = String(_grb_lo[_grbi]); writer.append_bulk_string_response(_gblo.unsafe_ptr(), _gblo.byte_length())
                var _gbla = String(_grb_la[_grbi]); writer.append_bulk_string_response(_gbla.unsafe_ptr(), _gbla.byte_length())
        return _grb_ji - i - 1
    else:
        writer.append_error_response("ERR wrong number of arguments for 'georadiusbymember' command")
        return 0
