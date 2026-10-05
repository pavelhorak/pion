"""Geohash arithmetic: thin wrappers over src/ffi/geo_math.c.

A point's sorted-set score is its 52-bit geohash: longitude and latitude are
each scaled into [0, 2^26) and interleaved, longitude in the odd bits. The
floating-point work lives in C on purpose — see geo_math.c: Redis's replies
depend on whether its compiler fused a multiply-add, and the platform's C
compiler makes the same choice for the same expression where Mojo would make
its own.

Encoding is Redis's (scale, truncate, interleave), not repeated bisection: the
two disagree for a point on a cell boundary — bisection's `x > mid` put
longitude 0 in the western half, Redis's floor puts it in the eastern — so
`GEOADD k 0 0 m` stored a different score than Redis.
"""
from std.ffi import external_call
from std.memory import stack_allocation

comptime GEO_STEP_MAX = 26
comptime GEO_LAT_MIN = -85.05112878
comptime GEO_LAT_MAX = 85.05112878
comptime GEO_LONG_MIN = -180.0
comptime GEO_LONG_MAX = 180.0


@fieldwise_init
struct GeoHashBits(Copyable, ImplicitlyCopyable):
    var step: Int
    var bits: UInt64


@fieldwise_init
struct GeoHashRange(Copyable, ImplicitlyCopyable):
    var min: Float64
    var max: Float64


@fieldwise_init
struct GeoHashArea(Copyable, ImplicitlyCopyable):
    var hash: GeoHashBits
    var latitude: GeoHashRange
    var longitude: GeoHashRange


def geohash_encode(latitude: Float64, longitude: Float64, step: Int) -> GeoHashBits:
    """The stored form: WGS84 within Redis's Mercator-safe latitude range."""
    return GeoHashBits(step, external_call["pion_geo_encode", UInt64](
        longitude, latitude, Int32(step), GEO_LONG_MIN, GEO_LONG_MAX, GEO_LAT_MIN, GEO_LAT_MAX))


def geohash_encode_wgs84(latitude: Float64, longitude: Float64, step: Int) -> GeoHashBits:
    """The textual GEOHASH form (gh #181): latitude range ±90, as Redis
    re-encodes for the GEOHASH reply."""
    return GeoHashBits(step, external_call["pion_geo_encode", UInt64](
        longitude, latitude, Int32(step), -180.0, 180.0, -90.0, 90.0))


def geohash_decode(hash: GeoHashBits) -> GeoHashArea:
    """A cell's bounds over the stored ranges."""
    var out = stack_allocation[4, Float64]()
    external_call["pion_geo_cell", NoneType](hash.bits, Int32(hash.step), out)
    return GeoHashArea(hash, GeoHashRange(out[0], out[1]), GeoHashRange(out[2], out[3]))


def geohash_decode_score(score: Float64, mut lon: Float64, mut lat: Float64):
    """A stored score's point: the centre of its cell, clamped."""
    var out = stack_allocation[2, Float64]()
    external_call["pion_geo_point", NoneType](score, out)
    lon = out[0]
    lat = out[1]


@always_inline
def geohash_align52(hash: GeoHashBits) -> UInt64:
    return hash.bits << UInt64(52 - hash.step * 2)


def geohash_distance(lon1d: Float64, lat1d: Float64, lon2d: Float64, lat2d: Float64) -> Float64:
    """Haversine great-circle distance in meters."""
    return external_call["pion_geo_distance", Float64](lon1d, lat1d, lon2d, lat2d)


def geohash_within(x1: Float64, y1: Float64, x2: Float64, y2: Float64, radius_m: Float64,
                   width_m: Float64, height_m: Float64, mut dist: Float64) -> Bool:
    """Is (x2, y2) inside the circle (width_m < 0) or box centred on (x1, y1)?"""
    var d = stack_allocation[1, Float64]()
    d[0] = 0
    var r = external_call["pion_geo_within", Int32](x1, y1, x2, y2, radius_m, width_m, height_m, d)
    dist = d[0]
    return r != 0


def geohash_search_cells(lon: Float64, lat: Float64, radius_m: Float64, width_m: Float64,
                         height_m: Float64) -> List[GeoHashBits]:
    """The nine cells a search walks, in Redis's order (centre, N, S, E, W,
    NE, NW, SE, SW); a ruled-out neighbour is step 0, bits 0."""
    var bits = stack_allocation[9, UInt64]()
    var steps = stack_allocation[9, Int32]()
    external_call["pion_geo_areas", NoneType](lon, lat, radius_m, width_m, height_m, bits, steps)
    var out = List[GeoHashBits]()
    for k in range(9):
        out.append(GeoHashBits(Int(steps[k]), bits[k]))
    return out^
