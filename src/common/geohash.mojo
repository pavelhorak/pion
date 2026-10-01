from std.math import ldexp

# Geohash parameters
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
    var lat_range = GeoHashRange(GEO_LAT_MIN, GEO_LAT_MAX)
    var long_range = GeoHashRange(GEO_LONG_MIN, GEO_LONG_MAX)
    var hash = GeoHashBits(step, 0)
    var is_even = True

    for _ in range(step * 2):
        if is_even:
            var mid = (long_range.min + long_range.max) / 2
            if longitude > mid:
                hash.bits = (hash.bits << 1) | 1
                long_range.min = mid
            else:
                hash.bits <<= 1
                long_range.max = mid
        else:
            var mid = (lat_range.min + lat_range.max) / 2
            if latitude > mid:
                hash.bits = (hash.bits << 1) | 1
                lat_range.min = mid
            else:
                hash.bits <<= 1
                lat_range.max = mid
        is_even = not is_even
    return hash

def geohash_encode_wgs84(latitude: Float64, longitude: Float64, step: Int) -> GeoHashBits:
    """Standard textual-geohash encode (gh #181): lat range ±90, unlike the
    internal ±85.05 mercator form. Redis re-encodes with this range for the
    GEOHASH reply, so the internal score bits must not be base32'd directly."""
    var lat_range = GeoHashRange(-90.0, 90.0)
    var long_range = GeoHashRange(GEO_LONG_MIN, GEO_LONG_MAX)
    var hash = GeoHashBits(step, 0)
    var is_even = True

    for _ in range(step * 2):
        if is_even:
            var mid = (long_range.min + long_range.max) / 2
            if longitude > mid:
                hash.bits = (hash.bits << 1) | 1
                long_range.min = mid
            else:
                hash.bits <<= 1
                long_range.max = mid
        else:
            var mid = (lat_range.min + lat_range.max) / 2
            if latitude > mid:
                hash.bits = (hash.bits << 1) | 1
                lat_range.min = mid
            else:
                hash.bits <<= 1
                lat_range.max = mid
        is_even = not is_even
    return hash

def geohash_decode(hash: GeoHashBits) -> GeoHashArea:
    var area = GeoHashArea(hash, GeoHashRange(GEO_LAT_MIN, GEO_LAT_MAX), GeoHashRange(GEO_LONG_MIN, GEO_LONG_MAX))
    var is_even = True

    for i in range(hash.step * 2):
        var bit = (hash.bits >> UInt64(hash.step * 2 - i - 1)) & 1
        if is_even:
            var mid = (area.longitude.min + area.longitude.max) / 2
            if bit:
                area.longitude.min = mid
            else:
                area.longitude.max = mid
        else:
            var mid = (area.latitude.min + area.latitude.max) / 2
            if bit:
                area.latitude.min = mid
            else:
                area.latitude.max = mid
        is_even = not is_even
    return area
