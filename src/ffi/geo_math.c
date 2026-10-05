/* Geo arithmetic for GEOADD / GEOPOS / GEODIST / the search family.

   Why C: Redis's answers depend on how its C compiler evaluates a handful of
   floating-point expressions. Clang (macOS) and gcc on AArch64 fuse `a + b*c`
   into one fused multiply-add, gcc on baseline x86-64 cannot, and the
   difference shows in the last digit of a coordinate (GEOPOS) and, rarely, in
   whether a point on a radius's edge is inside it. Written here, in the same
   expression shapes, and built by the same compiler family as the platform's
   Redis, the results agree bit for bit; Mojo would make its own choices.

   Conventions follow Redis: a cell is `step` bits per axis, interleaved with
   latitude in the even bits and longitude in the odd ones; scores are the
   52-bit (step 26) cell of the point within longitude ±180 and latitude
   ±85.05112878. Distances are meters on a sphere of radius 6372797.560856. */
#include <math.h>
#include <stdint.h>

#define GEO_LAT_MIN (-85.05112878)
#define GEO_LAT_MAX 85.05112878
#define GEO_LONG_MIN (-180.0)
#define GEO_LONG_MAX 180.0
#define GEO_EARTH_R 6372797.560856
#define GEO_MERCATOR_MAX 20037726.37
#define GEO_D_R (M_PI / 180.0)

static inline double geo_deg_rad(double ang) { return ang * GEO_D_R; }
static inline double geo_rad_deg(double ang) { return ang / GEO_D_R; }

static uint64_t geo_interleave(uint32_t xlo, uint32_t ylo) {
    uint64_t x = xlo, y = ylo;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;  y = (y | (y << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;   y = (y | (y << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;   y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;   y = (y | (y << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;   y = (y | (y << 1)) & 0x5555555555555555ULL;
    return x | (y << 1);
}

static uint64_t geo_deinterleave(uint64_t v) {
    uint64_t x = v, y = v >> 1;
    x = x & 0x5555555555555555ULL;                y = y & 0x5555555555555555ULL;
    x = (x | (x >> 1)) & 0x3333333333333333ULL;   y = (y | (y >> 1)) & 0x3333333333333333ULL;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0FULL;   y = (y | (y >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FFULL;   y = (y | (y >> 4)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFFULL;   y = (y | (y >> 8)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFFULL;  y = (y | (y >> 16)) & 0x00000000FFFFFFFFULL;
    return x | (y << 32);
}

/* The cell of (longitude, latitude) at `step` bits per axis, over the given
   ranges; 0 for a point outside them. */
uint64_t pion_geo_encode(double longitude, double latitude, int step,
                         double long_min, double long_max, double lat_min, double lat_max) {
    if (longitude > GEO_LONG_MAX || longitude < GEO_LONG_MIN ||
        latitude > GEO_LAT_MAX || latitude < GEO_LAT_MIN) return 0;
    if (latitude < lat_min || latitude > lat_max ||
        longitude < long_min || longitude > long_max) return 0;
    double lat_offset = (latitude - lat_min) / (lat_max - lat_min);
    double long_offset = (longitude - long_min) / (long_max - long_min);
    lat_offset *= (1ULL << step);
    long_offset *= (1ULL << step);
    return geo_interleave((uint32_t)lat_offset, (uint32_t)long_offset);
}

/* A cell's bounds over the stored ranges: out = {lat_min, lat_max,
   long_min, long_max}. */
void pion_geo_cell(uint64_t bits, int step, double* out) {
    double lat_scale = GEO_LAT_MAX - GEO_LAT_MIN;
    double long_scale = GEO_LONG_MAX - GEO_LONG_MIN;
    uint64_t sep = geo_deinterleave(bits);
    uint32_t ilato = (uint32_t)sep;
    uint32_t ilono = (uint32_t)(sep >> 32);
    out[0] = GEO_LAT_MIN + (ilato * 1.0 / (1ull << step)) * lat_scale;
    out[1] = GEO_LAT_MIN + ((ilato + 1) * 1.0 / (1ull << step)) * lat_scale;
    out[2] = GEO_LONG_MIN + (ilono * 1.0 / (1ull << step)) * long_scale;
    out[3] = GEO_LONG_MIN + ((ilono + 1) * 1.0 / (1ull << step)) * long_scale;
}

/* A stored score's point: the centre of its 52-bit cell, clamped.
   out = {longitude, latitude}. */
void pion_geo_point(double score, double* out) {
    double c[4];
    pion_geo_cell((uint64_t)score, 26, c);
    out[0] = (c[2] + c[3]) / 2;
    if (out[0] > GEO_LONG_MAX) out[0] = GEO_LONG_MAX;
    if (out[0] < GEO_LONG_MIN) out[0] = GEO_LONG_MIN;
    out[1] = (c[0] + c[1]) / 2;
    if (out[1] > GEO_LAT_MAX) out[1] = GEO_LAT_MAX;
    if (out[1] < GEO_LAT_MIN) out[1] = GEO_LAT_MIN;
}

double pion_geo_lat_distance(double lat1d, double lat2d) {
    return GEO_EARTH_R * fabs(geo_deg_rad(lat2d) - geo_deg_rad(lat1d));
}

/* Haversine great-circle distance. */
double pion_geo_distance(double lon1d, double lat1d, double lon2d, double lat2d) {
    double lat1r, lon1r, lat2r, lon2r, u, v, a;
    lon1r = geo_deg_rad(lon1d);
    lon2r = geo_deg_rad(lon2d);
    v = sin((lon2r - lon1r) / 2);
    if (v == 0.0) return pion_geo_lat_distance(lat1d, lat2d);
    lat1r = geo_deg_rad(lat1d);
    lat2r = geo_deg_rad(lat2d);
    u = sin((lat2r - lat1r) / 2);
    a = u * u + cos(lat1r) * cos(lat2r) * v * v;
    return 2.0 * GEO_EARTH_R * asin(sqrt(a));
}

/* Is the point (x2, y2) inside the shape centred on (x1, y1)? A circle when
   width_m < 0 (radius_m), else a width_m x height_m box. Its distance from
   the centre goes to *dist. */
int pion_geo_within(double x1, double y1, double x2, double y2,
                    double radius_m, double width_m, double height_m, double* dist) {
    if (width_m < 0) {
        *dist = pion_geo_distance(x1, y1, x2, y2);
        return *dist <= radius_m;
    }
    double lat_distance = pion_geo_lat_distance(y2, y1);
    if (lat_distance > height_m / 2) return 0;
    double lon_distance = pion_geo_distance(x2, y2, x1, y2);
    if (lon_distance > width_m / 2) return 0;
    *dist = pion_geo_distance(x1, y1, x2, y2);
    return 1;
}

static void geo_move_x(uint64_t* bits, int step, int d) {
    if (d == 0) return;
    uint64_t x = *bits & 0xaaaaaaaaaaaaaaaaULL;
    uint64_t y = *bits & 0x5555555555555555ULL;
    uint64_t zz = 0x5555555555555555ULL >> (64 - step * 2);
    if (d > 0) {
        x = x + (zz + 1);
    } else {
        x = x | zz;
        x = x - (zz + 1);
    }
    x &= (0xaaaaaaaaaaaaaaaaULL >> (64 - step * 2));
    *bits = x | y;
}

static void geo_move_y(uint64_t* bits, int step, int d) {
    if (d == 0) return;
    uint64_t x = *bits & 0xaaaaaaaaaaaaaaaaULL;
    uint64_t y = *bits & 0x5555555555555555ULL;
    uint64_t zz = 0xaaaaaaaaaaaaaaaaULL >> (64 - step * 2);
    if (d > 0) {
        y = y + (zz + 1);
    } else {
        y = y | zz;
        y = y - (zz + 1);
    }
    y &= (0x5555555555555555ULL >> (64 - step * 2));
    *bits = x | y;
}

static int geo_steps_for(double range_meters, double lat) {
    if (range_meters == 0) return 26;
    int step = 1;
    while (range_meters < GEO_MERCATOR_MAX) {
        range_meters *= 2;
        step++;
    }
    step -= 2;
    if (lat > 66 || lat < -66) {
        step--;
        if (lat > 80 || lat < -80) step--;
    }
    if (step < 1) step = 1;
    if (step > 26) step = 26;
    return step;
}

/* The nine cells a radius (width_m < 0) or box search walks, in Redis's
   order — centre, N, S, E, W, NE, NW, SE, SW — at one step for all. A
   neighbour the search cannot reach is bits 0 / step 0. */
void pion_geo_areas(double longitude, double latitude, double radius_m, double width_m,
                    double height_m, uint64_t* bits9, int* steps9) {
    int circle = width_m < 0;
    double height = circle ? radius_m : height_m / 2;
    double width = circle ? radius_m : width_m / 2;
    const double lat_delta = geo_rad_deg(height / GEO_EARTH_R);
    const double long_delta_top = geo_rad_deg(width / GEO_EARTH_R / cos(geo_deg_rad(latitude + lat_delta)));
    const double long_delta_bottom = geo_rad_deg(width / GEO_EARTH_R / cos(geo_deg_rad(latitude - lat_delta)));
    int southern = latitude < 0 ? 1 : 0;
    double min_lon = southern ? longitude - long_delta_bottom : longitude - long_delta_top;
    double max_lon = southern ? longitude + long_delta_bottom : longitude + long_delta_top;
    double min_lat = latitude - lat_delta;
    double max_lat = latitude + lat_delta;

    double radius_meters = circle ? radius_m :
        sqrt((width_m / 2) * (width_m / 2) + (height_m / 2) * (height_m / 2));
    int steps = geo_steps_for(radius_meters, latitude);

    for (int pass = 0; pass < 2; pass++) {
        uint64_t hash = pion_geo_encode(longitude, latitude, steps,
                                        GEO_LONG_MIN, GEO_LONG_MAX, GEO_LAT_MIN, GEO_LAT_MAX);
        uint64_t n[8];          /* N S E W NE NW SE SW */
        for (int k = 0; k < 8; k++) n[k] = hash;
        geo_move_y(&n[0], steps, 1);
        geo_move_y(&n[1], steps, -1);
        geo_move_x(&n[2], steps, 1);
        geo_move_x(&n[3], steps, -1);
        geo_move_x(&n[4], steps, 1);  geo_move_y(&n[4], steps, 1);
        geo_move_x(&n[5], steps, -1); geo_move_y(&n[5], steps, 1);
        geo_move_x(&n[6], steps, 1);  geo_move_y(&n[6], steps, -1);
        geo_move_x(&n[7], steps, -1); geo_move_y(&n[7], steps, -1);
        double area[4], c[4];
        pion_geo_cell(hash, steps, area);
        if (pass == 0) {
            /* The estimate can leave the box poking out of the neighbours:
               then one step coarser. */
            int decrease = 0;
            pion_geo_cell(n[0], steps, c); if (c[1] < max_lat) decrease = 1;
            pion_geo_cell(n[1], steps, c); if (c[0] > min_lat) decrease = 1;
            pion_geo_cell(n[2], steps, c); if (c[3] < max_lon) decrease = 1;
            pion_geo_cell(n[3], steps, c); if (c[2] > min_lon) decrease = 1;
            if (steps > 1 && decrease) {
                steps--;
                continue;
            }
        }
        int zero[8] = {0};
        if (steps >= 2) {
            if (area[0] < min_lat) { zero[1] = zero[7] = zero[6] = 1; }   /* S SW SE */
            if (area[1] > max_lat) { zero[0] = zero[4] = zero[5] = 1; }   /* N NE NW */
            if (area[2] < min_lon) { zero[3] = zero[7] = zero[5] = 1; }   /* W SW NW */
            if (area[3] > max_lon) { zero[2] = zero[6] = zero[4] = 1; }   /* E SE NE */
        }
        bits9[0] = hash; steps9[0] = steps;
        for (int k = 0; k < 8; k++) {
            bits9[k + 1] = zero[k] ? 0 : n[k];
            steps9[k + 1] = zero[k] ? 0 : steps;
        }
        return;
    }
}
