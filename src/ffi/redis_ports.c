/*
 * Copyright (c) 2009-2012, Redis Ltd.
 * Copyright (c) 2018, Redis Ltd.
 * Copyright (c) Valkey Contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ----------------------------------------------------------------------------
 *
 * Ports of two Valkey 9.0.3 commands, so that Pion answers them as Redis 7
 * does (#39):
 *   - LCS, from lcsCommand in src/t_string.c;
 *   - LOLWUT versions 5 and 6, from src/lolwut.c, src/lolwut5.c and
 *     src/lolwut6.c.
 * The Mojo side (src/commands/string_kv.mojo, src/commands/admin.mojo) does
 * the key lookups and option parsing; these functions do the work and build
 * the reply. Compiled as part of fcntl_wrap.c, which includes this file.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── LCS ─────────────────────────────────────────────────────────────────
   The dynamic-programming table of LCS lengths (uint32 per cell, refused past
   512 MB, Redis's default proto-max-bulk-len), then the walk back from the
   end that collects the string and, for IDX, the matching ranges, newest
   first. The reply is built here, as RESP, in a malloc'd buffer the caller
   frees with pion_lcs_free. mode: 0 the string, 1 LEN, 2 IDX. Returns the
   reply length, or -1 when a buffer could not be allocated. */

typedef struct { char *p; size_t n, cap; int oom; } LcsBuf;

static void lcs_add(LcsBuf *b, const char *s, size_t n) {
    if (b->oom || n == 0) return;
    if (b->n + n > b->cap) {
        size_t c = b->cap ? b->cap : 256;
        while (c < b->n + n) c *= 2;
        char *q = (char *)realloc(b->p, c);
        if (!q) { b->oom = 1; return; }
        b->p = q;
        b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
}

static void lcs_hdr(LcsBuf *b, char t, long long v) {
    char tmp[32];
    int l = snprintf(tmp, sizeof(tmp), "%c%lld\r\n", t, v);
    lcs_add(b, tmp, (size_t)l);
}

static int64_t lcs_err(LcsBuf *b, const char *msg, char **out) {
    lcs_add(b, "-", 1);
    lcs_add(b, msg, strlen(msg));
    lcs_add(b, "\r\n", 2);
    *out = b->p;
    return b->oom ? -1 : (int64_t)b->n;
}

int64_t pion_lcs(const char *a, int64_t alen64, const char *b, int64_t blen64, int64_t mode,
                 int64_t minmatchlen, int64_t withmatchlen, int64_t resp, char **out) {
    LcsBuf rb = {NULL, 0, 0, 0};
    *out = NULL;
    /* Detect string truncation or later overflows. */
    if (alen64 >= (int64_t)UINT32_MAX - 1 || blen64 >= (int64_t)UINT32_MAX - 1)
        return lcs_err(&rb, "ERR String too long for LCS", out);
    uint32_t alen = (uint32_t)alen64, blen = (uint32_t)blen64;
    unsigned long long lcssize = (unsigned long long)(alen + 1) * (blen + 1);
    unsigned long long lcsalloc = lcssize * sizeof(uint32_t);
    uint32_t *lcs = NULL;
    if (lcsalloc < SIZE_MAX && lcsalloc / lcssize == sizeof(uint32_t)) {
        if (lcsalloc > 512ULL * 1024 * 1024)
            return lcs_err(&rb, "ERR Insufficient memory, transient memory for LCS exceeds proto-max-bulk-len", out);
        lcs = (uint32_t *)malloc((size_t)lcsalloc);
    }
    if (!lcs)
        return lcs_err(&rb, "ERR Insufficient memory, failed allocating transient memory for LCS", out);

#define LCS(A, B) lcs[(B) + ((size_t)(A) * (blen + 1))]
    for (uint32_t i = 0; i <= alen; i++) {
        for (uint32_t j = 0; j <= blen; j++) {
            if (i == 0 || j == 0) {
                LCS(i, j) = 0;
            } else if (a[i - 1] == b[j - 1]) {
                LCS(i, j) = LCS(i - 1, j - 1) + 1;
            } else {
                uint32_t lcs1 = LCS(i - 1, j);
                uint32_t lcs2 = LCS(i, j - 1);
                LCS(i, j) = lcs1 > lcs2 ? lcs1 : lcs2;
            }
        }
    }

    uint32_t idx = LCS(alen, blen);
    int getidx = mode == 2, getlen = mode == 1;
    int computelcs = getidx || !getlen;
    char *result = NULL;
    if (computelcs) {
        result = (char *)malloc(idx ? idx : 1);
        if (!result) { free(lcs); return -1; }
    }
    /* IDX: the ranges are collected first, since their count leads them. */
    LcsBuf mb = {NULL, 0, 0, 0};
    long long arraylen = 0;
    uint32_t arange_start = alen, arange_end = 0, brange_start = 0, brange_end = 0;
    uint32_t i = alen, j = blen;
    while (computelcs && i > 0 && j > 0) {
        int emit_range = 0;
        if (a[i - 1] == b[j - 1]) {
            result[idx - 1] = a[i - 1];
            if (arange_start == alen) {
                arange_start = i - 1;
                arange_end = i - 1;
                brange_start = j - 1;
                brange_end = j - 1;
            } else if (arange_start == i && brange_start == j) {
                arange_start--;
                brange_start--;
            } else {
                emit_range = 1;
            }
            if (arange_start == 0 || brange_start == 0) emit_range = 1;
            idx--;
            i--;
            j--;
        } else {
            uint32_t lcs1 = LCS(i - 1, j);
            uint32_t lcs2 = LCS(i, j - 1);
            if (lcs1 > lcs2)
                i--;
            else
                j--;
            if (arange_start != alen) emit_range = 1;
        }
        uint32_t match_len = arange_end - arange_start + 1;
        if (emit_range) {
            if (minmatchlen == 0 || (long long)match_len >= minmatchlen) {
                if (getidx) {
                    lcs_hdr(&mb, '*', 2 + (withmatchlen ? 1 : 0));
                    lcs_hdr(&mb, '*', 2);
                    lcs_hdr(&mb, ':', arange_start);
                    lcs_hdr(&mb, ':', arange_end);
                    lcs_hdr(&mb, '*', 2);
                    lcs_hdr(&mb, ':', brange_start);
                    lcs_hdr(&mb, ':', brange_end);
                    if (withmatchlen) lcs_hdr(&mb, ':', match_len);
                    arraylen++;
                }
            }
            arange_start = alen; /* Restart at the next match. */
        }
    }

    if (getidx) {
        if (resp == 3) lcs_hdr(&rb, '%', 2);
        else lcs_hdr(&rb, '*', 4);
        lcs_add(&rb, "$7\r\nmatches\r\n", 13);
        lcs_hdr(&rb, '*', arraylen);
        lcs_add(&rb, mb.p, mb.n);
        lcs_add(&rb, "$3\r\nlen\r\n", 9);
        lcs_hdr(&rb, ':', LCS(alen, blen));
    } else if (getlen) {
        lcs_hdr(&rb, ':', idx);
    } else {
        lcs_hdr(&rb, '$', LCS(alen, blen));
        lcs_add(&rb, result, LCS(alen, blen));
        lcs_add(&rb, "\r\n", 2);
    }
#undef LCS
    int oom = rb.oom || mb.oom;
    free(mb.p);
    free(result);
    free(lcs);
    *out = rb.p;
    return oom ? -1 : (int64_t)rb.n;
}

void pion_lcs_free(char *p) { free(p); }

/* ── LOLWUT ──────────────────────────────────────────────────────────────
   The canvas, Schotter (version 5) and the skyline (version 6). `version` is
   5 or 6; `args` holds the version's numeric arguments (cols, squares per
   row, squares per column for 5; cols, rows for 6), `nargs` of them, already
   parsed as integers by the caller. `label` ends the text ("Pion ver. X").
   Returns the text's length and the malloc'd text in *out (pion_lcs_free);
   the caller replies it as a verbatim string. */

typedef struct lwCanvas {
    int width;
    int height;
    char *pixels;
} lwCanvas;

static lwCanvas *lwCreateCanvas(int width, int height, int bgcolor) {
    lwCanvas *canvas = (lwCanvas *)malloc(sizeof(*canvas));
    canvas->width = width;
    canvas->height = height;
    canvas->pixels = (char *)malloc((size_t)width * height + 1);
    memset(canvas->pixels, bgcolor, (size_t)width * height);
    return canvas;
}

static void lwFreeCanvas(lwCanvas *canvas) {
    free(canvas->pixels);
    free(canvas);
}

static void lwDrawPixel(lwCanvas *canvas, int x, int y, int color) {
    if (x < 0 || x >= canvas->width || y < 0 || y >= canvas->height) return;
    canvas->pixels[x + y * canvas->width] = color;
}

static int lwGetPixel(lwCanvas *canvas, int x, int y) {
    if (x < 0 || x >= canvas->width || y < 0 || y >= canvas->height) return 0;
    return canvas->pixels[x + y * canvas->width];
}

static void lwDrawLine(lwCanvas *canvas, int x1, int y1, int x2, int y2, int color) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy, e2;

    while (1) {
        lwDrawPixel(canvas, x1, y1, color);
        if (x1 == x2 && y1 == y2) break;
        e2 = err * 2;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

static void lwDrawSquare(lwCanvas *canvas, int x, int y, float size, float angle, int color) {
    int px[4], py[4];
    size /= 1.4142135623;
    size = round(size);
    float k = M_PI / 4 + angle;
    for (int j = 0; j < 4; j++) {
        px[j] = round(sin(k) * size + x);
        py[j] = round(cos(k) * size + y);
        k += M_PI / 2;
    }
    for (int j = 0; j < 4; j++) lwDrawLine(canvas, px[j], py[j], px[(j + 1) % 4], py[(j + 1) % 4], color);
}

/* Version 5: Schotter, by Georg Nees. */
static void lwTranslatePixelsGroup(int byte, char *output) {
    int code = 0x2800 + byte;
    output[0] = 0xE0 | (code >> 12);
    output[1] = 0x80 | ((code >> 6) & 0x3F);
    output[2] = 0x80 | (code & 0x3F);
}

static lwCanvas *lwDrawSchotter(int console_cols, int squares_per_row, int squares_per_col) {
    int canvas_width = console_cols * 2;
    int padding = canvas_width > 4 ? 2 : 0;
    float square_side = (float)(canvas_width - padding * 2) / squares_per_row;
    int canvas_height = square_side * squares_per_col + padding * 2;
    lwCanvas *canvas = lwCreateCanvas(canvas_width, canvas_height, 0);

    for (int y = 0; y < squares_per_col; y++) {
        for (int x = 0; x < squares_per_row; x++) {
            int sx = x * square_side + square_side / 2 + padding;
            int sy = y * square_side + square_side / 2 + padding;
            float angle = 0;
            if (y > 1) {
                float r1 = (float)rand() / (float)RAND_MAX / squares_per_col * y;
                float r2 = (float)rand() / (float)RAND_MAX / squares_per_col * y;
                float r3 = (float)rand() / (float)RAND_MAX / squares_per_col * y;
                if (rand() % 2) r1 = -r1;
                if (rand() % 2) r2 = -r2;
                if (rand() % 2) r3 = -r3;
                angle = r1;
                sx += r2 * square_side / 3;
                sy += r3 * square_side / 3;
            }
            lwDrawSquare(canvas, sx, sy, square_side, angle, 1);
        }
    }
    return canvas;
}

static void lwRenderBraille(lwCanvas *canvas, LcsBuf *text) {
    for (int y = 0; y < canvas->height; y += 4) {
        for (int x = 0; x < canvas->width; x += 2) {
            int byte = 0;
            if (lwGetPixel(canvas, x, y)) byte |= (1 << 0);
            if (lwGetPixel(canvas, x, y + 1)) byte |= (1 << 1);
            if (lwGetPixel(canvas, x, y + 2)) byte |= (1 << 2);
            if (lwGetPixel(canvas, x + 1, y)) byte |= (1 << 3);
            if (lwGetPixel(canvas, x + 1, y + 1)) byte |= (1 << 4);
            if (lwGetPixel(canvas, x + 1, y + 2)) byte |= (1 << 5);
            if (lwGetPixel(canvas, x, y + 3)) byte |= (1 << 6);
            if (lwGetPixel(canvas, x + 1, y + 3)) byte |= (1 << 7);
            char unicode[3];
            lwTranslatePixelsGroup(byte, unicode);
            lcs_add(text, unicode, 3);
        }
        if (y != canvas->height - 1) lcs_add(text, "\n", 1);
    }
}

/* Version 6: a skyline, after Plaguemon by hikikomori. */
static void lwRenderGrays(lwCanvas *canvas, LcsBuf *text) {
    for (int y = 0; y < canvas->height; y++) {
        for (int x = 0; x < canvas->width; x++) {
            int color = lwGetPixel(canvas, x, y);
            const char *ce;
            switch (color) {
            case 0: ce = "0;30;40m"; break;
            case 1: ce = "0;90;100m"; break;
            case 2: ce = "0;37;47m"; break;
            case 3: ce = "0;97;107m"; break;
            default: ce = "0;30;40m"; break;
            }
            char cell[32];
            int l = snprintf(cell, sizeof(cell), "\033[%s \033[0m", ce);
            lcs_add(text, cell, (size_t)l);
        }
        if (y != canvas->height - 1) lcs_add(text, "\n", 1);
    }
}

struct skyscraper {
    int xoff;
    int width;
    int height;
    int windows;
    int color;
};

static void generateSkyscraper(lwCanvas *canvas, struct skyscraper *si) {
    int starty = canvas->height - 1;
    int endy = starty - si->height + 1;
    for (int y = starty; y >= endy; y--) {
        for (int x = si->xoff; x < si->xoff + si->width; x++) {
            if (y == endy && (x <= si->xoff + 1 || x >= si->xoff + si->width - 2)) continue;
            int color = si->color;
            if (si->windows && x > si->xoff + 1 && x < si->xoff + si->width - 2 && y > endy + 1 && y < starty - 1) {
                int relx = x - (si->xoff + 1);
                int rely = y - (endy + 1);
                if (relx / 2 % 2 && rely % 2) {
                    do {
                        color = 1 + rand() % 2;
                    } while (color == si->color);
                    if (relx % 2) color = lwGetPixel(canvas, x - 1, y);
                }
            }
            lwDrawPixel(canvas, x, y, color);
        }
    }
}

static void generateSkyline(lwCanvas *canvas) {
    struct skyscraper si;
    for (int color = 2; color >= 1; color--) {
        si.color = color;
        for (int offset = -10; offset < canvas->width;) {
            offset += rand() % 8;
            si.xoff = offset;
            si.width = 10 + rand() % 9;
            if (color == 2)
                si.height = canvas->height / 2 + rand() % canvas->height / 2;
            else
                si.height = canvas->height / 2 + rand() % canvas->height / 3;
            si.windows = 0;
            generateSkyscraper(canvas, &si);
            if (color == 2)
                offset += si.width / 2;
            else
                offset += si.width + 1;
        }
    }
    si.color = 0;
    for (int offset = -10; offset < canvas->width;) {
        offset += rand() % 8;
        si.xoff = offset;
        si.width = 5 + rand() % 14;
        if (si.width % 4) si.width += (si.width % 3);
        si.height = canvas->height / 3 + rand() % canvas->height / 2;
        si.windows = 1;
        generateSkyscraper(canvas, &si);
        offset += si.width + 5;
    }
}

int64_t pion_lolwut(int64_t version, int64_t nargs, const int64_t *args, const char *label,
                    int64_t label_len, char **out) {
    LcsBuf text = {NULL, 0, 0, 0};
    *out = NULL;
    if (version == 5) {
        long cols = nargs > 0 ? (long)args[0] : 66;
        long squares_per_row = nargs > 1 ? (long)args[1] : 8;
        long squares_per_col = nargs > 2 ? (long)args[2] : 12;
        if (cols < 1) cols = 1;
        if (cols > 1000) cols = 1000;
        if (squares_per_row < 1) squares_per_row = 1;
        if (squares_per_row > 200) squares_per_row = 200;
        if (squares_per_col < 1) squares_per_col = 1;
        if (squares_per_col > 200) squares_per_col = 200;
        lwCanvas *canvas = lwDrawSchotter((int)cols, (int)squares_per_row, (int)squares_per_col);
        lwRenderBraille(canvas, &text);
        lwFreeCanvas(canvas);
        const char *credit = "\nGeorg Nees - schotter, plotter on paper, 1968. ";
        lcs_add(&text, credit, strlen(credit));
    } else if (version == 6) {
        long cols = nargs > 0 ? (long)args[0] : 80;
        long rows = nargs > 1 ? (long)args[1] : 20;
        if (cols < 1) cols = 1;
        if (cols > 1000) cols = 1000;
        if (rows < 1) rows = 1;
        if (rows > 1000) rows = 1000;
        lwCanvas *canvas = lwCreateCanvas((int)cols, (int)rows, 3);
        generateSkyline(canvas);
        lwRenderGrays(canvas, &text);
        lwFreeCanvas(canvas);
        const char *credit = "\nDedicated to the 8 bit game developers of past and present.\n"
                             "Original 8 bit image from Plaguemon by hikikomori. ";
        lcs_add(&text, credit, strlen(credit));
    }
    lcs_add(&text, label, (size_t)label_len);
    lcs_add(&text, "\n", 1);
    *out = text.p;
    return text.oom ? -1 : (int64_t)text.n;
}
