// gh #191 — standalone harness for the sparse block-score selector.
//
// Measures the scalar form currently in metal_wrap.m against a NEON
// multi-accumulator rewrite, and reports BOTH the timing and the correctness
// surface the issue calls out: max |score| delta, and whether the selected
// top-K block SET changes. Reassociation moves the last ulps, and a near-tied
// pair of blocks can swap — which is the only way this can hurt quality.
//
// clang -O3 -o sel_bench sel_bench.c -framework Accelerate
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <arm_neon.h>

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// ---------------------------------------------------------------- scalar ---
static void mean_scalar(const float *Q, const float *M, const uint8_t *hm,
                        uint32_t H_q, uint32_t D, uint32_t n_blocks,
                        float inv_sqrt_D, float *scores) {
    for (uint32_t b = 0; b < n_blocks; b++) scores[b] = 0.0f;
    for (uint32_t h = 0; h < H_q; h++) {
        const float *Qh = Q + (size_t)h * D;
        const float *Mh = M + (size_t)hm[h] * n_blocks * D;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *Mb = Mh + (size_t)b * D;
            float dot = 0.0f;
            for (uint32_t d = 0; d < D; d++) dot += Qh[d] * Mb[d];
            scores[b] += dot * inv_sqrt_D;
        }
    }
}

static void quest_scalar(const float *Q, const float *N, const float *X,
                         const uint8_t *hm, uint32_t H_q, uint32_t D,
                         uint32_t n_blocks, float inv_sqrt_D, float *scores) {
    for (uint32_t b = 0; b < n_blocks; b++) scores[b] = 0.0f;
    for (uint32_t h = 0; h < H_q; h++) {
        const float *Qh = Q + (size_t)h * D;
        const float *Nh = N + (size_t)hm[h] * n_blocks * D;
        const float *Xh = X + (size_t)hm[h] * n_blocks * D;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *Nb = Nh + (size_t)b * D;
            const float *Xb = Xh + (size_t)b * D;
            float ub = 0.0f;
            for (uint32_t d = 0; d < D; d++) {
                float qd = Qh[d], a = qd * Nb[d], c = qd * Xb[d];
                ub += (a > c) ? a : c;
            }
            scores[b] += ub * inv_sqrt_D;
        }
    }
}

// ------------------------------------------------------------------ NEON ---
// Four independent accumulators: the serial FADD chain is the bottleneck, not
// the multiplies. FMA latency on M-series is ~4 cycles with 1-cycle throughput,
// so one chain leaves ~3/4 of the FP pipeline idle regardless of vector width.
static void mean_neon(const float *Q, const float *M, const uint8_t *hm,
                      uint32_t H_q, uint32_t D, uint32_t n_blocks,
                      float inv_sqrt_D, float *scores) {
    for (uint32_t b = 0; b < n_blocks; b++) scores[b] = 0.0f;
    for (uint32_t h = 0; h < H_q; h++) {
        const float *Qh = Q + (size_t)h * D;
        const float *Mh = M + (size_t)hm[h] * n_blocks * D;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *Mb = Mh + (size_t)b * D;
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
            uint32_t d = 0;
            for (; d + 16 <= D; d += 16) {
                a0 = vfmaq_f32(a0, vld1q_f32(Qh + d),      vld1q_f32(Mb + d));
                a1 = vfmaq_f32(a1, vld1q_f32(Qh + d + 4),  vld1q_f32(Mb + d + 4));
                a2 = vfmaq_f32(a2, vld1q_f32(Qh + d + 8),  vld1q_f32(Mb + d + 8));
                a3 = vfmaq_f32(a3, vld1q_f32(Qh + d + 12), vld1q_f32(Mb + d + 12));
            }
            for (; d + 4 <= D; d += 4)
                a0 = vfmaq_f32(a0, vld1q_f32(Qh + d), vld1q_f32(Mb + d));
            float dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            for (; d < D; d++) dot += Qh[d] * Mb[d];
            scores[b] += dot * inv_sqrt_D;
        }
    }
}

static void quest_neon(const float *Q, const float *N, const float *X,
                       const uint8_t *hm, uint32_t H_q, uint32_t D,
                       uint32_t n_blocks, float inv_sqrt_D, float *scores) {
    for (uint32_t b = 0; b < n_blocks; b++) scores[b] = 0.0f;
    for (uint32_t h = 0; h < H_q; h++) {
        const float *Qh = Q + (size_t)h * D;
        const float *Nh = N + (size_t)hm[h] * n_blocks * D;
        const float *Xh = X + (size_t)hm[h] * n_blocks * D;
        for (uint32_t b = 0; b < n_blocks; b++) {
            const float *Nb = Nh + (size_t)b * D;
            const float *Xb = Xh + (size_t)b * D;
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
            uint32_t d = 0;
            for (; d + 16 <= D; d += 16) {
                float32x4_t q0 = vld1q_f32(Qh + d),      q1 = vld1q_f32(Qh + d + 4);
                float32x4_t q2 = vld1q_f32(Qh + d + 8),  q3 = vld1q_f32(Qh + d + 12);
                a0 = vaddq_f32(a0, vmaxq_f32(vmulq_f32(q0, vld1q_f32(Nb + d)),
                                             vmulq_f32(q0, vld1q_f32(Xb + d))));
                a1 = vaddq_f32(a1, vmaxq_f32(vmulq_f32(q1, vld1q_f32(Nb + d + 4)),
                                             vmulq_f32(q1, vld1q_f32(Xb + d + 4))));
                a2 = vaddq_f32(a2, vmaxq_f32(vmulq_f32(q2, vld1q_f32(Nb + d + 8)),
                                             vmulq_f32(q2, vld1q_f32(Xb + d + 8))));
                a3 = vaddq_f32(a3, vmaxq_f32(vmulq_f32(q3, vld1q_f32(Nb + d + 12)),
                                             vmulq_f32(q3, vld1q_f32(Xb + d + 12))));
            }
            for (; d + 4 <= D; d += 4) {
                float32x4_t q0 = vld1q_f32(Qh + d);
                a0 = vaddq_f32(a0, vmaxq_f32(vmulq_f32(q0, vld1q_f32(Nb + d)),
                                             vmulq_f32(q0, vld1q_f32(Xb + d))));
            }
            float ub = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            for (; d < D; d++) {
                float qd = Qh[d], a = qd * Nb[d], c = qd * Xb[d];
                ub += (a > c) ? a : c;
            }
            scores[b] += ub * inv_sqrt_D;
        }
    }
}

// ------------------------------------------------------------------ top-K ---
// BSD qsort_r puts the context FIRST, unlike the GNU signature.
static int cmp_desc(void *ctx, const void *x, const void *y) {
    const float *s = (const float *)ctx;
    uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y;
    if (s[a] > s[b]) return -1;
    if (s[a] < s[b]) return 1;
    return (a < b) ? -1 : (a > b);
}

static void topk(const float *s, uint32_t n, uint32_t k, uint32_t *out) {
    uint32_t *idx = malloc(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++) idx[i] = i;
    qsort_r(idx, n, sizeof(uint32_t), (void *)s, cmp_desc);
    memcpy(out, idx, k * sizeof(uint32_t));
    free(idx);
}

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

int main(int argc, char **argv) {
    uint32_t D = 128, H_q = 8, H_kv = 2, B = 64, K_top = 400;
    uint32_t N_ctx = (argc > 1) ? (uint32_t)atoi(argv[1]) : 65536;
    uint32_t n_blocks = (N_ctx + B - 1) / B;
    uint32_t iters = 200;

    srand(12345);
    float *Q = malloc((size_t)H_q * D * sizeof(float));
    size_t pre = (size_t)H_kv * n_blocks * D;
    float *M = malloc(pre * sizeof(float));
    float *Nn = malloc(pre * sizeof(float));
    float *X = malloc(pre * sizeof(float));
    uint8_t *hm = malloc(H_q);
    for (uint32_t h = 0; h < H_q; h++) hm[h] = (uint8_t)(h / (H_q / H_kv));
    for (size_t i = 0; i < (size_t)H_q * D; i++) Q[i] = frand();
    for (size_t i = 0; i < pre; i++) {
        M[i] = frand();
        float a = frand(), b = frand();
        Nn[i] = a < b ? a : b;
        X[i] = a < b ? b : a;
    }

    float *s_ref = malloc(n_blocks * sizeof(float));
    float *s_new = malloc(n_blocks * sizeof(float));
    uint32_t kk = K_top < n_blocks ? K_top : n_blocks;
    uint32_t *t_ref = malloc(kk * sizeof(uint32_t));
    uint32_t *t_new = malloc(kk * sizeof(uint32_t));

    printf("N=%u  n_blocks=%u  D=%u  H_q=%u  H_kv=%u  K_top=%u  iters=%u\n\n",
           N_ctx, n_blocks, D, H_q, H_kv, kk, iters);

    struct { const char *name; int quest; } cases[] = {
        {"block-mean", 0}, {"Quest UB", 1},
    };
    for (int c = 0; c < 2; c++) {
        double t0, t1, t2;
        // warm
        if (cases[c].quest) quest_scalar(Q, Nn, X, hm, H_q, D, n_blocks, 1.f, s_ref);
        else                mean_scalar(Q, M, hm, H_q, D, n_blocks, 1.f, s_ref);
        t0 = now_ms();
        for (uint32_t i = 0; i < iters; i++) {
            if (cases[c].quest) quest_scalar(Q, Nn, X, hm, H_q, D, n_blocks, 1.f, s_ref);
            else                mean_scalar(Q, M, hm, H_q, D, n_blocks, 1.f, s_ref);
        }
        t1 = now_ms();
        for (uint32_t i = 0; i < iters; i++) {
            if (cases[c].quest) quest_neon(Q, Nn, X, hm, H_q, D, n_blocks, 1.f, s_new);
            else                mean_neon(Q, M, hm, H_q, D, n_blocks, 1.f, s_new);
        }
        t2 = now_ms();

        double ms_ref = (t1 - t0) / iters, ms_new = (t2 - t1) / iters;
        float maxd = 0.0f, maxrel = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
            float d = fabsf(s_ref[b] - s_new[b]);
            if (d > maxd) maxd = d;
            float den = fabsf(s_ref[b]);
            if (den > 1e-6f && d / den > maxrel) maxrel = d / den;
        }
        topk(s_ref, n_blocks, kk, t_ref);
        topk(s_new, n_blocks, kk, t_new);
        // Set difference, not order difference: a swap of two adjacent ranks
        // inside the selected set changes nothing downstream.
        uint32_t *seen = calloc(n_blocks, sizeof(uint32_t));
        for (uint32_t i = 0; i < kk; i++) seen[t_ref[i]] = 1;
        uint32_t set_diff = 0;
        for (uint32_t i = 0; i < kk; i++) if (!seen[t_new[i]]) set_diff++;
        uint32_t order_diff = 0;
        for (uint32_t i = 0; i < kk; i++) if (t_ref[i] != t_new[i]) order_diff++;
        free(seen);

        printf("%-11s scalar %7.3f ms   neon %7.3f ms   %.2fx\n",
               cases[c].name, ms_ref, ms_new, ms_ref / ms_new);
        printf("            max|delta| %.3e   max rel %.3e\n", maxd, maxrel);
        printf("            top-%u set differs in %u blocks, order in %u positions\n\n",
               kk, set_diff, order_diff);
    }
    return 0;
}
