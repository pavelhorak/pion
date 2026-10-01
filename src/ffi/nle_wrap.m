// nle_wrap.m — Apple NaturalLanguage NLEmbedding wrapper for Pion auto-embed.
//
// Replaces the PyTorch + MiniLM-L6-v2 sidecar path on macOS with a system-
// framework call. NLEmbedding routes to the Apple Neural Engine where
// available (Apple Silicon), so we get hardware acceleration without writing
// CoreML conversion code. PionMesh's iOS app validates the same approach
// (../PionMesh/AGENTS.md line 291).
//
// Compiles with:
//   clang -c -fobjc-arc src/ffi/nle_wrap.m -o src/ffi/nle_wrap.o \
//     -framework Foundation -framework NaturalLanguage
// Links into pion-server via -framework NaturalLanguage on the mojo build line.
//
// Linux: this file is NOT compiled (pixi.toml gates the rule on macos-arm64).
// All callers in Mojo are wrapped in `comptime if CompilationTarget.is_macos():`
// guards, mirroring the metal_wrap.m pattern.

#import <Foundation/Foundation.h>
#import <NaturalLanguage/NaturalLanguage.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>

// Single shared NLEmbedding instance for English. Apple's API is thread-safe
// for query (`vectorForString:`); only init needs to be serialized.
static NLEmbedding *g_nle = nil;
static atomic_int g_nle_dimension = 0;
static pthread_mutex_t g_nle_mutex = PTHREAD_MUTEX_INITIALIZER;
static atomic_int g_nle_init_status = 0;  // 0=unstarted, 1=in_progress, 2=ready, -1..= error codes

// Initialize the NLEmbedding model for English sentence embeddings. Idempotent
// and thread-safe; all workers may call concurrently. Returns 0 on success,
// negative on error.
int32_t pion_nle_init(void) {
    int status = atomic_load(&g_nle_init_status);
    if (status == 2) return 0;
    if (status < 0) return status;

    pthread_mutex_lock(&g_nle_mutex);
    status = atomic_load(&g_nle_init_status);
    if (status == 2) { pthread_mutex_unlock(&g_nle_mutex); return 0; }
    if (status < 0) { pthread_mutex_unlock(&g_nle_mutex); return status; }

    @autoreleasepool {
        NLEmbedding *e = [NLEmbedding sentenceEmbeddingForLanguage:NLLanguageEnglish];
        if (!e) {
            fprintf(stderr, "[NLE] sentenceEmbedding for English unavailable on this OS\n");
            atomic_store(&g_nle_init_status, -1);
            pthread_mutex_unlock(&g_nle_mutex);
            return -1;
        }
        g_nle = e;
        atomic_store(&g_nle_dimension, (int)[e dimension]);
        atomic_store(&g_nle_init_status, 2);
        fprintf(stderr, "[NLE] initialized — sentence embedding (English), dim=%d\n",
                atomic_load(&g_nle_dimension));
    }
    pthread_mutex_unlock(&g_nle_mutex);
    return 0;
}

// Returns the embedding dimension (512 on macOS 12+ for English sentence
// embedding) once initialized; 0 before init or on error.
uint32_t pion_nle_dimension(void) {
    return (uint32_t)atomic_load(&g_nle_dimension);
}

// Embed a UTF-8 text into `out_buf` as `dim` float32 values.
// Returns the dimension written on success, 0 on text-not-recognized,
// negative on error.
//
// Thread-safe: NLEmbedding.vectorForString: is documented thread-safe on
// the same NLEmbedding instance.
int32_t pion_nle_embed(const char *text, uint32_t text_len,
                       float *out_buf, uint32_t out_capacity) {
    if (atomic_load(&g_nle_init_status) != 2) return -1;
    if (!text || text_len == 0 || !out_buf) return -2;
    int dim = atomic_load(&g_nle_dimension);
    if ((int)out_capacity < dim) return -3;

    @autoreleasepool {
        NSString *s = [[NSString alloc] initWithBytes:text length:text_len
                                              encoding:NSUTF8StringEncoding];
        if (!s) return -4;

        NSArray<NSNumber *> *vec = [g_nle vectorForString:s];
        if (!vec || (int)vec.count != dim) return 0;  // not recognized → 0 (caller falls back)

        // NSArray<NSNumber*> stores doubles. Downcast to float, AND normalize to
        // unit-norm so Pion's HNSW INT8 distance kernel (which assumes unit-norm
        // input — see semantic_cache.mojo line 33) reads cosine similarity
        // correctly. NLE's raw vectors have norm ≈ 11 at 512-dim, which would
        // make the INT8 quantizer saturate and the cosine threshold meaningless.
        double norm_sq = 0.0;
        for (int i = 0; i < dim; ++i) {
            double x = [vec[i] doubleValue];
            out_buf[i] = (float)x;
            norm_sq += x * x;
        }
        if (norm_sq > 1e-12) {
            float inv_norm = (float)(1.0 / sqrt(norm_sq));
            for (int i = 0; i < dim; ++i) out_buf[i] *= inv_norm;
        }
        return dim;
    }
}

// Returns 1 if NLE is initialized and ready, 0 otherwise.
int32_t pion_nle_available(void) {
    return atomic_load(&g_nle_init_status) == 2 ? 1 : 0;
}
