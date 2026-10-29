/* Link stubs for Mojo unit tests that import src.vector.hnsw on macOS.
 * HNSWGraph names the Metal hooks in src/ffi/metal_wrap.m behind
 * `comptime if is_macos()`; a test builds no Metal context, so availability
 * answers 0 and the rest are never reached. Linking metal_wrap.m instead would
 * pull in the Metal and Foundation frameworks for a CPU-only test. */
#include <stdint.h>

int32_t pion_metal_available(void) { return 0; }
int32_t pion_metal_register_vectors(int8_t *b, uint32_t n, uint32_t s) { (void)b; (void)n; (void)s; return -1; }
int32_t pion_metal_register_rerank_buffer(float *b, uint32_t n, uint32_t d) { (void)b; (void)n; (void)d; return -1; }
int32_t pion_metal_unregister_rerank_buffer(void) { return 0; }
