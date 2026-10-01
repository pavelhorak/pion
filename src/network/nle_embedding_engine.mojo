"""NLEmbeddingEngine — in-process Mojo wrapper for Apple's NLEmbedding.

Replaces the PyTorch + sentence-transformers + MiniLM-L6-v2 sidecar path
on macOS with a system-framework call. NLEmbedding routes to the Apple
Neural Engine where available — same approach PionMesh validated on iOS
(`../PionMesh/AGENTS.md` line 291).

Output dimension: 512 (Apple's English sentence embedding on macOS 12+).
This is different from the Linux PyTorch sidecar's 384-dim MiniLM-L6-v2
output, so semantic-cache HNSW indexes built on Mac with `--nle-embed`
are not interchangeable with Linux indexes. Pre-launch this is fine; if
data persistence across platforms ever matters, version the cache.

Linux: `pion_nle_*` symbols are not linked (gated in pixi.toml). All
external_calls below are wrapped in `comptime if CompilationTarget.is_macos():`
guards. Same pattern as MetalAttentionEngine.
"""

from std.ffi import external_call
from std.memory.unsafe_pointer import Pointer
from std.sys.info import CompilationTarget


struct NLEmbeddingEngine(Movable):
    """Apple NLEmbedding wrapper. `available` is True iff the system
    NaturalLanguage framework returned a sentence embedding for English on
    init (always true on supported macOS; false on Linux or older macOS)."""

    var enabled: Bool
    var available: Bool
    var dimension: UInt32

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.available = False
        self.dimension = UInt32(0)
        comptime if CompilationTarget.is_macos():
            if enabled:
                var rc = external_call["pion_nle_init", Int32]()
                if rc == 0:
                    self.available = True
                    self.dimension = external_call["pion_nle_dimension", UInt32]()
                else:
                    print("[NLE] init failed (rc=", rc, "); auto-embed will fall back to PyTorch sidecar")

    def __moveinit__(out self, deinit take: Self):
        self.enabled = take.enabled
        self.available = take.available
        self.dimension = take.dimension

    @always_inline
    def embed(mut self,
              text_ptr: Pointer[UInt8, MutUntrackedOrigin],
              text_len: Int,
              out_buf: Pointer[Float32, MutUntrackedOrigin],
              out_capacity: Int) -> Int:
        """Embed UTF-8 text into `out_buf`. Returns the dimension written
        (positive) on success, 0 if NLE didn't recognize the text, negative
        on error. Matches the contract InferenceBridge.embed_blocking uses."""
        if not self.available:
            return -1
        comptime if CompilationTarget.is_macos():
            return Int(external_call["pion_nle_embed", Int32](
                text_ptr, UInt32(text_len),
                out_buf, UInt32(out_capacity),
            ))
        else:
            return -1
