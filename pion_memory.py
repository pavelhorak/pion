"""
pion_memory — Python SDK for the Pion in-process memory library.

Zero-copy, sub-millisecond vector memory for AI workloads.
No server required. Callable during model inference.

Usage:
    from pion_memory import PionMemory
    import numpy as np

    mem = PionMemory(dim=1536, max_elements=100_000)

    # Stage vectors
    for i, embedding in enumerate(embeddings):
        mem.remember(i, embedding)

    # Build index (one-time, O(N log N))
    mem.optimize()

    # Search
    results = mem.recall(query_embedding, k=10)
    # results = [(id, score), ...]
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def _load_library() -> ctypes.CDLL:
    """Locate and load libpion_memory."""
    # Search order: env var, project root, standard locations
    search_paths = []

    env_path = os.environ.get("PION_MEMORY_LIB")
    if env_path:
        search_paths.append(env_path)

    # Project root (same dir as this file)
    here = Path(__file__).parent
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    search_paths.extend([
        str(here / f"libpion_memory{ext}"),
        str(here / f"libpion_memory.so"),
        str(here / f"libpion_memory.dylib"),
    ])

    for path in search_paths:
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError as e:
                raise RuntimeError(f"Found {path} but failed to load: {e}")

    raise FileNotFoundError(
        f"libpion_memory not found. Build it with:\n"
        f"  pixi run build-lib\n"
        f"Or set PION_MEMORY_LIB=/path/to/libpion_memory{ext}"
    )


def _configure_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Set argtypes and restype for all exported functions."""
    c_void_p = ctypes.c_void_p
    c_i32 = ctypes.c_int32
    c_f32 = ctypes.c_float
    c_f32_p = ctypes.POINTER(ctypes.c_float)
    c_i32_p = ctypes.POINTER(ctypes.c_int32)

    lib.pion_memory_create.argtypes = [c_i32, c_i32, c_i32, c_i32]
    lib.pion_memory_create.restype = c_void_p

    lib.pion_memory_destroy.argtypes = [c_void_p]
    lib.pion_memory_destroy.restype = None

    lib.pion_remember.argtypes = [c_void_p, c_i32, c_f32_p]
    lib.pion_remember.restype = c_i32

    lib.pion_optimize.argtypes = [c_void_p]
    lib.pion_optimize.restype = c_i32

    lib.pion_recall.argtypes = [c_void_p, c_f32_p, c_i32, c_i32, c_i32_p, c_f32_p]
    lib.pion_recall.restype = c_i32

    lib.pion_forget.argtypes = [c_void_p, c_i32]
    lib.pion_forget.restype = c_i32

    lib.pion_count.argtypes = [c_void_p]
    lib.pion_count.restype = c_i32

    lib.pion_is_ready.argtypes = [c_void_p]
    lib.pion_is_ready.restype = c_i32

    return lib


# Module-level lazy library load
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _configure_lib(_load_library())
    return _lib


class PionMemory:
    """In-process HNSW vector memory — no server, no network, no serialization.

    Args:
        dim:             Embedding dimension (e.g. 1536 for text-embedding-3-large)
        max_elements:    Maximum number of vectors (pre-allocates memory)
        M:               HNSW M parameter — connections per node (default 16)
        ef_construction: Build quality parameter (default 128)
        lib_path:        Optional explicit path to libpion_memory
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100_000,
        M: int = 16,
        ef_construction: int = 128,
        lib_path: Optional[str] = None,
    ):
        if lib_path:
            global _lib
            _lib = _configure_lib(ctypes.CDLL(lib_path))

        lib = _get_lib()
        self._lib = lib
        self._dim = dim
        self._max_k = 1024  # max results buffer

        self._handle = lib.pion_memory_create(
            ctypes.c_int32(dim),
            ctypes.c_int32(max_elements),
            ctypes.c_int32(M),
            ctypes.c_int32(ef_construction),
        )
        if not self._handle:
            raise MemoryError(
                f"pion_memory_create failed for dim={dim}, max_elements={max_elements}"
            )

        # Pre-allocate result buffers (reused across recall() calls)
        self._out_ids = (ctypes.c_int32 * self._max_k)()
        self._out_scores = (ctypes.c_float * self._max_k)()

    def remember(self, id: int, vector) -> None:
        """Stage a vector for indexing. Call optimize() when done adding.

        Args:
            id:     Integer ID (returned in recall results)
            vector: numpy array (float32, length=dim) or any array-like
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self._dim,):
            raise ValueError(f"Expected shape ({self._dim},), got {arr.shape}")
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self._lib.pion_remember(self._handle, ctypes.c_int32(id), ptr)
        if rc != 0:
            raise RuntimeError(f"pion_remember failed for id={id} (buffer full?)")

    def optimize(self) -> None:
        """Build the HNSW index. Must be called before recall().

        Call once after all remember() calls, or after each batch of inserts.
        Rebuilds the full index from staged vectors — O(N log N).
        """
        rc = self._lib.pion_optimize(self._handle)
        if rc != 0:
            raise RuntimeError("pion_optimize failed (no staged vectors?)")

    def recall(
        self,
        query,
        k: int = 10,
        ef: int = 150,
    ) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors of query.

        Args:
            query: numpy array (float32, length=dim) or any array-like
            k:     Number of results to return
            ef:    Search beam width (100–200 for ≥0.93 recall at 1536-dim)

        Returns:
            List of (id, l2_distance) tuples, sorted by distance ascending.

        Raises:
            RuntimeError: if optimize() has not been called.
        """
        arr = np.asarray(query, dtype=np.float32)
        if arr.shape != (self._dim,):
            raise ValueError(f"Expected shape ({self._dim},), got {arr.shape}")

        if k > self._max_k:
            raise ValueError(f"k={k} exceeds max buffer size {self._max_k}")

        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        count = self._lib.pion_recall(
            self._handle,
            ptr,
            ctypes.c_int32(k),
            ctypes.c_int32(ef),
            self._out_ids,
            self._out_scores,
        )
        if count < 0:
            raise RuntimeError(
                "pion_recall failed — did you call optimize() first?"
            )
        return [(int(self._out_ids[i]), float(self._out_scores[i]))
                for i in range(count)]

    def forget(self, id: int) -> None:
        """Remove a vector from the index. [Phase 1 — not yet implemented]"""
        raise NotImplementedError(
            "pion_forget is not implemented in Phase 0. Coming in Phase 1."
        )

    @property
    def count(self) -> int:
        """Number of indexed vectors (after optimize)."""
        return int(self._lib.pion_count(self._handle))

    @property
    def is_ready(self) -> bool:
        """True if the index is built and recall() can be called."""
        return bool(self._lib.pion_is_ready(self._handle))

    def __len__(self) -> int:
        return self.count

    def __del__(self):
        if hasattr(self, "_handle") and self._handle and hasattr(self, "_lib"):
            self._lib.pion_memory_destroy(self._handle)
            self._handle = None

    def __repr__(self) -> str:
        status = "ready" if self.is_ready else "not optimized"
        return (
            f"PionMemory(dim={self._dim}, count={self.count}, status={status})"
        )
