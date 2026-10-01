"""Codebase indexer — chunks source files and stores embeddings in Pion.

Walks a project directory, splits files into semantic chunks (functions,
classes, or fixed-size blocks), embeds each chunk, and stores them in
Pion's HNSW index for semantic retrieval.

Supports incremental indexing: only re-indexes files whose content hash
has changed since the last index run.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import redis as redis_lib

from .embeddings import embed_text, embed_texts, embed_dim

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_NAME = "__codebase__"
HASH_PREFIX = "cb:"
CHECKSUM_KEY = "__cb_checksums__"
COUNTER_KEY = "__cb_seq__"
OPTIMIZE_EVERY = 100  # auto-optimize after N new chunks

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".mojo", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql", ".lua",
    ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".rst",
}

# Directories to skip
SKIP_DIRS = {
    ".git", ".pixi", ".pixi-linux", "node_modules", "__pycache__", ".pytest_cache",
    ".venv", "venv", ".tox", "dist", "build", ".eggs", ".mypy_cache",
    ".ruff_cache", "target", ".next", ".nuxt", ".claude",
    # virtualenvs and local scratch / data dirs
    "venv", ".venv", "localtemp",
    "dataset", "models", ".kv_cache",
}

# Max file size to index (512KB)
MAX_FILE_SIZE = 512 * 1024

# Chunk size limits
CHUNK_MAX_LINES = 80
CHUNK_MIN_LINES = 5


@dataclass
class Chunk:
    """A semantic chunk of source code."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    kind: str = "block"  # "function", "class", "block"
    name: str = ""


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_file(file_path: str, content: str) -> list[Chunk]:
    """Split a file into semantic chunks.

    Strategy:
    1. Try to split on function/class boundaries (Python, Mojo, JS/TS, etc.)
    2. Fall back to fixed-size line blocks with overlap
    """
    lines = content.split("\n")
    if len(lines) < CHUNK_MIN_LINES:
        return [Chunk(
            file_path=file_path, start_line=1, end_line=len(lines),
            content=content, kind="file", name=os.path.basename(file_path),
        )]

    ext = os.path.splitext(file_path)[1].lower()

    # Try semantic chunking for supported languages
    if ext in (".py", ".mojo"):
        chunks = _chunk_python_like(file_path, lines)
        if chunks:
            return chunks
    elif ext in (".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs"):
        chunks = _chunk_brace_lang(file_path, lines)
        if chunks:
            return chunks

    # Fallback: fixed-size blocks
    return _chunk_fixed(file_path, lines)


def _chunk_python_like(file_path: str, lines: list[str]) -> list[Chunk]:
    """Chunk Python/Mojo files on def/class/fn/struct boundaries."""
    pattern = re.compile(r"^(class |def |fn |struct |async def )")
    chunks = []
    current_start = 0

    for i, line in enumerate(lines):
        if pattern.match(line.lstrip()) and i > current_start + CHUNK_MIN_LINES:
            # Emit previous block
            block = "\n".join(lines[current_start:i])
            if block.strip():
                name = _extract_name(lines[current_start])
                chunks.append(Chunk(
                    file_path=file_path, start_line=current_start + 1,
                    end_line=i, content=block, kind="function", name=name,
                ))
            current_start = i

    # Emit final block
    if current_start < len(lines):
        block = "\n".join(lines[current_start:])
        if block.strip():
            name = _extract_name(lines[current_start])
            chunks.append(Chunk(
                file_path=file_path, start_line=current_start + 1,
                end_line=len(lines), content=block, kind="function", name=name,
            ))

    # If chunks are too large, sub-split them
    result = []
    for chunk in chunks:
        chunk_lines = chunk.content.split("\n")
        if len(chunk_lines) > CHUNK_MAX_LINES:
            sub = _chunk_fixed(file_path, chunk_lines, offset=chunk.start_line - 1)
            result.extend(sub)
        else:
            result.append(chunk)

    return result


def _chunk_brace_lang(file_path: str, lines: list[str]) -> list[Chunk]:
    """Chunk brace-delimited languages on top-level function/class boundaries."""
    pattern = re.compile(
        r"^\s*(export\s+)?(async\s+)?(function|class|interface|struct|impl|fn|func|def|pub fn|pub struct)\s"
    )
    chunks = []
    current_start = 0

    for i, line in enumerate(lines):
        if pattern.match(line) and i > current_start + CHUNK_MIN_LINES:
            block = "\n".join(lines[current_start:i])
            if block.strip():
                chunks.append(Chunk(
                    file_path=file_path, start_line=current_start + 1,
                    end_line=i, content=block, kind="function",
                    name=_extract_name(lines[current_start]),
                ))
            current_start = i

    if current_start < len(lines):
        block = "\n".join(lines[current_start:])
        if block.strip():
            chunks.append(Chunk(
                file_path=file_path, start_line=current_start + 1,
                end_line=len(lines), content=block, kind="function",
                name=_extract_name(lines[current_start]),
            ))

    result = []
    for chunk in chunks:
        if len(chunk.content.split("\n")) > CHUNK_MAX_LINES:
            result.extend(_chunk_fixed(file_path, chunk.content.split("\n"), offset=chunk.start_line - 1))
        else:
            result.append(chunk)

    return result


def _chunk_fixed(file_path: str, lines: list[str], offset: int = 0) -> list[Chunk]:
    """Split into fixed-size blocks with 10-line overlap."""
    chunks = []
    stride = CHUNK_MAX_LINES - 10  # 10-line overlap
    for i in range(0, len(lines), max(stride, 1)):
        block_lines = lines[i:i + CHUNK_MAX_LINES]
        block = "\n".join(block_lines)
        if block.strip():
            chunks.append(Chunk(
                file_path=file_path, start_line=offset + i + 1,
                end_line=offset + i + len(block_lines),
                content=block, kind="block",
                name=f"{os.path.basename(file_path)}:{offset + i + 1}",
            ))
    return chunks


def _extract_name(line: str) -> str:
    """Extract function/class name from a definition line."""
    m = re.match(r"(?:export\s+)?(?:async\s+)?(?:def|fn|function|class|struct|impl|func|pub fn|pub struct)\s+(\w+)", line.strip())
    return m.group(1) if m else line.strip()[:50]


# ── Indexing Engine ───────────────────────────────────────────────────────────

class CodebaseIndexer:
    """Indexes a codebase into Pion's HNSW index."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1974):
        self.conn = redis_lib.Redis(
            host=host, port=port,
            decode_responses=False,
            socket_keepalive=True,
        )
        self._inserts_since_optimize = 0

    def _ensure_index(self):
        """Create the codebase index if it doesn't exist."""
        try:
            self.conn.execute_command("FT.INFO", INDEX_NAME)
        except Exception:
            dim = embed_dim()
            self.conn.execute_command(
                "FT.CREATE", INDEX_NAME,
                "SCHEMA", "vec", "VECTOR", "HNSW",
                "10",
                "TYPE", "FLOAT32",
                "DIM", str(dim),
                "DISTANCE_METRIC", "COSINE",
                "M", "16",
                "EF_CONSTRUCTION", "128",
            )

    def _get_checksum(self, file_path: str) -> Optional[str]:
        """Get stored checksum for a file."""
        val = self.conn.hget(CHECKSUM_KEY, file_path)
        return val.decode() if val else None

    def _set_checksum(self, file_path: str, checksum: str):
        """Store checksum for a file."""
        self.conn.hset(CHECKSUM_KEY, file_path, checksum)

    def _file_checksum(self, content: str) -> str:
        """Compute SHA256 of file content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _next_id(self) -> int:
        """Get next sequential chunk ID."""
        return self.conn.incr(COUNTER_KEY)

    def index_file(self, file_path: str, force: bool = False) -> int:
        """Index a single file. Returns number of chunks indexed.

        Skips if file content hasn't changed (unless force=True).
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return 0

        if len(content) > MAX_FILE_SIZE:
            return 0

        checksum = self._file_checksum(content)
        if not force:
            stored = self._get_checksum(file_path)
            if stored == checksum:
                return 0  # unchanged

        self._ensure_index()

        chunks = chunk_file(file_path, content)
        if not chunks:
            return 0

        # Remove old chunks for this file
        self._remove_file_chunks(file_path)

        # Embed and store chunks
        texts = [f"{c.file_path}:{c.start_line} ({c.kind} {c.name})\n{c.content}" for c in chunks]

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed for {file_path}: {e}") from e

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self._next_id()
            doc_key = f"{HASH_PREFIX}{chunk_id}"

            self.conn.hset(doc_key, mapping={
                "vec": embedding,
                "text": chunk.content.encode("utf-8"),
                "file_path": chunk.file_path.encode("utf-8"),
                "start_line": str(chunk.start_line).encode("utf-8"),
                "end_line": str(chunk.end_line).encode("utf-8"),
                "kind": chunk.kind.encode("utf-8"),
                "name": chunk.name.encode("utf-8"),
                "index": INDEX_NAME.encode("utf-8"),
            })

            self._inserts_since_optimize += 1

        self._set_checksum(file_path, checksum)

        # NOTE: Do NOT auto-optimize here. Pion's HNSW ingest buffer is freed
        # after FT.OPTIMIZE, so subsequent HSETs would silently skip vector
        # routing. All inserts must complete before a single FT.OPTIMIZE call.

        return len(chunks)

    def _remove_file_chunks(self, file_path: str):
        """Remove all chunks for a file (for re-indexing)."""
        # Scan for chunks with this file_path
        # Note: this is O(N) but only runs during re-index of a single file
        cursor = 0
        file_path_bytes = file_path.encode("utf-8")
        while True:
            cursor, keys = self.conn.scan(cursor, match=f"{HASH_PREFIX}*", count=500)
            for key in keys:
                stored_path = self.conn.hget(key, "file_path")
                if stored_path == file_path_bytes:
                    self.conn.delete(key)
            if cursor == 0:
                break

    def index_directory(
        self,
        root: str,
        extensions: set[str] | None = None,
        skip_dirs: set[str] | None = None,
        force: bool = False,
    ) -> IndexStats:
        """Index all matching files in a directory tree.

        Args:
            root: Root directory to walk.
            extensions: File extensions to include (default: CODE_EXTENSIONS).
            skip_dirs: Directory names to skip (default: SKIP_DIRS).
            force: Re-index all files even if unchanged.
        """
        if extensions is None:
            extensions = CODE_EXTENSIONS
        if skip_dirs is None:
            skip_dirs = SKIP_DIRS

        stats = IndexStats()
        t0 = time.time()

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue

                file_path = os.path.join(dirpath, filename)
                stats.files_scanned += 1

                try:
                    n = self.index_file(file_path, force=force)
                    if n > 0:
                        stats.files_indexed += 1
                        stats.chunks_created += n
                        stats.chunks_embedded += n
                    else:
                        stats.files_skipped += 1
                except Exception as e:
                    stats.errors.append(f"{file_path}: {e}")

        # Final optimize
        if stats.chunks_created > 0:
            self.optimize()

        stats.elapsed_s = time.time() - t0
        return stats

    def optimize(self):
        """Build/rebuild the HNSW index."""
        try:
            self.conn.execute_command("FT.OPTIMIZE", INDEX_NAME)
        except Exception:
            pass  # index may not exist yet

    def stats(self) -> dict:
        """Get index statistics."""
        try:
            raw = self.conn.execute_command("FT.INFO", INDEX_NAME)
            result = {}
            if isinstance(raw, list):
                for i in range(0, len(raw) - 1, 2):
                    k = raw[i].decode() if isinstance(raw[i], bytes) else str(raw[i])
                    v = raw[i + 1]
                    if isinstance(v, bytes):
                        v = v.decode()
                    result[k] = v
            return result
        except Exception:
            return {"status": "no index"}
