"""Codebase indexer — chunks source files and stores embeddings in Pion.

Walks a project directory, splits files into semantic chunks (functions,
classes, or fixed-size blocks), embeds each chunk, and stores them in
Pion's HNSW index for semantic retrieval.

Supports incremental indexing: only re-indexes files whose content hash
has changed since the last index run. Pion builds an index once
(ingest -> FT.OPTIMIZE -> search), so a change to a built index is applied by
rebuilding it from the vectors already stored with each chunk (`rebuild`).
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import redis as redis_lib

from .embeddings import embed_text, embed_texts, embed_dim

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_NAME = "__codebase__"
HASH_PREFIX = "cb:"
FILE_KEYS_PREFIX = "cbf:"         # cbf:<file_path> -> set of that file's chunk keys
CHECKSUM_KEY = "__cb_checksums__"
COUNTER_KEY = "__cb_seq__"
LAYOUT_KEY = "__cb_layout__"      # "2": chunk keys are tracked per file in cbf: sets
REBUILD_LOCK_KEY = "__cb_rebuild_lock__"
ROOTS_KEY = "__cb_roots__"        # absolute root -> the root as it was given to index_directory

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".mojo", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql", ".lua",
    ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".rst",
}

# Inside a git work tree the files to index are git's: tracked plus untracked
# but not ignored (`git ls-files --cached --others --exclude-standard`), so
# .gitignore decides and no source directory is dropped for its name. This
# list is applied on top, and is all that applies outside a git work tree. It
# used to hold `models` and `dataset` too, which dropped Django's ORM
# (django/db/models/) and any other source directory with those names.
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".pixi", ".pixi-linux", "node_modules", "__pycache__",
    ".pytest_cache", ".venv", "venv", ".tox", ".eggs", ".mypy_cache", ".ruff_cache",
    ".next", ".nuxt", ".claude", ".kv_cache",
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

def list_files(root: str, extensions: set[str] | None = None,
               skip_dirs: set[str] | None = None) -> list[str]:
    """The files to index under `root`, as paths that start with `root`.

    In a git work tree: tracked files plus untracked files that are not
    ignored. Elsewhere: a walk that skips `skip_dirs`. Either way only
    `extensions` are kept and `skip_dirs` components are dropped.
    """
    extensions = CODE_EXTENSIONS if extensions is None else extensions
    skip_dirs = SKIP_DIRS if skip_dirs is None else skip_dirs
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True, timeout=60,
        ).stdout.decode("utf-8", "surrogateescape")
        candidates = [os.path.join(root, rel) for rel in out.split("\0") if rel]
    except (OSError, subprocess.SubprocessError):
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
            candidates += [os.path.join(dirpath, f) for f in sorted(filenames)]
    files = []
    for path in candidates:
        parts = os.path.normpath(path).split(os.sep)
        if any(part in skip_dirs for part in parts[:-1]):
            continue
        if os.path.splitext(path)[1].lower() in extensions and os.path.isfile(path):
            files.append(path)
    return files


class CodebaseIndexer:
    """Indexes a codebase into Pion's HNSW index."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1974):
        self.conn = redis_lib.Redis(
            host=host, port=port,
            decode_responses=False,
            socket_keepalive=True,
        )

    # ── index lifecycle ──────────────────────────────────────────────────────

    def _index_exists(self) -> bool:
        try:
            self.conn.execute_command("FT.INFO", INDEX_NAME)
            return True
        except redis_lib.ResponseError:
            return False

    def _create_index(self):
        self.conn.execute_command(
            "FT.CREATE", INDEX_NAME,
            "SCHEMA", "vec", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(embed_dim()),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "128",
        )

    def _ensure_index(self):
        """Create the codebase index if it doesn't exist."""
        if not self._index_exists():
            self._create_index()
            self.conn.set(LAYOUT_KEY, "2")

    def optimize(self):
        """Build the HNSW graph from the chunks ingested since FT.CREATE."""
        try:
            self.conn.execute_command("FT.OPTIMIZE", INDEX_NAME)
        except redis_lib.ResponseError:
            pass  # index may not exist yet

    def rebuild(self, lock_wait_s: float = 60.0) -> int:
        """Apply changes to a built index by building it again.

        Pion's index contract is ingest -> FT.OPTIMIZE -> search: a vector
        written after FT.OPTIMIZE is stored with its hash but never enters the
        graph. So: drop the index (FT.DROPINDEX keeps the documents), create it
        again, send every chunk's vector again (read back from its own hash,
        so nothing is re-embedded), and optimize. Returns the chunk count.
        Searches fail for the moment between the drop and the optimize.
        """
        deadline = time.time() + lock_wait_s
        while not self.conn.set(REBUILD_LOCK_KEY, str(os.getpid()), nx=True, px=int(lock_wait_s * 1000)):
            if time.time() > deadline:
                raise TimeoutError("another rebuild of the codebase index is still running")
            time.sleep(0.1)
        try:
            keys = list(self.conn.scan_iter(f"{HASH_PREFIX}*", count=1000))
            vectors = []
            for i in range(0, len(keys), 500):
                p = self.conn.pipeline(transaction=False)
                for k in keys[i:i + 500]:
                    p.hget(k, "vec")
                vectors += p.execute()
            missing = [k for k, v in zip(keys, vectors) if not v]
            if missing:
                # Refuse before dropping anything: rebuilding without these
                # vectors would silently lose their chunks.
                raise RuntimeError(
                    f"{len(missing)} chunks (e.g. {missing[0]!r}) have no stored vector; "
                    "re-index with --force into a fresh server")
            try:
                self.conn.execute_command("FT.DROPINDEX", INDEX_NAME)
            except redis_lib.ResponseError:
                pass
            self._create_index()
            p = self.conn.pipeline(transaction=False)
            for n, (k, v) in enumerate(zip(keys, vectors), start=1):
                p.hset(k, "vec", v)
                if n % 500 == 0:
                    p.execute()
            p.execute()
            self.conn.execute_command("FT.OPTIMIZE", INDEX_NAME)
            return len(keys)
        finally:
            self.conn.delete(REBUILD_LOCK_KEY)

    def _finish(self, existed: bool):
        """Make the index searchable after a batch of changes."""
        if existed:
            self.rebuild()
        else:
            self.optimize()

    # ── files ────────────────────────────────────────────────────────────────

    def canonical_path(self, file_path: str) -> str:
        """The path a file is stored under. Chunks keep the path as the walk
        of their root produced it ("./pkg/x.py", "src/x.py"); a hook passes
        an absolute path, which is mapped back through the indexed roots so
        it replaces the file's chunks instead of adding a second copy."""
        if not os.path.isabs(file_path):
            return file_path
        real = os.path.realpath(file_path)       # macOS: /var is /private/var
        best = None
        for raw_abs, raw_given in self.conn.hgetall(ROOTS_KEY).items():
            root_abs = raw_abs.decode("utf-8", "replace")
            if real == root_abs or real.startswith(root_abs.rstrip(os.sep) + os.sep):
                if best is None or len(root_abs) > len(best[0]):
                    best = (root_abs, raw_given.decode("utf-8", "replace"))
        if best is None:
            return file_path
        return os.path.join(best[1], os.path.relpath(real, best[0]))

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

    def _remove_file_chunks(self, file_path: str):
        """Remove all chunks of a file."""
        set_key = f"{FILE_KEYS_PREFIX}{file_path}"
        keys = self.conn.smembers(set_key)
        if keys:
            self.conn.delete(*keys)
        self.conn.delete(set_key)
        if self.conn.get(LAYOUT_KEY) == b"2":
            return
        # An index written before chunk keys were tracked per file: scan.
        file_path_bytes = file_path.encode("utf-8")
        for key in self.conn.scan_iter(f"{HASH_PREFIX}*", count=500):
            if self.conn.hget(key, "file_path") == file_path_bytes:
                self.conn.delete(key)

    def remove_file(self, file_path: str):
        """Drop a file (deleted or emptied) from the index. No rebuild is
        needed: a document leaves the results when its hash is deleted."""
        self._remove_file_chunks(file_path)
        self.conn.hdel(CHECKSUM_KEY, file_path)

    def index_file(self, file_path: str, force: bool = False, finish: bool = True) -> int:
        """Index a single file. Returns number of chunks indexed.

        Skips if file content hasn't changed (unless force=True). With
        finish=True the index is searchable again on return: optimized if
        this call created it, rebuilt if it was already built.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return 0
        file_path = self.canonical_path(file_path)

        if len(content) > MAX_FILE_SIZE:
            return 0

        checksum = self._file_checksum(content)
        if not force:
            stored = self._get_checksum(file_path)
            if stored == checksum:
                return 0  # unchanged

        existed = self._index_exists()
        self._ensure_index()

        chunks = chunk_file(file_path, content) if content.strip() else []
        if not chunks:
            if self._get_checksum(file_path) is not None:
                self.remove_file(file_path)      # emptied: its old chunks go
            return 0

        # Remove old chunks for this file
        self._remove_file_chunks(file_path)

        # Embed and store chunks
        texts = [f"{c.file_path}:{c.start_line} ({c.kind} {c.name})\n{c.content}" for c in chunks]

        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            raise RuntimeError(f"Embedding failed for {file_path}: {e}") from e

        p = self.conn.pipeline(transaction=False)
        doc_keys = []
        for chunk, embedding in zip(chunks, embeddings):
            doc_key = f"{HASH_PREFIX}{self._next_id()}"
            doc_keys.append(doc_key)
            p.hset(doc_key, mapping={
                "vec": embedding,
                "text": chunk.content.encode("utf-8"),
                "file_path": chunk.file_path.encode("utf-8"),
                "start_line": str(chunk.start_line).encode("utf-8"),
                "end_line": str(chunk.end_line).encode("utf-8"),
                "kind": chunk.kind.encode("utf-8"),
                "name": chunk.name.encode("utf-8"),
                "index": INDEX_NAME.encode("utf-8"),
            })
        p.sadd(f"{FILE_KEYS_PREFIX}{file_path}", *doc_keys)
        p.execute()

        self._set_checksum(file_path, checksum)
        if finish:
            self._finish(existed)
        return len(chunks)

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
        stats = IndexStats()
        t0 = time.time()
        existed = self._index_exists()
        self.conn.hset(ROOTS_KEY, os.path.realpath(root), root)
        files = list_files(root, extensions, skip_dirs)
        changed = False

        for file_path in files:
            stats.files_scanned += 1
            try:
                n = self.index_file(file_path, force=force, finish=False)
                if n > 0:
                    stats.files_indexed += 1
                    stats.chunks_created += n
                    stats.chunks_embedded += n
                    changed = True
                else:
                    stats.files_skipped += 1
            except Exception as e:
                stats.errors.append(f"{file_path}: {e}")

        # Files indexed earlier under this root that are gone now
        present = set(files)
        prefix = os.path.join(root, "")
        for raw in self.conn.hkeys(CHECKSUM_KEY):
            path = raw.decode("utf-8", "replace")
            if (path.startswith(prefix) or os.path.dirname(path) == root) and path not in present:
                self.remove_file(path)

        if changed:
            self._finish(existed)

        stats.elapsed_s = time.time() - t0
        return stats

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
