"""Harvest real code snippets from the Pion repository for realistic prompts."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source files to harvest — mix of Python and Mojo, various subsystems
SOURCE_FILES = [
    ("pion_context/engine.py", "python"),
    ("pion_context/indexer.py", "python"),
    ("pion_context/embeddings.py", "python"),
    ("mcp/pion_mcp/server.py", "python"),
    ("flare_gateway/gateway.py", "python"),
    ("src/network/semantic_cache.mojo", "mojo"),
    ("src/commands/kv_cache.mojo", "mojo"),
    ("src/network/response_writer.mojo", "mojo"),
    ("src/common/hash_map.mojo", "mojo"),
    ("src/network/speculative_rag.mojo", "mojo"),
]

CHUNK_MIN_LINES = 15
CHUNK_MAX_LINES = 60


@dataclass
class CodeSnippet:
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int

    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)


def _chunk_lines(lines: list[str], min_lines: int, max_lines: int) -> list[tuple[int, int]]:
    """Split line list into chunk ranges, trying to break at blank lines."""
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + max_lines, len(lines))
        # Try to find a blank line near the end for a clean break
        best_break = end
        for j in range(end - 1, max(i + min_lines - 1, i), -1):
            if lines[j].strip() == "":
                best_break = j + 1
                break
        if best_break - i < min_lines and end - i >= min_lines:
            best_break = end
        chunks.append((i, best_break))
        i = best_break
    return chunks


def load_snippets() -> list[CodeSnippet]:
    """Load and chunk all source files into CodeSnippets."""
    snippets = []
    for rel_path, language in SOURCE_FILES:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            continue
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = text.splitlines(keepends=True)
        if len(lines) < CHUNK_MIN_LINES:
            continue
        for start, end in _chunk_lines(lines, CHUNK_MIN_LINES, CHUNK_MAX_LINES):
            content = "".join(lines[start:end]).rstrip()
            if len(content.strip()) < 50:
                continue
            snippets.append(CodeSnippet(
                file_path=rel_path,
                language=language,
                content=content,
                start_line=start + 1,
                end_line=end,
            ))
    return snippets


def get_snippet_pool(n: int = 20) -> list[CodeSnippet]:
    """Return a fixed-size pool of snippets (deterministic)."""
    all_snippets = load_snippets()
    # Take evenly spaced snippets to get diversity across files
    if len(all_snippets) <= n:
        return all_snippets
    step = len(all_snippets) / n
    return [all_snippets[int(i * step)] for i in range(n)]
