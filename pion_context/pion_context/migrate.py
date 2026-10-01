"""Migrate Claude Code memory files to Pion semantic cache.

Reads markdown memory files from ~/.claude/projects/*/memory/,
extracts content and metadata, and stores them in Pion using
the agent_remember pattern (HNSW-indexed semantic memory).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import redis as redis_lib

from .embeddings import embed_text, embed_dim

MEMORY_INDEX = "__agent_memory__"
MEMORY_PREFIX = "mem:"
MEMORY_SEQ_KEY = "__mem_seq__"


def parse_memory_file(path: str) -> Optional[dict]:
    """Parse a Claude Code memory markdown file.

    Expected format:
    ---
    name: memory name
    description: one-line description
    type: user|feedback|project|reference
    ---

    Memory content here.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    # Parse frontmatter
    frontmatter = {}
    body = content

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        body = fm_match.group(2).strip()

        for line in fm_text.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip()

    if not body:
        return None

    return {
        "name": frontmatter.get("name", os.path.basename(path)),
        "description": frontmatter.get("description", ""),
        "type": frontmatter.get("type", "project"),
        "body": body,
        "source_file": path,
    }


def migrate_memory_directory(
    memory_dir: str,
    host: str = "127.0.0.1",
    port: int = 1974,
    session_id: str = "migrated",
) -> dict:
    """Migrate all memory files in a directory to Pion.

    Returns stats dict with keys: files, stored, errors.
    """
    conn = redis_lib.Redis(host=host, port=port, decode_responses=False, socket_keepalive=True)

    # Ensure memory index exists
    dim = embed_dim()
    try:
        conn.execute_command("FT.INFO", MEMORY_INDEX)
    except Exception:
        conn.execute_command(
            "FT.CREATE", MEMORY_INDEX,
            "SCHEMA", "embedding", "VECTOR", "HNSW",
            "10",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "32",
        )

    stats = {"files": 0, "stored": 0, "errors": 0}
    memory_path = Path(memory_dir)

    if not memory_path.is_dir():
        print(f"Directory not found: {memory_dir}")
        return stats

    # Skip MEMORY.md (index file, not a memory)
    for md_file in sorted(memory_path.glob("*.md")):
        if md_file.name == "MEMORY.md":
            continue

        stats["files"] += 1
        parsed = parse_memory_file(str(md_file))
        if not parsed:
            stats["errors"] += 1
            continue

        # Embed the full text (name + description + body)
        text_to_embed = f"{parsed['name']}: {parsed['description']}\n{parsed['body']}"

        try:
            embedding = embed_text(text_to_embed[:2048])
        except Exception as e:
            print(f"  Embedding failed for {md_file.name}: {e}")
            stats["errors"] += 1
            continue

        # Store in Pion
        seq = conn.incr(MEMORY_SEQ_KEY)
        doc_key = f"{MEMORY_PREFIX}{seq}"

        conn.hset(doc_key, mapping={
            "text": parsed["body"].encode("utf-8"),
            "embedding": embedding,
            "session_id": session_id.encode("utf-8"),
            "timestamp": str(int(time.time())).encode("utf-8"),
            "memory_type": parsed["type"].encode("utf-8"),
            "memory_name": parsed["name"].encode("utf-8"),
            "source_file": parsed["source_file"].encode("utf-8"),
            "index": MEMORY_INDEX.encode("utf-8"),
        })
        stats["stored"] += 1
        print(f"  Migrated: {md_file.name} ({parsed['type']}: {parsed['name']})")

    # Optimize after migration
    if stats["stored"] > 0:
        try:
            conn.execute_command("FT.OPTIMIZE", MEMORY_INDEX)
            print("  Index optimized.")
        except Exception:
            pass

    return stats
