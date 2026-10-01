#!/usr/bin/env python3
"""Pion codebase search CLI.

Commands:
    pion-context index [--dir DIR] [--force]     Index a codebase
    pion-context index-file FILE                  Index a single file
    pion-context search QUERY [-k N]              Semantic code search
    pion-context context QUERY                    Full context retrieval
    pion-context migrate [--memory-dir DIR]       Migrate Claude memory files
    pion-context stats                            Show index statistics
    pion-context hook-reindex                     Hook: re-index file (reads stdin JSON)
    pion-context hook-session                     Hook: session context (reads stdin JSON)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def cmd_index(args):
    """Index a codebase directory."""
    from .indexer import CodebaseIndexer
    indexer = CodebaseIndexer(host=args.host, port=args.port)

    root = args.dir or os.getcwd()
    print(f"Indexing {root}...")
    stats = indexer.index_directory(root, force=args.force)

    print(f"\nIndexing complete:")
    print(f"  Files scanned:  {stats.files_scanned}")
    print(f"  Files indexed:  {stats.files_indexed}")
    print(f"  Files skipped:  {stats.files_skipped} (unchanged)")
    print(f"  Chunks created: {stats.chunks_created}")
    print(f"  Time:           {stats.elapsed_s:.1f}s")
    if stats.errors:
        print(f"  Errors:         {len(stats.errors)}")
        for e in stats.errors[:5]:
            print(f"    {e}")


def cmd_index_file(args):
    """Index a single file."""
    from .indexer import CodebaseIndexer
    indexer = CodebaseIndexer(host=args.host, port=args.port)

    n = indexer.index_file(args.file, force=True)
    if n > 0:
        indexer.optimize()
        print(f"Indexed {args.file}: {n} chunks")
    else:
        print(f"Skipped {args.file} (empty or too large)")


def cmd_search(args):
    """Semantic code search."""
    from .engine import ContextEngine
    engine = ContextEngine(host=args.host, port=args.port)

    results = engine.search_codebase(args.query, k=args.k)
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results):
        loc = f"{r.file_path}:{r.start_line}" if r.file_path else "?"
        print(f"\n--- [{i+1}] {loc} ({r.name}) ---")
        # Show first 10 lines of content
        lines = r.content.split("\n")[:10]
        print("\n".join(lines))
        if len(r.content.split("\n")) > 10:
            print(f"  ... ({len(r.content.split(chr(10)))} lines total)")


def cmd_context(args):
    """Full context retrieval (code + memory + cache)."""
    from .engine import ContextEngine
    engine = ContextEngine(host=args.host, port=args.port)

    results = engine.retrieve_context(args.query, code_k=args.k)
    formatted = engine.format_context(results)

    if formatted:
        print(formatted)
    else:
        print("No context found.")


def cmd_stats(args):
    """Show index statistics."""
    from .indexer import CodebaseIndexer
    indexer = CodebaseIndexer(host=args.host, port=args.port)

    stats = indexer.stats()
    if stats.get("status") == "no index":
        print("No codebase index found. Run: pion-context index")
        return

    print("Codebase Index Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_migrate(args):
    """Migrate Claude Code memory files to Pion semantic cache."""
    from .migrate import migrate_memory_directory

    memory_dir = args.memory_dir
    if not memory_dir:
        # Auto-detect: look for the current project's memory dir
        cwd = os.getcwd()
        # Claude stores memories at ~/.claude/projects/-path-to-project/memory/
        safe_path = cwd.replace("/", "-")
        candidate = os.path.expanduser(f"~/.claude/projects/{safe_path}/memory")
        if os.path.isdir(candidate):
            memory_dir = candidate
        else:
            print("Could not auto-detect memory directory.")
            print("Specify with: pion-context migrate --memory-dir PATH")
            return

    stats = migrate_memory_directory(memory_dir, host=args.host, port=args.port)
    print(f"\nMigration complete:")
    print(f"  Files processed: {stats['files']}")
    print(f"  Memories stored: {stats['stored']}")
    print(f"  Errors:          {stats['errors']}")


def cmd_hook_reindex(args):
    """Hook handler: re-index a file after Edit/Write.

    Reads PostToolUse JSON from stdin, extracts file_path, re-indexes it.
    Outputs JSON for Claude Code hook system.
    """
    from .indexer import CodebaseIndexer

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path or not os.path.isfile(file_path):
        sys.exit(0)

    # Check if the file extension is indexable
    ext = os.path.splitext(file_path)[1].lower()
    from .indexer import CODE_EXTENSIONS
    if ext not in CODE_EXTENSIONS:
        sys.exit(0)

    try:
        indexer = CodebaseIndexer(host=args.host, port=args.port)
        n = indexer.index_file(file_path, force=True)
        if n > 0:
            indexer.optimize()
    except Exception:
        pass  # non-blocking: don't fail the tool call

    sys.exit(0)


def cmd_hook_session(args):
    """Hook handler: inject relevant context at session start.

    Reads SessionStart JSON from stdin, retrieves project-level context
    from Pion, outputs additionalContext for Claude Code.
    """
    from .engine import ContextEngine

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    cwd = input_data.get("cwd", os.getcwd())
    project_name = os.path.basename(cwd)

    try:
        engine = ContextEngine(host=args.host, port=args.port)

        # Search for project-level context
        results = engine.retrieve_context(
            f"project overview architecture {project_name}",
            code_k=5,
            memory_k=3,
            check_cache=False,
        )

        if results:
            context = engine.format_context(results, max_chars=4000)
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": (
                        f"[Pion codebase search] Retrieved {len(results)} relevant items "
                        f"from the codebase index:\n\n{context}"
                    ),
                }
            }
            json.dump(output, sys.stdout)
    except Exception:
        pass  # non-blocking

    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        prog="pion-context",
        description="Pion semantic codebase search for Claude Code",
    )
    parser.add_argument("--host", default=os.environ.get("PION_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PION_PORT", "1974")))

    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index a codebase directory")
    p_index.add_argument("--dir", "-d", help="Root directory (default: cwd)")
    p_index.add_argument("--force", "-f", action="store_true", help="Re-index all files")
    p_index.set_defaults(func=cmd_index)

    # index-file
    p_file = sub.add_parser("index-file", help="Index a single file")
    p_file.add_argument("file", help="File path to index")
    p_file.set_defaults(func=cmd_index_file)

    # search
    p_search = sub.add_parser("search", help="Semantic code search")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")
    p_search.set_defaults(func=cmd_search)

    # context
    p_ctx = sub.add_parser("context", help="Full context retrieval")
    p_ctx.add_argument("query", help="Natural language query")
    p_ctx.add_argument("-k", type=int, default=8, help="Number of code results")
    p_ctx.set_defaults(func=cmd_context)

    # stats
    p_stats = sub.add_parser("stats", help="Show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    # migrate
    p_migrate = sub.add_parser("migrate", help="Migrate Claude memory files to Pion")
    p_migrate.add_argument("--memory-dir", help="Path to Claude memory directory")
    p_migrate.set_defaults(func=cmd_migrate)

    # hook-reindex (called by PostToolUse hook)
    p_hook_ri = sub.add_parser("hook-reindex", help="Hook: re-index after Edit/Write")
    p_hook_ri.set_defaults(func=cmd_hook_reindex)

    # hook-session (called by SessionStart hook)
    p_hook_ss = sub.add_parser("hook-session", help="Hook: session start context")
    p_hook_ss.set_defaults(func=cmd_hook_session)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
