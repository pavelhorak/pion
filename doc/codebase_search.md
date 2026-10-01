# Pion Codebase Search

Indexes the codebase into Pion's HNSW vector index for semantic search. Finds `verify_totp()` when the query says "MFA" — not just keyword matches.

## Setup

```bash
pip install -e pion_context                 # install CLI + library
./pion-server -w 1 --no-auto-detect         # start Pion
ollama pull nomic-embed-text                # embedding model (768d, padded to 1536d)
pion-context index --dir src/ --force       # index codebase (~50s for 95 files, 1174 chunks)
```

## CLI

```bash
pion-context search "hash map collision probing" -k 5    # semantic code search
pion-context context "WAL persistence recovery"          # code + memories + cache
pion-context stats                                       # index info
pion-context migrate                                     # migrate Claude memory files to Pion
pion-context index-file src/network/fast_path.mojo       # re-index single file
```

## Claude Code Hooks

`pion_context` has two hook handlers for Claude Code: `hook-session` injects
project context at session start, and `hook-reindex` re-indexes a file after an
edit. Add them to your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{
      "type": "command",
      "command": "python3 -m pion_context.cli hook-session",
      "timeout": 15,
      "statusMessage": "Loading context from Pion..."
    }]}],
    "PostToolUse": [{ "matcher": "Edit|Write", "hooks": [{
      "type": "command",
      "command": "python3 -m pion_context.cli hook-reindex",
      "timeout": 30,
      "async": true
    }]}]
  }
}
```

| Handler | Event | Behavior |
|------|-------|----------|
| `hook-session` | SessionStart | Queries Pion HNSW for project-level context and injects it via `additionalContext`. |
| `hook-reindex` | PostToolUse (Edit\|Write) | Re-indexes the changed file, asynchronously. |

**The hooks need Pion running on port 1974 with an indexed codebase.** Wrap
them in a script that exits 0 when the port is closed if you want them to skip
silently while Pion is down.

To test: start Pion and index (`pion-context index --dir src/ --force`), then
restart Claude Code; you should see "Loading context from Pion...".

## MCP Tools

Three codebase-specific tools added to `mcp/pion_mcp/server.py` (alongside the existing 35 tools):

| Tool | Description |
|------|-------------|
| `codebase_index(directory, force)` | Index a directory tree into Pion HNSW |
| `codebase_search(query, k)` | Semantic search over indexed code chunks |
| `codebase_context(query, code_k, memory_k)` | Unified retrieval: code + agent memories + semantic cache |

## Architecture

```
Claude Code Session
    │
    ├── SessionStart hook ──→ Pion FT.SEARCH ──→ inject relevant code context
    │
    ├── MCP: codebase_search("auth middleware") ──→ Pion HNSW ──→ results
    ├── MCP: agent_recall("past decisions") ──→ Pion memory index ──→ results
    │
    └── PostToolUse hook ──→ pion-context index-file ──→ re-index changed file

Pion Server (-w 1)
    ├── __codebase__        HNSW index of code chunks (field: "vec", 1536d)
    ├── __agent_memory__    HNSW index of conversation memories (field: "embedding")
    ├── __cb_checksums__    File checksums for incremental indexing
    └── AI.SEMANTIC_CACHE   Cached Q&A pairs
```

## Chunking Strategy

Files are split into semantic chunks before embedding:

| Language | Strategy | Boundaries |
|----------|----------|------------|
| Python, Mojo | Semantic | `def`, `fn`, `class`, `struct` |
| JS/TS, Go, Rust, Java, C/C++ | Semantic | `function`, `class`, `struct`, `impl`, `fn`, `func` |
| All others | Fixed-size | 80-line blocks with 10-line overlap |

- Max chunk: 80 lines. Min: 5 lines.
- Files > 512KB skipped.
- Incremental: SHA256 checksum per file, skip unchanged (unless `--force`).

## Key Implementation Details

### Single FT.OPTIMIZE rule

Pion frees the shared ingest buffer after FT.OPTIMIZE. All HSET inserts must complete **before** calling FT.OPTIMIZE once. Subsequent HSETs after optimize silently skip vector routing (vectors stored as hash fields but not HNSW-indexed).

For incremental inserts after optimize, use FT.DROPINDEX + FT.CREATE + re-insert + FT.OPTIMIZE.

### Embedding dimension padding

- Ollama `nomic-embed-text` outputs 768 dimensions
- Pion's server-side default `Vector Dim` is 1536 (`--dim`); `FT.CREATE … DIM <d>` is honored per index
- `pion_context` targets the 1536-dim default by padding 768d → 1536d with zeros and normalizing to unit norm (it could instead create a 768-dim index; padding keeps one server config)
- OpenAI `text-embedding-3-small` outputs 1536d natively (no padding needed)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PION_HOST` | `127.0.0.1` | Pion server host |
| `PION_PORT` | `1974` | Pion server port |
| `PION_EMBED_PROVIDER` | `ollama` | Provider: `ollama`, `openai`, `mock` |
| `PION_EMBED_MODEL` | `nomic-embed-text` | Model name |
| `PION_EMBED_DIM` | `1536` | Output dimension (after padding) |
| `PION_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `OPENAI_API_KEY` | (none) | Required for `openai` provider |

## Files

```
pion_context/
  pyproject.toml       # pip install -e pion_context
  README.md            # setup + usage guide
  pion_context/        # the package
    __init__.py
    indexer.py         # codebase walker, semantic chunker, HNSW storage
    engine.py          # unified retrieval (code + memories + cache)
    cli.py             # CLI: index, search, context, migrate, stats, hook handlers
    migrate.py         # convert Claude memory files to Pion semantic cache
    embeddings.py      # Ollama/OpenAI/mock providers, padding, normalization

mcp/pion_mcp/server.py # 3 new tools: codebase_index, codebase_search, codebase_context
```

## Verified Results

Indexed 95 source files (1,174 chunks) in 51 seconds with Ollama nomic-embed-text.

| Query | Top Result | Correct? |
|-------|-----------|----------|
| "hash map collision probing" | `hash_map.mojo:StripedHashMap` | Yes |
| "HNSW beam search algorithm" | `hnsw.mojo:_beam_search_1536_turbo3bit` | Yes |
| "WAL persistence recovery" | `wal.mojo:recover` | Yes |
| "TCP connection kqueue" | `replication.mojo:PrimaryReplicator`, `xdp.mojo:TCPConnection` | Yes |
| "fast path dispatch commands" | `fast_path.mojo:process_data_plane` | Yes |
