# Pion Codebase Search — Semantic Search over Your Repo for Claude Code

Replaces grep/glob with HNSW-powered semantic search. Every function, class, and module is embedded and indexed in Pion for sub-millisecond retrieval by meaning.

## Setup

```bash
# 1. Install the codebase search client
cd /path/to/Pion
pip install -e pion_context

# 2. Start Pion with vector support
./pion-server --profile vector -w 1

# 3. Index your codebase (one-time, ~30s for a 500-file project)
pion-context index --dir .

# 4. Start an embedding provider (pick one):
#    Option A: Ollama (default, free, local)
ollama pull nomic-embed-text
#    Option B: OpenAI (faster, requires API key)
export PION_EMBED_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

## Usage

### CLI

```bash
# Semantic code search
pion-context search "how does the hash map handle collisions"
pion-context search "authentication middleware" -k 5

# Full context retrieval (code + memories + cache)
pion-context context "race condition in queue processor"

# Index a single file (after editing)
pion-context index-file src/network/fast_path.mojo

# Migrate Claude Code memories to Pion
pion-context migrate

# Show index stats
pion-context stats
```

### MCP Tools (from Claude Code)

The MCP server exposes three new tools:

| Tool | Description |
|------|-------------|
| `codebase_index` | Index a directory tree into Pion |
| `codebase_search` | Semantic search over indexed code |
| `codebase_context` | Unified retrieval: code + memories + cache |

These work alongside the existing 25 MCP tools (`vector_search`, `agent_remember`, `semantic_cache_get`, etc.).

### Claude Code Hooks (automatic)

When `.claude/settings.json` is configured (done automatically):

- **PostToolUse (Edit|Write)**: After Claude edits a file, the hook re-indexes it in Pion automatically. Runs async — no delay on the conversation.
- **SessionStart**: On new sessions, queries Pion for project-level context and injects it into Claude's context window.

## Architecture

```
Claude Code Session
    │
    ├── SessionStart hook ──→ Pion FT.SEARCH ──→ inject relevant code context
    │
    ├── MCP: codebase_search("auth middleware") ──→ Pion HNSW ──→ results
    ├── MCP: agent_recall("past decisions") ──→ Pion memory index ──→ results
    ├── MCP: semantic_cache_get("similar question") ──→ cached answer
    │
    └── PostToolUse hook ──→ pion-context index-file ──→ re-index changed file

Pion Server (-w 1, --profile vector)
    ├── __codebase__        HNSW index of code chunks
    ├── __agent_memory__    HNSW index of conversation memories
    ├── __cb_checksums__    File checksums (incremental indexing)
    └── AI.SEMANTIC_CACHE   Cached Q&A pairs
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PION_HOST` | `127.0.0.1` | Pion server host |
| `PION_PORT` | `1974` | Pion server port |
| `PION_EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama`, `openai`, `mock` |
| `PION_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `PION_EMBED_DIM` | `768` | Embedding dimension |
| `PION_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `OPENAI_API_KEY` | (none) | Required for `openai` provider |

### Chunking Strategy

Files are split into semantic chunks:
- **Python/Mojo**: Split on `def`/`fn`/`class`/`struct` boundaries
- **JS/TS/Go/Rust/Java**: Split on function/class/struct boundaries
- **Fallback**: 80-line blocks with 10-line overlap

Max chunk size: 80 lines. Min: 5 lines. Files > 512KB are skipped.

### Incremental Indexing

Each file's SHA256 checksum is stored in Pion. On re-index:
1. Compute checksum of current file content
2. Compare with stored checksum
3. Skip if unchanged (unless `--force`)
4. On change: remove old chunks, re-embed, store new chunks
5. Auto-optimize HNSW index every 100 new chunks

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
