# Pion Codebase Search (pion-context)

**Status: experimental.** Indexes a repository's source into Pion for semantic code
search: functions, classes and blocks are embedded and searched by meaning. Read the
[known limitations](#known-limitations) before relying on it. No benefit to coding
agents has been measured.

## Setup

```bash
# 1. Install the codebase search client
cd /path/to/Pion
pip install -e pion_context

# 2. Start Pion with vector support
./pion-server --profile vector -w 1

# 3. Start an embedding provider (pick one):
#    Option A: Ollama (default, free, local)
ollama pull nomic-embed-text
#    Option B: OpenAI (requires an API key)
export PION_EMBED_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# 4. Index your codebase into that fresh server
pion-context index --dir .
```

Indexing makes one embedding request per chunk, so its time is set by the embedding
provider.

## Usage

### CLI

```bash
# Semantic code search
pion-context search "how does the hash map handle collisions"
pion-context search "authentication middleware" -k 5

# Code plus agent memories and the semantic cache (see the limitations)
pion-context context "race condition in queue processor"

# Migrate Claude Code memories to Pion
pion-context migrate

# Show index stats
pion-context stats
```

### MCP tools

pion-mcp (`mcp/`) exposes three tools over this package: `codebase_index`,
`codebase_search` and `codebase_context` (code, agent memories and the semantic cache).

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PION_HOST` | `127.0.0.1` | Pion server host |
| `PION_PORT` | `1974` | Pion server port |
| `PION_EMBED_PROVIDER` | `ollama` | Embedding provider: `ollama`, `openai`, `mock` |
| `PION_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `PION_EMBED_DIM` | `1536` | Index dimension; Ollama's 768-d vectors are zero-padded to it |
| `PION_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `OPENAI_API_KEY` | (none) | Required for `openai` provider |

### Chunking Strategy

Files are split into semantic chunks:
- **Python/Mojo**: Split on `def`/`fn`/`class`/`struct` boundaries
- **JS/TS/Go/Rust/Java**: Split on function/class/struct boundaries
- **Fallback**: 80-line blocks with 10-line overlap

Max chunk size: 80 lines. Min: 5 lines. Files over 512 KB are skipped, and so are some
directories (see the limitations).

### Checksums

Each file's SHA256 checksum is stored in Pion, and `pion-context index` skips files whose
checksum has not changed unless `--force` is given.

## Known limitations

- **Directories are skipped by name, anywhere in the tree.** `SKIP_DIRS` in `indexer.py`
  drops `.git`, `node_modules`, `build`, `dist`, `target`, `venv`, `dataset`, `models` and
  a few more wherever they occur, including source directories: Django's ORM,
  `django/db/models/`, is not indexed.
- **Re-indexing into a built index removes files from search.** Vectors written after
  `FT.OPTIMIZE` are not added to the HNSW graph. `pion-context index-file`, and the
  `hook-reindex` handler meant for a Claude Code PostToolUse hook, delete the file's old
  chunks and store new ones that no search returns. Index into a fresh server instead.
- **One FT index per server.** Pion serves one FT index at a time. pion-mcp's agent memory
  and semantic cache tools build their own indexes, and the last one built replaces
  `__codebase__`, so `context` and `codebase_context` cannot return code and memories from
  one server.
- **Embedding input is cut at 2,048 characters**, and `nomic-embed-text` is called without
  the `search_query:` / `search_document:` prefixes it was trained with.

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
