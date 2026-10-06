# Pion Codebase Search (pion-context)

**Status: experimental.** Indexes a repository's source into Pion for semantic code
search: functions, classes and blocks are embedded and searched by meaning. Read the
[known limitations](#known-limitations) before relying on it. No benefit to coding
agents has been measured.

What it does, and the test that checks it on every push
(`tests/test_pion_context_index.py`):

- **Indexes what git tracks.** In a git work tree the files are git's: tracked, plus
  untracked files that are not ignored, so `.gitignore` decides. No source directory is
  dropped for its name.
- **Keeps a built index current.** Re-indexing a file replaces its chunks and rebuilds
  the index from the vectors already stored with each chunk; nothing is re-embedded
  except the changed file. Deleted and emptied files leave the index.
- **Reports a search that cannot run.** `search` exits 1 with the reason (the embedding
  provider failed, or another index replaced this one) instead of printing no results.

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

# 5. Optional: let Claude Code keep the index current
pion-context install-hooks
```

Indexing time is dominated by embedding the chunks, which go to Ollama 64 per
request; a search embeds one query and runs one `FT.SEARCH`. No timing is published
with raw output yet, so measure on your own repository.

## Usage

### CLI

```bash
# Semantic code search
pion-context search "how does the hash map handle collisions"
pion-context search "authentication middleware" -k 5

# Code plus agent memories and the semantic cache (see the limitations)
pion-context context "race condition in queue processor"

# Re-index one file after editing it
pion-context index-file src/network/fast_path.mojo

# Migrate Claude Code memories to Pion
pion-context migrate

# Show index stats
pion-context stats
```

### Claude Code hooks

`pion-context install-hooks [--project-dir DIR]` adds two hooks to the project's
`.claude/settings.json`, keeping the settings already there. Running it again replaces
them rather than adding copies. The hooks run the Python that ran `install-hooks`, so they
use the environment pion-context is installed in.

- **SessionStart** queries the index for project-level context and adds it to the session.
- **PostToolUse** (Edit, Write, MultiEdit) re-indexes the edited file. It runs
  asynchronously, so the edit does not wait for it.

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

### Re-indexing

Each file's SHA256 checksum is stored in Pion, and `pion-context index` skips files whose
checksum has not changed unless `--force` is given. Pion builds an index once (ingest,
`FT.OPTIMIZE`, search), and a vector written after `FT.OPTIMIZE` does not enter the built
graph. So a change to a built index ends with a rebuild: drop the index (the chunks stay),
create it again, send every chunk's stored vector again, optimize. A rebuild that would
lose chunks, because some have no stored vector, refuses before it drops anything.

## Known limitations

- **One FT index per server.** Pion serves one FT index at a time. pion-mcp's agent memory
  and semantic cache tools build their own indexes, and the last one built replaces
  `__codebase__`; a search then fails with an error that names the index that replaced it.
  `context` and `codebase_context` cannot return code and memories from one server: give
  each its own server.
- **Searches fail during a rebuild**, between dropping the index and optimizing it again.
- **Outside a git work tree**, a walk is used instead, skipping version-control, virtualenv,
  cache and `node_modules` directories.
- **Embedding input is cut at 2,048 characters**, and `nomic-embed-text` is called without
  the `search_query:` / `search_document:` prefixes it was trained with.

## License

Apache-2.0. Pion's satellites are deliberately permissive so they can be vendored
into any stack; the Pion **server** itself is Apache-2.0 too,
with one closed binary library for its tuned vector kernels — see the top-level
`LICENSE` and `doc/licensing.md`.
