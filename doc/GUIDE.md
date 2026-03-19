# vecgrep User Guide

See also: [Configuration](CONFIG.md) | [Server API](SERVER.md)

## How it works

1. **Walk** — discovers files using the same engine as ripgrep (`.gitignore`-aware, binary detection)
2. **Chunk** — splits files into overlapping token-window chunks, snapped to line boundaries
3. **Embed** — runs each chunk through the embedding model (built-in or external) to produce a vector
4. **Index** — caches embeddings in a local SQLite database (`.vecgrep/index.db`), keyed by BLAKE3 content hash so only changed files are re-embedded
5. **Search** — cosine similarity via [sqlite-vec](https://github.com/asg017/sqlite-vec), fast enough for every-keystroke use

## Searching

```bash
vecgrep "error handling for network timeouts" ./src
vecgrep "match result { Ok(v) => v, Err(e) => return Err(e) }" ./src
vecgrep "sorting algorithm" --type rust
vecgrep -i "authentication"                     # interactive TUI
```

### Filtering results

```bash
vecgrep -l "retry logic" ./src                  # file paths only
vecgrep -c "error handling" ./src               # count per file
vecgrep --json "retry logic" | jq '.score'      # JSONL for scripting
vecgrep --json "error handling" ./src | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'
```

### Combining with other tools

```bash
# Semantic search to find files, then exact match
vecgrep -l "error handling" ./src | xargs rg "unwrap"

# Ripgrep to narrow files, vecgrep to rank by meaning
rg -l "TODO" ./src | xargs vecgrep "technical debt that should be refactored"

# Interactive TUI with xargs
rg -l "TODO" ./src | xargs vecgrep -i --query "technical debt"

# Git integration
vecgrep --json "authentication" ./src | jq -r '.file' | sort -u | xargs git blame
vecgrep -l "database queries" ./src | xargs git log --oneline -5 --

# Pretty-print with bat
vecgrep -l "configuration parsing" ./src | xargs bat --style=header,grid

# Watch mode
vecgrep -l "error handling" ./src | entr -r cargo test
```

## Path scoping

Results are scoped to the paths you specify, like ripgrep:

```bash
vecgrep "query" src/                            # only src/ results
cd src && vecgrep "query"                       # same — scoped to cwd
vecgrep --no-scope "query"                      # search entire project
```

- One invocation, one project root, one cache. Paths outside the root are rejected by default (`--skip-outside-root` to ignore them).
- Single directory: walks recursively, cleans up stale files in that subtree.
- Multiple directories: walks all, updates shared cache, skips stale cleanup.
- Explicit file paths: cached with an `explicit` flag for fast re-search, excluded from directory-only searches. Cleared when a directory walk rediscovers the file.

## Ignoring files

vecgrep respects `.gitignore` by default. For additional patterns:

```bash
vecgrep --ignore-file .vecgrep/ignore "query"
```

Or set it once in `.vecgrep/config.toml`:

```toml
ignore_files = [".vecgrep/ignore"]
```

Supports the full gitignore pattern language — globs, directory patterns, and negation (`!keep-this.log`). The flag can be specified multiple times.

## Index management

```bash
vecgrep --stats                                 # files, chunks, holes, DB size
vecgrep --reindex ./src                         # force full re-index
vecgrep --clear-cache                           # delete cached index
vecgrep --index-only ./src                      # build index without searching
vecgrep --show-root                             # print resolved project root
```

The index is a local cache. It rebuilds automatically when the schema version or embedding model changes. `Holes` are chunks whose remote embedding failed — they exist in the cache but never match queries.

## Embedding models

### Built-in: all-MiniLM-L6-v2

Ships inside the binary — no setup needed. 22M parameters, 384 dimensions, single-digit millisecond inference on CPU. Best score separation on our [benchmark](BENCHMARK.md).

### External models

For large codebases (1,000+ files), larger models improve accuracy:

```bash
vecgrep --embedder-url http://localhost:11434/v1/embeddings --embedder-model mxbai-embed-large "query"
```

Works with [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible API. Set it once in [config](CONFIG.md) to avoid repeating flags. The index rebuilds automatically when the model changes.
