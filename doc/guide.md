# vecgrep User Guide

Detailed documentation for [vecgrep](../README.md) — semantic grep with vector search.

## Usage examples

```bash
# Search for a concept
vecgrep "error handling for network timeouts" ./src

# Use a code snippet as query to find similar patterns
vecgrep "match result { Ok(v) => v, Err(e) => return Err(e) }" ./src

# Filter by file type
vecgrep "sorting algorithm" --type rust

# Interactive TUI mode
vecgrep -i "authentication"

# Combining with ripgrep — semantic search to find files, then exact match
vecgrep -l "error handling" ./src | xargs rg "unwrap"

# Reverse — ripgrep to narrow files, vecgrep to rank by meaning
rg -l "TODO" ./src | xargs vecgrep "technical debt that should be refactored"

# Interactive TUI with xargs — files as paths, query typed in TUI or via --query
rg -l "TODO" ./src | xargs vecgrep -i --query "technical debt"

# JSON output for scripting
vecgrep "retry logic" --json | jq '.score'

# Use an external embedding model via Ollama
vecgrep --embedder-url http://localhost:11434/v1/embeddings --embedder-model mxbai-embed-large "query" ./src

# Index management
vecgrep --stats              # show index statistics, including index holes
vecgrep --reindex ./src      # force full re-index
vecgrep --clear-cache        # delete cached index
vecgrep --index-only ./src   # build index without searching
vecgrep --show-root          # print resolved project root
```

### Server and scripting

```bash
# HTTP server mode (load model once, query via curl)
vecgrep --serve --port 8080 ./src
# => Listening on http://127.0.0.1:8080
curl -s "http://localhost:8080/search?q=error+handling&k=5"

# Check indexing status (useful for IDE plugins)
curl -s "http://localhost:8080/status"
# => {"status":"indexing","indexed":42,"total":380,"chunks":85,"version":"0.9.1","root":"/path/to/project"}
# => {"status":"ready","files":380,"chunks":850,"version":"0.9.1","root":"/path/to/project"}

# Use with fzf for interactive fuzzy semantic search
vecgrep --serve --port 8080 ./src &
fzf --bind "change:reload:curl -s 'http://localhost:8080/search?q={q}'" --preview 'echo {}'

# Filter high-confidence results and format as file:line
vecgrep --json "error handling" ./src | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'

# Find who wrote security-related code
vecgrep --json "authentication" ./src | jq -r '.file' | sort -u | xargs git blame

# Recent changes to files about database access
vecgrep -l "database queries" ./src | xargs git log --oneline -5 --

# Pretty-print matching files with bat
vecgrep -l "configuration parsing" ./src | xargs bat --style=header,grid

# Re-run tests when error-handling code changes
vecgrep -l "error handling" ./src | entr -r cargo test
```

## Ignoring files

vecgrep respects `.gitignore` by default. For additional project-specific ignore patterns, use `--ignore-file` with a file containing [gitignore syntax](https://git-scm.com/docs/gitignore):

```bash
# Create an ignore file in the .vecgrep/ directory (already gitignored)
cat > .vecgrep/ignore <<'EOF'
*.org
*.org_archive
*.min.js
vendor/
EOF

# Use it from the CLI
vecgrep --ignore-file .vecgrep/ignore "query"

# Or set it once in .vecgrep/config.toml
cat >> .vecgrep/config.toml <<'EOF'
ignore_files = [".vecgrep/ignore"]
EOF
```

The flag can be specified multiple times and supports the full gitignore pattern language — globs, directory patterns, and negation (`!keep-this.log`).

## How it works

1. **Walk** — discovers files using the same engine as ripgrep (`.gitignore`-aware, binary detection)
2. **Chunk** — splits files into overlapping token-window chunks, snapped to line boundaries
3. **Embed** — runs each chunk through the embedding model (built-in or external) to produce a vector
4. **Index** — caches embeddings in a local SQLite database (`.vecgrep/index.db`), keyed by BLAKE3 content hash so only changed files are re-embedded
5. **Search** — cosine similarity between query and all cached embeddings, returned as top-k results

Search is a vector KNN query via [sqlite-vec](https://github.com/asg017/sqlite-vec) — fast enough for on-every-keystroke use in interactive mode and the HTTP server.

## Path semantics

`vecgrep` accepts files, directories, or a mix of both. The project root is discovered once and the cache lives at that root, so different path selections still share the same index.

- One invocation uses one selected project root and one cache. Paths outside that root are rejected by default.
- **Results are scoped to requested paths**: `vecgrep "query" src/` only returns results from `src/`, not the entire index. Running from a subdirectory without explicit paths scopes results to that subdirectory — `cd src && vecgrep "query"` only shows `src/` results. Same behavior as ripgrep.
- `--no-scope` overrides path scoping and searches the entire project index.
- Single directory path: vecgrep walks that subtree recursively and performs stale cleanup for that subtree.
- Multiple directory paths: vecgrep walks all of them and updates the shared cache, but skips stale cleanup because the input is not one contiguous subtree.
- Explicit file paths: vecgrep indexes them with an `explicit` flag. They stay cached for fast re-search but are excluded from directory-only searches. Only the specific explicit files you pass appear in results — not all explicit files from prior invocations. When a directory walk rediscovers the file, the flag is cleared. Consistent across CLI, TUI, and `--serve`.
- `--skip-outside-root`: ignore outside-root paths instead of failing. Skipped paths are not indexed and cannot appear in results.
- No path given: equivalent to `.`.

## Embedding models

### Built-in: all-MiniLM-L6-v2

The binary ships with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a 22M-parameter model that produces 384-dimensional embeddings. It runs in single-digit milliseconds on CPU, indexes thousands of files in seconds, and has the best score separation on our [benchmark](../BENCHMARK.md) — meaning `--threshold` works reliably.

### External: Ollama / LM Studio / any OpenAI-compatible API

For large codebases (1,000+ files), larger models improve retrieval accuracy. Use `--embedder-url` and `--embedder-model` to connect to a local embedding server:

```bash
# Ollama
vecgrep --embedder-url http://localhost:11434/v1/embeddings --embedder-model mxbai-embed-large "query"

# LM Studio
vecgrep --embedder-url http://localhost:1234/v1/embeddings --embedder-model mxbai-embed-large "query"
```

Or set it once in `~/.config/vecgrep/config.toml`:

```toml
embedder_url = "http://localhost:11434/v1/embeddings"
embedder_model = "mxbai-embed-large"
```

The index automatically rebuilds when the model changes. See [BENCHMARK.md](../BENCHMARK.md) for model comparisons.

## Index behavior

`vecgrep --stats` reports file count, chunk count, database size, and `Holes` — chunks whose embedding failed and were stored as zero vectors. Holes can never match a query and are mainly relevant when using remote embedders.

The index database is a local cache. vecgrep automatically rebuilds it when the schema version changes, rather than trying to migrate older cache files in place.

## Configuration

Default values for CLI flags can be set in TOML config files. Two locations are checked, with this precedence:

1. **CLI flags** — always win
2. **Project config** — `.vecgrep/config.toml` in the project root
3. **Global config** — `~/.config/vecgrep/config.toml`

```toml
# External embedder (e.g., Ollama)
embedder_url = "http://localhost:11434/v1/embeddings"
embedder_model = "mxbai-embed-large"

# Search defaults
top_k = 20
threshold = 0.25

# File discovery
hidden = true
ignore_files = [".vecgrepignore"]
file_type = ["rust", "python"]
# file_type_not = ["markdown"]
# glob = ["!*.generated.*"]
# max_depth = 5

# TUI file opener — {file}, {line}, {end_line} placeholders
# open_cmd = "less -N +{line}G {file}"
# open_cmd = "nvim +{line} {file}"
# open_cmd = "vim +{line} {file}"
# open_cmd = "emacs +{line} {file}"
# open_cmd = "code -g {file}:{line}"
# open_cmd = "bat -n --highlight-line {line}:{end_line} --paging always --pager 'less +{line}G' {file}"

# Server
# port = 8080

# Behavior
# skip_outside_root = true
```

Project-level config is useful for per-repo settings (e.g., a different model or chunk size). Global config sets your personal defaults.

## Options reference

```
vecgrep [OPTIONS] <QUERY> [PATHS]...

Arguments:
  <QUERY>     Search query (natural language or code snippet)
  [PATHS]...  Files or directories to search [default: .]

Options:
  -k, --top-k <N>              Number of results [default: 10]
      --threshold <F>           Minimum similarity 0.0-1.0 [default: 0.2]
  -i, --interactive             Interactive TUI mode
  -t, --type <TYPE>             Filter by file type (rust, python, js, ...)
  -T, --type-not <TYPE>         Exclude file type
  -g, --glob <PATTERN>          Filter by glob
  -l, --files-with-matches      Print only file paths with matches
  -c, --count                   Print count of matching chunks per file
  -., --hidden                  Search hidden files and directories
  -L, --follow                  Follow symbolic links
  -d, --max-depth <N>           Limit directory traversal depth
      --ignore-file <PATH>      Additional ignore file (gitignore syntax, repeatable)
      --no-ignore               Don't respect .gitignore
      --type-list               Show all supported file types
      --color <WHEN>            When to use color (auto, always, never)
  -p, --pretty                  Alias for --color=always (force colors when piping)
      --embedder-url <URL>      OpenAI-compatible embeddings API URL
      --embedder-model <NAME>   Model name for --embedder-url
      --reindex                 Force full re-index
      --full-index              Wait for indexing to complete before starting interactive/server mode
      --index-only              Build index without searching
      --stats                   Show index statistics
      --clear-cache             Delete cached index
      --show-root               Print resolved project root and exit
      --skip-outside-root       Ignore paths outside the selected project root
      --no-scope                Search entire project index (ignore cwd scoping)
      --open-cmd <CMD>          Command to open files from TUI ({file}, {line}, {end_line})
      --json                    JSONL output (includes "root" field)
      --serve                   Start HTTP server mode
      --port <PORT>             Port for HTTP server [default: auto]
      --chunk-size <N>          Tokens per chunk [default: 256]
      --chunk-overlap <N>       Overlap tokens [default: 64]
```

## Server endpoints

When running with `--serve`, the HTTP server exposes:

| Endpoint | Description |
|---|---|
| `GET /search?q=<query>&k=<N>&threshold=<F>` | Semantic search, returns JSONL |
| `GET /status` | Pipeline status as JSON |

The `/status` endpoint returns:
```json
{"status":"indexing","indexed":42,"total":380,"chunks":85,"version":"0.9.1","root":"/path/to/project"}
{"status":"ready","files":380,"chunks":850,"version":"0.9.1","root":"/path/to/project","scope":["src"]}
```

`total` is `null` while the file walker is still scanning. `version` is the vecgrep binary version. `root` is the project root path. `scope` lists active path scopes (omitted when searching the full project). IDE plugins can poll this to show indexing progress or wait for readiness.

## Environment variables

- `VECGREP_MODEL_CACHE` — override model cache directory (default: system cache dir)
- `VECGREP_LOG` — enable debug logging, e.g. `VECGREP_LOG=debug`
