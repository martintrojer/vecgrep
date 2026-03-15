# vecgrep

Semantic grep — like [ripgrep](https://github.com/BurntSushi/ripgrep), but with vector search.

Search your codebase, notes, or Obsidian vault by meaning, not just text. Ask for "error handling for network timeouts" and find the relevant code, even if it doesn't contain those exact words.

**Local-first.** An embedding model ships inside the binary — no external services, no API keys, no GPU required. Your code never leaves your machine.

**Fast by default.** CLI searches wait for indexing to finish, so first-run results include newly discovered files. Interactive mode (`-i`) and the HTTP server (`--serve`) update results progressively as new files are indexed.

**Bring your own model.** Optionally connect to [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible embeddings API for access to larger models. See [BENCHMARK.md](BENCHMARK.md) for model comparisons.

## Usage

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

## More examples

```bash
# HTTP server mode (load model once, query via curl)
vecgrep --serve --port 8080 ./src
# => Listening on http://127.0.0.1:8080
curl -s "http://localhost:8080/search?q=error+handling&k=5"

# Use with fzf for interactive fuzzy semantic search
vecgrep --serve --port 8080 ./src &
fzf --bind "change:reload:curl -s 'http://localhost:8080/search?q={q}'" --preview 'echo {}'

# Security audit — find input handling code, then grep for dangerous patterns
vecgrep -l "parsing user input" ./src | xargs rg "eval|exec|unsafe"

# Find files about a concept and open them in your editor
vecgrep -l "authentication and session management" ./src | xargs $EDITOR

# Count how many chunks in each file relate to error handling
vecgrep -c "error handling" ./src

# Filter high-confidence results and format as file:line
vecgrep --json "error handling" ./src | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'

# Find who wrote security-related code
vecgrep --json "authentication" ./src | jq -r '.file' | sort -u | xargs git blame

# Recent changes to files about database access
vecgrep -l "database queries" ./src | xargs git log --oneline -5 --

# Pretty-print matching files with bat
vecgrep -l "configuration parsing" ./src | xargs bat --style=header,grid

# Generate a markdown TODO list from semantic matches
vecgrep --json "TODO" ./src | jq -r '"- [ ] \(.file):\(.start_line) — \(.text | split("\n") | first)"'

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
- Single directory path: vecgrep walks that subtree recursively and performs stale cleanup for that subtree.
- Multiple directory paths: vecgrep walks all of them and updates the shared cache, but skips stale cleanup because the input is not one contiguous subtree.
- Explicit file paths: vecgrep indexes only those files and skips stale cleanup. This is the right mode for commands like `rg -l ... | xargs vecgrep ...`.
- `--skip-outside-root`: ignore outside-root paths instead of failing. Skipped paths are not indexed and cannot appear in results.
- No path given: equivalent to `.`.

## Embedding models

### Built-in: all-MiniLM-L6-v2

The binary ships with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a 22M-parameter model that produces 384-dimensional embeddings. It runs in single-digit milliseconds on CPU, indexes thousands of files in seconds, and has the best score separation on our [benchmark](BENCHMARK.md) — meaning `--threshold` works reliably.

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

The index automatically rebuilds when the model changes. See [BENCHMARK.md](BENCHMARK.md) for model comparisons.

## Index behavior

`vecgrep --stats` reports file count, chunk count, database size, and `Holes` — chunks whose embedding failed and were stored as zero vectors. Holes can never match a query and are mainly relevant when using remote embedders.

The index database is a local cache. vecgrep automatically rebuilds it when the schema version changes, rather than trying to migrate older cache files in place.

## Install

Pre-built binaries for macOS and Linux are available on the [releases page](https://github.com/mtrojer/vecgrep/releases). Download the appropriate archive, extract it, and place the `vecgrep` binary on your `PATH`.

Install with mise:

```bash
mise use github:martintrojer/vecgrep
```

Install with cargo:

```bash
cargo install vecgrep
```

To build from source:

```bash
cargo install --path .
```

The first build downloads the ONNX model (~90 MB) from HuggingFace and caches it locally. Subsequent builds reuse the cached model.

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
context = 5

# File discovery
hidden = true
ignore_files = [".vecgrepignore"]
```

Project-level config is useful for per-repo settings (e.g., a different model or chunk size). Global config sets your personal defaults.

## Options

```
vecgrep [OPTIONS] <QUERY> [PATHS]...

Arguments:
  <QUERY>     Search query (natural language or code snippet)
  [PATHS]...  Files or directories to search [default: .]
              Like ripgrep, you can pass multiple paths. Directories
              are walked recursively, respecting .gitignore. Files
              are searched directly. All paths share one cache at the
              discovered project root (.git/, .vecgrep/, etc.). Stale
              cleanup only runs for single-directory walks, not for
              explicit file lists or multi-path mixes. Paths outside
              the selected root fail by default.

Options:
  -k, --top-k <N>              Number of results [default: 10]
      --threshold <F>           Minimum similarity 0.0–1.0 [default: 0.3]
  -i, --interactive             Interactive TUI mode
  -t, --type <TYPE>             Filter by file type (rust, python, js, ...)
  -T, --type-not <TYPE>         Exclude file type
  -g, --glob <PATTERN>          Filter by glob
  -C, --context <N>             Context lines around match [default: 3]
  -l, --files-with-matches      Print only file paths with matches
  -c, --count                   Print count of matching chunks per file
  -., --hidden                  Search hidden files and directories
  -L, --follow                  Follow symbolic links
  -d, --max-depth <N>           Limit directory traversal depth
      --ignore-file <PATH>      Additional ignore file (gitignore syntax, repeatable)
      --no-ignore               Don't respect .gitignore
      --type-list               Show all supported file types
      --color <WHEN>            When to use color (auto, always, never)
      --embedder-url <URL>      OpenAI-compatible embeddings API URL
      --embedder-model <NAME>   Model name for --embedder-url
      --reindex                 Force full re-index
      --full-index              Wait for indexing to complete before starting interactive/server mode
      --index-only              Build index without searching
      --stats                   Show index statistics
      --clear-cache             Delete cached index
      --show-root               Print resolved project root and exit
      --skip-outside-root       Ignore paths outside the selected project root
      --json                    JSONL output (includes "root" field)
      --serve                   Start HTTP server mode
      --port <PORT>             Port for HTTP server [default: auto]
      --chunk-size <N>          Tokens per chunk [default: 500]
      --chunk-overlap <N>       Overlap tokens [default: 100]
```

## Integrations

- [vecgrep.nvim](https://github.com/martintrojer/vecgrep.nvim) — Neovim plugin for semantic search via vecgrep's `--serve` mode

## Environment variables

- `VECGREP_MODEL_CACHE` — override model cache directory (default: system cache dir)
- `VECGREP_LOG` — enable debug logging, e.g. `VECGREP_LOG=debug`

## License

MIT
