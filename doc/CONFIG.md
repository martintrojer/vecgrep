# Configuration

See also: [User Guide](GUIDE.md) | [Server API](SERVER.md)

Default values for CLI flags can be set in TOML config files. Two locations are checked, with this precedence:

1. **CLI flags** — always win
2. **Project config** — `.vecgrep/config.toml` in the project root
3. **Global config** — `~/.config/vecgrep/config.toml`

Unknown fields are rejected — typos like `open-cmd` (should be `open_cmd`) produce a clear error message.

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

## Environment variables

- `VECGREP_MODEL_CACHE` — override model cache directory (default: system cache dir)
- `VECGREP_LOG` — enable debug logging, e.g. `VECGREP_LOG=debug`
