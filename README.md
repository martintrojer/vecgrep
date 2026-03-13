# vecgrep

Semantic grep — like [ripgrep](https://github.com/BurntSushi/ripgrep), but with vector search.

vecgrep uses a local embedding model ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) to search your codebase by meaning rather than exact text matches. The model is embedded directly in the binary — no external services, no API keys, fully offline.

**Fast by default.** After the first index build, searches return instantly — vecgrep queries the cached index without waiting for re-indexing. Changed files are indexed in the background for next time. Interactive mode (`-i`) and the HTTP server (`--serve`) feel real-time: queries take ~5ms, and results update progressively as new files are indexed.

## Usage

```bash
# Search for a concept
vecgrep "error handling for network timeouts" ./src

# Search with more results and a lower threshold
vecgrep "database connection pooling" ./src -k 20 -t 0.2

# Filter by file type
vecgrep "sorting algorithm" --type rust

# Interactive TUI mode
vecgrep -i "authentication"

# JSON output for scripting
vecgrep "retry logic" --json | jq '.score'

# HTTP server mode (load model once, query via curl)
# Omit --port to auto-pick a free port; the URL is printed to stderr
vecgrep --serve --port 8080 ./src
# => Listening on http://127.0.0.1:8080
curl -s "http://localhost:8080/search?q=error+handling&k=5"

# Use with fzf for interactive fuzzy semantic search
vecgrep --serve --port 8080 ./src &
fzf --bind "change:reload:curl -s 'http://localhost:8080/search?q={q}'" --preview 'echo {}'

# Index management
vecgrep --stats              # show index statistics
vecgrep --reindex ./src      # force full re-index
vecgrep --clear-cache        # delete cached index
vecgrep --index-only ./src   # build index without searching
vecgrep --show-root          # print resolved project root
```

## How it works

1. **Walk** — discovers files on a background thread using the same engine as ripgrep (`.gitignore`-aware, binary detection), streaming them through a bounded channel
2. **Chunk** — splits files into overlapping token-window chunks, snapped to line boundaries
3. **Embed** — runs each chunk through the ONNX model to produce a 384-dimensional vector
4. **Index** — caches embeddings in a local SQLite database (`.vecgrep/index.db`), keyed by BLAKE3 content hash so only changed files are re-embedded on subsequent runs
5. **Search** — computes cosine similarity between your query embedding and all cached chunk embeddings, returns top-k results

Walking and indexing overlap — the embedder processes files as the walker discovers them. Searches run against the cached index immediately; changed files are indexed in the background. Use `--full-index` to wait for indexing to complete before searching.

Search is a single matrix dot product against cached embeddings loaded in memory — no database in the hot path. This makes interactive mode and the HTTP server responsive enough for on-every-keystroke use.

## Why local-only?

vecgrep runs entirely on your machine. There are no API calls, no cloud services, no telemetry. Your code never leaves your computer.

This matters for:

- **Privacy** — proprietary codebases stay private
- **Speed** — no network round-trips; search is a local matrix multiply that takes <5ms
- **Availability** — works offline, on planes, behind firewalls, in air-gapped environments
- **Cost** — no API fees, no usage limits

## Model choice

vecgrep embeds [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) directly in the binary. This is a 22M-parameter sentence transformer that produces 384-dimensional embeddings.

Why this model:

- **Small and fast** — 90 MB (float32 ONNX), runs inference in single-digit milliseconds on CPU. No GPU required.
- **Good quality for its size** — consistently ranks near the top of [MTEB](https://huggingface.co/spaces/mteb/leaderboard) benchmarks among models under 100 MB. Handles both natural language and code well.
- **Standard BERT architecture** — wide ONNX Runtime support across platforms (x86, ARM, with optional CoreML/CUDA acceleration).
- **Battle-tested** — one of the most downloaded sentence-transformers models, with well-understood behaviour.

The model is downloaded once at build time from HuggingFace, cached locally, and compiled into the binary via `include_bytes!`. The resulting binary is fully self-contained.

## Install

Pre-built binaries for macOS and Linux are available on the [releases page](https://github.com/mtrojer/vecgrep/releases). Download the appropriate archive, extract it, and place the `vecgrep` binary on your `PATH`.

To build from source:

```bash
cargo install --path .
```

The first build downloads the ONNX model (~90 MB) from HuggingFace and caches it locally. Subsequent builds reuse the cached model.


## Options

```
vecgrep [OPTIONS] <QUERY> [PATHS]...

Arguments:
  <QUERY>     Search query (natural language or code snippet)
  [PATHS]...  Paths to search [default: .]

Options:
  -k, --top-k <N>              Number of results [default: 10]
      --threshold <F>           Minimum similarity 0.0–1.0 [default: 0.3]
  -i, --interactive             Interactive TUI mode
  -t, --type <TYPE>             Filter by file type (rust, python, js, ...)
  -T, --type-not <TYPE>         Exclude file type
  -g, --glob <PATTERN>          Filter by glob
  -C, --context <N>             Context lines around match [default: 3]
  -j, --threads <N>             Indexing threads
  -l, --files-with-matches      Print only file paths with matches
  -c, --count                   Print count of matching chunks per file
  -., --hidden                  Search hidden files and directories
  -L, --follow                  Follow symbolic links
  -d, --max-depth <N>           Limit directory traversal depth
      --no-ignore               Don't respect .gitignore
      --type-list               Show all supported file types
      --color <WHEN>            When to use color (auto, always, never)
      --reindex                 Force full re-index
      --full-index              Wait for indexing to complete before searching
      --index-only              Build index without searching
      --stats                   Show index statistics
      --clear-cache             Delete cached index
      --show-root               Print resolved project root and exit
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
