# vecgrep

Semantic grep — like [ripgrep](https://github.com/BurntSushi/ripgrep), but with vector search.

vecgrep uses a local embedding model ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) to search your codebase by meaning rather than exact text matches. The model is embedded directly in the binary — no external services, no API keys, fully offline.

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

# Index management
vecgrep --stats              # show index statistics
vecgrep --reindex ./src      # force full re-index
vecgrep --clear-cache        # delete cached index
vecgrep --index-only ./src   # build index without searching
```

## How it works

1. **Walk** — discovers files using the same engine as ripgrep (`.gitignore`-aware, binary detection)
2. **Chunk** — splits files into overlapping token-window chunks, snapped to line boundaries
3. **Embed** — runs each chunk through the ONNX model to produce a 384-dimensional vector
4. **Index** — caches embeddings in a local SQLite database (`.vecgrep/index.db`), keyed by BLAKE3 content hash so only changed files are re-embedded on subsequent runs
5. **Search** — computes cosine similarity between your query embedding and all cached chunk embeddings, returns top-k results

## Install

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
  -t, --threshold <F>          Minimum similarity 0.0–1.0 [default: 0.3]
  -i, --interactive             Interactive TUI mode
  -T, --type <TYPE>             Filter by file type (rust, python, js, ...)
  -g, --glob <PATTERN>          Filter by glob
  -C, --context <N>             Context lines around match [default: 3]
  -j, --threads <N>             Indexing threads
      --reindex                 Force full re-index
      --index-only              Build index without searching
      --stats                   Show index statistics
      --clear-cache             Delete cached index
      --json                    JSONL output
      --chunk-size <N>          Tokens per chunk [default: 500]
      --chunk-overlap <N>       Overlap tokens [default: 100]
```

## Environment variables

- `VECGREP_MODEL_CACHE` — override model cache directory (default: system cache dir)
- `VECGREP_LOG` — enable debug logging, e.g. `VECGREP_LOG=debug`
- `EDITOR` — editor opened by `Ctrl+O` in TUI mode

## License

MIT
