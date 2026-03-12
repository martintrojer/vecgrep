# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                    # debug build (downloads ~90MB ONNX model on first build)
cargo build --release          # release build (LTO enabled, ~109MB binary)
cargo test                     # run all tests (requires model, so first build must complete)
cargo test embedder::tests::test_embed_single  # run a single test
```

The first build downloads `all-MiniLM-L6-v2` model files from HuggingFace and caches them at `$VECGREP_MODEL_CACHE` or the system cache dir (`~/Library/Caches/vecgrep/models/` on macOS). Subsequent builds skip the download.

Debug logging: `VECGREP_LOG=debug cargo run -- "query" ./path`

## Before Committing

Always run these before committing:

```bash
cargo fmt
cargo clippy -- -D warnings
```

## Architecture

vecgrep is a semantic grep tool: it embeds a query and file chunks into vectors using a local ONNX model, then ranks chunks by cosine similarity.

**Data pipeline** (orchestrated in `main.rs`):

```
walker  →  chunker  →  embedder  →  index (SQLite)  →  search  →  output/tui/serve
(files)    (chunks)    (vectors)    (cache)             (rank)     (display)
```

**Key design decisions:**

- **Model embedded in binary**: `build.rs` downloads model + tokenizer at build time; they're compiled into the binary via `include_bytes!`. This means the ONNX model (~90MB) lives in `$OUT_DIR/models/` and the binary is self-contained.
- **ort API quirks**: `ort` v2.0.0-rc.12 errors are not `Send+Sync`, so `?` with `anyhow` doesn't work — all ort calls must use `.map_err(|e| anyhow::anyhow!("{}", e))`. `Session::run` requires `&mut self`.
- **Embeddings are L2-normalized**, so cosine similarity = dot product. The search module (`search.rs`) exploits this by doing a simple `embedding_matrix.dot(&query)`.
- **Cache invalidation**: BLAKE3 content hash per file. If model name or chunk params change (stored in `meta` table as JSON), the entire index is rebuilt.
- **Index location**: `.vecgrep/index.db` in the search root directory. Automatically added to `.gitignore`.
- **Embeddings stored as BLOB**: `Vec<f32>` → little-endian bytes in SQLite, reconstituted into `ndarray::Array2<f32>` for search.
- **CLI flags follow ripgrep conventions**: `-t` for type, `-g` for glob, `-C` for context, `-l` for files-with-matches, `-c` for count, `-.` for hidden, `-L` for follow, etc. Any new CLI flag must be checked against `rg --help` for compatibility — do not reuse a short flag that means something different in rg.

**Module responsibilities:**

| Module | Role |
|---|---|
| `embedder.rs` | ONNX session + tokenizer, batch inference with mean-pooling |
| `chunker.rs` | Split file content into overlapping token-window chunks, snapped to line boundaries |
| `index.rs` | SQLite schema (`meta`/`files`/`chunks`), upsert, stale removal, bulk load into ndarray |
| `search.rs` | Matrix dot-product scoring, top-k partial sort, threshold filter |
| `walker.rs` | `ignore` crate for .gitignore-aware file discovery with type/glob filters |
| `output.rs` | `termcolor` for ripgrep-style colored output, JSONL mode, TTY detection |
| `serve.rs` | `tiny_http` server for `--serve` mode; loads model once, serves queries over HTTP |
| `tui.rs` | `ratatui` interactive mode with debounced re-embedding on keystrokes |
