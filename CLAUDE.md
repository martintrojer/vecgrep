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

**Benchmarking models**: When swapping models in `build.rs` for benchmarking, you must clear both caches to avoid stale model files. The download cache uses filename-based keys (`model.onnx`) so different model URLs collide:
```bash
rm -rf ~/Library/Caches/vecgrep/models/
cargo clean
cargo test --test benchmark_models -- --nocapture
```

**Model benchmarks** (see [BENCHMARK.md](BENCHMARK.md)):
- Built-in all-MiniLM-L6-v2 wins at small scale (<500 docs) thanks to best separation (0.505)
- At large scale (6,500+ docs), Ollama models (mxbai-embed-large, embeddinggemma) beat MiniLM by ~3% MRR — richer representations help with more distractors
- All texts truncated to 1024 chars for fair cross-model comparison
- `benchmark_large` downloads CodeSearchNet from HuggingFace (cached locally)

**GPU/ANE acceleration**: We tested CoreML execution provider (Apple Neural Engine) via `ort`'s `coreml` feature flag. Findings:
- CoreML is **significantly slower** than CPU on real workloads (tested on a Markdown vault)
- For our small model (22M params), CPU inference is already <5ms per query
- The data transfer overhead between CPU↔ANE exceeds any compute savings at this model size
- Conclusion: CPU-only is the right choice. Don't revisit unless the model grows significantly (>100M params)

## Before Committing

Always run these before committing:

```bash
cargo fmt
cargo clippy -- -D warnings
```

## Architecture

vecgrep is a semantic grep tool: it embeds a query and file chunks into vectors, then ranks chunks by cosine similarity. Embeddings come from either the built-in ONNX model (default) or an external OpenAI-compatible API (`--embedder-url`).

**Data pipeline** (orchestrated in `main.rs`, streamed via `std::sync::mpsc`):

```
walker (thread) →  channel(64)  →  StreamingIndexer  →  index (SQLite)  →  search  →  output/tui/serve
(files)            (backpressure)   (chunk + embed)      (cache)            (rank)     (display)
```

The walker runs on a background thread feeding files through a bounded `sync_channel(batch_size * 2)`. The three modes consume the channel differently:

- **CLI default**: searches immediately against the cached index, then `drain_all()` indexes remaining files in the background for next time. Embedder and Index stay on the main thread.
- **CLI `--full-index`**: `drain_all()` blocks until all files are indexed (with threshold prompt), then searches. Same single-threaded ownership.
- **TUI/serve**: Use `EmbedWorker` — a background thread that owns the `Embedder`, `Index`, and `StreamingIndexer`. The UI thread communicates via channels, never blocking on embed calls. This ensures Esc always works in the TUI and the HTTP server stays responsive, even when Ollama is loading a model (which can block for 30+ seconds).

**EmbedWorker architecture** (TUI/serve only):
```
walker (thread) → file channel → EmbedWorker (thread, owns Embedder + Index + StreamingIndexer)
                                   ↑ search requests       ↓ search results
                                   ↑ (mpsc channel)        ↓ (mpsc channel)
                                 UI thread (TUI event loop / HTTP server)
                                   ↓ index progress
                                   ↓ (mpsc channel)
```
The worker prioritizes search requests over indexing: it checks for pending searches between every small batch (`WORKER_BATCH_SIZE` files). Search results are returned as `SearchOutcome` — either `Results(Vec<SearchResult>)` or `EmbedError(String)` — so the UI can distinguish "no matches" from "embedder failed." Index progress is sent on a separate channel. The worker shuts down cleanly via `Drop`.

**Key design decisions:**

- **Config file hierarchy**: `.vecgrep/config.toml` (project root) > `~/.config/vecgrep/config.toml` (global) > CLI defaults. Loaded via `config::load_config(project_root)` after project root discovery in `main.rs`. All fields are `Option<T>`, merged with `or()` precedence.
- **Local vs remote embedding — avoiding index holes**:
  - **Local** (`Embedder::Local`): `build.rs` downloads model + tokenizer at build time, compiled into the binary via `include_bytes!`. The chunker uses the real tokenizer for exact token counts. The ONNX model silently truncates at `MAX_SEQ_LEN` (256 tokens). No errors possible — every chunk gets an embedding.
  - **Remote** (`Embedder::Remote`): Uses `--embedder-url` with any OpenAI-compatible API (Ollama, LM Studio, etc.). No tokenizer available, so the chunker uses a character heuristic (~2.5 chars/token — URLs, markdown, and code tokenize densely). At startup, probes Ollama's `/api/show` for the model's context length to set accurate truncation limits. Falls back to 1200-char default for non-Ollama servers. Unlike the local model, Ollama **rejects** (HTTP 400) texts exceeding context length instead of truncating. The `chunk_size` is automatically capped to the model's context in `main.rs`. If a chunk still fails, the batch is retried one-at-a-time, and any remaining failures get a zero-vector embedding with a warning logged (including the filename via `pipeline.rs`). Zero vectors are index holes — they exist but never match queries. The goal is zero holes: correct chunking avoids them, the fallback chain catches edge cases.
- **ort API quirks**: `ort` v2.0.0-rc.12 errors are not `Send+Sync`, so `?` with `anyhow` doesn't work — all ort calls must use `.map_err(|e| anyhow::anyhow!("{}", e))`. `Session::run` requires `&mut self`.
- **Embeddings are L2-normalized**, so cosine similarity = dot product. Vector search uses `sqlite-vec`'s `vec0` virtual table with `distance_metric=cosine` — a single SQL query replaces the old ndarray matrix multiply.
- **Cache invalidation**: BLAKE3 content hash per file. If model name or chunk params change (stored in `meta` table as JSON), the entire index is rebuilt.
- **Index location**: `.vecgrep/index.db` in the project root directory. Automatically added to `.gitignore`. The project root is discovered by walking up from the search path looking for `.git/`, `.hg/`, `.jj/`, or `.vecgrep/`. Use `--show-root` to print it.
- **JSON output includes `root`**: All JSONL output (`--json` and `--serve`) includes a `"root"` field with the canonical project root path, so clients can resolve the project-root-relative `"file"` paths.
- **Embeddings stored in vec0 virtual table**: `sqlite-vec` handles vector storage and KNN search via the `vec_chunks` virtual table. Embeddings are passed as little-endian `f32` bytes using `zerocopy::IntoBytes` for zero-copy conversion. The `chunks` table stores text/metadata, `vec_chunks` stores vectors, joined by `chunk_id`. The `vec_chunks` dimension is baked into the virtual table at creation time — `create_tables()` uses `EMBEDDING_DIM` (384) as default. When switching models (e.g. local→Ollama with 1024-dim), `check_config()` detects the `IndexConfig` change, `clear()` drops `vec_chunks`, and `set_config()` recreates it with the correct dimension.
- **CLI flags follow ripgrep conventions**: `-t` for type, `-g` for glob, `-C` for context, `-l` for files-with-matches, `-c` for count, `-.` for hidden, `-L` for follow, `--ignore-file` for additional ignore files, etc. Any new CLI flag must be checked against `rg --help` for compatibility — do not reuse a short flag that means something different in rg.

**Module responsibilities:**

| Module | Role |
|---|---|
| `config.rs` | Load `~/.config/vecgrep/config.toml`, all fields `Option<T>`, merged with CLI args in `main.rs` |
| `embedder.rs` | `Embedder` enum: `Local` (ONNX + tokenizer) or `Remote` (OpenAI-compatible HTTP API). Single queries use CPU, batches use the configured backend |
| `chunker.rs` | Split file content into overlapping token-window chunks, snapped to line boundaries. Uses tokenizer when available, char-based heuristic otherwise |
| `pipeline.rs` | `StreamingIndexer` (channel consumer with `poll()`/`drain_all()`), `EmbedWorker` (background thread for non-blocking TUI/serve), `process_batch()` for chunk → embed → upsert |
| `paths.rs` | Path conversions: `to_project_relative()`, `to_cwd_relative()` |
| `index.rs` | SQLite schema (`meta`/`files`/`chunks`/`vec_chunks`), upsert, stale removal, vector search via sqlite-vec |
| `walker.rs` | `ignore` crate for .gitignore-aware file discovery; `walk_with()` helper, `walk_paths_streaming()` for channel-based walking |
| `output.rs` | `termcolor` for ripgrep-style colored output, JSONL mode, TTY detection |
| `serve.rs` | `tiny_http` server for `--serve` mode; `run_streaming()` interleaves indexing with request handling |
| `tui.rs` | `ratatui` interactive mode; `run_streaming()` interleaves indexing with the event loop |
