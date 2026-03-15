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

- **CLI default**: `drain_all()` blocks until indexing is complete, then searches the up-to-date index. Embedder and Index stay on the main thread.
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
- **Schema changes rebuild, not migrate**: `.vecgrep/index.db` is disposable cache state. `index.rs` tracks a `PRAGMA user_version` schema version; when it changes, vecgrep drops and recreates the cache instead of attempting in-place schema migrations.
- **Index location**: `.vecgrep/index.db` in the project root directory. Automatically added to `.gitignore`. The project root is discovered by walking up from the search path looking for `.git/`, `.hg/`, `.jj/`, or `.vecgrep/`. Use `--show-root` to print it.
- **One invocation, one root**: vecgrep is intentionally single-root. Paths outside the selected project root fail by default. `--skip-outside-root` turns that failure into “ignore these paths,” but skipped paths are not indexed and never appear in results.
- **`--stats` includes holes**: Stats now report failed chunks (`Holes`) in addition to files/chunks/DB size. Holes are chunks whose embedding failed and were stored as zero vectors, so they exist in the cache but never match a query.
- **JSON output includes `root`**: All JSONL output (`--json` and `--serve`) includes a `"root"` field with the canonical project root path, so clients can resolve the project-root-relative `"file"` paths.
- **Embeddings stored in vec0 virtual table**: `sqlite-vec` handles vector storage and KNN search via the `vec_chunks` virtual table. Embeddings are passed as little-endian `f32` bytes using `zerocopy::IntoBytes` for zero-copy conversion. The `chunks` table stores text/metadata, `vec_chunks` stores vectors, joined by `chunk_id`. The `vec_chunks` dimension is baked into the virtual table at creation time — `create_tables()` uses `EMBEDDING_DIM` (384) as default. When switching models (e.g. local→Ollama with 1024-dim), `check_config()` detects the `IndexConfig` change and `rebuild_for_config()` atomically recreates the cache with the correct dimension.
- **Write-path atomicity**: Index writes are intentionally wrapped in `BEGIN IMMEDIATE` transactions via a shared helper in `index.rs`. Keep multi-statement mutations atomic because the CLI can be interrupted at any time.
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

## Reviewed Decisions

These came out of a design review and should be treated as intentional unless requirements change:

- **CLI searches must not return unlabeled partial results**: Default CLI behavior waits for indexing to complete before searching. Progressive partial results are only for TUI and `--serve`, where that tradeoff is explicit.
- **Prefix-scoped stale removal is intentional**: Searching `src/` should not force project-wide stale cleanup. The current behavior favors speed for narrow searches over eagerly cleaning unrelated directories.
- **Mixed-root inputs are not merged**: Do not silently mix files from different project roots into one cache. Either run vecgrep separately per root or use `--skip-outside-root` to ignore the out-of-root paths.
- **Index warn threshold prompts only once**: Re-prompting as discovery continues would make large-vault indexing noisy and frustrating. Users can already abort at any time.
- **Config invalidation stays coarse-grained**: If `IndexConfig` changes, rebuild the cache. Do not add partial “embeddings are probably still valid” exceptions for chunking or overlap changes unless there is a very strong correctness story.
- **Index holes are currently surfaced via `--stats`**: Failed remote embeddings become zero vectors and are counted as `Holes`. That is the current user-visible surfacing mechanism; search output itself does not yet annotate them.

## Allium Spec

The repo includes an Allium spec at `vecgrep.allium`. Upstream Allium lives at https://github.com/juxt/allium. Treat the local spec as the clearest product-model description of vecgrep's intended behavior, not as a parser-checked source of truth.

- The spec is intentionally higher-level than the Rust code. It models root selection, path admission, config precedence, indexing/search lifecycles, and the CLI/TUI/server surfaces. It does **not** try to capture threading, exact chunking internals, or storage details.
- When refactoring, prefer moving the code **toward** the spec's shape: explicit invocation resolution, one selected root, admitted vs rejected paths, clear blocking vs progressive indexing behavior, and distinct CLI/TUI/server surfaces.
- Do not assume every helper or field in `main.rs` needs a direct one-to-one counterpart in the spec. The spec is an idealized behavior model, not a mandate to over-abstract the implementation.
- Use the spec to spot design drift:
  - duplicated user-visible policy across CLI/TUI/server
  - hidden precedence rules
  - mixed parsing/runtime state
  - behavior that depends on incidental implementation details rather than explicit lifecycle/state
- If code and spec diverge, decide explicitly whether the spec is wrong, the code is wrong, or the spec is intentionally aspirational before changing either.
