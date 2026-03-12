# TODO

## Code Review Findings

### Critical

- [ ] **Deduplicate `to_project_relative`** — identical copies in `main.rs:54`, `tui.rs:61`, `serve.rs:17`. Make it `pub` in one place (e.g. `walker.rs` or a `util` module) and import it everywhere.

- [ ] **Deduplicate walker builder setup** — `walk_paths` and `walk_paths_streaming` in `walker.rs` share ~90% identical WalkBuilder configuration. Extract into a helper that takes a callback:
  ```rust
  fn walk_with<F>(paths: &[String], opts: &WalkOptions, mut on_file: F) -> Result<()>
  where F: FnMut(WalkedFile) -> bool  // return false to stop
  ```

### Recommended

- [ ] **Deduplicate streaming drain logic** — TUI (`tui.rs:155-198`) and serve (`serve.rs:65-111`) have identical channel drain → hash check → batch → process_batch → rate-limited reload blocks. Extract into a `StreamingIndexer` helper in `pipeline.rs`.

- [ ] **Deduplicate `run_loop` / `run_streaming_loop` in `tui.rs`** — ~80% shared code (rendering, input handling, debounced search, preview cache). A single loop function with optional streaming state would eliminate ~150 lines.

- [ ] **Name the magic number `4`** — per-iteration batch size in TUI/serve streaming loops (`tui.rs:169`, `serve.rs:79`). Define `const STREAMING_BATCH_SIZE: usize = 4;` in `pipeline.rs`.

### Minor

- [ ] **`walk_paths` is effectively dead in production** — only used in walker tests after the streaming refactor. Will resolve naturally if walker dedup (above) happens.

- [ ] **Inline `is_streaming_mode`** — `main.rs:248`, used once. Replace with `args.interactive || args.serve` directly.

- [ ] **Narrow variable scope in CLI path** — `main.rs:242-245`: `batch`, `needs_indexing_count`, `threshold_prompted` are declared outside the if/else but only used in the else branch. Move inside.

## Test Review Findings

### Recommended

- [ ] **Tighten `test_process_batch_indexes_files` assertion** — change `assert!(chunk_count >= 2)` to `assert_eq!(chunk_count, 2)` since two single-line files at chunk_size=500 produce exactly 1 chunk each.

- [ ] **Fix type asymmetry in `test_walk_streaming_matches_walk_paths`** — comparing `Vec<&str>` with `Vec<String>`. Both sides should be `Vec<&str>`.

- [ ] **Assert partial count in `test_walk_streaming_receiver_drop`** — currently only checks `result.is_ok()`. Add `assert!(count < 100)` to verify early exit actually happened.

### Suggested New Tests

- [ ] **Multi-chunk file test for `process_batch`** — all current test files are tiny. A file large enough to produce 2+ chunks would exercise the chunk-grouping logic (`current_file` tracking in `pipeline.rs:49-69`).

- [ ] **Streaming skip test (hash match)** — no test exercises the path where a pre-indexed file is skipped during streaming because its hash matches.

- [ ] **Strengthen `test_process_batch_empty`** — also verify the index is empty afterward with `idx.load_all()`.
