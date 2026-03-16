# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)

## Code Review

### Recommended

- [ ] **Dead `context` flag**: `-C` / `context` is wired through CLI (`src/cli.rs:51,164`), config (`src/config.rs:13,59`), and invocation (`src/invocation.rs:133`) but never consumed. Implement context-line display, remove it entirely, or hide with `#[arg(hide = true)]`.
- [ ] **`drain_initial_indexing` / `drain_remaining_indexing` duplication**: Both do spinner + `drain_all` + clear (`src/main.rs:242,288`). Extract shared pattern into one function with optional threshold parameter (~20 lines saved).
- [ ] **`WalkOptions` should derive `Default`**: Constructed verbatim in 10+ places (`src/walker.rs:51`). Add `#[derive(Default)]` to enable struct-update syntax.

### Suggestions

- [ ] `SCORE_HIGH_THRESHOLD` / `SCORE_MEDIUM_THRESHOLD` are `pub` but only used within `output.rs:110-111` — make private.
- [ ] `has_project_marker` is `pub` but only used within `root.rs:36` — make private.
- [ ] `upsert_file` in `index.rs:241` is a one-line test-only wrapper — consider `#[cfg(test)]`.
- [ ] `unwrap_or(384)` in `remote.rs:76` is in a branch where `embedding_dim` is always `Some` — dead fallback.
- [ ] `admitted.clone()` in `invocation.rs:104` is unnecessary — `admitted` isn't used after, transfer ownership directly.
- [ ] Magic number `32` for batch size in `main.rs:173` — promote to a named constant.

## Test Review

### Critical

- [ ] **`test_batches_mixed_sizes` doesn't test batch splitting** (`src/embedder/remote.rs:384`): All texts fit in one batch (184 < 200 threshold). Use sizes that exceed the threshold and assert multiple batches are created.
- [ ] **`test_default_remote_max_chars` is tautological** (`src/embedder/remote.rs:519`): Asserts a constant equals itself. Delete it.
- [ ] **`test_estimate_tokens_short/conservative` mirror the formula** (`src/chunker.rs:280-296`): Re-derive `len * 2 / 5` by hand. Replace with property-based tests that verify estimates >= actual token counts.

### Recommended

- [ ] **`test_worker_reports_search_errors` — weak assertion** (`src/pipeline.rs:833`): Only checks error message is non-empty. Assert the message mentions "dimension" or the mismatched sizes.
- [ ] **`test_large_file_multiple_chunks` — loose bound** (`src/chunker.rs:150`): `chunks.len() > 1` passes even if chunker drops content. Verify all original content is covered.
- [ ] **Two integration tests missing `--threshold 0.0`** (`tests/integration.rs:430,454`): `test_full_index_indexes_before_search` and `test_default_mode_uses_cached_index` rely on default 0.3 threshold. Could flake on model changes.
- [ ] **`test_capped_chunk_size_reduces` missing the key case** (`src/invocation.rs:449`): Never asserts `capped_chunk_size(500, Some(256)) == 256` — the actual capping case.

### Suggestions

- [ ] No unit tests for `process_batch` with `explicit: true` files — integration tests cover it but unit coverage would catch regressions faster.
- [ ] `test_walk_nonexistent_path` documents an implicit contract (silent empty return) — name could be clearer.
