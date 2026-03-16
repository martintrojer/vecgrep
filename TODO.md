# TODO

## Code quality

### High
- [x] Collapse duplicated `remove_stale_files` / `remove_stale_files_under` in `index.rs` into one method with `prefix: Option<&str>`
- [x] Extract duplicated clear SQL (DROP/DELETE) in `index.rs` into a private `clear_all_data` helper
- [x] Remove dead `_stream_progress` parameter from `process_batch` in `pipeline.rs`

### Medium
- [ ] Eliminate `Invocation`/`Args` field duplication in `main.rs` — access values through `invocation.args.*` instead of copying them out
- [ ] Duplicated score-to-color thresholds in `tui.rs` — convert from `termcolor::Color` instead of duplicating logic from `output.rs`
- [ ] Unify or document the two `l2_norm` / `l2_norm_vec` functions in `embedder.rs`
- [ ] Mark `Index::clear()` as `#[cfg(test)]` — only used in tests

### Low
- [ ] Extract magic number `embed_batch_size = 64` in `pipeline.rs` to a named constant
- [ ] Extract error-response construction in `serve.rs` into a shared helper
- [ ] `with_transaction` closure in `index.rs` could take `&Connection` to prevent accidental nested transactions
- [ ] Redundant `let mut file = file;` rebinding in `pipeline.rs:recv_one`
- [ ] `SearchOutcome::request_id()` repeats the same match arm three times — consider restructuring the enum

## Test quality

### Medium
- [ ] Strengthen chunker test assertions: `test_large_file_multiple_chunks` should verify token budget, not just existence
- [ ] `test_overlap_larger_than_chunk` should verify forward progress and full file coverage, not just non-emptiness
- [ ] `test_single_very_long_line` should assert chunk count and content, not just `start_line`
- [ ] `test_cli_progress_reporter_finish_stops_thread` has zero assertions — add a timing check
- [ ] Config tests (`config.rs:235-282`) mutate env vars without RAII cleanup guards

### Low
- [ ] `test_semantic_similarity` should assert a minimum separation margin, not just `>`
- [ ] `test_stats` in `index.rs` queries raw SQL instead of `index.stats()` — misses bugs in the public API
- [ ] Remove duplicated `test_search_empty` between `index.rs` unit test and `integration.rs`
- [ ] Fix integration test `make_embedding` helper — uniform vectors all normalize to the same unit vector, masking ranking bugs
- [ ] Add walker test for symlink following (`follow: true`)
- [ ] Add test for `file_types_not` (negative type filter)
- [ ] `test_search_k_override` in `serve.rs` should use `assert_eq!(len, 1)` not `assert!(len <= 1)`

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
