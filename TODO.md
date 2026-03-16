# TODO

## Code quality

- [ ] `with_transaction` closure in `index.rs` could take `&Connection` — would make it impossible for callers to execute statements outside the transaction boundary. Requires refactoring all helper methods.
- [ ] Add `debug_assert!(self.conn.is_autocommit())` in `with_transaction` to guard against nested transactions
- [ ] Remove dead methods: `EmbedWorker::recv_results` and `Embedder::is_remote` (never called)
- [ ] Extract `const PROJECT_MARKERS` — the `[".git", ".hg", ".jj", ".vecgrep"]` list is duplicated in `find_project_root` and `has_project_marker`
- [ ] Inline `build_invocation` and `resolve_query` — trivial one-liners called once each
- [ ] TUI: `continue` on stale search results skips rendering and input handling — Esc can become unresponsive if stale results queue up. Should skip only the result processing, not the whole loop iteration.
- [ ] `serve.rs` has 8 parameters with `#[allow(clippy::too_many_arguments)]` — bundle into a config struct

## Tests

- [ ] `test_stats` never calls `stats()` — only tests `chunk_count()`. Add assertions for all `IndexStats` fields.
- [ ] `test_merge_with_empty` only tests one merge direction — add `merge(values, empty)` to catch precedence bugs
- [ ] Add test for `ignore_files` additive merge in `config::tests` (currently only covered by integration test)
- [ ] Add happy-path test for `parse_embeddings` — L2 normalization, dimension discovery, and index reordering are untested
- [ ] Add test for `rebuild_for_config` — verify old data is cleared and new config is stored
- [ ] Add test for `resolve_config` defaults — verify hardcoded defaults apply when both CLI and config omit values
- [ ] Add `test_search_top_k_zero` for the early-return edge case

## Spec-aligned simplifications (done)

- [x] Replace `FlowControl` with `Option<bool>`
- [x] Remove `RunMode::IndexOnly` variant
- [x] Consolidate `StaleRemovalScope::All` into `Prefix`
- [x] Merge `drain_initial_indexing` and `drain_initial_indexing_with_prompt`
- [x] Restructure `process_batch` to iterate per-file
- [x] Inline CLI spinner into `drain_all` callback
- [x] Remove double canonicalization of project root
- [x] Collapse path admission into one pass
- [x] Unify 3-layer config merge

## Future

- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
