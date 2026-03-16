# TODO

## Code quality
- [ ] `with_transaction` closure in `index.rs` could take `&Connection` — would require refactoring all helper methods
- [ ] `SearchOutcome::request_id()` repeats the same match arm three times — low value for only three variants

## Spec-aligned simplifications

These items came from comparing the Allium spec (`vecgrep.allium`) against the implementation.
The goal is to reduce complexity while staying true to the spec's behavioural model.

### Done

- [x] **Replace `FlowControl` with `Option<bool>`**
- [x] **Remove `RunMode::IndexOnly` variant**
- [x] **Consolidate `StaleRemovalScope::All` into `Prefix`**
- [x] **Merge `drain_initial_indexing` and `drain_initial_indexing_with_prompt`**
- [x] **Restructure `process_batch` to iterate per-file**
- [x] **Inline CLI spinner into `drain_all` callback**
- [x] **Remove double canonicalization of project root**
- [x] **Collapse path admission into one pass**

### Remaining

- [ ] **Unify 3-layer config merge**: Make clap args `Option<T>` (no defaults), then do `cli.or(project).or(global).unwrap_or(DEFAULT)` — matching the spec's `config` block. Eliminates `apply_config`, both macros, `cli_provided`, `with_config`, and the separate `merge()` in `config.rs`. Lower priority: high churn (~15 field type changes cascade through all access points) for moderate benefit (~80 lines). The current implementation works correctly and is well-tested.
- [ ] **Deduplicate stats handling**: `--stats` is checked in both `handle_pre_execution_actions` (before index) and `handle_post_index_actions` (after index). These serve different purposes (pre: skip embedder loading; post: show stats after reindex), so the duplication is functional, not accidental.

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
