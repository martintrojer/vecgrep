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
- [x] **Unify 3-layer config merge**


## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
