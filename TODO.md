# TODO

## Code quality
- [ ] `with_transaction` closure in `index.rs` could take `&Connection` — would require refactoring all helper methods
- [ ] `SearchOutcome::request_id()` repeats the same match arm three times — low value for only three variants

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
