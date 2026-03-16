# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
