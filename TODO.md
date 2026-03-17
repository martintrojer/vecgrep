# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
- [ ] `run()` in `main.rs` is 160+ lines with multi-phase early-exit checks — consider further decomposition if complexity grows
- [ ] Server tests in `serve.rs` share a single `OnceLock` server instance — works because the HTTP API is read-only, but if mutations are added later, tests will need isolation
- [ ] `event_loop` in `tui.rs` is ~280 lines — consider splitting into `handle_input`, `update_state`, and `render` if the TUI grows further
- [ ] `make_relative` in `paths.rs` allocates a `PathBuf` per call — potential micro-optimization if path rewriting appears in profiles
- [ ] `Args` struct in `cli.rs` has 30+ fields — approaching the threshold where grouping (e.g. `WalkArgs`, `OutputArgs`) would improve readability
