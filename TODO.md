# TODO

## Code Quality

### Duplication
- [ ] `score_to_color` reimplemented in TUI (`tui.rs`) because `output::score_to_color` returns `termcolor::Color` — consider a shared `ScoreTier -> color name` mapping

### Simplification
- [ ] `run()` in `main.rs` is 160+ lines with multi-phase early-exit checks — consider further decomposition if complexity grows
- [ ] `event_loop` in `tui.rs` is ~280 lines — consider splitting into `handle_input`, `update_state`, and `render` if the TUI grows further
- [ ] `Args` struct in `cli.rs` has 30+ fields — approaching the threshold where grouping (e.g. `WalkArgs`, `OutputArgs`) would improve readability
- [ ] `config::merge` is manual field-by-field merge of 21 fields — fragile when adding new config options

### Minor Cleanups
- [ ] `PipelineStatus::indexed()` returns `0` for `Ready` variant but is only called during `Indexing` — rename or remove the `Ready` arm
- [ ] Avoid `Box<dyn ToSql>` allocation in `index.rs` search for the common no-explicit-paths case
- [ ] `to_string_lossy().to_string()` in `paths.rs` — document why lossy conversion is acceptable or use `into_string()`
- [ ] Inconsistent stderr output: `initialize_embedder` uses raw `eprint!` while everything else uses the `status!` macro

## Tests

### Missing Coverage
- [ ] Remote embedder zero-vector fallback path (`remote.rs:62-79`) is untested — test that zero vectors are returned when `embedding_dim` is known, and errors propagate when it's `None`
- [ ] No test for schema version migration — opening an index with an old `PRAGMA user_version` should trigger a rebuild
- [ ] No test for `--no-scope` behavior at the search layer
- [ ] `resolve_query_flag` error path calls `process::exit()` (`main.rs:539-545`) — refactor to return `Result` to enable testing
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
- [ ] No test for `/status` with active path scopes (scope field presence untested — needs second test server)

### Test Quality
- [ ] Server tests in `serve.rs` share a single `OnceLock` server instance — works because the HTTP API is read-only, but if mutations are added later, tests will need isolation
- [ ] `test_status_returns_ready` uses `sleep(500ms)` instead of polling `/status` — fragile on slow CI
- [ ] `test_large_file_multiple_chunks` uses magic-number slack of 60 tokens (`chunker.rs:170`) — bound could be tighter or proportional

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] `make_relative` in `paths.rs` allocates a `PathBuf` per call — potential micro-optimization if path rewriting appears in profiles
