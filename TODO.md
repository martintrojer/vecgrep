# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
- [ ] `run()` in `main.rs` is 160+ lines with multi-phase early-exit checks — consider further decomposition if complexity grows
- [ ] Server tests in `serve.rs` share a single `OnceLock` server instance — works because the HTTP API is read-only, but if mutations are added later, tests will need isolation

## Code Review
- [ ] Deduplicate `handle_search` dispatch in `worker_loop` (`pipeline.rs`) — the `try_recv` and `recv_timeout` branches have identical match arms
- [ ] `search()` in `index.rs` post-filters by `path_scopes` after requesting `top_k` from sqlite-vec — scoped searches may return fewer results than requested; consider over-fetching (e.g. `top_k * 3`) when scopes are active
- [ ] Extract shared score-tier logic from `output::score_to_color` and `tui.rs` color mapping — return a High/Medium/Low enum instead of `termcolor::Color`, let each consumer map to its own color type
- [ ] `event_loop` in `tui.rs` is ~280 lines — consider splitting into `handle_input`, `update_state`, and `render` if the TUI grows further
- [ ] `status!` macro in `main.rs` could be a plain function — only used for simple format strings, a function would be easier to grep for
- [ ] `search()` in `index.rs` builds query strings via `format!` on every call — the no-explicit-paths case could use a `const` string
- [ ] `make_relative` in `paths.rs` allocates a `PathBuf` per call — potential micro-optimization if path rewriting appears in profiles
- [ ] `Args` struct in `cli.rs` has 30+ fields — approaching the threshold where grouping (e.g. `WalkArgs`, `OutputArgs`) would improve readability

## Test Review (completed)
- [x] Fixed `test_status_indexing_with_total_when_walker_done` — use `on_send()`, assert on total value
- [x] Added distance precondition to `test_search_threshold`
- [x] Added `test_exit_code_1_when_no_matches`
- [x] Added `test_json_output_structure`
- [x] Relaxed `test_first_run_status_output_is_compact` to pattern-based assertions
