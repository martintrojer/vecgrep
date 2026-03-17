# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
- [ ] `run()` in `main.rs` is 160+ lines with multi-phase early-exit checks — consider further decomposition if complexity grows
- [ ] Server tests in `serve.rs` share a single `OnceLock` server instance — works because the HTTP API is read-only, but if mutations are added later, tests will need isolation

## Critical review findings

- [x] README documents nonexistent `-C`/`--context` flag and `context` config key — removed from README
- [x] Show indexing progress as fraction (e.g. "42/380 files") — unbounded channel lets walker run ahead; progress shows N/M when total is known
- [ ] Add `-p`/`--pretty` short flag (ripgrep convention for forcing colors + headings when piping)
- [x] `--` separator — works natively via clap, no code changes needed. Clap even suggests it in error messages for dash-prefixed queries
- [x] Acknowledge binary size (~109MB with embedded model vs ~6MB for ripgrep) in README Install section
