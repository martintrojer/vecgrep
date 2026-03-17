# TODO

## Future
- [ ] Consider IVF partitioning for vec0 virtual table if index grows beyond 100K chunks (brute-force KNN is O(n) per query)
- [ ] Add unit test for `process_batch` with `explicit: true` files (integration tests cover it, but unit coverage would catch regressions faster)
- [ ] `run()` in `main.rs` is 160+ lines with multi-phase early-exit checks — consider further decomposition if complexity grows
- [ ] Server tests in `serve.rs` share a single `OnceLock` server instance — works because the HTTP API is read-only, but if mutations are added later, tests will need isolation

## Critical review findings

- [x] README documents nonexistent `-C`/`--context` flag and `context` config key — removed from README
- [ ] Show indexing progress as fraction (e.g. "42/380 files") — the walker runs ahead so total is available; currently only shows indexed count with no denominator
- [ ] Add `-p`/`--pretty` short flag (ripgrep convention for forcing colors + headings when piping)
- [ ] Document `--` separator support in README — rg users expect `vecgrep "pattern" -- file1 file2`; likely works via clap but not tested or mentioned
- [x] Acknowledge binary size (~109MB with embedded model vs ~6MB for ripgrep) in README Install section
