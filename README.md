# vecgrep

Semantic grep — like [ripgrep](https://github.com/BurntSushi/ripgrep), but with vector search.

<!-- TODO: add demo GIF/video here -->

Search your codebase by meaning, not just text. Ask for "error handling for network timeouts" and find the relevant code, even if it doesn't contain those exact words.

- **Local-first** — embedding model ships inside the binary. No API keys, no GPU, your code stays on your machine.
- **Fast** — indexes thousands of files in seconds, searches in milliseconds. Interactive TUI and HTTP server update results progressively during indexing.
- **Bring your own model** — optionally connect to [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible API. See [BENCHMARK.md](BENCHMARK.md).

## Install

Pre-built binaries on the [releases page](https://github.com/mtrojer/vecgrep/releases), or:

```bash
mise use github:martintrojer/vecgrep    # mise
cargo install vecgrep                    # cargo
```

### AI skill for coding agents

```bash
npx skills add martintrojer/vecgrep                    # Codex / generic
/plugin marketplace add martintrojer/vecgrep            # Claude Code
```

## Quick start

```bash
vecgrep "error handling" ./src                          # search by meaning
vecgrep -i "authentication"                             # interactive TUI
vecgrep -l "retry logic" | xargs rg "unwrap"            # combine with ripgrep
vecgrep --serve --port 8080 ./src                       # HTTP server for IDE plugins
vecgrep --embedder-url http://localhost:11434/v1/embeddings --embedder-model mxbai-embed-large "query"
```

## Key features

| Feature | |
|---|---|
| **TUI mode** (`-i`) | Live search with preview pane, score colors, configurable file opener (`--open-cmd`) |
| **HTTP server** (`--serve`) | `/search` and `/status` endpoints for IDE integration ([vecgrep.nvim](https://github.com/martintrojer/vecgrep.nvim)) |
| **Path scoping** | Results scoped to requested paths, like ripgrep. `--no-scope` to search entire project |
| **File type filters** | `-t rust`, `-T markdown`, `-g "*.rs"` — same flags as ripgrep |
| **JSONL output** | `--json` for scripting with jq |
| **Config files** | Project (`.vecgrep/config.toml`) and global (`~/.config/vecgrep/config.toml`) |
| **Cache** | BLAKE3 content hashing — only changed files are re-embedded |

## Documentation

- **[User Guide](doc/guide.md)** — full examples, options reference, path semantics, configuration, server API
- **[Benchmarks](BENCHMARK.md)** — model comparisons and retrieval accuracy

## How it works

1. **Walk** — `.gitignore`-aware file discovery (same engine as ripgrep)
2. **Chunk** — overlapping token-window chunks, snapped to line boundaries
3. **Embed** — built-in ONNX model or external API
4. **Index** — SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec), keyed by content hash
5. **Search** — cosine similarity KNN, fast enough for every-keystroke use

## License

MIT
