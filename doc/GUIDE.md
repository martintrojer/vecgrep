# vecgrep User Guide

See also: [Configuration](CONFIG.md) | [Server API](SERVER.md)

## How it works

1. **Walk** — discovers files using the same engine as ripgrep (`.gitignore`-aware, binary detection)
2. **Chunk** — splits files into overlapping token-window chunks, snapped to line boundaries
3. **Embed** — runs each chunk through the embedding model (built-in or external) to produce a vector
4. **Index** — caches embeddings in a local SQLite database (`.vecgrep/index.db`), keyed by BLAKE3 content hash so only changed files are re-embedded
5. **Search** — cosine similarity via [sqlite-vec](https://github.com/asg017/sqlite-vec), fast enough for every-keystroke use

## Searching

```bash
vecgrep "error handling for network timeouts" ./src
vecgrep "match result { Ok(v) => v, Err(e) => return Err(e) }" ./src
vecgrep "sorting algorithm" --type rust
vecgrep -i "authentication"                     # interactive TUI
```

Useful search controls:

```bash
vecgrep -k 20 --threshold 0.35 "retry backoff" ./src
vecgrep --chunk-size 384 --chunk-overlap 96 "parser state machine"
vecgrep -t rust -T test -g '*.rs' "index invalidation"
```

### Hybrid search

`--hybrid` combines lexical and semantic ranking. It is most useful for short, grep-like queries with strong identifiers, symbols, or exact phrases:

```bash
vecgrep --reindex --hybrid-index               # build a hybrid-capable index (always rebuilds from the project root)
vecgrep --hybrid "IndexConfig" ./src           # lexical + semantic ranking
vecgrep --hybrid "timeout error" ./src
```

- `--hybrid` is query-time behavior.
- `--hybrid-index` controls whether the index stores lexical data for hybrid queries.
- A hybrid query against a non-hybrid index fails with a hint to rebuild using `--reindex --hybrid-index`.
- A hybrid-capable index can still serve normal vector-only searches.
- To go back to a plain vector-only index, rebuild without `--hybrid-index`.

### Filtering results

```bash
vecgrep -l "retry logic" ./src                  # file paths only
vecgrep -c "error handling" ./src               # count per file
vecgrep --json "retry logic" | jq '.score'      # JSONL for scripting
vecgrep --json "error handling" ./src | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'
```

### Combining with other tools

```bash
# Semantic search to find files, then exact match
vecgrep -l "error handling" ./src | xargs rg "unwrap"

# Ripgrep to narrow files, vecgrep to rank by meaning
rg -l "TODO" ./src | xargs vecgrep "technical debt that should be refactored"

# Interactive TUI with xargs
rg -l "TODO" ./src | xargs vecgrep -i --query "technical debt"

# Git integration
vecgrep --json "authentication" ./src | jq -r '.file' | sort -u | xargs git blame
vecgrep -l "database queries" ./src | xargs git log --oneline -5 --

# Pretty-print with bat
vecgrep -l "configuration parsing" ./src | xargs bat --style=header,grid

# Watch mode
vecgrep -l "error handling" ./src | entr -r cargo test
```

## Path scoping

Results are scoped to the paths you specify, like ripgrep:

```bash
vecgrep "query" src/                            # only src/ results
cd src && vecgrep "query"                       # same — scoped to cwd
vecgrep --no-scope "query"                      # search entire project
```

- One invocation, one project root, one cache. Paths outside the root are rejected by default (`--skip-outside-root` to ignore them).
- Single directory: walks recursively, cleans up stale files in that subtree.
- Multiple directories: walks all, updates shared cache, skips stale cleanup.
- Explicit file paths: cached with an `explicit` flag for fast re-search, excluded from directory-only searches. Cleared when a directory walk rediscovers the file.

If you pass paths from multiple roots, vecgrep fails by default. Use `--skip-outside-root` to ignore the out-of-root ones instead:

```bash
vecgrep --skip-outside-root "query" ./src ../other-project/file.rs
```

## Ignoring files

vecgrep respects `.gitignore` by default. For additional patterns:

```bash
vecgrep --ignore-file .vecgrep/ignore "query"
```

Or set it once in `.vecgrep/config.toml`:

```toml
ignore_files = [".vecgrep/ignore"]
```

Supports the full gitignore pattern language — globs, directory patterns, and negation (`!keep-this.log`). The flag can be specified multiple times.

Traversal-related flags:

```bash
vecgrep -. "query"                              # include hidden files
vecgrep --skip-vcs -. "query"                  # include hidden, still skip .git/.hg/.jj
vecgrep --no-ignore "query"                    # ignore .gitignore/.ignore rules
vecgrep -L "query"                             # follow symlinks
vecgrep -d 3 "query"                           # cap traversal depth
```

## Index management

```bash
vecgrep --stats                                 # files, chunks, holes, DB size
vecgrep --reindex                               # force full re-index (always rebuilds from the project root; rejects paths/queries)
vecgrep --reindex --hybrid-index                # rebuild with lexical index support
vecgrep --index-only ./src                      # build index without searching
vecgrep --index-only --hybrid-index ./src       # build a hybrid-capable index without searching
vecgrep --clear-cache                           # delete cached index
vecgrep --show-root                             # print resolved project root
```

The index is a local cache. It rebuilds automatically when the schema version or embedding model changes. `Holes` are chunks whose remote embedding failed — they exist in the cache but never match queries.
Index capability follows the resolved `hybrid_index` setting. A plain `--reindex` rebuilds to that target configuration instead of preserving old index history.
Plain search and plain `--index-only` runs preserve an existing hybrid-capable index; they do not downgrade it.

For interactive and server mode, `--full-index` waits for indexing to finish before starting the UI or HTTP server:

```bash
vecgrep -i --full-index "query" ./src
vecgrep --serve --full-index ./src
```

## Embedding models

### Built-in: all-MiniLM-L6-v2

Ships inside the binary — no setup needed. 22M parameters, 384 dimensions, single-digit millisecond inference on CPU. Best score separation on our [benchmark](BENCHMARK.md).

### External models

For large codebases (1,000+ files), larger models improve accuracy:

```bash
vecgrep --embedder-url http://localhost:11434/v1/embeddings --embedder-model mxbai-embed-large "query"
```

Works with [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible API. Set it once in [config](CONFIG.md) to avoid repeating flags. The index rebuilds automatically when the model changes.

## Interactive and server mode

Use `-i` for the TUI and `--serve` for the HTTP API:

```bash
vecgrep -i "auth flow" ./src
vecgrep -i --query "auth flow" $(rg -l "auth" ./src)
vecgrep --serve ./src
vecgrep --serve --port 4123 ./src
```

- `--query` is mainly for `-i` and `--serve`, especially with `xargs`, because it forces all positionals to be treated as paths.
- `--open-cmd` sets the command used when opening a result from the TUI:

```bash
vecgrep -i --open-cmd "nvim +{line} {file}" "query"
vecgrep -i --open-cmd "bat -n --highlight-line {line}:{end_line} {file}" "query"
```

See also: [Server API](SERVER.md)

## Output and scripting

```bash
vecgrep --json "retry logic" ./src
vecgrep --color always "query" | less -R
vecgrep -p "query" | less -R                  # alias for --color=always
vecgrep -q "query" ./src                      # suppress progress/status on stderr
vecgrep --type-list                           # list supported `-t/--type` values
```

- `--json` emits JSONL with the project `root` included in every record.
- `--color auto|always|never` controls colored output explicitly.
- `-p/--pretty` forces color when piping.
- `-q/--quiet` suppresses indexing and status noise on stderr.
- `--type-list` prints the file-type names accepted by `-t/--type` and `-T/--type-not`.

## Complete flag reference

Search and ranking:

- `-k, --top-k <N>`: number of top results to return.
- `--threshold <0.0-1.0>`: minimum similarity score.
- `--hybrid`: combine semantic and lexical ranking.
- `--hybrid-index`: build lexical index support required by hybrid search.
- `--chunk-size <N>`: tokens per chunk.
- `--chunk-overlap <N>`: overlap between chunks.

Modes and lifecycle:

- `-i, --interactive`: start the TUI.
- `--serve`: start the HTTP server.
- `--port <PORT>`: fixed port for `--serve`; otherwise a free port is chosen.
- `--query <TEXT>`: explicit query for TUI/server mode; all positionals become paths.
- `--full-index`: finish indexing before opening the TUI/server.
- `--index-only`: build the index and exit.

Index and cache:

- `--reindex`: force a full rebuild.
- `--clear-cache`: delete `.vecgrep/index.db`.
- `--stats`: print index statistics.
- `--index-warn-threshold <N>`: prompt before indexing more than `N` files; `0` disables the prompt.
- `--show-root`: print the resolved project root.

Path selection and traversal:

- `-t, --type <NAME>`: include file type; repeatable.
- `-T, --type-not <NAME>`: exclude file type; repeatable.
- `-g, --glob <PATTERN>`: include matching paths; repeatable.
- `-., --hidden`: include hidden files and directories.
- `--skip-vcs`: still skip `.git`, `.hg`, and `.jj` when using `--hidden`.
- `--ignore-file <PATH>`: extra gitignore-format ignore file; repeatable.
- `--no-ignore`: ignore `.gitignore`, `.ignore`, and similar files.
- `-L, --follow`: follow symlinks.
- `-d, --max-depth <N>`: limit traversal depth.
- `--skip-outside-root`: ignore out-of-root paths instead of failing.
- `--no-scope`: search the full project index instead of the cwd or explicit-path scope.

Output control:

- `-l, --files-with-matches`: print file paths only.
- `-c, --count`: print count of matching chunks per file.
- `--json`: emit JSONL.
- `--color <auto|always|never>`: set color mode.
- `-p, --pretty`: alias for `--color=always`.
- `-q, --quiet`: suppress status output on stderr.
- `--type-list`: print supported file types.

Embedding backend:

- `--embedder-url <URL>`: use an OpenAI-compatible embeddings endpoint.
- `--embedder-model <NAME>`: model name for `--embedder-url`.

TUI integration:

- `--open-cmd <CMD>`: file opener template for TUI results. Supports `{file}`, `{line}`, and `{end_line}` placeholders.
