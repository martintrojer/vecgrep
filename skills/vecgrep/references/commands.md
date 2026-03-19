# Vecgrep Commands

## Search

```bash
vecgrep "error handling" ./src                 # semantic search
vecgrep "match Ok(v) => v" ./src               # code snippet as query
vecgrep "query" -t rust                        # filter by type
vecgrep "query" -T markdown                    # exclude type
vecgrep "query" -g '*.rs'                      # glob filter
vecgrep "query" --threshold 0.1               # lower threshold for more results
vecgrep --no-scope "query"                     # search entire project (ignore cwd scope)
```

## Output Modes

```bash
vecgrep -l "query"                             # file paths only
vecgrep -c "query"                             # count per file
vecgrep --json "query"                         # JSONL for scripting
vecgrep -p "query"                             # force colors when piping
```

## Interactive / Server

```bash
vecgrep -i "query"                             # TUI mode
vecgrep -i --full-index                        # TUI after full index
vecgrep --serve --port 8080                    # HTTP server
```

## Index Management

```bash
vecgrep --stats                                # index state
vecgrep --reindex                              # force rebuild
vecgrep --clear-cache                          # delete index (keeps config)
vecgrep --index-only ./src                     # build without searching
vecgrep --show-root                            # print project root
```

## Combining With Other Tools

```bash
vecgrep -l "error handling" | xargs rg "unwrap"
rg -l "TODO" | xargs vecgrep "technical debt"
vecgrep --json "auth" | jq -r '.file' | sort -u | xargs git blame
vecgrep --json "query" | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'
vecgrep -l "error handling" | entr -r cargo test
```

## Remote Embedder

```bash
vecgrep --embedder-url http://localhost:11434/v1/embeddings \
        --embedder-model mxbai-embed-large "query"
```

## Troubleshooting

```bash
vecgrep --show-root                            # verify root
vecgrep --stats                                # check holes, file count
vecgrep --reindex                              # rebuild if stale
vecgrep --skip-outside-root "query" ../other   # ignore outside-root paths
```
