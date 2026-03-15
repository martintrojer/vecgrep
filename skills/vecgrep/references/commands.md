# Vecgrep Commands

Use these examples when answering "how do I use vecgrep?" questions.

## Common Commands

```bash
# Search the current project
vecgrep "error handling"

# Search a specific directory
vecgrep "authentication flow" src/

# Build or refresh the index without searching
vecgrep --index-only .

# Force a full rebuild, then search
vecgrep --reindex "query text" .

# Force a full rebuild without searching
vecgrep --reindex .

# Show index statistics
vecgrep --stats .

# Show the resolved project root
vecgrep --show-root
```

## Filtering And Output

```bash
# Restrict by file type
vecgrep "database migrations" -t rust -t sql

# Exclude a file type
vecgrep "frontend state" -T minified

# Restrict by glob
vecgrep "render pipeline" -g '*.rs' -g '!target/**'

# JSONL output for scripting
vecgrep --json "login form"

# Only print matching files
vecgrep -l "oauth callback"

# Print counts per file
vecgrep -c "retry logic"
```

## Interactive And Server Modes

```bash
# Start the TUI
vecgrep --interactive

# Start the TUI after a full upfront index
vecgrep --interactive --full-index

# Start the HTTP server
vecgrep --serve

# Start the server after a full upfront index
vecgrep --serve --full-index
```

## Remote Embedder

```bash
vecgrep \
  --embedder-url http://localhost:11434/v1/embeddings \
  --embedder-model mxbai-embed-large \
  "query text"
```

## Troubleshooting Shortcuts

```bash
# Rebuild the cache if results seem stale
vecgrep --reindex .

# Check for failed remote embeddings
vecgrep --stats .

# Ignore outside-root paths instead of failing
vecgrep --skip-outside-root "query" ../other-path
```
