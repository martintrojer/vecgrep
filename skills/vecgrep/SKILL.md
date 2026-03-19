---
name: vecgrep
description: Use vecgrep for semantic code search — finding code by meaning, not just text. Use when you need to locate relevant code, understand a codebase, find similar patterns, or answer questions about what code does. Also covers indexing, filtering, troubleshooting, and configuration.
---

# Vecgrep — Semantic Code Search

vecgrep searches codebases by meaning using local embeddings. Use it when grep/ripgrep would require knowing the exact words, but you know the concept.

## When to Use

- **Finding code by concept**: "error handling", "authentication flow", "database connection pooling"
- **Finding similar patterns**: pass a code snippet as the query
- **Exploring unfamiliar codebases**: "how does this project handle configuration?"
- **Narrowing before exact search**: `vecgrep -l "auth" | xargs rg "token"`

## Quick Reference

```bash
# Semantic search
vecgrep "error handling for network timeouts" ./src

# Code snippet as query
vecgrep "match result { Ok(v) => v, Err(e) => return Err(e) }" ./src

# Filter by file type
vecgrep "sorting algorithm" -t rust

# Files only (for piping)
vecgrep -l "retry logic" ./src | xargs rg "unwrap"

# JSONL for structured output
vecgrep --json "authentication" ./src

# Search entire project from any subdirectory
vecgrep --no-scope "startup"

# Interactive TUI
vecgrep -i "query"
```

## Combining With Other Tools

```bash
# Semantic find → exact grep
vecgrep -l "error handling" ./src | xargs rg "unwrap"

# Exact find → semantic rank
rg -l "TODO" ./src | xargs vecgrep "technical debt"

# Semantic find → git blame
vecgrep --json "auth" ./src | jq -r '.file' | sort -u | xargs git blame

# Structured extraction
vecgrep --json "error handling" ./src | jq -r 'select(.score > 0.5) | "\(.file):\(.start_line)"'
```

## Index Management

```bash
vecgrep --stats              # check index state
vecgrep --reindex            # force full rebuild
vecgrep --clear-cache        # delete cached index (preserves config)
vecgrep --index-only ./src   # build index without searching
vecgrep --show-root          # print resolved project root
```

## Key Behaviors

- **Single root**: project root discovered from `.git/`, `.hg/`, `.jj/`, or `.vecgrep/`. Cache at `.vecgrep/index.db`.
- **Path scoping**: `vecgrep "query" src/` only returns `src/` results. `--no-scope` searches the full index.
- **Content hashing**: only changed files are re-embedded. Config/model changes trigger full rebuild.
- **Default threshold**: 0.2. Lower it (`--threshold 0.1`) if getting too few results on small/code-heavy repos.

## Filtering

```bash
vecgrep "query" -t rust -t python      # include types
vecgrep "query" -T markdown            # exclude types
vecgrep "query" -g '*.rs'              # glob filter
vecgrep "query" -l                     # file paths only
vecgrep "query" -c                     # count per file
vecgrep "query" --json                 # JSONL output
```

## Troubleshooting

1. Check root: `vecgrep --show-root`
2. Check index: `vecgrep --stats`
3. Stale results? `vecgrep --reindex`
4. Too few results? Lower threshold: `--threshold 0.1`
5. Missing files? Check they're inside the root and not gitignored
