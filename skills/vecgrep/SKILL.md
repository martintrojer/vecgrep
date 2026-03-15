---
name: vecgrep
description: Explain how to use vecgrep effectively for semantic search, indexing, filtering, TUI/server modes, and troubleshooting. Use when Codex needs to answer questions about vecgrep commands, flag selection, project-root behavior, local vs remote embedders, reindexing, stats, or common "why didn't this search/index work?" cases.
---

# Vecgrep Usage

Give practical, command-first guidance. Prefer short examples over long prose.

## Start Here

Treat these as the main user-facing modes:

- Plain CLI search: index first, then search the up-to-date index.
- `--index-only`: build or refresh the index without searching.
- `--reindex`: rebuild the index from scratch, with or without a query.
- `--stats`: report indexed files, chunks, holes, and DB size.
- `--interactive`: run the TUI with progressive indexing unless `--full-index` is set.
- `--serve`: run the HTTP server with progressive indexing unless `--full-index` is set.

For concrete commands, read [references/commands.md](references/commands.md).

## Explain Indexing Correctly

Be precise about these behaviors:

- Normal CLI searches wait for indexing to finish before searching.
- `--full-index` only changes interactive/server behavior; plain CLI already waits.
- `--reindex` clears cached index data and rebuilds it.
- Config changes that affect embeddings or chunking rebuild the cache automatically.
- `--stats` includes `Holes`, which are chunks stored with zero-vector embeddings after remote embedding failures.

If a user expects progressive partial results from a normal CLI search, correct that expectation explicitly.

## Explain Roots And Paths

Vecgrep is single-root by design.

- The project root is discovered by walking up for `.git/`, `.hg/`, `.jj/`, or `.vecgrep/`.
- The cache lives at `.vecgrep/index.db` under that root.
- Paths outside the selected root fail by default.
- `--skip-outside-root` ignores outside-root paths instead of failing.
- `--show-root` prints the resolved root.

When debugging missing files, check root selection before assuming indexing is broken.

## Explain Embedder Modes

Distinguish the two embedder paths:

- Local embedder: built-in ONNX model, tokenizer-aware chunking, silent model truncation, no index holes from embedding failures.
- Remote embedder: OpenAI-compatible API via `--embedder-url` and `--embedder-model`, heuristic chunk sizing, possible HTTP context-limit failures, and possible holes if fallback embedding attempts still fail.

When users report remote embedding failures, mention context-length limits and the `Holes` count from `--stats`.

## Troubleshooting Workflow

Use this order:

1. Confirm the root with `--show-root`.
2. Check index state with `--stats`.
3. If the cache may be stale or corrupted, suggest `--reindex`.
4. If the user changed model or chunk settings, explain that a rebuild is expected.
5. If using a remote embedder, check server URL, model name, and context limits.
6. If a path is missing, verify it is inside the selected root and not ignored by flags or ignore files.

## Response Style

Prefer:

- exact commands
- short explanations of why a flag matters
- explicit corrections when users confuse `--full-index`, `--reindex`, or root scoping

Avoid:

- implementation-detail dumps unless the user asks
- generic vector search explanations
- suggesting multi-root behavior that vecgrep does not support
