# Embedding Model Benchmarks

## Small-scale benchmark (curated, 122 docs)

Results from `cargo test --test benchmark_models -- --nocapture`.

122 corpus documents (code in 6 languages + technical text + 10 hard negatives), 50 queries with labeled relevance judgments.

### Built-in model candidates

| Metric | all-MiniLM-L6-v2 | bge-small-en-v1.5 | arctic-embed-s | arctic-embed-xs |
|---|---|---|---|---|
| **MRR** | 0.933 | **0.947** | 0.912 | 0.857 |
| **R@5** | **0.858** | 0.811 | 0.785 | 0.767 |
| **R@10** | **0.941** | 0.893 | 0.848 | 0.822 |
| **NDCG@10** | **0.893** | 0.866 | 0.829 | 0.770 |
| **Separation** | **0.505** | 0.330 | 0.224 | 0.188 |
| ONNX size | ~90 MB | ~130 MB | ~130 MB | ~85 MB |

### Ollama models (via `--embedder-url`)

| Metric | MiniLM (built-in) | mxbai-embed-large | embeddinggemma | arctic-embed2 | nomic-v2-moe |
|---|---|---|---|---|---|
| **MRR** | 0.933 | **0.946** | 0.918 | 0.922 | 0.907 |
| **R@5** | 0.858 | 0.805 | **0.881** | 0.846 | 0.804 |
| **NDCG@10** | **0.893** | 0.875 | **0.893** | 0.880 | 0.818 |
| **Separation** | **0.505** | 0.390 | 0.359 | 0.414 | 0.271 |

At small scale, MiniLM wins on recall, NDCG, and separation. Its wide separation (0.505) makes thresholding reliable.

## Large-scale benchmark (CodeSearchNet, ~6,500 docs)

Results from `cargo test --test benchmark_large -- --ignored --nocapture`.

~6,500 code functions (Python, JavaScript, Go) from CodeSearchNet with documentation strings as queries. 1,000 queries evaluated. **All texts truncated to 1024 chars** for fair cross-model comparison (matching MiniLM's effective 256-token context).

| Metric | MiniLM (built-in) | mxbai-embed-large | embeddinggemma | nomic-v2-moe | arctic-embed2 |
|---|---|---|---|---|---|
| **MRR** | 0.929 | **0.957** | 0.956 | 0.956 | 0.953 |
| **R@1** | 0.892 | **0.930** | 0.928 | 0.927 | 0.923 |
| **R@5** | 0.977 | 0.991 | **0.993** | 0.990 | 0.990 |
| **R@10** | 0.990 | 0.997 | **0.998** | **0.998** | 0.997 |
| **R@50** | 0.996 | 0.999 | 0.999 | **1.000** | 0.999 |
| **R@100** | 0.998 | **1.000** | 0.999 | **1.000** | 0.999 |
| Failures | 0 | 0 | 0 | 0 | 0 |

At scale, **all Ollama models beat MiniLM** by ~3 points on MRR. The larger models produce richer representations that discriminate better in a large corpus with many similar-looking distractors. The four Ollama models are remarkably close to each other.

## Analysis

**Scale matters.** MiniLM wins at 122 docs (better separation), but loses at 6,500 docs (weaker representations). The crossover likely happens around 500-1,000 documents.

**Current model choice: all-MiniLM-L6-v2** — best self-contained option. 90 MB embedded in binary, no external dependencies, ~30s to index 6,500 docs. R@5 of 0.977 is excellent for most codebases.

**For large codebases (1,000+ files): consider `--embedder-url`** with Ollama running mxbai-embed-large or embeddinggemma. ~3% better MRR at the cost of slower indexing (~5 min vs 30s for 6,500 docs).

## Models not benchmarked (require code changes)

- **MongoDB/mdbr-leaf-ir**: 1024-dim output with Dense projection layer. ONNX split into two files.
- **IBM granite-embedding-30m-english**: RoBERTa-based (no token_type_ids), CLS pooling.
- **qwen3-embedding**: Produces NaN embeddings for code snippets. Not suitable.

## Running benchmarks

**Small benchmark (curated):**
```bash
cargo test --test benchmark_models -- --nocapture
```

**Small benchmark with Ollama:**
```bash
VECGREP_EMBEDDER_URL=http://localhost:11434/v1/embeddings \
VECGREP_EMBEDDER_MODEL=mxbai-embed-large \
cargo test --test benchmark_models -- --nocapture
```

**Large benchmark (CodeSearchNet, downloads from HuggingFace on first run):**
```bash
cargo test --test benchmark_large -- --ignored --nocapture
```

**Large benchmark with Ollama:**
```bash
VECGREP_EMBEDDER_URL=http://localhost:11434/v1/embeddings \
VECGREP_EMBEDDER_MODEL=mxbai-embed-large \
cargo test --test benchmark_large -- --ignored --nocapture
```

**Swapping built-in model:** clear both caches to avoid stale files:
```bash
rm -rf ~/Library/Caches/vecgrep/models/
cargo clean
cargo test --test benchmark_models -- --nocapture
```

## Methodology

**Small benchmark** tests two capabilities:
1. **Retrieval** (50 queries over 122 documents): MRR, R@5, R@10, NDCG@10.
2. **Relevance separation** (8 similar + 8 dissimilar pairs): score gap between related and unrelated text.

**Large benchmark** tests retrieval at scale:
- 6,500 code functions from CodeSearchNet (Python, JavaScript, Go)
- Documentation strings as queries, code functions as relevant documents
- All texts truncated to 1024 chars for fair cross-model comparison
- Measures MRR, R@1, R@5, R@10, R@50, R@100
