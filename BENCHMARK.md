# Embedding Model Benchmarks

Results from `cargo test --test benchmark_models -- --nocapture`.

The benchmark suite tests retrieval quality on a curated dataset of 122 corpus documents (code snippets across Python/Rust/Go/JS/Java/C + technical text passages) with 50 queries and labeled relevance judgments. 10 hard negatives are included — documents that share keywords with a concept but serve a different purpose.

## Results

| Metric | all-MiniLM-L6-v2 | bge-small-en-v1.5 | arctic-embed-s | arctic-embed-xs |
|---|---|---|---|---|
| **MRR** | 0.933 | **0.947** | 0.912 | 0.857 |
| **R@5** | **0.858** | 0.811 | 0.785 | 0.767 |
| **R@10** | **0.941** | 0.893 | 0.848 | 0.822 |
| **NDCG@10** | **0.893** | 0.866 | 0.829 | 0.770 |
| **Separation** | **0.505** | 0.330 | 0.224 | 0.188 |
| Params | 22M | 33M | 33M | 22M |
| ONNX size | ~90 MB | ~130 MB | ~130 MB | ~85 MB |
| Max tokens | 256 | 512 | 512 | 512 |
| MTEB Retrieval | 41.95 | 51.7 | 52.0 | 50.1 |

## Analysis

**all-MiniLM-L6-v2** leads on recall (R@5, R@10), ranking quality (NDCG@10), and separation — the metrics that matter most for code search where you want all relevant results surfaced reliably.

**bge-small-en-v1.5** has the best MRR (0.947) — slightly better at putting the single best result first. But it trails on recall and separation, and is 40 MB larger.

**snowflake-arctic-embed-s** trails on all metrics despite the highest MTEB score. Its weak separation (0.224) means thresholding is less reliable.

**snowflake-arctic-embed-xs** is the weakest across the board. Hard negatives trip it up more than the others.

The MTEB retrieval benchmarks test on large, diverse document corpora. Our use case is different: mixed code/prose corpus with short queries and hard negatives that share vocabulary but differ in purpose. MiniLM's much wider separation (0.505 vs next-best 0.330) is the key differentiator.

## Current model

**all-MiniLM-L6-v2** — best recall, NDCG, and separation at the smallest binary size.

## Models not benchmarked (require code changes)

- **MongoDB/mdbr-leaf-ir**: 1024-dim output with Dense projection layer. ONNX split into two files. Needs embedder changes for Dense layer and different dims.
- **IBM granite-embedding-30m-english**: RoBERTa-based (no token_type_ids), uses CLS pooling instead of mean pooling. Needs embedder changes for different pooling and input format.

## Running benchmarks

**Important**: when swapping models for benchmarking, you must clear both the download cache and the build artifacts to avoid stale model files:

```bash
rm -rf ~/Library/Caches/vecgrep/models/
cargo clean
cargo test --test benchmark_models -- --nocapture
```

The download cache (`~/Library/Caches/vecgrep/models/`) uses filename-based caching (`model.onnx`), so different models from different URLs collide. The build artifacts in `target/` also cache the compiled-in model bytes via `include_bytes!`.

## Methodology

The benchmark tests two capabilities:

1. **Retrieval** (50 queries over 122 documents): measures MRR, R@5, R@10, and NDCG@10 against labeled relevance judgments. Corpus includes code in 6 languages, technical prose, and 10 hard negatives. Queries range from natural language descriptions to concept searches.
2. **Relevance separation** (8 similar + 8 dissimilar pairs): measures the gap between scores for semantically related vs unrelated text pairs. Higher separation means more reliable thresholding.

Quality gates: MRR ≥ 0.50, R@5 ≥ 0.35, R@10 ≥ 0.50, NDCG@10 ≥ 0.45, separation ≥ 0.15.
