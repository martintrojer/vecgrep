//! Embedding model benchmark suite.
//!
//! Tests retrieval quality on a curated dataset of code snippets and
//! technical text, with labeled relevance judgments.
//!
//! Run with: cargo test --test benchmark_models -- --nocapture

use serde::Deserialize;
use vecgrep::embedder::Embedder;

#[derive(Deserialize)]
struct BenchmarkData {
    corpus: Vec<Document>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    #[serde(rename = "type")]
    _doc_type: String,
    text: String,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    relevant: Vec<String>,
}

fn load_data() -> BenchmarkData {
    let data = include_str!("benchmark_data.json");
    serde_json::from_str(data).expect("failed to parse benchmark_data.json")
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Mean Reciprocal Rank: 1/rank of the first relevant result.
fn mrr(ranked: &[(usize, f32)], relevant_indices: &[usize]) -> f32 {
    for (rank_pos, (idx, _)) in ranked.iter().enumerate() {
        if relevant_indices.contains(idx) {
            return 1.0 / (rank_pos as f32 + 1.0);
        }
    }
    0.0
}

/// Recall at k: fraction of relevant documents found in top-k.
fn recall_at_k(ranked: &[(usize, f32)], relevant_indices: &[usize], k: usize) -> f32 {
    let found = ranked
        .iter()
        .take(k)
        .filter(|(idx, _)| relevant_indices.contains(idx))
        .count();
    found as f32 / relevant_indices.len() as f32
}

/// Normalized Discounted Cumulative Gain at k.
fn ndcg_at_k(ranked: &[(usize, f32)], relevant_indices: &[usize], k: usize) -> f32 {
    let dcg: f32 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, (idx, _))| {
            let rel = if relevant_indices.contains(idx) {
                1.0
            } else {
                0.0
            };
            rel / (rank as f32 + 2.0).log2()
        })
        .sum();

    // Ideal DCG: all relevant docs at the top
    let ideal_count = relevant_indices.len().min(k);
    let idcg: f32 = (0..ideal_count)
        .map(|rank| 1.0 / (rank as f32 + 2.0).log2())
        .sum();

    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

#[test]
fn benchmark_retrieval() {
    let data = load_data();
    let mut embedder = Embedder::new().unwrap();

    // Embed corpus
    let corpus_texts: Vec<&str> = data.corpus.iter().map(|d| d.text.as_str()).collect();
    let corpus_embeddings = embedder.embed_batch(&corpus_texts).unwrap();

    let id_to_idx: std::collections::HashMap<&str, usize> = data
        .corpus
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.as_str(), i))
        .collect();

    let mut total_mrr = 0.0;
    let mut total_recall_5 = 0.0;
    let mut total_recall_10 = 0.0;
    let mut total_ndcg_10 = 0.0;
    let mut query_count = 0;

    println!(
        "\n=== Retrieval Benchmark ({} queries, {} corpus docs) ===\n",
        data.queries.len(),
        data.corpus.len()
    );

    for query in &data.queries {
        let query_emb = embedder.embed(&query.text).unwrap();

        let mut ranked: Vec<(usize, f32)> = corpus_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| (i, cosine_sim(&query_emb, doc_emb)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let relevant_indices: Vec<usize> = query
            .relevant
            .iter()
            .filter_map(|id| id_to_idx.get(id.as_str()).copied())
            .collect();

        let q_mrr = mrr(&ranked, &relevant_indices);
        let q_r5 = recall_at_k(&ranked, &relevant_indices, 5);
        let q_r10 = recall_at_k(&ranked, &relevant_indices, 10);
        let q_ndcg = ndcg_at_k(&ranked, &relevant_indices, 10);

        total_mrr += q_mrr;
        total_recall_5 += q_r5;
        total_recall_10 += q_r10;
        total_ndcg_10 += q_ndcg;
        query_count += 1;

        let top5: Vec<String> = ranked
            .iter()
            .take(5)
            .map(|(idx, score)| {
                let id = &data.corpus[*idx].id;
                let marker = if relevant_indices.contains(idx) {
                    "✓"
                } else {
                    " "
                };
                format!("{marker}{id}({score:.3})")
            })
            .collect();

        println!(
            "  [MRR={q_mrr:.2} R@5={q_r5:.2}] {}: {}",
            query.id, query.text
        );
        println!("    top5: {}", top5.join(", "));
    }

    let n = query_count as f32;
    let mean_mrr = total_mrr / n;
    let mean_r5 = total_recall_5 / n;
    let mean_r10 = total_recall_10 / n;
    let mean_ndcg = total_ndcg_10 / n;

    println!("\n=== Summary ===");
    println!("  MRR:      {mean_mrr:.3}");
    println!("  R@5:      {mean_r5:.3}");
    println!("  R@10:     {mean_r10:.3}");
    println!("  NDCG@10:  {mean_ndcg:.3}");
    println!();

    // Quality gates
    assert!(
        mean_mrr >= 0.50,
        "MRR {mean_mrr:.3} too low (expected >= 0.50)"
    );
    assert!(
        mean_r5 >= 0.35,
        "R@5 {mean_r5:.3} too low (expected >= 0.35)"
    );
    assert!(
        mean_r10 >= 0.50,
        "R@10 {mean_r10:.3} too low (expected >= 0.50)"
    );
    assert!(
        mean_ndcg >= 0.45,
        "NDCG@10 {mean_ndcg:.3} too low (expected >= 0.45)"
    );
}

#[test]
fn benchmark_relevance_separation() {
    let mut embedder = Embedder::new().unwrap();

    println!("\n=== Relevance Separation ===\n");

    let similar_pairs: &[(&str, &str)] = &[
        ("error handling", "exception management and recovery"),
        ("database query", "SQL select statement execution"),
        ("HTTP server", "web server listening on a port"),
        ("user authentication", "login with password verification"),
        ("retry with backoff", "exponential delay between attempts"),
        ("file hashing", "compute SHA digest of contents"),
        ("connection pool", "reuse database connections"),
        ("rate limiting", "throttle API requests per client"),
    ];

    let dissimilar_pairs: &[(&str, &str)] = &[
        ("error handling", "chocolate cake recipe"),
        ("database query", "weather forecast for tomorrow"),
        ("HTTP server", "gardening tips for spring"),
        ("user authentication", "history of ancient Rome"),
        ("retry with backoff", "painting a watercolor landscape"),
        ("file hashing", "how to train a puppy"),
        ("connection pool", "knitting patterns for beginners"),
        ("rate limiting", "best hiking trails in Colorado"),
    ];

    let mut sim_scores = Vec::new();
    let mut dissim_scores = Vec::new();

    for (a, b) in similar_pairs {
        let ea = embedder.embed(a).unwrap();
        let eb = embedder.embed(b).unwrap();
        let score = cosine_sim(&ea, &eb);
        sim_scores.push(score);
        println!("  similar:    {score:.3}  \"{a}\" ↔ \"{b}\"");
    }

    for (a, b) in dissimilar_pairs {
        let ea = embedder.embed(a).unwrap();
        let eb = embedder.embed(b).unwrap();
        let score = cosine_sim(&ea, &eb);
        dissim_scores.push(score);
        println!("  dissimilar: {score:.3}  \"{a}\" ↔ \"{b}\"");
    }

    let avg_sim: f32 = sim_scores.iter().sum::<f32>() / sim_scores.len() as f32;
    let avg_dissim: f32 = dissim_scores.iter().sum::<f32>() / dissim_scores.len() as f32;
    let separation = avg_sim - avg_dissim;

    let min_sim = sim_scores.iter().cloned().fold(f32::MAX, f32::min);
    let max_dissim = dissim_scores.iter().cloned().fold(f32::MIN, f32::max);

    println!(
        "\n  Avg similar: {avg_sim:.3}  Avg dissimilar: {avg_dissim:.3}  Separation: {separation:.3}"
    );
    println!("  Min similar: {min_sim:.3}  Max dissimilar: {max_dissim:.3}");
    println!();

    assert!(
        separation >= 0.15,
        "Separation {separation:.3} too low (expected >= 0.15)"
    );
}
