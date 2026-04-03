//! Embedding model benchmark suite.
//!
//! Tests retrieval quality on a curated dataset of code snippets and
//! technical text, with labeled relevance judgments.
//!
//! Run with: cargo test --test benchmark_models -- --nocapture

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::Deserialize;
use std::collections::HashMap;
use vecgrep::embedder::Embedder;
use vecgrep::index::{hybrid_fused_score, hybrid_lexical_query};

/// Create an embedder from env vars or fall back to local.
/// Set VECGREP_EMBEDDER_URL and VECGREP_EMBEDDER_MODEL to use a remote embedder.
fn make_embedder() -> Embedder {
    if let (Ok(url), Ok(model)) = (
        std::env::var("VECGREP_EMBEDDER_URL"),
        std::env::var("VECGREP_EMBEDDER_MODEL"),
    ) {
        eprintln!("Using remote embedder: {} ({})", url, model);
        Embedder::new_remote(&url, &model)
    } else {
        Embedder::new_local().unwrap()
    }
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetrievalMode {
    Vector,
    Lexical,
    Hybrid,
}

impl RetrievalMode {
    fn label(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Lexical => "lexical",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Default)]
struct Metrics {
    mrr: f32,
    recall_5: f32,
    recall_10: f32,
    ndcg_10: f32,
    queries: usize,
}

impl Metrics {
    fn record(&mut self, ranked: &[(usize, f32)], relevant_indices: &[usize]) {
        self.mrr += mrr(ranked, relevant_indices);
        self.recall_5 += recall_at_k(ranked, relevant_indices, 5);
        self.recall_10 += recall_at_k(ranked, relevant_indices, 10);
        self.ndcg_10 += ndcg_at_k(ranked, relevant_indices, 10);
        self.queries += 1;
    }

    fn mean_mrr(&self) -> f32 {
        self.mrr / self.queries as f32
    }

    fn mean_recall_5(&self) -> f32 {
        self.recall_5 / self.queries as f32
    }

    fn mean_recall_10(&self) -> f32 {
        self.recall_10 / self.queries as f32
    }

    fn mean_ndcg_10(&self) -> f32 {
        self.ndcg_10 / self.queries as f32
    }
}

const LEXICAL_TOP_K: usize = 50;
const VECTOR_TOP_K: usize = 50;
struct LexicalIndex {
    conn: Connection,
}

impl LexicalIndex {
    fn build(corpus: &[Document]) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute(
            "CREATE VIRTUAL TABLE corpus_fts USING fts5(doc_id UNINDEXED, text)",
            [],
        )?;

        {
            let mut stmt = conn.prepare("INSERT INTO corpus_fts(doc_id, text) VALUES (?1, ?2)")?;
            for (idx, doc) in corpus.iter().enumerate() {
                stmt.execute(params![idx as i64, &doc.text])?;
            }
        }

        Ok(Self { conn })
    }

    fn rank(&self, query: &str) -> Result<Vec<(usize, f32)>> {
        let Some(match_query) = hybrid_lexical_query(query) else {
            return Ok(vec![]);
        };

        let mut stmt = self.conn.prepare(
            "SELECT doc_id, bm25(corpus_fts) AS rank \
             FROM corpus_fts \
             WHERE corpus_fts MATCH ?1 \
             ORDER BY rank \
             LIMIT ?2",
        )?;
        let mut ranked = stmt
            .query_map(params![match_query, LEXICAL_TOP_K as i64], |row| {
                Ok((row.get::<_, i64>(0)? as usize, row.get::<_, f32>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        Ok(ranked)
    }
}

fn load_data() -> BenchmarkData {
    let data = include_str!("benchmark_data.json");
    serde_json::from_str(data).expect("failed to parse benchmark_data.json")
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn rank_vector(query_embedding: &[f32], corpus_embeddings: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = corpus_embeddings
        .iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_sim(query_embedding, doc_emb)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

fn rank_hybrid(
    query: &str,
    vector_ranked: &[(usize, f32)],
    lexical_ranked: &[(usize, f32)],
) -> Vec<(usize, f32)> {
    let mut merged: HashMap<usize, (Option<usize>, Option<usize>)> = HashMap::new();

    for (rank, (idx, _)) in vector_ranked.iter().take(VECTOR_TOP_K).enumerate() {
        merged
            .entry(*idx)
            .and_modify(|candidate| candidate.0 = Some(rank))
            .or_insert((Some(rank), None));
    }

    for (rank, (idx, _)) in lexical_ranked.iter().take(LEXICAL_TOP_K).enumerate() {
        merged
            .entry(*idx)
            .and_modify(|candidate| candidate.1 = Some(rank))
            .or_insert((None, Some(rank)));
    }

    let mut ranked: Vec<(usize, f32)> = merged
        .into_iter()
        .map(|(idx, (vector_rank, lexical_rank))| {
            (idx, hybrid_fused_score(query, vector_rank, lexical_rank))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    ranked
}

fn render_top5(ranked: &[(usize, f32)], relevant_indices: &[usize], corpus: &[Document]) -> String {
    ranked
        .iter()
        .take(5)
        .map(|(idx, score)| {
            let id = &corpus[*idx].id;
            let marker = if relevant_indices.contains(idx) {
                "✓"
            } else {
                " "
            };
            format!("{marker}{id}({score:.3})")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_summary(mode: RetrievalMode, metrics: &Metrics) {
    println!("  {}:", mode.label());
    println!("    MRR:      {:.3}", metrics.mean_mrr());
    println!("    R@5:      {:.3}", metrics.mean_recall_5());
    println!("    R@10:     {:.3}", metrics.mean_recall_10());
    println!("    NDCG@10:  {:.3}", metrics.mean_ndcg_10());
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
    let mut embedder = make_embedder();
    let lexical_index = LexicalIndex::build(&data.corpus).unwrap();

    // Embed corpus
    let corpus_texts: Vec<&str> = data.corpus.iter().map(|d| d.text.as_str()).collect();
    let corpus_embeddings = embedder.embed_batch(&corpus_texts).unwrap();

    let id_to_idx: HashMap<&str, usize> = data
        .corpus
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.as_str(), i))
        .collect();

    let mut metrics_by_mode: HashMap<&'static str, Metrics> = HashMap::from([
        (RetrievalMode::Vector.label(), Metrics::default()),
        (RetrievalMode::Lexical.label(), Metrics::default()),
        (RetrievalMode::Hybrid.label(), Metrics::default()),
    ]);
    let mut top1_wins: HashMap<&'static str, usize> = HashMap::from([
        (RetrievalMode::Vector.label(), 0),
        (RetrievalMode::Lexical.label(), 0),
        (RetrievalMode::Hybrid.label(), 0),
    ]);

    println!(
        "\n=== Retrieval Benchmark ({} queries, {} corpus docs, modes: vector / lexical / hybrid) ===\n",
        data.queries.len(),
        data.corpus.len()
    );

    for query in &data.queries {
        let query_emb = embedder.embed(&query.text).unwrap();
        let vector_ranked = rank_vector(&query_emb, &corpus_embeddings);
        let lexical_ranked = lexical_index.rank(&query.text).unwrap();
        let hybrid_ranked = rank_hybrid(&query.text, &vector_ranked, &lexical_ranked);

        let relevant_indices: Vec<usize> = query
            .relevant
            .iter()
            .filter_map(|id| id_to_idx.get(id.as_str()).copied())
            .collect();

        metrics_by_mode
            .get_mut(RetrievalMode::Vector.label())
            .unwrap()
            .record(&vector_ranked, &relevant_indices);
        metrics_by_mode
            .get_mut(RetrievalMode::Lexical.label())
            .unwrap()
            .record(&lexical_ranked, &relevant_indices);
        metrics_by_mode
            .get_mut(RetrievalMode::Hybrid.label())
            .unwrap()
            .record(&hybrid_ranked, &relevant_indices);

        let per_mode = [
            (RetrievalMode::Vector, &vector_ranked),
            (RetrievalMode::Lexical, &lexical_ranked),
            (RetrievalMode::Hybrid, &hybrid_ranked),
        ];

        let best_mode = per_mode
            .iter()
            .map(|(mode, ranked)| (*mode, mrr(ranked, &relevant_indices)))
            .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap())
            .unwrap()
            .0;
        *top1_wins.get_mut(best_mode.label()).unwrap() += 1;

        println!("  {}: {}", query.id, query.text);
        for (mode, ranked) in per_mode {
            let q_mrr = mrr(ranked, &relevant_indices);
            let q_r5 = recall_at_k(ranked, &relevant_indices, 5);
            println!(
                "    {:>7} [MRR={q_mrr:.2} R@5={q_r5:.2}] {}",
                mode.label(),
                render_top5(ranked, &relevant_indices, &data.corpus)
            );
        }
    }

    println!("\n=== Summary ===");
    print_summary(
        RetrievalMode::Vector,
        metrics_by_mode.get(RetrievalMode::Vector.label()).unwrap(),
    );
    print_summary(
        RetrievalMode::Lexical,
        metrics_by_mode.get(RetrievalMode::Lexical.label()).unwrap(),
    );
    print_summary(
        RetrievalMode::Hybrid,
        metrics_by_mode.get(RetrievalMode::Hybrid.label()).unwrap(),
    );
    println!("  top-MRR wins:");
    println!(
        "    vector={} lexical={} hybrid={}",
        top1_wins[RetrievalMode::Vector.label()],
        top1_wins[RetrievalMode::Lexical.label()],
        top1_wins[RetrievalMode::Hybrid.label()]
    );
    println!();

    let vector_metrics = metrics_by_mode.get(RetrievalMode::Vector.label()).unwrap();
    let hybrid_metrics = metrics_by_mode.get(RetrievalMode::Hybrid.label()).unwrap();

    // Quality gates
    assert!(
        vector_metrics.mean_mrr() >= 0.50,
        "MRR {:.3} too low (expected >= 0.50)",
        vector_metrics.mean_mrr()
    );
    assert!(
        vector_metrics.mean_recall_5() >= 0.35,
        "R@5 {:.3} too low (expected >= 0.35)",
        vector_metrics.mean_recall_5()
    );
    assert!(
        vector_metrics.mean_recall_10() >= 0.50,
        "R@10 {:.3} too low (expected >= 0.50)",
        vector_metrics.mean_recall_10()
    );
    assert!(
        vector_metrics.mean_ndcg_10() >= 0.45,
        "NDCG@10 {:.3} too low (expected >= 0.45)",
        vector_metrics.mean_ndcg_10()
    );
    assert!(
        hybrid_metrics.mean_mrr() >= 0.50,
        "Hybrid MRR {:.3} too low (expected >= 0.50)",
        hybrid_metrics.mean_mrr()
    );
    assert!(
        hybrid_metrics.mean_recall_10() >= 0.50,
        "Hybrid R@10 {:.3} too low (expected >= 0.50)",
        hybrid_metrics.mean_recall_10()
    );
    assert!(
        hybrid_metrics.mean_mrr() + 0.08 >= vector_metrics.mean_mrr(),
        "Hybrid MRR {:.3} regressed too far below vector {:.3}",
        hybrid_metrics.mean_mrr(),
        vector_metrics.mean_mrr()
    );
    assert!(
        hybrid_metrics.mean_ndcg_10() + 0.08 >= vector_metrics.mean_ndcg_10(),
        "Hybrid NDCG@10 {:.3} regressed too far below vector {:.3}",
        hybrid_metrics.mean_ndcg_10(),
        vector_metrics.mean_ndcg_10()
    );
}

#[test]
fn benchmark_relevance_separation() {
    let mut embedder = make_embedder();

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
