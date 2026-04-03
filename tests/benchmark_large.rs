//! Large-scale embedding model benchmark using CodeSearchNet from HuggingFace.
//!
//! Downloads ~9,000 code functions with their documentation strings.
//! Each documentation string is a query, its code function is the relevant document.
//!
//! All texts are truncated to the same character limit (1024 chars, matching
//! MiniLM's effective 256-token context) for fair cross-model comparison.
//!
//! Run with: cargo test --test benchmark_large -- --ignored --nocapture
//! Set VECGREP_EMBEDDER_URL and VECGREP_EMBEDDER_MODEL for remote embedders.
//!
//! First run downloads data from HuggingFace (~30s). Cached in target/benchmark_cache/.

use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::PathBuf;
use vecgrep::embedder::Embedder;
use vecgrep::index::{hybrid_fused_score, hybrid_lexical_query};

const DATASET_API: &str = "https://datasets-server.huggingface.co/rows";
const ROWS_PER_PAGE: usize = 100;
const TARGET_CODE_SIZE: usize = 9_000;
const CACHE_FILE: &str = "target/benchmark_cache/codesearchnet_v3.json";
const LEXICAL_TOP_K: usize = 100;
const VECTOR_TOP_K: usize = 100;

/// Fixed truncation limit for fair comparison across all models.
/// 1024 chars ≈ 256 tokens (MiniLM's context window).
const MAX_TEXT_CHARS: usize = 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CachedData {
    corpus: Vec<CorpusDoc>,
    queries: Vec<QueryDoc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CorpusDoc {
    text: String,
    kind: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct QueryDoc {
    text: String,
    relevant_idx: usize,
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
    total_mrr: f32,
    recall_at_1: usize,
    recall_at_5: usize,
    recall_at_10: usize,
    recall_at_50: usize,
    recall_at_100: usize,
    queries: usize,
}

impl Metrics {
    fn record_rank(&mut self, rank: Option<usize>) {
        self.queries += 1;
        if let Some(r) = rank {
            self.total_mrr += 1.0 / r as f32;
            if r <= 1 {
                self.recall_at_1 += 1;
            }
            if r <= 5 {
                self.recall_at_5 += 1;
            }
            if r <= 10 {
                self.recall_at_10 += 1;
            }
            if r <= 50 {
                self.recall_at_50 += 1;
            }
            if r <= 100 {
                self.recall_at_100 += 1;
            }
        }
    }

    fn mean_mrr(&self) -> f32 {
        self.total_mrr / self.queries as f32
    }

    fn recall_at(&self, k: usize) -> f32 {
        let hits = match k {
            1 => self.recall_at_1,
            5 => self.recall_at_5,
            10 => self.recall_at_10,
            50 => self.recall_at_50,
            100 => self.recall_at_100,
            _ => panic!("unsupported recall@{k}"),
        };
        hits as f32 / self.queries as f32
    }
}

struct LexicalIndex {
    conn: Connection,
}

impl LexicalIndex {
    fn build(corpus: &[CorpusDoc]) -> Result<Self> {
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

fn cache_path() -> PathBuf {
    PathBuf::from(CACHE_FILE)
}

fn fetch_pages(
    agent: &ureq::Agent,
    dataset: &str,
    config: &str,
    split: &str,
    max_rows: usize,
    label: &str,
    filter: fn(&serde_json::Value) -> Option<(String, String, String)>,
) -> Vec<(String, String, String)> {
    let mut results: Vec<(String, String, String)> = Vec::new();
    let mut offset = 0;
    let mut retries = 0;

    while results.len() < max_rows {
        let url = format!(
            "{}?dataset={}&config={}&split={}&offset={}&length={}",
            DATASET_API, dataset, config, split, offset, ROWS_PER_PAGE
        );

        eprint!("\r  {} {}/{}...", label, results.len(), max_rows);

        match agent.get(&url).call() {
            Ok(resp) => {
                retries = 0;
                let body = resp.into_body().read_to_string().unwrap_or_default();
                if let Ok(api_resp) = serde_json::from_str::<serde_json::Value>(&body) {
                    let rows = api_resp["rows"].as_array();
                    if rows.is_none_or(|r| r.is_empty()) {
                        break;
                    }
                    for row in rows.unwrap() {
                        if let Some(triple) = filter(&row["row"]) {
                            results.push(triple);
                        }
                    }
                } else {
                    break;
                }
            }
            Err(e) => {
                if format!("{e}").contains("429") && retries < 3 {
                    retries += 1;
                    eprintln!("\n  Rate limited, waiting {}s...", retries * 5);
                    std::thread::sleep(std::time::Duration::from_secs(retries * 5));
                    continue;
                }
                eprintln!("\n  Warning: request failed: {}", e);
                break;
            }
        }

        offset += ROWS_PER_PAGE;
        if results.len() >= max_rows {
            results.truncate(max_rows);
        }
    }

    eprintln!("\r  {} {}/{}    ", label, results.len(), max_rows);
    results
}

fn download_dataset() -> CachedData {
    eprintln!("Downloading CodeSearchNet from HuggingFace...");

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(std::time::Duration::from_secs(60)))
            .build(),
    );

    let mut corpus = Vec::new();
    let mut queries = Vec::new();

    let code_languages = ["python", "javascript", "go"];
    let per_lang = TARGET_CODE_SIZE / code_languages.len();

    for lang in &code_languages {
        let pairs = fetch_pages(
            &agent,
            "code-search-net/code_search_net",
            lang,
            "test",
            per_lang,
            lang,
            |row| {
                let doc = row["func_documentation_string"]
                    .as_str()?
                    .trim()
                    .to_string();
                let code = row["func_code_string"].as_str()?.trim().to_string();
                let lang = row["language"].as_str().unwrap_or("unknown").to_string();
                if doc.len() >= 20 && code.len() >= 50 {
                    Some((code, doc, lang))
                } else {
                    None
                }
            },
        );

        for (code, doc, lang) in pairs {
            let idx = corpus.len();
            corpus.push(CorpusDoc {
                text: code,
                kind: lang,
            });
            queries.push(QueryDoc {
                text: doc,
                relevant_idx: idx,
            });
        }
    }

    eprintln!(
        "Downloaded {} corpus docs, {} queries.",
        corpus.len(),
        queries.len()
    );

    CachedData { corpus, queries }
}

fn load_or_download() -> CachedData {
    let path = cache_path();
    if path.exists() {
        eprintln!("Loading cached dataset from {}...", path.display());
        let data = std::fs::read_to_string(&path).unwrap();
        return serde_json::from_str(&data).unwrap();
    }

    let data = download_dataset();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string(&data).unwrap();
    std::fs::write(&path, json).ok();
    eprintln!("Cached dataset to {}", path.display());

    data
}

fn make_embedder() -> Embedder {
    if let (Ok(url), Ok(model)) = (
        std::env::var("VECGREP_EMBEDDER_URL"),
        std::env::var("VECGREP_EMBEDDER_MODEL"),
    ) {
        eprintln!("Using remote embedder: {} ({})", url, model);
        Embedder::new_remote(&url, &model)
    } else {
        eprintln!("Using built-in model");
        Embedder::new_local().unwrap()
    }
}

fn truncate(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        text
    } else {
        &text[..text.floor_char_boundary(max_chars)]
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn rank_vector(query_embedding: &[f32], corpus_embeddings: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = corpus_embeddings
        .iter()
        .enumerate()
        .map(|(i, doc_emb)| (i, cosine_sim(query_embedding, doc_emb)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}

fn rank_hybrid(
    query: &str,
    vector_ranked: &[(usize, f32)],
    lexical_ranked: &[(usize, f32)],
) -> Vec<(usize, f32)> {
    let mut merged: std::collections::HashMap<usize, (Option<usize>, Option<usize>)> =
        std::collections::HashMap::new();

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

fn rank_of(ranked: &[(usize, f32)], relevant: usize) -> Option<usize> {
    ranked
        .iter()
        .position(|(idx, _)| *idx == relevant)
        .map(|rank| rank + 1)
}

fn print_summary(mode: RetrievalMode, metrics: &Metrics) {
    println!("  {}:", mode.label());
    println!("    MRR:    {:.4}", metrics.mean_mrr());
    println!("    R@1:    {:.4}", metrics.recall_at(1));
    println!("    R@5:    {:.4}", metrics.recall_at(5));
    println!("    R@10:   {:.4}", metrics.recall_at(10));
    println!("    R@50:   {:.4}", metrics.recall_at(50));
    println!("    R@100:  {:.4}", metrics.recall_at(100));
}

/// Embed a batch, falling back to one-at-a-time on failure.
/// Returns zero vectors for texts that fail even individually.
fn embed_batch_resilient(
    embedder: &mut Embedder,
    texts: &[&str],
    dim: usize,
) -> (Vec<Vec<f32>>, usize) {
    match embedder.embed_batch(texts) {
        Ok(embeddings) => (embeddings, 0),
        Err(_) => {
            let mut results = Vec::new();
            let mut failed = 0;
            for text in texts {
                match embedder.embed(text) {
                    Ok(emb) => results.push(emb),
                    Err(_) => {
                        results.push(vec![0.0; dim]);
                        failed += 1;
                    }
                }
            }
            (results, failed)
        }
    }
}

#[test]
#[ignore] // Run with: cargo test --test benchmark_large -- --ignored --nocapture
fn benchmark_large_scale() {
    let data = load_or_download();
    let mut embedder = make_embedder();

    // Probe to discover embedding dimension
    let probe = embedder.embed("probe").unwrap();
    let dim = probe.len();

    let corpus_size = data.corpus.len();
    let query_count = data.queries.len().min(1000);
    let lexical_corpus: Vec<CorpusDoc> = data
        .corpus
        .iter()
        .map(|doc| CorpusDoc {
            text: truncate(&doc.text, MAX_TEXT_CHARS).to_string(),
            kind: doc.kind.clone(),
        })
        .collect();

    let mut kind_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for doc in &data.corpus {
        *kind_counts.entry(doc.kind.as_str()).or_default() += 1;
    }

    eprintln!(
        "\n=== Large-Scale Benchmark ({} queries, {} corpus docs, dim={}) ===",
        query_count, corpus_size, dim
    );
    eprintln!(
        "  Corpus: {:?}  truncation: {} chars\n",
        kind_counts, MAX_TEXT_CHARS
    );

    eprintln!("Building lexical index...");
    let lexical_index = LexicalIndex::build(&lexical_corpus).unwrap();

    // Embed corpus — all texts truncated to same limit for fair comparison
    eprintln!("Embedding corpus...");
    let corpus_truncated: Vec<&str> = data
        .corpus
        .iter()
        .map(|d| truncate(&d.text, MAX_TEXT_CHARS))
        .collect();

    let mut corpus_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut total_failed = 0;
    let batch_size = if embedder.embedding_dim() != vecgrep::embedder::EMBEDDING_DIM {
        4
    } else {
        64
    };

    for (i, batch) in corpus_truncated.chunks(batch_size).enumerate() {
        let (embeddings, failed) = embed_batch_resilient(&mut embedder, batch, dim);
        corpus_embeddings.extend(embeddings);
        total_failed += failed;
        if (i + 1) % 10 == 0 {
            eprint!("\r  {}/{}...", corpus_embeddings.len(), corpus_size);
        }
    }

    if total_failed > 0 {
        eprintln!(
            "\r  Embedded {} corpus docs ({} failed).    ",
            corpus_size, total_failed
        );
    } else {
        eprintln!("\r  Embedded {} corpus docs.    ", corpus_size);
    }

    // Evaluate queries — same truncation applied
    eprintln!("Evaluating queries...");
    let mut metrics_by_mode: std::collections::HashMap<&'static str, Metrics> =
        std::collections::HashMap::from([
            (RetrievalMode::Vector.label(), Metrics::default()),
            (RetrievalMode::Lexical.label(), Metrics::default()),
            (RetrievalMode::Hybrid.label(), Metrics::default()),
        ]);
    let mut query_failures = 0;

    for (qi, query) in data.queries.iter().take(query_count).enumerate() {
        let query_text = truncate(&query.text, MAX_TEXT_CHARS);
        let query_emb = match embedder.embed(query_text) {
            Ok(emb) => emb,
            Err(_) => {
                query_failures += 1;
                continue;
            }
        };

        let relevant = query.relevant_idx;
        let vector_ranked = rank_vector(&query_emb, &corpus_embeddings);
        let lexical_ranked = lexical_index.rank(query_text).unwrap();
        let hybrid_ranked = rank_hybrid(query_text, &vector_ranked, &lexical_ranked);

        metrics_by_mode
            .get_mut(RetrievalMode::Vector.label())
            .unwrap()
            .record_rank(rank_of(&vector_ranked, relevant));
        metrics_by_mode
            .get_mut(RetrievalMode::Lexical.label())
            .unwrap()
            .record_rank(rank_of(&lexical_ranked, relevant));
        metrics_by_mode
            .get_mut(RetrievalMode::Hybrid.label())
            .unwrap()
            .record_rank(rank_of(&hybrid_ranked, relevant));

        if (qi + 1) % 100 == 0 {
            eprint!("\r  {}/{}...", qi + 1, query_count);
        }
    }

    let evaluated = query_count - query_failures;

    eprintln!("\r                              ");
    println!(
        "\n=== Results ({} queries evaluated, {} corpus, {} chars max, modes: vector / lexical / hybrid) ===",
        evaluated, corpus_size, MAX_TEXT_CHARS
    );
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
    if total_failed > 0 || query_failures > 0 {
        println!(
            "  Failures: {} corpus, {} queries",
            total_failed, query_failures
        );
    }
    println!("\n  Corpus: {:?}", kind_counts);
    println!();
}
