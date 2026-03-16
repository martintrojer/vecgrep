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

use std::path::PathBuf;
use vecgrep::embedder::Embedder;

const DATASET_API: &str = "https://datasets-server.huggingface.co/rows";
const ROWS_PER_PAGE: usize = 100;
const TARGET_CODE_SIZE: usize = 9_000;
const CACHE_FILE: &str = "target/benchmark_cache/codesearchnet_v3.json";

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
    let mut total_mrr = 0.0;
    let mut recall_at_1 = 0;
    let mut recall_at_5 = 0;
    let mut recall_at_10 = 0;
    let mut recall_at_50 = 0;
    let mut recall_at_100 = 0;
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

        let mut scores: Vec<(usize, f32)> = corpus_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| (i, cosine_sim(&query_emb, doc_emb)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let relevant = query.relevant_idx;
        let rank = scores
            .iter()
            .position(|(idx, _)| *idx == relevant)
            .map(|r| r + 1);

        if let Some(r) = rank {
            total_mrr += 1.0 / r as f32;
            if r <= 1 {
                recall_at_1 += 1;
            }
            if r <= 5 {
                recall_at_5 += 1;
            }
            if r <= 10 {
                recall_at_10 += 1;
            }
            if r <= 50 {
                recall_at_50 += 1;
            }
            if r <= 100 {
                recall_at_100 += 1;
            }
        }

        if (qi + 1) % 100 == 0 {
            eprint!("\r  {}/{}...", qi + 1, query_count);
        }
    }

    let evaluated = query_count - query_failures;
    let n = evaluated as f32;
    let mrr = total_mrr / n;
    let r1 = recall_at_1 as f32 / n;
    let r5 = recall_at_5 as f32 / n;
    let r10 = recall_at_10 as f32 / n;
    let r50 = recall_at_50 as f32 / n;
    let r100 = recall_at_100 as f32 / n;

    eprintln!("\r                              ");
    println!(
        "\n=== Results ({} queries evaluated, {} corpus, {} chars max) ===",
        evaluated, corpus_size, MAX_TEXT_CHARS
    );
    println!("  MRR:    {mrr:.4}");
    println!("  R@1:    {r1:.4}");
    println!("  R@5:    {r5:.4}");
    println!("  R@10:   {r10:.4}");
    println!("  R@50:   {r50:.4}");
    println!("  R@100:  {r100:.4}");
    if total_failed > 0 || query_failures > 0 {
        println!(
            "  Failures: {} corpus, {} queries",
            total_failed, query_failures
        );
    }
    println!("\n  Corpus: {:?}", kind_counts);
    println!();
}
