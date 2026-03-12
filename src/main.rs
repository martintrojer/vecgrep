use anyhow::{Context, Result};
use clap::Parser;
use std::io::IsTerminal;
use std::path::Path;

use vecgrep::cli::Args;
use vecgrep::embedder::Embedder;
use vecgrep::index::Index;
use vecgrep::types::IndexConfig;
use vecgrep::{chunker, output, search, tui, walker};

/// Print a status message to stderr unless --quiet is set.
macro_rules! status {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            eprintln!($($arg)*);
        }
    };
}

fn main() {
    std::process::exit(match run() {
        Ok(matched) => {
            if matched {
                0
            } else {
                1
            }
        }
        Err(e) => {
            eprintln!("{:#}", e);
            2
        }
    });
}

/// Returns Ok(true) if matches were found, Ok(false) if no matches.
fn run() -> Result<bool> {
    // Initialize tracing from VECGREP_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("VECGREP_LOG"))
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Handle --type-list (no model or index needed)
    if args.type_list {
        walker::print_type_list();
        return Ok(true);
    }

    // Configure rayon thread pool
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let color_choice = output::resolve_color_choice(&args.color);

    // Determine the index root (first path, or cwd)
    let index_root = if args.paths.len() == 1 && Path::new(&args.paths[0]).is_dir() {
        Path::new(&args.paths[0]).to_path_buf()
    } else {
        std::env::current_dir()?
    };

    let quiet = args.quiet;

    // Handle --clear-cache
    if args.clear_cache {
        let cache_dir = index_root.join(".vecgrep");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
            status!(quiet, "Cache cleared.");
        } else {
            status!(quiet, "No cache found.");
        }
        if args.query.is_none() {
            return Ok(true);
        }
    }

    // Handle --stats (without loading model)
    if args.stats && args.query.is_none() && !args.index_only {
        let idx = Index::open(&index_root)?;
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        return Ok(true);
    }

    // Initialize embedder
    status!(quiet, "Loading model...");
    let mut embedder = Embedder::new().context("Failed to initialize embedder")?;
    status!(quiet, "Model loaded.");

    // Open or create index
    let idx = Index::open(&index_root)?;

    let config = IndexConfig {
        model_name: "all-MiniLM-L6-v2".to_string(),
        chunk_size: args.chunk_size,
        chunk_overlap: args.chunk_overlap,
    };

    // Check if config changed
    let config_valid = idx.check_config(&config)?;
    if !config_valid || args.reindex {
        if !config_valid {
            status!(quiet, "Index configuration changed, rebuilding...");
        }
        idx.clear()?;
    }
    idx.set_config(&config)?;

    // Walk files
    status!(quiet, "Scanning files...");
    let walk_opts = walker::WalkOptions {
        file_types: &args.file_type,
        file_types_not: &args.file_type_not,
        globs: &args.glob,
        hidden: args.hidden,
        follow: args.follow,
        no_ignore: args.no_ignore,
        max_depth: args.max_depth,
    };
    let files = walker::walk_paths(&args.paths, &walk_opts)?;
    status!(quiet, "Found {} files.", files.len());

    // Remove stale files from index
    let current_paths: Vec<String> = files.iter().map(|f| f.rel_path.clone()).collect();
    let removed = idx.remove_stale_files(&current_paths)?;
    if removed > 0 {
        status!(quiet, "Removed {} stale files from index.", removed);
    }

    // Find files that need (re-)indexing (with pre-computed hashes)
    let files_to_index: Vec<(&walker::WalkedFile, String)> = files
        .iter()
        .filter_map(|f| {
            let hash = blake3::hash(f.content.as_bytes()).to_hex().to_string();
            let needs_index = match idx.get_file_hash(&f.rel_path) {
                Ok(Some(stored_hash)) => stored_hash != hash,
                _ => true,
            };
            if needs_index {
                Some((f, hash))
            } else {
                None
            }
        })
        .collect();

    if !files_to_index.is_empty() {
        status!(quiet, "Indexing {} files...", files_to_index.len());

        let batch_size = 32;
        let file_batches: Vec<&[(&walker::WalkedFile, String)]> =
            files_to_index.chunks(batch_size).collect();

        for (batch_idx, file_batch) in file_batches.iter().enumerate() {
            let mut all_chunks = Vec::new();
            let mut chunk_file_info: Vec<(String, String)> = Vec::new();

            for (file, content_hash) in *file_batch {
                let file_chunks = chunker::chunk_file(
                    &file.rel_path,
                    &file.content,
                    args.chunk_size,
                    args.chunk_overlap,
                    embedder.tokenizer(),
                );

                for _ in &file_chunks {
                    chunk_file_info.push((file.rel_path.clone(), content_hash.clone()));
                }
                all_chunks.extend(file_chunks);
            }

            if all_chunks.is_empty() {
                continue;
            }

            // Embed all chunks in sub-batches
            let texts: Vec<&str> = all_chunks.iter().map(|c| c.text.as_str()).collect();
            let embed_batch_size = 64;
            let mut all_embeddings = Vec::new();
            for text_batch in texts.chunks(embed_batch_size) {
                let embeddings = embedder.embed_batch(text_batch)?;
                all_embeddings.extend(embeddings);
            }

            // Group chunks by file and insert into index
            let mut current_file: Option<String> = None;
            let mut file_chunks = Vec::new();
            let mut file_embeddings = Vec::new();
            let mut file_hash = String::new();

            for (i, chunk) in all_chunks.iter().enumerate() {
                let (ref path, ref hash) = chunk_file_info[i];

                if current_file.as_ref() != Some(path) {
                    if let Some(ref prev_path) = current_file {
                        idx.upsert_file(prev_path, &file_hash, &file_chunks, &file_embeddings)?;
                    }
                    current_file = Some(path.clone());
                    file_hash = hash.clone();
                    file_chunks = Vec::new();
                    file_embeddings = Vec::new();
                }

                file_chunks.push(chunk.clone());
                file_embeddings.push(all_embeddings[i].clone());
            }

            if let Some(ref prev_path) = current_file {
                idx.upsert_file(prev_path, &file_hash, &file_chunks, &file_embeddings)?;
            }

            if !quiet && std::io::stderr().is_terminal() {
                eprint!(
                    "\rIndexed {}/{} files...",
                    ((batch_idx + 1) * batch_size).min(files_to_index.len()),
                    files_to_index.len()
                );
            }
        }

        status!(quiet, "\nIndexing complete.");
    } else {
        status!(quiet, "Index is up to date.");
    }

    // Handle --index-only
    if args.index_only {
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        return Ok(true);
    }

    // Handle --stats after indexing
    if args.stats {
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        if args.query.is_none() {
            return Ok(true);
        }
    }

    // Need a query from here on (unless interactive mode)
    let query = match &args.query {
        Some(q) => q.clone(),
        None if args.interactive => String::new(),
        None => return Ok(true),
    };

    // Load all embeddings for search
    let (chunks, embedding_matrix) = idx.load_all()?;
    status!(quiet, "Loaded {} chunks for search.", chunks.len());

    if args.interactive {
        tui::interactive::run(
            &mut embedder,
            &chunks,
            &embedding_matrix,
            &query,
            args.top_k,
            args.threshold,
        )?;
        return Ok(true);
    }

    // Embed query
    let query_embedding = embedder.embed(&query)?;

    // Batch search
    let results = search::search(
        &query_embedding,
        &embedding_matrix,
        args.top_k,
        args.threshold,
        &chunks,
    );

    if results.is_empty() {
        status!(quiet, "No results found.");
        return Ok(false);
    }

    if args.json {
        output::print_json(&results)?;
    } else if args.files_with_matches {
        output::print_files_with_matches(&results, color_choice)?;
    } else if args.count {
        output::print_count(&results, color_choice)?;
    } else {
        output::print_results(&results, args.context, color_choice)?;
    }

    Ok(true)
}
