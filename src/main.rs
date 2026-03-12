use anyhow::{Context, Result};
use clap::Parser;
use std::path::Path;

use vecgrep::cli::Args;
use vecgrep::embedder::Embedder;
use vecgrep::index::Index;
use vecgrep::types::IndexConfig;
use vecgrep::{chunker, output, search, tui, walker};

fn main() -> Result<()> {
    // Initialize tracing from VECGREP_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("VECGREP_LOG"))
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Handle --type-list (no model or index needed)
    if args.type_list {
        walker::print_type_list();
        return Ok(());
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

    // Handle --clear-cache
    if args.clear_cache {
        let cache_dir = index_root.join(".vecgrep");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
            eprintln!("Cache cleared.");
        } else {
            eprintln!("No cache found.");
        }
        if args.query.is_none() {
            return Ok(());
        }
    }

    // Handle --stats (without loading model)
    if args.stats && args.query.is_none() && !args.index_only {
        let idx = Index::open(&index_root)?;
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        return Ok(());
    }

    // Initialize embedder
    eprintln!("Loading model...");
    let mut embedder = Embedder::new().context("Failed to initialize embedder")?;
    eprintln!("Model loaded.");

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
            eprintln!("Index configuration changed, rebuilding...");
        }
        idx.clear()?;
    }
    idx.set_config(&config)?;

    // Walk files
    eprintln!("Scanning files...");
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
    eprintln!("Found {} files.", files.len());

    // Remove stale files from index
    let current_paths: Vec<String> = files.iter().map(|f| f.rel_path.clone()).collect();
    let removed = idx.remove_stale_files(&current_paths)?;
    if removed > 0 {
        eprintln!("Removed {} stale files from index.", removed);
    }

    // Find files that need (re-)indexing
    let files_to_index: Vec<&walker::WalkedFile> = files
        .iter()
        .filter(|f| {
            let hash = blake3::hash(f.content.as_bytes()).to_hex().to_string();
            match idx.get_file_hash(&f.rel_path) {
                Ok(Some(stored_hash)) => stored_hash != hash,
                _ => true,
            }
        })
        .collect();

    if !files_to_index.is_empty() {
        eprintln!("Indexing {} files...", files_to_index.len());

        let tokenizer_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/models/tokenizer.json"));
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let batch_size = 32;
        let file_batches: Vec<&[&walker::WalkedFile]> = files_to_index.chunks(batch_size).collect();

        for (batch_idx, file_batch) in file_batches.iter().enumerate() {
            let mut all_chunks = Vec::new();
            let mut chunk_file_info: Vec<(String, String)> = Vec::new();

            for file in *file_batch {
                let content_hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
                let file_chunks = chunker::chunk_file(
                    &file.rel_path,
                    &file.content,
                    args.chunk_size,
                    args.chunk_overlap,
                    &tokenizer,
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

            eprint!(
                "\rIndexed {}/{} files...",
                ((batch_idx + 1) * batch_size).min(files_to_index.len()),
                files_to_index.len()
            );
        }

        eprintln!("\nIndexing complete.");
    } else {
        eprintln!("Index is up to date.");
    }

    // Handle --index-only
    if args.index_only {
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        return Ok(());
    }

    // Handle --stats after indexing
    if args.stats {
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        if args.query.is_none() {
            return Ok(());
        }
    }

    // Need a query from here on
    let query = match &args.query {
        Some(q) => q.clone(),
        None => return Ok(()),
    };

    // Load all embeddings for search
    let (chunks, embedding_matrix) = idx.load_all()?;
    eprintln!("Loaded {} chunks for search.", chunks.len());

    // Embed query
    let query_embedding = embedder.embed(&query)?;

    if args.interactive {
        match tui::interactive::run(
            &mut embedder,
            &chunks,
            &embedding_matrix,
            &query,
            args.top_k,
            args.threshold,
        )? {
            Some(result) => {
                if args.json {
                    output::print_json(&[result])?;
                } else {
                    output::print_results(&[result], args.context, color_choice)?;
                }
            }
            None => {
                eprintln!("No selection.");
            }
        }
        return Ok(());
    }

    // Batch search
    let results = search::search(
        &query_embedding,
        &embedding_matrix,
        args.top_k,
        args.threshold,
        &chunks,
    );

    if results.is_empty() {
        eprintln!("No results found.");
        return Ok(());
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

    Ok(())
}
