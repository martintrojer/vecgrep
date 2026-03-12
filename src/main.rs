use anyhow::{Context, Result};
use clap::Parser;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use vecgrep::cli::Args;
use vecgrep::embedder::Embedder;
use vecgrep::index::Index;
use vecgrep::types::IndexConfig;
use vecgrep::{chunker, output, search, serve, tui, walker};

/// Print a status message to stderr unless --quiet is set.
macro_rules! status {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            eprintln!($($arg)*);
        }
    };
}

/// Walk up from `start` to find the project root.
/// Stops at: .git/, .hg/, .jj/, or existing .vecgrep/.
/// Never walks above $HOME. Falls back to `start` if nothing found.
fn find_project_root(start: &Path) -> PathBuf {
    let start_canon = match start.canonicalize() {
        Ok(p) => p,
        Err(_) => return start.to_path_buf(),
    };

    let home = dirs::home_dir();
    let mut current = start_canon.as_path();
    loop {
        for marker in &[".git", ".hg", ".jj", ".vecgrep"] {
            if current.join(marker).exists() {
                return current.to_path_buf();
            }
        }

        // Stop at $HOME (never go above it)
        if let Some(ref h) = home {
            if current == h.as_path() {
                return start_canon;
            }
        }

        match current.parent() {
            Some(parent) if parent != current => current = parent,
            _ => return start_canon,
        }
    }
}

/// Convert a walker-relative path to a project-root-relative path.
fn to_project_relative(walker_path: &str, cwd_suffix: &Path) -> String {
    let stripped = walker_path.strip_prefix("./").unwrap_or(walker_path);
    if cwd_suffix.as_os_str().is_empty() {
        stripped.to_string()
    } else {
        format!("{}/{}", cwd_suffix.display(), stripped)
    }
}

/// Convert a project-root-relative path to a cwd-relative path for display.
fn to_display_path(project_path: &str, cwd_suffix: &Path) -> String {
    if cwd_suffix.as_os_str().is_empty() {
        return project_path.to_string();
    }
    let prefix = format!("{}/", cwd_suffix.display());
    if let Some(rest) = project_path.strip_prefix(&prefix) {
        rest.to_string()
    } else {
        // Path is outside cwd subtree, compute relative path
        make_relative(cwd_suffix, Path::new(project_path))
    }
}

/// Compute a relative path from `from` to `to`, where both are relative to the same root.
fn make_relative(from: &Path, to: &Path) -> String {
    let from_comps: Vec<_> = from.components().collect();
    let to_comps: Vec<_> = to.components().collect();

    let common = from_comps
        .iter()
        .zip(to_comps.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let ups = from_comps.len() - common;
    let mut result = PathBuf::new();
    for _ in 0..ups {
        result.push("..");
    }
    for comp in &to_comps[common..] {
        result.push(comp);
    }
    result.to_string_lossy().to_string()
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
    let cwd = std::env::current_dir()?;
    let index_root = if args.paths.len() == 1 && Path::new(&args.paths[0]).is_dir() {
        Path::new(&args.paths[0]).to_path_buf()
    } else {
        cwd.clone()
    };

    // Find project root and compute path relationships
    let project_root = find_project_root(&index_root);
    let project_root_canon = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.clone());
    let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
    let cwd_suffix = cwd_canon
        .strip_prefix(&project_root_canon)
        .unwrap_or(Path::new(""))
        .to_path_buf();
    let walk_prefix = {
        let walk_root_abs = if args.paths.len() == 1 && Path::new(&args.paths[0]).is_dir() {
            Path::new(&args.paths[0])
                .canonicalize()
                .unwrap_or_else(|_| cwd_canon.join(&args.paths[0]))
        } else {
            cwd_canon.clone()
        };
        walk_root_abs
            .strip_prefix(&project_root_canon)
            .map(|p| p.to_path_buf())
            .unwrap_or_default()
    };

    let quiet = args.quiet;

    // Handle --clear-cache
    if args.clear_cache {
        let cache_dir = project_root.join(".vecgrep");
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
        let idx = Index::open(&project_root)?;
        let stats = idx.stats()?;
        output::print_stats(stats.file_count, stats.chunk_count, stats.db_size_bytes);
        return Ok(true);
    }

    // Initialize embedder
    status!(quiet, "Loading model...");
    let mut embedder = Embedder::new().context("Failed to initialize embedder")?;
    status!(quiet, "Model loaded.");

    // Open or create index
    let idx = Index::open(&project_root)?;

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

    // Convert walker paths to project-root-relative
    let files: Vec<walker::WalkedFile> = files
        .into_iter()
        .map(|mut f| {
            f.rel_path = to_project_relative(&f.rel_path, &cwd_suffix);
            f
        })
        .collect();
    status!(quiet, "Found {} files.", files.len());

    // Remove stale files from index (scoped to walk prefix when in a subdirectory)
    let current_paths: Vec<String> = files.iter().map(|f| f.rel_path.clone()).collect();
    let removed = if walk_prefix.as_os_str().is_empty() {
        idx.remove_stale_files(&current_paths)?
    } else {
        let prefix = format!("{}/", walk_prefix.display());
        idx.remove_stale_files_under(&current_paths, &prefix)?
    };
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

    if args.serve {
        serve::run(
            &mut embedder,
            &chunks,
            &embedding_matrix,
            args.port,
            args.top_k,
            args.threshold,
            quiet,
        )?;
        return Ok(true);
    }

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

    // For non-JSON CLI output, convert paths from project-root-relative to cwd-relative
    let mut results = results;
    if !args.json && !cwd_suffix.as_os_str().is_empty() {
        for r in &mut results {
            r.chunk.file_path = to_display_path(&r.chunk.file_path, &cwd_suffix);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;

    // --- find_project_root tests ---

    #[test]
    fn test_find_root_with_git() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/deep")).unwrap();

        let root = find_project_root(&dir.path().join("src/deep"));
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_with_hg() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".hg")).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let root = find_project_root(&dir.path().join("sub"));
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_with_jj() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".jj")).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let root = find_project_root(&dir.path().join("sub"));
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_with_vecgrep() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".vecgrep")).unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let root = find_project_root(&dir.path().join("sub"));
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_at_project_root() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();

        let root = find_project_root(dir.path());
        assert_eq!(root, dir.path().canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_no_markers_falls_back() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();

        let root = find_project_root(&dir.path().join("sub"));
        // Falls back to canonicalized start
        assert_eq!(root, dir.path().join("sub").canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_vecgrep_at_lower_level_wins() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("sub/.vecgrep")).unwrap();

        // Starting from sub/, .vecgrep/ is found at sub/ before .git/ at parent
        let root = find_project_root(&dir.path().join("sub"));
        assert_eq!(root, dir.path().join("sub").canonicalize().unwrap());
    }

    #[test]
    fn test_find_root_nonexistent_path_falls_back() {
        let result = find_project_root(Path::new("/nonexistent/path/that/doesnt/exist"));
        // canonicalize fails, falls back to the input path
        assert_eq!(result, PathBuf::from("/nonexistent/path/that/doesnt/exist"));
    }

    // --- to_project_relative tests ---

    #[test]
    fn test_project_relative_empty_suffix_strips_dot_slash() {
        assert_eq!(to_project_relative("./main.rs", Path::new("")), "main.rs");
    }

    #[test]
    fn test_project_relative_empty_suffix_no_dot_slash() {
        assert_eq!(
            to_project_relative("src/main.rs", Path::new("")),
            "src/main.rs"
        );
    }

    #[test]
    fn test_project_relative_with_suffix_strips_dot_slash() {
        assert_eq!(
            to_project_relative("./main.rs", Path::new("src")),
            "src/main.rs"
        );
    }

    #[test]
    fn test_project_relative_with_suffix_no_dot_slash() {
        assert_eq!(
            to_project_relative("lib/foo.rs", Path::new("src")),
            "src/lib/foo.rs"
        );
    }

    #[test]
    fn test_project_relative_nested_suffix() {
        assert_eq!(
            to_project_relative("./mod.rs", Path::new("src/deep")),
            "src/deep/mod.rs"
        );
    }

    // --- to_display_path tests ---

    #[test]
    fn test_display_path_empty_suffix() {
        assert_eq!(to_display_path("src/main.rs", Path::new("")), "src/main.rs");
    }

    #[test]
    fn test_display_path_strips_prefix() {
        assert_eq!(to_display_path("src/main.rs", Path::new("src")), "main.rs");
    }

    #[test]
    fn test_display_path_strips_nested_prefix() {
        assert_eq!(
            to_display_path("src/deep/mod.rs", Path::new("src/deep")),
            "mod.rs"
        );
    }

    #[test]
    fn test_display_path_outside_subtree() {
        // cwd is src/, path is lib/foo.rs → ../lib/foo.rs
        assert_eq!(
            to_display_path("lib/foo.rs", Path::new("src")),
            "../lib/foo.rs"
        );
    }

    #[test]
    fn test_display_path_sibling_deep() {
        // cwd is src/a/, path is src/b/foo.rs → ../b/foo.rs
        assert_eq!(
            to_display_path("src/b/foo.rs", Path::new("src/a")),
            "../b/foo.rs"
        );
    }

    #[test]
    fn test_display_path_root_file_from_subdir() {
        // cwd is src/, path is README.md → ../README.md
        assert_eq!(
            to_display_path("README.md", Path::new("src")),
            "../README.md"
        );
    }

    // --- make_relative tests ---

    #[test]
    fn test_make_relative_same_dir() {
        assert_eq!(
            make_relative(Path::new("src"), Path::new("src/main.rs")),
            "main.rs"
        );
    }

    #[test]
    fn test_make_relative_sibling() {
        assert_eq!(
            make_relative(Path::new("src"), Path::new("lib/foo.rs")),
            "../lib/foo.rs"
        );
    }

    #[test]
    fn test_make_relative_deep_to_shallow() {
        assert_eq!(
            make_relative(Path::new("src/a/b"), Path::new("README.md")),
            "../../../README.md"
        );
    }

    #[test]
    fn test_make_relative_no_common_prefix() {
        assert_eq!(
            make_relative(Path::new("aaa"), Path::new("bbb/file.rs")),
            "../bbb/file.rs"
        );
    }

    #[test]
    fn test_make_relative_shared_prefix() {
        assert_eq!(
            make_relative(Path::new("src/a"), Path::new("src/b/c.rs")),
            "../b/c.rs"
        );
    }
}
