use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;

use vecgrep::cli::Args;
use vecgrep::embedder::Embedder;
use vecgrep::index::Index;
use vecgrep::invocation::{
    capped_chunk_size, CliOutputContext, Invocation, RunMode, StaleRemovalScope,
};
use vecgrep::pipeline::PipelineStatus;
use vecgrep::root::resolve_project_root;
use vecgrep::types::IndexConfig;
use vecgrep::types::SearchScope;
use vecgrep::{invocation, output, paths, pipeline, serve, tui, walker};

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

const BATCH_SIZE: usize = 32;
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_progress(frame_idx: usize, status: &PipelineStatus) {
    let frame = SPINNER_FRAMES[frame_idx % SPINNER_FRAMES.len()];
    match status {
        PipelineStatus::Scanning { indexed, chunks } => {
            eprint!("\r{} {}/?? files | {} chunks", frame, indexed, chunks);
        }
        PipelineStatus::Indexing {
            indexed,
            total,
            chunks,
        } => {
            eprint!(
                "\r{} {}/{} files | {} chunks",
                frame, indexed, total, chunks
            );
        }
        PipelineStatus::Ready { files, chunks } => {
            eprint!("\r{} {}/{} files | {} chunks", frame, files, files, chunks);
        }
    }
    std::io::stderr().flush().ok();
}

fn clear_progress_line() {
    eprint!("\r\x1b[2K");
    std::io::stderr().flush().ok();
}

type WalkerHandle = std::thread::JoinHandle<anyhow::Result<usize>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexDrainOutcome {
    Completed,
    Aborted,
}

struct ExecutionContext {
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    walker_handle: Option<WalkerHandle>,
    root: String,
}

fn initialize_embedder(invocation: &mut Invocation) -> Result<Embedder> {
    let quiet = invocation.args.quiet;
    let embedder = if let (Some(ref url), Some(ref model)) = (
        &invocation.args.embedder_url,
        &invocation.args.embedder_model,
    ) {
        status!(quiet, "Using external embedder: {} ({})", url, model);
        let mut embedder = Embedder::new_remote(url, model);
        embedder
            .embed("probe")
            .context("Failed to connect to external embedder")?;
        status!(quiet, "Embedding dimension: {}", embedder.embedding_dim());
        embedder
    } else {
        if !quiet {
            eprint!("Loading model...");
            std::io::stderr().flush().ok();
        }
        let embedder = Embedder::new_local().context("Failed to initialize embedder")?;
        if !quiet {
            eprintln!(" done.");
        }
        embedder
    };

    let original_chunk_size = invocation.args.chunk_size.unwrap();
    invocation.args.chunk_size = Some(capped_chunk_size(
        original_chunk_size,
        embedder.context_tokens(),
    ));
    if invocation.args.chunk_size.unwrap() < original_chunk_size {
        status!(
            quiet,
            "Reducing chunk_size from {} to {} (model context limit)",
            original_chunk_size,
            invocation.args.chunk_size.unwrap()
        );
    }

    Ok(embedder)
}

fn prepare_index(
    project_root: &Path,
    embedder: &Embedder,
    invocation: &Invocation,
    reindex: bool,
) -> Result<Index> {
    let idx = Index::open(project_root)?;
    let config = IndexConfig {
        model_name: embedder.model_name().to_string(),
        embedding_dim: embedder.embedding_dim(),
        chunk_size: invocation.args.chunk_size.unwrap(),
        chunk_overlap: invocation.args.chunk_overlap.unwrap(),
    };

    let config_valid = idx.check_config(&config)?;
    if !config_valid || reindex {
        if !config_valid {
            status!(
                invocation.args.quiet,
                "Index configuration changed, rebuilding..."
            );
        }
        idx.rebuild_for_config(&config)?;
    } else {
        idx.set_config(&config)?;
    }

    Ok(idx)
}

fn build_walk_options(args: &Args) -> walker::WalkOptions {
    walker::WalkOptions {
        file_types: args.file_type.clone(),
        file_types_not: args.file_type_not.clone(),
        globs: args.glob.clone(),
        ignore_files: args.ignore_file.clone(),
        hidden: args.hidden,
        follow: args.follow,
        no_ignore: args.no_ignore,
        max_depth: args.max_depth,
    }
}

fn prepare_execution(invocation: &mut Invocation) -> Result<ExecutionContext> {
    let embedder = initialize_embedder(invocation)?;
    let idx = prepare_index(
        &invocation.path_plan.project_root,
        &embedder,
        invocation,
        invocation.args.reindex,
    )?;

    // All paths go through the same walker pipeline. The walker marks
    // explicit file paths so the index can clean them up on subsequent walks.
    let walk_paths = invocation.args.paths.clone();

    let walk_opts = build_walk_options(&invocation.args);
    let (tx, rx) = std::sync::mpsc::channel::<walker::WalkedFile>();
    let stream_progress = Arc::new(walker::StreamProgress::new());
    let walker_progress = Arc::clone(&stream_progress);
    let walker_handle = std::thread::spawn(move || {
        walker::walk_paths_streaming_with_progress(&walk_paths, &walk_opts, tx, walker_progress)
    });

    let indexer = pipeline::StreamingIndexer::new(
        rx,
        invocation.args.chunk_size.unwrap(),
        invocation.args.chunk_overlap.unwrap(),
        BATCH_SIZE,
        &invocation.path_plan.cwd_suffix,
        Some(stream_progress),
    );

    Ok(ExecutionContext {
        embedder,
        idx,
        indexer,
        walker_handle: Some(walker_handle),
        root: invocation
            .path_plan
            .project_root
            .to_string_lossy()
            .to_string(),
    })
}

fn prompt_to_continue() -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Ok(true);
    }

    eprint!("Continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input.trim().eq_ignore_ascii_case("y"))
}

fn drain_initial_indexing(
    indexer: &mut pipeline::StreamingIndexer,
    embedder: &mut Embedder,
    idx: &Index,
    quiet: bool,
    threshold: usize,
    show_spinner: bool,
    mut confirm_continue: impl FnMut() -> Result<bool>,
) -> Result<IndexDrainOutcome> {
    let mut threshold_prompted = false;
    let mut aborted = false;
    let mut frame_idx = 0;
    indexer.drain_all(embedder, idx, |status| {
        let indexed = status.indexed();
        if !quiet && !threshold_prompted && threshold > 0 && indexed >= threshold {
            threshold_prompted = true;
            if show_spinner {
                clear_progress_line();
            }
            match status {
                PipelineStatus::Indexing { indexed, total, .. } => {
                    eprintln!(
                        "Warning: {} files need indexing ({} files found).",
                        indexed, total
                    );
                }
                _ => {
                    eprintln!(
                        "Warning: {} files need indexing so far (still scanning).",
                        indexed
                    );
                }
            }
            if !confirm_continue()? {
                eprintln!("Aborted.");
                aborted = true;
                return Ok(false);
            }
        }
        if show_spinner {
            render_progress(frame_idx, &status);
            frame_idx += 1;
        }
        Ok(true)
    })?;

    if show_spinner {
        clear_progress_line();
    }

    if aborted {
        Ok(IndexDrainOutcome::Aborted)
    } else {
        Ok(IndexDrainOutcome::Completed)
    }
}

fn print_index_stats(idx: &Index) -> Result<()> {
    let stats = idx.stats()?;
    output::print_stats(
        stats.file_count,
        stats.chunk_count,
        stats.failed_chunk_count,
        stats.db_size_bytes,
    );
    Ok(())
}

/// Returns `Some(exit_status)` if the invocation should stop early, `None` to continue.
fn handle_pre_execution_actions(
    args: &Args,
    path_plan: &invocation::PathPlan,
    quiet: bool,
) -> Result<Option<bool>> {
    if args.clear_cache {
        let cache_dir = path_plan.project_root.join(".vecgrep");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
            status!(quiet, "Cache cleared.");
        } else {
            status!(quiet, "No cache found.");
        }
        if args.query.is_none() {
            return Ok(Some(true));
        }
    }

    if args.stats && args.query.is_none() && !args.index_only {
        let idx = Index::open(&path_plan.project_root)?;
        print_index_stats(&idx)?;
        return Ok(Some(true));
    }

    Ok(None)
}

/// Returns `Some(exit_status)` if the invocation should stop early, `None` to continue.
fn handle_post_index_actions(args: &Args, idx: &Index) -> Result<Option<bool>> {
    if args.index_only {
        print_index_stats(idx)?;
        return Ok(Some(true));
    }

    if args.reindex && args.query.is_none() {
        return Ok(Some(true));
    }

    if args.stats {
        print_index_stats(idx)?;
        if args.query.is_none() {
            return Ok(Some(true));
        }
    }

    Ok(None)
}

fn join_walker(walker_handle: &mut Option<WalkerHandle>) -> Result<()> {
    if let Some(handle) = walker_handle.take() {
        match handle.join() {
            Ok(result) => {
                let _ = result?;
            }
            Err(_) => anyhow::bail!("Walker thread panicked"),
        }
    }
    Ok(())
}

fn run_serve_mode(
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    invocation: &Invocation,
    output: CliOutputContext<'_>,
    walker_handle: &mut Option<WalkerHandle>,
    scope: SearchScope,
) -> Result<bool> {
    serve::run_streaming(
        embedder,
        idx,
        indexer,
        serve::ServeConfig {
            port: invocation.args.port,
            default_top_k: invocation.args.top_k.unwrap(),
            default_threshold: invocation.args.threshold.unwrap(),
            quiet: output.quiet,
            root: output.root,
            scope,
        },
    )?;
    join_walker(walker_handle)?;
    Ok(true)
}

fn run_interactive_mode(
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    invocation: &Invocation,
    output: CliOutputContext<'_>,
    walker_handle: &mut Option<WalkerHandle>,
    scope: SearchScope,
) -> Result<bool> {
    tui::interactive::run_streaming(
        embedder,
        idx,
        indexer,
        &invocation.query,
        &invocation.args,
        output.cwd_suffix,
        scope,
    )?;
    join_walker(walker_handle)?;
    Ok(true)
}

fn render_cli_results(
    mut results: Vec<vecgrep::types::SearchResult>,
    args: &Args,
    color_choice: termcolor::ColorChoice,
    cwd_suffix: &Path,
    root: &str,
) -> Result<bool> {
    let found = !results.is_empty();
    if !found {
        return Ok(false);
    }

    if !args.json {
        paths::rewrite_results_to_cwd_relative(&mut results, cwd_suffix);
    }

    if args.json {
        output::print_json(&results, root)?;
    } else if args.files_with_matches {
        output::print_files_with_matches(&results, color_choice)?;
    } else if args.count {
        output::print_count(&results, color_choice)?;
    } else {
        output::print_results(&results, color_choice)?;
    }

    Ok(true)
}

fn run_cli_search(
    embedder: &mut Embedder,
    idx: &Index,
    args: &Args,
    query: &str,
    scope: &SearchScope,
    output: CliOutputContext<'_>,
) -> Result<bool> {
    let chunk_count = idx.chunk_count()?;
    status!(output.quiet, "Index has {} chunks.", chunk_count);

    let query_embedding = embedder.embed(query)?;
    let results = idx.search(
        &query_embedding,
        args.top_k.unwrap(),
        args.threshold.unwrap(),
        scope,
    )?;

    let found = render_cli_results(
        results,
        args,
        output.color_choice,
        output.cwd_suffix,
        output.root,
    )?;
    if !found {
        status!(output.quiet, "No results found.");
    }

    Ok(found)
}

/// When `--query` is provided, use it as the search query and treat all
/// positional args as paths. If a positional query was also parsed, move
/// it into paths. This makes `| xargs vecgrep -i --query "search"` work:
/// xargs-appended files become paths, the query is explicit.
fn resolve_query_flag(args: &mut Args) {
    if let Some(q) = args.query_flag.take() {
        use clap::error::ErrorKind;
        if !args.interactive && !args.serve {
            Args::command()
                .error(
                    ErrorKind::MissingRequiredArgument,
                    "--query requires --interactive (-i) or --serve",
                )
                .exit();
        }
        // Move positional "query" (if any) into paths — with --query,
        // all positionals are paths. This handles the xargs case:
        // `| xargs vecgrep -i --query "search" file1.rs file2.rs`
        // where clap assigns file1.rs to the query positional.
        if let Some(positional) = args.query.take() {
            if args.paths == ["."] {
                args.paths = vec![positional];
            } else {
                args.paths.insert(0, positional);
            }
        }
        args.query = Some(q);
    }
}

fn build_search_scope(arg_paths: &[String], cwd_suffix: &Path) -> SearchScope {
    let explicit_paths: Vec<String> = arg_paths
        .iter()
        .filter(|p| Path::new(p).is_file())
        .map(|p| paths::to_project_relative(p, cwd_suffix))
        .collect();
    let path_scopes: Vec<String> = arg_paths
        .iter()
        .map(|p| paths::to_project_relative(p, cwd_suffix))
        .filter(|p| !p.is_empty())
        .collect();
    SearchScope {
        explicit_paths,
        path_scopes,
    }
}

/// Returns Ok(true) if matches were found, Ok(false) if no matches.
fn run() -> Result<bool> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("VECGREP_LOG"))
        .with_writer(std::io::stderr)
        .init();

    let mut args = Args::parse();

    resolve_query_flag(&mut args);

    if args.type_list {
        walker::print_type_list();
        return Ok(true);
    }

    let cwd = std::env::current_dir()?;
    let project_root = resolve_project_root(&cwd, &args.paths);

    if args.show_root {
        println!("{}", project_root.display());
        return Ok(true);
    }

    let mut invocation = invocation::resolve_invocation(args, &cwd, &project_root)?;
    let quiet = invocation.args.quiet;

    if let Some(result) =
        handle_pre_execution_actions(&invocation.args, &invocation.path_plan, quiet)?
    {
        return Ok(result);
    }

    let ExecutionContext {
        mut embedder,
        idx,
        mut indexer,
        mut walker_handle,
        root,
    } = prepare_execution(&mut invocation)?;
    let output = CliOutputContext {
        color_choice: invocation.color_choice,
        cwd_suffix: &invocation.path_plan.cwd_suffix,
        quiet,
        root: &root,
    };
    if invocation.query.is_empty()
        && matches!(invocation.run_mode, RunMode::Cli)
        && !invocation.args.stats
        && !invocation.args.reindex
        && !invocation.args.index_only
    {
        return Ok(true);
    }

    let show_spinner = !quiet && std::io::stderr().is_terminal();

    let must_drain_before_search =
        matches!(invocation.run_mode, RunMode::Cli) || invocation.args.index_only;
    if invocation.args.full_index || must_drain_before_search {
        let drain_outcome = drain_initial_indexing(
            &mut indexer,
            &mut embedder,
            &idx,
            quiet,
            invocation.args.index_warn_threshold.unwrap(),
            show_spinner,
            prompt_to_continue,
        )?;
        if matches!(drain_outcome, IndexDrainOutcome::Aborted) {
            return Ok(false);
        }
        finish_indexing(
            &mut indexer,
            &idx,
            &invocation.path_plan.stale_removal_scope,
            quiet,
            &mut walker_handle,
        )?;
    }

    if let Some(result) = handle_post_index_actions(&invocation.args, &idx)? {
        return Ok(result);
    }

    let search_scope = build_search_scope(&invocation.args.paths, &invocation.path_plan.cwd_suffix);

    match invocation.run_mode {
        RunMode::Serve => {
            return run_serve_mode(
                embedder,
                idx,
                indexer,
                &invocation,
                output,
                &mut walker_handle,
                search_scope,
            );
        }
        RunMode::Interactive => {
            return run_interactive_mode(
                embedder,
                idx,
                indexer,
                &invocation,
                output,
                &mut walker_handle,
                search_scope,
            );
        }
        RunMode::Cli => {}
    }

    let found = run_cli_search(
        &mut embedder,
        &idx,
        &invocation.args,
        &invocation.query,
        &search_scope,
        output,
    )?;

    Ok(found)
}

fn finish_indexing(
    indexer: &mut pipeline::StreamingIndexer,
    idx: &Index,
    stale_removal_scope: &StaleRemovalScope,
    quiet: bool,
    walker_handle: &mut Option<WalkerHandle>,
) -> Result<()> {
    join_walker(walker_handle)?;

    let removed = match stale_removal_scope {
        StaleRemovalScope::Prefix(walk_prefix) => {
            if walk_prefix.as_os_str().is_empty() {
                idx.remove_stale_files(&indexer.all_paths, None)?
            } else {
                idx.remove_stale_files(&indexer.all_paths, Some(walk_prefix))?
            }
        }
        StaleRemovalScope::None => 0,
    };
    if removed > 0 {
        status!(quiet, "Removed {} stale files from index.", removed);
    }

    if indexer.indexed_count > 0 {
        status!(
            quiet,
            "Indexed {}/{} files, {} chunks.",
            indexer.indexed_count,
            indexer.all_paths.len(),
            indexer.indexed_chunks
        );
    } else {
        status!(
            quiet,
            "Index is up to date. {} files.",
            indexer.all_paths.len()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;
    use vecgrep::embedder::EMBEDDING_DIM;
    use vecgrep::types::{Chunk, IndexConfig};

    fn make_chunk(file_path: &str, text: &str) -> Chunk {
        Chunk {
            file_path: file_path.to_string(),
            text: text.to_string(),
            start_line: 1,
            end_line: 1,
        }
    }

    fn make_embedding() -> Vec<f32> {
        let value = 1.0 / (EMBEDDING_DIM as f32).sqrt();
        vec![value; EMBEDDING_DIM]
    }

    #[test]
    fn test_drain_initial_indexing_aborts_cleanly() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        tx.send(walker::WalkedFile {
            rel_path: "main.rs".to_string(),
            content: "fn search_target() {}\n".to_string(),
            explicit: false,
        })
        .unwrap();
        drop(tx);

        let mut indexer = pipeline::StreamingIndexer::new(rx, 500, 100, 1, Path::new(""), None);

        let outcome =
            drain_initial_indexing(&mut indexer, &mut embedder, &idx, false, 1, false, || {
                Ok(false)
            })
            .unwrap();

        assert_eq!(outcome, IndexDrainOutcome::Aborted);
        assert_eq!(idx.chunk_count().unwrap(), 1);
    }

    #[test]
    fn test_handle_pre_execution_actions_returns_after_stats_without_query() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let path_plan = invocation::PathPlan {
            project_root: dir.path().canonicalize().unwrap(),
            cwd_suffix: PathBuf::new(),
            stale_removal_scope: StaleRemovalScope::Prefix(PathBuf::new()),
        };
        let args = Args::parse_from(["vecgrep", "--stats"]);

        let outcome = handle_pre_execution_actions(&args, &path_plan, true).unwrap();

        assert_eq!(outcome, Some(true));
    }

    #[test]
    fn test_handle_pre_execution_actions_clear_cache_with_query_continues() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir(dir.path().join(".vecgrep")).unwrap();
        let path_plan = invocation::PathPlan {
            project_root: dir.path().canonicalize().unwrap(),
            cwd_suffix: PathBuf::new(),
            stale_removal_scope: StaleRemovalScope::Prefix(PathBuf::new()),
        };
        let args = Args::parse_from(["vecgrep", "--clear-cache", "needle"]);

        let outcome = handle_pre_execution_actions(&args, &path_plan, true).unwrap();

        assert_eq!(
            outcome, None,
            "should continue to search when query is present"
        );
        assert!(
            !dir.path().join(".vecgrep").exists(),
            "cache directory should be deleted"
        );
    }

    #[test]
    fn test_handle_post_index_actions_returns_after_index_only() {
        let index = Index::open_in_memory().unwrap();
        let args = Args::parse_from(["vecgrep", "--index-only"]);

        let outcome = handle_post_index_actions(&args, &index).unwrap();

        assert_eq!(outcome, Some(true));
    }

    #[test]
    fn test_finish_indexing_removes_stale_files_for_prefix_scope() {
        let index = Index::open_in_memory().unwrap();
        let config = IndexConfig {
            model_name: "test-model".to_string(),
            embedding_dim: EMBEDDING_DIM,
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config).unwrap();

        let embedding = make_embedding();
        index
            .upsert_file(
                "live.rs",
                "hash-live",
                &[make_chunk("live.rs", "fn live() {}")],
                std::slice::from_ref(&embedding),
                &[false],
            )
            .unwrap();
        index
            .upsert_file(
                "stale.rs",
                "hash-stale",
                &[make_chunk("stale.rs", "fn stale() {}")],
                std::slice::from_ref(&embedding),
                &[false],
            )
            .unwrap();

        let (_tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut indexer = pipeline::StreamingIndexer::new(rx, 500, 100, 1, Path::new(""), None);
        indexer.all_paths = vec!["live.rs".to_string()];
        indexer.indexed_count = 1;
        indexer.indexed_chunks = 1;

        finish_indexing(
            &mut indexer,
            &index,
            &StaleRemovalScope::Prefix(PathBuf::new()),
            true,
            &mut None,
        )
        .unwrap();

        assert_eq!(
            index.get_file_hash("live.rs").unwrap(),
            Some("hash-live".to_string())
        );
        assert_eq!(index.get_file_hash("stale.rs").unwrap(), None);
    }

    #[test]
    fn test_finish_indexing_prefix_scope_preserves_other_dirs() {
        let index = Index::open_in_memory().unwrap();
        let config = IndexConfig {
            model_name: "test-model".to_string(),
            embedding_dim: EMBEDDING_DIM,
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config).unwrap();

        let embedding = make_embedding();
        index
            .upsert_file(
                "src/live.rs",
                "hash-live",
                &[make_chunk("src/live.rs", "fn live() {}")],
                std::slice::from_ref(&embedding),
                &[false],
            )
            .unwrap();
        index
            .upsert_file(
                "src/stale.rs",
                "hash-stale",
                &[make_chunk("src/stale.rs", "fn stale() {}")],
                std::slice::from_ref(&embedding),
                &[false],
            )
            .unwrap();
        index
            .upsert_file(
                "tests/keep.rs",
                "hash-keep",
                &[make_chunk("tests/keep.rs", "fn keep() {}")],
                std::slice::from_ref(&embedding),
                &[false],
            )
            .unwrap();

        let (_tx, rx) = std::sync::mpsc::sync_channel(1);
        let mut indexer = pipeline::StreamingIndexer::new(rx, 500, 100, 1, Path::new(""), None);
        indexer.all_paths = vec!["src/live.rs".to_string()];

        finish_indexing(
            &mut indexer,
            &index,
            &StaleRemovalScope::Prefix(PathBuf::from("src")),
            true,
            &mut None,
        )
        .unwrap();

        assert_eq!(
            index.get_file_hash("src/live.rs").unwrap(),
            Some("hash-live".to_string())
        );
        assert_eq!(index.get_file_hash("src/stale.rs").unwrap(), None);
        assert_eq!(
            index.get_file_hash("tests/keep.rs").unwrap(),
            Some("hash-keep".to_string())
        );
    }

    #[test]
    fn test_query_flag_moves_positional_to_paths() {
        // Simulates: | xargs vecgrep -i --query "search" file1.rs file2.rs
        // clap assigns file1.rs to query positional, file2.rs to paths
        let mut args =
            Args::parse_from(["vecgrep", "--query", "search", "-i", "file1.rs", "file2.rs"]);
        assert_eq!(args.query.as_deref(), Some("file1.rs"));
        assert_eq!(args.query_flag.as_deref(), Some("search"));
        assert_eq!(args.paths, ["file2.rs"]);

        resolve_query_flag(&mut args);

        assert_eq!(args.query.as_deref(), Some("search"));
        assert_eq!(args.paths, ["file1.rs", "file2.rs"]);
    }

    #[test]
    fn test_query_flag_without_positional() {
        // vecgrep --query "search" -i
        let mut args = Args::parse_from(["vecgrep", "--query", "search", "-i"]);
        assert!(args.query.is_none());
        assert_eq!(args.query_flag.as_deref(), Some("search"));

        resolve_query_flag(&mut args);

        assert_eq!(args.query.as_deref(), Some("search"));
        assert_eq!(args.paths, ["."]);
    }

    #[test]
    fn test_no_query_flag_preserves_positional() {
        // vecgrep "search term" -i
        let mut args = Args::parse_from(["vecgrep", "search term", "-i"]);
        assert_eq!(args.query.as_deref(), Some("search term"));
        assert!(args.query_flag.is_none());

        resolve_query_flag(&mut args);

        assert_eq!(args.query.as_deref(), Some("search term"));
    }
}
