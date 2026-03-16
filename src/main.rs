use anyhow::{Context, Result};
use clap::Parser;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vecgrep::cli::Args;
use vecgrep::embedder::Embedder;
use vecgrep::index::Index;
use vecgrep::pipeline::CliIndexingProgress;
use vecgrep::types::IndexConfig;
use vecgrep::{output, paths, pipeline, serve, tui, walker};

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

fn has_project_marker(path: &Path) -> bool {
    [".git", ".hg", ".jj", ".vecgrep"]
        .iter()
        .any(|marker| path.join(marker).exists())
}

fn resolve_input_path(cwd: &Path, input: &str) -> PathBuf {
    let path = Path::new(input);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    absolute.canonicalize().unwrap_or(absolute)
}

fn capped_chunk_size(chunk_size: usize, context_tokens: Option<usize>) -> usize {
    match context_tokens {
        Some(ctx) => chunk_size.min(ctx),
        None => chunk_size,
    }
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

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn render_progress(frame_idx: usize, progress: CliIndexingProgress) {
    let frame = SPINNER_FRAMES[frame_idx % SPINNER_FRAMES.len()];
    eprint!(
        "\r{} {} files | {} chunks | {} walked",
        frame, progress.indexed_count, progress.indexed_chunks, progress.walked_count
    );
    std::io::stderr().flush().ok();
}

fn clear_progress_line() {
    eprint!("\r\x1b[2K");
    std::io::stderr().flush().ok();
}

#[derive(Debug)]
enum StaleRemovalScope {
    Prefix(PathBuf),
    None,
}

#[derive(Debug)]
struct PathPlan {
    project_root: PathBuf,
    cwd_suffix: PathBuf,
    stale_removal_scope: StaleRemovalScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Cli,
    Interactive,
    Serve,
}

type WalkerHandle = std::thread::JoinHandle<anyhow::Result<usize>>;

struct Invocation {
    args: Args,
    path_plan: PathPlan,
    query: String,
    run_mode: RunMode,
    color_choice: termcolor::ColorChoice,
}

#[derive(Clone, Copy)]
struct CliOutputContext<'a> {
    color_choice: termcolor::ColorChoice,
    cwd_suffix: &'a Path,
    quiet: bool,
    root: &'a str,
}

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

/// Returns a canonicalized project root path.
fn resolve_project_root(cwd: &Path, paths: &[String]) -> PathBuf {
    let cwd_project_root = find_project_root(cwd);

    if has_project_marker(&cwd_project_root) {
        cwd_project_root
    } else if paths.len() == 1 {
        let resolved = resolve_input_path(cwd, &paths[0]);
        if resolved.is_dir() {
            find_project_root(&resolved)
        } else {
            cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
        }
    } else {
        cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())
    }
}

/// Classify input paths as admitted/skipped/rejected in one pass.
/// Returns a PathPlan with admitted paths and an updated Args.
fn admit_paths(args: Args, cwd: &Path, project_root: &Path) -> Result<(Args, PathPlan)> {
    let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let cwd_suffix = cwd_canon
        .strip_prefix(project_root)
        .unwrap_or(Path::new(""))
        .to_path_buf();

    let mut admitted = Vec::new();
    let mut outside = Vec::new();
    let mut single_admitted_dir: Option<PathBuf> = None;

    for input in &args.paths {
        let absolute = resolve_input_path(cwd, input);
        if absolute.starts_with(project_root) {
            if args.paths.len() == 1 && absolute.is_dir() {
                single_admitted_dir = Some(absolute);
            }
            admitted.push(input.clone());
        } else {
            outside.push(input.clone());
        }
    }

    if !outside.is_empty() && !args.skip_outside_root {
        anyhow::bail!(
            "Path '{}' is outside the selected project root '{}'. Run vecgrep from that project, invoke it separately per root, or pass --skip-outside-root to ignore such paths.",
            outside[0],
            project_root.display()
        );
    }
    if !outside.is_empty() && args.skip_outside_root && !args.quiet {
        eprintln!(
            "Skipping {} path(s) outside project root {}.",
            outside.len(),
            project_root.display()
        );
    }
    if admitted.is_empty() {
        anyhow::bail!(
            "All provided paths are outside the selected project root '{}'.",
            project_root.display()
        );
    }

    let stale_removal_scope = match single_admitted_dir {
        Some(dir) => {
            let walk_prefix = dir
                .strip_prefix(project_root)
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            StaleRemovalScope::Prefix(walk_prefix)
        }
        None => StaleRemovalScope::None,
    };

    let args = Args {
        paths: admitted.clone(),
        ..args
    };
    let plan = PathPlan {
        project_root: project_root.to_path_buf(),
        cwd_suffix,
        stale_removal_scope,
    };
    Ok((args, plan))
}

fn resolve_query(args: &Args) -> String {
    args.query.clone().unwrap_or_default()
}

fn determine_run_mode(args: &Args) -> RunMode {
    if args.serve {
        RunMode::Serve
    } else if args.interactive {
        RunMode::Interactive
    } else {
        RunMode::Cli
    }
}

fn build_invocation(
    args: Args,
    path_plan: PathPlan,
    query: String,
    run_mode: RunMode,
    color_choice: termcolor::ColorChoice,
) -> Invocation {
    Invocation {
        args,
        path_plan,
        query,
        run_mode,
        color_choice,
    }
}

fn resolve_invocation(mut args: Args, cwd: &Path, project_root: &Path) -> Result<Invocation> {
    let config = vecgrep::config::load_config(project_root);
    resolve_config(&mut args, &config);
    let (args, path_plan) = admit_paths(args, cwd, project_root)?;
    let query = resolve_query(&args);
    let run_mode = determine_run_mode(&args);
    let color_choice = output::resolve_color_choice(args.color.as_ref().unwrap());

    Ok(build_invocation(
        args,
        path_plan,
        query,
        run_mode,
        color_choice,
    ))
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

    let batch_size = 32;
    let walk_opts = build_walk_options(&invocation.args);
    let (tx, rx) = std::sync::mpsc::sync_channel::<walker::WalkedFile>(batch_size * 2);
    let walk_paths = invocation.args.paths.clone();
    let stream_progress = Arc::new(walker::StreamProgress::new());
    let walker_progress = Arc::clone(&stream_progress);
    let walker_handle = std::thread::spawn(move || {
        walker::walk_paths_streaming_with_progress(&walk_paths, &walk_opts, tx, walker_progress)
    });

    let indexer = pipeline::StreamingIndexer::new(
        rx,
        invocation.args.chunk_size.unwrap(),
        invocation.args.chunk_overlap.unwrap(),
        batch_size,
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
    indexer.drain_all(embedder, idx, |progress| {
        if !quiet && !threshold_prompted && threshold > 0 && progress.indexed_count >= threshold {
            threshold_prompted = true;
            if show_spinner {
                clear_progress_line();
            }
            eprintln!(
                "Warning: {} files need indexing so far (still scanning).",
                progress.indexed_count
            );
            if !confirm_continue()? {
                eprintln!("Aborted.");
                aborted = true;
                return Ok(false);
            }
        }
        if show_spinner {
            render_progress(frame_idx, progress);
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

fn drain_remaining_indexing(
    indexer: &mut pipeline::StreamingIndexer,
    embedder: &mut Embedder,
    idx: &Index,
    show_spinner: bool,
) -> Result<()> {
    let mut frame_idx = 0;
    indexer.drain_all(embedder, idx, |progress| {
        if show_spinner {
            render_progress(frame_idx, progress);
            frame_idx += 1;
        }
        Ok(true)
    })?;

    if show_spinner {
        clear_progress_line();
    }

    Ok(())
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
    path_plan: &PathPlan,
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
) -> Result<bool> {
    serve::run_streaming(
        embedder,
        idx,
        indexer,
        invocation.args.port,
        invocation.args.top_k.unwrap(),
        invocation.args.threshold.unwrap(),
        output.quiet,
        output.root,
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
) -> Result<bool> {
    tui::interactive::run_streaming(
        embedder,
        idx,
        indexer,
        &invocation.query,
        invocation.args.top_k.unwrap(),
        invocation.args.threshold.unwrap(),
        output.cwd_suffix,
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
    top_k: usize,
    threshold: f32,
    output: CliOutputContext<'_>,
) -> Result<bool> {
    let chunk_count = idx.chunk_count()?;
    status!(output.quiet, "Index has {} chunks.", chunk_count);

    let query_embedding = embedder.embed(query)?;
    let results = idx.search(&query_embedding, top_k, threshold)?;

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

/// Merge CLI args with config file values: cli > config > hardcoded defaults.
/// Bool flags use ||: CLI flag present wins, else config value, else false.
/// Option fields use .or(): CLI value if present, else config value, else default.
fn resolve_config(args: &mut Args, config: &vecgrep::config::Config) {
    use vecgrep::cli::*;

    // Value fields: cli.or(config).or(default)
    args.top_k = args.top_k.or(config.top_k).or(Some(DEFAULT_TOP_K));
    args.threshold = args
        .threshold
        .or(config.threshold)
        .or(Some(DEFAULT_THRESHOLD));
    args.context = args.context.or(config.context).or(Some(DEFAULT_CONTEXT));
    args.chunk_size = args
        .chunk_size
        .or(config.chunk_size)
        .or(Some(DEFAULT_CHUNK_SIZE));
    args.chunk_overlap = args
        .chunk_overlap
        .or(config.chunk_overlap)
        .or(Some(DEFAULT_CHUNK_OVERLAP));
    args.index_warn_threshold = args
        .index_warn_threshold
        .or(config.index_warn_threshold)
        .or(Some(DEFAULT_INDEX_WARN_THRESHOLD));

    // Option fields: cli.or(config)
    args.embedder_url = args
        .embedder_url
        .take()
        .or_else(|| config.embedder_url.clone());
    args.embedder_model = args
        .embedder_model
        .take()
        .or_else(|| config.embedder_model.clone());
    args.max_depth = args.max_depth.or(config.max_depth);

    // Bool flags: CLI flag || config value
    args.full_index = args.full_index || config.full_index.unwrap_or(false);
    args.hidden = args.hidden || config.hidden.unwrap_or(false);
    args.follow = args.follow || config.follow.unwrap_or(false);
    args.no_ignore = args.no_ignore || config.no_ignore.unwrap_or(false);
    args.quiet = args.quiet || config.quiet.unwrap_or(false);

    // Ignore files: additive merge
    if let Some(ref config_files) = config.ignore_files {
        let cli_files = args.ignore_file.get_or_insert_with(Vec::new);
        for f in config_files {
            if !cli_files.contains(f) {
                cli_files.push(f.clone());
            }
        }
    }

    // Color: cli.or(config parsed).or(Auto)
    if args.color.is_none() {
        args.color = config.color.as_deref().and_then(|c| match c {
            "always" => Some(ColorChoice::Always),
            "never" => Some(ColorChoice::Never),
            _ => None,
        });
    }
    if args.color.is_none() {
        args.color = Some(ColorChoice::Auto);
    }
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

    let cwd = std::env::current_dir()?;
    let project_root = resolve_project_root(&cwd, &args.paths);

    // Handle --show-root (no model or index needed)
    if args.show_root {
        println!("{}", project_root.display());
        return Ok(true);
    }

    let mut invocation = resolve_invocation(args, &cwd, &project_root)?;
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
    {
        return Ok(true);
    }

    let show_spinner = !quiet && std::io::stderr().is_terminal();

    // CLI and --index-only always drain before proceeding so first-run searches
    // don't miss freshly discovered files. TUI/serve stay progressive unless
    // --full-index was requested explicitly.
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

    // TUI and serve: pass the indexer for progressive indexing.
    // If --full-index was used, the indexer is already drained.
    if matches!(invocation.run_mode, RunMode::Serve) {
        return run_serve_mode(
            embedder,
            idx,
            indexer,
            &invocation,
            output,
            &mut walker_handle,
        );
    }

    if matches!(invocation.run_mode, RunMode::Interactive) {
        return run_interactive_mode(
            embedder,
            idx,
            indexer,
            &invocation,
            output,
            &mut walker_handle,
        );
    }

    let found = run_cli_search(
        &mut embedder,
        &idx,
        &invocation.args,
        &invocation.query,
        invocation.args.top_k.unwrap(),
        invocation.args.threshold.unwrap(),
        output,
    )?;

    // TUI/serve may return here with indexing still in progress, but CLI has
    // already drained above.
    if !indexer.indexing_done {
        drain_remaining_indexing(&mut indexer, &mut embedder, &idx, show_spinner)?;
        finish_indexing(
            &mut indexer,
            &idx,
            &invocation.path_plan.stale_removal_scope,
            quiet,
            &mut walker_handle,
        )?;
    }

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
                let prefix = format!("{}/", walk_prefix.display());
                idx.remove_stale_files(&indexer.all_paths, Some(&prefix))?
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
            "Indexing complete. {} files, {} chunks, {} walked.",
            indexer.indexed_count,
            indexer.indexed_chunks,
            indexer.all_paths.len()
        );
    } else {
        status!(
            quiet,
            "Index is up to date. Scanned {} files.",
            indexer.all_paths.len()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;
    use tempfile::TempDir;
    use vecgrep::embedder::EMBEDDING_DIM;
    use vecgrep::types::{Chunk, IndexConfig};

    fn parse_args(argv: &[&str]) -> Args {
        Args::parse_from(argv)
    }

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
    fn test_drain_initial_indexing_aborts_cleanly() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        tx.send(walker::WalkedFile {
            rel_path: "main.rs".to_string(),
            content: "fn search_target() {}\n".to_string(),
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

    #[test]
    fn test_capped_chunk_size_reduces_local_default_to_model_context() {
        assert_eq!(capped_chunk_size(256, Some(256)), 256);
    }

    #[test]
    fn test_capped_chunk_size_keeps_smaller_values() {
        assert_eq!(capped_chunk_size(200, Some(256)), 200);
    }

    #[test]
    fn test_capped_chunk_size_without_context_is_unchanged() {
        assert_eq!(capped_chunk_size(500, None), 500);
    }

    #[test]
    fn test_admit_paths_sets_prefix_scope_for_single_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let src = dir.path().join("src").display().to_string();

        let args = parse_args(&["vecgrep", "needle", &src]);
        let (result_args, plan) = admit_paths(args, &cwd, &cwd).unwrap();

        assert_eq!(result_args.paths, vec![src]);
        match plan.stale_removal_scope {
            StaleRemovalScope::Prefix(ref prefix) => assert_eq!(prefix, Path::new("src")),
            _ => panic!("expected prefix stale removal scope"),
        }
    }

    #[test]
    fn test_admit_paths_sets_no_scope_for_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn main() {}").unwrap();
        let cwd = dir.path().canonicalize().unwrap();

        let args = parse_args(&["vecgrep", "needle", "lib.rs"]);
        let (_, plan) = admit_paths(args, &cwd, &cwd).unwrap();

        assert!(matches!(plan.stale_removal_scope, StaleRemovalScope::None));
    }

    #[test]
    fn test_admit_paths_rejects_all_outside_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let outside = TempDir::new().unwrap();

        let outside_path = outside.path().join("elsewhere.rs").display().to_string();
        let args = parse_args(&["vecgrep", "needle", &outside_path]);

        let err = admit_paths(args, &cwd, &cwd).unwrap_err();
        assert!(err
            .to_string()
            .contains("outside the selected project root"));
    }

    #[test]
    fn test_resolve_query_for_cli_and_serve_modes() {
        let cli_args = parse_args(&["vecgrep", "needle"]);
        assert_eq!(resolve_query(&cli_args), "needle");

        let serve_args = parse_args(&["vecgrep", "--serve"]);
        assert_eq!(resolve_query(&serve_args), "");
    }

    #[test]
    fn test_determine_run_mode_prefers_expected_mode() {
        let serve_args = parse_args(&["vecgrep", "--serve"]);
        assert_eq!(determine_run_mode(&serve_args), RunMode::Serve);

        let interactive_args = parse_args(&["vecgrep", "--interactive"]);
        assert_eq!(determine_run_mode(&interactive_args), RunMode::Interactive);

        let index_only_args = parse_args(&["vecgrep", "--index-only"]);
        assert_eq!(determine_run_mode(&index_only_args), RunMode::Cli);
    }

    #[test]
    fn test_build_invocation_carries_runtime_fields() {
        let args = parse_args(&[
            "vecgrep",
            "--top-k",
            "7",
            "--threshold",
            "0.45",
            "--chunk-size",
            "123",
            "--chunk-overlap",
            "17",
            "--full-index",
            "--quiet",
            "needle",
        ]);

        let invocation = build_invocation(
            args,
            PathPlan {
                project_root: PathBuf::from("."),
                cwd_suffix: PathBuf::new(),
                stale_removal_scope: StaleRemovalScope::None,
            },
            "needle".to_string(),
            RunMode::Cli,
            termcolor::ColorChoice::Never,
        );

        assert_eq!(invocation.args.chunk_size, Some(123));
        assert_eq!(invocation.args.chunk_overlap, Some(17));
        assert!(invocation.args.full_index);
        assert!(invocation.args.quiet);
        assert_eq!(invocation.args.top_k, Some(7));
        assert_eq!(invocation.args.threshold, Some(0.45));
    }

    #[test]
    fn test_resolve_config_sets_bool_and_color_when_cli_omits_them() {
        let mut args = parse_args(&["vecgrep", "needle"]);
        let config = vecgrep::config::Config {
            hidden: Some(true),
            follow: Some(true),
            no_ignore: Some(true),
            quiet: Some(true),
            full_index: Some(true),
            color: Some("always".to_string()),
            ..Default::default()
        };

        resolve_config(&mut args, &config);

        assert!(args.hidden);
        assert!(args.follow);
        assert!(args.no_ignore);
        assert!(args.quiet);
        assert!(args.full_index);
        assert_eq!(args.color, Some(vecgrep::cli::ColorChoice::Always));
    }

    #[test]
    fn test_resolve_config_does_not_override_explicit_cli_color() {
        let mut args = parse_args(&["vecgrep", "--color", "never", "needle"]);
        let config = vecgrep::config::Config {
            color: Some("always".to_string()),
            ..Default::default()
        };

        resolve_config(&mut args, &config);

        assert_eq!(args.color, Some(vecgrep::cli::ColorChoice::Never));
    }

    #[test]
    fn test_handle_pre_execution_actions_returns_after_stats_without_query() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let path_plan = PathPlan {
            project_root: dir.path().canonicalize().unwrap(),
            cwd_suffix: PathBuf::new(),
            stale_removal_scope: StaleRemovalScope::Prefix(PathBuf::new()),
        };
        let args = parse_args(&["vecgrep", "--stats"]);

        let outcome = handle_pre_execution_actions(&args, &path_plan, true).unwrap();

        assert_eq!(outcome, Some(true));
    }

    #[test]
    fn test_handle_post_index_actions_returns_after_index_only() {
        let index = Index::open_in_memory().unwrap();
        let args = parse_args(&["vecgrep", "--index-only"]);

        let outcome = handle_post_index_actions(&args, &index).unwrap();

        assert_eq!(outcome, Some(true));
    }

    #[test]
    fn test_finish_indexing_removes_stale_files_for_all_scope() {
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
    fn test_finish_indexing_removes_stale_files_only_under_prefix_scope() {
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
    fn test_resolve_invocation_applies_project_config_when_cli_omits_flag() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "top_k = 42\nthreshold = 0.15\nquiet = true\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&["vecgrep", "needle"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.args.top_k, Some(42));
        assert_eq!(invocation.args.threshold, Some(0.15));
        assert!(invocation.args.quiet);
    }

    #[test]
    fn test_resolve_invocation_keeps_cli_values_over_project_config() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "top_k = 42\nthreshold = 0.15\nquiet = true\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&["vecgrep", "--top-k", "7", "--threshold", "0.6", "needle"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.args.top_k, Some(7));
        assert_eq!(invocation.args.threshold, Some(0.6));
        assert!(invocation.args.quiet);
    }

    #[test]
    fn test_resolve_invocation_merges_ignore_files_additively() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "ignore_files = [\"from-config.ignore\", \"shared.ignore\"]\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&[
            "vecgrep",
            "--ignore-file",
            "from-cli.ignore",
            "--ignore-file",
            "shared.ignore",
            "needle",
        ]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(
            invocation.args.ignore_file,
            Some(vec![
                "from-cli.ignore".to_string(),
                "shared.ignore".to_string(),
                "from-config.ignore".to_string(),
            ])
        );
    }

    #[test]
    fn test_resolve_invocation_uses_empty_query_for_serve_mode() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();

        let args = parse_args(&["vecgrep", "--serve"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.run_mode, RunMode::Serve);
        assert!(invocation.query.is_empty());
    }
}
