use anyhow::{Context, Result};
use clap::Parser;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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

fn split_paths_by_root(
    paths: &[String],
    cwd: &Path,
    project_root: &Path,
) -> (Vec<String>, Vec<String>) {
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for input in paths {
        let resolved = resolve_input_path(cwd, input);
        if !resolved.starts_with(project_root) {
            outside.push(input.clone());
        } else {
            inside.push(input.clone());
        }
    }
    (inside, outside)
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

struct CliProgressRenderer {
    frames: &'static [&'static str],
    frame_idx: usize,
    rendered_once: bool,
}

impl CliProgressRenderer {
    fn new() -> Self {
        Self {
            frames: &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            frame_idx: 0,
            rendered_once: false,
        }
    }

    fn render(&mut self, progress: CliIndexingProgress) {
        let frame = self.frames[self.frame_idx % self.frames.len()];
        self.frame_idx += 1;
        self.rendered_once = true;
        eprint!(
            "\r{} {} files | {} chunks | {} walked",
            frame, progress.indexed_count, progress.indexed_chunks, progress.walked_count
        );
        std::io::stderr().flush().ok();
    }

    fn finish(&mut self) {
        if self.rendered_once {
            eprint!("\r\x1b[2K");
            std::io::stderr().flush().ok();
            self.rendered_once = false;
        }
    }
}

struct CliProgressReporter {
    progress: Arc<Mutex<Option<CliIndexingProgress>>>,
    running: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CliProgressReporter {
    fn new() -> Self {
        let progress = Arc::new(Mutex::new(None));
        let running = Arc::new(AtomicBool::new(true));
        let paused = Arc::new(AtomicBool::new(false));
        let thread_progress = Arc::clone(&progress);
        let thread_running = Arc::clone(&running);
        let thread_paused = Arc::clone(&paused);

        let handle = std::thread::spawn(move || {
            let mut renderer = CliProgressRenderer::new();
            while thread_running.load(Ordering::Relaxed) {
                if !thread_paused.load(Ordering::Relaxed) {
                    let latest = *thread_progress.lock().unwrap();
                    if let Some(progress) = latest {
                        renderer.render(progress);
                    }
                }
                thread::sleep(Duration::from_millis(80));
            }
            renderer.finish();
        });

        Self {
            progress,
            running,
            paused,
            handle: Some(handle),
        }
    }

    fn update(&self, progress: CliIndexingProgress) {
        *self.progress.lock().unwrap() = Some(progress);
    }

    fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }

    fn finish(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CliProgressReporter {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

enum StaleRemovalScope {
    All,
    Prefix(PathBuf),
    None,
}

struct PathPlan {
    project_root: PathBuf,
    project_root_canon: PathBuf,
    cwd_suffix: PathBuf,
    inside_paths: Vec<String>,
    outside_paths: Vec<String>,
    stale_removal_scope: StaleRemovalScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Cli,
    Interactive,
    Serve,
    IndexOnly,
}

struct RuntimeConfig {
    top_k: usize,
    threshold: f32,
    chunk_size: usize,
    chunk_overlap: usize,
    quiet: bool,
    full_index: bool,
    index_warn_threshold: usize,
    embedder_url: Option<String>,
    embedder_model: Option<String>,
    port: Option<u16>,
}

fn resolve_project_root(cwd: &Path, paths: &[String]) -> PathBuf {
    let cwd_project_root = find_project_root(cwd);

    if has_project_marker(&cwd_project_root) {
        cwd_project_root
    } else if paths.len() == 1 && Path::new(&paths[0]).is_dir() {
        find_project_root(Path::new(&paths[0]))
    } else {
        cwd.to_path_buf()
    }
}

fn build_path_plan(cwd: &Path, project_root: &Path, paths: &[String]) -> PathPlan {
    let project_root_canon = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let (inside_paths, outside_paths) = split_paths_by_root(paths, &cwd_canon, &project_root_canon);

    let cwd_suffix = cwd_canon
        .strip_prefix(&project_root_canon)
        .unwrap_or(Path::new(""))
        .to_path_buf();

    let stale_removal_scope = if inside_paths.len() == 1 && Path::new(&inside_paths[0]).is_dir() {
        let walk_root_abs = Path::new(&inside_paths[0])
            .canonicalize()
            .unwrap_or_else(|_| cwd_canon.join(&inside_paths[0]));
        let walk_prefix = walk_root_abs
            .strip_prefix(&project_root_canon)
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        if walk_prefix.as_os_str().is_empty() {
            StaleRemovalScope::All
        } else {
            StaleRemovalScope::Prefix(walk_prefix)
        }
    } else {
        StaleRemovalScope::None
    };

    PathPlan {
        project_root: project_root.to_path_buf(),
        project_root_canon,
        cwd_suffix,
        inside_paths,
        outside_paths,
        stale_removal_scope,
    }
}

fn apply_path_plan(args: &mut Args, plan: &PathPlan) -> Result<()> {
    if !plan.outside_paths.is_empty() && !args.skip_outside_root {
        anyhow::bail!(
            "Path '{}' is outside the selected project root '{}'. Run vecgrep from that project, invoke it separately per root, or pass --skip-outside-root to ignore such paths.",
            plan.outside_paths[0],
            plan.project_root_canon.display()
        );
    }
    if !plan.outside_paths.is_empty() && args.skip_outside_root && !args.quiet {
        eprintln!(
            "Skipping {} path(s) outside project root {}.",
            plan.outside_paths.len(),
            plan.project_root_canon.display()
        );
    }
    if plan.inside_paths.is_empty() {
        anyhow::bail!(
            "All provided paths are outside the selected project root '{}'.",
            plan.project_root_canon.display()
        );
    }
    args.paths = plan.inside_paths.clone();
    Ok(())
}

fn resolve_query(args: &Args) -> Result<String> {
    match &args.query {
        Some(q) => Ok(q.clone()),
        None if args.interactive => Ok(String::new()),
        None if args.serve || args.index_only || args.stats || args.reindex => Ok(String::new()),
        None => Ok(String::new()),
    }
}

fn determine_run_mode(args: &Args) -> RunMode {
    if args.serve {
        RunMode::Serve
    } else if args.interactive {
        RunMode::Interactive
    } else if args.index_only {
        RunMode::IndexOnly
    } else {
        RunMode::Cli
    }
}

fn build_runtime_config(args: &Args) -> RuntimeConfig {
    RuntimeConfig {
        top_k: args.top_k,
        threshold: args.threshold,
        chunk_size: args.chunk_size,
        chunk_overlap: args.chunk_overlap,
        quiet: args.quiet,
        full_index: args.full_index,
        index_warn_threshold: args.index_warn_threshold,
        embedder_url: args.embedder_url.clone(),
        embedder_model: args.embedder_model.clone(),
        port: args.port,
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

fn run_serve_mode(
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    port: Option<u16>,
    top_k: usize,
    threshold: f32,
    quiet: bool,
    root: &str,
    walker_handle: &mut Option<std::thread::JoinHandle<anyhow::Result<usize>>>,
) -> Result<bool> {
    serve::run_streaming(embedder, idx, indexer, port, top_k, threshold, quiet, root)?;
    if let Some(h) = walker_handle.take() {
        let _ = h.join();
    }
    Ok(true)
}

fn run_interactive_mode(
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    query: &str,
    top_k: usize,
    threshold: f32,
    cwd_suffix: &Path,
    walker_handle: &mut Option<std::thread::JoinHandle<anyhow::Result<usize>>>,
) -> Result<bool> {
    tui::interactive::run_streaming(embedder, idx, indexer, query, top_k, threshold, cwd_suffix)?;
    if let Some(h) = walker_handle.take() {
        let _ = h.join();
    }
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

    if !args.json && !cwd_suffix.as_os_str().is_empty() {
        for r in &mut results {
            r.chunk.file_path = paths::to_cwd_relative(&r.chunk.file_path, cwd_suffix);
        }
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

/// Apply config file values where CLI flags weren't explicitly provided.
fn apply_config(args: &mut Args, config: &vecgrep::config::Config) {
    // Option fields: apply if CLI is None
    if args.embedder_url.is_none() {
        args.embedder_url.clone_from(&config.embedder_url);
    }
    if args.embedder_model.is_none() {
        args.embedder_model.clone_from(&config.embedder_model);
    }
    if args.max_depth.is_none() {
        args.max_depth = config.max_depth;
    }

    // Fields with clap defaults: apply config if value matches the hardcoded default
    macro_rules! apply_default {
        ($field:ident, $default:expr) => {
            if args.$field == $default {
                if let Some(v) = config.$field {
                    args.$field = v;
                }
            }
        };
    }

    apply_default!(top_k, 10);
    apply_default!(threshold, 0.3);
    apply_default!(context, 3);
    apply_default!(chunk_size, 500);
    apply_default!(chunk_overlap, 100);
    apply_default!(index_warn_threshold, 1000);

    // Bool fields: apply config if CLI is false (default)
    macro_rules! apply_bool {
        ($field:ident) => {
            if !args.$field {
                if let Some(true) = config.$field {
                    args.$field = true;
                }
            }
        };
    }

    apply_bool!(full_index);
    apply_bool!(hidden);
    apply_bool!(follow);
    apply_bool!(no_ignore);
    apply_bool!(quiet);

    // Ignore files: merge config into CLI (additive)
    if let Some(ref config_files) = config.ignore_files {
        let cli_files = args.ignore_file.get_or_insert_with(Vec::new);
        for f in config_files {
            if !cli_files.contains(f) {
                cli_files.push(f.clone());
            }
        }
    }

    // Color: apply if CLI is Auto (default)
    if matches!(args.color, vecgrep::cli::ColorChoice::Auto) {
        if let Some(ref c) = config.color {
            match c.as_str() {
                "always" => args.color = vecgrep::cli::ColorChoice::Always,
                "never" => args.color = vecgrep::cli::ColorChoice::Never,
                _ => {}
            }
        }
    }
}

/// Returns Ok(true) if matches were found, Ok(false) if no matches.
fn run() -> Result<bool> {
    // Initialize tracing from VECGREP_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("VECGREP_LOG"))
        .with_writer(std::io::stderr)
        .init();

    let mut args = Args::parse();

    // Handle --type-list (no model or index needed)
    if args.type_list {
        walker::print_type_list();
        return Ok(true);
    }

    let color_choice = output::resolve_color_choice(&args.color);

    let cwd = std::env::current_dir()?;
    let project_root = resolve_project_root(&cwd, &args.paths);

    // Apply config: project (.vecgrep/config.toml) > global (~/.config/vecgrep/config.toml) > defaults
    let config = vecgrep::config::load_config(&project_root);
    apply_config(&mut args, &config);

    // Handle --show-root (no model or index needed)
    if args.show_root {
        let canon = project_root
            .canonicalize()
            .unwrap_or_else(|_| project_root.clone());
        println!("{}", canon.display());
        return Ok(true);
    }
    let path_plan = build_path_plan(&cwd, &project_root, &args.paths);
    apply_path_plan(&mut args, &path_plan)?;

    let mut runtime = build_runtime_config(&args);
    let quiet = runtime.quiet;

    // Handle --clear-cache
    if args.clear_cache {
        let cache_dir = path_plan.project_root.join(".vecgrep");
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
        let idx = Index::open(&path_plan.project_root)?;
        print_index_stats(&idx)?;
        return Ok(true);
    }

    // Initialize embedder
    let mut embedder =
        if let (Some(ref url), Some(ref model)) = (&runtime.embedder_url, &runtime.embedder_model) {
            status!(quiet, "Using external embedder: {} ({})", url, model);
            let mut e = Embedder::new_remote(url, model);
            // Probe to discover embedding dimension before building config
            e.embed("probe")
                .context("Failed to connect to external embedder")?;
            status!(quiet, "Embedding dimension: {}", e.embedding_dim());
            e
        } else {
            if !quiet {
                eprint!("Loading model...");
                std::io::stderr().flush().ok();
            }
            let e = Embedder::new_local().context("Failed to initialize embedder")?;
            if !quiet {
                eprintln!(" done.");
            }
            e
        };

    let original_chunk_size = runtime.chunk_size;
    runtime.chunk_size = capped_chunk_size(runtime.chunk_size, embedder.context_tokens());
    if runtime.chunk_size < original_chunk_size {
        status!(
            quiet,
            "Reducing chunk_size from {} to {} (model context limit)",
            original_chunk_size,
            runtime.chunk_size
        );
    }

    // Open or create index
    let idx = Index::open(&path_plan.project_root)?;

    let config = IndexConfig {
        model_name: embedder.model_name().to_string(),
        embedding_dim: embedder.embedding_dim(),
        chunk_size: runtime.chunk_size,
        chunk_overlap: runtime.chunk_overlap,
    };

    // Check if config changed
    let config_valid = idx.check_config(&config)?;
    if !config_valid || args.reindex {
        if !config_valid {
            status!(quiet, "Index configuration changed, rebuilding...");
        }
        idx.rebuild_for_config(&config)?;
    } else {
        idx.set_config(&config)?;
    }

    // Stream files from walker thread through a channel
    let walk_opts = walker::WalkOptions {
        file_types: args.file_type.clone(),
        file_types_not: args.file_type_not.clone(),
        globs: args.glob.clone(),
        ignore_files: args.ignore_file.clone(),
        hidden: args.hidden,
        follow: args.follow,
        no_ignore: args.no_ignore,
        max_depth: args.max_depth,
    };

    let batch_size = 32;
    // 2× batch_size so the walker stays ahead of the embedder — keeps the
    // model fed even on slow I/O (NFS, spinning disks, etc.).
    let (tx, rx) = std::sync::mpsc::sync_channel::<walker::WalkedFile>(batch_size * 2);
    let walk_paths = args.paths.clone();
    let stream_progress = Arc::new(walker::StreamProgress::new());
    let walker_progress = Arc::clone(&stream_progress);
    let walker_join_handle = std::thread::spawn(move || {
        walker::walk_paths_streaming_with_progress(&walk_paths, &walk_opts, tx, walker_progress)
    });

    let threshold = runtime.index_warn_threshold;
    let root_str = path_plan.project_root_canon.to_string_lossy().to_string();

    let indexer = pipeline::StreamingIndexer::new(
        rx,
        runtime.chunk_size,
        runtime.chunk_overlap,
        batch_size,
        &path_plan.cwd_suffix,
        Some(stream_progress),
    );

    let query = resolve_query(&args)?;
    let run_mode = determine_run_mode(&args);
    if query.is_empty() && matches!(run_mode, RunMode::Cli) && !args.stats && !args.reindex {
        return Ok(true);
    }

    let mut indexer = indexer;
    let mut walker_handle = Some(walker_join_handle);
    let mut progress_reporter =
        (!quiet && std::io::stderr().is_terminal()).then(CliProgressReporter::new);

    // CLI and --index-only always drain before proceeding so first-run searches
    // don't miss freshly discovered files. TUI/serve stay progressive unless
    // --full-index was requested explicitly.
    let must_drain_before_search = matches!(run_mode, RunMode::IndexOnly | RunMode::Cli);
    if runtime.full_index || must_drain_before_search {
        let mut threshold_prompted = false;
        indexer.drain_all(&mut embedder, &idx, |progress| {
            if !quiet && !threshold_prompted && threshold > 0 && progress.indexed_count >= threshold
            {
                threshold_prompted = true;
                if let Some(reporter) = progress_reporter.as_ref() {
                    reporter.pause();
                }
                eprintln!(
                    "Warning: {} files need indexing so far (still scanning).",
                    progress.indexed_count
                );
                if std::io::stdin().is_terminal() {
                    eprint!("Continue? [y/N] ");
                    std::io::stderr().flush().ok();
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input)?;
                    if !input.trim().eq_ignore_ascii_case("y") {
                        eprintln!("Aborted.");
                        return Ok(false);
                    }
                }
                if let Some(reporter) = progress_reporter.as_ref() {
                    reporter.resume();
                }
            }
            if let Some(reporter) = progress_reporter.as_ref() {
                reporter.update(progress);
            }
            Ok(true)
        })?;
        if let Some(reporter) = progress_reporter.take() {
            reporter.finish();
        }

        finish_indexing(
            &mut indexer,
            &idx,
            &path_plan.stale_removal_scope,
            quiet,
            &mut walker_handle,
        )?;
    }

    // Handle --index-only
    if args.index_only {
        print_index_stats(&idx)?;
        return Ok(true);
    }

    if args.reindex && args.query.is_none() {
        return Ok(true);
    }

    // Handle --stats after indexing
    if args.stats {
        print_index_stats(&idx)?;
        if args.query.is_none() {
            return Ok(true);
        }
    }

    // TUI and serve: pass the indexer for progressive indexing.
    // If --full-index was used, the indexer is already drained.
    if matches!(run_mode, RunMode::Serve) {
        return run_serve_mode(
            embedder,
            idx,
            indexer,
            runtime.port,
            runtime.top_k,
            runtime.threshold,
            quiet,
            &root_str,
            &mut walker_handle,
        );
    }

    if matches!(run_mode, RunMode::Interactive) {
        return run_interactive_mode(
            embedder,
            idx,
            indexer,
            &query,
            runtime.top_k,
            runtime.threshold,
            &path_plan.cwd_suffix,
            &mut walker_handle,
        );
    }

    // CLI: search after indexing has completed.
    let chunk_count = idx.chunk_count()?;
    status!(quiet, "Index has {} chunks.", chunk_count);

    let query_embedding = embedder.embed(&query)?;

    let results = idx.search(&query_embedding, runtime.top_k, runtime.threshold)?;

    let found = render_cli_results(results, &args, color_choice, &path_plan.cwd_suffix, &root_str)?;
    if !found {
        status!(quiet, "No results found.");
    }

    // TUI/serve may return here with indexing still in progress, but CLI has
    // already drained above.
    if !indexer.indexing_done {
        indexer.drain_all(&mut embedder, &idx, |progress| {
            if let Some(reporter) = progress_reporter.as_ref() {
                reporter.update(progress);
            }
            Ok(true)
        })?;
        if let Some(reporter) = progress_reporter.take() {
            reporter.finish();
        }

        finish_indexing(
            &mut indexer,
            &idx,
            &path_plan.stale_removal_scope,
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
    walker_handle: &mut Option<std::thread::JoinHandle<anyhow::Result<usize>>>,
) -> Result<()> {
    if let Some(handle) = walker_handle.take() {
        match handle.join() {
            Ok(result) => {
                result?;
            }
            Err(_) => anyhow::bail!("Walker thread panicked"),
        }
    }

    let removed = match stale_removal_scope {
        StaleRemovalScope::All => idx.remove_stale_files(&indexer.all_paths)?,
        StaleRemovalScope::Prefix(walk_prefix) => {
            let prefix = format!("{}/", walk_prefix.display());
            idx.remove_stale_files_under(&indexer.all_paths, &prefix)?
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
    fn test_cli_progress_reporter_finish_stops_thread() {
        let reporter = CliProgressReporter::new();
        reporter.update(CliIndexingProgress {
            indexed_count: 1,
            indexed_chunks: 4,
            walked_count: 2,
        });
        std::thread::sleep(Duration::from_millis(120));
        reporter.finish();
    }

    #[test]
    fn test_cli_progress_reporter_drop_stops_thread() {
        let reporter = CliProgressReporter::new();
        reporter.update(CliIndexingProgress {
            indexed_count: 3,
            indexed_chunks: 9,
            walked_count: 5,
        });
        reporter.pause();
        reporter.resume();

        let start = Instant::now();
        drop(reporter);
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "dropping reporter should not block for long"
        );
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
        assert_eq!(capped_chunk_size(500, Some(256)), 256);
    }

    #[test]
    fn test_capped_chunk_size_keeps_smaller_values() {
        assert_eq!(capped_chunk_size(200, Some(256)), 200);
    }

    #[test]
    fn test_capped_chunk_size_without_context_is_unchanged() {
        assert_eq!(capped_chunk_size(500, None), 500);
    }
}
