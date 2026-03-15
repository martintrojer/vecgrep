use anyhow::{Context, Result};
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, FromArgMatches};
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

struct ResolvedPath {
    input: String,
    absolute: PathBuf,
    inside_root: bool,
    is_dir: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunMode {
    Cli,
    Interactive,
    Serve,
    IndexOnly,
}

type WalkerHandle = std::thread::JoinHandle<anyhow::Result<usize>>;

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

#[derive(Clone, Copy)]
struct SearchConfig {
    top_k: usize,
    threshold: f32,
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

struct Invocation {
    args: Args,
    path_plan: PathPlan,
    runtime: RuntimeConfig,
    query: String,
    run_mode: RunMode,
    color_choice: termcolor::ColorChoice,
}

struct ExecutionContext {
    embedder: Embedder,
    idx: Index,
    indexer: pipeline::StreamingIndexer,
    walker_handle: Option<WalkerHandle>,
    root: String,
}

enum FlowControl {
    Continue,
    Return(bool),
}

fn resolve_input_paths(cwd: &Path, paths: &[String], project_root: &Path) -> Vec<ResolvedPath> {
    let project_root_canon = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());

    paths
        .iter()
        .map(|input| {
            let absolute = resolve_input_path(cwd, input);
            let is_dir = absolute.is_dir();
            let inside_root = absolute.starts_with(&project_root_canon);
            ResolvedPath {
                input: input.clone(),
                absolute,
                inside_root,
                is_dir,
            }
        })
        .collect()
}

fn resolve_project_root(cwd: &Path, paths: &[String]) -> PathBuf {
    let cwd_project_root = find_project_root(cwd);

    if has_project_marker(&cwd_project_root) {
        cwd_project_root
    } else if paths.len() == 1 {
        let resolved = resolve_input_path(cwd, &paths[0]);
        if resolved.is_dir() {
            find_project_root(&resolved)
        } else {
            cwd.to_path_buf()
        }
    } else {
        cwd.to_path_buf()
    }
}

fn build_path_plan(cwd: &Path, project_root: &Path, paths: &[ResolvedPath]) -> PathPlan {
    let project_root_canon = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let inside_paths = paths
        .iter()
        .filter(|path| path.inside_root)
        .map(|path| path.input.clone())
        .collect();
    let outside_paths = paths
        .iter()
        .filter(|path| !path.inside_root)
        .map(|path| path.input.clone())
        .collect();

    let cwd_suffix = cwd_canon
        .strip_prefix(&project_root_canon)
        .unwrap_or(Path::new(""))
        .to_path_buf();

    let stale_removal_scope = match paths
        .iter()
        .filter(|path| path.inside_root)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [path] if path.is_dir => {
            let walk_prefix = path
                .absolute
                .strip_prefix(&project_root_canon)
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            if walk_prefix.as_os_str().is_empty() {
                StaleRemovalScope::All
            } else {
                StaleRemovalScope::Prefix(walk_prefix)
            }
        }
        _ => StaleRemovalScope::None,
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

fn apply_path_plan(args: Args, plan: &PathPlan) -> Result<Args> {
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
    Ok(Args {
        paths: plan.inside_paths.clone(),
        ..args
    })
}

fn resolve_query(args: &Args) -> String {
    args.query.clone().unwrap_or_default()
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

fn search_config(runtime: &RuntimeConfig) -> SearchConfig {
    SearchConfig {
        top_k: runtime.top_k,
        threshold: runtime.threshold,
    }
}

fn with_config(mut args: Args, config: &vecgrep::config::Config, matches: &ArgMatches) -> Args {
    apply_config(&mut args, config, matches);
    args
}

fn resolve_invocation(
    args: Args,
    matches: &ArgMatches,
    cwd: &Path,
    project_root: &Path,
) -> Result<Invocation> {
    let config = vecgrep::config::load_config(project_root);
    let args = with_config(args, &config, matches);
    let resolved_paths = resolve_input_paths(cwd, &args.paths, project_root);
    let path_plan = build_path_plan(cwd, project_root, &resolved_paths);
    let args = apply_path_plan(args, &path_plan)?;

    Ok(Invocation {
        query: resolve_query(&args),
        run_mode: determine_run_mode(&args),
        color_choice: output::resolve_color_choice(&args.color),
        runtime: build_runtime_config(&args),
        args,
        path_plan,
    })
}

fn initialize_embedder(runtime: &mut RuntimeConfig) -> Result<Embedder> {
    let quiet = runtime.quiet;
    let embedder = if let (Some(ref url), Some(ref model)) =
        (&runtime.embedder_url, &runtime.embedder_model)
    {
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

    Ok(embedder)
}

fn prepare_index(
    project_root: &Path,
    embedder: &Embedder,
    runtime: &RuntimeConfig,
    reindex: bool,
) -> Result<Index> {
    let idx = Index::open(project_root)?;
    let config = IndexConfig {
        model_name: embedder.model_name().to_string(),
        embedding_dim: embedder.embedding_dim(),
        chunk_size: runtime.chunk_size,
        chunk_overlap: runtime.chunk_overlap,
    };

    let config_valid = idx.check_config(&config)?;
    if !config_valid || reindex {
        if !config_valid {
            status!(runtime.quiet, "Index configuration changed, rebuilding...");
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

fn prepare_execution(
    args: &Args,
    runtime: &mut RuntimeConfig,
    path_plan: &PathPlan,
) -> Result<ExecutionContext> {
    let embedder = initialize_embedder(runtime)?;
    let idx = prepare_index(&path_plan.project_root, &embedder, runtime, args.reindex)?;

    let batch_size = 32;
    let walk_opts = build_walk_options(args);
    let (tx, rx) = std::sync::mpsc::sync_channel::<walker::WalkedFile>(batch_size * 2);
    let walk_paths = args.paths.clone();
    let stream_progress = Arc::new(walker::StreamProgress::new());
    let walker_progress = Arc::clone(&stream_progress);
    let walker_handle = std::thread::spawn(move || {
        walker::walk_paths_streaming_with_progress(&walk_paths, &walk_opts, tx, walker_progress)
    });

    let indexer = pipeline::StreamingIndexer::new(
        rx,
        runtime.chunk_size,
        runtime.chunk_overlap,
        batch_size,
        &path_plan.cwd_suffix,
        Some(stream_progress),
    );

    Ok(ExecutionContext {
        embedder,
        idx,
        indexer,
        walker_handle: Some(walker_handle),
        root: path_plan.project_root_canon.to_string_lossy().to_string(),
    })
}

fn drain_initial_indexing(
    indexer: &mut pipeline::StreamingIndexer,
    embedder: &mut Embedder,
    idx: &Index,
    quiet: bool,
    threshold: usize,
    progress_reporter: &mut Option<CliProgressReporter>,
) -> Result<IndexDrainOutcome> {
    drain_initial_indexing_with_prompt(
        indexer,
        embedder,
        idx,
        quiet,
        threshold,
        progress_reporter,
        prompt_to_continue,
    )
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

fn drain_initial_indexing_with_prompt<F>(
    indexer: &mut pipeline::StreamingIndexer,
    embedder: &mut Embedder,
    idx: &Index,
    quiet: bool,
    threshold: usize,
    progress_reporter: &mut Option<CliProgressReporter>,
    mut confirm_continue: F,
) -> Result<IndexDrainOutcome>
where
    F: FnMut() -> Result<bool>,
{
    let mut threshold_prompted = false;
    let mut aborted = false;
    indexer.drain_all(embedder, idx, |progress| {
        if !quiet && !threshold_prompted && threshold > 0 && progress.indexed_count >= threshold {
            threshold_prompted = true;
            if let Some(reporter) = progress_reporter.as_ref() {
                reporter.pause();
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
    progress_reporter: &mut Option<CliProgressReporter>,
) -> Result<()> {
    indexer.drain_all(embedder, idx, |progress| {
        if let Some(reporter) = progress_reporter.as_ref() {
            reporter.update(progress);
        }
        Ok(true)
    })?;

    if let Some(reporter) = progress_reporter.take() {
        reporter.finish();
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

fn handle_pre_execution_actions(
    args: &Args,
    path_plan: &PathPlan,
    quiet: bool,
) -> Result<FlowControl> {
    if args.clear_cache {
        let cache_dir = path_plan.project_root.join(".vecgrep");
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
            status!(quiet, "Cache cleared.");
        } else {
            status!(quiet, "No cache found.");
        }
        if args.query.is_none() {
            return Ok(FlowControl::Return(true));
        }
    }

    if args.stats && args.query.is_none() && !args.index_only {
        let idx = Index::open(&path_plan.project_root)?;
        print_index_stats(&idx)?;
        return Ok(FlowControl::Return(true));
    }

    Ok(FlowControl::Continue)
}

fn handle_post_index_actions(args: &Args, idx: &Index) -> Result<FlowControl> {
    if args.index_only {
        print_index_stats(idx)?;
        return Ok(FlowControl::Return(true));
    }

    if args.reindex && args.query.is_none() {
        return Ok(FlowControl::Return(true));
    }

    if args.stats {
        print_index_stats(idx)?;
        if args.query.is_none() {
            return Ok(FlowControl::Return(true));
        }
    }

    Ok(FlowControl::Continue)
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
    port: Option<u16>,
    search: SearchConfig,
    output: CliOutputContext<'_>,
    walker_handle: &mut Option<WalkerHandle>,
) -> Result<bool> {
    serve::run_streaming(
        embedder,
        idx,
        indexer,
        port,
        search.top_k,
        search.threshold,
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
    query: &str,
    search: SearchConfig,
    output: CliOutputContext<'_>,
    walker_handle: &mut Option<WalkerHandle>,
) -> Result<bool> {
    tui::interactive::run_streaming(
        embedder,
        idx,
        indexer,
        query,
        search.top_k,
        search.threshold,
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
    search: SearchConfig,
    output: CliOutputContext<'_>,
) -> Result<bool> {
    let chunk_count = idx.chunk_count()?;
    status!(output.quiet, "Index has {} chunks.", chunk_count);

    let query_embedding = embedder.embed(query)?;
    let results = idx.search(&query_embedding, search.top_k, search.threshold)?;

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

/// Apply config file values where CLI flags weren't explicitly provided.
fn cli_provided(matches: &ArgMatches, id: &str) -> bool {
    matches.value_source(id) == Some(ValueSource::CommandLine)
}

/// Apply config file values where CLI flags weren't explicitly provided.
fn apply_config(args: &mut Args, config: &vecgrep::config::Config, matches: &ArgMatches) {
    // Option fields: apply if CLI is None
    if !cli_provided(matches, "embedder_url") {
        args.embedder_url.clone_from(&config.embedder_url);
    }
    if !cli_provided(matches, "embedder_model") {
        args.embedder_model.clone_from(&config.embedder_model);
    }
    if !cli_provided(matches, "max_depth") {
        args.max_depth = config.max_depth;
    }

    // Fields with clap defaults: apply config unless the CLI explicitly set them.
    macro_rules! apply_value {
        ($field:ident, $id:literal) => {
            if !cli_provided(matches, $id) {
                if let Some(v) = config.$field {
                    args.$field = v;
                }
            }
        };
    }

    apply_value!(top_k, "top_k");
    apply_value!(threshold, "threshold");
    apply_value!(context, "context");
    apply_value!(chunk_size, "chunk_size");
    apply_value!(chunk_overlap, "chunk_overlap");
    apply_value!(index_warn_threshold, "index_warn_threshold");

    // Bool fields: apply config unless the CLI explicitly set them.
    macro_rules! apply_bool {
        ($field:ident, $id:literal) => {
            if !cli_provided(matches, $id) {
                if let Some(true) = config.$field {
                    args.$field = true;
                }
            }
        };
    }

    apply_bool!(full_index, "full_index");
    apply_bool!(hidden, "hidden");
    apply_bool!(follow, "follow");
    apply_bool!(no_ignore, "no_ignore");
    apply_bool!(quiet, "quiet");

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
    if !cli_provided(matches, "color") {
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

    let matches = Args::command().get_matches();
    let args = Args::from_arg_matches(&matches).expect("clap validated matches");

    // Handle --type-list (no model or index needed)
    if args.type_list {
        walker::print_type_list();
        return Ok(true);
    }

    let cwd = std::env::current_dir()?;
    let project_root = resolve_project_root(&cwd, &args.paths);

    // Handle --show-root (no model or index needed)
    if args.show_root {
        let canon = project_root
            .canonicalize()
            .unwrap_or_else(|_| project_root.clone());
        println!("{}", canon.display());
        return Ok(true);
    }

    let Invocation {
        args,
        path_plan,
        mut runtime,
        query,
        run_mode,
        color_choice,
    } = resolve_invocation(args, &matches, &cwd, &project_root)?;
    let quiet = runtime.quiet;

    if let FlowControl::Return(result) = handle_pre_execution_actions(&args, &path_plan, quiet)? {
        return Ok(result);
    }

    let threshold = runtime.index_warn_threshold;
    let ExecutionContext {
        mut embedder,
        idx,
        mut indexer,
        mut walker_handle,
        root,
    } = prepare_execution(&args, &mut runtime, &path_plan)?;
    let search = search_config(&runtime);
    let output = CliOutputContext {
        color_choice,
        cwd_suffix: &path_plan.cwd_suffix,
        quiet,
        root: &root,
    };
    if query.is_empty() && matches!(run_mode, RunMode::Cli) && !args.stats && !args.reindex {
        return Ok(true);
    }

    let mut progress_reporter =
        (!quiet && std::io::stderr().is_terminal()).then(CliProgressReporter::new);

    // CLI and --index-only always drain before proceeding so first-run searches
    // don't miss freshly discovered files. TUI/serve stay progressive unless
    // --full-index was requested explicitly.
    let must_drain_before_search = matches!(run_mode, RunMode::IndexOnly | RunMode::Cli);
    if runtime.full_index || must_drain_before_search {
        let drain_outcome = drain_initial_indexing(
            &mut indexer,
            &mut embedder,
            &idx,
            quiet,
            threshold,
            &mut progress_reporter,
        )?;
        if matches!(drain_outcome, IndexDrainOutcome::Aborted) {
            return Ok(false);
        }
        finish_indexing(
            &mut indexer,
            &idx,
            &path_plan.stale_removal_scope,
            quiet,
            &mut walker_handle,
        )?;
    }

    if let FlowControl::Return(result) = handle_post_index_actions(&args, &idx)? {
        return Ok(result);
    }

    // TUI and serve: pass the indexer for progressive indexing.
    // If --full-index was used, the indexer is already drained.
    if matches!(run_mode, RunMode::Serve) {
        return run_serve_mode(
            embedder,
            idx,
            indexer,
            runtime.port,
            search,
            output,
            &mut walker_handle,
        );
    }

    if matches!(run_mode, RunMode::Interactive) {
        return run_interactive_mode(
            embedder,
            idx,
            indexer,
            &query,
            search,
            output,
            &mut walker_handle,
        );
    }

    let found = run_cli_search(&mut embedder, &idx, &args, &query, search, output)?;

    // TUI/serve may return here with indexing still in progress, but CLI has
    // already drained above.
    if !indexer.indexing_done {
        drain_remaining_indexing(&mut indexer, &mut embedder, &idx, &mut progress_reporter)?;
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
    walker_handle: &mut Option<WalkerHandle>,
) -> Result<()> {
    join_walker(walker_handle)?;

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
    use clap::{CommandFactory, FromArgMatches};
    use std::path::Path;
    use std::time::Instant;
    use tempfile::TempDir;
    use vecgrep::embedder::EMBEDDING_DIM;
    use vecgrep::types::{Chunk, IndexConfig};

    fn parse_args(argv: &[&str]) -> (Args, ArgMatches) {
        let matches = Args::command().get_matches_from(argv);
        let args = Args::from_arg_matches(&matches).expect("clap validated matches");
        (args, matches)
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
        let mut progress_reporter = None;

        let outcome = drain_initial_indexing_with_prompt(
            &mut indexer,
            &mut embedder,
            &idx,
            false,
            1,
            &mut progress_reporter,
            || Ok(false),
        )
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

    #[test]
    fn test_build_path_plan_sets_prefix_scope_for_single_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let src = dir.path().join("src").display().to_string();

        let resolved = resolve_input_paths(&cwd, std::slice::from_ref(&src), &cwd);
        let plan = build_path_plan(&cwd, &cwd, &resolved);

        assert_eq!(plan.inside_paths, vec![src]);
        assert!(plan.outside_paths.is_empty());
        match plan.stale_removal_scope {
            StaleRemovalScope::Prefix(ref prefix) => assert_eq!(prefix, Path::new("src")),
            _ => panic!("expected prefix stale removal scope"),
        }
    }

    #[test]
    fn test_build_path_plan_sets_no_scope_for_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn main() {}").unwrap();
        let cwd = dir.path().canonicalize().unwrap();

        let resolved = resolve_input_paths(&cwd, &["lib.rs".to_string()], &cwd);
        let plan = build_path_plan(&cwd, &cwd, &resolved);

        assert!(matches!(plan.stale_removal_scope, StaleRemovalScope::None));
    }

    #[test]
    fn test_apply_path_plan_rejects_all_outside_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let outside = TempDir::new().unwrap();

        let outside_path = outside.path().join("elsewhere.rs").display().to_string();
        let resolved = resolve_input_paths(&cwd, std::slice::from_ref(&outside_path), &cwd);
        let plan = build_path_plan(&cwd, &cwd, &resolved);
        let (args, _) = parse_args(&["vecgrep", "needle"]);

        let err = apply_path_plan(args, &plan).unwrap_err();
        assert!(err
            .to_string()
            .contains("outside the selected project root"));
    }

    #[test]
    fn test_resolve_query_for_cli_and_serve_modes() {
        let (cli_args, _) = parse_args(&["vecgrep", "needle"]);
        assert_eq!(resolve_query(&cli_args), "needle");

        let (serve_args, _) = parse_args(&["vecgrep", "--serve"]);
        assert_eq!(resolve_query(&serve_args), "");
    }

    #[test]
    fn test_determine_run_mode_prefers_expected_mode() {
        let (serve_args, _) = parse_args(&["vecgrep", "--serve"]);
        assert_eq!(determine_run_mode(&serve_args), RunMode::Serve);

        let (interactive_args, _) = parse_args(&["vecgrep", "--interactive"]);
        assert_eq!(determine_run_mode(&interactive_args), RunMode::Interactive);

        let (index_only_args, _) = parse_args(&["vecgrep", "--index-only"]);
        assert_eq!(determine_run_mode(&index_only_args), RunMode::IndexOnly);
    }

    #[test]
    fn test_build_runtime_and_search_config() {
        let (args, _) = parse_args(&[
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

        let runtime = build_runtime_config(&args);
        let search = search_config(&runtime);

        assert_eq!(runtime.chunk_size, 123);
        assert_eq!(runtime.chunk_overlap, 17);
        assert!(runtime.full_index);
        assert!(runtime.quiet);
        assert_eq!(search.top_k, 7);
        assert_eq!(search.threshold, 0.45);
    }

    #[test]
    fn test_apply_config_sets_bool_and_color_when_cli_omits_them() {
        let (mut args, matches) = parse_args(&["vecgrep", "needle"]);
        let config = vecgrep::config::Config {
            hidden: Some(true),
            follow: Some(true),
            no_ignore: Some(true),
            quiet: Some(true),
            full_index: Some(true),
            color: Some("always".to_string()),
            ..Default::default()
        };

        apply_config(&mut args, &config, &matches);

        assert!(args.hidden);
        assert!(args.follow);
        assert!(args.no_ignore);
        assert!(args.quiet);
        assert!(args.full_index);
        assert!(matches!(args.color, vecgrep::cli::ColorChoice::Always));
    }

    #[test]
    fn test_apply_config_does_not_override_explicit_cli_color() {
        let (mut args, matches) = parse_args(&["vecgrep", "--color", "never", "needle"]);
        let config = vecgrep::config::Config {
            color: Some("always".to_string()),
            ..Default::default()
        };

        apply_config(&mut args, &config, &matches);

        assert!(matches!(args.color, vecgrep::cli::ColorChoice::Never));
    }

    #[test]
    fn test_handle_pre_execution_actions_returns_after_stats_without_query() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let path_plan = PathPlan {
            project_root: dir.path().to_path_buf(),
            project_root_canon: dir.path().canonicalize().unwrap(),
            cwd_suffix: PathBuf::new(),
            inside_paths: vec![".".to_string()],
            outside_paths: Vec::new(),
            stale_removal_scope: StaleRemovalScope::All,
        };
        let (args, _) = parse_args(&["vecgrep", "--stats"]);

        let outcome = handle_pre_execution_actions(&args, &path_plan, true).unwrap();

        assert!(matches!(outcome, FlowControl::Return(true)));
    }

    #[test]
    fn test_handle_post_index_actions_returns_after_index_only() {
        let index = Index::open_in_memory().unwrap();
        let (args, _) = parse_args(&["vecgrep", "--index-only"]);

        let outcome = handle_post_index_actions(&args, &index).unwrap();

        assert!(matches!(outcome, FlowControl::Return(true)));
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
            &StaleRemovalScope::All,
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
        let (args, matches) = parse_args(&["vecgrep", "needle"]);
        let invocation = resolve_invocation(args, &matches, &cwd, &cwd).unwrap();

        assert_eq!(invocation.runtime.top_k, 42);
        assert_eq!(invocation.runtime.threshold, 0.15);
        assert!(invocation.runtime.quiet);
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
        let (args, matches) =
            parse_args(&["vecgrep", "--top-k", "7", "--threshold", "0.6", "needle"]);
        let invocation = resolve_invocation(args, &matches, &cwd, &cwd).unwrap();

        assert_eq!(invocation.runtime.top_k, 7);
        assert_eq!(invocation.runtime.threshold, 0.6);
        assert!(invocation.runtime.quiet);
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
        let (args, matches) = parse_args(&[
            "vecgrep",
            "--ignore-file",
            "from-cli.ignore",
            "--ignore-file",
            "shared.ignore",
            "needle",
        ]);
        let invocation = resolve_invocation(args, &matches, &cwd, &cwd).unwrap();

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

        let (args, matches) = parse_args(&["vecgrep", "--serve"]);
        let invocation = resolve_invocation(args, &matches, &cwd, &cwd).unwrap();

        assert_eq!(invocation.run_mode, RunMode::Serve);
        assert!(invocation.query.is_empty());
    }
}
