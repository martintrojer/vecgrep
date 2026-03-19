use clap::{Parser, ValueEnum};

#[derive(Clone, Debug, PartialEq, ValueEnum)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

#[derive(Parser, Debug)]
#[command(
    name = "vecgrep",
    about = "Semantic grep — like ripgrep, but with vector search",
    version
)]
pub struct Args {
    /// The search query (natural language or code snippet).
    #[arg(required_unless_present_any = ["reindex", "stats", "clear_cache", "index_only", "type_list", "interactive", "serve", "show_root", "query_flag"])]
    pub query: Option<String>,

    /// Paths to search (files or directories). Defaults to current directory.
    #[arg(default_values_t = vec![".".to_string()])]
    pub paths: Vec<String>,

    /// Explicit query for interactive/serve mode. When set, all positional
    /// arguments are treated as paths. Requires -i or --serve.
    #[arg(long = "query")]
    pub query_flag: Option<String>,

    /// Number of top results to return.
    #[arg(short = 'k', long = "top-k")]
    pub top_k: Option<usize>,

    /// Minimum similarity threshold (0.0–1.0).
    #[arg(long)]
    pub threshold: Option<f32>,

    /// Interactive TUI mode.
    #[arg(short = 'i', long, conflicts_with_all = ["serve", "index_only", "type_list", "show_root"])]
    pub interactive: bool,

    /// Filter by file type (e.g., rust, python, js). Can be specified multiple times.
    #[arg(short = 't', long = "type")]
    pub file_type: Option<Vec<String>>,

    /// Negative file type filter. Can be specified multiple times.
    #[arg(short = 'T', long = "type-not")]
    pub file_type_not: Option<Vec<String>>,

    /// Filter by glob pattern. Can be specified multiple times.
    #[arg(short = 'g', long)]
    pub glob: Option<Vec<String>>,

    /// Force full re-index.
    #[arg(long)]
    pub reindex: bool,

    /// Wait for indexing to complete before starting interactive or server mode.
    /// Normal CLI searches already wait for indexing to finish.
    #[arg(long)]
    pub full_index: bool,

    /// Build index without searching.
    #[arg(long, conflicts_with_all = ["interactive", "serve", "type_list", "show_root"])]
    pub index_only: bool,

    /// Show index statistics.
    #[arg(long, conflicts_with_all = ["interactive", "serve", "index_only", "type_list", "show_root"])]
    pub stats: bool,

    /// Delete cached index.
    #[arg(long)]
    pub clear_cache: bool,

    /// Output results as JSONL.
    #[arg(long)]
    pub json: bool,

    /// Tokens per chunk.
    #[arg(long)]
    pub chunk_size: Option<usize>,

    /// Overlap tokens between chunks.
    #[arg(long)]
    pub chunk_overlap: Option<usize>,

    // --- rg-compatible flags ---
    /// Suppress all status messages on stderr.
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Search hidden files and directories.
    #[arg(short = '.', long)]
    pub hidden: bool,

    /// Follow symbolic links.
    #[arg(short = 'L', long)]
    pub follow: bool,

    /// Print only the paths of files with matches (no content).
    #[arg(short = 'l', long)]
    pub files_with_matches: bool,

    /// Print a count of matching chunks per file.
    #[arg(short = 'c', long)]
    pub count: bool,

    /// Additional ignore file (gitignore syntax). Can be specified multiple times.
    #[arg(long)]
    pub ignore_file: Option<Vec<String>>,

    /// Don't respect ignore files (.gitignore, .ignore, etc.).
    #[arg(long)]
    pub no_ignore: bool,

    /// Limit directory traversal depth.
    #[arg(short = 'd', long)]
    pub max_depth: Option<usize>,

    /// Show all supported file types.
    #[arg(long, conflicts_with_all = ["interactive", "serve", "index_only", "show_root"])]
    pub type_list: bool,

    /// When to use colored output.
    #[arg(long, value_enum)]
    pub color: Option<ColorChoice>,

    /// Alias for --color=always (force colors when piping).
    #[arg(short = 'p', long)]
    pub pretty: bool,

    /// Warn and ask for confirmation when more than this many files need indexing.
    /// Set to 0 to disable.
    #[arg(long)]
    pub index_warn_threshold: Option<usize>,

    /// Print the resolved project root and exit.
    #[arg(long, conflicts_with_all = ["interactive", "serve", "index_only", "type_list"])]
    pub show_root: bool,

    /// URL of an OpenAI-compatible embeddings API
    /// (e.g., http://localhost:11434/v1/embeddings).
    /// When set, uses the external server instead of the built-in model.
    #[arg(long, requires = "embedder_model")]
    pub embedder_url: Option<String>,

    /// Model name to use with --embedder-url.
    #[arg(long, requires = "embedder_url")]
    pub embedder_model: Option<String>,

    /// Start HTTP server mode.
    #[arg(long, conflicts_with_all = ["interactive", "index_only", "type_list", "show_root"])]
    pub serve: bool,

    /// Port for HTTP server (default: auto-pick free port).
    #[arg(long, requires = "serve")]
    pub port: Option<u16>,

    /// Skip paths outside the selected project root instead of failing.
    #[arg(long)]
    pub skip_outside_root: bool,

    /// Search the entire project index, ignoring path scoping from cwd or arguments.
    #[arg(long)]
    pub no_scope: bool,

    /// Command to open a file from TUI results. Placeholders: {file}, {line}, {end_line}.
    /// Examples: "nvim +{line} {file}", "bat -n --highlight-line {line}:{end_line} {file}"
    /// Default: $PAGER or less.
    #[arg(long)]
    pub open_cmd: Option<String>,
}

/// Hardcoded defaults for config-overridable fields.
/// These match the spec's `config {}` block defaults.
pub const DEFAULT_TOP_K: usize = 10;
pub const DEFAULT_THRESHOLD: f32 = 0.2;
pub const DEFAULT_CHUNK_SIZE: usize = 256;
pub const DEFAULT_CHUNK_OVERLAP: usize = 64;
pub const DEFAULT_INDEX_WARN_THRESHOLD: usize = 1000;

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn rejects_interactive_and_serve_together() {
        let err = Args::try_parse_from(["vecgrep", "--interactive", "--serve"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn rejects_index_only_and_show_root_together() {
        let err = Args::try_parse_from(["vecgrep", "--index-only", "--show-root"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn configurable_fields_default_to_none() {
        let args = Args::parse_from(["vecgrep", "needle"]);
        assert_eq!(args.chunk_size, None);
        assert_eq!(args.chunk_overlap, None);
        assert_eq!(args.top_k, None);
        assert_eq!(args.threshold, None);
        assert_eq!(args.color, None);
        assert_eq!(args.index_warn_threshold, None);
    }

    #[test]
    fn explicit_cli_values_are_some() {
        let args = Args::parse_from(["vecgrep", "--top-k", "5", "--chunk-size", "128", "needle"]);
        assert_eq!(args.top_k, Some(5));
        assert_eq!(args.chunk_size, Some(128));
    }
}
