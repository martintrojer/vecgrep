use clap::{Parser, ValueEnum};

#[derive(Clone, Debug, ValueEnum)]
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
    #[arg(required_unless_present_any = ["reindex", "stats", "clear_cache", "index_only", "type_list", "interactive", "serve", "show_root"])]
    pub query: Option<String>,

    /// Paths to search (files or directories). Defaults to current directory.
    #[arg(default_values_t = vec![".".to_string()])]
    pub paths: Vec<String>,

    /// Number of top results to return.
    #[arg(short = 'k', long = "top-k", default_value_t = 10)]
    pub top_k: usize,

    /// Minimum similarity threshold (0.0–1.0).
    #[arg(long, default_value_t = 0.3)]
    pub threshold: f32,

    /// Interactive TUI mode.
    #[arg(short = 'i', long)]
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

    /// Context lines around match.
    #[arg(short = 'C', long, default_value_t = 3)]
    pub context: usize,

    /// Force full re-index.
    #[arg(long)]
    pub reindex: bool,

    /// Wait for indexing to complete before searching.
    /// By default, vecgrep searches immediately with the cached index
    /// and indexes changed files in the background.
    #[arg(long)]
    pub full_index: bool,

    /// Build index without searching.
    #[arg(long)]
    pub index_only: bool,

    /// Show index statistics.
    #[arg(long)]
    pub stats: bool,

    /// Delete cached index.
    #[arg(long)]
    pub clear_cache: bool,

    /// Output results as JSONL.
    #[arg(long)]
    pub json: bool,

    /// Tokens per chunk.
    #[arg(long, default_value_t = 500)]
    pub chunk_size: usize,

    /// Overlap tokens between chunks.
    #[arg(long, default_value_t = 100)]
    pub chunk_overlap: usize,

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

    /// Don't respect ignore files (.gitignore, .ignore, etc.).
    #[arg(long)]
    pub no_ignore: bool,

    /// Limit directory traversal depth.
    #[arg(short = 'd', long)]
    pub max_depth: Option<usize>,

    /// Show all supported file types.
    #[arg(long)]
    pub type_list: bool,

    /// When to use colored output.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// Warn and ask for confirmation when more than this many files need indexing.
    /// Set to 0 to disable.
    #[arg(long, default_value_t = 1000)]
    pub index_warn_threshold: usize,

    /// Print the resolved project root and exit.
    #[arg(long)]
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
    #[arg(long)]
    pub serve: bool,

    /// Port for HTTP server (default: auto-pick free port).
    #[arg(long, requires = "serve")]
    pub port: Option<u16>,
}
