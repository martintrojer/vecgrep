use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "vecgrep",
    about = "Semantic grep — like ripgrep, but with vector search",
    version
)]
pub struct Args {
    /// The search query (natural language or code snippet).
    #[arg(required_unless_present_any = ["reindex", "stats", "clear_cache", "index_only"])]
    pub query: Option<String>,

    /// Paths to search (files or directories). Defaults to current directory.
    #[arg(default_values_t = vec![".".to_string()])]
    pub paths: Vec<String>,

    /// Number of top results to return.
    #[arg(short = 'k', long = "top-k", default_value_t = 10)]
    pub top_k: usize,

    /// Minimum similarity threshold (0.0–1.0).
    #[arg(short = 't', long, default_value_t = 0.3)]
    pub threshold: f32,

    /// Interactive TUI mode.
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Filter by file type (e.g., rust, python, js).
    #[arg(short = 'T', long = "type")]
    pub file_type: Option<Vec<String>>,

    /// Filter by glob pattern.
    #[arg(short = 'g', long)]
    pub glob: Option<Vec<String>>,

    /// Context lines around match.
    #[arg(short = 'C', long, default_value_t = 3)]
    pub context: usize,

    /// Number of threads for indexing.
    #[arg(short = 'j', long)]
    pub threads: Option<usize>,

    /// Force full re-index.
    #[arg(long)]
    pub reindex: bool,

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
}
