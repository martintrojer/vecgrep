use serde::{Deserialize, Serialize};

/// A chunk of text from a file, with its position info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Path to the source file (relative to search root).
    pub file_path: String,
    /// The text content of this chunk.
    pub text: String,
    /// 1-based start line in the original file.
    pub start_line: usize,
    /// 1-based end line (inclusive) in the original file.
    pub end_line: usize,
}

/// A search result with its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    /// Cosine similarity score (0.0–1.0).
    pub score: f32,
}

/// Index configuration, stored in meta table to detect config changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    pub model_name: String,
    pub embedding_dim: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}
