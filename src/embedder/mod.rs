mod local;
mod remote;

use anyhow::Result;
use tokenizers::Tokenizer;

pub use local::LocalEmbedder;
pub use remote::RemoteEmbedder;

/// Maximum token sequence length for all-MiniLM-L6-v2.
const MAX_SEQ_LEN: usize = 256;

/// Embedding dimension for all-MiniLM-L6-v2 (built-in model).
pub const EMBEDDING_DIM: usize = 384;

/// Embedding backend — either built-in ONNX model or remote API.
pub enum Embedder {
    Local(Box<LocalEmbedder>),
    Remote(RemoteEmbedder),
}

impl Embedder {
    /// Create an embedder using the built-in ONNX model.
    pub fn new_local() -> Result<Self> {
        Ok(Embedder::Local(Box::new(LocalEmbedder::new()?)))
    }

    /// Create an embedder using a remote OpenAI-compatible API.
    pub fn new_remote(url: &str, model: &str) -> Self {
        Embedder::Remote(RemoteEmbedder::new(url, model))
    }

    /// Embed a single text string, returning a normalized embedding vector.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed a batch of texts, returning normalized embedding vectors.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        match self {
            Embedder::Local(local) => local.embed_batch(texts),
            Embedder::Remote(remote) => remote.embed_batch(texts),
        }
    }

    /// Get the embedding dimension. For remote embedders, this is discovered on first call.
    pub fn embedding_dim(&self) -> usize {
        match self {
            Embedder::Local(_) => EMBEDDING_DIM,
            Embedder::Remote(remote) => remote.embedding_dim.unwrap_or(0),
        }
    }

    /// Get a reference to the tokenizer (only available for built-in model).
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        match self {
            Embedder::Local(local) => Some(&local.tokenizer),
            Embedder::Remote(_) => None,
        }
    }

    /// Model name for index configuration.
    pub fn model_name(&self) -> &str {
        match self {
            Embedder::Local(_) => "all-MiniLM-L6-v2",
            Embedder::Remote(remote) => &remote.model,
        }
    }

    /// Max context in tokens for this embedder, if known.
    pub fn context_tokens(&self) -> Option<usize> {
        match self {
            Embedder::Local(_) => Some(MAX_SEQ_LEN),
            Embedder::Remote(r) => Some(r.max_chars * 2 / 5),
        }
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_single() {
        let mut embedder = Embedder::new_local().unwrap();
        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_embed_batch() {
        let mut embedder = Embedder::new_local().unwrap();
        let embeddings = embedder
            .embed_batch(&["hello world", "semantic search"])
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIM);
        assert_eq!(embeddings[1].len(), EMBEDDING_DIM);
        assert_ne!(embeddings[0], embeddings[1]);
    }

    #[test]
    fn test_semantic_similarity() {
        let mut embedder = Embedder::new_local().unwrap();
        let e1 = embedder.embed("error handling in network code").unwrap();
        let e2 = embedder
            .embed("dealing with exceptions in networking")
            .unwrap();
        let e3 = embedder.embed("chocolate cake recipe").unwrap();

        let sim_related: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let sim_unrelated: f32 = e1.iter().zip(e3.iter()).map(|(a, b)| a * b).sum();

        assert!(
            sim_related - sim_unrelated > 0.1,
            "semantic gap too small: related={sim_related:.3}, unrelated={sim_unrelated:.3}"
        );
    }

    #[test]
    fn test_model_name() {
        let local = Embedder::new_local().unwrap();
        assert_eq!(local.model_name(), "all-MiniLM-L6-v2");

        let remote =
            Embedder::new_remote("http://localhost:11434/v1/embeddings", "nomic-embed-text");
        assert_eq!(remote.model_name(), "nomic-embed-text");
    }

    #[test]
    fn test_tokenizer_availability() {
        let local = Embedder::new_local().unwrap();
        assert!(local.tokenizer().is_some());

        let remote = Embedder::new_remote("http://localhost:99999/v1/embeddings", "test");
        assert!(remote.tokenizer().is_none());
    }

    #[test]
    fn test_context_tokens_local() {
        let local = Embedder::new_local().unwrap();
        assert_eq!(local.context_tokens(), Some(MAX_SEQ_LEN));
    }

    #[test]
    fn test_context_tokens_remote_default() {
        let remote = Embedder::new_remote("http://localhost:99999/v1/embeddings", "test");
        let ctx = remote.context_tokens().unwrap();
        assert_eq!(ctx, 480);
    }

    #[test]
    fn test_context_tokens_with_known_limit() {
        let remote = RemoteEmbedder {
            url: String::new(),
            model: String::new(),
            embedding_dim: None,
            max_chars: 1280,
            agent: ureq::Agent::new_with_config(ureq::config::Config::default()),
        };
        let embedder = Embedder::Remote(remote);
        assert_eq!(embedder.context_tokens(), Some(512));
    }
}
