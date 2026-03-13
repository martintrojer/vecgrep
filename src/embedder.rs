use anyhow::Result;
use ndarray::{Array1, ArrayView2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/model.onnx"));
const TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/tokenizer.json"));

/// Maximum token sequence length for all-MiniLM-L6-v2.
const MAX_SEQ_LEN: usize = 256;

/// Embedding dimension for all-MiniLM-L6-v2 (built-in model).
pub const EMBEDDING_DIM: usize = 384;

/// Built-in ONNX model embedder.
pub struct LocalEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

/// Remote embedder using an OpenAI-compatible API.
pub struct RemoteEmbedder {
    url: String,
    model: String,
    embedding_dim: Option<usize>,
    agent: ureq::Agent,
}

/// Embedding backend — either built-in ONNX model or remote API.
pub enum Embedder {
    Local(Box<LocalEmbedder>),
    Remote(RemoteEmbedder),
}

impl Embedder {
    /// Create an embedder using the built-in ONNX model.
    pub fn new_local() -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_intra_threads(num_cpus::get())
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))?;

        let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Embedder::Local(Box::new(LocalEmbedder {
            session,
            tokenizer,
        })))
    }

    /// Create an embedder using a remote OpenAI-compatible API.
    pub fn new_remote(url: &str, model: &str) -> Self {
        Embedder::Remote(RemoteEmbedder {
            url: url.to_string(),
            model: model.to_string(),
            embedding_dim: None,
            agent: ureq::Agent::new_with_config(
                ureq::config::Config::builder()
                    .timeout_global(Some(std::time::Duration::from_secs(120)))
                    .build(),
            ),
        })
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

    /// Whether this is a remote embedder.
    pub fn is_remote(&self) -> bool {
        matches!(self, Embedder::Remote(_))
    }
}

impl LocalEmbedder {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);

        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let seq_len = ids.len().min(max_len);

            for j in 0..seq_len {
                input_ids[i * max_len + j] = ids[j] as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
                token_type_ids[i * max_len + j] = type_ids[j] as i64;
            }
        }

        let shape = vec![batch_size as i64, max_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;
        let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {}", e))?;
        let token_type_ids_tensor = Tensor::from_array((shape, token_type_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create token_type_ids tensor: {}", e))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract tensor: {}", e))?;

        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let hidden_size = dims[2];

        let token_embeddings =
            ArrayView2::from_shape((batch_size * dims[1], hidden_size), out_data)?;

        let attention_f32: Vec<f32> = attention_mask.iter().map(|&v| v as f32).collect();
        let attention_mask_2d = ArrayView2::from_shape((batch_size, max_len), &attention_f32)?;

        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = i * max_len;
            let end = start + max_len;
            let tokens = token_embeddings.slice(ndarray::s![start..end, ..]);
            let mask = attention_mask_2d.row(i);

            let mask_expanded = mask
                .to_owned()
                .insert_axis(Axis(1))
                .broadcast((max_len, hidden_size))
                .unwrap()
                .to_owned();

            let masked = &tokens * &mask_expanded;
            let summed = masked.sum_axis(Axis(0));
            let mask_sum = mask.sum().max(1e-9);
            let mean_pooled = summed / mask_sum;

            let norm = l2_norm(&mean_pooled);
            let normalized = mean_pooled / norm.max(1e-9);

            results.push(normalized.to_vec());
        }

        Ok(results)
    }
}

impl RemoteEmbedder {
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::new();

        // Sub-batch to avoid overwhelming the server with large payloads
        for batch in texts.chunks(16) {
            let input: Vec<&str> = batch.to_vec();
            let body = serde_json::json!({
                "model": self.model,
                "input": input,
            });

            let body_str = serde_json::to_string(&body).unwrap();
            let resp = self
                .agent
                .post(&self.url)
                .content_type("application/json")
                .send(body_str.as_bytes())
                .map_err(|e| anyhow::anyhow!("Embeddings API request failed: {e}"))?;
            let response_text = resp
                .into_body()
                .read_to_string()
                .map_err(|e| anyhow::anyhow!("Failed to read embeddings response: {e}"))?;
            let response: serde_json::Value = serde_json::from_str(&response_text)
                .map_err(|e| anyhow::anyhow!("Failed to parse embeddings response: {e}"))?;

            let data = response["data"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Response missing 'data' array"))?;

            let mut batch_embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
            for item in data {
                let index = item["index"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'index' in response"))?
                    as usize;
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Missing 'embedding' in response"))?
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                // Discover embedding dimension from first response
                if self.embedding_dim.is_none() {
                    self.embedding_dim = Some(embedding.len());
                    tracing::info!("Remote embedder dimension discovered: {}", embedding.len());
                }

                // L2 normalize
                let norm = l2_norm_vec(&embedding);
                let normalized = if norm > 1e-9 {
                    embedding.iter().map(|x| x / norm).collect()
                } else {
                    embedding
                };

                batch_embeddings.push((index, normalized));
            }

            // Sort by index to maintain input order
            batch_embeddings.sort_by_key(|(idx, _)| *idx);
            all_embeddings.extend(batch_embeddings.into_iter().map(|(_, emb)| emb));
        }

        Ok(all_embeddings)
    }
}

fn l2_norm(v: &Array1<f32>) -> f32 {
    v.mapv(|x| x * x).sum().sqrt()
}

fn l2_norm_vec(v: &[f32]) -> f32 {
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

        assert!(sim_related > sim_unrelated);
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

        let remote = Embedder::new_remote("http://localhost:11434/v1/embeddings", "test");
        assert!(remote.tokenizer().is_none());
    }
}
