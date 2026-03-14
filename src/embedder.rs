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
    max_chars: usize,
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
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(120)))
                .build(),
        );

        // Try to discover context length via Ollama's /api/show (best-effort)
        let max_chars =
            query_context_length(&agent, url, model).unwrap_or(DEFAULT_REMOTE_MAX_CHARS);

        Embedder::Remote(RemoteEmbedder {
            url: url.to_string(),
            model: model.to_string(),
            embedding_dim: None,
            max_chars,
            agent,
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

    /// Max context in estimated tokens for the remote model.
    /// Returns None for the local model (it handles its own context).
    pub fn context_tokens(&self) -> Option<usize> {
        match self {
            Embedder::Local(_) => None,
            Embedder::Remote(r) => Some(r.max_chars * 2 / 5), // reverse the 2.5 chars/token
        }
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

/// Default max chars per text for remote embedders.
/// Assumes 512-token context at ~2.5 chars/token (URLs, markdown, code
/// tokenize more densely than plain English).
const DEFAULT_REMOTE_MAX_CHARS: usize = 1200;

fn truncate_text(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        text
    } else {
        &text[..text.floor_char_boundary(max_chars)]
    }
}

/// Try to get model context length via Ollama's /api/show endpoint.
/// Returns max chars (context_length × 4) or None if not Ollama.
fn query_context_length(agent: &ureq::Agent, embedder_url: &str, model: &str) -> Option<usize> {
    // Derive Ollama base URL: http://host:port/v1/embeddings → http://host:port/api/show
    let base = embedder_url.split("/v1/").next()?;
    let show_url = format!("{}/api/show", base);

    let body = serde_json::json!({"model": model});
    let resp = agent
        .post(&show_url)
        .content_type("application/json")
        .send(serde_json::to_string(&body).unwrap().as_bytes())
        .ok()?;
    let text = resp.into_body().read_to_string().ok()?;
    let info: serde_json::Value = serde_json::from_str(&text).ok()?;

    let model_info = &info["model_info"];
    let context_tokens = model_info["bert.context_length"]
        .as_u64()
        .or_else(|| model_info["general.context_length"].as_u64())?
        as usize;

    // ~2.5 chars/token — URLs, markdown, and code tokenize densely.
    // Ollama rejects texts exceeding context length with HTTP 400.
    let max_chars = context_tokens * 5 / 2;
    tracing::info!(
        "Remote model context: {} tokens, truncating at {} chars",
        context_tokens,
        max_chars
    );
    Some(max_chars)
}

impl RemoteEmbedder {
    fn send_request(&self, texts: &[&str]) -> Result<serde_json::Value> {
        let truncated: Vec<&str> = texts
            .iter()
            .map(|t| truncate_text(t, self.max_chars))
            .collect();
        let body = serde_json::json!({
            "model": self.model,
            "input": truncated,
        });
        let body_str = serde_json::to_string(&body).unwrap();
        tracing::debug!(
            "Remote embed: {} texts, payload {} bytes, longest {} chars",
            truncated.len(),
            body_str.len(),
            truncated.iter().map(|t| t.len()).max().unwrap_or(0)
        );
        let resp = self
            .agent
            .post(&self.url)
            .content_type("application/json")
            .send(body_str.as_bytes());
        match resp {
            Ok(resp) => {
                let response_text = resp
                    .into_body()
                    .read_to_string()
                    .map_err(|e| anyhow::anyhow!("Failed to read response: {e}"))?;
                serde_json::from_str(&response_text)
                    .map_err(|e| anyhow::anyhow!("Failed to parse response: {e}"))
            }
            Err(e) => {
                tracing::debug!(
                    "Remote embed failed: {}, text lengths: {:?}, first 100 chars: {:?}",
                    e,
                    truncated.iter().map(|t| t.len()).collect::<Vec<_>>(),
                    truncated
                        .iter()
                        .map(|t| &t[..t.len().min(100)])
                        .collect::<Vec<_>>()
                );
                Err(anyhow::anyhow!("Embeddings API request failed: {e}"))
            }
        }
    }

    fn parse_embeddings(&mut self, response: &serde_json::Value) -> Result<Vec<Vec<f32>>> {
        let data = response["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Response missing 'data' array"))?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = Vec::new();
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

            embeddings.push((index, normalized));
        }

        embeddings.sort_by_key(|(idx, _)| *idx);
        Ok(embeddings.into_iter().map(|(_, emb)| emb).collect())
    }

    /// Split texts into batches based on total payload size.
    /// Each batch stays under max_chars total to avoid HTTP 400 errors.
    fn make_batches<'a>(&self, texts: &[&'a str]) -> Vec<Vec<&'a str>> {
        let mut batches = Vec::new();
        let mut current_batch: Vec<&str> = Vec::new();
        let mut current_size = 0;

        for text in texts {
            let truncated_len = text.len().min(self.max_chars);
            // If adding this text would exceed the limit, flush the batch
            if !current_batch.is_empty() && current_size + truncated_len > self.max_chars * 2 {
                batches.push(std::mem::take(&mut current_batch));
                current_size = 0;
            }
            current_batch.push(text);
            current_size += truncated_len;
        }
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }
        batches
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::new();

        for batch in self.make_batches(texts) {
            match self.send_request(&batch) {
                Ok(response) => {
                    all_embeddings.extend(self.parse_embeddings(&response)?);
                }
                Err(batch_err) => {
                    tracing::debug!(
                        "Batch of {} failed ({}), falling back to single. Lengths: {:?}",
                        batch.len(),
                        batch_err,
                        batch.iter().map(|t| t.len()).collect::<Vec<_>>()
                    );
                    for text in &batch {
                        match self.send_request(&[*text]) {
                            Ok(response) => {
                                all_embeddings.extend(self.parse_embeddings(&response)?);
                            }
                            Err(e) => {
                                let preview = &text[..text.len().min(80)];
                                tracing::warn!(
                                    "Skipping chunk ({} chars, starts with {:?}): {}",
                                    text.len(),
                                    preview,
                                    e
                                );
                                // Use zero vector so index positions stay aligned
                                let dim = self.embedding_dim.unwrap_or(384);
                                all_embeddings.push(vec![0.0; dim]);
                            }
                        }
                    }
                }
            }
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

        let remote = Embedder::new_remote("http://localhost:99999/v1/embeddings", "test");
        assert!(remote.tokenizer().is_none());
    }

    // --- truncate_text tests ---

    #[test]
    fn test_truncate_short_text() {
        assert_eq!(truncate_text("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_exact_limit() {
        assert_eq!(truncate_text("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_over_limit() {
        let result = truncate_text("hello world", 5);
        assert_eq!(result, "hello");
        assert!(result.len() <= 5);
    }

    #[test]
    fn test_truncate_unicode_boundary() {
        // "café" — the é is 2 bytes in UTF-8
        let text = "café";
        let result = truncate_text(text, 4);
        // Should not split in the middle of the é
        assert!(result.len() <= 4);
        assert!(result.is_char_boundary(result.len()));
    }

    // --- make_batches tests ---

    fn make_test_remote(max_chars: usize) -> RemoteEmbedder {
        RemoteEmbedder {
            url: String::new(),
            model: String::new(),
            embedding_dim: None,
            max_chars,
            agent: ureq::Agent::new_with_config(ureq::config::Config::default()),
        }
    }

    #[test]
    fn test_batches_single_small_text() {
        let remote = make_test_remote(100);
        let texts = vec!["hello"];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_batches_multiple_small_texts() {
        let remote = make_test_remote(100);
        // 4 texts × 5 chars = 20 chars total, well under 200 (max_chars * 2)
        let texts = vec!["hello", "world", "foo", "bar"];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn test_batches_split_on_payload_size() {
        let remote = make_test_remote(50);
        // max payload = 50 * 2 = 100 chars
        // Each text is 40 chars → 2 fit per batch, 3rd starts a new one
        let long = "a".repeat(40);
        let texts: Vec<&str> = vec![long.as_str(), long.as_str(), long.as_str()];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_batches_single_huge_text() {
        let remote = make_test_remote(50);
        // One text that exceeds max_chars — should still go in a batch (truncated at send)
        let huge = "x".repeat(200);
        let texts = vec![huge.as_str()];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_batches_mixed_sizes() {
        let remote = make_test_remote(100);
        // max payload = 200 chars
        let short = "hi"; // 2 chars
        let medium = "a".repeat(80); // 80 chars
        let long = "b".repeat(150); // 150 chars, truncated to 100
        let texts: Vec<&str> = vec![short, medium.as_str(), long.as_str(), short];

        let batches = remote.make_batches(&texts);
        // "hi"(2) + medium(80) = 82, under 200 → same batch
        // + long(100 truncated) = 182, under 200 → same batch
        // + "hi"(2) = 184, under 200 → same batch
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_batches_empty_input() {
        let remote = make_test_remote(100);
        let texts: Vec<&str> = vec![];
        let batches = remote.make_batches(&texts);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batches_preserves_order() {
        let remote = make_test_remote(20);
        // max payload = 40 chars, each text ~15 chars → 2 per batch
        let texts = vec!["aaaa-1111-bbbbb", "cccc-2222-ddddd", "eeee-3333-fffff"];
        let batches = remote.make_batches(&texts);

        // Flatten and verify order preserved
        let flat: Vec<&str> = batches.into_iter().flatten().collect();
        assert_eq!(flat, texts);
    }

    #[test]
    fn test_default_remote_max_chars() {
        assert_eq!(DEFAULT_REMOTE_MAX_CHARS, 1200);
    }

    #[test]
    fn test_context_tokens_local() {
        let local = Embedder::new_local().unwrap();
        assert!(local.context_tokens().is_none());
    }

    #[test]
    fn test_context_tokens_remote_default() {
        let remote = Embedder::new_remote("http://localhost:99999/v1/embeddings", "test");
        let ctx = remote.context_tokens().unwrap();
        // DEFAULT_REMOTE_MAX_CHARS=1200, reverse: 1200 * 2 / 5 = 480
        assert_eq!(ctx, 480);
    }

    #[test]
    fn test_context_tokens_with_known_limit() {
        // Simulate Ollama with 512-token context → max_chars = 512 * 5/2 = 1280
        let remote = RemoteEmbedder {
            url: String::new(),
            model: String::new(),
            embedding_dim: None,
            max_chars: 1280,
            agent: ureq::Agent::new_with_config(ureq::config::Config::default()),
        };
        let embedder = Embedder::Remote(remote);
        // 1280 * 2 / 5 = 512
        assert_eq!(embedder.context_tokens(), Some(512));
    }
}
