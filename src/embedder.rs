use anyhow::Result;
use ndarray::{Array1, ArrayView2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/model.onnx"));
const TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/tokenizer.json"));

/// Maximum token sequence length for all-MiniLM-L6-v2.
const MAX_SEQ_LEN: usize = 256;

/// Embedding dimension for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder {
    pub fn new() -> Result<Self> {
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

        Ok(Self { session, tokenizer })
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

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();

        // Find max length and cap at MAX_SEQ_LEN
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_SEQ_LEN))
            .max()
            .unwrap_or(0);

        // Build padded input tensors
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

        // Create ort Tensor values
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

        // Extract output: try_extract_tensor returns (&Shape, &[f32])
        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract tensor: {}", e))?;

        // Shape should be (batch_size, seq_len, hidden_size)
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let hidden_size = dims[2];

        // Reshape flat data into 3D using ndarray
        let token_embeddings =
            ArrayView2::from_shape((batch_size * dims[1], hidden_size), out_data)?;

        // Build attention mask as f32 for mean pooling
        let attention_f32: Vec<f32> = attention_mask.iter().map(|&v| v as f32).collect();
        let attention_mask_2d = ArrayView2::from_shape((batch_size, max_len), &attention_f32)?;

        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Get this sample's token embeddings: rows [i*max_len .. (i+1)*max_len]
            let start = i * max_len;
            let end = start + max_len;
            let tokens = token_embeddings.slice(ndarray::s![start..end, ..]);
            let mask = attention_mask_2d.row(i);

            // Expand mask to (seq_len, hidden)
            let mask_expanded = mask
                .to_owned()
                .insert_axis(Axis(1))
                .broadcast((max_len, hidden_size))
                .unwrap()
                .to_owned();

            // Masked sum
            let masked = &tokens * &mask_expanded;
            let summed = masked.sum_axis(Axis(0)); // (hidden,)
            let mask_sum = mask.sum().max(1e-9);
            let mean_pooled = summed / mask_sum;

            // L2 normalize
            let norm = l2_norm(&mean_pooled);
            let normalized = mean_pooled / norm.max(1e-9);

            results.push(normalized.to_vec());
        }

        Ok(results)
    }
}

fn l2_norm(v: &Array1<f32>) -> f32 {
    v.mapv(|x| x * x).sum().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_single() {
        let mut embedder = Embedder::new().unwrap();
        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // Check L2 normalization (should be ~1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_embed_batch() {
        let mut embedder = Embedder::new().unwrap();
        let embeddings = embedder
            .embed_batch(&["hello world", "semantic search"])
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), EMBEDDING_DIM);
        assert_eq!(embeddings[1].len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_semantic_similarity() {
        let mut embedder = Embedder::new().unwrap();
        let e1 = embedder.embed("error handling in network code").unwrap();
        let e2 = embedder
            .embed("dealing with exceptions in networking")
            .unwrap();
        let e3 = embedder.embed("chocolate cake recipe").unwrap();

        let sim_related: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        let sim_unrelated: f32 = e1.iter().zip(e3.iter()).map(|(a, b)| a * b).sum();

        // Related texts should have higher similarity
        assert!(sim_related > sim_unrelated);
    }
}
