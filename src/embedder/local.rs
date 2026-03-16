use anyhow::Result;
use ndarray::{ArrayView2, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::{l2_norm, MAX_SEQ_LEN};

const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/model.onnx"));
const TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/models/tokenizer.json"));

/// Built-in ONNX model embedder.
pub struct LocalEmbedder {
    pub(super) session: Session,
    pub(super) tokenizer: Tokenizer,
}

impl LocalEmbedder {
    pub fn new() -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_intra_threads(num_cpus::get())
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {}", e))?;

        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        tokenizer.with_padding(None);

        Ok(Self { session, tokenizer })
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
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

            let norm = l2_norm(mean_pooled.as_slice().unwrap());
            let normalized = mean_pooled / norm.max(1e-9);

            results.push(normalized.to_vec());
        }

        Ok(results)
    }
}
