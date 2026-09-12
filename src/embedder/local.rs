use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::{l2_normalize, ort_err, Embedding, MAX_SEQ_LEN};

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
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(num_cpus::get())
            .map_err(ort_err)?
            .commit_from_memory(MODEL_BYTES)
            .map_err(|e| ort_err(format_args!("Failed to load ONNX model: {e}")))?;

        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        tokenizer.with_padding(None);

        Ok(Self { session, tokenizer })
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
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

        let shape = [batch_size as i64, max_len as i64];
        let input_ids_tensor = Tensor::from_array((shape, input_ids)).map_err(ort_err)?;
        let attention_mask_tensor =
            Tensor::from_array((shape, attention_mask.clone())).map_err(ort_err)?;
        let token_type_ids_tensor = Tensor::from_array((shape, token_type_ids)).map_err(ort_err)?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = self.session.run(inputs).map_err(ort_err)?;

        let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;

        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let hidden_size = dims[2];
        // Model seq length should match the padded input length we sent.
        debug_assert_eq!(dims[1], max_len);

        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut mask_sum = 0.0f32;
            for j in 0..max_len {
                let m = attention_mask[i * max_len + j] as f32;
                if m == 0.0 {
                    continue;
                }
                mask_sum += m;
                let token_offset = (i * max_len + j) * hidden_size;
                for (h, cell) in pooled.iter_mut().enumerate() {
                    *cell += out_data[token_offset + h] * m;
                }
            }
            let denom = mask_sum.max(1e-9);
            for cell in &mut pooled {
                *cell /= denom;
            }
            l2_normalize(&mut pooled);
            results.push(Embedding {
                values: pooled,
                failed: false,
            });
        }

        Ok(results)
    }
}
