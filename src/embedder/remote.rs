use anyhow::Result;

use super::l2_norm;

/// Default max chars per text for remote embedders.
/// Assumes 512-token context at ~2.5 chars/token (URLs, markdown, code
/// tokenize more densely than plain English).
pub(super) const DEFAULT_REMOTE_MAX_CHARS: usize = 1200;

/// Remote embedder using an OpenAI-compatible API.
pub struct RemoteEmbedder {
    pub(super) url: String,
    pub(super) model: String,
    pub(super) embedding_dim: Option<usize>,
    pub(super) max_chars: usize,
    pub(super) agent: ureq::Agent,
}

impl RemoteEmbedder {
    pub fn new(url: &str, model: &str) -> Self {
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(120)))
                .http_status_as_error(false)
                .build(),
        );

        let max_chars =
            query_context_length(&agent, url, model).unwrap_or(DEFAULT_REMOTE_MAX_CHARS);

        Self {
            url: url.to_string(),
            model: model.to_string(),
            embedding_dim: None,
            max_chars,
            agent,
        }
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::new();

        for batch in self.make_batches(texts) {
            match self.send_request(&batch) {
                Ok(response) => {
                    all_embeddings.extend(self.parse_embeddings(&response, batch.len())?);
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
                                all_embeddings.extend(self.parse_embeddings(&response, 1)?);
                            }
                            Err(e) => {
                                // If we haven't discovered the embedding dimension yet,
                                // we can't create a valid zero vector — propagate the error
                                // so the caller knows the embedder is unreachable.
                                if self.embedding_dim.is_none() {
                                    return Err(e);
                                }
                                let preview = &text[..text.len().min(80)];
                                tracing::warn!(
                                    "Skipping chunk ({} chars, starts with {:?}): {}",
                                    text.len(),
                                    preview,
                                    e
                                );
                                // Use zero vector so index positions stay aligned
                                let dim = self.embedding_dim.unwrap();
                                all_embeddings.push(vec![0.0; dim]);
                            }
                        }
                    }
                }
            }
        }

        Ok(all_embeddings)
    }

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
                let status = resp.status().as_u16();
                let response_text = resp
                    .into_body()
                    .read_to_string()
                    .map_err(|e| anyhow::anyhow!("Failed to read response: {e}"))?;
                if status >= 400 {
                    let detail = extract_error_message(&response_text);
                    tracing::debug!(
                        "Remote embed failed (HTTP {}): {}, text lengths: {:?}",
                        status,
                        detail,
                        truncated.iter().map(|t| t.len()).collect::<Vec<_>>(),
                    );
                    return Err(anyhow::anyhow!(
                        "Embeddings API returned HTTP {}: {}",
                        status,
                        detail
                    ));
                }
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

    pub(super) fn parse_embeddings(
        &mut self,
        response: &serde_json::Value,
        expected_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let data = response["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Response missing 'data' array"))?;

        if data.len() != expected_len {
            anyhow::bail!(
                "Response returned {} embeddings for {} inputs",
                data.len(),
                expected_len
            );
        }

        let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; expected_len];
        for item in data {
            let index = item["index"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("Missing 'index' in response"))?
                as usize;
            if index >= expected_len {
                anyhow::bail!(
                    "Response embedding index {} is out of range for {} inputs",
                    index,
                    expected_len
                );
            }
            if embeddings[index].is_some() {
                anyhow::bail!("Response contained duplicate embedding index {}", index);
            }

            let embedding: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing 'embedding' in response"))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            if embedding.is_empty() {
                anyhow::bail!("Response contained empty embedding at index {}", index);
            }

            if self.embedding_dim.is_none() {
                self.embedding_dim = Some(embedding.len());
                tracing::info!("Remote embedder dimension discovered: {}", embedding.len());
            } else if self.embedding_dim != Some(embedding.len()) {
                anyhow::bail!(
                    "Response embedding dimension {} does not match expected {}",
                    embedding.len(),
                    self.embedding_dim.unwrap_or_default()
                );
            }

            // L2 normalize
            let norm = l2_norm(&embedding);
            let normalized = if norm > 1e-9 {
                embedding.iter().map(|x| x / norm).collect()
            } else {
                embedding
            };

            embeddings[index] = Some(normalized);
        }

        embeddings
            .into_iter()
            .enumerate()
            .map(|(index, emb)| {
                emb.ok_or_else(|| anyhow::anyhow!("Response missing embedding at index {}", index))
            })
            .collect()
    }

    /// Split texts into batches based on total payload size.
    pub(super) fn make_batches<'a>(&self, texts: &[&'a str]) -> Vec<Vec<&'a str>> {
        let mut batches = Vec::new();
        let mut current_batch: Vec<&str> = Vec::new();
        let mut current_size = 0;

        for text in texts {
            let truncated_len = text.len().min(self.max_chars);
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
}

fn truncate_text(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        text
    } else {
        &text[..text.floor_char_boundary(max_chars)]
    }
}

/// Try to get model context length via Ollama's /api/show endpoint.
fn query_context_length(agent: &ureq::Agent, embedder_url: &str, model: &str) -> Option<usize> {
    let base = embedder_url.split("/v1/").next()?;
    let show_url = format!("{}/api/show", base);

    let body = serde_json::json!({"model": model});
    let resp = agent
        .post(&show_url)
        .content_type("application/json")
        .send(serde_json::to_string(&body).unwrap().as_bytes())
        .ok()?;
    if resp.status().as_u16() >= 400 {
        return None;
    }
    let text = resp.into_body().read_to_string().ok()?;
    let info: serde_json::Value = serde_json::from_str(&text).ok()?;

    let model_info = &info["model_info"];
    let context_tokens = model_info["bert.context_length"]
        .as_u64()
        .or_else(|| model_info["general.context_length"].as_u64())?
        as usize;

    let max_chars = context_tokens * 5 / 2;
    tracing::info!(
        "Remote model context: {} tokens, truncating at {} chars",
        context_tokens,
        max_chars
    );
    Some(max_chars)
}

/// Extract a human-readable error message from an API error response body.
pub(super) fn extract_error_message(body: &str) -> String {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(msg) = json["error"]["message"].as_str() {
            return msg.to_string();
        }
        if let Some(msg) = json["error"].as_str() {
            return msg.to_string();
        }
        if let Some(msg) = json["message"].as_str() {
            return msg.to_string();
        }
    }
    body.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn make_test_remote(max_chars: usize) -> RemoteEmbedder {
        RemoteEmbedder {
            url: String::new(),
            model: String::new(),
            embedding_dim: None,
            max_chars,
            agent: ureq::Agent::new_with_config(ureq::config::Config::default()),
        }
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
        let text = "café";
        let result = truncate_text(text, 4);
        assert!(result.len() <= 4);
        assert!(result.is_char_boundary(result.len()));
    }

    // --- make_batches tests ---

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
        let texts = vec!["hello", "world", "foo", "bar"];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 4);
    }

    #[test]
    fn test_batches_split_on_payload_size() {
        let remote = make_test_remote(50);
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
        let huge = "x".repeat(200);
        let texts = vec![huge.as_str()];
        let batches = remote.make_batches(&texts);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_batches_mixed_sizes_splits_when_exceeding_threshold() {
        let remote = make_test_remote(100);
        let medium = "a".repeat(80);
        let long = "b".repeat(150);
        // medium(80) + long(100 truncated) = 180 < 200 threshold, fits in one batch
        // Adding another medium(80) → 260 > 200, forces a split
        let texts: Vec<&str> = vec![medium.as_str(), long.as_str(), medium.as_str()];
        let batches = remote.make_batches(&texts);
        assert_eq!(
            batches.len(),
            2,
            "expected 2 batches when payload exceeds threshold, got {}",
            batches.len()
        );
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
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
        let texts = vec!["aaaa-1111-bbbbb", "cccc-2222-ddddd", "eeee-3333-fffff"];
        let batches = remote.make_batches(&texts);
        let flat: Vec<&str> = batches.into_iter().flatten().collect();
        assert_eq!(flat, texts);
    }

    // --- parse_embeddings tests ---

    #[test]
    fn test_parse_embeddings_normalizes_and_reorders() {
        let mut remote = make_test_remote(100);
        let response = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                {"index": 0, "embedding": [3.0, 4.0, 0.0]}
            ]
        });

        let result = remote.parse_embeddings(&response, 2).unwrap();
        assert_eq!(result.len(), 2);

        for emb in &result {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
        }

        assert!(
            result[0][0] > 0.5,
            "index 0 should start with a large value"
        );
        assert!(result[1][0].abs() < 1e-5, "index 1 should start near zero");
        assert_eq!(remote.embedding_dim, Some(3));
    }

    #[test]
    fn test_parse_embeddings_rejects_missing_entries() {
        let mut remote = make_test_remote(100);
        let response = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [1.0, 0.0]}
            ]
        });

        let err = remote.parse_embeddings(&response, 2).unwrap_err();
        assert!(err
            .to_string()
            .contains("returned 1 embeddings for 2 inputs"));
    }

    #[test]
    fn test_parse_embeddings_rejects_duplicate_indices() {
        let mut remote = make_test_remote(100);
        let response = serde_json::json!({
            "data": [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]}
            ]
        });

        let err = remote.parse_embeddings(&response, 2).unwrap_err();
        assert!(err.to_string().contains("duplicate embedding index 0"));
    }

    #[test]
    fn test_parse_embeddings_rejects_out_of_range_index() {
        let mut remote = make_test_remote(100);
        let response = serde_json::json!({
            "data": [
                {"index": 2, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]}
            ]
        });

        let err = remote.parse_embeddings(&response, 2).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    // --- extract_error_message tests ---

    #[test]
    fn test_extract_error_message_openai_style() {
        let body = r#"{"error": {"message": "model not found", "type": "invalid_request"}}"#;
        assert_eq!(extract_error_message(body), "model not found");
    }

    #[test]
    fn test_extract_error_message_simple_error() {
        let body = r#"{"error": "something went wrong"}"#;
        assert_eq!(extract_error_message(body), "something went wrong");
    }

    #[test]
    fn test_extract_error_message_ollama_style() {
        let body = r#"{"message": "model 'mxbai' not found, try pulling it first"}"#;
        assert_eq!(
            extract_error_message(body),
            "model 'mxbai' not found, try pulling it first"
        );
    }

    #[test]
    fn test_extract_error_message_plain_text() {
        let body = "Internal Server Error";
        assert_eq!(extract_error_message(body), "Internal Server Error");
    }

    #[test]
    fn test_extract_error_message_truncates_long_body() {
        let body = "x".repeat(300);
        assert_eq!(extract_error_message(&body).len(), 200);
    }
}
