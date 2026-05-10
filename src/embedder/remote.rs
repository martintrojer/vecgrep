use anyhow::Result;

use super::{l2_normalize, tokens_to_chars};

/// Default max chars per text for remote embedders.
/// Assumes 480-token context (chosen to match historical 1200-char default)
/// at the shared chars-per-token ratio in `embedder::mod`.
const DEFAULT_REMOTE_MAX_CHARS: usize = tokens_to_chars(480);

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

            let mut embedding: Vec<f32> = item["embedding"]
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

            l2_normalize(&mut embedding);
            embeddings[index] = Some(embedding);
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

    let max_chars = tokens_to_chars(context_tokens);
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

    // --- HTTP-level integration tests for embed_batch fallback chain.
    // These cover the per-text retry, zero-vector fallback when dim is
    // known, and dim-unknown error propagation paths called out by
    // tr_remote_fallback_gap.

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// Scripted response: status code + body (kept simple, no headers).
    #[derive(Clone)]
    struct MockResponse {
        status: u16,
        body: String,
    }

    impl MockResponse {
        fn ok_embedding_dim(dim: usize, n: usize) -> Self {
            let mut data = Vec::new();
            for i in 0..n {
                let emb: Vec<f32> = (0..dim).map(|j| (i + j + 1) as f32 * 0.001).collect();
                data.push(serde_json::json!({"index": i, "embedding": emb}));
            }
            let body = serde_json::json!({ "data": data }).to_string();
            Self { status: 200, body }
        }
        fn http_400(message: &str) -> Self {
            Self {
                status: 400,
                body: serde_json::json!({"error": {"message": message}}).to_string(),
            }
        }
        fn http_500() -> Self {
            Self {
                status: 500,
                body: "Internal Server Error".to_string(),
            }
        }
    }

    /// Spawn a tiny HTTP server that pops responses from `script` for each
    /// request received. Returns the bound URL and a join handle. The
    /// server shuts down when `script` is exhausted (next connection is
    /// closed without response).
    fn spawn_mock_server(
        script: Vec<MockResponse>,
    ) -> (String, std::thread::JoinHandle<()>, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://127.0.0.1:{port}/v1/embeddings");
        let received_bodies = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = Arc::clone(&received_bodies);

        let handle = std::thread::spawn(move || {
            let script = script;
            let mut idx = 0;
            for stream in listener.incoming() {
                let mut stream = match stream {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let mut buf = [0u8; 8192];
                let n = match stream.read(&mut buf) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                // Capture the body for assertion. Find double-CRLF.
                if let Some(body_pos) = req.find("\r\n\r\n") {
                    received_clone
                        .lock()
                        .unwrap()
                        .push(req[body_pos + 4..].to_string());
                } else {
                    received_clone.lock().unwrap().push(String::new());
                }
                if idx >= script.len() {
                    break;
                }
                let resp = &script[idx];
                idx += 1;
                let response = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp.status,
                    resp.body.len(),
                    resp.body
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
                if idx >= script.len() {
                    break;
                }
            }
        });

        (url, handle, received_bodies)
    }

    /// Build a RemoteEmbedder pointing at `url` without contacting it for
    /// context discovery (which would consume the first scripted response).
    fn make_remote_at(url: &str) -> RemoteEmbedder {
        RemoteEmbedder {
            url: url.to_string(),
            model: "test-model".to_string(),
            embedding_dim: None,
            max_chars: 1000,
            agent: ureq::Agent::new_with_config(
                ureq::config::Config::builder()
                    .timeout_global(Some(std::time::Duration::from_secs(5)))
                    .http_status_as_error(false)
                    .build(),
            ),
        }
    }

    #[test]
    fn test_embed_batch_success_returns_parsed_embeddings() {
        let dim = 3;
        let (url, _h, _) = spawn_mock_server(vec![MockResponse::ok_embedding_dim(dim, 2)]);
        let mut remote = make_remote_at(&url);
        let result = remote.embed_batch(&["alpha", "beta"]).unwrap();
        assert_eq!(result.len(), 2, "two inputs → two embeddings");
        for emb in &result {
            assert_eq!(emb.len(), dim, "each embedding has the discovered dim");
        }
        assert_eq!(remote.embedding_dim, Some(dim));
    }

    #[test]
    fn test_embed_batch_400_then_per_text_retry_fills_zero_when_dim_known() {
        // First batch (2 texts) fails with 400. Per-text retry: first text
        // succeeds (so dim becomes known), second text fails with 400 again
        // — a zero vector is emitted at index 1.
        let dim = 3;
        let script = vec![
            MockResponse::http_400("input too long"), // initial batch
            MockResponse::ok_embedding_dim(dim, 1),   // per-text retry of text 0
            MockResponse::http_400("still bad"),      // per-text retry of text 1
        ];
        let (url, _h, _) = spawn_mock_server(script);
        let mut remote = make_remote_at(&url);
        let result = remote.embed_batch(&["alpha", "beta"]).unwrap();

        assert_eq!(result.len(), 2, "output length must equal input length");
        assert_eq!(result[0].len(), dim, "first chunk has full embedding");
        assert_eq!(
            result[1].len(),
            dim,
            "second chunk has zero vector at correct dim"
        );
        assert!(
            result[1].iter().all(|&v| v == 0.0),
            "second chunk must be a zero vector (failed retry, dim known)"
        );
        assert!(
            result[0].iter().any(|&v| v != 0.0),
            "first chunk must be a real embedding"
        );
    }

    #[test]
    fn test_embed_batch_500_with_unknown_dim_propagates_error() {
        // No prior successful request → dim is None. A 500 in the per-text
        // retry must surface as Err so the caller doesn't get a phantom
        // zero-length zero vector.
        let script = vec![
            MockResponse::http_500(), // initial batch
            MockResponse::http_500(), // per-text retry of text 0 (dim still unknown)
        ];
        let (url, _h, _) = spawn_mock_server(script);
        let mut remote = make_remote_at(&url);
        let err = remote.embed_batch(&["alpha"]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("500"),
            "error must surface upstream HTTP 500: {msg}"
        );
    }

    #[test]
    fn test_embed_batch_400_for_oversize_input_per_text_retry() {
        // Ollama-style behavior: HTTP 400 for input exceeding context.
        // Initial batch fails; per-text succeeds for both. Asserts
        // ordering and length preservation.
        let dim = 4;
        let script = vec![
            MockResponse::http_400("context length exceeded"),
            MockResponse::ok_embedding_dim(dim, 1),
            MockResponse::ok_embedding_dim(dim, 1),
        ];
        let (url, _h, _) = spawn_mock_server(script);
        let mut remote = make_remote_at(&url);
        let result = remote
            .embed_batch(&["first chunk text", "second chunk text"])
            .unwrap();
        assert_eq!(result.len(), 2);
        // Both should be real embeddings (not zero vectors).
        for (i, emb) in result.iter().enumerate() {
            assert_eq!(emb.len(), dim, "emb {i} has expected dim");
            assert!(
                emb.iter().any(|&v| v != 0.0),
                "emb {i} must be non-zero (per-text retry succeeded)"
            );
        }
    }

    #[test]
    fn test_query_context_length_returns_none_on_404() {
        // /api/show endpoint returning 404 must yield None (not panic)
        // so RemoteEmbedder::new falls back to DEFAULT_REMOTE_MAX_CHARS.
        let script = vec![MockResponse {
            status: 404,
            body: "{}".to_string(),
        }];
        let (url, _h, _) = spawn_mock_server(script);
        // Strip /v1/embeddings tail so query_context_length finds the base.
        let base_url = url.replace("/v1/embeddings", "/v1/embeddings");
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(5)))
                .http_status_as_error(false)
                .build(),
        );
        let result = query_context_length(&agent, &base_url, "missing-model");
        assert!(
            result.is_none(),
            "404 from /api/show must yield None, got {result:?}"
        );
    }
}
