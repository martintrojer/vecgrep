use std::net::TcpListener;

use anyhow::{Context, Result};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use url::Url;

use crate::embedder::Embedder;
use crate::index::Index;
use crate::output::format_json_result;
use crate::pipeline::StreamingIndexer;

fn json_content_header() -> Header {
    "Content-Type: application/x-ndjson"
        .parse()
        .expect("valid header")
}

fn json_error_header() -> Header {
    "Content-Type: application/json"
        .parse()
        .expect("valid header")
}

/// Handle a single HTTP search request. Returns Ok(()) after responding.
fn handle_request(
    request: Request,
    embedder: &mut Embedder,
    idx: &Index,
    default_top_k: usize,
    default_threshold: f32,
    root: &str,
) -> Result<()> {
    let json_content_type = json_content_header();
    let json_error_type = json_error_header();

    if request.method() != &Method::Get {
        let body = r#"{"error":"method not allowed"}"#;
        let resp = Response::from_string(body)
            .with_status_code(StatusCode(405))
            .with_header(json_error_type);
        let _ = request.respond(resp);
        return Ok(());
    }

    let raw_url = request.url().to_string();
    let full_url = format!("http://localhost{raw_url}");
    let parsed = match Url::parse(&full_url) {
        Ok(u) => u,
        Err(_) => {
            let body = r#"{"error":"invalid URL"}"#;
            let resp = Response::from_string(body)
                .with_status_code(StatusCode(400))
                .with_header(json_error_type);
            let _ = request.respond(resp);
            return Ok(());
        }
    };

    if parsed.path() != "/search" {
        let body = r#"{"error":"not found"}"#;
        let resp = Response::from_string(body)
            .with_status_code(StatusCode(404))
            .with_header(json_error_type);
        let _ = request.respond(resp);
        return Ok(());
    }

    let params: Vec<(String, String)> = parsed.query_pairs().into_owned().collect();
    let q = params
        .iter()
        .find(|(k, _)| k == "q")
        .map(|(_, v)| v.clone());

    let q = match q {
        Some(q) if !q.is_empty() => q,
        _ => {
            let body = r#"{"error":"missing or empty 'q' parameter"}"#;
            let resp = Response::from_string(body)
                .with_status_code(StatusCode(400))
                .with_header(json_error_type);
            let _ = request.respond(resp);
            return Ok(());
        }
    };

    let top_k: usize = params
        .iter()
        .find(|(k, _)| k == "k")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(default_top_k);

    let threshold: f32 = params
        .iter()
        .find(|(k, _)| k == "threshold")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(default_threshold);

    let query_embedding = match embedder.embed(&q) {
        Ok(e) => e,
        Err(e) => {
            let body = serde_json::json!({"error": format!("{e:#}")}).to_string();
            let resp = Response::from_string(body)
                .with_status_code(StatusCode(500))
                .with_header(json_error_type);
            let _ = request.respond(resp);
            return Ok(());
        }
    };

    let results = idx.search(&query_embedding, top_k, threshold)?;

    let mut body = String::new();
    for result in &results {
        let json = format_json_result(result, root);
        body.push_str(&json.to_string());
        body.push('\n');
    }

    let resp = Response::from_string(body).with_header(json_content_type);
    let _ = request.respond(resp);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_streaming(
    embedder: &mut Embedder,
    idx: &Index,
    mut indexer: StreamingIndexer,
    port: Option<u16>,
    default_top_k: usize,
    default_threshold: f32,
    quiet: bool,
    root: &str,
) -> Result<()> {
    let port = port.unwrap_or(0);
    let listener =
        TcpListener::bind(("127.0.0.1", port)).context("Failed to bind HTTP listener")?;
    let actual_port = listener.local_addr()?.port();

    let server = Server::from_listener(listener, None)
        .map_err(|e| anyhow::anyhow!("Failed to create HTTP server: {e}"))?;

    if !quiet {
        eprintln!("Listening on http://127.0.0.1:{actual_port}");
    }

    let mut indexing_announced = false;

    loop {
        if !indexer.indexing_done {
            indexer.poll(embedder, idx)?;
            if indexer.indexing_done && !indexing_announced {
                indexing_announced = true;
                if !quiet {
                    let chunk_count = idx.chunk_count().unwrap_or(0);
                    eprintln!(
                        "Indexing complete. {} files indexed, {} chunks ready.",
                        indexer.indexed_count, chunk_count
                    );
                }
            }
        }

        let request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(req)) => req,
            Ok(None) | Err(_) => continue,
        };

        handle_request(
            request,
            embedder,
            idx,
            default_top_k,
            default_threshold,
            root,
        )?;
    }
}

pub fn run(
    embedder: &mut Embedder,
    idx: &Index,
    port: Option<u16>,
    default_top_k: usize,
    default_threshold: f32,
    quiet: bool,
    root: &str,
) -> Result<()> {
    let port = port.unwrap_or(0);
    let listener =
        TcpListener::bind(("127.0.0.1", port)).context("Failed to bind HTTP listener")?;
    let actual_port = listener.local_addr()?.port();

    let server = Server::from_listener(listener, None)
        .map_err(|e| anyhow::anyhow!("Failed to create HTTP server: {e}"))?;

    if !quiet {
        eprintln!("Listening on http://127.0.0.1:{actual_port}");
    }

    for request in server.incoming_requests() {
        handle_request(
            request,
            embedder,
            idx,
            default_top_k,
            default_threshold,
            root,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder;
    use crate::types::Chunk;
    use std::io::{Read as _, Write as _};
    use std::net::TcpStream;
    use std::sync::OnceLock;
    use std::thread;
    use std::time::Duration;

    static SERVER_PORT: OnceLock<u16> = OnceLock::new();

    /// Start a shared test server (once) and return its port.
    fn test_port() -> u16 {
        *SERVER_PORT.get_or_init(|| {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);

            thread::spawn(move || {
                let mut embedder = Embedder::new_local().unwrap();
                let idx = Index::open_in_memory().unwrap();

                let texts = ["error handling in rust", "memory management", "HTTP server"];
                let chunks: Vec<Chunk> = texts
                    .iter()
                    .enumerate()
                    .map(|(i, text)| Chunk {
                        file_path: format!("test{i}.rs"),
                        text: text.to_string(),
                        start_line: 1,
                        end_line: 1,
                    })
                    .collect();

                let embeddings: Vec<Vec<f32>> =
                    texts.iter().map(|t| embedder.embed(t).unwrap()).collect();

                // Insert each file into the index
                for (i, (chunk, emb)) in chunks.iter().zip(embeddings.iter()).enumerate() {
                    idx.upsert_file(
                        &format!("test{i}.rs"),
                        &format!("hash{i}"),
                        &[chunk.clone()],
                        &[emb.clone()],
                    )
                    .unwrap();
                }

                run(&mut embedder, &idx, Some(port), 10, 0.3, true, "/test/root").unwrap();
            });

            for _ in 0..50 {
                if TcpStream::connect(format!("127.0.0.1:{port}")).is_ok() {
                    return port;
                }
                thread::sleep(Duration::from_millis(100));
            }
            panic!("test server did not start within 5 seconds");
        })
    }

    fn http_request(method: &str, port: u16, path: &str) -> (u16, String, String) {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        stream.flush().unwrap();

        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        let status_line = response.lines().next().unwrap_or("");
        let status_code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let content_type = response
            .lines()
            .find(|l| l.to_lowercase().starts_with("content-type:"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
            .unwrap_or_default();

        let body_start = response.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let body = response[body_start..].to_string();

        (status_code, body, content_type)
    }

    #[test]
    fn test_search_returns_jsonl() {
        let port = test_port();
        let (status, body, content_type) = http_request("GET", port, "/search?q=error+handling");

        assert_eq!(status, 200);
        assert_eq!(content_type, "application/x-ndjson");

        let lines: Vec<&str> = body.lines().collect();
        assert!(!lines.is_empty(), "expected at least one result");

        let json: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(json["root"], "/test/root");
        assert!(json["file"].as_str().unwrap().ends_with(".rs"));
        assert!(json["score"].as_f64().unwrap() > 0.0);
        assert!(!json["text"].as_str().unwrap().is_empty());
        assert!(json["start_line"].as_u64().unwrap() >= 1);
        assert!(json.get("end_line").is_some());
    }

    #[test]
    fn test_search_k_override() {
        let port = test_port();
        let (status, body, _) = http_request("GET", port, "/search?q=test&k=1");

        assert_eq!(status, 200);
        let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert!(
            lines.len() <= 1,
            "expected at most 1 result with k=1, got {}",
            lines.len()
        );
    }

    #[test]
    fn test_search_threshold_override() {
        let port = test_port();
        let (status, body, _) = http_request("GET", port, "/search?q=test&threshold=0.99");

        assert_eq!(status, 200);
        for line in body.lines().filter(|l| !l.is_empty()) {
            let json: serde_json::Value = serde_json::from_str(line).unwrap();
            let score = json["score"].as_f64().unwrap();
            assert!(score >= 0.99, "score {score} should be >= 0.99");
        }
    }

    #[test]
    fn test_missing_query_returns_400() {
        let port = test_port();
        let (status, body, content_type) = http_request("GET", port, "/search");

        assert_eq!(status, 400);
        assert_eq!(content_type, "application/json");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(json.get("error").is_some());
    }

    #[test]
    fn test_empty_query_returns_400() {
        let port = test_port();
        let (status, body, _) = http_request("GET", port, "/search?q=");

        assert_eq!(status, 400);
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("missing"));
    }

    #[test]
    fn test_not_found_returns_404() {
        let port = test_port();
        let (status, body, content_type) = http_request("GET", port, "/invalid");

        assert_eq!(status, 404);
        assert_eq!(content_type, "application/json");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"], "not found");
    }

    #[test]
    fn test_post_returns_405() {
        let port = test_port();
        let (status, body, content_type) = http_request("POST", port, "/search?q=test");

        assert_eq!(status, 405);
        assert_eq!(content_type, "application/json");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["error"], "method not allowed");
    }
}
