use std::net::TcpListener;

use anyhow::{Context, Result};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use url::Url;

use crate::embedder::Embedder;
use crate::index::Index;
use crate::output::format_json_result;
use crate::pipeline::{EmbedWorker, PipelineStatus, SearchOutcome, StreamingIndexer};
use crate::types::SearchScope;

fn respond(request: Request, response: Response<std::io::Cursor<Vec<u8>>>) {
    if let Err(err) = request.respond(response) {
        tracing::debug!("failed to send HTTP response: {err}");
    }
}

fn json_content_header() -> Header {
    "Content-Type: application/x-ndjson"
        .parse()
        .expect("valid header")
}

fn json_response_header() -> Header {
    "Content-Type: application/json"
        .parse()
        .expect("valid header")
}

fn error_response(status: u16, message: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::json!({"error": message}).to_string();
    Response::from_string(body)
        .with_status_code(StatusCode(status))
        .with_header(json_response_header())
}

/// Handle a single HTTP request. Returns Ok(()) after responding.
fn handle_request(
    request: Request,
    worker: &EmbedWorker,
    default_top_k: usize,
    default_threshold: f32,
    root: &str,
    pipeline_status: &PipelineStatus,
    path_scopes: &[String],
) -> Result<()> {
    if request.method() != &Method::Get {
        respond(request, error_response(405, "method not allowed"));
        return Ok(());
    }

    let raw_url = request.url().to_string();
    let full_url = format!("http://localhost{raw_url}");
    let parsed = match Url::parse(&full_url) {
        Ok(u) => u,
        Err(_) => {
            respond(request, error_response(400, "invalid URL"));
            return Ok(());
        }
    };

    match parsed.path() {
        "/search" => handle_search(
            request,
            &parsed,
            worker,
            default_top_k,
            default_threshold,
            root,
        ),
        "/status" => {
            let mut status = serde_json::to_value(pipeline_status).unwrap();
            status["version"] = serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string());
            status["root"] = serde_json::Value::String(root.to_string());
            if !path_scopes.is_empty() {
                status["scope"] = serde_json::json!(path_scopes);
            }
            let body = serde_json::to_string(&status).unwrap();
            let resp = Response::from_string(body).with_header(json_response_header());
            respond(request, resp);
            Ok(())
        }
        _ => {
            respond(request, error_response(404, "not found"));
            Ok(())
        }
    }
}

fn handle_search(
    request: Request,
    parsed: &Url,
    worker: &EmbedWorker,
    default_top_k: usize,
    default_threshold: f32,
    root: &str,
) -> Result<()> {
    let params: Vec<(String, String)> = parsed.query_pairs().into_owned().collect();
    let q = params
        .iter()
        .find(|(k, _)| k == "q")
        .map(|(_, v)| v.clone());

    let q = match q {
        Some(q) if !q.is_empty() => q,
        _ => {
            respond(
                request,
                error_response(400, "missing or empty 'q' parameter"),
            );
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

    let request_id = worker.search(&q, top_k, threshold);
    match worker.recv_result_for(request_id) {
        Some(SearchOutcome::Results { results, .. }) => {
            let mut body = String::new();
            for result in &results {
                let json = format_json_result(result, root);
                body.push_str(&json.to_string());
                body.push('\n');
            }
            let resp = Response::from_string(body).with_header(json_content_header());
            respond(request, resp);
        }
        Some(SearchOutcome::SearchError { message, .. }) => {
            respond(request, error_response(500, &message));
        }
        Some(SearchOutcome::EmbedError { message, .. }) => {
            respond(request, error_response(500, &message));
        }
        None => {
            respond(request, error_response(500, "worker unavailable"));
        }
    }
    Ok(())
}

pub struct ServeConfig<'a> {
    pub port: Option<u16>,
    pub default_top_k: usize,
    pub default_threshold: f32,
    pub quiet: bool,
    pub root: &'a str,
    pub scope: SearchScope,
}

pub fn run_streaming(
    embedder: Embedder,
    idx: Index,
    indexer: StreamingIndexer,
    config: ServeConfig<'_>,
) -> Result<()> {
    let port = config.port.unwrap_or(0);
    let listener =
        TcpListener::bind(("127.0.0.1", port)).context("Failed to bind HTTP listener")?;
    let actual_port = listener.local_addr()?.port();

    let server = Server::from_listener(listener, None)
        .map_err(|e| anyhow::anyhow!("Failed to create HTTP server: {e}"))?;

    if !config.quiet {
        eprintln!("Listening on http://127.0.0.1:{actual_port}");
    }

    let path_scopes = config.scope.path_scopes.clone();
    let worker = EmbedWorker::spawn(embedder, idx, indexer, config.scope);
    let mut indexing_announced = false;
    let mut pipeline_status = PipelineStatus::initial();

    loop {
        // Check index progress (non-blocking)
        if let Some(status) = worker.drain_progress() {
            pipeline_status = status;
            if let PipelineStatus::Ready { files, chunks } = status {
                if !indexing_announced {
                    indexing_announced = true;
                    if !config.quiet {
                        eprintln!("Indexed {files} files, {chunks} chunks ready.");
                    }
                }
            }
        }

        let request = match server.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(Some(req)) => req,
            Ok(None) | Err(_) => continue,
        };

        handle_request(
            request,
            &worker,
            config.default_top_k,
            config.default_threshold,
            config.root,
            &pipeline_status,
            &path_scopes,
        )?;
    }
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
            let listener =
                TcpListener::bind("127.0.0.1:0").expect("bind ephemeral test server port");
            let port = listener
                .local_addr()
                .expect("read bound test server address")
                .port();
            drop(listener);

            thread::spawn(move || {
                let mut embedder = Embedder::new_local().expect("initialize local test embedder");
                let idx = Index::open_in_memory().expect("open in-memory test index");

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

                let embeddings: Vec<Vec<f32>> = texts
                    .iter()
                    .map(|t| embedder.embed(t).expect("embed test fixture text"))
                    .collect();

                // Insert each file into the index
                for (i, (chunk, emb)) in chunks.iter().zip(embeddings.iter()).enumerate() {
                    idx.upsert_file(
                        &format!("test{i}.rs"),
                        &format!("hash{i}"),
                        &[chunk.clone()],
                        &[emb.clone()],
                        &[false],
                    )
                    .expect("insert fixture chunk into test index");
                }

                // Create a dummy indexer (no files to index) for run_streaming
                let (dummy_tx, dummy_rx) = std::sync::mpsc::sync_channel(0);
                drop(dummy_tx);
                let indexer =
                    StreamingIndexer::new(dummy_rx, 500, 100, 1, std::path::Path::new(""), None);
                run_streaming(
                    embedder,
                    idx,
                    indexer,
                    ServeConfig {
                        port: Some(port),
                        default_top_k: 10,
                        default_threshold: 0.3,
                        quiet: true,
                        root: "/test/root",
                        scope: SearchScope::default(),
                    },
                )
                .expect("run shared test HTTP server");
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
        let mut stream =
            TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect to test server");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("set test server read timeout");
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .expect("write HTTP request to test server");
        stream.flush().expect("flush HTTP request");

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read HTTP response from test server");

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

        let json: serde_json::Value =
            serde_json::from_str(lines[0]).expect("parse first JSONL response line");
        assert_eq!(json["root"], "/test/root");
        assert!(json["file"]
            .as_str()
            .expect("response includes file path")
            .ends_with(".rs"));
        assert!(json["score"].as_f64().expect("response includes score") > 0.0);
        assert!(!json["text"]
            .as_str()
            .expect("response includes matched text")
            .is_empty());
        assert!(
            json["start_line"]
                .as_u64()
                .expect("response includes start_line")
                >= 1
        );
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
    fn test_status_returns_ready() {
        let port = test_port();
        // Poll /status until the server reports "ready" (indexing complete)
        let mut body = String::new();
        for _ in 0..50 {
            let (s, b, _) = http_request("GET", port, "/status");
            if s == 200 {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&b) {
                    if json["status"] == "ready" {
                        body = b;
                        break;
                    }
                }
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(
            !body.is_empty(),
            "server did not reach ready status within 5s"
        );

        let (status, body, content_type) = http_request("GET", port, "/status");

        assert_eq!(status, 200);
        assert_eq!(content_type, "application/json");
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["status"], "ready");
        assert!(
            json["files"].as_u64().unwrap() >= 3,
            "expected at least 3 files, got: {json}"
        );
        assert!(
            json["chunks"].as_u64().unwrap() >= 3,
            "expected at least 3 chunks, got: {json}"
        );
        assert!(
            json["version"].as_str().is_some(),
            "expected version field, got: {json}"
        );
        assert_eq!(
            json["root"], "/test/root",
            "expected root field, got: {json}"
        );
        // No scope when SearchScope is default (project root)
        assert!(
            json.get("scope").is_none(),
            "expected no scope field for default scope, got: {json}"
        );
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
