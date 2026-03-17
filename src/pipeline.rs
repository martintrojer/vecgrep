use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

use crate::chunker;
use crate::embedder::Embedder;
use crate::index::Index;
use crate::paths;
use crate::types::{SearchResult, SearchScope};
use crate::walker::{StreamProgress, WalkedFile};

/// Pipeline state — single source of truth for indexing progress.
/// Used by CLI spinner, TUI status bar, serve endpoint, and future `/status` API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PipelineStatus {
    /// Walker and indexer running. `total` is `None` until the walker finishes.
    Indexing {
        indexed: usize,
        total: Option<usize>,
        chunks: usize,
    },
    /// All done.
    Ready { files: usize, chunks: usize },
}

impl PipelineStatus {
    pub fn indexed(&self) -> usize {
        match self {
            Self::Indexing { indexed, .. } => *indexed,
            Self::Ready { .. } => 0,
        }
    }
}

/// Manages incremental indexing from a streaming channel.
pub struct StreamingIndexer {
    rx: Receiver<WalkedFile>,
    pub indexing_done: bool,
    pub indexed_count: usize,
    pub indexed_chunks: usize,
    /// All file paths seen (for stale removal).
    pub all_paths: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    pub(crate) batch_size: usize,
    cwd_suffix: Box<Path>,
    stream_progress: Option<Arc<StreamProgress>>,
}

impl StreamingIndexer {
    pub fn new(
        rx: Receiver<WalkedFile>,
        chunk_size: usize,
        chunk_overlap: usize,
        batch_size: usize,
        cwd_suffix: &Path,
        stream_progress: Option<Arc<StreamProgress>>,
    ) -> Self {
        Self {
            rx,
            indexing_done: false,
            indexed_count: 0,
            indexed_chunks: 0,
            all_paths: Vec::new(),
            chunk_size,
            chunk_overlap,
            batch_size,
            cwd_suffix: cwd_suffix.into(),
            stream_progress,
        }
    }

    pub fn status(&self) -> PipelineStatus {
        if self.indexing_done {
            return PipelineStatus::Ready {
                files: self.all_paths.len(),
                chunks: self.indexed_chunks,
            };
        }

        let snapshot = self
            .stream_progress
            .as_ref()
            .map(|progress| progress.snapshot())
            .unwrap_or_default();

        let total = if snapshot.walk_done {
            Some(snapshot.walked_files.max(self.all_paths.len()))
        } else {
            None
        };

        PipelineStatus::Indexing {
            indexed: self.indexed_count,
            total,
            chunks: self.indexed_chunks,
        }
    }

    /// Receive one file from the channel, hash-check it, and return it if it needs indexing.
    /// Returns `None` if the file is up-to-date or the channel is empty/closed.
    fn recv_one(&mut self, idx: &Index, blocking: bool) -> Option<(WalkedFile, String)> {
        let mut file = if blocking {
            match self.rx.recv() {
                Ok(f) => f,
                Err(_) => {
                    self.indexing_done = true;
                    return None;
                }
            }
        } else {
            match self.rx.try_recv() {
                Ok(f) => f,
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => {
                    self.indexing_done = true;
                    return None;
                }
            }
        };
        file.rel_path = paths::to_project_relative(&file.rel_path, &self.cwd_suffix);
        self.all_paths.push(file.rel_path.clone());

        let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
        let needs_index = match idx.get_file_hash(&file.rel_path) {
            Ok(Some(stored_hash)) => stored_hash != hash,
            _ => true,
        };
        if needs_index {
            Some((file, hash))
        } else {
            None
        }
    }

    /// Drain up to `batch_size` files from the channel (non-blocking) and process them.
    /// Returns `true` if any new files were indexed (caller should re-search).
    pub fn poll(&mut self, embedder: &mut Embedder, idx: &Index) -> Result<bool> {
        if self.indexing_done {
            return Ok(false);
        }

        let mut batch: Vec<(WalkedFile, String)> = Vec::new();
        while batch.len() < self.batch_size && !self.indexing_done {
            match self.recv_one(idx, false) {
                Some(entry) => batch.push(entry),
                None => break,
            }
        }

        if !batch.is_empty() {
            self.indexed_count += batch.len();
            let chunk_count =
                process_batch(embedder, idx, &batch, self.chunk_size, self.chunk_overlap)?;
            self.indexed_chunks += chunk_count;
            return Ok(true);
        }

        Ok(false)
    }

    /// Blocking drain: process all files from the channel until it closes.
    /// Calls `on_batch` after each batch is processed, with the count indexed so far.
    /// Returns the total number of files indexed.
    pub fn drain_all<F>(
        &mut self,
        embedder: &mut Embedder,
        idx: &Index,
        mut on_batch: F,
    ) -> Result<usize>
    where
        F: FnMut(PipelineStatus) -> Result<bool>,
    {
        while !self.indexing_done {
            let mut batch: Vec<(WalkedFile, String)> = Vec::new();

            // First file: blocking recv
            if let Some(entry) = self.recv_one(idx, true) {
                batch.push(entry);
            }
            if self.indexing_done && batch.is_empty() {
                break;
            }

            // Fill rest of batch: non-blocking
            while batch.len() < self.batch_size && !self.indexing_done {
                match self.recv_one(idx, false) {
                    Some(entry) => batch.push(entry),
                    None => break,
                }
            }

            if !batch.is_empty() {
                self.indexed_count += batch.len();
                let chunk_count =
                    process_batch(embedder, idx, &batch, self.chunk_size, self.chunk_overlap)?;
                self.indexed_chunks += chunk_count;

                if !on_batch(self.status())? {
                    return Ok(self.indexed_count);
                }
            }
        }

        Ok(self.indexed_count)
    }
}

// --- Background embed worker for non-blocking TUI/serve modes ---

/// Worker batch size — small for responsiveness between search request checks.
const WORKER_BATCH_SIZE: usize = 2;

enum WorkerRequest {
    Search {
        request_id: u64,
        query: String,
        top_k: usize,
        threshold: f32,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum SearchOutcome {
    Results {
        request_id: u64,
        results: Vec<SearchResult>,
    },
    SearchError {
        request_id: u64,
        message: String,
    },
    EmbedError {
        request_id: u64,
        message: String,
    },
}

impl SearchOutcome {
    pub fn request_id(&self) -> u64 {
        let (Self::Results { request_id, .. }
        | Self::SearchError { request_id, .. }
        | Self::EmbedError { request_id, .. }) = self;
        *request_id
    }
}

/// Runs embedding and indexing on a background thread so the UI stays responsive.
pub struct EmbedWorker {
    req_tx: mpsc::Sender<WorkerRequest>,
    result_rx: mpsc::Receiver<SearchOutcome>,
    progress_rx: mpsc::Receiver<PipelineStatus>,
    next_request_id: AtomicU64,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl EmbedWorker {
    /// Spawn a background worker that owns the embedder, index, and indexer.
    /// `explicit_paths`: when `Some`, only these explicit files appear in results
    /// alongside normal files. When `None`, all explicit files are excluded.
    pub fn spawn(
        embedder: Embedder,
        idx: Index,
        mut indexer: StreamingIndexer,
        scope: SearchScope,
    ) -> Self {
        let (req_tx, req_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();

        indexer.batch_size = WORKER_BATCH_SIZE;

        let handle = std::thread::spawn(move || {
            worker_loop(
                embedder,
                idx,
                indexer,
                req_rx,
                result_tx,
                progress_tx,
                scope,
            );
        });

        Self {
            req_tx,
            result_rx,
            progress_rx,
            next_request_id: AtomicU64::new(1),
            handle: Some(handle),
        }
    }

    /// Send a search request to the worker.
    pub fn search(&self, query: &str, top_k: usize, threshold: f32) -> u64 {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.req_tx
            .send(WorkerRequest::Search {
                request_id,
                query: query.to_string(),
                top_k,
                threshold,
            })
            .ok();
        request_id
    }

    /// Non-blocking check for search results.
    pub fn try_recv_results(&self) -> Option<SearchOutcome> {
        self.result_rx.try_recv().ok()
    }

    /// Blocking wait for the result matching `request_id`, discarding older ones.
    pub fn recv_result_for(&self, request_id: u64) -> Option<SearchOutcome> {
        loop {
            let outcome = self.result_rx.recv().ok()?;
            if outcome.request_id() == request_id {
                return Some(outcome);
            }
        }
    }

    /// Drain all pending progress messages, returning the latest.
    pub fn drain_progress(&self) -> Option<PipelineStatus> {
        let mut last = None;
        while let Ok(p) = self.progress_rx.try_recv() {
            last = Some(p);
        }
        last
    }
}

impl Drop for EmbedWorker {
    fn drop(&mut self) {
        self.req_tx.send(WorkerRequest::Shutdown).ok();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_search(
    embedder: &mut Embedder,
    idx: &Index,
    request_id: u64,
    query: &str,
    top_k: usize,
    threshold: f32,
    scope: &SearchScope,
    result_tx: &mpsc::Sender<SearchOutcome>,
) {
    let outcome = match embedder.embed(query) {
        Ok(emb) => match idx.search(&emb, top_k, threshold, scope) {
            Ok(results) => SearchOutcome::Results {
                request_id,
                results,
            },
            Err(e) => SearchOutcome::SearchError {
                request_id,
                message: format!("{e:#}"),
            },
        },
        Err(e) => SearchOutcome::EmbedError {
            request_id,
            message: format!("{e:#}"),
        },
    };
    result_tx.send(outcome).ok();
}

fn worker_loop(
    mut embedder: Embedder,
    idx: Index,
    mut indexer: StreamingIndexer,
    req_rx: mpsc::Receiver<WorkerRequest>,
    result_tx: mpsc::Sender<SearchOutcome>,
    progress_tx: mpsc::Sender<PipelineStatus>,
    scope: SearchScope,
) {
    loop {
        // Priority 1: handle all pending search requests
        loop {
            match req_rx.try_recv() {
                Ok(WorkerRequest::Search {
                    request_id,
                    query,
                    top_k,
                    threshold,
                }) => handle_search(
                    &mut embedder,
                    &idx,
                    request_id,
                    &query,
                    top_k,
                    threshold,
                    &scope,
                    &result_tx,
                ),
                Ok(WorkerRequest::Shutdown) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        // Priority 2: index a small batch
        if !indexer.indexing_done {
            match indexer.poll(&mut embedder, &idx) {
                Ok(_) => {
                    progress_tx.send(indexer.status()).ok();
                }
                Err(e) => {
                    tracing::error!("Indexing error: {e:#}");
                }
            }
        } else {
            // No more indexing, wait for requests
            match req_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(WorkerRequest::Search {
                    request_id,
                    query,
                    top_k,
                    threshold,
                }) => handle_search(
                    &mut embedder,
                    &idx,
                    request_id,
                    &query,
                    top_k,
                    threshold,
                    &scope,
                    &result_tx,
                ),
                Ok(WorkerRequest::Shutdown) => return,
                Err(_) => {}
            }
        }
    }
}

/// Process a batch of files: chunk, embed, and upsert into the index.
/// Returns the number of chunks indexed.
pub fn process_batch(
    embedder: &mut Embedder,
    idx: &Index,
    files_with_hashes: &[(WalkedFile, String)],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<usize> {
    let mut total_chunks = 0;

    for (file, content_hash) in files_with_hashes {
        let chunks = chunker::chunk_file(
            &file.rel_path,
            &file.content,
            chunk_size,
            chunk_overlap,
            embedder.tokenizer(),
        );
        if chunks.is_empty() {
            continue;
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let embeddings = embedder.embed_batch(&texts)?;

        let failed: Vec<bool> = embeddings
            .iter()
            .map(|emb| {
                let is_zero = emb.iter().all(|&v| v == 0.0);
                if is_zero {
                    tracing::warn!("Zero embedding for chunk in file: {}", file.rel_path);
                }
                is_zero
            })
            .collect();

        idx.upsert_file_with_explicit(
            &file.rel_path,
            content_hash,
            &chunks,
            &embeddings,
            &failed,
            file.explicit,
        )?;
        total_chunks += chunks.len();
    }

    Ok(total_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walker;
    use std::sync::mpsc;

    #[test]
    fn test_process_batch_indexes_files() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let files = vec![
            (
                WalkedFile {
                    rel_path: "a.rs".to_string(),
                    content: "fn main() {}".to_string(),
                    explicit: false,
                },
                "hash_a".to_string(),
            ),
            (
                WalkedFile {
                    rel_path: "b.rs".to_string(),
                    content: "fn helper() {}".to_string(),
                    explicit: false,
                },
                "hash_b".to_string(),
            ),
        ];

        let chunk_count = process_batch(&mut embedder, &idx, &files, 500, 100).unwrap();
        assert_eq!(chunk_count, 2);

        assert_eq!(idx.chunk_count().unwrap(), chunk_count);

        // Verify both files are searchable
        let query_emb = embedder.embed("fn main").unwrap();
        let results = idx
            .search(&query_emb, 10, 0.0, &SearchScope::default())
            .unwrap();
        let paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        assert!(
            paths.contains(&"a.rs"),
            "expected a.rs in results, got: {paths:?}"
        );
        assert!(
            paths.contains(&"b.rs"),
            "expected b.rs in results, got: {paths:?}"
        );
    }

    #[test]
    fn test_process_batch_empty() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let chunk_count = process_batch(&mut embedder, &idx, &[], 500, 100).unwrap();
        assert_eq!(chunk_count, 0);
        assert_eq!(idx.chunk_count().unwrap(), 0);
    }

    #[test]
    fn test_streaming_into_process_batch() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("one.txt"), "first file content here").unwrap();
        std::fs::write(dir.path().join("two.txt"), "second file content here").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = walker::WalkOptions::default();

        let (tx, rx) = mpsc::channel();
        let handle =
            std::thread::spawn(move || walker::walk_paths_streaming(&paths, &opts, tx).unwrap());

        let mut batch: Vec<(WalkedFile, String)> = Vec::new();
        for file in rx.iter() {
            let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
            batch.push((file, hash));
        }
        let walk_count = handle.join().unwrap();
        assert_eq!(walk_count, 2);
        assert_eq!(batch.len(), 2);

        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();
        let chunk_count = process_batch(&mut embedder, &idx, &batch, 500, 100).unwrap();
        assert_eq!(chunk_count, 2);
        assert_eq!(idx.chunk_count().unwrap(), 2);
    }

    #[test]
    fn test_incremental_streaming_batches() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        // Batch 1
        let batch1 = vec![(
            WalkedFile {
                rel_path: "a.rs".to_string(),
                content: "fn alpha() {}".to_string(),
                explicit: false,
            },
            "hash_a1".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch1, 500, 100).unwrap();
        assert_eq!(idx.chunk_count().unwrap(), 1);

        // Batch 2
        let batch2 = vec![(
            WalkedFile {
                rel_path: "b.rs".to_string(),
                content: "fn beta() {}".to_string(),
                explicit: false,
            },
            "hash_b1".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch2, 500, 100).unwrap();
        assert_eq!(idx.chunk_count().unwrap(), 2);

        // Batch 3: re-index a.rs with new content
        let batch3 = vec![(
            WalkedFile {
                rel_path: "a.rs".to_string(),
                content: "fn alpha_v2() { updated }".to_string(),
                explicit: false,
            },
            "hash_a2".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch3, 500, 100).unwrap();
        assert_eq!(idx.chunk_count().unwrap(), 2);

        // Verify a.rs has the updated content
        let query_emb = embedder.embed("alpha_v2 updated").unwrap();
        let results = idx
            .search(&query_emb, 1, 0.0, &SearchScope::default())
            .unwrap();
        assert_eq!(results[0].chunk.file_path, "a.rs");
        assert!(
            results[0].chunk.text.contains("alpha_v2"),
            "expected updated content, got: {}",
            results[0].chunk.text
        );
    }

    #[test]
    fn test_process_batch_multi_chunk_file() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let lines: Vec<String> = (0..50)
            .map(|i| format!("Line {} with some content to fill tokens", i))
            .collect();
        let content = lines.join("\n");
        let files = vec![(
            WalkedFile {
                rel_path: "big.rs".to_string(),
                content,
                explicit: false,
            },
            "hash_big".to_string(),
        )];

        let chunk_count = process_batch(&mut embedder, &idx, &files, 10, 2).unwrap();
        assert!(
            chunk_count > 1,
            "expected multiple chunks, got {chunk_count}"
        );
        assert_eq!(idx.chunk_count().unwrap(), chunk_count);
    }

    #[test]
    fn test_streaming_skips_already_indexed() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let content = "fn already_indexed() {}";
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let files = vec![(
            WalkedFile {
                rel_path: "cached.rs".to_string(),
                content: content.to_string(),
                explicit: false,
            },
            hash.clone(),
        )];
        process_batch(&mut embedder, &idx, &files, 500, 100).unwrap();

        let (tx, rx) = mpsc::sync_channel(32);
        tx.send(WalkedFile {
            rel_path: "cached.rs".to_string(),
            content: content.to_string(),
            explicit: false,
        })
        .unwrap();
        drop(tx);

        let mut batch: Vec<(WalkedFile, String)> = Vec::new();
        for file in rx.iter() {
            let file_hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
            let needs_index = match idx.get_file_hash(&file.rel_path) {
                Ok(Some(stored_hash)) => stored_hash != file_hash,
                _ => true,
            };
            if needs_index {
                batch.push((file, file_hash));
            }
        }
        assert!(
            batch.is_empty(),
            "file should have been skipped (hash match)"
        );
        assert_eq!(idx.chunk_count().unwrap(), 1);
    }

    // --- PipelineStatus tests ---

    #[test]
    fn test_status_indexing_without_total_when_walker_active() {
        let (tx, rx) = mpsc::sync_channel(32);
        tx.send(WalkedFile {
            rel_path: "a.rs".to_string(),
            content: "fn a() {}".to_string(),
            explicit: false,
        })
        .unwrap();
        // Don't drop tx — walker appears still running

        let progress = Arc::new(walker::StreamProgress::new());
        // Simulate walker having sent 1 file
        progress.snapshot(); // just to confirm it works

        let mut indexer =
            StreamingIndexer::new(rx, 500, 100, 32, std::path::Path::new(""), Some(progress));

        let status = indexer.status();
        assert_eq!(
            status,
            PipelineStatus::Indexing {
                indexed: 0,
                total: None,
                chunks: 0,
            }
        );

        drop(tx);
        // Drain to mark indexing_done
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();
        indexer
            .drain_all(&mut embedder, &idx, |_| Ok(true))
            .unwrap();

        match indexer.status() {
            PipelineStatus::Ready { files, chunks } => {
                assert_eq!(files, 1);
                assert_eq!(chunks, 1);
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn test_status_indexing_with_total_when_walker_done() {
        let (tx, rx) = mpsc::sync_channel(32);
        let progress = Arc::new(walker::StreamProgress::new());

        // Send files through the progress tracker
        for i in 0..3 {
            tx.send(WalkedFile {
                rel_path: format!("f{i}.rs"),
                content: format!("fn f{i}() {{}}"),
                explicit: false,
            })
            .unwrap();
        }
        // Manually update progress to simulate walker sending
        for _ in 0..3 {
            progress.snapshot(); // read
        }
        progress.mark_done();
        drop(tx);

        let indexer =
            StreamingIndexer::new(rx, 500, 100, 32, std::path::Path::new(""), Some(progress));

        match indexer.status() {
            PipelineStatus::Indexing { total, .. } => {
                // Walker is done but indexer hasn't consumed files yet.
                // total should be Some because walk_done is true.
                assert!(total.is_some(), "expected total when walker is done");
            }
            other => panic!("expected Indexing, got {other:?}"),
        }
    }

    #[test]
    fn test_pipeline_status_serializes_as_tagged_json() {
        let indexing = PipelineStatus::Indexing {
            indexed: 42,
            total: Some(380),
            chunks: 85,
        };
        let json = serde_json::to_value(&indexing).unwrap();
        assert_eq!(json["status"], "indexing");
        assert_eq!(json["indexed"], 42);
        assert_eq!(json["total"], 380);
        assert_eq!(json["chunks"], 85);

        let scanning = PipelineStatus::Indexing {
            indexed: 10,
            total: None,
            chunks: 20,
        };
        let json = serde_json::to_value(&scanning).unwrap();
        assert_eq!(json["status"], "indexing");
        assert_eq!(json["total"], serde_json::Value::Null);

        let ready = PipelineStatus::Ready {
            files: 100,
            chunks: 200,
        };
        let json = serde_json::to_value(&ready).unwrap();
        assert_eq!(json["status"], "ready");
        assert_eq!(json["files"], 100);
        assert_eq!(json["chunks"], 200);
    }

    // --- EmbedWorker tests ---

    /// Helper: create an EmbedWorker with pre-indexed data and no streaming files.
    fn worker_with_data(texts: &[&str]) -> EmbedWorker {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        for (i, text) in texts.iter().enumerate() {
            let emb = embedder.embed(text).unwrap();
            let chunk = crate::types::Chunk {
                file_path: format!("file{i}.rs"),
                text: text.to_string(),
                start_line: 1,
                end_line: 1,
            };
            idx.upsert_file(
                &format!("file{i}.rs"),
                &format!("hash{i}"),
                &[chunk],
                &[emb],
                &[false],
            )
            .unwrap();
        }

        let (tx, rx) = mpsc::sync_channel(0);
        drop(tx); // no files to index
        let indexer = StreamingIndexer::new(rx, 500, 100, 1, std::path::Path::new(""), None);
        EmbedWorker::spawn(embedder, idx, indexer, SearchScope::default())
    }

    #[test]
    fn test_worker_search_after_indexing_done() {
        let worker = worker_with_data(&["error handling in rust", "memory management"]);

        let request_id = worker.search("error handling", 5, 0.0);
        let outcome = worker.recv_result_for(request_id).unwrap();
        match outcome {
            SearchOutcome::Results { results, .. } => {
                assert!(!results.is_empty(), "expected search results");
                assert!(results[0].score > 0.0);
            }
            SearchOutcome::EmbedError { message, .. } => {
                panic!("unexpected embed error: {message}")
            }
            SearchOutcome::SearchError { message, .. } => {
                panic!("unexpected search error: {message}")
            }
        }
    }

    #[test]
    fn test_worker_search_during_indexing() {
        let mut embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        // Pre-index one file so search has something to find
        let emb = embedder.embed("existing content").unwrap();
        let chunk = crate::types::Chunk {
            file_path: "existing.rs".to_string(),
            text: "existing content".to_string(),
            start_line: 1,
            end_line: 1,
        };
        idx.upsert_file("existing.rs", "hash0", &[chunk], &[emb], &[false])
            .unwrap();

        // Create a channel with files to index (keeps worker busy)
        let (tx, rx) = mpsc::sync_channel(32);
        for i in 0..10 {
            tx.send(WalkedFile {
                rel_path: format!("new{i}.rs"),
                content: format!("fn new_function_{i}() {{ }}"),
                explicit: false,
            })
            .unwrap();
        }
        // Don't drop tx yet — worker thinks indexing is still in progress

        let indexer = StreamingIndexer::new(rx, 500, 100, 1, std::path::Path::new(""), None);
        let worker = EmbedWorker::spawn(embedder, idx, indexer, SearchScope::default());

        // Search should work even while indexing is happening
        let request_id = worker.search("existing content", 5, 0.0);
        let outcome = worker.recv_result_for(request_id).unwrap();
        match outcome {
            SearchOutcome::Results { results, .. } => {
                assert!(!results.is_empty(), "search should work during indexing");
                assert_eq!(results[0].chunk.file_path, "existing.rs");
            }
            SearchOutcome::EmbedError { message, .. } => {
                panic!("unexpected embed error: {message}")
            }
            SearchOutcome::SearchError { message, .. } => {
                panic!("unexpected search error: {message}")
            }
        }

        drop(tx); // let indexing finish
        drop(worker);
    }

    #[test]
    fn test_worker_progress_reporting() {
        let embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let (tx, rx) = mpsc::sync_channel(32);
        for i in 0..3 {
            tx.send(WalkedFile {
                rel_path: format!("f{i}.rs"),
                content: format!("fn func_{i}() {{ }}"),
                explicit: false,
            })
            .unwrap();
        }
        drop(tx);

        let indexer = StreamingIndexer::new(rx, 500, 100, 2, std::path::Path::new(""), None);
        let worker = EmbedWorker::spawn(embedder, idx, indexer, SearchScope::default());

        // Wait for indexing to complete (50 × 50ms = 2.5s max)
        let mut final_status = None;
        for _ in 0..50 {
            if let Some(s) = worker.drain_progress() {
                if matches!(s, PipelineStatus::Ready { .. }) {
                    final_status = Some(s);
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let status = final_status.expect("should have received Ready status");
        assert_eq!(
            status,
            PipelineStatus::Ready {
                files: 3,
                chunks: 3
            }
        );
    }

    #[test]
    fn test_worker_shutdown_via_drop() {
        let worker = worker_with_data(&["test content"]);

        // Verify worker is alive by doing a search
        let request_id = worker.search("test", 1, 0.0);
        match worker.recv_result_for(request_id) {
            Some(SearchOutcome::Results { .. }) => {}
            other => panic!("expected Results, got: {other:?}"),
        }

        // Drop should shut down cleanly without hanging
        drop(worker);
    }

    #[test]
    fn test_worker_reports_search_errors() {
        let embedder = Embedder::new_local().unwrap();
        let idx = Index::open_in_memory().unwrap();
        idx.rebuild_for_config(&crate::types::IndexConfig {
            model_name: "remote-like".to_string(),
            embedding_dim: 1024,
            chunk_size: 500,
            chunk_overlap: 100,
        })
        .unwrap();

        let (tx, rx) = mpsc::sync_channel(0);
        drop(tx);
        let indexer = StreamingIndexer::new(rx, 500, 100, 1, std::path::Path::new(""), None);
        let worker = EmbedWorker::spawn(embedder, idx, indexer, SearchScope::default());

        let request_id = worker.search("dimension mismatch", 5, 0.0);
        match worker.recv_result_for(request_id) {
            Some(SearchOutcome::SearchError { message, .. }) => {
                assert!(
                    message.contains("dimension")
                        || message.contains("1024")
                        || message.contains("384"),
                    "expected dimension mismatch error, got: {message}"
                );
            }
            other => panic!("expected SearchError, got: {other:?}"),
        }
    }
}
