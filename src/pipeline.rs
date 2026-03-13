use anyhow::Result;
use std::path::Path;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use crate::chunker;
use crate::embedder::Embedder;
use crate::index::Index;
use crate::paths;
use crate::types::Chunk;
use crate::walker::WalkedFile;

/// Max files to drain from the channel per iteration in streaming modes.
pub const STREAMING_BATCH_SIZE: usize = 4;

/// Manages incremental indexing from a streaming channel.
pub struct StreamingIndexer {
    rx: Receiver<WalkedFile>,
    pub indexing_done: bool,
    pub indexed_count: usize,
    /// All file paths seen (for stale removal).
    pub all_paths: Vec<String>,
    last_reload: Instant,
    chunk_size: usize,
    chunk_overlap: usize,
    batch_size: usize,
    cwd_suffix: Box<Path>,
}

impl StreamingIndexer {
    pub fn new(
        rx: Receiver<WalkedFile>,
        chunk_size: usize,
        chunk_overlap: usize,
        batch_size: usize,
        cwd_suffix: &Path,
    ) -> Self {
        Self {
            rx,
            indexing_done: false,
            indexed_count: 0,
            all_paths: Vec::new(),
            last_reload: Instant::now() - Duration::from_secs(10),
            chunk_size,
            chunk_overlap,
            batch_size,
            cwd_suffix: cwd_suffix.into(),
        }
    }

    /// Receive one file from the channel, hash-check it, and return it if it needs indexing.
    /// Returns `None` if the file is up-to-date or the channel is empty/closed.
    fn recv_one(&mut self, idx: &Index, blocking: bool) -> Option<(WalkedFile, String)> {
        let file = if blocking {
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

        let mut file = file;
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

    /// Drain up to `batch_size` files from the channel (non-blocking), process them,
    /// and reload the index if enough time has passed.
    /// Returns `true` if the index data was reloaded (caller should re-search).
    pub fn poll(
        &mut self,
        embedder: &mut Embedder,
        idx: &Index,
        chunks: &mut Vec<Chunk>,
        embedding_matrix: &mut ndarray::Array2<f32>,
    ) -> Result<bool> {
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

        let mut reloaded = false;

        if !batch.is_empty() {
            self.indexed_count += batch.len();
            process_batch(embedder, idx, &batch, self.chunk_size, self.chunk_overlap)?;
            if self.last_reload.elapsed() >= Duration::from_secs(2) {
                let (new_chunks, new_matrix) = idx.load_all()?;
                *chunks = new_chunks;
                *embedding_matrix = new_matrix;
                self.last_reload = Instant::now();
                reloaded = true;
            }
        }

        // Final reload when indexing completes
        if self.indexing_done && self.indexed_count > 0 && !reloaded {
            let (new_chunks, new_matrix) = idx.load_all()?;
            *chunks = new_chunks;
            *embedding_matrix = new_matrix;
            reloaded = true;
        }

        Ok(reloaded)
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
        F: FnMut(usize) -> Result<bool>, // (indexed_so_far) -> continue?
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
                process_batch(embedder, idx, &batch, self.chunk_size, self.chunk_overlap)?;

                if !on_batch(self.indexed_count)? {
                    return Ok(self.indexed_count);
                }
            }
        }

        Ok(self.indexed_count)
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
    let mut all_chunks = Vec::new();
    let mut chunk_file_info: Vec<(String, String)> = Vec::new();

    for (file, content_hash) in files_with_hashes {
        let file_chunks = chunker::chunk_file(
            &file.rel_path,
            &file.content,
            chunk_size,
            chunk_overlap,
            embedder.tokenizer(),
        );

        for _ in &file_chunks {
            chunk_file_info.push((file.rel_path.clone(), content_hash.clone()));
        }
        all_chunks.extend(file_chunks);
    }

    if all_chunks.is_empty() {
        return Ok(0);
    }

    // Embed all chunks in sub-batches
    let texts: Vec<&str> = all_chunks.iter().map(|c| c.text.as_str()).collect();
    let embed_batch_size = 64;
    let mut all_embeddings = Vec::new();
    for text_batch in texts.chunks(embed_batch_size) {
        let embeddings = embedder.embed_batch(text_batch)?;
        all_embeddings.extend(embeddings);
    }

    // Group chunks by file and insert into index
    let mut current_file: Option<String> = None;
    let mut file_chunks = Vec::new();
    let mut file_embeddings = Vec::new();
    let mut file_hash = String::new();

    for (i, chunk) in all_chunks.iter().enumerate() {
        let (ref path, ref hash) = chunk_file_info[i];

        if current_file.as_ref() != Some(path) {
            if let Some(ref prev_path) = current_file {
                idx.upsert_file(prev_path, &file_hash, &file_chunks, &file_embeddings)?;
            }
            current_file = Some(path.clone());
            file_hash = hash.clone();
            file_chunks = Vec::new();
            file_embeddings = Vec::new();
        }

        file_chunks.push(chunk.clone());
        file_embeddings.push(all_embeddings[i].clone());
    }

    if let Some(ref prev_path) = current_file {
        idx.upsert_file(prev_path, &file_hash, &file_chunks, &file_embeddings)?;
    }

    Ok(all_chunks.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walker;
    use std::sync::mpsc;

    #[test]
    fn test_process_batch_indexes_files() {
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let files = vec![
            (
                WalkedFile {
                    rel_path: "a.rs".to_string(),
                    content: "fn main() {}".to_string(),
                },
                "hash_a".to_string(),
            ),
            (
                WalkedFile {
                    rel_path: "b.rs".to_string(),
                    content: "fn helper() {}".to_string(),
                },
                "hash_b".to_string(),
            ),
        ];

        let chunk_count = process_batch(&mut embedder, &idx, &files, 500, 100).unwrap();
        assert_eq!(chunk_count, 2); // two small files → one chunk each

        let (chunks, matrix) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), chunk_count);
        assert_eq!(matrix.nrows(), chunk_count);

        let paths: Vec<&str> = chunks.iter().map(|c| c.file_path.as_str()).collect();
        assert!(paths.contains(&"a.rs"));
        assert!(paths.contains(&"b.rs"));
    }

    #[test]
    fn test_process_batch_empty() {
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();

        let chunk_count = process_batch(&mut embedder, &idx, &[], 500, 100).unwrap();
        assert_eq!(chunk_count, 0);

        let (chunks, _) = idx.load_all().unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_streaming_into_process_batch() {
        // Walk real files via streaming, then process them through the pipeline
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("one.txt"), "first file content here").unwrap();
        std::fs::write(dir.path().join("two.txt"), "second file content here").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = walker::WalkOptions {
            file_types: None,
            file_types_not: None,
            globs: None,
            hidden: false,
            follow: false,
            no_ignore: false,
            max_depth: None,
        };

        let (tx, rx) = mpsc::sync_channel(32);
        let handle =
            std::thread::spawn(move || walker::walk_paths_streaming(&paths, &opts, tx).unwrap());

        // Collect files with hashes, simulating the main.rs receive loop
        let mut batch: Vec<(WalkedFile, String)> = Vec::new();
        for file in rx.iter() {
            let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
            batch.push((file, hash));
        }
        let walk_count = handle.join().unwrap();
        assert_eq!(walk_count, 2);
        assert_eq!(batch.len(), 2);

        // Process through pipeline
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();
        let chunk_count = process_batch(&mut embedder, &idx, &batch, 500, 100).unwrap();
        assert_eq!(chunk_count, 2); // small files = 1 chunk each

        // Verify index has searchable data
        let (chunks, matrix) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(matrix.nrows(), 2);
    }

    #[test]
    fn test_incremental_streaming_batches() {
        // Simulate the streaming pipeline processing files in multiple batches
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();

        // Batch 1
        let batch1 = vec![(
            WalkedFile {
                rel_path: "a.rs".to_string(),
                content: "fn alpha() {}".to_string(),
            },
            "hash_a1".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch1, 500, 100).unwrap();

        let (chunks, _) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), 1);

        // Batch 2
        let batch2 = vec![(
            WalkedFile {
                rel_path: "b.rs".to_string(),
                content: "fn beta() {}".to_string(),
            },
            "hash_b1".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch2, 500, 100).unwrap();

        let (chunks, _) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), 2);

        // Batch 3: re-index a.rs with new content
        let batch3 = vec![(
            WalkedFile {
                rel_path: "a.rs".to_string(),
                content: "fn alpha_v2() { updated }".to_string(),
            },
            "hash_a2".to_string(),
        )];
        process_batch(&mut embedder, &idx, &batch3, 500, 100).unwrap();

        let (chunks, _) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), 2); // still 2 files, a.rs was replaced
        let a_chunk = chunks.iter().find(|c| c.file_path == "a.rs").unwrap();
        assert!(a_chunk.text.contains("alpha_v2"));
    }

    #[test]
    fn test_process_batch_multi_chunk_file() {
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();

        // Generate content large enough for multiple chunks at chunk_size=10
        let lines: Vec<String> = (0..50)
            .map(|i| format!("Line {} with some content to fill tokens", i))
            .collect();
        let content = lines.join("\n");
        let files = vec![(
            WalkedFile {
                rel_path: "big.rs".to_string(),
                content,
            },
            "hash_big".to_string(),
        )];

        let chunk_count = process_batch(&mut embedder, &idx, &files, 10, 2).unwrap();
        assert!(
            chunk_count > 1,
            "expected multiple chunks, got {chunk_count}"
        );

        let (chunks, matrix) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), chunk_count);
        assert_eq!(matrix.nrows(), chunk_count);
        assert!(chunks.iter().all(|c| c.file_path == "big.rs"));
    }

    #[test]
    fn test_streaming_skips_already_indexed() {
        let mut embedder = Embedder::new().unwrap();
        let idx = Index::open_in_memory().unwrap();

        // Pre-index a file
        let content = "fn already_indexed() {}";
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        let files = vec![(
            WalkedFile {
                rel_path: "cached.rs".to_string(),
                content: content.to_string(),
            },
            hash.clone(),
        )];
        process_batch(&mut embedder, &idx, &files, 500, 100).unwrap();

        // Now simulate streaming: send the same file through a channel
        let (tx, rx) = mpsc::sync_channel(32);
        tx.send(WalkedFile {
            rel_path: "cached.rs".to_string(),
            content: content.to_string(),
        })
        .unwrap();
        drop(tx);

        // Drain and check — hash matches, so it should be skipped
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

        // Index still has original data
        let (chunks, _) = idx.load_all().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].file_path, "cached.rs");
    }
}
