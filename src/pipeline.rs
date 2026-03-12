use anyhow::Result;

use crate::chunker;
use crate::embedder::Embedder;
use crate::index::Index;
use crate::walker::WalkedFile;

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
        assert!(chunk_count >= 2);

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
}
