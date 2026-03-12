use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

use crate::embedder::EMBEDDING_DIM;
use crate::types::{Chunk, IndexConfig};

pub struct Index {
    conn: Connection,
}

impl Index {
    /// Open or create the index database at `.vecgrep/index.db` under the given root.
    pub fn open(root: &Path) -> Result<Self> {
        let index_dir = root.join(".vecgrep");
        std::fs::create_dir_all(&index_dir).context("Failed to create .vecgrep directory")?;

        // Auto-add .vecgrep/ to .gitignore
        let gitignore_path = root.join(".gitignore");
        ensure_gitignore_entry(&gitignore_path);

        let db_path = index_dir.join("index.db");
        let conn = Connection::open(&db_path).context("Failed to open index database")?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        let index = Self { conn };
        index.create_tables()?;
        Ok(index)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("Failed to open in-memory database")?;
        let index = Self { conn };
        index.create_tables()?;
        Ok(index)
    }

    fn create_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);",
        )?;
        Ok(())
    }

    /// Check if the index config matches. If not, clear and return false.
    pub fn check_config(&self, config: &IndexConfig) -> Result<bool> {
        let stored = self.get_meta("config")?;
        let config_json = serde_json::to_string(config)?;

        match stored {
            Some(s) if s == config_json => Ok(true),
            _ => {
                // Config changed, need full rebuild
                Ok(false)
            }
        }
    }

    /// Store the current config.
    pub fn set_config(&self, config: &IndexConfig) -> Result<()> {
        let config_json = serde_json::to_string(config)?;
        self.set_meta("config", &config_json)
    }

    /// Clear all data (for reindex or config change).
    pub fn clear(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;")?;
        Ok(())
    }

    /// Get the stored content hash for a file path.
    pub fn get_file_hash(&self, path: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT content_hash FROM files WHERE path = ?1")?;
        let hash = stmt
            .query_row(params![path], |row| row.get::<_, String>(0))
            .ok();
        Ok(hash)
    }

    /// Insert or update a file and its chunks.
    pub fn upsert_file(
        &self,
        path: &str,
        content_hash: &str,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        // Delete existing data for this file
        if let Ok(file_id) = self.get_file_id(path) {
            self.conn
                .execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
            self.conn
                .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        }

        // Insert file record
        self.conn.execute(
            "INSERT INTO files (path, content_hash) VALUES (?1, ?2)",
            params![path, content_hash],
        )?;
        let file_id = self.conn.last_insert_rowid();

        // Insert chunks with embeddings
        let mut stmt = self.conn.prepare(
            "INSERT INTO chunks (file_id, text, start_line, end_line, embedding) VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let embedding_blob = embedding_to_blob(embedding);
            stmt.execute(params![
                file_id,
                chunk.text,
                chunk.start_line as i64,
                chunk.end_line as i64,
                embedding_blob,
            ])?;
        }

        Ok(())
    }

    /// Remove files that are no longer present on disk.
    pub fn remove_stale_files(&self, current_paths: &[String]) -> Result<usize> {
        let stored_paths = self.all_file_paths()?;
        let current_set: std::collections::HashSet<&str> =
            current_paths.iter().map(|s| s.as_str()).collect();

        let mut removed = 0;
        for path in &stored_paths {
            if !current_set.contains(path.as_str()) {
                if let Ok(file_id) = self.get_file_id(path) {
                    self.conn
                        .execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
                    self.conn
                        .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }

    /// Load all chunks and embeddings from the index.
    pub fn load_all(&self) -> Result<(Vec<Chunk>, ndarray::Array2<f32>)> {
        let mut stmt = self.conn.prepare(
            "SELECT c.text, c.start_line, c.end_line, c.embedding, f.path
             FROM chunks c JOIN files f ON c.file_id = f.id",
        )?;

        let rows: Vec<(String, i64, i64, Vec<u8>, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let n = rows.len();
        let mut chunks = Vec::with_capacity(n);
        let mut embeddings_flat = Vec::with_capacity(n * EMBEDDING_DIM);

        for (text, start_line, end_line, blob, path) in rows {
            chunks.push(Chunk {
                file_path: path,
                text,
                start_line: start_line as usize,
                end_line: end_line as usize,
            });
            let embedding = blob_to_embedding(&blob);
            embeddings_flat.extend_from_slice(&embedding);
        }

        let embedding_matrix = if n > 0 {
            ndarray::Array2::from_shape_vec((n, EMBEDDING_DIM), embeddings_flat)?
        } else {
            ndarray::Array2::zeros((0, EMBEDDING_DIM))
        };

        Ok((chunks, embedding_matrix))
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        let file_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;

        let db_path = self.db_path()?;
        let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        Ok(IndexStats {
            file_count: file_count as usize,
            chunk_count: chunk_count as usize,
            db_size_bytes: db_size,
        })
    }

    fn db_path(&self) -> Result<PathBuf> {
        let path: String = self
            .conn
            .query_row("PRAGMA database_list", [], |r| r.get(2))?;
        Ok(PathBuf::from(path))
    }

    fn get_file_id(&self, path: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT id FROM files WHERE path = ?1", params![path], |r| {
                r.get(0)
            })?)
    }

    fn all_file_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(paths)
    }

    fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let val = stmt
            .query_row(params![key], |row| row.get::<_, String>(0))
            .ok();
        Ok(val)
    }

    fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }
}

pub struct IndexStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub db_size_bytes: u64,
}

/// Convert f32 embedding to bytes for SQLite BLOB storage.
pub(crate) fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert bytes from SQLite BLOB to f32 embedding.
pub(crate) fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub(crate) fn ensure_gitignore_entry(gitignore_path: &Path) {
    let entry = ".vecgrep/";
    let content = std::fs::read_to_string(gitignore_path).unwrap_or_default();
    if !content.lines().any(|line| line.trim() == entry) {
        let mut new_content = content;
        if !new_content.is_empty() && !new_content.ends_with('\n') {
            new_content.push('\n');
        }
        new_content.push_str(entry);
        new_content.push('\n');
        let _ = std::fs::write(gitignore_path, new_content);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, IndexConfig};
    use tempfile::TempDir;

    // --- Unit tests for blob conversion ---

    #[test]
    fn test_embedding_blob_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14, f32::MIN, f32::MAX];
        let blob = embedding_to_blob(&original);
        let recovered = blob_to_embedding(&blob);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_embedding_blob_empty() {
        let original: Vec<f32> = vec![];
        let blob = embedding_to_blob(&original);
        assert!(blob.is_empty());
        let recovered = blob_to_embedding(&blob);
        assert!(recovered.is_empty());
    }

    // --- Unit tests for ensure_gitignore_entry ---

    #[test]
    fn test_ensure_gitignore_new_file() {
        let dir = TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");
        ensure_gitignore_entry(&gitignore);
        let content = std::fs::read_to_string(&gitignore).unwrap();
        assert!(content.contains(".vecgrep/"));
    }

    #[test]
    fn test_ensure_gitignore_already_present() {
        let dir = TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");
        std::fs::write(&gitignore, ".vecgrep/\n").unwrap();
        ensure_gitignore_entry(&gitignore);
        let content = std::fs::read_to_string(&gitignore).unwrap();
        // Should appear exactly once
        assert_eq!(content.matches(".vecgrep/").count(), 1);
    }

    #[test]
    fn test_ensure_gitignore_appends() {
        let dir = TempDir::new().unwrap();
        let gitignore = dir.path().join(".gitignore");
        std::fs::write(&gitignore, "target/\nnode_modules/").unwrap();
        ensure_gitignore_entry(&gitignore);
        let content = std::fs::read_to_string(&gitignore).unwrap();
        assert!(content.contains("target/"));
        assert!(content.contains(".vecgrep/"));
        // Should have added a newline before the entry since the file didn't end with one
        assert!(content.contains("node_modules/\n.vecgrep/"));
    }

    // --- Integration tests using in-memory DB ---

    #[test]
    fn test_open_and_create_tables() {
        let index = Index::open_in_memory().unwrap();
        // Verify tables exist by querying them
        let count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM meta", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_config_roundtrip() {
        let index = Index::open_in_memory().unwrap();
        let config = IndexConfig {
            model_name: "test-model".to_string(),
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config).unwrap();
        assert!(index.check_config(&config).unwrap());
    }

    #[test]
    fn test_config_mismatch() {
        let index = Index::open_in_memory().unwrap();
        let config1 = IndexConfig {
            model_name: "model-a".to_string(),
            chunk_size: 500,
            chunk_overlap: 100,
        };
        let config2 = IndexConfig {
            model_name: "model-b".to_string(),
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config1).unwrap();
        assert!(!index.check_config(&config2).unwrap());
    }

    fn make_test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * seed).sin()).collect()
    }

    #[test]
    fn test_upsert_and_load() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let chunks = vec![
            Chunk {
                file_path: "test.rs".to_string(),
                text: "fn main() {}".to_string(),
                start_line: 1,
                end_line: 1,
            },
            Chunk {
                file_path: "test.rs".to_string(),
                text: "fn helper() {}".to_string(),
                start_line: 3,
                end_line: 3,
            },
        ];
        let embeddings = vec![make_test_embedding(dim, 1.0), make_test_embedding(dim, 2.0)];

        index
            .upsert_file("test.rs", "abc123", &chunks, &embeddings)
            .unwrap();

        let (loaded_chunks, loaded_matrix) = index.load_all().unwrap();
        assert_eq!(loaded_chunks.len(), 2);
        assert_eq!(loaded_matrix.nrows(), 2);
        assert_eq!(loaded_matrix.ncols(), dim);
        assert_eq!(loaded_chunks[0].text, "fn main() {}");
        assert_eq!(loaded_chunks[1].text, "fn helper() {}");
    }

    #[test]
    fn test_upsert_replaces() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let chunks_v1 = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "version 1".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb_v1 = vec![make_test_embedding(dim, 1.0)];
        index
            .upsert_file("a.rs", "hash1", &chunks_v1, &emb_v1)
            .unwrap();

        let chunks_v2 = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "version 2".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb_v2 = vec![make_test_embedding(dim, 2.0)];
        index
            .upsert_file("a.rs", "hash2", &chunks_v2, &emb_v2)
            .unwrap();

        let (loaded_chunks, _) = index.load_all().unwrap();
        assert_eq!(loaded_chunks.len(), 1);
        assert_eq!(loaded_chunks[0].text, "version 2");
    }

    #[test]
    fn test_get_file_hash() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let chunks = vec![Chunk {
            file_path: "test.rs".to_string(),
            text: "hello".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let embeddings = vec![make_test_embedding(dim, 1.0)];
        index
            .upsert_file("test.rs", "myhash", &chunks, &embeddings)
            .unwrap();

        assert_eq!(
            index.get_file_hash("test.rs").unwrap(),
            Some("myhash".to_string())
        );
        assert_eq!(index.get_file_hash("nonexistent.rs").unwrap(), None);
    }

    #[test]
    fn test_remove_stale_files() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        for name in &["a.rs", "b.rs", "c.rs"] {
            let chunks = vec![Chunk {
                file_path: name.to_string(),
                text: format!("content of {}", name),
                start_line: 1,
                end_line: 1,
            }];
            let emb = vec![make_test_embedding(dim, 1.0)];
            index.upsert_file(name, "hash", &chunks, &emb).unwrap();
        }

        // Only a.rs and c.rs remain on disk
        let current = vec!["a.rs".to_string(), "c.rs".to_string()];
        let removed = index.remove_stale_files(&current).unwrap();
        assert_eq!(removed, 1);

        let (loaded, _) = index.load_all().unwrap();
        assert_eq!(loaded.len(), 2);
        let paths: Vec<&str> = loaded.iter().map(|c| c.file_path.as_str()).collect();
        assert!(paths.contains(&"a.rs"));
        assert!(paths.contains(&"c.rs"));
        assert!(!paths.contains(&"b.rs"));
    }

    #[test]
    fn test_clear() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let chunks = vec![Chunk {
            file_path: "test.rs".to_string(),
            text: "hello".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let embeddings = vec![make_test_embedding(dim, 1.0)];
        index
            .upsert_file("test.rs", "hash", &chunks, &embeddings)
            .unwrap();
        index
            .set_config(&IndexConfig {
                model_name: "m".to_string(),
                chunk_size: 1,
                chunk_overlap: 0,
            })
            .unwrap();

        index.clear().unwrap();

        let (loaded, _) = index.load_all().unwrap();
        assert!(loaded.is_empty());
        assert_eq!(index.get_file_hash("test.rs").unwrap(), None);
    }

    #[test]
    fn test_stats() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        // stats() calls db_path() which fails for in-memory DBs, so we test the counts directly
        let file_count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        let chunk_count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(file_count, 0);
        assert_eq!(chunk_count, 0);

        // Add some data
        let chunks = vec![
            Chunk {
                file_path: "a.rs".to_string(),
                text: "one".to_string(),
                start_line: 1,
                end_line: 1,
            },
            Chunk {
                file_path: "a.rs".to_string(),
                text: "two".to_string(),
                start_line: 2,
                end_line: 2,
            },
        ];
        let emb = vec![make_test_embedding(dim, 1.0), make_test_embedding(dim, 2.0)];
        index.upsert_file("a.rs", "hash", &chunks, &emb).unwrap();

        let file_count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        let chunk_count: i64 = index
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(file_count, 1);
        assert_eq!(chunk_count, 2);
    }
}
