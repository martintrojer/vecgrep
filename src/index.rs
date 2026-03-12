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
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert bytes from SQLite BLOB to f32 embedding.
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn ensure_gitignore_entry(gitignore_path: &Path) {
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
