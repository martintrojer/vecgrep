use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::Once;
use zerocopy::IntoBytes;

use crate::embedder::EMBEDDING_DIM;
use crate::types::{Chunk, IndexConfig, SearchResult};

static SQLITE_VEC_INIT: Once = Once::new();

fn vec_table_ddl(dim: usize) -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
         chunk_id integer primary key, \
         embedding float[{dim}] distance_metric=cosine)"
    )
}

fn init_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub struct Index {
    conn: Connection,
}

impl Index {
    /// Open or create the index database at `.vecgrep/index.db` under the given root.
    pub fn open(root: &Path) -> Result<Self> {
        init_sqlite_vec();

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
        init_sqlite_vec();

        let conn = Connection::open_in_memory().context("Failed to open in-memory database")?;
        let index = Self { conn };
        index.create_tables()?;
        Ok(index)
    }

    fn create_tables(&self) -> Result<()> {
        // Migrate from old schema: if chunks table has an embedding column, drop data tables
        let has_old_schema = self
            .conn
            .prepare("SELECT embedding FROM chunks LIMIT 0")
            .is_ok();
        if has_old_schema {
            self.conn.execute_batch(
                "DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS files; DELETE FROM meta;",
            )?;
        }

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
                file_id INTEGER NOT NULL REFERENCES files(id),
                text TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);",
        )?;

        // Create vec_chunks with default dimension (set_config will recreate if needed)
        self.conn.execute(&vec_table_ddl(EMBEDDING_DIM), [])?;

        Ok(())
    }

    /// Check if the index config matches. If not, return false.
    pub fn check_config(&self, config: &IndexConfig) -> Result<bool> {
        let stored = self.get_meta("config")?;
        let config_json = serde_json::to_string(config)?;

        match stored {
            Some(s) if s == config_json => Ok(true),
            _ => Ok(false),
        }
    }

    /// Store the current config and ensure vec_chunks table has correct dimension.
    pub fn set_config(&self, config: &IndexConfig) -> Result<()> {
        let config_json = serde_json::to_string(config)?;
        self.set_meta("config", &config_json)?;

        // Ensure vec_chunks exists with the correct embedding dimension
        self.conn
            .execute(&vec_table_ddl(config.embedding_dim), [])?;

        Ok(())
    }

    /// Clear all data (for reindex or config change).
    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch(
            "DROP TABLE IF EXISTS vec_chunks; \
             DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;",
        )?;
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
        self.conn.execute("BEGIN", [])?;

        // Delete existing data for this file
        if let Ok(file_id) = self.get_file_id(path) {
            self.delete_file_by_id(file_id)?;
        }

        // Insert file record
        self.conn.execute(
            "INSERT INTO files (path, content_hash) VALUES (?1, ?2)",
            params![path, content_hash],
        )?;
        let file_id = self.conn.last_insert_rowid();

        // Insert chunks and their vector embeddings
        let mut chunk_stmt = self.conn.prepare(
            "INSERT INTO chunks (file_id, text, start_line, end_line) VALUES (?1, ?2, ?3, ?4)",
        )?;
        let mut vec_stmt = self
            .conn
            .prepare("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)")?;

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            chunk_stmt.execute(params![
                file_id,
                chunk.text,
                chunk.start_line as i64,
                chunk.end_line as i64,
            ])?;
            let chunk_id = self.conn.last_insert_rowid();
            vec_stmt.execute(params![chunk_id, embedding.as_slice().as_bytes()])?;
        }

        self.conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Search for chunks most similar to the query embedding.
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let mut stmt = self.conn.prepare(
            "SELECT c.text, c.start_line, c.end_line, f.path, v.distance \
             FROM vec_chunks v \
             JOIN chunks c ON c.id = v.chunk_id \
             JOIN files f ON f.id = c.file_id \
             WHERE v.embedding MATCH ?1 \
               AND k = ?2 \
             ORDER BY v.distance",
        )?;

        let results = stmt
            .query_map(params![query_embedding.as_bytes(), top_k as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, f64>(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .filter_map(|(text, start_line, end_line, path, distance)| {
                let score = 1.0 - distance as f32;
                if score >= threshold {
                    Some(SearchResult {
                        chunk: Chunk {
                            file_path: path,
                            text,
                            start_line: start_line as usize,
                            end_line: end_line as usize,
                        },
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(search_results)
    }

    /// Get the number of chunks in the index.
    pub fn chunk_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(count as usize)
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
                    self.delete_file_by_id(file_id)?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }

    /// Remove stale files, but only within the given path prefix.
    /// Files outside the prefix are left untouched.
    pub fn remove_stale_files_under(
        &self,
        current_paths: &[String],
        prefix: &str,
    ) -> Result<usize> {
        let stored_paths = self.all_file_paths()?;
        let current_set: std::collections::HashSet<&str> =
            current_paths.iter().map(|s| s.as_str()).collect();

        let mut removed = 0;
        for path in &stored_paths {
            if path.starts_with(prefix) && !current_set.contains(path.as_str()) {
                if let Ok(file_id) = self.get_file_id(path) {
                    self.delete_file_by_id(file_id)?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
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

    fn delete_file_by_id(&self, file_id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?1)",
            params![file_id],
        )?;
        self.conn
            .execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
        Ok(())
    }

    fn all_file_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
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
        assert!(content.contains("node_modules/\n.vecgrep/"));
    }

    // --- Integration tests using in-memory DB ---

    fn make_test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (i as f32 * seed).sin()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn test_open_and_create_tables() {
        let index = Index::open_in_memory().unwrap();
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
            embedding_dim: 384,
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
            embedding_dim: 384,
            chunk_size: 500,
            chunk_overlap: 100,
        };
        let config2 = IndexConfig {
            model_name: "model-b".to_string(),
            embedding_dim: 384,
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config1).unwrap();
        assert!(!index.check_config(&config2).unwrap());
    }

    #[test]
    fn test_upsert_and_search() {
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

        assert_eq!(index.chunk_count().unwrap(), 2);

        // Search with first embedding — should find itself as top match
        let results = index.search(&embeddings[0], 2, -1.0).unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results[0].score > 0.99,
            "top match score should be near 1.0, got {}",
            results[0].score
        );
        assert_eq!(results[0].chunk.text, "fn main() {}");
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

        assert_eq!(index.chunk_count().unwrap(), 1);

        let results = index.search(&emb_v2[0], 1, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "version 2");
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

        let current = vec!["a.rs".to_string(), "c.rs".to_string()];
        let removed = index.remove_stale_files(&current).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(index.chunk_count().unwrap(), 2);
    }

    #[test]
    fn test_remove_stale_files_under() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        for name in &["src/a.rs", "src/b.rs", "lib/c.rs", "README.md"] {
            let chunks = vec![Chunk {
                file_path: name.to_string(),
                text: format!("content of {}", name),
                start_line: 1,
                end_line: 1,
            }];
            let emb = vec![make_test_embedding(dim, 1.0)];
            index.upsert_file(name, "hash", &chunks, &emb).unwrap();
        }

        let current = vec!["src/a.rs".to_string()];
        let removed = index.remove_stale_files_under(&current, "src/").unwrap();
        assert_eq!(removed, 1);
        assert_eq!(index.chunk_count().unwrap(), 3);
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
                embedding_dim: 384,
                chunk_size: 1,
                chunk_overlap: 0,
            })
            .unwrap();

        index.clear().unwrap();

        assert_eq!(index.chunk_count().unwrap(), 0);
        assert_eq!(index.get_file_hash("test.rs").unwrap(), None);
    }

    #[test]
    fn test_search_empty() {
        let index = Index::open_in_memory().unwrap();
        let query = make_test_embedding(EMBEDDING_DIM, 1.0);
        let results = index.search(&query, 10, 0.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_threshold() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let emb1 = make_test_embedding(dim, 1.0);
        let emb2 = make_test_embedding(dim, 100.0); // very different

        let chunks = vec![
            Chunk {
                file_path: "a.rs".to_string(),
                text: "similar".to_string(),
                start_line: 1,
                end_line: 1,
            },
            Chunk {
                file_path: "b.rs".to_string(),
                text: "different".to_string(),
                start_line: 1,
                end_line: 1,
            },
        ];
        index
            .upsert_file("a.rs", "h1", &chunks[0..1], &[emb1.clone()])
            .unwrap();
        index
            .upsert_file("b.rs", "h2", &chunks[1..2], &[emb2])
            .unwrap();

        // Both chunks should exist
        assert_eq!(index.chunk_count().unwrap(), 2);

        // High threshold — only the near-exact match should pass
        let results = index.search(&emb1, 10, 0.99).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "similar");
    }

    #[test]
    fn test_stats() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

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

    #[test]
    fn test_stats_on_disk() {
        let dir = TempDir::new().unwrap();
        let index = Index::open(dir.path()).unwrap();
        let dim = EMBEDDING_DIM;
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

        let stats = index.stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.chunk_count, 2);
        assert!(stats.db_size_bytes > 0);
    }

    #[test]
    fn test_search_descending_order() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let query = make_test_embedding(dim, 1.0);

        // Insert the query itself as "exact" — guaranteed top match
        let chunks_exact = vec![Chunk {
            file_path: "exact.rs".to_string(),
            text: "exact".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        index
            .upsert_file("exact.rs", "h0", &chunks_exact, &[query.clone()])
            .unwrap();

        // Insert several other embeddings
        for i in 1..4 {
            let emb = make_test_embedding(dim, (i * 10) as f32);
            let chunks = vec![Chunk {
                file_path: format!("other{i}.rs"),
                text: format!("other{i}"),
                start_line: 1,
                end_line: 1,
            }];
            index
                .upsert_file(&format!("other{i}.rs"), &format!("h{i}"), &chunks, &[emb])
                .unwrap();
        }

        let results = index.search(&query, 10, -1.0).unwrap();
        assert!(results.len() >= 2);
        // The exact match must be first
        assert_eq!(results[0].chunk.text, "exact");
        assert!(results[0].score > 0.99);
        // All results must be in descending score order
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results not in descending order at position {}: {} < {}",
                i,
                results[i - 1].score,
                results[i].score
            );
        }
    }

    #[test]
    fn test_search_top_k_larger_than_results() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        for i in 0..3 {
            let chunks = vec![Chunk {
                file_path: format!("f{i}.rs"),
                text: format!("chunk {i}"),
                start_line: 1,
                end_line: 1,
            }];
            let emb = vec![make_test_embedding(dim, (i + 1) as f32)];
            index
                .upsert_file(&format!("f{i}.rs"), &format!("h{i}"), &chunks, &emb)
                .unwrap();
        }

        // top_k=100 but only 3 chunks — should return all 3
        let query = make_test_embedding(dim, 1.0);
        let results = index.search(&query, 100, -1.0).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_all_below_threshold() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let chunks = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "content".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb = vec![make_test_embedding(dim, 1.0)];
        index.upsert_file("a.rs", "h", &chunks, &emb).unwrap();

        // Use a very different query and high threshold
        let query = make_test_embedding(dim, 100.0);
        let results = index.search(&query, 10, 0.99).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear_then_reindex() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let config = IndexConfig {
            model_name: "m".to_string(),
            embedding_dim: dim,
            chunk_size: 500,
            chunk_overlap: 100,
        };

        // Initial index
        index.set_config(&config).unwrap();
        let chunks = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "original".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb = vec![make_test_embedding(dim, 1.0)];
        index.upsert_file("a.rs", "h1", &chunks, &emb).unwrap();
        assert_eq!(index.chunk_count().unwrap(), 1);

        // Clear drops vec_chunks, set_config recreates it
        index.clear().unwrap();
        index.set_config(&config).unwrap();

        assert_eq!(index.chunk_count().unwrap(), 0);
        let results = index.search(&emb[0], 10, -1.0).unwrap();
        assert!(results.is_empty());

        // Re-index and search again
        let chunks2 = vec![Chunk {
            file_path: "b.rs".to_string(),
            text: "rebuilt".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb2 = vec![make_test_embedding(dim, 2.0)];
        index.upsert_file("b.rs", "h2", &chunks2, &emb2).unwrap();

        let results = index.search(&emb2[0], 1, -1.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "rebuilt");
    }

    #[test]
    fn test_vec_chunks_consistency() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let vec_count = |idx: &Index| -> i64 {
            idx.conn
                .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
                .unwrap()
        };

        // Insert 2 files with 1 chunk each
        for (name, seed) in &[("a.rs", 1.0f32), ("b.rs", 2.0)] {
            let chunks = vec![Chunk {
                file_path: name.to_string(),
                text: format!("content of {name}"),
                start_line: 1,
                end_line: 1,
            }];
            let emb = vec![make_test_embedding(dim, *seed)];
            index.upsert_file(name, "hash", &chunks, &emb).unwrap();
        }
        assert_eq!(index.chunk_count().unwrap(), 2);
        assert_eq!(vec_count(&index), 2);

        // Upsert a.rs (replace) — should stay at 2
        let chunks = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "updated".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb = vec![make_test_embedding(dim, 3.0)];
        index.upsert_file("a.rs", "hash2", &chunks, &emb).unwrap();
        assert_eq!(index.chunk_count().unwrap(), 2);
        assert_eq!(vec_count(&index), 2);

        // Remove stale b.rs — should drop to 1
        let current = vec!["a.rs".to_string()];
        index.remove_stale_files(&current).unwrap();
        assert_eq!(index.chunk_count().unwrap(), 1);
        assert_eq!(vec_count(&index), 1);

        // Clear should zero out both tables
        index.clear().unwrap();
        assert_eq!(index.chunk_count().unwrap(), 0);
        // vec_chunks is dropped by clear(), recreate it via set_config
        let config = IndexConfig {
            model_name: "m".to_string(),
            embedding_dim: dim,
            chunk_size: 500,
            chunk_overlap: 100,
        };
        index.set_config(&config).unwrap();
        assert_eq!(vec_count(&index), 0);
    }
}
