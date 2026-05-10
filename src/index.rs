use anyhow::{Context, Result};
use rusqlite::{params, Connection, OpenFlags};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;
use zerocopy::IntoBytes;

use crate::embedder::EMBEDDING_DIM;
use crate::types::{Chunk, IndexConfig, SearchResult, SearchScope};

static SQLITE_VEC_INIT: Once = Once::new();
const SCHEMA_VERSION: i64 = 3;
const SCOPE_OVERFETCH: usize = 3;
const HYBRID_FETCH_MULTIPLIER: usize = 5;
const RRF_K: usize = 60;

fn vec_table_ddl(dim: usize) -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
         chunk_id integer primary key, \
         embedding float[{dim}] distance_metric=cosine)"
    )
}

pub fn hybrid_lexical_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

fn init_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

// --- Helper functions that take &Connection directly ---
// These can be called both inside and outside transactions.

fn get_file_id(conn: &Connection, path: &str) -> Result<i64> {
    Ok(
        conn.query_row("SELECT id FROM files WHERE path = ?1", params![path], |r| {
            r.get(0)
        })?,
    )
}

fn delete_file_by_id(conn: &Connection, file_id: i64) -> Result<()> {
    if hybrid_enabled(conn)? {
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?1)",
            params![file_id],
        )?;
    }
    conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?1)",
        params![file_id],
    )?;
    conn.execute("DELETE FROM chunks WHERE file_id = ?1", params![file_id])?;
    conn.execute("DELETE FROM files WHERE id = ?1", params![file_id])?;
    Ok(())
}

fn all_file_paths_with_explicit(conn: &Connection) -> Result<Vec<(String, bool)>> {
    let mut stmt = conn.prepare("SELECT path, explicit FROM files")?;
    let rows: Vec<(String, bool)> = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? != 0))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
    let val = stmt
        .query_row(params![key], |row| row.get::<_, String>(0))
        .ok();
    Ok(val)
}

fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

/// Project the first four columns of a search result row
/// (`text, start_line, end_line, path`) into a `Chunk`.
/// Both vector and lexical search return these in positions 0..=3.
fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<Chunk> {
    Ok(Chunk {
        text: row.get::<_, String>(0)?,
        start_line: row.get::<_, i64>(1)? as usize,
        end_line: row.get::<_, i64>(2)? as usize,
        file_path: row.get::<_, String>(3)?,
    })
}

/// Build the WHERE-clause fragment that filters out rows whose file is
/// `explicit = 1` unless the path appears in `explicit_paths`. The returned
/// SQL starts with a leading space and `AND ` so it can be concatenated
/// directly into a query body.
///
/// `next_placeholder_idx` is the 1-based index of the first SQL placeholder
/// available for the path values (e.g. `3` if `?1` and `?2` are already used).
/// Returns `(sql_fragment, boxed_params)` ready to be appended to the
/// caller's existing param list.
fn build_explicit_filter_clause(
    explicit_paths: &[String],
    next_placeholder_idx: usize,
) -> (String, Vec<Box<dyn rusqlite::types::ToSql>>) {
    if explicit_paths.is_empty() {
        return (" AND f.explicit = 0".to_string(), Vec::new());
    }
    let placeholders: Vec<String> = (0..explicit_paths.len())
        .map(|i| format!("?{}", i + next_placeholder_idx))
        .collect();
    let sql = format!(
        " AND (f.explicit = 0 OR f.path IN ({}))",
        placeholders.join(", ")
    );
    let params: Vec<Box<dyn rusqlite::types::ToSql>> = explicit_paths
        .iter()
        .map(|p| Box::new(p.clone()) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    (sql, params)
}

fn hybrid_enabled(conn: &Connection) -> Result<bool> {
    let Some(config_json) = get_meta(conn, "config")? else {
        return Ok(false);
    };
    let config: IndexConfig = serde_json::from_str(&config_json)?;
    Ok(config.hybrid)
}

fn sync_hybrid_schema(conn: &Connection, hybrid: bool) -> Result<()> {
    if hybrid {
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                chunk_id UNINDEXED
            );",
        )?;
    } else {
        conn.execute_batch("DROP TABLE IF EXISTS chunks_fts;")?;
    }
    Ok(())
}

fn clear_all_data(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS chunks_fts;
         DROP TABLE IF EXISTS vec_chunks;
         DELETE FROM chunks;
         DELETE FROM files;
         DELETE FROM meta;",
    )?;
    Ok(())
}

pub struct Index {
    conn: Connection,
    db_path: Option<PathBuf>,
}

impl Index {
    fn with_transaction<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        debug_assert!(
            self.conn.is_autocommit(),
            "with_transaction called inside an existing transaction"
        );
        self.conn.execute("BEGIN IMMEDIATE", [])?;
        match f(&self.conn) {
            Ok(value) => {
                self.conn.execute("COMMIT", [])?;
                Ok(value)
            }
            Err(err) => {
                let _ = self.conn.execute("ROLLBACK", []);
                Err(err)
            }
        }
    }

    /// Open or create the index database at `.vecgrep/index.db` under the given root.
    pub fn open(root: &Path) -> Result<Self> {
        init_sqlite_vec();

        let index_dir = root.join(".vecgrep");
        std::fs::create_dir_all(&index_dir).context("Failed to create .vecgrep directory")?;

        // Auto-add .vecgrep/ to .gitignore
        let gitignore_path = root.join(".gitignore");
        ensure_gitignore_entry(&gitignore_path);

        let db_path = index_dir.join("index.db");
        let conn = Self::open_connection(&db_path)?;

        let index = Self {
            conn,
            db_path: Some(db_path),
        };
        index.create_tables()?;
        Ok(index)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        init_sqlite_vec();

        let conn = Connection::open_in_memory().context("Failed to open in-memory database")?;
        let index = Self {
            conn,
            db_path: None,
        };
        index.create_tables()?;
        Ok(index)
    }

    fn open_connection(path: &Path) -> Result<Connection> {
        let conn = Connection::open(path).context("Failed to open index database")?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        Ok(conn)
    }

    pub fn open_reader(&self) -> Result<Option<Self>> {
        let Some(path) = &self.db_path else {
            return Ok(None);
        };

        init_sqlite_vec();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("Failed to open read-only index connection")?;
        Ok(Some(Self {
            conn,
            db_path: Some(path.clone()),
        }))
    }

    fn create_tables(&self) -> Result<()> {
        let current_version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;

        self.with_transaction(|conn| {
            if current_version != SCHEMA_VERSION {
                conn.execute_batch(
                    "DROP TABLE IF EXISTS chunks_fts;
                     DROP TABLE IF EXISTS vec_chunks;
                     DROP TABLE IF EXISTS chunks;
                     DROP TABLE IF EXISTS files;
                     DROP TABLE IF EXISTS meta;",
                )?;
            }

            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    explicit INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER NOT NULL REFERENCES files(id),
                    text TEXT NOT NULL,
                    embedding_failed INTEGER NOT NULL DEFAULT 0,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);",
            )?;
            conn.execute(&vec_table_ddl(EMBEDDING_DIM), [])?;
            conn.execute(&format!("PRAGMA user_version = {}", SCHEMA_VERSION), [])?;

            Ok(())
        })
    }

    /// Check if the index config matches. If not, return false.
    pub fn check_config(&self, config: &IndexConfig) -> Result<bool> {
        let stored = get_meta(&self.conn, "config")?;
        let config_json = serde_json::to_string(config)?;

        match stored {
            Some(s) if s == config_json => Ok(true),
            _ => Ok(false),
        }
    }

    pub fn stored_config(&self) -> Result<Option<IndexConfig>> {
        let Some(config_json) = get_meta(&self.conn, "config")? else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(&config_json)?))
    }

    pub fn is_hybrid_capable(&self) -> Result<bool> {
        Ok(self.stored_config()?.is_some_and(|config| config.hybrid))
    }

    /// Store the current config and ensure vec_chunks table has correct dimension.
    pub fn set_config(&self, config: &IndexConfig) -> Result<()> {
        self.with_transaction(|conn| {
            let config_json = serde_json::to_string(config)?;
            set_meta(conn, "config", &config_json)?;
            sync_hybrid_schema(conn, config.hybrid)?;

            // Create vec_chunks if it doesn't exist (new DB or after clear()).
            // The dimension is correct because clear() drops vec_chunks when
            // config changes, so this always creates with the right dimension.
            conn.execute(&vec_table_ddl(config.embedding_dim), [])?;

            Ok(())
        })
    }

    /// Rebuild all index data for a new configuration atomically.
    pub fn rebuild_for_config(&self, config: &IndexConfig) -> Result<()> {
        self.with_transaction(|conn| {
            clear_all_data(conn)?;

            let config_json = serde_json::to_string(config)?;
            set_meta(conn, "config", &config_json)?;
            sync_hybrid_schema(conn, config.hybrid)?;
            conn.execute(&vec_table_ddl(config.embedding_dim), [])?;

            Ok(())
        })
    }

    /// Clear all data (for testing).
    #[cfg(test)]
    pub fn clear(&self) -> Result<()> {
        self.with_transaction(|conn| clear_all_data(conn))
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
    /// `explicit` marks files that came from a direct file path (not a
    /// directory walk); they are skipped by stale removal and excluded from
    /// search by default.
    pub fn upsert_file(
        &self,
        path: &str,
        content_hash: &str,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
        embedding_failed: &[bool],
        explicit: bool,
    ) -> Result<()> {
        self.with_transaction(|conn| {
            // Delete existing data for this file
            if let Ok(file_id) = get_file_id(conn, path) {
                delete_file_by_id(conn, file_id)?;
            }

            // Insert file record
            conn.execute(
                "INSERT INTO files (path, content_hash, explicit) VALUES (?1, ?2, ?3)",
                params![path, content_hash, explicit as i64],
            )?;
            let file_id = conn.last_insert_rowid();

            // Insert chunks and their vector embeddings
            let mut chunk_stmt = conn.prepare(
                "INSERT INTO chunks (file_id, text, embedding_failed, start_line, end_line)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let mut vec_stmt =
                conn.prepare("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)")?;
            let hybrid = hybrid_enabled(conn)?;
            let mut fts_stmt = if hybrid {
                Some(conn.prepare(
                    "INSERT INTO chunks_fts (rowid, text, chunk_id) VALUES (?1, ?2, ?3)",
                )?)
            } else {
                None
            };

            for ((chunk, embedding), failed) in chunks
                .iter()
                .zip(embeddings.iter())
                .zip(embedding_failed.iter())
            {
                chunk_stmt.execute(params![
                    file_id,
                    chunk.text,
                    *failed as i64,
                    chunk.start_line as i64,
                    chunk.end_line as i64,
                ])?;
                let chunk_id = conn.last_insert_rowid();
                vec_stmt.execute(params![chunk_id, embedding.as_slice().as_bytes()])?;
                if let Some(ref mut fts_stmt) = fts_stmt {
                    fts_stmt.execute(params![chunk_id, chunk.text, chunk_id])?;
                }
            }

            Ok(())
        })
    }

    fn search_vector_candidates(
        &self,
        query_embedding: &[f32],
        candidate_limit: usize,
        scope: &SearchScope,
    ) -> Result<Vec<SearchResult>> {
        let explicit_paths = &scope.explicit_paths;
        let path_scopes = &scope.path_scopes;
        if candidate_limit == 0 {
            return Ok(vec![]);
        }

        let fetch_k = if path_scopes.is_empty() {
            candidate_limit
        } else {
            candidate_limit * SCOPE_OVERFETCH
        };

        let (filter_sql, explicit_params) = build_explicit_filter_clause(explicit_paths, 3);
        let query = format!(
            "SELECT c.text, c.start_line, c.end_line, f.path, v.distance \
             FROM vec_chunks v \
             JOIN chunks c ON c.id = v.chunk_id \
             JOIN files f ON f.id = c.file_id \
             WHERE v.embedding MATCH ?1 \
               AND k = ?2{filter_sql} \
             ORDER BY v.distance"
        );
        let mut stmt = self.conn.prepare(&query)?;

        let row_to_result = |row: &rusqlite::Row<'_>| {
            let chunk = row_to_chunk(row)?;
            let distance = row.get::<_, f64>(4)?;
            Ok(SearchResult {
                chunk,
                score: 1.0 - distance as f32,
            })
        };

        // Fast path: no explicit paths means no per-call heap allocation
        // for params (the common case). Falls back to a boxed Vec when
        // explicit paths must be bound positionally.
        let search_results: Vec<SearchResult> = if explicit_params.is_empty() {
            stmt.query_map(
                params![query_embedding.as_bytes(), fetch_k as i64],
                row_to_result,
            )?
            .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut all: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(2 + explicit_params.len());
            all.push(Box::new(query_embedding.as_bytes().to_vec()));
            all.push(Box::new(fetch_k as i64));
            all.extend(explicit_params);
            let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
            stmt.query_map(refs.as_slice(), row_to_result)?
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(scope_results(search_results, path_scopes, candidate_limit))
    }

    pub fn search_lexical(
        &self,
        query: &str,
        candidate_limit: usize,
        scope: &SearchScope,
    ) -> Result<Vec<SearchResult>> {
        let explicit_paths = &scope.explicit_paths;
        let path_scopes = &scope.path_scopes;
        if candidate_limit == 0 {
            return Ok(vec![]);
        }
        let Some(match_query) = hybrid_lexical_query(query) else {
            return Ok(vec![]);
        };

        let fetch_limit = if path_scopes.is_empty() {
            candidate_limit
        } else {
            candidate_limit * SCOPE_OVERFETCH
        };

        let (filter_sql, explicit_params) = build_explicit_filter_clause(explicit_paths, 3);
        let query = format!(
            "SELECT c.text, c.start_line, c.end_line, f.path, bm25(chunks_fts) AS rank \
             FROM chunks_fts \
             JOIN chunks c ON c.id = chunks_fts.chunk_id \
             JOIN files f ON f.id = c.file_id \
             WHERE chunks_fts MATCH ?1{filter_sql} \
             ORDER BY rank \
             LIMIT ?2"
        );
        let mut stmt = self.conn.prepare(&query)?;

        let row_to_result = |row: &rusqlite::Row<'_>| {
            Ok(SearchResult {
                chunk: row_to_chunk(row)?,
                score: 0.0,
            })
        };

        // Fast path: no explicit paths means no per-call heap allocation
        // for params (the common case).
        let search_results: Vec<SearchResult> = if explicit_params.is_empty() {
            stmt.query_map(params![match_query, fetch_limit as i64], row_to_result)?
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut all: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(2 + explicit_params.len());
            all.push(Box::new(match_query));
            all.push(Box::new(fetch_limit as i64));
            all.extend(explicit_params);
            let refs: Vec<&dyn rusqlite::types::ToSql> = all.iter().map(|p| p.as_ref()).collect();
            stmt.query_map(refs.as_slice(), row_to_result)?
                .collect::<Result<Vec<_>, _>>()?
        };

        Ok(scope_results(search_results, path_scopes, candidate_limit))
    }

    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
        scope: &SearchScope,
        lexical_results: Option<Vec<SearchResult>>,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let candidate_limit = top_k * HYBRID_FETCH_MULTIPLIER;
        let vector_results =
            self.search_vector_candidates(query_embedding, candidate_limit, scope)?;
        let lexical_results = match lexical_results {
            Some(results) => results,
            None => self.search_lexical(query, candidate_limit, scope)?,
        };

        Ok(fuse_ranked_results(
            query,
            vector_results,
            lexical_results,
            top_k,
            threshold,
        ))
    }

    /// Search for chunks most similar to the query embedding.
    /// See `SearchScope` for how explicit files and path scoping work.
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
        scope: &SearchScope,
    ) -> Result<Vec<SearchResult>> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let search_results: Vec<SearchResult> = self
            .search_vector_candidates(query_embedding, top_k, scope)?
            .into_iter()
            .filter(|result| result.score >= threshold)
            .collect();
        Ok(search_results.into_iter().take(top_k).collect())
    }

    /// Get the number of files in the index.
    pub fn file_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    /// Get the number of chunks in the index.
    pub fn chunk_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(count as usize)
    }

    /// Remove non-explicit stale files and clear the explicit flag on walked files.
    /// If `scope` is provided, only non-explicit files under that directory are
    /// considered stale. Explicit files are never removed (they stay cached for
    /// fast re-search). Files seen in the walk have their explicit flag cleared
    /// so they become normal cached entries.
    pub fn remove_stale_files(
        &self,
        current_paths: &[String],
        scope: Option<&Path>,
    ) -> Result<usize> {
        self.with_transaction(|conn| {
            let stored = all_file_paths_with_explicit(conn)?;
            let current_set: std::collections::HashSet<&str> =
                current_paths.iter().map(|s| s.as_str()).collect();

            let mut removed = 0;
            for (path, is_explicit) in &stored {
                if !current_set.contains(path.as_str()) {
                    // Only remove non-explicit files within the prefix scope.
                    // Explicit files stay cached for fast re-search.
                    if !is_explicit {
                        let in_scope = scope.is_none_or(|s| crate::paths::is_under(path, s));
                        if in_scope {
                            if let Ok(file_id) = get_file_id(conn, path) {
                                delete_file_by_id(conn, file_id)?;
                                removed += 1;
                            }
                        }
                    }
                }
            }

            // Clear explicit flag on files seen in the walk (handles hash-match case
            // where the file wasn't re-indexed but should become a normal entry)
            for path in current_paths {
                conn.execute(
                    "UPDATE files SET explicit = 0 WHERE path = ?1 AND explicit = 1",
                    params![path],
                )?;
            }

            Ok(removed)
        })
    }

    /// Get the number of chunks with failed embeddings (zero vectors).
    pub fn failed_chunk_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(embedding_failed), 0) FROM chunks",
            [],
            |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        let db_path = self.db_path()?;
        let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

        Ok(IndexStats {
            file_count: self.file_count()?,
            chunk_count: self.chunk_count()?,
            failed_chunk_count: self.failed_chunk_count()?,
            db_size_bytes: db_size,
        })
    }

    fn db_path(&self) -> Result<PathBuf> {
        let path: String = self
            .conn
            .query_row("PRAGMA database_list", [], |r| r.get(2))?;
        Ok(PathBuf::from(path))
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct ResultKey {
    file_path: String,
    start_line: usize,
    end_line: usize,
}

impl ResultKey {
    fn from_result(result: &SearchResult) -> Self {
        Self {
            file_path: result.chunk.file_path.clone(),
            start_line: result.chunk.start_line,
            end_line: result.chunk.end_line,
        }
    }
}

struct HybridCandidate {
    result: SearchResult,
    vector_rank: Option<usize>,
    lexical_rank: Option<usize>,
}

fn scope_results(
    results: Vec<SearchResult>,
    path_scopes: &[String],
    candidate_limit: usize,
) -> Vec<SearchResult> {
    if path_scopes.is_empty() {
        return results.into_iter().take(candidate_limit).collect();
    }

    results
        .into_iter()
        .filter(|result| {
            path_scopes
                .iter()
                .any(|scope| crate::paths::is_under(&result.chunk.file_path, Path::new(scope)))
        })
        .take(candidate_limit)
        .collect()
}

fn rank_component(rank: usize) -> f32 {
    1.0 / (RRF_K + rank + 1) as f32
}

fn lexical_weight(query: &str) -> f32 {
    let token_count = query
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .count();
    let has_code_like_marker = query.chars().any(|ch| {
        matches!(
            ch,
            '_' | ':' | '/' | '\\' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '=' | '<' | '>'
        ) || ch.is_ascii_digit()
    });

    if has_code_like_marker {
        0.65
    } else if token_count <= 2 {
        0.45
    } else if token_count == 3 {
        0.20
    } else {
        0.10
    }
}

pub fn hybrid_fused_score(
    query: &str,
    vector_rank: Option<usize>,
    lexical_rank: Option<usize>,
) -> f32 {
    let vector_weight = 1.0;
    let lexical_weight = lexical_weight(query);
    let lexical_only_weight = lexical_weight * 0.2;
    let max_score = (vector_weight + lexical_weight) * rank_component(0);
    let vector_score = vector_rank
        .map(|rank| vector_weight * rank_component(rank))
        .unwrap_or(0.0);
    let lexical_score = lexical_rank
        .map(|rank| {
            let weight = if vector_rank.is_some() {
                lexical_weight
            } else {
                lexical_only_weight
            };
            weight * rank_component(rank)
        })
        .unwrap_or(0.0);
    let score = vector_score + lexical_score;
    if max_score == 0.0 {
        0.0
    } else {
        score / max_score
    }
}

fn compare_scores_desc(lhs: f32, rhs: f32) -> Ordering {
    rhs.partial_cmp(&lhs).unwrap_or(Ordering::Equal)
}

fn fuse_ranked_results(
    query: &str,
    vector_results: Vec<SearchResult>,
    lexical_results: Vec<SearchResult>,
    top_k: usize,
    threshold: f32,
) -> Vec<SearchResult> {
    let mut merged: HashMap<ResultKey, HybridCandidate> = HashMap::new();

    for (rank, result) in vector_results.into_iter().enumerate() {
        let key = ResultKey::from_result(&result);
        merged
            .entry(key)
            .and_modify(|candidate| candidate.vector_rank = Some(rank))
            .or_insert(HybridCandidate {
                result,
                vector_rank: Some(rank),
                lexical_rank: None,
            });
    }

    for (rank, result) in lexical_results.into_iter().enumerate() {
        let key = ResultKey::from_result(&result);
        merged
            .entry(key)
            .and_modify(|candidate| candidate.lexical_rank = Some(rank))
            .or_insert(HybridCandidate {
                result,
                vector_rank: None,
                lexical_rank: Some(rank),
            });
    }

    let mut fused: Vec<SearchResult> = merged
        .into_values()
        .filter_map(|candidate| {
            let score = hybrid_fused_score(query, candidate.vector_rank, candidate.lexical_rank);
            if score >= threshold {
                Some(SearchResult {
                    score,
                    ..candidate.result
                })
            } else {
                None
            }
        })
        .collect();

    fused.sort_by(|lhs, rhs| {
        compare_scores_desc(lhs.score, rhs.score)
            .then_with(|| lhs.chunk.file_path.cmp(&rhs.chunk.file_path))
            .then_with(|| lhs.chunk.start_line.cmp(&rhs.chunk.start_line))
            .then_with(|| lhs.chunk.end_line.cmp(&rhs.chunk.end_line))
    });
    fused.truncate(top_k);
    fused
}

pub struct IndexStats {
    pub file_count: usize,
    pub chunk_count: usize,
    pub failed_chunk_count: usize,
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
        if let Err(e) = std::fs::write(gitignore_path, new_content) {
            tracing::warn!(
                "Failed to update .gitignore at {}: {}",
                gitignore_path.display(),
                e
            );
        }
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
            hybrid: false,
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
            hybrid: false,
        };
        let config2 = IndexConfig {
            model_name: "model-b".to_string(),
            embedding_dim: 384,
            chunk_size: 500,
            chunk_overlap: 100,
            hybrid: false,
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
            .upsert_file(
                "test.rs",
                "abc123",
                &chunks,
                &embeddings,
                &[false, false],
                false,
            )
            .unwrap();

        assert_eq!(index.chunk_count().unwrap(), 2);

        // Search with first embedding — should find itself as top match
        let results = index
            .search(&embeddings[0], 2, -1.0, &SearchScope::default())
            .unwrap();
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
            .upsert_file("a.rs", "hash1", &chunks_v1, &emb_v1, &[false], false)
            .unwrap();

        let chunks_v2 = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "version 2".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb_v2 = vec![make_test_embedding(dim, 2.0)];
        index
            .upsert_file("a.rs", "hash2", &chunks_v2, &emb_v2, &[false], false)
            .unwrap();

        assert_eq!(index.chunk_count().unwrap(), 1);

        let results = index
            .search(&emb_v2[0], 1, 0.0, &SearchScope::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "version 2");
    }

    #[test]
    fn test_search_lexical_finds_matches() {
        // Insert three distinct chunks so the lexical query must discriminate:
        //   notes.rs   : strong match for both query terms
        //   partial.rs : matches only one query term
        //   unrelated.rs : matches neither
        // A regression that returns every chunk indiscriminately would fail
        // the unrelated-not-included assertion. A regression that swapped
        // BM25 ranking would fail the ordering assertion.
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        index
            .set_config(&IndexConfig {
                model_name: "test-model".to_string(),
                embedding_dim: dim,
                chunk_size: 500,
                chunk_overlap: 100,
                hybrid: true,
            })
            .unwrap();

        let entries: [(&str, &str); 3] = [
            ("notes.rs", "retry timeout handling logic"),
            ("partial.rs", "timeout configuration values"),
            ("unrelated.rs", "chocolate cake recipe ingredients"),
        ];
        for (i, (path, text)) in entries.iter().enumerate() {
            let chunks = vec![Chunk {
                file_path: (*path).to_string(),
                text: (*text).to_string(),
                start_line: 1,
                end_line: 1,
            }];
            let emb = vec![make_test_embedding(dim, 1.0 + i as f32)];
            index
                .upsert_file(path, &format!("hash-{i}"), &chunks, &emb, &[false], false)
                .unwrap();
        }

        let results = index
            .search_lexical("timeout retry", 10, &SearchScope::default())
            .unwrap();

        let paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        assert!(
            paths.contains(&"notes.rs"),
            "strong match notes.rs missing: {paths:?}"
        );
        assert!(
            paths.contains(&"partial.rs"),
            "single-term match partial.rs missing: {paths:?}"
        );
        assert!(
            !paths.contains(&"unrelated.rs"),
            "unrelated.rs must not appear in results: {paths:?}"
        );
        assert_eq!(
            results.len(),
            2,
            "expected exactly 2 matches, got {paths:?}"
        );
        // BM25 ranking: a chunk matching both query terms must outrank one
        // matching a single term.
        assert_eq!(
            results[0].chunk.file_path, "notes.rs",
            "two-term match notes.rs must rank above one-term match: {paths:?}"
        );
        assert!(
            results[0].score >= results[1].score,
            "results must be in descending score order: {} >= {} ({paths:?})",
            results[0].score,
            results[1].score
        );
    }

    #[test]
    fn test_upsert_replaces_lexical_entries() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        index
            .set_config(&IndexConfig {
                model_name: "test-model".to_string(),
                embedding_dim: dim,
                chunk_size: 500,
                chunk_overlap: 100,
                hybrid: true,
            })
            .unwrap();

        let chunks_v1 = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "alpha token".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb_v1 = vec![make_test_embedding(dim, 1.0)];
        index
            .upsert_file("a.rs", "hash1", &chunks_v1, &emb_v1, &[false], false)
            .unwrap();

        let chunks_v2 = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "beta token".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb_v2 = vec![make_test_embedding(dim, 2.0)];
        index
            .upsert_file("a.rs", "hash2", &chunks_v2, &emb_v2, &[false], false)
            .unwrap();

        let old_results = index
            .search_lexical("alpha", 10, &SearchScope::default())
            .unwrap();
        assert!(old_results.is_empty());

        let new_results = index
            .search_lexical("beta", 10, &SearchScope::default())
            .unwrap();
        assert_eq!(new_results.len(), 1);
        assert_eq!(new_results[0].chunk.text, "beta token");
    }

    #[test]
    fn test_hybrid_capability_roundtrip() {
        let index = Index::open_in_memory().unwrap();
        assert!(!index.is_hybrid_capable().unwrap());

        index
            .set_config(&IndexConfig {
                model_name: "test-model".to_string(),
                embedding_dim: EMBEDDING_DIM,
                chunk_size: 500,
                chunk_overlap: 100,
                hybrid: true,
            })
            .unwrap();

        assert!(index.is_hybrid_capable().unwrap());
        assert_eq!(index.stored_config().unwrap().unwrap().hybrid, true);
    }

    #[test]
    fn test_rebuild_for_config_can_enable_hybrid_capability() {
        let index = Index::open_in_memory().unwrap();
        index
            .rebuild_for_config(&IndexConfig {
                model_name: "test-model".to_string(),
                embedding_dim: EMBEDDING_DIM,
                chunk_size: 500,
                chunk_overlap: 100,
                hybrid: true,
            })
            .unwrap();

        assert!(index.is_hybrid_capable().unwrap());
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
            .upsert_file("test.rs", "myhash", &chunks, &embeddings, &[false], false)
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
            index
                .upsert_file(name, "hash", &chunks, &emb, &[false], false)
                .unwrap();
        }

        let current = vec!["a.rs".to_string(), "c.rs".to_string()];
        let removed = index.remove_stale_files(&current, None).unwrap();
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
            index
                .upsert_file(name, "hash", &chunks, &emb, &[false], false)
                .unwrap();
        }

        let current = vec!["src/a.rs".to_string()];
        let removed = index
            .remove_stale_files(&current, Some(Path::new("src")))
            .unwrap();
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
            .upsert_file("test.rs", "hash", &chunks, &embeddings, &[false], false)
            .unwrap();
        index
            .set_config(&IndexConfig {
                model_name: "m".to_string(),
                embedding_dim: 384,
                chunk_size: 1,
                chunk_overlap: 0,
                hybrid: false,
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
        let results = index
            .search(&query, 10, 0.0, &SearchScope::default())
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_threshold() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let emb1 = make_test_embedding(dim, 1.0);
        let emb2 = make_test_embedding(dim, 100.0);

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
            .upsert_file(
                "a.rs",
                "h1",
                &chunks[0..1],
                &[emb1.clone()],
                &[false],
                false,
            )
            .unwrap();
        index
            .upsert_file("b.rs", "h2", &chunks[1..2], &[emb2], &[false], false)
            .unwrap();

        assert_eq!(index.chunk_count().unwrap(), 2);

        // Precondition: verify the two embeddings are actually dissimilar
        let all_results = index
            .search(&emb1, 10, -1.0, &SearchScope::default())
            .unwrap();
        let other_score = all_results
            .iter()
            .find(|r| r.chunk.text == "different")
            .expect("should find 'different' chunk")
            .score;
        assert!(
            other_score < 0.5,
            "test precondition: embeddings must be dissimilar, got {other_score}"
        );

        // High threshold — only the near-exact match should pass
        let results = index
            .search(&emb1, 10, 0.99, &SearchScope::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "similar");
    }

    #[test]
    fn test_stats_in_memory() {
        let index = Index::open_in_memory().unwrap();
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
        index
            .upsert_file("a.rs", "hash", &chunks, &emb, &[false, true], false)
            .unwrap();

        let stats = index.stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.chunk_count, 2);
        assert_eq!(stats.failed_chunk_count, 1);
        assert_eq!(stats.db_size_bytes, 0); // in-memory DB has no file
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
        index
            .upsert_file("a.rs", "hash", &chunks, &emb, &[false, true], false)
            .unwrap();

        let stats = index.stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.chunk_count, 2);
        assert_eq!(stats.failed_chunk_count, 1);
        assert!(stats.db_size_bytes > 0);
    }

    #[test]
    fn test_stats_reindex_replaces_failed_chunk_count() {
        let index = Index::open_in_memory().unwrap();
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

        let emb = vec![make_test_embedding(dim, 1.0), vec![0.0; dim]];
        index
            .upsert_file("a.rs", "hash1", &chunks, &emb, &[false, true], false)
            .unwrap();
        assert_eq!(index.stats().unwrap().failed_chunk_count, 1);

        let replacement_chunks = vec![Chunk {
            file_path: "a.rs".to_string(),
            text: "replacement".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let replacement_emb = vec![make_test_embedding(dim, 2.0)];
        index
            .upsert_file(
                "a.rs",
                "hash2",
                &replacement_chunks,
                &replacement_emb,
                &[false],
                false,
            )
            .unwrap();

        let stats = index.stats().unwrap();
        assert_eq!(stats.file_count, 1);
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.failed_chunk_count, 0);
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
            .upsert_file(
                "exact.rs",
                "h0",
                &chunks_exact,
                &[query.clone()],
                &[false],
                false,
            )
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
                .upsert_file(
                    &format!("other{i}.rs"),
                    &format!("h{i}"),
                    &chunks,
                    &[emb],
                    &[false],
                    false,
                )
                .unwrap();
        }

        let results = index
            .search(&query, 10, -1.0, &SearchScope::default())
            .unwrap();
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
                .upsert_file(
                    &format!("f{i}.rs"),
                    &format!("h{i}"),
                    &chunks,
                    &emb,
                    &[false],
                    false,
                )
                .unwrap();
        }

        // top_k=100 but only 3 chunks — should return all 3
        let query = make_test_embedding(dim, 1.0);
        let results = index
            .search(&query, 100, -1.0, &SearchScope::default())
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_top_k_zero() {
        let index = Index::open_in_memory().unwrap();
        let query = make_test_embedding(EMBEDDING_DIM, 1.0);
        let results = index
            .search(&query, 0, -1.0, &SearchScope::default())
            .unwrap();
        assert!(results.is_empty());
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
        index
            .upsert_file("a.rs", "h", &chunks, &emb, &[false], false)
            .unwrap();

        // Use a very different query and high threshold
        let query = make_test_embedding(dim, 100.0);
        let results = index
            .search(&query, 10, 0.99, &SearchScope::default())
            .unwrap();
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
            hybrid: false,
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
        index
            .upsert_file("a.rs", "h1", &chunks, &emb, &[false], false)
            .unwrap();
        assert_eq!(index.chunk_count().unwrap(), 1);

        // Clear drops vec_chunks, set_config recreates it
        index.clear().unwrap();
        index.set_config(&config).unwrap();

        assert_eq!(index.chunk_count().unwrap(), 0);
        let results = index
            .search(&emb[0], 10, -1.0, &SearchScope::default())
            .unwrap();
        assert!(results.is_empty());

        // Re-index and search again
        let chunks2 = vec![Chunk {
            file_path: "b.rs".to_string(),
            text: "rebuilt".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb2 = vec![make_test_embedding(dim, 2.0)];
        index
            .upsert_file("b.rs", "h2", &chunks2, &emb2, &[false], false)
            .unwrap();

        let results = index
            .search(&emb2[0], 1, -1.0, &SearchScope::default())
            .unwrap();
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
            index
                .upsert_file(name, "hash", &chunks, &emb, &[false], false)
                .unwrap();
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
        index
            .upsert_file("a.rs", "hash2", &chunks, &emb, &[false], false)
            .unwrap();
        assert_eq!(index.chunk_count().unwrap(), 2);
        assert_eq!(vec_count(&index), 2);

        // Remove stale b.rs — should drop to 1
        let current = vec!["a.rs".to_string()];
        index.remove_stale_files(&current, None).unwrap();
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
            hybrid: false,
        };
        index.set_config(&config).unwrap();
        assert_eq!(vec_count(&index), 0);
    }

    #[test]
    fn test_rebuild_for_config_with_different_dimension() {
        // Regression: open_in_memory creates vec_chunks at EMBEDDING_DIM (384).
        // rebuild_for_config must drop and recreate with the new dimension,
        // otherwise inserts fail with "Expected 384 dimensions but received N".
        let index = Index::open_in_memory().unwrap();
        let new_dim = 1024;
        let config = IndexConfig {
            model_name: "remote-model".to_string(),
            embedding_dim: new_dim,
            chunk_size: 500,
            chunk_overlap: 100,
            hybrid: false,
        };
        index.rebuild_for_config(&config).unwrap();

        let chunks = vec![Chunk {
            file_path: "test.rs".to_string(),
            text: "test content".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb = vec![make_test_embedding(new_dim, 1.0)];
        // This would fail with dimension mismatch if rebuild_for_config
        // didn't recreate vec_chunks with the correct dimension.
        index
            .upsert_file("test.rs", "hash", &chunks, &emb, &[false], false)
            .unwrap();

        assert_eq!(index.chunk_count().unwrap(), 1);

        let results = index
            .search(&emb[0], 1, -1.0, &SearchScope::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "test content");
    }

    #[test]
    fn test_rebuild_for_config_clears_old_data() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let config1 = IndexConfig {
            model_name: "model-a".to_string(),
            embedding_dim: dim,
            chunk_size: 500,
            chunk_overlap: 100,
            hybrid: false,
        };
        index.set_config(&config1).unwrap();
        let chunks = vec![Chunk {
            file_path: "old.rs".to_string(),
            text: "old data".to_string(),
            start_line: 1,
            end_line: 1,
        }];
        let emb = vec![make_test_embedding(dim, 1.0)];
        index
            .upsert_file("old.rs", "hash1", &chunks, &emb, &[false], false)
            .unwrap();
        assert_eq!(index.chunk_count().unwrap(), 1);

        let config2 = IndexConfig {
            model_name: "model-b".to_string(),
            embedding_dim: dim,
            chunk_size: 200,
            chunk_overlap: 50,
            hybrid: false,
        };
        index.rebuild_for_config(&config2).unwrap();

        assert_eq!(index.chunk_count().unwrap(), 0);
        assert_eq!(index.get_file_hash("old.rs").unwrap(), None);
        assert!(index.check_config(&config2).unwrap());
        assert!(!index.check_config(&config1).unwrap());
    }

    // --- Explicit flag tests ---

    #[test]
    fn test_search_filters_explicit_files_by_path() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        let emb_normal = make_test_embedding(dim, 1.0);
        let emb_explicit1 = make_test_embedding(dim, 2.0);
        let emb_explicit2 = make_test_embedding(dim, 3.0);

        index
            .upsert_file(
                "normal.rs",
                "h1",
                &[Chunk {
                    file_path: "normal.rs".to_string(),
                    text: "normal file".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb_normal.clone()],
                &[false],
                false,
            )
            .unwrap();
        index
            .upsert_file(
                "secret.log",
                "h2",
                &[Chunk {
                    file_path: "secret.log".to_string(),
                    text: "explicit file".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb_explicit1.clone()],
                &[false],
                true,
            )
            .unwrap();
        index
            .upsert_file(
                "other.log",
                "h3",
                &[Chunk {
                    file_path: "other.log".to_string(),
                    text: "another explicit file".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb_explicit2.clone()],
                &[false],
                true,
            )
            .unwrap();

        assert_eq!(index.chunk_count().unwrap(), 3);

        // explicit_paths=Some(["secret.log"]): normal + only secret.log
        let scope = SearchScope {
            explicit_paths: vec!["secret.log".to_string()],
            ..Default::default()
        };
        let results = index.search(&emb_explicit1, 10, -1.0, &scope).unwrap();
        let paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        assert!(
            paths.contains(&"secret.log"),
            "expected secret.log with explicit_paths=Some, got: {paths:?}"
        );
        assert!(
            paths.contains(&"normal.rs"),
            "expected normal.rs with explicit_paths=Some, got: {paths:?}"
        );
        assert!(
            !paths.contains(&"other.log"),
            "other.log should be excluded (not in explicit_paths), got: {paths:?}"
        );

        // explicit_paths=None: only normal file appears
        let results = index
            .search(&emb_normal, 10, -1.0, &SearchScope::default())
            .unwrap();
        let paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        assert!(
            paths.contains(&"normal.rs"),
            "expected normal.rs with explicit_paths=None"
        );
        assert!(
            !paths.contains(&"secret.log"),
            "secret.log should be excluded with explicit_paths=None"
        );
        assert!(
            !paths.contains(&"other.log"),
            "other.log should be excluded with explicit_paths=None"
        );
    }

    #[test]
    fn test_stale_removal_preserves_explicit_files() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let emb = make_test_embedding(dim, 1.0);

        // Add a normal file and an explicit file
        index
            .upsert_file(
                "stale.rs",
                "h1",
                &[Chunk {
                    file_path: "stale.rs".to_string(),
                    text: "stale".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb.clone()],
                &[false],
                false,
            )
            .unwrap();
        index
            .upsert_file(
                "cached.log",
                "h2",
                &[Chunk {
                    file_path: "cached.log".to_string(),
                    text: "cached explicit".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb.clone()],
                &[false],
                true,
            )
            .unwrap();

        // Walk found neither file — stale.rs should be removed, cached.log should survive
        let removed = index.remove_stale_files(&[], None).unwrap();
        assert_eq!(removed, 1, "only non-explicit stale file should be removed");
        assert_eq!(index.get_file_hash("stale.rs").unwrap(), None);
        assert_eq!(
            index.get_file_hash("cached.log").unwrap(),
            Some("h2".to_string()),
            "explicit file should be preserved"
        );
    }

    #[test]
    fn test_stale_removal_clears_explicit_flag_on_walked_files() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;
        let emb = make_test_embedding(dim, 1.0);

        // Add a file as explicit
        index
            .upsert_file(
                "main.rs",
                "h1",
                &[Chunk {
                    file_path: "main.rs".to_string(),
                    text: "fn main".to_string(),
                    start_line: 1,
                    end_line: 1,
                }],
                &[emb.clone()],
                &[false],
                true,
            )
            .unwrap();

        // Verify it's excluded from non-explicit search
        let results = index
            .search(&emb, 10, -1.0, &SearchScope::default())
            .unwrap();
        assert!(results.is_empty(), "explicit file should be excluded");

        // Simulate a directory walk that found main.rs (hash matched, not re-indexed)
        let walked = vec!["main.rs".to_string()];
        index.remove_stale_files(&walked, None).unwrap();

        // Now main.rs should appear in non-explicit search (flag cleared)
        let results = index
            .search(&emb, 10, -1.0, &SearchScope::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.file_path, "main.rs");
    }

    #[test]
    fn test_schema_version_change_triggers_rebuild() {
        init_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        let index = Index {
            conn,
            db_path: None,
        };
        index.create_tables().unwrap();

        // Insert data
        let dim = EMBEDDING_DIM;
        let emb = make_test_embedding(dim, 1.0);
        let chunk = Chunk {
            file_path: "old.rs".into(),
            text: "old data".into(),
            start_line: 1,
            end_line: 1,
        };
        index
            .upsert_file("old.rs", "hash", &[chunk], &[emb], &[false], false)
            .unwrap();
        assert_eq!(index.file_count().unwrap(), 1);

        // Simulate an old schema version
        index.conn.execute("PRAGMA user_version = 1", []).unwrap();

        // Re-create tables — should detect version mismatch and drop everything
        index.create_tables().unwrap();

        // All data should be gone (tables were dropped and recreated)
        assert_eq!(index.file_count().unwrap(), 0);
        assert_eq!(index.chunk_count().unwrap(), 0);

        // Schema version should be current
        let version: i64 = index
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn test_no_scope_returns_all_results() {
        let index = Index::open_in_memory().unwrap();
        let dim = EMBEDDING_DIM;

        // Insert files in different directories
        for (path, seed) in &[("src/a.rs", 1.0), ("tests/b.rs", 1.1), ("docs/c.md", 1.2)] {
            let emb = make_test_embedding(dim, *seed);
            let chunk = Chunk {
                file_path: path.to_string(),
                text: format!("content of {path}"),
                start_line: 1,
                end_line: 1,
            };
            index
                .upsert_file(
                    path,
                    &format!("hash-{path}"),
                    &[chunk],
                    &[emb],
                    &[false],
                    false,
                )
                .unwrap();
        }

        // SearchScope::default() (no scoping) should return all files
        let query = make_test_embedding(dim, 1.05);
        let results = index
            .search(&query, 10, -1.0, &SearchScope::default())
            .unwrap();

        let paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        assert_eq!(paths.len(), 3, "expected all 3 files, got: {paths:?}");
        assert!(paths.contains(&"src/a.rs"));
        assert!(paths.contains(&"tests/b.rs"));
        assert!(paths.contains(&"docs/c.md"));
    }
}
