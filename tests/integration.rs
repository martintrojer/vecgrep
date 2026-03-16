use vecgrep::embedder::EMBEDDING_DIM;
use vecgrep::index::Index;
use vecgrep::types::{Chunk, IndexConfig};

fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|i| (i as f32 * seed).sin()).collect();
    // L2 normalize
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[test]
fn test_index_and_search_roundtrip() {
    let index = Index::open_in_memory().unwrap();
    let dim = EMBEDDING_DIM;

    let chunks = vec![
        Chunk {
            file_path: "main.rs".to_string(),
            text: "fn main() { println!(\"hello\"); }".to_string(),
            start_line: 1,
            end_line: 1,
        },
        Chunk {
            file_path: "main.rs".to_string(),
            text: "fn helper() { }".to_string(),
            start_line: 3,
            end_line: 3,
        },
        Chunk {
            file_path: "lib.rs".to_string(),
            text: "pub fn add(a: i32, b: i32) -> i32 { a + b }".to_string(),
            start_line: 1,
            end_line: 1,
        },
    ];

    let embeddings = vec![
        make_embedding(dim, 1.0),
        make_embedding(dim, 2.0),
        make_embedding(dim, 3.0),
    ];

    // Store main.rs chunks
    index
        .upsert_file(
            "main.rs",
            "hash_main",
            &chunks[0..2],
            &embeddings[0..2],
            &[false, false],
        )
        .unwrap();

    // Store lib.rs chunk
    index
        .upsert_file(
            "lib.rs",
            "hash_lib",
            &chunks[2..3],
            &embeddings[2..3],
            &[false],
        )
        .unwrap();

    assert_eq!(index.chunk_count().unwrap(), 3);

    // Search with the first embedding as query — should find itself as top match
    let results = index.search(&embeddings[0], 3, 0.0, true).unwrap();
    assert!(!results.is_empty());
    // The top result should have a very high similarity (close to 1.0)
    assert!(results[0].score > 0.99);
}

#[test]
fn test_incremental_indexing() {
    let index = Index::open_in_memory().unwrap();
    let dim = EMBEDDING_DIM;

    let config = IndexConfig {
        model_name: "test-model".to_string(),
        embedding_dim: EMBEDDING_DIM,
        chunk_size: 500,
        chunk_overlap: 100,
    };
    index.set_config(&config).unwrap();

    // Index file a.rs
    let chunk_a = vec![Chunk {
        file_path: "a.rs".to_string(),
        text: "fn a() {}".to_string(),
        start_line: 1,
        end_line: 1,
    }];
    let emb_a = vec![make_embedding(dim, 1.0)];
    index
        .upsert_file("a.rs", "hash_a_v1", &chunk_a, &emb_a, &[false])
        .unwrap();

    // Index file b.rs
    let chunk_b = vec![Chunk {
        file_path: "b.rs".to_string(),
        text: "fn b() {}".to_string(),
        start_line: 1,
        end_line: 1,
    }];
    let emb_b = vec![make_embedding(dim, 2.0)];
    index
        .upsert_file("b.rs", "hash_b", &chunk_b, &emb_b, &[false])
        .unwrap();

    // Verify both are present
    assert_eq!(index.chunk_count().unwrap(), 2);

    // Now "modify" a.rs (new content hash and embedding)
    let chunk_a_v2 = vec![Chunk {
        file_path: "a.rs".to_string(),
        text: "fn a_modified() {}".to_string(),
        start_line: 1,
        end_line: 1,
    }];
    let emb_a_v2 = vec![make_embedding(dim, 3.0)];

    // Check hash — should differ
    let stored_hash = index.get_file_hash("a.rs").unwrap();
    assert_eq!(stored_hash, Some("hash_a_v1".to_string()));

    // Re-index only a.rs
    index
        .upsert_file("a.rs", "hash_a_v2", &chunk_a_v2, &emb_a_v2, &[false])
        .unwrap();

    // Verify: still 2 chunks total
    assert_eq!(index.chunk_count().unwrap(), 2);

    // Verify a.rs has the updated content
    let results = index.search(&emb_a_v2[0], 1, -1.0, true).unwrap();
    assert_eq!(results[0].chunk.text, "fn a_modified() {}");

    // Verify b.rs is still intact
    let results = index.search(&emb_b[0], 1, -1.0, true).unwrap();
    assert_eq!(results[0].chunk.text, "fn b() {}");

    // Verify updated hash
    let new_hash = index.get_file_hash("a.rs").unwrap();
    assert_eq!(new_hash, Some("hash_a_v2".to_string()));
}

// --- CLI flag tests using the binary ---

#[test]
fn test_show_root_at_git_root() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--show-root")
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let expected = dir.path().canonicalize().unwrap();
    assert_eq!(stdout.trim(), expected.display().to_string());
}

#[test]
fn test_show_root_from_subdirectory() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir_all(dir.path().join("src/deep")).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--show-root")
        .current_dir(dir.path().join("src/deep"))
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let expected = dir.path().canonicalize().unwrap();
    assert_eq!(stdout.trim(), expected.display().to_string());
}

#[test]
fn test_show_root_no_marker_falls_back_to_cwd() {
    let dir = tempfile::TempDir::new().unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--show-root")
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let expected = dir.path().canonicalize().unwrap();
    assert_eq!(stdout.trim(), expected.display().to_string());
}

#[test]
fn test_rejects_file_outside_selected_root() {
    let parent = tempfile::TempDir::new().unwrap();
    let repo = parent.path().join("repo");
    let other = parent.path().join("other");
    std::fs::create_dir(&repo).unwrap();
    std::fs::create_dir(&repo.join(".git")).unwrap();
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("outside.rs"), "fn outside_bug() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["bugs", "../other/outside.rs"])
        .current_dir(&repo)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("outside the selected project root"),
        "expected outside-root error, got: {stderr}"
    );
}

#[test]
fn test_rejects_directory_outside_selected_root() {
    let parent = tempfile::TempDir::new().unwrap();
    let repo = parent.path().join("repo");
    let other = parent.path().join("other");
    std::fs::create_dir(&repo).unwrap();
    std::fs::create_dir(&repo.join(".git")).unwrap();
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("outside.rs"), "fn outside_bug() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["bugs", "../other"])
        .current_dir(&repo)
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("outside the selected project root"),
        "expected outside-root error, got: {stderr}"
    );
}

#[test]
fn test_skip_outside_root_flag_skips_file_outside_selected_root() {
    let parent = tempfile::TempDir::new().unwrap();
    let repo = parent.path().join("repo");
    let other = parent.path().join("other");
    std::fs::create_dir(&repo).unwrap();
    std::fs::create_dir(&repo.join(".git")).unwrap();
    std::fs::create_dir(&other).unwrap();
    std::fs::write(repo.join("inside.rs"), "fn inside_bug() {}").unwrap();
    std::fs::write(other.join("outside.rs"), "fn outside_bug() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--skip-outside-root",
            "--threshold",
            "0.0",
            "bug",
            "inside.rs",
            "../other/outside.rs",
        ])
        .current_dir(&repo)
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("inside.rs"),
        "expected inside file in results, got: {stdout}"
    );
    assert!(
        !stdout.contains("../other/outside.rs"),
        "outside-root file should have been skipped, got: {stdout}"
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Skipping 1 path(s) outside project root"),
        "expected skip warning, got: {stderr}"
    );
}

#[test]
fn test_skip_outside_root_with_explicit_files_does_not_pollute() {
    let parent = tempfile::TempDir::new().unwrap();
    let repo = parent.path().join("repo");
    let other = parent.path().join("other");
    std::fs::create_dir(&repo).unwrap();
    std::fs::create_dir(&repo.join(".git")).unwrap();
    std::fs::create_dir(&other).unwrap();
    std::fs::write(repo.join("lib.rs"), "fn library_code() {}").unwrap();
    std::fs::write(other.join("outside.rs"), "fn outside_code() {}").unwrap();

    // Search with explicit inside file + skipped outside file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--skip-outside-root",
            "--quiet",
            "--threshold",
            "0.0",
            "code",
            "lib.rs",
            "../other/outside.rs",
        ])
        .current_dir(&repo)
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("lib.rs"),
        "expected inside explicit file in results, got: {stdout}"
    );

    // Directory walk should NOT include the explicit file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--quiet", "--threshold", "0.0", "code"])
        .current_dir(&repo)
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // lib.rs should appear because the directory walk finds it and clears
    // the explicit flag
    assert!(
        stdout.contains("lib.rs"),
        "lib.rs should appear in directory search (walk clears explicit flag), got: {stdout}"
    );
    // outside.rs was skipped, never indexed
    assert!(
        !stdout.contains("outside.rs"),
        "outside file should never have been indexed, got: {stdout}"
    );
}

#[test]
fn test_reindex_without_query_repopulates_index() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("main.rs"), "fn search_target() {}\n").unwrap();

    let initial = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["search target", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        initial.status.success(),
        "initial index failed: {}",
        String::from_utf8_lossy(&initial.stderr)
    );

    let reindex = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--reindex")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        reindex.status.success(),
        "reindex failed: {}",
        String::from_utf8_lossy(&reindex.stderr)
    );

    let stats = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--stats")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        stats.status.success(),
        "stats failed: {}",
        String::from_utf8_lossy(&stats.stderr)
    );

    let stderr = String::from_utf8(stats.stderr).unwrap();
    assert!(
        stderr.contains("  Files:  1"),
        "expected repopulated index, got: {stderr}"
    );
}

#[test]
fn test_reindex_with_query_repopulates_index_and_searches() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("main.rs"), "fn search_target() {}\n").unwrap();

    let initial = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["search target", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        initial.status.success(),
        "initial index failed: {}",
        String::from_utf8_lossy(&initial.stderr)
    );

    let reindex = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--reindex", "search target"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        reindex.status.success(),
        "reindex search failed: {}",
        String::from_utf8_lossy(&reindex.stderr)
    );

    let stdout = String::from_utf8(reindex.stdout).unwrap();
    assert!(
        stdout.contains("main.rs"),
        "expected search results after reindex, got: {stdout}"
    );
}

#[test]
fn test_full_index_indexes_before_search() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(
        dir.path().join("hello.rs"),
        "fn hello() { println!(\"hello world\"); }",
    )
    .unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--threshold", "0.0", "hello world"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("hello.rs"),
        "expected hello.rs in results, got: {stdout}"
    );
}

#[test]
fn test_default_mode_uses_cached_index() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(
        dir.path().join("cached.rs"),
        "fn cached_function() { return 42; }",
    )
    .unwrap();

    // First run: build the index with --full-index
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Second run: default mode should find cached results
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--threshold", "0.0", "cached function"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("cached.rs"),
        "expected cached.rs in results, got: {stdout}"
    );
}

#[test]
fn test_default_mode_without_index_builds_index_before_search() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("new.rs"), "fn brand_new() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--threshold", "0.0", "brand new function"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("new.rs"),
        "expected new.rs in results on first run, got: {stdout}"
    );
}

#[test]
fn test_quiet_hides_indexing_status_output() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("quiet.rs"), "fn quiet_mode() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--quiet", "--threshold", "0.0", "quiet mode"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("quiet.rs"),
        "expected quiet.rs in results, got: {stdout}"
    );

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.trim().is_empty(),
        "expected no stderr output with --quiet, got: {stderr:?}"
    );
}

#[test]
fn test_first_run_status_output_is_compact() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("compact.rs"), "fn compact_output() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--threshold", "0.0", "compact output"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Loading model... done."),
        "expected compact model loading line, got: {stderr:?}"
    );
    assert!(
        stderr.contains("Indexing complete. 1 files, 1 chunks, 1 walked."),
        "expected compact indexing summary, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("Model loaded."),
        "unexpected separate model-loaded line: {stderr:?}"
    );
    assert!(
        !stderr.contains("Scanning files..."),
        "unexpected separate scanning line: {stderr:?}"
    );
    assert!(
        !stderr.contains("Found 1 files."),
        "unexpected duplicate found-files line: {stderr:?}"
    );
    assert!(
        !stderr.contains("\n\n"),
        "unexpected blank line in status output: {stderr:?}"
    );
}

// --- Embedding dimension tests ---

#[test]
fn test_search_with_non_default_embedding_dim() {
    // Simulate an index built with a remote model (e.g., 1024-dim mxbai-embed-large)
    let index = Index::open_in_memory().unwrap();
    let dim = 1024;

    // Set config with non-default dimension so vec_chunks has the right dim
    let config = IndexConfig {
        model_name: "test-remote".to_string(),
        embedding_dim: dim,
        chunk_size: 500,
        chunk_overlap: 100,
    };
    index.rebuild_for_config(&config).unwrap();

    let chunks = vec![
        Chunk {
            file_path: "main.rs".to_string(),
            text: "fn main() {}".to_string(),
            start_line: 1,
            end_line: 1,
        },
        Chunk {
            file_path: "lib.rs".to_string(),
            text: "pub fn lib() {}".to_string(),
            start_line: 1,
            end_line: 1,
        },
    ];

    let embeddings = vec![make_embedding(dim, 1.0), make_embedding(dim, 2.0)];

    index
        .upsert_file(
            "main.rs",
            "hash1",
            &chunks[0..1],
            &embeddings[0..1],
            &[false],
        )
        .unwrap();
    index
        .upsert_file(
            "lib.rs",
            "hash2",
            &chunks[1..2],
            &embeddings[1..2],
            &[false],
        )
        .unwrap();

    assert_eq!(index.chunk_count().unwrap(), 2);

    // Search should work with the correct dimension
    let results = index.search(&embeddings[0], 2, 0.0, true).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].score > 0.99); // should find itself
}

#[test]
fn test_stats_reports_holes() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();

    let index = Index::open(dir.path()).unwrap();
    let dim = EMBEDDING_DIM;
    let chunks = vec![
        Chunk {
            file_path: "mixed.rs".to_string(),
            text: "fn healthy() {}".to_string(),
            start_line: 1,
            end_line: 1,
        },
        Chunk {
            file_path: "mixed.rs".to_string(),
            text: "fn broken() {}".to_string(),
            start_line: 1,
            end_line: 1,
        },
    ];
    let embeddings = vec![make_embedding(dim, 1.0), vec![0.0; dim]];

    index
        .upsert_file("mixed.rs", "hash1", &chunks, &embeddings, &[false, true])
        .unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--stats")
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Index statistics:"),
        "expected stats header, got: {stderr}"
    );
    assert!(
        stderr.contains("Files:  1"),
        "expected file count, got: {stderr}"
    );
    assert!(
        stderr.contains("Chunks: 2"),
        "expected chunk count, got: {stderr}"
    );
    assert!(
        stderr.contains("Holes:  1"),
        "expected hole count, got: {stderr}"
    );
}

// --- Index scoping tests ---

#[test]
fn test_index_shared_across_subdirectories() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::create_dir(dir.path().join("tests")).unwrap();
    std::fs::write(
        dir.path().join("src/app.rs"),
        "fn application_startup() { initialize(); }",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tests/app_test.rs"),
        "fn test_application_startup() { assert!(true); }",
    )
    .unwrap();

    // Index src/
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only", "./src"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Index tests/
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only", "./tests"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Search from project root — should find files from both directories
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "application startup"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("src/app.rs"),
        "expected src/app.rs in results, got: {stdout}"
    );
    assert!(
        stdout.contains("tests/app_test.rs"),
        "expected tests/app_test.rs in results, got: {stdout}"
    );
}

#[test]
fn test_subdirectory_index_does_not_remove_other_dirs() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::create_dir(dir.path().join("tests")).unwrap();
    std::fs::write(
        dir.path().join("src/main.rs"),
        "fn main_entry_point() { run(); }",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tests/integration.rs"),
        "fn integration_test_suite() { verify(); }",
    )
    .unwrap();

    // Index everything from project root
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Re-index only src/ (should NOT remove tests/ entries)
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only", "./src"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Search should still find files from both directories
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--json", "--threshold", "0.0", "function"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("src/main.rs"),
        "expected src/main.rs in results after re-indexing src/, got: {stdout}"
    );
    assert!(
        stdout.contains("tests/integration.rs"),
        "expected tests/integration.rs to survive src/-only re-index, got: {stdout}"
    );
}

#[test]
fn test_explicit_file_paths_do_not_prune_other_indexed_files() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn alpha_bug() {}").unwrap();
    std::fs::write(dir.path().join("b.rs"), "fn beta_bug() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--index-only", "a.rs"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .arg("--stats")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Files:  2"),
        "explicit file indexing should not prune other cached files: {stderr}"
    );
}

#[test]
fn test_stale_file_removed_within_walked_dir() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/keep.rs"), "fn keep_this() {}").unwrap();
    std::fs::write(
        dir.path().join("src/remove.rs"),
        "fn remove_this_later() {}",
    )
    .unwrap();

    // Index src/
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only", "./src"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Delete one file and re-index
    std::fs::remove_file(dir.path().join("src/remove.rs")).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only", "./src"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Search should only find the remaining file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--json", "keep or remove"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("remove.rs"),
        "deleted file should not appear in results, got: {stdout}"
    );
}

#[test]
fn test_search_from_subdirectory_shows_relative_paths() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "fn library_function() { compute(); }",
    )
    .unwrap();

    // Index from project root
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Search from src/ — paths should be relative to cwd
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "library function"])
        .current_dir(dir.path().join("src"))
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("lib.rs"),
        "expected lib.rs in results, got: {stdout}"
    );
    assert!(
        !stdout.contains("src/lib.rs"),
        "path should be relative to cwd (src/), not project root, got: {stdout}"
    );
}

// --- Ephemeral index / explicit file path tests ---

#[test]
fn test_explicit_ignored_file_does_not_pollute_persistent_index() {
    let dir = tempfile::TempDir::new().unwrap();
    // Initialize git repo so .gitignore is respected
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();
    std::fs::write(dir.path().join("normal.rs"), "fn normal_function() {}").unwrap();
    std::fs::write(dir.path().join("secret.log"), "secret log data to search").unwrap();

    // First: search an explicit gitignored file — should find it
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "secret log",
            &dir.path().join("secret.log").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("secret.log"),
        "expected secret.log in explicit file results, got: {stdout}"
    );

    // Second: search the directory — secret.log should NOT appear
    // because it's gitignored and was only in the ephemeral index
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--quiet", "--threshold", "0.0", "secret log"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains("secret.log"),
        "gitignored file from ephemeral should not pollute persistent index, got: {stdout}"
    );
}

#[test]
fn test_explicit_file_survives_directory_walk_and_is_reusable() {
    let dir = tempfile::TempDir::new().unwrap();
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();
    std::fs::write(dir.path().join("normal.rs"), "fn normal() {}").unwrap();
    std::fs::write(dir.path().join("debug.log"), "debug log data for search").unwrap();

    // 1. Search the explicit gitignored file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "debug log",
            &dir.path().join("debug.log").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("debug.log"),
        "expected debug.log in first explicit search, got: {stdout}"
    );

    // 2. Do a directory walk — should NOT delete the explicit file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--quiet", "--threshold", "0.0", "normal function"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // 3. Search the same explicit file again — should still be cached
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "debug log",
            &dir.path().join("debug.log").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("debug.log"),
        "explicit file should still be cached after directory walk, got: {stdout}"
    );
}

#[test]
fn test_explicit_file_reindexed_when_content_changes() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    let file_path = dir.path().join("evolving.rs");

    // 1. Index with original content
    std::fs::write(&file_path, "fn original_function() { return 1; }").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "original function",
            &file_path.to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("original_function"),
        "expected original content in first search, got: {stdout}"
    );

    // 2. Change the file content (new hash)
    std::fs::write(&file_path, "fn completely_rewritten() { return 42; }").unwrap();

    // 3. Search again with the same explicit file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "rewritten",
            &file_path.to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();

    // 4. Should have re-indexed (new hash) and return the new content
    assert!(
        stdout.contains("completely_rewritten"),
        "expected updated content after re-index, got: {stdout}"
    );
    assert!(
        !stdout.contains("original_function"),
        "old content should be replaced, got: {stdout}"
    );
}

#[test]
fn test_explicit_file_returns_results() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(
        dir.path().join("target.rs"),
        "fn search_target_function() { return 42; }",
    )
    .unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "search target",
            &dir.path().join("target.rs").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("target.rs"),
        "expected target.rs in results, got: {stdout}"
    );
}

#[test]
fn test_mixed_file_and_directory_search() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/lib.rs"), "fn library_code() {}").unwrap();
    std::fs::write(dir.path().join("notes.txt"), "some important notes here").unwrap();

    // Search with a directory AND an explicit file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "code",
            &dir.path().join("src").to_string_lossy(),
            &dir.path().join("notes.txt").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // Both should appear: lib.rs from persistent, notes.txt from ephemeral
    assert!(
        stdout.contains("lib.rs"),
        "expected lib.rs from directory walk, got: {stdout}"
    );
    assert!(
        stdout.contains("notes.txt"),
        "expected notes.txt from explicit file, got: {stdout}"
    );
}

#[test]
fn test_index_only_with_dir_does_not_include_explicit_file_results() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();

    // --index-only indexes the directory into persistent cache
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--index-only"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Verify the file is in the persistent index via --stats
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--stats"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Files:  1"),
        "expected 1 file in index, got: {stderr}"
    );
}

#[test]
fn test_search_explicit_file_with_query() {
    // When a query and explicit file paths are both provided,
    // the file goes to paths (not query) and is searched ephemerally.
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(
        dir.path().join("data.txt"),
        "important data for semantic search",
    )
    .unwrap();

    let file_path = dir.path().join("data.txt");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "semantic search",
            &file_path.to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("data.txt"),
        "expected data.txt in results, got: {stdout}"
    );

    // The file is in the persistent index (with explicit flag) but gets
    // cleaned up on the next directory walk
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--stats"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("Files:  1"),
        "explicit file should be in index with explicit flag, got: {stderr}"
    );

    // A directory walk should clean up the explicit file (data.txt is not
    // gitignored, so the walker finds it and clears the explicit flag)
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--quiet", "--threshold", "0.0", "data"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());
}

#[test]
fn test_multiple_explicit_files_all_searched() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("alpha.rs"), "fn alpha_function() {}").unwrap();
    std::fs::write(dir.path().join("beta.rs"), "fn beta_function() {}").unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args([
            "--quiet",
            "--threshold",
            "0.0",
            "function",
            &dir.path().join("alpha.rs").to_string_lossy(),
            &dir.path().join("beta.rs").to_string_lossy(),
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("alpha.rs"),
        "expected alpha.rs in results, got: {stdout}"
    );
    assert!(
        stdout.contains("beta.rs"),
        "expected beta.rs in results, got: {stdout}"
    );
}
