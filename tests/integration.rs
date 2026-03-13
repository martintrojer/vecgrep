use vecgrep::embedder::EMBEDDING_DIM;
use vecgrep::index::Index;
use vecgrep::search;
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

    // Create some chunks with distinct embeddings
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
        .upsert_file("main.rs", "hash_main", &chunks[0..2], &embeddings[0..2])
        .unwrap();

    // Store lib.rs chunk
    index
        .upsert_file("lib.rs", "hash_lib", &chunks[2..3], &embeddings[2..3])
        .unwrap();

    // Load all and search
    let (loaded_chunks, embedding_matrix): (Vec<Chunk>, ndarray::Array2<f32>) =
        index.load_all().unwrap();
    assert_eq!(loaded_chunks.len(), 3);
    assert_eq!(embedding_matrix.nrows(), 3);
    assert_eq!(embedding_matrix.ncols(), dim);

    // Search with the first embedding as query — should find itself as top match
    let results = search::search(&embeddings[0], &embedding_matrix, 3, 0.0, &loaded_chunks);
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
        .upsert_file("a.rs", "hash_a_v1", &chunk_a, &emb_a)
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
        .upsert_file("b.rs", "hash_b", &chunk_b, &emb_b)
        .unwrap();

    // Verify both are present
    let (chunks, _): (Vec<Chunk>, ndarray::Array2<f32>) = index.load_all().unwrap();
    assert_eq!(chunks.len(), 2);

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
        .upsert_file("a.rs", "hash_a_v2", &chunk_a_v2, &emb_a_v2)
        .unwrap();

    // Verify: still 2 chunks total, a.rs has new content
    let (chunks, _): (Vec<Chunk>, ndarray::Array2<f32>) = index.load_all().unwrap();
    assert_eq!(chunks.len(), 2);

    let a_chunk = chunks.iter().find(|c| c.file_path == "a.rs").unwrap();
    assert_eq!(a_chunk.text, "fn a_modified() {}");

    let b_chunk = chunks.iter().find(|c| c.file_path == "b.rs").unwrap();
    assert_eq!(b_chunk.text, "fn b() {}");

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
    // No .git, .hg, .jj, or .vecgrep — falls back to cwd

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
fn test_full_index_indexes_before_search() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(
        dir.path().join("hello.rs"),
        "fn hello() { println!(\"hello world\"); }",
    )
    .unwrap();

    // --full-index on a fresh repo: should index then find the file
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["--full-index", "hello world"])
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

    // Second run: default mode should find cached results instantly
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["cached function"])
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
fn test_default_mode_without_index_returns_no_results() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join("new.rs"), "fn brand_new() {}").unwrap();

    // Default mode on a fresh repo: no cached index, so no results
    // (files are indexed in the background for next time)
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_vecgrep"))
        .args(["brand new function"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    // Exit code 1 = no matches found
    assert_eq!(output.status.code(), Some(1));
}

// --- Index scoping tests ---

#[test]
fn test_index_shared_across_subdirectories() {
    // Index from src/, then search from tests/ — both should be in the same index
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
    // Index the whole project, then re-index only src/ — tests/ entries should survive
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
    // Use --threshold 0.0 to ensure all indexed files appear regardless of score
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
fn test_stale_file_removed_within_walked_dir() {
    // Index src/ with two files, delete one, re-index src/ — deleted file should be gone
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
    // Should show "lib.rs" not "src/lib.rs" since we're running from src/
    assert!(
        stdout.contains("lib.rs"),
        "expected lib.rs in results, got: {stdout}"
    );
    assert!(
        !stdout.contains("src/lib.rs"),
        "path should be relative to cwd (src/), not project root, got: {stdout}"
    );
}
