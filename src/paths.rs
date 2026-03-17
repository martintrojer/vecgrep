use std::path::{Path, PathBuf};

use crate::types::SearchResult;

/// Normalize a walker-emitted path: strip leading "./" if present.
fn normalize_walker_path(walker_path: &str) -> &str {
    walker_path.strip_prefix("./").unwrap_or(walker_path)
}

/// Convert a walker-relative path to a project-root-relative path.
pub fn to_project_relative(walker_path: &str, cwd_suffix: &Path) -> String {
    let normalized = normalize_walker_path(walker_path);
    if normalized == "." || normalized.is_empty() {
        cwd_suffix.to_string_lossy().to_string()
    } else if cwd_suffix.as_os_str().is_empty() {
        normalized.to_string()
    } else {
        cwd_suffix.join(normalized).to_string_lossy().to_string()
    }
}

/// Convert a project-root-relative path to a cwd-relative path.
pub fn to_cwd_relative(project_path: &str, cwd_suffix: &Path) -> String {
    if cwd_suffix.as_os_str().is_empty() {
        return project_path.to_string();
    }
    let project = Path::new(project_path);
    if let Ok(rest) = project.strip_prefix(cwd_suffix) {
        rest.to_string_lossy().to_string()
    } else {
        make_relative(cwd_suffix, project)
    }
}

/// Check if `path` is under `prefix` (or is `prefix` itself).
pub fn is_under(path: &str, prefix: &Path) -> bool {
    let path = Path::new(path);
    path == prefix || path.starts_with(prefix)
}

/// Rewrite search result paths from project-root-relative to cwd-relative.
pub fn rewrite_results_to_cwd_relative(results: &mut [SearchResult], cwd_suffix: &Path) {
    if cwd_suffix.as_os_str().is_empty() {
        return;
    }

    for result in results {
        result.chunk.file_path = to_cwd_relative(&result.chunk.file_path, cwd_suffix);
    }
}

/// Compute a relative path from `from` to `to`, where both are relative to the same root.
fn make_relative(from: &Path, to: &Path) -> String {
    let from_comps: Vec<_> = from.components().collect();
    let to_comps: Vec<_> = to.components().collect();

    let common = from_comps
        .iter()
        .zip(to_comps.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let ups = from_comps.len() - common;
    let mut result = PathBuf::new();
    for _ in 0..ups {
        result.push("..");
    }
    for comp in &to_comps[common..] {
        result.push(comp);
    }
    result.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- to_project_relative ---

    #[test]
    fn test_project_relative_empty_suffix_strips_dot_slash() {
        assert_eq!(to_project_relative("./main.rs", Path::new("")), "main.rs");
    }

    #[test]
    fn test_project_relative_empty_suffix_no_dot_slash() {
        assert_eq!(
            to_project_relative("src/main.rs", Path::new("")),
            "src/main.rs"
        );
    }

    #[test]
    fn test_project_relative_with_suffix() {
        assert_eq!(
            to_project_relative("./main.rs", Path::new("src")),
            "src/main.rs"
        );
    }

    #[test]
    fn test_project_relative_nested_suffix() {
        assert_eq!(
            to_project_relative("./mod.rs", Path::new("src/deep")),
            "src/deep/mod.rs"
        );
    }

    #[test]
    fn test_project_relative_dot_at_root() {
        assert_eq!(to_project_relative(".", Path::new("")), "");
    }

    #[test]
    fn test_project_relative_dot_in_subdir() {
        assert_eq!(to_project_relative(".", Path::new("src")), "src");
    }

    // --- to_cwd_relative ---

    #[test]
    fn test_cwd_relative_at_root() {
        assert_eq!(to_cwd_relative("src/main.rs", Path::new("")), "src/main.rs");
    }

    #[test]
    fn test_cwd_relative_strips_prefix() {
        assert_eq!(to_cwd_relative("src/main.rs", Path::new("src")), "main.rs");
    }

    #[test]
    fn test_cwd_relative_nested() {
        assert_eq!(
            to_cwd_relative("src/deep/mod.rs", Path::new("src/deep")),
            "mod.rs"
        );
    }

    #[test]
    fn test_cwd_relative_outside_subtree() {
        assert_eq!(
            to_cwd_relative("lib/foo.rs", Path::new("src")),
            "../lib/foo.rs"
        );
    }

    #[test]
    fn test_cwd_relative_root_file_from_subdir() {
        assert_eq!(
            to_cwd_relative("README.md", Path::new("src")),
            "../README.md"
        );
    }

    #[test]
    fn test_cwd_relative_sibling_deep() {
        assert_eq!(
            to_cwd_relative("src/b/foo.rs", Path::new("src/a")),
            "../b/foo.rs"
        );
    }

    // --- is_under ---

    #[test]
    fn test_is_under_direct_child() {
        assert!(is_under("src/main.rs", Path::new("src")));
    }

    #[test]
    fn test_is_under_exact_match() {
        assert!(is_under("src/main.rs", Path::new("src/main.rs")));
    }

    #[test]
    fn test_is_under_not_related() {
        assert!(!is_under("docs/guide.md", Path::new("src")));
    }

    #[test]
    fn test_is_under_trailing_slash() {
        assert!(is_under("src/main.rs", Path::new("src/")));
    }

    #[test]
    fn test_is_under_rejects_prefix_substring() {
        // "src-old/main.rs" should NOT match scope "src" — Path::starts_with
        // is component-aware, unlike string prefix matching.
        assert!(!is_under("src-old/main.rs", Path::new("src")));
        assert!(!is_under("srclib/foo.rs", Path::new("src")));
    }

    // --- rewrite ---

    #[test]
    fn test_rewrite_results_to_cwd_relative() {
        let mut results = vec![SearchResult {
            chunk: crate::types::Chunk {
                file_path: "src/main.rs".to_string(),
                text: "fn main() {}".to_string(),
                start_line: 1,
                end_line: 1,
            },
            score: 0.9,
        }];

        rewrite_results_to_cwd_relative(&mut results, Path::new("src"));

        assert_eq!(results[0].chunk.file_path, "main.rs");
    }
}
