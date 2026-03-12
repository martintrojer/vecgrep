use std::path::{Path, PathBuf};

/// Convert a walker-relative path to a project-root-relative path.
pub fn to_project_relative(walker_path: &str, cwd_suffix: &Path) -> String {
    let stripped = walker_path.strip_prefix("./").unwrap_or(walker_path);
    if cwd_suffix.as_os_str().is_empty() {
        stripped.to_string()
    } else {
        format!("{}/{}", cwd_suffix.display(), stripped)
    }
}

/// Convert a project-root-relative path to a cwd-relative path.
pub fn to_cwd_relative(project_path: &str, cwd_suffix: &Path) -> String {
    if cwd_suffix.as_os_str().is_empty() {
        return project_path.to_string();
    }
    let prefix = format!("{}/", cwd_suffix.display());
    if let Some(rest) = project_path.strip_prefix(&prefix) {
        rest.to_string()
    } else {
        make_relative(cwd_suffix, Path::new(project_path))
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
}
