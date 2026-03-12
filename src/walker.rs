use anyhow::Result;
use ignore::WalkBuilder;
use std::path::Path;

/// Discovered file with its content.
pub struct WalkedFile {
    /// Path relative to the search root.
    pub rel_path: String,
    /// File content (UTF-8).
    pub content: String,
}

/// Options for file walking, mapped from CLI flags.
pub struct WalkOptions<'a> {
    pub file_types: &'a Option<Vec<String>>,
    pub file_types_not: &'a Option<Vec<String>>,
    pub globs: &'a Option<Vec<String>>,
    pub hidden: bool,
    pub follow: bool,
    pub no_ignore: bool,
    pub max_depth: Option<usize>,
}

/// Walk the given paths, respecting .gitignore and filters.
pub fn walk_paths(paths: &[String], opts: &WalkOptions) -> Result<Vec<WalkedFile>> {
    let mut files = Vec::new();

    for search_path in paths {
        let search_path = Path::new(search_path);

        if search_path.is_file() {
            if let Some(f) = read_file(search_path)? {
                files.push(f);
            }
            continue;
        }

        let mut builder = WalkBuilder::new(search_path);
        builder
            .hidden(!opts.hidden)
            .follow_links(opts.follow)
            .git_ignore(!opts.no_ignore)
            .git_global(!opts.no_ignore)
            .git_exclude(!opts.no_ignore)
            .ignore(!opts.no_ignore);

        if let Some(depth) = opts.max_depth {
            builder.max_depth(Some(depth));
        }

        // Add type filters (select and negate)
        if opts.file_types.is_some() || opts.file_types_not.is_some() {
            let mut type_builder = ignore::types::TypesBuilder::new();
            type_builder.add_defaults();
            if let Some(types) = opts.file_types {
                for t in types {
                    type_builder.select(t);
                }
            }
            if let Some(types) = opts.file_types_not {
                for t in types {
                    type_builder.negate(t);
                }
            }
            let types_matcher = type_builder.build().map_err(|e| anyhow::anyhow!("{}", e))?;
            builder.types(types_matcher);
        }

        // Add glob filters
        if let Some(glob_patterns) = opts.globs {
            let mut overrides = ignore::overrides::OverrideBuilder::new(search_path);
            for pattern in glob_patterns {
                overrides
                    .add(pattern)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            }
            let ov = overrides.build().map_err(|e| anyhow::anyhow!("{}", e))?;
            builder.overrides(ov);
        }

        for entry in builder.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Walk error: {}", e);
                    continue;
                }
            };

            // Skip directories
            if entry.file_type().is_none_or(|ft| !ft.is_file()) {
                continue;
            }

            let path = entry.path();
            if let Some(f) = read_file(path)? {
                files.push(f);
            }
        }
    }

    Ok(files)
}

/// Print all supported file types (from the ignore crate).
pub fn print_type_list() {
    let mut type_builder = ignore::types::TypesBuilder::new();
    type_builder.add_defaults();
    let types = type_builder.build().unwrap();
    for def in types.definitions() {
        println!("{}: {}", def.name(), def.globs().join(", "));
    }
}

fn read_file(path: &Path) -> Result<Option<WalkedFile>> {
    let rel_path = path.to_string_lossy().to_string();

    match std::fs::read_to_string(path) {
        Ok(content) => {
            if content.is_empty() {
                return Ok(None);
            }
            Ok(Some(WalkedFile { rel_path, content }))
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::InvalidData {
                tracing::debug!("Skipping binary file: {}", rel_path);
            } else {
                tracing::warn!("Failed to read {}: {}", rel_path, e);
            }
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn default_opts<'a>() -> WalkOptions<'a> {
        WalkOptions {
            file_types: &None,
            file_types_not: &None,
            globs: &None,
            hidden: false,
            follow: false,
            no_ignore: false,
            max_depth: None,
        }
    }

    #[test]
    fn test_walk_single_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, "hello world").unwrap();

        let paths = vec![file.to_string_lossy().to_string()];
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].content, "hello world");
    }

    #[test]
    fn test_walk_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "aaa").unwrap();
        std::fs::write(dir.path().join("b.txt"), "bbb").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/c.txt"), "ccc").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_walk_skips_binary() {
        let dir = TempDir::new().unwrap();
        let bin_path = dir.path().join("data.bin");
        let mut f = std::fs::File::create(&bin_path).unwrap();
        f.write_all(&[0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81]).unwrap();

        // Also create a valid text file to ensure we get at least one result
        std::fs::write(dir.path().join("valid.txt"), "text content").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();

        let names: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("valid.txt")));
        // Binary file should not appear (either skipped by read or by ignore crate)
        assert!(!names.iter().any(|n| n.contains("data.bin")));
    }

    #[test]
    fn test_walk_skips_empty() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();
        std::fs::write(dir.path().join("notempty.txt"), "content").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].rel_path.contains("notempty.txt"));
    }

    #[test]
    fn test_walk_respects_gitignore() {
        let dir = TempDir::new().unwrap();
        // Initialize a git repo so .gitignore is respected
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        std::fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();
        std::fs::write(dir.path().join("app.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("debug.log"), "log data").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();

        let names: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("app.rs")));
        assert!(!names.iter().any(|n| n.contains("debug.log")));
    }

    #[test]
    fn test_walk_hidden_flag() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "secret").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "hello").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];

        // Without hidden flag: should not find hidden files
        let opts = default_opts();
        let files = walk_paths(&paths, &opts).unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(!names.iter().any(|n| n.contains(".hidden")));

        // With hidden flag: should find hidden files
        let opts = WalkOptions {
            hidden: true,
            ..default_opts()
        };
        let files = walk_paths(&paths, &opts).unwrap();
        let names: Vec<&str> = files.iter().map(|f| f.rel_path.as_str()).collect();
        assert!(names.iter().any(|n| n.contains(".hidden")));
    }

    #[test]
    fn test_walk_max_depth() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("top.txt"), "top level").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/deep.txt"), "deep").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let opts = WalkOptions {
            max_depth: Some(1),
            ..default_opts()
        };
        let files = walk_paths(&paths, &opts).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].rel_path.contains("top.txt"));
    }

    #[test]
    fn test_walk_type_filter() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("readme.md"), "# Hello").unwrap();
        std::fs::write(dir.path().join("data.json"), "{}").unwrap();

        let paths = vec![dir.path().to_string_lossy().to_string()];
        let rust_type = Some(vec!["rust".to_string()]);
        let opts = WalkOptions {
            file_types: &rust_type,
            file_types_not: &None,
            globs: &None,
            hidden: false,
            follow: false,
            no_ignore: false,
            max_depth: None,
        };
        let files = walk_paths(&paths, &opts).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].rel_path.contains("code.rs"));
    }
}
