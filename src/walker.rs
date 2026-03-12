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
