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

/// Walk the given paths, respecting .gitignore and filters.
/// Returns an iterator-like Vec of files with their content.
pub fn walk_paths(
    paths: &[String],
    file_types: &Option<Vec<String>>,
    globs: &Option<Vec<String>>,
) -> Result<Vec<WalkedFile>> {
    let mut files = Vec::new();

    for search_path in paths {
        let search_path = Path::new(search_path);

        if search_path.is_file() {
            // Single file
            if let Some(f) = read_file(search_path)? {
                files.push(f);
            }
            continue;
        }

        let mut builder = WalkBuilder::new(search_path);
        builder
            .hidden(true) // respect hidden/dot files
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true);

        // Add type filters
        if let Some(types) = file_types {
            let mut type_builder = ignore::types::TypesBuilder::new();
            type_builder.add_defaults();
            for t in types {
                type_builder.select(t);
            }
            let types_matcher = type_builder.build().map_err(|e| anyhow::anyhow!("{}", e))?;
            builder.types(types_matcher);
        }

        // Add glob filters
        if let Some(glob_patterns) = globs {
            for pattern in glob_patterns {
                builder.add_custom_ignore_filename("");
                // Use overrides for glob matching
                let mut overrides = ignore::overrides::OverrideBuilder::new(search_path);
                overrides
                    .add(pattern)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let ov = overrides.build().map_err(|e| anyhow::anyhow!("{}", e))?;
                builder.overrides(ov);
            }
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

fn read_file(path: &Path) -> Result<Option<WalkedFile>> {
    let rel_path = path.to_string_lossy().to_string();

    // Read as UTF-8, skip binary/non-UTF-8 files
    match std::fs::read_to_string(path) {
        Ok(content) => {
            // Skip empty files
            if content.is_empty() {
                return Ok(None);
            }
            Ok(Some(WalkedFile { rel_path, content }))
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::InvalidData {
                // Binary file, skip silently
                tracing::debug!("Skipping binary file: {}", rel_path);
            } else {
                tracing::warn!("Failed to read {}: {}", rel_path, e);
            }
            Ok(None)
        }
    }
}
