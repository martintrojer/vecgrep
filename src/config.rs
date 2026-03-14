use serde::Deserialize;
use std::path::PathBuf;

/// User configuration loaded from `~/.config/vecgrep/config.toml`.
/// All fields are optional — absent means "use CLI default".
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub embedder_url: Option<String>,
    pub embedder_model: Option<String>,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub context: Option<usize>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub full_index: Option<bool>,
    pub hidden: Option<bool>,
    pub follow: Option<bool>,
    pub no_ignore: Option<bool>,
    pub max_depth: Option<usize>,
    pub color: Option<String>,
    pub quiet: Option<bool>,
    pub index_warn_threshold: Option<usize>,
}

/// Return the global config path (`~/.config/vecgrep/config.toml`).
pub fn global_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("vecgrep").join("config.toml"))
}

/// Return the project config path (`<project_root>/.vecgrep/config.toml`).
pub fn project_config_path(project_root: &std::path::Path) -> PathBuf {
    project_root.join(".vecgrep").join("config.toml")
}

fn load_config_file(path: &std::path::Path) -> Option<Config> {
    let content = std::fs::read_to_string(path).ok()?;
    match toml::from_str(&content) {
        Ok(config) => Some(config),
        Err(e) => {
            eprintln!("Warning: failed to parse {}: {}", path.display(), e);
            None
        }
    }
}

/// Merge two configs. `override_config` values take precedence over `base`.
fn merge(base: Config, override_config: Config) -> Config {
    Config {
        embedder_url: override_config.embedder_url.or(base.embedder_url),
        embedder_model: override_config.embedder_model.or(base.embedder_model),
        top_k: override_config.top_k.or(base.top_k),
        threshold: override_config.threshold.or(base.threshold),
        context: override_config.context.or(base.context),
        chunk_size: override_config.chunk_size.or(base.chunk_size),
        chunk_overlap: override_config.chunk_overlap.or(base.chunk_overlap),
        full_index: override_config.full_index.or(base.full_index),
        hidden: override_config.hidden.or(base.hidden),
        follow: override_config.follow.or(base.follow),
        no_ignore: override_config.no_ignore.or(base.no_ignore),
        max_depth: override_config.max_depth.or(base.max_depth),
        color: override_config.color.or(base.color),
        quiet: override_config.quiet.or(base.quiet),
        index_warn_threshold: override_config
            .index_warn_threshold
            .or(base.index_warn_threshold),
    }
}

/// Load config with precedence: project (.vecgrep/config.toml) > global (~/.config/vecgrep/config.toml).
/// Returns default config if neither file exists.
pub fn load_config(project_root: &std::path::Path) -> Config {
    let global = global_config_path().and_then(|p| load_config_file(&p));
    let project = load_config_file(&project_config_path(project_root));

    match (global, project) {
        (Some(g), Some(p)) => merge(g, p),
        (Some(g), None) => g,
        (None, Some(p)) => p,
        (None, None) => Config::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
            embedder_url = "http://localhost:11434/v1/embeddings"
            embedder_model = "mxbai-embed-large"
            top_k = 20
            threshold = 0.25
            context = 5
            chunk_size = 400
            chunk_overlap = 80
            full_index = true
            hidden = true
            follow = false
            no_ignore = false
            max_depth = 10
            color = "always"
            quiet = false
            index_warn_threshold = 500
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(
            config.embedder_url.as_deref(),
            Some("http://localhost:11434/v1/embeddings")
        );
        assert_eq!(config.embedder_model.as_deref(), Some("mxbai-embed-large"));
        assert_eq!(config.top_k, Some(20));
        assert_eq!(config.threshold, Some(0.25));
        assert_eq!(config.context, Some(5));
        assert_eq!(config.chunk_size, Some(400));
        assert_eq!(config.chunk_overlap, Some(80));
        assert_eq!(config.full_index, Some(true));
        assert_eq!(config.hidden, Some(true));
        assert_eq!(config.max_depth, Some(10));
        assert_eq!(config.color.as_deref(), Some("always"));
    }

    #[test]
    fn test_parse_partial_config() {
        let toml = r#"
            top_k = 5
            threshold = 0.4
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.top_k, Some(5));
        assert_eq!(config.threshold, Some(0.4));
        assert!(config.embedder_url.is_none());
        assert!(config.hidden.is_none());
    }

    #[test]
    fn test_parse_empty_config() {
        let config: Config = toml::from_str("").unwrap();
        assert!(config.top_k.is_none());
        assert!(config.embedder_url.is_none());
    }

    #[test]
    fn test_invalid_toml_returns_default() {
        let result: Result<Config, _> = toml::from_str("not valid toml {{{{");
        assert!(result.is_err());
        // load_config() would return Config::default() in this case
    }

    #[test]
    fn test_unknown_fields_ignored() {
        let toml = r#"
            top_k = 5
            unknown_field = "ignored"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.top_k, Some(5));
    }

    #[test]
    fn test_merge_project_overrides_global() {
        let global: Config = toml::from_str(
            r#"
            top_k = 10
            threshold = 0.3
            embedder_url = "http://global:11434/v1/embeddings"
            embedder_model = "global-model"
        "#,
        )
        .unwrap();

        let project: Config = toml::from_str(
            r#"
            top_k = 20
            embedder_model = "project-model"
        "#,
        )
        .unwrap();

        let merged = merge(global, project);
        assert_eq!(merged.top_k, Some(20)); // project wins
        assert_eq!(merged.threshold, Some(0.3)); // global (project absent)
        assert_eq!(
            merged.embedder_url.as_deref(),
            Some("http://global:11434/v1/embeddings")
        ); // global (project absent)
        assert_eq!(merged.embedder_model.as_deref(), Some("project-model")); // project wins
    }

    #[test]
    fn test_merge_with_empty() {
        let config: Config = toml::from_str("top_k = 5").unwrap();
        let empty = Config::default();

        let merged = merge(empty, config);
        assert_eq!(merged.top_k, Some(5));
        assert!(merged.threshold.is_none());
    }

    #[test]
    fn test_load_from_project_root() {
        let dir = tempfile::TempDir::new().unwrap();
        let vecgrep_dir = dir.path().join(".vecgrep");
        std::fs::create_dir_all(&vecgrep_dir).unwrap();
        std::fs::write(
            vecgrep_dir.join("config.toml"),
            "top_k = 42\nhidden = true\n",
        )
        .unwrap();

        let config = load_config(dir.path());
        assert_eq!(config.top_k, Some(42));
        assert_eq!(config.hidden, Some(true));
    }
}
