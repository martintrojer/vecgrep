use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::cli::{
    Args, ColorChoice, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_INDEX_WARN_THRESHOLD,
    DEFAULT_THRESHOLD, DEFAULT_TOP_K,
};
use crate::root::resolve_input_path;
use crate::{config, output};

#[derive(Debug)]
pub enum StaleRemovalScope {
    Prefix(PathBuf),
    None,
}

#[derive(Debug)]
pub struct PathPlan {
    pub project_root: PathBuf,
    pub cwd_suffix: PathBuf,
    pub stale_removal_scope: StaleRemovalScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Cli,
    Interactive,
    Serve,
}

pub struct Invocation {
    pub args: Args,
    pub path_plan: PathPlan,
    pub query: String,
    pub run_mode: RunMode,
    pub color_choice: termcolor::ColorChoice,
}

#[derive(Clone, Copy)]
pub struct CliOutputContext<'a> {
    pub color_choice: termcolor::ColorChoice,
    pub cwd_suffix: &'a Path,
    pub quiet: bool,
    pub root: &'a str,
}

/// Classify input paths as admitted/skipped/rejected in one pass.
pub fn admit_paths(args: Args, cwd: &Path, project_root: &Path) -> Result<(Args, PathPlan)> {
    let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let cwd_suffix = cwd_canon
        .strip_prefix(project_root)
        .unwrap_or(Path::new(""))
        .to_path_buf();

    let mut admitted = Vec::new();
    let mut outside = Vec::new();
    let mut single_admitted_dir: Option<PathBuf> = None;

    for input in &args.paths {
        let absolute = resolve_input_path(cwd, input);
        if absolute.starts_with(project_root) {
            if args.paths.len() == 1 && absolute.is_dir() {
                single_admitted_dir = Some(absolute);
            }
            admitted.push(input.clone());
        } else {
            outside.push(input.clone());
        }
    }

    if !outside.is_empty() && !args.skip_outside_root {
        anyhow::bail!(
            "Path '{}' is outside the selected project root '{}'. Run vecgrep from that project, invoke it separately per root, or pass --skip-outside-root to ignore such paths.",
            outside[0],
            project_root.display()
        );
    }
    if !outside.is_empty() && args.skip_outside_root && !args.quiet {
        eprintln!(
            "Skipping {} path(s) outside project root {}.",
            outside.len(),
            project_root.display()
        );
    }
    if admitted.is_empty() {
        anyhow::bail!(
            "All provided paths are outside the selected project root '{}'.",
            project_root.display()
        );
    }

    let stale_removal_scope = match single_admitted_dir {
        Some(dir) => {
            let walk_prefix = dir
                .strip_prefix(project_root)
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            StaleRemovalScope::Prefix(walk_prefix)
        }
        None => StaleRemovalScope::None,
    };

    let args = Args {
        paths: admitted,
        ..args
    };
    let plan = PathPlan {
        project_root: project_root.to_path_buf(),
        cwd_suffix,
        stale_removal_scope,
    };
    Ok((args, plan))
}

fn determine_run_mode(args: &Args) -> RunMode {
    if args.serve {
        RunMode::Serve
    } else if args.interactive {
        RunMode::Interactive
    } else {
        RunMode::Cli
    }
}

/// Merge CLI args with config file values: cli > config > hardcoded defaults.
pub fn resolve_config(args: &mut Args, config: &config::Config) {
    // Value fields: cli.or(config).or(default)
    args.top_k = args.top_k.or(config.top_k).or(Some(DEFAULT_TOP_K));
    args.threshold = args
        .threshold
        .or(config.threshold)
        .or(Some(DEFAULT_THRESHOLD));
    args.chunk_size = args
        .chunk_size
        .or(config.chunk_size)
        .or(Some(DEFAULT_CHUNK_SIZE));
    args.chunk_overlap = args
        .chunk_overlap
        .or(config.chunk_overlap)
        .or(Some(DEFAULT_CHUNK_OVERLAP));
    args.index_warn_threshold = args
        .index_warn_threshold
        .or(config.index_warn_threshold)
        .or(Some(DEFAULT_INDEX_WARN_THRESHOLD));

    // Option fields: cli.or(config)
    args.embedder_url = args
        .embedder_url
        .take()
        .or_else(|| config.embedder_url.clone());
    args.embedder_model = args
        .embedder_model
        .take()
        .or_else(|| config.embedder_model.clone());
    args.max_depth = args.max_depth.or(config.max_depth);

    // Bool flags: CLI flag || config value
    args.full_index = args.full_index || config.full_index.unwrap_or(false);
    args.hidden = args.hidden || config.hidden.unwrap_or(false);
    args.follow = args.follow || config.follow.unwrap_or(false);
    args.no_ignore = args.no_ignore || config.no_ignore.unwrap_or(false);
    args.quiet = args.quiet || config.quiet.unwrap_or(false);

    // Ignore files: additive merge
    if let Some(ref config_files) = config.ignore_files {
        let cli_files = args.ignore_file.get_or_insert_with(Vec::new);
        for f in config_files {
            if !cli_files.contains(f) {
                cli_files.push(f.clone());
            }
        }
    }

    // --pretty/-p is an alias for --color=always
    if args.pretty && args.color.is_none() {
        args.color = Some(ColorChoice::Always);
    }

    // Color: cli.or(config parsed).or(Auto)
    if args.color.is_none() {
        args.color = config.color.as_deref().and_then(|c| match c {
            "always" => Some(ColorChoice::Always),
            "never" => Some(ColorChoice::Never),
            _ => None,
        });
    }
    if args.color.is_none() {
        args.color = Some(ColorChoice::Auto);
    }
}

pub fn resolve_invocation(mut args: Args, cwd: &Path, project_root: &Path) -> Result<Invocation> {
    let config = config::load_config(project_root);
    resolve_config(&mut args, &config);
    let (args, path_plan) = admit_paths(args, cwd, project_root)?;
    let query = args.query.clone().unwrap_or_default();
    let run_mode = determine_run_mode(&args);
    let color_choice = output::resolve_color_choice(args.color.as_ref().unwrap());

    Ok(Invocation {
        args,
        path_plan,
        query,
        run_mode,
        color_choice,
    })
}

pub fn capped_chunk_size(chunk_size: usize, context_tokens: Option<usize>) -> usize {
    match context_tokens {
        Some(ctx) => chunk_size.min(ctx),
        None => chunk_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tempfile::TempDir;

    fn parse_args(argv: &[&str]) -> Args {
        Args::parse_from(argv)
    }

    #[test]
    fn test_admit_paths_sets_prefix_scope_for_single_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let src = dir.path().join("src").display().to_string();

        let args = parse_args(&["vecgrep", "needle", &src]);
        let (result_args, plan) = admit_paths(args, &cwd, &cwd).unwrap();

        assert_eq!(result_args.paths, vec![src]);
        match plan.stale_removal_scope {
            StaleRemovalScope::Prefix(ref prefix) => assert_eq!(prefix, Path::new("src")),
            _ => panic!("expected prefix stale removal scope"),
        }
    }

    #[test]
    fn test_admit_paths_sets_no_scope_for_single_file() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join("lib.rs"), "fn main() {}").unwrap();
        let cwd = dir.path().canonicalize().unwrap();

        let args = parse_args(&["vecgrep", "needle", "lib.rs"]);
        let (_, plan) = admit_paths(args, &cwd, &cwd).unwrap();

        assert!(matches!(plan.stale_removal_scope, StaleRemovalScope::None));
    }

    #[test]
    fn test_admit_paths_rejects_all_outside_paths() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();
        let outside = TempDir::new().unwrap();

        let outside_path = outside.path().join("elsewhere.rs").display().to_string();
        let args = parse_args(&["vecgrep", "needle", &outside_path]);

        let err = admit_paths(args, &cwd, &cwd).unwrap_err();
        assert!(err
            .to_string()
            .contains("outside the selected project root"));
    }

    #[test]
    fn test_determine_run_mode_prefers_expected_mode() {
        let serve_args = parse_args(&["vecgrep", "--serve"]);
        assert_eq!(determine_run_mode(&serve_args), RunMode::Serve);

        let interactive_args = parse_args(&["vecgrep", "--interactive"]);
        assert_eq!(determine_run_mode(&interactive_args), RunMode::Interactive);

        let index_only_args = parse_args(&["vecgrep", "--index-only"]);
        assert_eq!(determine_run_mode(&index_only_args), RunMode::Cli);
    }

    #[test]
    fn test_cli_args_carry_explicit_values() {
        let args = parse_args(&[
            "vecgrep",
            "--top-k",
            "7",
            "--threshold",
            "0.45",
            "--chunk-size",
            "123",
            "--chunk-overlap",
            "17",
            "--full-index",
            "--quiet",
            "needle",
        ]);

        assert_eq!(args.chunk_size, Some(123));
        assert_eq!(args.chunk_overlap, Some(17));
        assert!(args.full_index);
        assert!(args.quiet);
        assert_eq!(args.top_k, Some(7));
        assert_eq!(args.threshold, Some(0.45));
    }

    #[test]
    fn test_resolve_config_sets_bool_and_color_when_cli_omits_them() {
        let mut args = parse_args(&["vecgrep", "needle"]);
        let config = config::Config {
            hidden: Some(true),
            follow: Some(true),
            no_ignore: Some(true),
            quiet: Some(true),
            full_index: Some(true),
            color: Some("always".to_string()),
            ..Default::default()
        };

        resolve_config(&mut args, &config);

        assert!(args.hidden);
        assert!(args.follow);
        assert!(args.no_ignore);
        assert!(args.quiet);
        assert!(args.full_index);
        assert_eq!(args.color, Some(ColorChoice::Always));
    }

    #[test]
    fn test_resolve_config_does_not_override_explicit_cli_color() {
        let mut args = parse_args(&["vecgrep", "--color", "never", "needle"]);
        let config = config::Config {
            color: Some("always".to_string()),
            ..Default::default()
        };

        resolve_config(&mut args, &config);

        assert_eq!(args.color, Some(ColorChoice::Never));
    }

    #[test]
    fn test_resolve_config_applies_hardcoded_defaults() {
        let mut args = parse_args(&["vecgrep", "needle"]);
        let config = config::Config::default();

        resolve_config(&mut args, &config);

        assert_eq!(args.top_k, Some(DEFAULT_TOP_K));
        assert_eq!(args.threshold, Some(DEFAULT_THRESHOLD));
        assert_eq!(args.chunk_size, Some(DEFAULT_CHUNK_SIZE));
        assert_eq!(args.chunk_overlap, Some(DEFAULT_CHUNK_OVERLAP));
        assert_eq!(
            args.index_warn_threshold,
            Some(DEFAULT_INDEX_WARN_THRESHOLD)
        );
        assert_eq!(args.color, Some(ColorChoice::Auto));
        assert!(!args.quiet);
        assert!(!args.hidden);
    }

    #[test]
    fn test_resolve_invocation_applies_project_config_when_cli_omits_flag() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "top_k = 42\nthreshold = 0.15\nquiet = true\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&["vecgrep", "needle"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.args.top_k, Some(42));
        assert_eq!(invocation.args.threshold, Some(0.15));
        assert!(invocation.args.quiet);
    }

    #[test]
    fn test_resolve_invocation_keeps_cli_values_over_project_config() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "top_k = 42\nthreshold = 0.15\nquiet = true\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&["vecgrep", "--top-k", "7", "--threshold", "0.6", "needle"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.args.top_k, Some(7));
        assert_eq!(invocation.args.threshold, Some(0.6));
        assert!(invocation.args.quiet);
    }

    #[test]
    fn test_resolve_invocation_merges_ignore_files_additively() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join(".vecgrep")).unwrap();
        std::fs::write(
            dir.path().join(".vecgrep/config.toml"),
            "ignore_files = [\"from-config.ignore\", \"shared.ignore\"]\n",
        )
        .unwrap();

        let cwd = dir.path().canonicalize().unwrap();
        let args = parse_args(&[
            "vecgrep",
            "--ignore-file",
            "from-cli.ignore",
            "--ignore-file",
            "shared.ignore",
            "needle",
        ]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(
            invocation.args.ignore_file,
            Some(vec![
                "from-cli.ignore".to_string(),
                "shared.ignore".to_string(),
                "from-config.ignore".to_string(),
            ])
        );
    }

    #[test]
    fn test_resolve_invocation_uses_empty_query_for_serve_mode() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let cwd = dir.path().canonicalize().unwrap();

        let args = parse_args(&["vecgrep", "--serve"]);
        let invocation = resolve_invocation(args, &cwd, &cwd).unwrap();

        assert_eq!(invocation.run_mode, RunMode::Serve);
        assert!(invocation.query.is_empty());
    }

    #[test]
    fn test_capped_chunk_size_reduces() {
        assert_eq!(capped_chunk_size(256, Some(256)), 256);
        assert_eq!(capped_chunk_size(200, Some(256)), 200);
        assert_eq!(capped_chunk_size(500, None), 500);
        assert_eq!(capped_chunk_size(500, Some(256)), 256);
    }
}
