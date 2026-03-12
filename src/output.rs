use std::collections::BTreeMap;
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

use std::io::IsTerminal;

use crate::cli;
use crate::types::SearchResult;

pub fn resolve_color_choice(color: &cli::ColorChoice) -> ColorChoice {
    match color {
        cli::ColorChoice::Always => ColorChoice::Always,
        cli::ColorChoice::Never => ColorChoice::Never,
        cli::ColorChoice::Auto => {
            if std::io::stdout().is_terminal() {
                ColorChoice::Auto
            } else {
                ColorChoice::Never
            }
        }
    }
}

/// Print search results in colored ripgrep-style format.
pub fn print_results(
    results: &[SearchResult],
    _context_lines: usize,
    color: ColorChoice,
) -> std::io::Result<()> {
    let mut stdout = StandardStream::stdout(color);

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            writeln!(stdout)?;
        }
        print_result(&mut stdout, result)?;
    }

    Ok(())
}

fn print_result(stdout: &mut StandardStream, result: &SearchResult) -> std::io::Result<()> {
    // File path in magenta
    stdout.set_color(ColorSpec::new().set_fg(Some(Color::Magenta)).set_bold(true))?;
    write!(stdout, "{}", result.chunk.file_path)?;
    stdout.reset()?;

    // Line range in green
    stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
    write!(
        stdout,
        ":{}:{}",
        result.chunk.start_line, result.chunk.end_line
    )?;
    stdout.reset()?;

    // Score with color based on value
    let score_color = score_to_color(result.score);
    write!(stdout, " [")?;
    stdout.set_color(ColorSpec::new().set_fg(Some(score_color)).set_bold(true))?;
    write!(stdout, "{:.3}", result.score)?;
    stdout.reset()?;
    writeln!(stdout, "]")?;

    // Content with line numbers
    for (i, line) in result.chunk.text.lines().enumerate() {
        let line_num = result.chunk.start_line + i;
        stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green)).set_dimmed(true))?;
        write!(stdout, "{:>5} ", line_num)?;
        stdout.reset()?;
        stdout.set_color(ColorSpec::new().set_fg(Some(Color::White)))?;
        writeln!(stdout, "{}", line)?;
        stdout.reset()?;
    }

    Ok(())
}

/// Print only unique file paths from results.
pub fn print_files_with_matches(
    results: &[SearchResult],
    color: ColorChoice,
) -> std::io::Result<()> {
    let mut stdout = StandardStream::stdout(color);
    let paths = collect_files_with_matches(results);

    for path in paths {
        stdout.set_color(ColorSpec::new().set_fg(Some(Color::Magenta)).set_bold(true))?;
        writeln!(stdout, "{}", path)?;
        stdout.reset()?;
    }

    Ok(())
}

/// Print count of matching chunks per file.
pub fn print_count(results: &[SearchResult], color: ColorChoice) -> std::io::Result<()> {
    let mut stdout = StandardStream::stdout(color);
    let counts = collect_counts(results);

    for (path, count) in &counts {
        stdout.set_color(ColorSpec::new().set_fg(Some(Color::Magenta)).set_bold(true))?;
        write!(stdout, "{}", path)?;
        stdout.reset()?;
        stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
        writeln!(stdout, ":{}", count)?;
        stdout.reset()?;
    }

    Ok(())
}

/// Score thresholds for color coding.
pub const SCORE_HIGH_THRESHOLD: f32 = 0.7;
pub const SCORE_MEDIUM_THRESHOLD: f32 = 0.5;

pub(crate) fn score_to_color(score: f32) -> Color {
    if score >= SCORE_HIGH_THRESHOLD {
        Color::Green
    } else if score >= SCORE_MEDIUM_THRESHOLD {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Print results as JSONL for scripting.
pub fn print_json(results: &[SearchResult]) -> std::io::Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    for result in results {
        let json = format_json_result(result);
        writeln!(handle, "{}", json)?;
    }
    Ok(())
}

/// Format a byte size into a human-readable string.
pub(crate) fn format_size(bytes: u64) -> String {
    if bytes > 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes > 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a search result as a JSON value.
pub(crate) fn format_json_result(result: &SearchResult) -> serde_json::Value {
    serde_json::json!({
        "file": result.chunk.file_path,
        "start_line": result.chunk.start_line,
        "end_line": result.chunk.end_line,
        "score": result.score,
        "text": result.chunk.text,
    })
}

/// Collect unique file paths from results (preserving first-seen order).
pub(crate) fn collect_files_with_matches(results: &[SearchResult]) -> Vec<&str> {
    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();
    for result in results {
        if seen.insert(result.chunk.file_path.as_str()) {
            paths.push(result.chunk.file_path.as_str());
        }
    }
    paths
}

/// Collect match counts grouped by file path.
pub(crate) fn collect_counts(results: &[SearchResult]) -> BTreeMap<&str, usize> {
    let mut counts = BTreeMap::new();
    for result in results {
        *counts.entry(result.chunk.file_path.as_str()).or_insert(0) += 1;
    }
    counts
}

/// Print index statistics.
pub fn print_stats(file_count: usize, chunk_count: usize, db_size: u64) {
    let size_str = format_size(db_size);

    eprintln!("Index statistics:");
    eprintln!("  Files:  {}", file_count);
    eprintln!("  Chunks: {}", chunk_count);
    eprintln!("  Size:   {}", size_str);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, SearchResult};

    fn make_result(file: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                file_path: file.to_string(),
                text: format!("content of {}", file),
                start_line: 1,
                end_line: 5,
            },
            score,
        }
    }

    #[test]
    fn test_score_to_color_high() {
        assert_eq!(score_to_color(0.9), Color::Green);
        assert_eq!(score_to_color(0.7), Color::Green);
    }

    #[test]
    fn test_score_to_color_medium() {
        assert_eq!(score_to_color(0.6), Color::Yellow);
        assert_eq!(score_to_color(0.5), Color::Yellow);
    }

    #[test]
    fn test_score_to_color_low() {
        assert_eq!(score_to_color(0.4), Color::Red);
        assert_eq!(score_to_color(0.0), Color::Red);
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(999), "999 B");
        assert_eq!(format_size(1000), "1000 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1_001), "1.0 KB");
        assert_eq!(format_size(50_000), "50.0 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1_000_001), "1.0 MB");
        assert_eq!(format_size(5_500_000), "5.5 MB");
    }

    #[test]
    fn test_format_json_result() {
        let result = make_result("test.rs", 0.85);
        let json = format_json_result(&result);
        assert_eq!(json["file"], "test.rs");
        assert_eq!(json["start_line"], 1);
        assert_eq!(json["end_line"], 5);
        assert_eq!(json["score"], 0.85f32 as f64);
        assert_eq!(json["text"], "content of test.rs");
    }

    #[test]
    fn test_collect_files_with_matches() {
        let results = vec![
            make_result("a.rs", 0.9),
            make_result("b.rs", 0.8),
            make_result("a.rs", 0.7), // duplicate
            make_result("c.rs", 0.6),
        ];
        let files = collect_files_with_matches(&results);
        assert_eq!(files, vec!["a.rs", "b.rs", "c.rs"]);
    }

    #[test]
    fn test_collect_counts() {
        let results = vec![
            make_result("a.rs", 0.9),
            make_result("b.rs", 0.8),
            make_result("a.rs", 0.7),
            make_result("a.rs", 0.6),
            make_result("b.rs", 0.5),
        ];
        let counts = collect_counts(&results);
        assert_eq!(counts[&"a.rs"], 3);
        assert_eq!(counts[&"b.rs"], 2);
    }
}
