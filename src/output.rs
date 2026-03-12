use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

use crate::types::SearchResult;

/// Print search results in colored ripgrep-style format.
pub fn print_results(results: &[SearchResult], context_lines: usize) -> std::io::Result<()> {
    let color_choice = if atty::is(atty::Stream::Stdout) {
        ColorChoice::Auto
    } else {
        ColorChoice::Never
    };
    let mut stdout = StandardStream::stdout(color_choice);

    for (i, result) in results.iter().enumerate() {
        if i > 0 {
            writeln!(stdout)?;
        }
        print_result(&mut stdout, result, context_lines)?;
    }

    Ok(())
}

fn print_result(
    stdout: &mut StandardStream,
    result: &SearchResult,
    _context_lines: usize,
) -> std::io::Result<()> {
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

fn score_to_color(score: f32) -> Color {
    if score >= 0.7 {
        Color::Green
    } else if score >= 0.5 {
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
        let json = serde_json::json!({
            "file": result.chunk.file_path,
            "start_line": result.chunk.start_line,
            "end_line": result.chunk.end_line,
            "score": result.score,
            "text": result.chunk.text,
        });
        writeln!(handle, "{}", json)?;
    }
    Ok(())
}

/// Print index statistics.
pub fn print_stats(file_count: usize, chunk_count: usize, db_size: u64) {
    let size_str = if db_size > 1_000_000 {
        format!("{:.1} MB", db_size as f64 / 1_000_000.0)
    } else if db_size > 1_000 {
        format!("{:.1} KB", db_size as f64 / 1_000.0)
    } else {
        format!("{} B", db_size)
    };

    eprintln!("Index statistics:");
    eprintln!("  Files:  {}", file_count);
    eprintln!("  Chunks: {}", chunk_count);
    eprintln!("  Size:   {}", size_str);
}
