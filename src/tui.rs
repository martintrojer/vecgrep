use crate::embedder::Embedder;
use crate::index::Index;
use crate::output;
use crate::paths;
use crate::pipeline::{EmbedWorker, PipelineStatus, SearchOutcome, StreamingIndexer};
use crate::types::SearchResult;
use crate::types::SearchScope;
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Terminal,
};
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Debug)]
enum SearchTrigger {
    None,
    AutoRefresh,
    UserInput,
}

/// Result of handling a single key press.
enum KeyOutcome {
    Continue,
    Quit,
    /// User pressed Enter on a selected result — caller should leave the TUI and
    /// run the open command for the given (file, start_line, end_line).
    Open(String, usize, usize),
}

/// Immutable runtime configuration for the TUI.
struct TuiConfig<'a> {
    top_k: usize,
    threshold: f32,
    hybrid: bool,
    cwd_suffix: &'a Path,
    path_scopes: &'a [String],
    open_cmd: Option<&'a str>,
}

/// All mutable state that lives across one TUI iteration.
struct TuiState {
    query: String,
    results: Vec<SearchResult>,
    list_state: ListState,
    show_preview: bool,
    last_search: Instant,
    pending_search: SearchTrigger,
    searching: bool,
    active_search_trigger: SearchTrigger,
    active_request_id: Option<u64>,
    search_error: Option<String>,
    debounce: Duration,
    auto_refresh_interval: Duration,
    last_auto_refresh: Instant,
    pipeline_status: PipelineStatus,
    preview_file_cache: Option<(String, String)>,
    preview_scroll: u16,
    last_selected: Option<usize>,
}

impl TuiState {
    fn new(initial_query: &str) -> Self {
        Self {
            query: initial_query.to_string(),
            results: Vec::new(),
            list_state: ListState::default(),
            show_preview: true,
            last_search: Instant::now() - Duration::from_secs(1),
            pending_search: SearchTrigger::UserInput,
            searching: false,
            active_search_trigger: SearchTrigger::None,
            active_request_id: None,
            search_error: None,
            debounce: Duration::from_millis(300),
            auto_refresh_interval: Duration::from_secs(3),
            last_auto_refresh: Instant::now(),
            pipeline_status: PipelineStatus::initial(),
            preview_file_cache: None,
            preview_scroll: 0,
            last_selected: None,
        }
    }

    /// Apply a search outcome from the worker. Ignores stale request ids.
    fn handle_search_outcome(&mut self, outcome: SearchOutcome, cwd_suffix: &Path) {
        if self.active_request_id != Some(outcome.request_id()) {
            return;
        }
        self.searching = false;
        self.active_search_trigger = SearchTrigger::None;
        match outcome {
            SearchOutcome::Results {
                results: new_results,
                ..
            } => {
                self.search_error = None;
                self.results = new_results;
                if !self.results.is_empty() {
                    self.list_state.select(Some(0));
                } else {
                    self.list_state.select(None);
                }
                paths::rewrite_results_to_cwd_relative(&mut self.results, cwd_suffix);
            }
            SearchOutcome::SearchError { message, .. } => {
                self.search_error = Some(format!("Search error: {message}"));
                self.results.clear();
                self.list_state.select(None);
            }
            SearchOutcome::EmbedError { message, .. } => {
                self.search_error = Some(format!("Embed error: {message}"));
                self.results.clear();
                self.list_state.select(None);
            }
        }
        self.last_selected = None;
    }

    /// Apply a pipeline progress update and decide whether to schedule an auto-refresh.
    fn handle_progress(&mut self, status: PipelineStatus) {
        let was_indexing = !matches!(self.pipeline_status, PipelineStatus::Ready { .. });
        self.pipeline_status = status;
        let is_ready = matches!(self.pipeline_status, PipelineStatus::Ready { .. });

        if self.query.is_empty() || self.searching || self.pending_search != SearchTrigger::None {
            return;
        }
        if was_indexing && is_ready {
            self.pending_search = SearchTrigger::AutoRefresh;
        } else if was_indexing && self.last_auto_refresh.elapsed() >= self.auto_refresh_interval {
            self.pending_search = SearchTrigger::AutoRefresh;
            self.last_auto_refresh = Instant::now();
        }
    }

    /// Refresh the cached preview file when the selection changes.
    fn update_preview_cache(&mut self) {
        let current_selected = self.list_state.selected();
        if current_selected == self.last_selected {
            return;
        }
        self.last_selected = current_selected;
        let Some(sel) = current_selected else {
            return;
        };
        let Some(result) = self.results.get(sel) else {
            return;
        };
        let path = &result.chunk.file_path;
        let needs_load = match &self.preview_file_cache {
            Some((cached_path, _)) => cached_path != path,
            None => true,
        };
        if needs_load {
            self.preview_file_cache = std::fs::read_to_string(path)
                .ok()
                .map(|content| (path.clone(), content));
        }
        self.preview_scroll = (result.chunk.start_line.saturating_sub(4)) as u16;
    }

    fn status_text(&self, hybrid: bool, path_scopes: &[String]) -> String {
        let index_status = self.pipeline_status.to_string();
        let scope_str = if path_scopes.is_empty() {
            String::new()
        } else {
            format!(" | scope: {}", path_scopes.join(", "))
        };
        let mode_str = if hybrid { " | mode: hybrid" } else { "" };

        if let Some(ref err) = self.search_error {
            err.clone()
        } else if self.active_search_trigger == SearchTrigger::UserInput {
            format!("Searching... | {index_status}{scope_str}{mode_str}")
        } else if !matches!(self.pipeline_status, PipelineStatus::Ready { .. }) {
            format!(
                "{} results | Indexing: {index_status}{scope_str}{mode_str}",
                self.results.len()
            )
        } else {
            format!(
                "{} results | {index_status}{scope_str}{mode_str}",
                self.results.len()
            )
        }
    }

    fn handle_key(&mut self, key: event::KeyEvent) -> KeyOutcome {
        match key.code {
            KeyCode::Esc => KeyOutcome::Quit,
            KeyCode::Enter => {
                if let Some(sel) = self.list_state.selected() {
                    if let Some(result) = self.results.get(sel) {
                        return KeyOutcome::Open(
                            result.chunk.file_path.clone(),
                            result.chunk.start_line,
                            result.chunk.end_line,
                        );
                    }
                }
                KeyOutcome::Continue
            }
            KeyCode::Tab => {
                self.show_preview = !self.show_preview;
                KeyOutcome::Continue
            }
            KeyCode::Up => {
                if let Some(sel) = self.list_state.selected() {
                    if sel > 0 {
                        self.list_state.select(Some(sel - 1));
                    }
                }
                KeyOutcome::Continue
            }
            KeyCode::Down => {
                if let Some(sel) = self.list_state.selected() {
                    if sel + 1 < self.results.len() {
                        self.list_state.select(Some(sel + 1));
                    }
                }
                KeyOutcome::Continue
            }
            KeyCode::PageUp => {
                self.preview_scroll = self.preview_scroll.saturating_sub(10);
                KeyOutcome::Continue
            }
            KeyCode::PageDown => {
                self.preview_scroll = self.preview_scroll.saturating_add(10);
                KeyOutcome::Continue
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.pending_search = SearchTrigger::UserInput;
                self.last_search = Instant::now();
                KeyOutcome::Continue
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.pending_search = SearchTrigger::UserInput;
                self.last_search = Instant::now();
                KeyOutcome::Continue
            }
            _ => KeyOutcome::Continue,
        }
    }

    /// Issue a search to the worker if the debounce window has elapsed.
    fn maybe_dispatch_search(&mut self, worker: &EmbedWorker, cfg: &TuiConfig<'_>) {
        if self.pending_search == SearchTrigger::None
            || self.searching
            || self.last_search.elapsed() < self.debounce
            || self.query.is_empty()
        {
            return;
        }
        self.active_search_trigger = self.pending_search;
        self.active_request_id =
            Some(worker.search(&self.query, cfg.top_k, cfg.threshold, cfg.hybrid));
        self.searching = true;
        self.pending_search = SearchTrigger::None;
        self.last_selected = None;
    }

    fn render(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        cfg: &TuiConfig<'_>,
    ) -> Result<()> {
        let status_text = self.status_text(cfg.hybrid, cfg.path_scopes);
        terminal.draw(|f| {
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(5),
                    Constraint::Length(1),
                ])
                .split(f.area());

            let query_block = Paragraph::new(self.query.as_str())
                .block(Block::default().borders(Borders::ALL).title(if cfg.hybrid {
                    " Query (hybrid search) "
                } else {
                    " Query (semantic search) "
                }))
                .style(Style::default().fg(Color::Yellow));
            f.render_widget(query_block, main_chunks[0]);

            if self.show_preview && !self.results.is_empty() {
                let result_area = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                    .split(main_chunks[1]);

                render_list(f, &self.results, &mut self.list_state, result_area[0]);
                render_preview(
                    f,
                    &self.results,
                    &self.list_state,
                    result_area[1],
                    &self.preview_file_cache,
                    self.preview_scroll,
                );
            } else {
                render_list(f, &self.results, &mut self.list_state, main_chunks[1]);
            }

            let key_style = Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD);
            let keymap: &[(&str, &str)] = &[
                (" Esc", ":quit "),
                ("Enter", ":view "),
                ("Tab", ":preview "),
                ("PgUp/PgDn", ":scroll "),
            ];
            let mut spans: Vec<Span> = keymap
                .iter()
                .flat_map(|(k, d)| [Span::styled(*k, key_style), Span::raw(*d)])
                .collect();
            spans.push(Span::raw(format!(" | {}", status_text)));
            let status = Line::from(spans);
            let status_bar = Paragraph::new(status).style(Style::default().bg(Color::DarkGray));
            f.render_widget(status_bar, main_chunks[2]);
        })?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_streaming(
    embedder: Embedder,
    idx: Index,
    indexer: StreamingIndexer,
    initial_query: &str,
    args: &crate::cli::Args,
    cwd_suffix: &Path,
    scope: SearchScope,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let path_scopes = scope.path_scopes.clone();
    let worker = EmbedWorker::spawn(embedder, idx, indexer, scope);

    let cfg = TuiConfig {
        top_k: args.top_k.unwrap(),
        threshold: args.threshold.unwrap(),
        hybrid: args.hybrid,
        cwd_suffix,
        path_scopes: &path_scopes,
        open_cmd: args.open_cmd.as_deref(),
    };
    let result = event_loop(&mut terminal, &worker, initial_query, &cfg);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    worker: &EmbedWorker,
    initial_query: &str,
    cfg: &TuiConfig<'_>,
) -> Result<()> {
    let mut state = TuiState::new(initial_query);

    // Initial search
    if !state.query.is_empty() {
        state.active_request_id =
            Some(worker.search(&state.query, cfg.top_k, cfg.threshold, cfg.hybrid));
        state.searching = true;
        state.active_search_trigger = SearchTrigger::UserInput;
        state.pending_search = SearchTrigger::None;
    }

    loop {
        if let Some(outcome) = worker.try_recv_results() {
            state.handle_search_outcome(outcome, cfg.cwd_suffix);
        }
        if let Some(status) = worker.drain_progress() {
            state.handle_progress(status);
        }
        state.update_preview_cache();
        state.render(terminal, cfg)?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match state.handle_key(key) {
                    KeyOutcome::Continue => {}
                    KeyOutcome::Quit => return Ok(()),
                    KeyOutcome::Open(file, line, end_line) => {
                        return open_in_editor(cfg.open_cmd, &file, line, end_line);
                    }
                }
            }
        }

        state.maybe_dispatch_search(worker, cfg);
    }
}

/// Drop the alternate screen, run the configured open command, then return.
fn open_in_editor(open_cmd: Option<&str>, file: &str, line: usize, end_line: usize) -> Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;

    let default_cmd = format!(
        "{} +{{line}}G {{file}}",
        std::env::var("PAGER").unwrap_or_else(|_| "less".to_string())
    );
    let cmd = open_cmd.unwrap_or(default_cmd.as_str());
    for warning in validate_open_cmd(cmd) {
        eprintln!("{warning}");
    }
    let expanded = expand_open_cmd(cmd, file, line, end_line);
    if let Err(e) = std::process::Command::new("sh")
        .arg("-c")
        .arg(&expanded)
        .status()
    {
        eprintln!("Failed to run '{}': {}", expanded, e);
    }
    Ok(())
}

fn render_list(
    f: &mut ratatui::Frame,
    results: &[SearchResult],
    list_state: &mut ListState,
    area: ratatui::layout::Rect,
) {
    let items: Vec<ListItem> = results
        .iter()
        .map(|r| {
            let score_color = output::score_to_color_ratatui(r.score);
            let line = Line::from(vec![
                Span::styled(
                    format!("[{:.3}] ", r.score),
                    Style::default()
                        .fg(score_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(&r.chunk.file_path, Style::default().fg(Color::Magenta)),
                Span::styled(
                    format!(":{}:{}", r.chunk.start_line, r.chunk.end_line),
                    Style::default().fg(Color::Green),
                ),
            ]);
            ListItem::new(line)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Results "))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    f.render_stateful_widget(list, area, list_state);
}

fn render_preview(
    f: &mut ratatui::Frame,
    results: &[SearchResult],
    list_state: &ListState,
    area: ratatui::layout::Rect,
    file_cache: &Option<(String, String)>,
    scroll: u16,
) {
    let content = if let Some(sel) = list_state.selected() {
        if let Some(result) = results.get(sel) {
            if let Some((_, ref file_content)) = file_cache {
                let chunk_start = result.chunk.start_line;
                let chunk_end = result.chunk.end_line;
                let highlight_style = Style::default().bg(Color::DarkGray);

                file_content
                    .lines()
                    .enumerate()
                    .map(|(i, line)| {
                        let line_num = i + 1;
                        let in_chunk = line_num >= chunk_start && line_num <= chunk_end;
                        let num_style = if in_chunk {
                            Style::default().fg(Color::Yellow).bg(Color::DarkGray)
                        } else {
                            Style::default().fg(Color::DarkGray)
                        };
                        let text_style = if in_chunk {
                            highlight_style
                        } else {
                            Style::default()
                        };
                        Line::from(vec![
                            Span::styled(format!("{:>5} ", line_num), num_style),
                            Span::styled(line, text_style),
                        ])
                    })
                    .collect()
            } else {
                vec![Line::raw("Unable to read file")]
            }
        } else {
            vec![Line::raw("No selection")]
        }
    } else {
        vec![Line::raw("No selection")]
    };

    let preview = Paragraph::new(content)
        .block(Block::default().borders(Borders::ALL).title(" Preview "))
        .scroll((scroll, 0));

    f.render_widget(preview, area);
}

/// Expand `{file}`, `{line}`, and `{end_line}` placeholders in an open-cmd template.
fn expand_open_cmd(cmd: &str, file: &str, line: usize, end_line: usize) -> String {
    cmd.replace("{file}", file)
        .replace("{line}", &line.to_string())
        .replace("{end_line}", &end_line.to_string())
}

/// Return warnings for an open-cmd template (missing `{file}`, unknown placeholders).
fn validate_open_cmd(cmd: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    if !cmd.contains("{file}") {
        warnings.push("Warning: --open-cmd missing {file} placeholder".to_string());
    }
    let valid = ["{file}", "{line}", "{end_line}"];
    for cap in cmd.split('{').skip(1) {
        if let Some(name) = cap.split('}').next() {
            let placeholder = format!("{{{name}}}");
            if !valid.contains(&placeholder.as_str()) {
                warnings.push(format!(
                    "Warning: unknown placeholder '{placeholder}' in --open-cmd"
                ));
            }
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_open_cmd_all_placeholders() {
        let result = expand_open_cmd("nvim +{line} {file}", "src/main.rs", 42, 50);
        assert_eq!(result, "nvim +42 src/main.rs");
    }

    #[test]
    fn test_expand_open_cmd_with_end_line() {
        let result = expand_open_cmd(
            "bat -n --highlight-line {line}:{end_line} {file}",
            "a.rs",
            10,
            20,
        );
        assert_eq!(result, "bat -n --highlight-line 10:20 a.rs");
    }

    #[test]
    fn test_expand_open_cmd_no_placeholders() {
        let result = expand_open_cmd("less", "file.rs", 1, 5);
        assert_eq!(result, "less");
    }

    #[test]
    fn test_validate_open_cmd_missing_file() {
        let warnings = validate_open_cmd("less +{line}G");
        assert!(warnings.iter().any(|w| w.contains("missing {file}")));
    }

    #[test]
    fn test_validate_open_cmd_unknown_placeholder() {
        let warnings = validate_open_cmd("{file} {col}");
        assert!(warnings
            .iter()
            .any(|w| w.contains("unknown placeholder '{col}'")));
    }

    #[test]
    fn test_validate_open_cmd_valid() {
        let warnings = validate_open_cmd("nvim +{line} {file}");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_open_cmd_all_valid_placeholders() {
        let warnings = validate_open_cmd("{file} {line} {end_line}");
        assert!(warnings.is_empty());
    }

    // --- TuiState behavior tests (tr_tui_event_loop_gap) ---
    //
    // These cover the testable parts of the TUI without the full
    // event_loop / pty refactor: state initialization, search outcome
    // dispatch (including out-of-order request handling), error rendering
    // in the status bar, and scope display. The bigger pieces called out
    // by the finding (pty integration test, ratatui TestBackend snapshot)
    // remain DEFERRED — see the task note.

    use crate::types::{Chunk, SearchResult};
    use std::path::Path;

    fn mk_result(path: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                file_path: path.to_string(),
                text: "text".to_string(),
                start_line: 1,
                end_line: 1,
            },
            score,
        }
    }

    #[test]
    fn test_tui_state_initial_query_is_prepopulated() {
        // Regression target: --query value must be in the input on first
        // frame (and must NOT be re-typed by an event handler).
        let st = TuiState::new("hello world");
        assert_eq!(st.query, "hello world");
        // Initial state must request a search so the pre-populated query
        // actually fires.
        assert_eq!(st.pending_search, SearchTrigger::UserInput);
        assert!(!st.searching);
        assert!(st.results.is_empty());
    }

    #[test]
    fn test_tui_state_handle_search_outcome_ignores_stale_request() {
        // Out-of-order arrival: state has active_request_id=2, an outcome
        // for request_id=1 must be dropped on the floor (no result
        // overwrite, no state change to searching).
        let mut st = TuiState::new("q");
        st.searching = true;
        st.active_request_id = Some(2);
        st.results = vec![mk_result("existing.rs", 0.5)];
        let stale = SearchOutcome::Results {
            request_id: 1,
            results: vec![mk_result("stale.rs", 0.9)],
        };
        st.handle_search_outcome(stale, Path::new(""));
        assert!(
            st.searching,
            "stale outcome must NOT clear `searching` flag"
        );
        assert_eq!(
            st.results.len(),
            1,
            "stale outcome must NOT overwrite results"
        );
        assert_eq!(
            st.results[0].chunk.file_path, "existing.rs",
            "existing result must remain unchanged"
        );
        assert_eq!(
            st.active_request_id,
            Some(2),
            "active_request_id must be unchanged"
        );
    }

    #[test]
    fn test_tui_state_handle_search_outcome_applies_matching_request() {
        let mut st = TuiState::new("q");
        st.searching = true;
        st.active_request_id = Some(7);
        st.active_search_trigger = SearchTrigger::UserInput;
        let outcome = SearchOutcome::Results {
            request_id: 7,
            results: vec![mk_result("a.rs", 0.8), mk_result("b.rs", 0.5)],
        };
        st.handle_search_outcome(outcome, Path::new(""));
        assert!(!st.searching, "matching outcome must clear `searching`");
        assert_eq!(st.results.len(), 2);
        assert_eq!(st.list_state.selected(), Some(0), "first row selected");
        assert!(st.search_error.is_none());
        assert_eq!(
            st.active_search_trigger,
            SearchTrigger::None,
            "trigger reset after results arrive"
        );
    }

    #[test]
    fn test_tui_state_handle_search_outcome_renders_error() {
        let mut st = TuiState::new("q");
        st.searching = true;
        st.active_request_id = Some(3);
        st.results = vec![mk_result("old.rs", 0.5)]; // stale results
        let outcome = SearchOutcome::EmbedError {
            request_id: 3,
            message: "connection refused".to_string(),
        };
        st.handle_search_outcome(outcome, Path::new(""));
        assert!(!st.searching);
        assert_eq!(
            st.results.len(),
            0,
            "results must be cleared on embed error"
        );
        let err = st.search_error.expect("error must be set");
        assert!(
            err.contains("Embed error") && err.contains("connection refused"),
            "error message missing expected text: {err}"
        );
    }

    #[test]
    fn test_tui_state_status_text_includes_scope_and_mode() {
        let mut st = TuiState::new("q");
        st.results = vec![mk_result("a.rs", 0.5)];
        st.pipeline_status = PipelineStatus::Ready {
            files: 10,
            chunks: 30,
        };
        let text = st.status_text(true, &["src".to_string(), "docs".to_string()]);
        assert!(text.contains("scope: src, docs"), "missing scope: {text}");
        assert!(text.contains("mode: hybrid"), "missing mode: {text}");
        assert!(text.contains("1 results"));
    }

    #[test]
    fn test_tui_state_status_text_renders_search_error_directly() {
        let mut st = TuiState::new("q");
        st.search_error = Some("Search error: boom".to_string());
        let text = st.status_text(false, &[]);
        assert_eq!(
            text, "Search error: boom",
            "search error must take precedence over status"
        );
    }
}
