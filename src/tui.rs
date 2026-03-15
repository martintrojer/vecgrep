pub mod interactive {
    use crate::embedder::Embedder;
    use crate::index::Index;
    use crate::output::{SCORE_HIGH_THRESHOLD, SCORE_MEDIUM_THRESHOLD};
    use crate::paths;
    use crate::pipeline::{EmbedWorker, SearchOutcome, StreamingIndexer};
    use crate::types::SearchResult;
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

    pub fn run_streaming(
        embedder: Embedder,
        idx: Index,
        indexer: StreamingIndexer,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
        cwd_suffix: &Path,
    ) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let worker = EmbedWorker::spawn(embedder, idx, indexer);

        let result = event_loop(
            &mut terminal,
            &worker,
            initial_query,
            top_k,
            threshold,
            cwd_suffix,
        );

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    fn event_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        worker: &EmbedWorker,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
        cwd_suffix: &Path,
    ) -> Result<()> {
        let mut query = initial_query.to_string();
        let mut results: Vec<SearchResult> = Vec::new();
        let mut list_state = ListState::default();
        let mut show_preview = true;
        let mut last_search = Instant::now() - Duration::from_secs(1);
        let mut needs_search = true;
        let mut searching = false;
        let mut active_request_id: Option<u64> = None;
        let mut search_error: Option<String> = None;
        let debounce = Duration::from_millis(300);

        // Index progress state
        let mut indexed_count: usize = 0;
        let mut chunk_count: usize = 0;
        let mut indexing_done = false;

        // Preview state
        let mut preview_file_cache: Option<(String, String)> = None;
        let mut preview_scroll: u16 = 0;
        let mut last_selected: Option<usize> = None;

        // Initial search
        if !query.is_empty() {
            active_request_id = Some(worker.search(&query, top_k, threshold));
            searching = true;
            needs_search = false;
        }

        loop {
            // 1. Check for search results (non-blocking)
            if let Some(outcome) = worker.try_recv_results() {
                if active_request_id != Some(outcome.request_id()) {
                    continue;
                }
                searching = false;
                match outcome {
                    SearchOutcome::Results {
                        results: new_results,
                        ..
                    } => {
                        search_error = None;
                        results = new_results;
                        if !results.is_empty() {
                            list_state.select(Some(0));
                        } else {
                            list_state.select(None);
                        }
                        if !cwd_suffix.as_os_str().is_empty() {
                            for r in &mut results {
                                r.chunk.file_path =
                                    paths::to_cwd_relative(&r.chunk.file_path, cwd_suffix);
                            }
                        }
                    }
                    SearchOutcome::EmbedError { message, .. } => {
                        search_error = Some(message);
                        results.clear();
                        list_state.select(None);
                    }
                }
                last_selected = None;
            }

            // 2. Check for index progress (non-blocking)
            if let Some(progress) = worker.drain_progress() {
                indexed_count = progress.indexed_count;
                chunk_count = progress.chunk_count;
                indexing_done = progress.indexing_done;
                // Re-search when new data is indexed
                if !query.is_empty() && !searching {
                    needs_search = true;
                }
            }

            // 3. Detect selection change and update preview cache/scroll
            let current_selected = list_state.selected();
            if current_selected != last_selected {
                last_selected = current_selected;
                if let Some(sel) = current_selected {
                    if let Some(result) = results.get(sel) {
                        let path = &result.chunk.file_path;
                        let needs_load = match &preview_file_cache {
                            Some((cached_path, _)) => cached_path != path,
                            None => true,
                        };
                        if needs_load {
                            if let Ok(content) = std::fs::read_to_string(path) {
                                preview_file_cache = Some((path.clone(), content));
                            } else {
                                preview_file_cache = None;
                            }
                        }
                        preview_scroll = (result.chunk.start_line.saturating_sub(4)) as u16;
                    }
                }
            }

            // 4. Render
            let preview_scroll_val = preview_scroll;
            let preview_cache_ref = &preview_file_cache;
            let status_text = if let Some(ref err) = search_error {
                format!("Embed error: {err}")
            } else if searching {
                format!("Searching... | {} chunks indexed", chunk_count)
            } else if !indexing_done {
                format!(
                    "{} results | Indexing: {} files...",
                    results.len(),
                    indexed_count
                )
            } else {
                format!("{} results | {} chunks indexed", results.len(), chunk_count)
            };

            terminal.draw(|f| {
                let main_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Min(5),
                        Constraint::Length(1),
                    ])
                    .split(f.area());

                let query_block = Paragraph::new(query.as_str())
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" Query (semantic search) "),
                    )
                    .style(Style::default().fg(Color::Yellow));
                f.render_widget(query_block, main_chunks[0]);

                if show_preview && !results.is_empty() {
                    let result_area = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                        .split(main_chunks[1]);

                    render_list(f, &results, &mut list_state, result_area[0]);
                    render_preview(
                        f,
                        &results,
                        &list_state,
                        result_area[1],
                        preview_cache_ref,
                        preview_scroll_val,
                    );
                } else {
                    render_list(f, &results, &mut list_state, main_chunks[1]);
                }

                let status = Line::from(vec![
                    Span::styled(
                        " Esc",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":quit "),
                    Span::styled(
                        "Enter",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":view "),
                    Span::styled(
                        "Tab",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":preview "),
                    Span::styled(
                        "PgUp/PgDn",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":scroll "),
                    Span::raw(format!(" | {}", status_text)),
                ]);
                let status_bar = Paragraph::new(status).style(Style::default().bg(Color::DarkGray));
                f.render_widget(status_bar, main_chunks[2]);
            })?;

            // 5. Handle input
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Enter => {
                            if let Some(sel) = list_state.selected() {
                                if let Some(result) = results.get(sel) {
                                    let line = result.chunk.start_line;
                                    let file = result.chunk.file_path.clone();

                                    disable_raw_mode()?;
                                    execute!(io::stdout(), LeaveAlternateScreen)?;

                                    let pager = std::env::var("PAGER")
                                        .unwrap_or_else(|_| "less".to_string());
                                    let _ = std::process::Command::new(&pager)
                                        .arg(format!("+{}G", line))
                                        .arg(&file)
                                        .status();

                                    return Ok(());
                                }
                            }
                        }
                        KeyCode::Tab => {
                            show_preview = !show_preview;
                        }
                        KeyCode::Up => {
                            if let Some(sel) = list_state.selected() {
                                if sel > 0 {
                                    list_state.select(Some(sel - 1));
                                }
                            }
                        }
                        KeyCode::Down => {
                            if let Some(sel) = list_state.selected() {
                                if sel + 1 < results.len() {
                                    list_state.select(Some(sel + 1));
                                }
                            }
                        }
                        KeyCode::PageUp => {
                            preview_scroll = preview_scroll.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            preview_scroll = preview_scroll.saturating_add(10);
                        }
                        KeyCode::Backspace => {
                            query.pop();
                            needs_search = true;
                            last_search = Instant::now();
                        }
                        KeyCode::Char(c) => {
                            query.push(c);
                            needs_search = true;
                            last_search = Instant::now();
                        }
                        _ => {}
                    }
                }
            }

            // 6. Debounced search
            if needs_search && !searching && last_search.elapsed() >= debounce && !query.is_empty()
            {
                active_request_id = Some(worker.search(&query, top_k, threshold));
                searching = true;
                needs_search = false;
                last_selected = None;
            }
        }
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
                let score_color = if r.score >= SCORE_HIGH_THRESHOLD {
                    Color::Green
                } else if r.score >= SCORE_MEDIUM_THRESHOLD {
                    Color::Yellow
                } else {
                    Color::Red
                };
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
}
