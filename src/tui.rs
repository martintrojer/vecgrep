pub mod interactive {
    use crate::embedder::Embedder;
    use crate::index::Index;
    use crate::output::{SCORE_HIGH_THRESHOLD, SCORE_MEDIUM_THRESHOLD};
    use crate::paths;
    use crate::pipeline::StreamingIndexer;
    use crate::search;
    use crate::types::{Chunk, SearchResult};
    use anyhow::Result;
    use crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ndarray::Array2;
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

    pub fn run(
        embedder: &mut Embedder,
        chunks: &[Chunk],
        embedding_matrix: &Array2<f32>,
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

        let result = event_loop(
            &mut terminal,
            embedder,
            chunks.to_vec(),
            embedding_matrix.clone(),
            initial_query,
            top_k,
            threshold,
            cwd_suffix,
            None,
            None,
        );

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    pub fn run_streaming(
        embedder: &mut Embedder,
        idx: &Index,
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

        let (chunks, embedding_matrix) = idx.load_all()?;

        let result = event_loop(
            &mut terminal,
            embedder,
            chunks,
            embedding_matrix,
            initial_query,
            top_k,
            threshold,
            cwd_suffix,
            Some(indexer),
            Some(idx),
        );

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn event_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        embedder: &mut Embedder,
        mut chunks: Vec<Chunk>,
        mut embedding_matrix: Array2<f32>,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
        cwd_suffix: &Path,
        mut indexer: Option<StreamingIndexer>,
        idx: Option<&Index>,
    ) -> Result<()> {
        let mut query = initial_query.to_string();
        let mut results: Vec<SearchResult> = Vec::new();
        let mut list_state = ListState::default();
        let mut show_preview = true;
        let mut last_search = Instant::now() - Duration::from_secs(1);
        let mut needs_search = true;
        let debounce = Duration::from_millis(300);

        // Preview state
        let mut preview_file_cache: Option<(String, String)> = None;
        let mut preview_scroll: u16 = 0;
        let mut last_selected: Option<usize> = None;

        // Initial search
        if !query.is_empty() && !chunks.is_empty() {
            if let Ok(emb) = embedder.embed(&query) {
                results = search::search(&emb, &embedding_matrix, top_k, threshold, &chunks);
                if !results.is_empty() {
                    list_state.select(Some(0));
                }
            }
            needs_search = false;
        }

        loop {
            // 1. Process streaming indexing (if active)
            if let (Some(ref mut si), Some(idx)) = (&mut indexer, idx) {
                if si.poll(embedder, idx, &mut chunks, &mut embedding_matrix)? {
                    needs_search = true;
                }
            }

            // 2. Detect selection change and update preview cache/scroll
            let current_selected = list_state.selected();
            if current_selected != last_selected {
                last_selected = current_selected;
                if let Some(sel) = current_selected {
                    if let Some(result) = results.get(sel) {
                        let path = &result.chunk.file_path;
                        let fs_path = paths::to_cwd_relative(path, cwd_suffix);
                        let needs_load = match &preview_file_cache {
                            Some((cached_path, _)) => cached_path != path,
                            None => true,
                        };
                        if needs_load {
                            if let Ok(content) = std::fs::read_to_string(&fs_path) {
                                preview_file_cache = Some((path.clone(), content));
                            } else {
                                preview_file_cache = None;
                            }
                        }
                        preview_scroll = (result.chunk.start_line.saturating_sub(4)) as u16;
                    }
                }
            }

            // 3. Render
            let preview_scroll_val = preview_scroll;
            let preview_cache_ref = &preview_file_cache;
            let status_text = match &indexer {
                Some(si) if !si.indexing_done => {
                    format!(
                        "{} results | Indexing: {} files...",
                        results.len(),
                        si.indexed_count
                    )
                }
                Some(_) => {
                    format!(
                        "{} results | {} chunks indexed",
                        results.len(),
                        chunks.len()
                    )
                }
                None => format!("{} results", results.len()),
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

            // 4. Handle input
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Enter => {
                            if let Some(sel) = list_state.selected() {
                                if let Some(result) = results.get(sel) {
                                    let line = result.chunk.start_line;
                                    let file =
                                        paths::to_cwd_relative(&result.chunk.file_path, cwd_suffix);

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

            // 5. Debounced search
            if needs_search && last_search.elapsed() >= debounce && !query.is_empty() {
                if let Ok(emb) = embedder.embed(&query) {
                    results = search::search(&emb, &embedding_matrix, top_k, threshold, &chunks);
                    if !results.is_empty() {
                        list_state.select(Some(0));
                    } else {
                        list_state.select(None);
                    }
                }
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
