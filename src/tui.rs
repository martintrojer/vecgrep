pub mod interactive {
    use crate::embedder::Embedder;
    use crate::search;
    use crate::types::{Chunk, SearchResult};
    use anyhow::Result;
    use crossterm::{
        event::{self, Event, KeyCode, KeyModifiers},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ndarray::Array2;
    use ratatui::{
        backend::CrosstermBackend,
        layout::{Constraint, Direction, Layout},
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
        Terminal,
    };
    use std::io;
    use std::time::{Duration, Instant};

    pub fn run(
        embedder: &mut Embedder,
        chunks: &[Chunk],
        embedding_matrix: &Array2<f32>,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
    ) -> Result<Option<SearchResult>> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = run_loop(
            &mut terminal,
            embedder,
            chunks,
            embedding_matrix,
            initial_query,
            top_k,
            threshold,
        );

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    fn run_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        embedder: &mut Embedder,
        chunks: &[Chunk],
        embedding_matrix: &Array2<f32>,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
    ) -> Result<Option<SearchResult>> {
        let mut query = initial_query.to_string();
        let mut results: Vec<SearchResult> = Vec::new();
        let mut list_state = ListState::default();
        let mut show_preview = true;
        let mut last_search = Instant::now() - Duration::from_secs(1);
        let mut needs_search = true;
        let debounce = Duration::from_millis(300);

        // Initial search
        if !query.is_empty() {
            if let Ok(emb) = embedder.embed(&query) {
                results = search::search(&emb, embedding_matrix, top_k, threshold, chunks);
                if !results.is_empty() {
                    list_state.select(Some(0));
                }
            }
            needs_search = false;
        }

        loop {
            terminal.draw(|f| {
                let main_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3), // Query input
                        Constraint::Min(5),    // Results area
                        Constraint::Length(1), // Status bar
                    ])
                    .split(f.area());

                // Query input
                let query_block = Paragraph::new(query.as_str())
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" Query (semantic search) "),
                    )
                    .style(Style::default().fg(Color::Yellow));
                f.render_widget(query_block, main_chunks[0]);

                // Results area - split into list and preview
                if show_preview && !results.is_empty() {
                    let result_area = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                        .split(main_chunks[1]);

                    render_list(f, &results, &mut list_state, result_area[0]);
                    render_preview(f, &results, &list_state, result_area[1]);
                } else {
                    render_list(f, &results, &mut list_state, main_chunks[1]);
                }

                // Status bar
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
                    Span::raw(":select "),
                    Span::styled(
                        "Tab",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":preview "),
                    Span::styled(
                        "Ctrl+O",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(":editor "),
                    Span::raw(format!(" | {} results", results.len())),
                ]);
                let status_bar = Paragraph::new(status).style(Style::default().bg(Color::DarkGray));
                f.render_widget(status_bar, main_chunks[2]);
            })?;

            // Handle input with debouncing
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => return Ok(None),
                        KeyCode::Enter => {
                            if let Some(idx) = list_state.selected() {
                                return Ok(results.get(idx).cloned());
                            }
                        }
                        KeyCode::Tab => {
                            show_preview = !show_preview;
                        }
                        KeyCode::Up => {
                            if let Some(idx) = list_state.selected() {
                                if idx > 0 {
                                    list_state.select(Some(idx - 1));
                                }
                            }
                        }
                        KeyCode::Down => {
                            if let Some(idx) = list_state.selected() {
                                if idx + 1 < results.len() {
                                    list_state.select(Some(idx + 1));
                                }
                            }
                        }
                        KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            // Open in editor
                            if let Some(idx) = list_state.selected() {
                                if let Some(result) = results.get(idx) {
                                    let editor = std::env::var("EDITOR")
                                        .unwrap_or_else(|_| "vim".to_string());
                                    let line = result.chunk.start_line;
                                    let file = &result.chunk.file_path;

                                    // Temporarily exit TUI
                                    disable_raw_mode()?;
                                    execute!(io::stdout(), LeaveAlternateScreen)?;

                                    let _ = std::process::Command::new(&editor)
                                        .arg(format!("+{}", line))
                                        .arg(file)
                                        .status();

                                    enable_raw_mode()?;
                                    execute!(io::stdout(), EnterAlternateScreen)?;
                                }
                            }
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

            // Debounced search
            if needs_search && last_search.elapsed() >= debounce && !query.is_empty() {
                if let Ok(emb) = embedder.embed(&query) {
                    results = search::search(&emb, embedding_matrix, top_k, threshold, chunks);
                    if !results.is_empty() {
                        list_state.select(Some(0));
                    } else {
                        list_state.select(None);
                    }
                }
                needs_search = false;
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
                let score_color = if r.score >= 0.7 {
                    Color::Green
                } else if r.score >= 0.5 {
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
    ) {
        let content = if let Some(idx) = list_state.selected() {
            if let Some(result) = results.get(idx) {
                let mut lines = Vec::new();
                for (i, line) in result.chunk.text.lines().enumerate() {
                    let line_num = result.chunk.start_line + i;
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("{:>5} ", line_num),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::raw(line),
                    ]));
                }
                lines
            } else {
                vec![Line::raw("No selection")]
            }
        } else {
            vec![Line::raw("No selection")]
        };

        let preview = Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).title(" Preview "))
            .wrap(Wrap { trim: false });

        f.render_widget(preview, area);
    }
}
