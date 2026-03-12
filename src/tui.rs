pub mod interactive {
    use crate::embedder::Embedder;
    use crate::index::Index;
    use crate::output::{SCORE_HIGH_THRESHOLD, SCORE_MEDIUM_THRESHOLD};
    use crate::pipeline;
    use crate::search;
    use crate::types::{Chunk, SearchResult};
    use crate::walker::WalkedFile;
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
    use std::sync::mpsc::{Receiver, TryRecvError};
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

        let result = run_loop(
            &mut terminal,
            embedder,
            chunks,
            embedding_matrix,
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

    /// Convert a walker-relative path to a project-root-relative path.
    fn to_project_relative(walker_path: &str, cwd_suffix: &Path) -> String {
        let stripped = walker_path.strip_prefix("./").unwrap_or(walker_path);
        if cwd_suffix.as_os_str().is_empty() {
            stripped.to_string()
        } else {
            format!("{}/{}", cwd_suffix.display(), stripped)
        }
    }

    /// Convert a project-root-relative path to a cwd-relative path for file I/O.
    fn to_cwd_path(project_path: &str, cwd_suffix: &Path) -> String {
        if cwd_suffix.as_os_str().is_empty() {
            return project_path.to_string();
        }
        let prefix = format!("{}/", cwd_suffix.display());
        if let Some(rest) = project_path.strip_prefix(&prefix) {
            rest.to_string()
        } else {
            // Path is outside cwd subtree, compute relative path
            let from_comps: Vec<_> = cwd_suffix.components().collect();
            let to = Path::new(project_path);
            let to_comps: Vec<_> = to.components().collect();
            let common = from_comps
                .iter()
                .zip(to_comps.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let mut result = std::path::PathBuf::new();
            for _ in 0..(from_comps.len() - common) {
                result.push("..");
            }
            for comp in &to_comps[common..] {
                result.push(comp);
            }
            result.to_string_lossy().to_string()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_streaming(
        embedder: &mut Embedder,
        idx: &Index,
        rx: Receiver<WalkedFile>,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
        chunk_size: usize,
        chunk_overlap: usize,
        cwd_suffix: &Path,
    ) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = run_streaming_loop(
            &mut terminal,
            embedder,
            idx,
            rx,
            initial_query,
            top_k,
            threshold,
            chunk_size,
            chunk_overlap,
            cwd_suffix,
        );

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_streaming_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        embedder: &mut Embedder,
        idx: &Index,
        rx: Receiver<WalkedFile>,
        initial_query: &str,
        top_k: usize,
        threshold: f32,
        chunk_size: usize,
        chunk_overlap: usize,
        cwd_suffix: &Path,
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

        // Streaming indexing state
        let mut indexing_done = false;
        let mut indexed_count: usize = 0;
        let mut last_reload = Instant::now() - Duration::from_secs(10);

        // Load initial data from index
        let (mut chunks, mut embedding_matrix) = idx.load_all()?;

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
            // 1. Process files from channel (non-blocking, up to 4 files per iteration)
            if !indexing_done {
                let mut batch: Vec<(WalkedFile, String)> = Vec::new();
                loop {
                    match rx.try_recv() {
                        Ok(mut file) => {
                            file.rel_path = to_project_relative(&file.rel_path, cwd_suffix);
                            let hash = blake3::hash(file.content.as_bytes()).to_hex().to_string();
                            let needs_index = match idx.get_file_hash(&file.rel_path) {
                                Ok(Some(stored_hash)) => stored_hash != hash,
                                _ => true,
                            };
                            if needs_index {
                                batch.push((file, hash));
                            }
                            if batch.len() >= 4 {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            indexing_done = true;
                            break;
                        }
                    }
                }
                if !batch.is_empty() {
                    indexed_count += batch.len();
                    pipeline::process_batch(embedder, idx, &batch, chunk_size, chunk_overlap)?;
                    // Rate-limit reloads to every 2 seconds
                    if last_reload.elapsed() >= Duration::from_secs(2) {
                        let (new_chunks, new_matrix) = idx.load_all()?;
                        chunks = new_chunks;
                        embedding_matrix = new_matrix;
                        needs_search = true;
                        last_reload = Instant::now();
                    }
                }
                // Final reload when indexing completes
                if indexing_done && indexed_count > 0 {
                    let (new_chunks, new_matrix) = idx.load_all()?;
                    chunks = new_chunks;
                    embedding_matrix = new_matrix;
                    needs_search = true;
                }
            }

            // 2. Detect selection change and update preview cache/scroll
            let current_selected = list_state.selected();
            if current_selected != last_selected {
                last_selected = current_selected;
                if let Some(idx) = current_selected {
                    if let Some(result) = results.get(idx) {
                        let path = &result.chunk.file_path;
                        let fs_path = to_cwd_path(path, cwd_suffix);
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
            let status_text = if indexing_done {
                format!(
                    "{} results | {} chunks indexed",
                    results.len(),
                    chunks.len()
                )
            } else {
                format!(
                    "{} results | Indexing: {} files...",
                    results.len(),
                    indexed_count
                )
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

            // 4. Handle input with debouncing
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Enter => {
                            if let Some(idx) = list_state.selected() {
                                if let Some(result) = results.get(idx) {
                                    let line = result.chunk.start_line;
                                    let file = to_cwd_path(&result.chunk.file_path, cwd_suffix);

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

    #[allow(clippy::too_many_arguments)]
    fn run_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        embedder: &mut Embedder,
        chunks: &[Chunk],
        embedding_matrix: &Array2<f32>,
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
        let debounce = Duration::from_millis(300);

        // Preview state
        let mut preview_file_cache: Option<(String, String)> = None; // (path, content)
        let mut preview_scroll: u16 = 0;
        let mut last_selected: Option<usize> = None;

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
            // Detect selection change and update preview cache/scroll
            let current_selected = list_state.selected();
            if current_selected != last_selected {
                last_selected = current_selected;
                if let Some(idx) = current_selected {
                    if let Some(result) = results.get(idx) {
                        let path = &result.chunk.file_path;
                        let fs_path = to_cwd_path(path, cwd_suffix);
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
                        // Auto-scroll to chunk (3 lines above start_line)
                        preview_scroll = (result.chunk.start_line.saturating_sub(4)) as u16;
                    }
                }
            }

            let preview_scroll_val = preview_scroll;
            let preview_cache_ref = &preview_file_cache;

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
                    Span::raw(format!(" | {} results", results.len())),
                ]);
                let status_bar = Paragraph::new(status).style(Style::default().bg(Color::DarkGray));
                f.render_widget(status_bar, main_chunks[2]);
            })?;

            // Handle input with debouncing
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Enter => {
                            if let Some(idx) = list_state.selected() {
                                if let Some(result) = results.get(idx) {
                                    let line = result.chunk.start_line;
                                    let file = to_cwd_path(&result.chunk.file_path, cwd_suffix);

                                    // Exit TUI
                                    disable_raw_mode()?;
                                    execute!(io::stdout(), LeaveAlternateScreen)?;

                                    // Open pager
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
                // Reset selection tracking so preview updates
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
        let content = if let Some(idx) = list_state.selected() {
            if let Some(result) = results.get(idx) {
                if let Some((_, ref file_content)) = file_cache {
                    let chunk_start = result.chunk.start_line;
                    let chunk_end = result.chunk.end_line;
                    let highlight_style = Style::default().bg(Color::DarkGray);

                    file_content
                        .lines()
                        .enumerate()
                        .map(|(i, line)| {
                            let line_num = i + 1; // 1-based
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

    #[cfg(test)]
    mod tests {
        use super::*;

        // --- to_cwd_path tests ---

        #[test]
        fn test_cwd_path_at_project_root() {
            // CWD is project root — path unchanged
            assert_eq!(to_cwd_path("src/main.rs", Path::new("")), "src/main.rs");
        }

        #[test]
        fn test_cwd_path_from_subdirectory() {
            // CWD is src/ — src/main.rs becomes main.rs
            assert_eq!(to_cwd_path("src/main.rs", Path::new("src")), "main.rs");
        }

        #[test]
        fn test_cwd_path_from_nested_subdirectory() {
            // CWD is src/deep/ — src/deep/mod.rs becomes mod.rs
            assert_eq!(
                to_cwd_path("src/deep/mod.rs", Path::new("src/deep")),
                "mod.rs"
            );
        }

        #[test]
        fn test_cwd_path_outside_subtree() {
            // CWD is src/ — lib/foo.rs becomes ../lib/foo.rs
            assert_eq!(to_cwd_path("lib/foo.rs", Path::new("src")), "../lib/foo.rs");
        }

        #[test]
        fn test_cwd_path_root_file_from_subdir() {
            // CWD is src/ — README.md becomes ../README.md
            assert_eq!(to_cwd_path("README.md", Path::new("src")), "../README.md");
        }

        #[test]
        fn test_cwd_path_sibling_deep() {
            // CWD is src/a/ — src/b/foo.rs becomes ../b/foo.rs
            assert_eq!(
                to_cwd_path("src/b/foo.rs", Path::new("src/a")),
                "../b/foo.rs"
            );
        }

        #[test]
        fn test_cwd_path_resolves_to_readable_file() {
            // Create a real file, verify to_cwd_path produces a path that read_to_string can open
            let dir = tempfile::TempDir::new().unwrap();
            let sub = dir.path().join("sub");
            std::fs::create_dir(&sub).unwrap();
            std::fs::write(sub.join("file.txt"), "hello").unwrap();

            // Simulate: project root is dir, CWD is dir, file is sub/file.txt
            let cwd_suffix = Path::new("");
            let fs_path = to_cwd_path("sub/file.txt", cwd_suffix);
            let full = dir.path().join(&fs_path);
            assert_eq!(std::fs::read_to_string(full).unwrap(), "hello");

            // Simulate: project root is dir, CWD is dir/sub, file is sub/file.txt
            let cwd_suffix = Path::new("sub");
            let fs_path = to_cwd_path("sub/file.txt", cwd_suffix);
            assert_eq!(fs_path, "file.txt");
            let full = sub.join(&fs_path);
            assert_eq!(std::fs::read_to_string(full).unwrap(), "hello");
        }
    }
}
