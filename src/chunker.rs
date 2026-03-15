use crate::types::Chunk;
use tokenizers::Tokenizer;

/// Estimate token count from text length.
/// Uses ~2.5 chars/token — URLs, markdown, and code tokenize more densely
/// than plain English (~4 chars/token). Conservative to avoid exceeding
/// remote model context limits.
fn estimate_tokens(text: &str) -> usize {
    // Multiply by 2 then divide by 5 to approximate dividing by 2.5
    (text.len() * 2).div_ceil(5)
}

/// Chunk a file's content into overlapping token-window chunks, snapping to line boundaries.
/// If `tokenizer` is `None`, uses a character-based heuristic for token counting.
pub fn chunk_file(
    file_path: &str,
    content: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    tokenizer: Option<&Tokenizer>,
) -> Vec<Chunk> {
    if content.is_empty() {
        return vec![];
    }

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    // Get token counts per line — use tokenizer if available, otherwise estimate
    let line_token_counts: Vec<usize> = match tokenizer {
        Some(tok) => lines
            .iter()
            .map(|line| {
                tok.encode(*line, false)
                    .map(|enc| enc.get_ids().len())
                    .unwrap_or(0)
            })
            .collect(),
        None => lines.iter().map(|line| estimate_tokens(line)).collect(),
    };

    let total_tokens: usize = line_token_counts.iter().sum();

    // If the entire file fits in one chunk, return it as a single chunk
    if total_tokens <= chunk_size {
        return vec![Chunk {
            file_path: file_path.to_string(),
            text: content.to_string(),
            start_line: 1,
            end_line: lines.len(),
        }];
    }

    let mut chunks = Vec::new();
    let mut start_line_idx = 0;

    while start_line_idx < lines.len() {
        let mut token_count = 0;
        let mut end_line_idx = start_line_idx;

        // Accumulate lines until we reach chunk_size tokens
        while end_line_idx < lines.len()
            && token_count + line_token_counts[end_line_idx] <= chunk_size
        {
            token_count += line_token_counts[end_line_idx];
            end_line_idx += 1;
        }

        // If we couldn't fit even one line, include it anyway to avoid infinite loops
        if end_line_idx == start_line_idx {
            end_line_idx = start_line_idx + 1;
        }

        let chunk_text: String = lines[start_line_idx..end_line_idx].join("\n");

        chunks.push(Chunk {
            file_path: file_path.to_string(),
            text: chunk_text,
            start_line: start_line_idx + 1, // 1-based
            end_line: end_line_idx,         // 1-based inclusive
        });

        // Advance by (chunk_size - overlap) worth of tokens, snapping to lines
        let advance_tokens = if chunk_size > chunk_overlap {
            chunk_size - chunk_overlap
        } else {
            chunk_size
        };

        let mut advanced = 0;
        let mut next_start = start_line_idx;
        while next_start < end_line_idx && advanced < advance_tokens {
            advanced += line_token_counts[next_start];
            next_start += 1;
        }

        // Ensure we advance at least one line
        if next_start == start_line_idx {
            next_start = start_line_idx + 1;
        }

        start_line_idx = next_start;
    }

    // Skip trailing fragments smaller than overlap
    if chunks.len() > 1 {
        let last = chunks.last().unwrap();
        let last_tokens: usize = (last.start_line..=last.end_line)
            .filter_map(|l| line_token_counts.get(l - 1))
            .sum();
        if last_tokens < chunk_overlap {
            chunks.pop();
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> Tokenizer {
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/models/tokenizer.json"));
        let mut tok = Tokenizer::from_bytes(bytes).unwrap();
        tok.with_padding(None);
        tok
    }

    #[test]
    fn test_small_file_single_chunk() {
        let tokenizer = make_tokenizer();
        let content = "hello world\nthis is a test";
        let chunks = chunk_file("test.txt", content, 500, 50, Some(&tokenizer));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
    }

    #[test]
    fn test_empty_content() {
        let tokenizer = make_tokenizer();
        let chunks = chunk_file("test.txt", "", 200, 50, Some(&tokenizer));
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_large_file_multiple_chunks() {
        let tokenizer = make_tokenizer();
        // Create content with many lines
        let lines: Vec<String> = (0..100)
            .map(|i| format!("This is line number {} with some content to fill tokens", i))
            .collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.txt", &content, 50, 10, Some(&tokenizer));
        assert!(chunks.len() > 1);

        // Verify all chunks have valid line ranges
        for chunk in &chunks {
            assert!(chunk.start_line >= 1);
            assert!(chunk.end_line >= chunk.start_line);
            assert!(chunk.end_line <= 100);
        }
    }

    #[test]
    fn test_overlap_larger_than_chunk() {
        let tokenizer = make_tokenizer();
        let lines: Vec<String> = (0..20)
            .map(|i| format!("Line {} with some words", i))
            .collect();
        let content = lines.join("\n");
        // overlap >= chunk_size: should not panic, advances by chunk_size
        let chunks = chunk_file("test.txt", &content, 10, 10, Some(&tokenizer));
        assert!(!chunks.is_empty());
        // overlap > chunk_size
        let chunks = chunk_file("test.txt", &content, 10, 20, Some(&tokenizer));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_single_very_long_line() {
        let tokenizer = make_tokenizer();
        // A single line with many words that exceeds chunk_size tokens
        let long_line = "word ".repeat(500);
        let chunks = chunk_file("test.txt", &long_line, 10, 2, Some(&tokenizer));
        // Should produce at least one chunk even though the line exceeds chunk_size
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn test_chunk_boundaries_are_line_aligned() {
        let tokenizer = make_tokenizer();
        let lines: Vec<String> = (0..50)
            .map(|i| format!("This is line number {}", i))
            .collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.txt", &content, 30, 5, Some(&tokenizer));

        for chunk in &chunks {
            // No chunk text should start or end with a partial line
            // (i.e., the text split on \n should match whole lines from the original)
            for line in chunk.text.lines() {
                assert!(
                    lines.contains(&line.to_string()),
                    "Chunk contains partial line: {:?}",
                    line
                );
            }
        }
    }

    #[test]
    fn test_chunks_overlap_and_cover_file() {
        let tokenizer = make_tokenizer();
        // Use enough lines and small enough chunk_size to ensure multiple chunks with overlap.
        let lines: Vec<String> = (0..200)
            .map(|i| format!("Line {} with some extra content to consume tokens", i))
            .collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.txt", &content, 200, 50, Some(&tokenizer));
        assert!(chunks.len() >= 2, "Need multiple chunks to test overlap");

        // First chunk starts at line 1
        assert_eq!(chunks[0].start_line, 1);

        for pair in chunks.windows(2) {
            // Adjacent chunks must not have gaps (next starts at or before previous ends + 1)
            assert!(
                pair[1].start_line <= pair[0].end_line + 1,
                "Gap between chunks: [{}-{}] and [{}-{}]",
                pair[0].start_line,
                pair[0].end_line,
                pair[1].start_line,
                pair[1].end_line
            );
        }
    }

    // --- estimate_tokens tests ---

    #[test]
    fn test_estimate_tokens_short() {
        // 5 chars → 2 tokens (5 * 2 / 5 = 2)
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_conservative() {
        // 1280 chars should estimate to 512 tokens (1280 * 2 / 5 = 512)
        // This matches Ollama's 512-token context limit
        let text = "x".repeat(1280);
        assert_eq!(estimate_tokens(&text), 512);
    }

    #[test]
    fn test_estimate_tokens_url_heavy() {
        // URL-heavy text should produce more tokens per char
        // estimate_tokens is conservative so it over-counts, which is safe
        let url_text = "https://example.com/path/to/resource?param=value&other=123\n";
        let estimated = estimate_tokens(url_text);
        // The estimate should be >= what the actual tokenizer would produce
        // At 2.5 chars/token, 58 chars → ~23 tokens
        assert!(
            estimated >= 20,
            "estimated {estimated} too low for URL text"
        );
    }

    // --- chunking without tokenizer (remote embedder path) ---

    #[test]
    fn test_chunk_without_tokenizer_respects_size() {
        // Without tokenizer, chunk_size=100 tokens ≈ 250 chars at 2.5 chars/token
        let lines: Vec<String> = (0..50)
            .map(|i| format!("Line {} with some content", i))
            .collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.txt", &content, 100, 20, None);

        assert!(chunks.len() > 1, "should produce multiple chunks");
        // All chunks should have text shorter than ~250 chars (100 tokens * 2.5)
        for chunk in &chunks {
            let estimated = estimate_tokens(&chunk.text);
            assert!(
                estimated <= 120, // some slack for line snapping
                "chunk has {estimated} estimated tokens, expected <=120"
            );
        }
    }

    #[test]
    fn test_chunk_without_tokenizer_covers_file() {
        let lines: Vec<String> = (0..30).map(|i| format!("Line {}", i)).collect();
        let content = lines.join("\n");
        let chunks = chunk_file("test.txt", &content, 20, 5, None);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_line, 1);

        // Check coverage — no gaps
        for pair in chunks.windows(2) {
            assert!(
                pair[1].start_line <= pair[0].end_line + 1,
                "Gap between chunks"
            );
        }
    }
}
