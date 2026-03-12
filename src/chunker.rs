use crate::types::Chunk;
use tokenizers::Tokenizer;

/// Chunk a file's content into overlapping token-window chunks, snapping to line boundaries.
pub fn chunk_file(
    file_path: &str,
    content: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    tokenizer: &Tokenizer,
) -> Vec<Chunk> {
    if content.is_empty() {
        return vec![];
    }

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    // Tokenize the full content to get token counts per line
    let line_token_counts: Vec<usize> = lines
        .iter()
        .map(|line| {
            tokenizer
                .encode(*line, false)
                .map(|enc| enc.get_ids().len())
                .unwrap_or(0)
        })
        .collect();

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
        Tokenizer::from_bytes(bytes).unwrap()
    }

    #[test]
    fn test_small_file_single_chunk() {
        let tokenizer = make_tokenizer();
        let content = "hello world\nthis is a test";
        let chunks = chunk_file("test.txt", content, 200, 50, &tokenizer);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
    }

    #[test]
    fn test_empty_content() {
        let tokenizer = make_tokenizer();
        let chunks = chunk_file("test.txt", "", 200, 50, &tokenizer);
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
        let chunks = chunk_file("test.txt", &content, 50, 10, &tokenizer);
        assert!(chunks.len() > 1);

        // Verify all chunks have valid line ranges
        for chunk in &chunks {
            assert!(chunk.start_line >= 1);
            assert!(chunk.end_line >= chunk.start_line);
            assert!(chunk.end_line <= 100);
        }
    }
}
