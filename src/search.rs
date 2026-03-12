use crate::types::SearchResult;
use ndarray::Array2;

/// Search for the most similar chunks to a query embedding.
///
/// Takes a query embedding and a matrix of chunk embeddings (rows = chunks, cols = embedding dims).
/// Returns top-k results above the threshold, sorted by descending similarity.
pub fn search(
    query_embedding: &[f32],
    embedding_matrix: &Array2<f32>,
    top_k: usize,
    threshold: f32,
    chunks: &[crate::types::Chunk],
) -> Vec<SearchResult> {
    let n_chunks = embedding_matrix.nrows();
    if n_chunks == 0 {
        return vec![];
    }

    // Compute cosine similarities via matrix-vector dot product.
    // Since both query and stored embeddings are L2-normalized,
    // cosine similarity = dot product.
    let query = ndarray::Array1::from_vec(query_embedding.to_vec());
    let scores = embedding_matrix.dot(&query);

    // Collect (index, score) pairs above threshold
    let mut scored: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= threshold)
        .map(|(i, &s)| (i, s))
        .collect();

    // Partial sort for top-k
    if scored.len() > top_k {
        scored.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
    }

    // Sort by descending score
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: chunks[idx].clone(),
            score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;
    use ndarray::Array2;

    fn dummy_chunk(text: &str) -> Chunk {
        Chunk {
            file_path: "test.rs".to_string(),
            text: text.to_string(),
            start_line: 1,
            end_line: 1,
        }
    }

    #[test]
    fn test_search_returns_top_k() {
        let dim = 4;
        let embeddings = Array2::from_shape_vec(
            (3, dim),
            vec![
                1.0, 0.0, 0.0, 0.0, // chunk 0
                0.0, 1.0, 0.0, 0.0, // chunk 1
                0.7, 0.7, 0.0, 0.0, // chunk 2
            ],
        )
        .unwrap();
        let chunks = vec![
            dummy_chunk("chunk 0"),
            dummy_chunk("chunk 1"),
            dummy_chunk("chunk 2"),
        ];
        let query = vec![1.0, 0.0, 0.0, 0.0]; // Most similar to chunk 0

        let results = search(&query, &embeddings, 2, 0.0, &chunks);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.text, "chunk 0");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_threshold() {
        let dim = 4;
        let embeddings = Array2::from_shape_vec(
            (2, dim),
            vec![
                1.0, 0.0, 0.0, 0.0, // chunk 0: similarity = 1.0
                0.0, 1.0, 0.0, 0.0, // chunk 1: similarity = 0.0
            ],
        )
        .unwrap();
        let chunks = vec![dummy_chunk("chunk 0"), dummy_chunk("chunk 1")];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results = search(&query, &embeddings, 10, 0.5, &chunks);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.text, "chunk 0");
    }

    #[test]
    fn test_search_empty() {
        let embeddings = Array2::from_shape_vec((0, 4), vec![]).unwrap();
        let chunks: Vec<Chunk> = vec![];
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = search(&query, &embeddings, 10, 0.0, &chunks);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_larger_than_results() {
        let dim = 4;
        let embeddings = Array2::from_shape_vec(
            (3, dim),
            vec![
                1.0, 0.0, 0.0, 0.0, // chunk 0
                0.0, 1.0, 0.0, 0.0, // chunk 1
                0.5, 0.5, 0.0, 0.0, // chunk 2
            ],
        )
        .unwrap();
        let chunks = vec![
            dummy_chunk("chunk 0"),
            dummy_chunk("chunk 1"),
            dummy_chunk("chunk 2"),
        ];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // top_k=100 but only 3 chunks — should return all 3
        let results = search(&query, &embeddings, 100, 0.0, &chunks);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_descending_order() {
        let dim = 4;
        let embeddings = Array2::from_shape_vec(
            (4, dim),
            vec![
                0.0, 0.0, 1.0, 0.0, // chunk 0: sim = 0.0
                0.5, 0.5, 0.0, 0.0, // chunk 1: sim = 0.5
                1.0, 0.0, 0.0, 0.0, // chunk 2: sim = 1.0
                0.7, 0.3, 0.0, 0.0, // chunk 3: sim = 0.7
            ],
        )
        .unwrap();
        let chunks = vec![
            dummy_chunk("chunk 0"),
            dummy_chunk("chunk 1"),
            dummy_chunk("chunk 2"),
            dummy_chunk("chunk 3"),
        ];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let results = search(&query, &embeddings, 10, 0.0, &chunks);
        // Verify descending order
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results not in descending order: {} < {}",
                results[i - 1].score,
                results[i].score
            );
        }
        // First result should be chunk 2 (sim = 1.0)
        assert_eq!(results[0].chunk.text, "chunk 2");
    }

    #[test]
    fn test_search_all_below_threshold() {
        let dim = 4;
        let embeddings = Array2::from_shape_vec(
            (2, dim),
            vec![
                0.0, 1.0, 0.0, 0.0, // chunk 0: sim = 0.0
                0.0, 0.0, 1.0, 0.0, // chunk 1: sim = 0.0
            ],
        )
        .unwrap();
        let chunks = vec![dummy_chunk("chunk 0"), dummy_chunk("chunk 1")];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Threshold of 0.5 — both have sim = 0.0
        let results = search(&query, &embeddings, 10, 0.5, &chunks);
        assert!(results.is_empty());
    }
}
