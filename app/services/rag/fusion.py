from app.services.rag.models import Citation


def reciprocal_rank_fusion(
    dense_citations: list[Citation],
    sparse_citations: list[Citation],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 5,
) -> list[Citation]:
    """Merge ranked lists from dense vector search and sparse lexical search using RRF.

    RRF formula:
        RRF_Score(d) = sum_{m in {dense, sparse}} ( weight_m / (k + rank_m(d)) )
    where rank_m(d) is the 1-based index of document d in ranking list m.
    """
    rrf_scores: dict[str, float] = {}
    citation_map: dict[str, Citation] = {}

    # 1. Process dense rankings
    for rank, cite in enumerate(dense_citations, start=1):
        chunk_id = cite.chunk_id
        citation_map[chunk_id] = cite
        score_contribution = dense_weight / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

    # 2. Process sparse rankings
    for rank, cite in enumerate(sparse_citations, start=1):
        chunk_id = cite.chunk_id
        if chunk_id not in citation_map:
            citation_map[chunk_id] = cite
        score_contribution = sparse_weight / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

    # 3. Sort by fused RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    # 4. Construct final ranked citations with fused similarity score
    fused_citations: list[Citation] = []
    for chunk_id, score in sorted_items[:top_k]:
        base_cite = citation_map[chunk_id]
        fused_cite = base_cite.model_copy(update={"similarity": score})
        fused_citations.append(fused_cite)

    return fused_citations


def weighted_score_fusion(
    dense_citations: list[Citation],
    sparse_citations: list[Citation],
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[Citation]:
    """Merge dense and sparse search results using convex weighted score combination.

    Scores from both lists are min-max normalized to [0, 1] before blending:
        Fused_Score(d) = alpha * dense_norm(d) + (1 - alpha) * sparse_norm(d)
    """
    bounded_alpha = max(0.0, min(1.0, alpha))
    citation_map: dict[str, Citation] = {}

    # Normalize dense scores
    dense_scores: dict[str, float] = {}
    if dense_citations:
        min_d = min(c.similarity for c in dense_citations)
        max_d = max(c.similarity for c in dense_citations)
        span_d = (max_d - min_d) if (max_d - min_d) > 1e-9 else 1.0
        for c in dense_citations:
            citation_map[c.chunk_id] = c
            dense_scores[c.chunk_id] = (c.similarity - min_d) / span_d

    # Normalize sparse scores
    sparse_scores: dict[str, float] = {}
    if sparse_citations:
        min_s = min(c.similarity for c in sparse_citations)
        max_s = max(c.similarity for c in sparse_citations)
        span_s = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0
        for c in sparse_citations:
            if c.chunk_id not in citation_map:
                citation_map[c.chunk_id] = c
            sparse_scores[c.chunk_id] = (c.similarity - min_s) / span_s

    # Combine scores
    fused_scores: dict[str, float] = {}
    all_chunk_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    for cid in all_chunk_ids:
        d_val = dense_scores.get(cid, 0.0)
        s_val = sparse_scores.get(cid, 0.0)
        fused_scores[cid] = (bounded_alpha * d_val) + ((1.0 - bounded_alpha) * s_val)

    sorted_items = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    fused_citations: list[Citation] = []
    for cid, score in sorted_items[:top_k]:
        base_cite = citation_map[cid]
        fused_citations.append(base_cite.model_copy(update={"similarity": score}))

    return fused_citations
