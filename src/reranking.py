"""Pipeline 2: Cross-Encoder re-ranking of the baseline Top-10 candidates."""
import time


def rerank(query, retrieved_chunks, cross_encoder, return_timing=False):
    """Score every (query, chunk) pair jointly and re-sort by Cross-Encoder score.

    The ms-marco Cross-Encoder outputs raw relevance logits (unbounded, not
    probabilities). They are only used for ordering here.
    """
    pairs = [(query, c["text"]) for c in retrieved_chunks]
    t0 = time.perf_counter()
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    t1 = time.perf_counter()

    scored = [dict(c, ce_score=float(s)) for c, s in zip(retrieved_chunks, scores)]
    # stable sort; ties keep the original baseline order
    scored.sort(key=lambda c: -c["ce_score"])
    for new_rank, c in enumerate(scored, start=1):
        c["reranked_rank"] = new_rank
        c["rank_change"] = c["baseline_rank"] - new_rank  # positive = promoted

    assert_same_candidates(retrieved_chunks, scored)
    if return_timing:
        return scored, {"rerank_s": t1 - t0}
    return scored


def assert_same_candidates(baseline, reranked):
    """Invariant: re-ranking only reorders; it never adds or drops candidates."""
    b = sorted(c["chunk_id"] for c in baseline)
    r = sorted(c["chunk_id"] for c in reranked)
    assert b == r, f"candidate sets differ: {b} vs {r}"
    assert len(r) == len(set(r)), "duplicate chunk ids in candidates"
