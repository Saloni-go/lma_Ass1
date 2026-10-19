"""Legacy prototype helpers (first scaffold). NOT used by the final pipeline; metrics are in src/stress_test.py and timings in src/benchmark_latency.py. Kept for reference."""
import time

import numpy as np
import pandas as pd


def first_relevant_rank(ranked_ids, relevant_ids):
    """1-based rank of the first relevant chunk, or None if absent from the list."""
    for r, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return r
    return None


def retrieval_metrics(ranked_ids, relevant_ids, k=3):
    """Hit@k, Recall@k and reciprocal rank for one query.

    ranked_ids    -- the ordered Top-10 candidate chunk ids for one pipeline
    relevant_ids  -- chunk ids manually judged to contain the supporting evidence
    Recall@k is computed over the relevant chunks that are reachable in the
    Top-10 candidate pool (re-ranking cannot recover chunks outside it).
    """
    top_k = ranked_ids[:k]
    rel_in_pool = [c for c in relevant_ids if c in ranked_ids]
    hits = [c for c in top_k if c in relevant_ids]
    rank = first_relevant_rank(ranked_ids, relevant_ids)
    return {
        f"hit@{k}": int(len(hits) > 0),
        f"recall@{k}": len(hits) / len(rel_in_pool) if rel_in_pool else np.nan,
        "first_rel_rank": rank,
        "rr": 1.0 / rank if rank else 0.0,
    }


def rank_comparison_table(reranked):
    """Per-candidate table: baseline rank vs re-ranked rank vs change."""
    df = pd.DataFrame(reranked)[[
        "chunk_id", "page_start", "page_end", "baseline_rank", "cosine_score",
        "reranked_rank", "ce_score", "rank_change",
    ]]
    return df.sort_values("baseline_rank").reset_index(drop=True)


def time_call(fn, repeats=5):
    """Run fn() `repeats` times and return a list of wall-clock durations (s)."""
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def summarise(values):
    v = np.asarray(values, dtype=float)
    return {"mean_ms": v.mean() * 1000, "std_ms": v.std(ddof=1) * 1000 if len(v) > 1 else 0.0,
            "min_ms": v.min() * 1000, "max_ms": v.max() * 1000, "n": len(v)}
