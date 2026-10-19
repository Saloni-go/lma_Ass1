"""High-level RAG pipelines over the shared chunk corpus.

Pipeline 1 (baseline):   Bi-Encoder Top-10 -> ranks 1-3 unchanged          -> Gemini (Answer 1)
Pipeline 2 (re-ranking): the SAME Top-10  -> Cross-Encoder re-sort -> top 3 -> Gemini (Answer 2)

Both pipelines accept a precomputed `retrieval` (the output of retrieve()) so
compare_pipelines() can run the vector search exactly once and hand the
identical candidate list to both. The prompt template, LLM model and
generation settings are shared; only the choice of the 3 context chunks differs.
"""
import time

from . import config
from .embeddings import get_device, load_bi_encoder, load_cross_encoder, set_seed
from .generation import (GENERATION_SETTINGS, PROMPT_TEMPLATE, build_prompt, describe_backend,
                         format_context, generate_answer, get_client)
from .process_pdf import load_chunks
from .reranking import rerank
from .retrieval import baseline_retrieve, corpus_fingerprint, load_or_build_index


class RAGPipeline:
    def __init__(self, chunks=None, gemini_model=config.GEMINI_MODEL, verbose=True):
        set_seed(config.SEED)
        self.device = get_device()
        self.chunks = chunks if chunks is not None else load_chunks()
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.source_pdf = self.chunks[0]["source"]
        self.corpus_fingerprint = corpus_fingerprint(self.chunks, config.BI_ENCODER_NAME)

        t0 = time.perf_counter()
        self.bi_encoder = load_bi_encoder(self.device)
        t1 = time.perf_counter()
        self.index, self.id_map, self.embeddings, built = load_or_build_index(
            self.chunks, self.bi_encoder, config.BI_ENCODER_NAME, config.INDEX_DIR)
        t2 = time.perf_counter()
        self.cross_encoder = load_cross_encoder(self.device)
        t3 = time.perf_counter()
        self.setup_timing = {"load_bi_encoder_s": t1 - t0,
                             ("build_index_s" if built else "load_cached_index_s"): t2 - t1,
                             "load_cross_encoder_s": t3 - t2}

        self.gemini_model = gemini_model
        self.client = get_client()  # None when no credentials are configured
        self.llm_backend = describe_backend()
        if verbose:
            print(f"device={self.device} | chunks={len(self.chunks)} | "
                  f"index: {self.index.ntotal} x {self.index.d} "
                  f"({'built and cached' if built else 'loaded from cache'}) | "
                  f"LLM={gemini_model} via {self.llm_backend}")

    # ------------------------------------------------------------------ retrieval
    def baseline_retrieve(self, query, top_k=config.TOP_K_RETRIEVE, return_timing=False):
        return baseline_retrieve(query, self.bi_encoder, self.index, self.chunks_by_id,
                                 self.id_map, top_k=top_k, return_timing=return_timing)

    def retrieve(self, query):
        """Run the Bi-Encoder search once; the result is shareable by both pipelines."""
        t0 = time.perf_counter()
        candidates, rt = self.baseline_retrieve(query, return_timing=True)
        t1 = time.perf_counter()
        return {"query": query, "candidates": candidates, "timing": {
            "query_embed_ms": rt["embed_s"] * 1000,
            "faiss_search_ms": rt["search_s"] * 1000,
            "retrieval_ms": (t1 - t0) * 1000,
        }}

    def _retrieval_for(self, query, retrieval):
        if retrieval is None:
            return self.retrieve(query)
        assert retrieval["query"] == query, "retrieval was computed for a different query"
        return retrieval

    # ------------------------------------------------------------------ generation
    def generate(self, query, context_chunks):
        """Call the LLM with the shared prompt. Never fabricates: on failure the
        answer is None and the error message is returned instead."""
        try:
            answer, latency = generate_answer(query, context_chunks, self.client,
                                              model=self.gemini_model)
            return {"answer": answer, "error": None, "generation_s": latency}
        except Exception as exc:
            return {"answer": None, "error": f"{type(exc).__name__}: {exc}", "generation_s": None}

    # ------------------------------------------------------------------ pipelines
    def baseline_answer(self, query, retrieval=None):
        """Pipeline 1: Top-10 by cosine similarity, the first 3 (no re-ranking) -> Answer 1."""
        r = self._retrieval_for(query, retrieval)
        candidates = r["candidates"]
        t0 = time.perf_counter()
        selected = candidates[:config.TOP_K_CONTEXT]
        t1 = time.perf_counter()
        gen = self.generate(query, selected)
        return {
            "query": query,
            "pipeline": "baseline",
            "candidates": candidates,                       # Top-10, shared with Pipeline 2
            "selected_chunk_ids": [c["chunk_id"] for c in selected],
            "prompt": build_prompt(query, selected),
            "answer": gen["answer"],
            "error": gen["error"],
            "llm_model": self.gemini_model,
            "timing": {
                **r["timing"],
                "selection_ms": (t1 - t0) * 1000,
                # retrieval-side latency of Pipeline 1 (what the user waits for before the LLM)
                "pipeline_retrieval_ms": r["timing"]["retrieval_ms"] + (t1 - t0) * 1000,
                "generation_ms": gen["generation_s"] * 1000 if gen["generation_s"] else None,
            },
        }

    def advanced_answer(self, query, retrieval=None):
        """Pipeline 2: re-score exactly the baseline Top-10 with the Cross-Encoder,
        take the 3 highest-scoring chunks -> Answer 2 (same model and prompt)."""
        r = self._retrieval_for(query, retrieval)
        candidates = r["candidates"]
        t0 = time.perf_counter()
        reranked, rr = rerank(query, candidates, self.cross_encoder, return_timing=True)
        selected = reranked[:config.TOP_K_CONTEXT]
        t1 = time.perf_counter()
        gen = self.generate(query, selected)
        return {
            "query": query,
            "pipeline": "cross_encoder_rerank",
            "candidates": candidates,                       # original Top-10, original ranks
            "reranked": [{k: c[k] for k in ("chunk_id", "baseline_rank", "cosine_score",
                                            "ce_score", "reranked_rank", "rank_change",
                                            "page_start", "page_end")} for c in reranked],
            "selected_chunk_ids": [c["chunk_id"] for c in selected],
            "prompt": build_prompt(query, selected),
            "answer": gen["answer"],
            "error": gen["error"],
            "llm_model": self.gemini_model,
            "timing": {
                **r["timing"],
                "ce_predict_ms": rr["rerank_s"] * 1000,     # model forward pass on 10 pairs
                "rerank_ms": (t1 - t0) * 1000,              # pairs + predict + sort + top-3
                "pipeline_retrieval_ms": r["timing"]["retrieval_ms"] + (t1 - t0) * 1000,
                "generation_ms": gen["generation_s"] * 1000 if gen["generation_s"] else None,
            },
        }

    def compare_pipelines(self, query):
        """Retrieve once, run both pipelines on the identical Top-10, verify fairness."""
        retrieval = self.retrieve(query)
        a1 = self.baseline_answer(query, retrieval)
        a2 = self.advanced_answer(query, retrieval)
        checks = self.verify_comparison(a1, a2)
        rank_changes = [
            {"chunk_id": c["chunk_id"], "baseline_rank": c["baseline_rank"],
             "reranked_rank": c["reranked_rank"], "rank_change": c["rank_change"],
             "cosine_score": c["cosine_score"], "ce_score": c["ce_score"],
             "pages": [c["page_start"], c["page_end"]]}
            for c in sorted(a2["reranked"], key=lambda c: c["baseline_rank"])
        ]
        b_top, r_top = a1["selected_chunk_ids"], a2["selected_chunk_ids"]
        return {
            "query": query,
            "source_pdf": self.source_pdf,
            "corpus_fingerprint": self.corpus_fingerprint,
            "llm_model": self.gemini_model,
            "candidates_top10": [{k: c[k] for k in ("chunk_id", "baseline_rank", "cosine_score",
                                                    "page_start", "page_end", "text")}
                                 for c in retrieval["candidates"]],
            "baseline_top3": b_top,
            "reranked_top3": r_top,
            "top3_overlap": sorted(set(b_top) & set(r_top)),
            "promoted_into_top3": [c for c in r_top if c not in b_top],
            "dropped_from_top3": [c for c in b_top if c not in r_top],
            "rank_changes": rank_changes,
            "answer_1": a1["answer"], "answer_1_error": a1["error"],
            "answer_2": a2["answer"], "answer_2_error": a2["error"],
            "prompt_1": a1["prompt"], "prompt_2": a2["prompt"],
            "timing": {"baseline": a1["timing"], "reranked": a2["timing"]},
            "checks": checks,
        }

    def verify_comparison(self, a1, a2):
        """Assert the experimental controls: only the top-3 context selection may differ."""
        ids1 = [c["chunk_id"] for c in a1["candidates"]]
        ids2 = [c["chunk_id"] for c in a2["candidates"]]
        reranked_ids = sorted(c["chunk_id"] for c in a2["reranked"])

        def strip_context(prompt, selected_ids):
            ctx = format_context([self.chunks_by_id[i] for i in selected_ids])
            return prompt.replace(ctx, "<CONTEXT>")

        checks = {
            "same_source_pdf": all(self.chunks_by_id[i]["source"] == self.source_pdf
                                   for i in ids1 + ids2),
            "same_top10_ids_and_order": ids1 == ids2,
            "same_top10_scores": [c["cosine_score"] for c in a1["candidates"]]
                                 == [c["cosine_score"] for c in a2["candidates"]],
            "reranked_set_equals_top10": reranked_ids == sorted(ids1),
            "reranked_top3_from_top10": set(a2["selected_chunk_ids"]) <= set(ids1),
            "both_use_3_chunks": len(a1["selected_chunk_ids"]) == len(a2["selected_chunk_ids"])
                                 == config.TOP_K_CONTEXT,
            "same_llm_model": a1["llm_model"] == a2["llm_model"],
            "same_prompt_outside_context": strip_context(a1["prompt"], a1["selected_chunk_ids"])
                                           == strip_context(a2["prompt"], a2["selected_chunk_ids"])
                                           == PROMPT_TEMPLATE.format(context="<CONTEXT>",
                                                                     query=a1["query"]),
            "generation_settings": dict(GENERATION_SETTINGS),
        }
        failed = [k for k, v in checks.items() if v is False]
        assert not failed, f"comparison controls violated: {failed}"
        return checks
