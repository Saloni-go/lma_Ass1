"""Stage 5: stress-test both pipelines on the verified question set.

Phase 1 (retrieval, all questions, no LLM):
    python -m src.stress_test retrieval
Phase 2 (Gemini answers for the selected cases only; cached, free-tier paced):
    python -m src.stress_test generate

Definitions (relevance labels = relevant_chunk_ids from data/evaluation/questions.json)
    Recall@3          |relevant ∩ top-3| / |relevant|
    Recall@10         |relevant ∩ top-10| / |relevant|   (same for both pipelines)
    MRR@10            1 / rank of the first relevant chunk in the 10-list (0 if none)
    evidence_cov@3    fraction of the question's evidence quotes that occur in at least
                      one top-3 chunk (a quote may occur in two overlapping chunks)
    promoted          relevant chunk with baseline rank 4-10 and re-ranked rank 1-3
    demoted           relevant chunk with baseline rank 1-3 and re-ranked rank 4-10
    outcome           compares evidence_cov@3 first; if equal, the reciprocal rank:
                      improved / unchanged / worsened
    rescue            relevant evidence missing from the baseline top-3 (cov@3 < 1),
                      present in the baseline top-10, and promoted into the re-ranked
                      top-3 so that cov@3 increases
    strict_rescue     rescue where the baseline top-3 contained NO relevant chunk
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone

from . import config
from .generation import GENERATION_SETTINGS
from .pipeline import RAGPipeline
from .reranking import rerank

QUESTIONS_JSON = config.DATA_DIR / "evaluation" / "questions.json"
OUT_DIR = config.RESULTS_DIR / "stress_test"
RETRIEVAL_JSON = OUT_DIR / "retrieval_results.json"
RETRIEVAL_CSV = OUT_DIR / "retrieval_results.csv"
ANSWERS_JSON = OUT_DIR / "generation_results.json"
K = config.TOP_K_CONTEXT
FREE_TIER_PAUSE_S = 13  # 5 requests/minute on the free tier


def recall_at(ids, relevant, k):
    return len(set(ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ids, relevant):
    return next((1.0 / r for r, c in enumerate(ids, 1) if c in relevant), 0.0)


def evidence_coverage(top_ids, evidence):
    top = set(top_ids)
    return sum(bool(top & set(e["chunk_ids"])) for e in evidence) / len(evidence)


def evaluate_question(rag, q):
    retrieval = rag.retrieve(q["question"])
    cands = retrieval["candidates"]
    reranked, rr = rerank(q["question"], cands, rag.cross_encoder, return_timing=True)
    base_ids = [c["chunk_id"] for c in cands]
    rr_ids = [c["chunk_id"] for c in reranked]
    rel, dis = set(q["relevant_chunk_ids"]), set(q["distractor_chunk_ids"])

    m = {}
    for name, ids in (("baseline", base_ids), ("reranked", rr_ids)):
        m[name] = {
            "top3": ids[:K],
            "recall@3": recall_at(ids, rel, K),
            "hit@3": int(bool(set(ids[:K]) & rel)),
            "mrr@10": reciprocal_rank(ids, rel),
            "evidence_cov@3": evidence_coverage(ids[:K], q["evidence"]),
            "distractors_in_top3": sorted(set(ids[:K]) & dis),
        }
    by_id = {c["chunk_id"]: c for c in reranked}
    promoted = sorted(c for c in rel & set(base_ids)
                      if by_id[c]["baseline_rank"] > K and by_id[c]["reranked_rank"] <= K)
    demoted = sorted(c for c in rel & set(base_ids)
                     if by_id[c]["baseline_rank"] <= K and by_id[c]["reranked_rank"] > K)

    b, r = m["baseline"], m["reranked"]
    key = lambda x: (x["evidence_cov@3"], x["mrr@10"])
    outcome = "improved" if key(r) > key(b) else "worsened" if key(r) < key(b) else "unchanged"
    # did the outcome come from top-3 evidence coverage, or only from a rank move?
    outcome_basis = ("none" if outcome == "unchanged" else
                     "evidence_cov@3" if r["evidence_cov@3"] != b["evidence_cov@3"] else "rank_only")
    rescue = b["evidence_cov@3"] < 1 and bool(promoted) and r["evidence_cov@3"] > b["evidence_cov@3"]

    return {
        "query_id": q["query_id"], "category": q["category"], "question": q["question"],
        "relevant_chunk_ids": sorted(rel), "distractor_chunk_ids": sorted(dis),
        "relevant_in_top10": sorted(rel & set(base_ids)),
        "recall@10": recall_at(base_ids, rel, 10),
        "baseline": b, "reranked": r,
        "relevant_rank_shifts": [
            {"chunk_id": c, "baseline_rank": by_id[c]["baseline_rank"],
             "reranked_rank": by_id[c]["reranked_rank"]} for c in sorted(rel & set(base_ids))],
        "promoted_into_top3": promoted, "demoted_from_top3": demoted,
        "outcome": outcome, "outcome_basis": outcome_basis, "rescue": rescue,
        "strict_rescue": rescue and b["hit@3"] == 0,
        "candidates": [{
            "chunk_id": c["chunk_id"], "baseline_rank": c["baseline_rank"],
            "cosine_score": round(c["cosine_score"], 4), "ce_score": round(c["ce_score"], 4),
            "reranked_rank": c["reranked_rank"], "rank_change": c["rank_change"],
            "relevant": c["chunk_id"] in rel, "distractor": c["chunk_id"] in dis,
            "pages": [c["page_start"], c["page_end"]],
        } for c in sorted(reranked, key=lambda c: c["baseline_rank"])],
        "timing_ms": {"retrieval": retrieval["timing"]["retrieval_ms"],
                      "ce_predict": rr["rerank_s"] * 1000},
    }


def aggregate(rows):
    n = len(rows)
    mean = lambda f: sum(f(r) for r in rows) / n
    agg = {"n_questions": n}
    for p in ("baseline", "reranked"):
        agg[p] = {k: round(mean(lambda r: r[p][k]), 4)
                  for k in ("recall@3", "hit@3", "mrr@10", "evidence_cov@3")}
    agg["recall@10"] = round(mean(lambda r: r["recall@10"]), 4)
    agg["questions_with_no_relevant_in_top10"] = [r["query_id"] for r in rows if not r["relevant_in_top10"]]
    agg["relevant_chunks_promoted_4to10_into_top3"] = sum(len(r["promoted_into_top3"]) for r in rows)
    agg["relevant_chunks_demoted_from_top3"] = sum(len(r["demoted_from_top3"]) for r in rows)
    for o in ("improved", "unchanged", "worsened"):
        agg[o] = [r["query_id"] for r in rows if r["outcome"] == o]
        agg[f"{o}_top3_coverage"] = [r["query_id"] for r in rows
                                     if r["outcome"] == o and r["outcome_basis"] == "evidence_cov@3"]
    agg["rescues"] = [r["query_id"] for r in rows if r["rescue"]]
    agg["strict_rescues"] = [r["query_id"] for r in rows if r["strict_rescue"]]
    return agg


def run_retrieval():
    questions = json.loads(QUESTIONS_JSON.read_text())["questions"]
    rag = RAGPipeline()
    rerank("warm-up", rag.retrieve("warm-up")["candidates"], rag.cross_encoder)
    rows = [evaluate_question(rag, q) for q in questions]

    by_cat = {}
    for cat in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = aggregate(sub)
    out = {"metadata": {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_pdf": rag.source_pdf, "corpus_fingerprint": rag.corpus_fingerprint,
        "bi_encoder": config.BI_ENCODER_NAME, "cross_encoder": config.CROSS_ENCODER_NAME,
        "definitions": __doc__.split("Definitions", 1)[1].strip()},
        "overall": aggregate(rows), "by_category": by_cat, "questions": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(RETRIEVAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "category", "relevant_chunk_ids", "relevant_rank_shifts",
                    "baseline_top3", "reranked_top3", "base_recall@3", "rr_recall@3",
                    "base_mrr@10", "rr_mrr@10", "base_evidence_cov@3", "rr_evidence_cov@3",
                    "recall@10", "promoted", "demoted", "outcome", "outcome_basis", "rescue",
                    "strict_rescue"])
        for r in rows:
            w.writerow([r["query_id"], r["category"], ";".join(map(str, r["relevant_chunk_ids"])),
                        "; ".join(f"{s['chunk_id']}:{s['baseline_rank']}->{s['reranked_rank']}"
                                  for s in r["relevant_rank_shifts"]),
                        ";".join(map(str, r["baseline"]["top3"])), ";".join(map(str, r["reranked"]["top3"])),
                        f"{r['baseline']['recall@3']:.3f}", f"{r['reranked']['recall@3']:.3f}",
                        f"{r['baseline']['mrr@10']:.3f}", f"{r['reranked']['mrr@10']:.3f}",
                        f"{r['baseline']['evidence_cov@3']:.3f}", f"{r['reranked']['evidence_cov@3']:.3f}",
                        f"{r['recall@10']:.3f}", ";".join(map(str, r["promoted_into_top3"])),
                        ";".join(map(str, r["demoted_from_top3"])), r["outcome"], r["outcome_basis"], r["rescue"],
                        r["strict_rescue"]])
    print_summary(out)


def print_summary(out):
    o = out["overall"]
    print(f"\n=== Retrieval metrics over {o['n_questions']} questions ===")
    print(f"{'':<16}{'Recall@3':>9}{'Hit@3':>7}{'MRR@10':>8}{'EvCov@3':>9}")
    for p in ("baseline", "reranked"):
        x = o[p]
        print(f"{p:<16}{x['recall@3']:>9.3f}{x['hit@3']:>7.3f}{x['mrr@10']:>8.3f}{x['evidence_cov@3']:>9.3f}")
    print(f"Recall@10 (shared candidate pool): {o['recall@10']:.3f}; "
          f"no relevant chunk in top-10: {o['questions_with_no_relevant_in_top10'] or 'none'}")
    print(f"relevant chunks promoted 4-10 -> top-3: {o['relevant_chunks_promoted_4to10_into_top3']} | "
          f"demoted top-3 -> 4-10: {o['relevant_chunks_demoted_from_top3']}")
    print(f"improved {len(o['improved'])} {o['improved']} (top-3 coverage gain: {o['improved_top3_coverage']})")
    print(f"unchanged {len(o['unchanged'])}")
    print(f"worsened {len(o['worsened'])} {o['worsened']} (top-3 coverage loss: {o['worsened_top3_coverage']})")
    print(f"rescues {o['rescues']} | strict rescues (baseline top-3 had no relevant chunk) {o['strict_rescues']}")

    print(f"\n{'category':<20}{'n':>3}{'R@3 base':>10}{'R@3 rr':>8}{'MRR base':>10}{'MRR rr':>8}  imp/unch/worse")
    for cat, a in out["by_category"].items():
        print(f"{cat:<20}{a['n_questions']:>3}{a['baseline']['recall@3']:>10.3f}{a['reranked']['recall@3']:>8.3f}"
              f"{a['baseline']['mrr@10']:>10.3f}{a['reranked']['mrr@10']:>8.3f}  "
              f"{len(a['improved'])}/{len(a['unchanged'])}/{len(a['worsened'])}")

    print(f"\n{'id':<6} {'relevant: base->rr rank':<28} {'base top3':<14} {'rr top3':<14} "
          f"{'cov3 b->r':<10} outcome")
    for r in out["questions"]:
        shifts = ", ".join(f"{s['chunk_id']}:#{s['baseline_rank']}->#{s['reranked_rank']}"
                           for s in r["relevant_rank_shifts"]) or "not in top-10"
        flag = " STRICT-RESCUE" if r["strict_rescue"] else " rescue" if r["rescue"] else ""
        print(f"{r['query_id']:<6} {shifts:<28} {str(r['baseline']['top3']):<14} "
              f"{str(r['reranked']['top3']):<14} "
              f"{r['baseline']['evidence_cov@3']:.2f}->{r['reranked']['evidence_cov@3']:.2f}  "
              f"{r['outcome']}{' (rank only)' if r['outcome_basis'] == 'rank_only' else ''}{flag}")


def run_generation(query_ids):
    """Answer 1 and Answer 2 for the given questions, with both pipelines sharing
    one retrieval. Results are cached per question so quota is not re-spent."""
    questions = {q["query_id"]: q for q in json.loads(QUESTIONS_JSON.read_text())["questions"]}
    cache = json.loads(ANSWERS_JSON.read_text()) if ANSWERS_JSON.exists() else {"cases": {}}
    rag = RAGPipeline()
    for qid in query_ids:
        prev = cache["cases"].get(qid)
        if prev and prev["answer_1"] and prev["answer_2"]:
            print(f"{qid}: cached")
            continue
        q = questions[qid]
        cmp = rag.compare_pipelines(q["question"])
        cache["cases"][qid] = {
            "query_id": qid, "category": q["category"], "question": q["question"],
            "reference_answer": q["reference_answer"], "evidence": q["evidence"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            "baseline_top3": cmp["baseline_top3"], "reranked_top3": cmp["reranked_top3"],
            "rank_changes": cmp["rank_changes"],
            "answer_1": cmp["answer_1"], "answer_1_error": cmp["answer_1_error"],
            "answer_2": cmp["answer_2"], "answer_2_error": cmp["answer_2_error"],
            "prompt_1": cmp["prompt_1"], "prompt_2": cmp["prompt_2"],
            "llm_model": cmp["llm_model"], "llm_backend": rag.llm_backend,
            "generation_settings": GENERATION_SETTINGS, "checks": cmp["checks"],
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ANSWERS_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        print(f"{qid}: answer_1 {'ok' if cmp['answer_1'] else 'ERROR'} | "
              f"answer_2 {'ok' if cmp['answer_2'] else 'ERROR'}")
        time.sleep(FREE_TIER_PAUSE_S * 2)  # two requests were just made


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if mode == "retrieval":
        run_retrieval()
    elif mode == "generate":
        ids = sys.argv[2:]
        if not ids:
            res = json.loads(RETRIEVAL_JSON.read_text())
            ids = res["overall"]["rescues"] + res["overall"]["worsened"]
        run_generation(ids)
    else:
        sys.exit("usage: python -m src.stress_test [retrieval | generate [QUERY_ID ...]]")
