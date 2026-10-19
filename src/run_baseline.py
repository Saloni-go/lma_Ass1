"""Stage 2 runner: Pipeline 1 on document-specific test queries.

Usage (from the project root):
    python -m src.run_baseline
"""
import json
import textwrap
from datetime import datetime, timezone

from . import config
from .pdf_processing import page_label
from .pipeline import RAGPipeline

# Each query targets a specific clause; `expected` notes where the evidence is
# (used for manual inspection, not by the pipeline).
TEST_QUERIES = [
    {"query": "What is the minimum attendance required in a course to be allowed to write the end-semester examination?",
     "expected": "R.14 (pp. 18-19): 85%; below that a W grade is awarded"},
    {"query": "What CGPA must B.Tech (Honours) students maintain to be awarded the Honours degree?",
     "expected": "R.30.1 (p. 25): average CGPA of 8.5, otherwise only B.Tech"},
    {"query": "Are dual degree students allowed to change their branch to the 4-year B.Tech programme?",
     "expected": "R.5.8 (p. 15): not eligible; may change discipline within the department"},
    {"query": "How many grade points does an E grade carry?",
     "expected": "R.21.1 grade table (p. 21): E = 4"},
    {"query": "What are the conditions for taking extra courses beyond the curriculum?",
     "expected": "R.6.4 (p. 15): no backlog, CGPA >= 7.0, max 15 credits, Dean's approval"},
]


def show(result):
    print("\n" + "=" * 100)
    print(f"QUERY: {result['query']}")
    print(f"expected evidence: {result['expected']}")
    print(f"\nTop-{len(result['candidates'])} baseline candidates (cosine similarity):")
    for c in result["candidates"]:
        mark = ">> LLM" if c["chunk_id"] in result["selected_chunk_ids"] else ""
        snippet = c["text"][:85].replace("\n", " ")
        print(f"  #{c['baseline_rank']:<2} chunk {c['chunk_id']:>2}  {page_label(c):<9} "
              f"cos={c['cosine_score']:.4f}  {mark:<6} {snippet}")
    print(f"\nTop-3 sent to Gemini: {result['selected_chunk_ids']}")
    t = result["timing"]
    gen = f"{t['generation_ms']:.0f} ms" if t["generation_ms"] else "n/a"
    print(f"timing: embed {t['query_embed_ms']:.1f} ms | search {t['faiss_search_ms']:.3f} ms | "
          f"retrieval {t['retrieval_ms']:.1f} ms | generation {gen}")
    if result["answer"]:
        print("\nANSWER 1:\n" + textwrap.indent(result["answer"], "  "))
    else:
        print(f"\nANSWER 1: <not generated> {result['error']}")


def main():
    rag = RAGPipeline()
    print(f"setup: {', '.join(f'{k}={v:.2f}' for k, v in rag.setup_timing.items())}")
    rag.baseline_retrieve("warm-up query")  # first call includes one-off kernel set-up

    results = []
    for q in TEST_QUERIES:
        r = rag.baseline_answer(q["query"])
        r["expected"] = q["expected"]
        results.append(r)
        show(r)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "pipeline": "baseline (Pipeline 1)",
            "bi_encoder": config.BI_ENCODER_NAME,
            "top_k_retrieve": config.TOP_K_RETRIEVE,
            "top_k_context": config.TOP_K_CONTEXT,
            "llm_model": rag.gemini_model,
            "llm_backend": rag.llm_backend,
            "device": rag.device,
            "n_chunks": len(rag.chunks),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "results": results,
    }
    config.BASELINE_RESULTS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_ok = sum(r["answer"] is not None for r in results)
    print(f"\nSaved {config.BASELINE_RESULTS_JSON.relative_to(config.PROJECT_ROOT)} "
          f"({len(results)} queries, {n_ok} answers generated)")


if __name__ == "__main__":
    main()
