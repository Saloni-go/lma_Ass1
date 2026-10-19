"""Stage 3 runner: Pipeline 1 vs Pipeline 2 on the same retrieved Top-10.

Usage (from the project root):
    python -m src.run_comparison
"""
import json
import textwrap
from datetime import datetime, timezone

from . import config
from .generation import GENERATION_SETTINGS
from .pipeline import RAGPipeline
from .run_baseline import TEST_QUERIES


def show(cmp, expected):
    print("\n" + "=" * 100)
    print(f"QUERY: {cmp['query']}")
    print(f"expected evidence: {expected}")
    print(f"\n{'base':>4} {'rerank':>6} {'change':>6}  {'chunk':>5}  {'pages':<7} "
          f"{'cosine':>7} {'CE score':>9}  text")
    for rc in sorted(cmp["rank_changes"], key=lambda r: r["reranked_rank"]):
        text = next(c["text"] for c in cmp["candidates_top10"] if c["chunk_id"] == rc["chunk_id"])
        pages = f"{rc['pages'][0]}" if rc["pages"][0] == rc["pages"][1] else f"{rc['pages'][0]}-{rc['pages'][1]}"
        tag = " *" if rc["reranked_rank"] <= config.TOP_K_CONTEXT else "  "
        print(f"#{rc['baseline_rank']:<3} #{rc['reranked_rank']:<4}{tag}{rc['rank_change']:+5d}  "
              f"{rc['chunk_id']:>5}  {pages:<7} {rc['cosine_score']:7.4f} {rc['ce_score']:9.3f}  "
              f"{text[:60].replace(chr(10), ' ')}")
    print("(* = in re-ranked Top-3; change = baseline rank - re-ranked rank, positive = promoted)")
    print(f"\nBaseline Top-3 -> Answer 1: {cmp['baseline_top3']}")
    print(f"Re-ranked Top-3 -> Answer 2: {cmp['reranked_top3']}")
    print(f"promoted into Top-3: {cmp['promoted_into_top3'] or 'none'} | "
          f"dropped: {cmp['dropped_from_top3'] or 'none'}")
    tb, tr = cmp["timing"]["baseline"], cmp["timing"]["reranked"]
    print(f"timing: shared retrieval {tb['retrieval_ms']:.1f} ms | CE re-rank {tr['rerank_ms']:.1f} ms | "
          f"P1 retrieval-side {tb['pipeline_retrieval_ms']:.1f} ms vs P2 {tr['pipeline_retrieval_ms']:.1f} ms | "
          f"gen1 {tb['generation_ms'] or float('nan'):.0f} ms, gen2 {tr['generation_ms'] or float('nan'):.0f} ms")
    for n in (1, 2):
        ans, err = cmp[f"answer_{n}"], cmp[f"answer_{n}_error"]
        print(f"\nANSWER {n}:\n" + (textwrap.indent(ans, "  ") if ans else f"  <not generated> {err}"))
    ok = all(v for k, v in cmp["checks"].items() if isinstance(v, bool))
    print(f"\ncontrol checks: {'all passed' if ok else cmp['checks']}")


def main():
    rag = RAGPipeline()
    print("setup: " + ", ".join(f"{k}={v:.2f}s" for k, v in rag.setup_timing.items()))
    # warm-up so the first measured query does not include one-off initialisation
    warm = rag.retrieve("warm-up query")
    from .reranking import rerank
    rerank("warm-up query", warm["candidates"], rag.cross_encoder)

    comparisons = []
    for q in TEST_QUERIES:
        cmp = rag.compare_pipelines(q["query"])
        cmp["expected"] = q["expected"]
        comparisons.append(cmp)
        show(cmp, q["expected"])

    n_changed = sum(set(c["baseline_top3"]) != set(c["reranked_top3"]) for c in comparisons)
    n_order = sum(c["baseline_top3"] != c["reranked_top3"] for c in comparisons)
    print("\n" + "=" * 100)
    print(f"SUMMARY: {len(comparisons)} queries | Top-3 set changed in {n_changed} | "
          f"Top-3 order changed in {n_order}")
    for c in comparisons:
        print(f"  {c['query'][:65]:<67} base {c['baseline_top3']}  rerank {c['reranked_top3']}")

    out = {
        "metadata": {
            "source_pdf": rag.source_pdf,
            "corpus_fingerprint": rag.corpus_fingerprint,
            "n_chunks": len(rag.chunks),
            "bi_encoder": config.BI_ENCODER_NAME,
            "cross_encoder": config.CROSS_ENCODER_NAME,
            "top_k_retrieve": config.TOP_K_RETRIEVE,
            "top_k_context": config.TOP_K_CONTEXT,
            "llm_model": rag.gemini_model,
            "llm_backend": rag.llm_backend,
            "generation_settings": GENERATION_SETTINGS,
            "device": rag.device,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "comparisons": comparisons,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.COMPARISON_RESULTS_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {config.COMPARISON_RESULTS_JSON.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
