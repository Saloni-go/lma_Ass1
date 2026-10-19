"""Stage 7: validated results tables, figures and results_summary.md.

Reads ONLY saved outputs of earlier stages:
    data/processed/chunks.json, data/evaluation/questions.json      (Tasks 1, 4)
    results/stress_test/*.json                                      (Task 5)
    results/latency/*                                               (Task 6)
validates them (missing values, duplicates, chunk IDs, relevance labels, and
recomputes the retrieval metrics from the stored candidate lists), then writes
    results/tables/*.csv|md, results/figures/*.png, results/results_summary.md

Usage (from the project root):
    python -m src.make_results
"""
import json
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .retrieval import corpus_fingerprint  # noqa: E402

R = config.RESULTS_DIR
TABLES, FIGS = R / "tables", config.FIGURES_DIR
SUMMARY_MD = R / "results_summary.md"
CATEGORY_ORDER = ["keyword_trap", "exception", "negation_constraint", "synonym_mismatch",
                  "document_specific", "control"]
CASE_ORDER = ["KT-02", "EX-05", "DS-02", "NG-06", "SY-03", "DS-05"]

# palette (dataviz reference instance; both pairs validated with validate_palette.js)
BASE, RR = "#2a78d6", "#eb6834"                      # pipelines: baseline / re-ranked
IMPROVED, UNCHANGED, WORSENED = "#6250d6", "#c9c8c3", "#e34948"
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#d3d2cd"


def load():
    j = lambda p: json.loads(p.read_text())
    return {
        "chunks": j(config.CHUNKS_JSON)["chunks"],
        "questions": j(config.DATA_DIR / "evaluation" / "questions.json")["questions"],
        "retrieval": j(R / "stress_test" / "retrieval_results.json"),
        "generation": j(R / "stress_test" / "generation_results.json")["cases"],
        "report": j(R / "stress_test" / "stress_test_report.json"),
        "lat_summary": j(R / "latency" / "latency_summary.json"),
        "lat_ret": pd.read_csv(R / "latency" / "raw_retrieval_timings.csv"),
        "lat_gen": pd.read_csv(R / "latency" / "raw_generation_timings.csv"),
        "lat_att": pd.read_csv(R / "latency" / "raw_generation_attempts.csv"),
    }


# ----------------------------------------------------------------- validation
def validate(d):
    checks = []
    add = lambda name, ok, detail="": checks.append({"check": name, "passed": bool(ok), "detail": detail})
    chunk_ids = {c["chunk_id"] for c in d["chunks"]}
    q = pd.DataFrame(d["questions"])
    rows = {r["query_id"]: r for r in d["retrieval"]["questions"]}

    # questions / relevance labels
    add("33 questions, unique query IDs", len(q) == 33 and q.query_id.is_unique, f"n={len(q)}")
    add("no duplicate question texts", q.question.str.strip().str.lower().is_unique)
    req = ["query_id", "category", "question", "reference_answer", "supporting_pages",
           "relevant_chunk_ids", "evidence"]
    missing = [f"{r.query_id}:{c}" for _, r in q.iterrows() for c in req
               if r[c] is None or (hasattr(r[c], "__len__") and len(r[c]) == 0)]
    add("no missing values in question fields", not missing, ", ".join(missing))
    bad = [(r.query_id, c) for _, r in q.iterrows()
           for c in list(r.relevant_chunk_ids) + list(r.distractor_chunk_ids) if c not in chunk_ids]
    add("all relevant/distractor chunk IDs exist in chunks.json", not bad, str(bad))
    overlap = [r.query_id for _, r in q.iterrows() if set(r.relevant_chunk_ids) & set(r.distractor_chunk_ids)]
    add("relevant and distractor labels disjoint", not overlap, str(overlap))
    add("all questions verified against the PDF (Task 4 flag)", q.verified.all())
    add("categories are the six defined ones", set(q.category) == set(CATEGORY_ORDER))

    # retrieval results vs labels; recompute metrics from the stored candidates
    add("one retrieval row per question", set(rows) == set(q.query_id) and
        len(d["retrieval"]["questions"]) == len(q))
    fp = corpus_fingerprint(d["chunks"], config.BI_ENCODER_NAME)
    add("retrieval run used the current chunk corpus (fingerprint)",
        d["retrieval"]["metadata"]["corpus_fingerprint"] == fp)
    mism = []
    for _, qq in q.iterrows():
        r = rows[qq.query_id]
        rel = set(qq.relevant_chunk_ids)
        c = pd.DataFrame(r["candidates"])
        if sorted(r["relevant_chunk_ids"]) != sorted(rel):
            mism.append(f"{qq.query_id}: labels differ")
        if len(c) != 10 or not c.chunk_id.is_unique or not set(c.chunk_id) <= chunk_ids:
            mism.append(f"{qq.query_id}: candidate list invalid")
        if sorted(c.baseline_rank) != list(range(1, 11)) or sorted(c.reranked_rank) != list(range(1, 11)):
            mism.append(f"{qq.query_id}: ranks not a permutation of 1..10")
        b_ids = list(c.sort_values("baseline_rank").chunk_id)
        r_ids = list(c.sort_values("reranked_rank").chunk_id)
        # re-ranked order must be the CE-score order; baseline order the cosine order
        if list(c.sort_values("ce_score", ascending=False, kind="stable").chunk_id) != r_ids:
            mism.append(f"{qq.query_id}: re-ranked order != CE score order")
        if list(c.sort_values("cosine_score", ascending=False, kind="stable").chunk_id) != b_ids:
            mism.append(f"{qq.query_id}: baseline order != cosine order")
        for name, ids in (("baseline", b_ids), ("reranked", r_ids)):
            m = r[name]
            rec3 = len(set(ids[:3]) & rel) / len(rel)
            mrr = next((1 / i for i, x in enumerate(ids, 1) if x in rel), 0.0)
            cov = np.mean([bool(set(ids[:3]) & set(e["chunk_ids"])) for e in qq.evidence])
            if m["top3"] != ids[:3] or abs(m["recall@3"] - rec3) > 1e-9 or abs(m["mrr@10"] - mrr) > 1e-9 \
                    or abs(m["evidence_cov@3"] - cov) > 1e-9:
                mism.append(f"{qq.query_id}: stored {name} metrics differ from recomputation")
    add("retrieval candidates, ranks and Recall@3/MRR@10/evidence coverage recomputed and match",
        not mism, "; ".join(mism))

    # generation cases
    gen_bad = []
    for qid, g in d["generation"].items():
        r = rows[qid]
        if g["baseline_top3"] != r["baseline"]["top3"] or g["reranked_top3"] != r["reranked"]["top3"]:
            gen_bad.append(f"{qid}: top-3 differs from retrieval run")
        if not g["answer_1"] or not g["answer_2"]:
            gen_bad.append(f"{qid}: missing answer")
        if not all(v for v in g["checks"].values() if isinstance(v, bool)):
            gen_bad.append(f"{qid}: fairness checks failed")
    add("generated answers: present, same top-3 as retrieval run, fairness checks passed",
        not gen_bad, "; ".join(gen_bad))
    reviewed = {c["query_id"] for c in d["report"]["generation_cases"]}
    add("every generated case has a manual review", reviewed == set(d["generation"]))
    rr_ = d["report"]["repeat_runs"]["cases"]
    backends = {g["llm_backend"] for g in d["generation"].values()} | {b for r in rr_ for b in r["llm_backend"]}
    add("repeat runs: 5 per case, fairness checks passed, verdicts agree with the key-fact check",
        {r["query_id"] for r in rr_} == set(d["generation"]) and all(r["n_runs"] == 5 and r["controls_passed_all_runs"]
        and r["answer_1_fact_check_agrees"] and r["answer_2_fact_check_agrees"] for r in rr_))
    add("all reviewed answers and the Gemini latency runs used one backend",
        len(backends) == 1 and backends == set(d["lat_gen"].backend), "; ".join(sorted(backends | set(d["lat_gen"].backend))))

    # latency
    lr, lg = d["lat_ret"], d["lat_gen"]
    tcols = ["query_embed_ms", "faiss_search_ms", "ce_score_ms", "sort_select_ms",
             "p1_retrieval_ms", "p2_retrieval_ms"]
    add(f"retrieval timings: {len(lr)} rows = {len(q)} questions x 5 repeats, no missing values",
        len(lr) == 5 * len(q) and set(lr.query_id) == set(q.query_id) and lr.groupby("query_id").size().eq(5).all() and lr[tcols].notna().all().all()
        and not lr.duplicated(["query_id", "repeat"]).any())
    gcols = ["gen1_ms", "gen2_ms", "p1_end_to_end_ms", "p2_end_to_end_ms"]
    add("generation timings: 15 completed pairs, unique (query, repeat), no missing values",
        len(lg) == 15 and not lg.duplicated(["query_id", "repeat"]).any() and lg[gcols].notna().all().all())
    add("generation settings identical in every completed pair", lg.settings.nunique() == 1)
    tab = {t["key"]: t for t in d["lat_summary"]["table"]}
    lat_bad = [k for k in tcols if abs(lr[k].mean() - tab[k]["mean"]) > 1e-6] + \
              [k for k in gcols if abs(lg[k].mean() - tab[k]["mean"]) > 1e-6]
    add("latency summary means recomputed from raw files and match", not lat_bad, str(lat_bad))
    ok_att = (d["lat_att"].status == "ok").sum()
    add("attempt log consistent (successful calls >= 2 x completed pairs)", ok_att >= 2 * len(lg),
        f"{ok_att} ok attempts for {len(lg)} pairs")
    return checks


# ----------------------------------------------------------------- tables
def md_table(df, floatfmt=3):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(df, name, floatfmt=3):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(md_table(df, floatfmt) + "\n")


def build_tables(d):
    rows = pd.DataFrame(d["retrieval"]["questions"])
    per_q = pd.DataFrame({
        "query_id": rows.query_id, "category": rows.category,
        "base_recall@3": rows.baseline.map(lambda m: m["recall@3"]),
        "rr_recall@3": rows.reranked.map(lambda m: m["recall@3"]),
        "base_hit@3": rows.baseline.map(lambda m: m["hit@3"]),
        "rr_hit@3": rows.reranked.map(lambda m: m["hit@3"]),
        "base_mrr@10": rows.baseline.map(lambda m: m["mrr@10"]),
        "rr_mrr@10": rows.reranked.map(lambda m: m["mrr@10"]),
        "base_evcov@3": rows.baseline.map(lambda m: m["evidence_cov@3"]),
        "rr_evcov@3": rows.reranked.map(lambda m: m["evidence_cov@3"]),
        "recall@10": rows["recall@10"], "outcome": rows.outcome, "outcome_basis": rows.outcome_basis,
        "rescue": rows.rescue, "strict_rescue": rows.strict_rescue,
    })
    metric_cols = [c for c in per_q.columns if c.startswith(("base_", "rr_")) or c == "recall@10"]

    g = per_q.groupby("category")[metric_cols].mean().reindex(CATEGORY_ORDER)
    g.insert(0, "n", per_q.groupby("category").size().reindex(CATEGORY_ORDER))
    overall = per_q[metric_cols].mean().to_frame().T
    overall.insert(0, "n", len(per_q))
    overall.index = ["ALL"]
    recall = pd.concat([g, overall]).reset_index().rename(columns={"index": "category"})
    save(recall, "recall_summary")

    oc = (per_q.assign(n=1).pivot_table(index="category", columns="outcome", values="n", aggfunc="sum",
                                        fill_value=0)
          .reindex(index=CATEGORY_ORDER, columns=["improved", "unchanged", "worsened"], fill_value=0))
    oc["improved_top3"] = per_q[(per_q.outcome == "improved") & (per_q.outcome_basis == "evidence_cov@3")] \
        .groupby("category").size().reindex(CATEGORY_ORDER, fill_value=0)
    oc["worsened_top3"] = per_q[(per_q.outcome == "worsened") & (per_q.outcome_basis == "evidence_cov@3")] \
        .groupby("category").size().reindex(CATEGORY_ORDER, fill_value=0)
    oc.loc["ALL"] = oc.sum()
    outcomes = oc.reset_index()
    save(outcomes, "outcome_counts", 0)

    shift_rows = []
    for _, r in rows.iterrows():
        cands = {c["chunk_id"]: c for c in r.candidates}
        for cid in r.relevant_chunk_ids:
            c = cands.get(cid)
            shift_rows.append({
                "query_id": r.query_id, "category": r.category, "relevant_chunk": cid,
                "baseline_rank": c["baseline_rank"] if c else "not in top-10",
                "reranked_rank": c["reranked_rank"] if c else "not in top-10",
                "rank_change": c["rank_change"] if c else "",
                "cosine_similarity": c["cosine_score"] if c else "",
                "cross_encoder_logit": c["ce_score"] if c else "",
                "entered_top3": bool(c and c["baseline_rank"] > 3 and c["reranked_rank"] <= 3),
                "left_top3": bool(c and c["baseline_rank"] <= 3 and c["reranked_rank"] > 3),
            })
    shifts = pd.DataFrame(shift_rows)
    save(shifts, "rank_shifts_relevant_chunks", 4)

    rescues = per_q[per_q.rescue | ((per_q.outcome == "worsened") & (per_q.outcome_basis == "evidence_cov@3"))]
    resc = rescues.merge(rows[["query_id", "relevant_rank_shifts"]], on="query_id")
    resc["relevant_rank_shifts"] = resc.relevant_rank_shifts.map(
        lambda s: "; ".join(f"chunk {x['chunk_id']}: #{x['baseline_rank']} -> #{x['reranked_rank']}" for x in s))
    resc["baseline_top3"] = resc.query_id.map(lambda q: d["generation"][q]["baseline_top3"])
    resc["reranked_top3"] = resc.query_id.map(lambda q: d["generation"][q]["reranked_top3"])
    review = {c["query_id"]: c["manual_review"] for c in d["report"]["generation_cases"]}
    resc["answer_1_verdict"] = resc.query_id.map(lambda q: review[q]["answer_1_verdict"])
    resc["answer_2_verdict"] = resc.query_id.map(lambda q: review[q]["answer_2_verdict"])
    resc["answer_change"] = resc.query_id.map(lambda q: review[q]["answer_change"])
    resc = resc[["query_id", "category", "relevant_rank_shifts", "baseline_top3", "reranked_top3",
                 "base_evcov@3", "rr_evcov@3", "rescue", "strict_rescue", "answer_1_verdict",
                 "answer_2_verdict", "answer_change"]]
    resc = resc.set_index("query_id").loc[[c for c in CASE_ORDER if c in set(resc.query_id)]].reset_index()
    save(resc, "rescues_and_regressions", 2)

    lat = pd.DataFrame(d["lat_summary"]["table"])[["scope", "component", "n", "mean", "median", "std", "p95"]]
    lat.columns = ["scope", "component", "n", "mean_ms", "median_ms", "std_ms", "p95_ms"]
    save(lat, "latency_comparison", 3)

    # side-by-side answers
    md = ["# Generated answers: baseline (Answer 1) vs re-ranked (Answer 2)", "",
          "Actual Gemini 2.5 Flash outputs from Stage 5 (`results/stress_test/generation_results.json`), "
          "same prompt template and settings for both answers; only the 3 context chunks differ. "
          "Verdicts are a manual review against the verified reference answer and the exact context "
          "(`src/stress_report.py`).", ""]
    for qid in CASE_ORDER:
        gcase, rv = d["generation"][qid], review[qid]
        md += [f"## {qid} ({gcase['category']}): {gcase['question']}", "",
               f"**Reference answer:** {gcase['reference_answer']}", "",
               f"**Retrieval:** {rv['retrieval_change']}", "",
               f"**Context:** baseline top-3 {gcase['baseline_top3']}, re-ranked top-3 {gcase['reranked_top3']}, "
               f"relevant {gcase['relevant_chunk_ids']}", "",
               "| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |", "|---|---|---|",
               "| Answer | " + gcase["answer_1"].replace("\n", "<br>").replace("|", "\\|") + " | "
               + gcase["answer_2"].replace("\n", "<br>").replace("|", "\\|") + " |",
               f"| Verdict | {rv['answer_1_verdict']} | {rv['answer_2_verdict']} |",
               f"| Missing clauses | {'; '.join(rv['missing_clauses']['answer_1']) or '-'} | "
               f"{'; '.join(rv['missing_clauses']['answer_2']) or '-'} |",
               f"| Unsupported claims | {'; '.join(rv['unsupported_claims']['answer_1']) or 'none found'} | "
               f"{'; '.join(rv['unsupported_claims']['answer_2']) or 'none found'} |",
               f"| Citation issues | {'; '.join(rv['citation_issues']['answer_1']) or '-'} | "
               f"{'; '.join(rv['citation_issues']['answer_2']) or '-'} |",
               "", f"**Answer change:** {rv['answer_change']}", ""]
    (TABLES / "answers_side_by_side.md").write_text("\n".join(md))
    return {"per_q": per_q, "recall": recall, "outcomes": outcomes, "shifts": shifts,
            "rescues": resc, "latency": lat}


# ----------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left"})


def fig_recall(t):
    rec = t["recall"].set_index("category")
    cats = CATEGORY_ORDER + ["ALL"]
    labels = [c.replace("_", "\n") + f"\n(n={int(rec.loc[c, 'n'])})" for c in cats]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.36
    for off, col, color, name in ((-w / 2, "base_recall@3", BASE, "Baseline (Bi-Encoder top-3)"),
                                  (w / 2, "rr_recall@3", RR, "Re-ranked (Cross-Encoder top-3)")):
        vals = rec.loc[cats, col].values
        ax.bar(x + off, vals, w, color=color, edgecolor=SURF, linewidth=2, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axvline(len(cats) - 1.5, color=GRID, linewidth=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall@3 (mean over questions)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Recall@3 by question category: baseline vs re-ranked ({int(rec.loc['ALL', 'n'])} verified questions)", pad=26)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "recall_at3_by_category.png", dpi=160)
    plt.close(fig)


def fig_rank_shifts(t):
    s = t["shifts"][t["shifts"].baseline_rank != "not in top-10"].copy()
    s["b"], s["r"] = s.baseline_rank.astype(int), s.reranked_rank.astype(int)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.axhspan(0.5, 3.5, color="#f0efec", zorder=0)
    ax.text(1.05, 3.1, "shaded: top-3 = context sent to the LLM", ha="left", va="center", fontsize=7.5, color=INK2)
    seen, labels = {}, []
    for _, r in s.sort_values(["entered_top3", "left_top3"]).iterrows():
        color = IMPROVED if r.entered_top3 else WORSENED if r.left_top3 else UNCHANGED
        lw, z = (2.2, 3) if color != UNCHANGED else (1.0, 2)
        seen[(r.b, r.r)] = seen.get((r.b, r.r), 0) + 1
        off = 0.0                            # identical paths overlap exactly (counted below)
        ax.plot([0, 1], [r.b + off, r.r + off], color=color, linewidth=lw, zorder=z, alpha=0.9 if z == 3 else 0.6,
                marker="o", markersize=5, markeredgecolor=SURF, markeredgewidth=1)
        if color != UNCHANGED:
            labels.append((r.r + off, f"{r.query_id}  chunk {r.relevant_chunk}: #{r.b} → #{r.r}"))
    same = seen.get((1, 1), 0)
    if same:
        ax.text(-0.04, 1, f"{same} chunks\n#1 → #1", ha="right", va="center", fontsize=7.5, color=INK2)
    last = -9
    for y, text in sorted(labels):            # nudge labels apart when ranks coincide
        y = max(y, last + 0.38)
        ax.text(1.05, y, text, va="center", fontsize=8, color=INK)
        last = y
    ax.set_xlim(-0.3, 2.0)
    ax.set_ylim(10.6, 0.4)
    ax.set_yticks(range(1, 11))
    ax.set_xticks([0, 1], ["Baseline rank\n(cosine)", "Re-ranked rank\n(Cross-Encoder)"])
    ax.set_ylabel("rank of relevant chunk within the shared top-10")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title(f"Rank shifts of the {len(s)} relevant chunks found in the top-10", pad=30)
    handles = [plt.Line2D([], [], color=IMPROVED, linewidth=2.2, label="entered top-3"),
               plt.Line2D([], [], color=WORSENED, linewidth=2.2, label="left top-3"),
               plt.Line2D([], [], color=UNCHANGED, linewidth=1.2, label="top-3 membership unchanged")]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    out = t["shifts"][t["shifts"].baseline_rank == "not in top-10"]
    note = ", ".join(f"{r.query_id} chunk {r.relevant_chunk}" for r in out.itertuples())
    ax.text(0, -0.11, f"Not in the top-10, so not re-rankable ({len(out)}): {note}.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGS / "rank_shifts_relevant_chunks.png", dpi=160)
    plt.close(fig)


def fig_top10(d):
    rows = {r["query_id"]: r for r in d["retrieval"]["questions"]}
    fig, axes = plt.subplots(len(CASE_ORDER), 2, figsize=(10, 2.15 * len(CASE_ORDER)))
    for i, qid in enumerate(CASE_ORDER):
        c = pd.DataFrame(rows[qid]["candidates"])
        tag = "rescue" if rows[qid]["rescue"] else "regression"
        for j, (order, score, color, xlabel) in enumerate((
                ("baseline_rank", "cosine_score", BASE, "cosine similarity (Bi-Encoder, range -1..1)"),
                ("reranked_rank", "ce_score", RR, "Cross-Encoder relevance logit (unbounded, own scale)"))):
            ax = axes[i, j]
            cc = c.sort_values(order)
            y = np.arange(10)
            ax.barh(y, cc[score], color=[color if rel else MUTED for rel in cc.relevant],
                    height=0.7, edgecolor=SURF, linewidth=1)
            ax.set_yticks(y, [f"#{r} chunk {cid}" for r, cid in zip(cc[order], cc.chunk_id)], fontsize=7)
            ax.invert_yaxis()
            ax.axhline(2.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
            ax.axvline(0, color=INK2, linewidth=0.6)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=7)
            if j == 0:
                ax.set_title(f"{qid} ({tag}) baseline order", fontsize=9)
                ax.set_xlim(0, 1)
            else:
                ax.set_title(f"{qid} re-ranked order", fontsize=9)
            if i == len(CASE_ORDER) - 1:
                ax.set_xlabel(xlabel, fontsize=8)
    fig.suptitle("Shared top-10 before and after re-ranking (coloured = verified relevant chunk; "
                 "dashed line = top-3 cut-off).\nLeft and right use different, non-comparable score scales.",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGS / "top10_before_after.png", dpi=160)
    plt.close(fig)


def fig_latency(d):
    lr, lg = d["lat_ret"], d["lat_gen"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    rng = np.random.default_rng(0)
    for k, (col, color, name) in enumerate((("p1_retrieval_ms", BASE, "Pipeline 1"),
                                            ("p2_retrieval_ms", RR, "Pipeline 2"))):
        v = lr[col].values
        a1.scatter(v, k + rng.uniform(-0.18, 0.18, len(v)), s=10, color=color, alpha=0.55, linewidths=0)
        a1.plot([np.median(v)] * 2, [k - 0.3, k + 0.3], color=INK, linewidth=2)
        a1.text(np.median(v), k - 0.36, f"median {np.median(v):.1f} ms", ha="center", fontsize=8, color=INK)
    a1.set_xscale("log")
    a1.set_yticks([0, 1], ["Pipeline 1\n(embed + FAISS)", "Pipeline 2\n(+ Cross-Encoder, 10 pairs)"])
    a1.set_ylim(1.6, -0.7)
    a1.set_xlabel("retrieval-side latency per query (ms, log scale)")
    a1.set_title(f"Retrieval side: {len(lr)} runs ({lr.query_id.nunique()} questions x 5)")
    a1.grid(axis="x", color=GRID, linewidth=0.6, which="both")
    a1.set_axisbelow(True)

    for _, r in lg.iterrows():
        a2.plot([0, 1], [r.p1_end_to_end_ms / 1000, r.p2_end_to_end_ms / 1000], color=MUTED, linewidth=1, zorder=1)
    a2.scatter(np.zeros(len(lg)), lg.p1_end_to_end_ms / 1000, color=BASE, s=22, zorder=2, label="Pipeline 1")
    a2.scatter(np.ones(len(lg)), lg.p2_end_to_end_ms / 1000, color=RR, s=22, zorder=2, label="Pipeline 2")
    for k, col in enumerate(("p1_end_to_end_ms", "p2_end_to_end_ms")):
        m = lg[col].median() / 1000
        a2.plot([k - 0.15, k + 0.15], [m, m], color=INK, linewidth=2, zorder=3)
        a2.text(k + (0.2 if k else -0.2), m, f"median {m:.2f} s", va="center", ha="left" if k else "right",
                fontsize=8, color=INK)
    a2.set_xticks([0, 1], ["Pipeline 1", "Pipeline 2"])
    a2.set_xlim(-0.8, 1.8)
    a2.set_ylabel("end-to-end latency incl. Gemini (s)")
    a2.set_title(f"End-to-end: {len(lg)} paired runs (lines join a pair)")
    a2.grid(axis="y", color=GRID, linewidth=0.6)
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "latency_comparison.png", dpi=160)
    plt.close(fig)


def fig_outcomes(t):
    oc = t["outcomes"].set_index("category").loc[CATEGORY_ORDER]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(oc))
    left = np.zeros(len(oc))
    for col, color, name in (("improved", IMPROVED, "improved"), ("unchanged", UNCHANGED, "unchanged"),
                             ("worsened", WORSENED, "worsened")):
        v = oc[col].values.astype(float)
        ax.barh(y, v, left=left, color=color, edgecolor=SURF, linewidth=2, height=0.6, label=name)
        for yi, l, vv in zip(y, left, v):
            if vv:
                ax.text(l + vv / 2, yi, f"{int(vv)}", ha="center", va="center", fontsize=8,
                        color=SURF if color != UNCHANGED else INK)
        left += v
    for yi, (cat, r) in zip(y, oc.iterrows()):
        ax.text(left[yi] + 0.1, yi, f"top-3 evidence: +{int(r.improved_top3)} / -{int(r.worsened_top3)}",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [c.replace("_", " ") for c in CATEGORY_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() + 3.2)
    ax.set_xlabel("number of questions")
    tot = t["outcomes"].set_index("category").loc["ALL"]
    ax.set_title(f"Per-question outcome of re-ranking: {int(tot.improved)} improved, {int(tot.unchanged)} "
                 f"unchanged, {int(tot.worsened)} worsened", pad=26)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGS / "outcome_counts_by_category.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- summary
def write_summary(d, t, checks):
    ret_meta = d["retrieval"]["metadata"]
    o = d["retrieval"]["overall"]
    ls = d["lat_summary"]
    tab = {x["key"]: x for x in ls["table"]}
    rec = t["recall"].set_index("category")
    review = {c["query_id"]: c["manual_review"] for c in d["report"]["generation_cases"]}
    chunk_meta = json.loads(config.CHUNKS_JSON.read_text())["metadata"]
    gen_meta = next(iter(d["generation"].values()))
    n_pass = sum(c["passed"] for c in checks)

    def pct(x):
        return f"{x:.3f}"

    all_ = rec.loc["ALL"]   # unrounded means; the JSON stores 4-dp values, which would round twice

    rows_by_q = {r["query_id"]: r for r in d["retrieval"]["questions"]}
    lines = [
        "# Results summary: Baseline RAG vs Cross-Encoder re-ranking", "",
        "Every number, table and figure below is generated by `src/make_results.py` from saved experiment "
        "outputs (Tasks 4-6), after the validation checks listed at the end. The answer verdicts "
        "(correct / partially correct / insufficient context) and the short interpretive sentences "
        "come from the manual review of the actual Gemini answers recorded in `src/stress_report.py`.", "",
        "## Configuration", "",
        f"- **Source:** `{chunk_meta['source_pdf']}`, IIT Madras B.Tech Ordinances & Regulations, "
        f"{chunk_meta['n_pages']} pages",
        f"- **Chunking:** RecursiveCharacterTextSplitter, {chunk_meta['chunk_size']} chars, overlap "
        f"{chunk_meta['chunk_overlap']} → {chunk_meta['n_chunks']} chunks",
        f"- **Bi-Encoder:** `{ret_meta['bi_encoder']}`; FAISS `IndexFlatIP` on L2-normalised vectors "
        "(= cosine similarity), top-10",
        f"- **Cross-Encoder:** `{ret_meta['cross_encoder']}`. It re-scores exactly the same top-10; top-3 go to the LLM",
        f"- **LLM:** `{gen_meta['llm_model']}`. Settings {json.dumps(gen_meta['generation_settings'])}, "
        "and one prompt template shared by both pipelines",
        f"- **Evaluation set:** {o['n_questions']} questions, verified against the PDF, in 6 categories; "
        "relevance labels = chunks that contain the evidence quotes",
        f"- **Hardware:** {ls['hardware']['cpu']}, {ls['hardware']['ram_gb']} GB RAM, GPU: {ls['hardware']['gpu']}",
        "",
        f"## Retrieval metrics (all {o['n_questions']} questions)", "",
        "| | Recall@3 | Hit@3 | MRR@10 | Evidence coverage@3 |", "|---|---|---|---|---|",
        f"| Baseline | {pct(all_['base_recall@3'])} | {pct(all_['base_hit@3'])} | "
        f"{pct(all_['base_mrr@10'])} | {pct(all_['base_evcov@3'])} |",
        f"| Re-ranked | {pct(all_['rr_recall@3'])} | {pct(all_['rr_hit@3'])} | "
        f"{pct(all_['rr_mrr@10'])} | {pct(all_['rr_evcov@3'])} |", "",
        f"- **Shared candidate pool:** Recall@10 = {pct(all_['recall@10'])}, identical for both pipelines. "
        f"No relevant chunk reached the top-10 for: {', '.join(o['questions_with_no_relevant_in_top10']) or 'none'}.",
        f"- **Chunks moved:** {o['relevant_chunks_promoted_4to10_into_top3']} relevant chunks were promoted "
        f"from ranks 4-10 into the top-3, and {o['relevant_chunks_demoted_from_top3']} were demoted out of it.",
        f"- **Per-question outcome:** {len(o['improved'])} improved, {len(o['unchanged'])} unchanged, "
        f"{len(o['worsened'])} worsened.",
        f"  - Only {len(o['improved_top3_coverage'])} improvements ({', '.join(o['improved_top3_coverage'])}) "
        f"and {len(o['worsened_top3_coverage'])} regressions ({', '.join(o['worsened_top3_coverage'])}) "
        "changed the evidence that reaches the LLM.",
        "  - The rest are rank moves only, e.g. a chunk moving within the top-3, or from #9 to #8.",
        "- **Metric definitions:**",
        "  - Recall@3 = |relevant ∩ top-3| / |relevant|.",
        "  - MRR@10 = 1/rank of the first relevant chunk (0 if none).",
        "  - Evidence coverage@3 = share of evidence quotes present in the top-3.",
        "", "### By category", "",
        md_table(t["recall"][["category", "n", "base_recall@3", "rr_recall@3", "base_mrr@10", "rr_mrr@10",
                              "base_evcov@3", "rr_evcov@3"]]), "",
        "![Recall@3 by category](figures/recall_at3_by_category.png)", "",
        "![Outcome counts](figures/outcome_counts_by_category.png)", "",
        "## Confirmed rescues and regressions", "",
        "A **rescue** means: relevant evidence missing from the baseline top-3 was in the baseline top-10 "
        "and was promoted into the top-3 by the Cross-Encoder. Cosine similarities and Cross-Encoder "
        "logits are on different scales, so only ranks are compared across pipelines.", "",
    ]
    for qid in CASE_ORDER:
        r, rv = rows_by_q[qid], review[qid]
        shifts = ", ".join(f"chunk {s['chunk_id']} #{s['baseline_rank']}→#{s['reranked_rank']}"
                           for s in r["relevant_rank_shifts"])
        kind = ("strict rescue" if r["strict_rescue"] else "rescue") if r["rescue"] else "regression"
        cands = {c["chunk_id"]: c for c in r["candidates"]}
        scores = "; ".join(f"chunk {s['chunk_id']}: cosine {cands[s['chunk_id']]['cosine_score']:.3f}, "
                           f"CE logit {cands[s['chunk_id']]['ce_score']:.2f}" for s in r["relevant_rank_shifts"])
        lines += [f"- **{qid}** ({r['category']}, {kind}): {shifts}.",
                  f"  - Top-3: {r['baseline']['top3']} → {r['reranked']['top3']}.",
                  f"  - Evidence coverage {r['baseline']['evidence_cov@3']:.2f} → {r['reranked']['evidence_cov@3']:.2f}.",
                  f"  - Scores: {scores}.",
                  f"  - Answer 1: *{rv['answer_1_verdict']}*. Answer 2: *{rv['answer_2_verdict']}*. "
                  f"Answer change: **{rv['answer_change']}**."]
    lines += ["", md_table(t["rescues"], 2), "",
              "![Rank shifts](figures/rank_shifts_relevant_chunks.png)", "",
              "![Top-10 before/after](figures/top10_before_after.png)", "",
              "## Answer changes (manual review of actual Gemini outputs)", "",
              f"- **Answers improved: {', '.join(d['report']['answer_level_summary']['improved'])}.** "
              "In every rescue, the baseline either declined for lack of evidence (KT-02, DS-02, NG-06) "
              "or left out the decisive clause (EX-05). The re-ranked answer was correct.",
              "- **Negation case NG-06** was found by a retrieval-only screen of 24 negation phrasings "
              "(`stress_test/negation_screen.json`): 3 screen hits, 2 rejected on checking, 1 kept.",
              f"- **Answers worsened: {', '.join(d['report']['answer_level_summary']['worsened'])}.** "
              "The key chunk dropped out of the top-3, and Answer 2 declined for lack of evidence.",
              f"- **Fabrications found:** {d['report']['answer_level_summary']['fabrications_found']}. "
              "When evidence was missing, the model said so instead of guessing.",
              "- **Citation errors:** one page-citation slip in the reviewed answers, a wrong page within a "
              "two-page chunk (NG-06 Answer 2).",
              "- **Retrieval vs answer quality:** in all six cases the answer moved in the same direction "
              "as the retrieval change, i.e. with whether the deciding chunk was in the context.",
              "- Full side-by-side answers: `tables/answers_side_by_side.md`.", "",
              "### Repeated generation (5 runs per pipeline per case)", "",
              "Each case was answered 5 times by both pipelines on the same backend (Vertex AI), with the same "
              "retrieval, prompt and settings. Verdicts were assigned after reading every answer. The key-fact "
              "column is an automatic regex check for the facts in the reference answer "
              "(`src/stress_repeats.py`).", "",
              "| Case | Answer 1 verdicts | Answer 2 verdicts | A1 with all key facts | A2 with all key facts |",
              "|---|---|---|---|---|"] + [
              f"| {r['query_id']} | " + ", ".join(f"{k} ×{v}" for k, v in r["answer_1_verdicts"].items()) + " | "
              + ", ".join(f"{k} ×{v}" for k, v in r["answer_2_verdicts"].items())
              + f" | {r['answer_1_all_key_facts']}/{r['n_runs']} | {r['answer_2_all_key_facts']}/{r['n_runs']} |"
              for r in d["report"]["repeat_runs"]["cases"]] + [
              "", "- The verdict was identical in all 5 runs for every case; only the wording changed "
              "(e.g. KT-02 Answer 2 quotes the conflicting O.5 clause in runs 2-3 only).",
              "- Full runs: `stress_test/generation_repeats.json`; summary: `stress_test/generation_repeats_summary.csv`.", "",
              "## Latency (ms unless stated)", "",
              "| Component | n | mean | median | std |", "|---|---|---|---|---|"]
    for k in ("query_embed_ms", "faiss_search_ms", "ce_score_ms", "sort_select_ms", "p1_retrieval_ms",
              "p2_retrieval_ms", "rerank_overhead_ms", "gen1_ms", "gen2_ms", "p1_end_to_end_ms", "p2_end_to_end_ms"):
        x = tab[k]
        lines.append(f"| {x['component']} | {x['n']} | {x['mean']:.3f} | {x['median']:.3f} | {x['std']:.3f} |")
    sc = ls["scaling"]
    lines += ["",
              f"- **Re-ranking cost:** it adds **{ls['rerank_overhead_mean_ms']:.0f} ms** per query (mean), "
              f"so the retrieval side is {ls['retrieval_slowdown_mean_ratio']:.1f}x slower (mean) and "
              f"{ls['retrieval_slowdown_median_ratio']:.1f}x slower (median).",
              f"- **End-to-end:** P2/P1 = {ls['end_to_end_slowdown_median_ratio']:.2f}x by median and "
              f"{ls['end_to_end_slowdown_mean_ratio']:.2f}x by mean.",
              f"  - Re-ranking is {ls['rerank_share_of_p2_end_to_end']:.1%} of P2's mean end-to-end time.",
              f"  - Gemini latency varies strongly between identical requests (Answer 1: "
              f"{d['lat_gen'].gen1_ms.min() / 1000:.1f}-{d['lat_gen'].gen1_ms.max() / 1000:.1f} s, Answer 2: "
              f"{d['lat_gen'].gen2_ms.min() / 1000:.1f}-{d['lat_gen'].gen2_ms.max() / 1000:.1f} s), far more than the "
              f"{ls['rerank_overhead_mean_ms']:.0f} ms re-ranking step, so the end-to-end ratios reflect Gemini variance. "
              f"With {len(d['lat_gen'])} pairs, the end-to-end difference is not statistically resolved.",
              f"- **Cross-Encoder scaling:** 10 pairs = {sc['cross_encoder_vs_pairs_ms']['10']['median']:.0f} ms; "
              f"the whole 85-chunk corpus = {sc['cross_encoder_vs_pairs_ms']['85']['median']:.0f} ms per query.",
              f"  - Linear extrapolation: about {sc['cross_encoder_full_corpus_extrapolation_ms']['10000'] / 60000:.0f} min "
              "per query for 10k chunks.",
              f"- **FAISS scaling:** exact search over 100k vectors (synthetic) = "
              f"{sc['faiss_vs_index_size_ms']['100000 (synthetic)']['median']:.1f} ms.",
              f"- **Cost estimate:** ≈ ${ls['cost_estimate_usd']['estimated_cost_usd']:.3f} for "
              f"{ls['cost_estimate_usd']['successful_calls']} Gemini answers (assumed list prices).",
              "", "![Latency comparison](figures/latency_comparison.png)", "",
              "![Latency breakdown](figures/latency_breakdown.png)", "",
              "![Scaling](figures/latency_scaling.png)", "",
              "## Limitations", ""]
    lines += [f"- {x}" for x in d["report"]["limitations"]]
    lines += ["- Latency was measured on one laptop CPU without a GPU. Gemini latency includes network and "
              "server queueing, and Vertex AI returned transient 429 errors, which were retried and are "
              "excluded from the measurements.",
              f"- All reviewed answers (and their 5-run repeats) and the Gemini latency runs used one backend: "
              f"{d['lat_gen'].backend.iloc[0]}, model {gen_meta['llm_model']}, identical settings.",
              "- Retrieval metrics depend on the relevance labels (chunks containing the evidence quotes). "
              "Chunk overlap means some quotes occur in two chunks.", "",
              "## Validation of the inputs", "", f"{n_pass}/{len(checks)} checks passed.", ""]
    lines += [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['check']}" + (f" ({c['detail']})" if c["detail"] and not c["passed"] else "")
              for c in checks]
    lines += ["", "## Files", "",
              "- `tables/`:",
              "  - `recall_summary`",
              "  - `outcome_counts`",
              "  - `rank_shifts_relevant_chunks`",
              "  - `rescues_and_regressions`",
              "  - `latency_comparison` (.csv + .md)",
              "  - `answers_side_by_side.md`",
              "- `figures/`:",
              "  - `recall_at3_by_category.png`",
              "  - `outcome_counts_by_category.png`",
              "  - `rank_shifts_relevant_chunks.png`",
              "  - `top10_before_after.png`",
              "  - `latency_comparison.png`",
              "  - `latency_breakdown.png`",
              "  - `latency_scaling.png`",
              "- Raw sources: `stress_test/`, `latency/`, `../data/evaluation/questions.json`"]
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main():
    d = load()
    checks = validate(d)
    failed = [c for c in checks if not c["passed"]]
    print(f"=== Validation: {len(checks) - len(failed)}/{len(checks)} checks passed ===")
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']}" + (f"  -> {c['detail']}" if not c["passed"] else ""))
    if failed:
        sys.exit("validation failed; no results written")

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    t = build_tables(d)
    style()
    fig_recall(t)
    fig_outcomes(t)
    fig_rank_shifts(t)
    fig_top10(d)
    fig_latency(d)
    write_summary(d, t, checks)

    o = d["retrieval"]["overall"]
    print("\n=== Recall@3 summary ===")
    print(t["recall"][["category", "n", "base_recall@3", "rr_recall@3", "base_mrr@10", "rr_mrr@10"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Outcomes ===")
    print(t["outcomes"].to_string(index=False))
    print("\n=== Rescues and regressions (answer verdicts = manual review) ===")
    print(t["rescues"][["query_id", "category", "relevant_rank_shifts", "base_evcov@3", "rr_evcov@3",
                        "answer_1_verdict", "answer_2_verdict", "answer_change"]].to_string(index=False))
    ls = d["lat_summary"]
    print(f"\n=== Latency === re-ranking +{ls['rerank_overhead_mean_ms']:.0f} ms/query "
          f"({ls['retrieval_slowdown_mean_ratio']:.1f}x retrieval side); end-to-end P2/P1 median "
          f"{ls['end_to_end_slowdown_median_ratio']:.2f}x")
    print(f"\nwrote {SUMMARY_MD.relative_to(config.PROJECT_ROOT)}, tables in "
          f"{TABLES.relative_to(config.PROJECT_ROOT)}/, figures in {FIGS.relative_to(config.PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
