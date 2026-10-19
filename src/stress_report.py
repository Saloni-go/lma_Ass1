"""Stage 5 report: merge measured retrieval results, generated answers and a
manual answer review into one JSON + CSV.

The retrieval metrics and the answers are measured outputs (stress_test.py).
MANUAL_REVIEW below is a human-style judgement of each generated answer against
the verified reference answer and the chunks that were actually in the prompt;
it was written after reading the answers and is labelled as such in the output.

Usage (from the project root):
    python -m src.stress_report
"""
import csv
import json

from .stress_repeats import CASES as REPEAT_CASES, N_RUNS, REPEATS_JSON, score
from .stress_test import ANSWERS_JSON, OUT_DIR, RETRIEVAL_JSON

REPORT_JSON = OUT_DIR / "stress_test_report.json"
CASES_CSV = OUT_DIR / "generation_cases.csv"

# verdict scale: correct | partially_correct | insufficient_context (model declined;
# no fabrication) | incorrect
# The detailed review is of run 1 (the primary answers in generation_results.json, Vertex AI).
MANUAL_REVIEW = {
    "KT-02": {
        "retrieval_change": "strict rescue: relevant chunk 21 (R.1.12) moved from baseline #4 to #1; the baseline top-3 [78, 84, 33] contained no relevant chunk, chunk 84 (R.30.1, the 'maintain CGPA 8.5 for award' distractor) was #2",
        "answer_1_verdict": "insufficient_context",
        "answer_1_notes": "Answered the distractor (average CGPA 8.5 to be AWARDED Honours, R.30.1) and stated that the context does not say when a student can opt, nor give a CGPA for opting. No fabrication, but the question is not answered.",
        "answer_2_verdict": "correct",
        "answer_2_notes": "8.5 and above at the end of the 3rd semester, opting during the 4th semester, first-attempt clearance and U/W ineligibility, all from chunk 21. Run 1 does not mention O.5 ('at the end of the fourth semester', chunk 12), which was also in the context; runs 2-3 do.",
        "missing_clauses": {"answer_1": ["CGPA 8.5 at end of 3rd semester", "opt during 4th semester", "cleared all courses in first attempt"], "answer_2": []},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": []},
        "answer_change": "improved",
    },
    "EX-05": {
        "retrieval_change": "partial rescue: chunk 61 stayed #1, chunk 62 (notional-attendance formula and the >= 85% condition) moved from #5 to #2; evidence coverage of the top-3 rose from 3/5 to 5/5 quotes",
        "answer_1_verdict": "partially_correct",
        "answer_1_notes": "Correct 'yes' with the medical-certificate and >40% conditions, but omits the second, decisive condition that notional attendance must reach 85%; as written it implies >40% attendance alone suffices.",
        "answer_2_verdict": "correct",
        "answer_2_notes": "Gives both conditions (>40% actual attendance, then notional attendance >= 85%, with the formula) and the casual-absenteeism / minor-illness exclusion.",
        "missing_clauses": {"answer_1": ["notional attendance must be >= 85% (formula excluding illness days)"], "answer_2": []},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": []},
        "answer_change": "improved",
    },
    "DS-02": {
        "retrieval_change": "strict rescue: chunk 31 (2013-onwards table, contains the HPF row) moved from #7 to #1; the baseline top-3 [33, 30, 32] held only the text paragraph and the two table tails without the HPF row",
        "answer_1_verdict": "insufficient_context",
        "answer_1_notes": "Correctly reports that the given excerpts do not mention HPF; quotes the 7-13 Pass/Fail total and the PPF (0-6) / PIL (1) rows accurately. No fabrication.",
        "answer_2_verdict": "correct",
        "answer_2_notes": "4 non-CGPA HPF credits from the 2013-onwards table (chunk 31). Chunk 29 (2011-2012 table, HPF = 2) was also in the context and was not confused with it.",
        "missing_clauses": {"answer_1": ["HPF = 4 credits"], "answer_2": []},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": []},
        "answer_change": "improved",
    },
    "NG-06": {
        "retrieval_change": "strict rescue: chunk 76 (R.25.1 tail, 'A course successfully completed cannot be repeated' + the 2014-15 exception) moved from #4 to #1; the baseline top-3 [25, 75, 46] held the 'No withdrawal ... will be allowed' registration clause (distractor), the U/W repetition rules and the n-4 registration rule",
        "answer_1_verdict": "insufficient_context",
        "answer_1_notes": "Lists only the U/W repetition rules from chunk 75 and concludes that 'there are no courses explicitly stated as non-repeatable'. True for its context and no fabrication, but a reader would take it to mean there is no restriction, which is the opposite of the rule.",
        "answer_2_verdict": "correct",
        "answer_2_notes": "Successfully completed courses cannot be repeated, plus the 2014-15 exception for students with all courses taken and CGPA below 5.",
        "missing_clauses": {"answer_1": ["a successfully completed course cannot be repeated", "2014-15 exception for CGPA below 5"], "answer_2": []},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": ["prohibition cited as [Chunk 76, p. 22]; the clause is on PDF page 23 (chunk 76 spans pp. 22-23)"]},
        "answer_change": "improved",
    },
    "SY-03": {
        "retrieval_change": "worsened: chunk 74 (the 'within three weeks' clause) fell from #2 to #4, out of the top-3; chunk 75 (re-examination procedure) went from #1 to #3; chunks 48 (dropping courses) and 72 (grade definitions) were promoted to #1 and #2",
        "answer_1_verdict": "correct",
        "answer_1_notes": "Three weeks from the commencement of the next semester after results (R.24.1), plus access to answer papers from chunk 75.",
        "answer_2_verdict": "insufficient_context",
        "answer_2_notes": "States that no time limit is given in the context and describes the answer-paper access and re-examination procedure from chunk 75; true for its context, but the answer to the question is lost. No fabrication.",
        "missing_clauses": {"answer_1": [], "answer_2": ["within three weeks from the commencement of the following semester"]},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": []},
        "answer_change": "worsened",
    },
    "DS-05": {
        "retrieval_change": "worsened: chunk 58 (attendance-code table) fell from #3 to #5, out of the top-3; replaced by chunk 59 (R.14.2 shortfall rule)",
        "answer_1_verdict": "correct",
        "answer_1_notes": "'G' for 90% attendance, from the table in chunk 58.",
        "answer_2_verdict": "insufficient_context",
        "answer_2_notes": "Reports only the <85% case (W grade with code P) and says codes for students at or above 85% are not described; true for its context. No fabrication.",
        "missing_clauses": {"answer_1": [], "answer_2": ["85-94% -> G (Good)"]},
        "unsupported_claims": {"answer_1": [], "answer_2": []},
        "citation_issues": {"answer_1": [], "answer_2": []},
        "answer_change": "worsened",
    },
}

# Verdicts for all 5 runs per case (generation_repeats.json), assigned after reading
# every answer; the automatic key-fact check (stress_repeats.KEY_FACTS) is reported
# next to them as a cross-check.
REPEAT_REVIEW = {
    "KT-02": {"answer_1": ["insufficient_context"] * 5, "answer_2": ["correct"] * 5,
              "notes": "Answer 1 always answers the award rule (R.30.1) and says the opt-in rule is missing. Answer 2 always gives 8.5 / end of 3rd semester / during 4th semester; runs 2-3 also quote O.5 ('end of the fourth semester'), runs 1, 4 and 5 do not."},
    "EX-05": {"answer_1": ["partially_correct"] * 5, "answer_2": ["correct"] * 5,
              "notes": "Answer 1 never mentions the notional-attendance condition; Answer 2 always does (runs 1 and 5 also give the formulas)."},
    "DS-02": {"answer_1": ["insufficient_context"] * 5, "answer_2": ["correct"] * 5,
              "notes": "Answer 1 always says HPF is not in the context; Answer 2 is always '4 HPF credits [Chunk 31]'."},
    "NG-06": {"answer_1": ["insufficient_context"] * 5, "answer_2": ["correct"] * 5,
              "notes": "Answer 1 lists the U/W repetition rules every time; runs 1 and 3 add 'no courses explicitly stated as non-repeatable', run 5 says the evidence is insufficient. Answer 2 always gives the prohibition and the 2014-15 exception (run 4 cites pp. 22-23 correctly; the other runs cite p. 22 only)."},
    "SY-03": {"answer_1": ["correct"] * 5, "answer_2": ["insufficient_context"] * 5,
              "notes": "Answer 1 always gives 'within three weeks'; Answer 2 always says no time limit is given."},
    "DS-05": {"answer_1": ["correct"] * 5, "answer_2": ["insufficient_context"] * 5,
              "notes": "Answer 1 always gives 'G'; Answer 2 always says the code for 90% is not specified."},
}

LIMITATIONS = [
    "33 questions, one PDF, one retrieval run per pipeline: the counts are indicative, not statistically significant. NG-06 was added after a retrieval-only screen of 24 negation phrasings (results/stress_test/negation_screen.json); 1 of the 3 screen hits survived checking, so the negation rescue was searched for, not found by chance.",
    "Answers were generated 5 times per pipeline for the six reviewed cases (60 Gemini calls, all on Vertex AI, the same backend as the latency benchmark). The verdict was the same in all 5 runs for every case; only the wording changed. Five runs show the pattern is stable, but cannot rule out rarer outcomes.",
    "Earlier single-run answers from the Gemini Developer API are kept in archived_developer_api_generation_results.json for reference only; they showed the same verdict pattern and are not used in the results.",
    "Answer verdicts are a single reviewer's judgement against the verified reference answers, not an independent annotation. The automatic key-fact check (regexes in src/stress_repeats.py) agreed with every verdict, but it only tests whether key facts are mentioned.",
    "Relevance labels are the chunks containing the evidence quotes; a chunk that answers the question with different wording would not count as relevant.",
    "C-04 (control) failed for both pipelines: the 12-credit rule is the last sentence of chunk 52, which is mostly about contact courses, and chunk 52 is not in the top-10. Re-ranking cannot recover chunks outside the candidate pool (a chunking issue, not a ranking issue).",
    "DS-01 (free-elective credits, 2011-2012): the relevant table tail (chunk 30) stays at #8-#9; the table half without its batch caption is hard for both models. Recorded as 'improved (rank only)', which is not a meaningful gain.",
]

FURTHER_EVALUATION = [
    "Grow the question set (for example 2-3 paraphrases per question) so that improved / worsened counts per category become meaningful.",
    "Have a second person review the answers independently and report agreement.",
    "Measure the effect of the candidate-pool size (top-20 or top-30 before re-ranking) on misses like C-04 and DS-01, together with the extra Cross-Encoder latency.",
    "Repeat table headers / batch captions in every table chunk, or chunk tables row-wise, and re-measure DS-01, DS-02 and DS-05.",
]

def main():
    retrieval = json.loads(RETRIEVAL_JSON.read_text())
    answers = json.loads(ANSWERS_JSON.read_text())["cases"]
    rows = {r["query_id"]: r for r in retrieval["questions"]}

    cases = []
    for qid, review in MANUAL_REVIEW.items():
        a, r = answers[qid], rows[qid]
        assert a["baseline_top3"] == r["baseline"]["top3"] and a["reranked_top3"] == r["reranked"]["top3"], \
            f"{qid}: generation run saw different top-3 than the retrieval run"
        cases.append({
            "query_id": qid, "category": a["category"], "question": a["question"],
            "reference_answer": a["reference_answer"], "evidence": a["evidence"],
            "relevant_chunk_ids": a["relevant_chunk_ids"],
            "relevant_rank_shifts": r["relevant_rank_shifts"],
            "baseline_top3": a["baseline_top3"], "reranked_top3": a["reranked_top3"],
            "metrics": {"baseline": r["baseline"], "reranked": r["reranked"]},
            "retrieval_outcome": r["outcome"], "rescue": r["rescue"], "strict_rescue": r["strict_rescue"],
            "answer_1": a["answer_1"], "answer_1_error": a["answer_1_error"],
            "answer_2": a["answer_2"], "answer_2_error": a["answer_2_error"],
            "llm_model": a["llm_model"], "llm_backend": a["llm_backend"],
            "controls_passed": all(v for v in a["checks"].values() if isinstance(v, bool)),
            "manual_review": review,
        })

    o = retrieval["overall"]
    report = {
        "metadata": {**retrieval["metadata"],
                     "manual_review_note": "answer verdicts are a manual assessment against the verified reference answers and the exact prompt context"},
        "retrieval_overall": o,
        "retrieval_by_category": retrieval["by_category"],
        "genuine_rescues_verified": [c["query_id"] for c in cases if c["rescue"]],
        "answer_level_summary": {
            "improved": [c["query_id"] for c in cases if c["manual_review"]["answer_change"] == "improved"],
            "worsened": [c["query_id"] for c in cases if c["manual_review"]["answer_change"] == "worsened"],
            "unchanged": [c["query_id"] for c in cases if c["manual_review"]["answer_change"] == "unchanged"],
            "fabrications_found": 0,
        },
        "generation_cases": cases,
        "limitations": LIMITATIONS,
        "further_evaluation": FURTHER_EVALUATION,
    }
    # repeated runs: verdicts (manual, REPEAT_REVIEW) next to the automatic key-fact check
    reps = json.loads(REPEATS_JSON.read_text())
    repeat_rows = []
    for qid in REPEAT_CASES:
        runs = [reps["runs"][f"{qid}#{i}"] for i in range(1, N_RUNS + 1)]
        assert all(r["baseline_top3"] == rows[qid]["baseline"]["top3"] and
                   r["reranked_top3"] == rows[qid]["reranked"]["top3"] for r in runs), f"{qid}: top-3 drift"
        rv = REPEAT_REVIEW[qid]
        row = {"query_id": qid, "n_runs": len(runs), "llm_backend": sorted({r["llm_backend"] for r in runs}),
               "controls_passed_all_runs": all(all(v for v in r["checks"].values() if isinstance(v, bool)) for r in runs),
               "notes": rv["notes"]}
        for a in ("answer_1", "answer_2"):
            facts = [score(qid, r[a]) for r in runs]
            row[f"{a}_verdicts"] = {v: rv[a].count(v) for v in sorted(set(rv[a]))}
            row[f"{a}_all_key_facts"] = sum(f["all_facts"] for f in facts)
            # cross-check: a run judged 'correct' must contain all key facts, and vice versa
            row[f"{a}_fact_check_agrees"] = all((v == "correct") == f["all_facts"] for v, f in zip(rv[a], facts))
        repeat_rows.append(row)
    report["repeat_runs"] = {"description": f"{N_RUNS} generations per pipeline per case on one backend; "
                             "verdicts are manual, the key-fact check is automatic (src/stress_repeats.py)",
                             "key_facts": reps["metadata"]["key_facts"], "cases": repeat_rows}
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with open(OUT_DIR / "generation_repeats_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "n_runs", "answer_1_verdicts", "answer_2_verdicts", "answer_1_all_key_facts",
                    "answer_2_all_key_facts", "fact_check_agrees", "backend", "notes"])
        for r in repeat_rows:
            w.writerow([r["query_id"], r["n_runs"], r["answer_1_verdicts"], r["answer_2_verdicts"],
                        f"{r['answer_1_all_key_facts']}/{r['n_runs']}", f"{r['answer_2_all_key_facts']}/{r['n_runs']}",
                        r["answer_1_fact_check_agrees"] and r["answer_2_fact_check_agrees"],
                        "; ".join(r["llm_backend"]), r["notes"]])

    with open(CASES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "category", "question", "relevant_rank_shifts", "baseline_top3",
                    "reranked_top3", "base_evidence_cov@3", "rr_evidence_cov@3", "retrieval_outcome",
                    "strict_rescue", "answer_1_verdict", "answer_2_verdict", "answer_change",
                    "answer_1_missing", "answer_2_missing", "citation_issues", "reference_answer",
                    "answer_1", "answer_2"])
        for c in cases:
            m = c["manual_review"]
            w.writerow([c["query_id"], c["category"], c["question"],
                        "; ".join(f"{s['chunk_id']}:{s['baseline_rank']}->{s['reranked_rank']}"
                                  for s in c["relevant_rank_shifts"]),
                        c["baseline_top3"], c["reranked_top3"],
                        c["metrics"]["baseline"]["evidence_cov@3"], c["metrics"]["reranked"]["evidence_cov@3"],
                        c["retrieval_outcome"], c["strict_rescue"], m["answer_1_verdict"],
                        m["answer_2_verdict"], m["answer_change"],
                        " | ".join(m["missing_clauses"]["answer_1"]), " | ".join(m["missing_clauses"]["answer_2"]),
                        " | ".join(m["citation_issues"]["answer_1"] + m["citation_issues"]["answer_2"]),
                        c["reference_answer"], c["answer_1"], c["answer_2"]])

    print(f"genuine rescues (verified): {report['genuine_rescues_verified']}")
    print(f"answer level: {report['answer_level_summary']}")
    for c in cases:
        m = c["manual_review"]
        print(f"  {c['query_id']:<6} retrieval {c['retrieval_outcome']:<9} | A1 {m['answer_1_verdict']:<21} "
              f"A2 {m['answer_2_verdict']:<21} -> answer {m['answer_change']}")
    print(f"\nSaved {REPORT_JSON.name}, {CASES_CSV.name} (+ retrieval_results.json/.csv, "
          f"generation_results.json) in {OUT_DIR}")


if __name__ == "__main__":
    main()
