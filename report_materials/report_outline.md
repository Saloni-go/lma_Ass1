# Report outline (for the student to write in their own words)

The assignment says the report must **not** be AI-generated and must be concise and in your own words. This file is therefore **not a report**. It gives:
- the four required sections, in the assignment's order;
- where the evidence for each section is;
- questions to answer yourself.

All facts, numbers, tables and verbatim answers are in `evidence_pack.md` (generated from the saved results) and in `../results/results_summary.md`.

Before you start, re-run `python -m src.make_results` and `python -m src.make_report_materials`, so the numbers you quote match the latest outputs.

---

## 1. Architecture comparison: Bi-Encoder vs Cross-Encoder

**Assignment asks:** briefly explain the core difference: independent embeddings + cosine similarity vs joint self-attention across tokens.

**Evidence to use:**
- `evidence_pack.md` §1: model names, the 384-dimensional embeddings, measured costs (one-time corpus embedding vs per-query cross-encoder cost), and the score ranges of cosine vs CE logits.
- `evidence_pack.md` §1, KT-02 worked example: the same 10 chunks scored by both models.
- Notebook §3 (embeddings and index) and §5 (cross-encoder).

**Questions to answer in your own words:**
1. What does the Bi-Encoder compute for a chunk, and when? Why can that be done before any query arrives?
2. What does the Cross-Encoder take as input? What can it "see" that a cosine between two separate vectors cannot? Think about tokens of the query attending to tokens of the chunk.
3. Why are a cosine similarity of 0.67 and a CE logit of 4.33 not comparable numbers? What *is* comparable between the two pipelines?
4. In the KT-02 table, chunk 84 (the "maintain CGPA 8.5 for the award" distractor) has a higher cosine than chunk 21 (the true answer), but a lower CE logit. In one or two sentences, explain why independent embeddings might do this.

---

## 2. Qualitative and rank-shift analysis

**Assignment asks:**
- document the rank changes for your failure queries, e.g. "chunk X moved from rank #8 → #1";
- compare Answer 1 vs Answer 2: hallucination, an omitted clause, or a misstated rule because the critical chunk sat outside the top 3?

**Evidence to use:**
- `evidence_pack.md` §2: for each case, the question, reference answer, verbatim source quotes, the distractor passage, rank shifts, both top-3 lists, both Gemini answers verbatim, and the review notes.
- `../results/tables/rescues_and_regressions.md`, `../results/tables/answers_side_by_side.md`.
- Figures:
  - `../results/figures/rank_shifts_relevant_chunks.png`
  - `../results/figures/top10_before_after.png`
  - `../results/figures/recall_at3_by_category.png`
  - `../results/figures/outcome_counts_by_category.png`

**Verified cases (measured, not assumed):**

| Case | Category | Rank shift | Answer change |
|---|---|---|---|
| KT-02 | keyword trap | chunk 21: #4 → #1 (strict rescue) | baseline answered the distractor and said the opt-in rule was missing → correct |
| EX-05 | exception | chunk 62: #5 → #2 | baseline omitted the "notional attendance ≥ 85%" condition → complete |
| DS-02 | document-specific (near-identical tables) | chunk 31: #7 → #1 (strict rescue) | "not in context" → correct (4 credits) |
| NG-06 | negation | chunk 76: #4 → #1 (strict rescue) | "no such restriction mentioned" → correct (completed courses cannot be repeated, plus the 2014-15 exception) |
| SY-03 | synonym mismatch (regression) | chunk 74: #2 → #4 | correct → "no time limit given" |
| DS-05 | table lookup (regression) | chunk 58: #3 → #5 | correct → "not specified" |

**Questions to answer in your own words:**
1. For each rescue: what made the baseline rank the wrong chunks higher? Use the "Why it is hard" notes and the distractor quotes as a starting point, but explain it yourself.
2. Did any baseline answer *hallucinate*? Check the review notes: in this project the baseline failures were refusals or omitted clauses, not invented facts. Why might the prompt rule "say when evidence is missing" matter here?
3. EX-05: what exactly is wrong with Answer 1, and why is that more dangerous than an outright refusal?
4. The two regressions (SY-03, DS-05) both involve a chunk the Cross-Encoder scored low (CE logits about −7 for SY-03, 0.67 for DS-05). What do they suggest about the Cross-Encoder's weaknesses? Consider tables, and wording that differs from MS MARCO-style passages.
5. Why is "7 improved / 22 unchanged / 4 worsened" a weaker statement than "4 rescues / 2 regressions of top-3 evidence"? Which one should a reader care about?
6. C-04 and DS-01 failed for both pipelines. Why can re-ranking not fix C-04 (relevant chunk not in the top-10)? What would you change: chunking, a larger k?
7. KT-02 extra observation: chunk 12 (ordinance O.5, "opt for B.Tech (Honours) at the end of the fourth semester") moved from #8 to #2. It is not in the relevance labels. Is it relevant? It even contradicts R.1.12 ("during the fourth semester"). How does that affect your discussion?

---

## 3. Latency profiling

**Assignment asks:** record and tabulate the runtime of baseline retrieval vs the re-ranking pipeline.

**Evidence to use:**
- `evidence_pack.md` §3: full latency table, hardware, method, tokens, cost, failed attempts.
- `../results/latency/latency_table.md` and `../results/latency/latency_analysis.md` (method details and limitations).
- Figures:
  - `../results/figures/latency_breakdown.png`
  - `../results/figures/latency_comparison.png`

**Key measured numbers** (copy them from the table, don't retype from memory):
- Pipeline 1 retrieval side ≈ 10.0 ms median; Pipeline 2 ≈ 264 ms median.
- Re-ranking adds ≈ 265 ms (mean), so retrieval is ≈ 27× slower.
- End-to-end with Gemini: ≈ 4.51 s vs 3.50 s (median); 4.50 s vs 6.03 s (mean).

**Questions to answer in your own words:**
1. Which single component dominates Pipeline 2's retrieval time, and why does the sorting step cost almost nothing?
2. Re-ranking is ≈ 27× slower on the retrieval side, yet end-to-end Pipeline 2 is *faster* by median (0.78×) and slower by mean (1.34×). Explain why neither ratio measures re-ranking. Where does most of the user-visible time go?
3. Why are both end-to-end ratios misleading here? Look at the 26 s Answer 2 outlier (C-01, repeat 5) in `latency_comparison.png`.
4. What would change on a GPU, or with a larger k?

---

## 4. Scaling question

**Assignment asks:** why we do not simply run a Cross-Encoder across the entire document collection from the start.

**Evidence to use:**
- `evidence_pack.md` §4: measured CE cost for 1-85 pairs (85 = the whole corpus ≈ 3 s per query), the extrapolation, and FAISS cost vs index size.
- `../results/figures/latency_scaling.png`.

**Questions to answer in your own words:**
1. The Cross-Encoder has to run once per (query, chunk) pair at query time. How does that cost grow with the corpus? How does Bi-Encoder retrieval grow?
2. Using the measured numbers, what would a Cross-Encoder over 10,000 or 100,000 chunks cost per query? These are extrapolations; say so.
3. Explain the two-stage idea in your own terms: what does each stage contribute, and what does it cost?
4. What is the main risk of the two-stage design? Hint: C-04, where the relevant chunk never reaches the top-10.

---

## Gaps and unfulfilled requirements (be honest about these in the report)

1. **The report itself is not written.** It is your task, by the assignment's rules.
2. **GitHub link:** the repository is prepared locally but not pushed. You need to create the remote and push (commands in the README).
3. **How the negation case was found:** NG-01 to NG-05 gave no negation rescue (NG-02 and NG-05 only moved inside the top-3). NG-06 was then found by screening 24 negation-style questions by retrieval only (`../results/stress_test/negation_screen.json`): 3 met the rank criterion and 2 were rejected because the question misdescribed the rule. Say in the report that this case was searched for, not written in advance.
   - Question to answer: Answer 1 for NG-06 says the context mentions no course that cannot be repeated. Why is that more misleading than a plain "I don't know"?
4. **Repeated generation:** each of the six cases was answered 5 times by both pipelines (60 calls). The verdict was the same in all 5 runs every time; only the wording changed. See `../results/stress_test/generation_repeats_summary.csv` and the table in `results_summary.md`. Five runs show the pattern is stable, but cannot rule out rarer outcomes.
5. **One backend:** all reviewed answers and the latency runs used Vertex AI (gemini-2.5-flash, identical settings). Earlier single-run answers from the free API key are archived and not used.
6. **Single reviewer:** the answer verdicts are one reviewer's judgement. Read the answers yourself and agree or disagree in the report.
7. **Small set:** 33 questions on one document, so the counts are indicative, not statistically significant.
