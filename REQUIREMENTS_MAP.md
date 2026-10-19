# Assignment requirements → where they are satisfied

Source: "RAG Assignment: Baseline vs. Re-Ranking" handout. Notebook = `RAG_Baseline_vs_Reranking.ipynb`, and "§" refers to its section numbers.

Status key: ✅ done and verified · ⚠️ partially done or needs the student · ❌ not done

## Submission

| Requirement | Status | Where |
|---|---|---|
| Code in a public or shared GitHub repository | ⚠️ | Repository prepared locally (`git init`, `.gitignore`, README). **Not pushed yet**; publishing commands are in the README. |
| Preferably a single, well-commented Jupyter notebook | ✅ | `RAG_Baseline_vs_Reranking.ipynb`: self-contained (22 functions defined inline, no `src/` imports), 21 code cells, executed end to end without errors |
| Report not AI-generated, concise, in the student's own words | ⚠️ | Not written, deliberately. `report_materials/report_outline.md` (structure + questions) and `report_materials/evidence_pack.md` (facts) support the student. |

## 1. Setup and environment

| Requirement | Status | Where |
|---|---|---|
| LLM: Gemini / other API / local model | ✅ | Gemini 2.5 Flash via `google-genai` (API key or Vertex AI), `src/generation.py`; notebook §4 |
| Embeddings | ✅ | `sentence-transformers/all-MiniLM-L6-v2`, `src/embeddings.py`; notebook §3 |
| A single, complex PDF (policy, regulation or rulebook) | ✅ | `data/document.pdf`: IIT Madras B.Tech Ordinances & Regulations, 25 pages, with tables, exceptions and batch variants; notebook §1 |

## 2. Two-stage retrieval over the same chunked corpus

| Requirement | Status | Where |
|---|---|---|
| Same chunked corpus for both pipelines | ✅ | `data/processed/chunks.json` (85 chunks, 1000/200); corpus fingerprint and `verify_comparison` checks; notebook §2 and §5 |
| P1(a) Index chunks in a vector store with a Bi-Encoder | ✅ | FAISS `IndexFlatIP` on normalised embeddings, cached in `data/index/`; `src/retrieval.py`; notebook §3 |
| P1(b) Top-10 by cosine / inner product | ✅ | Notebook §4 (top-10 table), `RAGPipeline.baseline_retrieve` |
| P1(c) Top-3 directly → LLM → Answer 1 | ✅ | Notebook §4 (live Answer 1), `RAGPipeline.baseline_answer` |
| P2(a) Take the same top-10 candidates | ✅ | Notebook §5 (no second search; same-set assertion), `compare_pipelines` |
| P2(b) Score all 10 (query, chunk) pairs with a Cross-Encoder | ✅ | `cross-encoder/ms-marco-MiniLM-L-6-v2`, `src/reranking.py`; notebook §5 |
| P2(c) Re-sort by Cross-Encoder score, take top-3 | ✅ | Notebook §5 (re-ranked table with cosine and CE logit in separate columns) |
| P2(d) Same prompt structure → Answer 2 | ✅ | Notebook §5 (live Answer 2 + prompt-identity assertion); `src/generation.py` `PROMPT_TEMPLATE` |

## 3. Failure-mode analysis

| Requirement | Status | Where |
|---|---|---|
| ≥ 2 failure queries where the baseline degrades and re-ranking restores relevance | ✅ | 4 verified rescues: **KT-02** (chunk 21 #4→#1), **EX-05** (chunk 62 #5→#2), **DS-02** (chunk 31 #7→#1), **NG-06** (chunk 76 #4→#1); notebook §6–7; `results/tables/rescues_and_regressions.md` |
| Keyword trap / distractor | ✅ | KT-02: distractor R.30.1 (chunk 84) outranked the answer R.1.12 (chunk 21); strict rescue |
| Exceptions & corner cases | ✅ | EX-05: the condonation exception's second condition (chunk 62) was promoted, and Answer 1 → Answer 2 went from omitting the clause to complete |
| Negation & constraints | ✅ | NG-06 "Which courses is a student not allowed to repeat?": the prohibition chunk 76 moved #4→#1 (strict rescue); the baseline top-3 held the "No withdrawal ... will be allowed" distractor (chunk 25). Answer 1 said no such restriction is mentioned; Answer 2 gave the rule and its 2014-15 exception. Found by a retrieval-only screen of 24 negation phrasings (`results/stress_test/negation_screen.json`, 1 of 3 hits kept); the original NG-01 to NG-05 only moved inside the top-3. |
| Other document-specific failure (vital chunk at ranks 4–10 promoted to the top-3) | ✅ | DS-02: the 2013 credit table among three near-identical tables, chunk 31 #7→#1 |
| (Also reported) cases where re-ranking hurt or could not help | ✅ | Regressions SY-03 and DS-05; C-04 and DS-01 not recoverable (`results/results_summary.md`) |

## 4. Analysis and report (evidence ready, text by the student)

| Report area | Status | Evidence |
|---|---|---|
| 1. Architecture comparison | ⚠️ text | `evidence_pack.md` §1 (models, measured costs, KT-02 dual-score table); notebook §3, §5 |
| 2a. Rank changes for the failure queries | ✅ data / ⚠️ text | `evidence_pack.md` §2; `results/figures/rank_shifts_relevant_chunks.png`, `top10_before_after.png`; `results/tables/rank_shifts_relevant_chunks.md` |
| 2b. Answer 1 vs Answer 2 (hallucination / omitted clause / misstated rule) | ✅ data / ⚠️ text | Verbatim answers and manual review: `results/tables/answers_side_by_side.md`, `evidence_pack.md` §2; notebook §7 |
| 3. Latency profiling table: baseline vs re-ranking | ✅ data / ⚠️ text | `results/latency/latency_table.md`, `latency_analysis.md`, `results/figures/latency_breakdown.png`, `latency_comparison.png`; notebook §8 |
| 4. Scaling question | ✅ data / ⚠️ text | `evidence_pack.md` §4; `results/figures/latency_scaling.png`; notebook §8 |

## Checks run (latest)

| Check | Result |
|---|---|
| Notebook executed headless (`nbconvert --execute`) | 21/21 code cells executed, 0 errors; no `src/` imports; live Answer 1/2 generated on Vertex AI; pages, chunks and all 33 top-3 lists identical to the saved outputs |
| Fresh extraction + chunking identical to saved `chunks.json` | yes (all 85 chunk hashes equal), notebook §2 |
| Live stress-test retrieval identical to saved results | yes, all 33 top-3 lists, notebook §6 |
| `src/make_results.py` validation | 19/19 checks passed (labels, IDs, duplicates, missing values, recomputed metrics, latency means) |
| Stress-test question verification | 33/33 questions' evidence quotes verified in the cleaned text, the raw PDF text and the chunks |
| Secrets | `.env` git-ignored; no API-key-like strings in tracked files |

## Known gaps (also listed in `report_materials/report_outline.md`)

1. Report text: by the student.
2. GitHub push: needs your authorisation and a remote.
3. The negation rescue (NG-06) was found by a targeted screen after the first 32 questions, not written in advance; say so.
4. Answers repeated 5× per pipeline per case on one backend (Vertex AI, same as the latency runs): verdicts identical across runs. Five runs cannot rule out rarer outcomes.
5. One manual reviewer (cross-checked by an automatic key-fact regex).
6. 33 questions only.
