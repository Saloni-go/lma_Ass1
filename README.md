# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunked PDF corpus:

1. **Baseline (Pipeline 1):** Bi-Encoder (`all-MiniLM-L6-v2`) + FAISS. The top-10 are retrieved and ranks 1–3 go to the LLM (Answer 1).
2. **Re-ranking (Pipeline 2):** the *same* top-10 are re-scored by a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). The new top-3 go to the LLM (Answer 2).

Both pipelines use Gemini 2.5 Flash (`google-genai`) with one identical prompt template and identical settings.

## Quick start

```bash
cd rag-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add Gemini credentials (see "Gemini credentials")
jupyter notebook RAG_Baseline_vs_Reranking.ipynb   # run all cells from the project root
```

- **`RAG_Baseline_vs_Reranking.ipynb`** is the single consolidated notebook. It is **self-contained**: every step is implemented and commented in its cells, and nothing is imported from `src/`.
  - It covers setup, extraction, chunking, embeddings and FAISS, both pipelines with live Gemini answers, the stress tests, rank shifts, latency, scaling and figures.
  - It reads only data files: the PDF, `data/evaluation/questions.json` and the saved results in `results/`.
  - Every recomputed step is checked against the saved outputs.
  - `src/` holds the same logic as standalone scripts, which were used to produce the saved results.
  - Expensive results (stress-test answers, the 5-repeat latency runs) are loaded from `results/`.
  - Flags in the first cell re-run them.
  - Fast steps are recomputed live and checked against the saved outputs.
  - To execute it headless: `PATH=.venv/bin:$PATH jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb`
- **`results/results_summary.md`:** all verified metrics, rescues, answer changes, timings and limitations.
- **`report_materials/`:** report outline and factual evidence pack. The report itself must be written by the student.
- **`REQUIREMENTS_MAP.md`:** each assignment requirement mapped to where it is satisfied.

## Knowledge source

`data/document.pdf` is the *IIT Madras Ordinances and Regulations, B.Tech / B.Tech (Honours), applicable to the 2010–2014 batches*. It is 25 pages and born-digital (no OCR needed). It suits stress-testing because it has:

- numbered clauses with exceptions ("except", "not eligible", "notwithstanding")
- batch-specific variants of the same rule
- three near-identical credit tables (pages 10–12)
- a grade-point table (p. 21) and an attendance-code table (p. 18)

Page numbers in all metadata are **1-based PDF page indices**. The cover page is unnumbered, so the page number printed on the paper is the PDF page − 1.

## Setup

```bash
cd rag-assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your Gemini credentials (see below)
```

## Gemini credentials

Answer generation needs Gemini. Credentials go in `rag-assignment/.env`, which is git-ignored. Never commit it or paste keys anywhere else. Choose **one** option.

### Option A: Vertex AI (recommended if you have Google Cloud credits)

Free-trial or promotional credits from the Google Cloud console are spent through Vertex AI.

1. In the [Google Cloud console](https://console.cloud.google.com/), select the project that has the credits.
   - Copy its **Project ID** (not the project name or number).
   - Check under **Billing** that the project is linked to the billing account holding the credits.
2. Enable the **Vertex AI API** for that project:
   - **APIs & Services → Library → "Vertex AI API" → Enable**, or
   - `gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID`
3. Authenticate on this machine with the Google account that owns or can access the project:
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   ```
   This creates local credentials (Application Default Credentials), so no key goes into `.env`.
4. Fill in `.env`:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=true
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=global
   GEMINI_MODEL=gemini-2.5-flash
   ```
   - If the model is unavailable at `global`, try a region such as `us-central1`.
   - If a model is retired, pick a current one from the Vertex AI Model Garden.

### Option B: Gemini Developer API key

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Put it in `.env`. Leave `GOOGLE_GENAI_USE_VERTEXAI` unset or `false`.
   ```
   GEMINI_API_KEY=your-api-key
   GEMINI_MODEL=gemini-2.5-flash
   ```

### Check the setup

Run `python -m src.run_baseline`. The first line shows which backend was picked, e.g. `LLM=gemini-2.5-flash via Vertex AI (project=..., location=global)`.

If credentials are missing or wrong, the API error is printed and saved in the results JSON (`answer: null`). Answers are never fabricated.

## Stage 1: PDF processing

```bash
python -m src.process_pdf
```

This produces:

| File | Contents |
|---|---|
| `data/processed/pages.json` | one record per PDF page: cleaned text, printed page number, table/image counts, word coverage, flags |
| `data/processed/chunks.json` | `{"metadata": {...}, "chunks": [...]}`; each chunk has `chunk_id`, `source`, `page_start`, `page_end`, `pages`, `text`, `char_count`, `start_index`, `end_index`, `chunk_hash` |

The script prints page and chunk statistics and sample chunks, then validates the output. It exits non-zero if any check fails.

### What the processing does

- **Extraction (PyMuPDF).**
  - Text blocks are read in reading order.
  - Tables are detected with `find_tables()` and written as Markdown rows, so every value stays in its column. Footnote markers such as `68-80**` (Honours) are preserved.
  - Text lines inside a table area are replaced by that table.
- **Normalisation.**
  - Applies NFKC, so the Greek question mark becomes `;`.
  - Wingdings bullets become `- `.
  - Curly quotes become ASCII quotes.
  - Words hyphenated across lines are re-joined.
  - Wrapped lines are joined, while list items and clause numbers (`R.14.2`, `(a)`) keep their own lines.
  - Letter-spaced headings are collapsed (`C O N T E N T S` → `CONTENTS`).
  - The printed page number is moved from the text into metadata.
- **No silent loss.**
  - Every page is kept, including the cover.
  - Pages are flagged as `empty`, `low_text`, `image_low_text`, `has_tables`, `garbled`, or `content_loss`.
  - A word-coverage check compares the cleaned text with PyMuPDF's raw text for each page.
- **Chunking.**
  - Uses LangChain `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` characters.
  - Separators are tried in this order: paragraph, line, sentence, word.
  - The whole document is split as one text, so clauses that continue onto the next page are not cut at the page break. Each chunk's character offsets are mapped back to the page(s) it covers.
  - Paragraph breaks are only inserted at real paragraph ends. A table caption stays attached to its table, so each batch's credit table starts with its batch label.
- **IDs.**
  - `chunk_id` is the 0-based position in document order. It is deterministic for a given PDF and configuration, and later equals the FAISS row index.
  - `chunk_hash` (SHA-1 prefix of the text) shows whether a re-run changed any chunk.
- **Validation.**
  - IDs are contiguous.
  - Every chunk has non-empty text, a correct `char_count`, and length ≤ 1000.
  - Page ranges are valid.
  - The offsets reproduce each chunk's text exactly.
  - Every non-whitespace character of the document is covered by some chunk.
  - There are no duplicate chunks.
  - `chunks.json` loads back identically.

All parameters live in `src/config.py`.

## Stage 2: Baseline RAG (Pipeline 1)

```bash
python -m src.run_baseline      # needs Gemini credentials in .env (see above) for the answers
```

- **Embeddings and index.**
  - The Bi-Encoder is `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional, L2-normalised embeddings.
  - FAISS `IndexFlatIP` does exact inner-product search. On unit vectors, inner product equals cosine similarity.
- **Caching.**
  - Embeddings, the index and the FAISS-row → `chunk_id` map are cached in `data/index/`.
  - The cache is keyed by a fingerprint of the chunk hashes and the model name. If the chunks change, the index is rebuilt automatically.
- **`RAGPipeline.baseline_retrieve(query, top_k=10)`** returns rank, `chunk_id`, cosine score, pages and text for each of the Top-10.
- **`RAGPipeline.baseline_answer(query)`** retrieves the Top-10, takes ranks 1–3 unchanged, and generates Answer 1 with Gemini. It returns:
  - the answer (or the API error; answers are never fabricated)
  - all 10 candidates, which Pipeline 2 must reuse
  - the selected IDs
  - the exact prompt
  - timings: query embedding, FAISS search, retrieval, generation and total
- **Generation (`src/generation.py`).**
  - Uses the `google-genai` SDK with either Vertex AI or a Gemini API key (see **Gemini credentials**). The model comes from `GEMINI_MODEL` (default `gemini-2.5-flash`).
  - Settings are `temperature=0` and `seed=42`.
  - One `PROMPT_TEMPLATE` is shared by both pipelines. It requires context-only answers with `[Chunk N, p. X]` citations, and an explicit statement when evidence is missing.
- **Output.** Results go to `results/baseline_results.json`.

## Stage 3: Cross-Encoder re-ranking (Pipeline 2) and comparison

```bash
python -m src.run_comparison
```

- **`RAGPipeline.advanced_answer(query, retrieval=None)`**
  - Takes the baseline Top-10 (`retrieve()` / `baseline_retrieve()`) and scores all 10 `(query, chunk_text)` pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - The scores are raw relevance logits, not probabilities, and are used only for ordering.
  - Sorts the candidates by score, highest first, and sends the top 3 to the same Gemini model with the same prompt.
  - Returns the original candidates and ranks, the scores, the re-ranked positions with rank changes, the new Top-3 IDs, Answer 2, and timings.
  - Runs no second search.
- **`RAGPipeline.compare_pipelines(query)`**
  - Runs the vector search **once** and passes the identical candidate list to `baseline_answer` and `advanced_answer`.
  - Reports both Top-3 lists, which chunks were promoted or dropped, and all 10 rank changes.
- **`verify_comparison`** asserts that the two pipelines share:
  - the source PDF and chunks
  - the Top-10 IDs, order and scores
  - the candidate set used for re-ranking
  - 3 context chunks each
  - the LLM model
  - a prompt that is identical outside the context block

  So the choice of the 3 context chunks is the only thing that differs.
- **Output.** Results go to `results/comparison_results.json`.
- **Notes.**
  - Gemini 2.5 Flash is not fully deterministic, even with `temperature=0` and a fixed seed: re-running an identical prompt can change the wording. Judge answer differences against the context chunks, not wording alone.
  - The free-tier key allows 5 requests per minute. On a 429 error, the client waits as long as the API asks.

## Stage 4: Stress-test question set

```bash
python -m src.build_questions
```

- **The set.**
  - 33 questions in `data/evaluation/questions.json` and `questions.csv`.
  - Categories: keyword trap (6), exception (6), negation/constraint (6), synonym mismatch (6), document-specific (5), control (4).
- **Each question records:**
  - ID, category and question
  - a reference answer
  - supporting PDF pages and relevant chunk IDs
  - the evidence quotes
  - known distractor chunks
  - why the question may be hard
- **Verification.**
  - Pages and chunk IDs are not typed by hand. They are derived from exact evidence quotes.
  - Each quote must appear in the cleaned page text, in the raw PyMuPDF text of that page (word-sequence match), and inside at least one chunk. Otherwise the build fails.
- **NG-06 was added later.** The first 32 questions produced no negation rescue, so 24 negation-style questions were screened by retrieval only; 3 met the rank criterion, 2 were rejected on checking, and NG-06 was kept. All 24 are listed in `results/stress_test/negation_screen.json`.
- **These are candidate stress tests only.** Whether the baseline actually fails a question, or re-ranking actually rescues it, is measured in the next stage.

## Stage 5: Stress test of both pipelines

```bash
python -m src.stress_test retrieval     # all questions: ranks, scores, Recall@3, MRR@10, evidence coverage
python -m src.stress_test generate      # single Gemini answer pair per rescued/worsened case (cached)
python -m src.stress_repeats generate   # 5 answer pairs per reviewed case, Vertex AI only (resumable)
python -m src.stress_repeats promote    # run 1 of each case becomes generation_results.json
python -m src.stress_report             # merge with the manual answer review -> report JSON + CSV
```

- Output goes to `results/stress_test/`:
  - `retrieval_results.json` / `.csv`
  - `generation_results.json` (primary answers = run 1 of the repeats)
  - `generation_repeats.json`, `generation_repeats_summary.csv` (all 5 runs per case)
  - `stress_test_report.json`
  - `generation_cases.csv`
  - `negation_screen.json` (the 24-question negation screen)
  - `archived_developer_api_generation_results.json` (earlier single-run answers on the API key; not used)
- Metric definitions are in the docstring of `src/stress_test.py`.
- A **rescue** is relevant evidence that was missing from the baseline Top-3 but was in the Top-10, and that the cross-encoder promoted into the Top-3.
- Answer verdicts in `src/stress_report.py` are a manual review and are labelled as such in the output.
- All reviewed answers come from Vertex AI, the same backend as the latency benchmark. Each case was generated 5 times per pipeline; the verdict was the same in all 5 runs for every case, and an automatic key-fact check (`KEY_FACTS` in `src/stress_repeats.py`) agrees with every verdict.

## Stage 6: Latency benchmark and scaling

```bash
python -m src.benchmark_latency retrieval    # all 33 questions x 5 repeats, component timings (no LLM)
python -m src.benchmark_latency scaling      # Cross-Encoder vs #pairs, FAISS vs index size
python -m src.benchmark_latency generation   # 3 questions x 5 paired Gemini runs (resumable, bounded retries)
python -m src.benchmark_latency report       # table, summary, charts from the raw files
```

- **Output** goes to `results/latency/`:
  - raw timings
  - `raw_generation_attempts.csv` (every API attempt, including failures)
  - `latency_table.md` / `.csv`
  - `latency_summary.json`
  - `latency_analysis.md`, the write-up
- **Charts** go to `results/figures/latency_*.png`.
- **What it measures.** Model loading, indexing and warm-up are excluded. Both pipelines share one retrieval per repetition, so the measured difference is exactly cross-encoder scoring plus sorting.

## Stage 7: Results and figures

```bash
python -m src.make_results
```

- **Inputs:** only saved outputs from Tasks 4–6.
- **Validation (19 checks):**
  - duplicate and missing values
  - chunk IDs and relevance labels
  - corpus fingerprint
  - Recall@3, MRR@10 and evidence coverage, recomputed from the stored candidates
  - latency means, recomputed from the raw CSVs

  If any check fails, nothing is written.
- **Output:**
  - `results/results_summary.md`
  - `results/tables/` (CSV + Markdown, plus `answers_side_by_side.md`)
  - five figures in `results/figures/`
- **Score scales:** cosine similarities and Cross-Encoder logits are plotted on separate axes and never compared directly.

## Project structure

```
rag-assignment/
├── RAG_Baseline_vs_Reranking.ipynb   # consolidated, executed notebook
├── REQUIREMENTS_MAP.md               # assignment requirement -> evidence
├── README.md, requirements.txt, .env.example, .gitignore
├── data/
│   ├── document.pdf                  # knowledge source
│   ├── processed/                    # pages.json, chunks.json (Stage 1)
│   ├── evaluation/                   # questions.json / .csv (Stage 4, verified)
│   └── index/                        # cached embeddings + FAISS index (git-ignored, rebuilt automatically)
├── results/
│   ├── results_summary.md            # Stage 7 summary
│   ├── tables/                       # recall, outcomes, rank shifts, rescues, latency, answers side by side
│   ├── figures/                      # all charts
│   ├── stress_test/                  # Stage 5 raw retrieval + generation results, manual review
│   ├── latency/                      # Stage 6 raw timings, attempts log, summary, analysis
│   ├── baseline_results.json         # Stage 2 run
│   └── comparison_results.json       # Stage 3 run
├── report_materials/
│   ├── report_outline.md             # structure + questions for the student
│   └── evidence_pack.md              # generated facts, tables, verbatim answers
└── src/
    ├── config.py                     # paths and parameters (single source of truth)
    ├── pdf_processing.py             # extraction, cleaning, chunking, validation
    ├── process_pdf.py                # Stage 1 runner
    ├── embeddings.py                 # model loading, seeding, embedding
    ├── retrieval.py                  # Pipeline 1: FAISS index (cached) + top-k search
    ├── reranking.py                  # Pipeline 2: Cross-Encoder re-ranking
    ├── generation.py                 # Gemini client (API key or Vertex AI), shared prompt
    ├── pipeline.py                   # RAGPipeline: baseline/advanced answers, compare_pipelines
    ├── run_baseline.py, run_comparison.py      # Stage 2 / 3 runners
    ├── build_questions.py            # Stage 4: verified stress-test questions
    ├── stress_test.py, stress_report.py        # Stage 5: metrics, generation, manual review
    ├── benchmark_latency.py          # Stage 6: latency + scaling + charts
    ├── make_results.py               # Stage 7: validation, tables, figures, summary
    ├── make_report_materials.py      # evidence pack
    └── evaluation.py                 # legacy prototype helpers (unused)
```

## Reproducing everything from scratch

```bash
python -m src.process_pdf              # Stage 1
python -m src.run_baseline             # Stage 2 (Gemini)
python -m src.run_comparison           # Stage 3 (Gemini)
python -m src.build_questions          # Stage 4
python -m src.stress_test retrieval    # Stage 5
python -m src.stress_test generate && python -m src.stress_report
python -m src.benchmark_latency retrieval && python -m src.benchmark_latency scaling \
  && python -m src.benchmark_latency generation && python -m src.benchmark_latency report   # Stage 6
python -m src.make_results             # Stage 7
python -m src.make_report_materials
```

Gemini answers are not fully deterministic, even at `temperature=0`, so regenerated answer texts can differ in wording.

## Sources

- **Document:** *Ordinances and Regulations, B.Tech / B.Tech (Honours), applicable from 2010 to 2014 batch*, Indian Institute of Technology Madras (`data/document.pdf`).
- **Models:**
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
  - Google Gemini 2.5 Flash, via the `google-genai` SDK
- **Libraries:** PyMuPDF, LangChain text splitters, sentence-transformers, FAISS, pandas, matplotlib (versions in `requirements.txt`).
- **Assignment:** "RAG Assignment: Baseline vs. Re-Ranking" (course handout).

## Publishing to GitHub

```bash
git init -b main                       # already done locally
git add .
git status                             # check that .env and data/index/ are NOT listed
git commit -m "RAG baseline vs cross-encoder re-ranking"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
