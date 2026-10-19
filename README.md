# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunked PDF corpus:

1. **Baseline (Pipeline 1):** Bi-Encoder (`all-MiniLM-L6-v2`) + FAISS. The top-10 chunks are retrieved and ranks 1–3 go to the LLM (Answer 1).
2. **Re-ranking (Pipeline 2):** the *same* top-10 are re-scored by a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). The new top-3 go to the LLM (Answer 2).

Both pipelines use Gemini 2.5 Flash (`google-genai`) with one identical prompt template and identical settings.

All the code is in one notebook: **`main.ipynb`**. Each step is explained in a Markdown cell and commented in the code.

## Quick start

```bash
cd rag-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add Gemini credentials (see "Gemini credentials")
jupyter notebook main.ipynb   # run all cells from the project folder
```

To execute it without opening Jupyter:

```bash
PATH=.venv/bin:$PATH jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

## What the notebook contains

| § | Step |
|---|---|
| 0 | Setup and configuration (all parameters in one cell) |
| 1 | PDF extraction with PyMuPDF: tables kept as Markdown rows, text normalised, a word-coverage check so no text is silently lost |
| 2 | Chunking with LangChain `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap) → 85 chunks |
| 3 | Bi-Encoder embeddings and a FAISS `IndexFlatIP` index (inner product on normalised vectors = cosine) |
| 4 | **Pipeline 1:** top-10 → top-3 → Gemini **Answer 1** |
| 5 | **Pipeline 2:** Cross-Encoder re-ranks the same top-10 → new top-3 → Gemini **Answer 2**, with a check that only the context differs |
| 6 | The 33 stress-test questions (evidence quotes verified against the PDF), the negation screen, and Recall@3, Hit@3, MRR@10, evidence coverage@3 for both pipelines |
| 7 | Rank shifts, Gemini answers for the rescues and regressions (5 runs per case), the manual review, Answer 1 vs Answer 2 |
| 8 | Latency benchmark and scaling (why the Cross-Encoder is not run on the whole collection) |
| 9 | 19 validation checks, result tables, figures and `results/results_summary.md` |

Every file in `data/processed/`, `data/evaluation/` and `results/` is produced by code in the notebook.

**Run flags** in the first code cell:

| Flag | Default | What it does |
|---|---|---|
| `RUN_LIVE_LLM` | on | 2 Gemini calls: the live Answer 1 / Answer 2 demo in §4–5 |
| `RERUN_STRESS_GENERATION` | off | 60 Gemini calls: regenerate the 6 stress cases × 5 runs × 2 answers (§7) |
| `RERUN_LATENCY_BENCHMARK` | off | ~4 min, no API: 33 questions × 5 timed queries + the scaling runs (§8) |
| `RERUN_LATENCY_GENERATION` | off | 30 Gemini calls: 3 questions × 5 paired runs for end-to-end latency (§8) |
| `WRITE_RESULTS` | off | rewrite every derived output file (corpus, questions, metrics, report JSON, tables, figures, summary) |

With the defaults, everything cheap is recomputed live (extraction, chunking, index, questions, negation screen, all retrieval metrics, validation, tables and figures) and compared with the saved files; the raw measurements of the three slow or paid steps are loaded from `results/`. A `RERUN_*` flag measures that step again and saves the new raw data. The manual answer review (`MANUAL_REVIEW` in §7) refers to the saved answers, so it has to be re-read if the answers are regenerated; the automatic key-fact check shows where it no longer agrees.

## Knowledge source

`data/document.pdf` is the *IIT Madras Ordinances and Regulations, B.Tech / B.Tech (Honours), applicable to the 2010–2014 batches*. It is 25 pages and born-digital (no OCR needed). It suits stress-testing because it has:

- numbered clauses with exceptions ("except", "not eligible", "notwithstanding")
- batch-specific variants of the same rule
- three near-identical credit tables (pages 10–12)
- a grade-point table (p. 21) and an attendance-code table (p. 18)

Page numbers in all metadata are **1-based PDF page indices**. The cover page is unnumbered, so the page number printed on the paper is the PDF page − 1.

## Gemini credentials

Answer generation needs Gemini. Credentials go in `rag-assignment/.env`, which is git-ignored. Never commit it or paste keys anywhere else. Choose **one** option.

### Option A: Vertex AI (Google Cloud credits)

1. In the [Google Cloud console](https://console.cloud.google.com/), select the project with the credits, copy its **Project ID**, and check under **Billing** that it is linked to the billing account holding the credits.
2. Enable the **Vertex AI API**: `gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID`
3. Authenticate on this machine:
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   ```
4. Fill in `.env`:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=true
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=global
   GEMINI_MODEL=gemini-2.5-flash
   ```

### Option B: Gemini Developer API key

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Put it in `.env` and leave `GOOGLE_GENAI_USE_VERTEXAI` unset or `false`:
   ```
   GEMINI_API_KEY=your-api-key
   GEMINI_MODEL=gemini-2.5-flash
   ```

The notebook prints which backend it uses (`LLM client: ...`). Without credentials, the live answers are skipped with a message; answers are never fabricated.

Gemini 2.5 Flash is not fully deterministic, even with `temperature=0` and a fixed seed, so re-running can change the wording of the answers.

## Project structure

Every file in `data/processed/`, `data/evaluation/` and `results/` is produced by the notebook (section in brackets).

```
rag-assignment/
├── main.ipynb                           # all the code, commented and executed
├── README.md
├── requirements.txt                     # Python dependencies
├── .env.example                         # template for the Gemini credentials (.env itself is git-ignored)
├── .gitignore
├── data/
│   ├── document.pdf                     # knowledge source (input)
│   ├── processed/
│   │   ├── pages.json                   # cleaned text + diagnostics per PDF page (§1)
│   │   └── chunks.json                  # the 85 chunks with page metadata (§2)
│   ├── evaluation/
│   │   ├── questions.json               # the 33 stress-test questions, evidence verified against the PDF (§6)
│   │   └── questions.csv                # same, as a table
│   └── index/                           # cached embeddings + FAISS index (git-ignored, rebuilt automatically, §3)
├── results/
│   ├── results_summary.md               # all metrics, rescues, answer changes, timings, limitations (§9)
│   ├── stress_test/
│   │   ├── negation_screen.json         # the 24-question negation screen that found NG-06 (§6)
│   │   ├── retrieval_results.json/.csv  # ranks, scores and metrics of both pipelines for all 33 questions (§6)
│   │   ├── generation_repeats.json      # 5 Gemini answer pairs per rescue/regression case (§7)
│   │   ├── generation_results.json      # run 1 of each case, the answers reviewed in detail (§7)
│   │   ├── stress_test_report.json      # retrieval results + answers + manual review (§7)
│   │   ├── generation_cases.csv         # the reviewed cases as a table (§7)
│   │   └── generation_repeats_summary.csv  # verdicts and key-fact check over the 5 runs (§7)
│   ├── latency/
│   │   ├── raw_retrieval_timings.csv    # 33 questions x 5 timed queries (§8)
│   │   ├── raw_generation_timings.csv   # 15 paired Gemini runs (§8)
│   │   ├── raw_generation_attempts.csv  # every Gemini API attempt, including failures (§8)
│   │   ├── scaling_measurements.json    # Cross-Encoder vs #pairs, FAISS vs index size (§8)
│   │   ├── latency_summary.json         # statistics, ratios, cost, scaling fit (§8)
│   │   ├── latency_table.csv/.md        # the latency table (§8)
│   │   └── latency_analysis.md          # written analysis of the latency results
│   ├── tables/                          # recall, outcomes, rank shifts, rescues, latency, answers side by side (§9)
│   └── figures/                         # the 7 charts (§7-9)
└── report/
    ├── RAG_Report.pdf                   # the submitted report
    └── RAG_Report.html                  # its source
```

## Sources

- **Document:** *Ordinances and Regulations, B.Tech / B.Tech (Honours), applicable from 2010 to 2014 batch*, Indian Institute of Technology Madras (`data/document.pdf`).
- **Models:**
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
  - Google Gemini 2.5 Flash, via the `google-genai` SDK
- **Libraries:** PyMuPDF, LangChain text splitters, sentence-transformers, FAISS, pandas, matplotlib (versions in `requirements.txt`).
