# Latency benchmark and scaling analysis

All numbers are measured with `time.perf_counter()` by `src/benchmark_latency.py`. Raw data is in this folder; charts are in `results/figures/latency_breakdown.png` and `results/figures/latency_scaling.png`. Units are milliseconds unless stated otherwise.

## Setup

- **Hardware.** 12th Gen Intel Core i5-1235U laptop CPU (10 cores / 12 threads, PyTorch uses 10 threads), 15.3 GB RAM, **no GPU**.
- **Software.** Linux 6.12, Python 3.10.12, torch 2.12.0, faiss-cpu 1.15.1, sentence-transformers 6.1.0.
- **Corpus.** 85 chunks, 384-dimensional embeddings (all-MiniLM-L6-v2), FAISS `IndexFlatIP` (exact search). The re-ranker is cross-encoder/ms-marco-MiniLM-L-6-v2, run on CPU.
- **LLM.** Gemini 2.5 Flash on Vertex AI (project `lma-ass1`, location `global`), `temperature=0`, `seed=42`, `max_output_tokens=8192`. Both pipelines use identical settings, and every completed run was checked for this.
- **Excluded from all query timings.** Model loading, index building or loading, and three warm-up queries.
- **Shared retrieval.** Each repetition runs the vector search once. Pipeline 1 takes ranks 1–3 from it; Pipeline 2 re-scores the same 10 candidates. The extra cost of Pipeline 2 is therefore exactly cross-encoder scoring plus sorting.
- **Repetitions.**
  - Retrieval side: 33 questions × 5 repeats = 165 runs, interleaved round-robin.
  - Generation: 3 questions (KT-02, EX-05, C-01) × 5 repeats = 15 paired runs.

## Results

| Component | n | mean | median | std | p95 |
|---|---|---|---|---|---|
| Query embedding (Bi-Encoder) | 165 | 10.22 | 9.96 | 1.88 | 13.86 |
| FAISS top-10 search (85 vectors) | 165 | 0.068 | 0.069 | 0.020 | 0.096 |
| Cross-Encoder scoring, 10 pairs | 165 | 265.04 | 253.34 | 50.41 | 341.89 |
| Sort by CE score + top-3 | 165 | 0.018 | 0.017 | 0.006 | 0.030 |
| **Pipeline 1 retrieval side** | 165 | **10.30** | **10.04** | 1.89 | 13.94 |
| **Pipeline 2 retrieval side** | 165 | **275.35** | **263.99** | 50.69 | 355.12 |
| Additional re-ranking time (CE + sort) | 165 | 265.06 | 253.36 | 50.41 | 341.90 |
| Gemini generation, Answer 1 | 15 | 4465 | 4474 | 2217 | 8388 |
| Gemini generation, Answer 2 | 15 | 5746 | 3256 | 6600 | 17132 |
| **Pipeline 1 end-to-end** | 15 | 4496 | **4510** | 2214 | 8401 |
| **Pipeline 2 end-to-end** | 15 | 6035 | **3505** | 6591 | 17369 |

The full table, including the id → chunk lookup and top-3 selection (each about 0.01 ms), is in `latency_table.md` / `latency_table.csv`.

- **Re-ranking cost.** It adds **≈ 265 ms per query** (mean), which makes the retrieval side **≈ 26–27× slower** (26.7× by mean, 26.3× by median). Almost all of it is the cross-encoder forward pass; sorting 10 scores takes 0.02 ms.
- **End-to-end effect.** With the LLM included, re-ranking is **4.4 %** of Pipeline 2's mean end-to-end time.
  - By median, Pipeline 2 looks *faster* end-to-end (0.78×: 3.50 s vs 4.51 s).
  - By mean, it looks slower (1.34×: 6.03 s vs 4.50 s). That comes from one 26.3 s Gemini response for Answer 2 (C-01, repeat 5), not from re-ranking. Answer 1 ranges from 1.8 to 9.2 s, Answer 2 from 1.1 to 26.3 s.
  - Neither ratio measures re-ranking. With 15 runs, Gemini's spread (std 2.2–6.6 s) is roughly 10–25× larger than the whole re-ranking step, so the end-to-end difference between the pipelines is **not resolvable** at this sample size.
- **Tokens and cost** (successful calls only, medians per answer):
  - Answer 1: 592 input, 92 output, 492 thinking tokens.
  - Answer 2: 738 input, 149 output, 367 thinking tokens.
  - Answer 2 prompts are longer because the re-ranked chunks happen to be longer.
  - All 30 completed calls together: 21,505 input and 15,663 output+thinking tokens. At the assumed list prices ($0.30 / 1M input, $2.50 / 1M output; thinking billed as output; check the current Vertex AI pricing page) that is **≈ $0.046, or ≈ $0.0015 per answer**.
  - The cross-encoder runs locally and has no API cost.
- **Failed attempts.**
  - 37 API attempts in total: 31 succeeded, 6 failed.
  - All 6 failures were `429 RESOURCE_EXHAUSTED` (Vertex shared capacity).
  - Retries were bounded: at most 3 attempts per answer with 2 s and 5 s waits, and a stop after 3 consecutive failed calls.
  - A pair counts only when both answers succeeded. In C-01, repeat 4, Answer 1 failed all 3 attempts; the successful Answer 2 of that pair was discarded and the pair was re-measured in a resumed run.
  - Failed attempts are logged in `raw_generation_attempts.csv` and are not counted as measurements.

## Scaling

The scaling measurements are in `scaling_measurements.json` and the right-hand chart.

| | measured |
|---|---|
| Cross-Encoder, 1 / 10 / 20 / 40 / 85 pairs (median) | 31 / 232 / 545 / 1197 / **2677** ms |
| Cross-Encoder cost per pair | 20–31 ms (rises with batch size); linear fit 32.0 ms/pair |
| FAISS exact search, N = 85 / 1k / 10k / 100k / 500k (median) | 0.037 / 0.053 / 0.57 / 9.8 / 52 ms (N > 85 uses synthetic random unit vectors) |
| Embedding the whole corpus (one-time, offline) | 2162 ms for 85 chunks ≈ 25.4 ms per chunk |
| Embedding one query (per query) | 10.0 ms median |

## Explanation

- **Why the Bi-Encoder is fast at query time.** Chunk embeddings are computed once, offline (≈ 2.2 s for this corpus), and cached in `data/index/`. At query time only the **query** is encoded (≈ 10 ms). Retrieval then reduces to a dot product between one 384-dimensional vector and every stored vector, which FAISS does in 0.07 ms for 85 chunks. Query and documents never interact inside the model, which is exactly what makes caching possible.
- **Fast candidate retrieval stays fast as the corpus grows.** Exact search grew from 0.037 ms (85 vectors) to 10 ms (100k) and 52 ms (500k synthetic vectors), roughly linear in N but with a tiny constant. Approximate indexes (IVF, HNSW) would push this further down for millions of chunks; we did not need them at this size.
- **Why the Cross-Encoder is expensive.** It reads the query and the chunk **together** in one transformer pass. Self-attention runs across the tokens of both, which is what lets it notice negations, exceptions and which entity a number belongs to.
  - The price is that nothing can be pre-computed: every (query, chunk) pair is a fresh forward pass over about 200–300 tokens, which costs ≈ 20–31 ms per pair on this CPU.
  - Ten pairs take ≈ 250 ms, and the whole 85-chunk corpus takes ≈ 2.7 s **per query**.
- **Scaling cost of "Cross-Encoder over everything".** Cross-encoder time grows linearly with the number of chunks scored. Extrapolating the measured fit (32 ms/pair, same CPU, an *extrapolation*, not a measurement):
  - ≈ 32 s per query for 1,000 chunks
  - ≈ 5 min for 10,000
  - ≈ 53 min for 100,000

  In the two-stage design, query cost is almost independent of corpus size. At 100k chunks: ≈ 10 ms embedding + ≈ 10 ms search (measured on synthetic vectors) + ≈ 253 ms re-ranking of 10 candidates ≈ 0.27 s.
- **The two-stage trade-off.**
  - The Bi-Encoder gives cheap, scalable *recall*: it gets the right chunk into the top 10.
  - The Cross-Encoder gives expensive, accurate *precision*: it orders those 10 so the right ones reach the top 3.
  - The cost of the precision is bounded by the candidate count (k = 10 → ≈ 265 ms), not by the corpus size. That cost is small next to LLM generation (≈ 3.3–4.5 s median here).
  - The limitation is that re-ranking cannot recover a chunk the first stage missed. The C-04 failure in Stage 5 is an example. Raising k improves recall at a linear cross-encoder cost (≈ 20–31 ms per extra candidate on this CPU).

## Limitations

- **Single machine.** A laptop CPU with frequency scaling and other processes running. Absolute numbers will differ on other hardware; a GPU would cut cross-encoder time sharply.
- **Small generation sample.**
  - 15 paired runs over 3 questions. Gemini latency includes network and server queueing, which we cannot control, and varies strongly between identical requests. The end-to-end comparison between pipelines is therefore not statistically resolved.
  - Vertex AI hit transient `429` capacity limits. Retried calls report only the successful request's latency, not the waiting time.
  - Answer 1 and Answer 2 for a pair were sent one after the other, so the two answers share network conditions but are not simultaneous.
- **Earlier free-tier attempt not counted.** Its one completed pair came from a different backend (Gemini Developer API), so it is not comparable. It is kept as `archived_freetier_generation_attempt.csv`.
- **Scaling caveats.**
  - FAISS timings above 85 vectors use synthetic random vectors, which is fine for exact search, whose cost depends only on N and d.
  - Cross-encoder cost beyond 85 pairs is a linear extrapolation.
  - The FAISS search time for the real 85-vector index differs between the two runs (0.069 ms in the per-query benchmark, 0.037 ms in the scaling run). At these sub-0.1 ms durations, timer and cache effects dominate.
- **Cost figure.** Token counts are measured; prices are assumed list prices, so the cost is an estimate. It excludes failed attempts and the two single test requests.
