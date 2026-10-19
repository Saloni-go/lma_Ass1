| Component | n | mean (ms) | median (ms) | std (ms) | p95 (ms) |
|---|---|---|---|---|---|
| **retrieval (33 questions x 5 repeats)** | | | | | |
| Query embedding (Bi-Encoder, 1 query) | 165 | 10.217 | 9.959 | 1.880 | 13.859 |
| FAISS top-10 search (85 vectors) | 165 | 0.068 | 0.069 | 0.020 | 0.096 |
| FAISS id -> chunk lookup | 165 | 0.012 | 0.011 | 0.003 | 0.015 |
| Pipeline 1: take top-3 | 165 | 0.001 | 0.001 | 0.000 | 0.001 |
| Cross-Encoder scoring (10 pairs) | 165 | 265.037 | 253.340 | 50.413 | 341.887 |
| Pipeline 2: sort by CE score + top-3 | 165 | 0.018 | 0.017 | 0.006 | 0.030 |
| Shared retrieval (embed + search + lookup) | 165 | 10.297 | 10.038 | 1.886 | 13.935 |
| Pipeline 1 retrieval side (total) | 165 | 10.298 | 10.039 | 1.886 | 13.936 |
| Pipeline 2 retrieval side (total) | 165 | 275.352 | 263.994 | 50.690 | 355.117 |
| Additional re-ranking time (CE + sort) | 165 | 265.055 | 253.357 | 50.413 | 341.904 |
| **with generation (3 questions x 5 repeats)** | | | | | |
| Gemini generation, Answer 1 (P1 context) | 15 | 4465.213 | 4473.500 | 2217.392 | 8387.500 |
| Gemini generation, Answer 2 (P2 context) | 15 | 5745.780 | 3256.300 | 6600.380 | 17132.490 |
| Pipeline 1 end-to-end | 15 | 4495.860 | 4510.400 | 2214.414 | 8401.250 |
| Pipeline 2 end-to-end | 15 | 6034.640 | 3504.700 | 6591.336 | 17369.190 |
