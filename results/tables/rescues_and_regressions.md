| query_id | category | relevant_rank_shifts | baseline_top3 | reranked_top3 | base_evcov@3 | rr_evcov@3 | rescue | strict_rescue | answer_1_verdict | answer_2_verdict | answer_change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KT-02 | keyword_trap | chunk 21: #4 -> #1 | [78, 84, 33] | [21, 12, 78] | 0.00 | 1.00 | True | True | insufficient_context | correct | improved |
| EX-05 | exception | chunk 61: #1 -> #1; chunk 62: #5 -> #2 | [61, 60, 58] | [61, 62, 60] | 0.60 | 1.00 | True | False | partially_correct | correct | improved |
| DS-02 | document_specific | chunk 31: #7 -> #1 | [33, 30, 32] | [31, 33, 29] | 0.00 | 1.00 | True | True | insufficient_context | correct | improved |
| NG-06 | negation_constraint | chunk 76: #4 -> #1 | [25, 75, 46] | [76, 59, 75] | 0.00 | 1.00 | True | True | insufficient_context | correct | improved |
| SY-03 | synonym_mismatch | chunk 74: #2 -> #4; chunk 75: #1 -> #3 | [75, 74, 59] | [48, 72, 75] | 1.00 | 0.50 | False | False | correct | insufficient_context | worsened |
| DS-05 | document_specific | chunk 58: #3 -> #5 | [60, 61, 58] | [60, 59, 61] | 1.00 | 0.00 | False | False | correct | insufficient_context | worsened |
