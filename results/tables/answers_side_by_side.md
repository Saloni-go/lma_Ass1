# Generated answers: baseline (Answer 1) vs re-ranked (Answer 2)

Actual Gemini 2.5 Flash outputs from Stage 5 (`results/stress_test/generation_results.json`), same prompt template and settings for both answers; only the 3 context chunks differ. Verdicts are a manual review against the verified reference answer and the exact context (`src/stress_report.py`).

## KT-02 (keyword_trap): What CGPA does a student need in order to opt for the B.Tech (Honours) programme, and when can they opt?

**Reference answer:** A CGPA of 8.5 and above at the end of the third semester; the option is exercised during the fourth semester. The student must also have cleared all prescribed courses in the first attempt; students with a 'U' or 'W' grade in any course are not eligible.

**Retrieval:** strict rescue: relevant chunk 21 (R.1.12) moved from baseline #4 to #1; the baseline top-3 [78, 84, 33] contained no relevant chunk, chunk 84 (R.30.1, the 'maintain CGPA 8.5 for award' distractor) was #2

**Context:** baseline top-3 [78, 84, 33], re-ranked top-3 [21, 12, 78], relevant [21]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | To be awarded the B.Tech (Honours) degree, students must maintain an average CGPA of 8.5 [Chunk 84, p. 25]. If this CGPA is not maintained, they will be awarded only a B.Tech Degree [Chunk 84, p. 25].<br><br>The provided context does not specify when a student can opt for the B.Tech (Honours) programme, nor does it state a CGPA requirement for *opting* into the programme itself, only for the *award* of the degree. | A student needs a CGPA of 8.5 and above at the end of their third semester to opt for the B.Tech (Honours) programme [Chunk 21, p. 8]. They can opt for this programme during the fourth semester [Chunk 21, p. 8].<br><br>Additionally, to be eligible, students must have cleared all prescribed courses in the curriculum in the first attempt and must not have obtained a "U" or "W" grade in any course [Chunk 21, p. 8]. |
| Verdict | insufficient_context | correct |
| Missing clauses | CGPA 8.5 at end of 3rd semester; opt during 4th semester; cleared all courses in first attempt | - |
| Unsupported claims | none found | none found |
| Citation issues | - | - |

**Answer change:** improved

## EX-05 (exception): Can attendance below 85% be condoned for a student who was hospitalised?

**Reference answer:** Yes, for prolonged illness and/or hospitalisation with a medical certificate: the student must have more than 40% actual attendance, and the notional attendance (classes attended x 100 / (classes held in the semester minus classes held during the prolonged illness/hospitalisation)) must be at least 85%. Casual absenteeism and minor illnesses are not condoned.

**Retrieval:** partial rescue: chunk 61 stayed #1, chunk 62 (notional-attendance formula and the >= 85% condition) moved from #5 to #2; evidence coverage of the top-3 rose from 3/5 to 5/5 quotes

**Context:** baseline top-3 [61, 60, 58], re-ranked top-3 [61, 62, 60], relevant [61, 62]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | Yes, attendance below 85% can be condoned for a student who was hospitalized, provided they meet a specific condition. A student must have more than 40% attendance (calculated as per normal cases) to be eligible for condonation in cases of prolonged illness and/or hospitalization with a medical certificate [Chunk 61, p. 19].<br><br>This is in contrast to cases of minor illnesses and casual absenteeism, for which condonation of attendance is not permissible [Chunk 60, p. 19; Chunk 61, p. 19]. | Yes, attendance below 85% can be condoned for a student who was hospitalized, provided they meet specific conditions.<br><br>For cases of prolonged illness and/or hospitalization with a medical certificate, a student must have more than 40% attendance, calculated based on the actual number of classes attended out of the total classes held till the date of compilation, to be eligible for condonation [Chunk 61, p. 19].<br><br>If eligible, a "Notional % of attendance" is calculated as:<br>(Actual no. of classes attended x 100) / ((Total no. of classes held in the semester) - (No. of classes held during the days of prolonged illness and or hospitalization))<br>This notional percentage must be ≥ 85% for condonation to be granted [Chunk 62, p. 19].<br><br>It is important to note that condonation is not permissible for cases of casual absenteeism or minor illnesses if attendance is below 85% [Chunk 60, p. 19; Chunk 61, p. 19]. |
| Verdict | partially_correct | correct |
| Missing clauses | notional attendance must be >= 85% (formula excluding illness days) | - |
| Unsupported claims | none found | none found |
| Citation issues | - | - |

**Answer change:** improved

## DS-02 (document_specific): From the 2013 batch onwards, how many Humanities Pass/Fail (HPF) credits are required?

**Reference answer:** 4 non-CGPA credits under the Humanities Pass/Fail option (HPF) in the table for the 2013 batch onwards (the 2010 and 2011-2012 tables specify 2).

**Retrieval:** strict rescue: chunk 31 (2013-onwards table, contains the HPF row) moved from #7 to #1; the baseline top-3 [33, 30, 32] held only the text paragraph and the two table tails without the HPF row

**Context:** baseline top-3 [33, 30, 32], re-ranked top-3 [31, 33, 29], relevant [31]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | The provided context does not explicitly state the number of Humanities Pass/Fail (HPF) credits required for students from the 2013 batch onwards.<br><br>The context indicates that for the 2013 batch onwards, 7-13 credits of coursework done under the Pass/Fail option are not counted towards CGPA calculation [Chunk 33, p. 12]. Chunk 32, which applies to this batch, lists "Pass/ Fail Electives" (PPF) as 0-6 credits and "Industrial Lecture (Pass/Fail)" (PIL) as 1 credit under the Pass/Fail category [Chunk 32, p. 12]. However, "Humanities Pass/Fail" (HPF) is not mentioned. | For the 2013 batch onwards, 4 Humanities Pass/Fail (HPF) credits are required, which are non-CGPA credits [Chunk 31, p. 12]. |
| Verdict | insufficient_context | correct |
| Missing clauses | HPF = 4 credits | - |
| Unsupported claims | none found | none found |
| Citation issues | - | - |

**Answer change:** improved

## NG-06 (negation_constraint): Which courses is a student not allowed to repeat?

**Reference answer:** A course that has been successfully completed cannot be repeated. Exception: from academic year 2014-15, students who have taken all core and elective courses but still fail to reach the minimum CGPA of 5 (and so are not eligible for the degree) may repeat core or elective courses to earn the minimum CGPA.

**Retrieval:** strict rescue: chunk 76 (R.25.1 tail, 'A course successfully completed cannot be repeated' + the 2014-15 exception) moved from #4 to #1; the baseline top-3 [25, 75, 46] held the 'No withdrawal ... will be allowed' registration clause (distractor), the U/W repetition rules and the n-4 registration rule

**Context:** baseline top-3 [25, 75, 46], re-ranked top-3 [76, 59, 75], relevant [76]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | The provided context does not state which courses a student is *not allowed* to repeat. It only specifies conditions under which courses *must* or *may* be repeated:<br>*   A student who earns a U or W grade in a core course has to repeat it compulsorily when the course is offered next [Chunk 75, p. 22].<br>*   A student securing a U or W grade in an elective course may repeat it if they desire to get a successful grade [Chunk 75, p. 22].<br><br>Therefore, based on the provided information, there are no courses explicitly stated as non-repeatable. | A student is generally not allowed to repeat a course that has been successfully completed [Chunk 76, p. 22].<br><br>However, for students from the academic year 2014-15 onwards, if they have completed all core and elective courses but fail to achieve a minimum CGPA of 5 and are thus ineligible for a degree, they are permitted to repeat core or elective courses to meet the minimum CGPA requirement [Chunk 76, p. 22]. |
| Verdict | insufficient_context | correct |
| Missing clauses | a successfully completed course cannot be repeated; 2014-15 exception for CGPA below 5 | - |
| Unsupported claims | none found | none found |
| Citation issues | - | prohibition cited as [Chunk 76, p. 22]; the clause is on PDF page 23 (chunk 76 spans pp. 22-23) |

**Answer change:** improved

## SY-03 (synonym_mismatch): How much time does a student have to dispute a grade they think is unfair?

**Reference answer:** The student can contact the teacher concerned within three weeks from the commencement of the semester immediately following the announcement of results; if the teacher finds the case genuine, he/she may re-examine and forward a revised grade to the Dean of Academic Courses through the Chairman of the Class Committee.

**Retrieval:** worsened: chunk 74 (the 'within three weeks' clause) fell from #2 to #4, out of the top-3; chunk 75 (re-examination procedure) went from #1 to #3; chunks 48 (dropping courses) and 72 (grade definitions) were promoted to #1 and #2

**Context:** baseline top-3 [75, 74, 59], re-ranked top-3 [48, 72, 75], relevant [74, 75]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | A student has three weeks from the commencement of the semester immediately following the announcement of the results to contact the teacher concerned if they feel aggrieved about a grade [Chunk 74, p. 22]. The student can access their answer paper(s) in the end semester examination, which may be shown to them by the teachers [Chunk 75, p. 22]. If the teacher finds the case genuine, they may re-examine it and forward a revised grade, if any, to the Dean of Academic Courses through the Chairman of the Class Committee with justification [Chunk 75, p. 22]. | The provided context does not specify a time limit for a student to dispute a grade they think is unfair.<br><br>The ordinances state that a student shall have access to their end-semester examination answer paper(s) which may be shown by the teachers concerned. If the teacher deems the case genuine, they may re-examine it and forward a revised grade with justification [Chunk 75, p. 22]. However, no deadline for initiating this process is mentioned. |
| Verdict | correct | insufficient_context |
| Missing clauses | - | within three weeks from the commencement of the following semester |
| Unsupported claims | none found | none found |
| Citation issues | - | - |

**Answer change:** worsened

## DS-05 (document_specific): What attendance code does a student with 90% attendance in a course receive?

**Reference answer:** 'G' (Good), which covers attendance of 85 to 94%.

**Retrieval:** worsened: chunk 58 (attendance-code table) fell from #3 to #5, out of the top-3; replaced by chunk 59 (R.14.2 shortfall rule)

**Context:** baseline top-3 [60, 61, 58], re-ranked top-3 [60, 59, 61], relevant [58]

| | Answer 1 (baseline top-3) | Answer 2 (re-ranked top-3) |
|---|---|---|
| Answer | A student with 90% attendance in a course receives an attendance code of "G" [Chunk 58, p. 18]. | The provided context does not explicitly state what attendance code a student with 90% attendance in a course receives. The regulations only specify that if a student has less than 85% attendance, they should be assigned a "W" grade in that subject along with "P" as the attendance code [Chunk 60, p. 18]. For attendance not ≥ 85%, a "W" grade shall be awarded [Chunk 61, p. 19]. The context does not describe attendance codes for students meeting or exceeding the 85% minimum requirement. |
| Verdict | correct | insufficient_context |
| Missing clauses | - | 85-94% -> G (Good) |
| Unsupported claims | none found | none found |
| Citation issues | - | - |

**Answer change:** worsened
