"""Stage 4: document-specific stress-test questions with verified references.

Every question carries `evidence`: exact quotes from the source document. Pages
and relevant chunk IDs are DERIVED from those quotes (never typed by hand), and
each quote is verified against
  1. the cleaned page text (data/processed/pages.json), and
  2. the raw PyMuPDF text of the same page (word-sequence match),
so a reference answer cannot cite text that is not in the PDF. `distractor`
quotes mark passages that plausibly attract retrieval for the wrong reason;
they are verified the same way.

These are CANDIDATE stress tests. Whether the baseline actually fails (or the
Cross-Encoder actually rescues) a question is measured in a later stage.

Usage (from the project root):
    python -m src.build_questions
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter

import pymupdf

from . import config
from .pdf_processing import QUOTE_MAP

EVAL_DIR = config.DATA_DIR / "evaluation"

CATEGORIES = {
    "keyword_trap": "High lexical overlap with a different (wrong) passage",
    "exception": "General rule plus an 'except / however / unless' condition",
    "negation_constraint": "Prohibitions, 'not allowed', role or count limits",
    "synonym_mismatch": "Query wording differs from the document's terminology",
    "document_specific": "Batch-specific variants and near-identical tables",
    "control": "Easy, direct lookup (sanity check)",
}

QUESTIONS = [
    # ------------------------------------------------------------ keyword traps
    {
        "id": "KT-01", "category": "keyword_trap",
        "question": "If a student's absence from NCC classes is unauthorized, how must the attendance deficiency be made up?",
        "reference_answer": "It has to be made up during the following year, and an NCC deficiency is made up in NSO/NSS programmes at four times the deficiency (each NCC/NSS class missed is made up by four NSO classes). (If the absence is on valid grounds, it is instead made up in the extra NCC/NSO/NSS programmes offered in May of the same year.)",
        "evidence": [
            "If the absence is unauthorized, the deficiency will have to be made up during the following year.",
            "The deficiency in NCC will have to be made up in NSO / NSS programmes at the rate of four times the deficiency",
            "If the absence is on valid grounds, the deficiency will be made up in the extra NCC / NSO / NSS programmes to be offered during May of the same year.",
        ],
        "distractor": ["whatever may be the reason for the shortfall in attendance"],
        "difficulty": "Shares 'attendance', 'deficiency/shortfall' and 'absence' with the much larger academic-attendance section (R.14), which may outrank the short NCC sub-clause R.27.5.",
    },
    {
        "id": "KT-02", "category": "keyword_trap",
        "question": "What CGPA does a student need in order to opt for the B.Tech (Honours) programme, and when can they opt?",
        "reference_answer": "A CGPA of 8.5 and above at the end of the third semester; the option is exercised during the fourth semester. The student must also have cleared all prescribed courses in the first attempt; students with a 'U' or 'W' grade in any course are not eligible.",
        "evidence": [
            "Students admitted to B.Tech Programme can opt for B.Tech (Honours) programme during the fourth semester.",
            "The student should have a CGPA of 8.5 and above at the end of their third semester and should have cleared all the courses as prescribed in the curriculum in the first attempt.",
        ],
        "distractor": ["the B.Tech (Honours) students must maintain an average CGPA of 8.5"],
        "difficulty": "R.30.1 (p. 25) contains the same keywords ('B.Tech (Honours)', 'CGPA of 8.5') but is about keeping 8.5 to be awarded the degree, not about eligibility to opt in (R.1.12, p. 8).",
    },
    {
        "id": "KT-03", "category": "keyword_trap",
        "question": "What minimum CGPA must a student keep to avoid being relieved from the programme, and what happens if it drops below that?",
        "reference_answer": "5.0, calculated over successfully completed courses at the end of each semester. If it is below 5.0 (or the student earns less than the average 22 credits), a warning is issued with intimation to the parent; if in the following semester the CGPA is still below 5.0 or the student cannot earn an average of 15 credits (R.10.1), he/she is relieved from the programme at the end of that semester.",
        "evidence": [
            "A student should have a minimum CGPA of 5.0 (see R.26.2), calculated for the courses successfully completed at the end of each semester.",
            "the student will be issued a warning under intimation to the parent.",
            "In the following semester, if the CGPA is less than 5.0 or the student could not earn an average of 15 credits, as per R.10.1, he/she will be relieved from the programme at the end of the semester.",
        ],
        "distractor": ["secured a CGPA of 5.0 in passed subjects only."],
        "difficulty": "'CGPA of 5.0' also appears in degree eligibility (R.29.1e) and in the 2014-15 course-repetition rule (R.25.1); only R.10.3 describes the warning-then-relieve procedure.",
    },
    {
        "id": "KT-04", "category": "keyword_trap",
        "question": "How many credits must a student admitted in the 2013 batch earn for the award of the B.Tech degree?",
        "reference_answer": "184 credits. The 180-credit figure applies to batches before 2013.",
        "evidence": [
            "designed to have a total of 180 credits (184 credits from 2013) for the award of the B.Tech degree.",
            "The minimum number of credits to be earned for the award of the B.Tech. Degree is 180 (184 from 2013)",
        ],
        "distractor": [],
        "difficulty": "The wrong value (180) is printed right next to the right one; the answer depends on the parenthetical batch qualifier, which retrieval scores cannot distinguish.",
    },
    {
        "id": "KT-05", "category": "keyword_trap",
        "question": "How many times can a student take the supplementary examination in a course?",
        "reference_answer": "Only once; no second chance is given. If the student fails the supplementary examination, the course has to be repeated in a regular semester when it is offered.",
        "evidence": [
            "A student is eligible for only one Supplementary Examination in any course. No second chance is given.",
            "In case a student fails in the Supplementary Examination he/she has to repeat the course in a regular semester, as and when it is offered.",
        ],
        "distractor": ["may permit the student to appear for a second make-up examination."],
        "difficulty": "The make-up examination rules (R.20) use near-identical vocabulary ('examination', 'missed', 'second') and DO allow a second make-up in exceptional cases; mixing them up flips the answer.",
    },
    {
        "id": "KT-06", "category": "keyword_trap",
        "question": "Who acts as Chairman of the Class Committee for a department's own courses, and who nominates them?",
        "reference_answer": "One professor of the department concerned, preferably not associated with teaching the class, nominated by the Head of the Department concerned. (For the common courses of the first three semesters, the Chairman is instead a professor not associated with teaching the class, nominated by the Dean of Academic Courses.)",
        "evidence": [
            "one professor of the Department concerned, preferably not associated with teaching of the class, to be nominated by the Head of the Department concerned, to act as the Chairman of the Class Committee.",
        ],
        "distractor": ["One professor-not associated with teaching the particular class to be nominated by the Dean of Academic Courses to act as the Chairman of the Class Committee."],
        "difficulty": "R.4.2 (common courses) and R.4.3 (department courses) contain almost the same sentence; only the nominating authority differs (Dean vs Head of Department).",
    },
    # ------------------------------------------------------------ exceptions
    {
        "id": "EX-01", "category": "exception",
        "question": "Which students who have completed their first semester are not eligible to be considered for a change of branch?",
        "reference_answer": "Engineering Design Dual Degree students. All other students who have successfully completed the first semester are eligible for consideration, subject to the availability of vacancies.",
        "evidence": [
            "All students who have successfully completed the first semester of the course, except Engineering Design Dual Degree students, will be eligible for consideration for change of branch, subject to the availability of vacancies.",
        ],
        "distractor": ["Such change will be considered only at the end of the first semester."],
        "difficulty": "The exception is a mid-sentence clause in R.5.3; other change-of-branch clauses (R.5.2, R.5.4-5.6) match the query just as well lexically.",
    },
    {
        "id": "EX-02", "category": "exception",
        "question": "Can a dual degree student change to a different discipline or branch?",
        "reference_answer": "They are not eligible for a branch/programme change to the 4-year B.Tech programme. However, they are eligible for a change of discipline within the same Department if it offers more than one discipline under their dual degree programme (from 2012, Applied Mechanics is treated as a discipline for AE/CE/EE/NA). Dual degree students other than Engineering Design students are also eligible for change of branch at the end of the first semester as per the Dual Degree Regulations.",
        "evidence": [
            "Dual degree students are not eligible for branch/ programme change to 4-year B.Tech programme.",
            "However, they are eligible for change of discipline within the same Department, in case the Department offers more than one discipline under their dual degree programme.",
            "From 2012 batch, Applied Mechanics will be treated as one of the discipline in the case of AE/CE/EE/NA Departments.",
            "The Dual Degree students except Engineering Design students are also eligible for change of branch at the end of the first semester as per the Dual Degree Regulations.",
        ],
        "distractor": ["Change over from 4-year B.Tech programme to 5-year dual degree programme is permitted within the same Department"],
        "difficulty": "A prohibition followed by two exceptions. In the Task 3 runs, both pipelines answered a related question with only the prohibition, although this chunk was in the context.",
    },
    {
        "id": "EX-03", "category": "exception",
        "question": "Must a student earn at least 15 credits in every semester, including the final one?",
        "reference_answer": "A student must maintain an average of 15 credits per semester (including the first two semesters) to continue. However, in the final semester a student may earn less than 15 credits if that is sufficient to fulfil the requirements for the award of the degree.",
        "evidence": [
            "Thereafter, he/she must maintain an average of 15 credits per semester, including the first two semesters, to continue his/ her studies.",
            "However, in the final semester, a student may earn less than 15 credits if it is sufficient for him/ her to fulfill the requirements for the award of the degree.",
        ],
        "distractor": ["The curriculum for any semester, except for the final semester will normally carry credits between 21 and 29."],
        "difficulty": "The relaxation is a 'However' sentence after the general rule; R.7.1 (21-29 credits, also excepting the final semester) is a lexically similar distractor.",
    },
    {
        "id": "EX-04", "category": "exception",
        "question": "Can a student repeat a course they have already passed in order to improve their CGPA?",
        "reference_answer": "Generally no: a successfully completed course cannot be repeated. Exception: from academic year 2014-15, students who have taken all core and elective courses but still fail to get the minimum CGPA of 5 (and so are not eligible for the degree) are permitted to repeat core or elective courses to earn the minimum CGPA.",
        "evidence": [
            "A course successfully completed cannot be repeated.",
            "However from academic year 2014-15 where students have taken all core and elective courses but still fail to get a minimum CGPA of 5",
        ],
        "distractor": ["A student who earns a U or W grade in a core course has to repeat it compulsorily when the course is offered next."],
        "difficulty": "The general course-repetition rule (U/W grades, p. 22) is the obvious match; the passed-course prohibition and its 2014-15 exception continue on p. 23.",
    },
    {
        "id": "EX-05", "category": "exception",
        "question": "Can attendance below 85% be condoned for a student who was hospitalised?",
        "reference_answer": "Yes, for prolonged illness and/or hospitalisation with a medical certificate: the student must have more than 40% actual attendance, and the notional attendance (classes attended x 100 / (classes held in the semester minus classes held during the prolonged illness/hospitalisation)) must be at least 85%. Casual absenteeism and minor illnesses are not condoned.",
        "evidence": [
            "For calculation of attendance in case of prolonged illness and/or hospitalization with medical certificate:",
            "A student should have more than 40% attendance calculated as per (a) above to be eligible for condonation.",
            "which should be ≥ 85% for condonation",
            "(Total no. of classes held in the semester) - (No. of classes held during the days of prolonged illness and or hospitalization)",
            "Such cases will not come under the purview of condonation of attendance.",
        ],
        "distractor": ["whatever may be the reason for the shortfall in attendance, will not be permitted to appear in the end-semester examination"],
        "difficulty": "R.14.2 says students below 85% are barred 'whatever may be the reason', which contradicts the condonation route in R.14.4(b); the general rule is more likely to rank first.",
    },
    {
        "id": "EX-06", "category": "exception",
        "question": "Can a student who missed the make-up examination be given another one?",
        "reference_answer": "Not normally. However, in exceptional cases of prolonged illness that make the student miss the make-up examination, the Chairman of the Senate, in consultation with the Dean of Academic Courses or with the approval of the Senate, may permit a second make-up examination.",
        "evidence": [
            "A student who misses this make-up examination will not normally be given another make-up examination.",
            "However, in exceptional cases of prolonged illness resulting in the student missing a make-up examination, the Chairman of the Senate, in consultation with the Dean of Academic Courses or with approval of the Senate may permit the student to appear for a second make-up examination.",
        ],
        "distractor": ["A student is eligible for only one Supplementary Examination in any course. No second chance is given."],
        "difficulty": "Mirror of KT-05: the supplementary-exam rule ('No second chance') is a strong lexical match with the opposite answer.",
    },
    # ------------------------------------------------------------ negation / constraints
    {
        "id": "NG-01", "category": "negation_constraint",
        "question": "Which students are not allowed to take the NCC programme?",
        "reference_answer": "Foreign nationals and women students; the NCC programme is not available to them. (All students must take either NCC or NSO/NSS.)",
        "evidence": [
            "All students admitted to the B.Tech programme will have to take either NCC or NSO / NSS",
            "NCC programme is not available for Foreign nationals and women students.",
        ],
        "distractor": [],
        "difficulty": "Negative restriction phrased as 'not available for'; many chunks mention NCC/NSO/NSS (attendance, grades X/Y, continuation rule R.10.2).",
    },
    {
        "id": "NG-02", "category": "negation_constraint",
        "question": "Can a student who joined an IIT through JEE appear for JEE again the following year?",
        "reference_answer": "No. Candidates who join any of the IITs, IT-BHU Varanasi or ISMU Dhanbad through JEE/JEE (Advanced) are NOT permitted to appear in subsequent JEE/JEE (Advanced) from the following year.",
        "evidence": [
            "will NOT be permitted to appear in subsequent JEE(s) / JEE (Advanced) from 2013 from the following year",
        ],
        "distractor": ["A student can have only two attempts to write JEE / JEE (Advanced) from 2013."],
        "difficulty": "The two-attempts rule in the same chunk suggests a second attempt is allowed; the answer hinges on the capitalised NOT in R.1.11.",
    },
    {
        "id": "NG-03", "category": "negation_constraint",
        "question": "Is a student allowed to withdraw from a contact course?",
        "reference_answer": "No. Withdrawal from a contact course is not permitted.",
        "evidence": ["Withdrawal from a contact course is not permitted."],
        "distractor": ["he/she may drop courses if the student maintains a minimum attendance grade G"],
        "difficulty": "One-line prohibition (R.9.6) at the top of p. 17; R.7.1 allows dropping ordinary courses, so a context without R.9.6 suggests 'yes'.",
    },
    {
        "id": "NG-04", "category": "negation_constraint",
        "question": "Can a third-semester student register for an M.Tech course?",
        "reference_answer": "No. Only students in the V or higher semesters are eligible to register for M.Tech/M.Sc. courses, and they must obtain the consent of the teacher offering the course.",
        "evidence": [
            "only students belonging to the V or higher semesters are eligible to register for M.Tech/M.Sc. courses, and every such student desiring to register for such a course must obtain the consent of the teacher offering the course.",
        ],
        "distractor": [],
        "difficulty": "Constraint expressed positively ('only ... V or higher'); the query's 'third semester' must be compared with a Roman numeral. The sentence sits at the end of a chunk that leads into Table-1.",
    },
    {
        "id": "NG-05", "category": "negation_constraint",
        "question": "Is an attendance shortage caused by casual absenteeism condoned?",
        "reference_answer": "No. For casual absenteeism no condonation of attendance is permissible; a student with less than 85% attendance is assigned a 'W' grade in that subject with 'P' as the attendance code.",
        "evidence": [
            "For cases of casual absenteeism, no condonation of attendance is permissible.",
            "If a student has less than 85% attendance, he/she should be assigned \"W\" grade in that subject along with \"P\" as the attendance code.",
        ],
        "distractor": ["A student should have more than 40% attendance calculated as per (a) above to be eligible for condonation."],
        "difficulty": "Condonation passages (which allow condonation for prolonged illness) are semantically close; Bi-Encoder embeddings may not separate 'no condonation' from 'eligible for condonation'.",
    },
    {
        # Added after a screen of 24 negation phrasings (see results/stress_test/negation_screen.json).
        "id": "NG-06", "category": "negation_constraint",
        "question": "Which courses is a student not allowed to repeat?",
        "reference_answer": "A course that has been successfully completed cannot be repeated. Exception: from academic year 2014-15, students who have taken all core and elective courses but still fail to reach the minimum CGPA of 5 (and so are not eligible for the degree) may repeat core or elective courses to earn the minimum CGPA.",
        "evidence": [
            "A course successfully completed cannot be repeated.",
            "However from academic year 2014-15 where students have taken all core and elective courses but still fail to get a minimum CGPA of 5 and they are not eligible to get degree, these students are permitted to repeat the core or elective courses to earn the minimum CGPA.",
        ],
        "distractor": ["No withdrawal from any of the courses for which a student has registered will be allowed, except as per regulation 7.1."],
        "difficulty": "The prohibition is one short sentence; R.25.1 (U/W courses must or may be repeated) and the 'No withdrawal ... will be allowed' registration clause share the words 'repeat', 'course' and 'not allowed' and look like answers.",
    },
    # ------------------------------------------------------------ synonym mismatch
    {
        "id": "SY-01", "category": "synonym_mismatch",
        "question": "What happens to a student found guilty of harassing or bullying junior students?",
        "reference_answer": "This is ragging: once a student's involvement in ragging is established, the student is dismissed from the Institution and will not be admitted into any other Institution. Ragging is a criminal, non-bailable offence, and collective punishment is possible if individuals cannot be identified.",
        "evidence": [
            "Ragging of any dimension is a criminal and non-bailable offence in our country.",
            "Once the involvement of a student is established in ragging, the offending student will be dismissed from the Institution and will not be admitted into any other Institution.",
            "Avenues also exist for collective punishment, if individuals can not be identified in this inhuman act.",
        ],
        "distractor": ["Any act of indiscipline of a student reported to the Dean, Academic Courses, will be referred to a Discipline and Welfare Committee"],
        "difficulty": "The query never uses the document's term 'ragging'; the general discipline clause R.13.2 is a plausible but incomplete match.",
    },
    {
        "id": "SY-02", "category": "synonym_mismatch",
        "question": "For how long can a student take a break from studies for health reasons?",
        "reference_answer": "The Dean of Academic Courses may permit a student to withdraw for a semester or longer for ill health or other valid reasons, but normally only for a maximum continuous period of two semesters. Semesters withdrawn on medical grounds are excluded from the 12-semester maximum duration.",
        "evidence": [
            "A student may be permitted by the Dean of Academic Courses to withdraw from the programme for a semester or longer for reasons of ill health or other valid reasons.",
            "Normally, a student will be permitted to withdraw from the programme only for a maximum continuous period of two semesters.",
            "excluding semesters withdrawn on medical grounds",
        ],
        "distractor": [],
        "difficulty": "'Take a break' vs the document's 'temporary withdrawal'; 'health' vs 'ill health / medical grounds'.",
    },
    {
        "id": "SY-03", "category": "synonym_mismatch",
        "question": "How much time does a student have to dispute a grade they think is unfair?",
        "reference_answer": "The student can contact the teacher concerned within three weeks from the commencement of the semester immediately following the announcement of results; if the teacher finds the case genuine, he/she may re-examine and forward a revised grade to the Dean of Academic Courses through the Chairman of the Class Committee.",
        "evidence": [
            "In case any student feels aggrieved, he/she can contact the teacher concerned within three weeks from the commencement of the semester immediately following the announcement of the results.",
            "If the teacher feels that the case is genuine, he/she may re-examine the case and forward a revised grade, if any, to the Dean of Academic Courses through the Chairman of the Class Committee",
        ],
        "distractor": [],
        "difficulty": "'Dispute a grade' vs 'feels aggrieved' / 're-examination of answer papers'; many grade-related chunks (grade table, grade card, awarding grades) compete.",
    },
    {
        "id": "SY-04", "category": "synonym_mismatch",
        "question": "What share of students receive a need-based scholarship, and do they have to pay tuition fees?",
        "reference_answer": "25% of the students admitted or 25% of the sanctioned strength, whichever is less, receive the Merit-Cum-Means (MCM) scholarship for the entire programme, and they are exempted from tuition fees. In addition, 10% of the students admitted or 10% of the sanctioned strength, whichever is less, are exempted from tuition fees.",
        "evidence": [
            "25% of the students admitted or 25% of the sanctioned strength, whichever is less, will be awarded Merit-Cum-Means (MCM) scholarship",
            "These students are exempted from paying tuition fees.",
            "In addition, 10% of the students admitted or 10% of the sanctioned strength, whichever is less, are exempted from payment of tuition fees",
        ],
        "distractor": ["The award of merit scholarships to the B.Tech/ B.Tech (Honours) students will be governed by the regulations framed by the Senate"],
        "difficulty": "'Need-based' corresponds only loosely to 'Merit-Cum-Means'; ordinance O.4 on merit scholarships is a keyword distractor without the numbers.",
    },
    {
        "id": "SY-05", "category": "synonym_mismatch",
        "question": "If I spend a semester abroad on an exchange programme, do those credits and grades count at IIT Madras?",
        "reference_answer": "The credits earned abroad count towards the minimum credits required for the degree. The IIT Madras transcript shows only the courses, credits and grades completed at IITM plus the total number of credits (without grades) earned at the other university, and the CGPA based on IITM credits alone is used for the award of prizes. (The document does not say anything further about how foreign grades affect the regular CGPA.)",
        "evidence": [
            "IIT Madras transcripts will only indicate the courses, credits and grades completed at IITM and the total no. of credits (without grades) earned in other Universities in a particular semester.",
            "The CGPA calculation based on credits earned/completed at IIT Madras alone is to be considered for award of prizes.",
            "The credits earned at Universities abroad will be taken into account for calculation of minimum required credits for the award of the degree.",
        ],
        "distractor": [],
        "difficulty": "'Exchange programme' / 'abroad' vs 'credit transfer'; the passage is an un-numbered section between R.29 and R.30, and a faithful answer must not overstate what the document says about CGPA.",
    },
    {
        "id": "SY-06", "category": "synonym_mismatch",
        "question": "Can I sign up for the next semester if I still owe money to the hostel?",
        "reference_answer": "No. A student is permitted to enroll only if he/she has cleared all dues to the Institute, Hostel, Library and NCC Unit up to the end of the previous semester (and is not debarred on disciplinary grounds).",
        "evidence": [
            "will be permitted to enroll only if he/she has cleared all dues to the Institute, Hostel, Library and the NCC Unit up to the end of the previous semester",
            "provided he / she is not debarred from enrolment, on disciplinary grounds.",
        ],
        "distractor": ["no dues to the Institute, Hostels, Libraries, NCC / NSS / NSO etc."],
        "difficulty": "'Sign up' / 'owe money' vs 'enrolment' / 'cleared all dues'; the degree-eligibility clause R.29.1(f) also mentions dues.",
    },
    # ------------------------------------------------------------ document-specific
    {
        "id": "DS-01", "category": "document_specific",
        "question": "For the 2011-2012 batches, how many CGPA credits are prescribed for free electives?",
        "reference_answer": "0-9 CGPA credits, with a course-wise minimum credit requirement of 9 (Table-1 for 2011-2012 batches). For comparison, the 2010 table gives 6 (minimum 6) and the 2013-onwards table gives 6-9 (minimum 6).",
        "evidence": [
            "(for 2011-2012 batches)",
            "| 6. | FREE ELECTIVES |  | 0-9 |  | 9 |",
        ],
        "distractor": ["| 6. | FREE ELECTIVES |  | 6 |  | 6 |", "| 6. | FREE ELECTIVES |  | 6-9 |  | 6 |"],
        "difficulty": "Three near-identical tables (pp. 10-12). The FREE ELECTIVES row falls in the second half of each split table, a chunk that does not contain the batch caption.",
    },
    {
        "id": "DS-02", "category": "document_specific",
        "question": "From the 2013 batch onwards, how many Humanities Pass/Fail (HPF) credits are required?",
        "reference_answer": "4 non-CGPA credits under the Humanities Pass/Fail option (HPF) in the table for the 2013 batch onwards (the 2010 and 2011-2012 tables specify 2).",
        "evidence": [
            "(for 2013 batch onwards)",
            "|  | Pass/Fail Option | HPF |  | 4 |  |",
        ],
        "distractor": ["|  | Pass/Fail Option | HPF |  | 2 |  |"],
        "difficulty": "The three tables differ in a single cell; retrieval must pick the 2013 table out of three almost identical chunks.",
    },
    {
        "id": "DS-03", "category": "document_specific",
        "question": "For a student of the 2010 batch, does an E grade make them ineligible for the B.Tech (Honours) degree?",
        "reference_answer": "Yes. For the 2010 batch, students with an 'E', 'U' or 'W' grade in any course are not eligible for the B.Tech (Honours) degree; from the 2011 batch onwards only 'U' or 'W' grades disqualify.",
        "evidence": [
            "Students who have obtained \"E\" or \"U\" or \"W\" grade in any course are not eligible for B.Tech (Honours) degree.",
            "From 2011 batch onwards In addition to B.Tech regulations specified above, the B.Tech (Honours) students must earn minimum of 12 additional PMT credits.",
        ],
        "distractor": ["Students who have obtained \"U\" or \"W\" grade in any course are not eligible for B.Tech (Honours) degree."],
        "difficulty": "The 2010 and 2011+ variants are adjacent and nearly identical; R.1.12 also lists only U/W, so a model may answer 'no'.",
    },
    {
        "id": "DS-04", "category": "document_specific",
        "question": "How are marks split between quizzes and the end-semester examination in the common courses of the first three semesters?",
        "reference_answer": "Two quizzes of 20 marks each (40%) and the end-semester examination (60%).",
        "evidence": [
            # table row as extracted, plus the same cells in the PDF's raw reading order
            {"quote": "For common courses of first three semesters: 2 Quizzes – 20 marks each End Semester Examination | 40% 60% |",
             "raw": "For common courses of first three semesters: 2 Quizzes – 20 marks each 40% End Semester Examination 60%"},
        ],
        "distractor": ["The weightage for theory courses of higher semesters of the B.Tech programme should not be below 40% each for both sessional assessment and end semester examination."],
        "difficulty": "The values live in a table cell; the adjacent higher-semester rule also contains '40%' and 'end semester examination'.",
    },
    {
        "id": "DS-05", "category": "document_specific",
        "question": "What attendance code does a student with 90% attendance in a course receive?",
        "reference_answer": "'G' (Good), which covers attendance of 85 to 94%.",
        "evidence": ["| 85 to 94% | Good | G |"],
        "distractor": [],
        "difficulty": "Requires reading a small table and mapping 90% into the 85-94% band; the many '85%' attendance chunks compete.",
    },
    # ------------------------------------------------------------ controls
    {
        "id": "C-01", "category": "control",
        "question": "How long is the end-semester examination in a lecture-based subject?",
        "reference_answer": "Three hours; there is one end-semester examination of three hours in each lecture-based subject.",
        "evidence": ["There will be one end-semester examination of three hours duration in each lecture based subject."],
        "distractor": [],
        "difficulty": "Control: direct paraphrase of R.17.1.",
    },
    {
        "id": "C-02", "category": "control",
        "question": "What is the medium of instruction for the B.Tech programme?",
        "reference_answer": "English (for instruction, examinations and project reports).",
        "evidence": ["The medium of instruction, examination and project reports will be English."],
        "distractor": [],
        "difficulty": "Control: single-sentence rule R.2.9.",
    },
    {
        "id": "C-03", "category": "control",
        "question": "How many grade points does an S grade carry?",
        "reference_answer": "10 grade points.",
        "evidence": ["| S | 10 |  |"],
        "distractor": [],
        "difficulty": "Control: direct table lookup (grade table R.21.1).",
    },
    {
        "id": "C-04", "category": "control",
        "question": "How many credits must a student earn in the first semester to continue in the programme?",
        "reference_answer": "Not less than 12 credits in the first semester (and 28 credits by the end of the second semester).",
        "evidence": ["A student should earn not less than 12 credits in the first semester, and 28 credits at the end of the second semester."],
        "distractor": [],
        "difficulty": "Control: direct statement in R.10.1.",
    },
]


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def words(s):
    s = unicodedata.normalize("NFKC", s).translate(QUOTE_MAP).replace("-\n", "")
    return re.findall(r"[a-z0-9]+", s.lower())


def contains_seq(hay, needle):
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def locate(item, pages, raw_words, chunks):
    """Return (quote, pages, chunk_ids) for an evidence item; raise if it cannot be verified.

    An item is a quote string, or {"quote", "raw"} for table cells whose order
    in the raw PDF text differs from the extracted Markdown row; then "raw" is
    checked against the raw PDF text instead of "quote".
    """
    quote, raw_quote = (item["quote"], item["raw"]) if isinstance(item, dict) else (item, item)
    q = norm(quote)
    found_pages = [p["page"] for p in pages if q in norm(p["text"])]
    if not found_pages:
        raise ValueError(f"not in cleaned text: {quote!r}")
    for pg in found_pages:
        if not contains_seq(raw_words[pg], words(raw_quote)):
            raise ValueError(f"not in raw PDF text of page {pg}: {raw_quote!r}")
    chunk_ids = [c["chunk_id"] for c in chunks if q in norm(c["text"])]
    if not chunk_ids:
        raise ValueError(f"split across chunks (no single chunk contains it): {quote!r}")
    return quote, found_pages, chunk_ids


def build():
    pages = json.loads(config.PAGES_JSON.read_text())["pages"]
    chunks_doc = json.loads(config.CHUNKS_JSON.read_text())
    chunks = chunks_doc["chunks"]
    doc = pymupdf.open(config.SOURCE_PDF)
    raw_words = {i + 1: words(page.get_text("text")) for i, page in enumerate(doc)}

    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    out, errors = [], []
    for q in QUESTIONS:
        assert q["category"] in CATEGORIES, q["id"]
        try:
            ev = [dict(zip(("quote", "pages", "chunk_ids"), locate(e, pages, raw_words, chunks)))
                  for e in q["evidence"]]
            ds = [dict(zip(("quote", "pages", "chunk_ids"), locate(d, pages, raw_words, chunks)))
                  for d in q["distractor"]]
        except ValueError as exc:
            errors.append(f"{q['id']}: {exc}")
            continue
        rel = sorted({c for e in ev for c in e["chunk_ids"]})
        out.append({
            "query_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "supporting_pages": sorted({p for e in ev for p in e["pages"]}),
            "relevant_chunk_ids": rel,
            "evidence": ev,
            "distractor_chunk_ids": sorted({c for d in ds for c in d["chunk_ids"]} - set(rel)),
            "distractors": ds,
            "difficulty_reason": q["difficulty"],
            "verified": True,
        })
    return out, errors, chunks_doc["metadata"]


def main():
    questions, errors, chunk_meta = build()
    if errors:
        print("VERIFICATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_pdf": chunk_meta["source_pdf"],
        "source_sha256": chunk_meta["source_sha256"],
        "chunking": {k: chunk_meta[k] for k in ("chunk_size", "chunk_overlap", "n_chunks")},
        "categories": CATEGORIES,
        "n_questions": len(questions),
        "page_numbering": chunk_meta["page_numbering"],
        "verification": "every evidence/distractor quote matched in cleaned page text, in the raw "
                        "PyMuPDF page text (word sequence) and inside at least one chunk",
        "note": "Candidate stress tests only; failures/rescues are not established until measured.",
    }
    (EVAL_DIR / "questions.json").write_text(
        json.dumps({"metadata": meta, "questions": questions}, indent=2, ensure_ascii=False))
    with open(EVAL_DIR / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "category", "question", "reference_answer", "supporting_pages",
                    "relevant_chunk_ids", "distractor_chunk_ids", "evidence_quotes",
                    "difficulty_reason"])
        for q in questions:
            w.writerow([q["query_id"], q["category"], q["question"], q["reference_answer"],
                        ";".join(map(str, q["supporting_pages"])),
                        ";".join(map(str, q["relevant_chunk_ids"])),
                        ";".join(map(str, q["distractor_chunk_ids"])),
                        " || ".join(e["quote"] for e in q["evidence"]),
                        q["difficulty_reason"]])

    counts = Counter(q["category"] for q in questions)
    print(f"Verified {len(questions)} questions against {meta['source_pdf']} "
          f"(all evidence quotes found in cleaned text, raw PDF text and chunks)\n")
    print(f"{'category':<22}{'n':>3}  description")
    for cat, desc in CATEGORIES.items():
        print(f"{cat:<22}{counts[cat]:>3}  {desc}")
    print(f"{'TOTAL':<22}{len(questions):>3}")
    multi = [q["query_id"] for q in questions if len(q["relevant_chunk_ids"]) > 1]
    print(f"\nQuestions whose evidence spans >1 chunk: {len(multi)} {multi}")
    print(f"Questions with a verified distractor chunk: "
          f"{sum(bool(q['distractor_chunk_ids']) for q in questions)}")
    print(f"\nSaved {(EVAL_DIR / 'questions.json').relative_to(config.PROJECT_ROOT)} and "
          f"{(EVAL_DIR / 'questions.csv').relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
