"""Stage 5b: repeated generation for the reviewed stress-test cases, on one backend.

Each case is answered N_RUNS times by both pipelines (one shared retrieval per run,
identical prompt and settings), so answer differences caused by the context can be
separated from run-to-run variation of Gemini. Vertex AI only: the script refuses to
run on another backend, so every reviewed answer comes from the same backend as the
latency benchmark.

Usage (from the project root):
    python -m src.stress_repeats            # generate (resumes; saved after every run)
    python -m src.stress_repeats promote    # copy run 1 of each case to generation_results.json

Run 1 becomes the "primary" answer pair that is reviewed in detail (stress_report.py);
all runs are scored for the key facts of the reference answer (KEY_FACTS below).
"""
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone

from .generation import GENERATION_SETTINGS
from .pipeline import RAGPipeline
from .stress_test import ANSWERS_JSON, OUT_DIR, QUESTIONS_JSON

CASES = ["KT-02", "EX-05", "DS-02", "NG-06", "SY-03", "DS-05"]
N_RUNS = 5
PAUSE_S = 2
REPEATS_JSON = OUT_DIR / "generation_repeats.json"
ARCHIVE_JSON = OUT_DIR / "archived_developer_api_generation_results.json"

# The facts an answer must state to be judged correct, taken from the verified
# reference answers. Each fact is a list of alternative regexes (case-insensitive).
KEY_FACTS = {
    "KT-02": {"CGPA 8.5": [r"8\.5"],
              "measured at the end of the 3rd semester": [r"(third|3rd) semester"],
              "opt during the 4th semester": [r"(fourth|4th) semester"]},
    "EX-05": {"actual attendance above 40%": [r"40\s?%"],
              "notional attendance must reach 85%": [r"notional"]},
    "DS-02": {"HPF = 4 credits": [r"HPF[^.\n]{0,120}?\b(4|four)\b", r"\b(4|four)\b[^.\n]{0,60}HPF"]},
    "NG-06": {"successfully completed courses cannot be repeated": [r"successfully completed"],
              "2014-15 exception (CGPA below 5)": [r"2014"]},
    "SY-03": {"within three weeks": [r"(three|3) weeks"]},
    "DS-05": {"attendance code G": [r"[\"'“‘*]G[\"'”’*]", r"\bG\b\s*\(", r"\(G\)", r"code\s+(is\s+)?G\b"]},
}


def score(qid, answer):
    found = {name: any(re.search(p, answer or "", re.I) for p in pats)
             for name, pats in KEY_FACTS[qid].items()}
    return {"facts_found": found, "n_found": sum(found.values()), "n_facts": len(found),
            "all_facts": all(found.values())}


def load():
    return json.loads(REPEATS_JSON.read_text()) if REPEATS_JSON.exists() else {"runs": {}}


def generate():
    questions = {q["query_id"]: q for q in json.loads(QUESTIONS_JSON.read_text())["questions"]}
    rag = RAGPipeline()
    if "Vertex AI" not in rag.llm_backend:
        sys.exit(f"refusing to run: backend is '{rag.llm_backend}', expected Vertex AI (see .env)")
    data = load()
    data["metadata"] = {"llm_model": rag.gemini_model, "llm_backend": rag.llm_backend,
                        "generation_settings": GENERATION_SETTINGS, "n_runs_per_case": N_RUNS,
                        "cases": CASES, "key_facts": KEY_FACTS}
    failures = 0
    for run in range(1, N_RUNS + 1):          # round-robin: run 1 of every case, then run 2, ...
        for qid in CASES:
            key = f"{qid}#{run}"
            if key in data["runs"] and data["runs"][key]["answer_1"] and data["runs"][key]["answer_2"]:
                continue
            q = questions[qid]
            cmp = rag.compare_pipelines(q["question"])
            ok = bool(cmp["answer_1"] and cmp["answer_2"])
            failures = 0 if ok else failures + 1
            data["runs"][key] = {
                "query_id": qid, "run": run, "baseline_top3": cmp["baseline_top3"],
                "reranked_top3": cmp["reranked_top3"], "rank_changes": cmp["rank_changes"],
                "answer_1": cmp["answer_1"], "answer_1_error": cmp["answer_1_error"],
                "answer_2": cmp["answer_2"], "answer_2_error": cmp["answer_2_error"],
                "prompt_1": cmp["prompt_1"], "prompt_2": cmp["prompt_2"],
                "llm_model": cmp["llm_model"], "llm_backend": rag.llm_backend,
                "checks": cmp["checks"],
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            REPEATS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"{key}: answer_1 {'ok' if cmp['answer_1'] else 'ERROR'} | "
                  f"answer_2 {'ok' if cmp['answer_2'] else 'ERROR'}", flush=True)
            if failures >= 3:
                sys.exit("3 consecutive failed cases; stopping (rerun to resume)")
            time.sleep(PAUSE_S)
    summarise(data)


def summarise(data=None):
    data = data or load()
    print(f"\n{'case':<6} {'A1 all facts':>12} {'A2 all facts':>12}   per-fact hits A1 | A2")
    for qid in CASES:
        runs = [data["runs"][f"{qid}#{r}"] for r in range(1, N_RUNS + 1) if f"{qid}#{r}" in data["runs"]]
        s1 = [score(qid, r["answer_1"]) for r in runs]
        s2 = [score(qid, r["answer_2"]) for r in runs]
        facts = list(KEY_FACTS[qid])
        hits = lambda ss: [sum(s["facts_found"][f] for s in ss) for f in facts]
        print(f"{qid:<6} {sum(s['all_facts'] for s in s1):>8}/{len(runs)} {sum(s['all_facts'] for s in s2):>10}/{len(runs)}"
              f"   {hits(s1)} | {hits(s2)}")


def promote():
    """Make run 1 of each case the primary answer pair in generation_results.json.
    The earlier Developer-API answers are kept in an archive file, never deleted."""
    data = load()
    questions = {q["query_id"]: q for q in json.loads(QUESTIONS_JSON.read_text())["questions"]}
    if ANSWERS_JSON.exists() and not ARCHIVE_JSON.exists():
        shutil.copy(ANSWERS_JSON, ARCHIVE_JSON)
    cases = {}
    for qid in CASES:
        r, q = data["runs"][f"{qid}#1"], questions[qid]
        cases[qid] = {
            "query_id": qid, "category": q["category"], "question": q["question"],
            "reference_answer": q["reference_answer"], "evidence": q["evidence"],
            "relevant_chunk_ids": q["relevant_chunk_ids"],
            **{k: r[k] for k in ("baseline_top3", "reranked_top3", "rank_changes", "answer_1",
                                 "answer_1_error", "answer_2", "answer_2_error", "prompt_1",
                                 "prompt_2", "llm_model", "llm_backend", "checks", "generated_utc")},
            "generation_settings": data["metadata"]["generation_settings"],
            "source": f"run 1 of {N_RUNS} in generation_repeats.json",
        }
    ANSWERS_JSON.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False))
    print(f"wrote {ANSWERS_JSON.name} from run 1; earlier answers archived in {ARCHIVE_JSON.name}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    {"generate": generate, "promote": promote, "summary": summarise}[mode]()
