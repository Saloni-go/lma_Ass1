"""Stage 6: latency benchmark of both pipelines + scaling measurements.

Phases (from the project root):
    python -m src.benchmark_latency retrieval   # all questions x REPEATS, no LLM
    python -m src.benchmark_latency generation  # GEN_QUERIES x REPEATS, both answers (Gemini)
    python -m src.benchmark_latency scaling     # CE cost vs #pairs, FAISS cost vs index size
    python -m src.benchmark_latency report      # summary table + charts from the raw files

Timing rules
  - Models are loaded and the index is built/loaded BEFORE timing; one warm-up
    query runs through every component first. Initialisation is never timed.
  - Every duration comes from time.perf_counter() around exactly one step.
  - Per repetition the vector search runs ONCE; Pipeline 1 and Pipeline 2 share
    those candidates (as in compare_pipelines), so the only extra work in
    Pipeline 2 is Cross-Encoder scoring + sorting.
  - Repetitions are interleaved (round-robin over queries) to spread CPU-frequency
    and background-load drift over all queries.
"""
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import faiss
import numpy as np
import torch

from . import config
from .embeddings import embed_texts
from .pipeline import RAGPipeline

OUT = config.RESULTS_DIR / "latency"
RAW_RETRIEVAL = OUT / "raw_retrieval_timings.csv"
RAW_GENERATION = OUT / "raw_generation_timings.csv"          # completed pairs only
RAW_GEN_ATTEMPTS = OUT / "raw_generation_attempts.csv"        # every API attempt, ok or failed
SCALING_JSON = OUT / "scaling_measurements.json"
SUMMARY_JSON = OUT / "latency_summary.json"
TABLE_CSV = OUT / "latency_table.csv"
TABLE_MD = OUT / "latency_table.md"
QUESTIONS_JSON = config.DATA_DIR / "evaluation" / "questions.json"

REPEATS = 5
GEN_QUERIES = ["KT-02", "EX-05", "C-01"]   # a rescue, a partial rescue, a control
K, TOP_K = config.TOP_K_CONTEXT, config.TOP_K_RETRIEVE


def questions():
    return json.loads(QUESTIONS_JSON.read_text())["questions"]


def hardware():
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        mem_gb = round(int(open("/proc/meminfo").readline().split()[1]) / 1024**2, 1)
    except (OSError, ValueError):
        mem_gb = None
    import sentence_transformers
    return {"cpu": cpu, "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "ram_gb": mem_gb,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)",
            "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "faiss": faiss.__version__,
            "sentence_transformers": sentence_transformers.__version__}


def timed_query(rag, query):
    """One timed pass: shared retrieval, Pipeline-1 selection, Pipeline-2 re-ranking.

    Returns component durations in ms plus the two top-3 lists. Each step is a
    contiguous perf_counter segment, so the totals are measured, not summed guesses.
    """
    t0 = time.perf_counter()
    q = embed_texts(rag.bi_encoder, [query])                       # query embedding
    t1 = time.perf_counter()
    scores, rows = rag.index.search(q, TOP_K)                     # FAISS top-10
    t2 = time.perf_counter()
    cands = [rag.chunks_by_id[rag.id_map[int(r)]] for r in rows[0]]  # id -> chunk lookup
    t3 = time.perf_counter()
    p1_top3 = cands[:K]                                           # Pipeline 1 selection
    t4 = time.perf_counter()
    pairs = [(query, c["text"]) for c in cands]
    ce = rag.cross_encoder.predict(pairs, show_progress_bar=False)  # CE scoring, 10 pairs
    t5 = time.perf_counter()
    order = sorted(range(len(cands)), key=lambda i: -float(ce[i]))  # sort + top-3
    p2_top3 = [cands[i] for i in order[:K]]
    t6 = time.perf_counter()
    ms = lambda a, b: (b - a) * 1000
    return {
        "query_embed_ms": ms(t0, t1), "faiss_search_ms": ms(t1, t2), "id_lookup_ms": ms(t2, t3),
        "p1_select_ms": ms(t3, t4), "ce_score_ms": ms(t4, t5), "sort_select_ms": ms(t5, t6),
        "shared_retrieval_ms": ms(t0, t3),
        "p1_retrieval_ms": ms(t0, t4),                    # embed + search + lookup + top-3
        "p2_retrieval_ms": ms(t0, t3) + ms(t4, t6),        # same shared part + CE + sort
        "rerank_overhead_ms": ms(t4, t6),
        "p1_top3": [c["chunk_id"] for c in p1_top3], "p2_top3": [c["chunk_id"] for c in p2_top3],
        "n_pairs": len(pairs),
    }


def warm_up(rag):
    for _ in range(3):
        timed_query(rag, "warm-up query about attendance and grades")


# ---------------------------------------------------------------- phase 1
def run_retrieval():
    qs = questions()
    rag = RAGPipeline()
    warm_up(rag)
    rows = []
    for rep in range(1, REPEATS + 1):
        for q in qs:
            r = timed_query(rag, q["question"])
            rows.append({"repeat": rep, "query_id": q["query_id"], "category": q["category"],
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RAW_RETRIEVAL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})
    print(f"saved {len(rows)} timed queries ({len(qs)} questions x {REPEATS} repeats) -> {RAW_RETRIEVAL}")


# ---------------------------------------------------------------- phase 2
GEN_TIMEOUT_MS = 60_000          # per-request HTTP timeout
GEN_MAX_ATTEMPTS = 3             # 1 try + 2 retries per answer
GEN_RETRY_WAITS_S = (2, 5)       # short, bounded back-off
GEN_MAX_CONSECUTIVE_FAILURES = 3 # stop the run instead of waiting on a dead quota
# Gemini 2.5 Flash list prices (USD per 1M tokens, text, standard tier); thinking
# tokens are billed as output. Check the current Vertex AI pricing page before
# relying on the estimate.
PRICE_PER_M = {"input": 0.30, "output": 2.50}


def timed_generate(rag, query, ctx, attempt_log, label):
    """One Gemini answer with bounded retries. Returns a result dict or None.

    Every attempt (successful or not) is appended to attempt_log. The reported
    latency is the perf_counter duration of the successful request only.
    """
    from google.genai import types
    from .generation import GENERATION_SETTINGS, build_prompt

    prompt = build_prompt(query, ctx)
    cfg = types.GenerateContentConfig(**GENERATION_SETTINGS,
                                      http_options=types.HttpOptions(timeout=GEN_TIMEOUT_MS))
    for attempt in range(1, GEN_MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            resp = rag.client.models.generate_content(model=rag.gemini_model, contents=prompt, config=cfg)
            ms = (time.perf_counter() - t0) * 1000
            if not (resp.text or "").strip():
                raise RuntimeError(f"empty response (finish reason: {resp.candidates[0].finish_reason})")
            u = resp.usage_metadata
            out = {"ms": ms, "attempt": attempt,
                   "input_tokens": u.prompt_token_count or 0,
                   "output_tokens": u.candidates_token_count or 0,
                   "thinking_tokens": u.thoughts_token_count or 0}
            attempt_log.append({**label, "attempt": attempt, "status": "ok", "ms": round(ms, 1), "error": ""})
            return out
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            code = getattr(exc, "code", None)
            attempt_log.append({**label, "attempt": attempt, "status": "failed", "ms": round(ms, 1),
                                "error": f"{type(exc).__name__} {code or ''}: {str(exc)[:160]}"})
            non_retryable = isinstance(code, int) and 400 <= code < 500 and code != 429
            if non_retryable or attempt == GEN_MAX_ATTEMPTS:
                return None
            time.sleep(GEN_RETRY_WAITS_S[attempt - 1])
    return None


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_generation():
    """GEN_QUERIES x REPEATS paired runs (Answer 1 + Answer 2 on one shared retrieval).

    A pair is stored as a completed measurement only if BOTH answers succeeded;
    failed attempts go to the attempts log only. Results are written after every
    pair, and a rerun resumes, skipping completed (query, repeat) pairs.
    """
    from .generation import GENERATION_SETTINGS

    qs = {q["query_id"]: q for q in questions()}
    rag = RAGPipeline()
    if rag.client is None:
        sys.exit(f"no Gemini credentials ({rag.llm_backend})")
    warm_up(rag)
    rows = read_csv(RAW_GENERATION) if RAW_GENERATION.exists() else []
    attempts = read_csv(RAW_GEN_ATTEMPTS) if RAW_GEN_ATTEMPTS.exists() else []
    done = {(r["query_id"], int(r["repeat"])) for r in rows}
    consecutive_failures = 0

    for rep in range(1, REPEATS + 1):
        for qid in GEN_QUERIES:
            if (qid, rep) in done:
                continue
            query = qs[qid]["question"]
            r = timed_query(rag, query)                       # shared retrieval + re-ranking
            ctx = {1: [rag.chunks_by_id[i] for i in r["p1_top3"]],
                   2: [rag.chunks_by_id[i] for i in r["p2_top3"]]}
            gen = {}
            for n in (1, 2):                                   # identical prompt template + settings
                label = {"query_id": qid, "repeat": rep, "answer": n, "backend": rag.llm_backend}
                gen[n] = timed_generate(rag, query, ctx[n], attempts, label)
                consecutive_failures = 0 if gen[n] else consecutive_failures + 1
                write_rows(RAW_GEN_ATTEMPTS, attempts)
                if consecutive_failures >= GEN_MAX_CONSECUTIVE_FAILURES:
                    print(f"STOP: {consecutive_failures} consecutive failed calls; last error: "
                          f"{attempts[-1]['error']}")
                    return
            if gen[1] and gen[2]:
                row = {"query_id": qid, "repeat": rep, "backend": rag.llm_backend,
                       "model": rag.gemini_model, "settings": json.dumps(GENERATION_SETTINGS, sort_keys=True),
                       "p1_top3": ";".join(map(str, r["p1_top3"])), "p2_top3": ";".join(map(str, r["p2_top3"]))}
                row.update({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
                for n in (1, 2):
                    row.update({f"gen{n}_ms": round(gen[n]["ms"], 1), f"gen{n}_attempt": gen[n]["attempt"],
                                f"gen{n}_input_tokens": gen[n]["input_tokens"],
                                f"gen{n}_output_tokens": gen[n]["output_tokens"],
                                f"gen{n}_thinking_tokens": gen[n]["thinking_tokens"]})
                row["p1_end_to_end_ms"] = round(r["p1_retrieval_ms"] + gen[1]["ms"], 1)
                row["p2_end_to_end_ms"] = round(r["p2_retrieval_ms"] + gen[2]["ms"], 1)
                rows.append(row)
                write_rows(RAW_GENERATION, rows)
                print(f"{qid} rep {rep}: gen1 {gen[1]['ms']:.0f} ms, gen2 {gen[2]['ms']:.0f} ms "
                      f"(tokens in {gen[1]['input_tokens']}/{gen[2]['input_tokens']})", flush=True)
            else:
                print(f"{qid} rep {rep}: NOT completed (see attempts log)", flush=True)
    print(f"{len(rows)} completed paired runs in {RAW_GENERATION}")


# ---------------------------------------------------------------- phase 3
def run_scaling():
    qs = questions()
    rag = RAGPipeline()
    warm_up(rag)
    probe = [q["question"] for q in qs[::7]][:5]          # 5 spread-out questions
    res = {"probe_queries": probe}

    # (a) Cross-Encoder cost vs number of (query, chunk) pairs, real chunks.
    #     n = 85 is the whole corpus: "run the Cross-Encoder on everything".
    ce = []
    for n in (1, 5, 10, 20, 40, len(rag.chunks)):
        for rep in range(REPEATS):
            for query in probe:
                full = rag.baseline_retrieve(query, top_k=len(rag.chunks))
                pairs = [(query, c["text"]) for c in full[:n]]
                t0 = time.perf_counter()
                rag.cross_encoder.predict(pairs, show_progress_bar=False)
                ce.append({"n_pairs": n, "repeat": rep + 1, "ms": (time.perf_counter() - t0) * 1000})
    res["cross_encoder_vs_pairs"] = ce

    # (b) Bi-Encoder: one-time (offline) cost of embedding the corpus vs per-query cost.
    texts = [c["text"] for c in rag.chunks]
    corpus = []
    for rep in range(3):
        t0 = time.perf_counter()
        embed_texts(rag.bi_encoder, texts)
        corpus.append((time.perf_counter() - t0) * 1000)
    res["bi_encoder_corpus_embedding_ms"] = {"n_chunks": len(texts), "runs_ms": corpus}

    # (c) FAISS exact inner-product search vs index size. Sizes above the real
    #     corpus use SYNTHETIC random unit vectors (same dim, same index type):
    #     search cost depends on N and d, not on the vector contents.
    rng = np.random.default_rng(config.SEED)
    q = embed_texts(rag.bi_encoder, probe)
    faiss_rows = []
    for n in (len(rag.chunks), 1_000, 10_000, 100_000, 500_000):
        if n == len(rag.chunks):
            index, synthetic = rag.index, False
        else:
            x = rng.standard_normal((n, rag.index.d), dtype=np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            index, synthetic = faiss.IndexFlatIP(rag.index.d), True
            index.add(x)
        index.search(q[:1], TOP_K)  # warm-up
        for rep in range(REPEATS):
            for i in range(len(probe)):
                t0 = time.perf_counter()
                index.search(q[i:i + 1], TOP_K)
                faiss_rows.append({"n_vectors": n, "synthetic": synthetic, "repeat": rep + 1,
                                   "ms": (time.perf_counter() - t0) * 1000})
        if synthetic:
            del index, x
    res["faiss_vs_index_size"] = faiss_rows

    OUT.mkdir(parents=True, exist_ok=True)
    SCALING_JSON.write_text(json.dumps(res, indent=1))
    print(f"saved scaling measurements -> {SCALING_JSON}")


# ---------------------------------------------------------------- report
def stats(values):
    v = [float(x) for x in values if x not in ("", None)]
    return {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0, "min": min(v), "max": max(v),
            "p95": float(np.percentile(v, 95))}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_report():
    ret = read_csv(RAW_RETRIEVAL)
    gen = read_csv(RAW_GENERATION) if RAW_GENERATION.exists() else []
    gen_ok = [r for r in gen if r["gen1_ms"] and r["gen2_ms"]]
    scal = json.loads(SCALING_JSON.read_text()) if SCALING_JSON.exists() else None

    comp = [("query_embed_ms", "Query embedding (Bi-Encoder, 1 query)"),
            ("faiss_search_ms", "FAISS top-10 search (85 vectors)"),
            ("id_lookup_ms", "FAISS id -> chunk lookup"),
            ("p1_select_ms", "Pipeline 1: take top-3"),
            ("ce_score_ms", "Cross-Encoder scoring (10 pairs)"),
            ("sort_select_ms", "Pipeline 2: sort by CE score + top-3"),
            ("shared_retrieval_ms", "Shared retrieval (embed + search + lookup)"),
            ("p1_retrieval_ms", "Pipeline 1 retrieval side (total)"),
            ("p2_retrieval_ms", "Pipeline 2 retrieval side (total)"),
            ("rerank_overhead_ms", "Additional re-ranking time (CE + sort)")]
    n_ret_q = len({r["query_id"] for r in ret})
    table = [{"scope": f"retrieval ({n_ret_q} questions x {REPEATS} repeats)", "component": label, "key": key,
              **stats(r[key] for r in ret)} for key, label in comp]
    if gen_ok:
        n_q = len({r["query_id"] for r in gen_ok})
        for key, label in (("gen1_ms", "Gemini generation, Answer 1 (P1 context)"),
                           ("gen2_ms", "Gemini generation, Answer 2 (P2 context)"),
                           ("p1_end_to_end_ms", "Pipeline 1 end-to-end"),
                           ("p2_end_to_end_ms", "Pipeline 2 end-to-end")):
            table.append({"scope": f"with generation ({n_q} questions x {REPEATS} repeats)",
                          "component": label, "key": key, **stats(r[key] for r in gen_ok)})

    by = {t["key"]: t for t in table}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware(), "units": "milliseconds (ms)", "repeats": REPEATS,
        "llm_model": config.GEMINI_MODEL, "retrieval_questions": n_ret_q, "table": table,
        "rerank_overhead_mean_ms": by["rerank_overhead_ms"]["mean"],
        "retrieval_slowdown_mean_ratio": by["p2_retrieval_ms"]["mean"] / by["p1_retrieval_ms"]["mean"],
        "retrieval_slowdown_median_ratio": by["p2_retrieval_ms"]["median"] / by["p1_retrieval_ms"]["median"],
    }
    if gen_ok:
        summary["end_to_end_slowdown_mean_ratio"] = by["p2_end_to_end_ms"]["mean"] / by["p1_end_to_end_ms"]["mean"]
        summary["end_to_end_slowdown_median_ratio"] = by["p2_end_to_end_ms"]["median"] / by["p1_end_to_end_ms"]["median"]
        summary["rerank_share_of_p2_end_to_end"] = by["rerank_overhead_ms"]["mean"] / by["p2_end_to_end_ms"]["mean"]
        att = read_csv(RAW_GEN_ATTEMPTS) if RAW_GEN_ATTEMPTS.exists() else []
        failed = [a for a in att if a["status"] != "ok"]
        summary["generation_attempts"] = {
            "completed_paired_runs": len(gen_ok), "planned_paired_runs": len(GEN_QUERIES) * REPEATS,
            "api_attempts_total": len(att), "api_attempts_ok": len(att) - len(failed),
            "api_attempts_failed": len(failed),
            "failed_attempts": [{k: a[k] for k in ("query_id", "repeat", "answer", "attempt", "error")} for a in failed],
            "backend": sorted({r["backend"] for r in gen_ok}),
            "settings_identical_across_pipelines": len({r["settings"] for r in gen_ok}) == 1,
            "note": "failed attempts are never counted as measurements; latency = successful request only",
        }
        tok = {}
        for n in (1, 2):
            for kind in ("input", "output", "thinking"):
                tok[f"answer{n}_{kind}"] = stats(r[f"gen{n}_{kind}_tokens"] for r in gen_ok)
        tot_in = sum(int(r[f"gen{n}_input_tokens"]) for r in gen_ok for n in (1, 2))
        tot_out = sum(int(r[f"gen{n}_output_tokens"]) + int(r[f"gen{n}_thinking_tokens"])
                      for r in gen_ok for n in (1, 2))
        cost = tot_in / 1e6 * PRICE_PER_M["input"] + tot_out / 1e6 * PRICE_PER_M["output"]
        summary["tokens"] = tok
        summary["cost_estimate_usd"] = {
            "successful_calls": 2 * len(gen_ok), "input_tokens": tot_in,
            "output_plus_thinking_tokens": tot_out, "prices_per_1M_usd": PRICE_PER_M,
            "estimated_cost_usd": round(cost, 5),
            "per_answer_usd": round(cost / (2 * len(gen_ok)), 6),
            "note": "estimate from recorded token counts x list prices; excludes failed attempts "
                    "(normally not billed) and the single test requests",
        }
    if scal:
        ce_by_n = {}
        for r in scal["cross_encoder_vs_pairs"]:
            ce_by_n.setdefault(r["n_pairs"], []).append(r["ms"])
        ce_stats = {n: stats(v) for n, v in sorted(ce_by_n.items())}
        ns = np.array(sorted(ce_stats))
        med = np.array([ce_stats[n]["median"] for n in ns])
        slope, intercept = np.polyfit(ns, med, 1)
        faiss_by_n = {}
        for r in scal["faiss_vs_index_size"]:
            faiss_by_n.setdefault((r["n_vectors"], r["synthetic"]), []).append(r["ms"])
        emb = scal["bi_encoder_corpus_embedding_ms"]
        summary["scaling"] = {
            "cross_encoder_vs_pairs_ms": {str(n): s for n, s in ce_stats.items()},
            "cross_encoder_linear_fit_ms": {"per_pair": slope, "intercept": intercept,
                                            "fit_on": "median ms vs n_pairs, measured n = 1..85"},
            "cross_encoder_full_corpus_extrapolation_ms": {
                str(n): slope * n + intercept for n in (1_000, 10_000, 100_000, 1_000_000)},
            "faiss_vs_index_size_ms": {f"{n}{' (synthetic)' if syn else ''}": stats(v)
                                       for (n, syn), v in sorted(faiss_by_n.items())},
            "bi_encoder_corpus_embedding": {"n_chunks": emb["n_chunks"], **stats(emb["runs_ms"]),
                                            "per_chunk_ms": statistics.median(emb["runs_ms"]) / emb["n_chunks"]},
        }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    with open(TABLE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "component", "n", "mean_ms", "median_ms", "std_ms", "min_ms", "p95_ms", "max_ms"])
        for t in table:
            w.writerow([t["scope"], t["component"], t["n"]] +
                       [f"{t[k]:.3f}" for k in ("mean", "median", "std", "min", "p95", "max")])
    lines = ["| Component | n | mean (ms) | median (ms) | std (ms) | p95 (ms) |", "|---|---|---|---|---|---|"]
    scope = None
    for t in table:
        if t["scope"] != scope:
            scope = t["scope"]
            lines.append(f"| **{scope}** | | | | | |")
        lines.append(f"| {t['component']} | {t['n']} | {t['mean']:.3f} | {t['median']:.3f} | "
                     f"{t['std']:.3f} | {t['p95']:.3f} |")
    TABLE_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nadditional re-ranking time (mean): {summary['rerank_overhead_mean_ms']:.1f} ms | "
          f"retrieval-side slowdown P2/P1: {summary['retrieval_slowdown_mean_ratio']:.1f}x (mean), "
          f"{summary['retrieval_slowdown_median_ratio']:.1f}x (median)")
    if gen_ok:
        print(f"end-to-end P2/P1: {summary['end_to_end_slowdown_mean_ratio']:.3f}x (mean), "
              f"{summary['end_to_end_slowdown_median_ratio']:.3f}x (median) | "
              f"re-ranking share of P2 end-to-end: {summary['rerank_share_of_p2_end_to_end']:.1%}")
        ga, ce_ = summary["generation_attempts"], summary["cost_estimate_usd"]
        print(f"generation: {ga['completed_paired_runs']}/{ga['planned_paired_runs']} paired runs completed | "
              f"API attempts {ga['api_attempts_ok']} ok, {ga['api_attempts_failed']} failed | backend {ga['backend']} | "
              f"identical settings: {ga['settings_identical_across_pipelines']}")
        t = summary["tokens"]
        print("tokens (mean) A1 in/out/think: "
              f"{t['answer1_input']['mean']:.0f}/{t['answer1_output']['mean']:.0f}/{t['answer1_thinking']['mean']:.0f} | "
              f"A2: {t['answer2_input']['mean']:.0f}/{t['answer2_output']['mean']:.0f}/{t['answer2_thinking']['mean']:.0f}")
        print(f"cost estimate: {ce_['input_tokens']} in + {ce_['output_plus_thinking_tokens']} out/think tokens "
              f"-> ${ce_['estimated_cost_usd']:.4f} total (${ce_['per_answer_usd']:.5f}/answer)")
    if scal:
        s = summary["scaling"]
        print("CE ms vs pairs (median):", {n: round(v["median"], 1) for n, v in s["cross_encoder_vs_pairs_ms"].items()})
        print(f"CE linear fit: {s['cross_encoder_linear_fit_ms']['per_pair']:.2f} ms/pair; extrapolated full-corpus CE:",
              {n: f"{v / 1000:.1f} s" for n, v in s["cross_encoder_full_corpus_extrapolation_ms"].items()})
        print("FAISS ms vs N (median):", {n: round(v["median"], 3) for n, v in s["faiss_vs_index_size_ms"].items()})
        print(f"Bi-Encoder corpus embedding: {s['bi_encoder_corpus_embedding']['median']:.0f} ms for "
              f"{s['bi_encoder_corpus_embedding']['n_chunks']} chunks "
              f"({s['bi_encoder_corpus_embedding']['per_chunk_ms']:.1f} ms/chunk, one-time)")
    make_charts(summary)


def make_charts(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference palette (dataviz skill), categorical slots in fixed order
    C = {"embed": "#2a78d6", "search": "#eb6834", "ce": "#1baf7a", "sort": "#eda100", "gen": "#6250d6"}
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})
    by = {t["key"]: t for t in summary["table"]}
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: retrieval-side breakdown (stacked medians) + end-to-end
    has_gen = "gen1_ms" in by
    fig, axes = plt.subplots(1, 2 if has_gen else 1, figsize=(11 if has_gen else 6.5, 3.9))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    segs = [("query_embed_ms", "Query embedding", C["embed"]),
            ("faiss_search_ms", "FAISS search + lookup", C["search"]),
            ("ce_score_ms", "Cross-Encoder (10 pairs)", C["ce"]),
            ("sort_select_ms", "Sort + top-3", C["sort"])]
    rows = {"Pipeline 1\n(baseline)": ["query_embed_ms", "faiss_search_ms"],
            "Pipeline 2\n(re-ranked)": ["query_embed_ms", "faiss_search_ms", "ce_score_ms", "sort_select_ms"]}
    for y, (name, keys) in enumerate(rows.items()):
        left = 0
        for key, label, color in segs:
            if key not in keys:
                continue
            w = by[key]["median"] + (by["id_lookup_ms"]["median"] if key == "faiss_search_ms" else 0)
            ax.barh(y, w, left=left, color=color, height=0.5, edgecolor=SURF, linewidth=2,
                    label=label if y == 1 else None)
            left += w
        total = by["p1_retrieval_ms" if y == 0 else "p2_retrieval_ms"]["median"]
        ax.text(left + 4, y, f"{total:.1f} ms", va="center", color=INK)
    ax.set_yticks(range(len(rows)), list(rows))
    ax.invert_yaxis()
    ax.set_xlabel("median latency per query (ms)")
    ax.set_title(f"Retrieval side (excl. LLM), {summary['retrieval_questions']} questions x {REPEATS} repeats", color=INK, loc="left", fontsize=10, pad=44)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, by["p2_retrieval_ms"]["median"] * 1.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    ax.text(0, -0.32, f"FAISS search + lookup ({by['faiss_search_ms']['median'] + by['id_lookup_ms']['median']:.2f} ms) "
            f"and sort ({by['sort_select_ms']['median']:.2f} ms) are too small to see.",
            transform=ax.transAxes, fontsize=8, color=INK2)

    if has_gen:
        ax = axes[1]
        for y, (ret_key, gen_key, name) in enumerate((("p1_retrieval_ms", "gen1_ms", "Pipeline 1"),
                                                      ("p2_retrieval_ms", "gen2_ms", "Pipeline 2"))):
            r, g = by[ret_key]["median"], by[gen_key]["median"]
            ax.barh(y, r, color=C["ce"] if y else C["embed"], height=0.5, edgecolor=SURF, linewidth=2,
                    label=("Retrieval side (P1)" if y == 0 else "Retrieval side incl. re-ranking (P2)"))
            ax.barh(y, g, left=r, color=C["gen"], height=0.5, edgecolor=SURF, linewidth=2,
                    label="Gemini generation" if y == 1 else None)
            e2e = by["p1_end_to_end_ms" if y == 0 else "p2_end_to_end_ms"]["median"]
            ax.text(r + g + 40, y, f"sum of medians {(r + g) / 1000:.2f} s\nmedian end-to-end {e2e / 1000:.2f} s",
                    va="center", color=INK, fontsize=8)
        ax.set_yticks([0, 1], ["Pipeline 1\n(baseline)", "Pipeline 2\n(re-ranked)"])
        ax.invert_yaxis()
        ax.set_xlabel("median latency per query (ms)")
        ax.set_title(f"End-to-end incl. Gemini ({by['gen1_ms']['n']} paired runs)", color=INK, loc="left", fontsize=10, pad=44)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(by["p1_end_to_end_ms"]["median"], by["p2_end_to_end_ms"]["median"]) * 1.5)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "latency_breakdown.png", dpi=160)
    plt.close(fig)

    # ---- Figure 2: scaling (two separate panels, no dual axis)
    s = summary.get("scaling")
    if not s:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    ce = s["cross_encoder_vs_pairs_ms"]
    ns = [int(n) for n in ce]
    a1.plot(ns, [ce[str(n)]["median"] for n in ns], color=C["ce"], linewidth=2, marker="o", markersize=6)
    a1.fill_between(ns, [ce[str(n)]["min"] for n in ns], [ce[str(n)]["max"] for n in ns],
                    color=C["ce"], alpha=0.15, linewidth=0)
    a1.annotate(f"10 pairs: {ce['10']['median']:.0f} ms", (10, ce["10"]["median"]), textcoords="offset points",
                xytext=(10, -4), color=INK, fontsize=9, ha="left", va="top")
    a1.annotate(f"{ce[str(ns[-1])]['median']:.0f} ms", (ns[-1], ce[str(ns[-1])]["median"]),
                textcoords="offset points", xytext=(-10, 8), color=INK, fontsize=9, ha="right")
    per_pair = [ce[str(n)]["median"] / n for n in ns if n >= 5]
    a1.set_xlabel("number of (query, chunk) pairs scored")
    a1.set_ylabel("Cross-Encoder time (ms)")
    a1.set_title(f"Cross-Encoder: roughly linear, {min(per_pair):.0f}-{max(per_pair):.0f} ms per pair\n"
                 f"5 queries x 5 repeats, band = min-max; 85 = whole corpus",
                 color=INK, loc="left", fontsize=10)
    a1.grid(color=GRID, linewidth=0.8)
    a1.set_axisbelow(True)

    fz = s["faiss_vs_index_size_ms"]
    xs = [int(k.split()[0]) for k in fz]
    ys = [fz[k]["median"] for k in fz]
    a2.plot(xs, ys, color=C["search"], linewidth=2, marker="o", markersize=6)
    a2.set_xscale("log")
    a2.set_yscale("log")
    for k, x, y in zip(fz, xs, ys):
        a2.annotate(f"{y:.2f} ms" if y < 10 else f"{y:.0f} ms", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=INK, fontsize=8)
    a2.set_xlabel("vectors in index (log; > 85 are synthetic)")
    a2.set_ylabel("FAISS top-10 search (ms, log)")
    a2.set_title("Exact FAISS search, 1 query, 384-d", color=INK, loc="left", fontsize=10)
    a2.grid(color=GRID, linewidth=0.8, which="major")
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "latency_scaling.png", dpi=160)
    plt.close(fig)
    print(f"charts -> {config.FIGURES_DIR / 'latency_breakdown.png'}, {config.FIGURES_DIR / 'latency_scaling.png'}")


if __name__ == "__main__":
    {"retrieval": run_retrieval, "generation": run_generation,
     "scaling": run_scaling, "report": run_report}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
