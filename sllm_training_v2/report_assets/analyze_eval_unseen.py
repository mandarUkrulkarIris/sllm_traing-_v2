import os
import json
import collections
import statistics

EVAL_DIR = r"D:\Dev\custom_llm_outputs\completed_unseen"
TRAIN_SOURCE_DIR = r"C:\Users\mandar.ukrulkar\Downloads\data_raw_in_out_jsons_200726\output"
MIXED_STATS_PATH = r"D:\Dev\sllm_training_v2\report_assets\eval_stats.json"
OUT_JSON = r"D:\Dev\sllm_training_v2\report_assets\eval_stats_unseen.json"
MODEL_PATH = r"D:\Dev\qwen_podman\models\v2\merged_Qwen3.5-4B_v16v2_clean_dataset_210726_fp16-Q6_K.gguf"

TRAIN_JOB_IDS = set(os.listdir(TRAIN_SOURCE_DIR)) if os.path.isdir(TRAIN_SOURCE_DIR) else set()

OVERALL_FIELDS = [
    "table_type_lexical_match_rate",
    "table_type_embedding_match_rate",
    "row_type_accuracy",
    "row_is_signed_negative_accuracy",
    "row_direction_accuracy",
    "row_note_ref_value_accuracy",
    "row_contributing_rows_accuracy",
    "column_type_accuracy",
    "column_direction_accuracy",
    "column_contributing_columns_accuracy",
    "cell_concept_id_lexical_accuracy",
    "cell_concept_id_embedding_accuracy",
    "cell_unit_accuracy",
    "cell_scale_accuracy",
    "cell_scale_multiplier_accuracy",
    "cell_reporting_period_accuracy",
    "cell_reporting_period_normalized_accuracy",
    "cell_concept_meaning_lexical_match_rate",
    "cell_concept_meaning_avg_lexical_similarity",
    "cell_concept_meaning_embedding_match_rate",
    "cell_concept_meaning_avg_embedding_similarity",
]

PER_TABLE_NUMERIC_FIELDS = [
    "row_type_accuracy",
    "column_type_accuracy",
    "concept_id_lexical_accuracy",
    "concept_id_embedding_accuracy",
    "concept_meaning_avg_lexical_similarity",
    "concept_meaning_avg_embedding_similarity",
]

jobs = []
missing_compare = []
per_table_rows = []
job_docnames = {}

for jobid in sorted(os.listdir(EVAL_DIR)):
    jdir = os.path.join(EVAL_DIR, jobid)
    if not os.path.isdir(jdir):
        continue
    job_json_path = os.path.join(jdir, "job.json")
    docname = "UNKNOWN"
    if os.path.isfile(job_json_path):
        with open(job_json_path, "r", encoding="utf-8") as f:
            jm = json.load(f)
        docx = jm.get("docx", "")
        docname = os.path.basename(docx) if docx else "UNKNOWN"
    job_docnames[jobid] = docname

    manifest_path = os.path.join(jdir, "customllm_manifest_customllm.json")
    time_taken = None
    n_tables_manifest = None
    n_errors = 0
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        time_taken = man.get("time_taken")
        tbls = man.get("tables", [])
        n_tables_manifest = len(tbls)
        n_errors = sum(1 for t in tbls if t.get("error"))

    summary_path = os.path.join(jdir, "compare", "compare_summary.json")
    if not os.path.isfile(summary_path):
        missing_compare.append({"job_id": jobid, "docname": docname})
        continue

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    overall = summary.get("overall", {})
    per_table = summary.get("per_table", [])

    jobs.append({
        "job_id": jobid,
        "docname": docname,
        "tables_compared": overall.get("tables_compared"),
        "time_taken_sec": time_taken,
        "n_tables_manifest": n_tables_manifest,
        "n_inference_errors": n_errors,
        "overall": overall,
    })

    for pt in per_table:
        row = dict(pt)
        row["job_id"] = jobid
        row["docname"] = docname
        detail_path = os.path.join(jdir, "compare", f"compare_{pt['table_id']}.json")
        row["row_count_compared"] = None
        row["column_count_compared"] = None
        row["cell_count_compared"] = None
        if os.path.isfile(detail_path):
            try:
                with open(detail_path, "r", encoding="utf-8") as f:
                    detail = json.load(f)
                row["row_count_compared"] = detail.get("rows", {}).get("row_count_compared")
                row["column_count_compared"] = detail.get("columns", {}).get("column_count_compared")
                row["cell_count_compared"] = detail.get("cells", {}).get("cell_count_compared")
            except Exception:
                pass
        per_table_rows.append(row)

# ---- sanity check: this folder is expected to be 100% held out of the training corpus ----
n_in_sample = sum(1 for j in jobs if j["job_id"] in TRAIN_JOB_IDS)

# ---- weighted (by tables_compared) aggregate of job-level "overall" metrics ----
weighted_sum = collections.defaultdict(float)
weighted_wt = collections.defaultdict(float)
for j in jobs:
    w = j["tables_compared"] or 0
    for field in OVERALL_FIELDS:
        v = j["overall"].get(field)
        if v is not None and w:
            weighted_sum[field] += v * w
            weighted_wt[field] += w

weighted_overall = {
    field: (weighted_sum[field] / weighted_wt[field] if weighted_wt[field] else None)
    for field in OVERALL_FIELDS
}

# ---- true micro (table-level) aggregate directly from flattened per_table rows ----
micro_overall = {}
lex_match_vals = [1 if r.get("table_type_lexical_match") else 0 for r in per_table_rows]
emb_match_vals = [1 if r.get("table_type_embedding_match") else 0 for r in per_table_rows]
micro_overall["table_type_lexical_match_rate"] = sum(lex_match_vals) / len(lex_match_vals) if lex_match_vals else None
micro_overall["table_type_embedding_match_rate"] = sum(emb_match_vals) / len(emb_match_vals) if emb_match_vals else None
for field in PER_TABLE_NUMERIC_FIELDS:
    vals = [r[field] for r in per_table_rows if r.get(field) is not None]
    micro_overall[field] = statistics.mean(vals) if vals else None
    micro_overall[field + "__n"] = len(vals)

# ---- per-table distributions for charts ----
row_type_acc_vals = [r["row_type_accuracy"] for r in per_table_rows if r.get("row_type_accuracy") is not None]
column_type_acc_vals = [r["column_type_accuracy"] for r in per_table_rows if r.get("column_type_accuracy") is not None]
concept_id_lex_vals = [r["concept_id_lexical_accuracy"] for r in per_table_rows if r.get("concept_id_lexical_accuracy") is not None]
concept_id_emb_vals = [r["concept_id_embedding_accuracy"] for r in per_table_rows if r.get("concept_id_embedding_accuracy") is not None]

# ---- per-document aggregate (mean row_type_accuracy, table_type match rate) for best/worst ----
doc_bucket = collections.defaultdict(list)
for r in per_table_rows:
    doc_bucket[r["docname"]].append(r)

doc_scores = []
for docname, rows in doc_bucket.items():
    tt_lex = sum(1 if r.get("table_type_lexical_match") else 0 for r in rows) / len(rows)
    row_accs = [r["row_type_accuracy"] for r in rows if r.get("row_type_accuracy") is not None]
    doc_scores.append({
        "docname": docname,
        "n_tables": len(rows),
        "table_type_lexical_match_rate": tt_lex,
        "mean_row_type_accuracy": statistics.mean(row_accs) if row_accs else None,
    })
doc_scores.sort(key=lambda d: (d["mean_row_type_accuracy"] if d["mean_row_type_accuracy"] is not None else 1))

# ---- worst individual tables (lowest row_type_accuracy) ----
worst_tables = sorted(
    [r for r in per_table_rows if r.get("row_type_accuracy") is not None],
    key=lambda r: r["row_type_accuracy"]
)[:20]

# ---- best individual tables: perfect (or near-perfect) score, ranked by size ----
def is_clean_sweep(r):
    for field in ("row_type_accuracy", "column_type_accuracy", "concept_id_lexical_accuracy"):
        v = r.get(field)
        if v is not None and v < 1.0:
            return False
    return r.get("row_type_accuracy") == 1.0

best_tables = sorted(
    [r for r in per_table_rows if is_clean_sweep(r)],
    key=lambda r: (r.get("cell_count_compared") or 0),
    reverse=True,
)[:20]

# ---- does table complexity (size) correlate with accuracy? ----
sized_rows = [r for r in per_table_rows if r.get("row_count_compared") is not None and r.get("row_type_accuracy") is not None]
sized_rows_sorted = sorted(sized_rows, key=lambda r: r["row_count_compared"])
complexity_buckets = []
if sized_rows_sorted:
    n = len(sized_rows_sorted)
    bucket_edges = [0, n // 4, n // 2, 3 * n // 4, n]
    bucket_labels = ["Small (Q1)", "Medium (Q2)", "Large (Q3)", "Very large (Q4)"]
    for (lo, hi), label in zip(zip(bucket_edges[:-1], bucket_edges[1:]), bucket_labels):
        chunk = sized_rows_sorted[lo:hi]
        if not chunk:
            continue
        row_counts_in_bucket = [c["row_count_compared"] for c in chunk]
        complexity_buckets.append({
            "label": label,
            "n_tables": len(chunk),
            "row_count_range": [min(row_counts_in_bucket), max(row_counts_in_bucket)],
            "mean_row_type_accuracy": statistics.mean([c["row_type_accuracy"] for c in chunk]),
        })

# ---- inference throughput ----
times = [j["time_taken_sec"] for j in jobs if j["time_taken_sec"] is not None]
tables_per_job = [j["tables_compared"] for j in jobs if j["tables_compared"] is not None]
per_table_latency = []
for j in jobs:
    if j["time_taken_sec"] and j["n_tables_manifest"]:
        per_table_latency.append(j["time_taken_sec"] / j["n_tables_manifest"])

total_errors = sum(j["n_inference_errors"] for j in jobs)

# ---- benchmark comparison: this pure-unseen run vs. the earlier mixed-corpus report ----
# (that earlier report's 543-table sample was 34 in-sample + this exact 26-job held-out set)
BENCHMARK_FIELDS = [
    "table_type_lexical_match_rate",
    "row_type_accuracy",
    "column_type_accuracy",
    "cell_concept_id_lexical_accuracy",
]
mixed_benchmark = None
if os.path.isfile(MIXED_STATS_PATH):
    with open(MIXED_STATS_PATH, "r", encoding="utf-8") as f:
        mixed_stats = json.load(f)
    mixed_benchmark = {
        "n_jobs": mixed_stats.get("n_jobs_with_compare"),
        "n_tables": mixed_stats.get("total_tables_compared"),
        "weighted_overall": {k: mixed_stats["weighted_overall"].get(k) for k in BENCHMARK_FIELDS},
    }

benchmark_comparison = {
    "unseen": {
        "n_jobs": len(jobs),
        "n_tables": sum(j["tables_compared"] or 0 for j in jobs),
        "weighted_overall": {k: weighted_overall.get(k) for k in BENCHMARK_FIELDS},
    },
    "mixed_report_2026_07_24": mixed_benchmark,
}

stats = {
    "model_path": MODEL_PATH,
    "eval_dir": EVAL_DIR,
    "n_in_sample_sanity_check": n_in_sample,
    "benchmark_comparison": benchmark_comparison,
    "n_jobs_total": len(jobs) + len(missing_compare),
    "n_jobs_with_compare": len(jobs),
    "n_jobs_missing_compare": len(missing_compare),
    "missing_compare_jobs": missing_compare,
    "total_tables_compared": sum(j["tables_compared"] or 0 for j in jobs),
    "total_inference_errors": total_errors,
    "weighted_overall": weighted_overall,
    "micro_overall": micro_overall,
    "row_type_acc_vals": row_type_acc_vals,
    "column_type_acc_vals": column_type_acc_vals,
    "concept_id_lex_vals": concept_id_lex_vals,
    "concept_id_emb_vals": concept_id_emb_vals,
    "doc_scores_worst20": doc_scores[:20],
    "doc_scores_best20": list(reversed(doc_scores[-20:])),
    "n_documents": len(doc_scores),
    "worst_tables": [
        {"job_id": r["job_id"], "docname": r["docname"], "table_id": r["table_id"],
         "row_type_accuracy": r["row_type_accuracy"],
         "table_type_lexical_match": r.get("table_type_lexical_match")}
        for r in worst_tables
    ],
    "best_tables": [
        {"job_id": r["job_id"], "docname": r["docname"], "table_id": r["table_id"],
         "row_type_accuracy": r["row_type_accuracy"],
         "row_count_compared": r.get("row_count_compared"),
         "cell_count_compared": r.get("cell_count_compared")}
        for r in best_tables
    ],
    "complexity_buckets": complexity_buckets,
    "inference_time_describe": {
        "n_jobs": len(times),
        "total_time_sec": sum(times) if times else None,
        "mean_time_sec": statistics.mean(times) if times else None,
        "median_time_sec": statistics.median(times) if times else None,
    },
    "per_table_latency_sec_describe": {
        "mean": statistics.mean(per_table_latency) if per_table_latency else None,
        "median": statistics.median(per_table_latency) if per_table_latency else None,
        "min": min(per_table_latency) if per_table_latency else None,
        "max": max(per_table_latency) if per_table_latency else None,
    },
    "jobs_table_counts_describe": {
        "min": min(tables_per_job) if tables_per_job else None,
        "max": max(tables_per_job) if tables_per_job else None,
        "mean": statistics.mean(tables_per_job) if tables_per_job else None,
        "median": statistics.median(tables_per_job) if tables_per_job else None,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("n_jobs_total", stats["n_jobs_total"])
print("n_jobs_with_compare", stats["n_jobs_with_compare"])
print("total_tables_compared", stats["total_tables_compared"])
print("total_inference_errors", stats["total_inference_errors"])
print("n_documents", stats["n_documents"])
print("n_in_sample_sanity_check (should be 0)", n_in_sample)
print()
print("=== weighted_overall (job-level, weighted by tables_compared) ===")
for k, v in weighted_overall.items():
    print(f"{k}: {v}")
print()
print("=== micro_overall (table-level, flattened) ===")
for k, v in micro_overall.items():
    print(f"{k}: {v}")
print()
print("inference_time_describe", stats["inference_time_describe"])
print("per_table_latency_sec_describe", stats["per_table_latency_sec_describe"])
print()
print("=== benchmark_comparison (this unseen run vs. the 2026-07-24 mixed report) ===")
print(json.dumps(benchmark_comparison, indent=2))
print()
print("=== best_tables (top 10) ===")
for t in stats["best_tables"][:10]:
    print(t)
print()
print("=== complexity_buckets (does size correlate with accuracy?) ===")
for b in complexity_buckets:
    print(b)
print("Wrote", OUT_JSON)
