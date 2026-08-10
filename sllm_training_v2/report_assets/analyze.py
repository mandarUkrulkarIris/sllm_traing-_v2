import os
import json
import collections
import statistics
import re

BASE_DIRS = [r"C:\Users\mandar.ukrulkar\Downloads\data_raw_in_out_jsons_200726\output"]
CLEAN_JSONL = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.clean.jsonl"
RAW_JSONL = r"D:\Dev\sllm_training_v2\dataset_v16v2_200726.jsonl"
OUT_JSON = r"D:\Dev\sllm_training_v2\report_assets\stats.json"

# ---------- 1. Source corpus: walk job dirs ----------
jobs = []
doc_table_files = collections.Counter()   # docname -> count of *_input.json files present
doc_job_ids = collections.defaultdict(list)
provider_counter = collections.Counter()
classified_counter = collections.Counter()
job_table_counts = []
job_financial_counts = []
job_skipped_counts = []
missing_job_json = []
duplicate_docx = collections.defaultdict(list)

for base in BASE_DIRS:
    if not os.path.isdir(base):
        continue
    source_tag = os.path.basename(base)
    for jobid in sorted(os.listdir(base)):
        jdir = os.path.join(base, jobid)
        if not os.path.isdir(jdir):
            continue
        job_json_path = os.path.join(jdir, "job.json")
        if not os.path.isfile(job_json_path):
            missing_job_json.append(jdir)
            continue
        with open(job_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        docx = meta.get("docx", "")
        docname = os.path.basename(docx) if docx else "UNKNOWN"
        input_files = [f for f in os.listdir(jdir) if f.endswith("_input.json")]
        n_inputs = len(input_files)

        jobs.append({
            "job_id": meta.get("job_id", jobid),
            "source_dir": source_tag,
            "docx": docx,
            "docname": docname,
            "table_count": meta.get("table_count"),
            "financial_table_count": meta.get("financial_table_count"),
            "skipped_table_count": meta.get("skipped_table_count"),
            "classified": meta.get("classified"),
            "ai_provider": meta.get("ai_provider"),
            "n_input_files": n_inputs,
        })
        doc_table_files[docname] += n_inputs
        doc_job_ids[docname].append(meta.get("job_id", jobid))
        provider_counter[meta.get("ai_provider", "unknown")] += 1
        classified_counter[str(meta.get("classified"))] += 1
        if meta.get("table_count") is not None:
            job_table_counts.append(meta["table_count"])
        if meta.get("financial_table_count") is not None:
            job_financial_counts.append(meta["financial_table_count"])
        if meta.get("skipped_table_count") is not None:
            job_skipped_counts.append(meta["skipped_table_count"])
        duplicate_docx[docname].append(meta.get("job_id", jobid))

dup_docs = {k: v for k, v in duplicate_docx.items() if len(v) > 1}

total_jobs = len(jobs)
total_docs = len(doc_table_files)
total_input_files = sum(doc_table_files.values())
total_table_count_meta = sum(job_table_counts)
total_financial_meta = sum(job_financial_counts)
total_skipped_meta = sum(job_skipped_counts)

# ---------- 2. Clean / raw training dataset EDA ----------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

clean_rows = load_jsonl(CLEAN_JSONL)
raw_rows = load_jsonl(RAW_JSONL)

def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1

def describe(values):
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "p25": pct(s, 0.25),
        "median": pct(s, 0.5),
        "mean": statistics.mean(s),
        "p75": pct(s, 0.75),
        "p90": pct(s, 0.90),
        "p99": pct(s, 0.99),
        "max": s[-1],
    }

table_type_counter = collections.Counter()
row_type_counter = collections.Counter()
column_type_counter = collections.Counter()
prompt_lens = []
completion_lens = []
row_counts = []
col_counts = []
cell_counts = []
concept_id_filled = 0
concept_id_total = 0
note_ref_filled = 0
note_ref_total = 0
confidence_values = []
parse_errors = 0

for r in clean_rows:
    prompt = r.get("prompt", "")
    completion_str = r.get("completion", "")
    prompt_lens.append(len(prompt))
    completion_lens.append(len(completion_str))
    try:
        comp = json.loads(completion_str)
    except Exception:
        parse_errors += 1
        continue
    tt = comp.get("table_type")
    if isinstance(tt, dict):
        tt_val = tt.get("value")
        if tt.get("confidence") is not None:
            confidence_values.append(tt["confidence"])
    else:
        tt_val = tt
    table_type_counter[str(tt_val)] += 1

    rows_obj = comp.get("rows", {})
    cols_obj = comp.get("columns", {})
    cells_obj = comp.get("cells", {})
    row_counts.append(len(rows_obj))
    col_counts.append(len(cols_obj))
    n_cells = 0
    for _, rowcells in cells_obj.items():
        if isinstance(rowcells, dict):
            n_cells += len(rowcells)
    cell_counts.append(n_cells)

    for _, rowinfo in rows_obj.items():
        if isinstance(rowinfo, dict):
            row_type_counter[str(rowinfo.get("row_type"))] += 1
            nrv = rowinfo.get("note_ref_value")
            note_ref_total += 1
            if nrv and any(v not in (None, "null") for v in (nrv if isinstance(nrv, list) else [nrv])):
                note_ref_filled += 1

    for _, colinfo in cols_obj.items():
        if isinstance(colinfo, dict):
            column_type_counter[str(colinfo.get("column_type"))] += 1

    for _, rowcells in cells_obj.items():
        if isinstance(rowcells, dict):
            for _, cellinfo in rowcells.items():
                if isinstance(cellinfo, dict):
                    concept_id_total += 1
                    if cellinfo.get("concept_id"):
                        concept_id_filled += 1

# raw vs clean comparison
raw_table_type_counter = collections.Counter()
for r in raw_rows:
    try:
        comp = json.loads(r.get("completion", ""))
        tt = comp.get("table_type")
        tt_val = tt.get("value") if isinstance(tt, dict) else tt
        raw_table_type_counter[str(tt_val)] += 1
    except Exception:
        pass

stats = {
    "corpus": {
        "total_jobs": total_jobs,
        "total_documents": total_docs,
        "total_input_files": total_input_files,
        "total_table_count_meta": total_table_count_meta,
        "total_financial_table_count_meta": total_financial_meta,
        "total_skipped_table_count_meta": total_skipped_meta,
        "provider_counter": dict(provider_counter),
        "classified_counter": dict(classified_counter),
        "job_table_counts_describe": describe(job_table_counts),
        "job_financial_counts_describe": describe(job_financial_counts),
        "job_skipped_counts_describe": describe(job_skipped_counts),
        "missing_job_json_count": len(missing_job_json),
        "duplicate_docx_docs": {k: v for k, v in list(dup_docs.items())[:50]},
        "duplicate_docx_count": len(dup_docs),
    },
    "doc_table_counts": dict(doc_table_files),
    "dataset": {
        "raw_total_records": len(raw_rows),
        "clean_total_records": len(clean_rows),
        "removed_by_dedup": len(raw_rows) - len(clean_rows),
        "parse_errors": parse_errors,
        "prompt_len_describe": describe(prompt_lens),
        "completion_len_describe": describe(completion_lens),
        "row_counts_describe": describe(row_counts),
        "col_counts_describe": describe(col_counts),
        "cell_counts_describe": describe(cell_counts),
        "table_type_counter": dict(table_type_counter),
        "raw_table_type_counter": dict(raw_table_type_counter),
        "row_type_counter": dict(row_type_counter),
        "column_type_counter": dict(column_type_counter),
        "confidence_describe": describe(confidence_values),
        "note_ref_fill_rate": note_ref_filled / note_ref_total if note_ref_total else None,
        "concept_id_fill_rate": concept_id_filled / concept_id_total if concept_id_total else None,
        "distinct_table_types": len(table_type_counter),
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("total_jobs", total_jobs)
print("total_documents", total_docs)
print("total_input_files", total_input_files)
print("total_table_count_meta", total_table_count_meta)
print("total_financial_meta", total_financial_meta)
print("duplicate_docx_count", len(dup_docs))
print("raw_records", len(raw_rows), "clean_records", len(clean_rows))
print("distinct_table_types (clean)", len(table_type_counter))
print("Wrote", OUT_JSON)
