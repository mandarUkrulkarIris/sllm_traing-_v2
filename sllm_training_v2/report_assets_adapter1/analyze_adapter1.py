import os
import json
import collections
import statistics

BASE_DIR = r"D:\Dev\sllm_training_v2_gitrepo\sllm_traing-_v2\sllm_training_v2"
EXISTING_STATS = os.path.join(BASE_DIR, "report_assets", "stats.json")
CANON_REPORT = os.path.join(BASE_DIR, "report_assets", "canonicalization_report.json")
PRECANON_JSONL = os.path.join(BASE_DIR, "dataprep", "dataset_v16v2_200726.clean.jsonl")
CANON_JSONL = os.path.join(BASE_DIR, "dataprep", "dataset_v16v2_200726.clean.canonical.jsonl")
OUT_JSON = os.path.join(BASE_DIR, "report_assets_adapter1", "stats_adapter1.json")

# ---------- 1. Corpus + job-level stats are unchanged from the clean-dataset report ----------
with open(EXISTING_STATS, "r", encoding="utf-8") as f:
    existing = json.load(f)

corpus = existing["corpus"]
doc_table_counts = existing["doc_table_counts"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


precanon_rows = load_jsonl(PRECANON_JSONL)
canon_rows = load_jsonl(CANON_JSONL)


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

for r in canon_rows:
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

# pre-canonicalization table_type distribution (dataset_v16v2_200726.clean.jsonl), for before/after comparison
precanon_table_type_counter = collections.Counter()
for r in precanon_rows:
    try:
        comp = json.loads(r.get("completion", ""))
        tt = comp.get("table_type")
        tt_val = tt.get("value") if isinstance(tt, dict) else tt
        precanon_table_type_counter[str(tt_val)] += 1
    except Exception:
        pass

with open(CANON_REPORT, "r", encoding="utf-8") as f:
    canon_report = json.load(f)

stats = {
    "corpus": corpus,
    "doc_table_counts": doc_table_counts,
    "dataset": {
        "precanon_total_records": len(precanon_rows),
        "canon_total_records": len(canon_rows),
        "removed_by_canonicalization": len(precanon_rows) - len(canon_rows),
        "parse_errors": parse_errors,
        "prompt_len_describe": describe(prompt_lens),
        "completion_len_describe": describe(completion_lens),
        "row_counts_describe": describe(row_counts),
        "col_counts_describe": describe(col_counts),
        "cell_counts_describe": describe(cell_counts),
        "table_type_counter": dict(table_type_counter),
        "precanon_table_type_counter": dict(precanon_table_type_counter),
        "row_type_counter": dict(row_type_counter),
        "column_type_counter": dict(column_type_counter),
        "confidence_describe": describe(confidence_values),
        "note_ref_fill_rate": note_ref_filled / note_ref_total if note_ref_total else None,
        "concept_id_fill_rate": concept_id_filled / concept_id_total if concept_id_total else None,
        "distinct_table_types": len(table_type_counter),
    },
    "canonicalization": canon_report,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("canon_records", len(canon_rows), "precanon_records", len(precanon_rows))
print("distinct_table_types (canonical)", len(table_type_counter))
print("distinct_table_types (pre-canon)", len(precanon_table_type_counter))
print("Wrote", OUT_JSON)
